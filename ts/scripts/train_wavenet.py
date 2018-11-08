import argparse
import sys
import os

import numpy as np
import random
import torch
from sklearn import metrics
from numpy import genfromtxt
from torch.autograd import Variable
from tqdm import trange, tqdm
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from ..models import BasicRNN, LSTNet, DiscretizedMixturelogisticLoss
from ..data_loader import UnlabeledDataLoader, WaveNetDataSet, WaveNetUDataSet, MergeDataset

from ..wavenet_vocoder import WaveNet
from ..wavenet_vocoder.mixture import discretized_mix_logistic_loss


class WaveNetModel(nn.Module):
    def __init__(self, num_mixtures, num_layers, device=None):
        super().__init__()
        self.device = device
        self.num_layers = num_layers
        self.num_mixtures = num_mixtures
        self.wavenet = WaveNet(
            out_channels=num_mixtures * 3,
            layers=num_layers,
            scalar_input=True,
            use_speaker_embedding=False,
            legacy=False)
        self.criterion = DiscretizedMixturelogisticLoss()
        self.receptive_field = self.wavenet.receptive_field

    def set_device(self, device):
        self.device = device

    def save(self, save_path):
        dirname = os.path.dirname(save_path)
        os.makedirs(dirname, exist_ok=True)
        torch.save(self, save_path)

    def load_weights(self, model_path):
        model = torch.load(model_path, map_location=self.device)
        self.wavenet = model.wavenet
        print('Model is loaded from \'{}\'...'.format(model_path))
        return model

    @staticmethod
    def load(model_path, device=None):
        if device is None:
            device = torch.device('cpu')
        model = torch.load(model_path, map_location=device)
        model.set_device(device)

        print('Model is loaded from \'{}\'...'.format(model_path))
        return model

    def eval_dataset(self, dataloader):
        self.eval()

        Y_pred = []
        Y_true = []

        for X, Y in tqdm(dataloader):
            X = X.to(self.device)
            Y = Y.to(self.device)

            X_losses = []
            num_steps = Y.shape[1]
            num_variables = X.shape[1]
            for X_j in X.chunk(num_variables, dim=1):
                outputs = self.forward(X_j[:, :, :-1])
                losses = self.compute_loss(outputs, X_j[:, :, 1:])
                X_losses.append(losses)

            # C x B x T -> B X T
            X_losses = torch.stack(X_losses).sum(dim=0)
            X_losses = X_losses[:, -num_steps:]

            Y_pred.append(X_losses.data.cpu().numpy().reshape((-1, )))
            Y_true.append(Y.data.cpu().numpy().reshape((-1, )))

        Y_pred = np.concatenate(Y_pred, axis=0)
        Y_true = np.concatenate(Y_true, axis=0)
        fp_list, tp_list, thresholds = metrics.roc_curve(Y_true, Y_pred)

        auc = metrics.auc(fp_list, tp_list)
        return auc

    def forward(self, X):
        """forward

        :param X: [B, C, T]

        return: [B, C, T]
        """

        # -> [B, 1, T]
        outputs = self.wavenet(X)
        return outputs

    def compute_loss(self, X_hat, X):
        num_steps = X.size()[2]
        losses = self.criterion(X_hat, X.squeeze(1).unsqueeze(2))
        losses = losses.squeeze(2)
        assert losses.size() == (X.size()[0], num_steps)
        # [B, T]
        return losses

    def start_train(self,
                    dataloader_trn,
                    dataloader_val,
                    lr,
                    beta1,
                    beta2,
                    num_epochs,
                    lr_decay=1.,
                    valid_iter=10,
                    max_patience=10,
                    max_num_trial=10,
                    save_path=None):
        optim = torch.optim.Adam(
            self.parameters(), lr=lr, betas=(beta1, beta2))

        if dataloader_val is not None:
            best_auc = self.eval_dataset(dataloader_val)
            best_iter = 0
            if save_path is not None:
                self.save(save_path + '/model.best')
        else:
            best_auc = best_iter = 0

        patience = num_trial = curr_iter = 0
        t_epoch = trange(num_epochs)
        for epoch in t_epoch:
            self.train()

            t = tqdm(dataloader_trn)
            for i, (X, _) in enumerate(t):
                X = X.to(self.device)
                num_variables = X.shape[1]
                loss_item = 0.
                for X_j in X.chunk(num_variables, dim=1):
                    optim.zero_grad()
                    outputs = self.forward(X_j[:, :, :-1])
                    losses = self.compute_loss(outputs, X_j[:, :, 1:])
                    loss = losses.mean()
                    loss.backward()
                    loss_item += loss.item()
                    optim.step()

                loss_item /= num_variables

                t_epoch.set_postfix(
                    epoch='{}/{}'.format(epoch, num_epochs),
                    batch='{}/{}'.format(i, len(dataloader_trn)),
                    patience=patience,
                    best_auc=best_auc,
                    best_iter=best_iter,
                    loss=loss.item())
                curr_iter += 1

                if curr_iter % valid_iter == 0:
                    if dataloader_val is not None:
                        auc = self.eval_dataset(dataloader_val)
                        if auc > best_auc:
                            best_auc = auc
                            best_iter = curr_iter
                            self.save(save_path + '/model.best')
                            patience = 0
                        else:
                            patience += 1
                            if patience >= max_patience:
                                num_trial += 1
                                if num_trial >= max_num_trial:
                                    print('early stop!', file=sys.stderr)
                                    return
                                lr = lr * lr_decay
                                print(
                                    'load previously best model and decay learning rate to %f'
                                    % lr,
                                    file=sys.stderr)
                                self.load_weights(save_path + '/model.best')
                    else:
                        self.save(save_path + '/model.current')


def train(args):
    # configurate seed
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = torch.device(
        'cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
    wavenet = WaveNetModel(
        num_mixtures=args.num_mixtures,
        num_layers=args.num_layers,
        device=device)
    wavenet.to(device)

    # labelled data sets
    trn_sets = []
    val_sets = []
    for path in args.data_paths:
        if 'yahoo' in path:
            data_set = WaveNetDataSet(
                path,
                receptive_field=wavenet.receptive_field,
                device=device,
                trn_ratio=0.50,
                val_ratio=0.75)
        else:
            data_set = WaveNetDataSet(
                path, receptive_field=wavenet.receptive_field, device=device)

        trn_sets.append(data_set.trn_set.with_horizon(args.horizon))
        val_sets.append(data_set.val_set.with_horizon(args.horizon))

    # unlabelled data sets
    for path in args.u_data_paths:
        data_set = WaveNetUDataSet(
            data_path=path,
            receptive_field=wavenet.receptive_field,
            horizon=args.horizon,
            device=device)
        trn_sets.append(data_set)

    trn_set = MergeDataset(*trn_sets)
    val_set = MergeDataset(*val_sets, sub_sample=args.sub_sample)

    dataloader_trn = DataLoader(
        trn_set, batch_size=args.batch_size, shuffle=True, num_workers=1)
    dataloader_val = DataLoader(
        val_set, batch_size=args.batch_size, num_workers=1)

    os.makedirs(args.save_path, exist_ok=True)
    wavenet.start_train(
        dataloader_trn=dataloader_trn,
        # dataloader_val=None,
        dataloader_val=dataloader_val,
        valid_iter=args.valid_iter,
        lr=args.lr,
        lr_decay=args.lr_decay,
        beta1=args.beta1,
        beta2=args.beta2,
        num_epochs=args.num_epochs,
        max_patience=args.max_patience,
        max_num_trial=args.max_num_trial,
        save_path=args.save_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-paths', nargs='+', required=True)
    parser.add_argument('--u-data-paths', nargs='+')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--num-layers', type=int, default=10)
    parser.add_argument('--num-mixtures', type=int, default=10)
    parser.add_argument('--save-path', default='models/wavenet')
    parser.add_argument('--num-epochs', type=int, default=1000)
    parser.add_argument('--max-num-trial', type=int, default=5)
    parser.add_argument('--max-patience', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--horizon', type=int, default=512)
    parser.add_argument('--valid-iter', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--lr-decay', type=float, default=0.5)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--seed', type=int, default=1127)
    parser.add_argument('--sub-sample', type=float)
    return parser.parse_args()


def main():
    args = parse_args()
    train(args)


if __name__ == '__main__':
    main()
