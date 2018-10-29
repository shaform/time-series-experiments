import argparse
import os

import numpy as np
import random
import torch
from sklearn import metrics
from numpy import genfromtxt
from torch.autograd import Variable
from tqdm import trange, tqdm
from torch.nn.utils import clip_grad_norm_

from ..models import BasicRNN, LSTNet
from ..data_loader import ForcastDataSet


class ForcastRNN(object):
    def __init__(self,
                 latent_size,
                 output_size,
                 input_size,
                 window_size,
                 model_type,
                 rnn_hidden_size,
                 cnn_hidden_size,
                 cnn_kernel_size,
                 rnn_skip_hidden_size,
                 skip_size,
                 highway_size,
                 dropout_rate,
                 predict_x,
                 grad_clip=10.,
                 device=None):
        self.device = device
        self.latent_size = latent_size
        self.window_size = window_size
        self.criteria = torch.nn.MSELoss()
        self.grad_clip = grad_clip

        self.input_size = input_size
        self.output_size = output_size
        self.rnn_hidden_size = rnn_hidden_size
        self.cnn_hidden_size = cnn_hidden_size
        self.rnn_skip_hidden = rnn_skip_hidden_size
        self.cnn_kernel_size = cnn_kernel_size
        self.skip_size = skip_size
        self.highway_size = highway_size
        self.model_type = model_type
        self.dropout_rate = dropout_rate

        if model_type == 'gru':
            self.rnn = BasicRNN(
                latent_size=latent_size,
                input_size=input_size,
                output_size=output_size,
                predict_x=predict_x)
        elif model_type == 'lstnet':
            self.rnn = LSTNet(
                rnn_hidden_size=rnn_hidden_size,
                cnn_hidden_size=cnn_hidden_size,
                cnn_kernel_size=cnn_kernel_size,
                rnn_skip_hidden_size=rnn_skip_hidden_size,
                skip_size=skip_size,
                highway_size=highway_size,
                dropout_rate=dropout_rate,
                window_size=window_size,
                output_size=output_size)

        self.criteria.to(device)
        self.rnn.to(device)

    def save(self, save_path):
        dirname = os.path.dirname(save_path)
        os.makedirs(dirname, exist_ok=True)
        torch.save(self, save_path)

    def eval_dataset(self, dataloader):
        self.rnn.eval()
        MSE = []
        for X, Y in dataloader:
            X = X.transpose(0, 1).contiguous()
            pp = self.rnn(X, predict_all=True)
            if isinstance(pp, tuple):
                pY, _ = pp
            else:
                pY = pp
            pY = pY[-1]
            MSE.append(self.criteria(pY, Y).item())
        loss = sum(MSE) / len(MSE)
        return loss

    def train(self,
              dataloader,
              lr,
              beta1,
              beta2,
              num_epochs,
              val_dataloader=None,
              save_path=None,
              finetune=False):
        optim = torch.optim.Adam(
            self.rnn.parameters(), lr=lr, betas=(beta1, beta2))

        if val_dataloader is not None:
            best_loss = self.eval_dataset(val_dataloader)
            best_epoch = 0
            if save_path is not None:
                self.save(save_path + '.best')
        else:
            best_loss = best_epoch = 0

        for epoch in trange(num_epochs):
            self.rnn.train()
            t = tqdm(dataloader)
            for i, (batch_data, batch_labels) in enumerate(t):
                real_batch_size = batch_data.shape[0]
                batch_data = batch_data.transpose(0, 1).contiguous()
                batch_labels = batch_labels.unsqueeze(0)
                all_labels = torch.cat([batch_data[1:], batch_labels],
                                       0).contiguous()
                optim.zero_grad()

                predicts, predicts_x = self.rnn(batch_data, predict_all=True)
                loss1 = self.criteria(predicts, all_labels)
                loss2 = self.criteria(predicts_x, all_labels)
                loss = loss1 + loss2
                loss.backward()
                clip_grad_norm_(self.rnn.parameters(), self.grad_clip)
                optim.step()

                t.set_postfix(
                    epoch='{}/{}'.format(epoch, num_epochs),
                    batch='{}/{}'.format(i, len(dataloader)),
                    best_loss=best_loss,
                    best_epoch=best_epoch,
                    loss1=loss1.item(),
                    loss2=loss2.item(),
                )

            if save_path is not None:
                if val_dataloader is not None:
                    loss = self.eval_dataset(val_dataloader)
                    if loss < best_loss:
                        best_loss = loss
                        best_epoch = epoch
                        self.save(save_path + '.best')
                    else:
                        print(loss, 'greater than', best_loss)
                else:
                    self.save(save_path + '.{}'.format(epoch))


def train(data_path, cuda, latent_size, window_size, save_path, num_epochs,
          batch_size, lr, beta1, beta2, grad_clip, model_type, rnn_hidden_size,
          cnn_hidden_size, cnn_kernel_size, rnn_skip_hidden_size, skip_size,
          highway_size, dropout_rate, predict_x, test, seed):
    # configurate seed
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    device = torch.device(
        'cuda' if cuda and torch.cuda.is_available() else 'cpu')
    dataloader = ForcastDataSet(
        data_path=data_path,
        window_size=window_size,
        batch_size=batch_size,
        device=device)
    if test:
        rnn = torch.load(save_path + '.best')
        loss = rnn.eval_dataset(dataloader.tst_set)
        print(loss)
    else:
        rnn = ForcastRNN(
            input_size=dataloader.num_variables,
            output_size=dataloader.num_variables,
            predict_x=predict_x,
            latent_size=latent_size,
            window_size=window_size,
            device=device,
            model_type=model_type,
            rnn_hidden_size=rnn_hidden_size,
            cnn_hidden_size=cnn_hidden_size,
            cnn_kernel_size=cnn_kernel_size,
            rnn_skip_hidden_size=rnn_skip_hidden_size,
            skip_size=skip_size,
            highway_size=highway_size,
            dropout_rate=dropout_rate,
            grad_clip=grad_clip)
        rnn.train(
            dataloader.trn_set,
            lr,
            beta1,
            beta2,
            num_epochs,
            save_path=save_path,
            val_dataloader=dataloader.val_set)

        rnn.save(save_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', required=True)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--latent-size', type=int, default=10)
    parser.add_argument('--window-size', type=int, default=25)
    parser.add_argument('--save-path', default='models/gan')
    parser.add_argument('--num-epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--model-type', default='gru')
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument(
        '--grad-clip', type=float, default=10.0, help='gradient clipping')
    parser.add_argument('--seed', type=int, default=1127)
    parser.add_argument(
        '--cnn-hidden-size',
        type=int,
        default=10,
        help='number of CNN hidden units for LSTNet')
    parser.add_argument(
        '--rnn-hidden-size',
        type=int,
        default=10,
        help='number of RNN hidden units for LSTNet')
    parser.add_argument(
        '--cnn-kernel-size',
        type=int,
        default=6,
        help='the kernel size of the CNN layers for LSTNet')
    parser.add_argument(
        '--highway-size',
        type=int,
        default=10,
        help='The window size of the highway component for LSTNet')
    parser.add_argument(
        '--skip-size',
        type=int,
        default=10,
        help='skip-length of RNN-skip layer for LSTNet')
    parser.add_argument(
        '--rnn-skip-hidden-size',
        type=int,
        default=5,
        help='hidden units nubmer of RNN-skip layer for LSTNet')
    parser.add_argument('--dropout-rate', type=float, default=0.)
    parser.add_argument('--predict-x', action='store_true')
    parser.add_argument('--test', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    train(**vars(args))


if __name__ == '__main__':
    main()
