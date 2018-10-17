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
from ..data_loader import UnlabeledDataLoader


class RNN(object):
    def __init__(self,
                 latent_size,
                 window_size,
                 model_type,
                 rnn_hidden_size,
                 cnn_hidden_size,
                 cnn_kernel_size,
                 rnn_skip_hidden_size,
                 skip_size,
                 highway_size,
                 dropout_rate,
                 bidirectional=True,
                 grad_clip=10.,
                 device=None):
        self.device = device
        self.latent_size = latent_size
        self.window_size = window_size
        self.criteria = torch.nn.MSELoss()
        self.bidirectional = bidirectional
        self.grad_clip = grad_clip

        self.rnn_hidden_size = rnn_hidden_size
        self.cnn_hidden_size = cnn_hidden_size
        self.rnn_skip_hidden = rnn_skip_hidden_size
        self.cnn_kernel_size = cnn_kernel_size
        self.skip_size = skip_size
        self.highway_size = highway_size
        self.model_type = model_type
        self.dropout_rate = dropout_rate

        if model_type == 'gru':
            self.forward_rnn = BasicRNN(
                latent_size=latent_size, input_size=1, output_size=1)
            if bidirectional:
                self.backward_rnn = BasicRNN(
                    latent_size=latent_size, input_size=1, output_size=1)
        elif model_type == 'lstnet':
            self.forward_rnn = LSTNet(
                rnn_hidden_size=rnn_hidden_size,
                cnn_hidden_size=cnn_hidden_size,
                cnn_kernel_size=cnn_kernel_size,
                rnn_skip_hidden_size=rnn_skip_hidden_size,
                skip_size=skip_size,
                highway_size=highway_size,
                dropout_rate=dropout_rate,
                window_size=window_size,
                output_size=1)
            if bidirectional:
                self.backward_rnn = BasicRNN(
                    latent_size=latent_size, input_size=1, output_size=1)

        self.criteria.to(device)
        self.forward_rnn.to(device)
        if bidirectional:
            self.backward_rnn.to(device)

    def save(self, save_path):
        dirname = os.path.dirname(save_path)
        os.makedirs(dirname, exist_ok=True)
        torch.save(self, save_path)

    def eval_dataset(self, dataloader):
        self.forward_rnn.eval()
        if self.bidirectional:
            self.backward_rnn.eval()
        Y_pred = []
        L_true = []
        for Y, L in dataloader:
            Y = Y.transpose(0, 1).contiguous()
            forward_Y, backward_Y = torch.split(
                Y, [self.window_size, self.window_size], dim=0)
            backward_Y = torch.flip(backward_Y, dims=(0, ))
            forward_y = backward_Y[-1]
            backward_y = forward_Y[-1]
            Y_pred_batch = None
            for i in range(Y.shape[2]):
                a_forward_y = self.forward_rnn(forward_Y[:, :, i:i + 1])
                a_forward_y = (a_forward_y - forward_y[:, i:i + 1])**2
                if self.bidirectional:
                    a_backward_y = self.backward_rnn(backward_Y[:, :, i:i + 1])
                    a_backward_y = (a_backward_y - backward_y[:, i:i + 1])**2
                    a_Y = a_forward_y + a_backward_y
                else:
                    a_Y = a_forward_y
                if Y_pred_batch is None:
                    Y_pred_batch = a_Y
                else:
                    Y_pred_batch += a_Y
            Y_pred.append(Y_pred_batch.data.cpu().numpy())
            L_true.append(L)
        Y_pred = np.concatenate(Y_pred, axis=0)
        L_true = np.concatenate(L_true, axis=0)
        fp_list, tp_list, thresholds = metrics.roc_curve(L_true, Y_pred)

        auc = metrics.auc(fp_list, tp_list)
        return auc

    def train(self,
              dataloader,
              lr,
              beta1,
              beta2,
              num_epochs,
              val_dataloader=None,
              save_path=None,
              finetune=False):
        if self.bidirectional:
            optim = torch.optim.Adam(
                list(self.forward_rnn.parameters()) + list(
                    self.backward_rnn.parameters()),
                lr=lr,
                betas=(beta1, beta2))
        else:
            optim = torch.optim.Adam(
                self.forward_rnn.parameters(), lr=lr, betas=(beta1, beta2))
        if val_dataloader is not None:
            best_auc = self.eval_dataset(val_dataloader)
            best_epoch = 0
            if save_path is not None:
                self.save(save_path + '.best')
        else:
            best_auc = best_epoch = 0

        for epoch in trange(num_epochs):
            self.forward_rnn.train()
            if self.bidirectional:
                self.backward_rnn.train()
            t = tqdm(dataloader)
            for i, batch_data in enumerate(t):
                real_batch_size = batch_data.shape[0]
                batch_data = batch_data.transpose(
                    0, 1).unsqueeze(-1).contiguous()
                forward_x, backward_x = torch.split(
                    batch_data, [self.window_size, self.window_size], dim=0)
                backward_x = torch.flip(backward_x, dims=(0, ))
                # [wind_size, batch, 1]

                forward_y = backward_x[-1]
                backward_y = forward_x[-1]
                # [1, batch, 1]

                optim.zero_grad()
                pred_forward_y = self.forward_rnn(forward_x)
                f_loss = self.criteria(pred_forward_y, forward_y)

                if self.bidirectional:
                    pred_backward_y = self.backward_rnn(backward_x)
                    b_loss = self.criteria(pred_backward_y, backward_y)
                    loss = (f_loss + b_loss) / 2.
                    f_loss_item = f_loss.item()
                    b_loss_item = b_loss.item()
                else:
                    loss = f_loss
                    f_loss_item = f_loss.item()
                    b_loss_item = 0.

                loss.backward()
                if self.bidirectional:
                    clip_grad_norm_(
                        list(self.backward_rnn.parameters()) + list(
                            self.forward_rnn.parameters()), self.grad_clip)
                else:
                    clip_grad_norm_(self.forward_rnn.parameters(),
                                    self.grad_clip)
                optim.step()

                t.set_postfix(
                    epoch='{}/{}'.format(epoch, num_epochs),
                    batch='{}/{}'.format(i, len(dataloader)),
                    best_auc=best_auc,
                    best_epoch=best_epoch,
                    f_loss=f_loss_item,
                    b_loss=b_loss_item)

            if save_path is not None:
                if val_dataloader is not None:
                    auc = self.eval_dataset(val_dataloader)
                    if auc > best_auc:
                        best_auc = auc
                        best_epoch = epoch
                        self.save(save_path + '.best')
                else:
                    self.save(save_path + '.{}'.format(epoch))


def train(data_paths, cuda, latent_size, window_size, save_path, num_epochs,
          batch_size, lr, beta1, beta2, grad_clip, model_type, rnn_hidden_size,
          cnn_hidden_size, cnn_kernel_size, rnn_skip_hidden_size, skip_size,
          highway_size, dropout_rate, seed):
    # configurate seed
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    device = torch.device(
        'cuda' if cuda and torch.cuda.is_available() else 'cpu')
    rnn = RNN(
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
    dataloader = UnlabeledDataLoader(
        data_paths=data_paths,
        window_size=window_size,
        batch_size=batch_size,
        device=device)
    rnn.train(dataloader, lr, beta1, beta2, num_epochs, save_path=save_path)

    rnn.save(save_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-paths', nargs='+', required=True)
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
    return parser.parse_args()


def main():
    args = parse_args()
    train(**vars(args))


if __name__ == '__main__':
    main()
