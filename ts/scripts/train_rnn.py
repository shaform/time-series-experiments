import argparse
import os

import numpy as np
import random
import torch
from numpy import genfromtxt
from torch.autograd import Variable
from tqdm import trange, tqdm
from torch.nn.utils import clip_grad_norm_

from ..models import BasicRNN
from ..data_loader import UnlabeledDataLoader


class RNN(object):
    def __init__(self,
                 latent_size,
                 window_size,
                 bidirectional=True,
                 grad_clip=10.,
                 device=None):
        self.device = device
        self.latent_size = latent_size
        self.window_size = window_size
        self.criteria = torch.nn.MSELoss()
        self.bidirectional = bidirectional
        self.grad_clip = grad_clip
        self.forward_rnn = BasicRNN(
            latent_size=latent_size, input_size=1, output_size=1)
        if bidirectional:
            self.backward_rnn = BasicRNN(
                latent_size=latent_size, input_size=1, output_size=1)
        self.backward_rnn.to(device)

        self.criteria.to(device)
        self.forward_rnn.to(device)

    def save(self, save_path):
        dirname = os.path.dirname(save_path)
        os.makedirs(dirname, exist_ok=True)
        torch.save(self, save_path)

    def train(self,
              dataloader,
              lr,
              beta1,
              beta2,
              num_epochs,
              save_path=None,
              finetune=False):
        if self.bidirectional:
            optim = torch.optim.Adam(
                list(self.forward_rnn.parameters()) + list(
                    self.backward_rnn.parameters()),
                lr=lr,
                betas=(beta1, beta2))
            self.backward_rnn.train()
        else:
            optim = torch.optim.Adam(
                self.forward_rnn.parameters(), lr=lr, betas=(beta1, beta2))
        self.forward_rnn.train()
        for epoch in trange(num_epochs):
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
                    f_loss=f_loss_item,
                    b_loss=b_loss_item)

                if save_path is not None:
                    self.save(save_path + '.{}'.format(epoch))


def train(data_paths, cuda, latent_size, window_size, save_path, num_epochs,
          batch_size, lr, beta1, beta2, grad_clip, seed):
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
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument(
        '--grad-clip',
        type=float,
        default=10.0,
        help='gradient clipping for RNN')
    parser.add_argument('--seed', type=int, default=1127)
    return parser.parse_args()


def main():
    args = parse_args()
    train(**vars(args))


if __name__ == '__main__':
    main()
