import argparse
import os

import numpy as np
import random
import torch
from numpy import genfromtxt
from torch.autograd import Variable
from tqdm import trange, tqdm

from ..models import BasicGenerator, BasicDiscriminator
from ..data_loader import UnlabeledDataLoader


class GAN(object):
    def __init__(self, latent_size, window_size, device=None):
        self.device = device
        self.latent_size = latent_size
        self.window_size = window_size
        self.criteria = torch.nn.BCEWithLogitsLoss()
        self.gen = BasicGenerator(
            latent_size=latent_size,
            input_size=window_size,
            output_size=window_size)
        self.disc = BasicDiscriminator(input_size=window_size * 2)

        self.criteria.to(device)
        self.gen.to(device)
        self.disc.to(device)

    def save(self, save_path):
        dirname = os.path.dirname(save_path)
        os.makedirs(dirname, exist_ok=True)
        torch.save(self, save_path)

    def train(self, dataloader, lr, beta1, beta2, num_epochs, save_path=None, finetune=False):
        g_optim = torch.optim.Adam(
            self.gen.parameters(), lr=lr, betas=(beta1, beta2))
        d_optim = torch.optim.Adam(
            self.disc.parameters(), lr=lr, betas=(beta1, beta2))
        self.gen.train()
        self.disc.train()
        if finetune:
            for layer in self.disc.model[:-1]:
                for param in layer.parameters():
                    param.require_grad = True
        for epoch in trange(num_epochs):
            t = tqdm(dataloader)
            for i, batch_data in enumerate(t):
                real_batch_size = batch_data.shape[0]
                zeros = Variable(
                    torch.zeros(
                        real_batch_size,
                        1,
                        dtype=torch.float,
                        device=self.device),
                    requires_grad=False)
                ones = Variable(
                    torch.ones(
                        real_batch_size,
                        1,
                        dtype=torch.float,
                        device=self.device),
                    requires_grad=False)
                c, r = torch.split(
                    batch_data, [self.window_size, self.window_size], dim=-1)

                # generator
                g_optim.zero_grad()
                z = Variable(
                    torch.tensor(
                        np.random.normal(0, 1,
                                         (real_batch_size, self.latent_size)),
                        dtype=torch.float,
                        device=self.device),
                    requires_grad=False)
                g_half = self.gen(z, c)
                fake = torch.cat([c, g_half], dim=-1)

                g_loss = self.criteria(self.disc(fake), ones)

                g_loss.backward()
                g_optim.step()

                # discriminator
                d_optim.zero_grad()

                d_real_loss = self.criteria(self.disc(batch_data), ones)
                d_fake_loss = self.criteria(self.disc(fake.detach()), zeros)
                d_g_loss = (d_real_loss + d_fake_loss) / 2

                # discriminator for random real data
                batch_perm = batch_data[torch.randperm(real_batch_size)]
                batch_cat = torch.cat([batch_data, batch_perm], dim=-1)
                diff = np.random.randint(0, 2 * self.window_size + 1)
                batch_drift = batch_cat[:, diff:diff + 2 * self.window_size]
                drift = ones * (
                    np.abs(diff - self.window_size) / self.window_size)
                d_drift_loss = self.criteria(
                    self.disc(batch_drift.detach()), drift)
                d_loss = d_g_loss + d_drift_loss

                d_loss.backward()
                d_optim.step()

                if finetune:
                    t.set_postfix(
                        epoch='{}/{}'.format(epoch, num_epochs),
                        batch='{}/{}'.format(i, len(dataloader)),
                        d_loss=d_loss.item())
                else:
                    t.set_postfix(
                            f_mean=g_half.mean().item(),
                        r_mean=r.mean().item(),
                        epoch='{}/{}'.format(epoch, num_epochs),
                        batch='{}/{}'.format(i, len(dataloader)),
                        d_loss=d_loss.item(),
                        g_loss=g_loss.item())

            if save_path is not None:
                self.save(save_path + '.{}'.format(epoch))


def train(data_paths, cuda, latent_size, window_size, save_path, num_epochs,
          batch_size, lr, beta1, beta2, seed):
    # configurate seed
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    device = torch.device(
        'cuda' if cuda and torch.cuda.is_available() else 'cpu')
    gan = GAN(latent_size=latent_size, window_size=window_size, device=device)
    dataloader = UnlabeledDataLoader(
        data_paths=data_paths,
        window_size=window_size,
        batch_size=batch_size,
        device=device)
    gan.train(dataloader, lr, beta1, beta2, num_epochs, save_path=save_path)

    gan.save(save_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-paths', nargs='+', required=True)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--latent-size', type=int, default=64)
    parser.add_argument('--window-size', type=int, default=25)
    parser.add_argument('--save-path', default='models/gan')
    parser.add_argument('--num-epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--seed', type=int, default=1127)
    return parser.parse_args()


def main():
    args = parse_args()
    train(**vars(args))


if __name__ == '__main__':
    main()
