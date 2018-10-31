import argparse
import time
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
                output_size=output_size)
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

    def load(self, load_path):
        model = torch.load(load_path)
        self.rnn = model.rnn

    def eval_dataset(self, dataloader):
        self.rnn.eval()
        with torch.no_grad():
            MSE = []
            for X, Y in dataloader:
                X = X.transpose(0, 1).contiguous()
                pY = self.rnn(X)
                MSE.append(self.criteria(pY, Y).item())
            loss = sum(MSE) / len(MSE)
        self.rnn.train()
        return loss

    def train(self,
              dataloader,
              lr,
              beta1,
              beta2,
              num_epochs,
              weight_decay=0.,
              patience=5,
              log_every=50,
              max_num_trial=5,
              lr_decay=0.5,
              valid_niter=0,
              val_dataloader=None,
              save_path=None,
              finetune=False):
        optim = torch.optim.Adam(
            self.rnn.parameters(), lr=lr, betas=(beta1, beta2))

        if val_dataloader is not None:
            best_loss = self.eval_dataset(val_dataloader)
            if save_path is not None:
                self.save(save_path + '.best')
        else:
            best_loss = 0

        train_iter = patience = hit_patience = num_trial = 0
        report_loss = report_examples = 0
        cumulative_examples = 0
        begin_time = time.time()
        self.rnn.train()
        for epoch in trange(num_epochs):
            t = tqdm(dataloader)
            for i, (batch_data, batch_labels) in enumerate(t):
                real_batch_size = batch_data.shape[0]
                batch_data = batch_data.transpose(0, 1).contiguous()
                batch_labels = batch_labels.unsqueeze(0)
                optim.zero_grad()

                predicts = self.rnn(batch_data)
                loss = self.criteria(predicts, batch_labels)
                loss.backward()
                clip_grad_norm_(self.rnn.parameters(), self.grad_clip)
                optim.step()
                report_loss += loss.data.cpu().numpy() * real_batch_size
                report_examples += real_batch_size
                cumulative_examples += real_batch_size

                train_iter += 1
                if train_iter % log_every == 0:
                    print(
                        'epoch %d, iter %d, avg. loss %.2f, '
                        'cum. examples %d, time elapsed %.2f sec' %
                        (epoch, train_iter, report_loss / report_examples,
                         cumulative_examples, time.time() - begin_time),
                        file=sys.stderr)
                    report_loss = report_examples = 0.

                if train_iter % valid_niter == 0 and val_dataloader is not None and save_path is not None:
                    loss = self.eval_dataset(val_dataloader)
                    print(
                        'validation: iter %d, dev. loss %f' % (train_iter,
                                                               loss),
                        file=sys.stderr)

                    if loss < best_loss:
                        best_loss = loss
                        hit_patience = 0
                        print(
                            'save currently the best model to [%s]' %
                            save_path,
                            file=sys.stderr)
                        self.save(os.path.join(save_path, 'model.best'))
                        state = {
                            'optimizer': optim.state_dict(),
                        }
                        torch.save(state, os.path.join(save_path, 'opt.pth'))
                    elif hit_patience < patience:
                        hit_patience += 1
                        print(
                            'hit patience %d' % hit_patience, file=sys.stderr)
                        if hit_patience == patience:
                            num_trial += 1
                        print('hit #%d trial' % num_trial, file=sys.stderr)
                        if num_trial == max_num_trial:
                            print('early stop!', file=sys.stderr)
                            return

                        lr = lr * lr_decay
                        print(
                            'load previously best model and decay learning rate to %f'
                            % lr,
                            file=sys.stderr)
                        self.load(os.path.join(save_path, 'model.best'))
                        optim = torch.optim.Adam(
                            self.rnn.parameters(), lr=lr, weight_decay)

                        print(
                            'restore parameters of the optimizers',
                            file=sys.stderr)
                        # You may also need to load the state of the optimizer saved before
                        state = torch.load(os.path.join(save_path, 'opt.pth'))
                        optim.load_state_dict(state['optimizer'])

                        hit_patience = 0

                t.set_postfix(
                    epoch='{}/{}'.format(epoch, num_epochs),
                    batch='{}/{}'.format(i, len(dataloader)),
                    best_loss=best_loss,
                    loss=loss.item(),
                )

            if save_path is not None:
                self.save(os.path.join(save_path, 'model.current'))


def train(args):
    # configurate seed
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = torch.device(
        'cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
    dataloader = ForcastDataSet(
        data_path=args.data_path,
        window_size=args.window_size,
        batch_size=args.batch_size,
        device=args.device)
    if test:
        rnn = torch.load(args.save_path + '.best')
        loss = rnn.eval_dataset(dataloader.tst_set)
        print(loss)
    else:
        rnn = ForcastRNN(
            input_size=dataloader.num_variables,
            output_size=dataloader.num_variables,
            latent_size=args.latent_size,
            window_size=args.window_size,
            device=args.device,
            model_type=args.model_type,
            rnn_hidden_size=args.rnn_hidden_size,
            cnn_hidden_size=args.cnn_hidden_size,
            cnn_kernel_size=args.cnn_kernel_size,
            rnn_skip_hidden_size=args.rnn_skip_hidden_size,
            skip_size=args.skip_size,
            highway_size=args.highway_size,
            dropout_rate=args.dropout_rate,
            grad_clip=args.grad_clip)
        rnn.train(
            dataloader=dataloader.trn_set,
            lr=args.lr,
            beta1=args.beta1,
            beta2=args.beta2,
            patience=args.patience,
            log_every=args.log_every,
            valid_niter=args.valid_niter,
            max_num_trial=args.max_num_trial,
            num_epochs=args.num_epochs,
            save_path=args.save_path,
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
    parser.add_argument('--model-type', default='lstnet')
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument(
        '--grad-clip', type=float, default=10.0, help='gradient clipping')
    parser.add_argument('--log-every', type=int, default=50)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--max-num-trial', type=int, default=5)
    parser.add_argument('--lr-decay', type=float, default=0.5)
    parser.add_argument('--valid-niter', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=1126)
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
    parser.add_argument('--dropout-rate', type=float, default=0.2)
    parser.add_argument('--test', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    train(args)


if __name__ == '__main__':
    main()
