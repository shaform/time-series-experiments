import argparse
import sys
import time
import os

import numpy as np
import random
import torch
from sklearn import metrics
from tqdm import trange, tqdm
from torch.nn.utils import clip_grad_norm_

from ..models import BasicRNN, LSTNet
from ..data_loader import LabeledDataSet


class CPDRNN(object):
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
                 bidirectional,
                 predict_horizon=False,
                 predict_each=False,
                 grad_clip=10.,
                 device=None):
        self.device = device
        self.latent_size = latent_size
        self.window_size = window_size
        self.criteria = torch.nn.MSELoss()
        self.grad_clip = grad_clip
        self.bidirectional = bidirectional

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
        self.predict_horizon = predict_horizon
        self.predict_each = predict_each

        if predict_each:
            actual_input_size = 1
        else:
            actual_input_size = self.input_size

        if predict_horizon:
            if predict_each:
                actual_output_size = self.window_size
            else:
                actual_output_size = self.window_size * self.output_size
        else:
            if predict_each:
                actual_output_size = 1
            else:
                actual_output_size = self.output_size

        if model_type == 'gru':
            self.rnn = BasicRNN(
                latent_size=latent_size,
                input_size=actual_input_size,
                output_size=actual_output_size)
            if bidirectional:
                self.back_rnn = BasicRNN(
                    latent_size=latent_size,
                    input_size=actual_input_size,
                    output_size=actual_output_size)
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
                input_size=actual_input_size,
                output_size=actual_output_size)
            if bidirectional:
                self.back_rnn = LSTNet(
                    rnn_hidden_size=rnn_hidden_size,
                    cnn_hidden_size=cnn_hidden_size,
                    cnn_kernel_size=cnn_kernel_size,
                    rnn_skip_hidden_size=rnn_skip_hidden_size,
                    skip_size=skip_size,
                    highway_size=highway_size,
                    dropout_rate=dropout_rate,
                    window_size=window_size,
                    input_size=actual_input_size,
                    output_size=actual_output_size)

        self.criteria.to(device)
        self.rnn.to(device)
        if bidirectional:
            self.back_rnn.to(device)

    def set_train(self):
        self.rnn.train()
        if self.bidirectional:
            self.back_rnn.train()

    def set_eval(self):
        self.rnn.eval()
        if self.bidirectional:
            self.back_rnn.eval()

    def parameters(self):
        for p in self.rnn.parameters():
            yield p
        if self.bidirectional:
            for p in self.back_rnn.parameters():
                yield p

    def save(self, save_path):
        dirname = os.path.dirname(save_path)
        os.makedirs(dirname, exist_ok=True)
        torch.save(self, save_path)

    def load(self, load_path):
        model = torch.load(load_path)
        self.rnn = model.rnn
        if self.bidirectional:
            self.back_rnn = model.back_rnn

    def eval_dataset(self, dataloader):
        self.set_eval()
        Y_pred = []
        L_true = []
        for X, L in dataloader:
            X = X.transpose(0, 1).contiguous()

            forward_X, backward_X = torch.split(
                X, [self.window_size, self.window_size], dim=0)

            backward_X = torch.flip(backward_X, dims=(0, ))
            if self.predict_horizon:
                forward_y = backward_X.transpose(0, 1)
                backward_y = forward_X.transpose(0, 1)
            else:
                forward_y = backward_X[-1]
                backward_y = forward_X[-1]

            if self.predict_each:
                forward_y_hat = []
                for i in range(X.shape[2]):
                    forward_y_hat_each = self.rnn(forward_X[:, :, i:i + 1])
                    forward_y_hat.append(forward_y_hat_each.unsqueeze(-1))
                forward_y_hat = torch.cat(forward_y_hat, dim=-1)
            else:
                forward_y_hat = self.rnn(forward_X)
            # -> [B, C]

            forward_y_scores = ((
                forward_y_hat.contiguous().view(forward_y_hat.size(0), -1) -
                forward_y.contiguous().view(forward_y_hat.size(0), -1))
                                **2).sum(-1)
            if len(forward_y_scores.size()) > 1:
                forward_y_scores = forward_y_scores.sum(-1)
            # -> B
            scores = forward_y_scores

            if self.bidirectional:
                if self.predict_each:
                    backward_y_hat = []
                    for i in range(X.shape[2]):
                        backward_y_hat_each = self.back_rnn(
                            backward_X[:, :, i:i + 1])
                        backward_y_hat.append(
                            backward_y_hat_each.unsqueeze(-1))
                    backward_y_hat = torch.cat(backward_y_hat, dim=-1)
                else:
                    backward_y_hat = self.back_rnn(backward_X)
                backward_y_scores = ((backward_y_hat.contiguous().view(
                    backward_y_hat.size(0), -1) - backward_y.contiguous().view(
                        backward_y_hat.size(0), -1))**2).sum(-1)
                if len(backward_y_scores.size()) > 1:
                    backward_y_scores = backward_y_scores.sum(-1)
                scores += backward_y_scores

            Y_pred.append(scores.data.cpu().numpy())
            L_true.append(L)
        Y_pred = np.concatenate(Y_pred, axis=0)
        L_true = np.concatenate(L_true, axis=0)
        fp_list, tp_list, thresholds = metrics.roc_curve(L_true, Y_pred)

        auc = metrics.auc(fp_list, tp_list)
        self.set_train()
        return auc

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
            self.parameters(),
            lr=lr,
            betas=(beta1, beta2),
            weight_decay=weight_decay)

        if val_dataloader is not None:
            best_auc = 0
            if save_path is not None:
                self.save(save_path + '.best')
        else:
            best_auc = 0

        train_iter = hit_patience = num_trial = 0
        report_loss = report_examples = 0
        cumulative_examples = 0
        begin_time = time.time()
        self.set_train()
        for epoch in trange(num_epochs):
            t = tqdm(dataloader)
            for i, X in enumerate(t):
                real_batch_size = X.shape[0]
                if len(X.size()) == 2:
                    X = X.transpose(0, 1).unsqueeze(-1).contiguous()
                else:
                    X = X.transpose(0, 1).contiguous()
                forward_X, backward_X = torch.split(
                    X, [self.window_size, self.window_size], dim=0)
                backward_X = torch.flip(backward_X, dims=(0, ))
                # [wind_size, batch, 1]

                if self.predict_horizon:
                    forward_y = backward_X.transpose(0, 1)
                    backward_y = forward_X.transpose(0, 1)
                else:
                    forward_y = backward_X[-1]
                    backward_y = forward_X[-1]

                optim.zero_grad()
                if self.predict_each:
                    forward_y_hat = []
                    for i in range(X.shape[2]):
                        forward_y_hat_each = self.rnn(forward_X[:, :, i:i + 1])
                        forward_y_hat.append(forward_y_hat_each.unsqueeze(-1))
                    forward_y_hat = torch.cat(forward_y_hat, dim=-1)
                else:
                    forward_y_hat = self.rnn(forward_X)
                f_loss = self.criteria(
                    forward_y_hat.contiguous().view(forward_y_hat.size(0), -1),
                    forward_y.contiguous().view(forward_y_hat.size(0), -1))

                if self.bidirectional:
                    if self.predict_each:
                        backward_y_hat = []
                        for i in range(X.shape[2]):
                            backward_y_hat_each = self.back_rnn(
                                backward_X[:, :, i:i + 1])
                            backward_y_hat.append(
                                backward_y_hat_each.unsqueeze(-1))
                        backward_y_hat = torch.cat(backward_y_hat, dim=-1)
                    else:
                        backward_y_hat = self.back_rnn(backward_X)
                    b_loss = self.criteria(
                        backward_y_hat.view(
                            backward_y_hat.contiguous().size(0), -1),
                        backward_y.contiguous().view(
                            backward_y_hat.size(0), -1))
                    loss = (f_loss + b_loss) / 2.
                else:
                    loss = f_loss

                loss.backward()
                clip_grad_norm_(self.parameters(), self.grad_clip)
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
                    auc = self.eval_dataset(val_dataloader)
                    print(
                        'validation: iter %d, dev. auc %f' % (train_iter, auc),
                        file=sys.stderr)

                    if auc > best_auc:
                        best_auc = auc
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
                                self.parameters(),
                                lr=lr,
                                weight_decay=weight_decay,
                                betas=(beta1, beta2))

                            print(
                                'restore parameters of the optimizers',
                                file=sys.stderr)
                            # You may also need to load the state of the optimizer saved before
                            state = torch.load(
                                os.path.join(save_path, 'opt.pth'))
                            optim.load_state_dict(state['optimizer'])

                            hit_patience = 0

                t.set_postfix(
                    epoch='{}/{}'.format(epoch, num_epochs),
                    batch='{}/{}'.format(i, len(dataloader)),
                    best_auc=best_auc,
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
    if 'yahoo' in args.data_path:
        dataloader = LabeledDataSet(
            batch_size=args.batch_size,
            data_path=args.data_path,
            window_size=args.window_size,
            device=device,
            shuffle=False,
            trn_ratio=0.50,
            val_ratio=0.75)
    else:
        dataloader = LabeledDataSet(
            batch_size=args.batch_size,
            data_path=args.data_path,
            window_size=args.window_size,
            device=device,
            shuffle=False)
    if args.test:
        rnn = torch.load(os.path.join(args.save_path, 'model.best'))
        loss = rnn.eval_dataset(dataloader.tst_set)
        print(loss)
    else:
        rnn = CPDRNN(
            input_size=dataloader.num_variables,
            output_size=dataloader.num_variables,
            latent_size=args.latent_size,
            window_size=args.window_size,
            device=device,
            bidirectional=args.bidirectional,
            model_type=args.model_type,
            rnn_hidden_size=args.rnn_hidden_size,
            cnn_hidden_size=args.cnn_hidden_size,
            cnn_kernel_size=args.cnn_kernel_size,
            rnn_skip_hidden_size=args.rnn_skip_hidden_size,
            skip_size=args.skip_size,
            highway_size=args.highway_size,
            dropout_rate=args.dropout_rate,
            predict_horizon=args.predict_horizon,
            predict_each=args.predict_each,
            grad_clip=args.grad_clip)
        rnn.train(
            dataloader=dataloader.trn_set.unlabelled(full=True),
            lr=args.lr,
            beta1=args.beta1,
            beta2=args.beta2,
            patience=args.patience,
            log_every=args.log_every,
            valid_niter=args.valid_niter,
            max_num_trial=args.max_num_trial,
            num_epochs=args.num_epochs,
            save_path=args.save_path,
            weight_decay=args.weight_decay,
            val_dataloader=dataloader.val_set)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', required=True)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--latent-size', type=int, default=10)
    parser.add_argument('--window-size', type=int, default=25)
    parser.add_argument('--save-path', default='models/gan')
    parser.add_argument('--num-epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=10)
    parser.add_argument('--model-type', default='lstnet')
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument(
        '--grad-clip', type=float, default=10.0, help='gradient clipping')
    parser.add_argument('--log-every', type=int, default=5)
    parser.add_argument('--patience', type=int, default=10)
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
    parser.add_argument('--bidirectional', action='store_true')
    parser.add_argument('--predict-horizon', action='store_true')
    parser.add_argument('--predict-each', action='store_true')
    parser.add_argument('--weight-decay', type=float, default=0.001)
    return parser.parse_args()


def main():
    args = parse_args()
    train(args)


if __name__ == '__main__':
    main()
