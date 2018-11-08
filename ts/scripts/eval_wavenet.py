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
from .train_wavenet import WaveNetModel

def test(args):
    # configurate seed
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = torch.device(
        'cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
    wavenet = WaveNetModel.load(os.path.join(args.load_path, 'model.best'), device=device)

    # labelled data sets
    data_sets = []
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

        data_sets.append(data_set)

    aucs = []
    for data_set in data_sets:
        data_set.tst_set.with_horizon(args.horizon)
        if args.finetune:
            dataloader_trn = DataLoader(
                data_set.trn_set.with_horizon(args.horizon), batch_size=args.batch_size, num_workers=4, shuffle=True)
            dataloader_val = DataLoader(
                data_set.val_set.with_horizon(args.horizon), batch_size=args.batch_size, num_workers=1)
            wavenet.start_train(
                dataloader_trn=dataloader_trn,
                dataloader_val=dataloader_val,
                valid_iter=args.valid_iter,
                lr=args.lr,
                lr_decay=args.lr_decay,
                beta1=args.beta1,
                beta2=args.beta2,
                num_epochs=args.num_epochs,
                max_patience=args.max_patience,
                max_num_trial=args.max_num_trial,
                save_path=os.path.join(args.load_path, os.path.basename(data_set.data_path)))

        dataloader_tst = DataLoader(
            data_set.tst_set.with_horizon(args.horizon), batch_size=args.batch_size, num_workers=1)
        auc = wavenet.eval_dataset(dataloader_tst)
        aucs.append(auc)

    print('{}\t{}'.format(args.label, sum(aucs) / len(aucs)))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--label', required=True)
    parser.add_argument('--data-paths', nargs='+', required=True)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--load-path', default='models/wavenet')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--horizon', type=int, default=512)
    parser.add_argument('--finetune', action='store_true')
    parser.add_argument('--num-epochs', type=int, default=1000)
    parser.add_argument('--max-num-trial', type=int, default=5)
    parser.add_argument('--max-patience', type=int, default=5)
    parser.add_argument('--valid-iter', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--lr-decay', type=float, default=0.5)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--seed', type=int, default=1127)
    return parser.parse_args()


def main():
    args = parse_args()
    test(args)


if __name__ == '__main__':
    main()
