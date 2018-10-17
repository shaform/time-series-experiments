import argparse
import os

import torch
from sklearn import metrics
import numpy as np
import sklearn

from ..data_loader import LabeledDataSet
from ..scripts.train_rnn import RNN
parser = argparse.ArgumentParser()
parser.add_argument('--model-path', required=True)
parser.add_argument('--bidirectional', action='store_true')
parser.add_argument('--finetune', action='store_true')
parser.add_argument('--window-size', type=int, default=25)
args = parser.parse_args()

data_sets = [
    [
        'parsed_data/labeled/beedance-1.npy',
        'parsed_data/labeled/beedance-2.npy',
        'parsed_data/labeled/beedance-3.npy',
        'parsed_data/labeled/beedance-4.npy',
        'parsed_data/labeled/beedance-5.npy',
        'parsed_data/labeled/beedance-6.npy'
    ],
    ['parsed_data/labeled/fishkiller.npy'],
    ['parsed_data/labeled/hasc.npy'],
    [
        'parsed_data/labeled/yahoo-7.npy', 'parsed_data/labeled/yahoo-8.npy',
        'parsed_data/labeled/yahoo-16.npy', 'parsed_data/labeled/yahoo-22.npy',
        'parsed_data/labeled/yahoo-27.npy', 'parsed_data/labeled/yahoo-33.npy',
        'parsed_data/labeled/yahoo-37.npy', 'parsed_data/labeled/yahoo-42.npy',
        'parsed_data/labeled/yahoo-45.npy', 'parsed_data/labeled/yahoo-46.npy',
        'parsed_data/labeled/yahoo-50.npy', 'parsed_data/labeled/yahoo-51.npy',
        'parsed_data/labeled/yahoo-54.npy', 'parsed_data/labeled/yahoo-55.npy',
        'parsed_data/labeled/yahoo-56.npy'
    ],
]

for data_set_list in data_sets:
    print('=====', data_set_list, '====')
    aucs = []
    for data_set in data_set_list:
        d = LabeledDataSet(
            data_set, device=torch.device('cuda'), shuffle=False)
        rnn = torch.load(args.model_path)
        if args.finetune:
            finetune_path = args.model_path + '.' + os.path.basename(data_set)
            best_path = finetune_path + '.best'
            if not os.path.exists(best_path):
                rnn.train(
                    d.trn_set.unlabelled(),
                    lr=0.00005,
                    beta1=0.5,
                    beta2=0.999,
                    num_epochs=20,
                    val_dataloader=d.val_set,
                    save_path=finetune_path)
            rnn = torch.load(best_path)

        rnn.forward_rnn.eval()
        if args.bidirectional:
            rnn.backward_rnn.eval()
        Y_pred = []
        L_true = []
        for Y, L in d.tst_set:
            Y = Y.transpose(0, 1).contiguous()
            forward_Y, backward_Y = torch.split(
                Y, [args.window_size, args.window_size], dim=0)
            backward_Y = torch.flip(backward_Y, dims=(0, ))
            forward_y = backward_Y[-1]
            backward_y = forward_Y[-1]
            Y_pred_batch = None
            for i in range(Y.shape[2]):
                a_forward_y = rnn.forward_rnn(forward_Y[:, :, i:i + 1])
                # print(a_forward_y.size())
                # [batch, 1]
                # print(a_forward_y.size(), forward_y.size())
                a_forward_y = (a_forward_y - forward_y[:, i:i + 1])**2
                # print(a_forward_y.size())
                if args.bidirectional:
                    a_backward_y = rnn.backward_rnn(backward_Y[:, :, i:i + 1])
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

        auc_pre = metrics.auc(fp_list, tp_list)
        aucs.append(auc_pre)

    print(sum(aucs) / len(aucs), aucs)
