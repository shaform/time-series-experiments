import torch
from sklearn import metrics
import numpy as np
import sklearn

from ..data_loader import LabeledDataSet
from ..scripts.train_gan import GAN

data_sets = [
'parsed_data/labeled/fishkiller.npy',
'parsed_data/labeled/beedance-1.npy',
'parsed_data/labeled/beedance-2.npy',
'parsed_data/labeled/beedance-3.npy',
'parsed_data/labeled/beedance-4.npy',
'parsed_data/labeled/beedance-5.npy',
'parsed_data/labeled/beedance-6.npy',
'parsed_data/labeled/yahoo-7.npy',
'parsed_data/labeled/yahoo-8.npy',
'parsed_data/labeled/yahoo-16.npy',
'parsed_data/labeled/yahoo-22.npy',
'parsed_data/labeled/yahoo-27.npy',
'parsed_data/labeled/yahoo-33.npy',
'parsed_data/labeled/yahoo-37.npy',
'parsed_data/labeled/yahoo-42.npy',
'parsed_data/labeled/yahoo-45.npy',
'parsed_data/labeled/yahoo-46.npy',
'parsed_data/labeled/yahoo-50.npy',
'parsed_data/labeled/yahoo-51.npy',
'parsed_data/labeled/yahoo-54.npy',
'parsed_data/labeled/yahoo-55.npy',
'parsed_data/labeled/yahoo-56.npy',
'parsed_data/labeled/hasc.npy'
]

for data_set in data_sets:
    print('=====', data_set, '====')
    d = LabeledDataSet(data_set, device=torch.device('cuda'), shuffle=True)
    gan = torch.load('models/gan')
    gan.disc.eval()
    Y_pred = []
    L_true = []
    for Y, L in d.tst_set:
        Y_pred_batch = None
        for i in range(Y.shape[2]):
            a_Y = gan.disc(Y[:, :, i])
            if Y_pred_batch is None:
                Y_pred_batch = a_Y
            else:
                Y_pred_batch += a_Y
        Y_pred.append(-Y_pred_batch.data.cpu().numpy())
        L_true.append(L)
    Y_pred = np.concatenate(Y_pred, axis=0)
    L_true = np.concatenate(L_true, axis=0)
    fp_list, tp_list, thresholds = metrics.roc_curve(L_true, Y_pred)

    auc_pre = metrics.auc(fp_list, tp_list)

    # finetune
    gan.disc.train()
    gan.train(d.trn_set.unlabelled(), lr=0.0001, beta1=0.5, beta2=0.999, num_epochs=1, save_path=None, finetune=True)


    gan.disc.eval()
    Y_pred = []
    L_true = []
    for Y, L in d.tst_set:
        Y_pred_batch = None
        for i in range(Y.shape[2]):
            a_Y = gan.disc(Y[:, :, i])
            if Y_pred_batch is None:
                Y_pred_batch = a_Y
            else:
                Y_pred_batch += a_Y
        Y_pred.append(-Y_pred_batch.data.cpu().numpy())
        L_true.append(L)
    Y_pred = np.concatenate(Y_pred, axis=0)
    L_true = np.concatenate(L_true, axis=0)
    fp_list, tp_list, thresholds = metrics.roc_curve(L_true, Y_pred)

    auc = metrics.auc(fp_list, tp_list)
    print('aut pre=', auc_pre, 'auc=', auc)
