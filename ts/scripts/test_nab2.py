import argparse
import os

import torch
from sklearn import metrics
import numpy as np
import sklearn
import tqdm

from ..data_loader import LabeledDataSet
from ..data_loader import UnlabeledDataLoader
from ..scripts.train_rnn import RNN
parser = argparse.ArgumentParser()
parser.add_argument('--model-path', required=True)
parser.add_argument('--model-type', default='gru')
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--window-size', type=int, default=25)
args = parser.parse_args()

device = torch.device('cuda') if args.cuda else torch.device('cpu')

data_sets = [
    [
        'nab_data/artificialNoAnomaly/art_daily_no_noise.csv.npy',
        'nab_data/artificialNoAnomaly/art_daily_perfect_square_wave.csv.npy',
        'nab_data/artificialNoAnomaly/art_daily_small_noise.csv.npy',
        'nab_data/artificialNoAnomaly/art_flatline.csv.npy',
        'nab_data/artificialNoAnomaly/art_noisy.csv.npy',
        'nab_data/realAWSCloudwatch/ec2_cpu_utilization_c6585a.csv.npy',
        'nab_data/artificialWithAnomaly/art_daily_flatmiddle.csv.npy',
        'nab_data/artificialWithAnomaly/art_daily_jumpsdown.csv.npy',
        'nab_data/artificialWithAnomaly/art_daily_jumpsup.csv.npy',
        'nab_data/artificialWithAnomaly/art_daily_nojump.csv.npy',
        'nab_data/artificialWithAnomaly/art_load_balancer_spikes.csv.npy',
        'nab_data/artificialWithAnomaly/art_increase_spike_density.csv.npy',
        'nab_data/realAWSCloudwatch/ec2_cpu_utilization_53ea38.csv.npy',
        'nab_data/realAWSCloudwatch/ec2_cpu_utilization_5f5533.csv.npy',
        'nab_data/realAWSCloudwatch/ec2_cpu_utilization_77c1ca.csv.npy',
        'nab_data/realAWSCloudwatch/ec2_cpu_utilization_825cc2.csv.npy',
        'nab_data/realAWSCloudwatch/ec2_cpu_utilization_ac20cd.csv.npy',
        'nab_data/realAWSCloudwatch/ec2_cpu_utilization_fe7f93.csv.npy',
        'nab_data/realAWSCloudwatch/ec2_disk_write_bytes_1ef3de.csv.npy',
        'nab_data/realAWSCloudwatch/ec2_disk_write_bytes_c0d644.csv.npy',
        'nab_data/realAWSCloudwatch/ec2_network_in_257a54.csv.npy',
        'nab_data/realAWSCloudwatch/ec2_network_in_5abac7.csv.npy',
        'nab_data/realAWSCloudwatch/elb_request_count_8c0756.csv.npy',
        'nab_data/realAWSCloudwatch/grok_asg_anomaly.csv.npy',
        'nab_data/realAWSCloudwatch/iio_us-east-1_i-a2eb1cd9_NetworkIn.csv.npy',
        'nab_data/realAWSCloudwatch/rds_cpu_utilization_cc0c53.csv.npy',
        'nab_data/realAWSCloudwatch/rds_cpu_utilization_e47b3b.csv.npy',
        'nab_data/realAWSCloudwatch/ec2_cpu_utilization_24ae8d.csv.npy',
        'nab_data/realAdExchange/exchange-2_cpm_results.csv.npy',
        'nab_data/realAdExchange/exchange-2_cpc_results.csv.npy',
        'nab_data/realAdExchange/exchange-3_cpc_results.csv.npy',
        'nab_data/realAdExchange/exchange-3_cpm_results.csv.npy',
        'nab_data/realAdExchange/exchange-4_cpc_results.csv.npy',
        'nab_data/realAdExchange/exchange-4_cpm_results.csv.npy',
        'nab_data/realKnownCause/ambient_temperature_system_failure.csv.npy',
        'nab_data/realKnownCause/cpu_utilization_asg_misconfiguration.csv.npy',
        'nab_data/realKnownCause/ec2_request_latency_system_failure.csv.npy',
        'nab_data/realKnownCause/machine_temperature_system_failure.csv.npy',
        'nab_data/realKnownCause/nyc_taxi.csv.npy',
        'nab_data/realKnownCause/rogue_agent_key_hold.csv.npy',
        'nab_data/realKnownCause/rogue_agent_key_updown.csv.npy',
        'nab_data/realTraffic/TravelTime_387.csv.npy',
        'nab_data/realTraffic/TravelTime_451.csv.npy',
        'nab_data/realTraffic/occupancy_6005.csv.npy',
        'nab_data/realTraffic/occupancy_t4013.csv.npy',
        'nab_data/realTraffic/speed_6005.csv.npy',
        'nab_data/realTraffic/speed_7578.csv.npy',
        'nab_data/realTraffic/speed_t4013.csv.npy',
        'nab_data/realTweets/Twitter_volume_KO.csv.npy',
        'nab_data/realTweets/Twitter_volume_PFE.csv.npy',
        'nab_data/realTweets/Twitter_volume_UPS.csv.npy',
        'nab_data/realTweets/Twitter_volume_AAPL.csv.npy',
        'nab_data/realTweets/Twitter_volume_AMZN.csv.npy',
        'nab_data/realTweets/Twitter_volume_CRM.csv.npy',
        'nab_data/realTweets/Twitter_volume_CVS.csv.npy',
        'nab_data/realTweets/Twitter_volume_FB.csv.npy',
        'nab_data/realTweets/Twitter_volume_GOOG.csv.npy',
        'nab_data/realTweets/Twitter_volume_IBM.csv.npy',
    ],
]

outfile = open('testk2', 'w')
for data_set_list in data_sets:
    print('=====', data_set_list, '====')
    aucs = []
    for data_set in data_set_list:
        d = LabeledDataSet(
            data_set,
            batch_size=50,
            device=device,
            shuffle=False,
            trn_ratio=0.0,
            val_ratio=0.0,
            normal='minmax')
        rnn = torch.load(args.model_path, map_location=device)
        rnn.forward_rnn.eval()
        Y_pred = []
        L_true = []
        for Y, L in tqdm.tqdm(d.tst_set):
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
                a_Y = a_forward_y
                if Y_pred_batch is None:
                    Y_pred_batch = a_Y
                else:
                    Y_pred_batch += a_Y
            Y_pred.append(Y_pred_batch.data.cpu().numpy())
            L_true.append(L)
        Y_pred = np.concatenate(Y_pred, axis=0).flatten()
        outfile.write(data_set + '\t' + ' '.join(str(x)
                                                 for x in Y_pred) + '\n')
        L_true = np.concatenate(L_true, axis=0).flatten()
        fp_list, tp_list, thresholds = metrics.roc_curve(L_true, Y_pred)

        auc_pre = metrics.auc(fp_list, tp_list)
        aucs.append(auc_pre)

    print(sum(aucs) / len(aucs), aucs)
