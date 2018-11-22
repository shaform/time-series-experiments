import random
import math

import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset


class MergeDataset(Dataset):
    def __init__(self, *datasets, sub_sample=None):
        self.datasets = datasets
        self.lens = [len(d) for d in datasets]
        self.sub_sample = sub_sample
        total = sum(self.lens)

        if sub_sample:
            self.indices = np.random.choice(total, int(total * sub_sample))
            self.indices.sort()

    def __getitem__(self, idx):
        if self.sub_sample:
            idx = self.indices[idx]

        for i, length in enumerate(self.lens):
            if idx < length:
                return self.datasets[i][idx]
            else:
                idx -= length

    def __len__(self):
        if self.sub_sample:
            return self.indices.shape[0]
        else:
            return sum(self.lens)


class WaveNetUDataSet(Dataset):
    def __init__(self,
                 data_path,
                 receptive_field=25,
                 horizon=None,
                 device=None):
        if horizon is not None and horizon <= receptive_field:
            horizon = receptive_field + 1

        self.data_path = data_path
        self.device = device
        self.receptive_field = receptive_field
        self.horizon = horizon
        self.load_data()

    def load_data(self):
        self.data = np.load(self.data_path)
        # (N, ) to (N, 1)
        self.data = self.data.reshape((self.data.shape[0], -1))

        # to [0, 1]
        mini = np.min(self.data, axis=0)
        ptp = np.ptp(self.data, axis=0)

        self.data = (self.data - mini) / ptp
        self.num_steps, self.num_variables = self.data.shape

    def with_horizon(self, horizon):
        return WaveNetUDataSet(
            data_path=self.data_path,
            receptive_field=self.receptive_field,
            horizon=horizon,
            device=self.device)

    def __getitem__(self, idx):
        # always zeros
        labels = np.array([0])

        if self.horizon is None:
            data = self.data[:self.end]
        else:
            start = idx * (self.horizon - self.receptive_field)
            end = min(self.num_steps, start + self.horizon)
            data = self.data[start:end]

        data = torch.tensor(data, dtype=torch.float).permute(1, 0).contiguous()
        labels = torch.tensor(labels, dtype=torch.long)

        return data, labels

    def __len__(self):
        # total all at once now
        if self.horizon is None:
            return 1
        else:
            return math.ceil((max(1, (self.num_steps - self.receptive_field)))
                             / (self.horizon - self.receptive_field))


class WaveNetDataSet(object):
    def __init__(self,
                 data_path,
                 receptive_field=25,
                 trn_ratio=0.6,
                 val_ratio=0.8,
                 device=None):
        self.data_path = data_path
        self.device = device
        self.receptive_field = receptive_field
        self.trn_ratio = trn_ratio
        self.val_ratio = val_ratio
        self.load_data(trn_ratio=trn_ratio, val_ratio=val_ratio)

    def load_data(self, trn_ratio=0.6, val_ratio=0.8):
        data_dict = np.load(self.data_path).item()
        self.data = data_dict['Y']
        # (N, ) to (N, 1)
        self.data = self.data.reshape((self.data.shape[0], -1))

        # to [0, 1]
        mini = np.min(self.data, axis=0)
        ptp = np.ptp(self.data, axis=0)

        self.data = (self.data - mini) / ptp

        self.labels = data_dict['L']
        self.labels = self.labels.reshape((-1, ))
        # to (N,)

        self.num_steps, self.num_variables = self.data.shape

        self.trn_end = int(np.ceil(self.num_steps * trn_ratio))
        self.val_end = int(np.ceil(self.num_steps * val_ratio))

        self.trn_set = LabeledWaveNetDataSet(
            self,
            0,
            self.trn_end,
            receptive_field=self.receptive_field,
            device=self.device)
        self.val_set = LabeledWaveNetDataSet(
            self,
            self.trn_end,
            self.val_end,
            receptive_field=self.receptive_field,
            device=self.device)
        self.tst_set = LabeledWaveNetDataSet(
            self,
            self.val_end,
            self.num_steps,
            receptive_field=self.receptive_field,
            device=self.device)


class LabeledWaveNetDataSet(Dataset):
    def __init__(self,
                 data_set,
                 start,
                 end,
                 receptive_field=25,
                 horizon=None,
                 device=None):
        if horizon is not None and horizon <= receptive_field:
            horizon = receptive_field + 1

        self.data_set = data_set
        self.start = start
        self.end = end
        self.receptive_field = receptive_field
        self.num_variables = self.data_set.data.shape[1]
        self.num_steps = end - start
        self.horizon = horizon
        self.device = device

        self.data_begin = max(0, self.start - self.receptive_field)

    def with_horizon(self, horizon):
        return LabeledWaveNetDataSet(
            data_set=self.data_set,
            start=self.start,
            end=self.end,
            receptive_field=self.receptive_field,
            horizon=horizon,
            device=self.device)

    def __len__(self):
        # total all at once now
        if self.horizon is None:
            return 1
        else:
            return math.ceil(
                (max(1, (self.end - self.receptive_field) - self.data_begin)) /
                (self.horizon - self.receptive_field))

    def __getitem__(self, idx):
        if self.horizon is None:
            data = self.data_set.data[self.data_begin:self.end]
            labels = self.data_set.labels[self.start:self.end]
        else:
            start = self.data_begin + idx * (
                self.horizon - self.receptive_field)
            end = min(self.end, start + self.horizon)
            data = self.data_set.data[start:end]
            if start == self.data_begin:
                labels = self.data_set.labels[self.start:end]
            else:
                labels = self.data_set.labels[start + self.receptive_field:end]

        data = torch.tensor(data, dtype=torch.float).permute(1, 0).contiguous()
        labels = torch.tensor(labels, dtype=torch.long)

        return data, labels


class ForcastDataLoader(object):
    def __init__(self,
                 data_set,
                 start,
                 end,
                 shuffle=True,
                 window_size=25,
                 batch_size=10,
                 device=None):
        self.data_set = data_set
        self.start = start
        self.end = end
        self.shuffle = shuffle

        self.window_size = window_size
        self.batch_size = batch_size
        self.device = device

    def __len__(self):
        return math.ceil((self.end - self.start) / self.batch_size)

    def __iter__(self):
        return self.yield_batches(shuffle=self.shuffle)

    def yield_batches(self, shuffle=True):
        indices = np.arange(max(self.window_size, self.start), self.end)
        if shuffle:
            indices = np.random.permutation(indices)

        for i in range(0, indices.shape[0], self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            real_batch_size = batch_indices.shape[0]
            batch_data = np.zeros((real_batch_size, self.window_size,
                                   self.data_set.num_variables))
            batch_labels = np.zeros((real_batch_size,
                                     self.data_set.num_variables))

            for j, index in enumerate(batch_indices):
                batch_data[j] = self.data_set.data[index -
                                                   self.window_size:index]
                batch_labels[j] = self.data_set.data[index]

            # create tensor on target device
            batch_data = Variable(
                torch.tensor(
                    batch_data, dtype=torch.float, device=self.device),
                requires_grad=False)
            batch_labels = Variable(
                torch.tensor(
                    batch_labels, dtype=torch.float, device=self.device),
                requires_grad=False)

            yield batch_data, batch_labels


class SingleUnlabeledDataLoader(object):
    def __init__(self, data_path, window_size=25, batch_size=10, device=None):
        self.data_path = data_path
        self.window_size = window_size
        self.batch_size = batch_size
        self.device = device
        self.load_data()

    def load_data(self):
        self.data = np.load(self.data_path)
        # (N, ) to (N, 1)
        self.data = self.data.reshape((self.data.shape[0], -1))

        mean = np.mean(self.data, axis=0)
        std = np.std(self.data, axis=0)

        self.data = (self.data - mean) / std

        self.num_steps, self.num_variables = self.data.shape

    def __len__(self):
        return math.ceil(
            (self.num_variables *
             (self.num_steps - self.window_size * 2 + 1)) / self.batch_size)

    def yield_batches(self, shuffle=True):
        indices = np.arange(self.window_size,
                            self.num_steps - self.window_size + 1)
        indices = np.concatenate([indices] * self.num_variables)
        dim_indices = np.concatenate(
            [[i] * (self.num_steps - self.window_size * 2 + 1)
             for i in range(self.num_variables)])

        if shuffle:
            perm = np.random.permutation(np.arange(indices.shape[0]))
            indices = indices[perm]
            dim_indices = dim_indices[perm]

        for i in range(0, indices.shape[0], self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            batch_dim_indices = dim_indices[i:i + self.batch_size]
            real_batch_size = batch_indices.shape[0]
            batch_data = np.zeros((real_batch_size, self.window_size * 2))
            for j, index in enumerate(batch_indices):
                batch_data[j] = self.data[index - self.window_size:index + self
                                          .window_size, batch_dim_indices[j]]

            # create tensor on target device
            batch_data = Variable(
                torch.tensor(
                    batch_data, dtype=torch.float, device=self.device),
                requires_grad=False)

            yield batch_data


class UnlabeledDataLoader(object):
    def __init__(self,
                 data_paths,
                 shuffle=True,
                 window_size=25,
                 batch_size=10,
                 device=None):
        self.data_paths = data_paths
        self.shuffle = shuffle
        if isinstance(self.data_paths, str):
            self.data_paths = [self.data_paths]

        self.window_size = window_size
        self.batch_size = batch_size
        self.device = device
        self.load_data()

    def load_data(self):
        self.data = [
            SingleUnlabeledDataLoader(data_path, self.window_size,
                                      self.batch_size, self.device)
            for data_path in self.data_paths
        ]

    def __len__(self):
        return sum([len(data) for data in self.data])

    def __iter__(self):
        return self.yield_batches(shuffle=self.shuffle)

    def yield_batches(self, shuffle=True):
        yielders = [data.yield_batches(shuffle=shuffle) for data in self.data]
        if not shuffle:
            for yielder in yielders:
                yield from yielder

        else:
            while len(yielders) > 0:
                random.shuffle(yielders)
                try:
                    batch_data = next(yielders[-1])
                    yield batch_data
                except StopIteration:
                    yielders.pop()


class ForcastDataSet(object):
    def __init__(self,
                 data_path,
                 shuffle=True,
                 window_size=25,
                 batch_size=10,
                 trn_ratio=0.6,
                 val_ratio=0.8,
                 device=None):
        self.data_path = data_path
        self.shuffle = shuffle

        self.window_size = window_size
        self.batch_size = batch_size
        self.device = device
        self.load_data(trn_ratio=trn_ratio, val_ratio=val_ratio)

    def load_data(self, trn_ratio=0.6, val_ratio=0.8):
        self.data = np.load(self.data_path)
        # (N, ) to (N, 1)
        self.data = self.data.reshape((self.data.shape[0], -1))

        mean = np.mean(self.data, axis=0)
        std = np.std(self.data, axis=0)

        self.data = (self.data - mean) / std

        self.num_steps, self.num_variables = self.data.shape
        self.trn_end = int(np.ceil(self.num_steps * trn_ratio))
        self.val_end = int(np.ceil(self.num_steps * val_ratio))

        self.trn_set = ForcastDataLoader(
            self,
            self.window_size,
            self.trn_end,
            shuffle=self.shuffle,
            window_size=self.window_size,
            batch_size=self.batch_size,
            device=self.device)
        self.val_set = ForcastDataLoader(
            self,
            self.trn_end,
            self.val_end,
            shuffle=False,
            window_size=self.window_size,
            batch_size=self.batch_size,
            device=self.device)
        self.tst_set = ForcastDataLoader(
            self,
            self.val_end,
            self.num_steps,
            shuffle=False,
            window_size=self.window_size,
            batch_size=self.batch_size,
            device=self.device)


class LabeledDataLoader(object):
    def __init__(self,
                 data_set,
                 start,
                 end,
                 unlabelled=False,
                 shuffle=True,
                 window_size=25,
                 batch_size=10,
                 device=None,
                 full=False):
        self.data_set = data_set
        self.start = start
        self.end = end
        self.shuffle = shuffle
        self.window_size = window_size
        self.batch_size = batch_size
        self.is_unlabelled = unlabelled
        self.full = full
        self.num_variables = self.data_set.data.shape[1]
        self.device = device

    def unlabelled(self, full=False):
        return LabeledDataLoader(
            self.data_set,
            self.start,
            self.end,
            True,
            self.shuffle,
            self.window_size,
            self.batch_size,
            self.device,
            full=full)

    def __len__(self):
        if self.is_unlabelled and not self.full:
            return math.ceil((self.end - self.start) *
                             self.data_set.data.shape[1] / self.batch_size)
        else:
            return math.ceil((self.end - self.start) / self.batch_size)

    def __iter__(self):
        if self.is_unlabelled:
            return self.yield_unlabelled_batches(
                shuffle=self.shuffle, full=self.full)
        else:
            return self.yield_batches(shuffle=self.shuffle)

    def yield_unlabelled_batches(self, shuffle=True, full=False):
        for batch_data, _ in self.yield_batches(shuffle=shuffle):
            if full:
                yield batch_data
            else:
                for i in range(batch_data.shape[2]):
                    yield batch_data[:, :, i]

    def yield_batches(self, shuffle=True):
        indices = np.arange(self.start, self.end)

        if shuffle:
            indices = np.random.permutation(indices)

        for i in range(0, indices.shape[0], self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            batch_data = []
            batch_labels = []
            for j in batch_indices:
                batch_data.append(self.data_set.data[j - self.window_size:j +
                                                     self.window_size])
                batch_labels.append(self.data_set.labels[j])

            batch_data = np.stack(batch_data)
            batch_labels = np.concatenate(batch_labels).reshape(-1, 1)

            # create tensor on target device
            batch_data = Variable(
                torch.tensor(
                    batch_data, dtype=torch.float, device=self.device),
                requires_grad=False)
            batch_labels = Variable(
                torch.tensor(
                    batch_labels, dtype=torch.float, device=self.device),
                requires_grad=False)

            yield batch_data, batch_labels


class LabeledDataSet(object):
    def __init__(self,
                 data_path,
                 shuffle=True,
                 window_size=25,
                 batch_size=10,
                 trn_ratio=0.6,
                 val_ratio=0.8,
                 device=None):
        self.data_path = data_path
        self.shuffle = shuffle

        self.window_size = window_size
        self.batch_size = batch_size
        self.device = device
        self.load_data(trn_ratio=trn_ratio, val_ratio=val_ratio)

    def load_data(self, trn_ratio=0.6, val_ratio=0.8):
        data_dict = np.load(self.data_path).item()
        self.data = data_dict['Y']
        # (N, ) to (N, 1)
        self.data = self.data.reshape((self.data.shape[0], -1))

        mean = np.mean(self.data, axis=0)
        std = np.std(self.data, axis=0)

        self.data = (self.data - mean) / std

        self.labels = data_dict['L']
        self.labels = self.labels.reshape((self.labels.shape[0], -1))

        self.num_steps, self.num_variables = self.data.shape

        self.trn_end = int(np.ceil(self.num_steps * trn_ratio))
        self.val_end = int(np.ceil(self.num_steps * val_ratio))

        # XXX: dirty trick to augment the last buffer, copied from previous work
        self.data = np.concatenate([self.data, self.data[-self.window_size:]])

        self.trn_set = LabeledDataLoader(
            self,
            self.window_size,
            self.trn_end,
            shuffle=self.shuffle,
            window_size=self.window_size,
            batch_size=self.batch_size,
            device=self.device)
        self.val_set = LabeledDataLoader(
            self,
            self.trn_end,
            self.val_end,
            shuffle=False,
            window_size=self.window_size,
            batch_size=self.batch_size,
            device=self.device)
        self.tst_set = LabeledDataLoader(
            self,
            self.val_end,
            self.num_steps,
            shuffle=False,
            window_size=self.window_size,
            batch_size=self.batch_size,
            device=self.device)
