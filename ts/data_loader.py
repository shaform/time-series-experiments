import random
import math

import numpy as np
import torch
from torch.autograd import Variable


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


class LabeledDataLoader(object):
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
        return self.end - self.start

    def __iter__(self):
        return self.yield_batches(shuffle=self.shuffle)

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

    def change_point(self, labels):
        labels = (labels != np.roll(labels, 1)).astype(np.float32)
        labels[0] = 0.
        return labels

    def load_data(self, trn_ratio=0.6, val_ratio=0.8):
        data_dict = np.load(self.data_path).item()
        self.data = data_dict['Y']
        # (N, ) to (N, 1)
        self.data = self.data.reshape((self.data.shape[0], -1))

        mean = np.mean(self.data, axis=0)
        std = np.std(self.data, axis=0)

        self.data = (self.data - mean) / std

        self.labels = self.change_point(data_dict['L'])
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
