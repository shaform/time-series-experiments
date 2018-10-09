import random
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

    def yield_batches(self, shuffle=True):
        indices = np.arange(self.window_size,
                            self.num_steps - self.window_size + 1)
        indices = np.concatenate([indices] * self.num_variables)
        dim_indices = np.concatenate(
            [[i] * (self.num_steps - self.window_size * 2 + 1)
             for i in range(self.num_variables)])

        if shuffle:
            perm = np.random.permutation(indices)
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
    def __init__(self, data_paths, window_size=25, batch_size=10, device=None):
        self.data_paths = data_paths
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
