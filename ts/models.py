import torch.nn as nn
import torch.nn.functional as F
import torch


class BasicGenerator(nn.Module):
    def __init__(self, latent_size, input_size, output_size):
        super().__init__()

        def block(num_inputs, num_outputs):
            return [nn.Linear(num_inputs, num_outputs), nn.ReLU()]

        self.model = nn.Sequential(*block(latent_size + input_size, 128),
                                   *block(128, 256), nn.Linear(
                                       256, output_size))

    def forward(self, z, c):
        inputs = torch.cat([z, c], dim=-1)
        return self.model(inputs)


class BasicDiscriminator(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, 128), nn.LeakyReLU(0.2), nn.Linear(128, 256),
            nn.LeakyReLU(0.2), nn.Linear(256, 1))

    def forward(self, inputs):
        return self.model(inputs)
