import torch.nn as nn
import torch.nn.functional as F
import torch

# Simple GRU baseline
class BasicRNN(nn.Module):
    def __init__(self, latent_size, input_size, output_size, rnn_type='gru'):
        super().__init__()

        self.output_size = output_size
        self.input_size = input_size
        self.latent_size = latent_size

        if rnn_type == 'gru':
            self.rnn_layer = nn.GRU(self.input_size, self.latent_size)
        else:
            self.rnn_layer = nn.LSTM(self.input_size, self.latent_size)
        self.fc_layer = nn.Sequential(
            nn.Linear(self.latent_size, self.output_size),
        )

    # X_p:      batch_size x seq_len x var_dim
    # X_p_enc:  batch_size x seq_len x RNN_hid_dim
    # h_t:      1 x batch_size x RNN_hid_dim
    # y_t:      batch_size x var_dim
    def forward(self, inputs):
        outputs, hidden = self.rnn_layer(inputs)
        # outputs: [seq_len, batch, num_directions * hidden_size]
        outputs = outputs[-1]

        outputs = self.fc_layer(outputs)
        # [batch, output_size]

        return outputs

class BasicGenerator(nn.Module):
    def __init__(self, latent_size, input_size, output_size):
        super().__init__()

        def block(num_inputs, num_outputs):
            return [nn.Linear(num_inputs, num_outputs), nn.ReLU()]

        self.model = nn.Sequential(*block(latent_size + input_size, 128),
                                   *block(128, 256), *block(256, 128),
                                   nn.Linear(128, output_size))

    def forward(self, z, c):
        inputs = torch.cat([z, c], dim=-1)
        return self.model(inputs)


class BasicDiscriminator(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, 128), nn.LeakyReLU(0.2), nn.Linear(128, 256),
            nn.LeakyReLU(0.2), nn.Linear(256, 128), nn.LeakyReLU(0.2),
            nn.Linear(128, 1))

    def forward(self, inputs):
        return self.model(inputs)
