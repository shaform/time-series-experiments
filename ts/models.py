import torch.nn as nn
import torch.nn.functional as F
import torch


class LSTNet(nn.Module):
    def __init__(self,
                 rnn_hidden_size,
                 cnn_hidden_size,
                 cnn_kernel_size,
                 rnn_skip_hidden_size,
                 skip_size,
                 window_size,
                 highway_size,
                 output_size,
                 dropout_rate,
                 output_func='linear'):
        super().__init__()
        self.window_size = window_size
        self.output_size = output_size
        self.rnn_hidden_size = rnn_hidden_size
        self.cnn_hidden_size = cnn_hidden_size
        self.rnn_skip_hidden = rnn_skip_hidden_size
        self.cnn_kernel_size = cnn_kernel_size
        self.skip_size = skip_size
        self.highway_size = highway_size

        self.conv1 = nn.Conv2d(
            1,
            self.cnn_hidden_size,
            kernel_size=(self.cnn_kernel_size, self.output_size))
        self.gru1 = nn.GRU(self.cnn_hidden_size, self.rnn_hidden_size)
        self.dropout = nn.Dropout(p=dropout_rate)

        # RNN-skip layer
        if self.skip_size > 0:
            self.pt = (
                self.window_size - self.cnn_kernel_size) // self.skip_size
            self.gru_skip = nn.GRU(self.cnn_hidden_size, self.rnn_skip_hidden)
            self.linear1 = nn.Linear(
                self.rnn_hidden_size + self.skip_size * self.rnn_skip_hidden,
                self.output_size)
        else:
            self.linear1 = nn.Linear(self.rnn_hidden_size, self.output_size)

        # Highway layer
        if self.highway_size > 0:
            self.highway = nn.Linear(self.highway_size, 1)

        self.output_func = None
        if output_func == 'sigmoid':
            self.output_func = torch.sigmoid
        elif output_func == 'tanh':
            self.output_func = torch.tanh

    def forward(self, inputs):
        # inputs: [seq, batch, output_size]
        batch_size = inputs.size(1)

        # CNN
        outputs = inputs.transpose(0, 1).contiguous().view(
            -1, 1, self.window_size, self.output_size)
        outputs = F.relu(self.conv1(outputs))
        outputs = self.dropout(outputs)
        # [batch, hidden, seq, 1] -> [batch, hidden, seq]
        outputs = torch.squeeze(outputs, 3)

        # RNN
        # -> [seq, batch, hidden]
        r = outputs.permute(2, 0, 1).contiguous()
        _, r = self.gru1(r)
        # -> [batch, hidden_size]
        r = self.dropout(torch.squeeze(r, 0))

        # RNN-GRUskip
        if (self.skip_size > 0):
            s = outputs[:, :, -self.pt * self.skip_size:].contiguous()
            s = s.view(batch_size, self.cnn_hidden_size, self.pt,
                       self.skip_size)
            s = s.permute(2, 0, 3, 1).contiguous()
            s = s.view(self.pt, batch_size * self.skip_size,
                       self.cnn_hidden_size)
            _, s = self.gru_skip(s)
            s = s.view(batch_size, self.skip_size * self.rnn_skip_hidden)
            s = self.dropout(s)
            r = torch.cat((r, s), 1)

        res = self.linear1(r)

        # highway
        if self.highway_size > 0:
            z = inputs[-self.highway_size:, :, :]
            z = z.permute(1, 2, 0).contiguous().view(-1, self.highway_size)
            z = self.highway(z)
            z = z.view(-1, self.output_size)
            res = res + z

        if self.output_func is not None:
            res = self.output_func(res)

        return res


# Simple GRU baseline
class BasicRNN(nn.Module):
    def __init__(self,
                 latent_size,
                 input_size,
                 output_size,
                 predict_x=False,
                 rnn_type='gru'):
        super().__init__()

        self.output_size = output_size
        self.input_size = input_size
        self.latent_size = latent_size
        self.predict_x = predict_x

        if rnn_type == 'gru':
            self.rnn_layer = nn.GRU(self.input_size, self.latent_size)
        else:
            self.rnn_layer = nn.LSTM(self.input_size, self.latent_size)
        self.fc_layer = nn.Sequential(
            nn.Linear(self.latent_size, self.output_size), )

        if predict_x:
            self.pred_layer = nn.Sequential(
                nn.Linear(self.latent_size, self.output_size), )
            self.attn_layer = nn.Sequential(
                nn.Linear(self.latent_size + self.output_size * 2, 1), )

    def forward(self, inputs):
        # if self.predict_x:
        #     attn = None
        #     output_pred = None
        #     outputs = []
        #     outputs_pred = []
        #     for i, inputs_t in enumerate(inputs.chunk(input.size(0), dim=0)):
        #         if attn is not None:
        #             inputs_t = attn * input_t + (1 - attn) * output_pred
        #             o_t, h_t = self.rnn_layer(inputs_t)
        #             output = self.fc_layer(o_t[-1])
        #             output_pred = self.pred_layer(o_t[-1])

        #             output_attn = torch.cat([output, output_pred, h_t], dim=1)
        #             attn = F.sigmoid(self.attn_layer(output_attn))
        #         outputs += [output]
        #         outputs_pred += [output_pred]
        #     outputs = torch.stack(outputs, 1)
        #     outputs_pred = torch.stack(outputs, 1)
        #     return outputs, outputs_pred
        # else:
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
