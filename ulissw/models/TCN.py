import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

from .TCN_base import Chomp1d


class TCN_block(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilations, drop_prob=0.2):
        super().__init__()
        n_convs = len(dilations)
        conv_list = []

        for i in range(n_convs):
            if i == 0:
                in_channels = n_inputs
            else:
                in_channels = n_outputs

            padding = (kernel_size-1) * dilations[i]
            conv = weight_norm(nn.Conv1d(in_channels, n_outputs, kernel_size,
                                         stride=stride, padding=padding, dilation=dilations[i]))
            conv.weight.data.normal_(0, 0.01)
            chomp = Chomp1d(padding)
            relu = nn.ReLU()
            dropout = nn.Dropout(drop_prob)

            conv_list += [conv, chomp, relu, dropout]


        self.net = nn.Sequential(*conv_list)
        self.resize = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        if self.resize is not None:
            self.resize.weight.data.normal_(0, 0.01)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        out = self.net(x)
        res = x if self.resize is None else self.resize(x)
        return self.relu(out + res)


class TCN(nn.Module):
    model_type = 'tcn'
    def __init__(self, input_size, num_inputs=1, output_size=16, 
                 num_blocks=1, block_dilations=[1, 3, 6, 12, 24],
                 n_channels=128, kernel_size=6, dropout=0.2):
        super().__init__()
        if not isinstance(n_channels, list):
            n_channels = [n_channels]*num_blocks
        if not isinstance(kernel_size, list):
            kernel_size = [kernel_size]*num_blocks

        blocks = []
        for i in range(num_blocks):
            in_channels = num_inputs if i == 0 else n_channels[i-1]
            out_channels = n_channels[i]

            blocks += [TCN_block(in_channels, out_channels, kernel_size[i], 
                                 stride=1, dilations=block_dilations,
                                 drop_prob=dropout)]

        self.network = nn.Sequential(*blocks)
        self.fc = nn.Linear(input_size*n_channels[-1], output_size)

    def forward(self, x):
        x = self.network(x)
        return self.fc(x.flatten(1))