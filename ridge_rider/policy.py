import torch
import torch.nn as nn
from torch.nn import LSTM, Linear


class Policy(nn.Module):
    def __init__(self, x_dim, y_dim):
        super(Policy, self).__init__()

        self.h_dim = y_dim
        num_layers = 2

        self.rnn = LSTM(x_dim, self.h_dim, num_layers=num_layers, bias=False)

    def get_m(self):
        return (torch.rand(2, 1, self.h_dim),
                torch.rand(2, 1, self.h_dim))

    def forward(self, x, m):
        (h, m) = self.rnn(x.view(1, 1, -1), m)
        y = h[-1]
        return y, m
