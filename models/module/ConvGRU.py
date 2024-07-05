
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn import init


class convGRU(nn.Module):
    def __init__(self, in_channels1, in_channels2, inner):
        super(convGRU, self).__init__()

        self.conv_x_r = nn.Conv2d(in_channels1, inner, 1)
        self.conv_h_r = nn.Conv2d(in_channels2, inner, 1)

        self.conv = nn.Conv2d(in_channels1, inner, 1)
        self.conv_u = nn.Conv2d(in_channels2, inner, 1)

    def forward(self, x, h_t_1):
        r_t = torch.sigmoid(self.conv_x_r(x) + self.conv_h_r(h_t_1))
        h_hat_t = self.conv(x) + self.conv_u(h_t_1)
        h_t = torch.mul(r_t, torch.tanh(h_t_1)) + torch.mul((1 - r_t), torch.tanh(h_hat_t))
        return h_t
