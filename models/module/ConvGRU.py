import torch
import torch.nn as nn

# myConvGRU
# class convGRU(nn.Module):
#     def __init__(self, in_channels1, in_channels2, inner):
#         super(convGRU, self).__init__()
#
#         self.conv_x_z = nn.Conv2d(in_channels1, inner, 1)
#         self.conv_h_z = nn.Conv2d(in_channels2, inner, 1)
#         self.conv_x_r = nn.Conv2d(in_channels1, inner, 1)
#         self.conv_h_r = nn.Conv2d(in_channels2, inner, 1)
#         self.conv = nn.Conv2d(in_channels1, inner,1)
#         self.conv_u = nn.Conv2d(in_channels2, inner,1)
#
#
#
#     def forward(self, x, h_t_1):
#
#         z_t = torch.sigmoid(self.conv_x_z(x) + self.conv_h_z(h_t_1))
#         r_t = torch.sigmoid(self.conv_x_r(x) + self.conv_h_r(h_t_1))
#         h_hat_t = torch.tanh(self.conv(x) + self.conv_u(torch.mul(r_t, h_t_1)))
#         h_t = torch.mul((1 - z_t), h_t_1) + torch.mul(z_t, h_hat_t)
#         return h_t


# https://github.com/jacobkimmel/pytorch_convgru/blob/master/convgru.py
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn import init
# class convGRU(nn.Module):
#     """
#     Generate a convolutional GRU cell
#     """
#     def __init__(self, input_size, input_size1, hidden_size, kernel_size=1):
#         super(convGRU, self).__init__()
#         # padding = kernel_size // 2
#         padding = 0
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.reset_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)
#         self.update_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)
#         self.out_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)
#
#         init.orthogonal(self.reset_gate.weight)
#         init.orthogonal(self.update_gate.weight)
#         init.orthogonal(self.out_gate.weight)
#         init.constant(self.reset_gate.bias, 0.)
#         init.constant(self.update_gate.bias, 0.)
#         init.constant(self.out_gate.bias, 0.)
#
#
#     def forward(self, input_, prev_state):
#
#         # get batch and spatial sizes
#         batch_size = input_.data.size()[0]
#         spatial_size = input_.data.size()[2:]
#
#         # generate empty prev_state, if None is provided
#         if prev_state is None:
#             state_size = [batch_size, self.hidden_size] + list(spatial_size)
#             if torch.cuda.is_available():
#                 prev_state = Variable(torch.zeros(state_size)).cuda()
#             else:
#                 prev_state = Variable(torch.zeros(state_size))
#
#         # data size is [batch, channel, height, width]
#         stacked_inputs = torch.cat([input_, prev_state], dim=1)
#         update = F.sigmoid(self.update_gate(stacked_inputs))
#         reset = F.sigmoid(self.reset_gate(stacked_inputs))
#         out_inputs = F.tanh(self.out_gate(torch.cat([input_, prev_state * reset], dim=1)))
#         new_state = prev_state * (1 - update) + out_inputs * update
#
#         return new_state





# class convGRU(nn.Module):
#     def __init__(self, in_channels1, in_channels2, inner):
#         super(convGRU, self).__init__()
#
#         self.conv_x_r = nn.Conv2d(in_channels1, inner, 1)
#         self.conv_h_r = nn.Conv2d(in_channels2, inner, 1)
#
#         self.conv = nn.Conv2d(in_channels1, inner, 1)
#         self.conv_u = nn.Conv2d(in_channels2, inner, 1)
#
#     def forward(self, x, h_t_1):
#         r_t = torch.sigmoid(self.conv_x_r(x) + self.conv_h_r(h_t_1))
#         h_hat_t = self.conv(x) + self.conv_u(h_t_1)
#         # 一直用的这个版本，似乎不是最优
#         h_t = torch.mul(r_t, torch.tanh(h_t_1)) + torch.mul((1 - r_t), torch.tanh(h_hat_t))
#         #这个好像更好，不对，在2013数据集上很差
#         # h_t = torch.mul((1 - r_t), torch.tanh(h_t_1)) + torch.mul(r_t, torch.tanh(h_hat_t))
#         # 这个一般般
#         # h_t = torch.mul((1 - r_t), h_t_1) + torch.mul(r_t, torch.tanh(h_hat_t))
#         return h_t




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
