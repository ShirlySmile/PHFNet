import torch.nn as nn
import torch.nn.functional as F

import torch
from models.module.function import adaptive_instance_normalization, calc_mean_std
from models.module.ConvGRU import convGRU



def conv3x3_bn_relu(in_channel, out_channel):
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, 3, 1, 1),
        nn.InstanceNorm2d(out_channel, affine=True),
        nn.ReLU(inplace=True),
    )


def conv1x1_bn_relu(in_channel, out_channel):
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, 1),
        nn.InstanceNorm2d(out_channel, affine=True),
        nn.ReLU(inplace=True),
    )


def downsample2x(in_channel, out_channel):
    return nn.Sequential(
        nn.MaxPool2d(2),
        nn.Conv2d(in_channel, out_channel, 1),
        nn.InstanceNorm2d(out_channel, affine=True),
        nn.ReLU(inplace=True),
    )


class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction_ratio):
        super(SEBlock, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.seq = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        v = self.gap(x)
        score = self.seq(v.view(v.size(0), v.size(1)))
        y = x * score.view(score.size(0), score.size(1), 1, 1)
        return y



def repeat_block(block_channel, r=4):
    layers = [
        nn.Sequential(
            SEBlock(block_channel, r),
        )]
    return nn.Sequential(*layers)


class NormInteraction(nn.Module):
    def __init__(self, channel=128):
        super(NormInteraction, self).__init__()

        self.conv1x1 = nn.Sequential(nn.Conv2d(channel + channel, channel, 1, bias=False),
                                     nn.BatchNorm2d(channel),
                                     nn.ReLU(inplace=False))

    def forward(self, branch1, branch2):
        branch1_norm = adaptive_instance_normalization(branch1)
        branch2_norm = adaptive_instance_normalization(branch2)

        C = torch.cat([branch1, branch2], 1)
        FusionStyle = self.conv1x1(C)
        style_mean, style_std = calc_mean_std(FusionStyle)
        size = branch2_norm.size()
        branch1_align = branch1_norm * style_std.expand(size) + style_mean.expand(size)
        branch2_align = branch2_norm * style_std.expand(size) + style_mean.expand(size)

        align_fusion = branch1_align + branch2_align

        return align_fusion




class PHFNet(nn.Module):
    def __init__(self, in_channels1, in_channels2, num_classes):
        super(PHFNet, self).__init__()

        block1_channels = 32
        block2_channels = 64
        block3_channels = 64
        block4_channels = 96

        self.feature_ops1 = nn.ModuleList([

            conv1x1_bn_relu(in_channels1, block1_channels),
            conv3x3_bn_relu(block1_channels, block1_channels),
            conv3x3_bn_relu(block1_channels, block1_channels),
            repeat_block(block1_channels),
            nn.Identity(),

            downsample2x(block1_channels, block2_channels),
            repeat_block(block2_channels),
            nn.Identity(),

            downsample2x(block2_channels, block3_channels),
            repeat_block(block3_channels),
            nn.Identity(),

            downsample2x(block3_channels, block4_channels),
            repeat_block(block4_channels),
            nn.Identity(),
        ])

        self.feature_ops2 = nn.ModuleList([

            conv1x1_bn_relu(in_channels2, block1_channels),
            conv3x3_bn_relu(block1_channels, block1_channels),
            conv3x3_bn_relu(block1_channels, block1_channels),
            repeat_block(block1_channels),
            nn.Identity(),

            downsample2x(block1_channels, block2_channels),
            repeat_block(block2_channels),
            nn.Identity(),

            downsample2x(block2_channels, block3_channels),
            repeat_block(block3_channels),
            nn.Identity(),

            downsample2x(block3_channels, block4_channels),
            repeat_block(block4_channels),
            nn.Identity(),
        ])

        self.innerfusion = nn.ModuleList([
            NormInteraction(block1_channels),
            NormInteraction(block2_channels),
            NormInteraction(block3_channels),
            NormInteraction(block4_channels),
        ])

        self.fuse_3x3convs = nn.ModuleList([
            convGRU(block4_channels, block3_channels, block3_channels),
            convGRU(block3_channels, block2_channels, block2_channels),
            convGRU(block2_channels, block1_channels, block1_channels),
        ])

        self.cls_pred_conv = nn.Conv2d(block1_channels, num_classes, 1)

        self.conv_bn_relu = conv1x1_bn_relu(block4_channels, block4_channels)

    def forward(self, x1, x2):

        feat_list1 = []
        for op in self.feature_ops1:
            x1 = op(x1)
            if isinstance(op, nn.Identity):
                feat_list1.append(x1)

        feat_list2 = []
        for op in self.feature_ops2:
            x2 = op(x2)
            if isinstance(op, nn.Identity):
                feat_list2.append(x2)

        inner_feat_list = [self.innerfusion[i](feat_list1[i], feat_list2[i]) for i, feat in enumerate(feat_list1)]


        inner_feat_list.reverse()

        feat = inner_feat_list[0]
        feat = self.conv_bn_relu(feat)
        out_feat_list = [feat]


        for i in range(len(inner_feat_list) - 1):
            b, c, h1, w1 = inner_feat_list[i + 1].size()
            b, c, h2, w2 = inner_feat_list[i].size()
            scale = h1 / h2
            inner0 = F.interpolate(out_feat_list[i], scale_factor=scale, mode='bilinear', align_corners=False)
            out = self.fuse_3x3convs[i](inner0, inner_feat_list[i + 1])

            out_feat_list.append(out)
        final_feat = out_feat_list[-1]
        logit = self.cls_pred_conv(final_feat)

        return logit


if __name__ == "__main__":

    HS = torch.randn(1, 180, 336, 496)
    SAR = torch.randn(1, 4, 336, 496)

    y = torch.randint(0, 5, (1, 336, 496))
    grf_net = PHFNet(in_channels1=180, in_channels2=4, num_classes=7)
    grf_net.cuda()

    out = grf_net(HS.cuda(), SAR.cuda())


