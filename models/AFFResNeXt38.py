# New
"""
resneXt for cifar with pytorch
Reference:
[1] S. Xie, G. Ross, P. Dollar, Z. Tu and K. He Aggregated residual transformations for deep neural networks. In CVPR, 2017
"""

import torch
import torch.nn as nn
import math


class ASKCFuse(nn.Module):
    # 可以参考官方代码里面的 https://github.com/YimianDai/open-aff/blob/master/aff_pytorch/aff_net/fusion.py
    def __init__(self, channels=64, r=16):
        super(ASKCFuse, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels)
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x, residual):
        xa = x + residual
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = torch.sigmoid(xlg)

        xo = 2 * x * wei + 2 * residual * (1 - wei)
        return xo


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, channels, cardinality, botteneck_width, stride, downsample=False):
        super(Bottleneck, self).__init__()
        D = int(math.floor(channels * (botteneck_width / 64)))
        group_width = D * cardinality

        self.relu = nn.ReLU(inplace=True)

        self.split_transforms = nn.Sequential(
            nn.Conv2d(in_channels, group_width, kernel_size=1, bias=False),
            nn.BatchNorm2d(group_width),
            self.relu,
            nn.Conv2d(group_width, group_width, kernel_size=3, stride=stride, padding=1, groups=cardinality, bias=False),
            nn.BatchNorm2d(group_width),
            self.relu,
            nn.Conv2d(group_width, channels * 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels * 4)
        )

        # 是否对残差进行 1x1 卷积
        if downsample:
            self.downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True, count_include_pad=False),
                nn.Conv2d(in_channels, channels * 4, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(channels * 4)
            )
        else:
            self.downsample = nn.Sequential()

        self.attention = ASKCFuse(channels=channels * 4)

    def forward(self, x): 
        residual = self.downsample(x)
        out = self.split_transforms(x)

        out = self.attention(out, residual)
        out = self.relu(out)

        return out


class ResNeXt_Cifar(nn.Module):

    def __init__(self, block, num_blocks_in_layer, cardinality, bottleneck_width, num_classes=100, channel=3):
        super(ResNeXt_Cifar, self).__init__()
        self.in_channels = 64
        self.cardinality = cardinality
        self.bottleneck_width = bottleneck_width

        self.layers = nn.Sequential(
            nn.Conv2d(channel, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            *self._make_layer(block, 64, num_blocks_in_layer[0], stride=1),
            *self._make_layer(block, 128, num_blocks_in_layer[1], stride=2),
            *self._make_layer(block, 256, num_blocks_in_layer[2], stride=2),
        )

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256 * block.expansion, num_classes)

    def _make_layer(self, block, channels, blocks, stride):

        layers = []
        layers.append(block(self.in_channels, channels, self.cardinality, self.bottleneck_width, stride, downsample=True))

        self.in_channels = channels * block.expansion
        for _ in range(blocks-1):
            layers.append(block(self.in_channels, channels, self.cardinality, self.bottleneck_width, stride=1, downsample=False))

        return layers

    def forward(self, x):
        x = self.layers(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resneXt_cifar(n, cardinality, bottleneck_width, **kwargs):
    assert (n - 2) % 9 == 0
    n = int((n - 2) / 9)
    model = ResNeXt_Cifar(Bottleneck, [n, n, n], cardinality, bottleneck_width, **kwargs)
    return model


def AFFResNeXt38_32x4d_100():
    return resneXt_cifar(n=38, cardinality=32, bottleneck_width=4, num_classes=100)


def AFFResNeXt38_32x4d_10():
    return resneXt_cifar(n=38, cardinality=32, bottleneck_width=4, num_classes=10)


def AFFResNeXt38_32x4d_3_1c():
    return resneXt_cifar(n=38, cardinality=32, bottleneck_width=4, num_classes=3, channel=1)
