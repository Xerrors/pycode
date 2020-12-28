import torch.nn as nn


class FireBlock(nn.Module):
    """ fire block in Squeeze Net """

    def __init__(self, in_num, out_num, sr=0.125, pct=0.5):
        super(FireBlock, self).__init__()
        s1_num = int(out_num * sr)
        self.s1 = nn.Conv2d(in_num, s1_num, kernel_size=1)
        self.e1 = nn.Conv2d(s1_num, int(out_num * (1 - pct)), kernel_size=1)
        self.e3 = nn.Conv2d(s1_num, int(out_num * pct), kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.s1(x))
        out1 = self.e1(out)
        out2 = self.e3(out)
        out = self.relu(torch.cat((out1, out2), 1))
        return out


class SqueezeNet(nn.Module):
    """ Squeeze Net <http://arxiv.org/abs/1602.07360> """

    def __init__(self):
        super(SqueezeNet, self).__init__()
        base_e = 128
        sr = 0.125
        pct = 0.5

        self.fire = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=7, stride=(1, 2)),
            nn.MaxPool2d(kernel_size=3, stride=2),
            FireBlock(96, 128, sr, pct),
            FireBlock(128, 128, sr, pct),
            FireBlock(128, 256, sr, pct),
            nn.MaxPool2d(kernel_size=3, stride=2),
            FireBlock(256, 256, sr, pct),
            FireBlock(256, 384, sr, pct),
            FireBlock(384, 384, sr, pct),
            FireBlock(384, 512, sr, pct),
            nn.MaxPool2d(kernel_size=3, stride=2),
            FireBlock(512, 512, sr, pct),
            nn.Dropout(p=0.5),
            nn.Conv2d(512, 3, kernel_size=1, stride=1),
            nn.AdaptiveAvgPool2d(1)
        )

    def forward(self, x):
        out = self.fire(x)
        out = out.view(out.size(0), -1)
        # out = torch.softmax(out, dim=1)
        return out



""" 下面的部分是我从 Pytorch 上面找到的 """

import torch
import torch.nn as nn
import torch.nn.init as init
from typing import Any

__all__ = ['SqueezeNet', 'squeezenet1_0', 'squeezenet1_1']

model_urls = {
    'squeezenet1_0': 'https://download.pytorch.org/models/squeezenet1_0-a815701f.pth',
    'squeezenet1_1': 'https://download.pytorch.org/models/squeezenet1_1-f364aa15.pth',
}


class Fire(nn.Module):

    def __init__(
            self,
            inplanes: int,
            squeeze_planes: int,
            expand1x1_planes: int,
            expand3x3_planes: int
    ) -> None:
        super(Fire, self).__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes,
                                   kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes,
                                   kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))
        ], 1)


class SqueezeNet_Torch(nn.Module):

    def __init__(
            self,
            version: str = '1_0',
            num_classes: int = 1000,
            channel=3
    ) -> None:
        super(SqueezeNet_Torch, self).__init__()
        self.num_classes = num_classes
        if version == '1_0':
            self.features = nn.Sequential(
                nn.Conv2d(channel, 96, kernel_size=7, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(96, 16, 64, 64),
                Fire(128, 16, 64, 64),
                Fire(128, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 32, 128, 128),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(512, 64, 256, 256),
            )
        elif version == '1_1':
            self.features = nn.Sequential(
                nn.Conv2d(channel, 64, kernel_size=3, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(64, 16, 64, 64),
                Fire(128, 16, 64, 64),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(128, 32, 128, 128),
                Fire(256, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                Fire(512, 64, 256, 256),
            )
        else:
            raise ValueError("Unsupported SqueezeNet version {version}:"
                             "1_0 or 1_1 expected".format(version=version))

        # Final convolution is initialized differently from the rest
        final_conv = nn.Conv2d(512, self.num_classes, kernel_size=1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            final_conv,
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m is final_conv:
                    init.normal_(m.weight, mean=0.0, std=0.01)
                else:
                    init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return torch.flatten(x, 1)


def _squeezenet(version: str, pretrained: bool, progress: bool, **kwargs: Any) -> SqueezeNet_Torch:
    model = SqueezeNet_Torch(version, **kwargs)
    return model


def squeezenet1_0(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> SqueezeNet_Torch:
    return _squeezenet('1_0', pretrained, progress, **kwargs)


def squeezenet1_1(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> SqueezeNet_Torch:
    return _squeezenet('1_1', pretrained, progress, **kwargs)
