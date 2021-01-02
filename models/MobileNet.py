import torch.nn as nn


class InvertedResiduals(nn.Module):
    """
    Inverted residuals block.
    """

    def __init__(self, in_panel, out_panel, stride, expand_ratio):
        super(InvertedResiduals, self).__init__()

        hidden_panels = in_panel * expand_ratio

        layers = []
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_panel, hidden_panels, 1, stride=1, bias=False),
                nn.BatchNorm2d(hidden_panels),
                nn.ReLU6(inplace=True)
            ])

        layers.extend([
            nn.Conv2d(hidden_panels, hidden_panels, 3, stride=stride, padding=1, groups=hidden_panels, bias=False),
            nn.BatchNorm2d(hidden_panels),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden_panels, out_panel, 1, stride=1, bias=False),
            nn.BatchNorm2d(out_panel)
        ])

        self.residuals = nn.Sequential(*layers)


    def forward(self, x):
        out = self.residuals(x)
        return out

class MobileNetV2(nn.Module):
    """
    Mobile Net.
    """
    def __init__(self, channel=3, num_classes=1000, inverted_residual_setting=None):
        super(MobileNetV2, self).__init__()

        input_channel = 32
        self.last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]


            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 64, 2, 2],
                [6, 320, 1, 1],
            ]

        features = [nn.Sequential(
            nn.Conv2d(channel, input_channel, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(input_channel),
            nn.ReLU6(inplace=True)
        )]

        for t, c, n, s in inverted_residual_setting:
            output_channel = c
            for i in range(n):
                stride = s if i==0 else 1
                features.append(InvertedResiduals(input_channel, output_channel, stride=stride, expand_ratio=t))
                input_channel = output_channel

        features.append(nn.Sequential(
            nn.Conv2d(input_channel, self.last_channel, 1, bias=False)
        ))

        self.features = nn.Sequential(*features)

        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes),
        )

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)


    def forward(self, x):
        x = self.features(x)
        x = self.avg_pool(x).reshape(x.shape[0], -1)
        x = self.classifier(x)
        return x






