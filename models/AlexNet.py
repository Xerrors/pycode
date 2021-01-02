from torch import nn
import torch

class AlexNet(nn.Module):
    def __init__(self, channel=3, num_classes=1000):#imagenet数量
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=channel, out_channels=32, kernel_size=3, stride=2, bias=False),
            # nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, bias=False),
            # nn.BatchNorm2d(64),
            nn.ReLU6(inplace=True),

            nn.Conv2d(in_channels=64, out_channels=16, kernel_size=1, bias=False),
            # nn.BatchNorm2d(64),
            nn.ReLU6(inplace=True),

            nn.Conv2d(in_channels=16, out_channels=128, kernel_size=3, bias=False),
            # nn.BatchNorm2d(128),
            nn.ReLU6(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, groups=4, padding=2, bias=False),
            # nn.BatchNorm2d(64),
            nn.ReLU6(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
            nn.Dropout(),

            nn.Conv2d(in_channels=32, out_channels=num_classes, kernel_size=1, bias=False),
            nn.AdaptiveAvgPool2d(1)
        )
        
    def forward(self, x):
        x = self.layers(x)

        out = torch.flatten(x, 1)
        
        return out