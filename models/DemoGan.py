import torch
import torch.nn as nn


class Discriminator(nn.Module):
    """
    定义判别器 Discriminator: 使用多层网络来作为判别器
    将图片28x28展开成784，然后通过多层感知器，中间经过斜率设置为0.2的LeakyReLU激活函数，
    最后接sigmoid激活函数得到一个0到1之间的概率进行二分类。
    """

    def __init__(self):
        super(Discriminator, self).__init__()
        self.dis = nn.Sequential(
            nn.Linear(784, 256),  # 输入特征数为784，输出为256
            nn.LeakyReLU(0.2),  # 进行非线性映射
            nn.Linear(256, 256),  # 进行一个线性映射
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()  # 也是一个激活函数，二分类问题中，
            # sigmoid可以班实数映射到【0,1】，作为概率值，
            # 多分类用softmax函数
        )

    def forward(self, x):
        x = self.dis(x)
        return x


class Generator(nn.Module):
    """
    定义生成器 Generator: 输入一个100维的0～1之间的高斯分布，然后通过第一层线性变换将其映射到256维,
    然后通过LeakyReLU激活函数，接着进行一个线性变换，再经过一个LeakyReLU激活函数，
    然后经过线性变换将其变成784维，最后经过Tanh激活函数是希望生成的假的图片数据分布
    能够在-1～1之间。
    """

    def __init__(self):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            nn.Linear(100, 256),  # 用线性变换将输入映射到256维
            nn.ReLU(True),  # relu激活
            nn.Linear(256, 256),  # 线性变换
            nn.ReLU(True),  # relu激活
            nn.Linear(256, 784),  # 线性变换
            nn.Tanh()  # Tanh激活使得生成数据分布在【-1,1】之间
        )

    def forward(self, x):
        x = self.gen(x)
        return x


def DemoGanMnist(device):
    """用于测试的 GAN 网络"""
    return Generator().to(device), Discriminator.to(device)