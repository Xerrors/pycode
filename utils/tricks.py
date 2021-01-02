import math
import torch
import torch.nn.functional as F
import torch.nn as nn

from torch.autograd import Variable

import numpy as np


class LabelSmoothingLoss(nn.Module):
    # ref: https://github.com/pytorch/pytorch/issues/7455#issuecomment-513062631
    def __init__(self, classes, smoothing=0.1, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


class Disout(nn.Module):
    """
    Beyond Dropout: Feature Map Distortion to Regularize Deep Neural Networks
    https://arxiv.org/abs/2002.11022
    Args:
        dist_prob (float): probability of an element to be distorted.
        block_size (int): size of the block to be distorted.
        alpha: the intensity of distortion.
    Shape:
        - Input: `(N, C, H, W)`
        - Output: `(N, C, H, W)`
    """

    def __init__(self, dist_prob, block_size=6, alpha=1.0):
        super(Disout, self).__init__()

        self.dist_prob = dist_prob
        self.weight_behind = None

        self.alpha = alpha
        self.block_size = block_size

    def forward(self, x):

        if not self.training:
            return x
        else:
            x = x.clone()
            if x.dim() == 4:
                width = x.size(2)
                height = x.size(3)

                seed_drop_rate = self.dist_prob * (width * height) / self.block_size ** 2 / (
                            (width - self.block_size + 1) * (height - self.block_size + 1))

                valid_block_center = torch.zeros(width, height, device=x.device).float()
                valid_block_center[int(self.block_size // 2):(width - (self.block_size - 1) // 2),
                int(self.block_size // 2):(height - (self.block_size - 1) // 2)] = 1.0

                valid_block_center = valid_block_center.unsqueeze(0).unsqueeze(0)

                randdist = torch.rand(x.shape, device=x.device)

                block_pattern = ((1 - valid_block_center + float(1 - seed_drop_rate) + randdist) >= 1).float()

                if self.block_size == width and self.block_size == height:
                    block_pattern = torch.min(block_pattern.view(x.size(0), x.size(1), x.size(2) * x.size(3)), dim=2)[
                        0].unsqueeze(-1).unsqueeze(-1)
                else:
                    block_pattern = -F.max_pool2d(input=-block_pattern, kernel_size=(self.block_size, self.block_size),
                                                  stride=(1, 1), padding=self.block_size // 2)

                if self.block_size % 2 == 0:
                    block_pattern = block_pattern[:, :, :-1, :-1]
                percent_ones = block_pattern.sum() / float(block_pattern.numel())

                if not (self.weight_behind is None) and not (len(self.weight_behind) == 0):
                    wtsize = self.weight_behind.size(3)
                    weight_max = self.weight_behind.max(dim=0, keepdim=True)[0]
                    sig = torch.ones(weight_max.size(), device=weight_max.device)
                    sig[torch.rand(weight_max.size(), device=sig.device) < 0.5] = -1
                    weight_max = weight_max * sig
                    weight_mean = weight_max.mean(dim=(2, 3), keepdim=True)
                    if wtsize == 1:
                        weight_mean = 0.1 * weight_mean
                    # print(weight_mean)
                mean = torch.mean(x).clone().detach()
                var = torch.var(x).clone().detach()

                if not (self.weight_behind is None) and not (len(self.weight_behind) == 0):
                    dist = self.alpha * weight_mean * (var ** 0.5) * torch.randn(*x.shape, device=x.device)
                else:
                    dist = self.alpha * 0.01 * (var ** 0.5) * torch.randn(*x.shape, device=x.device)

            x = x * block_pattern
            dist = dist * (1 - block_pattern)
            x = x + dist
            x = x / percent_ones
            return x


class LinearScheduler(nn.Module):
    def __init__(self, disout, start_value, stop_value, nr_steps):
        super(LinearScheduler, self).__init__()
        self.disout = disout
        self.i = 0
        self.drop_values = np.linspace(start=start_value, stop=stop_value, num=nr_steps)

    def forward(self, x):
        return self.disout(x)

    def step(self):
        if self.i < len(self.drop_values):
            self.disout.dist_prob = self.drop_values[self.i]
        self.i += 1


def mixup_data(inputs, labels, alpha=1.0, device=torch.device("cpu")):
    """
    paper: https://arxiv.org/abs/1710.09412
    GitHub: https://github.com/hongyi-zhang/mixup
    source: https://github.com/hongyi-zhang/mixup/blob/80000cea34/cifar/easy_mixup.py
    """
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    batch_size = inputs.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_inputs = lam * inputs + (1 - lam) * inputs[index, :]
    labels_a, labels_b = labels, labels[index]

    return Variable(mixed_inputs), Variable(labels_a), Variable(labels_b), lam


def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]


def init_net_weight(net, name="kaiming"):
    """
    对权重进行初始化，Kaiming, Xavier
    """
    for layer in net.modules():
        # 对卷积层和线性层进行初始化
        if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
            if name == "xavier" or name == "Xavier":
                layer.weight.data = torch.nn.init.xavier_uniform_(layer.weight)
            if name == "kaiming" or name == "Kaiming":
                pass

# custom weights initialization called on netG and netD
def weights_init(m):
    """
    Just Demo For DCGAN
    From the DCGAN paper, the authors specify that all model weights shall be randomly initialized from a
    Normal distribution with mean=0, stdev=0.02. The weights_init function takes an initialized model as
    input and reinitializes all convolutional, convolutional-transpose, and batch normalization layers to
    meet this criteria. This function is applied to the models immediately after initialization.
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def warm_up_with_cosine(args, T_max=None):
    """ 余弦学习率 实际上只是半个周期，余弦从 1 到 0 罢了
        <https://zhuanlan.zhihu.com/p/148487894>
    args:
        - args: 参数
        - T_max: 关于周期的，如果设置周期，将按照半周期衰减，否则从 1 衰减到 0 
    """
    if T_max:
        warm_up_with_cosine_lr = lambda epoch: epoch / args.warmup_epochs if epoch <= args.warmup_epochs else 0.5 * (
            math.cos((epoch - args.warmup_epochs) % (T_max*2) / (T_max*2) * math.pi) + 1)
    else:
        warm_up_with_cosine_lr = lambda epoch: epoch / args.warmup_epochs if epoch <= args.warmup_epochs else 0.5 * (
            math.cos((epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs) * math.pi) + 1)
    
    return warm_up_with_cosine_lr

