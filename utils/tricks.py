import torch
import torch.nn.functional as F
import torch.nn as nn

from torch.autograd import Variable

import numpy as np


class LabelSmoothCELoss(nn.Module):
    """
    标签平滑，参考：https://zhuanlan.zhihu.com/p/148487894
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred, label, smoothing=0.1):
        classes = pred.size()[1]
        pred = F.softmax(pred, dim=1)
        one_hot_label = F.one_hot(label, classes).float()
        smoothed_one_hot_label = (1.0 - smoothing) * one_hot_label + smoothing / classes
        loss = (-torch.log(pred)) * smoothed_one_hot_label
        loss = loss.sum(axis=1, keepdim=False)
        loss = loss.mean()

        return loss


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
    mixed_labels = lam * labels_a + (1-lam) * labels_b

    return Variable(mixed_inputs), Variable(mixed_labels), lam


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

