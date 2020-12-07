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

