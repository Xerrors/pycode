import torch
import torch.nn.functional as F
import torch.nn as nn

class LabelSmoothCELoss(nn.Module):
    """
    标签平滑，参考：https://zhuanlan.zhihu.com/p/148487894
    """
    def __init__(self):
        super().__init__()
        self.CrossEntropyLoss = nn.CrossEntropyLoss()

    def forward(self, pred, label, smoothing=0.1):
        one_hot_label = F.one_hot(label, pred.size(1)).float()
        label = (1.0 - smoothing) * one_hot_label + smoothing / pred.size(1)
        loss = self.CrossEntropyLoss(pred, label)
        return loss

def phrase_labels(labels, classes, lam, label_smoothing):
    eta = 0.1 # label_smoothing 的参数
    one_hot_label = F.one_hot(label, classes).float()

    if label_smoothing:
        label_1 = (1.0 - eta) * one_hot_label + eta / classes
        label_2 = label_1[::-1]



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

