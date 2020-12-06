import os
import sys
import time
import math

import torch
import torch.optim as optim
import torch.nn as nn

import logging

# 自定义的一些训练优化技巧以及常用函数，放在主文件里面有点乱，就整理出去了
from utils.tricks import LabelSmoothingLoss, init_net_weight, add_weight_decay, mixup_data
from utils.functions import print_and_log, sys_flush_log
from trainer import ClassifyTrainer

# from models import DemoNet_Gray as Network
# from data import MNIST as Dataset
# NAME = 'DemoNet_Gray'

# 加载本次训练所需要的模型以及数据
from models import AFFResNeXt38_32x4d_100 as Network
from dataset import CIFAR100 as Dataset


trainer = ClassifyTrainer('AFFResNeXt38_32x4d_100')

def test(net, data, device, criterion):
    """对训练结果进行测试"""

    net.eval() # 改为测试模式，对 BN 有影响，具体为啥还需要学习
    correct = 0
    loss = 0.0
    with torch.no_grad():
        for (images, labels) in data:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
    cur_acc = correct / len(test_loader.dataset)
    return cur_acc, loss / len(test_loader)


if __name__ == '__main__':
    # 获取参数
    args = trainer.args

    # 加载数据
    train_loader, test_loader = Dataset(args.batch_size, args.num_worker, args.auto_aug)

    # 定义网络
    net = Network().to(trainer.device)
    if args.init_weight:
        init_net_weight(net, args.init_weight)  # 参数初始化

    # 保存训练过程中的 loss 信息
    train_losses = []
    test_losses = []
    acc_list = []
    best_acc = args.base_line
    start_epoch = 0

    # 权重衰减 https://discuss.pytorch.org/t/weight-decay-in-the-optimizers-is-a-bad-idea-especially-with-batchnorm/16994/3
    weight_decay = args.weight_decay
    if weight_decay and args.no_wd:
        parameters = add_weight_decay(net, weight_decay)
        weight_decay = 0.
    else:
        parameters = net.parameters()

    # 优化器
    optimizer = optim.SGD(
        parameters,
        lr=args.base_lr,
        momentum=args.momentum,
        weight_decay=weight_decay,
        nesterov=args.nesterov)

    # 余弦学习率 实际上只是半个周期，余弦从 1 到 0 罢了 https://zhuanlan.zhihu.com/p/148487894
    if args.warmup and args.cosine:
        # T_max=20 # 带周期的
        # warm_up_with_cosine_lr = lambda epoch: epoch / args.warmup_epochs if epoch <= args.warmup_epochs else 0.5 * (
        #         math.cos((epoch - args.warmup_epochs) % (T_max*2) / (T_max*2) * math.pi) + 1)

        # 从 1 到 0 Cosine
        warm_up_with_cosine_lr = lambda epoch: epoch / args.warmup_epochs if epoch <= args.warmup_epochs else 0.5 * (
                math.cos((epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs) * math.pi) + 1)
        
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_up_with_cosine_lr)
    else:
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=eval(args.milestones),
            gamma=args.lr_decay
        ) # 步进的，按照里程碑对学习率进行修改

    # 损失函数
    if args.label_smoothing:
        criterion = LabelSmoothingLoss(classes=100).to(trainer.device) # 标签平滑损失函数
    else:
        criterion = nn.CrossEntropyLoss().to(trainer.device)

    # 断点续训，功能似乎还没有实现，需要验证
    if len(args.checkpoint) > 0:
        checkpoint = torch.load(args.checkpoint)  # 加载断点
        net.load_state_dict(checkpoint['net'])  # 加载模型可学习参数
        optimizer.load_state_dict(checkpoint['optimizer'])  # 加载优化器参数
        scheduler.load_state_dict(checkpoint['scheduler'])
        train_losses = checkpoint['train_losses']
        test_losses = checkpoint['test_losses']
        acc_list = checkpoint['acc_list']
        start_epoch = len(train_losses)  # 设置开始的epoch
        best_acc = max(max(acc_list), best_acc)

    # 训练
    print("Start Training! Trining on {}.".format(trainer.device))
    for epoch in range(start_epoch, args.epochs):
        net.train() # 切换为训练模式
        running_loss = 0.0  # 运行时的 loss
        batch_num = len(train_loader)
        epoch_start_time = time.time()  # 记录起始时间

        for i, (inputs, labels) in enumerate(train_loader, 0):
            inputs, labels = inputs.to(trainer.device), labels.to(trainer.device)
            optimizer.zero_grad()
            # mixup 数据增强
            if args.mixup and args.mixup_off_epoch <= args.epochs - epoch:
                inputs, labels_a, labels_b, lam = mixup_data(inputs, labels, args.mixup_alpha, trainer.device)
                outputs = net(inputs)
                loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
            else:
                outputs = net(inputs)
                loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # 输出效果，后面的 Sys 输出也只是为了实现类似进度条的东西
            running_loss += loss.item()
            logging.debug(loss.item())
            sys_flush_log('Epoch-{:^3d}[{:>3.2f}%] '.format(epoch, (i + 1) / batch_num * 100))

        scheduler.step()  # 更新学习率

        # 计算本 Epoch 的损失以及测试集上的准确率
        train_loss = running_loss / batch_num
        cur_acc, test_loss = test(net, test_loader, trainer.device, criterion)
        acc_list.append(cur_acc)  # 准确率数组
        train_losses.append(train_loss)  # 训练损失数组
        test_losses.append(test_loss)  # 测试集损失数组
        # 输出损失信息并记录到日志
        use_time = time.time() - epoch_start_time  # 一个 epoch 的用时
        print_and_log("Epoch-{:^3d} T-Loss: {:.3f}, E-Loss: {:.3f}, Time: {:.2f}s, Need: {:.2f}h, LR: {:.4f}, Acc: {:.2f}%".format(
            epoch,
            train_loss, test_loss,
            use_time, use_time / 3600 * (args.epochs - epoch),
            optimizer.state_dict()['param_groups'][0]['lr'],
            cur_acc * 100))

        # 保存断点
        trainer.save_checkpoint(net, optimizer, scheduler, train_losses, test_losses, acc_list, best_acc, epoch)
        best_acc = max(cur_acc, best_acc)

    # 输出以及数据的保存，对训练无影响
    print_and_log('### Finished Training! Best Acc: {:2f}%'.format(best_acc * 100))
    train_info = {"training_loss": train_losses, "test_loss": test_losses, "acc_list": acc_list}
    torch.save(train_info, trainer.model_dir + "/training_info.pth")
    new_model_dir = trainer.model_dir + '_' + str(best_acc * 10000)[:4]
    os.rename(trainer.model_dir, new_model_dir)  # 更改文件夹名称，加上准确率
    os.remove(new_model_dir+"/checkpoint.pth")
