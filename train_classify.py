import os
import sys
import time
import math

import torch
import torch.optim as optim
import torch.nn as nn
from torchsummary import summary

import logging

# 自定义的一些训练优化技巧以及常用函数，放在主文件里面有点乱，就整理出去了
from utils.tricks import LabelSmoothingLoss, init_net_weight, add_weight_decay, mixup_data, warm_up_with_cosine
from utils.functions import print_and_log, sys_flush_log, calc_accuracy
from trainer import ClassifyTrainer

# 加载本次训练所需要的模型以及数据
from models import squeezenet1_1 as Network
from dataset import TAUUrbanAcousticScenes2020_3classDevelopment_Feature as Dataset

trainer = ClassifyTrainer('SequeezeNetInTAU', 3)


def test(net, dataloader, criterion, device):
    """对训练结果进行测试"""

    net.eval()  # 改为测试模式，对 BN 有影响，具体为啥还需要学习
    correct = 0
    loss = 0.0
    logging.debug("Start Test")

    with torch.no_grad():
        for (inputs, labels) in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            loss += criterion(outputs, labels).item()
            correct += calc_accuracy(outputs, labels)

    cur_acc = correct / len(dataloader.dataset)
    test_loss = loss / len(dataloader)

    return cur_acc, test_loss


if __name__ == '__main__':
    # 获取参数
    args = trainer.args

    # 加载数据
    train_loader, test_loader = Dataset(args.batch_size, args.num_worker)

    # 定义网络
    net = Network(channel=1, num_classes=trainer.classes).to(trainer.device)

    if args.init_weight:
        init_net_weight(net, args.init_weight)  # 参数初始化
    
    summary(net, (1,64,400))

    # 权重衰减 https://discuss.pytorch.org/t/weight-decay-in-the-optimizers-is-a-bad-idea-especially-with-batchnorm/16994/3
    weight_decay = args.weight_decay
    if weight_decay and args.no_wd:
        parameters = add_weight_decay(net, weight_decay)
        weight_decay = 0.
    else:
        parameters = net.parameters()

    # SGD 优化器
    # optimizer = optim.SGD(
    #     parameters,
    #     lr=args.base_lr,
    #     momentum=args.momentum,
    #     weight_decay=weight_decay,
    #     nesterov=args.nesterov)

    optimizer = optim.AdamW(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=True)

    # warmup 和 余弦衰减
    if args.warmup and args.cosine:
        warm_up_with_cosine_lr = warm_up_with_cosine(args)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_up_with_cosine_lr)
    else:
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=eval(args.milestones),
            gamma=args.lr_decay
        )  # 步进的，按照里程碑对学习率进行修改

    # 损失函数
    if args.label_smoothing:
        criterion = LabelSmoothingLoss(classes=trainer.classes).to(trainer.device)  # 标签平滑损失函数
    else:
        criterion = nn.CrossEntropyLoss().to(trainer.device)

    # 断点续训，功能似乎还没有实现，需要验证
    if args.checkpoint:
        net, optimizer, scheduler = trainer.start_with_checkpoint(net, optimizer, scheduler)

    # 训练
    print("Start Training! Trining on {}.".format(trainer.device))
    for epoch in range(trainer.epoch, args.epochs):
        net.train()  # 切换为训练模式
        running_loss = 0.0  # 运行时的 loss
        correct = 0
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


            running_loss += loss.item()
            correct += calc_accuracy(outputs, labels)

            logging.debug(loss.item())
            logging.debug(correct / args.batch_size / (i+1) * 100)
            sys_flush_log('Epoch-{:^3d}[{:>3.2f}%] '.format(epoch, (i + 1) / batch_num * 100))


        scheduler.step()  # 更新学习率

        # 计算本 Epoch 的损失以及测试集上的准确率
        train_loss = running_loss / batch_num
        cur_acc, test_loss = test(net, test_loader, criterion, trainer.device)

        trainer.acc_list.append(cur_acc)  # 准确率数组
        trainer.train_losses.append(train_loss)  # 训练损失数组
        trainer.test_losses.append(test_loss)  # 测试集损失数组

        # 输出损失信息并记录到日志
        use_time = time.time() - epoch_start_time  # 一个 epoch 的用时
        print_and_log(
            "[{:^3d}/{}], Loss: {:.3f}/{:.3f}, Time: {:.2f}s/{:.2f}h, LR: {:.4f}, Acc: {:.2f}%/{:.2f}%".format(
                epoch, args.epochs, train_loss, test_loss,
                use_time, use_time / 3600 * (args.epochs - epoch),
                optimizer.state_dict()['param_groups'][0]['lr'],
                correct / len(train_loader.dataset) * 100, cur_acc * 100))

        # 保存断点
        trainer.save_checkpoint(net, optimizer, scheduler)

    trainer.done()
