import torch
import torch.optim as optim
import torch.nn as nn
from torchsummary import summary

import logging

# 自定义的一些训练优化技巧以及常用函数，放在主文件里面有点乱，就整理出去了
from utils.tricks import LabelSmoothingLoss, init_net_weight, add_weight_decay, mixup_data, warm_up_with_cosine
from utils.functions import calc_accuracy
from trainer import ClassifyTrainer

# 加载本次训练所需要的模型以及数据
from models import DD_CNN as Network
from dataset import TAUUrbanAcousticScenes2020_3classDevelopment_Feature as Dataset

trainer = ClassifyTrainer('DD_CNN_TAU', 3)


def test(model, dataloader, test_criterion, device):
    """对训练结果进行测试"""

    model.eval()  # 改为测试模式，对 BN 有影响，具体为啥还需要学习
    correct = 0
    t_loss = 0.0

    with torch.no_grad():
        for (test_input, test_labels) in dataloader:
            test_input, test_labels = test_input.to(device), test_labels.to(device)
            out = model(test_input)
            correct += calc_accuracy(out, test_labels)
            t_loss += test_criterion(out, test_labels).item()

    test_acc = correct / len(dataloader.dataset)
    t_loss = t_loss / len(dataloader)

    return test_acc, t_loss


if __name__ == '__main__':
    # 获取参数
    args = trainer.args

    # 加载数据
    train_loader, test_loader = Dataset(args.batch_size, args.num_worker)

    # 定义网络
    net = Network(channel=1, num_classes=trainer.classes).to(torch.device("cuda"))

    if args.init_weight:
        init_net_weight(net, args.init_weight)  # 参数初始化

    summary(net, test_loader.dataset[0][0].shape)

    net = net.to(trainer.device)

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

    optimizer = optim.AdamW(parameters, lr=args.base_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=weight_decay, amsgrad=True)

    # warmup 和 余弦衰减
    if args.warmup and args.cosine:
        warm_up_with_cosine_lr = warm_up_with_cosine(args)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_up_with_cosine_lr)
    elif args.milestones:
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=eval(args.milestones),
            gamma=args.lr_decay
        )  # 步进的，按照里程碑对学习率进行修改
    else:
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1)

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
        trainer.train(train_loader)

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

            trainer.step(i, loss, calc_accuracy(outputs, labels))

        scheduler.step()  # 更新学习率

        # 计算本 Epoch 的损失以及测试集上的准确率
        cur_acc, test_loss = test(net, test_loader, criterion, trainer.device)

        # 保存断点，并输出训练情况
        trainer.save_checkpoint(net, optimizer, scheduler, cur_acc, test_loss)

    trainer.done()
