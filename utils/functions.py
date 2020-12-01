import torch
import logging
import argparse
import time
import os

import matplotlib.pyplot as plt


def judge_device(gpu="0"):
    """判断当前可用设备，并对服务器的GPU进行配置"""

    if torch.cuda.is_available() and gpu != "X":
        device = torch.device("cuda:"+gpu)
    else:
        device = torch.device("cpu")

    return device


def save_pic(train_losses, test_losses, acc_list, save_dir):
    """保存训练中的训练详情的图像"""

    epochs = list(range(len(acc_list)))
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('epochs')
    ax1.plot(epochs, train_losses, dashes=[2, 2, 10, 2], label='Train Loss')
    ax1.plot(epochs, test_losses, dashes=[6, 2], label='Test Loss')
    # ax1.legend() # 会遮挡准确率曲线

    ax2 = ax1.twinx()
    ax2.plot(epochs, acc_list, 'r', label="Acc")

    fig.tight_layout()
    plt.savefig(save_dir+'/training_info.jpg')


def parse_args():
    """对命令行参数进行解析"""

    parser = argparse.ArgumentParser()
    # tricks
    parser.add_argument('--auto-aug', action='store_true')
    parser.add_argument('--cosine', action='store_true')
    parser.add_argument('--no-wd', action='store_true')
    parser.add_argument('--mixup', action='store_true')
    parser.add_argument('--mixup-alpha', type=float, default=1)
    parser.add_argument('--mixup-off-epoch', type=int, default=40)
    parser.add_argument('--label-smoothing', action='store_true')
    parser.add_argument('--warmup', action='store_true')
    parser.add_argument('--warmup-epochs', type=int, default=10)
    # run config
    parser.add_argument('--num-worker', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=640)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--gpu', type=str, default="0")
    # optim config
    parser.add_argument('--base-lr', type=float, default=0.2)
    parser.add_argument('--lr-decay', type=float, default=0.1)
    parser.add_argument('--weight-decay', type=float, default=0.0001)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--nesterov', action='store_false')
    parser.add_argument('--milestones', type=str, default='[300,450]')
    # others
    parser.add_argument('--base-line', type=float, default=0.6)
    parser.add_argument('--checkpoint', type=str, default="")
    parser.add_argument('--init-weight', type=str, default="kaiming")

    args = parser.parse_args()
    return args


def save_checkpoint(path, model, optimizer, scheduler, epoch, acc_list, train_losses, test_losses, best_acc):
    """保存断点和最优模型，同时还打印日志"""

    checkpoint_path = path+"/checkpoint.pth"
    best_model_path = path+'/best_model.pth'
    checkpoint = {
        "net": model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        "train_losses": train_losses,
        "test_losses": test_losses,
        "acc_list": acc_list
    }
    torch.save(checkpoint, checkpoint_path)
    save_pic(train_losses, test_losses, acc_list, path)

    # 保存最优模型
    if acc_list[-1] > best_acc:
        torch.save(model.state_dict(), best_model_path)
        logging.info("[  BEST  ] - A best model in epoch {} - {:.3f}%".format(
            epoch, acc_list[-1] * 100
        ))


def log_parms(NAME, args, device):
    """解析参数，输出运行时的配置信息，如训练信息、优化器信息、Tricks等等，返回训练信息所在的文件夹"""

    training_start_time = str(time.time()).split('.')[0]
    model_dir = './log/' + NAME + '/' + training_start_time
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    log_path = model_dir + '/training.log'
    if args.debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logging.basicConfig(
        filename=log_path,
        format='[%(levelname)s] %(asctime)s: %(message)s',
        level=level)

    msg = "\n"
    msg += "Name: {}\nDevice: {}\nPath: {}\nDebug: {}\n---\n".format(NAME, device, model_dir, args.debug)
    msg += "Epochs: {}\nBatch Size: {}\nNum Worker: {}\n---\n".format(args.epochs, args.batch_size, args.num_worker)
    msg += "Base LR: {}\nLR Decay: {}\nWeight Decay: {}\nMomentum: {}\nMilestones: {}\nInit Weight: {}\n---\n".format(
        args.base_lr, args.lr_decay, args.weight_decay, args.momentum, args.milestones, args.init_weight
    )
    msg += "Nesterov: {}\nAuto Augment: {}\nCosine: {}\nLabel Smoothing: {}\nNo Weight Decay: {}\n---\n".format(
        args.nesterov, args.auto_aug, args.cosine, args.label_smoothing, args.no_wd
    )
    if args.mixup:
        msg += "Mixup: {}\nMixup Alpha: {}\nMixup Off Epoch: {}\n---\n".format(
            args.mixup, args.mixup_alpha, args.mixup_off_epoch)

    if args.warmup:
        msg += "Warmup: {}\nWarmup Epochs: {}\n---\n".format(args.warmup, args.warmup_epochs)

    if args.checkpoint:
        msg += "Checkpoint: {}\n---\n".format(args.checkpoint)

    print_and_log(msg)
    return model_dir

def print_and_log(msg, level="info"):
    """向控制台和日志输出信息"""
    print(msg)

    if level == "info":
        logging.info(msg)
    elif level == "debug":
        logging.debug(msg)


    
