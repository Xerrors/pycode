import torch
import logging

from trainer import MyTrainer
from utils.functions import print_and_log, save_pic


class ClassifyTrainer(MyTrainer):
    """用于图像分类的训练器"""
    def __init__(self, NAME):
        super(ClassifyTrainer, self).__init__(NAME=NAME)

    def init_parser(self):
        super(ClassifyTrainer, self).init_parser()
        # tricks
        self.parser.add_argument('--auto-aug', action='store_true')
        self.parser.add_argument('--cosine', action='store_true')
        self.parser.add_argument('--no-wd', action='store_true')
        self.parser.add_argument('--mixup', action='store_true')
        self.parser.add_argument('--mixup-alpha', type=float, default=1)
        self.parser.add_argument('--mixup-off-epoch', type=int, default=40)
        self.parser.add_argument('--label-smoothing', action='store_true')
        self.parser.add_argument('--warmup', action='store_true')
        self.parser.add_argument('--warmup-epochs', type=int, default=10)
        # optim config
        self.parser.add_argument('--base-lr', type=float, default=0.2)
        self.parser.add_argument('--lr-decay', type=float, default=0.1)
        self.parser.add_argument('--weight-decay', type=float, default=0.0001)
        self.parser.add_argument('--momentum', type=float, default=0.9)
        self.parser.add_argument('--nesterov', action='store_false')
        self.parser.add_argument('--milestones', type=str, default='[300,450]')
        # others
        self.parser.add_argument('--base-line', type=float, default=0.6)
        self.parser.add_argument('--checkpoint', type=str, default="")
        self.parser.add_argument('--init-weight', type=str, default="kaiming")

    def log_args(self):
        super(ClassifyTrainer, self).log_args()
        args = self.args
        msg = "The parameters of image classifier\n---\n"
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

    def save_checkpoint(self, model, optimizer, scheduler, train_losses, test_losses, acc_list, best_acc, epoch):
        checkpoint_path = self.model_dir + "/checkpoint.pth"
        best_model_path = self.model_dir + '/best_model.pth'
        checkpoint = {
            "net": model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            "train_losses": train_losses,
            "test_losses": test_losses,
            "acc_list": acc_list
        }
        torch.save(checkpoint, checkpoint_path)
        save_pic(self.model_dir, {
            "train_losses": train_losses,
            "test_losses": test_losses
        }, {
             "acc_list": acc_list
         })

        # 保存最优模型
        if acc_list[-1] > best_acc:
            torch.save(model.state_dict(), best_model_path)
            logging.info("> Best Acc: {:.3f}% in epoch: {}".format(acc_list[-1] * 100, epoch))
