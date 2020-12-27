import torch
import logging

from trainer import MyTrainer
from utils.functions import print_and_log, save_pic


class ClassifyTrainer(MyTrainer):
    """用于图像分类的训练器"""
    def __init__(self, NAME, classes=100):
        self.classes = classes
        super(ClassifyTrainer, self).__init__(NAME=NAME)

        self.train_losses = []
        self.test_losses = []
        self.acc_list = []
        self.epoch = 0
        self.best_acc = self.args.base_line
        self.checkpoint_path = self.model_dir + "/checkpoint.pth"
        self.best_model_path = self.model_dir + '/best_model.pth'


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
        self.parser.add_argument('--milestones', type=str, default='[50,80]')
        # others
        self.parser.add_argument('--base-line', type=float, default=0.6)
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


    def save_checkpoint(self, model, optimizer, scheduler):

        self.epoch = len(self.acc_list)

        checkpoint = {
            "net": model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            "train_losses": self.train_losses,
            "test_losses": self.test_losses,
            "acc_list": self.acc_list,
            "best_acc": self.best_acc
        }
        torch.save(checkpoint, self.checkpoint_path)

        save_pic(self.model_dir, {
            "train_losses": self.train_losses,
            "test_losses": self.test_losses
        }, {
             "acc_list": self.acc_list
         })

        # 保存最优模型
        if self.acc_list[-1] > self.best_acc:
            self.best_acc = self.acc_list[-1]
            torch.save(model.state_dict(), self.best_model_path)
            logging.info("> Best Acc: {:.3f}% in epoch: {}".format(self.acc_list[-1] * 100, self.epoch))

    
    def start_with_checkpoint(self, net, optimizer, scheduler):
        """ 从断点处加载数据 """

        checkpoint = torch.load(self.checkpoint_path)

        net.load_state_dict(checkpoint['net']) # 加载模型可学习参数
        optimizer.load_state_dict(checkpoint['optimizer'])  # 加载优化器参数
        scheduler.load_state_dict(checkpoint['scheduler'])

        self.train_losses = checkpoint['train_losses']
        self.test_losses = checkpoint['test_losses']
        self.acc_list = checkpoint['acc_list']
        self.best_acc = checkpoint['best_acc']
        self.epoch = len(train_losses)  # 设置开始的epoch

        print_and_log("This Mode Will Training From Epoch {} Base On Checkpoint That Stored In {}".format(
            self.epoch, self.checkpoint_path
        ))

        return net, optimizer, scheduler 

    
    def done(self):
        """ 输出以及数据的保存，对训练无影响 """
        print_and_log('> Finished Training! Best Acc: {:2f}%'.format(self.best_acc * 100))
        train_info = {"training_loss": self.train_losses, "test_loss": self.test_losses, "acc_list": self.acc_list}
        torch.save(train_info, os.path.join(self.model_dir, "training_info.pth"))

        new_model_dir = self.model_dir + '_' + str(best_acc * 10000)[:4]
        os.rename(self.model_dir, new_model_dir)  # 更改文件夹名称，加上准确率
        os.remove(os.path.join(new_model_dir, "checkpoint.pth"))


