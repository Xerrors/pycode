import torch
import logging
import argparse
import time
import os

from utils.functions import print_and_log, save_pic

class MyTrainer():
    """参数解析器类"""
    def __init__(self, NAME="Demo"):
        self.NAME = NAME
        self.init_parser() # 初始化解析器
        self.log_args()    # 对训练参数进行输出

    def init_parser(self):
        """可在此处自定义 parser"""
        parser = argparse.ArgumentParser()
        parser.add_argument('--num-worker', type=int, default=8)
        parser.add_argument('--epochs', type=int, default=640)
        parser.add_argument('--batch-size', type=int, default=128)
        parser.add_argument('--debug', action='store_true')
        parser.add_argument('--gpu', type=str, default="0")
        self.parser = parser

    def log_args(self):
        self.args = self.parser.parse_args()
        # 判断当前可用设备，并对服务器的GPU进行配置
        if torch.cuda.is_available() and self.args.gpu != "X":
            self.device = torch.device("cuda:" + self.args.gpu)
        else:
            self.device = torch.device("cpu")

        # 创建此次训练的文件夹，模型、断点、损失信息都会保存在这个文件夹下面
        training_start_time = str(time.time()).split('.')[0]
        self.model_dir = './log/' + self.NAME + '/' + training_start_time
        if not os.path.isdir(self.model_dir):
            os.mkdir(self.model_dir)

        # 配置 log 的路径以及配置信息
        if self.args.debug:
            level = logging.DEBUG
        else:
            level = logging.INFO
        logging.basicConfig(
            filename=self.model_dir + '/training.log',
            format='[%(levelname)s] %(asctime)s: %(message)s',
            level=level)

        msg = "---\n"
        msg += "Name: {}\nDevice: {}\nDebug: {}\n---\n".format(self.NAME, self.device, self.args.debug)
        msg += "Epochs: {}\nBatch Size: {}\nNum Worker: {}\n---\n".format(
            self.args.epochs, self.args.batch_size, self.args.num_worker)
        print_and_log(msg)
