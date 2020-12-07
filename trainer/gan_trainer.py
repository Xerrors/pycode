import torch
import logging

from trainer import MyTrainer
from utils.functions import print_and_log, save_pic


class GanTrainer(MyTrainer):
    """GANs 网络的训练器"""

    def __init__(self, NAME):
        super(GanTrainer, self).__init__(NAME=NAME)


    # def save_checkpoint(self, ):