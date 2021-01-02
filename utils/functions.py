import logging
import sys
import os
import torch
import matplotlib.pyplot as plt

def save_pic(save_dir, ax1_data, ax2_data=None):
    """
    保存训练的时候的图片
    :param ax1_data: dict 左边的纵坐标显示的数据
    :param ax2_data: dict 右边的纵坐标显示的数据
    :param save_dir: 保存的路径
    :return: None
    """
    # TODO 修改 legend 的位置为左上角
    if len(ax1_data) == 0:
        print_and_log("SAVE_PIC: 传入的数据 ax1_data 为空", "debug")
        return

    fig, ax1 = plt.subplots()
    iters = list(range(len(list(ax1_data.values())[0])))
    ax1.set_xlabel('iters')
    for key, value in ax1_data.items():
        ax1.plot(iters, value, dashes=[6, 2], label=key)
    ax1.legend(loc=3) # 会遮挡准确率曲线

    if ax2_data:
        ax2 = ax1.twinx()
        for key, value in ax2_data.items():
            ax2.plot(iters, value, label=key)

    fig.tight_layout()
    plt.savefig(save_dir+'/training_info.jpg')
    plt.close()


def print_and_log(msg, level="info"):
    """向控制台和日志输出信息"""
    if level == "info":
        logging.info(msg)
    elif level == "debug":
        logging.debug(msg)
    print(msg)


def sys_flush_log(msg):
    # V1
    # sys.stdout.write('\r')
    # sys.stdout.write(msg)
    # sys.stdout.flush()

    # V2
    print('\r', msg, end="")


# from: https://github.com/MTG/DCASE-models/blob/e371511b49/dcase_models/util/files.py
def mkdir_if_not_exists(path, parents=False):
    """ Make dir if does not exists.
    If parents is True, also creates all parents needed.
    Parameters
    ----------
    path : str
        Path to folder to be created.
    parents : bool, optional
        If True, also creates all parents needed.
    """
    if not os.path.exists(path):
        if parents:
            os.makedirs(path)
        else:
            os.mkdir(path)
    
def calc_accuracy(predictions, labels):
    """ 比较准确率 """
    pred = torch.max(predictions, 1)[1]
    correct = pred.eq(labels.data.view_as(pred)).sum()
    return correct