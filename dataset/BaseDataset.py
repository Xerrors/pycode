import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

data_dir = "./data"


def CIFAR10(batch_size=128, num_worker=8, auto_aug=False, resize=None):
    """引入CIFAR10数据集"""
    train_augments = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ]
    # 自动增强
    if auto_aug:
        from utils.autoaugment import CIFAR10Policy # 引入已经别人已经训练好的优化策略
        train_augments.insert(0, CIFAR10Policy())

    if resize:
        train_augments.insert(0, transforms.Resize(resize))

    transform_train = transforms.Compose(train_augments)
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    train_set = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_train)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_worker)

    test_set = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_test)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=num_worker)

    return train_loader, test_loader


def CIFAR100(batch_size=128, num_worker=8, auto_aug=False, resize=None):
    """Cifar100"""
    train_augments = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ]
    if auto_aug:
        # 引入已经别人已经训练好的优化策略
        from utils.autoaugment import CIFAR10Policy
        train_augments.insert(0, CIFAR10Policy())

    if resize:
        train_augments.insert(0, transforms.Resize(resize))

    transform_train = transforms.Compose(train_augments)
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    train_set = torchvision.datasets.CIFAR100(root=data_dir, train=True, download=False, transform=transform_train)
    train_loader = DataLoader(train_set, drop_last=True, batch_size=batch_size, shuffle=True, num_workers=num_worker)

    test_set = torchvision.datasets.CIFAR100(root=data_dir, train=False, download=False, transform=transform_test)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_worker)

    return train_loader, test_loader


def MNIST(batch_size=128, num_worker=8, resize=None):
    actions = [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ]

    if resize:
        actions.insert(0, transforms.Resize(resize))

    transform_train = transforms.Compose(actions)
    transform_test = transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_set = torchvision.datasets.MNIST(root=data_dir, train=True, download=True, transform=transform_train)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_worker)

    test_set = torchvision.datasets.MNIST(root=data_dir, train=False, download=True, transform=transform_test)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=num_worker)

    return train_loader, test_loader
