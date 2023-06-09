'''
Small Try：CIFAR-10
'''
import torchvision as tv
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage
import torch
from torch.utils.data import DataLoader, Dataset

# 分类数据

classes = ('plane', 'car', 'bird', 'cat', 'deer',
         'dog', 'frog', 'horse', 'ship', 'truck')

# 定义数据集


show = ToPILImage()  # 可以把Tensor转换成Image, 方便可视化


def main():
    # 第一次运行程序torchvision会自动下载CIFAR-10数据集
    # 大约100M，需要一定时间
    # 如果已经下载有CIFAR-10，可通过root参数指定

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # TrainSet
    trainset = tv.datasets.CIFAR10(
        root='/media/benben/0ECABB60A248B50C',
        train=True,
        download=False,
        transform=transform
    )

    trainloader = DataLoader(
        trainset,
        batch_size=4,
        shuffle=True,
        num_workers=2
    )

    testset = tv.datasets.CIFAR10(
        root='/media/benben/0ECABB60A248B50C',
        train=True,
        download=False,
        transform=transform
    )

    testdataloader = DataLoader(
        trainset,
        batch_size=4,
        shuffle=False,
        num_workers=2
    )

    # 试验一下
    (data, label) = trainset[100]
    print(classes[label])
    show((data + 1) / 2).resize((100, 100))


if __name__ == '__main__':
    main()
