'''
第五章 PyTorch常用工具模块
在训练神经网络过程中，需要用到很多工具，其中最重要的三部分是：数据、可视化和GPU加速。本章主要介绍Pytorch在这几方面的工具模块，合理使用这些工具能够极大地提高编码效率。
'''

'''
5.1 数据处理
在解决深度学习问题的过程中，往往需要花费大量的精力去处理数据，包括图像、文本、语音或其它二进制数据等。
数据的处理对训练神经网络来说十分重要，良好的数据处理不仅会加速模型训练，更会提高模型效果。考虑到这点，PyTorch提供了几个高效便捷的工具，以便使用者进行数据处理或增强等操作，
同时可通过并行化加速数据加载。

5.1.1 数据加载
在PyTorch中，数据加载可通过自定义的数据集对象。数据集对象被抽象为Dataset类，实现自定义的数据集需要继承Dataset，并实现两个Python魔法方法：

__getitem__：返回一条数据，或一个样本。obj[index]等价于obj.__getitem__(index)
__len__：返回样本的数量。len(obj)等价于obj.__len__()
这里我们以Kaggle经典挑战赛"Dogs vs. Cat"的数据为例，来详细讲解如何处理数据。"Dogs vs. Cats"是一个分类问题，
判断一张图片是狗还是猫，其所有图片都存放在一个文件夹下，根据文件名的前缀判断是狗还是猫。
'''
# 深度学习包
import torch
from torch.utils import data

import os
import numpy as np
from PIL import Image


class DogCat(data.Dataset):
    def __init__(self, root):
        imgs = os.listdir(root)
        # 所有图片的绝对路径
        # 这里没有加载图片，只是指定路径，当调用__getitem__的时候才会读取图片
        self.imgs = [os.path.join(root, img) for img in imgs]

    def __getitem__(self, index):
        img_path = self.imgs[index]
        # dog -- 1, cat -- 0
        label = 1 if 'dog' in img_path.split('/')[-1] else 0  # 第二种写法想一想
        pil_img = Image.open(img_path)
        array = np.asarray(pil_img)
        data = torch.from_numpy(array)
        return data, label

    def __len__(self):
        return len(self.imgs)


dataset = DogCat('dogcat')  # 这里用的相对路径
# print(dataset)
img, label = dataset[0]
# print(img, label)

# for img, label in dataset:
#     print('--------------------------------')
#     print(img.size(), img.float().mean(), label)

'''
通过上面的代码，我们学习了如何自定义自己的数据集，并可以依次获取。但这里返回的数据不适合实际使用，因其具有如下两方面问题：

返回样本的形状不一，因每张图片的大小不一样，这对于需要取batch训练的神经网络来说很不友好
返回样本的数值较大，未归一化至[-1, 1]
针对上述问题，PyTorch提供了torchvision[^1]。它是一个视觉工具包，提供了很多视觉图像处理的工具，其中transforms模块提供了对PIL Image对象和Tensor对象的常用操作。

对PIL Image的操作包括：

Scale：调整图片尺寸，长宽比保持不变
CenterCrop、RandomCrop、RandomResizedCrop： 裁剪图片
Pad：填充
ToTensor：将PIL Image对象转成Tensor，会自动将[0, 255]归一化至[0, 1]
对Tensor的操作包括：

Normalize：标准化，即减均值，除以标准差
ToPILImage：将Tensor转为PIL Image对象

如果要对图片进行多个操作，可通过Compose函数将这些操作拼接起来，类似于nn.Sequential。注意，这些操作定义后是以函数的形式存在，真正使用时需调用它的__call__方法，这点类似于nn.Module。
例如要将图片调整为
，首先应构建这个操作trans = Resize((224, 224))，然后调用trans(img)。下面我们就用transforms的这些操作来优化上面实现的dataset。 
[^1]: https://github.com/pytorch/vision/
'''

import os
from PIL import Image
import numpy as np
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize(224),  # 缩放图片（Image）,保存长宽比不变，最短边为224pixel
    transforms.CenterCrop(224),  # 从图片中间切出来
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
])


class DogCat(data.Dataset):
    def __init__(self, root, transforms=None):
        imgs = os.listdir(root)

        # 所有图片的绝对路径
        # 这里没有加载图片，只是指定路径，当调用__getitem__的时候才会读取图片
        self.imgs = [os.path.join(root, img) for img in imgs]
        self.transforms = transforms

    def __getitem__(self, index):
        img_path = self.imgs[index]
        # dog -- 1, cat -- 0
        label = 0 if 'dog' in img_path.split('/')[-1] else 1  # 第二种写法想一想
        data = Image.open(img_path)
        if self.transforms:
            data = self.transforms(data)
        return data, label

    def __len__(self):
        return len(self.imgs)


dataset = DogCat('dogcat', transforms=transform)
img, label = dataset[0]
for img, label in dataset:
    print(img.size(), label)

'''
除了上述操作之外，transforms还可通过Lambda封装自定义的转换策略。例如想对PIL Image进行随机旋转，则可写成这样trans=T.Lambda(lambda img: img.rotate(random()*360))。

torchvision已经预先实现了常用的Dataset，包括前面使用过的CIFAR-10，以及ImageNet、COCO、MNIST、LSUN等数据集，可通过诸如torchvision.datasets.CIFAR10来调用，具体使用方法请参看官方文档^1。
在这里介绍一个会经常使用到的Dataset——ImageFolder，它的实现和上述的DogCat很相似。ImageFolder假设所有的文件按文件夹保存，每个文件夹下存储同一个类别的图片，文件夹名为类名，其构造函数如下：

ImageFolder(root, transform=None, target_transform=None, loader=default_loader)
它主要有四个参数：

root：在root指定的路径下寻找图片
transform：对PIL Image进行的转换操作，transform的输入是使用loader读取图片的返回对象
target_transform：对label的转换
loader：给定路径后如何读取图片，默认读取为RGB格式的PIL Image对象
label是按照文件夹名顺序排序后存成字典，即{类名:类序号(从0开始)}，一般来说最好直接将文件夹命名为从0开始的数字，这样会和ImageFolder实际的label一致，如果不是这种命名规范，
建议看看self.class_to_idx属性以了解label和文件夹名的映射关系。
'''

from torchvision.datasets import ImageFolder
dataset = ImageFolder('dogcat_2')

print(dataset.class_to_idx)
print(dataset.imgs)
# dataset[0][0].show()

'''
Dataset只负责数据的抽象，一次调用__getitem__只返回一个样本。前面提到过，在训练神经网络时，最好是对一个batch的数据进行操作，同时还需要对数据进行shuffle和并行加速等。对此，PyTorch提供了DataLoader帮助我们实现这些功能。

DataLoader的函数定义如下： DataLoader(dataset, batch_size=1, shuffle=False, sampler=None, num_workers=0, collate_fn=default_collate, pin_memory=False, drop_last=False)

dataset：加载的数据集(Dataset对象)
batch_size：batch size
shuffle:：是否将数据打乱
sampler： 样本抽样，后续会详细介绍
num_workers：使用多进程加载的进程数，0代表不使用多进程
collate_fn： 如何将多个样本数据拼接成一个batch，一般使用默认的拼接方式即可
pin_memory：是否将数据保存在pin memory区，pin memory中的数据转到GPU会快一些
drop_last：dataset中的数据个数可能不是batch_size的整数倍，drop_last为True会将多出来不足一个batch的数据丢弃
'''

from torch.utils.data import DataLoader

dataloader = DataLoader(dataset, batch_size=3, shuffle=True, num_workers=0, drop_last=False)

dataiter = iter(dataloader)
print(dataiter)
imgs, labels = next(dataiter) # batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'>
# 可以看到上面的dataset没有transform
print(imgs, labels)
