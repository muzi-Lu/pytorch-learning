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
# imgs, labels = next(dataiter) # batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'>
# 可以看到上面的dataset没有transform
# print(imgs, labels)

'''
在数据处理中，有时会出现某个样本无法读取等问题，比如某张图片损坏。这时在__getitem__函数中将出现异常，此时最好的解决方案即是将出错的样本剔除。
如果实在是遇到这种情况无法处理，则可以返回None对象，然后在Dataloader中实现自定义的collate_fn，将空对象过滤掉。
但要注意，在这种情况下dataloader返回的batch数目会少于batch_size。
'''

class NewDogCat(DogCat): # 继承前面实现的DogCat数据集
    def __getitem__(self, index):
        try:
            # 调用父类的获取函数，即 DogCat.__getitem__(self, index)
            return super(NewDogCat,self).__getitem__(index)
        except:
            return None, None

from torch.utils.data.dataloader import default_collate # 导入默认的拼接方式
def my_collate_fn(batch):
    '''
    batch中每个元素形如(data, label)
    '''
    # 过滤为None的数据
    batch = list(filter(lambda x:x[0] is not None, batch))
    if len(batch) == 0: return torch.Tensor()
    return default_collate(batch) # 用默认方式拼接过滤后的batch数据

dataset = NewDogCat('dogcat_wrong/', transforms=transform)

print(dataset[5][0].shape)

dataloader = DataLoader(dataset, 2, collate_fn=my_collate_fn, num_workers=1,shuffle=True)
for batch_datas, batch_labels in dataloader:
    print(batch_datas.size(), batch_labels.size())

'''
来看一下上述batch_size的大小。其中第2个的batch_size为1，这是因为有一张图片损坏，导致其无法正常返回。
而最后1个的batch_size也为1，这是因为共有9张（包括损坏的文件）图片，无法整除2（batch_size），因此最后一个batch的数据会少于batch_szie，
可通过指定drop_last=True来丢弃最后一个不足batch_size的batch。

对于诸如样本损坏或数据集加载异常等情况，还可以通过其它方式解决。例如但凡遇到异常情况，就随机取一张图片代替：
'''

class NewDogCat(DogCat):
    def __getitem__(self, index):
        try:
            return super(NewDogCat, self).__getitem__(index)
        except:
            new_index = np.random.randint(0, len(self)-1)
            return self[new_index]

'''
相比较丢弃异常图片而言，这种做法会更好一些，因为它能保证每个batch的数目仍是batch_size。但在大多数情况下，最好的方式还是对数据进行彻底清洗。

DataLoader里面并没有太多的魔法方法，它封装了Python的标准库multiprocessing，使其能够实现多进程加速。在此提几点关于Dataset和DataLoader使用方面的建议：

高负载的操作放在__getitem__中，如加载图片等。
dataset中应尽量只包含只读对象，避免修改任何可变对象，利用多线程进行操作。
第一点是因为多进程会并行的调用__getitem__函数，将负载高的放在__getitem__函数中能够实现并行加速。 第二点是因为dataloader使用多进程加载，
如果在Dataset实现中使用了可变对象，可能会有意想不到的冲突。
在多线程/多进程中，修改一个可变对象，需要加锁，但是dataloader的设计使得其很难加锁（在实际使用中也应尽量避免锁的存在），因此最好避免在dataset中修改可变对象。
例如下面就是一个不好的例子，在多进程处理中self.num可能与预期不符，这种问题不会报错，因此难以发现。如果一定要修改可变对象，建议使用Python标准库Queue中的相关数据结构。
'''


'''
使用Python multiprocessing库的另一个问题是，在使用多进程时，如果主程序异常终止（比如用Ctrl+C强行退出），相应的数据加载进程可能无法正常退出。这时你可能会发现程序已经退出了，但GPU显存和内存依旧被占用着，或通过top、ps aux依旧能够看到已经退出的程序，这时就需要手动强行杀掉进程。建议使用如下命令：

ps x | grep <cmdline> | awk '{print $1}' | xargs kill
ps x：获取当前用户的所有进程
grep <cmdline>：找到已经停止的PyTorch程序的进程，例如你是通过python train.py启动的，那你就需要写grep 'python train.py'
awk '{print $1}'：获取进程的pid
xargs kill：杀掉进程，根据需要可能要写成xargs kill -9强制杀掉进程
在执行这句命令之前，建议先打印确认一下是否会误杀其它进程

ps x | grep <cmdline> | ps x
PyTorch中还单独提供了一个sampler模块，用来对数据进行采样。
常用的有随机采样器：RandomSampler，当dataloader的shuffle参数为True时，系统会自动调用这个采样器，实现打乱数据。默认的是采用SequentialSampler，
它会按顺序一个一个进行采样。这里介绍另外一个很有用的采样方法： WeightedRandomSampler，它会根据每个样本的权重选取数据，在样本比例不均衡的问题中，可用它来进行重采样。


构建WeightedRandomSampler时需提供两个参数：每个样本的权重weights、共选取的样本总数num_samples，以及一个可选参数replacement。
权重越大的样本被选中的概率越大，待选取的样本数目一般小于全部的样本数目。replacement用于指定是否可以重复选取某一个样本，默认为True，即允许在一个epoch中重复采样某一个数据。
如果设为False，则当某一类的样本被全部选取完，但其样本数目仍未达到num_samples时，sampler将不会再从该类中选择数据，此时可能导致weights参数失效。下面举例说明。
'''

dataset = DogCat('dogcat/', transforms=transform)

# 狗的图片被取出的概率是猫的概率的两倍
# 两类图片被取出的概率与weights的绝对大小无关，只和比值有关
weights = [2 if label == 1 else 1 for data, label in dataset]
print(weights)


from torch.utils.data.sampler import  WeightedRandomSampler
sampler = WeightedRandomSampler(weights,\
                                num_samples=9,\
                                replacement=True)
dataloader = DataLoader(dataset,
                        batch_size=3,
                        sampler=sampler)
for datas, labels in dataloader:
    print(labels.tolist())

'''
可见猫狗样本比例约为1:2，另外一共只有8个样本，但是却返回了9个，说明肯定有被重复返回的，这就是replacement参数的作用，下面将replacement设为False试试
'''

sampler = WeightedRandomSampler(weights, 8, replacement=False)
dataloader = DataLoader(dataset, batch_size=4, sampler=sampler)
for datas, labels in dataloader:
    print(labels.tolist())

'''
在这种情况下，num_samples等于dataset的样本总数，为了不重复选取，sampler会将每个样本都返回，这样就失去weight参数的意义了。

从上面的例子可见sampler在样本采样中的作用：如果指定了sampler，shuffle将不再生效，并且sampler.num_samples会覆盖dataset的实际大小，即一个epoch返回的图片总数取决于sampler.num_samples。
'''

'''
5.2 计算机视觉工具包：torchvision
计算机视觉是深度学习中最重要的一类应用，为了方便研究者使用，PyTorch团队专门开发了一个视觉工具包torchvion，这个包独立于PyTorch，需通过pip instal torchvision安装。

在之前的例子中我们已经见识到了它的部分功能，这里再做一个系统性的介绍。torchvision主要包含三部分：

models：提供深度学习中各种经典网络的网络结构以及预训练好的模型，包括AlexNet、VGG系列、ResNet系列、Inception系列等。
datasets： 提供常用的数据集加载，设计上都是继承torhc.utils.data.Dataset，主要包括MNIST、CIFAR10/100、ImageNet、COCO等。
transforms：提供常用的数据预处理操作，主要包括对Tensor以及PIL Image对象的操作。
'''

from torchvision import models
from torch import nn
# 加载预训练好的模型，如果不存在会进行下载
# 预训练好的模型保存在 ~/.torch/models/下面
resnet_34 = models.squeezenet1_0(pretrained=True)

# 修改最后的全连接层为10分类问题（默认是ImageNet上的1000分类）
resnet_34.fc=nn.Linear(512, 10)

'''
5.3 可视化工具
在训练神经网络时，我们希望能更直观地了解训练情况，包括损失曲线、输入图片、输出图片、卷积核的参数分布等信息。
这些信息能帮助我们更好地监督网络的训练过程，并为参数优化提供方向和依据。
最简单的办法就是打印输出，但其只能打印数值信息，不够直观，同时无法查看分布、图片、声音等。
在本节，我们将介绍两个深度学习中常用的可视化工具：Tensorboard和Visdom。

5.3.1 Tensorboard
Tensorboard最初是作为TensorFlow的可视化工具迅速流行开来。作为和TensorFlow深度集成的工具，Tensorboard能够展现你的TensorFlow网络计算图，绘制图像生成的定量指标图以及附加数据。
但同时Tensorboard也是一个相对独立的工具，只要用户保存的数据遵循相应的格式，tensorboard就能读取这些数据并进行可视化。
这里我们将主要介绍如何在PyTorch中使用tensorboardX^1进行训练损失的可视化。
 TensorboardX是将Tensorboard的功能抽取出来，使得非TensorFlow用户也能使用它进行可视化，几乎支持原生TensorBoard的全部功能。
 
 tensorboard的安装主要分为以下两步：

安装TensorFlow：如果电脑中已经安装完TensorFlow可以跳过这一步，如果电脑中尚未安装，建议安装CPU-Only的版本，具体安装教程参见TensorFlow官网^1，
或使用pip直接安装，推荐使用清华的软件源^2。
安装tensorboard: pip install tensorboard
安装tensorboardX：可通过pip install tensorboardX命令直接安装。
tensorboardX的使用非常简单。首先用如下命令启动tensorboard：

tensorboard --logdir <your/running/dir> --port <your_bind_port>
下面举例说明tensorboardX的使用。
'''

from tensorboardX import SummaryWriter

# 构建logger对象，logdir用来指定log文件的保存路径
# flush_secs用来指定刷新同步间隔

logger = SummaryWriter(log_dir='experiment_cnn', flush_secs=1)

for ii in range(100):
    print('data/loss', 10-ii**0.5)
    print('data/accuracy', ii**0.5/10)

# for ii in range(100):
#     logger.add_scalar('data/loss', 10-ii**0.5)
#     logger.add_scalar('data/accuracy', ii**0.5/10)

'''
5.4 使用GPU加速：cuda
这部分内容在前面介绍Tensor、Module时大都提到过，这里将做一个总结，并深入介绍相关应用。

在PyTorch中以下数据结构分为CPU和GPU两个版本：

Tensor
nn.Module（包括常用的layer、loss function，以及容器Sequential等）
它们都带有一个.cuda方法，调用此方法即可将其转为对应的GPU对象。注意，tensor.cuda会返回一个新对象，这个新对象的数据已转移至GPU，而之前的tensor还在原来的设备上（CPU）。而module.cuda则会将所有的数据都迁移至GPU，并返回自己。所以module = module.cuda()和module.cuda()所起的作用一致。

nn.Module在GPU与CPU之间的转换，本质上还是利用了Tensor在GPU和CPU之间的转换。nn.Module的cuda方法是将nn.Module下的所有parameter（包括子module的parameter）都转移至GPU，而Parameter本质上也是tensor(Tensor的子类)。

下面将举例说明，这部分代码需要你具有两块GPU设备。

P.S. 为什么将数据转移至GPU的方法叫做.cuda而不是.gpu，就像将数据转移至CPU调用的方法是.cpu？这是因为GPU的编程接口采用CUDA，而目前并不是所有的GPU都支持CUDA，只有部分Nvidia的GPU才支持。PyTorch未来可能会支持AMD的GPU，而AMD GPU的编程接口采用OpenCL，因此PyTorch还预留着.cl方法，用于以后支持AMD等的GPU。


'''