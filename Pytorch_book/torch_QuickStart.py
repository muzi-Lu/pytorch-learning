from __future__ import print_function
 # python2.x and python3.x use same print function
import torch

print(torch.__version__)

# 构建一个 5x3 的Tensor矩阵， 但只是分配了空间，未进行初始化
x = torch.Tensor(5, 3)
# 进行赋值
x = torch.Tensor([[1, 2], [3, 4]])
print(x)

# 使用[0, 1]均匀分布随机初始化二维数组
x = torch.randn(5, 6)
print(x)
# 矩阵大小
print(x.size()) # 会给出torch.Size的类型
print(x.type) # 类型：Tensor object
# 输出行列大小
print('行大小：', x.size(0))
print('列大小：', x.size(1))

# 另外一种说法
print('行大小：', x.size()[0])
print('列大小：', x.size()[1])
# print('高大小：', x.size(2)) # IndexError: Dimension out of range (expected to be in range of [-2, 1], but got 2)


# 使用[0, 1]均匀分布随机初始化3维数组
y = torch.randn(5, 6, 3)
print(x)
# 输出行列高大小
print('行大小：', y.size(0))
print('列大小：', y.size(1))
print('列大小：', y.size(2))

# 张量加法
x = torch.Tensor([
    [1, 2],
    [3, 4]
])

y = torch.Tensor([
    [1, 2],
    [3, 4]
])
# 第一种
print(x+y)
# 第二种
print(torch.add(x, y))
# 第三种
result = torch.Tensor(2, 2)
torch.add(x, y, out=result)
print(result)

# 另外的加法方式
print('最初的y')
print(y)

print('第一种加法, y的结果')
print(y.add(x)) # 普通加法， 不改变y的内容
print(y)

print('第二种加法, y的结果')
print(y.add_(x))# inplace加法， y变了
print(y)

'''
Note:函数名后面带下划线_的函数会改变Tensor本身
'''

'''
Tensor还支持很多操作，包括数学运算，线性代数，选择，切片等等，其接口设计与Numpy非常相近

Tensor和Numpy的数组之间的互操作非常容易且快速，对于与Tensor不支持的操作，可以先用Numpy数组处理，之后再转回Tensor
'''

a = torch.ones(6)
print(a)
# Tensor --> Numpy
b = a.numpy()
print(b)
# 这么打印类型
print('数组类型：', type(b))
print('数组数据类型:', b.dtype)

import numpy as np
a = np.ones(6)
# Numpy --> Tensor
b = torch.from_numpy(a)
print(a)
print(b)

'''
Tensor 和 Numpy 对象之间共享内存，所以他们之间转换的很快，而且不怎么消耗资源，但是也意味着，如果一个改变，另外一个也要改变
'''
# 这里a也变就很奇怪
b.add_(1)
print(a)
print(b)

'''
如果想获得元素的一个值，可以使用scalar.item。直接使用tensor[idx]得到的还是一个tensor，一个 0-dim 的tensor, 一般称为scalar
'''
scalar = b[0]
print(scalar) # 拿到的还是一个tensor
print(scalar.size()) # 0-dim
print(scalar.item()) # 使用scalar.item()能从中取出python对象的数值

# 注意和scalar的区别
tensor = torch.Tensor([2])
print('Tensor:', tensor)
print('Scalar:', scalar)

print('Tensor Size:', tensor.size()) # 1-dim
print('scalar Size:', scalar.size()) # 0-dim

print('Tensor Item:', tensor.item()) # 1-dim
print('scalar Item:', scalar.item()) # 0-dim

'''
此外在pytorch中还有一个和np.array很类似的结构：torch.tensor,使用方法非常接近
'''

tensor = torch.Tensor([3, 4])

old_tensor = tensor
new_tensor = old_tensor.clone()
new_tensor[0] = 1111
print(old_tensor, new_tensor)

'''
需要注意的是, torch.Tensor以及tensor.clone()总是会进行数据拷贝，新的tensor和原本的数据不再共享内存，所以想要共享内存的话，建议使用
torch.from_numpy或者tensor.detach()来新建一个tensor，二者共享内存
'''

new_tensor = old_tensor.detach()
new_tensor[0] = 1111
print(old_tensor, new_tensor)

'''
Tensor可以通过.cuda的方法转为GPU的Tensor,从而享受GPU带来的加速运算
'''
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
x = x.to(device)
y = y.to(device)
print('x in cuda', x)
print('y in cuda', y)

z = x+y
print('z in cuda', z)

'''
此外，还可以使用tensor.cuda()的方法将tensor拷贝到GPU上，但是这种方法不太推荐
'''

'''
autograd:自动微分
深度学习的算法本质上是通过反向传播求导数，而PyTorch的autograd模块实现了功能。在Tensor上的所有操作，autograd都能为其自动提供微分
，避免了手动计算导数的复杂过程

从0.4开始， Variable正式合并入Tensor, Variable本来实现的自动微分功能， Tensor就能支持。
要想使用Tensor的autograd功能，只要tensor.requries_grad = True
'''

x = torch.ones(2, 2, requires_grad=True)
# 也可以这么写
x = torch.ones(2, 2)
x.requires_grad = True

print('x:', x)

y = x.sum()
print('y:{0}  y.grad_fn:{1}'.format(y, y.grad_fn))

# 反向传播 计算梯度
y.backward()
print(x.grad) # 为什么是这个？

x = torch.Tensor([
    [2, 3],
    [4, 5]
])
x.requires_grad = True
y = x.sum()
print('y:{0}  y.grad_fn:{1}'.format(y, y.grad_fn))
y.backward()
print(x.grad) # 为什么是这个？ # 知道怎么算了 y = x11+x12+x21+x22

'''
神经网络
Autograd 实现了反向传播功能，但是直接写深度学习的代码在很多情况下还是很复杂的，torch.nn是专门为神经网络设计的模块化接口。
nn构建在Autograd之上，可用于定义和运行神经网络.nn.Module是nn中最重要的类，可以把他看成一个网络的封装，包含网络各层定义以及forward方法，
调用forward(input)方法，可返回前向传播的结构。
下面以最早的卷积神经网络：LeNet
'''

import torch.nn as nn
import torch.nn.functional as F

'''
定义网络
定义网络时，需要继承nn.Module,并实现它的forward方法，把网络中具有可学习的参数的层放在构造函数__init__中。如果某一层(如Relu)不具有可学习的参数，
则既可以放在构造函数中，也可以不放，但建议不放在里面，而是用forward中使用nn.functional代替
'''
class Net(nn.Module):
    def __init__(self):
        # nn.Module子类的函数必须在构造函数中执行父类的构造函数
        # 下式等价于nn.Module.__init__(self)
        super(Net, self).__init__()

        # 卷积层‘1’表示输入图片为单通道， ‘6’表示输出通道数， ‘5’表示卷积核为5
        self.conv1 = nn.Conv2d(1, 6, 5)
        # Conv Layer
        self.conv2 = nn.Conv2d(6, 16, 5)
        # 全连接层, y=Wx+b
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # 卷积 --> 激活 --> 池化
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x), 2))
        # reshape ,-1表示自适应
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()
print(net)

