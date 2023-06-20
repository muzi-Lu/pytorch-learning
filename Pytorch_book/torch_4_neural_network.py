import torch
import torch.nn as nn

class Linear(nn.Module):
    def __init__(self, in_features, out_features):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.w = nn.Parameter(torch.randn(self.in_features, self.out_features))
        self.b = nn.Parameter(torch.randn(self.out_features))

    def forward(self, x):
        x = x.mm(self.w) # x @(self.w)
        return x + self.b.expand_as(x)

layer = Linear(4, 3)
input = torch.rand(2, 4)
output = layer(input)
print(output)

for name, parameter in layer.named_parameters():
    print(name, parameter)

# 可见，全连接层的实现非常简单，代码不超过20

'''
1.自定义的Linear必须继承nn.Module,并且其构造函数需要调用nn.Module的构造函数，即
super(Linear, self).__init__(),或者nn.Module.__init__(self)
2.在构造函数__init__的必须自己定义可学习的参数，并封装成Parameter，
在上面的例子中，把w和b封装成了parameter
parameter是一种特殊的tensor，但其默认需要求导，可以通过nn.Parameter??，直接查看Parameter类的源码
3.forward函数实现前向传播过程，其输入可以是一个或者多个tensor
4.无需写反向传播函数，nn.Module能够利用autograd自动实现反向传播，这点比Function简单
5.使用时，直观上可将layer看成数学概念上的函数，调用layer(input)就可以得到input对应的结果
等价于layers.__call__(input),在__call__函数中，主要调用的是layer.forward(x),另外还对钩子进行了一些处理，
所以实际使用中应尽量使用layer(x),而不是layer.forward(x),钩子下面介绍
6.Module中的科学系参数可以通过name_parameters()或者parameters()返回迭代器，前者会给每个parameter附上名字
'''

# 多层感知机
class Perception(nn.Module):
    def __init__(self, in_feature, hidden_feature, out_feature):
        nn.Module.__init__(self)
        self.layer1 = Linear(in_feature, hidden_feature)
        self.layer2 = Linear(hidden_feature, out_feature)

    def forward(self, x):
        x = self.layer1(x)
        x = torch.sigmoid(x)
        x = self.layer2(x)
        return x

print("------------------------------------------------------")
perception = Perception(3, 4, 1)
for name, param in perception.named_parameters():
    print(name, param, param.size())

'''
4.1 常用神经网络层
4.1.1 图像相关层
图像相关曾主要包括卷积层(Conv),池化层（Pool）等，这些层在实际使用中可以分为1d，2d
,3d，池化方式又可以分为平均值池化(AvgPool),最大值池化(MaxPool),自适应池化(AdaptiveAvgPool)。
而除了常用的前向卷积外，还有逆卷积。
'''

print("------------------------------------------------------")
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage
to_tensor = ToTensor()
to_pil = ToPILImage()

img = Image.open('run_25_05_23_10_28_loss_fig.png')
print(img.size)
# img.show('loss')

input = to_tensor(img).unsqueeze(0)
input = to_tensor(img).squeeze(1)
print(input.shape)

# 锐化卷积核
kernel = torch.ones(3, 3) / -9
kernel[1][1] = 1
print(kernel)
conv = nn.Conv2d(1, 1, (3, 3), 1, bias=False)
conv.weight.data = kernel.view(1, 1, 3, 3)

out = conv(input)
to_pil(out.data.squeeze(0))
