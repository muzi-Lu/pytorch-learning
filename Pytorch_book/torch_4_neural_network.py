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

img = Image.open('img.png')
print(img.size)
# img.show('lena')

input = to_tensor(img).unsqueeze(0)
print(input.shape)

# 锐化卷积核
kernel = torch.ones(3, 3) / -9
kernel[1][1] = 1
print(kernel)
conv = nn.Conv2d(4, 1, (3, 3), 1, bias=False)
conv.weight.data = kernel.view(1, 1, 3, 3)

out = conv(input[:, 1:2, :, :]) # 注意学，注意看
out_img = to_pil(out.data.squeeze(0))
# out_img.show()

'''
除了上述的使用，图像的卷积操作还有各种变体，具体可以参照此处动图[^2]介绍。 [^2]: https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
'''

# 池化层可以看作是一种特殊的卷积层，用来下采样。但池化层没有可学习参数，其weight是固定的。
pool = nn.AvgPool2d(2, 2)
print(pool.parameters(), pool.named_parameters())# 这里打印的是对象
print(list(pool.parameters()), list(pool.named_parameters()))# 这里打印的是参数

out = pool(input)
pool_img = to_pil(out.data.squeeze(0))
# pool_img.show()

'''
除了卷积层，池化层，深度学习还常用到以下几层：

Linear:全连接层
BatchNorm:批规范化层，分为1D，2D，3D。除了标准的BatchNorm之外，还有风格迁移中常用到的InstanceNorm层
Dropout:dropout层，用于防止过拟合，同样分为1D，2D，3D
'''

# 输入 batch_size = 2, dim = 3
input = torch.ones(2, 3)
input = torch.rand(2, 3)
print(input)
linear = nn.Linear(3, 4)
print(list(linear.parameters()))
h = linear(input)
print(h)

'''
所以他这里是这么算的
input x (linear)T + b
'''


print('--------------------------------------------')
# wocao 这个地方不明白啊
# 4 channel,初始化标准差为4， 均值为0 这个地方是怎么算的
bn = nn.BatchNorm1d(4)
bn.weight.data = torch.ones(4) * 4
bn.bias.data = torch.zeros(4)

bn_out = bn(h)
print('bn_out:', bn_out)
# 注意输出的均值和方差
# 方差是标准差的平方，计算无偏方差分母会减1
# 使用unbiased = Flase, 分母不减一
print('Mean:', bn_out.mean(0), 'Var:', bn_out.var(0, unbiased = False))

print('--------------------------------------------')

dropout = nn.Dropout(0.5)
o = dropout(bn_out)
print(o)

'''
以上很多例子中都对module的属性直接操作，其大多数是可学习参数，一般会随着学习的进行而不断改变。实际使用中除非需要使用特殊的初始化，应尽量不要直接修改这些参数。
'''

'''
4.1.2 激活函数
PyTorch实现了常见的激活函数，其具体的接口信息可参见官方文档[^3]，这些激活函数可作为独立的layer使用。这里将介绍最常用的激活函数ReLU，其数学表达式为：
[^3]: http://pytorch.org/docs/nn.html#non-linear-activations
'''

print('--------------------------------------------')
relu = nn.ReLU(inplace=True)
input = torch.rand(4, 4)
print(input)
output = relu(input)
print(output)

'''
在以上的例子中，基本上都是将每一层的输出直接作为下一层的输入，这种网络称为前馈传播网络（feedforward neural network）。
对于此类网络如果每次都写复杂的forward函数会有些麻烦，在此就有两种简化方式，ModuleList和Sequential。
其中Sequential是一个特殊的module，它包含几个子Module，前向传播时会将输入一层接一层的传递下去。
ModuleList也是一个特殊的module，可以包含几个子module，可以像用list一样使用它，但不能直接把输入传给ModuleList。下面举例说明。
'''

# Sequential的三种写法
net1 = nn.Sequential()
net1.add_module('conv', nn.Conv2d(3, 3, 3))
net1.add_module('batchnorm', nn.BatchNorm2d(3))
net1.add_module('activate_layer', nn.ReLU())

net2 = nn.Sequential(
    nn.Conv2d(3, 3, 3),
    nn.BatchNorm2d(3),
    nn.ReLU()
)

from collections import OrderedDict
net3 = nn.Sequential(OrderedDict([
    ('conv1', nn.Conv2d(3, 3, 3)),
    ('bn1', nn.BatchNorm2d(3)),
    ('relu1', nn.ReLU())
]))

print(net1)
print(net2)
print(net3)

print(net1.conv, net2[0], net3.conv1)

print('--------------------------------------------')
# 为什么输出不一样呢，是kernel不一样吧
input = torch.rand(1, 3, 4, 4)
output1 = net1(input)
output2 = net2(input)
output3 = net3(input)
output4 = net3.relu1(net1.batchnorm(net1.conv(input)))

print('output1:', output1, '/n'+'output2:', output2, '/n','output3:', output3, '/n','output4:', output4, '/n')

print('--------------------------------------------')
modellist = nn.ModuleList([nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2)])
input = torch.randn(1, 3)
for model in modellist:
    input = model(input)
output = input
print(output)

# 但是这样会报错，因为modellist没有实现forward方法
# output = modellist(input) 注释掉了

'''
看到这里，读者可能会问，为何不直接使用Python中自带的list，而非要多此一举呢？这是因为ModuleList是Module的子类，当在Module中使用它的时候，就能自动识别为子module。

下面举例说明。
'''

class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.list = [nn.Linear(3, 4), nn.ReLU()]
        self.module_list = nn.ModuleList([nn.Conv2d(3, 3, 3), nn.ReLU()])
    def forward(self):
        pass
model = MyModule()
print(model)

for name, param in model.named_parameters():
    print(name, param.size())

'''
可见，list中的子module并不能被主module所识别，而ModuleList中的子module能够被主module所识别。这意味着如果用list保存子module，将无法调整其参数，因其未加入到主module的参数中。

除ModuleList之外还有ParameterList，其是一个可以包含多个parameter的类list对象。在实际应用中，使用方式与ModuleList类似。
如果在构造函数__init__中用到list、tuple、dict等对象时，一定要思考是否应该用ModuleList或ParameterList代替。
'''


'''

4.1.3 循环神经网络层(RNN)
近些年随着深度学习和自然语言处理的结合加深，RNN的使用也越来越多，关于RNN的基础知识，推荐阅读colah的文章[^4]入门。
PyTorch中实现了如今最常用的三种RNN：RNN（vanilla RNN）、LSTM和GRU。此外还有对应的三种RNNCell。

RNN和RNNCell层的区别在于前者一次能够处理整个序列，而后者一次只处理序列中一个时间点的数据，前者封装更完备更易于使用，后者更具灵活性。
实际上RNN层的一种后端实现方式就是调用RNNCell来实现的。 [^4]: http://colah.github.io/posts/2015-08-Understanding-LSTMs/
'''
print('--------------------------------------------')
torch.manual_seed(1000)
# 输入：batchsize = 3, seq_length = 2, 序列每个元素4维
input = torch.rand(2, 3, 4)
# lstm 输入向量4维， 隐藏元3,1层
lstm = nn.LSTM(4, 3, 1)
print(lstm)
# c初始状态：1层，batch_size=3, 3个隐藏元
h0 = torch.randn(1, 3, 3)
c0 = torch.randn(1, 3, 3)
out, hn = lstm(input, (h0, c0))
print(out)

print('--------------------------------------------')
torch.manual_seed(1000)
input = torch.randn(2, 3, 4)
# 一个LSTMCell对应只能是一层
lstm = nn.LSTMCell(4, 3)
hx = torch.randn(3, 3)
cx = torch.rand(3, 3)
out = []
for i_ in input:
    hx, cx = lstm(i_, (hx, cx))
    out.append(hx)
torch.stack(out)

print('--------------------------------------------')
# 词向量在自然语言中应用十分普及，PyTorch同样提供了Embedding层。
# 有4个词，每个词用5维的向量表示
# embedding = nn.Embedding(4, 5)
# # 可以用预训练好的词向量初始化embedding
# embedding.weight.data = torch.arange(0,20).view(4,5)
#
# input = torch.arange(3, 0, -1).long()
# output = embedding(input)
# RuntimeError: isDifferentiableType(variable.scalar_type()) INTERNAL ASSERT FAILED at "/pytorch/torch/csrc/autograd/functions/utils.h":64, please report a bug to PyTorch.


'''
4.1.4 损失函数
在深度学习中要用到各种各样的损失函数（loss function），这些损失函数可看作是一种特殊的layer，PyTorch也将这些损失函数实现为nn.Module的子类。
然而在实际使用中通常将这些loss function专门提取出来，和主模型互相独立。
详细的loss使用请参照文档[^5]，这里以分类中最常用的交叉熵损失CrossEntropyloss为例说明。 [^5]: http://pytorch.org/docs/nn.html#loss-functions
'''

# batch_size = 3, 计算对应每个类别的分数（只有两类）
score = torch.randn(3, 2)
print(score)
# 三个样本分别属于1，0，1类，label必须是LongTensor
label = torch.Tensor([1, 0, 1]).long()

# loss与普通的layer无差异
criterion = nn.CrossEntropyLoss()
loss = criterion(score, label)
print(loss)

'''
4.2 优化器
PyTorch将深度学习中常用的优化方法全部封装在torch.optim中，其设计十分灵活，能够很方便的扩展成自定义的优化方法。

所有的优化方法都是继承基类optim.Optimizer，并实现了自己的优化步骤。下面就以最基本的优化方法——随机梯度下降法（SGD）举例说明。这里需重点掌握：

优化方法的基本使用方法
如何对模型的不同部分设置不同的学习率
如何调整学习率

'''

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 6, 5, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 400)
        x = self.classifier(x)
        return x

net = Net()

from torch import optim
optimizer = optim.SGD(net.parameters(), lr=0.001)
print(optimizer)
optimizer.zero_grad()

input = torch.randn(1, 3, 32, 32)
output = net(input) # 前向传播
output.backward(output) # 假的反向传播

optimizer.step() # 执行优化

# 为不同子网络设置不同的学习率，在finetune中经常用到
# 如果对某个参数不指定学习率，就使用最外层的默认学习率
optimizer =optim.SGD([
                {'params': net.features.parameters()}, # 学习率为1e-5
                {'params': net.classifier.parameters(), 'lr': 1e-2}
            ], lr=1e-5)
print(optimizer)

'''
对于如何调整学习率，主要有两种做法。一种是修改optimizer.param_groups中对应的学习率，另一种是更简单也是较为推荐的做法——新建优化器，由于optimizer十分轻量级，构建开销很小，故而可以构建新的optimizer。
但是后者对于使用动量的优化器（如Adam），会丢失动量等状态信息，可能会造成损失函数的收敛出现震荡等情况。
'''


# 只为两个全连接层设置较大的学习率，其余层的学习率较小
special_layers = nn.ModuleList([net.classifier[0], net.classifier[3]])
special_layers_params = list(map(id, special_layers.parameters()))
base_params = filter(lambda p: id(p) not in special_layers_params,
                     net.parameters())

optimizer = torch.optim.SGD([
            {'params': base_params},
            {'params': special_layers.parameters(), 'lr': 0.01}
        ], lr=0.001)
print(optimizer)

# 方法1: 调整学习率，新建一个optimizer
old_lr = 0.1
optimizer1 =optim.SGD([
                {'params': net.features.parameters()},
                {'params': net.classifier.parameters(), 'lr': old_lr*0.1}
            ], lr=1e-5)
print(optimizer1)

# 方法2: 调整学习率, 手动decay, 保存动量
for param_group in optimizer.param_groups:
    param_group['lr'] *= 0.1 # 学习率为之前的0.1倍
print(optimizer)

'''
4.3 nn.functional
梅东
nn中还有一个很常用的模块：nn.functional，nn中的大多数layer，在functional中都有一个与之相对应的函数。
nn.functional中的函数和nn.Module的主要区别在于，用nn.Module实现的layers是一个特殊的类，都是由class layer(nn.Module)定义，会自动提取可学习的参数。
而nn.functional中的函数更像是纯函数，由def function(input)定义。下面举例说明functional的使用，并指出二者的不同之处。
'''

input = torch.randn(2, 3)
model = nn.Linear(3, 4)
output1 = model(input)
output2 = nn.functional.linear(input, model.weight, model.bias)
print(output1 == output2)


b = nn.functional.relu(input)
b2 = nn.ReLU()(input)
print(b == b2)

'''
此时读者可能会问，应该什么时候使用nn.Module，什么时候使用nn.functional呢？
答案很简单，如果模型有可学习的参数，最好用nn.Module，否则既可以使用nn.functional也可以使用nn.Module，二者在性能上没有太大差异，具体的使用取决于个人的喜好。
如激活函数（ReLU、sigmoid、tanh），池化（MaxPool）等层由于没有可学习参数，则可以使用对应的functional函数代替，而对于卷积、全连接等具有可学习参数的网络建议使用nn.Module。
下面举例说明，如何在模型中搭配使用nn.Module和nn.functional。另外虽然dropout操作也没有可学习操作，但建议还是使用nn.Dropout而不是nn.functional.dropout，
因为dropout在训练和测试两个阶段的行为有所差别，使用nn.Module对象能够通过model.eval操作加以区分。
'''

from torch.nn import functional as F
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.pool(F.relu(self.conv1(x)), 2)
        x = F.pool(F.relu(self.conv1(x)), 2)
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

'''
对于不具备可学习参数的层（激活层、池化层等），将它们用函数代替，这样则可以不用放置在构造函数__init__中。
对于有可学习参数的模块，也可以用functional来代替，只不过实现起来较为繁琐，需要手动定义参数parameter，如前面实现自定义的全连接层，就可将weight和bias两个参数单独拿出来，
在构造函数中初始化为parameter。

'''


'''
在深度学习中参数的初始化十分重要，良好的初始化能让模型更快收敛，并达到更高水平，而糟糕的初始化则可能使得模型迅速瘫痪。
PyTorch中nn.Module的模块参数都采取了较为合理的初始化策略，因此一般不用我们考虑，当然我们也可以用自定义初始化去代替系统的默认初始化。
而当我们在使用Parameter时，自定义初始化则尤为重要，因t.Tensor()返回的是内存中的随机数，很可能会有极大值，这在实际训练网络中会造成溢出或者梯度消失。 # 自己的程序会不会是这个问题

PyTorch中nn.init模块就是专门为初始化而设计，如果某种初始化策略nn.init不提供，用户也可以自己直接初始化。
'''

# 利用nn.init初始化
from torch.nn import init
linear = nn.Linear(3, 4)

torch.manual_seed(1)
# 等价于linear.weight.data.normal_(0, std)
init.xavier_normal_(linear.weight)
print(linear.weight)

# 直接初始化
import math
torch.manual_seed(1)

# xavier初始化的计算公式
std = math.sqrt(2)/math.sqrt(7.)
linear.weight.data.normal_(0,std)
print(linear.weight)

'''
如果想要更深入地理解nn.Module，究其原理是很有必要的。首先来看看nn.Module基类的构造函数：

def __init__(self):
    self._parameters = OrderedDict()
    self._modules = OrderedDict()
    self._buffers = OrderedDict()
    self._backward_hooks = OrderedDict()
    self._forward_hooks = OrderedDict()
    self.training = True
其中每个属性的解释如下：

_parameters：字典，保存用户直接设置的parameter，self.param1 = nn.Parameter(t.randn(3, 3))会被检测到，在字典中加入一个key为'param'，value为对应parameter的item。
而self.submodule = nn.Linear(3, 4)中的parameter则不会存于此。

_modules：子module，通过self.submodel = nn.Linear(3, 4)指定的子module会保存于此。
_buffers：缓存。如batchnorm使用momentum机制，每次前向传播需用到上一次前向传播的结果。
_backward_hooks与_forward_hooks：钩子技术，用来提取中间变量，类似variable的hook。
training：BatchNorm与Dropout层在训练阶段和测试阶段中采取的策略不同，通过判断training值来决定前向传播策略。
上述几个属性中，_parameters、_modules和_buffers这三个字典中的键值，都可以通过self.key方式获得，效果等价于self._parameters['key'].

'''

print('--------------------------------------------')
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.param1 = nn.Parameter(torch.rand(3, 3))
        self.submodel1 = nn.Linear(3, 4)

    def forward(self, input):
        x = self.param1.mm(input)
        x = self.submodel1(x)
        return x

net = Net()
print(net)

print(net.param1)
print(net.submodel1)

for name, param in net.named_parameters():
    print(name, param.size())

print(net._modules)
print(net._parameters)

'''
nn.Module在实际使用中可能层层嵌套，一个module包含若干个子module，每一个子module又包含了更多的子module。
为方便用户访问各个子module，nn.Module实现了很多方法，如函数children可以查看直接子module，函数module可以查看所有的子module（包括当前module）。
与之相对应的还有函数named_childen和named_modules，其能够在返回module列表的同时返回它们的名字。
'''

input = torch.arange(0, 12).view(3, 4).float()
model = nn.Dropout()
output = model(input)
print(output)


model.training  = False
# 在测试阶段，dropout什么都不做
output = model(input)
print(output)

'''
对于batchnorm、dropout、instancenorm等在训练和测试阶段行为差距巨大的层，如果在测试时不将其training值设为True，则可能会有很大影响，这在实际使用中要千万注意。
虽然可通过直接设置training属性，来将子module设为train和eval模式，但这种方式较为繁琐，因如果一个模型具有多个dropout层，就需要为每个dropout层指定training属性。
更为推荐的做法是调用model.train()函数，它会将当前module及其子module中的所有training属性都设为True，相应的，model.eval()函数会把training属性都设为False。
'''

'''
Important:
nn.Module对象在构造函数中的行为看起来有些怪异，如果想要真正掌握其原理，就需要看两个魔法方法__getattr__和__setattr__。在Python中有两个常用的buildin方法getattr和setattr，getattr(obj, 'attr1')等价于obj.attr，如果getattr函数无法找到所需属性，Python会转而调用obj.__getattr__('attr1')方法，即getattr函数无法找到的交给__getattr__函数处理，没有实现__getattr__或者__getattr__也无法处理的就会raise AttributeError。setattr(obj, 'name', value)等价于obj.name=value，如果obj对象实现了__setattr__方法，setattr会直接调用obj.__setattr__('name', value)，否则调用buildin方法。总结一下：

result = obj.name会调用buildin函数getattr(obj, 'name')，如果该属性找不到，会调用obj.__getattr__('name')
obj.name = value会调用buildin函数setattr(obj, 'name', value)，如果obj对象实现了__setattr__方法，setattr会直接调用obj.__setattr__('name', value')
nn.Module实现了自定义的__setattr__函数，当执行module.name=value时，会在__setattr__中判断value是否为Parameter或nn.Module对象，如果是则将这些对象加到_parameters和_modules两个字典中，而如果是其它类型的对象，如Variable、list、dict等，则调用默认的操作，将这个值保存在__dict__中

'''

'''
4.6 nn和autograd的关系
nn.Module利用的也是autograd技术，其主要工作是实现前向传播。在forward函数中，nn.Module对输入的tensor进行的各种操作，本质上都是用到了autograd技术。
这里需要对比autograd.Function和nn.Module之间的区别：

autograd.Function利用了Tensor对autograd技术的扩展，为autograd实现了新的运算op，不仅要实现前向传播还要手动实现反向传播
nn.Module利用了autograd技术，对nn的功能进行扩展，实现了深度学习中更多的层。只需实现前向传播功能，autograd即会自动实现反向传播
nn.functional是一些autograd操作的集合，是经过封装的函数

作为两大类扩充PyTorch接口的方法，我们在实际使用中应该如何选择呢？如果某一个操作，在autograd中尚未支持，那么只能实现Function接口对应的前向传播和反向传播。
如果某些时候利用autograd接口比较复杂，则可以利用Function将多个操作聚合，实现优化，正如第三章所实现的Sigmoid一样，比直接利用autograd低级别的操作要快。
而如果只是想在深度学习中增加某一层，使用nn.Module进行封装则更为简单高效。
'''