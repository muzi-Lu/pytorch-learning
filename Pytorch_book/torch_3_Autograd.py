'''

用Tensor训练网络很方便，但从上一小节最后的线性回归例子来看，反向传播过程需要手动实现。这对于像线性回归等较为简单的模型来说，还可以应付，但实际使用中经常出现非常复杂的网络结构，此时如果手动实现反向传播，不仅费时费力，而且容易出错，难以检查。
torch.autograd就是为方便用户使用，而专门开发的一套自动求导引擎，它能够根据输入和前向传播过程自动构建计算图，并执行反向传播。

计算图(Computation Graph)是现代深度学习框架如PyTorch和TensorFlow等的核心，其为高效自动求导算法——反向传播(Back Propogation)提供了理论支持，
了解计算图在实际写程序过程中会有极大的帮助。本节将涉及一些基础的计算图知识，但并不要求读者事先对此有深入的了解。关于计算图的基础知识推荐阅读Christopher Olah的文章^1。
# colah.github.io/posts/2015-08-Backprop
'''

from __future__ import print_function
import torch

# 在创建tensor的时候指定requires_grad
a = torch.randn(3, 4, requires_grad=True)
# 或者
a = torch.randn(3, 4).requires_grad_()
# 或者
a = torch.randn(3, 4)
a.requires_grad = True
print(a)

b = torch.zeros(3, 4, requires_grad=True)
print(b)

c = a.add(b)
print(c)

d = c.sum()
d.backward() # 这个就是反向传播

print(d) # d还是一个requires_grad=True的Tensor,对他的操作需要慎重
print(d.requires_grad)

print('a:', a.grad)
print('b:', b.grad)

# 此处虽然没有指定c需要求导，但c依赖于a，而a需要求导
# 因此c的requires_grad属性会自动设置为True
print(a.requires_grad, b.requires_grad, c.requires_grad)

# 由用户创建的variable属于叶子节点，对应的grad_fn是None
print(a.is_leaf, b.is_leaf, c.is_leaf)

# c.grad是None,因C不是叶子节点，他的梯度是用于计算a的梯度
# 所以虽然c.requires_grad = True,但其梯度计算完之后被释放
# print(c.grad is None)

def f(x):
    y = x**2 * torch.exp(x)
    return y

def gradf(x):
    dx = 2 * x * torch.exp(x) + x**2 * torch.exp(x)
    return dx

x = torch.randn(3, 4, requires_grad=True)
print('x:', x)
y = f(x)
print('y:', y)

y.backward(torch.ones(y.size()))
print('x.grad:', x.grad)

print('gradf(x):', gradf(x))
# 这个为什么不一样阿

# Pytorch中autograd的底层用了计算图，计算图是一种特殊的有向五环图，用于记录算子和变量之间的关系，一般用矩阵表示算子，椭圆表示变量，如表达式
# z = wx + b 可以拆分为 y = wx 和 z = y + b。
x = torch.ones(1)
b = torch.randn(1, requires_grad=True)
w = torch.randn(1, requires_grad=True)
y = w * x
z = y + b

print(x.requires_grad, b.requires_grad, w.requires_grad)

# 虽然没有指定y.requires_grad为True,但由于y依赖于需要求导的w
# 故而requires_grad为True

print(y.requires_grad)

print(x.is_leaf, w.is_leaf, b.is_leaf)

print(y.is_leaf, z.is_leaf)

# grad_fn可以查看这个variable的反向传播函数
# z是add函数的输出，所以他的反向传播函数是AddBackward
print(z.grad_fn) # <AddBackward0 object at 0x7fab39ea6590>

# next_functions保存grad_fn的输入，是一个tuple，tuple的元素也是Function
# 第一个是y，他是乘法的输出，所以对应的反向传播函数y.grad_fn是MulBackward
# 第二个是b，它是叶子节点，由用户创建，grad_fn为None，但是有
print(z.grad_fn.next_functions)

# 第一个是w,叶子节点，需要求导，梯度是累加的
# 第二个是x,叶子节点，不需要求导，所以为None
print(y.grad_fn.next_functions)

# 叶子节点的grad_fn是None
print(w.grad_fn, x.grad_fn)

# 计算w的梯度的时候，需要用