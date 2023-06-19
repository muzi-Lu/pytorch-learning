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
d.backward()  # 这个就是反向传播

print(d)  # d还是一个requires_grad=True的Tensor,对他的操作需要慎重
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
    y = x ** 2 * torch.exp(x)
    return y


def gradf(x):
    dx = 2 * x * torch.exp(x) + x ** 2 * torch.exp(x)
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
print(z.grad_fn)  # <AddBackward0 object at 0x7fab39ea6590>

# next_functions保存grad_fn的输入，是一个tuple，tuple的元素也是Function
# 第一个是y，他是乘法的输出，所以对应的反向传播函数y.grad_fn是MulBackward
# 第二个是b，它是叶子节点，由用户创建，grad_fn为None，但是有
print(z.grad_fn.next_functions)

# 第一个是w,叶子节点，需要求导，梯度是累加的
# 第二个是x,叶子节点，不需要求导，所以为None
print(y.grad_fn.next_functions)

# 叶子节点的grad_fn是None
print(w.grad_fn, x.grad_fn)

# 计算w的梯度的时候，需要用x的数值，这些数值在前向过程中会保存成buffer,在计算完梯度之后会自动清空，
# 为了能够多次反向传播需要指定retain_graph来保留这些buffer

# 使用retain_graph来保存buffer
z.backward(retain_graph=True)
print(w.grad)

# 多次反向传播，梯度累加，也就是w中AccumulateGrad的含义
z.backward()  # 如果有第三次的话这个地方也要加一个反向传播
print(w.grad)


# 多次反向传播，梯度累加，也就是w中AccumulateGrad的含义
# z.backward()
# print(w.grad)


# Pytorch使用的是动态图，他的计算图在每次前向传播都需要从头开始构建，所以它能够使用Python控制语句。
# 根据需求创建计算图

def abs(x):
    if x.data[0] > 0:
        return x
    else:
        return -x


x = torch.ones(1, requires_grad=True)
y = abs(x)
y.backward()
print(x)

x = -1 * torch.ones(1)
print(x)
x = x.requires_grad_() # 如果这行注释的话，试试下面运行
print(x)
y = abs(x)
print(y)
y.backward()
print(x.grad)

# 因为x是可以有requires_grad
cc = x*3
print(cc.requires_grad)

def f(x):
    result = 1
    for ii in x:
        if ii.item() > 0:
            result = ii * result
    return result

x = torch.arange(-2, 4, dtype=torch.float32).requires_grad_()
print(x)
y = f(x)
print(y)
y.backward()
print(x.grad)

'''
变量的requires_grad属性默认为False，如果某一个节点requires_grad被设置为True，那么所有依赖它的节点requires_grad都是True。这其实很好理解，对于
，x.requires_grad = True，当需要计算时，根据链式法则，，自然也需要求，所以y.requires_grad会被自动标为True.
有些时候我们可能不希望autograd对tensor求导。认为求导需要缓存许多中间结构，增加额外的内存/显存开销，那么我们可以关闭自动求导。
对于不需要反向传播的情景（如inference，即测试推理时），关闭自动求导可实现一定程度的速度提升，并节省约一半显存，因其不需要分配空间计算梯度。
'''

x = torch.ones(1, requires_grad=True)
w = torch.rand(1, requires_grad=True)
y = x * w
# y依赖于w，而w.requires_grad = True
print(x.requires_grad, w.requires_grad, y.requires_grad)

with torch.no_grad():
    x = torch.ones(1)
    w = torch.ones(1, requires_grad=True)
    y = x * w
    # y依赖于w和x，虽然w.requires_grad = True，但是y的requires_grad依旧为False
    print(x.requires_grad, w.requires_grad, y.requires_grad)

grad_enabled = torch.set_grad_enabled(True)
print(grad_enabled)

# 如果我们想要修改tensor的数值，但是又不希望被autograd记录，那么我么可以对tensor.data进行操作
a = torch.ones(3, 4, requires_grad=True)
b = torch.ones(3, 4, requires_grad=True)
c = a * b
print(a.data) # 还是一个Tensor

print(a.data.requires_grad) # 独立计算图之外了

d = a.data.sigmoid_()
print(d.requires_grad)

'''
在反向传播过程中非叶子节点的导数计算完之后即被清空。若想查看这些变量的梯度，有两种方法：

使用autograd.grad函数
使用hook

autograd.grad和hook方法都是很强大的工具，更详细的用法参考官方api文档，这里举例说明基础的使用。推荐使用hook方法，但是在实际使用中应尽量避免修改grad的值
'''

x = torch.ones(3, requires_grad=True)
w = torch.rand(3, requires_grad=True)
y = x * w
# y依赖于w，而w.requires_grad = True
z = y.sum()
print(x.requires_grad, w.requires_grad, y.requires_grad)

# 非叶子节点grad计算完之后自动清空，y.grad是None
z.backward()
# print(x.grad, w.grad, y.grad)
# UserWarning: The .grad attribute of a Tensor that is not a leaf Tensor is being accessed.
# Its .grad attribute won't be populated during autograd.backward(). If you indeed want the gradient for a non-leaf Tensor, use .retain_grad() on the non-leaf Tensor.
# If you access the non-leaf Tensor by mistake, make sure you access the leaf Tensor instead. See github.com/pytorch/pytorch/pull/30531 for more informations.
# print(x.grad, w.grad, y.grad)

# 第一种方法：使用grad获取中间变量的梯度
x = torch.ones(3, requires_grad=True)
w = torch.rand(3, requires_grad=True)
y = x * w
z = y.sum()
# z对y的梯度，隐式调用backward()
print(torch.autograd.grad(z, y))

# 第二种方法：使用hook
# hook是一个函数，输入是梯度，不应该有返回值
def variable_hook(grad):
    print('y的梯度：',grad)

x = torch.ones(3, requires_grad=True)
w = torch.rand(3, requires_grad=True)
y = x * w
# 注册hook
hook_handle = y.register_hook(variable_hook)
z = y.sum()
z.backward()


# 除非你每次都要用hook，否则用完之后记得移除hook
hook_handle.remove()
