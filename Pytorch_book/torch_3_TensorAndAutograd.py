'''
3.1 Tensor

'''
import torch
'''
3.1.1 基础操作
tensor的接口和Numpy相似，以方便用户使用。但不熟悉Numpy也先没有关系

从torch接口的角度进行分类可以分为两类：
1.torch.function
2.tensor.function

从存储的角度来讲，tensor的操作可以分为两类：
1.不会修改自身数据的
2.会修改自身数据的
'''

'''
create tensor
函数	功能
Tensor(*sizes)	基础构造函数
tensor(data,)	类似np.array的构造函数
ones(*sizes)	全1Tensor
zeros(*sizes)	全0Tensor
eye(*sizes)	    对角线为1，其他为0
arange(s,e,step)	从s到e，步长为step
linspace(s,e,steps)	从s到e，均匀切分成steps份
rand/randn(*sizes)	均匀/标准分布
normal(mean,std)/uniform(from,to)	正态分布/均匀分布
randperm(m)	    随机排列
'''
a = torch.Tensor(2, 3)
print('a:', a)

b = torch.Tensor(
    [
    [1, 2],
    [3, 4]
    ])
print('b:', b)

c = b.tolist()
print('c:', c)

# tensor.size() 返回 torch.size()对象，它是tuple的子类，但其使用方法与tuple略有不同
b_size = b.size()
print('b_size:', b_size)

# b中元素总个数， 2*3， 等价于b.nelement()
b_sumnum = b.numel()
print('b_numsum:', b_sumnum)

# 创建一个和b形状一样的tensor
e = torch.Tensor(b_size)
print('e:', e)
# c创建一个元素为2和3的tensor
f = torch.Tensor((2, 3))
print('f:', f)
print(e, f)

# 除了tensor.size(), 还可以利用tensor.shape直接查看tensor的形状，tensor.shape等价于tensor.size()
print('b.shape = b.size():', b.shape)

# 其他创建tensor的方法：
print(torch.ones(2, 3))
print(torch.zeros(2, 3))
print(torch.arange(1, 4, 2))
print(torch.linspace(1, 10, 3))
print(torch.randn(2, 3, device=torch.device('cpu')))
print(torch.randperm(5))
print(torch.eye(2, 3, dtype=torch.int))

scalar = torch.tensor(3.14159)
print('scalar: %s, shape of sclar: %s' % (scalar, scalar.shape))

vector = torch.Tensor([1, 2])
print('vector: %s, shape of vector: %s' % (vector, vector.shape))

vector = torch.Tensor(1, 2)
print('vector: %s, shape of vector: %s' % (vector, vector.shape))
vector = torch.Tensor(4, 3)
print('vector: %s, shape of vector: %s' % (vector, vector.shape))

matrix = torch.Tensor([
    [0.1, 1.2],
    [2.2, 3.1],
    [4.9, 5.2]
])
print(matrix, matrix.shape)

row_tensor = torch.Tensor([[0.111, 0.222, 0.333]],
                          # dtype=torch.float64, # 为啥这个地方会报错
                          device=torch.device('cpu'))
print(row_tensor)

'''
常用Tensor操作
通过tensor.view方法可以调整tensor形状但必须保证调整前后元素总数一致，view不会改变自身数据，返回的新tensor与源tensor共享内存，
改变一个，另外一个也跟着改变。

在实际应用中可能经常需要添加或减少某一个维度，这个时候squeeze和unsqueeze这两个函数也就有用了
'''
a = torch.arange(0, 6)
print(a)
a = a.view(2, 3)
print(a)

b = a.view(-1, 3) # 当某一维为-1的时候，会自动计算他的大小
print(b)

b = b.unsqueeze(1) # 注意形状， 在第一维(下标从0开始)上增加‘1’
print(a[:, None].shape) # 这个梅东
print(b.shape)

print(b.unsqueeze(-2).shape) # -2表示倒数第二个维度

c = b.view(1, 2, 1, 1, 3)
print(c.shape)
c = c.squeeze(0)
print(c.shape)

c = c.squeeze() # 把所有维度为1的压缩
print(c.shape)

a[1] = 100
print(b)

'''
resize是另一种可以调整size的方法, 但与view不同, 它可以修改tensor的大小。如果新大小超过了原大小，会自动分配新的内存空间，而如果新大小小于原大小，则之前的数据依然会被保存
'''

b.resize_(1, 3)
print(b)

b.resize_(3, 3)
print(b)
