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

b = a.view(-1, 3)  # 当某一维为-1的时候，会自动计算他的大小
print(b)

b = b.unsqueeze(1)  # 注意形状， 在第一维(下标从0开始)上增加‘1’
print('why:', a[:, None].shape)  # 这个梅东
'''
# None类似于np.newaxis, 为a新增了一个轴
'''
print(b.shape)

print(b.unsqueeze(-2).shape)  # -2表示倒数第二个维度

c = b.view(1, 2, 1, 1, 3)
print(c.shape)
c = c.squeeze(0)
print(c.shape)

c = c.squeeze()  # 把所有维度为1的压缩
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

'''
索引操作
Tensor支持与numpy.ndarray类似的索引操作，语法上也类似，下面通过一些例子，讲解常用的索引操作。如无特殊说明，索引出来的结果与原tensor共享内存，也即修改一个，另一个会跟着修改。
'''

# 第0行(下标从0开始)
a = torch.randn(3, 4)
print('a:', a)

# 第0列
print('first row:', a[0])
print('Column 0:', a[:, 0])

# The second element in row 0, equivalent to a[0, 2]
print('a[0, 2]:', a[0][2])

# 第0行最后一个元素
print('a[0, -1]:', a[0][-1])

# 前两行
print('a[0-1]:', a[:2])

# 前两行，第0,1列
print('a[0-1][0-1]:', a[:2, :2])

# # 注意两者的区别：形状不同
print(a[0:1, :2])  # 第0行，前两列
print(a[0, :2])  # 注意两者的区别：形状不同

# None类似于np.newaxis, 为a新增了一个轴
'''
exp
'''
print(a[None].shape)
print(a[:, None].shape)
print(a[:, :, None].shape)
# 等价于a.view(1, a.shape[0], a.shape[1])


print(a[:, None, :, None, None].shape)

print(a > 1)
print(a[a > 1])  # 等价于a.masked_select(a>1)
# 选择结果与原tensor不共享内存空间)

print(a[torch.LongTensor([0, 1])])  # 第0行和第1行)

'''
表3-2常用的选择函数

函数	功能	
index_select(input, dim, index)	在指定维度dim上选取，比如选取某些行、某些列	
masked_select(input, mask)	例子如上，a[a>0]，使用ByteTensor进行选取	
non_zero(input)	非0元素的下标	
gather(input, dim, index)	根据index，在dim维度上选取数据，输出的size与index一样	
https://zhuanlan.zhihu.com/p/462008911
'''

a = torch.arange(0, 16).view(4, 4)
print(a)

# 选取对角线的元素
index = torch.LongTensor([[0, 1, 2, 3]])
print(index)
print(a.gather(0, index))  # messi

# 选取反对角线上的元素
index = torch.LongTensor([[3, 2, 1, 0]]).t()
print(a.gather(1, index))

# 选取反对角线上的元素，注意与上面的不同
index = torch.LongTensor([[3, 2, 1, 0]])
print(a.gather(0, index))

# # tensor.gather
# tensor = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
# index = torch.tensor([[0, 2], [1, 3], [2, 0]])
#
# # 在第二维上按照索引取值
# result = tensor.gather(1, index)
# print(result)

# 选取两个对角线上的元素
index = torch.LongTensor([[0, 1, 2, 3], [3, 2, 1, 0]]).t()
b = a.gather(1, index)
print(b)

'''
与gather相对应的逆操作是scatter_，gather把数据从input中按index取出，而scatter_是把取出的数据再放回去。注意scatter_函数是inplace操作。
'''

c = torch.zeros(4, 4)
c.scatter_(1, index, b.float())
print(c)

'''
对tensor的任何索引操作仍是一个tensor，想要获取标准的python对象数值，需要调用tensor.item(), 这个方法只对包含一个元素的tensor适用
'''
print(a[0][0])
print(a[0][0].item())
d = a[0:1, 0:1, None]
print(d.shape)
print(d.item())

'''
高级索引
PyTorch在0.2版本中完善了索引操作，目前已经支持绝大多数numpy的高级索引[^10]。高级索引可以看成是普通索引操作的扩展，但是高级索引操作的结果一般不和原始的Tensor共享内存。 
[^10]: https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html#advanced-indexing
'''
x = torch.arange(0, 27).view(3, 3, 3)
print(x)
print(x[[1, 2], [1, 2], [2, 0]])
print(x[[2, 1, 0], [0], [1]])

'''
Tensor类型
Tensor有不同的数据类型，如表3-3所示，每种类型分别对应有CPU和GPU版本(HalfTensor除外)。默认的tensor是FloatTensor，可通过t.set_default_tensor_type 来修改默认tensor类型
(如果默认类型为GPU tensor，则所有操作都将在GPU上进行)。Tensor的类型对分析内存占用很有帮助。例如对于一个size为(1000, 1000, 1000)的FloatTensor，
它有1000*1000*1000=10^9个元素，每个元素占32bit/8 = 4Byte内存，所以共占大约4GB内存/显存。HalfTensor是专门为GPU版本设计的，同样的元素个数，显存占用只有FloatTensor的一半，
所以可以极大缓解GPU显存不足的问题，但由于HalfTensor所能表示的数值大小和精度有限^2，所以可能出现溢出等问题。
'''

'''
表3-3: tensor数据类型

Data    type	                     dtype	                 CPU tensor	        GPU tensor
32-bit floating point	torch.float32 or torch.float	torch.FloatTensor	torch.cuda.FloatTensor
64-bit floating point	torch.float64 or torch.double	torch.DoubleTensor	torch.cuda.DoubleTensor
16-bit floating point	torch.float16 or torch.half	    torch.HalfTensor	torch.cuda.HalfTensor
8-bit integer(unsigned) torch.uint8	                    torch.ByteTensor	torch.cuda.ByteTensor
8-bit integer(signed)	torch.int8	                    torch.CharTensor	torch.cuda.CharTensor
16-bit integer(signed)	torch.int16 or torch.short	torch.ShortTensor	torch.cuda.ShortTensor
32-bit integer(signed)	torch.int32 or torch.int	torch.IntTensor	torch.cuda.IntTensor
64-bit integer(signed)	torch.int64 or torch.long	torch.LongTensor	torch.cuda.LongTensor


各数据类型之间可以互相转换，type(new_type)是通用的做法，同时还有float、long、half等快捷方法。CPU tensor与GPU tensor之间的互相转换通过tensor.cuda和tensor.cpu方法实现，
此外还可以使用tensor.to(device)。Tensor还有一个new方法，用法与t.Tensor一样，会调用该tensor对应类型的构造函数，生成与当前tensor类型一致的tensor。torch.*_like(tensora) 
可以生成和tensora拥有同样属性(类型，形状，cpu/gpu)的新tensor。
 tensor.new_*(new_shape) 新建一个不同形状的tensor。
'''

'''
逐元素操作

'''
a = torch.arange(0, 6).view(2, 3).float()
print(torch.cos(a))

print(a % 3)

print(a ** 2)

print(torch.clamp(a, min=3))

'''
归并操作
此类操作会输出形状小于输入形状，并可以沿着某一维度进行指定操作。如加法sum，既可以计算整个
tensor的和，也可以计算tensor中每一行或者每一列的和。常见的归并操作如表3-5所示。

    函数                               功能
mean/sum/median/mode              均值/和/中位数/众数
norm/dist                            范数/距离
std/var                              标准差/方差
cumsum/cumprod                        累加/累乘   
'''

b = torch.ones(2, 3)
print(b.sum(dim=0, keepdim=True))
print(b.sum(dim=0, keepdim=False))

print(b.sum(dim=1))

'''
比较操作
'''
a = torch.linspace(0, 15, 6).view(2, 3)
print(a)
b = torch.linspace(15, 0, 6).view(2, 3)
print(b)
print(a > b)
print(a[a > b])
print(torch.max(a))

'''
函数	功能
trace	对角线元素之和(矩阵的迹)
diag	对角线元素
triu/tril	矩阵的上三角/下三角，可指定偏移量
mm/bmm	矩阵乘法，batch的矩阵乘法
addmm/addbmm/addmv/addr/badbmm..	矩阵运算
t	转置
dot/cross	内积/外积
inverse	求逆矩阵
svd	奇异值分解
'''

a = torch.randn(3, 3)
print(a)
b = a.t()
print(b, b.is_contiguous())
print(b.contiguous())

'''
3.1.2 Tensor和Numpy
Tensor和Numpy数组之间具有很高的相似性，彼此之间的互操作也非常简单高效。需要注意的是，Numpy和Tensor共享内存。
由于Numpy历史悠久，支持丰富的操作，所以当遇到Tensor不支持的操作时，可先转成Numpy数组，处理后再转回tensor，其转换开销很小。
'''
import numpy as np

a = np.ones([2, 3], dtype=np.float32)
print(a)

b = torch.from_numpy(a)
print(b)

a[0, 1] = 100
b[0][0] = 2
print(a)
print(b)

c = b.numpy()
print(c, c.dtype)

# 注意： 当numpy的数据类型和Tensor的类型不一样的时候，数据会被复制，不会共享内存。

a = np.ones([2, 3])
print(a.dtype)

b = torch.Tensor(a) # 此处进行拷贝，不共享内存
print(b.dtype)

c = torch.from_numpy(a) # 注意c的类型（DoubleTensor）
print(c)

a[0, 1] = 100
print(b)
print(c)
# 注意： 不论输入的类型是什么，t.tensor都会进行数据拷贝，不会共享内存

'''
广播法则(broadcast)是科学运算中经常使用的一个技巧，它在快速执行向量化的同时不会占用额外的内存/显存。 Numpy的广播法则定义如下：

让所有输入数组都向其中shape最长的数组看齐，shape中不足的部分通过在前面加1补齐
两个数组要么在某一个维度的长度一致，要么其中一个为1，否则不能计算
当输入数组的某个维度的长度为1时，计算时沿此维度复制扩充成一样的形状
PyTorch当前已经支持了自动广播法则，但是笔者还是建议读者通过以下两个函数的组合手动实现广播法则，这样更直观，更不易出错：

unsqueeze或者view，或者tensor[None],：为数据某一维的形状补1，实现法则1
expand或者expand_as，重复数组，实现法则3；该操作不会复制数组，所以不会占用额外的空间。
注意，repeat实现与expand相类似的功能，但是repeat会把相同数据复制多份，因此会占用额外的空间。
'''

a = torch.ones(3, 2)
b = torch.zeros(2, 3, 1)
# 自动广播法则
# 第一步：a是2维,b是3维，所以先在较小的a前面补1 ，
#               即：a.unsqueeze(0)，a的形状变成（1，3，2），b的形状是（2，3，1）,
# 第二步:   a和b在第一维和第三维形状不一样，其中一个为1 ，
#               可以利用广播法则扩展，两个形状都变成了（2，3，2）
print(a+b)

# 手动广播法则
# 或者 a.view(1,3,2).expand(2,3,2)+b.expand(2,3,2)
a[None].expand(2, 3, 2) + b.expand(2, 3, 2)

# expand不会占用额外空间，只会在需要的时候才扩充，可极大节省内存
e = a.unsqueeze(0).expand(10000000000000, 3, 2)