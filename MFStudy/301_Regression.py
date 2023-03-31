import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.nn as nn

# usage of torch.unsqueeze
'''
x = torch.tensor([1, 2, 3])
print(x)
print(x.shape)
x = torch.unsqueeze(x, dim=0)
print(x)
print(x.shape)
x = torch.unsqueeze(x, dim=1)
print(x)
print(x.shape) 
'''

x_pre = torch.linspace(-1, 1, 100)
x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
y = x.pow(2) + 0.2*torch.rand(x.size())
# print(x_pre, x_pre.shape)
# print(x, x.shape)

x, y = Variable(x), Variable(y)

# plt.figure()
# plt.scatter(x.data.numpy(), y.data.numpy())
# plt.show()

class Net(nn.Module):
    def __init__(self, n_input, n_hidden_1, n_hidden_2, n_output):
        super(Net, self).__init__()
        self.hidden_layer_1 = nn.Linear(n_input, n_hidden_1)
        self.hidden_layer_2 = nn.Linear(n_hidden_1, n_hidden_2)
        self.output_layer = nn.Linear(n_hidden_2, n_output)

    def forward(self, x):
        x = self.hidden_layer_1(x)
        x = F.relu(x)
        x = self.hidden_layer_2(x)
        x = F.relu(x)
        x = self.output_layer(x)
        return x

net = Net(1, 16, 8, 1)
# print(net) #show Network art
plt.ion()
plt.show()

# define the optimizer and loss_function
optimizer = torch.optim.SGD(net.parameters(), lr=0.25)
criterion = torch.nn.MSELoss()

n_epochs = 300
for epoch in range(n_epochs):
    pred = net(x)
    loss = criterion(pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 5 == 0:
        # plot and show learning process
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), pred.data.numpy(), c='r', lw=5)
        plt.text(0.5, 0, 'Loss=%.4f ' % loss.item(), fontdict={'size': 20, 'color': 'blue'})
        plt.text(0.5, 1, 'epoch=%d ' % epoch, fontdict={'size': 20, 'color': 'blue'})
        plt.pause(0.5)
plt.ioff()
plt.show()

'''
hidden_layer_number = 1
lr = 0.5
    hidden=64
        完成实验9次，3次出现优化不动，2次出现优化不完全，4次优化正常
        其他3次均能正常优化
        分析原因：不清楚为啥，去找赵又没有看到模型参数的函数
    hidden=32
        完成实验5次，4次出现优化不动，0次出现优化不完全，1次优化正常
    hidden=16
        完成实验5次，1次出现优化不动，1次出现优化不完全，3次优化正常
    hidden=8
        完成实验5次，0次出现优化不动，2次出现优化不完全，3次优化正常
lr=0.25
    hidden=64
    完成实验5次，0次出现优化不动，1次出现优化不完全，4次优化正常
    hidden=32
    完成实验5次，0次出现优化不动，0次出现优化不完全，5次优化正常
    hidden=16
    完成实验5次，0次出现优化不动，0次出现优化不完全，5次优化正常
    hidden=8
    完成实验5次，0次出现优化不动，0次出现优化不完全，5次优化正常

hidden_layer_number = 2
lr = 0.5
    hidden_layer_1 =32  &&  hidden_layer_2=16
        完成实验4次，0次出现优化不动，2次出现优化不完全，2次优化正常(这个正常感觉也不太正常)
    hidden_layer_1 =32  &&  hidden_layer_2=8
        完成实验4次，2次出现优化不动，1次出现优化不完全，1次优化正常(也不太正常)
    hidden_layer_1 =32  &&  hidden_layer_2=4
        完成实验4次，0次出现优化不动，0次出现优化不完全，0次优化正常
lr=0.25
    hidden=64
    完成实验5次，0次出现优化不动，1次出现优化不完全，4次优化正常
    hidden=32
    完成实验5次，0次出现优化不动，0次出现优化不完全，5次优化正常
    hidden=16
    完成实验5次，0次出现优化不动，0次出现优化不完全，5次优化正常
    hidden=8
    完成实验5次，0次出现优化不动，0次出现优化不完全，5次优化正常
'''