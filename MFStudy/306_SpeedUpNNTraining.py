import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader, Dataset

# hyperparameters
LR = 0.01
BATCH_SIZE = 32
EPOCH = 12

x = torch.unsqueeze(torch.linspace(-1, 1, 1000), dim=1)
# x_1 = torch.unsqueeze(torch.linspace(-1, 1, 1000), dim=0)
# print(x.shape)
y = x.pow(2) + 0.1*torch.normal(torch.zeros(*x.size()))
# print(*x.size())
# print(torch.zeros(*x.size()).shape)
# print(torch.normal(torch.zeros(*x.size())))

torch_dataset = TensorDataset(x, y)
dataloader = DataLoader(torch_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=False)
# plt.scatter(x.numpy(), y.numpy())
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

net_SGD = Net(1, 16, 8, 1)
net_Momentum = Net(1, 16, 8, 1)
net_RMSprop = Net(1, 16, 8, 1)
net_Adam = Net(1, 16, 8, 1)
nets = [net_SGD, net_Momentum, net_RMSprop, net_Adam]

optimizer_SGD = optim.SGD(net_SGD.parameters(), lr=LR)
optimizer_Momentum = optim.SGD(net_Momentum.parameters(), lr=LR, momentum=0.8)
optimizer_RMSprop = optim.RMSprop(net_RMSprop.parameters(), lr=LR, alpha=0.9)
optimizer_Adam = optim.Adam(net_Adam.parameters(), lr=LR, betas=(0.9, 0.99))
optimizers = [optimizer_SGD, optimizer_Momentum, optimizer_RMSprop, optimizer_Adam]

critizer = nn.MSELoss()
losses_his  = [[], [], [], []]

for epoch in range(EPOCH):
    print('epoch:', epoch)
    for step, [batch_x, batch_y] in enumerate(dataloader):
       # b_x = Variable(batch_x)
       # b_y = Variable(batch_y)

        for net, opt, l_his in zip(nets, optimizers, losses_his):
            output = net(batch_x)
            loss = critizer(output, batch_y)

            opt.zero_grad()
            loss.backward()
            opt.step()
            # l_his.append(loss.item())
            l_his.append(loss.data.numpy())

labels = ['SGD', 'Momentum', 'RMSprop', 'Adam']
for i, l_his in enumerate(losses_his):
    plt.plot(l_his, label=labels[i])
    plt.legend(loc='best')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.ylim((0, 0.2))
plt.show()



# 看笔记或者或者源码


