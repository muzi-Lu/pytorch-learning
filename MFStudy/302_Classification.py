import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt

n_data = torch.ones(100, 2)
# print(n_data)
x0 = torch.normal(2*n_data, 1)
y0 = torch.zeros(100)
x1 = torch.normal(-2*n_data, 1)
y1 = torch.ones(100)
# print(x0.shape, y0.shape, x1.shape, y1.shape)
x = torch.cat((x0, x1), 0).type(torch.FloatTensor)
# print(x, x.shape)
y = torch.cat((y0, y1), 0).type(torch.LongTensor)
# print(y, y.shape)

x, y = Variable(x), Variable(y)

plt.figure()
plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:,-1], c=y.data.numpy(),s=100, lw=0)
plt.show()

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

net = Net(2, 16, 8, 2)
net_2 = torch.nn.Sequential(
    torch.nn.Linear(2, 10),
    torch.nn.ReLU(),
    torch.nn.Linear(10, 2),
)
# print(net)

plt.ion()
plt.show()

# define the optimizer and loss_function
optimizer = torch.optim.SGD(net.parameters(), lr=0.05)
criterion = torch.nn.CrossEntropyLoss()

n_epochs = 100
for epoch in range(n_epochs):
    out = net(x)
    loss = criterion(out, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 2 == 0:
        # plot and show learning process
        plt.cla()
        prediction = torch.max(F.softmax(out), 1)[1]  # do not understand
        pred_y = prediction.data.numpy().squeeze() # do not understand
        target_y = y.data.numpy()
        plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y, s=100, lw=0)
        accuracy = sum(pred_y == target_y) / 200
        plt.text(1.5, -4, 'Accuracy=%.2f' % accuracy, fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.1)
plt.ioff()
plt.show()
