import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# HyperParameters
EPOCH = 10
BATCH_SIZE = 64
TIME_STEP = 28
INPUT_SIZE = 28
LR = 0.01
DOWNLOAD_MNIST = False

# I was changed in dev branch


# Prepare the datasets and the dataloader
training_data = dsets.MNIST(
    root='/media/benben/0ECABB60A248B50C/MNIST',
    train=True,
    transform=transforms.ToTensor(),
    download=False
)
# print(training_data.train_data.size())
# print(training_data.train_labels.size())
# plt.imshow(training_data.train_data[0].numpy(), cmap='gray')
# plt.title('%i' % training_data.train_labels[0])
# plt.show()
# train_data = dsets.MNIST(root)

train_loader = torch.utils.data.DataLoader(dataset=training_data, batch_size=BATCH_SIZE, shuffle=True)

test_data = dsets.MNIST(
    root='/media/benben/0ECABB60A248B50C/MNIST',
    train=False,
    transform=transforms.ToTensor()
)
test_x = test_data.test_data.type(torch.FloatTensor)[:]/255
test_y = test_data.test_labels.numpy()[:]

# build your model
class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=64,
            num_layers=1, # number of hidden layer
            batch_first=True # (batch, time_step, input)的batch_first = True结果
        )

        self.Linear = nn.Linear(64, 10)
    def forward(self, x):
        r_out, (h_n, h_c) = self.rnn(x, None)
        # x (batch, time_step, input_size)
        # (h_n, h_c) represents the hidden and states
        out = self.Linear(r_out[:, -1, :])
        return out
        # x.shape (batch, time_step, input_size)

rnn = RNN()
print(rnn)

# choose your loss_function and optimizer

optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)
critizer = nn.CrossEntropyLoss()

# Forward and Backpropagation

for epoch in range(EPOCH):
    for step, (x, y) in enumerate(train_loader):
        b_x = Variable(x.view(-1, 28, 28)) # reshape x to (batch, time_step, input_size)
        b_y = Variable(y)

        output = rnn(b_x)
        loss = critizer(output, b_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            test_output = rnn(test_x)
            pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
            # accuracy = sum(pred_y == test_y) /test_output.size
            accuracy = float((pred_y == test_y).astype(int).sum())/ float(test_y.size) # why
            print('Epoch:', epoch, '|train loss:%.4f' % loss.data.numpy(), '| test accuracy:%.4f' % accuracy)


