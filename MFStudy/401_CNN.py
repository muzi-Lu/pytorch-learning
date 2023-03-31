# Not Finised (want to add Sklearn part)
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
import torchvision
import matplotlib.pyplot as plt

# Hyper Parameters
Epoch = 10
BATCH_SIZE = 50
LR = 0.001
DOWNLOAD_MNIST = False

train_data = torchvision.datasets.MNIST(
    root='/media/benben/0ECABB60A248B50C/MNIST',
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=DOWNLOAD_MNIST
)

# print(training_data.train_data.size())
# print(training_data.train_labels.size())
# # print(training_data.train_data.type)
# plt.imshow(training_data.train_data[1].numpy(), cmap='gray')
# plt.show()

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=False)

test_data = torchvision.datasets.MNIST(root="/media/benben/0ECABB60A248B50C/MNIST", train=False)
test_x = Variable(torch.unsqueeze(test_data.test_data, dim=1), volatile=True).type(torch.FloatTensor)[:]/255. #这一步要多研究
test_y = test_data.test_labels[:]

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.out = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x) #(batch, 32, 7, 7)
        x = x.view(x.size(0), -1) # (batch, 32*7*7)
        output = self.out(x)
        return output, x

cnn = CNN()
# print(cnn)

optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
critizer = torch.nn.CrossEntropyLoss()

from matplotlib import cm
# try: from sklearn

for epoch in range(Epoch):
    for step, (b_x, b_y) in enumerate(train_loader):
        output = cnn(b_x)[0]
        loss = critizer(output, b_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            test_output, last_layer = cnn(test_x)
            print('test_output size:', test_output.shape,'last_layer:', last_layer.shape)
            pred_y = torch.max(test_output, 1)[1].data.numpy() #sloved : max的第2个参数1代表着同行内列间比较
            accuracy = float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
            print('epoch:', epoch, '| train loss :%.4f' % loss.data.numpy(), '| test accuracy: %.5f' % accuracy)

test_output, _ = cnn(test_x)
pred_y = torch.max(test_output, 1)[1].data.numpy()
print(pred_y, 'prediction number')
print(test_y.numpy(), 'real number')

torch.save(cnn, 'MNIST_Digals_Nets.pkl')