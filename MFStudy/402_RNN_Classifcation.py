import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# HyperParameters
EPOCH = 2
BATCH_SIZE = 64
TIME_STEP = 28
INPUT_SIZE = 28
LR = 0.1
DOWNLOAD_MNIST = False

# Prepare the datasets and the dataloader
# train_data = dsets.MNIST(root)

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

    def forward(self):
        pass
