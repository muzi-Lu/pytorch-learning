import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

# Define Hyper Parameters
EPOCH = 10
BATCH_SIZE = 64
LR_G = 0.001
LR_D = 0.001
ART_COMPONENTS = 15
PAINT = [np.linspace(-1, 1, ART_COMPONENTS) for _ in range(BATCH_SIZE)]
print(PAINT)
PAINT_POINTS = np.vstack([np.linspace(-1, 1, ART_COMPONENTS) for _ in range(BATCH_SIZE)]) # not understand
print(PAINT_POINTS.shape)