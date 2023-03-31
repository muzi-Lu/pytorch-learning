# Import necessary packages.
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
# "ConcatDataset" and "Subset" are possibly useful when doing semi-supervised learning.
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torchvision.datasets import DatasetFolder

# This is for the progress bar.
from tqdm.auto import tqdm

##### Transforms Here #####

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()

        ##### Networks Architecture #####

        ##### Input [3, 128, 128] #####
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),  # [64, 128, 128]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0), # [64, 64, 64]

            nn.Conv2d(64, 128, 3, 1, 1), # [128, 64, 64]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0), # [128, 32 ,32]

            nn.Conv2d(128, 256, 3, 1, 1), # [256, 32, 32]
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0), # [256, 16, 16]

            nn.Conv2d(256, 512, 3, 1, 1), # [512, 16, 16]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0), # [512, 8, 8]

            nn.Conv2d(512, 1024, 3, 1, 1), #[1024, 8, 8]
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0) # [1024, 4, 4]
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(1024*4*4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 11)
        )

    def forward(self, x):
        # input(x):[batch_size, 3, 128, 128]
        # outputs:[batch_size ,11]

        # Extract features by CNN layers
        x = self.cnn_layers(x)

        # The extract features maps must be flatten before going to fully_connected layers
        x = x.view(x.size()[0], -1) # x.size()[0] -->batchsize -1 -> W*H*F

        # The features are transformed by fully-connected layers to obtain the final logits
        x = self.fc_layers(x)
        return x