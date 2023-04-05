import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Subset, Dataset
from torchvision.datasets import DatasetFolder, VisionDataset

from tqdm import tqdm
import random

from Data import FoodDataset
from Model import Classifier
##### self-made #####


if __name__ == '__main__':
    ##### Configurations #####
    # "Cuda" only when GPUS are available
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize a model. and put it on the specified device
    model = Classifier().to(device)

    # The number of batch size
    batch_size = 64

    # The number of Epochs
    n_epochs = 8

    # If no improvement in 'patience' epochs, early stopping
    patience = 5

    # For the classification task, we use cross-entropy the measurement of performances
    criterion = nn.CrossEntropyLoss()

    # Initialize optimizer, you may fine-tune some hyperparameters such as learning rate on your own
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003, weight_decay=1e-5)
    ##### One : Prepare Datasets and Dataloaders #####

    train_tfm = transforms.Compose([

        ##### add more transforms here #####
        transforms.Resize((128, 128)),
        # transforms.RandomCrop(112), # 这个还要看一下图片大小吧
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.RandomRotation(30),
        ##### added more here #####
        transforms.ToTensor(),
    ])

    test_tfm = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    train_set = FoodDataset("/media/benben/0ECABB60A248B50C/HWHomework/datasets/3/train", tfm=train_tfm)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    valid_set = FoodDataset("/media/benben/0ECABB60A248B50C/HWHomework/datasets/3/valid", tfm=test_tfm)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)


    ##### Starting Training #####
    # Initialize trackers, these are not parameters and should not be changed

    stale = 0
    best_acc = 0

    for epoch in range(n_epochs):
        # Make sure your model is in train code before training
        model.train()

        #These are used to record information in training
        train_loss = []
        train_accs = []

        for batch in tqdm(train_loader):
            imgs, labels = batch
            print(imgs.shape, labels.shape)

            # put images into model and Forward the data (make sure data and model are on the same device)
            logits = model(imgs.to(device))

            # Calculate the cross-entropy loss
            loss = criterion(logits, labels.to(device))

            # Gradients stored in the parameters in the previous step should be cleard out first
            # Note: This can be done in the first step
            optimizer.zero_grad()

            # Compute the gradients for parameters
            loss.backward()

            # Clip the gradient norms for stable training to prevent the gradient from exploding
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)

            # Undate the parameters with computed gradient
            optimizer.step()

            acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

            # Record the loss and accuracy.
            train_loss.append(loss.item())
            train_accs.append(acc)

        train_loss = sum(train_loss) / len(train_loss)
        train_acc = sum(train_accs) / len(train_accs)




