import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2
import os
from os import path

class ImageDataset(Dataset):
    def __init__(self, image_dir, transform):
        self.image_dir = image_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return self.images

    def __getitem__(self, index):
        image_path = os.path.join(self.image_dir, self.images[index])

        image = Image.open(image_path)
        if self.transform is not None:
            image = self.transform(image)
        return image