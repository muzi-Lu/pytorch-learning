import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from PIL import Image
import cv2
from ImageDataset import ImageDataset

def main():
    path = ''
    transformer = transforms.Compose(
        transforms.Resize(256, 256),
        transforms.ToTensor()
    )
    imagedataset = ImageDataset(path, transformer)
    dataloader = DataLoader(imagedataset, batch_size=32)


if __name__ == '__main__':
    main()

