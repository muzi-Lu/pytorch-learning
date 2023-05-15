from torch.utils.data import Dataset
from pathlib import Path
from torchvision.transforms import transforms
import os.path
from PIL import Image

class MyDataset(Dataset):
    def __init__(
            self,
            folder,
            image_size
    ):
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for p in Path(f'{folder}').glob(f'**/*.jpg')]

        '''
        TODO:
        Data Augmentation
        '''

        self.transforms = transforms.Compose(
            transforms.Resize(image_size),
            transforms.ToTensor()
        )

    def __init__(self):
        return len(self.paths)
    
    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        return self.transforms(img)
