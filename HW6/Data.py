from torch.utils.data import Dataset
from pathlib import Path
from torchvision.transforms import transforms
import os.path

class MyDataset(Dataset):
    def __init__(
            self,
            folder,
            image_size
    ):
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for p in Path(f'{folder}').glob(f'**/*.jpg')]

        self.transforms = transforms.Compose(
            transforms.Resize(image_size),
            transforms.ToTensor()
        )