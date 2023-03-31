import os.path

from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms
from PIL import Image
train_tfm = transforms.Compose([

        ##### add more transforms here #####
        transforms.Resize((128, 128)),
        ##### added more here #####
        transforms.ToTensor(),
    ])

test_tfm = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    ])

class FoodDataset(Dataset):
    def __init__(self, path, tfm=test_tfm, files=None):
        super(FoodDataset).__init__()
        self.path = path
        self.files = sorted([os.path.join(path, x) for x in os.listdir(path) if x.endswith(".jpg")])
        if files != None:
            self.files = files
        transforms = tfm

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        im = Image.open(fname)
        im = transforms(im)

        try:
            label = int(fname.split("/")[-1].split(" ")[0])
        except:
            label = -1
        return im, label