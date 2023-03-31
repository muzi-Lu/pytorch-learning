import torch
from torch.utils.data import Dataset, DataLoader, random_split

class COVID19Dataset(Dataset):

    def __init__(self, x, y=None):
        self.x = torch.FloatTensor(x)
        if y is None:
            self.y = y
        else:
            self.y = torch.FloatTensor(y)

    def __getitem__(self, idx):
        if self.y is None:
            return self.x[idx]
        else:
            return self.x[idx], self.y[idx]

    def __len__(self):
        return len(self.x)

