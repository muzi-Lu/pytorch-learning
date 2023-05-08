import torch
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader
import numpy as np

class ExampleDataset(data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

data = np.random.randint(0, 100, size=20)
print(data)
dataset = ExampleDataset(data)
# print(dataset.__len__())
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
for batch in dataloader:
    print(batch)