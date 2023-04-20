import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

class PoseDataset(Dataset):
    '''
    super(PoseDataset, self).__init__()
    错了，你要写一个初始化函数，里面才是这个
    '''
    def __init__(self):
        super(PoseDataset, self).__init__()
