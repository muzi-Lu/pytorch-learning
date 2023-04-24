import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

'''
写一个7Scenes的位姿dataset
'''
class PoseDataset(Dataset):

    def __init__(self, scene, data_path, train, transform, target_transform, mode, df, trainskip, testskip, hwf, ret_idx):
        pass
    """
          :param scene: scene name[chess, red kitchen, heads, ...]
          :param data_path: root 7Scenes data dir
          :param train: if train is True, return training images, else return testing images
          :param transform: transform images
          :param target_transform: transform poses
          :param mode: 0 just is color images, 1 is Nerf color images resized and 0-1
          :param df: downscale factor
          :param trainskip: due to 7Scenes is large and now use trainskip
          :param testskip: same as above
          :param hwf: H,W,Focal from colmap
          :param ret_idx: bool, currently used in NeRF-W
    """


    def name(self):
        return 'PoseDataset'

    def __getitem__(self, index):
        pass

    def __index__(self):
        pass
