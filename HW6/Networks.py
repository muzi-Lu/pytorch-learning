import torch
import torch.nn as nn
import torch.nn.functional as F


def Upsample(dim, dim_out=None):
    '''
    上采样
    :param dim:
    :param dim_out:
    :return:
    '''
    pass


def Downsample(dim, dim_out=None):
    '''
    下采样
    :param dim:
    :param dim_out:
    :return:
    '''
    pass


class Residual(nn.Module):
    '''
    残差模块
    '''

    def __init__(self, fn):
        pass

    def forward(self, x, *args, **kwargs):
        pass


class WeightStandardizedConv2d(nn.Conv2d):
    def forward(self, x):
        pass


# sinusoidal Pos Emb

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        pass

    def forward(self, x):
        pass


class RandomOrLearnedSinusoidalPoseEmb(nn.Module):
    def __init__(self, dim, is_random=False):
        pass

    def forward(self, x):
        pass


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        pass

    def forward(self, x, scale_shift = None):
        pass


class ResnetBlock(nn.Module):
    def __init__(self):
        pass

    def forward(self, x):
        pass


class Linear_Attention(nn.Module):
    def __init__(self):
        pass

    def forward(self, x):
        pass


class Attention(nn.Module):
    def __init__(self):
        pass

    def forward(self, x):
        pass


class Unet(nn.Module):
    def __init__(self):
        pass

    def forward(self, x):
        pass