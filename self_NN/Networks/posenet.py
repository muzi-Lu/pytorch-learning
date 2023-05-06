import torch
import torch.nn as nn
import torch.nn.functional as F
from .base.basenet import BaseNet
from .base.InceptionBlock import GoogleNet


class Regression(nn.Module):
    """
    Pose regreesion module
    Args:
        read code precisely
        regid: id to map the length of the last dimension of the input feature maps
        with_embedding: if set True, output activations before pose regression
        together with regressed poses, otherwise only pose
    Return:
         xyz: global camera position
         wpqr: global camera orientation in quaternion
    """

    def __init__(self, regid, with_embedding=False):
        pass

    def forward(self, x):
        pass


class PoseNet(BaseNet):
    """
    PoseNet model in 2015ICCV
    Not Finished
    """

    def __init__(self, config, with_embedding=False):
        pass

    def forward(self, x):
        pass
