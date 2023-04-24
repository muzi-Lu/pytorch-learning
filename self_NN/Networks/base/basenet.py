import torch
import torch.nn as nn
from collections import OrderedDict

class BaseNet(nn.Module):
    def __init__(self, config):
        super(BaseNet, self).__init__()
        self.config = config

