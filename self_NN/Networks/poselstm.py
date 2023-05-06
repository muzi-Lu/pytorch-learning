import torch
import torch.nn as nn
import torch.nn.functional as F
from base.basenet import BaseNet
from base.InceptionBlock import GoogleNet
class FourDirectionalLSTM(nn.Module):
    def __init__(self, seq_size, origin_feat_size, hidden_size):
        super(FourDirectionalLSTM, self).__init__()

    def init_hidden_(self, batch_size, device):
        pass