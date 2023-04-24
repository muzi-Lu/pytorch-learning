import torch
import torch.nn as nn

class InceptionBlock(nn.Module):
    def __init__(self, block_in, branch_out):
        super(InceptionBlock, self).__init__()
        self.branch_x1 = nn.Sequential(
            nn.Conv2d(block_in, branch_out[0], kernel_size=1),
            nn.ReLU()
        )
        self.branch_x3 = nn.Sequential(
            nn.Conv2d(block_in, branch_out[1][0], kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(branch_out[1][0], branch_out[1][1], kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.branch_x5 = nn.Sequential(
            nn.Conv2d(block_in, branch_out[2][0], kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(branch_out[2][0], branch_out[2][1], kernel_size=5, padding=2),
            nn.ReLU()
        )
        self.branch_proj = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(block_in, branch_out[3], kernel_size=1),
            nn.ReLU()
        )

        def forward(self, x):
            br_out0 = self.branch_x1(x)
            br_out1 = self.branch_x3(x)
            br_out2 = self.branch_x5(x)
            br_out3 = self.branch_proj(x)
            output = [br_out0, br_out1, br_out2, br_out3]
            return torch.cat(output, 1)

class GoogleNet(nn.Module):
    def __init__(self, with_aux=False):
        super(GoogleNet, self).__init__()
        self.with_aux = with_aux
        self.localrespnorm = nn.LocalResponseNorm()
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)