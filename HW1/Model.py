import torch.nn as nn

class COVID19Model(nn.Module):
    def __init__(self, input_dim):
        super(COVID19Model, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.layers(x)
        x = x.squeeze(1) #(B, 1) --> (B)
        return x