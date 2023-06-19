import torch
import torch.nn as nn

class Linear(nn.Module):
    def __init__(self, in_features, out_features):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.w = nn.Parameter(torch.randn(self.in_features, self.out_features))
        self.b = nn.Parameter(torch.randn(self.out_features))

    def forward(self, x):
        x = x.mm(self.w) # x @(self.w)
        return x + self.b.expand_as(x)

layer = Linear(4, 3)
input = torch.rand(2, 4)
output = layer(input)
print(output)

for name, parameter in layer.named_parameters():
    print(name, parameter)