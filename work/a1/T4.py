import torch.nn as nn


class ScaledShift(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.scale = nn.Parameter(torch.randn(size))
        self.bias = nn.Prameter(torch.zeros(size))

    def forward(self, X):
        return X * self.scale + self.bias
