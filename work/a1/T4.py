import torch
import torch.nn as nn


class ScaledShift(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(size))
        self.bias = nn.Parameter(torch.zeros(size))

    def forward(self, X):
        return X * self.scale + self.bias
