import torch.nn as nn
import torch


class PixelNorm(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + 1e-8)
    