import torch.nn as nn
import torch


class PixelNorm(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, z):
        return z / torch.sqrt(torch.mean(z ** 2, dim=1, keepdim=True) + 1e-8)
    