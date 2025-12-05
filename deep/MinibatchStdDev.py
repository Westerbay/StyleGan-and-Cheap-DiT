import torch.nn as nn
import torch


class MinibatchStdDev(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        B, C, H, W = x.shape
        if B < 2:
            return x
        std = x.std(dim=0, unbiased=False)
        std = std.mean().view(1, 1, 1, 1)
        std_map = std.repeat(B, 1, H, W)
        return torch.cat([x, std_map], dim=1)
