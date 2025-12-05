from deep.AdaIN import AdaIN

import torch.nn as nn
import torch


class StyleBlock(nn.Module):

    def __init__(self, in_ch, out_ch, w_dim):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.bias = nn.Parameter(torch.zeros(1, out_ch, 1, 1))
        self.adain = AdaIN(out_ch, w_dim)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x, w):
        B, C, H, W = x.shape
        noise = torch.randn(B, 1, H, W, device=x.device)
        x = self.conv(x)
        x = x + noise * self.bias
        x = self.adain(x, w)
        return self.activation(x)
