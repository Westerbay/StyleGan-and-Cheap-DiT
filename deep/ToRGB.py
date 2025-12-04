import torch.nn as nn


class ToRGB(nn.Module):

    def __init__(self, in_ch, out_ch=3):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        return self.conv(x)
