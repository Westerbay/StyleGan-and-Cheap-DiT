import torch.nn as nn


class DiscriminatorBlock(nn.Module):

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.activation = nn.LeakyReLU(0.2)
        self.down = nn.AvgPool2d(2)

    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        return self.down(x)
    