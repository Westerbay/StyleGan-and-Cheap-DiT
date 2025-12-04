from deep.DiscriminatorBlock import DiscriminatorBlock

import torch.nn as nn


class Discriminator(nn.Module):

    CHANNELS = {
        4:   512,
        8:   512,
        16:  512,
        32:  256,
        64:  128,
        128: 64,
        256: 32,
    }
    RESOLUTIONS = [8, 16, 32, 64, 128, 256]

    def __init__(self):
        super().__init__()
        blocks, in_ch = [], 3
        for res in reversed(self.RESOLUTIONS):
            out_ch = self.CHANNELS[res]
            blocks.append(DiscriminatorBlock(in_ch, out_ch))
            in_ch = out_ch
        self.feature = nn.Sequential(*blocks)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.CHANNELS[4] * 4 * 4, 1)
        )

    def forward(self, x):
        x = self.feature(x)
        return self.classifier(x)
