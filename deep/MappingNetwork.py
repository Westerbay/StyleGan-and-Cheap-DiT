from deep.PixelNorm import PixelNorm

import torch.nn as nn


class MappingNetwork(nn.Module):

    def __init__(self, z_dim, w_dim, num_layers=8):
        super().__init__()
        self.pixel_norm = PixelNorm()
        layers = []
        for _ in range(num_layers-1):
            layers.append(nn.Linear(w_dim, w_dim))
            layers.append(nn.LeakyReLU(0.2))
            
        self.mlp = nn.Sequential(
            nn.Linear(z_dim, w_dim),
            nn.LeakyReLU(0.2),
            *layers
        )

    def forward(self, z):
        z = self.pixel_norm(z)
        return self.mlp(z)
