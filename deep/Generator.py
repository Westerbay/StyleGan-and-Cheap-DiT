from deep.MappingNetwork import MappingNetwork
from deep.SynthesisNetwork import SynthesisNetwork

import torch.nn as nn
import torch


class Generator(nn.Module):

    def __init__(self, z_dim, w_dim, mapping_layers=8):
        super().__init__()
        self.z_dim = z_dim
        self.mapping = MappingNetwork(z_dim, w_dim, mapping_layers)
        self.synthesis = SynthesisNetwork(w_dim)

    def forward(self, n, device="cpu"):
        z = torch.randn(n, self.z_dim, device=device)
        w = self.mapping(z)
        return self.synthesis(w)
