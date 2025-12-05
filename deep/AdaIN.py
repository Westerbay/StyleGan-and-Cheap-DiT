import torch.nn as nn
import torch


class AdaIN(nn.Module):

    def __init__(self, out_ch, w_dim):
        super().__init__()
        self.linear = nn.Linear(w_dim, out_ch * 2)

    def forward(self, x, w):
        B, C, H, W = x.shape
        y = self.linear(w) # Styles
        scale, bias = y[:, :C], y[:, C:]
        scale = scale.view(B, C, 1, 1)
        bias = bias.view(B, C, 1, 1)
        mean = torch.mean(x, dim=[2, 3], keepdim=True)
        std = torch.sqrt(torch.var(x, dim=[2,3], keepdim=True) + 1e-8)
        x_norm = (x - mean) / std
        return scale * x_norm + bias