from deep.StyleBlock import StyleBlock
from deep.ToRGB import ToRGB

import torch.nn.functional as F
import torch.nn as nn
import torch


class SynthesisNetwork(nn.Module):

    CHANNELS = {
        4:   512,
        8:   512,
        16:  512,
        32:  256,
        64:  128,
        128: 64,
        256: 32,
    }
    RESOLUTIONS = [4, 8, 16, 32, 64, 128, 256]

    def __init__(self,  w_dim=512):
        super().__init__()
        self.w_dim = w_dim
        self.const_input = nn.Parameter(
            torch.randn(1, self.CHANNELS[4], 4, 4)
        )
        self.style_blocks = nn.ModuleDict()
        self.to_rgbs = nn.ModuleDict()
        in_ch = self.CHANNELS[4]
        self.style_blocks["4x4_1"] = StyleBlock(in_ch, in_ch, w_dim)
        self.style_blocks["4x4_2"] = StyleBlock(in_ch, in_ch, w_dim)
        self.to_rgbs["4"] = ToRGB(self.CHANNELS[4])        
        for res in self.RESOLUTIONS[1:]:
            out_ch = self.CHANNELS[res]
            self.style_blocks[f"{res}x_up"] = StyleBlock(in_ch, out_ch, w_dim)
            self.style_blocks[f"{res}x_2"] = StyleBlock(out_ch, out_ch, w_dim)
            self.to_rgbs[str(res)] = ToRGB(out_ch)
            in_ch = out_ch

    def forward(self, w):
        B = w.shape[0]
        x = self.const_input.repeat(B, 1, 1, 1)
        x = self.style_blocks["4x4_1"](x, w)
        x = self.style_blocks["4x4_2"](x, w)
        img = self.to_rgbs["4"](x)
        for res in self.RESOLUTIONS[1:]:
            img = F.interpolate(img, scale_factor=2, mode="nearest")
            x = F.interpolate(x, scale_factor=2, mode="nearest")
            x = self.style_blocks[f"{res}x_up"](x, w)
            x = self.style_blocks[f"{res}x_2"](x, w)
            rgb = self.to_rgbs[str(res)](x)
            img = img + rgb
        return img
