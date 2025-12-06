import torch.nn as nn
import torch


class TransformerDenoiser(nn.Module):

    MLP_RATIO = 4
    ACTIVATION_TRANSFORMER = "gelu"
    
    def __init__(
        self, 
        img_size, 
        patch_size,
        time_steps,
        in_ch,
        d_model,
        nhead,
        depth
    ):
        super().__init__()

        assert img_size % patch_size == 0

        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        self.patch_emb = nn.Conv2d(in_ch, d_model, patch_size, patch_size)
        self.pos_patch_emb = nn.Embedding(self.num_patches, d_model)
        self.time_emb = nn.Embedding(time_steps, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * self.MLP_RATIO,
            activation=self.ACTIVATION_TRANSFORMER,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=depth
        )

        self.reconstruct = nn.ConvTranspose2d(
            d_model, in_ch, patch_size, patch_size
        )

    def forward(self, x, t):
        h = self.patch_emb(x)
        B, d_model, H_p, W_p = h.shape
        h = h.flatten(2) # B, d_model, H_p*W_p
        h = h.transpose(1, 2)

        positions = torch.arange(self.num_patches, device=x.device)
        pos_patch_emb = self.pos_patch_emb(positions)[None, :, :]
        h = h + pos_patch_emb

        time_emb = self.time_emb(t) # B, d_model
        h = h + time_emb[:, None, :]

        h = self.transformer(h)

        h = h.transpose(1, 2).view(B, d_model, H_p, W_p)
        out = self.reconstruct(h)
        return out
