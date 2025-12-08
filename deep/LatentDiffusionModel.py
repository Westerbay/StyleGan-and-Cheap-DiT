from deep.VAE import VAE
from deep.DDPM import DDPM
from deep.TransformerDenoiser import TransformerDenoiser

import torch.nn as nn
import torch


class LatentDiffusionModel(nn.Module):

    def __init__(
        self,
        time_steps,
        img_size,
        device,
        d_model,
        nhead,
        depth,
        patch_size,
        latent_ch,
    ):
        super().__init__()
        self.device = device
        self.img_size = img_size

        self.latent_ch = latent_ch
        self.latent_size = img_size // 8

        self.vae = VAE(in_ch=3, latent_ch=latent_ch).to(device)
        self.ddpm = DDPM(time_steps=time_steps, device=device)

        self.denoiser = TransformerDenoiser(
            img_size=self.latent_size,
            patch_size=patch_size,
            time_steps=time_steps,
            in_ch=latent_ch,
            d_model=d_model,
            nhead=nhead,
            depth=depth,
        ).to(device)

        self.time_steps = time_steps
        self.mse_loss = nn.MSELoss()

    def encode(self, x, detach=False):
        mu, logvar = self.vae.encode(x)
        z = self.vae.reparameterize(mu, logvar)
        if detach:
            z = z.detach()
        return z, mu, logvar

    def decode(self, z):
        return self.vae.decode(z)

    def diffusion_training_step(self, x):
        """
        x : images dans [-1,1]
        VAE gelé -> on ne propage pas dans le VAE.
        """
        B = x.size(0)
        device = self.device

        with torch.no_grad():
            z0, _, _ = self.encode(x)

        t = torch.randint(0, self.time_steps, (B,), device=device).long()
        zt, noise = self.ddpm.forward_sample(z0, t)

        pred = self.denoiser(zt, t)
        return self.mse_loss(noise, pred)

    @torch.no_grad()
    def sample(self, batch_size):
        self.denoiser.eval()
        self.vae.eval()

        z = torch.randn(
            batch_size,
            self.latent_ch,
            self.latent_size,
            self.latent_size,
            device=self.device,
        )

        for i in reversed(range(self.time_steps)):
            t = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
            z = self.ddpm.backward_sample(self.denoiser, z, t)

        x = self.decode(z)  # [-1,1]
        return x