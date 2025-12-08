from deep.LatentDiffusionModel import LatentDiffusionModel

from data.ImageDataset import ImageDataset

from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

import torch.nn.functional as F
import torch
import os


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT = "artwork"   
OUTPUT_DIR = "samples_ldm"
IMG_SIZE = 256
LATENT_CH = 4

VAE_EPOCHS = 20
DIFFUSION_EPOCHS = 500
BATCH_SIZE = 32
LR_VAE = 1e-4
LR_DENOISER = 5e-5

TIME_STEPS = 1000
SAMPLE_EVERY = 10

D_MODEL = 512
NHEAD = 8
DEPTH = 8
PATCH_SIZE = 1


# Slide 65 du cours
def vae_kl_loss(x, pred, mu, logvar, kl_weight=1e-3):
    recon = F.mse_loss(pred, x)
    kl = 0.5 * torch.mean(mu ** 2 + torch.exp(logvar) - logvar - 1)
    return recon + kl_weight * kl, recon, kl # Beta VAE

def train_vae(vae, dataloader, epochs, lr, device):
    vae.train()
    optimizer = torch.optim.AdamW(vae.parameters(), lr=lr)

    for epoch in range(epochs + 1):
        pbar = tqdm(dataloader, desc=f"[VAE] Epoch {epoch}/{epochs}")
        for x in pbar:
            x = x.to(device)

            pred, mu, logvar = vae(x)
            loss, recon, kl = vae_kl_loss(x, pred, mu, logvar)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "recon": f"{recon.item():.4f}",
                "kl": f"{kl.item():.4f}",
            })
        if epoch % SAMPLE_EVERY == 0 or epoch == epochs:
            vae.eval()
            with torch.no_grad():
                x = next(iter(dataloader))
                x = x.to(device)
                pred, _, _ = vae(x)
                grid = torch.cat([
                    (x[:8] + 1) / 2.0,
                    (pred[:8] + 1) / 2.0
                ], dim=0)
                save_image(
                    grid, 
                    os.path.join(OUTPUT_DIR, "vae", f"vae_reconstruction_{epoch}.png"), 
                    nrow=8
                )
            vae.train()

def train_diffusion(ldm, dataloader, epochs, lr, device):
    # STOPPER VAE, Constant
    for p in ldm.vae.parameters():
        p.requires_grad = False

    ldm.denoiser.train()
    optimizer = torch.optim.AdamW(ldm.denoiser.parameters(), lr=lr)

    for epoch in range(epochs + 1):
        pbar = tqdm(dataloader, desc=f"[LDM] Epoch {epoch}/{epochs}")
        for x in pbar:
            x = x.to(device)

            loss = ldm.diffusion_training_step(x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
            })
        if epoch % SAMPLE_EVERY == 0:
            torch.save(ldm.state_dict(), "ldm.pth")
            ldm.eval()
            with torch.no_grad():
                samples = ldm.sample(batch_size=8)
                samples = (samples + 1) / 2.0
                save_image(samples, os.path.join(OUTPUT_DIR, "diffusion", f"epoch_{epoch}.png"), nrow=4)                
            ldm.denoiser.train()

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "vae"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "diffusion"), exist_ok=True)
    dataset = ImageDataset(DATA_ROOT, IMG_SIZE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    print("Dataset size:", len(dataset))

    ldm = LatentDiffusionModel(
        time_steps=TIME_STEPS,
        img_size=IMG_SIZE,
        device=DEVICE,
        d_model=D_MODEL,
        nhead=NHEAD,
        depth=DEPTH,
        patch_size=PATCH_SIZE,
        latent_ch=LATENT_CH,
    ).to(DEVICE)

    print("VAE Training")
    train_vae(ldm.vae, dataloader, VAE_EPOCHS, LR_VAE, DEVICE)

    print("Diffusion Training")
    train_diffusion(ldm, dataloader, DIFFUSION_EPOCHS, LR_DENOISER, DEVICE)

    torch.save(ldm.state_dict(), "ldm.pth")
    print("Model saved !")


if __name__ == "__main__":
    main()
