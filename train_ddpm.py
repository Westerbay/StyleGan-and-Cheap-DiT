from deep.DDPM import DDPM
from deep.TransformerDenoiser import TransformerDenoiser

from data.ImageDataset import ImageDataset

from torch.utils.data import DataLoader
from torchvision.utils import save_image

import torch.nn as nn
import torch
import numpy as np
import os


def save_samples(fake_imgs, epoch):
    os.makedirs("./samples_ddpm", exist_ok=True)
    fake = (fake_imgs + 1) / 2 # Image [-1, 1]
    filename = f"./samples_ddpm/epoch_{epoch}.png"
    save_image(fake[:16], filename, nrow=4)
    print(f"[+] Saved samples: {filename}")

def train(
    transformer, ddpm, loader, 
    epochs, lr,
    img_size, time_steps,
    save_every, sample_every,
    device
):
    transformer = transformer.to(device)
    ddpm = ddpm.to(device)
    optimizer = torch.optim.AdamW(transformer.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        global_loss = 0
        for x_0 in loader:
            x_0 = x_0.to(device)
            B, C, H, W = x_0.shape
            t = torch.randint(0, time_steps, (B,), device=device)
            x_t, noise = ddpm.forward_sample(x_0, t)
            noise_pred = transformer(x_t, t)
            loss = criterion(noise_pred, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            global_loss += loss.item()        
        avg_loss = global_loss / len(loader)
        print(f"[Epoch {epoch}] Average loss per batch : {avg_loss}")
        if epoch % sample_every == 0:
            with torch.no_grad():
                samples = ddpm.sample(transformer, img_size=img_size, batch_size=16)
                save_samples(samples, epoch)
        if epoch % save_every == 0:
            torch.save(transformer.state_dict(), "transformer.pth")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    img_size = 256
    dataset = ImageDataset("artwork", img_size)
    loader = DataLoader(dataset, shuffle=True, batch_size=128)
    print("Dataset length:", len(dataset))

    time_steps = 1000
    transformer = TransformerDenoiser(
        img_size=img_size,
        patch_size=16,
        time_steps=time_steps,
        in_ch=3,
        d_model=512,
        nhead=8,
        depth=8
    )
    ddpm = DDPM(time_steps, device)
    train(
        transformer, ddpm, loader,
        epochs=500, lr=1e-4, 
        img_size=img_size, time_steps=time_steps,
        save_every=50, sample_every=10,
        device=device
    )
