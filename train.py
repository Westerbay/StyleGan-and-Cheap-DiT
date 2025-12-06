from deep.Generator import Generator
from deep.Discriminator import Discriminator

from data.ImageDataset import ImageDataset

from torch.utils.data import DataLoader
from torchvision.utils import save_image

import torch
import torch.nn.functional as F
import numpy as np
import os


def save_samples(fake_imgs, epoch):
    os.makedirs("./samples", exist_ok=True)
    fake = (fake_imgs + 1) / 2 # Image [-1, 1]
    filename = f"./samples/epoch_{epoch}.png"
    save_image(fake[:16], filename, nrow=4)
    print(f"[+] Saved samples: {filename}")

def train(
    generator, discriminator,
    loader, epochs,
    lr, betas,
    display_every, save_every,
    device="cuda"
):
    G = generator.to(device)
    D = discriminator.to(device)
    G.train()
    D.train()

    opt_G = torch.optim.Adam(G.parameters(), lr=lr, betas=betas)
    opt_D = torch.optim.Adam(D.parameters(), lr=lr, betas=betas)

    for epoch in range(epochs):
        running_D, running_G = [], []
        for real in loader:
            batch_size = real.shape[0]
            real = real.to(device)
            fake = G(batch_size, device).detach()

            pred_fake = D(fake)
            pred_real = D(real)

            loss_D = F.softplus(pred_fake).mean() + F.softplus(-pred_real).mean()
            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()
            running_D.append(loss_D.item())

            fake = G(batch_size, device)
            pred_fake = D(fake)

            loss_G = F.softplus(-pred_fake).mean()

            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()
            running_G.append(loss_G.item())

        print(f"[Epoch {epoch}]  D={np.mean(running_D):.4f}  G={np.mean(running_G):.4f}")
        if epoch % display_every == 0:            
            save_samples(fake, epoch)
        if epoch % save_every == 0:
            torch.save(G.state_dict(), "generator.pth")
            torch.save(D.state_dict(), "discriminator.pth")
            print(f"Epoch {epoch} saved !")


if __name__ == "__main__":
    w_dim, z_dim = 512, 512
    generator = Generator(z_dim=z_dim, w_dim=w_dim)
    discriminator = Discriminator()

    batch_size = 32
    dataset = ImageDataset("landscapes", 256)
    loader = DataLoader(dataset=dataset, shuffle=True, batch_size=batch_size)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train(
        generator, discriminator,
        loader, epochs=10_000,
        lr=1e-4, betas=(0.0, 0.99),
        display_every=20, save_every=200,
        device=device
    )
