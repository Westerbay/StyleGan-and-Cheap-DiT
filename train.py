from deep.MappingNetwork import MappingNetwork
from deep.SynthesisNetwork import SynthesisNetwork
from deep.Discriminator import Discriminator

from data.ImageDataset import ImageDataset

from torch.utils.data import DataLoader
from torchvision.utils import save_image

import torch
import torch.nn.functional as F
import os


def save_samples(fake_imgs, epoch):
    os.makedirs("./samples", exist_ok=True)
    fake = (fake_imgs + 1) / 2
    filename = f"./samples/epoch_{epoch}.png"
    save_image(fake[:16], filename, nrow=4)
    print(f"[+] Saved samples: {filename}")

def train(
    generator, discriminator, mapping,
    loader, epochs,
    lr, betas,
    z_dim, batch_size,
    display_every, save_every,
    device="cuda"
):
    G = generator.to(device)
    M = mapping.to(device)
    D = discriminator.to(device)

    opt_G = torch.optim.Adam(list(G.parameters()) + list(M.parameters()), lr=lr, betas=(0.0, 0.99))
    opt_D = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.0, 0.99))

    for epoch in range(epochs):
        loss_D, loss_G = torch.tensor(0), torch.tensor(0)
        for real in loader:
            real = real.to(device)
            z = torch.randn(batch_size, z_dim).to(device)
            w = M(z)
            fake = G(w).detach()

            pred_fake = D(fake)
            pred_real = D(real)

            loss_D = F.softplus(pred_fake).mean() + F.softplus(-pred_real).mean()
            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()

            z = torch.randn(batch_size, z_dim).to(device)
            w = M(z)
            fake = G(w)
            pred_fake = D(fake)

            loss_G = F.softplus(-pred_fake).mean()

            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()

        print(f"[Epoch {epoch}]  D={loss_D.item():.4f}  G={loss_G.item():.4f}")
        if epoch % display_every == 0:            
            save_samples(fake, epoch)
        if epoch % save_every == 0:
            torch.save(G.state_dict(), "generator.pth")
            torch.save(D.state_dict(), "discriminator.pth")
            print(f"Epoch {epoch} saved !")


if __name__ == "__main__":
    w_dim, z_dim = 512, 512
    mapping = MappingNetwork(z_dim=z_dim, w_dim=w_dim)
    generator = SynthesisNetwork(w_dim=z_dim)
    discriminator = Discriminator()

    batch_size = 32
    dataset = ImageDataset("pokemon_preprocessed", (256, 256))
    loader = DataLoader(dataset=dataset, shuffle=True, batch_size=batch_size)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train(
        generator, discriminator, mapping,
        loader, epochs=10_000,
        lr=1e-4, betas=(0.0, 0.99),
        z_dim=z_dim, batch_size=batch_size,
        display_every=20, save_every=200,
        device=device
    )
