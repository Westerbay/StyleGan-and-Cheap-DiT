from deep.Generator import Generator
from deep.Discriminator import Discriminator

from data.ImageDataset import ImageDataset

from torch.utils.data import DataLoader
from torchvision.utils import save_image

import torch
import torch.nn.functional as F
import numpy as np
import os


def save_samples(fake_imgs):
    os.makedirs("./generated", exist_ok=True)
    fake = (fake_imgs + 1) / 2 # Image [-1, 1]
    filename = f"./generated/gan_output.png"
    save_image(fake[:16], filename, nrow=4)
    print(f"[+] Saved samples: {filename}")

def generate(
    generator, device
):
    G = generator.to(device)
    G.eval()

    with torch.no_grad():
        fake = G(16, device).detach()
        save_samples(fake)


if __name__ == "__main__":
    w_dim, z_dim = 512, 512
    generator = Generator(z_dim=z_dim, w_dim=w_dim)
    state_dict = torch.load("models/generator.pth", map_location="cpu")
    generator.load_state_dict(state_dict)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    generate(
        generator, device
    )
