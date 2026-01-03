from deep.LatentDiffusionModel import LatentDiffusionModel

from data.ImageDataset import ImageDataset

from torch.utils.data import DataLoader
from torchvision.utils import save_image

import torch.nn.functional as F
import torch
import os


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = "generated"

IMG_SIZE = 256
LATENT_CH = 4

TIME_STEPS = 1000
D_MODEL = 512
NHEAD = 8
DEPTH = 8
PATCH_SIZE = 1

def generate(ldm):
    ldm.eval()    
    with torch.no_grad():
        samples = ldm.sample(batch_size=1)
        samples = (samples + 1) / 2.0
        save_image(samples, os.path.join(OUTPUT_DIR, "output_ldm.png"), nrow=4)                

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
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
    state_dict = torch.load("models/ldm.pth", map_location="cpu")
    ldm.load_state_dict(state_dict)
    generate(ldm)

if __name__ == "__main__":
    main()
