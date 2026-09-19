"""Load migrated checkpoints and exercise one CPU inference step per model."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from deep.Generator import Generator
from deep.Discriminator import Discriminator
from api.service import LDMService

torch.set_num_threads(2)
with torch.inference_mode():
    generator = Generator(z_dim=512, w_dim=512).eval()
    generator.load_state_dict(torch.load("models/generator.pth", map_location="cpu", weights_only=True))
    images = generator(1, "cpu")
    assert images.shape == (1, 3, 256, 256)
    assert torch.isfinite(images).all()
    discriminator = Discriminator().eval()
    discriminator.load_state_dict(torch.load("models/discriminator.pth", map_location="cpu", weights_only=True))
    assert torch.isfinite(discriminator(images)).all()
    service = LDMService()
    service.load_model()
    service.start(1)
    step, image = service.step()
    assert step == 1 and image.shape == (1, 3, 256, 256)
    assert torch.isfinite(image).all()
print("All three checkpoints loaded; GAN inference and one diffusion step passed.")
