from api.session import GenerationSession
from deep.LatentDiffusionModel import LatentDiffusionModel

import torch


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 256
LATENT_CH = 4
TIME_STEPS = 1000
D_MODEL = 512
NHEAD = 8
DEPTH = 8
PATCH_SIZE = 1
MODEL_PATH = "models/ldm.pth"

class LDMService:
    
    def __init__(self):
        self.ldm = None
        self.session: GenerationSession | None = None

    def load_model(self):
        self.ldm = LatentDiffusionModel(
            time_steps=TIME_STEPS,
            img_size=IMG_SIZE,
            device=DEVICE,
            d_model=D_MODEL,
            nhead=NHEAD,
            depth=DEPTH,
            patch_size=PATCH_SIZE,
            latent_ch=LATENT_CH,
        ).to(DEVICE)

        state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
        self.ldm.load_state_dict(state_dict)
        self.ldm.eval()

    def start(self, batch_size: int):
        self.session = GenerationSession(self.ldm, batch_size)
        return {
            "time_steps": self.ldm.time_steps,
            "batch_size": batch_size,
        }

    def step(self):
        if self.session is None:
            raise RuntimeError("No active session")

        return self.session.next()
