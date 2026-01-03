import os
from api.service import LDMService

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from torchvision.utils import save_image
from contextlib import asynccontextmanager

import torch
import uvicorn
import base64
import io


# =====================
# Config
# =====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = "generated"
OUTPUT_IMAGE = os.path.join(OUTPUT_DIR, "step.png")

IMG_SIZE = 256
LATENT_CH = 4
TIME_STEPS = 1000
D_MODEL = 512
NHEAD = 8
DEPTH = 8
PATCH_SIZE = 1
MODEL_PATH = "models/ldm.pth"


@asynccontextmanager
async def lifespan(app: FastAPI):
    service.load_model()
    yield
    service.session = None
    service.ldm = None


# =====================
# App
# =====================
app = FastAPI(
    title="Step-by-step LDM API",
    lifespan=lifespan
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
service = LDMService()

os.makedirs(OUTPUT_DIR, exist_ok=True)

@app.get("/start")
def start(batch_size: int = 1):
    return service.start(batch_size)

@app.get("/step")
def step():
    try:
        step_idx, x = service.step()
        if x is None:
            return {"message": "Generation finished"}

        x = (x + 1) / 2.0
        images = []
        for i in range(x.size(0)):
            buffer = io.BytesIO()
            save_image(x[i], buffer, format="PNG")
            buffer.seek(0)

            images.append(
                base64.b64encode(buffer.read()).decode("utf-8")
            )

        return JSONResponse({
            "step": step_idx,
            "images": images,
        })

    except RuntimeError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7050)
