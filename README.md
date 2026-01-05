# StyleGAN & Cheap DiT (Minimal Latent DiT)

This repository is a **pedagogical and experimental project** exploring two approaches for **256×256 image generation**:

1. A **minimal StyleGAN1** implementation (mapping network, AdaIN, synthesis network)
2. A **minimal / “cheap” DiT** (*Diffusion Transformer*)  
   - first attempted directly in pixel space (too slow, hard to converge)
   - then moved to **latent space** using an **ultra-minimal VAE**

The focus is on understanding **training stability**, **mode collapse**, **convergence issues**, and **compute trade-offs**, rather than pushing state-of-the-art image quality.

---

## Project Goal

- **Generate 256×256 images** using a homemade **StyleGAN1**
- Encounter classic GAN problems:
  - **training instability**
  - **mode collapse**
- Explore a modern alternative:
  - a **minimal DiT** (diffusion + transformer) in pixel space  
  - ➜ **problem**: very slow training and poor convergence
- Final approach:
  - **Latent DiT** (diffusion in latent space)
  - powered by an **ultra-minimal VAE**
  - ➜ much cheaper computation and faster experimentation

---

## Visual Overview

### StyleGAN, trained with landscapes

From `samples_gan/`:

![StyleGAN epoch 400](samples_gan/epoch_400.png)

### Latent Diffusion / DiT — diffusion & VAE, trained with paintings

From `samples_ldm/`:

**Diffusion samples**

![LDM epoch 200](samples_ldm/diffusion/epoch_200.png)

**VAE reconstructions**

![VAE reconstruction 20](samples_ldm/vae/vae_reconstruction_20.png)

---

## Repository Structure

```
.
├── api/                      # (Optional) service/session wrappers
│   ├── service.py
│   └── session.py
├── data/                     # dataset + preprocessing
│   ├── ImageDataset.py
│   └── ImagePreprocess.py
├── deep/                     # core DL building blocks
│   ├── Generator.py
│   ├── Discriminator.py
│   ├── MappingNetwork.py
│   ├── SynthesisNetwork.py
│   ├── StyleBlock.py
│   ├── AdaIN.py
│   ├── PixelNorm.py
│   ├── MinibatchStdDev.py
│   ├── DDPM.py
│   ├── VAE.py
│   ├── TransformerDenoiser.py
│   └── LatentDiffusionModel.py
├── models/                   # saved models
│   ├── generator.pth
│   ├── discriminator.pth
│   └── ldm.pth
├── resources/                # reference papers (PDFs)
├── samples_gan/              # GAN training samples
├── samples_ldm/              # diffusion + VAE samples
├── preprocess.py             # dataset preprocessing
├── train_gan.py              # StyleGAN training
├── generate_gan.py           # GAN inference
├── train_latentdiff.py       # Latent Diffusion training
├── generate_ldm.py           # Latent Diffusion inference
└── api_ldm.py                # LDM API
```

---

## High-Level Technical Overview

### 1) Minimal StyleGAN1 (256×256)

- **Mapping Network**: maps noise `z` → style vector `w`
- **Synthesis Network**: progressively builds the image
- **AdaIN**: injects style information at multiple layers
- **Stabilization tricks** (kept minimal on purpose):
  - `PixelNorm`
  - `MinibatchStdDev` in the discriminator
  - careful LR and architecture choices

The goal is clarity and compactness, at the cost of **higher sensitivity** to hyperparameters and **mode collapse**.

---

### 2) “Cheap DiT” → Latent DiT

- Initial idea: diffusion + transformer directly on 256×256 images  
  ❌ extremely expensive and unstable
- Final approach: **Latent Diffusion**
  - images are compressed using a **very small VAE**
  - diffusion operates in latent space
  - denoising is handled by a **TransformerDenoiser**

**Advantages**
- drastically reduced compute
- faster iterations
- easier experimentation on consumer GPUs

**Limitations**
- minimal VAE ⇒ imperfect reconstructions
- minimal DiT ⇒ quality heavily depends on tuning

---

## User Interface (Docker) 

You can use the user interface to try the DiT model.

```bash
docker pull registry.gitlab.com/westerbay/stylegan-and-cheap-dit/stylegan-and-cheap-dit:v0.3

# With CPU
docker run -p 5173:5173 registry.gitlab.com/westerbay/stylegan-and-cheap-dit/stylegan-and-cheap-dit:v0.3
# With CUDA
docker run -p 5173:5173 --gpus all registry.gitlab.com/westerbay/stylegan-and-cheap-dit/stylegan-and-cheap-dit:v0.3
```

---

## Installation

Recommended: Python ≥ 3.10 (repo already uses 3.12 bytecode).

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch torchvision    
pip install numpy pillow tqdm
pip install fastapi uvicorn # Optional
```

Or

```bash
pip install -r requirements.txt
```


---

## Usage

### 1) Preprocess the dataset

```bash
python preprocess.py
```

Typical steps:
- Resize / Normalize images
- Color Jitter
- Transpose / Rotate

---

### 2) Train StyleGAN

```bash
python train_gan.py
```

- models → `models/`
- visual samples → `samples_gan/`

---

### 3) Generate images with StyleGAN

```bash
git lfs install   # For pretrained model
git lfs pull      # For pretrained model

python generate_gan.py
```

- visual samples → `generated/`

---

### 4) Train Latent Diffusion (VAE + diffusion)

```bash
python train_latentdiff.py
```

- VAE reconstructions → `samples_ldm/vae/`
- diffusion samples → `samples_ldm/diffusion/`
- model weights → `models/ldm.pth`

---

### 5) Generate images with Latent Diffusion

```bash
git lfs install   # For pretrained model
git lfs pull      # For pretrained model

python generate_ldm.py
```

- visual samples → `generated/`

---

## Training Issues & Observations

### Mode Collapse (GAN)

Symptoms:
- low diversity
- repeated patterns
- stalled visual progress

---

### Slow / Unstable Convergence (Pixel-space DiT)

- diffusion at 256×256 is compute-heavy
- naive transformer denoisers diverge easily
- latent diffusion drastically simplifies the problem

---

## References (`resources/`)

- *A Style-Based Generator Architecture for Generative Adversarial Networks*  
  (Karras et al., CVPR 2019)
- *Denoising Diffusion Probabilistic Models*  
  (Ho et al., NeurIPS 2020)
- NVIDIA StyleGAN slides / documentation

---

