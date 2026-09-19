"""Reject missing, empty or unhydrated LFS checkpoints before a container build."""
from pathlib import Path


for name in ("generator.pth", "discriminator.pth", "ldm.pth"):
    path = Path("models") / name
    if not path.is_file() or path.stat().st_size < 1024:
        raise SystemExit(f"Missing model weights: {path}. Run git lfs pull first.")
    with path.open("rb") as checkpoint:
        if checkpoint.read(80).startswith(b"version https://git-lfs.github.com/spec/v1"):
            raise SystemExit(f"LFS pointer found: {path}. Run git lfs pull first.")
    print(f"{path}: {path.stat().st_size:,} bytes")
