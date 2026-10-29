"""Shared paths and small helpers."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODELS = ROOT / "models"
UPLOADS = ROOT / "uploads"
CONFIGS = ROOT / "configs"
FIGURES = ROOT / "figures"

for p in (DATA, MODELS, UPLOADS, FIGURES, DATA / "text", DATA / "audio" / "crema_d", DATA / "audio" / "banglaser"):
    p.mkdir(parents=True, exist_ok=True)


def device_str() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"
