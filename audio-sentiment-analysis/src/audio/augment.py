"""Waveform then spectrogram-level augmentations (thesis order)."""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np


def load_wav(path: Path, sr: int = 16000) -> Tuple[np.ndarray, int]:
    """Load mono float32 audio. Prefer soundfile; fall back to librosa for odd CREMA-D files."""
    y = None
    file_sr = None
    try:
        import soundfile as sf

        y, file_sr = sf.read(str(path), always_2d=False)
    except Exception:
        y = None

    if y is None:
        import librosa

        y, file_sr = librosa.load(str(path), sr=None, mono=True)

    y = np.asarray(y, dtype=np.float32)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if len(y) == 0:
        raise ValueError(f"empty audio: {path}")
    if file_sr != sr:
        n_out = max(1, int(len(y) * sr / float(file_sr)))
        x_old = np.linspace(0.0, 1.0, num=len(y), endpoint=False)
        x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
        y = np.interp(x_new, x_old, y).astype(np.float32)
    return y, sr


def temporal_shift(y: np.ndarray, max_frac: float = 0.2) -> np.ndarray:
    n = len(y)
    delta = int(np.random.uniform(-max_frac, max_frac) * n)
    out = np.zeros_like(y)
    if delta >= 0:
        out[delta:] = y[: n - delta]
    else:
        out[: n + delta] = y[-delta:]
    return out


def noise_inject(y: np.ndarray, alpha_low: float = 0.005, alpha_high: float = 0.015) -> np.ndarray:
    alpha = float(np.random.uniform(alpha_low, alpha_high))
    noise = np.random.randn(len(y)).astype(np.float32) * (alpha * (y.std() + 1e-8))
    return y + noise


def time_stretch(y: np.ndarray, sr: int) -> np.ndarray:
    import librosa

    beta = float(np.random.uniform(0.9, 1.1))
    return librosa.effects.time_stretch(y, rate=beta).astype(np.float32)


def pitch_shift(y: np.ndarray, sr: int, semitones=(-3.0, -1.5, 0.0, 1.5, 3.0)) -> np.ndarray:
    import librosa

    n = float(np.random.choice(semitones))
    if n == 0:
        return y
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=n).astype(np.float32)


def augment_once(y: np.ndarray, sr: int, dataset: str = "crema_d") -> np.ndarray:
    """Apply staged pipeline once. CREMA-D uses milder pitch by default."""
    y = temporal_shift(y, max_frac=0.2)
    y = noise_inject(y)
    y = time_stretch(y, sr)
    semis = (-2.0, -1.0, 0.0, 1.0, 2.0) if dataset == "crema_d" else (-3.0, -1.5, 0.0, 1.5, 3.0)
    y = pitch_shift(y, sr, semis)
    return y


def save_wav(path: Path, y: np.ndarray, sr: int = 16000) -> None:
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), y, sr)
