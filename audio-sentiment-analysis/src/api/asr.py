"""Turn a recording into words and a language guess."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def _model():
    from faster_whisper import WhisperModel

    return WhisperModel("small", device="cpu", compute_type="int8")


def transcribe(path: Path) -> dict:
    segments, info = _model().transcribe(str(path), beam_size=1)
    text = " ".join(part.text.strip() for part in segments).strip()
    probability = getattr(info, "language_probability", None)
    return {
        "text": text,
        "language": (info.language or "").lower(),
        "probability": float(probability) if probability is not None else None,
    }
