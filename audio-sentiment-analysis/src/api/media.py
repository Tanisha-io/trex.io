"""Pull a PCM waveform out of an uploaded audio or video file."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def to_pcm_wav(src: Path, dest: Path, rate: int = 16000) -> Path:
    """Decode the first audio stream to 16-bit mono WAV."""
    import av
    import soundfile as sf

    container = av.open(str(src))
    try:
        stream = next((item for item in container.streams if item.type == "audio"), None)
        if stream is None:
            raise ValueError("no audio stream")
        resampler = av.audio.resampler.AudioResampler(format="s16", layout="mono", rate=rate)
        pieces: list[np.ndarray] = []
        for frame in container.decode(stream):
            pieces.extend(_samples(resampler.resample(frame)))
        pieces.extend(_samples(resampler.resample(None)))
    finally:
        container.close()
    if not pieces:
        raise ValueError("no audio stream")
    audio = np.concatenate(pieces).astype(np.float32) / 32768.0
    dest.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dest), audio, rate, subtype="PCM_16")
    return dest


def _samples(frames) -> list[np.ndarray]:
    if frames is None:
        return []
    if not isinstance(frames, (list, tuple)):
        frames = [frames]
    out = []
    for frame in frames:
        out.append(np.asarray(frame.to_ndarray()).reshape(-1))
    return out
