"""Pull a PCM waveform out of an uploaded audio or video file."""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import numpy as np

# Long movies / big uploads: keep a short front slice for demos.
MAX_SECONDS = float(os.getenv("MAX_AUDIO_SECONDS", "90"))
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(120 * 1024 * 1024)))


def to_pcm_wav(src: Path, dest: Path, rate: int = 16000) -> Path:
    """Decode the first audio stream to 16-bit mono WAV (trimmed for large media)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    size = src.stat().st_size
    trimmed = False

    if size > MAX_UPLOAD_BYTES or _looks_like_video(src):
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg:
            try:
                _ffmpeg_extract(ffmpeg, src, dest, rate, MAX_SECONDS)
                return dest
            except Exception:
                pass  # fall through to PyAV

    try:
        dest, trimmed = _pyav_extract(src, dest, rate, MAX_SECONDS)
    except Exception:
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise
        _ffmpeg_extract(ffmpeg, src, dest, rate, MAX_SECONDS)
        trimmed = True

    # Marker file for API notes (optional)
    if trimmed or size > 15 * 1024 * 1024:
        (dest.with_suffix(".trim.txt")).write_text(
            f"trimmed_to_seconds={MAX_SECONDS}; source_bytes={size}\n",
            encoding="utf-8",
        )
    return dest


def _looks_like_video(src: Path) -> bool:
    return src.suffix.lower() in {
        ".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".mpeg", ".mpg",
    }


def _ffmpeg_extract(ffmpeg: str, src: Path, dest: Path, rate: int, seconds: float) -> None:
    cmd = [
        ffmpeg, "-y",
        "-i", str(src),
        "-t", str(seconds),
        "-vn",
        "-ac", "1",
        "-ar", str(rate),
        "-f", "wav",
        str(dest),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not dest.exists() or dest.stat().st_size < 44:
        raise RuntimeError(proc.stderr[-500:] if proc.stderr else "ffmpeg failed")


def _pyav_extract(src: Path, dest: Path, rate: int, seconds: float) -> tuple[Path, bool]:
    import av
    import soundfile as sf

    container = av.open(str(src))
    trimmed = False
    try:
        stream = next((item for item in container.streams if item.type == "audio"), None)
        if stream is None:
            raise ValueError("no audio stream")
        resampler = av.audio.resampler.AudioResampler(format="s16", layout="mono", rate=rate)
        pieces: list[np.ndarray] = []
        max_samples = int(seconds * rate)
        total = 0
        for frame in container.decode(stream):
            for chunk in _samples(resampler.resample(frame)):
                remain = max_samples - total
                if remain <= 0:
                    trimmed = True
                    break
                if len(chunk) > remain:
                    chunk = chunk[:remain]
                    trimmed = True
                pieces.append(chunk)
                total += len(chunk)
            if total >= max_samples:
                trimmed = True
                break
        if total < max_samples:
            for chunk in _samples(resampler.resample(None)):
                remain = max_samples - total
                if remain <= 0:
                    break
                if len(chunk) > remain:
                    chunk = chunk[:remain]
                    trimmed = True
                pieces.append(chunk)
                total += len(chunk)
    finally:
        container.close()
    if not pieces:
        raise ValueError("no audio stream")
    audio = np.concatenate(pieces).astype(np.float32) / 32768.0
    sf.write(str(dest), audio, rate, subtype="PCM_16")
    return dest, trimmed


def _samples(frames) -> list[np.ndarray]:
    if frames is None:
        return []
    if not isinstance(frames, (list, tuple)):
        frames = [frames]
    out = []
    for frame in frames:
        out.append(np.asarray(frame.to_ndarray()).reshape(-1))
    return out
