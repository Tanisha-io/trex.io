"""Turn a recording into words and a language guess."""
from __future__ import annotations

import os
import sys
import wave
from functools import lru_cache
from pathlib import Path

# Light default (was faster-whisper "small").
_DEFAULT_MODEL = "base"
# Cap how much audio ASR listens to (large videos already trimmed in media.py).
_ASR_SECONDS = float(os.getenv("ASR_MAX_SECONDS", "60"))


def _model_name() -> str:
    return (os.getenv("WHISPER_MODEL") or _DEFAULT_MODEL).strip() or _DEFAULT_MODEL


def _backend() -> str:
    explicit = (os.getenv("WHISPER_BACKEND") or "").strip().lower()
    if explicit in {"faster", "openai", "google"}:
        return explicit
    # Windows: prefer Google web ASR first (Torch/faster-whisper often broken locally).
    return "google" if sys.platform.startswith("win") else "faster"


@lru_cache(maxsize=1)
def _faster_model():
    from faster_whisper import WhisperModel

    device = (os.getenv("WHISPER_DEVICE") or "cpu").strip() or "cpu"
    compute = (os.getenv("WHISPER_COMPUTE_TYPE") or "int8").strip() or "int8"
    return WhisperModel(_model_name(), device=device, compute_type=compute)


@lru_cache(maxsize=1)
def _openai_model():
    import whisper

    return whisper.load_model(_model_name())


def transcribe(path: Path, language_hint: str | None = None) -> dict:
    """
    Try backends in order. Returns text + language.
    Raises the last error if every backend fails.
    """
    errors: list[Exception] = []
    order = []
    preferred = _backend()
    if preferred == "faster":
        order = ["faster", "openai", "google"]
    elif preferred == "openai":
        order = ["openai", "google", "faster"]
    else:
        order = ["google", "openai", "faster"]

    clipped = _clip_wav(path, _ASR_SECONDS)
    for name in order:
        try:
            if name == "faster":
                return _transcribe_faster(clipped, language_hint)
            if name == "openai":
                return _transcribe_openai(clipped, language_hint)
            if name == "google":
                return _transcribe_google(clipped, language_hint)
        except Exception as exc:  # noqa: BLE001 — try next backend
            errors.append(exc)
            continue
    if errors:
        raise errors[-1]
    raise RuntimeError("No ASR backend available")


def _clip_wav(path: Path, seconds: float) -> Path:
    """Write a short front slice so ASR stays cheap on long media."""
    try:
        with wave.open(str(path), "rb") as src:
            rate = src.getframerate()
            nchan = src.getnchannels()
            width = src.getsampwidth()
            max_frames = int(rate * seconds)
            frames = src.readframes(max_frames)
            total = src.getnframes()
        if total <= max_frames:
            return path
        out = path.with_name(path.stem + f".asr{int(seconds)}.wav")
        with wave.open(str(out), "wb") as dest:
            dest.setnchannels(nchan)
            dest.setsampwidth(width)
            dest.setframerate(rate)
            dest.writeframes(frames)
        return out
    except Exception:
        return path


def _transcribe_faster(path: Path, language_hint: str | None) -> dict:
    # VAD needs onnxruntime; keep it off by default so demos don't hard-fail.
    use_vad = (os.getenv("WHISPER_VAD") or "").strip().lower() in {"1", "true", "yes"}
    kwargs = {"beam_size": 1, "vad_filter": use_vad}
    hint = _whisper_lang(language_hint)
    if hint:
        kwargs["language"] = hint
    try:
        segments, info = _faster_model().transcribe(str(path), **kwargs)
    except RuntimeError as exc:
        if use_vad and "onnxruntime" in str(exc).lower():
            kwargs["vad_filter"] = False
            segments, info = _faster_model().transcribe(str(path), **kwargs)
        else:
            raise
    text = " ".join(part.text.strip() for part in segments).strip()
    probability = getattr(info, "language_probability", None)
    return {
        "text": text,
        "language": (info.language or hint or "").lower(),
        "probability": float(probability) if probability is not None else None,
        "backend": "faster-whisper",
    }


def _transcribe_openai(path: Path, language_hint: str | None) -> dict:
    kwargs = {"fp16": False}
    hint = _whisper_lang(language_hint)
    if hint:
        kwargs["language"] = hint
    result = _openai_model().transcribe(str(path), **kwargs)
    language = (result.get("language") or hint or "").lower()
    return {
        "text": (result.get("text") or "").strip(),
        "language": language,
        "probability": None,
        "backend": "openai-whisper",
    }


def _transcribe_google(path: Path, language_hint: str | None) -> dict:
    """Network fallback — no local Torch required."""
    import speech_recognition as sr

    recognizer = sr.Recognizer()
    with sr.AudioFile(str(path)) as source:
        audio = recognizer.record(source, duration=min(_ASR_SECONDS, 55))
    lang = "bn-IN" if (language_hint or "").startswith("bn") else "en-US"
    text = recognizer.recognize_google(audio, language=lang)
    return {
        "text": (text or "").strip(),
        "language": "bn" if lang.startswith("bn") else "en",
        "probability": None,
        "backend": "google-speech",
    }


def _whisper_lang(hint: str | None) -> str | None:
    if not hint:
        return None
    hint = hint.lower()
    if hint.startswith("en"):
        return "en"
    if hint.startswith("bn"):
        return "bn"
    return None
