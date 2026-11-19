"""The local page: score a recording, or open a saved example."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src import UPLOADS
from src.api.cues import library
import opensmile
import numpy as np

STATIC = Path(__file__).resolve().parent / "static"
app = FastAPI(title="Voice emotion")
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/example")
def example(language: str = "bn", emotion: str | None = None):
    dataset = "banglaser" if language == "bn" else "crema_d"
    lib = library(dataset)
    row = lib.pick(emotion)
    payload = lib.score_row(row)
    payload["source"] = "library"
    payload["frames"] = None
    payload["transcript"] = None
    payload["text_emotion"] = None
    payload["fusion"] = None
    payload["note"] = (
        "This is a saved example. The words, the text score, and the combined "
        "score need a recording."
    )
    return payload


@app.post("/api/analyze")
async def analyze(
    language: str = Form("bn"),
    text: str = Form(""),
    audio: UploadFile | None = File(None),
):
    fallback_lang = _supported_language(language) or "bn"
    if audio is None or not audio.filename:
        if text.strip():
            return {
                "ok": False,
                "error": "Paste is not enough here. Choose a file that contains sound.",
            }
        return {"ok": False, "error": "Choose a file that contains sound."}

    dest = UPLOADS / Path(audio.filename).name
    raw = await audio.read()
    dest.write_bytes(raw)
    wav = UPLOADS / f"{dest.stem}.analysis.wav"
    trim_note = ""
    try:
        from src.api.media import MAX_SECONDS, to_pcm_wav

        wav = to_pcm_wav(dest, wav)
        marker = wav.with_suffix(".trim.txt")
        if marker.exists():
            trim_note = f" Large media trimmed to first {MAX_SECONDS:.0f}s."
    except Exception:
        return {
            "ok": False,
            "error": "Could not read audio from this file. Use a file that contains sound.",
        }

    try:
        from src.audio.features import extract_egemaps

        values = extract_egemaps(wav)
        frames = _frames(wav)
    except Exception:
        return {
            "ok": False,
            "error": "Could not read the voice in this file. A library clip still works.",
        }

    asr_note = ""
    try:
        from src.api.asr import transcribe

        heard = transcribe(wav, language_hint=fallback_lang)
        detected_code = _supported_language(heard["language"]) or fallback_lang
        heard_lang = heard["language"] or fallback_lang
        heard_prob = heard["probability"]
        transcript = (heard.get("text") or "").strip() or None
    except Exception as exc:
        heard_lang = fallback_lang
        heard_prob = None
        transcript = text.strip() or None
        detected_code = fallback_lang
        asr_note = (
            f"ASR unavailable ({type(exc).__name__}); "
            f"scored voice only using language={fallback_lang}."
        )

    detected = {
        "language": detected_code or heard_lang or "",
        "language_label": _language_label(detected_code or heard_lang),
        "language_probability": heard_prob,
        "emotion": None,
    }
    if detected_code is None:
        return {
            "ok": True,
            "source": "upload",
            "transcript": transcript,
            "detected": detected,
            "note": f"Heard {detected['language_label']}. Emotion scoring covers English and Bengali." + trim_note,
            "frames": frames,
        }

    dataset = "banglaser" if detected_code == "bn" else "crema_d"
    row = pd.Series(values)
    payload = library(dataset).score_row(row)
    payload["source"] = "upload"
    payload["frames"] = frames
    payload["true_label"] = None
    payload["transcript"] = transcript
    payload["note"] = (asr_note + trim_note).strip()
    if text.strip() and not payload["transcript"]:
        payload["transcript"] = text.strip()

    if payload.get("transcript"):
        try:
            _add_text_and_fusion(detected_code, payload)
        except Exception:
            payload["text_emotion"] = None
            payload["fusion"] = None
    else:
        payload["text_emotion"] = None
        payload["fusion"] = None

    detected["emotion"] = payload["emotion"]
    payload["detected"] = detected
    return payload


def _supported_language(code: str) -> str | None:
    code = (code or "").lower()
    if code.startswith("en"):
        return "en"
    if code.startswith("bn"):
        return "bn"
    return None


def _language_label(code: str) -> str:
    supported = _supported_language(code)
    if supported == "en":
        return "English"
    if supported == "bn":
        return "Bengali"
    return code or "unknown"


def _add_text_and_fusion(language: str, payload: dict) -> None:
    payload["text_emotion"] = None
    payload["fusion"] = None
    transcript = (payload.get("transcript") or "").strip()
    if not transcript:
        return
    import numpy as np

    from src import CONFIGS
    from src.api.text_emotion import text_model
    from src.fusion.fuse import fuse, load_mapping, map_text_to_audio

    text = text_model(language).score(transcript)
    payload["text_emotion"] = text
    key = "english_text_to_crema_d" if language == "en" else "bengali_text_to_banglaser"
    mapping = load_mapping(CONFIGS / "fusion_mapping.yaml", key)
    if language == "en":
        if "happiness" in mapping:
            mapping.setdefault("joy", mapping["happiness"])
        geo = load_mapping(CONFIGS / "fusion_mapping.yaml", "goemotions_to_crema_d")
        if "love" in geo:
            mapping.setdefault("love", geo["love"])
    elif "joy" in mapping:
        mapping.setdefault("love", mapping["joy"])
    audio_labels = payload["profiles"][0]["emotions"]
    audio_map = {item["label"]: item["probability"] for item in payload["probabilities"]}
    audio = np.array([audio_map[label] for label in audio_labels], dtype=np.float64)
    text_map = {item["label"]: item["probability"] for item in text["probabilities"]}
    mapped = map_text_to_audio(text_map, mapping, audio_labels)
    final, voice_weight, text_weight = fuse(audio, mapped, payload["confidence"], text["confidence"])
    order = np.argsort(-final)
    payload["fusion"] = {
        "emotion": audio_labels[int(order[0])],
        "confidence": float(final[int(order[0])]),
        "voice_weight": float(voice_weight),
        "text_weight": float(text_weight),
        "probabilities": [
            {"label": audio_labels[int(i)], "probability": float(final[int(i)])}
            for i in order
        ],
    }


def _frames(path: Path):
    """Loudness and pitch across the clip, for the small trace under the charts."""
    

    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
    )
    frame = smile.process_file(str(path))

    def series(name: str) -> list:
        if name not in frame.columns:
            return []
        values = frame[name].to_numpy(dtype=float)
        if len(values) > 280:
            edges = np.linspace(0, len(values), 281).astype(int)
            values = np.array([values[a:b].mean() for a, b in zip(edges[:-1], edges[1:]) if b > a])
        return [round(float(v), 4) for v in values]

    return {
        "loudness": series("Loudness_sma3"),
        "pitch": series("F0semitoneFrom27.5Hz_sma3nz"),
    }
