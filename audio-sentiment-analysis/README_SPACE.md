---
title: Voice Emotion
sdk: docker
app_port: 7860
---

# Voice emotion (hosted demo)

Lightweight CPU demo of the multilingual emotion UI.

## Hugging Face Space (Docker)

**Note (2025+):** Docker and Gradio Spaces on free `cpu-basic` need a [Hugging Face PRO](https://huggingface.co/pro) plan. Static Spaces stay free but cannot run this API.

With PRO, from the repo root after `hf auth login`:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/deploy-hf-space.ps1 -SpaceId "qetzi/voice-emotion"
```

Weights can also live in the free model repo `qetzi/voice-emotion-models`; the container downloads them via `entrypoint.sh` if `models_final/` is missing.

## Render / other Docker hosts (alternative)

1. Push this repo to GitHub (already done).
2. Create a Web Service from `audio-sentiment-analysis/Dockerfile`.
3. Set env `MODEL_REPO=qetzi/voice-emotion-models` (public download at start).
4. Give the service enough disk/RAM for ~3GB weights + Whisper (free tiers often OOM — a small paid instance is safer).
5. Put the public URL in `../demo-config.js` as `EMOTION_DEMO_URL`.

## Local

```bash
pip install -r requirements.txt
# place trained weights in models_final/
set WHISPER_MODEL=base
python -m uvicorn src.api.app:app --host 127.0.0.1 --port 8765
```
