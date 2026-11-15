---
title: Voice Emotion
sdk: docker
app_port: 7860
---

# Voice emotion (HF Space)

Lightweight CPU demo of the multilingual emotion UI.

## Local

```bash
pip install -r requirements.txt
# place trained weights in models_final/
set WHISPER_MODEL=base
python -m uvicorn src.api.app:app --host 127.0.0.1 --port 8765
```

## Deploy (Hugging Face Space)

1. Create a Docker Space.
2. Upload this folder + `models_final/` (gitignored in trex.io).
3. Space builds the Dockerfile and serves on port 7860.
4. Point the portfolio emotion card at the Space URL.

`WHISPER_MODEL` defaults to **base** (lighter than former `small`). Docker sets `WHISPER_BACKEND=faster`. On Windows locally, the app falls back to openai-whisper, or voice-only scoring if Torch/Whisper cannot load.
