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

From the repo root (after `hf auth login`):

```powershell
powershell -File scripts/deploy-hf-space.ps1
# or: powershell -File scripts/deploy-hf-space.ps1 -SpaceId "YourHFUser/voice-emotion"
```

That creates a public Docker Space, uploads this folder + `models_final/` (~2.7GB), and prints the app URL. Put that URL in `../demo-config.js` (`EMOTION_DEMO_URL`) and push so GitHub Pages links to it.

Manual alternative: create a Docker Space, upload this folder + weights, wait for the build on port 7860.

`WHISPER_MODEL` defaults to **base** (lighter than former `small`). Docker sets `WHISPER_BACKEND=faster`. On Windows locally, the app falls back to openai-whisper, or voice-only scoring if Torch/Whisper cannot load.
