# Multilingual Audio Sentiment Analysis

English and Bengali emotion from speech and from the words. IIT Kharagpur MTP, Tanisha Panda, 20EE38017, April 2025.

## Run

From this folder (needs `models_final/` on disk — not in git):

```bash
pip install -r requirements.txt
set WHISPER_MODEL=base
python -m uvicorn src.api.app:app --host 127.0.0.1 --port 8765
```

Open http://127.0.0.1:8765. Score a file that contains sound, or open a library clip.

ASR uses Whisper **`base`** (lighter than the old `small`). On Windows the app uses **openai-whisper** by default (more stable); Linux/Docker uses **faster-whisper**. Override with `WHISPER_MODEL=tiny|base|small` and `WHISPER_BACKEND=openai|faster`.

For a public demo, see [README_SPACE.md](README_SPACE.md) (Hugging Face Docker Space).

## Layout

- `src/api` — page
- `src/audio`, `src/text`, `src/fusion` — models
- `configs`, `notebooks`, `figures`

More detail is in [DATASETS.md](DATASETS.md), [TRAINING.md](TRAINING.md), and [RESULTS.md](RESULTS.md).
