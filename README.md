# Multilingual Audio Sentiment Analysis

English and Bengali emotion from speech and from the words. IIT Kharagpur MTP, Tanisha Panda, 20EE38017, April 2025.

## Run

```bash
pip install -r requirements.txt
python -m uvicorn src.api.app:app --host 127.0.0.1 --port 8765
```

Open http://127.0.0.1:8765. Score a file that contains sound, or open a library clip.

## Layout

- `src/api` — page
- `src/audio`, `src/text`, `src/fusion` — models
- `configs`, `notebooks`, `figures`

More detail is in [DATASETS.md](DATASETS.md), [TRAINING.md](TRAINING.md), and [RESULTS.md](RESULTS.md).
