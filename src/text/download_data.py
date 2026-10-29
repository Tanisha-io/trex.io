"""Download all text datasets used by the Chapter 4–5 pipelines."""

from __future__ import annotations

import io
import json
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "text"

CROWDFLOWER_URL = (
    "https://raw.githubusercontent.com/tlkh/text-emotion-classification/"
    "master/dataset/original/text_emotion.csv"
)
UBMEC_XLSX_URL = (
    "https://github.com/Sakibsourav019/UBMEC-Unified-Bangla-Multi-class-Emotion-Corpus-/"
    "raw/main/UBMEC%20Corpus_Sakib(updated).xlsx"
)
GOEMOTIONS_BASE = (
    "https://raw.githubusercontent.com/google-research/google-research/"
    "master/goemotions/data"
)
GOEMOTIONS_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise", "neutral",
]
DAIR_ID2LABEL = {0: "sadness", 1: "joy", 2: "love", 3: "anger", 4: "fear", 5: "surprise"}


def _get(url: str, timeout: int = 180) -> bytes:
    headers = {"User-Agent": "Mozilla/5.0 audio-sentiment-analysis"}
    resp = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
    resp.raise_for_status()
    return resp.content


def download_english_single_label() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    from datasets import load_dataset

    ds = load_dataset("dair-ai/emotion")
    frames = []
    for split in ds:
        df = ds[split].to_pandas()
        df["label"] = df["label"].map(lambda x: DAIR_ID2LABEL.get(int(x), x))
        df["split"] = split
        frames.append(df[["text", "label", "split"]])
    path = DATA_DIR / "text_emotions.csv"
    pd.concat(frames, ignore_index=True).to_csv(path, index=False)
    (DATA_DIR / "crowdflower_text_emotion.csv").write_bytes(_get(CROWDFLOWER_URL))
    return path


def download_bengali_single_label() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / "bengali_emotions.csv"
    last_error = None
    try:
        df = pd.read_excel(io.BytesIO(_get(UBMEC_XLSX_URL)))
        cols = {str(c).strip().lower(): c for c in df.columns}
        text_col = next(cols[k] for k in cols if k in {"text", "review", "comment", "data", "sentence", "reviews"})
        label_col = next(cols[k] for k in cols if k in {"label", "emotion", "class", "sentiment", "emotions"})
        out = df.rename(columns={text_col: "text", label_col: "label"})[["text", "label"]]
        out["label"] = out["label"].astype(str).str.strip().str.lower()
        out.to_csv(path, index=False)
        return path
    except Exception as exc:
        last_error = exc

    from datasets import load_dataset

    try:
        ds = load_dataset("sustcsenlp/bn_emotion_noisy_dataset")
        emotions = ("Joy", "Anger", "Sadness", "Fear", "Surprise", "Love")
        rows = []
        for row in ds["train"]:
            text = row.get("Data") or row.get("text") or ""
            hits = [e for e in emotions if int(row.get(e, 0) or 0) == 1]
            if len(hits) == 1:
                rows.append({"text": text, "label": hits[0].lower()})
        if rows:
            pd.DataFrame(rows).to_csv(path, index=False)
            return path
    except Exception as exc:
        last_error = exc

    raise RuntimeError(f"Bengali dataset download failed: {last_error}")


def download_goemotions() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    import numpy as np

    frames = []
    for split in ("train", "dev", "test"):
        dest = DATA_DIR / f"goemotions_{split}.tsv"
        dest.write_bytes(_get(f"{GOEMOTIONS_BASE}/{split}.tsv"))
        df = pd.read_csv(dest, sep="\t", header=None, names=["text", "emotion_ids", "id"])
        mat = np.zeros((len(df), len(GOEMOTIONS_LABELS)), dtype=np.int32)
        for i, raw in enumerate(df["emotion_ids"].astype(str)):
            for tok in str(raw).split(","):
                tok = tok.strip()
                if tok.isdigit():
                    mat[i, int(tok)] = 1
        out = pd.DataFrame({"text": df["text"], "split": split})
        for j, name in enumerate(GOEMOTIONS_LABELS):
            out[name] = mat[:, j]
        frames.append(out)
    path = DATA_DIR / "goemotions_simplified.parquet"
    pd.concat(frames, ignore_index=True).to_parquet(path, index=False)
    return path


def main() -> None:
    print("English:", download_english_single_label())
    print("Bengali:", download_bengali_single_label())
    print("GoEmotions:", download_goemotions())


if __name__ == "__main__":
    main()
