"""Load single-label text CSVs or HuggingFace fallbacks."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src import DATA


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c.lower().strip(): c for c in df.columns}
    text_col = None
    label_col = None
    for key in ("text", "content", "sentence", "tweet", "comment"):
        if key in cols:
            text_col = cols[key]
            break
    for key in ("emotion", "label", "sentiment", "emotions"):
        if key in cols:
            label_col = cols[key]
            break
    if text_col is None or label_col is None:
        raise ValueError(
            f"Need text + emotion columns. Found: {list(df.columns)}. "
            "Rename to text,emotion or content,label."
        )
    out = df[[text_col, label_col]].copy()
    out.columns = ["text", "emotion"]
    out["text"] = out["text"].astype(str)
    out["emotion"] = out["emotion"].astype(str).str.lower().str.strip()
    out = out.dropna().drop_duplicates()
    return out.reset_index(drop=True)


def load_english_csv(path: Optional[Path] = None) -> pd.DataFrame:
    path = path or (DATA / "text" / "text_emotions.csv")
    if path.exists():
        return _normalize_columns(pd.read_csv(path))
    # Public fallback close to common emotion CSVs
    from datasets import load_dataset

    ds = load_dataset("dair-ai/emotion", "split")
    frames = []
    for split in ("train", "validation", "test"):
        frames.append(pd.DataFrame(ds[split]))
    df = pd.concat(frames, ignore_index=True)
    # dair-ai uses int labels
    id2label = ds["train"].features["label"].names
    df["emotion"] = df["label"].map(lambda i: id2label[i])
    df = df.rename(columns={"text": "text"})
    return df[["text", "emotion"]]


def load_bengali_csv(path: Optional[Path] = None) -> pd.DataFrame:
    path = path or (DATA / "text" / "bengali_emotions.csv")
    if not path.exists():
        raise FileNotFoundError(
            f"Bengali CSV not found at {path}. Place your custom corpus there "
            "with columns text,emotion."
        )
    return _normalize_columns(pd.read_csv(path))


def split_xy(
    df: pd.DataFrame, test_size: float = 0.2, seed: int = 42
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, LabelEncoder]:
    le = LabelEncoder()
    y = le.fit_transform(df["emotion"])
    x_train, x_test, y_train, y_test = train_test_split(
        df["text"], y, test_size=test_size, random_state=seed, stratify=y
    )
    return x_train, x_test, y_train, y_test, le
