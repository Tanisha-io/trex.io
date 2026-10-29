"""GoEmotions multi-label: Dummy → Ridge → BERT."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

from src import MODELS, device_str
from src.hf_args import make_training_args


GOEMOTIONS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization", "relief",
    "remorse", "sadness", "surprise", "neutral",
]


def load_goemotions():
    from datasets import load_dataset

    # Newer huggingface_hub requires namespace/name (bare "go_emotions" fails)
    try:
        return load_dataset("google-research-datasets/go_emotions", "simplified")
    except Exception as first:
        try:
            return load_dataset("go_emotions", "simplified")
        except Exception:
            raise first


def labels_to_multihot(batch_labels, n=28):
    arr = np.zeros((len(batch_labels), n), dtype=np.float32)
    for i, labs in enumerate(batch_labels):
        for lab in labs:
            if 0 <= int(lab) < n:
                arr[i, int(lab)] = 1.0
    return arr


def light_clean(text: str) -> str:
    import re

    t = str(text).lower()
    t = re.sub(r"http\S+|www\.\S+", " ", t)
    t = re.sub(r"<[^>]+>", " ", t)
    t = re.sub(r"[^\w\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def train_baselines(out_dir: Path) -> dict:
    ds = load_goemotions()
    train_text = [light_clean(t) for t in ds["train"]["text"]]
    test_text = [light_clean(t) for t in ds["test"]["text"]]
    y_train = labels_to_multihot(ds["train"]["labels"])
    y_test = labels_to_multihot(ds["test"]["labels"])

    # Dummy: always Neutral (index 27)
    dummy = np.zeros_like(y_test)
    dummy[:, 27] = 1
    neutral_acc = (y_test[:, 27] == 1).mean()

    tfidf = TfidfVectorizer(max_features=1000)
    x_train = tfidf.fit_transform(train_text)
    x_test = tfidf.transform(test_text)
    ridge = MultiOutputClassifier(RidgeClassifier())
    ridge.fit(x_train, y_train)
    pred = ridge.predict(x_test)
    ridge_f1 = f1_score(y_test, pred, average="macro", zero_division=0)

    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(tfidf, out_dir / "tfidf.joblib")
    joblib.dump(ridge, out_dir / "ridge.joblib")
    metrics = {
        "dummy_neutral_support_rate": float(neutral_acc),
        "ridge_macro_f1": float(ridge_f1),
    }
    (out_dir / "baseline_metrics.json").write_text(json.dumps(metrics, indent=2))
    print("Baselines:", metrics)
    return metrics


def train_bert(out_dir: Path, epochs: int = 4, batch_size: int = 16, max_length: int = 128, lr: float = 5e-5):
    import torch
    from datasets import Dataset
    from sklearn.metrics import classification_report
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer

    ds = load_goemotions()
    model_name = "bert-base-uncased"

    def to_rows(split):
        texts = [light_clean(t) for t in split["text"]]
        labels = labels_to_multihot(split["labels"]).tolist()
        return Dataset.from_dict({"text": texts, "labels": labels})

    train_ds = to_rows(ds["train"])
    val_ds = to_rows(ds["validation"])
    test_ds = to_rows(ds["test"])

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tok(batch):
        enc = tokenizer(batch["text"], truncation=True, padding="max_length", max_length=max_length)
        enc["labels"] = batch["labels"]
        return enc

    train_ds = train_ds.map(tok, batched=True)
    val_ds = val_ds.map(tok, batched=True)
    test_ds = test_ds.map(tok, batched=True)
    # Keep numpy format — set_format("torch") breaks on some Colab torchvision builds
    keep = ["input_ids", "attention_mask", "labels"]
    train_ds = train_ds.remove_columns([c for c in train_ds.column_names if c not in keep])
    val_ds = val_ds.remove_columns([c for c in val_ds.column_names if c not in keep])
    test_ds = test_ds.remove_columns([c for c in test_ds.column_names if c not in keep])

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=28,
        problem_type="multi_label_classification",
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    args = make_training_args(str(out_dir / "runs"), epochs, batch_size, lr, logging_steps=100)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        probs = 1 / (1 + np.exp(-logits))
        preds = (probs >= 0.5).astype(int)
        try:
            auc = roc_auc_score(labels, probs, average="weighted")
        except ValueError:
            auc = float("nan")
        return {
            "weighted_f1": f1_score(labels, preds, average="weighted", zero_division=0),
            "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
            "micro_f1": f1_score(labels, preds, average="micro", zero_division=0),
            "weighted_roc_auc": auc,
        }

    from transformers import default_data_collator

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    metrics = trainer.evaluate(test_ds)
    trainer.save_model(str(out_dir / "best"))
    tokenizer.save_pretrained(str(out_dir / "best"))
    (out_dir / "bert_metrics.json").write_text(json.dumps(metrics, indent=2))
    (out_dir / "label_names.json").write_text(json.dumps(GOEMOTIONS, indent=2))
    print(f"GoEmotions BERT on {device_str()}:", metrics)
    return metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["baseline", "bert", "both"], default="both")
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=16)
    args = p.parse_args()
    out = MODELS / "text" / "multilabel_goemotions"
    if args.mode in ("baseline", "both"):
        train_baselines(out / "baseline")
    if args.mode in ("bert", "both"):
        train_bert(out / "bert", epochs=args.epochs, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
