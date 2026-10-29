"""Train classical + transformer single-label text emotion models."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

from src import MODELS, device_str
from src.hf_args import make_training_args
from src.text.data import load_bengali_csv, load_english_csv, split_xy
from src.text.preprocess import (
    batch_apply,
    preprocess_bengali_classical,
    preprocess_bengali_mbert,
    preprocess_english_bert,
    preprocess_english_classical,
)


def classical_models(language: str) -> dict:
    ngram = (1, 2) if language == "bn" else (1, 1)
    vectorizer = Pipeline(
        [
            ("count", CountVectorizer(ngram_range=ngram, max_features=50000)),
            ("tfidf", TfidfTransformer()),
        ]
    )
    clfs = {
        "naive_bayes": MultinomialNB(),
        "logreg": LogisticRegression(max_iter=2000, n_jobs=-1),
        "svm": LinearSVC(),
        "rf": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        "xgboost": XGBClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            random_state=42,
            n_jobs=-1,
            eval_metric="mlogloss",
        ),
    }
    return vectorizer, clfs


def train_classical(language: str, out_dir: Path) -> dict:
    if language == "en":
        df = load_english_csv()
        prep = preprocess_english_classical
    else:
        df = load_bengali_csv()
        prep = preprocess_bengali_classical

    x_train, x_test, y_train, y_test, le = split_xy(df)
    x_train = batch_apply(x_train, prep)
    x_test = batch_apply(x_test, prep)

    vectorizer, clfs = classical_models(language)
    x_train_v = vectorizer.fit_transform(x_train)
    x_test_v = vectorizer.transform(x_test)

    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, out_dir / "tfidf_pipeline.joblib")
    joblib.dump(le, out_dir / "label_encoder.joblib")

    results = {}
    for name, clf in clfs.items():
        if language == "bn" and name not in ("logreg", "svm", "rf"):
            continue
        clf.fit(x_train_v, y_train)
        pred = clf.predict(x_test_v)
        acc = accuracy_score(y_test, pred)
        f1 = f1_score(y_test, pred, average="macro")
        results[name] = {"accuracy": acc, "macro_f1": f1}
        joblib.dump(clf, out_dir / f"{name}.joblib")
        print(f"[{language}] {name}: acc={acc:.4f} macro_f1={f1:.4f}")
        print(classification_report(y_test, pred, target_names=list(le.classes_), zero_division=0))

    (out_dir / "classical_metrics.json").write_text(json.dumps(results, indent=2))
    return results


def train_transformer(
    language: str,
    out_dir: Path,
    epochs: int = 3,
    batch_size: int = 16,
    max_length: int = 128,
    lr: float = 5e-5,
) -> dict:
    import torch
    from datasets import Dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer

    if language == "en":
        df = load_english_csv()
        model_name = "bert-base-uncased"
        prep = preprocess_english_bert
    else:
        df = load_bengali_csv()
        model_name = "bert-base-multilingual-cased"
        prep = preprocess_bengali_mbert

    x_train, x_test, y_train, y_test, le = split_xy(df)
    n_labels = len(le.classes_)
    train_df = Dataset.from_dict(
        {"text": batch_apply(x_train, prep), "label": list(map(int, y_train))}
    )
    test_df = Dataset.from_dict(
        {"text": batch_apply(x_test, prep), "label": list(map(int, y_test))}
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tok(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=max_length)

    train_df = train_df.map(tok, batched=True)
    test_df = test_df.map(tok, batched=True)
    # Keep numpy format — set_format("torch") breaks on some Colab torchvision builds
    keep = ["input_ids", "attention_mask", "label"]
    train_df = train_df.remove_columns([c for c in train_df.column_names if c not in keep])
    test_df = test_df.remove_columns([c for c in test_df.column_names if c not in keep])

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=n_labels)
    out_dir.mkdir(parents=True, exist_ok=True)

    args = make_training_args(str(out_dir / "runs"), epochs, batch_size, lr)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "macro_f1": f1_score(labels, preds, average="macro"),
        }

    from transformers import default_data_collator

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_df,
        eval_dataset=test_df,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    metrics = trainer.evaluate()
    trainer.save_model(str(out_dir / "best"))
    tokenizer.save_pretrained(str(out_dir / "best"))
    joblib.dump(le, out_dir / "label_encoder.joblib")
    (out_dir / "transformer_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"[{language}] transformer on {device_str()}: {metrics}")
    return metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--language", choices=["en", "bn"], required=True)
    p.add_argument("--mode", choices=["classical", "transformer", "both"], default="both")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=16)
    args = p.parse_args()

    out = MODELS / "text" / f"single_{args.language}"
    if args.mode in ("classical", "both"):
        train_classical(args.language, out / "classical")
    if args.mode in ("transformer", "both"):
        train_transformer(args.language, out / "transformer", epochs=args.epochs, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
