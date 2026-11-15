"""Text emotion from the saved BERT weights, without PyTorch.

The fine-tuned checkpoints are BertForSequenceClassification safetensors.
This is the eval-time forward pass: no dropout, post-norm, erf GELU.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
from safetensors import safe_open

from src import ROOT
from src.text.preprocess import preprocess_bengali_mbert, preprocess_english_bert

FINAL = ROOT / "models_final" / "text"
HEADS = 12
HEAD_DIM = 64


def _linear(x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return x @ weight.T + bias


def _norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray) -> np.ndarray:
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return gamma * (x - mean) / np.sqrt(var + 1e-12) + beta


def _gelu(x: np.ndarray) -> np.ndarray:
    from scipy.special import erf

    return 0.5 * x * (1.0 + erf(x / np.sqrt(2.0)))


def _softmax(x: np.ndarray) -> np.ndarray:
    z = x - x.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


class TextEmotion:
    def __init__(self, language: str):
        name = "single_en" if language == "en" else "single_bn"
        folder = FINAL / name / "transformer"
        best = folder / "best"
        self.language = language
        self.head = "English BERT" if language == "en" else "Bengali mBERT"
        self.files = [
            str(best / "model.safetensors"),
            str(best / "tokenizer.json"),
            str(folder / "label_encoder.joblib"),
        ]
        self.labels = [str(c) for c in joblib.load(folder / "label_encoder.joblib").classes_]
        from tokenizers import Tokenizer

        self.tokenizer = Tokenizer.from_file(str(best / "tokenizer.json"))
        self.tokenizer.no_padding()
        self.tokenizer.enable_truncation(max_length=128)
        self.weights = {}
        with safe_open(best / "model.safetensors", framework="numpy") as handle:
            for key in handle.keys():
                self.weights[key] = handle.get_tensor(key)

    def _encode(self, text: str) -> tuple[np.ndarray, np.ndarray]:
        cleaned = preprocess_english_bert(text) if self.language == "en" else preprocess_bengali_mbert(text)
        enc = self.tokenizer.encode(cleaned or text)
        ids = list(enc.ids)
        types = list(enc.type_ids) or [0] * len(ids)
        mask = list(enc.attention_mask) or [1] * len(ids)
        if mask and len(mask) == len(ids) and sum(mask) < len(ids):
            keep = int(sum(mask))
            ids, types = ids[:keep], types[:keep]
        if not ids or ids[0] != 101:
            ids = [101] + ids
            types = [0] + types
        if ids[-1] != 102:
            ids.append(102)
            types.append(0)
        ids = ids[:128]
        types = types[: len(ids)]
        return np.asarray(ids, dtype=np.int64), np.asarray(types, dtype=np.int64)

    def _forward(self, ids: np.ndarray, types: np.ndarray) -> np.ndarray:
        w = self.weights
        positions = np.arange(len(ids))
        hidden = (
            w["bert.embeddings.word_embeddings.weight"][ids]
            + w["bert.embeddings.position_embeddings.weight"][positions]
            + w["bert.embeddings.token_type_embeddings.weight"][types]
        )
        hidden = _norm(hidden, w["bert.embeddings.LayerNorm.gamma"], w["bert.embeddings.LayerNorm.beta"])
        mask = np.ones(len(ids), dtype=np.float32)
        for layer in range(12):
            prefix = f"bert.encoder.layer.{layer}."
            q = _linear(hidden, w[prefix + "attention.self.query.weight"], w[prefix + "attention.self.query.bias"])
            k = _linear(hidden, w[prefix + "attention.self.key.weight"], w[prefix + "attention.self.key.bias"])
            v = _linear(hidden, w[prefix + "attention.self.value.weight"], w[prefix + "attention.self.value.bias"])
            seq = q.shape[0]
            q = q.reshape(seq, HEADS, HEAD_DIM).transpose(1, 0, 2)
            k = k.reshape(seq, HEADS, HEAD_DIM).transpose(1, 0, 2)
            v = v.reshape(seq, HEADS, HEAD_DIM).transpose(1, 0, 2)
            scores = (q @ k.transpose(0, 2, 1)) / np.sqrt(HEAD_DIM)
            scores = scores + (1.0 - mask)[None, None, :] * -1e4
            ctx = (_softmax(scores) @ v).transpose(1, 0, 2).reshape(seq, -1)
            attn = _linear(ctx, w[prefix + "attention.output.dense.weight"], w[prefix + "attention.output.dense.bias"])
            hidden = _norm(hidden + attn, w[prefix + "attention.output.LayerNorm.gamma"], w[prefix + "attention.output.LayerNorm.beta"])
            mid = _gelu(_linear(hidden, w[prefix + "intermediate.dense.weight"], w[prefix + "intermediate.dense.bias"]))
            out = _linear(mid, w[prefix + "output.dense.weight"], w[prefix + "output.dense.bias"])
            hidden = _norm(hidden + out, w[prefix + "output.LayerNorm.gamma"], w[prefix + "output.LayerNorm.beta"])
        pooled = np.tanh(_linear(hidden[0], w["bert.pooler.dense.weight"], w["bert.pooler.dense.bias"]))
        return _linear(pooled, w["classifier.weight"], w["classifier.bias"])

    def score(self, text: str) -> dict:
        ids, types = self._encode(text)
        logits = self._forward(ids, types)
        probs = _softmax(logits)
        order = np.argsort(-probs)
        top = int(order[0])
        return {
            "head": self.head,
            "files": self.files,
            "emotion": self.labels[top],
            "confidence": float(probs[top]),
            "probabilities": [
                {"label": self.labels[int(i)], "probability": float(probs[int(i)])}
                for i in order
            ],
        }


@lru_cache(maxsize=2)
def text_model(language: str) -> TextEmotion:
    return TextEmotion("en" if language == "en" else "bn")
