"""Cross-modal mapping + confidence-weighted fusion + WAS/CED metrics."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import yaml


def load_mapping(path: Path, key: str) -> Dict[str, Dict[str, float]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw = data[key]
    out: Dict[str, Dict[str, float]] = {}
    for text_lab, dist in raw.items():
        s = sum(float(v) for v in dist.values()) or 1.0
        out[text_lab] = {a: float(v) / s for a, v in dist.items()}
    return out


def map_text_to_audio(
    text_probs: Mapping[str, float],
    mapping: Mapping[str, Mapping[str, float]],
    audio_labels: Sequence[str],
) -> np.ndarray:
    vec = np.zeros(len(audio_labels), dtype=np.float64)
    idx = {lab: i for i, lab in enumerate(audio_labels)}
    for t_lab, p in text_probs.items():
        dist = mapping.get(t_lab)
        if not dist:
            continue
        for a_lab, m in dist.items():
            if a_lab in idx:
                vec[idx[a_lab]] += float(p) * float(m)
    s = vec.sum()
    if s > 0:
        vec /= s
    else:
        vec[:] = 1.0 / len(audio_labels)
    return vec


def fuse(
    audio_probs: np.ndarray,
    text_mapped: np.ndarray,
    audio_conf: float,
    text_conf: float,
) -> Tuple[np.ndarray, float, float]:
    c_a = max(float(audio_conf), 1e-8)
    c_t = max(float(text_conf), 1e-8)
    w1 = c_a / (c_a + c_t)
    w2 = 1.0 - w1
    final = w1 * audio_probs + w2 * text_mapped
    s = final.sum()
    if s > 0:
        final = final / s
    return final, w1, w2


def weighted_alignment_score(
    audio_probs: np.ndarray,
    text_mapped: np.ndarray,
    w1: float,
    w2: float,
) -> float:
    a = int(audio_probs.argmax())
    t = int(text_mapped.argmax())
    agree = 1.0 if a == t else 0.0
    na = np.linalg.norm(audio_probs) + 1e-8
    nt = np.linalg.norm(text_mapped) + 1e-8
    cos = float(np.dot(audio_probs, text_mapped) / (na * nt))
    return w1 * agree + w2 * cos


def cross_entropy_divergence(
    audio_probs: np.ndarray,
    text_mapped: np.ndarray,
    eps: float = 1e-5,
) -> float:
    p = np.clip(audio_probs, eps, 1.0)
    q = np.clip(text_mapped, eps, 1.0)
    return float(-(p * np.log(q)).sum())


def top_emotions(probs: np.ndarray, labels: Sequence[str], k: int = 3) -> List[dict]:
    order = np.argsort(-probs)[:k]
    return [{"label": labels[i], "probability": float(probs[i])} for i in order]
