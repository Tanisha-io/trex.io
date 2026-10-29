"""Demo fusion on synthetic probs + optional real model dirs later."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src import CONFIGS, ROOT
from src.fusion import (
    cross_entropy_divergence,
    fuse,
    load_mapping,
    map_text_to_audio,
    top_emotions,
    weighted_alignment_score,
)


def main():
    mapping_path = CONFIGS / "fusion_mapping.yaml"
    # English demo: text single-label space -> CREMA-D
    mapping = load_mapping(mapping_path, "english_text_to_crema_d")
    audio_labels = ["anger", "disgust", "fear", "happy", "neutral", "sad"]

    # fake model outputs
    text_probs = {"sadness": 0.7, "neutral": 0.2, "fear": 0.1}
    # map keys to mapping names used in yaml
    text_probs = {"sadness" if k == "sadness" else k: v for k, v in text_probs.items()}
    # yaml uses "sadness" in goemotions but english_text uses "sadness"? check labels
    # english_text_to_crema_d uses: anger, disgust, fear, happiness, sadness, surprise, neutral
    text_probs = {"sadness": 0.7, "neutral": 0.2, "fear": 0.1}

    audio_probs = np.array([0.05, 0.05, 0.15, 0.05, 0.20, 0.50])  # sad-heavy
    mapped = map_text_to_audio(text_probs, mapping, audio_labels)
    final, w1, w2 = fuse(audio_probs, mapped, audio_conf=0.62, text_conf=0.38)
    was = weighted_alignment_score(audio_probs, mapped, w1, w2)
    ced = cross_entropy_divergence(audio_probs, mapped)

    out = {
        "audio_top": top_emotions(audio_probs, audio_labels),
        "text_mapped_top": top_emotions(mapped, audio_labels),
        "fused_top": top_emotions(final, audio_labels),
        "modality_weights": {"audio": w1, "text": w2},
        "WAS": was,
        "CED": ced,
    }
    print(json.dumps(out, indent=2))
    demo_path = ROOT / "models" / "fusion_demo.json"
    demo_path.parent.mkdir(parents=True, exist_ok=True)
    demo_path.write_text(json.dumps(out, indent=2))
    print("wrote", demo_path)


if __name__ == "__main__":
    main()
