"""Print GoEmotions / text metrics from models_final or models/."""
from __future__ import annotations

import json
from pathlib import Path

from src import MODELS, ROOT


def main():
    roots = [
        ROOT / "models_final" / "text",
        MODELS / "text",
        Path("/content/drive/MyDrive/audio-sentiment-analysis/models_final/text"),
        Path("/content/audio-sentiment-analysis/models/text"),
    ]
    found = []
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("*metrics.json"):
            found.append(p)
    if not found:
        print("No *metrics.json found under models_final/text or models/text")
        return
    for p in sorted(found):
        print("=" * 60)
        print(p)
        try:
            print(json.dumps(json.loads(p.read_text()), indent=2)[:2000])
        except Exception as e:
            print("read error", e)


if __name__ == "__main__":
    main()
