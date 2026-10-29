"""Shared matplotlib helpers for thesis figures."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "axes.grid": True,
            "grid.alpha": 0.25,
        }
    )


def savefig(fig: plt.Figure, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {path}")
    return path


# eGeMAPS column groups used in thesis Ch.6 line plots
LOUDNESS_KEYS = (
    "loudness_sma3_amean",
    "loudness_sma3_percentile80.0",
    "loudness_sma3_pctlrange0-2",
    "loudness_sma3_meanRisingSlope",
    "loudness_sma3_meanFallingSlope",
    "loudness_sma3_stddevNorm",
)

PITCH_KEYS = (
    "F0semitoneFrom27.5Hz_sma3nz_amean",
    "F0semitoneFrom27.5Hz_sma3nz_percentile20.0",
    "F0semitoneFrom27.5Hz_sma3nz_percentile50.0",
    "F0semitoneFrom27.5Hz_sma3nz_percentile80.0",
    "F0semitoneFrom27.5Hz_sma3nz_pctlrange0-2",
    "F0semitoneFrom27.5Hz_sma3nz_stddevNorm",
)

MFCC_KEYS = (
    "mfcc1_sma3_amean",
    "mfcc2_sma3_amean",
    "mfcc3_sma3_amean",
    "mfcc4_sma3_amean",
)

VQ_KEYS = (
    "jitterLocal_sma3nz_amean",
    "jitterLocal_sma3nz_stddevNorm",
    "shimmerLocaldB_sma3nz_amean",
    "HNRdBACF_sma3nz_amean",
    "logRelF0-H1-H2_sma3nz_amean",
    "logRelF0-H1-A3_sma3nz_amean",
)

META = {"path", "emotion"}
