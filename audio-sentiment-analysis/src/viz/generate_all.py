"""Regenerate thesis List-of-Figures graphs (Fig 1.1, 6.1–6.23).

Usage:
  python -m src.viz.generate_all
  python -m src.viz.generate_all --models-root /path/to/models_final
  python -m src.viz.generate_all --skip-dnn   # skip LSTM/BiLSTM CMs if no GPU/weights

Requires for audio figures:
  models/.../classical/features.csv  (from train_classical)
  optional: random_forest.joblib, xgboost.joblib, label_encoder.joblib, scaler.joblib, kept_features.joblib
  optional DNN: models/.../dnn/best.pt + features.npz + label_encoder.joblib
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src import DATA, MODELS, ROOT
from src.viz._style import (
    LOUDNESS_KEYS,
    META,
    MFCC_KEYS,
    PITCH_KEYS,
    VQ_KEYS,
    apply_style,
    savefig,
)

FIGURES = ROOT / "figures"


def _resolve_classical(models_root: Path, dataset: str) -> Path:
    for cand in (
        models_root / "audio" / dataset / "classical",
        models_root / dataset / "classical",
        MODELS / "audio" / dataset / "classical",
    ):
        if (cand / "features.csv").exists():
            return cand
    return models_root / "audio" / dataset / "classical"


def _resolve_dnn(models_root: Path, dataset: str) -> Path:
    for cand in (
        models_root / "audio" / dataset / "dnn",
        models_root / dataset / "dnn",
        MODELS / "audio" / dataset / "dnn",
    ):
        if (cand / "best.pt").exists() or (cand / "features.npz").exists():
            return cand
    return models_root / "audio" / dataset / "dnn"


def load_features(classical_dir: Path) -> Optional[pd.DataFrame]:
    path = classical_dir / "features.csv"
    if not path.exists():
        print(f"  missing {path}")
        return None
    df = pd.read_csv(path)
    if "emotion" not in df.columns:
        print(f"  no emotion column in {path}")
        return None
    return df


def present_cols(df: pd.DataFrame, keys: Sequence[str]) -> List[str]:
    return [k for k in keys if k in df.columns]


def match_group(df: pd.DataFrame, needles: Sequence[str], fallback_sub: str) -> List[str]:
    cols = present_cols(df, needles)
    if cols:
        return cols
    return [c for c in df.columns if fallback_sub.lower() in c.lower() and c not in META][:8]


# ---------------------------------------------------------------------------
# Fig 1.1 — System architecture
# ---------------------------------------------------------------------------
def fig_1_1_architecture(out: Path) -> Path:
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    apply_style()
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis("off")
    ax.set_title("Fig 1.1 — Multilingual multimodal emotion detection", pad=12)

    def box(x, y, w, h, text, color="#E8F1F8"):
        p = FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.15",
            linewidth=1.2, edgecolor="#234", facecolor=color,
        )
        ax.add_patch(p)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8, wrap=True)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(
            FancyArrowPatch(
                (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12,
                linewidth=1.0, color="#345",
            )
        )

    box(0.3, 3.4, 1.8, 1.2, "Audio / Video\ninput", "#F7E9D7")
    box(2.5, 5.4, 2.2, 1.4, "Language-Aware ASR\n(Whisper)\ntranscript + lang", "#DCEFDA")
    box(2.5, 1.4, 2.2, 1.4, "Acoustic features\nOpenSMILE eGeMAPS\n→ VIF filter", "#DCEFDA")

    box(5.2, 6.2, 2.4, 1.0, "EN text: BERT\n(single / multi-label)", "#E6E0F8")
    box(5.2, 4.8, 2.4, 1.0, "BN text: mBERT\n(single / multi-label)", "#E6E0F8")
    box(5.2, 2.0, 2.4, 1.0, "EN audio: LSTM", "#E6E0F8")
    box(5.2, 0.6, 2.4, 1.0, "BN audio: BiLSTM", "#E6E0F8")

    box(8.2, 3.2, 3.2, 1.8, "Confidence-weighted\nfusion + mapping M\n(text → audio space)", "#F8E0E0")
    box(8.6, 1.0, 2.4, 1.0, "Final emotion\nP_final", "#F7E9D7")

    arrow(2.1, 4.0, 2.5, 5.8)
    arrow(2.1, 3.6, 2.5, 2.2)
    arrow(4.7, 6.1, 5.2, 6.7)
    arrow(4.7, 5.7, 5.2, 5.3)
    arrow(4.7, 2.1, 5.2, 2.5)
    arrow(4.7, 1.7, 5.2, 1.1)
    arrow(7.6, 6.7, 8.2, 4.6)
    arrow(7.6, 5.3, 8.2, 4.4)
    arrow(7.6, 2.5, 8.2, 3.8)
    arrow(7.6, 1.1, 8.2, 3.6)
    arrow(9.8, 3.2, 9.8, 2.0)

    return savefig(fig, out / "fig_1_1_system_architecture.png")


# ---------------------------------------------------------------------------
# Fig 6.1 — Augmentation waveforms
# ---------------------------------------------------------------------------
def _demo_waveform(sr: int = 16000, seconds: float = 2.0) -> np.ndarray:
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    # speech-like envelope + harmonics
    env = np.exp(-1.5 * np.abs(t - 0.9)) * (0.4 + 0.6 * (t > 0.15) * (t < 1.7))
    y = env * (
        0.6 * np.sin(2 * np.pi * 180 * t)
        + 0.25 * np.sin(2 * np.pi * 360 * t)
        + 0.1 * np.sin(2 * np.pi * 540 * t)
    )
    return y.astype(np.float32)


def fig_6_1_augmentation(out: Path, sample_wav: Optional[Path] = None) -> Path:
    import matplotlib.pyplot as plt
    from src.audio.augment import load_wav, noise_inject, temporal_shift

    apply_style()
    sr = 16000
    if sample_wav is not None and Path(sample_wav).exists():
        y0, sr = load_wav(Path(sample_wav), sr=16000)
        y0 = y0[: sr * 2]
    else:
        y0 = None
        for ds in ("crema_d", "banglaser"):
            d = DATA / "audio" / ds
            wavs = sorted(d.rglob("*.wav")) if d.exists() else []
            if wavs:
                try:
                    y0, sr = load_wav(wavs[0], sr=16000)
                    y0 = y0[: sr * 2]
                    print(f"  aug demo wav: {wavs[0].name}")
                    break
                except Exception:
                    continue
        if y0 is None:
            print("  no wav found - using synthetic speech-like waveform")
            y0 = _demo_waveform(sr)

    def _time_stretch_np(y: np.ndarray, rate: float = 0.95) -> np.ndarray:
        n_out = max(1, int(len(y) / rate))
        x_old = np.linspace(0.0, 1.0, num=len(y), endpoint=False)
        x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
        return np.interp(x_new, x_old, y).astype(np.float32)

    def _pitch_shift_np(y: np.ndarray, n_steps: float = -3.0) -> np.ndarray:
        factor = 2.0 ** (n_steps / 12.0)
        stretched = _time_stretch_np(y, rate=factor)
        if len(stretched) >= len(y):
            return stretched[: len(y)]
        out = np.zeros_like(y)
        out[: len(stretched)] = stretched
        return out

    try:
        from src.audio.augment import pitch_shift, time_stretch

        y_stretch = time_stretch(y0.copy(), sr)
        y_pitch = pitch_shift(y0.copy(), sr, semitones=(-3.0,))
    except Exception:
        y_stretch = _time_stretch_np(y0.copy(), rate=0.95)
        y_pitch = _pitch_shift_np(y0.copy(), n_steps=-3.0)

    np.random.seed(42)
    panels = [
        ("Temporal shifting", temporal_shift(y0.copy(), max_frac=0.15)),
        ("Noise injection", noise_inject(y0.copy())),
        ("Time stretching", y_stretch),
        ("Pitch shifting", y_pitch),
    ]

    fig, axes = plt.subplots(4, 1, figsize=(10, 9), sharex=True)
    fig.suptitle("Fig 6.1 - Data Augmentation Results", y=0.995)
    t0 = np.arange(len(y0)) / sr
    for ax, (title, y1) in zip(axes, panels):
        t1 = np.arange(len(y1)) / sr
        ax.plot(t0, y0, color="C0", alpha=0.75, label="original", linewidth=0.8)
        ax.plot(t1, y1, color="C3", alpha=0.75, label="augmented", linewidth=0.8)
        ax.set_ylabel("amp")
        ax.set_title(title, loc="left", fontsize=10)
        ax.legend(loc="upper right", fontsize=7)
    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    return savefig(fig, out / "fig_6_1_data_augmentation.png")


# ---------------------------------------------------------------------------
# Fig 6.2–6.3 — Correlation heatmaps
# ---------------------------------------------------------------------------
def fig_heatmap(df: pd.DataFrame, title: str, out_path: Path, max_feats: int = 60) -> Path:
    import matplotlib.pyplot as plt
    import seaborn as sns

    apply_style()
    feat_cols = [c for c in df.columns if c not in META]
    # keep a representative subset if huge (thesis shows block structure)
    if len(feat_cols) > max_feats:
        # prefer known groups first
        preferred = []
        for keys in (PITCH_KEYS, LOUDNESS_KEYS, MFCC_KEYS, VQ_KEYS):
            preferred.extend([k for k in keys if k in feat_cols])
        rest = [c for c in feat_cols if c not in preferred]
        feat_cols = (preferred + rest)[:max_feats]

    corr = df[feat_cols].astype(float).corr()
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(
        corr, ax=ax, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
        xticklabels=False, yticklabels=False, square=True, cbar_kws={"shrink": 0.7},
    )
    ax.set_title(title)
    fig.tight_layout()
    return savefig(fig, out_path)


# ---------------------------------------------------------------------------
# Fig 6.4–6.11 — Line plots by emotion
# ---------------------------------------------------------------------------
def fig_line_by_emotion(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    title: str,
    out_path: Path,
) -> Optional[Path]:
    import matplotlib.pyplot as plt

    cols = [c for c in feature_cols if c in df.columns]
    if not cols:
        print(f"  skip {out_path.name}: no matching feature columns")
        return None

    apply_style()
    means = df.groupby("emotion")[cols].mean()
    emotions = list(means.index)
    x = np.arange(len(emotions))

    fig, ax = plt.subplots(figsize=(10, 5))
    for col in cols:
        ax.plot(x, means[col].values, marker="o", label=col.split("_sma")[0][:40])
    ax.set_xticks(x)
    ax.set_xticklabels(emotions, rotation=25, ha="right")
    ax.set_ylabel("Mean feature value")
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
    fig.tight_layout()
    return savefig(fig, out_path)


# ---------------------------------------------------------------------------
# Fig 6.12–6.13 — MFCC pairplots
# ---------------------------------------------------------------------------
def fig_mfcc_pairplot(df: pd.DataFrame, title: str, out_path: Path) -> Optional[Path]:
    import matplotlib.pyplot as plt
    import seaborn as sns

    cols = present_cols(df, MFCC_KEYS)
    if len(cols) < 2:
        # fuzzy match
        cols = [c for c in df.columns if c.startswith("mfcc") and "amean" in c][:4]
    if len(cols) < 2:
        print(f"  skip {out_path.name}: need MFCC amean columns")
        return None

    apply_style()
    sub = df[cols + ["emotion"]].dropna()
    # subsample for speed/legibility
    if len(sub) > 2500:
        sub = sub.sample(2500, random_state=42)

    g = sns.pairplot(
        sub, vars=cols, hue="emotion", diag_kind="kde",
        plot_kws={"s": 8, "alpha": 0.45}, height=2.2,
    )
    g.fig.suptitle(title, y=1.02)
    g.fig.tight_layout()
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    g.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(g.fig)
    print(f"  wrote {path}")
    return path


# ---------------------------------------------------------------------------
# Fig 6.14–6.17 — Top-10 importance + KDE
# ---------------------------------------------------------------------------
def _top_features(
    classical_dir: Path, df: pd.DataFrame, k: int = 10
) -> Tuple[List[str], Optional[np.ndarray]]:
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder

    kept_path = classical_dir / "kept_features.joblib"
    rf_path = classical_dir / "random_forest.joblib"

    if kept_path.exists():
        kept = list(joblib.load(kept_path))
    else:
        kept = [c for c in df.columns if c not in META]

    X = df[kept].astype(float).fillna(0.0)
    y = LabelEncoder().fit_transform(df["emotion"])

    if rf_path.exists():
        rf = joblib.load(rf_path)
        if hasattr(rf, "feature_importances_") and len(rf.feature_importances_) == len(kept):
            imp = rf.feature_importances_
        else:
            rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            rf.fit(X, y)
            imp = rf.feature_importances_
    else:
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        imp = rf.feature_importances_

    order = np.argsort(imp)[::-1][:k]
    return [kept[i] for i in order], imp[order]


def fig_top10_importance(names: Sequence[str], values: np.ndarray, title: str, out_path: Path) -> Path:
    import matplotlib.pyplot as plt

    apply_style()
    fig, ax = plt.subplots(figsize=(9, 5))
    y = np.arange(len(names))[::-1]
    ax.barh(y, values[::-1], color="steelblue")
    ax.set_yticks(y)
    ax.set_yticklabels([n[:48] for n in names][::-1], fontsize=8)
    ax.set_xlabel("Gini importance")
    ax.set_title(title)
    fig.tight_layout()
    return savefig(fig, out_path)


def fig_top10_kde(df: pd.DataFrame, names: Sequence[str], title: str, out_path: Path) -> Path:
    import matplotlib.pyplot as plt
    import seaborn as sns

    apply_style()
    n = len(names)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 2.6 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, feat in zip(axes, names):
        for emo, g in df.groupby("emotion"):
            vals = g[feat].astype(float).dropna()
            if len(vals) < 5:
                continue
            sns.kdeplot(vals, ax=ax, label=str(emo), linewidth=1.2)
        ax.set_title(feat[:42], fontsize=8)
        ax.set_xlabel("")
        ax.legend(fontsize=6, loc="best")
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle(title, y=1.01)
    fig.tight_layout()
    return savefig(fig, out_path)


# ---------------------------------------------------------------------------
# Fig 6.18–6.23 — Confusion matrices
# ---------------------------------------------------------------------------
def _split_xy(classical_dir: Path, df: pd.DataFrame):
    import joblib
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    kept_path = classical_dir / "kept_features.joblib"
    kept = list(joblib.load(kept_path)) if kept_path.exists() else [c for c in df.columns if c not in META]
    le_path = classical_dir / "label_encoder.joblib"
    if le_path.exists():
        le = joblib.load(le_path)
        y = le.transform(df["emotion"])
    else:
        le = LabelEncoder()
        y = le.fit_transform(df["emotion"])
    X = df[kept].astype(float).fillna(0.0).to_numpy()
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler_path = classical_dir / "scaler.joblib"
    if scaler_path.exists():
        scaler = joblib.load(scaler_path)
        x_test = scaler.transform(x_test)
    else:
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)
    return x_train, x_test, y_train, y_test, le, kept


def fig_confusion(y_true, y_pred, labels: Sequence[str], title: str, out_path: Path) -> Path:
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    apply_style()
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", ax=ax,
        xticklabels=labels, yticklabels=labels,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    return savefig(fig, out_path)


def classical_confusion(
    classical_dir: Path, df: pd.DataFrame, model_name: str, title: str, out_path: Path
) -> Optional[Path]:
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier

    x_train, x_test, y_train, y_test, le, kept = _split_xy(classical_dir, df)
    path = classical_dir / f"{model_name}.joblib"
    if path.exists():
        clf = joblib.load(path)
    elif model_name == "random_forest":
        clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        clf.fit(x_train, y_train)
    else:
        clf = XGBClassifier(
            n_estimators=100, learning_rate=0.1, random_state=42, n_jobs=-1, eval_metric="mlogloss"
        )
        clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    return fig_confusion(y_test, pred, list(le.classes_), title, out_path)


def dnn_confusion(dnn_dir: Path, dataset: str, title: str, out_path: Path) -> Optional[Path]:
    import joblib
    import torch
    from sklearn.model_selection import train_test_split

    from src.audio.train_dnn import EmotionLSTM

    metrics_path = dnn_dir / "dnn_metrics.json"
    use_aug = False
    if metrics_path.exists():
        use_aug = bool(json.loads(metrics_path.read_text(encoding="utf-8")).get("augment"))
    npz = dnn_dir / ("features_aug.npz" if use_aug else "features.npz")
    if not npz.exists():
        npz = dnn_dir / ("features.npz" if use_aug else "features_aug.npz")
    ckpt = dnn_dir / "best.pt"
    le_path = dnn_dir / "label_encoder.joblib"
    if not (npz.exists() and ckpt.exists() and le_path.exists()):
        print(f"  skip DNN CM: need feature cache, best.pt, label_encoder under {dnn_dir}")
        return None

    data = np.load(npz)
    X = data["x"] if "x" in data.files else data["X"]
    y = data["y"] if "y" in data.files else data["Y"]
    le = joblib.load(le_path)
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    norm_path = dnn_dir / "norm.npz"
    if norm_path.exists():
        norm = np.load(norm_path)
        X_test = (X_test - norm["mean"]) / norm["std"]

    bidirectional = dataset == "banglaser"
    hidden, layers, dense = (256, 2, 128) if bidirectional else (128, 3, 64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EmotionLSTM(
        n_feat=X.shape[-1],
        n_classes=len(le.classes_),
        bidirectional=bidirectional,
        hidden=hidden,
        layers=layers,
        dense=dense,
    ).to(device)
    state = torch.load(ckpt, map_location=device)
    if isinstance(state, dict) and "model" in state:
        model.load_state_dict(state["model"])
    else:
        model.load_state_dict(state)
    model.eval()

    preds = []
    with torch.no_grad():
        for i in range(0, len(X_test), 64):
            batch = torch.tensor(X_test[i : i + 64], dtype=torch.float32, device=device)
            logits = model(batch)
            preds.append(logits.argmax(dim=1).cpu().numpy())
    y_pred = np.concatenate(preds)
    return fig_confusion(y_test, y_pred, list(le.classes_), title, out_path)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def generate_dataset_figures(
    dataset: str,
    classical_dir: Path,
    dnn_dir: Path,
    out: Path,
    skip_dnn: bool = False,
) -> Dict[str, Optional[Path]]:
    written: Dict[str, Optional[Path]] = {}
    tag = "CREMA-D" if dataset == "crema_d" else "BanglaSER"
    print(f"\n=== {tag} ({dataset}) ===")
    df = load_features(classical_dir)
    if df is None:
        print("  skipping feature/model figures (no features.csv)")
        return written

    # heatmaps
    if dataset == "banglaser":
        written["6.2"] = fig_heatmap(df, "Fig 6.2 — Heatmap of BanglaSER features", out / "fig_6_2_heatmap_banglaser.png")
    else:
        written["6.3"] = fig_heatmap(df, "Fig 6.3 — Heatmap of CREMA-D features", out / "fig_6_3_heatmap_crema_d.png")

    loud = match_group(df, LOUDNESS_KEYS, "loudness")
    pitch = match_group(df, PITCH_KEYS, "F0semitone")
    mfcc = match_group(df, MFCC_KEYS, "mfcc")
    vq = present_cols(df, VQ_KEYS) or [
        c for c in df.columns
        if any(s in c.lower() for s in ("jitter", "shimmer", "hnr", "logrelf0"))
    ][:6]

    if dataset == "crema_d":
        written["6.4"] = fig_line_by_emotion(df, loud, "Fig 6.4 — Loudness features across emotions (CREMA-D)", out / "fig_6_4_loudness_crema_d.png")
        written["6.5"] = fig_line_by_emotion(df, pitch, "Fig 6.5 — Pitch features across emotions (CREMA-D)", out / "fig_6_5_pitch_crema_d.png")
        written["6.6"] = fig_line_by_emotion(df, mfcc, "Fig 6.6 — MFCC features across emotions (CREMA-D)", out / "fig_6_6_mfcc_crema_d.png")
        written["6.7"] = fig_line_by_emotion(df, vq, "Fig 6.7 — Voice quality across emotions (CREMA-D)", out / "fig_6_7_vq_crema_d.png")
        written["6.13"] = fig_mfcc_pairplot(df, "Fig 6.13 — MFCC pairplot (CREMA-D)", out / "fig_6_13_mfcc_pairplot_crema_d.png")
        written["6.15"] = None
        written["6.17"] = None
        names, vals = _top_features(classical_dir, df, k=10)
        written["6.15"] = fig_top10_importance(names, vals, "Fig 6.15 — Top 10 features (CREMA-D)", out / "fig_6_15_top10_crema_d.png")
        written["6.17"] = fig_top10_kde(df, names, "Fig 6.17 — KDE of top 10 features (CREMA-D)", out / "fig_6_17_kde_crema_d.png")
        written["6.18"] = classical_confusion(classical_dir, df, "random_forest", "Fig 6.18 — RF confusion (CREMA-D)", out / "fig_6_18_cm_rf_crema_d.png")
        written["6.20"] = classical_confusion(classical_dir, df, "xgboost", "Fig 6.20 — XGBoost confusion (CREMA-D)", out / "fig_6_20_cm_xgb_crema_d.png")
        if not skip_dnn:
            written["6.22"] = dnn_confusion(dnn_dir, dataset, "Fig 6.22 — DNN confusion (CREMA-D)", out / "fig_6_22_cm_dnn_crema_d.png")
    else:
        written["6.8"] = fig_line_by_emotion(df, loud, "Fig 6.8 — Loudness features across emotions (BanglaSER)", out / "fig_6_8_loudness_banglaser.png")
        written["6.9"] = fig_line_by_emotion(df, pitch, "Fig 6.9 — Pitch features across emotions (BanglaSER)", out / "fig_6_9_pitch_banglaser.png")
        written["6.10"] = fig_line_by_emotion(df, mfcc, "Fig 6.10 — MFCC across emotions (BanglaSER)", out / "fig_6_10_mfcc_banglaser.png")
        written["6.11"] = fig_line_by_emotion(df, vq, "Fig 6.11 — Voice quality across emotions (BanglaSER)", out / "fig_6_11_vq_banglaser.png")
        written["6.12"] = fig_mfcc_pairplot(df, "Fig 6.12 — MFCC pairplot (BanglaSER)", out / "fig_6_12_mfcc_pairplot_banglaser.png")
        names, vals = _top_features(classical_dir, df, k=10)
        written["6.14"] = fig_top10_importance(names, vals, "Fig 6.14 — Top 10 features (BanglaSER)", out / "fig_6_14_top10_banglaser.png")
        written["6.16"] = fig_top10_kde(df, names, "Fig 6.16 — KDE of top 10 features (BanglaSER)", out / "fig_6_16_kde_banglaser.png")
        written["6.19"] = classical_confusion(classical_dir, df, "random_forest", "Fig 6.19 — RF confusion (BanglaSER)", out / "fig_6_19_cm_rf_banglaser.png")
        written["6.21"] = classical_confusion(classical_dir, df, "xgboost", "Fig 6.21 — XGBoost confusion (BanglaSER)", out / "fig_6_21_cm_xgb_banglaser.png")
        if not skip_dnn:
            written["6.23"] = dnn_confusion(dnn_dir, dataset, "Fig 6.23 — DNN confusion (BanglaSER)", out / "fig_6_23_cm_dnn_banglaser.png")

    return written


def generate_all(
    out_dir: Optional[Path] = None,
    models_root: Optional[Path] = None,
    sample_wav: Optional[Path] = None,
    skip_dnn: bool = False,
) -> Dict[str, Optional[Path]]:
    warnings.filterwarnings("ignore", category=UserWarning)
    out = Path(out_dir or FIGURES)
    out.mkdir(parents=True, exist_ok=True)
    models_root = Path(models_root or MODELS)
    results: Dict[str, Optional[Path]] = {}

    print("Generating thesis figures ->", out)
    results["1.1"] = fig_1_1_architecture(out)
    results["6.1"] = fig_6_1_augmentation(out, sample_wav=sample_wav)

    for ds in ("crema_d", "banglaser"):
        classical = _resolve_classical(models_root, ds)
        dnn = _resolve_dnn(models_root, ds)
        results.update(
            generate_dataset_figures(ds, classical, dnn, out, skip_dnn=skip_dnn)
        )

    n_ok = sum(1 for v in results.values() if v is not None)
    print(f"\nDone: {n_ok}/{len(results)} figure files written under {out}")
    missing = [k for k, v in results.items() if v is None]
    if missing:
        print("Skipped / missing:", ", ".join(missing))
        print(
            "Tip: place classical features at "
            "models/audio/<crema_d|banglaser>/classical/features.csv "
            "(and joblib/DNN weights for CMs)."
        )
    return results
    return results


def main():
    p = argparse.ArgumentParser(description="Regenerate thesis LoF figures")
    p.add_argument("--out", type=Path, default=FIGURES)
    p.add_argument("--models-root", type=Path, default=None, help="Root containing audio/<ds>/classical")
    p.add_argument("--sample-wav", type=Path, default=None, help="Wav for Fig 6.1")
    p.add_argument("--skip-dnn", action="store_true")
    args = p.parse_args()
    generate_all(
        out_dir=args.out,
        models_root=args.models_root,
        sample_wav=args.sample_wav,
        skip_dnn=args.skip_dnn,
    )


if __name__ == "__main__":
    main()
