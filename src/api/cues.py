"""Detection scores plus the acoustic profiles used on the interface."""
from __future__ import annotations

import re
from functools import lru_cache
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

from src import ROOT

FINAL = ROOT / "models_final" / "audio"

# Same measurements as the thesis line plots and top-10 charts.
PROFILE_FEATURES = {
    "banglaser": [
        "loudness_sma3_stddevRisingSlope",
        "loudness_sma3_stddevFallingSlope",
        "loudness_sma3_pctlrange0-2",
        "F0semitoneFrom27.5Hz_sma3nz_pctlrange0-2",
    ],
    "crema_d": [
        "spectralFluxV_sma3nz_amean",
        "spectralFluxUV_sma3nz_amean",
        "loudness_sma3_stddevFallingSlope",
        "loudness_sma3_meanRisingSlope",
    ],
}
EMOTION_ORDER = {
    "banglaser": ["angry", "happy", "natural", "sad", "surprised"],
    "crema_d": ["anger", "disgust", "fear", "happy", "neutral", "sad"],
}
# Small-multiple groups. Each value is the corpus z-score, averaged inside the group.
FAMILY_FEATURES = {
    "banglaser": [
        (
            "Loudness slope",
            [
                "loudness_sma3_stddevRisingSlope",
                "loudness_sma3_stddevFallingSlope",
                "loudness_sma3_meanRisingSlope",
                "loudness_sma3_meanFallingSlope",
            ],
        ),
        (
            "Loudness level",
            [
                "loudness_sma3_amean",
                "loudness_sma3_pctlrange0-2",
                "loudness_sma3_percentile20.0",
                "loudness_sma3_percentile80.0",
            ],
        ),
        (
            "Pitch",
            [
                "F0semitoneFrom27.5Hz_sma3nz_amean",
                "F0semitoneFrom27.5Hz_sma3nz_pctlrange0-2",
                "F0semitoneFrom27.5Hz_sma3nz_stddevNorm",
            ],
        ),
        (
            "Voice quality",
            [
                "jitterLocal_sma3nz_amean",
                "shimmerLocaldB_sma3nz_amean",
                "HNRdBACF_sma3nz_amean",
            ],
        ),
    ],
    "crema_d": [
        (
            "Spectral flux",
            [
                "spectralFluxV_sma3nz_amean",
                "spectralFluxUV_sma3nz_amean",
                "spectralFlux_sma3_amean",
            ],
        ),
        (
            "Loudness",
            [
                "loudness_sma3_amean",
                "loudness_sma3_stddevFallingSlope",
                "loudness_sma3_meanRisingSlope",
                "loudness_sma3_pctlrange0-2",
            ],
        ),
        (
            "Pitch",
            [
                "F0semitoneFrom27.5Hz_sma3nz_amean",
                "F0semitoneFrom27.5Hz_sma3nz_pctlrange0-2",
                "F0semitoneFrom27.5Hz_sma3nz_stddevNorm",
            ],
        ),
        (
            "Voice quality",
            [
                "jitterLocal_sma3nz_amean",
                "shimmerLocaldB_sma3nz_amean",
                "HNRdBACF_sma3nz_amean",
            ],
        ),
    ],
}

# The 88 eGeMAPSv02 functionals in the report's feature table.
FEATURE_GROUPS = (
    (
        "Pitch and frequency",
        [
            "F0semitoneFrom27.5Hz_sma3nz_amean",
            "F0semitoneFrom27.5Hz_sma3nz_stddevNorm",
            "F0semitoneFrom27.5Hz_sma3nz_percentile20.0",
            "F0semitoneFrom27.5Hz_sma3nz_percentile50.0",
            "F0semitoneFrom27.5Hz_sma3nz_percentile80.0",
            "F0semitoneFrom27.5Hz_sma3nz_pctlrange0-2",
            "F0semitoneFrom27.5Hz_sma3nz_meanRisingSlope",
            "F0semitoneFrom27.5Hz_sma3nz_stddevRisingSlope",
            "F0semitoneFrom27.5Hz_sma3nz_meanFallingSlope",
            "F0semitoneFrom27.5Hz_sma3nz_stddevFallingSlope",
            "jitterLocal_sma3nz_amean",
            "jitterLocal_sma3nz_stddevNorm",
            "F1frequency_sma3nz_amean",
            "F1frequency_sma3nz_stddevNorm",
            "F2frequency_sma3nz_amean",
            "F2frequency_sma3nz_stddevNorm",
            "F3frequency_sma3nz_amean",
            "F3frequency_sma3nz_stddevNorm",
            "F1bandwidth_sma3nz_amean",
            "F1bandwidth_sma3nz_stddevNorm",
            "F2bandwidth_sma3nz_amean",
            "F2bandwidth_sma3nz_stddevNorm",
            "F3bandwidth_sma3nz_amean",
            "F3bandwidth_sma3nz_stddevNorm",
        ],
    ),
    (
        "Loudness and energy",
        [
            "loudness_sma3_amean",
            "loudness_sma3_stddevNorm",
            "loudness_sma3_percentile20.0",
            "loudness_sma3_percentile50.0",
            "loudness_sma3_percentile80.0",
            "loudness_sma3_pctlrange0-2",
            "loudness_sma3_meanRisingSlope",
            "loudness_sma3_stddevRisingSlope",
            "loudness_sma3_meanFallingSlope",
            "loudness_sma3_stddevFallingSlope",
            "shimmerLocaldB_sma3nz_amean",
            "shimmerLocaldB_sma3nz_stddevNorm",
            "loudnessPeaksPerSec",
        ],
    ),
    (
        "Spectrum",
        [
            "alphaRatioV_sma3nz_amean",
            "alphaRatioV_sma3nz_stddevNorm",
            "hammarbergIndexV_sma3nz_amean",
            "hammarbergIndexV_sma3nz_stddevNorm",
            "slopeV0-500_sma3nz_amean",
            "slopeV0-500_sma3nz_stddevNorm",
            "slopeV500-1500_sma3nz_amean",
            "slopeV500-1500_sma3nz_stddevNorm",
            "alphaRatioUV_sma3nz_amean",
            "hammarbergIndexUV_sma3nz_amean",
            "slopeUV0-500_sma3nz_amean",
            "slopeUV500-1500_sma3nz_amean",
            "spectralFlux_sma3_amean",
            "spectralFlux_sma3_stddevNorm",
            "spectralFluxV_sma3nz_amean",
            "spectralFluxV_sma3nz_stddevNorm",
            "spectralFluxUV_sma3nz_amean",
        ],
    ),
    (
        "MFCC",
        [
            "mfcc1_sma3_amean",
            "mfcc1_sma3_stddevNorm",
            "mfcc2_sma3_amean",
            "mfcc2_sma3_stddevNorm",
            "mfcc3_sma3_amean",
            "mfcc3_sma3_stddevNorm",
            "mfcc4_sma3_amean",
            "mfcc4_sma3_stddevNorm",
            "mfcc1V_sma3nz_amean",
            "mfcc1V_sma3nz_stddevNorm",
            "mfcc2V_sma3nz_amean",
            "mfcc2V_sma3nz_stddevNorm",
            "mfcc3V_sma3nz_amean",
            "mfcc3V_sma3nz_stddevNorm",
            "mfcc4V_sma3nz_amean",
            "mfcc4V_sma3nz_stddevNorm",
        ],
    ),
    (
        "Harmonics and formant energy",
        [
            "HNRdBACF_sma3nz_amean",
            "HNRdBACF_sma3nz_stddevNorm",
            "logRelF0-H1-H2_sma3nz_amean",
            "logRelF0-H1-H2_sma3nz_stddevNorm",
            "logRelF0-H1-A3_sma3nz_amean",
            "logRelF0-H1-A3_sma3nz_stddevNorm",
            "F1amplitudeLogRelF0_sma3nz_amean",
            "F1amplitudeLogRelF0_sma3nz_stddevNorm",
            "F2amplitudeLogRelF0_sma3nz_amean",
            "F2amplitudeLogRelF0_sma3nz_stddevNorm",
            "F3amplitudeLogRelF0_sma3nz_amean",
            "F3amplitudeLogRelF0_sma3nz_stddevNorm",
        ],
    ),
    (
        "Timing and level",
        [
            "VoicedSegmentsPerSec",
            "MeanVoicedSegmentLengthSec",
            "StddevVoicedSegmentLengthSec",
            "MeanUnvoicedSegmentLength",
            "StddevUnvoicedSegmentLength",
            "equivalentSoundLevel_dBp",
        ],
    ),
)

SPOKES = {
    "Level": ["loudness_sma3_amean"],
    "Movement": [
        "loudness_sma3_stddevRisingSlope",
        "loudness_sma3_stddevFallingSlope",
        "loudness_sma3_meanRisingSlope",
        "loudness_sma3_meanFallingSlope",
    ],
    "Pitch": ["F0semitoneFrom27.5Hz_sma3nz_amean"],
    "Timbre change": [
        "spectralFlux_sma3_amean",
        "spectralFluxV_sma3nz_amean",
        "spectralFluxUV_sma3nz_amean",
    ],
    "Voice quality": [
        "jitterLocal_sma3nz_amean",
        "shimmerLocaldB_sma3nz_amean",
        "HNRdBACF_sma3nz_amean",
    ],
}
# Higher HNR is a clearer voice, so the spoke grows when the voice is rougher.
INVERT = {"HNRdBACF_sma3nz_amean"}

GLOSS = {
    "loudness_sma3_amean": "overall loudness",
    "loudness_sma3_stddevNorm": "loudness variation",
    "loudness_sma3_percentile20.0": "loudness of the quiet parts",
    "loudness_sma3_percentile80.0": "loudness of the loud parts",
    "loudness_sma3_meanRisingSlope": "the rise in loudness",
    "loudness_sma3_meanFallingSlope": "the fall in loudness",
    "loudness_sma3_stddevRisingSlope": "how sharply loudness surges",
    "loudness_sma3_stddevFallingSlope": "how sharply loudness falls",
    "spectralFlux_sma3_amean": "how fast the timbre changes",
    "spectralFluxV_sma3nz_amean": "timbre change while the voice is sounding",
    "spectralFluxUV_sma3nz_amean": "timbre change in the unvoiced parts",
    "F0semitoneFrom27.5Hz_sma3nz_amean": "how high the voice sits",
    "F0semitoneFrom27.5Hz_sma3nz_stddevNorm": "pitch variation",
    "jitterLocal_sma3nz_amean": "pitch roughness",
    "shimmerLocaldB_sma3nz_amean": "loudness roughness",
    "HNRdBACF_sma3nz_amean": "harmonic clarity",
    "mfcc1_sma3_amean": "the broad shape of the spectrum",
    "mfcc2_sma3_amean": "the second spectral contour",
}


def gloss(feature: str) -> str:
    if feature in GLOSS:
        return GLOSS[feature]
    text = feature
    for raw, pretty in (
        ("_sma3nz", ""),
        ("_sma3", ""),
        ("stddevNorm", " variation"),
        ("stddevRisingSlope", " surge"),
        ("stddevFallingSlope", " fall"),
        ("meanRisingSlope", " rise"),
        ("meanFallingSlope", " fall"),
        ("percentile20.0", " low end"),
        ("percentile50.0", " midpoint"),
        ("percentile80.0", " high end"),
        ("pctlrange0-2", " range"),
        ("amean", ""),
        ("mfcc1V", "first voiced cepstrum"),
        ("mfcc2V", "second voiced cepstrum"),
        ("mfcc3V", "third voiced cepstrum"),
        ("mfcc4V", "fourth voiced cepstrum"),
        ("mfcc1", "first cepstrum"),
        ("mfcc2", "second cepstrum"),
        ("mfcc3", "third cepstrum"),
        ("mfcc4", "fourth cepstrum"),
        ("hammarbergIndexV", "voiced spectral tilt"),
        ("hammarbergIndexUV", "unvoiced spectral tilt"),
        ("hammarbergIndex", "spectral tilt"),
        ("slopeUV0-500", "unvoiced low-frequency slope"),
        ("slopeUV500-1500", "unvoiced mid-frequency slope"),
        ("slopeV0-500", "voiced low-frequency slope"),
        ("slopeV500-1500", "voiced mid-frequency slope"),
        ("spectralFluxUV", "unvoiced timbre change"),
        ("spectralFluxV", "voiced timbre change"),
        ("spectralFlux", "timbre change"),
        ("alphaRatioUV", "unvoiced brightness"),
        ("alphaRatioV", "voiced brightness"),
        ("alphaRatio", "brightness"),
        ("StddevVoicedSegmentLengthSec", "voiced-segment spread"),
        ("MeanVoicedSegmentLengthSec", "average voiced-segment length"),
        ("StddevUnvoicedSegmentLengthSec", "unvoiced-segment spread"),
        ("MeanUnvoicedSegmentLengthSec", "average unvoiced-segment length"),
        ("equivalentSoundLevel_dBp", "equivalent sound level"),
        ("loudnessPeaksPerSec", "loudness peaks per second"),
        ("VoicedSegmentsPerSec", "voiced segments per second"),
        ("logRelF0-H1-H2", "low-harmonic difference"),
        ("logRelF0-H1-A3", "high-harmonic difference"),
        ("F3amplitudeLogRelF0", "third-formant energy"),
        ("F2amplitudeLogRelF0", "second-formant energy"),
        ("F1amplitudeLogRelF0", "first-formant energy"),
        ("F3bandwidth", "third-formant bandwidth"),
        ("F2bandwidth", "second-formant bandwidth"),
        ("F1bandwidth", "first-formant bandwidth"),
        ("F3frequency", "third formant"),
        ("F2frequency", "second formant"),
        ("F1frequency", "first formant"),
        ("F0frequency", "pitch"),
        ("F0semitoneFrom27.5Hz", "pitch"),
        ("jitterLocal", "pitch roughness"),
        ("shimmerLocaldB", "loudness roughness"),
        ("HNRdBACF", "harmonic clarity"),
        ("loudness", "loudness"),
    ):
        text = text.replace(raw, pretty)
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    return " ".join(text.replace("_", " ").split()).lower()


class VoiceLibrary:
    def __init__(self, dataset: str):
        folder = FINAL / dataset / "classical"
        self.dataset = dataset
        self.kept: List[str] = list(joblib.load(folder / "kept_features.joblib"))
        self.scaler = joblib.load(folder / "scaler.joblib")
        self.encoder = joblib.load(folder / "label_encoder.joblib")
        self.model = joblib.load(folder / "xgboost.joblib")
        self.labels = [str(c) for c in self.encoder.classes_]
        frame = pd.read_csv(folder / "features.csv")
        self.frame = frame
        cols = [c for c in frame.columns if c not in {"path", "emotion"}]
        self.mean = frame[cols].mean()
        self.std = frame[cols].std().replace(0, np.nan).fillna(1.0)
        self.proto = frame.groupby("emotion")[cols].mean()

    def z(self, feature: str, value: float) -> float:
        return float((value - self.mean[feature]) / self.std[feature])

    def spoke_values(self, values: pd.Series) -> List[float]:
        out = []
        for feats in SPOKES.values():
            zs = []
            for feat in feats:
                if feat not in values.index:
                    continue
                sign = -1.0 if feat in INVERT else 1.0
                zs.append(sign * self.z(feat, float(values[feat])))
            out.append(float(np.mean(zs)) if zs else 0.0)
        return out

    def prototype_spokes(self, emotion: str) -> List[float]:
        if emotion not in self.proto.index:
            return [0.0] * len(SPOKES)
        return self.spoke_values(self.proto.loc[emotion])

    def profiles(self, row: pd.Series, predicted: str) -> List[dict]:
        order = [e for e in EMOTION_ORDER[self.dataset] if e in self.proto.index]
        charts = []
        for feat in PROFILE_FEATURES[self.dataset]:
            if feat not in self.proto.columns:
                continue
            means = [float(self.proto.loc[e, feat]) for e in order]
            clip = float(row[feat]) if feat in row.index and pd.notna(row[feat]) else float("nan")
            nearest = order[int(np.argmin([abs(m - clip) for m in means]))]
            charts.append(
                {
                    "feature": feat,
                    "gloss": gloss(feat),
                    "emotions": order,
                    "means": means,
                    "clip": clip,
                    "nearest": nearest,
                    "predicted": predicted,
                }
            )
        return charts

    def _present(self, feats: List[str]) -> List[str]:
        return [f for f in feats if f in self.mean.index and f in self.proto.columns]

    def _mean_z(self, feats: List[str], value_of) -> float:
        return float(np.mean([self.z(f, value_of(f)) for f in feats]))

    def families(self, row: pd.Series) -> List[dict]:
        order = [e for e in EMOTION_ORDER[self.dataset] if e in self.proto.index]
        charts = []
        for name, feats in FAMILY_FEATURES[self.dataset]:
            present = self._present(feats)
            if not present:
                continue
            means = [
                self._mean_z(present, lambda f, emotion=e: float(self.proto.loc[emotion, f]))
                for e in order
            ]
            clip = self._mean_z(
                present,
                lambda f: float(row[f]) if f in row.index and pd.notna(row[f]) else float(self.mean[f]),
            )
            nearest = order[int(np.argmin([abs(m - clip) for m in means]))]
            charts.append(
                {
                    "name": name,
                    "emotions": order,
                    "means": means,
                    "clip": clip,
                    "nearest": nearest,
                }
            )
        return charts

    def _chart(self, feat: str, order: List[str], row: pd.Series, predicted: str) -> dict:
        means = [float(self.proto.loc[e, feat]) for e in order]
        clip = float(row[feat]) if feat in row.index and pd.notna(row[feat]) else float("nan")
        nearest = order[int(np.argmin([abs(m - clip) for m in means]))]
        return {
            "feature": feat,
            "gloss": gloss(feat),
            "emotions": order,
            "means": means,
            "clip": clip,
            "nearest": nearest,
            "predicted": predicted,
        }

    def groups(self, row: pd.Series, predicted: str) -> List[dict]:
        order = [e for e in EMOTION_ORDER[self.dataset] if e in self.proto.index]
        out = []
        for name, feats in FEATURE_GROUPS:
            charts = [self._chart(feat, order, row, predicted) for feat in feats if feat in self.proto.columns]
            if charts:
                out.append({"name": name, "charts": charts})
        return out

    def _ensure_kde(self) -> None:
        if getattr(self, "_kde_ready", False):
            return
        from scipy.stats import gaussian_kde

        rf = joblib.load(FINAL / self.dataset / "classical" / "random_forest.joblib")
        importance = np.asarray(rf.feature_importances_, dtype=float)
        order = np.argsort(importance)[::-1][:10]
        self._top = [(self.kept[int(i)], float(importance[int(i)])) for i in order]
        self._kde = {}
        for feat, _score in self._top:
            fitted = {}
            for emotion, group in self.frame.groupby("emotion"):
                vals = group[feat].astype(float).to_numpy()
                vals = vals[np.isfinite(vals)]
                model = None
                if len(vals) >= 5 and float(np.std(vals)) > 0:
                    try:
                        model = gaussian_kde(vals)
                    except Exception:
                        model = None
                fitted[str(emotion)] = {"model": model, "values": vals}
            self._kde[feat] = fitted
        self._kde_ready = True

    def kdes(self, row: pd.Series) -> List[dict]:
        self._ensure_kde()
        plots = []
        for feat, importance in self._top:
            fitted = self._kde[feat]
            clip = float(row[feat]) if feat in row.index and pd.notna(row[feat]) else float(self.mean[feat])
            columns = [item["values"] for item in fitted.values() if len(item["values"])]
            pooled = np.concatenate(columns) if columns else np.array([clip])
            lo = float(np.percentile(pooled, 2))
            hi = float(np.percentile(pooled, 98))
            pad = (hi - lo) * 0.08 or 0.05
            lo -= pad
            hi += pad
            lo = min(lo, clip)
            hi = max(hi, clip)
            if hi <= lo:
                hi = lo + 1.0
            xs = np.linspace(lo, hi, 64)
            curves = []
            for emotion in EMOTION_ORDER[self.dataset]:
                item = fitted.get(emotion)
                if item is None or item["model"] is None:
                    continue
                model = item["model"]
                ys = np.asarray(model(xs), dtype=float)
                at = float(np.asarray(model([clip])).reshape(-1)[0])
                curves.append(
                    {
                        "emotion": emotion,
                        "y": [round(float(v), 5) for v in ys],
                        "at": round(at, 5),
                    }
                )
            nearest = max(curves, key=lambda curve: curve["at"])["emotion"] if curves else ""
            plots.append(
                {
                    "feature": feat,
                    "gloss": gloss(feat),
                    "importance": round(importance, 4),
                    "x": [round(float(v), 5) for v in xs],
                    "curves": curves,
                    "clip": clip,
                    "nearest": nearest,
                }
            )
        return plots

    def score_row(self, row: pd.Series) -> dict:
        raw = np.array([[float(row[c]) if pd.notna(row[c]) else 0.0 for c in self.kept]], dtype=np.float32)
        scaled = self.scaler.transform(raw)
        proba = self.model.predict_proba(scaled)[0]
        order = np.argsort(-proba)
        top = int(order[0])
        second = int(order[1])
        pred = self.labels[top]
        runner = self.labels[second]
        import xgboost as xgb

        contrib = self.model.get_booster().predict(xgb.DMatrix(scaled), pred_contribs=True)
        contrib = np.asarray(contrib)
        # (1, n_classes, n_features + bias)
        weights = contrib[0, top, :-1]
        ranked = [int(i) for i in np.argsort(-np.abs(weights))]
        contributions = [
            {
                "feature": self.kept[idx],
                "gloss": gloss(self.kept[idx]),
                "value": float(weights[idx]),
            }
            for idx in ranked[:8]
        ]
        preferred = [i for i in ranked[:10] if self.kept[i] in GLOSS]
        order_idx = preferred + [i for i in ranked if i not in preferred]
        cues = []
        for idx in order_idx:
            if len(cues) == 3:
                break
            feat = self.kept[idx]
            if feat not in self.mean.index:
                continue
            clip_z = self.z(feat, float(row[feat]) if pd.notna(row[feat]) else 0.0)
            pred_z = self.z(feat, float(self.proto.loc[pred, feat])) if pred in self.proto.index else 0.0
            run_z = self.z(feat, float(self.proto.loc[runner, feat])) if runner in self.proto.index else 0.0
            cues.append(
                {
                    "feature": feat,
                    "gloss": gloss(feat),
                    "sentence": _sentence(gloss(feat), clip_z, pred_z, pred),
                    "clip": clip_z,
                    "predicted": pred_z,
                    "runner_up": run_z,
                    "contribution": float(weights[int(idx)]),
                }
            )
        profiles = self.profiles(row, pred)
        return {
            "language": "bn" if self.dataset == "banglaser" else "en",
            "dataset": self.dataset,
            "head": "BanglaSER XGBoost" if self.dataset == "banglaser" else "CREMA-D XGBoost",
            "emotion": pred,
            "runner_up": runner,
            "confidence": float(proba[top]),
            "margin": float(proba[top] - proba[second]),
            "probabilities": [
                {"label": self.labels[i], "probability": float(proba[i])}
                for i in order
            ],
            "profiles": profiles,
            "families": self.families(row),
            "groups": self.groups(row, pred),
            "kdes": self.kdes(row),
            "contributions": contributions,
            "profile_hits": sum(1 for chart in profiles if chart["nearest"] == pred),
            "cues": cues,
            "true_label": str(row["emotion"]) if "emotion" in row.index else None,
        }

    def pick(self, emotion: Optional[str] = None) -> pd.Series:
        pool = self.frame
        if emotion:
            matched = pool[pool["emotion"] == emotion]
            if len(matched):
                pool = matched
        if emotion in self.labels and len(pool) > 1:
            raw = pool[self.kept].fillna(0).to_numpy(dtype=np.float32)
            target = self.model.predict_proba(self.scaler.transform(raw))[:, self.labels.index(emotion)]
            strong = np.flatnonzero(target >= 0.85)
            if len(strong) == 0:
                strong = np.flatnonzero(target >= np.quantile(target, 0.8))
            feats = [f for f in PROFILE_FEATURES[self.dataset] if f in pool.columns and f in self.proto.columns]
            if len(strong) and feats and emotion in self.proto.index:
                proto = self.proto.loc[emotion, feats].to_numpy(dtype=float)
                scale = self.std[feats].to_numpy(dtype=float)
                scale = np.where(scale == 0, 1.0, scale)
                vals = pool.iloc[strong][feats].to_numpy(dtype=float)
                dist = np.nanmean(np.abs(vals - proto) / scale, axis=1)
                return pool.iloc[int(strong[int(np.nanargmin(dist))])]
        return pool.sample(1).iloc[0]


def _sentence(name: str, clip_z: float, proto_z: float, label: str) -> str:
    gap = clip_z - proto_z
    if gap > 0.25:
        relation = "higher"
    elif gap < -0.25:
        relation = "lower"
    else:
        return f"{name.capitalize()} matches the average {label} clip."
    return f"{name.capitalize()} is {relation} than the average {label} clip."


@lru_cache(maxsize=2)
def library(dataset: str) -> VoiceLibrary:
    return VoiceLibrary(dataset)
