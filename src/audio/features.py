"""OpenSMILE eGeMAPS feature extraction + utterance table builder."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

_SMILE_FUNC = None
_SMILE_LLD = None


def _smile_functionals():
    global _SMILE_FUNC
    if _SMILE_FUNC is None:
        import opensmile

        print("Initializing OpenSMILE eGeMAPS functionals (once)...")
        _SMILE_FUNC = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals,
        )
    return _SMILE_FUNC


def _smile_lld():
    global _SMILE_LLD
    if _SMILE_LLD is None:
        import opensmile

        print("Initializing OpenSMILE eGeMAPS LLDs (once)...")
        _SMILE_LLD = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
        )
    return _SMILE_LLD


def extract_egemaps(path: Path) -> Dict[str, float]:
    smile = _smile_functionals()
    df = smile.process_file(str(path))
    row = df.iloc[0].to_dict()
    return {k: float(v) for k, v in row.items()}


def extract_lld_sequence(path: Path, max_frames: int = 200) -> np.ndarray:
    """Frame-level LLDs for LSTM (padded/truncated)."""
    smile = _smile_lld()
    df = smile.process_file(str(path))
    arr = df.to_numpy(dtype=np.float32)
    if arr.shape[0] >= max_frames:
        arr = arr[:max_frames]
    else:
        pad = np.zeros((max_frames - arr.shape[0], arr.shape[1]), dtype=np.float32)
        arr = np.vstack([arr, pad])
    return arr


CREMA_MAP = {
    "ANG": "anger",
    "DIS": "disgust",
    "FEA": "fear",
    "HAP": "happy",
    "NEU": "neutral",
    "SAD": "sad",
}


def parse_crema_label(path: Path) -> Optional[str]:
    # e.g. 1001_DFA_ANG_XX.wav
    parts = path.stem.split("_")
    if len(parts) >= 3 and parts[2] in CREMA_MAP:
        return CREMA_MAP[parts[2]]
    return None


BANGLASER_EMO = {
    # RAVDESS-style numeric id used by BanglaSER / B-SER filenames
    # Confirmed from dataset docs: 02 = sad; remaining map to thesis labels
    "01": "angry",
    "02": "sad",
    "03": "happy",
    "04": "surprised",
    "05": "natural",
}


def parse_banglaser_label(path: Path) -> Optional[str]:
    """Parse BanglaSER labels from folder name or 7-field filename."""
    name = path.stem.lower()
    parent = path.parent.name.lower()
    for lab in ("angry", "happy", "natural", "neutral", "sad", "surprised", "surprise"):
        if lab in name or lab in parent:
            if lab in ("surprise",):
                return "surprised"
            if lab == "neutral":
                return "natural"
            return lab

    # e.g. 03-01-02-01-03-03-02.wav -> emotion is 3rd field
    parts = path.stem.split("-")
    if len(parts) >= 3 and parts[2] in BANGLASER_EMO:
        return BANGLASER_EMO[parts[2]]
    return None


def build_feature_table(
    audio_dir: Path,
    dataset: str,
    cache_csv: Optional[Path] = None,
    chunk_size: int = 250,
) -> pd.DataFrame:
    """Extract functionals for all wavs. Reuses one Smile; resumes from cache_csv."""
    labeled: List[tuple[Path, str]] = []
    for wav in sorted(audio_dir.rglob("*.wav")):
        label = parse_crema_label(wav) if dataset == "crema_d" else parse_banglaser_label(wav)
        if label is not None:
            labeled.append((wav, label))
    if not labeled:
        raise FileNotFoundError(f"No labeled wav files under {audio_dir}")

    done_paths = set()
    rows: List[dict] = []
    if cache_csv is not None and cache_csv.exists():
        prev = pd.read_csv(cache_csv)
        rows = prev.to_dict(orient="records")
        done_paths = set(prev["path"].astype(str))
        print(f"Resuming feature extract: {len(done_paths)} already done")

    pending = [(p, lab) for p, lab in labeled if str(p) not in done_paths]
    print(f"Extracting OpenSMILE features for {len(pending)} / {len(labeled)} files")
    smile = _smile_functionals()

    for start in range(0, len(pending), chunk_size):
        batch = pending[start : start + chunk_size]
        paths = [str(p) for p, _ in batch]
        labels = [lab for _, lab in batch]
        feat_df = smile.process_files(paths)
        # process_files returns MultiIndex (file, start, end) or similar
        if isinstance(feat_df.index, pd.MultiIndex):
            # one row per file typically at level 0
            file_level = feat_df.index.get_level_values(0)
            for path_str, lab in zip(paths, labels):
                # match by path string / basename
                mask = file_level.astype(str) == path_str
                if not mask.any():
                    # fallback: match basename
                    mask = pd.Index([Path(x).name for x in file_level.astype(str)]) == Path(path_str).name
                    sub = feat_df.loc[mask]
                else:
                    sub = feat_df.loc[mask]
                if len(sub) == 0:
                    print("skip (no features):", path_str)
                    continue
                row = {k: float(v) for k, v in sub.iloc[0].to_dict().items()}
                row["path"] = path_str
                row["emotion"] = lab
                rows.append(row)
        else:
            for i, (path_str, lab) in enumerate(zip(paths, labels)):
                row = {k: float(v) for k, v in feat_df.iloc[i].to_dict().items()}
                row["path"] = path_str
                row["emotion"] = lab
                rows.append(row)

        print(f"  {min(start + chunk_size, len(pending))}/{len(pending)}")
        if cache_csv is not None:
            cache_csv.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(cache_csv, index=False)

    return pd.DataFrame(rows)
