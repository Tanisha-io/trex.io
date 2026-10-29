"""Random Forest + XGBoost on VIF-pruned OpenSMILE functionals."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

from src import DATA, MODELS
from src.audio.features import build_feature_table
from src.audio.vif_select import vif_select


META = {"path", "emotion"}


def train_dataset(dataset: str, audio_dir: Path, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    table_path = out_dir / "features.csv"
    if table_path.exists() and len(pd.read_csv(table_path)) > 100:
        df = pd.read_csv(table_path)
        print(f"Loaded cached features: {len(df)} rows from {table_path}")
    else:
        df = build_feature_table(audio_dir, dataset, cache_csv=table_path, chunk_size=250)
        df.to_csv(table_path, index=False)
        print(f"Wrote {len(df)} feature rows -> {table_path}")

    feat_cols = [c for c in df.columns if c not in META]
    kept, hist = vif_select(df, feat_cols, threshold=10.0)
    hist.to_csv(out_dir / "vif_dropped.csv", index=False)
    joblib.dump(kept, out_dir / "kept_features.joblib")

    le = LabelEncoder()
    y = le.fit_transform(df["emotion"])
    x = df[kept].astype(float).fillna(0.0).to_numpy()
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    results = {}
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(x_train, y_train)
    pred = rf.predict(x_test)
    results["random_forest"] = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "macro_f1": float(f1_score(y_test, pred, average="macro")),
    }
    print("RF", results["random_forest"])
    print(classification_report(y_test, pred, target_names=list(le.classes_), zero_division=0))
    joblib.dump(rf, out_dir / "random_forest.joblib")

    xgb = XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42, n_jobs=-1, eval_metric="mlogloss")
    xgb.fit(x_train, y_train)
    pred = xgb.predict(x_test)
    results["xgboost"] = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "macro_f1": float(f1_score(y_test, pred, average="macro")),
    }
    print("XGB", results["xgboost"])
    print(classification_report(y_test, pred, target_names=list(le.classes_), zero_division=0))
    joblib.dump(xgb, out_dir / "xgboost.joblib")
    joblib.dump(scaler, out_dir / "scaler.joblib")
    joblib.dump(le, out_dir / "label_encoder.joblib")
    (out_dir / "classical_metrics.json").write_text(json.dumps(results, indent=2))
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["crema_d", "banglaser"], required=True)
    args = p.parse_args()
    audio_dir = DATA / "audio" / args.dataset
    out = MODELS / "audio" / args.dataset / "classical"
    train_dataset(args.dataset, audio_dir, out)


if __name__ == "__main__":
    main()
