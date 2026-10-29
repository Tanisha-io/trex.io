"""LSTM (CREMA-D) / BiLSTM (BanglaSER) on OpenSMILE LLD sequences.

Resumable: incremental features.npz build + optional continue from best.pt.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional, Tuple

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset

from src import DATA, MODELS, device_str
from src.audio.augment import augment_once, load_wav, save_wav
from src.audio.features import extract_lld_sequence, parse_banglaser_label, parse_crema_label


class SeqDataset(Dataset):
    def __init__(self, xs: np.ndarray, ys: np.ndarray):
        self.xs = torch.tensor(xs, dtype=torch.float32)
        self.ys = torch.tensor(ys, dtype=torch.long)

    def __len__(self):
        return len(self.ys)

    def __getitem__(self, i):
        return self.xs[i], self.ys[i]


class EmotionLSTM(nn.Module):
    def __init__(
        self,
        n_feat: int,
        n_classes: int,
        bidirectional: bool = False,
        hidden: int = 128,
        layers: int = 3,
        dense: int = 64,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_feat,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=0.2 if layers > 1 else 0.0,
        )
        h_out = hidden * (2 if bidirectional else 1)
        self.bidirectional = bidirectional
        self.head = nn.Sequential(
            nn.Linear(h_out, dense),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(dense, n_classes),
        )

    def forward(self, x):
        out, (h_n, _c_n) = self.lstm(x)
        if self.bidirectional:
            # h_n is (layers * 2, batch, hidden), forward then backward per layer.
            # The last forward state has walked the whole clip. The last backward
            # state has walked from the end back to the start. out[:, -1, hidden:]
            # is only the first backward step, whose previous hidden is still 0.
            summary = torch.cat((h_n[-2], h_n[-1]), dim=-1)
        else:
            summary = out[:, -1, :]
        return self.head(summary)


def list_labeled_wavs(audio_dir: Path, dataset: str) -> List[Tuple[Path, str]]:
    items = []
    for wav in sorted(audio_dir.rglob("*.wav")):
        lab = parse_crema_label(wav) if dataset == "crema_d" else parse_banglaser_label(wav)
        if lab:
            items.append((wav, lab))
    return items


def filter_readable(items: List[Tuple[Path, str]]) -> List[Tuple[Path, str]]:
    """Drop clips that neither soundfile nor librosa can open (rare CREMA-D bad files)."""
    ok = []
    for path, lab in items:
        try:
            y, _ = load_wav(path)
            if len(y) < 16:
                raise ValueError("too short")
            ok.append((path, lab))
        except Exception as e:
            print(f"  drop unreadable {path.name}: {e}")
    print(f"readable wavs: {len(ok)}/{len(items)}")
    return ok


def maybe_augment_dir(
    items: List[Tuple[Path, str]], dataset: str, ratio: int, cache: Path
) -> List[Tuple[Path, str]]:
    cache.mkdir(parents=True, exist_ok=True)
    out = list(items)
    total = len(items) * ratio
    done = 0
    skipped = 0
    for path, lab in items:
        try:
            y0, sr0 = load_wav(path)
        except Exception as e:
            skipped += 1
            print(f"  skip unreadable {path.name}: {e}")
            done += ratio
            continue
        for k in range(ratio):
            dest = cache / f"{path.stem}_aug{k}.wav"
            if not dest.exists():
                try:
                    y = augment_once(y0.copy(), sr0, dataset=dataset)
                    save_wav(dest, y, sr0)
                except Exception as e:
                    skipped += 1
                    print(f"  skip aug {path.name}#{k}: {e}")
                    done += 1
                    continue
            if dest.exists():
                out.append((dest, lab))
            done += 1
            if done % 200 == 0 or done >= total:
                print(f"  augment {min(done, total)}/{total} (skipped={skipped})")
    print(f"augment done: {len(out)} clips (orig+aug), skipped_ops={skipped}")
    return out


TRAIN_DNN_PATCH = "bilstm-readout-v3-20261019"
READOUT_BIDIRECTIONAL = "forward_end+backward_start"
READOUT_FORWARD = "last_frame"


def _atomic_savez_compressed(path: Path, **arrays) -> None:
    """Write npz via temp file so a crash mid-write can't leave a corrupt cache.

    NumPy appends '.npz' unless the filename already ends with '.npz', so temps must
    use a '*.tmp.npz' name (never '*.npz.tmp').
    """
    import os
    import tempfile

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".tmp.npz", dir=str(path.parent))
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        # mkstemp created an empty file; remove so savez creates cleanly
        tmp.unlink(missing_ok=True)
        np.savez_compressed(str(tmp), **arrays)
        if not tmp.exists():
            # Defensive: some NumPy builds append an extra .npz
            alt = Path(str(tmp) + ".npz")
            if alt.exists():
                tmp = alt
            else:
                raise FileNotFoundError(f"savez did not create {tmp}")
        os.replace(str(tmp), str(path))
    except Exception:
        for junk in (tmp, Path(str(tmp) + ".npz")):
            try:
                junk.unlink(missing_ok=True)
            except OSError:
                pass
        # Last resort: direct write (better than crashing the whole run)
        np.savez_compressed(str(path), **arrays)


def build_arrays_resumable(
    items: List[Tuple[Path, str]],
    cache_npz: Path,
    max_frames: int = 200,
    save_every: int = 100,
) -> Tuple[np.ndarray, np.ndarray, LabelEncoder]:
    """Extract LLDs with checkpointing so Colab disconnects don't lose hours."""
    labels = [lab for _, lab in items]
    le = LabelEncoder()
    y_enc = le.fit_transform(labels)
    paths = [str(p) for p, _ in items]

    xs: List[Optional[np.ndarray]] = [None] * len(items)
    start_i = 0
    partial = cache_npz.with_name(cache_npz.stem + "_partial.npz")
    if partial.exists():
        try:
            blob = np.load(partial, allow_pickle=True)
            saved_paths = list(blob["paths"])
            saved_x = blob["x"]
            if len(saved_paths) <= len(paths) and all(
                saved_paths[i] == paths[i] for i in range(len(saved_paths))
            ):
                for i in range(len(saved_paths)):
                    xs[i] = saved_x[i]
                start_i = len(saved_paths)
                print(f"Resuming LLD extract from {start_i}/{len(items)}")
            else:
                print("Partial cache path mismatch — rebuilding from scratch")
                partial.unlink(missing_ok=True)
        except Exception as e:
            print(f"Corrupt partial cache ({e}); deleting and rebuilding")
            partial.unlink(missing_ok=True)
            xs = [None] * len(items)
            start_i = 0

    for i in range(start_i, len(items)):
        path, _ = items[i]
        xs[i] = extract_lld_sequence(path, max_frames=max_frames)
        if (i + 1) % 50 == 0 or (i + 1) == len(items):
            print(f"  lld features {i+1}/{len(items)}")
        if (i + 1) % save_every == 0 or (i + 1) == len(items):
            done_x = np.stack(xs[: i + 1])
            _atomic_savez_compressed(
                partial,
                x=done_x,
                y=y_enc[: i + 1],
                paths=np.array(paths[: i + 1], dtype=object),
            )

    x = np.stack(xs)
    y = np.asarray(y_enc)
    _atomic_savez_compressed(cache_npz, x=x, y=y)
    if partial.exists():
        partial.unlink(missing_ok=True)
    return x, y, le


def make_model(dataset: str, n_feat: int, n_classes: int) -> EmotionLSTM:
    if dataset == "crema_d":
        return EmotionLSTM(
            n_feat=n_feat,
            n_classes=n_classes,
            bidirectional=False,
            hidden=128,
            layers=3,
            dense=64,
        )
    return EmotionLSTM(
        n_feat=n_feat,
        n_classes=n_classes,
        bidirectional=True,
        hidden=256,
        layers=2,
        dense=128,
    )


def _arch_meta(dataset: str, n_feat: int, n_classes: int, max_frames: int, readout: str) -> dict:
    return {
        "dataset": dataset,
        "n_feat": int(n_feat),
        "n_classes": int(n_classes),
        "max_frames": max_frames,
        "bidirectional": dataset != "crema_d",
        "hidden": 128 if dataset == "crema_d" else 256,
        "layers": 3 if dataset == "crema_d" else 2,
        "dense": 64 if dataset == "crema_d" else 128,
        "readout": readout,
    }


def train_dnn(
    dataset: str,
    epochs: int = 20,
    batch_size: int = 32,
    max_frames: int = 200,
    augment: bool = False,
    resume: bool = True,
):
    print(f"train_dnn patch: {TRAIN_DNN_PATCH}")
    audio_dir = DATA / "audio" / dataset
    out_dir = MODELS / "audio" / dataset / "dnn"
    out_dir.mkdir(parents=True, exist_ok=True)

    cache_npz = out_dir / ("features_aug.npz" if augment else "features.npz")
    le_path = out_dir / "label_encoder.joblib"

    loaded = False
    if cache_npz.exists():
        try:
            print(f"Loading cached LLD arrays from {cache_npz}")
            blob = np.load(cache_npz, allow_pickle=True)
            x, y = blob["x"], blob["y"]
            if le_path.exists():
                le = joblib.load(le_path)
            else:
                le = LabelEncoder()
                le.classes_ = np.array([str(i) for i in range(int(y.max()) + 1)])
                print("WARNING: label_encoder missing; class names may be wrong")
            loaded = True
        except Exception as e:
            print(f"Corrupt {cache_npz}: {e}; deleting and rebuilding")
            cache_npz.unlink(missing_ok=True)

    if not loaded:
        items = list_labeled_wavs(audio_dir, dataset)
        if not items:
            raise FileNotFoundError(
                f"No labeled wavs in {audio_dir} and no feature cache at {cache_npz}"
            )
        print(f"labeled wavs: {len(items)}")
        items = filter_readable(items)
        if not items:
            raise FileNotFoundError(f"No readable wavs in {audio_dir}")

        if augment:
            ratio = 4 if dataset == "crema_d" else 3
            print(f"Augmenting {ratio}:1 (slow first run)...")
            items = maybe_augment_dir(
                items, dataset, ratio=ratio, cache=DATA / "audio" / f"{dataset}_aug"
            )
        else:
            print("Augmentation OFF (use --augment to enable thesis 4:1 / 3:1)")

        print("Extracting LLD sequences (resumable)...")
        x, y, le = build_arrays_resumable(items, cache_npz, max_frames=max_frames)
        joblib.dump(le, le_path)
        print(f"Cached -> {cache_npz}")

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    mean = x_train.mean(axis=(0, 1), keepdims=True)
    std = x_train.std(axis=(0, 1), keepdims=True) + 1e-8
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std
    np.savez(out_dir / "norm.npz", mean=mean, std=std)

    n_classes = len(le.classes_) if hasattr(le, "classes_") else int(y.max()) + 1
    model = make_model(dataset, n_feat=x.shape[-1], n_classes=n_classes)
    device = torch.device(device_str())
    model.to(device)

    best_path = out_dir / "best.pt"
    start_epoch = 1
    best_acc = 0.0
    readout = READOUT_BIDIRECTIONAL if dataset != "crema_d" else READOUT_FORWARD
    arch_path = out_dir / "arch.json"
    checkpoint_ok = True
    if dataset != "crema_d" and arch_path.exists():
        previous = json.loads(arch_path.read_text(encoding="utf-8"))
        checkpoint_ok = previous.get("readout") == readout
    elif dataset != "crema_d":
        checkpoint_ok = False
    if resume and best_path.exists() and checkpoint_ok:
        model.load_state_dict(torch.load(best_path, map_location=device))
        print(f"Loaded weights from {best_path}")
        model.eval()
        with torch.no_grad():
            logits = model(torch.tensor(x_test, dtype=torch.float32, device=device))
            pred = logits.argmax(dim=-1).cpu().numpy()
            best_acc = float(accuracy_score(y_test, pred))
        print(f"Current best_acc from checkpoint: {best_acc:.4f}")
    elif resume and best_path.exists():
        print(
            f"Skipping {best_path}: it was trained with the last-frame readout. "
            "Bidirectional training starts from scratch."
        )

    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    weight_tensor = torch.ones(n_classes, dtype=torch.float32, device=device)
    for c, w in zip(classes, weights):
        weight_tensor[int(c)] = float(w)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss(weight=weight_tensor)
    train_loader = DataLoader(SeqDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(SeqDataset(x_test, y_test), batch_size=batch_size)

    last_f1 = 0.0
    last_preds, last_trues, last_probs = [], [], []
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        total = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(yb)
        model.eval()
        preds, trues, probs = [], [], []
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(device)
                logits = model(xb)
                p = torch.softmax(logits, dim=-1).cpu().numpy()
                pred = p.argmax(axis=1)
                preds.extend(pred.tolist())
                trues.extend(yb.numpy().tolist())
                probs.append(p)
        acc = accuracy_score(trues, preds)
        f1 = f1_score(trues, preds, average="macro")
        last_f1, last_preds, last_trues, last_probs = f1, preds, trues, probs
        print(f"epoch {epoch}: loss={total/len(y_train):.4f} acc={acc:.4f} macro_f1={f1:.4f}")
        if acc >= best_acc:
            best_acc = acc
            torch.save(model.state_dict(), best_path)
            arch_path.write_text(json.dumps(_arch_meta(
                dataset, x.shape[-1], n_classes, max_frames, readout
            ), indent=2))
            print(f"  saved best -> {best_path}")

    probs = np.vstack(last_probs)
    try:
        auc = roc_auc_score(
            np.eye(n_classes)[last_trues], probs, average="weighted", multi_class="ovr"
        )
    except ValueError:
        auc = float("nan")
    metrics = {
        "accuracy": float(best_acc),
        "last_macro_f1": float(last_f1),
        "weighted_roc_auc": float(auc),
        "device": device_str(),
        "augment": augment,
        "epochs": epochs,
    }
    target_names = list(le.classes_) if hasattr(le, "classes_") else None
    print(classification_report(last_trues, last_preds, target_names=target_names, zero_division=0))
    joblib.dump(le, le_path)
    (out_dir / "dnn_metrics.json").write_text(json.dumps(metrics, indent=2))
    meta = _arch_meta(dataset, x.shape[-1], n_classes, max_frames, readout)
    arch_path.write_text(json.dumps(meta, indent=2))
    print("METRICS", metrics)
    return metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["crema_d", "banglaser"], required=True)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--augment", action="store_true")
    p.add_argument("--no-resume", action="store_true")
    args = p.parse_args()
    train_dnn(
        args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        augment=args.augment,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()
