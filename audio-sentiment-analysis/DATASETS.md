# Download audio datasets into Colab (/content), not Drive

Run these **after** the slim project copy. Keep wavs on Colab disk to avoid filling Drive.

## CREMA-D (English) — AudioWAV only

Official source: [CheyneyComputerScience/CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D) (Git LFS).  
Do **not** use “Download ZIP” — that gives empty LFS stubs.

```python
# ~1–2 GB of wavs (full repo with video is ~7.5 GB — we pull AudioWAV only)
!apt-get -qq install -y git-lfs
!git lfs install

import os, shutil
from pathlib import Path
os.chdir("/content")
if Path("CREMA-D").exists():
    shutil.rmtree("CREMA-D")

# Clone without downloading all LFS blobs first
!GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 https://github.com/CheyneyComputerScience/CREMA-D.git
%cd /content/CREMA-D
!git lfs pull --include="AudioWAV/*"

dst = Path("/content/audio-sentiment-analysis/data/audio/crema_d")
dst.mkdir(parents=True, exist_ok=True)
# Prefer AudioWAV; fall back to AudioMP3 if needed
src = Path("/content/CREMA-D/AudioWAV")
assert src.exists() and any(src.glob("*.wav")), "AudioWAV missing — LFS pull failed"
!cp -r /content/CREMA-D/AudioWAV/. {dst}/
print("CREMA-D wavs:", len(list(dst.rglob("*.wav"))))
```

Filenames like `1001_DFA_ANG_XX.wav` — emotion is the 3rd token (`ANG`,`HAP`,`NEU`,…).

## BanglaSER (Bengali)

Official source: [Mendeley BanglaSER v5](https://data.mendeley.com/datasets/t9h6p943xy/5) (~0.6–1 GB).

1. On your PC browser: open that link → **Download All**
2. Upload the zip to Drive (or Colab Files panel), e.g. `MyDrive/BanglaSER.zip`
3. In Colab:

```python
from pathlib import Path
import zipfile, shutil

# adjust path if you put the zip elsewhere
z = Path("/content/drive/MyDrive/BanglaSER.zip")
assert z.exists(), "Upload BanglaSER.zip to MyDrive first"
out = Path("/content/BanglaSER_raw")
if out.exists():
    shutil.rmtree(out)
out.mkdir()
with zipfile.ZipFile(z) as zf:
    zf.extractall(out)

dst = Path("/content/audio-sentiment-analysis/data/audio/banglaser")
if dst.exists():
    shutil.rmtree(dst)
dst.mkdir(parents=True)

# flatten all wavs into banglaser/
n = 0
for wav in out.rglob("*.wav"):
    shutil.copy2(wav, dst / wav.name)
    n += 1
print("BanglaSER wavs:", n)
```

Filenames look like `03-01-02-01-03-03-02.wav` (emotion = 3rd number).

## Then train

```python
import os
os.chdir("/content/audio-sentiment-analysis")
!python -m src.audio.train_classical --dataset crema_d
# LSTM needs GPU:
!python -m src.audio.train_dnn --dataset crema_d --epochs 20 --batch-size 32
```

Optional later: copy only `models_final/audio` back to Drive (not the wavs).
