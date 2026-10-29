# Training setup (Google Colab + Drive)

## Workflow (no Colab file-upload widget)

1. On the PC, use this project folder.
2. In browser Drive: **New → Folder upload** that folder to **My Drive** as `audio-sentiment-analysis`.
3. Colab: open `notebooks/01_train_text_colab.ipynb` from Drive, enable **T4 GPU**.
4. Notebook:
   - copies code Drive → `/content`
   - trains on `/content` (HF cache off Drive)
   - copies **only final models** → `MyDrive/audio-sentiment-analysis/models_final/`

Do **not** save Trainer `runs/` or HF caches onto Drive.

## Commands (inside notebook /content)

```bash
python -m src.text.train_single_label --language en --mode transformer --epochs 3 --batch-size 16
python -m src.text.train_multilabel --mode both --epochs 4 --batch-size 16
```

## Final artifacts location

`My Drive/audio-sentiment-analysis/models_final/text/`  
`My Drive/audio-sentiment-analysis/models_final/audio/`
