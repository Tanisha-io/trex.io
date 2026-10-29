# Training

Colab with a T4 GPU. Upload this folder to Drive as `audio-sentiment-analysis`, then run:

1. `notebooks/01_train_text_colab.ipynb`
2. `notebooks/02_train_audio_colab.ipynb`

```bash
python -m src.text.train_single_label --language en --mode transformer --epochs 3 --batch-size 16
python -m src.text.train_multilabel --mode both --epochs 4 --batch-size 16
```

Finished models go to `models_final/text` and `models_final/audio`.
