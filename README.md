# Multilingual Audio Sentiment Analysis

IIT Kharagpur MTP: **Computational Framework for Multilingual Audio Sentiment Analysis** (Tanisha Panda, 20EE38017, April 2025).

English and Bengali emotion from speech and from the words, scored separately and then combined. The page is at `http://127.0.0.1:8765`.

## Run the interface

Python 3.11, from this folder:

```bash
pip install -r requirements.txt
python -m uvicorn src.api.app:app --host 127.0.0.1 --port 8765
```

Choose any file that contains sound. The page hears the language, writes the transcript, and scores English and Bengali. A saved library clip can be opened without a file.

`models_final/` holds the voice and text weights. Those files are not in git: the largest checkpoint is hundreds of megabytes. Put that folder next to `src/` before scoring. `data/`, `models/`, and `uploads/` stay local for the same reason.

## What the project does

A dual-track system:

1. **Language-aware ASR** (Whisper) turns audio/video into text + language id (`en` / `bn`).
2. **Text branch** classifies emotions with BERT (English) and mBERT (Bengali), single-label and multi-label.
3. **Audio branch** extracts OpenSMILE acoustics, drops collinear features (VIF > 10), then classifies with RF / XGBoost / LSTM or BiLSTM.
4. **Fusion** maps text emotions onto audio emotion space and mixes the two distributions with confidence weights.
5. **API** returns transcription, fused emotions, and modality weights.

```
audio/video
    ├─ Whisper ASR ──► language + transcript ──► BERT / mBERT
    └─ waveform ──► OpenSMILE ──► VIF ──► LSTM / BiLSTM
                           └──────────────────────► confidence-weighted fusion ──► JSON
```

---

## Datasets (place under `data/`)

| Track | Language | Source | Notes |
| --- | --- | --- | --- |
| Single-label text | English | `text_emotions.csv` | Classical + BERT comparison |
| Single-label text | Bengali | Custom CSV (~10k) | Joy / anger / sadness imbalance; Banglish present |
| Multi-label text | English | [GoEmotions](https://github.com/google-research/google-research/tree/master/goemotions) | 58,009 Reddit comments, 27 emotions + Neutral |
| Multi-label text | Bengali | Derived from the 10k single-label set | Map via Plutchik wheel (planned in thesis; optional v2) |
| Audio | English | [CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D) | ~7,442 clips, 6 emotions, 1–2 s |
| Audio | Bengali | [BanglaSER](https://github.com/sazzadcsedu/BN-Speech-Corpus) | ~1,080 clips, 5 emotions, 3–5 s |

**Audio labels**

- CREMA-D: Anger, Disgust, Fear, Happy, Neutral, Sad
- BanglaSER: Angry, Happy, Natural, Sad, Surprised

**Text single-label (Bengali mix, thesis):** Joy 17.8%, Anger 15.2%, Sadness 16.3%, remainder other classes.

---

## Stage 1 — Single-label text

### English preprocessing (`text_emotions.csv`)

Lowercase → emoji dictionary substitution → strip punctuation/digits → regex tokens → NLTK + custom stopwords → WordNet lemmatizer → `CountVectorizer` + `TfidfTransformer` for classical models.

BERT path: keep more context; use `BertTokenizer`, max length 128. Do not lemmatize/stopword for transformers.

### Bengali preprocessing (custom CSV)

Lowercase → drop non-Bengali (Latin) characters → strip punctuation and Bengali/Arabic digits → regex tokens for Bengali script → Bengali stopword list → **no lemmatization** → TF-IDF with n-grams `(1, 2)` for classical models.

mBERT path: `BertTokenizer` from `bert-base-multilingual-cased`, max length 128.

### Models to train

**English:** Naive Bayes, Logistic Regression, SVM, Random Forest, XGBoost, BERT (`bert-base-uncased`).  
**Bengali:** Logistic Regression, mBERT (plus optional SVM / RF).

### Thesis targets (sanity check, not exact reproduction guaranteed)

| Model | Acc | F1 |
| --- | --- | --- |
| English BERT | 91.78% | 89.91 |
| English XGBoost | 88.83% | — |
| Bengali Logistic Regression | 43.70% | ~0.433 |
| Bengali mBERT | 43.35% | ~0.433 |

Bengali is expected to stay weak until the original custom corpus is recovered.

---

## Stage 2 — Multi-label text (GoEmotions)

### Preprocessing (BERT)

Lowercase → strip emojis, slang-heavy noise, HTML, punctuation → `BertTokenizer` pad/truncate to **128** → labels as multi-hot tensors (28 classes). **No** stopword removal or lemmatization.

### Baselines

1. Dummy: always Neutral.
2. Ridge via `MultiOutputClassifier` on TF-IDF (1,000 features) after spaCy tokenize + lemmatize + stopwords.

### BERT

- Head: `[CLS]` → linear → 28 logits
- Loss: `BCEWithLogitsLoss`
- Optimizer: AdamW, **lr = 5e-5**
- Epochs: **4**

### Thesis targets

- Dummy Neutral accuracy ~52.3%
- Ridge macro F1 ~0.25
- BERT weighted ROC-AUC **0.89**, weighted F1 **0.55**, macro F1 **0.46**, micro F1 **0.57**
- Strong: gratitude, love, amusement
- Weak / zero: grief, relief, pride, nervousness

Optional later: class-weighted BCE or oversampling for rare labels.

---

## Stage 3 — Audio emotion

### 3a. Augmentation (label-preserving, staged)

**Waveform first** (before STFT):

| Transform | Spec |
| --- | --- |
| Temporal shift | Cyclic shift \(\delta \sim U(-0.2T, 0.2T)\), zero-pad to keep length. CREMA-D ~200 ms; BanglaSER can use larger windows (clips are 3–5 s). |
| Noise | \(x' = x + \alpha \mathcal{N}(0, \sigma_x^2)\), \(\alpha \in [0.005, 0.015]\) (0.5–1.5% of amplitude) |

**Then spectrogram / phase vocoder:**

| Transform | Spec |
| --- | --- |
| Time stretch | \(\beta \in [0.9, 1.1]\) |
| Pitch shift | \(n \in \{-3,-1.5,0,1.5,3\}\) semitones; BanglaSER ±3, CREMA-D can use ±2 |

**Synthetic:real ratios:** CREMA-D **4:1**, BanglaSER **3:1**.

Order matters: time-domain ops before spectral ops so STFT relations stay consistent.

### 3b. Features (OpenSMILE)

Extract **53 LLDs** aggregated to **~88–90 functionals** in four groups:

- Pitch (~20): `F0semitoneFrom27.5Hz_*`
- Loudness (~11): `loudness_sma3_*`
- MFCC (~16): `mfcc1–4` means / voiced stats
- Voice quality (~6): jitter, shimmer, HNR

Use the **eGeMAPS** (or ComParE if matching 90 dims) config so names match the thesis (`loudness_sma3_amean`, `spectralFlux`, etc.).

### 3c. VIF filter (threshold 10)

Iteratively drop features with VIF > 10.

| Dataset | Start | After VIF |
| --- | --- | --- |
| CREMA-D | 90 feats, 7,442 samples | **79** feats |
| BanglaSER | — | **87** feats (only a few pitch percentiles/range dropped) |

CREMA-D drops typically include upper/lower pitch percentiles, pitch range, spectral flux variability, unvoiced HNR, equivalent sound level, some MFCC means, H1-A3.

Persist the kept feature name lists; inference must use the same columns.

### 3d. Feature importance (for analysis, not required at serve time)

Random Forest Gini importance.

- **Bengali:** loudness **dynamics** (rising/falling slopes, slope std), spectral flux, low-order MFCCs.
- **English:** **absolute** loudness (mean, p80, median), loudness range, MFCC variability.

Train **separate** audio models per language. Do not share one universal acoustic model.

### 3e. Classifiers

**Random Forest:** 100 trees, `random_state=42`, VIF features.

**XGBoost:** 100 estimators, `learning_rate=0.1`, `random_state=42`.

**DNN (production audio models):**

| | CREMA-D (English) | BanglaSER (Bengali) |
| --- | --- | --- |
| Sequential model | 3 × LSTM(128) | 2 × BiLSTM(256) |
| Dense | 64 | 128 |
| Features | 79 | 87 |
| Optimizer | Adam | Adam |
| Input | Sequence of LLD frames **or** repeated/stat functionals reshaped as time; match training notebooks |

If OpenSMILE functionals are utterance-level (1 vector), either:

- extract **frame-level LLDs** and pad sequences for LSTM, or
- treat the functional vector as a single timestep (weaker; not what the architecture implies).

Prefer **frame-level LLD sequences** + masking.

### Thesis audio accuracy (approx.)

| Model | CREMA-D | BanglaSER |
| --- | --- | --- |
| Random Forest | 52% | 81% |
| XGBoost | 54% | 82% |
| DNN | ~79–83% (thesis: +27–29 pp vs classical on English) | ~84–91% (thesis: +2–10 pp); ROC-AUC ~0.93 |

English DNN ROC-AUC ~0.84. Anger is easiest in English; Fear/Disgust hardest. BanglaSER Natural/Angry/Sad are strong; Happy/Surprised overlap.

---

## Stage 4 — Multimodal fusion + API

### Runtime flow

1. `POST /analyze` with audio or video.
2. Save sanitized filename under `uploads/`.
3. Whisper **base**: transcript + language (`en` / `bn`). Code-mix is a known failure mode.
4. Text model for that language → \(P_{\text{text}}\) over text labels.
5. Audio model for that language → \(P_{\text{audio}}\) over audio labels.
6. Map text distribution into audio space with matrix \(M\).
7. Confidence-weighted mix:

\[
P_{\text{final}}(e_i) = w_1 P_{\text{audio}}(e_i) + w_2 \sum_j M_{ij} P_{\text{text}}(e_j)
\]

\[
w_1 = \frac{c_{\text{audio}}}{c_{\text{audio}} + c_{\text{text}}},\quad w_2 = 1 - w_1
\]

Confidence = max softmax (audio) / max sigmoid or calibrated probability (text).

**Example mapping (thesis):** textual *disgust* → weighted *angry* + *natural* in audio space. Build \(M\) as a sparse expert matrix (text label → audio labels), then row-normalize.

### Response contract

```json
{
  "transcription": "I can’t believe this is happening...",
  "emotions": [
    {"label": "sadness", "probability": 0.76},
    {"label": "neutral", "probability": 0.34}
  ],
  "modality_weights": {
    "audio": 0.62,
    "text": 0.38
  }
}
```

### Offline fusion metrics

- **WAS:** agreement of top labels plus cosine of mapped distributions, weighted by \(w_1, w_2\).
- **CED:** cross-entropy between audio and (mapped) text distributions, \(\epsilon = 10^{-5}\).

Thesis latency note: ~890 ms / 10 s clip on RTX 3090 (float32); int8 quant ~210 ms. Optional v2.

Optional reject: \(\max(P_{\text{final}}) < 0.5\) and \(|w_1 - w_2| > 0.3\).

---

## Repo layout

```
audio-sentiment-analysis/
  src/
    api/                # the page: app.py, voice scores, transcript, text score
    audio/              # OpenSMILE features and training
    text/               # text preprocessing and training
    fusion/             # combine voice and text scores
    viz/                # thesis figures
  configs/
  notebooks/
  figures/              # thesis figures
  data/                 # local only, not in git
  models/               # local only, not in git
  models_final/         # local weights, not in git
  uploads/              # scored files, not in git
  requirements.txt
```

---

## Implementation order

1. Data ingest + label YAML (English/Bengali text + audio).
2. Text single-label classical + BERT/mBERT.
3. GoEmotions multi-label Ridge + BERT.
4. Audio: augment → OpenSMILE → VIF → RF/XGB → LSTM/BiLSTM.
5. Cross-modal \(M\), fusion, WAS/CED.
6. FastAPI + Whisper end-to-end.
7. Optional: Plutchik Bengali multi-label, code-mix ASR, int8 quant.

---

## Stack

Python 3.10+, PyTorch, Hugging Face Transformers, scikit-learn, XGBoost, librosa / audiomentations, OpenSMILE (`opensmile` Python package), Whisper, FastAPI, uvicorn, pandas, statsmodels (VIF), spaCy + NLTK.

---

## Training (Google Colab)

**Colab is recommended if you have no local GPU.** You do not need Colab if you already have CUDA.

See **[TRAINING.md](TRAINING.md)** and:

1. `notebooks/01_train_text_colab.ipynb` — text models first  
2. `notebooks/02_train_audio_colab.ipynb` — audio models second  

Upload this folder to Drive as `MyDrive/audio-sentiment-analysis`, enable a GPU runtime, then run those notebooks in order.

---

## Blockers until original assets return

- Custom **Bengali text CSV** is not a public standard set; Bengali text numbers will not match without it.
- Exact OpenSMILE config (eGeMAPS vs custom 90-d set) must be matched for VIF drop lists.
- Original fusion matrix \(M\) was not printed in full; we reconstruct from emotion taxonomy + the disgust→angry/natural example.
