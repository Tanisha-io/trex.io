# Results scorecard (from your Colab PDFs + earlier classical text logs)

Date captured from notebooks: ~8 Sep 2026.

## 1. Text — English single-label (`dair-ai/emotion` fallback)

| Model | Accuracy | Macro F1 | Notes |
| --- | ---: | ---: | --- |
| Naive Bayes | 0.678 | 0.421 | Weak on love / surprise |
| Logistic Regression | 0.876 | 0.821 | Solid classical |
| **SVM** | **0.897** | **0.861** | Best classical |
| Random Forest | 0.894 | 0.853 | Close to SVM |
| XGBoost | 0.881 | 0.846 | Slightly behind SVM/RF |
| **BERT** (`bert-base-uncased`, 3 ep) | **0.943** | **0.910** | Best text single-label |

Thesis target (different CSV): BERT ~91.8% / F1 0.90. Your BERT **beats** that on the HF emotion set.

**Use in final pipeline (EN text single-label): BERT.**

---

## 2. Text — GoEmotions multi-label

| Model | Metric | Value |
| --- | --- | ---: |
| Dummy (always Neutral) | Neutral rate / support proxy | 0.329 |
| Ridge (TF-IDF) | Macro F1 | 0.274 |
| BERT multi-label | In progress in PDF | Training shown mid-run (~25% steps); finals were copied to Drive (419 MB `models_final/text`) |

Thesis targets: Ridge ~0.25 F1, BERT ROC-AUC ~0.89 / weighted F1 ~0.55.

**Action:** open `models_final/text/**/bert_metrics.json` (or `transformer_metrics.json`) on Drive and confirm eval ROC-AUC / F1. If training finished, **use GoEmotions BERT** for multi-label EN text.

---

## 3. Audio — CREMA-D (English, 7442 wavs)

| Model | Accuracy | Macro F1 | Best classes | Weak classes |
| --- | ---: | ---: | --- | --- |
| Random Forest | 0.525 | 0.512 | anger (F1 0.68), sad (0.61) | fear 0.37, disgust 0.38 |
| **XGBoost** | **0.541** | **0.536** | anger 0.67, sad 0.60, neutral 0.57 | disgust 0.42, fear 0.43 |
| LSTM DNN | Incomplete in PDF | — | LLD extract started; full 20-epoch metrics not in export | Need re-run / longer session |

Thesis classical ~52–54% — **you matched that**. Thesis DNN ~79%+ — **not yet achieved** (run didn’t finish).

**Interim EN audio head: XGBoost.**  
**Target EN audio head: finish LSTM** (`train_dnn` with cached `features.npz`, no need to re-extract).

---

## 4. Audio — BanglaSER (1467 / 1466 labeled wavs)

| Model | Accuracy | Macro F1 | Notes |
| --- | ---: | ---: | --- |
| Random Forest | 0.786 | 0.789 | Strong; natural best |
| **XGBoost** | **0.796** | **0.802** | Best finished Bangla model; natural F1 ~0.97 |
| BiLSTM DNN | ~0.46 after 3 epochs only | 0.42 | Under-trained vs classical; PDF cut off early |

Thesis classical Bangla ~81–82%, DNN higher. Your classical is **close**; DNN needs full 20 epochs (and preferably `--augment`).

**Interim BN audio head: XGBoost.**  
**Target BN audio head: complete BiLSTM**, only replace XGB if it beats ~0.80.

---

## 5. What to use as the **final** multimodal system now

| Stage | Language | Model to ship |
| --- | --- | --- |
| ASR | en/bn | Whisper base |
| Text | English | BERT single-label + GoEmotions BERT (if metrics file OK) |
| Text | Bengali | mBERT / LogReg when CSV available |
| Audio | English | **XGB now** → **LSTM when training finishes** |
| Audio | Bengali | **XGB now** → **BiLSTM if it exceeds XGB** |
| Fusion | both | Confidence-weighted late fusion + `configs/fusion_mapping.yaml` |

Artifacts already on Drive:
- Text: `models_final/text/` (~419 MB)
- Audio: `models_final/audio/` (~189 MB, 28 files)

---

## 6. Improvements that will raise the final model

Priority order:

1. **Finish CREMA-D LSTM** — load cached LLDs, run full 20 epochs; expect the big thesis jump (~+25 pp).
2. **Finish BanglaSER BiLSTM** — don’t stop at epoch 3; compare to XGB 0.80 before swapping.
3. **Confirm GoEmotions BERT** metrics JSON; if weak on rare labels, add class-weighted BCE.
4. **Class imbalance** — CREMA-D fear/disgust; text surprise/love — class weights or focal loss.
5. **Enable `--augment`** for DNN once the unaugmented run works (thesis 4:1 / 3:1).
6. **Fusion tuning** — fit modality weights / mapping \(M\) on a small paired set; track WAS & CED.
7. **Calibration + reject** — temperature scale; reject when \(\max P < 0.5\) and weights disagree.
8. **Bengali text corpus** — still missing; blocks full multilingual text path.

---

## 7. Quick “done vs not done”

| Item | Status |
| --- | --- |
| EN classical text | Done |
| EN BERT single-label | Done (~94% / 0.91 F1) |
| GoEmotions BERT | Likely done (weights saved); metrics to confirm |
| CREMA-D RF/XGB | Done (~52–54%) |
| CREMA-D LSTM | Incomplete → use `notebooks/03_finish_gaps_colab.ipynb` |
| BanglaSER RF/XGB | Done (~79–80%) |
| BanglaSER BiLSTM | Incomplete (~46% @ 3 ep) → same notebook |
| Fusion API | Scaffold in `src/fusion/` (WAS/CED demo) |

### Resume command (Colab, after uploading updated `src/`)

```bash
python -m src.text.print_metrics
python -m src.audio.train_dnn --dataset crema_d --epochs 20 --batch-size 32
python -m src.audio.train_dnn --dataset banglaser --epochs 20 --batch-size 16
python -m src.fusion.demo
```

DNN now: resumable LLD cache, class-weighted loss, loads `best.pt` if present.
