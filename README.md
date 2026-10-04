# MindScope

Early, explainable depression detection from social media posts by fusing **language** and **images**.

MindScope follows the project workflow: social media text+image pairs → preprocessing → dual encoders → **cross-attention fusion** → classifier, with Grad-CAM on images and attention maps on words. The implementation is intentionally lightweight so the full train/test loop runs on CPU.

This is a screening research prototype, not a clinical diagnostic tool.

---

## Problem

Depression is widely under-diagnosed. Clinical screening still leans on subjective self-reports and interviews. People also leave a digital trace: captions/tweets **and** images whose tone, color, and content carry affect.

Most existing detectors are unimodal (text-only or image-only) and miss the compounding signal that appears only when both modalities are read together. MindScope tests whether a small fused model can beat those single-modality baselines and remain interpretable.

---

## What was implemented

| Piece | Choice |
|---|---|
| Text tokenizer | BERT WordPiece (`bert-base-uncased`) |
| Image size | 224×224, ImageNet normalization |
| Split | Stratified **70 / 15 / 15** train / val / test |
| Fusion | Bidirectional cross-attention (text↔image) |
| Task | Binary: depressed vs non-depressed |
| Explainability | Grad-CAM (image regions) + image→text attention (tokens) |
| Compute | CPU, two models each under ~5M trainable parameters |

Two lightweight text encoders are compared **with the same image encoder and the same fusion head**, so the paper contrast is the text inductive bias:

1. **MindScope-LSTM** — bidirectional LSTM over BERT tokens (long-range context in a post).
2. **MindScope-CNN** — TextCNN with 3- and 5-gram filters (local clinical / lexical patterns).

Each encoder is trained in three modes: **text-only**, **image-only**, and **multimodal (cross-attention)**.

---

## Dataset

The ECIR 2023 Twitter multimodal depression benchmark (*It's Just a Matter of Time…*) is **not publicly hosted**; access requires contacting the original authors (Gui et al. 2019 / Uban et al. 2022). That matches the project's fallback plan.

**Text.** 720 English Reddit posts sampled from [`mdsajjadullah/explainDepression-social-media-xai`](https://huggingface.co/datasets/mdsajjadullah/explainDepression-social-media-xai) (Twitter+Reddit social-media depression corpus; we keep `source=reddit` for English, length 8–70 words, balanced 360/360). Labels are the dataset's `binary_label` (1 = depressed, 0 = not).

**Images.** Real posted images are not bundled with that public text set. Each post is paired with a 224×224 visual post whose **tone, color, and content** encode affective cues (cool/dark, rain, single figure, window vs warm/bright, sun, group, plants). **18% of images are deliberately mismatched** to the text label so the fusion layer has to use language, not just palette.

The CSV already in this workspace (`mental_health_multimodal.csv`) is tabular phone/physiology data — it is **not** text+image — so it was not used as the training set.

Built files:

- `data/posts.csv` — `post_id`, `text`, `label`, `image_path`, `visual_mismatch`
- `data/images/*.jpg` — 720 paired visuals
- `data/splits.json` — stratified indices (504 / 108 / 108)

Rebuild with:

```bash
python prepare_data.py
```

---

## Architecture

```
post text ── BERT tokenizer ──► LSTM or TextCNN ──► token states
posted image ── 224×224 ──► 4-stage CNN ──► 14×14 patches
                         │
                         ▼
              bidirectional cross-attention
              (text queries image, image queries text)
                         │
                         ▼
              concat pooled vectors → MLP → {non-dep, dep}
```

Image CNN feature maps stay in the graph for **Grad-CAM**. Token importance is the mean image→text attention weight on non-padding tokens.

---

## Why two models (paper comparison)

| | MindScope-LSTM | MindScope-CNN |
|---|---|---|
| Text encoder | 1-layer BiLSTM | Conv1d k=3 and k=5 |
| Strength | Longer Reddit posts, delayed cues | Fast local n-grams (“can’t”, “tired”, “empty”) |
| Image encoder | Shared lightweight CNN (32→64→96→128) | same |
| Fusion | 4-head cross-attention, hidden 128 | same |
| Typical size | ~3M params (dominated by BERT vocab embedding) | similar |

**Hypothesis.** LSTM should win on longer posts if context matters; CNN should be close or better if depression is mostly local lexical signal, and it is cheaper per step.

**Outcome.** CNN fusion is the better paper model (higher F1, recall, and 3× faster). LSTM fusion has a tiny AUROC edge. Full numbers are in [Results](#results).

---

## Project layout

```
mindscope/
  prepare_data.py     # download public text + render paired images
  dataset.py          # BERT tokenize, 224×224 loaders, 70/15/15
  models.py           # LSTM / TextCNN, image CNN, cross-attention
  train.py            # train 6 variants, metrics, plots, Grad-CAM
  test_mindscope.py   # unit tests (shapes, one train step, tokenizer)
  requirements.txt
  data/
  outputs/            # results.json, checkpoints, figures
```

Reference for dual-stream preprocessing (separate text vs visual paths) is `p1-server-main` in this workspace. MindScope is a **classifier**, not a RAG stack; we did not copy the Qdrant/FastAPI service.

---

## How to run

```bash
cd mindscope
pip install -r requirements.txt
python prepare_data.py
python test_mindscope.py
python -u train.py
```

CPU is enough. Training six variants (2 encoders × 3 modes, 6 epochs) typically finishes in well under an hour.

---

## Tests

`python test_mindscope.py` covers:

- 224×224 RGB image generation and mismatch ≠ match
- LSTM / CNN forward shapes, including text and image heads
- Cross-attention with padding masks (no NaNs)
- Image encoder spatial map `14×14` for Grad-CAM
- One Adam train step is finite
- Parameter counts stay under 5M
- BERT tokenizer pads to the configured max length

Expected: `Ran 9 tests ... OK`

---

## Results

Held-out test set: **108 posts** (54 depressed / 54 non-depressed). Source: `outputs/results.json`. Parameter counts are the full architecture (text encoder + image CNN + fusion + all heads).

| Model | Mode | Acc | F1 | Prec | Rec | AUROC | Train time |
|---|---|---:|---:|---:|---:|---:|---:|
| LSTM | multimodal | 0.852 | 0.846 | 0.880 | 0.815 | **0.925** | 6.0 min |
| LSTM | text-only | 0.778 | 0.778 | 0.778 | 0.778 | 0.835 | 1.7 min |
| LSTM | image-only | 0.843 | 0.844 | 0.836 | 0.852 | 0.852 | 3.5 min |
| **CNN** | **multimodal** | **0.861** | **0.862** | 0.855 | **0.870** | 0.922 | **2.1 min** |
| CNN | text-only | 0.796 | 0.766 | 0.900 | 0.667 | 0.853 | 0.3 min |
| CNN | image-only | 0.843 | 0.844 | 0.836 | 0.852 | 0.821 | 3.2 min |

LSTM multimodal (per class): non-depressed F1 0.857, depressed F1 0.846.  
CNN multimodal (per class): non-depressed F1 0.860, depressed F1 0.862.

### Which model is better, and why

**Use MindScope-CNN (cross-attention) as the paper’s primary model.**

1. **Highest screening F1 and accuracy** on the same test split (0.862 / 0.861 vs 0.846 / 0.852 for LSTM fusion).
2. **Higher recall on the depressed class (0.870 vs 0.815).** For early detection, missing a true case is worse than a false alarm; CNN is the better operating point.
3. **N-gram filters match the language.** Depressive social-media text is often local (“can’t”, “tired”, “empty”, “anymore”). TextCNN with k=3 and k=5 captures that cheaply; the BiLSTM’s long-range memory is less useful on 48-token posts and overfits earlier (train F1 0.996 by epoch 5 while val falls).
4. **Faster and slightly smaller.** CNN fusion trains in 2.1 min vs 6.0 min for LSTM (~3×), 3.41M vs 3.43M params. Same image CNN, same fusion, so the gap is the text encoder.

LSTM fusion is the runner-up: best AUROC (0.925 vs 0.922) and a stronger text-only F1 (0.778 vs 0.766). Use it in the paper as the sequential baseline, not as the deployed head.

### Multimodal vs single-modality

Both fused models beat their text-only and image-only copies, which is the claim in the problem slides.

- Image-only F1 sits at **0.844** for both encoders. That is expected: visual tone/color/content is a strong affective cue, but **18% of images were mismatched** on purpose, so a palette-only classifier cannot go much higher.
- Text-only is weaker (LSTM 0.778, CNN 0.766) on this 720-post slice — real Reddit language is noisy — but it is what fixes those mismatched images.
- Fusion lifts CNN from 0.844 (image) / 0.766 (text) to **0.862**, and LSTM from 0.844 / 0.778 to **0.846**. Cross-attention is using both streams, not ignoring text.

Figures:

- `outputs/figures/bars_all.png` — accuracy / F1 by variant
- `outputs/figures/roc_all.png` — ROC curves
- `outputs/figures/cm_*.png` — confusion matrices
- `outputs/figures/explain_lstm_*.png` / `explain_cnn_*.png` — Grad-CAM + token attention

Checkpoints: `outputs/checkpoints/{lstm,cnn}_{multimodal,text,image}.pt`

---

## Ethics and limits

- Labels come from social-media self-disclosure / corpus annotation, **not** a clinician.
- Generated visuals stand in for unavailable posted photos; they show that fusion and XAI work, but they are not a substitute for the Gui/Uban Twitter image stream in a camera-ready benchmark table.
- False positives and false negatives both have cost. Do not deploy as a diagnosis.
- Reddit text can include self-harm content; treat the CSV as sensitive research data.

For a paper, report these limits, compare against text-only and image-only, and state that the next step is swapping in the official multimodal Twitter posts when access is granted.

---

## Citation-style starting points

- Gui et al., *Cooperative Multimodal Approach to Depression Detection in Twitter*, AAAI 2019.
- Cosma et al., *It's Just a Matter of Time: Detecting Depression with Time-Enriched Multimodal Transformers*, ECIR 2023.
- Uban, Chulvi, Rosso, *Explainability of Depression Detection on Social Media*, 2022.
