# BlendEmo

Official minimal training release for the paper:

**BlendEmo: Structured Set Prediction and Pair-Conditioned Ratio Modeling for Blended Emotion Recognition**

This repository contains the core cross-validation training code for **BlendEmo**, our structured multimodal method for blended emotion recognition on the **BlEmoRe / BLEMORE** benchmark.

## Overview

Blended emotion recognition is not a standard flat classification problem. A model must:

1. decide whether a clip is **single-emotion** or **mixed-emotion**,
2. recover the correct **emotion set**,
3. and, for mixed clips, estimate the **relative salience ratio**.

BlendEmo addresses this by explicitly factorizing prediction into two coupled stages:

- **structured set prediction**: predict one valid state from the constrained task label space,
- **pair-conditioned ratio prediction**: if the sample is mixed, predict the discrete dominance ratio inside the selected pair.

The final decoder is fixed and task-consistent, so the output always corresponds to a valid BLEMORE label.

## Method Summary

BlendEmo uses four pre-extracted modality streams:

- **HiCMAE**
- **WavLM**
- **OpenFace**
- **VideoMAEv2**

The model has three main stages.

### 1. Modality encoding

Each modality is encoded independently and projected into:

- a **shared representation** for cross-modal interaction,
- a **private representation** for complementary fusion.

In the current implementation:

- HiCMAE and VideoMAEv2 use lightweight encoders,
- WavLM and OpenFace use sequence encoders.

### 2. Cross-modal fusion

The shared branch is fused through:

- audio-guided residual interaction,
- prompt-token fusion,
- relation-graph fusion,
- router-based pooling.

The private branch contributes a complementary multiplicative fusion term.

The final fused representation combines:

- additive summary,
- multiplicative summary,
- routed summary.

### 3. Structured prediction

BlendEmo predicts:

- a **16-way structured set class**
  - 6 single-emotion classes
  - 10 valid blended-pair classes
- a **3-way ratio class** for mixed emotions
  - `70/30`
  - `50/50`
  - `30/70`

The ratio head is **pair-conditioned**: it uses pair embeddings and pair-specific scalar evidence instead of a single global ratio head.

## Why This Design Matters

The main idea of the paper is that **pair identity and salience are related but not identical decisions**.

A flat classifier can assign plausible scores to two emotions but still fail the benchmark because:

- the predicted pair is invalid,
- the wrong pair is selected,
- or the salience ratio is wrong.

BlendEmo improves this by:

- first predicting a valid set state,
- then predicting the ratio only inside the selected pair.

This matches the hierarchical structure of the benchmark.

## Repository Structure

```text
BlendEmo/
├── run_train.sh
└── src/
    ├── train.py
    ├── model.py
    ├── backbone.py
    ├── encoders.py
    ├── data.py
    ├── common.py
    ├── labels.py
    └── metrics.py
```

## Data Layout

By default, `run_train.sh` expects the BLEMORE data directory at:

```text
../ble-datasets
```

Inside that directory, the following files/folders are expected:

```text
ble-datasets/
├── train_metadata.csv
└── pre_extracted_train_data/
    ├── HiCMAE_train/
    ├── WavLM_large_train/
    ├── OpenFace_3_train/
    └── VideoMAEv2_train/
```

The training code reads:

- `train_metadata.csv`
- `pre_extracted_train_data/HiCMAE_train/*.npy`
- `pre_extracted_train_data/WavLM_large_train/*.npy`
- `pre_extracted_train_data/OpenFace_3_train/*.csv`
- `pre_extracted_train_data/VideoMAEv2_train/*.npy`

## Environment

This minimal release does not ship a frozen environment file. At minimum, you need:

- Python 3.9+
- PyTorch
- NumPy
- pandas
- tqdm

A CUDA-enabled PyTorch installation is recommended for actual training.

## Training

### Default cross-validation run

```bash
bash run_train.sh
```

Default outputs:

- `outputs_cv/train.log`
- `outputs_cv/results.csv`
- `outputs_cv/checkpoints/`

### Override data path or output path

```bash
DATA_DIR=/path/to/ble-datasets OUTPUT_DIR=/path/to/output bash run_train.sh
```

### Direct Python entry

```bash
python src/train.py \
  --data_dir ../ble-datasets \
  --output_dir ./outputs_cv
```

## Training Configuration Used in the Repository Script

The provided `run_train.sh` uses:

- hidden dimension: `192`
- dropout: `0.35`
- LMF rank: `10`
- prompt layers: `1`
- batch size: `24`
- learning rate: `3e-4`
- weight decay: `6e-3`
- epochs: `40`
- warmup epochs: `3`
- patience: `12`
- feature-stat frame cap: `80`
- mixup alpha: `0.2`
- set loss weight: `0.45`
- ratio loss weight: `0.25`
- soft emotion loss weight: `0.9`
- mix loss weight: `0.4`
- salience loss weight: `0.35`
- R-Drop weight: `0.8`
- label smoothing: `0.05`
- focal gamma: `2.0`

## Reproducing the Paper Setting

The paper reports the main table under a **common epoch-20 checkpoint** for fair comparison across methods.

This repository's default script is a generic cross-validation training entrypoint. If you want to match the paper's fixed-checkpoint setting more closely, use the training entry directly and specify:

```bash
python src/train.py \
  --data_dir ../ble-datasets \
  --output_dir ./outputs_cv_epoch20 \
  --selection_mode fixed_epoch \
  --fixed_epoch 20
```
