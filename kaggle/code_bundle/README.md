# Scale_Error_DC — YOLO26m Accuracy-First End-to-End

A single-configuration, Python-only object-detection project for training **YOLO26m** on
**Scale_Error_DC v26** using a **Kaggle Tesla T4** through the **Kaggle CLI**.

There is deliberately **one training config only**:

```text
configs/train.yaml
```

No `YOLO26s`, no 640-vs-768 comparison, no hyperparameter sweep. The repository spends the GPU
budget on one stronger run intended to maximize a defensible final result.

---

## Final model decision

```text
Model       : yolo26m.pt
Input       : 960 × 960
Epochs max  : 120
Patience    : 15
Batch       : 8 (fixed; AutoBatch and batch=16 both OOM'd on Kaggle T4)
Optimizer   : MuSGD
AMP         : enabled
Seed        : 42
Main metric : mAP50-95 on held-out test
```

Why this choice:

- YOLO26m is the accuracy-oriented middle model in the YOLO26 family.
- 960 px preserves more detail than 640 for thin/small industrial defects.
- `yolo26m.pt` is pretrained, so the run is transfer learning rather than training from scratch.
- optimizer/loss values are informed by the published YOLO26-M training recipe.
- augmentation is moderated for industrial imagery instead of copying the very aggressive COCO
  augmentation recipe unchanged.
- early stopping protects against wasting epochs after validation stops improving.

This configuration is designed to *favor accuracy*, but no code can honestly guarantee a target mAP
before the dataset is actually trained and evaluated.

---

## Dataset

Roboflow Universe:

```text
workspace : datacpn
project   : scale_error_dc-orqpx
version   : 26
task      : object detection
images    : 7,346
classes   : 5
```

Classes:

```text
DiVat
DiVatLoiLom
LoiChi
LoiNhua
LoiTray
```

The public v26 release has no test split. The pipeline therefore:

```text
original train (5578) --------------------> final train
original valid (1768)
        |
        +-- multilabel stratified --------> final val  (~884)
        |
        +-- multilabel stratified --------> final test (~884)
```

The held-out test set is not used for training or early stopping.

Class balance in the Roboflow train split is heavily skewed (LoiChi=4629 images
vs LoiTray=223, a ~21:1 ratio), and 59% of all boxes cover under 0.25% of the
image area. Every model trained on this dataset under-performs on the rarest
classes as missed detections, not misclassifications (confirmed via confusion
matrices — see `docs/FULL_DOCUMENTATION.md`). Hardlink-based rare-class
oversampling was tried and measured worse across every class (mAP50-95 0.385
&rarr; 0.356) — duplicate images add no new information and appear to push
the model toward overfitting the repeated samples rather than generalizing,
so it was reverted. A future attempt at improving rare-class recall should
use loss-level class weighting or true copy-paste with segmentation masks
instead of image duplication.

---

## End-to-end pipeline

```text
Roboflow Scale_Error_DC v26
        ↓
YOLO26 export download
        ↓
annotation validation
        ↓
dataset statistics
        ↓
SHA1 duplicate/leakage audit
        ↓
multilabel-stratified val/test split
        ↓
YOLO26m pretrained checkpoint
        ↓
accuracy-first fine-tuning @ 960
        ↓
best.pt
        ↓
held-out test evaluation
        ↓
error analysis (FP/FN/wrong class)
        ↓
latency benchmark
        ↓
sample predictions
        ↓
ONNX export
        ↓
scale_error_yolo26_results.zip
```

---

## Project tree

```text
scale-error-yolo26/
├── configs/
│   └── train.yaml
├── src/
│   └── scale_error/
│       ├── data/
│       ├── training/
│       ├── evaluation/
│       ├── inference/
│       ├── cli.py
│       └── config.py
├── demo/
│   └── app.py
├── tests/
├── docs/
│   └── FULL_DOCUMENTATION.md
├── kaggle_output/            (YOLO26m results)
├── kaggle_output_yolo11/     (YOLO11m results)
├── kaggle_output_rtdetr_final/ (RT-DETR-L results)
├── pyproject.toml
└── README.md
```

Training itself ran as standalone Kaggle Notebook scripts (not checked into
this repo); the directories above hold each run's downloaded `artifacts/`
(manifest, metrics, weights, plots) for inspection.

---

## Important output files

```text
artifacts/
├── dataset/
│   ├── dataset_report.json
│   ├── generated_split_report.json
│   ├── leakage_report.json
│   └── runtime_data.yaml
├── runs/
│   └── yolo26m_960/
│       └── weights/
│           ├── best.pt
│           └── last.pt
├── evaluation/
│   ├── metrics_default_nms.json
│   ├── metrics_nms_free.json
│   ├── error_analysis.csv
│   └── error_analysis_summary.json
├── benchmark/
├── predictions/
├── exports/
└── manifest.json
```

For the coursework report, use `best.pt` and the held-out `test` results, not `last.pt`.

---

## Metrics to report

At minimum:

```text
Precision
Recall
mAP@50
mAP@50:95
per-class AP
Confusion Matrix
False Positives
False Negatives
wrong-class matches
mean / median / p95 latency
FPS
```

Also show several failure examples. A strong report explains *why* the model fails, not only its mAP.

---

## Local training

Install:

```bash
python -m pip install -e ".[dev]"
```

Prepare:

```bash
scale-error prepare \
  --output data/raw/scale_error_dc \
  --workspace datacpn \
  --project scale_error_dc-orqpx \
  --version 26 \
  --split-output data/processed
```

Train the single configuration:

```bash
scale-error train \
  --data data/processed/runtime_data.yaml \
  --config configs/train.yaml \
  --project-dir artifacts/runs
```

---

## One rule for this repository

Do not add extra training configs just to make the repository look larger.

The experimental claim is:

> Fine-tune one accuracy-oriented YOLO26m configuration on Scale_Error_DC, evaluate it correctly,
> analyze its failure modes, and produce a reproducible deployment artifact.

That is enough for a clean deep-learning project.
