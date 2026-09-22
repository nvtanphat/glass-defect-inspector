# FULL DOCUMENTATION
## Industrial Defect Detection on Scale_Error_DC with YOLO26m

### 1. Project objective

The project builds an end-to-end deep-learning pipeline for detecting five industrial defect
categories in the Scale_Error_DC dataset. The engineering goal is not simply to call
`model.train()`. The pipeline must make the experiment reproducible, prevent accidental test
leakage, validate the labels before expensive GPU training, preserve all important artifacts, and
produce enough evidence to defend the model in a coursework presentation.

The project uses exactly one model/training configuration:

```text
YOLO26m pretrained checkpoint
imgsz = 960
epochs <= 120
NVIDIA GPU (Tesla T4 or equivalent)
```

### 2. Why only one configuration

A single configuration is useful when GPU budget is limited and the goal is to deliver one strong
result rather than a large ablation table. It also prevents the common coursework mistake of
running many arbitrary variants without understanding any of them.

One configuration does **not** mean no scientific discipline. Dataset validation, a held-out test,
error analysis, latency measurement, and reproducibility logging are still mandatory.

### 3. Dataset

Public source:

```text
Roboflow Universe
workspace = datacpn
project   = scale_error_dc-orqpx
version   = 26
```

Task:

```text
Object Detection
```

Classes:

```text
0 DiVat
1 DiVatLoiLom
2 LoiChi
3 LoiNhua
4 LoiTray
```

The project page reports approximately 7.35k images. Version 26 is the fixed dataset revision used by
the code.

### 4. Annotation representation

YOLO detection annotations use one text row per object:

```text
class_id x_center y_center width height
```

The four geometry values are normalized to `[0, 1]`.

Example:

```text
2 0.512 0.438 0.074 0.031
```

means:

- class = `LoiChi`
- box center is at 51.2% image width and 43.8% image height
- box width is 7.4% of image width
- box height is 3.1% of image height

The validation code rejects malformed rows, out-of-range class IDs, invalid normalized coordinates,
and non-positive box dimensions.

### 5. Dataset split policy

The public v26 release does not provide a test split. Using the same validation data for both early
stopping and final reporting would overestimate generalization.

Therefore:

```text
Roboflow train  -> train
Roboflow valid  -> stratified 50% val + 50% test
```

The split uses multilabel stratification at **image level**. If an image contains multiple defect
classes, its class-presence vector participates in stratification.

Seed:

```text
42
```

This makes the split reproducible.

### 6. Leakage audit

Before training, the code computes SHA1 hashes of image bytes across train/val/test and records
cross-split identical images.

Why this matters:

```text
same image in train and test
        ↓
model has effectively seen test content
        ↓
test mAP becomes overly optimistic
```

The pipeline records detected duplicates in:

```text
artifacts/dataset/leakage_report.json
```

A final academic report should disclose any leakage found rather than hiding it.

### 7. Dataset statistics

The preparation stage records:

- number of images per split
- number of objects
- missing label files
- empty label files
- object count per class
- small-box ratio
- tiny-box ratio

Definitions used by the diagnostic report:

```text
small box: normalized area < 1%
tiny box : normalized area < 0.25%
```

These are diagnostic thresholds, not COCO's official pixel-area definitions.

### 8. Why YOLO26m

YOLO26 provides multiple model scales. The `m` variant is selected because this repository
prioritizes detection quality over minimum latency while still targeting a single 16 GB-class T4.

Conceptually:

```text
image
  ↓
backbone
  ↓
multi-scale feature representation
  ↓
neck / feature fusion
  ↓
dual detection heads
  ├── one-to-many → normal accuracy path → NMS
  └── one-to-one  → end-to-end path       → no NMS required
```

YOLO26 also includes training changes designed to improve the accuracy/latency tradeoff and small
target assignment.

### 9. Why pretrained weights

The run starts from:

```text
yolo26m.pt
```

and does **not** initialize random weights.

Transfer learning is preferred because Scale_Error_DC has thousands rather than millions of images.
The pretrained checkpoint already contains generic visual representations learned from large-scale
detection pretraining/fine-tuning.

### 10. Why input size 960

A small or thin defect may occupy very few pixels. Resizing an image too aggressively can erase
useful texture/edge information.

Main run:

```text
imgsz = 960
```

This is higher than YOLO's common 640 baseline while still being practical for a T4 with AutoBatch.

Tradeoff:

```text
higher resolution
    + more defect detail
    + potentially better small-object recall
    - more VRAM
    - slower training
    - slower inference
```

The project accepts that compute cost because accuracy is the priority.

### 11. Final configuration

Canonical file:

```text
configs/train.yaml
```

Configuration:

```yaml
model: yolo26m.pt
epochs: 120
imgsz: 960
batch: 0.70
device: 0
workers: 4

seed: 42
deterministic: true
patience: 30

amp: true
cache: false

optimizer: MuSGD
lr0: 0.00038
lrf: 0.882
momentum: 0.948
weight_decay: 0.00027
warmup_epochs: 0.99
box: 9.83
cls: 0.65
dfl: 0.96

mosaic: 0.80
close_mosaic: 10
mixup: 0.05
copy_paste: 0.0
scale: 0.50
translate: 0.10
fliplr: 0.50
flipud: 0.0
degrees: 0.0
shear: 0.0
perspective: 0.0
hsv_h: 0.013
hsv_s: 0.353
hsv_v: 0.194
```

### 12. Why MuSGD

The official YOLO26 training recipe uses MuSGD for the released checkpoints. MuSGD combines SGD-like
updates with Muon-style orthogonalized updates for suitable weight matrices.

This project uses the published YOLO26-M optimizer schedule as an informed starting point:

```text
lr0          = 0.00038
lrf          = 0.882
momentum     = 0.948
weight_decay = 0.00027
warmup       ≈ 1 epoch
```

This is more explicit and reproducible than `optimizer=auto`.

### 13. Loss weights

YOLO26 does not use the old Distribution Focal Loss in the same way as older YOLO generations. In
the current YOLO26 recipe, the `dfl` gain weights a normalized box-distance L1 term.

For M:

```text
box = 9.83
cls = 0.65
dfl = 0.96
```

The project retains these recipe-informed gains rather than inventing arbitrary numbers.

### 14. Augmentation strategy

The official COCO recipe for YOLO26m is aggressive. This project intentionally does **not** copy it
blindly because fixed-camera industrial inspection differs from generic COCO scenes.

Used:

```text
mosaic    0.80
mixup     0.05
scale     0.50
translate 0.10
fliplr    0.50
```

Disabled:

```text
flipud
degrees
shear
perspective
```

Reason: geometric transforms that are unrealistic for the production camera can create synthetic
samples that reduce rather than improve real defect accuracy.

Color jitter is retained at moderate levels because lighting can change between captures.

### 15. Close mosaic

```text
close_mosaic = 10
```

Mosaic is disabled in the final ten epochs. The model can then converge on normal single-image
geometry rather than seeing mosaic-composed images right until training stops.

### 16. AMP

```text
amp = true
```

Automatic Mixed Precision reduces memory use and improves Tensor Core utilization on the T4. It is
a standard choice for modern detector training.

### 17. AutoBatch

```text
batch = 0.70
```

This asks Ultralytics to target approximately 70% GPU memory rather than hard-coding a batch that
may OOM when the runtime changes.

This is important for `YOLO26m + 960`.

### 18. Early stopping

```text
epochs   = 120
patience = 30
```

120 is a maximum, not a requirement to consume every epoch. If validation fitness does not improve
for the patience window, training can stop and `best.pt` remains the checkpoint used for final
evaluation.

### 19. Checkpoint policy

Important files:

```text
best.pt  → best validation checkpoint → use this
last.pt  → final training state       → do not automatically call this "best"
```

Academic results must be generated from `best.pt`.

### 20. Evaluation paths in YOLO26

The pipeline measures:

```text
default:
one-to-many head → NMS

end-to-end:
one-to-one head → nms=False
```

Why measure both:

- default path is the conventional accuracy path
- NMS-free path is useful for simplified deployment
- latency and accuracy may differ

Do not report the faster path as "better" if its accuracy is materially lower.

### 21. Precision

```text
Precision = TP / (TP + FP)
```

High precision means predicted defects are usually real.

Industrial interpretation:

```text
low precision → too many false alarms
```

### 22. Recall

```text
Recall = TP / (TP + FN)
```

High recall means real defects are rarely missed.

Industrial interpretation:

```text
low recall → defective items may pass inspection
```

For many quality-control systems, recall is operationally very important.

### 23. mAP@50

AP is calculated from the precision-recall curve. `mAP@50` averages AP over classes at IoU 0.50.

It is easier than mAP50-95 and often looks high. Do not use it alone.

### 24. mAP@50:95

This averages AP over IoU thresholds:

```text
0.50, 0.55, ..., 0.95
```

It penalizes poor localization more strongly and should be the main headline metric.

### 25. Per-class AP

A high average can hide a weak rare class. Always report each class separately.

Example report structure:

| Class | AP50 | AP50-95 | Recall |
|---|---:|---:|---:|
| DiVat | ... | ... | ... |
| DiVatLoiLom | ... | ... | ... |
| LoiChi | ... | ... | ... |
| LoiNhua | ... | ... | ... |
| LoiTray | ... | ... | ... |

### 26. Confusion matrix

The confusion matrix answers:

```text
which classes are confused with which?
```

For visually similar defects, this may be more informative than aggregate mAP.

### 27. Error analysis

The project creates:

```text
error_analysis.csv
```

Each test image records:

- GT object count
- predicted object count
- correct-class IoU matches
- wrong-class matches
- false negatives
- false positives

The diagnostic matching IoU is 0.50.

This error-analysis CSV is separate from official Ultralytics mAP computation.

### 28. Failure cases to inspect manually

After training, inspect at least:

```text
1. smallest defects
2. low-contrast defects
3. multiple defects close together
4. reflective/highlight areas
5. background patterns resembling scratches
6. partially visible defects
7. each class's highest-confidence false positive
8. each class's false negatives
```

### 29. Latency benchmark

The benchmark:

- warms up the network
- synchronizes CUDA before/after timing
- runs held-out test images
- records mean latency
- records median latency
- records p95 latency
- derives FPS

This is more defensible than copying an FPS number from a different GPU.

### 30. ONNX export

Final export:

```text
format  = ONNX
imgsz   = 960
dynamic = true
nms     = false
```

The NMS-free export corresponds to YOLO26's end-to-end deployment path.

### 31. Reproducibility manifest

Every training run writes:

```text
artifacts/manifest.json
```

It records:

- Python
- OS/platform
- Torch
- CUDA
- GPU name
- Ultralytics version
- Roboflow version
- seed
- image size
- model
- training parameters
- metrics
- benchmark results
- elapsed runtime

### 32. Execution model

The repository is code-first. Training is not performed in an interactive notebook, but via reproducible CLI commands:

Flow:

```text
local repo / environment
   ↓
scale-error prepare
   ↓
scale-error train
   ↓
NVIDIA CUDA GPU
   ↓
artifact outputs (runs/weights/best.pt)
   ↓
scale-error evaluate
```

This makes the training pipeline versionable, modular, and reproducible.

### 33. Environment Secrets

Required:

```text
ROBOFLOW_API_KEY
```

It must be stored in environment variables or `.env`.

Never put:

```python
api_key = "actual-secret"
```

inside the repository.

### 34. Expected artifacts

```text
artifacts/
├── dataset/
├── runs/yolo26m_960/
├── evaluation/
├── benchmark/
├── predictions/
├── exports/
├── model_summary.txt
└── manifest.json
```

The whole artifact directory is also bundled into:

```text
scale_error_yolo26_results.zip
```

### 35. What counts as a good result

There is no honest fixed mAP guarantee before the run.

A good result is one that is:

1. measured on held-out data
2. reproducible
3. strong across all important classes
4. not created by train/test leakage
5. accompanied by error analysis
6. usable at acceptable latency

A suspiciously high result with leakage is weaker scientifically than a slightly lower clean result.

### 36. What to show in presentation

Recommended slide/report order:

```text
Problem
↓
Dataset + classes
↓
YOLO26m architecture overview
↓
Why pretrained YOLO26m
↓
Data validation + split
↓
Single training configuration
↓
Training curves
↓
mAP / Precision / Recall
↓
Per-class results
↓
Confusion matrix
↓
Correct detections
↓
Failure cases
↓
Latency
↓
Conclusion
```

### 37. Final technical claim

The correct conclusion is not:

> YOLO26m is the best detector in the world.

The defensible conclusion is:

> Under the fixed Scale_Error_DC v26 split and the documented YOLO26m-960 training protocol, the
> trained checkpoint achieved the reported held-out metrics and exhibited the documented
> accuracy/latency/failure characteristics.

That wording is technically correct and suitable for an academic project.
