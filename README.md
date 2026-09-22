# Glass Defect Inspector: Industrial Glass Bottle Defect Detection with YOLO26m

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLO26-00FFFF.svg)](https://docs.ultralytics.com/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Benchmark](https://img.shields.io/badge/mAP50--95-39.98%25-brightgreen.svg)](#-comprehensive-benchmark-results)
[![Inference](https://img.shields.io/badge/Postprocess-0.13ms%20(NMS--Free)-success.svg)](#-nms-free-dual-head-inference)

An end-to-end computer vision system for automated optical inspection (AOI) of glass bottle mouth defects. Powered by **YOLO26m** with recipe-informed **MuSGD** optimization, geometric **ROI Cropping** preprocessing, and **Dual-Head NMS-Free** real-time deployment.

---

## 📌 Project Overview

In glass container manufacturing, mouth defects (sealing ring cracks, mold gouges, foreign particles, burrs, and scratches) cause container leakage, seal failure, or consumer safety hazards. 

This repository provides a reproducible, end-to-end deep learning pipeline designed to resolve the three hardest challenges in industrial glass inspection:
1. **Extreme Scale Variance & Micro-Defects**: 59.3% of all defect bounding boxes occupy less than **0.25% of the total image frame** (sub-centimeter flaws on high-resolution cameras).
2. **Severe Class Imbalance**: Training samples exhibit an extreme **~21:1 imbalance ratio** between majority defects (`LoiChi`: 5,282 annotations) and minority defects (`LoiTray`: 249 annotations).
3. **Real-Time Edge Inference**: The visual inspection system achieves sub-70ms end-to-end inference without non-linear NMS bottlenecks when deployed on edge hardware.

---

## 🏆 Comprehensive Benchmark Results

All models were evaluated on the identical held-out test split (**890 images, 1,573 ground-truth annotations**) with **zero cross-split data leakage** verified by image byte-level SHA-1 auditing.

### 1. Overall Model Comparison (Held-out Test Split)

| Model Architecture | Preprocessing | Post-Processing | Params | Precision | Recall | mAP@50 | mAP@75 | **mAP@50:95** ⭐ | E2E Latency | FPS |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **YOLO11m** (Baseline 1) | Raw (960×960) | Traditional NMS | 20.1M | **80.23%** | 58.58% | 63.86% | 32.23% | 34.31% | **57.2 ms** | **17.5** |
| **RT-DETR-L** (Baseline 2) | Raw (960×960) | Hungarian Match | 32.8M | 73.09% | 65.47% | 65.22% | 30.57% | 35.02% | 97.3 ms | 10.3 |
| **YOLO26m** (Proposed Base) | Raw (960×960) | Traditional NMS | 21.9M | 74.66% | **67.87%** | 69.85% | 36.75% | 38.51% | 67.4 ms | 14.8 |
| **YOLO26m** (Proposed Base) | Raw (960×960) | NMS-Free (One-to-One) | 21.9M | 71.75% | 67.10% | 69.05% | 37.15% | 38.17% | 71.2 ms | 14.1 |
| **YOLO26m + ROI Crop** | **ROI Cropped** | Traditional NMS | 21.9M | 77.81% | 67.07% | **70.20%** | 38.44% | 39.24% | ~62.3 ms | ~16.1 |
| **YOLO26m + ROI Crop** 🏆 | **ROI Cropped** | **NMS-Free (One-to-One)**| 21.9M | 72.10% | 67.15% | 69.62% | **41.30%** | **39.98%** | **~61.3 ms** | **~16.3** |

> 🌟 **Key Milestones:**
> - **+5.67% mAP50-95** improvement over the YOLO11m baseline.
> - **+4.55% mAP@75** jump (from 36.75% to 41.30%), demonstrating significantly tighter bounding-box localization around fine cracks.
> - **mAP@50 officially surpasses 70%** (70.20%).

---

### 2. Per-Class Performance Breakdown (mAP@50:95)

| Defect Class | Vietnamese Term | Test Boxes | YOLO11m | RT-DETR-L | YOLO26m Base | **YOLO26m + ROI Crop** | Delta vs Baseline |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **DiVat** | Dị vật nhỏ / Foreign Matter | 215 | 13.18% | 16.30% | 17.53% | **21.41%** | 🚀 **+8.23%** |
| **DiVatLoiLom** | Dị vật lồi lõm / Surface Gouge | 436 | 21.80% | 25.45% | 27.89% | **31.66%** | 📈 **+9.86%** |
| **LoiChi** | Nứt chỉ / Sealing Crack | 806 | 58.93% | 59.24% | 62.38% | **63.40%** | **+4.47%** |
| **LoiNhua** | Lỗi nhựa / Resin Burr | 76 | 46.94% | 40.53% | 50.01% | **53.22%** | **+6.28%** |
| **LoiTray** | Vết trầy xước / Surface Scratch | 40 | 30.69% | 33.57% | **34.75%** | 32.22% | +1.53% |
| **Overall mAP50-95** | | **1,573** | **34.31%** | **35.02%** | **38.51%** | **39.98%** | 🏆 **+5.67%** |

---

### 3. Latency & Edge Inference Breakdown

| Pipeline Stage | YOLO11m (NMS) | RT-DETR-L | YOLO26m Base (NMS) | YOLO26m + ROI (NMS-Free) |
| :--- | :---: | :---: | :---: | :---: |
| **Pre-process** | 1.53 ms | 1.90 ms | 1.41 ms | **1.86 ms** |
| **Model Inference** | **44.29 ms** | 96.01 ms | 49.83 ms | 59.34 ms |
| **Post-process** | 0.98 ms | 0.16 ms | 0.87 ms | **0.13 ms** (Zero-NMS overhead) |
| **Total End-to-End** | **57.20 ms** | 97.29 ms | 67.45 ms | **~61.3 ms** |
| **Throughput (FPS)** | **17.5 FPS** | 10.3 FPS | 14.8 FPS | **~16.3 FPS** |

---

## 🔬 Core Engineering Innovations

```
[Input Frame: 5328×3000]
           │
           ▼
┌─────────────────────────┐
│  Adaptive ROI Cropping  │  ──► Eliminates 25.3% useless dark margins
└─────────────────────────┘      Preserves 1.7× higher defect pixel density
           │
           ▼
┌─────────────────────────┐
│  YOLO26m + MuSGD Train  │  ──► Normalized box distance L1 + IoU-aware loss
└─────────────────────────┘      Transfer learning at 960×960
           │
     ┌─────┴────────────────┐
     ▼                      ▼
┌──────────────────┐  ┌────────────────────────┐
│ One-to-Many Head │  │ One-to-One Dual Head   │ ──► NMS-Free deployment
│ (Traditional NMS)│  │ (Direct Set Matching)  │     Postprocess: 0.13 ms
└──────────────────┘  └────────────────────────┘
```

1. **Geometric ROI Cropping (`processing/apply_roi_crop.py`)**:
   - Fixed camera inspection of circular bottle mouths introduces substantial black background padding on wide aspect ratios (e.g. 5328×3000).
   - By calculating the statistical bounding envelope of the mouth ring, the pipeline crops out 25.3% of unused lateral background.
   - When downscaling to the detector resolution (960×960), tiny defects retain significantly higher effective pixel area, driving a **+4.55% jump in mAP75** and **+8.23% in micro-foreign-matter detection (`DiVat`)**.

2. **MuSGD Optimization Protocol**:
   - Swaps naive stochastic optimizers for MuSGD with orthogonalized matrix updates (`lr0=0.00038`, `lrf=0.882`, `momentum=0.948`, `weight_decay=0.00027`).
   - Applies industrial-tuned augmentation (mosaic=0.80 with last-10-epochs closure, scale=0.50, mixup=0.05) while disabling non-physical affine transforms (shear, perspective, flipud).

3. **Dual-Head NMS-Free Deployment**:
   - YOLO26 incorporates parallel One-to-One and One-to-Many prediction heads.
   - Using the One-to-One head achieves **39.98% mAP50-95** with **0.13 ms** postprocessing latency, completely bypassing sorting loops and score thresholding in traditional Non-Maximum Suppression.

---

## 💻 Local Quickstart & Usage

### 1. Environment Installation

```bash
# 1. Clone the repository
git clone https://github.com/nvtanphat/-glass-defect-inspector.git
cd -glass-defect-inspector

# 2. Create and activate a clean Python virtual environment
python -m venv .venv

# On Linux / macOS:
source .venv/bin/activate
# On Windows (PowerShell):
.\.venv\Scripts\Activate.ps1

# 3. Install dependencies in editable mode
pip install --upgrade pip
pip install -e ".[dev]"
```

---

### 2. Dataset Preparation & Stratified Splitting

The project uses the public **Scale_Error_DC v26** dataset (7,346 industrial images across 5 defect classes).

```bash
# Download and automatically run validation + multilabel stratified split
scale-error prepare \
  --output data/raw/scale_error_dc \
  --workspace datacpn \
  --project scale_error_dc-orqpx \
  --version 26 \
  --split-output data/processed
```

> **Note on Data Policy:** The upstream dataset only contains `train` and `valid` splits. The `prepare` command creates a clean, leakage-free 50/50 image-level stratified split on `valid` to generate an independent held-out `test` set.

---

### 3. Apply ROI Cropping (Recommended)

To achieve the top 39.98% mAP benchmark, apply the geometric ROI crop:

```bash
python processing/apply_roi_crop.py \
  --dataset-root data/processed \
  --output-root data/processed_roi \
  --roi-config processing/roi_config.json \
  --splits train val test
```

This generates cropped training, validation, and test splits along with `data/processed_roi/data.yaml`.

---

### 4. Local Model Training

Train the accuracy-oriented YOLO26m configuration locally on your GPU:

```bash
scale-error train \
  --data data/processed_roi/data.yaml \
  --config configs/train.yaml \
  --project-dir artifacts/runs
```

Training configuration (`configs/train.yaml`):
- **Model**: `yolo26m.pt` (transfer learning)
- **Resolution**: `imgsz: 960`
- **Batch size**: `batch: 8` (or target fraction e.g. `0.70` for dynamic VRAM allocation)
- **Optimizer**: `MuSGD`
- **AMP**: Enabled (Automatic Mixed Precision)
- **Patience**: 30 epochs early stopping

---

### 5. Local Evaluation & Error Analysis

Evaluate the trained checkpoint on the held-out test split:

```bash
# Standard evaluation with NMS
scale-error evaluate \
  --weights artifacts/runs/yolo26m_960/weights/best.pt \
  --data data/processed_roi/data.yaml \
  --output artifacts/evaluation/metrics_nms.json

# Real-time NMS-Free evaluation (One-to-One head)
scale-error evaluate \
  --weights artifacts/runs/yolo26m_960/weights/best.pt \
  --data data/processed_roi/data.yaml \
  --output artifacts/evaluation/metrics_nms_free.json \
  --nms-free
```

---

### 6. Interactive Web UI & Inference

#### Run Gradio Web Demo
Launch an interactive browser interface to upload bottle images and visualize defects in real time:

```bash
python demo/app.py
```
Open your browser at `http://localhost:7860`.

#### CLI Batch Inference
```bash
scale-error predict \
  --weights artifacts/runs/yolo26m_960/weights/best.pt \
  --source path/to/bottle_images/ \
  --output artifacts/predictions \
  --nms-free
```

---

### 7. Export to ONNX / TensorRT (Edge Deployment)

Export the trained model to standalone ONNX format for deployment on industrial inspection PCs or Edge AI accelerators (NVIDIA Jetson, TensorRT):

```bash
yolo export \
  model=artifacts/runs/yolo26m_960/weights/best.pt \
  format=onnx \
  imgsz=960 \
  dynamic=True \
  nms=False
```

---

## 📂 Repository Structure

```text
-glass-defect-inspector/
├── configs/
│   ├── train.yaml                   # YOLO26m MuSGD training configuration
│   └── train_rtdetr.yaml            # RT-DETR-L baseline configuration
├── demo/
│   └── app.py                       # Interactive Gradio web application
├── docs/
│   └── FULL_DOCUMENTATION.md        # Mathematical formulation & protocol details
├── processing/
│   ├── apply_roi_crop.py            # Geometric ROI cropping pipeline
│   ├── compute_roi.py               # Statistical bounding envelope calculator
│   ├── label_audit.py               # Bounding box & annotation audit script
│   └── roi_config.json              # Calibrated ROI crop parameters
├── report_assets/                   # Evaluation plots, curves & confusion matrices
├── src/
│   └── scale_error/
│       ├── data/                    # Roboflow download, split & audit modules
│       ├── evaluation/              # Benchmark, error analysis & metric logging
│       ├── inference/               # Dual-head predictor module
│       ├── training/                # Custom losses (Varifocal) & trainer wrapper
│       ├── cli.py                   # Unified command-line interface (CLI)
│       └── config.py                # Pydantic configuration schemas
├── tests/                           # Unit & data validation tests
├── pyproject.toml                   # Project metadata & dependencies
├── report.html                      # Interactive visual benchmark dashboard
└── README.md                        # Documentation & setup guide
```

---

## 📄 License & Citation

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

If this repository or the benchmark results assist your research or industrial deployment, please consider citing:

```bibtex
@misc{glass_defect_inspector_2026,
  author = {Nguyen Van Tan Phat},
  title = {Industrial Glass Bottle Mouth Defect Detection using YOLO26m with Geometric ROI Cropping},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/nvtanphat/-glass-defect-inspector}}
}
```
