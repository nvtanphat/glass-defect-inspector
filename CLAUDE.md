# CLAUDE.md — Glass Defect Inspector (scale-error-yolo26)

## Project Overview
Industrial glass bottle mouth defect detection system addressing **Scale Error** (tiny defects < 1% of image area) and **Low Contrast** (transparent glass defects) using:
- **FBDNet-M (SOTA Proposed):** Frequency Boundary Diffusion Network on D-FINE-M with AFTA (Adaptive Frequency Attention), SFIF (Spatial-Frequency Fusion), and DCFSB (Dense Cross-scale Screening Blocks).
- **D-FINE-M Baseline:** Fine-grained Distribution Refinement on HGNetv2-B2.
- **YOLO26m + ROI + VFL:** YOLO26 baseline with Adaptive ROI Cropping and Varifocal Loss.

## Dataset & Protocol
- **Dataset:** Roboflow `scale_error_dc-orqpx` v26.
- **5 Defect Classes:** `0: DiVat`, `1: DiVatLoiLom`, `2: LoiChi`, `3: LoiNhua`, `4: LoiTray`.
- **Split:** Multilabel Stratified Split (Seed 42), 0% Data Leakage.
  - Train: 5,578 images (9,166 boxes)
  - Val: 878 images (1,555 boxes)
  - **Held-out Test: 890 images (1,573 boxes)**.
- **Canonical Input Size:** 960×960 with Adaptive ROI Crop on 5328×3000 source images.

## Benchmark Numbers (Single Source of Truth)
Never fabricate or round numbers. All numbers derived from `docs/reports/BENCHMARK_REPORT.md` and `results/metrics_*.json`:
- **FBDNet-M:** **42.42%** mAP@50:95 (best ep 89) / **42.13%** (final ep 99) | **75.43%** mAP@50 | **28.86%** mAP_s | 22.4M params | 17.5 FPS (T4)
  - Per-class mAP@50:95: DiVat 26.72% | DiVatLoiLom 33.13% | LoiChi 63.19% | LoiNhua 46.57% | LoiTray 41.04%
- **D-FINE-M Baseline:** **41.33%** mAP@50:95 (best ep 82) / **40.49%** (rerun ep 99) | **72.25%** mAP@50 | **28.64%** mAP_s | 19.2M params | 17.5 FPS (T4)
  - Per-class mAP@50:95: DiVat 25.31% | DiVatLoiLom 31.90% | LoiChi 63.82% | LoiNhua 51.78% | LoiTray 29.65%
- **YOLO26m + ROI + VFL:** **40.34%** mAP@50:95 | **70.96%** mAP@50 | 17.1 FPS
- **YOLO26m Base:** 38.51% mAP@50:95 | 69.85% mAP@50
- **RT-DETR-L:** 35.02% mAP@50:95 | 65.22% mAP@50
- **YOLO11m:** 34.31% mAP@50:95 | 63.86% mAP@50

*(Note: An obsolete historical experiment on 1,483 boxes reported 44.40% for D-FINE-M. This has been deprecated; the canonical benchmark is on 1,573 boxes).*

## Standard Commands
- **Interactive Demo:** `python app.py` or `.\run_demo.bat`
- **Verify Benchmarks:** `python tools/check_benchmarks.py`
- **Evaluate D-FINE / FBDNet:** `python tools/eval_dfine.py`
- **Grad-CAM Visualization:** `python tools/visualize_gradcam.py --image <path> --weights <path>`

## Codebase Rules
1. **No loose root files:** Never write scripts, outputs, notebooks, or temp logs directly to root. Use `tools/`, `notebooks/`, `docs/reports/`.
2. **Never commit weights or credentials:** `.pth`, `.pt`, `kaggle-output/`, `.kaggle_accounts.json`, `.env` must remain in `.gitignore`.
3. **Preserve modularity:** Core logic belongs in `src/scale_error/`.
