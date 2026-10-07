# TỔNG HỢP KẾT QUẢ — Glass Defect Inspector (scale-error-yolo26)

> File này gộp **toàn bộ số liệu** nằm rải rác trong các thư mục `kaggle*` thành một chỗ.
> Ngày tổng hợp: 2026-09-28.
> Test split giữ nguyên: **890 ảnh / 1.573 ground-truth box** (xem mục Dataset).
> Classes (thứ tự index): `0 DiVat`, `1 DiVatLoiLom`, `2 LoiChi`, `3 LoiNhua`, `4 LoiTray`.

---

## 1. Bảng tổng hợp (held-out test, 890 ảnh)

| Model | Backbone | Pre-process | mAP@50 | mAP@75 | mAP@50:95 | Latency (T4) | FPS |
|---|---|---|---|---|---|---|---|
| YOLO11m (Baseline 1) | CSPDarknet | Raw 960×960 | 63.86% | 32.23% | 34.31% | 57.2 ms | 17.5 |
| RT-DETR-L (Baseline 2) | HGNetv2-B4 | Raw 960×960 | 65.22% | 30.57% | 35.02% | 97.3 ms | 10.3 |
| YOLO26m Base | YOLO26 | Raw 960×960 | 69.85% | 36.75% | 38.51% | 67.4 ms | 14.8 |
| YOLO26m (eval lại, `best.pt`) | YOLO26 | Raw 960×960 | 68.79% | 37.59% | 39.00% | 54.0 ms (inference) | — |
| **YOLO26m + ROI + VFL (SOTA)** 🏆 | YOLO26 | ROI Crop | 70.96% | 39.69% | **40.34%** | 58.4 ms | 17.1 |
| **D-FINE-M (Fine-grained Regression)** 🏆 | HGNetv2-B2 | ROI Crop | **76.90%** | **46.00%** | **44.40%** | 57.27 ms | 17.5 |

Benchmark dùng 100 ảnh, `imgsz=960`, end-to-end (pre + infer + post), GPU T4.

---

## 2. Chi tiết từng run

### 2.1 YOLO26m Base — `kaggle_output/`
| | NMS | NMS-Free |
|---|---|---|
| mAP50:95 | **38.51%** | 38.17% |
| mAP50 | 69.85% | 69.05% |
| mAP75 | 36.75% | 37.15% |
| Precision | 74.66% | 71.75% |
| Recall | 67.87% | 67.10% |
| Latency (infer / e2e) | 49.8 ms / 67.4 ms | 50.5 ms / 67.4 ms* |

Per-class mAP50:95 (NMS): DiVat 17.53 · DiVatLoiLom 27.89 · LoiChi 62.38 · LoiNhua 50.01 · LoiTray 34.75
Per-class mAP50:95 (NMS-Free): DiVat 18.26 · DiVatLoiLom 28.05 · LoiChi 60.60 · LoiNhua 50.54 · LoiTray 33.42

Error analysis (890 ảnh test): **428 FN · 277 FP · 38 wrong-class**
Nguồn: `kaggle_output/artifacts/evaluation/metrics_*.json`, `error_analysis_summary.json`, `kaggle_output/artifacts/benchmark/benchmark_*.json`

### 2.2 YOLO26m eval lại (`best.pt`) — `kaggle_eval_results/` (+ bản sao `kaggle_weights_download/`)
| | NMS=True |
|---|---|
| mAP50:95 | **39.00%** |
| mAP50 | 68.79% |
| mAP75 | 37.59% |
| Precision | 72.87% |
| Recall | 61.37% |
| Latency (infer) | 54.0 ms |

Per-class mAP50:95: DiVat 18.40 · DiVatLoiLom 30.78 · LoiChi 62.30 · LoiNhua 50.55 · LoiTray 32.99
Nguồn: `kaggle_eval_results/metrics_nms_True.json` (= `kaggle_weights_download/artifacts/evaluation/metrics_nms_True.json`, **2 file trùng nhau**)

### 2.3 YOLO11m (Baseline 1) — `kaggle_output_yolo11/`
| | NMS | NMS-Free |
|---|---|---|
| mAP50:95 | **34.31%** | 34.31% |
| mAP50 | 63.86% | 63.86% |
| mAP75 | 32.23% | 32.23% |
| Precision | 80.23% | 80.23% |
| Recall | 58.58% | 58.58% |
| E2E latency | 57.2 ms (17.48 FPS) | — |

Per-class mAP50:95: DiVat 13.18 · DiVatLoiLom 21.80 · LoiChi 58.93 · LoiNhua 46.94 · LoiTray 30.69
Error analysis: **478 FN · 270 FP · 46 wrong-class**
Nguồn: `kaggle_output_yolo11/artifacts/{evaluation,benchmark}/…`

### 2.4 RT-DETR-L (Baseline 2) — `kaggle_output_rtdetr_final/`
| | kết quả |
|---|---|
| mAP50:95 | **35.02%** |
| mAP50 | 65.22% |
| mAP75 | 30.57% |
| Precision | 73.09% |
| Recall | 65.47% |
| E2E latency | 97.3 ms (10.28 FPS) |

Per-class mAP50:95: DiVat 16.30 · DiVatLoiLom 25.45 · LoiChi 59.24 · LoiNhua 40.53 · LoiTray 33.57
Error analysis: **320 FN · 945 FP · 59 wrong-class** (FP cao nhất — hay báo thừa)
Nguồn: `kaggle_output_rtdetr_final/artifacts/{evaluation,benchmark}/…`

### 2.5 D-FINE-M (mô hình tốt nhất) — `kaggle_output_dfine_m/`, `artifacts/dfine_m/`
| Metric | Giá trị |
|---|---|
| **mAP50:95** | **44.40%** |
| mAP50 | 76.90% |
| mAP75 | 46.00% |
| mAP small / medium / large | 35.3% / 45.8% / 55.9% |
| Parameters | 19,3 M (HGNetv2-B2) |
| Input | 960×960 |
| Latency (forward) | 57.27 ms → 17.5 FPS |
| Epochs | 100 (Stage 1: 88 epoch multi-scale, Stage 2: 12 epoch FDR refinement) |
| Pretrained | COCO `dfine_m_coco.pth` |
| Test | 890 ảnh / 1.573 box |

Nguồn: `kaggle_output_dfine_m/metrics_dfine_m.json`, `kaggle_output_dfine_m/benchmark_summary.txt`,
log từng epoch: `kaggle_output_dfine_m/output_dfine_m/log.txt`, `results_report/output_dfine_m/{log.txt,train_stdout.log}`

### 2.6 YOLO26m + ROI + VFL (SOTA) — chỉ còn số liệu trong README
mAP50:95 **40.34%** · mAP50 70.96% · mAP75 39.69% · Precision 73.59% · Recall 66.31% · 58.4 ms · 17.1 FPS
Train: 100 epoch, Varifocal Loss + class weights, MuSGD, NMS-Free (one-to-one), ROI Crop.

⚠️ Kernel train run này **bị lỗi** (`git clone` repo thất bại → `FileNotFoundError`),
log rỗng 0 byte. Weight `best.pt` 125 MB đã chuyển về `weights/yolo26m_roi_vfl_best.pt`.
Số liệu trên lấy từ `README.md` / `results_report/benchmark_summary.txt` — **không còn file metrics gốc**.

---

## 3. Dataset & split

| Split | Ảnh | Box | Ghi chú |
|---|---|---|---|
| Roboflow train | 5.578 | 9.166 | giữ nguyên |
| Roboflow valid | 1.768 | 3.128 | chia tiếp 50/50 bằng multilabel-stratify, seed 42 |
| → val | 878 | 1.555 | |
| → **test (held-out)** | **890** | **1.573** | |

- Strategy: `keep Roboflow train; multilabel-stratify original valid 50/50 into val/test`, seed 42.
- **Leakage: 0** duplicate giữa các split.
- Box nhỏ dominante: ~85% box < 1% diện tích ảnh, ~64% box < 0,25%.
- Class counts (train): LoiChi 5.282 · DiVatLoiLom 2.011 · DiVat 1.300 · LoiNhua 324 · LoiTray 249
- Class counts (test): LoiChi 806 · DiVatLoiLom 436 · DiVat 215 · LoiNhua 76 · LoiTray 40
- Label errors: 0 fatal; 1 file label rỗng ở valid.
Nguồn: `kaggle_output/artifacts/dataset/{dataset_report,split_report,leakage_report,generated_split_report}.json`

---

## 4. Hyperparameter chính

**YOLO26m (từ `kaggle_eval_results/args.yaml`)**
`model=yolo26m.pt` · `epochs=120` · `patience=15` · `batch=8` · `imgsz=960` · `optimizer=MuSGD`
`seed=42, deterministic` · `lr0=0.00038` · `lrf=0.882` · `momentum=0.948` · `weight_decay=0.00027`
`mosaic=0.3` · `mixup=0.05` · `close_mosaic=10` · `cls_remap=true` · `amp=true` · `device=0` (T4)

**D-FINE-M**: 100 epoch = stage 1 (88 epoch, multi-scale) + stage 2 (12 epoch FDR refinement), pretrained COCO.

---

## 5. Vị trí file (sau khi dọn 2026-09-28)

Thư mục `kaggle*` đã **xóa hết**; weight duy nhất được gom về `weights/`.
Cột "Nguồn gốc" chỉ để đối chiếu (đường dẫn này không còn tồn tại).

| Nội dung | Hiện tại | Nguồn gốc (đã xóa) |
|---|---|---|
| D-FINE-M final (44.40%) | `weights/dfine_m_best.pth` | `kaggle_output_dfine_m/best.pth` |
| D-FINE-M stage-1 ckpt | `weights/dfine_m_stg1.pth` | `kaggle/checkpoint_dfine/best_stg1.pth` |
| D-FINE-M epoch 87 / last | `weights/dfine_m_last_ep87.pth`, `weights/dfine_m_last.pth` | `kaggle/checkpoint_dfine/` |
| D-FINE pretrained COCO | `weights/dfine_m_coco.pth` | `kaggle_output_dfine_m/weights/` |
| D-FINE-M local copy (bản 26/09) | `artifacts/dfine_m/best.pth` | — (giữ nguyên) |
| YOLO26m eval `best.pt` (39.00%) | `weights/yolo26m.pt` ← demo load cái này | `kaggle_eval_results/best.pt` |
| YOLO26m run 960 | `weights/yolo26m_960_best.pt` | `kaggle_output/artifacts/runs/yolo26m_960/weights/best.pt` |
| YOLO26m base | `weights/yolo26m_base.pt` | `kaggle_output/yolo26m.pt` |
| **YOLO26m + ROI + VFL (40.34%)** | `weights/yolo26m_roi_vfl_best.pt` | `kaggle_outputs/.../yolo26m/weights/best.pt` |
| YOLO11m / RT-DETR weights, exports, predictions, curves | — | đã xóa (chỉ còn số liệu ở mục 2) |
| Metrics, benchmark, error analysis, dataset report, args.yaml | **đã gộp vào file này** | `kaggle_*/artifacts/…` |
| D-FINE train log | `results_report/output_dfine_m/{log.txt,train_stdout.log}` | giữ nguyên |
| Benchmark table | `results_report/benchmark_summary.txt` | giữ nguyên |
| Kernel script D-FINE | `kaggle/kernel_dfine/{run_dfine.py,kernel-metadata.json}` | giữ nguyên |

Ghi chú: `kaggle/pull_kernel_output.py` (script tải output kernel) đã bị xóa — nếu cần chạy lại
thì phải viết lại (README dòng 311 vẫn còn nhắc tới). Dataset ROI local vẫn ở `data/processed/roi/` (4,8 GB),
bản copy trên Kaggle (`kaggle/output_v4/data_roi`, 455 MB) đã xóa; repo D-FINE clone lại từ GitHub nếu cần.
