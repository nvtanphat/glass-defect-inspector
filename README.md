# Glass Defect Inspector: Industrial Glass Bottle Defect Detection

[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLO26-00FFFF.svg)](https://docs.ultralytics.com/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Test mAP50-95](https://img.shields.io/badge/Best%20mAP50--95-44.40%25-brightgreen.svg)](#-benchmark-results-held-out-test-split)
[![Inference Speed](https://img.shields.io/badge/Speed-17.5%20FPS%20(57.3ms)-orange.svg)](#3-phân-rã-độ-trễ-suy-luận-tesla-t4-gpu)

Pipeline phát hiện lỗi miệng chai thủy tinh công nghiệp hỗ trợ **4 kiến trúc mô hình**: **YOLO26m**, **YOLO11m**, **RT-DETR-L**, và **D-FINE-M**, kết hợp tiền xử lý **ROI Cropping**, hàm mất mát **Varifocal Loss + Class Weighting**, và cơ chế suy luận **NMS-Free**.

---

## 📊 Benchmark Results (Held-out Test Split)

Toàn bộ mô hình được đánh giá trên cùng một tập **Test độc lập** (890 ảnh, 1.573 nhãn ground-truth), hoàn toàn tách biệt khỏi tập huấn luyện (`train`) và kiểm định (`val`).

### 1. So sánh tổng thể trên tập Test

| Mô hình | Backbone | Tiền xử lý | mAP@50 | mAP@75 | **mAP@50:95** | Độ trễ (T4) | FPS |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **YOLO11m** (Baseline 1) | CSPDarknet | Raw 960×960 | 63.86% | 32.23% | 34.31% | 57.2 ms | 17.5 |
| **RT-DETR-L** (Baseline 2) | HGNetv2-B4 | Raw 960×960 | 65.22% | 30.57% | 35.02% | 97.3 ms | 10.3 |
| **YOLO26m Base** | YOLO26 | Raw 960×960 | 69.85% | 36.75% | 38.51% | 67.4 ms | 14.8 |
| **YOLO26m + ROI + VFL** | YOLO26 | ROI Crop | 70.96% | 39.69% | 40.34% | 58.4 ms | 17.1 |
| **D-FINE-M** 🏆 | HGNetv2-B2 | ROI Crop | **76.90%** | **46.00%** | **44.40%** | 57.27 ms | 17.5 |

> **D-FINE-M** đạt kết quả tốt nhất trên mọi chỉ số mAP, vượt YOLO26m+ROI+VFL khoảng **+6% mAP@50** và **+4% mAP@50:95**, với tốc độ suy luận tương đương.

---

### 2. Chi tiết hiệu năng từng lớp lỗi (mAP@50:95 trên tập Test)

| Lớp lỗi | Định nghĩa lỗi | Số nhãn Test | YOLO11m | RT-DETR-L | YOLO26m + ROI + VFL | **D-FINE-M** |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **DiVat** | Dị vật nhỏ | 215 | 13.18% | 16.30% | 20.85% | — |
| **DiVatLoiLom** | Dị vật lồi lõm | 436 | 21.80% | 25.45% | 30.95% | — |
| **LoiChi** | Nứt chỉ viền miệng | 806 | 58.93% | 59.24% | 62.16% | — |
| **LoiNhua** | Lỗi ba-via nhựa | 76 | 46.94% | 40.53% | 53.19% | — |
| **LoiTray** | Vết trầy xước | 40 | 30.69% | 33.57% | 34.56% | — |
| **Trung bình** | | **1.573** | **34.31%** | **35.02%** | **40.34%** | **44.40%** |

> D-FINE-M per-class metrics: mAP_small=35.3%, mAP_medium=45.8%, mAP_large=55.9%.

---

### 3. Phân rã độ trễ suy luận (Tesla T4 GPU)

| Giai đoạn suy luận | YOLO11m (NMS) | RT-DETR-L | YOLO26m + ROI + VFL (NMS-Free) | D-FINE-M |
| :--- | :---: | :---: | :---: | :---: |
| **Tổng độ trễ** | 57.20 ms | 97.29 ms | 58.42 ms | 57.27 ms |
| **Tốc độ (FPS)** | 17.5 | 10.3 | 17.1 | 17.5 |

---

## 🏗️ Kiến trúc 4 mô hình

| Mô hình | Framework | Hàm mất mát | Optimizer | Đặc điểm |
| :--- | :---: | :---: | :---: | :--- |
| **YOLO26m** | Ultralytics | Varifocal Loss + Class Weights | MuSGD | NMS-Free Dual-Head, ROI Crop |
| **YOLO11m** | Ultralytics | BCE (default) | Auto SGD | Baseline nhanh, precision cao |
| **RT-DETR-L** | Ultralytics | Hungarian Match | AdamW | Transformer encoder-decoder |
| **D-FINE-M** | D-FINE | Fine-grained Distribution Refinement | AdamW | 2-stage training (88+12 epochs), HGNetv2 backbone |

---

## 🛠️ Các phương pháp kỹ thuật chính

1. **Cắt vùng quan tâm (ROI Cropping — `processing/apply_roi_crop.py`)**:
   - Loại bỏ vùng nền đen cố định không chứa sản phẩm xung quanh miệng chai (tọa độ: `[1448, 900, 4228, 2300]`, giảm 25.3% diện tích thừa).
   - Tăng mật độ điểm ảnh hữu ích của các khuyết tật kích thước nhỏ khi đưa về 960×960.

2. **Varifocal Loss & Trọng số nghịch đảo tần suất (`losses.py`)**:
   - Thay thế BCE bằng Varifocal Loss (VFL) với continuous soft targets từ TaskAlignedAssigner.
   - Nhân trọng số nghịch đảo tần suất lớp để cân bằng gradient giữa lớp đa số (`LoiChi`: 5.282 nhãn) và thiểu số (`LoiTray`: 249 nhãn).

3. **D-FINE Fine-grained Distribution Refinement**:
   - Stage 1 (88 epochs): Multi-scale augmentation với PhotometricDistort, ZoomOut, IoUCrop.
   - Stage 2 (12 epochs): FDR refinement — tắt augmentation, tinh chỉnh phân phối hồi quy.
   - Backbone HGNetv2-B2 pretrained COCO, AdamW optimizer.

4. **Suy luận NMS-Free (Dual Head — YOLO26m)**:
   - Sử dụng One-to-One head gán nhãn 1-1, loại bỏ hoàn toàn NMS (post-process 0.12 ms).

---

## 🚀 Hướng dẫn cài đặt và chạy Local

### 1. Cài đặt môi trường

```bash
# Clone repository
git clone https://github.com/nvtanphat/-glass-defect-inspector.git
cd -glass-defect-inspector

# Tạo môi trường ảo
python -m venv .venv

# Kích hoạt (Windows PowerShell)
.\.venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate

# Cài đặt thư viện
pip install --upgrade pip
pip install -e ".[dev,demo]"
```

---

### 2. Chuẩn bị dữ liệu và phân chia tập Test

```bash
scale-error prepare \
  --output data/raw/scale_error_dc \
  --workspace datacpn \
  --project scale_error_dc-orqpx \
  --version 26 \
  --split-output data/processed
```

---

### 3. Tiền xử lý ROI Crop

```bash
python processing/apply_roi_crop.py \
  --dataset-root data/processed \
  --output-root data/processed_roi \
  --roi-config processing/roi_config.json \
  --splits train val test
```

---

### 4. Huấn luyện mô hình

#### YOLO26m + ROI + VFL (Ultralytics)

```bash
scale-error train \
  --data data/processed_roi/data.yaml \
  --config configs/train_yolo26m.yaml \
  --project-dir artifacts/runs \
  --dataset-report data/processed/dataset_report.json
```

#### YOLO11m (Ultralytics)

```bash
scale-error train \
  --data data/processed_roi/data.yaml \
  --config configs/train_yolo11m.yaml \
  --project-dir artifacts/runs
```

#### RT-DETR-L (Ultralytics)

```bash
scale-error train \
  --data data/processed_roi/data.yaml \
  --config configs/train_rtdetr.yaml \
  --project-dir artifacts/runs
```

#### D-FINE-M (Framework riêng)

```bash
scale-error train-dfine \
  --data data/processed_roi/data.yaml \
  --dfine-repo results/D-FINE \
  --weights results/weights/dfine_m_coco.pth \
  --output-dir artifacts/runs/dfine_m
```

---

### 5. Đánh giá trên tập Test

#### Ultralytics models (YOLO26m / YOLO11m / RT-DETR-L)

```bash
# NMS-Free (One-to-One Head)
scale-error evaluate \
  --weights artifacts/runs/yolo26m/weights/best.pt \
  --data data/processed_roi/data.yaml \
  --output artifacts/evaluation/metrics_nms_free.json \
  --nms-free

# NMS truyền thống
scale-error evaluate \
  --weights artifacts/runs/yolo26m/weights/best.pt \
  --data data/processed_roi/data.yaml \
  --output artifacts/evaluation/metrics_nms.json
```

#### D-FINE-M

```bash
scale-error evaluate-dfine \
  --weights results/best.pth \
  --dfine-repo results/D-FINE \
  --data data/processed_roi/data.yaml \
  --output artifacts/evaluation/metrics_dfine.json
```

---

### 6. Giao diện Web Dashboard & Dự đoán

#### Giao diện Web Gradio Dashboard

```bash
python demo/app.py
```

Mở trình duyệt tại `http://localhost:7860`. Tính năng:
- **Thẻ KPI thời gian thực**: Tổng lỗi, số chủng loại, độ trễ (ms).
- **Tab trực quan hóa kép**: Bounding box phân loại lỗi + ROI Preview.
- **Bảng dữ liệu chi tiết**: Tọa độ XYXY, confidence, diện tích (px²).
- **Chế độ suy luận**: NMS-Free (0.12ms post-process) hoặc Standard NMS.

#### Dự đoán qua CLI

```bash
scale-error predict \
  --weights artifacts/runs/yolo26m/weights/best.pt \
  --source path/to/images/ \
  --output artifacts/predictions \
  --nms-free
```

---

### 7. Xuất định dạng ONNX

```bash
yolo export \
  model=artifacts/runs/yolo26m/weights/best.pt \
  format=onnx \
  imgsz=960 \
  dynamic=True \
  nms=False
```

---

## 📁 Cấu trúc thư mục

```text
-glass-defect-inspector/
├── configs/
│   ├── train_yolo26m.yaml           # YOLO26m (MuSGD + VFL)
│   ├── train_yolo11m.yaml           # YOLO11m (Auto SGD)
│   ├── train_rtdetr.yaml            # RT-DETR-L (AdamW)
│   └── train_dfine_m.yaml           # D-FINE-M (reference config)
├── demo/
│   └── app.py                       # Giao diện Gradio Dashboard
├── docs/
│   └── FULL_DOCUMENTATION.md        # Tài liệu chi tiết
├── kaggle/                          # Scripts push & train trên Kaggle cloud GPU
│   ├── bundle_and_upload.py         # Đóng gói + upload code lên Kaggle
│   ├── code_bundle/                 # Source code bundle (upload as dataset)
│   ├── code_bundle_upload/          # Metadata upload
│   ├── kernel/                      # Kernel training YOLO26m
│   └── kernel_resume/               # Kernel resume training
├── processing/
│   ├── apply_roi_crop.py            # Script cắt ROI
│   ├── compute_roi.py               # Tính tọa độ ROI
│   ├── label_audit.py               # Kiểm tra nhãn
│   └── roi_config.json              # Config ROI
├── report_assets/                   # Đồ thị, confusion matrix
├── src/
│   └── scale_error/
│       ├── data/                    # Tải, chia tập, audit dữ liệu
│       ├── evaluation/
│       │   ├── evaluator.py         # Eval Ultralytics (YOLO/RT-DETR)
│       │   ├── evaluator_dfine.py   # Eval D-FINE (COCO format)
│       │   ├── benchmark.py         # Benchmark tốc độ
│       │   └── error_analysis.py    # Phân tích lỗi chi tiết
│       ├── inference/               # Module suy luận
│       ├── training/
│       │   ├── trainer.py           # Trainer Ultralytics (YOLO/RT-DETR)
│       │   ├── trainer_dfine.py     # Trainer D-FINE (YOLO→COCO convert)
│       │   └── losses.py            # Varifocal Loss + Class Weights
│       ├── cli.py                   # CLI: train / train-dfine / evaluate / predict
│       └── config.py                # Schema cấu hình
├── results/                         # Kết quả training (weights, logs, metrics)
│   ├── D-FINE/                      # D-FINE repository + configs
│   ├── best.pth                     # D-FINE best checkpoint
│   └── weights/                     # Pretrained weights
├── pyproject.toml                   # Package definition
├── report.html                      # Báo cáo trực quan
└── README.md
```

---

## 📄 License

Dự án được phân phối dưới giấy phép [MIT License](LICENSE).
