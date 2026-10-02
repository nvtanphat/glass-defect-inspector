# Glass Defect Inspector: Industrial Glass Bottle Defect Detection

[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLO26-00FFFF.svg)](https://docs.ultralytics.com/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Test mAP50-95](https://img.shields.io/badge/Best%20mAP50--95-44.40%25-brightgreen.svg)](#-benchmark-results-held-out-test-split)
[![Forward Speed](https://img.shields.io/badge/D--FINE%20forward-17.5%20FPS%20(57.3ms)-orange.svg)](#3-phân-rã-độ-trễ-suy-luận-tesla-t4-gpu)

Pipeline phát hiện lỗi miệng chai thủy tinh công nghiệp hỗ trợ **4 kiến trúc mô hình**: **YOLO26m**, **YOLO11m**, **RT-DETR-L**, và **D-FINE-M**. ROI Cropping, Varifocal Loss + Class Weighting được dùng cho biến thể YOLO26m nâng cao; báo cáo chính dùng suy luận NMS mặc định. D-FINE-M dùng ROI Crop và loss gốc của framework D-FINE.

---

## 📊 Kết quả đánh giá (Held-out Test Split)

> **Trạng thái:** YOLO26m + ROI + VFL đã được cập nhật từ lần retrain mới trên ROI giữ đủ nhãn. D-FINE-M vẫn tạm dùng kết quả cũ trong khi chờ job retrain hoàn tất.

Toàn bộ run dùng cùng **890 ảnh nguồn** của tập Test độc lập. YOLO26m + ROI + VFL mới được đánh giá trên đủ **1.573 box**. D-FINE-M trong bảng vẫn là kết quả lịch sử trên ROI cũ còn khoảng **1.483 box**, do đó chưa nên so sánh chính thức cho đến khi D-FINE retrain hoàn tất.

### 1. So sánh tổng thể trên tập Test

| Mô hình | Backbone | Tiền xử lý | mAP@50 | mAP@75 | **mAP@50:95** | Độ trễ (T4) | FPS |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **YOLO11m** (Baseline 1) | CSPDarknet | Raw 960×960 | 63.86% | 32.23% | 34.31% | 57.2 ms | 17.5 |
| **RT-DETR-L** (Baseline 2) | HGNetv2-B4 | Raw 960×960 | 65.22% | 30.57% | 35.02% | 97.3 ms | 10.3 |
| **YOLO26m Base** | YOLO26 | Raw 960×960 | 69.85% | 36.75% | 38.51% | 67.4 ms | 14.8 |
| **YOLO26m + ROI + VFL** | YOLO26 | ROI Crop | 69.22% | 39.91% | 40.12% | 95.59 ms | 10.5 |
| **D-FINE-M** 🏆 | HGNetv2-B2 | ROI Crop | **76.90%** | **46.00%** | **44.40%** | 57.27 ms (forward-only) | 17.5* |

> YOLO26m mới được báo cáo bằng chế độ NMS mặc định. Chỉ so sánh chính thức với D-FINE sau khi D-FINE retrain trên cùng 890 ảnh Test và 1.573 annotation. D-FINE cũ chỉ có latency forward-only nên chưa thể so sánh tốc độ end-to-end.

---

### 2. Chi tiết hiệu năng từng lớp lỗi (mAP@50:95 trên tập Test)

| Lớp lỗi | Định nghĩa lỗi | Số nhãn Test | YOLO11m | RT-DETR-L | YOLO26m + ROI + VFL | **D-FINE-M** |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **DiVat** | Dị vật nhỏ | 215 | 13.18% | 16.30% | 20.61% | — |
| **DiVatLoiLom** | Dị vật lồi lõm | 436 | 21.80% | 25.45% | 30.92% | — |
| **LoiChi** | Nứt chỉ viền miệng | 806 | 58.93% | 59.24% | 62.61% | — |
| **LoiNhua** | Lỗi ba-via nhựa | 76 | 46.94% | 40.53% | 52.23% | — |
| **LoiTray** | Vết trầy xước | 40 | 30.69% | 33.57% | 34.21% | — |
| **Trung bình** | | **1.573** | **34.31%** | **35.02%** | **40.12%** | **44.40%** |

> Số nhãn theo lớp trong bảng là của tập Raw (1.573 box). ROI artifact lịch sử còn 1.483 box: DiVat 198, DiVatLoiLom 383, LoiChi 788, LoiNhua 74, LoiTray 40. D-FINE-M per-class AP không còn; chỉ còn AP theo kích thước: small=35.3%, medium=45.8%, large=55.9%.

---

### 3. Phân rã độ trễ suy luận (Tesla T4 GPU)

| Giai đoạn suy luận | YOLO11m (NMS) | RT-DETR-L | YOLO26m + ROI + VFL (NMS) | D-FINE-M |
| :--- | :---: | :---: | :---: | :---: |
| **Độ trễ được lưu** | 57.20 ms E2E | 97.29 ms E2E | 95.59 ms E2E | 57.27 ms forward-only |
| **Tốc độ suy ra** | 17.5 FPS | 10.3 FPS | 10.5 FPS | 17.5 FPS* |

> \* FPS của D-FINE được suy ra từ forward latency, không bao gồm đầy đủ preprocess/postprocess như benchmark E2E của ba model Ultralytics.

---

## 🏗️ Kiến trúc 4 mô hình

| Mô hình | Framework | Hàm mất mát | Optimizer | Đặc điểm |
| :--- | :---: | :---: | :---: | :--- |
| **YOLO26m** | Ultralytics | Varifocal Loss + Class Weights | MuSGD | ROI Crop, NMS mặc định |
| **YOLO11m** | Ultralytics | BCE (default) | Auto SGD | Baseline nhanh, precision cao |
| **RT-DETR-L** | Ultralytics | Hungarian Match | AdamW | Transformer encoder-decoder |
| **D-FINE-M** | D-FINE | Fine-grained Distribution Refinement | AdamW | 2-stage training (70+30 epochs), HGNetv2 backbone |

---

## 🛠️ Các phương pháp kỹ thuật chính

1. **Cắt vùng quan tâm (ROI Cropping — `processing/apply_roi_crop.py`)**:
   - Với ảnh 5328×3000, config hiện tại dùng ROI xấp xỉ `[752, 0, 4732, 3000]`, giữ toàn bộ 9.166/1.555/1.573 box của train/val/test trong lần kiểm tra local. Ảnh 1024×1024 và kích thước chưa có cấu hình được giữ nguyên.
   - Tăng mật độ điểm ảnh hữu ích của các khuyết tật kích thước nhỏ khi đưa về 960×960.

2. **Varifocal Loss & Trọng số nghịch đảo tần suất (`losses.py`)**:
   - Thay thế BCE bằng Varifocal Loss (VFL) với continuous soft targets từ TaskAlignedAssigner.
   - Nhân trọng số nghịch đảo tần suất lớp để cân bằng gradient giữa lớp đa số (`LoiChi`: 5.282 nhãn) và thiểu số (`LoiTray`: 249 nhãn).

3. **D-FINE Fine-grained Distribution Refinement**:
   - Stage 1 (70 epochs): augmentation nhẹ với PhotometricDistort và IoUCrop.
   - Stage 2 (30 epochs): FDR refinement — tắt augmentation, tinh chỉnh phân phối hồi quy.
   - Backbone HGNetv2-B2 pretrained COCO, AdamW optimizer.

4. **Hậu xử lý NMS (YOLO26m)**:
   - Báo cáo chính dùng NMS theo từng lớp với IoU 0.7, phù hợp khi nhiều box cùng dự đoán một khuyết tật.

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
  --data data/processed/runtime_data.yaml \
  --config configs/train_yolo11m.yaml \
  --project-dir artifacts/runs
```

#### RT-DETR-L (Ultralytics)

```bash
scale-error train \
  --data data/processed/runtime_data.yaml \
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

### 6. Dự đoán qua CLI

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
