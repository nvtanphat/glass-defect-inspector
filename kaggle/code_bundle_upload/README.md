# Glass Defect Inspector: Industrial Glass Bottle Defect Detection with YOLO26m

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLO26-00FFFF.svg)](https://docs.ultralytics.com/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Test mAP50-95](https://img.shields.io/badge/Test%20mAP50--95-40.34%25-brightgreen.svg)](#-benchmark-results-held-out-test-split)

Pipeline phát hiện lỗi miệng chai thủy tinh công nghiệp dựa trên **YOLO26m**, kết hợp tiền xử lý **ROI Cropping**, hàm mất mát **Varifocal Loss + Class Weighting**, tối ưu hóa **MuSGD**, và cơ chế suy luận **NMS-Free Dual-Head**.

---

## 📊 Benchmark Results (Held-out Test Split)

Toàn bộ mô hình được đánh giá trên cùng một tập **Test độc lập** (Held-out Test Split gồm **890 ảnh, 1.573 nhãn lỗi ground-truth**), hoàn toàn tách biệt khỏi tập huấn luyện (`train`) và tập kiểm định (`val`).

### 1. So sánh tổng thể trên tập Test

| Mô hình | Tiền xử lý | Hàm mất mát | Optimizer | Epochs | Cơ chế Post-process | Precision | Recall | mAP@50 | mAP@75 | **mAP@50:95** | Độ trễ (T4) | FPS |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **YOLO11m** (Baseline 1) | Raw (960×960) | BCE | Auto (SGD) | 102 | NMS truyền thống | **80.23%** | 58.58% | 63.86% | 32.23% | 34.31% | **57.2 ms** | **17.5** |
| **RT-DETR-L** (Baseline 2) | Raw (960×960) | Hungarian Match | AdamW | 120 | Hungarian Set Match | 73.09% | 65.47% | 65.22% | 30.57% | 35.02% | 97.3 ms | 10.3 |
| **YOLO26m Base** | Raw (960×960) | BCE | MuSGD | 66 | NMS truyền thống | 74.66% | 67.87% | 69.85% | 36.75% | 38.51% | 67.4 ms | 14.8 |
| **YOLO26m Base** | Raw (960×960) | BCE | MuSGD | 66 | NMS-Free (One-to-One) | 71.75% | 67.10% | 69.05% | 37.15% | 38.17% | 71.2 ms | 14.1 |
| **YOLO26m + ROI + VFL** | ROI Crop | Varifocal + Class Weights | MuSGD | 100 | NMS truyền thống | 72.99% | **68.22%** | **71.28%** | 37.18% | 39.37% | 59.8 ms | 16.7 |
| **YOLO26m + ROI + VFL** 🏆 | **ROI Crop** | **Varifocal + Class Weights** | **MuSGD** | **100** | **NMS-Free (One-to-One)** | 73.59% | 66.31% | 70.96% | **39.69%** | **40.34%** | 58.4 ms | 17.1 |

---

### 2. Chi tiết hiệu năng từng lớp lỗi (mAP@50:95 trên tập Test)

| Lớp lỗi | Định nghĩa lỗi | Số nhãn Test | YOLO11m | RT-DETR-L | YOLO26m Base | **YOLO26m + ROI + VFL** | Chênh lệch vs Baseline |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **DiVat** | Dị vật nhỏ | 215 | 13.18% | 16.30% | 17.53% | **20.85%** | +7.67% |
| **DiVatLoiLom** | Dị vật lồi lõm | 436 | 21.80% | 25.45% | 27.89% | **30.95%** | +9.15% |
| **LoiChi** | Nứt chỉ viền miệng | 806 | 58.93% | 59.24% | 62.38% | **62.16%** | +3.23% |
| **LoiNhua** | Lỗi ba-via nhựa | 76 | 46.94% | 40.53% | 50.01% | **53.19%** | +6.25% |
| **LoiTray** | Vết trầy xước | 40 | 30.69% | 33.57% | 34.75% | **34.56%** | +3.87% |
| **Trung bình (All Classes)** | | **1.573** | **34.31%** | **35.02%** | **38.51%** | **40.34%** | **+6.03%** |

---

### 3. Phân rã độ trễ suy luận (Tesla T4 GPU)

| Giai đoạn suy luận | YOLO11m (NMS) | RT-DETR-L | YOLO26m Base (NMS) | YOLO26m + ROI + VFL (NMS-Free) |
| :--- | :---: | :---: | :---: | :---: |
| **Pre-process** | 1.53 ms | 1.90 ms | 1.41 ms | 2.10 ms |
| **Inference (Forward)** | 44.29 ms | 96.01 ms | 49.83 ms | 56.20 ms |
| **Post-process** | 0.98 ms | 0.16 ms | 0.87 ms | **0.12 ms** |
| **Tổng độ trễ End-to-End** | 57.20 ms | 97.29 ms | 67.45 ms | 58.42 ms |
| **Tốc độ (FPS)** | 17.5 FPS | 10.3 FPS | 14.8 FPS | 17.1 FPS |

---

## 🛠️ Các phương pháp kỹ thuật chính

1. **Cắt vùng quan tâm (ROI Cropping - `processing/apply_roi_crop.py`)**:
   - Loại bỏ vùng nền đen cố định không chứa sản phẩm xung quanh miệng chai (giảm 25.3% diện tích thừa).
   - Tăng mật độ điểm ảnh hữu ích của các khuyết tật kích thước nhỏ khi đưa về kích thước đầu vào 960×960.

2. **Varifocal Loss & Trọng số nghịch đảo tần suất (`losses.py`)**:
   - Thay thế BCE bằng Varifocal Loss (VFL) với continuous soft targets lấy từ TaskAlignedAssigner.
   - Nhân trọng số nghịch đảo tần suất lớp để cân bằng độ đóng góp gradient giữa lớp đa số (`LoiChi`: 5.282 nhãn train) và thiểu số (`LoiTray`: 249 nhãn train).

3. **Giao thức tối ưu MuSGD**:
   - Tối ưu ma trận trọng số với momentum 0.948, weight decay 0.00027, learning rate ban đầu 0.00038.
   - Tắt các phép biến đổi hình học không phù hợp với ảnh camera công nghiệp cố định (shear, perspective, flipud).

4. **Suy luận NMS-Free (Dual Head)**:
   - Sử dụng One-to-One head của YOLO26 để gán nhãn 1-1 trực tiếp, loại bỏ hoàn toàn bước Non-Maximum Suppression (thời gian post-process chỉ 0.12 ms).

---

## 🚀 Hướng dẫn cài đặt và chạy Local

### 1. Cài đặt môi trường

```bash
# Clone repository
git clone https://github.com/nvtanphat/-glass-defect-inspector.git
cd -glass-defect-inspector

# Tạo môi trường ảo
python -m venv .venv

# Kích hoạt môi trường (Windows PowerShell)
.\.venv\Scripts\Activate.ps1
# Hoặc trên Linux/macOS:
# source .venv/bin/activate

# Cài đặt thư viện
pip install --upgrade pip
pip install -e ".[dev]"
```

---

### 2. Chuẩn bị dữ liệu và phân chia tập Test

Dự án sử dụng bộ dữ liệu **Scale_Error_DC v26** (Roboflow). Lệnh sau tải và chia tập stratified split để tạo tập `test` độc lập:

```bash
scale-error prepare \
  --output data/raw/scale_error_dc \
  --workspace datacpn \
  --project scale_error_dc-orqpx \
  --version 26 \
  --split-output data/processed
```

---

### 3. Thực hiện tiền xử lý ROI Crop

```bash
python processing/apply_roi_crop.py \
  --dataset-root data/processed \
  --output-root data/processed_roi \
  --roi-config processing/roi_config.json \
  --splits train val test
```

---

### 4. Huấn luyện mô hình

```bash
scale-error train \
  --data data/processed_roi/data.yaml \
  --config configs/train.yaml \
  --project-dir artifacts/runs \
  --dataset-report data/processed/dataset_report.json
```

Cấu hình huấn luyện chính (`configs/train.yaml`):
- Model: `yolo26m.pt`
- Image size: `960`
- Optimizer: `MuSGD`
- Loss: Varifocal Loss (VFL) kết hợp trọng số lớp nghịch đảo tần suất (tự động nạp từ `dataset_report.json`)
- Epochs: 100 (với early stopping 30)

---

### 5. Đánh giá trên tập Test

Chạy đánh giá checkpoint tốt nhất trên tập Test (held-out):

```bash
# Đánh giá NMS-Free (One-to-One Head)
scale-error evaluate \
  --weights artifacts/runs/yolo26m/weights/best.pt \
  --data data/processed_roi/data.yaml \
  --output artifacts/evaluation/metrics_nms_free.json \
  --nms-free

# Đánh giá NMS truyền thống
scale-error evaluate \
  --weights artifacts/runs/yolo26m/weights/best.pt \
  --data data/processed_roi/data.yaml \
  --output artifacts/evaluation/metrics_nms.json
```

---

### 6. Dự đoán & Web Demo

```bash
# Giao diện trực quan Gradio
python demo/app.py
```
Mở trình duyệt tại `http://localhost:7860`.

```bash
# Dự đoán qua CLI
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
│   ├── train.yaml                   # Cấu hình huấn luyện YOLO26m (MuSGD)
│   └── train_rtdetr.yaml            # Cấu hình huấn luyện RT-DETR-L
├── demo/
│   └── app.py                       # Giao diện web Gradio
├── docs/
│   └── FULL_DOCUMENTATION.md        # Tài liệu chi tiết công thức và thông số
├── processing/
│   ├── apply_roi_crop.py            # Script thực hiện cắt ROI
│   ├── compute_roi.py               # Script tính toán tọa độ ROI từ bounding box
│   ├── label_audit.py               # Kiểm tra tính toàn vẹn của nhãn
│   └── roi_config.json              # Thông số cấu hình ROI
├── report_assets/                   # Đồ thị đánh giá và ma trận nhầm lẫn
├── src/
│   └── scale_error/
│       ├── data/                    # Tải dữ liệu, chia tập và audit
│       ├── evaluation/              # Hàm đánh giá split="test" và tính metric
│       ├── inference/               # Module suy luận One-to-One / NMS
│       ├── training/                # Hàm mất mát Varifocal và trainer
│       ├── cli.py                   # Giao diện dòng lệnh CLI
│       └── config.py                # Schema cấu hình Pydantic
├── tests/                           # Kiểm thử đơn vị
├── pyproject.toml                   # Định nghĩa package và dependencies
├── report.html                      # Báo cáo kết quả trực quan
└── README.md                        # Hướng dẫn sử dụng
```

---

## 📄 License

Dự án được phân phối dưới giấy phép [MIT License](LICENSE).
