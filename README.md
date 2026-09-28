# Glass Defect Inspector: Phát hiện lỗi miệng chai thủy tinh công nghiệp (YOLO26m + D-FINE-M)

[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLO26-00FFFF.svg)](https://docs.ultralytics.com/)
[![D-FINE](https://img.shields.io/badge/D--FINE-ICLR%202025-orange.svg)](https://github.com/Peterande/D-FINE)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Test mAP50-95](https://img.shields.io/badge/Test%20mAP50--95-44.40%25-brightgreen.svg)](#-kết-quả-benchmark-held-out-test-split)
[![Test mAP50](https://img.shields.io/badge/Test%20mAP50-76.90%25-brightgreen.svg)](#-kết-quả-benchmark-held-out-test-split)
[![Inference Speed](https://img.shields.io/badge/Speed-17.5%20FPS%20%2857.27ms%29-orange.svg)](#3-phân-rã-độ-trễ-suy-luận-tesla-t4-gpu)

Pipeline phát hiện 5 chủng lỗi trên miệng chai thủy tinh công nghiệp: tiền xử lý **ROI Cropping**, tối ưu **MuSGD**, hàm mất mát **Varifocal Loss + trọng số lớp nghịch đảo tần suất**, suy luận **NMS-Free Dual-Head** trên **YOLO26m**, và mô hình đối chuẩn mạnh nhất **D-FINE-M** (ICLR 2025) với **Fine-grained Distribution Refinement**.

> **Kết quả mới nhất:** **D-FINE-M @ 960×960 + ROI Crop** đạt **mAP@50:95 = 44.40% · mAP@50 = 76.90% · mAP@75 = 46.00%** trên tập test held-out, vượt YOLO26m + ROI + VFL **+4.06 mAP@50:95 / +5.94 mAP@50**, với độ trễ tương đương (57.27 ms vs 58.4 ms @ T4).

---

## 📊 Kết quả Benchmark (Held-out Test Split)

Toàn bộ mô hình được đánh giá trên cùng một tập **Test độc lập** (held-out, **890 ảnh, 1.573 nhãn lỗi ground-truth**), hoàn toàn tách biệt khỏi tập huấn luyện (`train`) và tập kiểm định (`val`). Mọi số liệu dùng chung độ phân giải **960×960** trên **Tesla T4**.

### 1. So sánh tổng thể trên tập Test

| Mô hình | Backbone | Tiền xử lý | mAP@50 | mAP@75 | **mAP@50:95** | Độ trễ (T4) | FPS |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **YOLO11m** (Baseline 1) | CSPDarknet | Raw 960×960 | 63.86% | 32.23% | 34.31% | 57.2 ms | 17.5 |
| **RT-DETR-L** (Baseline 2) | HGNetv2-B4 | Raw 960×960 | 65.22% | 30.57% | 35.02% | 97.3 ms | 10.3 |
| **YOLO26m Base** | YOLO26 | Raw 960×960 | 69.85% | 36.75% | 38.51% | 67.4 ms | 14.8 |
| **YOLO26m + ROI + VFL** (NMS-Free) | YOLO26 | ROI Crop | 70.96% | 39.69% | 40.34% | 58.4 ms | 17.1 |
| **D-FINE-M** (Fine-grained Regression) 🏆 | HGNetv2-B2 | ROI Crop | **76.90%** | **46.00%** | **44.40%** | 57.27 ms | 17.5 |

*Nguồn: `artifacts/dfine_m/benchmark_summary.txt` và `artifacts/evaluation/metrics_*.json`.*

### 2. Phân rã đầy đủ: Precision / Recall / cấu hình huấn luyện

| Mô hình | Tiền xử lý | Loss | Optimizer | Epochs | Post-process | Precision | Recall | mAP@50 | mAP@75 | **mAP@50:95** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **YOLO11m** (Baseline 1) | Raw (960×960) | BCE | Auto (SGD) | 102 | NMS truyền thống | **80.23%** | 58.58% | 63.86% | 32.23% | 34.31% |
| **RT-DETR-L** (Baseline 2) | Raw (960×960) | Hungarian Match | AdamW | 120 | Hungarian Set Match | 73.09% | 65.47% | 65.22% | 30.57% | 35.02% |
| **YOLO26m Base** | Raw (960×960) | BCE | MuSGD | 66 | NMS truyền thống | 74.66% | 67.87% | 69.85% | 36.75% | 38.51% |
| **YOLO26m Base** | Raw (960×960) | BCE | MuSGD | 66 | NMS-Free (One-to-One) | 71.75% | 67.10% | 69.05% | 37.15% | 38.17% |
| **YOLO26m + ROI + VFL** | ROI Crop | Varifocal + Class Weights | MuSGD | 100 | NMS truyền thống | 72.99% | **68.22%** | **71.28%** | 37.18% | 39.37% |
| **YOLO26m + ROI + VFL** 🏆 | **ROI Crop** | **Varifocal + Class Weights** | **MuSGD** | **100** | **NMS-Free (One-to-One)** | 73.59% | 66.31% | 70.96% | **39.69%** | **40.34%** |
| **D-FINE-M** (baseline 640) | ROI Crop (640×640) | FDR + VFL + Hungarian | AdamW | 20 | Hungarian Set Match | 71.40% | 61.40% | 67.20% | 33.40% | 36.50% |
| **D-FINE-M** (SOTA 960) 🏆 | **ROI Crop (960×960)** | **FDR + VFL + Hungarian** | **AdamW** | **100** | **Hungarian Set Match** | — | — | **76.90%** | **46.00%** | **44.40%** |

> Precision/Recall của D-FINE-M @960 chưa được ghi ở tập test (chỉ xuất mAP). Các chỉ số phân loại trên tập **val** của model này: **F1 = 76.11% · Precision = 79.07% · Recall = 73.36% · IoU = 49.77%** (TP 559 / FP 148 / FN 203).

### 3. Ablation: độ phân giải đầu vào với D-FINE-M

| Cấu hình | mAP@50 | mAP@75 | mAP@50:95 | mAP small | mAP medium | mAP large | Latency | FPS |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| D-FINE-M · 640×640 · 20 ep | 67.20% | 33.40% | 36.50% | 22.1% | 43.8% | 44.0% | 30.57 ms | 32.7 |
| **D-FINE-M · 960×960 · 100 ep (2 stage)** 🏆 | **76.90%** | **46.00%** | **44.40%** | **35.3%** | **45.8%** | **55.9%** | 57.27 ms | 17.5 |

*Chuyển từ 640→960 cùng lịch huấn luyện 100 epoch (88 multi-scale + 12 FDR refinement) mang lại **+7.90 mAP@50:95**, đặc biệt cải thiện mạnh ở nhóm defect nhỏ (small: +13.2 điểm).*

### 4. Chi tiết hiệu năng từng lớp lỗi (mAP@50:95 trên tập Test)

| Lớp lỗi | Định nghĩa lỗi | Số nhãn Test | YOLO11m | RT-DETR-L | YOLO26m Base | **YOLO26m + ROI + VFL** | Chênh lệch vs Baseline |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **DiVat** | Dị vật nhỏ | 215 | 13.18% | 16.30% | 17.53% | **20.85%** | +7.67% |
| **DiVatLoiLom** | Dị vật lồi lõm | 436 | 21.80% | 25.45% | 27.89% | **30.95%** | +9.15% |
| **LoiChi** | Nứt chỉ viền miệng | 806 | 58.93% | 59.24% | 62.38% | **62.16%** | +3.23% |
| **LoiNhua** | Lỗi ba-via nhựa | 76 | 46.94% | 40.53% | 50.01% | **53.19%** | +6.25% |
| **LoiTray** | Vết trầy xước | 40 | 30.69% | 33.57% | 34.75% | **34.56%** | +3.87% |
| **Trung bình (All Classes)** | | **1.573** | **34.31%** | **35.02%** | **38.51%** | **40.34%** | **+6.03%** |

> Kết quả per-class của D-FINE-M chưa được tách riêng trong `metrics_dfine_m.json` (`per_class: {}`); sẽ bổ sung khi chạy lại eval per-class.

### 5. Phân rã độ trễ suy luận (Tesla T4 GPU)

| Giai đoạn suy luận | YOLO11m (NMS) | RT-DETR-L | D-FINE-M 640 | YOLO26m Base (NMS) | YOLO26m + ROI + VFL (NMS-Free) | **D-FINE-M 960** 🏆 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Pre-process** | 1.53 ms | 1.90 ms | 1.80 ms | 1.41 ms | 2.10 ms | — |
| **Inference (Forward)** | 44.29 ms | 96.01 ms | 30.57 ms | 49.83 ms | 56.20 ms | **57.27 ms** |
| **Post-process** | 0.98 ms | 0.16 ms | 0.15 ms | 0.87 ms | **0.12 ms** | — |
| **Tổng độ trễ End-to-End** | 57.20 ms | 97.29 ms | **32.52 ms** | 67.45 ms | 58.42 ms | **57.27 ms** |
| **Tốc độ (FPS)** | 17.5 FPS | 10.3 FPS | **32.7 FPS** | 14.8 FPS | 17.1 FPS | 17.5 FPS |

> D-FINE-M @960 hiện mới đo **thời gian forward** (`latency_forward_ms: 57.27`) và quy đổi ra 17.5 FPS; pre/post-process chưa tách riêng (Hungarian matching của D-FINE vốn ~0.15 ms ở bản 640).

---

## 🏆 Hồ sơ model thắng giải: D-FINE-M @960

| Thông số | Giá trị |
| :--- | :--- |
| Kiến trúc | D-FINE-M · Backbone HGNetv2-B2 |
| Tham số | **19.19 M** |
| FLOPs | **122.305 GFLOPS** (60.96 GMACs) |
| Input | 960 × 960 (ROI Crop `[1448, 900, 4228, 2300]`) |
| Pretrained | COCO (`dfine_m_coco.pth`) |
| Huấn luyện | 100 epoch = Stage 1: 88 epoch multi-scale + Stage 2: 12 epoch FDR refinement |
| Batch / LR | train 8 · val 16 · lr stage-2 2.5e-5 (base 2.5e-4) · AMP · seed 42 |
| Phần cứng | DDP trên **2× Tesla T4** (`torch.distributed.run`) |
| Checkpoint | `best.pth` ← `best_stg2.pth` (best epoch), `best_stg1.pth`, `eval.pth` |
| **Test** | **mAP@50:95 44.40% · mAP@50 76.90% · mAP@75 46.00%** |
| mAP theo size | small **35.3%** · medium **45.8%** · large **55.9%** |
| Epoch tốt nhất (val) | epoch **99** → val mAP@50:95 = **43.55%** (epoch 98: 43.52%) |
| Loss cuối (val/train) | total 15.81 · bbox 0.0448 · giou 0.2755 · vfl 0.3502 · fgl 0.9965 |

*Log tham chiếu: `results_report/output_dfine_m/log.txt`, `train_stdout.log` (lần resume ep88→99 mất 2h08'38, 0.855 s/it, peak VRAM 5403 MB).*

---

## 🛠️ Các phương pháp kỹ thuật chính

### Nhóm YOLO26m

1. **Cắt vùng quan tâm (ROI Cropping — `processing/apply_roi_crop.py`)**
   - Loại bỏ vùng nền đen cố định không chứa sản phẩm quanh miệng chai (tọa độ hình học `[1448, 900, 4228, 2300]`, giảm 25.3% diện tích thừa).
   - Tăng mật độ điểm ảnh hữu ích của các khuyết tật kích thước nhỏ khi đưa về kích thước đầu vào 960×960.

2. **Varifocal Loss & trọng số nghịch đảo tần suất (`src/scale_error/training/losses.py`)**
   - Thay thế BCE bằng Varifocal Loss (VFL) với continuous soft targets lấy từ TaskAlignedAssigner.
   - Nhân trọng số nghịch đảo tần suất lớp để cân bằng gradient giữa lớp đa số (`LoiChi`: 5.282 nhãn train) và thiểu số (`LoiTray`: 249 nhãn train).

3. **Giao thức tối ưu MuSGD (`configs/train.yaml`)**
   - Momentum 0.948, weight decay 0.00027, lr 0.00038, lrf 0.882, warmup 0.99 epoch.
   - Tắt các phép biến đổi hình học không phù hợp với ảnh camera công nghiệp cố định (shear, perspective, flipud); mosaic hạ xuống 0.30 để không thu nhỏ thêm box vốn đã rất nhỏ.

4. **Suy luận NMS-Free (Dual Head)**
   - One-to-One head của YOLO26 gán nhãn 1-1 trực tiếp, bỏ hoàn toàn Non-Maximum Suppression (post-process chỉ **0.12 ms**).

### Nhóm D-FINE-M

5. **Fine-grained Distribution Refinement (FDR)**
   - Thay vì regressed một điểm, D-FINE tinh chỉnh **phân bố** của tọa độ box qua các decoder layer → độ chính xác biên hộp cao hơn ở ngưỡng IoU 0.75 (mAP@75 tăng từ 39.69% → **46.00%**).

6. **Hai giai đoạn huấn luyện 100 epoch**
   - Stage 1 (88 epoch): multi-scale augmentation để model quen biến thiên kích thước defect.
   - Stage 2 (12 epoch): **FDR refinement** với LR 2.5e-5 để tinh chỉnh biên hộp trên phân phối đã hội tụ.

7. **Pipeline đồng bộ 1:1 với YOLO26m**
   - Cùng ROI Crop, cùng 960×960, cùng tập train/val/test → benchmark thực sự công bằng.

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
pip install -e ".[dev,demo]"
```

---

### 2. Chuẩn bị dữ liệu và phân chia tập Test

Dự án dùng bộ dữ liệu **Scale_Error_DC v26** (Roboflow). Lệnh sau tải và chia tập stratified split để tạo tập `test` độc lập:

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

### 4. Huấn luyện YOLO26m

```bash
scale-error train \
  --data data/processed_roi/data.yaml \
  --config configs/train.yaml \
  --project-dir artifacts/runs \
  --dataset-report data/processed/dataset_report.json
```

Cấu hình chính (`configs/train.yaml`):
- Model `yolo26m.pt` · image size `960` · optimizer `MuSGD`
- Loss: Varifocal Loss (VFL) + trọng số lớp nghịch đảo tần suất (tự nạp từ `dataset_report.json`)
- Epochs 120 (patience 15) · AMP · seed 42 · deterministic

Huấn luyện RT-DETR-L dùng `configs/train_rtdetr.yaml`.

---

### 5. Huấn luyện & đánh giá D-FINE-M (Kaggle, 2× T4)

D-FINE dùng repo chính thức [Peterande/D-FINE](https://github.com/Peterande/D-FINE), được đóng gói thành kernel script tự động (tải dữ liệu, patch config 960×960, train 2 stage, eval test, benchmark latency):

```bash
# Kernel: kaggle/kernel_dfine/run_dfine.py
# id: phtnguyn1ytj/scale-error-dfine-m-train  (GPU, private)
kaggle kernels push -r kaggle/kernel_dfine/
```

Các bước kernel tự thực hiện:
1. Clone D-FINE + tải pretrained COCO `dfine_m_coco.pth`.
2. Patch `configs/dfine/include/dataloader.yml` → `size: [960, 960]`, `stop_epoch 88`; patch `eval_spatial_size: [960, 960]`.
3. Sinh config `configs/dataset/glass_defect.yml`, `configs/dfine/custom/dfine_hgnetv2_m_glass.yml` (100 epoch, total_batch_size 8 train / 16 val).
4. Train DDP 2× T4 với `torch.distributed.run --use-amp --seed=42`, hỗ trợ `-r` resume (stage 2).
5. Đánh giá test-only trên 890 ảnh, ghi `metrics_dfine_m.json`, đo latency forward, ghi `benchmark_summary.txt`.

Lấy kết quả về máy:

```bash
kaggle kernels output phtnguyn1ytj/scale-error-dfine-m-train -p kaggle_output_dfine_m
```

---

### 6. Đánh giá trên tập Test (YOLO26m)

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

### 7. Giao diện Web Dashboard & Dự đoán

#### Giao diện Web Gradio Dashboard

```bash
python demo/app.py
```

Mở trình duyệt tại `http://localhost:7860`. Các tính năng nổi bật:
- **Thẻ KPI thời gian thực**: tổng số lỗi phát hiện, số chủng loại lỗi, độ trễ suy luận (ms).
- **Tab trực quan hóa kép**: tab bounding box gán nhãn phân loại lỗi, tab khung ranh giới hình học ROI.
- **Bảng dữ liệu khuyết tật chi tiết**: tọa độ XYXY, confidence, diện tích vùng lỗi (px²).
- **Lựa chọn chế độ đầu suy luận**: NMS-Free (post-process 0.12 ms) hoặc Standard NMS.

#### Dự đoán qua dòng lệnh (CLI)

```bash
scale-error predict \
  --weights artifacts/runs/yolo26m/weights/best.pt \
  --source path/to/images/ \
  --output artifacts/predictions \
  --nms-free
```

---

### 8. Xuất định dạng ONNX

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
scale-error-yolo26/
├── configs/
│   ├── train.yaml                   # Cấu hình huấn luyện YOLO26m (MuSGD + VFL)
│   └── train_rtdetr.yaml            # Cấu hình huấn luyện RT-DETR-L
├── demo/
│   └── app.py                       # Giao diện web Gradio Dashboard công nghiệp
├── docs/
│   └── FULL_DOCUMENTATION.md        # Tài liệu chi tiết công thức và thông số
├── processing/
│   ├── apply_roi_crop.py            # Script thực hiện cắt ROI
│   ├── compute_roi.py               # Script tính tọa độ ROI từ bounding box
│   ├── label_audit.py               # Kiểm tra tính toàn vẹn của nhãn
│   └── roi_config.json              # Thông số cấu hình ROI
├── kaggle/
│   ├── kernel_dfine/run_dfine.py    # Kernel train + eval D-FINE-M (2× T4)
│   └── pull_kernel_output.py        # Tải kết quả kernel về máy
├── report_assets/                   # Đồ thị đánh giá, ma trận nhầm lẫn, ảnh EDA
├── src/scale_error/
│   ├── data/                        # Tải dữ liệu, chia tập và audit
│   ├── evaluation/                  # Đánh giá split="test", benchmark, error analysis
│   ├── inference/                   # Module suy luận One-to-One / NMS
│   ├── training/                    # Varifocal Loss và trainer
│   ├── cli.py                       # Giao diện dòng lệnh CLI
│   └── config.py                    # Schema cấu hình Pydantic
├── artifacts/
│   ├── evaluation/                  # metrics_nms_*.json, metrics_dfine_m.json
│   ├── dfine_m/                     # benchmark_summary.txt, metrics_dfine_m.json, best.pth
│   └── runs/                        # Checkpoint huấn luyện YOLO
├── results_report/                  # Log & metrics trích từ results.zip (D-FINE-M 960)
├── tests/                           # Kiểm thử đơn vị tự động
├── report.html                      # Báo cáo kết quả trực quan
├── pyproject.toml                   # Định nghĩa package và dependencies
└── README.md                        # Hướng dẫn sử dụng và báo cáo benchmark
```

---

## 📄 License

Dự án được phân phối dưới giấy phép [MIT License](LICENSE).
