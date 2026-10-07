# Glass Defect Inspector: Industrial Glass Bottle Defect Detection with FBDNet-M & YOLO26

[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![SOTA mAP50-95](https://img.shields.io/badge/FBDNet--M%20mAP50--95-42.42%25-brightgreen.svg)](#-kết-quả-đánh-giá-held-out-test-split)
[![mAP50](https://img.shields.io/badge/FBDNet--M%20mAP50-75.43%25-blueviolet.svg)](#-kết-quả-đánh-giá-held-out-test-split)
[![Speed](https://img.shields.io/badge/Forward%20Speed-17.5%20FPS%20(57.3ms)-orange.svg)](#3-độ-trễ-suy-luận-tesla-t4-gpu)

Hệ thống thị giác máy tính công nghiệp phục vụ **phát hiện và phân loại khuyết tật miệng chai thủy tinh** với độ chính xác cao. Dự án giải quyết triệt để bài toán **vật thể siêu nhỏ (Scale Error)**, **độ tương phản cực thấp (Low Contrast)** và **mất cân bằng dữ liệu nghiêm trọng** trong dây chuyền sản xuất thực tế.

---

## 🌟 Đóng góp nổi bật (Key Highlights)

1. **Kiến trúc SOTA mới đề xuất — FBDNet-M (Frequency Boundary Diffusion Network):**
   - **AFTA (Adaptive Frequency-domain / Target Attention):** Tích hợp vào backbone HGNetv2 để trích xuất tín hiệu tần số cao, khuếch đại các vết nứt mỏng và trầy xước trong suốt.
   - **SFIF (Spatial-Frequency Interaction Fusion):** Hợp nhất tương hỗ giữa miền không gian và miền tần số tại encoder ẩn.
   - **DCFSB (Dense Cross-scale Feature Screening Block):** Ngăn chặn hiện tượng tiêu biến đặc trưng của các dị vật li ti qua các tầng đa tỷ lệ (multi-scale).
   - Đạt đỉnh **42.42% mAP@50:95** và **75.43% mAP@50**, vượt trội so với toàn bộ các baseline tiêu chuẩn.

2. **Quy trình tiền xử lý ROI thích ứng (Adaptive ROI Cropping):**
   - Tự động định vị và cắt vùng miệng chai từ ảnh nguồn độ phân giải cao (5328×3000), loại bỏ 80% phông nền thừa và gia tăng mật độ điểm ảnh hữu ích cho các khuyết tật nhỏ.

3. **Bản địa hóa 5 lớp khuyết tật công nghiệp chuẩn mực:**
   - `DiVat` (Dị vật nhỏ), `DiVatLoiLom` (Dị vật lồi lõm), `LoiChi` (Nứt chỉ viền), `LoiNhua` (Ba-via nhựa), `LoiTray` (Vết trầy xước).

---

## 📊 Kết quả đánh giá (Held-out Test Split)

> **Giao thức thử nghiệm chuẩn (Benchmark Protocol):**
> Toàn bộ mô hình được đánh giá trên cùng tập kiểm thử độc lập **Held-out Test (890 ảnh, 1.573 ground-truth bounding box)**, phân chia theo phương pháp Multilabel Stratified Split (Seed 42), đảm bảo 0% data leakage.

### 1. Bảng so sánh tổng thể các mô hình

| Mô hình | Kiến trúc Backbone & Kỹ thuật | mAP@50 | mAP@75 | **mAP@50:95** | mAP_s *(lỗi nhỏ)* | Params | Tốc độ (T4) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **FBDNet-M (Đề xuất)** 🏆 | **HGNetv2FBD + FBDHybridEncoder + ROI** | **75.43%** | **41.90%** | **42.42%** | **28.86%** | 22.4M | 17.5 FPS |
| **D-FINE-M Baseline** | HGNetv2-B2 + HybridEncoder + ROI | 72.25% | 40.59% | **41.33%** | 28.64% | 19.2M | 17.5 FPS |
| **YOLO26m + ROI + VFL** | YOLO26 + ROI Crop + Varifocal Loss | 70.96% | 39.69% | **40.34%** | — | ~21M | 17.1 FPS |
| **YOLO26m Base** | YOLO26 (Raw 960×960) | 69.85% | 36.75% | **38.51%** | — | ~21M | 14.8 FPS |
| **RT-DETR-L** | HGNetv2-B4 + Transformer | 65.22% | 30.57% | **35.02%** | — | 32.0M | 10.3 FPS |
| **YOLO11m** | CSPDarknet (Baseline tiêu chuẩn) | 63.86% | 32.23% | **34.31%** | — | 20.1M | 17.5 FPS |

---

### 2. Chi tiết hiệu năng theo từng lớp khuyết tật (mAP@50:95)

| Lớp khuyết tật | Mô tả đặc trưng | Số nhãn | YOLO11m | RT-DETR-L | YOLO26m+ROI | D-FINE-M | **FBDNet-M** 🏆 | Tăng trưởng vs Baseline |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **DiVat** | Dị vật nhỏ / li ti | 215 | 13.18% | 16.30% | 20.61% | 25.31% | **26.72%** | **+1.41%** 🟢 |
| **DiVatLoiLom** | Dị vật lồi lõm | 436 | 21.80% | 25.45% | 30.92% | 31.90% | **33.13%** | **+1.23%** 🟢 |
| **LoiChi** | Nứt chỉ viền miệng | 806 | 58.93% | 59.24% | 62.61% | **63.82%** | 63.19% | Ổn định (~90% @50) |
| **LoiNhua** | Ba-via nhựa | 76 | 46.94% | 40.53% | **52.23%** | 51.78% | 46.57% | Phân hóa đặc thù |
| **LoiTray** | Vết trầy xước | 40 | 30.69% | 33.57% | 34.21% | 29.65% | **41.04%** | **+11.39%** 🔥 |
| **Trung bình** | **mAP@50:95** | **1.573** | **34.31%** | **35.02%** | **40.34%** | **40.49%** | **42.13%** | **+1.64%** 🚀 |

> 💡 **Nhận xét chuyên sâu:**
> - **Đột phá ở lớp `LoiTray` (+11.39% mAP@50:95 và +17.89% mAP@50):** Vết trầy trên thủy tinh có độ tương phản cực kỳ thấp. Nhánh chú ý miền tần số (AFTA) đã hỗ trợ mô hình bóc tách hoàn hảo tín hiệu biên cạnh sắc nét mà các backbone thông thường bỏ sót.
> - **Cải thiện ở khuyết tật nhỏ (`DiVat` & `DiVatLoiLom`):** Tăng đều đặn +1.2% đến +1.4%, đồng thời chỉ số `mAP_s` đạt **28.86%** khẳng định năng lực định vị vật thể nhỏ của FBDNet.

---

### 3. Độ trễ suy luận (Tesla T4 GPU)

| Mô hình | Tiền xử lý | Thời gian suy luận (Latency) | Tốc độ khung hình (FPS) |
| :--- | :---: | :---: | :---: |
| **FBDNet-M / D-FINE-M** | ROI Crop (960×960) | **57.27 ms** | **17.5 FPS** |
| **YOLO26m + ROI + VFL** | ROI Crop (960×960) | 58.40 ms | 17.1 FPS |
| **YOLO11m** | Raw (960×960) | 57.20 ms | 17.5 FPS |
| **RT-DETR-L** | Raw (960×960) | 97.29 ms | 10.3 FPS |

---

## 🏗️ Cấu trúc dự án chuẩn Enterprise

Mã nguồn được cấu trúc theo tiêu chuẩn mô-đun hóa, sạch sẽ, sẵn sàng triển khai thực tế (Production-Ready):

```text
scale-error-yolo26/
├── configs/                     # Cấu hình YAML (Mô hình & Thực nghiệm)
│   ├── experiments/             # Config ablation study (p2, frequency, balanced)
│   └── train_*.yaml             # Config huấn luyện (YOLO26, D-FINE, RT-DETR)
├── data/                        # [Git-ignored] Thư mục chứa dữ liệu ảnh
├── demo/                        # Ứng dụng Web Demo tương tác & giao diện UI replica
│   └── stitch_image_ui_replica/ # Giao diện ghép ảnh công nghiệp
├── docs/                        # Tài liệu đặc tả kỹ thuật và báo cáo
│   └── reports/                 # Báo cáo kết quả đầy đủ, file Word và phân tích lỗi
├── notebooks/                   # Jupyter Notebook nghiên cứu dữ liệu (EDA)
│   └── 01_dataset_eda_scale_error_dc.ipynb
├── src/                         # Core Python Package (scale_error)
│   └── scale_error/
│       ├── core/                # Module cấu hình, logging, registry
│       ├── data/                # Data loaders, audit, ROI cropping
│       ├── models/              # Kiến trúc mô hình (FBDNet: afta, sfif, dcfsb)
│       ├── inference/           # Engine suy luận end-to-end
│       └── evaluation/          # Đánh giá metrics & phân tích sai số
├── tools/                       # CLI Scripts vận hành cho kỹ sư
│   ├── eval_dfine.py            # Đánh giá benchmark D-FINE / FBDNet
│   ├── check_benchmarks.py      # Trích xuất tự động toàn bộ metric từ log
│   └── visualize_gradcam.py     # Trực quan hóa bản đồ nhiệt Grad-CAM
├── tests/                       # Unit tests
├── weights/                     # Checkpoints chính thức (FBDNet SOTA, D-FINE, YOLO26)
├── kaggle/                      # Pipeline huấn luyện đám mây & auto-resume tự động
├── app.py                       # Giao diện Web Application chính
├── run_demo.bat                 # Khởi chạy demo 1-click
├── pyproject.toml               # Quản lý dependencies & metadata dự án
└── README.md                    # Tài liệu hướng dẫn chính
```

---

## 🚀 Hướng dẫn cài đặt & Thực thi

### 1. Cài đặt môi trường

```bash
# 1. Clone repository
git clone https://github.com/nvtanphat/-glass-defect-inspector.git
cd -glass-defect-inspector

# 2. Tạo môi trường ảo
python -m venv .venv

# Kích hoạt trên Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# Kích hoạt trên Linux / macOS:
# source .venv/bin/activate

# 3. Cài đặt dependencies
pip install --upgrade pip
pip install -e ".[dev,demo]"
```

---

### 2. Khởi chạy Web Demo (Interactive UI)

Chạy ứng dụng demo phát hiện khuyết tật trực quan trên trình duyệt:

```bash
# Cách 1: Sử dụng script 1-click trên Windows
run_demo.bat

# Cách 2: Chạy trực tiếp bằng Python
python app.py
```

---

### 3. Đánh giá và kiểm thử Benchmark (Evaluation)

Để kiểm tra lại toàn bộ số liệu benchmark của các mô hình:

```bash
# Trích xuất bảng kết quả từ tất cả các log thực nghiệm
python tools/check_benchmarks.py

# Đánh giá chi tiết mô hình D-FINE / FBDNet trên tập test
python tools/eval_dfine.py
```

---

### 4. Trực quan hóa bản đồ chú ý (Grad-CAM)

Để phân tích vùng mạng nơ-ron tập trung khi phát hiện vết nứt/trầy:

```bash
python tools/visualize_gradcam.py \
  --image demo/sample_bottle.jpg \
  --weights weights/dfine_m_fbdnet_best.pth \
  --output report_assets/gradcam_result.png
```

---

## 📑 Phân chia dữ liệu (Dataset Protocol)

* **Nguồn ảnh:** Dây chuyền chụp ảnh miệng chai thủy tinh công nghiệp độ phân giải cao.
* **Quy tắc phân chia:** Multilabel Stratified Split (Seed 42) chia đều các lớp khuyết tật hiếm:
  - **Train:** 5.578 ảnh (9.166 ground-truth boxes)
  - **Val:** 878 ảnh (1.555 ground-truth boxes)
  - **Test (Held-out):** 890 ảnh (1.573 ground-truth boxes)
* **Tỷ lệ box nhỏ:** ~85% số box chiếm diện tích nhỏ hơn 1% toàn ảnh.

---

## 📄 License

Dự án được phân phối dưới giấy phép [MIT License](LICENSE).
Mọi đóng góp học thuật và công nghiệp xin vui lòng liên hệ qua GitHub Issues hoặc Email của tác giả.
