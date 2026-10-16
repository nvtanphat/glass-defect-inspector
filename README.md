# Glass Defect Inspector

Dự án nhận diện năm loại lỗi trên chai thủy tinh: pipeline train/evaluate local bằng Python và một web FastAPI + PostgreSQL để kiểm tra ảnh, lưu lịch sử, xác nhận kết quả và xuất báo cáo.

## Hiệu năng & Benchmark SOTA

Kết quả đánh giá thực nghiệm trên tập kiểm thử độc lập chính thức (**890 ảnh, 1.573 khuyết tật**) với chuẩn hóa ROI và đánh giá bằng `COCOeval` (NVIDIA Tesla T4 GPU FP16):

| Mô hình / Kiến trúc | mAP@50 (%) | mAP@50:95 (%) | Độ trễ (ms) | Tốc độ (FPS) | Đặc tính triển khai |
|:---|:---:|:---:|:---:|:---:|:---|
| 👑 **Unified Heterogeneous Triad** | **76,43** | **44,45** | 293,5 ms | ~3,4 FPS | KCS chuyên sâu, kỷ lục SOTA tuyệt đối (đạt đỉnh cả 5/5 lớp) |
| **Dual-Model WBF Ensemble** | 75,68 | 43,71 | 76,3 ms | ~13,1 FPS | Cân bằng tối ưu giữa độ chính xác vượt trội và tốc độ xử lý |
| **FBDNet-M (Single)** | 72,47 | 41,81 | 57,3 ms | ~17,5 FPS | Mô hình đơn lẻ cân bằng, phát hiện tốt cạnh biên khuyết tật |
| **YOLO26m-ROI-VFL (Single)** | 69,13 | 40,09 | 20,7 ms | **~48,3 FPS** | Dây chuyền thời gian thực (Real-Time), xử lý trực tiếp trên chuyền |

👉 Xem phân tích chi tiết từng lớp lỗi, trần lý thuyết Oracle và bảng đối chiếu toàn bộ các mô hình tại [Báo cáo Benchmark Toàn Diện](docs/research/reports/BENCHMARK_REPORT.md).

## Cài đặt

Python 3.11 trở lên. Chạy từ root dự án:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -e ".[app,dev]"
```

Training GPU cần PyTorch/torchvision tương thích CUDA của máy. Dataset, checkpoint và runtime D-FINE tùy biến được cung cấp riêng ở local.

## Train và evaluate local

```bash
python scripts/train.py train --data data/processed/runtime_data.yaml --config configs/train_yolo26m.yaml
python scripts/train.py evaluate --data data/processed/runtime_data.yaml --weights artifacts/runs/yolo26m/weights/best.pt
python scripts/train_fbdnet.py --exp both --weights weights/dfine_m_fbdnet_best.pth
python scripts/evaluate_fbdnet_full.py --device cuda --output artifacts/evaluation/fbdnet_m
```

Chọn đường dẫn checkpoint thực tế của lần train. Runner FBDNet fine-tune từ weights có sẵn với optimizer mới, không resume trạng thái training. Xem [training local](docs/TRAINING.md) để chuẩn bị dữ liệu/runtime và chạy D-FINE/AHRD.

## Chạy web

Chỉ có một giao diện ở `src/glass_inspector/web/`, được FastAPI phục vụ. Khởi động bằng `python -m glass_inspector` hoặc `glass-inspector`.

Docker: tạo `inspector.env` từ `.env.example`, thay mật khẩu database và `INSPECTOR_API_TOKEN`:

```bash
docker compose --env-file inspector.env up -d --build
```

Mở http://127.0.0.1:8501 và đăng nhập bằng `INSPECTOR_API_TOKEN`. Docker tự chạy migrations; database và ảnh kiểm định nằm trong volumes. Image hiện dùng inference CPU.

Python trực tiếp: cấu hình `DATABASE_URL`, `INSPECTOR_API_TOKEN`, rồi chạy:

```bash
python -m alembic upgrade head
python -m glass_inspector
```

Xem [vận hành web](docs/INSPECTOR.md). Model thiếu checkpoint/runtime được đánh dấu unavailable. Không có detection được chuyển sang **Cần xác nhận**, không tự kết luận đạt.

## Source và tài liệu

- `src/scale_error/`: dữ liệu, model, loss, training và evaluation.
- `src/glass_inspector/`: web, dịch vụ kiểm định, database và inference.
- `scripts/`, `configs/`, `processing/`: lệnh `.py`, cấu hình và ROI.
- `tests/`: unit, research, integration và browser tests.
- [Cấu trúc dự án](docs/project-layout.md), [training](docs/TRAINING.md), [nghiên cứu](docs/research/README.md).

Dataset, weights, credentials, runtime ngoài và output không nằm trong Git. Benchmark trong tài liệu nghiên cứu là số liệu lịch sử, không phải phép đo mới sau khi dọn dự án.

## Kiểm tra

```bash
python -m ruff check src tests scripts migrations
python -m pytest tests/unit tests/research -q
```

Integration cần PostgreSQL đã migrate và `INSPECTOR_TEST_DATABASE_URL`. Browser tests cần server, `INSPECTOR_E2E_URL`, `INSPECTOR_E2E_TOKEN` và Chromium. Training smoke chỉ được thu thập khi đặt `RUN_TRAINING_TESTS=1`.
