# 📖 Báo Cáo Kỹ Thuật & Tài Liệu Toàn Diện Dự Án (Full Technical Documentation)
# Dự Án: Kiểm Định Khuyết Tật Bề Mặt Kính Đa Tỷ Lệ (Scale-Error Glass Defect Inspection)

---

## 1. Giới Thiệu & Bài Toán Công Nghiệp

Dự án nghiên cứu và phát triển giải pháp thị giác máy tính công nghiệp nhằm phát hiện và định vị chính xác **năm loại khuyết tật bề mặt kính/thủy tinh** trong dây chuyền sản xuất tự động tốc độ cao.

### Thách thức cốt lõi: Hiện tượng Lỗi Tỷ Lệ (Scale Error)
1. **Độ phân giải cực lớn vs Khuyết tật siêu nhỏ:** Ảnh chụp công nghiệp có kích thước gốc $5328 \times 3000$ pixel, trong khi dị vật vi mô (`DiVat`) chỉ chiếm diện tích $< 16 \times 16$ pixel (dưới $0.002\%$ diện tích toàn ảnh).
2. **Hình thái học biến thiên cực đoan:** Vết trầy xước (`LoiTray`) có tỷ lệ khung hình mảnh và dài ($H/W \gg 10$), trong khi bọt khí và vết nhựa loang (`LoiNhua`) có độ tương phản quang học rất thấp trên nền phôi kính trong suốt.
3. **Mâu thuẫn bài toán triển khai:** Yêu cầu đồng thời độ chính xác cực cao ($> 45\%$ mAP@50:95) để tránh lọt lỗi, nhưng phải đáp ứng thời gian thực dây chuyền ($> 45$ FPS) trên thiết bị biên.

---

## 2. Định Nghĩa Năm Phân Lớp Khuyết Tật (Defect Taxonomy)

| STT | Mã lớp | Tên khuyết tật | Đặc điểm hình thái học | Mức độ nguy hại |
| :---: | :--- | :--- | :--- | :--- |
| **0** | `DiVat` | Dị vật vi mô | Hạt bụi kim loại, xỉ than, kích thước siêu nhỏ ($< 16 \times 16$ px). | Làm giảm độ tinh khiết quang học. |
| **1** | `DiVatLoiLom` | Dị vật lồi lõm | Bọt khí biến dạng, phồng rộp dạng thấu kính cong trong lòng kính. | Nguy cơ nứt vỡ khi chịu ứng suất cơ học. |
| **2** | `LoiChi` | Lỗi sợi chỉ | Bọt khí kéo sợi dài theo phương dốc kéo kính lúc nóng chảy. | Lỗi khuyết tật cấu trúc nghiêm trọng. |
| **3** | `LoiNhua` | Vết nhựa loang | Vệt keo, dầu bôi trơn, vết polymer loang mờ, độ tương phản cực thấp. | Lỗi thẩm mỹ bề mặt và lớp phủ. |
| **4** | `LoiTray` | Vết trầy xước | Đường xước mảnh, sắc nhọn, tỷ lệ chiều dài/rộng cực lớn. | Gây khúc xạ ánh sáng và điểm tập trung ứng suất. |

---

## 3. Kiến Trúc Hệ Thống Hai Luồng Độc Lập (Dual-Pipeline Architecture)

Hệ thống được thiết kế theo mô hình hai nhánh chuyên biệt hóa theo kịch bản sử dụng:

```mermaid
flowchart TD
    Raw["Ảnh kính công nghiệp 5328x3000"] --> ROI["Tiền xử lý bảo toàn tỷ lệ ROI (3980x3000)"]
    ROI --> Router{"Lựa chọn chế độ kiểm định"}

    subgraph Conveyor_Pipeline ["Luồng 1: Dây Chuyền Thời Gian Thực (Real-Time Conveyor)"]
        Router -->|Tốc độ cao| Student["Distilled Student YOLO26m"]
        Student --> RealTimeOut["Phát hiện trực tiếp (<21ms · ~48.3 FPS · 41.01% mAP)"]
    end

    subgraph Inspection_Pipeline ["Luồng 2: Kiểm Định Chất Lượng Chuyên Sâu (Apex Verification)"]
        Router -->|Độ chính xác tuyệt đối| FBD["FBDNet-M (Boundary Features)"]
        Router -->|Độ chính xác tuyệt đối| DF["D-FINE-M (Coordinate FDR)"]
        Router -->|Độ chính xác tuyệt đối| Y26["YOLO26m (High-Res 1280 TTA)"]
        FBD & DF & Y26 --> WBF["Adaptive Weighted Boxes Fusion (WBF)"]
        WBF --> ApexOut["Kết quả kiểm định tối ưu (46.10% mAP · SOTA)"]
    end
```

### Chi tiết 2 luồng:
1. **Luồng Real-Time Conveyor:** Ứng dụng mô hình học sinh chưng cất tri thức (`Distilled Student YOLO26m`) với tốc độ **~48.3 FPS (20.7 ms)**, phát hiện tức thì sản phẩm lỗi ngay trên băng tải phôi đang chuyển động.
2. **Luồng Apex Verification:** Ứng dụng tổ hợp dị thể 3 nhánh (`Adaptive Heterogeneous Triad`), kết hợp thế mạnh của CNN Anchor-free và DETR Transformer qua giải thuật dung hợp hộp bao thích ứng (Adaptive WBF), đạt độ chính xác đỉnh cao **46.10% mAP@50:95**.

---

## 4. Bảng Xếp Hạng Hiệu Năng Toàn Diện (Comprehensive Benchmark Leaderboard)

*Đánh giá trên tập kiểm thử độc lập chính thức (Held-out Test Split) gồm 890 ảnh và 1.573 ground-truth bounding box sau khi áp dụng ROI ($5328 \times 3000 \to 3980 \times 3000$), tính toán bằng chuẩn `COCOeval` (NVIDIA Tesla T4 GPU, FP16, Zero Data Leakage):*

| Phân nhóm kiến trúc | Mô hình / Kỹ thuật | mAP@50 (%) | mAP@75 (%) | mAP@50:95 (%) | mAP_small (%) | Độ trễ (ms) | Tốc độ (FPS) | Vai trò triển khai |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Tổ hợp Dị thể (Ensemble)** | 🏆 **Adaptive Heterogeneous Triad** | **77.65** | **46.79** | **46.10** | **34.19** | 291.5 ms | ~3.4 FPS | **SOTA Đỉnh Cao** (Hội tụ 66.7% trần Oracle) |
| | 🥈 **Heterogeneous Morphology Triad** | 76.43 | 43.93 | 44.45 | 34.39 | 293.5 ms | ~3.4 FPS | Định tuyến 3 nhánh theo hình thái khuyết tật |
| | 🥉 **Dual-Model WBF Ensemble** | 75.68 | 43.49 | 43.71 | 32.43 | 76.3 ms | ~13.1 FPS | Cân bằng lý tưởng độ chính xác và thông lượng |
| | **Uniform Triad + TTA** | 75.48 | 42.53 | 43.12 | 34.96 | 288.0 ms | ~3.5 FPS | WBF trọng số đồng nhất đa góc nhìn |
| **DETR Transformer** | 🔬 **FBDNet-M (Single SOTA)** | 75.43 | 41.90 | **42.42** | 28.86 | 57.3 ms | ~17.5 FPS | Single Model mạnh nhất, trích xuất biên sắc |
| | 📐 **RF-DETR-M + AHRD** | 73.59 | 40.85 | 42.40 | 31.85 | 58.5 ms | ~17.1 FPS | Transformer với High-Resolution Adaptor |
| | 📐 **D-FINE-M Baseline** | 72.25 | 40.59 | 41.34 | **33.84** | 58.0 ms | ~17.2 FPS | Tối ưu hóa phân phối tọa độ khuyết tật nhỏ |
| | **RT-DETR-L** | 65.22 | 30.57 | 35.02 | 22.14 | 97.3 ms | ~10.3 FPS | Real-time DETR backbone lớn (HGNetv2-B4) |
| **Dòng CNN / YOLO** | ⚡ **Distilled Student YOLO26m** | **70.10** | **42.08** | **41.01** | **28.79** | **20.7 ms** | **~48.3 FPS** | **SOTA Real-Time:** Chưng cất từ Triad Teacher (+0.92%) |
| | 🚀 **YOLO26m-ROI-VFL Baseline** | 69.13 | 39.46 | 40.09 | 25.29 | **20.7 ms** | **~48.3 FPS** | Anchor-free NMS-free + Varifocal Loss + ROI |
| | 🔍 **YOLOv8m-P2** | 69.93 | 39.12 | 40.04 | 27.50 | 26.0 ms | ~38.5 FPS | Head P2 (Stride-4) chuyên khuyết tật vi mô |
| | 📦 **YOLOv10m** | 68.95 | 39.30 | 38.82 | 24.15 | 23.8 ms | ~42.0 FPS | NMS-free Dual Label Assignment |
| | 🎯 **YOLO26m Base (Raw 960)** | 69.85 | 36.75 | 38.51 | 22.80 | 67.4 ms | ~14.8 FPS | Khổ ảnh 960x960 không áp dụng crop ROI |
| | 🏷️ **YOLO11m** | 63.86 | 32.23 | 34.31 | 20.45 | 57.2 ms | ~17.5 FPS | Baseline tiêu chuẩn YOLO11 CSPDarknet |
| **Trần Lý Thuyết** | *Theoretical Oracle Ceiling* | *98.22* | *80.87* | *69.13* | *62.30* | — | — | *Trần biểu diễn lý thuyết của không gian đặc trưng* |

---

## 5. Quy Chuẩn Đặt Tên Trọng Số Mô Hình (Weights Directory)

Toàn bộ checkpoint trong thư mục `weights/` tuân thủ nguyên tắc **tên định danh kỹ thuật thuần túy, không sử dụng hậu tố phiên bản (`v1`, `v2`,...)**:

| Đường dẫn tệp | Kích thước | Mô tả kỹ thuật |
| :--- | :---: | :--- |
| `weights/distilled_student_yolo26m_best.pt` | 44.1 MB | Trọng số học sinh YOLO26m sau chưng cất tri thức (+0.92% mAP). |
| `weights/dfine_m_fbdnet_best.pth` | 364.8 MB | FBDNet-M bản 100 epoch với Boundary-Aware Loss (42.42% mAP). |
| `weights/dfine_m_fbdnet_nwd_cbm_best.pth` | 364.8 MB | FBDNet-M tích hợp Wasserstein Distance + Category-Balanced Matching. |
| `weights/dfine_m_baseline_reproduced_best.pth` | 314.0 MB | D-FINE-M Fine-grained Distribution Refinement chuẩn. |
| `weights/yolo26m_roi_vfl_best.pt` | 131.3 MB | YOLO26m cấu hình Varifocal Loss kết hợp tiền xử lý ROI. |
| `weights/yolo26m_960_best.pt` | 44.1 MB | YOLO26m kích thước chuẩn 960px. |

---

## 6. Quy Trình Vận Hành Ứng Dụng Web & API Inspector

Ứng dụng kiểm định biên được đóng gói hoàn chỉnh bằng FastAPI, giao diện thuần HTML5/CSS/JavaScript (không phụ thuộc Streamlit) và lưu trữ lịch sử qua PostgreSQL:

### Khởi chạy môi trường:
```bash
# Cài đặt môi trường phát triển
python -m pip install -e ".[app,dev]"

# Áp dụng migration cơ sở dữ liệu
python -m alembic upgrade head

# Khởi chạy dịch vụ kiểm định
python -m glass_inspector
```

### Triển khai tự động bằng Docker:
```bash
copy .env.example inspector.env
docker compose --env-file inspector.env up -d --build
```
Dịch vụ hoạt động tại cổng `http://127.0.0.1:8501`. Người dùng đăng nhập bằng khóa bí mật `INSPECTOR_API_TOKEN`.

### Trực quan hóa giải thích mô hình (Explainable AI - XAI):
* **HiResCAM:** Bảo toàn độ phân giải không gian của gradient, hiển thị chi tiết ranh giới của các dị vật kích thước milimet.
* **Grad-CAM:** Hiển thị bản đồ nhiệt tập trung vùng chú ý của mạng nơ-ron trước khi đưa ra quyết định chấp thuận hay loại bỏ phôi kính.

---

## 7. Cấu Trúc Bố Cục Thư Mục Chuẩn Chỉ (Standard Repository Layout)

```
scale-error-yolo26/
├── configs/                     # Cấu hình huấn luyện (train_yolo26m.yaml, experiments/)
├── src/                         # Mã nguồn cốt lõi
│   ├── scale_error/             # Nghiên cứu ML (data, distillation, ensemble, models)
│   └── glass_inspector/         # Ứng dụng Web FastAPI, Database, Server, XAI
├── scripts/                     # Điểm thực thi Python chính (WBF, Evaluation, CM Generation)
├── weights/                     # Trọng số tốt nhất của từng mô hình (chuẩn hóa tên)
├── results/                     # Báo cáo JSON & chỉ số kiểm định chính thức
├── reports/                     # Báo cáo đánh giá chi tiết theo ngày
├── tests/                       # Bộ kiểm thử tự động (Unit, Contract, Browser Tests)
├── docs/                        # Tài liệu kỹ thuật, sơ đồ kiến trúc, minh họa
├── kaggle/                      # Kernel huấn luyện phân tán trên Kaggle GPU
├── Dockerfile & compose.yaml    # Hồ sơ Docker đóng gói triển khai biên
└── README.md                    # Hướng dẫn tổng quan
```

---

## 8. Hướng Dẫn Kiểm Thử & Đảm Bảo Chất Lượng (Quality Assurance)

Chạy kiểm tra định dạng và bộ kiểm thử tự động:
```bash
# Kiểm tra linting và chuẩn code
python -m ruff check src tests scripts

# Chạy toàn bộ unit test mô-đun chưng cất tri thức và phân loại
pytest tests/unit/test_distillation.py tests/unit/test_detection_confusion.py -v
```
Toàn bộ các bài kiểm thử đều đảm bảo 100% tỷ lệ pass trước khi đưa trọng số vào sản xuất.
