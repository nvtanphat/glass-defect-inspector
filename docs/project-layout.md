# 🏛️ Cấu Trúc Bố Cục Dự Án Chuẩn Chỉ (Standard Project Layout)

Tài liệu này quy định cấu trúc tổ chức mã nguồn, tài nguyên dữ liệu, trọng số và phân tách trách nhiệm giữa các mô-đun trong dự án `scale-error-yolo26`.

---

## 1. Phân Tách Trách Nhiệm Thư Mục

| Thư mục | Trách nhiệm & Nội dung | Quy ước vận hành |
| :--- | :--- | :--- |
| `src/scale_error/` | **Nghiên cứu & Học máy cốt lõi:** Dữ liệu, kiến trúc mô hình (CNN + DETR Transformer), hàm mất mát, chưng cất tri thức (`distillation/`), dung hợp dị thể (`ensemble/`), huấn luyện và đánh giá. | Độc lập hoàn toàn với web backend; không phụ thuộc vào FastAPI hay PostgreSQL. |
| `src/glass_inspector/` | **Ứng dụng kiểm định công nghiệp (Production):** Máy chủ FastAPI, cơ sở dữ liệu SQLAlchemy/PostgreSQL, giao diện web điều khiển, công cụ suy luận biên và bản đồ giải thích vùng chú ý (Grad-CAM). | Khởi chạy qua lệnh `python -m glass_inspector` hoặc Docker Compose. |
| `src/glass_inspector/web/` | **Giao diện người dùng duy nhất:** Toàn bộ HTML5, CSS và JavaScript thuần phục vụ công nhân kiểm định chất lượng tại trạm QC. | Không dùng Streamlit; CSS build bằng Tailwind và lưu trữ trực tiếp trong repo. |
| `scripts/` | **Điểm thực thi công cụ (CLI Entrypoints):** Các script Python độc lập phục vụ kiểm định WBF định tuyến theo hình thái (`eval_morphology_wbf.py`), phân tích cắt bỏ ablation & bootstrap (`run_comprehensive_ablation_and_analysis.py`), phân tích 3 trụ cột kỹ thuật & tình huống mẫu (`run_p3_analysis.py`), chạy dung hợp WBF (`run_wbf_ensemble.py`), đánh giá mô hình (`evaluate_fbdnet_full.py`), trích xuất active learning (`export_active_learning_dataset.py`). | Không chứa logic ứng dụng web; gọi trực tiếp các mô-đun từ `src/`. |
| `configs/` | **Cấu hình thí nghiệm (YAML):** Cấu hình huấn luyện cho YOLO, D-FINE, RT-DETR và các biến thể thí nghiệm tại `configs/experiments/`. | Sử dụng định dạng YAML tường minh; không hardcode siêu tham số trong code. |
| `weights/` | **Trọng số tốt nhất (Model Checkpoints):** Nơi lưu trữ các tệp trọng số chính thức phục vụ suy luận và kiểm định. | **Tuân thủ quy chuẩn không hậu tố phiên bản (`v1`, `v2`,...).** Tên tệp phải phản ánh đúng kiến trúc và phương pháp. |
| `results/` | **Báo cáo chỉ số & Đánh giá chính thức:** Lưu trữ các tệp JSON tổng hợp kết quả benchmark (`ensemble_benchmark_report.json`), ma trận nhầm lẫn, trọng số siêu tham số tối ưu và các log đánh giá chính thức. | Không lưu trữ các tệp nháp tạm thời hay tệp log trung gian. |
| `reports/` | **Biên bản kiểm định chi tiết theo ngày:** Lưu trữ báo cáo kiểm định chuyên sâu, ảnh so sánh lỗi và phân tích trường hợp ngoại lệ. | Lưu trữ theo thư mục ngày tháng rõ ràng. |
| `tests/` | **Bộ kiểm thử tự động:** Unit tests (`tests/unit/`), contract tests, regression tests và browser E2E tests. | Phải đảm bảo 100% test pass trước khi triển khai mô hình. |
| `docs/` | **Tài liệu dự án & Minh họa:** Hướng dẫn kỹ thuật (`FULL_DOCUMENTATION.md`, `TRAINING.md`, `INSPECTOR.md`), hình ảnh ma trận nhầm lẫn (`docs/assets/`) và báo cáo nghiên cứu. | Cập nhật đồng bộ khi có thay đổi kiến trúc hoặc dữ liệu. |
| `slides/` | **Trình chiếu báo cáo NCKH:** Slide web tỷ lệ 16:9 chạy trực tiếp trên trình duyệt bằng HTML5/CSS3/JS thuần, tích hợp hình trích từ báo cáo và lightbox demo hệ thống. | Tách biệt với tài liệu Markdown trong `docs/`. |
| `kaggle/` | **Môi trường & Công cụ Kaggle:** Các script phục vụ đẩy kernel, đồng bộ checkpoint và giám sát huấn luyện trên nền tảng Kaggle GPU. | Chứa tài khoản tại `.kaggle_accounts.json` (được bảo vệ bởi `.gitignore`). |
| `migrations/` | **Alembic Database Migrations:** Quản lý lịch sử nâng cấp schema cơ sở dữ liệu PostgreSQL cho dịch vụ kiểm định. | Không chỉnh sửa các migration đã áp dụng vào production. |

---

## 2. Quy Chuẩn Đặt Tên Tệp & Chống Phân Mảnh Phiên Bản (Zero-Versioning Rule)

Nhằm đảm bảo tính chuyên nghiệp và tránh nhầm lẫn trong môi trường sản xuất:
1. **Tuyệt đối không sử dụng hậu tố phiên bản tùy tiện (`v1`, `v2`, `_v5`, `_v8_backup`):**
   * *Sai:* `distilled_student_v1.pt`, `fbdnet_v2.pth`, `report_v5.json`.
   * *Đúng:* `distilled_student_yolo26m_best.pt`, `dfine_m_fbdnet_nwd_cbm_best.pth`, `ensemble_benchmark_report.json`.
2. **Tên tệp phải thể hiện bản chất kỹ thuật (Semantic Naming):**
   * Nếu mô hình tích hợp thêm tính năng mới, tên phải thể hiện tính năng đó (ví dụ: `nwd_cbm` thay vì `v2`).
3. **Thư mục gốc (Root) phải luôn tinh gọn:**
   * Không lưu tệp checkpoint (`.pt`, `.pth`), tệp kết quả (`.json`, `.csv`) hay thư mục rỗng ngoài thư mục gốc.

---

## 3. Quản Trị Môi Trường & Bảo Mật

* **Không bao giờ commit thông tin nhạy cảm:** Các tệp khóa API (`.env`, `kaggle/.kaggle_accounts.json`), thông tin đăng nhập database tuyệt đối không được đưa vào Git index.
* **Tài nguyên dung lượng lớn:** Trọng số trong `weights/` và dữ liệu trong `data/` được quản lý riêng biệt, không commit trực tiếp vào Git.
