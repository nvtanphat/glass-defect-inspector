# Cấu trúc dự án

| Thư mục | Nội dung |
|---|---|
| `src/scale_error/` | Dữ liệu, model, loss, training, evaluation và prediction |
| `src/glass_inspector/` | FastAPI, database, inference và dịch vụ kiểm định |
| `src/glass_inspector/web/` | Giao diện HTML/JavaScript/CSS duy nhất |
| `scripts/` | Entry point Python cho training, evaluation và kiểm chứng |
| `configs/`, `processing/` | Cấu hình thí nghiệm, ROI và xử lý nhãn |
| `migrations/` | Alembic migrations |
| `tests/` | Unit, integration, browser và nghiên cứu |
| `docs/` | Hướng dẫn, hình minh họa và benchmark lịch sử |

`python -m glass_inspector` và `glass-inspector` khởi động cùng ứng dụng. Đã bỏ launcher Streamlit và demo dư. Build CSS bằng `npm ci` rồi `npm run build` trong `src/glass_inspector/web/`. CSS build nằm trong Git; `node_modules/` thì không.

`data/`, `weights/`, `results/`, `runs/`, `artifacts/`, `reports/`, `scratch/`, `var/` là tài nguyên local/output. Không xóa dataset, checkpoint hoặc database để dọn source. Runtime D-FINE tùy biến đặt ở `results/D-FINE/`, dùng bởi training và inference.

Không commit credentials, `.env`, checkpoint, công cụ vận hành tài khoản bên ngoài hoặc evidence riêng. Không thay kiến trúc/loss hay sửa số liệu benchmark để làm sạch tài liệu.
