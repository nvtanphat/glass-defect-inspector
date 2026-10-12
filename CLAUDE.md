# Quy tắc dự án

- Nghiên cứu/training ở `src/scale_error/`; web duy nhất ở `src/glass_inspector/`.
- Lệnh `.py`: xem `docs/TRAINING.md`; chạy web bằng `python -m glass_inspector`.
- Công cụ ở `scripts/`, tài liệu ở `docs/`, output ở `artifacts/`, `runs/`, `reports/`.
- Không commit dataset, weights, credentials hoặc công cụ vận hành tài khoản bên ngoài.
- Giữ nguyên dataset/checkpoint, kiến trúc/loss khi dọn cấu trúc.
- Không tạo benchmark giả hoặc xem số liệu lịch sử là phép đo mới.
- Chạy lint, unit/research tests; integration/E2E cần PostgreSQL/server riêng.
