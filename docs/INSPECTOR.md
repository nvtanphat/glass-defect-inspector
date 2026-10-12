# Vận hành Glass Defect Inspector

FastAPI phục vụ một giao diện web, PostgreSQL lưu lịch sử. `python -m glass_inspector` và `glass-inspector` khởi động cùng server ở http://127.0.0.1:8501. Chạy một worker để bảo toàn giới hạn inference/tài nguyên model.

## Python trực tiếp

Cài Python 3.11 trở lên, PostgreSQL, tạo database/user riêng và đặt biến môi trường:

```powershell
python -m pip install -e ".[app,dev]"
$env:DATABASE_URL = "postgresql+psycopg://inspector:YOUR_PASSWORD@127.0.0.1:5432/inspector"
$env:INSPECTOR_API_TOKEN = "YOUR_INTERNAL_PASSWORD"
python -m alembic upgrade head
python -m glass_inspector
```

Linux/macOS dùng `export`. Server Python đọc biến môi trường, không tự load `inspector.env`. Đổi host/port bằng `INSPECTOR_HOST`, `INSPECTOR_PORT`. `INSPECTOR_STORAGE` mặc định `var/inspections`; `INSPECTOR_ROOT` chọn root tài nguyên khi cài wheel ngoài checkout.

## Docker

Sao chép `.env.example` thành `inspector.env`, thay mật khẩu/token:

```bash
docker compose --env-file inspector.env up -d --build
docker compose --env-file inspector.env ps
```

Compose chạy migrations trước server. Image dùng inference CPU; `weights/` và `results/D-FINE/` mount read-only. Database/ảnh nằm trong volumes `postgres-data`, `inspection-images`. Giữ volumes để giữ lịch sử; không dùng `down -v` nếu cần dữ liệu.

## Model và kiểm định

Đặt checkpoint đã chọn trong `weights/`. Xem `src/glass_inspector/registry.py` để biết tên file/config cần thiết. D-FINE/FBDNet cần runtime tùy biến tương thích tại `results/D-FINE/`. Thiếu tài nguyên thì model unavailable; không thay bằng prediction giả.

Đăng nhập bằng `INSPECTOR_API_TOKEN`, chọn ảnh/model, kiểm tra, xem lịch sử và xác nhận kết quả. Trạng thái gồm Đạt, Không đạt, Cần xác nhận. Không có detection được chuyển sang Cần xác nhận. Review lưu cùng inspection; có thể xuất JSON/CSV/PDF.

Cookie đăng nhập HttpOnly, SameSite=Strict, tối đa tám giờ, ký theo token cấu hình. Đổi token làm cookie cũ mất hiệu lực. Đây là mật khẩu nội bộ dùng chung, chưa có tài khoản riêng từng người. Automation dùng bearer token. Khi đưa server ra ngoài máy local, cấu hình HTTPS/reverse proxy và quyền truy cập phù hợp.

## Kiểm tra và bảo trì

```bash
python -m alembic check
python -m pytest tests/unit tests/research -q
python scripts/verify_inspector.py --url http://127.0.0.1:8501
```

Integration cần `INSPECTOR_TEST_DATABASE_URL` tới database test riêng đã migrate. Browser tests cần `INSPECTOR_E2E_URL`, `INSPECTOR_E2E_TOKEN`, Chromium (`python -m playwright install chromium`). Không chạy tests lên database vận hành.

Backup database và ảnh cùng nhau. Ghi runtime, ROI, checkpoint hash, dependency versions khi chuyển máy. Không sửa migration đã áp dụng; thêm revision khi đổi schema. Xem [training](TRAINING.md) và [cấu trúc source](project-layout.md).
