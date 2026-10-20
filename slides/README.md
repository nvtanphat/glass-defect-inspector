# GLASS DEFECT INSPECTOR — 16 SLIDE NCKH (bản cập nhật)

Website trình chiếu tỷ lệ 16:9 dựa trên `BaoCaoSanPham_MucLucDong(2).docx`, tháng 10/2026. Mã nguồn chạy trực tiếp trên trình duyệt, không cần bundler hay Node.js.

## Mở trình chiếu

1. Giữ `index.html`, `style.css`, `script.js` và thư mục `assets/` cùng cấp.
2. Mở `index.html` bằng Chrome hoặc Edge.
3. Nhấn `F` để toàn màn hình. `←`/`→` để chuyển slide; `O` để mở danh mục; `P` để in hoặc xuất PDF; trên điện thoại có thể vuốt ngang.
4. **Slide 6–8 & Slide 15:** bấm vào các ảnh kiến trúc hoặc ảnh demo hệ thống để mở **hình nguyên bản từ tài liệu/Word** ở độ phân giải cao. Bấm `Esc` để đóng. Slide 13 có chức năng chuyển và phóng to confusion matrix. Slide 15 có các tab chuyển màn hình demo (Workspace, Grad-CAM, Review, Dashboard).

## Nội dung trình chiếu

- **Slide 5 – ROI và Data Augmentation:** phân biệt cắt ROI (5328×3000 → 3980×3000), resize/letterbox (960×960), thay đổi tọa độ hộp bao, clip boxes sát biên; giữ lại 74,70% bề rộng; đối chiếu Mosaic 0,30, MixUp 0,05, Scale 0,50, lật ngang 0,50, lật dọc/xoay/shear/perspective bằng 0. Ghi rõ thiết lập huấn luyện YOLO26m (batch 8, seed 42, tối đa 100 epoch, dừng sớm patience 15) và lịch dừng augmentation ở epoch 88 của D-FINE-M. Tọa độ khởi đầu x0,y0 của ROI không được bịa ra khi tài liệu chưa xác nhận.
- **Slide 6 – Tổng quan mô hình:** dùng ảnh mẫu và hai hình trích từ Word, không dùng biểu tượng giả lập kiến trúc hay chỉ số hiệu năng.
- **Slide 7 – Kiến trúc FBDNet-M:** hiển thị Hình 2.6 trích trực tiếp từ báo cáo, làm nổi bật HGNetv2-FBD / AFTA, FBDHybridEncoder / SFIF / DCFSB và D-FINE Decoder; không có điểm benchmark.
- **Slide 8 – Adaptive Heterogeneous Ensemble:** sử dụng Hình 2.7 trích trực tiếp từ báo cáo, mô tả nhánh, định tuyến theo lớp, TTA và WBF; không có điểm benchmark.
- **Slide 9 – Knowledge Distillation:** tách các bước teacher → đề xuất/hiệu chỉnh nhãn → student, với hình ảnh ví dụ từ báo cáo; chỉ có số lượng ảnh và thao tác nhãn, **không có mAP hay latency**.
- **Slide 10–14:** tiếp tục là phần giao thức và kết quả, phân biệt E1/F1/KD1 và các lần đo lịch sử. Các giá trị được giữ đúng: FBDNet-M 42,42% (F1), Adaptive Triad 46,10% (E1), Student 41,01% (KD1).
- **Slide 15 – Hệ thống kiểm định thực tế (Glass Defect Inspector):** tích hợp 4 ảnh demo trực quan từ tài liệu hệ thống (`inspector-workspace.png`, `inspector-gradcam.png`, `inspector-review.png`, `inspector-dashboard.png`) với tab tương tác chuyển đổi và lightbox phóng to ảnh gốc; trình bày quy trình suy luận đa mô hình (YOLO26m, FBDNet, Ensemble) với crop ROI tự động, giải thích bản đồ nhiệt XAI Grad-CAM/HiResCAM và phân luồng QA trên nền tảng FastAPI + PostgreSQL + Docker.
- **Slide 16 – Kết luận & Hướng phát triển:** tổng kết bài toán, lựa chọn mô hình theo bài toán và các hướng mở rộng tiếp theo.

## In PDF

Bấm `P` → `Save as PDF` → bật `Background graphics` nếu muốn giữ màu nền. CSS print đã được kiểm thử để xuất đúng 16 trang 16:9.

## Thành phần

- `index.html` — nội dung 16 slide, biểu đồ SVG và các ảnh nguồn.
- `style.css` — bố cục, responsive, animation và chế độ in.
- `script.js` — điều hướng, toàn màn hình, overview, ảnh zoom, tooltip, demo tab controller.
- `data.json` — benchmark 14 cấu hình từ bảng 3.1; `null` nghĩa là chưa có số liệu.
- `assets/` — ảnh gốc trích từ báo cáo và bộ ảnh chụp hệ thống Glass Defect Inspector (`inspector-*.png`).

