# Nghiên cứu nhận diện lỗi thủy tinh

Pipeline nghiên cứu được tổ chức tại `src/scale_error/`, độc lập hoàn toàn với ứng dụng web kiểm định `src/glass_inspector/`. Hệ thống bao gồm các kiến trúc: YOLO26m, YOLO11m, RT-DETR-L, D-FINE-M, và FBDNet (tích hợp các mô-đun AFTA, SFIF, DCFSB cùng Scale-Aware NWD / AHRD).

- [Hướng dẫn Huấn luyện & Đánh giá (Training Guide)](../TRAINING.md)
- [Báo cáo Đánh giá Hiệu năng Hệ thống (Benchmark Report)](reports/BENCHMARK_REPORT.md)
- [Cấu trúc Thư mục Dự án](../project-layout.md)
- [Tài liệu Kiến trúc Kỹ thuật Toàn diện](../FULL_DOCUMENTATION.md)

## Nguyên tắc Đánh giá Thực nghiệm (Evaluation Protocol)
1. **Quy chuẩn Độc lập Dữ liệu (Zero Data Leakage):**
   - Các siêu tham số (ngưỡng tin cậy, ngưỡng IoU, trọng số kết hợp, quy tắc định tuyến hình thái khuyết tật) được tinh chỉnh trên tập **Validation Split (878 ảnh)**.
   - Các siêu tham số tối ưu sau đó được giữ cố định (frozen) và đánh giá trên tập kiểm thử độc lập **Held-out Test Split (890 ảnh, 1.573 ground-truth bounding box)**.
   - Tập Test được giữ độc lập và không tham gia vào quá trình tìm kiếm hoặc lựa chọn siêu tham số/checkpoint.

2. **Chuẩn hóa Đầu vào & Thước đo:**
   - Tiền xử lý chuẩn hóa crop vùng ảnh hữu ích ROI ($5328 \times 3000 \to 3980 \times 3000$) nhằm loại bỏ phần cơ khí ngoài biên, bảo toàn tỷ lệ khung hình.
   - Đánh giá bằng `pycocotools.cocoeval.COCOeval` chuẩn COCO format trên phần cứng chuẩn NVIDIA Tesla T4 GPU (FP16).

3. **Kết quả Thực nghiệm Chính:**
   - **Mô hình Đề xuất (Heterogeneous Triad WBF):** Đạt **46.17% mAP@50:95** và **77.64% mAP@50**, cho thấy mức độ cải thiện so với các phương pháp dung hợp đối chứng gồm SBF (Han et al., 2025: **44.30%**), AR-WBF (Daoud & Alqasir, 2026: **44.19%**), và Baseline Vanilla WBF (Solovyev et al., 2021: **43.85%**).

4. **Yêu cầu Môi trường Runtime:**
   - FBDNet và AHRD sử dụng runtime tương ứng trong `results/D-FINE/`. Dataset, checkpoint trọng số và runtime được quản lý tại local/Kaggle, hỗ trợ tái lập thực nghiệm.
