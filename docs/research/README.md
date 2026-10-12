# Nghiên cứu nhận diện lỗi thủy tinh

Pipeline ở `src/scale_error/`, độc lập với web `src/glass_inspector/`. Model/config gồm YOLO26m, YOLO11m, RT-DETR-L và D-FINE-M. FBDNet bổ sung AFTA, SFIF, DCFSB; E0/E1 đánh giá Scale-Aware NWD. AHRD có runner riêng.

- [Train/evaluate local](../TRAINING.md)
- [Benchmark lịch sử](reports/BENCHMARK_REPORT.md)
- [Cấu trúc source](../project-layout.md)

Benchmark là số liệu lịch sử đã lưu, không phải phép đo mới sau khi dọn code. Khi chạy lại, lưu output, seed, split, ROI, checkpoint hash và thiết bị thực tế. So sánh cùng protocol, kiểm tra capture-family leakage trước khi diễn giải cải thiện. Test không dùng chọn checkpoint/hyperparameter.

FBDNet/AHRD cần runtime D-FINE tùy biến tương ứng trong `results/D-FINE/`. Dataset, checkpoint, runtime do người chạy cung cấp ở local, không phân phối qua Git.
