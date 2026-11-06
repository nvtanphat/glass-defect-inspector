"""Gói thư viện scale_error: Hệ thống phát hiện khuyết tật bề mặt kính đa tỉ lệ.

Bao gồm các mô-đun:
- data: Tiền xử lý, tải dữ liệu Roboflow, kiểm định và phân tầng đa nhãn.
- models: Kiến trúc FBDNet-M (AFTA, DCFSB, SFIF, HGNetv2-FBD) và AHRD cho RF-DETR.
- training: Bộ huấn luyện YOLO và D-FINE-M, hàm mất mát NWD và Varifocal Loss.
- distillation: Bộ tạo nhãn đồng thuận và chưng cất tri thức Student-Teacher.
- ensemble: Thuật toán Weighted Boxes Fusion (WBF) định tuyến thích ứng hình thái.
- evaluation: Đánh giá COCO mAP, ma trận nhầm lẫn và phân tích lỗi.
- inference: Bộ suy luận trực tiếp trên ảnh và luồng video.
"""

__version__ = "0.1.0"
