"""Mô-đun dự đoán suy luận trực tiếp với mô hình YOLO."""

from __future__ import annotations

from pathlib import Path

from ultralytics import YOLO


def predict(
    weights: str | Path,
    source: str | Path,
    project: str | Path = "artifacts/predictions",
    imgsz: int = 960,
    conf: float = 0.25,
    nms: bool = True,
):
    """Thực hiện suy luận dự đoán khuyết tật trên ảnh hoặc thư mục nguồn và lưu ảnh kết quả.

    Tham số:
        weights: Đường dẫn file trọng số mô hình (.pt).
        source: Đường dẫn file ảnh nguồn, thư mục ảnh hoặc video.
        project: Thư mục lưu kết quả xuất ra.
        imgsz: Kích thước ảnh suy luận (mặc định: 960).
        conf: Ngưỡng điểm tin cậy phát hiện (mặc định: 0.25).
        nms: Có áp dụng triệt tiêu phi cực đại NMS hay không.
    """
    model = YOLO(str(weights))
    return model.predict(
        source=str(source),
        imgsz=imgsz,
        conf=conf,
        nms=nms,
        save=True,
        project=str(project),
        name="predict",
        exist_ok=True,
    )
