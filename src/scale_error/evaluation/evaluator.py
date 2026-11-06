"""Mô-đun đánh giá mô hình Ultralytics (YOLO26m / YOLO11m / RT-DETR-L).

Thực hiện kiểm định mô hình trên tập test và trích xuất các chỉ số COCO mAP (mAP50, mAP75, mAP50:95)
cùng tốc độ tiền xử lý, suy luận và hậu xử lý NMS.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ultralytics import RTDETR, YOLO


def _jsonable(value: Any):
    """Chuyển đổi các cấu trúc dữ liệu PyTorch/NumPy sang định dạng tuần tự hóa JSON."""
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return str(value)


def evaluate(
    weights: str | Path,
    data_yaml: str | Path,
    output_json: str | Path,
    imgsz: int = 960,
    device: int | str = 0,
    nms: bool = True,
) -> dict:
    """Đánh giá mô hình trên tập dữ liệu kiểm thử (split='test').

    Tham số:
        weights: Đường dẫn file trọng số (.pt).
        data_yaml: Đường dẫn file cấu hình data.yaml.
        output_json: Đường dẫn file JSON xuất kết quả.
        imgsz: Kích thước ảnh đánh giá (mặc định: 960x960).
        device: Chỉ số thiết bị GPU hoặc 'cpu'.
        nms: Có kích hoạt NMS hay không.
    """
    weights = Path(weights)
    model_cls = (
        RTDETR
        if "rtdetr" in weights.stem.lower() or "rtdetr" in str(weights.parent).lower()
        else YOLO
    )
    model = model_cls(str(weights))
    metrics = model.val(
        data=str(Path(data_yaml).resolve()),
        split="test",
        imgsz=imgsz,
        device=device,
        plots=True,
        nms=nms,
    )
    payload = {
        "results_dict": _jsonable(getattr(metrics, "results_dict", {})),
        "map50_95": float(metrics.box.map),
        "map50": float(metrics.box.map50),
        "map75": float(metrics.box.map75),
        "per_class_map50_95": _jsonable(metrics.box.maps),
        "speed": _jsonable(getattr(metrics, "speed", {})),
        "nms": nms,
    }
    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
