"""Mô-đun đo kiểm hiệu năng tốc độ suy luận (Latency & FPS Benchmark).

Đo độ trễ trung bình, trung vị, phân vị 95% (P95) và tốc độ khung hình (FPS)
của mô hình trên GPU/CPU với đồng bộ hóa CUDA chính xác.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO

from scale_error.data.validate import list_images


def benchmark(
    weights: str | Path,
    image_dir: str | Path,
    output_json: str | Path,
    imgsz: int = 960,
    max_images: int = 100,
    nms: bool = True,
) -> dict:
    """Đo kiểm thời gian xử lý suy luận end-to-end trên tập ảnh thử nghiệm.

    Tham số:
        weights: Đường dẫn file trọng số mô hình (.pt).
        image_dir: Thư mục chứa ảnh benchmark.
        output_json: Đường dẫn file JSON xuất kết quả đo kiểm.
        imgsz: Kích thước ảnh đầu vào (mặc định: 960x960).
        max_images: Số lượng ảnh tối đa để chạy đo (mặc định: 100 ảnh).
        nms: Có áp dụng NMS hay không.

    Trả về:
        dict: Chứa các chỉ số mean_ms, median_ms, p95_ms, và fps.
    """
    images = list_images(image_dir)[:max_images]
    if not images:
        raise RuntimeError("Không tìm thấy ảnh benchmark trong thư mục chỉ định.")

    model = YOLO(str(weights))

    # Khởi động làm nóng GPU (Warm-up 5 ảnh đầu)
    for image in images[: min(5, len(images))]:
        model.predict(str(image), imgsz=imgsz, device=0, verbose=False, nms=nms)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Thực hiện đo đạc độ trễ từng ảnh
    times_ms = []
    for image in images:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        model.predict(str(image), imgsz=imgsz, device=0, verbose=False, nms=nms)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times_ms.append((time.perf_counter() - t0) * 1000)

    arr = np.asarray(times_ms, dtype=np.float64)
    mean_ms = float(arr.mean())
    payload = {
        "images": len(images),
        "imgsz": imgsz,
        "nms": nms,
        "mean_ms_per_image_end_to_end": mean_ms,
        "median_ms": float(np.median(arr)),
        "p95_ms": float(np.percentile(arr, 95)),
        "fps_from_mean_latency": float(1000.0 / mean_ms),
    }
    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
