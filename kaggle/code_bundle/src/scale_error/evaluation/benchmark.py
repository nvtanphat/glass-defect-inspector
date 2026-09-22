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
    images = list_images(image_dir)[:max_images]
    if not images:
        raise RuntimeError("No benchmark images found.")

    model = YOLO(str(weights))

    for image in images[: min(5, len(images))]:
        model.predict(str(image), imgsz=imgsz, device=0, verbose=False, nms=nms)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

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
