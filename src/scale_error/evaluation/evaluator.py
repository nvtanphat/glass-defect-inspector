from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ultralytics import RTDETR, YOLO


def _jsonable(value: Any):
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
