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
