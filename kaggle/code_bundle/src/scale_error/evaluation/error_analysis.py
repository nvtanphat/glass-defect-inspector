from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from ultralytics import YOLO

from scale_error.data.validate import label_path_for_image, list_images, parse_yolo_label_file


def _xywhn_to_xyxy(box: list[float], width: int, height: int) -> np.ndarray:
    _cls, x, y, w, h = box
    cx, cy, bw, bh = x * width, y * height, w * width, h * height
    return np.array([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], dtype=float)


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-9)


def analyze(
    weights: str | Path,
    test_images_dir: str | Path,
    test_labels_dir: str | Path,
    output_csv: str | Path,
    nc: int,
    imgsz: int = 768,
    conf: float = 0.25,
    match_iou: float = 0.5,
) -> Path:
    model = YOLO(str(weights))
    images_root = Path(test_images_dir)
    labels_root = Path(test_labels_dir)
    rows = []

    for image_path in list_images(images_root):
        label_path = label_path_for_image(image_path, images_root, labels_root)
        gt, _ = parse_yolo_label_file(label_path, nc)

        result = model.predict(
            str(image_path), imgsz=imgsz, conf=conf, device=0, verbose=False
        )[0]
        h, w = result.orig_shape
        gt_xyxy = [_xywhn_to_xyxy(box, w, h) for box in gt]
        gt_cls = [int(box[0]) for box in gt]

        pred_xyxy = (
            result.boxes.xyxy.detach().cpu().numpy()
            if result.boxes is not None
            else np.empty((0, 4))
        )
        pred_cls = (
            result.boxes.cls.detach().cpu().numpy().astype(int)
            if result.boxes is not None
            else np.empty((0,), dtype=int)
        )

        unmatched_gt = set(range(len(gt_xyxy)))
        unmatched_pred = set(range(len(pred_xyxy)))
        correct = 0
        wrong_class = 0

        candidates = []
        for gi, gbox in enumerate(gt_xyxy):
            for pi, pbox in enumerate(pred_xyxy):
                candidates.append((_iou(gbox, pbox), gi, pi))
        candidates.sort(reverse=True, key=lambda x: x[0])

        for iou, gi, pi in candidates:
            if iou < match_iou or gi not in unmatched_gt or pi not in unmatched_pred:
                continue
            unmatched_gt.remove(gi)
            unmatched_pred.remove(pi)
            if gt_cls[gi] == int(pred_cls[pi]):
                correct += 1
            else:
                wrong_class += 1

        rows.append(
            {
                "image": str(image_path),
                "gt_objects": len(gt),
                "pred_objects": len(pred_xyxy),
                "matched_correct_class": correct,
                "wrong_class_matches": wrong_class,
                "false_negatives": len(unmatched_gt),
                "false_positives": len(unmatched_pred),
            }
        )

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["image"])
        writer.writeheader()
        writer.writerows(rows)
    return output_csv
