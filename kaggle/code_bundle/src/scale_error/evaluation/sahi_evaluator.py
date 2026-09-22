"""SAHI (Slicing Aided Hyper Inference) evaluator for high-resolution defect detection.

Implements high-resolution slice inference, non-maximum merging (NMM), coordinate
remapping, and rigorous COCO metric evaluation against original ground-truth annotations.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torchvision.ops as ops
from ultralytics import YOLO


def calculate_slice_coords(img_w: int, img_h: int, slice_w: int = 960, slice_h: int = 960, overlap_ratio: float = 0.20) -> list[tuple[int, int, int, int]]:
    """Generate (x1, y1, x2, y2) bounds covering the entire image with overlap."""
    step_x = max(1, int(slice_w * (1.0 - overlap_ratio)))
    step_y = max(1, int(slice_h * (1.0 - overlap_ratio)))

    x_starts = []
    x = 0
    while x + slice_w <= img_w:
        x_starts.append(x)
        x += step_x
    if not x_starts or x_starts[-1] + slice_w < img_w:
        x_starts.append(max(0, img_w - slice_w))
    x_starts = sorted(list(set(x_starts)))

    y_starts = []
    y = 0
    while y + slice_h <= img_h:
        y_starts.append(y)
        y += step_y
    if not y_starts or y_starts[-1] + slice_h < img_h:
        y_starts.append(max(0, img_h - slice_h))
    y_starts = sorted(list(set(y_starts)))

    slices = []
    for sy in y_starts:
        for sx in x_starts:
            slices.append((sx, sy, min(img_w, sx + slice_w), min(img_h, sy + slice_h)))
    return slices


def box_iou(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """Compute IoU matrix between boxes1 [N, 4] and boxes2 [M, 4] in (x1, y1, x2, y2)."""
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)), dtype=np.float32)

    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = np.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = np.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[:, :, 0] * wh[:, :, 1]

    union = area1[:, None] + area2[None, :] - inter
    return inter / np.clip(union, 1e-6, None)


class SAHIPredictor:
    def __init__(
        self,
        weights: str | Path,
        slice_w: int = 960,
        slice_h: int = 960,
        overlap_ratio: float = 0.20,
        conf_thresh: float = 0.25,
        iou_thresh: float = 0.50,
        device: int | str = 0,
    ):
        self.model = YOLO(str(weights))
        self.slice_w = slice_w
        self.slice_h = slice_h
        self.overlap_ratio = overlap_ratio
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.device = device

    def predict_image(self, img: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
        """Run sliced inference and merge overlapping predictions.

        Returns:
            (predictions [N, 6]: x1, y1, x2, y2, score, class_id),
            timing_dict (ms)
        """
        img_h, img_w = img.shape[:2]

        t0 = time.perf_counter()
        slice_coords = calculate_slice_coords(img_w, img_h, self.slice_w, self.slice_h, self.overlap_ratio)
        tiles = [img[sy:ey, sx:ex] for sx, sy, ex, ey in slice_coords]
        t_slice = (time.perf_counter() - t0) * 1000.0

        t1 = time.perf_counter()
        # Batch inference over tiles
        results = self.model.predict(
            tiles,
            imgsz=self.slice_w,
            conf=self.conf_thresh,
            device=self.device,
            verbose=False,
        )
        t_infer = (time.perf_counter() - t1) * 1000.0

        t2 = time.perf_counter()
        shifted_boxes = []
        shifted_scores = []
        shifted_classes = []

        for (sx, sy, _, _), res in zip(slice_coords, results):
            if res.boxes is None or len(res.boxes) == 0:
                continue
            b = res.boxes.xyxy.cpu().numpy()
            s = res.boxes.conf.cpu().numpy()
            c = res.boxes.cls.cpu().numpy()

            # Shift x, y by slice start offset
            b[:, [0, 2]] += sx
            b[:, [1, 3]] += sy

            shifted_boxes.append(b)
            shifted_scores.append(s)
            shifted_classes.append(c)

        if not shifted_boxes:
            t_merge = (time.perf_counter() - t2) * 1000.0
            return np.empty((0, 6), dtype=np.float32), {
                "slice_ms": t_slice,
                "inference_ms": t_infer,
                "merge_ms": t_merge,
                "total_e2e_ms": t_slice + t_infer + t_merge,
            }

        all_boxes = np.concatenate(shifted_boxes, axis=0)
        all_scores = np.concatenate(shifted_scores, axis=0)
        all_classes = np.concatenate(shifted_classes, axis=0)

        # Non-Maximum Merging across slices
        t_boxes = torch.from_numpy(all_boxes).float()
        t_scores = torch.from_numpy(all_scores).float()
        t_cls = torch.from_numpy(all_classes).long()

        keep_idx = ops.batched_nms(t_boxes, t_scores, t_cls, self.iou_thresh).numpy()
        merged = np.hstack([
            all_boxes[keep_idx],
            all_scores[keep_idx, None],
            all_classes[keep_idx, None],
        ])
        t_merge = (time.perf_counter() - t2) * 1000.0

        return merged, {
            "slice_ms": t_slice,
            "inference_ms": t_infer,
            "merge_ms": t_merge,
            "total_e2e_ms": t_slice + t_infer + t_merge,
        }


def evaluate_sahi(
    weights: str | Path,
    test_images_dir: str | Path,
    test_labels_dir: str | Path,
    output_json: str | Path,
    slice_w: int = 960,
    slice_h: int = 960,
    overlap_ratio: float = 0.20,
    conf_thresh: float = 0.25,
    iou_thresh: float = 0.50,
    device: int | str = 0,
    num_classes: int = 5,
    class_names: dict[int, str] | None = None,
) -> dict[str, Any]:
    test_images_dir = Path(test_images_dir)
    test_labels_dir = Path(test_labels_dir)
    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    if class_names is None:
        class_names = {0: "DiVat", 1: "DiVatLoiLom", 2: "LoiChi", 3: "LoiNhua", 4: "LoiTray"}

    predictor = SAHIPredictor(
        weights=weights,
        slice_w=slice_w,
        slice_h=slice_h,
        overlap_ratio=overlap_ratio,
        conf_thresh=conf_thresh,
        iou_thresh=iou_thresh,
        device=device,
    )

    image_files = sorted(list(test_images_dir.glob("*.jpg")) + list(test_images_dir.glob("*.png")))
    print(f"Running SAHI evaluation on {len(image_files)} test images...")

    all_preds_by_img = []
    all_gts_by_img = []
    e2e_times = []
    slice_times = []
    infer_times = []
    merge_times = []

    for idx, img_p in enumerate(image_files):
        img = cv2.imread(str(img_p))
        if img is None:
            continue
        h, w = img.shape[:2]

        preds, timings = predictor.predict_image(img)
        all_preds_by_img.append(preds)
        e2e_times.append(timings["total_e2e_ms"])
        slice_times.append(timings["slice_ms"])
        infer_times.append(timings["inference_ms"])
        merge_times.append(timings["merge_ms"])

        # Load ground truth
        lbl_p = test_labels_dir / f"{img_p.stem}.txt"
        gt_boxes = []
        if lbl_p.exists():
            for line in lbl_p.read_text(encoding="utf-8").strip().splitlines():
                parts = line.strip().split()
                if len(parts) >= 5:
                    c = int(float(parts[0]))
                    xc, yc, bw, bh = map(float, parts[1:5])
                    x1 = (xc - bw / 2.0) * w
                    y1 = (yc - bh / 2.0) * h
                    x2 = (xc + bw / 2.0) * w
                    y2 = (yc + bh / 2.0) * h
                    gt_boxes.append([x1, y1, x2, y2, c])
        all_gts_by_img.append(np.array(gt_boxes, dtype=np.float32) if gt_boxes else np.empty((0, 5), dtype=np.float32))

        if (idx + 1) % 100 == 0 or (idx + 1) == len(image_files):
            print(f"  [{idx+1}/{len(image_files)}] mean latency: {np.mean(e2e_times):.1f} ms")

    # Compute COCO-style metrics
    iou_thresholds = np.linspace(0.5, 0.95, 10)
    per_class_ap = {c: [] for c in range(num_classes)}
    recalls_50 = []

    for c in range(num_classes):
        # Accumulate detections and ground truths for class c
        c_preds = []
        c_gts_count = 0

        for img_preds, img_gts in zip(all_preds_by_img, all_gts_by_img):
            p_mask = img_preds[:, 5] == c if len(img_preds) else np.zeros(0, dtype=bool)
            g_mask = img_gts[:, 4] == c if len(img_gts) else np.zeros(0, dtype=bool)

            cur_p = img_preds[p_mask] if len(img_preds) else np.empty((0, 6))
            cur_g = img_gts[g_mask, :4] if len(img_gts) else np.empty((0, 4))
            c_gts_count += len(cur_g)

            if len(cur_p):
                # Sort detections by confidence descending
                order = np.argsort(-cur_p[:, 4])
                c_preds.append((cur_p[order], cur_g))
            else:
                c_preds.append((np.empty((0, 6)), cur_g))

        if c_gts_count == 0:
            per_class_ap[c] = [0.0] * len(iou_thresholds)
            continue

        class_ap_thresholds = []
        for iou_th in iou_thresholds:
            tp = []
            fp = []

            for cur_p, cur_g in c_preds:
                if len(cur_p) == 0:
                    continue
                matched = np.zeros(len(cur_g), dtype=bool)
                ious = box_iou(cur_p[:, :4], cur_g)

                for p_idx in range(len(cur_p)):
                    best_iou = -1.0
                    best_gt = -1
                    for g_idx in range(len(cur_g)):
                        if not matched[g_idx] and ious[p_idx, g_idx] > best_iou:
                            best_iou = ious[p_idx, g_idx]
                            best_gt = g_idx
                    if best_iou >= iou_th:
                        tp.append(1)
                        fp.append(0)
                        matched[best_gt] = True
                    else:
                        tp.append(0)
                        fp.append(1)

            if len(tp) == 0:
                class_ap_thresholds.append(0.0)
                continue

            tp_cumsum = np.cumsum(tp)
            fp_cumsum = np.cumsum(fp)
            rec = tp_cumsum / float(c_gts_count)
            prec = tp_cumsum / np.maximum(tp_cumsum + fp_cumsum, 1e-6)

            # 101-point interpolated AP
            mrec = np.concatenate(([0.0], rec, [1.0]))
            mpre = np.concatenate(([0.0], prec, [0.0]))
            for i in range(len(mpre) - 2, -1, -1):
                mpre[i] = max(mpre[i], mpre[i + 1])
            i = np.where(mrec[1:] != mrec[:-1])[0]
            ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
            class_ap_thresholds.append(float(ap))

            if np.isclose(iou_th, 0.5):
                recalls_50.append(float(rec[-1]) if len(rec) else 0.0)

        per_class_ap[c] = class_ap_thresholds

    map50_95 = float(np.mean([np.mean(aps) for aps in per_class_ap.values()]))
    map50 = float(np.mean([aps[0] for aps in per_class_ap.values()]))
    map75 = float(np.mean([aps[5] for aps in per_class_ap.values()]))
    mean_recall = float(np.mean(recalls_50)) if recalls_50 else 0.0

    class_breakdown = {}
    for c, aps in per_class_ap.items():
        name = class_names.get(c, str(c))
        class_breakdown[name] = {
            "ap50_95": float(np.mean(aps)),
            "ap50": float(aps[0]),
            "ap75": float(aps[5]),
        }

    e2e_arr = np.array(e2e_times)
    metrics_report = {
        "mAP50_95": map50_95,
        "mAP50": map50,
        "mAP75": map75,
        "recall": mean_recall,
        "per_class": class_breakdown,
        "speed": {
            "mean_slice_ms": float(np.mean(slice_times)),
            "mean_inference_ms": float(np.mean(infer_times)),
            "mean_merge_ms": float(np.mean(merge_times)),
            "mean_e2e_ms": float(np.mean(e2e_arr)),
            "median_e2e_ms": float(np.median(e2e_arr)),
            "p95_e2e_ms": float(np.percentile(e2e_arr, 95)),
            "fps": float(1000.0 / np.mean(e2e_arr)) if len(e2e_arr) else 0.0,
        },
        "config": {
            "slice_w": slice_w,
            "slice_h": slice_h,
            "overlap_ratio": overlap_ratio,
            "conf_thresh": conf_thresh,
            "iou_thresh": iou_thresh,
            "total_test_images": len(image_files),
        },
    }

    output_json.write_text(json.dumps(metrics_report, indent=2), encoding="utf-8")
    print(f"\n=== SAHI Evaluation Results Saved to {output_json} ===")
    print(f"mAP50-95: {map50_95*100:.2f}% | mAP50: {map50*100:.2f}% | mAP75: {map75*100:.2f}% | Recall: {mean_recall*100:.2f}%")
    print(f"Mean Latency: {np.mean(e2e_arr):.1f} ms | FPS: {1000.0/np.mean(e2e_arr):.1f}")
    return metrics_report
