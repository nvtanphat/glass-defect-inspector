"""Per-Class COCO Evaluator for Object Detection.

Computes:
- Overall: mAP50:95, mAP50, mAP75, AP_small, AP_medium, AP_large
- Per-class: AP50:95, AP50, AP75, Recall
- Confusion / detection statistics

Works seamlessly with pycocotools / faster_coco_eval.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


CLASS_NAMES_DEFAULT = ["DiVat", "DiVatLoiLom", "LoiChi", "LoiNhua", "LoiTray"]


def evaluate_coco_predictions(
    gt_json_path: str | Path,
    predictions: list[dict[str, Any]],
    class_names: list[str] = CLASS_NAMES_DEFAULT,
    output_json_path: str | Path | None = None,
) -> dict[str, Any]:
    """Evaluate bounding box predictions against a COCO ground-truth JSON file.

    Args:
        gt_json_path: Path to COCO ground truth JSON file.
        predictions: List of dicts, each with keys:
            ['image_id', 'category_id', 'bbox' (x, y, w, h), 'score']
        class_names: List of class names ordered by category id.
        output_json_path: Optional path to save evaluation metrics JSON.

    Returns:
        Structured dictionary containing overall and per-class metrics.
    """
    try:
        from faster_coco_eval import COCO, COCOeval
    except ImportError:
        try:
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval
        except ImportError:
            raise ImportError("Neither faster_coco_eval nor pycocotools is installed.")

    gt_json_path = str(Path(gt_json_path).resolve())
    coco_gt = COCO(gt_json_path)

    if not predictions:
        print("[Evaluator] Warning: predictions list is empty.")
        empty_metrics = {
            "mAP50_95": 0.0,
            "mAP50": 0.0,
            "mAP75": 0.0,
            "AP_small": 0.0,
            "AP_medium": 0.0,
            "AP_large": 0.0,
            "per_class": {
                name: {"AP50_95": 0.0, "AP50": 0.0, "AP75": 0.0, "Recall": 0.0}
                for name in class_names
            },
        }
        if output_json_path:
            Path(output_json_path).parent.mkdir(parents=True, exist_ok=True)
            Path(output_json_path).write_text(json.dumps(empty_metrics, indent=2), encoding="utf-8")
        return empty_metrics

    coco_dt = coco_gt.loadRes(predictions)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # coco_eval.stats:
    # 0: AP @ IoU=0.50:0.95, area=all, maxDets=100
    # 1: AP @ IoU=0.50, area=all, maxDets=100
    # 2: AP @ IoU=0.75, area=all, maxDets=100
    # 3: AP @ IoU=0.50:0.95, area=small, maxDets=100
    # 4: AP @ IoU=0.50:0.95, area=medium, maxDets=100
    # 5: AP @ IoU=0.50:0.95, area=large, maxDets=100
    # 6: AR @ IoU=0.50:0.95, area=all, maxDets=1
    # 7: AR @ IoU=0.50:0.95, area=all, maxDets=10
    # 8: AR @ IoU=0.50:0.95, area=all, maxDets=100
    # 9: AR @ IoU=0.50:0.95, area=small, maxDets=100
    # 10: AR @ IoU=0.50:0.95, area=medium, maxDets=100
    # 11: AR @ IoU=0.50:0.95, area=large, maxDets=100
    stats = coco_eval.stats

    overall_metrics = {
        "mAP50_95": round(float(stats[0]) * 100, 2) if stats[0] >= 0 else 0.0,
        "mAP50": round(float(stats[1]) * 100, 2) if stats[1] >= 0 else 0.0,
        "mAP75": round(float(stats[2]) * 100, 2) if stats[2] >= 0 else 0.0,
        "AP_small": round(float(stats[3]) * 100, 2) if stats[3] >= 0 else 0.0,
        "AP_medium": round(float(stats[4]) * 100, 2) if stats[4] >= 0 else 0.0,
        "AP_large": round(float(stats[5]) * 100, 2) if stats[5] >= 0 else 0.0,
        "AR50_95": round(float(stats[8]) * 100, 2) if stats[8] >= 0 else 0.0,
        "AR_small": round(float(stats[9]) * 100, 2) if stats[9] >= 0 else 0.0,
        "AR_medium": round(float(stats[10]) * 100, 2) if stats[10] >= 0 else 0.0,
        "AR_large": round(float(stats[11]) * 100, 2) if stats[11] >= 0 else 0.0,
    }

    # Extract per-class AP from precision array:
    # precision shape: [T, R, K, A, M]
    # T: 10 IoU thresholds [0.50, 0.55, ..., 0.95]
    # R: 101 recall thresholds [0.0, 0.01, ..., 1.0]
    # K: categories
    # A: 4 area ranges [all, small, medium, large]
    # M: 3 maxDets [1, 10, 100]
    prec = coco_eval.eval["precision"]  # numpy ndarray
    rec = coco_eval.eval["recall"]  # numpy ndarray [T, K, A, M]
    cat_ids = coco_eval.params.catIds

    per_class_metrics = {}
    for k_idx, cat_id in enumerate(cat_ids):
        cat_name = class_names[cat_id] if cat_id < len(class_names) else f"class_{cat_id}"

        # AP 50:95 (all areas, maxDets=100)
        p_all_thresh = prec[:, :, k_idx, 0, -1]
        valid_p = p_all_thresh[p_all_thresh >= 0]
        ap_50_95 = float(np.mean(valid_p)) * 100 if len(valid_p) > 0 else 0.0

        # AP 50 (IoU index 0)
        p_50 = prec[0, :, k_idx, 0, -1]
        valid_p50 = p_50[p_50 >= 0]
        ap_50 = float(np.mean(valid_p50)) * 100 if len(valid_p50) > 0 else 0.0

        # AP 75 (IoU index 5)
        p_75 = prec[5, :, k_idx, 0, -1]
        valid_p75 = p_75[p_75 >= 0]
        ap_75 = float(np.mean(valid_p75)) * 100 if len(valid_p75) > 0 else 0.0

        # Recall (max recall across IoUs for maxDets=100)
        r_all = rec[:, k_idx, 0, -1]
        valid_r = r_all[r_all >= 0]
        recall = float(np.mean(valid_r)) * 100 if len(valid_r) > 0 else 0.0

        # AP small (area index 1)
        p_small = prec[:, :, k_idx, 1, -1]
        valid_psmall = p_small[p_small >= 0]
        ap_small = float(np.mean(valid_psmall)) * 100 if len(valid_psmall) > 0 else 0.0

        per_class_metrics[cat_name] = {
            "AP50_95": round(ap_50_95, 2),
            "AP50": round(ap_50, 2),
            "AP75": round(ap_75, 2),
            "AP_small": round(ap_small, 2),
            "Recall": round(recall, 2),
        }

    full_results = {
        **overall_metrics,
        "per_class": per_class_metrics,
        "num_predictions": len(predictions),
    }

    if output_json_path:
        out_p = Path(output_json_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(full_results, indent=2), encoding="utf-8")
        print(f"[Evaluator] Saved comprehensive COCO metrics to: {out_p}")

    return full_results
