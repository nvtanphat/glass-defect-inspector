"""Bộ đánh giá độ chính xác COCO chi tiết theo từng phân lớp khuyết tật (Per-Class COCO Evaluator).

Tính toán các chỉ số:
- Tổng thể: mAP@50:95, mAP@50, mAP@75, AP_small, AP_medium, AP_large, AR_small
- Từng phân lớp: AP@50:95, AP@50, AP@75, AP_small, Recall
- Tương thích hoàn toàn với pycocotools và faster_coco_eval.
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
    """Đánh giá các dự đoán bounding box đối chiếu với file JSON nhãn Ground Truth chuẩn COCO.

    Tham số:
        gt_json_path: Đường dẫn tới file JSON ground truth chuẩn COCO.
        predictions: Danh sách các từ điển dự đoán gồm các khóa:
            ['image_id', 'category_id', 'bbox' (x, y, w, h), 'score']
        class_names: Danh sách tên lớp sắp xếp theo thứ tự ID (0 đến 4).
        output_json_path: Đường dẫn tùy chọn để lưu file kết quả JSON.

    Trả về:
        dict: Chứa đầy đủ các chỉ số mAP tổng quan và chi tiết từng lớp khuyết tật.
    """
    try:
        from faster_coco_eval import COCO, COCOeval
    except ImportError:
        try:
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval
        except ImportError:
            raise ImportError("Yêu cầu cài đặt faster_coco_eval hoặc pycocotools.")

    gt_json_path = str(Path(gt_json_path).resolve())
    coco_gt = COCO(gt_json_path)

    if not predictions:
        print("[Evaluator] Cảnh báo: Danh sách dự đoán rỗng.")
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

    # Nạp kết quả dự đoán và khởi tạo đối tượng đánh giá COCOeval
    coco_dt = coco_gt.loadRes(predictions)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # Thống kê tổng hợp coco_eval.stats:
    # 0: AP @ IoU=0.50:0.95 (tất cả quy mô)
    # 1: AP @ IoU=0.50 (tất cả quy mô)
    # 2: AP @ IoU=0.75 (tất cả quy mô)
    # 3: AP @ IoU=0.50:0.95 (vật thể nhỏ: diện tích < 32^2 px)
    # 4: AP @ IoU=0.50:0.95 (vật thể vừa: 32^2 <= diện tích < 96^2 px)
    # 5: AP @ IoU=0.50:0.95 (vật thể lớn: diện tích >= 96^2 px)
    # 8: AR @ IoU=0.50:0.95 (Recall trung bình)
    # 9: AR @ IoU=0.50:0.95 (Recall vật thể nhỏ)
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

    # Trích xuất AP từng phân lớp từ mảng precision [T, R, K, A, M]
    prec = coco_eval.eval["precision"]
    rec = coco_eval.eval["recall"]
    cat_ids = coco_eval.params.catIds

    per_class_metrics = {}
    for k_idx, cat_id in enumerate(cat_ids):
        cat_name = class_names[cat_id] if cat_id < len(class_names) else f"class_{cat_id}"

        # AP 50:95 (toàn bộ diện tích)
        p_all_thresh = prec[:, :, k_idx, 0, -1]
        valid_p = p_all_thresh[p_all_thresh >= 0]
        ap_50_95 = float(np.mean(valid_p)) * 100 if len(valid_p) > 0 else 0.0

        # AP 50 (ngưỡng IoU = 0.50)
        p_50 = prec[0, :, k_idx, 0, -1]
        valid_p50 = p_50[p_50 >= 0]
        ap_50 = float(np.mean(valid_p50)) * 100 if len(valid_p50) > 0 else 0.0

        # AP 75 (ngưỡng IoU = 0.75)
        p_75 = prec[5, :, k_idx, 0, -1]
        valid_p75 = p_75[p_75 >= 0]
        ap_75 = float(np.mean(valid_p75)) * 100 if len(valid_p75) > 0 else 0.0

        # Độ thu hồi (Recall)
        r_all = rec[:, k_idx, 0, -1]
        valid_r = r_all[r_all >= 0]
        recall = float(np.mean(valid_r)) * 100 if len(valid_r) > 0 else 0.0

        # AP cho khuyết tật nhỏ (diện tích mức small)
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
        print(f"[Evaluator] Đã lưu báo cáo COCO mAP chi tiết tại: {out_p}")

    return full_results
