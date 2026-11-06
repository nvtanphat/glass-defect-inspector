"""Ma trận nhầm lẫn phát hiện đối tượng: Hàng là nhãn thực tế (Ground Truth),
cột là dự đoán (Predictions).

Xử lý triệt để:
- True Positive (TP): Dự đoán đúng phân lớp và IoU >= ngưỡng.
- False Positive (FP): Báo động giả từ nền (Background) hoặc phân loại nhầm lớp khác.
- False Negative (FN): Bỏ sót khuyết tật vào cột Background.
"""

from collections import defaultdict

import numpy as np


def box_iou_xywh(boxes_a, boxes_b):
    """Tính ma trận IoU giữa hai tập hợp bounding box định dạng [x, y, w, h]."""
    a = np.asarray(boxes_a, dtype=float).reshape(-1, 4)
    b = np.asarray(boxes_b, dtype=float).reshape(-1, 4)
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Tọa độ bounding box phải là số hữu hạn.")
    if (a[:, 2:] < 0).any() or (b[:, 2:] < 0).any():
        raise ValueError("Kích thước chiều rộng/cao của hộp không được âm.")
    intersection_start = np.maximum(a[:, None, :2], b[None, :, :2])
    intersection_end = np.minimum(a[:, None, :2] + a[:, None, 2:], b[None, :, :2] + b[None, :, 2:])
    intersection = np.maximum(intersection_end - intersection_start, 0).prod(axis=2)
    union = a[:, 2:].prod(axis=1)[:, None] + b[:, 2:].prod(axis=1)[None, :] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def detection_confusion(images, annotations, predictions, nc=5, confidence=0.5, iou=0.5):
    """Ghép cặp IoU cao nhất một lần duy nhất, bất kể phân lớp, để lỗi phân lớp
    luôn hiển thị rõ ràng.

    Hộp nhãn thực không khớp chiếm cột background (cột nc); hộp dự đoán không khớp
    chiếm hàng background (hàng nc). Ô background/background luôn bằng 0.
    """
    if not 0 <= confidence <= 1 or not 0 <= iou <= 1:
        raise ValueError("Các ngưỡng phải nằm trong khoảng [0, 1]")
    ids = {image["id"] for image in images}
    if len(ids) != len(images):
        raise ValueError("Phát hiện ID ảnh bị trùng lặp.")
    gt_by_image, pred_by_image = defaultdict(list), defaultdict(list)
    for annotation in annotations:
        if annotation["image_id"] not in ids or not 0 <= annotation["category_id"] < nc:
            raise ValueError("ID hoặc phân lớp của annotation không hợp lệ.")
        gt_by_image[annotation["image_id"]].append(annotation)
    for prediction in predictions:
        if prediction["image_id"] not in ids or not 0 <= prediction["category_id"] < nc:
            raise ValueError("ID hoặc phân lớp của prediction không hợp lệ.")
        if not np.isfinite(prediction["score"]) or not 0 <= prediction["score"] <= 1:
            raise ValueError("Điểm tin cậy confidence không hợp lệ.")
        if prediction["score"] >= confidence:
            pred_by_image[prediction["image_id"]].append(prediction)
    matrix = np.zeros((nc + 1, nc + 1), dtype=np.int64)
    errors = []
    for image in images:
        image_id = image["id"]
        truth, predicted = gt_by_image[image_id], pred_by_image[image_id]
        overlaps = box_iou_xywh([g["bbox"] for g in truth], [p["bbox"] for p in predicted])
        candidates = np.argwhere(overlaps >= iou)
        # Sắp xếp ưu tiên cặp có IoU cao nhất trước
        pairs = sorted(candidates.tolist(), key=lambda pair: (-overlaps[tuple(pair)], *pair))
        used_gt, used_pred = set(), set()
        wrong_class = 0
        for gt_index, pred_index in pairs:
            if gt_index in used_gt or pred_index in used_pred:
                continue
            used_gt.add(gt_index)
            used_pred.add(pred_index)
            gt_class = truth[gt_index]["category_id"]
            pred_class = predicted[pred_index]["category_id"]
            matrix[gt_class, pred_class] += 1
            wrong_class += int(gt_class != pred_class)
        # Nhãn thực bị bỏ sót (False Negative)
        for index, gt in enumerate(truth):
            if index not in used_gt:
                matrix[gt["category_id"], nc] += 1
        # Dự đoán sai báo động giả (False Alarm)
        for index, prediction in enumerate(predicted):
            if index not in used_pred:
                matrix[nc, prediction["category_id"]] += 1
        errors.append(
            dict(
                image_id=image_id,
                missed=len(truth) - len(used_gt),
                false_alarms=len(predicted) - len(used_pred),
                wrong_class=wrong_class,
            )
        )
    assert matrix[:nc].sum() == len(annotations)
    assert matrix[:, :nc].sum() == sum(map(len, pred_by_image.values()))
    return matrix, errors


def confusion_statistics(matrix, class_names):
    """Tính toán bảng thống kê chi tiết từ ma trận nhầm lẫn: Precision, Recall, F1, FN, FP."""
    nc = len(class_names)
    matrix = np.asarray(matrix)
    if matrix.shape != (nc + 1, nc + 1) or (matrix < 0).any():
        raise ValueError("Ma trận nhầm lẫn không hợp lệ về kích thước hoặc chứa giá trị âm.")
    stats = []
    for index, name in enumerate(class_names):
        tp = int(matrix[index, index])
        fn = int(matrix[index].sum()) - tp
        fp = int(matrix[:, index].sum()) - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        stats.append(
            dict(
                class_name=name,
                support=tp + fn,
                tp=tp,
                fp=fp,
                fn=fn,
                precision=precision,
                recall=recall,
                f1=2 * precision * recall / (precision + recall) if precision + recall else 0.0,
                missed_to_background=int(matrix[index, nc]),
                false_alarms_from_background=int(matrix[nc, index]),
                misclassified=int(matrix[index, :nc].sum()) - tp,
            )
        )
    tp = sum(row["tp"] for row in stats)
    fp = sum(row["fp"] for row in stats)
    fn = sum(row["fn"] for row in stats)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return stats, dict(
        tp=tp,
        fp=fp,
        fn=fn,
        precision=precision,
        recall=recall,
        f1=2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        missed_to_background=int(matrix[:nc, nc].sum()),
        false_alarms_from_background=int(matrix[nc, :nc].sum()),
        wrong_class=int(matrix[:nc, :nc].sum()) - tp,
    )
