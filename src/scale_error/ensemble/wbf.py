"""Weighted Boxes Fusion (WBF) and Score Calibration for Glass Defect Detection.

Pure NumPy / Python implementation of Weighted Boxes Fusion (WBF)
tailored for extreme small defects, high aspect-ratio cracks/scratches,
and class-imbalanced multi-model inspection.

References:
    Solovyev et al., "Weighted boxes fusion: Ensembling methods for object detection",
    Image and Vision Computing, 2021.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

CLASS_NAMES = ["DiVat", "DiVatLoiLom", "LoiChi", "LoiNhua", "LoiTray"]

# Default per-class model weights matrix for [FBDNet-M (Transformer), YOLO26m/YOLOv10m (CNN)]
# Columns: [Model 0: FBDNet-M / RF-DETR, Model 1: YOLO26m / YOLOv10m]
DEFAULT_PER_CLASS_WEIGHTS: dict[str, list[float]] = {
    # DiVat: Hạt vi mô cực nhỏ -> FBDNet (DCFSB / NWD) chiếm ưu thế áp đảo
    "DiVat": [1.8, 0.7],
    # DiVatLoiLom: Lõm quang học -> FBDNet nhỉnh hơn
    "DiVatLoiLom": [1.4, 0.9],
    # LoiChi: Sợi mảnh -> Cả hai đều >63%, trọng số cân bằng
    "LoiChi": [1.0, 1.0],
    # LoiNhua: Biên mềm diện tích -> YOLO áp đảo (53.87% vs 46.57%)
    "LoiNhua": [0.7, 1.9],
    # LoiTray: Vết xước dài mảnh -> FBDNet áp đảo nhờ AFTA (41.04% vs 29.53%)
    "LoiTray": [2.0, 0.6],
}

# Calibrated Per-Class dynamic IoU thresholds:
DEFAULT_PER_CLASS_IOU: dict[str, float] = {
    "DiVat": 0.55,
    "DiVatLoiLom": 0.41,
    "LoiChi": 0.65,
    "LoiNhua": 0.59,
    "LoiTray": 0.43,
}

# Optimal Per-Class Heterogeneous Routing Hyperparameters (mAP@50:95: 46.10%, mAP@50: 77.65%):
DEFAULT_OPTIMAL_HYPERPARAMS: dict[str, dict] = {
    "DiVat": {
        "mode": "3M_MS_TTA",
        "iou_thr": 0.55,
        "weights": [1.0, 2.5, 0.8],
        "coord_ratio": [0.33, 0.33, 0.34],
        "boost_3": 1.25,
        "boost_2": 1.05,
        "solo_decay": 0.85,
        "conf_floor": 0.12,
        "final_ap": 26.28,
    },
    "DiVatLoiLom": {
        "mode": "2M_FBD_YOLO_MS",
        "iou_thr": 0.41,
        "weights": [2.0, 1.0],
        "coord_ratio": [0.60, 0.40],
        "boost_3": 1.0,
        "boost_2": 1.22,
        "solo_decay": 0.85,
        "conf_floor": 0.03,
        "final_ap": 35.81,
    },
    "LoiChi": {
        "mode": "2M_FBD_YOLO",
        "iou_thr": 0.65,
        "weights": [2.5, 0.5],
        "coord_ratio": [0.75, 0.25],
        "boost_3": 1.0,
        "boost_2": 1.05,
        "solo_decay": 0.85,
        "conf_floor": 0.05,
        "final_ap": 66.88,
    },
    "LoiNhua": {
        "mode": "2M_FBD_YOLO",
        "iou_thr": 0.59,
        "weights": [0.8, 2.0],
        "coord_ratio": [0.60, 0.40],
        "boost_3": 1.0,
        "boost_2": 1.05,
        "solo_decay": 0.85,
        "conf_floor": 0.05,
        "final_ap": 57.79,
    },
    "LoiTray": {
        "mode": "3M_MS_TTA",
        "iou_thr": 0.43,
        "weights": [0.8, 0.8, 2.5],
        "coord_ratio": [0.45, 0.45, 0.10],
        "boost_3": 1.10,
        "boost_2": 1.05,
        "solo_decay": 0.85,
        "conf_floor": 0.18,
        "final_ap": 43.73,
    },
}

DEFAULT_APEX_HYPERPARAMS = DEFAULT_OPTIMAL_HYPERPARAMS


def compute_iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Compute pairwise IoU between two sets of boxes [x1, y1, x2, y2]."""
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)

    w_a = np.maximum(0.0, boxes_a[:, 2] - boxes_a[:, 0])
    h_a = np.maximum(0.0, boxes_a[:, 3] - boxes_a[:, 1])
    area_a = w_a * h_a

    w_b = np.maximum(0.0, boxes_b[:, 2] - boxes_b[:, 0])
    h_b = np.maximum(0.0, boxes_b[:, 3] - boxes_b[:, 1])
    area_b = w_b * h_b

    iw = np.maximum(
        0.0,
        np.minimum(boxes_a[:, 2:3], boxes_b[:, 2:3].T)
        - np.maximum(boxes_a[:, 0:1], boxes_b[:, 0:1].T),
    )
    ih = np.maximum(
        0.0,
        np.minimum(boxes_a[:, 3:4], boxes_b[:, 3:4].T)
        - np.maximum(boxes_a[:, 1:2], boxes_b[:, 1:2].T),
    )
    inter = iw * ih

    union = area_a[:, None] + area_b[None, :] - inter
    union = np.maximum(union, 1e-8)
    return inter / union


def calibrate_scores(
    scores: np.ndarray,
    model_type: str = "yolo",
    temperature: float = 1.0,
    gamma: float = 1.0,
) -> np.ndarray:
    """Calibrate confidence scores across heterogeneous detector families.

    DETR/D-FINE models produce different score distributions compared to YOLO
    (DETR sigmoid outputs are generally more conservative, while YOLO uses Focal/BCE
    with peaky scores). This calibration prevents YOLO scores from overwhelming DETR.
    """
    scores = np.asarray(scores, dtype=np.float32)
    if len(scores) == 0:
        return scores

    if model_type.lower() in {"detr", "dfine", "fbdnet"}:
        # Slight boost for confident DETR detections
        calibrated = np.power(scores, 1.0 / (gamma * temperature))
    elif model_type.lower() in {"yolo", "yolov8", "yolov10", "yolo26"}:
        # Soften peaky YOLO scores
        calibrated = np.power(scores, gamma * temperature)
    else:
        calibrated = scores

    return np.clip(calibrated, 0.0, 1.0)


def weighted_boxes_fusion(
    boxes_list: list[np.ndarray | list],
    scores_list: list[np.ndarray | list],
    labels_list: list[np.ndarray | list],
    weights: list[float] | None = None,
    iou_thr: float = 0.58,
    skip_box_thr: float = 0.05,
    conf_type: str = "avg",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Execute Weighted Boxes Fusion over multiple models predictions.

    Args:
        boxes_list: List of boxes arrays for each model, shape (N_i, 4) in [x1, y1, x2, y2].
        scores_list: List of confidence score arrays for each model, shape (N_i,).
        labels_list: List of integer class labels for each model, shape (N_i,).
        weights: Model importance weights, length equal to len(boxes_list).
        iou_thr: IoU threshold for clustering boxes of the same defect.
        skip_box_thr: Pre-filter threshold to drop negligible noise boxes.
        conf_type: Method to compute cluster confidence:
            'avg': Simple average of matched boxes scaled by model count.
            'max': Maximum confidence in cluster.
            'weighted': Weighted average of matched confidences.

    Returns:
        (fused_boxes, fused_scores, fused_labels)
    """
    num_models = len(boxes_list)
    if num_models == 0:
        return np.empty((0, 4)), np.empty((0,)), np.empty((0,), dtype=int)

    if weights is None:
        weights = [1.0] * num_models
    weights = [float(w) for w in weights]
    total_model_weight = sum(weights)

    # Collect and pre-filter boxes
    filtered_boxes: list[list[float]] = []
    filtered_scores: list[float] = []
    filtered_labels: list[int] = []
    filtered_model_indices: list[int] = []

    for m_idx in range(num_models):
        b_arr = np.asarray(boxes_list[m_idx], dtype=np.float32)
        s_arr = np.asarray(scores_list[m_idx], dtype=np.float32)
        l_arr = np.asarray(labels_list[m_idx], dtype=int)

        if len(b_arr) == 0:
            continue

        keep = s_arr >= skip_box_thr
        b_arr = b_arr[keep]
        s_arr = s_arr[keep]
        l_arr = l_arr[keep]

        for b, s, lbl in zip(b_arr, s_arr, l_arr, strict=False):
            filtered_boxes.append(b.tolist())
            filtered_scores.append(float(s))
            filtered_labels.append(int(lbl))
            filtered_model_indices.append(m_idx)

    if not filtered_boxes:
        return np.empty((0, 4)), np.empty((0,)), np.empty((0,), dtype=int)

    unique_labels = sorted(list(set(filtered_labels)))
    final_boxes: list[list[float]] = []
    final_scores: list[float] = []
    final_labels: list[int] = []

    # Process class-by-class independently
    for target_label in unique_labels:
        cls_indices = [i for i, lbl in enumerate(filtered_labels) if lbl == target_label]
        cls_boxes = [filtered_boxes[i] for i in cls_indices]
        cls_scores = [filtered_scores[i] for i in cls_indices]
        cls_models = [filtered_model_indices[i] for i in cls_indices]

        # Sort descending by score
        order = np.argsort(cls_scores)[::-1]
        cls_boxes = [cls_boxes[i] for i in order]
        cls_scores = [cls_scores[i] for i in order]
        cls_models = [cls_models[i] for i in order]

        # Clusters: list of lists containing matching item dicts
        clusters: list[list[dict]] = []

        for b, s, m in zip(cls_boxes, cls_scores, cls_models, strict=False):
            b_arr = np.array(b, dtype=np.float32)
            matched_cluster_idx = -1
            best_iou = iou_thr

            for c_idx, cluster in enumerate(clusters):
                # Calculate cluster center box
                cluster_boxes = np.array([item["box"] for item in cluster], dtype=np.float32)
                cluster_weights = np.array(
                    [item["weight"] * item["score"] for item in cluster],
                    dtype=np.float32,
                )
                sum_w = np.sum(cluster_weights)
                center_box = (
                    np.sum(cluster_boxes * cluster_weights[:, None], axis=0) / max(sum_w, 1e-8)
                )

                # IoU between current box and cluster center
                iou = compute_iou_matrix(b_arr[None, :], center_box[None, :])[0, 0]
                if iou > best_iou:
                    best_iou = iou
                    matched_cluster_idx = c_idx

            item_data = {"box": b, "score": s, "model": m, "weight": weights[m]}
            if matched_cluster_idx >= 0:
                clusters[matched_cluster_idx].append(item_data)
            else:
                clusters.append([item_data])

        # Compute fused box for each cluster
        for cluster in clusters:
            c_boxes = np.array([item["box"] for item in cluster], dtype=np.float32)
            c_scores = np.array([item["score"] for item in cluster], dtype=np.float32)
            c_weights = np.array([item["weight"] for item in cluster], dtype=np.float32)

            box_weights = c_weights * c_scores
            sum_bw = np.sum(box_weights)
            fused_b = np.sum(c_boxes * box_weights[:, None], axis=0) / max(sum_bw, 1e-8)

            if conf_type == "max":
                fused_s = float(np.max(c_scores))
            elif conf_type == "weighted":
                fused_s = float(np.sum(c_scores * c_weights) / max(np.sum(c_weights), 1e-8))
            else:  # 'avg' scaled by number of models
                fused_s = float(np.sum(c_scores * c_weights) / max(total_model_weight, 1e-8))

            final_boxes.append(fused_b.tolist())
            final_scores.append(fused_s)
            final_labels.append(target_label)

    if not final_boxes:
        return np.empty((0, 4)), np.empty((0,)), np.empty((0,), dtype=int)

    final_boxes_np = np.array(final_boxes, dtype=np.float32)
    final_scores_np = np.array(final_scores, dtype=np.float32)
    final_labels_np = np.array(final_labels, dtype=int)

    # Sort final predictions by confidence score descending
    final_order = np.argsort(final_scores_np)[::-1]
    return final_boxes_np[final_order], final_scores_np[final_order], final_labels_np[final_order]


def per_class_weighted_boxes_fusion(
    boxes_list: list[np.ndarray | list],
    scores_list: list[np.ndarray | list],
    labels_list: list[np.ndarray | list],
    per_class_weights: dict[str, list[float]] | None = None,
    class_names: Sequence[str] = CLASS_NAMES,
    iou_thr: float = 0.58,
    skip_box_thr: float = 0.05,
    conf_type: str = "avg",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Execute WBF using per-class adaptive weights matrix.

    Allows specific models to dominate on classes where they have proven SOTA advantages
    (e.g., FBDNet on LoiTray & DiVat; YOLO on LoiNhua).
    """
    weights_map = per_class_weights or DEFAULT_PER_CLASS_WEIGHTS
    num_models = len(boxes_list)
    all_fused_boxes = []
    all_fused_scores = []
    all_fused_labels = []

    unique_classes = set()
    for l_arr in labels_list:
        unique_classes.update(np.asarray(l_arr, dtype=int).tolist())

    for cls_idx in sorted(list(unique_classes)):
        cls_name = class_names[cls_idx] if 0 <= cls_idx < len(class_names) else f"class_{cls_idx}"
        class_weights = weights_map.get(cls_name, [1.0] * num_models)
        if len(class_weights) < num_models:
            class_weights = class_weights + [1.0] * (num_models - len(class_weights))

        # Filter predictions for this specific class
        cls_boxes_list = []
        cls_scores_list = []
        cls_labels_list = []

        for m_idx in range(num_models):
            b = np.asarray(boxes_list[m_idx], dtype=np.float32)
            s = np.asarray(scores_list[m_idx], dtype=np.float32)
            lbl_arr = np.asarray(labels_list[m_idx], dtype=int)

            if len(lbl_arr) == 0:
                cls_boxes_list.append(np.empty((0, 4)))
                cls_scores_list.append(np.empty((0,)))
                cls_labels_list.append(np.empty((0,), dtype=int))
                continue

            mask = lbl_arr == cls_idx
            cls_boxes_list.append(b[mask])
            cls_scores_list.append(s[mask])
            cls_labels_list.append(lbl_arr[mask])

        fb, fs, fl = weighted_boxes_fusion(
            boxes_list=cls_boxes_list,
            scores_list=cls_scores_list,
            labels_list=cls_labels_list,
            weights=class_weights[:num_models],
            iou_thr=iou_thr,
            skip_box_thr=skip_box_thr,
            conf_type=conf_type,
        )

        if len(fb) > 0:
            all_fused_boxes.append(fb)
            all_fused_scores.append(fs)
            all_fused_labels.append(fl)

    if not all_fused_boxes:
        return np.empty((0, 4)), np.empty((0,)), np.empty((0,), dtype=int)

    fused_b = np.vstack(all_fused_boxes)
    fused_s = np.concatenate(all_fused_scores)
    fused_l = np.concatenate(all_fused_labels)

    order = np.argsort(fused_s)[::-1]
    return fused_b[order], fused_s[order], fused_l[order]


def fuse_detection_lists(
    detection_lists: list[list[tuple[str, float, int, int, int, int]]],
    weights: list[float] | None = None,
    per_class_weights: dict[str, list[float]] | None = None,
    class_names: Sequence[str] = CLASS_NAMES,
    iou_thr: float = 0.58,
    skip_box_thr: float = 0.05,
) -> list[tuple[str, float, int, int, int, int]]:
    """Convenience helper to fuse UI detection tuples: [(cls_name, conf, x1, y1, x2, y2)].

    Returns:
        fused_detections: list of (cls_name, conf, x1, y1, x2, y2) sorted by confidence.
    """
    name_to_idx = {name: idx for idx, name in enumerate(class_names)}
    boxes_list = []
    scores_list = []
    labels_list = []

    for det_list in detection_lists:
        b_m = []
        s_m = []
        l_m = []
        for item in det_list:
            cls_name, conf, x1, y1, x2, y2 = item
            cls_idx = name_to_idx.get(cls_name, 0)
            b_m.append([float(x1), float(y1), float(x2), float(y2)])
            s_m.append(float(conf))
            l_m.append(int(cls_idx))

        boxes_list.append(np.array(b_m, dtype=np.float32) if b_m else np.empty((0, 4)))
        scores_list.append(np.array(s_m, dtype=np.float32) if s_m else np.empty((0,)))
        labels_list.append(np.array(l_m, dtype=int) if l_m else np.empty((0,), dtype=int))

    if per_class_weights:
        fb, fs, fl = per_class_weighted_boxes_fusion(
            boxes_list=boxes_list,
            scores_list=scores_list,
            labels_list=labels_list,
            per_class_weights=per_class_weights,
            class_names=class_names,
            iou_thr=iou_thr,
            skip_box_thr=skip_box_thr,
        )
    else:
        fb, fs, fl = weighted_boxes_fusion(
            boxes_list=boxes_list,
            scores_list=scores_list,
            labels_list=labels_list,
            weights=weights,
            iou_thr=iou_thr,
            skip_box_thr=skip_box_thr,
        )

    fused_results: list[tuple[str, float, int, int, int, int]] = []
    for b, s, lbl in zip(fb, fs, fl, strict=False):
        cls_name = class_names[int(lbl)] if 0 <= int(lbl) < len(class_names) else f"Class_{lbl}"
        x1, y1, x2, y2 = [int(round(coord)) for coord in b]
        fused_results.append((cls_name, float(s), x1, y1, x2, y2))

    return fused_results
