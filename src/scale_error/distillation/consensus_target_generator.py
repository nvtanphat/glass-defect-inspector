"""Consensus target synthesis and sub-pixel boundary refinement for knowledge distillation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

CLASS_NAMES = ["DiVat", "DiVatLoiLom", "LoiChi", "LoiNhua", "LoiTray"]

# Per-class geometric clustering thresholds from optimal Apex Ensemble
DEFAULT_CLASS_IOU: dict[str, float] = {
    "DiVat": 0.55,         # Compact micro particles: prevent merging adjacent particles
    "DiVatLoiLom": 0.41,   # Optical refraction bubbles
    "LoiChi": 0.65,        # Elongated directional line bubbles: tight boundary fit
    "LoiNhua": 0.59,       # Soft-textured resin stain
    "LoiTray": 0.43,       # High aspect-ratio scratch: prevent false splitting
}

DEFAULT_CLASS_DISCOVERY_FLOOR: dict[str, float] = {
    "DiVat": 0.35,
    "DiVatLoiLom": 0.30,
    "LoiChi": 0.35,
    "LoiNhua": 0.35,
    "LoiTray": 0.30,
}


@dataclass
class PerClassDistillationConfig:
    """Per-class configuration for distillation target generation."""

    class_names: Sequence[str] = field(default_factory=lambda: CLASS_NAMES)
    iou_thresholds: dict[str, float] = field(default_factory=lambda: DEFAULT_CLASS_IOU)
    discovery_floors: dict[str, float] = field(
        default_factory=lambda: DEFAULT_CLASS_DISCOVERY_FLOOR
    )
    gt_match_threshold: float = 0.42
    solo_discovery_floor: float = 0.55
    subpixel_alpha: float = 0.50  # 0.5 * GT + 0.5 * Teacher


def compute_box_iou(box1: Sequence[float], box2: Sequence[float]) -> float:
    """Calculate IoU between two bounding boxes in [x1, y1, x2, y2] format."""
    x1 = max(float(box1[0]), float(box2[0]))
    y1 = max(float(box1[1]), float(box2[1]))
    x2 = min(float(box1[2]), float(box2[2]))
    y2 = min(float(box1[3]), float(box2[3]))

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    if inter <= 0.0:
        return 0.0

    a1 = max(0.0, float(box1[2]) - float(box1[0])) * max(0.0, float(box1[3]) - float(box1[1]))
    a2 = max(0.0, float(box2[2]) - float(box2[0])) * max(0.0, float(box2[3]) - float(box2[1]))
    union = a1 + a2 - inter
    return inter / union if union > 0.0 else 0.0


class ConsensusTargetGenerator:
    """Synthesizes high-fidelity distillation targets from multiple teacher models."""

    def __init__(self, config: PerClassDistillationConfig | None = None) -> None:
        self.config = config or PerClassDistillationConfig()

    def fuse_teacher_predictions(
        self,
        teacher_predictions_by_model: list[list[tuple[int, list[float], float]]],
        model_weights: list[float] | None = None,
    ) -> list[tuple[int, list[float], float, int]]:
        """Cluster and fuse predictions from multiple teacher models.

        Args:
            teacher_predictions_by_model: List of [(class_id, [x1, y1, x2, y2], conf)] per model.
            model_weights: Importance weight for each teacher model.

        Returns:
            List of (class_id, fused_box, consensus_score, agreement_count).
        """
        num_models = len(teacher_predictions_by_model)
        if num_models == 0:
            return []

        weights = model_weights or [1.0] * num_models

        # Group detections by class ID
        per_class_items: dict[int, list[tuple[int, np.ndarray, float]]] = {}
        for m_idx, det_list in enumerate(teacher_predictions_by_model):
            for c_id, box, score in det_list:
                per_class_items.setdefault(c_id, []).append(
                    (m_idx, np.array(box, dtype=np.float32), float(score))
                )

        fused_results: list[tuple[int, list[float], float, int]] = []

        for c_id, items in per_class_items.items():
            c_name = (
                self.config.class_names[c_id]
                if 0 <= c_id < len(self.config.class_names)
                else f"class_{c_id}"
            )
            iou_thr = self.config.iou_thresholds.get(c_name, 0.50)

            # Sort descending by score
            items.sort(key=lambda x: x[2], reverse=True)

            clusters: list[dict] = []
            for m_idx, b_arr, s in items:
                best_iou = iou_thr
                best_c_idx = -1
                for c_idx, cluster in enumerate(clusters):
                    iou = compute_box_iou(b_arr, cluster["center"])
                    if iou > best_iou:
                        best_iou = iou
                        best_c_idx = c_idx

                if best_c_idx >= 0:
                    clusters[best_c_idx]["items"].append((m_idx, b_arr, s))
                else:
                    clusters.append({
                        "center": b_arr.copy(),
                        "items": [(m_idx, b_arr, s)],
                    })

            for cluster in clusters:
                c_items = cluster["items"]
                distinct_models = len(set(item[0] for item in c_items))

                total_weight = 0.0
                sum_box = np.zeros(4, dtype=np.float32)
                sum_score = 0.0

                for m_idx, b_arr, s in c_items:
                    w = weights[m_idx]
                    total_weight += w
                    sum_box += w * s * b_arr
                    sum_score += w * s

                fused_box = (sum_box / max(sum_score, 1e-6)).tolist()
                base_score = sum_score / max(total_weight, 1e-6)
                # Multi-teacher consensus bonus
                multiplier = (distinct_models / float(num_models)) ** 0.35
                consensus_score = min(1.0, float(base_score * multiplier))

                fused_results.append((c_id, fused_box, consensus_score, distinct_models))

        return fused_results

    def refine_ground_truth(
        self,
        gt_boxes: list[dict],
        fused_teachers: list[tuple[int, list[float], float, int]],
        image_width: int,
        image_height: int,
    ) -> tuple[list[dict], dict[str, int]]:
        """Perform sub-pixel boundary refinement and consensus defect discovery.

        Args:
            gt_boxes: List of dicts with keys {"cls": int, "box": [x1, y1, x2, y2]}.
            fused_teachers: Output from fuse_teacher_predictions.
            image_width: Full image width in pixels.
            image_height: Full image height in pixels.

        Returns:
            (refined_targets, stats_dict)
        """
        stats = {
            "initial_gt": len(gt_boxes),
            "refined_gt": 0,
            "discovered": 0,
            "total_distilled": 0,
        }

        matched_teacher_indices = set()
        refined_targets: list[dict] = []

        # Step 1: Sub-pixel boundary refinement of annotated GT
        for gt in gt_boxes:
            c_id = gt["cls"]
            gt_b = gt["box"]
            c_name = (
                self.config.class_names[c_id]
                if 0 <= c_id < len(self.config.class_names)
                else f"class_{c_id}"
            )
            match_thr = self.config.iou_thresholds.get(c_name, self.config.gt_match_threshold)

            best_iou = match_thr
            best_t_idx = -1
            for t_idx, (t_cls, t_box, _t_score, _t_agree) in enumerate(fused_teachers):
                if t_cls == c_id:
                    iou = compute_box_iou(gt_b, t_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_t_idx = t_idx

            if best_t_idx >= 0:
                t_box = fused_teachers[best_t_idx][1]
                a = self.config.subpixel_alpha
                refined_b = [
                    (1.0 - a) * gt_b[0] + a * t_box[0],
                    (1.0 - a) * gt_b[1] + a * t_box[1],
                    (1.0 - a) * gt_b[2] + a * t_box[2],
                    (1.0 - a) * gt_b[3] + a * t_box[3],
                ]
                matched_teacher_indices.add(best_t_idx)
                stats["refined_gt"] += 1
                box_to_use = refined_b
                conf_score = fused_teachers[best_t_idx][2]
            else:
                box_to_use = gt_b
                conf_score = 1.0  # Ground truth default confidence

            refined_targets.append({
                "cls": c_id,
                "box": box_to_use,
                "score": float(conf_score),
                "is_discovered": False,
            })

        # Step 2: Teacher consensus discovery of unlabeled micro-defects
        for t_idx, (t_cls, t_box, t_score, t_agree) in enumerate(fused_teachers):
            if t_idx in matched_teacher_indices:
                continue

            c_name = (
                self.config.class_names[t_cls]
                if 0 <= t_cls < len(self.config.class_names)
                else f"class_{t_cls}"
            )
            discovery_floor = self.config.discovery_floors.get(c_name, 0.35)

            # Accept if consensus of >= 2 teachers exceeds floor, or solo teacher is high confidence
            is_valid_consensus = (t_agree >= 2 and t_score >= discovery_floor)
            is_high_conf_solo = (t_agree == 1 and t_score >= self.config.solo_discovery_floor)

            if is_valid_consensus or is_high_conf_solo:
                # Ensure box stays within image bounds
                x1 = max(0.0, min(float(t_box[0]), float(image_width)))
                y1 = max(0.0, min(float(t_box[1]), float(image_height)))
                x2 = max(0.0, min(float(t_box[2]), float(image_width)))
                y2 = max(0.0, min(float(t_box[3]), float(image_height)))

                if (x2 - x1) > 2.0 and (y2 - y1) > 2.0:
                    refined_targets.append({
                        "cls": t_cls,
                        "box": [x1, y1, x2, y2],
                        "score": float(t_score),
                        "is_discovered": True,
                    })
                    stats["discovered"] += 1

        stats["total_distilled"] = len(refined_targets)
        return refined_targets, stats

    def to_yolo_format(
        self,
        targets: list[dict],
        image_width: int,
        image_height: int,
    ) -> list[str]:
        """Convert pixel-coordinate targets into normalized YOLO text lines: `cls cx cy w h`."""
        lines = []
        for t in targets:
            c_id = t["cls"]
            x1, y1, x2, y2 = t["box"]
            cx = ((x1 + x2) / 2.0) / float(image_width)
            cy = ((y1 + y2) / 2.0) / float(image_height)
            bw = (x2 - x1) / float(image_width)
            bh = (y2 - y1) / float(image_height)

            if 0.0 < cx < 1.0 and 0.0 < cy < 1.0 and bw > 0.0005 and bh > 0.0005:
                lines.append(f"{c_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        return lines
