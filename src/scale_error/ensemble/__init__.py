"""Mô-đun tổ hợp mô hình (Ensemble) cho Glass Defect Inspector.

Cung cấp thuật toán Weighted Boxes Fusion (WBF), gán trọng số thích ứng theo từng phân lớp,
hiệu chuẩn điểm tin cậy và pipeline tổ hợp Heterogeneous Triad WBF
(FBDNet-M + D-FINE-M + YOLO26m TTA).
"""

from scale_error.ensemble.ensemble_pipeline import GlassDefectEnsemble
from scale_error.ensemble.wbf import (
    calibrate_scores,
    fuse_detection_lists,
    per_class_weighted_boxes_fusion,
    weighted_boxes_fusion,
)

__all__ = [
    "weighted_boxes_fusion",
    "per_class_weighted_boxes_fusion",
    "calibrate_scores",
    "fuse_detection_lists",
    "GlassDefectEnsemble",
]
