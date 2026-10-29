"""Ensemble modules for Glass Defect Inspector (scale-error-yolo26).

Provides Weighted Boxes Fusion (WBF), per-class adaptive weighting,
score calibration, and multi-model ensemble pipelines.
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
