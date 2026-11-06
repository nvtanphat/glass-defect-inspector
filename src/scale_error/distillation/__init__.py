"""Mô-đun chưng cất tri thức (Knowledge Distillation) cho Glass Defect Inspector.

Chuyển giao tri thức từ cụm mô hình giáo viên đa cấu trúc (Triad Teacher: FBDNet + D-FINE + YOLO)
sang một mô hình học sinh YOLO26m duy nhất đạt tốc độ cao phục vụ sản xuất thực tế.
"""

from __future__ import annotations

from scale_error.distillation.consensus_target_generator import (
    ConsensusTargetGenerator,
    PerClassDistillationConfig,
)
from scale_error.distillation.losses import (
    ConfidenceWeightedVFLLoss,
    compute_smooth_class_weights,
)

__all__ = [
    "ConsensusTargetGenerator",
    "PerClassDistillationConfig",
    "ConfidenceWeightedVFLLoss",
    "compute_smooth_class_weights",
]
