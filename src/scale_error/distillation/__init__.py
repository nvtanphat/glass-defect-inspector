"""Distillation module for Glass Defect Inspector.

Transfers multi-teacher ensemble knowledge (FBDNet + D-FINE + YOLO)
into a single high-speed YOLO26 student model.
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
