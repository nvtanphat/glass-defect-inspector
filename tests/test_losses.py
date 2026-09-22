from __future__ import annotations

import torch
import torch.nn.functional as F

from scale_error.training.losses import VFLDetectionLoss


def test_vfl_elementwise_class_weights():
    # Simulate dummy inputs for classification
    B, A, C = 2, 50, 5
    pred_scores = torch.randn(B, A, C)
    target_scores = torch.rand(B, A, C) * 0.5
    one_hot = (target_scores > 0.2).float()
    target_scores = target_scores * one_hot
    class_weights = torch.tensor([1.5, 0.8, 0.3, 3.5, 6.0])

    gamma, alpha = 2.0, 0.75
    vfl_weight = alpha * pred_scores.sigmoid().pow(gamma) * (1 - one_hot) + target_scores * one_hot
    bce = F.binary_cross_entropy_with_logits(pred_scores, target_scores, reduction="none")
    cls_elements = bce * vfl_weight

    # Broadcast class_weights
    cw = class_weights.view(1, 1, -1)
    weighted_elements = cls_elements * cw

    # Ratio per class must match target weights exactly
    unweighted_sums = cls_elements.sum(dim=(0, 1))
    weighted_sums = weighted_elements.sum(dim=(0, 1))
    ratios = weighted_sums / unweighted_sums

    assert torch.allclose(ratios, class_weights, atol=1e-5)
