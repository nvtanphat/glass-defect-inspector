"""Scale-Aware Normalized Wasserstein Distance (NWD) Auxiliary Loss.

Implements Gaussian Wasserstein-2 distance with scale-adaptive weighting
specifically designed for small defect detection in industrial glass bottles.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn


class ScaleAwareNWDLoss(nn.Module):
    """Scale-Aware Normalized Wasserstein Distance (NWD) Auxiliary Loss.

    Computes the 2-Wasserstein distance between 2D Gaussian distributions
    modeling predicted and ground-truth bounding boxes, normalized by a constant C,
    and applies a scale-aware weighting factor to prioritize small defects.

    Args:
        c (float): Normalization constant in pixels. Default: 16.0.
        scale_aware (bool): Whether to apply scale-aware weighting. Default: True.
        tau (float): Characteristic defect scale threshold in pixels. Default: 32.0.
        alpha_scale (float): Weight modulation factor for small defects. Default: 1.0.
        min_weight (float): Minimum scale weight clamp. Default: 0.5.
        max_weight (float): Maximum scale weight clamp. Default: 2.0.
        img_size (tuple or list): Image (height, width) for unnormalizing boxes. Default: (960, 960).
        eps (float): Numerical stability epsilon. Default: 1e-7.
    """

    def __init__(
        self,
        c: float = 16.0,
        scale_aware: bool = True,
        tau: float = 32.0,
        alpha_scale: float = 1.0,
        min_weight: float = 0.5,
        max_weight: float = 2.0,
        img_size: tuple[int, int] | list[int] = (960, 960),
        eps: float = 1e-7,
    ):
        super().__init__()
        self.c = float(c)
        self.scale_aware = bool(scale_aware)
        self.tau = float(tau)
        self.alpha_scale = float(alpha_scale)
        self.min_weight = float(min_weight)
        self.max_weight = float(max_weight)
        self.img_size = tuple(img_size)
        self.eps = float(eps)

    def forward(
        self,
        pred_boxes: torch.Tensor,
        target_boxes: torch.Tensor,
        num_boxes: float | int | torch.Tensor = 1.0,
    ) -> torch.Tensor:
        """Compute scale-aware NWD loss between pred and target boxes.

        Args:
            pred_boxes: (N, 4) in cxcywh normalized format [0, 1].
            target_boxes: (N, 4) in cxcywh normalized format [0, 1].
            num_boxes: Normalization divisor (total target boxes across batch/world).

        Returns:
            Scalar tensor representing the scale-aware NWD loss.
        """
        if pred_boxes.numel() == 0 or target_boxes.numel() == 0:
            return pred_boxes.sum() * 0.0

        # Unnormalize to pixel coordinates (cx, cy, w, h)
        # img_size is (height, width)
        h_img, w_img = self.img_size[0], self.img_size[1]
        scale_factor = pred_boxes.new_tensor([w_img, h_img, w_img, h_img])

        pred_px = pred_boxes * scale_factor
        tgt_px = target_boxes * scale_factor

        cx_p, cy_p, w_p, h_p = pred_px.unbind(-1)
        cx_t, cy_t, w_t, h_t = tgt_px.unbind(-1)

        # 2D Gaussian Wasserstein-2 distance squared:
        # W_2^2 = ||mu_p - mu_t||^2 + ||sigma_p - sigma_t||_F^2
        # where sigma = diag(w/2, h/2)
        # ||sigma_p - sigma_t||_F^2 = 0.25 * ((w_p - w_t)^2 + (h_p - h_t)^2)
        center_dist_sq = (cx_p - cx_t).pow(2) + (cy_p - cy_t).pow(2)
        wh_dist_sq = 0.25 * ((w_p - w_t).pow(2) + (h_p - h_t).pow(2))
        w2_sq = center_dist_sq + wh_dist_sq

        # Clamped Wasserstein distance in pixels with smooth epsilon
        w_dist = torch.sqrt(torch.clamp(w2_sq, min=0.0) + self.eps)

        # Normalized Wasserstein Distance similarity in (0, 1]
        nwd = torch.exp(-w_dist / self.c)

        # Raw NWD loss: 1 - NWD
        # Subtract epsilon offset so that identical boxes produce exactly 0.0 loss
        zero_offset = 1.0 - math.exp(-math.sqrt(self.eps) / self.c)
        loss_nwd = torch.clamp(1.0 - nwd - zero_offset, min=0.0)

        # Scale-Aware Weighting
        if self.scale_aware:
            # Defect scale: geometric mean of target bounding box dimensions in pixels
            target_scale = torch.sqrt(torch.clamp(w_t * h_t, min=self.eps))
            # Exponential decay weight: tiny defects (scale << tau) receive higher attention
            scale_weights = 1.0 + self.alpha_scale * torch.exp(-target_scale / self.tau)
            scale_weights = torch.clamp(scale_weights, min=self.min_weight, max=self.max_weight)
            # Detach scale weights to prevent gradient flow through scale assignment
            loss_nwd = loss_nwd * scale_weights.detach()

        if isinstance(num_boxes, (int, float)):
            divisor = max(float(num_boxes), 1.0)
        else:
            divisor = torch.clamp(num_boxes, min=1.0)

        return loss_nwd.sum() / divisor
