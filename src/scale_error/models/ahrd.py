"""Adaptive High-Resolution Detail Branch (AHRD) Module.

Integrated into RF-DETR-M / D-FINE-M for tiny glass defect detection.
Hypothesis:
    Tiny defect classes ('DiVat' and 'DiVatLoiLom') lose critical spatial information
    at stride-8 and deeper feature maps. However, shallow stride-4 features of glass
    contain strong non-defect background noise (reflections, mold seams, optical glare).
    AHRD introduces a semantic-guided gate:
        G = sigmoid(Conv([D, Up(S)]))
        D_gated = G * D
    to selectively filter glass background noise while retaining sub-pixel defect edges,
    and fusions D_gated into the high-resolution feature P3 via residual connection
    with learnable alpha (initialized to 0.0 for strict identity preservation).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["AHRDModule"]


class AHRDModule(nn.Module):
    """Adaptive High-Resolution Detail Branch (AHRD).

    Formulation:
        D: Stride-4 shallow feature [B, C_d, H/4, W/4] from backbone stage 1.
        S: Stride-8 deep semantic feature [B, C_s, H/8, W/8] from backbone stage 2.
        Up(S): Bilinear upsampling of S to match D's spatial resolution [B, C_s, H/4, W/4].
        G = sigmoid(Conv([D, Up(S)])): Semantic-guided gating mask [B, C_d, H/4, W/4].
        D_gated = G * D: Filtered detail feature.
        D_proj = Downsample(D_gated): Lightweight stride-2 projection [B, C_s, H/8, W/8].
        P3_fused = S + alpha * D_proj: Residual fusion with learnable alpha (init=0.0).
    """

    def __init__(
        self,
        in_channels_d: int = 96,
        in_channels_s: int = 384,
        out_channels: int = 384,
    ):
        super().__init__()
        self.in_channels_d = in_channels_d
        self.in_channels_s = in_channels_s
        self.out_channels = out_channels

        # G = sigmoid(Conv([D, Up(S)]))
        self.gate_conv = nn.Sequential(
            nn.Conv2d(
                in_channels_d + in_channels_s,
                in_channels_d,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels_d),
            nn.Sigmoid(),
        )

        # Lightweight stride-2 depthwise-separable downsampling projection
        self.downsample = nn.Sequential(
            nn.Conv2d(
                in_channels_d,
                in_channels_d,
                kernel_size=3,
                stride=2,
                padding=1,
                groups=in_channels_d,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels_d),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels_d, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        # Learnable residual scale initialized to 0.0
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, d: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """Forward pass of AHRD module.

        Args:
            d: Stride-4 shallow feature [B, C_d, H/4, W/4]
            s: Stride-8 semantic feature [B, C_s, H/8, W/8]

        Returns:
            fused_s: Fused feature matching s shape [B, C_s, H/8, W/8]
        """
        up_s = F.interpolate(s, size=d.shape[2:], mode="bilinear", align_corners=False)
        cat_feat = torch.cat([d, up_s], dim=1)
        g = self.gate_conv(cat_feat)
        d_gated = g * d
        d_proj = self.downsample(d_gated)
        return s + self.alpha * d_proj
