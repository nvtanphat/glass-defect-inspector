"""Adaptive Frequency-domain / Target Attention (AFTA) Module.

FBDNet-inspired adaptation for HGNetv2 backbone stage outputs.
Enhances weak small-defect signals and separates background texture
through adaptive channel and spatial frequency gating.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AFTA(nn.Module):
    """Adaptive Frequency-domain / Target Attention (AFTA).

    Args:
        channels: Number of input/output channels (e.g., 384, 768, 1536).
        reduction: Channel reduction ratio for MLP attention bottleneck.
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.channels = channels
        mid_channels = max(channels // reduction, 32)

        # 1. Frequency/Channel attention branch (dual pooling: mean + max)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, mid_channels, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(mid_channels, channels, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

        # 2. High-pass spatial edge/target attention branch
        # Depthwise 3x3 with Laplacian-inspired contour emphasis
        self.spatial_dw = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, groups=channels, bias=False
        )
        self.spatial_norm = nn.BatchNorm2d(channels)
        self.spatial_act = nn.GELU()
        self.spatial_pw = nn.Conv2d(channels, 1, kernel_size=1, bias=False)
        self.spatial_gate = nn.Sigmoid()

        # 3. Learnable residual scale (zero-initialized for stable transfer learning)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Channel modulation
        c_att = self.channel_gate(x)
        x_c = x * c_att

        # Spatial contour modulation
        s_feat = self.spatial_act(self.spatial_norm(self.spatial_dw(x_c)))
        s_att = self.spatial_gate(self.spatial_pw(s_feat))
        x_s = x_c * s_att

        # Residual update
        return x + self.gamma * x_s
