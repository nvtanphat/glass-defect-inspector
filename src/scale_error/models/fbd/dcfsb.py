"""Dense Cross-scale Feature Screening / Fusion Block (DCFSB).

FBDNet-inspired adaptation placed within the multi-scale FPN/PAN fusion stages.
Prevents weak small-defect signals from being suppressed or overwhelmed
by dominant high-contrast background features during multi-scale fusion.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DCFSB(nn.Module):
    """Dense Cross-scale Feature Screening / Fusion Block.

    Args:
        channels: Feature dimension (typically hidden_dim = 256).
    """

    def __init__(self, channels: int = 256):
        super().__init__()
        self.channels = channels

        # 1. Feature contrast extraction (unsharp masking difference)
        self.contrast_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)

        # 2. Multi-receptive field screening convolutions (dilated depthwise)
        self.dw_conv1 = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, dilation=1, groups=channels, bias=False
        )
        self.dw_conv2 = nn.Conv2d(
            channels, channels, kernel_size=3, padding=2, dilation=2, groups=channels, bias=False
        )
        self.norm = nn.BatchNorm2d(channels)
        self.act = nn.SiLU()

        # 3. Adaptive screening selection gate (suppress dominant, amplify subtle anomalies)
        self.screening_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, kernel_size=1, bias=False),
            nn.SiLU(),
            nn.Conv2d(channels // 4, channels, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

        # 4. Output projection
        self.out_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.out_norm = nn.BatchNorm2d(channels)

        # 5. Residual scale initialized to zero for stable transfer learning
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # High-frequency local contrast extraction
        local_mean = self.contrast_pool(x)
        contrast = x - local_mean

        # Multi-scale receptive processing on contrast-enhanced features
        feat1 = self.dw_conv1(x + contrast)
        feat2 = self.dw_conv2(x + contrast)
        screened_feat = self.act(self.norm(feat1 + feat2))

        # Adaptive channel screening
        gate = self.screening_gate(screened_feat)
        screened = screened_feat * gate

        out = self.out_norm(self.out_proj(screened))
        return x + self.gamma * out
