"""Scientific Experiment Modules for D-FINE Glass Defect Detection.

Modules:
- E1: P2FusionModule (High-resolution shallow spatial feature fusion)
- E2: FrequencyAwareFusionModule (Local high-frequency decomposition & gating)
- E3: P2FrequencyFusionModule (Unified P2 + Frequency-aware feature encoder)
- E4: LongTailBalancedSampler (Annotation-driven instance & image balanced sampling)

Designed to be lightweight, mathematically sound, with identity initialization
(alpha=0) to prevent early training disruption of pretrained weights.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Sampler


class P2FusionModule(nn.Module):
    """E1: Lightweight P2-to-P3 Fusion Module.

    Hypothesis:
        P3 (stride 8, 120x120 @960) loses sub-pixel spatial detail critical
        for tiny glass scratches (LoiTray) and cracks (LoiChi).
        P2 (stride 4, 240x240, 96 ch) retains fine boundaries.

    Architecture:
        - Downsamples P2 to stride 8 using Depthwise-Separable Conv (k=3, s=2).
        - Projects P2 channels (96 -> 384) to match P3.
        - Gated fusion with learnable alpha (init=0.0).
        - Preserves exact 384-channel output for downstream HybridEncoder compatibility.
    """

    def __init__(self, in_p2: int = 96, in_p3: int = 384, out_channels: int = 384):
        super().__init__()
        self.in_p2 = in_p2
        self.in_p3 = in_p3
        self.out_channels = out_channels

        # Lightweight stride-2 depthwise separable convolution
        self.downsample = nn.Sequential(
            nn.Conv2d(in_p2, in_p2, kernel_size=3, stride=2, padding=1, groups=in_p2, bias=False),
            nn.BatchNorm2d(in_p2),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_p2, in_p3, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(in_p3),
        )

        # Gated fusion to learn which spatial details from P2 should be preserved
        self.gate = nn.Sequential(
            nn.Conv2d(in_p3 * 2, in_p3, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_p3),
            nn.Sigmoid(),
        )

        self.proj_out = nn.Sequential(
            nn.Conv2d(in_p3, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

        # Learnable residual weight initialized to 0.0 (identity at start)
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, p2: torch.Tensor, p3: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            p2: Tensor of shape [B, 96, H/4, W/4]
            p3: Tensor of shape [B, 384, H/8, W/8]

        Returns:
            fused_p3: Tensor of shape [B, 384, H/8, W/8]
        """
        # Downsample P2 to match P3 resolution: [B, 96, H/4, W/4] -> [B, 384, H/8, W/8]
        p2_ds = self.downsample(p2)

        # Gated fusion
        g = self.gate(torch.cat([p3, p2_ds], dim=1))
        fused = p3 + self.alpha * (g * p2_ds)
        return self.proj_out(fused)


class FrequencyAwareFusionModule(nn.Module):
    """E2: Lightweight Frequency-Aware Feature Refinement.

    Hypothesis:
        Glass bottle images contain pronounced high-frequency non-defect signals
        (reflections, illumination glares, molded ring edges). Both scratches
        and cracks are also high-frequency.
        A frequency-aware gate selectively attenuates background glass high frequencies
        while enhancing defect edges.

    Mechanism:
        low = LocalLowPass(x) (via 3x3 average pooling)
        high = x - low
        gate = Sigmoid(Conv1x1(concat([x, high])))
        out = x + alpha * (gate * high)
    """

    def __init__(self, channels: int = 384, kernel_size: int = 3):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.pad = kernel_size // 2

        self.gate_conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid(),
        )

        # Learnable alpha initialized to 0.0
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Decompose x into low/high frequency and apply gated residual."""
        low = F.avg_pool2d(x, kernel_size=self.kernel_size, stride=1, padding=self.pad)
        high = x - low
        gate = self.gate_conv(torch.cat([x, high], dim=1))
        return x + self.alpha * (gate * high)


class P2FrequencyFusionModule(nn.Module):
    """E3: Combined P2 High-Resolution and Frequency-Aware Refinement.

    Hypothesis:
        P2 injects fine boundary details, and Frequency-Aware gating separates
        the injected defect details from glass background artifacts.
    """

    def __init__(self, in_p2: int = 96, in_p3: int = 384, out_channels: int = 384):
        super().__init__()
        self.p2_fusion = P2FusionModule(in_p2=in_p2, in_p3=in_p3, out_channels=out_channels)
        self.freq_module = FrequencyAwareFusionModule(channels=out_channels)

    def forward(self, p2: torch.Tensor, p3: torch.Tensor) -> torch.Tensor:
        fused = self.p2_fusion(p2, p3)
        return self.freq_module(fused)


class LongTailBalancedSampler(Sampler[int]):
    """E4: Long-Tail Balanced Sampler for Object Detection.

    Hypothesis:
        Severe class imbalance (e.g. LoiChi >> LoiTray/LoiNhua) causes gradients
        to be dominated by majority classes.
        Sampling images with probability proportional to the inverse class
        frequency of the rarest instance in the image improves minority class recall
        without duplicating images on disk or increasing steps_per_epoch.

    Formula:
        Instance frequency: N_c = sum_{i} count(c in image_i)
        Class weight: w_c = (max_k N_k / (N_c + 1)) ** gamma  (gamma in [0.2, 0.4])
        Image weight: W_i = max_{c in image_i} w_c (smoothed)
        P(image_i) = W_i / sum(W)
    """

    def __init__(
        self,
        image_annotations: list[list[int]],
        num_classes: int = 5,
        gamma: float = 0.25,
        replacement: bool = True,
        seed: int = 42,
    ):
        """Initialize the balanced sampler.

        Args:
            image_annotations: List of list of category IDs for each image index.
                E.g. [[2, 2], [0], [4], ...]
            num_classes: Number of distinct classes.
            gamma: Exponent smoothing parameter (0.0 = uniform, 1.0 = full inverse freq).
            replacement: Whether to sample with replacement.
            seed: Random seed.
        """
        self.num_samples = len(image_annotations)
        self.replacement = replacement
        self.seed = seed

        # 1. Compute instance frequency per class
        class_counts = np.zeros(num_classes, dtype=np.int64)
        for labels in image_annotations:
            for l in labels:
                if 0 <= l < num_classes:
                    class_counts[l] += 1

        max_count = max(class_counts.max(), 1)
        class_weights = np.power(max_count / (class_counts + 1.0), gamma)

        # 2. Assign weight to each image based on its rarest object
        image_weights = np.zeros(self.num_samples, dtype=np.float64)
        for i, labels in enumerate(image_annotations):
            if len(labels) == 0:
                image_weights[i] = 1.0
            else:
                image_weights[i] = max(class_weights[l] for l in labels if 0 <= l < num_classes)

        # Normalize to probability distribution
        self.probabilities = image_weights / image_weights.sum()
        self.class_counts = class_counts.tolist()
        self.class_weights = class_weights.tolist()

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        indices = torch.multinomial(
            torch.from_numpy(self.probabilities).float(),
            num_samples=self.num_samples,
            replacement=self.replacement,
            generator=generator,
        ).tolist()
        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples
