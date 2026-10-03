"""Spatial-Frequency Interaction Fusion (SFIF) Module.

FBDNet-inspired adaptation placed after input projection (hidden_dim=256).
Fuses spatial convolution representations with 2D-FFT frequency components
to separate small defect signals from high-frequency background noise.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SFIF(nn.Module):
    """Spatial-Frequency Interaction Fusion.

    Args:
        channels: Feature dimension (typically hidden_dim = 256).
    """

    def __init__(self, channels: int = 256):
        super().__init__()
        self.channels = channels

        # 1. Spatial branch (depthwise + pointwise conv)
        self.spatial_branch = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )

        # 2. Frequency branch: 1x1 conv on complex frequency spectrum representation
        # Real + Imaginary treated as 2*channels
        self.freq_conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels * 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels * 2),
            nn.SiLU(),
            nn.Conv2d(channels * 2, channels * 2, kernel_size=1, bias=False),
        )
        self.freq_norm = nn.BatchNorm2d(channels)

        # 3. Interactive cross-gating
        self.gate_s2f = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.gate_f2s = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )

        # 4. Fusion projection
        self.fuse_proj = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(),
        )

        # 5. Residual scale initialized to zero for stable transfer learning
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        orig_dtype = x.dtype

        # --- Spatial branch ---
        x_s = self.spatial_branch(x)

        # --- Frequency branch (2D Real FFT) ---
        # cuFFT requires float32 for non-power-of-two dimensions (e.g. 120, 60, 30)
        with torch.amp.autocast(device_type=x.device.type if x.is_cuda else "cpu", enabled=False):
            x_fp32 = x.float()
            x_fft = torch.fft.rfft2(x_fp32, norm="ortho")
            x_cat = torch.cat([x_fft.real, x_fft.imag], dim=1)

            x_freq_cat = self.freq_conv(x_cat)
            x_real_out, x_imag_out = torch.chunk(x_freq_cat, 2, dim=1)
            x_fft_out = torch.complex(x_real_out, x_imag_out)

            # Inverse FFT back to spatial domain
            x_f = torch.fft.irfft2(x_fft_out, s=(h, w), norm="ortho")
            x_f = self.freq_norm(x_f).to(orig_dtype)

        # --- Interactive Cross-Gating ---
        g_s = self.gate_s2f(x_s)
        g_f = self.gate_f2s(x_f)

        f_sf = x_f * g_s
        f_fs = x_s * g_f

        # Fusion
        fused = self.fuse_proj(torch.cat([f_sf, f_fs], dim=1))

        return x + self.gamma * fused
