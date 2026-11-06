"""Mô-đun Hợp nhất Tương tác Không gian - Tần số (Spatial-Frequency Interaction Fusion - SFIF).

Thiết kế lấy cảm hứng từ FBDNet, tích hợp ngay sau tầng chiếu đầu vào (hidden_dim=256).
Mô-đun kết hợp:
- Nhánh không gian (Spatial branch): Tích chập cục bộ nắm bắt hình thái biên cạnh.
- Nhánh tần số (Frequency branch): Biến đổi Fourier 2D (2D-FFT) phân tích phổ tần số
  để phân tách tín hiệu khuyết tật cực nhỏ khỏi nhiễu nền tần số cao.
- Cổng tương tác chéo (Interactive cross-gating): Điều chế hai chiều giữa không gian và tần số.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SFIF(nn.Module):
    """Mô-đun hợp nhất tương tác miền không gian và miền tần số (SFIF).

    Tham số:
        channels (int): Số chiều đặc trưng (thường là hidden_dim = 256).
    """

    def __init__(self, channels: int = 256):
        super().__init__()
        self.channels = channels

        # 1. Nhánh không gian (Tích chập depthwise 3x3 + pointwise 1x1)
        self.spatial_branch = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )

        # 2. Nhánh tần số: Tích chập 1x1 trên biểu diễn phổ phức
        # (Phần thực + Phần ảo nối kênh thành 2*channels)
        self.freq_conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels * 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels * 2),
            nn.SiLU(),
            nn.Conv2d(channels * 2, channels * 2, kernel_size=1, bias=False),
        )
        self.freq_norm = nn.BatchNorm2d(channels)

        # 3. Cổng điều chế tương tác chéo (Interactive Cross-Gating)
        # Không gian điều khiển tần số (Spatial-to-Frequency Gate)
        self.gate_s2f = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, kernel_size=1, bias=False),
            nn.SiLU(),
            nn.Conv2d(channels // 4, channels, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )
        # Tần số điều khiển không gian (Frequency-to-Spatial Gate)
        self.gate_f2s = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, kernel_size=1, bias=False),
            nn.SiLU(),
            nn.Conv2d(channels // 4, channels, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

        # 4. Tầng chiếu hợp nhất đặc trưng (Fusion Projection)
        self.fuse_proj = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(),
        )

        # 5. Hệ số tỷ lệ phần dư học được (Khởi tạo bằng 0 để ổn định transfer learning)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        orig_dtype = x.dtype

        # --- Nhánh không gian (Spatial Branch) ---
        x_s = self.spatial_branch(x)

        # --- Nhánh tần số qua biến đổi 2D Real FFT ---
        # cuFFT yêu cầu float32 đối với các kích thước không phải lũy thừa của 2
        # (ví dụ: 120, 60, 30)
        with torch.amp.autocast(device_type=x.device.type if x.is_cuda else "cpu", enabled=False):
            x_fp32 = x.float()
            x_fft = torch.fft.rfft2(x_fp32, norm="ortho")
            x_cat = torch.cat([x_fft.real, x_fft.imag], dim=1)

            x_freq_cat = self.freq_conv(x_cat)
            x_real_out, x_imag_out = torch.chunk(x_freq_cat, 2, dim=1)
            x_fft_out = torch.complex(x_real_out, x_imag_out)

            # Biến đổi ngược Fourier (Inverse FFT) về lại miền không gian
            x_f = torch.fft.irfft2(x_fft_out, s=(h, w), norm="ortho")
            x_f = self.freq_norm(x_f).to(orig_dtype)

        # --- Cổng tương tác chéo (Cross-Gating) ---
        g_s = self.gate_s2f(x_s)
        g_f = self.gate_f2s(x_f)

        f_sf = x_f * g_s
        f_fs = x_s * g_f

        # --- Hợp nhất hai miền đặc trưng và cộng phần dư ---
        fused = self.fuse_proj(torch.cat([f_sf, f_fs], dim=1))

        return x + self.gamma * fused
