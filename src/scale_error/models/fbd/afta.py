"""Mô-đun Chú ý Tần số Thích ứng / Mục tiêu (Adaptive Frequency-domain / Target Attention - AFTA).

Thiết kế lấy cảm hứng từ FBDNet, tích hợp tại các tầng đặc trưng đầu ra của backbone HGNetv2.
Mô-đun có nhiệm vụ:
- Tăng cường tín hiệu biên của các khuyết tật cực nhỏ hoặc khuyết tật dạng sợi mảnh
  (LoiChi, LoiTray).
- Tách biệt đặc trưng khuyết tật khỏi vân nền và nhiễu quang học bề mặt thủy tinh.
- Điều chế đồng thời qua cổng kênh (Channel Attention) và cổng không gian (Spatial Contour Gate).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AFTA(nn.Module):
    """Mô-đun chú ý thích ứng miền tần số và mục tiêu (AFTA).

    Tham số:
        channels (int): Số kênh đầu vào và đầu ra (ví dụ: 384, 768, 1536
            ứng với các stage backbone).
        reduction (int): Tỷ lệ thu giảm số kênh trong khối MLP chú ý kênh (mặc định: 16).
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.channels = channels
        mid_channels = max(channels // reduction, 32)

        # 1. Nhánh chú ý kênh/tần số (Channel Attention Branch):
        # Trích xuất trọng số tầm quan trọng của từng kênh đặc trưng qua pooling trung bình toàn cục
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, mid_channels, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(mid_channels, channels, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

        # 2. Nhánh chú ý không gian làm nổi bật đường biên/mục tiêu (Spatial Edge Attention):
        # Sử dụng tích chập Depthwise 3x3 với hàm kích hoạt GELU để khuếch đại
        # đường viền biên dạng Laplacian
        self.spatial_dw = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, groups=channels, bias=False
        )
        self.spatial_norm = nn.BatchNorm2d(channels)
        self.spatial_act = nn.GELU()
        self.spatial_pw = nn.Conv2d(channels, 1, kernel_size=1, bias=False)
        self.spatial_gate = nn.Sigmoid()

        # 3. Hệ số tỷ lệ phần dư học được (Khởi tạo bằng 0 để ổn định giai đoạn đầu
        # của transfer learning)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Bước 1: Điều chế đặc trưng theo kênh (Channel modulation)
        c_att = self.channel_gate(x)
        x_c = x * c_att

        # Bước 2: Điều chế làm nổi bật đường viền không gian (Spatial contour modulation)
        s_feat = self.spatial_act(self.spatial_norm(self.spatial_dw(x_c)))
        s_att = self.spatial_gate(self.spatial_pw(s_feat))
        x_s = x_c * s_att

        # Bước 3: Cộng nối phần dư với hệ số gamma học được (Residual update)
        return x + self.gamma * x_s
