"""Khối Lọc và Hợp nhất Đặc trưng Đa tầng Dày đặc
(Dense Cross-scale Feature Screening / Fusion Block - DCFSB).

Thiết kế lấy cảm hứng từ FBDNet, đặt tại các tầng hợp nhất đa tỉ lệ FPN/PAN.
Mục tiêu cốt lõi:
- Ngăn ngừa tín hiệu khuyết tật cực nhỏ (DiVat, DiVatLoiLom) bị lấn át hoặc dập tắt bởi các
  đặc trưng nền có độ tương phản cao trong quá trình kết hợp đặc trưng đa tầng.
- Trích xuất độ tương phản cục bộ qua unsharp masking.
- Khám phá trường tiếp nhận đa kích thước qua tích chập giãn (dilated depthwise convolutions).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DCFSB(nn.Module):
    """Khối lọc và sàng lọc đặc trưng đa tỉ lệ dày đặc (DCFSB).

    Tham số:
        channels (int): Số chiều đặc trưng ẩn (thường là hidden_dim = 256).
    """

    def __init__(self, channels: int = 256):
        super().__init__()
        self.channels = channels

        # 1. Trích xuất độ tương phản cục bộ (Unsharp Masking Difference):
        # Lấy hiệu số giữa đặc trưng gốc và giá trị trung bình lân cận 3x3 để làm bật cạnh vi mô
        self.contrast_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)

        # 2. Các nhánh tích chập lọc đa trường tiếp nhận (Multi-receptive field screening):
        # Tích chập depthwise với hệ số giãn dilation=1 và dilation=2
        self.dw_conv1 = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, dilation=1, groups=channels, bias=False
        )
        self.dw_conv2 = nn.Conv2d(
            channels, channels, kernel_size=3, padding=2, dilation=2, groups=channels, bias=False
        )
        self.norm = nn.BatchNorm2d(channels)
        self.act = nn.SiLU()

        # 3. Cổng lựa chọn thích ứng (Adaptive Screening Gate):
        # Ức chế các đặc trưng nền chiếm ưu thế và khuếch đại các bất thường khuyết tật tinh vi
        self.screening_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, kernel_size=1, bias=False),
            nn.SiLU(),
            nn.Conv2d(channels // 4, channels, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

        # 4. Phép chiếu đặc trưng đầu ra (Output projection)
        self.out_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.out_norm = nn.BatchNorm2d(channels)

        # 5. Hệ số tỷ lệ phần dư học được (Khởi tạo bằng 0 để ổn định transfer learning)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Bước 1: Trích xuất độ tương phản tần số cao cục bộ
        local_mean = self.contrast_pool(x)
        contrast = x - local_mean

        # Bước 2: Xử lý đa trường tiếp nhận trên các đặc trưng tăng cường độ tương phản
        feat1 = self.dw_conv1(x + contrast)
        feat2 = self.dw_conv2(x + contrast)
        screened_feat = self.act(self.norm(feat1 + feat2))

        # Bước 3: Sàng lọc kênh thích ứng thông qua cổng gating
        gate = self.screening_gate(screened_feat)
        screened = screened_feat * gate

        # Bước 4: Chiếu chuẩn hóa và cộng phần dư
        out = self.out_norm(self.out_proj(screened))
        return x + self.gamma * out
