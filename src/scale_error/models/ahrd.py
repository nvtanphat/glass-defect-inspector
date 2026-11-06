"""Mô-đun Nhánh Chi tiết Độ phân giải cao Thích ứng (Adaptive High-Resolution Detail Branch - AHRD).

Tích hợp vào RF-DETR-M / D-FINE-M nhằm cải thiện phát hiện khuyết tật cực nhỏ trên kính.
Giả thuyết thiết kế:
    Các phân lớp khuyết tật vi mô ('DiVat' và 'DiVatLoiLom') bị mất thông tin không gian quan trọng
    ở tầng đặc trưng stride-8 hoặc sâu hơn. Tuy nhiên, các đặc trưng nông stride-4 của thủy tinh
    lại chứa nhiều nhiễu nền phi khuyết tật (vết phản quang, viền khuôn gá, vết chói).
    AHRD đưa vào cơ chế cổng định hướng ngữ nghĩa (semantic-guided gate):
        G = sigmoid(Conv([D, Up(S)]))
        D_gated = G * D
    để lọc chọn lọc nhiễu nền thủy tinh trong khi giữ lại các cạnh khuyết tật mức sub-pixel,
    sau đó hợp nhất D_gated vào đặc trưng P3 độ phân giải cao thông qua kết nối phần dư
    với hệ số alpha học được.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["AHRDModule"]


class AHRDModule(nn.Module):
    """Mô-đun Nhánh Chi tiết Độ phân giải cao Thích ứng (AHRD).

    Công thức:
        D: Đặc trưng nông Stride-4 [B, C_d, H/4, W/4] từ backbone stage 1.
        S: Đặc trưng ngữ nghĩa sâu Stride-8 [B, C_s, H/8, W/8] từ backbone stage 2.
        Up(S): Nội suy song tuyến tính đưa S về cùng kích thước không gian với D.
        G = sigmoid(Conv([D, Up(S)])): Mặt nạ cổng lọc định hướng ngữ nghĩa.
        D_gated = G * D: Đặc trưng chi tiết đã lọc bỏ nhiễu nền.
        D_proj = Downsample(D_gated): Tầng chiếu giảm mẫu nhẹ nhàng stride-2.
        P3_fused = S + alpha * D_proj: Hợp nhất phần dư với hệ số alpha (khởi tạo 0.0).
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

        # Cổng lọc định hướng ngữ nghĩa: G = sigmoid(Conv([D, Up(S)]))
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

        # Tích chập tách biệt theo chiều sâu (Depthwise-separable) stride-2 giảm mẫu nhẹ
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

        # Hệ số phần dư học được (Khởi tạo 0.0 để bảo toàn trọng số pretrained ban đầu)
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, d: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """Lan truyền tiến của mô-đun AHRD.

        Tham số:
            d: Đặc trưng nông Stride-4 [B, C_d, H/4, W/4]
            s: Đặc trưng ngữ nghĩa Stride-8 [B, C_s, H/8, W/8]

        Trả về:
            Tensor đặc trưng hợp nhất có cùng kích thước với s [B, C_s, H/8, W/8]
        """
        up_s = F.interpolate(s, size=d.shape[2:], mode="bilinear", align_corners=False)
        cat_feat = torch.cat([d, up_s], dim=1)
        g = self.gate_conv(cat_feat)
        d_gated = g * d
        d_proj = self.downsample(d_gated)
        return s + self.alpha * d_proj
