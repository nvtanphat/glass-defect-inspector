"""Các mô-đun thử nghiệm khoa học cho phát hiện khuyết tật kính D-FINE / FBDNet.

Bao gồm:
- E1: P2FusionModule (Hợp nhất đặc trưng không gian nông độ phân giải cao P2 sang P3)
- E2: FrequencyAwareFusionModule (Phân tách tần số cao cục bộ và điều chế qua cổng)
- E3: P2FrequencyFusionModule (Bộ mã hóa tích hợp P2 và đặc trưng nhận biết tần số)
- E4: LongTailBalancedSampler (Bộ lấy mẫu cân bằng theo phân bố nhãn thiểu số long-tail)

Thiết kế gọn nhẹ, vững chắc về mặt toán học, khởi tạo dạng đồng nhất (alpha=0.0)
để không gây xáo trộn trọng số tiền huấn luyện trong các epoch đầu.
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
    """E1: Mô-đun hợp nhất nhẹ nhàng từ P2 sang P3.

    Giả thuyết:
        P3 (stride 8, kích thước 120x120 tại ảnh 960) bị mất chi tiết không gian
        sub-pixel quan trọng cho các vết trầy xước mảnh (LoiTray) và vết nứt sợi (LoiChi).
        P2 (stride 4, 240x240, 96 kênh) giữ trọn vẹn các đường biên vi mô.

    Kiến trúc:
        - Giảm mẫu P2 về stride 8 bằng tích chập Depthwise-Separable (k=3, s=2).
        - Chiếu số kênh P2 từ 96 lên 384 để khớp với P3.
        - Hợp nhất có cổng gating với hệ số alpha học được (khởi tạo 0.0).
    """

    def __init__(self, in_p2: int = 96, in_p3: int = 384, out_channels: int = 384):
        super().__init__()
        self.in_p2 = in_p2
        self.in_p3 = in_p3
        self.out_channels = out_channels

        # Tích chập Depthwise-Separable stride-2 giảm mẫu nhẹ
        self.downsample = nn.Sequential(
            nn.Conv2d(in_p2, in_p2, kernel_size=3, stride=2, padding=1, groups=in_p2, bias=False),
            nn.BatchNorm2d(in_p2),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_p2, in_p3, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(in_p3),
        )

        # Cổng điều khiển hợp nhất để học chi tiết không gian nào từ P2 cần được giữ lại
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

        # Hệ số phần dư khởi tạo 0.0 (tương đương ánh xạ đồng nhất ở đầu quá trình huấn luyện)
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, p2: torch.Tensor, p3: torch.Tensor) -> torch.Tensor:
        """Lan truyền tiến: Hợp nhất P2 vào P3."""
        # Giảm mẫu P2 để khớp kích thước phân giải của P3: [B, 96, H/4, W/4] -> [B, 384, H/8, W/8]
        p2_ds = self.downsample(p2)

        # Hợp nhất qua cổng gating
        g = self.gate(torch.cat([p3, p2_ds], dim=1))
        fused = p3 + self.alpha * (g * p2_ds)
        return self.proj_out(fused)


class FrequencyAwareFusionModule(nn.Module):
    """E2: Mô-đun tinh chỉnh đặc trưng nhận biết miền tần số (Frequency-Aware Feature Refinement).

    Giả thuyết:
        Ảnh bề mặt chai thủy tinh chứa các tín hiệu tần số cao không phải lỗi
        (vết chói, phản chiếu, vân gá). Vết trầy và sợi chỉ cũng là tín hiệu tần số cao.
        Cổng nhận biết tần số sẽ chọn lọc triệt tiêu nhiễu nền thủy tinh trong khi
        khuếch đại đường biên khuyết tật.
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

        # Hệ số alpha học được khởi tạo 0.0
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Phân rã x thành thành phần tần số thấp / cao và áp dụng cổng phần dư."""
        low = F.avg_pool2d(x, kernel_size=self.kernel_size, stride=1, padding=self.pad)
        high = x - low
        gate = self.gate_conv(torch.cat([x, high], dim=1))
        return x + self.alpha * (gate * high)


class P2FrequencyFusionModule(nn.Module):
    """E3: Kết hợp hợp nhất P2 độ phân giải cao và tinh chỉnh nhận biết tần số."""

    def __init__(self, in_p2: int = 96, in_p3: int = 384, out_channels: int = 384):
        super().__init__()
        self.p2_fusion = P2FusionModule(in_p2=in_p2, in_p3=in_p3, out_channels=out_channels)
        self.freq_module = FrequencyAwareFusionModule(channels=out_channels)

    def forward(self, p2: torch.Tensor, p3: torch.Tensor) -> torch.Tensor:
        fused = self.p2_fusion(p2, p3)
        return self.freq_module(fused)


class LongTailBalancedSampler(Sampler[int]):
    """E4: Bộ lấy mẫu cân bằng phân bố đuôi dài (Long-Tail Balanced Sampler).

    Giả thuyết:
        Sự mất cân bằng lớp trầm trọng (LoiChi >> LoiTray/LoiNhua) khiến gradient
        bị chi phối bởi các lớp đa số.
        Lấy mẫu ảnh với xác suất tỷ lệ thuận với nghịch đảo tần suất của vật thể hiếm nhất trong ảnh
        giúp tăng cường độ thu hồi cho lớp thiểu số mà không cần nhân bản file ảnh trên đĩa.
    """

    def __init__(
        self,
        image_annotations: list[list[int]],
        num_classes: int = 5,
        gamma: float = 0.25,
        replacement: bool = True,
        seed: int = 42,
    ):
        """Khởi tạo bộ lấy mẫu cân bằng.

        Tham số:
            image_annotations: Danh sách danh sách ID phân lớp cho từng ảnh index.
            num_classes: Số lượng phân lớp (5).
            gamma: Tham số làm mịn số mũ (0.0: phân bố đều, 1.0: nghịch đảo hoàn toàn).
            replacement: Có hoàn lại sau khi lấy mẫu hay không.
            seed: Hạt giống ngẫu nhiên.
        """
        self.num_samples = len(image_annotations)
        self.replacement = replacement
        self.seed = seed

        # 1. Tính tần suất xuất hiện instance của từng phân lớp
        class_counts = np.zeros(num_classes, dtype=np.int64)
        for labels in image_annotations:
            for l in labels:
                if 0 <= l < num_classes:
                    class_counts[l] += 1

        max_count = max(class_counts.max(), 1)
        class_weights = np.power(max_count / (class_counts + 1.0), gamma)

        # 2. Gán trọng số cho từng ảnh dựa trên lớp hiếm nhất hiện diện trong ảnh
        image_weights = np.zeros(self.num_samples, dtype=np.float64)
        for i, labels in enumerate(image_annotations):
            if len(labels) == 0:
                image_weights[i] = 1.0
            else:
                image_weights[i] = max(class_weights[l] for l in labels if 0 <= l < num_classes)

        # Chuẩn hóa thành phân phối xác suất
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
