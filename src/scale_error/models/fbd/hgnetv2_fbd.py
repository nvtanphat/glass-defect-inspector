"""HGNetv2 kết hợp mô-đun Chú ý Tần số Thích ứng / Mục tiêu (HGNetv2FBD).

Mở rộng mạng backbone HGNetv2 của D-FINE bằng cách tích hợp các mô-đun AFTA
tại các tầng đặc trưng đầu ra [C3: 384, C4: 768, C5: 1536] nhằm tăng cường
tín hiệu đặc trưng khuyết tật yếu trên bề mặt chai thủy tinh công nghiệp.
"""

from __future__ import annotations

import torch
import torch.nn as nn

try:
    from ...core import register
    from .hgnetv2 import HGNetv2
    from ...zoo.dfine.fbd_modules.afta import AFTA
except Exception:
    try:
        from src.core import register
        from src.nn.backbone.hgnetv2 import HGNetv2
        from .afta import AFTA
    except Exception:

        def register():
            def deco(cls):
                return cls

            return deco

        from .afta import AFTA

        HGNetv2 = nn.Module

__all__ = ["HGNetv2FBD"]


@register()
class HGNetv2FBD(HGNetv2):
    """Mạng backbone HGNetv2 tăng cường với các khối AFTA theo từng stage."""

    def __init__(
        self,
        *args,
        use_afta: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.use_afta = use_afta

        # Số kênh đầu ra cho phiên bản B2: stage 1 (384), stage 2 (768), stage 3 (1536)
        out_channels = [self._out_channels[i] for i in self.return_idx]
        if self.use_afta:
            self.afta = nn.ModuleList([AFTA(ch) for ch in out_channels])
        else:
            self.afta = None

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        # Trích xuất đặc trưng đa tầng từ HGNetv2 tiêu chuẩn
        feats = super().forward(x)

        # Điều chế và tăng cường qua mô-đun AFTA
        if self.use_afta and self.afta is not None:
            feats = [self.afta[i](f) for i, f in enumerate(feats)]

        return feats
