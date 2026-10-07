"""HGNetv2 with Adaptive Frequency-domain / Target Attention (HGNetv2FBD).

Extends D-FINE's HGNetv2 backbone by placing AFTA modules at stage output features
[C3: 384, C4: 768, C5: 1536] to enhance weak defect features in industrial glass bottles.
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
    """HGNetv2 backbone with optional AFTA stage enhancements."""

    def __init__(
        self,
        *args,
        use_afta: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.use_afta = use_afta

        # Return channels for B2: stage 1 (384), stage 2 (768), stage 3 (1536)
        # Using self._out_channels according to return_idx
        out_channels = [self._out_channels[i] for i in self.return_idx]
        if self.use_afta:
            self.afta = nn.ModuleList([AFTA(ch) for ch in out_channels])
        else:
            self.afta = None

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        feats = super().forward(x)

        if self.use_afta and self.afta is not None:
            feats = [self.afta[i](f) for i, f in enumerate(feats)]

        return feats
