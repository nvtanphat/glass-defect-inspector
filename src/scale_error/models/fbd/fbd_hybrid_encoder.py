"""FBDHybridEncoder: HybridEncoder with Spatial-Frequency Interaction Fusion (SFIF)
and Dense Cross-scale Feature Screening / Fusion Blocks (DCFSB).

FBDNet-inspired adaptation for D-FINE-M. Preserves all baseline architecture
while enabling independent toggling of SFIF and DCFSB modules for rigorous ablation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from ...core import register
    from .fbd_modules.dcfsb import DCFSB
    from .fbd_modules.sfif import SFIF
    from .hybrid_encoder import HybridEncoder
except Exception:
    try:
        from src.core import register
        from src.zoo.dfine.hybrid_encoder import HybridEncoder
        from .dcfsb import DCFSB
        from .sfif import SFIF
    except Exception:

        def register():
            def deco(cls):
                return cls

            return deco

        from .dcfsb import DCFSB
        from .sfif import SFIF

        HybridEncoder = nn.Module

__all__ = ["FBDHybridEncoder"]


@register()
class FBDHybridEncoder(HybridEncoder):
    """HybridEncoder extended with FBDNet-inspired SFIF and DCFSB modules."""

    def __init__(
        self,
        *args,
        use_sfif: bool = True,
        use_dcfsb: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.use_sfif = use_sfif
        self.use_dcfsb = use_dcfsb

        # 1. SFIF applied after input projection
        if use_sfif:
            self.sfif = nn.ModuleList([SFIF(self.hidden_dim) for _ in self.in_channels])
        else:
            self.sfif = None

        # 2. DCFSB applied inside multi-scale FPN and PAN fusion stages
        if use_dcfsb:
            num_fusions = len(self.in_channels) - 1
            self.dcfsb_fpn = nn.ModuleList([DCFSB(self.hidden_dim) for _ in range(num_fusions)])
            self.dcfsb_pan = nn.ModuleList([DCFSB(self.hidden_dim) for _ in range(num_fusions)])
        else:
            self.dcfsb_fpn = None
            self.dcfsb_pan = None

    def forward(self, feats: list[torch.Tensor]) -> list[torch.Tensor]:
        assert len(feats) == len(self.in_channels)

        # 1. Input projection (C3: 384, C4: 768, C5: 1536 -> hidden_dim: 256)
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]

        # 2. SFIF: Spatial-Frequency Interaction Fusion
        if self.use_sfif and self.sfif is not None:
            proj_feats = [self.sfif[i](feat) for i, feat in enumerate(proj_feats)]

        # 3. Transformer Encoder layers (as in baseline)
        if self.num_encoder_layers > 0:
            for i, enc_ind in enumerate(self.use_encoder_idx):
                h, w = proj_feats[enc_ind].shape[2:]
                src_flatten = proj_feats[enc_ind].flatten(2).permute(0, 2, 1)
                if self.training or self.eval_spatial_size is None:
                    pos_embed = self.build_2d_sincos_position_embedding(
                        w, h, self.hidden_dim, self.pe_temperature
                    ).to(src_flatten.device)
                else:
                    pos_embed = getattr(self, f"pos_embed{enc_ind}", None).to(src_flatten.device)

                memory: torch.Tensor = self.encoder[i](src_flatten, pos_embed=pos_embed)
                proj_feats[enc_ind] = (
                    memory.permute(0, 2, 1).reshape(-1, self.hidden_dim, h, w).contiguous()
                )

        # 4. Top-Down FPN Broadcasting & Fusion with DCFSB
        inner_outs = [proj_feats[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            block_idx = len(self.in_channels) - 1 - idx
            feat_heigh = inner_outs[0]
            feat_low = proj_feats[idx - 1]
            feat_heigh = self.lateral_convs[block_idx](feat_heigh)
            inner_outs[0] = feat_heigh
            upsample_feat = F.interpolate(feat_heigh, scale_factor=2.0, mode="nearest")
            inner_out = self.fpn_blocks[block_idx](torch.concat([upsample_feat, feat_low], dim=1))
            if self.use_dcfsb and self.dcfsb_fpn is not None:
                inner_out = self.dcfsb_fpn[block_idx](inner_out)
            inner_outs.insert(0, inner_out)

        # 5. Bottom-Up PAN Downsampling & Fusion with DCFSB
        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_height = inner_outs[idx + 1]
            downsample_feat = self.downsample_convs[idx](feat_low)
            out = self.pan_blocks[idx](torch.concat([downsample_feat, feat_height], dim=1))
            if self.use_dcfsb and self.dcfsb_pan is not None:
                out = self.dcfsb_pan[idx](out)
            outs.append(out)

        return outs
