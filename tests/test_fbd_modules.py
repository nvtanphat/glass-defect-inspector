"""Unit tests for FBDNet-inspired adaptation modules (AFTA, SFIF, DCFSB)
and their integration into HGNetv2FBD and FBDHybridEncoder.
"""

import sys
from pathlib import Path

# Stub broken tensorflow in local site-packages if present
import types as _types
from importlib.machinery import ModuleSpec as _ModuleSpec
_tf_stub = _types.ModuleType("tensorflow")
_tf_stub.__version__ = "stub"
_tf_stub.__spec__ = _ModuleSpec("tensorflow", loader=None)
_tf_stub.__path__ = []
_tf_stub.io = _types.SimpleNamespace(
    gfile=_types.SimpleNamespace(join=lambda *a: "/".join(str(x) for x in a))
)
sys.modules["tensorflow"] = _tf_stub

import pytest
import torch

# Ensure D-FINE is in python path
ROOT = Path(__file__).resolve().parents[1]
DFINE_ROOT = ROOT / "results" / "D-FINE"
if str(DFINE_ROOT) not in sys.path:
    sys.path.insert(0, str(DFINE_ROOT))

from src.zoo.dfine.fbd_modules.afta import AFTA
from src.zoo.dfine.fbd_modules.sfif import SFIF
from src.zoo.dfine.fbd_modules.dcfsb import DCFSB
from src.nn.backbone.hgnetv2_fbd import HGNetv2FBD
from src.zoo.dfine.fbd_hybrid_encoder import FBDHybridEncoder


def test_afta_preserves_shape():
    """Verify AFTA preserves input shape across all HGNetv2-B2 stage channels."""
    channels_list = [384, 768, 1536]
    resolutions = [(120, 120), (60, 60), (30, 30)]

    for ch, (h, w) in zip(channels_list, resolutions):
        x = torch.randn(2, ch, h, w)
        m = AFTA(ch)
        y = m(x)

        assert y.shape == x.shape, f"AFTA shape mismatch for ch={ch}: {y.shape} vs {x.shape}"
        assert torch.isfinite(y).all(), f"AFTA produced non-finite values for ch={ch}"


def test_sfif_preserves_shape():
    """Verify SFIF preserves input shape at hidden_dim=256."""
    resolutions = [(120, 120), (60, 60), (30, 30)]
    for h, w in resolutions:
        x = torch.randn(2, 256, h, w)
        m = SFIF(256)
        y = m(x)

        assert y.shape == x.shape, f"SFIF shape mismatch at {h}x{w}: {y.shape} vs {x.shape}"
        assert torch.isfinite(y).all(), f"SFIF produced non-finite values at {h}x{w}"


def test_dcfsb_preserves_shape():
    """Verify DCFSB preserves input shape at hidden_dim=256."""
    resolutions = [(120, 120), (60, 60), (30, 30)]
    for h, w in resolutions:
        x = torch.randn(2, 256, h, w)
        m = DCFSB(256)
        y = m(x)

        assert y.shape == x.shape, f"DCFSB shape mismatch at {h}x{w}: {y.shape} vs {x.shape}"
        assert torch.isfinite(y).all(), f"DCFSB produced non-finite values at {h}x{w}"


def test_hgnetv2_fbd_forward():
    """Verify HGNetv2FBD stage output channels and shapes for 960x960 input."""
    # Test with miniature spatial size to run fast on CPU
    x = torch.randn(2, 3, 256, 256)
    backbone = HGNetv2FBD(
        name="B2",
        return_idx=[1, 2, 3],
        use_lab=True,
        pretrained=False,
        use_afta=True,
    )
    outs = backbone(x)

    assert len(outs) == 3
    assert outs[0].shape[1] == 384, f"C3 channel mismatch: {outs[0].shape[1]}"
    assert outs[1].shape[1] == 768, f"C4 channel mismatch: {outs[1].shape[1]}"
    assert outs[2].shape[1] == 1536, f"C5 channel mismatch: {outs[2].shape[1]}"


def test_fbd_hybrid_encoder_forward():
    """Verify FBDHybridEncoder multi-scale outputs for P3 (120x120), P4 (60x60), P5 (30x30)."""
    P3 = torch.randn(2, 384, 120, 120)
    P4 = torch.randn(2, 768, 60, 60)
    P5 = torch.randn(2, 1536, 30, 30)

    encoder = FBDHybridEncoder(
        in_channels=[384, 768, 1536],
        feat_strides=[8, 16, 32],
        hidden_dim=256,
        use_encoder_idx=[2],
        num_encoder_layers=1,
        depth_mult=0.67,
        use_sfif=True,
        use_dcfsb=True,
    )

    outs = encoder([P3, P4, P5])

    assert len(outs) == 3
    assert outs[0].shape == (2, 256, 120, 120), f"P3 output shape mismatch: {outs[0].shape}"
    assert outs[1].shape == (2, 256, 60, 60), f"P4 output shape mismatch: {outs[1].shape}"
    assert outs[2].shape == (2, 256, 30, 30), f"P5 output shape mismatch: {outs[2].shape}"


def test_pretrained_weight_transfer():
    """Verify non-strict loading of dfine_m_coco.pth into FBDNet architecture."""
    weights_path = ROOT / "weights" / "dfine_m_coco.pth"
    if not weights_path.exists():
        pytest.skip(f"Pretrained weights not found at {weights_path}")

    state_dict = torch.load(weights_path, map_location="cpu")
    if "model" in state_dict:
        state_dict = state_dict["model"]

    # Initialize FBDHybridEncoder
    encoder = FBDHybridEncoder(
        in_channels=[384, 768, 1536],
        feat_strides=[8, 16, 32],
        hidden_dim=256,
        use_encoder_idx=[2],
        num_encoder_layers=1,
        depth_mult=0.67,
        use_sfif=True,
        use_dcfsb=True,
    )

    # Prefix filter encoder keys from full model
    encoder_state = {
        k.removeprefix("encoder."): v
        for k, v in state_dict.items()
        if k.startswith("encoder.")
    }

    incompatible = encoder.load_state_dict(encoder_state, strict=False)

    # All missing keys MUST only be inside sfif and dcfsb modules
    for mk in incompatible.missing_keys:
        assert mk.startswith("sfif.") or mk.startswith("dcfsb_"), (
            f"Unexpected missing key outside FBD modules: {mk}"
        )

    # No unexpected keys within encoder scope
    assert len(incompatible.unexpected_keys) == 0, (
        f"Unexpected keys found: {incompatible.unexpected_keys[:5]}"
    )
