"""Unit tests for Adaptive High-Resolution Detail Branch (AHRD)."""

from __future__ import annotations

import sys
import types
from importlib.machinery import ModuleSpec

# Safe tensorboard stub for test environment
_tf_stub = types.ModuleType("tensorflow")
_tf_stub.__version__ = "stub"
_tf_stub.__spec__ = ModuleSpec("tensorflow", loader=None)
_tf_stub.__path__ = []
_tf_stub.io = types.SimpleNamespace(gfile=types.SimpleNamespace(join=lambda *a: "/".join(str(x) for x in a)))
sys.modules["tensorflow"] = _tf_stub

from pathlib import Path
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
DFINE_DIR = ROOT / "results" / "D-FINE"
if str(DFINE_DIR) not in sys.path:
    sys.path.insert(0, str(DFINE_DIR))

from scale_error.models.ahrd import AHRDModule
from src.zoo.dfine.ahrd import DFINE_AHRD, AHRDModule as ZooAHRDModule


def test_ahrd_module_shapes_and_identity():
    """Verify AHRDModule output shape, identity at alpha=0, and gradient propagation."""
    module = AHRDModule(in_channels_d=96, in_channels_s=384, out_channels=384)

    # D: stride 4 (H=240, W=240), S: stride 8 (H=120, W=120)
    d = torch.randn(2, 96, 60, 60, requires_grad=True)
    s = torch.randn(2, 384, 30, 30, requires_grad=True)

    out = module(d, s)

    # 1. Shape preservation
    assert out.shape == s.shape, f"Expected shape {s.shape}, got {out.shape}"

    # 2. Strict identity at initialization (alpha = 0.0)
    assert module.alpha.item() == 0.0, "alpha must be initialized to 0.0"
    assert torch.allclose(out, s, atol=1e-7), "Output must strictly equal s when alpha=0.0"

    # 3. Gradient check
    loss = out.sum()
    loss.backward()
    assert s.grad is not None and s.grad.abs().sum() > 0
    assert module.alpha.grad is not None


def test_ahrd_parameter_count():
    """Verify that AHRD adds minimal parameters (< 0.5M)."""
    module = AHRDModule(in_channels_d=96, in_channels_s=384, out_channels=384)
    param_count = sum(p.numel() for p in module.parameters())
    assert param_count < 500_000, f"AHRD parameter count {param_count} exceeds 500k budget"
    assert param_count == 453_601, f"Expected 453,601 params, got {param_count}"
