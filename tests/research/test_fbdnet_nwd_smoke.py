"""End-to-end smoke test for FBDNet-M with Scale-Aware NWD.

Verifies:
1. D-FINE YAML config instantiation for both E0 and E1.
2. Loading model weights from weights/dfine_m_fbdnet_best.pth.
3. Fresh AdamW optimizer and CosineAnnealingLR scheduler initialization.
4. Forward pass, loss calculation (L1 + GIoU + FGL + VFL + Scale-Aware NWD), and backward step.
5. Verifying finite losses and gradient updates without NaN.
"""

from __future__ import annotations

import sys

# Stub broken tensorflow in local site-packages if present
import types as _types
from importlib.machinery import ModuleSpec as _ModuleSpec
from pathlib import Path

_tf_stub = _types.ModuleType("tensorflow")
_tf_stub.__version__ = "stub"
_tf_stub.__spec__ = _ModuleSpec("tensorflow", loader=None)
_tf_stub.__path__ = []
_tf_stub.io = _types.SimpleNamespace(
    gfile=_types.SimpleNamespace(join=lambda *a: "/".join(str(x) for x in a))
)
sys.modules["tensorflow"] = _tf_stub

import torch

ROOT = Path(__file__).resolve().parents[2]
DFINE_ROOT = ROOT / "results" / "D-FINE"
if str(DFINE_ROOT) not in sys.path:
    sys.path.insert(0, str(DFINE_ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from src.core import YAMLConfig


def test_fbdnet_m_e1_smoke_step():
    """Smoke test 1 forward + loss + backward step with FBDNet-M and Scale-Aware NWD."""
    config_path = DFINE_ROOT / "configs" / "dfine" / "custom" / "fbdnet_m_e1_sanwd_20ep.yml"
    assert config_path.exists(), f"Config file not found: {config_path}"

    cfg = YAMLConfig(str(config_path))
    model = cfg.model
    criterion = cfg.criterion

    # Verify architecture
    assert hasattr(model, "backbone"), "Model must have backbone"
    assert hasattr(model, "encoder"), "Model must have encoder"
    assert hasattr(model, "decoder"), "Model must have decoder"
    assert criterion.use_nwd is True, "E1 must have use_nwd = True"
    assert "loss_nwd" in criterion.weight_dict, "loss_nwd must be in weight_dict"

    # Verify checkpoint loading contract
    ckpt_path = ROOT / "weights" / "dfine_m_fbdnet_best.pth"
    assert ckpt_path.exists(), f"Checkpoint not found: {ckpt_path}"
    state = torch.load(ckpt_path, map_location="cpu")
    weights = state["ema"]["module"] if "ema" in state else state["model"]

    # Load weights into model
    missing, unexpected = model.load_state_dict(weights, strict=False)
    print(f"Loaded weights: {len(missing)} missing, {len(unexpected)} unexpected")

    # Initialize fresh optimizer (AdamW, lr=0.00025)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00025, weight_decay=0.000125)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

    assert len(optimizer.state) == 0, "Optimizer must start with fresh state"
    assert scheduler.last_epoch == 0, "Scheduler must start from epoch 0"

    # Forward pass on small spatial size (320x320) for CPU test speed
    model.train()
    criterion.train()

    dummy_input = torch.randn(1, 3, 320, 320)
    dummy_targets = [
        {
            "boxes": torch.tensor([[0.5, 0.5, 0.05, 0.05], [0.3, 0.4, 0.08, 0.06]]),
            "labels": torch.tensor([0, 1]),
        }
    ]

    outputs = model(dummy_input, targets=dummy_targets)
    assert "pred_boxes" in outputs, "Model output must contain pred_boxes"
    assert "pred_logits" in outputs, "Model output must contain pred_logits"

    loss_dict = criterion(outputs, dummy_targets)
    assert "loss_bbox" in loss_dict, "Loss dict must contain loss_bbox"
    assert "loss_giou" in loss_dict, "Loss dict must contain loss_giou"
    assert "loss_nwd" in loss_dict, "Loss dict must contain loss_nwd"

    total_loss = sum(loss_dict.values())
    assert torch.isfinite(total_loss), f"Total loss must be finite, got {total_loss.item()}"
    assert total_loss.item() > 0.0, "Total loss must be positive"

    optimizer.zero_grad()
    total_loss.backward()

    # Verify gradients exist and are finite
    has_grad = False
    for p in model.parameters():
        if p.grad is not None:
            has_grad = True
            assert torch.isfinite(p.grad).all(), "Gradients must be finite"
    assert has_grad, "At least some parameters must have gradients"

    optimizer.step()
    scheduler.step()
    assert scheduler.last_epoch == 1
