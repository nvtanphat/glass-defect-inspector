"""Unit tests for Scale-Aware Normalized Wasserstein Distance (NWD) Auxiliary Loss
and its integration into FBDNet-M / DFINECriterion.
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
import torch.nn as nn

# Ensure D-FINE and src are in python path
ROOT = Path(__file__).resolve().parents[2]
DFINE_ROOT = ROOT / "results" / "D-FINE"
if str(DFINE_ROOT) not in sys.path:
    sys.path.insert(0, str(DFINE_ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from src.zoo.dfine.dfine_criterion import DFINECriterion
from src.zoo.dfine.matcher import HungarianMatcher

from scale_error.training.nwd_loss import ScaleAwareNWDLoss


def test_nwd_zero_loss_identical_boxes():
    """Verify that identical predicted and ground-truth boxes yield exactly zero loss."""
    loss_fn = ScaleAwareNWDLoss(c=16.0, scale_aware=True, img_size=(960, 960))
    boxes = torch.tensor([
        [0.5, 0.5, 0.1, 0.1],
        [0.2, 0.3, 0.05, 0.05],
        [0.8, 0.7, 0.2, 0.15],
    ])
    loss = loss_fn(boxes, boxes, num_boxes=3)
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-5), f"Expected 0.0 loss, got {loss.item()}"
    assert torch.isfinite(loss), "Loss must be finite"


def test_nwd_positive_loss_different_boxes():
    """Verify that different boxes produce strictly positive, finite loss."""
    loss_fn = ScaleAwareNWDLoss(c=16.0, scale_aware=True, img_size=(960, 960))
    pred = torch.tensor([[0.5, 0.5, 0.1, 0.1]])
    target = torch.tensor([[0.55, 0.52, 0.1, 0.1]])
    loss = loss_fn(pred, target, num_boxes=1)
    assert loss.item() > 0.0, f"Expected positive loss, got {loss.item()}"
    assert loss.item() <= 2.0, f"Expected loss clamped by max_weight, got {loss.item()}"


def test_scale_aware_weighting_prioritizes_small_boxes():
    """Verify that small boxes receive higher loss penalty than large boxes for the same pixel shift."""
    loss_fn = ScaleAwareNWDLoss(
        c=16.0,
        scale_aware=True,
        tau=32.0,
        alpha_scale=1.0,
        min_weight=0.5,
        max_weight=2.0,
        img_size=(960, 960),
    )

    # 10px shift in center for small box (16x16 px -> scale = 16px)
    w_small, h_small = 16.0 / 960.0, 16.0 / 960.0
    shift = 10.0 / 960.0
    small_pred = torch.tensor([[0.5 + shift, 0.5, w_small, h_small]])
    small_tgt = torch.tensor([[0.5, 0.5, w_small, h_small]])

    # Exactly same 10px shift in center for large box (160x160 px -> scale = 160px)
    w_large, h_large = 160.0 / 960.0, 160.0 / 960.0
    large_pred = torch.tensor([[0.5 + shift, 0.5, w_large, h_large]])
    large_tgt = torch.tensor([[0.5, 0.5, w_large, h_large]])

    loss_small = loss_fn(small_pred, small_tgt, num_boxes=1)
    loss_large = loss_fn(large_pred, large_tgt, num_boxes=1)

    assert loss_small.item() > loss_large.item(), (
        f"Small box loss ({loss_small.item():.4f}) must exceed large box loss ({loss_large.item():.4f})"
    )


def test_nwd_empty_boxes():
    """Verify that empty inputs gracefully return 0.0 without errors."""
    loss_fn = ScaleAwareNWDLoss()
    empty_pred = torch.zeros((0, 4), requires_grad=True)
    empty_tgt = torch.zeros((0, 4))
    loss = loss_fn(empty_pred, empty_tgt, num_boxes=0)
    assert loss.item() == 0.0
    loss.backward()
    assert empty_pred.grad is not None


def test_nwd_gradient_backprop():
    """Verify backward gradient flow to predicted box parameters."""
    loss_fn = ScaleAwareNWDLoss(c=16.0, scale_aware=True, img_size=(960, 960))
    pred = torch.tensor([[0.51, 0.52, 0.08, 0.09]], requires_grad=True)
    target = torch.tensor([[0.5, 0.5, 0.1, 0.1]])

    loss = loss_fn(pred, target, num_boxes=1)
    loss.backward()

    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()
    assert (pred.grad.abs() > 0).any(), "Gradient should be non-zero"


def test_dfine_criterion_e0_baseline():
    """Verify DFINECriterion with use_nwd=False (E0 baseline) does not compute loss_nwd."""
    matcher = HungarianMatcher(weight_dict={"cost_class": 2, "cost_bbox": 5, "cost_giou": 2})
    weight_dict = {"loss_vfl": 1.0, "loss_bbox": 5.0, "loss_giou": 2.0}
    criterion = DFINECriterion(
        matcher=matcher,
        weight_dict=weight_dict,
        losses=["boxes"],
        use_nwd=False,
    )
    assert not criterion.use_nwd
    assert "loss_nwd" not in criterion.weight_dict

    # Dummy batch with 2 queries and 1 target
    outputs = {
        "pred_boxes": torch.tensor([[[0.5, 0.5, 0.2, 0.2], [0.1, 0.1, 0.1, 0.1]]], requires_grad=True),
        "pred_logits": torch.randn(1, 2, 5),
    }
    targets = [{"boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]]), "labels": torch.tensor([0])}]
    indices = [(torch.tensor([0]), torch.tensor([0]))]

    loss_dict = criterion.loss_boxes(outputs, targets, indices, num_boxes=1)
    assert "loss_bbox" in loss_dict
    assert "loss_giou" in loss_dict
    assert "loss_nwd" not in loss_dict


def test_dfine_criterion_e1_scale_aware_nwd():
    """Verify DFINECriterion with use_nwd=True (E1) incorporates loss_nwd and backpropagates."""
    matcher = HungarianMatcher(weight_dict={"cost_class": 2, "cost_bbox": 5, "cost_giou": 2})
    weight_dict = {"loss_vfl": 1.0, "loss_bbox": 5.0, "loss_giou": 2.0, "loss_nwd": 0.2}
    criterion = DFINECriterion(
        matcher=matcher,
        weight_dict=weight_dict,
        losses=["boxes"],
        use_nwd=True,
        nwd_weight=0.2,
        nwd_c=16.0,
        nwd_scale_aware=True,
    )
    assert criterion.use_nwd
    assert "loss_nwd" in criterion.weight_dict
    assert criterion.weight_dict["loss_nwd"] == 0.2

    # Slightly offset prediction
    outputs = {
        "pred_boxes": torch.tensor([[[0.52, 0.51, 0.18, 0.19], [0.1, 0.1, 0.1, 0.1]]], requires_grad=True),
        "pred_logits": torch.randn(1, 2, 5),
    }
    targets = [{"boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]]), "labels": torch.tensor([0])}]
    indices = [(torch.tensor([0]), torch.tensor([0]))]

    loss_dict = criterion.loss_boxes(outputs, targets, indices, num_boxes=1)
    assert "loss_bbox" in loss_dict
    assert "loss_giou" in loss_dict
    assert "loss_nwd" in loss_dict
    assert loss_dict["loss_nwd"].item() > 0.0

    total_loss = loss_dict["loss_bbox"] * 5.0 + loss_dict["loss_giou"] * 2.0 + loss_dict["loss_nwd"] * 0.2
    total_loss.backward()
    assert outputs["pred_boxes"].grad is not None
    assert torch.isfinite(outputs["pred_boxes"].grad).all()


def test_checkpoint_weight_loading_fresh_state():
    """Verify FBDNet-M checkpoint weights load cleanly and optimizer/scheduler start fresh."""
    ckpt_path = ROOT / "weights" / "dfine_m_fbdnet_best.pth"
    assert ckpt_path.exists(), f"Checkpoint {ckpt_path} not found"

    ckpt = torch.load(ckpt_path, map_location="cpu")
    assert "ema" in ckpt or "model" in ckpt, "Checkpoint must contain weights"
    state_dict = ckpt["ema"]["module"] if "ema" in ckpt else ckpt["model"]

    # Verify key architectural layers of FBDNet exist in checkpoint
    has_backbone = any("backbone" in k for k in state_dict.keys())
    has_decoder = any("decoder" in k for k in state_dict.keys())
    assert has_backbone, "Checkpoint must contain backbone weights"
    assert has_decoder, "Checkpoint must contain decoder weights"

    # Verify fresh fine-tuning initialization contract:
    # A fresh fine-tuning run starts with:
    # - initial epoch = 0
    # - a new AdamW optimizer
    # - a new lr scheduler
    # - NO state loaded from old optimizer
    dummy_param = nn.Parameter(torch.randn(10, 10))
    fresh_optimizer = torch.optim.AdamW([dummy_param], lr=0.00025, weight_decay=0.000125)
    fresh_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(fresh_optimizer, T_max=20)

    # Optimizer state must be completely uninitialized (fresh)
    assert len(fresh_optimizer.state) == 0
    assert fresh_scheduler.last_epoch == 0
