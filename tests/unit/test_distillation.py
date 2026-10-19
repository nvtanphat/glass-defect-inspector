"""Unit tests for knowledge distillation modules and loss functions."""

from __future__ import annotations

import torch

from src.scale_error.distillation.consensus_target_generator import (
    ConsensusTargetGenerator,
    PerClassDistillationConfig,
    compute_box_iou,
)
from src.scale_error.distillation.losses import (
    compute_smooth_class_weights,
    make_vfl_loss_elements,
)


def test_smooth_class_weights_clamps_extreme_imbalance():
    # Extreme unbalanced class counts (e.g., class 2 has 5500, class 4 has 300)
    counts = {0: 1400, 1: 2400, 2: 5500, 3: 370, 4: 310}
    weights = compute_smooth_class_weights(
        counts, num_classes=5, power=0.5, min_weight=0.6, max_weight=2.5
    )

    assert weights.shape == (5,)
    assert (weights >= 0.6).all()
    assert (weights <= 2.5).all()
    # Ratio between maximum and minimum weight should be controlled (< 5.0), not 17.5x
    ratio = float(weights.max() / weights.min())
    assert ratio <= 4.2


def test_compute_box_iou():
    box1 = [10.0, 10.0, 50.0, 50.0]  # area = 1600
    box2 = [10.0, 10.0, 50.0, 50.0]
    assert compute_box_iou(box1, box2) == 1.0

    box3 = [60.0, 60.0, 100.0, 100.0]  # disjoint
    assert compute_box_iou(box1, box3) == 0.0

    box4 = [30.0, 10.0, 50.0, 50.0]  # half overlap
    iou = compute_box_iou(box1, box4)
    assert 0.4 < iou < 0.6


def test_fuse_teacher_predictions_with_agreement():
    generator = ConsensusTargetGenerator()

    # Model 0 (FBDNet), Model 1 (DFINE), Model 2 (YOLO) all detect a defect at ~[20, 20, 60, 60]
    t0 = [(0, [20.0, 20.0, 60.0, 60.0], 0.85)]
    t1 = [(0, [21.0, 19.0, 59.0, 61.0], 0.90)]
    t2 = [(0, [19.0, 20.0, 60.0, 59.0], 0.80)]

    fused = generator.fuse_teacher_predictions([t0, t1, t2])
    assert len(fused) == 1
    c_id, box, score, agreement = fused[0]
    assert c_id == 0
    assert agreement == 3
    assert score > 0.80
    assert 19.0 <= box[0] <= 21.0


def test_refine_ground_truth_subpixel_and_discovery():
    cfg = PerClassDistillationConfig(subpixel_alpha=0.5, gt_match_threshold=0.40)
    generator = ConsensusTargetGenerator(config=cfg)

    # 1 annotated GT box
    gt_boxes = [{"cls": 0, "box": [100.0, 100.0, 200.0, 200.0]}]

    # Teacher candidates:
    # - Candidate 1 matches GT with slight offset -> should be refined
    # - Candidate 2 is a new defect discovered by 2 teachers (agreement=2) -> should be discovered
    fused_teachers = [
        (0, [104.0, 102.0, 202.0, 198.0], 0.92, 3),   # matched
        (4, [500.0, 500.0, 520.0, 650.0], 0.75, 2),   # discovered scratch (LoiTray)
    ]

    refined, stats = generator.refine_ground_truth(
        gt_boxes=gt_boxes,
        fused_teachers=fused_teachers,
        image_width=1000,
        image_height=1000,
    )

    assert stats["initial_gt"] == 1
    assert stats["refined_gt"] == 1
    assert stats["discovered"] == 1
    assert stats["total_distilled"] == 2

    # Check refined box coordinates: 0.5 * 100 + 0.5 * 104 = 102.0
    r_box = refined[0]["box"]
    assert r_box[0] == 102.0
    assert r_box[1] == 101.0
    assert refined[0]["is_discovered"] is False
    assert refined[1]["is_discovered"] is True

    # YOLO formatting check
    yolo_lines = generator.to_yolo_format(refined, image_width=1000, image_height=1000)
    assert len(yolo_lines) == 2
    assert yolo_lines[0].startswith("0 ")
    assert yolo_lines[1].startswith("4 ")


def test_make_vfl_loss_elements():
    pred_scores = torch.randn(2, 5, 20)
    target_scores = torch.zeros(2, 5, 20)
    # Set positive targets
    target_scores[0, 0, 5] = 0.85
    target_scores[1, 2, 10] = 0.90

    cw = torch.tensor([1.2, 1.0, 0.8, 1.5, 1.4], dtype=torch.float32)
    loss = make_vfl_loss_elements(pred_scores, target_scores, class_weights=cw)

    assert loss.shape == pred_scores.shape
    assert not torch.isnan(loss).any()
    assert (loss >= 0.0).all()
