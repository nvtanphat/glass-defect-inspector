from __future__ import annotations

import numpy as np
from scale_error.evaluation.sahi_evaluator import box_iou, calculate_slice_coords


def test_calculate_slice_coords():
    coords = calculate_slice_coords(1920, 1080, slice_w=960, slice_h=960, overlap_ratio=0.20)
    assert len(coords) > 0
    # Every coordinate must fit within boundaries
    for x1, y1, x2, y2 in coords:
        assert 0 <= x1 < x2 <= 1920
        assert 0 <= y1 < y2 <= 1080


def test_box_iou():
    b1 = np.array([[0, 0, 10, 10], [10, 10, 20, 20]], dtype=np.float32)
    b2 = np.array([[0, 0, 10, 10]], dtype=np.float32)
    ious = box_iou(b1, b2)
    assert ious.shape == (2, 1)
    assert np.isclose(ious[0, 0], 1.0)
    assert np.isclose(ious[1, 0], 0.0)
