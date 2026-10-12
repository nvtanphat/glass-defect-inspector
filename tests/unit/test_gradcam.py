"""Spatial target matching and honest heatmap behavior for detector explanations."""

import numpy as np
import pytest
import torch

from glass_inspector.gradcam import normalize_cam, select_query


def test_saved_detection_selects_matching_box_instead_of_remote_high_score():
    logits = torch.tensor([[1.0, -2.0], [9.0, -2.0], [2.0, -2.0]])
    boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0], [50.0, 50.0, 60.0, 60.0], [1.0, 1.0, 9.0, 9.0]])
    assert select_query(logits, boxes, [0, 0, 10, 10], 0) == (2, 0)


def test_no_detection_selects_raw_candidate_without_manufacturing_a_box():
    logits = torch.tensor([[-2.0, -1.0], [0.0, 3.0]])
    assert select_query(logits, torch.zeros(2, 4)) == (1, 1)


def test_cam_preserves_spatial_signal_and_reports_flat_maps():
    features = [torch.tensor([[[[0.0, 1.0], [2.0, 3.0]]]])]
    cam, metadata = normalize_cam(features, [torch.ones_like(features[0])], (2, 2))
    assert np.allclose(cam, [[0, 1 / 3], [2 / 3, 1]])
    assert metadata == {"flat_cam": False, "nonzero_gradient": True}
    flat, metadata = normalize_cam(features, [torch.zeros_like(features[0])], (2, 2))
    assert not flat.any()
    assert metadata == {"flat_cam": True, "nonzero_gradient": False}


def test_invalid_gradients_do_not_produce_fake_visuals():
    features = [torch.ones(1, 1, 2, 2)]
    with pytest.raises(ValueError, match="Nonfinite"):
        normalize_cam(features, [torch.full_like(features[0], float("nan"))], (2, 2))
    with pytest.raises(ValueError, match="connected"):
        normalize_cam(features, [None], (2, 2))
