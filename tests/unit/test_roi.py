from PIL import Image

from glass_inspector.inference_engine import compute_slice_intervals, get_roi_crop
from processing.apply_roi_crop import _crop_boxes_to_roi


def test_reference_resolution_uses_label_preserving_roi():
    cropped, coords = get_roi_crop(Image.new("RGB", (5328, 3000)))
    assert coords == (752, 0, 4732, 3000)
    assert cropped.size == (3980, 3000)


def test_1024_images_pass_through_unchanged():
    image = Image.new("RGB", (1024, 1024))
    cropped, coords = get_roi_crop(image)
    assert coords is None
    assert cropped is image


def test_unknown_resolution_passes_through():
    image = Image.new("RGB", (3000, 1800))
    assert get_roi_crop(image) == (image, None)


def test_tiling_covers_boundaries():
    intervals = compute_slice_intervals(3980)
    assert intervals[0][0] == 0 and intervals[-1][1] == 3980
    assert all(b - a == 960 for a, b in intervals)
    assert all(intervals[i + 1][0] <= intervals[i][1] for i in range(len(intervals) - 1))


def test_roi_clips_and_rejects_boxes_outside_crop():
    boxes = [[0, 0.5, 0.5, 0.4, 0.2], [1, 0.05, 0.5, 0.01, 0.1]]
    result = _crop_boxes_to_roi(boxes, (0.25, 0, 0.75, 1), 0.1)
    assert len(result) == 1
    assert abs(result[0][1] - 0.5) < 1e-9
    assert abs(result[0][3] - 0.8) < 1e-9
