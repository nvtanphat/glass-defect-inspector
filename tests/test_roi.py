import numpy as np

from demo.app import crop_roi, roi_for_image


def test_reference_resolution_uses_label_preserving_roi():
    image = np.zeros((3000, 5328, 3), dtype=np.uint8)
    cropped, coords = crop_roi(image)
    assert coords == (752, 0, 4732, 3000)
    assert cropped.shape[:2] == (3000, 3980)


def test_1024_images_pass_through_unchanged():
    image = np.zeros((1024, 1024, 3), dtype=np.uint8)
    cropped, coords = crop_roi(image)
    assert coords == (0, 0, 1024, 1024)
    assert cropped.shape == image.shape


def test_unknown_resolution_passes_through():
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    assert roi_for_image(image) == (0, 0, 640, 480)
