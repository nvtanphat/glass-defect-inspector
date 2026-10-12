import numpy as np
import pytest

from scale_error.evaluation.detection_confusion import (
    box_iou_xywh,
    confusion_statistics,
    detection_confusion,
)


def gt(image, category, box):
    return {"image_id": image, "category_id": category, "bbox": box}


def pred(image, category, box, score=0.9):
    return gt(image, category, box) | {"score": score}


def test_correct_wrong_class_missed_and_empty_ground_truth_image():
    images = [{"id": index} for index in range(4)]
    annotations = [gt(0, 0, [0, 0, 10, 10]), gt(1, 1, [0, 0, 10, 10]), gt(2, 2, [0, 0, 10, 10])]
    predictions = [
        pred(0, 0, [0, 0, 10, 10]),
        pred(1, 2, [0, 0, 10, 10]),
        pred(3, 1, [5, 5, 10, 10]),
    ]
    matrix, errors = detection_confusion(images, annotations, predictions, nc=3)
    np.testing.assert_array_equal(matrix, [[1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 1, 0, 0]])
    rows, totals = confusion_statistics(matrix, ["a", "b", "c"])
    assert totals["tp"] == 1 and totals["fp"] == 2 and totals["fn"] == 2
    assert totals["wrong_class"] == 1
    assert totals["precision"] == pytest.approx(1 / 3)
    assert rows[1]["misclassified"] == 1
    assert errors[3]["false_alarms"] == 1


def test_duplicate_prediction_cannot_match_ground_truth_twice():
    matrix, _ = detection_confusion(
        [{"id": 1}],
        [gt(1, 0, [0, 0, 10, 10])],
        [pred(1, 0, [0, 0, 10, 10]), pred(1, 0, [0, 0, 10, 10])],
        nc=1,
    )
    np.testing.assert_array_equal(matrix, [[1, 0], [1, 0]])


def test_highest_iou_pair_is_selected_before_lower_overlap():
    matrix, _ = detection_confusion(
        [{"id": 1}],
        [gt(1, 0, [0, 0, 10, 10]), gt(1, 1, [4, 0, 10, 10])],
        [pred(1, 1, [4, 0, 10, 10]), pred(1, 0, [0, 0, 10, 10])],
        nc=2,
    )
    np.testing.assert_array_equal(matrix, [[1, 0, 0], [0, 1, 0], [0, 0, 0]])


def test_low_confidence_prediction_is_excluded_and_empty_output_is_supported():
    matrix, _ = detection_confusion(
        [{"id": 1}],
        [gt(1, 0, [0, 0, 10, 10])],
        [pred(1, 0, [0, 0, 10, 10], 0.49)],
        nc=1,
    )
    np.testing.assert_array_equal(matrix, [[0, 1], [0, 0]])
    empty, _ = detection_confusion([{"id": 1}], [], [], nc=1)
    assert empty.sum() == 0


def test_pixel_iou_and_invalid_predictions():
    assert box_iou_xywh([[0, 0, 10, 10]], [[5, 0, 10, 10]])[0, 0] == pytest.approx(1 / 3)
    with pytest.raises(ValueError):
        detection_confusion([{"id": 1}], [], [pred(2, 0, [0, 0, 10, 10])])
    with pytest.raises(ValueError):
        detection_confusion([{"id": 1}], [], [pred(1, 0, [0, 0, 10, 10], float("nan"))])
