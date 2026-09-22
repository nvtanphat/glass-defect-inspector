from pathlib import Path

from scale_error.data.validate import parse_yolo_label_file


def test_valid_yolo_label(tmp_path: Path):
    p = tmp_path / "a.txt"
    p.write_text("2 0.5 0.5 0.2 0.1\n", encoding="utf-8")
    boxes, errors = parse_yolo_label_file(p, nc=5)
    assert not errors
    assert len(boxes) == 1


def test_invalid_bbox_detected(tmp_path: Path):
    p = tmp_path / "a.txt"
    p.write_text("2 1.2 0.5 0.2 0.1\n", encoding="utf-8")
    _, errors = parse_yolo_label_file(p, nc=5)
    assert errors
