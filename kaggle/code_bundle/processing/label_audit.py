"""Detect duplicate / near-duplicate bounding boxes within the same image.

`src/scale_error/data/validate.py` already checks YOLO label *format*
(missing files, out-of-range values, bad class ids) and found zero errors on
this dataset. What it does not check is *semantic* label quality: two boxes
drawn for the same physical defect (a common manual-annotation slip) will
pass format validation cleanly but inflate box counts and can teach the
detector to expect duplicate predictions on real defects.

This script flags image/label pairs where two or more boxes overlap heavily
(IoU above a threshold), regardless of class, so they can be reviewed by eye
before any further pipeline changes (ROI crop, tiling, etc.) risk baking the
duplication in.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from scale_error.data.validate import (  # noqa: E402
    label_path_for_image,
    list_images,
    parse_yolo_label_file,
    read_names,
)


def _xywhn_to_xyxy(box: list[float]) -> tuple[float, float, float, float]:
    _cls, x, y, w, h = box
    return (x - w / 2, y - h / 2, x + w / 2, y + h / 2)


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def find_duplicate_boxes(
    images_root: Path,
    labels_root: Path,
    names: dict[int, str],
    iou_threshold: float = 0.7,
) -> list[dict]:
    flagged = []
    for image_path in list_images(images_root):
        label_path = label_path_for_image(image_path, images_root, labels_root)
        boxes, _errors = parse_yolo_label_file(label_path, len(names))
        if len(boxes) < 2:
            continue

        xyxy = [_xywhn_to_xyxy(b) for b in boxes]
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                iou = _iou(xyxy[i], xyxy[j])
                if iou < iou_threshold:
                    continue
                flagged.append(
                    {
                        "image": str(image_path),
                        "label": str(label_path),
                        "iou": round(iou, 4),
                        "box_a": {"class": names[int(boxes[i][0])], "xywhn": boxes[i][1:]},
                        "box_b": {"class": names[int(boxes[j][0])], "xywhn": boxes[j][1:]},
                        "same_class": int(boxes[i][0]) == int(boxes[j][0]),
                    }
                )
    return flagged


def audit_split(split_dir: Path, names: dict[int, str], iou_threshold: float) -> dict:
    images_root = split_dir / "images"
    labels_root = split_dir / "labels"
    flagged = find_duplicate_boxes(images_root, labels_root, names, iou_threshold)
    return {
        "split_dir": str(split_dir),
        "iou_threshold": iou_threshold,
        "images_scanned": len(list_images(images_root)),
        "duplicate_pairs_found": len(flagged),
        "images_affected": len({f["image"] for f in flagged}),
        "duplicates": flagged,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, help="Folder containing data.yaml")
    parser.add_argument(
        "--splits", nargs="+", default=["train", "valid"], help="Split folder names to scan"
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.7,
        help="Boxes with IoU at or above this are flagged as duplicates (default: 0.7)",
    )
    parser.add_argument(
        "--output",
        default="processing/label_audit_report.json",
        help="Where to write the JSON report",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).resolve()
    names = read_names(dataset_root / "data.yaml")

    report = {"dataset_root": str(dataset_root), "splits": {}}
    for split in args.splits:
        split_dir = dataset_root / split
        if not (split_dir / "images").exists():
            continue
        report["splits"][split] = audit_split(split_dir, names, args.iou_threshold)

    total_pairs = sum(s["duplicate_pairs_found"] for s in report["splits"].values())
    total_images = sum(s["images_affected"] for s in report["splits"].values())
    report["summary"] = {
        "total_duplicate_pairs": total_pairs,
        "total_images_affected": total_images,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Scanned {sum(s['images_scanned'] for s in report['splits'].values())} images.")
    print(f"Duplicate/near-duplicate box pairs (IoU >= {args.iou_threshold}): {total_pairs}")
    print(f"Images affected: {total_images}")
    print(f"Full report: {output_path}")


if __name__ == "__main__":
    main()
