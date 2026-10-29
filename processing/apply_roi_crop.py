"""Apply the fixed per-resolution ROI crops from roi_config.json to a split's
images and YOLO labels, writing a cropped copy under a new root.

Companion to compute_roi.py, which only proposes the ROI boxes. This script
performs the actual crop:

  - looks up each image's ROI by its (width, height) resolution group
  - images whose resolution has no group in roi_config.json (too few boxes to
    estimate a stable ROI, or a size seen only outside train) are copied
    through unchanged, so a mixed-resolution dataset never loses images
  - crops the image to the ROI box
  - remaps every YOLO label box into ROI-relative normalized coordinates,
    clipping at the ROI edges; a box whose clipped area is a small fraction
    of its original area (default 10%, i.e. mostly cropped away) is dropped
    instead of kept as a sliver, since a near-invisible fragment is not a
    usable training signal for defects that are already only a few pixels
    wide (DiVat, LoiChi, LoiTray)
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from scale_error.data.validate import (
    label_path_for_image,
    list_images,
    parse_yolo_label_file,
    read_names,
)  # noqa: E402


def _crop_boxes_to_roi(
    boxes: list[list[float]],
    roi_xyxyn: tuple[float, float, float, float],
    min_area_keep_frac: float,
) -> list[list[float]]:
    roi_x1, roi_y1, roi_x2, roi_y2 = roi_xyxyn
    roi_w = roi_x2 - roi_x1
    roi_h = roi_y2 - roi_y1

    out: list[list[float]] = []
    for cls, x, y, w, h in boxes:
        bx1, by1, bx2, by2 = x - w / 2, y - h / 2, x + w / 2, y + h / 2
        orig_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        if orig_area <= 0:
            continue

        cx1, cy1 = max(bx1, roi_x1), max(by1, roi_y1)
        cx2, cy2 = min(bx2, roi_x2), min(by2, roi_y2)
        clipped_w, clipped_h = max(0.0, cx2 - cx1), max(0.0, cy2 - cy1)
        clipped_area = clipped_w * clipped_h
        if clipped_area <= 0 or (clipped_area / orig_area) < min_area_keep_frac:
            continue

        new_x1 = (cx1 - roi_x1) / roi_w
        new_y1 = (cy1 - roi_y1) / roi_h
        new_x2 = (cx2 - roi_x1) / roi_w
        new_y2 = (cy2 - roi_y1) / roi_h
        new_w = new_x2 - new_x1
        new_h = new_y2 - new_y1
        if new_w <= 0 or new_h <= 0:
            continue
        out.append([cls, new_x1 + new_w / 2, new_y1 + new_h / 2, new_w, new_h])
    return out


def _write_label(path: Path, boxes: list[list[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{int(cls)} {x:.6f} {y:.6f} {w:.6f} {h:.6f}" for cls, x, y, w, h in boxes]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def apply_roi_split(
    split_dir: Path,
    output_dir: Path,
    roi_config: dict,
    nc: int,
    min_area_keep_frac: float,
) -> dict:
    images_root = split_dir / "images"
    labels_root = split_dir / "labels"
    out_images_root = output_dir / "images"
    out_labels_root = output_dir / "labels"

    groups = roi_config["groups"]
    stats = {"images_cropped": 0, "images_passthrough": 0, "boxes_in": 0, "boxes_out": 0}

    for image_path in list_images(images_root):
        img = cv2.imread(str(image_path))
        if img is None:
            continue
        h_px, w_px = img.shape[:2]
        key = f"{w_px}x{h_px}"

        rel = image_path.relative_to(images_root)
        out_image_path = out_images_root / rel
        out_label_path = (out_labels_root / rel).with_suffix(".txt")
        out_image_path.parent.mkdir(parents=True, exist_ok=True)

        label_path = label_path_for_image(image_path, images_root, labels_root)
        boxes, _errors = parse_yolo_label_file(label_path, nc)
        stats["boxes_in"] += len(boxes)

        group = groups.get(key)
        if group is None:
            shutil.copy2(image_path, out_image_path)
            if boxes:
                _write_label(out_label_path, boxes)
            stats["images_passthrough"] += 1
            stats["boxes_out"] += len(boxes)
            continue

        roi_xyxyn = tuple(group["roi_xyxyn"])
        x1n, y1n, x2n, y2n = roi_xyxyn
        x1, y1 = int(round(x1n * w_px)), int(round(y1n * h_px))
        x2, y2 = int(round(x2n * w_px)), int(round(y2n * h_px))
        x2, y2 = max(x2, x1 + 1), max(y2, y1 + 1)

        cropped = img[y1:y2, x1:x2]
        cv2.imwrite(str(out_image_path), cropped)

        new_boxes = _crop_boxes_to_roi(boxes, roi_xyxyn, min_area_keep_frac)
        if new_boxes:
            _write_label(out_label_path, new_boxes)
        stats["images_cropped"] += 1
        stats["boxes_out"] += len(new_boxes)

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root", required=True, help="Root with train/ (and optionally valid/) YOLO splits"
    )
    parser.add_argument("--roi-config", default="processing/roi_config.json")
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "valid"],
        help="Split subdirectory names under --dataset-root to crop (default: train valid)",
    )
    parser.add_argument(
        "--min-area-keep-frac",
        type=float,
        default=0.10,
        help="Drop a box if less than this fraction of its original area survives the crop (default: 0.10)",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).resolve()
    output_root = Path(args.output_root).resolve()
    roi_config = json.loads(Path(args.roi_config).read_text(encoding="utf-8"))
    names = read_names(dataset_root / "data.yaml")

    summary = {}
    for split in args.splits:
        split_dir = dataset_root / split
        if not (split_dir / "images").exists():
            print(f"Skipping {split}: no images/ dir at {split_dir}")
            continue
        print(f"Cropping {split} ...")
        stats = apply_roi_split(
            split_dir, output_root / split, roi_config, len(names), args.min_area_keep_frac
        )
        summary[split] = stats
        print(
            f"  {split}: {stats['images_cropped']} cropped, {stats['images_passthrough']} passthrough, "
            f"boxes {stats['boxes_in']} -> {stats['boxes_out']}"
        )

    # data.yaml for the cropped dataset, pointing at the new image roots
    out_data_yaml = output_root / "data.yaml"
    split_map = {"train": "train", "valid": "val", "val": "val", "test": "test"}
    lines = []
    for split in args.splits:
        mapped = split_map.get(split, split)
        lines.append(f"{mapped}: {(output_root / split / 'images').resolve()}")
    lines.append("")
    lines.append(f"nc: {len(names)}")
    lines.append(f"names: {[names[i] for i in range(len(names))]}")
    out_data_yaml.write_text("\n".join(lines) + "\n", encoding="utf-8")

    summary_path = output_root / "roi_crop_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_data_yaml}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
