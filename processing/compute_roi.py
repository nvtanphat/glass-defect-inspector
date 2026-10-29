"""Compute fixed ROI crop boxes from the training bbox distribution, per
source image resolution group.

Rationale: the camera/bottle position is fixed *within* a given capture setup,
so every defect's bounding box is a sample of where the bottle body can be.
Taking the envelope of the 0.5th-99.5th percentile of box edges (not raw
min/max, to ignore a handful of extreme outliers) and padding it out by a
wide margin gives a crop that:

  - keeps the full bottle body (defects are always on/near it, never outside)
  - drops dead background on all four sides
  - is small enough to make defects relatively larger after resizing to the
    training imgsz, without risking clipping real bottle area

IMPORTANT: this dataset mixes image resolutions (5328x3000: ~62% of train,
1024x1024: ~38%, plus a handful of 1598x898), and their bbox distributions
differ enough (box width averages ~0.027 vs ~0.074 normalized) to indicate
different camera/capture setups rather than just a resize of the same shot.
A single normalized ROI would misalign one group, so this script computes
one ROI per resolution group and requires images to be routed through the
matching ROI at crop/inference time (see `apply_roi_crop.py`).

This does NOT crop images/labels itself - it only computes and reports the
proposed ROIs per group. Run `apply_roi_crop.py` once these look right.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from scale_error.data.validate import (
    label_path_for_image,
    list_images,
    parse_yolo_label_file,
    read_names,
)  # noqa: E402


def collect_box_edges_by_size(
    images_root: Path, labels_root: Path, nc: int
) -> dict[tuple[int, int], dict[str, list[float]]]:
    by_size: dict[tuple[int, int], dict[str, list[float]]] = defaultdict(
        lambda: {"x1": [], "y1": [], "x2": [], "y2": []}
    )
    for image_path in list_images(images_root):
        img = cv2.imread(str(image_path))
        if img is None:
            continue
        size = (img.shape[1], img.shape[0])  # (width, height)

        label_path = label_path_for_image(image_path, images_root, labels_root)
        boxes, _errors = parse_yolo_label_file(label_path, nc)
        bucket = by_size[size]
        for _cls, x, y, w, h in boxes:
            bucket["x1"].append(x - w / 2)
            bucket["y1"].append(y - h / 2)
            bucket["x2"].append(x + w / 2)
            bucket["y2"].append(y + h / 2)
    return by_size


def compute_roi(
    x1s: np.ndarray,
    y1s: np.ndarray,
    x2s: np.ndarray,
    y2s: np.ndarray,
    edge_percentile: float,
    margin: float,
) -> dict:
    left = float(np.percentile(x1s, edge_percentile))
    top = float(np.percentile(y1s, edge_percentile))
    right = float(np.percentile(x2s, 100 - edge_percentile))
    bottom = float(np.percentile(y2s, 100 - edge_percentile))

    width = right - left
    height = bottom - top

    roi_x1 = max(0.0, left - width * margin)
    roi_y1 = max(0.0, top - height * margin)
    roi_x2 = min(1.0, right + width * margin)
    roi_y2 = min(1.0, bottom + height * margin)

    return {
        "raw_envelope_xyxyn": [left, top, right, bottom],
        "roi_xyxyn": [roi_x1, roi_y1, roi_x2, roi_y2],
        "roi_width_frac": roi_x2 - roi_x1,
        "roi_height_frac": roi_y2 - roi_y1,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--edge-percentile",
        type=float,
        default=0.5,
        help="Trim this percentile from each side before taking the envelope (default: 0.5)",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.25,
        help="Extra padding as a fraction of envelope width/height on each side (default: 0.25)",
    )
    parser.add_argument(
        "--min-boxes-per-group",
        type=int,
        default=30,
        help="Skip resolution groups with fewer boxes than this (too few to estimate a stable ROI)",
    )
    parser.add_argument("--output", default="processing/roi_config.json")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).resolve()
    names = read_names(dataset_root / "data.yaml")
    images_root = dataset_root / args.split / "images"
    labels_root = dataset_root / args.split / "labels"

    print(f"Scanning {images_root} ...")
    by_size = collect_box_edges_by_size(images_root, labels_root, len(names))

    result = {
        "source_split": args.split,
        "dataset_root": str(dataset_root),
        "edge_percentile": args.edge_percentile,
        "margin": args.margin,
        "groups": {},
    }

    for size, edges in sorted(by_size.items(), key=lambda kv: -len(kv[1]["x1"])):
        n_boxes = len(edges["x1"])
        key = f"{size[0]}x{size[1]}"
        if n_boxes < args.min_boxes_per_group:
            print(f"{key}: only {n_boxes} boxes, skipping (below --min-boxes-per-group)")
            continue
        x1s, y1s, x2s, y2s = (np.array(edges[k]) for k in ("x1", "y1", "x2", "y2"))
        roi = compute_roi(x1s, y1s, x2s, y2s, args.edge_percentile, args.margin)
        roi["image_size_wh"] = list(size)
        roi["n_boxes"] = n_boxes
        result["groups"][key] = roi
        print(f"{key}: n_boxes={n_boxes}")
        print(f"  raw envelope xyxyn: {[round(v, 4) for v in roi['raw_envelope_xyxyn']]}")
        print(f"  ROI xyxyn (with margin): {[round(v, 4) for v in roi['roi_xyxyn']]}")
        print(
            f"  ROI covers {roi['roi_width_frac'] * 100:.1f}% x {roi['roi_height_frac'] * 100:.1f}% of the frame"
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
