"""Tile-based dataset slicing pipeline for high-resolution images (SAHI-style).

Implements spatial preservation by slicing high-resolution ROI-cropped images
into 960x960 tiles with 20% overlap, properly handling:
1. Boundary annotation truncation (retains box only if area ratio >= min_area_ratio).
2. Negative background sampling (keeps a controlled fraction of empty tiles to mitigate false positives).
3. Generates complete YOLO-compatible sliced dataset with data.yaml.
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import cv2
import yaml


def calculate_slice_intervals(total_dim: int, slice_dim: int, overlap_ratio: float) -> list[int]:
    """Calculate 1D start coordinates with overlap covering total dimension."""
    if total_dim <= slice_dim:
        return [0]
    step = int(slice_dim * (1.0 - overlap_ratio))
    step = max(1, step)
    coords = []
    pos = 0
    while pos + slice_dim <= total_dim:
        coords.append(pos)
        pos += step
    if coords[-1] + slice_dim < total_dim:
        coords.append(total_dim - slice_dim)
    return sorted(list(set(coords)))


def slice_image_and_labels(
    image_path: Path,
    label_path: Path | None,
    output_img_dir: Path,
    output_lbl_dir: Path,
    slice_w: int = 960,
    slice_h: int = 960,
    overlap_ratio: float = 0.20,
    min_area_ratio: float = 0.30,
    negative_ratio: float = 0.20,
    rng: random.Random | None = None,
) -> tuple[int, int]:
    """Slice a single image and transform its bounding boxes.

    Returns: (num_positive_slices, num_negative_slices)
    """
    if rng is None:
        rng = random.Random(42)

    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Warning: Could not read image {image_path}, skipping.")
        return 0, 0

    img_h, img_w = img.shape[:2]

    # Load original boxes (class, x1, y1, x2, y2 in pixels)
    boxes = []
    if label_path is not None and label_path.exists():
        for line in label_path.read_text(encoding="utf-8").strip().splitlines():
            parts = line.strip().split()
            if len(parts) >= 5:
                cls_id = int(float(parts[0]))
                xc, yc, w, h = map(float, parts[1:5])
                x1 = (xc - w / 2.0) * img_w
                y1 = (yc - h / 2.0) * img_h
                x2 = (xc + w / 2.0) * img_w
                y2 = (yc + h / 2.0) * img_h
                boxes.append((cls_id, x1, y1, x2, y2))

    x_starts = calculate_slice_intervals(img_w, slice_w, overlap_ratio)
    y_starts = calculate_slice_intervals(img_h, slice_h, overlap_ratio)

    stem = image_path.stem
    pos_count = 0
    neg_count = 0

    for sy in y_starts:
        for sx in x_starts:
            tile_x2 = sx + slice_w
            tile_y2 = sy + slice_h

            tile_boxes = []
            for cls_id, x1, y1, x2, y2 in boxes:
                # Intersect with tile
                ix1 = max(x1, sx)
                iy1 = max(y1, sy)
                ix2 = min(x2, tile_x2)
                iy2 = min(y2, tile_y2)

                if ix2 > ix1 and iy2 > iy1:
                    inter_area = (ix2 - ix1) * (iy2 - iy1)
                    orig_area = (x2 - x1) * (y2 - y1)
                    if orig_area > 0 and (inter_area / orig_area) >= min_area_ratio:
                        # Shift to tile coordinate frame [0, slice_w] x [0, slice_h]
                        tx = ((ix1 + ix2) / 2.0 - sx) / float(slice_w)
                        ty = ((iy1 + iy2) / 2.0 - sy) / float(slice_h)
                        tw = (ix2 - ix1) / float(slice_w)
                        th = (iy2 - iy1) / float(slice_h)
                        # Clamp to [0, 1]
                        tx = min(max(tx, 0.0), 1.0)
                        ty = min(max(ty, 0.0), 1.0)
                        tw = min(max(tw, 0.0), 1.0)
                        th = min(max(th, 0.0), 1.0)
                        tile_boxes.append((cls_id, tx, ty, tw, th))

            is_positive = len(tile_boxes) > 0

            # Decide whether to keep
            if is_positive:
                keep = True
                pos_count += 1
            else:
                # Sample negative background tiles
                keep = rng.random() < negative_ratio
                if keep:
                    neg_count += 1

            if keep:
                tile_name = f"{stem}_x{sx}_y{sy}"
                tile_img = img[sy:tile_y2, sx:tile_x2]

                # Save tile image
                out_img_path = output_img_dir / f"{tile_name}.jpg"
                cv2.imwrite(str(out_img_path), tile_img, [cv2.IMWRITE_JPEG_QUALITY, 95])

                # Save tile label
                out_lbl_path = output_lbl_dir / f"{tile_name}.txt"
                if is_positive:
                    lines = [f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}" for c, x, y, w, h in tile_boxes]
                    out_lbl_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
                else:
                    out_lbl_path.write_text("", encoding="utf-8")

    return pos_count, neg_count


def slice_dataset(
    dataset_root: str | Path,
    output_root: str | Path,
    splits: tuple[str, ...] = ("train", "val"),
    slice_w: int = 960,
    slice_h: int = 960,
    overlap_ratio: float = 0.20,
    min_area_ratio: float = 0.30,
    negative_ratio: float = 0.20,
    seed: int = 42,
) -> Path:
    dataset_root = Path(dataset_root).resolve()
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    print(f"=== Slicing dataset from {dataset_root} to {output_root} ===")
    print(f"Parameters: slice={slice_w}x{slice_h}, overlap={overlap_ratio*100:.0f}%, "
          f"min_area_ratio={min_area_ratio}, negative_ratio={negative_ratio}")

    for split in splits:
        img_dir = dataset_root / split / "images"
        lbl_dir = dataset_root / split / "labels"

        if not img_dir.exists():
            print(f"Split {split} not found at {img_dir}, skipping.")
            continue

        out_img_dir = output_root / split / "images"
        out_lbl_dir = output_root / split / "labels"
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_lbl_dir.mkdir(parents=True, exist_ok=True)

        images = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
        print(f"\nProcessing {split} split ({len(images)} images)...")

        total_pos = 0
        total_neg = 0

        for i, img_path in enumerate(images):
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            pos, neg = slice_image_and_labels(
                img_path,
                lbl_path,
                out_img_dir,
                out_lbl_dir,
                slice_w=slice_w,
                slice_h=slice_h,
                overlap_ratio=overlap_ratio,
                min_area_ratio=min_area_ratio,
                negative_ratio=negative_ratio,
                rng=rng,
            )
            total_pos += pos
            total_neg += neg

            if (i + 1) % 500 == 0 or (i + 1) == len(images):
                print(f"  [{i+1}/{len(images)}] Generated {total_pos} pos tiles, {total_neg} neg tiles")

        print(f"Finished {split}: total {total_pos + total_neg} tiles ({total_pos} pos, {total_neg} neg)")

    # Copy or build data.yaml
    src_yaml = dataset_root / "data.yaml"
    if src_yaml.exists():
        data = yaml.safe_load(src_yaml.read_text(encoding="utf-8"))
    else:
        data = {"names": {0: "DiVat", 1: "DiVatLoiLom", 2: "LoiChi", 3: "LoiNhua", 4: "LoiTray"}}

    data["train"] = str((output_root / "train" / "images").resolve())
    if (output_root / "val" / "images").exists():
        data["val"] = str((output_root / "val" / "images").resolve())
    if (dataset_root / "test" / "images").exists():
        # Keep test pointed to ROI or original images for true full-image SAHI evaluation
        data["test"] = str((dataset_root / "test" / "images").resolve())

    out_yaml = output_root / "data.yaml"
    out_yaml.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"\nCreated sliced data.yaml at {out_yaml}")
    return output_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Slice dataset into overlapping tiles for SAHI fine-tuning.")
    parser.add_argument("--dataset-root", required=True, help="Input dataset directory containing train/val/test")
    parser.add_argument("--output-root", required=True, help="Output directory for sliced tiles")
    parser.add_argument("--splits", nargs="+", default=["train", "val"], help="Splits to slice")
    parser.add_argument("--slice-w", type=int, default=960, help="Tile width")
    parser.add_argument("--slice-h", type=int, default=960, help="Tile height")
    parser.add_argument("--overlap", type=float, default=0.20, help="Overlap ratio between adjacent tiles")
    parser.add_argument("--min-area-ratio", type=float, default=0.30, help="Minimum remaining box area ratio to keep truncated boxes")
    parser.add_argument("--negative-ratio", type=float, default=0.20, help="Fraction of empty tiles to retain")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible negative sampling")

    args = parser.parse_args()
    slice_dataset(
        args.dataset_root,
        args.output_root,
        splits=tuple(args.splits),
        slice_w=args.slice_w,
        slice_h=args.slice_h,
        overlap_ratio=args.overlap,
        min_area_ratio=args.min_area_ratio,
        negative_ratio=args.negative_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
