from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Iterable

import yaml

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def list_images(folder: str | Path) -> list[Path]:
    folder = Path(folder)
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def label_path_for_image(image_path: Path, images_root: Path, labels_root: Path) -> Path:
    rel = image_path.relative_to(images_root)
    return (labels_root / rel).with_suffix(".txt")


def _normalize_names(names: object) -> dict[int, str]:
    if isinstance(names, list):
        return {i: str(v) for i, v in enumerate(names)}
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    raise ValueError("data.yaml 'names' must be a list or dict.")


def read_names(data_yaml: str | Path) -> dict[int, str]:
    with Path(data_yaml).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return _normalize_names(data["names"])


def parse_yolo_label_file(path: str | Path, nc: int) -> tuple[list[list[float]], list[str]]:
    path = Path(path)
    boxes: list[list[float]] = []
    errors: list[str] = []
    if not path.exists():
        return boxes, errors

    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        raw = raw.strip()
        if not raw:
            continue
        parts = raw.split()
        if len(parts) != 5:
            errors.append(f"{path}:{line_no}: expected 5 fields, got {len(parts)}")
            continue
        try:
            cls_f, x, y, w, h = map(float, parts)
        except ValueError:
            errors.append(f"{path}:{line_no}: non-numeric value")
            continue
        cls = int(cls_f)
        if cls_f != cls:
            errors.append(f"{path}:{line_no}: class id must be integer-like")
        if not (0 <= cls < nc):
            errors.append(f"{path}:{line_no}: class id {cls} outside [0,{nc - 1}]")
        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
            errors.append(f"{path}:{line_no}: invalid normalized bbox {x} {y} {w} {h}")
        boxes.append([float(cls), x, y, w, h])
    return boxes, errors


def validate_split(
    split_dir: str | Path,
    names: dict[int, str],
) -> dict:
    split_dir = Path(split_dir)
    images_root = split_dir / "images"
    labels_root = split_dir / "labels"
    images = list_images(images_root)

    class_counts = Counter()
    errors: list[str] = []
    missing_labels = 0
    empty_labels = 0
    total_boxes = 0
    small_boxes = 0
    tiny_boxes = 0

    for image in images:
        label = label_path_for_image(image, images_root, labels_root)
        if not label.exists():
            missing_labels += 1
            continue
        boxes, file_errors = parse_yolo_label_file(label, len(names))
        errors.extend(file_errors)
        if not boxes:
            empty_labels += 1
        for cls, _x, _y, w, h in boxes:
            class_counts[int(cls)] += 1
            total_boxes += 1
            area = w * h
            if area < 0.01:
                small_boxes += 1
            if area < 0.0025:
                tiny_boxes += 1

    return {
        "images": len(images),
        "total_boxes": total_boxes,
        "missing_label_files": missing_labels,
        "empty_label_files": empty_labels,
        "class_counts": {
            names[i]: int(class_counts.get(i, 0)) for i in range(len(names))
        },
        "small_box_ratio_area_lt_1pct": (small_boxes / total_boxes if total_boxes else 0.0),
        "tiny_box_ratio_area_lt_0_25pct": (tiny_boxes / total_boxes if total_boxes else 0.0),
        "errors": errors,
    }


def validate_dataset(dataset_root: str | Path, report_path: str | Path | None = None) -> dict:
    dataset_root = Path(dataset_root).resolve()
    data_yaml = dataset_root / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"Missing data.yaml: {data_yaml}")

    names = read_names(data_yaml)
    expected = {"DiVat", "DiVatLoiLom", "LoiChi", "LoiNhua", "LoiTray"}
    actual = set(names.values())

    report = {
        "dataset_root": str(dataset_root),
        "names": names,
        "class_set_matches_scale_error_dc": actual == expected,
        "splits": {},
    }
    for split in ("train", "valid", "val", "test"):
        if (dataset_root / split / "images").exists():
            report["splits"][split] = validate_split(dataset_root / split, names)

    fatal_errors = []
    for split_data in report["splits"].values():
        fatal_errors.extend(split_data["errors"])
    report["fatal_label_error_count"] = len(fatal_errors)

    if report_path is not None:
        p = Path(report_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    if fatal_errors:
        preview = "\n".join(fatal_errors[:20])
        raise ValueError(f"Dataset has {len(fatal_errors)} invalid YOLO label rows.\n{preview}")
    return report


def _sha1(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def hash_leakage_report(split_image_dirs: dict[str, str | Path]) -> dict:
    hashes: dict[str, dict[str, list[str]]] = {}
    for split, folder in split_image_dirs.items():
        per_hash: dict[str, list[str]] = {}
        for image in list_images(folder):
            digest = _sha1(image)
            per_hash.setdefault(digest, []).append(str(image))
        hashes[split] = per_hash

    cross_split = []
    split_names = list(hashes)
    for i, a in enumerate(split_names):
        for b in split_names[i + 1 :]:
            common = set(hashes[a]).intersection(hashes[b])
            for digest in sorted(common):
                cross_split.append(
                    {
                        "sha1": digest,
                        "split_a": a,
                        "files_a": hashes[a][digest],
                        "split_b": b,
                        "files_b": hashes[b][digest],
                    }
                )
    return {
        "cross_split_duplicate_groups": cross_split,
        "cross_split_duplicate_count": len(cross_split),
    }
