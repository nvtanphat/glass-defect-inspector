from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np
import yaml
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

from .validate import label_path_for_image, list_images, parse_yolo_label_file, read_names


def _link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _image_multihot(
    image_paths: list[Path],
    images_root: Path,
    labels_root: Path,
    nc: int,
) -> np.ndarray:
    y = np.zeros((len(image_paths), nc), dtype=np.int8)
    for i, image in enumerate(image_paths):
        label = label_path_for_image(image, images_root, labels_root)
        boxes, _ = parse_yolo_label_file(label, nc)
        for cls, *_ in boxes:
            y[i, int(cls)] = 1
    return y


def split_original_valid(
    dataset_root: str | Path,
    output_root: str | Path,
    test_size: float = 0.5,
    seed: int = 42,
) -> tuple[Path, Path]:
    dataset_root = Path(dataset_root).resolve()
    output_root = Path(output_root).resolve()

    source = dataset_root / "valid"
    images_root = source / "images"
    labels_root = source / "labels"
    if not images_root.exists():
        raise FileNotFoundError(f"Expected Roboflow valid split at {images_root}")

    names = read_names(dataset_root / "data.yaml")
    images = list_images(images_root)
    y = _image_multihot(images, images_root, labels_root, len(names))

    splitter = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=test_size, random_state=seed
    )
    val_idx, test_idx = next(splitter.split(np.zeros(len(images)), y))

    for split_name, indices in (("val", val_idx), ("test", test_idx)):
        for idx in indices:
            image = images[int(idx)]
            rel = image.relative_to(images_root)
            label = label_path_for_image(image, images_root, labels_root)
            _link_or_copy(image, output_root / split_name / "images" / rel)
            if label.exists():
                _link_or_copy(label, (output_root / split_name / "labels" / rel).with_suffix(".txt"))

    runtime_yaml = output_root / "runtime_data.yaml"
    payload = {
        "train": str((dataset_root / "train" / "images").resolve()),
        "val": str((output_root / "val" / "images").resolve()),
        "test": str((output_root / "test" / "images").resolve()),
        "names": {i: names[i] for i in range(len(names))},
    }
    with runtime_yaml.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)

    return output_root, runtime_yaml
