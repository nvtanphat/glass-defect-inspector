"""D-FINE-M training wrapper.

Wraps the D-FINE framework (https://github.com/Peterande/D-FINE) to provide
a consistent interface with the rest of the scale_error training pipeline.

D-FINE uses its own training framework (not Ultralytics), so this module
handles the necessary setup: YOLO-to-COCO conversion, config generation,
and subprocess-based training invocation.

Usage (local):
    python -m scale_error.training.trainer_dfine \
        --data data/processed_roi/runtime_data.yaml \
        --dfine-repo results/D-FINE \
        --weights results/weights/dfine_m_coco.pth \
        --output-dir artifacts/runs/dfine_m

Usage (in pipeline):
    from scale_error.training.trainer_dfine import train_dfine
    train_dfine(data_yaml, dfine_repo, pretrained_weights, output_dir)
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import yaml


def _yolo_to_coco_json(images_dir: Path, labels_dir: Path, output_json: Path, class_names: list[str]) -> Path:
    """Convert YOLO-format labels to COCO JSON format required by D-FINE.

    Args:
        images_dir: Directory containing images.
        labels_dir: Directory containing YOLO .txt label files.
        output_json: Path to write the COCO JSON file.
        class_names: List of class names in order.

    Returns:
        Path to the generated COCO JSON file.
    """
    from PIL import Image

    images_list = []
    annotations = []
    ann_id = 1

    image_files = sorted(
        [f for f in images_dir.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
    )

    for img_id, img_path in enumerate(image_files, start=1):
        img = Image.open(img_path)
        w, h = img.size
        images_list.append({
            "id": img_id,
            "file_name": img_path.name,
            "width": w,
            "height": h,
        })

        label_path = labels_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            continue

        for line in label_path.read_text(encoding="utf-8").strip().splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            # Convert YOLO (cx, cy, w, h) normalized → COCO (x, y, w, h) absolute
            x = (cx - bw / 2) * w
            y = (cy - bh / 2) * h
            abs_w = bw * w
            abs_h = bh * h
            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": cls_id,
                "bbox": [round(x, 2), round(y, 2), round(abs_w, 2), round(abs_h, 2)],
                "area": round(abs_w * abs_h, 2),
                "iscrowd": 0,
            })
            ann_id += 1

    categories = [{"id": i, "name": name} for i, name in enumerate(class_names)]

    coco = {
        "images": images_list,
        "annotations": annotations,
        "categories": categories,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(coco, indent=2), encoding="utf-8")
    print(f"[D-FINE] Converted {len(images_list)} images, {len(annotations)} annotations → {output_json}")
    return output_json


def _generate_dataset_config(
    train_images_dir: Path,
    train_json: Path,
    val_images_dir: Path,
    val_json: Path,
    num_classes: int,
    output_yml: Path,
) -> Path:
    """Generate a D-FINE dataset YAML config."""
    config = {
        "task": "detection",
        "evaluator": {"type": "CocoEvaluator", "iou_types": ["bbox"]},
        "num_classes": num_classes,
        "remap_mscoco_category": False,
        "train_dataloader": {
            "type": "DataLoader",
            "dataset": {
                "type": "CocoDetection",
                "img_folder": str(train_images_dir),
                "ann_file": str(train_json),
                "return_masks": False,
                "transforms": {"type": "Compose", "ops": None},
            },
            "shuffle": True,
            "num_workers": 4,
            "drop_last": True,
            "collate_fn": {"type": "BatchImageCollateFunction"},
        },
        "val_dataloader": {
            "type": "DataLoader",
            "dataset": {
                "type": "CocoDetection",
                "img_folder": str(val_images_dir),
                "ann_file": str(val_json),
                "return_masks": False,
                "transforms": {"type": "Compose", "ops": None},
            },
            "shuffle": False,
            "num_workers": 4,
            "drop_last": False,
            "collate_fn": {"type": "BatchImageCollateFunction"},
        },
    }

    output_yml.parent.mkdir(parents=True, exist_ok=True)
    # Write as YAML-like format
    output_yml.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    print(f"[D-FINE] Generated dataset config → {output_yml}")
    return output_yml


def prepare_coco_data(
    data_yaml: str | Path,
    output_dir: str | Path,
) -> tuple[Path, Path, Path, Path, list[str]]:
    """Prepare COCO-format data from a YOLO runtime_data.yaml.

    Returns:
        (train_images_dir, train_json, val_images_dir, val_json, class_names)
    """
    data_yaml = Path(data_yaml)
    output_dir = Path(output_dir)
    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))

    names = cfg["names"]
    if isinstance(names, dict):
        class_names = [names[k] for k in sorted(names.keys(), key=int)]
    else:
        class_names = list(names)

    train_images = Path(cfg["train"])
    val_images = Path(cfg["val"])
    train_labels = train_images.parent / "labels" if not (train_images.parent.parent / "labels").exists() \
        else train_images.parent.parent / "labels" / train_images.parent.name
    val_labels = val_images.parent / "labels" if not (val_images.parent.parent / "labels").exists() \
        else val_images.parent.parent / "labels" / val_images.parent.name

    # Try common YOLO directory structures
    for candidate in [
        train_images.parent.parent / "labels" / "train",
        train_images.parent.parent / "labels",
        train_images.with_name("labels"),
    ]:
        if candidate.exists():
            train_labels = candidate
            break

    for candidate in [
        val_images.parent.parent / "labels" / "val",
        val_images.parent.parent / "labels",
        val_images.with_name("labels"),
    ]:
        if candidate.exists():
            val_labels = candidate
            break

    train_json = output_dir / "train" / "train.json"
    val_json = output_dir / "val" / "val.json"

    _yolo_to_coco_json(train_images, train_labels, train_json, class_names)
    _yolo_to_coco_json(val_images, val_labels, val_json, class_names)

    # Symlink images to output dir for D-FINE
    train_out_images = output_dir / "train"
    val_out_images = output_dir / "val"

    # Copy or link images
    for img in train_images.iterdir():
        if img.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
            dst = train_out_images / img.name
            if not dst.exists():
                shutil.copy2(img, dst)

    for img in val_images.iterdir():
        if img.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
            dst = val_out_images / img.name
            if not dst.exists():
                shutil.copy2(img, dst)

    return train_out_images, train_json, val_out_images, val_json, class_names


def train_dfine(
    data_yaml: str | Path,
    dfine_repo: str | Path,
    pretrained_weights: str | Path,
    output_dir: str | Path,
    epochs: int = 100,
    eval_size: int = 960,
    batch_size: int = 8,
) -> Path:
    """Train D-FINE-M model.

    Args:
        data_yaml: Path to runtime_data.yaml (YOLO format).
        dfine_repo: Path to D-FINE repository root.
        pretrained_weights: Path to pretrained weights (e.g., dfine_m_coco.pth).
        output_dir: Directory for training outputs.
        epochs: Number of training epochs.
        eval_size: Evaluation image size.
        batch_size: Training batch size.

    Returns:
        Path to the output directory containing best.pth.
    """
    dfine_repo = Path(dfine_repo).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Prepare COCO-format data
    print("[D-FINE] Preparing COCO-format dataset...")
    coco_data_dir = output_dir / "data_coco"
    train_imgs, train_json, val_imgs, val_json, class_names = prepare_coco_data(
        data_yaml, coco_data_dir
    )
    num_classes = len(class_names)

    # 2. Generate dataset config
    dataset_yml = output_dir / "dataset_glass.yml"
    _generate_dataset_config(train_imgs, train_json, val_imgs, val_json, num_classes, dataset_yml)

    # 3. Find model config
    model_config = dfine_repo / "configs" / "dfine" / "custom" / "dfine_hgnetv2_m_glass.yml"
    if not model_config.exists():
        model_config = dfine_repo / "configs" / "dfine" / "dfine_hgnetv2_m_coco.yml"

    # 4. Run training
    print(f"[D-FINE] Starting training: {epochs} epochs, eval_size={eval_size}")
    cmd = [
        sys.executable, str(dfine_repo / "train.py"),
        "-c", str(model_config),
        "-t", str(Path(pretrained_weights).resolve()),
        "--use-amp",
        "--seed", "42",
        "--output-dir", str(output_dir),
        "--summary-dir", str(output_dir / "summary"),
    ]

    print(f"[D-FINE] Command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=str(dfine_repo))

    print(f"[D-FINE] Training complete. Output: {output_dir}")
    return output_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train D-FINE-M model")
    parser.add_argument("--data", required=True, help="Path to runtime_data.yaml")
    parser.add_argument("--dfine-repo", required=True, help="Path to D-FINE repository")
    parser.add_argument("--weights", required=True, help="Path to pretrained weights (dfine_m_coco.pth)")
    parser.add_argument("--output-dir", default="artifacts/runs/dfine_m", help="Output directory")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--eval-size", type=int, default=960)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    train_dfine(
        args.data, args.dfine_repo, args.weights, args.output_dir,
        epochs=args.epochs, eval_size=args.eval_size, batch_size=args.batch_size,
    )
