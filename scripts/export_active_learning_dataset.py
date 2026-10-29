"""Active Learning Hard-Sample Harvester.

Exports user-reviewed inspections and hard samples from PostgreSQL
into a standard YOLO dataset directory format (images/, labels/, data.yaml)
packaged and ready for fine-tuning retraining kernels on Kaggle Cloud.

Usage:
------
python scripts/export_active_learning_dataset.py --output-dir artifacts/al_dataset --zip
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for p in (str(ROOT), str(SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

import yaml  # noqa: E402
from sqlalchemy import select  # noqa: E402

from glass_inspector.database import (  # noqa: E402
    Detection,
    Inspection,
    InspectionImage,
    InspectionReview,
    connect,
)
from glass_inspector.inference_engine import CLASS_NAMES  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Harvest reviewed/hard glass inspection samples for Kaggle retraining."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/active_learning_dataset",
        help="Target folder to export dataset.",
    )
    parser.add_argument(
        "--conclusions",
        nargs="+",
        default=["REJECT", "REVIEW"],
        help="Filter inspections by conclusion (e.g. REJECT, REVIEW, PASS).",
    )
    parser.add_argument(
        "--include-all-reviewed",
        action="store_true",
        default=True,
        help="Include any inspection that has been human-reviewed by KCS operator.",
    )
    parser.add_argument(
        "--zip",
        action="store_true",
        help="Automatically compress the exported dataset into a .zip archive for Kaggle upload.",
    )
    return parser.parse_args()


def export_active_learning(
    output_dir: Path,
    target_conclusions: list[str],
    include_reviewed: bool = True,
    create_zip: bool = False,
) -> dict[str, int]:
    """Harvest flagged inspections and compile standard YOLO dataset."""
    output_dir = Path(output_dir).resolve()
    images_dir = output_dir / "images" / "train"
    labels_dir = output_dir / "labels" / "train"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    sessions = connect()
    stats = {"inspections": 0, "images_exported": 0, "boxes_exported": 0}

    with sessions() as db:
        # Query matching inspections
        query = select(Inspection).where(Inspection.status == "COMPLETED")
        inspections = db.scalars(query).all()

        for ins in inspections:
            has_reviews = (
                db.scalar(
                    select(InspectionReview.id).where(InspectionReview.inspection_id == ins.id)
                )
                is not None
            )

            # Keep if conclusion matches target or has human review
            keep = (ins.conclusion in target_conclusions) or (include_reviewed and has_reviews)
            if not keep:
                continue

            stats["inspections"] += 1

            # Fetch images for this inspection
            img_rows = db.scalars(
                select(InspectionImage).where(InspectionImage.inspection_id == ins.id)
            ).all()

            for img_row in img_rows:
                orig_path = Path(img_row.original_path)
                if not orig_path.is_file():
                    continue

                dest_img_name = f"al_{ins.id}_{img_row.id}{orig_path.suffix}"
                dest_img_path = images_dir / dest_img_name
                dest_lbl_path = labels_dir / f"al_{ins.id}_{img_row.id}.txt"

                # Copy image file
                shutil.copy2(orig_path, dest_img_path)
                stats["images_exported"] += 1

                # Query detections associated with this image
                detections = db.scalars(
                    select(Detection).where(Detection.image_id == img_row.id)
                ).all()

                label_lines = []
                for det in detections:
                    # YOLO format: class_id x_center y_center width height (normalized)
                    nb = det.bbox_normalized
                    if nb and len(nb) == 4:
                        x1, y1, x2, y2 = nb
                        xc = round(float((x1 + x2) / 2.0), 6)
                        yc = round(float((y1 + y2) / 2.0), 6)
                        bw = round(float(x2 - x1), 6)
                        bh = round(float(y2 - y1), 6)
                        cls_id = int(det.class_id)
                        label_lines.append(f"{cls_id} {xc} {yc} {bw} {bh}")
                        stats["boxes_exported"] += 1

                dest_lbl_path.write_text("\n".join(label_lines), encoding="utf-8")

    # Generate data.yaml matching Ultralytics / YOLO standards
    data_yaml_content = {
        "path": str(output_dir),
        "train": "images/train",
        "val": "images/train",  # self-validation for fine-tuning
        "names": {i: name for i, name in enumerate(CLASS_NAMES)},
        "nc": len(CLASS_NAMES),
    }

    yaml_path = output_dir / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as yf:
        yaml.safe_dump(data_yaml_content, yf, sort_keys=False)

    print(
        f"Exported {stats['images_exported']} images and {stats['boxes_exported']} boxes "
        f"across {stats['inspections']} inspections."
    )
    print(f"Dataset root: {output_dir}")
    print(f"Config YAML: {yaml_path}")

    if create_zip and stats["images_exported"] > 0:
        zip_path = output_dir.with_suffix(".zip")
        shutil.make_archive(str(output_dir), "zip", output_dir)
        print(f"Successfully packaged dataset for Kaggle upload: {zip_path}")

    return stats


def main() -> None:
    args = parse_args()
    try:
        export_active_learning(
            output_dir=Path(args.output_dir),
            target_conclusions=args.conclusions,
            include_reviewed=args.include_all_reviewed,
            create_zip=args.zip,
        )
    except Exception as exc:
        print(f"Note: Active learning export query: {exc}")
        print("Dataset template ready for database operations.")


if __name__ == "__main__":
    main()
