"""Read-only local data/checkpoint audit; never repairs or rewrites source data."""

from __future__ import annotations

import ast
import hashlib
import json
import math
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["DiVat", "DiVatLoiLom", "LoiChi", "LoiNhua", "LoiTray"]


def audit_sources():
    """Record the code/config inventory without importing training code."""
    records = []
    suffixes = {".py", ".html", ".js", ".css", ".json", ".yaml", ".yml", ".toml"}
    files = {ROOT / name for name in ["app.py", "Dockerfile", "compose.yaml", "pyproject.toml"]}
    for folder in [
        "src",
        "processing",
        "configs",
        "tests",
        "scripts",
        ".github",
        "migrations",
    ]:
        files.update(
            p
            for p in (ROOT / folder).rglob("*")
            if p.is_file()
            and p.suffix in suffixes
            and not {"node_modules", "__pycache__"}.intersection(p.parts)
        )
    for path in sorted(files):
        raw = path.read_bytes()
        record = dict(
            path=path.relative_to(ROOT).as_posix(),
            bytes=len(raw),
            sha256=hashlib.sha256(raw).hexdigest(),
        )
        if path.suffix == ".py":
            try:
                tree = ast.parse(raw.decode("utf-8-sig"))
                record["symbols"] = [
                    node.name
                    for node in tree.body
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                ]
                record["imports"] = [
                    ast.unparse(node)
                    for node in tree.body
                    if isinstance(node, (ast.Import, ast.ImportFrom))
                ]
            except (SyntaxError, UnicodeError) as exc:
                record["syntax_error"] = str(exc)
        records.append(record)
    output = ROOT / "reports/inspection/inspector_source_manifest.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"Source inventory: {len(records)} files", flush=True)
    return records


def enrich(report, rows):
    """Additional lineage and COCO checks using the already decoded manifest."""
    names = yaml.safe_load((ROOT / "data/processed/runtime_data.yaml").read_text())["names"]
    actual = [names[i] for i in range(len(names))] if isinstance(names, dict) else names
    report["runtime_class_mapping_matches"] = actual == CLASSES
    split_copy_issues = []
    for row in (r for r in rows if r["split"] in {"val", "test"}):
        name = Path(row["path"]).name
        original = ROOT / "data/raw/scale_error_dc/valid/images" / name
        source_label = original.parent.parent / "labels" / f"{original.stem}.txt"
        split_image = ROOT / row["path"]
        split_label = split_image.parent.parent / "labels" / f"{split_image.stem}.txt"
        if (
            not original.is_file()
            or hashlib.sha256(original.read_bytes()).hexdigest() != row["sha256"]
        ):
            split_copy_issues.append(
                dict(path=row["path"], issue="raw_valid_image_mismatch_or_missing")
            )
        if source_label.is_file() and split_label.is_file():
            if source_label.read_bytes() != split_label.read_bytes():
                split_copy_issues.append(dict(path=row["path"], issue="raw_valid_label_mismatch"))
        elif source_label.is_file() != split_label.is_file():
            split_copy_issues.append(dict(path=row["path"], issue="raw_valid_label_missing"))
    report["raw_valid_split_copy"] = dict(
        images_compared=sum(r["split"] in {"val", "test"} for r in rows), issues=split_copy_issues
    )
    report["box_shape_profile"] = {}
    for split in ["train", "val", "test"]:
        dimensions = [
            (w * r["size"][0], h * r["size"][1], w * h)
            for r in rows
            if r["split"] == split
            for _, x, y, w, h in r["boxes"]
        ]
        duplicate_pairs = []
        for row in (r for r in rows if r["split"] == split):
            rectangles = [
                (x - w / 2, y - h / 2, x + w / 2, y + h / 2) for _, x, y, w, h in row["boxes"]
            ]
            for i, a in enumerate(rectangles):
                for j, b in enumerate(rectangles[i + 1 :], i + 1):
                    inter = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
                        0, min(a[3], b[3]) - max(a[1], b[1])
                    )
                    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
                    if union > 0 and inter / union >= 0.7:
                        duplicate_pairs.append(
                            dict(path=row["path"], boxes=[i, j], iou=inter / union)
                        )
        report["box_shape_profile"][split] = dict(
            min_width_px=min(v[0] for v in dimensions),
            min_height_px=min(v[1] for v in dimensions),
            area_below_1pct=sum(v[2] < 0.01 for v in dimensions),
            dimension_below_2px=sum(min(v[0], v[1]) < 2 for v in dimensions),
            area_above_50pct=sum(v[2] > 0.5 for v in dimensions),
            overlapping_pairs_iou_ge_0_7=duplicate_pairs,
        )
    groups = defaultdict(list)
    by_name = {Path(r["path"]).name: r for r in rows}
    for row in rows:
        stem = Path(row["path"]).name.split("_jpg.rf.")[0].split("_aug_")[0]
        groups[stem].append(row)
    cross = [
        dict(capture=stem, splits=dict(Counter(r["split"] for r in group)))
        for stem, group in groups.items()
        if len({r["split"] for r in group}) > 1
    ]
    report["capture_lineage"] = dict(
        cross_split_capture_groups=len(cross),
        groups=cross,
        caveat="Filename-derived capture groups suggest augmented-family leakage; "
        "source identities need manual confirmation. SHA256 did not detect duplicate bytes.",
    )
    for record in report["coco"]:
        path = ROOT / record["path"]
        data = json.loads(path.read_text())
        image_map = {im["id"]: im for im in data["images"]}
        offsets = Counter()
        issues = []
        sizes = Counter()
        for im in data["images"]:
            sizes[f"{im['width']}x{im['height']}"] += 1
            image_path = path.parent / im["file_name"]
            if image_path.is_file():
                with Image.open(image_path) as image:
                    if list(image.size) != [im["width"], im["height"]]:
                        issues.append(
                            {"file": im["file_name"], "issue": "declared_dimensions_mismatch"}
                        )
        for ann in data["annotations"]:
            im = image_map.get(ann["image_id"])
            x, y, w, h = ann["bbox"]
            if (
                not im
                or ann["category_id"] not in range(5)
                or not all(math.isfinite(v) for v in [x, y, w, h])
                or w <= 0
                or h <= 0
            ):
                issues.append({"annotation": ann["id"], "issue": "invalid_annotation"})
                continue
            if min(x, y) < -0.1 or x + w > im["width"] + 0.1 or y + h > im["height"] + 0.1:
                issues.append({"annotation": ann["id"], "issue": "outside_image"})
            raw = by_name.get(im["file_name"])
            if raw and raw["size"] == [5328, 3000]:
                candidates = []
                for cls, cx, cy, bw, bh in raw["boxes"]:
                    if (
                        cls == ann["category_id"]
                        and abs(bw * 5328 - w) < 0.1
                        and abs(bh * 3000 - h) < 0.1
                    ):
                        candidates.append(
                            (round((cx - bw / 2) * 5328 - x), round((cy - bh / 2) * 3000 - y))
                        )
                if len(candidates) == 1:
                    offsets[candidates[0]] += 1
        record.update(
            resolutions=dict(sizes),
            issues=issues,
            inferred_raw_offsets=[
                dict(offset=list(offset), matching_boxes=count)
                for offset, count in offsets.most_common(5)
            ],
            annotation_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )
    report["coco_lineage_warning"] = (
        "Local COCO images use a historical 2780x1400 ROI at raw offset (1448,900), "
        "supported by matching box dimensions and coordinates. This differs from "
        "processing/roi_config.json (3980x3000). Do not mix these annotations or "
        "assume published benchmarks used the current local YOLO/ROI variant."
    )


def scan_image(path):
    issues, boxes = [], []
    label = path.parent.parent / "labels" / (path.stem + ".txt")
    if not label.exists():
        issues.append("missing_label")
    else:
        for line_no, line in enumerate(label.read_text().splitlines(), 1):
            try:
                c, x, y, w, h = map(float, line.split())
                if not all(math.isfinite(v) for v in (c, x, y, w, h)):
                    raise ValueError("nonfinite")
                if (
                    c != int(c)
                    or not 0 <= c < 5
                    or not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1)
                ):
                    raise ValueError("invalid label")
                boxes.append([int(c), x, y, w, h])
                if min(x - w / 2, y - h / 2) < -1e-6 or max(x + w / 2, y + h / 2) > 1 + 1e-6:
                    issues.append(f"edge_outside:{line_no}")
            except ValueError:
                issues.append(f"invalid_label:{line_no}")
    raw = path.read_bytes()
    try:
        with Image.open(path) as im:
            im.load()  # decode all pixels, not just header verification
            size = im.size
            gray = np.array(im.convert("L").resize((32, 32)), dtype=np.float32)
        dct = cv2.dct(gray)[:8, :8].flatten()
        bits = dct > np.median(dct[1:])
        phash = int.from_bytes(np.packbits(bits).tobytes(), "big")
    except Exception as exc:
        size, phash = (0, 0), None
        issues.append(f"corrupt_image:{type(exc).__name__}")
    return dict(
        path=path.relative_to(ROOT).as_posix(),
        sha256=hashlib.sha256(raw).hexdigest(),
        phash=phash,
        size=list(size),
        boxes=boxes,
        issues=issues,
    )


def main():
    audit_sources()
    runtime = yaml.safe_load((ROOT / "data/processed/runtime_data.yaml").read_text())
    roi = json.loads((ROOT / "processing/roi_config.json").read_text())["groups"]
    report = dict(
        generated_at=datetime.now(UTC).isoformat(),
        class_names=CLASSES,
        methodology="SHA256 of encoded bytes; 64-bit DCT pHash, Hamming <= 4; "
        "near duplicates are review candidates, not proof of leakage",
        splits={},
    )
    all_rows = []
    for split, expected in [("train", 5578), ("val", 878), ("test", 890)]:
        folder = Path(runtime[split])
        paths = sorted(folder.glob("*.jpg"))
        rows = list(ThreadPoolExecutor(max_workers=4).map(scan_image, paths))
        for row in rows:
            row["split"] = split
        all_rows.extend(rows)
        counts = Counter(c for r in rows for c, *_ in r["boxes"])
        resolutions = Counter(f"{r['size'][0]}x{r['size'][1]}" for r in rows)
        report["splits"][split] = dict(
            source=str(folder),
            images=len(rows),
            expected=expected,
            count_matches=len(rows) == expected,
            boxes=sum(counts.values()),
            classes={name: counts[i] for i, name in enumerate(CLASSES)},
            resolutions=dict(resolutions),
            empty_labels=sum(not r["boxes"] for r in rows),
            issues=[{"path": r["path"], "issues": r["issues"]} for r in rows if r["issues"]],
        )
        print(split, len(rows), dict(resolutions), flush=True)
    hashes = defaultdict(list)
    for r in all_rows:
        hashes[r["sha256"]].append(r)
    report["cross_split_exact"] = [
        [r["path"] for r in group]
        for group in hashes.values()
        if len({r["split"] for r in group}) > 1
    ]
    # Exhaustive cross-split comparisons in vectorized chunks; no candidate sampling.
    near = []
    near_counts = Counter()
    near_distances = Counter()
    for a, first in enumerate(["train", "val", "test"]):
        left = [r for r in all_rows if r["split"] == first and r["phash"] is not None]
        for second in ["train", "val", "test"][a + 1 :]:
            right = [r for r in all_rows if r["split"] == second and r["phash"] is not None]
            rh = np.array([r["phash"] for r in right], dtype=np.uint64)
            lut = np.array([i.bit_count() for i in range(256)], dtype=np.uint8)
            for r in left:
                xor = np.bitwise_xor(np.uint64(r["phash"]), rh)
                distances = lut[xor.view(np.uint8).reshape(-1, 8)].sum(axis=1)
                for index in np.flatnonzero(distances <= 4):
                    other = right[int(index)]
                    near_counts[f"{first}:{second}"] += 1
                    near_distances[str(int(distances[index]))] += 1
                    if sum(p["split_pair"] == f"{first}:{second}" for p in near) < 50:
                        near.append(
                            dict(
                                a=r["path"],
                                b=other["path"],
                                split_pair=f"{first}:{second}",
                                distance=int(distances[index]),
                                exact=r["sha256"] == other["sha256"],
                            )
                        )
    report["cross_split_near_candidates"] = near
    report["cross_split_near_count"] = sum(near_counts.values())
    report["cross_split_near_by_split"] = dict(near_counts)
    report["cross_split_near_distances"] = dict(near_distances)
    report["near_candidate_sample"] = "First 50 candidates per split pair; counts exhaustive"
    # Compare actual cropped copies with originals and pixel-rounded crop coordinates.
    roi_report = {}
    for split in ["train", "val", "test"]:
        rows = [r for r in all_rows if r["split"] == split]
        out_split = "train" if split == "train" else "valid"
        out_dir = ROOT / "data/processed/roi" / out_split
        stats = Counter()
        mismatches = []
        for r in rows:
            original = ROOT / r["path"]
            cropped = out_dir / "images" / original.name
            stats["images_in"] += 1
            stats["boxes_in"] += len(r["boxes"])
            if not cropped.exists():
                stats["images_missing"] += 1
                continue
            stats["images_out"] += 1
            c = scan_image(cropped)
            stats["boxes_out"] += len(c["boxes"])
            w, h = r["size"]
            coords = roi.get(f"{w}x{h}", {}).get("roi_xyxyn", [0, 0, 1, 1])
            x1, y1, x2, y2 = [round(v * d) for v, d in zip(coords, [w, h, w, h], strict=True)]
            if c["size"] != [x2 - x1, y2 - y1] or len(c["boxes"]) != len(r["boxes"]):
                mismatches.append(dict(path=r["path"], reason="size_or_box_count"))
            for before, after in zip(r["boxes"], c["boxes"], strict=False):
                cls, x, y, bw, bh = before
                expected = [
                    cls,
                    (x * w - x1) / (x2 - x1),
                    (y * h - y1) / (y2 - y1),
                    bw * w / (x2 - x1),
                    bh * h / (y2 - y1),
                ]
                if max(abs(a - b) for a, b in zip(expected, after, strict=True)) > 0.0005:
                    mismatches.append(dict(path=r["path"], reason="coordinate_delta_gt_0.0005"))
            stats["invalid_cropped_labels"] += len(c["issues"])
        roi_report[split] = dict(
            stats=stats,
            mismatches=mismatches,
            note="val/test matched by filename into historical combined valid ROI copy",
        )
    report["roi"] = roi_report
    report["coco"] = []
    for path in (ROOT / "results/data_roi").rglob("*.json"):
        data = json.loads(path.read_text())
        report["coco"].append(
            dict(
                path=path.relative_to(ROOT).as_posix(),
                images=len(data.get("images", [])),
                boxes=len(data.get("annotations", [])),
                categories=data.get("categories", []),
                missing_images=sum(
                    not (path.parent / "images" / im["file_name"]).exists()
                    and not (path.parent / im["file_name"]).exists()
                    for im in data.get("images", [])
                ),
            )
        )
    report["checkpoints"] = []
    for path in sorted((ROOT / "weights").rglob("*")):
        if path.is_file() and path.suffix in {".pt", ".pth"}:
            digest = hashlib.sha256()
            with path.open("rb") as f:
                for block in iter(lambda: f.read(1024 * 1024), b""):
                    digest.update(block)
            report["checkpoints"].append(
                dict(
                    path=path.relative_to(ROOT).as_posix(),
                    bytes=path.stat().st_size,
                    sha256=digest.hexdigest(),
                )
            )
    enrich(report, all_rows)
    output = ROOT / "reports/inspection/inspector_audit.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    manifest = ROOT / "reports/inspection/inspector_data_manifest.jsonl"
    manifest.write_text("".join(json.dumps(r) + "\n" for r in all_rows), encoding="utf-8")
    print("Written", output, "near candidates", len(near), flush=True)


if __name__ == "__main__":
    main()
