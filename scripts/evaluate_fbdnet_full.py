"""Evaluate the original FBDNet checkpoint, including reproducible detector Grad-CAM."""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from evaluate_fbdnet import (
    CLASS_NAMES,
    ROOT,
    decode_roi,
    load_dfine_model,
    prepare_ground_truth,
    sha256,
)
from PIL import ImageDraw
from torchvision import transforms

from scale_error.evaluation.detection_confusion import (
    box_iou_xywh,
    confusion_statistics,
    detection_confusion,
)

COLORS = ["#2463a6", "#b27c16", "#bd5827", "#687b38", "#b35487"]


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def write_csv(path, rows):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def valid_mean(values):
    values = np.asarray(values)
    valid = values[values >= 0]
    return float(valid.mean()) if valid.size else None


def evaluate_coco(truth, predictions):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    coco = COCO()
    coco.dataset = truth
    coco.createIndex()
    if predictions:
        detected = coco.loadRes(predictions)
    else:
        detected = COCO()
        detected.dataset = dict(
            images=truth["images"], categories=truth["categories"], annotations=[]
        )
        detected.createIndex()
    evaluator = COCOeval(coco, detected, "bbox")
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    keys = [
        "mAP50_95",
        "mAP50",
        "mAP75",
        "AP_small",
        "AP_medium",
        "AP_large",
        "AR1",
        "AR10",
        "AR100",
        "AR_small",
        "AR_medium",
        "AR_large",
    ]
    overall = {
        key: float(value) if value >= 0 else None
        for key, value in zip(keys, evaluator.stats, strict=True)
    }
    precision, recall = evaluator.eval["precision"], evaluator.eval["recall"]
    per_class = []
    for k, name in enumerate(CLASS_NAMES):
        row = dict(
            class_name=name,
            mAP50_95=valid_mean(precision[:, :, k, 0, -1]),
            mAP50=valid_mean(precision[0, :, k, 0, -1]),
            mAP75=valid_mean(precision[5, :, k, 0, -1]),
        )
        for area, suffix in enumerate(["all", "small", "medium", "large"]):
            if area:
                row[f"AP_{suffix}"] = valid_mean(precision[:, :, k, area, -1])
            row[f"AR_{suffix}"] = valid_mean(recall[:, k, area, -1])
        per_class.append(row)
    return overall, per_class, evaluator


def plot_confusion(output, matrix, confidence, iou, prefix="confusion"):
    labels = CLASS_NAMES + ["Background"]
    normalized = np.divide(
        matrix,
        matrix.sum(1, keepdims=True),
        out=np.zeros_like(matrix, dtype=float),
        where=matrix.sum(1, keepdims=True) > 0,
    )
    for name, values, vmax in [
        ("counts", matrix, max(1, matrix.max())),
        ("normalized", normalized * 100, 100),
    ]:
        fig, ax = plt.subplots(figsize=(9, 8), dpi=180)
        im = ax.imshow(values, cmap="Blues", vmin=0, vmax=vmax)
        ax.set_xticks(range(6), labels, rotation=30, ha="right")
        ax.set_yticks(range(6), labels)
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("Ground truth class")
        ax.set_title(
            f"FBDNet-M | Test confusion matrix ({name})\nConfidence >= {confidence:g}, IoU >= {iou:g}"
        )
        for i in range(6):
            for j in range(6):
                label = f"{values[i, j]:.1f}%" if name == "normalized" else str(int(values[i, j]))
                ax.text(
                    j,
                    i,
                    label,
                    ha="center",
                    va="center",
                    color="white" if values[i, j] > vmax * 0.55 else "#15334d",
                )
        fig.colorbar(
            im, ax=ax, shrink=0.8, label="Row percentage" if name == "normalized" else "Boxes"
        )
        fig.tight_layout()
        fig.savefig(output / f"{prefix}_{name}.png", bbox_inches="tight")
        plt.close(fig)
    write_csv(
        output / f"{prefix}_matrix.csv",
        [
            dict(ground_truth=name, **dict(zip(labels, row.tolist(), strict=True)))
            for name, row in zip(labels, matrix, strict=True)
        ],
    )


def curves(output, ground_truth, predictions, evaluator, iou):
    pr_rows = []
    fig, ax = plt.subplots(figsize=(8, 6), dpi=180)
    for k, name in enumerate(CLASS_NAMES):
        values = evaluator.eval["precision"][0, :, k, 0, -1]
        valid = values >= 0
        ax.plot(evaluator.params.recThrs[valid], values[valid], color=COLORS[k], label=name)
        pr_rows += [
            dict(class_name=name, recall=float(r), precision=float(p), iou=0.5)
            for r, p in zip(evaluator.params.recThrs[valid], values[valid], strict=True)
        ]
    ax.set(
        xlim=(0, 1),
        ylim=(0, 1),
        xlabel="Recall",
        ylabel="COCO interpolated precision",
        title="FBDNet-M | PR curves at IoU 0.50 (maxDets=100)",
    )
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / "pr_curve.png")
    plt.close(fig)
    if pr_rows:
        write_csv(output / "pr_curve.csv", pr_rows)
    rows = []
    thresholds = sorted(
        set([0.0, 0.25, 0.5, 0.99] + [round(x, 2) for x in np.arange(0.05, 1, 0.05)])
    )
    for threshold in thresholds:
        matrix, _ = detection_confusion(
            ground_truth["images"],
            ground_truth["annotations"],
            predictions,
            confidence=threshold,
            iou=iou,
        )
        per_class, totals = confusion_statistics(matrix, CLASS_NAMES)
        rows += [
            dict(
                confidence=threshold,
                class_name=row["class_name"],
                precision=row["precision"],
                recall=row["recall"],
                f1=row["f1"],
                tp=row["tp"],
                fp=row["fp"],
                fn=row["fn"],
            )
            for row in per_class
        ]
        rows.append(
            dict(
                confidence=threshold,
                class_name="micro",
                **{key: totals[key] for key in ["precision", "recall", "f1", "tp", "fp", "fn"]},
            )
        )
    write_csv(output / "confidence_sweep.csv", rows)
    for metric in ["precision", "recall", "f1"]:
        fig, ax = plt.subplots(figsize=(8, 6), dpi=180)
        for k, name in enumerate(CLASS_NAMES + ["micro"]):
            subset = [row for row in rows if row["class_name"] == name]
            ax.plot(
                [row["confidence"] for row in subset],
                [row[metric] for row in subset],
                color=COLORS[k] if k < 5 else "#292929",
                label=name,
                linewidth=1.6 if k < 5 else 2.5,
            )
        ax.set(
            xlim=(0, 1),
            ylim=(0, 1),
            xlabel="Confidence threshold",
            ylabel=metric.capitalize(),
            title=f"FBDNet-M | {metric.upper()} vs confidence (IoU {iou:g})",
        )
        ax.legend()
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(output / f"{metric}_confidence_curve.png")
        plt.close(fig)
    return max((row for row in rows if row["class_name"] == "micro"), key=lambda row: row["f1"])


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def gradcam_for_target(model, tensor, target, roi_size):
    """Explain one decoder query's class logit at the first AFTA feature (P3)."""
    activation = []
    hook = model.backbone.afta[0].register_forward_hook(
        lambda module, args, result: activation.append(result)
    )
    try:
        with torch.enable_grad():
            inputs = tensor.detach().clone().requires_grad_(True)
            outputs = model(inputs)
            boxes = outputs["pred_boxes"][0].detach().cpu().numpy()
            width, height = roi_size
            pixel_boxes = np.column_stack(
                [
                    (boxes[:, 0] - boxes[:, 2] / 2) * width,
                    (boxes[:, 1] - boxes[:, 3] / 2) * height,
                    boxes[:, 2] * width,
                    boxes[:, 3] * height,
                ]
            )
            overlaps = box_iou_xywh([target["bbox"]], pixel_boxes)[0]
            category = target["category_id"]
            scores = outputs["pred_logits"][0, :, category].sigmoid().detach().cpu().numpy()
            candidates = np.flatnonzero(overlaps >= 0.5)
            query = (
                int(candidates[np.argmax(scores[candidates])])
                if len(candidates)
                else int(overlaps.argmax())
            )
            logit = outputs["pred_logits"][0, query, category]
            feature = activation[0]
            (gradient,) = torch.autograd.grad(logit, feature)
            weights = gradient.mean(dim=(-2, -1), keepdim=True)
            cam = torch.relu((weights * feature).sum(1))[0].detach().cpu().numpy()
            if not np.isfinite(cam).all() or not torch.isfinite(gradient).all():
                raise RuntimeError("Nonfinite Grad-CAM")
            dynamic_range = float(cam.max() - cam.min())
            cam = (cam - cam.min()) / dynamic_range if dynamic_range > 1e-12 else np.zeros_like(cam)
            cam = torch.nn.functional.interpolate(
                torch.from_numpy(cam)[None, None],
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )[0, 0].numpy()
            return cam, dict(
                query=query,
                target_class=CLASS_NAMES[category],
                score=float(scores[query]),
                query_gt_iou=float(overlaps[query]),
                target_logit=float(logit.detach()),
                target_layer="backbone.afta.0 (P3)",
                nonzero_gradient=bool(gradient.abs().sum() > 0),
                flat_cam=dynamic_range <= 1e-12,
                predicted_bbox=pixel_boxes[query].tolist(),
            )
    finally:
        hook.remove()


def render_examples(
    output, model, transform, manifest, truth, predictions, device, per_class_count
):
    directory = output / "gradcam"
    directory.mkdir()
    annotations, by_image = defaultdict(list), defaultdict(list)
    for annotation in truth["annotations"]:
        annotations[annotation["category_id"]].append(annotation)
    for prediction in predictions:
        by_image[prediction["image_id"]].append(prediction)
    records = {row["image_id"]: row for row in manifest}
    images = {row["id"]: row for row in truth["images"]}
    chosen, results = [], []
    for category in range(5):
        candidates = annotations[category]
        ranked = []
        for target in candidates:
            same_class = [p for p in by_image[target["image_id"]] if p["category_id"] == category]
            overlaps = box_iou_xywh([target["bbox"]], [p["bbox"] for p in same_class])[0]
            confidence = max(
                (
                    p["score"]
                    for p, overlap in zip(same_class, overlaps, strict=True)
                    if overlap >= 0.5
                ),
                default=0.0,
            )
            image = images[target["image_id"]]
            ranked.append((confidence, target["area"] / (image["width"] * image["height"]), target))
        if not ranked:
            continue
        # Confident localization, difficult localization, and smallest relative-area defect.
        orders = [
            sorted(ranked, key=lambda r: (-r[0], r[2]["id"])),
            sorted(ranked, key=lambda r: (r[0], r[2]["id"])),
            sorted(ranked, key=lambda r: (r[1], r[2]["id"])),
        ]
        used = set()
        for index in range(per_class_count):
            available = [row for row in orders[index % 3] if row[2]["image_id"] not in used]
            if not available:
                break
            row = available[0]
            used.add(row[2]["image_id"])
            chosen.append((row[2], ["highest-score", "difficult", "smallest"][index % 3]))
    for target, reason in chosen:
        record, image = records[target["image_id"]], images[target["image_id"]]
        original = decode_roi(record)
        tensor = transform(original).unsqueeze(0).to(device)
        cam, detail = gradcam_for_target(model, tensor, target, (image["width"], image["height"]))
        marked = original.copy()
        drawing = ImageDraw.Draw(marked)
        x, y, w, h = target["bbox"]
        drawing.rectangle([x, y, x + w, y + h], outline="#25ce68", width=3)
        px, py, pw, ph = detail["predicted_bbox"]
        drawing.rectangle([px, py, px + pw, py + ph], outline="#ffad22", width=3)
        rgb = np.asarray(original) / 255.0
        colored = plt.get_cmap("inferno")(cam)[..., :3]
        overlay = np.clip(0.6 * rgb + 0.4 * colored, 0, 1)
        fig, axes = plt.subplots(1, 4, figsize=(18, 5), dpi=160)
        axes[0].imshow(marked)
        axes[0].set_title("ROI | GT: green, target query: orange")
        axes[1].imshow(cam, cmap="inferno", vmin=0, vmax=1)
        axes[1].set_title("Grad-CAM | AFTA P3")
        axes[2].imshow(overlay)
        axes[2].set_title("Overlay")
        axes[3].imshow(overlay)
        padding = max(w, h, 20) * 1.5
        axes[3].set_xlim(max(0, x - padding), min(image["width"], x + w + padding))
        axes[3].set_ylim(min(image["height"], y + h + padding), max(0, y - padding))
        axes[3].set_title("Zoom around ground truth")
        for ax in axes:
            ax.axis("off")
        fig.suptitle(
            f"FBDNet-M | {detail['target_class']} | {reason} | score {detail['score']:.4f} | IoU {detail['query_gt_iou']:.3f}"
        )
        fig.tight_layout()
        filename = f"{detail['target_class']}_{reason}_image{target['image_id']}.png"
        fig.savefig(directory / filename, bbox_inches="tight")
        plt.close(fig)
        np.savez_compressed(directory / filename.replace(".png", ".npz"), cam=cam)
        results.append(
            detail
            | dict(
                file=filename,
                image_id=target["image_id"],
                annotation_id=target["id"],
                source_image=record["path"],
                selection=reason,
                ground_truth_bbox=target["bbox"],
            )
        )
        print(
            f"GRADCAM {len(results)}/{len(chosen)} {filename} flat={detail['flat_cam']}", flush=True
        )
    dump(directory / "manifest.json", results)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=ROOT / "weights/dfine_m_fbdnet_best.pth")
    parser.add_argument("--data", type=Path, default=ROOT / "data/processed/runtime_data.yaml")
    parser.add_argument("--roi", type=Path, default=ROOT / "processing/roi_config.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--gradcam-per-class", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--benchmark-repeats", type=int, default=30)
    parser.add_argument("--limit", type=int, help="Explicit smoke subset; never a full-test result")
    args = parser.parse_args()
    if (
        args.batch_size < 1
        or args.threads < 1
        or args.benchmark_repeats < 1
        or args.warmup < 0
        or args.gradcam_per_class < 0
    ):
        parser.error("Invalid resource limits")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if not 0 <= args.confidence <= 1 or not 0 <= args.iou <= 1:
        parser.error("Thresholds must be in [0, 1]")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("GPU requested but unavailable; refusing CPU fallback")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise RuntimeError("Output directory must be empty")
    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    np.random.seed(42)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    truth, manifest = prepare_ground_truth(args.data, args.roi, args.limit)
    checkpoint_hash = sha256(args.weights)
    bundle_manifest_path = ROOT / "bundle_manifest.json"
    if bundle_manifest_path.exists():
        bundle = json.loads(bundle_manifest_path.read_text(encoding="utf-8"))
        if checkpoint_hash != bundle["checkpoint_sha256"]:
            raise RuntimeError("Uploaded checkpoint hash mismatch")
        expected = {row["path"]: row for row in bundle["input_manifest"]}
        for row in manifest:
            if any(
                row[key] != expected[row["path"]][key]
                for key in ["sha256", "label_sha256", "crop_xyxy", "boxes"]
            ):
                raise RuntimeError(f"Uploaded test input mismatch: {row['path']}")
    dump(output / "ground_truth.json", truth)
    model, postprocessor = load_dfine_model(str(args.weights))
    if (
        type(model.backbone).__name__ != "HGNetv2FBD"
        or type(model.encoder).__name__ != "FBDHybridEncoder"
    ):
        raise RuntimeError("Unexpected architecture")
    model.to(device).eval()
    postprocessor.to(device).eval()
    protocol = dict(
        status="PROCESSING",
        pilot=bool(args.limit),
        checkpoint=args.weights.name,
        checkpoint_sha256=checkpoint_hash,
        roi_sha256=sha256(args.roi),
        images=len(manifest),
        boxes_before_roi=sum(row["boxes"] for row in manifest),
        boxes_after_roi=len(truth["annotations"]),
        input_size=[960, 960],
        confidence=args.confidence,
        iou=args.iou,
        device=str(device),
        device_name=torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        precision="float32; no AMP; TF32 disabled",
        batch_size=args.batch_size,
        architecture=dict(
            backbone=type(model.backbone).__name__,
            encoder=type(model.encoder).__name__,
            decoder=type(model.decoder).__name__,
        ),
        parameters=sum(p.numel() for p in model.parameters()),
        checkpoint_bytes=args.weights.stat().st_size,
        matching="Class-agnostic descending IoU, one-to-one; rows=GT, columns=predicted",
        coco_protocol="IoU .50:.95, maxDets 1/10/100; all prediction scores",
        input_manifest=manifest,
        torch_version=torch.__version__,
        training=False,
        runtime_source_sha256={
            p.relative_to(ROOT).as_posix(): sha256(p)
            for p in sorted((ROOT / "results/D-FINE/src").rglob("*.py"))
        },
    )
    dump(output / "protocol.json", protocol)
    print(
        f"AUDIT PASS {protocol['images']} images / {protocol['boxes_after_roi']} boxes | {checkpoint_hash}",
        flush=True,
    )
    print(
        f"ARCHITECTURE {protocol['architecture']} | parameters {protocol['parameters']} | {protocol['device_name']}",
        flush=True,
    )
    transform = transforms.Compose([transforms.Resize((960, 960)), transforms.ToTensor()])
    sample = transform(decode_roi(manifest[0])).unsqueeze(0).to(device)
    sample_size = torch.tensor(
        [[truth["images"][0]["width"], truth["images"][0]["height"]]], device=device
    )
    latencies = []
    with torch.inference_mode():
        for _ in range(args.warmup):
            postprocessor(model(sample), sample_size)
        for _ in range(args.benchmark_repeats):
            synchronize(device)
            started = time.perf_counter()
            postprocessor(model(sample), sample_size)
            synchronize(device)
            latencies.append((time.perf_counter() - started) * 1000)
    predictions, forward_seconds = [], []
    begin_all = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with (
        (output / "predictions.jsonl").open("x", encoding="utf-8") as stream,
        torch.inference_mode(),
    ):
        for offset in range(0, len(manifest), args.batch_size):
            batch = manifest[offset : offset + args.batch_size]
            inputs = torch.stack([transform(decode_roi(row)) for row in batch]).to(device)
            sizes = torch.tensor(
                [
                    [
                        row["crop_xyxy"][2] - row["crop_xyxy"][0],
                        row["crop_xyxy"][3] - row["crop_xyxy"][1],
                    ]
                    for row in batch
                ],
                device=device,
            )
            synchronize(device)
            started = time.perf_counter()
            labels, boxes, scores = postprocessor(model(inputs), sizes)
            synchronize(device)
            forward_seconds.append(time.perf_counter() - started)
            for index, row in enumerate(batch):
                detected = []
                for category, box, score in zip(
                    labels[index].tolist(),
                    boxes[index].tolist(),
                    scores[index].tolist(),
                    strict=True,
                ):
                    if (
                        not 0 <= category < 5
                        or not np.isfinite(box).all()
                        or not np.isfinite(score)
                    ):
                        raise RuntimeError("Invalid prediction")
                    x1, y1, x2, y2 = box
                    detected.append(
                        dict(
                            image_id=row["image_id"],
                            category_id=int(category),
                            bbox=[x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)],
                            score=score,
                        )
                    )
                predictions.extend(detected)
                stream.write(
                    json.dumps(
                        dict(image_id=row["image_id"], predictions=detected), allow_nan=False
                    )
                    + "\n"
                )
            stream.flush()
            if offset % (args.batch_size * 10) == 0 or offset + len(batch) == len(manifest):
                print(
                    f"INFERENCE {offset + len(batch)}/{len(manifest)} elapsed={time.perf_counter() - begin_all:.1f}s",
                    flush=True,
                )
    inference_elapsed = time.perf_counter() - begin_all
    performance = dict(
        batch_forward_ms_per_image=1000 * sum(forward_seconds) / len(manifest),
        batch_forward_fps=len(manifest) / sum(forward_seconds),
        inference_with_io_seconds=inference_elapsed,
        inference_with_io_fps=len(manifest) / inference_elapsed,
        batch1_mean_ms=float(np.mean(latencies)),
        batch1_median_ms=float(np.median(latencies)),
        batch1_p95_ms=float(np.percentile(latencies, 95)),
        batch1_fps=1000 / float(np.mean(latencies)),
        batch1_raw_ms=latencies,
        warmup=args.warmup,
        benchmark_repeats=args.benchmark_repeats,
        timing_scope="Model + postprocessor; batch1 excludes image decode, resize, and host-to-device copy",
        peak_inference_gpu_memory_mb=torch.cuda.max_memory_allocated(device) / 2**20
        if device.type == "cuda"
        else None,
    )
    overall, per_class_ap, evaluator = evaluate_coco(truth, predictions)
    np.savez_compressed(
        output / "coco_eval_arrays.npz",
        precision=evaluator.eval["precision"],
        recall=evaluator.eval["recall"],
        scores=evaluator.eval["scores"],
        iou_thresholds=evaluator.params.iouThrs,
        recall_thresholds=evaluator.params.recThrs,
    )
    matrix, errors = detection_confusion(
        truth["images"], truth["annotations"], predictions, confidence=args.confidence, iou=args.iou
    )
    per_class_operating, totals = confusion_statistics(matrix, CLASS_NAMES)
    per_class = [
        ap | operating for ap, operating in zip(per_class_ap, per_class_operating, strict=True)
    ]
    write_csv(output / "per_class_metrics.csv", per_class)
    plot_confusion(output, matrix, args.confidence, args.iou)
    alternative, _ = detection_confusion(
        truth["images"], truth["annotations"], predictions, confidence=0.25, iou=args.iou
    )
    plot_confusion(output, alternative, 0.25, args.iou, prefix="confusion_conf025")
    best_threshold = curves(output, truth, predictions, evaluator, args.iou)
    print("COCO METRICS\n" + json.dumps(overall, indent=2), flush=True)
    print("PER CLASS\n" + json.dumps(per_class, indent=2), flush=True)
    print("OPERATING METRICS\n" + json.dumps(totals, indent=2), flush=True)
    print("PERFORMANCE\n" + json.dumps(performance, indent=2), flush=True)
    gradcam = render_examples(
        output, model, transform, manifest, truth, predictions, device, args.gradcam_per_class
    )
    if sha256(args.weights) != checkpoint_hash:
        raise RuntimeError("Checkpoint changed during evaluation")
    protocol["status"] = "PILOT_COMPLETED" if args.limit else "COMPLETED"
    result = dict(
        generated_at=datetime.now(UTC).isoformat(),
        model="FBDNet-M",
        protocol=protocol,
        coco_metrics=overall,
        per_class=per_class,
        operating_metrics=totals,
        confusion_matrix=matrix.tolist(),
        image_errors=errors,
        performance=performance,
        best_micro_f1_on_test_sweep=best_threshold,
        gradcam=gradcam,
        predictions_sha256=sha256(output / "predictions.jsonl"),
        ground_truth_sha256=sha256(output / "ground_truth.json"),
    )
    dump(output / "evaluation.json", result)
    dump(output / "protocol.json", protocol)
    report = [
        "# FBDNet-M test evaluation" + (" — PILOT ONLY" if args.limit else ""),
        "",
        f"{len(manifest)} images / {len(truth['annotations'])} boxes. Checkpoint SHA256: `{checkpoint_hash}`.",
        "",
        "| COCO metric | Value |",
        "|---|---:|",
    ]
    report += [
        f"| {key} | {'N/A' if value is None else f'{value * 100:.4f}%'} |"
        for key, value in overall.items()
    ]
    report += [
        "",
        "| Class | GT | AP50 | AP75 | AP50:95 | Precision | Recall | F1 | TP | FP | FN |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in per_class:
        values = [
            "N/A" if row[k] is None else f"{row[k] * 100:.4f}%"
            for k in ["mAP50", "mAP75", "mAP50_95", "precision", "recall", "f1"]
        ]
        report.append(
            f"| {row['class_name']} | {row['support']} | "
            + " | ".join(values)
            + f" | {row['tp']} | {row['fp']} | {row['fn']} |"
        )
    report += [
        "",
        f"Operating point: confidence {args.confidence}, IoU {args.iou}. Micro P/R/F1: "
        f"{totals['precision'] * 100:.4f}% / {totals['recall'] * 100:.4f}% / {totals['f1'] * 100:.4f}%.",
        "",
        "![Confusion counts](confusion_counts.png)",
        "![Row-normalized confusion](confusion_normalized.png)",
        "![COCO PR](pr_curve.png)",
        "![F1 vs confidence](f1_confidence_curve.png)",
        "",
        f"Device: {protocol['device_name']}; FP32; input 960×960. Parameters: {protocol['parameters']:,}.",
        f"Batch1 mean/median/p95: {performance['batch1_mean_ms']:.4f}/{performance['batch1_median_ms']:.4f}/{performance['batch1_p95_ms']:.4f} ms. "
        f"Model + postprocessor FPS: {performance['batch1_fps']:.4f} (excludes IO/preprocessing/transfers).",
        "",
        "## Grad-CAM",
        "",
        "Target: one decoder query's class logit; feature: backbone.afta.0 (P3). "
        "Choose the highest class score among queries overlapping the target GT at IoU >= .5; otherwise the maximum-IoU query. "
        "GT labels select explanatory targets only and never enter inference or metrics. Maps do not prove model correctness.",
        "",
    ]
    report += [
        f"![{row['target_class']} {row['selection']}](gradcam/{row['file']})" for row in gradcam
    ]
    report += [
        "",
        "## Protocol and interpretation",
        "",
        "- Original local runtime and strict checkpoint load. No training, tuning, augmentation, NMS, or tiling.",
        "- Canonical original test images, local ROI, JPEG quality 95 for crops, RGB resize to 960×960, pixels /255.",
        "- COCO AP includes all scores; P/R/F1/confusion use the stated fixed threshold and class-agnostic one-to-one IoU matching.",
        "- Rows are GT; columns are predictions. Background column = unmatched GT; background row = unmatched detections. No object-detection true negatives.",
        "- Metrics with no applicable GT are N/A. FP/FN include class confusions.",
        "- The best F1 threshold in the sweep is descriptive on test, not a tuned deployment threshold.",
        "- Old 42.42% training-validation AP is not this held-out test result. Split capture/augmentation leakage was not resolved by this evaluation.",
        "- Exact unrounded numbers, per-class size AP/AR, timings, source hashes, and image errors are in evaluation.json; arrays and CSVs support recomputation.",
        "- No FLOPs figure is supplied: unsupported FFT/attention operators would make a generic partial count misleading.",
        "",
    ]
    (output / "report.md").write_text("\n".join(report), encoding="utf-8")
    artifact_rows = [
        dict(path=p.relative_to(output).as_posix(), bytes=p.stat().st_size, sha256=sha256(p))
        for p in sorted(output.rglob("*"))
        if p.is_file()
    ]
    dump(output / "artifacts.json", artifact_rows)
    print(
        f"{protocol['status']}: {output} | {len(gradcam)} Grad-CAM figures | {len(artifact_rows)} artifacts",
        flush=True,
    )


if __name__ == "__main__":
    main()
