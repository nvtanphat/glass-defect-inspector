#!/usr/bin/env python3
"""Evaluate D-FINE-M on Kaggle T4 GPU:
1. Load 44.4% D-FINE-M checkpoint
2. Test split ROI inference (890 images) on GPU
3. Calculate official COCO mAP (pycocotools)
4. Calculate and plot Confusion Matrix (IoU >= 0.5, conf >= 0.5)
5. Calculate per-class Precision, Recall, F1, TP, FP, FN
6. Render visual predictions on raw sample images
"""

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

WORKING = Path("/kaggle/working")
INPUT = Path("/kaggle/input")
REPO_DIR = WORKING / "D-FINE"

CLASSES = ["DiVat", "DiVatLoiLom", "LoiChi", "LoiNhua", "LoiTray"]
COLORS = ["#e63946", "#f4a261", "#2a9d8f", "#457b9d", "#9b5de5"]
CONF_TH = 0.5
IOU_TH = 0.5
BATCH = 16

ROI_CONFIG = {
    "1280x1024": (0.288281, 0.448242, 0.627344, 0.627930),
    "1920x1080": (0.334896, 0.414815, 0.677604, 0.638889),
    "1920x1200": (0.334375, 0.473333, 0.677083, 0.675000),
    "2048x1536": (0.329590, 0.496094, 0.650879, 0.654297),
    "2592x1944": (0.332562, 0.498457, 0.653935, 0.654835),
    "3840x2160": (0.334896, 0.499074, 0.656250, 0.655093),
}


def log(msg: str) -> None:
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {msg}", flush=True)


def setup_environment() -> None:
    log("Installing dependencies...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "pycocotools", "faster-coco-eval", "scikit-learn", "matplotlib", "seaborn", "calflops", "scipy"],
        check=True,
    )
    if not REPO_DIR.exists():
        log("Cloning D-FINE repository...")
        subprocess.run(["git", "clone", "https://github.com/Peterande/D-FINE.git", str(REPO_DIR)], check=True)


def find_dataset_root() -> Path:
    candidates = list(INPUT.glob("**/data.yaml"))
    if not candidates:
        raise FileNotFoundError("Could not find data.yaml in /kaggle/input")
    return candidates[0].parent


def find_weights() -> Path:
    candidates = list(INPUT.glob("**/*dfine*.pth")) + list(INPUT.glob("**/best*.pth"))
    if not candidates:
        raise FileNotFoundError("Could not find D-FINE weights in /kaggle/input")
    log(f"Found weights candidates: {candidates}")
    return candidates[0]


def parse_label_file(label_path: Path) -> list[list[float]]:
    if not label_path.exists():
        return []
    boxes = []
    for line in label_path.read_text(encoding="utf-8").strip().splitlines():
        parts = line.strip().split()
        if len(parts) >= 5:
            boxes.append([float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])])
    return boxes


def write_label_file(path: Path, boxes: list[list[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{int(b[0])} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}" for b in boxes]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def crop_boxes_to_roi(boxes: list[list[float]], roi_xyxyn: tuple[float, float, float, float], min_area_keep_frac: float = 0.10) -> list[list[float]]:
    rx1, ry1, rx2, ry2 = roi_xyxyn
    rw, rh = rx2 - rx1, ry2 - ry1
    if rw <= 0 or rh <= 0:
        return []
    kept = []
    for box in boxes:
        cls_id, cx, cy, bw, bh = box
        bx1, by1 = cx - bw / 2, cy - bh / 2
        bx2, by2 = cx + bw / 2, cy + bh / 2
        orig_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        if orig_area <= 0:
            continue
        ix1, iy1 = max(bx1, rx1), max(by1, ry1)
        ix2, iy2 = min(bx2, rx2), min(by2, ry2)
        inter_w, inter_h = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter_area = inter_w * inter_h
        if inter_area <= 0 or (inter_area / orig_area) < min_area_keep_frac:
            continue
        nx = (ix1 - rx1) / rw
        ny = (iy1 - ry1) / rh
        nw = inter_w / rw
        nh = inter_h / rh
        ncx = nx + nw / 2
        ncy = ny + nh / 2
        kept.append([cls_id, max(0.0, min(1.0, ncx)), max(0.0, min(1.0, ncy)), max(0.0, min(1.0, nw)), max(0.0, min(1.0, nh))])
    return kept


def apply_roi_crop_to_split(src_images_dir: Path, src_labels_dir: Path, dst_images_dir: Path, dst_labels_dir: Path) -> None:
    import cv2
    dst_images_dir.mkdir(parents=True, exist_ok=True)
    dst_labels_dir.mkdir(parents=True, exist_ok=True)
    img_files = sorted([f for f in src_images_dir.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}])
    log(f"Applying ROI crop to {len(img_files)} images...")
    for img_path in img_files:
        label_path = src_labels_dir / f"{img_path.stem}.txt"
        boxes = parse_label_file(label_path)
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h_px, w_px = img.shape[:2]
        key = f"{w_px}x{h_px}"
        dst_img_path = dst_images_dir / img_path.name
        dst_lbl_path = dst_labels_dir / f"{img_path.stem}.txt"
        if key not in ROI_CONFIG or ROI_CONFIG[key] == (0.0, 0.0, 1.0, 1.0):
            shutil.copy2(img_path, dst_img_path)
            if boxes:
                write_label_file(dst_lbl_path, boxes)
            continue
        roi_xyxyn = ROI_CONFIG[key]
        x1n, y1n, x2n, y2n = roi_xyxyn
        x1, y1 = int(round(x1n * w_px)), int(round(y1n * h_px))
        x2, y2 = int(round(x2n * w_px)), int(round(y2n * h_px))
        x2, y2 = max(x2, x1 + 1), max(y2, y1 + 1)
        cv2.imwrite(str(dst_img_path), img[y1:y2, x1:x2])
        new_boxes = crop_boxes_to_roi(boxes, roi_xyxyn, min_area_keep_frac=0.10)
        if new_boxes:
            write_label_file(dst_lbl_path, new_boxes)


def convert_yolo_to_coco(images_dir: Path, labels_dir: Path, output_json: Path, class_names: list[str]) -> Path:
    from PIL import Image
    images_list, annotations = [], []
    ann_id = 1
    image_files = sorted([f for f in images_dir.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}])
    for img_id, img_path in enumerate(image_files, start=1):
        with Image.open(img_path) as img:
            w, h = img.size
        images_list.append({"id": img_id, "file_name": img_path.name, "width": w, "height": h})
        label_path = labels_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            continue
        for line in label_path.read_text(encoding="utf-8").strip().splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            x, y = (cx - bw / 2) * w, (cy - bh / 2) * h
            abs_w, abs_h = bw * w, bh * h
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
    coco = {"images": images_list, "annotations": annotations, "categories": categories}
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(coco, indent=2), encoding="utf-8")
    log(f"COCO JSON written: {output_json} ({len(images_list)} images, {len(annotations)} annotations)")
    return output_json


def prepare_test_dataset(dataset_root: Path) -> tuple[Path, Path, Path]:
    import numpy as np
    valid_img_dir = dataset_root / "valid" / "images"
    valid_lbl_dir = dataset_root / "valid" / "labels"
    image_files = sorted([f for f in valid_img_dir.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}])

    image_labels = {}
    for img_path in image_files:
        lbl_path = valid_lbl_dir / f"{img_path.stem}.txt"
        boxes = parse_label_file(lbl_path)
        classes = {int(b[0]) for b in boxes}
        image_labels[img_path.stem] = classes

    Y = np.zeros((len(image_files), len(CLASSES)), dtype=int)
    for i, img_path in enumerate(image_files):
        for c in image_labels.get(img_path.stem, set()):
            if c < len(CLASSES):
                Y[i, c] = 1

    try:
        from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
        mskf = MultilabelStratifiedKFold(n_splits=2, shuffle=True, random_state=42)
        _, test_idx = next(mskf.split(image_files, Y))
    except Exception:
        rng = np.random.RandomState(42)
        indices = rng.permutation(len(image_files))
        test_idx = indices[len(image_files) // 2:]

    test_imgs = {image_files[i].name for i in test_idx}
    raw_test_dir = WORKING / "raw_test" / "images"
    raw_test_lbl = WORKING / "raw_test" / "labels"
    raw_test_dir.mkdir(parents=True, exist_ok=True)
    raw_test_lbl.mkdir(parents=True, exist_ok=True)

    for img_name in test_imgs:
        shutil.copy2(valid_img_dir / img_name, raw_test_dir / img_name)
        lbl_file = valid_lbl_dir / f"{Path(img_name).stem}.txt"
        if lbl_file.exists():
            shutil.copy2(lbl_file, raw_test_lbl / lbl_file.name)

    log(f"Test split isolated: {len(test_imgs)} images in {raw_test_dir}")
    roi_test_dir = WORKING / "data_roi" / "test"
    roi_test_lbl = WORKING / "data_roi" / "test_labels"
    apply_roi_crop_to_split(raw_test_dir, raw_test_lbl, roi_test_dir, roi_test_lbl)

    coco_json = convert_yolo_to_coco(roi_test_dir, roi_test_lbl, roi_test_dir / "test.json", CLASSES)
    return raw_test_dir, roi_test_dir, coco_json


def run_inference(model, images_info, roi_img_dir: Path, device: str):
    import torch
    import torchvision.transforms as T
    from PIL import Image

    tf = T.Compose([T.Resize((960, 960)), T.ToTensor()])
    preds = {}
    t0 = time.time()
    n_total = len(images_info)

    for i in range(0, n_total, BATCH):
        batch = images_info[i:i + BATCH]
        tensors, sizes = [], []
        for info in batch:
            im = Image.open(roi_img_dir / info["file_name"]).convert("RGB")
            sizes.append([im.size[0], im.size[1]])
            tensors.append(tf(im))
        samples = torch.stack(tensors, dim=0).to(device)
        orig_sizes = torch.tensor(sizes, dtype=torch.int64).to(device)
        with torch.no_grad():
            labels, boxes, scores = model(samples, orig_sizes)
        for j, info in enumerate(batch):
            preds[info["id"]] = (
                boxes[j].cpu().numpy(),
                scores[j].cpu().numpy(),
                labels[j].cpu().numpy(),
            )
        done = i + len(batch)
        if done % 100 == 0 or done == n_total:
            el = time.time() - t0
            log(f"Inference progress: {done}/{n_total} images ({el:.1f}s, {el/done*1000:.1f}ms/img)")
    return preds


def compute_coco_eval(preds, gt_json_path: Path):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    coco = COCO(str(gt_json_path))
    coco_dt = []
    for img_id, (boxes, scores, labels) in preds.items():
        for k in range(len(scores)):
            x0, y0, x1, y1 = [float(v) for v in boxes[k]]
            coco_dt.append({
                "image_id": int(img_id),
                "category_id": int(labels[k]),
                "bbox": [x0, y0, max(0.0, x1 - x0), max(0.0, y1 - y0)],
                "score": float(scores[k]),
            })
    coco_dt = coco.loadRes(coco_dt)
    ev = COCOeval(coco, coco_dt, "bbox")
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    return {
        "mAP50_95": float(ev.stats[0]),
        "mAP50": float(ev.stats[1]),
        "mAP75": float(ev.stats[2]),
        "mAP_small": float(ev.stats[3]),
        "mAP_medium": float(ev.stats[4]),
        "mAP_large": float(ev.stats[5]),
    }


def iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    sa = (a[2] - a[0]) * (a[3] - a[1])
    sb = (b[2] - b[0]) * (b[3] - b[1])
    ua = sa + sb - inter
    return inter / ua if ua > 0 else 0.0


def compute_confusion_matrix(preds, gt_by_image):
    import numpy as np
    n = len(CLASSES)
    mat = np.zeros((n + 1, n + 1), dtype=int)
    stats = {c: {"tp": 0, "fp": 0, "fn": 0} for c in range(n)}

    for img_id, gts in gt_by_image.items():
        boxes, scores, labels = preds[img_id]
        keep = np.argsort(-scores)
        matched = set()
        for k in keep:
            if scores[k] < CONF_TH:
                break
            best, best_iou = -1, IOU_TH
            for gi, g in enumerate(gts):
                if gi in matched:
                    continue
                v = iou(boxes[k], [g["bbox"][0], g["bbox"][1], g["bbox"][0] + g["bbox"][2], g["bbox"][1] + g["bbox"][3]])
                if v >= best_iou:
                    best, best_iou = gi, v
            if best >= 0:
                matched.add(best)
                gt_c = gts[best]["category_id"]
                pr_c = int(labels[k])
                mat[gt_c, pr_c] += 1
                if gt_c == pr_c:
                    stats[pr_c]["tp"] += 1
                else:
                    stats[gt_c]["fn"] += 1
                    stats[pr_c]["fp"] += 1
            else:
                pr_c = int(labels[k])
                mat[n, pr_c] += 1
                stats[pr_c]["fp"] += 1
        for gi, g in enumerate(gts):
            if gi not in matched:
                mat[g["category_id"], n] += 1
                stats[g["category_id"]]["fn"] += 1
    return mat, stats


def plot_confusion(mat, out_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = CLASSES + ["background"]
    fig, ax = plt.subplots(figsize=(8.5, 7.5), dpi=150)
    im = ax.imshow(mat, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(len(labels)), labels, fontsize=9)
    ax.set_xlabel("Predicted Class", fontsize=11, fontweight="bold")
    ax.set_ylabel("Ground Truth Class", fontsize=11, fontweight="bold")
    ax.set_title("D-FINE-M · Confusion Matrix (Held-out Test 890 Images)\nIoU ≥ 0.50, Conf ≥ 0.50 (Greedy Matching)", fontsize=11, fontweight="bold")
    mx = max(mat.max(), 1)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            ax.text(j, i, str(v), ha="center", va="center", fontsize=8, color="white" if v > 0.55 * mx else "#12314f")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(str(out_path), bbox_inches="tight")
    plt.close(fig)
    log(f"Saved confusion matrix plot: {out_path}")


def draw_visual_samples(preds, gt_by_image, gt_json_path: Path, raw_test_dir: Path, out_dir: Path):
    from PIL import Image, ImageDraw, ImageFont
    import numpy as np

    def score(item):
        return len(item[1])
    chosen = [i for i, _ in sorted(gt_by_image.items(), key=score, reverse=True)[:6]]
    info_by_id = {im["id"]: im for im in json.load(open(gt_json_path, encoding="utf-8"))["images"]}

    for idx, img_id in enumerate(chosen):
        info = info_by_id[img_id]
        raw_path = raw_test_dir / info["file_name"]
        if not raw_path.exists():
            continue
        im = Image.open(raw_path).convert("RGB")
        W, H = im.size
        key = f"{W}x{H}"
        rx1, ry1 = ROI_CONFIG.get(key, (0, 0, 1, 1))[:2]
        ox, oy = rx1 * W, ry1 * H
        draw = ImageDraw.Draw(im)
        try:
            font = ImageFont.truetype("arial.ttf", max(18, W // 220))
        except Exception:
            font = ImageFont.load_default()

        # GT: Blue boxes
        for g in gt_by_image[img_id]:
            gx, gy, gw, gh = g["bbox"]
            draw.rectangle([gx + ox, gy + oy, gx + gw + ox, gy + gh + oy], outline="#1d9bf0", width=4)

        # Preds: Class-colored boxes
        boxes, scores, labels = preds[img_id]
        for k in np.argsort(-scores):
            if scores[k] < CONF_TH:
                break
            c = int(labels[k])
            x0, y0, x1, y1 = boxes[k]
            draw.rectangle([x0 + ox, y0 + oy, x1 + ox, y1 + oy], outline=COLORS[c], width=5)
            draw.text((x0 + ox, y0 + oy - 24), f"{CLASSES[c]} {scores[k]:.2f}", fill=COLORS[c], font=font)

        out_file = out_dir / f"dfine_m_pred_{idx + 1}.jpg"
        im.save(out_file, quality=90)
        log(f"Saved visual prediction sample: {out_file}")


def main() -> None:
    log("=== Starting D-FINE-M Kaggle Evaluation ===")
    setup_environment()
    sys.path.insert(0, str(REPO_DIR))

    import torch
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    log(f"Inference device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    dataset_root = find_dataset_root()
    weights_path = find_weights()
    log(f"Using dataset from: {dataset_root}")
    log(f"Using weights from: {weights_path}")

    raw_test_dir, roi_test_dir, gt_json_path = prepare_test_dataset(dataset_root)

    # Build custom config for glass defect (5 classes)
    dataset_yml_path = REPO_DIR / "configs" / "dataset" / "glass_defect_test.yml"
    dataset_yml_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_yml_path.write_text("""task: detection
evaluator:
  type: CocoEvaluator
  iou_types: ['bbox', ]
num_classes: 5
remap_mscoco_category: False
train_dataloader:
  type: DataLoader
  dataset:
    type: CocoDetection
    img_folder: /kaggle/working/data_roi/train
    ann_file: /kaggle/working/data_roi/train/train.json
    return_masks: False
    transforms:
      type: Compose
      ops: ~
  shuffle: True
  num_workers: 4
  drop_last: True
  collate_fn:
    type: BatchImageCollateFunction
val_dataloader:
  type: DataLoader
  dataset:
    type: CocoDetection
    img_folder: /kaggle/working/data_roi/test
    ann_file: /kaggle/working/data_roi/test/test.json
    return_masks: False
    transforms:
      type: Compose
      ops: ~
  shuffle: False
  num_workers: 4
  drop_last: False
  collate_fn:
    type: BatchImageCollateFunction
""", encoding="utf-8")

    custom_cfg_path = REPO_DIR / "configs" / "dfine" / "custom" / "dfine_hgnetv2_m_glass_test.yml"
    custom_cfg_path.parent.mkdir(parents=True, exist_ok=True)
    custom_cfg_path.write_text("""__include__: [
  '../../dataset/glass_defect_test.yml',
  '../../runtime.yml',
  '../include/dataloader.yml',
  '../include/optimizer.yml',
  '../include/dfine_hgnetv2.yml',
]
output_dir: /kaggle/working/output_dfine_m
eval_spatial_size: [960, 960]
DFINE:
  backbone: HGNetv2
HGNetv2:
  name: 'B2'
  return_idx: [1, 2, 3]
  freeze_at: -1
  freeze_norm: False
  use_lab: True
  pretrained: False
DFINETransformer:
  num_layers: 4
  eval_idx: -1
HybridEncoder:
  in_channels: [384, 768, 1536]
  hidden_dim: 256
  depth_mult: 0.67
val_dataloader:
  total_batch_size: 16
  num_workers: 4
  dataset:
    transforms:
      ops:
        - {type: Resize, size: [960, 960]}
        - {type: ConvertPILImage, dtype: 'float32', scale: True}
  collate_fn:
    type: BatchImageCollateFunction
""", encoding="utf-8")

    # Build model using YAMLConfig
    from src.core import YAMLConfig
    cfg_path = custom_cfg_path
    log(f"Loading YAML config: {cfg_path}")
    cfg = YAMLConfig(str(cfg_path))
    cfg.yaml_cfg["eval_spatial_size"] = [960, 960]

    ckpt = torch.load(str(weights_path), map_location=device)
    if "ema" in ckpt:
        state = ckpt["ema"]["module"]
        log(f"Loaded EMA weights (epoch {ckpt.get('last_epoch', 'unknown')})")
    elif "model" in ckpt:
        state = ckpt["model"]
        log("Loaded model weights")
    else:
        state = ckpt
        log("Loaded direct weights")

    cfg.model.load_state_dict(state)

    class DeployedModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            return self.postprocessor(outputs, orig_target_sizes)

    model = DeployedModel().to(device)
    model.eval()

    gt_json = json.load(open(gt_json_path, encoding="utf-8"))
    images_info = sorted(gt_json["images"], key=lambda x: x["id"])
    gt_by_image = {}
    for a in gt_json["annotations"]:
        gt_by_image.setdefault(a["image_id"], []).append(a)
    gt_by_image = {i: gt_by_image.get(i, []) for i in (im["id"] for im in images_info)}

    log(f"Running inference on {len(images_info)} images...")
    preds = run_inference(model, images_info, roi_test_dir, device)

    log("Computing COCO mAP metrics...")
    coco_metrics = compute_coco_eval(preds, gt_json_path)
    log(f"COCO Metrics: {coco_metrics}")

    log("Computing Confusion Matrix...")
    mat, stats = compute_confusion_matrix(preds, gt_by_image)
    plot_confusion(mat, WORKING / "dfine_m_cm.png")

    per_class = {}
    print("\n" + "=" * 60)
    print("DETAILED PER-CLASS METRICS (IoU >= 0.50, Conf >= 0.50):")
    print("=" * 60)
    for c, name in enumerate(CLASSES):
        tp, fp, fn = stats[c]["tp"], stats[c]["fp"], stats[c]["fn"]
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        per_class[name] = {
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f1, 4),
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }
        print(f"  {name:15s} | Precision: {p:6.2%} | Recall: {r:6.2%} | F1: {f1:6.2%} | TP: {tp:3d} | FP: {fp:3d} | FN: {fn:3d}")
    print("=" * 60 + "\n")

    # Save artifacts
    results = {
        "model": "D-FINE-M (Reference 44.4%)",
        "weights": weights_path.name,
        "coco_metrics": coco_metrics,
        "per_class": per_class,
        "confusion_matrix": mat.tolist(),
        "class_order": CLASSES + ["background"],
    }
    (WORKING / "dfine_m_confusion.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (WORKING / "metrics_dfine_m.json").write_text(json.dumps(coco_metrics, indent=2), encoding="utf-8")

    draw_visual_samples(preds, gt_by_image, gt_json_path, raw_test_dir, WORKING)
    log("=== Evaluation complete! All artifacts saved to /kaggle/working/ ===")


if __name__ == "__main__":
    main()
