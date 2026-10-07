"""Self-contained Kaggle Runner for RF-DETR-M Baseline (E0).

Experiment E0: RF-DETR-M Baseline (Seed 42, AHRD = OFF)
Hypothesis:
    Serves as the pure canonical control baseline to evaluate the small-defect
    bottleneck and measure the exact impact of AHRD.
    Uses pure HGNetv2-B2 (return_idx=[1, 2, 3]) + HybridEncoder + DFINETransformer.

Single independent variable: AHRD = OFF.
Seed: 42.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

# Kaggle environment paths
WORKING = Path("/kaggle/working")
DATA_ROI = WORKING / "data_roi"
OUTPUT_DIR = WORKING / "output_rfdetr_m_baseline_seed42"
DFINE_REPO = WORKING / "D-FINE"
DFINE_WEIGHTS = WORKING / "dfine_m_coco.pth"

# PyTorch CUDA memory management
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

SEED = 42
TOTAL_BATCH_SIZE = 8
MICRO_BATCH_SIZE = 4
ACCUMULATE_STEPS = TOTAL_BATCH_SIZE // MICRO_BATCH_SIZE
os.environ["DFINE_ACCUMULATE_STEPS"] = str(ACCUMULATE_STEPS)

ROI_CONFIG = {
    "5328x3000": (0.14113178913288288, 0.0, 0.8881345303115614, 1.0),
    "1024x1024": (0.0, 0.0, 1.0, 1.0),
}

CLASS_NAMES = ["DiVat", "DiVatLoiLom", "LoiChi", "LoiNhua", "LoiTray"]


def log(msg: str) -> None:
    print(f"[rfdetr-baseline-seed{SEED}] {msg}", flush=True)


def patch_dist_utils_for_nccl_cpu() -> None:
    dist_file = DFINE_REPO / "src" / "misc" / "dist_utils.py"
    if not dist_file.exists():
        return
    text = dist_file.read_text(encoding="utf-8")
    if "_safe_all_gather" in text:
        return
    patch_code = '''
# Hotfix: prevent RuntimeError: No backend type associated with device type cpu under NCCL
import torch.distributed as _dist
_orig_all_gather = _dist.all_gather
def _safe_all_gather(tensor_list, tensor, *args, **kwargs):
    try:
        return _orig_all_gather(tensor_list, tensor, *args, **kwargs)
    except RuntimeError as e:
        if "device type cpu" in str(e) or "No backend type" in str(e):
            dev = torch.cuda.current_device()
            cuda_tensor = tensor.to(dev)
            cuda_list = [t.to(dev) for t in tensor_list]
            work = _orig_all_gather(cuda_list, cuda_tensor, *args, **kwargs)
            for cpu_t, cuda_t in zip(tensor_list, cuda_list):
                cpu_t.copy_(cuda_t.cpu())
            return work
        raise
_dist.all_gather = _safe_all_gather
'''
    text = patch_code + "\n" + text
    text = text.replace("find_unused_parameters: bool = False", "find_unused_parameters: bool = True")
    text = text.replace("find_unused_parameters=find_unused_parameters", "find_unused_parameters=True")
    dist_file.write_text(text, encoding="utf-8")
    log("Applied NCCL CPU all_gather try-except and find_unused_parameters=True hotfix")

    runtime_yml = DFINE_REPO / "configs" / "runtime.yml"
    if runtime_yml.exists():
        rtext = runtime_yml.read_text(encoding="utf-8")
        rtext = rtext.replace("find_unused_parameters: False", "find_unused_parameters: True")
        runtime_yml.write_text(rtext, encoding="utf-8")


def patch_gradient_accumulation(source: str) -> str:
    replacements = {
        "import math\nimport sys": "import math\nimport os\nimport sys",
        """    for i, (samples, targets) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        global_step = epoch * len(data_loader) + i
        metas = dict(epoch=epoch, step=i, global_step=global_step, epoch_step=len(data_loader))
""": """    accumulate_steps = int(os.environ.get(\"DFINE_ACCUMULATE_STEPS\", \"1\"))
    optimizer_steps = math.ceil(len(data_loader) / accumulate_steps)
    optimizer.zero_grad()
    for i, (samples, targets) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        optimizer_step = i // accumulate_steps
        global_step = epoch * optimizer_steps + optimizer_step
        metas = dict(
            epoch=epoch,
            step=optimizer_step,
            global_step=global_step,
            epoch_step=optimizer_steps,
        )
        should_step = (i + 1) % accumulate_steps == 0 or i + 1 == len(data_loader)
""",
        """            scaler.scale(loss).backward()

            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
""": """            loss_to_backprop = loss / accumulate_steps
            scaler.scale(loss_to_backprop).backward()

            if should_step:
                if max_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
""",
        """            loss.backward()

            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            optimizer.step()
            optimizer.zero_grad()
""": """            loss_to_backprop = loss / accumulate_steps
            loss_to_backprop.backward()

            if should_step:
                if max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                optimizer.step()
                optimizer.zero_grad()
""",
        """        if lr_warmup_scheduler:
            lr_warmup_scheduler.step()
""": """        if lr_warmup_scheduler and should_step:
            lr_warmup_scheduler.step()
""",
        """        if ema:
            ema.update(model)
""": """        if ema and should_step:
            ema.update(model)
""",
    }
    for old, new in replacements.items():
        if old in source:
            source = source.replace(old, new, 1)
    return source


def patch_dfine_engine() -> None:
    engine_file = DFINE_REPO / "src" / "solver" / "det_engine.py"
    if engine_file.exists():
        content = engine_file.read_text(encoding="utf-8")
        patched = patch_gradient_accumulation(content)
        engine_file.write_text(patched, encoding="utf-8")
        log(f"Patched gradient accumulation: micro={MICRO_BATCH_SIZE}, accumulate={ACCUMULATE_STEPS}")


def install_dependencies() -> None:
    log("Installing dependencies...")
    subprocess.run(
        [
            sys.executable, "-m", "pip", "install",
            "iterative-stratification==0.1.9",
            "PyYAML>=6.0.2",
            "Pillow>=10.0",
            "pycocotools>=2.0.7",
            "tensorboard>=2.15",
            "faster-coco-eval>=1.3.3",
        ],
        check=True,
    )
    if not DFINE_REPO.exists():
        log("Cloning D-FINE repository...")
        subprocess.run(
            ["git", "clone", "--depth", "1", "https://github.com/Peterande/D-FINE.git", str(DFINE_REPO)],
            check=True,
        )
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", str(DFINE_REPO / "requirements.txt")],
            check=True,
        )


def download_pretrained_weights() -> Path:
    if DFINE_WEIGHTS.exists() and DFINE_WEIGHTS.stat().st_size > 70 * 1024 * 1024:
        log(f"Pretrained weights exist at {DFINE_WEIGHTS}")
        return DFINE_WEIGHTS

    url = "https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_m_coco.pth"
    log(f"Downloading pretrained weights from {url}...")
    urllib.request.urlretrieve(url, str(DFINE_WEIGHTS))
    log(f"Downloaded weights successfully: {DFINE_WEIGHTS.stat().st_size / (1024*1024):.1f} MB")
    return DFINE_WEIGHTS


def find_dataset_root() -> Path:
    candidates = [
        p for p in sorted(Path("/kaggle/input").rglob("data.yaml"))
        if "dataset_processed" not in p.parts and "dataset_roi" not in p.parts
    ]
    if not candidates:
        raise FileNotFoundError("Raw dataset root containing data.yaml was not found in /kaggle/input")
    log(f"Found dataset root at {candidates[0].parent}")
    return candidates[0].parent


def parse_label_file(p: Path) -> list[list[float]]:
    boxes = []
    if not p.exists():
        return boxes
    for line in p.read_text(encoding="utf-8").strip().splitlines():
        parts = line.strip().split()
        if len(parts) >= 5:
            boxes.append([float(x) for x in parts[:5]])
    return boxes


def apply_roi_crop_to_split(src_img_dir: Path, src_lbl_dir: Path, out_img_dir: Path, out_lbl_dir: Path) -> None:
    from PIL import Image

    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    img_files = sorted([f for f in src_img_dir.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}])
    for img_p in img_files:
        lbl_p = src_lbl_dir / f"{img_p.stem}.txt"
        with Image.open(img_p) as im:
            w, h = im.size
            dim_key = f"{w}x{h}"
            roi = ROI_CONFIG.get(dim_key, (0.0, 0.0, 1.0, 1.0))
            rx0, ry0, rx1, ry1 = roi
            x0_px = int(round(rx0 * w))
            y0_px = int(round(ry0 * h))
            x1_px = int(round(rx1 * w))
            y1_px = int(round(ry1 * h))
            crop_im = im.crop((x0_px, y0_px, x1_px, y1_px))
            crop_im.save(out_img_dir / img_p.name, quality=95)

        rw = (rx1 - rx0) * w
        rh = (ry1 - ry0) * h

        out_lines = []
        if lbl_p.exists():
            for line in lbl_p.read_text(encoding="utf-8").strip().splitlines():
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls_id = int(parts[0])
                cx = float(parts[1]) * w
                cy = float(parts[2]) * h
                bw = float(parts[3]) * w
                bh = float(parts[4]) * h

                new_cx = (cx - x0_px) / rw
                new_cy = (cy - y0_px) / rh
                new_bw = bw / rw
                new_bh = bh / rh

                if 0.0 < new_cx < 1.0 and 0.0 < new_cy < 1.0:
                    out_lines.append(f"{cls_id} {new_cx:.6f} {new_cy:.6f} {new_bw:.6f} {new_bh:.6f}")

        (out_lbl_dir / f"{img_p.stem}.txt").write_text("\n".join(out_lines), encoding="utf-8")


def convert_yolo_to_coco(img_dir: Path, lbl_dir: Path, out_json: Path, class_names: list[str]) -> Path:
    from PIL import Image

    images, annotations = [], []
    ann_id = 1
    img_files = sorted([f for f in img_dir.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}])

    for img_id, img_p in enumerate(img_files, start=1):
        with Image.open(img_p) as im:
            w, h = im.size
        images.append({"id": img_id, "file_name": img_p.name, "width": w, "height": h})

        lbl_p = lbl_dir / f"{img_p.stem}.txt"
        if not lbl_p.exists():
            continue

        for line in lbl_p.read_text(encoding="utf-8").strip().splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            abs_x = (cx - bw / 2) * w
            abs_y = (cy - bh / 2) * h
            abs_w = bw * w
            abs_h = bh * h
            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": cls_id,
                "bbox": [round(abs_x, 2), round(abs_y, 2), round(abs_w, 2), round(abs_h, 2)],
                "area": round(abs_w * abs_h, 2),
                "iscrowd": 0,
            })
            ann_id += 1

    categories = [{"id": i, "name": name} for i, name in enumerate(class_names)]
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps({"images": images, "annotations": annotations, "categories": categories}, indent=2), encoding="utf-8")
    return out_json


def prepare_dataset(dataset_root: Path) -> dict[str, Path]:
    import numpy as np
    from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

    raw_train_imgs = dataset_root / "train" / "images"
    raw_train_lbls = dataset_root / "train" / "labels"
    raw_val_imgs = dataset_root / "valid" / "images" if (dataset_root / "valid").exists() else dataset_root / "val" / "images"
    raw_val_lbls = dataset_root / "valid" / "labels" if (dataset_root / "valid").exists() else dataset_root / "val" / "labels"

    split_cache = WORKING / "dataset_split"
    split_val_imgs = split_cache / "val" / "images"
    split_val_lbls = split_cache / "val" / "labels"
    split_test_imgs = split_cache / "test" / "images"
    split_test_lbls = split_cache / "test" / "labels"
    for d in [split_val_imgs, split_val_lbls, split_test_imgs, split_test_lbls]:
        d.mkdir(parents=True, exist_ok=True)

    val_images = sorted([f for f in raw_val_imgs.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}])
    nc = len(CLASS_NAMES)
    y = np.zeros((len(val_images), nc), dtype=np.int8)
    for i, img_p in enumerate(val_images):
        lbl_p = raw_val_lbls / f"{img_p.stem}.txt"
        for box in parse_label_file(lbl_p):
            c = int(box[0])
            if 0 <= c < nc:
                y[i, c] = 1

    splitter = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
    val_idx, test_idx = next(splitter.split(np.zeros(len(val_images)), y))
    log(f"Stratified split: {len(val_idx)} val images, {len(test_idx)} test images")

    for idx in val_idx:
        shutil.copy2(val_images[int(idx)], split_val_imgs / val_images[int(idx)].name)
        lbl_p = raw_val_lbls / f"{val_images[int(idx)].stem}.txt"
        if lbl_p.exists():
            shutil.copy2(lbl_p, split_val_lbls / lbl_p.name)

    for idx in test_idx:
        shutil.copy2(val_images[int(idx)], split_test_imgs / val_images[int(idx)].name)
        lbl_p = raw_val_lbls / f"{val_images[int(idx)].stem}.txt"
        if lbl_p.exists():
            shutil.copy2(lbl_p, split_test_lbls / lbl_p.name)

    roi_train = DATA_ROI / "train"
    roi_val = DATA_ROI / "val"
    roi_test = DATA_ROI / "test"

    apply_roi_crop_to_split(raw_train_imgs, raw_train_lbls, roi_train, roi_train / "labels")
    apply_roi_crop_to_split(split_val_imgs, split_val_lbls, roi_val, roi_val / "labels")
    apply_roi_crop_to_split(split_test_imgs, split_test_lbls, roi_test, roi_test / "labels")

    train_json = convert_yolo_to_coco(roi_train, roi_train / "labels", roi_train / "train.json", CLASS_NAMES)
    val_json = convert_yolo_to_coco(roi_val, roi_val / "labels", roi_val / "val.json", CLASS_NAMES)
    test_json = convert_yolo_to_coco(roi_test, roi_test / "labels", roi_test / "test.json", CLASS_NAMES)

    return {
        "train_img": roi_train,
        "train_json": train_json,
        "val_img": roi_val,
        "val_json": val_json,
        "test_img": roi_test,
        "test_json": test_json,
    }


def write_baseline_config(paths: dict[str, Path]) -> Path:
    import yaml

    nc = len(CLASS_NAMES)
    ds_train_cfg = {
        "task": "detection",
        "evaluator": {"type": "CocoEvaluator", "iou_types": ["bbox"]},
        "num_classes": nc,
        "remap_mscoco_category": False,
        "train_dataloader": {
            "type": "DataLoader",
            "dataset": {
                "type": "CocoDetection",
                "img_folder": str(paths["train_img"]),
                "ann_file": str(paths["train_json"]),
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
                "img_folder": str(paths["val_img"]),
                "ann_file": str(paths["val_json"]),
                "return_masks": False,
                "transforms": {"type": "Compose", "ops": None},
            },
            "shuffle": False,
            "num_workers": 4,
            "drop_last": False,
            "collate_fn": {"type": "BatchImageCollateFunction"},
        },
    }
    ds_path = DFINE_REPO / "configs" / "dataset" / "glass_defect_baseline.yml"
    ds_path.write_text(yaml.safe_dump(ds_train_cfg, sort_keys=False), encoding="utf-8")

    model_cfg = f"""# Experiment E0: RF-DETR-M Baseline (Seed {SEED}, AHRD = OFF)
__include__: [
  '../../dataset/glass_defect_baseline.yml',
  '../../runtime.yml',
  '../include/dataloader.yml',
  '../include/optimizer.yml',
  '../include/dfine_hgnetv2.yml',
]

output_dir: {OUTPUT_DIR}
eval_spatial_size: [960, 960]

model: DFINE

DFINE:
  backbone: HGNetv2
  encoder: HybridEncoder
  decoder: DFINETransformer

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

optimizer:
  type: AdamW
  params:
    -
      params: '^(?=.*backbone)(?!.*norm|bn).*$'
      lr: 0.000025
    -
      params: '^(?=.*backbone)(?=.*norm|bn).*$'
      lr: 0.000025
      weight_decay: 0.
    -
      params: '^(?=.*(?:encoder|decoder))(?=.*(?:norm|bn|bias)).*$'
      weight_decay: 0.

  lr: 0.00025
  betas: [0.9, 0.999]
  weight_decay: 0.000125

epochs: 100

train_dataloader:
  total_batch_size: 8
  num_workers: 4
  dataset:
    transforms:
      ops:
        - {{type: RandomPhotometricDistort, p: 0.5}}
        - {{type: RandomZoomOut, fill: 0}}
        - {{type: RandomIoUCrop, p: 0.8}}
        - {{type: SanitizeBoundingBoxes, min_size: 1}}
        - {{type: RandomHorizontalFlip}}
        - {{type: Resize, size: [960, 960]}}
        - {{type: SanitizeBoundingBoxes, min_size: 1}}
        - {{type: ConvertPILImage, dtype: 'float32', scale: True}}
        - {{type: ConvertBoxes, fmt: 'cxcywh', normalize: True}}
      policy:
        name: stop_epoch
        epoch: 88
        ops: ['RandomPhotometricDistort', 'RandomZoomOut', 'RandomIoUCrop']
  collate_fn:
    type: BatchImageCollateFunction
    base_size: 960
    base_size_repeat: 3
    stop_epoch: 88
    ema_restart_decay: 0.9999

val_dataloader:
  total_batch_size: 16
  num_workers: 4
  dataset:
    transforms:
      ops:
        - {{type: Resize, size: [960, 960]}}
        - {{type: ConvertPILImage, dtype: 'float32', scale: True}}
  collate_fn:
    type: BatchImageCollateFunction

ema:
  warmups: 0

lr_warmup_scheduler:
  warmup_duration: 0
"""
    model_cfg_path = DFINE_REPO / "configs" / "dfine" / "custom" / "rfdetr_m_baseline_seed42.yml"
    model_cfg_path.parent.mkdir(parents=True, exist_ok=True)
    model_cfg_path.write_text(model_cfg, encoding="utf-8")
    return model_cfg_path


def run_training(config_path: Path, weights_path: Path) -> Path:
    import torch

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    n_gpus = torch.cuda.device_count()
    log(f"Detected {n_gpus} GPUs. Launching training with seed {SEED}...")

    if n_gpus > 1:
        cmd = [
            "torchrun",
            f"--nproc_per_node={n_gpus}",
            str(DFINE_REPO / "train.py"),
            "-c", str(config_path),
            "-t", str(weights_path),
            "--seed", str(SEED),
            "--use-amp",
            "--output-dir", str(OUTPUT_DIR),
        ]
    else:
        cmd = [
            sys.executable,
            str(DFINE_REPO / "train.py"),
            "-c", str(config_path),
            "-t", str(weights_path),
            "--seed", str(SEED),
            "--use-amp",
            "--output-dir", str(OUTPUT_DIR),
        ]

    log(f"Executing: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    candidates = [
        OUTPUT_DIR / "best_stg2.pth",
        OUTPUT_DIR / "best_stg1.pth",
        OUTPUT_DIR / "best.pth",
    ]
    for c in candidates:
        if c.exists():
            log(f"Found best checkpoint at {c}")
            return c
    raise FileNotFoundError(f"No checkpoint found in {OUTPUT_DIR}")


def run_evaluation(best_ckpt: Path, config_path: Path, test_json: Path, test_img_dir: Path) -> dict[str, Any]:
    log("Running post-training evaluation on test split...")
    eval_script = DFINE_REPO / "eval_baseline.py"
    eval_code = f"""
import json, os, sys, time
from pathlib import Path
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

sys.path.insert(0, "{DFINE_REPO}")
from src.core import YAMLConfig
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

CLASS_NAMES = ["DiVat", "DiVatLoiLom", "LoiChi", "LoiNhua", "LoiTray"]
SCALE_BINS = {{
    "lt_8px": lambda w, h: min(w, h) < 8.0,
    "8_to_16px": lambda w, h: 8.0 <= min(w, h) < 16.0,
    "16_to_32px": lambda w, h: 16.0 <= min(w, h) < 32.0,
    "gte_32px": lambda w, h: min(w, h) >= 32.0,
}}

def compute_iou(b1, b2):
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    xa, ya = max(x1, x2), max(y1, y2)
    xb, yb = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
    inter = max(0.0, xb - xa) * max(0.0, yb - ya)
    union = w1 * h1 + w2 * h2 - inter
    return inter / union if union > 0 else 0.0

cfg = YAMLConfig("{config_path}")
ckpt = torch.load("{best_ckpt}", map_location="cpu")
state = ckpt.get("ema", {{}}).get("module", ckpt.get("model", ckpt))
cfg.model.load_state_dict(state, strict=False)

class FullModel(torch.nn.Module):
    def __init__(self, m, p):
        super().__init__()
        self.model = m.deploy()
        self.postprocessor = p.deploy()
    def forward(self, im, sz):
        return self.postprocessor(self.model(im), sz)

dev = "cuda:0" if torch.cuda.is_available() else "cpu"
model = FullModel(cfg.model, cfg.postprocessor).to(dev)
model.eval()

gt_data = json.loads(Path("{test_json}").read_text(encoding="utf-8"))
images = gt_data["images"]
tf = T.Compose([T.Resize((960, 960)), T.ToTensor()])
preds = []

with torch.no_grad():
    for i in range(0, len(images), 4):
        batch = images[i:i+4]
        tensors, sizes = [], []
        for info in batch:
            im = Image.open(Path("{test_img_dir}") / info["file_name"]).convert("RGB")
            tensors.append(tf(im))
            sizes.append([info["height"], info["width"]])
        outs = model(torch.stack(tensors).to(dev), torch.tensor(sizes, dtype=torch.float32).to(dev))
        for j, info in enumerate(batch):
            out = outs[j]
            b_np = out["boxes"].cpu().numpy()
            s_np = out["scores"].cpu().numpy()
            l_np = out["labels"].cpu().numpy()
            for b, s, l in zip(b_np, s_np, l_np):
                preds.append({{
                    "image_id": info["id"],
                    "category_id": int(l),
                    "bbox": [float(b[0]), float(b[1]), float(b[2]-b[0]), float(b[3]-b[1])],
                    "score": float(s),
                }})

coco_gt = COCO("{test_json}")
coco_dt = coco_gt.loadRes(preds)
ev = COCOeval(coco_gt, coco_dt, "bbox")
ev.evaluate()
ev.accumulate()
ev.summarize()
st = ev.stats

per_class = {{}}
for cid, cname in enumerate(CLASS_NAMES):
    ev.params.catIds = [cid]
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    per_class[cname] = {{
        "mAP50_95": round(float(ev.stats[0])*100, 2),
        "mAP50": round(float(ev.stats[1])*100, 2),
        "mAP75": round(float(ev.stats[2])*100, 2),
    }}

scale_stats = {{b: {{"gt": 0, "tp": 0, "fn": 0, "recall": 0.0}} for b in SCALE_BINS}}
valid_preds = [p for p in preds if p["score"] >= 0.5]
p_by_img = {{}}
for p in valid_preds:
    p_by_img.setdefault(p["image_id"], []).append(p)

for ann in gt_data["annotations"]:
    w, h = ann["bbox"][2], ann["bbox"][3]
    m_bin = next((b for b, fn in SCALE_BINS.items() if fn(w, h)), None)
    if not m_bin: continue
    scale_stats[m_bin]["gt"] += 1
    matched = False
    for p in p_by_img.get(ann["image_id"], []):
        if p["category_id"] == ann["category_id"] and compute_iou(ann["bbox"], p["bbox"]) >= 0.5:
            matched = True
            break
    if matched: scale_stats[m_bin]["tp"] += 1
    else: scale_stats[m_bin]["fn"] += 1

for b, d in scale_stats.items():
    d["recall"] = round((d["tp"] / d["gt"])*100, 2) if d["gt"] > 0 else 0.0

report = {{
    "mAP50_95": round(float(st[0])*100, 2),
    "mAP50": round(float(st[1])*100, 2),
    "mAP75": round(float(st[2])*100, 2),
    "AP_s": round(float(st[3])*100, 2),
    "AP_m": round(float(st[4])*100, 2),
    "AP_l": round(float(st[5])*100, 2),
    "per_class": per_class,
    "scale_breakdown": scale_stats,
}}
Path("{OUTPUT_DIR}/baseline_evaluation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print("EVALUATION RESULT:")
print(json.dumps(report, indent=2))
"""
    eval_script.write_text(eval_code, encoding="utf-8")
    subprocess.run([sys.executable, str(eval_script)], check=True)


def main() -> None:
    log("=" * 70)
    log(f"STARTING EXPERIMENT E0: RF-DETR-M BASELINE (SEED {SEED}, AHRD = OFF)")
    log("=" * 70)

    install_dependencies()
    patch_dist_utils_for_nccl_cpu()
    patch_dfine_engine()
    weights_path = download_pretrained_weights()

    dataset_root = find_dataset_root()
    paths = prepare_dataset(dataset_root)
    cfg_path = write_baseline_config(paths)

    best_ckpt = run_training(cfg_path, weights_path)
    run_evaluation(best_ckpt, cfg_path, paths["test_json"], paths["test_img"])
    log("=" * 70)
    log(f"EXPERIMENT E0 (SEED {SEED}) COMPLETED SUCCESSFULLY!")
    log("=" * 70)


if __name__ == "__main__":
    main()
