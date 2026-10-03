"""FBDNet-M Adaptation on D-FINE: Industrial Glass Bottle Defect Detection.

Architecture:
- Backbone: HGNetv2FBD with Adaptive Frequency-domain / Target Attention (AFTA) at C3/C4/C5.
- Encoder: FBDHybridEncoder with:
    1. Spatial-Frequency Interaction Fusion (SFIF) at projected features (hidden_dim=256).
    2. Dense Cross-scale Feature Screening / Fusion Blocks (DCFSB) at multi-scale FPN/PAN fusion.
- Decoder: DFINETransformer (frozen/clean baseline compatibility).
- Pretrained: Transferred non-strictly from D-FINE-M checkpoint with zero-initialized residual gates.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

WORKING = Path("/kaggle/working")
EXPERIMENTS_ROOT = WORKING / "glass-defect-experiments"
EXP_DIR = EXPERIMENTS_ROOT / "fbdnet_m"
ARTIFACTS = WORKING / "artifacts"
DFINE_REPO = WORKING / "D-FINE"

TOTAL_BATCH_SIZE = 8
MICRO_BATCH_SIZE = 4
ACCUMULATE_STEPS = TOTAL_BATCH_SIZE // MICRO_BATCH_SIZE
EPOCHS = 20

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ["DFINE_ACCUMULATE_STEPS"] = str(ACCUMULATE_STEPS)


def log(msg: str) -> None:
    print(f"[fbdnet-m] {msg}", flush=True)


def inspect_environment() -> dict:
    log("Inspecting Kaggle Environment...")
    import torch
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    vram_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2) if torch.cuda.is_available() else 0.0
    free_disk_gb = round(shutil.disk_usage(WORKING)[2] / (1024**3), 2)

    env_info = {
        "python": sys.version.split()[0],
        "pytorch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "gpu_name": gpu_name,
        "vram_gb": vram_gb,
        "free_disk_gb": free_disk_gb,
        "micro_batch_size": MICRO_BATCH_SIZE,
        "accumulate_steps": ACCUMULATE_STEPS,
        "effective_batch_size": TOTAL_BATCH_SIZE,
        "epochs": EPOCHS,
    }
    log("=" * 60)
    for k, v in env_info.items():
        log(f"  {k:22s}: {v}")
    log("=" * 60)
    return env_info


def install_dependencies() -> None:
    log("Installing dependencies and cloning official D-FINE...")
    subprocess.run(
        [
            sys.executable, "-m", "pip", "install",
            "iterative-stratification==0.1.9",
            "PyYAML>=6.0.2",
            "Pillow>=10.0",
            "pycocotools>=2.0.7",
            "tensorboard>=2.15",
        ],
        check=True,
    )
    if not DFINE_REPO.exists():
        subprocess.run(
            ["git", "clone", "--depth", "1", "https://github.com/Peterande/D-FINE.git", str(DFINE_REPO)],
            check=True,
        )
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", str(DFINE_REPO / "requirements.txt")],
        check=True,
    )


def patch_dfine_engine() -> None:
    """Enable gradient accumulation in D-FINE train loop to prevent T4 OOM."""
    engine_file = DFINE_REPO / "src" / "solver" / "det_engine.py"
    if not engine_file.exists():
        return
    text = engine_file.read_text(encoding="utf-8")
    if "DFINE_ACCUMULATE_STEPS" in text:
        return

    log("Patching D-FINE engine for gradient accumulation...")
    text = text.replace(
        "import math\nimport sys",
        "import math\nimport os\nimport sys"
    )
    old_loop = """    for i, (samples, targets) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        global_step = epoch * len(data_loader) + i
        metas = dict(epoch=epoch, step=i, global_step=global_step, epoch_step=len(data_loader))
"""
    new_loop = """    accumulate_steps = int(os.environ.get("DFINE_ACCUMULATE_STEPS", "1"))
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
"""
    text = text.replace(old_loop, new_loop)

    old_backward = """            scaler.scale(loss).backward()

            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
"""
    new_backward = """            loss_scaled = loss / accumulate_steps
            scaler.scale(loss_scaled).backward()

            if should_step:
                if max_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
"""
    text = text.replace(old_backward, new_backward)
    engine_file.write_text(text, encoding="utf-8")
    log("Gradient accumulation patch applied successfully.")


def setup_fbdnet_modules() -> None:
    """Inject AFTA, SFIF, DCFSB, HGNetv2FBD, and FBDHybridEncoder into D-FINE."""
    log("Injecting FBDNet modules into D-FINE codebase...")

    fbd_modules_dir = DFINE_REPO / "src" / "zoo" / "dfine" / "fbd_modules"
    fbd_modules_dir.mkdir(parents=True, exist_ok=True)

    # 1. afta.py
    afta_code = '''"""Adaptive Frequency-domain / Target Attention (AFTA) Module."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class AFTA(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.channels = channels
        mid_channels = max(channels // reduction, 32)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, mid_channels, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(mid_channels, channels, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )
        self.spatial_dw = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, groups=channels, bias=False
        )
        self.spatial_norm = nn.BatchNorm2d(channels)
        self.spatial_act = nn.GELU()
        self.spatial_pw = nn.Conv2d(channels, 1, kernel_size=1, bias=False)
        self.spatial_gate = nn.Sigmoid()
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c_att = self.channel_gate(x)
        x_c = x * c_att
        s_feat = self.spatial_act(self.spatial_norm(self.spatial_dw(x_c)))
        s_att = self.spatial_gate(self.spatial_pw(s_feat))
        x_s = x_c * s_att
        return x + self.gamma * x_s
'''
    (fbd_modules_dir / "afta.py").write_text(afta_code, encoding="utf-8")

    # 2. sfif.py
    sfif_code = '''"""Spatial-Frequency Interaction Fusion (SFIF) Module."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class SFIF(nn.Module):
    def __init__(self, channels: int = 256):
        super().__init__()
        self.channels = channels
        self.spatial_branch = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.freq_conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels * 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels * 2),
            nn.SiLU(),
            nn.Conv2d(channels * 2, channels * 2, kernel_size=1, bias=False),
        )
        self.freq_norm = nn.BatchNorm2d(channels)
        self.gate_s2f = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.gate_f2s = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.fuse_proj = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(),
        )
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        orig_dtype = x.dtype
        x_s = self.spatial_branch(x)

        # cuFFT requires float32 for non-power-of-two dimensions (e.g. 120, 60, 30)
        with torch.amp.autocast(device_type=x.device.type if x.is_cuda else "cpu", enabled=False):
            x_fp32 = x.float()
            x_fft = torch.fft.rfft2(x_fp32, norm="ortho")
            x_cat = torch.cat([x_fft.real, x_fft.imag], dim=1)
            x_freq_cat = self.freq_conv(x_cat)
            x_real_out, x_imag_out = torch.chunk(x_freq_cat, 2, dim=1)
            x_f = torch.fft.irfft2(torch.complex(x_real_out, x_imag_out), s=(h, w), norm="ortho")
            x_f = self.freq_norm(x_f).to(orig_dtype)

        g_s = self.gate_s2f(x_s)
        g_f = self.gate_f2s(x_f)
        f_sf = x_f * g_s
        f_fs = x_s * g_f
        fused = self.fuse_proj(torch.cat([f_sf, f_fs], dim=1))
        return x + self.gamma * fused
'''
    (fbd_modules_dir / "sfif.py").write_text(sfif_code, encoding="utf-8")

    # 3. dcfsb.py
    dcfsb_code = '''"""Dense Cross-scale Feature Screening / Fusion Block (DCFSB)."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class DCFSB(nn.Module):
    def __init__(self, channels: int = 256):
        super().__init__()
        self.channels = channels
        self.contrast_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.dw_conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, dilation=1, groups=channels, bias=False)
        self.dw_conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=2, dilation=2, groups=channels, bias=False)
        self.norm = nn.BatchNorm2d(channels)
        self.act = nn.SiLU()
        self.screening_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, kernel_size=1, bias=False),
            nn.SiLU(),
            nn.Conv2d(channels // 4, channels, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )
        self.out_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.out_norm = nn.BatchNorm2d(channels)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local_mean = self.contrast_pool(x)
        contrast = x - local_mean
        feat1 = self.dw_conv1(x + contrast)
        feat2 = self.dw_conv2(x + contrast)
        screened_feat = self.act(self.norm(feat1 + feat2))
        gate = self.screening_gate(screened_feat)
        screened = screened_feat * gate
        out = self.out_norm(self.out_proj(screened))
        return x + self.gamma * out
'''
    (fbd_modules_dir / "dcfsb.py").write_text(dcfsb_code, encoding="utf-8")

    (fbd_modules_dir / "__init__.py").write_text(
        'from .afta import AFTA\nfrom .sfif import SFIF\nfrom .dcfsb import DCFSB\n__all__ = ["AFTA", "SFIF", "DCFSB"]\n',
        encoding="utf-8"
    )

    # 4. hgnetv2_fbd.py in src/nn/backbone/
    backbone_dir = DFINE_REPO / "src" / "nn" / "backbone"
    hgnetv2_fbd_code = '''"""HGNetv2 with Adaptive Frequency-domain / Target Attention (HGNetv2FBD)."""
from __future__ import annotations
import torch
import torch.nn as nn
from ...core import register
from .hgnetv2 import HGNetv2
from ...zoo.dfine.fbd_modules.afta import AFTA

__all__ = ["HGNetv2FBD"]

@register()
class HGNetv2FBD(HGNetv2):
    def __init__(self, *args, use_afta: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_afta = use_afta
        out_channels = [self._out_channels[i] for i in self.return_idx]
        if self.use_afta:
            self.afta = nn.ModuleList([AFTA(ch) for ch in out_channels])
        else:
            self.afta = None

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        feats = super().forward(x)
        if self.use_afta and self.afta is not None:
            feats = [self.afta[i](f) for i, f in enumerate(feats)]
        return feats
'''
    (backbone_dir / "hgnetv2_fbd.py").write_text(hgnetv2_fbd_code, encoding="utf-8")

    # Update backbone __init__.py
    bb_init = backbone_dir / "__init__.py"
    bb_text = bb_init.read_text(encoding="utf-8")
    if "HGNetv2FBD" not in bb_text:
        bb_text += "\nfrom .hgnetv2_fbd import HGNetv2FBD\n"
        bb_init.write_text(bb_text, encoding="utf-8")

    # 5. fbd_hybrid_encoder.py in src/zoo/dfine/
    dfine_dir = DFINE_REPO / "src" / "zoo" / "dfine"
    encoder_code = '''"""FBDHybridEncoder: HybridEncoder with SFIF and DCFSB modules."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from ...core import register
from .fbd_modules.dcfsb import DCFSB
from .fbd_modules.sfif import SFIF
from .hybrid_encoder import HybridEncoder

__all__ = ["FBDHybridEncoder"]

@register()
class FBDHybridEncoder(HybridEncoder):
    def __init__(self, *args, use_sfif: bool = True, use_dcfsb: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_sfif = use_sfif
        self.use_dcfsb = use_dcfsb

        if use_sfif:
            self.sfif = nn.ModuleList([SFIF(self.hidden_dim) for _ in self.in_channels])
        else:
            self.sfif = None

        if use_dcfsb:
            num_fusions = len(self.in_channels) - 1
            self.dcfsb_fpn = nn.ModuleList([DCFSB(self.hidden_dim) for _ in range(num_fusions)])
            self.dcfsb_pan = nn.ModuleList([DCFSB(self.hidden_dim) for _ in range(num_fusions)])
        else:
            self.dcfsb_fpn = None
            self.dcfsb_pan = None

    def forward(self, feats: list[torch.Tensor]) -> list[torch.Tensor]:
        assert len(feats) == len(self.in_channels)
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]

        if self.use_sfif and self.sfif is not None:
            proj_feats = [self.sfif[i](feat) for i, feat in enumerate(proj_feats)]

        if self.num_encoder_layers > 0:
            for i, enc_ind in enumerate(self.use_encoder_idx):
                h, w = proj_feats[enc_ind].shape[2:]
                src_flatten = proj_feats[enc_ind].flatten(2).permute(0, 2, 1)
                if self.training or self.eval_spatial_size is None:
                    pos_embed = self.build_2d_sincos_position_embedding(
                        w, h, self.hidden_dim, self.pe_temperature
                    ).to(src_flatten.device)
                else:
                    pos_embed = getattr(self, f"pos_embed{enc_ind}", None).to(src_flatten.device)

                memory: torch.Tensor = self.encoder[i](src_flatten, pos_embed=pos_embed)
                proj_feats[enc_ind] = memory.permute(0, 2, 1).reshape(-1, self.hidden_dim, h, w).contiguous()

        inner_outs = [proj_feats[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            block_idx = len(self.in_channels) - 1 - idx
            feat_heigh = inner_outs[0]
            feat_low = proj_feats[idx - 1]
            feat_heigh = self.lateral_convs[block_idx](feat_heigh)
            inner_outs[0] = feat_heigh
            upsample_feat = F.interpolate(feat_heigh, scale_factor=2.0, mode="nearest")
            inner_out = self.fpn_blocks[block_idx](torch.concat([upsample_feat, feat_low], dim=1))
            if self.use_dcfsb and self.dcfsb_fpn is not None:
                inner_out = self.dcfsb_fpn[block_idx](inner_out)
            inner_outs.insert(0, inner_out)

        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_height = inner_outs[idx + 1]
            downsample_feat = self.downsample_convs[idx](feat_low)
            out = self.pan_blocks[idx](torch.concat([downsample_feat, feat_height], dim=1))
            if self.use_dcfsb and self.dcfsb_pan is not None:
                out = self.dcfsb_pan[idx](out)
            outs.append(out)

        return outs
'''
    (dfine_dir / "fbd_hybrid_encoder.py").write_text(encoder_code, encoding="utf-8")

    # Update dfine __init__.py
    dfine_init = dfine_dir / "__init__.py"
    dfine_text = dfine_init.read_text(encoding="utf-8")
    if "FBDHybridEncoder" not in dfine_text:
        dfine_text += "\nfrom .fbd_hybrid_encoder import FBDHybridEncoder\n"
        dfine_init.write_text(dfine_text, encoding="utf-8")

    log("FBDNet modules successfully injected into D-FINE!")


def find_dataset_root() -> Path:
    inputs = Path("/kaggle/input")
    candidates = [
        p for p in inputs.rglob("data.yaml")
        if "scale-error" in str(p).lower() or "yolov8" in str(p).lower()
    ]
    if candidates:
        log(f"Found data.yaml at {candidates[0]}")
        return candidates[0].parent

    all_data_yamls = list(inputs.rglob("data.yaml"))
    if all_data_yamls:
        log(f"Fallback data.yaml at {all_data_yamls[0]}")
        return all_data_yamls[0].parent

    raise FileNotFoundError("Could not find data.yaml in /kaggle/input")


def find_checkpoint(target_name: str = "dfine_m_coco.pth") -> Path | None:
    inputs = Path("/kaggle/input")
    for p in inputs.rglob(f"*{target_name}*"):
        if p.is_file() and p.suffix.lower() == ".pth":
            log(f"Found checkpoint: {p}")
            return p
    for p in inputs.rglob("*.pth"):
        if p.is_file() and ("best" in p.name.lower() or "coco" in p.name.lower()):
            log(f"Fallback checkpoint: {p}")
            return p
    return None


def convert_yolo_to_coco(images_dir: Path, labels_dir: Path, output_json: Path, class_names: list[str]) -> None:
    from PIL import Image
    images_list = []
    annotations = []
    ann_id = 1
    image_files = sorted([f for f in images_dir.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}])

    for img_id, img_path in enumerate(image_files, start=1):
        try:
            with Image.open(img_path) as img:
                w, h = img.size
        except Exception:
            w, h = 960, 960

        images_list.append({"id": img_id, "file_name": img_path.name, "width": w, "height": h})
        label_file = labels_dir / f"{img_path.stem}.txt"
        if not label_file.exists():
            continue

        for line in label_file.read_text(encoding="utf-8").strip().splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
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
    coco_data = {"images": images_list, "annotations": annotations, "categories": categories}
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(coco_data, indent=2), encoding="utf-8")
    log(f"Converted {len(images_list)} images, {len(annotations)} boxes -> {output_json}")


def main() -> None:
    log("Starting FBDNet-M adaptation training on Kaggle...")
    env_info = inspect_environment()
    install_dependencies()
    patch_dfine_engine()
    setup_fbdnet_modules()

    import yaml
    dataset_root = find_dataset_root()
    data_yaml_path = dataset_root / "data.yaml"
    cfg = yaml.safe_load(data_yaml_path.read_text(encoding="utf-8"))
    class_names = [cfg["names"][i] for i in sorted(cfg["names"].keys())] if isinstance(cfg["names"], dict) else cfg["names"]

    # Setup directories
    coco_dir = ARTIFACTS / "coco"
    coco_dir.mkdir(parents=True, exist_ok=True)
    train_json = coco_dir / "train.json"
    val_json = coco_dir / "val.json"

    train_img_dir = dataset_root / "train" / "images"
    train_lbl_dir = dataset_root / "train" / "labels"
    val_img_dir = dataset_root / "valid" / "images" if (dataset_root / "valid").exists() else dataset_root / "val" / "images"
    val_lbl_dir = dataset_root / "valid" / "labels" if (dataset_root / "valid").exists() else dataset_root / "val" / "labels"

    convert_yolo_to_coco(train_img_dir, train_lbl_dir, train_json, class_names)
    convert_yolo_to_coco(val_img_dir, val_lbl_dir, val_json, class_names)

    # 1. Dataset YAML
    dataset_yml_path = DFINE_REPO / "configs" / "dataset" / "glass_defect_fbdnet.yml"
    dataset_config = {
        "task": "detection",
        "evaluator": {"type": "CocoEvaluator", "iou_types": ["bbox"]},
        "num_classes": len(class_names),
        "remap_mscoco_category": False,
        "train_dataloader": {
            "type": "DataLoader",
            "dataset": {
                "type": "CocoDetection",
                "img_folder": str(train_img_dir),
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
                "img_folder": str(val_img_dir),
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
    dataset_yml_path.write_text(yaml.safe_dump(dataset_config, sort_keys=False), encoding="utf-8")

    # 2. Experiment YAML (FBDNet-M full configuration)
    exp_cfg_path = DFINE_REPO / "configs" / "dfine" / "custom" / "train_dfine_fbdnet_m.yml"
    exp_config = {
        "__include__": [
            "../../dataset/glass_defect_fbdnet.yml",
            "../../runtime.yml",
            "../include/dataloader.yml",
            "../include/optimizer.yml",
            "../include/dfine_hgnetv2.yml",
        ],
        "output_dir": str(EXP_DIR),
        "eval_spatial_size": [960, 960],
        "DFINE": {
            "backbone": "HGNetv2FBD",
            "encoder": "FBDHybridEncoder",
        },
        "HGNetv2FBD": {
            "name": "B2",
            "return_idx": [1, 2, 3],
            "freeze_at": -1,
            "freeze_norm": False,
            "use_lab": True,
            "pretrained": False,
            "use_afta": True,
        },
        "DFINETransformer": {
            "num_layers": 4,
            "eval_idx": -1,
        },
        "FBDHybridEncoder": {
            "in_channels": [384, 768, 1536],
            "feat_strides": [8, 16, 32],
            "hidden_dim": 256,
            "use_encoder_idx": [2],
            "num_encoder_layers": 1,
            "depth_mult": 0.67,
            "use_sfif": True,
            "use_dcfsb": True,
        },
        "optimizer": {
            "type": "AdamW",
            "params": [
                {"params": "^(?=.*backbone)(?!.*norm|bn).*$", "lr": 0.000025},
                {"params": "^(?=.*backbone)(?=.*norm|bn).*$", "lr": 0.000025, "weight_decay": 0.0},
                {"params": "^(?=.*(?:encoder|decoder))(?=.*(?:norm|bn|bias)).*$", "weight_decay": 0.0},
            ],
            "lr": 0.00025,
            "betas": [0.9, 0.999],
            "weight_decay": 0.000125,
        },
        "epochs": EPOCHS,
        "train_dataloader": {
            "total_batch_size": MICRO_BATCH_SIZE,
            "num_workers": 4,
            "dataset": {
                "transforms": {
                    "ops": [
                        {"type": "RandomPhotometricDistort", "p": 0.3},
                        {"type": "RandomIoUCrop", "p": 0.5},
                        {"type": "SanitizeBoundingBoxes", "min_size": 1},
                        {"type": "RandomHorizontalFlip"},
                        {"type": "Resize", "size": [960, 960]},
                        {"type": "SanitizeBoundingBoxes", "min_size": 1},
                        {"type": "ConvertPILImage", "dtype": "float32", "scale": True},
                        {"type": "ConvertBoxes", "fmt": "cxcywh", "normalize": True},
                    ],
                    "policy": {
                        "name": "stop_epoch",
                        "epoch": 16,
                        "ops": ["RandomPhotometricDistort", "RandomIoUCrop"],
                    },
                },
            },
            "collate_fn": {
                "type": "BatchImageCollateFunction",
                "base_size": 960,
                "base_size_repeat": 3,
                "stop_epoch": 16,
                "ema_restart_decay": 0.9999,
            },
        },
        "val_dataloader": {
            "total_batch_size": 8,
            "num_workers": 4,
            "dataset": {
                "transforms": {
                    "ops": [
                        {"type": "Resize", "size": [960, 960]},
                        {"type": "ConvertPILImage", "dtype": "float32", "scale": True},
                    ],
                },
            },
            "collate_fn": {"type": "BatchImageCollateFunction"},
        },
        "ema": {"warmups": 0},
        "lr_warmup_scheduler": {"warmup_duration": 0},
    }
    exp_cfg_path.write_text(yaml.safe_dump(exp_config, sort_keys=False), encoding="utf-8")
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(exp_cfg_path, EXP_DIR / "fbdnet_m_config.yaml")

    # Locate pretrained checkpoint (dfine_m_coco.pth or resume70)
    pretrained_ckpt = find_checkpoint("dfine_m_coco.pth")
    if not pretrained_ckpt:
        pretrained_ckpt = find_checkpoint("resume")

    log("=" * 60)
    log(f"Launching FBDNet-M ({EPOCHS} Epochs) Training...")
    log(f"Pretrained weights: {pretrained_ckpt}")
    log("=" * 60)

    train_cmd = [
        sys.executable, str(DFINE_REPO / "train.py"),
        "-c", str(exp_cfg_path),
        "--use-amp",
        "--seed", "42",
    ]
    if pretrained_ckpt:
        train_cmd.extend(["-t", str(pretrained_ckpt)])

    log(f"Running command: {' '.join(train_cmd)}")
    with open(EXP_DIR / "training_stdout.log", "w", encoding="utf-8") as out_fp:
        proc = subprocess.Popen(
            train_cmd,
            cwd=str(DFINE_REPO),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        for line in proc.stdout:
            sys.stdout.write(line)
            out_fp.write(line)
            out_fp.flush()
        proc.wait()

    if proc.returncode != 0:
        log(f"Training exited with code {proc.returncode}")
        sys.exit(proc.returncode)

    # Copy final model artifacts to /kaggle/working/
    for fn in ["best_stg1.pth", "best_stg2.pth", "best.pth", "last.pth"]:
        p = EXP_DIR / fn
        if p.exists():
            shutil.copy2(p, WORKING / f"fbdnet_m_{fn}")
            log(f"Saved artifact -> /kaggle/working/fbdnet_m_{fn}")

    log_txt = EXP_DIR / "log.txt"
    if log_txt.exists():
        shutil.copy2(log_txt, WORKING / "fbdnet_m_log.txt")
        # Generate clean summary CSV
        csv_rows = []
        for line in log_txt.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    d = json.loads(line)
                    bbox = d.get("test_coco_eval_bbox", [])
                    csv_rows.append({
                        "epoch": d.get("epoch"),
                        "train_loss": round(d.get("train_loss", 0), 4),
                        "val_mAP50_95": round(bbox[0] * 100, 2) if len(bbox) > 0 else None,
                        "val_mAP50": round(bbox[1] * 100, 2) if len(bbox) > 1 else None,
                        "val_mAP75": round(bbox[2] * 100, 2) if len(bbox) > 2 else None,
                    })
                except Exception:
                    pass
        if csv_rows:
            csv_path = WORKING / "fbdnet_m_training_log.csv"
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
                writer.writeheader()
                writer.writerows(csv_rows)
            log(f"Saved training log CSV -> {csv_path}")

    log("FBDNet-M Kaggle run completed successfully!")


if __name__ == "__main__":
    main()
