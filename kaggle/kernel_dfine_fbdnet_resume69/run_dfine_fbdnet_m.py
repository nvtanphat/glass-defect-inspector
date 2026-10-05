"""Self-contained Kaggle entrypoint for FBDNet-M on D-FINE-M.

Architecture:
- Backbone: HGNetv2FBD with Adaptive Frequency-domain / Target Attention (AFTA) at C3/C4/C5.
- Encoder: FBDHybridEncoder with:
    1. Spatial-Frequency Interaction Fusion (SFIF) at projected features (hidden_dim=256).
    2. Dense Cross-scale Feature Screening / Fusion Blocks (DCFSB) at multi-scale FPN/PAN fusion.
- Decoder: DFINETransformer (clean baseline compatibility).

Recipe (aligned 100% with reference D-FINE-M):
- RandomPhotometricDistort (p=0.5)
- RandomZoomOut (fill=0)
- RandomIoUCrop (p=0.8)
- stop_epoch = 88
- global batch = 8 (4 per GPU with 2-GPU DDP torchrun)
- epochs = 100
- Full ROI preprocessing and stratified train/val/test splits inline
- Robust NCCL CPU all_gather handling and faster-coco-eval compatibility
- Checkpoint resolution: best_stg2.pth -> best_stg1.pth -> best.pth
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Any

# Kaggle paths
WORKING = Path("/kaggle/working")
DATA_ROI = WORKING / "data_roi"
OUTPUT_DIR = WORKING / "output_fbdnet_m"
DFINE_REPO = WORKING / "D-FINE"
DFINE_WEIGHTS = WORKING / "dfine_m_coco.pth"

# PyTorch CUDA memory management for Tesla T4
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

ROI_CONFIG = {
    "5328x3000": (0.14113178913288288, 0.0, 0.8881345303115614, 1.0),
    "1024x1024": (0.0, 0.0, 1.0, 1.0),
}


def log(msg: str) -> None:
    print(f"[scale-error-fbdnet-m] {msg}", flush=True)


def patch_dist_utils_for_nccl_cpu() -> None:
    dist_file = DFINE_REPO / "src" / "misc" / "dist_utils.py"
    if not dist_file.exists():
        return
    text = dist_file.read_text(encoding="utf-8")
    if "_safe_all_gather" in text:
        return
    patch_code = '''
# Hotfix: prevent RuntimeError: No backend type associated with device type cpu under NCCL in PyTorch 2.6+
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
    log("Applied NCCL CPU all_gather try-except and find_unused_parameters=True hotfix to dist_utils.py")

    runtime_yml = DFINE_REPO / "configs" / "runtime.yml"
    if runtime_yml.exists():
        rtext = runtime_yml.read_text(encoding="utf-8")
        rtext = rtext.replace("find_unused_parameters: False", "find_unused_parameters: True")
        runtime_yml.write_text(rtext, encoding="utf-8")
        log("Updated runtime.yml: find_unused_parameters: True")


def patch_faster_coco_eval_file() -> None:
    try:
        import faster_coco_eval.utils.pytorch.coco_eval as ce
        file_path = Path(ce.__file__)
        text = file_path.read_text(encoding="utf-8")
        target_pattern = 'device = torch.device(f"cuda:{torch.cuda.current_device()}") if torch.cuda.is_available() else torch.device("cpu")'
        old_pattern = 'device = torch.device("cuda") if dist.get_backend() == "nccl" else torch.device("cpu")'
        if old_pattern in text:
            text = text.replace(old_pattern, target_pattern)
            file_path.write_text(text, encoding="utf-8")
            log(f"Patched {file_path.name} (replaced nccl check with cuda check)")
        elif 'device = torch.device("cpu")' in text:
            text = text.replace('device = torch.device("cpu")', target_pattern)
            file_path.write_text(text, encoding="utf-8")
            log(f"Patched {file_path.name} (replaced cpu device with cuda check)")
    except Exception as e:
        log(f"Warning: could not patch faster_coco_eval file: {e}")


def install_dependencies() -> None:
    log("Installing dependencies and setting up D-FINE repository...")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "iterative-stratification==0.1.9",
            "PyYAML>=6.0.2",
            "Pillow>=10.0",
            "pycocotools>=2.0.7",
            "tensorboard>=2.15",
            "opencv-python-headless>=4.8",
            "faster-coco-eval>=1.8.0",
        ],
        check=True,
    )
    if not DFINE_REPO.exists():
        subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "https://github.com/Peterande/D-FINE.git",
                str(DFINE_REPO),
            ],
            check=True,
        )
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", str(DFINE_REPO / "requirements.txt")],
        check=True,
    )
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--upgrade", "faster-coco-eval>=1.8.0"],
        check=True,
    )
    patch_dist_utils_for_nccl_cpu()
    patch_faster_coco_eval_file()


def setup_fbdnet_modules() -> None:
    log("Injecting FBDNet modules (AFTA, SFIF, DCFSB) into D-FINE codebase...")
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
            nn.Conv2d(channels, channels // 4, kernel_size=1, bias=False),
            nn.SiLU(),
            nn.Conv2d(channels // 4, channels, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )
        self.gate_f2s = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, kernel_size=1, bias=False),
            nn.SiLU(),
            nn.Conv2d(channels // 4, channels, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )
        self.fuse_proj = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        b, c, h, w = x.shape
        x_s = self.spatial_branch(x)

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
        encoding="utf-8",
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
                    pos_embed = getattr(self, f"pos_embed{enc_ind}", None)
                    if pos_embed is not None:
                        pos_embed = pos_embed.to(src_flatten.device)
                memory = self.encoder[i](src_flatten, pos_embed=pos_embed)
                proj_feats[enc_ind] = memory.permute(0, 2, 1).reshape(-1, self.hidden_dim, h, w)

        inner_outs = [proj_feats[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_heigh = inner_outs[0]
            feat_low = proj_feats[idx - 1]
            upsample_feat = F.interpolate(feat_heigh, scale_factor=2., mode='nearest')
            inner_out = self.fpn_blocks[len(self.in_channels) - 1 - idx](torch.cat([upsample_feat, feat_low], dim=1))
            if self.use_dcfsb and self.dcfsb_fpn is not None:
                fusion_idx = len(self.in_channels) - 1 - idx
                inner_out = self.dcfsb_fpn[fusion_idx](inner_out)
            inner_outs.insert(0, inner_out)

        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_height = inner_outs[idx + 1]
            downsample_feat = self.downsample_convs[idx](feat_low)
            out = self.pan_blocks[idx](torch.cat([downsample_feat, feat_height], dim=1))
            if self.use_dcfsb and self.dcfsb_pan is not None:
                out = self.dcfsb_pan[idx](out)
            outs.append(out)

        return outs
'''
    (dfine_dir / "fbd_hybrid_encoder.py").write_text(encoder_code, encoding="utf-8")
    dfine_init = dfine_dir / "__init__.py"
    df_text = dfine_init.read_text(encoding="utf-8")
    if "FBDHybridEncoder" not in df_text:
        df_text += "\nfrom .fbd_hybrid_encoder import FBDHybridEncoder\n"
        dfine_init.write_text(df_text, encoding="utf-8")
    log("FBDNet modules injected successfully.")


def find_dataset_root() -> Path:
    candidates = [
        p
        for p in Path("/kaggle/input").rglob("data.yaml")
        if "repo" not in str(p).lower() and "working" not in str(p).lower()
    ]
    if candidates:
        log(f"Found dataset data.yaml at: {candidates[0]}")
        return candidates[0].parent
    raise FileNotFoundError("Could not find raw dataset data.yaml under /kaggle/input")


def find_checkpoint(pattern: str) -> Path | None:
    for p in Path("/kaggle/input").rglob(pattern):
        if p.is_file():
            return p
    return None


def crop_boxes_to_roi(
    boxes: list[list[float]],
    roi_xyxyn: tuple[float, float, float, float],
    min_area_keep_frac: float = 0.10,
) -> list[list[float]]:
    roi_x1, roi_y1, roi_x2, roi_y2 = roi_xyxyn
    roi_w = roi_x2 - roi_x1
    roi_h = roi_y2 - roi_y1

    out: list[list[float]] = []
    for cls, x, y, w, h in boxes:
        bx1, by1, bx2, by2 = x - w / 2, y - h / 2, x + w / 2, y + h / 2
        orig_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        if orig_area <= 0:
            continue

        cx1, cy1 = max(bx1, roi_x1), max(by1, roi_y1)
        cx2, cy2 = min(bx2, roi_x2), min(by2, roi_y2)
        clipped_w, clipped_h = max(0.0, cx2 - cx1), max(0.0, cy2 - cy1)
        clipped_area = clipped_w * clipped_h
        if clipped_area <= 0 or (clipped_area / orig_area) < min_area_keep_frac:
            continue

        new_x1 = (cx1 - roi_x1) / roi_w
        new_y1 = (cy1 - roi_y1) / roi_h
        new_x2 = (cx2 - roi_x1) / roi_w
        new_y2 = (cy2 - roi_y1) / roi_h
        new_w = new_x2 - new_x1
        new_h = new_y2 - new_y1
        if new_w <= 0 or new_h <= 0:
            continue
        out.append([cls, new_x1 + new_w / 2, new_y1 + new_h / 2, new_w, new_h])
    return out


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


def apply_roi_crop_to_split(
    src_images_dir: Path,
    src_labels_dir: Path,
    dst_images_dir: Path,
    dst_labels_dir: Path,
) -> None:
    import cv2

    dst_images_dir.mkdir(parents=True, exist_ok=True)
    dst_labels_dir.mkdir(parents=True, exist_ok=True)

    img_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    img_files = sorted([f for f in src_images_dir.iterdir() if f.suffix.lower() in img_extensions])
    log(f"Applying ROI crop to {len(img_files)} images from {src_images_dir}...")

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

        cropped = img[y1:y2, x1:x2]
        cv2.imwrite(str(dst_img_path), cropped)

        new_boxes = crop_boxes_to_roi(boxes, roi_xyxyn, min_area_keep_frac=0.10)
        if new_boxes:
            write_label_file(dst_lbl_path, new_boxes)


def convert_yolo_to_coco(
    images_dir: Path,
    labels_dir: Path,
    output_json: Path,
    class_names: list[str],
) -> Path:
    from PIL import Image

    images_list = []
    annotations = []
    ann_id = 1
    image_files = sorted([
        f for f in images_dir.iterdir()
        if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    ])

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
    coco = {"images": images_list, "annotations": annotations, "categories": categories}
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(coco, indent=2), encoding="utf-8")
    log(f"COCO JSON written: {output_json} ({len(images_list)} images, {len(annotations)} annotations)")
    return output_json


def prepare_dataset(dataset_root: Path) -> tuple[dict[str, Path], list[str]]:
    import numpy as np
    import yaml
    from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

    raw_cfg = yaml.safe_load((dataset_root / "data.yaml").read_text(encoding="utf-8"))
    raw_names = raw_cfg["names"]
    if isinstance(raw_names, dict):
        class_names = [raw_names[k] for k in sorted(raw_names.keys(), key=int)]
    else:
        class_names = list(raw_names)
    nc = len(class_names)
    log(f"Classes ({nc}): {class_names}")

    raw_train_imgs = dataset_root / "train" / "images"
    raw_train_lbls = dataset_root / "train" / "labels"
    raw_val_imgs = dataset_root / "valid" / "images" if (dataset_root / "valid").exists() else dataset_root / "val" / "images"
    raw_val_lbls = dataset_root / "valid" / "labels" if (dataset_root / "valid").exists() else dataset_root / "val" / "labels"

    split_cache_dir = WORKING / "dataset_split"
    split_val_imgs = split_cache_dir / "val" / "images"
    split_val_lbls = split_cache_dir / "val" / "labels"
    split_test_imgs = split_cache_dir / "test" / "images"
    split_test_lbls = split_cache_dir / "test" / "labels"
    for d in [split_val_imgs, split_val_lbls, split_test_imgs, split_test_lbls]:
        d.mkdir(parents=True, exist_ok=True)

    val_images = sorted([
        f for f in raw_val_imgs.iterdir()
        if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    ])
    y = np.zeros((len(val_images), nc), dtype=np.int8)
    for i, img_p in enumerate(val_images):
        lbl_p = raw_val_lbls / f"{img_p.stem}.txt"
        for box in parse_label_file(lbl_p):
            c = int(box[0])
            if 0 <= c < nc:
                y[i, c] = 1

    splitter = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
    val_idx, test_idx = next(splitter.split(np.zeros(len(val_images)), y))
    log(f"Stratified split of valid: {len(val_idx)} val images, {len(test_idx)} test images")

    for idx in val_idx:
        img_p = val_images[int(idx)]
        shutil.copy2(img_p, split_val_imgs / img_p.name)
        lbl_p = raw_val_lbls / f"{img_p.stem}.txt"
        if lbl_p.exists():
            shutil.copy2(lbl_p, split_val_lbls / lbl_p.name)

    for idx in test_idx:
        img_p = val_images[int(idx)]
        shutil.copy2(img_p, split_test_imgs / img_p.name)
        lbl_p = raw_val_lbls / f"{img_p.stem}.txt"
        if lbl_p.exists():
            shutil.copy2(lbl_p, split_test_lbls / lbl_p.name)

    roi_train_dir = DATA_ROI / "train"
    roi_val_dir = DATA_ROI / "val"
    roi_test_dir = DATA_ROI / "test"

    apply_roi_crop_to_split(raw_train_imgs, raw_train_lbls, roi_train_dir, roi_train_dir / "labels")
    apply_roi_crop_to_split(split_val_imgs, split_val_lbls, roi_val_dir, roi_val_dir / "labels")
    apply_roi_crop_to_split(split_test_imgs, split_test_lbls, roi_test_dir, roi_test_dir / "labels")
    if split_cache_dir.exists():
        shutil.rmtree(split_cache_dir, ignore_errors=True)

    train_json = convert_yolo_to_coco(roi_train_dir, roi_train_dir / "labels", roi_train_dir / "train.json", class_names)
    val_json = convert_yolo_to_coco(roi_val_dir, roi_val_dir / "labels", roi_val_dir / "val.json", class_names)
    test_json = convert_yolo_to_coco(roi_test_dir, roi_test_dir / "labels", roi_test_dir / "test.json", class_names)

    paths = {
        "train_img": roi_train_dir,
        "train_json": train_json,
        "val_img": roi_val_dir,
        "val_json": val_json,
        "test_img": roi_test_dir,
        "test_json": test_json,
    }
    return paths, class_names


def setup_fbdnet_configs(paths: dict[str, Path], class_names: list[str]) -> tuple[Path, Path]:
    import yaml

    nc = len(class_names)

    # 1. glass_defect_fbdnet.yml
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
    ds_train_path = DFINE_REPO / "configs" / "dataset" / "glass_defect_fbdnet.yml"
    ds_train_path.parent.mkdir(parents=True, exist_ok=True)
    ds_train_path.write_text(yaml.safe_dump(ds_train_cfg, sort_keys=False), encoding="utf-8")

    # 2. glass_defect_fbdnet_test.yml
    ds_test_cfg = {
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
                "img_folder": str(paths["test_img"]),
                "ann_file": str(paths["test_json"]),
                "return_masks": False,
                "transforms": {"type": "Compose", "ops": None},
            },
            "shuffle": False,
            "num_workers": 4,
            "drop_last": False,
            "collate_fn": {"type": "BatchImageCollateFunction"},
        },
    }
    ds_test_path = DFINE_REPO / "configs" / "dataset" / "glass_defect_fbdnet_test.yml"
    ds_test_path.write_text(yaml.safe_dump(ds_test_cfg, sort_keys=False), encoding="utf-8")

    # 3. Model recipe matching reference D-FINE-M with FBDNet-M enhancements
    base_model_cfg = {
        "__include__": [
            "../../dataset/glass_defect_fbdnet.yml",
            "../../runtime.yml",
            "../include/dataloader.yml",
            "../include/optimizer.yml",
            "../include/dfine_hgnetv2.yml",
        ],
        "output_dir": str(OUTPUT_DIR),
        "eval_spatial_size": [960, 960],
        "find_unused_parameters": True,
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
        "DFINETransformer": {"num_layers": 4, "eval_idx": -1},
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
                {
                    "params": "^(?=.*backbone)(?!.*norm|bn).*$",
                    "lr": 0.000025,
                },
                {
                    "params": "^(?=.*backbone)(?=.*norm|bn).*$",
                    "lr": 0.000025,
                    "weight_decay": 0.0,
                },
                {
                    "params": "^(?=.*(?:encoder|decoder))(?=.*(?:norm|bn|bias)).*$",
                    "weight_decay": 0.0,
                },
            ],
            "lr": 0.00025,
            "betas": [0.9, 0.999],
            "weight_decay": 0.000125,
        },
        "epochs": 100,
        "train_dataloader": {
            "total_batch_size": 8,
            "num_workers": 4,
            "dataset": {
                "transforms": {
                    "ops": [
                        {"type": "RandomPhotometricDistort", "p": 0.5},
                        {"type": "RandomZoomOut", "fill": 0},
                        {"type": "RandomIoUCrop", "p": 0.8},
                        {"type": "SanitizeBoundingBoxes", "min_size": 1},
                        {"type": "RandomHorizontalFlip"},
                        {"type": "Resize", "size": [960, 960]},
                        {"type": "SanitizeBoundingBoxes", "min_size": 1},
                        {"type": "ConvertPILImage", "dtype": "float32", "scale": True},
                        {"type": "ConvertBoxes", "fmt": "cxcywh", "normalize": True},
                    ],
                    "policy": {
                        "name": "stop_epoch",
                        "epoch": 88,
                        "ops": ["RandomPhotometricDistort", "RandomZoomOut", "RandomIoUCrop"],
                    },
                }
            },
            "collate_fn": {
                "type": "BatchImageCollateFunction",
                "base_size": 960,
                "base_size_repeat": 3,
                "stop_epoch": 88,
                "ema_restart_decay": 0.9999,
            },
        },
        "val_dataloader": {
            "total_batch_size": 16,
            "num_workers": 4,
            "dataset": {
                "transforms": {
                    "ops": [
                        {"type": "Resize", "size": [960, 960]},
                        {"type": "ConvertPILImage", "dtype": "float32", "scale": True},
                    ]
                }
            },
            "collate_fn": {"type": "BatchImageCollateFunction"},
        },
        "ema": {"warmups": 0},
        "lr_warmup_scheduler": {"warmup_duration": 0},
    }

    train_exp_path = DFINE_REPO / "configs" / "dfine" / "custom" / "train_dfine_fbdnet_m.yml"
    train_exp_path.parent.mkdir(parents=True, exist_ok=True)
    train_exp_path.write_text(yaml.safe_dump(base_model_cfg, sort_keys=False), encoding="utf-8")

    test_model_cfg = dict(base_model_cfg)
    test_model_cfg["__include__"] = [
        "../../dataset/glass_defect_fbdnet_test.yml",
        "../../runtime.yml",
        "../include/dataloader.yml",
        "../include/optimizer.yml",
        "../include/dfine_hgnetv2.yml",
    ]
    test_exp_path = DFINE_REPO / "configs" / "dfine" / "custom" / "train_dfine_fbdnet_m_test.yml"
    test_exp_path.write_text(yaml.safe_dump(test_model_cfg, sort_keys=False), encoding="utf-8")

    log(f"FBDNet-M experiment configs generated: {train_exp_path} and {test_exp_path}")
    return train_exp_path, test_exp_path


def parse_coco_eval_output(stdout: str) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for line in stdout.splitlines():
        line = line.strip()
        if "Average Precision" in line and "IoU=0.50:0.95" in line and "area=   all" in line:
            try:
                metrics["mAP50_95"] = round(float(line.split("=")[-1].strip()) * 100, 2)
            except ValueError:
                pass
        elif "Average Precision" in line and "IoU=0.50 " in line and "area=   all" in line:
            try:
                metrics["mAP50"] = round(float(line.split("=")[-1].strip()) * 100, 2)
            except ValueError:
                pass
        elif "Average Precision" in line and "IoU=0.75" in line and "area=   all" in line:
            try:
                metrics["mAP75"] = round(float(line.split("=")[-1].strip()) * 100, 2)
            except ValueError:
                pass
    return metrics


def main() -> None:
    log("Starting FBDNet-M 100-epoch training with reference D-FINE recipe...")
    install_dependencies()
    setup_fbdnet_modules()

    import torch

    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    device_name = torch.cuda.get_device_name(0) if num_gpus > 0 else "CPU"
    log(f"Detected {num_gpus} CUDA device(s): {device_name}")

    dataset_root = find_dataset_root()
    paths, class_names = prepare_dataset(dataset_root)
    train_cfg_path, test_cfg_path = setup_fbdnet_configs(paths, class_names)

    # Weights resolution
    if not DFINE_WEIGHTS.exists():
        coco_candidates = list(Path("/kaggle/input").rglob("dfine_m_coco.pth"))
        if coco_candidates:
            log(f"Using attached COCO weights from {coco_candidates[0]}")
            shutil.copy2(coco_candidates[0], DFINE_WEIGHTS)
        else:
            log("Downloading dfine_m_coco.pth from GitHub release...")
            urllib.request.urlretrieve(
                "https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_m_coco.pth",
                DFINE_WEIGHTS,
            )

    resume_ckpt = find_checkpoint("last.pth")
    if not resume_ckpt:
        log("last.pth not found in attached datasets. Attempting direct API fetch from prior kernel session...")
        try:
            import base64
            from kaggle.api.kaggle_api_extended import ApiListKernelSessionOutputRequest, KaggleApi
            if "KAGGLE_KEY" not in os.environ and not Path("~/.kaggle/kaggle.json").expanduser().exists():
                os.environ["KAGGLE_USERNAME"] = "baohuynh05"
                os.environ["KAGGLE_KEY"] = "fb0e3d800d4ed28d2dab13ec7fbed646"
            api = KaggleApi()
            api.authenticate()
            with api.build_kaggle_client() as client:
                req = ApiListKernelSessionOutputRequest()
                req.user_name = "baohuynh05"
                req.kernel_slug = "scale-error-dfine-fbdnet-m-resume46"
                api._set_paging(req, 1)
                res = client.kernels.kernels_api_client.list_kernel_session_output(req)
                prefix = json.loads(base64.b64decode(res.next_page_token).decode())["GcsPageToken"].split("/output/")[0]
                target_token = base64.b64encode(json.dumps({"GcsPageToken": f"{prefix}/output/output_fbdnet_m"}).encode()).decode()
                req2 = ApiListKernelSessionOutputRequest()
                req2.user_name = "baohuynh05"
                req2.kernel_slug = "scale-error-dfine-fbdnet-m-resume46"
                api._set_paging(req2, 200, target_token)
                res2 = client.kernels.kernels_api_client.list_kernel_session_output(req2)
                f_map = {f.file_name: f.url for f in (res2.files or [])}
            
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            for fname, dest_name in [("output_fbdnet_m/last.pth", "last.pth"), ("output_fbdnet_m/best_stg1.pth", "best_stg1.pth"), ("output_fbdnet_m/log.txt", "log.txt")]:
                if fname in f_map:
                    log(f"Downloading {dest_name} directly inside Kaggle VM...")
                    urllib.request.urlretrieve(f_map[fname], OUTPUT_DIR / dest_name)
                    log(f"Successfully downloaded {dest_name} ({(OUTPUT_DIR / dest_name).stat().st_size / (1024*1024):.1f} MB)")
            if (OUTPUT_DIR / "last.pth").exists():
                resume_ckpt = OUTPUT_DIR / "last.pth"
        except Exception as e:
            log(f"Warning: Failed to fetch prior checkpoint via Kaggle API: {e}")

    if resume_ckpt and "coco" not in str(resume_ckpt).lower():
        log(f"Resuming training from: {resume_ckpt}")
        ckpt_flag = "-r"
        ckpt_path = str(resume_ckpt)
    else:
        log(f"Starting training with COCO tuning weights: {DFINE_WEIGHTS}")
        ckpt_flag = "-t"
        ckpt_path = str(DFINE_WEIGHTS)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Restore prior best checkpoint and log if available from resume dataset
    prior_best = find_checkpoint('best_stg1.pth')
    if prior_best and prior_best.exists() and not (OUTPUT_DIR / 'best_stg1.pth').exists():
        shutil.copy2(prior_best, OUTPUT_DIR / 'best_stg1.pth')
        log(f'Restored prior best checkpoint from {prior_best}')
    prior_log = find_checkpoint('log.txt')
    if prior_log and prior_log.exists() and not (OUTPUT_DIR / 'log.txt').exists():
        shutil.copy2(prior_log, OUTPUT_DIR / 'log.txt')
        log(f'Restored prior log.txt from {prior_log}')

    # Execute training
    if num_gpus >= 2:
        log("Launching 2-GPU DDP training with torchrun (global batch size 8, 4 per GPU)...")
        train_cmd = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            f"--nproc_per_node={num_gpus}",
            "--master_port=7777",
            str(DFINE_REPO / "train.py"),
            "-c",
            str(train_cfg_path),
            ckpt_flag,
            ckpt_path,
            "--use-amp",
            "--seed",
            "42",
            "--output-dir",
            str(OUTPUT_DIR),
            "--summary-dir",
            str(OUTPUT_DIR / "summary"),
        ]
    else:
        log("Launching single-device training...")
        train_cmd = [
            sys.executable,
            str(DFINE_REPO / "train.py"),
            "-c",
            str(train_cfg_path),
            ckpt_flag,
            ckpt_path,
            "--use-amp",
            "--seed",
            "42",
            "--output-dir",
            str(OUTPUT_DIR),
            "--summary-dir",
            str(OUTPUT_DIR / "summary"),
        ]

    log(f"Running command: {' '.join(train_cmd)}")
    with open(OUTPUT_DIR / "train_stdout.log", "w", encoding="utf-8") as out_fp:
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
        log(f"FBDNet-M training failed with return code {proc.returncode}")
        sys.exit(proc.returncode)

    log("Training completed! Resolving best checkpoint...")
    best_ckpt = None
    for ckpt_name in ["best_stg2.pth", "best_stg1.pth", "best.pth"]:
        candidate = OUTPUT_DIR / ckpt_name
        if candidate.exists():
            best_ckpt = candidate
            break

    if best_ckpt is None:
        raise FileNotFoundError(f"No best checkpoint found in {OUTPUT_DIR}")

    log(f"Selected best checkpoint for evaluation: {best_ckpt.name} ({best_ckpt})")
    shutil.copy2(best_ckpt, WORKING / best_ckpt.name)

    # Clean up data_roi to prevent Kaggle output clutter (>18,000 files)
    if DATA_ROI.exists():
        shutil.rmtree(DATA_ROI, ignore_errors=True)
        log("Cleaned up temporary DATA_ROI directory")

    # Copy checkpoints and logs to /kaggle/working/ for simple downloading
    for name in ["best_stg2.pth", "best_stg1.pth", "best.pth", "last.pth"]:
        src_f = OUTPUT_DIR / name
        if src_f.exists():
            shutil.copy2(src_f, WORKING / f"fbdnet_m_{name}")
            log(f"Saved artifact: /kaggle/working/fbdnet_m_{name}")

    if (OUTPUT_DIR / "log.txt").exists():
        shutil.copy2(OUTPUT_DIR / "log.txt", WORKING / "fbdnet_m_log.txt")

    # Evaluate on held-out test split
    log("Running final evaluation on held-out test split...")
    eval_cmd = [
        sys.executable,
        str(DFINE_REPO / "train.py"),
        "-c",
        str(test_cfg_path),
        "-r",
        str(best_ckpt),
        "--test-only",
        "--device",
        "cuda:0" if num_gpus > 0 else "cpu",
    ]
    log(f"Evaluation command: {' '.join(eval_cmd)}")
    eval_proc = subprocess.run(
        eval_cmd,
        cwd=str(DFINE_REPO),
        capture_output=True,
        text=True,
    )
    print(eval_proc.stdout)
    (WORKING / "test_evaluation_stdout.log").write_text(eval_proc.stdout, encoding="utf-8")

    metrics = parse_coco_eval_output(eval_proc.stdout)
    metrics["checkpoint"] = best_ckpt.name
    metrics["model"] = "FBDNet-M (100 Epochs Full Recipe)"
    metrics_path = WORKING / "metrics_fbdnet_m.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    log(f"Evaluation metrics saved to {metrics_path}: {metrics}")
    log("FBDNet-M 100-epoch execution complete!")


if __name__ == "__main__":
    main()
