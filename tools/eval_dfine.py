"""Chay inference D-FINE-M tren tap test ROI (890 anh) va tao:
  1) Xac minh mAP bang pycocotools (ky vong ~44.40 / mAP50 ~76.90)
  2) Confusion matrix tai IoU=0.5, conf=0.5  -> report_assets/dfine_m_cm.png
  3) Anh du doan ve tren anh goc (map toa do ROI ve anh goc) -> report_assets/dfine_m_pred_*.jpg
  4) Precision/Recall/F1 tung lop -> results_report/dfine_m_confusion.json
"""
import json
import os
import random
import sys
import time

# tensorflow bi con trong site-packages (thu muc rong, khong co __init__) lam
# torch.utils.tensorboard / tensorboard.compat that bai -> thay bang stub nhe
try:
    import tensorflow as _tf_probe  # noqa: F401
    _tf_broken = not hasattr(_tf_probe, "__version__")
except Exception:
    _tf_broken = True
if _tf_broken:
    import types as _types
    _tf_stub = _types.ModuleType("tensorflow")
    _tf_stub.__version__ = "stub"
    from importlib.machinery import ModuleSpec as _ModuleSpec
    _tf_stub.__spec__ = _ModuleSpec("tensorflow", loader=None)
    _tf_stub.__path__ = []
    _tf_stub.io = _types.SimpleNamespace(
        gfile=_types.SimpleNamespace(join=lambda *a: "/".join(str(x) for x in a))
    )
    sys.modules["tensorflow"] = _tf_stub

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "results", "D-FINE"))

import numpy as np
import torch
torch.set_num_threads(8)
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from src.core import YAMLConfig

CFG = os.path.join(ROOT, "results/D-FINE/configs/dfine/custom/dfine_hgnetv2_m_glass_test.yml")
CKPT = os.path.join(ROOT, "weights/dfine_m_best.pth")
GT_JSON = os.path.join(ROOT, "results/data_roi/test/test.json")
ROI_IMG_DIR = os.path.join(ROOT, "results/data_roi/test")
RAW_IMG_DIR = os.path.join(ROOT, "data/processed/test/images")
ROI_CONFIG = os.path.join(ROOT, "processing/roi_config.json")

OUT_DIR = os.path.join(ROOT, "report_assets")
RES_DIR = os.path.join(ROOT, "results_report")
CONF_TH = 0.5
IOU_TH = 0.5
N_SAMPLES = 6
BATCH = 4

CLASSES = ["DiVat", "DiVatLoiLom", "LoiChi", "LoiNhua", "LoiTray"]
COLORS = ["#e63946", "#f4a261", "#2a9d8f", "#457b9d", "#9b5de5"]
BATCH = 4


def build_model():
    cfg = YAMLConfig(CFG)
    ckpt = torch.load(CKPT, map_location="cpu")
    if "ema" in ckpt:
        state = ckpt["ema"]["module"]
        print("Dung trong so EMA (epoch %s)" % ckpt.get("last_epoch"))
    else:
        state = ckpt["model"]
        print("Dung trong so model thuc")
    cfg.model.load_state_dict(state)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            return self.postprocessor(outputs, orig_target_sizes)

    m = Model()
    m.eval()
    return m


def run_inference(model, images_info):
    """images_info: list of (file_name, w, h). Tra ve dict image_id -> (boxes, scores, labels)."""
    tf = T.Compose([T.Resize((960, 960)), T.ToTensor()])
    preds = {}
    t0 = time.time()
    for i in range(0, len(images_info), BATCH):
        batch = images_info[i:i + BATCH]
        tensors, sizes = [], []
        pils = []
        for info in batch:
            im = Image.open(os.path.join(ROI_IMG_DIR, info["file_name"])).convert("RGB")
            pils.append(im)
            sizes.append([im.size[0], im.size[1]])
            tensors.append(tf(im))
        samples = torch.stack(tensors, dim=0)
        orig_sizes = torch.tensor(sizes, dtype=torch.int64)
        with torch.no_grad():
            labels, boxes, scores = model(samples, orig_sizes)
        for j, info in enumerate(batch):
            preds[info["id"]] = (
                boxes[j].cpu().numpy(),
                scores[j].cpu().numpy(),
                labels[j].cpu().numpy(),
            )
        done = i + len(batch)
        if done % 40 == 0 or done == len(images_info):
            el = time.time() - t0
            print(f"  {done}/{len(images_info)} anh  ({el:.0f}s, "
                  f"{el / done:.2f}s/anh, con lai ~{el / done * (len(images_info) - done):.0f}s)",
                  flush=True)
    return preds


def coco_eval(preds):
    coco = COCO(GT_JSON)
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
    return {"mAP50_95": float(ev.stats[0]), "mAP50": float(ev.stats[1]),
            "mAP75": float(ev.stats[2]), "mAP_small": float(ev.stats[3]),
            "mAP_medium": float(ev.stats[4]), "mAP_large": float(ev.stats[5])}


def iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    ua = ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return inter / ua if ua > 0 else 0.0


def confusion_matrix(preds, gt_by_image):
    """Ghep nhan GT voi du doan (greedy theo diem, IoU>=0.5)."""
    n = len(CLASSES)
    mat = np.zeros((n + 1, n + 1), dtype=int)  # hang = GT (cot cuoi = background), cot = pred (hang cuoi = FP)
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
                v = iou(boxes[k], g["bbox"])
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


def plot_confusion(mat, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = CLASSES + ["background"]
    fig, ax = plt.subplots(figsize=(8.2, 7), dpi=140)
    im = ax.imshow(mat, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(len(labels)), labels, fontsize=9)
    ax.set_xlabel("Dự đoán", fontsize=11)
    ax.set_ylabel("Ground truth", fontsize=11)
    ax.set_title("D-FINE-M · Confusion matrix trên test set (890 ảnh)\n"
                 "IoU ≥ 0.5 · conf ≥ 0.5 (greedy matching)", fontsize=11)
    mx = max(mat.max(), 1)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if v:
                ax.text(j, i, str(v), ha="center", va="center", fontsize=8,
                        color="white" if v > 0.55 * mx else "#12314f")
            else:
                ax.text(j, i, "0", ha="center", va="center", fontsize=8, color="#9db4c8")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print("DA LUU", out_path)


def roi_offset(raw_w, raw_h, roi_cfg):
    g = roi_cfg["groups"].get(f"{raw_w}x{raw_h}")
    if g is None:
        return 0.0, 0.0
    x0, y0 = g["roi_xyxyn"][0], g["roi_xyxyn"][1]
    return x0 * raw_w, y0 * raw_h


def draw_samples(preds, gt_by_image, roi_cfg, out_path_prefix):
    """Ve du doan tren anh GOC (map toa do ROI ve toa do anh goc)."""
    n = len(CLASSES)
    # chon anh: nhieu GT + phong phu lop
    def score(item):
        img_id, gts = item
        return len(gts)
    chosen = [i for i, _ in sorted(gt_by_image.items(), key=score, reverse=True)[:N_SAMPLES]]

    info_by_id = {im["id"]: im for im in json.load(open(GT_JSON, encoding="utf-8"))["images"]}
    paths = []
    for idx, img_id in enumerate(chosen):
        info = info_by_id[img_id]
        raw_path = os.path.join(RAW_IMG_DIR, info["file_name"])
        if not os.path.exists(raw_path):
            cand = [f for f in os.listdir(RAW_IMG_DIR) if info["file_name"].split("_rf.")[0] in f]
            if not cand:
                print("SKIP (khong tim thay anh goc):", info["file_name"])
                continue
            raw_path = os.path.join(RAW_IMG_DIR, cand[0])
        im = Image.open(raw_path).convert("RGB")
        W, H = im.size
        ox, oy = roi_offset(W, H, roi_cfg)
        draw = ImageDraw.Draw(im)
        try:
            font = ImageFont.truetype("arial.ttf", max(18, W // 220))
        except Exception:
            font = ImageFont.load_default()

        # GT: vien xanh duong
        for g in gt_by_image[img_id]:
            x0, y0, x1, y1 = g["bbox"]
            draw.rectangle([x0 + ox, y0 + oy, x1 + ox, y1 + oy], outline="#1d9bf0", width=4)
        # pred conf>=0.5: vien mau lop
        boxes, scores, labels = preds[img_id]
        for k in np.argsort(-scores):
            if scores[k] < CONF_TH:
                break
            c = int(labels[k])
            x0, y0, x1, y1 = boxes[k]
            draw.rectangle([x0 + ox, y0 + oy, x1 + ox, y1 + oy],
                           outline=COLORS[c], width=5)
            draw.text((x0 + ox, y0 + oy - 26), f"{CLASSES[c]} {scores[k]:.2f}",
                      fill=COLORS[c], font=font)

        # nho lai de dua vao README
        scale = min(1.0, 1500 / W)
        if scale < 1.0:
            im = im.resize((int(W * scale), int(H * scale)), Image.LANCZOS)
        out = f"{out_path_prefix}_{idx + 1}.jpg"
        im.save(out, quality=88)
        paths.append(out)
        print("DA LUU", out, f"(GT {len(gt_by_image[img_id])} box)")
    return paths


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(RES_DIR, exist_ok=True)
    roi_cfg = json.load(open(ROI_CONFIG, encoding="utf-8"))

    gt_json = json.load(open(GT_JSON, encoding="utf-8"))
    images_info = sorted(gt_json["images"], key=lambda x: x["id"])
    gt_by_image = {}
    for a in gt_json["annotations"]:
        gt_by_image.setdefault(a["image_id"], []).append(a)
    gt_by_image = {i: gt_by_image.get(i, []) for i in (im["id"] for im in images_info)}

    print(f"Tong so anh test: {len(images_info)}, tong GT box: "
          f"{sum(len(v) for v in gt_by_image.values())}")
    model = build_model()
    preds = run_inference(model, images_info)

    print("\n=== COCO mAP (pycocotools) ===")
    metrics = coco_eval(preds)

    print("\n=== Confusion matrix ===")
    mat, stats = confusion_matrix(preds, gt_by_image)
    plot_confusion(mat, os.path.join(OUT_DIR, "dfine_m_cm.png"))

    per_class = {}
    for c, name in enumerate(CLASSES):
        tp, fp, fn = stats[c]["tp"], stats[c]["fp"], stats[c]["fn"]
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        per_class[name] = {"tp": tp, "fp": fp, "fn": fn,
                           "precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4)}
        print(f"  {name:14s} P={p:.3f} R={r:.3f} F1={f1:.3f} (tp={tp} fp={fp} fn={fn})")

    out = {"checkpoint": CKPT, "conf_threshold": CONF_TH, "iou_threshold": IOU_TH,
           "coco_metrics": metrics, "per_class": per_class,
           "confusion_matrix": mat.tolist(), "class_order": CLASSES}
    with open(os.path.join(RES_DIR, "dfine_m_confusion.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("DA LUU results_report/dfine_m_confusion.json")

    draw_samples(preds, gt_by_image, roi_cfg, os.path.join(OUT_DIR, "dfine_m_pred"))


if __name__ == "__main__":
    main()
