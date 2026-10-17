import gzip
import json
import pickle
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from scale_error.evaluation.detection_confusion import detection_confusion  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ASSETS_DIR = ROOT / "docs" / "assets"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

CLASSES = ["DiVat", "DiVatLoiLom", "LoiChi", "LoiNhua", "LoiTray"]
LABELS = CLASSES + ["background"]


def compute_iou(b1, b2):
    x1 = max(float(b1[0]), float(b2[0]))
    y1 = max(float(b1[1]), float(b2[1]))
    x2 = min(float(b1[2]), float(b2[2]))
    y2 = min(float(b1[3]), float(b2[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    a1 = max(0.0, float(b1[2]) - float(b1[0])) * max(0.0, float(b1[3]) - float(b1[1]))
    a2 = max(0.0, float(b2[2]) - float(b2[0])) * max(0.0, float(b2[3]) - float(b2[1]))
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def fast_wbf_engine(
    boxes_list,
    scores_list,
    weights,
    iou_thr,
    coord_ratio,
    boost_3,
    boost_2,
    solo_decay,
    conf_floor,
):
    num_m = len(boxes_list)
    items = []
    for m in range(num_m):
        for b, s in zip(boxes_list[m], scores_list[m], strict=False):
            if s >= conf_floor:
                items.append((np.array(b, dtype=np.float32), float(s), m))
    if not items:
        return [], []
    items.sort(key=lambda x: x[1], reverse=True)
    clusters = []
    for b_curr, s_curr, m_curr in items:
        best_iou = iou_thr
        best_idx = -1
        for c_idx, c_info in enumerate(clusters):
            iou = compute_iou(b_curr, c_info["center"])
            if iou > best_iou:
                best_iou = iou
                best_idx = c_idx
        if best_idx >= 0:
            clusters[best_idx]["members"].append((m_curr, b_curr, s_curr))
        else:
            clusters.append({"center": b_curr.copy(), "members": [(m_curr, b_curr, s_curr)]})

    out_boxes = []
    out_scores = []
    for c in clusters:
        m_list = c["members"]
        if num_m == 3:
            m0 = [x for x in m_list if x[0] == 0]
            m1 = [x for x in m_list if x[0] == 1]
            m2 = [x for x in m_list if x[0] == 2]
            if m0 and m1 and m2:
                cw_sum = coord_ratio[0] + coord_ratio[1] + coord_ratio[2]
                box = (
                    m0[0][1] * (coord_ratio[0] / cw_sum)
                    + m1[0][1] * (coord_ratio[1] / cw_sum)
                    + m2[0][1] * (coord_ratio[2] / cw_sum)
                ).tolist()
                s_avg = (
                    m0[0][2] * weights[0] + m1[0][2] * weights[1] + m2[0][2] * weights[2]
                ) / (weights[0] + weights[1] + weights[2])
                score = min(1.0, s_avg * boost_3)
            elif m0 and m1:
                cw_sum = coord_ratio[0] + coord_ratio[1]
                box = (
                    m0[0][1] * (coord_ratio[0] / cw_sum) + m1[0][1] * (coord_ratio[1] / cw_sum)
                ).tolist()
                s_avg = (m0[0][2] * weights[0] + m1[0][2] * weights[1]) / (
                    weights[0] + weights[1]
                )
                score = min(1.0, s_avg * boost_2)
            elif m0 and m2:
                cw_sum = coord_ratio[0] + coord_ratio[2]
                box = (
                    m0[0][1] * (coord_ratio[0] / cw_sum) + m2[0][1] * (coord_ratio[2] / cw_sum)
                ).tolist()
                s_avg = (m0[0][2] * weights[0] + m2[0][2] * weights[2]) / (
                    weights[0] + weights[2]
                )
                score = min(1.0, s_avg * boost_2)
            elif m1 and m2:
                cw_sum = coord_ratio[1] + coord_ratio[2]
                box = (
                    m1[0][1] * (coord_ratio[1] / cw_sum) + m2[0][1] * (coord_ratio[2] / cw_sum)
                ).tolist()
                s_avg = (m1[0][2] * weights[1] + m2[0][2] * weights[2]) / (
                    weights[1] + weights[2]
                )
                score = min(1.0, s_avg * boost_2)
            elif m0:
                box = m0[0][1].tolist()
                score = m0[0][2] * solo_decay
            elif m1:
                box = m1[0][1].tolist()
                score = m1[0][2] * (solo_decay * 0.98)
            else:
                box = m2[0][1].tolist()
                score = m2[0][2] * (solo_decay * 0.94)
        else:
            m0 = [x for x in m_list if x[0] == 0]
            m1 = [x for x in m_list if x[0] == 1]
            if m0 and m1:
                cw_sum = coord_ratio[0] + coord_ratio[1]
                box = (
                    m0[0][1] * (coord_ratio[0] / cw_sum) + m1[0][1] * (coord_ratio[1] / cw_sum)
                ).tolist()
                s_avg = (m0[0][2] * weights[0] + m1[0][2] * weights[1]) / (
                    weights[0] + weights[1]
                )
                score = min(1.0, s_avg * boost_2)
            elif m0:
                box = m0[0][1].tolist()
                score = m0[0][2] * solo_decay
            else:
                box = m1[0][1].tolist()
                score = m1[0][2] * (solo_decay * 0.95)

        if score >= conf_floor:
            out_boxes.append(box)
            out_scores.append(float(score))

    return out_boxes, out_scores


def plot_cm(mat, title, out_path):
    fig, ax = plt.subplots(figsize=(8.5, 7.5), dpi=200)
    im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=1.0)
    ax.set_xticks(range(len(LABELS)))
    ax.set_xticklabels(LABELS, rotation=35, ha="right", fontsize=9.5, fontweight="medium")
    ax.set_yticks(range(len(LABELS)))
    ax.set_yticklabels(LABELS, fontsize=9.5, fontweight="medium")
    ax.set_xlabel("True Class (Ground Truth)", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_ylabel("Predicted Class", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_title(title, fontsize=12.5, fontweight="bold", pad=14)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            if val > 0.005:
                color = "white" if val > 0.50 else "#08306b"
                ax.text(
                    j,
                    i,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color=color,
                    fontweight="bold" if val > 0.4 else "normal",
                )

    fig.colorbar(im, ax=ax, shrink=0.82, label="Normalized Detection Proportion")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    gt_path = ROOT / "results" / "ground_truth.json"
    cache_path = ROOT / "results" / "ensemble_predictions_cache.pkl.gz"
    opt_hp_path = ROOT / "results" / "optimal_ensemble_hyperparameters.json"

    if not gt_path.exists() or not cache_path.exists():
        print("Missing required predictions cache or ground truth.")
        return

    print("Loading Ground Truth, Hyperparameters, and Prediction Cache (890 test images)...")
    with open(gt_path, encoding="utf-8") as f:
        gt = json.load(f)

    with open(opt_hp_path, encoding="utf-8") as f:
        opt_hp = json.load(f)

    with gzip.open(cache_path, "rb") as f:
        cache = pickle.load(f)

    # 1. Evaluate Single Models
    models_to_eval = [
        ("f_orig", "FBDNet-M (SOTA Proposed 42.42%)", "fbdnet_m_cm.png"),
        ("d_orig", "D-FINE-M Baseline (41.34%)", "dfine_m_cm.png"),
        ("y_orig", "YOLO26m-ROI-VFL (40.09%)", "yolo26m_roi_vfl_cm.png"),
    ]

    for key, title, fname in models_to_eval:
        print(f"Evaluating {title}...")
        preds = []
        for img_id, m_dict in cache.items():
            data = m_dict[key]
            for c_idx in range(5):
                boxes, scores = data[c_idx]
                for b, s in zip(boxes, scores, strict=False):
                    x1, y1, x2, y2 = b
                    preds.append(
                        {
                            "image_id": img_id,
                            "category_id": c_idx,
                            "bbox": [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)],
                            "score": float(s),
                        }
                    )

        matrix, _ = detection_confusion(
            gt["images"], gt["annotations"], preds, nc=5, confidence=0.25, iou=0.5
        )
        row_sums = matrix.sum(axis=1, keepdims=True)
        norm_cm = np.divide(
            matrix, row_sums, out=np.zeros_like(matrix, dtype=float), where=row_sums > 0
        )

        out_file = ASSETS_DIR / fname
        plot_cm(norm_cm, f"{title} Normalized Confusion Matrix (Test Split)", out_file)

    # 2. Evaluate Adaptive Heterogeneous Triad (Apex Master Ensemble 46.10%)
    print("Evaluating Adaptive Heterogeneous Triad (Apex Master Ensemble 46.10%)...")
    apex_preds = []
    for c_idx, c_name in enumerate(CLASSES):
        cfg = opt_hp[c_name]
        mode = cfg["mode"]
        iou_thr = cfg["iou_thr"]
        weights = cfg["weights"]
        coord_ratio = cfg["coord_ratio"]
        boost_3 = cfg.get("boost_3", 1.25)
        boost_2 = cfg.get("boost_2", 1.12)
        solo_decay = cfg.get("solo_decay", 0.95)
        conf_floor = cfg.get("conf_floor", 0.05)

        for img_id, s in cache.items():
            if mode == "3M_MS_TTA":
                b_list = [s["f_tta"][c_idx][0], s["d_tta"][c_idx][0], s["y_ms_tta"][c_idx][0]]
                s_list = [s["f_tta"][c_idx][1], s["d_tta"][c_idx][1], s["y_ms_tta"][c_idx][1]]
            elif mode == "2M_FBD_YOLO_MS":
                b_list = [s["f_orig"][c_idx][0], s["y_ms_orig"][c_idx][0]]
                s_list = [s["f_orig"][c_idx][1], s["y_ms_orig"][c_idx][1]]
            elif mode == "2M_FBD_YOLO":
                b_list = [s["f_orig"][c_idx][0], s["y_orig"][c_idx][0]]
                s_list = [s["f_orig"][c_idx][1], s["y_orig"][c_idx][1]]
            else:
                b_list = [s["f_orig"][c_idx][0], s["y_orig"][c_idx][0]]
                s_list = [s["f_orig"][c_idx][1], s["y_orig"][c_idx][1]]

            fb, fs = fast_wbf_engine(
                b_list,
                s_list,
                weights,
                iou_thr,
                coord_ratio,
                boost_3,
                boost_2,
                solo_decay,
                conf_floor,
            )
            for b, score in zip(fb, fs, strict=False):
                apex_preds.append(
                    {
                        "image_id": img_id,
                        "category_id": c_idx,
                        "bbox": [b[0], b[1], max(0.0, b[2] - b[0]), max(0.0, b[3] - b[1])],
                        "score": score,
                    }
                )

    matrix, _ = detection_confusion(
        gt["images"], gt["annotations"], apex_preds, nc=5, confidence=0.25, iou=0.5
    )
    row_sums = matrix.sum(axis=1, keepdims=True)
    norm_cm = np.divide(
        matrix, row_sums, out=np.zeros_like(matrix, dtype=float), where=row_sums > 0
    )

    out_file = ASSETS_DIR / "apex_ensemble_cm.png"
    plot_cm(
        norm_cm,
        "Adaptive Heterogeneous Triad (Apex Ensemble 46.10%) Confusion Matrix",
        out_file,
    )

    print("All confusion matrices generated and saved successfully!")


if __name__ == "__main__":
    main()
