"""Evaluate morphology-specific WBF on Val and Test sets."""

import gzip
import json
import pickle

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

CLASS_NAMES = ["DiVat", "DiVatLoiLom", "LoiChi", "LoiNhua", "LoiTray"]


def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    a1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    a2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def fast_wbf_engine(
    boxes_list,
    scores_list,
    weights,
    iou_thr,
    coord_ratio,
    boost_3=1.0,
    boost_2=1.0,
    solo_decay=1.0,
    conf_floor=0.01,
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
                s_avg = (m0[0][2] * weights[0] + m1[0][2] * weights[1] + m2[0][2] * weights[2]) / (
                    weights[0] + weights[1] + weights[2]
                )
                score = min(1.0, s_avg * boost_3)
            elif m0 and m1:
                cw_sum = coord_ratio[0] + coord_ratio[1]
                box = (
                    m0[0][1] * (coord_ratio[0] / cw_sum) + m1[0][1] * (coord_ratio[1] / cw_sum)
                ).tolist()
                s_avg = (m0[0][2] * weights[0] + m1[0][2] * weights[1]) / (weights[0] + weights[1])
                score = min(1.0, s_avg * boost_2)
            elif m0 and m2:
                cw_sum = coord_ratio[0] + coord_ratio[2]
                box = (
                    m0[0][1] * (coord_ratio[0] / cw_sum) + m2[0][1] * (coord_ratio[2] / cw_sum)
                ).tolist()
                s_avg = (m0[0][2] * weights[0] + m2[0][2] * weights[2]) / (weights[0] + weights[2])
                score = min(1.0, s_avg * boost_2)
            elif m1 and m2:
                cw_sum = coord_ratio[1] + coord_ratio[2]
                box = (
                    m1[0][1] * (coord_ratio[1] / cw_sum) + m2[0][1] * (coord_ratio[2] / cw_sum)
                ).tolist()
                s_avg = (m1[0][2] * weights[1] + m2[0][2] * weights[2]) / (weights[1] + weights[2])
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
                s_avg = (m0[0][2] * weights[0] + m1[0][2] * weights[1]) / (weights[0] + weights[1])
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


def generate_class_predictions(c_idx, cfg, raw_streams):
    mode = cfg["mode"]
    iou_thr = cfg["iou_thr"]
    weights = cfg["weights"]
    coord_ratio = cfg.get("coord_ratio")
    if coord_ratio is None:
        coord_ratio = [1.0 / len(weights)] * len(weights)
    boost_3 = cfg.get("boost_3", 1.0)
    boost_2 = cfg.get("boost_2", cfg.get("consensus_boost", 1.0))
    solo_decay = cfg.get("solo_decay", 1.0)
    conf_floor = cfg.get("conf_floor", 0.01)

    preds_c = []
    for img_id, s in raw_streams.items():
        if mode == "3M_MS_TTA":
            b_list = [s["f_tta"][c_idx][0], s["d_tta"][c_idx][0], s["y_ms_tta"][c_idx][0]]
            s_list = [s["f_tta"][c_idx][1], s["d_tta"][c_idx][1], s["y_ms_tta"][c_idx][1]]
        elif mode == "3M_TTA":
            b_list = [s["f_tta"][c_idx][0], s["d_tta"][c_idx][0], s["y_tta"][c_idx][0]]
            s_list = [s["f_tta"][c_idx][1], s["d_tta"][c_idx][1], s["y_tta"][c_idx][1]]
        elif mode == "2M_FBD_YOLO_MS":
            b_list = [s["f_orig"][c_idx][0], s["y_ms_orig"][c_idx][0]]
            s_list = [s["f_orig"][c_idx][1], s["y_ms_orig"][c_idx][1]]
        elif mode == "2M_FBD_YOLO":
            b_list = [s["f_orig"][c_idx][0], s["y_orig"][c_idx][0]]
            s_list = [s["f_orig"][c_idx][1], s["y_orig"][c_idx][1]]
        elif mode == "2M_DFINE_YOLO_MS":
            b_list = [s["d_orig"][c_idx][0], s["y_ms_orig"][c_idx][0]]
            s_list = [s["d_orig"][c_idx][1], s["y_ms_orig"][c_idx][1]]
        elif mode == "2M_FBD_DFINE":
            b_list = [s["f_tta"][c_idx][0], s["d_tta"][c_idx][0]]
            s_list = [s["f_tta"][c_idx][1], s["d_tta"][c_idx][1]]
        elif mode == "3M_ORIGINAL":
            b_list = [s["f_orig"][c_idx][0], s["d_orig"][c_idx][0], s["y_orig"][c_idx][0]]
            s_list = [s["f_orig"][c_idx][1], s["d_orig"][c_idx][1], s["y_orig"][c_idx][1]]
        else:
            raise ValueError(f"Unknown mode: {mode}")

        fb, fs = fast_wbf_engine(
            b_list, s_list, weights, iou_thr, coord_ratio, boost_3, boost_2, solo_decay, conf_floor
        )
        for b, score in zip(fb, fs, strict=False):
            preds_c.append(
                {
                    "image_id": img_id,
                    "category_id": c_idx,
                    "bbox": [b[0], b[1], max(0.0, b[2] - b[0]), max(0.0, b[3] - b[1])],
                    "score": score,
                }
            )
    return preds_c


def evaluate_all(gt_coco, all_preds):
    coco_gt = COCO()
    coco_gt.dataset = gt_coco
    coco_gt.createIndex()
    coco_dt = coco_gt.loadRes(all_preds) if all_preds else coco_gt.loadRes([])
    ev = COCOeval(coco_gt, coco_dt, "bbox")
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    stats = ev.stats
    mAP50_95 = float(stats[0] * 100)
    mAP50 = float(stats[1] * 100)
    mAP75 = float(stats[2] * 100)
    mAP_s = float(stats[3] * 100)

    per_class = {}
    for c_idx, c_name in enumerate(CLASS_NAMES):
        ev_c = COCOeval(coco_gt, coco_dt, "bbox")
        ev_c.params.catIds = [c_idx]
        ev_c.evaluate()
        ev_c.accumulate()
        ev_c.summarize()
        per_class[c_name] = round(float(ev_c.stats[0] * 100), 2)
    return {
        "mAP50_95": round(mAP50_95, 2),
        "mAP50": round(mAP50, 2),
        "mAP75": round(mAP75, 2),
        "mAP_s": round(mAP_s, 2),
        "per_class": per_class,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate morphology-specific WBF ensemble.")
    parser.add_argument(
        "--split", choices=["test", "val"], default="test", help="Evaluation split (default: test)"
    )
    parser.add_argument(
        "--config",
        default="results/optimal_ensemble_hyperparameters.json",
        help="Path to frozen hyperparameters JSON",
    )
    parser.add_argument(
        "--clean", action="store_true", help="Exclude near-duplicate / overlap images"
    )
    args = parser.parse_args()

    # Load frozen hyperparameters
    with open(args.config, encoding="utf-8") as f:
        ensemble_config = json.load(f)

    if args.split == "test":
        streams_path = "results/ensemble_predictions_cache.pkl.gz"
        gt_path = "results/ground_truth.json"
        with gzip.open(streams_path, "rb") as f:
            streams = pickle.load(f)
        with open(gt_path, encoding="utf-8") as f:
            gt = json.load(f)
    else:
        streams_path = "kaggle/output_wbf_routing_val/ensemble_outputs/val_streams.json.gz"
        gt_path = "kaggle/output_wbf_routing_val/ensemble_outputs/val_ground_truth.json"
        with gzip.open(streams_path, "rt", encoding="utf-8") as f:
            streams = {int(k): v for k, v in json.load(f)["streams"].items()}
        with open(gt_path, encoding="utf-8") as f:
            gt = json.load(f)

    print(f"Loaded {len(streams)} prediction streams for split: {args.split.upper()}")

    # Generate predictions
    all_preds = []
    for c_idx, c_name in enumerate(CLASS_NAMES):
        preds_c = generate_class_predictions(c_idx, ensemble_config[c_name], streams)
        all_preds.extend(preds_c)

    if args.clean and args.split == "test":
        near_dup_files = {
            "MyImage-20250723185242068_jpg.rf.9734aa09c104c66dbb57963ea7b91b12.jpg",
            "MyImage-20250730182600653_aug_0_6786_jpg.rf.8b60ad820652281c882dab0a093eb04e.jpg",
            "MyImage-20250730201039789_aug_0_1883_jpg.rf.ac6bc1ef2a5b0cc6f11731d32948e4b4.jpg",
            "MyImage-20250730201039789_aug_0_1883_jpg.rf.ed5f32c458f291c1cb34a62bb6033bc0.jpg",
            "MyImage-20250730204135919_jpg.rf.b6c2f23b539814877bbfd9184edd2e80.jpg",
        }
        dup_ids = {img["id"] for img in gt["images"] if img["file_name"] in near_dup_files}
        gt = {
            "images": [i for i in gt["images"] if i["id"] not in dup_ids],
            "annotations": [a for a in gt["annotations"] if a["image_id"] not in dup_ids],
            "categories": gt["categories"],
        }
        all_preds = [p for p in all_preds if p["image_id"] not in dup_ids]
        print(f"Applied clean filter: {len(gt['images'])} images remaining.")

    results = evaluate_all(gt, all_preds)
    print(f"Ensemble Evaluation Results ({args.split.upper()}):")
    print(json.dumps(results, indent=2))
