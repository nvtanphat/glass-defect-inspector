"""Weighted Boxes Fusion (WBF) Ensemble CLI for Glass Defect Inspector.

Usage Examples:
---------------
1. Run single image ensemble:
   python scripts/run_wbf_ensemble.py --image data/raw/sample_1.jpg --output results/sample.json

2. Fuse precomputed offline model predictions:
   python scripts/run_wbf_ensemble.py --pred-files fbdnet.json yolo.json --gt-file gt.json

3. Custom models with per-class adaptive WBF:
   python scripts/run_wbf_ensemble.py --models weights/fbdnet.pth weights/yolo.pt --iou-thr 0.58
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add project root to sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.scale_error.ensemble.ensemble_pipeline import GlassDefectEnsemble  # noqa: E402
from src.scale_error.ensemble.wbf import DEFAULT_PER_CLASS_WEIGHTS  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Weighted Boxes Fusion (WBF) Ensemble for Glass Defect Inspection."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            str(ROOT / "weights" / "dfine_m_fbdnet_best.pth"),
            str(ROOT / "weights" / "yolo26m_roi_vfl_best.pt"),
        ],
        help="Paths to model weights checkpoints (default: FBDNet-M + YOLO26m-ROI-VFL).",
    )
    parser.add_argument(
        "--model-weights",
        nargs="+",
        type=float,
        default=None,
        help="Global model importance weights. If omitted, uses per-class adaptive weights.",
    )
    parser.add_argument(
        "--iou-thr",
        type=float,
        default=0.58,
        help="WBF IoU clustering threshold (default: 0.58; 0.55-0.62 for micro defects).",
    )
    parser.add_argument(
        "--skip-box-thr",
        type=float,
        default=0.05,
        help="Confidence pre-filtering threshold (default: 0.05).",
    )
    parser.add_argument(
        "--conf-thr",
        type=float,
        default=0.15,
        help="Detection confidence threshold (default: 0.15).",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to single image for live ensemble inference.",
    )
    parser.add_argument(
        "--pred-files",
        nargs="+",
        default=None,
        help="Paths to offline COCO prediction JSON files to fuse without running GPU inference.",
    )
    parser.add_argument(
        "--gt-file",
        type=str,
        default=str(ROOT / "reports" / "fbdnet_m" / "ground_truth.json"),
        help="Path to COCO ground truth JSON file for offline evaluation.",
    )
    parser.add_argument(
        "--hyperparams",
        type=str,
        default=str(ROOT / "results" / "optimal_ensemble_hyperparameters.json"),
        help="Path to optimal hyperparameters JSON file (if available).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(ROOT / "results" / "ensemble_results.json"),
        help="Path to save output predictions / report.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 80)
    print("GLASS DEFECT INSPECTOR: WEIGHTED BOXES FUSION (WBF) ENSEMBLE ENGINE")
    print("=" * 80)
    print(f"• Models configured: {[Path(m).name for m in args.models]}")
    print(f"• IoU Clustering Threshold: {args.iou_thr}")
    print(f"• Pre-filter Skip Box Threshold: {args.skip_box_thr}")
    print("• Per-Class Adaptive Weights Matrix:")
    for cls_name, w in DEFAULT_PER_CLASS_WEIGHTS.items():
        print(f"    - {cls_name:<15}: Model0 (Transformer)={w[0]}, Model1 (CNN)={w[1]}")
    print("=" * 80)

    opt_hp_path = Path(args.hyperparams)
    loaded_per_class_weights = None
    if opt_hp_path.is_file():
        try:
            with open(opt_hp_path, encoding="utf-8") as f:
                opt_data = json.load(f)
            loaded_per_class_weights = {k: v.get("weights", [1.0, 1.0]) for k, v in opt_data.items()}
            print(f"• Loaded optimal per-class hyperparameters from: {opt_hp_path}")
        except Exception as e:
            print(f"• Notice: could not parse {opt_hp_path}: {e}")

    ensemble = GlassDefectEnsemble(
        model_paths=args.models,
        weights=args.model_weights,
        per_class_weights=loaded_per_class_weights,
        iou_threshold=args.iou_thr,
        skip_box_threshold=args.skip_box_thr,
        root_dir=ROOT,
    )

    # Mode 1: Offline fusion of precomputed JSON files
    if args.pred_files:
        print("\n[MODE] Offline Predictions Fusion (Zero GPU Compute)...")
        _ = ensemble.evaluate_offline_from_json(
            prediction_json_files=args.pred_files,
            ground_truth_json=args.gt_file,
            output_report_file=args.output,
        )
        print("\n>> Offline Fusion Completed Successfully!")
        print(f">> Report saved to: {args.output}")
        return

    # Mode 2: Single Image live inference
    if args.image:
        print(f"\n[MODE] Single Image Live Ensemble Inference on: {args.image}...")
        img_path = Path(args.image)
        if not img_path.is_file():
            print(f"Error: Image file not found: {img_path}", file=sys.stderr)
            sys.exit(1)

        result = ensemble.predict_image(
            image=img_path,
            conf_threshold=args.conf_thr,
            use_per_class_weights=True,
        )

        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

        print("\n>> Ensemble Inference Results:")
        print(f"   • Total Defects Detected: {result['num_detections']}")
        for idx, det in enumerate(result["detections"]):
            c_name, c_conf, c_box = det["class_name"], det["confidence"], det["bbox"]
            print(f"     [{idx+1}] {c_name:<12} | Conf: {c_conf:.3f} | BBox: {c_box}")
        t_total = result["timing_ms"]["total_pipeline_ms"]
        t_wbf = result["timing_ms"]["fusion_wbf_ms"]
        print(f"   • Total Latency: {t_total} ms (WBF: {t_wbf} ms)")
        print(f">> Saved details to: {out_path}")
        return

    print("\n[NOTE] No --image or --pred-files specified.")
    print("The ensemble module is fully coded, compiled, and ready on disk.")
    print("When ready, run one of the following commands:")
    print("  python scripts/run_wbf_ensemble.py --image <path_to_glass_image.jpg>")
    print("  python scripts/run_wbf_ensemble.py --pred-files file1.json file2.json")


if __name__ == "__main__":
    main()
