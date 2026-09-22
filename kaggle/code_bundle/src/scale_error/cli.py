from __future__ import annotations

import argparse
from pathlib import Path

from scale_error.data.roboflow_io import download_dataset
from scale_error.data.split import split_original_valid
from scale_error.data.validate import validate_dataset
from scale_error.evaluation.evaluator import evaluate
from scale_error.inference.predictor import predict
from scale_error.training.trainer import train


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="scale-error")
    sub = p.add_subparsers(dest="cmd", required=True)

    prepare = sub.add_parser("prepare")
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--workspace", default="datacpn")
    prepare.add_argument("--project", default="scale_error_dc-orqpx")
    prepare.add_argument("--version", type=int, default=26)
    prepare.add_argument("--split-output", default="data/processed")

    tr = sub.add_parser("train")
    tr.add_argument("--data", required=True)
    tr.add_argument("--config", required=True)
    tr.add_argument("--project-dir", default="artifacts/runs")

    ev = sub.add_parser("evaluate")
    ev.add_argument("--weights", required=True)
    ev.add_argument("--data", required=True)
    ev.add_argument("--output", default="artifacts/evaluation/metrics.json")
    ev.add_argument("--nms-free", action="store_true")

    pr = sub.add_parser("predict")
    pr.add_argument("--weights", required=True)
    pr.add_argument("--source", required=True)
    pr.add_argument("--output", default="artifacts/predictions")
    pr.add_argument("--nms-free", action="store_true")

    return p


def main() -> None:
    args = build_parser().parse_args()

    if args.cmd == "prepare":
        dataset_root = download_dataset(
            output_dir=args.output,
            workspace=args.workspace,
            project=args.project,
            version=args.version,
        )
        validate_dataset(dataset_root, Path(args.split_output) / "dataset_report.json")
        _, runtime_yaml = split_original_valid(dataset_root, args.split_output)
        print(runtime_yaml)

    elif args.cmd == "train":
        train(args.data, args.config, args.project_dir)

    elif args.cmd == "evaluate":
        evaluate(
            args.weights,
            args.data,
            args.output,
            nms=not args.nms_free,
        )

    elif args.cmd == "predict":
        predict(
            args.weights,
            args.source,
            args.output,
            nms=not args.nms_free,
        )


if __name__ == "__main__":
    main()
