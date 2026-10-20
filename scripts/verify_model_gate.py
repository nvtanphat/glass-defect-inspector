"""Model Quality and Integrity Gate Verification CLI.

Audits all registered models in the Glass Inspector Model Registry,
validates weights checksums, verifies checkpoint integrity, executes
smoke-test inference on real sample images, and measures operational latency.

Usage:
------
python scripts/verify_model_gate.py [--strict] [--output results/model_gate_report.json]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.glass_inspector.paths import SAMPLES_DIR  # noqa: E402
from src.glass_inspector.registry import ModelRegistry  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run automated verification gate on all registered defect models."
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any model in the registry is unavailable (default: requires default READY).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/model_gate_report.json",
        help="Path to save audit report JSON.",
    )
    return parser.parse_args()


def run_model_gate(strict: bool = False, output_path: str | Path | None = None) -> bool:
    print("=" * 80)
    print("  GLASS INSPECTOR MODEL INTEGRITY & PERFORMANCE GATE")
    print("=" * 80)

    registry = ModelRegistry()
    sample_img = SAMPLES_DIR / "sample_bottle_1.jpg"
    if not sample_img.is_file():
        print(f"Error: Required probe image missing: {sample_img}")
        return False

    print(f"Probe image: {sample_img.name} ({round(sample_img.stat().st_size / 1024, 1)} KB)\n")

    t_start = time.perf_counter()
    registry.probe(sample_img)
    total_probe_time = round((time.perf_counter() - t_start) * 1000, 1)

    all_passed = True
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "probe_time_ms": total_probe_time,
        "models": {},
    }

    print(f"{'Model ID':<22} | {'Status':<12} | {'Size':<10} | {'Latency':<12} | {'Result':<10}")
    print("-" * 75)

    for mid, rec in registry.records.items():
        status = rec.get("status", "UNAVAILABLE")
        size_str = f"{rec.get('size_mb', 0.0)} MB" if rec.get("size_mb") else "N/A"
        latency_str = f"{rec.get('probe_latency_ms', 'N/A')} ms"
        reason = rec.get("reason") or "OK"

        is_ok = status == "READY"
        if not is_ok:
            if strict or rec.get("is_default", False):
                all_passed = False

        status_disp = f"[OK] {status}" if is_ok else f"[!] {status}"
        print(f"{mid:<22} | {status_disp:<12} | {size_str:<10} | {latency_str:<12} | {reason[:20]}")

        report["models"][mid] = {
            "name": rec.get("name"),
            "status": status,
            "filename": rec.get("filename"),
            "size_mb": rec.get("size_mb"),
            "sha256": rec.get("sha256"),
            "latency_ms": rec.get("probe_latency_ms"),
            "reason": reason,
        }

    print("-" * 75)
    print(f"Total gate evaluation finished in {total_probe_time} ms.\n")

    if output_path:
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Report saved to: {out_file}")

    if all_passed:
        print("\n>>> MODEL GATE STATUS: PASSED (System is ready for production inspection)")
    else:
        print("\n>>> MODEL GATE STATUS: FAILED (One or more critical models failed verification)")

    return all_passed


def main() -> None:
    args = parse_args()
    success = run_model_gate(strict=args.strict, output_path=args.output)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
