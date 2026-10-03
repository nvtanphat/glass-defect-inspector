"""Repair D-FINE epoch counters after stage-2 EMA refreshes stale metadata."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--actual-epoch", type=int, required=True)
    args = parser.parse_args()

    state = torch.load(args.source, map_location="cpu", weights_only=False)
    if not isinstance(state, dict):
        raise TypeError("Expected a dictionary checkpoint")

    old_epoch = state.get("last_epoch")
    state["last_epoch"] = args.actual_epoch

    scheduler = state.get("lr_scheduler")
    if isinstance(scheduler, dict):
        scheduler["last_epoch"] = args.actual_epoch + 1
        if "_step_count" in scheduler:
            scheduler["_step_count"] = args.actual_epoch + 2

    args.destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.destination.with_suffix(args.destination.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(args.destination)
    print(
        f"Saved repaired checkpoint: {args.destination} "
        f"(last_epoch {old_epoch} -> {args.actual_epoch})"
    )


if __name__ == "__main__":
    main()
