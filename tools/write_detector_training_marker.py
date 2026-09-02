from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path


def epoch_from_checkpoint(path: Path) -> int | None:
    match = re.search(r"checkpoint(\d+)\.pth$", path.name)
    return int(match.group(1)) if match else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify detector training completion and write its evidence marker.")
    sub = parser.add_subparsers(dest="detector", required=True)
    rt = sub.add_parser("rtdetr")
    rt.add_argument("--checkpoint", type=Path, required=True)
    rt.add_argument("--config", type=Path, required=True)
    rt.add_argument("--requested-epochs", type=int, required=True)
    rt.add_argument("--resumed-from", type=Path)
    rt.add_argument("--out", type=Path, required=True)
    yo = sub.add_parser("yolo")
    yo.add_argument("--results-csv", type=Path, required=True)
    yo.add_argument("--last-checkpoint", type=Path, required=True)
    yo.add_argument("--requested-epochs", type=int, required=True)
    yo.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if args.detector == "rtdetr":
        if not args.checkpoint.is_file() or not args.config.is_file():
            raise FileNotFoundError("RT-DETR completion evidence is missing")
        checkpoint_epoch = epoch_from_checkpoint(args.checkpoint)
        if checkpoint_epoch is not None and checkpoint_epoch + 1 < args.requested_epochs:
            raise RuntimeError(f"checkpoint epoch {checkpoint_epoch} does not prove {args.requested_epochs} epochs")
        evidence = {
            "detector": "rtdetr",
            "completed": True,
            "requested_epochs": args.requested_epochs,
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_epoch": checkpoint_epoch,
            "config": str(args.config.resolve()),
            "resumed_from": str(args.resumed_from.resolve()) if args.resumed_from else None,
        }
        out = args.out
    else:
        if not args.results_csv.is_file() or not args.last_checkpoint.is_file():
            raise FileNotFoundError("YOLO completion evidence is missing")
        with args.results_csv.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) < args.requested_epochs:
            raise RuntimeError(f"results.csv has {len(rows)} rows, expected at least {args.requested_epochs}")
        evidence = {
            "detector": "yolo",
            "completed": True,
            "requested_epochs": args.requested_epochs,
            "results_rows": len(rows),
            "results_csv": str(args.results_csv.resolve()),
            "last_checkpoint": str(args.last_checkpoint.resolve()),
        }
        out = args.out
    evidence["written_at_utc"] = datetime.now(timezone.utc).isoformat()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
