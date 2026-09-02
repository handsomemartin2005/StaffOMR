from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


METRICS = ("iou", "boundary_f1", "leakage", "staff_leakage", "area_ratio")


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "instances": len(rows),
        **{metric: float(np.mean([row[metric] for row in rows])) if rows else 0.0 for metric in METRICS},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize a source-domain SAM2 class gate.")
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--sam-classes", default="beam")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    sam_classes = {item.strip() for item in args.sam_classes.split(",") if item.strip()}
    paired: defaultdict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for line in args.rows.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        paired[(row["page"], str(row["symbol_id"]))][row["source"]] = row
    baseline = []
    gated = []
    for variants in paired.values():
        box = variants["bbox_mask"]
        sam = variants["sam2_zero_shot"]
        baseline.append(box)
        gated.append(sam if sam["class"] in sam_classes else box)
    payload = {
        "sam_classes": sorted(sam_classes),
        "baseline": aggregate(baseline),
        "class_gated": aggregate(gated),
        "delta": {
            metric: aggregate(gated)[metric] - aggregate(baseline)[metric] for metric in METRICS
        },
        "protocol": "class gate fixed from MUSCIMA++ source-domain per-class mask diagnostics",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
