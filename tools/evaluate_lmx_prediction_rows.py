from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate sample-keyed LMX predictions from a JSON rows file.")
    parser.add_argument("--pred-json", type=Path, required=True)
    parser.add_argument("--gt-root", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--olimpic-root", type=Path, default=Path(".local-tools/olimpic-icdar24"))
    args = parser.parse_args()
    sys.path.insert(0, str((args.olimpic_root / "zeus").resolve()))
    from ser_metric import ser_metric

    payload = json.loads(args.pred_json.read_text(encoding="utf-8"))
    gold, pred, rows = [], [], []
    for row in payload.get("rows", []):
        sample = str(row["sample"])
        gt_path = args.gt_root / f"{sample}.lmx.txt"
        if not gt_path.exists():
            continue
        gt = gt_path.read_text(encoding="utf-8").strip()
        hypothesis = str(row.get("prediction") or "").strip()
        gold.append(gt)
        pred.append(hypothesis)
        metric = ser_metric([gt], [hypothesis])
        rows.append({"sample": sample, **metric, "gt_tokens": len(gt.split()), "pred_tokens": len(hypothesis.split())})
    summary = ser_metric(gold, pred) if rows else {"SER": None, "SERnotuplets": None}
    result = {"samples": len(rows), "summary": summary, "rows": rows}
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"samples": len(rows), **summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
