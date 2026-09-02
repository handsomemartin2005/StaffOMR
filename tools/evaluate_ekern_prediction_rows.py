from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluate_kern_text_proxy_metrics import chars_from_tokens, metric_block, normalized_tokens


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate sample-keyed normalized eKern proxy predictions.")
    parser.add_argument("--pred-json", type=Path, required=True)
    parser.add_argument("--gt-root", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.pred_json.read_text(encoding="utf-8"))
    gold: list[str] = []
    pred: list[str] = []
    rows = []
    for row in payload.get("rows", []):
        sample = str(row["sample"])
        gt_path = args.gt_root / f"{sample}.ekern.txt"
        if not gt_path.exists():
            continue
        gt_tokens = normalized_tokens(gt_path.read_text(encoding="utf-8"))
        pred_tokens = str(row.get("prediction") or "").split()
        gold.extend(gt_tokens)
        pred.extend(pred_tokens)
        rows.append({"sample": sample, "gt_tokens": len(gt_tokens), "pred_tokens": len(pred_tokens)})
    result = {
        "samples": len(rows),
        "SER_proxy": metric_block(gold, pred)["error_percent"] if rows else None,
        "CER_proxy": metric_block(chars_from_tokens(gold), chars_from_tokens(pred))["error_percent"] if rows else None,
        "gt_tokens": len(gold),
        "pred_tokens": len(pred),
        "rows": rows,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
