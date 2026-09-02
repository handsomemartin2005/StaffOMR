from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluate_kern_text_proxy_metrics import chars_from_tokens, data_lines, metric_block, normalized_tokens, semantic_to_pseudo_ekern


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate eKern proxy SER/CER over a directory of per-page semantics.")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--gt-root", type=Path, required=True)
    parser.add_argument("--prediction-relpath", default="semantics/semantics_v2_1.json")
    parser.add_argument("--gt-suffix", default=".ekern.txt")
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args()
    gt_tokens: list[str] = []
    pred_tokens: list[str] = []
    gt_lines: list[str] = []
    pred_lines: list[str] = []
    rows = []
    for page_dir in sorted(p for p in args.source_root.iterdir() if p.is_dir()):
        pred_path = page_dir / args.prediction_relpath
        gt_path = args.gt_root / f"{page_dir.name}{args.gt_suffix}"
        if not pred_path.exists() or not gt_path.exists():
            continue
        gt = gt_path.read_text(encoding="utf-8")
        pred = semantic_to_pseudo_ekern(json.loads(pred_path.read_text(encoding="utf-8")))
        gtok, ptok = normalized_tokens(gt), normalized_tokens(pred)
        glines, plines = data_lines(gt), data_lines(pred)
        gt_tokens.extend(gtok)
        pred_tokens.extend(ptok)
        gt_lines.extend(glines)
        pred_lines.extend(plines)
        rows.append({"sample": page_dir.name, "gt_tokens": len(gtok), "pred_tokens": len(ptok)})
    result = {
        "samples": len(rows),
        "SER_proxy": metric_block(gt_tokens, pred_tokens)["error_percent"],
        "CER_proxy": metric_block(chars_from_tokens(gt_tokens), chars_from_tokens(pred_tokens))["error_percent"],
        "LER_proxy": metric_block(gt_lines, pred_lines)["error_percent"],
        "gt_tokens": len(gt_tokens),
        "pred_tokens": len(pred_tokens),
        "rows": rows,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
