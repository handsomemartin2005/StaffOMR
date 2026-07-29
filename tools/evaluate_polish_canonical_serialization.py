from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parent.parent
TOOLS = REPO / "tools"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from evaluate_kern_text_proxy_metrics import (  # noqa: E402
    legacy_semantic_to_pseudo_ekern,
    metric_block,
    normalized_tokens,
)
from tools.polish_canonical_serialization import semantic_to_canonical_ekern  # noqa: E402


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def bag_oracle(ref: list[str], hyp: list[str]) -> dict[str, Any]:
    matches = sum((Counter(ref) & Counter(hyp)).values())
    minimum_errors = max(len(ref) - matches, len(hyp) - matches)
    error_rate = minimum_errors / len(ref) if ref else float(bool(hyp))
    return {
        "matches": matches,
        "recall_percent": 100.0 * matches / len(ref) if ref else 0.0,
        "precision_percent": 100.0 * matches / len(hyp) if hyp else 0.0,
        "minimum_errors": minimum_errors,
        "SER_lower_bound": 100.0 * error_rate,
    }


def evaluate(source_root: Path, gt_root: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    corpus_ref: list[str] = []
    corpus_legacy: list[str] = []
    corpus_hyp: list[str] = []
    page_bag_rows: list[dict[str, Any]] = []
    for semantics_path in sorted(source_root.glob("*/semantics/semantics_v2_1.json")):
        sample = semantics_path.parent.parent.name
        gt_path = gt_root / f"{sample}.ekern.txt"
        if not gt_path.exists():
            continue
        ref = normalized_tokens(gt_path.read_text(encoding="utf-8"))
        semantics = read_json(semantics_path)
        legacy = normalized_tokens(legacy_semantic_to_pseudo_ekern(semantics))
        hyp = normalized_tokens(semantic_to_canonical_ekern(semantics))
        legacy_metrics = metric_block(ref, legacy)
        canonical_metrics = metric_block(ref, hyp)
        page_bag = bag_oracle(ref, hyp)
        page_bag_rows.append(page_bag)
        rows.append(
            {
                "sample": sample,
                "parts": len(semantics.get("parts") or []),
                "ref_tokens": len(ref),
                "pred_tokens": len(hyp),
                "legacy": legacy_metrics,
                "canonical": canonical_metrics,
                "SER_delta": canonical_metrics["error_percent"] - legacy_metrics["error_percent"],
                "bag_oracle": page_bag,
            }
        )
        corpus_ref.extend(ref)
        corpus_legacy.extend(legacy)
        corpus_hyp.extend(hyp)
    legacy_metrics = metric_block(corpus_ref, corpus_legacy)
    canonical_metrics = metric_block(corpus_ref, corpus_hyp)
    page_bag_errors = sum(row["minimum_errors"] for row in page_bag_rows)
    page_bag_matches = sum(row["matches"] for row in page_bag_rows)
    return {
        "protocol": "Polish eKern proxy with deterministic canonical reconstruction",
        "samples": len(rows),
        "rows": rows,
        "corpus": {
            "legacy": legacy_metrics,
            "canonical": canonical_metrics,
            "SER_delta": canonical_metrics["error_percent"] - legacy_metrics["error_percent"],
            "bag_oracle": {
                "page_constrained": {
                    "matches": page_bag_matches,
                    "ref_tokens": len(corpus_ref),
                    "pred_tokens": len(corpus_hyp),
                    "SER_lower_bound": 100.0 * page_bag_errors / len(corpus_ref),
                },
                "corpus_relaxed": bag_oracle(corpus_ref, corpus_hyp),
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate canonical Polish score serialization.")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--gt-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = evaluate(args.source_root, args.gt_root)
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
