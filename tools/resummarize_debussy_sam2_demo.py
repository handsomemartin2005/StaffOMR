from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
TOOLS = ROOT / "tools"
for path in (TOOLS,):
    value = str(path)
    while value in sys.path:
        sys.path.remove(value)
    sys.path.insert(0, value)

from run_debussy_sam2_selective_demo import aggregate, write_json, write_markdown


KEYS = (
    "no_sam2",
    "box_only_sam2",
    "rebuilt_box",
    "selective_sam2",
    "selective_contact",
    "safe_residual",
    "safe_rerank",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute SAM2 demo attribution from per-page comparisons.")
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    old = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    pages = list(old["pages"])
    rows = [
        json.loads((root / page / "comparison.json").read_text(encoding="utf-8")) for page in pages
    ]
    aggregated = {key: aggregate(rows, key) for key in KEYS}
    stronger_frozen = max(aggregated["no_sam2"]["f1"], aggregated["box_only_sam2"]["f1"])
    best_mask = max(aggregated["selective_sam2"]["f1"], aggregated["selective_contact"]["f1"])
    rebuilt = float(aggregated["rebuilt_box"]["f1"])
    frozen_delta = float(best_mask) - float(stronger_frozen)
    rebuilt_delta = float(best_mask) - rebuilt
    safe_delta = max(
        float(aggregated["safe_residual"]["f1"]), float(aggregated["safe_rerank"]["f1"])
    ) - rebuilt
    summary = {
        **old,
        "rows": rows,
        "aggregate": aggregated,
        "best_selective_delta": frozen_delta,
        "best_mask_delta_over_rebuilt_box": rebuilt_delta,
        "safe_delta_over_rebuilt_box": safe_delta,
        "decision": (
            "run_full" if frozen_delta > 0 and rebuilt_delta > 0 else "stop_and_visualization_only"
        ),
    }
    write_json(root / "summary.json", summary)
    write_markdown(summary, root / "summary.md")
    print(
        json.dumps(
            {
                "best_selective_delta": frozen_delta,
                "best_mask_delta_over_rebuilt_box": rebuilt_delta,
                "safe_delta_over_rebuilt_box": safe_delta,
                "decision": summary["decision"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
