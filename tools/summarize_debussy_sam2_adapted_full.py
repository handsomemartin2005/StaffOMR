from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

from paper_evidence_metrics import aggregate_counts, paired_bootstrap_delta


DEFAULT_SUMMARY = ROOT / "outputs/sam2_adapted_beam_full24/summary.json"


def paired_summary(
    rows: list[dict[str, Any]],
    candidate: str,
    baseline: str,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    candidate_rows = [row[candidate] for row in rows]
    baseline_rows = [row[baseline] for row in rows]
    effect = paired_bootstrap_delta(candidate_rows, baseline_rows, samples=samples, seed=seed)
    wins = ties = losses = 0
    for candidate_row, baseline_row in zip(candidate_rows, baseline_rows):
        candidate_f1 = float(aggregate_counts([candidate_row])["f1"])
        baseline_f1 = float(aggregate_counts([baseline_row])["f1"])
        if candidate_f1 > baseline_f1:
            wins += 1
        elif candidate_f1 < baseline_f1:
            losses += 1
        else:
            ties += 1
    return {
        "candidate": candidate,
        "baseline": baseline,
        "effect": effect,
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "statistically_positive_95": float(effect["ci95_low"]) > 0.0,
    }


def write_markdown(payload: dict[str, Any], path: Path) -> None:
    aggregate = payload["aggregate"]
    comparisons = payload["comparisons"]
    lines = [
        "# Full-24 domain-adapted SAM2 result",
        "",
        "Frozen Debussy detector, balanced reconstruction, and Event-F1 evaluator. The adapted variant applies the writer-disjoint MUSCIMA++ beam decoder only to beam symbols; all other classes use box fallback.",
        "",
        "| Variant | Matches | Predicted | Gold | Event-F1 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    labels = {
        "no_sam2": "No SAM2",
        "zero_shot_sam2": "Zero-shot SAM2",
        "adapted_beam_sam2": "Area-trained beam SAM2",
        "adapted_relation_sam2": "Area-trained beam SAM2 + safe relation pruning",
    }
    for key, label in labels.items():
        item = aggregate[key]
        lines.append(
            f"| {label} | {item['matches']} | {item['predicted_events']} | "
            f"{item['gold_events']} | {item['f1']:.3f} |"
        )
    lines.extend(["", "| Comparison | Delta | 95% paired bootstrap CI | W/T/L |", "| --- | ---: | --- | ---: |"])
    for item in comparisons:
        effect = item["effect"]
        lines.append(
            f"| {labels[item['candidate']]} minus {labels[item['baseline']]} | "
            f"{effect['full_minus_ablated']:+.3f} | [{effect['ci95_low']:.3f}, {effect['ci95_high']:.3f}] | "
            f"{item['wins']}/{item['ties']}/{item['losses']} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "The adapted decoder has the best point estimate and increases matched events, but both principal 95% confidence intervals include zero. Report this as a positive full-set trend, not as a statistically significant or guaranteed gain. Safe relation pruning is retained only as an optional conservative gate because its aggregate Event-F1 is unchanged.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize full Debussy adapted-SAM2 attribution.")
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260722)
    args = parser.parse_args()
    source = json.loads(args.summary.read_text(encoding="utf-8"))
    comparisons = [
        paired_summary(source["rows"], "adapted_beam_sam2", "no_sam2", args.bootstrap_samples, args.seed),
        paired_summary(
            source["rows"], "adapted_beam_sam2", "zero_shot_sam2", args.bootstrap_samples, args.seed + 1
        ),
        paired_summary(
            source["rows"], "adapted_relation_sam2", "adapted_beam_sam2", args.bootstrap_samples, args.seed + 2
        ),
    ]
    payload = {
        "protocol": source["protocol"],
        "pages": source["pages"],
        "aggregate": source["aggregate"],
        "comparisons": comparisons,
        "decision": "positive_point_estimate_not_statistically_significant",
        "publication_status": "full24_result_with_paired_bootstrap",
    }
    out = args.summary.resolve().parent
    (out / "attribution.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(payload, out / "conclusion.md")
    print(json.dumps({"decision": payload["decision"], "comparisons": comparisons}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
