from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import run_debussy_abcd_ablation as debussy_eval


DATA = ROOT / "data" / "ijcv_samples" / "debussy-omr-transfer24"
ABLATION = ROOT / "outputs" / "debussy_abcd_ablation"
SELECTIVE = ROOT / "outputs" / "sam2_selective_demo"
DEFAULT_OUT = ROOT / "outputs" / "sam2_event_recovery_oracle"


def event_counter(path: Path) -> Counter[tuple[str, Any]]:
    return Counter(debussy_eval.note_events(path))


def f1(matches: int, predicted: int, gold: int) -> float:
    return 200.0 * matches / (predicted + gold) if predicted + gold else 0.0


def recovery_oracle(
    gold: Counter[Any], base: Counter[Any], candidate: Counter[Any]
) -> dict[str, int | float]:
    base_true = gold & base
    candidate_true = gold & candidate
    recovered = candidate_true - base_true
    damaged = base_true - candidate_true
    recovery_count = sum(recovered.values())
    base_matches = sum(base_true.values())
    base_predicted = sum(base.values())
    gold_count = sum(gold.values())
    oracle_matches = base_matches + recovery_count
    oracle_predicted = base_predicted + recovery_count
    return {
        "base_matches": base_matches,
        "candidate_matches": sum(candidate_true.values()),
        "recovered_events": recovery_count,
        "damaged_events": sum(damaged.values()),
        "base_predicted": base_predicted,
        "candidate_predicted": sum(candidate.values()),
        "gold": gold_count,
        "base_f1": f1(base_matches, base_predicted, gold_count),
        "candidate_f1": f1(sum(candidate_true.values()), sum(candidate.values()), gold_count),
        "conservative_oracle_matches": oracle_matches,
        "conservative_oracle_predicted": oracle_predicted,
        "conservative_oracle_f1": f1(oracle_matches, oracle_predicted, gold_count),
    }


def counter_union(counters: list[Counter[Any]]) -> Counter[Any]:
    result: Counter[Any] = Counter()
    for counter in counters:
        result |= counter
    return result


def aggregate(rows: list[dict[str, Any]], key: str) -> dict[str, int | float]:
    recovered = sum(int(row[key]["recovered_events"]) for row in rows)
    damaged = sum(int(row[key]["damaged_events"]) for row in rows)
    base_matches = sum(int(row[key]["base_matches"]) for row in rows)
    candidate_matches = sum(int(row[key]["candidate_matches"]) for row in rows)
    base_predicted = sum(int(row[key]["base_predicted"]) for row in rows)
    candidate_predicted = sum(int(row[key]["candidate_predicted"]) for row in rows)
    gold = sum(int(row[key]["gold"]) for row in rows)
    oracle_matches = base_matches + recovered
    oracle_predicted = base_predicted + recovered
    return {
        "base_matches": base_matches,
        "candidate_matches": candidate_matches,
        "recovered_events": recovered,
        "damaged_events": damaged,
        "base_predicted": base_predicted,
        "candidate_predicted": candidate_predicted,
        "gold": gold,
        "base_f1": f1(base_matches, base_predicted, gold),
        "candidate_f1": f1(candidate_matches, candidate_predicted, gold),
        "conservative_oracle_matches": oracle_matches,
        "conservative_oracle_predicted": oracle_predicted,
        "conservative_oracle_f1": f1(oracle_matches, oracle_predicted, gold),
    }


def write_markdown(summary: dict[str, Any], path: Path) -> None:
    labels = {
        "box_only_sam2": "Full box-prompt SAM2",
        "selective_sam2": "Quality-gated SAM2",
        "selective_contact": "Quality-gated + mask contact",
        "all_mask_union": "Union of all mask branches",
    }
    lines = [
        "# SAM2 event-recovery oracle",
        "",
        "The oracle keeps every prediction from the no-SAM2 box baseline and uses labels only to add correct events recovered by a mask branch. It is diagnostic and not deployable.",
        "",
        "| Candidate evidence | Candidate F1 | Recovered | Damaged | Conservative oracle F1 | Oracle delta over box |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key, label in labels.items():
        row = summary["aggregate"][key]
        delta = float(row["conservative_oracle_f1"]) - float(row["base_f1"])
        lines.append(
            f"| {label} | {row['candidate_f1']:.3f} | {row['recovered_events']} | {row['damaged_events']} | {row['conservative_oracle_f1']:.3f} | {delta:+.3f} |"
        )
    lines.extend(
        [
            "",
            f"Decision: **{summary['decision']}**.",
            "",
            "A positive oracle gap shows only that useful mask-derived events exist; an automatic pre-SAM2 gate still needs observable uncertainty features and a held-out evaluation.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure the label-aware event recovery ceiling of existing SAM2 branches.")
    parser.add_argument("--pages", default="test_0004,test_0005,test_0006")
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--selective-root", type=Path, default=SELECTIVE)
    args = parser.parse_args()
    pages = [item.strip() for item in args.pages.split(",") if item.strip()]
    selective_root = args.selective_root.resolve()
    rows: list[dict[str, Any]] = []

    for page in pages:
        paths = {
            "gold": DATA / f"{page}.musicxml",
            "base": ABLATION / "A1B1C0D1" / page / "semantics" / "score.musicxml",
            "box_only_sam2": ABLATION / "A1B1C1D1" / page / "semantics" / "score.musicxml",
            "selective_sam2": selective_root / "selective_sam2" / page / "semantics" / "score.musicxml",
            "selective_contact": selective_root / "selective_contact" / page / "semantics" / "score.musicxml",
        }
        for required in paths.values():
            if not required.exists():
                raise FileNotFoundError(required)
        counters = {key: event_counter(value) for key, value in paths.items()}
        row: dict[str, Any] = {"page": page}
        for key in ("box_only_sam2", "selective_sam2", "selective_contact"):
            row[key] = recovery_oracle(counters["gold"], counters["base"], counters[key])
        union = counter_union(
            [counters["box_only_sam2"], counters["selective_sam2"], counters["selective_contact"]]
        )
        row["all_mask_union"] = recovery_oracle(counters["gold"], counters["base"], union)
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False))

    keys = ("box_only_sam2", "selective_sam2", "selective_contact", "all_mask_union")
    aggregate_rows = {key: aggregate(rows, key) for key in keys}
    union_row = aggregate_rows["all_mask_union"]
    oracle_delta = float(union_row["conservative_oracle_f1"]) - float(union_row["base_f1"])
    summary = {
        "protocol": "label-aware conservative event-recovery oracle over frozen outputs",
        "pages": pages,
        "rows": rows,
        "aggregate": aggregate_rows,
        "best_oracle_delta_over_no_sam2_box": oracle_delta,
        "decision": "develop_pre_sam2_gate" if oracle_delta >= 1.0 else "do_not_invest_in_gate",
        "publication_status": "diagnostic_upper_bound_not_a_deployable_result",
    }
    out_root = args.out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(summary, out_root / "summary.md")
    print(json.dumps(aggregate_rows, ensure_ascii=False, indent=2))
    print({"oracle_delta": oracle_delta, "decision": summary["decision"]})


if __name__ == "__main__":
    main()
