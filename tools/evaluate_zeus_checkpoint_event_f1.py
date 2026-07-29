from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.build_transfer_error_decomposition import kern_events, zeus_events
from tools.paper_evidence_metrics import aggregate_counts, multiset_counts, musicxml_events


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a frozen Zeus LMX prediction with order-invariant event F1."
    )
    parser.add_argument("--dataset", choices=("debussy", "polish"), required=True)
    parser.add_argument("--predicted-lmx", type=Path, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-pages", type=int, default=24)
    args = parser.parse_args()

    prediction_path = args.predicted_lmx.resolve()
    prediction_lines = prediction_path.read_text(encoding="utf-8").splitlines()
    if len(prediction_lines) != args.expected_pages:
        raise ValueError(
            f"expected {args.expected_pages} prediction lines, got {len(prediction_lines)}"
        )

    data_root = ROOT / "data" / "ijcv_samples"
    olimpic_root = ROOT / ".local-tools" / "olimpic-icdar24"
    rows: list[dict[str, object]] = []
    parse_errors: list[dict[str, str]] = []
    for index, line in enumerate(prediction_lines):
        page_id = f"test_{index:04d}"
        if args.dataset == "debussy":
            gold = musicxml_events(
                data_root / "debussy-omr-transfer24" / f"{page_id}.musicxml"
            )
        else:
            gold = kern_events(
                data_root / "polish-scores-test24" / f"{page_id}.ekern.txt"
            )
        try:
            predicted = zeus_events(line, olimpic_root)
        except Exception as exc:  # Preserve failed predictions in the denominator.
            predicted = []
            parse_errors.append(
                {"page_id": page_id, "error": f"{type(exc).__name__}: {exc}"}
            )
        rows.append({"page_id": page_id, **multiset_counts(gold, predicted)})

    payload = {
        "dataset": args.dataset,
        "pages": args.expected_pages,
        "checkpoint": args.checkpoint,
        "metric": "order_invariant_duration_pitch_event_f1",
        "summary": aggregate_counts(rows),
        "parse_errors": parse_errors,
        "rows": rows,
        "provenance": {
            "predicted_lmx": str(prediction_path),
            "predicted_lmx_sha256": hashlib.sha256(prediction_path.read_bytes()).hexdigest(),
            "evaluator": str(Path(__file__).resolve()),
        },
        "caveat": "This content metric ignores reading order, measure structure, and voice assignment; it is not LMX SER.",
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2, ensure_ascii=False))
    print(f"parse_errors={len(parse_errors)}")
    print(f"output={output}")


if __name__ == "__main__":
    main()
