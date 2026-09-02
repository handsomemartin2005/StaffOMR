from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.paper_evidence_metrics import aggregate_counts


ORACLES = ("oracle_pitch", "oracle_duration", "oracle_count")


def oracle_counts_from_decomposition(row: Mapping[str, Any]) -> dict[str, dict[str, int]]:
    joint = row["joint"]
    pitch = row["pitch"]
    duration = row["duration"]
    predicted = int(joint["predicted"])
    gold = int(joint["gold"])
    return {
        "oracle_pitch": {
            "matches": int(duration["matches"]),
            "predicted": predicted,
            "gold": gold,
        },
        "oracle_duration": {
            "matches": int(pitch["matches"]),
            "predicted": predicted,
            "gold": gold,
        },
        "oracle_count": {
            "matches": int(joint["matches"]),
            "predicted": min(predicted, gold),
            "gold": gold,
        },
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build explicitly labeled oracle upper-bound diagnostics.")
    parser.add_argument("--dataset", choices=("debussy",), required=True)
    parser.add_argument("--input", type=Path, default=Path("outputs/paper_evidence_20260720/error_decomposition/per_page.jsonl"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    source_rows = _read_jsonl(args.input)
    rows: list[dict[str, Any]] = []
    for source in source_rows:
        oracles = oracle_counts_from_decomposition(source)
        rows.append(
            {
                "page_id": source["page_id"],
                "method": source["method"],
                "uses_target_annotations": True,
                **oracles,
            }
        )
    methods = sorted({row["method"] for row in rows})
    summary = {
        method: {
            oracle: aggregate_counts([row[oracle] for row in rows if row["method"] == method])
            for oracle in ORACLES
        }
        for method in methods
    }
    payload = {
        "dataset": "HugoSchtr/debussy-omr-system-lvl transfer24",
        "diagnostic_only": True,
        "uses_target_annotations": True,
        "warning": "Oracle diagnostic; uses target annotations. Do not report as zero-shot or deployable performance.",
        "definitions": {
            "oracle_pitch": "Best possible joint matches if pitch were corrected while predicted durations and event count stay fixed; equals duration-only overlap.",
            "oracle_duration": "Best possible joint matches if duration were corrected while predicted pitches and event count stay fixed; equals pitch-only overlap.",
            "oracle_count": "Remove excess predictions up to gold count without changing retained event identities or adding matches.",
        },
        "summary": summary,
        "source": str(args.input.resolve()),
    }
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_jsonl(out / "per_page.jsonl", rows)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
