from __future__ import annotations

import argparse
import json
import re
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.paper_evidence_metrics import (
    Event,
    aggregate_counts,
    decompose_event_errors,
    musicxml_element_events,
    musicxml_events,
)


METHODS = ("staff", "smt", "zeus")
PROJECTIONS = ("joint", "pitch", "duration")
_DURATION = re.compile(r"^(\d+)(\.*)")
_PITCH = re.compile(r"([A-Ga-g]+)([#n-]*)")


def _kern_atom(atom: str) -> Event | None:
    match = _DURATION.match(atom.strip())
    if match is None:
        return None
    denominator = match.group(1)
    base = Fraction(8 if denominator == "0" else 16 if set(denominator) == {"0"} else 4, int(denominator) or 1)
    duration = base * sum(Fraction(1, 2**index) for index in range(len(match.group(2)) + 1))
    suffix = atom[match.end() :]
    if "r" in suffix:
        return Event("R", duration)
    pitch = _PITCH.search(suffix)
    if pitch is None:
        return None
    letters = pitch.group(1)
    octave = 3 - (len(letters) - 1) if letters[0].isupper() else 4 + (len(letters) - 1)
    accidental = pitch.group(2).count("#") - pitch.group(2).count("-")
    return Event(f"{letters[0].upper()}{accidental}@{octave}", duration)


def kern_events(path: Path) -> list[Event]:
    events: list[Event] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        for field in line.split("\t"):
            if not field or field[0] in "*=!" or field == ".":
                continue
            for atom in field.split():
                event = _kern_atom(atom)
                if event is not None:
                    events.append(event)
    return events


def zeus_events(line: str, olimpic_root: Path) -> list[Event]:
    path = str(olimpic_root.resolve())
    if path not in sys.path:
        sys.path.insert(0, path)
    from app.linearization.Delinearizer import Delinearizer
    from app.symbolic.part_to_score import part_to_score

    delinearizer = Delinearizer()
    delinearizer.process_text(line)
    return musicxml_element_events(
        part_to_score(delinearizer.part_element).getroot(), implicit_rest=False
    )


def build_report(
    pages: Sequence[Mapping[str, Any]], expected_pages: int
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if len(pages) != expected_pages:
        raise ValueError(f"expected {expected_pages} pages, got {len(pages)}")
    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for page in pages:
        page_id = str(page["page_id"])
        if page_id in seen_ids:
            raise ValueError(f"duplicate page ID: {page_id}")
        seen_ids.add(page_id)
        predictions = page["predictions"]
        if set(predictions) != set(METHODS):
            raise ValueError(f"page {page_id} methods must be {METHODS}, got {sorted(predictions)}")
        gold = page["gold"]
        for method in METHODS:
            decomposition = decompose_event_errors(gold, predictions[method])
            rows.append({"page_id": page_id, "method": method, **decomposition})

    summary: dict[str, Any] = {}
    for method in METHODS:
        method_rows = [row for row in rows if row["method"] == method]
        summary[method] = {
            projection: aggregate_counts([row[projection] for row in method_rows])
            for projection in PROJECTIONS
        }
        summary[method]["pitch_correct_duration_wrong_opportunities"] = sum(
            int(row["pitch_correct_duration_wrong_opportunities"]) for row in method_rows
        )
        summary[method]["duration_correct_pitch_wrong_opportunities"] = sum(
            int(row["duration_correct_pitch_wrong_opportunities"]) for row in method_rows
        )
    return {"pages": expected_pages, "methods": list(METHODS), "summary": summary}, rows


def _debussy_pages(expected_pages: int) -> list[dict[str, Any]]:
    data = ROOT / "data" / "ijcv_samples" / "debussy-omr-transfer24"
    staff_root = ROOT / "outputs" / "ijcv_repro" / "debussy_staff_transfer24"
    staff_smoke = ROOT / "outputs" / "ijcv_repro" / "debussy_staff_smoke"
    smt_root = ROOT / "outputs" / "ijcv_repro" / "smt_fp_grandstaff_debussy_transfer24"
    zeus_path = ROOT / "outputs" / "ijcv_repro" / "zeus_grandstaff_on_debussy_transfer24" / "debussy-omr-transfer24-test.predicted.lmx"
    olimpic_root = ROOT / ".local-tools" / "olimpic-icdar24"
    zeus_lines = zeus_path.read_text(encoding="utf-8").splitlines()
    if len(zeus_lines) < expected_pages:
        raise ValueError(f"Zeus has {len(zeus_lines)} lines, expected {expected_pages}")
    pages: list[dict[str, Any]] = []
    for index in range(expected_pages):
        page_id = f"test_{index:04d}"
        staff_base = staff_smoke if index == 0 else staff_root
        pages.append(
            {
                "page_id": page_id,
                "gold": musicxml_events(data / f"{page_id}.musicxml"),
                "predictions": {
                    "staff": musicxml_events(staff_base / page_id / "semantics" / "score_v2_1.musicxml"),
                    "smt": kern_events(smt_root / f"{page_id}.bekern.txt"),
                    "zeus": zeus_events(zeus_lines[index], olimpic_root),
                },
            }
        )
    return pages


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build pitch/duration/joint transfer error decomposition.")
    parser.add_argument("--dataset", choices=("debussy",), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected-pages", type=int, default=24)
    args = parser.parse_args()

    pages = _debussy_pages(args.expected_pages)
    report, rows = build_report(pages, args.expected_pages)
    report.update(
        {
            "dataset": "HugoSchtr/debussy-omr-system-lvl transfer24",
            "protocol": "zero-shot frozen predictions; diagnostic marginal event projections",
            "metric_note": "Pitch-only and duration-only projections are diagnostics; joint Event-F1 is the accepted comparison metric.",
        }
    )
    accepted_path = ROOT / "outputs" / "ijcv_repro" / "order_invariant_event_f1" / "debussy_transfer24.json"
    accepted = json.loads(accepted_path.read_text(encoding="utf-8"))
    for method in METHODS:
        observed = float(report["summary"][method]["joint"]["f1"])
        expected = float(accepted["summary"][method]["f1"])
        if abs(observed - expected) > 1e-9:
            raise RuntimeError(f"Joint F1 reproduction failed for {method}: {observed} != {expected}")

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    report["sources"] = [str(accepted_path), str(ROOT / "data" / "ijcv_samples" / "debussy-omr-transfer24")]
    (out / "metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_jsonl(out / "per_page.jsonl", rows)
    print(json.dumps(report["summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
