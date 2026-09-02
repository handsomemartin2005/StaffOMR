#!/usr/bin/env python
"""Prune V2.1 weak-rule over-recognition before semantic export."""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


NOTE_PRUNABLE_CLASSES = {"filled_notehead", "open_notehead"}
SYMBOL_PRUNABLE_CLASSES = {
    "filled_notehead",
    "open_notehead",
    "stem",
    "beam",
    "ledger_line",
    "slur_or_tie",
    "augmentation_dot",
}

PRESETS: dict[str, dict[str, float | int]] = {
    "recall": {
        "column_cap": 8,
        "column_xtol": 14,
        "no_stem_min_conf": 0.0,
        "open_no_stem_min_conf": 0.0,
        "isolated_unknown_min_conf": 0.0,
        "neighbor_xtol": 18,
    },
    "light": {
        "column_cap": 4,
        "column_xtol": 14,
        "no_stem_min_conf": 0.60,
        "open_no_stem_min_conf": 0.32,
        "isolated_unknown_min_conf": 0.55,
        "neighbor_xtol": 16,
    },
    "balanced": {
        "column_cap": 3,
        "column_xtol": 14,
        "no_stem_min_conf": 0.70,
        "open_no_stem_min_conf": 0.34,
        "isolated_unknown_min_conf": 0.60,
        "neighbor_xtol": 14,
    },
    "aggressive": {
        "column_cap": 3,
        "column_xtol": 14,
        "no_stem_min_conf": 0.75,
        "open_no_stem_min_conf": 0.34,
        "isolated_unknown_min_conf": 0.60,
        "neighbor_xtol": 14,
    },
    "visual_strict": {
        "column_cap": 2,
        "column_xtol": 14,
        "no_stem_min_conf": 0.70,
        "open_no_stem_min_conf": 0.34,
        "isolated_unknown_min_conf": 0.60,
        "neighbor_xtol": 14,
    },
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def note_x(note: dict[str, Any]) -> float:
    return float((note.get("center") or [0.0, 0.0])[0])


def note_y(note: dict[str, Any]) -> float:
    return float((note.get("center") or [0.0, 0.0])[1])


def note_score(note: dict[str, Any]) -> float:
    score = float(note.get("confidence") or 0.0)
    if note.get("stem_id"):
        score += 0.25
    if note.get("beam_ids"):
        score += 0.18
    if note.get("accidental_id"):
        score += 0.10
    if note.get("ledger_line_ids"):
        score += 0.03
    if note.get("duration_hint") == "notehead_only_unknown":
        score -= 0.22
    if note.get("notehead_class") == "filled_notehead":
        score += 0.02
    return score


def has_stemmed_neighbor(note: dict[str, Any], notes: list[dict[str, Any]], xtol: float) -> bool:
    x = note_x(note)
    staff = note.get("staff")
    for other in notes:
        if other is note or other.get("staff") != staff:
            continue
        if abs(note_x(other) - x) > xtol:
            continue
        if other.get("stem_id") or other.get("beam_ids"):
            return True
    return False


def initial_drop_reason(
    note: dict[str, Any],
    all_notes: list[dict[str, Any]],
    params: dict[str, float | int],
) -> str | None:
    confidence = float(note.get("confidence") or 0.0)
    has_rhythm_relation = bool(note.get("stem_id") or note.get("beam_ids"))
    is_unknown = note.get("duration_hint") == "notehead_only_unknown"
    if (
        is_unknown
        and not has_stemmed_neighbor(note, all_notes, float(params["neighbor_xtol"]))
        and confidence < float(params["isolated_unknown_min_conf"])
    ):
        return "detached_unknown_notehead_low_confidence"
    if (
        note.get("notehead_class") == "filled_notehead"
        and not has_rhythm_relation
        and confidence < float(params["no_stem_min_conf"])
    ):
        return "filled_notehead_without_rhythm_relation_low_confidence"
    if (
        note.get("notehead_class") == "open_notehead"
        and not has_rhythm_relation
        and confidence < float(params["open_no_stem_min_conf"])
    ):
        return "open_notehead_without_rhythm_relation_low_confidence"
    return None


def cluster_by_staff_x(notes: list[dict[str, Any]], xtol: float) -> list[list[dict[str, Any]]]:
    clusters: list[list[dict[str, Any]]] = []
    by_staff: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for note in notes:
        by_staff[note.get("staff")].append(note)
    for _staff, staff_notes in by_staff.items():
        current: list[dict[str, Any]] = []
        for note in sorted(staff_notes, key=lambda item: (note_x(item), note_y(item), str(item.get("id")))):
            if not current or abs(note_x(note) - note_x(current[-1])) <= xtol:
                current.append(note)
            else:
                clusters.append(current)
                current = [note]
        if current:
            clusters.append(current)
    return clusters


def prune_notes(
    notes_payload: dict[str, Any],
    params: dict[str, float | int],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    original_notes = list(notes_payload.get("notes", []))
    dropped: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []

    for note in original_notes:
        reason = initial_drop_reason(note, original_notes, params)
        if reason:
            item = copy.deepcopy(note)
            item["drop_reason"] = reason
            item["drop_score"] = note_score(note)
            dropped.append(item)
        else:
            candidates.append(note)

    cap = int(params["column_cap"])
    for cluster in cluster_by_staff_x(candidates, float(params["column_xtol"])):
        if len(cluster) <= cap:
            continue
        keep_ids = {
            str(note.get("id"))
            for note in sorted(cluster, key=lambda item: note_score(item), reverse=True)[:cap]
        }
        for note in cluster:
            if str(note.get("id")) not in keep_ids:
                item = copy.deepcopy(note)
                item["drop_reason"] = "staff_x_column_cap"
                item["drop_score"] = note_score(note)
                item["column_size"] = len(cluster)
                dropped.append(item)

    dropped_ids = {str(note.get("id")) for note in dropped}
    kept_notes = [
        copy.deepcopy(note)
        for note in original_notes
        if str(note.get("id")) not in dropped_ids
    ]
    kept_notes.sort(key=lambda item: (item.get("staff") if item.get("staff") is not None else 9999, note_x(item), note_y(item), str(item.get("id"))))

    pruned = copy.deepcopy(notes_payload)
    pruned["notes"] = kept_notes
    pruned["events"] = sorted(
        [*kept_notes, *pruned.get("rests", []), *pruned.get("marks", [])],
        key=lambda item: (item.get("staff") if item.get("staff") is not None else 9999, note_x(item), note_y(item), str(item.get("id"))),
    )
    old_summary = dict(pruned.get("summary") or {})
    duration_counts = Counter(str(note.get("duration_hint") or "unknown") for note in kept_notes)
    pruned["summary"] = {
        **old_summary,
        "notes_before_pruning": len(original_notes),
        "notes": len(kept_notes),
        "notes_pruned": len(dropped),
        "events": len(pruned["events"]),
        "notes_without_stem": sum(1 for note in kept_notes if note.get("stem_id") is None),
        "notes_with_stem": sum(1 for note in kept_notes if note.get("stem_id") is not None),
        "notes_with_beam": sum(1 for note in kept_notes if note.get("beam_ids")),
        "duration_hints": dict(sorted(duration_counts.items())),
    }
    pruned["overrecognition_pruning"] = {
        "version": "v2_1_overrecognition_prune",
        "params": params,
        "drop_reasons": dict(Counter(str(item["drop_reason"]) for item in dropped)),
    }
    return pruned, dropped


def used_symbol_ids(notes_payload: dict[str, Any]) -> set[str]:
    used: set[str] = set()
    for collection in ("notes", "rests", "marks"):
        for event in notes_payload.get(collection, []) or []:
            for symbol_id in event.get("source_symbol_ids", []) or []:
                if symbol_id is not None:
                    used.add(str(symbol_id))
            for key in (
                "notehead_id",
                "stem_id",
                "rest_id",
                "accidental_id",
                "augmentation_dot_id",
                "dynamic_id",
                "hairpin_id",
                "pedal_id",
            ):
                value = event.get(key)
                if value is not None:
                    used.add(str(value))
            for key in ("beam_ids", "ledger_line_ids", "slur_or_tie_ids"):
                for value in event.get(key, []) or []:
                    used.add(str(value))
    return used


def prune_shapes(
    shapes_payload: dict[str, Any],
    pruned_notes_payload: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    used = used_symbol_ids(pruned_notes_payload)
    kept_symbols: list[dict[str, Any]] = []
    dropped_symbols: list[dict[str, Any]] = []
    for symbol in shapes_payload.get("symbols", []) or []:
        symbol_id = str(symbol.get("id"))
        cls = str(symbol.get("class"))
        if cls in SYMBOL_PRUNABLE_CLASSES and symbol_id not in used:
            item = copy.deepcopy(symbol)
            item["drop_reason"] = "unused_after_overrecognition_note_pruning"
            dropped_symbols.append(item)
        else:
            kept_symbols.append(copy.deepcopy(symbol))

    kept_ids = {str(symbol.get("id")) for symbol in kept_symbols}
    kept_relations = []
    for relation in shapes_payload.get("relations", []) or []:
        source = str(relation.get("source"))
        targets = [str(item) for item in relation.get("targets", []) or []]
        if source in kept_ids and all(target in kept_ids for target in targets):
            kept_relations.append(copy.deepcopy(relation))

    pruned = copy.deepcopy(shapes_payload)
    pruned["symbols"] = kept_symbols
    pruned["relations"] = kept_relations
    pruned["dropped_symbols"] = [*pruned.get("dropped_symbols", []), *dropped_symbols]
    pruned["counts"] = dict(Counter(str(symbol.get("class")) for symbol in kept_symbols))
    pruned["overrecognition_pruning"] = {
        "version": "v2_1_overrecognition_prune",
        "symbols_before_pruning": len(shapes_payload.get("symbols", []) or []),
        "symbols_after_pruning": len(kept_symbols),
        "symbols_pruned": len(dropped_symbols),
        "relations_before_pruning": len(shapes_payload.get("relations", []) or []),
        "relations_after_pruning": len(kept_relations),
        "drop_reasons": dict(Counter(str(item["class"]) for item in dropped_symbols)),
    }
    return pruned, dropped_symbols


def main() -> None:
    parser = argparse.ArgumentParser(description="Prune V2.1 over-recognized note and symbol candidates.")
    parser.add_argument("--notes-json", type=Path, required=True)
    parser.add_argument("--shapes-json", type=Path, required=True)
    parser.add_argument("--out-notes-json", type=Path, required=True)
    parser.add_argument("--out-shapes-json", type=Path, required=True)
    parser.add_argument("--out-report", type=Path, required=True)
    parser.add_argument("--preset", choices=sorted(PRESETS), default="balanced")
    parser.add_argument("--column-cap", type=int)
    parser.add_argument("--column-xtol", type=float)
    parser.add_argument("--no-stem-min-conf", type=float)
    parser.add_argument("--open-no-stem-min-conf", type=float)
    parser.add_argument("--isolated-unknown-min-conf", type=float)
    parser.add_argument("--neighbor-xtol", type=float)
    args = parser.parse_args()

    params = dict(PRESETS[args.preset])
    overrides = {
        "column_cap": args.column_cap,
        "column_xtol": args.column_xtol,
        "no_stem_min_conf": args.no_stem_min_conf,
        "open_no_stem_min_conf": args.open_no_stem_min_conf,
        "isolated_unknown_min_conf": args.isolated_unknown_min_conf,
        "neighbor_xtol": args.neighbor_xtol,
    }
    for key, value in overrides.items():
        if value is not None:
            params[key] = value

    notes_payload = read_json(args.notes_json)
    shapes_payload = read_json(args.shapes_json)
    pruned_notes, dropped_notes = prune_notes(notes_payload, params)
    pruned_shapes, dropped_symbols = prune_shapes(shapes_payload, pruned_notes)
    write_json(args.out_notes_json, pruned_notes)
    write_json(args.out_shapes_json, pruned_shapes)

    report = {
        "preset": args.preset,
        "params": params,
        "inputs": {
            "notes_json": str(args.notes_json),
            "shapes_json": str(args.shapes_json),
        },
        "outputs": {
            "notes_json": str(args.out_notes_json),
            "shapes_json": str(args.out_shapes_json),
        },
        "notes": {
            "before": len(notes_payload.get("notes", []) or []),
            "after": len(pruned_notes.get("notes", []) or []),
            "pruned": len(dropped_notes),
            "drop_reasons": dict(Counter(str(item["drop_reason"]) for item in dropped_notes)),
        },
        "symbols": {
            "before": len(shapes_payload.get("symbols", []) or []),
            "after": len(pruned_shapes.get("symbols", []) or []),
            "pruned": len(dropped_symbols),
            "pruned_by_class": dict(Counter(str(item.get("class")) for item in dropped_symbols)),
            "counts_after": pruned_shapes.get("counts", {}),
        },
        "relations": {
            "before": len(shapes_payload.get("relations", []) or []),
            "after": len(pruned_shapes.get("relations", []) or []),
        },
    }
    write_json(args.out_report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
