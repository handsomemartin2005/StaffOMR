from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from compose_v2_1_marks import compose_marks


NOTEHEAD_CLASSES = {"filled_notehead", "open_notehead"}
ACCIDENTAL_CLASSES = {"sharp", "flat", "natural"}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def center(symbol: dict[str, Any]) -> tuple[float, float]:
    x0, y0, x1, y1 = [float(v) for v in symbol["bbox"]]
    return 0.5 * (x0 + x1), 0.5 * (y0 + y1)


def symbol_staff(symbol: dict[str, Any]) -> int | None:
    value = (symbol.get("attributes") or {}).get("staff")
    return int(value) if value is not None else None


def detector_class(symbol: dict[str, Any]) -> str:
    attrs = symbol.get("attributes") or {}
    return str(attrs.get("detector_class") or attrs.get("fine_class") or symbol["class"])


def symbol_space(symbol: dict[str, Any], staves_by_index: dict[int, dict[str, Any]]) -> float:
    attrs = symbol.get("attributes") or {}
    if attrs.get("staff_space") is not None:
        return float(attrs["staff_space"])
    staff = symbol_staff(symbol)
    if staff is not None and staff in staves_by_index:
        return float(staves_by_index[staff].get("space") or 12.0)
    return 12.0


def pitch_from_staff(symbol: dict[str, Any], staves_by_index: dict[int, dict[str, Any]]) -> dict[str, Any]:
    attrs = symbol.get("attributes") or {}
    cx, cy = center(symbol)
    if attrs.get("pitch_step") is not None:
        return {
            "pitch_step": int(round(float(attrs["pitch_step"]))),
            "pitch_y_error": float(attrs.get("pitch_y_error") or 0.0),
            "pitch_source": "symbol_attribute",
        }
    staff_index = symbol_staff(symbol)
    staff = staves_by_index.get(staff_index) if staff_index is not None else None
    if staff is None:
        return {"pitch_step": None, "pitch_y_error": None, "pitch_source": "missing_staff"}
    lines = [float(v) for v in staff.get("lines", [])]
    if len(lines) < 2:
        return {"pitch_step": None, "pitch_y_error": None, "pitch_source": "missing_staff_lines"}
    space = float(staff.get("space") or abs(lines[1] - lines[0]) or 12.0)
    bottom = max(lines)
    step = int(round((bottom - cy) / max(1.0, 0.5 * space)))
    pitch_y = bottom - step * 0.5 * space
    return {"pitch_step": step, "pitch_y_error": cy - pitch_y, "pitch_source": "staff_geometry"}


def relation_maps(relations: list[dict[str, Any]]) -> dict[str, dict[str, list[dict[str, Any]]]]:
    by_source: dict[str, dict[str, list[dict[str, Any]]]] = {}
    by_target: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for relation in relations:
        rel_type = str(relation.get("type"))
        source = str(relation.get("source"))
        by_source.setdefault(rel_type, {}).setdefault(source, []).append(relation)
        for target in relation.get("targets", []):
            by_target.setdefault(rel_type, {}).setdefault(str(target), []).append(relation)
    return {"by_source": by_source, "by_target": by_target}


def relation_targets(relation: dict[str, Any]) -> list[str]:
    return [str(item) for item in relation.get("targets", [])]


def best_stem_for_note(note_id: str, maps: dict[str, dict[str, list[dict[str, Any]]]]) -> tuple[str | None, float | None]:
    relations = maps["by_source"].get("notehead_stem_attachment", {}).get(note_id, [])
    candidates: list[tuple[float, str]] = []
    for relation in relations:
        score = float(relation.get("score") or 0.0)
        for target in relation_targets(relation):
            candidates.append((score, target))
    if not candidates:
        return None, None
    score, stem_id = max(candidates, key=lambda item: item[0])
    return stem_id, score


def beams_for_stem(stem_id: str | None, maps: dict[str, dict[str, list[dict[str, Any]]]]) -> list[dict[str, Any]]:
    if stem_id is None:
        return []
    return maps["by_target"].get("beam_stem_group", {}).get(stem_id, [])


def ledgers_for_note(note_id: str, maps: dict[str, dict[str, list[dict[str, Any]]]]) -> list[dict[str, Any]]:
    return maps["by_target"].get("ledger_line_notehead_attachment", {}).get(note_id, [])


def slurs_for_note(note_id: str, maps: dict[str, dict[str, list[dict[str, Any]]]]) -> list[dict[str, Any]]:
    return maps["by_target"].get("slur_tie_notehead_endpoints", {}).get(note_id, [])


def nearby_accidental(
    note: dict[str, Any],
    accidentals: list[dict[str, Any]],
    staves_by_index: dict[int, dict[str, Any]],
) -> dict[str, Any] | None:
    nx, ny = center(note)
    staff = symbol_staff(note)
    space = symbol_space(note, staves_by_index)
    candidates: list[tuple[float, dict[str, Any]]] = []
    for accidental in accidentals:
        if staff is not None and symbol_staff(accidental) != staff:
            continue
        ax, ay = center(accidental)
        dx = nx - ax
        dy = abs(ny - ay)
        if 0 <= dx <= 3.2 * space and dy <= 1.25 * space:
            candidates.append((math.hypot(dx, dy), accidental))
    if not candidates:
        return None
    return min(candidates, key=lambda item: item[0])[1]


def nearby_augmentation_dot(
    event_symbol: dict[str, Any],
    dots: list[dict[str, Any]],
    staves_by_index: dict[int, dict[str, Any]],
    used_dot_ids: set[str],
) -> dict[str, Any] | None:
    ex, ey = center(event_symbol)
    staff = symbol_staff(event_symbol)
    space = symbol_space(event_symbol, staves_by_index)
    candidates: list[tuple[float, dict[str, Any]]] = []
    for dot in dots:
        dot_id = str(dot.get("id"))
        if dot_id in used_dot_ids:
            continue
        if staff is not None and symbol_staff(dot) != staff:
            continue
        dx0, dy0, dx1, dy1 = [float(value) for value in dot["bbox"]]
        dx = 0.5 * (dx0 + dx1) - ex
        dy = abs(0.5 * (dy0 + dy1) - ey)
        if 0.05 * space <= dx <= 1.85 * space and dy <= 0.75 * space:
            candidates.append((math.hypot(dx, dy), dot))
    if not candidates:
        return None
    dot = min(candidates, key=lambda item: item[0])[1]
    used_dot_ids.add(str(dot.get("id")))
    return dot


def beam_symbol_count_hint(beam: dict[str, Any] | None) -> int:
    if beam is None:
        return 1
    attrs = beam.get("attributes") or {}
    for key in ("beam_count", "beam_level", "parallel_beams"):
        value = attrs.get(key)
        if isinstance(value, (int, float)) and value >= 1:
            return max(1, min(5, int(round(value))))
    height = float(beam["bbox"][3]) - float(beam["bbox"][1])
    space = float(attrs.get("staff_space") or 12.0)
    ratio = height / max(1.0, space)
    if ratio >= 3.4:
        return 4
    if ratio >= 2.7:
        return 3
    if ratio >= 2.15:
        return 2
    return 1


def beam_count_for_relations(
    beam_relations: list[dict[str, Any]],
    symbols_by_id: dict[str, dict[str, Any]],
) -> int:
    source_ids = list(dict.fromkeys(str(relation.get("source")) for relation in beam_relations if relation.get("source")))
    if not source_ids:
        return 0
    hinted = [beam_symbol_count_hint(symbols_by_id.get(source_id)) for source_id in source_ids]
    return max(len(source_ids), max(hinted, default=1))


def duration_from_beam_count(beam_count: int) -> str:
    return {
        1: "eighth",
        2: "16th",
        3: "32nd",
        4: "64th",
    }.get(min(max(beam_count, 1), 4), "eighth_or_shorter")


def duration_hint(note: dict[str, Any], stem_id: str | None, beam_count: int) -> str:
    note_class = note["class"]
    if note_class == "open_notehead":
        return "half" if stem_id else "whole"
    if note_class == "filled_notehead":
        if beam_count:
            return duration_from_beam_count(beam_count)
        if stem_id:
            return "quarter"
        return "notehead_only_unknown"
    return "unknown"


def rest_duration_hint(rest: dict[str, Any]) -> str:
    fine = detector_class(rest)
    if fine.startswith("rest_"):
        suffix = fine.replace("rest_", "")
        return {
            "8th": "eighth",
            "16th": "16th",
            "32nd": "32nd",
            "64th": "64th",
            "128th": "64th",
            "quarter": "quarter",
            "half": "half",
            "whole": "whole",
            "double_whole": "whole",
        }.get(suffix, suffix)
    kind = str((rest.get("attributes") or {}).get("kind") or "")
    if "eighth" in kind:
        return "eighth_or_quarter"
    if "half" in kind:
        return "half"
    if "whole" in kind:
        return "whole"
    return "unknown"


def event_sort_key(item: dict[str, Any]) -> tuple[int, float, float, str]:
    staff = item.get("staff")
    x, y = item.get("center", [0.0, 0.0])
    return (9999 if staff is None else int(staff), float(x), float(y), str(item.get("id")))


def build_note_objects(payload: dict[str, Any]) -> dict[str, Any]:
    symbols = payload.get("symbols", [])
    symbols_by_id = {str(symbol["id"]): symbol for symbol in symbols}
    staves_by_index = {int(staff["index"]): staff for staff in payload.get("staves", []) if staff.get("index") is not None}
    maps = relation_maps(payload.get("relations", []))
    noteheads = [symbol for symbol in symbols if symbol.get("class") in NOTEHEAD_CLASSES]
    rests = [symbol for symbol in symbols if symbol.get("class") == "rest"]
    accidentals = [symbol for symbol in symbols if symbol.get("class") in ACCIDENTAL_CLASSES]
    augmentation_dots = [symbol for symbol in symbols if symbol.get("class") == "augmentation_dot"]
    used_dot_ids: set[str] = set()

    notes: list[dict[str, Any]] = []
    for index, notehead in enumerate(sorted(noteheads, key=event_sort_key), start=1):
        note_id = str(notehead["id"])
        stem_id, stem_score = best_stem_for_note(note_id, maps)
        beam_relations = beams_for_stem(stem_id, maps)
        beam_ids = [str(relation.get("source")) for relation in beam_relations]
        beam_count = beam_count_for_relations(beam_relations, symbols_by_id)
        ledger_relations = ledgers_for_note(note_id, maps)
        ledger_ids = [str(relation.get("source")) for relation in ledger_relations]
        slur_relations = slurs_for_note(note_id, maps)
        slur_ids = [str(relation.get("source")) for relation in slur_relations]
        accidental = nearby_accidental(notehead, accidentals, staves_by_index)
        dot = nearby_augmentation_dot(notehead, augmentation_dots, staves_by_index, used_dot_ids)
        cx, cy = center(notehead)
        pitch = pitch_from_staff(notehead, staves_by_index)
        confidence_parts = [float(notehead.get("confidence") or 0.0)]
        if stem_score is not None:
            confidence_parts.append(stem_score)
        confidence = sum(confidence_parts) / max(1, len(confidence_parts))
        notes.append(
            {
                "id": f"note_{index:05d}",
                "type": "note",
                "staff": symbol_staff(notehead),
                "center": [cx, cy],
                "notehead_id": note_id,
                "notehead_class": notehead["class"],
                "stem_id": stem_id,
                "beam_ids": list(dict.fromkeys(beam_ids)),
                "beam_count": beam_count,
                "ledger_line_ids": list(dict.fromkeys(ledger_ids)),
                "accidental_id": str(accidental["id"]) if accidental else None,
                "accidental": accidental["class"] if accidental else None,
                "augmentation_dot_id": str(dot["id"]) if dot else None,
                "dot_count": 1 if dot else 0,
                "slur_or_tie_ids": list(dict.fromkeys(slur_ids)),
                **pitch,
                "duration_hint": duration_hint(notehead, stem_id, beam_count),
                "confidence": confidence,
                "source_symbol_ids": [
                    item
                    for item in [
                        note_id,
                        stem_id,
                        *(list(dict.fromkeys(beam_ids))),
                        *(list(dict.fromkeys(ledger_ids))),
                        str(accidental["id"]) if accidental else None,
                        str(dot["id"]) if dot else None,
                    ]
                    if item is not None
                ],
            }
        )

    rest_events: list[dict[str, Any]] = []
    for index, rest in enumerate(sorted(rests, key=event_sort_key), start=1):
        cx, cy = center(rest)
        dot = nearby_augmentation_dot(rest, augmentation_dots, staves_by_index, used_dot_ids)
        rest_events.append(
            {
                "id": f"rest_{index:05d}",
                "type": "rest",
                "staff": symbol_staff(rest),
                "center": [cx, cy],
                "rest_id": str(rest["id"]),
                "augmentation_dot_id": str(dot["id"]) if dot else None,
                "dot_count": 1 if dot else 0,
                "duration_hint": rest_duration_hint(rest),
                "confidence": float(rest.get("confidence") or 0.0),
                "source_symbol_ids": [item for item in [str(rest["id"]), str(dot["id"]) if dot else None] if item is not None],
            }
        )

    mark_payload = compose_marks(symbols, payload.get("staves", []), include_singletons=True)
    marks = mark_payload["marks"]

    events = sorted([*notes, *rest_events, *marks], key=event_sort_key)
    detached_counts = {
        "notes_without_stem": sum(1 for note in notes if note["stem_id"] is None),
        "notes_with_stem": sum(1 for note in notes if note["stem_id"] is not None),
        "notes_with_beam": sum(1 for note in notes if note["beam_ids"]),
        "notes_with_accidental": sum(1 for note in notes if note["accidental_id"] is not None),
        "notes_with_dot": sum(1 for note in notes if note["dot_count"]),
        "notes_with_ledger": sum(1 for note in notes if note["ledger_line_ids"]),
        "rests": len(rest_events),
        "rests_with_dot": sum(1 for rest in rest_events if rest["dot_count"]),
        "marks": len(marks),
        "composite_marks": mark_payload["summary"]["composite_marks"],
        "dynamic_composites": mark_payload["summary"]["dynamic_composites"],
        "pedal_composites": mark_payload["summary"]["pedal_composites"],
    }
    by_duration: dict[str, int] = {}
    for event in events:
        if event.get("type") not in {"note", "rest"}:
            continue
        key = str(event.get("duration_hint") or "unknown")
        by_duration[key] = by_duration.get(key, 0) + 1

    return {
        "version": "v2.1_note_objects",
        "input": payload.get("input"),
        "source_shapes_json": payload.get("source_shapes_json"),
        "stack": {
            **payload.get("stack", {}),
            "note_assembly": "geometry_relation_note_objects_v2_1",
        },
        "staves": payload.get("staves", []),
        "notes": notes,
        "rests": rest_events,
        "marks": marks,
        "events": events,
        "summary": {
            "symbols": len(symbols),
            "relations": len(payload.get("relations", [])),
            "notes": len(notes),
            "rests": len(rest_events),
            "marks": len(marks),
            "events": len(events),
            **detached_counts,
            "duration_hints": dict(sorted(by_duration.items())),
        },
        "definitions": {
            "note": "One notehead plus optional linked stem, beams, ledger lines, accidental, and slur/tie references.",
            "mark": "A dynamic, hairpin, or pedal mark. Adjacent detector primitives such as p+p, s+f+z, or Ped.+* are grouped into composite marks.",
            "duration_hint": "Heuristic duration class inferred from notehead class and attached stem/beam count/rest class; not final MusicXML duration.",
            "pitch_step": "Staff-relative half-step index from lower staff line when available; this is not yet converted to absolute note name.",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Assemble V2.1 symbols and relations into note/rest event objects.")
    parser.add_argument("--shapes-json", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args()

    payload = read_json(args.shapes_json)
    result = build_note_objects({**payload, "source_shapes_json": str(args.shapes_json)})
    write_json(args.out_json, result)
    print(
        json.dumps(
            {
                "out_json": str(args.out_json),
                **result["summary"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

