from __future__ import annotations

import argparse
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


LETTERS = ["C", "D", "E", "F", "G", "A", "B"]
BASE_DIATONIC = {
    "treble": 4 * 7 + 2,  # E4, bottom line
    "bass": 2 * 7 + 4,  # G2, bottom line
}
DIVISIONS = 16
DURATION_UNITS = {
    "whole": 64,
    "half": 32,
    "quarter": 16,
    "eighth": 8,
    "16th": 4,
    "32nd": 2,
    "64th": 1,
    "eighth_or_shorter": 8,
    "eighth_or_quarter": 8,
    "notehead_only_unknown": 16,
    "unknown": 16,
}
MUSICXML_TYPES = {
    "whole": "whole",
    "half": "half",
    "quarter": "quarter",
    "eighth": "eighth",
    "16th": "16th",
    "32nd": "32nd",
    "64th": "64th",
    "eighth_or_shorter": "eighth",
    "eighth_or_quarter": "eighth",
    "notehead_only_unknown": "quarter",
    "unknown": "quarter",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def event_sort_key(event: dict[str, Any]) -> tuple[int, float, float, str]:
    staff = event.get("staff")
    x, y = event.get("center", [0.0, 0.0])
    return (9999 if staff is None else int(staff), float(x), float(y), str(event.get("id")))


def load_shapes(notes_payload: dict[str, Any], shapes_path: Path | None) -> dict[str, Any] | None:
    if shapes_path:
        return read_json(shapes_path)
    source = notes_payload.get("source_shapes_json")
    if source and Path(source).exists():
        return read_json(Path(source))
    return None


def barlines_by_staff(shapes_payload: dict[str, Any] | None) -> dict[int, list[float]]:
    result: dict[int, list[float]] = {}
    if not shapes_payload:
        return result
    for symbol in shapes_payload.get("symbols", []):
        if symbol.get("class") != "barline":
            continue
        attrs = symbol.get("attributes") or {}
        if attrs.get("staff") is None:
            continue
        x0, _, x1, _ = [float(v) for v in symbol["bbox"]]
        result.setdefault(int(attrs["staff"]), []).append(0.5 * (x0 + x1))
    return {staff: sorted(set(round(x, 1) for x in xs)) for staff, xs in result.items()}


def measure_index_for_x(x: float, staff: dict[str, Any], barlines: list[float]) -> int:
    left = float(staff.get("x0") or 0.0)
    usable = [bar for bar in barlines if bar > left + 2.0]
    return 1 + sum(1 for bar in usable if x >= bar)


def accidental_alter(accidental: str | None) -> int | None:
    return {"sharp": 1, "flat": -1, "natural": 0}.get(accidental or "")


def pitch_from_step(staff: dict[str, Any], pitch_step: int | None, accidental: str | None) -> dict[str, Any]:
    clef = str(staff.get("clef_type") or "treble")
    base = BASE_DIATONIC.get(clef, BASE_DIATONIC["treble"])
    if pitch_step is None:
        absolute = 4 * 7
        approximate = True
    else:
        absolute = base + int(pitch_step)
        approximate = False
    octave = math.floor(absolute / 7)
    letter = LETTERS[absolute % 7]
    result: dict[str, Any] = {"step": letter, "octave": octave, "approximate": approximate}
    alter = accidental_alter(accidental)
    if alter is not None:
        result["alter"] = alter
        result["accidental"] = accidental
    return result


def event_duration(event: dict[str, Any]) -> tuple[int, str]:
    hint = str(event.get("duration_hint") or "unknown")
    units = DURATION_UNITS.get(hint, DURATION_UNITS["unknown"])
    dot_count = max(0, int(event.get("dot_count") or 0))
    dotted_units = units
    addend = units
    for _ in range(dot_count):
        addend /= 2
        dotted_units += addend
    return max(1, int(round(dotted_units))), MUSICXML_TYPES.get(hint, "quarter")


def symbols_by_id(shapes_payload: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not shapes_payload:
        return {}
    return {str(symbol.get("id")): symbol for symbol in shapes_payload.get("symbols", []) if symbol.get("id") is not None}


def stem_orientation(event: dict[str, Any], symbol_map: dict[str, dict[str, Any]]) -> str | None:
    stem_id = event.get("stem_id")
    if not stem_id:
        return None
    stem = symbol_map.get(str(stem_id))
    if not stem:
        return None
    attrs = stem.get("attributes") or {}
    direction = str(attrs.get("direction") or "").lower()
    if direction in {"up", "down", "none"}:
        return direction

    bbox = stem.get("bbox")
    center = event.get("center")
    if bbox and center:
        stem_x = 0.5 * (float(bbox[0]) + float(bbox[2]))
        return "up" if stem_x >= float(center[0]) else "down"
    return None


def build_semantics(notes_payload: dict[str, Any], shapes_payload: dict[str, Any] | None) -> dict[str, Any]:
    staves = {int(staff["index"]): staff for staff in notes_payload.get("staves", []) if staff.get("index") is not None}
    barlines = barlines_by_staff(shapes_payload)
    symbol_map = symbols_by_id(shapes_payload)
    events = sorted(notes_payload.get("events", []), key=event_sort_key)
    score_parts: list[dict[str, Any]] = []
    for staff_index, staff in sorted(staves.items()):
        measures: dict[int, list[dict[str, Any]]] = {}
        for event in events:
            if event.get("staff") != staff_index:
                continue
            x, _ = event.get("center", [0.0, 0.0])
            measure = measure_index_for_x(float(x), staff, barlines.get(staff_index, []))
            semantic_event = {
                "id": event.get("id"),
                "type": event.get("type"),
                "x": float(x),
                "center": event.get("center"),
                "measure": measure,
                "duration_hint": event.get("duration_hint"),
                "dot_count": event.get("dot_count", 0),
                "duration_units": event_duration(event)[0],
                "source_symbol_ids": event.get("source_symbol_ids", []),
                "confidence": event.get("confidence"),
            }
            if event.get("type") == "note":
                semantic_event.update(
                    {
                        "pitch": pitch_from_step(staff, event.get("pitch_step"), event.get("accidental")),
                        "pitch_step": event.get("pitch_step"),
                        "notehead_id": event.get("notehead_id"),
                        "stem_id": event.get("stem_id"),
                        "stem_orientation": stem_orientation(event, symbol_map),
                        "beam_ids": event.get("beam_ids", []),
                        "beam_count": event.get("beam_count", 0),
                        "ledger_line_ids": event.get("ledger_line_ids", []),
                        "accidental": event.get("accidental"),
                        "slur_or_tie_ids": event.get("slur_or_tie_ids", []),
                    }
                )
            elif event.get("type") == "rest":
                semantic_event["rest_id"] = event.get("rest_id")
            elif event.get("type") == "mark":
                semantic_event.update(
                    {
                        "kind": event.get("kind"),
                        "text": event.get("text"),
                        "bbox": event.get("bbox"),
                        "composite": bool(event.get("composite")),
                    }
                )
            measures.setdefault(measure, []).append(semantic_event)
        score_parts.append(
            {
                "id": f"P{staff_index + 1}",
                "name": f"Staff {staff_index + 1}",
                "staff": staff_index,
                "clef_type": staff.get("clef_type"),
                "barlines": barlines.get(staff_index, []),
                "measures": [
                    {"number": number, "events": sorted(items, key=lambda item: (item["x"], item["id"] or ""))}
                    for number, items in sorted(measures.items())
                ],
            }
        )
    return {
        "version": "v2.1_semantic_score_draft",
        "input": notes_payload.get("input"),
        "source_notes_json": notes_payload.get("source_notes_json"),
        "source_shapes_json": notes_payload.get("source_shapes_json"),
        "divisions": DIVISIONS,
        "parts": score_parts,
        "summary": {
            "parts": len(score_parts),
            "measures": sum(len(part["measures"]) for part in score_parts),
            "events": sum(len(measure["events"]) for part in score_parts for measure in part["measures"]),
            "notes": sum(
                1
                for part in score_parts
                for measure in part["measures"]
                for event in measure["events"]
                if event["type"] == "note"
            ),
            "rests": sum(
                1
                for part in score_parts
                for measure in part["measures"]
                for event in measure["events"]
                if event["type"] == "rest"
            ),
            "marks": sum(
                1
                for part in score_parts
                for measure in part["measures"]
                for event in measure["events"]
                if event["type"] == "mark"
            ),
            "composite_marks": sum(
                1
                for part in score_parts
                for measure in part["measures"]
                for event in measure["events"]
                if event["type"] == "mark" and event.get("composite")
            ),
        },
        "limitations": [
            "Draft export: staff-relative pitch is converted to approximate diatonic pitch using detected clef.",
            "Draft export: rhythm is inferred from duration_hint and is not measure-balanced.",
            "Draft export: voices, tuplets, cross-staff beaming, repeats, and key/time signatures are not fully decoded.",
        ],
    }


def add_text(parent: ET.Element, tag: str, text: str | int | float) -> ET.Element:
    child = ET.SubElement(parent, tag)
    child.text = str(text)
    return child


def beam_statuses_for_events(events: list[dict[str, Any]]) -> dict[str, list[tuple[int, str]]]:
    statuses: dict[str, list[tuple[int, str]]] = {}
    beam_groups: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        if event.get("type") != "note":
            continue
        for beam_id in event.get("beam_ids", []) or []:
            beam_groups.setdefault(str(beam_id), []).append(event)

    for group in beam_groups.values():
        ordered = sorted(group, key=lambda item: (float(item.get("x") or 0.0), str(item.get("id") or "")))
        if not ordered:
            continue
        for index, event in enumerate(ordered):
            event_id = str(event.get("id"))
            beam_count = max(1, int(event.get("beam_count") or 1))
            if len(ordered) == 1:
                value = "forward hook"
            elif index == 0:
                value = "begin"
            elif index == len(ordered) - 1:
                value = "end"
            else:
                value = "continue"
            existing_levels = {number for number, _ in statuses.get(event_id, [])}
            for number in range(1, beam_count + 1):
                if number not in existing_levels:
                    statuses.setdefault(event_id, []).append((number, value))

    for event in events:
        if event.get("type") != "note" or not event.get("beam_count"):
            continue
        event_id = str(event.get("id"))
        existing_levels = {number for number, _ in statuses.get(event_id, [])}
        beam_count = max(1, int(event.get("beam_count") or 1))
        for number in range(1, beam_count + 1):
            if number not in existing_levels:
                statuses.setdefault(event_id, []).append((number, "continue"))

    return {event_id: sorted(values, key=lambda item: item[0]) for event_id, values in statuses.items()}


def slur_statuses_for_events(events: list[dict[str, Any]]) -> dict[str, list[tuple[int, str]]]:
    statuses: dict[str, list[tuple[int, str]]] = {}
    slur_groups: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        if event.get("type") != "note":
            continue
        for slur_id in event.get("slur_or_tie_ids", []) or []:
            slur_groups.setdefault(str(slur_id), []).append(event)

    ordered_groups = sorted(
        slur_groups.items(),
        key=lambda item: (
            min(float(event.get("x") or 0.0) for event in item[1]) if item[1] else 0.0,
            item[0],
        ),
    )
    for index, (_slur_id, group) in enumerate(ordered_groups, start=1):
        ordered = sorted(group, key=lambda item: (float(item.get("x") or 0.0), str(item.get("id") or "")))
        if len(ordered) < 2:
            continue
        number = 1 + ((index - 1) % 6)
        statuses.setdefault(str(ordered[0].get("id")), []).append((number, "start"))
        statuses.setdefault(str(ordered[-1].get("id")), []).append((number, "stop"))
    return {event_id: sorted(values, key=lambda item: item[0]) for event_id, values in statuses.items()}


def add_attributes(measure: ET.Element, clef_type: str | None) -> None:
    attrs = ET.SubElement(measure, "attributes")
    add_text(attrs, "divisions", DIVISIONS)
    key = ET.SubElement(attrs, "key")
    add_text(key, "fifths", 0)
    time = ET.SubElement(attrs, "time")
    add_text(time, "beats", 4)
    add_text(time, "beat-type", 4)
    clef = ET.SubElement(attrs, "clef")
    if clef_type == "bass":
        add_text(clef, "sign", "F")
        add_text(clef, "line", 4)
    else:
        add_text(clef, "sign", "G")
        add_text(clef, "line", 2)


def add_musicxml_note(measure: ET.Element, event: dict[str, Any], chord: bool = False) -> None:
    note = ET.SubElement(measure, "note")
    if chord:
        ET.SubElement(note, "chord")
    if event["type"] == "rest":
        ET.SubElement(note, "rest")
    else:
        pitch_data = event.get("pitch") or {"step": "C", "octave": 4}
        pitch = ET.SubElement(note, "pitch")
        add_text(pitch, "step", pitch_data.get("step", "C"))
        if pitch_data.get("alter") is not None:
            add_text(pitch, "alter", int(pitch_data["alter"]))
        add_text(pitch, "octave", pitch_data.get("octave", 4))
    duration_units = int(event.get("duration_units") or DIVISIONS)
    add_text(note, "duration", max(1, duration_units))
    add_text(note, "voice", 1)
    _, note_type = event_duration(event)
    add_text(note, "type", note_type)
    for _ in range(max(0, int(event.get("dot_count") or 0))):
        ET.SubElement(note, "dot")
    if event.get("type") == "note" and event.get("accidental"):
        add_text(note, "accidental", event["accidental"])
    if event.get("type") == "note" and event.get("stem_orientation"):
        add_text(note, "stem", event["stem_orientation"])
    for number, value in event.get("musicxml_beams", []) or []:
        beam = ET.SubElement(note, "beam", {"number": str(number)})
        beam.text = str(value)
    for number, value in event.get("musicxml_slurs", []) or []:
        notations = note.find("notations")
        if notations is None:
            notations = ET.SubElement(note, "notations")
        ET.SubElement(notations, "slur", {"number": str(number), "type": str(value)})
    if event.get("type") == "note" and event.get("beam_count"):
        notations = note.find("notations")
        if notations is None:
            notations = ET.SubElement(note, "notations")
        technical = ET.SubElement(notations, "technical")
        add_text(technical, "other-technical", f"beam_count={event.get('beam_count')}")


def add_musicxml_direction(measure: ET.Element, event: dict[str, Any]) -> None:
    kind = str(event.get("kind") or "mark")
    text = str(event.get("text") or kind)
    placement = "below" if kind in {"dynamic", "dynamic_hairpin", "pedal"} else "above"
    direction = ET.SubElement(measure, "direction", {"placement": placement})
    direction_type = ET.SubElement(direction, "direction-type")
    add_text(direction_type, "words", text)
    add_text(direction, "voice", 1)


def semantic_to_musicxml(semantics: dict[str, Any]) -> ET.ElementTree:
    root = ET.Element("score-partwise", {"version": "4.0"})
    work = ET.SubElement(root, "work")
    add_text(work, "work-title", "StaffOMR V2.1 draft export")
    identification = ET.SubElement(root, "identification")
    encoding = ET.SubElement(identification, "encoding")
    add_text(encoding, "software", "StaffOMR V2.1 semantic draft exporter")
    part_list = ET.SubElement(root, "part-list")
    for part in semantics.get("parts", []):
        score_part = ET.SubElement(part_list, "score-part", {"id": part["id"]})
        add_text(score_part, "part-name", part.get("name") or part["id"])

    for part in semantics.get("parts", []):
        part_el = ET.SubElement(root, "part", {"id": part["id"]})
        for measure_data in part.get("measures", []):
            measure_el = ET.SubElement(part_el, "measure", {"number": str(measure_data["number"])})
            if int(measure_data["number"]) == 1:
                add_attributes(measure_el, part.get("clef_type"))
            previous_x: float | None = None
            events = measure_data.get("events", [])
            beam_statuses = beam_statuses_for_events(events)
            slur_statuses = slur_statuses_for_events(events)
            for event in events:
                if event.get("type") == "mark":
                    add_musicxml_direction(measure_el, event)
                    continue
                event_for_xml = dict(event)
                event_for_xml["musicxml_beams"] = beam_statuses.get(str(event.get("id")), [])
                event_for_xml["musicxml_slurs"] = slur_statuses.get(str(event.get("id")), [])
                chord = False
                if event.get("type") == "note" and previous_x is not None and abs(float(event["x"]) - previous_x) <= 1.5:
                    chord = True
                add_musicxml_note(measure_el, event_for_xml, chord=chord)
                if event.get("type") == "note":
                    previous_x = float(event["x"])
    return ET.ElementTree(root)


def write_musicxml(path: Path, tree: ET.ElementTree) -> None:
    ET.indent(tree, space="  ")
    path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(path, encoding="utf-8", xml_declaration=True)


def write_linearized(path: Path, semantics: dict[str, Any]) -> None:
    lines: list[str] = []
    for part in semantics.get("parts", []):
        lines.append(f"part {part['id']} staff={part['staff']} clef={part.get('clef_type')}")
        for measure in part.get("measures", []):
            tokens: list[str] = []
            for event in measure.get("events", []):
                if event["type"] == "rest":
                    tokens.append(f"rest:{event.get('duration_hint')}")
                    continue
                if event["type"] == "mark":
                    kind = event.get("kind") or "mark"
                    text = event.get("text") or "?"
                    marker = "*" if event.get("composite") else ""
                    tokens.append(f"mark:{kind}:{text}{marker}")
                    continue
                pitch = event.get("pitch") or {}
                acc = event.get("accidental")
                suffix = f"{pitch.get('step', '?')}{pitch.get('octave', '?')}"
                if acc:
                    suffix += f"({acc})"
                tokens.append(f"{suffix}:{event.get('duration_hint')}")
            lines.append(f"  measure {measure['number']}: {' '.join(tokens)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export V2.1 note objects to draft semantic JSON and MusicXML.")
    parser.add_argument("--notes-json", type=Path, required=True)
    parser.add_argument("--shapes-json", type=Path)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-musicxml", type=Path)
    parser.add_argument("--out-linearized", type=Path)
    args = parser.parse_args()

    notes_payload = read_json(args.notes_json)
    notes_payload = {**notes_payload, "source_notes_json": str(args.notes_json)}
    shapes_payload = load_shapes(notes_payload, args.shapes_json)
    semantics = build_semantics(notes_payload, shapes_payload)
    write_json(args.out_json, semantics)
    if args.out_musicxml:
        write_musicxml(args.out_musicxml, semantic_to_musicxml(semantics))
    if args.out_linearized:
        write_linearized(args.out_linearized, semantics)
    print(
        json.dumps(
            {
                "out_json": str(args.out_json),
                "out_musicxml": str(args.out_musicxml) if args.out_musicxml else None,
                "out_linearized": str(args.out_linearized) if args.out_linearized else None,
                **semantics["summary"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

