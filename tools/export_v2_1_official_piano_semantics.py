from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

from export_v2_1_semantics import (
    DIVISIONS,
    MUSICXML_TYPES,
    add_text,
    beam_statuses_for_events,
    event_duration,
    slur_statuses_for_events,
)


STEP_ORDER = {"C": 0, "D": 1, "E": 2, "F": 3, "G": 4, "A": 5, "B": 6}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def clef_sign_line(clef_type: str | None) -> tuple[str, int]:
    if clef_type == "bass":
        return "F", 4
    return "G", 2


def pitch_key(event: dict[str, Any]) -> tuple[int, int, str]:
    pitch = event.get("pitch") or {}
    step = str(pitch.get("step") or "C")
    try:
        octave = int(pitch.get("octave") or 4)
    except (TypeError, ValueError):
        octave = 4
    try:
        alter = int(pitch.get("alter") or 0)
    except (TypeError, ValueError):
        alter = 0
    return (octave * 7 + STEP_ORDER.get(step, 0), alter, str(event.get("id") or ""))


def sort_key(event: dict[str, Any]) -> tuple[float, float, str]:
    center = event.get("center") or [event.get("x") or 0.0, 0.0]
    return (float(event.get("x") or center[0] or 0.0), float(center[1] or 0.0), str(event.get("id") or ""))


def staff_voice(staff_number: int) -> str:
    return "1" if staff_number == 1 else str(1 + (staff_number - 1) * 4)


def keep_event(event: dict[str, Any], min_note_confidence: float) -> bool:
    if event.get("type") != "note":
        return True
    confidence = event.get("confidence")
    if confidence is None:
        return True
    try:
        return float(confidence) >= min_note_confidence
    except (TypeError, ValueError):
        return True


def make_attributes(parts: list[dict[str, Any]], include_key: bool, include_time: bool) -> ET.Element:
    attrs = ET.Element("attributes")
    add_text(attrs, "divisions", DIVISIONS)
    if include_key:
        key = ET.SubElement(attrs, "key")
        add_text(key, "fifths", 0)
    if include_time:
        time = ET.SubElement(attrs, "time")
        add_text(time, "beats", 4)
        add_text(time, "beat-type", 4)
    add_text(attrs, "staves", max(1, len(parts)))
    for staff_number, part in enumerate(parts, start=1):
        sign, line = clef_sign_line(part.get("clef_type"))
        clef = ET.SubElement(attrs, "clef", {"number": str(staff_number)})
        add_text(clef, "sign", sign)
        add_text(clef, "line", line)
    return attrs


def make_backup(duration_units: int) -> ET.Element:
    backup = ET.Element("backup")
    add_text(backup, "duration", max(1, int(duration_units)))
    return backup


def note_type(event: dict[str, Any]) -> str:
    if event.get("musicxml_type_override"):
        return str(event["musicxml_type_override"])
    hint = str(event.get("duration_hint") or "unknown")
    return MUSICXML_TYPES.get(hint, "quarter")


def add_musicxml_note(
    measure: ET.Element,
    event: dict[str, Any],
    *,
    chord: bool,
    staff_number: int,
    voice: str,
    emit_staff: bool = True,
) -> None:
    attrs: dict[str, str] = {}
    if event.get("x") is not None:
        attrs["default-x"] = f"{float(event['x']):.2f}"
    center = event.get("center") or []
    if len(center) >= 2:
        attrs["default-y"] = f"{float(center[1]):.2f}"
    note = ET.SubElement(measure, "note", attrs)
    if chord and event.get("type") == "note":
        ET.SubElement(note, "chord")
    if event.get("type") == "rest":
        ET.SubElement(note, "rest")
    else:
        pitch_data = event.get("pitch") or {"step": "C", "octave": 4}
        pitch = ET.SubElement(note, "pitch")
        add_text(pitch, "step", pitch_data.get("step", "C"))
        if pitch_data.get("alter") is not None:
            add_text(pitch, "alter", int(pitch_data["alter"]))
        add_text(pitch, "octave", pitch_data.get("octave", 4))
    duration_units = int(event.get("duration_units") or event_duration(event)[0])
    add_text(note, "duration", max(1, duration_units))
    for _number, value in event.get("musicxml_ties", []) or []:
        ET.SubElement(note, "tie", {"type": str(value)})
    add_text(note, "voice", voice)
    add_text(note, "type", note_type(event))
    for _ in range(max(0, int(event.get("dot_count") or 0))):
        ET.SubElement(note, "dot")
    if event.get("type") == "note" and event.get("accidental"):
        add_text(note, "accidental", event["accidental"])
    if event.get("type") == "note" and event.get("stem_orientation"):
        add_text(note, "stem", event["stem_orientation"])
    if emit_staff:
        add_text(note, "staff", staff_number)
    for number, value in event.get("musicxml_beams", []) or []:
        beam = ET.SubElement(note, "beam", {"number": str(number)})
        beam.text = str(value)
    for number, value in event.get("musicxml_slurs", []) or []:
        notations = note.find("notations")
        if notations is None:
            notations = ET.SubElement(note, "notations")
        ET.SubElement(notations, "slur", {"number": str(number), "type": str(value)})
    for number, value in event.get("musicxml_ties", []) or []:
        notations = note.find("notations")
        if notations is None:
            notations = ET.SubElement(note, "notations")
        ET.SubElement(notations, "tied", {"number": str(number), "type": str(value)})
    if event.get("type") == "note" and event.get("beam_count") and not event.get("musicxml_beams"):
        notations = note.find("notations")
        if notations is None:
            notations = ET.SubElement(note, "notations")
        technical = ET.SubElement(notations, "technical")
        add_text(technical, "other-technical", f"beam_count={event.get('beam_count')}")


def add_musicxml_direction(measure: ET.Element, event: dict[str, Any], voice: str, emit_staff: bool = True) -> None:
    kind = str(event.get("kind") or "mark")
    text = str(event.get("text") or kind)
    placement = "below" if kind in {"dynamic", "dynamic_hairpin", "pedal"} else "above"
    direction = ET.SubElement(measure, "direction", {"placement": placement})
    direction_type = ET.SubElement(direction, "direction-type")
    add_text(direction_type, "words", text)
    add_text(direction, "voice", voice)
    if emit_staff:
        add_text(direction, "staff", 1)


def onset_groups(events: list[dict[str, Any]], x_tolerance: float) -> list[list[dict[str, Any]]]:
    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_x: float | None = None
    for event in sorted(events, key=sort_key):
        if event.get("type") == "mark":
            groups.append([event])
            current = []
            current_x = None
            continue
        if event.get("type") == "rest":
            groups.append([event])
            current = []
            current_x = None
            continue
        x = float(event.get("x") or 0.0)
        if current and current_x is not None and abs(x - current_x) <= x_tolerance:
            current.append(event)
            current_x = sum(float(item.get("x") or 0.0) for item in current) / len(current)
            continue
        current = [event]
        current_x = x
        groups.append(current)
    return groups


def chord_profile(event: dict[str, Any]) -> tuple[int, str, int]:
    return (
        int(event.get("duration_units") or event_duration(event)[0]),
        note_type(event),
        max(0, int(event.get("dot_count") or 0)),
    )


def normalize_chord_group(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(events) <= 1:
        return [dict(event) for event in events]
    profile = Counter(chord_profile(event) for event in events).most_common(1)[0][0]
    duration_units, type_text, dot_count = profile
    result = []
    for event in events:
        item = dict(event)
        item["duration_units"] = duration_units
        item["musicxml_type_override"] = type_text
        item["dot_count"] = dot_count
        result.append(item)
    return result


def pitch_identity(event: dict[str, Any]) -> tuple[str, int, int] | None:
    pitch = event.get("pitch") or {}
    if not pitch:
        return None
    return (
        str(pitch.get("step") or "C"),
        int(pitch.get("octave") or 4),
        int(pitch.get("alter") or 0),
    )


def split_slur_tie_statuses(
    events: list[dict[str, Any]],
) -> tuple[dict[str, list[tuple[int, str]]], dict[str, list[tuple[int, str]]]]:
    slurs: dict[str, list[tuple[int, str]]] = {}
    ties: dict[str, list[tuple[int, str]]] = {}
    groups: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        if event.get("type") != "note":
            continue
        for curve_id in event.get("slur_or_tie_ids", []) or []:
            groups.setdefault(str(curve_id), []).append(event)
    ordered_groups = sorted(
        groups.items(),
        key=lambda item: (
            min(float(event.get("x") or 0.0) for event in item[1]) if item[1] else 0.0,
            item[0],
        ),
    )
    for index, (_curve_id, group) in enumerate(ordered_groups, start=1):
        ordered = sorted(group, key=lambda item: (float(item.get("x") or 0.0), str(item.get("id") or "")))
        if len(ordered) < 2:
            continue
        first, last = ordered[0], ordered[-1]
        number = 1 + ((index - 1) % 6)
        target = ties if pitch_identity(first) == pitch_identity(last) else slurs
        target.setdefault(str(first.get("id")), []).append((number, "start"))
        target.setdefault(str(last.get("id")), []).append((number, "stop"))
    return (
        {event_id: sorted(values, key=lambda item: item[0]) for event_id, values in slurs.items()},
        {event_id: sorted(values, key=lambda item: item[0]) for event_id, values in ties.items()},
    )


def measures_by_number(part: dict[str, Any]) -> dict[int, list[dict[str, Any]]]:
    result: dict[int, list[dict[str, Any]]] = {}
    for measure in part.get("measures", []) or []:
        try:
            number = int(measure.get("number") or 1)
        except (TypeError, ValueError):
            number = 1
        result[number] = list(measure.get("events", []) or [])
    return result


def voice_duration(events: list[dict[str, Any]]) -> int:
    total = 0
    last_x: float | None = None
    for group in onset_groups([event for event in events if event.get("type") in {"note", "rest"}], 8.0):
        non_marks = [event for event in group if event.get("type") in {"note", "rest"}]
        if not non_marks:
            continue
        first = non_marks[0]
        if first.get("type") == "note":
            x = float(first.get("x") or 0.0)
            if last_x is not None and abs(x - last_x) <= 8.0:
                continue
            last_x = x
        total += int(first.get("duration_units") or event_duration(first)[0])
    return total


def semantic_to_official_piano(
    semantics: dict[str, Any],
    *,
    include_key: bool,
    include_time: bool,
    emit_directions: bool,
    x_tolerance: float,
    backup_policy: str,
    fixed_measure_units: int,
    min_note_confidence: float,
    infer_same_pitch_ties: bool = False,
) -> tuple[ET.ElementTree, dict[str, int]]:
    parts = sorted(semantics.get("parts", []) or [], key=lambda item: int(item.get("staff") or 0))
    by_part = [measures_by_number(part) for part in parts]
    measure_numbers = sorted({number for measures in by_part for number in measures})

    # GrandStaff pages occasionally contain four detected horizontal staves even
    # though the official piano representation has two logical staves.  Keep the
    # detected streams as independent voices, but fold the upper/lower halves to
    # logical staff 1/2.  Emitting staff:4 is invalid for the official vocabulary.
    collapse_to_piano = len(parts) > 3
    attribute_parts = [parts[0], parts[-1]] if collapse_to_piano else parts
    piano_split = (len(parts) + 1) // 2

    root = ET.Element("score-partwise", {"version": "4.0"})
    work = ET.SubElement(root, "work")
    add_text(work, "work-title", "StaffOMR V2.1 official-style piano export")
    identification = ET.SubElement(root, "identification")
    encoding = ET.SubElement(identification, "encoding")
    add_text(encoding, "software", "StaffOMR V2.1 official-style semantic exporter")
    part_list = ET.SubElement(root, "part-list")
    score_part = ET.SubElement(part_list, "score-part", {"id": "P1"})
    add_text(score_part, "part-name", "Piano")
    score_instrument = ET.SubElement(score_part, "score-instrument", {"id": "P1-I1"})
    add_text(score_instrument, "instrument-name", "Piano")
    part_el = ET.SubElement(root, "part", {"id": "P1"})

    stats = {
        "parts": len(parts),
        "measures": len(measure_numbers),
        "notes": 0,
        "rests": 0,
        "directions": 0,
        "chord_notes": 0,
        "staff_tags": 0,
        "backups": 0,
        "tie_endpoints": 0,
    }
    emit_staff = len(parts) >= 2

    for measure_index, number in enumerate(measure_numbers):
        measure_el = ET.SubElement(part_el, "measure", {"number": str(number)})
        if measure_index == 0:
            measure_el.append(make_attributes(attribute_parts, include_key=include_key, include_time=include_time))
        emitted_voice = False
        for staff_offset, measures in enumerate(by_part):
            source_staff_number = staff_offset + 1
            staff_number = (
                1 if staff_offset < piano_split else 2
            ) if collapse_to_piano else source_staff_number
            # Preserve a distinct voice for every detected stream after folding,
            # using the official per-staff voice ranges (1-4 and 5-8).
            if collapse_to_piano:
                group_start = 0 if staff_number == 1 else piano_split
                local_voice = staff_offset - group_start
                voice = str(1 + (staff_number - 1) * 4 + local_voice)
            else:
                voice = staff_voice(source_staff_number)
            events = sorted(
                [event for event in measures.get(number, []) if keep_event(event, min_note_confidence)],
                key=sort_key,
            )
            notes_and_rests = [event for event in events if event.get("type") in {"note", "rest"}]
            if not notes_and_rests:
                if emit_directions:
                    for event in events:
                        if event.get("type") == "mark":
                            add_musicxml_direction(measure_el, event, voice, emit_staff=emit_staff)
                            stats["directions"] += 1
                continue
            if emitted_voice:
                backup_units = voice_duration(events) if backup_policy == "voice-sum" else fixed_measure_units
                measure_el.append(make_backup(backup_units))
                stats["backups"] += 1
            if emit_directions:
                for event in events:
                    if event.get("type") == "mark":
                        add_musicxml_direction(measure_el, event, voice, emit_staff=emit_staff)
                        stats["directions"] += 1
            beam_statuses = beam_statuses_for_events(notes_and_rests)
            if infer_same_pitch_ties:
                slur_statuses, tie_statuses = split_slur_tie_statuses(notes_and_rests)
            else:
                slur_statuses = slur_statuses_for_events(notes_and_rests)
                tie_statuses = {}
            for group in onset_groups(notes_and_rests, x_tolerance):
                if group[0].get("type") == "rest":
                    event_for_xml = dict(group[0])
                    add_musicxml_note(
                        measure_el,
                        event_for_xml,
                        chord=False,
                        staff_number=staff_number,
                        voice=voice,
                        emit_staff=emit_staff,
                    )
                    stats["rests"] += 1
                    stats["staff_tags"] += int(emit_staff)
                    continue
                ordered = normalize_chord_group(sorted(group, key=pitch_key))
                for chord_index, event in enumerate(ordered):
                    event_for_xml = dict(event)
                    event_for_xml["musicxml_beams"] = beam_statuses.get(str(event.get("id")), [])
                    event_for_xml["musicxml_slurs"] = slur_statuses.get(str(event.get("id")), [])
                    event_for_xml["musicxml_ties"] = tie_statuses.get(str(event.get("id")), [])
                    add_musicxml_note(
                        measure_el,
                        event_for_xml,
                        chord=chord_index > 0,
                        staff_number=staff_number,
                        voice=voice,
                        emit_staff=emit_staff,
                    )
                    stats["notes"] += 1
                    stats["staff_tags"] += int(emit_staff)
                    if chord_index > 0:
                        stats["chord_notes"] += 1
                    stats["tie_endpoints"] += len(event_for_xml["musicxml_ties"])
            emitted_voice = True

    return ET.ElementTree(root), stats


def write_musicxml(path: Path, tree: ET.ElementTree) -> None:
    ET.indent(tree, space="  ")
    path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(path, encoding="utf-8", xml_declaration=True)


def export_file(
    semantics_path: Path,
    output_path: Path,
    *,
    include_key: bool,
    include_time: bool,
    emit_directions: bool,
    x_tolerance: float,
    backup_policy: str,
    fixed_measure_units: int,
    min_note_confidence: float,
    infer_same_pitch_ties: bool = False,
) -> dict[str, Any]:
    semantics = read_json(semantics_path)
    tree, stats = semantic_to_official_piano(
        semantics,
        include_key=include_key,
        include_time=include_time,
        emit_directions=emit_directions,
        x_tolerance=x_tolerance,
        backup_policy=backup_policy,
        fixed_measure_units=fixed_measure_units,
        min_note_confidence=min_note_confidence,
        infer_same_pitch_ties=infer_same_pitch_ties,
    )
    write_musicxml(output_path, tree)
    return {"input": str(semantics_path), "output": str(output_path), **stats}


def discover_inputs(source_root: Path, semantics_relpath: str) -> list[Path]:
    return sorted(path for path in source_root.glob(f"*/{semantics_relpath}") if path.is_file())


def main() -> None:
    parser = argparse.ArgumentParser(description="Export V2.1 semantics as one official-style piano MusicXML part.")
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--semantics-json", type=Path)
    parser.add_argument("--semantics-relpath", default="semantics/semantics_v2_1.json")
    parser.add_argument("--output-name", default="score_v2_1_official_semantic.musicxml")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--out-summary", type=Path, required=True)
    parser.add_argument("--include-key", action="store_true")
    parser.add_argument("--include-time", action="store_true")
    parser.add_argument("--emit-directions", action="store_true")
    parser.add_argument("--x-tolerance", type=float, default=8.0)
    parser.add_argument("--backup-policy", choices=("fixed", "voice-sum"), default="fixed")
    parser.add_argument("--fixed-measure-units", type=int, default=64)
    parser.add_argument("--min-note-confidence", type=float, default=0.0)
    parser.add_argument("--infer-same-pitch-ties", action="store_true")
    args = parser.parse_args()

    if args.semantics_json:
        inputs = [args.semantics_json]
    elif args.source_root:
        inputs = discover_inputs(args.source_root, args.semantics_relpath)
    else:
        raise SystemExit("--semantics-json or --source-root is required")

    rows = []
    for input_path in inputs:
        if args.output:
            output_path = args.output
        else:
            output_path = input_path.parent / args.output_name
        rows.append(
            export_file(
                input_path,
                output_path,
                include_key=args.include_key,
                include_time=args.include_time,
                emit_directions=args.emit_directions,
                x_tolerance=args.x_tolerance,
                backup_policy=args.backup_policy,
                fixed_measure_units=args.fixed_measure_units,
                min_note_confidence=args.min_note_confidence,
                infer_same_pitch_ties=args.infer_same_pitch_ties,
            )
        )

    summary = {
        "files": len(rows),
        "include_key": bool(args.include_key),
        "include_time": bool(args.include_time),
        "emit_directions": bool(args.emit_directions),
        "x_tolerance": args.x_tolerance,
        "backup_policy": args.backup_policy,
        "fixed_measure_units": args.fixed_measure_units,
        "min_note_confidence": args.min_note_confidence,
        "infer_same_pitch_ties": bool(args.infer_same_pitch_ties),
        "notes": sum(row["notes"] for row in rows),
        "rests": sum(row["rests"] for row in rows),
        "directions": sum(row["directions"] for row in rows),
        "chord_notes": sum(row["chord_notes"] for row in rows),
        "staff_tags": sum(row["staff_tags"] for row in rows),
        "backups": sum(row["backups"] for row in rows),
        "tie_endpoints": sum(row["tie_endpoints"] for row in rows),
    }
    payload = {"summary": summary, "rows": rows}
    write_json(args.out_summary, payload)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
