from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

from sanitize_musicxml_chords import sanitize_musicxml_chords


STEP_ORDER = {"C": 0, "D": 1, "E": 2, "F": 3, "G": 4, "A": 5, "B": 6}
DIVISIONS = 16


def child_text(parent: ET.Element, tag: str, default: str = "") -> str:
    child = parent.find(tag)
    return child.text if child is not None and child.text is not None else default


def first_child_index(parent: ET.Element, tag: str) -> int | None:
    for index, child in enumerate(list(parent)):
        if child.tag == tag:
            return index
    return None


def remove_children(parent: ET.Element, tag: str) -> None:
    for child in list(parent):
        if child.tag == tag:
            parent.remove(child)


def add_text(parent: ET.Element, tag: str, text: str | int) -> ET.Element:
    child = ET.SubElement(parent, tag)
    child.text = str(text)
    return child


def make_default_attributes() -> ET.Element:
    attrs = ET.Element("attributes")
    add_text(attrs, "divisions", DIVISIONS)
    key = ET.SubElement(attrs, "key")
    add_text(key, "fifths", 0)
    time = ET.SubElement(attrs, "time")
    add_text(time, "beats", 4)
    add_text(time, "beat-type", 4)
    clef = ET.SubElement(attrs, "clef")
    add_text(clef, "sign", "G")
    add_text(clef, "line", 2)
    return attrs


def ensure_first_measure_attributes(root: ET.Element) -> int:
    inserted = 0
    for part in root.findall("part"):
        first_measure = part.find("measure")
        if first_measure is None:
            continue
        attrs = first_measure.find("attributes")
        if attrs is None:
            first_measure.insert(0, make_default_attributes())
            inserted += 1
            continue
        if attrs.find("divisions") is None:
            division = ET.Element("divisions")
            division.text = str(DIVISIONS)
            attrs.insert(0, division)
            inserted += 1
    return inserted


def normalize_single_staff_tags(root: ET.Element) -> int:
    removed = 0
    for part in root.findall("part"):
        declared = []
        for staves in part.findall("./measure/attributes/staves"):
            try:
                declared.append(int(staves.text or "1"))
            except ValueError:
                declared.append(1)
        if not declared or max(declared) > 1:
            continue
        for parent_tag in ("note", "direction"):
            for parent in part.iter(parent_tag):
                for staff in list(parent.findall("staff")):
                    if (staff.text or "1").strip() == "1":
                        parent.remove(staff)
                        removed += 1
    return removed


def pitch_key(note: ET.Element) -> tuple[int, int, str]:
    pitch = note.find("pitch")
    if pitch is None:
        return (999, 999, ET.tostring(note, encoding="unicode"))
    step = child_text(pitch, "step", "C")
    octave_text = child_text(pitch, "octave", "4")
    try:
        octave = int(octave_text)
    except ValueError:
        octave = 4
    alter_text = child_text(pitch, "alter", "0")
    try:
        alter = int(float(alter_text))
    except ValueError:
        alter = 0
    return (octave * 7 + STEP_ORDER.get(step, 0), alter, ET.tostring(note, encoding="unicode"))


def insert_dots(note: ET.Element, dot_count: int) -> None:
    remove_children(note, "dot")
    type_index = first_child_index(note, "type")
    insert_at = len(list(note)) if type_index is None else type_index + 1
    for _ in range(max(0, dot_count)):
        note.insert(insert_at, ET.Element("dot"))
        insert_at += 1


def clone_with_chord_state(
    note: ET.Element,
    *,
    chord: bool,
    duration: str | None,
    note_type: str | None,
    dot_count: int,
) -> ET.Element:
    result = ET.fromstring(ET.tostring(note, encoding="utf-8"))
    remove_children(result, "chord")
    if chord and result.find("rest") is None:
        result.insert(0, ET.Element("chord"))
    if duration is not None:
        duration_el = result.find("duration")
        if duration_el is None:
            duration_el = ET.SubElement(result, "duration")
        duration_el.text = duration
    if note_type is not None:
        type_el = result.find("type")
        if type_el is None:
            type_el = ET.SubElement(result, "type")
        type_el.text = note_type
    insert_dots(result, dot_count)
    return result


def chord_profile(note: ET.Element) -> tuple[str, str, int]:
    return child_text(note, "duration", ""), child_text(note, "type", ""), len(note.findall("dot"))


def chord_groups(measure: ET.Element) -> list[tuple[int, int, list[ET.Element]]]:
    children = list(measure)
    groups: list[tuple[int, int, list[ET.Element]]] = []
    index = 0
    while index < len(children):
        child = children[index]
        if child.tag != "note":
            index += 1
            continue
        start = index
        group = [child]
        index += 1
        while index < len(children):
            candidate = children[index]
            if candidate.tag == "note" and candidate.find("chord") is not None:
                group.append(candidate)
                index += 1
                continue
            break
        groups.append((start, index, group))
    return groups


def fix_measure(measure: ET.Element) -> dict[str, int]:
    stats = {
        "chord_groups": 0,
        "notes_reordered": 0,
        "durations_normalized": 0,
        "dots_normalized": 0,
        "rest_chords_removed": 0,
    }
    for note in measure.findall("note"):
        if note.find("rest") is not None and note.find("chord") is not None:
            remove_children(note, "chord")
            stats["rest_chords_removed"] += 1
    children = list(measure)
    replacements: dict[int, list[ET.Element]] = {}
    for start, end, group in chord_groups(measure):
        pitched = [note for note in group if note.find("pitch") is not None]
        if len(pitched) <= 1:
            continue
        stats["chord_groups"] += 1
        ordered = sorted(group, key=pitch_key)
        if [ET.tostring(note, encoding="unicode") for note in ordered] != [
            ET.tostring(note, encoding="unicode") for note in group
        ]:
            stats["notes_reordered"] += len(group)
        profiles = Counter(chord_profile(note) for note in group)
        duration_text, note_type_text, dot_count = max(
            profiles.items(),
            key=lambda item: (item[1], item[0][2], item[0][0], item[0][1]),
        )[0]
        duration = duration_text or None
        note_type = note_type_text or None
        durations = {child_text(note, "duration", "") for note in group}
        types = {child_text(note, "type", "") for note in group}
        dot_counts = {len(note.findall("dot")) for note in group}
        if len(durations) > 1 or len(types) > 1:
            stats["durations_normalized"] += len(group)
        if len(dot_counts) > 1:
            stats["dots_normalized"] += len(group)
        replacements[start] = [
            clone_with_chord_state(note, chord=(idx > 0), duration=duration, note_type=note_type, dot_count=dot_count)
            for idx, note in enumerate(ordered)
        ]
        for remove_index in range(start + 1, end):
            replacements[remove_index] = []

    if not replacements:
        return stats
    for child in children:
        measure.remove(child)
    for index, child in enumerate(children):
        for new_child in replacements.get(index, [child]):
            measure.append(new_child)
    return stats


def fix_file(input_path: Path, output_path: Path) -> dict[str, Any]:
    tree = ET.parse(input_path)
    total = {
        "chord_groups": 0,
        "notes_reordered": 0,
        "durations_normalized": 0,
        "dots_normalized": 0,
        "rest_chords_removed": 0,
        "first_measure_attributes_inserted": ensure_first_measure_attributes(tree.getroot()),
        "single_staff_tags_removed": normalize_single_staff_tags(tree.getroot()),
    }
    for measure in tree.getroot().iter("measure"):
        stats = fix_measure(measure)
        for key, value in stats.items():
            total[key] += value
    ET.indent(tree, space="  ")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    return {"input": str(input_path), "output": str(output_path), **total}


def discover_inputs(source_root: Path, prediction_relpath: str) -> list[Path]:
    return sorted(path for path in source_root.glob(f"*/{prediction_relpath}") if path.is_file())


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair MusicXML chord order and duration consistency.")
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--prediction-relpath", default="semantics/score_v2_1.musicxml")
    parser.add_argument("--output-name", default="score_v2_1_chordfix.musicxml")
    parser.add_argument("--out-summary", type=Path, required=True)
    args = parser.parse_args()

    if args.input:
        inputs = [args.input]
    elif args.source_root:
        inputs = discover_inputs(args.source_root, args.prediction_relpath)
    else:
        raise SystemExit("--input or --source-root is required")

    rows = []
    for input_path in inputs:
        output_path = input_path.with_name(args.output_name)
        row = fix_file(input_path, output_path)
        row["orphan_chords_removed"] = sanitize_musicxml_chords(output_path)
        rows.append(row)
    summary = {
        "files": len(rows),
        "chord_groups": sum(row["chord_groups"] for row in rows),
        "notes_reordered": sum(row["notes_reordered"] for row in rows),
        "durations_normalized": sum(row["durations_normalized"] for row in rows),
        "dots_normalized": sum(row["dots_normalized"] for row in rows),
        "orphan_chords_removed": sum(row["orphan_chords_removed"] for row in rows),
    }
    payload = {"summary": summary, "rows": rows}
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
