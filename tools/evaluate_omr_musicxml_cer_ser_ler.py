#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import zipfile
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET


def strip_ns(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def children_named(el: ET.Element, name: str) -> list[ET.Element]:
    return [child for child in list(el) if strip_ns(child.tag) == name]


def first_child(el: ET.Element, name: str) -> ET.Element | None:
    for child in list(el):
        if strip_ns(child.tag) == name:
            return child
    return None


def child_text(el: ET.Element, name: str, default: str = "") -> str:
    child = first_child(el, name)
    if child is None or child.text is None:
        return default
    return child.text.strip()


def iter_named(el: ET.Element, name: str):
    for item in el.iter():
        if strip_ns(item.tag) == name:
            yield item


def read_musicxml_tree(path: Path) -> ET.ElementTree:
    if path.suffix.lower() == ".mxl":
        with zipfile.ZipFile(path) as zf:
            xml_name = None
            if "META-INF/container.xml" in zf.namelist():
                container = ET.fromstring(zf.read("META-INF/container.xml"))
                for rootfile in iter_named(container, "rootfile"):
                    candidate = rootfile.attrib.get("full-path")
                    if candidate:
                        xml_name = candidate
                        break
            if xml_name is None:
                candidates = [
                    name
                    for name in zf.namelist()
                    if name.lower().endswith(".xml") and not name.startswith("META-INF/")
                ]
                if not candidates:
                    raise ValueError(f"No MusicXML XML payload found in {path}")
                xml_name = candidates[0]
            return ET.ElementTree(ET.fromstring(zf.read(xml_name)))
    return ET.parse(path)


def normalize_text(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"\s+", " ", value)
    return value


def note_pitch(note: ET.Element) -> str:
    pitch = first_child(note, "pitch")
    if pitch is None:
        return "unpitched"
    step = child_text(pitch, "step", "?")
    alter = child_text(pitch, "alter", "0")
    octave = child_text(pitch, "octave", "?")
    return f"{step}{alter}:{octave}"


def note_type(note: ET.Element) -> str:
    note_type_text = child_text(note, "type")
    if not note_type_text:
        note_type_text = f"duration={child_text(note, 'duration', '?')}"
    dots = len(children_named(note, "dot"))
    return f"{note_type_text}{'.' * dots}"


def note_voice(note: ET.Element) -> str:
    return child_text(note, "voice", "1") or "1"


def note_staff(note: ET.Element) -> str:
    return child_text(note, "staff", "1") or "1"


def note_beam_count(note: ET.Element) -> int:
    return len(children_named(note, "beam"))


def note_event(note: ET.Element, ignore_beams: bool) -> tuple[str, str]:
    is_rest = first_child(note, "rest") is not None
    is_chord = first_child(note, "chord") is not None
    if is_rest:
        event_class = "rest"
        token = f"rest|type={note_type(note)}|voice={note_voice(note)}|staff={note_staff(note)}"
    else:
        event_class = "chord_note" if is_chord else "note"
        token = (
            f"{event_class}|pitch={note_pitch(note)}|type={note_type(note)}"
            f"|voice={note_voice(note)}|staff={note_staff(note)}"
        )
    if not ignore_beams:
        token += f"|beams={note_beam_count(note)}"
    return event_class, token


def direction_text(direction: ET.Element) -> str:
    texts: list[str] = []
    for words in iter_named(direction, "words"):
        if words.text:
            texts.append(normalize_text(words.text))
    for dynamics in iter_named(direction, "dynamics"):
        names = [strip_ns(child.tag) for child in list(dynamics)]
        texts.extend(normalize_text(name) for name in names)
    if not texts:
        for child in iter_named(direction, "other-direction"):
            if child.text:
                texts.append(normalize_text(child.text))
    return "+".join(text for text in texts if text) or "direction"


def direction_event(direction: ET.Element) -> tuple[str, str]:
    placement = direction.attrib.get("placement", "")
    staff = child_text(direction, "staff", "1") or "1"
    text = direction_text(direction)
    return "direction", f"direction|text={text}|placement={placement}|staff={staff}"


def attributes_events(attributes: ET.Element) -> list[tuple[str, str]]:
    events: list[tuple[str, str]] = []
    key = first_child(attributes, "key")
    if key is not None:
        events.append(("key", f"key|fifths={child_text(key, 'fifths', '0')}|mode={child_text(key, 'mode', '')}"))
    time = first_child(attributes, "time")
    if time is not None:
        events.append(("time", f"time|beats={child_text(time, 'beats', '?')}|beat_type={child_text(time, 'beat-type', '?')}"))
    for index, clef in enumerate(children_named(attributes, "clef"), start=1):
        number = clef.attrib.get("number", str(index))
        events.append(("clef", f"clef|number={number}|sign={child_text(clef, 'sign', '?')}|line={child_text(clef, 'line', '?')}"))
    return events


def extract_musicxml_sequences(
    path: Path,
    *,
    include_attributes: bool,
    ignore_directions: bool,
    ignore_beams: bool,
) -> dict[str, list[str]]:
    tree = read_musicxml_tree(path)
    root = tree.getroot()
    class_tokens: list[str] = []
    semantic_tokens: list[str] = []
    line_tokens: list[str] = []

    for part_index, part in enumerate(children_named(root, "part"), start=1):
        part_id = part.attrib.get("id", f"P{part_index}")
        for measure_index, measure in enumerate(children_named(part, "measure"), start=1):
            measure_no = measure.attrib.get("number", str(measure_index))
            measure_class_tokens: list[str] = []
            measure_semantic_tokens: list[str] = []
            for child in list(measure):
                name = strip_ns(child.tag)
                events: list[tuple[str, str]] = []
                if include_attributes and name == "attributes":
                    events.extend(attributes_events(child))
                elif name == "note":
                    events.append(note_event(child, ignore_beams))
                elif not ignore_directions and name == "direction":
                    events.append(direction_event(child))
                else:
                    continue
                for event_class, token in events:
                    class_tokens.append(event_class)
                    semantic_tokens.append(token)
                    measure_class_tokens.append(event_class)
                    measure_semantic_tokens.append(token)
            line_tokens.append(
                f"part={part_id}|measure={measure_no}|classes={' '.join(measure_class_tokens)}|events={' '.join(measure_semantic_tokens)}"
            )
    return {
        "class_tokens": class_tokens,
        "semantic_tokens": semantic_tokens,
        "line_tokens": line_tokens,
    }


def levenshtein_ops(ref: list[str], hyp: list[str]) -> dict[str, int]:
    n = len(ref)
    m = len(hyp)
    previous = list(range(m + 1))
    current = [0] * (m + 1)
    back: list[list[str]] = [[""] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        back[i][0] = "D"
    for j in range(1, m + 1):
        back[0][j] = "I"

    for i in range(1, n + 1):
        current[0] = i
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                best_cost, op = previous[j - 1], "M"
            else:
                best_cost, op = previous[j - 1] + 1, "S"
            if previous[j] + 1 < best_cost:
                best_cost, op = previous[j] + 1, "D"
            if current[j - 1] + 1 < best_cost:
                best_cost, op = current[j - 1] + 1, "I"
            current[j] = best_cost
            back[i][j] = op
        previous, current = current, previous

    i, j = n, m
    ops = {"substitutions": 0, "deletions": 0, "insertions": 0, "matches": 0}
    while i > 0 or j > 0:
        op = back[i][j]
        if op == "M":
            ops["matches"] += 1
            i -= 1
            j -= 1
        elif op == "S":
            ops["substitutions"] += 1
            i -= 1
            j -= 1
        elif op == "D":
            ops["deletions"] += 1
            i -= 1
        else:
            ops["insertions"] += 1
            j -= 1
    return ops


def error_rate(ops: dict[str, int], ref_len: int) -> float:
    if ref_len == 0:
        return 0.0 if ops["insertions"] == 0 else 1.0
    return (ops["substitutions"] + ops["deletions"] + ops["insertions"]) / ref_len


def metric_from_sequences(ref: list[str], hyp: list[str]) -> dict[str, Any]:
    ops = levenshtein_ops(ref, hyp)
    return {
        "rate": error_rate(ops, len(ref)),
        "ops": ops,
        "ref_len": len(ref),
        "hyp_len": len(hyp),
    }


def compare_pair(
    reference: Path,
    prediction: Path,
    *,
    include_attributes: bool,
    ignore_directions: bool,
    ignore_beams: bool,
) -> dict[str, Any]:
    ref = extract_musicxml_sequences(
        reference,
        include_attributes=include_attributes,
        ignore_directions=ignore_directions,
        ignore_beams=ignore_beams,
    )
    hyp = extract_musicxml_sequences(
        prediction,
        include_attributes=include_attributes,
        ignore_directions=ignore_directions,
        ignore_beams=ignore_beams,
    )
    cer = metric_from_sequences(ref["class_tokens"], hyp["class_tokens"])
    ser = metric_from_sequences(ref["semantic_tokens"], hyp["semantic_tokens"])
    ler = metric_from_sequences(ref["line_tokens"], hyp["line_tokens"])
    return {
        "reference": str(reference.resolve()),
        "prediction": str(prediction.resolve()),
        "CER": cer["rate"],
        "CES": cer["rate"],
        "SER": ser["rate"],
        "LER": ler["rate"],
        "metrics": {
            "CER": cer,
            "SER": ser,
            "LER": ler,
        },
        "counts": {
            "ref_class_tokens": len(ref["class_tokens"]),
            "hyp_class_tokens": len(hyp["class_tokens"]),
            "ref_semantic_tokens": len(ref["semantic_tokens"]),
            "hyp_semantic_tokens": len(hyp["semantic_tokens"]),
            "ref_lines": len(ref["line_tokens"]),
            "hyp_lines": len(hyp["line_tokens"]),
        },
    }


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def manifest_pages(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("pages"), list):
        return payload["pages"]
    raise ValueError("Manifest must be a list or an object with a 'pages' list")


def resolve_prediction_map(page: dict[str, Any]) -> dict[str, str]:
    predictions: dict[str, str] = {}
    nested = page.get("predictions")
    if isinstance(nested, dict):
        predictions.update({str(key): str(value) for key, value in nested.items() if value})
    for key, value in page.items():
        if key in {"page", "name", "gt", "reference", "reference_musicxml", "predictions"}:
            continue
        if isinstance(value, str) and value:
            predictions[key] = value
    return predictions


def evaluate_manifest(
    manifest_path: Path,
    *,
    include_attributes: bool,
    ignore_directions: bool,
    ignore_beams: bool,
    missing_ok: bool,
) -> list[dict[str, Any]]:
    base = manifest_path.parent
    rows: list[dict[str, Any]] = []
    for page in manifest_pages(read_json(manifest_path)):
        page_name = str(page.get("page") or page.get("name") or f"page_{len(rows) + 1:02d}")
        reference_raw = page.get("gt") or page.get("reference") or page.get("reference_musicxml")
        if not reference_raw:
            if missing_ok:
                continue
            raise ValueError(f"{page_name}: missing gt/reference path")
        reference = (base / str(reference_raw)).resolve() if not Path(str(reference_raw)).is_absolute() else Path(str(reference_raw))
        if not reference.exists():
            if missing_ok:
                continue
            raise FileNotFoundError(f"{page_name}: reference not found: {reference}")
        for method, pred_raw in sorted(resolve_prediction_map(page).items()):
            prediction = (base / pred_raw).resolve() if not Path(pred_raw).is_absolute() else Path(pred_raw)
            if not prediction.exists():
                if missing_ok:
                    rows.append({"page": page_name, "method": method, "exists": False, "prediction": str(prediction)})
                    continue
                raise FileNotFoundError(f"{page_name}/{method}: prediction not found: {prediction}")
            result = compare_pair(
                reference,
                prediction,
                include_attributes=include_attributes,
                ignore_directions=ignore_directions,
                ignore_beams=ignore_beams,
            )
            rows.append({"page": page_name, "method": method, "exists": True, **result})
    return rows


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Strict MusicXML CER/SER/LER",
        "",
        "Requires GT/reference MusicXML per page. CES is emitted as an alias of CER for compatibility with naming variants.",
        "",
        "| Page | Method | CER/CES | SER | LER | Ref/Hyp symbols | Ref/Hyp lines |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload.get("results", []):
        if not row.get("exists", True):
            lines.append(f"| {row.get('page')} | {row.get('method')} | - | - | - | - | - |")
            continue
        counts = row["counts"]
        lines.append(
            "| {page} | {method} | {cer:.6f} | {ser:.6f} | {ler:.6f} | {rs}/{hs} | {rl}/{hl} |".format(
                page=row["page"],
                method=row["method"],
                cer=row["CER"],
                ser=row["SER"],
                ler=row["LER"],
                rs=counts["ref_semantic_tokens"],
                hs=counts["hyp_semantic_tokens"],
                rl=counts["ref_lines"],
                hl=counts["hyp_lines"],
            )
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Strict GT-required CER/SER/LER evaluation for MusicXML/MXL OMR outputs.")
    parser.add_argument("--manifest", type=Path, help="JSON manifest with pages, GT MusicXML, and method prediction paths.")
    parser.add_argument("--reference", type=Path, help="Single-pair reference/GT MusicXML or MXL.")
    parser.add_argument("--prediction", type=Path, help="Single-pair predicted MusicXML or MXL.")
    parser.add_argument("--method", default="prediction", help="Method name for single-pair mode.")
    parser.add_argument("--page", default="single", help="Page name for single-pair mode.")
    parser.add_argument("--out-json", type=Path, default=Path("outputs/musicxml_metrics/cer_ser_ler.json"))
    parser.add_argument("--out-md", type=Path)
    parser.add_argument("--include-attributes", action="store_true", help="Include key/time/clef tokens in CER/SER/LER.")
    parser.add_argument("--ignore-directions", action="store_true", help="Ignore MusicXML direction text/dynamics.")
    parser.add_argument("--ignore-beams", action="store_true", help="Ignore beam tag counts in semantic tokens.")
    parser.add_argument("--missing-ok", action="store_true", help="Skip missing GT/prediction files instead of failing.")
    args = parser.parse_args()

    if args.manifest:
        rows = evaluate_manifest(
            args.manifest,
            include_attributes=args.include_attributes,
            ignore_directions=args.ignore_directions,
            ignore_beams=args.ignore_beams,
            missing_ok=args.missing_ok,
        )
    elif args.reference and args.prediction:
        rows = [
            {
                "page": args.page,
                "method": args.method,
                "exists": True,
                **compare_pair(
                    args.reference,
                    args.prediction,
                    include_attributes=args.include_attributes,
                    ignore_directions=args.ignore_directions,
                    ignore_beams=args.ignore_beams,
                ),
            }
        ]
    else:
        raise SystemExit("Provide either --manifest or both --reference and --prediction.")

    payload = {
        "definitions": {
            "CER": "Class/event sequence edit rate over normalized MusicXML events. CES is an alias of CER in this script.",
            "SER": "Semantic event edit rate over normalized MusicXML note/rest/direction tokens including pitch, duration, voice, staff, and beam count unless ignored.",
            "LER": "Line/layout edit rate over part-measure lines. One line token represents one MusicXML part+measure event sequence.",
            "normalization": {
                "include_attributes": args.include_attributes,
                "ignore_directions": args.ignore_directions,
                "ignore_beams": args.ignore_beams,
            },
        },
        "results": rows,
    }
    write_json(args.out_json, payload)
    if args.out_md:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(render_markdown(payload), encoding="utf-8")

    print(
        json.dumps(
            {
                "out_json": str(args.out_json),
                "out_md": str(args.out_md) if args.out_md else None,
                "results": [
                    {
                        "page": row.get("page"),
                        "method": row.get("method"),
                        "CER": row.get("CER"),
                        "CES": row.get("CES"),
                        "SER": row.get("SER"),
                        "LER": row.get("LER"),
                    }
                    for row in rows
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
