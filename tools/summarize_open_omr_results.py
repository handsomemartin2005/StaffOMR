#!/usr/bin/env python3
"""Summarize StaffOMR V2.1, Audiveris, and oemer outputs for the five-page OMR demo."""

from __future__ import annotations

import argparse
import json
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET


PAGES = [f"page_{idx:02d}" for idx in range(1, 6)]


def strip_ns(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def children_named(el: ET.Element, name: str) -> list[ET.Element]:
    return [child for child in list(el) if strip_ns(child.tag) == name]


def first_child(el: ET.Element, name: str) -> ET.Element | None:
    for child in list(el):
        if strip_ns(child.tag) == name:
            return child
    return None


def child_text(el: ET.Element, name: str) -> str | None:
    child = first_child(el, name)
    if child is None:
        return None
    return child.text


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
                xml_candidates = [
                    name
                    for name in zf.namelist()
                    if name.lower().endswith(".xml") and not name.startswith("META-INF/")
                ]
                if not xml_candidates:
                    raise ValueError(f"No MusicXML file found in {path}")
                xml_name = xml_candidates[0]
            return ET.ElementTree(ET.fromstring(zf.read(xml_name)))
    return ET.parse(path)


def summarize_musicxml(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {"exists": False, "path": str(path) if path else None}
    try:
        tree = read_musicxml_tree(path)
        root = tree.getroot()
    except Exception as exc:  # pragma: no cover - diagnostic path
        return {"exists": False, "path": str(path), "error": str(exc)}

    parts = list(iter_named(root, "part"))
    measures = list(iter_named(root, "measure"))
    notes = list(iter_named(root, "note"))
    type_counts: Counter[str] = Counter()
    rest_count = 0
    pitched_count = 0
    chord_count = 0
    beam_count = 0
    for note in notes:
        if first_child(note, "rest") is not None:
            rest_count += 1
        if first_child(note, "pitch") is not None:
            pitched_count += 1
        if first_child(note, "chord") is not None:
            chord_count += 1
        if first_child(note, "beam") is not None:
            beam_count += 1
        note_type = child_text(note, "type")
        if note_type:
            type_counts[note_type] += 1

    return {
        "exists": True,
        "path": str(path.resolve()),
        "parts": len(parts),
        "measures": len(measures),
        "notes_total": len(notes),
        "pitched_notes": pitched_count,
        "rests": rest_count,
        "chord_notes": chord_count,
        "notes_with_beam_tag": beam_count,
        "note_type_counts": dict(sorted(type_counts.items())),
    }


def first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def summarize_text_flags(log_path: Path | None) -> dict[str, Any]:
    if log_path is None or not log_path.exists():
        return {"path": str(log_path) if log_path else None, "exists": False}
    text = log_path.read_text(errors="ignore")
    return {
        "path": str(log_path.resolve()),
        "exists": True,
        "failed_page_step": "Error in reaching PAGE" in text,
        "low_interline_warning": "interline" in text and "too low" in text,
        "cuda_provider_warning": "Failed to create CUDAExecutionProvider" in text
        or "cublasLt64_13.dll" in text,
    }


def load_staff_summary(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {row["page"]: row for row in rows}


def summarize_audiveris(audiveris_root: Path, page: str) -> dict[str, Any]:
    page_dir = audiveris_root / page
    mxl = first_existing([page_dir / f"{page}.mxl", page_dir / f"{page}.musicxml", page_dir / f"{page}.xml"])
    summary = summarize_musicxml(mxl)
    summary["omr"] = str((page_dir / f"{page}.omr").resolve()) if (page_dir / f"{page}.omr").exists() else None
    summary["console_log"] = summarize_text_flags(page_dir / f"{page}.console.log")
    return summary


def summarize_oemer(oemer_root: Path, page: str) -> dict[str, Any]:
    page_dir = oemer_root / page
    musicxml = first_existing(
        [
            page_dir / f"{page}.musicxml",
            page_dir / f"{page}.mxl",
            page_dir / f"{page}.xml",
            page_dir / "output.musicxml",
            page_dir / "output.xml",
        ]
    )
    summary = summarize_musicxml(musicxml)
    teaser = page_dir / f"{page}_teaser.png"
    summary["teaser"] = str(teaser.resolve()) if teaser.exists() else None
    summary["stdout_log"] = summarize_text_flags(page_dir / f"{page}.stdout.log")
    summary["stderr_log"] = summarize_text_flags(page_dir / f"{page}.stderr.log")
    legacy_log = page_dir / f"{page}.console.log"
    if legacy_log.exists():
        summary["console_log"] = summarize_text_flags(legacy_log)
    return summary


def compact_staff_row(row: dict[str, Any] | None) -> dict[str, Any]:
    if not row:
        return {"exists": False}
    keys = [
        "notes",
        "rests",
        "notes_with_stem",
        "notes_with_beam",
        "semantic_parts",
        "semantic_measures",
        "semantic_events",
        "duration_hints",
        "musicxml",
        "comparison",
    ]
    result = {"exists": True}
    for key in keys:
        if key in row:
            result[key] = row[key]
    return result


def render_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Open OMR Comparison",
        "",
        "Counts are structural MusicXML/output counts, not CER/SER/LER. They are intended to show coarse recognition coverage for the same five input images.",
        "",
        "| Page | StaffOMR notes/rests | StaffOMR stem/beam | Audiveris notes/rests/beam | oemer notes/rests/beam | Notes |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        staff = row["staffomr_v2_1"]
        aud = row["audiveris"]
        oemer = row["oemer"]
        staff_notes = f"{staff.get('notes', '-')}/{staff.get('rests', '-')}" if staff.get("exists") else "-"
        staff_rel = f"{staff.get('notes_with_stem', '-')}/{staff.get('notes_with_beam', '-')}" if staff.get("exists") else "-"
        aud_counts = "-"
        notes = []
        if aud.get("exists"):
            aud_counts = f"{aud.get('notes_total', '-')}/{aud.get('rests', '-')}/{aud.get('notes_with_beam_tag', '-')}"
        elif aud.get("console_log", {}).get("failed_page_step"):
            notes.append("Audiveris failed PAGE")
        if row.get("audiveris_x2_preprocess", {}).get("exists"):
            x2 = row["audiveris_x2_preprocess"]
            notes.append(
                f"Audiveris x2: {x2.get('notes_total', '-')}/{x2.get('rests', '-')}/{x2.get('notes_with_beam_tag', '-')}"
            )
        oemer_counts = "-"
        if oemer.get("exists"):
            oemer_counts = f"{oemer.get('notes_total', '-')}/{oemer.get('rests', '-')}/{oemer.get('notes_with_beam_tag', '-')}"
        elif oemer.get("stderr_log", {}).get("exists") or oemer.get("console_log", {}).get("exists"):
            notes.append("oemer incomplete/error")
        lines.append(
            f"| {row['page']} | {staff_notes} | {staff_rel} | {aud_counts} | {oemer_counts} | {'; '.join(notes)} |"
        )
    lines.extend(["", "Generated by `tools/summarize_open_omr_results.py`."])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--staff-summary", type=Path, default=Path("outputs/wechat_5_latest_model/v2_1_postprocess_summary.json"))
    parser.add_argument("--audiveris-root", type=Path, default=Path("outputs/open_omr/audiveris_5_10_2/wechat5"))
    parser.add_argument("--oemer-root", type=Path, default=Path("outputs/open_omr/oemer_0_1_8"))
    parser.add_argument("--out-json", type=Path, default=Path("outputs/open_omr/open_omr_vs_staffomr_summary.json"))
    parser.add_argument("--out-md", type=Path, default=Path("outputs/open_omr/open_omr_vs_staffomr_summary.md"))
    args = parser.parse_args()

    staff_rows = load_staff_summary(args.staff_summary)
    rows: list[dict[str, Any]] = []
    for page in PAGES:
        row: dict[str, Any] = {
            "page": page,
            "staffomr_v2_1": compact_staff_row(staff_rows.get(page)),
            "audiveris": summarize_audiveris(args.audiveris_root, page),
            "oemer": summarize_oemer(args.oemer_root, page),
        }
        if page == "page_05":
            row["audiveris_x2_preprocess"] = summarize_audiveris(args.audiveris_root, "page_05_x2")
        rows.append(row)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    args.out_md.write_text(render_markdown(rows), encoding="utf-8")
    print(json.dumps({"out_json": str(args.out_json), "out_md": str(args.out_md), "pages": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
