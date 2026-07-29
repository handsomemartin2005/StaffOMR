from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path


def sanitize_musicxml_chords(path: Path) -> int:
    """Remove chord markers that cannot refer to a preceding pitched note."""
    tree = ET.parse(path)
    root = tree.getroot()
    removed = 0

    for measure in root.iter("measure"):
        previous_note: ET.Element | None = None
        for element in measure:
            if element.tag != "note":
                continue
            chord = element.find("chord")
            if chord is not None and (
                previous_note is None or previous_note.find("pitch") is None
            ):
                element.remove(chord)
                removed += 1
            previous_note = element

    if removed:
        tree.write(path, encoding="utf-8", xml_declaration=True)
    return removed


def sanitize_tree(source_root: Path, prediction_relpath: Path) -> tuple[int, int]:
    files_changed = 0
    markers_removed = 0
    for sample_dir in sorted(path for path in source_root.iterdir() if path.is_dir()):
        prediction = sample_dir / prediction_relpath
        if not prediction.is_file():
            continue
        removed = sanitize_musicxml_chords(prediction)
        if removed:
            files_changed += 1
            markers_removed += removed
    return files_changed, markers_removed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--prediction-relpath", type=Path, required=True)
    args = parser.parse_args()
    files, markers = sanitize_tree(args.source_root, args.prediction_relpath)
    print({"files_changed": files, "chord_markers_removed": markers})


if __name__ == "__main__":
    main()
