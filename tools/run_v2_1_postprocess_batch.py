from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


def run(command: list[str], cwd: Path) -> None:
    print("RUN", " ".join(command), flush=True)
    subprocess.run(command, cwd=str(cwd), check=True)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def page_dirs(root: Path) -> list[Path]:
    return sorted(path for path in root.glob("page_*_v2_1_*") if path.is_dir())


def make_contact_sheet(items: list[dict[str, Any]], out_path: Path) -> None:
    if not items:
        return
    thumb_w = 760
    label_h = 36
    pad = 20
    rows = []
    for item in items:
        image = Image.open(item["comparison"]).convert("RGB")
        thumb_h = int(image.height * thumb_w / image.width)
        rows.append((item["page"], image.resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)))
    width = thumb_w + 2 * pad
    height = pad + sum(label_h + img.height + pad for _, img in rows)
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except Exception:
        font = ImageFont.load_default()
    y = pad
    for label, image in rows:
        draw.rectangle([pad, y, pad + thumb_w, y + label_h], fill=(28, 32, 38))
        draw.text((pad + 10, y + 6), label, fill="white", font=font)
        y += label_h
        canvas.paste(image, (pad, y))
        draw.rectangle([pad, y, pad + image.width, y + image.height], outline=(170, 170, 170), width=1)
        y += image.height + pad
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-run V2.1 note, semantic, and reconstruction postprocess for existing page runs.")
    parser.add_argument("--root", type=Path, default=Path("outputs/wechat_5_latest_model"))
    parser.add_argument("--out-summary", type=Path)
    parser.add_argument("--contact-sheet", type=Path)
    args = parser.parse_args()

    repo = Path.cwd()
    py = sys.executable
    root = args.root
    summary: list[dict[str, Any]] = []
    for run_dir in page_dirs(root):
        page = run_dir.name.split("_v2_1_", 1)[0]
        shapes_json = run_dir / "symbols" / "symbols_v2_1_shapes.json"
        ocr_json = run_dir / "ocr" / "symbols_v2_ocr.json"
        if not shapes_json.exists():
            print(f"SKIP {run_dir}: missing {shapes_json}", flush=True)
            continue
        notes_json = run_dir / "notes" / "notes_v2_1_postprocess.json"
        semantics_json = run_dir / "semantics" / "semantics_v2_1.json"
        musicxml_path = run_dir / "semantics" / "score_v2_1.musicxml"
        linearized_path = run_dir / "semantics" / "score_v2_1_linearized.txt"
        reconstruction = run_dir / "visuals" / "v2_reconstructed_postprocess.png"
        comparison = run_dir / "visuals" / "input_vs_v2_reconstructed_postprocess.png"

        run([py, "tools/assemble_v2_1_notes.py", "--shapes-json", str(shapes_json), "--out-json", str(notes_json)], repo)
        run(
            [
                py,
                "tools/export_v2_1_semantics.py",
                "--notes-json",
                str(notes_json),
                "--shapes-json",
                str(shapes_json),
                "--out-json",
                str(semantics_json),
                "--out-musicxml",
                str(musicxml_path),
                "--out-linearized",
                str(linearized_path),
            ],
            repo,
        )
        reconstruct_cmd = [
            py,
            "tools/reconstruct_v2_symbols.py",
            "--symbols-json",
            str(shapes_json),
            "--out",
            str(reconstruction),
            "--comparison",
            str(comparison),
        ]
        if ocr_json.exists():
            reconstruct_cmd.extend(["--ocr-json", str(ocr_json)])
        run(reconstruct_cmd, repo)

        notes_payload = read_json(notes_json)
        semantics_payload = read_json(semantics_json)
        summary.append(
            {
                "page": page,
                "run": str(run_dir),
                "notes": notes_payload["summary"].get("notes"),
                "rests": notes_payload["summary"].get("rests"),
                "marks": notes_payload["summary"].get("marks"),
                "composite_marks": notes_payload["summary"].get("composite_marks"),
                "dynamic_composites": notes_payload["summary"].get("dynamic_composites"),
                "pedal_composites": notes_payload["summary"].get("pedal_composites"),
                "notes_with_stem": notes_payload["summary"].get("notes_with_stem"),
                "notes_with_beam": notes_payload["summary"].get("notes_with_beam"),
                "duration_hints": notes_payload["summary"].get("duration_hints"),
                "semantic_parts": semantics_payload["summary"].get("parts"),
                "semantic_measures": semantics_payload["summary"].get("measures"),
                "semantic_events": semantics_payload["summary"].get("events"),
                "notes_json": str(notes_json.resolve()),
                "semantics_json": str(semantics_json.resolve()),
                "musicxml": str(musicxml_path.resolve()),
                "linearized": str(linearized_path.resolve()),
                "comparison": str(comparison.resolve()),
            }
        )

    out_summary = args.out_summary or root / "v2_1_postprocess_summary.json"
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    contact_sheet = args.contact_sheet or root / "v2_1_postprocess_contact_sheet.png"
    make_contact_sheet(summary, contact_sheet)
    print(json.dumps({"summary": str(out_summary), "contact_sheet": str(contact_sheet), "pages": len(summary)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
