from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_fonts() -> tuple[ImageFont.ImageFont, ImageFont.ImageFont, ImageFont.ImageFont]:
    try:
        return (
            ImageFont.truetype("arial.ttf", 22),
            ImageFont.truetype("arial.ttf", 30),
            ImageFont.truetype("arial.ttf", 46),
        )
    except Exception:
        font = ImageFont.load_default()
        return font, font, font


def draw_symbol_reconstruction(symbols_payload: dict[str, Any], ocr_payload: dict[str, Any] | None, out_path: Path) -> None:
    input_path = Path(symbols_payload["input"])
    image = Image.open(input_path).convert("RGB")
    width, height = image.size
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    small_font, mid_font, clef_font = load_fonts()

    for staff in symbols_payload.get("staves", []):
        x0, x1 = int(staff["x0"]), int(staff["x1"])
        for y in staff["lines"]:
            yy = int(round(y))
            draw.line([(x0, yy), (x1, yy)], fill=(0, 0, 0), width=2)
        clef_bbox = staff.get("clef_bbox")
        if clef_bbox:
            label = "G" if staff.get("clef_type") == "treble" else "F"
            draw.text((int(clef_bbox[0]), int(clef_bbox[1])), label, fill=(0, 0, 0), font=clef_font)

    symbols = symbols_payload.get("symbols", [])
    by_class: dict[str, list[dict[str, Any]]] = {}
    for symbol in symbols:
        by_class.setdefault(symbol["class"], []).append(symbol)

    def box(symbol: dict[str, Any]) -> list[int]:
        return [int(round(v)) for v in symbol["bbox"]]

    for symbol in by_class.get("barline", []):
        x0, y0, x1, y1 = box(symbol)
        draw.line([(x0 + (x1 - x0) // 2, y0), (x0 + (x1 - x0) // 2, y1)], fill=(0, 0, 0), width=max(2, x1 - x0))
    for symbol in by_class.get("ledger_line", []):
        x0, y0, x1, y1 = box(symbol)
        draw.line([(x0, (y0 + y1) // 2), (x1, (y0 + y1) // 2)], fill=(0, 0, 0), width=max(2, y1 - y0))
    for symbol in by_class.get("beam", []):
        x0, y0, x1, y1 = box(symbol)
        draw.rectangle([x0, y0, x1, y1], fill=(0, 0, 0))
    for symbol in by_class.get("stem", []):
        x0, y0, x1, y1 = box(symbol)
        draw.line([((x0 + x1) // 2, y0), ((x0 + x1) // 2, y1)], fill=(0, 0, 0), width=max(2, x1 - x0))
    for symbol in by_class.get("slur_or_tie", []):
        x0, y0, x1, y1 = box(symbol)
        draw.arc([x0, y0, x1, y1], 190, 350, fill=(0, 0, 0), width=2)
    for symbol in by_class.get("filled_notehead", []):
        x0, y0, x1, y1 = box(symbol)
        draw.ellipse([x0, y0, x1, y1], fill=(0, 0, 0), outline=(0, 0, 0), width=2)
    for symbol in by_class.get("open_notehead", []):
        x0, y0, x1, y1 = box(symbol)
        draw.ellipse([x0, y0, x1, y1], fill="white", outline=(0, 0, 0), width=3)
    for symbol_class, text in [("sharp", "#"), ("flat", "b"), ("natural", "N")]:
        for symbol in by_class.get(symbol_class, []):
            x0, y0, _, _ = box(symbol)
            draw.text((x0, max(0, y0 - 4)), text, fill=(0, 0, 0), font=mid_font)
    for symbol in by_class.get("rest", []):
        x0, y0, x1, y1 = box(symbol)
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        draw.rectangle([x0, cy - 5, x1, cy + 5], fill=(0, 0, 0))
        draw.ellipse([cx - 6, cy + 8, cx + 6, cy + 20], fill=(0, 0, 0))
    for symbol_class, label in [("treble_clef", "G"), ("bass_clef", "F")]:
        for symbol in by_class.get(symbol_class, []):
            x0, y0, _, _ = box(symbol)
            draw.text((x0, y0), label, fill=(0, 0, 0), font=clef_font)

    def detector_class(symbol: dict[str, Any]) -> str:
        attributes = symbol.get("attributes") or {}
        return str(attributes.get("detector_class") or attributes.get("fine_class") or symbol["class"])

    def centered_text(symbol: dict[str, Any], label: str, font: ImageFont.ImageFont = mid_font) -> None:
        x0, y0, x1, y1 = box(symbol)
        draw.text((x0, max(0, y0 - 4)), label, fill=(0, 0, 0), font=font)
        if x1 - x0 > 4 and y1 - y0 > 4:
            draw.rectangle([x0, y0, x1, y1], outline=(0, 0, 0), width=1)

    for symbol in symbols:
        symbol_class = symbol["class"]
        fine_class = detector_class(symbol)
        x0, y0, x1, y1 = box(symbol)
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        label_class = fine_class if fine_class != symbol_class else symbol_class
        if label_class in {"augmentation_dot", "repeat_dot"}:
            radius = max(2, min(x1 - x0, y1 - y0) // 2)
            draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=(0, 0, 0))
        elif label_class.startswith("flag_"):
            direction = -1 if label_class.endswith("_up") else 1
            draw.line([(x0, y0), (x1, cy + direction * max(4, y1 - y0))], fill=(0, 0, 0), width=3)
            draw.arc([x0, min(y0, cy), x1, max(y1, cy + direction * (y1 - y0))], 270, 80, fill=(0, 0, 0), width=2)
        elif label_class.startswith("time_sig_"):
            suffix = label_class.replace("time_sig_", "")
            label = {"common": "C", "cut_common": "C/"}.get(suffix, suffix)
            centered_text(symbol, label, clef_font)
        elif label_class.startswith("dynamic_"):
            if label_class == "dynamic_crescendo_hairpin":
                draw.line([(x0, cy), (x1, y0)], fill=(0, 0, 0), width=2)
                draw.line([(x0, cy), (x1, y1)], fill=(0, 0, 0), width=2)
            elif label_class == "dynamic_diminuendo_hairpin":
                draw.line([(x0, y0), (x1, cy)], fill=(0, 0, 0), width=2)
                draw.line([(x0, y1), (x1, cy)], fill=(0, 0, 0), width=2)
            else:
                centered_text(symbol, label_class.replace("dynamic_", ""), mid_font)
        elif label_class.startswith("artic_"):
            mark = {
                "artic_staccato": ".",
                "artic_staccatissimo": "'",
                "artic_accent": ">",
                "artic_tenuto": "-",
                "artic_marcato": "^",
            }.get(label_class, ".")
            centered_text(symbol, mark, mid_font)
        elif label_class == "fermata":
            centered_text(symbol, "U", mid_font)
        elif label_class.startswith("tuplet_"):
            centered_text(symbol, label_class.replace("tuplet_", ""), small_font)
        elif label_class == "fingering":
            centered_text(symbol, "1", small_font)
        elif label_class == "pedal_mark":
            centered_text(symbol, "Ped", small_font)
        elif label_class == "pedal_up":
            centered_text(symbol, "*", small_font)
        elif label_class.startswith("ornament_"):
            centered_text(symbol, "~", mid_font)
        elif label_class == "arpeggiato":
            centered_text(symbol, "~", mid_font)
        elif label_class in {"brace", "bracket"}:
            draw.line([(cx, y0), (cx, y1)], fill=(0, 0, 0), width=3)
        elif label_class == "repeat_barline":
            draw.line([(x0, y0), (x0, y1)], fill=(0, 0, 0), width=2)
            draw.line([(x1, y0), (x1, y1)], fill=(0, 0, 0), width=2)

    if ocr_payload:
        for item in ocr_payload.get("ocr_results", []):
            bbox = item.get("bbox") or item.get("symbol_bbox")
            if not bbox:
                continue
            x0, y0, _, _ = [int(round(v)) for v in bbox]
            draw.text((x0, max(0, y0 - 4)), item.get("text", ""), fill=(0, 0, 0), font=small_font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def make_comparison(input_path: Path, reconstruction_path: Path, out_path: Path) -> None:
    source = Image.open(input_path).convert("RGB")
    reconstruction = Image.open(reconstruction_path).convert("RGB")
    thumb_w = 760
    label_h = 44
    pad = 24
    labels = [("Input original", source), ("V2 reconstructed recognition", reconstruction)]
    thumbs = []
    for label, image in labels:
        thumb_h = int(image.height * thumb_w / image.width)
        thumbs.append((label, image.resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)))
    height = max(image.height for _, image in thumbs)
    canvas = Image.new("RGB", (thumb_w * 2 + pad * 3, height + label_h + pad * 2), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", 28)
    except Exception:
        font = ImageFont.load_default()
    for idx, (label, image) in enumerate(thumbs):
        x = pad + idx * (thumb_w + pad)
        y = pad
        draw.rectangle([x, y, x + thumb_w, y + label_h], fill=(28, 32, 38))
        draw.text((x + 12, y + 8), label, fill="white", font=font)
        canvas.paste(image, (x, y + label_h))
        draw.rectangle([x, y + label_h, x + thumb_w, y + label_h + image.height], outline=(170, 170, 170), width=1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Reconstruct a score-like image from V2 symbol JSON.")
    parser.add_argument("--symbols-json", type=Path, required=True)
    parser.add_argument("--ocr-json", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--comparison", type=Path)
    args = parser.parse_args()

    symbols_payload = read_json(args.symbols_json)
    ocr_payload = read_json(args.ocr_json) if args.ocr_json else None
    draw_symbol_reconstruction(symbols_payload, ocr_payload, args.out)
    if args.comparison:
        make_comparison(Path(symbols_payload["input"]), args.out, args.comparison)
    print(args.out)
    if args.comparison:
        print(args.comparison)


if __name__ == "__main__":
    main()
