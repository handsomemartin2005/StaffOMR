from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from omr_v2_common import clamp_bbox, color_for_class, read_json, write_json


def load_ocr_engine():
    try:
        from rapidocr_onnxruntime import RapidOCR
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "RapidOCR is not installed. Install it with: python -m pip install rapidocr-onnxruntime opencv-python-headless"
        ) from exc
    return RapidOCR()


def clean_text(text: str) -> str:
    text = text.replace("|", "I")
    text = re.sub(r"\s+", " ", text).strip()
    text = text.replace("D. S.", "D.S.").replace("D.S .", "D.S.")
    return text


def normalize_score_text(raw_text: str) -> str:
    compact = clean_text(raw_text)
    lower = compact.lower()
    lower_alnum = re.sub(r"[^a-z0-9.]+", " ", lower)
    if re.search(r"\bp\s*no\.?\b", lower_alnum) or lower_alnum.startswith("pno"):
        return "Pno."
    if "8va" in lower:
        return "8va"
    if "coda" in lower and "to" in lower:
        return "To Coda"
    if "coda" in lower and "al" in lower and ("d.s" in lower or re.search(r"\bs\b", lower)):
        return "D.S. al Coda"
    if "coda" in lower and "al" in lower:
        return "al Coda"
    if "coda" in lower:
        return "Coda"
    return compact


def sort_ocr_items(items: list[list[Any]]) -> list[list[Any]]:
    def key(item: list[Any]) -> tuple[float, float]:
        pts = item[0]
        xs = [float(p[0]) for p in pts]
        ys = [float(p[1]) for p in pts]
        return (sum(ys) / len(ys), sum(xs) / len(xs))

    return sorted(items, key=key)


def run_ocr_on_bbox(
    ocr: Any,
    image: Image.Image,
    bbox: list[int],
    scale: int = 3,
) -> dict[str, Any]:
    crop = image.crop(tuple(bbox)).convert("RGB")
    if scale > 1:
        crop = crop.resize((crop.width * scale, crop.height * scale), Image.Resampling.LANCZOS)
    result, elapsed = ocr(np.array(crop))
    items = sort_ocr_items(result or [])
    raw_parts = [str(item[1]) for item in items]
    confidences = [float(item[2]) for item in items]
    raw_text = clean_text(" ".join(raw_parts))
    normalized = normalize_score_text(raw_text)
    return {
        "engine": "rapidocr_onnxruntime",
        "bbox": bbox,
        "scale": scale,
        "raw_text": raw_text,
        "text": normalized,
        "confidence": float(sum(confidences) / len(confidences)) if confidences else 0.0,
        "elapsed": elapsed,
        "items": [
            {
                "text": str(item[1]),
                "confidence": float(item[2]),
                "box": item[0],
            }
            for item in items
        ],
    }


def expanded_text_bbox(symbol: dict[str, Any], image_size: tuple[int, int]) -> list[int] | None:
    x0, y0, x1, y1 = symbol["bbox"]
    attrs = symbol.get("attributes", {})
    space = float(attrs.get("staff_space") or max(12, y1 - y0))
    known_text = str(attrs.get("text") or attrs.get("template_text") or "").lower()
    if "8va" in known_text or (x1 - x0) <= 4.2 * space:
        return clamp_bbox([x0 - 0.7 * space, y0 - 0.8 * space, x1 + 1.0 * space, y1 + 0.8 * space], *image_size)
    # Left expansion catches D.S. before "al Coda"; right expansion catches short split words.
    pad_left = 6.0 * space
    pad_right = 2.0 * space
    pad_y = 1.4 * space
    return clamp_bbox([x0 - pad_left, y0 - pad_y, x1 + pad_right, y1 + pad_y], *image_size)


def instrument_label_regions(staves: list[dict[str, Any]], image_size: tuple[int, int]) -> list[dict[str, Any]]:
    regions = []
    for idx in range(0, len(staves) - 1, 2):
        upper = staves[idx]
        lower = staves[idx + 1]
        space = float(upper.get("space") or lower.get("space") or 16)
        x1 = min(float(upper.get("x0", 90)), float(lower.get("x0", 90))) - 2
        bbox = clamp_bbox(
            [
                0,
                float(upper["y1"]) - 1.1 * space,
                max(60, x1),
                float(lower["y0"]) + 2.2 * space,
            ],
            *image_size,
        )
        if bbox is None:
            continue
        regions.append(
            {
                "id": f"ocr_instrument_{idx // 2:03d}",
                "class": "text_region",
                "bbox": bbox,
                "confidence": 1.0,
                "source": "v2_instrument_region",
                "attributes": {
                    "staff_pair": [upper["index"], lower["index"]],
                    "staff": upper["index"],
                    "text_role": "instrument_label",
                    "staff_space": space,
                },
            }
        )
    return regions


def attach_ocr_to_symbol(symbol: dict[str, Any], ocr_result: dict[str, Any], source_suffix: str = "+ocr") -> dict[str, Any]:
    item = dict(symbol)
    attrs = dict(item.get("attributes") or {})
    if "text" in attrs:
        attrs["template_text"] = attrs["text"]
    attrs["text"] = ocr_result["text"]
    attrs["ocr"] = ocr_result
    item["attributes"] = attrs
    item["confidence"] = max(float(item.get("confidence", 0.0)), float(ocr_result.get("confidence", 0.0)))
    if source_suffix not in item.get("source", ""):
        item["source"] = f"{item.get('source', 'unknown')}{source_suffix}"
    return item


def draw_ocr_overlay(image: Image.Image, symbols: list[dict[str, Any]], out_path: Path) -> None:
    canvas = image.convert("RGB")
    draw = ImageDraw.Draw(canvas, "RGBA")
    for symbol in symbols:
        if symbol["class"] != "text_region":
            continue
        attrs = symbol.get("attributes") or {}
        text = attrs.get("text")
        ocr = attrs.get("ocr")
        if not text and not ocr:
            continue
        x0, y0, x1, y1 = symbol["bbox"]
        color = color_for_class("text_region")
        draw.rectangle([x0, y0, x1, y1], outline=color, width=3)
        label = str(text or ocr.get("text"))
        label_box = [x0, max(0, y0 - 18), x0 + max(50, 8 * len(label) + 8), y0]
        draw.rectangle(label_box, fill=(255, 255, 255, 230), outline=color)
        draw.text((label_box[0] + 3, label_box[1] + 3), label, fill=color)
        if ocr:
            ox0, oy0, ox1, oy1 = ocr["bbox"]
            draw.rectangle([ox0, oy0, ox1, oy1], outline=(0, 170, 60, 180), width=2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run OCR on score text regions and attach results to V2 symbols.")
    parser.add_argument("--symbols-json", type=Path, default=Path("outputs/v2_neural_symbols/symbols_v2_with_sam2.json"))
    parser.add_argument("--out-json", type=Path, default=Path("outputs/v2_ocr/symbols_v2_ocr.json"))
    parser.add_argument("--overlay", type=Path, default=Path("outputs/v2_ocr/ocr_overlay.png"))
    parser.add_argument("--scale", type=int, default=3)
    parser.add_argument("--min-confidence", type=float, default=0.45)
    args = parser.parse_args()

    payload = read_json(args.symbols_json)
    image = Image.open(payload["input"]).convert("RGB")
    ocr = load_ocr_engine()
    symbols = payload.get("symbols", [])
    updated = []
    ocr_results = []

    for symbol in symbols:
        if symbol.get("class") != "text_region":
            updated.append(symbol)
            continue
        bbox = expanded_text_bbox(symbol, image.size)
        if bbox is None:
            updated.append(symbol)
            continue
        result = run_ocr_on_bbox(ocr, image, bbox, scale=args.scale)
        if result["confidence"] >= args.min_confidence and result["text"]:
            item = attach_ocr_to_symbol(symbol, result)
            updated.append(item)
            ocr_results.append({"symbol_id": item["id"], **result})
        else:
            updated.append(symbol)

    for region in instrument_label_regions(payload.get("staves", []), image.size):
        result = run_ocr_on_bbox(ocr, image, region["bbox"], scale=args.scale)
        if result["confidence"] < args.min_confidence or normalize_score_text(result["raw_text"]) != "Pno.":
            continue
        item = attach_ocr_to_symbol(region, result)
        updated.append(item)
        ocr_results.append({"symbol_id": item["id"], **result})

    payload["symbols"] = updated
    payload["stack"] = dict(payload.get("stack") or {})
    payload["stack"]["ocr"] = "rapidocr_onnxruntime"
    payload["ocr_results"] = ocr_results
    payload["counts"] = dict(payload.get("counts") or {})
    payload["counts"]["text_region"] = sum(1 for item in updated if item.get("class") == "text_region")
    payload["ocr_overlay"] = str(args.overlay)
    write_json(args.out_json, payload)
    draw_ocr_overlay(image, updated, args.overlay)
    print(
        json.dumps(
            {
                "out_json": str(args.out_json),
                "overlay": str(args.overlay),
                "ocr_results": len(ocr_results),
                "texts": [
                    {
                        "symbol_id": item["symbol_id"],
                        "text": item["text"],
                        "confidence": item["confidence"],
                        "raw_text": item["raw_text"],
                    }
                    for item in ocr_results
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
