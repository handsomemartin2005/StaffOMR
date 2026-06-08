from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from omr_v2_common import (
    V2_EXPANDED_CLASS_NAMES,
    V2_ID_TO_CLASS,
    V2_CLASS_TO_ID,
    V2_CLASS_NAMES,
    base_class_for_symbol_class,
    clamp_bbox,
    count_by_class,
    draw_symbol_overlay,
    enrich_symbols_with_staff_geometry,
    read_json,
    run_v1_symbol_pipeline,
    v1_result_to_weak_symbols,
    write_json,
)


def xywh_to_xyxy(bbox: list[float]) -> list[float]:
    x, y, w, h = bbox
    return [x, y, x + w, y + h]


def class_from_prediction(item: dict[str, Any], categories: dict[int, str]) -> str | None:
    if "class" in item:
        return item["class"]
    if "category" in item:
        return item["category"]
    if "category_id" in item:
        category_id = int(item["category_id"])
        return categories.get(category_id, V2_ID_TO_CLASS.get(category_id))
    return None


def load_detector_symbols(
    path: Path,
    image_size: tuple[int, int],
    bbox_format: str,
    min_confidence: float,
) -> list[dict[str, Any]]:
    payload = read_json(path)
    categories: dict[int, str] = dict(V2_ID_TO_CLASS)
    raw_items: list[dict[str, Any]]

    if isinstance(payload, list):
        raw_items = payload
    elif isinstance(payload, dict) and "symbols" in payload:
        raw_items = payload["symbols"]
    elif isinstance(payload, dict) and "annotations" in payload:
        categories.update({int(item["id"]): item["name"] for item in payload.get("categories", [])})
        raw_items = payload["annotations"]
    else:
        raise ValueError(f"Unsupported detector prediction format: {path}")

    symbols = []
    for idx, item in enumerate(raw_items):
        confidence = float(item.get("confidence", item.get("score", 1.0)))
        if confidence < min_confidence:
            continue
        symbol_class = class_from_prediction(item, categories)
        if symbol_class is None:
            continue
        attributes = dict(item.get("attributes", {}))
        detector_class = str(attributes.get("detector_class") or symbol_class)
        if symbol_class in V2_CLASS_TO_ID:
            output_class = symbol_class
        elif symbol_class in V2_EXPANDED_CLASS_NAMES:
            output_class = base_class_for_symbol_class(symbol_class) or symbol_class
            attributes.setdefault("detector_class", detector_class)
            if output_class != symbol_class:
                attributes.setdefault("fine_class", symbol_class)
        else:
            continue
        raw_bbox = item.get("bbox")
        if raw_bbox is None:
            continue
        xyxy = xywh_to_xyxy(raw_bbox) if bbox_format == "xywh" else raw_bbox
        bbox = clamp_bbox(xyxy, *image_size)
        if bbox is None:
            continue
        symbols.append(
            {
                "id": item.get("id", f"det_{idx:06d}"),
                "class": output_class,
                "bbox": bbox,
                "confidence": confidence,
                "source": item.get("source", "deim_dfine_external"),
                "attributes": attributes,
            }
        )
    return symbols


def attach_sam2_mask_metadata(symbols: list[dict[str, Any]], mask_json: Path | None) -> list[dict[str, Any]]:
    if mask_json is None:
        return symbols
    payload = read_json(mask_json)
    masks = payload.get("masks", payload if isinstance(payload, dict) else [])
    by_id = {str(item.get("symbol_id")): item for item in masks if item.get("symbol_id") is not None}
    enriched = []
    for symbol in symbols:
        item = dict(symbol)
        mask = by_id.get(str(symbol["id"]))
        if mask is not None:
            item["mask"] = mask
            item["source"] = f"{item.get('source', 'unknown')}+sam2"
        enriched.append(item)
    return enriched


def bbox_iou(a: list[int], b: list[int]) -> float:
    x0 = max(a[0], b[0])
    y0 = max(a[1], b[1])
    x1 = min(a[2], b[2])
    y1 = min(a[3], b[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    if inter <= 0:
        return 0.0
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def merge_rule_symbols(
    detector_symbols: list[dict[str, Any]],
    v1_symbols: list[dict[str, Any]],
    rule_classes: set[str],
    rule_fallback_iou: float,
    fusion_preference: str,
) -> list[dict[str, Any]]:
    if not rule_classes:
        return detector_symbols
    if fusion_preference == "rule":
        merged = []
        for symbol in v1_symbols:
            if symbol["class"] not in rule_classes:
                continue
            item = dict(symbol)
            item["id"] = f"rule_{item['id']}"
            item["source"] = f"retained_rule_{item.get('source', 'v1')}"
            merged.append(item)
        for symbol in detector_symbols:
            if symbol["class"] in rule_classes:
                continue
            merged.append(symbol)
        return merged

    merged = list(detector_symbols)
    for symbol in v1_symbols:
        if symbol["class"] not in rule_classes:
            continue
        if any(
            existing["class"] == symbol["class"] and bbox_iou(existing["bbox"], symbol["bbox"]) >= rule_fallback_iou
            for existing in merged
        ):
            continue
        item = dict(symbol)
        item["id"] = f"rule_{item['id']}"
        item["source"] = f"retained_rule_{item.get('source', 'v1')}"
        merged.append(item)
    return merged


def parse_class_list(value: str) -> set[str]:
    if value.strip().lower() == "all":
        return set(V2_CLASS_NAMES)
    return {item.strip() for item in value.split(",") if item.strip()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the V2 symbol-recognition pipeline shell.")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--pdf-dpi", type=int, default=220)
    parser.add_argument(
        "--detector-predictions",
        type=Path,
        help="Optional DEIM-D-FINE prediction JSON/COCO. Defaults to V1 weak detector output.",
    )
    parser.add_argument("--bbox-format", choices=("xywh", "xyxy"), default="xywh")
    parser.add_argument("--min-confidence", type=float, default=0.05)
    parser.add_argument(
        "--sam2-mask-json",
        type=Path,
        help="Optional SAM2 mask metadata keyed by symbol_id. SAM2 execution is intentionally external for now.",
    )
    parser.add_argument(
        "--merge-rule-classes",
        default="text_region,barline",
        help="Comma-separated V1 rule classes to retain when detector predictions are supplied.",
    )
    parser.add_argument("--rule-fallback-iou", type=float, default=0.50)
    parser.add_argument(
        "--fusion-preference",
        choices=("detector", "rule"),
        default="detector",
        help="Use detector output as primary, or use retained rule classes as primary fallback.",
    )
    parser.add_argument("--out-json", type=Path, default=Path("outputs/v2_neural_symbols/symbols_v2.json"))
    parser.add_argument("--overlay", type=Path, default=Path("outputs/v2_neural_symbols/overlay_v2.png"))
    args = parser.parse_args()

    v1_result = run_v1_symbol_pipeline(args.input, pdf_dpi=args.pdf_dpi)
    image = v1_result["image"]
    v1_symbols = v1_result_to_weak_symbols(v1_result)
    retained_rule_classes: set[str] = set()
    if args.detector_predictions is None:
        detector_name = "v1_weak_detector_until_deim_dfine_is_connected"
        symbols = v1_symbols
    else:
        detector_name = "deim_dfine_external_predictions"
        symbols = load_detector_symbols(args.detector_predictions, image.size, args.bbox_format, args.min_confidence)
        retained_rule_classes = parse_class_list(args.merge_rule_classes)
        symbols = merge_rule_symbols(
            symbols,
            v1_symbols,
            retained_rule_classes,
            args.rule_fallback_iou,
            args.fusion_preference,
        )

    symbols = enrich_symbols_with_staff_geometry(symbols, v1_result["staves"])
    symbols = attach_sam2_mask_metadata(symbols, args.sam2_mask_json)
    draw_symbol_overlay(image, symbols, args.overlay)

    payload = {
        "version": "v2_symbol_recognition",
        "stack": {
            "staff_geometry": "v1_rules",
            "detector": detector_name,
            "retained_rule_classes": sorted(retained_rule_classes),
            "fusion_preference": args.fusion_preference,
            "segmentation": "sam2_external_mask_metadata" if args.sam2_mask_json else "not_run",
            "embedding": "dinov2_external_embedding_stage",
        },
        "input": str(v1_result["input"]),
        "threshold": v1_result["threshold"],
        "staff_space": v1_result["staff_space"],
        "staves": [
            {
                "index": staff.index,
                "lines": staff.lines,
                "x0": staff.x0,
                "x1": staff.x1,
                "y0": staff.y0,
                "y1": staff.y1,
                "space": staff.space,
                "scan_x0": staff.scan_x0,
                "clef_bbox": staff.clef_bbox,
                "clef_type": staff.clef_type,
            }
            for staff in v1_result["staves"]
        ],
        "symbols": symbols,
        "counts": count_by_class(symbols),
        "overlay": str(args.overlay),
    }
    write_json(args.out_json, payload)
    print(
        json.dumps(
            {
                "input": payload["input"],
                "symbols": len(symbols),
                "counts": payload["counts"],
                "out_json": str(args.out_json),
                "overlay": str(args.overlay),
                "detector": detector_name,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
