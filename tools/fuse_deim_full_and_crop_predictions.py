from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torchvision
from PIL import Image

from omr_v2_common import clamp_bbox, count_by_class, draw_symbol_overlay, read_json, write_json


DEFAULT_SUPPLEMENT_CLASSES = {
    "stem",
    "beam",
    "ledger_line",
}


DEFAULT_CLASS_THRESHOLDS = {
    "filled_notehead": 0.55,
    "open_notehead": 0.50,
    "stem": 0.45,
    "beam": 0.75,
    "ledger_line": 0.45,
}


def parse_classes(value: str) -> set[str]:
    if value.strip().lower() == "all":
        return set()
    return {item.strip() for item in value.split(",") if item.strip()}


def parse_thresholds(value: str | None) -> dict[str, float]:
    thresholds = dict(DEFAULT_CLASS_THRESHOLDS)
    if not value:
        return thresholds
    for item in value.split(","):
        if not item.strip():
            continue
        if "=" not in item:
            raise ValueError(f"Invalid class threshold item: {item!r}. Expected class=value.")
        name, raw_threshold = item.split("=", 1)
        thresholds[name.strip()] = float(raw_threshold)
    return thresholds


def symbol_class(symbol: dict[str, Any]) -> str:
    return str(symbol.get("class") or "")


def detector_class(symbol: dict[str, Any]) -> str:
    attributes = symbol.get("attributes") or {}
    return str(attributes.get("detector_class") or symbol_class(symbol))


def symbol_score(symbol: dict[str, Any]) -> float:
    return float(symbol.get("confidence", symbol.get("score", 1.0)) or 0.0)


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


def source_priority(symbol: dict[str, Any]) -> float:
    source = str(symbol.get("source") or "")
    # Full-page detections are the anchor result. Crop detections only win when
    # they are meaningfully higher confidence for the same class/bbox.
    return 0.03 if "crop" not in source else 0.0


def image_size_from_payload(payloads: list[dict[str, Any]], input_path: Path | None) -> tuple[int, int]:
    for payload in payloads:
        inference = payload.get("inference")
        if isinstance(inference, dict) and isinstance(inference.get("image_size"), list):
            width, height = inference["image_size"][:2]
            return int(width), int(height)
    if input_path is not None:
        with Image.open(input_path) as image:
            return image.size
    raise ValueError("Cannot determine image size; pass --input for overlay/bbox validation.")


def normalized_symbols(payload: dict[str, Any], image_size: tuple[int, int], source_tag: str) -> list[dict[str, Any]]:
    width, height = image_size
    result: list[dict[str, Any]] = []
    for idx, symbol in enumerate(payload.get("symbols", [])):
        bbox = clamp_bbox(symbol.get("bbox"), width, height)
        if bbox is None:
            continue
        item = dict(symbol)
        item["bbox"] = bbox
        item.setdefault("id", f"{source_tag}_{idx:06d}")
        item.setdefault("source", source_tag)
        result.append(item)
    return result


def filter_crop_symbols(
    crop_symbols: list[dict[str, Any]],
    full_symbols: list[dict[str, Any]],
    supplement_classes: set[str],
    thresholds: dict[str, float],
    default_threshold: float,
    require_missing: bool,
    full_overlap_iou: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    full_by_class: dict[str, list[dict[str, Any]]] = {}
    for symbol in full_symbols:
        full_by_class.setdefault(symbol_class(symbol), []).append(symbol)
    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for symbol in crop_symbols:
        cls = symbol_class(symbol)
        if supplement_classes and cls not in supplement_classes:
            item = dict(symbol)
            item["drop_reason"] = "class_not_supplemented"
            dropped.append(item)
            continue
        threshold = thresholds.get(cls, default_threshold)
        if symbol_score(symbol) < threshold:
            item = dict(symbol)
            item["drop_reason"] = "below_supplement_threshold"
            item["supplement_threshold"] = threshold
            dropped.append(item)
            continue
        if require_missing:
            overlapping = [
                full_symbol
                for full_symbol in full_by_class.get(cls, [])
                if bbox_iou(symbol["bbox"], full_symbol["bbox"]) >= full_overlap_iou
            ]
            if overlapping:
                item = dict(symbol)
                item["drop_reason"] = "covered_by_full_detection"
                item["full_overlap_iou"] = max(bbox_iou(symbol["bbox"], full_symbol["bbox"]) for full_symbol in overlapping)
                dropped.append(item)
                continue
        item = dict(symbol)
        attributes = dict(item.get("attributes") or {})
        attributes["supplement_threshold"] = threshold
        attributes["supplement_source"] = "crop"
        item["attributes"] = attributes
        item["source"] = f"{item.get('source', 'deim_dfine_crop')}+supplement"
        kept.append(item)
    return kept, dropped


def nms_symbols(symbols: list[dict[str, Any]], nms_iou: float, max_detections: int, id_prefix: str) -> list[dict[str, Any]]:
    if not symbols:
        return []
    buckets: dict[str, list[int]] = {}
    for idx, symbol in enumerate(symbols):
        buckets.setdefault(detector_class(symbol), []).append(idx)

    keep_indices: list[int] = []
    for indices in buckets.values():
        boxes = torch.tensor([symbols[idx]["bbox"] for idx in indices], dtype=torch.float32)
        scores = torch.tensor([symbol_score(symbols[idx]) + source_priority(symbols[idx]) for idx in indices], dtype=torch.float32)
        local_keep = torchvision.ops.nms(boxes, scores, nms_iou)
        keep_indices.extend(indices[int(local_idx)] for local_idx in local_keep.tolist())

    keep_indices.sort(key=lambda idx: symbol_score(symbols[idx]) + source_priority(symbols[idx]), reverse=True)
    if max_detections > 0:
        keep_indices = keep_indices[:max_detections]

    fused: list[dict[str, Any]] = []
    for out_idx, idx in enumerate(keep_indices):
        item = dict(symbols[idx])
        old_id = str(item.get("id", ""))
        item["id"] = f"{id_prefix}_{out_idx:06d}"
        attributes = dict(item.get("attributes") or {})
        attributes["pre_fuse_id"] = old_id
        item["attributes"] = attributes
        fused.append(item)
    return fused


def anchored_fuse(full_symbols: list[dict[str, Any]], crop_kept: list[dict[str, Any]], nms_iou: float, max_detections: int) -> list[dict[str, Any]]:
    crop_deduped = nms_symbols(crop_kept, nms_iou, max_detections=0, id_prefix="fused_crop")
    fused: list[dict[str, Any]] = []
    for idx, symbol in enumerate(full_symbols):
        item = dict(symbol)
        item["id"] = f"fused_full_{idx:06d}"
        attributes = dict(item.get("attributes") or {})
        attributes["pre_fuse_id"] = symbol.get("id")
        attributes["fuse_role"] = "full_anchor"
        item["attributes"] = attributes
        fused.append(item)
    fused.extend(crop_deduped)
    fused.sort(key=lambda symbol: symbol_score(symbol), reverse=True)
    if max_detections > 0:
        fused = fused[:max_detections]
    return fused


def main() -> None:
    parser = argparse.ArgumentParser(description="Fuse full-page DEIM detections with high-confidence crop supplements.")
    parser.add_argument("--full-json", type=Path, required=True)
    parser.add_argument("--crop-json", type=Path, required=True)
    parser.add_argument("--input", type=Path, help="Original image path, used for overlay and bbox validation.")
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--overlay", type=Path)
    parser.add_argument("--supplement-classes", default=",".join(sorted(DEFAULT_SUPPLEMENT_CLASSES)))
    parser.add_argument("--class-thresholds", help="Comma-separated overrides, for example filled_notehead=0.5,beam=0.6")
    parser.add_argument("--default-threshold", type=float, default=0.55)
    parser.add_argument("--require-missing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--full-overlap-iou", type=float, default=0.10)
    parser.add_argument("--nms-iou", type=float, default=0.45)
    parser.add_argument("--max-detections", type=int, default=2500)
    args = parser.parse_args()

    full_payload = read_json(args.full_json)
    crop_payload = read_json(args.crop_json)
    image_size = image_size_from_payload([full_payload, crop_payload], args.input)
    full_symbols = normalized_symbols(full_payload, image_size, "deim_dfine_full")
    crop_symbols = normalized_symbols(crop_payload, image_size, "deim_dfine_crop")
    supplement_classes = parse_classes(args.supplement_classes)
    thresholds = parse_thresholds(args.class_thresholds)
    crop_kept, crop_dropped = filter_crop_symbols(
        crop_symbols,
        full_symbols,
        supplement_classes,
        thresholds,
        args.default_threshold,
        args.require_missing,
        args.full_overlap_iou,
    )

    fused_symbols = anchored_fuse(full_symbols, crop_kept, args.nms_iou, args.max_detections)
    payload = {
        "version": "v2_deim_dfine_full_crop_supplement_predictions",
        "input": str(args.input) if args.input else full_payload.get("input") or crop_payload.get("input"),
        "full_json": str(args.full_json),
        "crop_json": str(args.crop_json),
        "full_symbols": len(full_symbols),
        "crop_symbols": len(crop_symbols),
        "crop_supplement_kept": len(crop_kept),
        "crop_supplement_dropped": len(crop_dropped),
        "supplement_classes": sorted(supplement_classes) if supplement_classes else "all",
        "class_thresholds": thresholds,
        "default_threshold": args.default_threshold,
        "require_missing": args.require_missing,
        "full_overlap_iou": args.full_overlap_iou,
        "nms_iou": args.nms_iou,
        "max_detections": args.max_detections,
        "symbols": fused_symbols,
        "counts": count_by_class(fused_symbols),
        "supplement_counts": count_by_class(crop_kept),
        "dropped_counts": count_by_class(crop_dropped),
    }
    if args.overlay:
        image_path = Path(args.input or payload["input"])
        image = Image.open(image_path).convert("RGB")
        try:
            draw_symbol_overlay(image, fused_symbols, args.overlay)
        finally:
            image.close()
        payload["overlay"] = str(args.overlay)
    write_json(args.out_json, payload)
    print(
        json.dumps(
            {
                "out_json": str(args.out_json),
                "full_symbols": len(full_symbols),
                "crop_symbols": len(crop_symbols),
                "crop_supplement_kept": len(crop_kept),
                "symbols": len(fused_symbols),
                "counts": payload["counts"],
                "supplement_counts": payload["supplement_counts"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
