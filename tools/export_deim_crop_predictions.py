from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

import torch
import torchvision
from PIL import Image

from export_deim_predictions import load_class_names, load_deim_model, predict
from omr_v2_common import clamp_bbox, count_by_class, draw_symbol_overlay, run_v1_symbol_pipeline, write_json


def grand_staff_groups(staves: list[Any]) -> list[list[Any]]:
    groups: list[list[Any]] = []
    index = 0
    while index < len(staves):
        if index + 1 < len(staves):
            groups.append([staves[index], staves[index + 1]])
            index += 2
        else:
            groups.append([staves[index]])
            index += 1
    return groups


def system_groups(staves: list[Any], gap_spaces: float) -> list[list[Any]]:
    """Group staves into page systems using large vertical gaps as boundaries."""
    ordered = sorted(staves, key=lambda staff: (float(staff.y0), float(staff.x0)))
    if not ordered:
        return []
    spaces = [float(staff.space) for staff in ordered if float(staff.space) > 0]
    fallback_space = statistics.median(spaces) if spaces else 12.0
    groups: list[list[Any]] = [[ordered[0]]]
    for previous, current in zip(ordered, ordered[1:]):
        gap = float(current.y0) - float(previous.y1)
        local_space = max(float(getattr(previous, "space", fallback_space) or fallback_space), float(getattr(current, "space", fallback_space) or fallback_space))
        if gap > gap_spaces * local_space:
            groups.append([current])
        else:
            groups[-1].append(current)
    return groups


def base_crop_boxes(
    staves: list[Any],
    image_size: tuple[int, int],
    crop_mode: str,
    horizontal_pad_spaces: float,
    vertical_pad_spaces: float,
    system_gap_spaces: float,
) -> list[dict[str, Any]]:
    width, height = image_size
    if crop_mode == "staff":
        groups = [[staff] for staff in staves]
    elif crop_mode == "grand_staff":
        groups = grand_staff_groups(staves)
    else:
        groups = system_groups(staves, system_gap_spaces)

    boxes: list[dict[str, Any]] = []
    for group_idx, group in enumerate(groups):
        space = float(max(staff.space for staff in group))
        x0 = min(staff.x0 for staff in group) - horizontal_pad_spaces * space
        x1 = max(staff.x1 for staff in group) + horizontal_pad_spaces * space
        y0 = min(staff.y0 for staff in group) - vertical_pad_spaces * space
        y1 = max(staff.y1 for staff in group) + vertical_pad_spaces * space
        bbox = clamp_bbox([x0, y0, x1, y1], width, height)
        if bbox is None:
            continue
        boxes.append(
            {
                "id": f"{crop_mode}_{group_idx:03d}",
                "staff_indices": [int(staff.index) for staff in group],
                "bbox": bbox,
            }
        )
    return boxes


def tile_boxes(base_boxes: list[dict[str, Any]], image_size: tuple[int, int], tiles: int, overlap: float) -> list[dict[str, Any]]:
    if tiles <= 1:
        return base_boxes
    width, height = image_size
    tiled: list[dict[str, Any]] = []
    for base in base_boxes:
        x0, y0, x1, y1 = base["bbox"]
        base_w = x1 - x0
        step = base_w / tiles
        tile_w = step * (1.0 + max(0.0, overlap))
        for tile_idx in range(tiles):
            tx0 = x0 + tile_idx * step - (tile_w - step) * 0.5
            tx1 = tx0 + tile_w
            bbox = clamp_bbox([tx0, y0, tx1, y1], width, height)
            if bbox is None:
                continue
            item = dict(base)
            item["id"] = f"{base['id']}_tile{tile_idx:02d}"
            item["base_bbox"] = base["bbox"]
            item["tile_index"] = tile_idx
            item["bbox"] = bbox
            tiled.append(item)
    return tiled


def dedupe_symbols(symbols: list[dict[str, Any]], dedupe_iou: float, max_detections: int) -> list[dict[str, Any]]:
    if not symbols:
        return []
    buckets: dict[str, list[int]] = {}
    for idx, symbol in enumerate(symbols):
        detector_class = str(symbol.get("attributes", {}).get("detector_class") or symbol["class"])
        buckets.setdefault(detector_class, []).append(idx)

    keep_indices: list[int] = []
    for indices in buckets.values():
        boxes = torch.tensor([symbols[idx]["bbox"] for idx in indices], dtype=torch.float32)
        scores = torch.tensor([float(symbols[idx].get("confidence", 1.0)) for idx in indices], dtype=torch.float32)
        kept_local = torchvision.ops.nms(boxes, scores, dedupe_iou)
        keep_indices.extend(indices[int(local_idx)] for local_idx in kept_local.tolist())

    keep_indices.sort(key=lambda idx: float(symbols[idx].get("confidence", 1.0)), reverse=True)
    if max_detections > 0:
        keep_indices = keep_indices[:max_detections]

    deduped: list[dict[str, Any]] = []
    for out_idx, idx in enumerate(keep_indices):
        item = dict(symbols[idx])
        item["id"] = f"crop_deim_{out_idx:06d}"
        deduped.append(item)
    return deduped


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DEIM-D-FINE inference on staff/system crops and remap boxes to the full page.")
    parser.add_argument("--deim-root", type=Path, default=Path(".local-tools/DEIM-main"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--pdf-dpi", type=int, default=220)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--overlay", type=Path, required=True)
    parser.add_argument("--crop-dir", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--nms-iou", type=float, default=0.50)
    parser.add_argument("--dedupe-iou", type=float, default=0.35)
    parser.add_argument("--max-detections-per-crop", type=int, default=500)
    parser.add_argument("--max-detections", type=int, default=2500)
    parser.add_argument("--taxonomy", choices=("base", "expanded", "expanded_clean"), default="base")
    parser.add_argument("--class-names-json", type=Path)
    parser.add_argument("--crop-mode", choices=("system", "grand_staff", "staff"), default="staff")
    parser.add_argument("--tiles", type=int, default=3)
    parser.add_argument("--tile-overlap", type=float, default=0.25)
    parser.add_argument("--horizontal-pad-spaces", type=float, default=2.0)
    parser.add_argument("--vertical-pad-spaces", type=float, default=4.0)
    parser.add_argument("--system-gap-spaces", type=float, default=7.0)
    parser.add_argument(
        "--resize-mode",
        choices=("auto", "stretch", "keep-aspect", "original"),
        default="stretch",
        help="DEIM preprocessing for each crop. Stretch intentionally magnifies thin staff crops.",
    )
    parser.add_argument("--input-size", type=int, default=640)
    args = parser.parse_args()

    v1_result = run_v1_symbol_pipeline(args.input, pdf_dpi=args.pdf_dpi)
    page: Image.Image = v1_result["image"].convert("RGB")
    page_w, page_h = page.size
    class_names = load_class_names(args.class_names_json, args.taxonomy)
    model = load_deim_model(args.deim_root, args.config, args.checkpoint, args.device)

    crop_dir = args.crop_dir or args.out_json.parent / "crops"
    crop_dir.mkdir(parents=True, exist_ok=True)

    bases = base_crop_boxes(
        v1_result["staves"],
        page.size,
        args.crop_mode,
        args.horizontal_pad_spaces,
        args.vertical_pad_spaces,
        args.system_gap_spaces,
    )
    crops = tile_boxes(bases, page.size, args.tiles, args.tile_overlap)
    all_symbols: list[dict[str, Any]] = []
    crop_summaries: list[dict[str, Any]] = []

    for crop_idx, crop in enumerate(crops):
        x0, y0, x1, y1 = crop["bbox"]
        crop_image = page.crop((x0, y0, x1, y1))
        crop_path = crop_dir / f"{crop['id']}.png"
        crop_image.save(crop_path)
        _, crop_symbols = predict(
            model,
            crop_path,
            args.device,
            args.threshold,
            args.nms_iou,
            args.max_detections_per_crop,
            class_names,
            resize_mode=args.resize_mode,
            input_size=args.input_size,
        )
        remapped = []
        for local_idx, symbol in enumerate(crop_symbols):
            bbox = symbol["bbox"]
            mapped_bbox = clamp_bbox([bbox[0] + x0, bbox[1] + y0, bbox[2] + x0, bbox[3] + y0], page_w, page_h)
            if mapped_bbox is None:
                continue
            item = dict(symbol)
            item["id"] = f"crop{crop_idx:03d}_{local_idx:04d}"
            item["bbox"] = mapped_bbox
            item["source"] = "deim_dfine_crop"
            attributes = dict(item.get("attributes") or {})
            attributes.update(
                {
                    "crop_id": crop["id"],
                    "crop_bbox": crop["bbox"],
                    "crop_mode": args.crop_mode,
                    "staff_indices": crop["staff_indices"],
                }
            )
            item["attributes"] = attributes
            remapped.append(item)
        all_symbols.extend(remapped)
        crop_summaries.append(
            {
                "id": crop["id"],
                "bbox": crop["bbox"],
                "staff_indices": crop["staff_indices"],
                "raw_symbols": len(crop_symbols),
                "remapped_symbols": len(remapped),
                "counts": count_by_class(remapped),
            }
        )

    symbols = dedupe_symbols(all_symbols, args.dedupe_iou, args.max_detections)
    draw_symbol_overlay(page, symbols, args.overlay)
    payload = {
        "version": "v2_deim_dfine_crop_predictions",
        "input": str(v1_result["input"]),
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "threshold": args.threshold,
        "nms_iou": args.nms_iou,
        "dedupe_iou": args.dedupe_iou,
        "taxonomy": args.taxonomy,
        "class_names": class_names,
        "crop_mode": args.crop_mode,
        "tiles": args.tiles,
        "tile_overlap": args.tile_overlap,
        "system_gap_spaces": args.system_gap_spaces,
        "resize_mode": args.resize_mode,
        "input_size": args.input_size,
        "base_crops": bases,
        "crops": crop_summaries,
        "symbols_before_dedupe": len(all_symbols),
        "symbols": symbols,
        "counts": count_by_class(symbols),
        "overlay": str(args.overlay),
    }
    write_json(args.out_json, payload)
    print(
        json.dumps(
            {
                "out_json": str(args.out_json),
                "overlay": str(args.overlay),
                "base_crops": len(bases),
                "crops": len(crops),
                "symbols_before_dedupe": len(all_symbols),
                "symbols": len(symbols),
                "counts": payload["counts"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
