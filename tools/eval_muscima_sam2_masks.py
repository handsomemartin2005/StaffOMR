from __future__ import annotations

import argparse
import json
import re
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parent.parent
TOOLS = ROOT / "tools"
for path in (TOOLS,):
    value = str(path)
    while value in sys.path:
        sys.path.remove(value)
    sys.path.insert(0, value)

ANNOTATIONS = ROOT / "data/muscima_pp/MUSCIMA-pp_v2.0/v2.0/data/annotations"
IMAGES = ROOT / "data/muscima_pp/CVC_MUSCIMA_PP_Annotated-Images/fulls"
DEFAULT_CHECKPOINT = ROOT / "outputs/models/sam2/sam2.1_hiera_tiny.pt"
DEFAULT_CLASSES = ("noteheadFull", "noteheadHalf", "stem", "beam")


def decode_rle(mask_text: str, height: int, width: int) -> np.ndarray:
    values: list[np.ndarray] = []
    total = 0
    for token in mask_text.split():
        value_text, count_text = token.split(":", 1)
        count = int(count_text)
        total += count
        values.append(np.full(count, int(value_text), dtype=np.uint8))
    if total != height * width:
        raise ValueError(f"RLE length {total} != mask size {height}x{width}")
    return np.concatenate(values).reshape(height, width).astype(bool)


def _dilate(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    result = mask.copy()
    for _ in range(radius):
        padded = np.pad(result, 1)
        neighbors = [
            padded[dy : dy + result.shape[0], dx : dx + result.shape[1]]
            for dy in range(3)
            for dx in range(3)
        ]
        result = np.logical_or.reduce(neighbors)
    return result


def _erode(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1, constant_values=False)
    neighbors = [
        padded[dy : dy + mask.shape[0], dx : dx + mask.shape[1]]
        for dy in range(3)
        for dx in range(3)
    ]
    return np.logical_and.reduce(neighbors)


def mask_metrics(pred: np.ndarray, gold: np.ndarray, staff: np.ndarray | None = None) -> dict[str, float]:
    pred = pred.astype(bool)
    gold = gold.astype(bool)
    intersection = int(np.logical_and(pred, gold).sum())
    union = int(np.logical_or(pred, gold).sum())
    pred_area = int(pred.sum())
    gold_area = int(gold.sum())
    pred_boundary = np.logical_xor(pred, _erode(pred))
    gold_boundary = np.logical_xor(gold, _erode(gold))
    boundary_precision = (
        np.logical_and(pred_boundary, _dilate(gold_boundary)).sum() / max(1, pred_boundary.sum())
    )
    boundary_recall = (
        np.logical_and(gold_boundary, _dilate(pred_boundary)).sum() / max(1, gold_boundary.sum())
    )
    boundary_f1 = (
        2.0 * boundary_precision * boundary_recall / (boundary_precision + boundary_recall)
        if boundary_precision + boundary_recall
        else 0.0
    )
    outside = np.logical_and(pred, np.logical_not(gold))
    staff_leak = 0.0
    if staff is not None:
        staff_leak = float(np.logical_and(outside, staff).sum()) / max(1, pred_area)
    return {
        "iou": intersection / union if union else 1.0,
        "boundary_f1": float(boundary_f1),
        "leakage": float(outside.sum()) / max(1, pred_area),
        "staff_leakage": staff_leak,
        "area_ratio": pred_area / max(1, gold_area),
    }


def parse_nodes(path: Path) -> list[dict[str, Any]]:
    nodes = []
    for node in ET.parse(path).getroot().findall("Node"):
        height = int(node.findtext("Height", "0"))
        width = int(node.findtext("Width", "0"))
        nodes.append(
            {
                "id": node.findtext("Id", ""),
                "class": node.findtext("ClassName", ""),
                "top": int(node.findtext("Top", "0")),
                "left": int(node.findtext("Left", "0")),
                "height": height,
                "width": width,
                "mask": decode_rle(node.findtext("Mask", ""), height, width),
            }
        )
    return nodes


def annotation_writer_id(path: Path) -> int:
    match = re.search(r"W-(\d+)", path.stem)
    if not match:
        raise ValueError(f"Cannot parse writer id: {path.name}")
    return int(match.group(1))


def parse_writer_spec(value: str) -> set[int]:
    result: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            left, right = part.split("-", 1)
            result.update(range(int(left), int(right) + 1))
        else:
            result.add(int(part))
    return result


def place_local(node: dict[str, Any], shape: tuple[int, int]) -> np.ndarray:
    result = np.zeros(shape, dtype=bool)
    top, left = node["top"], node["left"]
    bottom = min(shape[0], top + node["height"])
    right = min(shape[1], left + node["width"])
    result[top:bottom, left:right] = node["mask"][: bottom - top, : right - left]
    return result


def aggregate(rows: list[dict[str, Any]], source: str, class_name: str | None = None) -> dict[str, Any]:
    selected = [
        row for row in rows if row["source"] == source and (class_name is None or row["class"] == class_name)
    ]
    metrics = ("iou", "boundary_f1", "leakage", "staff_leakage", "area_ratio")
    return {
        "instances": len(selected),
        **{
            metric: float(np.mean([row[metric] for row in selected])) if selected else 0.0
            for metric in metrics
        },
    }


def main() -> None:
    import refine_masks_sam2

    parser = argparse.ArgumentParser(description="Evaluate zero-shot SAM2 masks against MUSCIMA++ instance masks.")
    parser.add_argument("--pages", type=int, default=2)
    parser.add_argument("--writers", default="", help="Optional writer split, e.g. 41-50.")
    parser.add_argument("--per-class", type=int, default=15)
    parser.add_argument("--classes", default=",".join(DEFAULT_CLASSES))
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--model-cfg", default="configs/sam2.1/sam2.1_hiera_t.yaml")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--decoder-state", type=Path)
    parser.add_argument(
        "--single-mask",
        action="store_true",
        help="Evaluate the single-mask token used by decoder fine-tuning.",
    )
    parser.add_argument("--mask-threshold", type=float, help="Override the SAM2 logit threshold.")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/muscima_sam2_mask_baseline")
    args = parser.parse_args()
    classes = tuple(item.strip() for item in args.classes.split(",") if item.strip())
    all_paths = sorted(ANNOTATIONS.glob("*.xml"))
    if args.writers:
        writers = parse_writer_spec(args.writers)
        all_paths = [path for path in all_paths if annotation_writer_id(path) in writers]
    xml_paths = all_paths[: args.pages]
    if len(xml_paths) != args.pages:
        raise RuntimeError(f"Requested {args.pages} pages, found {len(xml_paths)}")
    predictor = refine_masks_sam2.load_predictor(
        args.checkpoint.resolve(), args.model_cfg, refine_masks_sam2.choose_device(args.device)
    )
    if args.mask_threshold is not None:
        predictor.mask_threshold = float(args.mask_threshold)
    rows: list[dict[str, Any]] = []

    import torch

    if args.decoder_state:
        state = torch.load(args.decoder_state.resolve(), map_location=predictor.device, weights_only=True)
        predictor.model.sam_mask_decoder.load_state_dict(state)
    sam_source = "sam2_adapted" if args.decoder_state else "sam2_zero_shot"

    with torch.inference_mode():
        for xml_path in xml_paths:
            page_id = xml_path.stem
            image_path = IMAGES / f"{page_id}.png"
            image = np.array(Image.open(image_path).convert("RGB"), copy=True)
            nodes = parse_nodes(xml_path)
            page_shape = image.shape[:2]
            staff_mask = np.zeros(page_shape, dtype=bool)
            for node in nodes:
                if node["class"] == "staffLine":
                    staff_mask |= place_local(node, page_shape)
            selected: list[dict[str, Any]] = []
            by_class: defaultdict[str, int] = defaultdict(int)
            for node in nodes:
                if node["class"] in classes and by_class[node["class"]] < args.per_class:
                    selected.append(node)
                    by_class[node["class"]] += 1

            predictor.set_image(image)
            for node in selected:
                box = np.asarray(
                    [node["left"], node["top"], node["left"] + node["width"], node["top"] + node["height"]],
                    dtype=np.float32,
                )
                masks, scores, _ = predictor.predict(box=box, multimask_output=not args.single_mask)
                best = int(np.argmax(scores))
                pred = np.asarray(masks[best], dtype=bool)
                gold = place_local(node, page_shape)
                bbox = np.zeros(page_shape, dtype=bool)
                bbox[node["top"] : node["top"] + node["height"], node["left"] : node["left"] + node["width"]] = True
                for source, candidate in ((sam_source, pred), ("bbox_mask", bbox)):
                    rows.append(
                        {
                            "page": page_id,
                            "symbol_id": node["id"],
                            "class": node["class"],
                            "source": source,
                            "sam2_score": float(scores[best]) if source == "sam2_zero_shot" else None,
                            **mask_metrics(candidate, gold, staff_mask),
                        }
                    )
            print({"page": page_id, "instances": len(selected), "classes": dict(by_class)})

    summary = {
        "protocol": "MUSCIMA++ source-domain GT boxes; full-page SAM2 encoding; selected mask token by SAM2 score",
        "pages": [path.stem for path in xml_paths],
        "classes": list(classes),
        "per_class_per_page": args.per_class,
        "checkpoint": str(args.checkpoint.resolve()),
        "decoder_state": str(args.decoder_state.resolve()) if args.decoder_state else None,
        "multimask_output": not args.single_mask,
        "mask_threshold": predictor.mask_threshold,
        "writers": sorted({annotation_writer_id(path) for path in xml_paths}),
        "overall": {
            source: aggregate(rows, source) for source in ("bbox_mask", sam_source)
        },
        "per_class": {
            class_name: {
                source: aggregate(rows, source, class_name)
                for source in ("bbox_mask", sam_source)
            }
            for class_name in classes
        },
        "publication_status": "small_source_domain_diagnostic_not_target_domain_result",
    }
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "rows.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary["overall"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
