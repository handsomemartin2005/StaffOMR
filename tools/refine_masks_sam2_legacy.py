from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from omr_v2_common import read_json, write_json


def choose_device(requested: str) -> str:
    if requested != "auto":
        return requested
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def mask_to_image(mask: np.ndarray) -> Image.Image:
    return Image.fromarray(np.where(mask, 255, 0).astype(np.uint8), mode="L")


def symbol_center(symbol: dict[str, Any]) -> tuple[float, float]:
    x0, y0, x1, y1 = [float(v) for v in symbol["bbox"]]
    return 0.5 * (x0 + x1), 0.5 * (y0 + y1)


def symbol_staff(symbol: dict[str, Any]) -> int | None:
    staff = (symbol.get("attributes") or {}).get("staff")
    return int(staff) if staff is not None else None


def symbol_staff_space(symbol: dict[str, Any]) -> float:
    attrs = symbol.get("attributes") or {}
    return float(attrs.get("staff_space") or 12.0)


def staff_lines_for_symbol(symbol: dict[str, Any], staves: list[dict[str, Any]]) -> list[float]:
    staff_idx = symbol_staff(symbol)
    if staff_idx is None:
        return []
    for staff in staves:
        if int(staff.get("index", -1)) == staff_idx:
            return [float(y) for y in staff.get("lines", [])]
    return []


def clamp_point(point: tuple[float, float], width: int, height: int) -> list[float]:
    x, y = point
    return [float(max(0.0, min(width - 1.0, x))), float(max(0.0, min(height - 1.0, y)))]


def line_sample_points(bbox: list[float], axis: str, fractions: tuple[float, ...]) -> list[tuple[float, float]]:
    x0, y0, x1, y1 = [float(v) for v in bbox]
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    if axis == "vertical":
        return [(cx, y0 + fraction * (y1 - y0)) for fraction in fractions]
    return [(x0 + fraction * (x1 - x0), cy) for fraction in fractions]


def positive_points(symbol: dict[str, Any]) -> list[tuple[float, float]]:
    cls = str(symbol.get("class") or "")
    bbox = symbol["bbox"]
    if cls in {"stem", "barline"}:
        return line_sample_points(bbox, "vertical", (0.28, 0.50, 0.72))
    if cls in {"beam", "ledger_line", "slur_or_tie"}:
        return line_sample_points(bbox, "horizontal", (0.25, 0.50, 0.75))
    return [symbol_center(symbol)]


def negative_staff_points(
    symbol: dict[str, Any],
    staves: list[dict[str, Any]],
    width: int,
    height: int,
) -> list[tuple[float, float]]:
    cls = str(symbol.get("class") or "")
    if cls in {"filled_notehead", "open_notehead", "ledger_line", "barline"}:
        return []
    staff_lines = staff_lines_for_symbol(symbol, staves)
    if not staff_lines:
        return []
    x0, y0, x1, y1 = [float(v) for v in symbol["bbox"]]
    cx, _ = symbol_center(symbol)
    space = symbol_staff_space(symbol)
    y_margin = max(1.5, 0.25 * space)
    points: list[tuple[float, float]] = []
    for y in staff_lines:
        if not (y0 - y_margin <= y <= y1 + y_margin):
            continue
        if cls == "stem":
            for x in (x0 - 0.55 * space, x1 + 0.55 * space):
                points.append((x, y))
        else:
            for fraction in (0.25, 0.50, 0.75):
                points.append((x0 + fraction * (x1 - x0), y))
    deduped: list[tuple[float, float]] = []
    seen: set[tuple[int, int]] = set()
    for x, y in points:
        clamped = clamp_point((x, y), width, height)
        key = (int(round(clamped[0])), int(round(clamped[1])))
        if key in seen:
            continue
        seen.add(key)
        deduped.append((clamped[0], clamped[1]))
    return deduped


def build_prompt_points(
    symbol: dict[str, Any],
    staves: list[dict[str, Any]],
    image_size: tuple[int, int],
    strategy: str,
) -> tuple[np.ndarray | None, np.ndarray | None, dict[str, Any]]:
    if strategy == "box_only":
        return None, None, {"strategy": strategy, "positive_points": [], "negative_points": []}

    width, height = image_size
    positives = [clamp_point(point, width, height) for point in positive_points(symbol)]
    negatives: list[list[float]] = []
    if strategy == "box_pos_neg_staff":
        negatives = [clamp_point(point, width, height) for point in negative_staff_points(symbol, staves, width, height)]

    points = positives + negatives
    if not points:
        return None, None, {"strategy": strategy, "positive_points": [], "negative_points": []}
    labels = [1] * len(positives) + [0] * len(negatives)
    metadata = {
        "strategy": strategy,
        "positive_points": positives,
        "negative_points": negatives,
    }
    return np.array(points, dtype=np.float32), np.array(labels, dtype=np.int32), metadata


def load_predictor(checkpoint: Path, model_cfg: str, device: str):
    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "SAM2 is not installed. Install Meta's SAM2 package and provide a checkpoint/config before running "
            "this tool. Expected imports: sam2.build_sam.build_sam2 and sam2.sam2_image_predictor.SAM2ImagePredictor."
        ) from exc
    model = build_sam2(model_cfg, str(checkpoint), device=device)
    return SAM2ImagePredictor(model)


def refine_payload(
    predictor,
    payload: dict[str, Any],
    checkpoint: Path,
    model_cfg: str,
    prompt_strategy: str,
    out_json: Path,
    mask_dir: Path,
    limit: int | None = None,
    multimask_output: bool = False,
) -> dict[str, Any]:
    import torch

    image = Image.open(payload["input"]).convert("RGB")
    image_size = image.size
    symbols = payload["symbols"][:limit] if limit else payload["symbols"]
    staves = payload.get("staves", [])
    with torch.inference_mode():
        predictor.set_image(np.array(image))
        mask_items = []
        mask_dir.mkdir(parents=True, exist_ok=True)
        for idx, symbol in enumerate(symbols):
            box = np.array(symbol["bbox"], dtype=np.float32)
            point_coords, point_labels, prompt_metadata = build_prompt_points(symbol, staves, image_size, prompt_strategy)
            masks, scores, _ = predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=box,
                multimask_output=multimask_output,
            )
            best_idx = int(np.argmax(scores))
            mask = masks[best_idx].astype(bool)
            mask_path = mask_dir / f"{idx:06d}_{symbol['id']}_{symbol['class']}.png"
            mask_to_image(mask).save(mask_path)
            mask_items.append(
                {
                    "symbol_id": symbol["id"],
                    "class": symbol["class"],
                    "bbox": symbol["bbox"],
                    "mask_path": str(mask_path),
                    "mask_score": float(scores[best_idx]),
                    "mask_area": int(mask.sum()),
                    "source": "sam2_box_prompt",
                    "prompt_strategy": prompt_strategy,
                    "prompt": prompt_metadata,
                }
            )
    result = {
        "symbols_json": None,
        "input": payload["input"],
        "checkpoint": str(checkpoint),
        "model_cfg": model_cfg,
        "prompt_strategy": prompt_strategy,
        "masks": mask_items,
    }
    write_json(out_json, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Refine V2 symbol boxes into masks with SAM2.")
    parser.add_argument("--symbols-json", type=Path, default=Path("outputs/v2_neural_symbols/symbols_v2.json"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-cfg", required=True, help="SAM2 model config, for example sam2_hiera_l.yaml.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out-json", type=Path, default=Path("outputs/v2_sam2_masks/masks.json"))
    parser.add_argument("--mask-dir", type=Path, default=Path("outputs/v2_sam2_masks/masks"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--multimask-output", action="store_true")
    parser.add_argument(
        "--prompt-strategy",
        choices=("box_only", "box_positive", "box_pos_neg_staff"),
        default="box_only",
        help="SAM2 prompt ablation strategy. box_only matches the original behavior.",
    )
    args = parser.parse_args()

    payload: dict[str, Any] = read_json(args.symbols_json)
    device = choose_device(args.device)
    predictor = load_predictor(args.checkpoint, args.model_cfg, device)
    result = refine_payload(predictor, payload, args.checkpoint, args.model_cfg, args.prompt_strategy, args.out_json, args.mask_dir, args.limit, args.multimask_output)
    result["symbols_json"] = str(args.symbols_json)
    write_json(args.out_json, result)
    print(json.dumps({"masks": len(result["masks"]), "out_json": str(args.out_json)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
