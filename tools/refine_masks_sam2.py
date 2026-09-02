from __future__ import annotations

import argparse
import gc
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

try:
    from . import refine_masks_sam2_legacy as _LEGACY
except ImportError:
    import refine_masks_sam2_legacy as _LEGACY


ROOT = Path(__file__).resolve().parent.parent
LEGACY_STRATEGIES = ("box_only", "box_positive", "box_pos_neg_staff")
METHOD_STRATEGY = "method_aligned"
DEFAULT_PROMPT_STRATEGY = METHOD_STRATEGY
STRATEGIES = (*LEGACY_STRATEGIES, METHOD_STRATEGY)
NOTEHEAD_CLASSES = frozenset({"filled_notehead", "open_notehead"})
ACCIDENTAL_CLASSES = frozenset({"flat", "natural", "sharp"})

choose_device = _LEGACY.choose_device
load_predictor = _LEGACY.load_predictor
read_json = _LEGACY.read_json
write_json = _LEGACY.write_json
mask_to_image = _LEGACY.mask_to_image
build_prompt_points = _LEGACY.build_prompt_points


def _clamp_box(box: list[float], width: int, height: int) -> list[int]:
    x0, y0, x1, y1 = box
    left = max(0, min(width - 1, int(round(x0))))
    top = max(0, min(height - 1, int(round(y0))))
    right = max(left + 1, min(width, int(round(x1))))
    bottom = max(top + 1, min(height, int(round(y1))))
    return [left, top, right, bottom]


def _expand_box(
    box: list[int], amount_x: float, amount_y: float, width: int, height: int
) -> list[int]:
    return _clamp_box(
        [box[0] - amount_x, box[1] - amount_y, box[2] + amount_x, box[3] + amount_y],
        width,
        height,
    )


def adaptive_prompt_box(
    symbol: dict[str, Any], image_size: tuple[int, int]
) -> tuple[list[int], dict[str, Any]]:
    """Return the class/staff-scale box used as the actual SAM2 box prompt."""

    width, height = image_size
    bbox = _clamp_box([float(v) for v in symbol["bbox"]], width, height)
    box_width = max(1, bbox[2] - bbox[0])
    box_height = max(1, bbox[3] - bbox[1])
    attrs = dict(symbol.get("attributes") or {})
    staff_space = float(attrs.get("staff_space") or max(8.0, 0.5 * (box_width + box_height)))
    cls = str(symbol.get("class") or "")

    if cls in NOTEHEAD_CLASSES:
        prompt = _expand_box(bbox, 0.18 * box_width, 0.18 * box_height, width, height)
        rule = "notehead_fractional_expand"
    elif cls == "stem":
        prompt = _expand_box(bbox, 0.25 * staff_space, 0.08 * staff_space, width, height)
        rule = "stem_staff_scaled_expand"
    elif cls == "beam":
        prompt = _expand_box(bbox, 0.20 * staff_space, 0.30 * staff_space, width, height)
        rule = "beam_staff_scaled_expand"
    elif cls == "ledger_line":
        prompt = _expand_box(bbox, 0.30 * staff_space, 0.20 * staff_space, width, height)
        rule = "ledger_staff_scaled_expand"
    elif cls == "barline":
        prompt = _expand_box(bbox, 0.20 * staff_space, 0.10 * staff_space, width, height)
        rule = "barline_staff_scaled_expand"
    elif cls == "slur_or_tie":
        prompt = _expand_box(bbox, 0.35 * staff_space, 0.35 * staff_space, width, height)
        rule = "arc_staff_scaled_expand"
    elif cls in ACCIDENTAL_CLASSES or cls in {"bass_clef", "treble_clef", "rest"}:
        prompt = _expand_box(bbox, 0.12 * box_width, 0.12 * box_height, width, height)
        rule = "outline_fractional_expand"
    else:
        prompt = _expand_box(bbox, 0.08 * staff_space, 0.08 * staff_space, width, height)
        rule = "default_staff_scaled_expand"

    return prompt, {
        "detector_box": bbox,
        "adaptive_box": prompt,
        "adaptive_box_rule": rule,
        "staff_space": staff_space,
    }


def image_crop_box(
    prompt_box: list[int], image_size: tuple[int, int], staff_space: float, context_scale: float
) -> list[int]:
    width, height = image_size
    box_width = max(1, prompt_box[2] - prompt_box[0])
    box_height = max(1, prompt_box[3] - prompt_box[1])
    margin_x = context_scale * max(staff_space, 0.25 * box_width)
    margin_y = context_scale * max(staff_space, 0.25 * box_height)
    return _expand_box(prompt_box, margin_x, margin_y, width, height)


def _local_box(global_box: list[int], crop_box: list[int]) -> np.ndarray:
    return np.asarray(
        [
            global_box[0] - crop_box[0],
            global_box[1] - crop_box[1],
            global_box[2] - crop_box[0],
            global_box[3] - crop_box[1],
        ],
        dtype=np.float32,
    )


def _local_points(
    point_coords: np.ndarray | None, point_labels: np.ndarray | None, crop_box: list[int]
) -> tuple[np.ndarray | None, np.ndarray | None, list[list[float]], list[int]]:
    if point_coords is None or point_labels is None:
        return None, None, [], []
    local = np.asarray(point_coords, dtype=np.float32).copy()
    local[:, 0] -= crop_box[0]
    local[:, 1] -= crop_box[1]
    crop_width = crop_box[2] - crop_box[0]
    crop_height = crop_box[3] - crop_box[1]
    keep = (
        (local[:, 0] >= 0)
        & (local[:, 0] < crop_width)
        & (local[:, 1] >= 0)
        & (local[:, 1] < crop_height)
    )
    local = local[keep]
    labels = np.asarray(point_labels, dtype=np.int32)[keep]
    if len(local) == 0:
        return None, None, [], []
    return local, labels, local.tolist(), labels.tolist()


def filter_negative_points_outside_support(
    point_coords: np.ndarray | None,
    point_labels: np.ndarray | None,
    support_box: list[int],
) -> tuple[np.ndarray | None, np.ndarray | None, int]:
    """Drop staff-line negatives that fall inside the detector's target support."""

    if point_coords is None or point_labels is None:
        return point_coords, point_labels, 0
    coords = np.asarray(point_coords, dtype=np.float32)
    labels = np.asarray(point_labels, dtype=np.int32)
    x0, y0, x1, y1 = (float(value) for value in support_box)
    negative_inside = (
        (labels == 0)
        & (coords[:, 0] >= x0)
        & (coords[:, 0] <= x1)
        & (coords[:, 1] >= y0)
        & (coords[:, 1] <= y1)
    )
    return coords[~negative_inside], labels[~negative_inside], int(negative_inside.sum())


def _full_size_mask(mask: np.ndarray, crop_box: list[int], image_size: tuple[int, int]) -> np.ndarray:
    width, height = image_size
    crop_width = crop_box[2] - crop_box[0]
    crop_height = crop_box[3] - crop_box[1]
    local = np.asarray(mask, dtype=bool)
    if local.shape != (crop_height, crop_width):
        local = np.asarray(
            Image.fromarray(local.astype(np.uint8) * 255).resize(
                (crop_width, crop_height), Image.Resampling.NEAREST
            )
        ) > 0
    full = np.zeros((height, width), dtype=bool)
    full[crop_box[1] : crop_box[3], crop_box[0] : crop_box[2]] = local
    return full


def _refine_method_aligned(
    predictor,
    payload: dict[str, Any],
    checkpoint: Path,
    model_cfg: str,
    out_json: Path,
    mask_dir: Path,
    limit: int | None,
    multimask_output: bool,
    crop_context_scale: float,
) -> dict[str, Any]:
    import torch

    image = Image.open(payload["input"]).convert("RGB")
    image_size = image.size
    symbols = payload["symbols"][:limit] if limit else payload["symbols"]
    staves = payload.get("staves", [])
    mask_dir.mkdir(parents=True, exist_ok=True)
    mask_items: list[dict[str, Any]] = []
    oom_retries = 0

    with torch.inference_mode():
        for idx, symbol in enumerate(symbols):
            if idx and idx % 32 == 0 and torch.cuda.is_available():
                gc.collect()
                torch.cuda.empty_cache()

            prompt_box, box_metadata = adaptive_prompt_box(symbol, image_size)
            crop_box = image_crop_box(
                prompt_box,
                image_size,
                float(box_metadata["staff_space"]),
                crop_context_scale,
            )
            crop = image.crop(tuple(crop_box))
            point_coords, point_labels, _ = build_prompt_points(
                symbol, staves, image_size, "box_pos_neg_staff"
            )
            point_coords, point_labels, removed_support_negatives = filter_negative_points_outside_support(
                point_coords,
                point_labels,
                list(box_metadata["detector_box"]),
            )
            point_metadata = {
                "point_rule": "class_points_with_support_filtered_staff_negatives",
                "positive_points": (
                    point_coords[point_labels == 1].tolist()
                    if point_coords is not None and point_labels is not None
                    else []
                ),
                "negative_points": (
                    point_coords[point_labels == 0].tolist()
                    if point_coords is not None and point_labels is not None
                    else []
                ),
                "removed_support_negatives": removed_support_negatives,
            }
            local_points, local_labels, local_points_list, local_labels_list = _local_points(
                point_coords, point_labels, crop_box
            )
            local_prompt_box = _local_box(prompt_box, crop_box)

            for attempt in range(3):
                try:
                    predictor.set_image(np.array(crop, copy=True))
                    masks, scores, _ = predictor.predict(
                        point_coords=local_points,
                        point_labels=local_labels,
                        box=local_prompt_box,
                        multimask_output=multimask_output,
                    )
                    break
                except RuntimeError as exc:
                    if "out of memory" not in str(exc).lower() or attempt == 2:
                        raise
                    oom_retries += 1
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            best_idx = int(np.argmax(scores))
            full_mask = _full_size_mask(masks[best_idx], crop_box, image_size)
            mask_path = mask_dir / f"{idx:06d}_{symbol['id']}_{symbol['class']}.png"
            mask_to_image(full_mask).save(mask_path)
            prompt_metadata = {
                "strategy": METHOD_STRATEGY,
                **box_metadata,
                "image_crop": crop_box,
                "box_local_to_crop": local_prompt_box.tolist(),
                "positive_points": point_metadata.get("positive_points", []),
                "negative_points": point_metadata.get("negative_points", []),
                "point_rule": point_metadata["point_rule"],
                "removed_support_negatives": point_metadata["removed_support_negatives"],
                "points_local_to_crop": local_points_list,
                "point_labels": local_labels_list,
                "crop_context_scale": crop_context_scale,
            }
            mask_items.append(
                {
                    "symbol_id": symbol["id"],
                    "class": symbol["class"],
                    "bbox": symbol["bbox"],
                    "mask_path": str(mask_path),
                    "mask_score": float(scores[best_idx]),
                    "mask_area": int(full_mask.sum()),
                    "source": "sam2_method_aligned",
                    "prompt_strategy": METHOD_STRATEGY,
                    "prompt": prompt_metadata,
                }
            )

    result = {
        "symbols_json": None,
        "input": payload["input"],
        "checkpoint": str(checkpoint),
        "model_cfg": model_cfg,
        "prompt_strategy": METHOD_STRATEGY,
        "cuda_oom_retries": oom_retries,
        "crop_context_scale": crop_context_scale,
        "masks": mask_items,
    }
    write_json(out_json, result)
    return result


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
    crop_context_scale: float = 1.0,
) -> dict[str, Any]:
    if prompt_strategy in LEGACY_STRATEGIES:
        return _LEGACY.refine_payload(
            predictor,
            payload,
            checkpoint,
            model_cfg,
            prompt_strategy,
            out_json,
            mask_dir,
            limit,
            multimask_output,
        )
    if prompt_strategy != METHOD_STRATEGY:
        raise ValueError(f"Unknown prompt strategy: {prompt_strategy}")
    return _refine_method_aligned(
        predictor,
        payload,
        checkpoint,
        model_cfg,
        out_json,
        mask_dir,
        limit,
        multimask_output,
        crop_context_scale,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Refine V2 symbol boxes into masks with SAM2.")
    parser.add_argument("--symbols-json", type=Path, default=Path("outputs/v2/symbols/symbols_v2.json"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-cfg", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out-json", type=Path, default=Path("outputs/v2/sam2/masks_all.json"))
    parser.add_argument("--mask-dir", type=Path, default=Path("outputs/v2/sam2/masks"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--multimask-output", action="store_true")
    parser.add_argument("--prompt-strategy", choices=STRATEGIES, default=DEFAULT_PROMPT_STRATEGY)
    parser.add_argument("--crop-context-scale", type=float, default=1.0)
    args = parser.parse_args()
    if args.crop_context_scale < 0:
        parser.error("--crop-context-scale must be non-negative")

    payload = read_json(args.symbols_json)
    device = choose_device(args.device)
    predictor = load_predictor(args.checkpoint, args.model_cfg, device)
    result = refine_payload(
        predictor,
        payload,
        args.checkpoint,
        args.model_cfg,
        args.prompt_strategy,
        args.out_json,
        args.mask_dir,
        args.limit,
        args.multimask_output,
        args.crop_context_scale,
    )
    result["symbols_json"] = str(args.symbols_json)
    write_json(args.out_json, result)
    print({"out_json": str(args.out_json), "masks": len(result["masks"]), "strategy": args.prompt_strategy})


if __name__ == "__main__":
    main()
