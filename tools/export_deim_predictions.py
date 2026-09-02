from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import torch.nn.functional as F
from PIL import Image

from omr_v2_common import (
    base_class_for_symbol_class,
    class_names_for_taxonomy,
    clamp_bbox,
    draw_symbol_overlay,
    read_json,
    write_json,
)


def load_deim_model(deim_root: Path, config: Path, checkpoint: Path, device: str) -> nn.Module:
    sys.path.insert(0, str(deim_root.resolve()))
    from engine.core import YAMLConfig  # type: ignore

    cfg = YAMLConfig(str(config.resolve()), resume=str(checkpoint.resolve()))
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    try:
        raw = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        raw = torch.load(checkpoint, map_location="cpu")
    if "ema" in raw:
        state = raw["ema"]["module"]
    elif "model" in raw:
        state = raw["model"]
    else:
        state = raw
    missing, unexpected = cfg.model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(
            json.dumps(
                {
                    "checkpoint_load": "non_strict",
                    "missing": len(missing),
                    "unexpected": len(unexpected),
                    "missing_preview": list(missing)[:20],
                    "unexpected_preview": list(unexpected)[:20],
                },
                ensure_ascii=False,
            )
        )

    class DeployModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images: torch.Tensor, orig_target_sizes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            outputs = self.model(images)
            return self.postprocessor(outputs, orig_target_sizes)

    model = DeployModel().to(device)
    model.eval()
    return model


def nms_filter(
    labels: torch.Tensor,
    boxes: torch.Tensor,
    scores: torch.Tensor,
    threshold: float,
    nms_iou: float,
    max_detections: int,
) -> list[int]:
    keep_candidates = torch.nonzero(scores >= threshold, as_tuple=False).flatten()
    if keep_candidates.numel() == 0:
        return []
    labels = labels[keep_candidates]
    boxes = boxes[keep_candidates]
    scores = scores[keep_candidates]

    kept_local: list[torch.Tensor] = []
    for label in torch.unique(labels):
        mask = labels == label
        local_indices = torch.nonzero(mask, as_tuple=False).flatten()
        local_keep = torchvision.ops.nms(boxes[local_indices], scores[local_indices], nms_iou)
        kept_local.append(local_indices[local_keep])
    if not kept_local:
        return []
    keep = torch.cat(kept_local)
    keep = keep[torch.argsort(scores[keep], descending=True)]
    keep = keep[:max_detections]
    return keep_candidates[keep].cpu().tolist()


def tile_starts(length: int, tile_size: int, overlap: float) -> list[int]:
    tile_size = max(1, min(int(tile_size), int(length)))
    if tile_size >= length:
        return [0]
    overlap = max(0.0, min(float(overlap), 0.9))
    step = max(1, int(round(tile_size * (1.0 - overlap))))
    starts = list(range(0, max(1, length - tile_size + 1), step))
    last = length - tile_size
    if not starts or starts[-1] != last:
        starts.append(last)
    return sorted(set(starts))


def load_class_names(path: Path | None, taxonomy: str) -> list[str]:
    if path is None:
        return class_names_for_taxonomy(taxonomy)
    payload = read_json(path)
    if isinstance(payload, dict) and isinstance(payload.get("class_names"), list):
        return [str(item) for item in payload["class_names"]]
    if isinstance(payload, dict) and isinstance(payload.get("categories"), list):
        return [
            str(item["name"])
            for item in sorted(payload["categories"], key=lambda item: int(item["id"]))
        ]
    raise ValueError(f"Cannot load class names from {path}")


def predict_raw_image(
    model: nn.Module,
    image: Image.Image,
    device: str,
    resize_mode: str = "stretch",
    input_size: int = 640,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    width, height = image.size
    unletterbox: tuple[float, float, float] | None = None
    if resize_mode == "stretch":
        transform = T.Compose([T.Resize((input_size, input_size)), T.ToTensor()])
        tensor = transform(image).unsqueeze(0).to(device)
        target_size = (width, height)
    elif resize_mode == "keep-aspect":
        max_side = max(width, height)
        scale = min(1.0, input_size / max(1, max_side))
        resized_w = max(1, int(round(width * scale)))
        resized_h = max(1, int(round(height * scale)))
        resized = image.resize((resized_w, resized_h), Image.Resampling.BILINEAR)
        canvas = Image.new("RGB", (input_size, input_size), "white")
        pad_x = int((input_size - resized_w) // 2)
        pad_y = int((input_size - resized_h) // 2)
        canvas.paste(resized, (pad_x, pad_y))
        tensor = T.ToTensor()(canvas).unsqueeze(0).to(device)
        target_size = (input_size, input_size)
        unletterbox = (float(scale), float(pad_x), float(pad_y))
    elif resize_mode == "original":
        tensor = T.ToTensor()(image).unsqueeze(0).to(device)
        target_size = (width, height)
    else:
        raise ValueError(f"Unsupported resize mode: {resize_mode}")
    # HGNet/DFINE backbones are safest with padded tensor sizes divisible by 32.
    pad_h = (32 - tensor.shape[-2] % 32) % 32
    pad_w = (32 - tensor.shape[-1] % 32) % 32
    if pad_h or pad_w:
        tensor = F.pad(tensor, (0, pad_w, 0, pad_h), value=1.0)
    orig_size = torch.tensor([[target_size[0], target_size[1]]], device=device)
    with torch.no_grad():
        labels, boxes, scores = model(tensor, orig_size)
    labels = labels[0].detach().cpu()
    boxes = boxes[0].detach().cpu()
    scores = scores[0].detach().cpu()
    if unletterbox is not None:
        scale, pad_x, pad_y = unletterbox
        if scale > 0:
            boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / scale
            boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / scale
    return (
        labels,
        boxes,
        scores,
    )


def resolve_resize_mode(requested: str, image_size: tuple[int, int]) -> str:
    if requested != "auto":
        return requested
    # The current DEIM config has fixed positional embeddings, so true original-size
    # inference is not valid. Keep aspect ratio and letterbox into the trained square.
    return "keep-aspect"


def symbols_from_predictions(
    labels: torch.Tensor,
    boxes: torch.Tensor,
    scores: torch.Tensor,
    keep_indices: list[int],
    image_size: tuple[int, int],
    class_names: list[str],
    id_prefix: str = "deim",
    source: str = "deim_dfine",
) -> list[dict[str, Any]]:
    width, height = image_size
    symbols: list[dict[str, Any]] = []
    for out_idx, pred_idx in enumerate(keep_indices):
        label = int(labels[pred_idx].item())
        if label < 0 or label >= len(class_names):
            continue
        bbox = clamp_bbox([float(v) for v in boxes[pred_idx].tolist()], width, height)
        if bbox is None:
            continue
        detector_class = class_names[label]
        output_class = base_class_for_symbol_class(detector_class) or detector_class
        attributes: dict[str, Any] = {
            "zero_based_label": label,
            "detector_class": detector_class,
        }
        if output_class != detector_class:
            attributes["fine_class"] = detector_class
        symbols.append(
            {
                "id": f"{id_prefix}_{out_idx:06d}",
                "class": output_class,
                "bbox": bbox,
                "confidence": float(scores[pred_idx].item()),
                "source": source,
                "attributes": attributes,
            }
        )
    return symbols


def predict_full_image(
    model: nn.Module,
    image: Image.Image,
    device: str,
    threshold: float,
    nms_iou: float,
    max_detections: int,
    class_names: list[str],
    resize_mode: str,
    input_size: int,
) -> list[dict[str, Any]]:
    labels, boxes, scores = predict_raw_image(model, image, device, resize_mode, input_size)
    keep_indices = nms_filter(labels, boxes, scores, threshold, nms_iou, max_detections)
    return symbols_from_predictions(labels, boxes, scores, keep_indices, image.size, class_names)


def predict_tiled_image(
    model: nn.Module,
    image: Image.Image,
    device: str,
    threshold: float,
    nms_iou: float,
    max_detections: int,
    class_names: list[str],
    tile_size: int,
    tile_overlap: float,
    tile_edge_margin: int,
    tile_full_height: bool,
    resize_mode: str,
    input_size: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    width, height = image.size
    tile_w = min(max(1, int(tile_size)), width)
    tile_h = height if tile_full_height else min(max(1, int(tile_size)), height)
    x_starts = tile_starts(width, tile_w, tile_overlap)
    y_starts = tile_starts(height, tile_h, tile_overlap)

    all_labels: list[int] = []
    all_boxes: list[torch.Tensor] = []
    all_scores: list[float] = []
    tile_summaries: list[dict[str, Any]] = []
    edge_margin = max(0, int(tile_edge_margin))

    for tile_idx, y0 in enumerate(y_starts):
        for x_idx, x0 in enumerate(x_starts):
            x1 = min(width, x0 + tile_w)
            y1 = min(height, y0 + tile_h)
            crop = image.crop((x0, y0, x1, y1))
            labels, boxes, scores = predict_raw_image(model, crop, device, resize_mode, input_size)
            keep_indices = nms_filter(labels, boxes, scores, threshold, nms_iou, max_detections)

            kept = 0
            dropped_edge = 0
            crop_w, crop_h = crop.size
            for pred_idx in keep_indices:
                box = boxes[pred_idx].clone()
                touches_left = x0 > 0 and float(box[0].item()) <= edge_margin
                touches_right = x1 < width and float(box[2].item()) >= crop_w - edge_margin
                touches_top = y0 > 0 and float(box[1].item()) <= edge_margin
                touches_bottom = y1 < height and float(box[3].item()) >= crop_h - edge_margin
                if touches_left or touches_right or touches_top or touches_bottom:
                    dropped_edge += 1
                    continue
                box[0] += x0
                box[2] += x0
                box[1] += y0
                box[3] += y0
                all_labels.append(int(labels[pred_idx].item()))
                all_boxes.append(box)
                all_scores.append(float(scores[pred_idx].item()))
                kept += 1

            tile_summaries.append(
                {
                    "id": f"tile_y{tile_idx:02d}_x{x_idx:02d}",
                    "bbox": [x0, y0, x1, y1],
                    "raw_keep": len(keep_indices),
                    "kept": kept,
                    "dropped_edge": dropped_edge,
                }
            )

    if not all_boxes:
        return [], tile_summaries

    merged_labels = torch.tensor(all_labels, dtype=torch.long)
    merged_boxes = torch.stack(all_boxes).to(torch.float32)
    merged_scores = torch.tensor(all_scores, dtype=torch.float32)
    keep_indices = nms_filter(merged_labels, merged_boxes, merged_scores, 0.0, nms_iou, max_detections)
    symbols = symbols_from_predictions(
        merged_labels,
        merged_boxes,
        merged_scores,
        keep_indices,
        image.size,
        class_names,
        source="deim_dfine_tiled",
    )
    return symbols, tile_summaries


def predict_with_metadata(
    model: nn.Module,
    input_path: Path,
    device: str,
    threshold: float,
    nms_iou: float,
    max_detections: int,
    class_names: list[str],
    inference_mode: str = "full",
    tile_size: int = 1280,
    tile_overlap: float = 0.25,
    tile_aspect_trigger: float = 2.2,
    tile_edge_margin: int = 24,
    resize_mode: str = "stretch",
    input_size: int = 640,
) -> tuple[Image.Image, list[dict[str, Any]], dict[str, Any]]:
    image = Image.open(input_path).convert("RGB")
    width, height = image.size
    aspect = width / max(1, height)
    effective_resize_mode = resolve_resize_mode(resize_mode, image.size)
    if inference_mode == "auto":
        use_tiles = aspect >= tile_aspect_trigger and width > min(width, tile_size)
    elif inference_mode == "tile":
        use_tiles = True
    elif inference_mode == "full":
        use_tiles = False
    else:
        raise ValueError(f"Unsupported inference mode: {inference_mode}")

    metadata: dict[str, Any] = {
        "requested_inference_mode": inference_mode,
        "image_size": [width, height],
        "tile_size": int(tile_size),
        "tile_overlap": float(tile_overlap),
        "tile_aspect_trigger": float(tile_aspect_trigger),
        "tile_edge_margin": int(tile_edge_margin),
        "requested_resize_mode": resize_mode,
        "resize_mode": effective_resize_mode,
        "input_size": int(input_size),
    }
    if use_tiles:
        symbols, tile_summaries = predict_tiled_image(
            model,
            image,
            device,
            threshold,
            nms_iou,
            max_detections,
            class_names,
            tile_size,
            tile_overlap,
            tile_edge_margin,
            tile_full_height=inference_mode == "auto" and aspect >= tile_aspect_trigger,
            resize_mode=effective_resize_mode,
            input_size=input_size,
        )
        metadata.update(
            {
                "actual_inference_mode": "tile",
                "tile_count": len(tile_summaries),
                "tiles": tile_summaries,
            }
        )
    else:
        symbols = predict_full_image(
            model,
            image,
            device,
            threshold,
            nms_iou,
            max_detections,
            class_names,
            effective_resize_mode,
            input_size,
        )
        metadata.update({"actual_inference_mode": "full", "tile_count": 0, "tiles": []})
    return image, symbols, metadata


def predict(
    model: nn.Module,
    input_path: Path,
    device: str,
    threshold: float,
    nms_iou: float,
    max_detections: int,
    class_names: list[str],
    inference_mode: str = "full",
    tile_size: int = 1280,
    tile_overlap: float = 0.25,
    tile_aspect_trigger: float = 2.2,
    tile_edge_margin: int = 24,
    resize_mode: str = "stretch",
    input_size: int = 640,
) -> tuple[Image.Image, list[dict[str, Any]]]:
    image, symbols, _ = predict_with_metadata(
        model,
        input_path,
        device,
        threshold,
        nms_iou,
        max_detections,
        class_names,
        inference_mode=inference_mode,
        tile_size=tile_size,
        tile_overlap=tile_overlap,
        tile_aspect_trigger=tile_aspect_trigger,
        tile_edge_margin=tile_edge_margin,
        resize_mode=resize_mode,
        input_size=input_size,
    )
    return image, symbols


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DEIM-D-FINE checkpoint inference and export V2 symbols.")
    parser.add_argument("--deim-root", type=Path, default=Path(".local-tools/DEIM-main"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, default=Path("ds2_dense/ds2_dense/images/lg-2267728-aug-beethoven--page-2.png"))
    parser.add_argument("--out-json", type=Path, default=Path("outputs/v2_deim_predictions/predictions.json"))
    parser.add_argument("--overlay", type=Path, default=Path("outputs/v2_deim_predictions/overlay.png"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--nms-iou", type=float, default=0.50)
    parser.add_argument("--max-detections", type=int, default=1000)
    parser.add_argument("--taxonomy", choices=("base", "expanded", "expanded_clean"), default="base")
    parser.add_argument("--inference-mode", choices=("full", "tile", "auto"), default="auto")
    parser.add_argument("--tile-size", type=int, default=1280)
    parser.add_argument("--tile-overlap", type=float, default=0.25)
    parser.add_argument("--tile-aspect-trigger", type=float, default=2.2)
    parser.add_argument("--tile-edge-margin", type=int, default=24)
    parser.add_argument(
        "--resize-mode",
        choices=("auto", "stretch", "keep-aspect", "original"),
        default="auto",
        help="DEIM preprocessing: auto chooses aspect-preserving letterbox; stretch is the legacy square resize; original is experimental and usually invalid for this config.",
    )
    parser.add_argument("--input-size", type=int, default=640, help="Square size for stretch, or long-side limit for keep-aspect.")
    parser.add_argument(
        "--class-names-json",
        type=Path,
        help="Optional COCO annotation or config summary JSON containing categories/class_names.",
    )
    args = parser.parse_args()

    class_names = load_class_names(args.class_names_json, args.taxonomy)
    model = load_deim_model(args.deim_root, args.config, args.checkpoint, args.device)
    image, symbols, inference_metadata = predict_with_metadata(
        model,
        args.input,
        args.device,
        args.threshold,
        args.nms_iou,
        args.max_detections,
        class_names,
        inference_mode=args.inference_mode,
        tile_size=args.tile_size,
        tile_overlap=args.tile_overlap,
        tile_aspect_trigger=args.tile_aspect_trigger,
        tile_edge_margin=args.tile_edge_margin,
        resize_mode=args.resize_mode,
        input_size=args.input_size,
    )
    draw_symbol_overlay(image, symbols, args.overlay)
    payload = {
        "version": "v2_deim_dfine_predictions",
        "input": str(args.input),
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "threshold": args.threshold,
        "nms_iou": args.nms_iou,
        "taxonomy": args.taxonomy,
        "inference": inference_metadata,
        "class_names": class_names,
        "symbols": symbols,
        "overlay": str(args.overlay),
    }
    write_json(args.out_json, payload)
    print(
        json.dumps(
            {
                "symbols": len(symbols),
                "out_json": str(args.out_json),
                "overlay": str(args.overlay),
                "checkpoint": str(args.checkpoint),
                "inference": {
                    "actual_inference_mode": inference_metadata["actual_inference_mode"],
                    "tile_count": inference_metadata["tile_count"],
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
