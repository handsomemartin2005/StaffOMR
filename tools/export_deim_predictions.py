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


def predict(
    model: nn.Module,
    input_path: Path,
    device: str,
    threshold: float,
    nms_iou: float,
    max_detections: int,
    class_names: list[str],
) -> tuple[Image.Image, list[dict[str, Any]]]:
    image = Image.open(input_path).convert("RGB")
    width, height = image.size
    transform = T.Compose([T.Resize((640, 640)), T.ToTensor()])
    tensor = transform(image).unsqueeze(0).to(device)
    orig_size = torch.tensor([[width, height]], device=device)
    with torch.no_grad():
        labels, boxes, scores = model(tensor, orig_size)
    labels = labels[0].detach().cpu()
    boxes = boxes[0].detach().cpu()
    scores = scores[0].detach().cpu()
    keep_indices = nms_filter(labels, boxes, scores, threshold, nms_iou, max_detections)

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
                "id": f"deim_{out_idx:06d}",
                "class": output_class,
                "bbox": bbox,
                "confidence": float(scores[pred_idx].item()),
                "source": "deim_dfine",
                "attributes": attributes,
            }
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
    parser.add_argument("--taxonomy", choices=("base", "expanded"), default="base")
    parser.add_argument(
        "--class-names-json",
        type=Path,
        help="Optional COCO annotation or config summary JSON containing categories/class_names.",
    )
    args = parser.parse_args()

    class_names = load_class_names(args.class_names_json, args.taxonomy)
    model = load_deim_model(args.deim_root, args.config, args.checkpoint, args.device)
    image, symbols = predict(
        model,
        args.input,
        args.device,
        args.threshold,
        args.nms_iou,
        args.max_detections,
        class_names,
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
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
