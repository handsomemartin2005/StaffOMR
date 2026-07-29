from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torchvision import transforms as T

from omr_v2_common import clamp_bbox, count_by_class, draw_symbol_overlay, read_json, run_v1_symbol_pipeline, write_json
from train_symbol_crop_classifier import SmallCropCNN


CLEF_CLASSES = {"treble_clef", "bass_clef", "c_clef_alto", "c_clef_tenor"}


def symbol_score(symbol: dict[str, Any]) -> float:
    return float(symbol.get("confidence", symbol.get("score", 0.0)) or 0.0)


def symbol_center(symbol: dict[str, Any]) -> tuple[float, float]:
    x0, y0, x1, y1 = [float(v) for v in symbol["bbox"]]
    return (x0 + x1) * 0.5, (y0 + y1) * 0.5


def load_classifier(path: Path, device: torch.device) -> tuple[SmallCropCNN, list[str], T.Compose]:
    checkpoint = torch.load(path, map_location=device)
    class_names = [str(name) for name in checkpoint["class_names"]]
    model = SmallCropCNN(len(class_names)).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    image_size = int(checkpoint.get("image_size") or 96)
    transform = T.Compose(
        [
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    return model, class_names, transform


def staff_roi_bbox(
    staff: Any,
    image_size: tuple[int, int],
    pad_spaces: float,
    fallback_width_spaces: float,
    roi_mode: str,
) -> list[int] | None:
    width, height = image_size
    space = float(staff.space)
    use_component = roi_mode == "component" or (roi_mode == "auto" and staff.clef_bbox is not None)
    if use_component and staff.clef_bbox is not None:
        x0, y0, x1, y1 = [float(v) for v in staff.clef_bbox]
        pad_x = pad_spaces * space
        pad_y = max(pad_spaces * space, 0.45 * (y1 - y0))
        bbox = [x0 - pad_x, y0 - pad_y, x1 + pad_x, y1 + pad_y]
    else:
        bbox = [
            float(staff.x0) - 0.35 * space,
            float(staff.y0) - 3.4 * space,
            float(staff.x0) + fallback_width_spaces * space,
            float(staff.y1) + 3.4 * space,
        ]
    return clamp_bbox(bbox, width, height)


def staff_symbol_bbox(staff: Any, klass: str, image_size: tuple[int, int], mode: str) -> list[int] | None:
    if mode == "roi":
        return None
    width, height = image_size
    space = float(staff.space)
    staff_x0 = float(staff.x0)
    staff_y0 = float(staff.y0)
    staff_y1 = float(staff.y1)
    staff_mid_y = 0.5 * (staff_y0 + staff_y1)
    if klass == "bass_clef":
        bbox = [
            staff_x0 - 0.15 * space,
            staff_y0 - 0.20 * space,
            staff_x0 + 2.55 * space,
            staff_y1 + 0.75 * space,
        ]
    elif klass in {"c_clef_alto", "c_clef_tenor"}:
        bbox = [
            staff_x0 - 0.10 * space,
            staff_mid_y - 2.20 * space,
            staff_x0 + 2.25 * space,
            staff_mid_y + 2.20 * space,
        ]
    else:
        bbox = [
            staff_x0 - 0.15 * space,
            staff_mid_y - 2.85 * space,
            staff_x0 + 2.35 * space,
            staff_mid_y + 2.85 * space,
        ]
    return clamp_bbox(bbox, width, height)


def has_existing_left_clef(symbols: list[dict[str, Any]], staff: Any, min_confidence: float) -> bool:
    space = float(staff.space)
    left_limit = float(staff.x0) + 5.2 * space
    y0 = float(staff.y0) - 3.8 * space
    y1 = float(staff.y1) + 3.8 * space
    for symbol in symbols:
        if str(symbol.get("class") or "") not in CLEF_CLASSES:
            continue
        if symbol_score(symbol) < min_confidence:
            continue
        cx, cy = symbol_center(symbol)
        bx0, by0, bx1, by1 = [float(v) for v in symbol["bbox"]]
        if cx <= left_limit and y0 <= cy <= y1 and (by1 - by0) >= 1.4 * space and (bx1 - bx0) >= 0.45 * space:
            return True
    return False


@torch.no_grad()
def classify_crop(
    model: SmallCropCNN,
    class_names: list[str],
    transform: T.Compose,
    image: Image.Image,
    bbox: list[int],
    device: torch.device,
) -> tuple[str, float, float, list[float]]:
    crop = image.crop(tuple(bbox)).convert("RGB")
    tensor = transform(crop).unsqueeze(0).to(device)
    probs = torch.softmax(model(tensor), dim=1)[0].detach().cpu()
    order = torch.argsort(probs, descending=True)
    best_idx = int(order[0].item())
    second = float(probs[int(order[1].item())].item()) if len(order) > 1 else 0.0
    best = float(probs[best_idx].item())
    return class_names[best_idx], best, best - second, [float(value) for value in probs.tolist()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Add neural clef ROI classifier predictions to a detector prediction JSON.")
    parser.add_argument("--detector-json", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--classifier", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--overlay", type=Path)
    parser.add_argument("--pdf-dpi", type=int, default=220)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threshold", type=float, default=0.70)
    parser.add_argument("--min-margin", type=float, default=0.18)
    parser.add_argument("--existing-clef-min-confidence", type=float, default=0.50)
    parser.add_argument("--roi-mode", choices=("fixed", "component", "auto"), default="fixed")
    parser.add_argument("--symbol-bbox-mode", choices=("compact", "roi"), default="compact")
    parser.add_argument("--pad-spaces", type=float, default=0.70)
    parser.add_argument("--fallback-width-spaces", type=float, default=5.2)
    parser.add_argument("--replace-existing", action="store_true")
    args = parser.parse_args()

    payload = read_json(args.detector_json)
    symbols = list(payload.get("symbols", []))
    device = torch.device(args.device)
    model, class_names, transform = load_classifier(args.classifier, device)

    v1_result = run_v1_symbol_pipeline(args.input, pdf_dpi=args.pdf_dpi)
    image: Image.Image = v1_result["image"].convert("RGB")
    image_size = image.size

    supplements: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for staff in v1_result["staves"]:
        roi = staff_roi_bbox(staff, image_size, args.pad_spaces, args.fallback_width_spaces, args.roi_mode)
        if roi is None:
            continue
        if not args.replace_existing and has_existing_left_clef(symbols, staff, args.existing_clef_min_confidence):
            dropped.append({"staff": staff.index, "roi": roi, "reason": "existing_detector_clef"})
            continue
        klass, confidence, margin, probs = classify_crop(model, class_names, transform, image, roi, device)
        if klass not in CLEF_CLASSES:
            dropped.append({"staff": staff.index, "roi": roi, "reason": "non_clef_prediction", "class": klass, "confidence": confidence, "margin": margin})
            continue
        if confidence < args.threshold or margin < args.min_margin:
            dropped.append({"staff": staff.index, "roi": roi, "reason": "below_threshold", "class": klass, "confidence": confidence, "margin": margin})
            continue
        space = float(staff.space)
        symbol_bbox = staff_symbol_bbox(staff, klass, image_size, args.symbol_bbox_mode) or roi
        supplements.append(
            {
                "id": f"clef_roi_{len(supplements):04d}",
                "class": klass,
                "bbox": symbol_bbox,
                "confidence": confidence,
                "source": "neural_clef_roi_classifier",
                "attributes": {
                    "detector_class": klass,
                    "fine_class": klass,
                    "staff": int(staff.index),
                    "staff_space": space,
                    "clef_roi": True,
                    "classifier_roi_bbox": roi,
                    "symbol_bbox_mode": args.symbol_bbox_mode,
                    "classifier_margin": margin,
                    "classifier_class_names": class_names,
                    "classifier_probs": probs,
                    "rule_geometry_source": "staff_left_roi",
                    "rule_geometry_only": True,
                },
            }
        )

    fused_symbols = symbols + supplements
    out_payload = dict(payload)
    out_payload["version"] = "v2_deim_with_neural_clef_roi_classifier"
    out_payload["base_detector_json"] = str(args.detector_json)
    out_payload["clef_roi_classifier"] = {
        "classifier": str(args.classifier),
        "threshold": args.threshold,
        "min_margin": args.min_margin,
        "supplements": len(supplements),
        "dropped": dropped,
        "counts": count_by_class(supplements),
    }
    out_payload["symbols"] = fused_symbols
    out_payload["counts"] = count_by_class(fused_symbols)
    if args.overlay:
        draw_symbol_overlay(image, fused_symbols, args.overlay)
        out_payload["overlay"] = str(args.overlay)
    write_json(args.out_json, out_payload)
    print(
        json.dumps(
            {
                "out_json": str(args.out_json),
                "base_symbols": len(symbols),
                "supplements": len(supplements),
                "supplement_counts": count_by_class(supplements),
                "symbols": len(fused_symbols),
                "counts": out_payload["counts"],
                "dropped": len(dropped),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
