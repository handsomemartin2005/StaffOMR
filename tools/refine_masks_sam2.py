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
    args = parser.parse_args()

    try:
        import torch
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "SAM2 is not installed. Install Meta's SAM2 package and provide a checkpoint/config before running "
            "this tool. Expected imports: sam2.build_sam.build_sam2 and sam2.sam2_image_predictor.SAM2ImagePredictor."
        ) from exc

    payload: dict[str, Any] = read_json(args.symbols_json)
    image = Image.open(payload["input"]).convert("RGB")
    symbols = payload["symbols"][: args.limit] if args.limit else payload["symbols"]
    device = choose_device(args.device)

    model = build_sam2(args.model_cfg, str(args.checkpoint), device=device)
    predictor = SAM2ImagePredictor(model)
    with torch.inference_mode():
        predictor.set_image(np.array(image))
        mask_items = []
        args.mask_dir.mkdir(parents=True, exist_ok=True)
        for idx, symbol in enumerate(symbols):
            box = np.array(symbol["bbox"], dtype=np.float32)
            masks, scores, _ = predictor.predict(box=box, multimask_output=args.multimask_output)
            best_idx = int(np.argmax(scores))
            mask = masks[best_idx].astype(bool)
            mask_path = args.mask_dir / f"{idx:06d}_{symbol['id']}_{symbol['class']}.png"
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
                }
            )

    result = {
        "symbols_json": str(args.symbols_json),
        "input": payload["input"],
        "checkpoint": str(args.checkpoint),
        "model_cfg": args.model_cfg,
        "masks": mask_items,
    }
    write_json(args.out_json, result)
    print(json.dumps({"masks": len(mask_items), "out_json": str(args.out_json)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
