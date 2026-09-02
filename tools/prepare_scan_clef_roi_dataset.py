from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from add_clef_roi_classifier_predictions import staff_roi_bbox
from omr_v2_common import run_v1_symbol_pipeline, write_json


CATEGORIES = [
    {"id": 1, "name": "treble_clef"},
    {"id": 2, "name": "bass_clef"},
]


def label_from_staff(staff: Any) -> str | None:
    clef_type = str(getattr(staff, "clef_type", "") or "")
    if clef_type == "treble":
        return "treble_clef"
    if clef_type == "bass":
        return "bass_clef"
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare weakly labelled scan-domain clef ROI crops from staff geometry.")
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--pdf-dpi", type=int, default=220)
    parser.add_argument("--roi-mode", choices=("fixed", "component", "auto"), default="fixed")
    parser.add_argument("--pad-spaces", type=float, default=0.70)
    parser.add_argument("--fallback-width-spaces", type=float, default=5.2)
    parser.add_argument("--val-ratio", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=19)
    args = parser.parse_args()

    image_dir = args.out_dir / "images"
    overlay_dir = args.out_dir / "overlays"
    image_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)
    cat_id = {cat["name"]: int(cat["id"]) for cat in CATEGORIES}

    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []
    image_id = 1
    ann_id = 1
    for input_path in args.inputs:
        v1_result = run_v1_symbol_pipeline(input_path, pdf_dpi=args.pdf_dpi)
        page: Image.Image = v1_result["image"].convert("RGB")
        overlay = page.copy()
        draw = ImageDraw.Draw(overlay)
        for staff in v1_result["staves"]:
            label = label_from_staff(staff)
            if label is None:
                continue
            roi = staff_roi_bbox(staff, page.size, args.pad_spaces, args.fallback_width_spaces, args.roi_mode)
            if roi is None:
                continue
            x0, y0, x1, y1 = roi
            crop = page.crop((x0, y0, x1, y1)).convert("RGB")
            file_name = f"{input_path.stem}_staff{int(staff.index):03d}_{label}.png"
            crop.save(image_dir / file_name)
            width, height = crop.size
            images.append({"id": image_id, "file_name": file_name, "width": width, "height": height})
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": cat_id[label],
                    "bbox": [0, 0, width, height],
                    "area": width * height,
                    "iscrowd": 0,
                }
            )
            metadata.append(
                {
                    "file_name": file_name,
                    "source": str(input_path),
                    "staff": int(staff.index),
                    "weak_label": label,
                    "roi": roi,
                }
            )
            color = (255, 140, 0) if label == "treble_clef" else (40, 120, 255)
            draw.rectangle(roi, outline=color, width=3)
            draw.text((x0, max(0, y0 - 14)), f"{staff.index}:{label}", fill=color)
            image_id += 1
            ann_id += 1
        overlay.save(overlay_dir / f"{input_path.stem}_clef_roi_overlay.png")

    rng = random.Random(args.seed)
    order = list(range(len(images)))
    rng.shuffle(order)
    val_count = int(round(len(order) * max(0.0, min(0.9, args.val_ratio))))
    val_indices = set(order[:val_count])

    def subset_payload(indices: set[int] | None) -> dict[str, Any]:
        if indices is None:
            subset_images = images
            subset_ids = {image["id"] for image in images}
        else:
            subset_images = [image for idx, image in enumerate(images) if idx in indices]
            subset_ids = {image["id"] for image in subset_images}
        subset_annotations = [ann for ann in annotations if int(ann["image_id"]) in subset_ids]
        return {"images": subset_images, "annotations": subset_annotations, "categories": CATEGORIES}

    if val_count == 0:
        train_payload = subset_payload(None)
        val_payload = subset_payload(None)
    else:
        train_indices = set(range(len(images))) - val_indices
        train_payload = subset_payload(train_indices)
        val_payload = subset_payload(val_indices)

    ann_dir = args.out_dir / "annotations"
    write_json(ann_dir / "instances_train.json", train_payload)
    write_json(ann_dir / "instances_val.json", val_payload)
    write_json(args.out_dir / "metadata.json", metadata)
    print(
        json.dumps(
            {
                "out_dir": str(args.out_dir),
                "images": len(images),
                "annotations": len(annotations),
                "train_images": len(train_payload["images"]),
                "val_images": len(val_payload["images"]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
