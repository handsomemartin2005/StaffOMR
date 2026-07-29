from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from PIL import Image

import traditional_omr_demo as v1
from omr_v2_common import V2_ID_TO_CLASS, clamp_bbox, read_json


def safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return value.strip("._") or "item"


def resolve_image_path(file_name: str, image_root: Path | None) -> Path:
    path = Path(file_name)
    if path.exists():
        return path
    if image_root is not None:
        candidate = image_root / file_name
        if candidate.exists():
            return candidate
        candidate = image_root / path.name
        if candidate.exists():
            return candidate
    return path


def padded_bbox(
    xywh: list[float],
    image_size: tuple[int, int],
    pad_ratio: float,
    pad_px: int,
) -> list[int] | None:
    x, y, w, h = xywh
    pad = max(float(pad_px), max(w, h) * pad_ratio)
    return clamp_bbox([x - pad, y - pad, x + w + pad, y + h + pad], *image_size)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build symbol crop images from V2 COCO labels.")
    parser.add_argument("--coco", type=Path, default=Path("outputs/v2_weak_labels/weak_symbols_coco.json"))
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/v2_symbol_crops"))
    parser.add_argument("--pad-ratio", type=float, default=0.18)
    parser.add_argument("--pad-px", type=int, default=4)
    parser.add_argument("--min-size", type=int, default=4)
    parser.add_argument("--pdf-dpi", type=int, default=220)
    args = parser.parse_args()

    coco: dict[str, Any] = read_json(args.coco)
    images = {item["id"]: item for item in coco["images"]}
    categories = {item["id"]: item["name"] for item in coco.get("categories", [])}
    metadata = []
    image_cache: dict[int, Image.Image] = {}
    image_paths: dict[int, Path] = {}

    for ann in coco["annotations"]:
        image_info = images[ann["image_id"]]
        image_id = int(image_info["id"])
        if image_id not in image_cache:
            image_path = resolve_image_path(image_info["file_name"], args.image_root)
            image_paths[image_id] = image_path
            image_cache[image_id] = v1.load_input_image(image_path, dpi=args.pdf_dpi)

        image = image_cache[image_id]
        crop_box = padded_bbox(ann["bbox"], image.size, args.pad_ratio, args.pad_px)
        if crop_box is None:
            continue
        x0, y0, x1, y1 = crop_box
        if min(x1 - x0, y1 - y0) < args.min_size:
            continue

        class_name = categories.get(ann["category_id"], V2_ID_TO_CLASS.get(ann["category_id"], "unknown"))
        image_stem = safe_name(Path(image_info["file_name"]).stem)
        crop_name = f"img{image_id:04d}_ann{int(ann['id']):06d}_{image_stem}.png"
        crop_path = args.out_dir / class_name / crop_name
        crop_path.parent.mkdir(parents=True, exist_ok=True)
        image.crop((x0, y0, x1, y1)).save(crop_path)

        metadata.append(
            {
                "crop_path": str(crop_path),
                "class": class_name,
                "category_id": ann["category_id"],
                "annotation_id": ann["id"],
                "image_id": image_id,
                "image_path": str(image_paths[image_id]),
                "bbox_xywh": ann["bbox"],
                "crop_bbox_xyxy": crop_box,
                "source": ann.get("source", "unknown"),
                "score": ann.get("score"),
                "attributes": ann.get("attributes", {}),
            }
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = args.out_dir / "metadata.jsonl"
    metadata_path.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in metadata),
        encoding="utf-8",
    )
    summary_path = args.out_dir / "summary.json"
    counts: dict[str, int] = {}
    for item in metadata:
        counts[item["class"]] = counts.get(item["class"], 0) + 1
    summary_path.write_text(
        json.dumps(
            {
                "coco": str(args.coco),
                "crops": len(metadata),
                "metadata": str(metadata_path),
                "counts": dict(sorted(counts.items())),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(summary_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
