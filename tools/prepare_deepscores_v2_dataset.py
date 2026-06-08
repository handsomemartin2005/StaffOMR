from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from omr_v2_common import (
    canonical_deepscores_class,
    class_names_for_taxonomy,
    write_json,
    zero_based_class_to_id,
)


def load_deepscores(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def select_images(
    images: list[dict[str, Any]],
    limit: int | None,
    include_image_names: list[str],
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen = set()
    by_name = {item["filename"]: item for item in images}
    for name in include_image_names:
        item = by_name.get(name)
        if item is None or item["filename"] in seen:
            continue
        selected.append(item)
        seen.add(item["filename"])
    for image in images:
        if limit is not None and len(selected) >= limit:
            break
        if image["filename"] in seen:
            continue
        selected.append(image)
        seen.add(image["filename"])
    return selected


def normalize_limit(value: int | None) -> int | None:
    if value is None or value < 0:
        return None
    return value


def xyxy_to_xywh(box: list[float], width: int, height: int) -> list[float] | None:
    x0, y0, x1, y1 = [float(v) for v in box]
    x0 = max(0.0, min(float(width), x0))
    y0 = max(0.0, min(float(height), y0))
    x1 = max(0.0, min(float(width), x1))
    y1 = max(0.0, min(float(height), y1))
    if x1 <= x0 or y1 <= y0:
        return None
    return [x0, y0, x1 - x0, y1 - y0]


def convert_split(
    data: dict[str, Any],
    images: list[dict[str, Any]],
    skip_classes: set[str],
    min_area: float,
    taxonomy: str,
    class_names: list[str],
    class_to_id: dict[str, int],
) -> tuple[dict[str, Any], dict[str, int]]:
    coco_images = []
    coco_annotations = []
    class_counts = {name: 0 for name in class_names}
    ann_out_id = 1

    for image in images:
        image_id = int(image["id"])
        width = int(image["width"])
        height = int(image["height"])
        coco_images.append(
            {
                "id": image_id,
                "file_name": image["filename"],
                "width": width,
                "height": height,
            }
        )
        for ann_id in image.get("ann_ids", []):
            ann = data["annotations"].get(str(ann_id))
            if ann is None:
                continue
            symbol_class = canonical_deepscores_class(ann, data["categories"], taxonomy=taxonomy)
            if symbol_class is None or symbol_class == "staff" or symbol_class in skip_classes:
                continue
            if symbol_class not in class_to_id:
                continue
            bbox = xyxy_to_xywh(ann["a_bbox"], width, height)
            if bbox is None:
                continue
            area = bbox[2] * bbox[3]
            if area < min_area:
                continue
            coco_annotations.append(
                {
                    "id": ann_out_id,
                    "image_id": image_id,
                    "category_id": class_to_id[symbol_class],
                    "bbox": bbox,
                    "area": area,
                    "iscrowd": 0,
                    "source_ann_id": str(ann_id),
                }
            )
            class_counts[symbol_class] += 1
            ann_out_id += 1

    coco = {
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": [{"id": idx, "name": name} for idx, name in enumerate(class_names)],
    }
    return coco, {key: value for key, value in class_counts.items() if value}


def parse_names(values: list[str] | None) -> list[str]:
    if not values:
        return []
    names = []
    for value in values:
        names.extend([item.strip() for item in value.split(",") if item.strip()])
    return names


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert DeepScores dense annotations into V2-symbol COCO for DEIM.")
    parser.add_argument("--train-json", type=Path, default=Path("ds2_dense/ds2_dense/deepscores_train.json"))
    parser.add_argument("--val-json", type=Path, default=Path("ds2_dense/ds2_dense/deepscores_train.json"))
    parser.add_argument("--image-root", type=Path, default=Path("ds2_dense/ds2_dense/images"))
    parser.add_argument("--out-root", type=Path, default=Path("outputs/v2_deim_deepscores_dataset"))
    parser.add_argument("--train-limit", type=int, default=8)
    parser.add_argument("--val-limit", type=int, default=1)
    parser.add_argument("--include-train-image", action="append", default=["lg-2267728-aug-beethoven--page-2.png"])
    parser.add_argument("--include-val-image", action="append", default=["lg-2267728-aug-beethoven--page-2.png"])
    parser.add_argument("--skip-class", action="append", default=["text_region"])
    parser.add_argument("--min-area", type=float, default=1.0)
    parser.add_argument("--taxonomy", choices=("base", "expanded"), default="base")
    args = parser.parse_args()

    class_names = class_names_for_taxonomy(args.taxonomy)
    class_to_id = zero_based_class_to_id(class_names)
    train_data = load_deepscores(args.train_json)
    val_data = load_deepscores(args.val_json)
    train_images = select_images(train_data["images"], normalize_limit(args.train_limit), parse_names(args.include_train_image))
    val_images = select_images(val_data["images"], normalize_limit(args.val_limit), parse_names(args.include_val_image))
    train_coco, train_counts = convert_split(
        train_data,
        train_images,
        set(parse_names(args.skip_class)),
        args.min_area,
        args.taxonomy,
        class_names,
        class_to_id,
    )
    val_coco, val_counts = convert_split(
        val_data,
        val_images,
        set(parse_names(args.skip_class)),
        args.min_area,
        args.taxonomy,
        class_names,
        class_to_id,
    )

    ann_dir = args.out_root / "annotations"
    write_json(ann_dir / "instances_train.json", train_coco)
    write_json(ann_dir / "instances_val.json", val_coco)
    summary = {
        "out_root": str(args.out_root),
        "image_root": str(args.image_root),
        "train_json": str(args.train_json),
        "val_json": str(args.val_json),
        "train_images": len(train_coco["images"]),
        "train_annotations": len(train_coco["annotations"]),
        "train_counts": train_counts,
        "val_images": len(val_coco["images"]),
        "val_annotations": len(val_coco["annotations"]),
        "val_counts": val_counts,
        "taxonomy": args.taxonomy,
        "class_names": class_names,
        "num_classes": len(class_names),
        "category_ids": "zero_based_for_deim",
        "train_ann_file": str(ann_dir / "instances_train.json"),
        "val_ann_file": str(ann_dir / "instances_val.json"),
    }
    write_json(args.out_root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
