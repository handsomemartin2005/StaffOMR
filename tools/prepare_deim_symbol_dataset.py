from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from omr_v2_common import read_json, write_json


def remap_category_ids(coco: dict[str, Any], zero_based: bool) -> dict[str, Any]:
    if not zero_based:
        return coco
    sorted_categories = sorted(coco["categories"], key=lambda item: item["id"])
    id_map = {category["id"]: idx for idx, category in enumerate(sorted_categories)}
    return {
        "images": coco["images"],
        "annotations": [
            {
                **ann,
                "category_id": id_map[ann["category_id"]],
            }
            for ann in coco["annotations"]
        ],
        "categories": [
            {
                **category,
                "id": id_map[category["id"]],
            }
            for category in sorted_categories
        ],
        "category_id_map": id_map,
    }


def rewrite_split(coco: dict[str, Any], out_root: Path, split: str, image_root: Path | None) -> dict[str, Any]:
    image_dir = out_root / "images" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    rewritten = {
        "images": [],
        "annotations": coco["annotations"],
        "categories": coco["categories"],
    }
    for image in coco["images"]:
        src = Path(image["file_name"])
        if not src.exists() and image_root is not None:
            candidate = image_root / image["file_name"]
            if candidate.exists():
                src = candidate
            else:
                candidate = image_root / Path(image["file_name"]).name
                if candidate.exists():
                    src = candidate
        if not src.exists():
            raise FileNotFoundError(src)
        dst = image_dir / src.name
        if not dst.exists() or dst.stat().st_size != src.stat().st_size:
            shutil.copy2(src, dst)
        item = dict(image)
        item["file_name"] = src.name
        rewritten["images"].append(item)
    return rewritten


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare V2 symbol COCO labels for DEIM-D-FINE training.")
    parser.add_argument("--coco", type=Path, default=Path("outputs/v2_weak_labels/weak_symbols_coco.json"))
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--out-root", type=Path, default=Path("outputs/v2_deim_dataset"))
    parser.add_argument("--same-val", action="store_true", help="Use the same images/annotations for val smoke runs.")
    parser.add_argument("--one-based-category-ids", action="store_true", help="Keep original category IDs instead of remapping to 0..N-1.")
    args = parser.parse_args()

    original_coco = read_json(args.coco)
    coco = remap_category_ids(original_coco, zero_based=not args.one_based_category_ids)
    train = rewrite_split(coco, args.out_root, "train", args.image_root)
    val = rewrite_split(coco, args.out_root, "val", args.image_root) if args.same_val else train
    ann_dir = args.out_root / "annotations"
    write_json(ann_dir / "instances_train.json", train)
    write_json(ann_dir / "instances_val.json", val)
    summary = {
        "out_root": str(args.out_root),
        "train_images": len(train["images"]),
        "train_annotations": len(train["annotations"]),
        "val_images": len(val["images"]),
        "val_annotations": len(val["annotations"]),
        "num_classes": len(coco["categories"]),
        "category_id_map": coco.get("category_id_map"),
        "train_img_folder": str(args.out_root / "images" / "train"),
        "train_ann_file": str(ann_dir / "instances_train.json"),
        "val_img_folder": str(args.out_root / "images" / "val"),
        "val_ann_file": str(ann_dir / "instances_val.json"),
    }
    write_json(args.out_root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
