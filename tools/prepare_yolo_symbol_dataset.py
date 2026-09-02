from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path


def link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def convert_split(annotation: Path, image_root: Path, out_root: Path, split: str, category_map: dict[int, int]) -> dict:
    payload = json.loads(annotation.read_text(encoding="utf-8"))
    by_image: dict[int, list[dict]] = defaultdict(list)
    for ann in payload.get("annotations", []):
        if not ann.get("iscrowd"):
            by_image[int(ann["image_id"])].append(ann)
    images_out = out_root / "images" / split
    labels_out = out_root / "labels" / split
    rows = []
    for image in payload.get("images", []):
        file_name = Path(str(image["file_name"]))
        source = image_root / file_name
        if not source.exists():
            source = image_root / file_name.name
        if not source.exists():
            raise FileNotFoundError(source)
        target = images_out / file_name.name
        link_or_copy(source, target)
        width, height = float(image["width"]), float(image["height"])
        labels = []
        for ann in by_image.get(int(image["id"]), []):
            x, y, w, h = [float(v) for v in ann["bbox"]]
            cx, cy = (x + 0.5 * w) / width, (y + 0.5 * h) / height
            labels.append(f"{category_map[int(ann['category_id'])]} {cx:.8f} {cy:.8f} {w/width:.8f} {h/height:.8f}")
        labels_out.mkdir(parents=True, exist_ok=True)
        (labels_out / f"{target.stem}.txt").write_text("\n".join(labels) + ("\n" if labels else ""), encoding="utf-8")
        rows.append({"image": str(target), "labels": len(labels)})
    return {"split": split, "images": len(rows), "annotations": sum(row["labels"] for row in rows)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert the common expanded-clean COCO symbol data to YOLO detection format.")
    parser.add_argument("--dataset-root", type=Path, default=Path("outputs/v2_deim_ds_all_expanded_clean"))
    parser.add_argument("--image-root", type=Path, default=Path("ds2_dense/ds2_dense/images"))
    parser.add_argument("--out-root", type=Path, default=Path("outputs/yolo_symbol_dataset_expanded_clean"))
    args = parser.parse_args()
    train_ann = args.dataset_root / "annotations/instances_train.json"
    val_ann = args.dataset_root / "annotations/instances_val.json"
    categories = sorted(json.loads(train_ann.read_text(encoding="utf-8"))["categories"], key=lambda item: int(item["id"]))
    category_map = {int(item["id"]): index for index, item in enumerate(categories)}
    summaries = [
        convert_split(train_ann, args.image_root, args.out_root, "train", category_map),
        convert_split(val_ann, args.image_root, args.out_root, "val", category_map),
    ]
    names = {index: str(item["name"]) for index, item in enumerate(categories)}
    yaml_text = f"path: {args.out_root.resolve().as_posix()}\ntrain: images/train\nval: images/val\nnames:\n" + "".join(f"  {index}: {name}\n" for index, name in names.items())
    yaml_path = args.out_root / "dataset.yaml"
    yaml_path.write_text(yaml_text, encoding="utf-8")
    summary = {"dataset_yaml": str(yaml_path), "classes": len(names), "names": names, "splits": summaries}
    (args.out_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"dataset_yaml": str(yaml_path), "classes": len(names), "splits": summaries}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
