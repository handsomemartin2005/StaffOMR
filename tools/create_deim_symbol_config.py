from __future__ import annotations

import argparse
import json
from pathlib import Path

from omr_v2_common import class_names_for_taxonomy, read_json, write_json


def posix(path: Path) -> str:
    return path.resolve().as_posix()


def class_names_from_coco(path: Path) -> list[str] | None:
    if not path.exists():
        return None
    data = read_json(path)
    categories = data.get("categories", [])
    if not categories:
        return None
    return [item["name"] for item in sorted(categories, key=lambda item: int(item["id"]))]


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a DEIM-D-FINE config for V2 symbol detection.")
    parser.add_argument("--dataset-root", type=Path, default=Path("outputs/v2_deim_deepscores_dataset"))
    parser.add_argument("--image-root", type=Path, default=Path("ds2_dense/ds2_dense/images"))
    parser.add_argument("--out-config", type=Path, default=Path(".local-tools/DEIM-main/configs/deim_dfine/deim_symbol_v2.yml"))
    parser.add_argument("--deim-output-dir", type=Path, default=Path("outputs/v2_deim_runs/v2_symbol_detector"))
    parser.add_argument("--hgnetv2-model-dir", type=Path, default=Path("outputs/models/deim/hgnetv2"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--train-batch-size", type=int, default=1)
    parser.add_argument("--val-batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--checkpoint-freq", type=int, default=1)
    parser.add_argument("--print-freq", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.0008)
    parser.add_argument("--backbone-lr", type=float, default=0.0002)
    parser.add_argument("--no-pretrained-backbone", action="store_true")
    parser.add_argument("--taxonomy", choices=("base", "expanded"), default="base")
    args = parser.parse_args()

    train_ann = args.dataset_root / "annotations" / "instances_train.json"
    val_ann = args.dataset_root / "annotations" / "instances_val.json"
    class_names = class_names_from_coco(train_ann) or class_names_for_taxonomy(args.taxonomy)
    epochs = max(1, args.epochs)
    pretrained = "False" if args.no_pretrained_backbone else "True"
    cfg = f"""__include__: [
  './deim_hgnetv2_n_coco.yml'
]

output_dir: {posix(args.deim_output_dir)}

num_classes: {len(class_names)}
remap_mscoco_category: False
sync_bn: False
use_ema: False
epoches: {epochs}
flat_epoch: {epochs}
no_aug_epoch: 0
checkpoint_freq: {max(1, args.checkpoint_freq)}
print_freq: {max(1, args.print_freq)}
warmup_iter: 1

HGNetv2:
  pretrained: {pretrained}
  local_model_dir: {posix(args.hgnetv2_model_dir)}

optimizer:
  type: AdamW
  params:
    -
      params: '^(?=.*backbone)(?!.*norm|bn).*$'
      lr: {args.backbone_lr}
    -
      params: '^(?=.*backbone)(?=.*norm|bn).*$'
      lr: {args.backbone_lr}
      weight_decay: 0.
    -
      params: '^(?=.*(?:encoder|decoder))(?=.*(?:norm|bn|bias)).*$'
      weight_decay: 0.
  lr: {args.lr}
  betas: [0.9, 0.999]
  weight_decay: 0.0001

lr_warmup_scheduler:
  type: LinearWarmup
  warmup_duration: 1

train_dataloader:
  total_batch_size: {args.train_batch_size}
  shuffle: True
  num_workers: {args.num_workers}
  drop_last: False
  dataset:
    img_folder: {posix(args.image_root)}
    ann_file: {posix(train_ann)}
    transforms:
      type: Compose
      ops:
        - {{type: Resize, size: [640, 640], }}
        - {{type: SanitizeBoundingBoxes, min_size: 1}}
        - {{type: ConvertPILImage, dtype: 'float32', scale: True}}
        - {{type: ConvertBoxes, fmt: 'cxcywh', normalize: True}}
      policy: ~
      mosaic_prob: -0.1
  collate_fn:
    type: BatchImageCollateFunction
    base_size: 640
    base_size_repeat: ~
    mixup_prob: 0.0
    mixup_epochs: [0, 0]
    stop_epoch: {epochs}

val_dataloader:
  total_batch_size: {args.val_batch_size}
  shuffle: False
  num_workers: {args.num_workers}
  drop_last: False
  dataset:
    img_folder: {posix(args.image_root)}
    ann_file: {posix(val_ann)}
    transforms:
      type: Compose
      ops:
        - {{type: Resize, size: [640, 640], }}
        - {{type: ConvertPILImage, dtype: 'float32', scale: True}}
"""
    args.out_config.parent.mkdir(parents=True, exist_ok=True)
    args.out_config.write_text(cfg, encoding="utf-8")
    summary = {
        "out_config": str(args.out_config),
        "deim_output_dir": str(args.deim_output_dir),
        "train_ann": str(train_ann),
        "val_ann": str(val_ann),
        "image_root": str(args.image_root),
        "epochs": epochs,
        "taxonomy": args.taxonomy,
        "class_names": class_names,
        "num_classes": len(class_names),
        "train_batch_size": args.train_batch_size,
        "val_batch_size": args.val_batch_size,
        "num_workers": args.num_workers,
        "checkpoint_freq": max(1, args.checkpoint_freq),
        "print_freq": max(1, args.print_freq),
        "pretrained_backbone": not args.no_pretrained_backbone,
    }
    write_json(args.out_config.with_suffix(".summary.json"), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
