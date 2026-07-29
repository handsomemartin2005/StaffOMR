from __future__ import annotations

import argparse
import json
from pathlib import Path


def posix(path: Path) -> str:
    return path.resolve().as_posix()


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a fair RT-DETRv2 symbol-detector config on the same COCO data as DEIM.")
    parser.add_argument("--dataset-root", type=Path, default=Path("outputs/v2_deim_ds_all_expanded_clean"))
    parser.add_argument("--image-root", type=Path, default=Path("ds2_dense/ds2_dense/images"))
    parser.add_argument("--out-config", type=Path, default=Path(".local-tools/DEIM-main/configs/deim_rtdetrv2/rtdetrv2_symbol_expanded_clean.yml"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/rtdetr_runs/rtdetrv2_symbol_expanded_clean"))
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--train-batch-size", type=int, default=1)
    parser.add_argument("--val-batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()
    train_ann = args.dataset_root / "annotations/instances_train.json"
    val_ann = args.dataset_root / "annotations/instances_val.json"
    categories = json.loads(train_ann.read_text(encoding="utf-8"))["categories"]
    num_classes = len(categories)
    epochs = max(1, args.epochs)
    config = f"""__include__: [
  './rtdetrv2_r18vd_120e_coco.yml'
]

output_dir: {posix(args.output_dir)}
num_classes: {num_classes}
remap_mscoco_category: False
sync_bn: False
use_ema: False
epoches: {epochs}
checkpoint_freq: 10
print_freq: 50

PResNet:
  depth: 18
  freeze_at: -1
  freeze_norm: False
  pretrained: True

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
        - {{type: Resize, size: [640, 640]}}
        - {{type: SanitizeBoundingBoxes, min_size: 1}}
        - {{type: ConvertPILImage, dtype: 'float32', scale: True}}
        - {{type: ConvertBoxes, fmt: 'cxcywh', normalize: True}}
      policy: ~
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
        - {{type: Resize, size: [640, 640]}}
        - {{type: ConvertPILImage, dtype: 'float32', scale: True}}
        - {{type: ConvertBoxes, fmt: 'cxcywh', normalize: True}}
"""
    args.out_config.parent.mkdir(parents=True, exist_ok=True)
    args.out_config.write_text(config, encoding="utf-8")
    summary = {
        "architecture": "RT-DETRv2-R18",
        "dataset_root": str(args.dataset_root),
        "image_root": str(args.image_root),
        "train_ann": str(train_ann),
        "val_ann": str(val_ann),
        "num_classes": num_classes,
        "epochs": epochs,
        "out_config": str(args.out_config),
        "output_dir": str(args.output_dir),
        "fairness": "Same expanded-clean COCO annotations, 640x640 input, batch size and 400-epoch budget as the DEIM detector ablation target.",
    }
    args.out_config.with_suffix(".summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
