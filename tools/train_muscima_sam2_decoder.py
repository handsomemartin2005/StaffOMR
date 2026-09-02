from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parent.parent
TOOLS = ROOT / "tools"
for path in (TOOLS,):
    value = str(path)
    while value in sys.path:
        sys.path.remove(value)
    sys.path.insert(0, value)

from eval_muscima_sam2_masks import ANNOTATIONS, IMAGES, parse_nodes, place_local


DEFAULT_CHECKPOINT = ROOT / "outputs/models/sam2/sam2.1_hiera_tiny.pt"


def writer_id(path: Path) -> int:
    match = re.search(r"W-(\d+)", path.stem)
    if not match:
        raise ValueError(f"Cannot parse writer id: {path.name}")
    return int(match.group(1))


def parse_writer_range(value: str) -> set[int]:
    result: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            left, right = part.split("-", 1)
            result.update(range(int(left), int(right) + 1))
        else:
            result.add(int(part))
    return result


def select_documents(writers: set[int], limit: int | None) -> list[Path]:
    paths = [path for path in sorted(ANNOTATIONS.glob("*.xml")) if writer_id(path) in writers]
    return paths[:limit] if limit else paths


def select_nodes(nodes: list[dict[str, Any]], classes: set[str], max_per_class: int) -> list[dict[str, Any]]:
    counts = {class_name: 0 for class_name in classes}
    selected = []
    for node in nodes:
        class_name = str(node["class"])
        if class_name in classes and counts[class_name] < max_per_class:
            selected.append(node)
            counts[class_name] += 1
    return selected


def soft_erode(image):
    import torch.nn.functional as F

    return -F.max_pool2d(-image, kernel_size=3, stride=1, padding=1)


def soft_dilate(image):
    import torch.nn.functional as F

    return F.max_pool2d(image, kernel_size=3, stride=1, padding=1)


def soft_skeleton(image, iterations: int = 5):
    import torch

    opened = soft_dilate(soft_erode(image))
    skeleton = torch.relu(image - opened)
    current = image
    for _ in range(iterations):
        current = soft_erode(current)
        opened = soft_dilate(soft_erode(current))
        delta = torch.relu(current - opened)
        skeleton = skeleton + torch.relu(delta - skeleton * delta)
    return skeleton


def decoder_loss(logits, target, roi):
    import torch
    import torch.nn.functional as F

    probability = torch.sigmoid(logits)
    pixel_weight = roi * (1.0 + 4.0 * target)
    bce = (F.binary_cross_entropy_with_logits(logits, target, reduction="none") * pixel_weight).sum()
    bce = bce / pixel_weight.sum().clamp_min(1.0)
    pred_roi = probability * roi
    target_roi = target * roi
    intersection = (pred_roi * target_roi).sum(dim=(1, 2, 3))
    dice = 1.0 - ((2.0 * intersection + 1.0) / (
        pred_roi.sum(dim=(1, 2, 3)) + target_roi.sum(dim=(1, 2, 3)) + 1.0
    )).mean()
    pred_boundary = torch.relu(soft_dilate(probability) - soft_erode(probability)) * roi
    target_boundary = torch.relu(soft_dilate(target) - soft_erode(target)) * roi
    boundary = F.l1_loss(pred_boundary, target_boundary, reduction="sum") / roi.sum().clamp_min(1.0)
    pred_skeleton = soft_skeleton(pred_roi)
    target_skeleton = soft_skeleton(target_roi)
    tprec = (pred_skeleton * target_roi).sum() / pred_skeleton.sum().clamp_min(1e-6)
    tsens = (target_skeleton * pred_roi).sum() / target_skeleton.sum().clamp_min(1e-6)
    cldice = 1.0 - (2.0 * tprec * tsens + 1e-6) / (tprec + tsens + 1e-6)
    total = bce + dice + 0.2 * boundary + 0.2 * cldice
    return total, {
        "bce": float(bce.detach()),
        "dice": float(dice.detach()),
        "boundary": float(boundary.detach()),
        "cldice": float(cldice.detach()),
    }


def downsample_targets(target, size: tuple[int, int] = (256, 256), mode: str = "nearest"):
    import torch.nn.functional as F

    if mode not in {"nearest", "area"}:
        raise ValueError(f"Unsupported target downsampling mode: {mode}")
    return F.interpolate(target, size=size, mode=mode)


def make_targets(
    nodes: list[dict[str, Any]],
    page_shape: tuple[int, int],
    device,
    target_mode: str = "nearest",
):
    import torch

    full = np.stack([place_local(node, page_shape) for node in nodes]).astype(np.float32)
    target = torch.from_numpy(full[:, None]).to(device)
    target = downsample_targets(target, mode=target_mode)
    roi = torch.zeros_like(target)
    height, width = page_shape
    for index, node in enumerate(nodes):
        x0 = node["left"] / width * 256.0
        y0 = node["top"] / height * 256.0
        x1 = (node["left"] + node["width"]) / width * 256.0
        y1 = (node["top"] + node["height"]) / height * 256.0
        pad_x = max(2.0, x1 - x0)
        pad_y = max(2.0, y1 - y0)
        left = max(0, int(x0 - pad_x))
        top = max(0, int(y0 - pad_y))
        right = min(256, int(np.ceil(x1 + pad_x)))
        bottom = min(256, int(np.ceil(y1 + pad_y)))
        roi[index, :, top:bottom, left:right] = 1.0
    return target, roi


def forward_decoder(predictor, boxes: np.ndarray):
    import torch

    _, _, _, transformed_boxes = predictor._prep_prompts(None, None, boxes, None, True)
    box_coords = transformed_boxes.reshape(-1, 2, 2)
    box_labels = torch.tensor([[2, 3]], dtype=torch.int, device=transformed_boxes.device).repeat(
        transformed_boxes.size(0), 1
    )
    with torch.no_grad():
        sparse, dense = predictor.model.sam_prompt_encoder(
            points=(box_coords, box_labels), boxes=None, masks=None
        )
    high_res = [feature[-1].unsqueeze(0) for feature in predictor._features["high_res_feats"]]
    low_res, iou_predictions, _, _ = predictor.model.sam_mask_decoder(
        image_embeddings=predictor._features["image_embed"][-1].unsqueeze(0),
        image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse,
        dense_prompt_embeddings=dense,
        multimask_output=False,
        repeat_image=len(boxes) > 1,
        high_res_features=high_res,
    )
    return low_res, iou_predictions


def document_batches(nodes: list[dict[str, Any]], batch_size: int):
    for start in range(0, len(nodes), batch_size):
        yield nodes[start : start + batch_size]


def evaluate(
    predictor,
    paths: list[Path],
    classes: set[str],
    max_per_class: int,
    batch_size: int,
    target_mode: str = "nearest",
):
    import torch

    totals = {
        "intersection": 0,
        "union": 0,
        "pred_boundary_match": 0,
        "gold_boundary_match": 0,
        "pred_boundary": 0,
        "gold_boundary": 0,
    }
    instances = 0
    predictor.model.eval()
    with torch.inference_mode():
        for path in paths:
            image = np.array(Image.open(IMAGES / f"{path.stem}.png").convert("RGB"), copy=True)
            nodes = select_nodes(parse_nodes(path), classes, max_per_class)
            predictor.set_image(image)
            for batch in document_batches(nodes, batch_size):
                boxes = np.asarray(
                    [[n["left"], n["top"], n["left"] + n["width"], n["top"] + n["height"]] for n in batch],
                    dtype=np.float32,
                )
                logits, _ = forward_decoder(predictor, boxes)
                target, _ = make_targets(batch, image.shape[:2], logits.device, target_mode=target_mode)
                pred = torch.sigmoid(logits) > 0.5
                gold_threshold = 0.05 if target_mode == "area" else 0.5
                gold = target > gold_threshold
                totals["intersection"] += int((pred & gold).sum())
                totals["union"] += int((pred | gold).sum())
                pred_eroded = (-torch.nn.functional.max_pool2d(-pred.float(), 3, 1, 1)).bool()
                gold_eroded = (-torch.nn.functional.max_pool2d(-gold.float(), 3, 1, 1)).bool()
                pb = pred & ~pred_eroded
                gb = gold & ~gold_eroded
                dilated_gold = torch.nn.functional.max_pool2d(gb.float(), 3, 1, 1).bool()
                dilated_pred = torch.nn.functional.max_pool2d(pb.float(), 3, 1, 1).bool()
                totals["pred_boundary_match"] += int((pb & dilated_gold).sum())
                totals["gold_boundary_match"] += int((gb & dilated_pred).sum())
                totals["pred_boundary"] += int(pb.sum())
                totals["gold_boundary"] += int(gb.sum())
                instances += len(batch)
    precision = totals["pred_boundary_match"] / max(1, totals["pred_boundary"])
    recall = totals["gold_boundary_match"] / max(1, totals["gold_boundary"])
    return {
        "instances": instances,
        "iou": totals["intersection"] / max(1, totals["union"]),
        "boundary_f1": 2.0 * precision * recall / max(1e-9, precision + recall),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune the SAM2 mask decoder on MUSCIMA++ beam/stem masks.")
    parser.add_argument("--classes", default="beam,stem")
    parser.add_argument("--train-writers", default="1-30")
    parser.add_argument("--val-writers", default="31-40")
    parser.add_argument("--train-pages", type=int, default=2)
    parser.add_argument("--val-pages", type=int, default=1)
    parser.add_argument("--max-per-class", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument(
        "--target-mode",
        choices=("nearest", "area"),
        default="area",
        help="Downsampling for full-page instance masks. Area preserves thin-symbol occupancy.",
    )
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--model-cfg", default="configs/sam2.1/sam2.1_hiera_t.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/muscima_sam2_decoder_smoke")
    args = parser.parse_args()

    import torch
    import refine_masks_sam2

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    classes = {item.strip() for item in args.classes.split(",") if item.strip()}
    train_paths = select_documents(parse_writer_range(args.train_writers), args.train_pages)
    val_paths = select_documents(parse_writer_range(args.val_writers), args.val_pages)
    if {writer_id(path) for path in train_paths} & {writer_id(path) for path in val_paths}:
        raise RuntimeError("Writer leakage between train and validation splits")
    predictor = refine_masks_sam2.load_predictor(
        args.checkpoint.resolve(), args.model_cfg, refine_masks_sam2.choose_device(args.device)
    )
    for parameter in predictor.model.parameters():
        parameter.requires_grad = False
    for parameter in predictor.model.sam_mask_decoder.parameters():
        parameter.requires_grad = True
    optimizer = torch.optim.AdamW(predictor.model.sam_mask_decoder.parameters(), lr=args.lr, weight_decay=1e-4)

    before = evaluate(
        predictor,
        val_paths,
        classes,
        args.max_per_class,
        args.batch_size,
        target_mode=args.target_mode,
    )
    history = []
    predictor.model.train()
    for epoch in range(args.epochs):
        for path in train_paths:
            image = np.array(Image.open(IMAGES / f"{path.stem}.png").convert("RGB"), copy=True)
            nodes = select_nodes(parse_nodes(path), classes, args.max_per_class)
            random.shuffle(nodes)
            predictor.set_image(image)
            for batch in document_batches(nodes, args.batch_size):
                boxes = np.asarray(
                    [[n["left"], n["top"], n["left"] + n["width"], n["top"] + n["height"]] for n in batch],
                    dtype=np.float32,
                )
                optimizer.zero_grad(set_to_none=True)
                logits, _ = forward_decoder(predictor, boxes)
                target, roi = make_targets(
                    batch,
                    image.shape[:2],
                    logits.device,
                    target_mode=args.target_mode,
                )
                loss, parts = decoder_loss(logits.float(), target, roi)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(predictor.model.sam_mask_decoder.parameters(), 1.0)
                optimizer.step()
                history.append({"epoch": epoch, "page": path.stem, "loss": float(loss.detach()), **parts})
                print(history[-1])
    after = evaluate(
        predictor,
        val_paths,
        classes,
        args.max_per_class,
        args.batch_size,
        target_mode=args.target_mode,
    )

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    torch.save(predictor.model.sam_mask_decoder.state_dict(), out / "mask_decoder.pt")
    summary = {
        "protocol": "MUSCIMA++ writer-disjoint smoke; frozen image/prompt encoders; train mask decoder only",
        "classes": sorted(classes),
        "train_pages": [path.stem for path in train_paths],
        "val_pages": [path.stem for path in val_paths],
        "train_writers": sorted({writer_id(path) for path in train_paths}),
        "val_writers": sorted({writer_id(path) for path in val_paths}),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "target_mode": args.target_mode,
        "validation_gold_threshold": 0.05 if args.target_mode == "area" else 0.5,
        "validation_before": before,
        "validation_after": after,
        "validation_delta": {key: after[key] - before[key] for key in ("iou", "boundary_f1")},
        "history": history,
        "publication_status": "smoke_test_only",
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"before": before, "after": after, "delta": summary["validation_delta"]}, indent=2))


if __name__ == "__main__":
    main()
