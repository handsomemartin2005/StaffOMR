from __future__ import annotations

import argparse
import copy
import json
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

from eval_muscima_sam2_masks import IMAGES
from train_muscima_beam_notehead_relation_head import (
    ANNOTATIONS,
    build_dataset,
    edge_metrics,
    parse_nodes,
    parse_writer_range,
    writer_id,
)


DEFAULT_CHECKPOINT = ROOT / "outputs/models/sam2/sam2.1_hiera_tiny.pt"
DEFAULT_DECODER = ROOT / "outputs/muscima_sam2_decoder_beam_area_1ep/mask_decoder.pt"
DEFAULT_RELATION_DIR = ROOT / "outputs/muscima_beam_notehead_relation_head"
DEFAULT_OUT = ROOT / "outputs/muscima_sam2_relation_gate_5p"


def crop_mask(mask: np.ndarray) -> dict[str, Any]:
    mask = np.asarray(mask, dtype=bool)
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return {"mask": np.zeros((1, 1), dtype=bool), "mask_left": 0, "mask_top": 0}
    left = int(xs.min())
    right = int(xs.max()) + 1
    top = int(ys.min())
    bottom = int(ys.max()) + 1
    return {"mask": mask[top:bottom, left:right].copy(), "mask_left": left, "mask_top": top}


def predict_beam_masks(predictor, paths: list[Path]) -> dict[tuple[str, str], dict[str, Any]]:
    import torch

    result: dict[tuple[str, str], dict[str, Any]] = {}
    with torch.inference_mode():
        for path in paths:
            image = np.array(Image.open(IMAGES / f"{path.stem}.png").convert("RGB"), copy=True)
            beams = [node for node in parse_nodes(path) if node["class"] == "beam"]
            predictor.set_image(image)
            for beam in beams:
                box = np.asarray(
                    [
                        beam["left"],
                        beam["top"],
                        beam["left"] + beam["width"],
                        beam["top"] + beam["height"],
                    ],
                    dtype=np.float32,
                )
                masks, _, _ = predictor.predict(box=box, multimask_output=False)
                result[(path.stem, str(beam["id"]))] = crop_mask(masks[0])
            print({"page": path.stem, "beams": len(beams)})
    return result


def main() -> None:
    import joblib
    import torch
    import refine_masks_sam2

    parser = argparse.ArgumentParser(description="Evaluate relation features from predicted SAM2 beam masks.")
    parser.add_argument("--writers", default="41-50")
    parser.add_argument("--pages", type=int, default=5)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--decoder-state", type=Path, default=DEFAULT_DECODER)
    parser.add_argument("--relation-dir", type=Path, default=DEFAULT_RELATION_DIR)
    parser.add_argument("--model-cfg", default="configs/sam2.1/sam2.1_hiera_t.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mask-threshold", type=float, default=0.0)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    writers = parse_writer_range(args.writers)
    paths = [path for path in sorted(ANNOTATIONS.glob("*.xml")) if writer_id(path) in writers][: args.pages]
    if len(paths) != args.pages:
        raise RuntimeError(f"Requested {args.pages} pages, found {len(paths)}")
    relation_summary = json.loads((args.relation_dir / "summary.json").read_text(encoding="utf-8"))
    box_model = joblib.load(args.relation_dir / "box_only_head.joblib")
    mask_model = joblib.load(args.relation_dir / "box_mask_head.joblib")
    threshold_box = float(relation_summary["validation"]["box_only_head"]["threshold"])
    threshold_mask = float(relation_summary["validation"]["box_mask_head"]["threshold"])
    threshold_safe = float(
        relation_summary["validation_safe_pruning"]["box_mask_head"]["threshold"]
    )

    predictor = refine_masks_sam2.load_predictor(args.checkpoint.resolve(), args.model_cfg, args.device)
    predictor.mask_threshold = float(args.mask_threshold)
    base_decoder = copy.deepcopy(predictor.model.sam_mask_decoder.state_dict())
    zero_overrides = predict_beam_masks(predictor, paths)
    adapted_state = torch.load(args.decoder_state.resolve(), map_location=predictor.device, weights_only=True)
    predictor.model.sam_mask_decoder.load_state_dict(adapted_state)
    adapted_overrides = predict_beam_masks(predictor, paths)
    predictor.model.sam_mask_decoder.load_state_dict(base_decoder)

    gt_dataset = build_dataset(paths)
    zero_dataset = build_dataset(paths, zero_overrides)
    adapted_dataset = build_dataset(paths, adapted_overrides)
    if not np.array_equal(gt_dataset.labels, zero_dataset.labels) or not np.array_equal(
        gt_dataset.labels, adapted_dataset.labels
    ):
        raise RuntimeError("Candidate ordering changed while replacing masks")
    results = {
        "box_only_head": edge_metrics(
            gt_dataset.labels,
            box_model.predict_proba(gt_dataset.box_features)[:, 1],
            threshold_box,
            gt_dataset.gold_edges,
        ),
        "gt_mask_head": edge_metrics(
            gt_dataset.labels,
            mask_model.predict_proba(gt_dataset.mask_features)[:, 1],
            threshold_mask,
            gt_dataset.gold_edges,
        ),
        "zero_shot_mask_head": edge_metrics(
            zero_dataset.labels,
            mask_model.predict_proba(zero_dataset.mask_features)[:, 1],
            threshold_mask,
            gt_dataset.gold_edges,
        ),
        "adapted_mask_head": edge_metrics(
            adapted_dataset.labels,
            mask_model.predict_proba(adapted_dataset.mask_features)[:, 1],
            threshold_mask,
            gt_dataset.gold_edges,
        ),
        "adapted_safe_pruning_head": edge_metrics(
            adapted_dataset.labels,
            mask_model.predict_proba(adapted_dataset.mask_features)[:, 1],
            threshold_safe,
            gt_dataset.gold_edges,
        ),
    }
    summary = {
        "protocol": "frozen relation heads and validation thresholds; held-out writers; GT beam boxes; predicted mask feature replacement",
        "pages": [path.stem for path in paths],
        "writers": sorted({writer_id(path) for path in paths}),
        "candidates": int(len(gt_dataset.labels)),
        "gold_edges": int(gt_dataset.gold_edges),
        "missed_gold_edges": int(gt_dataset.missed_gold_edges),
        "mask_threshold": predictor.mask_threshold,
        "relation_thresholds": {"box": threshold_box, "mask": threshold_mask, "safe_pruning": threshold_safe},
        "results": results,
        "deltas": {
            "gt_mask_over_box": results["gt_mask_head"]["edge_f1"] - results["box_only_head"]["edge_f1"],
            "zero_shot_mask_over_box": results["zero_shot_mask_head"]["edge_f1"] - results["box_only_head"]["edge_f1"],
            "adapted_mask_over_box": results["adapted_mask_head"]["edge_f1"] - results["box_only_head"]["edge_f1"],
            "adapted_over_zero_shot": results["adapted_mask_head"]["edge_f1"] - results["zero_shot_mask_head"]["edge_f1"],
        },
        "publication_status": "small_source_domain_predicted_mask_relation_gate",
    }
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"results": results, "deltas": summary["deltas"]}, indent=2))


if __name__ == "__main__":
    main()
