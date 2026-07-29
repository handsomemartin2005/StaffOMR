from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parent.parent
TOOLS = ROOT / "tools"
SAM2_ROOT = ROOT / ".local-tools" / "sam2-main"
DATA = ROOT / "data" / "ijcv_samples" / "debussy-omr-transfer24"
FROZEN = ROOT / "outputs" / "ijcv_repro" / "debussy_staff_transfer24"
ABLATION = ROOT / "outputs" / "debussy_abcd_ablation"
DEFAULT_DECODER = ROOT / "outputs" / "muscima_sam2_decoder_beam_area_1ep" / "mask_decoder.pt"
DEFAULT_RELATION_DIR = ROOT / "outputs" / "muscima_beam_notehead_relation_head"
DEFAULT_CHECKPOINT = ROOT / "outputs" / "models" / "sam2" / "sam2.1_hiera_tiny.pt"
DEFAULT_OUT = ROOT / "outputs" / "sam2_adapted_beam_demo"

for path in (SAM2_ROOT, TOOLS):
    value = str(path)
    while value in sys.path:
        sys.path.remove(value)
    sys.path.insert(0, value)

import refine_masks_sam2
import run_debussy_abcd_ablation as debussy_eval
from train_muscima_beam_notehead_relation_head import pair_features


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def filtered_payload(payload: dict[str, Any], classes: set[str]) -> dict[str, Any]:
    result = copy.deepcopy(payload)
    result["symbols"] = [item for item in result.get("symbols", []) if item.get("class") in classes]
    return result


def attach_available_masks(
    payload: dict[str, Any], masks_payload: dict[str, Any], classes: set[str]
) -> tuple[dict[str, Any], dict[str, int]]:
    result = copy.deepcopy(payload)
    masks = {str(item["symbol_id"]): item for item in masks_payload.get("masks", [])}
    selected = attached = missing = 0
    for symbol in result.get("symbols", []):
        if symbol.get("class") not in classes:
            continue
        selected += 1
        item = masks.get(str(symbol["id"]))
        if item is None:
            missing += 1
            continue
        symbol["mask"] = item
        source = str(symbol.get("source") or "unknown")
        if "+sam2_adapted" not in source:
            symbol["source"] = f"{source}+sam2_adapted"
        attached += 1
    result["sam2_prompt_strategy"] = "beam_adapted_box_only"
    return result, {"selected": selected, "attached": attached, "missing": missing}


def relation_signature(relation: dict[str, Any]) -> tuple[str, str, tuple[str, ...]]:
    return (
        str(relation.get("type") or ""),
        str(relation.get("source") or ""),
        tuple(str(item) for item in relation.get("targets", [])),
    )


def relation_delta(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, int]:
    left = {relation_signature(item) for item in reference.get("relations", [])}
    right = {relation_signature(item) for item in candidate.get("relations", [])}
    beam_left = {item for item in left if item[0] == "beam_stem_group"}
    beam_right = {item for item in right if item[0] == "beam_stem_group"}
    return {
        "relations_added": len(right - left),
        "relations_removed": len(left - right),
        "beam_relations_added": len(beam_right - beam_left),
        "beam_relations_removed": len(beam_left - beam_right),
    }


def _feature_node(symbol: dict[str, Any]) -> dict[str, Any]:
    x0, y0, x1, y1 = [float(value) for value in symbol["bbox"]]
    return {
        "left": x0,
        "top": y0,
        "width": max(1.0, x1 - x0),
        "height": max(1.0, y1 - y0),
    }


def _beam_feature_node(symbol: dict[str, Any]) -> dict[str, Any]:
    result = _feature_node(symbol)
    points = ((symbol.get("mask") or {}).get("points") or [])
    if len(points) >= 3:
        left = int(math.floor(min(float(point[0]) for point in points)))
        top = int(math.floor(min(float(point[1]) for point in points)))
        right = int(math.ceil(max(float(point[0]) for point in points))) + 1
        bottom = int(math.ceil(max(float(point[1]) for point in points))) + 1
        canvas = Image.new("1", (max(1, right - left), max(1, bottom - top)), 0)
        polygon = [(float(point[0]) - left, float(point[1]) - top) for point in points]
        ImageDraw.Draw(canvas).polygon(polygon, fill=1)
        result.update(
            {
                "mask": np.asarray(canvas, dtype=bool),
                "mask_left": left,
                "mask_top": top,
            }
        )
    else:
        width = max(1, int(round(result["width"])))
        height = max(1, int(round(result["height"])))
        result["mask"] = np.ones((height, width), dtype=bool)
    return result


def apply_relation_head_filter(
    shapes: dict[str, Any], model: Any, threshold: float
) -> tuple[dict[str, Any], dict[str, int]]:
    result = copy.deepcopy(shapes)
    symbols = {str(item["id"]): item for item in result.get("symbols", [])}
    kept_relations = []
    report = {
        "beam_relations_seen": 0,
        "targets_seen": 0,
        "targets_kept": 0,
        "targets_removed": 0,
        "relations_removed": 0,
        "missing_features": 0,
    }
    for relation in result.get("relations", []):
        if relation.get("type") != "beam_stem_group":
            kept_relations.append(relation)
            continue
        report["beam_relations_seen"] += 1
        beam = symbols.get(str(relation.get("source")))
        if beam is None:
            report["missing_features"] += len(relation.get("targets", []))
            kept_relations.append(relation)
            continue
        beam_node = _beam_feature_node(beam)
        kept_targets = []
        scores: dict[str, float] = {}
        for target_id in relation.get("targets", []):
            report["targets_seen"] += 1
            stem = symbols.get(str(target_id))
            notehead_id = str(((stem or {}).get("attributes") or {}).get("notehead_id") or "")
            notehead = symbols.get(notehead_id)
            if notehead is None:
                report["missing_features"] += 1
                kept_targets.append(target_id)
                continue
            _, features = pair_features(beam_node, _feature_node(notehead))
            score = float(model.predict_proba(np.asarray([features], dtype=np.float64))[0, 1])
            scores[str(target_id)] = score
            if score >= threshold:
                kept_targets.append(target_id)
                report["targets_kept"] += 1
            else:
                report["targets_removed"] += 1
        if kept_targets:
            relation["targets"] = kept_targets
            relation.setdefault("evidence", {})["relation_head_scores"] = scores
            relation["evidence"]["relation_head_threshold"] = threshold
            kept_relations.append(relation)
        else:
            report["relations_removed"] += 1
    result["relations"] = kept_relations
    result.setdefault("ablation", {})["beam_notehead_relation_head"] = True
    result["relation_head_report"] = report
    return result, report


def build_prediction(
    module: Any,
    payload: dict[str, Any],
    symbols_path: Path,
    out: Path,
    relation_model: Any | None = None,
    relation_threshold: float = 0.61,
) -> dict[str, Any]:
    shapes = module.extract_shapes(payload, symbols_path)
    shapes.setdefault("ablation", {}).update(
        {
            "C_sam2": True,
            "D_relation_graph": True,
            "sam2_prompt_strategy": "beam_adapted_box_only",
        }
    )
    relation_report = None
    if relation_model is not None:
        shapes, relation_report = apply_relation_head_filter(shapes, relation_model, relation_threshold)
    notes = module.build_note_objects({**shapes, "source_shapes_json": "sam2_adapted_beam_demo"})
    params = dict(module.PRESETS["balanced"])
    pruned_notes, _ = module.prune_notes(notes, params)
    pruned_shapes, _ = module.prune_shapes(shapes, pruned_notes)
    semantics = module.build_semantics(pruned_notes, pruned_shapes)

    write_json(out / "symbols" / "shapes.json", pruned_shapes)
    write_json(out / "notes" / "notes.json", pruned_notes)
    write_json(out / "semantics" / "semantics_v2_1.json", semantics)
    score_path = out / "semantics" / "score.musicxml"
    score_path.parent.mkdir(parents=True, exist_ok=True)
    module.write_musicxml(score_path, module.semantic_to_musicxml(semantics))
    return {"score_path": score_path, "shapes": pruned_shapes, "relation_report": relation_report}


def aggregate(rows: list[dict[str, Any]], key: str) -> dict[str, float | int]:
    matches = sum(int(row[key]["matches"]) for row in rows)
    predicted = sum(int(row[key]["predicted"]) for row in rows)
    gold = sum(int(row[key]["gold"]) for row in rows)
    return {
        "matches": matches,
        "predicted_events": predicted,
        "gold_events": gold,
        "precision": 100.0 * matches / predicted if predicted else 0.0,
        "recall": 100.0 * matches / gold if gold else 0.0,
        "f1": 200.0 * matches / (predicted + gold) if predicted + gold else 0.0,
    }


def write_markdown(summary: dict[str, Any], path: Path) -> None:
    variants = (
        ("no_sam2", "No SAM2 / box fallback"),
        ("zero_shot_sam2", "Original zero-shot SAM2"),
        ("adapted_beam_sam2", "Area-trained beam-only SAM2"),
        ("adapted_relation_sam2", "Adapted SAM2 + learned relation filter"),
    )
    lines = [
        "# Domain-adapted SAM2 Debussy demo",
        "",
        "Frozen detector and downstream reconstruction; only the beam mask source changes.",
        "",
        "| Variant | Precision | Recall | Event F1 | Predicted events |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for key, label in variants:
        item = summary["aggregate"][key]
        lines.append(
            f"| {label} | {item['precision']:.3f} | {item['recall']:.3f} | "
            f"{item['f1']:.3f} | {item['predicted_events']} |"
        )
    lines.extend(
        [
            "",
            f"Adapted delta over no SAM2: {summary['delta_over_no_sam2']:+.3f} Event-F1 points.",
            f"Adapted delta over zero-shot SAM2: {summary['delta_over_zero_shot']:+.3f} Event-F1 points.",
            f"Adapted + relation delta over zero-shot SAM2: {summary['relation_delta_over_zero_shot']:+.3f} Event-F1 points.",
            "",
            "This is a three-page stop/go demo, not a significance result.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a beam-adapted SAM2 decoder on a small Debussy demo.")
    parser.add_argument("--pages", default="test_0019,test_0015,test_0010")
    parser.add_argument("--classes", default="beam")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--decoder-state", type=Path, default=DEFAULT_DECODER)
    parser.add_argument("--relation-dir", type=Path, default=DEFAULT_RELATION_DIR)
    parser.add_argument("--model-cfg", default="configs/sam2.1/sam2.1_hiera_t.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mask-threshold", type=float, default=0.0)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--force-masks", action="store_true")
    args = parser.parse_args()

    import joblib
    import torch

    pages = [item.strip() for item in args.pages.split(",") if item.strip()]
    classes = {item.strip() for item in args.classes.split(",") if item.strip()}
    module = debussy_eval.load_archived_ablation_module()
    predictor = refine_masks_sam2.load_predictor(args.checkpoint.resolve(), args.model_cfg, args.device)
    state = torch.load(args.decoder_state.resolve(), map_location=predictor.device, weights_only=True)
    predictor.model.sam_mask_decoder.load_state_dict(state)
    predictor.mask_threshold = float(args.mask_threshold)
    relation_model = joblib.load(args.relation_dir / "box_mask_head.joblib")
    relation_summary = read_json(args.relation_dir / "summary.json")
    relation_threshold = float(
        relation_summary["validation_safe_pruning"]["box_mask_head"]["threshold"]
    )
    out_root = args.out_root.resolve()
    rows = []

    for page_id in pages:
        started = time.perf_counter()
        symbols_path = FROZEN / page_id / "symbols" / "symbols_v2.json"
        gt = DATA / f"{page_id}.musicxml"
        base = read_json(symbols_path)
        masks_path = out_root / page_id / "sam2" / "masks.json"
        if args.force_masks or not masks_path.exists():
            masks = refine_masks_sam2.refine_payload(
                predictor,
                filtered_payload(base, classes),
                args.checkpoint,
                args.model_cfg,
                "box_only",
                masks_path,
                out_root / page_id / "sam2" / "masks",
                multimask_output=False,
            )
            masks["decoder_state"] = str(args.decoder_state.resolve())
            masks["mask_threshold"] = predictor.mask_threshold
            write_json(masks_path, masks)
        else:
            masks = read_json(masks_path)

        adapted_payload, attachment = attach_available_masks(base, masks, classes)
        adapted_symbols = out_root / page_id / "symbols" / "symbols_adapted_beam_sam2.json"
        write_json(adapted_symbols, adapted_payload)
        built = build_prediction(module, adapted_payload, adapted_symbols, out_root / "adapted" / page_id)
        relation_built = build_prediction(
            module,
            adapted_payload,
            adapted_symbols,
            out_root / "adapted_relation" / page_id,
            relation_model=relation_model,
            relation_threshold=relation_threshold,
        )
        no_sam_shapes = read_json(ABLATION / "A1B1C0D1" / page_id / "symbols" / "shapes.json")
        row = {
            "page": page_id,
            "no_sam2": debussy_eval.counts(
                gt, ABLATION / "A1B1C0D1" / page_id / "semantics" / "score.musicxml"
            ),
            "zero_shot_sam2": debussy_eval.counts(
                gt, ABLATION / "A1B1C1D1" / page_id / "semantics" / "score.musicxml"
            ),
            "adapted_beam_sam2": debussy_eval.counts(gt, built["score_path"]),
            "adapted_relation_sam2": debussy_eval.counts(gt, relation_built["score_path"]),
            "attachment": attachment,
            "relation_head_report": relation_built["relation_report"],
            "relation_delta_over_no_sam2": relation_delta(no_sam_shapes, built["shapes"]),
            "elapsed_seconds": time.perf_counter() - started,
        }
        rows.append(row)
        write_json(out_root / page_id / "comparison.json", row)
        print(row)

    aggregate_rows = {
        key: aggregate(rows, key)
        for key in ("no_sam2", "zero_shot_sam2", "adapted_beam_sam2", "adapted_relation_sam2")
    }
    adapted_f1 = float(aggregate_rows["adapted_beam_sam2"]["f1"])
    relation_f1 = float(aggregate_rows["adapted_relation_sam2"]["f1"])
    summary = {
        "protocol": "writer-disjoint area-trained beam decoder; predicted beam boxes; all non-beam classes use box fallback",
        "pages": pages,
        "classes": sorted(classes),
        "decoder_state": str(args.decoder_state.resolve()),
        "mask_threshold": predictor.mask_threshold,
        "relation_threshold": relation_threshold,
        "rows": rows,
        "aggregate": aggregate_rows,
        "delta_over_no_sam2": adapted_f1 - float(aggregate_rows["no_sam2"]["f1"]),
        "delta_over_zero_shot": adapted_f1 - float(aggregate_rows["zero_shot_sam2"]["f1"]),
        "relation_delta_over_zero_shot": relation_f1 - float(aggregate_rows["zero_shot_sam2"]["f1"]),
        "relation_delta_over_adapted": relation_f1 - adapted_f1,
        "publication_status": "three_page_demo_only_not_significance_test",
    }
    write_json(out_root / "summary.json", summary)
    write_markdown(summary, out_root / "summary.md")
    print(json.dumps(summary["aggregate"], ensure_ascii=False, indent=2))
    print({"delta_over_no_sam2": summary["delta_over_no_sam2"], "summary": str(out_root / "summary.json")})


if __name__ == "__main__":
    main()
