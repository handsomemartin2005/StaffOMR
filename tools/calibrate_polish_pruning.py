from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from assemble_v2_1_notes import build_note_objects
from evaluate_v2_1_symbol_classifier_proxy import evaluate_shapes
from evaluate_kern_text_proxy_metrics import normalized_tokens, semantic_to_pseudo_ekern
from export_v2_1_semantics import build_semantics
from prune_v2_1_overrecognition import PRESETS
from prune_v2_1_overrecognition import prune_notes, prune_shapes
from sweep_v2_1_symbol_classifier_fusion import apply_family_fusion, classify_symbols, load_classifier, read_json
from tune_v2_1_recall_balance import aggregate


def evaluate_root(source_root: Path, gt_root: Path, classifier_bundle, params_by_name: dict[str, dict], device: torch.device) -> dict[str, dict]:
    model, class_names, transform, checkpoint = classifier_bundle
    per_variant: dict[str, list[dict]] = {name: [] for name in params_by_name}
    for page_dir in sorted(path for path in source_root.iterdir() if path.is_dir()):
        shapes_path = page_dir / "symbols/symbols_v2_1_shapes.json"
        gt_path = gt_root / f"{page_dir.name}.ekern.txt"
        if not shapes_path.exists() or not gt_path.exists():
            continue
        shapes = read_json(shapes_path)
        predictions = classify_symbols(shapes, model, class_names, transform, checkpoint, device, 128)
        fused, _ = apply_family_fusion(shapes, predictions, conditional_threshold=0.55, family_mass_threshold=0.50, enabled_families={"notehead", "accidental"})
        for name, params in params_by_name.items():
            per_variant[name].append(evaluate_shapes(fused, gt_path, params))
        print(json.dumps({"page": page_dir.name, "variants": len(params_by_name)}), flush=True)
    return {name: {"metrics": aggregate(rows), "samples": len(rows)} for name, rows in per_variant.items()}


def materialize_rows(
    source_root: Path,
    gt_root: Path,
    classifier_bundle,
    params: dict,
    device: torch.device,
    include_targets: bool,
) -> list[dict]:
    model, class_names, transform, checkpoint = classifier_bundle
    rows = []
    for page_dir in sorted(path for path in source_root.iterdir() if path.is_dir()):
        shapes_path = page_dir / "symbols/symbols_v2_1_shapes.json"
        if not shapes_path.exists():
            continue
        shapes = read_json(shapes_path)
        predictions = classify_symbols(shapes, model, class_names, transform, checkpoint, device, 128)
        fused, _ = apply_family_fusion(shapes, predictions, conditional_threshold=0.55, family_mass_threshold=0.50, enabled_families={"notehead", "accidental"})
        notes = build_note_objects(fused)
        pruned_notes, _ = prune_notes(notes, params)
        pruned_shapes, _ = prune_shapes(fused, pruned_notes)
        semantics = build_semantics(pruned_notes, pruned_shapes)
        src = " ".join(normalized_tokens(semantic_to_pseudo_ekern(semantics)))
        row = {"sample": page_dir.name, "src": src, "tgt": src, "src_tokens": len(src.split())}
        if include_targets:
            gt_path = gt_root / f"{page_dir.name}.ekern.txt"
            if not gt_path.exists():
                continue
            tgt = " ".join(normalized_tokens(gt_path.read_text(encoding="utf-8")))
            row.update({"tgt": tgt, "tgt_tokens": len(tgt.split()), "gt": str(gt_path)})
        rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Select pruning only on Polish train83 and evaluate the fixed choice on test24.")
    parser.add_argument("--train-source", type=Path, required=True)
    parser.add_argument("--train-gt", type=Path, required=True)
    parser.add_argument("--test-source", type=Path, required=True)
    parser.add_argument("--test-gt", type=Path, required=True)
    parser.add_argument("--classifier", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    bundle = load_classifier(args.classifier, device)
    params_by_name = {name: dict(params) for name, params in PRESETS.items() if name != "none"}
    train = evaluate_root(args.train_source, args.train_gt, bundle, params_by_name, device)
    best_name = min(train, key=lambda name: float(train[name]["metrics"]["ser_proxy_percent"]))
    test = evaluate_root(args.test_source, args.test_gt, bundle, {best_name: params_by_name[best_name]}, device)
    train_pairs_path = args.out_json.with_name(args.out_json.stem + "_train_pairs.jsonl")
    test_inputs_path = args.out_json.with_name(args.out_json.stem + "_test_inputs.jsonl")
    train_rows = materialize_rows(args.train_source, args.train_gt, bundle, params_by_name[best_name], device, include_targets=True)
    test_rows = materialize_rows(args.test_source, args.test_gt, bundle, params_by_name[best_name], device, include_targets=False)
    write_jsonl(train_pairs_path, train_rows)
    write_jsonl(test_inputs_path, test_rows)
    result = {
        "protocol": "Pruning preset selected only on Polish train83; test24 is evaluated once with the fixed preset.",
        "crop_fusion": {"conditional_threshold": 0.55, "family_mass_threshold": 0.50, "families": ["notehead", "accidental"]},
        "train_candidates": train,
        "selected": {"name": best_name, "params": params_by_name[best_name]},
        "test": test[best_name],
        "materialized": {
            "train_pairs": str(train_pairs_path),
            "train_rows": len(train_rows),
            "test_inputs": str(test_inputs_path),
            "test_rows": len(test_rows),
            "test_ground_truth_read_for_inputs": False,
        },
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"selected": result["selected"], "test": result["test"]["metrics"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
