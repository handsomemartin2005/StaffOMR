from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from calibrate_lmx_export_on_train import export_fix_eval
from export_v2_1_semantics import build_semantics
from prune_v2_1_overrecognition import PRESETS, prune_notes, prune_shapes, read_json, write_json


PRESET_NAMES = ("recall", "light", "balanced", "aggressive", "visual_strict")


def source_files(page_dir: Path) -> tuple[Path, Path]:
    notes = page_dir / "notes/notes_v2_1.json"
    shapes_candidates = (
        page_dir / "symbols/symbols_v2_1_shapes_crop_fused.json",
        page_dir / "symbols/symbols_v2_1_shapes.json",
    )
    shapes = next((path for path in shapes_candidates if path.exists()), shapes_candidates[-1])
    if not notes.exists() or not shapes.exists():
        raise FileNotFoundError(f"Missing pre-pruning inputs in {page_dir}: notes={notes.exists()} shapes={shapes.exists()}")
    return notes, shapes


def materialize(source_root: Path, destination_root: Path, preset: str) -> int:
    params = dict(PRESETS[preset])
    completed = 0
    for page_dir in sorted(path for path in source_root.iterdir() if path.is_dir()):
        output = destination_root / page_dir.name / "semantics/semantics_v2_1.json"
        if output.exists():
            completed += 1
            continue
        notes_path, shapes_path = source_files(page_dir)
        notes = read_json(notes_path)
        shapes = read_json(shapes_path)
        pruned_notes, _ = prune_notes(notes, params)
        pruned_shapes, _ = prune_shapes(shapes, pruned_notes)
        semantics = build_semantics(pruned_notes, pruned_shapes)
        semantics["training_domain_tuning"] = {"prune_preset": preset, "params": params}
        write_json(output, semantics)
        completed += 1
        if completed % 50 == 0:
            print(json.dumps({"preset": preset, "materialized": completed}, ensure_ascii=False), flush=True)
    return completed


def main() -> None:
    parser = argparse.ArgumentParser(description="Select pruning and official-LMX export settings only on a training domain, then evaluate a fixed choice on test.")
    parser.add_argument("--train-source", type=Path, required=True)
    parser.add_argument("--train-gt", type=Path, required=True)
    parser.add_argument("--test-source", type=Path, required=True)
    parser.add_argument("--test-gt", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--x-tolerances", type=float, nargs="+", default=[4, 6, 8])
    parser.add_argument("--confidences", type=float, nargs="+", default=[0.60, 0.65, 0.70])
    args = parser.parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)

    candidates: list[dict[str, Any]] = []
    train_counts = {}
    for preset in PRESET_NAMES:
        train_view = args.out_root / "train_views" / preset
        train_counts[preset] = materialize(args.train_source, train_view, preset)
        for xtol in args.x_tolerances:
            for confidence in args.confidences:
                tag = f"train1000_{preset}_x{xtol:g}_c{int(round(confidence * 100)):02d}"
                row = export_fix_eval(train_view, args.train_gt, tag, xtol, confidence, args.out_root / "metrics")
                row["prune_preset"] = preset
                row["pred_lmx_dir"] = str((args.out_root / "metrics" / f"{tag}_ser_pred_lmx").resolve())
                candidates.append(row)
                print(json.dumps({"candidate": tag, "SER": row["SER"], "SERnotuplets": row["SERnotuplets"]}, ensure_ascii=False), flush=True)

    best = min(candidates, key=lambda row: float(row["SER"]))
    test_view = args.out_root / "test_view" / str(best["prune_preset"])
    test_count = materialize(args.test_source, test_view, str(best["prune_preset"]))
    test_tag = f"test200_selected_{best['prune_preset']}_x{float(best['x_tolerance']):g}_c{int(round(float(best['confidence']) * 100)):02d}"
    test = export_fix_eval(test_view, args.test_gt, test_tag, float(best["x_tolerance"]), float(best["confidence"]), args.out_root / "metrics")
    test["prune_preset"] = best["prune_preset"]
    test["pred_lmx_dir"] = str((args.out_root / "metrics" / f"{test_tag}_ser_pred_lmx").resolve())
    result = {
        "protocol": "Pruning preset, x tolerance, and confidence selected exclusively on train1000; test200 is evaluated once with the fixed joint choice.",
        "train_pages": train_counts,
        "test_pages": test_count,
        "candidates": candidates,
        "best_train": best,
        "test": test,
        "train_pred_lmx_dir": best["pred_lmx_dir"],
        "test_pred_lmx_dir": test["pred_lmx_dir"],
    }
    selection = args.out_root / "selection.json"
    write_json(selection, result)
    print(json.dumps({"selection": str(selection), "best_train": best, "test": test}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
