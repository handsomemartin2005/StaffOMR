from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import extract_symbol_shapes_v2_1 as shape_mod
from assemble_v2_1_notes import build_note_objects
from evaluate_kern_text_proxy_metrics import chars_from_tokens, data_lines, metric_block, normalized_tokens, semantic_to_pseudo_ekern
from export_v2_1_official_piano_semantics import export_file
from export_v2_1_semantics import build_semantics
from prune_v2_1_overrecognition import PRESETS, prune_notes, prune_shapes
from run_abcd_module_ablation import Page, discover_pages, read_json, write_json


VARIANTS: dict[str, dict[str, Any]] = {
    "baseline": {},
    "note_dist_090": {"note_stem_distance_scale": 0.90},
    "note_dist_180": {"note_stem_distance_scale": 1.80},
    "note_x_070": {"note_stem_x_scale": 0.70},
    "note_x_120": {"note_stem_x_scale": 1.20},
    "max_notes_3": {"max_notes_per_stem": 3},
    "max_notes_9": {"max_notes_per_stem": 9},
    "beam_endpoint_030": {"beam_endpoint_scale": 0.30},
    "beam_endpoint_060": {"beam_endpoint_scale": 0.60},
    "ledger_dist_140": {"ledger_distance_scale": 1.40},
    "ledger_dist_230": {"ledger_distance_scale": 2.30},
    "slur_endpoint_200": {"slur_endpoint_scale": 2.0},
    "slur_endpoint_400": {"slur_endpoint_scale": 4.0},
}


def base_shapes(page: Page) -> dict[str, Any]:
    path = page.page_dir / "symbols/symbols_v2_1_shapes.json"
    payload = read_json(path)
    for symbol in payload.get("symbols", []):
        symbol.pop("relations", None)
    return payload


def process(page: Page, dataset: str, out_root: Path, variant: str, params: dict[str, Any]) -> None:
    out = out_root / dataset / variant / page.sample
    semantics_path = out / "semantics/semantics_v2_1.json"
    official_path = out / "semantics/score_official.musicxml"
    if semantics_path.exists() and (dataset == "polish" or official_path.exists()):
        return
    shapes = copy.deepcopy(base_shapes(page))
    shapes["relations"] = shape_mod.build_relations(shapes.get("symbols", []), **params)
    shapes["graph_parameter_ablation"] = {"variant": variant, "params": params}
    notes = build_note_objects({**shapes, "source_shapes_json": "graph_parameter_ablation"})
    pruned_notes, _ = prune_notes(notes, dict(PRESETS["balanced"]))
    pruned_shapes, _ = prune_shapes(shapes, pruned_notes)
    semantics = build_semantics(pruned_notes, pruned_shapes)
    write_json(out / "symbols/shapes.json", pruned_shapes)
    write_json(out / "notes/notes.json", pruned_notes)
    write_json(semantics_path, semantics)
    if dataset != "polish":
        x_tolerance, confidence = (6.0, 0.65) if dataset == "grandstaff" else (8.0, 0.69)
        export_file(
            semantics_path,
            official_path,
            include_key=False,
            include_time=False,
            emit_directions=False,
            x_tolerance=x_tolerance,
            backup_policy="fixed",
            fixed_measure_units=64,
            min_note_confidence=confidence,
            infer_same_pitch_ties=False,
        )


def aggregate_polish(pages: list[Page], root: Path) -> None:
    rows = []
    for variant, params in VARIANTS.items():
        gt_tokens: list[str] = []
        pred_tokens: list[str] = []
        gt_lines: list[str] = []
        pred_lines: list[str] = []
        for page in pages:
            gt_text = page.gt.read_text(encoding="utf-8")
            pred = semantic_to_pseudo_ekern(read_json(root / "polish" / variant / page.sample / "semantics/semantics_v2_1.json"))
            gt_tokens.extend(normalized_tokens(gt_text))
            pred_tokens.extend(normalized_tokens(pred))
            gt_lines.extend(data_lines(gt_text))
            pred_lines.extend(data_lines(pred))
        rows.append(
            {
                "variant": variant,
                "params": params,
                "SER_proxy": metric_block(gt_tokens, pred_tokens)["error_percent"],
                "CER_proxy": metric_block(chars_from_tokens(gt_tokens), chars_from_tokens(pred_tokens))["error_percent"],
                "LER_proxy": metric_block(gt_lines, pred_lines)["error_percent"],
            }
        )
    write_json(root / "polish/graph_parameter_summary.json", {"dataset": "polish", "protocol": "OFAT graph parameter ablation", "rows": rows})


def main() -> None:
    parser = argparse.ArgumentParser(description="One-factor-at-a-time ablation of V2.1 graph construction parameters.")
    parser.add_argument("--dataset", choices=("grandstaff", "olimpic", "polish"), required=True)
    parser.add_argument("--out-root", type=Path, default=Path("outputs/graph_parameter_ablation"))
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    pages = discover_pages(args.dataset)
    if args.limit is not None:
        pages = pages[: args.limit]
    for index, page in enumerate(pages, start=1):
        for variant, params in VARIANTS.items():
            process(page, args.dataset, args.out_root, variant, params)
        write_json(args.out_root / args.dataset / "progress.json", {"pages_completed": index, "pages_total": len(pages), "variants": len(VARIANTS)})
        print(json.dumps({"dataset": args.dataset, "page": page.sample, "index": index, "total": len(pages)}, ensure_ascii=False), flush=True)
    if args.dataset == "polish":
        aggregate_polish(pages, args.out_root)
    write_json(args.out_root / args.dataset / "summary.json", {"dataset": args.dataset, "pages": len(pages), "variants": VARIANTS})


if __name__ == "__main__":
    main()
