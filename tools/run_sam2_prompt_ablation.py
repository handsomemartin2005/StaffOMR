from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path

from assemble_v2_1_notes import build_note_objects
from evaluate_kern_text_proxy_metrics import chars_from_tokens, data_lines, metric_block, normalized_tokens, semantic_to_pseudo_ekern
from export_v2_1_official_piano_semantics import export_file
from export_v2_1_semantics import build_semantics
from prune_v2_1_overrecognition import PRESETS, prune_notes, prune_shapes
from run_abcd_module_ablation import Page, discover_pages, read_json, write_json
from refine_masks_sam2 import load_predictor, refine_payload


STRATEGIES = ("box_only", "box_positive", "box_pos_neg_staff")
NATIVE = {"grandstaff": "box_only", "olimpic": "box_positive", "polish": "box_only"}


def run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def shapes_for(page: Page, dataset: str, strategy: str, out: Path, predictor) -> dict:
    shapes_path = out / "symbols/shapes.json"
    if shapes_path.exists():
        return read_json(shapes_path)
    if strategy == NATIVE[dataset]:
        shapes = read_json(page.page_dir / "symbols/symbols_v2_1_shapes.json")
        write_json(shapes_path, shapes)
        return shapes
    symbols_path = page.page_dir / "symbols/symbols_v2.json"
    masks_path = out / "sam2/masks.json"
    payload = read_json(symbols_path)
    result = refine_payload(
        predictor,
        payload,
        Path("outputs/models/sam2/sam2.1_hiera_tiny.pt"),
        "configs/sam2.1/sam2.1_hiera_t.yaml",
        strategy,
        masks_path,
        out / "sam2/masks",
    )
    result["symbols_json"] = str(symbols_path)
    write_json(masks_path, result)
    extractor = Path("tools/extract_symbol_shapes_v2_1.py")
    run(
        [
            sys.executable,
            str(extractor),
            "--symbols-json",
            str(symbols_path),
            "--mask-json",
            str(masks_path),
            "--out-json",
            str(shapes_path),
        ]
    )
    return read_json(shapes_path)


def process(page: Page, dataset: str, strategy: str, root: Path, predictor) -> None:
    out = root / dataset / strategy / page.sample
    semantics_path = out / "semantics/semantics_v2_1.json"
    official_path = out / "semantics/score_official.musicxml"
    if semantics_path.exists() and (dataset == "polish" or official_path.exists()):
        return
    shapes = copy.deepcopy(shapes_for(page, dataset, strategy, out, predictor))
    shapes["sam2_prompt_ablation"] = strategy
    notes = build_note_objects({**shapes, "source_shapes_json": "sam2_prompt_ablation"})
    pruned_notes, _ = prune_notes(notes, dict(PRESETS["balanced"]))
    pruned_shapes, _ = prune_shapes(shapes, pruned_notes)
    semantics = build_semantics(pruned_notes, pruned_shapes)
    write_json(out / "symbols/shapes_pruned.json", pruned_shapes)
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
    for strategy in STRATEGIES:
        gt_tokens: list[str] = []
        pred_tokens: list[str] = []
        gt_lines: list[str] = []
        pred_lines: list[str] = []
        for page in pages:
            gt = page.gt.read_text(encoding="utf-8")
            pred = semantic_to_pseudo_ekern(read_json(root / "polish" / strategy / page.sample / "semantics/semantics_v2_1.json"))
            gt_tokens.extend(normalized_tokens(gt))
            pred_tokens.extend(normalized_tokens(pred))
            gt_lines.extend(data_lines(gt))
            pred_lines.extend(data_lines(pred))
        rows.append(
            {
                "strategy": strategy,
                "SER_proxy": metric_block(gt_tokens, pred_tokens)["error_percent"],
                "CER_proxy": metric_block(chars_from_tokens(gt_tokens), chars_from_tokens(pred_tokens))["error_percent"],
                "LER_proxy": metric_block(gt_lines, pred_lines)["error_percent"],
            }
        )
    write_json(root / "polish/summary_metrics.json", {"dataset": "polish", "rows": rows})


def main() -> None:
    parser = argparse.ArgumentParser(description="Full-dataset SAM2 prompt-strategy ablation reusing fixed DEIM predictions.")
    parser.add_argument("--dataset", choices=("grandstaff", "olimpic", "polish"), required=True)
    parser.add_argument("--out-root", type=Path, default=Path("outputs/sam2_prompt_full_ablation"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--start-index", type=int, default=1, help="1-based first dataset page to process.")
    parser.add_argument("--end-index", type=int, help="1-based inclusive last dataset page to process.")
    args = parser.parse_args()
    if args.start_index < 1:
        parser.error("--start-index must be at least 1")
    if args.end_index is not None and args.end_index < args.start_index:
        parser.error("--end-index must be greater than or equal to --start-index")
    sam2_root = str(Path(".local-tools/sam2-main").resolve())
    if sam2_root not in sys.path:
        sys.path.insert(0, sam2_root)
    predictor = load_predictor(Path("outputs/models/sam2/sam2.1_hiera_tiny.pt"), "configs/sam2.1/sam2.1_hiera_t.yaml", "cuda")
    pages = discover_pages(args.dataset)
    if args.limit is not None:
        pages = pages[: args.limit]
    all_pages = pages
    total_pages = len(all_pages)
    end_index = min(args.end_index or total_pages, total_pages)
    pages = all_pages[args.start_index - 1 : end_index]
    for index, page in enumerate(pages, start=args.start_index):
        for strategy in STRATEGIES:
            process(page, args.dataset, strategy, args.out_root, predictor)
        write_json(args.out_root / args.dataset / "progress.json", {"pages_completed": index, "pages_total": total_pages, "strategies": STRATEGIES})
        print(json.dumps({"dataset": args.dataset, "page": page.sample, "index": index, "total": total_pages}, ensure_ascii=False), flush=True)
    completed_index = end_index if pages else min(args.start_index - 1, total_pages)
    if completed_index == total_pages:
        if args.dataset == "polish":
            aggregate_polish(all_pages, args.out_root)
        write_json(args.out_root / args.dataset / "summary.json", {"dataset": args.dataset, "pages": total_pages, "strategies": STRATEGIES})


if __name__ == "__main__":
    main()
