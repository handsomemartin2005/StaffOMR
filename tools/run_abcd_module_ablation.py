from __future__ import annotations

import argparse
import copy
import json
import re
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

import extract_symbol_shapes_v2_1 as shape_mod
from assemble_v2_1_notes import build_note_objects
from evaluate_kern_text_proxy_metrics import (
    chars_from_tokens,
    data_lines,
    metric_block,
    normalized_tokens,
    semantic_to_pseudo_ekern,
    split_raw_tokens,
)
from export_v2_1_official_piano_semantics import export_file
from export_v2_1_semantics import build_semantics, semantic_to_musicxml, write_musicxml
from prune_v2_1_overrecognition import PRESETS, prune_notes, prune_shapes


@dataclass(frozen=True)
class Page:
    sample: str
    page_dir: Path
    gt: Path


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def discover_pages(dataset: str) -> list[Page]:
    if dataset == "grandstaff":
        root = Path("outputs/ijcv_repro/ours_grandstaff_lmx_test200_zero_v2_1_sam2_cropfusion")
        gt_root = Path("data/ijcv_samples/grandstaff-lmx-test200-flat")
        return [Page(p.name, p, gt_root / f"{p.name}.lmx.txt") for p in sorted(root.iterdir()) if p.is_dir() and (p / "symbols/symbols_v2.json").exists()]
    if dataset == "olimpic":
        roots = [
            (Path("outputs/public_olimpic_test50_v2_1_sam2_box_positive"), Path("data/public_olimpic/test_offset0000_count0050")),
            (Path("outputs/public_olimpic_test50b_v2_1_sam2_box_positive"), Path("data/public_olimpic/test_offset0050_count0050")),
            (Path("outputs/public_olimpic_test50c_v2_1_sam2_box_positive"), Path("data/public_olimpic/test_offset0100_count0050")),
            (Path("outputs/public_olimpic_test50d_v2_1_sam2_box_positive"), Path("data/public_olimpic/test_offset0150_count0050")),
        ]
        pages: list[Page] = []
        for root, gt_root in roots:
            pages.extend(Page(p.name, p, gt_root / f"{p.name}.lmx.txt") for p in sorted(root.iterdir()) if p.is_dir() and (p / "symbols/symbols_v2.json").exists())
        return pages
    if dataset == "polish":
        root = Path("outputs/ijcv_dwd_dataset_runs")
        gt_root = Path("data/ijcv_samples/polish-scores-test24")
        pattern = re.compile(r"^polish_test(\d{4})_v2_1_sam2_neural_clef_roi_rl_balance_test24$")
        pages = []
        for p in sorted(root.iterdir()):
            match = pattern.match(p.name)
            if match and (p / "symbols/symbols_v2.json").exists():
                sample = f"test_{match.group(1)}"
                pages.append(Page(sample, p, gt_root / f"{sample}.ekern.txt"))
        return pages
    raise ValueError(dataset)


def detector_symbol(symbol: dict[str, Any]) -> bool:
    source = str(symbol.get("source") or "").lower()
    return "deim" in source or "detector" in source or str(symbol.get("id") or "").startswith("deim_")


def filter_modules(payload: dict[str, Any], use_a: bool, use_b: bool) -> dict[str, Any]:
    result = copy.deepcopy(payload)
    symbols = []
    for symbol in result.get("symbols", []):
        is_b = detector_symbol(symbol)
        keep = (use_b and is_b) or (use_a and not is_b)
        if keep:
            symbols.append(symbol)
    result["symbols"] = symbols
    if not use_a:
        result["staves"] = []
        result["staff_space"] = None
        for symbol in result["symbols"]:
            attrs = dict(symbol.get("attributes") or {})
            for key in ("staff", "staff_space", "pitch_step", "pitch_y_error", "clef_type", "staff_lines"):
                attrs.pop(key, None)
            symbol["attributes"] = attrs
    result.setdefault("ablation", {}).update({"A_rule_staff_prior": use_a, "B_deim_detector": use_b})
    return result


def extract_shapes(payload: dict[str, Any], symbols_path: Path) -> dict[str, Any]:
    if not payload.get("symbols"):
        return {
            **payload,
            "version": "v2.1_symbol_shapes_ablation",
            "symbols": [],
            "dropped_symbols": [],
            "relations": [],
            "counts": {},
            "v2_1_summary": {"symbols": 0, "dropped_symbols": 0, "synthetic_symbols": 0, "relations": 0, "geometry_counts": {}, "postprocess": {}},
        }
    input_path = Path(payload["input"])
    image = Image.open(input_path).convert("RGB")
    image_size = image.size
    gray = np.array(image.convert("L"))
    staves = payload.get("staves", [])
    anchors = [Path.cwd(), symbols_path.parent]
    symbols = [shape_mod.enrich_symbol_shape(symbol, image_size, staves, {}, anchors, 96, 96) for symbol in payload.get("symbols", [])]
    symbols, dropped, synthetic, postprocess = shape_mod.clean_and_augment_symbols(symbols, staves, image_size, gray)
    relations = shape_mod.build_relations(symbols)
    geometry_counts: dict[str, int] = {}
    for symbol in symbols:
        status = str(symbol.get("geometry_check", "unknown"))
        geometry_counts[status] = geometry_counts.get(status, 0) + 1
    return {
        **payload,
        "version": "v2.1_symbol_shapes_ablation",
        "symbols": symbols,
        "dropped_symbols": dropped,
        "relations": relations,
        "counts": shape_mod.count_by_class(symbols),
        "v2_1_summary": {
            "symbols": len(symbols),
            "dropped_symbols": len(dropped),
            "synthetic_symbols": len(synthetic),
            "relations": len(relations),
            "geometry_counts": dict(sorted(geometry_counts.items())),
            "postprocess": dict(sorted(postprocess.items())),
        },
    }


def build_variant(
    page: Page,
    out_dir: Path,
    use_a: bool,
    use_b: bool,
    use_c: bool,
    use_d: bool,
    dataset: str,
    shape_cache: dict[tuple[bool, bool, bool], dict[str, Any]],
) -> dict[str, Any]:
    variant = f"A{int(use_a)}B{int(use_b)}C{int(use_c)}D{int(use_d)}"
    page_out = out_dir / variant / page.sample
    semantics_path = page_out / "semantics/semantics_v2_1.json"
    official_path = page_out / "semantics/score_official.musicxml"
    if semantics_path.exists() and (dataset == "polish" or official_path.exists()):
        return {"variant": variant, "sample": page.sample, "status": "skipped_existing"}

    cache_key = (use_a, use_b, use_c)
    if cache_key not in shape_cache:
        symbols_path = page.page_dir / "symbols" / ("symbols_v2_with_sam2.json" if use_c else "symbols_v2.json")
        payload = filter_modules(read_json(symbols_path), use_a, use_b)
        shape_cache[cache_key] = extract_shapes(payload, symbols_path)
    shapes = copy.deepcopy(shape_cache[cache_key])
    shapes.setdefault("ablation", {}).update({"C_sam2": use_c, "D_relation_graph": use_d})
    if not use_d:
        shapes["relations"] = []
        shapes.setdefault("v2_1_summary", {})["relations"] = 0

    notes = build_note_objects({**shapes, "source_shapes_json": "in_memory_ablation"})
    params = dict(PRESETS["balanced"])
    pruned_notes, _ = prune_notes(notes, params)
    pruned_shapes, _ = prune_shapes(shapes, pruned_notes)
    semantics = build_semantics(pruned_notes, pruned_shapes)
    write_json(page_out / "symbols/shapes.json", pruned_shapes)
    write_json(page_out / "notes/notes.json", pruned_notes)
    write_json(semantics_path, semantics)
    write_musicxml(page_out / "semantics/score.musicxml", semantic_to_musicxml(semantics))
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
    return {
        "variant": variant,
        "sample": page.sample,
        "status": "completed",
        "symbols": len(pruned_shapes.get("symbols", [])),
        "relations": len(pruned_shapes.get("relations", [])),
        "events": int((pruned_notes.get("summary") or {}).get("events") or 0),
    }


def polish_aggregate(pages: list[Page], out_dir: Path) -> dict[str, Any]:
    summaries = []
    for flags in product((False, True), repeat=4):
        variant = "".join(f"{name}{int(value)}" for name, value in zip("ABCD", flags))
        gt_norm: list[str] = []
        pred_norm: list[str] = []
        gt_raw: list[str] = []
        pred_raw: list[str] = []
        gt_lines: list[str] = []
        pred_lines: list[str] = []
        for page in pages:
            gt_text = page.gt.read_text(encoding="utf-8")
            semantics = read_json(out_dir / variant / page.sample / "semantics/semantics_v2_1.json")
            pred_text = semantic_to_pseudo_ekern(semantics)
            gt_norm.extend(normalized_tokens(gt_text))
            pred_norm.extend(normalized_tokens(pred_text))
            gt_raw.extend(split_raw_tokens(gt_text))
            pred_raw.extend(split_raw_tokens(pred_text))
            gt_lines.extend(data_lines(gt_text))
            pred_lines.extend(data_lines(pred_text))
        ser = metric_block(gt_norm, pred_norm)["error_percent"]
        cer = metric_block(chars_from_tokens(gt_norm), chars_from_tokens(pred_norm))["error_percent"]
        raw_ler = metric_block(gt_lines, pred_lines)["error_percent"]
        summaries.append({"variant": variant, "samples": len(pages), "SER_proxy": ser, "CER_proxy": cer, "LER_proxy": raw_ler, "gt_tokens": len(gt_norm), "pred_tokens": len(pred_norm)})
    result = {"dataset": "polish", "protocol": "concatenated eKern proxy", "variants": summaries}
    write_json(out_dir / "polish_abcd_summary.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full 2^4 ablation over A=rule/staff prior, B=DEIM, C=SAM2, D=relation graph.")
    parser.add_argument("--dataset", choices=("grandstaff", "olimpic", "polish"), required=True)
    parser.add_argument("--out-root", type=Path, default=Path("outputs/abcd_ablation"))
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    pages = discover_pages(args.dataset)
    if args.limit is not None:
        pages = pages[: args.limit]
    out_dir = args.out_root / args.dataset
    rows = []
    for index, page in enumerate(pages, start=1):
        shape_cache: dict[tuple[bool, bool, bool], dict[str, Any]] = {}
        for flags in product((False, True), repeat=4):
            rows.append(build_variant(page, out_dir, *flags, dataset=args.dataset, shape_cache=shape_cache))
        write_json(out_dir / "progress.json", {"dataset": args.dataset, "pages_completed": index, "pages_total": len(pages), "rows": rows})
        print(json.dumps({"dataset": args.dataset, "page": page.sample, "index": index, "total": len(pages)}, ensure_ascii=False), flush=True)
    summary: dict[str, Any] = {"dataset": args.dataset, "pages": len(pages), "variants": 16, "rows": len(rows)}
    if args.dataset == "polish":
        summary["metrics"] = polish_aggregate(pages, out_dir)
    write_json(out_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
