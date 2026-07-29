from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from collections import Counter
from fractions import Fraction
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
TOOLS = ROOT / "tools"
SAM2_ROOT = ROOT / ".local-tools" / "sam2-main"
DATA = ROOT / "data" / "ijcv_samples" / "debussy-omr-transfer24"
FROZEN = ROOT / "outputs" / "ijcv_repro" / "debussy_staff_transfer24"
BASELINE = ROOT / "outputs" / "debussy_abcd_ablation" / "A1B1C1D1"
DEFAULT_OUT = ROOT / "outputs" / "sam2_method_aligned_demo"
PROMPT_TEMPLATE_CLASSES = frozenset(
    {
        "filled_notehead",
        "open_notehead",
        "stem",
        "beam",
        "flat",
        "natural",
        "sharp",
        "ledger_line",
        "slur_or_tie",
    }
)
DEFAULT_MASK_SELECTION = "safe_residual"

for path in (SAM2_ROOT, TOOLS):
    value = str(path)
    while value in sys.path:
        sys.path.remove(value)
    sys.path.insert(0, value)

import refine_masks_sam2
import run_debussy_abcd_ablation as debussy_eval
from run_debussy_sam2_selective_demo import (
    build_prediction as build_shapes_prediction,
    fuse_mask_safe_residual,
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def semantics_events(path: Path) -> list[tuple[str, Fraction]]:
    """Read pitch-duration events using the semantics file's declared time scale."""

    payload = read_json(path)
    divisions = max(1, int(payload.get("divisions") or 1))
    events: list[tuple[str, Fraction]] = []
    for part in payload.get("parts", []):
        for measure in part.get("measures", []):
            for event in measure.get("events", []):
                if event.get("type") not in {"note", "rest"}:
                    continue
                pitch = event.get("pitch")
                if pitch is None:
                    pitch_token = "R"
                else:
                    pitch_token = (
                        f"{pitch.get('step', 'C')}"
                        f"{int(pitch.get('alter') or 0)}@{int(pitch.get('octave') or 4)}"
                    )
                duration = int(event.get("duration_units") or 0)
                events.append((pitch_token, Fraction(duration, divisions)))
    return events


def semantic_counts(gt: Path, semantics: Path) -> dict[str, int]:
    gold = Counter(debussy_eval.note_events(gt))
    hypothesis = Counter(semantics_events(semantics))
    return {
        "matches": sum((gold & hypothesis).values()),
        "predicted": sum(hypothesis.values()),
        "gold": sum(gold.values()),
    }


def select_prompt_masks(
    box_only_payload: dict[str, Any],
    staff_aware_payload: dict[str, Any],
    eligible_classes: set[str] | frozenset[str] | None = None,
) -> dict[str, Any]:
    """Choose the higher-confidence SAM2 candidate for each fixed detector instance."""

    box_records = {str(item["symbol_id"]): item for item in box_only_payload.get("masks", [])}
    staff_records = {str(item["symbol_id"]): item for item in staff_aware_payload.get("masks", [])}
    if box_records.keys() != staff_records.keys():
        missing_box = sorted(staff_records.keys() - box_records.keys())
        missing_staff = sorted(box_records.keys() - staff_records.keys())
        raise ValueError(
            f"prompt candidates do not cover the same symbols: missing_box={missing_box[:3]}, "
            f"missing_staff={missing_staff[:3]}"
        )

    selected_records = []
    counts = {"staff_aware": 0, "box_only": 0}
    for staff_record in staff_aware_payload.get("masks", []):
        symbol_id = str(staff_record["symbol_id"])
        box_record = box_records[symbol_id]
        box_score = float(box_record.get("mask_score") or 0.0)
        staff_score = float(staff_record.get("mask_score") or 0.0)
        class_is_eligible = eligible_classes is None or str(staff_record.get("class") or "") in eligible_classes
        selected_prompt = "staff_aware" if class_is_eligible and staff_score >= box_score else "box_only"
        chosen = copy.deepcopy(staff_record if selected_prompt == "staff_aware" else box_record)
        chosen["selected_prompt"] = selected_prompt
        chosen["alternative_mask_scores"] = {
            "box_only": box_score,
            "staff_aware": staff_score,
        }
        selected_records.append(chosen)
        counts[selected_prompt] += 1

    result = copy.deepcopy(staff_aware_payload)
    result["prompt_strategy"] = "staff_aware_score_gated"
    result["selection_rule"] = "higher_sam2_mask_score_per_symbol_no_gold"
    result["selection_counts"] = counts
    result["eligible_classes"] = sorted(eligible_classes) if eligible_classes is not None else None
    result["masks"] = selected_records
    return result


def build_safe_residual_shapes(
    box_shapes: dict[str, Any], staff_aware_shapes: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, int]]:
    """Preserve the box-only graph and add only positive staff-aware contact evidence."""

    return fuse_mask_safe_residual(box_shapes, staff_aware_shapes)


def attach_masks(payload: dict[str, Any], masks_payload: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(payload)
    masks = {str(item["symbol_id"]): item for item in masks_payload.get("masks", [])}
    missing = []
    for symbol in result.get("symbols", []):
        item = masks.get(str(symbol["id"]))
        if item is None:
            missing.append(str(symbol["id"]))
            continue
        symbol["mask"] = item
        source = str(symbol.get("source") or "unknown")
        if "+sam2" not in source:
            symbol["source"] = f"{source}+sam2"
    if missing:
        raise ValueError(f"Method-aligned masks missing for {len(missing)} symbols; first={missing[:3]}")
    result["sam2_prompt_strategy"] = refine_masks_sam2.METHOD_STRATEGY
    return result


def build_prediction(
    module: Any,
    payload: dict[str, Any],
    symbols_path: Path,
    out: Path,
    prompt_strategy: str = "method_aligned",
) -> dict[str, Any]:
    shapes = module.extract_shapes(payload, symbols_path)
    shapes.setdefault("ablation", {}).update(
        {"C_sam2": True, "D_relation_graph": True, "sam2_prompt_strategy": prompt_strategy}
    )
    notes = module.build_note_objects({**shapes, "source_shapes_json": f"{prompt_strategy}_demo"})
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
    return {
        "score_path": score_path,
        "semantics_path": out / "semantics" / "semantics_v2_1.json",
        "symbols": len(pruned_shapes.get("symbols", [])),
        "relations": len(pruned_shapes.get("relations", [])),
        "events": int((pruned_notes.get("summary") or {}).get("events") or 0),
    }


def aggregate(rows: list[dict[str, Any]], key: str) -> dict[str, float | int]:
    matches = sum(int(row[key]["matches"]) for row in rows)
    predicted = sum(int(row[key]["predicted"]) for row in rows)
    gold = sum(int(row[key]["gold"]) for row in rows)
    precision = 100.0 * matches / predicted if predicted else 0.0
    recall = 100.0 * matches / gold if gold else 0.0
    f1 = 200.0 * matches / (predicted + gold) if predicted + gold else 0.0
    return {
        "matches": matches,
        "predicted_events": predicted,
        "gold_events": gold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def write_markdown(summary: dict[str, Any], path: Path) -> None:
    baseline = summary["aggregate"]["box_only_baseline"]
    method = summary["aggregate"]["method_aligned"]
    lines = [
        "# SAM2 method-aligned Debussy demo",
        "",
        "Frozen detector symbols and balanced downstream settings; only the SAM2 prompt/image path changes.",
        "",
        "| Variant | Pages | Precision | Recall | Event F1 | Predicted events |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        f"| Existing box-only baseline | {len(summary['pages'])} | {baseline['precision']:.3f} | {baseline['recall']:.3f} | {baseline['f1']:.3f} | {baseline['predicted_events']} |",
        f"| Method-aligned dynamic crop + box + points | {len(summary['pages'])} | {method['precision']:.3f} | {method['recall']:.3f} | {method['f1']:.3f} | {method['predicted_events']} |",
        "",
        f"Event-F1 delta: {summary['f1_delta']:+.3f} points.",
        "",
        "This is a small deterministic demo, not a publication-level significance test.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare frozen box-only SAM2 against the method-aligned dynamic crop/box/point path."
    )
    parser.add_argument(
        "--pages", default="test_0001,test_0002,test_0003", help="Comma-separated Debussy page ids."
    )
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--checkpoint", type=Path, default=ROOT / "outputs" / "models" / "sam2" / "sam2.1_hiera_tiny.pt"
    )
    parser.add_argument("--model-cfg", default="configs/sam2.1/sam2.1_hiera_t.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--crop-context-scale", type=float, default=1.0)
    parser.add_argument("--force-masks", action="store_true")
    parser.add_argument(
        "--selection",
        choices=("staff_aware", "score_gated", "family_score_gated", "safe_residual"),
        default=DEFAULT_MASK_SELECTION,
        help="Use staff-aware masks directly or a per-symbol higher-SAM-score safety fallback.",
    )
    args = parser.parse_args()

    page_ids = [item.strip() for item in args.pages.split(",") if item.strip()]
    if not page_ids:
        parser.error("--pages must contain at least one page id")
    for page_id in page_ids:
        if not (DATA / f"{page_id}.musicxml").exists():
            parser.error(f"Unknown Debussy page: {page_id}")

    module = debussy_eval.load_archived_ablation_module()
    predictor = refine_masks_sam2.load_predictor(args.checkpoint, args.model_cfg, args.device)
    out_root = args.out_root.resolve()
    rows = []

    for page_id in page_ids:
        started = time.perf_counter()
        page_out = out_root / page_id
        symbols_path = FROZEN / page_id / "symbols" / "symbols_v2.json"
        base_payload = read_json(symbols_path)
        masks_path = page_out / "sam2" / "masks.json"
        if args.force_masks or not masks_path.exists():
            masks_payload = refine_masks_sam2.refine_payload(
                predictor,
                base_payload,
                args.checkpoint,
                args.model_cfg,
                refine_masks_sam2.METHOD_STRATEGY,
                masks_path,
                page_out / "sam2" / "masks",
                crop_context_scale=args.crop_context_scale,
            )
            masks_payload["symbols_json"] = str(symbols_path)
            write_json(masks_path, masks_payload)
        else:
            masks_payload = read_json(masks_path)

        selected_masks_payload = masks_payload
        prompt_strategy = refine_masks_sam2.METHOD_STRATEGY
        if args.selection in {"score_gated", "family_score_gated"}:
            box_masks_path = FROZEN / page_id / "sam2" / "masks_all.json"
            if not box_masks_path.exists():
                raise FileNotFoundError(f"Missing box-only masks for score gating: {box_masks_path}")
            eligible_classes = PROMPT_TEMPLATE_CLASSES if args.selection == "family_score_gated" else None
            selected_masks_payload = select_prompt_masks(
                read_json(box_masks_path),
                masks_payload,
                eligible_classes=eligible_classes,
            )
            write_json(page_out / "sam2" / "selected_masks.json", selected_masks_payload)
            prompt_strategy = (
                "staff_aware_family_score_gated"
                if args.selection == "family_score_gated"
                else "staff_aware_score_gated"
            )

        method_payload = attach_masks(base_payload, selected_masks_payload)
        method_symbols_path = page_out / "symbols" / "symbols_with_method_sam2.json"
        write_json(method_symbols_path, method_payload)
        residual_report = None
        if args.selection == "safe_residual":
            staff_aware_shapes = module.extract_shapes(method_payload, method_symbols_path)
            write_json(page_out / "symbols" / "staff_aware_shapes.json", staff_aware_shapes)
            box_shapes_path = BASELINE / page_id / "symbols" / "shapes.json"
            if not box_shapes_path.exists():
                raise FileNotFoundError(f"Missing frozen box-only shapes: {box_shapes_path}")
            residual_shapes, residual_report = build_safe_residual_shapes(
                read_json(box_shapes_path),
                staff_aware_shapes,
            )
            built = build_shapes_prediction(
                module,
                residual_shapes,
                page_out,
                "staff_aware_safe_residual",
            )
            built["semantics_path"] = page_out / "semantics" / "semantics_v2_1.json"
            prompt_strategy = "staff_aware_safe_residual"
        else:
            built = build_prediction(
                module,
                method_payload,
                method_symbols_path,
                page_out,
                prompt_strategy=prompt_strategy,
            )

        gt = DATA / f"{page_id}.musicxml"
        baseline_semantics = BASELINE / page_id / "semantics" / "semantics_v2_1.json"
        if not baseline_semantics.exists():
            raise FileNotFoundError(f"Missing frozen box-only baseline: {baseline_semantics}")
        mask_scores = [float(item["mask_score"]) for item in masks_payload.get("masks", [])]
        row = {
            "page": page_id,
            "symbols_input": len(base_payload.get("symbols", [])),
            "mask_score_mean": sum(mask_scores) / len(mask_scores) if mask_scores else None,
            "mask_score_below_0_5": sum(score < 0.5 for score in mask_scores),
            "box_only_baseline": semantic_counts(gt, baseline_semantics),
            "method_aligned": semantic_counts(gt, built["semantics_path"]),
            "method_symbols": built["symbols"],
            "method_relations": built["relations"],
            "method_events": built["events"],
            "selection": args.selection,
            "selection_counts": selected_masks_payload.get("selection_counts"),
            "residual_report": residual_report,
            "elapsed_seconds": time.perf_counter() - started,
        }
        rows.append(row)
        write_json(page_out / "comparison.json", row)
        print(row)

    baseline = aggregate(rows, "box_only_baseline")
    method = aggregate(rows, "method_aligned")
    summary = {
        "protocol": (
            "frozen detector; box-only baseline versus method-aligned dynamic crop+box+points; "
            "pitch-duration events read from semantics JSON with declared divisions"
        ),
        "mask_selection": args.selection,
        "pages": page_ids,
        "rows": rows,
        "aggregate": {"box_only_baseline": baseline, "method_aligned": method},
        "f1_delta": float(method["f1"]) - float(baseline["f1"]),
        "publication_status": "demo_only_not_significance_test",
    }
    write_json(out_root / "summary.json", summary)
    write_markdown(summary, out_root / "summary.md")
    print(json.dumps(summary["aggregate"], ensure_ascii=False, indent=2))
    print({"f1_delta": summary["f1_delta"], "summary": str(out_root / "summary.json")})


if __name__ == "__main__":
    main()
