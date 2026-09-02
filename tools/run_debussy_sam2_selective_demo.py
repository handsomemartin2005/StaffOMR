from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
TOOLS = ROOT / "tools"
DATA = ROOT / "data" / "ijcv_samples" / "debussy-omr-transfer24"
FROZEN = ROOT / "outputs" / "ijcv_repro" / "debussy_staff_transfer24"
ABLATION = ROOT / "outputs" / "debussy_abcd_ablation"
DEFAULT_OUT = ROOT / "outputs" / "sam2_selective_demo"

# Keep the recovered source directory first when the runner is invoked directly.
for path in (TOOLS,):
    value = str(path)
    while value in sys.path:
        sys.path.remove(value)
    sys.path.insert(0, value)

import run_debussy_abcd_ablation as debussy_eval


QUALITY_RULES: dict[str, tuple[float, float, float]] = {
    "beam": (0.70, 0.20, 1.60),
    "stem": (0.60, 0.10, 1.80),
    "ledger_line": (0.65, 0.10, 1.60),
    "slur_or_tie": (0.72, 0.10, 1.20),
    "barline": (0.65, 0.10, 1.60),
    "filled_notehead": (0.75, 0.50, 1.50),
    "open_notehead": (0.75, 0.50, 1.50),
}
NOTEHEAD_CLASSES = frozenset({"filled_notehead", "open_notehead"})


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def symbol_space(symbol: dict[str, Any]) -> float:
    attrs = symbol.get("attributes") or {}
    bbox = symbol.get("bbox") or [0, 0, 8, 8]
    return float(attrs.get("staff_space") or max(8.0, 0.5 * ((bbox[2] - bbox[0]) + (bbox[3] - bbox[1]))))


def symbol_center(symbol: dict[str, Any]) -> tuple[float, float]:
    attrs = symbol.get("attributes") or {}
    center = attrs.get("center")
    if center:
        return float(center[0]), float(center[1])
    x0, y0, x1, y1 = symbol["bbox"]
    return 0.5 * (x0 + x1), 0.5 * (y0 + y1)


def dense_notehead_ids(symbols: list[dict[str, Any]]) -> set[str]:
    noteheads = [item for item in symbols if item.get("class") in NOTEHEAD_CLASSES]
    dense: set[str] = set()
    for index, left in enumerate(noteheads):
        lx, ly = symbol_center(left)
        lspace = symbol_space(left)
        lstaff = (left.get("attributes") or {}).get("staff")
        for right in noteheads[index + 1 :]:
            rstaff = (right.get("attributes") or {}).get("staff")
            if lstaff is not None and rstaff is not None and lstaff != rstaff:
                continue
            rx, ry = symbol_center(right)
            threshold = 1.35 * max(lspace, symbol_space(right))
            if math.hypot(lx - rx, ly - ry) <= threshold:
                dense.add(str(left["id"]))
                dense.add(str(right["id"]))
    return dense


def pre_sam2_ambiguous_ids(symbols: list[dict[str, Any]]) -> set[str]:
    """Select relation-ambiguous symbols using detector boxes only."""
    selected = dense_notehead_ids(symbols)
    noteheads = [item for item in symbols if item.get("class") in NOTEHEAD_CLASSES]
    for beam in (item for item in symbols if item.get("class") == "beam"):
        bx0, by0, bx1, by1 = [float(value) for value in beam["bbox"]]
        bstaff = (beam.get("attributes") or {}).get("staff")
        space = symbol_space(beam)
        nearby: list[str] = []
        for notehead in noteheads:
            nstaff = (notehead.get("attributes") or {}).get("staff")
            if bstaff is not None and nstaff is not None and bstaff != nstaff:
                continue
            nx, ny = symbol_center(notehead)
            if bx0 - space <= nx <= bx1 + space and by0 - 5.0 * space <= ny <= by1 + 5.0 * space:
                nearby.append(str(notehead["id"]))
        if len(nearby) >= 2:
            selected.add(str(beam["id"]))
            selected.update(nearby)
    return selected


def mask_quality(
    symbol: dict[str, Any], mask: dict[str, Any], dense_noteheads: set[str]
) -> tuple[bool, str, dict[str, float]]:
    cls = str(symbol.get("class") or "")
    if cls.startswith("flag"):
        rule = (0.65, 0.10, 1.60)
    else:
        rule = QUALITY_RULES.get(cls)
    if rule is None:
        return False, "class_not_selected", {}
    if cls in NOTEHEAD_CLASSES and str(symbol["id"]) not in dense_noteheads:
        return False, "notehead_not_dense", {}

    x0, y0, x1, y1 = [float(v) for v in symbol["bbox"]]
    bbox_area = max(1.0, (x1 - x0) * (y1 - y0))
    score = float(mask.get("mask_score") or 0.0)
    area_ratio = float(mask.get("mask_area") or 0.0) / bbox_area
    min_score, min_ratio, max_ratio = rule
    details = {
        "score": score,
        "area_ratio": area_ratio,
        "min_score": min_score,
        "min_area_ratio": min_ratio,
        "max_area_ratio": max_ratio,
    }
    if score < min_score:
        return False, "score_below_threshold", details
    if not min_ratio <= area_ratio <= max_ratio:
        return False, "area_ratio_out_of_range", details
    return True, "accepted", details


def selective_payload(
    payload: dict[str, Any],
    masks_payload: dict[str, Any],
    pre_gate_ids: set[str] | None = None,
    allowed_classes: set[str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    result = copy.deepcopy(payload)
    masks = {str(item["symbol_id"]): item for item in masks_payload.get("masks", [])}
    dense = dense_notehead_ids(result.get("symbols", []))
    reasons: Counter[str] = Counter()
    accepted_by_class: Counter[str] = Counter()
    quality_by_id: dict[str, dict[str, Any]] = {}

    for symbol in result.get("symbols", []):
        symbol_id = str(symbol["id"])
        symbol_class = str(symbol.get("class") or "")
        if allowed_classes is not None and symbol_class not in allowed_classes:
            reasons["class_gate_fallback"] += 1
            quality_by_id[symbol_id] = {"accepted": False, "reason": "class_gate_fallback"}
            continue
        if pre_gate_ids is not None and symbol_id not in pre_gate_ids:
            reasons["pre_gate_box_confident"] += 1
            quality_by_id[symbol_id] = {"accepted": False, "reason": "pre_gate_box_confident"}
            continue
        mask = masks.get(symbol_id)
        if mask is None:
            reasons["mask_missing"] += 1
            continue
        accepted, reason, details = mask_quality(symbol, mask, dense)
        reasons[reason] += 1
        quality_by_id[symbol_id] = {"accepted": accepted, "reason": reason, **details}
        if not accepted:
            continue
        symbol["mask"] = mask
        source = str(symbol.get("source") or "unknown")
        if "+sam2" not in source:
            symbol["source"] = f"{source}+sam2"
        accepted_by_class[str(symbol.get("class") or "unknown")] += 1

    report = {
        "input_symbols": len(result.get("symbols", [])),
        "dense_noteheads": len(dense),
        "accepted_masks": sum(accepted_by_class.values()),
        "accepted_by_class": dict(sorted(accepted_by_class.items())),
        "reasons": dict(sorted(reasons.items())),
        "quality_by_symbol": quality_by_id,
        "quality_rules": {key: list(value) for key, value in QUALITY_RULES.items()},
        "pre_gate_enabled": pre_gate_ids is not None,
        "pre_gate_selected": len(pre_gate_ids or set()),
        "allowed_classes": sorted(allowed_classes) if allowed_classes is not None else None,
    }
    result["sam2_selection"] = report
    return result, report


def _points(symbol: dict[str, Any], boundary: bool = False) -> list[tuple[float, float]]:
    if boundary:
        mask = symbol.get("mask") or {}
        values = mask.get("points") or []
        if values:
            return [(float(point[0]), float(point[1])) for point in values]
    skeleton = symbol.get("skeleton") or {}
    values = skeleton.get("points") or []
    return [(float(point[0]), float(point[1])) for point in values]


def _minimum_distance(
    left: list[tuple[float, float]], right: list[tuple[float, float]]
) -> float | None:
    if not left or not right:
        return None
    return min(math.hypot(ax - bx, ay - by) for ax, ay in left for bx, by in right)


def _has_sam2_geometry(symbol: dict[str, Any] | None) -> bool:
    return symbol is not None and str(symbol.get("mask_source") or "").startswith("sam2_")


def fuse_mask_contact_relations(shapes: dict[str, Any]) -> tuple[dict[str, Any], dict[str, int]]:
    result = copy.deepcopy(shapes)
    symbols = {str(item["id"]): item for item in result.get("symbols", [])}
    kept = []
    stats: Counter[str] = Counter()

    for relation in result.get("relations", []):
        relation_type = relation.get("type")
        source = symbols.get(str(relation.get("source")))
        if not _has_sam2_geometry(source):
            kept.append(relation)
            stats["bbox_fallback_relations"] += 1
            continue

        updated = copy.deepcopy(relation)
        evidence = dict(updated.get("evidence") or {})
        space = symbol_space(source)

        if relation_type == "beam_stem_group":
            source_points = _points(source)
            target_rows = []
            for target_id in relation.get("targets", []):
                target = symbols.get(str(target_id))
                target_points = _points(target) if target else []
                endpoints = target_points[:1] + target_points[-1:] if target_points else []
                distance = _minimum_distance(source_points, endpoints)
                if distance is not None:
                    target_rows.append((str(target_id), distance))
            threshold = 0.35 * space
            accepted_targets = [target_id for target_id, distance in target_rows if distance <= threshold]
            if target_rows and not accepted_targets:
                stats["beam_relations_removed"] += 1
                continue
            if accepted_targets:
                removed = len(updated.get("targets", [])) - len(accepted_targets)
                stats["beam_targets_removed"] += max(0, removed)
                updated["targets"] = accepted_targets
                distances = [distance for target_id, distance in target_rows if target_id in accepted_targets]
                contact_score = sum(max(0.0, 1.0 - distance / max(1e-6, threshold)) for distance in distances) / len(distances)
                updated["score"] = 0.7 * float(updated.get("score") or 0.0) + 0.3 * contact_score
                evidence["mask_contact"] = {
                    "mean_distance": sum(distances) / len(distances),
                    "threshold": threshold,
                    "contact_score": contact_score,
                }
                stats["beam_relations_fused"] += 1

        elif relation_type == "notehead_stem_attachment" and source.get("class") in NOTEHEAD_CLASSES:
            boundary = _points(source, boundary=True)
            target = symbols.get(str((relation.get("targets") or [None])[0]))
            target_points = _points(target) if target else []
            endpoints = target_points[:1] + target_points[-1:] if target_points else []
            distance = _minimum_distance(boundary, endpoints)
            threshold = 0.55 * space
            if distance is not None:
                if distance > threshold:
                    stats["note_stem_relations_removed"] += 1
                    continue
                contact_score = max(0.0, 1.0 - distance / max(1e-6, threshold))
                updated["score"] = 0.7 * float(updated.get("score") or 0.0) + 0.3 * contact_score
                evidence["mask_contact"] = {
                    "distance": distance,
                    "threshold": threshold,
                    "contact_score": contact_score,
                }
                stats["note_stem_relations_fused"] += 1

        updated["evidence"] = evidence
        kept.append(updated)

    for index, relation in enumerate(kept):
        relation["id"] = f"rel_{index:06d}"
    result["relations"] = kept
    result.setdefault("v2_1_summary", {})["relations"] = len(kept)
    result.setdefault("ablation", {})["mask_contact_fusion"] = True
    return result, dict(sorted(stats.items()))


def fuse_mask_safe_residual(
    box_shapes: dict[str, Any], mask_shapes: dict[str, Any], weight: float = 0.15
) -> tuple[dict[str, Any], dict[str, int]]:
    """Add positive mask-contact evidence without changing box graph topology."""
    result = copy.deepcopy(box_shapes)
    box_symbols = {str(item["id"]): item for item in result.get("symbols", [])}
    mask_symbols = {str(item["id"]): item for item in mask_shapes.get("symbols", [])}
    stats: Counter[str] = Counter()

    for relation in result.get("relations", []):
        source_id = str(relation.get("source"))
        source = mask_symbols.get(source_id)
        if not _has_sam2_geometry(source):
            stats["box_only_relations"] += 1
            continue
        relation_type = relation.get("type")
        space = symbol_space(source)
        contact_scores: list[float] = []

        if relation_type == "beam_stem_group":
            source_points = _points(source)
            threshold = 0.35 * space
            for target_id in relation.get("targets", []):
                target = box_symbols.get(str(target_id))
                target_points = _points(target) if target else []
                endpoints = target_points[:1] + target_points[-1:] if target_points else []
                distance = _minimum_distance(source_points, endpoints)
                if distance is not None and distance <= threshold:
                    contact_scores.append(max(0.0, 1.0 - distance / max(1e-6, threshold)))

        elif relation_type == "notehead_stem_attachment" and source.get("class") in NOTEHEAD_CLASSES:
            boundary = _points(source, boundary=True)
            threshold = 0.55 * space
            for target_id in relation.get("targets", []):
                target = box_symbols.get(str(target_id))
                target_points = _points(target) if target else []
                endpoints = target_points[:1] + target_points[-1:] if target_points else []
                distance = _minimum_distance(boundary, endpoints)
                if distance is not None and distance <= threshold:
                    contact_scores.append(max(0.0, 1.0 - distance / max(1e-6, threshold)))

        if not contact_scores:
            stats["mask_no_positive_contact"] += 1
            continue
        contact_score = sum(contact_scores) / len(contact_scores)
        old_score = float(relation.get("score") or 0.0)
        relation["score"] = old_score + weight * contact_score * max(0.0, 1.0 - old_score)
        evidence = dict(relation.get("evidence") or {})
        evidence["mask_safe_residual"] = {
            "contact_score": contact_score,
            "weight": weight,
            "topology_preserved": True,
        }
        relation["evidence"] = evidence
        stats["mask_relations_boosted"] += 1

    result.setdefault("ablation", {})["mask_safe_residual"] = True
    stats["relations_before"] = len(box_shapes.get("relations", []))
    stats["relations_after"] = len(result.get("relations", []))
    return result, dict(sorted(stats.items()))


def apply_mask_rerank(
    residual_shapes: dict[str, Any],
    box_uncertainty_margin: float = 0.15,
    min_contact: float = 0.35,
    min_contact_margin: float = 0.15,
) -> tuple[dict[str, Any], dict[str, int]]:
    """Let mask contact resolve only uncertain notehead-stem candidate ties."""
    result = copy.deepcopy(residual_shapes)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for relation in result.get("relations", []):
        if relation.get("type") == "notehead_stem_attachment":
            grouped.setdefault(str(relation.get("source")), []).append(relation)
    remove_ids: set[str] = set()
    stats: Counter[str] = Counter()

    for relations in grouped.values():
        if len(relations) < 2:
            continue
        by_box = sorted(relations, key=lambda row: float(row.get("score") or 0.0), reverse=True)
        box_margin = float(by_box[0].get("score") or 0.0) - float(by_box[1].get("score") or 0.0)
        if box_margin > box_uncertainty_margin:
            stats["box_confident_groups"] += 1
            continue

        scored: list[tuple[float, dict[str, Any]]] = []
        for relation in relations:
            evidence = (relation.get("evidence") or {}).get("mask_safe_residual") or {}
            scored.append((float(evidence.get("contact_score") or 0.0), relation))
        scored.sort(key=lambda item: item[0], reverse=True)
        contact_margin = scored[0][0] - scored[1][0]
        if scored[0][0] < min_contact or contact_margin < min_contact_margin:
            stats["mask_ambiguous_groups"] += 1
            continue
        if str(scored[0][1].get("id")) == str(by_box[0].get("id")):
            stats["mask_agrees_with_box"] += 1
            continue
        for _, relation in scored[1:]:
            remove_ids.add(str(relation.get("id")))
        stats["mask_reranked_groups"] += 1

    if remove_ids:
        result["relations"] = [
            relation for relation in result.get("relations", []) if str(relation.get("id")) not in remove_ids
        ]
        for index, relation in enumerate(result["relations"]):
            relation["id"] = f"rel_{index:06d}"
    result.setdefault("v2_1_summary", {})["relations"] = len(result.get("relations", []))
    result.setdefault("ablation", {})["mask_score_consumer"] = True
    stats["relations_removed"] = len(remove_ids)
    return result, dict(sorted(stats.items()))


def build_prediction(
    module: Any, shapes: dict[str, Any], out: Path, variant: str
) -> dict[str, Any]:
    working = copy.deepcopy(shapes)
    working.setdefault("ablation", {}).update({"sam2_variant": variant, "D_relation_graph": True})
    notes = module.build_note_objects({**working, "source_shapes_json": variant})
    params = dict(module.PRESETS["balanced"])
    pruned_notes, _ = module.prune_notes(notes, params)
    pruned_shapes, _ = module.prune_shapes(working, pruned_notes)
    semantics = module.build_semantics(pruned_notes, pruned_shapes)
    write_json(out / "symbols" / "shapes.json", pruned_shapes)
    write_json(out / "notes" / "notes.json", pruned_notes)
    write_json(out / "semantics" / "semantics_v2_1.json", semantics)
    score_path = out / "semantics" / "score.musicxml"
    score_path.parent.mkdir(parents=True, exist_ok=True)
    module.write_musicxml(score_path, module.semantic_to_musicxml(semantics))
    return {
        "score_path": score_path,
        "symbols": len(pruned_shapes.get("symbols", [])),
        "relations": len(pruned_shapes.get("relations", [])),
        "events": int((pruned_notes.get("summary") or {}).get("events") or 0),
    }


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
    labels = {
        "no_sam2": "No SAM2 / bbox fallback",
        "box_only_sam2": "Full box-only SAM2",
        "rebuilt_box": "Rebuilt box graph (same code path)",
        "selective_sam2": "Selective quality-gated SAM2",
        "selective_contact": "Selective SAM2 + mask contact",
        "safe_residual": "Box graph + safe mask residual",
        "safe_rerank": "Safe residual + uncertainty rerank",
    }
    lines = [
        "# Selective SAM2 Debussy demo",
        "",
        f"Protocol: {summary['protocol']}.",
        f"Pages: {', '.join(summary['pages'])}.",
        "",
        "| Variant | Precision | Recall | Event F1 | Predicted events |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for key, label in labels.items():
        row = summary["aggregate"][key]
        lines.append(
            f"| {label} | {row['precision']:.3f} | {row['recall']:.3f} | {row['f1']:.3f} | {row['predicted_events']} |"
        )
    lines.extend(
        [
            "",
            f"Best selective delta over the stronger frozen baseline: {summary['best_selective_delta']:+.3f} Event-F1 points.",
            f"Decision: **{summary['decision']}**.",
            "",
            "This three-page demo is a stop/go diagnostic, not a significance claim.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate selective quality-gated SAM2 on fresh Debussy pages.")
    parser.add_argument("--pages", default="test_0004,test_0005,test_0006")
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--pre-gate",
        action="store_true",
        help="Gate SAM2 attachment using only box-observable relation ambiguity.",
    )
    parser.add_argument(
        "--mask-classes",
        default="",
        help="Optional comma-separated source-domain class gate, e.g. beam.",
    )
    args = parser.parse_args()
    page_ids = [item.strip() for item in args.pages.split(",") if item.strip()]
    allowed_classes = {item.strip() for item in args.mask_classes.split(",") if item.strip()} or None
    module = debussy_eval.load_archived_ablation_module()
    out_root = args.out_root.resolve()
    rows = []

    for page_id in page_ids:
        started = time.perf_counter()
        page_dir = FROZEN / page_id
        symbols_path = page_dir / "symbols" / "symbols_v2.json"
        masks_path = page_dir / "sam2" / "masks_all.json"
        gt = DATA / f"{page_id}.musicxml"
        for required in (symbols_path, masks_path, gt):
            if not required.exists():
                raise FileNotFoundError(required)

        payload = read_json(symbols_path)
        pre_gate_ids = pre_sam2_ambiguous_ids(payload.get("symbols", [])) if args.pre_gate else None
        selective, gate_report = selective_payload(
            payload,
            read_json(masks_path),
            pre_gate_ids,
            allowed_classes,
        )
        selective_path = out_root / page_id / "symbols" / "symbols_selective_sam2.json"
        write_json(selective_path, selective)
        shapes = module.extract_shapes(selective, selective_path)
        shapes.setdefault("ablation", {})["sam2_quality_gate"] = True

        selective_built = build_prediction(
            module, shapes, out_root / "selective_sam2" / page_id, "selective_sam2"
        )
        contact_shapes, contact_report = fuse_mask_contact_relations(shapes)
        contact_built = build_prediction(
            module,
            contact_shapes,
            out_root / "selective_contact" / page_id,
            "selective_contact",
        )

        box_shapes_path = ABLATION / "A1B1C0D1" / page_id / "symbols" / "shapes.json"
        box_shapes = read_json(box_shapes_path)
        rebuilt_box = build_prediction(
            module,
            box_shapes,
            out_root / "rebuilt_box" / page_id,
            "rebuilt_box",
        )
        residual_shapes, residual_report = fuse_mask_safe_residual(box_shapes, shapes)
        residual_built = build_prediction(
            module,
            residual_shapes,
            out_root / "safe_residual" / page_id,
            "safe_residual",
        )
        reranked_shapes, rerank_report = apply_mask_rerank(residual_shapes)
        reranked_built = build_prediction(
            module,
            reranked_shapes,
            out_root / "safe_rerank" / page_id,
            "safe_rerank",
        )

        no_sam_score = ABLATION / "A1B1C0D1" / page_id / "semantics" / "score.musicxml"
        full_sam_score = ABLATION / "A1B1C1D1" / page_id / "semantics" / "score.musicxml"
        row = {
            "page": page_id,
            "no_sam2": debussy_eval.counts(gt, no_sam_score),
            "box_only_sam2": debussy_eval.counts(gt, full_sam_score),
            "rebuilt_box": debussy_eval.counts(gt, rebuilt_box["score_path"]),
            "selective_sam2": debussy_eval.counts(gt, selective_built["score_path"]),
            "selective_contact": debussy_eval.counts(gt, contact_built["score_path"]),
            "safe_residual": debussy_eval.counts(gt, residual_built["score_path"]),
            "safe_rerank": debussy_eval.counts(gt, reranked_built["score_path"]),
            "gate_report": gate_report,
            "contact_report": contact_report,
            "residual_report": residual_report,
            "rerank_report": rerank_report,
            "selective_symbols": selective_built["symbols"],
            "selective_relations": selective_built["relations"],
            "contact_relations": contact_built["relations"],
            "elapsed_seconds": time.perf_counter() - started,
        }
        rows.append(row)
        write_json(out_root / page_id / "comparison.json", row)
        print({key: value for key, value in row.items() if key not in {"gate_report"}})

    keys = (
        "no_sam2",
        "box_only_sam2",
        "rebuilt_box",
        "selective_sam2",
        "selective_contact",
        "safe_residual",
        "safe_rerank",
    )
    aggregate_rows = {key: aggregate(rows, key) for key in keys}
    stronger_baseline = max(aggregate_rows["no_sam2"]["f1"], aggregate_rows["box_only_sam2"]["f1"])
    best_selective = max(
        aggregate_rows["selective_sam2"]["f1"],
        aggregate_rows["selective_contact"]["f1"],
    )
    delta = float(best_selective) - float(stronger_baseline)
    rebuilt_box_f1 = float(aggregate_rows["rebuilt_box"]["f1"])
    mask_delta_over_rebuilt_box = float(best_selective) - rebuilt_box_f1
    safe_delta_over_rebuilt_box = max(
        float(aggregate_rows["safe_residual"]["f1"]),
        float(aggregate_rows["safe_rerank"]["f1"]),
    ) - rebuilt_box_f1
    summary = {
        "protocol": (
            f"source-domain class gate {sorted(allowed_classes)} plus frozen post-mask thresholds"
            if allowed_classes is not None
            else (
                "box-observable pre-SAM2 ambiguity gate plus frozen post-mask thresholds"
                if args.pre_gate
                else "frozen detector and fixed thresholds from test_0001--test_0003"
            )
        ),
        "pages": page_ids,
        "rows": rows,
        "aggregate": aggregate_rows,
        "best_selective_delta": delta,
        "best_mask_delta_over_rebuilt_box": mask_delta_over_rebuilt_box,
        "safe_delta_over_rebuilt_box": safe_delta_over_rebuilt_box,
        "decision": (
            "run_full"
            if delta > 0 and mask_delta_over_rebuilt_box > 0
            else "stop_and_visualization_only"
        ),
        "publication_status": "demo_only_not_significance_test",
    }
    write_json(out_root / "summary.json", summary)
    write_markdown(summary, out_root / "summary.md")
    print(json.dumps(aggregate_rows, ensure_ascii=False, indent=2))
    print({"best_selective_delta": delta, "decision": summary["decision"]})


if __name__ == "__main__":
    main()
