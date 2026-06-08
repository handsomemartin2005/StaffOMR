from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean
from typing import Any

from omr_v2_common import read_json, write_json


def blank_class_row() -> dict[str, Any]:
    return {
        "symbols": 0,
        "polygon": 0,
        "skeleton": 0,
        "geometry_pass": 0,
        "geometry_warn": 0,
        "geometry_fail": 0,
        "fill_ratios": [],
        "staff_overlap_ratios": [],
        "mask_scores": [],
    }


def add_value(target: list[float], value: Any) -> None:
    if value is None:
        return
    try:
        target.append(float(value))
    except (TypeError, ValueError):
        return


def finalize_row(row: dict[str, Any]) -> dict[str, Any]:
    total = max(1, int(row["symbols"]))
    result = {key: value for key, value in row.items() if not isinstance(value, list)}
    result["polygon_rate"] = row["polygon"] / total
    result["skeleton_rate"] = row["skeleton"] / total
    result["geometry_pass_rate"] = row["geometry_pass"] / total
    result["geometry_warn_rate"] = row["geometry_warn"] / total
    result["geometry_fail_rate"] = row["geometry_fail"] / total
    result["avg_fill_ratio"] = mean(row["fill_ratios"]) if row["fill_ratios"] else None
    result["avg_staff_overlap_ratio"] = mean(row["staff_overlap_ratios"]) if row["staff_overlap_ratios"] else None
    result["avg_mask_score"] = mean(row["mask_scores"]) if row["mask_scores"] else None
    return result


def relation_summary(relations: list[dict[str, Any]]) -> dict[str, Any]:
    by_type: dict[str, dict[str, Any]] = {}
    for relation in relations:
        key = relation.get("type", "unknown")
        row = by_type.setdefault(key, {"count": 0, "scores": []})
        row["count"] += 1
        add_value(row["scores"], relation.get("score"))
    finalized = {}
    for key, row in by_type.items():
        finalized[key] = {
            "count": row["count"],
            "avg_score": mean(row["scores"]) if row["scores"] else None,
        }
    return finalized


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize weak V2.1 shape and relation metrics.")
    parser.add_argument("--shapes-json", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, default=Path("outputs/v2_1/evaluation_v2_1_shapes.json"))
    args = parser.parse_args()

    payload: dict[str, Any] = read_json(args.shapes_json)
    rows: dict[str, dict[str, Any]] = {}
    totals = blank_class_row()
    thin_classes = {"stem", "beam", "ledger_line", "barline", "slur_or_tie"}
    thin_total = 0
    thin_with_skeleton = 0
    for symbol in payload.get("symbols", []):
        cls = symbol.get("class", "unknown")
        row = rows.setdefault(cls, blank_class_row())
        for target in (row, totals):
            target["symbols"] += 1
            if symbol.get("mask") and symbol["mask"].get("points"):
                target["polygon"] += 1
            if symbol.get("skeleton") and symbol["skeleton"].get("points"):
                target["skeleton"] += 1
            status = symbol.get("geometry_check", "unknown")
            if status == "pass":
                target["geometry_pass"] += 1
            elif status == "warn":
                target["geometry_warn"] += 1
            elif status == "fail":
                target["geometry_fail"] += 1
            shape = symbol.get("shape") or {}
            add_value(target["fill_ratios"], shape.get("fill_ratio"))
            add_value(target["staff_overlap_ratios"], shape.get("staff_overlap_ratio"))
            add_value(target["mask_scores"], symbol.get("mask_score"))
        if cls in thin_classes:
            thin_total += 1
            if symbol.get("skeleton") and symbol["skeleton"].get("points"):
                thin_with_skeleton += 1

    relations = payload.get("relations", [])
    relation_scores = [float(r.get("score")) for r in relations if r.get("score") is not None]
    result = {
        "shapes_json": str(args.shapes_json),
        "input": payload.get("input"),
        "symbols": len(payload.get("symbols", [])),
        "relations": len(relations),
        "overall": finalize_row(totals),
        "per_class": {key: finalize_row(value) for key, value in sorted(rows.items())},
        "relations_by_type": relation_summary(relations),
        "weak_scores": {
            "thin_skeleton_coverage": thin_with_skeleton / max(1, thin_total),
            "relation_consistency_score": mean(relation_scores) if relation_scores else None,
            "geometry_sanity_pass_rate": totals["geometry_pass"] / max(1, totals["symbols"]),
        },
        "definitions": {
            "geometry_sanity_pass_rate": "Share of symbols whose class-specific geometry checks emitted no warnings or failures.",
            "thin_skeleton_coverage": "Share of stem/beam/ledger_line/barline/slur_or_tie symbols with a V2.1 skeleton polyline.",
            "relation_consistency_score": "Mean heuristic score over V2.1 relation edges; it is not ground-truth accuracy.",
        },
    }
    write_json(args.out_json, result)
    print(
        {
            "out_json": str(args.out_json),
            "symbols": result["symbols"],
            "relations": result["relations"],
            "geometry_sanity_pass_rate": result["weak_scores"]["geometry_sanity_pass_rate"],
            "thin_skeleton_coverage": result["weak_scores"]["thin_skeleton_coverage"],
        }
    )


if __name__ == "__main__":
    main()
