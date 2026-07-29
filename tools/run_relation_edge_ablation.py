from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.paper_evidence_metrics import (
    aggregate_counts,
    multiset_counts,
    musicxml_events,
    paired_bootstrap_delta,
)


ACTIVE_ORDER = (
    "notehead_stem_attachment",
    "beam_stem_group",
    "slur_tie_notehead_endpoints",
)
PLANNED_INACTIVE = ("ledger_line_notehead_attachment", "accidental_note_attachment")


def filter_relation_type(shapes: Mapping[str, Any], removed_type: str) -> dict[str, Any]:
    filtered = copy.deepcopy(dict(shapes))
    relations = filtered.get("relations") or []
    filtered["relations"] = [edge for edge in relations if edge.get("type") != removed_type]
    return filtered


def load_archived_module() -> Any:
    from tools import run_abcd_module_ablation

    return run_abcd_module_ablation


def load_assemble_module() -> Any:
    from tools import assemble_v2_1_notes

    return assemble_v2_1_notes


def build_notes_from_shapes(assemble_module: Any, shapes: Mapping[str, Any]) -> dict[str, Any]:
    return assemble_module.build_note_objects(dict(shapes))


def apply_relation_ablation_to_notes(
    original: Mapping[str, Any], reassembled: Mapping[str, Any], removed_type: str
) -> dict[str, Any]:
    patched = copy.deepcopy(dict(original))
    keys_by_type = {
        "notehead_stem_attachment": (
            "stem_id",
            "beam_ids",
            "beam_count",
            "duration_hint",
            "source_symbol_ids",
        ),
        "beam_stem_group": ("beam_ids", "beam_count", "duration_hint", "source_symbol_ids"),
        "slur_tie_notehead_endpoints": ("slur_or_tie_ids", "source_symbol_ids"),
    }
    keys = keys_by_type[removed_type]
    for section in ("notes", "events"):
        rebuilt_by_id = {
            str(item.get("id")): item for item in reassembled.get(section, []) if item.get("id")
        }
        for item in patched.get(section, []):
            rebuilt = rebuilt_by_id.get(str(item.get("id")))
            if rebuilt is None:
                if removed_type == "notehead_stem_attachment":
                    item["stem_id"] = None
                    item["beam_ids"] = []
                    item["beam_count"] = 0
                elif removed_type == "beam_stem_group":
                    item["beam_ids"] = []
                    item["beam_count"] = 0
                elif removed_type == "slur_tie_notehead_endpoints":
                    item["slur_or_tie_ids"] = []
                continue
            for key in keys:
                if key in rebuilt:
                    item[key] = copy.deepcopy(rebuilt[key])
                else:
                    item.pop(key, None)
    return patched


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _variant_name(removed_type: str | None) -> str:
    return "full" if removed_type is None else f"without_{removed_type}"


def run_ablation(
    out: Path, limit: int, bootstrap_samples: int, seed: int
) -> dict[str, Any]:
    source_root = ROOT / "outputs" / "debussy_abcd_ablation" / "A1B1C1D1"
    gold_root = ROOT / "data" / "ijcv_samples" / "debussy-omr-transfer24"
    accepted_path = ROOT / "outputs" / "ijcv_repro" / "order_invariant_event_f1" / "debussy_transfer24.json"
    accepted = json.loads(accepted_path.read_text(encoding="utf-8"))
    accepted_pages = {row["page_id"]: row["staff"] for row in accepted["per_page_counts"]}
    module = load_archived_module()
    assemble = load_assemble_module()

    edge_counts: Counter[str] = Counter()
    edge_pages: defaultdict[str, int] = defaultdict(int)
    cached: list[tuple[str, Path, dict[str, Any], dict[str, Any], Path]] = []
    for index in range(limit):
        page_id = f"test_{index:04d}"
        page_root = source_root / page_id
        notes_path = page_root / "notes" / "notes.json"
        shapes_path = page_root / "symbols" / "shapes.json"
        gold_path = gold_root / f"{page_id}.musicxml"
        if not notes_path.exists() or not shapes_path.exists() or not gold_path.exists():
            raise FileNotFoundError(f"Missing frozen page artifacts for {page_id}")
        original_notes = json.loads(notes_path.read_text(encoding="utf-8"))
        shapes = json.loads(shapes_path.read_text(encoding="utf-8"))
        seen: set[str] = set()
        for edge in shapes.get("relations") or []:
            edge_type = str(edge.get("type"))
            edge_counts[edge_type] += 1
            seen.add(edge_type)
        for edge_type in seen:
            edge_pages[edge_type] += 1
        cached.append((page_id, gold_path, original_notes, shapes, shapes_path))

    unknown = sorted(set(edge_counts) - set(ACTIVE_ORDER))
    if unknown:
        raise RuntimeError(f"Unreviewed relation types present: {unknown}")
    active = [edge_type for edge_type in ACTIVE_ORDER if edge_counts[edge_type] > 0]
    variants: list[str | None] = [None, *active]
    rows: list[dict[str, Any]] = []
    out.mkdir(parents=True, exist_ok=True)
    for removed_type in variants:
        variant = _variant_name(removed_type)
        for page_id, gold_path, original_notes, shapes, shapes_path in cached:
            variant_shapes = shapes if removed_type is None else filter_relation_type(shapes, removed_type)
            if removed_type is None:
                notes = original_notes
            else:
                reassembled = build_notes_from_shapes(assemble, variant_shapes)
                notes = apply_relation_ablation_to_notes(original_notes, reassembled, removed_type)
            semantics = module.build_semantics(notes, variant_shapes)
            tree = module.semantic_to_musicxml(semantics)
            prediction = out / "variants" / variant / page_id / "score.musicxml"
            prediction.parent.mkdir(parents=True, exist_ok=True)
            tree.write(prediction, encoding="utf-8", xml_declaration=True)
            counts = multiset_counts(musicxml_events(gold_path), musicxml_events(prediction), "joint")
            row = {
                "page_id": page_id,
                "variant": variant,
                "removed_relation_type": removed_type,
                "source_shapes": str(shapes_path),
                **counts,
            }
            rows.append(row)
            if removed_type is None and counts != accepted_pages[page_id]:
                raise RuntimeError(
                    f"Full reproduction failed for {page_id}: {counts} != {accepted_pages[page_id]}"
                )

    summaries: dict[str, Any] = {}
    full_rows = [row for row in rows if row["variant"] == "full"]
    for removed_type in variants:
        variant = _variant_name(removed_type)
        variant_rows = [row for row in rows if row["variant"] == variant]
        entry: dict[str, Any] = {"summary": aggregate_counts(variant_rows)}
        if removed_type is not None:
            effect = paired_bootstrap_delta(
                full_rows, variant_rows, samples=bootstrap_samples, seed=seed + active.index(removed_type)
            )
            wins = ties = losses = 0
            for full, ablated in zip(full_rows, variant_rows):
                full_f1 = float(aggregate_counts([full])["f1"])
                ablated_f1 = float(aggregate_counts([ablated])["f1"])
                if full_f1 > ablated_f1:
                    wins += 1
                elif full_f1 < ablated_f1:
                    losses += 1
                else:
                    ties += 1
            entry.update({"effect": effect, "full_wins": wins, "ties": ties, "full_losses": losses})
        summaries[variant] = entry

    accepted_subset = [accepted_pages[f"test_{index:04d}"] for index in range(limit)]
    observed_full = float(summaries["full"]["summary"]["f1"])
    expected_full = float(aggregate_counts(accepted_subset)["f1"])
    reproduction_difference = abs(observed_full - expected_full)
    if reproduction_difference > 1e-9:
        raise RuntimeError(
            f"Full aggregate reproduction failed: {observed_full} != {expected_full}"
        )

    payload = {
        "dataset": "HugoSchtr/debussy-omr-system-lvl transfer24",
        "pages": limit,
        "protocol": "cached frozen A1B1C1D1 shapes; remove one active relation family and deterministically rebuild semantics",
        "metric": "order_invariant_duration_pitch_event_f1",
        "seed": seed,
        "bootstrap_samples": bootstrap_samples,
        "relation_inventory": {
            edge_type: {"edges": edge_counts[edge_type], "pages": edge_pages[edge_type]}
            for edge_type in ACTIVE_ORDER
        },
        "inactive_planned_relations": list(PLANNED_INACTIVE),
        "variants": summaries,
        "reproduction_gate": {
            "observed_f1": observed_full,
            "accepted_f1": expected_full,
            "absolute_f1_difference": reproduction_difference,
            "tolerance": 1e-9,
            "passed": True,
        },
        "sources": [str(source_root), str(gold_root), str(accepted_path)],
    }
    (out / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_jsonl(out / "per_page.jsonl", rows)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Ablate active relation edge families on frozen Debussy outputs.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=24)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260720)
    args = parser.parse_args()
    if not 1 <= args.limit <= 24:
        parser.error("--limit must be between 1 and 24")
    payload = run_ablation(args.out.resolve(), args.limit, args.bootstrap_samples, args.seed)
    print(json.dumps(payload["variants"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
