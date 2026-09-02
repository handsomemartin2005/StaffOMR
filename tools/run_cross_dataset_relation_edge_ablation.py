from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.paper_evidence_metrics import aggregate_counts, multiset_counts, musicxml_events, paired_bootstrap_delta
from tools.run_relation_edge_ablation import (
    apply_relation_ablation_to_notes,
    build_notes_from_shapes,
    filter_relation_type,
    load_archived_module,
    load_assemble_module,
)

ACTIVE = ("notehead_stem_attachment", "beam_stem_group")


def dataset_config(name: str) -> dict[str, Any]:
    roots = {
        "grandstaff": (
            ROOT / "outputs/abcd_ablation/grandstaff/A1B1C1D1",
            ROOT / "data/ijcv_samples/grandstaff-lmx-test200-flat",
        ),
        "olimpic": (
            ROOT / "outputs/abcd_ablation/olimpic/A1B1C1D1",
            ROOT / "data/ijcv_samples/olimpic-test200-flat",
        ),
    }
    source_root, gold_root = roots[name]
    page_ids = sorted(path.name for path in source_root.iterdir() if path.is_dir())
    return {"name": name, "source_root": source_root, "gold_root": gold_root, "page_ids": page_ids}


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def run_dataset_ablation(name: str, out: Path, limit: int, bootstrap_samples: int, seed: int) -> dict[str, Any]:
    config = dataset_config(name)
    page_ids = config["page_ids"][:limit]
    if len(page_ids) != limit:
        raise RuntimeError(f"{name}: requested {limit} pages but found {len(page_ids)}")
    module, assemble = load_archived_module(), load_assemble_module()
    cached = []
    inventory: Counter[str] = Counter()
    inventory_pages: defaultdict[str, int] = defaultdict(int)
    for page_id in page_ids:
        page = config["source_root"] / page_id
        notes = json.loads((page / "notes/notes.json").read_text(encoding="utf-8"))
        shapes = json.loads((page / "symbols/shapes.json").read_text(encoding="utf-8"))
        seen = set()
        for edge in shapes.get("relations") or []:
            kind = str(edge.get("type")); inventory[kind] += 1; seen.add(kind)
        for kind in seen: inventory_pages[kind] += 1
        cached.append((page_id, notes, shapes, config["gold_root"] / f"{page_id}.musicxml", page / "semantics/score.musicxml"))

    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    reproduction_failures = []
    for removed in (None, *ACTIVE):
        variant = "full" if removed is None else f"without_{removed}"
        for page_id, original, shapes, gold, frozen in cached:
            variant_shapes = shapes if removed is None else filter_relation_type(shapes, removed)
            notes = original
            if removed is not None:
                rebuilt = build_notes_from_shapes(assemble, variant_shapes)
                notes = apply_relation_ablation_to_notes(original, rebuilt, removed)
            tree = module.semantic_to_musicxml(module.build_semantics(notes, variant_shapes))
            pred = out / "variants" / variant / page_id / "score.musicxml"
            pred.parent.mkdir(parents=True, exist_ok=True)
            tree.write(pred, encoding="utf-8", xml_declaration=True)
            counts = multiset_counts(musicxml_events(gold), musicxml_events(pred), "joint")
            rows.append({"page_id": page_id, "variant": variant, "removed_relation_type": removed, **counts})
            if removed is None:
                check = aggregate_counts([multiset_counts(musicxml_events(frozen), musicxml_events(pred), "joint")])
                if check["f1"] != 100.0: reproduction_failures.append(page_id)

    summaries: dict[str, Any] = {}
    full = [row for row in rows if row["variant"] == "full"]
    for index, removed in enumerate((None, *ACTIVE)):
        variant = "full" if removed is None else f"without_{removed}"
        selected = [row for row in rows if row["variant"] == variant]
        entry: dict[str, Any] = {"summary": aggregate_counts(selected)}
        if removed is not None:
            entry["effect"] = paired_bootstrap_delta(full, selected, samples=bootstrap_samples, seed=seed + index)
        summaries[variant] = entry
    payload = {
        "dataset": name, "pages": limit, "metric": "order_invariant_duration_pitch_event_f1",
        "protocol": "frozen A1B1C1D1 outputs; remove one Debussy-gated active relation family",
        "relation_inventory": {key: {"edges": value, "pages": inventory_pages[key]} for key, value in inventory.items()},
        "variants": summaries,
        "reproduction_gate": {"passed": not reproduction_failures, "failed_pages": reproduction_failures},
    }
    (out / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_jsonl(out / "per_page.jsonl", rows)
    if reproduction_failures: raise RuntimeError(f"Frozen full reproduction failed: {reproduction_failures[:5]}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("grandstaff", "olimpic"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260720)
    args = parser.parse_args()
    result = run_dataset_ablation(args.dataset, args.out.resolve(), args.limit, args.bootstrap_samples, args.seed)
    print(json.dumps(result["variants"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
