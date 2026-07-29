from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parent.parent
TOOLS = REPO / "tools"
for candidate in (REPO, TOOLS):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from evaluate_kern_text_proxy_metrics import (  # noqa: E402
    legacy_semantic_to_pseudo_ekern,
    metric_block,
    normalized_tokens,
)
from tools.polish_canonical_serialization import semantic_to_canonical_ekern  # noqa: E402
from tools.polish_structural_oracles import (  # noqa: E402
    bag_oracle,
    chord_diagnostics,
    event_binding_oracle,
    exact_onset_order_oracle,
    flatten_gold,
    flatten_predicted,
    parse_ekern_hierarchy,
    predicted_onsets,
    sha256_files,
    staff_assignment_oracle,
    voice_observability,
    write_json,
)


def evaluate(source_root: Path, gt_root: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    semantics_paths = sorted(source_root.glob("*/semantics/semantics_v2_1.json"))
    gt_paths: list[Path] = []
    corpus_ref: list[str] = []
    corpus_legacy: list[str] = []
    corpus_r0: list[str] = []
    corpus_r3: list[str] = []
    gold_onsets_all = []
    pred_onsets_all = []
    r3_totals = {
        "matched_onset_groups": 0,
        "gold_onset_groups": 0,
        "pred_onset_groups": 0,
        "tokens_in_exact_groups": 0,
    }
    predicted_systems = 0
    r2_rows: list[dict[str, Any]] = []
    r4_rows: list[dict[str, Any]] = []
    r5_rows: list[dict[str, Any]] = []
    r6_rows: list[dict[str, Any]] = []
    bag_rows: list[dict[str, Any]] = []
    for semantics_path in semantics_paths:
        sample = semantics_path.parent.parent.name
        gt_path = gt_root / f"{sample}.ekern.txt"
        if not gt_path.exists():
            continue
        gt_paths.append(gt_path)
        semantics = json.loads(semantics_path.read_text(encoding="utf-8"))
        gold_onsets = parse_ekern_hierarchy(gt_path.read_text(encoding="utf-8"), normalized_tokens)
        pred = predicted_onsets(semantics, normalized_tokens)
        predicted_systems += len({onset.system for onset in pred})
        ref = flatten_gold(gold_onsets)
        legacy = normalized_tokens(legacy_semantic_to_pseudo_ekern(semantics))
        r0 = normalized_tokens(semantic_to_canonical_ekern(semantics))
        r3, r3_structure = exact_onset_order_oracle(gold_onsets, pred)
        for key in r3_totals:
            r3_totals[key] += int(r3_structure[key])
        r2 = staff_assignment_oracle(gold_onsets, pred)
        r4 = chord_diagnostics(gold_onsets, pred)
        r5 = voice_observability(gold_onsets, pred)
        r6 = event_binding_oracle(ref, flatten_predicted(pred))
        bag = bag_oracle(ref, flatten_predicted(pred))
        r2_rows.append(r2)
        r4_rows.append(r4)
        r5_rows.append(r5)
        r6_rows.append(r6)
        bag_rows.append(bag)
        rows.append(
            {
                "sample": sample,
                "legacy_staff_block": metric_block(ref, legacy),
                "R0_canonical": metric_block(ref, r0),
                "R2_staff": r2,
                "R3_exact_onset_order": {**metric_block(ref, r3), **r3_structure},
                "R4_chord": r4,
                "R5_voice": r5,
                "R6_binding": r6,
                "bag_oracle": bag,
            }
        )
        corpus_ref.extend(ref)
        corpus_legacy.extend(legacy)
        corpus_r0.extend(r0)
        corpus_r3.extend(r3)
        gold_onsets_all.extend(gold_onsets)
        pred_onsets_all.extend(pred)

    r0_metric = metric_block(corpus_ref, corpus_r0)
    legacy_metric = metric_block(corpus_ref, corpus_legacy)
    r3_metric = metric_block(corpus_ref, corpus_r3)
    ref_tokens = sum(row["ref_tokens"] for row in bag_rows)
    r2_constrained_errors = sum(row["staff_constrained_errors"] for row in r2_rows)
    r2_oracle_errors = sum(row["oracle_reassigned_errors"] for row in r2_rows)
    r2_summary = {
        "scope": "page-constrained",
        "ref_tokens": ref_tokens,
        "staff_constrained_matches": sum(row["staff_constrained_matches"] for row in r2_rows),
        "staff_constrained_SER_lower_bound": 100.0 * r2_constrained_errors / ref_tokens,
        "oracle_reassigned_matches": sum(row["oracle_reassigned_matches"] for row in r2_rows),
        "oracle_reassigned_SER_lower_bound": 100.0 * r2_oracle_errors / ref_tokens,
        "potential_SER_improvement": 100.0 * (r2_constrained_errors - r2_oracle_errors) / ref_tokens,
    }
    r4_summary = {
        "gold_chords": sum(row["gold_chords"] for row in r4_rows),
        "pred_chords": sum(row["pred_chords"] for row in r4_rows),
        "exact_chord_matches": sum(row["exact_chord_matches"] for row in r4_rows),
        "proxy_SER_effect": 0.0,
        "proxy_SER_effect_reason": "Current normalization splits spaces/tabs and discards chord-cell boundaries.",
    }
    r5_summary = {
        "gold_events_in_split_voice_spines": sum(
            row["gold_events_in_split_voice_spines"] for row in r5_rows
        ),
        "gold_multivoice_onsets": sum(row["gold_multivoice_onsets"] for row in r5_rows),
        "predicted_events_with_voice_field": sum(
            row["predicted_events_with_voice_field"] for row in r5_rows
        ),
        "oracle_status": "unavailable",
        "reason": "Prediction semantics do not expose voice assignment.",
    }
    r6_errors = sum(row["minimum_errors"] for row in r6_rows)
    r6_summary = {
        "scope": "page-constrained",
        "current_token_bag_matches": sum(row["current_token_bag_matches"] for row in r6_rows),
        "oracle_note_binding_matches": sum(row["oracle_note_binding_matches"] for row in r6_rows),
        "fixed_rest_matches": sum(row["fixed_rest_matches"] for row in r6_rows),
        "oracle_total_matches": sum(row["oracle_total_matches"] for row in r6_rows),
        "oracle_SER_lower_bound": 100.0 * r6_errors / ref_tokens,
        "uses_target_annotations": True,
    }
    bag_errors = sum(row["minimum_errors"] for row in bag_rows)
    bag_page_summary = {
        "scope": "page-constrained",
        "matches": sum(row["matches"] for row in bag_rows),
        "ref_tokens": ref_tokens,
        "pred_tokens": sum(row["pred_tokens"] for row in bag_rows),
        "SER_lower_bound": 100.0 * bag_errors / ref_tokens,
    }
    return {
        "dataset": "PRAIG/polish-scores test24",
        "diagnostic_only": True,
        "uses_target_annotations": True,
        "warning": "Oracle diagnostics use Polish test annotations and are not deployable or zero-shot results.",
        "observability": {
            "R0_canonical": "available",
            "R1_system": "unavailable: dataset has image+transcription only; eKern has no page-system boundaries",
            "R2_staff": "available as a staff-constrained versus reassigned token-bag lower bound",
            "R3_onset": "available for exact predicted onset groups matched to gold eKern time rows",
            "R4_chord": "structural counts available; proxy SER is invariant because normalization removes cell boundaries",
            "R5_voice": "unavailable for prediction accuracy: gold spine splits exist but prediction semantics contain no voice field",
            "R6_binding": "available as a max-flow upper bound preserving predicted pitch and duration marginals",
            "bag_oracle": "available",
        },
        "samples": len(rows),
        "summary": {
            "legacy_staff_block": legacy_metric,
            "R0_canonical": {
                **r0_metric,
                "SER_delta_from_legacy": r0_metric["error_percent"] - legacy_metric["error_percent"],
            },
            "R1_system": None,
            "R2_staff": r2_summary,
            "R3_exact_onset_order": {
                **r3_metric,
                **r3_totals,
                "SER_delta_from_R0": r3_metric["error_percent"] - r0_metric["error_percent"],
            },
            "R4_chord": r4_summary,
            "R5_voice": r5_summary,
            "R6_binding": r6_summary,
            "bag_oracle": {
                "page_constrained": bag_page_summary,
                "corpus_relaxed": bag_oracle(corpus_ref, flatten_predicted(pred_onsets_all)),
            },
        },
        "provenance": {
            "source_root": str(source_root.resolve()),
            "gt_root": str(gt_root.resolve()),
            "sample_manifest": [path.parent.parent.name for path in semantics_paths if (gt_root / f"{path.parent.parent.name}.ekern.txt").exists()],
            "semantic_output_sha256": sha256_files(semantics_paths),
            "gold_transcription_sha256": sha256_files(gt_paths),
            "serializer_version": "canonical_event_linearizer_v1",
            "metric_protocol": "evaluate_kern_text_proxy_metrics.normalized_tokens + rapidfuzz Levenshtein",
            "normalization": "metadata lines and null cells removed; spaces/tabs flattened; token reduced to duration:pitch",
            "predicted_systems": predicted_systems,
            "gold_systems": None,
            "historical_metric_reconciliation": {
                "96.1367": "This run: full A1B1C1D1/box-only semantics with the legacy staff-block serializer.",
                "98.73": "July 11 scan-aug crop-fusion snapshot with fixed OLiMPiC-selected thresholds; different semantic outputs/configuration.",
                "98.86": "July 11 direct StaffOMR-SAM V2.1 snapshot without crop fusion; different semantic outputs/configuration.",
                "94.694": "A1B1C0D0 ablation with relation graph D disabled; not the full-stack result.",
            },
        },
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run observable Polish structural oracle diagnostics.")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--gt-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = evaluate(args.source_root, args.gt_root)
    write_json(args.output, payload)
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
