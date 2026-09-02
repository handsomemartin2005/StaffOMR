from __future__ import annotations

from pathlib import Path
from typing import Any

from assemble_v2_1_notes import build_note_objects
from evaluate_ekern_our_event_metrics import (
    ekern_text_to_event_tokens,
    evaluate,
    semantics_to_event_tokens,
)
from evaluate_kern_text_proxy_metrics import evaluate_texts, semantic_to_pseudo_ekern
from export_v2_1_semantics import build_semantics
from prune_v2_1_overrecognition import prune_notes, prune_shapes


def evaluate_shapes(
    shapes: dict[str, Any], gt_path: Path, params: dict[str, Any]
) -> dict[str, Any]:
    """Python 3.11-compatible copy of the Polish calibration metric helper."""
    notes = build_note_objects(shapes)
    pruned_notes, dropped_notes = prune_notes(notes, params)
    pruned_shapes, dropped_symbols = prune_shapes(shapes, pruned_notes)
    semantics = build_semantics(pruned_notes, pruned_shapes)
    gt_text = gt_path.read_text(encoding="utf-8")
    gt_tokens = ekern_text_to_event_tokens(gt_text)
    pred_tokens = semantics_to_event_tokens(semantics)
    event_metrics = evaluate(gt_tokens, pred_tokens)
    proxy_metrics = evaluate_texts(gt_text, semantic_to_pseudo_ekern(semantics))
    event = event_metrics["event_error"]
    counts = event_metrics["counts"]
    ref_len = max(1, int(event.get("ref_len") or 0))

    def op_percent(key: str) -> float:
        return 100.0 * float((event.get("ops") or {}).get(key, 0)) / ref_len

    return {
        "gt_events": counts["gt_events"],
        "pred_events": counts["pred_events"],
        "event_error_percent": event["error_percent"],
        "pitch_error_percent": event_metrics["pitch_error"]["error_percent"],
        "duration_error_percent": event_metrics["duration_error"]["error_percent"],
        "cer_proxy_percent": proxy_metrics["normalized"]["CER_proxy"]["error_percent"],
        "ser_proxy_percent": proxy_metrics["normalized"]["SER_proxy"]["error_percent"],
        "under_percent": op_percent("deletions"),
        "over_percent": op_percent("insertions"),
        "substitution_percent": op_percent("substitutions"),
        "count_gap_percent": 100.0
        * abs(counts["pred_events"] - counts["gt_events"])
        / max(1, counts["gt_events"]),
        "notes_before": len(notes.get("notes", []) or []),
        "notes_after": len(pruned_notes.get("notes", []) or []),
        "notes_pruned": len(dropped_notes),
        "symbols_pruned": len(dropped_symbols),
    }
