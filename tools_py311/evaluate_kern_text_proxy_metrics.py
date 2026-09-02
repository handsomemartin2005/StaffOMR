from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Sequence

from rapidfuzz.distance import Levenshtein


_PYC = Path(__file__).with_suffix(".pyc")
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
_SPEC = importlib.util.spec_from_file_location("_evaluate_kern_text_proxy_metrics_legacy", _PYC)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Cannot load legacy metric helpers from {_PYC}")
_LEGACY = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_LEGACY)


normalized_tokens = _LEGACY.normalized_tokens
chars_from_tokens = _LEGACY.chars_from_tokens
legacy_semantic_to_pseudo_ekern = _LEGACY.semantic_to_pseudo_ekern
from tools.polish_canonical_serialization import semantic_to_canonical_ekern


def semantic_to_pseudo_ekern(semantics: dict[str, Any]) -> str:
    """Use canonical piano reading order only for Polish page semantics."""

    dataset = str(semantics.get("dataset") or "").lower()
    input_hint = str(semantics.get("input") or "").replace("\\", "/").lower()
    if dataset == "polish" or "polish" in input_hint:
        return semantic_to_canonical_ekern(semantics)
    return legacy_semantic_to_pseudo_ekern(semantics)
split_raw_tokens = _LEGACY.split_raw_tokens
data_lines = _LEGACY.data_lines


def metric_block(ref: Sequence[Any], hyp: Sequence[Any]) -> dict[str, Any]:
    """Compute the legacy metric using compact native edit operations."""
    counts = {
        "substitutions": 0,
        "deletions": 0,
        "insertions": 0,
        "matches": 0,
    }
    for operation in Levenshtein.editops(ref, hyp):
        if operation.tag == "replace":
            counts["substitutions"] += 1
        elif operation.tag == "delete":
            counts["deletions"] += 1
        elif operation.tag == "insert":
            counts["insertions"] += 1
    counts["matches"] = len(ref) - counts["substitutions"] - counts["deletions"]
    errors = counts["substitutions"] + counts["deletions"] + counts["insertions"]
    error_rate = errors / len(ref) if ref else float(errors > 0)
    return {
        "ref_len": len(ref),
        "hyp_len": len(hyp),
        "ops": counts,
        "error_rate": error_rate,
        "error_percent": 100.0 * error_rate,
    }


_LEGACY.metric_block = metric_block
evaluate_texts = _LEGACY.evaluate_texts
