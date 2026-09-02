from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Sequence

from rapidfuzz.distance import Levenshtein
from evaluate_v1_v2_metrics import error_rate_from_ops, levenshtein_ops

try:
    from .polish_canonical_serialization import semantic_to_canonical_ekern
except ImportError:
    from polish_canonical_serialization import semantic_to_canonical_ekern


DURATION_TO_KERN = {
    "whole": "1",
    "half": "2",
    "quarter": "4",
    "eighth": "8",
    "16th": "16",
    "32nd": "32",
    "64th": "64",
    "eighth_or_shorter": "8",
    "eighth_or_quarter": "8",
    "notehead_only_unknown": "4",
    "unknown": "4",
    None: "4",
}
STEP_ORDER = {"C": 0, "D": 1, "E": 2, "F": 3, "G": 4, "A": 5, "B": 6}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def kern_pitch(step: str, octave: int | None, alter: int | None) -> str:
    step = (step or "C").upper()
    octave = 4 if octave is None else int(octave)
    if octave >= 4:
        pitch = step.lower() * (octave - 3)
    else:
        pitch = step.upper() * (4 - octave)
    accidental = {1: "#", -1: "-", 0: "n"}.get(alter, "")
    return f"{pitch}{accidental}"


def semantic_event_to_ekern(event: dict[str, Any]) -> str | None:
    duration = DURATION_TO_KERN.get(event.get("duration_hint"), "4")
    dots = "." * max(0, int(event.get("dot_count") or 0))
    if event.get("type") == "rest":
        return f"{duration}{dots}@r"
    if event.get("type") != "note":
        return None
    pitch = event.get("pitch") or {}
    token = kern_pitch(str(pitch.get("step") or "C"), pitch.get("octave"), pitch.get("alter"))
    return f"{duration}{dots}@{token}"


def legacy_semantic_to_pseudo_ekern(semantics: dict[str, Any]) -> str:
    lines: list[str] = ["**ekern"]
    for part in semantics.get("parts", []):
        clef = str(part.get("clef_type") or "")
        if clef == "bass":
            lines.append("*clefF4")
        elif clef:
            lines.append("*clefG2")
        for measure in part.get("measures", []):
            lines.append(f"={measure.get('number')}")
            tokens = [
                token
                for event in measure.get("events", [])
                for token in [semantic_event_to_ekern(event)]
                if token
            ]
            if tokens:
                lines.append(" ".join(tokens))
    lines.append("*-")
    return "\n".join(lines) + "\n"


def semantic_to_pseudo_ekern(semantics: dict[str, Any]) -> str:
    """Use canonical piano reading order only for Polish page semantics."""

    dataset = str(semantics.get("dataset") or "").lower()
    input_hint = str(semantics.get("input") or "").replace("\\", "/").lower()
    if dataset == "polish" or "polish" in input_hint:
        return semantic_to_canonical_ekern(semantics)
    return legacy_semantic_to_pseudo_ekern(semantics)


def semantic_json_to_text(path: Path) -> str:
    return semantic_to_pseudo_ekern(read_json(path))


def data_lines(text: str) -> list[str]:
    result: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("!") or stripped.startswith("**") or stripped.startswith("*") or stripped.startswith("="):
            continue
        result.append(stripped)
    return result


def split_raw_tokens(text: str) -> list[str]:
    tokens: list[str] = []
    for line in data_lines(text):
        for cell in re.split(r"[\t ]+", line):
            cell = cell.strip()
            if not cell or cell == ".":
                continue
            tokens.append(cell)
    return tokens


def normalize_token(token: str) -> str | None:
    token = token.strip()
    if not token or token == ".":
        return None
    token = token.replace("@.", ".@")
    duration_match = re.match(r"^(\d+)(\.*)", token)
    duration = duration_match.group(1) if duration_match else "?"
    dots = duration_match.group(2) if duration_match else ""
    if "r" in token:
        return f"{duration}{dots}:r"
    pitch_match = re.search(r"([A-Ga-g]+)([#n-]*)", token)
    if not pitch_match:
        return None
    pitch = pitch_match.group(1)
    accidental = pitch_match.group(2)
    return f"{duration}{dots}:{pitch}{accidental}"


def normalized_tokens(text: str) -> list[str]:
    out: list[str] = []
    for token in split_raw_tokens(text):
        normalized = normalize_token(token)
        if normalized:
            out.append(normalized)
    return out


def chars_from_tokens(tokens: list[str]) -> list[str]:
    return list(" ".join(tokens))


def metric_block(ref: Sequence[Any], hyp: Sequence[Any]) -> dict[str, Any]:
    ops = {
        "substitutions": 0,
        "deletions": 0,
        "insertions": 0,
        "matches": 0,
    }
    for operation in Levenshtein.editops(ref, hyp):
        if operation.tag == "replace":
            ops["substitutions"] += 1
        elif operation.tag == "delete":
            ops["deletions"] += 1
        elif operation.tag == "insert":
            ops["insertions"] += 1
    ops["matches"] = len(ref) - ops["substitutions"] - ops["deletions"]
    errors = ops["substitutions"] + ops["deletions"] + ops["insertions"]
    error_rate = errors / len(ref) if ref else float(errors > 0)
    return {
        "ref_len": len(ref),
        "hyp_len": len(hyp),
        "ops": ops,
        "error_rate": error_rate,
        "error_percent": 100.0 * error_rate,
    }


def evaluate_texts(gt_text: str, pred_text: str) -> dict[str, Any]:
    gt_norm_tokens = normalized_tokens(gt_text)
    pred_norm_tokens = normalized_tokens(pred_text)
    gt_raw_tokens = split_raw_tokens(gt_text)
    pred_raw_tokens = split_raw_tokens(pred_text)
    return {
        "definitions": {
            "CER_proxy": "Levenshtein error over characters after lightweight ekern token normalization.",
            "SER_proxy": "Levenshtein error over normalized ekern-like tokens.",
            "LER_proxy": "Levenshtein error over non-header data lines.",
            "warning": "This is an IJCV-style proxy for current V2.1 outputs, not the official bekern tokenizer/evaluator.",
        },
        "normalized": {
            "CER_proxy": metric_block(chars_from_tokens(gt_norm_tokens), chars_from_tokens(pred_norm_tokens)),
            "SER_proxy": metric_block(gt_norm_tokens, pred_norm_tokens),
        },
        "raw": {
            "CER_proxy": metric_block(chars_from_tokens(gt_raw_tokens), chars_from_tokens(pred_raw_tokens)),
            "SER_proxy": metric_block(gt_raw_tokens, pred_raw_tokens),
            "LER_proxy": metric_block(data_lines(gt_text), data_lines(pred_text)),
        },
        "counts": {
            "gt_raw_tokens": len(gt_raw_tokens),
            "pred_raw_tokens": len(pred_raw_tokens),
            "gt_normalized_tokens": len(gt_norm_tokens),
            "pred_normalized_tokens": len(pred_norm_tokens),
            "gt_data_lines": len(data_lines(gt_text)),
            "pred_data_lines": len(data_lines(pred_text)),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute IJCV-style proxy CER/SER/LER for ekern text and V2.1 semantics.")
    parser.add_argument("--gt-ekern", type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pred-ekern", type=Path)
    group.add_argument("--pred-semantics-json", type=Path)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-pred-ekern", type=Path)
    args = parser.parse_args()

    gt_text = args.gt_ekern.read_text(encoding="utf-8")
    if args.pred_semantics_json:
        pred_text = semantic_json_to_text(args.pred_semantics_json)
    else:
        assert args.pred_ekern is not None
        pred_text = args.pred_ekern.read_text(encoding="utf-8")

    if args.out_pred_ekern:
        args.out_pred_ekern.parent.mkdir(parents=True, exist_ok=True)
        args.out_pred_ekern.write_text(pred_text, encoding="utf-8")

    result = {
        "gt_ekern": str(args.gt_ekern),
        "pred_ekern": str(args.pred_ekern) if args.pred_ekern else str(args.out_pred_ekern) if args.out_pred_ekern else None,
        "pred_semantics_json": str(args.pred_semantics_json) if args.pred_semantics_json else None,
        **evaluate_texts(gt_text, pred_text),
    }
    write_json(args.out_json, result)
    summary = {
        "out_json": str(args.out_json),
        "normalized_CER_percent": result["normalized"]["CER_proxy"]["error_percent"],
        "normalized_SER_percent": result["normalized"]["SER_proxy"]["error_percent"],
        "raw_LER_percent": result["raw"]["LER_proxy"]["error_percent"],
        "counts": result["counts"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
