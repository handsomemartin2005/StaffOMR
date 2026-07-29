from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from evaluate_v1_v2_metrics import error_rate_from_ops, levenshtein_ops


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


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def is_data_line(line: str) -> bool:
    stripped = line.strip()
    return bool(stripped) and not stripped.startswith(("!", "**", "*", "="))


def kern_pitch(step: str, octave: int | None, alter: int | None) -> str:
    step = (step or "C").upper()
    octave = 4 if octave is None else int(octave)
    if octave >= 4:
        pitch = step.lower() * (octave - 3)
    else:
        pitch = step.upper() * (4 - octave)
    accidental = {1: "#", -1: "-", 0: "n"}.get(alter, "")
    return f"{pitch}{accidental}"


def normalize_kern_pitch(raw: str) -> str | None:
    body = raw.replace("@", "")
    match = re.search(r"([A-Ga-g]+)([#n-]*)", body)
    if not match:
        return None
    return f"{match.group(1)}{match.group(2)}"


def parse_ekern_token(token: str) -> dict[str, str] | None:
    token = token.strip()
    if not token or token == ".":
        return None
    duration_match = re.match(r"^(\d+)", token)
    if not duration_match:
        return None
    duration = duration_match.group(1)
    body = token[duration_match.end() :]
    dots = "." * body.count(".")
    if "r" in body.lower():
        return {"event": f"R|{duration}{dots}", "pitch": "R", "duration": f"{duration}{dots}"}
    pitch = normalize_kern_pitch(body)
    if not pitch:
        return None
    return {"event": f"N|{duration}{dots}|{pitch}", "pitch": pitch, "duration": f"{duration}{dots}"}


def ekern_text_to_event_tokens(text: str) -> dict[str, list[str]]:
    events: list[str] = []
    pitches: list[str] = []
    durations: list[str] = []
    for line in text.splitlines():
        if not is_data_line(line):
            continue
        for cell in line.split("\t"):
            for raw_token in cell.split():
                parsed = parse_ekern_token(raw_token)
                if not parsed:
                    continue
                events.append(parsed["event"])
                pitches.append(parsed["pitch"])
                durations.append(parsed["duration"])
    return {"events": events, "pitches": pitches, "durations": durations}


def semantic_event_to_token(event: dict[str, Any]) -> dict[str, str] | None:
    duration = DURATION_TO_KERN.get(event.get("duration_hint"), "4")
    dots = "." * max(0, int(event.get("dot_count") or 0))
    if event.get("type") == "rest":
        return {"event": f"R|{duration}{dots}", "pitch": "R", "duration": f"{duration}{dots}"}
    if event.get("type") != "note":
        return None
    pitch = event.get("pitch") or {}
    pitch_token = kern_pitch(str(pitch.get("step") or "C"), pitch.get("octave"), pitch.get("alter"))
    return {"event": f"N|{duration}{dots}|{pitch_token}", "pitch": pitch_token, "duration": f"{duration}{dots}"}


def semantics_to_event_tokens(semantics: dict[str, Any]) -> dict[str, list[str]]:
    events: list[str] = []
    pitches: list[str] = []
    durations: list[str] = []
    for part in semantics.get("parts", []):
        for measure in part.get("measures", []):
            for event in measure.get("events", []):
                parsed = semantic_event_to_token(event)
                if not parsed:
                    continue
                events.append(parsed["event"])
                pitches.append(parsed["pitch"])
                durations.append(parsed["duration"])
    return {"events": events, "pitches": pitches, "durations": durations}


def metric_block(ref: list[str], hyp: list[str]) -> dict[str, Any]:
    ops = levenshtein_ops(ref, hyp)
    err = error_rate_from_ops(ops, len(ref))
    return {
        "ref_len": len(ref),
        "hyp_len": len(hyp),
        "ops": ops,
        "error_rate": err,
        "error_percent": 100.0 * err,
    }


def evaluate(gt: dict[str, list[str]], pred: dict[str, list[str]]) -> dict[str, Any]:
    return {
        "event_error": metric_block(gt["events"], pred["events"]),
        "pitch_error": metric_block(gt["pitches"], pred["pitches"]),
        "duration_error": metric_block(gt["durations"], pred["durations"]),
        "counts": {
            "gt_events": len(gt["events"]),
            "pred_events": len(pred["events"]),
            "gt_pitches": len(gt["pitches"]),
            "pred_pitches": len(pred["pitches"]),
            "gt_durations": len(gt["durations"]),
            "pred_durations": len(pred["durations"]),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Align ekern GT into our event-token metric space and evaluate V2.1 semantics."
    )
    parser.add_argument("--gt-ekern", type=Path, required=True)
    parser.add_argument("--pred-semantics-json", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-gt-tokens", type=Path)
    parser.add_argument("--out-pred-tokens", type=Path)
    args = parser.parse_args()

    gt_text = args.gt_ekern.read_text(encoding="utf-8")
    pred_semantics = read_json(args.pred_semantics_json)
    gt_tokens = ekern_text_to_event_tokens(gt_text)
    pred_tokens = semantics_to_event_tokens(pred_semantics)
    result = {
        "gt_ekern": str(args.gt_ekern),
        "pred_semantics_json": str(args.pred_semantics_json),
        "definitions": {
            "event_error": "Levenshtein error over our normalized note/rest event tokens: type + duration + pitch.",
            "pitch_error": "Levenshtein error over pitch/rest tokens only.",
            "duration_error": "Levenshtein error over duration tokens only.",
            "scope": "This aligns the IJCV dataset GT into our metric space. It does not convert the IJCV paper's model results into our metric.",
        },
        **evaluate(gt_tokens, pred_tokens),
    }
    write_json(args.out_json, result)
    if args.out_gt_tokens:
        write_json(args.out_gt_tokens, gt_tokens)
    if args.out_pred_tokens:
        write_json(args.out_pred_tokens, pred_tokens)
    print(
        json.dumps(
            {
                "out_json": str(args.out_json),
                "event_error_percent": result["event_error"]["error_percent"],
                "pitch_error_percent": result["pitch_error"]["error_percent"],
                "duration_error_percent": result["duration_error"]["error_percent"],
                "counts": result["counts"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
