from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


BOS = "<BOS>"
EOS = "<EOS>"
SEP = "\u001f"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def context_key(prev: str, token: str, nxt: str) -> str:
    return SEP.join((prev, token, nxt))


def boundary_key(left: str, right: str) -> str:
    return SEP.join((left, right))


def collect_counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    token_opportunities: Counter[str] = Counter()
    context_opportunities: Counter[str] = Counter()
    boundary_opportunities: Counter[str] = Counter()
    substitutions: dict[str, Counter[str]] = defaultdict(Counter)
    substitutions_context: dict[str, Counter[str]] = defaultdict(Counter)
    deletions: Counter[str] = Counter()
    deletions_context: Counter[str] = Counter()
    insertions: dict[str, Counter[str]] = defaultdict(Counter)

    for row in rows:
        src = str(row["src"]).split()
        tgt = str(row["tgt"]).split()
        padded = [BOS, *src, EOS]
        for i, token in enumerate(src, start=1):
            token_opportunities[token] += 1
            context_opportunities[context_key(padded[i - 1], token, padded[i + 1])] += 1
        for i in range(len(padded) - 1):
            boundary_opportunities[boundary_key(padded[i], padded[i + 1])] += 1

        matcher = SequenceMatcher(a=src, b=tgt, autojunk=False)
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                continue
            src_chunk = src[i1:i2]
            tgt_chunk = tgt[j1:j2]
            paired = min(len(src_chunk), len(tgt_chunk))
            for offset in range(paired):
                source_token = src_chunk[offset]
                target_token = tgt_chunk[offset]
                if source_token == target_token:
                    continue
                absolute = i1 + offset
                prev = src[absolute - 1] if absolute > 0 else BOS
                nxt = src[absolute + 1] if absolute + 1 < len(src) else EOS
                substitutions[source_token][target_token] += 1
                substitutions_context[context_key(prev, source_token, nxt)][target_token] += 1
            for offset in range(paired, len(src_chunk)):
                absolute = i1 + offset
                token = src[absolute]
                prev = src[absolute - 1] if absolute > 0 else BOS
                nxt = src[absolute + 1] if absolute + 1 < len(src) else EOS
                deletions[token] += 1
                deletions_context[context_key(prev, token, nxt)] += 1
            remaining = tgt_chunk[paired:]
            if remaining:
                left = src[i2 - 1] if i2 > 0 else BOS
                right = src[i2] if i2 < len(src) else EOS
                insertions[boundary_key(left, right)][" ".join(remaining)] += 1

    return {
        "token_opportunities": token_opportunities,
        "context_opportunities": context_opportunities,
        "boundary_opportunities": boundary_opportunities,
        "substitutions": substitutions,
        "substitutions_context": substitutions_context,
        "deletions": deletions,
        "deletions_context": deletions_context,
        "insertions": insertions,
    }


def best_rule(counts: Counter[str], opportunities: int, min_support: int, min_confidence: float) -> str | None:
    if not counts or opportunities <= 0:
        return None
    value, support = counts.most_common(1)[0]
    if support < min_support or support / opportunities < min_confidence:
        return None
    return value


def compile_model(counts: dict[str, Any], min_support: int, min_confidence: float) -> dict[str, Any]:
    substitutions = {}
    for token, choices in counts["substitutions"].items():
        rule = best_rule(choices, counts["token_opportunities"][token], min_support, min_confidence)
        if rule is not None:
            substitutions[token] = rule
    substitutions_context = {}
    for key, choices in counts["substitutions_context"].items():
        rule = best_rule(choices, counts["context_opportunities"][key], min_support, min_confidence)
        if rule is not None:
            substitutions_context[key] = rule
    deletions = {
        token
        for token, support in counts["deletions"].items()
        if support >= min_support and support / counts["token_opportunities"][token] >= min_confidence
    }
    deletions_context = {
        key
        for key, support in counts["deletions_context"].items()
        if support >= min_support and support / counts["context_opportunities"][key] >= min_confidence
    }
    insertions = {}
    for key, choices in counts["insertions"].items():
        rule = best_rule(choices, counts["boundary_opportunities"][key], min_support, min_confidence)
        if rule is not None:
            insertions[key] = rule.split()
    return {
        "min_support": min_support,
        "min_confidence": min_confidence,
        "substitutions": substitutions,
        "substitutions_context": substitutions_context,
        "deletions": sorted(deletions),
        "deletions_context": sorted(deletions_context),
        "insertions": insertions,
    }


def apply_model(text: str, model: dict[str, Any]) -> str:
    tokens = text.split()
    padded = [BOS, *tokens, EOS]
    delete_global = set(model.get("deletions", []))
    delete_context = set(model.get("deletions_context", []))
    output: list[str] = []
    for i, token in enumerate(tokens, start=1):
        prev, nxt = padded[i - 1], padded[i + 1]
        insertion = model.get("insertions", {}).get(boundary_key(prev, token), [])
        output.extend(insertion)
        key = context_key(prev, token, nxt)
        if key in delete_context or token in delete_global:
            continue
        replacement = model.get("substitutions_context", {}).get(key)
        if replacement is None:
            replacement = model.get("substitutions", {}).get(token, token)
        output.append(replacement)
    output.extend(model.get("insertions", {}).get(boundary_key(tokens[-1] if tokens else BOS, EOS), []))
    return " ".join(output)


def load_ser_metric(olimpic_root: Path):
    sys.path.insert(0, str((olimpic_root / "zeus").resolve()))
    from ser_metric import ser_metric

    return ser_metric


def evaluate(rows: list[dict[str, Any]], model: dict[str, Any], metric) -> dict[str, float]:
    gold = [str(row["tgt"]) for row in rows]
    pred = [apply_model(str(row["src"]), model) for row in rows]
    return metric(gold, pred)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a conservative high-confidence LMX edit corrector and apply it without target-test labels.")
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--out-model", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-lmx", type=Path, required=True)
    parser.add_argument("--selection-json", type=Path, required=True)
    parser.add_argument("--olimpic-root", type=Path, default=Path(".local-tools/olimpic-icdar24"))
    parser.add_argument("--dev-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = read_jsonl(args.train_jsonl)
    if len(rows) < 5:
        raise ValueError(f"Need at least 5 training rows, got {len(rows)}")
    shuffled = list(rows)
    random.Random(args.seed).shuffle(shuffled)
    dev_n = max(1, int(round(len(shuffled) * args.dev_ratio)))
    dev_rows, fit_rows = shuffled[:dev_n], shuffled[dev_n:]
    metric = load_ser_metric(args.olimpic_root)
    fit_counts = collect_counts(fit_rows)
    candidates = [{"name": "identity", "model": compile_model(fit_counts, 10**9, 1.0)}]
    for support in (2, 5, 10, 20):
        for confidence in (0.70, 0.80, 0.90, 0.97):
            candidates.append({"name": f"s{support}_c{int(confidence*100)}", "model": compile_model(fit_counts, support, confidence)})
    scored = []
    for candidate in candidates:
        metrics = evaluate(dev_rows, candidate["model"], metric)
        scored.append({"name": candidate["name"], "SER": metrics["SER"], "SERnotuplets": metrics.get("SERnotuplets"), "model": candidate["model"]})
    best = min(scored, key=lambda row: float(row["SER"]))
    selected_support = int(best["model"]["min_support"])
    selected_confidence = float(best["model"]["min_confidence"])
    final_model = compile_model(collect_counts(rows), selected_support, selected_confidence)
    final_model.update({"selection": best["name"], "train_rows": len(rows), "fit_rows": len(fit_rows), "dev_rows": len(dev_rows)})
    write_json(args.out_model, final_model)
    write_json(
        args.selection_json,
        {
            "protocol": "Hyperparameters selected on a deterministic training-domain holdout; target test labels are not read.",
            "best": {key: value for key, value in best.items() if key != "model"},
            "candidates": [{key: value for key, value in row.items() if key != "model"} for row in scored],
        },
    )

    inputs = read_jsonl(args.input_jsonl)
    pred_rows = [{"sample": str(row["sample"]), "prediction": apply_model(str(row["src"]), final_model)} for row in inputs]
    write_json(args.out_json, {"rows": pred_rows, "model": str(args.out_model), "target_ground_truth_read": False})
    args.out_lmx.parent.mkdir(parents=True, exist_ok=True)
    args.out_lmx.write_text("\n".join(row["prediction"] for row in pred_rows) + "\n", encoding="utf-8")
    print(json.dumps({"train_rows": len(rows), "inputs": len(inputs), "selection": best["name"], "dev_SER": best["SER"], "out_json": str(args.out_json)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
