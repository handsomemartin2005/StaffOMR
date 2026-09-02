from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from evaluate_ekern_our_event_metrics import ekern_text_to_event_tokens, evaluate, semantics_to_event_tokens
from evaluate_kern_text_proxy_metrics import evaluate_texts, semantic_to_pseudo_ekern
from export_v2_1_semantics import build_semantics
from prune_v2_1_overrecognition import PRESETS, prune_notes, prune_shapes, read_json, write_json


def action_key(action: dict[str, Any]) -> tuple[Any, ...]:
    return (
        action["source_suffix"],
        int(action["column_cap"]),
        float(action["column_xtol"]),
        float(action["no_stem_min_conf"]),
        float(action["open_no_stem_min_conf"]),
        float(action["isolated_unknown_min_conf"]),
        float(action["neighbor_xtol"]),
    )


def sampled_actions(source_suffixes: list[str], iterations: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    caps = [2, 3, 4, 5, 6, 8, 10, 12]
    column_xtols = [10.0, 12.0, 14.0, 18.0, 22.0]
    no_stem = [0.0, 0.35, 0.50, 0.60, 0.70, 0.80]
    open_no_stem = [0.0, 0.22, 0.32, 0.42, 0.55]
    isolated = [0.0, 0.35, 0.50, 0.60, 0.75]
    neighbor = [12.0, 14.0, 18.0, 22.0, 28.0]

    actions: list[dict[str, Any]] = []
    for source in source_suffixes:
        for preset in ("recall", "light", "balanced", "aggressive", "visual_strict"):
            base = dict(PRESETS[preset])
            actions.append({"source_suffix": source, "preset_seed": preset, **base})
    seen = {action_key(action) for action in actions}
    while len(actions) < iterations:
        source = rng.choice(source_suffixes)
        action = {
            "source_suffix": source,
            "preset_seed": "random",
            "column_cap": rng.choice(caps),
            "column_xtol": rng.choice(column_xtols),
            "no_stem_min_conf": rng.choice(no_stem),
            "open_no_stem_min_conf": rng.choice(open_no_stem),
            "isolated_unknown_min_conf": rng.choice(isolated),
            "neighbor_xtol": rng.choice(neighbor),
        }
        key = action_key(action)
        if key in seen:
            continue
        seen.add(key)
        actions.append(action)
    return actions


def metric_ops_percent(metric: dict[str, Any], key: str) -> float:
    ref_len = max(1, int(metric.get("ref_len") or 0))
    return 100.0 * float((metric.get("ops") or {}).get(key, 0)) / ref_len


def histogram(items: list[str]) -> Counter[str]:
    return Counter(str(item) for item in items)


def histogram_l1_percent(ref: list[str], hyp: list[str]) -> float:
    ref_hist = histogram(ref)
    hyp_hist = histogram(hyp)
    keys = set(ref_hist) | set(hyp_hist)
    diff = sum(abs(ref_hist.get(key, 0) - hyp_hist.get(key, 0)) for key in keys)
    return 100.0 * diff / max(1, len(ref))


def evaluate_sample(
    run_dir: Path,
    gt_ekern: Path,
    params: dict[str, float | int],
    full_eval: bool,
) -> dict[str, Any]:
    notes_payload = read_json(run_dir / "notes" / "notes_v2_1.json")
    shapes_payload = read_json(run_dir / "symbols" / "symbols_v2_1_shapes.json")
    pruned_notes, dropped_notes = prune_notes(notes_payload, params)
    pruned_shapes, dropped_symbols = prune_shapes(shapes_payload, pruned_notes)
    semantics = build_semantics(pruned_notes, pruned_shapes)

    gt_text = gt_ekern.read_text(encoding="utf-8")
    gt_tokens = ekern_text_to_event_tokens(gt_text)
    pred_tokens = semantics_to_event_tokens(semantics)
    gt_events = len(gt_tokens["events"])
    pred_events = len(pred_tokens["events"])
    if full_eval:
        event_metrics = evaluate(gt_tokens, pred_tokens)
        proxy_metrics = evaluate_texts(gt_text, semantic_to_pseudo_ekern(semantics))
        event = event_metrics["event_error"]
        counts = event_metrics["counts"]
        event_error_percent = event["error_percent"]
        pitch_error_percent = event_metrics["pitch_error"]["error_percent"]
        duration_error_percent = event_metrics["duration_error"]["error_percent"]
        cer_proxy_percent = proxy_metrics["normalized"]["CER_proxy"]["error_percent"]
        ser_proxy_percent = proxy_metrics["normalized"]["SER_proxy"]["error_percent"]
        under_percent = metric_ops_percent(event, "deletions")
        over_percent = metric_ops_percent(event, "insertions")
        substitution_percent = metric_ops_percent(event, "substitutions")
    else:
        counts = {"gt_events": gt_events, "pred_events": pred_events}
        under_percent = 100.0 * max(0, gt_events - pred_events) / max(1, gt_events)
        over_percent = 100.0 * max(0, pred_events - gt_events) / max(1, gt_events)
        duration_error_percent = histogram_l1_percent(gt_tokens["durations"], pred_tokens["durations"])
        pitch_error_percent = histogram_l1_percent(gt_tokens["pitches"], pred_tokens["pitches"])
        substitution_percent = 0.5 * (duration_error_percent + pitch_error_percent)
        event_error_percent = under_percent + over_percent + 0.5 * substitution_percent
        cer_proxy_percent = event_error_percent
        ser_proxy_percent = event_error_percent
    return {
        "run_dir": str(run_dir),
        "gt_ekern": str(gt_ekern),
        "gt_events": counts["gt_events"],
        "pred_events": counts["pred_events"],
        "event_error_percent": event_error_percent,
        "pitch_error_percent": pitch_error_percent,
        "duration_error_percent": duration_error_percent,
        "cer_proxy_percent": cer_proxy_percent,
        "ser_proxy_percent": ser_proxy_percent,
        "under_percent": under_percent,
        "over_percent": over_percent,
        "substitution_percent": substitution_percent,
        "count_gap_percent": 100.0 * abs(counts["pred_events"] - counts["gt_events"]) / max(1, counts["gt_events"]),
        "notes_before": len(notes_payload.get("notes", []) or []),
        "notes_after": len(pruned_notes.get("notes", []) or []),
        "notes_pruned": len(dropped_notes),
        "symbols_pruned": len(dropped_symbols),
        "drop_reasons": dict(Counter(str(item.get("drop_reason")) for item in dropped_notes)),
    }


def aggregate(samples: list[dict[str, Any]]) -> dict[str, Any]:
    total_gt = sum(int(item["gt_events"]) for item in samples)
    total_pred = sum(int(item["pred_events"]) for item in samples)
    total_pruned = sum(int(item["notes_pruned"]) for item in samples)
    weighted_keys = [
        "event_error_percent",
        "pitch_error_percent",
        "duration_error_percent",
        "cer_proxy_percent",
        "ser_proxy_percent",
        "under_percent",
        "over_percent",
        "substitution_percent",
        "count_gap_percent",
    ]
    result: dict[str, Any] = {
        "gt_events_total": total_gt,
        "pred_events_total": total_pred,
        "pred_gt_ratio": total_pred / max(1, total_gt),
        "notes_before_total": sum(int(item["notes_before"]) for item in samples),
        "notes_after_total": sum(int(item["notes_after"]) for item in samples),
        "notes_pruned_total": total_pruned,
        "notes_pruned_per_gt_percent": 100.0 * total_pruned / max(1, total_gt),
    }
    for key in weighted_keys:
        result[key] = sum(float(item[key]) * int(item["gt_events"]) for item in samples) / max(1, total_gt)
    return result


def cost_from_metrics(metrics: dict[str, Any], args: argparse.Namespace) -> float:
    return (
        args.event_weight * float(metrics["event_error_percent"])
        + args.duration_weight * float(metrics["duration_error_percent"])
        + args.cer_weight * float(metrics["cer_proxy_percent"])
        + args.under_weight * float(metrics["under_percent"])
        + args.over_weight * float(metrics["over_percent"])
        + args.balance_weight * float(metrics["count_gap_percent"])
        + args.prune_weight * float(metrics["notes_pruned_per_gt_percent"])
    )


def render_markdown(payload: dict[str, Any], limit: int) -> str:
    lines = [
        "# V2.1 Recall/Over-recognition Policy Search",
        "",
        "Reward is the negative weighted cost over event error, duration error, CER proxy, deletion pressure, insertion pressure, event-count gap, and pruning pressure.",
        "",
        "## Best Policy",
        "",
    ]
    best = payload["best"]
    metrics = best["metrics"]
    lines.extend(
        [
            f"- source_suffix: `{best['action']['source_suffix']}`",
            f"- params: `{json.dumps(best['params'], ensure_ascii=False)}`",
            f"- reward: `{best['reward']:.4f}`",
            f"- cost: `{best['cost']:.4f}`",
            f"- pred/GT events: `{metrics['pred_events_total']}/{metrics['gt_events_total']}` (`{metrics['pred_gt_ratio']:.3f}`)",
            f"- event error: `{metrics['event_error_percent']:.2f}%`",
            f"- under/over: `{metrics['under_percent']:.2f}% / {metrics['over_percent']:.2f}%`",
            f"- eKern CER proxy: `{metrics['cer_proxy_percent']:.2f}%`",
            f"- notes pruned: `{metrics['notes_pruned_total']}`",
            "",
            "## Top Full-Eval Policies",
            "",
            "| rank | source | cost | pred/GT | event err% | under% | over% | CER proxy% | pruned | params |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    top_items = payload.get("full_eval_results") or payload["results"]
    for idx, item in enumerate(top_items[:limit], start=1):
        m = item["metrics"]
        lines.append(
            "| "
            + " | ".join(
                [
                    str(idx),
                    item["action"]["source_suffix"],
                    f"{item['cost']:.2f}",
                    f"{m['pred_events_total']}/{m['gt_events_total']}",
                    f"{m['event_error_percent']:.2f}",
                    f"{m['under_percent']:.2f}",
                    f"{m['over_percent']:.2f}",
                    f"{m['cer_proxy_percent']:.2f}",
                    str(m["notes_pruned_total"]),
                    "`" + json.dumps(item["params"], ensure_ascii=False) + "`",
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune V2.1 recall/over-recognition balance with a lightweight bandit-style policy search.")
    parser.add_argument("--runs-root", type=Path, default=Path("outputs/ijcv_dwd_dataset_runs"))
    parser.add_argument("--gt-root", type=Path, default=Path("data/ijcv_samples/polish-scores"))
    parser.add_argument("--pages", nargs="+", default=[f"{idx:04d}" for idx in range(5)])
    parser.add_argument(
        "--source-suffixes",
        nargs="+",
        default=[
            "v2_1_sam2_neural_clef_roi_recall_notile",
            "v2_1_sam2_neural_clef_roi_recall_tile",
        ],
    )
    parser.add_argument("--iterations", type=int, default=96)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out-json", type=Path, default=Path("outputs/ijcv_dwd_dataset_runs/rl_balance_tuning/policy_search_results.json"))
    parser.add_argument("--out-md", type=Path, default=Path("outputs/ijcv_dwd_dataset_runs/rl_balance_tuning/policy_search_results.md"))
    parser.add_argument("--event-weight", type=float, default=0.45)
    parser.add_argument("--duration-weight", type=float, default=0.12)
    parser.add_argument("--cer-weight", type=float, default=0.20)
    parser.add_argument("--under-weight", type=float, default=0.10)
    parser.add_argument("--over-weight", type=float, default=0.18)
    parser.add_argument("--balance-weight", type=float, default=0.18)
    parser.add_argument("--prune-weight", type=float, default=0.04)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--full-eval-top-k", type=int, default=8)
    args = parser.parse_args()

    actions = sampled_actions(args.source_suffixes, args.iterations, args.seed)
    results: list[dict[str, Any]] = []
    for index, action in enumerate(actions, start=1):
        source = str(action["source_suffix"])
        params = {
            "column_cap": int(action["column_cap"]),
            "column_xtol": float(action["column_xtol"]),
            "no_stem_min_conf": float(action["no_stem_min_conf"]),
            "open_no_stem_min_conf": float(action["open_no_stem_min_conf"]),
            "isolated_unknown_min_conf": float(action["isolated_unknown_min_conf"]),
            "neighbor_xtol": float(action["neighbor_xtol"]),
        }
        sample_results: list[dict[str, Any]] = []
        missing = False
        for page in args.pages:
            run_dir = args.runs_root / f"polish_test{page}_{source}"
            gt_ekern = args.gt_root / f"test_{page}.ekern.txt"
            if not run_dir.exists() or not gt_ekern.exists():
                missing = True
                break
            sample_results.append(evaluate_sample(run_dir, gt_ekern, params, full_eval=False))
        if missing:
            continue
        metrics = aggregate(sample_results)
        cost = cost_from_metrics(metrics, args)
        results.append(
            {
                "index": index,
                "action": action,
                "params": params,
                "metrics": metrics,
                "cost": cost,
                "reward": -cost,
                "samples": sample_results,
            }
        )
        if index % 20 == 0:
            print(json.dumps({"screened": index, "current_best_fast_cost": min(item["cost"] for item in results)}, ensure_ascii=False), flush=True)

    if not results:
        raise SystemExit("No actions could be evaluated; check --source-suffixes and --pages.")
    results.sort(key=lambda item: item["cost"])
    full_results: list[dict[str, Any]] = []
    for item in results[: max(1, args.full_eval_top_k)]:
        sample_results = []
        action = item["action"]
        params = item["params"]
        for page in args.pages:
            run_dir = args.runs_root / f"polish_test{page}_{action['source_suffix']}"
            gt_ekern = args.gt_root / f"test_{page}.ekern.txt"
            sample_results.append(evaluate_sample(run_dir, gt_ekern, params, full_eval=True))
        metrics = aggregate(sample_results)
        cost = cost_from_metrics(metrics, args)
        full_results.append(
            {
                **item,
                "fast_metrics": item["metrics"],
                "fast_cost": item["cost"],
                "metrics": metrics,
                "cost": cost,
                "reward": -cost,
                "samples": sample_results,
            }
        )
    full_results.sort(key=lambda item: item["cost"])
    final_results = [*full_results, *results[max(1, args.full_eval_top_k) :]]
    payload = {
        "search": {
            "method": "two_stage_finite_action_bandit_policy_search",
            "iterations_requested": args.iterations,
            "actions_evaluated": len(results),
            "full_eval_top_k": args.full_eval_top_k,
            "seed": args.seed,
            "pages": args.pages,
            "source_suffixes": args.source_suffixes,
            "weights": {
                "event": args.event_weight,
                "duration": args.duration_weight,
                "cer": args.cer_weight,
                "under": args.under_weight,
                "over": args.over_weight,
                "balance": args.balance_weight,
                "prune": args.prune_weight,
            },
        },
        "best": full_results[0],
        "full_eval_results": full_results,
        "results": final_results,
    }
    write_json(args.out_json, payload)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(render_markdown(payload, args.top_k), encoding="utf-8")
    print(
        json.dumps(
            {
                "out_json": str(args.out_json),
                "out_md": str(args.out_md),
                "best_source": payload["best"]["action"]["source_suffix"],
                "best_params": payload["best"]["params"],
                "best_cost": payload["best"]["cost"],
                "best_metrics": payload["best"]["metrics"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
