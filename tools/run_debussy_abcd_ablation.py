from __future__ import annotations

import argparse
import json
import random
import xml.etree.ElementTree as ET
from collections import Counter
from fractions import Fraction
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "ijcv_samples" / "debussy-omr-transfer24"
STAFF_FULL = ROOT / "outputs" / "ijcv_repro" / "debussy_staff_transfer24"
STAFF_SMOKE = ROOT / "outputs" / "ijcv_repro" / "debussy_staff_smoke"
DEFAULT_OUT = ROOT / "outputs" / "debussy_abcd_ablation"


def load_archived_ablation_module() -> Any:
    try:
        from . import run_abcd_module_ablation
    except ImportError:
        import run_abcd_module_ablation
    return run_abcd_module_ablation


def discover_pages(module: Any, start_index: int, limit: int | None) -> list[Any]:
    pages = []
    for index in range(24):
        sample = f"test_{index:04d}"
        page_dir = (STAFF_SMOKE if index == 0 else STAFF_FULL) / sample
        gt = DATA / f"{sample}.musicxml"
        symbols = page_dir / "symbols" / "symbols_v2.json"
        if not symbols.exists() or not gt.exists():
            raise FileNotFoundError(
                f"Missing frozen Debussy artifact for {sample}: symbols={symbols.exists()}, gt={gt.exists()}"
            )
        pages.append(module.Page(sample=sample, page_dir=page_dir, gt=gt))
    pages = pages[start_index:]
    return pages[:limit] if limit is not None else pages


def note_events(path: Path) -> list[tuple[str, Fraction]]:
    """Return the exact duration-pitch event representation used by the main Debussy table."""
    root = ET.parse(path).getroot()
    events: list[tuple[str, Fraction]] = []
    for part in root.findall(".//part"):
        divisions = 1
        for measure in part.findall("measure"):
            value = measure.findtext("attributes/divisions")
            if value:
                divisions = max(1, int(value))
            for note in measure.findall("note"):
                pitch = note.find("pitch")
                if pitch is None:
                    pitch_token = "R"
                else:
                    step = pitch.findtext("step", "C")
                    alter = pitch.findtext("alter", "0")
                    octave = pitch.findtext("octave", "4")
                    pitch_token = f"{step}{alter}@{octave}"
                duration = int(note.findtext("duration", "0") or 0)
                events.append((pitch_token, Fraction(duration, divisions)))
    return events


def counts(gt: Path, pred: Path) -> dict[str, int]:
    gold = Counter(note_events(gt))
    hypothesis = Counter(note_events(pred))
    return {
        "matches": sum((gold & hypothesis).values()),
        "predicted": sum(hypothesis.values()),
        "gold": sum(gold.values()),
    }


def scores(rows: list[dict[str, int]]) -> dict[str, float | int]:
    matches = sum(row["matches"] for row in rows)
    predicted = sum(row["predicted"] for row in rows)
    gold = sum(row["gold"] for row in rows)
    precision = 100.0 * matches / predicted if predicted else 0.0
    recall = 100.0 * matches / gold if gold else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "matches": matches,
        "predicted_events": predicted,
        "gold_events": gold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "output_to_gold_ratio": 100.0 * predicted / gold if gold else 0.0,
    }


def bootstrap_delta(
    full: list[dict[str, int]], ablated: list[dict[str, int]], samples: int, seed: int
) -> dict[str, float | int]:
    if len(full) != len(ablated):
        raise ValueError("Paired bootstrap requires equal page counts")
    observed = float(scores(full)["f1"]) - float(scores(ablated)["f1"])
    rng = random.Random(seed)
    deltas = []
    for _ in range(samples):
        indices = [rng.randrange(len(full)) for _ in full]
        deltas.append(
            float(scores([full[i] for i in indices])["f1"])
            - float(scores([ablated[i] for i in indices])["f1"])
        )
    deltas.sort()
    low = deltas[int(0.025 * samples)]
    high = deltas[min(samples - 1, int(0.975 * samples))]
    return {
        "full_minus_ablated": observed,
        "ci95_low": low,
        "ci95_high": high,
        "bootstrap_samples": samples,
    }


def render_figures(payload: dict[str, Any], out_dir: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image, ImageDraw, ImageOps

    figures = out_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    variants = payload["variants"]

    ordered = sorted(variants, key=lambda name: variants[name]["summary"]["f1"], reverse=True)
    values = [variants[name]["summary"]["f1"] for name in ordered]
    colors = ["#c83e4d" if name == "A1B1C1D1" else "#4c78a8" for name in ordered]
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.bar(ordered, values, color=colors)
    ax.set_ylabel("Order-invariant event F1 (%)")
    ax.set_xlabel("A/B/C/D variant")
    ax.set_title("Debussy zero-shot A/B/C/D ablation")
    ax.tick_params(axis="x", rotation=55)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    overview = figures / "debussy_abcd_event_f1.png"
    fig.savefig(overview, dpi=220)
    plt.close(fig)

    loo = payload["leave_one_out"]
    labels = ["w/o A: staff prior", "w/o B: DEIM", "w/o C: SAM2", "w/o D: relation graph"]
    keys = ["A", "B", "C", "D"]
    effects = [loo[key]["bootstrap"]["full_minus_ablated"] for key in keys]
    lows = [effect - loo[key]["bootstrap"]["ci95_low"] for effect, key in zip(effects, keys)]
    highs = [loo[key]["bootstrap"]["ci95_high"] - effect for effect, key in zip(effects, keys)]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    bar_colors = ["#2ca02c" if effect >= 0 else "#d62728" for effect in effects]
    ax.barh(labels, effects, xerr=[lows, highs], capsize=4, color=bar_colors, alpha=0.85)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("Full minus ablated event F1 (percentage points)")
    ax.set_title("Leave-one-module-out contribution on Debussy")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    contribution = figures / "debussy_leave_one_out_contribution.png"
    fig.savefig(contribution, dpi=220)
    plt.close(fig)

    def page_f1(row: dict[str, int]) -> float:
        denominator = row["predicted"] + row["gold"]
        return 200.0 * row["matches"] / denominator if denominator else 0.0

    full_pages = variants["A1B1C1D1"]["per_page_counts"]
    no_d_pages = variants["A1B1C1D0"]["per_page_counts"]
    page_ids = [row["page_id"].replace("test_", "") for row in full_pages]
    full_f1 = [page_f1(row) for row in full_pages]
    no_d_f1 = [page_f1(row) for row in no_d_pages]
    x = list(range(len(page_ids)))
    fig, ax = plt.subplots(figsize=(12, 5.2))
    ax.plot(x, full_f1, marker="o", linewidth=1.8, color="#c83e4d", label="Full (with D)")
    ax.plot(x, no_d_f1, marker="s", linewidth=1.5, color="#4c78a8", label="w/o D")
    ax.fill_between(x, no_d_f1, full_f1, where=[a >= b for a, b in zip(full_f1, no_d_f1)], color="#59a14f", alpha=0.18)
    ax.set_xticks(x)
    ax.set_xticklabels(page_ids, rotation=55)
    ax.set_xlabel("Debussy transfer24 system index")
    ax.set_ylabel("Per-system event F1 (%)")
    ax.set_title("Relation graph contribution across Debussy systems")
    ax.grid(alpha=0.22)
    ax.legend()
    fig.tight_layout()
    per_page_d = figures / "debussy_relation_graph_per_page.png"
    fig.savefig(per_page_d, dpi=220)
    plt.close(fig)

    sample_root = STAFF_SMOKE / "test_0000"
    paths = [
        DATA / "test_0000.png",
        sample_root / "deim" / "overlay_deim_with_clef_roi.png",
        sample_root / "symbols" / "overlay_v2_with_sam2.png",
        sample_root / "visuals" / "v2_1_relation_overlay.png",
        sample_root / "visuals" / "v2_reconstructed_from_symbols.png",
    ]
    titles = ["Input", "DEIM detections", "SAM2 masks", "Relation graph", "Reconstruction"]
    thumbs = []
    target = (720, 260)
    for path, title in zip(paths, titles):
        image = Image.open(path).convert("RGB")
        image.thumbnail(target)
        canvas = Image.new("RGB", (target[0], target[1] + 38), "white")
        canvas.paste(image, ((target[0] - image.width) // 2, 36 + (target[1] - image.height) // 2))
        ImageDraw.Draw(canvas).text((12, 10), title, fill="black")
        thumbs.append(canvas)
    panel = Image.new("RGB", (target[0], len(thumbs) * (target[1] + 38)), "white")
    y = 0
    for thumb in thumbs:
        panel.paste(ImageOps.expand(thumb, border=1, fill="#b0b0b0"), (0, y))
        y += target[1] + 38
    qualitative = figures / "debussy_pipeline_qualitative_test0000.png"
    panel.save(qualitative)
    return [str(overview), str(contribution), str(per_page_d), str(qualitative)]


def write_markdown(payload: dict[str, Any], path: Path) -> None:
    lines = [
        "# Debussy zero-shot A/B/C/D ablation",
        "",
        "A=rule/staff prior, B=DEIM, C=SAM2, D=relation graph. All variants use the frozen Debussy transfer24 set and the same balanced pruning/export policy.",
        "",
        "| Variant | Precision | Recall | Event F1 | Pred/GT |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name in sorted(payload["variants"]):
        row = payload["variants"][name]["summary"]
        lines.append(
            f"| {name} | {row['precision']:.3f} | {row['recall']:.3f} | {row['f1']:.3f} | {row['output_to_gold_ratio']:.1f}% |"
        )
    lines.extend(
        [
            "",
            "## Leave-one-out effect",
            "",
            "| Removed module | Full - ablated F1 | 95% paired bootstrap CI |",
            "| --- | ---: | ---: |",
        ]
    )
    for key, row in payload["leave_one_out"].items():
        boot = row["bootstrap"]
        lines.append(
            f"| {key} | {boot['full_minus_ablated']:+.3f} | [{boot['ci95_low']:+.3f}, {boot['ci95_high']:+.3f}] |"
        )
    lines.extend(["", "Metric caveat: event F1 measures unordered pitch-duration content recovery, not ordered SER or TEDn."])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the frozen 2^4 STAFF ablation on Debussy transfer24.")
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--skip-run", action="store_true")
    parser.add_argument("--run-only", action="store_true")
    args = parser.parse_args()

    out_dir = args.out_root.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    module = load_archived_ablation_module()
    pages = discover_pages(module, args.start_index, args.limit)

    if not args.skip_run:
        progress = []
        for page in pages:
            shape_cache: dict[tuple[bool, bool, bool], dict[str, Any]] = {}
            for a in (False, True):
                for b in (False, True):
                    for c in (False, True):
                        for d in (False, True):
                            result = module.build_variant(
                                page, out_dir, a, b, c, d, "debussy", shape_cache
                            )
                            progress.append(result)
                            progress_name = f"progress_{args.start_index:04d}_{pages[-1].sample}.json"
                            (out_dir / progress_name).write_text(
                                json.dumps(progress, indent=2, ensure_ascii=False), encoding="utf-8"
                            )

    if args.run_only:
        print(json.dumps({"completed_pages": [page.sample for page in pages]}, indent=2))
        return

    variants: dict[str, Any] = {}
    for a in (0, 1):
        for b in (0, 1):
            for c in (0, 1):
                for d in (0, 1):
                    name = f"A{a}B{b}C{c}D{d}"
                    per_page = []
                    for page in pages:
                        pred = out_dir / name / page.sample / "semantics" / "score.musicxml"
                        if not pred.exists():
                            raise FileNotFoundError(f"Missing ablation prediction: {pred}")
                        per_page.append({"page_id": page.sample, **counts(page.gt, pred)})
                    variants[name] = {"summary": scores(per_page), "per_page_counts": per_page}

    full_name = "A1B1C1D1"
    ablated_names = {"A": "A0B1C1D1", "B": "A1B0C1D1", "C": "A1B1C0D1", "D": "A1B1C1D0"}
    leave_one_out = {}
    for offset, (module_name, variant_name) in enumerate(ablated_names.items()):
        leave_one_out[module_name] = {
            "variant": variant_name,
            "bootstrap": bootstrap_delta(
                variants[full_name]["per_page_counts"],
                variants[variant_name]["per_page_counts"],
                args.bootstrap_samples,
                args.seed + offset,
            ),
        }

    payload = {
        "dataset": "HugoSchtr/debussy-omr-system-lvl transfer24",
        "pages": len(pages),
        "protocol": "frozen zero-shot A/B/C/D; no Debussy adaptation or test-set parameter selection",
        "metric": "order_invariant_duration_pitch_event_f1",
        "module_definitions": {
            "A": "rule and staff prior",
            "B": "DEIM detections",
            "C": "SAM2 masks",
            "D": "relation graph",
        },
        "variants": variants,
        "leave_one_out": leave_one_out,
        "seed": args.seed,
        "caveat": "Event-F1 measures content recovery, not ordered SER or TEDn.",
    }
    figures = render_figures(payload, out_dir)
    payload["figures"] = figures
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(payload, out_dir / "summary.md")
    print(json.dumps({"summary": str(summary_path), "figures": figures}, indent=2))


if __name__ == "__main__":
    main()
