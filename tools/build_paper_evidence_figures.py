from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parent.parent
METHOD_COLORS = {"staff": "#0072B2", "smt": "#D55E00", "zeus": "#009E73"}


def select_ranked_pages(
    rows: Sequence[Mapping[str, Any]], value_key: str
) -> dict[str, Mapping[str, Any]]:
    if not rows:
        raise ValueError("Cannot select representatives from empty rows")
    ordered = sorted(rows, key=lambda row: (float(row[value_key]), str(row["page_id"])))
    return {"worst": ordered[0], "median": ordered[len(ordered) // 2], "best": ordered[-1]}


def _page_f1(counts: Mapping[str, int]) -> float:
    p = counts["matches"] / counts["predicted"] if counts["predicted"] else 0.0
    r = counts["matches"] / counts["gold"] if counts["gold"] else 0.0
    return 100.0 * 2 * p * r / (p + r) if p + r else 0.0


def _save(fig: plt.Figure, out: Path, stem: str) -> list[str]:
    paths = []
    for suffix in ("png", "pdf"):
        path = out / f"{stem}.{suffix}"
        fig.savefig(path, dpi=300 if suffix == "png" else None, bbox_inches="tight")
        paths.append(str(path))
    plt.close(fig)
    return paths


def _scatter_figure(out: Path) -> list[str]:
    sources = [
        ROOT / "outputs/ijcv_repro/order_invariant_event_f1/polish_test24.json",
        ROOT / "outputs/ijcv_repro/order_invariant_event_f1/debussy_transfer24.json",
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10, 9), constrained_layout=True)
    for row_index, source in enumerate(sources):
        payload = json.loads(source.read_text(encoding="utf-8"))
        for col_index, baseline in enumerate(("smt", "zeus")):
            ax = axes[row_index, col_index]
            staff = [_page_f1(row["staff"]) for row in payload["per_page_counts"]]
            other = [_page_f1(row[baseline]) for row in payload["per_page_counts"]]
            upper = max([1.0, *staff, *other]) * 1.05
            ax.scatter(other, staff, color=METHOD_COLORS[baseline], alpha=0.8, edgecolor="white")
            ax.plot([0, upper], [0, upper], linestyle="--", color="#777777", linewidth=1)
            wins = sum(a > b for a, b in zip(staff, other))
            ties = sum(abs(a - b) < 1e-12 for a, b in zip(staff, other))
            losses = len(staff) - wins - ties
            ax.set(xlim=(0, upper), ylim=(0, upper), xlabel=f"{baseline.upper()} page F1", ylabel="STAFF page F1")
            ax.set_title(f"{payload['dataset']}\nSTAFF W/T/L = {wins}/{ties}/{losses}")
            ax.grid(alpha=0.2)
    return _save(fig, out, "paired_page_event_f1")


def _decomposition_figure(out: Path) -> list[str]:
    source = ROOT / "outputs/paper_evidence_20260720/error_decomposition/metrics.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    methods = ("staff", "smt", "zeus")
    projections = ("pitch", "duration", "joint")
    labels = ("Pitch only", "Duration only", "Joint")
    x = np.arange(len(projections))
    fig, ax = plt.subplots(figsize=(8.5, 5.2), constrained_layout=True)
    width = 0.24
    for index, method in enumerate(methods):
        values = [payload["summary"][method][projection]["f1"] for projection in projections]
        bars = ax.bar(x + (index - 1) * width, values, width, label=method.upper(), color=METHOD_COLORS[method])
        ax.bar_label(bars, fmt="%.1f", fontsize=8, padding=2)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Event F1 (%) ↑")
    ax.set_title("Debussy zero-shot error decomposition")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    return _save(fig, out, "debussy_pitch_duration_joint")


def _forest_figure(out: Path) -> list[str]:
    source = ROOT / "outputs/paper_evidence_20260720/relation_edge_ablation/summary.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    rows = []
    for variant, entry in payload["variants"].items():
        if variant == "full":
            continue
        effect = entry["effect"]
        rows.append((variant.removeprefix("without_"), effect["full_minus_ablated"], effect["ci95_low"], effect["ci95_high"]))
    fig, ax = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)
    y = np.arange(len(rows))
    means = np.array([row[1] for row in rows])
    low = np.array([row[2] for row in rows])
    high = np.array([row[3] for row in rows])
    ax.errorbar(means, y, xerr=[means - low, high - means], fmt="o", color="#0072B2", capsize=5)
    ax.axvline(0, color="#777777", linestyle="--", linewidth=1)
    ax.set_yticks(y, [row[0].replace("_", "–") for row in rows])
    ax.set_xlabel("Full − edge-ablated Event F1 (points) ↑")
    ax.set_title("Debussy relation edge-family contribution (95% paired bootstrap CI)")
    ax.grid(axis="x", alpha=0.2)
    return _save(fig, out, "debussy_relation_edge_forest")


def _density_figure(out: Path) -> list[str]:
    source = ROOT / "outputs/ijcv_repro/order_invariant_event_f1/debussy_transfer24.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    rows = sorted(payload["per_page_counts"], key=lambda row: (row["staff"]["gold"], row["page_id"]))
    bins = np.array_split(rows, 3)
    fig, ax = plt.subplots(figsize=(8.5, 5.0), constrained_layout=True)
    x = np.arange(3)
    width = 0.24
    for index, method in enumerate(("staff", "smt", "zeus")):
        values = []
        for group in bins:
            counts = {
                "matches": sum(row[method]["matches"] for row in group),
                "predicted": sum(row[method]["predicted"] for row in group),
                "gold": sum(row[method]["gold"] for row in group),
            }
            values.append(_page_f1(counts))
        ax.bar(x + (index - 1) * width, values, width, label=method.upper(), color=METHOD_COLORS[method])
    labels = [f"Q{i+1}\n{min(r['staff']['gold'] for r in group)}–{max(r['staff']['gold'] for r in group)} gold" for i, group in enumerate(bins)]
    ax.set_xticks(x, labels)
    ax.set_ylabel("Aggregated Event F1 (%) ↑")
    ax.set_title("Debussy performance by predeclared gold-event density tertile")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    return _save(fig, out, "debussy_density_stratification")


def _qualitative_figure(out: Path) -> tuple[list[str], dict[str, str]]:
    source = ROOT / "outputs/debussy_abcd_ablation/summary.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    full = {row["page_id"]: row for row in payload["variants"]["A1B1C1D1"]["per_page_counts"]}
    no_d = {row["page_id"]: row for row in payload["variants"]["A1B1C1D0"]["per_page_counts"]}
    deltas = [{"page_id": page_id, "delta": _page_f1(full[page_id]) - _page_f1(no_d[page_id])} for page_id in full]
    selected = select_ranked_pages(deltas, "delta")
    fig, axes = plt.subplots(3, 2, figsize=(13, 10), constrained_layout=True)
    rank_order = ("best", "median", "worst")
    selected_ids: dict[str, str] = {}
    for row_index, rank in enumerate(rank_order):
        item = selected[rank]
        page_id = str(item["page_id"])
        selected_ids[rank] = page_id
        input_path = ROOT / f"data/ijcv_samples/debussy-omr-transfer24/{page_id}.png"
        shapes_path = ROOT / f"outputs/debussy_abcd_ablation/A1B1C1D1/{page_id}/symbols/shapes.json"
        shapes = json.loads(shapes_path.read_text(encoding="utf-8"))
        overlay_path = Path(shapes["overlay"])
        if not overlay_path.is_absolute():
            overlay_path = ROOT / overlay_path
        for col, path in enumerate((input_path, overlay_path)):
            image = Image.open(path).convert("RGB")
            axes[row_index, col].imshow(image)
            axes[row_index, col].axis("off")
        axes[row_index, 0].set_title(f"{rank}: {page_id} input")
        axes[row_index, 1].set_title(
            f"relations | Full D {_page_f1(full[page_id]):.2f}, w/o D {_page_f1(no_d[page_id]):.2f}, Δ {float(item['delta']):+.2f}"
        )
    return _save(fig, out, "debussy_relation_best_median_worst"), selected_ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Build publication figures for the 2026-07-20 evidence package.")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Microsoft YaHei", "DejaVu Sans"], "axes.unicode_minus": False})
    manifest: dict[str, Any] = {"figures": []}
    jobs = [
        ("paired_page_event_f1", _scatter_figure, ["polish_test24.json", "debussy_transfer24.json"]),
        ("debussy_pitch_duration_joint", _decomposition_figure, ["error_decomposition/metrics.json"]),
        ("debussy_relation_edge_forest", _forest_figure, ["relation_edge_ablation/summary.json"]),
        ("debussy_density_stratification", _density_figure, ["debussy_transfer24.json"]),
    ]
    for name, builder, sources in jobs:
        paths = builder(out)
        manifest["figures"].append({"name": name, "paths": paths, "sources": sources})
    paths, selected = _qualitative_figure(out)
    manifest["figures"].append({"name": "debussy_relation_best_median_worst", "paths": paths, "sources": ["debussy_abcd_ablation/summary.json"], "selection": selected})
    (out / "figure_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
