from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from PIL import Image


ROOT = Path(__file__).resolve().parent.parent
METHODS = ("staff", "smt", "zeus")
METHOD_COLORS = {"staff": "#0072B2", "smt": "#D55E00", "zeus": "#009E73"}
METHOD_MARKERS = {"staff": "o", "smt": "s", "zeus": "D"}
METHOD_LABELS = {"staff": "RIGOR-OMR", "smt": "SMT", "zeus": "Zeus"}
JOINT_COLOR = "#4D4D4D"
PITCH_COLOR = "#0072B2"
DURATION_COLOR = "#E69F00"
QUALITATIVE_SAMPLE_ID = "test_0006"
QUALITATIVE_CROP = (965, 390, 1170, 515)
QUALITATIVE_NOTEHEAD_IDS = ("deim_000003", "deim_000013")
QUALITATIVE_STEM_IDS = ("synthetic_stem_00012", "synthetic_stem_00013")
QUALITATIVE_BEAM_ID = "deim_000069"
HEAD_COLOR = "#D55E00"
STEM_COLOR = "#7B3294"
BEAM_COLOR = "#0072B2"
STAFF_COLOR = "#009E73"


def select_ranked_pages(
    rows: Sequence[Mapping[str, Any]], value_key: str
) -> dict[str, Mapping[str, Any]]:
    """Select deterministic worst, median, and best representatives."""

    if not rows:
        raise ValueError("Cannot select representatives from empty rows")
    ordered = sorted(rows, key=lambda row: (float(row[value_key]), str(row["page_id"])))
    return {"worst": ordered[0], "median": ordered[len(ordered) // 2], "best": ordered[-1]}


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _page_f1(counts: Mapping[str, int]) -> float:
    predicted = int(counts["predicted"])
    gold = int(counts["gold"])
    matches = int(counts["matches"])
    return 100.0 * (2.0 * matches) / (predicted + gold) if predicted + gold else 0.0


def _clean_axis(ax: plt.Axes, *, grid_axis: str | None = "x") -> None:
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    if grid_axis:
        ax.grid(axis=grid_axis, color="#D9D9D9", linewidth=0.55, alpha=0.65, zorder=0)


def _panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.02, 1.04, label, transform=ax.transAxes, ha="left", va="bottom", fontweight="bold", fontsize=8.5)


def _save(fig: plt.Figure, out: Path, stem: str) -> list[str]:
    paths: list[str] = []
    for suffix in ("png", "pdf"):
        path = out / f"{stem}.{suffix}"
        fig.savefig(path, dpi=300 if suffix == "png" else None, bbox_inches="tight", pad_inches=0.03)
        paths.append(str(path))
    plt.close(fig)
    return paths


def _configure_publication_fonts() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 7.8,
            "axes.labelsize": 7.8,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "legend.fontsize": 7.2,
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _zero_shot_payloads() -> list[tuple[str, dict[str, Any]]]:
    return [
        ("Polish", _json(ROOT / "outputs/ijcv_repro/order_invariant_event_f1/polish_test24.json")),
        ("Debussy", _json(ROOT / "outputs/ijcv_repro/order_invariant_event_f1/debussy_transfer24.json")),
    ]


def build_zero_shot_summary(out: Path) -> list[str]:
    payloads = _zero_shot_payloads()
    rows: list[tuple[str, str, float, float, float, tuple[int, int, int]]] = []
    for dataset, payload in payloads:
        for baseline in ("smt", "zeus"):
            boot = payload["paired_bootstrap"][f"staff_minus_{baseline}"]
            diffs = np.asarray(
                [_page_f1(item["staff"]) - _page_f1(item[baseline]) for item in payload["per_page_counts"]],
                dtype=float,
            )
            wins = int(np.sum(diffs > 1e-12))
            ties = int(np.sum(np.abs(diffs) <= 1e-12))
            losses = int(len(diffs) - wins - ties)
            rows.append(
                (
                    dataset,
                    baseline,
                    float(boot["staff_minus_baseline"]),
                    float(boot["ci95_low"]),
                    float(boot["ci95_high"]),
                    (wins, ties, losses),
                )
            )

    differences = np.asarray([row[2] for row in rows], dtype=float)
    wtl = np.asarray([row[5] for row in rows], dtype=int)
    np.testing.assert_allclose(
        np.round(differences, 3),
        [13.352, 15.751, 6.449, 0.736],
        atol=1e-9,
    )
    np.testing.assert_array_equal(
        wtl,
        [[24, 0, 0], [24, 0, 0], [22, 0, 2], [15, 0, 9]],
    )

    fig, ax = plt.subplots(figsize=(6.35, 2.25))
    y_positions = np.array([3.15, 2.15, 0.85, -0.15])
    for y, (dataset, baseline, mean, low, high, wtl) in zip(y_positions, rows):
        ax.errorbar(
            mean,
            y,
            xerr=[[mean - low], [high - mean]],
            fmt=METHOD_MARKERS[baseline],
            markersize=5.3,
            color=METHOD_COLORS[baseline],
            ecolor=METHOD_COLORS[baseline],
            elinewidth=1.15,
            capsize=2.4,
            markeredgecolor="white",
            markeredgewidth=0.45,
            zorder=3,
        )
        ax.text(
            19.25,
            y,
            f"{mean:+.1f}   {wtl[0]}/{wtl[1]}/{wtl[2]}",
            ha="left",
            va="center",
            fontsize=7.2,
            color="#333333",
        )

    ax.axvline(0, color="#777777", linestyle=(0, (3, 2)), linewidth=0.8, zorder=1)
    ax.axhline(1.50, color="#D3D3D3", linewidth=0.65)
    ax.set_yticks(
        y_positions,
        ["Polish   vs. SMT", "Polish   vs. Zeus", "Debussy   vs. SMT", "Debussy   vs. Zeus"],
    )
    ax.set_xlim(-3.5, 25.2)
    ax.set_xticks([-3, 0, 3, 6, 9, 12, 15, 18])
    ax.set_ylim(-0.60, 3.62)
    ax.set_xlabel("Event-F1 improvement over baseline (points)")
    _clean_axis(ax)
    fig.subplots_adjust(left=0.22, right=0.985, top=0.96, bottom=0.23)
    return _save(fig, out, "zero_shot_main_results")


def build_relation_forest(out: Path) -> list[str]:
    sources = [
        ("Debussy", ROOT / "outputs/paper_evidence_20260720/relation_edge_ablation/summary.json", ("notehead_stem_attachment", "beam_stem_group")),
        ("GrandStaff", ROOT / "outputs/paper_evidence_20260720/relation_edge_ablation_grandstaff/summary.json", ("notehead_stem_attachment", "beam_stem_group")),
        ("OLiMPiC", ROOT / "outputs/paper_evidence_20260720/relation_edge_ablation_olimpic/summary.json", ("notehead_stem_attachment", "beam_stem_group")),
    ]
    relation_labels = {
        "notehead_stem_attachment": "notehead-stem",
        "beam_stem_group": "beam-stem",
    }
    relation_markers = {"notehead_stem_attachment": "o", "beam_stem_group": "s"}
    relation_colors = {"notehead_stem_attachment": "#0072B2", "beam_stem_group": "#D55E00"}

    rows: list[tuple[str, str, float, float, float]] = []
    for dataset, source, relations in sources:
        payload = _json(source)
        for relation in relations:
            effect = payload["variants"][f"without_{relation}"]["effect"]
            rows.append((dataset, relation, float(effect["full_minus_ablated"]), float(effect["ci95_low"]), float(effect["ci95_high"])))

    effects = np.asarray([row[2] for row in rows], dtype=float)
    np.testing.assert_allclose(
        np.round(effects, 3),
        [1.470, 1.403, 0.669, 0.583, 1.911, 1.543],
        atol=1e-9,
    )

    y_positions = np.array([5.70, 4.70, 3.10, 2.10, 0.50, -0.50])
    fig, ax = plt.subplots(figsize=(6.35, 2.55))
    for y, (dataset, relation, mean, low, high) in zip(y_positions, rows):
        ax.errorbar(
            mean,
            y,
            xerr=[[mean - low], [high - mean]],
            fmt=relation_markers[relation],
            markersize=5.5,
            color=relation_colors[relation],
            ecolor=relation_colors[relation],
            elinewidth=1.2,
            capsize=2.4,
            markerfacecolor=relation_colors[relation],
            markeredgewidth=1.0,
            zorder=3,
        )
        ax.text(
            high + 0.06,
            y,
            f"{mean:.2f} [{low:.2f}, {high:.2f}]",
            ha="left",
            va="center",
            fontsize=7.2,
            color="#333333",
        )
    ax.axvline(0, color="#777777", linestyle=(0, (3, 2)), linewidth=0.8, zorder=1)
    ax.axhline(3.90, color="#D0D0D0", linewidth=0.65)
    ax.axhline(1.30, color="#D0D0D0", linewidth=0.65)
    ax.set_yticks(y_positions, [relation_labels[r] for _, r, *_ in rows])
    for y, dataset in ((6.17, "Debussy"), (3.57, "GrandStaff"), (0.97, "OLiMPiC")):
        ax.text(
            -0.19,
            y,
            dataset,
            transform=ax.get_yaxis_transform(),
            ha="left",
            va="bottom",
            fontsize=7.6,
            fontweight="bold",
            clip_on=False,
        )
    ax.set_xlim(-0.20, 3.02)
    ax.set_xticks([0, 0.5, 1.0, 1.5, 2.0, 2.5])
    ax.set_ylim(-0.92, 6.37)
    ax.set_xlabel("Event-F1 drop after removing relation (points)")
    _clean_axis(ax)
    handles = [
        Line2D([], [], linestyle="none", marker=relation_markers[r], markersize=5, color=relation_colors[r], label=relation_labels[r])
        for r in relation_markers
    ]
    ax.legend(
        handles=handles,
        frameon=False,
        fontsize=7.2,
        ncol=2,
        loc="lower right",
        bbox_to_anchor=(1.0, 1.0),
        handletextpad=0.25,
        columnspacing=0.9,
    )
    fig.subplots_adjust(left=0.255, right=0.985, top=0.87, bottom=0.20)
    return _save(fig, out, "cross_domain_relation_forest")


def build_binding_gap(out: Path) -> list[str]:
    payload = _json(ROOT / "outputs/paper_evidence_20260720/error_decomposition/metrics.json")
    projections = ("joint", "pitch", "duration")
    scores = np.asarray(
        [
            [float(payload["summary"][method][projection]["f1"]) for projection in projections]
            for method in METHODS
        ],
        dtype=float,
    )
    np.testing.assert_allclose(
        np.round(scores, 3),
        [[8.517, 26.119, 33.901], [2.068, 4.553, 16.530], [7.781, 20.318, 29.550]],
        atol=1e-9,
    )

    fig, ax = plt.subplots(figsize=(3.25, 2.45))
    x = np.arange(len(METHODS), dtype=float)
    series = (
        ("Joint", 0, -0.16, JOINT_COLOR, "D"),
        ("Pitch-only", 1, 0.00, PITCH_COLOR, "o"),
        ("Duration-only", 2, 0.16, DURATION_COLOR, "s"),
    )
    for label, column, offset, color, marker in series:
        values = scores[:, column]
        positions = x + offset
        ax.scatter(
            positions,
            values,
            s=34,
            marker=marker,
            color=color,
            edgecolor="white",
            linewidth=0.5,
            label=label,
            zorder=3,
        )
        for position, value in zip(positions, values):
            ax.annotate(
                f"{value:.1f}",
                (position, value),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=6.9,
                color=color,
            )

    ax.set_xticks(x, [METHOD_LABELS[method] for method in METHODS])
    ax.set_xlim(-0.48, 2.48)
    ax.set_ylim(0, 38.0)
    ax.set_yticks([0, 10, 20, 30])
    ax.set_ylabel("Debussy Event-F1 (%)")
    _clean_axis(ax, grid_axis="y")
    ax.legend(
        frameon=False,
        fontsize=6.7,
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        handletextpad=0.22,
        columnspacing=0.58,
    )
    fig.subplots_adjust(left=0.18, right=0.98, top=0.82, bottom=0.19)
    return _save(fig, out, "debussy_binding_gap")


def load_qualitative_case(root: Path = ROOT) -> dict[str, Any]:
    summary_path = root / "outputs/debussy_abcd_ablation/summary.json"
    shapes_path = root / "outputs/debussy_abcd_ablation/A1B1C1D1/test_0006/symbols/shapes.json"
    run_root = root / "outputs/ijcv_repro/debussy_staff_transfer24/test_0006"
    input_path = root / "data/ijcv_samples/debussy-omr-transfer24/test_0006.png"
    overview_path = run_root / "visuals/v2_1_skeleton_overlay.png"
    mask_manifest_path = run_root / "sam2/masks_all.json"

    summary = _json(summary_path)
    rows = summary["variants"]["A1B1C1D1"]["per_page_counts"]
    scored = [(_page_f1(row), row["page_id"]) for row in rows]
    event_f1, sample_id = max(scored, key=lambda item: item[0])
    if sample_id != QUALITATIVE_SAMPLE_ID:
        raise AssertionError(f"best full-pipeline system changed: {sample_id} ({event_f1:.6f})")
    np.testing.assert_allclose(event_f1, 24.4725738397, atol=1e-9)

    shapes = _json(shapes_path)
    symbols = {symbol["id"]: symbol for symbol in shapes["symbols"]}
    required_ids = (*QUALITATIVE_NOTEHEAD_IDS, *QUALITATIVE_STEM_IDS, QUALITATIVE_BEAM_ID)
    missing_symbols = [symbol_id for symbol_id in required_ids if symbol_id not in symbols]
    if missing_symbols:
        raise KeyError(f"missing selected symbols: {missing_symbols}")

    masks = _json(mask_manifest_path)["masks"]
    mask_records = {record["symbol_id"]: record for record in masks}
    missing_masks = [
        symbol_id
        for symbol_id in (*QUALITATIVE_NOTEHEAD_IDS, QUALITATIVE_BEAM_ID)
        if symbol_id not in mask_records
    ]
    if missing_masks:
        raise KeyError(f"missing selected masks: {missing_masks}")

    head_stem_edges = {
        (relation["source"], target)
        for relation in shapes["relations"]
        if relation["type"] == "notehead_stem_attachment"
        for target in relation["targets"]
        if relation["source"] in QUALITATIVE_NOTEHEAD_IDS and target in QUALITATIVE_STEM_IDS
    }
    beam_stem_edges = {
        (relation["source"], target)
        for relation in shapes["relations"]
        if relation["type"] == "beam_stem_group"
        for target in relation["targets"]
        if relation["source"] == QUALITATIVE_BEAM_ID and target in QUALITATIVE_STEM_IDS
    }
    direct_notehead_beam_edges = {
        (relation["source"], target)
        for relation in shapes["relations"]
        for target in relation["targets"]
        if (relation["source"] in QUALITATIVE_NOTEHEAD_IDS and target == QUALITATIVE_BEAM_ID)
        or (relation["source"] == QUALITATIVE_BEAM_ID and target in QUALITATIVE_NOTEHEAD_IDS)
    }
    expected_head_edges = set(zip(QUALITATIVE_NOTEHEAD_IDS, QUALITATIVE_STEM_IDS))
    expected_beam_edges = {(QUALITATIVE_BEAM_ID, stem_id) for stem_id in QUALITATIVE_STEM_IDS}
    if head_stem_edges != expected_head_edges or beam_stem_edges != expected_beam_edges:
        raise AssertionError("selected typed relation chain no longer matches the approved case")
    if direct_notehead_beam_edges:
        raise AssertionError("selected graph unexpectedly contains a direct notehead-beam relation")

    return {
        "provenance": {
            "summary_path": summary_path,
            "shapes_path": shapes_path,
            "mask_manifest_path": mask_manifest_path,
        },
        "sample_id": sample_id,
        "event_f1": event_f1,
        "crop": QUALITATIVE_CROP,
        "input_path": input_path,
        "overview_path": overview_path,
        "staves": shapes["staves"],
        "symbols": symbols,
        "notehead_ids": QUALITATIVE_NOTEHEAD_IDS,
        "stem_ids": QUALITATIVE_STEM_IDS,
        "beam_id": QUALITATIVE_BEAM_ID,
        "mask_records": mask_records,
        "head_stem_edges": sorted(head_stem_edges),
        "beam_stem_edges": sorted(beam_stem_edges),
        "direct_notehead_beam_edges": sorted(direct_notehead_beam_edges),
    }


def _bbox_center(symbol: Mapping[str, Any]) -> tuple[float, float]:
    x0, y0, x1, y1 = (float(value) for value in symbol["bbox"])
    return (0.5 * (x0 + x1), 0.5 * (y0 + y1))


def _show_local_background(ax: plt.Axes, image: np.ndarray, crop: tuple[int, int, int, int]) -> None:
    x0, y0, x1, y1 = crop
    ax.imshow(image)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _mask_rgba(mask_path: Path, color: str, alpha: float) -> np.ndarray:
    with Image.open(mask_path) as image:
        mask = np.asarray(image.convert("L")) > 0
    rgb = np.asarray(to_rgb(color), dtype=float)
    rgba = np.zeros((*mask.shape, 4), dtype=float)
    rgba[mask, :3] = rgb
    rgba[mask, 3] = alpha
    return rgba


def build_qualitative_pipeline(out: Path) -> list[str]:
    _configure_publication_fonts()
    case = load_qualitative_case()
    with Image.open(case["input_path"]) as input_image:
        image = np.asarray(input_image.convert("RGB"))
    crop = case["crop"]
    x0, y0, x1, y1 = crop
    symbols = case["symbols"]
    selected_ids = (*case["notehead_ids"], *case["stem_ids"], case["beam_id"])
    object_colors = {
        **{symbol_id: HEAD_COLOR for symbol_id in case["notehead_ids"]},
        **{symbol_id: STEM_COLOR for symbol_id in case["stem_ids"]},
        case["beam_id"]: BEAM_COLOR,
    }

    with Image.open(case["overview_path"]) as overview_image:
        overview = np.asarray(overview_image.convert("RGB"))

    fig = plt.figure(figsize=(7.0, 2.78))
    grid = fig.add_gridspec(2, 5, height_ratios=(1.0, 0.92), hspace=0.10, wspace=0.035)
    overview_ax = fig.add_subplot(grid[0, :])
    overview_ax.imshow(overview)
    overview_ax.add_patch(
        Rectangle(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            facecolor="none",
            edgecolor="#D55E00",
            linewidth=1.4,
        )
    )
    overview_ax.text(
        0.5 * (x0 + x1),
        max(8, y0 - 18),
        "zoomed below",
        ha="center",
        va="bottom",
        fontsize=6.4,
        color="#D55E00",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 0.8},
    )
    overview_ax.set_title("(a) Full-system structure", fontsize=7.2, pad=2.0)
    overview_ax.set_xticks([])
    overview_ax.set_yticks([])
    for spine in overview_ax.spines.values():
        spine.set_visible(False)

    axes = [fig.add_subplot(grid[1, index]) for index in range(5)]
    titles = ("(b) Staff", "(c) Accepted symbols", "(d) Masks", "(e) Relations", "(f) Graph")
    for ax, title in zip(axes, titles):
        _show_local_background(ax, image, crop)
        ax.set_title(title, fontsize=7.2, pad=2.0)

    local_staff = min(
        case["staves"],
        key=lambda staff: abs(np.mean(staff["lines"]) - 0.5 * (y0 + y1)),
    )
    axes[0].add_patch(
        Rectangle(
            (x0, min(local_staff["lines"]) - 3),
            x1 - x0,
            max(local_staff["lines"]) - min(local_staff["lines"]) + 6,
            facecolor=STAFF_COLOR,
            edgecolor=STAFF_COLOR,
            linewidth=0.8,
            alpha=0.10,
        )
    )
    for y in local_staff["lines"]:
        axes[0].plot([x0, x1], [y, y], color=STAFF_COLOR, linewidth=0.9)

    for symbol_id in selected_ids:
        bx0, by0, bx1, by1 = symbols[symbol_id]["bbox"]
        axes[1].add_patch(
            Rectangle(
                (bx0, by0),
                bx1 - bx0,
                by1 - by0,
                facecolor="none",
                edgecolor=object_colors[symbol_id],
                linewidth=1.0,
            )
        )

    for symbol_id in (*case["notehead_ids"], case["beam_id"]):
        record = case["mask_records"][symbol_id]
        mask_path = ROOT / Path(record["mask_path"])
        axes[2].imshow(_mask_rgba(mask_path, object_colors[symbol_id], 0.52))
    axes[2].set_xlim(x0, x1)
    axes[2].set_ylim(y1, y0)

    for head_id, stem_id in case["head_stem_edges"]:
        hx, hy = _bbox_center(symbols[head_id])
        sx, sy = _bbox_center(symbols[stem_id])
        axes[3].plot([hx, sx], [hy, sy], color=HEAD_COLOR, linewidth=1.5, marker="o", markersize=2.2)
    for beam_id, stem_id in case["beam_stem_edges"]:
        bx, by = _bbox_center(symbols[beam_id])
        sx, sy = _bbox_center(symbols[stem_id])
        axes[3].plot([sx, bx], [sy, by], color=BEAM_COLOR, linewidth=1.5, marker="o", markersize=2.2)

    for image_artist in list(axes[4].images):
        image_artist.remove()
    axes[4].set_facecolor("white")
    axes[4].set_xlim(x0, x1)
    axes[4].set_ylim(y1, y0)
    for symbol_id in (*case["notehead_ids"], case["beam_id"]):
        record = case["mask_records"][symbol_id]
        mask_path = ROOT / Path(record["mask_path"])
        axes[4].imshow(_mask_rgba(mask_path, object_colors[symbol_id], 0.78))
    for stem_id in case["stem_ids"]:
        bx0, by0, bx1, by1 = symbols[stem_id]["bbox"]
        axes[4].add_patch(Rectangle((bx0, by0), bx1 - bx0, by1 - by0, color=STEM_COLOR, alpha=0.82))
    for head_id, stem_id in case["head_stem_edges"]:
        hx, hy = _bbox_center(symbols[head_id])
        sx, sy = _bbox_center(symbols[stem_id])
        axes[4].plot([hx, sx], [hy, sy], color=HEAD_COLOR, linewidth=1.4)
    for beam_id, stem_id in case["beam_stem_edges"]:
        bx, by = _bbox_center(symbols[beam_id])
        sx, sy = _bbox_center(symbols[stem_id])
        axes[4].plot([sx, bx], [sy, by], color=BEAM_COLOR, linewidth=1.4)

    handles = [
        Patch(facecolor=HEAD_COLOR, label="notehead"),
        Patch(facecolor=STEM_COLOR, label="stem"),
        Patch(facecolor=BEAM_COLOR, label="beam"),
        Line2D([], [], color=HEAD_COLOR, linewidth=1.5, label="notehead-stem"),
        Line2D([], [], color=BEAM_COLOR, linewidth=1.5, label="beam-stem"),
    ]
    fig.legend(handles=handles, frameon=False, ncol=5, loc="lower center", bbox_to_anchor=(0.5, -0.005), fontsize=6.2)
    fig.subplots_adjust(left=0.01, right=0.995, top=0.93, bottom=0.13)
    return _save(fig, out, "debussy_qualitative_pipeline")


def build_polish_structure_diagnosis(out: Path) -> list[str]:
    # These audited proxy-SER values are supplied in the paper's approved figure specification.
    legacy = 96.1367
    stages = (
        ("Canonical SER", 95.9589),
        ("Staff-constrained diagnosis", 91.7498),
        ("Page-constrained diagnosis", 85.2584),
        ("Pitch-duration rebinding", 77.1959),
    )
    ser = np.asarray([value for _, value in stages], dtype=float)
    np.testing.assert_allclose(
        np.round(ser, 2),
        [95.96, 91.75, 85.26, 77.20],
        atol=1e-9,
    )

    fig, ax = plt.subplots(figsize=(6.35, 2.40))
    fig.patch.set_facecolor("#F3F3F3")
    ax.set_facecolor("#F3F3F3")
    y_positions = np.array([3.0, 2.0, 1.0, 0.0])
    colors = [METHOD_COLORS["staff"], "#56B4E9", "#009E73", "#CC79A7"]

    ax.scatter(ser[0], y_positions[0], s=52, color=colors[0], edgecolor="white", linewidth=0.6, zorder=4)
    ax.text(ser[0] - 0.45, y_positions[0], f"{ser[0]:.2f}", ha="right", va="center", fontsize=7.4, color="#222222")
    for index in range(1, len(stages)):
        previous = ser[index - 1]
        current = ser[index]
        y = y_positions[index]
        ax.plot(
            [previous, previous, current],
            [y_positions[index - 1], y, y],
            color="#A9A9A9",
            linewidth=1.05,
            zorder=1,
        )
        ax.annotate(
            "",
            xy=(current, y),
            xytext=(previous, y),
            arrowprops={"arrowstyle": "-|>", "color": colors[index], "linewidth": 1.6, "shrinkA": 0, "shrinkB": 3},
        )
        ax.scatter(current, y, s=48, color=colors[index], edgecolor="white", linewidth=0.6, zorder=4)
        ax.text(current - 0.45, y, f"{current:.2f}", ha="right", va="center", fontsize=7.4, color="#222222")
        drop = previous - current
        ax.text(
            (previous + current) / 2,
            y + 0.17,
            f"-{drop:.2f} pp",
            ha="center",
            va="bottom",
            fontsize=7.1,
            color=colors[index],
            fontweight="bold",
        )

    ax.text(
        0.985,
        0.055,
        f"Legacy → canonical: {95.9589 - legacy:.2f} pp",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.1,
        color="#555555",
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "white", "edgecolor": "#C8C8C8", "linewidth": 0.6},
    )
    ax.set_yticks(y_positions, [label for label, _ in stages])
    ax.set_xlim(74.7, 98.4)
    ax.set_xticks([76, 80, 84, 88, 92, 96])
    ax.set_ylim(-0.48, 3.52)
    ax.set_xlabel("Structural error rate (%)")
    ax.set_title("Diagnostic lower bounds—not deployable results", fontsize=8.2, fontweight="bold", pad=8)
    _clean_axis(ax)
    fig.subplots_adjust(left=0.285, right=0.985, top=0.82, bottom=0.23)
    return _save(fig, out, "polish_structure_diagnosis")


def build_per_item_small_multiples(out: Path) -> list[str]:
    payloads = _zero_shot_payloads()
    rows: list[tuple[str, str, dict[str, Any]]] = []
    for dataset, payload in payloads:
        for baseline in ("smt", "zeus"):
            rows.append((dataset, baseline, payload))
    fig, axes = plt.subplots(2, 2, figsize=(6.9, 4.15), sharex=True, sharey=True)
    for ax, (dataset, baseline, payload) in zip(axes.flat, rows):
        diffs = np.asarray([_page_f1(item["staff"]) - _page_f1(item[baseline]) for item in payload["per_page_counts"]], dtype=float)
        boot = payload["paired_bootstrap"][f"staff_minus_{baseline}"]
        mean = float(boot["staff_minus_baseline"])
        low = float(boot["ci95_low"])
        high = float(boot["ci95_high"])
        rng = np.random.default_rng(20260720)
        jitter = rng.uniform(-0.11, 0.11, len(diffs))
        ax.scatter(diffs, jitter, s=16, color=METHOD_COLORS[baseline], alpha=0.48, edgecolor="none", zorder=2)
        ax.errorbar(
            mean,
            0.34,
            xerr=[[mean - low], [high - mean]],
            fmt="D",
            markersize=5.1,
            color="#222222",
            ecolor="#222222",
            elinewidth=1.2,
            capsize=2.5,
            markerfacecolor="white",
            markeredgewidth=1.0,
            zorder=4,
        )
        wins = int(np.sum(diffs > 1e-12))
        ties = int(np.sum(np.abs(diffs) <= 1e-12))
        losses = int(len(diffs) - wins - ties)
        ax.axvline(0, color="#777777", linestyle=(0, (3, 2)), linewidth=0.75)
        ax.set_ylim(-0.25, 0.55)
        ax.set_yticks([])
        ax.text(0.02, 0.94, f"{dataset}: RIGOR-OMR - {METHOD_LABELS[baseline]}", transform=ax.transAxes, ha="left", va="top", fontsize=7.5, fontweight="bold")
        ax.text(0.98, 0.94, f"W/T/L {wins}/{ties}/{losses}", transform=ax.transAxes, ha="right", va="top", fontsize=7.0, color="#555555")
        ax.text(0.98, 0.09, f"Δ {mean:+.2f}  95% CI [{low:.2f}, {high:.2f}]", transform=ax.transAxes, ha="right", va="bottom", fontsize=7.0, color="#333333")
        _clean_axis(ax)
    axes[1, 0].set_xlabel("Per-item Event-F1 difference (points) ↑")
    axes[1, 1].set_xlabel("Per-item Event-F1 difference (points) ↑")
    axes[0, 0].set_xlim(-16, 39)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.98, bottom=0.14, hspace=0.23, wspace=0.17)
    return _save(fig, out, "per_item_difference_small_multiples")


def build_density_points(out: Path) -> list[str]:
    payload = _json(ROOT / "outputs/ijcv_repro/order_invariant_event_f1/debussy_transfer24.json")
    rows = sorted(payload["per_page_counts"], key=lambda row: (row["staff"]["gold"], row["page_id"]))
    bins: Sequence[np.ndarray] = np.array_split(np.asarray(rows, dtype=object), 3)
    fig, ax = plt.subplots(figsize=(3.25, 2.5))
    x = np.arange(3, dtype=float)
    offsets = {"staff": -0.14, "smt": 0.0, "zeus": 0.14}
    for method in METHODS:
        values: list[float] = []
        for group in bins:
            counts = {
                "matches": sum(int(row[method]["matches"]) for row in group),
                "predicted": sum(int(row[method]["predicted"]) for row in group),
                "gold": sum(int(row[method]["gold"]) for row in group),
            }
            values.append(_page_f1(counts))
        ax.scatter(
            x + offsets[method],
            values,
            s=34,
            marker=METHOD_MARKERS[method],
            color=METHOD_COLORS[method],
            edgecolor="white",
            linewidth=0.5,
            label=METHOD_LABELS[method],
            zorder=3,
        )
    labels = []
    for i, group in enumerate(bins):
        low = min(int(row["staff"]["gold"]) for row in group)
        high = max(int(row["staff"]["gold"]) for row in group)
        labels.append(f"Q{i + 1}\n{low}-{high}")
    ax.set_xticks(x, labels)
    ax.set_xlim(-0.48, 2.48)
    ax.set_ylabel("Aggregated Event-F1 (%) ↑")
    ax.set_xlabel("Gold-event density tertile (8 systems each)")
    _clean_axis(ax, grid_axis="y")
    ax.legend(frameon=False, fontsize=6.9, ncol=3, loc="lower left", bbox_to_anchor=(-0.03, 1.01), handletextpad=0.25, columnspacing=0.7)
    fig.subplots_adjust(left=0.20, right=0.98, top=0.82, bottom=0.25)
    return _save(fig, out, "debussy_density_stratification")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the approved AAAI result-visualization set.")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    _configure_publication_fonts()

    jobs = [
        ("zero_shot_main_results", build_zero_shot_summary, ["polish_test24.json", "debussy_transfer24.json"]),
        ("cross_domain_relation_forest", build_relation_forest, ["relation_edge_ablation/*/summary.json"]),
        ("debussy_binding_gap", build_binding_gap, ["error_decomposition/metrics.json"]),
        (
            "debussy_qualitative_pipeline",
            build_qualitative_pipeline,
            [
                "debussy_abcd_ablation/summary.json",
                "A1B1C1D1/test_0006/symbols/shapes.json",
                "test_0006/sam2/masks_all.json",
            ],
        ),
        ("polish_structure_diagnosis", build_polish_structure_diagnosis, ["approved Polish structural-diagnostic specification"]),
        ("per_item_difference_small_multiples", build_per_item_small_multiples, ["polish_test24.json", "debussy_transfer24.json"]),
        ("debussy_density_stratification", build_density_points, ["debussy_transfer24.json"]),
    ]
    manifest: dict[str, Any] = {"figures": []}
    for name, builder, sources in jobs:
        paths = builder(out)
        manifest["figures"].append({"name": name, "paths": paths, "sources": sources})
    (out / "figure_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
