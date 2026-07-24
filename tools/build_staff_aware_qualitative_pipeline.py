from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from PIL import Image

try:
    from tools import refine_masks_sam2 as sam2_refiner
except ModuleNotFoundError:
    import refine_masks_sam2 as sam2_refiner


ROOT = Path(__file__).resolve().parent.parent
SAMPLE_ID = "test_0006"
CROP = (965, 390, 1170, 515)
NOTEHEAD_IDS = ("deim_000003", "deim_000013")
STEM_IDS = ("synthetic_stem_00012", "synthetic_stem_00013")
BEAM_ID = "deim_000069"
PROMPT_BEAM_IDS = (BEAM_ID, "deim_000278")

HEAD_COLOR = "#D55E00"
STEM_COLOR = "#7B3294"
BEAM_COLOR = "#0072B2"
STAFF_COLOR = "#009E73"
POSITIVE_COLOR = "#00A65A"
NEGATIVE_COLOR = "#CC3311"


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _f1(counts: Mapping[str, int]) -> float:
    denominator = int(counts["predicted"]) + int(counts["gold"])
    return 200.0 * int(counts["matches"]) / denominator if denominator else 0.0


def _relation_edges(shapes: Mapping[str, Any], relation_type: str, source_ids: set[str]) -> set[tuple[str, str]]:
    return {
        (relation["source"], target)
        for relation in shapes["relations"]
        if relation["type"] == relation_type and relation["source"] in source_ids
        for target in relation["targets"]
        if target in set(STEM_IDS)
    }


def prompt_marker_groups(
    symbol: Mapping[str, Any],
    staves: list[dict[str, Any]],
    image_size: tuple[int, int],
) -> dict[str, list[list[float]]]:
    coords, labels, _ = sam2_refiner.build_prompt_points(
        symbol, staves, image_size, "box_pos_neg_staff"
    )
    if coords is None or labels is None:
        raise AssertionError(f"prompt generation returned no points for {symbol['id']}")
    _, metadata = sam2_refiner.adaptive_prompt_box(dict(symbol), image_size)
    filtered_coords, filtered_labels, removed = sam2_refiner.filter_negative_points_outside_support(
        coords, labels, metadata["detector_box"]
    )
    coords_array = np.asarray(coords, dtype=np.float32)
    labels_array = np.asarray(labels, dtype=np.int32)
    filtered_array = np.asarray(filtered_coords, dtype=np.float32)
    filtered_label_array = np.asarray(filtered_labels, dtype=np.int32)
    retained = filtered_array[filtered_label_array == 0]
    x0, y0, x1, y1 = map(float, metadata["detector_box"])
    suppressed_mask = (
        (labels_array == 0)
        & (coords_array[:, 0] >= x0)
        & (coords_array[:, 0] <= x1)
        & (coords_array[:, 1] >= y0)
        & (coords_array[:, 1] <= y1)
    )
    suppressed = coords_array[suppressed_mask]
    if len(suppressed) != removed:
        raise AssertionError("suppressed prompt reconstruction disagrees with the SAM2 filter")
    return {
        "positive": coords_array[labels_array == 1].tolist(),
        "retained_negative": retained.tolist(),
        "suppressed_negative": suppressed.tolist(),
    }


def load_case(root: Path = ROOT) -> dict[str, Any]:
    run_root = root / "outputs/sam2_staff_aware_probe_support_filter"
    page_root = run_root / SAMPLE_ID
    summary_path = run_root / "summary.json"
    shapes_path = page_root / "symbols/shapes.json"
    masks_path = page_root / "sam2/masks.json"
    input_path = root / f"data/ijcv_samples/debussy-omr-transfer24/{SAMPLE_ID}.png"
    overview_path = root / f"outputs/ijcv_repro/debussy_staff_transfer24/{SAMPLE_ID}/visuals/v2_1_skeleton_overlay.png"

    summary = _json(summary_path)
    if summary["mask_selection"] != "safe_residual" or SAMPLE_ID not in summary["pages"]:
        raise AssertionError("summary does not contain a test_0006 safe-residual run")
    matching_rows = [row for row in summary["rows"] if row["page"] == SAMPLE_ID]
    if len(matching_rows) != 1:
        raise AssertionError(f"expected one {SAMPLE_ID} summary row, found {len(matching_rows)}")
    row = matching_rows[0]
    shapes = _json(shapes_path)
    symbols = {symbol["id"]: symbol for symbol in shapes["symbols"]}
    required_ids = (*NOTEHEAD_IDS, *STEM_IDS, BEAM_ID)
    missing = [symbol_id for symbol_id in required_ids if symbol_id not in symbols]
    if missing:
        raise KeyError(f"missing selected symbols: {missing}")

    masks_payload = _json(masks_path)
    mask_records = {record["symbol_id"]: record for record in masks_payload["masks"]}
    missing_masks = [symbol_id for symbol_id in (*NOTEHEAD_IDS, BEAM_ID) if symbol_id not in mask_records]
    if missing_masks:
        raise KeyError(f"missing selected staff-aware masks: {missing_masks}")

    head_edges = _relation_edges(shapes, "notehead_stem_attachment", set(NOTEHEAD_IDS))
    beam_edges = _relation_edges(shapes, "beam_stem_group", {BEAM_ID})
    expected_heads = set(zip(NOTEHEAD_IDS, STEM_IDS))
    expected_beams = {(BEAM_ID, stem_id) for stem_id in STEM_IDS}
    if not expected_heads.issubset(head_edges) or beam_edges != expected_beams:
        raise AssertionError("selected safe-residual relation chain changed")

    source_symbols_path = Path(masks_payload["symbols_json"])
    if not source_symbols_path.is_absolute():
        source_symbols_path = root / source_symbols_path
    source_payload = _json(source_symbols_path)
    source_symbols = {symbol["id"]: symbol for symbol in source_payload["symbols"]}
    with Image.open(input_path) as input_image:
        image_size = input_image.size
    prompt_markers = {
        symbol_id: prompt_marker_groups(source_symbols[symbol_id], source_payload["staves"], image_size)
        for symbol_id in PROMPT_BEAM_IDS
    }
    for symbol_id, marker_groups in prompt_markers.items():
        record_prompt = mask_records[symbol_id]["prompt"]
        if len(marker_groups["retained_negative"]) != len(record_prompt["negative_points"]):
            raise AssertionError(f"retained negative count changed for {symbol_id}")
        if len(marker_groups["suppressed_negative"]) != int(record_prompt["removed_support_negatives"]):
            raise AssertionError(f"suppressed negative count changed for {symbol_id}")

    return {
        "sample_id": SAMPLE_ID,
        "crop": CROP,
        "input_path": input_path,
        "overview_path": overview_path,
        "staves": shapes["staves"],
        "symbols": symbols,
        "mask_records": mask_records,
        "prompt_markers": prompt_markers,
        "notehead_ids": NOTEHEAD_IDS,
        "stem_ids": STEM_IDS,
        "beam_id": BEAM_ID,
        "head_stem_edges": sorted(expected_heads),
        "beam_stem_edges": sorted(beam_edges),
        "box_event_f1": _f1(row["box_only_baseline"]),
        "event_f1": _f1(row["method_aligned"]),
        "box_predicted_events": int(row["box_only_baseline"]["predicted"]),
        "predicted_events": int(row["method_aligned"]["predicted"]),
        "provenance": {"summary": summary_path, "shapes": shapes_path, "masks": masks_path},
    }


def _configure_fonts() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 7.8,
            "axes.labelsize": 7.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _show_crop(ax: plt.Axes, image: np.ndarray, crop: tuple[int, int, int, int]) -> None:
    x0, y0, x1, y1 = crop
    ax.imshow(image)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _center(symbol: Mapping[str, Any]) -> tuple[float, float]:
    x0, y0, x1, y1 = map(float, symbol["bbox"])
    return 0.5 * (x0 + x1), 0.5 * (y0 + y1)


def _mask_rgba(mask_path: Path, color: str, alpha: float) -> np.ndarray:
    with Image.open(mask_path) as image:
        mask = np.asarray(image.convert("L")) > 0
    rgba = np.zeros((*mask.shape, 4), dtype=float)
    rgba[mask, :3] = np.asarray(to_rgb(color))
    rgba[mask, 3] = alpha
    return rgba


def _draw_box(ax: plt.Axes, box: list[float], *, color: str, linewidth: float, linestyle: str) -> None:
    x0, y0, x1, y1 = box
    ax.add_patch(
        Rectangle(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            facecolor="none",
            edgecolor=color,
            linewidth=linewidth,
            linestyle=linestyle,
        )
    )


def build(out: Path, root: Path = ROOT) -> list[str]:
    _configure_fonts()
    case = load_case(root)
    out.mkdir(parents=True, exist_ok=True)
    with Image.open(case["input_path"]) as source:
        image = np.asarray(source.convert("RGB"))
    with Image.open(case["overview_path"]) as source:
        overview = np.asarray(source.convert("RGB"))

    crop = case["crop"]
    x0, y0, x1, y1 = crop
    symbols = case["symbols"]
    selected_ids = (*case["notehead_ids"], *case["stem_ids"], case["beam_id"])
    colors = {
        **{symbol_id: HEAD_COLOR for symbol_id in case["notehead_ids"]},
        **{symbol_id: STEM_COLOR for symbol_id in case["stem_ids"]},
        case["beam_id"]: BEAM_COLOR,
    }

    fig = plt.figure(figsize=(7.0, 2.78))
    grid = fig.add_gridspec(2, 5, height_ratios=(1.0, 0.92), hspace=0.10, wspace=0.035)
    overview_ax = fig.add_subplot(grid[0, :])
    overview_ax.imshow(overview)
    overview_ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor="none", edgecolor=HEAD_COLOR, linewidth=1.4))
    overview_ax.text(
        0.5 * (x0 + x1),
        max(8, y0 - 18),
        "zoomed below",
        ha="center",
        va="bottom",
        fontsize=6.4,
        color=HEAD_COLOR,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 0.8},
    )
    overview_ax.set_title("(a) Full-system structure", fontsize=7.2, pad=2.0)
    overview_ax.set_xticks([])
    overview_ax.set_yticks([])
    for spine in overview_ax.spines.values():
        spine.set_visible(False)

    axes = [fig.add_subplot(grid[1, index]) for index in range(5)]
    titles = (
        "(b) Staff-aware prompts",
        "(c) Accepted symbols",
        "(d) Staff-aware masks",
        "(e) Safe-residual relations",
        "(f) Graph",
    )
    for ax, title in zip(axes, titles):
        _show_crop(ax, image, crop)
        ax.set_title(title, fontsize=7.0, pad=2.0)

    local_staff = min(case["staves"], key=lambda staff: abs(np.mean(staff["lines"]) - 0.5 * (y0 + y1)))
    for y in local_staff["lines"]:
        axes[0].plot([x0, x1], [y, y], color=STAFF_COLOR, linewidth=0.65, alpha=0.72)
    for symbol_id in (*case["notehead_ids"], case["beam_id"]):
        prompt = case["mask_records"][symbol_id]["prompt"]
        _draw_box(axes[0], prompt["detector_box"], color="#666666", linewidth=0.55, linestyle=":")
        _draw_box(axes[0], prompt["adaptive_box"], color=colors[symbol_id], linewidth=1.0, linestyle="--")
        if prompt["positive_points"]:
            px, py = zip(*prompt["positive_points"])
            axes[0].scatter(px, py, s=10, c=POSITIVE_COLOR, edgecolors="white", linewidths=0.35, zorder=6)
        if prompt["negative_points"]:
            nx, ny = zip(*prompt["negative_points"])
            axes[0].scatter(nx, ny, s=12, c=NEGATIVE_COLOR, marker="x", linewidths=0.7, zorder=6)
    axes[0].text(
        0.02,
        0.03,
        "beam: 3 staff-line negatives suppressed",
        transform=axes[0].transAxes,
        fontsize=4.8,
        color="#333333",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 0.5},
    )

    for symbol_id in selected_ids:
        _draw_box(axes[1], symbols[symbol_id]["bbox"], color=colors[symbol_id], linewidth=1.0, linestyle="-")

    for symbol_id in (*case["notehead_ids"], case["beam_id"]):
        record = case["mask_records"][symbol_id]
        axes[2].imshow(_mask_rgba(Path(record["mask_path"]), colors[symbol_id], 0.52))
    axes[2].set_xlim(x0, x1)
    axes[2].set_ylim(y1, y0)

    for head_id, stem_id in case["head_stem_edges"]:
        hx, hy = _center(symbols[head_id])
        sx, sy = _center(symbols[stem_id])
        axes[3].plot([hx, sx], [hy, sy], color=HEAD_COLOR, linewidth=1.5, marker="o", markersize=2.2)
    for beam_id, stem_id in case["beam_stem_edges"]:
        bx, by = _center(symbols[beam_id])
        sx, sy = _center(symbols[stem_id])
        axes[3].plot([sx, bx], [sy, by], color=BEAM_COLOR, linewidth=1.5, marker="o", markersize=2.2)

    for artist in list(axes[4].images):
        artist.remove()
    axes[4].set_facecolor("white")
    axes[4].set_xlim(x0, x1)
    axes[4].set_ylim(y1, y0)
    for symbol_id in (*case["notehead_ids"], case["beam_id"]):
        record = case["mask_records"][symbol_id]
        axes[4].imshow(_mask_rgba(Path(record["mask_path"]), colors[symbol_id], 0.78))
    for stem_id in case["stem_ids"]:
        bx0, by0, bx1, by1 = symbols[stem_id]["bbox"]
        axes[4].add_patch(Rectangle((bx0, by0), bx1 - bx0, by1 - by0, color=STEM_COLOR, alpha=0.82))
    for head_id, stem_id in case["head_stem_edges"]:
        hx, hy = _center(symbols[head_id])
        sx, sy = _center(symbols[stem_id])
        axes[4].plot([hx, sx], [hy, sy], color=HEAD_COLOR, linewidth=1.4)
    for beam_id, stem_id in case["beam_stem_edges"]:
        bx, by = _center(symbols[beam_id])
        sx, sy = _center(symbols[stem_id])
        axes[4].plot([sx, bx], [sy, by], color=BEAM_COLOR, linewidth=1.4)

    handles = [
        Patch(facecolor=HEAD_COLOR, label="notehead"),
        Patch(facecolor=STEM_COLOR, label="stem"),
        Patch(facecolor=BEAM_COLOR, label="beam"),
        Line2D([], [], color=POSITIVE_COLOR, marker="o", linestyle="none", markersize=3.2, label="positive prompt"),
        Line2D([], [], color=HEAD_COLOR, linewidth=1.5, label="notehead-stem"),
        Line2D([], [], color=BEAM_COLOR, linewidth=1.5, label="beam-stem"),
    ]
    fig.legend(handles=handles, frameon=False, ncol=6, loc="lower center", bbox_to_anchor=(0.5, -0.005), fontsize=5.7)
    fig.subplots_adjust(left=0.01, right=0.995, top=0.93, bottom=0.13)

    paths: list[str] = []
    for suffix in ("png", "pdf"):
        path = out / f"debussy_qualitative_pipeline.{suffix}"
        fig.savefig(path, dpi=300 if suffix == "png" else None, bbox_inches="tight", pad_inches=0.03)
        paths.append(str(path))
    plt.close(fig)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the test_0006 staff-aware qualitative pipeline figure.")
    parser.add_argument("--out", type=Path, default=ROOT / "output/pdf")
    args = parser.parse_args()
    for path in build(args.out):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
