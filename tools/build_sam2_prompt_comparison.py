from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from PIL import Image


ROOT = Path(__file__).resolve().parent.parent
CLASS_COLORS = {
    "filled_notehead": "#E66101",
    "open_notehead": "#E66101",
    "stem": "#9C27B0",
    "beam": "#0086D1",
}


def binary_mask_metrics(box_only: np.ndarray, staff_aware: np.ndarray) -> dict[str, float | int]:
    box = np.asarray(box_only, dtype=bool)
    aware = np.asarray(staff_aware, dtype=bool)
    if box.shape != aware.shape:
        raise ValueError("binary masks must have the same shape")
    intersection = int(np.logical_and(box, aware).sum())
    union = int(np.logical_or(box, aware).sum())
    return {
        "box_only_area": int(box.sum()),
        "staff_aware_area": int(aware.sum()),
        "intersection": intersection,
        "union": union,
        "iou": intersection / union if union else 1.0,
        "changed_pixels": int(np.logical_xor(box, aware).sum()),
    }


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_mask_path(value: str, manifest: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    root_candidate = ROOT / path
    if root_candidate.exists():
        return root_candidate
    return manifest.parent / path


def _load_mask(record: dict[str, Any], manifest: Path) -> np.ndarray:
    with Image.open(_resolve_mask_path(str(record["mask_path"]), manifest)) as image:
        return np.asarray(image.convert("L")) > 0


def _rgba(mask: np.ndarray, color: str, alpha: float) -> np.ndarray:
    rgb = np.asarray(plt.matplotlib.colors.to_rgb(color), dtype=float)
    rgba = np.zeros((*mask.shape, 4), dtype=float)
    rgba[mask, :3] = rgb
    rgba[mask, 3] = alpha
    return rgba


def _parse_crop(value: str | None, records: list[dict[str, Any]], image_shape: tuple[int, int]) -> tuple[int, int, int, int]:
    height, width = image_shape
    if value:
        parts = [int(item.strip()) for item in value.split(",")]
        if len(parts) != 4:
            raise ValueError("--crop must be x0,y0,x1,y1")
        x0, y0, x1, y1 = parts
    else:
        boxes = [record.get("prompt", {}).get("image_crop") or record["bbox"] for record in records]
        x0 = min(int(box[0]) for box in boxes)
        y0 = min(int(box[1]) for box in boxes)
        x1 = max(int(box[2]) for box in boxes)
        y1 = max(int(box[3]) for box in boxes)
        padding = max(8, int(round(0.08 * max(x1 - x0, y1 - y0))))
        x0, y0, x1, y1 = x0 - padding, y0 - padding, x1 + padding, y1 + padding
    return max(0, x0), max(0, y0), min(width, x1), min(height, y1)


def _show_crop(ax: plt.Axes, image: np.ndarray, crop: tuple[int, int, int, int]) -> None:
    x0, y0, x1, y1 = crop
    ax.imshow(image)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def build_comparison(
    image_path: Path,
    box_manifest: Path,
    staff_manifest: Path,
    symbol_ids: list[str],
    output: Path,
    crop_value: str | None = None,
) -> dict[str, Any]:
    box_payload = _read_json(box_manifest)
    staff_payload = _read_json(staff_manifest)
    box_records = {str(item["symbol_id"]): item for item in box_payload["masks"]}
    staff_records = {str(item["symbol_id"]): item for item in staff_payload["masks"]}
    missing = [symbol_id for symbol_id in symbol_ids if symbol_id not in box_records or symbol_id not in staff_records]
    if missing:
        raise KeyError(f"selected symbols missing from one or both manifests: {missing}")

    with Image.open(image_path) as source:
        image = np.asarray(source.convert("RGB"))
    selected_staff = [staff_records[symbol_id] for symbol_id in symbol_ids]
    crop = _parse_crop(crop_value, selected_staff, image.shape[:2])

    box_union = np.zeros(image.shape[:2], dtype=bool)
    staff_union = np.zeros(image.shape[:2], dtype=bool)
    per_symbol: dict[str, Any] = {}
    loaded: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for symbol_id in symbol_ids:
        box_mask = _load_mask(box_records[symbol_id], box_manifest)
        staff_mask = _load_mask(staff_records[symbol_id], staff_manifest)
        if box_mask.shape != image.shape[:2] or staff_mask.shape != image.shape[:2]:
            raise ValueError(f"mask shape for {symbol_id} does not match the source image")
        loaded[symbol_id] = (box_mask, staff_mask)
        box_union |= box_mask
        staff_union |= staff_mask
        prompt = staff_records[symbol_id].get("prompt", {})
        per_symbol[symbol_id] = {
            "class": staff_records[symbol_id].get("class"),
            **binary_mask_metrics(box_mask, staff_mask),
            "box_only_score": box_records[symbol_id].get("mask_score"),
            "staff_aware_score": staff_records[symbol_id].get("mask_score"),
            "adaptive_box_rule": prompt.get("adaptive_box_rule"),
            "positive_points": len(prompt.get("positive_points", [])),
            "negative_points": len(prompt.get("negative_points", [])),
        }

    fig, axes = plt.subplots(1, 4, figsize=(9.2, 2.55))
    titles = ("(a) Box-only mask", "(b) Staff-aware prompts", "(c) Staff-aware mask", "(d) Pixel difference")
    for ax, title in zip(axes, titles):
        _show_crop(ax, image, crop)
        ax.set_title(title, fontsize=9, pad=4)

    for symbol_id in symbol_ids:
        record = staff_records[symbol_id]
        cls = str(record.get("class") or "")
        color = CLASS_COLORS.get(cls, "#4D4D4D")
        box_mask, staff_mask = loaded[symbol_id]
        axes[0].imshow(_rgba(box_mask, color, 0.58))
        axes[2].imshow(_rgba(staff_mask, color, 0.58))

        prompt = record.get("prompt", {})
        detector_box = prompt.get("detector_box") or record["bbox"]
        adaptive_box = prompt.get("adaptive_box") or record["bbox"]
        for box, linestyle, linewidth in ((detector_box, ":", 1.0), (adaptive_box, "--", 1.5)):
            bx0, by0, bx1, by1 = (float(value) for value in box)
            axes[1].add_patch(
                Rectangle((bx0, by0), bx1 - bx0, by1 - by0, fill=False, edgecolor=color, linestyle=linestyle, linewidth=linewidth)
            )
        positive = np.asarray(prompt.get("positive_points", []), dtype=float)
        negative = np.asarray(prompt.get("negative_points", []), dtype=float)
        if positive.size:
            axes[1].scatter(positive[:, 0], positive[:, 1], s=24, marker="o", color="#00A651", edgecolor="white", linewidth=0.45, zorder=5)
        if negative.size:
            axes[1].scatter(negative[:, 0], negative[:, 1], s=30, marker="x", color="#D62728", linewidth=1.4, zorder=6)

    overlap = np.logical_and(box_union, staff_union)
    box_only_pixels = np.logical_and(box_union, ~staff_union)
    staff_only_pixels = np.logical_and(staff_union, ~box_union)
    axes[3].imshow(_rgba(overlap, "#7B61A8", 0.50))
    axes[3].imshow(_rgba(box_only_pixels, "#00BFC4", 0.82))
    axes[3].imshow(_rgba(staff_only_pixels, "#E7298A", 0.82))
    for ax in axes:
        ax.set_xlim(crop[0], crop[2])
        ax.set_ylim(crop[3], crop[1])

    overall = binary_mask_metrics(box_union, staff_union)
    handles = [
        Line2D([], [], color="#4D4D4D", linestyle=":", label="detector box"),
        Line2D([], [], color="#4D4D4D", linestyle="--", label="adaptive box"),
        Line2D([], [], marker="o", linestyle="", markerfacecolor="#00A651", markeredgecolor="white", label="positive point"),
        Line2D([], [], marker="x", linestyle="", color="#D62728", label="staff-line negative"),
        Patch(facecolor="#00BFC4", label="box-only only"),
        Patch(facecolor="#E7298A", label="staff-aware only"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False, fontsize=7.2, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(
        f"Same detections, different SAM2 prompts | union IoU={overall['iou']:.3f}, changed pixels={overall['changed_pixels']}",
        fontsize=9.5,
        y=0.99,
    )
    fig.subplots_adjust(left=0.01, right=0.995, top=0.84, bottom=0.20, wspace=0.025)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)

    report = {
        "image": str(image_path),
        "box_only_manifest": str(box_manifest),
        "staff_aware_manifest": str(staff_manifest),
        "symbol_ids": symbol_ids,
        "crop": list(crop),
        "overall": overall,
        "per_symbol": per_symbol,
        "output": str(output),
    }
    report_path = output.with_suffix(".json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize box-only versus staff-aware SAM2 prompts and masks.")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--box-manifest", type=Path, required=True)
    parser.add_argument("--staff-manifest", type=Path, required=True)
    parser.add_argument("--symbol-ids", required=True, help="Comma-separated symbol ids present in both manifests.")
    parser.add_argument("--crop", help="Optional x0,y0,x1,y1 crop; otherwise inferred from staff-aware crops.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    symbol_ids = [item.strip() for item in args.symbol_ids.split(",") if item.strip()]
    if not symbol_ids:
        parser.error("--symbol-ids must contain at least one id")
    report = build_comparison(
        args.image,
        args.box_manifest,
        args.staff_manifest,
        symbol_ids,
        args.output,
        args.crop,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
