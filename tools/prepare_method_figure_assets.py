#!/usr/bin/env python3
"""Prepare traceable raster assets for the StaffOMR-SAM method figure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / "outputs" / "ijcv_dwd_dataset_runs" / (
    "fp_grandstaff_test0000_v2_1_sam2_neural_clef_roi_rl_balance_test136"
)

SYSTEM_BOX = (30, 40, 2070, 500)
SCORE_CROP_BOX = (715, 38, 1579, 511)
TARGET_SYMBOL_ID = "deim_000522"
ASSEMBLED_BEAM_ID = "deim_000006"
REPRESENTATIVE_CANDIDATE_IDS = (
    "deim_000349",  # filled notehead
    "deim_000003",  # open notehead
    "deim_000006",  # beam
    "deim_000008",  # open notehead
    "deim_000005",  # open notehead
    TARGET_SYMBOL_ID,  # selected filled notehead
)

BOX_COLOR = (178, 119, 61, 255)
BOX_COLOR_LIGHT = (196, 145, 92, 190)
STAFF_COLOR = (75, 128, 105, 230)
MASK_COLOR = (61, 102, 171, 155)
POSITIVE_COLOR = (45, 117, 182, 255)
NEGATIVE_COLOR = (163, 62, 62, 255)
LABEL_FONT = Path("C:/Windows/Fonts/times.ttf")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs" / "paper_method_figure_assets" / "grandstaff_test0000",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def center(symbol: dict) -> tuple[float, float]:
    x0, y0, x1, y1 = symbol["bbox"]
    return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)


def inside(box: tuple[int, int, int, int], symbol: dict) -> bool:
    x, y = center(symbol)
    return box[0] <= x <= box[2] and box[1] <= y <= box[3]


def select_representative_candidates(symbols: list[dict]) -> list[dict]:
    by_id = {item["id"]: item for item in symbols}
    selected = [by_id[item_id] for item_id in REPRESENTATIVE_CANDIDATE_IDS]
    if any(not inside(SCORE_CROP_BOX, item) for item in selected):
        raise ValueError("A representative candidate lies outside SCORE_CROP_BOX")
    return sorted(selected, key=lambda item: (center(item)[1], center(item)[0]))


def draw_candidate_overlay(
    image: Image.Image, symbols: list[dict], target_symbol_id: str
) -> Image.Image:
    crop = image.crop(SCORE_CROP_BOX).convert("RGBA")
    draw = ImageDraw.Draw(crop)
    font = ImageFont.truetype(str(LABEL_FONT), 20) if LABEL_FONT.exists() else None
    ox, oy = SCORE_CROP_BOX[:2]
    for symbol in symbols:
        x0, y0, x1, y1 = symbol["bbox"]
        if symbol["class"].endswith("notehead"):
            x0, y0, x1, y1 = x0 - 4, y0 - 4, x1 + 4, y1 + 4
        local = (x0 - ox, y0 - oy, x1 - ox, y1 - oy)
        is_target = symbol["id"] == target_symbol_id
        draw.rectangle(
            local,
            outline=BOX_COLOR if is_target else BOX_COLOR_LIGHT,
            width=4 if is_target else 2,
        )
        if is_target:
            label_y = max(1, local[1] - 22)
            draw.text((local[0] + 2, label_y), "b_i", fill=BOX_COLOR, font=font)
    return crop.convert("RGB")


def draw_staff_geometry(image: Image.Image, shapes: dict) -> Image.Image:
    crop = image.crop(SYSTEM_BOX).convert("RGBA")
    draw = ImageDraw.Draw(crop)
    ox, oy = SYSTEM_BOX[:2]
    for staff in shapes["staves"]:
        if not any(SYSTEM_BOX[1] <= y <= SYSTEM_BOX[3] for y in staff["lines"]):
            continue
        x0 = max(staff["x0"], SYSTEM_BOX[0]) - ox
        x1 = min(staff["x1"], SYSTEM_BOX[2]) - ox
        for y in staff["lines"]:
            draw.line((x0, y - oy, x1, y - oy), fill=STAFF_COLOR, width=3)
    return crop.convert("RGB")


def expanded_box(bbox: list[int], width: int, height: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = bbox
    return (
        max(0, x0 - 65),
        max(0, y0 - 55),
        min(width, x1 + 65),
        min(height, y1 + 55),
    )


def mask_overlay(image_crop: Image.Image, mask_crop: Image.Image) -> Image.Image:
    base = image_crop.convert("RGBA")
    mask = mask_crop.convert("L")
    tint = Image.new("RGBA", base.size, MASK_COLOR)
    alpha = mask.point(lambda value: MASK_COLOR[3] if value > 0 else 0)
    tint.putalpha(alpha)
    return Image.alpha_composite(base, tint).convert("RGB")


def draw_prompt_crop(
    image_crop: Image.Image,
    target_bbox: list[int],
    crop_box: tuple[int, int, int, int],
    staff_lines: list[float],
) -> Image.Image:
    result = image_crop.convert("RGBA")
    draw = ImageDraw.Draw(result)
    ox, oy = crop_box[:2]
    x0, y0, x1, y1 = target_bbox
    local_bbox = (x0 - ox, y0 - oy, x1 - ox, y1 - oy)
    draw.rectangle(local_bbox, outline=BOX_COLOR, width=3)

    px = int((x0 + x1) / 2 - ox)
    py = int((y0 + y1) / 2 - oy)
    radius = 5
    draw.ellipse((px - radius, py - radius, px + radius, py + radius), fill=POSITIVE_COLOR)

    for staff_y in staff_lines:
        if crop_box[1] <= staff_y <= crop_box[3]:
            nx = min(result.width - 8, int(x1 - ox + 22))
            ny = int(staff_y - oy)
            draw.line((nx - 5, ny - 5, nx + 5, ny + 5), fill=NEGATIVE_COLOR, width=2)
            draw.line((nx - 5, ny + 5, nx + 5, ny - 5), fill=NEGATIVE_COLOR, width=2)
    return result.convert("RGB")


def build_assembled_beam_group(
    image: Image.Image,
    pruned_shapes: dict,
    mask_records: dict[str, dict],
) -> tuple[Image.Image, Image.Image, Image.Image, dict]:
    symbols = {item["id"]: item for item in pruned_shapes["symbols"]}
    relation = next(
        item
        for item in pruned_shapes["relations"]
        if item["type"] == "beam_stem_group" and item["source"] == ASSEMBLED_BEAM_ID
    )

    beam = symbols[ASSEMBLED_BEAM_ID]
    stems = [symbols[item_id] for item_id in relation["targets"]]
    stem_ids = {item["id"] for item in stems}
    attachment_relations = [
        item
        for item in pruned_shapes["relations"]
        if item["type"] == "notehead_stem_attachment"
        and item["score"] >= 0.5
        and any(target in stem_ids for target in item["targets"])
    ]
    notehead_ids = sorted({item["source"] for item in attachment_relations})
    noteheads = [symbols[item_id] for item_id in notehead_ids]

    composite = Image.new("L", image.size, 0)
    sam_member_ids = [ASSEMBLED_BEAM_ID, *notehead_ids]
    for member_id in sam_member_ids:
        record = mask_records[member_id]
        member_mask = Image.open(ROOT / record["mask_path"]).convert("L")
        composite = ImageChops.lighter(composite, member_mask)

    stem_layer = Image.new("L", image.size, 0)
    stem_draw = ImageDraw.Draw(stem_layer)
    for stem in stems:
        x0, y0, x1, y1 = map(int, stem["bbox"])
        stem_draw.rectangle((x0, y0, max(x0 + 2, x1), y1), fill=255)
    composite = ImageChops.lighter(composite, stem_layer)

    mask_box = composite.getbbox()
    if mask_box is None:
        raise ValueError(f"Empty assembled mask for {ASSEMBLED_BEAM_ID}")
    x0, y0, x1, y1 = mask_box
    crop_box = (
        max(0, x0 - 35),
        max(0, y0 - 8),
        min(image.width, x1 + 35),
        min(image.height, y1 + 20),
    )
    image_crop = image.crop(crop_box)
    mask_crop = composite.crop(crop_box)
    overlay = mask_overlay(image_crop, mask_crop)

    metadata = {
        "type": "relation_assembled_note_group_mask",
        "beam_id": ASSEMBLED_BEAM_ID,
        "relation_id": relation["id"],
        "relation_type": relation["type"],
        "relation_score": relation["score"],
        "beam_mask_source": beam.get("mask_source"),
        "beam_mask_score": beam.get("mask_score"),
        "stem_ids": [item["id"] for item in stems],
        "stem_source": "v2_1_geometry_synthetic",
        "notehead_ids": notehead_ids,
        "notehead_attachment_threshold": 0.5,
        "notehead_attachment_relations": [
            {
                "id": item["id"],
                "source": item["source"],
                "targets": item["targets"],
                "score": item["score"],
            }
            for item in attachment_relations
        ],
        "notehead_mask_sources": {
            item_id: mask_records[item_id]["source"] for item_id in notehead_ids
        },
        "crop_xyxy": crop_box,
        "warning": (
            "Derived union of primitive SAM2 masks and relation-supported synthetic stems; "
            "not a single SAM2 group-mask prediction."
        ),
    }
    return image_crop, mask_crop, overlay, metadata


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    predictions_path = run_dir / "deim" / "predictions_with_clef_roi.json"
    shapes_path = run_dir / "symbols" / "symbols_v2_1_shapes.json"
    masks_path = run_dir / "sam2" / "masks_all.json"
    pruned_shapes_path = run_dir / "symbols" / "symbols_v2_1_shapes_pruned_balanced.json"

    predictions = load_json(predictions_path)
    shapes = load_json(shapes_path)
    masks = load_json(masks_path)
    pruned_shapes = load_json(pruned_shapes_path)
    input_path = (ROOT / predictions["input"]).resolve()
    image = Image.open(input_path).convert("RGB")

    target = next(item for item in predictions["symbols"] if item["id"] == TARGET_SYMBOL_ID)
    selected = select_representative_candidates(predictions["symbols"])
    target_mask = next(item for item in masks["masks"] if item["symbol_id"] == TARGET_SYMBOL_ID)
    mask_records = {item["symbol_id"]: item for item in masks["masks"]}
    mask_path = (ROOT / target_mask["mask_path"]).resolve()
    mask = Image.open(mask_path).convert("L")
    if mask.size != image.size:
        raise ValueError(f"Expected a full-page SAM mask, got {mask.size}; image is {image.size}")

    image.save(out_dir / "01_score_image_I_full.png")
    image.crop(SYSTEM_BOX).save(out_dir / "02_score_image_I_system.png")
    image.crop(SCORE_CROP_BOX).save(out_dir / "03_score_crop_clean.png")
    draw_candidate_overlay(image, selected, target["id"]).save(
        out_dir / "04_score_crop_selected_candidates.png"
    )
    draw_staff_geometry(image, shapes).save(out_dir / "05_staff_geometry_estimation.png")

    target_crop_box = expanded_box(target["bbox"], image.width, image.height)
    target_image_crop = image.crop(target_crop_box)
    target_mask_crop = mask.crop(target_crop_box)
    target_image_crop.save(out_dir / "06_sam_image_crop_li.png")
    target_mask_crop.save(out_dir / "07_sam_mask_binary.png")
    mask_overlay(target_image_crop, target_mask_crop).save(out_dir / "08_sam_mask_overlay_Mi.png")

    staff = min(
        shapes["staves"],
        key=lambda item: abs(sum(item["lines"]) / len(item["lines"]) - center(target)[1]),
    )
    draw_prompt_crop(target_image_crop, target["bbox"], target_crop_box, staff["lines"]).save(
        out_dir / "09_staff_aware_prompt_design_only.png"
    )

    group_crop, group_mask, group_overlay, group_metadata = build_assembled_beam_group(
        image, pruned_shapes, mask_records
    )
    group_crop.save(out_dir / "10_assembled_note_group_crop.png")
    group_mask.save(out_dir / "11_assembled_note_group_mask.png")
    group_overlay.save(out_dir / "12_assembled_note_group_overlay.png")

    manifest = {
        "selection_rule": (
            "Representative successful case with complete traceable intermediate outputs. "
            "Candidate overlay shows six selected primitive-symbol detections for clarity, "
            "not the complete detector output."
        ),
        "detector_output_unit": "primitive_symbol",
        "candidate_visualization_scope": "selected_representative_subset",
        "recommended_panel_title": "Selected symbol candidates",
        "recommended_caption_note": (
            "For clarity, only representative primitive-symbol detections are visualized."
        ),
        "user_reference_crop_registration": {
            "method": "SIFT feature matching with RANSAC homography",
            "good_matches": 83,
            "inliers": 50,
            "mapped_crop_xyxy_rounded": SCORE_CROP_BOX,
        },
        "dataset_sample": "GrandStaff full-page public test sample test_0000",
        "input_image": str(input_path),
        "run_directory": str(run_dir),
        "detector_predictions": str(predictions_path.resolve()),
        "shape_and_staff_geometry": str(shapes_path.resolve()),
        "pruned_relation_graph": str(pruned_shapes_path.resolve()),
        "sam2_metadata": str(masks_path.resolve()),
        "sam2_checkpoint": masks.get("checkpoint"),
        "system_crop_xyxy": SYSTEM_BOX,
        "score_crop_xyxy": SCORE_CROP_BOX,
        "selected_candidate_count": len(selected),
        "selected_candidates": [
            {
                "id": item["id"],
                "class": item["class"],
                "confidence": item["confidence"],
                "bbox_xyxy": item["bbox"],
            }
            for item in selected
        ],
        "sam_target": {
            "id": target["id"],
            "class": target["class"],
            "confidence": target["confidence"],
            "bbox_xyxy": target["bbox"],
            "crop_xyxy": target_crop_box,
            "mask_path": str(mask_path),
            "mask_score": target_mask["mask_score"],
            "prompt_strategy": target_mask["prompt_strategy"],
        },
        "design_only_asset": {
            "path": "09_staff_aware_prompt_design_only.png",
            "warning": (
                "Constructed method-design visualization. The persisted SAM2 mask for this "
                "sample used box_only, so the added positive/negative points must not be "
                "presented as the executed prompt that produced the mask."
            ),
        },
        "assembled_note_group": group_metadata,
    }
    with (out_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=True)

    print(json.dumps({"out_dir": str(out_dir), "assets": 12, "candidates": len(selected)}))


if __name__ == "__main__":
    main()
