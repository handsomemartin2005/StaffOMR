from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from omr_v2_common import count_by_class, read_json, write_json


NOTEHEAD_CLASSES = {"filled_notehead", "open_notehead"}
ACCIDENTAL_CLASSES = {"sharp", "flat", "natural"}
SKELETON_CLASSES = {"stem", "beam", "ledger_line", "barline", "slur_or_tie"}
RELATION_CLASSES = NOTEHEAD_CLASSES | {"stem", "beam", "ledger_line", "slur_or_tie"}


def clamp_bbox(bbox: list[float], width: int, height: int) -> list[int]:
    x0, y0, x1, y1 = bbox
    return [
        max(0, min(width - 1, int(math.floor(x0)))),
        max(0, min(height - 1, int(math.floor(y0)))),
        max(0, min(width, int(math.ceil(x1)))),
        max(0, min(height, int(math.ceil(y1)))),
    ]


def expand_bbox(bbox: list[int], amount_x: float, amount_y: float, width: int, height: int) -> list[int]:
    x0, y0, x1, y1 = bbox
    return clamp_bbox([x0 - amount_x, y0 - amount_y, x1 + amount_x, y1 + amount_y], width, height)


def prompt_box_for_symbol(symbol: dict[str, Any], width: int, height: int) -> tuple[list[int], list[str]]:
    bbox = [int(round(v)) for v in symbol["bbox"]]
    attrs = symbol.get("attributes") or {}
    space = float(attrs.get("staff_space") or max(8.0, 0.5 * ((bbox[2] - bbox[0]) + (bbox[3] - bbox[1]))))
    cls = symbol["class"]
    decisions: list[str] = []
    if cls in NOTEHEAD_CLASSES:
        prompt = expand_bbox(bbox, 0.18 * (bbox[2] - bbox[0]), 0.18 * (bbox[3] - bbox[1]), width, height)
        decisions.append("prompt_box_expanded_for_notehead")
    elif cls == "stem":
        prompt = expand_bbox(bbox, 0.25 * space, 0.08 * space, width, height)
        decisions.append("prompt_box_narrow_vertical_stem")
    elif cls == "beam":
        prompt = expand_bbox(bbox, 0.20 * space, 0.30 * space, width, height)
        decisions.append("prompt_box_expanded_for_beam_bar")
    elif cls == "ledger_line":
        prompt = expand_bbox(bbox, 0.30 * space, 0.20 * space, width, height)
        decisions.append("prompt_box_short_horizontal_ledger")
    elif cls == "barline":
        prompt = expand_bbox(bbox, 0.20 * space, 0.10 * space, width, height)
        decisions.append("prompt_box_tall_vertical_barline")
    elif cls == "slur_or_tie":
        prompt = expand_bbox(bbox, 0.35 * space, 0.35 * space, width, height)
        decisions.append("prompt_box_expanded_for_arc")
    elif cls in ACCIDENTAL_CLASSES or cls in {"rest", "treble_clef", "bass_clef"}:
        prompt = expand_bbox(bbox, 0.12 * (bbox[2] - bbox[0]), 0.12 * (bbox[3] - bbox[1]), width, height)
        decisions.append("prompt_box_slightly_expanded_for_symbol_outline")
    else:
        prompt = expand_bbox(bbox, 0.08 * space, 0.08 * space, width, height)
        decisions.append("prompt_box_default")
    return prompt, decisions


def resolve_existing_path(value: str | None, anchors: list[Path]) -> Path | None:
    if not value:
        return None
    raw = Path(value)
    candidates = [raw]
    if not raw.is_absolute():
        candidates.extend(anchor / raw for anchor in anchors)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def load_external_masks(mask_json: Path | None) -> dict[str, dict[str, Any]]:
    if mask_json is None:
        return {}
    payload = read_json(mask_json)
    masks = payload.get("masks", payload if isinstance(payload, list) else [])
    return {str(item["symbol_id"]): item for item in masks if item.get("symbol_id") is not None}


def bbox_mask(symbol: dict[str, Any], image_size: tuple[int, int]) -> np.ndarray:
    width, height = image_size
    mask = np.zeros((height, width), dtype=bool)
    x0, y0, x1, y1 = [int(round(v)) for v in symbol["bbox"]]
    mask[max(0, y0) : min(height, y1), max(0, x0) : min(width, x1)] = True
    return mask


def load_symbol_mask(
    symbol: dict[str, Any],
    image_size: tuple[int, int],
    external_masks: dict[str, dict[str, Any]],
    anchors: list[Path],
) -> tuple[np.ndarray, dict[str, Any], list[str]]:
    width, height = image_size
    mask_info = dict(symbol.get("mask") or external_masks.get(str(symbol["id"])) or {})
    decisions: list[str] = []
    mask_path = resolve_existing_path(mask_info.get("mask_path"), anchors)
    if mask_path is None:
        decisions.append("mask_missing_bbox_fallback_used")
        return bbox_mask(symbol, image_size), {"source": "bbox_fallback", "mask_path": None}, decisions

    image = Image.open(mask_path).convert("L")
    if image.size != (width, height):
        image = image.resize((width, height), Image.Resampling.NEAREST)
        decisions.append("mask_resized_to_input_size")
    mask = np.array(image) > 0
    mask_info["mask_path"] = str(mask_path)
    mask_info.setdefault("source", "sam2_box_prompt")
    decisions.append("mask_loaded")
    return mask, mask_info, decisions


def crop_to_prompt(mask: np.ndarray, prompt_box: list[int]) -> np.ndarray:
    constrained = np.zeros_like(mask, dtype=bool)
    x0, y0, x1, y1 = prompt_box
    constrained[y0:y1, x0:x1] = mask[y0:y1, x0:x1]
    return constrained


def component_filter(mask: np.ndarray, prompt_box: list[int], cls: str, staff_space: float) -> tuple[np.ndarray, list[str]]:
    if not mask.any():
        return mask, ["empty_mask_after_prompt_crop"]
    min_area = max(4, int(0.015 * staff_space * staff_space))
    if cls in {"stem", "ledger_line", "barline"}:
        min_area = max(3, int(0.010 * staff_space * staff_space))
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    keep = np.zeros_like(mask, dtype=bool)
    px0, py0, px1, py1 = prompt_box
    kept = 0
    for label in range(1, num):
        x, y, w, h, area = [int(v) for v in stats[label]]
        if area < min_area:
            continue
        if x + w < px0 or x > px1 or y + h < py0 or y > py1:
            continue
        keep |= labels == label
        kept += 1
    decisions = [f"connected_components_kept_{kept}"]
    if kept == 0:
        decisions.append("component_filter_empty_reverted_to_prompt_mask")
        return mask, decisions
    return keep, decisions


def mask_bbox(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)]


def contour_polygon(mask: np.ndarray, max_points: int) -> dict[str, Any] | None:
    bbox = mask_bbox(mask)
    if bbox is None:
        return None
    x0, y0, x1, y1 = bbox
    crop = mask[y0:y1, x0:x1].astype(np.uint8)
    contours, _ = cv2.findContours(crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    perimeter = cv2.arcLength(contour, True)
    epsilon = max(1.0, 0.006 * perimeter)
    approx = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
    points = [[int(px + x0), int(py + y0)] for px, py in approx]
    if len(points) > max_points:
        step = int(math.ceil(len(points) / max_points))
        points = points[::step]
    return {"type": "polygon", "points": points}


def zhang_suen_thinning(mask: np.ndarray, max_iter: int = 128) -> np.ndarray:
    image = (mask > 0).astype(np.uint8)
    if image.size == 0 or not image.any():
        return image.astype(bool)
    changed = True
    iteration = 0
    while changed and iteration < max_iter:
        changed = False
        iteration += 1
        for subiter in (0, 1):
            padded = np.pad(image, 1, mode="constant")
            p2 = padded[:-2, 1:-1]
            p3 = padded[:-2, 2:]
            p4 = padded[1:-1, 2:]
            p5 = padded[2:, 2:]
            p6 = padded[2:, 1:-1]
            p7 = padded[2:, :-2]
            p8 = padded[1:-1, :-2]
            p9 = padded[:-2, :-2]
            neighbors = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
            transitions = (
                ((p2 == 0) & (p3 == 1)).astype(np.uint8)
                + ((p3 == 0) & (p4 == 1)).astype(np.uint8)
                + ((p4 == 0) & (p5 == 1)).astype(np.uint8)
                + ((p5 == 0) & (p6 == 1)).astype(np.uint8)
                + ((p6 == 0) & (p7 == 1)).astype(np.uint8)
                + ((p7 == 0) & (p8 == 1)).astype(np.uint8)
                + ((p8 == 0) & (p9 == 1)).astype(np.uint8)
                + ((p9 == 0) & (p2 == 1)).astype(np.uint8)
            )
            if subiter == 0:
                condition = (p2 * p4 * p6 == 0) & (p4 * p6 * p8 == 0)
            else:
                condition = (p2 * p4 * p8 == 0) & (p2 * p6 * p8 == 0)
            remove = (image == 1) & (neighbors >= 2) & (neighbors <= 6) & (transitions == 1) & condition
            if remove.any():
                image[remove] = 0
                changed = True
    return image.astype(bool)


def pca_angle_and_lengths(points: np.ndarray) -> tuple[float | None, float, float]:
    if len(points) < 2:
        return None, 0.0, 0.0
    centered = points - points.mean(axis=0)
    cov = np.cov(centered.T)
    values, vectors = np.linalg.eigh(cov)
    order = np.argsort(values)[::-1]
    values = values[order]
    vectors = vectors[:, order]
    main = vectors[:, 0]
    angle = math.degrees(math.atan2(float(main[1]), float(main[0])))
    major = float(4.0 * math.sqrt(max(values[0], 0.0)))
    minor = float(4.0 * math.sqrt(max(values[1], 0.0))) if len(values) > 1 else 0.0
    return angle, major, minor


def fallback_centerline(symbol: dict[str, Any]) -> dict[str, Any] | None:
    x0, y0, x1, y1 = [int(round(v)) for v in symbol["bbox"]]
    cx = int(round(0.5 * (x0 + x1)))
    cy = int(round(0.5 * (y0 + y1)))
    cls = symbol["class"]
    if cls in {"stem", "barline"}:
        points = [[cx, y0], [cx, y1]]
        orientation = "vertical"
    elif cls in {"beam", "ledger_line"}:
        points = [[x0, cy], [x1, cy]]
        orientation = "horizontal_or_oblique"
    elif cls == "slur_or_tie":
        points = [[x0, cy], [cx, y1], [x1, cy]]
        orientation = "curved"
    else:
        return None
    length = polyline_length(points)
    return {
        "type": "polyline",
        "points": points,
        "length": length,
        "orientation": orientation,
        "source": "bbox_centerline_fallback",
    }


def polyline_length(points: list[list[int]]) -> float:
    if len(points) < 2:
        return 0.0
    total = 0.0
    for a, b in zip(points, points[1:]):
        total += math.hypot(b[0] - a[0], b[1] - a[1])
    return float(total)


def skeleton_polyline(mask: np.ndarray, symbol: dict[str, Any], max_points: int) -> dict[str, Any] | None:
    if symbol["class"] not in SKELETON_CLASSES:
        return None
    bbox = mask_bbox(mask)
    if bbox is None:
        return fallback_centerline(symbol)
    x0, y0, x1, y1 = bbox
    crop = mask[y0:y1, x0:x1]
    if crop.size > 180_000:
        return fallback_centerline(symbol)
    skeleton = zhang_suen_thinning(crop)
    ys, xs = np.where(skeleton)
    if len(xs) < 2:
        return fallback_centerline(symbol)
    points = np.stack([xs.astype(float) + x0, ys.astype(float) + y0], axis=1)
    angle, _, _ = pca_angle_and_lengths(points)
    if len(points) > max_points:
        centered = points - points.mean(axis=0)
        if angle is None:
            order = np.argsort(points[:, 0] + 0.01 * points[:, 1])
        else:
            rad = math.radians(angle)
            axis = np.array([math.cos(rad), math.sin(rad)])
            order = np.argsort(centered @ axis)
        points = points[order]
        idxs = np.linspace(0, len(points) - 1, max_points).round().astype(int)
        points = points[idxs]
    else:
        order = np.argsort(points[:, 0] + 0.01 * points[:, 1])
        points = points[order]
    int_points = [[int(round(px)), int(round(py))] for px, py in points]
    orientation = classify_orientation(angle, symbol["class"])
    return {
        "type": "polyline",
        "points": int_points,
        "length": polyline_length(int_points),
        "orientation": orientation,
        "angle_degrees": angle,
        "source": "mask_skeleton",
    }


def classify_orientation(angle: float | None, cls: str) -> str:
    if cls == "slur_or_tie":
        return "curved_candidate"
    if angle is None:
        return "unknown"
    folded = abs(((angle + 90.0) % 180.0) - 90.0)
    vertical_error = abs(90.0 - folded)
    if vertical_error <= 20.0:
        return "vertical"
    if folded <= 20.0:
        return "horizontal"
    return "oblique"


def staff_lines_for_symbol(symbol: dict[str, Any], staves: list[dict[str, Any]]) -> tuple[list[float], float]:
    attrs = symbol.get("attributes") or {}
    staff_idx = attrs.get("staff")
    if staff_idx is not None:
        for staff in staves:
            if int(staff.get("index", -1)) == int(staff_idx):
                return [float(y) for y in staff.get("lines", [])], float(staff.get("space") or attrs.get("staff_space") or 1.0)
    return [], float(attrs.get("staff_space") or 1.0)


def shape_descriptor(
    mask: np.ndarray,
    symbol: dict[str, Any],
    staves: list[dict[str, Any]],
    prompt_box: list[int],
) -> dict[str, Any]:
    area = int(mask.sum())
    bbox = mask_bbox(mask)
    sx0, sy0, sx1, sy1 = [int(round(v)) for v in symbol["bbox"]]
    symbol_bbox_area = max(1, (sx1 - sx0) * (sy1 - sy0))
    px0, py0, px1, py1 = prompt_box
    prompt_area = max(1, (px1 - px0) * (py1 - py0))
    if bbox is None:
        return {
            "area": 0,
            "bbox_area": prompt_area,
            "prompt_area": prompt_area,
            "symbol_bbox_area": symbol_bbox_area,
            "fill_ratio": 0.0,
            "symbol_bbox_fill_ratio": 0.0,
            "aspect_ratio": 0.0,
            "centroid": None,
            "mask_bbox": None,
            "staff_overlap_pixels": 0,
            "staff_overlap_ratio": 0.0,
        }
    ys, xs = np.where(mask)
    points = np.stack([xs.astype(float), ys.astype(float)], axis=1)
    angle, major, minor = pca_angle_and_lengths(points)
    mx0, my0, mx1, my1 = bbox
    width = max(1, mx1 - mx0)
    height = max(1, my1 - my0)
    staff_lines, space = staff_lines_for_symbol(symbol, staves)
    staff_overlap = 0
    for y in staff_lines:
        yy0 = max(0, int(round(y - max(1.0, 0.08 * space))))
        yy1 = min(mask.shape[0], int(round(y + max(1.0, 0.08 * space))) + 1)
        staff_overlap += int(mask[yy0:yy1, :].sum())
    return {
        "area": area,
        "bbox_area": prompt_area,
        "prompt_area": prompt_area,
        "symbol_bbox_area": symbol_bbox_area,
        "fill_ratio": float(area / prompt_area),
        "symbol_bbox_fill_ratio": float(area / symbol_bbox_area),
        "aspect_ratio": float(width / height),
        "centroid": [float(xs.mean()), float(ys.mean())] if len(xs) else None,
        "mask_bbox": bbox,
        "orientation_angle_degrees": angle,
        "major_length": major,
        "minor_length": minor,
        "staff_overlap_pixels": staff_overlap,
        "staff_overlap_ratio": float(staff_overlap / max(1, area)),
    }


def geometry_check(symbol: dict[str, Any], shape: dict[str, Any], skeleton: dict[str, Any] | None) -> dict[str, Any]:
    cls = symbol["class"]
    attrs = symbol.get("attributes") or {}
    space = float(attrs.get("staff_space") or 1.0)
    failures: list[str] = []
    warnings: list[str] = []
    aspect = float(shape.get("aspect_ratio") or 0.0)
    fill_ratio = float(shape.get("fill_ratio") or 0.0)
    area = int(shape.get("area") or 0)
    if area <= 0:
        failures.append("empty_shape")
    if cls in NOTEHEAD_CLASSES:
        if not 0.45 <= aspect <= 2.4:
            warnings.append("notehead_aspect_outside_prior")
        if fill_ratio < 0.18:
            warnings.append("notehead_low_fill_ratio")
    elif cls == "stem":
        if aspect > 0.55:
            warnings.append("stem_not_thin_vertical")
        if skeleton and skeleton.get("orientation") != "vertical":
            warnings.append("stem_skeleton_not_vertical")
        if skeleton and float(skeleton.get("length") or 0.0) < 1.5 * space:
            warnings.append("stem_skeleton_short")
    elif cls == "beam":
        if aspect < 1.8:
            warnings.append("beam_not_elongated")
        if skeleton and skeleton.get("orientation") == "vertical":
            warnings.append("beam_skeleton_vertical")
    elif cls == "ledger_line":
        if aspect < 1.8:
            warnings.append("ledger_not_short_horizontal")
    elif cls == "barline":
        if aspect > 0.7:
            warnings.append("barline_not_vertical")
    elif cls == "slur_or_tie":
        if skeleton is None:
            warnings.append("slur_missing_skeleton")
    if failures:
        status = "fail"
    elif warnings:
        status = "warn"
    else:
        status = "pass"
    return {"status": status, "failures": failures, "warnings": warnings}


def point_to_bbox_distance(point: list[float], bbox: list[int]) -> float:
    x, y = point
    x0, y0, x1, y1 = bbox
    dx = max(x0 - x, 0.0, x - x1)
    dy = max(y0 - y, 0.0, y - y1)
    return math.hypot(dx, dy)


def bbox_intersects(a: list[int], b: list[int]) -> bool:
    return min(a[2], b[2]) >= max(a[0], b[0]) and min(a[3], b[3]) >= max(a[1], b[1])


def symbol_center(symbol: dict[str, Any]) -> list[float]:
    x0, y0, x1, y1 = [float(v) for v in symbol["bbox"]]
    return [0.5 * (x0 + x1), 0.5 * (y0 + y1)]


def symbol_staff(symbol: dict[str, Any]) -> int | None:
    staff = (symbol.get("attributes") or {}).get("staff")
    return int(staff) if staff is not None else None


def symbol_space(symbol: dict[str, Any]) -> float:
    return float((symbol.get("attributes") or {}).get("staff_space") or 12.0)


def skeleton_endpoints(symbol: dict[str, Any]) -> list[list[float]]:
    skeleton = symbol.get("skeleton")
    if skeleton and skeleton.get("points"):
        points = skeleton["points"]
        return [[float(v) for v in points[0]], [float(v) for v in points[-1]]]
    x0, y0, x1, y1 = [float(v) for v in symbol["bbox"]]
    cls = symbol["class"]
    if cls in {"stem", "barline"}:
        cx = 0.5 * (x0 + x1)
        return [[cx, y0], [cx, y1]]
    cy = 0.5 * (y0 + y1)
    return [[x0, cy], [x1, cy]]


def add_relation(
    relations: list[dict[str, Any]],
    symbols_by_id: dict[str, dict[str, Any]],
    relation_type: str,
    source: str,
    targets: list[str],
    score: float,
    evidence: dict[str, Any],
) -> None:
    relation_id = f"rel_{len(relations):06d}"
    item = {
        "id": relation_id,
        "type": relation_type,
        "source": source,
        "targets": targets,
        "score": float(max(0.0, min(1.0, score))),
        "evidence": evidence,
    }
    relations.append(item)
    for symbol_id in [source, *targets]:
        symbol = symbols_by_id.get(symbol_id)
        if symbol is not None:
            symbol.setdefault("relations", []).append(relation_id)


def build_relations(
    symbols: list[dict[str, Any]],
    *,
    note_stem_distance_scale: float = 1.35,
    note_stem_x_scale: float = 0.95,
    max_notes_per_stem: int = 6,
    beam_expand_x_scale: float = 0.35,
    beam_expand_y_scale: float = 0.55,
    beam_endpoint_scale: float = 0.45,
    ledger_y_scale: float = 1.65,
    ledger_distance_scale: float = 1.85,
    slur_endpoint_scale: float = 3.0,
) -> list[dict[str, Any]]:
    symbols_by_id = {str(item["id"]): item for item in symbols}
    noteheads = [s for s in symbols if s["class"] in NOTEHEAD_CLASSES]
    stems = [s for s in symbols if s["class"] == "stem"]
    beams = [s for s in symbols if s["class"] == "beam"]
    ledgers = [s for s in symbols if s["class"] == "ledger_line"]
    slurs = [s for s in symbols if s["class"] == "slur_or_tie"]
    relations: list[dict[str, Any]] = []

    for stem in stems:
        staff = symbol_staff(stem)
        space = symbol_space(stem)
        endpoints = skeleton_endpoints(stem)
        candidates: list[tuple[float, float, dict[str, Any], dict[str, Any]]] = []
        sx, _ = symbol_center(stem)
        sx0, sy0, sx1, sy1 = [float(v) for v in stem["bbox"]]
        for note in noteheads:
            if staff is not None and symbol_staff(note) != staff:
                continue
            dist = min(point_to_bbox_distance(endpoint, note["bbox"]) for endpoint in endpoints)
            nx, ny = symbol_center(note)
            x_dist = abs(sx - nx)
            vertical_cover = sy0 - 0.45 * space <= ny <= sy1 + 0.45 * space
            side_touch = (
                abs(float(note["bbox"][0]) - sx) <= note_stem_x_scale * space
                or abs(float(note["bbox"][2]) - sx) <= note_stem_x_scale * space
            )
            threshold = max(8.0, note_stem_distance_scale * space)
            if dist <= threshold or (vertical_cover and x_dist <= note_stem_x_scale * space) or (vertical_cover and side_touch):
                score_dist = min(dist, x_dist if vertical_cover else dist)
                evidence = {
                    "endpoint_distance": dist,
                    "x_distance": x_dist,
                    "vertical_cover": vertical_cover,
                    "side_touch": side_touch,
                    "threshold": threshold,
                }
                candidates.append((score_dist, x_dist, note, evidence))
        if not candidates:
            continue
        candidates.sort(key=lambda item: (item[0], item[1], symbol_center(item[2])[1]))
        attached_note_ids: set[str] = set()
        for dist, _x_dist, note, evidence in candidates[:max_notes_per_stem]:
            note_id = str(note["id"])
            if note_id in attached_note_ids:
                continue
            attached_note_ids.add(note_id)
            threshold = float(evidence["threshold"])
            score = 1.0 - min(1.0, dist / threshold)
            add_relation(
                relations,
                symbols_by_id,
                "notehead_stem_attachment",
                str(note["id"]),
                [str(stem["id"])],
                score,
                evidence,
            )

    for beam in beams:
        staff = symbol_staff(beam)
        space = symbol_space(beam)
        expanded = expand_bbox([int(v) for v in beam["bbox"]], beam_expand_x_scale * space, beam_expand_y_scale * space, 100_000, 100_000)
        attached: list[tuple[float, dict[str, Any]]] = []
        for stem in stems:
            if staff is not None and symbol_staff(stem) != staff:
                continue
            endpoints = skeleton_endpoints(stem)
            distances = [point_to_bbox_distance(endpoint, expanded) for endpoint in endpoints]
            dist = min(distances)
            if dist <= max(5.0, beam_endpoint_scale * space) or bbox_intersects(stem["bbox"], expanded):
                attached.append((dist, stem))
        if len(attached) >= 2:
            attached.sort(key=lambda item: symbol_center(item[1])[0])
            distances = [item[0] for item in attached]
            score = 1.0 - min(1.0, float(np.mean(distances)) / max(1.0, 0.65 * space))
            add_relation(
                relations,
                symbols_by_id,
                "beam_stem_group",
                str(beam["id"]),
                [str(stem["id"]) for _, stem in attached],
                score,
                {"attached_stems": len(attached), "mean_endpoint_distance": float(np.mean(distances))},
            )

    for ledger in ledgers:
        staff = symbol_staff(ledger)
        space = symbol_space(ledger)
        lx0, ly0, lx1, ly1 = [float(v) for v in ledger["bbox"]]
        lcx = 0.5 * (lx0 + lx1)
        lcy = 0.5 * (ly0 + ly1)
        candidates = []
        for note in noteheads:
            if staff is not None and symbol_staff(note) != staff:
                continue
            nx0, ny0, nx1, ny1 = [float(v) for v in note["bbox"]]
            ncx, ncy = 0.5 * (nx0 + nx1), 0.5 * (ny0 + ny1)
            x_overlap = min(lx1, nx1 + 0.75 * space) - max(lx0, nx0 - 0.75 * space)
            dist = math.hypot(lcx - ncx, lcy - ncy)
            if x_overlap >= 0 and abs(lcy - ncy) <= ledger_y_scale * space:
                candidates.append((dist, note))
        if not candidates:
            continue
        dist, note = min(candidates, key=lambda item: item[0])
        threshold = max(8.0, ledger_distance_scale * space)
        score = 1.0 - min(1.0, dist / threshold)
        add_relation(
            relations,
            symbols_by_id,
            "ledger_line_notehead_attachment",
            str(ledger["id"]),
            [str(note["id"])],
            score,
            {"center_distance": dist, "threshold": threshold},
        )

    for slur in slurs:
        staff = symbol_staff(slur)
        space = symbol_space(slur)
        endpoints = skeleton_endpoints(slur)
        targets: list[str] = []
        distances: list[float] = []
        for endpoint in endpoints:
            candidates = []
            for note in noteheads:
                if staff is not None and symbol_staff(note) != staff:
                    continue
                candidates.append((point_to_bbox_distance(endpoint, note["bbox"]), note))
            if candidates:
                dist, note = min(candidates, key=lambda item: item[0])
                if dist <= slur_endpoint_scale * space:
                    targets.append(str(note["id"]))
                    distances.append(dist)
        targets = list(dict.fromkeys(targets))
        if len(targets) >= 2:
            score = 1.0 - min(1.0, float(np.mean(distances)) / max(1.0, slur_endpoint_scale * space))
            add_relation(
                relations,
                symbols_by_id,
                "slur_tie_notehead_endpoints",
                str(slur["id"]),
                targets[:2],
                score,
                {"endpoint_distances": distances[:2]},
            )

    return relations


def enrich_symbol_shape(
    symbol: dict[str, Any],
    image_size: tuple[int, int],
    staves: list[dict[str, Any]],
    external_masks: dict[str, dict[str, Any]],
    anchors: list[Path],
    max_polygon_points: int,
    max_skeleton_points: int,
) -> dict[str, Any]:
    width, height = image_size
    item = dict(symbol)
    item["attributes"] = dict(symbol.get("attributes") or {})
    prompt_box, decisions = prompt_box_for_symbol(item, width, height)
    mask, mask_info, mask_decisions = load_symbol_mask(item, image_size, external_masks, anchors)
    decisions.extend(mask_decisions)
    mask = crop_to_prompt(mask, prompt_box)
    decisions.append("mask_constrained_to_prompt_box")
    staff_lines, space = staff_lines_for_symbol(item, staves)
    mask, component_decisions = component_filter(mask, prompt_box, item["class"], space)
    decisions.extend(component_decisions)
    polygon = contour_polygon(mask, max_polygon_points)
    skeleton = skeleton_polyline(mask, item, max_skeleton_points)
    shape = shape_descriptor(mask, item, staves, prompt_box)
    check = geometry_check(item, shape, skeleton)
    item["prompt_box"] = prompt_box
    item["mask"] = polygon
    item["mask_source"] = mask_info.get("source", "unknown")
    item["mask_score"] = mask_info.get("mask_score")
    item["mask_area"] = mask_info.get("mask_area", shape["area"])
    item["skeleton"] = skeleton
    item["shape"] = shape
    item["geometry_check"] = check["status"]
    item["geometry_check_details"] = check
    item["postprocess_decisions"] = decisions
    if staff_lines:
        item["shape"]["staff_line_overlap_policy"] = staff_overlap_policy(item["class"])
    return item


def staff_overlap_policy(cls: str) -> str:
    if cls in NOTEHEAD_CLASSES:
        return "allowed_inside_symbol"
    if cls == "ledger_line":
        return "allowed_as_symbol_candidate"
    if cls in {"stem", "slur_or_tie"}:
        return "treated_as_potential_interference"
    if cls == "beam":
        return "preserved_when_thick_or_stem_connected"
    return "class_specific_tolerance"


def bbox_iou(a: list[int], b: list[int]) -> float:
    inter_x0 = max(a[0], b[0])
    inter_y0 = max(a[1], b[1])
    inter_x1 = min(a[2], b[2])
    inter_y1 = min(a[3], b[3])
    inter = max(0, inter_x1 - inter_x0) * max(0, inter_y1 - inter_y0)
    if inter <= 0:
        return 0.0
    area_a = max(1, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(1, (b[2] - b[0]) * (b[3] - b[1]))
    return float(inter / (area_a + area_b - inter))


def bbox_overlap_min_ratio(a: list[int], b: list[int]) -> float:
    inter_x0 = max(a[0], b[0])
    inter_y0 = max(a[1], b[1])
    inter_x1 = min(a[2], b[2])
    inter_y1 = min(a[3], b[3])
    inter = max(0, inter_x1 - inter_x0) * max(0, inter_y1 - inter_y0)
    if inter <= 0:
        return 0.0
    area_a = max(1, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(1, (b[2] - b[0]) * (b[3] - b[1]))
    return float(inter / min(area_a, area_b))


def staff_lookup(staves: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    return {int(staff["index"]): staff for staff in staves if staff.get("index") is not None}


def staff_for_symbol(symbol: dict[str, Any], staves_by_index: dict[int, dict[str, Any]]) -> dict[str, Any] | None:
    staff = symbol_staff(symbol)
    if staff is None:
        return None
    return staves_by_index.get(staff)


def box_height(symbol: dict[str, Any]) -> float:
    x0, y0, x1, y1 = [float(v) for v in symbol["bbox"]]
    return max(0.0, y1 - y0)


def box_width(symbol: dict[str, Any]) -> float:
    x0, y0, x1, y1 = [float(v) for v in symbol["bbox"]]
    return max(0.0, x1 - x0)


def symbol_confidence(symbol: dict[str, Any]) -> float:
    return float(symbol.get("confidence") or 0.0)


def has_nearby_noteheads(
    symbol: dict[str, Any],
    noteheads: list[dict[str, Any]],
    min_count: int,
    vertical_spaces: float,
    horizontal_pad_spaces: float = 0.8,
) -> bool:
    staff = symbol_staff(symbol)
    space = symbol_space(symbol)
    x0, y0, x1, y1 = [float(v) for v in symbol["bbox"]]
    cy = 0.5 * (y0 + y1)
    count = 0
    for note in noteheads:
        if staff is not None and symbol_staff(note) != staff:
            continue
        nx, ny = symbol_center(note)
        if x0 - horizontal_pad_spaces * space <= nx <= x1 + horizontal_pad_spaces * space:
            if abs(ny - cy) <= vertical_spaces * space:
                count += 1
                if count >= min_count:
                    return True
    return False


def slur_has_note_endpoints(slur: dict[str, Any], noteheads: list[dict[str, Any]]) -> bool:
    staff = symbol_staff(slur)
    space = symbol_space(slur)
    endpoints = skeleton_endpoints(slur)
    matched: list[str] = []
    for endpoint in endpoints:
        candidates = []
        for note in noteheads:
            if staff is not None and symbol_staff(note) != staff:
                continue
            candidates.append((point_to_bbox_distance(endpoint, note["bbox"]), str(note["id"])))
        if not candidates:
            continue
        dist, note_id = min(candidates, key=lambda item: item[0])
        if dist <= 2.2 * space:
            matched.append(note_id)
    return len(set(matched)) >= 2


def should_drop_symbol(
    symbol: dict[str, Any],
    noteheads: list[dict[str, Any]],
    staves_by_index: dict[int, dict[str, Any]],
) -> str | None:
    cls = symbol["class"]
    attrs = symbol.get("attributes") or {}
    shape = symbol.get("shape") or {}
    space = symbol_space(symbol)
    normalized_height = box_height(symbol) / max(1.0, space)
    normalized_width = box_width(symbol) / max(1.0, space)
    staff_overlap = float(shape.get("staff_overlap_ratio") or 0.0)

    if cls in {"dynamic_crescendo_hairpin", "dynamic_diminuendo_hairpin"}:
        if staff_overlap >= 0.45 and normalized_height <= 0.30:
            return "dropped_hairpin_like_staff_line"
        if normalized_width >= 6.0 and normalized_height <= 0.18:
            return "dropped_overlong_thin_hairpin"

    if cls == "beam":
        if staff_overlap >= 0.55 and normalized_height <= 0.24:
            return "dropped_beam_like_staff_line"
        if normalized_width >= 8.0 and not has_nearby_noteheads(symbol, noteheads, 2, vertical_spaces=3.0):
            return "dropped_beam_without_nearby_noteheads"

    if cls == "slur_or_tie":
        if staff_overlap >= 0.20 and normalized_height <= 1.10 and not slur_has_note_endpoints(symbol, noteheads):
            return "dropped_slur_like_staff_or_beam_line"
        if normalized_width >= 8.5 and normalized_height <= 0.40:
            return "dropped_overlong_thin_slur"

    if cls in {"treble_clef", "bass_clef"}:
        staff = staff_for_symbol(symbol, staves_by_index)
        if staff is not None:
            cx, _ = symbol_center(symbol)
            left_band = float(staff.get("x0") or 0.0) + 4.2 * space
            if cx > left_band and symbol_confidence(symbol) < 0.90 and not str(symbol.get("source", "")).startswith("retained_rule"):
                return "dropped_midstaff_false_clef"

    if cls == "rest":
        box_symbol = [int(round(v)) for v in symbol["bbox"]]
        for note in noteheads:
            if symbol_staff(symbol) is not None and symbol_staff(symbol) != symbol_staff(note):
                continue
            if bbox_overlap_min_ratio(box_symbol, [int(round(v)) for v in note["bbox"]]) >= 0.35:
                return "dropped_rest_notehead_overlap"

    if attrs.get("detector_class") in {"timeSig0", "timeSig1", "timeSig2", "timeSig3", "timeSig4"}:
        staff = staff_for_symbol(symbol, staves_by_index)
        if staff is not None:
            cx, _ = symbol_center(symbol)
            left_band = float(staff.get("x0") or 0.0) + 9.0 * space
            if cx > left_band and symbol_confidence(symbol) < 0.60:
                return "dropped_midstaff_false_time_signature"

    return None


def dedupe_noteheads(symbols: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    noteheads = [symbol for symbol in symbols if symbol["class"] in NOTEHEAD_CLASSES]
    ordered = sorted(
        noteheads,
        key=lambda symbol: (
            symbol_staff(symbol) if symbol_staff(symbol) is not None else 9999,
            symbol_center(symbol)[0],
            symbol_center(symbol)[1],
            -symbol_confidence(symbol),
        ),
    )
    dropped_ids: set[str] = set()
    for idx, current in enumerate(ordered):
        if str(current["id"]) in dropped_ids:
            continue
        cbox = [int(round(v)) for v in current["bbox"]]
        for other in ordered[idx + 1 :]:
            if str(other["id"]) in dropped_ids:
                continue
            if symbol_staff(current) != symbol_staff(other):
                continue
            obox = [int(round(v)) for v in other["bbox"]]
            if bbox_overlap_min_ratio(cbox, obox) < 0.52:
                continue
            current_filled = current["class"] == "filled_notehead"
            other_filled = other["class"] == "filled_notehead"
            current_score = symbol_confidence(current) + (0.08 if current_filled else 0.0)
            other_score = symbol_confidence(other) + (0.08 if other_filled else 0.0)
            loser = other if current_score >= other_score else current
            dropped_ids.add(str(loser["id"]))
            if loser is current:
                break

    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for symbol in symbols:
        if str(symbol["id"]) in dropped_ids:
            item = dict(symbol)
            item["drop_reason"] = "dropped_duplicate_notehead"
            dropped.append(item)
        else:
            kept.append(symbol)
    return kept, dropped


def stem_exists_for_note(note: dict[str, Any], stems: list[dict[str, Any]]) -> bool:
    staff = symbol_staff(note)
    space = symbol_space(note)
    nx, ny = symbol_center(note)
    for stem in stems:
        if staff is not None and symbol_staff(stem) != staff:
            continue
        sx, _ = symbol_center(stem)
        sx0, sy0, sx1, sy1 = [float(v) for v in stem["bbox"]]
        if abs(sx - nx) <= 0.80 * space and sy0 - 0.4 * space <= ny <= sy1 + 0.4 * space:
            return True
        if min(point_to_bbox_distance(endpoint, note["bbox"]) for endpoint in skeleton_endpoints(stem)) <= 1.1 * space:
            return True
    return False


def beam_y_at_x(beam: dict[str, Any], x: float) -> float:
    points = skeleton_endpoints(beam)
    (x0, y0), (x1, y1) = points[0], points[-1]
    if abs(x1 - x0) <= 1e-6:
        return 0.5 * (y0 + y1)
    t = max(0.0, min(1.0, (x - x0) / (x1 - x0)))
    return y0 + t * (y1 - y0)


def nearest_beam_for_note(note: dict[str, Any], beams: list[dict[str, Any]]) -> dict[str, Any] | None:
    staff = symbol_staff(note)
    space = symbol_space(note)
    nx, ny = symbol_center(note)
    candidates: list[tuple[float, dict[str, Any]]] = []
    for beam in beams:
        if staff is not None and symbol_staff(beam) != staff:
            continue
        bx0, by0, bx1, by1 = [float(v) for v in beam["bbox"]]
        if not (bx0 - 0.85 * space <= nx <= bx1 + 0.85 * space):
            continue
        by = beam_y_at_x(beam, nx)
        vertical = abs(ny - by)
        if not (0.55 * space <= vertical <= 4.8 * space):
            continue
        horizontal_penalty = 0.0 if bx0 <= nx <= bx1 else min(abs(nx - bx0), abs(nx - bx1)) / max(1.0, space)
        candidates.append((vertical / max(1.0, space) + 0.45 * horizontal_penalty, beam))
    if not candidates:
        return None
    return min(candidates, key=lambda item: item[0])[1]


def pitch_step_for_direction(note: dict[str, Any]) -> int | None:
    attrs = note.get("attributes") or {}
    if attrs.get("pitch_step") is None:
        return None
    try:
        return int(round(float(attrs["pitch_step"])))
    except Exception:
        return None


def median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def synthetic_stem_symbol(
    index: int,
    note: dict[str, Any],
    beam: dict[str, Any] | None,
    image_size: tuple[int, int],
) -> dict[str, Any] | None:
    image_width, image_height = image_size
    space = symbol_space(note)
    nx0, ny0, nx1, ny1 = [float(v) for v in note["bbox"]]
    nx, ny = symbol_center(note)
    if beam is not None:
        beam_y = beam_y_at_x(beam, nx)
        direction = "up" if beam_y < ny else "down"
        stem_x = nx1 if direction == "up" else nx0
        y0, y1 = sorted([beam_y, ny])
        source = "synthetic_stem_from_notehead_beam"
        beam_id = str(beam["id"])
    else:
        if note["class"] != "filled_notehead":
            return None
        step = pitch_step_for_direction(note)
        direction = "down" if step is not None and step >= 4 else "up"
        stem_x = nx0 if direction == "down" else nx1
        stem_length = 3.35 * space
        if direction == "up":
            y0, y1 = ny - stem_length, ny
        else:
            y0, y1 = ny, ny + stem_length
        source = "synthetic_stem_from_filled_notehead"
        beam_id = None

    if y1 - y0 < 1.1 * space or y1 - y0 > 6.0 * space:
        return None
    half_width = max(1.0, min(2.5, 0.06 * space))
    stem_x = max(0.0, min(float(image_width), stem_x))
    y0 = max(0.0, min(float(image_height), y0))
    y1 = max(0.0, min(float(image_height), y1))
    if y1 - y0 < 1.1 * space:
        return None
    bbox = [stem_x - half_width, y0, stem_x + half_width, y1]
    points = [[int(round(stem_x)), int(round(y0))], [int(round(stem_x)), int(round(y1))]]
    bbox_int = [int(round(v)) for v in bbox]
    return {
        "id": f"synthetic_stem_{index:05d}",
        "class": "stem",
        "bbox": bbox_int,
        "confidence": min(0.62, max(0.35, symbol_confidence(note) * (0.76 if beam is not None else 0.58))),
        "source": "v2_1_geometry_synthetic",
        "attributes": {
            "staff": symbol_staff(note),
            "staff_space": space,
            "direction": direction,
            "generated_from": source,
            "notehead_id": str(note["id"]),
            "beam_id": beam_id,
            "center": [stem_x, 0.5 * (y0 + y1)],
            "normalized_width": (2.0 * half_width) / max(1.0, space),
            "normalized_height": (y1 - y0) / max(1.0, space),
        },
        "prompt_box": bbox_int,
        "mask": None,
        "mask_source": "synthetic_geometry",
        "mask_score": None,
        "mask_area": max(1, int(round((2.0 * half_width) * (y1 - y0)))),
        "skeleton": {
            "type": "polyline",
            "points": points,
            "length": float(y1 - y0),
            "orientation": "vertical",
            "angle_degrees": 90.0,
            "source": "synthetic_geometry",
        },
        "shape": {
            "area": max(1, int(round((2.0 * half_width) * (y1 - y0)))),
            "bbox_area": max(1, (bbox_int[2] - bbox_int[0]) * (bbox_int[3] - bbox_int[1])),
            "prompt_area": max(1, (bbox_int[2] - bbox_int[0]) * (bbox_int[3] - bbox_int[1])),
            "symbol_bbox_area": max(1, (bbox_int[2] - bbox_int[0]) * (bbox_int[3] - bbox_int[1])),
            "fill_ratio": 1.0,
            "symbol_bbox_fill_ratio": 1.0,
            "aspect_ratio": (2.0 * half_width) / max(1.0, y1 - y0),
            "centroid": [stem_x, 0.5 * (y0 + y1)],
            "mask_bbox": bbox_int,
            "orientation_angle_degrees": 90.0,
            "major_length": float(y1 - y0),
            "minor_length": float(2.0 * half_width),
            "staff_overlap_pixels": 0,
            "staff_overlap_ratio": 0.0,
            "staff_line_overlap_policy": "synthetic_relation_support",
        },
        "geometry_check": "pass",
        "geometry_check_details": {"status": "pass", "failures": [], "warnings": []},
        "postprocess_decisions": [source],
    }


def nearest_stem_for_note(note: dict[str, Any], stems: list[dict[str, Any]]) -> dict[str, Any] | None:
    staff = symbol_staff(note)
    space = symbol_space(note)
    candidates: list[tuple[float, dict[str, Any]]] = []
    for stem in stems:
        if staff is not None and symbol_staff(stem) != staff:
            continue
        endpoint_dist = min(point_to_bbox_distance(endpoint, note["bbox"]) for endpoint in skeleton_endpoints(stem))
        sx, _ = symbol_center(stem)
        nx, ny = symbol_center(note)
        sx0, sy0, sx1, sy1 = [float(v) for v in stem["bbox"]]
        vertical_cover = sy0 - 0.35 * space <= ny <= sy1 + 0.35 * space
        x_dist = abs(sx - nx)
        if endpoint_dist <= 1.25 * space or (vertical_cover and x_dist <= 0.85 * space):
            candidates.append((min(endpoint_dist, x_dist), stem))
    if not candidates:
        return None
    return min(candidates, key=lambda item: item[0])[1]


def stem_direction_for_note(note: dict[str, Any], stem: dict[str, Any]) -> str:
    direction = (stem.get("attributes") or {}).get("direction")
    if direction in {"up", "down"}:
        return str(direction)
    _, ny = symbol_center(note)
    _, sy0, _, sy1 = [float(v) for v in stem["bbox"]]
    return "up" if ny - sy0 >= sy1 - ny else "down"


def stem_beam_terminal(note: dict[str, Any], stem: dict[str, Any]) -> tuple[float, float]:
    endpoints = skeleton_endpoints(stem)
    direction = stem_direction_for_note(note, stem)
    return min(endpoints, key=lambda point: point[1]) if direction == "up" else max(endpoints, key=lambda point: point[1])


def supported_by_source_line(
    gray: np.ndarray,
    p0: tuple[float, float],
    p1: tuple[float, float],
    space: float,
) -> bool:
    height, width = gray.shape[:2]
    length = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
    if length < 0.55 * space:
        return False
    samples = max(10, int(round(length / 5.0)))
    radius = max(1, int(round(0.10 * space)))
    hits = 0
    usable = 0
    for idx in range(samples):
        t = (idx + 0.5) / samples
        if t < 0.13 or t > 0.87:
            continue
        x = p0[0] + (p1[0] - p0[0]) * t
        y = p0[1] + (p1[1] - p0[1]) * t
        xi = int(round(x))
        yi = int(round(y))
        if xi < 0 or xi >= width or yi < 0 or yi >= height:
            continue
        usable += 1
        x0 = max(0, xi - radius)
        x1 = min(width, xi + radius + 1)
        y0 = max(0, yi - radius)
        y1 = min(height, yi + radius + 1)
        if np.any(gray[y0:y1, x0:x1] <= 165):
            hits += 1
    return usable >= 6 and hits / usable >= 0.48


def synthetic_beam_symbol(
    index: int,
    group: list[tuple[dict[str, Any], dict[str, Any], tuple[float, float]]],
) -> dict[str, Any] | None:
    if len(group) < 2:
        return None
    terminals = [item[2] for item in group]
    space = median([symbol_space(item[0]) for item in group])
    x0 = min(point[0] for point in terminals)
    x1 = max(point[0] for point in terminals)
    if x1 - x0 < 0.55 * space:
        return None
    slope = fit_line_slope(terminals)
    if slope is None:
        slope = 0.0
    slope = max(-1.2, min(1.2, slope))
    intercept = median([point[1] - slope * point[0] for point in terminals])
    y0_line = slope * x0 + intercept
    y1_line = slope * x1 + intercept
    pad_y = max(2.0, 0.16 * space)
    bbox = [
        int(round(x0)),
        int(round(min(y0_line, y1_line) - pad_y)),
        int(round(x1)),
        int(round(max(y0_line, y1_line) + pad_y)),
    ]
    points = [[int(round(x0)), int(round(y0_line))], [int(round(x1)), int(round(y1_line))]]
    return {
        "id": f"synthetic_beam_{index:05d}",
        "class": "beam",
        "bbox": bbox,
        "confidence": 0.52,
        "source": "v2_1_geometry_synthetic",
        "attributes": {
            "staff": symbol_staff(group[0][0]),
            "staff_space": space,
            "generated_from": "source_supported_stem_terminals",
            "notehead_ids": [str(item[0]["id"]) for item in group],
            "stem_ids": [str(item[1]["id"]) for item in group],
            "slope": slope,
            "beam_count": 1,
            "center": [0.5 * (x0 + x1), 0.5 * (y0_line + y1_line)],
            "normalized_width": (x1 - x0) / max(1.0, space),
            "normalized_height": (2.0 * pad_y) / max(1.0, space),
        },
        "prompt_box": bbox,
        "mask": None,
        "mask_source": "synthetic_geometry",
        "mask_score": None,
        "mask_area": max(1, int(round((x1 - x0) * (2.0 * pad_y)))),
        "skeleton": {
            "type": "polyline",
            "points": points,
            "length": float(math.hypot(x1 - x0, y1_line - y0_line)),
            "orientation": "horizontal_or_oblique",
            "angle_degrees": math.degrees(math.atan2(y1_line - y0_line, x1 - x0)),
            "source": "synthetic_geometry",
        },
        "shape": {
            "area": max(1, int(round((x1 - x0) * (2.0 * pad_y)))),
            "bbox_area": max(1, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])),
            "prompt_area": max(1, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])),
            "symbol_bbox_area": max(1, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])),
            "fill_ratio": 1.0,
            "symbol_bbox_fill_ratio": 1.0,
            "aspect_ratio": (x1 - x0) / max(1.0, 2.0 * pad_y),
            "centroid": [0.5 * (x0 + x1), 0.5 * (y0_line + y1_line)],
            "mask_bbox": bbox,
            "orientation_angle_degrees": math.degrees(math.atan2(y1_line - y0_line, x1 - x0)),
            "major_length": float(math.hypot(x1 - x0, y1_line - y0_line)),
            "minor_length": float(2.0 * pad_y),
            "staff_overlap_pixels": 0,
            "staff_overlap_ratio": 0.0,
            "staff_line_overlap_policy": "synthetic_relation_support",
        },
        "geometry_check": "pass",
        "geometry_check_details": {"status": "pass", "failures": [], "warnings": []},
        "postprocess_decisions": ["synthetic_beam_from_source_supported_stem_terminals"],
    }


def fit_line_slope(points: list[tuple[float, float]]) -> float | None:
    if len(points) < 2:
        return None
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    denom = sum((x - mean_x) ** 2 for x in xs)
    if denom <= 1e-6:
        return None
    return sum((x - mean_x) * (y - mean_y) for x, y in points) / denom


def infer_synthetic_beams(
    noteheads: list[dict[str, Any]],
    stems: list[dict[str, Any]],
    existing_beams: list[dict[str, Any]],
    gray: np.ndarray,
) -> list[dict[str, Any]]:
    if existing_beams:
        return []
    by_staff: dict[int | None, list[dict[str, Any]]] = {}
    for note in noteheads:
        by_staff.setdefault(symbol_staff(note), []).append(note)

    synthetic: list[dict[str, Any]] = []
    for staff, notes in by_staff.items():
        entries: list[tuple[dict[str, Any], dict[str, Any], tuple[float, float], str]] = []
        for note in sorted(notes, key=lambda item: (symbol_center(item)[0], symbol_center(item)[1])):
            stem = nearest_stem_for_note(note, stems)
            if stem is None:
                continue
            entries.append((note, stem, stem_beam_terminal(note, stem), stem_direction_for_note(note, stem)))
        if len(entries) < 2:
            continue

        group: list[tuple[dict[str, Any], dict[str, Any], tuple[float, float]]] = []
        for left, right in zip(entries, entries[1:]):
            left_note, left_stem, left_terminal, left_direction = left
            right_note, right_stem, right_terminal, right_direction = right
            space = median([symbol_space(left_note), symbol_space(right_note)])
            lx, ly = symbol_center(left_note)
            rx, ry = symbol_center(right_note)
            x_gap = rx - lx
            if not (0.35 * space <= x_gap <= 2.7 * space):
                if len(group) >= 2:
                    beam = synthetic_beam_symbol(len(synthetic), group)
                    if beam is not None:
                        synthetic.append(beam)
                group = []
                continue
            if left_direction != right_direction:
                if len(group) >= 2:
                    beam = synthetic_beam_symbol(len(synthetic), group)
                    if beam is not None:
                        synthetic.append(beam)
                group = []
                continue
            if abs(left_terminal[1] - right_terminal[1]) > 1.35 * space:
                if len(group) >= 2:
                    beam = synthetic_beam_symbol(len(synthetic), group)
                    if beam is not None:
                        synthetic.append(beam)
                group = []
                continue
            if not supported_by_source_line(gray, left_terminal, right_terminal, space):
                if len(group) >= 2:
                    beam = synthetic_beam_symbol(len(synthetic), group)
                    if beam is not None:
                        synthetic.append(beam)
                group = []
                continue
            if not group:
                group = [(left_note, left_stem, left_terminal)]
            group.append((right_note, right_stem, right_terminal))

        if len(group) >= 2:
            beam = synthetic_beam_symbol(len(synthetic), group)
            if beam is not None:
                synthetic.append(beam)
    return synthetic


def clean_and_augment_symbols(
    symbols: list[dict[str, Any]],
    staves: list[dict[str, Any]],
    image_size: tuple[int, int],
    gray: np.ndarray,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    staves_by_index = staff_lookup(staves)
    deduped, duplicate_noteheads = dedupe_noteheads(symbols)
    noteheads = [symbol for symbol in deduped if symbol["class"] in NOTEHEAD_CLASSES]

    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = list(duplicate_noteheads)
    reason_counts: dict[str, int] = {}
    for symbol in deduped:
        reason = should_drop_symbol(symbol, noteheads, staves_by_index)
        if reason is None:
            kept.append(symbol)
        else:
            item = dict(symbol)
            item["drop_reason"] = reason
            dropped.append(item)
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
    if duplicate_noteheads:
        reason_counts["dropped_duplicate_notehead"] = len(duplicate_noteheads)

    noteheads = [symbol for symbol in kept if symbol["class"] in NOTEHEAD_CLASSES]
    stems = [symbol for symbol in kept if symbol["class"] == "stem"]
    beams = [symbol for symbol in kept if symbol["class"] == "beam"]
    synthetic: list[dict[str, Any]] = []
    synthetic_index = 0
    for note in sorted(noteheads, key=lambda item: (symbol_staff(item) or 9999, symbol_center(item)[0], symbol_center(item)[1])):
        if stem_exists_for_note(note, [*stems, *synthetic]):
            continue
        beam = nearest_beam_for_note(note, beams)
        stem = synthetic_stem_symbol(synthetic_index, note, beam, image_size)
        if stem is None:
            continue
        synthetic.append(stem)
        synthetic_index += 1

    all_stems = [symbol for symbol in [*kept, *synthetic] if symbol["class"] == "stem"]
    kept_beams = [symbol for symbol in kept if symbol["class"] == "beam"]
    synthetic_beams = infer_synthetic_beams(noteheads, all_stems, kept_beams, gray)
    augmented = [*kept, *synthetic, *synthetic_beams]
    summary = {
        **reason_counts,
        "synthetic_stems_added": len(synthetic),
        "synthetic_beams_added": len(synthetic_beams),
    }
    return augmented, dropped, [*synthetic, *synthetic_beams], summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract V2.1 polygon, skeleton, shape descriptors, and relation graph.")
    parser.add_argument("--symbols-json", type=Path, required=True)
    parser.add_argument("--mask-json", type=Path, help="Optional SAM2 mask metadata JSON when symbols do not already include mask fields.")
    parser.add_argument("--out-json", type=Path, default=Path("outputs/v2_1/symbols_v2_1_shapes.json"))
    parser.add_argument("--max-polygon-points", type=int, default=96)
    parser.add_argument("--max-skeleton-points", type=int, default=96)
    parser.add_argument("--note-stem-distance-scale", type=float, default=1.35)
    parser.add_argument("--note-stem-x-scale", type=float, default=0.95)
    parser.add_argument("--max-notes-per-stem", type=int, default=6)
    parser.add_argument("--beam-expand-x-scale", type=float, default=0.35)
    parser.add_argument("--beam-expand-y-scale", type=float, default=0.55)
    parser.add_argument("--beam-endpoint-scale", type=float, default=0.45)
    parser.add_argument("--ledger-y-scale", type=float, default=1.65)
    parser.add_argument("--ledger-distance-scale", type=float, default=1.85)
    parser.add_argument("--slur-endpoint-scale", type=float, default=3.0)
    args = parser.parse_args()

    payload = read_json(args.symbols_json)
    input_path = Path(payload["input"])
    image = Image.open(input_path).convert("RGB")
    image_size = image.size
    gray = np.array(image.convert("L"))
    external_masks = load_external_masks(args.mask_json)
    anchors = [Path.cwd(), args.symbols_json.parent]
    if args.mask_json:
        anchors.append(args.mask_json.parent)
    staves = payload.get("staves", [])
    symbols = [
        enrich_symbol_shape(
            symbol,
            image_size,
            staves,
            external_masks,
            anchors,
            args.max_polygon_points,
            args.max_skeleton_points,
        )
        for symbol in payload.get("symbols", [])
    ]
    symbols, dropped_symbols, synthetic_symbols, postprocess_summary = clean_and_augment_symbols(symbols, staves, image_size, gray)
    relations = build_relations(
        symbols,
        note_stem_distance_scale=args.note_stem_distance_scale,
        note_stem_x_scale=args.note_stem_x_scale,
        max_notes_per_stem=args.max_notes_per_stem,
        beam_expand_x_scale=args.beam_expand_x_scale,
        beam_expand_y_scale=args.beam_expand_y_scale,
        beam_endpoint_scale=args.beam_endpoint_scale,
        ledger_y_scale=args.ledger_y_scale,
        ledger_distance_scale=args.ledger_distance_scale,
        slur_endpoint_scale=args.slur_endpoint_scale,
    )
    geometry_counts: dict[str, int] = {}
    for symbol in symbols:
        status = symbol.get("geometry_check", "unknown")
        geometry_counts[status] = geometry_counts.get(status, 0) + 1
    result = {
        **payload,
        "version": "v2.1_symbol_shapes",
        "stack": {
            **payload.get("stack", {}),
            "shape_refinement": "staff_aware_box_to_mask_v2_1",
            "relations": "geometry_heuristic_relation_graph_v2_1",
            "symbol_cleanup": "staff_aware_false_line_filter_and_synthetic_stems_v2_1",
        },
        "symbols": symbols,
        "dropped_symbols": dropped_symbols,
        "relations": relations,
        "counts": count_by_class(symbols),
        "v2_1_summary": {
            "symbols": len(symbols),
            "dropped_symbols": len(dropped_symbols),
            "synthetic_symbols": len(synthetic_symbols),
            "relations": len(relations),
            "geometry_counts": dict(sorted(geometry_counts.items())),
            "postprocess": dict(sorted(postprocess_summary.items())),
        },
    }
    write_json(args.out_json, result)
    print(
        {
            "out_json": str(args.out_json),
            "symbols": len(symbols),
            "dropped_symbols": len(dropped_symbols),
            "synthetic_symbols": len(synthetic_symbols),
            "relations": len(relations),
            "geometry_counts": dict(sorted(geometry_counts.items())),
            "postprocess": dict(sorted(postprocess_summary.items())),
        }
    )


if __name__ == "__main__":
    main()

