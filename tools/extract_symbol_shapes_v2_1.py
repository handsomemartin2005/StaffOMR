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


def build_relations(symbols: list[dict[str, Any]]) -> list[dict[str, Any]]:
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
        candidates = []
        for note in noteheads:
            if staff is not None and symbol_staff(note) != staff:
                continue
            dist = min(point_to_bbox_distance(endpoint, note["bbox"]) for endpoint in endpoints)
            candidates.append((dist, note))
        if not candidates:
            continue
        dist, note = min(candidates, key=lambda item: item[0])
        threshold = max(8.0, 1.15 * space)
        if dist <= threshold:
            score = 1.0 - min(1.0, dist / threshold)
            add_relation(
                relations,
                symbols_by_id,
                "notehead_stem_attachment",
                str(note["id"]),
                [str(stem["id"])],
                score,
                {"endpoint_distance": dist, "threshold": threshold},
            )

    for beam in beams:
        staff = symbol_staff(beam)
        space = symbol_space(beam)
        expanded = expand_bbox([int(v) for v in beam["bbox"]], 0.35 * space, 0.55 * space, 100_000, 100_000)
        attached: list[tuple[float, dict[str, Any]]] = []
        for stem in stems:
            if staff is not None and symbol_staff(stem) != staff:
                continue
            endpoints = skeleton_endpoints(stem)
            distances = [point_to_bbox_distance(endpoint, expanded) for endpoint in endpoints]
            dist = min(distances)
            if dist <= max(5.0, 0.45 * space) or bbox_intersects(stem["bbox"], expanded):
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
            if x_overlap >= 0 and abs(lcy - ncy) <= 1.65 * space:
                candidates.append((dist, note))
        if not candidates:
            continue
        dist, note = min(candidates, key=lambda item: item[0])
        threshold = max(8.0, 1.85 * space)
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
                if dist <= 3.0 * space:
                    targets.append(str(note["id"]))
                    distances.append(dist)
        targets = list(dict.fromkeys(targets))
        if len(targets) >= 2:
            score = 1.0 - min(1.0, float(np.mean(distances)) / max(1.0, 3.0 * space))
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract V2.1 polygon, skeleton, shape descriptors, and relation graph.")
    parser.add_argument("--symbols-json", type=Path, required=True)
    parser.add_argument("--mask-json", type=Path, help="Optional SAM2 mask metadata JSON when symbols do not already include mask fields.")
    parser.add_argument("--out-json", type=Path, default=Path("outputs/v2_1/symbols_v2_1_shapes.json"))
    parser.add_argument("--max-polygon-points", type=int, default=96)
    parser.add_argument("--max-skeleton-points", type=int, default=96)
    args = parser.parse_args()

    payload = read_json(args.symbols_json)
    input_path = Path(payload["input"])
    image = Image.open(input_path).convert("RGB")
    image_size = image.size
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
    relations = build_relations(symbols)
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
        },
        "symbols": symbols,
        "relations": relations,
        "counts": count_by_class(symbols),
        "v2_1_summary": {
            "symbols": len(symbols),
            "relations": len(relations),
            "geometry_counts": dict(sorted(geometry_counts.items())),
        },
    }
    write_json(args.out_json, result)
    print(
        {
            "out_json": str(args.out_json),
            "symbols": len(symbols),
            "relations": len(relations),
            "geometry_counts": dict(sorted(geometry_counts.items())),
        }
    )


if __name__ == "__main__":
    main()
