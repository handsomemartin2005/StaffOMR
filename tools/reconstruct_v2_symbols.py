from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from PIL import Image, ImageChops, ImageDraw, ImageFont

from compose_v2_1_marks import compose_marks


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_fonts() -> tuple[ImageFont.ImageFont, ImageFont.ImageFont, ImageFont.ImageFont]:
    try:
        return (
            ImageFont.truetype("arial.ttf", 22),
            ImageFont.truetype("arial.ttf", 30),
            ImageFont.truetype("arial.ttf", 46),
        )
    except Exception:
        font = ImageFont.load_default()
        return font, font, font


NOTEHEAD_CLASSES = {"filled_notehead", "open_notehead"}
REST_FINE_CLASSES = {
    "rest",
    "rest_double_whole",
    "rest_whole",
    "rest_half",
    "rest_quarter",
    "rest_8th",
    "rest_16th",
    "rest_32nd",
    "rest_64th",
    "rest_128th",
}


def box(symbol: dict[str, Any]) -> list[int]:
    return [int(round(v)) for v in symbol["bbox"]]


def bbox_area(bbox: list[int]) -> int:
    return max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])


def overlap_area(a: list[int], b: list[int]) -> int:
    x0 = max(a[0], b[0])
    y0 = max(a[1], b[1])
    x1 = min(a[2], b[2])
    y1 = min(a[3], b[3])
    return max(0, x1 - x0) * max(0, y1 - y0)


def center(symbol: dict[str, Any]) -> tuple[float, float]:
    x0, y0, x1, y1 = box(symbol)
    return 0.5 * (x0 + x1), 0.5 * (y0 + y1)


def staff_space(symbol: dict[str, Any]) -> float:
    attrs = symbol.get("attributes") or {}
    return float(attrs.get("staff_space") or 12.0)


def detector_class(symbol: dict[str, Any]) -> str:
    attributes = symbol.get("attributes") or {}
    return str(attributes.get("detector_class") or attributes.get("fine_class") or symbol["class"])


def skeleton_points(symbol: dict[str, Any]) -> list[list[float]]:
    skeleton = symbol.get("skeleton") or {}
    points = skeleton.get("points") or []
    return [[float(point[0]), float(point[1])] for point in points if len(point) >= 2]


def polyline_endpoints(symbol: dict[str, Any]) -> tuple[tuple[float, float], tuple[float, float]]:
    points = skeleton_points(symbol)
    if len(points) >= 2:
        return (points[0][0], points[0][1]), (points[-1][0], points[-1][1])
    x0, y0, x1, y1 = box(symbol)
    if symbol["class"] in {"stem", "barline"}:
        cx = 0.5 * (x0 + x1)
        return (cx, float(y0)), (cx, float(y1))
    cy = 0.5 * (y0 + y1)
    return (float(x0), cy), (float(x1), cy)


def point_to_bbox_distance(point: tuple[float, float], bbox: list[int]) -> float:
    x, y = point
    x0, y0, x1, y1 = bbox
    dx = max(x0 - x, 0.0, x - x1)
    dy = max(y0 - y, 0.0, y - y1)
    return math.hypot(dx, dy)


def stem_terminal_for_beam(stem: dict[str, Any], beam: dict[str, Any]) -> tuple[float, float]:
    endpoints = polyline_endpoints(stem)
    beam_bbox = box(beam)
    return min(endpoints, key=lambda point: point_to_bbox_distance(point, beam_bbox))


def draw_thick_segment(
    draw: ImageDraw.ImageDraw,
    p0: tuple[float, float],
    p1: tuple[float, float],
    width: int,
    fill: tuple[int, int, int] = (0, 0, 0),
) -> None:
    draw.line([(int(round(p0[0])), int(round(p0[1]))), (int(round(p1[0])), int(round(p1[1])))], fill=fill, width=width)


def fit_slope(points: list[tuple[float, float]]) -> float | None:
    if len(points) < 2:
        return None
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    denom = sum((x - mean_x) ** 2 for x in xs)
    if denom <= 1e-6:
        return None
    slope = sum((x - mean_x) * (y - mean_y) for x, y in points) / denom
    return max(-1.8, min(1.8, slope))


def beam_slope(beam: dict[str, Any], attached_stems: list[dict[str, Any]] | None = None) -> float:
    attrs = beam.get("attributes") or {}
    if isinstance(attrs.get("slope"), (int, float)):
        return float(attrs["slope"])
    points = [(px, py) for px, py in skeleton_points(beam)]
    slope = fit_slope(points)
    if slope is not None:
        return slope
    if attached_stems and len(attached_stems) >= 2:
        terminals = [stem_terminal_for_beam(stem, beam) for stem in attached_stems]
        slope = fit_slope(terminals)
        if slope is not None:
            return slope
    return 0.0


def beam_center_segment(
    beam: dict[str, Any],
    attached_stems: list[dict[str, Any]] | None = None,
) -> tuple[tuple[float, float], tuple[float, float]]:
    x0, y0, x1, y1 = box(beam)
    attrs = beam.get("attributes") or {}
    if isinstance(attrs.get("slope"), (int, float)):
        slope = float(attrs["slope"])
        cx = 0.5 * (x0 + x1)
        cy = 0.5 * (y0 + y1)
        line_x0 = float(x0)
        line_x1 = float(x1)
        if attached_stems and len(attached_stems) >= 2:
            terminals = [stem_terminal_for_beam(stem, beam) for stem in attached_stems]
            stem_x0 = min(point[0] for point in terminals)
            stem_x1 = max(point[0] for point in terminals)
            line_x0 = min(line_x0, stem_x0)
            line_x1 = max(line_x1, stem_x1)
        return (line_x0, cy + slope * (line_x0 - cx)), (line_x1, cy + slope * (line_x1 - cx))

    points = sorted([(px, py) for px, py in skeleton_points(beam)], key=lambda point: point[0])
    if len(points) >= 2 and (points[-1][0] - points[0][0]) >= 0.4 * staff_space(beam):
        return points[0], points[-1]

    slope = beam_slope(beam, attached_stems)
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    return (float(x0), cy + slope * (x0 - cx)), (float(x1), cy + slope * (x1 - cx))


def estimate_beam_count_from_pixels(
    source_gray: Image.Image | None,
    beam: dict[str, Any],
    slope: float,
) -> int | None:
    if source_gray is None:
        return None
    x0, y0, x1, y1 = box(beam)
    space = staff_space(beam)
    pad = max(1, int(round(0.15 * space)))
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(source_gray.width, x1 + pad)
    y1 = min(source_gray.height, y1 + pad)
    if x1 <= x0 + 2 or y1 <= y0 + 2:
        return None

    crop = source_gray.crop((x0, y0, x1, y1))
    pixels = crop.load()
    width, height = crop.size
    hist: dict[int, int] = {}
    black = 0
    for yy in range(height):
        for xx in range(width):
            if pixels[xx, yy] <= 190:
                projection = yy - slope * xx
                key = int(round(projection))
                hist[key] = hist.get(key, 0) + 1
                black += 1
    if black < max(8, int(0.6 * width)):
        return None

    keys = list(range(min(hist), max(hist) + 1))
    smoothed: dict[int, float] = {}
    for key in keys:
        smoothed[key] = (
            hist.get(key - 1, 0) * 0.25
            + hist.get(key, 0) * 0.5
            + hist.get(key + 1, 0) * 0.25
        )
    peak = max(smoothed.values(), default=0.0)
    if peak <= 0:
        return None

    threshold = max(2.0, peak * 0.34, width * 0.10)
    clusters: list[list[int]] = []
    current: list[int] = []
    max_gap = max(1, int(round(0.16 * space)))
    last_active: int | None = None
    for key in keys:
        if smoothed[key] >= threshold:
            if current and last_active is not None and key - last_active > max_gap:
                clusters.append(current)
                current = []
            current.append(key)
            last_active = key
    if current:
        clusters.append(current)

    meaningful = []
    for cluster in clusters:
        mass = sum(smoothed[key] for key in cluster)
        if mass >= max(peak * 0.55, width * 0.18):
            meaningful.append(cluster)
    if not meaningful:
        return None
    return max(1, min(4, len(meaningful)))


def estimate_beam_count(
    beam: dict[str, Any],
    source_gray: Image.Image | None,
    slope: float,
) -> int:
    attrs = beam.get("attributes") or {}
    for key in ("beam_count", "beam_level", "parallel_beams"):
        value = attrs.get(key)
        if isinstance(value, (int, float)) and value >= 1:
            return max(1, min(4, int(round(value))))
    if attrs.get("beam_group_index") is not None or attrs.get("link_indices") is not None:
        return 1

    pixel_count = estimate_beam_count_from_pixels(source_gray, beam, slope)
    if pixel_count is not None:
        return pixel_count

    x0, y0, x1, y1 = box(beam)
    height = max(1.0, y1 - y0)
    space = staff_space(beam)
    ratio = height / max(1.0, space)
    if ratio >= 3.2:
        return 4
    if ratio >= 2.5:
        return 3
    if ratio >= 2.0:
        return 2
    return 1


def median(values: list[float]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def beam_staff_key(beam: dict[str, Any]) -> int | None:
    value = (beam.get("attributes") or {}).get("staff")
    return int(value) if value is not None else None


def beam_direction_key(beam: dict[str, Any]) -> str:
    return str((beam.get("attributes") or {}).get("direction") or "")


def beam_bbox_center(beam: dict[str, Any]) -> tuple[float, float]:
    x0, y0, x1, y1 = box(beam)
    return 0.5 * (x0 + x1), 0.5 * (y0 + y1)


def can_merge_beam_item(
    group: list[dict[str, Any]],
    item: dict[str, Any],
) -> bool:
    last = group[-1]["beam"]
    beam = item["beam"]
    if beam_staff_key(last) != beam_staff_key(beam):
        return False
    if beam_direction_key(last) != beam_direction_key(beam):
        return False

    lx0, _, lx1, _ = box(last)
    bx0, _, _, _ = box(beam)
    space = median([staff_space(entry["beam"]) for entry in [*group, item]])
    gap = bx0 - lx1
    if gap < -0.5 * space or gap > max(24.0, 3.8 * space):
        return False

    centers = [beam_bbox_center(entry["beam"]) for entry in group]
    slope = fit_slope(centers)
    if slope is None:
        slopes = [beam_slope(entry["beam"]) for entry in group if beam_slope(entry["beam"]) is not None]
        slope = sum(slopes) / len(slopes) if slopes else beam_slope(last)
    beam_s = beam_slope(beam)
    if abs(beam_s - slope) > 0.75:
        return False

    last_cx, last_cy = beam_bbox_center(last)
    cx, cy = beam_bbox_center(beam)
    expected = last_cy + slope * (cx - last_cx)
    return abs(cy - expected) <= max(10.0, 1.5 * space)


def group_beam_items_for_render(items: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    ordered = sorted(
        items,
        key=lambda item: (
            9999 if beam_staff_key(item["beam"]) is None else beam_staff_key(item["beam"]),
            beam_direction_key(item["beam"]),
            box(item["beam"])[0],
            box(item["beam"])[1],
        ),
    )
    groups: list[list[dict[str, Any]]] = []
    for item in ordered:
        placed = False
        for group in reversed(groups):
            if can_merge_beam_item(group, item):
                group.append(item)
                placed = True
                break
        if not placed:
            groups.append([item])
    return groups


def estimate_merged_beam_count(
    group: list[dict[str, Any]],
    source_gray: Image.Image | None,
    slope: float,
    intercept: float,
) -> int:
    beams = [item["beam"] for item in group]
    if not beams:
        return 1
    space = median([staff_space(beam) for beam in beams])
    direct_counts = [estimate_beam_count(beam, source_gray, slope) for beam in beams]
    best_direct = max(direct_counts, default=1)
    if best_direct > 1:
        return best_direct

    offsets = sorted(beam_bbox_center(beam)[1] - (slope * beam_bbox_center(beam)[0] + intercept) for beam in beams)
    if len(offsets) < 2:
        return 1
    clusters: list[list[float]] = []
    gap_threshold = max(3.0, 0.30 * space)
    for offset in offsets:
        if not clusters or abs(offset - clusters[-1][-1]) > gap_threshold:
            clusters.append([offset])
        else:
            clusters[-1].append(offset)
    meaningful = [cluster for cluster in clusters if len(cluster) >= 2 or len(beams) <= 4]
    return max(1, min(4, max(best_direct, len(meaningful))))


def draw_merged_beam_group(
    draw: ImageDraw.ImageDraw,
    group: list[dict[str, Any]],
    source_gray: Image.Image | None = None,
) -> None:
    beams = [item["beam"] for item in group]
    if not beams:
        return
    space = median([staff_space(beam) for beam in beams])
    centers = [beam_bbox_center(beam) for beam in beams]
    slope = fit_slope(centers)
    if slope is None:
        slopes = [beam_slope(beam) for beam in beams]
        slope = sum(slopes) / len(slopes) if slopes else 0.0
    slope = max(-1.8, min(1.8, slope))

    intercept = median([cy - slope * cx for cx, cy in centers])
    x0 = min(box(beam)[0] for beam in beams)
    x1 = max(box(beam)[2] for beam in beams)
    stems = [stem for item in group for stem in item.get("stems", [])]
    if len(stems) >= 2:
        x0 = min(x0, int(round(min(center(stem)[0] for stem in stems))))
        x1 = max(x1, int(round(max(center(stem)[0] for stem in stems))))
    if x1 <= x0:
        return

    thickness = max(2, min(4, int(round(0.16 * space))))
    count = estimate_merged_beam_count(group, source_gray, slope, intercept)
    gap = max(3, int(round(0.40 * space)))
    total = count - 1
    for idx in range(count):
        offset = (idx - 0.5 * total) * gap
        draw_thick_segment(
            draw,
            (float(x0), slope * x0 + intercept + offset),
            (float(x1), slope * x1 + intercept + offset),
            thickness,
        )


def paste_black_cutout(
    canvas: Image.Image,
    source: Image.Image,
    bbox: list[int],
    black_threshold: int = 210,
    pad: int = 2,
) -> bool:
    x0, y0, x1, y1 = [int(round(v)) for v in bbox]
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(source.width, x1 + pad)
    y1 = min(source.height, y1 + pad)
    if x1 <= x0 or y1 <= y0:
        return False
    crop = source.crop((x0, y0, x1, y1))
    mask = crop.convert("L").point(lambda value: 255 if value <= black_threshold else 0)
    canvas.paste(Image.new("RGB", crop.size, "black"), (x0, y0), mask)
    return True


def draw_rule_beam(
    draw: ImageDraw.ImageDraw,
    beam: dict[str, Any],
    attached_stems: list[dict[str, Any]] | None = None,
    source_gray: Image.Image | None = None,
) -> None:
    space = staff_space(beam)
    thickness = max(2, min(4, int(round(0.16 * space))))
    p0, p1 = beam_center_segment(beam, attached_stems)
    if math.hypot(p1[0] - p0[0], p1[1] - p0[1]) < max(4.0, 0.35 * space):
        return

    slope = beam_slope(beam, attached_stems)
    count = estimate_beam_count(beam, source_gray, slope)
    gap = max(3, int(round(0.40 * space)))
    total = count - 1
    for idx in range(count):
        offset = (idx - 0.5 * total) * gap
        draw_thick_segment(draw, (p0[0], p0[1] + offset), (p1[0], p1[1] + offset), thickness)


def draw_stem(draw: ImageDraw.ImageDraw, stem: dict[str, Any]) -> None:
    p0, p1 = polyline_endpoints(stem)
    width = max(1, min(3, int(round(0.12 * staff_space(stem)))))
    draw_thick_segment(draw, p0, p1, width)


def draw_slur_or_tie(
    draw: ImageDraw.ImageDraw,
    slur: dict[str, Any],
    related_noteheads: list[dict[str, Any]] | None = None,
) -> None:
    x0, y0, x1, y1 = box(slur)
    width = max(1, min(3, int(round(0.10 * staff_space(slur)))))
    if related_noteheads and len(related_noteheads) >= 2:
        left, right = sorted(related_noteheads[:2], key=lambda item: center(item)[0])
        sx, sy = center(left)
        ex, ey = center(right)
        if ex <= sx:
            ex = sx + max(4.0, 0.5 * staff_space(slur))
        above = y1 <= min(sy, ey) or (0.5 * (y0 + y1) < 0.5 * (sy + ey))
        arch = max(5.0, min(1.2 * staff_space(slur), 0.25 * abs(ex - sx)))
        ctrl_y = min(sy, ey) - arch if above else max(sy, ey) + arch
        points = []
        for idx in range(24):
            t = idx / 23.0
            x = (1 - t) * (1 - t) * sx + 2 * (1 - t) * t * (0.5 * (sx + ex)) + t * t * ex
            y = (1 - t) * (1 - t) * sy + 2 * (1 - t) * t * ctrl_y + t * t * ey
            points.append((int(round(x)), int(round(y))))
        draw.line(points, fill=(0, 0, 0), width=width)
        return
    draw.arc([x0, y0, x1, y1], 190, 350, fill=(0, 0, 0), width=width)


def filter_conflicting_rests_for_render(symbols: list[dict[str, Any]]) -> list[dict[str, Any]]:
    blockers = [
        symbol
        for symbol in symbols
        if symbol["class"] in NOTEHEAD_CLASSES or symbol["class"] in {"stem", "beam"}
    ]
    filtered: list[dict[str, Any]] = []
    for symbol in symbols:
        if symbol["class"] != "rest":
            filtered.append(symbol)
            continue
        rest_box = box(symbol)
        rest_area = max(1, bbox_area(rest_box))
        conflict = False
        for blocker in blockers:
            inter = overlap_area(rest_box, box(blocker))
            if inter <= 0:
                continue
            ratio = inter / rest_area
            threshold = 0.18 if blocker["class"] in NOTEHEAD_CLASSES else 0.28
            if ratio >= threshold:
                conflict = True
                break
        if not conflict:
            filtered.append(symbol)
    return filtered


def build_relations_by_type(payload: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    relations: dict[str, list[dict[str, Any]]] = {}
    for relation in payload.get("relations", []):
        relations.setdefault(str(relation.get("type")), []).append(relation)
    return relations


def draw_cutout_symbols(
    canvas: Image.Image,
    symbols_payload: dict[str, Any],
    symbols: list[dict[str, Any]],
    black_threshold: int,
) -> None:
    source = Image.open(symbols_payload["input"]).convert("RGB")
    gray = source.convert("L")
    black_pixels = gray.point(lambda value: 255 if value <= black_threshold else 0)
    symbol_regions = Image.new("L", source.size, 0)
    region_draw = ImageDraw.Draw(symbol_regions)
    for symbol in symbols:
        region_draw.rectangle(box(symbol), fill=255)
    symbol_black_pixels = ImageChops.multiply(symbol_regions, black_pixels)
    canvas.paste(Image.new("RGB", canvas.size, "black"), mask=symbol_black_pixels)


def draw_ocr_text(
    draw: ImageDraw.ImageDraw,
    ocr_payload: dict[str, Any] | None,
    font: ImageFont.ImageFont,
    instrument_only: bool = False,
) -> None:
    if not ocr_payload:
        return
    for item in ocr_payload.get("ocr_results", []):
        if instrument_only and not str(item.get("symbol_id", "")).startswith("ocr_instrument_"):
            continue
        bbox = item.get("bbox") or item.get("symbol_bbox")
        if not bbox:
            continue
        x0, y0, _, _ = [int(round(v)) for v in bbox]
        draw.text((x0, max(0, y0 - 4)), item.get("text", ""), fill=(0, 0, 0), font=font)


def draw_symbolic_clef(draw: ImageDraw.ImageDraw, symbol: dict[str, Any]) -> None:
    x0, y0, x1, y1 = box(symbol)
    width = max(1, x1 - x0)
    height = max(1, y1 - y0)
    stroke = max(2, int(round(0.10 * min(width, height))))
    if symbol["class"] == "bass_clef":
        head_r = max(2, int(round(0.16 * height)))
        head_cx = x0 + int(round(0.28 * width))
        head_cy = y0 + int(round(0.44 * height))
        draw.ellipse(
            [head_cx - head_r, head_cy - head_r, head_cx + head_r, head_cy + head_r],
            fill=(0, 0, 0),
        )
        arc_box = [
            x0 + int(round(0.14 * width)),
            y0 + int(round(0.10 * height)),
            x1 + int(round(0.10 * width)),
            y0 + int(round(0.86 * height)),
        ]
        draw.arc(arc_box, start=250, end=80, fill=(0, 0, 0), width=stroke)
        dot_r = max(2, int(round(0.055 * height)))
        dot_x = x0 + int(round(0.72 * width))
        for dot_y in (y0 + int(round(0.35 * height)), y0 + int(round(0.58 * height))):
            draw.ellipse([dot_x - dot_r, dot_y - dot_r, dot_x + dot_r, dot_y + dot_r], fill=(0, 0, 0))
        return

    cx = x0 + int(round(0.48 * width))
    draw.line(
        [(cx, y0 + int(round(0.04 * height))), (cx, y1 - int(round(0.04 * height)))],
        fill=(0, 0, 0),
        width=stroke,
    )
    outer = [
        x0 + int(round(0.05 * width)),
        y0 + int(round(0.22 * height)),
        x1 - int(round(0.05 * width)),
        y1 - int(round(0.06 * height)),
    ]
    draw.arc(outer, start=25, end=335, fill=(0, 0, 0), width=stroke)
    inner = [
        x0 + int(round(0.22 * width)),
        y0 + int(round(0.40 * height)),
        x1 - int(round(0.18 * width)),
        y1 - int(round(0.20 * height)),
    ]
    draw.arc(inner, start=205, end=565, fill=(0, 0, 0), width=stroke)
    top = [
        x0 + int(round(0.40 * width)),
        y0 + int(round(0.02 * height)),
        x1 - int(round(0.02 * width)),
        y0 + int(round(0.36 * height)),
    ]
    draw.arc(top, start=115, end=310, fill=(0, 0, 0), width=stroke)


def draw_symbol_reconstruction(
    symbols_payload: dict[str, Any],
    ocr_payload: dict[str, Any] | None,
    out_path: Path,
    render_mode: str = "symbolic",
    cutout_black_threshold: int = 210,
) -> None:
    input_path = Path(symbols_payload["input"])
    image = Image.open(input_path).convert("RGB")
    source_gray = image.convert("L")
    width, height = image.size
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    small_font, mid_font, clef_font = load_fonts()

    symbols = filter_conflicting_rests_for_render(symbols_payload.get("symbols", []))
    mark_payload = compose_marks(symbols, symbols_payload.get("staves", []), include_singletons=False)
    composite_marks = mark_payload["marks"]
    composite_consumed_ids = set(mark_payload["consumed_symbol_ids"])
    render_symbols = [symbol for symbol in symbols if str(symbol.get("id")) not in composite_consumed_ids]
    symbols_by_id = {str(symbol["id"]): symbol for symbol in symbols}
    relations_by_type = build_relations_by_type(symbols_payload)
    by_class: dict[str, list[dict[str, Any]]] = {}
    for symbol in render_symbols:
        by_class.setdefault(symbol["class"], []).append(symbol)
    clef_staffs = {
        int((symbol.get("attributes") or {}).get("staff"))
        for symbol in render_symbols
        if symbol["class"] in {"treble_clef", "bass_clef"} and (symbol.get("attributes") or {}).get("staff") is not None
    }

    for staff in symbols_payload.get("staves", []):
        x0, x1 = int(staff["x0"]), int(staff["x1"])
        for y in staff["lines"]:
            yy = int(round(y))
            draw.line([(x0, yy), (x1, yy)], fill=(0, 0, 0), width=2)
        clef_bbox = staff.get("clef_bbox")
        if clef_bbox and int(staff.get("index", -1)) not in clef_staffs:
            paste_black_cutout(canvas, image, [int(round(v)) for v in clef_bbox], cutout_black_threshold, pad=3)

    if render_mode == "cutout":
        draw_cutout_symbols(canvas, symbols_payload, render_symbols, cutout_black_threshold)
        draw_ocr_text(draw, ocr_payload, small_font, instrument_only=True)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(out_path)
        return

    for symbol in by_class.get("barline", []):
        x0, y0, x1, y1 = box(symbol)
        draw.line([(x0 + (x1 - x0) // 2, y0), (x0 + (x1 - x0) // 2, y1)], fill=(0, 0, 0), width=max(2, x1 - x0))
    for symbol in by_class.get("ledger_line", []):
        x0, y0, x1, y1 = box(symbol)
        draw.line([(x0, (y0 + y1) // 2), (x1, (y0 + y1) // 2)], fill=(0, 0, 0), width=max(2, y1 - y0))
    for symbol in by_class.get("stem", []):
        draw_stem(draw, symbol)

    drawn_beams: set[str] = set()
    beam_items: list[dict[str, Any]] = []
    for relation in relations_by_type.get("beam_stem_group", []):
        beam = symbols_by_id.get(str(relation.get("source")))
        if beam is None:
            continue
        stems = [symbols_by_id[target] for target in map(str, relation.get("targets", [])) if target in symbols_by_id]
        beam_items.append({"beam": beam, "stems": stems})
    for group in group_beam_items_for_render(beam_items):
        if len(group) >= 2:
            draw_merged_beam_group(draw, group, source_gray)
            drawn_beams.update(str(item["beam"]["id"]) for item in group)
            continue
        beam = group[0]["beam"]
        draw_rule_beam(draw, beam, group[0].get("stems", []), source_gray)
        drawn_beams.add(str(beam["id"]))
    for symbol in by_class.get("beam", []):
        if str(symbol["id"]) not in drawn_beams:
            draw_rule_beam(draw, symbol, source_gray=source_gray)

    drawn_slurs: set[str] = set()
    for relation in relations_by_type.get("slur_tie_notehead_endpoints", []):
        slur = symbols_by_id.get(str(relation.get("source")))
        if slur is None:
            continue
        notes = [symbols_by_id[target] for target in map(str, relation.get("targets", [])) if target in symbols_by_id]
        draw_slur_or_tie(draw, slur, notes)
        drawn_slurs.add(str(slur["id"]))
    for symbol in by_class.get("slur_or_tie", []):
        if str(symbol["id"]) not in drawn_slurs:
            draw_slur_or_tie(draw, symbol)

    for symbol in by_class.get("filled_notehead", []):
        x0, y0, x1, y1 = box(symbol)
        draw.ellipse([x0, y0, x1, y1], fill=(0, 0, 0), outline=(0, 0, 0), width=2)
    for symbol in by_class.get("open_notehead", []):
        x0, y0, x1, y1 = box(symbol)
        draw.ellipse([x0, y0, x1, y1], fill="white", outline=(0, 0, 0), width=3)
    for symbol_class, text in [("sharp", "#"), ("flat", "b"), ("natural", "N")]:
        for symbol in by_class.get(symbol_class, []):
            x0, y0, _, _ = box(symbol)
            draw.text((x0, max(0, y0 - 4)), text, fill=(0, 0, 0), font=mid_font)
    for symbol_class in ("treble_clef", "bass_clef"):
        for symbol in by_class.get(symbol_class, []):
            draw_symbolic_clef(draw, symbol)

    def centered_text(symbol: dict[str, Any], label: str, font: ImageFont.ImageFont = mid_font) -> None:
        x0, y0, x1, y1 = box(symbol)
        draw.text((x0, max(0, y0 - 4)), label, fill=(0, 0, 0), font=font)
        if x1 - x0 > 4 and y1 - y0 > 4:
            draw.rectangle([x0, y0, x1, y1], outline=(0, 0, 0), width=1)

    for mark in composite_marks:
        x0, y0, _, _ = [int(round(value)) for value in mark["bbox"]]
        font = small_font if mark.get("kind") == "pedal" else mid_font
        draw.text((x0, max(0, y0 - 4)), str(mark.get("text") or ""), fill=(0, 0, 0), font=font)

    for symbol in render_symbols:
        symbol_class = symbol["class"]
        fine_class = detector_class(symbol)
        x0, y0, x1, y1 = box(symbol)
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        label_class = fine_class if fine_class != symbol_class else symbol_class
        if label_class in {"augmentation_dot", "repeat_dot"}:
            radius = max(2, min(x1 - x0, y1 - y0) // 2)
            draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=(0, 0, 0))
        elif label_class.startswith("flag_"):
            direction = -1 if label_class.endswith("_up") else 1
            draw.line([(x0, y0), (x1, cy + direction * max(4, y1 - y0))], fill=(0, 0, 0), width=3)
            draw.arc([x0, min(y0, cy), x1, max(y1, cy + direction * (y1 - y0))], 270, 80, fill=(0, 0, 0), width=2)
        elif label_class in REST_FINE_CLASSES:
            if label_class in {"rest_whole", "rest_half", "rest_double_whole"}:
                bar_h = max(3, min(7, int(round(0.22 * staff_space(symbol)))))
                draw.rectangle([x0, cy - bar_h // 2, x1, cy + bar_h // 2], fill=(0, 0, 0))
            elif label_class == "rest_quarter":
                draw.line([(cx, y0), (x0, cy), (cx, y1)], fill=(0, 0, 0), width=2)
                draw.line([(cx, y0), (x1, cy)], fill=(0, 0, 0), width=2)
            else:
                draw.line([(x0, y0), (cx, cy), (x0, y1)], fill=(0, 0, 0), width=2)
                radius = max(2, min(4, (y1 - y0) // 5))
                draw.ellipse([x1 - 2 * radius, cy - radius, x1, cy + radius], fill=(0, 0, 0))
        elif label_class.startswith("time_sig_"):
            suffix = label_class.replace("time_sig_", "")
            label = {"common": "C", "cut_common": "C/"}.get(suffix, suffix)
            centered_text(symbol, label, clef_font)
        elif label_class.startswith("dynamic_"):
            if label_class == "dynamic_crescendo_hairpin":
                draw.line([(x0, cy), (x1, y0)], fill=(0, 0, 0), width=2)
                draw.line([(x0, cy), (x1, y1)], fill=(0, 0, 0), width=2)
            elif label_class == "dynamic_diminuendo_hairpin":
                draw.line([(x0, y0), (x1, cy)], fill=(0, 0, 0), width=2)
                draw.line([(x0, y1), (x1, cy)], fill=(0, 0, 0), width=2)
            else:
                centered_text(symbol, label_class.replace("dynamic_", ""), mid_font)
        elif label_class.startswith("artic_"):
            mark = {
                "artic_staccato": ".",
                "artic_staccatissimo": "'",
                "artic_accent": ">",
                "artic_tenuto": "-",
                "artic_marcato": "^",
            }.get(label_class, ".")
            centered_text(symbol, mark, mid_font)
        elif label_class == "fermata":
            centered_text(symbol, "U", mid_font)
        elif label_class.startswith("tuplet_"):
            centered_text(symbol, label_class.replace("tuplet_", ""), small_font)
        elif label_class == "fingering":
            centered_text(symbol, "1", small_font)
        elif label_class == "pedal_mark":
            centered_text(symbol, "Ped", small_font)
        elif label_class == "pedal_up":
            centered_text(symbol, "*", small_font)
        elif label_class.startswith("ornament_"):
            centered_text(symbol, "~", mid_font)
        elif label_class == "arpeggiato":
            centered_text(symbol, "~", mid_font)
        elif label_class in {"brace", "bracket"}:
            draw.line([(cx, y0), (cx, y1)], fill=(0, 0, 0), width=3)
        elif label_class == "repeat_barline":
            draw.line([(x0, y0), (x0, y1)], fill=(0, 0, 0), width=2)
            draw.line([(x1, y0), (x1, y1)], fill=(0, 0, 0), width=2)

    draw_ocr_text(draw, ocr_payload, small_font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def make_comparison(input_path: Path, reconstruction_path: Path, out_path: Path) -> None:
    source = Image.open(input_path).convert("RGB")
    reconstruction = Image.open(reconstruction_path).convert("RGB")
    thumb_w = 760
    label_h = 44
    pad = 24
    labels = [("Input original", source), ("V2 reconstructed recognition", reconstruction)]
    thumbs = []
    for label, image in labels:
        thumb_h = int(image.height * thumb_w / image.width)
        thumbs.append((label, image.resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)))
    height = max(image.height for _, image in thumbs)
    canvas = Image.new("RGB", (thumb_w * 2 + pad * 3, height + label_h + pad * 2), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", 28)
    except Exception:
        font = ImageFont.load_default()
    for idx, (label, image) in enumerate(thumbs):
        x = pad + idx * (thumb_w + pad)
        y = pad
        draw.rectangle([x, y, x + thumb_w, y + label_h], fill=(28, 32, 38))
        draw.text((x + 12, y + 8), label, fill="white", font=font)
        canvas.paste(image, (x, y + label_h))
        draw.rectangle([x, y + label_h, x + thumb_w, y + label_h + image.height], outline=(170, 170, 170), width=1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Reconstruct a score-like image from V2 symbol JSON.")
    parser.add_argument("--symbols-json", type=Path, required=True)
    parser.add_argument("--ocr-json", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--comparison", type=Path)
    parser.add_argument("--render-mode", choices=("symbolic", "cutout"), default="symbolic")
    parser.add_argument("--cutout-black-threshold", type=int, default=210)
    args = parser.parse_args()

    symbols_payload = read_json(args.symbols_json)
    ocr_payload = read_json(args.ocr_json) if args.ocr_json else None
    draw_symbol_reconstruction(
        symbols_payload,
        ocr_payload,
        args.out,
        render_mode=args.render_mode,
        cutout_black_threshold=args.cutout_black_threshold,
    )
    if args.comparison:
        make_comparison(Path(symbols_payload["input"]), args.out, args.comparison)
    print(args.out)
    if args.comparison:
        print(args.comparison)


if __name__ == "__main__":
    main()
