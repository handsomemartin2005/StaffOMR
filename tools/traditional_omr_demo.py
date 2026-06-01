from __future__ import annotations

import argparse
import json
import math
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, JpegImagePlugin  # noqa: F401 - registers PDF/JPEG saving


DEFAULT_INPUT = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "lg2267728_v18"
    / "input"
    / "lg-2267728-aug-beethoven--page-2.png"
)
LOCAL_FALLBACK_INPUT = Path(__file__).resolve().parents[1] / "dataset" / "call of silence.pdf"
DEFAULT_OUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "traditional_omr_demo"


@dataclass
class StaffLine:
    y: float
    thickness: int
    x0: int
    x1: int
    strength: int


@dataclass
class Staff:
    index: int
    lines: list[float]
    x0: int
    x1: int
    y0: float
    y1: float
    space: float
    clef_bbox: list[int] | None
    clef_type: str | None
    scan_x0: int


@dataclass
class NoteHead:
    staff: int
    x: int
    y: float
    pitch_step: int
    kind: str
    score: float
    black_density: float
    center_density: float


@dataclass
class Stem:
    staff: int
    note_index: int
    x: int
    y0: int
    y1: int
    direction: str
    length: int
    score: float


@dataclass
class Barline:
    staff: int
    x: int
    y0: int
    y1: int
    kind: str
    score: float


@dataclass
class Rest:
    staff: int
    bbox: list[int]
    kind: str
    score: float
    density: float


@dataclass
class Beam:
    staff: int
    bbox: list[int]
    kind: str
    score: float
    density: float


@dataclass
class BeamLink:
    staff: int
    stem_indices: list[int]
    x0: int
    y0: int
    x1: int
    y1: int
    direction: str
    score: float
    hit_ratio: float
    density: float


@dataclass
class BeamGroup:
    staff: int
    direction: str
    stem_indices: list[int]
    link_indices: list[int]
    points: list[list[int]]
    bbox: list[int]
    slope: float
    score: float
    hit_ratio: float
    density: float


@dataclass
class LedgerLine:
    staff: int
    note_index: int
    bbox: list[int]
    side: str
    line_number: int
    score: float


@dataclass
class Accidental:
    staff: int
    bbox: list[int]
    kind: str
    score: float
    density: float


@dataclass
class TextRegion:
    staff: int
    bbox: list[int]
    text: str
    score: float
    component_count: int


@dataclass
class SlurTie:
    staff: int
    bbox: list[int]
    kind: str
    score: float
    density: float


def otsu_threshold(gray: np.ndarray) -> int:
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    total = gray.size
    sum_total = np.dot(np.arange(256), hist)
    sum_b = 0.0
    weight_b = 0.0
    best_var = -1.0
    best_t = 180
    for t in range(256):
        weight_b += hist[t]
        if weight_b == 0:
            continue
        weight_f = total - weight_b
        if weight_f == 0:
            break
        sum_b += t * hist[t]
        mean_b = sum_b / weight_b
        mean_f = (sum_total - sum_b) / weight_f
        between = weight_b * weight_f * (mean_b - mean_f) ** 2
        if between > best_var:
            best_var = between
            best_t = t
    return int(best_t)


def load_input_image(path: Path, dpi: int = 220) -> Image.Image:
    if path.suffix.lower() == ".pdf":
        from pdf2image import convert_from_path

        return convert_from_path(str(path), dpi=dpi, first_page=1, last_page=1)[0].convert("RGB")
    return Image.open(path).convert("RGB")


def binarize(gray: np.ndarray) -> tuple[np.ndarray, int]:
    threshold = min(212, max(170, otsu_threshold(gray) + 35))
    mask = gray < threshold
    mask = remove_small_components(mask, min_area=3)
    return mask, threshold


def remove_small_components(mask: np.ndarray, min_area: int = 3) -> np.ndarray:
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    cleaned = mask.copy()
    for y in range(height):
        xs = np.where(mask[y] & ~seen[y])[0]
        for x in xs:
            if seen[y, x] or not mask[y, x]:
                continue
            q: deque[tuple[int, int]] = deque([(y, int(x))])
            seen[y, x] = True
            pixels: list[tuple[int, int]] = []
            while q:
                cy, cx = q.popleft()
                pixels.append((cy, cx))
                for ny in (cy - 1, cy, cy + 1):
                    for nx in (cx - 1, cx, cx + 1):
                        if ny == cy and nx == cx:
                            continue
                        if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True
                            q.append((ny, nx))
            if len(pixels) < min_area:
                for py, px in pixels:
                    cleaned[py, px] = False
    return cleaned


def save_binary(mask: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.where(mask, 0, 255).astype(np.uint8), mode="L").save(out_path)


def longest_true_run(values: np.ndarray) -> tuple[int, int]:
    best_start = 0
    best_end = -1
    start = None
    for i, value in enumerate(values):
        if value and start is None:
            start = i
        elif not value and start is not None:
            if i - 1 - start > best_end - best_start:
                best_start, best_end = start, i - 1
            start = None
    if start is not None and len(values) - 1 - start > best_end - best_start:
        best_start, best_end = start, len(values) - 1
    return best_start, best_end


def cluster_rows(rows: np.ndarray, max_gap: int = 1) -> list[tuple[int, int]]:
    groups: list[tuple[int, int]] = []
    start = None
    prev = None
    for row in rows:
        row_i = int(row)
        if start is None:
            start = prev = row_i
        elif row_i - int(prev) <= max_gap:
            prev = row_i
        else:
            groups.append((int(start), int(prev)))
            start = prev = row_i
    if start is not None:
        groups.append((int(start), int(prev)))
    return groups


def find_staff_line_candidates(mask: np.ndarray) -> list[StaffLine]:
    height, width = mask.shape
    projection = mask.sum(axis=1)
    strong = np.percentile(projection, 99.5)
    threshold = max(width * 0.22, strong * 0.62)
    row_groups = cluster_rows(np.where(projection > threshold)[0], max_gap=1)

    lines: list[StaffLine] = []
    for y0, y1 in row_groups:
        rows = mask[y0 : y1 + 1, :]
        col_hits = rows.sum(axis=0) > 0
        x0, x1 = longest_true_run(col_hits)
        if x1 <= x0:
            continue
        run = x1 - x0 + 1
        if run < width * 0.35:
            continue
        weights = projection[y0 : y1 + 1].astype(np.float64)
        ys = np.arange(y0, y1 + 1, dtype=np.float64)
        center = float(np.dot(ys, weights) / max(1.0, weights.sum()))
        lines.append(
            StaffLine(
                y=center,
                thickness=y1 - y0 + 1,
                x0=int(x0),
                x1=int(x1),
                strength=int(weights.max()),
            )
        )
    return sorted(lines, key=lambda item: item.y)


def estimate_staff_space(lines: list[StaffLine]) -> float:
    ys = np.array([line.y for line in lines], dtype=np.float64)
    diffs = np.diff(ys)
    useful = diffs[(diffs >= 6) & (diffs <= 80)]
    if useful.size == 0:
        raise RuntimeError("Could not estimate staff spacing from horizontal lines.")
    rounded = np.rint(useful).astype(int)
    values, counts = np.unique(rounded, return_counts=True)
    mode = values[np.argmax(counts)]
    near = useful[np.abs(useful - mode) <= 2.0]
    return float(np.median(near if near.size else useful))


def group_staves(lines: list[StaffLine], space: float) -> list[Staff]:
    staves: list[Staff] = []
    tol = max(2.4, space * 0.22)
    i = 0
    while i <= len(lines) - 5:
        chunk = lines[i : i + 5]
        diffs = np.diff([line.y for line in chunk])
        if np.all(np.abs(diffs - space) <= tol):
            x0 = int(np.median([line.x0 for line in chunk]))
            x1 = int(np.median([line.x1 for line in chunk]))
            ys = [float(line.y) for line in chunk]
            staves.append(
                Staff(
                    index=len(staves),
                    lines=ys,
                    x0=x0,
                    x1=x1,
                    y0=ys[0],
                    y1=ys[-1],
                    space=space,
                    clef_bbox=None,
                    clef_type=None,
                    scan_x0=x0 + int(round(4.2 * space)),
                )
            )
            i += 5
        else:
            i += 1
    return staves


def remove_staff_lines(mask: np.ndarray, staves: list[Staff]) -> np.ndarray:
    cleaned = mask.copy()
    height, width = mask.shape
    for staff in staves:
        band = max(1, int(round(staff.space * 0.10)))
        for y in staff.lines:
            y0 = max(0, int(round(y)) - band)
            y1 = min(height, int(round(y)) + band + 1)
            x0 = max(0, staff.x0 - int(round(staff.space)))
            x1 = min(width, staff.x1 + int(round(staff.space)))
            cleaned[y0:y1, x0:x1] = False
    return cleaned


def find_long_run_mask(mask: np.ndarray, staves: list[Staff]) -> np.ndarray:
    run_mask = np.zeros_like(mask, dtype=bool)
    height, width = mask.shape
    for staff in staves:
        space = staff.space
        min_run = max(18, int(round(1.75 * space)))
        y0 = max(0, int(round(staff.y0 - 5.0 * space)))
        y1 = min(height, int(round(staff.y1 + 5.0 * space)))
        x0 = max(0, int(round(staff.scan_x0 - 2.0 * space)))
        x1 = min(width, int(round(staff.x1 + 1.0 * space)))
        for y in range(y0, y1):
            row = mask[y, x0:x1]
            xs = np.where(row)[0]
            for start, end in cluster_rows(xs, max_gap=1):
                run_len = end - start + 1
                if run_len >= min_run:
                    yy0 = max(0, y - 1)
                    yy1 = min(height, y + 2)
                    run_mask[yy0:yy1, x0 + start : x0 + end + 1] = True
    return run_mask


def components_in_roi(mask: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> list[list[int]]:
    height, width = mask.shape
    x0, x1 = max(0, x0), min(width, x1)
    y0, y1 = max(0, y0), min(height, y1)
    roi = mask[y0:y1, x0:x1]
    seen = np.zeros_like(roi, dtype=bool)
    boxes: list[list[int]] = []
    h, w = roi.shape
    for yy in range(h):
        xs = np.where(roi[yy] & ~seen[yy])[0]
        for xx in xs:
            if seen[yy, xx] or not roi[yy, xx]:
                continue
            q: deque[tuple[int, int]] = deque([(yy, int(xx))])
            seen[yy, xx] = True
            min_x = max_x = int(xx)
            min_y = max_y = yy
            area = 0
            while q:
                cy, cx = q.popleft()
                area += 1
                min_x = min(min_x, cx)
                max_x = max(max_x, cx)
                min_y = min(min_y, cy)
                max_y = max(max_y, cy)
                for ny in (cy - 1, cy, cy + 1):
                    for nx in (cx - 1, cx, cx + 1):
                        if ny == cy and nx == cx:
                            continue
                        if 0 <= ny < h and 0 <= nx < w and roi[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True
                            q.append((ny, nx))
            boxes.append([x0 + min_x, y0 + min_y, x0 + max_x + 1, y0 + max_y + 1, area])
    return boxes


def detect_clefs(cleaned: np.ndarray, staves: list[Staff]) -> None:
    for staff in staves:
        space = staff.space
        roi_x0 = int(round(staff.x0 - 0.25 * space))
        roi_x1 = int(round(staff.x0 + 4.2 * space))
        roi_y0 = int(round(staff.y0 - 3.2 * space))
        roi_y1 = int(round(staff.y1 + 3.2 * space))
        boxes = components_in_roi(cleaned, roi_x0, roi_y0, roi_x1, roi_y1)
        candidates = []
        union_boxes = []
        for x0, y0, x1, y1, area in boxes:
            bw = x1 - x0
            bh = y1 - y0
            if x1 < staff.x0 - 0.05 * space:
                continue
            if area >= 0.12 * space * space and bw >= 0.18 * space and bh >= 0.45 * space:
                union_boxes.append([x0, y0, x1, y1, area])
            if bh >= 2.0 * space and bw >= 0.55 * space and area >= 0.55 * space * space:
                candidates.append([x0, y0, x1, y1, area])
        if candidates:
            best = max(candidates, key=lambda b: b[4])
            staff.clef_bbox = [int(best[0]), int(best[1]), int(best[2]), int(best[3])]
            staff.scan_x0 = max(staff.scan_x0, int(best[2] + 1.1 * space))
        elif union_boxes:
            x0 = min(box[0] for box in union_boxes)
            y0 = min(box[1] for box in union_boxes)
            x1 = max(box[2] for box in union_boxes)
            y1 = max(box[3] for box in union_boxes)
            if (x1 - x0) >= 0.75 * space and (y1 - y0) >= 1.6 * space:
                staff.clef_bbox = [int(x0), int(y0), int(x1), int(y1)]
                staff.scan_x0 = max(staff.scan_x0, int(x1 + 1.1 * space))
        staff.clef_type = "treble" if staff.index % 2 == 0 else "bass"


def bbox_intersects(a: list[int], b: list[int], margin: int = 0) -> bool:
    return min(a[2], b[2] + margin) > max(a[0], b[0] - margin) and min(a[3], b[3] + margin) > max(a[1], b[1] - margin)


def bbox_area(bbox: list[int]) -> int:
    return max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])


def merged_bbox(boxes: list[list[int]]) -> list[int]:
    return [
        int(min(box[0] for box in boxes)),
        int(min(box[1] for box in boxes)),
        int(max(box[2] for box in boxes)),
        int(max(box[3] for box in boxes)),
    ]


def merge_nearby_boxes(boxes: list[list[int]], pad_x: float, pad_y: float) -> list[list[list[int]]]:
    groups: list[list[list[int]]] = []
    for box in sorted(boxes, key=lambda item: (item[0], item[1])):
        placed = False
        for group in groups:
            gx0, gy0, gx1, gy1 = merged_bbox(group)
            expanded = [int(gx0 - pad_x), int(gy0 - pad_y), int(gx1 + pad_x), int(gy1 + pad_y)]
            if bbox_intersects(box[:4], expanded):
                group.append(box)
                placed = True
                break
        if not placed:
            groups.append([box])

    changed = True
    while changed:
        changed = False
        merged: list[list[list[int]]] = []
        while groups:
            group = groups.pop(0)
            gx0, gy0, gx1, gy1 = merged_bbox(group)
            expanded = [int(gx0 - pad_x), int(gy0 - pad_y), int(gx1 + pad_x), int(gy1 + pad_y)]
            i = 0
            while i < len(groups):
                if bbox_intersects(expanded, merged_bbox(groups[i])):
                    group.extend(groups.pop(i))
                    gx0, gy0, gx1, gy1 = merged_bbox(group)
                    expanded = [int(gx0 - pad_x), int(gy0 - pad_y), int(gx1 + pad_x), int(gy1 + pad_y)]
                    changed = True
                else:
                    i += 1
            merged.append(group)
        groups = merged
    return groups


def count_projection_groups(crop: np.ndarray, axis: int, threshold: int, max_gap: int = 1) -> int:
    if crop.size == 0:
        return 0
    values = crop.sum(axis=axis)
    hits = np.where(values >= threshold)[0]
    return len(cluster_rows(hits, max_gap=max_gap))


def classify_accidental_crop(crop: np.ndarray, bbox: list[int], area: int, space: float) -> tuple[str | None, float, float]:
    bw = bbox[2] - bbox[0]
    bh = bbox[3] - bbox[1]
    if bw <= 0 or bh <= 0:
        return None, 0.0, 0.0
    density = area / max(1, bw * bh)
    if not (0.35 * space <= bw <= 2.15 * space and 0.85 * space <= bh <= 3.65 * space):
        return None, 0.0, density
    if not (0.08 <= density <= 0.82):
        return None, 0.0, density

    row_groups = count_projection_groups(crop, axis=1, threshold=max(2, int(round(0.34 * bw))), max_gap=1)
    col_groups = count_projection_groups(crop, axis=0, threshold=max(2, int(round(0.26 * bh))), max_gap=1)
    tall_cols = int((crop.sum(axis=0) >= max(2, int(round(0.48 * bh)))).sum())
    lower = crop[crop.shape[0] // 2 :, :]
    upper = crop[: crop.shape[0] // 2, :]
    lower_density = float(lower.mean()) if lower.size else 0.0
    upper_density = float(upper.mean()) if upper.size else 0.0

    sharp_like = (
        bh >= 1.65 * space
        and 0.50 * space <= bw <= 1.55 * space
        and row_groups >= 3
        and tall_cols >= 4
    )
    if sharp_like and (col_groups >= 1 or density >= 0.32):
        score = min(1.0, 0.45 + 0.10 * row_groups + 0.10 * col_groups + 0.25 * density)
        return "sharp", float(score), density
    if bh >= 1.65 * space and row_groups >= 2 and col_groups >= 2 and tall_cols >= 3 and density >= 0.18:
        score = min(1.0, 0.34 + 0.09 * row_groups + 0.12 * col_groups + 0.22 * density)
        return "natural", float(score), density
    return None, 0.0, density


def detect_accidentals(cleaned: np.ndarray, staves: list[Staff], text_regions: list[TextRegion] | None = None) -> list[Accidental]:
    accidentals: list[Accidental] = []
    text_regions = text_regions or []
    height, width = cleaned.shape
    for staff in staves:
        space = staff.space
        x0 = max(0, int(round(staff.scan_x0 - 0.7 * space)))
        x1 = min(width, int(round(staff.x1 + 0.25 * space)))
        y0 = max(0, int(round(staff.y0 - 3.2 * space)))
        y1 = min(height, int(round(staff.y1 + 3.2 * space)))
        all_boxes = components_in_roi(cleaned, x0, y0, x1, y1)
        parts = []
        for box in all_boxes:
            bx0, by0, bx1, by1, area = box
            bw = bx1 - bx0
            bh = by1 - by0
            if area < max(5, int(0.025 * space * space)):
                continue
            if bw > 2.4 * space or bh > 3.9 * space:
                continue
            if staff.clef_bbox is not None and bbox_intersects([bx0, by0, bx1, by1], staff.clef_bbox, margin=int(round(0.2 * space))):
                continue
            if any(bbox_intersects([bx0, by0, bx1, by1], region.bbox, margin=int(round(0.15 * space))) for region in text_regions):
                continue
            parts.append(box)
        for group in merge_nearby_boxes(parts, pad_x=0.28 * space, pad_y=0.38 * space):
            bbox = merged_bbox(group)
            area = int(sum(box[4] for box in group))
            crop = cleaned[bbox[1] : bbox[3], bbox[0] : bbox[2]]
            kind, score, density = classify_accidental_crop(crop, bbox, area, space)
            if kind is None:
                continue
            if any(bbox_intersects(bbox, region.bbox, margin=int(round(0.15 * space))) for region in text_regions):
                continue
            if any(near_text_component_cluster(box, all_boxes, staff) for box in group):
                continue
            cx = 0.5 * (bbox[0] + bbox[2])
            cy = 0.5 * (bbox[1] + bbox[3])
            if cx < staff.scan_x0 - 0.8 * space or cx > staff.x1 + 0.3 * space:
                continue
            if nearest_pitch_distance(staff, cy) > 1.35 * space:
                continue
            accidentals.append(Accidental(staff.index, bbox, kind, score, density))
    return non_max_suppress_boxes(accidentals, key=lambda item: item.score, margin=4)


def near_text_component_cluster(box: list[int], boxes: list[list[int]], staff: Staff) -> bool:
    if not is_text_component(box, staff):
        return False
    space = staff.space
    cy = 0.5 * (box[1] + box[3])
    neighbors = 0
    for other in boxes:
        if other is box:
            continue
        if not is_text_component(other, staff):
            continue
        oy = 0.5 * (other[1] + other[3])
        if abs(cy - oy) > 0.68 * space:
            continue
        gap = max(0, max(other[0] - box[2], box[0] - other[2]))
        if gap <= 1.75 * space:
            neighbors += 1
    return neighbors >= 2


def mask_regions(shape: tuple[int, int], regions: list, pad_x: int = 0, pad_y: int = 0) -> np.ndarray:
    height, width = shape
    mask = np.zeros(shape, dtype=bool)
    for region in regions:
        x0, y0, x1, y1 = region.bbox
        x0 = max(0, int(x0 - pad_x))
        x1 = min(width, int(x1 + pad_x))
        y0 = max(0, int(y0 - pad_y))
        y1 = min(height, int(y1 + pad_y))
        mask[y0:y1, x0:x1] = True
    return mask


def choose_nearest_staff(staves: list[Staff], y: float) -> Staff:
    return min(staves, key=lambda staff: min(abs(y - staff.y0), abs(y - staff.y1), abs(y - 0.5 * (staff.y0 + staff.y1))))


def is_text_component(box: list[int], staff: Staff) -> bool:
    x0, y0, x1, y1, area = box
    bw = x1 - x0
    bh = y1 - y0
    if bw <= 0 or bh <= 0:
        return False
    space = staff.space
    density = area / max(1, bw * bh)
    if not (0.10 * space <= bw <= 1.85 * space and 0.25 * space <= bh <= 1.75 * space):
        return False
    if not (0.08 <= density <= 0.82):
        return False
    if bw <= 0.22 * space and bh >= 1.45 * space:
        return False
    return True


def recognize_text_region(mask: np.ndarray, bbox: list[int], space: float) -> tuple[str, float]:
    candidates = ("To Coda", "D.S. al Coda", "Final Coda", "al Coda", "8va")
    fonts = [
        Path("C:/Windows/Fonts/times.ttf"),
        Path("C:/Windows/Fonts/georgia.ttf"),
        Path("C:/Windows/Fonts/arial.ttf"),
    ]
    crop = mask[bbox[1] : bbox[3], bbox[0] : bbox[2]]
    if crop.size == 0:
        return "text", 0.0
    best_label = "text"
    best_score = 0.0
    target_size = (max(1, crop.shape[1]), max(1, crop.shape[0]))
    crop_aspect = target_size[0] / max(1, target_size[1])
    for label in candidates:
        if label == "8va" and target_size[0] > 5.2 * space:
            continue
        if label != "8va" and target_size[0] < 4.2 * space:
            continue
        for font_path in fonts:
            if not font_path.exists():
                continue
            for size in range(max(8, int(round(0.65 * space))), max(11, int(round(1.75 * space))) + 1, 2):
                try:
                    font = ImageFont.truetype(str(font_path), size)
                except OSError:
                    continue
                probe = Image.new("L", (1, 1), 255)
                probe_draw = ImageDraw.Draw(probe)
                tb = probe_draw.textbbox((0, 0), label, font=font)
                tw = max(1, tb[2] - tb[0])
                th = max(1, tb[3] - tb[1])
                template_img = Image.new("L", (tw + 6, th + 6), 255)
                template_draw = ImageDraw.Draw(template_img)
                template_draw.text((3 - tb[0], 3 - tb[1]), label, font=font, fill=0)
                template = np.array(template_img) < 190
                ys, xs = np.where(template)
                if xs.size == 0:
                    continue
                template = template[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
                template_aspect = template.shape[1] / max(1, template.shape[0])
                aspect_score = min(crop_aspect, template_aspect) / max(crop_aspect, template_aspect)
                if aspect_score < 0.42:
                    continue
                rendered = Image.fromarray(np.where(template, 0, 255).astype(np.uint8), mode="L").resize(
                    (target_size[0], target_size[1]), Image.Resampling.NEAREST
                )
                tmpl = np.array(rendered) < 190
                inter = int((crop & tmpl).sum())
                denom = int(crop.sum() + tmpl.sum() - inter)
                score = (inter / max(1, denom)) * aspect_score
                if score > best_score:
                    best_label = label
                    best_score = float(score)
    if best_score >= 0.16:
        return best_label, best_score
    return "text", best_score


def detect_text_regions(cleaned: np.ndarray, staves: list[Staff], accidentals: list[Accidental]) -> list[TextRegion]:
    height, width = cleaned.shape
    accidental_mask = mask_regions(cleaned.shape, accidentals, pad_x=2, pad_y=2)
    source = cleaned & ~accidental_mask
    text_regions: list[TextRegion] = []
    seen_boxes: list[list[int]] = []
    for staff in staves:
        space = staff.space
        x0 = max(0, int(round(staff.scan_x0 - 0.5 * space)))
        x1 = min(width, int(round(staff.x1 + 0.5 * space)))
        y0 = max(0, int(round(staff.y0 - 4.3 * space)))
        y1 = min(height, int(round(staff.y1 + 4.3 * space)))
        boxes = [box for box in components_in_roi(source, x0, y0, x1, y1) if is_text_component(box, staff)]
        if not boxes:
            continue
        boxes = sorted(boxes, key=lambda b: (0.5 * (b[1] + b[3]), b[0]))
        rows: list[list[list[int]]] = []
        for box in boxes:
            cy = 0.5 * (box[1] + box[3])
            placed = False
            for row in rows:
                row_cy = np.median([0.5 * (b[1] + b[3]) for b in row])
                if abs(cy - row_cy) <= 0.58 * space:
                    row.append(box)
                    placed = True
                    break
            if not placed:
                rows.append([box])

        for row in rows:
            row = sorted(row, key=lambda b: b[0])
            phrase: list[list[int]] = []
            for box in row:
                if not phrase:
                    phrase = [box]
                    continue
                prev = phrase[-1]
                gap = box[0] - prev[2]
                vertical_overlap = min(box[3], prev[3]) - max(box[1], prev[1])
                if gap <= 1.35 * space and vertical_overlap >= -0.25 * space:
                    phrase.append(box)
                else:
                    region = make_text_region_from_phrase(source, phrase, staff)
                    if region is not None:
                        text_regions.append(region)
                    phrase = [box]
            region = make_text_region_from_phrase(source, phrase, staff)
            if region is not None:
                text_regions.append(region)

    kept: list[TextRegion] = []
    for region in sorted(text_regions, key=lambda item: item.score, reverse=True):
        if any(bbox_intersects(region.bbox, other, margin=3) for other in seen_boxes):
            continue
        seen_boxes.append(region.bbox)
        kept.append(region)
    return sorted(kept, key=lambda item: (item.staff, item.bbox[1], item.bbox[0]))


def make_text_region_from_phrase(mask: np.ndarray, phrase: list[list[int]], staff: Staff) -> TextRegion | None:
    if len(phrase) < 3:
        return None
    bbox = merged_bbox(phrase)
    space = staff.space
    bw = bbox[2] - bbox[0]
    bh = bbox[3] - bbox[1]
    if bw < 1.8 * space or bh > 2.15 * space:
        return None
    center_y = 0.5 * (bbox[1] + bbox[3])
    in_staff_core = staff.y0 - 0.35 * space <= center_y <= staff.y1 + 0.35 * space
    if in_staff_core and nearest_pitch_distance(staff, center_y) <= 0.22 * space:
        return None
    label, match_score = recognize_text_region(mask, bbox, space)
    compactness = min(1.0, bw / max(1.0, 5.0 * space))
    score = max(match_score, 0.34 + 0.08 * min(5, len(phrase)) + 0.20 * compactness)
    return TextRegion(staff=staff.index, bbox=[int(v) for v in bbox], text=label, score=float(min(1.0, score)), component_count=len(phrase))


def make_notehead_masks(space: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rx = max(5, int(round(space * 0.78)))
    ry = max(4, int(round(space * 0.50)))
    yy, xx = np.mgrid[-ry : ry + 1, -rx : rx + 1]
    outer = (xx / rx) ** 2 + (yy / ry) ** 2 <= 1.0
    inner = (xx / max(1, rx * 0.50)) ** 2 + (yy / max(1, ry * 0.48)) ** 2 <= 1.0
    surround = ((xx / max(1, rx * 1.35)) ** 2 + (yy / max(1, ry * 1.35)) ** 2 <= 1.0) & ~outer
    ring = outer & ~inner
    left_ring = ring & (xx <= -0.18 * rx)
    right_ring = ring & (xx >= 0.18 * rx)
    top_ring = ring & (yy <= -0.15 * ry)
    bottom_ring = ring & (yy >= 0.15 * ry)
    return outer, inner, surround, left_ring, right_ring, top_ring, bottom_ring


def legal_pitch_centers(staff: Staff) -> list[tuple[int, float]]:
    half = staff.space / 2.0
    top = staff.lines[0]
    # Extended range covers ledger-line notes above and below the staff.
    return [(step, top + step * half) for step in range(-10, 19)]


def score_notehead(
    cleaned: np.ndarray,
    cx: int,
    cy: float,
    outer: np.ndarray,
    inner: np.ndarray,
    surround: np.ndarray,
    left_ring: np.ndarray,
    right_ring: np.ndarray,
    top_ring: np.ndarray,
    bottom_ring: np.ndarray,
) -> tuple[str | None, float, float, float]:
    ry = outer.shape[0] // 2
    rx = outer.shape[1] // 2
    y = int(round(cy))
    if y - ry < 0 or y + ry + 1 > cleaned.shape[0] or cx - rx < 0 or cx + rx + 1 > cleaned.shape[1]:
        return None, 0.0, 0.0, 0.0
    crop = cleaned[y - ry : y + ry + 1, cx - rx : cx + rx + 1]
    outer_density = float(crop[outer].mean())
    inner_density = float(crop[inner].mean())
    surround_density = float(crop[surround].mean()) if surround.any() else 0.0
    ring_density = float(crop[outer & ~inner].mean())
    outer_black = crop & outer
    row_hits = int((outer_black.sum(axis=1) >= max(2, int(0.18 * outer.shape[1]))).sum())
    col_hits = int((outer_black.sum(axis=0) >= max(2, int(0.18 * outer.shape[0]))).sum())
    row_coverage = row_hits / max(1, outer.shape[0])
    col_coverage = col_hits / max(1, outer.shape[1])
    left_density = float(crop[left_ring].mean()) if left_ring.any() else 0.0
    right_density = float(crop[right_ring].mean()) if right_ring.any() else 0.0
    top_density = float(crop[top_ring].mean()) if top_ring.any() else 0.0
    bottom_density = float(crop[bottom_ring].mean()) if bottom_ring.any() else 0.0
    side_balance = min(left_density, right_density) / max(left_density, right_density, 1e-6)

    filled_score = outer_density - 0.38 * surround_density
    open_score = 0.72 * ring_density + 0.30 * max(0.0, 0.35 - inner_density) - 0.22 * surround_density
    if (
        filled_score >= 0.36
        and inner_density >= 0.30
        and row_coverage >= 0.30
        and col_coverage >= 0.30
        and left_density >= 0.20
        and right_density >= 0.20
        and side_balance >= 0.25
    ):
        return "filled", filled_score, outer_density, inner_density
    if (
        open_score >= 0.31
        and ring_density >= 0.31
        and inner_density <= 0.28
        and row_coverage >= 0.30
        and col_coverage >= 0.30
        and left_density >= 0.23
        and right_density >= 0.23
        and max(top_density, bottom_density) >= 0.18
        and side_balance >= 0.30
    ):
        return "open", open_score, outer_density, inner_density
    if (
        filled_score >= 0.34
        and outer_density >= 0.34
        and inner_density >= 0.30
        and row_coverage >= 0.30
        and col_coverage >= 0.30
        and max(left_density, right_density) >= 0.20
        and surround_density <= 0.42
        and side_balance >= 0.24
    ):
        return "filled", filled_score, outer_density, inner_density
    return None, max(filled_score, open_score), outer_density, inner_density


def refine_notehead_kind(
    cleaned: np.ndarray,
    raw_mask: np.ndarray | None,
    cx: int,
    cy: float,
    outer: np.ndarray,
    inner: np.ndarray,
) -> tuple[str, float, float]:
    ry = outer.shape[0] // 2
    rx = outer.shape[1] // 2
    y = int(round(cy))
    if y - ry < 0 or y + ry + 1 > cleaned.shape[0] or cx - rx < 0 or cx + rx + 1 > cleaned.shape[1]:
        return "filled", 0.0, 0.0
    crop = cleaned[y - ry : y + ry + 1, cx - rx : cx + rx + 1]
    ring = outer & ~inner
    yy, xx = np.mgrid[-ry : ry + 1, -rx : rx + 1]
    core = (xx / max(1, rx * 0.34)) ** 2 + (yy / max(1, ry * 0.30)) ** 2 <= 1.0
    outer_density = float(crop[outer].mean())
    ring_density = float(crop[ring].mean())
    inner_density = float(crop[inner].mean())
    core_density = float(crop[core].mean()) if core.any() else inner_density
    ring_contrast = ring_density - core_density

    open_conf = 0.70 * ring_density + 0.45 * max(0.0, 0.28 - core_density) + 0.30 * max(0.0, ring_contrast)
    filled_conf = 0.70 * core_density + 0.35 * inner_density + 0.20 * outer_density
    if raw_mask is not None:
        raw_crop = raw_mask[y - ry : y + ry + 1, cx - rx : cx + rx + 1]
        raw_inner_density = float(raw_crop[inner].mean())
        raw_core_density = float(raw_crop[core].mean()) if core.any() else raw_inner_density
        raw_outer_density = float(raw_crop[outer].mean())
        raw_filled = raw_core_density >= 0.68 and raw_inner_density >= 0.58 and raw_outer_density >= 0.62
        cleaned_is_erased_by_staff = raw_core_density - core_density >= 0.42 or raw_inner_density - inner_density >= 0.42
        if raw_filled and cleaned_is_erased_by_staff:
            return "filled", raw_outer_density, raw_inner_density
    if open_conf >= 0.43 and core_density <= 0.24 and ring_density >= 0.31 and open_conf >= filled_conf * 0.92:
        return "open", outer_density, inner_density
    return "filled", outer_density, inner_density


def notehead_candidate_x_positions(cleaned: np.ndarray, staff: Staff, cy: float, step_x: int) -> list[int]:
    height, width = cleaned.shape
    space = staff.space
    x_start = max(0, int(round(staff.scan_x0)))
    x_end = min(width, int(round(staff.x1 - space)))
    candidates = set(range(x_start, x_end, step_x))

    y0 = max(0, int(round(cy - 0.72 * space)))
    y1 = min(height, int(round(cy + 0.72 * space)) + 1)
    if y1 <= y0:
        return sorted(candidates)

    band = cleaned[y0:y1, x_start:x_end]
    if band.size == 0:
        return sorted(candidates)
    col_counts = band.sum(axis=0)
    hits = np.where(col_counts >= max(2, int(round(0.10 * (y1 - y0)))))[0]
    for run_start, run_end in cluster_rows(hits, max_gap=2):
        run_w = run_end - run_start + 1
        if run_w < max(4, int(round(0.26 * space))) or run_w > int(round(3.2 * space)):
            continue
        weights = col_counts[run_start : run_end + 1].astype(np.float64)
        if weights.sum() <= 0:
            continue
        xs = np.arange(x_start + run_start, x_start + run_end + 1, dtype=np.float64)
        center = int(round(float(np.dot(xs, weights) / weights.sum())))
        for offset in (0, -0.18 * space, 0.18 * space):
            x = int(round(center + offset))
            if x_start <= x < x_end:
                candidates.add(x)
    return sorted(candidates)


def detect_noteheads(cleaned: np.ndarray, staves: list[Staff], raw_mask: np.ndarray | None = None) -> list[NoteHead]:
    detections: list[NoteHead] = []
    for staff in staves:
        outer, inner, surround, left_ring, right_ring, top_ring, bottom_ring = make_notehead_masks(staff.space)
        step_x = max(2, int(round(staff.space / 4.0)))
        for pitch_step, cy in legal_pitch_centers(staff):
            local: list[NoteHead] = []
            for cx in notehead_candidate_x_positions(cleaned, staff, cy, step_x):
                kind, score, outer_density, inner_density = score_notehead(
                    cleaned,
                    cx,
                    cy,
                    outer,
                    inner,
                    surround,
                    left_ring,
                    right_ring,
                    top_ring,
                    bottom_ring,
                )
                if kind is None:
                    continue
                kind, outer_density, inner_density = refine_notehead_kind(cleaned, raw_mask, cx, cy, outer, inner)
                local.append(
                    NoteHead(
                        staff=staff.index,
                        x=int(cx),
                        y=float(cy),
                        pitch_step=int(pitch_step),
                        kind=kind,
                        score=float(score),
                        black_density=float(outer_density),
                        center_density=float(inner_density),
                    )
                )
            detections.extend(non_max_suppress_notes(local, min_dx=max(5, int(round(staff.space * 0.72)))))
    return non_max_suppress_notes(detections, min_dx=8)


def non_max_suppress_notes(notes: list[NoteHead], min_dx: int) -> list[NoteHead]:
    kept: list[NoteHead] = []
    for note in sorted(notes, key=lambda n: n.score, reverse=True):
        conflict = False
        for other in kept:
            if note.staff != other.staff:
                continue
            if abs(note.x - other.x) <= min_dx and abs(note.y - other.y) <= max(4.0, 1.10 * min_dx):
                conflict = True
                break
        if not conflict:
            kept.append(note)
    return sorted(kept, key=lambda n: (n.staff, n.x, n.y))


def longest_vertical_run(column: np.ndarray, max_gap: int = 7) -> tuple[int, int, int]:
    best_start = 0
    best_end = -1
    start = None
    last_true = None
    gap = 0
    for i, value in enumerate(column):
        if value and start is None:
            start = i
            last_true = i
            gap = 0
        elif value and start is not None:
            last_true = i
            gap = 0
        elif not value and start is not None:
            gap += 1
            if gap > max_gap:
                end = int(last_true if last_true is not None else i - gap)
                if end - int(start) > best_end - best_start:
                    best_start, best_end = int(start), end
                start = None
                last_true = None
                gap = 0
    if start is not None:
        end = int(last_true if last_true is not None else len(column) - 1)
        if end - int(start) > best_end - best_start:
            best_start, best_end = int(start), end
    return best_start, best_end, max(0, best_end - best_start + 1)


def detect_stems(cleaned: np.ndarray, notes: list[NoteHead], staves: list[Staff]) -> list[Stem]:
    stems: list[Stem] = []
    height, width = cleaned.shape
    for idx, note in enumerate(notes):
        staff = staves[note.staff]
        space = staff.space
        min_len = int(round(space * 2.1))
        max_len = int(round(space * 4.8))
        y = int(round(note.y))
        best: Stem | None = None
        for direction in ("up", "down"):
            if direction == "up":
                x_candidates = range(
                    int(round(note.x + 0.25 * space)),
                    int(round(note.x + 1.10 * space)) + 1,
                )
                yy0 = max(0, y - max_len)
                yy1 = min(height, y + int(round(0.65 * space)))
            else:
                x_candidates = range(
                    int(round(note.x - 1.10 * space)),
                    int(round(note.x - 0.25 * space)) + 1,
                )
                yy0 = max(0, y - int(round(0.65 * space)))
                yy1 = min(height, y + max_len)
            for x in x_candidates:
                if x < 0 or x >= width or yy1 <= yy0:
                    continue
                start, end, length = longest_vertical_run(cleaned[yy0:yy1, x])
                if length < min_len:
                    continue
                y0 = yy0 + start
                y1 = yy0 + end
                score = length / max(1.0, max_len)
                candidate = Stem(
                    staff=note.staff,
                    note_index=idx,
                    x=int(x),
                    y0=int(y0),
                    y1=int(y1),
                    direction=direction,
                    length=int(length),
                    score=float(score),
                )
                if best is None or candidate.length > best.length:
                    best = candidate
        if best is not None:
            stems.append(best)
    return stems


def detect_ledger_lines(mask: np.ndarray, notes: list[NoteHead], staves: list[Staff]) -> list[LedgerLine]:
    ledger_lines: list[LedgerLine] = []
    height, width = mask.shape
    for note_idx, note in enumerate(notes):
        staff = staves[note.staff]
        space = staff.space
        side = None
        if note.y < staff.y0 - 0.45 * space:
            side = "above"
            positions = [
                staff.y0 - i * space
                for i in range(1, int(math.ceil((staff.y0 - note.y) / space)) + 2)
            ]
        elif note.y > staff.y1 + 0.45 * space:
            side = "below"
            positions = [
                staff.y1 + i * space
                for i in range(1, int(math.ceil((note.y - staff.y1) / space)) + 2)
            ]
        else:
            continue

        for line_number, y_line in enumerate(positions, start=1):
            if abs(y_line - note.y) > max(space * 1.25, abs(note.y - (staff.y0 if side == "above" else staff.y1)) + space):
                continue
            y = int(round(y_line))
            if y < 0 or y >= height:
                continue
            x0 = max(0, int(round(note.x - 1.45 * space)))
            x1 = min(width, int(round(note.x + 1.45 * space)))
            y0 = max(0, y - max(1, int(round(0.14 * space))))
            y1 = min(height, y + max(1, int(round(0.14 * space))) + 1)
            band = mask[y0:y1, x0:x1].any(axis=0)
            runs = cluster_rows(np.where(band)[0], max_gap=1)
            best_run = None
            best_len = 0
            for run_start, run_end in runs:
                abs_start = x0 + run_start
                abs_end = x0 + run_end
                crosses_note = abs_start <= note.x + 0.35 * space and abs_end >= note.x - 0.35 * space
                run_len = abs_end - abs_start + 1
                if crosses_note and run_len > best_len:
                    best_run = (abs_start, abs_end)
                    best_len = run_len
            if best_run is None:
                continue
            if not (0.75 * space <= best_len <= 3.0 * space):
                continue
            bx0, bx1 = best_run
            ledger_lines.append(
                LedgerLine(
                    staff=staff.index,
                    note_index=note_idx,
                    bbox=[int(bx0), int(y0), int(bx1), int(y1)],
                    side=side,
                    line_number=int(line_number),
                    score=float(min(1.0, best_len / max(1.0, 1.8 * space))),
                )
            )
    return non_max_suppress_ledger_lines(ledger_lines)


def non_max_suppress_ledger_lines(lines: list[LedgerLine]) -> list[LedgerLine]:
    kept: list[LedgerLine] = []
    for line in sorted(lines, key=lambda item: item.score, reverse=True):
        x0, y0, x1, y1 = line.bbox
        duplicate = False
        for other in kept:
            ox0, oy0, ox1, oy1 = other.bbox
            if line.staff != other.staff or line.side != other.side:
                continue
            if abs(0.5 * (x0 + x1) - 0.5 * (ox0 + ox1)) <= 8 and abs(0.5 * (y0 + y1) - 0.5 * (oy0 + oy1)) <= 4:
                duplicate = True
                break
        if not duplicate:
            kept.append(line)
    return sorted(kept, key=lambda item: (item.staff, item.note_index, item.line_number))


def detect_barlines(mask: np.ndarray, staves: list[Staff]) -> list[Barline]:
    raw_barlines: list[Barline] = []
    height, width = mask.shape
    for staff in staves:
        space = staff.space
        y0 = max(0, int(round(staff.y0 - 0.35 * space)))
        y1 = min(height, int(round(staff.y1 + 0.35 * space)))
        min_len = int(round(3.55 * space))
        x0 = max(0, int(round(staff.scan_x0 - 0.8 * space)))
        x1 = min(width, int(round(staff.x1 + 0.4 * space)))
        hits: list[tuple[int, int, int]] = []
        for x in range(x0, x1):
            run_start, run_end, run_len = longest_vertical_run(mask[y0:y1, x], max_gap=1)
            if run_len >= min_len:
                hits.append((x, y0 + run_start, y0 + run_end))
        if not hits:
            continue
        xs = np.array([hit[0] for hit in hits])
        groups = cluster_rows(xs, max_gap=2)
        for gx0, gx1 in groups:
            group_hits = [hit for hit in hits if gx0 <= hit[0] <= gx1]
            if not group_hits:
                continue
            x = int(round(np.median([hit[0] for hit in group_hits])))
            by0 = int(min(hit[1] for hit in group_hits))
            by1 = int(max(hit[2] for hit in group_hits))
            length = by1 - by0 + 1
            if length < min_len:
                continue
            kind = "system_start" if abs(x - staff.x0) <= 1.2 * space else "barline"
            raw_barlines.append(
                Barline(
                    staff=staff.index,
                    x=x,
                    y0=by0,
                    y1=by1,
                    kind=kind,
                    score=float(length / max(1.0, y1 - y0)),
                )
            )
    if len(staves) < 2:
        return raw_barlines

    filtered: list[Barline] = []
    for bar in raw_barlines:
        staff = staves[bar.staff]
        same_x = [
            other
            for other in raw_barlines
            if other.staff != bar.staff
            and abs(other.x - bar.x) <= max(2.0, 0.60 * staff.space)
            and abs(staves[other.staff].y0 - staff.y0) <= 13.0 * staff.space
        ]
        if same_x or bar.kind == "system_start":
            filtered.append(bar)
    return filtered if filtered else raw_barlines


def erase_detected_notes_and_stems(
    cleaned: np.ndarray,
    notes: list[NoteHead],
    stems: list[Stem],
    staves: list[Staff],
) -> np.ndarray:
    pruned = cleaned.copy()
    height, width = pruned.shape
    for note in notes:
        staff = staves[note.staff]
        rx = max(5, int(round(staff.space * 0.95)))
        ry = max(4, int(round(staff.space * 0.65)))
        cx = int(round(note.x))
        cy = int(round(note.y))
        x0 = max(0, cx - rx)
        x1 = min(width, cx + rx + 1)
        y0 = max(0, cy - ry)
        y1 = min(height, cy + ry + 1)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        ellipse = ((xx - cx) / max(1, rx)) ** 2 + ((yy - cy) / max(1, ry)) ** 2 <= 1.0
        pruned[y0:y1, x0:x1][ellipse] = False
    for stem in stems:
        x0 = max(0, stem.x - 2)
        x1 = min(width, stem.x + 3)
        y0 = max(0, stem.y0 - 2)
        y1 = min(height, stem.y1 + 3)
        pruned[y0:y1, x0:x1] = False
    return pruned


def bbox_overlaps_point_box(bbox: list[int], cx: float, cy: float, margin: float) -> bool:
    x0, y0, x1, y1 = bbox
    return (x0 - margin) <= cx <= (x1 + margin) and (y0 - margin) <= cy <= (y1 + margin)


def near_barline(bbox: list[int], barlines: list[Barline], staff_index: int, margin: float) -> bool:
    x0, _, x1, _ = bbox
    cx = 0.5 * (x0 + x1)
    return any(bar.staff == staff_index and abs(bar.x - cx) <= margin for bar in barlines)


def near_beam_group(bbox: list[int], beam_groups: list[BeamGroup], staff_index: int, margin: float) -> bool:
    x0, y0, x1, y1 = bbox
    for group in beam_groups:
        if group.staff != staff_index:
            continue
        gx0, gy0, gx1, gy1 = group.bbox
        if min(x1, gx1 + margin) >= max(x0, gx0 - margin) and min(y1, gy1 + margin) >= max(y0, gy0 - margin):
            return True
    return False


def nearest_pitch_distance(staff: Staff, y: float) -> float:
    return min(abs(y - pitch_y) for _, pitch_y in legal_pitch_centers(staff))


def near_stem(bbox: list[int], stems: list[Stem], staff_index: int, space: float) -> bool:
    x0, y0, x1, y1 = bbox
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    for stem in stems:
        if stem.staff != staff_index:
            continue
        x_close = x0 - 0.75 * space <= stem.x <= x1 + 0.75 * space
        y_overlap = min(stem.y1, y1 + 0.8 * space) >= max(stem.y0, y0 - 0.8 * space)
        if x_close and y_overlap:
            return True
        if abs(stem.x - cx) <= 0.45 * space and stem.y0 - space <= cy <= stem.y1 + space:
            return True
    return False


def looks_like_notehead_component(
    bbox: list[int],
    area: int,
    density: float,
    staff: Staff,
    stems: list[Stem],
) -> bool:
    x0, y0, x1, y1 = bbox
    bw = x1 - x0
    bh = y1 - y0
    if bw <= 0 or bh <= 0:
        return False
    space = staff.space
    cy = 0.5 * (y0 + y1)
    pitch_dist = nearest_pitch_distance(staff, cy)
    aspect = bw / max(1, bh)
    notehead_sized = 0.55 * space <= bw <= 1.75 * space and 0.38 * space <= bh <= 1.10 * space
    elliptical = 0.75 <= aspect <= 2.20 and density >= 0.42
    on_pitch_grid = pitch_dist <= 0.24 * space
    if notehead_sized and elliptical and on_pitch_grid:
        return True
    if notehead_sized and on_pitch_grid and near_stem(bbox, stems, staff.index, space):
        return True
    return False


def detect_secondary_symbols(
    pruned: np.ndarray,
    staves: list[Staff],
    notes: list[NoteHead],
    stems: list[Stem],
    beam_groups: list[BeamGroup],
    barlines: list[Barline],
) -> tuple[list[Rest], list[Beam], list[SlurTie]]:
    rests: list[Rest] = []
    beams: list[Beam] = []
    slurs: list[SlurTie] = []
    for staff in staves:
        space = staff.space
        roi_x0 = int(round(staff.scan_x0))
        roi_x1 = int(round(staff.x1 + space))
        roi_y0 = int(round(staff.y0 - 3.0 * space))
        roi_y1 = int(round(staff.y1 + 3.0 * space))
        boxes = components_in_roi(pruned, roi_x0, roi_y0, roi_x1, roi_y1)
        staff_notes = [note for note in notes if note.staff == staff.index]
        for x0, y0, x1, y1, area in boxes:
            bw = x1 - x0
            bh = y1 - y0
            if area < max(4, int(0.08 * space * space)):
                continue
            bbox = [int(x0), int(y0), int(x1), int(y1)]
            density = float(area / max(1, bw * bh))
            cx = 0.5 * (x0 + x1)
            cy = 0.5 * (y0 + y1)
            in_staff_body = staff.y0 - 1.15 * space <= cy <= staff.y1 + 1.15 * space
            if near_barline(bbox, barlines, staff.index, margin=0.45 * space):
                continue
            if near_beam_group(bbox, beam_groups, staff.index, margin=0.20 * space):
                continue
            if near_stem(bbox, stems, staff.index, space) and bw <= 2.2 * space:
                continue
            if looks_like_notehead_component(bbox, area, density, staff, stems):
                continue

            close_note = any(bbox_overlaps_point_box(bbox, note.x, note.y, margin=0.35 * space) for note in staff_notes)
            if not close_note and in_staff_body:
                rest_aspect = bw / max(1, bh)
                middle_rest_zone = staff.lines[1] - 0.35 * space <= cy <= staff.lines[3] + 0.35 * space
                if (
                    0.55 * space <= bw <= 2.2 * space
                    and 0.14 * space <= bh <= 0.52 * space
                    and rest_aspect >= 2.05
                    and density >= 0.35
                    and middle_rest_zone
                ):
                    kind = "whole_or_half_rest"
                    rests.append(Rest(staff.index, bbox, kind, float(density), density))
                    continue
                if 0.25 * space <= bw <= 1.45 * space and 0.9 * space <= bh <= 3.2 * space and density <= 0.62:
                    kind = "quarter_or_eighth_rest"
                    rests.append(Rest(staff.index, bbox, kind, float((area / (space * space)) / max(1.0, bh / space)), density))
                    continue

            if bw >= 1.45 * space and bh <= 0.95 * space and density >= 0.38:
                score = min(1.0, (bw / max(1.0, 4.0 * space)) * density)
                beams.append(Beam(staff.index, bbox, "beam_candidate", float(score), density))
                continue

            if bw >= 1.8 * space and 0.18 * space <= bh <= 1.7 * space and 0.035 <= density <= 0.36:
                score = min(1.0, (bw / max(1.0, 5.0 * space)) * (0.45 - density))
                kind = "tie_or_slur_candidate" if bw <= 6.5 * space else "slur_candidate"
                slurs.append(SlurTie(staff.index, bbox, kind, float(score), density))
    return (
        non_max_suppress_boxes(rests, key=lambda item: item.score, margin=3),
        non_max_suppress_boxes(beams, key=lambda item: item.score, margin=3),
        non_max_suppress_boxes(slurs, key=lambda item: item.score, margin=3),
    )


def stem_terminal(stem: Stem) -> tuple[int, int]:
    if stem.direction == "up":
        return stem.x, stem.y0
    return stem.x, stem.y1


def score_line_corridor(mask: np.ndarray, x0: int, y0: int, x1: int, y1: int, half_width: int) -> tuple[float, float]:
    if x1 == x0:
        return 0.0, 0.0
    if x1 < x0:
        x0, y0, x1, y1 = x1, y1, x0, y0
    width = mask.shape[1]
    height = mask.shape[0]
    dx = x1 - x0
    margin = max(2, int(round(dx * 0.08)))
    xs = np.arange(max(0, x0 + margin), min(width, x1 - margin + 1), dtype=np.int32)
    if xs.size < max(6, dx * 0.45):
        return 0.0, 0.0
    ys = y0 + (y1 - y0) * ((xs - x0) / max(1, dx))
    hits = 0
    samples = 0
    black = 0
    total = 0
    for x, y_float in zip(xs, ys):
        y = int(round(float(y_float)))
        yy0 = max(0, y - half_width)
        yy1 = min(height, y + half_width + 1)
        if yy1 <= yy0:
            continue
        stripe = mask[yy0:yy1, int(x)]
        samples += 1
        stripe_black = int(stripe.sum())
        if stripe_black:
            hits += 1
        black += stripe_black
        total += int(stripe.size)
    if samples == 0 or total == 0:
        return 0.0, 0.0
    return hits / samples, black / total


def detect_beam_links_from_stems(cleaned: np.ndarray, stems: list[Stem], staves: list[Staff]) -> list[BeamLink]:
    links: list[BeamLink] = []
    for staff in staves:
        space = staff.space
        for direction in ("up", "down"):
            indexed = [
                (idx, stem)
                for idx, stem in enumerate(stems)
                if stem.staff == staff.index and stem.direction == direction and stem.length >= 1.85 * space
            ]
            indexed.sort(key=lambda item: item[1].x)
            for left_pos, (left_idx, left) in enumerate(indexed):
                lx, ly = stem_terminal(left)
                best_link: BeamLink | None = None
                for right_idx, right in indexed[left_pos + 1 : left_pos + 5]:
                    rx, ry = stem_terminal(right)
                    dx = rx - lx
                    if dx < 1.10 * space:
                        continue
                    if dx > 7.50 * space:
                        break
                    dy = ry - ly
                    if abs(dy) > 2.60 * space:
                        continue
                    slope = dy / max(1, dx)
                    if abs(slope) > 0.85:
                        continue

                    half_width = max(2, int(round(0.22 * space)))
                    hit_ratio, density = score_line_corridor(cleaned, lx, ly, rx, ry, half_width)
                    if hit_ratio < 0.58 or density < 0.16:
                        # Some engravers join stems to the lower/upper edge of a thick beam.
                        # Try nearby parallel lines before rejecting.
                        offsets = (-0.38 * space, 0.38 * space, -0.72 * space, 0.72 * space)
                        best_hit, best_density = hit_ratio, density
                        best_offset = 0.0
                        for offset in offsets:
                            ohit, odensity = score_line_corridor(
                                cleaned,
                                lx,
                                int(round(ly + offset)),
                                rx,
                                int(round(ry + offset)),
                                half_width,
                            )
                            if ohit + odensity > best_hit + best_density:
                                best_hit, best_density = ohit, odensity
                                best_offset = offset
                        hit_ratio, density = best_hit, best_density
                        ly2 = int(round(ly + best_offset))
                        ry2 = int(round(ry + best_offset))
                    else:
                        ly2, ry2 = ly, ry

                    if hit_ratio >= 0.58 and density >= 0.16:
                        score = min(1.0, 0.72 * hit_ratio + 0.55 * density)
                        candidate = BeamLink(
                            staff=staff.index,
                            stem_indices=[int(left_idx), int(right_idx)],
                            x0=int(lx),
                            y0=int(ly2),
                            x1=int(rx),
                            y1=int(ry2),
                            direction=direction,
                            score=float(score),
                            hit_ratio=float(hit_ratio),
                            density=float(density),
                        )
                        if best_link is None or candidate.score > best_link.score:
                            best_link = candidate
                if best_link is not None:
                    links.append(best_link)
    return suppress_beam_links(links, staves)


def suppress_beam_links(links: list[BeamLink], staves: list[Staff]) -> list[BeamLink]:
    kept: list[BeamLink] = []
    for link in sorted(links, key=lambda item: item.score, reverse=True):
        staff = staves[link.staff]
        conflict = False
        for other in kept:
            if link.staff != other.staff or link.direction != other.direction:
                continue
            if abs(link.x0 - other.x0) <= 0.45 * staff.space and abs(link.x1 - other.x1) <= 0.45 * staff.space:
                conflict = True
                break
        if not conflict:
            kept.append(link)
    return sorted(kept, key=lambda item: (item.staff, item.x0, item.x1, item.y0))


def group_beam_links(links: list[BeamLink], stems: list[Stem], staves: list[Staff]) -> list[BeamGroup]:
    groups: list[BeamGroup] = []
    indexed_links = list(enumerate(links))
    for staff in staves:
        space = staff.space
        for direction in ("up", "down"):
            staff_stems = [
                (idx, stem)
                for idx, stem in enumerate(stems)
                if stem.staff == staff.index and stem.direction == direction
            ]
            staff_stems.sort(key=lambda item: stem_terminal(item[1])[0])
            rank = {idx: pos for pos, (idx, _) in enumerate(staff_stems)}
            candidates: list[tuple[float, int, BeamLink]] = []
            for link_idx, link in indexed_links:
                if link.staff != staff.index or link.direction != direction:
                    continue
                left_idx, right_idx = link.stem_indices
                if left_idx not in rank or right_idx not in rank:
                    continue
                rank_gap = rank[right_idx] - rank[left_idx]
                if rank_gap <= 0 or rank_gap > 2:
                    continue
                dx = max(1, link.x1 - link.x0)
                if dx > 4.8 * space:
                    continue
                slope = (link.y1 - link.y0) / dx
                if abs(slope) > 0.75:
                    continue
                continuity_penalty = 0.14 * (rank_gap - 1) + 0.015 * max(0.0, dx / space - 2.8)
                quality = link.score - continuity_penalty
                if quality < 0.58:
                    continue
                candidates.append((quality, link_idx, link))

            outgoing: dict[int, tuple[float, int, BeamLink]] = {}
            for quality, link_idx, link in sorted(candidates, key=lambda item: item[0], reverse=True):
                left_idx, _ = link.stem_indices
                if left_idx not in outgoing:
                    outgoing[left_idx] = (quality, link_idx, link)

            incoming: dict[int, tuple[float, int, BeamLink]] = {}
            for quality, link_idx, link in sorted(outgoing.values(), key=lambda item: item[0], reverse=True):
                _, right_idx = link.stem_indices
                if right_idx not in incoming:
                    incoming[right_idx] = (quality, link_idx, link)

            edges = {item[2].stem_indices[0]: item for item in incoming.values()}
            edge_lefts = set(edges.keys())
            edge_rights = {item[2].stem_indices[1] for item in edges.values()}
            starts = sorted(edge_lefts - edge_rights, key=lambda idx: rank.get(idx, 10**9))
            if not starts:
                starts = sorted(edge_lefts, key=lambda idx: rank.get(idx, 10**9))

            visited_edges: set[int] = set()
            for start in starts:
                path: list[tuple[int, BeamLink]] = []
                current = start
                seen_stems: set[int] = set()
                while current in edges and current not in seen_stems:
                    seen_stems.add(current)
                    _, link_idx, link = edges[current]
                    if link_idx in visited_edges:
                        break
                    path.append((link_idx, link))
                    visited_edges.add(link_idx)
                    current = link.stem_indices[1]
                groups.extend(split_beam_path(path, staff, slope_tolerance=0.22))

            for left_idx, (_, link_idx, link) in edges.items():
                if link_idx in visited_edges:
                    continue
                groups.extend(split_beam_path([(link_idx, link)], staff, slope_tolerance=0.22))
                visited_edges.add(link_idx)
    return groups


def split_beam_path(path: list[tuple[int, BeamLink]], staff: Staff, slope_tolerance: float) -> list[BeamGroup]:
    if not path:
        return []
    result: list[BeamGroup] = []
    current: list[tuple[int, BeamLink]] = []
    slopes: list[float] = []
    for item in path:
        _, link = item
        slope = (link.y1 - link.y0) / max(1, link.x1 - link.x0)
        if current and abs(slope - float(np.median(slopes))) > slope_tolerance:
            result.append(make_beam_group(current, staff))
            current = []
            slopes = []
        current.append(item)
        slopes.append(float(slope))
    if current:
        result.append(make_beam_group(current, staff))
    return result


def make_beam_group(items: list[tuple[int, BeamLink]], staff: Staff) -> BeamGroup:
    link_indices = [int(idx) for idx, _ in items]
    links = [link for _, link in items]
    stem_indices = [int(links[0].stem_indices[0])]
    points = [[int(links[0].x0), int(links[0].y0)]]
    for link in links:
        if stem_indices[-1] != int(link.stem_indices[0]):
            stem_indices.append(int(link.stem_indices[0]))
            points.append([int(link.x0), int(link.y0)])
        stem_indices.append(int(link.stem_indices[1]))
        points.append([int(link.x1), int(link.y1)])
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    slopes = [(link.y1 - link.y0) / max(1, link.x1 - link.x0) for link in links]
    bbox_pad = max(2, int(round(0.35 * staff.space)))
    return BeamGroup(
        staff=int(staff.index),
        direction=links[0].direction,
        stem_indices=stem_indices,
        link_indices=link_indices,
        points=points,
        bbox=[min(xs) - bbox_pad, min(ys) - bbox_pad, max(xs) + bbox_pad, max(ys) + bbox_pad],
        slope=float(np.median(slopes)),
        score=float(np.mean([link.score for link in links])),
        hit_ratio=float(np.mean([link.hit_ratio for link in links])),
        density=float(np.mean([link.density for link in links])),
    )


def non_max_suppress_boxes(items: list, key, margin: int = 2) -> list:
    kept = []
    for item in sorted(items, key=key, reverse=True):
        x0, y0, x1, y1 = item.bbox
        conflict = False
        for other in kept:
            if item.staff != other.staff:
                continue
            ox0, oy0, ox1, oy1 = other.bbox
            inter_x0 = max(x0, ox0) - margin
            inter_y0 = max(y0, oy0) - margin
            inter_x1 = min(x1, ox1) + margin
            inter_y1 = min(y1, oy1) + margin
            if inter_x1 > inter_x0 and inter_y1 > inter_y0:
                conflict = True
                break
        if not conflict:
            kept.append(item)
    return sorted(kept, key=lambda item: (item.staff, item.bbox[0], item.bbox[1]))


def draw_overlay(
    image: Image.Image,
    staves: list[Staff],
    accidentals: list[Accidental],
    text_regions: list[TextRegion],
    notes: list[NoteHead],
    stems: list[Stem],
    ledger_lines: list[LedgerLine],
    barlines: list[Barline],
    rests: list[Rest],
    beams: list[Beam],
    beam_links: list[BeamLink],
    beam_groups: list[BeamGroup],
    slurs: list[SlurTie],
    out_path: Path,
) -> None:
    canvas = image.convert("RGB")
    draw = ImageDraw.Draw(canvas, "RGBA")
    for staff in staves:
        for y in staff.lines:
            draw.line([(staff.x0, y), (staff.x1, y)], fill=(0, 110, 255, 170), width=2)
        for _, y in legal_pitch_centers(staff):
            draw.line([(staff.scan_x0, y), (staff.x1, y)], fill=(0, 180, 120, 55), width=1)
        draw.rectangle(
            [staff.x0, staff.y0 - 1.2 * staff.space, staff.x1, staff.y1 + 1.2 * staff.space],
            outline=(0, 160, 80, 190),
            width=3,
        )
        if staff.clef_bbox is not None:
            draw.rectangle(staff.clef_bbox, outline=(255, 140, 0, 230), width=4)
            draw.text((staff.clef_bbox[0], max(0, staff.clef_bbox[1] - 18)), staff.clef_type or "clef", fill=(255, 140, 0, 255))
            draw.line(
                [(staff.scan_x0, staff.y0 - 1.1 * staff.space), (staff.scan_x0, staff.y1 + 1.1 * staff.space)],
                fill=(255, 140, 0, 120),
                width=2,
            )

    for note in notes:
        staff = staves[note.staff]
        rx = max(5, int(round(staff.space * 0.78)))
        ry = max(4, int(round(staff.space * 0.50)))
        color = (235, 30, 40, 210) if note.kind == "filled" else (20, 190, 230, 220)
        draw.ellipse([note.x - rx, note.y - ry, note.x + rx, note.y + ry], outline=color, width=3)
        draw.ellipse([note.x - 2, note.y - 2, note.x + 2, note.y + 2], fill=color)

    for accidental in accidentals:
        draw.rectangle(accidental.bbox, outline=(255, 70, 180, 230), width=3)
        draw.text((accidental.bbox[0], max(0, accidental.bbox[1] - 16)), accidental.kind, fill=(255, 70, 180, 255))

    for text_region in text_regions:
        draw.rectangle(text_region.bbox, outline=(70, 90, 255, 230), width=3)
        draw.text((text_region.bbox[0], max(0, text_region.bbox[1] - 16)), text_region.text, fill=(70, 90, 255, 255))

    for stem in stems:
        draw.line([(stem.x, stem.y0), (stem.x, stem.y1)], fill=(190, 20, 220, 220), width=3)

    for ledger in ledger_lines:
        draw.rectangle(ledger.bbox, outline=(120, 70, 0, 230), width=2)
        x0, y0, x1, y1 = ledger.bbox
        draw.line([(x0, 0.5 * (y0 + y1)), (x1, 0.5 * (y0 + y1))], fill=(120, 70, 0, 230), width=3)

    for bar in barlines:
        draw.line([(bar.x, bar.y0), (bar.x, bar.y1)], fill=(255, 135, 0, 230), width=4)

    for rest in rests:
        draw.rectangle(rest.bbox, outline=(255, 210, 0, 230), width=3)

    for beam in beams:
        draw.rectangle(beam.bbox, outline=(20, 155, 40, 95), width=2)

    for link in beam_links:
        draw.line([(link.x0, link.y0), (link.x1, link.y1)], fill=(0, 180, 40, 55), width=2)

    for group in beam_groups:
        if len(group.points) < 2:
            continue
        draw.line([tuple(point) for point in group.points], fill=(0, 210, 40, 245), width=6)
        draw.rectangle(group.bbox, outline=(0, 150, 40, 150), width=2)
        r = 4
        for x, y in group.points:
            draw.ellipse([x - r, y - r, x + r, y + r], fill=(0, 210, 40, 245))

    for slur in slurs:
        draw.rectangle(slur.bbox, outline=(150, 80, 255, 220), width=2)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def draw_notehead_weight_regions(
    image: Image.Image,
    staves: list[Staff],
    notes: list[NoteHead],
    out_path: Path,
) -> None:
    base = Image.blend(image.convert("RGB"), Image.new("RGB", image.size, "white"), 0.30).convert("RGBA")
    overlay = Image.new("RGBA", image.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")

    for staff in staves:
        for y in staff.lines:
            draw.line([(staff.x0, y), (staff.x1, y)], fill=(30, 110, 255, 115), width=1)

    font = ImageFont.load_default()
    legend_x = 18
    legend_y = 18
    draw.rectangle([legend_x - 8, legend_y - 8, legend_x + 360, legend_y + 74], fill=(255, 255, 255, 210), outline=(0, 0, 0, 120))
    draw.text((legend_x, legend_y), "orange: surround / weight map region", fill=(210, 110, 0, 255), font=font)
    draw.text((legend_x, legend_y + 18), "red: filled notehead body; cyan: open notehead body", fill=(40, 40, 40, 255), font=font)
    draw.text((legend_x, legend_y + 36), "purple: inner hole/core region used by filled/open check", fill=(120, 30, 170, 255), font=font)
    draw.text((legend_x, legend_y + 54), "center label: note index, kind, score", fill=(40, 40, 40, 255), font=font)

    for idx, note in enumerate(notes):
        staff = staves[note.staff]
        space = staff.space
        rx = max(5, int(round(space * 0.78)))
        ry = max(4, int(round(space * 0.50)))
        inner_rx = max(1, int(round(rx * 0.50)))
        inner_ry = max(1, int(round(ry * 0.48)))
        surround_rx = max(rx + 1, int(round(rx * 1.35)))
        surround_ry = max(ry + 1, int(round(ry * 1.35)))
        cx = int(round(note.x))
        cy = int(round(note.y))

        weight_box = [cx - surround_rx, cy - surround_ry, cx + surround_rx, cy + surround_ry]
        note_box = [cx - rx, cy - ry, cx + rx, cy + ry]
        inner_box = [cx - inner_rx, cy - inner_ry, cx + inner_rx, cy + inner_ry]
        body_color = (235, 30, 40, 230) if note.kind == "filled" else (20, 190, 230, 235)

        draw.ellipse(weight_box, fill=(255, 185, 0, 28), outline=(255, 145, 0, 185), width=1)
        draw.rectangle(weight_box, outline=(255, 145, 0, 90), width=1)
        draw.ellipse(note_box, fill=body_color[:3] + (34,), outline=body_color, width=3)
        draw.rectangle(note_box, outline=body_color[:3] + (120,), width=1)
        draw.ellipse(inner_box, outline=(130, 40, 190, 230), width=2)
        draw.line([(cx - 5, cy), (cx + 5, cy)], fill=(0, 0, 0, 210), width=1)
        draw.line([(cx, cy - 5), (cx, cy + 5)], fill=(0, 0, 0, 210), width=1)

        if idx % 2 == 0:
            label_y = cy - surround_ry - 12
        else:
            label_y = cy + surround_ry + 2
        label = f"{idx}:{note.kind[0]}:{note.score:.2f}"
        draw.text((cx - surround_rx, label_y), label, fill=body_color, font=font)

    canvas = Image.alpha_composite(base, overlay).convert("RGB")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def load_clef_font(size: int) -> ImageFont.ImageFont:
    for path in (
        Path("C:/Windows/Fonts/seguisym.ttf"),
        Path("C:/Windows/Fonts/symbol.ttf"),
        Path("C:/Windows/Fonts/arial.ttf"),
    ):
        if path.exists():
            try:
                return ImageFont.truetype(str(path), size)
            except OSError:
                continue
    return ImageFont.load_default()


def draw_clef_symbol(draw: ImageDraw.ImageDraw, staff: Staff) -> None:
    if staff.clef_bbox is None:
        return
    x0, y0, x1, y1 = staff.clef_bbox
    bbox_w = max(1, x1 - x0)
    bbox_h = max(1, y1 - y0)
    if staff.clef_type == "treble":
        symbol = chr(0x1D11E)
        fallback = "G"
        font_size = max(24, int(round(bbox_h * 0.95)))
    else:
        symbol = chr(0x1D122)
        fallback = "F"
        font_size = max(20, int(round(bbox_h * 1.05)))

    font = load_clef_font(font_size)
    try:
        text_bbox = draw.textbbox((0, 0), symbol, font=font)
    except UnicodeEncodeError:
        symbol = fallback
        font = load_clef_font(max(18, int(round(bbox_h * 0.55))))
        text_bbox = draw.textbbox((0, 0), symbol, font=font)
    text_w = text_bbox[2] - text_bbox[0]
    text_h = text_bbox[3] - text_bbox[1]
    tx = x0 + 0.5 * (bbox_w - text_w) - text_bbox[0]
    ty = y0 + 0.5 * (bbox_h - text_h) - text_bbox[1]
    draw.text((tx, ty), symbol, font=font, fill="black")


def draw_reconstruction(
    size: tuple[int, int],
    staves: list[Staff],
    accidentals: list[Accidental],
    text_regions: list[TextRegion],
    notes: list[NoteHead],
    stems: list[Stem],
    ledger_lines: list[LedgerLine],
    barlines: list[Barline],
    rests: list[Rest],
    beams: list[Beam],
    beam_links: list[BeamLink],
    beam_groups: list[BeamGroup],
    slurs: list[SlurTie],
    out_path: Path,
) -> None:
    canvas = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(canvas)
    for staff in staves:
        for y in staff.lines:
            draw.line([(staff.x0, y), (staff.x1, y)], fill="black", width=2)
        draw_clef_symbol(draw, staff)
    for accidental in accidentals:
        x0, y0, x1, y1 = accidental.bbox
        symbol = {"sharp": "#", "flat": "b", "natural": "n"}.get(accidental.kind, "#")
        font = load_clef_font(max(10, int(round((y1 - y0) * 0.85))))
        draw.text((x0, y0 - max(1, int(round(0.08 * (y1 - y0))))), symbol, font=font, fill="black")
    for text_region in text_regions:
        x0, y0, x1, y1 = text_region.bbox
        label = text_region.text if text_region.text != "text" else "TEXT"
        font = load_clef_font(max(10, int(round((y1 - y0) * 0.95))))
        draw.text((x0, y0), label, font=font, fill="black")
    for bar in barlines:
        draw.line([(bar.x, bar.y0), (bar.x, bar.y1)], fill="black", width=3)
    for ledger in ledger_lines:
        x0, y0, x1, y1 = ledger.bbox
        draw.line([(x0, 0.5 * (y0 + y1)), (x1, 0.5 * (y0 + y1))], fill="black", width=2)
    for rest in rests:
        draw.rectangle(rest.bbox, outline="black", width=2)
        x0, y0, x1, y1 = rest.bbox
        if rest.kind.startswith("whole"):
            draw.rectangle([x0, y0, x1, y1], fill="black")
    for beam in beams:
        draw.rectangle(beam.bbox, fill="black")
    for group in beam_groups:
        if len(group.points) >= 2:
            draw.line([tuple(point) for point in group.points], fill="black", width=5)
    for slur in slurs:
        x0, y0, x1, y1 = slur.bbox
        draw.arc([x0, y0, x1, y1 + max(4, (y1 - y0))], start=185, end=355, fill="black", width=2)
    stem_by_note = {stem.note_index: stem for stem in stems}
    for idx, note in enumerate(notes):
        staff = staves[note.staff]
        rx = max(5, int(round(staff.space * 0.78)))
        ry = max(4, int(round(staff.space * 0.50)))
        box = [note.x - rx, note.y - ry, note.x + rx, note.y + ry]
        if note.kind == "filled":
            draw.ellipse(box, fill="black")
        else:
            draw.ellipse(box, outline="black", width=3)
        stem = stem_by_note.get(idx)
        if stem is not None:
            draw.line([(stem.x, stem.y0), (stem.x, stem.y1)], fill="black", width=2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def write_result_json(
    path: Path,
    input_path: Path,
    threshold: int,
    lines: list[StaffLine],
    staves: list[Staff],
    accidentals: list[Accidental],
    text_regions: list[TextRegion],
    notes: list[NoteHead],
    stems: list[Stem],
    ledger_lines: list[LedgerLine],
    barlines: list[Barline],
    rests: list[Rest],
    beams: list[Beam],
    beam_links: list[BeamLink],
    beam_groups: list[BeamGroup],
    slurs: list[SlurTie],
) -> None:
    payload = {
        "input": str(input_path),
        "threshold": threshold,
        "staff_line_candidates": [asdict(line) for line in lines],
        "staves": [asdict(staff) for staff in staves],
        "accidentals": [asdict(accidental) for accidental in accidentals],
        "text_regions": [asdict(region) for region in text_regions],
        "noteheads": [asdict(note) for note in notes],
        "stems": [asdict(stem) for stem in stems],
        "ledger_lines": [asdict(line) for line in ledger_lines],
        "barlines": [asdict(barline) for barline in barlines],
        "rests": [asdict(rest) for rest in rests],
        "beams": [asdict(beam) for beam in beams],
        "beam_links": [asdict(link) for link in beam_links],
        "beam_groups": [asdict(group) for group in beam_groups],
        "slurs_or_ties": [asdict(slur) for slur in slurs],
        "counts": {
            "staff_lines": len(lines),
            "staves": len(staves),
            "accidentals": len(accidentals),
            "text_regions": len(text_regions),
            "noteheads": len(notes),
            "filled_noteheads": sum(1 for note in notes if note.kind == "filled"),
            "open_noteheads": sum(1 for note in notes if note.kind == "open"),
            "stems": len(stems),
            "ledger_lines": len(ledger_lines),
            "barlines": len(barlines),
            "rests": len(rests),
            "beams": len(beams),
            "beam_links": len(beam_links),
            "beam_groups": len(beam_groups),
            "slurs_or_ties": len(slurs),
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Traditional OMR geometry demo on one score image.")
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--pdf-dpi", type=int, default=220)
    args = parser.parse_args()

    input_path = args.input
    if input_path is None:
        input_path = DEFAULT_INPUT if DEFAULT_INPUT.exists() else LOCAL_FALLBACK_INPUT

    image = load_input_image(input_path, dpi=args.pdf_dpi)
    gray = np.array(image.convert("L"))
    mask, threshold = binarize(gray)

    staff_lines = find_staff_line_candidates(mask)
    staff_space = estimate_staff_space(staff_lines)
    staves = group_staves(staff_lines, staff_space)
    cleaned = remove_staff_lines(mask, staves)
    detect_clefs(cleaned, staves)
    long_run_mask = find_long_run_mask(cleaned, staves)
    preliminary_text_regions = detect_text_regions(cleaned, staves, [])
    accidentals = detect_accidentals(cleaned, staves, preliminary_text_regions)
    text_regions = detect_text_regions(cleaned, staves, accidentals)
    accidental_mask = mask_regions(cleaned.shape, accidentals, pad_x=2, pad_y=2)
    text_mask = mask_regions(cleaned.shape, text_regions, pad_x=2, pad_y=2)
    note_mask = cleaned & ~long_run_mask & ~accidental_mask & ~text_mask
    notes = detect_noteheads(note_mask, staves, raw_mask=mask)
    stems = detect_stems(cleaned, notes, staves)
    ledger_lines = detect_ledger_lines(mask, notes, staves)
    barlines = detect_barlines(mask, staves)
    beam_links = detect_beam_links_from_stems(cleaned, stems, staves)
    beam_groups = group_beam_links(beam_links, stems, staves)
    pruned = erase_detected_notes_and_stems(cleaned, notes, stems, staves)
    rests, beams, slurs = detect_secondary_symbols(pruned, staves, notes, stems, beam_groups, barlines)

    binary_path = args.out_dir / "binary_cleaned_input.png"
    overlay_path = args.out_dir / "overlay_detected_staff_notes.png"
    notehead_debug_path = args.out_dir / "notehead_weight_regions.png"
    reconstructed_path = args.out_dir / "reconstructed_from_detection.png"
    pdf_path = args.out_dir / "demo_visual_result.pdf"
    json_path = args.out_dir / "detection_result.json"

    save_binary(mask, binary_path)
    draw_overlay(image, staves, accidentals, text_regions, notes, stems, ledger_lines, barlines, rests, beams, beam_links, beam_groups, slurs, overlay_path)
    draw_notehead_weight_regions(image, staves, notes, notehead_debug_path)
    draw_reconstruction(image.size, staves, accidentals, text_regions, notes, stems, ledger_lines, barlines, rests, beams, beam_links, beam_groups, slurs, reconstructed_path)
    write_result_json(
        json_path,
        input_path,
        threshold,
        staff_lines,
        staves,
        accidentals,
        text_regions,
        notes,
        stems,
        ledger_lines,
        barlines,
        rests,
        beams,
        beam_links,
        beam_groups,
        slurs,
    )

    # A compact PDF is convenient for quick manual review.
    binary = Image.open(binary_path).convert("RGB")
    overlay = Image.open(overlay_path).convert("RGB")
    notehead_debug = Image.open(notehead_debug_path).convert("RGB")
    reconstruction = Image.open(reconstructed_path).convert("RGB")
    binary.save(pdf_path, save_all=True, append_images=[overlay, notehead_debug, reconstruction])

    print(json.dumps({
        "input": str(input_path),
        "threshold": threshold,
        "staff_space": round(staff_space, 3),
        "staves": len(staves),
        "staff_lines": len(staff_lines),
        "accidentals": len(accidentals),
        "text_regions": len(text_regions),
        "noteheads": len(notes),
        "filled_noteheads": sum(1 for note in notes if note.kind == "filled"),
        "open_noteheads": sum(1 for note in notes if note.kind == "open"),
        "stems": len(stems),
        "ledger_lines": len(ledger_lines),
        "barlines": len(barlines),
        "rests": len(rests),
        "beams": len(beams),
        "beam_links": len(beam_links),
        "beam_groups": len(beam_groups),
        "slurs_or_ties": len(slurs),
        "binary": str(binary_path),
        "overlay": str(overlay_path),
        "notehead_debug": str(notehead_debug_path),
        "reconstructed": str(reconstructed_path),
        "pdf": str(pdf_path),
        "json": str(json_path),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
