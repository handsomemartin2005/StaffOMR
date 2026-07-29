from __future__ import annotations

from typing import Any


DYNAMIC_LETTER_CLASSES = {
    "dynamic_p": "p",
    "dynamic_f": "f",
    "dynamic_m": "m",
    "dynamic_s": "s",
    "dynamic_z": "z",
}
DYNAMIC_HAIRPIN_CLASSES = {
    "dynamic_crescendo_hairpin": "crescendo",
    "dynamic_diminuendo_hairpin": "diminuendo",
}
PEDAL_TEXT = {
    "pedal_mark": "Ped.",
    "pedal_up": "*",
}


def detector_class(symbol: dict[str, Any]) -> str:
    attrs = symbol.get("attributes") or {}
    return str(attrs.get("detector_class") or attrs.get("fine_class") or symbol.get("class"))


def symbol_staff(symbol: dict[str, Any]) -> int | None:
    value = (symbol.get("attributes") or {}).get("staff")
    return int(value) if value is not None else None


def bbox(symbol: dict[str, Any]) -> list[float]:
    return [float(value) for value in symbol["bbox"]]


def center(symbol: dict[str, Any]) -> tuple[float, float]:
    x0, y0, x1, y1 = bbox(symbol)
    return 0.5 * (x0 + x1), 0.5 * (y0 + y1)


def union_bbox(symbols: list[dict[str, Any]]) -> list[float]:
    boxes = [bbox(symbol) for symbol in symbols]
    return [
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    ]


def bbox_iou(left: dict[str, Any], right: dict[str, Any]) -> float:
    lx0, ly0, lx1, ly1 = bbox(left)
    rx0, ry0, rx1, ry1 = bbox(right)
    ix0 = max(lx0, rx0)
    iy0 = max(ly0, ry0)
    ix1 = min(lx1, rx1)
    iy1 = min(ly1, ry1)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    left_area = max(0.0, lx1 - lx0) * max(0.0, ly1 - ly0)
    right_area = max(0.0, rx1 - rx0) * max(0.0, ry1 - ry0)
    union = left_area + right_area - inter
    return inter / union if union > 0 else 0.0


def deduplicate_overlapping_candidates(
    symbols: list[dict[str, Any]],
    *,
    iou_threshold: float = 0.78,
) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for symbol in sorted(symbols, key=lambda item: float(item.get("confidence") or 0.0), reverse=True):
        if any(symbol_staff(symbol) == symbol_staff(item) and bbox_iou(symbol, item) >= iou_threshold for item in kept):
            continue
        kept.append(symbol)
    return kept


def symbol_space(symbol: dict[str, Any], staves_by_index: dict[int, dict[str, Any]]) -> float:
    attrs = symbol.get("attributes") or {}
    if attrs.get("staff_space") is not None:
        return float(attrs["staff_space"])
    staff = symbol_staff(symbol)
    if staff is not None and staff in staves_by_index:
        return float(staves_by_index[staff].get("space") or 12.0)
    return 12.0


def average_confidence(symbols: list[dict[str, Any]]) -> float:
    values = [float(symbol.get("confidence") or 0.0) for symbol in symbols]
    return sum(values) / max(1, len(values))


def mark_center(mark_bbox: list[float]) -> list[float]:
    x0, y0, x1, y1 = mark_bbox
    return [0.5 * (x0 + x1), 0.5 * (y0 + y1)]


def is_valid_dynamic_text(text: str) -> bool:
    if len(text) < 2:
        return False
    if set(text) == {"p"} and len(text) <= 4:
        return True
    if set(text) == {"f"} and len(text) <= 4:
        return True
    return text in {
        "mp",
        "mf",
        "fp",
        "sf",
        "sfz",
        "sfp",
        "fz",
    }


def same_vertical_band(
    left: dict[str, Any],
    right: dict[str, Any],
    staves_by_index: dict[int, dict[str, Any]],
) -> bool:
    _, ly = center(left)
    _, ry = center(right)
    space = max(symbol_space(left, staves_by_index), symbol_space(right, staves_by_index))
    lh = bbox(left)[3] - bbox(left)[1]
    rh = bbox(right)[3] - bbox(right)[1]
    return abs(ly - ry) <= max(0.9 * space, 0.65 * max(lh, rh))


def close_horizontally(
    left: dict[str, Any],
    right: dict[str, Any],
    staves_by_index: dict[int, dict[str, Any]],
    *,
    multiplier: float,
) -> bool:
    lx0, _, lx1, _ = bbox(left)
    rx0, _, rx1, _ = bbox(right)
    gap = rx0 - lx1
    space = max(symbol_space(left, staves_by_index), symbol_space(right, staves_by_index))
    avg_width = 0.5 * ((lx1 - lx0) + (rx1 - rx0))
    return -0.65 * space <= gap <= max(multiplier * space, 2.8 * avg_width)


def group_by_staff_and_y(
    symbols: list[dict[str, Any]],
    staves_by_index: dict[int, dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    groups: list[list[dict[str, Any]]] = []
    for symbol in sorted(symbols, key=lambda item: (symbol_staff(item) is None, symbol_staff(item) or -1, center(item)[1])):
        staff = symbol_staff(symbol)
        _, cy = center(symbol)
        space = symbol_space(symbol, staves_by_index)
        matched: list[dict[str, Any]] | None = None
        for group in groups:
            first = group[0]
            if symbol_staff(first) != staff:
                continue
            group_y = sum(center(item)[1] for item in group) / len(group)
            if abs(cy - group_y) <= max(0.9 * space, 8.0):
                matched = group
                break
        if matched is None:
            groups.append([symbol])
        else:
            matched.append(symbol)
    return groups


def build_mark(
    *,
    mark_id: str,
    kind: str,
    text: str,
    symbols: list[dict[str, Any]],
    composite: bool,
) -> dict[str, Any]:
    box = union_bbox(symbols)
    return {
        "id": mark_id,
        "type": "mark",
        "kind": kind,
        "text": text,
        "staff": symbol_staff(symbols[0]),
        "center": mark_center(box),
        "bbox": box,
        "confidence": average_confidence(symbols),
        "source_symbol_ids": [str(symbol["id"]) for symbol in symbols],
        "composite": composite,
    }


def compose_dynamic_marks(
    symbols: list[dict[str, Any]],
    staves_by_index: dict[int, dict[str, Any]],
    *,
    include_singletons: bool,
    start_index: int,
) -> tuple[list[dict[str, Any]], set[str], int]:
    candidates = deduplicate_overlapping_candidates(
        [symbol for symbol in symbols if detector_class(symbol) in DYNAMIC_LETTER_CLASSES]
    )
    marks: list[dict[str, Any]] = []
    consumed: set[str] = set()
    index = start_index
    for band in group_by_staff_and_y(candidates, staves_by_index):
        ordered = sorted(band, key=lambda item: (bbox(item)[0], bbox(item)[1]))
        group: list[dict[str, Any]] = []

        def flush() -> None:
            nonlocal index
            if not group:
                return
            text = "".join(DYNAMIC_LETTER_CLASSES[detector_class(item)] for item in sorted(group, key=lambda item: bbox(item)[0]))
            if len(group) >= 2 and is_valid_dynamic_text(text):
                source = sorted(group, key=lambda item: bbox(item)[0])
                marks.append(
                    build_mark(
                        mark_id=f"mark_{index:05d}",
                        kind="dynamic",
                        text=text,
                        symbols=source,
                        composite=True,
                    )
                )
                consumed.update(str(item["id"]) for item in source)
                index += 1
            elif include_singletons:
                for item in sorted(group, key=lambda item: bbox(item)[0]):
                    marks.append(
                        build_mark(
                            mark_id=f"mark_{index:05d}",
                            kind="dynamic",
                            text=DYNAMIC_LETTER_CLASSES[detector_class(item)],
                            symbols=[item],
                            composite=False,
                        )
                    )
                    index += 1

        for symbol in ordered:
            if not group:
                group = [symbol]
                continue
            previous = group[-1]
            if same_vertical_band(previous, symbol, staves_by_index) and close_horizontally(
                previous,
                symbol,
                staves_by_index,
                multiplier=1.65,
            ):
                group.append(symbol)
                continue
            flush()
            group = [symbol]
        flush()
    return marks, consumed, index


def compose_pedal_marks(
    symbols: list[dict[str, Any]],
    staves_by_index: dict[int, dict[str, Any]],
    *,
    include_singletons: bool,
    start_index: int,
) -> tuple[list[dict[str, Any]], set[str], int]:
    candidates = [symbol for symbol in symbols if detector_class(symbol) in PEDAL_TEXT]
    marks: list[dict[str, Any]] = []
    consumed: set[str] = set()
    index = start_index
    for band in group_by_staff_and_y(candidates, staves_by_index):
        ordered = sorted(band, key=lambda item: (bbox(item)[0], bbox(item)[1]))
        group: list[dict[str, Any]] = []

        def flush() -> None:
            nonlocal index
            if not group:
                return
            source = sorted(group, key=lambda item: bbox(item)[0])
            classes = {detector_class(item) for item in source}
            is_composite = len(source) >= 2 and "pedal_mark" in classes and "pedal_up" in classes
            if is_composite or include_singletons:
                text = "".join(PEDAL_TEXT[detector_class(item)] for item in source)
                marks.append(
                    build_mark(
                        mark_id=f"mark_{index:05d}",
                        kind="pedal",
                        text=text,
                        symbols=source,
                        composite=is_composite,
                    )
                )
                if is_composite:
                    consumed.update(str(item["id"]) for item in source)
                index += 1

        for symbol in ordered:
            if not group:
                group = [symbol]
                continue
            previous = group[-1]
            if same_vertical_band(previous, symbol, staves_by_index) and close_horizontally(
                previous,
                symbol,
                staves_by_index,
                multiplier=2.7,
            ):
                group.append(symbol)
                continue
            flush()
            group = [symbol]
        flush()
    return marks, consumed, index


def compose_hairpin_marks(
    symbols: list[dict[str, Any]],
    *,
    include_singletons: bool,
    start_index: int,
) -> tuple[list[dict[str, Any]], set[str], int]:
    if not include_singletons:
        return [], set(), start_index
    marks: list[dict[str, Any]] = []
    index = start_index
    for symbol in sorted(
        [item for item in symbols if detector_class(item) in DYNAMIC_HAIRPIN_CLASSES],
        key=lambda item: (symbol_staff(item) is None, symbol_staff(item) or -1, bbox(item)[0], bbox(item)[1]),
    ):
        cls = detector_class(symbol)
        marks.append(
            build_mark(
                mark_id=f"mark_{index:05d}",
                kind="dynamic_hairpin",
                text=DYNAMIC_HAIRPIN_CLASSES[cls],
                symbols=[symbol],
                composite=False,
            )
        )
        index += 1
    return marks, set(), index


def compose_marks(
    symbols: list[dict[str, Any]],
    staves: list[dict[str, Any]] | None = None,
    *,
    include_singletons: bool = False,
) -> dict[str, Any]:
    staves_by_index = {int(staff["index"]): staff for staff in staves or [] if staff.get("index") is not None}
    marks: list[dict[str, Any]] = []
    consumed: set[str] = set()
    index = 1

    dynamic_marks, dynamic_consumed, index = compose_dynamic_marks(
        symbols,
        staves_by_index,
        include_singletons=include_singletons,
        start_index=index,
    )
    marks.extend(dynamic_marks)
    consumed.update(dynamic_consumed)

    pedal_marks, pedal_consumed, index = compose_pedal_marks(
        symbols,
        staves_by_index,
        include_singletons=include_singletons,
        start_index=index,
    )
    marks.extend(pedal_marks)
    consumed.update(pedal_consumed)

    hairpin_marks, hairpin_consumed, index = compose_hairpin_marks(
        symbols,
        include_singletons=include_singletons,
        start_index=index,
    )
    marks.extend(hairpin_marks)
    consumed.update(hairpin_consumed)

    marks.sort(key=lambda mark: (mark.get("staff") is None, mark.get("staff") or -1, mark["center"][0], mark["center"][1], mark["id"]))
    return {
        "marks": marks,
        "consumed_symbol_ids": sorted(consumed),
        "summary": {
            "marks": len(marks),
            "composite_marks": sum(1 for mark in marks if mark.get("composite")),
            "dynamic_composites": sum(1 for mark in marks if mark.get("kind") == "dynamic" and mark.get("composite")),
            "pedal_composites": sum(1 for mark in marks if mark.get("kind") == "pedal" and mark.get("composite")),
        },
    }
