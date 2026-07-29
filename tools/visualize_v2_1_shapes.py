from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from omr_v2_common import color_for_class, read_json


def load_font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:
        return ImageFont.load_default()


def polygon_points(symbol: dict[str, Any]) -> list[tuple[int, int]]:
    mask = symbol.get("mask") or {}
    return [(int(x), int(y)) for x, y in mask.get("points", [])]


def skeleton_points(symbol: dict[str, Any]) -> list[tuple[int, int]]:
    skeleton = symbol.get("skeleton") or {}
    return [(int(x), int(y)) for x, y in skeleton.get("points", [])]


def relation_anchor(symbol: dict[str, Any]) -> tuple[int, int]:
    skeleton = skeleton_points(symbol)
    if skeleton:
        return skeleton[len(skeleton) // 2]
    polygon = polygon_points(symbol)
    if polygon:
        xs = [point[0] for point in polygon]
        ys = [point[1] for point in polygon]
        return int(sum(xs) / len(xs)), int(sum(ys) / len(ys))
    x0, y0, x1, y1 = [int(round(v)) for v in symbol["bbox"]]
    return (x0 + x1) // 2, (y0 + y1) // 2


def draw_staff(draw: ImageDraw.ImageDraw, payload: dict[str, Any]) -> None:
    for staff in payload.get("staves", []):
        x0, x1 = int(staff["x0"]), int(staff["x1"])
        for y in staff.get("lines", []):
            yy = int(round(y))
            draw.line([(x0, yy), (x1, yy)], fill=(40, 40, 40, 70), width=1)


def draw_polygon_overlay(payload: dict[str, Any], out_path: Path) -> None:
    image = Image.open(payload["input"]).convert("RGB")
    canvas = image.convert("RGBA")
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    draw_staff(draw, payload)
    font = load_font(13)
    for symbol in payload.get("symbols", []):
        color = color_for_class(symbol["class"])
        fill = (color[0], color[1], color[2], 42)
        outline = (color[0], color[1], color[2], 230)
        points = polygon_points(symbol)
        if len(points) >= 3:
            draw.polygon(points, fill=fill)
            draw.line(points + [points[0]], fill=outline, width=2)
        else:
            x0, y0, x1, y1 = [int(round(v)) for v in symbol["bbox"]]
            draw.rectangle([x0, y0, x1, y1], outline=outline, width=1)
        if symbol.get("geometry_check") in {"fail", "warn"}:
            x0, y0, _, _ = [int(round(v)) for v in symbol["bbox"]]
            draw.text((x0, max(0, y0 - 12)), symbol["geometry_check"], fill=(220, 20, 20, 230), font=font)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.alpha_composite(canvas, overlay).convert("RGB").save(out_path)


def draw_skeleton_overlay(payload: dict[str, Any], out_path: Path) -> None:
    image = Image.open(payload["input"]).convert("RGB")
    canvas = image.convert("RGBA")
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    draw_staff(draw, payload)
    for symbol in payload.get("symbols", []):
        points = skeleton_points(symbol)
        if len(points) < 2:
            continue
        color = color_for_class(symbol["class"])
        draw.line(points, fill=(color[0], color[1], color[2], 245), width=3)
        r = 3
        for point in (points[0], points[-1]):
            draw.ellipse([point[0] - r, point[1] - r, point[0] + r, point[1] + r], fill=(255, 255, 255, 230), outline=color, width=2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.alpha_composite(canvas, overlay).convert("RGB").save(out_path)


def draw_relation_overlay(payload: dict[str, Any], out_path: Path, label_relations: bool = False) -> None:
    image = Image.open(payload["input"]).convert("RGB")
    canvas = image.convert("RGBA")
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    draw_staff(draw, payload)
    symbols_by_id = {str(symbol["id"]): symbol for symbol in payload.get("symbols", [])}
    relation_colors = {
        "notehead_stem_attachment": (255, 45, 60, 235),
        "beam_stem_group": (20, 150, 50, 235),
        "ledger_line_notehead_attachment": (150, 80, 0, 235),
        "slur_tie_notehead_endpoints": (130, 60, 230, 235),
    }
    font = load_font(13)
    for relation in payload.get("relations", []):
        source = symbols_by_id.get(str(relation.get("source")))
        if source is None:
            continue
        source_point = relation_anchor(source)
        color = relation_colors.get(relation.get("type"), (0, 0, 0, 220))
        for target_id in relation.get("targets", []):
            target = symbols_by_id.get(str(target_id))
            if target is None:
                continue
            target_point = relation_anchor(target)
            draw.line([source_point, target_point], fill=color, width=2)
        if label_relations:
            draw.text((source_point[0] + 4, source_point[1] + 4), relation.get("type", "rel"), fill=color, font=font)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.alpha_composite(canvas, overlay).convert("RGB").save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize V2.1 polygons, skeletons, and relation graph.")
    parser.add_argument("--shapes-json", type=Path, required=True)
    parser.add_argument("--polygon-overlay", type=Path)
    parser.add_argument("--skeleton-overlay", type=Path)
    parser.add_argument("--relation-overlay", type=Path)
    parser.add_argument("--label-relations", action="store_true")
    args = parser.parse_args()

    payload: dict[str, Any] = read_json(args.shapes_json)
    if args.polygon_overlay:
        draw_polygon_overlay(payload, args.polygon_overlay)
        print(args.polygon_overlay)
    if args.skeleton_overlay:
        draw_skeleton_overlay(payload, args.skeleton_overlay)
        print(args.skeleton_overlay)
    if args.relation_overlay:
        draw_relation_overlay(payload, args.relation_overlay, label_relations=args.label_relations)
        print(args.relation_overlay)


if __name__ == "__main__":
    main()
