from __future__ import annotations

import argparse
import json
from pathlib import Path

from omr_v2_common import (
    count_by_class,
    draw_symbol_overlay,
    enrich_symbols_with_staff_geometry,
    merge_coco_documents,
    run_v1_symbol_pipeline,
    symbols_to_coco,
    v1_result_to_weak_symbols,
    write_json,
)


def collect_inputs(args: argparse.Namespace) -> list[Path | None]:
    inputs: list[Path | None] = []
    if args.input:
        inputs.extend(args.input)
    if args.input_list:
        for line in args.input_list.read_text(encoding="utf-8").splitlines():
            value = line.strip()
            if not value or value.startswith("#"):
                continue
            inputs.append(Path(value))
    if not inputs:
        inputs.append(None)
    return inputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Export V1 rule detections as weak V2 symbol labels.")
    parser.add_argument("--input", type=Path, action="append", help="Input PNG/PDF path. Can be repeated.")
    parser.add_argument("--input-list", type=Path, help="Text file with one input path per line.")
    parser.add_argument("--pdf-dpi", type=int, default=220)
    parser.add_argument("--out-coco", type=Path, default=Path("outputs/v2_weak_labels/weak_symbols_coco.json"))
    parser.add_argument("--out-json", type=Path, default=Path("outputs/v2_weak_labels/weak_symbols_v2.json"))
    parser.add_argument("--overlay-dir", type=Path, default=Path("outputs/v2_weak_labels/overlays"))
    parser.add_argument("--no-overlays", action="store_true")
    args = parser.parse_args()

    coco_docs = []
    pages = []
    total_counts: dict[str, int] = {}
    for image_id, input_path in enumerate(collect_inputs(args), start=1):
        result = run_v1_symbol_pipeline(input_path, pdf_dpi=args.pdf_dpi)
        symbols = enrich_symbols_with_staff_geometry(v1_result_to_weak_symbols(result), result["staves"])
        counts = count_by_class(symbols)
        for name, count in counts.items():
            total_counts[name] = total_counts.get(name, 0) + count

        input_resolved = result["input"]
        image = result["image"]
        coco_docs.append(symbols_to_coco(image_id, input_resolved, image.size, symbols))
        pages.append(
            {
                "image_id": image_id,
                "input": str(input_resolved),
                "width": image.width,
                "height": image.height,
                "threshold": result["threshold"],
                "staff_space": result["staff_space"],
                "staves": len(result["staves"]),
                "symbols": symbols,
                "counts": counts,
            }
        )

        if not args.no_overlays:
            overlay_name = f"{image_id:04d}_{input_resolved.stem}_weak_overlay.png"
            draw_symbol_overlay(image, symbols, args.overlay_dir / overlay_name)

    coco = merge_coco_documents(coco_docs)
    write_json(args.out_coco, coco)
    write_json(
        args.out_json,
        {
            "version": "v2_weak_labels_from_v1",
            "pages": pages,
            "total_counts": dict(sorted(total_counts.items())),
            "coco": str(args.out_coco),
        },
    )

    print(
        json.dumps(
            {
                "pages": len(pages),
                "annotations": len(coco["annotations"]),
                "out_coco": str(args.out_coco),
                "out_json": str(args.out_json),
                "counts": dict(sorted(total_counts.items())),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
