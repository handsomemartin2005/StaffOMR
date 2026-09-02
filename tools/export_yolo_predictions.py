from __future__ import annotations

import argparse
import json
from pathlib import Path

from omr_v2_common import base_class_for_symbol_class


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Ultralytics YOLO symbol detections using the common StaffOMR detector JSON contract.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--class-names-json", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--overlay", type=Path)
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--nms-iou", type=float, default=0.50)
    parser.add_argument("--max-detections", type=int, default=1000)
    parser.add_argument("--input-size", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--taxonomy", default="expanded_clean")
    args = parser.parse_args()
    from PIL import Image
    from ultralytics import YOLO

    categories = sorted(json.loads(args.class_names_json.read_text(encoding="utf-8"))["categories"], key=lambda item: int(item["id"]))
    class_names = [str(item["name"]) for item in categories]
    model = YOLO(str(args.model))
    result = model.predict(source=str(args.input), conf=args.threshold, iou=args.nms_iou, max_det=args.max_detections, imgsz=args.input_size, device=args.device, verbose=False)[0]
    symbols = []
    boxes = result.boxes
    if boxes is not None:
        for index, (bbox, confidence, label) in enumerate(zip(boxes.xyxy.cpu().tolist(), boxes.conf.cpu().tolist(), boxes.cls.cpu().tolist())):
            label_index = int(label)
            detector_class = class_names[label_index] if 0 <= label_index < len(class_names) else f"class_{label_index}"
            output_class = base_class_for_symbol_class(detector_class) or detector_class
            symbols.append(
                {
                    "id": f"yolo_{index:06d}",
                    "class": output_class,
                    "bbox": [int(round(value)) for value in bbox],
                    "confidence": float(confidence),
                    "source": "yolo_ultralytics",
                    "attributes": {"zero_based_label": label_index, "detector_class": detector_class, "fine_class": detector_class},
                }
            )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": "v2_yolo_predictions",
        "input": str(args.input),
        "model": str(args.model),
        "threshold": args.threshold,
        "nms_iou": args.nms_iou,
        "taxonomy": args.taxonomy,
        "class_names": class_names,
        "symbols": symbols,
        "inference": {"input_size": args.input_size, "image_size": list(Image.open(args.input).size)},
        "overlay": str(args.overlay) if args.overlay else None,
    }
    args.out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.overlay:
        args.overlay.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(result.plot()[:, :, ::-1]).save(args.overlay)
    print(json.dumps({"out_json": str(args.out_json), "symbols": len(symbols)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
