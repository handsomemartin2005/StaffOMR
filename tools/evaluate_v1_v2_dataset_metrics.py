from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

from evaluate_v1_v2_metrics import (
    error_rate_from_ops,
    iou,
    levenshtein_ops,
    pred_staff_symbols,
    reading_sequence,
)
from export_deim_predictions import load_class_names, load_deim_model, predict
from infer_neural_symbols_v2 import merge_rule_symbols, parse_class_list
from omr_v2_common import (
    canonical_deepscores_class,
    count_by_class,
    enrich_symbols_with_staff_geometry,
    read_json,
    run_v1_symbol_pipeline,
    v1_result_to_weak_symbols,
    write_json,
)


TOKEN_CLASSES = {
    "filled_notehead",
    "open_notehead",
    "stem",
    "beam",
    "barline",
    "ledger_line",
    "sharp",
    "flat",
    "natural",
    "rest",
    "treble_clef",
    "bass_clef",
    "text_region",
    "slur_or_tie",
}


def load_gt_by_image(
    annotation_json: Path,
    taxonomy: str,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    data = read_json(annotation_json)
    by_image: dict[str, list[dict[str, Any]]] = {}
    for image in data["images"]:
        symbols: list[dict[str, Any]] = []
        for ann_id in image.get("ann_ids", []):
            ann = data["annotations"].get(str(ann_id))
            if ann is None:
                continue
            symbol_class = canonical_deepscores_class(ann, data["categories"], taxonomy=taxonomy)
            if symbol_class is None:
                continue
            x0, y0, x1, y1 = ann["a_bbox"]
            symbols.append(
                {
                    "id": f"gt_{ann_id}",
                    "class": symbol_class,
                    "bbox": [float(x0), float(y0), float(x1), float(y1)],
                    "source": "deepscores",
                }
            )
        by_image[image["filename"]] = symbols
    return data, by_image


def blank_accumulator(name: str) -> dict[str, Any]:
    return {
        "name": name,
        "pages": 0,
        "cer_ops": {"substitutions": 0, "deletions": 0, "insertions": 0, "matches": 0},
        "cer_ref_len": 0,
        "cer_hyp_len": 0,
        "symbol": {
            "gt": 0,
            "pred": 0,
            "matches": 0,
            "true_positive": 0,
            "substitutions": 0,
            "deletions": 0,
            "insertions": 0,
        },
        "line": {
            "gt": 0,
            "pred": 0,
            "matches": 0,
            "true_positive": 0,
            "substitutions": 0,
            "deletions": 0,
            "insertions": 0,
        },
        "gt_counts": {},
        "pred_counts": {},
    }


def add_counts(target: dict[str, int], source: dict[str, int]) -> None:
    for key, value in source.items():
        target[key] = target.get(key, 0) + int(value)


def match_counts(
    gt: list[dict[str, Any]],
    pred: list[dict[str, Any]],
    iou_threshold: float,
    include_classes: set[str] | None = None,
) -> dict[str, int]:
    gt_filtered = [item for item in gt if include_classes is None or item["class"] in include_classes]
    pred_filtered = [item for item in pred if include_classes is None or item["class"] in include_classes]
    pairs = []
    for gi, g in enumerate(gt_filtered):
        for pi, p in enumerate(pred_filtered):
            overlap = iou(g["bbox"], p["bbox"])
            if overlap >= iou_threshold:
                pairs.append((overlap, gi, pi))
    pairs.sort(reverse=True)
    gt_used: set[int] = set()
    pred_used: set[int] = set()
    matches = 0
    substitutions = 0
    true_positive = 0
    for _, gi, pi in pairs:
        if gi in gt_used or pi in pred_used:
            continue
        gt_used.add(gi)
        pred_used.add(pi)
        matches += 1
        if gt_filtered[gi]["class"] == pred_filtered[pi]["class"]:
            true_positive += 1
        else:
            substitutions += 1
    return {
        "gt": len(gt_filtered),
        "pred": len(pred_filtered),
        "matches": matches,
        "true_positive": true_positive,
        "substitutions": substitutions,
        "deletions": len(gt_filtered) - len(gt_used),
        "insertions": len(pred_filtered) - len(pred_used),
    }


def add_match_counts(target: dict[str, int], source: dict[str, int]) -> None:
    for key, value in source.items():
        target[key] += int(value)


def add_page_result(
    acc: dict[str, Any],
    gt_symbols: list[dict[str, Any]],
    pred_symbols: list[dict[str, Any]],
    gt_staff: list[dict[str, Any]],
    pred_staff: list[dict[str, Any]],
    symbol_iou: float,
    line_iou: float,
) -> None:
    gt_classes = {item["class"] for item in gt_symbols if item["class"] != "staff"}
    gt_eval = [item for item in gt_symbols if item["class"] in gt_classes]
    pred_eval = [item for item in pred_symbols if item["class"] in gt_classes]
    ref_seq = reading_sequence(gt_eval, include_classes=gt_classes)
    hyp_seq = reading_sequence(pred_eval, include_classes=gt_classes)
    cer_ops = levenshtein_ops(ref_seq, hyp_seq)
    for key, value in cer_ops.items():
        acc["cer_ops"][key] += value
    acc["cer_ref_len"] += len(ref_seq)
    acc["cer_hyp_len"] += len(hyp_seq)
    add_match_counts(acc["symbol"], match_counts(gt_eval, pred_eval, symbol_iou))
    add_match_counts(acc["line"], match_counts(gt_staff, pred_staff, line_iou, include_classes={"staff"}))
    add_counts(acc["gt_counts"], count_by_class(gt_eval))
    add_counts(acc["pred_counts"], count_by_class(pred_eval))
    acc["pages"] += 1


def finalize(acc: dict[str, Any]) -> dict[str, Any]:
    symbol = acc["symbol"]
    line = acc["line"]
    symbol_den = max(1, symbol["gt"])
    line_den = max(1, line["gt"])
    symbol_precision_den = symbol["true_positive"] + symbol["substitutions"] + symbol["insertions"]
    symbol_recall_den = symbol["true_positive"] + symbol["substitutions"] + symbol["deletions"]
    precision = symbol["true_positive"] / symbol_precision_den if symbol_precision_den else 0.0
    recall = symbol["true_positive"] / symbol_recall_den if symbol_recall_den else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    line_ser = (line["substitutions"] + line["deletions"] + line["insertions"]) / line_den
    return {
        **acc,
        "cer": error_rate_from_ops(acc["cer_ops"], acc["cer_ref_len"]),
        "ser": (symbol["substitutions"] + symbol["deletions"] + symbol["insertions"]) / symbol_den,
        "ler": line_ser,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def load_prediction_payload(path: Path | None) -> dict[str, list[dict[str, Any]]]:
    if path is None:
        return {}
    payload = read_json(path)
    by_image = payload.get("predictions_by_image", {})
    if isinstance(by_image, dict):
        return by_image
    raise ValueError(f"Unsupported prediction cache: {path}")


def prefer_detector_classes(symbols: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for symbol in symbols:
        item = dict(symbol)
        attributes = dict(item.get("attributes") or {})
        if attributes.get("detector_class"):
            item["class"] = str(attributes["detector_class"])
        item["attributes"] = attributes
        normalized.append(item)
    return normalized


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate V1 and V2 CER/SER/LER over a full DeepScores split.")
    parser.add_argument("--annotation-json", type=Path, default=Path("ds2_dense/ds2_dense/deepscores_test.json"))
    parser.add_argument("--image-root", type=Path, default=Path("ds2_dense/ds2_dense/images"))
    parser.add_argument("--deim-root", type=Path, default=Path(".local-tools/DEIM-main"))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--prediction-cache", type=Path)
    parser.add_argument("--out-json", type=Path, default=Path("outputs/v2_metrics/evaluation_v1_v2_dataset.json"))
    parser.add_argument("--prediction-out-json", type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--nms-iou", type=float, default=0.50)
    parser.add_argument("--max-detections", type=int, default=1500)
    parser.add_argument("--symbol-iou", type=float, default=0.30)
    parser.add_argument("--line-iou", type=float, default=0.50)
    parser.add_argument("--merge-rule-classes", default="all")
    parser.add_argument("--rule-fallback-iou", type=float, default=0.50)
    parser.add_argument("--max-pages", type=int, default=-1)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--taxonomy", choices=("base", "expanded"), default="base")
    parser.add_argument(
        "--class-names-json",
        type=Path,
        help="Optional COCO annotation or config summary JSON containing categories/class_names.",
    )
    parser.add_argument(
        "--prefer-detector-class",
        action="store_true",
        help="Evaluate detector fine classes from attributes.detector_class when available.",
    )
    args = parser.parse_args()

    class_names = load_class_names(args.class_names_json, args.taxonomy)
    data, gt_by_image = load_gt_by_image(args.annotation_json, args.taxonomy)
    images = data["images"] if args.max_pages < 0 else data["images"][: args.max_pages]
    cached_predictions = load_prediction_payload(args.prediction_cache)
    model = None
    if not cached_predictions:
        if args.config is None or args.checkpoint is None:
            raise ValueError("--config and --checkpoint are required when --prediction-cache is not supplied")
        model = load_deim_model(args.deim_root, args.config, args.checkpoint, args.device)

    accumulators = {
        "v1_rules": blank_accumulator("v1_rules"),
        "v2_detector_only": blank_accumulator("v2_detector_only"),
        "v2_detector_plus_rule_fallback": blank_accumulator("v2_detector_plus_rule_fallback"),
    }
    predictions_by_image: dict[str, list[dict[str, Any]]] = {}
    failures: list[dict[str, str]] = []
    rule_classes = parse_class_list(args.merge_rule_classes)
    started = time.time()

    for index, image_info in enumerate(images, start=1):
        image_name = image_info["filename"]
        input_path = args.image_root / image_name
        gt_all = gt_by_image[image_name]
        gt_staff = [item for item in gt_all if item["class"] == "staff"]
        gt_symbols = [item for item in gt_all if item["class"] != "staff"]

        if image_name in cached_predictions:
            detector_symbols = cached_predictions[image_name]
        else:
            assert model is not None
            _, detector_symbols = predict(
                model,
                input_path,
                args.device,
                args.threshold,
                args.nms_iou,
                args.max_detections,
                class_names,
            )
        detector_metric_symbols = prefer_detector_classes(detector_symbols) if args.prefer_detector_class else detector_symbols
        predictions_by_image[image_name] = detector_symbols
        try:
            v1_result = run_v1_symbol_pipeline(input_path)
        except Exception as exc:
            failures.append(
                {
                    "image": image_name,
                    "stage": "v1_staff_geometry",
                    "error": str(exc),
                }
            )
            v1_symbols = []
            pred_staff = []
            detector_symbols = [dict(item) for item in detector_metric_symbols]
            hybrid_symbols = [dict(item) for item in detector_metric_symbols]
        else:
            v1_symbols = enrich_symbols_with_staff_geometry(v1_result_to_weak_symbols(v1_result), v1_result["staves"])
            pred_staff = pred_staff_symbols(v1_result)
            detector_symbols = enrich_symbols_with_staff_geometry(detector_metric_symbols, v1_result["staves"])
            hybrid_symbols = merge_rule_symbols(
                detector_symbols,
                v1_symbols,
                rule_classes,
                args.rule_fallback_iou,
                "detector",
            )
            hybrid_symbols = enrich_symbols_with_staff_geometry(hybrid_symbols, v1_result["staves"])

        add_page_result(
            accumulators["v1_rules"],
            gt_symbols,
            v1_symbols,
            gt_staff,
            pred_staff,
            args.symbol_iou,
            args.line_iou,
        )
        add_page_result(
            accumulators["v2_detector_only"],
            gt_symbols,
            detector_symbols,
            gt_staff,
            pred_staff,
            args.symbol_iou,
            args.line_iou,
        )
        add_page_result(
            accumulators["v2_detector_plus_rule_fallback"],
            gt_symbols,
            hybrid_symbols,
            gt_staff,
            pred_staff,
            args.symbol_iou,
            args.line_iou,
        )

        if args.progress_every > 0 and (index % args.progress_every == 0 or index == len(images)):
            elapsed = time.time() - started
            print(
                json.dumps(
                    {
                        "progress": f"{index}/{len(images)}",
                        "image": image_name,
                        "elapsed_sec": round(elapsed, 1),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    results = {
        "annotation_json": str(args.annotation_json),
        "image_root": str(args.image_root),
        "taxonomy": args.taxonomy,
        "class_names": class_names,
        "prefer_detector_class": args.prefer_detector_class,
        "pages": len(images),
        "failures": failures,
        "definitions": {
            "CER": "Levenshtein edit distance over reading-order symbol-class token sequence, divided by GT token count.",
            "SER": "Object-level symbol error rate: (class substitutions + unmatched GT + unmatched predictions) / GT symbols, after greedy IoU matching.",
            "LER": "Staff/staff-region error rate using the same SER formula on staff boxes.",
            "symbol_iou": args.symbol_iou,
            "line_iou": args.line_iou,
        },
        "models": [finalize(item) for item in accumulators.values()],
    }
    write_json(args.out_json, results)
    if args.prediction_out_json:
        write_json(
            args.prediction_out_json,
            {
                "version": "v2_deim_dataset_predictions",
                "annotation_json": str(args.annotation_json),
                "config": str(args.config) if args.config else None,
                "checkpoint": str(args.checkpoint) if args.checkpoint else None,
                "threshold": args.threshold,
                "nms_iou": args.nms_iou,
                "max_detections": args.max_detections,
                "predictions_by_image": predictions_by_image,
            },
        )
    print(
        json.dumps(
            {
                "out_json": str(args.out_json),
                "pages": len(images),
                "models": [
                    {
                        "name": item["name"],
                        "CER": item["cer"],
                        "SER": item["ser"],
                        "LER": item["ler"],
                        "precision": item["precision"],
                        "recall": item["recall"],
                        "f1": item["f1"],
                    }
                    for item in results["models"]
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
