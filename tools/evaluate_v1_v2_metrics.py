from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from omr_v2_common import (
    canonical_deepscores_class,
    count_by_class,
    enrich_symbols_with_staff_geometry,
    read_json,
    run_v1_symbol_pipeline,
    v1_result_to_weak_symbols,
    write_json,
)


TOKEN_CODES = {
    "filled_notehead": "n",
    "open_notehead": "o",
    "stem": "s",
    "beam": "b",
    "barline": "|",
    "ledger_line": "l",
    "sharp": "#",
    "flat": "f",
    "natural": "=",
    "rest": "r",
    "treble_clef": "g",
    "bass_clef": "F",
    "text_region": "x",
    "slur_or_tie": "u",
}


def bbox_xyxy_area(box: list[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def iou(a: list[float], b: list[float]) -> float:
    x0 = max(a[0], b[0])
    y0 = max(a[1], b[1])
    x1 = min(a[2], b[2])
    y1 = min(a[3], b[3])
    inter = bbox_xyxy_area([x0, y0, x1, y1])
    union = bbox_xyxy_area(a) + bbox_xyxy_area(b) - inter
    return inter / union if union > 0 else 0.0


def levenshtein_ops(ref: list[str], hyp: list[str]) -> dict[str, int]:
    n = len(ref)
    m = len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    back = [[""] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
        back[i][0] = "D"
    for j in range(1, m + 1):
        dp[0][j] = j
        back[0][j] = "I"
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                best = (dp[i - 1][j - 1], "M")
            else:
                best = (dp[i - 1][j - 1] + 1, "S")
            delete = (dp[i - 1][j] + 1, "D")
            insert = (dp[i][j - 1] + 1, "I")
            best = min(best, delete, insert, key=lambda item: item[0])
            dp[i][j], back[i][j] = best
    i = n
    j = m
    ops = {"substitutions": 0, "deletions": 0, "insertions": 0, "matches": 0}
    while i > 0 or j > 0:
        op = back[i][j]
        if op == "M":
            ops["matches"] += 1
            i -= 1
            j -= 1
        elif op == "S":
            ops["substitutions"] += 1
            i -= 1
            j -= 1
        elif op == "D":
            ops["deletions"] += 1
            i -= 1
        else:
            ops["insertions"] += 1
            j -= 1
    return ops


def error_rate_from_ops(ops: dict[str, int], ref_len: int) -> float:
    if ref_len == 0:
        return 0.0 if ops["insertions"] == 0 else 1.0
    return (ops["substitutions"] + ops["deletions"] + ops["insertions"]) / ref_len


def load_deepscores_page(
    annotation_json: Path,
    image_name: str,
    taxonomy: str,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    data = json.loads(annotation_json.read_text(encoding="utf-8"))
    matches = [img for img in data["images"] if img["filename"] == image_name]
    if not matches:
        raise ValueError(f"{image_name} not found in {annotation_json}")
    image_info = matches[0]
    gt = []
    for ann_id in image_info.get("ann_ids", []):
        ann = data["annotations"].get(str(ann_id))
        if ann is None:
            continue
        symbol_class = canonical_deepscores_class(ann, data["categories"], taxonomy=taxonomy)
        if symbol_class is None:
            continue
        x0, y0, x1, y1 = ann["a_bbox"]
        gt.append(
            {
                "id": f"gt_{ann_id}",
                "class": symbol_class,
                "bbox": [float(x0), float(y0), float(x1), float(y1)],
                "source": "deepscores",
            }
        )
    return data, image_info, gt


def load_v2_symbols(path: Path, min_confidence: float, prefer_detector_class: bool) -> list[dict[str, Any]]:
    payload = read_json(path)
    symbols = payload["symbols"] if isinstance(payload, dict) and "symbols" in payload else payload
    loaded = []
    for idx, symbol in enumerate(symbols):
        confidence = float(symbol.get("confidence", symbol.get("score", 1.0)))
        if confidence < min_confidence:
            continue
        if "class" not in symbol or "bbox" not in symbol:
            continue
        attributes = dict(symbol.get("attributes") or {})
        symbol_class = symbol["class"]
        if prefer_detector_class and attributes.get("detector_class"):
            symbol_class = str(attributes["detector_class"])
        loaded.append(
            {
                "id": symbol.get("id", f"v2_{idx:06d}"),
                "class": symbol_class,
                "bbox": [float(v) for v in symbol["bbox"]],
                "confidence": confidence,
                "source": symbol.get("source", "v2"),
                "attributes": attributes,
            }
        )
    return loaded


def pred_staff_symbols(v1_result: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "id": f"pred_staff_{staff.index}",
            "class": "staff",
            "bbox": [
                float(staff.x0),
                float(staff.y0 - 0.25 * staff.space),
                float(staff.x1),
                float(staff.y1 + 0.25 * staff.space),
            ],
            "source": "v1_staff_geometry",
        }
        for staff in v1_result["staves"]
    ]


def reading_sequence(symbols: list[dict[str, Any]], include_classes: set[str] | None = None) -> list[str]:
    filtered = [
        symbol
        for symbol in symbols
        if symbol["class"] != "staff" and (include_classes is None or symbol["class"] in include_classes)
    ]
    filtered.sort(key=lambda item: (0.5 * (item["bbox"][1] + item["bbox"][3]), 0.5 * (item["bbox"][0] + item["bbox"][2]), item["class"]))
    return [TOKEN_CODES.get(item["class"], item["class"]) for item in filtered]


def match_symbols(
    gt: list[dict[str, Any]],
    pred: list[dict[str, Any]],
    iou_threshold: float,
    include_classes: set[str] | None = None,
) -> dict[str, Any]:
    gt_filtered = [item for item in gt if include_classes is None or item["class"] in include_classes]
    pred_filtered = [item for item in pred if include_classes is None or item["class"] in include_classes]
    pairs = []
    for gi, g in enumerate(gt_filtered):
        for pi, p in enumerate(pred_filtered):
            overlap = iou(g["bbox"], p["bbox"])
            if overlap >= iou_threshold:
                pairs.append((overlap, gi, pi))
    pairs.sort(reverse=True)
    gt_used = set()
    pred_used = set()
    matches = []
    substitutions = 0
    true_positive = 0
    for overlap, gi, pi in pairs:
        if gi in gt_used or pi in pred_used:
            continue
        gt_used.add(gi)
        pred_used.add(pi)
        same = gt_filtered[gi]["class"] == pred_filtered[pi]["class"]
        if same:
            true_positive += 1
        else:
            substitutions += 1
        matches.append(
            {
                "iou": overlap,
                "gt_class": gt_filtered[gi]["class"],
                "pred_class": pred_filtered[pi]["class"],
                "same_class": same,
            }
        )
    deletions = len(gt_filtered) - len(gt_used)
    insertions = len(pred_filtered) - len(pred_used)
    denominator = max(1, len(gt_filtered))
    ser = (substitutions + deletions + insertions) / denominator
    precision_den = true_positive + substitutions + insertions
    recall_den = true_positive + substitutions + deletions
    precision = true_positive / precision_den if precision_den else 0.0
    recall = true_positive / recall_den if recall_den else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "gt": len(gt_filtered),
        "pred": len(pred_filtered),
        "matches": len(matches),
        "true_positive": true_positive,
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
        "ser": ser,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "match_details": matches[:200],
    }


def per_class_summary(gt: list[dict[str, Any]], pred: list[dict[str, Any]], iou_threshold: float) -> dict[str, Any]:
    classes = sorted({item["class"] for item in gt if item["class"] != "staff"} | {item["class"] for item in pred if item["class"] != "staff"})
    return {
        name: {
            key: value
            for key, value in match_symbols(gt, pred, iou_threshold, include_classes={name}).items()
            if key != "match_details"
        }
        for name in classes
    }


def evaluate_prediction(
    name: str,
    gt_symbols: list[dict[str, Any]],
    pred_symbols: list[dict[str, Any]],
    gt_staff: list[dict[str, Any]],
    pred_staff: list[dict[str, Any]],
    symbol_iou: float,
    line_iou: float,
) -> dict[str, Any]:
    gt_classes = {item["class"] for item in gt_symbols if item["class"] != "staff"}
    pred_eval = [item for item in pred_symbols if item["class"] in gt_classes]
    gt_eval = [item for item in gt_symbols if item["class"] in gt_classes]
    symbol_metrics = match_symbols(gt_eval, pred_eval, symbol_iou)
    line_metrics = match_symbols(gt_staff, pred_staff, line_iou, include_classes={"staff"})
    ref_seq = reading_sequence(gt_eval, include_classes=gt_classes)
    hyp_seq = reading_sequence(pred_eval, include_classes=gt_classes)
    cer_ops = levenshtein_ops(ref_seq, hyp_seq)
    return {
        "name": name,
        "cer": error_rate_from_ops(cer_ops, len(ref_seq)),
        "cer_ops": cer_ops,
        "cer_ref_len": len(ref_seq),
        "cer_hyp_len": len(hyp_seq),
        "ser": symbol_metrics["ser"],
        "symbol_metrics": {key: value for key, value in symbol_metrics.items() if key != "match_details"},
        "ler": line_metrics["ser"],
        "line_metrics": {key: value for key, value in line_metrics.items() if key != "match_details"},
        "gt_counts": count_by_class(gt_eval),
        "pred_counts": count_by_class(pred_eval),
        "per_class": per_class_summary(gt_eval, pred_eval, symbol_iou),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate V1 and current V2 outputs with CER/SER/LER against DeepScores GT.")
    parser.add_argument("--annotation-json", type=Path, default=Path("ds2_dense/ds2_dense/deepscores_train.json"))
    parser.add_argument("--image-root", type=Path, default=Path("ds2_dense/ds2_dense/images"))
    parser.add_argument("--image-name", default="lg-2267728-aug-beethoven--page-2.png")
    parser.add_argument("--symbol-iou", type=float, default=0.30)
    parser.add_argument("--line-iou", type=float, default=0.50)
    parser.add_argument("--v2-symbols-json", type=Path, help="Optional real V2 symbol JSON to evaluate.")
    parser.add_argument("--v2-name", default="v2_current")
    parser.add_argument("--v2-min-confidence", type=float, default=0.05)
    parser.add_argument("--taxonomy", choices=("base", "expanded"), default="base")
    parser.add_argument(
        "--prefer-detector-class",
        action="store_true",
        help="Evaluate detector fine classes from attributes.detector_class when available.",
    )
    parser.add_argument("--out-json", type=Path, default=Path("outputs/v2_metrics/evaluation_v1_v2.json"))
    args = parser.parse_args()

    _, image_info, gt_all = load_deepscores_page(args.annotation_json, args.image_name, args.taxonomy)
    gt_staff = [item for item in gt_all if item["class"] == "staff"]
    gt_symbols = [item for item in gt_all if item["class"] != "staff"]
    input_path = args.image_root / args.image_name
    v1_result = run_v1_symbol_pipeline(input_path)
    v1_symbols = enrich_symbols_with_staff_geometry(v1_result_to_weak_symbols(v1_result), v1_result["staves"])
    staff_pred = pred_staff_symbols(v1_result)

    if args.v2_symbols_json:
        v2_symbols = load_v2_symbols(args.v2_symbols_json, args.v2_min_confidence, args.prefer_detector_class)
        v2_name = args.v2_name
    else:
        v2_symbols = [dict(item, source="v2_current_weak_detector") for item in v1_symbols]
        v2_name = "v2_current_weak"
    v2_staff_pred = staff_pred

    results = {
        "image": image_info,
        "input_path": str(input_path),
        "annotation_json": str(args.annotation_json),
        "definitions": {
            "CER": "Levenshtein edit distance over reading-order symbol-class token sequence, divided by GT token count.",
            "SER": "Object-level symbol error rate: (class substitutions + unmatched GT + unmatched predictions) / GT symbols, after greedy IoU matching.",
            "LER": "Staff-line/staff-region error rate using the same SER formula on staff boxes.",
            "symbol_iou": args.symbol_iou,
            "line_iou": args.line_iou,
        },
        "gt_counts_all_mapped": count_by_class(gt_all),
        "models": [
            evaluate_prediction("v1_rules", gt_symbols, v1_symbols, gt_staff, staff_pred, args.symbol_iou, args.line_iou),
            evaluate_prediction(v2_name, gt_symbols, v2_symbols, gt_staff, v2_staff_pred, args.symbol_iou, args.line_iou),
        ],
    }
    write_json(args.out_json, results)
    print(
        json.dumps(
            {
                "out_json": str(args.out_json),
                "image": args.image_name,
                "gt_counts": results["gt_counts_all_mapped"],
                "models": [
                    {
                        "name": item["name"],
                        "CER": item["cer"],
                        "SER": item["ser"],
                        "LER": item["ler"],
                        "precision": item["symbol_metrics"]["precision"],
                        "recall": item["symbol_metrics"]["recall"],
                        "f1": item["symbol_metrics"]["f1"],
                    }
                    for item in results["models"]
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
