from __future__ import annotations

import argparse
import json
import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
ANNOTATIONS = ROOT / "data/muscima_pp/MUSCIMA-pp_v2.0/v2.0/data/annotations"
DEFAULT_OUT = ROOT / "outputs/muscima_beam_notehead_relation_head"
NOTEHEAD_CLASSES = frozenset({"noteheadFull", "noteheadFullSmall"})
BOX_FEATURE_NAMES = (
    "x_outside_noteheight",
    "x_position_in_beam",
    "x_position_centered_abs",
    "horizontal_overlap_note_width",
    "vertical_bbox_gap_noteheight",
    "signed_center_dy_noteheight",
    "center_dy_abs_noteheight",
    "log_beam_width_noteheight",
    "beam_height_noteheight",
    "note_width_noteheight",
)
MASK_FEATURE_NAMES = (
    "mask_vertical_gap_noteheight",
    "signed_mask_center_dy_noteheight",
    "mask_center_dy_abs_noteheight",
    "mask_slope",
    "mask_fill_ratio",
    "mask_fit_thickness_noteheight",
)


@dataclass
class RelationDataset:
    box_features: np.ndarray
    mask_features: np.ndarray
    labels: np.ndarray
    gold_edges: int
    missed_gold_edges: int
    pages: list[str]


def writer_id(path: Path) -> int:
    match = re.search(r"W-(\d+)", path.stem)
    if not match:
        raise ValueError(f"Cannot parse writer id: {path.name}")
    return int(match.group(1))


def parse_writer_range(value: str) -> set[int]:
    result: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            left, right = part.split("-", 1)
            result.update(range(int(left), int(right) + 1))
        else:
            result.add(int(part))
    return result


def decode_rle(mask_text: str, height: int, width: int) -> np.ndarray:
    chunks = []
    total = 0
    for token in mask_text.split():
        value, count = token.split(":", 1)
        count_int = int(count)
        chunks.append(np.full(count_int, int(value), dtype=np.uint8))
        total += count_int
    if total != height * width:
        raise ValueError(f"RLE length {total} != {height}x{width}")
    return np.concatenate(chunks).reshape(height, width).astype(bool)


def parse_nodes(path: Path) -> list[dict[str, Any]]:
    result = []
    for element in ET.parse(path).getroot().findall("Node"):
        height = int(element.findtext("Height", "0"))
        width = int(element.findtext("Width", "0"))
        result.append(
            {
                "id": element.findtext("Id", ""),
                "class": element.findtext("ClassName", ""),
                "top": int(element.findtext("Top", "0")),
                "left": int(element.findtext("Left", "0")),
                "height": height,
                "width": width,
                "mask": decode_rle(element.findtext("Mask", ""), height, width),
                "inlinks": (element.findtext("Inlinks") or "").split(),
            }
        )
    return result


def is_candidate(beam: dict[str, Any], notehead: dict[str, Any]) -> bool:
    note_width = max(1.0, float(notehead["width"]))
    note_height = max(1.0, float(notehead["height"]))
    note_x = float(notehead["left"]) + 0.5 * note_width
    note_y = float(notehead["top"]) + 0.5 * note_height
    beam_x = float(beam["left"]) + 0.5 * float(beam["width"])
    beam_y = float(beam["top"]) + 0.5 * float(beam["height"])
    horizontal = beam["left"] - 4.0 * note_width <= note_x <= beam["left"] + beam["width"] + 4.0 * note_width
    vertical = abs(note_y - beam_y) <= 16.0 * note_height
    return bool(horizontal and vertical)


def pair_features(beam: dict[str, Any], notehead: dict[str, Any]) -> tuple[list[float], list[float]]:
    beam_left = float(beam["left"])
    beam_top = float(beam["top"])
    beam_width = max(1.0, float(beam["width"]))
    beam_height = max(1.0, float(beam["height"]))
    beam_right = beam_left + beam_width
    beam_bottom = beam_top + beam_height
    note_left = float(notehead["left"])
    note_top = float(notehead["top"])
    note_width = max(1.0, float(notehead["width"]))
    note_height = max(1.0, float(notehead["height"]))
    note_right = note_left + note_width
    note_bottom = note_top + note_height
    note_x = note_left + 0.5 * note_width
    note_y = note_top + 0.5 * note_height

    x_outside = max(beam_left - note_x, note_x - beam_right, 0.0) / note_height
    x_relative = (note_x - beam_left) / beam_width
    x_overlap = max(0.0, min(beam_right, note_right) - max(beam_left, note_left)) / note_width
    y_gap = max(note_top - beam_bottom, beam_top - note_bottom, 0.0) / note_height
    center_dy = (note_y - (beam_top + 0.5 * beam_height)) / note_height
    box = [
        x_outside,
        x_relative,
        abs(x_relative - 0.5),
        x_overlap,
        y_gap,
        center_dy,
        abs(center_dy),
        math.log1p(beam_width / note_height),
        beam_height / note_height,
        note_width / note_height,
    ]

    mask = np.asarray(beam["mask"], dtype=bool)
    mask_left = float(beam.get("mask_left", beam_left))
    mask_top = float(beam.get("mask_top", beam_top))
    local_y, local_x = np.nonzero(mask)
    if len(local_x):
        global_x = local_x.astype(np.float64) + mask_left
        global_y = local_y.astype(np.float64) + mask_top
        design = np.column_stack((global_x, np.ones_like(global_x)))
        slope, intercept = np.linalg.lstsq(design, global_y, rcond=None)[0]
        mask_y = float(slope * note_x + intercept)
        mask_gap = max(note_top - mask_y, mask_y - note_bottom, 0.0) / note_height
        mask_center_dy = (note_y - mask_y) / note_height
        fill_ratio = len(local_x) / max(1.0, beam_width * beam_height)
        thickness = float(np.std(global_y - (slope * global_x + intercept))) / note_height
    else:
        slope = 0.0
        mask_gap = y_gap
        mask_center_dy = center_dy
        fill_ratio = 0.0
        thickness = 0.0
    mask_features = [
        mask_gap,
        mask_center_dy,
        abs(mask_center_dy),
        float(slope),
        fill_ratio,
        thickness,
    ]
    return box, box + mask_features


def build_dataset(
    paths: list[Path],
    mask_overrides: dict[tuple[str, str], dict[str, Any]] | None = None,
) -> RelationDataset:
    box_rows: list[list[float]] = []
    mask_rows: list[list[float]] = []
    labels: list[int] = []
    gold_edges = 0
    missed = 0
    for path in paths:
        nodes = parse_nodes(path)
        by_id = {str(node["id"]): node for node in nodes}
        noteheads = [node for node in nodes if node["class"] in NOTEHEAD_CLASSES]
        beams = [node for node in nodes if node["class"] == "beam"]
        for beam in beams:
            feature_beam = beam
            if mask_overrides is not None:
                override = mask_overrides.get((path.stem, str(beam["id"])))
                if override is not None:
                    feature_beam = {**beam, **override}
            positives = {
                linked_id
                for linked_id in beam["inlinks"]
                if linked_id in by_id and by_id[linked_id]["class"] in NOTEHEAD_CLASSES
            }
            gold_edges += len(positives)
            covered: set[str] = set()
            for notehead in noteheads:
                if not is_candidate(beam, notehead):
                    continue
                box, with_mask = pair_features(feature_beam, notehead)
                label = int(str(notehead["id"]) in positives)
                box_rows.append(box)
                mask_rows.append(with_mask)
                labels.append(label)
                if label:
                    covered.add(str(notehead["id"]))
            missed += len(positives - covered)
    return RelationDataset(
        box_features=np.asarray(box_rows, dtype=np.float64),
        mask_features=np.asarray(mask_rows, dtype=np.float64),
        labels=np.asarray(labels, dtype=np.int64),
        gold_edges=gold_edges,
        missed_gold_edges=missed,
        pages=[path.stem for path in paths],
    )


def edge_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float, gold_edges: int) -> dict[str, Any]:
    predicted = scores >= threshold
    true_positive = int(np.logical_and(labels == 1, predicted).sum())
    false_positive = int(np.logical_and(labels == 0, predicted).sum())
    false_negative = int(gold_edges - true_positive)
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, gold_edges)
    f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
    return {
        "threshold": float(threshold),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": precision,
        "recall": recall,
        "edge_f1": f1,
    }


def calibrate_threshold(labels: np.ndarray, scores: np.ndarray, gold_edges: int) -> dict[str, Any]:
    candidates = [edge_metrics(labels, scores, float(value), gold_edges) for value in np.linspace(0.01, 0.99, 99)]
    return max(candidates, key=lambda item: (item["edge_f1"], item["precision"], item["threshold"]))


def calibrate_pruning_threshold(
    labels: np.ndarray,
    scores: np.ndarray,
    gold_edges: int,
    minimum_recall: float = 0.99,
) -> dict[str, Any]:
    candidates = [edge_metrics(labels, scores, float(value), gold_edges) for value in np.linspace(0.01, 0.99, 99)]
    feasible = [item for item in candidates if item["recall"] >= minimum_recall]
    if not feasible:
        return min(candidates, key=lambda item: (abs(item["recall"] - minimum_recall), -item["precision"]))
    return max(feasible, key=lambda item: (item["precision"], item["threshold"]))


def select_paths(writers: set[int]) -> list[Path]:
    return [path for path in sorted(ANNOTATIONS.glob("*.xml")) if writer_id(path) in writers]


def dataset_summary(dataset: RelationDataset) -> dict[str, Any]:
    return {
        "pages": len(dataset.pages),
        "candidates": int(len(dataset.labels)),
        "positive_candidates": int(dataset.labels.sum()),
        "gold_edges": int(dataset.gold_edges),
        "missed_gold_edges": int(dataset.missed_gold_edges),
        "candidate_recall": (dataset.gold_edges - dataset.missed_gold_edges) / max(1, dataset.gold_edges),
    }


def main() -> None:
    from joblib import dump
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    parser = argparse.ArgumentParser(description="Train writer-disjoint MUSCIMA++ beam-notehead relation heads.")
    parser.add_argument("--train-writers", default="1-30")
    parser.add_argument("--val-writers", default="31-40")
    parser.add_argument("--test-writers", default="41-50")
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    split_writers = {
        "train": parse_writer_range(args.train_writers),
        "validation": parse_writer_range(args.val_writers),
        "source_test": parse_writer_range(args.test_writers),
    }
    if any(
        split_writers[left] & split_writers[right]
        for left, right in (("train", "validation"), ("train", "source_test"), ("validation", "source_test"))
    ):
        raise RuntimeError("Writer leakage across relation-head splits")
    datasets = {name: build_dataset(select_paths(writers)) for name, writers in split_writers.items()}
    train = datasets["train"]
    validation = datasets["validation"]
    source_test = datasets["source_test"]

    models = {
        "box_only_head": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0, random_state=args.seed),
        ),
        "box_mask_head": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0, random_state=args.seed),
        ),
    }
    models["box_only_head"].fit(train.box_features, train.labels)
    models["box_mask_head"].fit(train.mask_features, train.labels)

    validation_scores = {
        "geometry_rule": np.exp(-validation.box_features[:, 0] - 0.25 * validation.box_features[:, 4]),
        "box_only_head": models["box_only_head"].predict_proba(validation.box_features)[:, 1],
        "box_mask_head": models["box_mask_head"].predict_proba(validation.mask_features)[:, 1],
    }
    test_scores = {
        "geometry_rule": np.exp(-source_test.box_features[:, 0] - 0.25 * source_test.box_features[:, 4]),
        "box_only_head": models["box_only_head"].predict_proba(source_test.box_features)[:, 1],
        "box_mask_head": models["box_mask_head"].predict_proba(source_test.mask_features)[:, 1],
    }
    calibrated = {
        name: calibrate_threshold(validation.labels, scores, validation.gold_edges)
        for name, scores in validation_scores.items()
    }
    pruning_calibrated = calibrate_pruning_threshold(
        validation.labels,
        validation_scores["box_mask_head"],
        validation.gold_edges,
        minimum_recall=0.99,
    )
    test_results = {
        name: edge_metrics(source_test.labels, scores, calibrated[name]["threshold"], source_test.gold_edges)
        for name, scores in test_scores.items()
    }

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    dump(models["box_only_head"], out / "box_only_head.joblib")
    dump(models["box_mask_head"], out / "box_mask_head.joblib")
    summary = {
        "protocol": "MUSCIMA++ writer-disjoint beam-notehead relation prediction; validation-only threshold calibration",
        "splits": {name: dataset_summary(dataset) for name, dataset in datasets.items()},
        "writers": {name: sorted(values) for name, values in split_writers.items()},
        "feature_names": {
            "box_only_head": list(BOX_FEATURE_NAMES),
            "box_mask_head": list(BOX_FEATURE_NAMES + MASK_FEATURE_NAMES),
        },
        "validation": calibrated,
        "validation_safe_pruning": {
            "minimum_recall": 0.99,
            "box_mask_head": pruning_calibrated,
        },
        "source_test": test_results,
        "source_test_deltas": {
            "box_head_over_rule": test_results["box_only_head"]["edge_f1"] - test_results["geometry_rule"]["edge_f1"],
            "mask_features_over_box_head": test_results["box_mask_head"]["edge_f1"] - test_results["box_only_head"]["edge_f1"],
        },
        "publication_status": "source_domain_relation_gate_gt_masks_not_target_domain_result",
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"source_test": test_results, "deltas": summary["source_test_deltas"]}, indent=2))


if __name__ == "__main__":
    main()
