from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


RELATION_SPECS = {
    "notehead_stem_attachment": ({"filled_notehead", "open_notehead"}, {"stem"}),
    "beam_stem_group": ({"beam"}, {"stem"}),
    "ledger_line_notehead_attachment": ({"ledger_line"}, {"filled_notehead", "open_notehead"}),
    "slur_tie_notehead_endpoints": ({"slur_or_tie"}, {"filled_notehead", "open_notehead"}),
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def sample_dirs(source_root: Path) -> list[Path]:
    return sorted(
        path
        for path in source_root.iterdir()
        if path.is_dir() and (path / "symbols" / "symbols_v2_1_shapes.json").exists()
    )


def bbox(symbol: dict[str, Any]) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = [float(value) for value in symbol["bbox"]]
    return x0, y0, x1, y1


def center(symbol: dict[str, Any]) -> tuple[float, float]:
    x0, y0, x1, y1 = bbox(symbol)
    return 0.5 * (x0 + x1), 0.5 * (y0 + y1)


def staff(symbol: dict[str, Any]) -> int | None:
    value = (symbol.get("attributes") or {}).get("staff")
    return int(value) if value is not None else None


def staff_space(symbol: dict[str, Any]) -> float:
    value = (symbol.get("attributes") or {}).get("staff_space")
    return float(value) if value is not None else 12.0


def overlap_1d(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def gap_1d(a0: float, a1: float, b0: float, b1: float) -> float:
    if a1 < b0:
        return b0 - a1
    if b1 < a0:
        return a0 - b1
    return 0.0


def features(source: dict[str, Any], target: dict[str, Any], relation_type: str) -> list[float]:
    sx0, sy0, sx1, sy1 = bbox(source)
    tx0, ty0, tx1, ty1 = bbox(target)
    scx, scy = center(source)
    tcx, tcy = center(target)
    space = max(1.0, 0.5 * (staff_space(source) + staff_space(target)))
    sw, sh = sx1 - sx0, sy1 - sy0
    tw, th = tx1 - tx0, ty1 - ty0
    x_overlap = overlap_1d(sx0, sx1, tx0, tx1) / max(1.0, min(sw, tw))
    y_overlap = overlap_1d(sy0, sy1, ty0, ty1) / max(1.0, min(sh, th))
    x_gap = gap_1d(sx0, sx1, tx0, tx1) / space
    y_gap = gap_1d(sy0, sy1, ty0, ty1) / space
    dx = (tcx - scx) / space
    dy = (tcy - scy) / space
    type_index = list(RELATION_SPECS).index(relation_type)
    return [
        dx,
        dy,
        abs(dx),
        abs(dy),
        math.hypot(dx, dy),
        sw / space,
        sh / space,
        tw / space,
        th / space,
        x_overlap,
        y_overlap,
        x_gap,
        y_gap,
        1.0 if staff(source) is not None and staff(source) == staff(target) else 0.0,
        float(source.get("confidence") or 0.0),
        float(target.get("confidence") or 0.0),
        float(type_index),
    ]


def compatible_targets(
    source: dict[str, Any],
    symbols: list[dict[str, Any]],
    target_classes: set[str],
    *,
    max_distance_spaces: float,
) -> list[dict[str, Any]]:
    scx, scy = center(source)
    source_staff = staff(source)
    source_space = max(1.0, staff_space(source))
    candidates = []
    for target in symbols:
        if target.get("class") not in target_classes:
            continue
        if target["id"] == source["id"]:
            continue
        if source_staff is not None and staff(target) is not None and staff(target) != source_staff:
            continue
        tcx, tcy = center(target)
        dist = math.hypot(tcx - scx, tcy - scy) / source_space
        if dist <= max_distance_spaces:
            candidates.append((dist, target))
    return [target for _, target in sorted(candidates, key=lambda item: item[0])]


def extract_examples(
    sample: str,
    shapes_path: Path,
    *,
    max_neg_per_positive: int,
    max_distance_spaces: float,
    rng: random.Random,
) -> list[dict[str, Any]]:
    payload = read_json(shapes_path)
    symbols = payload.get("symbols", [])
    by_id = {str(symbol["id"]): symbol for symbol in symbols}
    positive_pairs: dict[str, set[tuple[str, str]]] = {name: set() for name in RELATION_SPECS}
    examples: list[dict[str, Any]] = []

    for relation in payload.get("relations", []):
        rel_type = str(relation.get("type") or "")
        if rel_type not in RELATION_SPECS:
            continue
        source = by_id.get(str(relation.get("source")))
        if source is None:
            continue
        for target_id in relation.get("targets", []) or []:
            target = by_id.get(str(target_id))
            if target is None:
                continue
            positive_pairs[rel_type].add((str(source["id"]), str(target["id"])))
            examples.append(
                {
                    "sample": sample,
                    "relation_type": rel_type,
                    "source": str(source["id"]),
                    "target": str(target["id"]),
                    "label": 1,
                    "features": features(source, target, rel_type),
                    "heuristic_score": float(relation.get("score") or 0.0),
                }
            )

    for rel_type, pairs in positive_pairs.items():
        source_classes, target_classes = RELATION_SPECS[rel_type]
        by_source: dict[str, set[str]] = {}
        for source_id, target_id in pairs:
            by_source.setdefault(source_id, set()).add(target_id)
        for source_id, positive_targets in by_source.items():
            source = by_id.get(source_id)
            if source is None or source.get("class") not in source_classes:
                continue
            candidates = [
                target
                for target in compatible_targets(
                    source,
                    symbols,
                    target_classes,
                    max_distance_spaces=max_distance_spaces,
                )
                if str(target["id"]) not in positive_targets
            ]
            if len(candidates) > max_neg_per_positive:
                candidates = rng.sample(candidates[: max_neg_per_positive * 4], min(max_neg_per_positive, len(candidates)))
            for target in candidates:
                examples.append(
                    {
                        "sample": sample,
                        "relation_type": rel_type,
                        "source": source_id,
                        "target": str(target["id"]),
                        "label": 0,
                        "features": features(source, target, rel_type),
                        "heuristic_score": 0.0,
                    }
                )
    return examples


def split_samples(samples: list[str], test_fraction: float, rng: random.Random) -> tuple[set[str], set[str]]:
    shuffled = list(samples)
    rng.shuffle(shuffled)
    test_count = max(1, int(round(len(shuffled) * test_fraction)))
    test = set(shuffled[:test_count])
    train = set(shuffled[test_count:])
    if not train:
        train = set(shuffled[:-1])
        test = {shuffled[-1]}
    return train, test


def train_one_relation(
    relation_type: str,
    examples: list[dict[str, Any]],
    train_samples: set[str],
    test_samples: set[str],
) -> tuple[Any | None, dict[str, Any]]:
    rel_examples = [item for item in examples if item["relation_type"] == relation_type]
    train = [item for item in rel_examples if item["sample"] in train_samples]
    test = [item for item in rel_examples if item["sample"] in test_samples]
    train_labels = [int(item["label"]) for item in train]
    test_labels = [int(item["label"]) for item in test]
    if len(set(train_labels)) < 2 or len(set(test_labels)) < 2:
        return None, {
            "relation_type": relation_type,
            "train_examples": len(train),
            "test_examples": len(test),
            "status": "skipped_insufficient_class_balance",
        }
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            ("logreg", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )
    train_x = np.asarray([item["features"] for item in train], dtype=np.float32)
    test_x = np.asarray([item["features"] for item in test], dtype=np.float32)
    train_y = np.asarray(train_labels, dtype=np.int64)
    test_y = np.asarray(test_labels, dtype=np.int64)
    model.fit(train_x, train_y)
    probabilities = model.predict_proba(test_x)[:, 1]
    predictions = (probabilities >= 0.5).astype(np.int64)
    metrics = {
        "relation_type": relation_type,
        "train_examples": len(train),
        "test_examples": len(test),
        "train_positives": int(train_y.sum()),
        "test_positives": int(test_y.sum()),
        "accuracy": float(accuracy_score(test_y, predictions)),
        "roc_auc": float(roc_auc_score(test_y, probabilities)),
        "average_precision": float(average_precision_score(test_y, probabilities)),
        "status": "trained",
    }
    return model, metrics


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# V2.1 Weak-Supervised Relation Scorer",
        "",
        f"Source root: `{payload['source_root']}`",
        f"Train samples: {', '.join(payload['train_samples'])}",
        f"Test samples: {', '.join(payload['test_samples'])}",
        "",
        "| Relation | Status | Train | Test | Pos train/test | Acc | ROC-AUC | AP |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["metrics"]:
        lines.append(
            "| {relation_type} | {status} | {train_examples} | {test_examples} | {train_positives}/{test_positives} | {accuracy} | {roc_auc} | {average_precision} |".format(
                relation_type=row["relation_type"],
                status=row["status"],
                train_examples=row.get("train_examples", 0),
                test_examples=row.get("test_examples", 0),
                train_positives=row.get("train_positives", "-"),
                test_positives=row.get("test_positives", "-"),
                accuracy=f"{row.get('accuracy', 0.0):.4f}" if row["status"] == "trained" else "-",
                roc_auc=f"{row.get('roc_auc', 0.0):.4f}" if row["status"] == "trained" else "-",
                average_precision=f"{row.get('average_precision', 0.0):.4f}" if row["status"] == "trained" else "-",
            )
        )
    lines.extend(
        [
            "",
            "## Caveat",
            "",
            "Labels are weak labels from the current geometry relation graph. Treat this as a low-cost scorer prototype, not as ground-truth supervised evidence.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train weak-supervised CPU relation scorers from V2.1 relation graphs.")
    parser.add_argument("--source-root", type=Path, nargs="+", default=[Path("outputs/musicxml_fused_chunk_eval")])
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/v2_1_relation_scorer"))
    parser.add_argument("--test-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-neg-per-positive", type=int, default=3)
    parser.add_argument("--max-distance-spaces", type=float, default=8.0)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    dirs = []
    for source_root in args.source_root:
        dirs.extend(sample_dirs(source_root))
    all_examples: list[dict[str, Any]] = []
    for sample_dir in dirs:
        all_examples.extend(
            extract_examples(
                sample_dir.name,
                sample_dir / "symbols" / "symbols_v2_1_shapes.json",
                max_neg_per_positive=args.max_neg_per_positive,
                max_distance_spaces=args.max_distance_spaces,
                rng=rng,
            )
        )
    samples = sorted({item["sample"] for item in all_examples})
    train_samples, test_samples = split_samples(samples, args.test_fraction, rng)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    metrics = []
    model_paths: dict[str, str] = {}
    for relation_type in RELATION_SPECS:
        model, row = train_one_relation(relation_type, all_examples, train_samples, test_samples)
        metrics.append(row)
        if model is not None:
            model_path = args.out_dir / f"{relation_type}.joblib"
            joblib.dump(model, model_path)
            model_paths[relation_type] = str(model_path)
    payload = {
        "source_root": ", ".join(str(path) for path in args.source_root),
        "out_dir": str(args.out_dir),
        "seed": args.seed,
        "train_samples": sorted(train_samples),
        "test_samples": sorted(test_samples),
        "examples": len(all_examples),
        "model_paths": model_paths,
        "feature_names": [
            "dx_space",
            "dy_space",
            "abs_dx_space",
            "abs_dy_space",
            "center_distance_space",
            "source_width_space",
            "source_height_space",
            "target_width_space",
            "target_height_space",
            "x_overlap_min_ratio",
            "y_overlap_min_ratio",
            "x_gap_space",
            "y_gap_space",
            "same_staff",
            "source_confidence",
            "target_confidence",
            "relation_type_index",
        ],
        "metrics": metrics,
    }
    summary_json = args.out_dir / "relation_scorer_summary.json"
    summary_md = args.out_dir / "relation_scorer_summary.md"
    write_json(summary_json, payload)
    summary_md.write_text(render_markdown(payload), encoding="utf-8")
    print(
        json.dumps(
            {
                "summary_json": str(summary_json),
                "summary_md": str(summary_md),
                "examples": len(all_examples),
                "trained": sorted(model_paths),
                "metrics": metrics,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
