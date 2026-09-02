from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import joblib
import numpy as np
from PIL import Image
from torchvision import transforms as T

from assemble_v2_1_notes import build_note_objects
from evaluate_olimpic_official_metrics import gold_cost_from_lmx, linearize_musicxml, load_olimpic_modules
from export_v2_1_official_piano_semantics import semantic_to_official_piano, write_musicxml
from export_v2_1_semantics import build_semantics
from run_v2_1_ablation_sweeps import make_note_postprocess
from train_symbol_crop_classifier import SmallCropCNN
from train_v2_1_relation_scorer import RELATION_SPECS, compatible_targets, features


FAMILIES = {
    "notehead": ("filled_notehead", "open_notehead"),
    "accidental": ("sharp", "flat", "natural"),
    "rest_duration": ("rest_whole", "rest_half", "rest_quarter", "rest_8th", "rest_16th"),
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def sample_dirs(source_root: Path, shapes_relpath: str) -> list[Path]:
    return sorted(path for path in source_root.iterdir() if path.is_dir() and (path / shapes_relpath).exists())


def clamp_crop(
    bbox: list[float],
    image_size: tuple[int, int],
    pad_ratio: float,
    pad_px: int,
) -> tuple[int, int, int, int] | None:
    width, height = image_size
    x0, y0, x1, y1 = [float(value) for value in bbox]
    pad = max(float(pad_px), max(x1 - x0, y1 - y0) * float(pad_ratio))
    crop = (
        max(0, int(x0 - pad)),
        max(0, int(y0 - pad)),
        min(width, int(x1 + pad + 0.999)),
        min(height, int(y1 + pad + 0.999)),
    )
    return crop if crop[2] > crop[0] and crop[3] > crop[1] else None


def load_classifier(
    path: Path,
    device: torch.device,
) -> tuple[SmallCropCNN, list[str], T.Compose, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device)
    class_names = [str(name) for name in checkpoint["class_names"]]
    model = SmallCropCNN(len(class_names)).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    image_size = int(checkpoint.get("image_size") or 96)
    transform = T.Compose(
        [
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    return model, class_names, transform, checkpoint


@torch.no_grad()
def classify_symbols(
    shapes: dict[str, Any],
    model: SmallCropCNN,
    class_names: list[str],
    transform: T.Compose,
    checkpoint: dict[str, Any],
    device: torch.device,
    batch_size: int,
) -> dict[str, dict[str, float]]:
    image_path = Path(str(shapes["input"]))
    image = Image.open(image_path).convert("RGB")
    eligible = set(name for family in FAMILIES.values() for name in family)
    items: list[tuple[str, torch.Tensor]] = []
    for symbol in shapes.get("symbols", []):
        symbol_class = str(symbol.get("class"))
        attrs = symbol.get("attributes") or {}
        fine_class = str(attrs.get("detector_class") or attrs.get("fine_class") or symbol_class)
        if symbol_class not in eligible and fine_class not in eligible:
            continue
        crop = clamp_crop(
            list(symbol["bbox"]),
            image.size,
            float(checkpoint.get("pad_ratio") or 0.18),
            int(checkpoint.get("pad_px") or 4),
        )
        if crop is None:
            continue
        items.append((str(symbol["id"]), transform(image.crop(crop))))
    result: dict[str, dict[str, float]] = {}
    for start in range(0, len(items), max(1, batch_size)):
        batch = items[start : start + max(1, batch_size)]
        probabilities = torch.softmax(model(torch.stack([tensor for _, tensor in batch]).to(device)), dim=1).cpu()
        for (symbol_id, _), probs in zip(batch, probabilities):
            result[symbol_id] = {name: float(value) for name, value in zip(class_names, probs.tolist())}
    return result


def family_for_class(class_name: str) -> tuple[str, tuple[str, ...]] | None:
    for family_name, family in FAMILIES.items():
        if class_name in family:
            return family_name, family
    return None


def family_for_symbol(symbol: dict[str, Any]) -> tuple[str, tuple[str, ...], str, bool] | None:
    symbol_class = str(symbol.get("class") or "")
    direct = family_for_class(symbol_class)
    if direct is not None:
        family_name, family = direct
        return family_name, family, symbol_class, False
    attrs = symbol.get("attributes") or {}
    fine_class = str(attrs.get("detector_class") or attrs.get("fine_class") or "")
    fine = family_for_class(fine_class)
    if fine is None:
        return None
    family_name, family = fine
    return family_name, family, fine_class, True


def apply_family_fusion(
    shapes: dict[str, Any],
    predictions: dict[str, dict[str, float]],
    *,
    conditional_threshold: float,
    family_mass_threshold: float,
    enabled_families: set[str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    result = copy.deepcopy(shapes)
    changes: Counter[str] = Counter()
    accepted = 0
    rejected_mass = 0
    rejected_conditional = 0
    for symbol in result.get("symbols", []):
        family_info = family_for_symbol(symbol)
        probs = predictions.get(str(symbol.get("id")))
        if family_info is None or probs is None:
            continue
        family_name, family, old_class, fine_attribute = family_info
        if enabled_families is not None and family_name not in enabled_families:
            continue
        family_scores = {name: float(probs.get(name, 0.0)) for name in family}
        family_mass = sum(family_scores.values())
        if family_mass < family_mass_threshold:
            rejected_mass += 1
            continue
        new_class, new_score = max(family_scores.items(), key=lambda item: item[1])
        conditional_score = new_score / max(1e-12, family_mass)
        if conditional_score < conditional_threshold:
            rejected_conditional += 1
            continue
        accepted += 1
        attrs = dict(symbol.get("attributes") or {})
        attrs["crop_classifier_fusion"] = {
            "family": family_name,
            "old_class": old_class,
            "new_class": new_class,
            "family_mass": family_mass,
            "conditional_score": conditional_score,
            "probabilities": probs,
        }
        symbol["attributes"] = attrs
        if new_class != old_class:
            if fine_attribute:
                attrs["fine_class"] = new_class
                attrs["detector_class"] = new_class
            else:
                symbol["class"] = new_class
            changes[f"{old_class}->{new_class}"] += 1
    stack = dict(result.get("stack") or {})
    stack["symbol_crop_classifier_fusion"] = {
        "conditional_threshold": conditional_threshold,
        "family_mass_threshold": family_mass_threshold,
        "families": FAMILIES,
        "enabled_families": sorted(enabled_families) if enabled_families is not None else sorted(FAMILIES),
    }
    result["stack"] = stack
    return result, {
        "eligible_predictions": len(predictions),
        "accepted": accepted,
        "rejected_family_mass": rejected_mass,
        "rejected_conditional": rejected_conditional,
        "changes": dict(changes),
        "changes_total": sum(changes.values()),
    }


def load_relation_models(model_dir: Path) -> dict[str, Any]:
    models = {}
    for relation_type in RELATION_SPECS:
        path = model_dir / f"{relation_type}.joblib"
        if path.exists():
            models[relation_type] = joblib.load(path)
    return models


def relation_target_scores(
    relation: dict[str, Any],
    symbols_by_id: dict[str, dict[str, Any]],
    model: Any,
) -> list[tuple[str, float]]:
    source = symbols_by_id.get(str(relation.get("source")))
    if source is None:
        return []
    relation_type = str(relation.get("type") or "")
    targets = [symbols_by_id.get(str(target_id)) for target_id in relation.get("targets", []) or []]
    targets = [target for target in targets if target is not None]
    if not targets:
        return []
    values = model.predict_proba(
        np.asarray([features(source, target, relation_type) for target in targets], dtype=np.float32)
    )[:, 1]
    return [(str(target["id"]), float(value)) for target, value in zip(targets, values)]


def apply_relation_rescore(
    shapes: dict[str, Any],
    models: dict[str, Any],
    *,
    threshold: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    result = copy.deepcopy(shapes)
    symbols_by_id = {str(symbol["id"]): symbol for symbol in result.get("symbols", [])}
    old_relations = list(result.get("relations", []))
    new_relations = []
    targets_before = 0
    targets_after = 0
    for relation in old_relations:
        relation_type = str(relation.get("type") or "")
        model = models.get(relation_type)
        targets_before += len(relation.get("targets", []) or [])
        if model is None:
            new_relations.append(relation)
            targets_after += len(relation.get("targets", []) or [])
            continue
        scored = relation_target_scores(relation, symbols_by_id, model)
        kept = [(target_id, score) for target_id, score in scored if score >= threshold]
        minimum = 2 if relation_type == "slur_tie_notehead_endpoints" else 1
        if len(kept) < minimum:
            continue
        item = copy.deepcopy(relation)
        item["targets"] = [target_id for target_id, _ in kept]
        item["score"] = float(sum(score for _, score in kept) / len(kept))
        evidence = dict(item.get("evidence") or {})
        evidence["learned_relation_scorer"] = {
            "mode": "rescore_existing",
            "threshold": threshold,
            "target_scores": {target_id: score for target_id, score in scored},
        }
        item["evidence"] = evidence
        new_relations.append(item)
        targets_after += len(kept)
    result["relations"] = new_relations
    return result, {
        "accepted": targets_after,
        "changes": {
            "relations_removed": len(old_relations) - len(new_relations),
            "targets_removed": targets_before - targets_after,
        },
        "changes_total": (len(old_relations) - len(new_relations)) + (targets_before - targets_after),
    }


def apply_relation_regeneration(
    shapes: dict[str, Any],
    models: dict[str, Any],
    *,
    threshold: float,
    max_distance_spaces: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    result = copy.deepcopy(shapes)
    symbols = list(result.get("symbols", []))
    preserved = [relation for relation in result.get("relations", []) if str(relation.get("type") or "") not in models]
    old_modeled = [relation for relation in result.get("relations", []) if str(relation.get("type") or "") in models]
    regenerated = []
    target_count = 0
    relation_index = 0
    for relation_type, model in models.items():
        source_classes, target_classes = RELATION_SPECS[relation_type]
        for source in symbols:
            if str(source.get("class") or "") not in source_classes:
                continue
            candidates = compatible_targets(
                source,
                symbols,
                target_classes,
                max_distance_spaces=max_distance_spaces,
            )
            if not candidates:
                continue
            probabilities = model.predict_proba(
                np.asarray([features(source, target, relation_type) for target in candidates], dtype=np.float32)
            )[:, 1]
            ranked = sorted(
                [(target, float(score)) for target, score in zip(candidates, probabilities) if float(score) >= threshold],
                key=lambda item: item[1],
                reverse=True,
            )
            if relation_type == "notehead_stem_attachment":
                ranked = ranked[:1]
            elif relation_type == "ledger_line_notehead_attachment":
                ranked = ranked[:2]
            elif relation_type == "slur_tie_notehead_endpoints":
                ranked = ranked[:2] if len(ranked) >= 2 else []
            if not ranked:
                continue
            if relation_type in {"beam_stem_group", "slur_tie_notehead_endpoints"}:
                target_groups = [ranked]
            else:
                target_groups = [[item] for item in ranked]
            for group in target_groups:
                regenerated.append(
                    {
                        "id": f"learned_rel_{relation_index:06d}",
                        "type": relation_type,
                        "source": str(source["id"]),
                        "targets": [str(target["id"]) for target, _ in group],
                        "score": float(sum(score for _, score in group) / len(group)),
                        "evidence": {
                            "learned_relation_scorer": {
                                "mode": "regenerate_candidates",
                                "threshold": threshold,
                                "max_distance_spaces": max_distance_spaces,
                                "target_scores": {str(target["id"]): score for target, score in group},
                            }
                        },
                    }
                )
                relation_index += 1
                target_count += len(group)
    result["relations"] = [*preserved, *regenerated]
    return result, {
        "accepted": target_count,
        "changes": {
            "modeled_relations_removed": len(old_modeled),
            "learned_relations_added": len(regenerated),
        },
        "changes_total": len(old_modeled) + len(regenerated),
    }


def rebuild_with_best_lightweight_policy(shapes: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    notes = build_note_objects(shapes)
    transform = make_note_postprocess(min_rest_conf=0.80)
    return transform(notes, shapes)


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# V2.1 Symbol Crop Classifier Fusion Sweep",
        "",
        f"Source root: `{payload['source_root']}`",
        f"GT root: `{payload['gt_root']}`",
        f"Classifier: `{payload['classifier']}`",
        "",
        "Family-constrained fusion only allows swaps inside notehead, accidental, or rest-duration families. It never changes symbols across families.",
        "",
        "| Variant | Samples | Failures | SER | SER no tuplets | Changes | Accepted |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["summary"]:
        lines.append(
            "| {variant} | {samples} | {failures} | {official_SER:.6f} | {official_SERnotuplets:.6f} | {changes_total} | {accepted} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "Lower SER is better. The baseline rebuild uses the same rest-confidence 0.80 and official exporter note-confidence 0.69 policy as the current lightweight benchmark path.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep family-constrained crop-classifier fusion under official LMX SER.")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--gt-root", type=Path, required=True)
    parser.add_argument("--classifier", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    parser.add_argument("--shapes-relpath", default="symbols/symbols_v2_1_shapes.json")
    parser.add_argument("--olimpic-root", type=Path, default=Path(".local-tools/olimpic-icdar24"))
    parser.add_argument("--conditional-thresholds", type=float, nargs="+", default=[0.55, 0.70, 0.85, 0.95])
    parser.add_argument("--family-mass-thresholds", type=float, nargs="+", default=[0.30, 0.50])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--skip-classifier-variants", action="store_true")
    parser.add_argument("--relation-model-dir", type=Path)
    parser.add_argument("--relation-thresholds", type=float, nargs="+", default=[0.30, 0.50, 0.70, 0.85])
    parser.add_argument("--relation-modes", nargs="+", choices=("rescore", "regenerate"), default=["rescore", "regenerate"])
    parser.add_argument("--max-distance-spaces", type=float, default=8.0)
    parser.add_argument("--infer-same-pitch-ties", action="store_true")
    parser.add_argument("--enabled-families", nargs="+", choices=tuple(FAMILIES), default=list(FAMILIES))
    args = parser.parse_args()

    device = torch.device(args.device)
    model, class_names, transform, checkpoint = load_classifier(args.classifier, device)
    Linearizer, MxlFile, _TEDn_lmx_xml, ser_metric = load_olimpic_modules(args.olimpic_root)
    relation_models = load_relation_models(args.relation_model_dir) if args.relation_model_dir else {}
    variants: list[dict[str, Any]] = [{"name": "baseline", "kind": "baseline"}]
    if not args.skip_classifier_variants:
        variants.extend(
            {
                "name": f"family_c{conditional:.2f}_m{mass:.2f}",
                "kind": "classifier",
                "conditional": conditional,
                "mass": mass,
            }
            for conditional in args.conditional_thresholds
            for mass in args.family_mass_thresholds
        )
    if relation_models:
        for mode in args.relation_modes:
            variants.extend(
                {
                    "name": f"relation_{mode}_t{threshold:.2f}",
                    "kind": f"relation_{mode}",
                    "threshold": threshold,
                }
                for threshold in args.relation_thresholds
            )
    aggregate: dict[str, dict[str, Any]] = {
        name: {"gold": [], "pred": [], "rows": [], "failures": [], "changes": Counter(), "accepted": 0}
        for name in [variant["name"] for variant in variants]
    }

    for sample_dir in sample_dirs(args.source_root, args.shapes_relpath):
        sample = sample_dir.name
        shapes = read_json(sample_dir / args.shapes_relpath)
        predictions = classify_symbols(shapes, model, class_names, transform, checkpoint, device, args.batch_size)
        gold_path = args.gt_root / f"{sample}.lmx.txt"
        if not gold_path.exists():
            for variant in variants:
                aggregate[variant["name"]]["failures"].append({"sample": sample, "reason": "missing_gold"})
            continue
        gold_lmx = gold_path.read_text(encoding="utf-8").strip()
        for variant in variants:
            name = variant["name"]
            if variant["kind"] == "baseline":
                new_shapes = copy.deepcopy(shapes)
                fusion_stats = {"accepted": 0, "changes": {}, "changes_total": 0}
            elif variant["kind"] == "classifier":
                new_shapes, fusion_stats = apply_family_fusion(
                    shapes,
                    predictions,
                    conditional_threshold=float(variant["conditional"]),
                    family_mass_threshold=float(variant["mass"]),
                    enabled_families=set(args.enabled_families),
                )
            elif variant["kind"] == "relation_rescore":
                new_shapes, fusion_stats = apply_relation_rescore(
                    shapes,
                    relation_models,
                    threshold=float(variant["threshold"]),
                )
            else:
                new_shapes, fusion_stats = apply_relation_regeneration(
                    shapes,
                    relation_models,
                    threshold=float(variant["threshold"]),
                    max_distance_spaces=args.max_distance_spaces,
                )
            notes, prune_stats = rebuild_with_best_lightweight_policy(new_shapes)
            semantics = build_semantics(notes, new_shapes)
            tree, export_stats = semantic_to_official_piano(
                semantics,
                include_key=False,
                include_time=False,
                emit_directions=False,
                x_tolerance=8.0,
                backup_policy="fixed",
                fixed_measure_units=64,
                min_note_confidence=0.69,
                infer_same_pitch_ties=args.infer_same_pitch_ties,
            )
            sample_out = args.out_root / name / sample
            xml_path = sample_out / "semantics" / "score_v2_1_crop_fusion.musicxml"
            shapes_path = sample_out / "symbols" / "symbols_v2_1_shapes.json"
            notes_path = sample_out / "notes" / "notes_v2_1.json"
            write_musicxml(xml_path, tree)
            write_json(shapes_path, new_shapes)
            write_json(notes_path, notes)
            try:
                pred_lmx = linearize_musicxml(xml_path, Linearizer, MxlFile)
            except Exception as exc:
                pred_lmx = ""
                aggregate[name]["failures"].append({"sample": sample, "reason": repr(exc)})
            pred_lmx_path = sample_out / "semantics" / "score_v2_1_crop_fusion.lmx.txt"
            pred_lmx_path.write_text(pred_lmx + "\n", encoding="utf-8")
            row_metrics = ser_metric([gold_lmx], [pred_lmx])
            aggregate[name]["gold"].append(gold_lmx)
            aggregate[name]["pred"].append(pred_lmx)
            aggregate[name]["accepted"] += int(fusion_stats["accepted"])
            aggregate[name]["changes"].update(fusion_stats["changes"])
            aggregate[name]["rows"].append(
                {
                    "sample": sample,
                    **row_metrics,
                    "gold_tokens": gold_cost_from_lmx(gold_lmx),
                    "pred_tokens": len(pred_lmx.split()),
                    "fusion": fusion_stats,
                    "prune": prune_stats,
                    "export": export_stats,
                    "prediction": str(xml_path),
                    "reference_lmx": str(gold_path),
                    "prediction_lmx": str(pred_lmx_path),
                }
            )
        print(json.dumps({"sample": sample, "variants": len(variants)}, ensure_ascii=False), flush=True)

    summary = []
    variant_rows = []
    for variant in variants:
        name = variant["name"]
        item = aggregate[name]
        metrics = ser_metric(item["gold"], item["pred"]) if item["gold"] else {"SER": None, "SERnotuplets": None}
        row = {
            "variant": name,
            "kind": variant["kind"],
            "conditional_threshold": variant.get("conditional"),
            "family_mass_threshold": variant.get("mass"),
            "relation_threshold": variant.get("threshold"),
            "samples": len(item["rows"]),
            "failures": len(item["failures"]),
            "official_SER": metrics["SER"],
            "official_SERnotuplets": metrics["SERnotuplets"],
            "changes": dict(item["changes"]),
            "changes_total": sum(item["changes"].values()),
            "accepted": item["accepted"],
        }
        summary.append(row)
        variant_rows.append({**row, "rows": item["rows"], "failures_detail": item["failures"]})
    summary.sort(key=lambda item: (float("inf") if item["official_SER"] is None else item["official_SER"], item["variant"]))
    payload = {
        "source_root": str(args.source_root),
        "gt_root": str(args.gt_root),
        "classifier": str(args.classifier),
        "out_root": str(args.out_root),
        "policy": {
            "families": FAMILIES,
            "rest_min_confidence": 0.80,
            "official_export_note_min_confidence": 0.69,
            "x_tolerance": 8.0,
            "infer_same_pitch_ties": bool(args.infer_same_pitch_ties),
            "enabled_families": args.enabled_families,
        },
        "summary": summary,
        "variants": variant_rows,
    }
    write_json(args.out_json, payload)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps({"out_json": str(args.out_json), "out_md": str(args.out_md), "best": summary[0]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
