from __future__ import annotations

import argparse
import copy
import json
import statistics
from pathlib import Path
from typing import Any, Callable

from assemble_v2_1_notes import build_note_objects
from evaluate_omr_musicxml_cer_ser_ler import compare_pair
from export_v2_1_semantics import build_semantics, semantic_to_musicxml, write_json, write_linearized, write_musicxml


NoteTransform = Callable[[dict[str, Any], dict[str, Any] | None], tuple[dict[str, Any], dict[str, Any]]]
ShapeTransform = Callable[[dict[str, Any]], tuple[dict[str, Any], dict[str, Any]]]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sample_dirs(source_root: Path) -> list[Path]:
    return sorted(
        path
        for path in source_root.iterdir()
        if path.is_dir()
        and (path / "notes" / "notes_v2_1.json").exists()
        and (path / "symbols" / "symbols_v2_1_shapes.json").exists()
    )


def reference_for_sample(gt_root: Path, sample: str) -> Path | None:
    for suffix in (".musicxml", ".mxl", ".xml"):
        candidate = gt_root / f"{sample}{suffix}"
        if candidate.exists():
            return candidate
    return None


def event_sort_key(event: dict[str, Any]) -> tuple[int, float, float, str]:
    staff = event.get("staff")
    x, y = event.get("center", [0.0, 0.0])
    return (9999 if staff is None else int(staff), float(x), float(y), str(event.get("id") or ""))


def rebuild_events(notes_payload: dict[str, Any]) -> None:
    notes = notes_payload.get("notes", [])
    rests = notes_payload.get("rests", [])
    marks = notes_payload.get("marks", [])
    notes_payload["events"] = sorted([*notes, *rests, *marks], key=event_sort_key)
    by_duration: dict[str, int] = {}
    for event in notes_payload["events"]:
        if event.get("type") not in {"note", "rest"}:
            continue
        key = str(event.get("duration_hint") or "unknown")
        by_duration[key] = by_duration.get(key, 0) + 1
    summary = dict(notes_payload.get("summary") or {})
    summary.update(
        {
            "notes": len(notes),
            "rests": len(rests),
            "marks": len(marks),
            "events": len(notes_payload["events"]),
            "notes_without_stem": sum(1 for note in notes if note.get("stem_id") is None),
            "notes_with_stem": sum(1 for note in notes if note.get("stem_id") is not None),
            "notes_with_beam": sum(1 for note in notes if note.get("beam_ids")),
            "notes_with_accidental": sum(1 for note in notes if note.get("accidental_id") is not None),
            "notes_with_ledger": sum(1 for note in notes if note.get("ledger_line_ids")),
            "duration_hints": dict(sorted(by_duration.items())),
        }
    )
    notes_payload["summary"] = summary


def confidence(item: dict[str, Any]) -> float:
    value = item.get("confidence")
    return float(value) if value is not None else 0.0


def make_note_postprocess(
    *,
    drop_marks: bool = False,
    min_unknown_note_conf: float | None = None,
    min_all_note_conf: float | None = None,
    min_rest_conf: float | None = None,
    drop_unknown_rests: bool = False,
    beam_cap: int | None = None,
) -> NoteTransform:
    def transform(notes_payload: dict[str, Any], shapes_payload: dict[str, Any] | None) -> tuple[dict[str, Any], dict[str, Any]]:
        del shapes_payload
        result = copy.deepcopy(notes_payload)
        before = {
            "notes": len(result.get("notes", [])),
            "rests": len(result.get("rests", [])),
            "marks": len(result.get("marks", [])),
        }
        dropped = {"notes": 0, "rests": 0, "marks": 0, "beam_count_capped": 0}
        kept_notes = []
        for note in result.get("notes", []):
            hint = str(note.get("duration_hint") or "")
            if min_all_note_conf is not None and confidence(note) < min_all_note_conf:
                dropped["notes"] += 1
                continue
            if hint == "notehead_only_unknown" and min_unknown_note_conf is not None and confidence(note) < min_unknown_note_conf:
                dropped["notes"] += 1
                continue
            if beam_cap is not None and int(note.get("beam_count") or 0) > beam_cap:
                note = dict(note)
                note["beam_count"] = beam_cap
                note["beam_ids"] = list(note.get("beam_ids") or [])[:beam_cap]
                note["duration_hint"] = {1: "eighth", 2: "16th", 3: "32nd", 4: "64th"}.get(beam_cap, "eighth")
                dropped["beam_count_capped"] += 1
            kept_notes.append(note)
        kept_rests = []
        for rest in result.get("rests", []):
            if min_rest_conf is not None and confidence(rest) < min_rest_conf:
                dropped["rests"] += 1
                continue
            if drop_unknown_rests and str(rest.get("duration_hint") or "") == "unknown":
                dropped["rests"] += 1
                continue
            kept_rests.append(rest)
        marks = result.get("marks", [])
        if drop_marks:
            dropped["marks"] = len(marks)
            marks = []
        result["notes"] = kept_notes
        result["rests"] = kept_rests
        result["marks"] = marks
        stack = dict(result.get("stack") or {})
        stack["ablation_postprocess"] = {
            "drop_marks": drop_marks,
            "min_unknown_note_conf": min_unknown_note_conf,
            "min_all_note_conf": min_all_note_conf,
            "min_rest_conf": min_rest_conf,
            "drop_unknown_rests": drop_unknown_rests,
            "beam_cap": beam_cap,
        }
        result["stack"] = stack
        rebuild_events(result)
        return result, {"before": before, "dropped": dropped, "after": result.get("summary", {})}

    return transform


def make_relation_filter(
    *,
    thresholds: dict[str, float] | None = None,
    top1_notehead_stem: bool = False,
    top1_beam_stem: bool = False,
) -> ShapeTransform:
    thresholds = thresholds or {}

    def keep_relation(relation: dict[str, Any]) -> bool:
        rel_type = str(relation.get("type") or "")
        threshold = thresholds.get(rel_type)
        if threshold is None:
            return True
        return float(relation.get("score") or 0.0) >= threshold

    def top1_key(relation: dict[str, Any]) -> str:
        if relation.get("type") == "notehead_stem_attachment":
            return str(relation.get("source"))
        if relation.get("type") == "beam_stem_group":
            targets = relation.get("targets") or []
            return str(targets[0]) if targets else str(relation.get("id"))
        return str(relation.get("id"))

    def transform(shapes_payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        result = copy.deepcopy(shapes_payload)
        before = len(result.get("relations", []))
        relations = [relation for relation in result.get("relations", []) if keep_relation(relation)]
        if top1_notehead_stem or top1_beam_stem:
            grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
            passthrough: list[dict[str, Any]] = []
            for relation in relations:
                rel_type = str(relation.get("type") or "")
                if (rel_type == "notehead_stem_attachment" and top1_notehead_stem) or (
                    rel_type == "beam_stem_group" and top1_beam_stem
                ):
                    grouped.setdefault((rel_type, top1_key(relation)), []).append(relation)
                else:
                    passthrough.append(relation)
            for group in grouped.values():
                passthrough.append(max(group, key=lambda item: float(item.get("score") or 0.0)))
            relations = sorted(passthrough, key=lambda item: str(item.get("id") or ""))
        result["relations"] = relations
        stack = dict(result.get("stack") or {})
        stack["relation_scorer_ablation"] = {
            "thresholds": thresholds,
            "top1_notehead_stem": top1_notehead_stem,
            "top1_beam_stem": top1_beam_stem,
        }
        result["stack"] = stack
        return result, {"relations_before": before, "relations_after": len(relations), "relations_dropped": before - len(relations)}

    return transform


def variants() -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {
        "baseline": {"kind": "notes", "transform": make_note_postprocess()},
        "drop_marks": {"kind": "notes", "transform": make_note_postprocess(drop_marks=True)},
        "drop_unknown_note_conf035": {"kind": "notes", "transform": make_note_postprocess(min_unknown_note_conf=0.35)},
        "drop_unknown_note_conf045": {"kind": "notes", "transform": make_note_postprocess(min_unknown_note_conf=0.45)},
        "drop_unknown_note_conf055": {"kind": "notes", "transform": make_note_postprocess(min_unknown_note_conf=0.55)},
        "drop_unknown_note_conf065": {"kind": "notes", "transform": make_note_postprocess(min_unknown_note_conf=0.65)},
        "drop_all_note_conf035": {"kind": "notes", "transform": make_note_postprocess(min_all_note_conf=0.35)},
        "drop_rest_conf030": {"kind": "notes", "transform": make_note_postprocess(min_rest_conf=0.30)},
        "drop_rest_conf040": {"kind": "notes", "transform": make_note_postprocess(min_rest_conf=0.40)},
        "drop_rest_conf050": {"kind": "notes", "transform": make_note_postprocess(min_rest_conf=0.50)},
        "drop_rest_conf060": {"kind": "notes", "transform": make_note_postprocess(min_rest_conf=0.60)},
        "drop_rest_conf070": {"kind": "notes", "transform": make_note_postprocess(min_rest_conf=0.70)},
        "drop_rest_conf080": {"kind": "notes", "transform": make_note_postprocess(min_rest_conf=0.80)},
        "drop_marks_unknown_conf045": {
            "kind": "notes",
            "transform": make_note_postprocess(drop_marks=True, min_unknown_note_conf=0.45),
        },
        "beam_cap1": {"kind": "notes", "transform": make_note_postprocess(beam_cap=1)},
        "beam_cap2": {"kind": "notes", "transform": make_note_postprocess(beam_cap=2)},
        "rel_stem_min050": {
            "kind": "shapes",
            "transform": make_relation_filter(thresholds={"notehead_stem_attachment": 0.50}),
        },
        "rel_stem_min075": {
            "kind": "shapes",
            "transform": make_relation_filter(thresholds={"notehead_stem_attachment": 0.75}),
        },
        "rel_ledger_min050": {
            "kind": "shapes",
            "transform": make_relation_filter(thresholds={"ledger_line_notehead_attachment": 0.50}),
        },
        "rel_beam_min090": {
            "kind": "shapes",
            "transform": make_relation_filter(thresholds={"beam_stem_group": 0.90}),
        },
        "rel_top1_stem": {"kind": "shapes", "transform": make_relation_filter(top1_notehead_stem=True)},
        "rel_top1_beam": {"kind": "shapes", "transform": make_relation_filter(top1_beam_stem=True)},
        "rel_top1_stem_beam": {
            "kind": "shapes",
            "transform": make_relation_filter(top1_notehead_stem=True, top1_beam_stem=True),
        },
    }
    return result


def export_prediction(
    *,
    notes_payload: dict[str, Any],
    shapes_payload: dict[str, Any] | None,
    out_dir: Path,
) -> dict[str, Path]:
    notes_path = out_dir / "notes" / "notes_v2_1.json"
    shapes_path = out_dir / "symbols" / "symbols_v2_1_shapes.json"
    semantics_path = out_dir / "semantics" / "semantics_v2_1.json"
    musicxml_path = out_dir / "semantics" / "score_v2_1.musicxml"
    linearized_path = out_dir / "semantics" / "score_v2_1_linearized.txt"
    write_json(notes_path, notes_payload)
    if shapes_payload is not None:
        write_json(shapes_path, shapes_payload)
    semantics = build_semantics({**notes_payload, "source_notes_json": str(notes_path)}, shapes_payload)
    write_json(semantics_path, semantics)
    write_musicxml(musicxml_path, semantic_to_musicxml(semantics))
    write_linearized(linearized_path, semantics)
    return {
        "notes": notes_path,
        "shapes": shapes_path,
        "semantics": semantics_path,
        "musicxml": musicxml_path,
        "linearized": linearized_path,
    }


def flatten_metrics(row: dict[str, Any]) -> dict[str, Any]:
    ser_ops = row["metrics"]["SER"]["ops"]
    cer_ops = row["metrics"]["CER"]["ops"]
    return {
        "CER": row["CER"],
        "SER": row["SER"],
        "LER": row["LER"],
        "SER_substitutions": ser_ops["substitutions"],
        "SER_deletions": ser_ops["deletions"],
        "SER_insertions": ser_ops["insertions"],
        "SER_ref_len": row["metrics"]["SER"]["ref_len"],
        "SER_hyp_len": row["metrics"]["SER"]["hyp_len"],
        "CER_insertions": cer_ops["insertions"],
        "CER_ref_len": row["metrics"]["CER"]["ref_len"],
        "CER_hyp_len": row["metrics"]["CER"]["hyp_len"],
    }


def mean(values: list[float]) -> float:
    return float(statistics.fmean(values)) if values else 0.0


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_variant: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_variant.setdefault(row["variant"], []).append(row)
    summaries = []
    for variant, items in sorted(by_variant.items()):
        summaries.append(
            {
                "variant": variant,
                "samples": len(items),
                "mean_CER": mean([float(item["CER"]) for item in items]),
                "mean_SER": mean([float(item["SER"]) for item in items]),
                "mean_LER": mean([float(item["LER"]) for item in items]),
                "SER_insertions": sum(int(item["SER_insertions"]) for item in items),
                "SER_deletions": sum(int(item["SER_deletions"]) for item in items),
                "SER_substitutions": sum(int(item["SER_substitutions"]) for item in items),
                "SER_ref_len": sum(int(item["SER_ref_len"]) for item in items),
                "SER_hyp_len": sum(int(item["SER_hyp_len"]) for item in items),
            }
        )
    baseline = next((item for item in summaries if item["variant"] == "baseline"), None)
    if baseline is not None:
        for item in summaries:
            item["delta_SER_vs_baseline"] = item["mean_SER"] - baseline["mean_SER"]
            item["delta_CER_vs_baseline"] = item["mean_CER"] - baseline["mean_CER"]
            item["delta_insertions_vs_baseline"] = item["SER_insertions"] - baseline["SER_insertions"]
            item["delta_hyp_len_vs_baseline"] = item["SER_hyp_len"] - baseline["SER_hyp_len"]
    return sorted(summaries, key=lambda item: (item["mean_SER"], item["mean_CER"], item["variant"]))


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# V2.1 Lightweight Ablation Sweeps",
        "",
        f"Source root: `{payload['source_root']}`",
        f"GT root: `{payload['gt_root']}`",
        "",
        "## Variant Summary",
        "",
        "| Variant | Samples | CER | SER | LER | Delta SER | SER ins/del/sub | Hyp/Ref tokens |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["summary"]:
        lines.append(
            "| {variant} | {samples} | {mean_CER:.6f} | {mean_SER:.6f} | {mean_LER:.6f} | {delta_SER_vs_baseline:+.6f} | {SER_insertions}/{SER_deletions}/{SER_substitutions} | {SER_hyp_len}/{SER_ref_len} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `drop_*` variants operate after note assembly and before MusicXML export.",
            "- `rel_*` variants rebuild notes from filtered relation graphs, then export with the same semantic exporter.",
            "- Lower CER/SER/LER is better. Negative delta SER means improvement over baseline under this strict evaluator.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run lightweight V2.1 MusicXML and relation ablation sweeps.")
    parser.add_argument("--source-root", type=Path, default=Path("outputs/musicxml_fused_chunk_eval"))
    parser.add_argument("--gt-root", type=Path, default=Path("data/musicxml_examples/xmlsamples"))
    parser.add_argument("--out-root", type=Path, default=Path("outputs/v2_1_light_ablation_sweeps"))
    parser.add_argument("--include-attributes", action="store_true")
    parser.add_argument("--ignore-directions", action="store_true")
    parser.add_argument("--ignore-beams", action="store_true")
    parser.add_argument("--rebuild-notes", action="store_true", help="Rebuild notes from shapes before note-level variants.")
    parser.add_argument("--variants", nargs="*", help="Optional subset of variant names.")
    args = parser.parse_args()

    all_variants = variants()
    selected_names = args.variants or list(all_variants)
    unknown = sorted(set(selected_names) - set(all_variants))
    if unknown:
        raise SystemExit(f"Unknown variants: {', '.join(unknown)}")

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for sample_dir in sample_dirs(args.source_root):
        sample = sample_dir.name
        reference = reference_for_sample(args.gt_root, sample)
        if reference is None:
            failures.append({"sample": sample, "reason": "missing_reference"})
            continue
        notes_payload = read_json(sample_dir / "notes" / "notes_v2_1.json")
        shapes_payload = read_json(sample_dir / "symbols" / "symbols_v2_1_shapes.json")
        base_notes_payload = (
            build_note_objects({**shapes_payload, "source_shapes_json": str(sample_dir / "symbols" / "symbols_v2_1_shapes.json")})
            if args.rebuild_notes
            else notes_payload
        )
        for variant_name in selected_names:
            variant = all_variants[variant_name]
            out_dir = args.out_root / variant_name / sample
            try:
                if variant["kind"] == "notes":
                    new_notes, transform_summary = variant["transform"](base_notes_payload, shapes_payload)
                    new_shapes = shapes_payload
                else:
                    new_shapes, transform_summary = variant["transform"](shapes_payload)
                    new_notes = build_note_objects({**new_shapes, "source_shapes_json": str(out_dir / "symbols" / "symbols_v2_1_shapes.json")})
                paths = export_prediction(notes_payload=new_notes, shapes_payload=new_shapes, out_dir=out_dir)
                metrics = compare_pair(
                    reference,
                    paths["musicxml"],
                    include_attributes=args.include_attributes,
                    ignore_directions=args.ignore_directions,
                    ignore_beams=args.ignore_beams,
                )
                rows.append(
                    {
                        "sample": sample,
                        "variant": variant_name,
                        "kind": variant["kind"],
                        "reference": str(reference),
                        "prediction": str(paths["musicxml"]),
                        "transform": transform_summary,
                        **flatten_metrics(metrics),
                    }
                )
            except Exception as exc:  # pragma: no cover - recorded for batch sweeps
                failures.append({"sample": sample, "variant": variant_name, "reason": repr(exc)})

    payload = {
        "source_root": str(args.source_root),
        "gt_root": str(args.gt_root),
        "out_root": str(args.out_root),
        "normalization": {
            "include_attributes": args.include_attributes,
            "ignore_directions": args.ignore_directions,
            "ignore_beams": args.ignore_beams,
            "rebuild_notes": args.rebuild_notes,
        },
        "summary": summarize(rows),
        "rows": rows,
        "failures": failures,
    }
    args.out_root.mkdir(parents=True, exist_ok=True)
    summary_json = args.out_root / "ablation_summary.json"
    summary_md = args.out_root / "ablation_summary.md"
    write_json(summary_json, payload)
    summary_md.write_text(render_markdown(payload), encoding="utf-8")
    print(
        json.dumps(
            {
                "summary_json": str(summary_json),
                "summary_md": str(summary_md),
                "variants": len(payload["summary"]),
                "rows": len(rows),
                "failures": len(failures),
                "best": payload["summary"][:5],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
