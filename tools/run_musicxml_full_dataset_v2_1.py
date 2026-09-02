from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from traditional_omr_demo import load_input_image


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run(command: list[str], cwd: Path, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PYTHONUTF8", "1")
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("RUN " + " ".join(command) + "\n")
        handle.flush()
        subprocess.run(command, cwd=str(cwd), stdout=handle, stderr=subprocess.STDOUT, check=True, env=env)


def source_for_sample(gt_root: Path, sample: str) -> Path:
    for suffix in (".pdf", ".png", ".jpg", ".jpeg"):
        path = gt_root / f"{sample}{suffix}"
        if path.exists():
            return path
    raise FileNotFoundError(f"No PDF/PNG source found for {sample} in {gt_root}")


def render_input(source: Path, out_png: Path, dpi: int) -> Path:
    if out_png.exists():
        return out_png
    image = load_input_image(source, dpi=dpi)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_png)
    return out_png


def sample_names(gt_root: Path, requested: list[str] | None) -> list[str]:
    if requested:
        return requested
    return sorted(path.stem for path in gt_root.glob("*.musicxml"))


def pipeline_command(args: argparse.Namespace, sample: str, input_png: Path, out_dir: Path) -> list[str]:
    command = [
        sys.executable,
        "tools/run_v2_full_pipeline.py",
        "--input",
        str(input_png),
        "--pdf-dpi",
        str(args.pdf_dpi),
        "--out-root",
        str(out_dir),
        "--detector",
        args.detector,
        "--merge-rule-classes",
        args.merge_rule_classes,
        "--fusion-preference",
        args.fusion_preference,
        "--skip-dinov2",
        "--skip-eval",
    ]
    if args.detector == "deim":
        command.extend(
            [
                "--deim-root", str(args.deim_root),
                "--deim-dataset-root", str(args.deim_dataset_root),
                "--deim-config", str(args.deim_config),
                "--deim-checkpoint", str(args.deim_checkpoint),
                "--skip-deim-dataset-prepare", "--skip-deim-config", "--skip-deim-train",
                "--deim-device", args.device,
                "--deim-threshold", str(args.deim_threshold),
                "--deim-nms-iou", str(args.deim_nms_iou),
                "--deim-max-detections", str(args.deim_max_detections),
                "--deim-taxonomy", args.deim_taxonomy,
            ]
        )
    elif args.detector == "yolo":
        command.extend(
            [
                "--yolo-python", str(args.yolo_python),
                "--yolo-model", str(args.yolo_model),
                "--yolo-class-names-json", str(args.yolo_class_names_json),
                "--yolo-device", args.yolo_device,
                "--yolo-threshold", str(args.yolo_threshold),
                "--yolo-nms-iou", str(args.yolo_nms_iou),
                "--yolo-max-detections", str(args.yolo_max_detections),
                "--yolo-input-size", str(args.yolo_input_size),
                "--deim-taxonomy", args.deim_taxonomy,
            ]
        )
    if args.use_sam2:
        command.extend(
            [
                "--sam2-checkpoint",
                str(args.sam2_checkpoint),
                "--sam2-cfg",
                args.sam2_cfg,
                "--sam2-pythonpath",
                str(args.sam2_pythonpath),
                "--sam2-prompt-strategy",
                args.sam2_prompt_strategy,
            ]
        )
    else:
        command.append("--skip-sam2")
    if args.clef_roi_classifier:
        command.extend(
            [
                "--clef-roi-classifier",
                str(args.clef_roi_classifier),
                "--clef-roi-threshold",
                str(args.clef_roi_threshold),
                "--clef-roi-min-margin",
                str(args.clef_roi_min_margin),
                "--clef-roi-mode",
                args.clef_roi_mode,
            ]
        )
    command.extend(["--v2-1-prune-preset", args.v2_1_prune_preset])
    if args.v2_1_symbol_crop_classifier:
        command.extend(
            [
                "--v2-1-symbol-crop-classifier",
                str(args.v2_1_symbol_crop_classifier),
                "--v2-1-symbol-crop-conditional-threshold",
                str(args.v2_1_symbol_crop_conditional_threshold),
                "--v2-1-symbol-crop-family-mass-threshold",
                str(args.v2_1_symbol_crop_family_mass_threshold),
                "--v2-1-symbol-crop-device",
                args.v2_1_symbol_crop_device,
            ]
        )
    return command


def output_complete(out_dir: Path) -> bool:
    required = [
        out_dir / "symbols" / "symbols_v2_1_shapes.json",
        out_dir / "notes" / "notes_v2_1.json",
        out_dir / "semantics" / "score_v2_1.musicxml",
        out_dir / "summary.json",
    ]
    return all(path.exists() for path in required)


def build_eval_manifest(gt_root: Path, out_root: Path, samples: list[str]) -> dict[str, Any]:
    pages = []
    for sample in samples:
        reference = gt_root / f"{sample}.musicxml"
        prediction = out_root / sample / "semantics" / "score_v2_1.musicxml"
        if reference.exists() and prediction.exists():
            pages.append(
                {
                    "page": sample,
                    "reference_musicxml": str(reference.resolve()),
                    "predictions": {"v2_1_best_stg1": str(prediction.resolve())},
                }
            )
    return {"pages": pages}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run V2.1 best checkpoint over the full MusicXML example dataset.")
    parser.add_argument("--gt-root", type=Path, default=Path("data/musicxml_examples/xmlsamples"))
    parser.add_argument("--input-root", type=Path, default=Path("outputs/musicxml_full_dataset_inputs_s220"))
    parser.add_argument("--out-root", type=Path, default=Path("outputs/musicxml_full_dataset_v2_1_best_stg1"))
    parser.add_argument("--samples", nargs="*")
    parser.add_argument("--pdf-dpi", type=int, default=220)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--detector", choices=("deim", "yolo", "weak"), default="deim")
    parser.add_argument("--deim-root", type=Path, default=Path(".local-tools/DEIM-main"))
    parser.add_argument("--deim-dataset-root", type=Path, default=Path("outputs/v2_deim_ds_all_expanded_clean"))
    parser.add_argument("--deim-config", type=Path, default=Path(".local-tools/DEIM-main/configs/deim_dfine/deim_symbol_v2_expanded_clean_5070ti.yml"))
    parser.add_argument("--deim-checkpoint", type=Path, default=Path("outputs/v2_deim_runs/v2_symbol_all_expanded_clean_5070ti/best_stg1.pth"))
    parser.add_argument("--deim-threshold", type=float, default=0.05)
    parser.add_argument("--deim-nms-iou", type=float, default=0.50)
    parser.add_argument("--deim-max-detections", type=int, default=1000)
    parser.add_argument("--deim-taxonomy", default="expanded_clean")
    parser.add_argument("--yolo-python", type=Path, default=Path(".local-tools/yolo-venv/Scripts/python.exe"))
    parser.add_argument("--yolo-model", type=Path, default=Path("outputs/yolo_runs/yolo11n_symbol_expanded_clean/weights/best.pt"))
    parser.add_argument("--yolo-class-names-json", type=Path, default=Path("outputs/v2_deim_ds_all_expanded_clean/annotations/instances_train.json"))
    parser.add_argument("--yolo-device", default="0")
    parser.add_argument("--yolo-threshold", type=float, default=0.05)
    parser.add_argument("--yolo-nms-iou", type=float, default=0.50)
    parser.add_argument("--yolo-max-detections", type=int, default=1000)
    parser.add_argument("--yolo-input-size", type=int, default=640)
    parser.add_argument("--merge-rule-classes", default="all")
    parser.add_argument("--fusion-preference", choices=("detector", "rule"), default="detector")
    parser.add_argument("--use-sam2", action="store_true")
    parser.add_argument("--sam2-checkpoint", type=Path, default=Path("outputs/models/sam2/sam2.1_hiera_tiny.pt"))
    parser.add_argument("--sam2-cfg", default="configs/sam2.1/sam2.1_hiera_t.yaml")
    parser.add_argument("--sam2-pythonpath", type=Path, default=Path(".local-tools/sam2-main"))
    parser.add_argument(
        "--sam2-prompt-strategy",
        choices=("box_only", "box_positive", "box_pos_neg_staff"),
        default="box_only",
    )
    parser.add_argument("--clef-roi-classifier", type=Path)
    parser.add_argument("--clef-roi-threshold", type=float, default=0.40)
    parser.add_argument("--clef-roi-min-margin", type=float, default=0.03)
    parser.add_argument("--clef-roi-mode", choices=("fixed", "component", "auto"), default="fixed")
    parser.add_argument(
        "--v2-1-prune-preset",
        choices=("none", "recall", "light", "balanced", "aggressive", "visual_strict"),
        default="balanced",
    )
    parser.add_argument("--v2-1-symbol-crop-classifier", type=Path)
    parser.add_argument("--v2-1-symbol-crop-conditional-threshold", type=float, default=0.55)
    parser.add_argument("--v2-1-symbol-crop-family-mass-threshold", type=float, default=0.50)
    parser.add_argument("--v2-1-symbol-crop-device", default="cpu")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    repo = Path.cwd()
    samples = sample_names(args.gt_root, args.samples)
    rows = []
    failures = []
    for index, sample in enumerate(samples, start=1):
        out_dir = args.out_root / sample
        row: dict[str, Any] = {"sample": sample, "index": index, "total": len(samples), "out_dir": str(out_dir)}
        try:
            source = source_for_sample(args.gt_root, sample)
            input_png = render_input(source, args.input_root / f"{sample}.png", args.pdf_dpi)
            row.update({"source": str(source), "input_png": str(input_png)})
            if output_complete(out_dir) and not args.force:
                row["status"] = "skipped_existing"
            else:
                run(pipeline_command(args, sample, input_png, out_dir), repo, out_dir / "logs" / "pipeline.log")
                row["status"] = "completed"
            if output_complete(out_dir):
                notes = read_json(out_dir / "notes" / "notes_v2_1.json")
                metrics = read_json(out_dir / "metrics" / "evaluation_v2_1_shapes.json") if (out_dir / "metrics" / "evaluation_v2_1_shapes.json").exists() else {}
                row.update(
                    {
                        "notes": notes.get("summary", {}).get("notes"),
                        "rests": notes.get("summary", {}).get("rests"),
                        "notes_with_stem": notes.get("summary", {}).get("notes_with_stem"),
                        "notes_with_beam": notes.get("summary", {}).get("notes_with_beam"),
                        "geometry_pass": metrics.get("weak_scores", {}).get("geometry_sanity_pass_rate"),
                        "relation_consistency": metrics.get("weak_scores", {}).get("relation_consistency_score"),
                    }
                )
            else:
                row["status"] = "incomplete"
        except Exception as exc:  # pragma: no cover - batch reporting
            row["status"] = "failed"
            row["error"] = repr(exc)
            failures.append(row)
        rows.append(row)
        write_json(args.out_root / "full_dataset_progress.json", {"rows": rows, "failures": failures})
        print(json.dumps(row, ensure_ascii=False), flush=True)

    manifest = build_eval_manifest(args.gt_root, args.out_root, samples)
    manifest_path = args.out_root / "strict_eval_manifest.json"
    write_json(manifest_path, manifest)
    summary = {
        "samples": samples,
        "rows": rows,
        "failures": failures,
        "manifest": str(manifest_path),
        "completed": sum(1 for row in rows if row.get("status") in {"completed", "skipped_existing"} and output_complete(Path(row["out_dir"]))),
    }
    write_json(args.out_root / "full_dataset_summary.json", summary)
    print(json.dumps({"summary": str(args.out_root / "full_dataset_summary.json"), **summary}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
