from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def run(command: list[str], cwd: Path, extra_pythonpath: Path | None = None) -> None:
    print("RUN", " ".join(command), flush=True)
    env = None
    if extra_pythonpath is not None:
        env = dict(os.environ)
        existing = env.get("PYTHONPATH")
        pieces = [str(extra_pythonpath.resolve())]
        if existing:
            pieces.append(existing)
        env["PYTHONPATH"] = os.pathsep.join(pieces)
    subprocess.run(command, cwd=str(cwd), check=True, env=env)


def checkpoint_from_run(run_dir: Path) -> Path:
    for name in ("last.pth", "best_stg2.pth", "best_stg1.pth", "checkpoint0000.pth"):
        path = run_dir / name
        if path.exists():
            return path
    checkpoints = sorted(run_dir.glob("checkpoint*.pth"))
    if checkpoints:
        return checkpoints[-1]
    raise FileNotFoundError(f"No DEIM checkpoint found in {run_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the current V2 OMR symbol-recognition pipeline end to end.")
    parser.add_argument("--workdir", type=Path, default=Path.cwd())
    parser.add_argument("--input", type=Path, default=Path("ds2_dense/ds2_dense/images/lg-2267728-aug-beethoven--page-2.png"))
    parser.add_argument("--pdf-dpi", type=int, default=220)
    parser.add_argument("--out-root", type=Path, default=Path("outputs/v2_full_run"))
    parser.add_argument("--detector", choices=("deim", "yolo", "weak"), default="deim")
    parser.add_argument("--deim-root", type=Path, default=Path(".local-tools/DEIM-main"))
    parser.add_argument("--deim-train-json", type=Path, default=Path("ds2_dense/ds2_dense/deepscores_train.json"))
    parser.add_argument("--deim-val-json", type=Path, default=Path("ds2_dense/ds2_dense/deepscores_test.json"))
    parser.add_argument("--deim-image-root", type=Path, default=Path("ds2_dense/ds2_dense/images"))
    parser.add_argument("--deim-dataset-root", type=Path, default=Path("outputs/v2_deim_deepscores_dataset"))
    parser.add_argument("--deim-config", type=Path, default=Path(".local-tools/DEIM-main/configs/deim_dfine/deim_symbol_v2.yml"))
    parser.add_argument("--deim-run-dir", type=Path, default=Path("outputs/v2_deim_runs/v2_symbol_detector"))
    parser.add_argument("--deim-checkpoint", type=Path)
    parser.add_argument("--deim-tuning-checkpoint", type=Path)
    parser.add_argument("--skip-deim-dataset-prepare", action="store_true")
    parser.add_argument("--skip-deim-config", action="store_true")
    parser.add_argument("--skip-deim-train", action="store_true")
    parser.add_argument("--deim-epochs", type=int, default=4)
    parser.add_argument("--deim-train-limit", type=int, default=8)
    parser.add_argument("--deim-val-limit", type=int, default=1)
    parser.add_argument("--deim-train-batch-size", type=int, default=1)
    parser.add_argument("--deim-val-batch-size", type=int, default=1)
    parser.add_argument("--deim-num-workers", type=int, default=0)
    parser.add_argument("--deim-device", default="cuda")
    parser.add_argument("--deim-use-amp", action="store_true")
    parser.add_argument("--deim-threshold", type=float, default=0.05)
    parser.add_argument("--deim-nms-iou", type=float, default=0.50)
    parser.add_argument("--deim-max-detections", type=int, default=1000)
    parser.add_argument("--deim-inference-mode", choices=("full", "tile", "auto"), default="auto")
    parser.add_argument("--deim-tile-size", type=int, default=1280)
    parser.add_argument("--deim-tile-overlap", type=float, default=0.25)
    parser.add_argument("--deim-resize-mode", choices=("auto", "stretch", "keep-aspect", "original"), default="auto")
    parser.add_argument("--deim-input-size", type=int, default=640)
    parser.add_argument("--deim-taxonomy", choices=("base", "expanded", "expanded_clean"), default="base")
    parser.add_argument("--yolo-python", type=Path, default=Path(".local-tools/yolo-venv/Scripts/python.exe"))
    parser.add_argument("--yolo-model", type=Path, default=Path("outputs/yolo_runs/yolo11n_symbol_expanded_clean/weights/best.pt"))
    parser.add_argument("--yolo-class-names-json", type=Path, default=Path("outputs/v2_deim_ds_all_expanded_clean/annotations/instances_train.json"))
    parser.add_argument("--yolo-device", default="0")
    parser.add_argument("--yolo-threshold", type=float, default=0.05)
    parser.add_argument("--yolo-nms-iou", type=float, default=0.50)
    parser.add_argument("--yolo-max-detections", type=int, default=1000)
    parser.add_argument("--yolo-input-size", type=int, default=640)
    parser.add_argument("--clef-roi-classifier", type=Path, help="Optional neural clef ROI classifier checkpoint used to supplement DEIM clef detections.")
    parser.add_argument("--clef-roi-threshold", type=float, default=0.40)
    parser.add_argument("--clef-roi-min-margin", type=float, default=0.03)
    parser.add_argument("--clef-roi-mode", choices=("fixed", "component", "auto"), default="fixed")
    parser.add_argument("--merge-rule-classes", default="all")
    parser.add_argument("--rule-fallback-iou", type=float, default=0.50)
    parser.add_argument("--fusion-preference", choices=("detector", "rule"), default="detector")
    parser.add_argument("--sam2-checkpoint", type=Path, default=Path("outputs/models/sam2/sam2.1_hiera_tiny.pt"))
    parser.add_argument("--sam2-cfg", default="configs/sam2.1/sam2.1_hiera_t.yaml")
    parser.add_argument("--sam2-pythonpath", type=Path, default=Path(".local-tools/sam2-main"))
    parser.add_argument(
        "--sam2-prompt-strategy",
        choices=("box_only", "box_positive", "box_pos_neg_staff"),
        default="box_only",
    )
    parser.add_argument("--skip-dinov2", action="store_true")
    parser.add_argument("--skip-sam2", action="store_true")
    parser.add_argument("--skip-v2-1", action="store_true")
    parser.add_argument(
        "--v2-1-prune-preset",
        choices=("none", "recall", "light", "balanced", "aggressive", "visual_strict"),
        default="aggressive",
        help="Postprocess V2.1 notes/symbols to reduce weak-rule over-recognition before semantic export.",
    )
    parser.add_argument("--v2-1-prune-column-cap", type=int)
    parser.add_argument("--v2-1-prune-column-xtol", type=float)
    parser.add_argument("--v2-1-prune-no-stem-min-conf", type=float)
    parser.add_argument("--v2-1-prune-open-no-stem-min-conf", type=float)
    parser.add_argument("--v2-1-prune-isolated-unknown-min-conf", type=float)
    parser.add_argument("--v2-1-prune-neighbor-xtol", type=float)
    parser.add_argument(
        "--v2-1-symbol-crop-classifier",
        type=Path,
        help="Optional family-constrained crop classifier applied after V2.1 shape extraction.",
    )
    parser.add_argument("--v2-1-symbol-crop-conditional-threshold", type=float, default=0.55)
    parser.add_argument("--v2-1-symbol-crop-family-mass-threshold", type=float, default=0.50)
    parser.add_argument("--v2-1-symbol-crop-device", default="cpu")
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()

    root = args.workdir.resolve()
    out = args.out_root
    weak_coco = out / "weak_labels" / "weak_symbols_coco.json"
    weak_json = out / "weak_labels" / "weak_symbols_v2.json"
    crop_dir = out / "symbol_crops"
    symbols_json = out / "symbols" / "symbols_v2.json"
    symbols_sam2_json = out / "symbols" / "symbols_v2_with_sam2.json"
    symbols_v2_1_json = out / "symbols" / "symbols_v2_1_shapes.json"
    notes_v2_1_json = out / "notes" / "notes_v2_1.json"
    symbols_v2_1_semantics_json = symbols_v2_1_json
    notes_v2_1_semantics_json = notes_v2_1_json
    semantics_v2_1_json = out / "semantics" / "semantics_v2_1.json"
    musicxml_v2_1_path = out / "semantics" / "score_v2_1.musicxml"
    linearized_v2_1_path = out / "semantics" / "score_v2_1_linearized.txt"
    ocr_json = out / "ocr" / "symbols_v2_ocr.json"
    deim_predictions = out / "deim" / "predictions.json"
    yolo_predictions = out / "yolo" / "predictions.json"
    deim_train_ann = args.deim_dataset_root / "annotations" / "instances_train.json"

    py = sys.executable
    input_args = ["--input", str(args.input)] if args.input else []
    run(
        [
            py,
            "tools/export_v1_weak_labels.py",
            *input_args,
            "--pdf-dpi",
            str(args.pdf_dpi),
            "--out-coco",
            str(weak_coco),
            "--out-json",
            str(weak_json),
            "--overlay-dir",
            str(out / "weak_labels" / "overlays"),
        ],
        root,
    )
    run([py, "tools/build_symbol_crops.py", "--coco", str(weak_coco), "--out-dir", str(crop_dir)], root)
    if not args.skip_dinov2:
        run(
            [
                py,
                "tools/embed_symbol_crops_dinov2.py",
                "--metadata",
                str(crop_dir / "metadata.jsonl"),
                "--out-dir",
                str(out / "dinov2"),
                "--batch-size",
                "64",
                "--cluster-k",
                "14",
            ],
            root,
        )
    detector_summary: dict[str, str | None] = {"type": args.detector, "checkpoint": None, "predictions": None}
    detector_threshold = args.deim_threshold if args.detector == "deim" else args.yolo_threshold
    if args.detector == "deim":
        image_name = args.input.name
        if not args.skip_deim_dataset_prepare:
            run(
                [
                    py,
                    "tools/prepare_deepscores_v2_dataset.py",
                    "--train-json",
                    str(args.deim_train_json),
                    "--val-json",
                    str(args.deim_val_json),
                    "--image-root",
                    str(args.deim_image_root),
                    "--out-root",
                    str(args.deim_dataset_root),
                    "--train-limit",
                    str(args.deim_train_limit),
                    "--val-limit",
                    str(args.deim_val_limit),
                    "--include-train-image",
                    image_name,
                    "--include-val-image",
                    image_name,
                    "--taxonomy",
                    args.deim_taxonomy,
                ],
                root,
            )
        elif not deim_train_ann.exists():
            raise FileNotFoundError(f"Cannot load DEIM class names; missing {deim_train_ann}")
        if not args.skip_deim_config:
            run(
                [
                    py,
                    "tools/create_deim_symbol_config.py",
                    "--dataset-root",
                    str(args.deim_dataset_root),
                    "--image-root",
                    str(args.deim_image_root),
                    "--out-config",
                    str(args.deim_config),
                    "--deim-output-dir",
                    str(args.deim_run_dir),
                    "--epochs",
                    str(args.deim_epochs),
                    "--train-batch-size",
                    str(args.deim_train_batch_size),
                    "--val-batch-size",
                    str(args.deim_val_batch_size),
                    "--num-workers",
                    str(args.deim_num_workers),
                    "--taxonomy",
                    args.deim_taxonomy,
                ],
                root,
            )
        elif not args.deim_config.exists():
            raise FileNotFoundError(f"Cannot run DEIM inference; missing config {args.deim_config}")
        if args.deim_checkpoint is None and not args.skip_deim_train:
            train_cmd = [
                py,
                "train.py",
                "-c",
                str(args.deim_config.resolve()),
                "-d",
                args.deim_device,
                "--output-dir",
                str(args.deim_run_dir.resolve()),
            ]
            if args.deim_tuning_checkpoint:
                train_cmd.extend(["-t", str(args.deim_tuning_checkpoint.resolve())])
            if args.deim_use_amp:
                train_cmd.append("--use-amp")
            run(train_cmd, (root / args.deim_root).resolve())
        checkpoint = args.deim_checkpoint if args.deim_checkpoint else checkpoint_from_run(args.deim_run_dir)
        detector_summary["checkpoint"] = str(checkpoint)
        run(
            [
                py,
                "tools/export_deim_predictions.py",
                "--deim-root",
                str(args.deim_root),
                "--config",
                str(args.deim_config),
                "--checkpoint",
                str(checkpoint),
                "--input",
                str(args.input),
                "--out-json",
                str(deim_predictions),
                "--overlay",
                str(out / "deim" / "overlay_deim_predictions.png"),
                "--device",
                args.deim_device,
                "--threshold",
                str(args.deim_threshold),
                "--nms-iou",
                str(args.deim_nms_iou),
                "--max-detections",
                str(args.deim_max_detections),
                "--inference-mode",
                args.deim_inference_mode,
                "--tile-size",
                str(args.deim_tile_size),
                "--tile-overlap",
                str(args.deim_tile_overlap),
                "--resize-mode",
                args.deim_resize_mode,
                "--input-size",
                str(args.deim_input_size),
                "--taxonomy",
                args.deim_taxonomy,
                "--class-names-json",
                str(deim_train_ann),
            ],
            root,
        )
        detector_predictions_for_symbols = deim_predictions
        if args.clef_roi_classifier is not None:
            clef_predictions = out / "deim" / "predictions_with_clef_roi.json"
            run(
                [
                    py,
                    "tools/add_clef_roi_classifier_predictions.py",
                    "--detector-json",
                    str(deim_predictions),
                    "--input",
                    str(args.input),
                    "--classifier",
                    str(args.clef_roi_classifier),
                    "--out-json",
                    str(clef_predictions),
                    "--overlay",
                    str(out / "deim" / "overlay_deim_with_clef_roi.png"),
                    "--device",
                    args.deim_device,
                    "--threshold",
                    str(args.clef_roi_threshold),
                    "--min-margin",
                    str(args.clef_roi_min_margin),
                    "--roi-mode",
                    args.clef_roi_mode,
                ],
                root,
            )
            detector_summary["base_predictions"] = str(deim_predictions)
            detector_summary["clef_roi_classifier"] = str(args.clef_roi_classifier)
            detector_predictions_for_symbols = clef_predictions
        detector_summary["predictions"] = str(detector_predictions_for_symbols)
        run(
            [
                py,
                "tools/infer_neural_symbols_v2.py",
                *input_args,
                "--pdf-dpi",
                str(args.pdf_dpi),
                "--detector-predictions",
                str(detector_predictions_for_symbols),
                "--bbox-format",
                "xyxy",
                "--min-confidence",
                str(args.deim_threshold),
                "--merge-rule-classes",
                args.merge_rule_classes,
                "--rule-fallback-iou",
                str(args.rule_fallback_iou),
                "--fusion-preference",
                args.fusion_preference,
                "--out-json",
                str(symbols_json),
                "--overlay",
                str(out / "symbols" / "overlay_v2.png"),
            ],
            root,
        )
    elif args.detector == "yolo":
        if not args.yolo_python.exists():
            raise FileNotFoundError(f"Cannot run YOLO inference; missing Python environment {args.yolo_python}")
        if not args.yolo_model.exists():
            raise FileNotFoundError(f"Cannot run YOLO inference; missing model {args.yolo_model}")
        if not args.yolo_class_names_json.exists():
            raise FileNotFoundError(f"Cannot map YOLO classes; missing {args.yolo_class_names_json}")
        detector_summary["checkpoint"] = str(args.yolo_model)
        run(
            [
                str(args.yolo_python),
                "tools/export_yolo_predictions.py",
                "--model",
                str(args.yolo_model),
                "--input",
                str(args.input),
                "--class-names-json",
                str(args.yolo_class_names_json),
                "--out-json",
                str(yolo_predictions),
                "--overlay",
                str(out / "yolo" / "overlay_yolo_predictions.png"),
                "--device",
                args.yolo_device,
                "--threshold",
                str(args.yolo_threshold),
                "--nms-iou",
                str(args.yolo_nms_iou),
                "--max-detections",
                str(args.yolo_max_detections),
                "--input-size",
                str(args.yolo_input_size),
                "--taxonomy",
                args.deim_taxonomy,
            ],
            root,
        )
        detector_predictions_for_symbols = yolo_predictions
        if args.clef_roi_classifier is not None:
            clef_predictions = out / "yolo" / "predictions_with_clef_roi.json"
            run(
                [
                    py,
                    "tools/add_clef_roi_classifier_predictions.py",
                    "--detector-json",
                    str(yolo_predictions),
                    "--input",
                    str(args.input),
                    "--classifier",
                    str(args.clef_roi_classifier),
                    "--out-json",
                    str(clef_predictions),
                    "--overlay",
                    str(out / "yolo" / "overlay_yolo_with_clef_roi.png"),
                    "--device",
                    args.deim_device,
                    "--threshold",
                    str(args.clef_roi_threshold),
                    "--min-margin",
                    str(args.clef_roi_min_margin),
                    "--roi-mode",
                    args.clef_roi_mode,
                ],
                root,
            )
            detector_summary["base_predictions"] = str(yolo_predictions)
            detector_summary["clef_roi_classifier"] = str(args.clef_roi_classifier)
            detector_predictions_for_symbols = clef_predictions
        detector_summary["predictions"] = str(detector_predictions_for_symbols)
        run(
            [
                py,
                "tools/infer_neural_symbols_v2.py",
                *input_args,
                "--pdf-dpi",
                str(args.pdf_dpi),
                "--detector-predictions",
                str(detector_predictions_for_symbols),
                "--bbox-format",
                "xyxy",
                "--min-confidence",
                str(args.yolo_threshold),
                "--merge-rule-classes",
                args.merge_rule_classes,
                "--rule-fallback-iou",
                str(args.rule_fallback_iou),
                "--fusion-preference",
                args.fusion_preference,
                "--out-json",
                str(symbols_json),
                "--overlay",
                str(out / "symbols" / "overlay_v2.png"),
            ],
            root,
        )
    else:
        run(
            [
                py,
                "tools/infer_neural_symbols_v2.py",
                *input_args,
                "--pdf-dpi",
                str(args.pdf_dpi),
                "--out-json",
                str(symbols_json),
                "--overlay",
                str(out / "symbols" / "overlay_v2.png"),
            ],
            root,
        )
    if not args.skip_sam2:
        if not args.sam2_checkpoint.exists():
            raise FileNotFoundError(args.sam2_checkpoint)
        masks_json = out / "sam2" / "masks_all.json"
        run(
            [
                py,
                "tools/refine_masks_sam2.py",
                "--symbols-json",
                str(symbols_json),
                "--checkpoint",
                str(args.sam2_checkpoint),
                "--model-cfg",
                args.sam2_cfg,
                "--prompt-strategy",
                args.sam2_prompt_strategy,
                "--out-json",
                str(masks_json),
                "--mask-dir",
                str(out / "sam2" / "masks"),
            ],
            root,
            extra_pythonpath=root / args.sam2_pythonpath,
        )
        run(
            [
                py,
                "tools/infer_neural_symbols_v2.py",
                *input_args,
                "--pdf-dpi",
                str(args.pdf_dpi),
                *(["--detector-predictions", str(detector_predictions_for_symbols), "--bbox-format", "xyxy", "--min-confidence", str(detector_threshold)] if args.detector in {"deim", "yolo"} else []),
                *(["--merge-rule-classes", args.merge_rule_classes, "--rule-fallback-iou", str(args.rule_fallback_iou), "--fusion-preference", args.fusion_preference] if args.detector in {"deim", "yolo"} else []),
                "--sam2-mask-json",
                str(masks_json),
                "--out-json",
                str(symbols_sam2_json),
                "--overlay",
                str(out / "symbols" / "overlay_v2_with_sam2.png"),
            ],
            root,
        )
    else:
        symbols_sam2_json = symbols_json
    symbols_for_downstream = symbols_sam2_json
    v2_1_outputs = None
    if not args.skip_v2_1:
        symbols_v2_1_base_json = symbols_v2_1_json
        run(
            [
                py,
                "tools/extract_symbol_shapes_v2_1.py",
                "--symbols-json",
                str(symbols_sam2_json),
                "--out-json",
                str(symbols_v2_1_json),
            ],
            root,
        )
        crop_classifier_outputs: dict[str, str | float] | None = None
        if args.v2_1_symbol_crop_classifier is not None:
            symbols_v2_1_fused_json = out / "symbols" / "symbols_v2_1_shapes_crop_fused.json"
            crop_classifier_summary = out / "metrics" / "symbol_crop_classifier_fusion.json"
            run(
                [
                    py,
                    "tools/apply_v2_1_symbol_crop_classifier.py",
                    "--shapes-json",
                    str(symbols_v2_1_base_json),
                    "--classifier",
                    str(args.v2_1_symbol_crop_classifier),
                    "--out-json",
                    str(symbols_v2_1_fused_json),
                    "--out-summary",
                    str(crop_classifier_summary),
                    "--conditional-threshold",
                    str(args.v2_1_symbol_crop_conditional_threshold),
                    "--family-mass-threshold",
                    str(args.v2_1_symbol_crop_family_mass_threshold),
                    "--device",
                    args.v2_1_symbol_crop_device,
                ],
                root,
            )
            symbols_v2_1_json = symbols_v2_1_fused_json
            crop_classifier_outputs = {
                "classifier": str(args.v2_1_symbol_crop_classifier),
                "base_symbols": str(symbols_v2_1_base_json),
                "fused_symbols": str(symbols_v2_1_fused_json),
                "summary": str(crop_classifier_summary),
                "conditional_threshold": args.v2_1_symbol_crop_conditional_threshold,
                "family_mass_threshold": args.v2_1_symbol_crop_family_mass_threshold,
            }
        polygon_overlay = out / "visuals" / "v2_1_polygon_overlay.png"
        skeleton_overlay = out / "visuals" / "v2_1_skeleton_overlay.png"
        relation_overlay = out / "visuals" / "v2_1_relation_overlay.png"
        run(
            [
                py,
                "tools/visualize_v2_1_shapes.py",
                "--shapes-json",
                str(symbols_v2_1_json),
                "--polygon-overlay",
                str(polygon_overlay),
                "--skeleton-overlay",
                str(skeleton_overlay),
                "--relation-overlay",
                str(relation_overlay),
            ],
            root,
        )
        v2_1_metrics = out / "metrics" / "evaluation_v2_1_shapes.json"
        run(
            [
                py,
                "tools/evaluate_v2_1_shape_metrics.py",
                "--shapes-json",
                str(symbols_v2_1_json),
                "--out-json",
                str(v2_1_metrics),
            ],
            root,
        )
        run(
            [
                py,
                "tools/assemble_v2_1_notes.py",
                "--shapes-json",
                str(symbols_v2_1_json),
                "--out-json",
                str(notes_v2_1_json),
            ],
            root,
        )
        prune_outputs: dict[str, str] | None = None
        if args.v2_1_prune_preset != "none":
            symbols_v2_1_pruned_json = out / "symbols" / f"symbols_v2_1_shapes_pruned_{args.v2_1_prune_preset}.json"
            notes_v2_1_pruned_json = out / "notes" / f"notes_v2_1_pruned_{args.v2_1_prune_preset}.json"
            prune_report_json = out / "metrics" / f"overrecognition_prune_{args.v2_1_prune_preset}_report.json"
            run(
                [
                    py,
                    "tools/prune_v2_1_overrecognition.py",
                    "--preset",
                    args.v2_1_prune_preset,
                    *(["--column-cap", str(args.v2_1_prune_column_cap)] if args.v2_1_prune_column_cap is not None else []),
                    *(["--column-xtol", str(args.v2_1_prune_column_xtol)] if args.v2_1_prune_column_xtol is not None else []),
                    *(["--no-stem-min-conf", str(args.v2_1_prune_no_stem_min_conf)] if args.v2_1_prune_no_stem_min_conf is not None else []),
                    *(["--open-no-stem-min-conf", str(args.v2_1_prune_open_no_stem_min_conf)] if args.v2_1_prune_open_no_stem_min_conf is not None else []),
                    *(["--isolated-unknown-min-conf", str(args.v2_1_prune_isolated_unknown_min_conf)] if args.v2_1_prune_isolated_unknown_min_conf is not None else []),
                    *(["--neighbor-xtol", str(args.v2_1_prune_neighbor_xtol)] if args.v2_1_prune_neighbor_xtol is not None else []),
                    "--notes-json",
                    str(notes_v2_1_json),
                    "--shapes-json",
                    str(symbols_v2_1_json),
                    "--out-notes-json",
                    str(notes_v2_1_pruned_json),
                    "--out-shapes-json",
                    str(symbols_v2_1_pruned_json),
                    "--out-report",
                    str(prune_report_json),
                ],
                root,
            )
            notes_v2_1_semantics_json = notes_v2_1_pruned_json
            symbols_v2_1_semantics_json = symbols_v2_1_pruned_json
            prune_outputs = {
                "preset": args.v2_1_prune_preset,
                "symbols": str(symbols_v2_1_pruned_json),
                "notes": str(notes_v2_1_pruned_json),
                "report": str(prune_report_json),
            }
        run(
            [
                py,
                "tools/export_v2_1_semantics.py",
                "--notes-json",
                str(notes_v2_1_semantics_json),
                "--shapes-json",
                str(symbols_v2_1_semantics_json),
                "--out-json",
                str(semantics_v2_1_json),
                "--out-musicxml",
                str(musicxml_v2_1_path),
                "--out-linearized",
                str(linearized_v2_1_path),
            ],
            root,
        )
        symbols_for_downstream = symbols_v2_1_semantics_json
        v2_1_outputs = {
            "symbols": str(symbols_v2_1_json),
            "symbol_crop_classifier": crop_classifier_outputs,
            "notes": str(notes_v2_1_json),
            "pruned": prune_outputs,
            "symbols_for_semantics": str(symbols_v2_1_semantics_json),
            "notes_for_semantics": str(notes_v2_1_semantics_json),
            "semantics": str(semantics_v2_1_json),
            "musicxml": str(musicxml_v2_1_path),
            "linearized": str(linearized_v2_1_path),
            "polygon_overlay": str(polygon_overlay),
            "skeleton_overlay": str(skeleton_overlay),
            "relation_overlay": str(relation_overlay),
            "metrics": str(v2_1_metrics),
        }
    run(
        [
            py,
            "tools/ocr_score_text.py",
            "--symbols-json",
            str(symbols_for_downstream),
            "--out-json",
            str(ocr_json),
            "--overlay",
            str(out / "ocr" / "ocr_overlay.png"),
        ],
        root,
    )
    if not args.skip_eval:
        run(
            [
                py,
                "tools/evaluate_v1_v2_metrics.py",
                "--v2-symbols-json",
                str(symbols_for_downstream),
                "--v2-name",
                f"v2_{args.detector}_sam2",
                "--out-json",
                str(out / "metrics" / "evaluation_v1_v2.json"),
                "--taxonomy",
                args.deim_taxonomy,
                *(["--prefer-detector-class"] if args.deim_taxonomy in {"expanded", "expanded_clean"} else []),
            ],
            root,
        )
    reconstruction_path = out / "visuals" / "v2_reconstructed_from_symbols.png"
    reconstruction_comparison_path = out / "visuals" / "input_vs_v2_reconstructed.png"
    run(
        [
            py,
            "tools/reconstruct_v2_symbols.py",
            "--symbols-json",
            str(symbols_for_downstream),
            "--ocr-json",
            str(ocr_json),
            "--out",
            str(reconstruction_path),
            "--comparison",
            str(reconstruction_comparison_path),
        ],
        root,
    )

    summary = {
        "weak_coco": str(weak_coco),
        "crops": str(crop_dir),
        "dinov2": None if args.skip_dinov2 else str(out / "dinov2" / "summary.json"),
        "detector": detector_summary,
        "deim_taxonomy": args.deim_taxonomy,
        "symbols": str(symbols_json),
        "symbols_with_sam2": None if args.skip_sam2 else str(symbols_sam2_json),
        "v2_1": v2_1_outputs,
        "ocr": str(ocr_json),
        "ocr_overlay": str(out / "ocr" / "ocr_overlay.png"),
        "metrics": None if args.skip_eval else str(out / "metrics" / "evaluation_v1_v2.json"),
        "reconstruction": str(reconstruction_path),
        "reconstruction_comparison": str(reconstruction_comparison_path),
    }
    summary_path = out / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
