# Traditional OMR Demo Baseline

This repository contains a compact, reproducible demo for exploring how far traditional image processing can go on dense piano-score OMR before moving to neural-network models.

The long-term goal is not to replace neural OMR with hand-written rules. The current rule-based pipeline is a diagnostic baseline: it exposes which parts of score recognition are geometrically stable, which parts fail, and which errors should become neural detector/classifier/segmentation tasks.

## Repository Contents

```text
tools/traditional_omr_demo.py
docs/traditional_omr_research_proposal.md
docs/traditional-omr-process-walkthrough.pptx
docs/contact-sheet.png
examples/lg2267728_v18/input/lg-2267728-aug-beethoven--page-2.png
examples/lg2267728_v18/results/
```

Large local datasets are intentionally not committed. The demo includes only one input image and the current `v18` outputs.

## Run the Demo

```bash
pip install -r requirements.txt
python tools/traditional_omr_demo.py --out-dir outputs/traditional_omr_demo
```

The default input is:

```text
examples/lg2267728_v18/input/lg-2267728-aug-beethoven--page-2.png
```

The script writes:

- `binary_cleaned_input.png`
- `overlay_detected_staff_notes.png`
- `notehead_weight_regions.png`
- `reconstructed_from_detection.png`
- `demo_visual_result.pdf`
- `detection_result.json`

## Current Baseline Counts

For the included `v18` example:

| Element | Count |
|---|---:|
| staff lines | 50 |
| staves | 10 |
| noteheads | 457 |
| filled noteheads | 415 |
| open noteheads | 42 |
| stems | 357 |
| ledger lines | 288 |
| barlines | 46 |
| rests | 9 |
| accidentals | 21 |
| text regions | 5 |
| beam links | 141 |
| beam groups | 68 |

## Research Direction

The next stage should convert the observed failure modes into neural-network tasks:

- notehead detection and filled/open classification
- accidental/rest/clef/text symbol detection
- beam, slur, tie, and stem relation modeling
- segmentation for staff and fine symbols
- graph or sequence decoding to MusicXML / Linearized MusicXML

The structured OMR output can then support LLM-based music education workflows, including score explanation, practice feedback, rhythm checking, accidental-scope explanation, and student error localization.

## V2 Neural Symbol Recognition

V2 keeps the V1 staff geometry rules and starts replacing brittle symbol rules with neural modules:

```text
V1 staff/staff_space localization
-> DEIM-D-FINE symbol detection
-> SAM2 mask refinement from detected boxes
-> DINOv2 embeddings for assisted labeling, retrieval, clustering, and ReID-style deduplication
-> RapidOCR text recognition for score text such as To Coda, D.S. al Coda, 8va, and Pno.
```

Install the V2 dependencies:

```bash
pip install -r requirements-v2.txt
```

For a rented Linux GPU server such as RTX PRO 6000 Blackwell 96GB, use the server bootstrap instead of manually installing packages:

```bash
bash scripts/setup_pro6000_server.sh
```

After copying the DeepScores dense dataset into `ds2_dense/ds2_dense`, start the expanded 81-class detector training:

```bash
bash scripts/run_pro6000_v2_expanded.sh
```

The server runbook is documented in `docs/pro6000_server_runbook_zh.md`.

Run the complete target-page V2 pipeline with DEIM-D-FINE tuning, DINOv2, SAM2, OCR, and V1/V2 metrics:

```bash
python -m gdown 1ZPEhiU9nhW4M5jLnYOFwTSLQC1Ugf62e -O outputs/models/deim/deim_hgnetv2_n_coco.pth
python tools/run_v2_full_pipeline.py \
  --out-root outputs/v2_deim_full_tuned100 \
  --deim-dataset-root outputs/v2_deim_ds_target_tuned100 \
  --deim-run-dir outputs/v2_deim_runs/v2_symbol_target100_tuned \
  --deim-epochs 100 \
  --deim-train-limit 1 \
  --deim-val-limit 1 \
  --deim-tuning-checkpoint outputs/models/deim/deim_hgnetv2_n_coco.pth
```

Use `--deim-train-limit -1 --deim-val-limit -1` to prepare the full local DeepScores split instead of the target-page smoke set. Use `--detector weak` for the old V1-weak V2 shell.

The first V2 detector used the default `base` taxonomy, a 14-class geometry-oriented label set. To train a finer symbol-recognition head for dots, flags, rest types, time signatures, dynamics, articulations, and ornaments, use the expanded taxonomy:

```bash
python tools/prepare_deepscores_v2_dataset.py \
  --taxonomy expanded \
  --train-json ds2_dense/ds2_dense/deepscores_train.json \
  --val-json ds2_dense/ds2_dense/deepscores_test.json \
  --image-root ds2_dense/ds2_dense/images \
  --out-root outputs/v2_deim_ds_all_expanded \
  --train-limit -1 \
  --val-limit -1 \
  --skip-class text_region

python tools/create_deim_symbol_config.py \
  --taxonomy expanded \
  --dataset-root outputs/v2_deim_ds_all_expanded \
  --image-root ds2_dense/ds2_dense/images \
  --out-config .local-tools/DEIM-main/configs/deim_dfine/deim_symbol_v2_expanded.yml \
  --deim-output-dir outputs/v2_deim_runs/v2_symbol_all_expanded \
  --epochs 400
```

Because `expanded` changes `num_classes` from 14 to 81, it should be treated as a new detector head. The export and evaluation scripts accept `--taxonomy expanded` and `--class-names-json outputs/v2_deim_ds_all_expanded/annotations/instances_train.json` so predictions keep detector fine classes in `attributes.detector_class` while preserving the old base-class fields where possible.

The pipeline writes:

```text
outputs/v2_deim_full_tuned100/deim/predictions.json
outputs/v2_deim_full_tuned100/symbols/symbols_v2_with_sam2.json
outputs/v2_deim_full_tuned100/symbols/symbols_v2_1_shapes.json
outputs/v2_deim_full_tuned100/visuals/v2_1_polygon_overlay.png
outputs/v2_deim_full_tuned100/visuals/v2_1_skeleton_overlay.png
outputs/v2_deim_full_tuned100/visuals/v2_1_relation_overlay.png
outputs/v2_deim_full_tuned100/ocr/symbols_v2_ocr.json
outputs/v2_deim_full_tuned100/ocr/ocr_overlay.png
outputs/v2_deim_full_tuned100/metrics/evaluation_v1_v2.json
outputs/v2_deim_full_tuned100/metrics/evaluation_v2_1_shapes.json
```

V2.1 adds staff-aware box-to-mask shape extraction after V2 detection/SAM2. It writes polygon contours, skeleton polylines for thin symbols, weak geometry checks, and a heuristic relation graph for notehead-stem, beam-stem, ledger-notehead, and slur/tie endpoints:

```bash
python tools/extract_symbol_shapes_v2_1.py \
  --symbols-json outputs/v2_deim_full_tuned100/symbols/symbols_v2_with_sam2.json \
  --out-json outputs/v2_deim_full_tuned100/symbols/symbols_v2_1_shapes.json

python tools/visualize_v2_1_shapes.py \
  --shapes-json outputs/v2_deim_full_tuned100/symbols/symbols_v2_1_shapes.json \
  --polygon-overlay outputs/v2_deim_full_tuned100/visuals/v2_1_polygon_overlay.png \
  --skeleton-overlay outputs/v2_deim_full_tuned100/visuals/v2_1_skeleton_overlay.png \
  --relation-overlay outputs/v2_deim_full_tuned100/visuals/v2_1_relation_overlay.png

python tools/evaluate_v2_1_shape_metrics.py \
  --shapes-json outputs/v2_deim_full_tuned100/symbols/symbols_v2_1_shapes.json \
  --out-json outputs/v2_deim_full_tuned100/metrics/evaluation_v2_1_shapes.json
```

The detailed V2 design is documented in `docs/neural_symbol_recognition_v2_technical_plan.md`.
中文方法设计文档见 `docs/neural_symbol_recognition_v2_method_design_zh.md`.
V2.1 的五线谱几何约束 Box-to-Mask 设计见 `docs/neural_symbol_recognition_v2_1_method_design_zh.md`.
