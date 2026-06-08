#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

VENV_DIR="${VENV_DIR:-.venv-v2}"
if [[ -f "$VENV_DIR/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
fi

RUN_DIR="${RUN_DIR:-outputs/v2_deim_runs/v2_symbol_all_expanded_pro6000}"
CONFIG="${CONFIG:-.local-tools/DEIM-main/configs/deim_dfine/deim_symbol_v2_expanded_pro6000.yml}"
DATASET_ROOT="${DATASET_ROOT:-outputs/v2_deim_ds_all_expanded}"
INPUT="${INPUT:-ds2_dense/ds2_dense/images/lg-2267728-aug-beethoven--page-2.png}"
OUT_ROOT="${OUT_ROOT:-outputs/v2_1_runs/pro6000_sample}"
DEVICE="${DEVICE:-cuda}"
CHECKPOINT="${CHECKPOINT:-}"
SAM2_CHECKPOINT="${SAM2_CHECKPOINT:-outputs/models/sam2/sam2.1_hiera_tiny.pt}"
SAM2_CFG="${SAM2_CFG:-configs/sam2.1/sam2.1_hiera_t.yaml}"

if [[ -z "$CHECKPOINT" ]]; then
  for candidate in "$RUN_DIR/best_stg2.pth" "$RUN_DIR/best_stg1.pth" "$RUN_DIR/last.pth"; do
    if [[ -f "$candidate" ]]; then
      CHECKPOINT="$candidate"
      break
    fi
  done
fi

if [[ -z "$CHECKPOINT" || ! -f "$CHECKPOINT" ]]; then
  echo "No trained checkpoint found. Expected one of:" >&2
  echo "  $RUN_DIR/best_stg2.pth" >&2
  echo "  $RUN_DIR/best_stg1.pth" >&2
  echo "  $RUN_DIR/last.pth" >&2
  exit 1
fi

if [[ ! -f "$CONFIG" ]]; then
  echo "Missing DEIM config: $CONFIG" >&2
  echo "Run scripts/run_pro6000_v2_expanded.sh first, or pass CONFIG=..." >&2
  exit 1
fi

if [[ ! -f "$DATASET_ROOT/annotations/instances_train.json" ]]; then
  echo "Missing expanded dataset metadata: $DATASET_ROOT/annotations/instances_train.json" >&2
  echo "Run scripts/run_pro6000_v2_expanded.sh first, or pass DATASET_ROOT=..." >&2
  exit 1
fi

sam2_args=()
if [[ -f "$SAM2_CHECKPOINT" ]]; then
  sam2_args=(--sam2-checkpoint "$SAM2_CHECKPOINT" --sam2-cfg "$SAM2_CFG")
else
  echo "SAM2 checkpoint not found at $SAM2_CHECKPOINT; running V2.1 with bbox fallback masks." >&2
  sam2_args=(--skip-sam2)
fi

python tools/run_v2_full_pipeline.py \
  --input "$INPUT" \
  --out-root "$OUT_ROOT" \
  --detector deim \
  --deim-root .local-tools/DEIM-main \
  --deim-dataset-root "$DATASET_ROOT" \
  --deim-config "$CONFIG" \
  --deim-run-dir "$RUN_DIR" \
  --deim-checkpoint "$CHECKPOINT" \
  --skip-deim-dataset-prepare \
  --skip-deim-config \
  --skip-deim-train \
  --deim-taxonomy expanded \
  --deim-device "$DEVICE" \
  --deim-threshold "${DEIM_THRESHOLD:-0.05}" \
  --deim-nms-iou "${DEIM_NMS_IOU:-0.50}" \
  --deim-max-detections "${DEIM_MAX_DETECTIONS:-2000}" \
  --skip-dinov2 \
  "${sam2_args[@]}"

echo "V2.1 outputs:"
echo "  $OUT_ROOT/summary.json"
echo "  $OUT_ROOT/visuals/input_vs_v2_reconstructed.png"
echo "  $OUT_ROOT/visuals/v2_1_relation_overlay.png"
