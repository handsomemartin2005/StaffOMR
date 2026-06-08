#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  if command -v python3.11 >/dev/null 2>&1; then
    PYTHON_BIN="python3.11"
  else
    PYTHON_BIN="python3"
  fi
fi

CREATE_VENV="${CREATE_VENV:-1}"
VENV_DIR="${VENV_DIR:-.venv-v2}"
DEIM_ROOT="${DEIM_ROOT:-.local-tools/DEIM-main}"
DEIM_REPO_URL="${DEIM_REPO_URL:-https://github.com/ShihuaHuang95/DEIM.git}"
DEIM_REF="${DEIM_REF:-}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
INSTALL_TORCH="${INSTALL_TORCH:-auto}"

echo "[1/7] Python"
"$PYTHON_BIN" - <<'PY'
import sys
version = sys.version_info
print(sys.version)
if version < (3, 10):
    raise SystemExit("Python 3.10+ is required")
PY

if [[ "$CREATE_VENV" == "1" ]]; then
  echo "[2/7] Virtual environment: $VENV_DIR"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
else
  echo "[2/7] Using current Python environment"
fi

python -m pip install --upgrade pip setuptools wheel

echo "[3/7] CUDA/PyTorch"
python - <<'PY' >/tmp/staffomr_torch_check.txt 2>&1 || true
import torch
print(torch.__version__, torch.version.cuda, torch.cuda.is_available())
raise SystemExit(0 if torch.cuda.is_available() else 1)
PY
if [[ "$INSTALL_TORCH" == "1" ]] || [[ "$INSTALL_TORCH" == "auto" && ! -s /tmp/staffomr_torch_check.txt ]]; then
  python -m pip install --index-url "$TORCH_INDEX_URL" torch torchvision
elif [[ "$INSTALL_TORCH" == "auto" ]]; then
  if ! grep -q "True" /tmp/staffomr_torch_check.txt; then
    python -m pip install --index-url "$TORCH_INDEX_URL" torch torchvision
  else
    cat /tmp/staffomr_torch_check.txt
  fi
fi

python - <<'PY'
import torch
print({"torch": torch.__version__, "cuda_runtime": torch.version.cuda, "cuda_available": torch.cuda.is_available()})
if not torch.cuda.is_available():
    raise SystemExit("Torch was installed, but CUDA is not available. Check the server driver/container image.")
print(torch.cuda.get_device_name(0))
PY

echo "[4/7] Project dependencies"
python -m pip install -r requirements-v2-server.txt

echo "[5/7] DEIM checkout"
mkdir -p "$(dirname "$DEIM_ROOT")"
if [[ ! -d "$DEIM_ROOT/.git" && ! -f "$DEIM_ROOT/train.py" ]]; then
  git clone "$DEIM_REPO_URL" "$DEIM_ROOT"
fi
if [[ -n "$DEIM_REF" && -d "$DEIM_ROOT/.git" ]]; then
  git -C "$DEIM_ROOT" fetch --all --tags
  git -C "$DEIM_ROOT" checkout "$DEIM_REF"
fi
python tools/patch_deim_for_staff.py --deim-root "$DEIM_ROOT"

tmp_req="$(mktemp)"
python - "$DEIM_ROOT/requirements.txt" "$tmp_req" <<'PY'
from pathlib import Path
import sys
src, dst = map(Path, sys.argv[1:])
skip = {"torch", "torchvision"}
lines = []
for raw in src.read_text(encoding="utf-8").splitlines():
    name = raw.strip().split("==")[0].split(">=")[0].split("<")[0].strip().lower()
    if not raw.strip() or raw.lstrip().startswith("#") or name in skip:
        continue
    lines.append(raw)
dst.write_text("\n".join(lines) + "\n", encoding="utf-8")
PY
python -m pip install -r "$tmp_req"
rm -f "$tmp_req"

echo "[6/7] DEIM COCO tuning checkpoint"
mkdir -p outputs/models/deim
if [[ ! -f outputs/models/deim/deim_hgnetv2_n_coco.pth ]]; then
  python -m gdown 1ZPEhiU9nhW4M5jLnYOFwTSLQC1Ugf62e -O outputs/models/deim/deim_hgnetv2_n_coco.pth
fi

echo "[7/7] Final verification"
DEIM_ROOT="$DEIM_ROOT" python - <<'PY'
import os
import torch
from pathlib import Path
deim_root = Path(os.environ["DEIM_ROOT"])
checks = {
    "deim_train": (deim_root / "train.py").exists(),
    "checkpoint": Path("outputs/models/deim/deim_hgnetv2_n_coco.pth").exists(),
    "cuda": torch.cuda.is_available(),
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
}
print(checks)
if not all([checks["deim_train"], checks["checkpoint"], checks["cuda"]]):
    raise SystemExit("Server setup verification failed")
PY

echo "Setup complete."
echo "Next: copy ds2_dense/ into this repo, then run:"
echo "  bash scripts/run_pro6000_v2_expanded.sh"
