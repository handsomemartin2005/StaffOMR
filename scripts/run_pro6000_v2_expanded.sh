#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

VENV_DIR="${VENV_DIR:-.venv-v2}"
if [[ -f "$VENV_DIR/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
fi

ulimit -n "${ULIMIT_NOFILE:-65535}" 2>/dev/null || true
export PYTORCH_SHARING_STRATEGY="${PYTORCH_SHARING_STRATEGY:-file_system}"

python tools/launch_v2_expanded_training.py "$@"
