from __future__ import annotations

import json
from pathlib import Path
from typing import Any


CURRENT_EPOCH_PROTOCOL = {
    "RT-DETR training epochs": 250,
    "YOLO training epochs": 50,
}


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_path(raw_path: str, repo_root: Path) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else repo_root / path


def _coverage_value(
    failure: dict[str, Any], repo_root: Path
) -> tuple[int | None, int]:
    label = str(failure.get("label", ""))
    expected = int(failure.get("expected") or 0)
    source = _resolve_path(str(failure.get("path", "")), repo_root)
    if not source.is_file():
        return None, expected

    payload = _load(source)
    if label in CURRENT_EPOCH_PROTOCOL:
        expected = CURRENT_EPOCH_PROTOCOL[label]
        if not payload.get("completed"):
            return None, expected
        requested = int(payload.get("requested_epochs") or 0)
        if label == "YOLO training epochs":
            requested = min(requested, int(payload.get("results_rows") or 0))
        return requested, expected
    if label.endswith("full training predictions"):
        return int(payload.get("full_train_pages") or 0), expected
    if label.endswith(" test"):
        return int(payload.get("test_pages") or 0), expected
    return None, expected


def reconcile_integrity(integrity_path: Path, repo_root: Path | None = None) -> bool:
    integrity_path = Path(integrity_path)
    repo_root = Path.cwd() if repo_root is None else Path(repo_root)
    payload = _load(integrity_path)
    unresolved: list[dict[str, Any]] = []
    resolved: list[dict[str, Any]] = list(payload.get("resolved_coverage", []))

    for failure in payload.get("coverage_failures", []):
        actual, expected = _coverage_value(failure, repo_root)
        checked = {**failure, "actual": actual, "expected": expected}
        if actual is not None and actual >= expected:
            resolved.append(checked)
        else:
            unresolved.append(checked)

    payload["coverage_failures"] = unresolved
    payload["coverage_failure_count"] = len(unresolved)
    payload["resolved_coverage"] = resolved
    payload["complete"] = (
        int(payload.get("missing_count") or 0) == 0 and not unresolved
    )
    integrity_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return bool(payload["complete"])

