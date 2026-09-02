from __future__ import annotations

import json
import shutil
from pathlib import Path

from run_abcd_module_ablation import discover_pages


def count(root: Path, pattern: str) -> int:
    return len(list(root.glob(pattern))) if root.exists() else 0


def main() -> None:
    checks: list[dict] = []

    def exact(name: str, actual: int, expected: int) -> None:
        checks.append({"name": name, "actual": actual, "expected": expected, "pass": actual == expected})

    def exists(path: str) -> None:
        target = Path(path)
        checks.append({"name": f"exists:{path}", "actual": target.exists(), "expected": True, "pass": target.exists()})

    exact("GrandStaff train1000 images", count(Path("data/ijcv_samples/grandstaff-lmx-train1000"), "*.png"), 1000)
    exact("GrandStaff train1000 LMX", count(Path("data/ijcv_samples/grandstaff-lmx-train1000"), "*.lmx.txt"), 1000)
    exact("GrandStaff test200 images", count(Path("data/ijcv_samples/grandstaff-lmx-test200-flat"), "*.jpg"), 200)
    exact("GrandStaff test200 LMX", count(Path("data/ijcv_samples/grandstaff-lmx-test200-flat"), "*.lmx.txt"), 200)
    exact("OLiMPiC test200 images", count(Path("data/ijcv_samples/olimpic-test200-flat"), "*.png"), 200)
    exact("OLiMPiC test200 LMX", count(Path("data/ijcv_samples/olimpic-test200-flat"), "*.lmx.txt"), 200)
    exact("Polish test24 images", count(Path("data/ijcv_samples/polish-scores-test24"), "*.png"), 24)
    exact("Polish test24 eKern", count(Path("data/ijcv_samples/polish-scores-test24"), "*.ekern.txt"), 24)
    exact("ABCD GrandStaff discoverable pages", len(discover_pages("grandstaff")), 200)
    exact("ABCD OLiMPiC discoverable pages", len(discover_pages("olimpic")), 200)
    exact("ABCD Polish discoverable pages", len(discover_pages("polish")), 24)
    manifest_path = Path("outputs/ijcv_repro/ours_olimpic_test200_artifact_view/manifest.json")
    manifest_pages = json.loads(manifest_path.read_text(encoding="utf-8"))["pages"] if manifest_path.exists() else 0
    exact("OLiMPiC pre-pruning artifact view", int(manifest_pages), 200)
    for path in (
        "outputs/v2_deim_runs/v2_symbol_all_expanded_clean_5070ti/checkpoint0399.pth",
        "outputs/models/sam2/sam2.1_hiera_tiny.pt",
        "outputs/v2_1_clef_roi_classifier_scan_pseudo/symbol_crop_classifier.pt",
        "outputs/v2_1_clef_crop_classifier_scan_aug_pad12/symbol_crop_classifier.pt",
        ".local-tools/DEIM-main/configs/deim_rtdetrv2/rtdetrv2_symbol_expanded_clean.yml",
        "outputs/v2_deim_ds_all_expanded_clean/annotations/instances_train.json",
    ):
        exists(path)
    free_gb = shutil.disk_usage(Path.cwd()).free / (1024**3)
    checks.append({"name": "workspace free disk GB", "actual": round(free_gb, 2), "expected": ">=100", "pass": free_gb >= 100})
    failures = [row for row in checks if not row["pass"]]
    result = {"pass": not failures, "checks": checks, "failures": failures}
    output = Path("outputs/ijcv_repro/remaining_experiments_preflight.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
