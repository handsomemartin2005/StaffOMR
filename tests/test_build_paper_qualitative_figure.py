from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pytest
from PIL import Image

from tools import build_paper_evidence_figures as figures


def test_load_qualitative_case_uses_frozen_best_system() -> None:
    case = figures.load_qualitative_case()

    assert case["provenance"] == {
        "summary_path": figures.ROOT / "outputs/debussy_abcd_ablation/summary.json",
        "shapes_path": figures.ROOT / "outputs/debussy_abcd_ablation/A1B1C1D1/test_0006/symbols/shapes.json",
        "mask_manifest_path": figures.ROOT / "outputs/ijcv_repro/debussy_staff_transfer24/test_0006/sam2/masks_all.json",
    }
    assert case["sample_id"] == "test_0006"
    assert case["event_f1"] == pytest.approx(24.4725738397, abs=1e-9)
    assert case["crop"] == (965, 390, 1170, 515)
    assert case["overview_path"] == (
        figures.ROOT
        / "outputs/ijcv_repro/debussy_staff_transfer24/test_0006/visuals/v2_1_skeleton_overlay.png"
    )
    assert case["overview_path"].exists()
    with Image.open(case["input_path"]) as image:
        assert image.size == (3524, 732)
    with Image.open(case["overview_path"]) as image:
        assert image.size == (3524, 732)
    assert set(case["notehead_ids"]) == {"deim_000003", "deim_000013"}
    assert set(case["stem_ids"]) == {"synthetic_stem_00012", "synthetic_stem_00013"}
    assert case["beam_id"] == "deim_000069"
    assert case["mask_records"]["deim_000003"]["mask_score"] == pytest.approx(0.8905225992)
    assert case["mask_records"]["deim_000069"]["mask_score"] == pytest.approx(0.8875870109)
    assert set(case["head_stem_edges"]) == {
        ("deim_000003", "synthetic_stem_00012"),
        ("deim_000013", "synthetic_stem_00013"),
    }
    assert set(case["beam_stem_edges"]) == {
        ("deim_000069", "synthetic_stem_00012"),
        ("deim_000069", "synthetic_stem_00013"),
    }
    assert case["direct_notehead_beam_edges"] == []


def test_build_qualitative_pipeline_writes_overview_and_local_stages(tmp_path: Path) -> None:
    paths = [Path(path) for path in figures.build_qualitative_pipeline(tmp_path)]

    assert plt.rcParams["font.family"] == ["serif"]
    assert plt.rcParams["font.serif"][:2] == ["Times New Roman", "Times"]
    assert {path.suffix for path in paths} == {".png", ".pdf"}
    assert all(path.exists() and path.stat().st_size > 0 for path in paths)
    with Image.open(next(path for path in paths if path.suffix == ".png")) as preview:
        width, height = preview.size
    # The paper uses this at full two-column width: the overview remains on
    # top, with the five local stages aligned in one row below it.
    assert 2.0 <= width / height <= 3.5
    assert height >= 500
