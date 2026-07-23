from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from tools import build_paper_evidence_figures as figures


def test_load_qualitative_case_uses_frozen_best_system() -> None:
    case = figures.load_qualitative_case()

    assert case["sample_id"] == "test_0006"
    assert case["event_f1"] == pytest.approx(24.4725738397, abs=1e-9)
    assert case["crop"] == (965, 390, 1170, 515)
    assert Image.open(case["input_path"]).size == (3524, 732)
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
