from __future__ import annotations

from pathlib import Path

from PIL import Image

from tools import build_staff_aware_qualitative_pipeline as figure


def test_load_case_uses_support_filtered_prompts_and_safe_residual_graph() -> None:
    case = figure.load_case()

    assert case["sample_id"] == "test_0006"
    assert case["event_f1"] > case["box_event_f1"]
    assert case["predicted_events"] == case["box_predicted_events"] == 138
    beam_prompt = case["mask_records"]["deim_000069"]["prompt"]
    assert beam_prompt["adaptive_box_rule"] == "beam_staff_scaled_expand"
    assert len(beam_prompt["positive_points"]) == 3
    assert beam_prompt["negative_points"] == []
    assert beam_prompt["removed_support_negatives"] == 3
    assert set(case["head_stem_edges"]) == {
        ("deim_000003", "synthetic_stem_00012"),
        ("deim_000013", "synthetic_stem_00013"),
    }
    assert set(case["beam_stem_edges"]) == {
        ("deim_000069", "synthetic_stem_00012"),
        ("deim_000069", "synthetic_stem_00013"),
    }


def test_prompt_marker_groups_include_retained_and_suppressed_negatives() -> None:
    case = figure.load_case()
    accepted = case["prompt_markers"]["deim_000069"]
    retained = case["prompt_markers"]["deim_000278"]

    assert len(accepted["positive"]) == 3
    assert accepted["retained_negative"] == []
    assert len(accepted["suppressed_negative"]) == 3
    assert len(retained["positive"]) == 3
    assert len(retained["retained_negative"]) == 3
    assert retained["suppressed_negative"] == []
    x0, y0, x1, y1 = case["crop"]
    assert all(x0 <= x <= x1 and y0 <= y <= y1 for x, y in retained["retained_negative"])


def test_build_writes_single_page_pdf_and_renderable_preview(tmp_path: Path) -> None:
    paths = [Path(path) for path in figure.build(tmp_path)]

    assert {path.suffix for path in paths} == {".png", ".pdf"}
    assert all(path.exists() and path.stat().st_size > 0 for path in paths)
    with Image.open(next(path for path in paths if path.suffix == ".png")) as preview:
        width, height = preview.size
    assert 2.0 <= width / height <= 3.5
    assert height >= 500
