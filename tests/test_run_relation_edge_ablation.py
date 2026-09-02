import copy
from pathlib import Path
import subprocess
import sys

from tools.run_relation_edge_ablation import (
    apply_relation_ablation_to_notes,
    build_notes_from_shapes,
    filter_relation_type,
    load_assemble_module,
    run_ablation,
)


def test_filter_relation_type_removes_only_selected_edges_without_mutation() -> None:
    shapes = {
        "version": "x",
        "symbols": [{"id": "n1", "class": "filled_notehead"}],
        "relations": [
            {"id": "r1", "type": "notehead_stem_attachment", "source": "n1"},
            {"id": "r2", "type": "beam_stem_group", "source": "b1"},
            {"id": "r3", "type": "slur_tie_notehead_endpoints", "source": "s1"},
        ],
    }
    original = copy.deepcopy(shapes)

    filtered = filter_relation_type(shapes, "beam_stem_group")

    assert shapes == original
    assert filtered["symbols"] == shapes["symbols"]
    assert [edge["id"] for edge in filtered["relations"]] == ["r1", "r3"]


def test_filter_with_absent_type_preserves_relation_order() -> None:
    shapes = {
        "relations": [
            {"id": "r1", "type": "notehead_stem_attachment"},
            {"id": "r2", "type": "beam_stem_group"},
        ]
    }
    filtered = filter_relation_type(shapes, "absent")
    assert filtered == shapes
    assert filtered is not shapes


def test_cli_can_be_invoked_directly() -> None:
    root = Path(__file__).resolve().parent.parent
    result = subprocess.run(
        [sys.executable, "tools/run_relation_edge_ablation.py", "--help"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_one_page_smoke_uses_page_gate_without_full_dataset_aggregate_gate(tmp_path) -> None:
    payload = run_ablation(tmp_path / "smoke", limit=1, bootstrap_samples=10, seed=3)
    assert payload["pages"] == 1
    assert payload["variants"]["full"]["summary"]["matches"] == 25
    assert payload["variants"]["full"]["summary"]["predicted_events"] == 291


def test_filtered_relations_are_reassembled_into_consumed_note_fields() -> None:
    root = Path(__file__).resolve().parent.parent
    shapes_path = root / "outputs/debussy_abcd_ablation/A1B1C1D1/test_0000/symbols/shapes.json"
    import json

    shapes = json.loads(shapes_path.read_text(encoding="utf-8"))
    assemble = load_assemble_module()
    full = build_notes_from_shapes(assemble, shapes)
    ablated = build_notes_from_shapes(
        assemble, filter_relation_type(shapes, "notehead_stem_attachment")
    )

    full_stems = sum(event.get("stem_id") is not None for event in full["events"])
    ablated_stems = sum(event.get("stem_id") is not None for event in ablated["events"])
    assert full_stems > 0
    assert ablated_stems == 0

    original_notes_path = root / "outputs/debussy_abcd_ablation/A1B1C1D1/test_0000/notes/notes.json"
    original = json.loads(original_notes_path.read_text(encoding="utf-8"))
    patched = apply_relation_ablation_to_notes(
        original, ablated, "notehead_stem_attachment"
    )
    assert sum(event.get("stem_id") is not None for event in patched["events"]) == 0
    assert patched["events"][0]["type"] == original["events"][0]["type"]
    assert patched["events"][0]["center"] == original["events"][0]["center"]


def test_full_dataset_uses_aggregate_reproduction_gate(tmp_path) -> None:
    payload = run_ablation(tmp_path / "full", limit=24, bootstrap_samples=10, seed=4)
    assert payload["reproduction_gate"]["absolute_f1_difference"] <= 0.01
