from __future__ import annotations

from tools.polish_canonical_serialization import (
    canonical_parts,
    group_piano_systems,
    semantic_to_canonical_ekern,
    system_measure_boundaries,
)


def note(
    event_id: str,
    x: float,
    y: float,
    step: str,
    octave: int,
    *,
    stem_id: str | None = None,
) -> dict:
    return {
        "id": event_id,
        "type": "note",
        "x": x,
        "center": [x, y],
        "duration_hint": "quarter",
        "dot_count": 0,
        "pitch": {"step": step, "octave": octave},
        "stem_id": stem_id,
    }


def part(staff: int, clef: str, events: list[dict], barlines: list[float] | None = None) -> dict:
    return {
        "id": f"P{staff + 1}",
        "staff": staff,
        "clef_type": clef,
        "barlines": barlines or [],
        "measures": [{"number": 99, "events": events}],
    }


def test_parts_and_systems_follow_physical_page_order() -> None:
    semantics = {
        "parts": [
            part(3, "bass", [note("d", 20, 400, "D", 3)]),
            part(0, "treble", [note("a", 20, 100, "C", 5)]),
            part(2, "treble", [note("c", 20, 300, "C", 5)]),
            part(1, "bass", [note("b", 20, 200, "D", 3)]),
        ]
    }
    ordered = canonical_parts(semantics)
    systems = group_piano_systems(ordered)
    assert [[item["staff"] for item in system] for system in systems] == [[0, 1], [2, 3]]


def test_measure_boundaries_align_staves_and_reject_single_staff_false_positive() -> None:
    system = [
        part(0, "treble", [], [100.0, 200.0, 301.0]),
        part(1, "bass", [], [102.0, 300.0]),
    ]
    assert system_measure_boundaries(system, tolerance=4.0) == [101.0, 300.5]


def test_canonical_serialization_is_system_then_onset_then_lower_upper_staff() -> None:
    semantics = {
        "parts": [
            part(
                0,
                "treble",
                [
                    note("upper-late", 80, 100, "G", 5),
                    note("upper-high", 21, 100, "E", 5, stem_id="s1"),
                    note("upper-low", 20, 110, "C", 5, stem_id="s1"),
                ],
                [50],
            ),
            part(1, "bass", [note("lower", 22, 200, "C", 3)], [51]),
            part(2, "treble", [note("next-system", 10, 300, "D", 5)]),
            part(3, "bass", [note("next-system-lower", 11, 400, "D", 3)]),
        ]
    }
    text = semantic_to_canonical_ekern(semantics, onset_tolerance=4.0, barline_tolerance=4.0)
    data = [line for line in text.splitlines() if line and not line.startswith("*") and not line.startswith("=")]
    assert data == [
        "4@C\t4@cc 4@ee",
        ".\t4@gg",
        "4@D\t4@dd",
    ]


def test_canonical_serialization_is_deterministic_for_unordered_input() -> None:
    first = part(0, "treble", [note("b", 10, 100, "G", 4), note("a", 10, 100, "C", 4)])
    second = part(1, "bass", [])
    a = semantic_to_canonical_ekern({"parts": [first, second]})
    first["measures"][0]["events"].reverse()
    b = semantic_to_canonical_ekern({"parts": [second, first]})
    assert a == b
    assert ".\t4@c 4@g" in a


def test_proxy_dispatch_can_identify_polish_without_changing_other_datasets() -> None:
    import sys
    from pathlib import Path

    tools_dir = Path(__file__).resolve().parent.parent / "tools"
    sys.path.insert(0, str(tools_dir))
    try:
        import evaluate_kern_text_proxy_metrics as metrics

        semantics = {
            "input": "data/ijcv_samples/polish-scores-test24/test_0000.png",
            "parts": [part(0, "treble", [note("n", 10, 100, "C", 4)])],
        }
        assert "\t" in metrics.semantic_to_pseudo_ekern(semantics).splitlines()[0]
        semantics["input"] = "data/fp-grandstaff/example.png"
        assert metrics.semantic_to_pseudo_ekern(semantics).splitlines()[0] == "**ekern"
    finally:
        sys.path.remove(str(tools_dir))
