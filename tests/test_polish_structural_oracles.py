from __future__ import annotations

from tools.polish_structural_oracles import (
    event_binding_oracle,
    parse_ekern_hierarchy,
    split_normalized_event,
)


def normalize(cell: str) -> list[str]:
    return [token for token in cell.replace(".", "").split() if token]


def test_ekern_parser_preserves_staff_onset_chord_and_voice_spines() -> None:
    text = """**ekern\t**ekern
*clefF4\t*clefG2
=\t=
4C\t4c 4e
*\t*^
.\t8g\t8b
*\t*v\t*v
4D\t4d
*-\t*-
"""
    onsets = parse_ekern_hierarchy(text, normalize)
    assert len(onsets) == 3
    assert [(event.base_staff, event.chord_index) for event in onsets[0].events] == [
        (0, 0),
        (1, 0),
        (1, 1),
    ]
    assert [event.base_staff for event in onsets[1].events] == [1, 1]
    assert [event.voice_path for event in onsets[1].events] == [(0,), (1,)]
    assert [event.voice_path for event in onsets[2].events] == [(), ()]


def test_split_normalized_event_separates_duration_and_pitch() -> None:
    assert split_normalized_event("16.:cc#") == ("16.", "cc#")
    assert split_normalized_event("4:r") == ("4", "r")
    assert split_normalized_event("metadata") is None


def test_binding_oracle_preserves_marginals_but_can_swap_duration_assignments() -> None:
    # Predicted pitches and durations are both correct, but paired to the wrong notes.
    gold = ["4:c", "8:e", "4:r"]
    predicted = ["8:c", "4:e", "4:r"]
    result = event_binding_oracle(gold, predicted)
    assert result["current_token_bag_matches"] == 1
    assert result["oracle_note_binding_matches"] == 2
    assert result["fixed_rest_matches"] == 1
    assert result["oracle_total_matches"] == 3
    assert result["oracle_SER_lower_bound"] == 0.0
