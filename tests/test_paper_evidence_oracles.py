from tools.build_oracle_diagnostics import oracle_counts_from_decomposition


def test_oracle_pitch_and_duration_use_only_the_opposite_fixed_attribute() -> None:
    row = {
        "joint": {"matches": 2, "predicted": 7, "gold": 5},
        "pitch": {"matches": 4, "predicted": 7, "gold": 5},
        "duration": {"matches": 3, "predicted": 7, "gold": 5},
    }
    result = oracle_counts_from_decomposition(row)
    assert result["oracle_pitch"] == {"matches": 3, "predicted": 7, "gold": 5}
    assert result["oracle_duration"] == {"matches": 4, "predicted": 7, "gold": 5}
    assert result["oracle_count"] == {"matches": 2, "predicted": 5, "gold": 5}


def test_oracle_count_handles_empty_and_underprediction_without_inventing_matches() -> None:
    empty = {
        "joint": {"matches": 0, "predicted": 0, "gold": 4},
        "pitch": {"matches": 0, "predicted": 0, "gold": 4},
        "duration": {"matches": 0, "predicted": 0, "gold": 4},
    }
    under = {
        "joint": {"matches": 1, "predicted": 2, "gold": 4},
        "pitch": {"matches": 2, "predicted": 2, "gold": 4},
        "duration": {"matches": 1, "predicted": 2, "gold": 4},
    }
    assert oracle_counts_from_decomposition(empty)["oracle_count"]["matches"] == 0
    assert oracle_counts_from_decomposition(under)["oracle_count"] == {
        "matches": 1,
        "predicted": 2,
        "gold": 4,
    }
