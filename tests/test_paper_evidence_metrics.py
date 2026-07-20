from fractions import Fraction

import pytest

from tools.paper_evidence_metrics import (
    Event,
    aggregate_counts,
    decompose_event_errors,
    multiset_counts,
    paired_bootstrap_delta,
)


def test_multiset_projections_handle_duplicates_rests_and_overprediction() -> None:
    gold = [
        Event("C0@4", Fraction(1, 1)),
        Event("C0@4", Fraction(1, 1)),
        Event("D0@4", Fraction(1, 2)),
        Event("R", Fraction(1, 1)),
    ]
    pred = [
        Event("C0@4", Fraction(1, 1)),
        Event("C0@4", Fraction(1, 2)),
        Event("D0@4", Fraction(1, 2)),
        Event("R", Fraction(2, 1)),
        Event("E0@4", Fraction(1, 1)),
    ]

    joint = multiset_counts(gold, pred, "joint")
    pitch = multiset_counts(gold, pred, "pitch")
    duration = multiset_counts(gold, pred, "duration")

    assert joint == {"matches": 2, "predicted": 5, "gold": 4}
    assert pitch == {"matches": 4, "predicted": 5, "gold": 4}
    assert duration == {"matches": 3, "predicted": 5, "gold": 4}


def test_empty_prediction_has_zero_scores_and_preserves_gold_denominator() -> None:
    row = multiset_counts([Event("R", Fraction(1, 1))], [], "joint")
    summary = aggregate_counts([row])

    assert summary["matches"] == 0
    assert summary["predicted_events"] == 0
    assert summary["gold_events"] == 1
    assert summary["precision"] == 0.0
    assert summary["recall"] == 0.0
    assert summary["f1"] == 0.0
    assert summary["output_to_gold_ratio"] == 0.0


def test_error_decomposition_reports_marginal_opportunities() -> None:
    gold = [Event("C0@4", Fraction(1, 1)), Event("D0@4", Fraction(1, 2))]
    pred = [Event("C0@4", Fraction(1, 2)), Event("E0@4", Fraction(1, 2))]

    result = decompose_event_errors(gold, pred)

    assert result["joint"]["matches"] == 0
    assert result["pitch"]["matches"] == 1
    assert result["duration"]["matches"] == 1
    assert result["pitch_correct_duration_wrong_opportunities"] == 1
    assert result["duration_correct_pitch_wrong_opportunities"] == 1


def test_paired_bootstrap_is_deterministic_and_validates_pair_count() -> None:
    full = [
        {"matches": 2, "predicted": 2, "gold": 2},
        {"matches": 1, "predicted": 2, "gold": 2},
    ]
    ablated = [
        {"matches": 1, "predicted": 2, "gold": 2},
        {"matches": 0, "predicted": 2, "gold": 2},
    ]

    first = paired_bootstrap_delta(full, ablated, samples=100, seed=7)
    second = paired_bootstrap_delta(full, ablated, samples=100, seed=7)

    assert first == second
    assert first["full_minus_ablated"] == pytest.approx(50.0)
    with pytest.raises(ValueError, match="equal page counts"):
        paired_bootstrap_delta(full, ablated[:1], samples=100, seed=7)
