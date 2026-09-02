from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

from summarize_debussy_sam2_adapted_full import paired_summary


def test_paired_summary_counts_page_wins_ties_losses() -> None:
    rows = [
        {"candidate": {"matches": 2, "predicted": 2, "gold": 2}, "baseline": {"matches": 1, "predicted": 2, "gold": 2}},
        {"candidate": {"matches": 1, "predicted": 2, "gold": 2}, "baseline": {"matches": 1, "predicted": 2, "gold": 2}},
        {"candidate": {"matches": 0, "predicted": 2, "gold": 2}, "baseline": {"matches": 1, "predicted": 2, "gold": 2}},
    ]
    result = paired_summary(rows, "candidate", "baseline", samples=20, seed=3)
    assert (result["wins"], result["ties"], result["losses"]) == (1, 1, 1)
