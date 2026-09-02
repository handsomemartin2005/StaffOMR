from __future__ import annotations

import sys
import unittest
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from run_debussy_sam2_event_oracle import counter_union, recovery_oracle


class EventRecoveryOracleTests(unittest.TestCase):
    def test_oracle_adds_only_candidate_true_positives_missing_from_base(self) -> None:
        gold = Counter({"a": 1, "b": 1, "c": 1})
        base = Counter({"a": 1, "wrong": 1})
        candidate = Counter({"b": 1, "c": 1, "wrong2": 1})
        result = recovery_oracle(gold, base, candidate)
        self.assertEqual(result["recovered_events"], 2)
        self.assertEqual(result["damaged_events"], 1)
        self.assertEqual(result["conservative_oracle_matches"], 3)
        self.assertEqual(result["conservative_oracle_predicted"], 4)

    def test_counter_union_keeps_maximum_multiplicity(self) -> None:
        combined = counter_union([Counter({"a": 1, "b": 2}), Counter({"a": 2, "b": 1})])
        self.assertEqual(combined, Counter({"a": 2, "b": 2}))


if __name__ == "__main__":
    unittest.main()
