from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

from train_muscima_beam_notehead_relation_head import (
    BOX_FEATURE_NAMES,
    MASK_FEATURE_NAMES,
    calibrate_pruning_threshold,
    edge_metrics,
    is_candidate,
    pair_features,
    parse_writer_range,
)


class BeamNoteheadRelationHeadTests(unittest.TestCase):
    def test_pair_features_include_mask_geometry(self) -> None:
        beam = {
            "left": 10,
            "top": 10,
            "width": 20,
            "height": 4,
            "mask": np.ones((4, 20), dtype=bool),
        }
        notehead = {"left": 14, "top": 30, "width": 6, "height": 6}
        box, mask = pair_features(beam, notehead)
        self.assertEqual(len(box), len(BOX_FEATURE_NAMES))
        self.assertEqual(len(mask), len(BOX_FEATURE_NAMES) + len(MASK_FEATURE_NAMES))
        self.assertTrue(np.isfinite(mask).all())
        self.assertTrue(is_candidate(beam, notehead))

    def test_mask_origin_override_changes_only_mask_features(self) -> None:
        beam = {
            "left": 10,
            "top": 10,
            "width": 20,
            "height": 4,
            "mask": np.ones((2, 20), dtype=bool),
        }
        notehead = {"left": 14, "top": 30, "width": 6, "height": 6}
        box_a, mask_a = pair_features(beam, notehead)
        moved = {**beam, "mask_top": 20}
        box_b, mask_b = pair_features(moved, notehead)
        self.assertEqual(box_a, box_b)
        self.assertNotEqual(mask_a[-5], mask_b[-5])

    def test_edge_metrics_count_missed_candidates_as_false_negative(self) -> None:
        result = edge_metrics(
            np.asarray([1, 0, 1]),
            np.asarray([0.9, 0.8, 0.1]),
            threshold=0.5,
            gold_edges=3,
        )
        self.assertEqual(result["true_positive"], 1)
        self.assertEqual(result["false_positive"], 1)
        self.assertEqual(result["false_negative"], 2)

    def test_writer_range(self) -> None:
        self.assertEqual(parse_writer_range("1-3,5"), {1, 2, 3, 5})

    def test_pruning_calibration_respects_minimum_recall(self) -> None:
        result = calibrate_pruning_threshold(
            np.asarray([1, 1, 0]),
            np.asarray([0.9, 0.2, 0.1]),
            gold_edges=2,
            minimum_recall=1.0,
        )
        self.assertLessEqual(result["threshold"], 0.2)
        self.assertEqual(result["recall"], 1.0)


if __name__ == "__main__":
    unittest.main()
