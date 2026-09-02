from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

from build_sam2_prompt_comparison import binary_mask_metrics


class Sam2PromptComparisonTests(unittest.TestCase):
    def test_binary_mask_metrics_reports_overlap_and_changed_pixels(self) -> None:
        box_only = np.asarray([[1, 1, 0], [0, 1, 0]], dtype=bool)
        staff_aware = np.asarray([[1, 0, 0], [0, 1, 1]], dtype=bool)

        metrics = binary_mask_metrics(box_only, staff_aware)

        self.assertEqual(metrics["box_only_area"], 3)
        self.assertEqual(metrics["staff_aware_area"], 3)
        self.assertEqual(metrics["intersection"], 2)
        self.assertEqual(metrics["union"], 4)
        self.assertAlmostEqual(metrics["iou"], 0.5)
        self.assertEqual(metrics["changed_pixels"], 2)

    def test_binary_mask_metrics_rejects_mismatched_shapes(self) -> None:
        with self.assertRaisesRegex(ValueError, "same shape"):
            binary_mask_metrics(np.zeros((2, 2), dtype=bool), np.zeros((3, 2), dtype=bool))


if __name__ == "__main__":
    unittest.main()
