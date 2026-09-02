from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

from eval_muscima_sam2_masks import decode_rle, mask_metrics, parse_writer_spec


class MuscimaSam2MaskEvalTests(unittest.TestCase):
    def test_decode_rle_row_major(self) -> None:
        decoded = decode_rle("0:2 1:2 0:2", 2, 3)
        np.testing.assert_array_equal(decoded, np.asarray([[0, 0, 1], [1, 0, 0]], dtype=bool))

    def test_identical_masks_are_perfect(self) -> None:
        mask = np.asarray([[0, 1, 0], [1, 1, 0], [0, 0, 0]], dtype=bool)
        result = mask_metrics(mask, mask)
        self.assertEqual(result["iou"], 1.0)
        self.assertEqual(result["boundary_f1"], 1.0)
        self.assertEqual(result["leakage"], 0.0)
        self.assertEqual(result["area_ratio"], 1.0)

    def test_parse_writer_spec(self) -> None:
        self.assertEqual(parse_writer_spec("41-43,50"), {41, 42, 43, 50})


if __name__ == "__main__":
    unittest.main()
