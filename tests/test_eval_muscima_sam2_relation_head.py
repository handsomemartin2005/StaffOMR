from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

from eval_muscima_sam2_relation_head import crop_mask


class Sam2RelationHeadEvalTests(unittest.TestCase):
    def test_crop_mask_preserves_global_origin(self) -> None:
        mask = np.zeros((8, 9), dtype=bool)
        mask[3:5, 4:7] = True
        result = crop_mask(mask)
        self.assertEqual(result["mask_left"], 4)
        self.assertEqual(result["mask_top"], 3)
        np.testing.assert_array_equal(result["mask"], np.ones((2, 3), dtype=bool))

    def test_crop_empty_mask_is_valid(self) -> None:
        result = crop_mask(np.zeros((2, 3), dtype=bool))
        self.assertEqual(result["mask"].shape, (1, 1))
        self.assertFalse(result["mask"].any())


if __name__ == "__main__":
    unittest.main()
