from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import refine_masks_sam2 as sam2_refiner


class MethodAlignedPromptTests(unittest.TestCase):
    def test_default_prompt_strategy_is_method_aligned(self) -> None:
        self.assertEqual(sam2_refiner.DEFAULT_PROMPT_STRATEGY, sam2_refiner.METHOD_STRATEGY)

    def test_support_filter_removes_only_negative_points_inside_detector_box(self) -> None:
        points = np.asarray(
            [[15, 15], [12, 12], [5, 15], [25, 15]],
            dtype=np.float32,
        )
        labels = np.asarray([1, 0, 0, 0], dtype=np.int32)

        kept_points, kept_labels, removed = sam2_refiner.filter_negative_points_outside_support(
            points,
            labels,
            [10, 10, 20, 20],
        )

        np.testing.assert_array_equal(
            kept_points,
            np.asarray([[15, 15], [5, 15], [25, 15]], dtype=np.float32),
        )
        np.testing.assert_array_equal(kept_labels, np.asarray([1, 0, 0], dtype=np.int32))
        self.assertEqual(removed, 1)

    def test_notehead_box_is_expanded_before_sam(self) -> None:
        symbol = {
            "id": "n1",
            "class": "filled_notehead",
            "bbox": [10, 20, 30, 40],
            "attributes": {"staff_space": 12.0},
        }
        box, metadata = sam2_refiner.adaptive_prompt_box(symbol, (100, 80))
        self.assertEqual(box, [6, 16, 34, 44])
        self.assertEqual(metadata["detector_box"], [10, 20, 30, 40])
        self.assertEqual(metadata["adaptive_box_rule"], "notehead_fractional_expand")

    def test_crop_contains_adaptive_box_with_context(self) -> None:
        crop = sam2_refiner.image_crop_box([20, 30, 40, 50], (100, 100), 10.0, 1.0)
        self.assertEqual(crop, [10, 20, 50, 60])

    def test_points_and_box_are_translated_to_crop_coordinates(self) -> None:
        crop = [10, 20, 70, 80]
        box = sam2_refiner._local_box([20, 30, 40, 50], crop)
        np.testing.assert_array_equal(box, np.asarray([10, 10, 30, 30], dtype=np.float32))
        points = np.asarray([[20, 30], [65, 75], [5, 5]], dtype=np.float32)
        labels = np.asarray([1, 0, 0], dtype=np.int32)
        local, kept_labels, _, _ = sam2_refiner._local_points(points, labels, crop)
        np.testing.assert_array_equal(local, np.asarray([[10, 10], [55, 55]], dtype=np.float32))
        np.testing.assert_array_equal(kept_labels, np.asarray([1, 0], dtype=np.int32))

    def test_crop_mask_is_mapped_back_to_full_image(self) -> None:
        local = np.ones((4, 5), dtype=bool)
        full = sam2_refiner._full_size_mask(local, [2, 3, 7, 7], (10, 9))
        self.assertEqual(full.shape, (9, 10))
        self.assertEqual(int(full.sum()), 20)
        self.assertTrue(full[3:7, 2:7].all())


if __name__ == "__main__":
    unittest.main()
