from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

from train_muscima_sam2_decoder import decoder_loss, downsample_targets, parse_writer_range, writer_id


class TrainMuscimaSam2DecoderTests(unittest.TestCase):
    def test_writer_split_helpers(self) -> None:
        self.assertEqual(parse_writer_range("1-3,5"), {1, 2, 3, 5})
        self.assertEqual(writer_id(Path("CVC-MUSCIMA_W-31_N-02_D-ideal.xml")), 31)

    def test_decoder_loss_is_finite_and_differentiable(self) -> None:
        logits = torch.zeros((2, 1, 16, 16), requires_grad=True)
        target = torch.zeros_like(logits)
        target[:, :, 4:12, 7:9] = 1.0
        roi = torch.ones_like(logits)
        loss, parts = decoder_loss(logits, target, roi)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(set(parts), {"bce", "dice", "boundary", "cldice"})
        loss.backward()
        self.assertIsNotNone(logits.grad)

    def test_area_downsampling_preserves_thin_line_occupancy(self) -> None:
        target = torch.zeros((1, 1, 8, 8))
        target[:, :, 1, :] = 1.0
        nearest = downsample_targets(target, size=(2, 2), mode="nearest")
        area = downsample_targets(target, size=(2, 2), mode="area")
        self.assertEqual(float(nearest.sum()), 0.0)
        self.assertGreater(float(area.sum()), 0.0)
        self.assertTrue(torch.all((area >= 0.0) & (area <= 1.0)))

    def test_downsampling_rejects_unknown_mode(self) -> None:
        with self.assertRaises(ValueError):
            downsample_targets(torch.zeros((1, 1, 8, 8)), size=(2, 2), mode="bilinear")


if __name__ == "__main__":
    unittest.main()
