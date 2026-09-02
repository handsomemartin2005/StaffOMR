from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

from run_debussy_sam2_adapted_demo import (
    apply_relation_head_filter,
    attach_available_masks,
    filtered_payload,
    relation_delta,
)


class ConstantModel:
    def __init__(self, positive: float) -> None:
        self.positive = positive

    def predict_proba(self, features):
        import numpy as np

        return np.asarray([[1.0 - self.positive, self.positive] for _ in features])


class AdaptedSam2DemoTests(unittest.TestCase):
    def test_beam_only_filter_and_attachment_preserve_fallback(self) -> None:
        payload = {
            "symbols": [
                {"id": "beam", "class": "beam", "source": "detector"},
                {"id": "note", "class": "filled_notehead", "source": "detector"},
            ]
        }
        selected = filtered_payload(payload, {"beam"})
        self.assertEqual([item["id"] for item in selected["symbols"]], ["beam"])
        attached, report = attach_available_masks(
            payload, {"masks": [{"symbol_id": "beam", "mask_area": 8}]}, {"beam"}
        )
        by_id = {item["id"]: item for item in attached["symbols"]}
        self.assertIn("mask", by_id["beam"])
        self.assertNotIn("mask", by_id["note"])
        self.assertEqual(report, {"selected": 1, "attached": 1, "missing": 0})

    def test_relation_delta_counts_beam_changes(self) -> None:
        reference = {
            "relations": [{"type": "beam_stem_group", "source": "b", "targets": ["s1"]}]
        }
        candidate = {
            "relations": [{"type": "beam_stem_group", "source": "b", "targets": ["s2"]}]
        }
        result = relation_delta(reference, candidate)
        self.assertEqual(result["beam_relations_added"], 1)
        self.assertEqual(result["beam_relations_removed"], 1)

    def test_relation_head_can_remove_low_scoring_target(self) -> None:
        shapes = {
            "symbols": [
                {"id": "b", "class": "beam", "bbox": [0, 0, 20, 4], "mask": {"points": [[0, 0], [20, 0], [20, 4], [0, 4]]}},
                {"id": "n", "class": "filled_notehead", "bbox": [5, 20, 11, 26]},
                {"id": "s", "class": "stem", "bbox": [8, 4, 10, 20], "attributes": {"notehead_id": "n"}},
            ],
            "relations": [{"type": "beam_stem_group", "source": "b", "targets": ["s"], "evidence": {}}],
        }
        filtered, report = apply_relation_head_filter(shapes, ConstantModel(0.2), threshold=0.61)
        self.assertEqual(filtered["relations"], [])
        self.assertEqual(report["targets_removed"], 1)


if __name__ == "__main__":
    unittest.main()
