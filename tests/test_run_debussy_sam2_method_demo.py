from __future__ import annotations

import json
import sys
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

from run_debussy_sam2_method_demo import (
    aggregate,
    attach_masks,
    build_safe_residual_shapes,
    DEFAULT_MASK_SELECTION,
    select_prompt_masks,
    semantics_events,
)


class DebussySam2MethodDemoTests(unittest.TestCase):
    def test_default_mask_consumer_is_safe_residual(self) -> None:
        self.assertEqual(DEFAULT_MASK_SELECTION, "safe_residual")

    def test_safe_residual_integration_preserves_box_topology(self) -> None:
        box_shapes = {
            "symbols": [
                {"id": "beam", "class": "beam", "bbox": [0, 0, 20, 4], "attributes": {"staff_space": 10}},
                {"id": "stem", "class": "stem", "bbox": [10, 2, 12, 20], "skeleton": {"points": [[11, 2], [11, 20]]}},
            ],
            "relations": [{"id": "r", "type": "beam_stem_group", "source": "beam", "targets": ["stem"], "score": 0.5}],
        }
        staff_shapes = {
            "symbols": [
                {"id": "beam", "class": "beam", "bbox": [0, 0, 20, 4], "attributes": {"staff_space": 10}, "mask_source": "sam2_method_aligned", "skeleton": {"points": [[0, 2], [11, 2], [20, 2]]}}
            ]
        }

        result, report = build_safe_residual_shapes(box_shapes, staff_shapes)

        self.assertEqual(result["relations"][0]["targets"], ["stem"])
        self.assertGreater(result["relations"][0]["score"], 0.5)
        self.assertEqual(report["relations_before"], report["relations_after"])

    def test_select_prompt_masks_uses_higher_sam_score_without_gold_labels(self) -> None:
        box_only = {
            "masks": [
                {"symbol_id": "n1", "mask_score": 0.80, "prompt_strategy": "box_only"},
                {"symbol_id": "n2", "mask_score": 0.70, "prompt_strategy": "box_only"},
            ]
        }
        staff_aware = {
            "masks": [
                {"symbol_id": "n1", "mask_score": 0.90, "prompt_strategy": "method_aligned"},
                {"symbol_id": "n2", "mask_score": 0.60, "prompt_strategy": "method_aligned"},
            ]
        }

        selected = select_prompt_masks(box_only, staff_aware)

        self.assertEqual(
            [item["selected_prompt"] for item in selected["masks"]],
            ["staff_aware", "box_only"],
        )
        self.assertEqual(selected["selection_counts"], {"staff_aware": 1, "box_only": 1})
        self.assertEqual(selected["masks"][0]["alternative_mask_scores"], {"box_only": 0.8, "staff_aware": 0.9})

    def test_select_prompt_masks_limits_staff_aware_to_template_families(self) -> None:
        box_only = {
            "masks": [
                {"symbol_id": "beam1", "class": "beam", "mask_score": 0.70},
                {"symbol_id": "clef1", "class": "treble_clef", "mask_score": 0.60},
            ]
        }
        staff_aware = {
            "masks": [
                {"symbol_id": "beam1", "class": "beam", "mask_score": 0.80},
                {"symbol_id": "clef1", "class": "treble_clef", "mask_score": 0.95},
            ]
        }

        selected = select_prompt_masks(box_only, staff_aware, eligible_classes={"beam"})

        self.assertEqual(
            [item["selected_prompt"] for item in selected["masks"]],
            ["staff_aware", "box_only"],
        )
        self.assertEqual(selected["eligible_classes"], ["beam"])

    def test_semantics_events_normalizes_duration_by_declared_divisions(self) -> None:
        payload = {
            "divisions": 16,
            "parts": [
                {
                    "measures": [
                        {
                            "events": [
                                {
                                    "type": "note",
                                    "pitch": {"step": "F", "alter": -1, "octave": 3},
                                    "duration_units": 16,
                                },
                                {"type": "rest", "duration_units": 8},
                                {"type": "mark", "duration_units": 16},
                            ]
                        }
                    ]
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "semantics.json"
            path.write_text(json.dumps(payload), encoding="utf-8")

            events = semantics_events(path)

        self.assertEqual(events, [("F-1@3", Fraction(1, 1)), ("R", Fraction(1, 2))])

    def test_attach_masks_preserves_detector_box_and_adds_mask(self) -> None:
        payload = {
            "symbols": [
                {"id": "n1", "class": "filled_notehead", "bbox": [1, 2, 3, 4], "source": "deim"}
            ]
        }
        masks = {
            "masks": [
                {
                    "symbol_id": "n1",
                    "bbox": [1, 2, 3, 4],
                    "mask_path": "mask.png",
                    "source": "sam2_method_aligned",
                }
            ]
        }
        result = attach_masks(payload, masks)
        self.assertEqual(result["symbols"][0]["bbox"], [1, 2, 3, 4])
        self.assertEqual(result["symbols"][0]["mask"]["source"], "sam2_method_aligned")
        self.assertEqual(result["symbols"][0]["source"], "deim+sam2")

    def test_aggregate_matches_debussy_event_f1_definition(self) -> None:
        rows = [{"variant": {"matches": 2, "predicted": 3, "gold": 5}}]
        result = aggregate(rows, "variant")
        self.assertEqual(result["f1"], 50.0)


if __name__ == "__main__":
    unittest.main()
