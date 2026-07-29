from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

from run_debussy_sam2_selective_demo import (
    apply_mask_rerank,
    dense_notehead_ids,
    fuse_mask_contact_relations,
    fuse_mask_safe_residual,
    mask_quality,
    pre_sam2_ambiguous_ids,
    selective_payload,
)


class SelectiveSam2Tests(unittest.TestCase):
    def test_dense_noteheads_are_selected(self) -> None:
        symbols = [
            {"id": "a", "class": "filled_notehead", "bbox": [0, 0, 10, 10], "attributes": {"staff": 0, "staff_space": 10}},
            {"id": "b", "class": "open_notehead", "bbox": [5, 5, 15, 15], "attributes": {"staff": 0, "staff_space": 10}},
            {"id": "c", "class": "filled_notehead", "bbox": [100, 100, 110, 110], "attributes": {"staff": 0, "staff_space": 10}},
        ]
        self.assertEqual(dense_notehead_ids(symbols), {"a", "b"})

    def test_pre_gate_selects_beam_group_but_not_isolated_notehead(self) -> None:
        symbols = [
            {"id": "n1", "class": "filled_notehead", "bbox": [10, 30, 20, 40]},
            {"id": "n2", "class": "filled_notehead", "bbox": [40, 30, 50, 40]},
            {"id": "n3", "class": "filled_notehead", "bbox": [200, 30, 210, 40]},
            {"id": "b1", "class": "beam", "bbox": [5, 10, 55, 15]},
        ]
        for symbol in symbols:
            symbol["attributes"] = {"staff_space": 10, "staff": 1}
        selected = pre_sam2_ambiguous_ids(symbols)
        self.assertTrue({"b1", "n1", "n2"}.issubset(selected))
        self.assertNotIn("n3", selected)

    def test_source_domain_class_gate_falls_back_for_non_beam(self) -> None:
        payload = {
            "symbols": [
                {"id": "b", "class": "beam", "bbox": [0, 0, 20, 4]},
                {"id": "n", "class": "filled_notehead", "bbox": [0, 0, 10, 10]},
            ]
        }
        masks = {
            "masks": [
                {"symbol_id": "b", "mask_score": 0.9, "mask_area": 40},
                {"symbol_id": "n", "mask_score": 0.9, "mask_area": 80},
            ]
        }
        result, report = selective_payload(payload, masks, allowed_classes={"beam"})
        by_id = {item["id"]: item for item in result["symbols"]}
        self.assertIn("mask", by_id["b"])
        self.assertNotIn("mask", by_id["n"])
        self.assertEqual(report["accepted_by_class"], {"beam": 1})

    def test_beam_gate_accepts_only_plausible_mask(self) -> None:
        symbol = {"id": "beam", "class": "beam", "bbox": [0, 0, 100, 10]}
        accepted, _, _ = mask_quality(symbol, {"mask_score": 0.8, "mask_area": 800}, set())
        rejected, reason, _ = mask_quality(symbol, {"mask_score": 0.5, "mask_area": 800}, set())
        self.assertTrue(accepted)
        self.assertFalse(rejected)
        self.assertEqual(reason, "score_below_threshold")

    def test_contact_fusion_removes_noncontacting_beam_relation(self) -> None:
        shapes = {
            "symbols": [
                {
                    "id": "beam",
                    "class": "beam",
                    "bbox": [0, 0, 20, 4],
                    "attributes": {"staff_space": 10},
                    "mask_source": "sam2_box_prompt",
                    "skeleton": {"points": [[0, 2], [20, 2]]},
                },
                {
                    "id": "stem",
                    "class": "stem",
                    "bbox": [50, 40, 52, 60],
                    "skeleton": {"points": [[51, 40], [51, 60]]},
                },
            ],
            "relations": [
                {"id": "r", "type": "beam_stem_group", "source": "beam", "targets": ["stem"], "score": 1.0, "evidence": {}}
            ],
            "v2_1_summary": {},
        }
        result, report = fuse_mask_contact_relations(shapes)
        self.assertEqual(result["relations"], [])
        self.assertEqual(report["beam_relations_removed"], 1)

    def test_safe_residual_preserves_topology_and_never_decreases_score(self) -> None:
        box = {
            "symbols": [
                {"id": "beam", "class": "beam", "bbox": [0, 0, 20, 4], "attributes": {"staff_space": 10}},
                {"id": "stem", "class": "stem", "bbox": [10, 2, 12, 20], "skeleton": {"points": [[11, 2], [11, 20]]}},
            ],
            "relations": [{"id": "r", "type": "beam_stem_group", "source": "beam", "targets": ["stem"], "score": 0.5}],
        }
        masks = {
            "symbols": [{"id": "beam", "class": "beam", "bbox": [0, 0, 20, 4], "attributes": {"staff_space": 10}, "mask_source": "sam2_method_aligned", "skeleton": {"points": [[0, 2], [11, 2], [20, 2]]}}]
        }
        result, report = fuse_mask_safe_residual(box, masks)
        self.assertEqual(len(result["relations"]), len(box["relations"]))
        self.assertEqual(result["relations"][0]["targets"], ["stem"])
        self.assertGreater(result["relations"][0]["score"], box["relations"][0]["score"])
        self.assertEqual(report["mask_relations_boosted"], 1)
        self.assertEqual(report["relations_before"], report["relations_after"])

    def test_mask_rerank_changes_only_uncertain_disagreement(self) -> None:
        shapes = {
            "relations": [
                {"id": "r1", "type": "notehead_stem_attachment", "source": "n", "targets": ["s1"], "score": 0.55, "evidence": {"mask_safe_residual": {"contact_score": 0.2}}},
                {"id": "r2", "type": "notehead_stem_attachment", "source": "n", "targets": ["s2"], "score": 0.50, "evidence": {"mask_safe_residual": {"contact_score": 0.8}}},
            ],
            "v2_1_summary": {},
        }
        result, report = apply_mask_rerank(shapes)
        self.assertEqual(len(result["relations"]), 1)
        self.assertEqual(result["relations"][0]["targets"], ["s2"])
        self.assertEqual(report["mask_reranked_groups"], 1)


if __name__ == "__main__":
    unittest.main()
