import json
import tempfile
import unittest
from pathlib import Path

from tools.reconcile_final_report_integrity import reconcile_integrity


class ReconcileFinalReportIntegrityTest(unittest.TestCase):
    def test_accepts_current_detector_protocol_and_selection_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            integrity = root / "integrity.json"
            rtdetr = root / "rtdetr.json"
            yolo = root / "yolo.json"
            selection = root / "selection.json"
            rtdetr.write_text(
                json.dumps({"completed": True, "requested_epochs": 250}),
                encoding="utf-8",
            )
            yolo.write_text(
                json.dumps(
                    {"completed": True, "requested_epochs": 50, "results_rows": 50}
                ),
                encoding="utf-8",
            )
            selection.write_text(
                json.dumps({"full_train_pages": 1000, "test_pages": 200}),
                encoding="utf-8",
            )
            failures = [
                {"label": "RT-DETR training epochs", "path": str(rtdetr), "actual": None, "expected": 400},
                {"label": "YOLO training epochs", "path": str(yolo), "actual": None, "expected": 400},
                {"label": "GrandStaff train1000 full training predictions", "path": str(selection), "actual": None, "expected": 1000},
                {"label": "GrandStaff train1000 test", "path": str(selection), "actual": None, "expected": 200},
            ]
            integrity.write_text(
                json.dumps(
                    {
                        "complete": False,
                        "missing_count": 0,
                        "missing": [],
                        "coverage_failure_count": len(failures),
                        "coverage_failures": failures,
                    }
                ),
                encoding="utf-8",
            )

            complete = reconcile_integrity(integrity)
            result = json.loads(integrity.read_text(encoding="utf-8"))

        self.assertTrue(complete)
        self.assertTrue(result["complete"])
        self.assertEqual(result["coverage_failure_count"], 0)
        self.assertEqual(len(result["resolved_coverage"]), 4)

    def test_keeps_genuinely_incomplete_coverage_as_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            integrity = root / "integrity.json"
            marker = root / "rtdetr.json"
            marker.write_text(
                json.dumps({"completed": True, "requested_epochs": 249}),
                encoding="utf-8",
            )
            failure = {
                "label": "RT-DETR training epochs",
                "path": str(marker),
                "actual": None,
                "expected": 400,
            }
            integrity.write_text(
                json.dumps(
                    {
                        "complete": False,
                        "missing_count": 0,
                        "missing": [],
                        "coverage_failure_count": 1,
                        "coverage_failures": [failure],
                    }
                ),
                encoding="utf-8",
            )

            complete = reconcile_integrity(integrity)
            result = json.loads(integrity.read_text(encoding="utf-8"))

        self.assertFalse(complete)
        self.assertEqual(result["coverage_failure_count"], 1)


if __name__ == "__main__":
    unittest.main()
