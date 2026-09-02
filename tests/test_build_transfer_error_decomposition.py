from fractions import Fraction
from pathlib import Path
import subprocess
import sys

import pytest

from tools.build_transfer_error_decomposition import build_report
from tools.paper_evidence_metrics import Event


def test_build_report_requires_complete_method_coverage_and_reconstructs_aggregates() -> None:
    pages = [
        {
            "page_id": "test_0000",
            "gold": [Event("C0@4", Fraction(1, 1))],
            "predictions": {
                "staff": [Event("C0@4", Fraction(1, 1))],
                "smt": [],
                "zeus": [Event("D0@4", Fraction(1, 1))],
            },
        },
        {
            "page_id": "test_0001",
            "gold": [Event("R", Fraction(1, 2))],
            "predictions": {
                "staff": [Event("R", Fraction(1, 1))],
                "smt": [Event("R", Fraction(1, 2))],
                "zeus": [],
            },
        },
    ]

    report, rows = build_report(pages, expected_pages=2)

    assert len(rows) == 6
    assert [(row["page_id"], row["method"]) for row in rows] == [
        ("test_0000", "staff"),
        ("test_0000", "smt"),
        ("test_0000", "zeus"),
        ("test_0001", "staff"),
        ("test_0001", "smt"),
        ("test_0001", "zeus"),
    ]
    assert report["summary"]["staff"]["joint"]["matches"] == 1
    assert report["summary"]["smt"]["joint"]["matches"] == 1
    assert report["summary"]["zeus"]["joint"]["matches"] == 0
    assert report["summary"]["staff"]["duration"]["matches"] == 1


def test_build_report_rejects_missing_pages_or_methods() -> None:
    page = {
        "page_id": "test_0000",
        "gold": [],
        "predictions": {"staff": [], "smt": []},
    }
    with pytest.raises(ValueError, match="expected 2 pages"):
        build_report([page], expected_pages=2)
    with pytest.raises(ValueError, match="methods"):
        build_report([page], expected_pages=1)


def test_cli_can_be_invoked_directly_from_repository_root() -> None:
    root = Path(__file__).resolve().parent.parent
    result = subprocess.run(
        [sys.executable, "tools/build_transfer_error_decomposition.py", "--help"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
