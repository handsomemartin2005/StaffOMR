from tools.build_paper_evidence_figures import select_ranked_pages


def test_select_ranked_pages_is_deterministic_with_page_id_tie_breaking() -> None:
    rows = [
        {"page_id": "p3", "delta": -1.0},
        {"page_id": "p2", "delta": 2.0},
        {"page_id": "p1", "delta": 2.0},
        {"page_id": "p4", "delta": 0.5},
        {"page_id": "p5", "delta": 0.0},
    ]
    selected = select_ranked_pages(rows, value_key="delta")
    assert selected["worst"]["page_id"] == "p3"
    assert selected["median"]["page_id"] == "p4"
    assert selected["best"]["page_id"] == "p2"
