from pathlib import Path
import subprocess
import sys

from tools.run_cross_dataset_relation_edge_ablation import dataset_config, run_dataset_ablation


def test_dataset_config_resolves_all_frozen_pages() -> None:
    for name in ("grandstaff", "olimpic"):
        config = dataset_config(name)
        assert len(config["page_ids"]) == 200
        first = config["page_ids"][0]
        assert (config["source_root"] / first / "notes" / "notes.json").exists()
        assert (config["gold_root"] / f"{first}.musicxml").exists()


def test_one_page_smoke_reproduces_frozen_full_output(tmp_path: Path) -> None:
    payload = run_dataset_ablation(
        "grandstaff", tmp_path / "grandstaff", limit=1, bootstrap_samples=10, seed=7
    )
    assert payload["pages"] == 1
    assert payload["reproduction_gate"]["passed"] is True
    assert set(payload["variants"]) == {
        "full",
        "without_notehead_stem_attachment",
        "without_beam_stem_group",
    }


def test_cli_can_be_invoked_directly() -> None:
    root = Path(__file__).resolve().parent.parent
    result = subprocess.run(
        [sys.executable, "tools/run_cross_dataset_relation_edge_ablation.py", "--help"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
