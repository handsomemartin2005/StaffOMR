# Anonymous Code and Data Supplement

This archive supports inspection and reproduction of the RIGOR-OMR experiments. It contains the complete project Python source used for preprocessing, modular recognition, relation construction, deterministic reconstruction, evaluation, and paper-evidence generation. It also contains focused tests, compact result records, and the numerical LaTeX table inputs used by the manuscript.

## Contents

- `tools/`: Python source and run scripts. No compiled bytecode is included.
- `tests/`: regression and integrity tests.
- `paper_tables/`: numerical table bodies used by the paper.
- `result_records/`: representative JSON and JSONL summaries, with machine-specific paths removed.
- `requirements.txt`: minimal repository requirements.
- `requirements-supplement.txt`: additional packages used by the full modular pipeline.
- `MANIFEST.sha256`: SHA-256 digest of every other archived file.

## Local checks

From the archive root, run:

```text
python -m compileall -q tools
python -m pytest -q tests/test_polish_canonical_serialization.py tests/test_refine_masks_sam2_method_aligned.py
```

The remaining tests are included for inspection; some expect the large frozen image artifacts that are intentionally omitted from this size-limited archive.

The principal entry points are:

```text
python tools/run_musicxml_full_dataset_v2_1.py --help
python tools/refine_masks_sam2.py --help
python tools/extract_symbol_shapes_v2_1.py --help
python tools/assemble_v2_1_notes.py --help
python tools/export_v2_1_semantics.py --help
python tools/run_cross_dataset_relation_edge_ablation.py --help
python tools/evaluate_omr_musicxml_cer_ser_ler.py --help
python tools/build_paper_evidence_figures.py --help
```

Dataset roots, prediction caches, checkpoints, and output directories are command-line arguments. The compact records permit numerical and statistical auditing without redistributing raw score images. Full detector or mask-refiner inference additionally requires the corresponding third-party packages, public datasets, and model assets; those are excluded because of size or redistribution terms.

No external paper-repository pointer, author identity, email address, personal filesystem path, Git metadata, model weight, or raw third-party dataset is included.
