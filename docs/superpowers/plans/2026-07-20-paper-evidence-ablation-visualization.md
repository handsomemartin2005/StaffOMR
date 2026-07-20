# STAFF Paper Evidence Ablation and Visualization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce source-traceable error decomposition, relation-edge ablations, oracle diagnostics, efficiency records, and publication figures without overwriting accepted paper results.

**Architecture:** A small tested metrics library provides event parsing, multiset scores, paired bootstrap, and oracle diagnostics. Dataset-specific CLIs consume frozen STAFF/SMT/Zeus artifacts and write normalized JSONL records. A relation-edge runner filters cached Debussy relation lists and rebuilds MusicXML through the archived deterministic exporter, so the mechanism experiment does not retrain or rerun the detector. A separate plotting CLI consumes only machine-readable summaries.

**Tech Stack:** Python 3.11, standard library XML/JSON/CSV, pytest, matplotlib, psutil, PyTorch CUDA telemetry where available.

## Global Constraints

- Preserve all existing accepted JSON, MusicXML, LMX, figures, logs, and dirty worktree files.
- Write new results only below `outputs/paper_evidence_20260720/`.
- Use paired bootstrap seed `20260720` and 10,000 samples.
- Never label target-annotation oracle diagnostics as zero-shot or deployable.
- Do not exceed one GPU inference process or 15,000 MiB VRAM.
- Until the existing OLiMPiC TEDn process exits, offline analysis uses at most two CPU workers.
- Run only relation types actually present in frozen Debussy shapes: `notehead_stem_attachment`, `beam_stem_group`, and `slur_tie_notehead_endpoints`. Record absent planned types as inactive rather than fabricating variants.

---

### Task 1: Tested Event Metrics and Error Decomposition Primitives

**Files:**
- Create: `tests/test_paper_evidence_metrics.py`
- Create: `tools/paper_evidence_metrics.py`

**Interfaces:**
- Produces: `Event(pitch: str, duration: Fraction)`, `musicxml_events(path)`, `multiset_counts(gold, pred, projection)`, `aggregate_counts(rows)`, `paired_bootstrap_delta(full, ablated, samples, seed)`, and `decompose_event_errors(gold, pred)`.
- Consumed by: Tasks 2, 3, and 4.

- [ ] **Step 1: Write failing tests for joint, pitch-only, and duration-only projections**

Create synthetic events with duplicates, rests, an empty prediction, and over-prediction. Assert exact match counts, denominators, F1, and prediction/gold ratios for all three projections.

- [ ] **Step 2: Run the focused test and verify RED**

Run: `python -m pytest tests/test_paper_evidence_metrics.py -q`

Expected: collection fails because `tools.paper_evidence_metrics` does not exist.

- [ ] **Step 3: Implement the minimum metrics library**

Use immutable `Event` records and `collections.Counter`. Parse MusicXML durations as `Fraction(duration, divisions)`. Preserve rests as pitch token `R`; normalize pitched notes as `STEP<alter>@<octave>`.

- [ ] **Step 4: Run tests and verify GREEN**

Run: `python -m pytest tests/test_paper_evidence_metrics.py -q`

Expected: all Task 1 tests pass.

- [ ] **Step 5: Commit Task 1**

Run:

```powershell
git add tests/test_paper_evidence_metrics.py tools/paper_evidence_metrics.py
git commit -m "test: add paper evidence event metrics"
```

### Task 2: Debussy Error Decomposition and Reproduction Gate

**Files:**
- Create: `tests/test_build_transfer_error_decomposition.py`
- Create: `tools/build_transfer_error_decomposition.py`
- Create at runtime: `outputs/paper_evidence_20260720/error_decomposition/metrics.json`
- Create at runtime: `outputs/paper_evidence_20260720/error_decomposition/per_page.jsonl`

**Interfaces:**
- Consumes: Task 1 metrics; frozen Debussy gold MusicXML; STAFF MusicXML; SMT beKern; Zeus LMX through the already validated projection routines from `.worktrees/staff-graph-decoder/tools/evaluate_transfer_event_f1.py`.
- Produces: one normalized per-page row per method and aggregate joint/pitch/duration metrics.

- [ ] **Step 1: Write failing tests for complete page/method coverage and aggregate reconstruction**

Use a three-page temporary fixture. Assert that missing pages are explicit failures, aggregate counts equal the sum of per-page counts, and method order is deterministic.

- [ ] **Step 2: Run the focused test and verify RED**

Run: `python -m pytest tests/test_build_transfer_error_decomposition.py -q`

Expected: import or missing-function failure.

- [ ] **Step 3: Implement the Debussy dataset adapter and CLI**

Reuse the validated beKern and LMX projection semantics, but convert them into `Event` objects so joint, pitch-only, and duration-only counts share one implementation. Require exactly 24 aligned IDs.

- [ ] **Step 4: Run tests and verify GREEN**

Run: `python -m pytest tests/test_build_transfer_error_decomposition.py -q`

Expected: all Task 2 tests pass.

- [ ] **Step 5: Run the full Debussy decomposition**

Run:

```powershell
python tools/build_transfer_error_decomposition.py `
  --dataset debussy `
  --out outputs/paper_evidence_20260720/error_decomposition `
  --expected-pages 24
```

Expected: exit 0, 72 method-page rows, and joint F1 values reproducing STAFF `8.517034068136272`, SMT `2.0683020683020685`, and Zeus `7.780689377141706` within `1e-9`.

- [ ] **Step 6: Commit Task 2**

Run:

```powershell
git add tests/test_build_transfer_error_decomposition.py tools/build_transfer_error_decomposition.py
git commit -m "feat: decompose transfer event errors"
```

### Task 3: Cached Debussy Relation Edge-Family Ablation

**Files:**
- Create: `tests/test_run_relation_edge_ablation.py`
- Create: `tools/run_relation_edge_ablation.py`
- Create at runtime: `outputs/paper_evidence_20260720/relation_edge_ablation/summary.json`
- Create at runtime: `outputs/paper_evidence_20260720/relation_edge_ablation/per_page.jsonl`
- Create at runtime: `outputs/paper_evidence_20260720/relation_edge_ablation/variants/`

**Interfaces:**
- Consumes: Task 1 metrics, frozen `outputs/debussy_abcd_ablation/A1B1C1D1/*/{notes,symbols}`, and deterministic functions from `tools_py311/run_abcd_module_ablation.pyc`.
- Produces: full rebuild plus one leave-one-edge-family-out MusicXML variant for each active relation type, paired bootstrap effects, relation counts, and win/tie/loss counts.

- [ ] **Step 1: Write failing tests for relation filtering and immutable inputs**

Construct a shapes fixture containing the three active relation types. Assert that filtering one type removes only that type, preserves relation order and all non-relation fields, and does not mutate the input dictionary.

- [ ] **Step 2: Write a failing reproduction-gate test**

On a temporary minimal fixture, assert that the full relation list is passed unchanged to the semantics builder and that full-minus-full yields zero bootstrap delta.

- [ ] **Step 3: Run the focused tests and verify RED**

Run: `python -m pytest tests/test_run_relation_edge_ablation.py -q`

Expected: import or missing-function failure.

- [ ] **Step 4: Implement relation filtering, deterministic rebuild, and summary aggregation**

Discover active relation types from all 24 frozen shape files. Record planned-but-absent edge families as inactive. Rebuild full and ablated MusicXML using `build_semantics` and `semantic_to_musicxml`; never rerun DEIM or SAM2.

- [ ] **Step 5: Run tests and verify GREEN**

Run: `python -m pytest tests/test_run_relation_edge_ablation.py -q`

Expected: all Task 3 tests pass.

- [ ] **Step 6: Run one-page smoke and verify the full gate**

Run:

```powershell
python tools/run_relation_edge_ablation.py `
  --limit 1 `
  --out outputs/paper_evidence_20260720/relation_edge_ablation_smoke `
  --bootstrap-samples 1000 `
  --seed 20260720
```

Expected: the rebuilt full page has the same joint event counts as accepted A1B1C1D1 test_0000: matches `25`, predicted `291`, gold `90`.

- [ ] **Step 7: Run the full 24-page relation ablation**

Run:

```powershell
python tools/run_relation_edge_ablation.py `
  --out outputs/paper_evidence_20260720/relation_edge_ablation `
  --bootstrap-samples 10000 `
  --seed 20260720
```

Expected: full rebuild F1 equals `8.517034068136272` within `1e-9`; every active variant has 24 outputs or explicit failure rows.

- [ ] **Step 8: Commit Task 3**

Run:

```powershell
git add tests/test_run_relation_edge_ablation.py tools/run_relation_edge_ablation.py
git commit -m "feat: add relation edge family ablations"
```

### Task 4: Oracle Upper-Bound Diagnostics

**Files:**
- Create: `tests/test_paper_evidence_oracles.py`
- Create: `tools/build_oracle_diagnostics.py`
- Create at runtime: `outputs/paper_evidence_20260720/oracle_diagnostics/metrics.json`
- Create at runtime: `outputs/paper_evidence_20260720/oracle_diagnostics/per_page.jsonl`

**Interfaces:**
- Consumes: Task 1 event records and Task 2 aligned per-page inputs.
- Produces: explicitly labeled oracle-pitch, oracle-duration, and oracle-count diagnostics.

- [ ] **Step 1: Write failing oracle tests**

Cover duplicates, rests, chords, empty predictions, over-prediction, and deterministic tie-breaking. Assert that no oracle changes both pitch and duration simultaneously.

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m pytest tests/test_paper_evidence_oracles.py -q`

Expected: import or missing-function failure.

- [ ] **Step 3: Implement bounded oracle matching**

Use deterministic maximum-overlap assignments within equal-duration groups for oracle pitch and equal-pitch groups for oracle duration. Oracle count may remove excess predictions but may not modify retained identities.

- [ ] **Step 4: Run tests and verify GREEN**

Run: `python -m pytest tests/test_paper_evidence_oracles.py -q`

Expected: all Task 4 tests pass.

- [ ] **Step 5: Run Debussy oracle diagnostics**

Run:

```powershell
python tools/build_oracle_diagnostics.py `
  --dataset debussy `
  --out outputs/paper_evidence_20260720/oracle_diagnostics
```

Expected: exit 0; every output table and JSON object carries `uses_target_annotations: true`.

- [ ] **Step 6: Commit Task 4**

Run:

```powershell
git add tests/test_paper_evidence_oracles.py tools/build_oracle_diagnostics.py
git commit -m "feat: add bounded oracle diagnostics"
```

### Task 5: Publication Visualization Builder

**Files:**
- Create: `tests/test_build_paper_evidence_figures.py`
- Create: `tools/build_paper_evidence_figures.py`
- Create at runtime: `outputs/paper_evidence_20260720/figures/figure_manifest.json`
- Create at runtime: `outputs/paper_evidence_20260720/figures/*.png`
- Create at runtime: `outputs/paper_evidence_20260720/figures/*.pdf`

**Interfaces:**
- Consumes: accepted Polish/Debussy comparison JSON, Task 2 error metrics, Task 3 edge ablation, Task 4 oracle diagnostics, and frozen qualitative assets.
- Produces: paired scatter, metric decomposition, forest plot, density stratification, and fixed-rank qualitative panels.

- [ ] **Step 1: Write failing tests for deterministic selection and manifest completeness**

Assert that best/median/worst are selected by sorted numeric delta with page ID tie-breaking, every figure lists source JSON files, and plotting fails when expected page IDs are missing.

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m pytest tests/test_build_paper_evidence_figures.py -q`

Expected: import or missing-function failure.

- [ ] **Step 3: Implement the plotting CLI**

Use a color-blind-safe palette, Chinese-capable installed fonts, metric arrows, exact method names, and 300-DPI PNG plus PDF output.

- [ ] **Step 4: Run tests and verify GREEN**

Run: `python -m pytest tests/test_build_paper_evidence_figures.py -q`

Expected: all Task 5 tests pass.

- [ ] **Step 5: Generate and inspect figures**

Run:

```powershell
python tools/build_paper_evidence_figures.py `
  --out outputs/paper_evidence_20260720/figures
```

Expected: all manifest entries exist, PNG files exceed 10 KiB, PDFs are non-empty, and qualitative selections match the recorded ranks.

- [ ] **Step 6: Commit Task 5**

Run:

```powershell
git add tests/test_build_paper_evidence_figures.py tools/build_paper_evidence_figures.py
git commit -m "feat: add publication evidence figures"
```

### Task 6: Fixed-Hardware Efficiency Harness and Experiment Queue

**Files:**
- Create: `tests/test_measure_paper_method_efficiency.py`
- Create: `tools/measure_paper_method_efficiency.py`
- Create: `tools/start_paper_evidence_queue.ps1`
- Create at runtime: `outputs/paper_evidence_20260720/efficiency/metrics.json`
- Create at runtime: `outputs/paper_evidence_20260720/efficiency/runs.jsonl`
- Create at runtime: `outputs/paper_evidence_20260720/logs/`

**Interfaces:**
- Consumes: fixed commands for STAFF, SMT, and Zeus; `nvidia-smi`; process telemetry.
- Produces: timestamped run records, exit status, wall time, RAM, optional CUDA memory, parameters where inspectable, and conversion failures.

- [ ] **Step 1: Write failing tests for command records, failure preservation, and memory guards**

Use a short successful subprocess and a failing subprocess. Assert both are retained and a mocked GPU sample above 15,000 MiB terminates the child with `memory_guard_triggered: true`.

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m pytest tests/test_measure_paper_method_efficiency.py -q`

Expected: import or missing-function failure.

- [ ] **Step 3: Implement the harness and serialized queue**

Record environment and command provenance. The PowerShell queue first runs Tasks 2 and 3, then figures, then efficiency methods one at a time. It must not overwrite a successful completed run unless `-Force` is supplied.

- [ ] **Step 4: Run tests and verify GREEN**

Run: `python -m pytest tests/test_measure_paper_method_efficiency.py -q`

Expected: all Task 6 tests pass.

- [ ] **Step 5: Start the first experiment queue in a hidden background process**

Run:

```powershell
$log = 'outputs/paper_evidence_20260720/logs/queue-launch.log'
Start-Process powershell -WindowStyle Hidden -ArgumentList @(
  '-NoProfile',
  '-ExecutionPolicy', 'Bypass',
  '-File', 'tools/start_paper_evidence_queue.ps1'
) -RedirectStandardOutput $log -RedirectStandardError 'outputs/paper_evidence_20260720/logs/queue-launch.err.log' -PassThru
```

Expected: a live process ID is returned and the queue log records the current stage.

- [ ] **Step 6: Commit Task 6**

Run:

```powershell
git add tests/test_measure_paper_method_efficiency.py tools/measure_paper_method_efficiency.py tools/start_paper_evidence_queue.ps1
git commit -m "feat: add paper evidence experiment queue"
```

### Task 7: Evidence Packet Addendum and Final Audit

**Files:**
- Create: `outputs/paper_reports/STAFF_新增消融与可视化证据_20260720.md`
- Modify: `outputs/paper_reports/STAFF_论文可支撑材料总包_中文版_20260719.md`

**Interfaces:**
- Consumes: completed machine-readable results and figure manifest.
- Produces: claim-safe Chinese paper addendum with supported, unsupported, and pending results.

- [ ] **Step 1: Run all focused tests**

Run: `python -m pytest tests/test_paper_evidence_metrics.py tests/test_build_transfer_error_decomposition.py tests/test_run_relation_edge_ablation.py tests/test_paper_evidence_oracles.py tests/test_build_paper_evidence_figures.py tests/test_measure_paper_method_efficiency.py -q`

Expected: all tests pass.

- [ ] **Step 2: Verify result integrity**

Run an audit that reconstructs every aggregate from per-page JSONL, confirms all expected page IDs, checks every figure path and source, and rejects any significance wording whose interval crosses zero.

- [ ] **Step 3: Write the Chinese addendum using only completed values**

Include exact protocols, tables, confidence intervals, figure links, interpretation, limits, and pending efficiency rows. Do not invent missing baseline measurements.

- [ ] **Step 4: Commit the final evidence scripts and documentation**

Stage only the new evidence addendum and intentional packet update. Preserve unrelated dirty files.

