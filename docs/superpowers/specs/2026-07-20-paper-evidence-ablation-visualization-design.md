# Paper Evidence Ablation and Visualization Design

Date: 2026-07-20

## Goal

Complete the minimum reviewer-critical evidence that is still missing from the STAFF paper: decompose the relation graph into interpretable edge families, identify whether transfer errors arise from pitch, duration, count, or global structure, measure deployment cost on a fixed transfer subset, and produce publication-ready visualizations that connect each claim to real outputs.

## Existing Evidence That Must Not Be Repeated

The following experiments are already complete and remain immutable inputs:

- zero-shot STAFF/SMT/Zeus Event-F1 on Polish test24 and Debussy transfer24;
- unified OLiMPiC test200 Event-F1, SER, SER-no-tuples, and TEDn;
- Debussy 16-way A/B/C/D factorial ablation;
- full-stack relation-graph D ablations on GrandStaff, OLiMPiC, and Polish;
- DEIM, RT-DETR, and YOLO detector replacement;
- SAM2 prompt ablation;
- relation-graph parameter sensitivity;
- existing qualitative Debussy pipeline visualization.

New experiments must add a distinct reviewer-facing answer rather than another threshold sweep.

## Central Claims and Missing Evidence

| Claim | Existing support | Missing evidence addressed here |
|---|---|---|
| Explicit relation reasoning improves transfer | Full D on/off ablations | Which edge family creates the gain |
| STAFF recovers content but struggles with strict structure | Event-F1 and OLiMPiC SER/TEDn | Pitch-duration-structure error decomposition |
| STAFF is inspectable | Intermediate outputs exist | Representative best/median/worst panels with matched quantitative labels |
| Modular deployment may be practical | Design-level argument only | Fixed-hardware runtime and memory measurements |

## Experimental Units

### E1: Relation Edge-Family Ablation

Use the frozen Debussy transfer24 inputs and the accepted A1B1C1D1 policy. The frozen shapes inventory contains 4,613 edges across three active relation families:

1. `notehead_stem_attachment`: 3,900 edges on 24/24 pages;
2. `beam_stem_group`: 585 edges on 23/24 pages;
3. `slur_tie_notehead_endpoints`: 128 edges on 21/24 pages.

Ledger-line--notehead and accidental--note were planned relation families but are absent from the frozen Debussy relation lists. They are recorded as inactive and are not presented as executed ablations.

Variants:

- full relation graph;
- remove one edge family at a time;
- no relation graph control, reusing the existing A1B1C1D0 result.

Primary metric: order-invariant duration-pitch Event-F1.  
Secondary metrics: Precision, Recall, prediction/gold ratio, per-page win/tie/loss count.  
Statistics: paired bootstrap with 10,000 samples and seed 20260720.

The full variant must reproduce the existing Debussy F1 within 1e-9 when it reuses cached outputs or within 0.01 points when it reruns deterministic conversion code. A variant is considered paper-relevant when its full-minus-ablated confidence interval excludes zero or when it exposes a consistent semantic failure supported by qualitative examples.

Expansion gate: only edge families with a Debussy absolute effect of at least 0.25 F1 or a confidence interval excluding zero are evaluated on the larger GrandStaff/OLiMPiC sets. This prevents expensive cross-dataset runs for inactive relations.

### E2: Transfer Error Decomposition

Use existing prediction and gold files; do not rerun neural inference.

For each method and dataset, compute:

- pitch-only multiset Precision/Recall/F1;
- duration-only multiset Precision/Recall/F1;
- joint duration-pitch Event-F1;
- prediction/gold count ratio;
- exact-event matches;
- pitch-correct but duration-wrong opportunities;
- duration-correct but pitch-wrong opportunities.

For OLiMPiC LMX, additionally align predicted and gold token streams and report edit mass for:

- pitch tokens;
- duration tokens;
- structural tokens including barlines, voices, tuplets, and grouping symbols;
- insertions;
- deletions;
- substitutions that cannot be assigned to one semantic class.

All category definitions must be deterministic and stored in the output JSON. The decomposition is diagnostic and must not replace official SER/TEDn.

### E3: Oracle Upper-Bound Diagnostics

Oracle experiments use target annotations only to locate bottlenecks and are never reported as deployable method results.

Compute three bounded diagnostics from aligned event multisets or sequences:

- oracle pitch: preserve predicted event count and durations while substituting the best-aligned gold pitches;
- oracle duration: preserve predicted event count and pitches while substituting the best-aligned gold durations;
- oracle count: cap insertion/deletion mass to the gold event count without changing the remaining predicted pitch-duration identities.

Each oracle must state exactly what information is replaced and what remains predicted. Results appear in a separate table labeled `Oracle diagnostic; uses target annotations`.

The implementation must include synthetic unit cases for duplicates, rests, chords, empty predictions, and over-prediction before it is run on paper datasets.

### E4: Fixed-Hardware Efficiency Measurement

Measure STAFF, SMT, and Zeus on the same Debussy transfer24 images where the runnable environment permits. Run one warm-up image followed by the fixed 24-image subset.

Record:

- total wall-clock time;
- median and interquartile per-image time;
- peak process RAM;
- peak CUDA allocated and reserved memory when applicable;
- model parameter count when it can be obtained from the loaded checkpoint;
- conversion failures.

Hardware record: NVIDIA GeForce RTX 5070 Ti, 16,303 MiB reported VRAM. Each method runs alone. GPU memory must remain below 15,000 MiB. A method that cannot share the same runtime is reported with its own environment and marked non-comparable for absolute wall-clock time.

This experiment supports inference-cost and operational evidence only. It does not support adaptation-cost claims unless training time, trainable parameters, and target-label budgets are measured separately.

### E5: Publication Visualizations

Generate the following figures only from machine-readable results:

1. paired per-page STAFF versus SMT/Zeus scatter plots for Polish and Debussy, with the identity line and win/tie/loss counts;
2. pitch-only, duration-only, and joint Event-F1 grouped bars;
3. relation edge-family forest plot showing full-minus-ablated effect and 95% confidence intervals;
4. error-mass composition for OLiMPiC, separating pitch, duration, structure, insertion, and deletion;
5. score-density or gold-event-count stratification with fixed quantile bins defined jointly before method comparison;
6. Debussy best, median, and worst relation-graph panels, each showing input, detections/relations, reconstruction, F1 with D, F1 without D, and the page delta;
7. OLiMPiC representative failure panels selected by predeclared quantiles rather than visual preference.

All plots use Chinese and English-safe fonts, vector-friendly dimensions, color-blind-safe colors, metric-direction arrows, and source JSON paths in a figure manifest. PNG is required; PDF or SVG is additionally produced where the plotting backend supports it.

## Execution Order and Gates

### Stage 1: Offline evidence

Run E2 and the non-oracle portions of E5 from existing cached predictions. This stage is CPU-only and may run while the existing OLiMPiC TEDn process completes.

Gate: every aggregate value must be exactly reproducible from its per-page or per-token records.

### Stage 2: Debussy mechanism evidence

Implement edge-family switches, validate them on one Debussy page, and run E1 across transfer24. GPU runs must be serialized; do not exceed one inference process.

Gate: full graph reproduction passes and each variant produces 24 aligned outputs or explicit failure records.

### Stage 3: Oracle diagnostics and efficiency

Implement unit-tested E3, then measure E4 one method at a time. If a baseline environment is not runnable, retain the failure reason rather than estimating cost.

Gate: oracle labels are isolated from all deployable result tables; efficiency logs contain hardware, command, start/end time, and exit status.

### Stage 4: Selective cross-dataset confirmation

Expand only active relation families from E1. Do not tune a family on OLiMPiC or Polish test labels. Use the frozen Debussy decision and report all expanded results.

## Resource and Concurrency Policy

- GPU: at most one STAFF/SMT/Zeus inference process; abort before 15,000 MiB VRAM.
- CPU: reserve capacity for the already-running OLiMPiC TEDn process; new offline analysis uses at most two workers until it finishes.
- RAM: stream page/token records rather than loading all shape JSON files simultaneously.
- Existing dirty worktrees and outputs are preserved. New files use new directories and never overwrite accepted evidence.

## Output Layout

All new artifacts are written under:

```text
outputs/paper_evidence_20260720/
  manifest.json
  logs/
  error_decomposition/
    metrics.json
    per_page.jsonl
  oracle_diagnostics/
    metrics.json
    per_page.jsonl
  relation_edge_ablation/
    summary.json
    per_page.jsonl
    variants/
  efficiency/
    metrics.json
    runs.jsonl
  figures/
    figure_manifest.json
```

The manifest records source files, commands, git commit, Python executable, seeds, sample IDs, and completion state.

## Validation and Failure Handling

- Never silently drop a page. Missing outputs become explicit failures.
- Never replace failed baseline outputs with reported paper numbers.
- Never select qualitative examples by subjective appearance; use fixed rank rules.
- Never call an oracle result zero-shot or deployable.
- Never call a difference significant unless the 95% paired interval excludes zero.
- Every plotting script must fail if required rows or expected sample IDs are missing.
- Existing accepted result JSON files remain read-only inputs.

## Deliverables

1. executable experiment plan with exact commands and expected outputs;
2. source-controlled analysis and visualization scripts with tests;
3. running logs for the first experiment queue;
4. machine-readable summaries and publication-ready figures;
5. an addendum to the Chinese paper evidence packet stating which new claims are supported, unsupported, or still pending.
