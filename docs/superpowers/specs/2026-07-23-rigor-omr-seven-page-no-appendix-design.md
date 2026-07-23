# RIGOR-OMR Seven-Page No-Appendix Design

## Goal

Produce an AAAI-27 main-track manuscript with no appendix in the submission PDF, at most seven pages of non-reference content, and at most two additional reference-only pages. Preserve the complete core evidence package by compressing redundant prose and merging appendix evidence into the main paper rather than hiding it in supplementary material.

## Venue Constraint

The submission PDF may contain at most seven pages of non-reference content; pages eight and nine, if used, must contain references only. The manuscript must not place an appendix after the references. Ethical, limitation, diagnostic, and reproducibility statements needed to assess the work must remain within the seven content pages.

## Main-Paper Story

The experiment section will follow four reviewer questions:

1. **Comparison:** Does RIGOR-OMR recover more pitch--duration content under zero-shot domain shift, and does it produce evaluable structured output?
2. **Ablation:** Which modules and relation families produce the gain?
3. **Visualization:** What does the system retain from a real Debussy system, from full-system structure to local masks and typed reconstruction?
4. **Headroom and limits:** Which attributes are already recovered, which bindings remain unresolved, and which structural improvements are diagnostic rather than deployable?

The current fine-grained experiment subsections will be merged into compact blocks for setup, comparison, ablation, and qualitative/diagnostic analysis. Repeated explanations of the same F1 values, interval interpretation, output ratios, and relation conclusions will be removed.

## Figure Set

The seven-page body will retain five core figures:

1. RIGOR-OMR method overview.
2. Zero-shot paired improvement forest plot.
3. Cross-domain relation-ablation forest plot.
4. A composite `test_0006` qualitative figure.
5. Debussy joint, pitch-only, and duration-only recovery figure.

Figure 4 will use the real stored `test_0006` artifact `outputs/ijcv_repro/debussy_staff_transfer24/test_0006/visuals/v2_1_skeleton_overlay.png`, not the clipboard copy. Its top row will show the complete system-level structural overlay and mark the approved crop `(965, 390, 1170, 515)`. Its bottom row will retain the existing five local stages: staff localization, accepted symbols, SAM2 masks, typed relations, and graph-only reconstruction. The caption will state that the two stems are geometry-derived, that the graph contains notehead--stem and beam--stem edges rather than a direct notehead--beam predictor, and that SAM2 supplies inspectable intermediate geometry without a verified endpoint-metric gain.

The per-item small-multiples plot, density-tertile plot, and Polish diagnostic waterfall will not remain as separate figures because their essential information is already represented by the main forest plot, compact prose, or the new structural table.

## Ablation Compression

Staff geometry (A) and symbol detection (B) will be introduced as a non-separable perception gate before the compact factorial table. A assigns detected objects to staff-normalized coordinates needed for pitch and grouping, while B supplies the symbol instances. If either is absent, the pipeline cannot form a valid pitch--duration event. Consequently, the 12 configurations with `A=0` or `B=0` are structurally equivalent zero-output cases and can be summarized in one row without discarding a nonzero result.

The main ablation table will contain five rows:

- `AB+C+D`;
- `AB+D`;
- `AB+C`;
- `AB`;
- `A=0 or B=0` (12 configurations, all zero).

It will retain precision, recall, Event-F1, and predicted/gold ratio. The prose will state the paired interval for removing D and the null interval for removing C once, rather than repeating every value in several paragraphs.

## Appendix-to-Main Evidence Mapping

| Former appendix material | Main-paper treatment |
| --- | --- |
| Complete 16-row factorial table | Losslessly collapsed into the five-row A/B-gated factorial table, including predicted/gold ratios. |
| Per-item zero-shot differences | Removed as a redundant figure; the main paired forest plot retains corpus differences, confidence intervals, and wins/ties/losses. |
| Pitch/duration recovery and partial-match counts | Moved into the main text with `debussy_binding_gap.pdf`; retain RIGOR-OMR/SMT/Zeus marginal F1 values and the 527 pitch-correct/duration-wrong and 760 duration-correct/pitch-wrong counts. |
| Polish structural diagnostic ledger | Condensed to a four-row main-text table: canonical SER 95.9589, staff-constrained 91.7498, page-constrained 85.2584, and pitch--duration rebinding 77.1959. The latter three rows remain explicitly labeled privileged diagnostic lower bounds. Legacy-to-canonical change of 0.1778 points is stated in prose. Low-coverage R3 and invalid cross-page matching are omitted from the primary table but their exclusion is explained. |
| Complete relation-effect scores | Main relation table retains all seven ablations, full and ablated Event-F1, differences, and 95% intervals. |
| Relation inventory | Moved into one compact reproducibility sentence reporting edge/page counts for Debussy, GrandStaff, and OLiMPiC, together with 10,000 bootstrap resamples, seed 20260720, and 200/200 completion gates. |
| Density tertiles | Removed as exploratory, non-confirmatory evidence without confidence intervals. |
| Oracle-diagnostics table | Replaced by the validated marginal pitch/duration projection figure and partial-match counts; no target-annotation oracle is presented as deployable performance. |

## Table Set

The main paper will use compact tables for:

1. zero-shot Polish/Debussy comparison;
2. strict OLiMPiC evaluation;
3. the collapsed A/B-gated factorial ablation;
4. complete relation-family ablations;
5. the four-row Polish structural diagnostic ladder.

The standalone dataset-role table may be removed if page pressure remains; its sample counts and roles will be retained in the setup paragraph. Table captions will carry definitions that otherwise require separate prose.

## File and Package Changes

The implementation will revise:

- `tools/build_paper_evidence_figures.py`;
- `tests/test_build_paper_qualitative_figure.py`;
- `paper/aaai27/main.tex`;
- `paper/aaai27/sections/experiments.tex`;
- `paper/aaai27/sections/conclusion.tex` if compression is required;
- `paper/aaai27/table/factorial_summary.tex`;
- `paper/aaai27/table/relation_effects.tex`;
- a compact Polish diagnostic table file under `paper/aaai27/table/`;
- `paper/aaai27/figure/debussy_qualitative_pipeline.pdf`.

The submission package will delete `appendix.tex` and exclude unused appendix-only tables and figures. `main.tex` will end the content after the conclusion and then invoke the bibliography. The source package will contain only files reachable from `main.tex`, plus the required style, bibliography, and reproducibility-checklist sources.

## Verification Gates

The revision is acceptable only when all of the following hold:

- the focused qualitative-figure tests pass;
- `latexmk -g -pdf -interaction=nonstopmode -halt-on-error main.tex` succeeds;
- pages 1--7 contain all non-reference content and no content section begins or continues on page 8;
- any pages after page 7 contain references only;
- the PDF contains no appendix heading, appendix figure/table labels, or references to an appendix;
- all figure/table references resolve and the final log has no undefined citations or references;
- Figure 4 is readable at AAAI double-column width and uses embedded Times New Roman text;
- the final PDF contains no unembedded or Type 3 fonts;
- a clean ZIP contains only reachable submission files and independently compiles to the same page structure.

