# RIGOR-OMR Figure Redesign and Package Synchronization

Date: 2026-07-23

## Scope and authoritative source

The authoritative manuscript is the user-edited archive:

`C:\Users\A\Downloads\RIGOR_OMR__Relation_Aware_Interpretable_Geometry_for_Zero_Shot_Cross_Domain_Optical_Music_Recognition (1).zip`

All manuscript edits start from that archive. The older local manuscript must not replace text from this version. After verification, the revised package is synchronized into `D:\PyCharmPojects\Staff\paper\aaai27` by overwriting only package files that have authoritative counterparts, while preserving unrelated local files and build artifacts.

No experimental value is inferred or fabricated. Figure data come from the frozen values already present in the manuscript and plotting script.

## Visual narrative

The main paper contains three result figures with one purpose each:

1. Performance: how much RIGOR-OMR improves over each baseline.
2. Mechanism: how much the two core relation families contribute.
3. Headroom: which event attributes are already recovered and where joint binding remains difficult.

The Polish structural analysis is a diagnostic rather than a deployable result, so it moves to the appendix.

All plots use Times New Roman, restrained colors, consistent line weights, and only annotations needed to read the result. Rasterized text is not permitted; exported PDFs must contain embedded fonts.

## Figure 2: zero-shot improvement forest

Replace the current two-panel plot with one four-row horizontal forest plot:

| Corpus | Comparison | Mean improvement | W/T/L |
|---|---|---:|---:|
| Polish | vs. SMT | +13.4 | 24/0/0 |
| Polish | vs. Zeus | +15.8 | 24/0/0 |
| Debussy | vs. SMT | +6.4 | 22/0/2 |
| Debussy | vs. Zeus | +0.7 | 15/0/9 |

- X-axis: `Event-F1 improvement over baseline (points)`.
- Points show paired mean differences; horizontal bars show paired 95% bootstrap intervals.
- A zero reference line remains.
- The right annotation combines the signed mean and wins/ties/losses without separate `Delta` or `W/T/L` headers.
- Absolute Event-F1 values remain in the table and are not duplicated in this figure.
- Proposed caption: “Paired Event-F1 improvements of RIGOR-OMR over each baseline. Points show corpus differences and bars show paired 95% bootstrap intervals; right annotations report the mean difference and per-item wins/ties/losses.”

## Figure 3: relation contribution forest

Retain a single forest plot and remove the `slur/tie` rows. Plot six rows, grouped by corpus, for only:

- notehead–stem;
- beam–stem.

Values are displayed to two decimals using the existing paired bootstrap intervals:

| Corpus | Relation | Drop | 95% interval |
|---|---|---:|---:|
| Debussy | notehead–stem | 1.47 | [0.28, 2.65] |
| Debussy | beam–stem | 1.40 | [0.21, 2.61] |
| GrandStaff | notehead–stem | 0.67 | [0.01, 1.32] |
| GrandStaff | beam–stem | 0.58 | [-0.07, 1.25] |
| OLiMPiC | notehead–stem | 1.91 | [1.47, 2.36] |
| OLiMPiC | beam–stem | 1.54 | [1.13, 1.97] |

- X-axis: `Event-F1 drop after removing relation (points)`.
- Corpus names sit outside the numeric plotting area as row-group labels.
- A zero reference line remains.
- The existing textual statement that `slur/tie` removal produced zero change remains in the experiment text.
- Proposed caption: “Event-F1 decrease after removing each relation family from frozen full-pipeline outputs. Points show corpus differences and bars show paired 95% bootstrap intervals.”

## Figure 4: pitch-duration result profile

Replace the dumbbell gap plot with a grouped point plot. The x-axis contains the methods `RIGOR-OMR`, `SMT`, and `Zeus`; each method has three consistently encoded series:

| Method | Joint | Pitch-only | Duration-only |
|---|---:|---:|---:|
| RIGOR-OMR | 8.5 | 26.1 | 33.9 |
| SMT | 2.1 | 4.6 | 16.5 |
| Zeus | 7.8 | 20.3 | 29.6 |

- Values are plotted from the existing full-precision data and labeled to one decimal.
- No derived `+17.6`, `+25.4`, or similar gap annotations appear.
- The legend contains only `Joint`, `Pitch-only`, and `Duration-only`.
- Proposed caption: “Joint, pitch-only, and duration-only Event-F1 on Debussy for the three frozen methods.”

## Appendix figure: Polish structural diagnostic ladder

Remove the Polish structural diagnostic figure from the main experiment section. Add a horizontal step/waterfall figure to the existing appendix section `Polish Structural Diagnostic Ledger`:

| Stage | SER | Change from previous stage |
|---|---:|---:|
| Canonical SER | 95.96 | — |
| Staff-constrained diagnosis | 91.75 | -4.21 |
| Page-constrained diagnosis | 85.26 | -6.49 |
| Pitch–duration rebinding | 77.20 | -8.06 |

- A distinct gray background/title states `Diagnostic lower bounds—not deployable results`.
- A small separate annotation states `Legacy to canonical: -0.18 pp`.
- The caption clearly distinguishes the deployable canonical score from privileged diagnostic lower bounds.
- Proposed caption: “Polish structural diagnostic ladder. Canonical SER is deployable; subsequent values are page-constrained diagnostic lower bounds and are not deployable system results. The inset reports the legacy-to-canonical serializer change.”
- With the current appendix ordering, this figure follows the per-item distribution figure and precedes the density plot; appendix figure numbering updates naturally.

## LaTeX integration

- Preserve the user's revised manuscript prose as the base.
- Update `sections/experiments.tex` to include only the three main figures and to refer readers to the appendix for the Polish diagnostic.
- Update `appendix.tex` to include the redesigned Polish diagnostic in the existing diagnostic section.
- Keep all tables under `table/` and all figures under `figure/`.
- Do not add standalone `analysis.tex` or `limitations.tex`; their material remains integrated into the experiment/conclusion structure already present in the user archive.
- Fix two local cross-reference defects found in the authoritative archive without changing claims:
  - add `\label{sec:factorial}` to the component-ablation subsection referenced by the methodology;
  - change the relation-edge table reference from `tab:relation-edge` to the existing label `tab:edges`.

## Reproducible generation

Modify `D:\PyCharmPojects\Staff\tools\build_paper_evidence_figures.py` so the four redesigned PDFs are reproducible from the frozen arrays already encoded there. Preserve Times New Roman configuration and export vector PDF outputs into `paper/aaai27/figure/` after synchronization.

The expected changed manuscript files are:

- `sections/experiments.tex`;
- `appendix.tex`;
- the four corresponding PDFs under `figure/`.

The reusable local plotting script is also updated. Other manuscript files remain byte-for-byte from the user archive except for build-generated files and the two explicit cross-reference fixes within `sections/experiments.tex`.

## Verification and deliverables

Verification consists of:

1. Regenerate all four redesigned figure PDFs from the script.
2. Compile the manuscript with `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`.
3. Confirm there are no undefined references, missing files, or LaTeX errors.
4. Inspect the compiled PDF pages containing Figures 2–4 and the appendix diagnostic for clipping, overlap, float spill after the conclusion, legibility at AAAI two-column size, and correct Times New Roman rendering.
5. Confirm the main-paper visual sequence is Performance → Mechanism → Headroom and the Polish diagnostic appears only in the appendix.
6. Create a new Overleaf-ready ZIP instead of overwriting the supplied archive, and create a matching preview PDF.
7. Synchronize the verified package into `D:\PyCharmPojects\Staff\paper\aaai27` non-destructively.

The ZIP contains only the clean LaTeX source package and required figures, tables, bibliography, style/class, checklist, and appendix files; it excludes build auxiliaries, temporary extraction directories, old preview PDFs, and obsolete package archives.
