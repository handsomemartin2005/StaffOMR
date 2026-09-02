# RIGOR-OMR Qualitative Result Figure

Date: 2026-07-23

## Relationship to the existing figure design

This document supersedes only the main-paper Figure 4 portion of
`2026-07-23-rigor-omr-figure-redesign-design.md`. Figures 2 and 3 and the
appendix Polish diagnostic remain unchanged.

The main experiment sequence becomes:

1. Performance: paired zero-shot improvements over the baselines.
2. Mechanism: relation-family ablations across domains.
3. Qualitative evidence: staff localization, detections, masks, and typed
   relation reconstruction on one local group from a real best-performing
   system.

The existing pitch--duration point plot moves to the appendix so that the main
paper still contains three focused result figures.

## Evidence source and selection rule

Use only artifacts produced by the existing full A+B+C+D pipeline. No symbol,
mask, relation, or reconstruction may be manually added or corrected.

The selected system is Debussy `test_0006`, because it has the highest
full-pipeline per-system Event-F1 among the 24 Debussy transfer systems:
24.473. The figure caption states this selection rule so the example is not
presented as a representative or randomly sampled case.

The selected local group contains the following accepted full-pipeline
artifacts:

- noteheads `deim_000003` and `deim_000013`;
- synthetic stems `synthetic_stem_00012` and `synthetic_stem_00013`;
- beam `deim_000069`;
- notehead SAM2 mask score 0.891 for `deim_000003`;
- beam SAM2 mask score 0.888 for `deim_000069`;
- actual `notehead_stem_attachment` and `beam_stem_group` edges.

Every stage uses the same source crop, approximately `(965, 390, 1170, 515)`
in the 3524-by-732 input coordinate system. The crop may be expanded by at most
one staff space on any side for label clearance, but its musical content must
not change.

The implementation does not contain a direct notehead--beam edge. The visual
must therefore show the true composition
`notehead -> stem -> beam` and must not label it as a direct head--beam
relation.

## Figure layout

Create one compact two-column `figure*` PDF containing five equal-width local
panels in a single horizontal row. Do not include a full-system overview. The
panels reuse the same crop and coordinate system so readers can follow the
evidence without repeatedly relocating the symbols.

### Panel (a): local staff localization

- Show the selected local image crop.
- Draw the stored five staff-line coordinates that cross the crop and a light
  staff-band boundary derived from the saved staff geometry.
- Do not add a full-page locator inset or repeat the unannotated input as a
  separate panel.

### Panel (b): accepted core detections

- Reuse the crop from panel (a).
- Draw boxes only for the selected accepted noteheads, synthetic stems, and
  beam.
- Do not print class names, IDs, or confidence values inside the panel.
- Keep box strokes subordinate to the notation.

### Panel (c): real mask evidence

- Show the selected local image crop.
- Overlay the actual box-prompted SAM2 masks for the two noteheads and beam.
- Use translucent fills so the original symbol boundaries remain visible.
- If the prompt box is shown, use one thin dashed outline and explain it in the
  legend rather than placing text inside the crop.
- This panel demonstrates intermediate geometric inspectability only. It must
  not imply that SAM2 improves the endpoint Event-F1.

### Panel (d): typed relations on the image

- Reuse exactly the same crop and coordinate system as panels (a)--(c).
- Draw actual notehead--stem edges in one color and beam--stem edges in a
  second color.
- Draw small node markers only where needed to distinguish the two edge
  families.
- Do not include unrelated relations or the cluttered full debug overlay.

### Panel (e): graph-only reconstruction

- Remove the image background and redraw only the selected detected objects
  and their actual typed edges in normalized local coordinates.
- Preserve the same left-to-right geometry as panels (a)--(d).
- Label the two edge types once in a compact legend.
- This panel is a visualization of the stored relation graph, not a manually
  corrected score and not a direct notehead--beam predictor.

## Visual language

- Use Times New Roman for every generated label, legend, and panel marker.
- Use restrained, colorblind-safe colors with one consistent mapping across
  panels (b)--(e).
- Keep the raster score image grayscale and use vector annotations in the PDF.
- Avoid gradients, shadows, heavy borders, and per-object text labels.
- Keep the five-panel strip below approximately one quarter of the text height,
  including panel titles and legend.
- Use at most one decimal place for the selection score if it appears in the
  figure; exact values remain in the caption or prose.

## Manuscript integration

In `sections/experiments.tex`:

- replace the current pitch--duration Figure 4 environment with the new
  qualitative `figure*`;
- rename the visualization subsection to describe qualitative intermediate
  evidence and typed reconstruction;
- add a short paragraph explaining what panels (a)--(e) show;
- retain the existing statement that SAM2 has no verified end-metric benefit;
- do not claim perfect segmentation or deployment-ready transcription.

In `appendix.tex`:

- add the existing `debussy_binding_gap.pdf` as an appendix figure near the
  additional diagnostic results;
- retain its current quantitative interpretation and do not change any result
  values.

All figure files remain under `paper/aaai27/figure/`; all tables remain under
`paper/aaai27/table/`. No standalone analysis or limitations TeX file is added.

## Proposed caption

"Qualitative evidence from the Debussy system with the highest full-pipeline
Event-F1 (`test_0006`, 24.473). The same local beam group is followed through
(a) staff localization, (b) accepted notehead, stem, and beam detections,
(c) actual box-prompted SAM2 masks, (d) stored notehead--stem and beam--stem
relations, and (e) graph-only reconstruction. Masks expose intermediate
geometry; the ablation results do not establish an endpoint-metric gain from
SAM2."

The LaTeX version will use typographic dashes and escaped punctuation as
required.

## Reproducible generation and deliverables

Extend `tools/build_paper_evidence_figures.py` with a qualitative-figure builder
that reads the real input image, `masks_all.json`, and the full A+B+C+D
`shapes.json`. The builder validates the selected IDs and relation types before
rendering. It writes:

- `paper/aaai27/figure/debussy_qualitative_pipeline.pdf` for LaTeX;
- an optional PNG preview outside the clean Overleaf package.

The final Overleaf package includes the generated PDF but excludes raw masks,
full debug overlays, temporary contact sheets, build auxiliaries, old ZIPs, and
preview-only images.

## Verification

1. Assert that `test_0006` remains the highest full-pipeline Debussy system in
   the frozen summary used for selection.
2. Assert that all selected node IDs exist and the two displayed edge types are
   present in `shapes.json`.
3. Assert that the displayed mask files and input image exist and share the
   expected coordinate system.
4. Generate the PDF twice and verify deterministic output geometry.
5. Compile the AAAI manuscript without missing files or undefined references.
6. Render the figure page and inspect box density, mask alignment, relation-edge
   visibility, Times New Roman text, clipping, strip height, and two-column
   legibility.
7. Confirm the qualitative figure remains before the analysis/conclusion and
   the pitch--duration plot appears only in the appendix.
8. Rebuild and independently compile a clean Overleaf ZIP.
