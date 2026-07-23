# RIGOR-OMR Qualitative Result Figure

Date: 2026-07-23

## Relationship to the existing figure design

This document supersedes only the main-paper Figure 4 portion of
`2026-07-23-rigor-omr-figure-redesign-design.md`. Figures 2 and 3 and the
appendix Polish diagnostic remain unchanged.

The main experiment sequence becomes:

1. Performance: paired zero-shot improvements over the baselines.
2. Mechanism: relation-family ablations across domains.
3. Qualitative evidence: detections, masks, and typed relation reconstruction
   on one real best-performing system.

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

The implementation does not contain a direct notehead--beam edge. The visual
must therefore show the true composition
`notehead -> stem -> beam` and must not label it as a direct head--beam
relation.

## Figure layout

Create one two-column `figure*` PDF with a wide overview above three aligned
local panels.

### Panel (a): accepted detection overview

- Show the complete `test_0006` system.
- Draw boxes only for accepted core objects used by the structured pipeline:
  filled/open noteheads, stems, and beams.
- Do not print a class name or confidence value beside every box.
- Use a single callout rectangle to identify the local group used in panels
  (b)--(d).
- Keep the score image legible; box strokes must remain subordinate to the
  notation.

### Panel (b): real mask evidence

- Show the selected local image crop.
- Overlay the actual box-prompted SAM2 masks for the two noteheads and beam.
- Use translucent fills so the original symbol boundaries remain visible.
- If the prompt box is shown, use one thin dashed outline and explain it in the
  legend rather than placing text inside the crop.
- This panel demonstrates intermediate geometric inspectability only. It must
  not imply that SAM2 improves the endpoint Event-F1.

### Panel (c): typed relations on the image

- Reuse exactly the same crop and coordinate system as panel (b).
- Draw actual notehead--stem edges in one color and beam--stem edges in a
  second color.
- Draw small node markers only where needed to distinguish the two edge
  families.
- Do not include unrelated relations or the cluttered full debug overlay.

### Panel (d): graph-only reconstruction

- Remove the image background and redraw only the selected detected objects
  and their actual typed edges in normalized local coordinates.
- Preserve the same left-to-right geometry as panels (b) and (c).
- Label the two edge types once in a compact legend.
- This panel is a visualization of the stored relation graph, not a manually
  corrected score and not a direct notehead--beam predictor.

## Visual language

- Use Times New Roman for every generated label, legend, and panel marker.
- Use restrained, colorblind-safe colors with one consistent mapping across
  panels (b)--(d).
- Keep the raster score image grayscale and use vector annotations in the PDF.
- Avoid gradients, shadows, heavy borders, and per-object text labels.
- Use at most one decimal place for the selection score if it appears in the
  figure; exact values remain in the caption or prose.

## Manuscript integration

In `sections/experiments.tex`:

- replace the current pitch--duration Figure 4 environment with the new
  qualitative `figure*`;
- rename the visualization subsection to describe qualitative intermediate
  evidence and typed reconstruction;
- add a short paragraph explaining what panels (a)--(d) show;
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
Event-F1 (`test_0006`, 24.473). (a) Accepted notehead, stem, and beam detections,
with the selected local group highlighted. (b) Actual box-prompted SAM2 masks
on the selected noteheads and beam. (c) Stored notehead--stem and beam--stem
relations over the same crop. (d) The corresponding graph-only reconstruction.
Masks expose intermediate geometry; the ablation results do not establish an
endpoint-metric gain from SAM2."

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
   visibility, Times New Roman text, clipping, and two-column legibility.
7. Confirm the qualitative figure remains before the analysis/conclusion and
   the pitch--duration plot appears only in the appendix.
8. Rebuild and independently compile a clean Overleaf ZIP.
