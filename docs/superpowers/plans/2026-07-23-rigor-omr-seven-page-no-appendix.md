# RIGOR-OMR Seven-Page No-Appendix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert the current AAAI-27 manuscript into a self-contained seven-page main paper with no appendix, while preserving the complete core comparison, ablation, qualitative, diagnostic, and reproducibility evidence.

**Architecture:** Collapse structurally redundant factorial rows, merge appendix evidence into compact main-text tables/figures/sentences, and extend the existing `test_0006` renderer into a full-system-plus-local composite. Iterate LaTeX compression against rendered page boundaries, then package only reachable sources.

**Tech Stack:** Python 3.11, Matplotlib/Pillow/NumPy, pytest, AAAI-27 LaTeX, latexmk, Poppler.

## Global Constraints

- Pages 1--7 may contain non-reference content; pages 8--9, if present, are references only.
- The PDF and ZIP contain no appendix.
- A and B are explained as a non-separable perception gate before their 12 zero-output configurations are collapsed.
- All four valid `AB x C x D` configurations remain in the main paper.
- Figure text uses embedded Times New Roman.
- No experimental number or claim may be invented or strengthened.

---

### Task 1: Extend the qualitative-figure contract

**Files:**
- Modify: `tests/test_build_paper_qualitative_figure.py`
- Modify: `tools/build_paper_evidence_figures.py`

**Interfaces:**
- Consumes: `load_qualitative_case(root: Path = ROOT) -> dict[str, Any]`.
- Produces: a case dictionary containing the real full-system overlay path and a composite `build_qualitative_pipeline(out: Path) -> list[str]` artifact.

- [ ] **Step 1: Add failing assertions**

```python
case = figures.load_qualitative_case()
assert case["overview_path"].name == "v2_1_skeleton_overlay.png"
assert case["overview_path"].exists()
assert case["crop"] == (965, 390, 1170, 515)
```

Also assert that the generated PNG is wider than tall, contains enough height for two rows, and its PDF embeds Times New Roman.

- [ ] **Step 2: Run the focused suite and confirm the new assertion fails**

Run: `$env:PYTHONPATH='.'; python -m pytest -q tests/test_build_paper_qualitative_figure.py`

Expected: failure because `overview_path` is absent.

- [ ] **Step 3: Implement the composite renderer**

Add the stored overview path to `load_qualitative_case`. Render the full overlay across the top row with the exact crop rectangle, and retain the five approved local panels below. Preserve real masks, symbol IDs, and typed edges.

- [ ] **Step 4: Regenerate and inspect**

Generate PNG/PDF under `tmp/rigor_omr_qualitative_20260723`, copy only the PDF to `paper/aaai27/figure`, run the focused tests, inspect the PNG, and check `pdffonts`.

---

### Task 2: Merge appendix tables into the main evidence package

**Files:**
- Modify: `paper/aaai27/table/factorial_summary.tex`
- Modify: `paper/aaai27/table/relation_effects.tex`
- Create: `paper/aaai27/table/polish_diagnostics_compact.tex`

**Interfaces:**
- Consumes: existing appendix tables and validated numeric artifacts.
- Produces: five-row factorial, seven-row complete relation ablation, and four-row structural diagnosis tables.

- [ ] **Step 1: Expand the collapsed factorial table**

Retain precision, recall, Event-F1, and Pred./Gold for `AB+C+D`, `AB+D`, `AB+C`, `AB`, plus one zero row labeled as 12 configurations missing A or B.

- [ ] **Step 2: Make the relation table complete**

Use columns `Target`, `Removed`, `Full`, `Ablated`, `Delta [95% CI]` for the seven existing ablations. Keep consistent three-decimal precision and bold only intervals excluding zero.

- [ ] **Step 3: Add the compact Polish ladder**

Use four rows for canonical, staff-constrained, page-constrained, and rebinding SER. Label privileged rows as diagnostic lower bounds in the caption and table body.

- [ ] **Step 4: Run static table checks**

Run `rg -n "hline|\\|Appendix|tab:.*full" paper/aaai27/table paper/aaai27/sections/experiments.tex` and inspect all matches. Expected: no vertical rules, no appendix references, no full-table labels reachable from the manuscript.

---

### Task 3: Rewrite and compress the experiment section

**Files:**
- Modify: `paper/aaai27/sections/experiments.tex`
- Modify: `paper/aaai27/sections/conclusion.tex`
- Modify: `paper/aaai27/main.tex`

**Interfaces:**
- Consumes: the five main figures and five compact tables.
- Produces: setup, comparison, ablation, qualitative/headroom, analysis/limitations flow with no appendix references.

- [ ] **Step 1: Compress setup and comparison**

Fold dataset roles into setup and remove `table/datasets.tex` from the main flow. State the metric/protocol once. Compress result narration to claim plus strongest evidence instead of restating every table cell.

- [ ] **Step 2: Explain and report the A/B gate**

Before the factorial table, explain that A supplies staff-normalized assignment and B supplies symbol instances; without either, no valid event can be formed. State that 12 zero-output rows are therefore collapsed without loss.

- [ ] **Step 3: Merge relation evidence**

Keep the cross-domain forest and complete compact relation table. Add one reproducibility sentence with edge/page inventories, 10,000 resamples, seed 20260720, and 200/200 completion gates.

- [ ] **Step 4: Merge qualitative and attribute evidence**

Insert the composite `test_0006` figure and `debussy_binding_gap.pdf`. Retain the marginal scores and the 527/760 partial-binding counts, while preserving the SAM2 null-result boundary.

- [ ] **Step 5: Merge structural diagnosis and limitations**

Insert the compact Polish table, state the 0.1778 serializer change, explain the exclusion of low-coverage R3 and invalid cross-page matching, then end with concise limitations/responsible use.

- [ ] **Step 6: Remove the appendix from the document graph**

Delete `\input{appendix}` from `main.tex`; bibliography becomes the final document content.

---

### Task 4: Enforce the seven-page content boundary

**Files:**
- Modify as required: `paper/aaai27/sections/*.tex`, `paper/aaai27/table/*.tex`
- Generate: `paper/aaai27/main.pdf`
- Generate: `paper/aaai27/RIGOR_OMR_7PAGE_preview.pdf`

**Interfaces:**
- Consumes: revised manuscript sources.
- Produces: an AAAI-27 PDF with content ending on page 7 and references only thereafter.

- [ ] **Step 1: Force a clean build**

Run: `latexmk -g -pdf -interaction=nonstopmode -halt-on-error main.tex`.

- [ ] **Step 2: Check the content/reference boundary**

Use `pdftotext -f 7 -l 9 -layout` and rendered page PNGs to confirm the conclusion/limitations end by page 7 and page 8 onward contains references only.

- [ ] **Step 3: Compress iteratively if needed**

Cut repeated motivation, repeated numeric narration, and redundant transitions first. Do not reduce template font sizes or margins and do not remove any protected evidence from the design.

- [ ] **Step 4: Inspect all pages visually**

Render the final PDF at 150 dpi. Check float order, table width, caption legibility, clipped content, and the composite Figure 4.

- [ ] **Step 5: Validate fonts and logs**

Expected: no undefined citations/references, no missing files, no Type 3 or unembedded fonts, and at most nine total pages.

---

### Task 5: Delete appendix-only artifacts and package Overleaf sources

**Files:**
- Delete: `paper/aaai27/appendix.tex`
- Delete/exclude: appendix-only tables and figures not reachable from `main.tex`
- Create: `paper/aaai27/RIGOR_OMR_AAAI27_7PAGE_NO_APPENDIX.zip`

**Interfaces:**
- Consumes: the verified reachable source graph.
- Produces: a clean, independently compilable Overleaf ZIP.

- [ ] **Step 1: Resolve reachable files**

Build an explicit whitelist from `main.tex`, section inputs, table inputs, bibliography, style files, and referenced figures.

- [ ] **Step 2: Remove exact appendix-only local artifacts**

Remove only `appendix.tex` and the obsolete appendix-only table/figure files identified by the whitelist. Preserve unrelated worktree files.

- [ ] **Step 3: Create a clean ZIP**

Stage the whitelist in a new timestamped directory and compress its contents. Exclude PDFs generated from `main.tex`, auxiliaries, previews, old ZIPs, raw masks, and debug files.

- [ ] **Step 4: Independently verify**

Extract the ZIP into another new directory, compile from zero, and repeat page-boundary, log, font, and manifest checks.

- [ ] **Step 5: Run final focused regression**

Run: `$env:PYTHONPATH='.'; python -m pytest -q tests/test_build_paper_qualitative_figure.py`.

Expected: all focused tests pass. Record the known unrelated `select_ranked_pages` collection failure separately.

---

### Task 6: Fill the seventh content page with retained evidence

**Files:**
- Restore: `paper/aaai27/figure/per_item_difference_small_multiples.pdf`
- Create: `paper/aaai27/table/datasets.tex`
- Create: `paper/aaai27/table/relation_inventory.tex`
- Modify: `paper/aaai27/sections/experiments.tex`
- Modify: `paper/aaai27/main.tex`
- Regenerate: `paper/aaai27/RIGOR_OMR_AAAI27_7PAGE_preview.pdf`
- Regenerate: `paper/aaai27/RIGOR_OMR_AAAI27_7PAGE_NO_APPENDIX.zip`

**Interfaces:**
- Consumes: validated former-appendix figure/table assets and the compiled seven-page draft.
- Produces: a nine-page submission PDF whose first seven pages contain the full main text and whose pages eight and nine contain references only.

- [ ] **Step 1: Restore only the approved evidence assets**

Extract `figure/per_item_difference_small_multiples.pdf` from the last qualitative-ready source package. Recreate the compact dataset-role and relation-inventory tables from their validated former-appendix values. Do not restore the density-tertile figure, Polish waterfall, appendix driver, or unused full tables.

- [ ] **Step 2: Integrate the restored evidence into Experiments**

Place the dataset table in Protocol and Baselines, the per-item distribution after the aggregate comparison figure, and the relation inventory after the complete relation ablation. Replace the dense inventory sentence with a reproducibility sentence. Add count-correct values `11.118/5.562/9.234`, Polish R3 `95.9293` at `141/5002` coverage, and the invalid cross-page mathematical reference `57.0793` in concise diagnostic prose.

- [ ] **Step 3: Enforce the content/reference boundary**

Insert `\clearpage` immediately before `\bibliography{reference}` in `paper/aaai27/main.tex`. Compile with:

```powershell
latexmk -g -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Expected: at most nine pages; Conclusion ends on page 7; References begins on page 8; pages 8--9 contain no non-reference section.

- [ ] **Step 4: Tune density without weakening legibility**

If the conclusion ends before page 7, enlarge neither decorative figures nor fonts. Restore supported protocol/diagnostic explanation or adjust the per-item figure width within the AAAI columns. If content spills to page 8, first remove repeated numeric narration and transitions. Do not alter template margins or body font size.

- [ ] **Step 5: Run visual and static verification**

Render pages 6--9 with `pdftoppm`, inspect float order and whitespace, and run `pdffonts`. Search the final log for undefined references/citations, missing files, and overfull boxes. Expected: no such issues, no Type 3 fonts, and all fonts embedded.

- [ ] **Step 6: Rebuild and independently compile the Overleaf ZIP**

Stage only files reachable from `main.tex`, plus the style, bibliography, and reproducibility checklist sources. Include the new figure and two tables; exclude Appendix, density/Polish diagnostic figures, auxiliaries, previews, and old archives. Extract into a fresh temporary directory and compile from zero. Expected: the extracted package reproduces the same nine-page boundary.

---

### Task 7: Rebalance pages 6--7 and restore the wide qualitative figure

**Files:**
- Modify: `tests/test_build_paper_qualitative_figure.py`
- Modify: `tools/build_paper_evidence_figures.py`
- Modify: `paper/aaai27/head.tex`
- Modify: `paper/aaai27/sections/experiments.tex`
- Regenerate: `paper/aaai27/figure/debussy_qualitative_pipeline.pdf`
- Regenerate: `paper/aaai27/RIGOR_OMR_AAAI27_7PAGE_preview.pdf`

**Interfaces:**
- Consumes: the six validated result figures and the current eight-page PDF.
- Produces: balanced pages 6--7, with two full-width quantitative figures on page 6 and the approved full-width `test_0006` landscape figure on page 7.

- [ ] **Step 1: Restore the landscape qualitative-figure contract**

Change the focused test to require a generated preview ratio between `2.0` and `3.5` and height at least `500` pixels. Run:

```powershell
$env:PYTHONPATH='.'; python -m pytest -q tests/test_build_paper_qualitative_figure.py::test_build_qualitative_pipeline_writes_overview_and_local_stages
```

Expected: FAIL because the current single-column preview ratio is approximately 1.0.

- [ ] **Step 2: Restore the approved full-width renderer**

Use a `7.0 x 2.78` inch figure with the full-system overlay across the first row and five aligned local panels across the second row. Keep the existing crop, masks, typed relations, graph content, and Times New Roman font settings. Run the focused test suite and expect `2 passed`.

- [ ] **Step 3: Group the two quantitative visualizations on page 6**

Replace the independent `figure*` environments for the per-item plot and relation forest with one full-width float-page block containing two full-width minipages. Each minipage keeps its own `\captionof{figure}` and label, so Figure 3 and Figure 4 remain separately numbered. Flush this grouped block before the qualitative subsection.

- [ ] **Step 4: Place the qualitative visualization deterministically on page 7**

Load `cuted` and `balance` in `paper/aaai27/head.tex`. Place the landscape `test_0006` image in a non-floating `strip` block at the top of the final content page. Continue with binding analysis, Polish diagnostics, limitations, and conclusion in balanced columns. Keep `\clearpage` before the bibliography.

- [ ] **Step 5: Compile and inspect pages 5--8**

Run:

```powershell
latexmk -g -pdf -interaction=nonstopmode -halt-on-error main.tex
pdftoppm -f 5 -l 8 -png -r 150 main.pdf tmp/pdfs/final-layout/page
```

Expected: page 6 contains the two full-width quantitative visualizations without empty columns; page 7 contains the wide qualitative figure and balanced final prose; Conclusion is on page 7; References begins on page 8.

- [ ] **Step 6: Verify without rebuilding the ZIP**

Run the focused qualitative tests, inspect `main.log`, and run `pdffonts main.pdf`. Expected: no undefined references/citations, missing files, or overfull boxes; no Type 3 or unembedded fonts. Copy `main.pdf` to `RIGOR_OMR_AAAI27_7PAGE_preview.pdf`. Do not create or modify any ZIP archive.

---

### Task 8: Replace forced float pages with standard AAAI flow

**Files:**
- Modify: `paper/aaai27/head.tex`
- Modify: `paper/aaai27/main.tex`
- Modify: `paper/aaai27/sections/experiments.tex`
- Regenerate: `paper/aaai27/RIGOR_OMR_AAAI27_7PAGE_preview.pdf`

**Interfaces:**
- Consumes: the validated landscape qualitative figure and all existing experiment prose, tables, labels, and numeric results.
- Produces: standard two-column AAAI flow with no float-only pages or large page-5 gap.

- [ ] **Step 1: Record the failing layout evidence**

Render pages 5--7 and extract their text. Confirm that page 5 ends immediately after Figure 2, page 6 contains only Figure 3/4, and page 7 is a `strip`-composed page. Search `experiments.tex` for `figure*[p]`, `\FloatBarrier`, and `strip` to identify the causal controls.

- [ ] **Step 2: Restore standard float environments**

Remove `cuted` from `head.tex`. Replace the grouped `[p]` block with separate `figure*[!t]` environments for Figure 3 and Figure 4. Replace the qualitative `strip` with `figure*[!t]`, and make the binding plot a one-column `figure[!t]`. Leave all labels and figure files unchanged.

- [ ] **Step 3: Restore standard section ownership**

Move `\input{sections/conclusion}` back to `main.tex` after Experiments. Remove that input from the right-column minipage and dissolve both diagnostic minipages into ordinary prose, figure, table, Limitations, and Conclusion flow.

- [ ] **Step 4: Compile and render pages 5--8**

Run `latexmk -g -pdf -interaction=nonstopmode -halt-on-error main.tex`, then render pages 5--8. Expected: page 5 continues with experiment prose, Figures 3/4 appear as top-aligned full-width figures, Figure 5 spans both columns at the top of page 7, the remaining content flows in normal columns, and References begins on page 8.

- [ ] **Step 5: Run final static and regression gates**

Run `$env:PYTHONPATH='.'; pytest tests/test_build_paper_qualitative_figure.py -q`, inspect the log for overfull/undefined diagnostics, and run `pdffonts`. Expected: `2 passed`, no blocking LaTeX diagnostics, and no Type 3 or unembedded fonts. Update only the preview PDF; do not create or modify a ZIP.
