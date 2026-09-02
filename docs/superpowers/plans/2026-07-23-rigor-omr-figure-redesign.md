# RIGOR-OMR Figure Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Revise the user-edited RIGOR-OMR manuscript so its experiment section reads comparison → ablation → visualization, contains three focused main-paper figures, moves the Polish structural diagnostic to the appendix, and ships as a verified Overleaf ZIP synchronized to the local paper directory.

**Architecture:** Treat the `(1).zip` archive as the immutable source of manuscript prose, generate all result figures through the existing Matplotlib script, and edit only the experiment/appendix integration required by the approved design. Build and visually inspect an isolated extracted copy first, then synchronize only the authoritative package files and verified figures into `paper/aaai27` without deleting unrelated local content.

**Tech Stack:** Python 3, Matplotlib, NumPy, LaTeX/AAAI class, latexmk, Poppler PDF rendering, PowerShell, ZIP archives.

## Global Constraints

- The authoritative input is `C:\Users\A\Downloads\RIGOR_OMR__Relation_Aware_Interpretable_Geometry_for_Zero_Shot_Cross_Domain_Optical_Music_Recognition (1).zip`.
- Preserve user-edited prose and never replace it with the older local manuscript.
- Order experiment evidence as comparison → ablation → visualization.
- Use Times New Roman in every regenerated figure and keep vector PDF text.
- Use only frozen values already present in the manuscript and plotting script.
- Keep tables under `table/`, figures under `figure/`, and do not add standalone `analysis.tex` or `limitations.tex`.
- Synchronize non-destructively into `D:\PyCharmPojects\Staff\paper\aaai27`.
- Create a new ZIP and preview PDF; do not overwrite the supplied archive.

---

### Task 1: Establish the authoritative working copy and acceptance checks

**Files:**
- Read: `C:\Users\A\Downloads\RIGOR_OMR__Relation_Aware_Interpretable_Geometry_for_Zero_Shot_Cross_Domain_Optical_Music_Recognition (1).zip`
- Modify in place: `D:\PyCharmPojects\Staff\tmp\rigor_omr_user_base_20260723\`

**Interfaces:**
- Consumes: the user-edited archive.
- Produces: an isolated manuscript tree whose source manifest is the basis for packaging and local synchronization.

- [ ] **Step 1: Verify the extraction corresponds to the supplied archive**

```powershell
$zip = 'C:\Users\A\Downloads\RIGOR_OMR__Relation_Aware_Interpretable_Geometry_for_Zero_Shot_Cross_Domain_Optical_Music_Recognition (1).zip'
$work = 'D:\PyCharmPojects\Staff\tmp\rigor_omr_user_base_20260723'
Get-FileHash -Algorithm SHA256 -LiteralPath $zip
Get-ChildItem -LiteralPath $work -Recurse -File | ForEach-Object { $_.FullName.Substring($work.Length + 1) } | Sort-Object
```

Expected: the archive exists, the extracted tree contains `main.tex`, `head.tex`, `appendix.tex`, `reference.bib`, `sections/`, `table/`, and `figure/`, and no path escapes the working directory.

- [ ] **Step 2: Compile the untouched user version to preserve a baseline log**

```powershell
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Expected: compilation completes; the only known source warnings are unresolved `sec:factorial` and `tab:relation-edge` references.

### Task 2: Redesign the four result figures reproducibly

**Files:**
- Modify: `D:\PyCharmPojects\Staff\tools\build_paper_evidence_figures.py`
- Generate: `D:\PyCharmPojects\Staff\tmp\rigor_omr_user_base_20260723\figure\zero_shot_main_results.pdf`
- Generate: `D:\PyCharmPojects\Staff\tmp\rigor_omr_user_base_20260723\figure\cross_domain_relation_forest.pdf`
- Generate: `D:\PyCharmPojects\Staff\tmp\rigor_omr_user_base_20260723\figure\debussy_binding_gap.pdf`
- Generate: `D:\PyCharmPojects\Staff\tmp\rigor_omr_user_base_20260723\figure\polish_structure_diagnosis.pdf`

**Interfaces:**
- Consumes: the existing frozen NumPy arrays inside each `build_*` function and an output `Path`.
- Produces: each `build_*` function returns the file paths written by `save_both`, including a vector PDF used by LaTeX.

- [ ] **Step 1: Add pre-render value assertions**

Inside the relevant functions, assert the exact display-driving arrays before plotting:

```python
np.testing.assert_allclose(np.round(diff, 3), [13.352, 15.751, 6.449, 0.736], atol=1e-9)
np.testing.assert_array_equal(wtl, [[24, 0, 0], [24, 0, 0], [22, 0, 2], [15, 0, 9]])
np.testing.assert_allclose(np.round(effect, 3), [1.470, 1.403, 0.669, 0.583, 1.911, 1.543], atol=1e-9)
np.testing.assert_allclose(
    np.round(scores, 3),
    [[8.517, 26.119, 33.901], [2.068, 4.553, 16.530], [7.781, 20.318, 29.550]],
    atol=1e-9,
)
np.testing.assert_allclose(np.round(ser, 2), [95.96, 91.75, 85.26, 77.20], atol=1e-9)
```

- [ ] **Step 2: Run the script and verify the assertions initially expose any mismatched source ordering**

```powershell
python tools\build_paper_evidence_figures.py --out tmp\rigor_omr_user_base_20260723\figure
```

Expected before the rewrite: at least the zero-shot/relation arrays require reordering or filtering to match the approved row order, or the command writes the older plot geometry that fails the visual acceptance criteria.

- [ ] **Step 3: Replace `build_zero_shot_summary` with the four-row forest encoding**

Implement one axis with rows `Polish vs. SMT`, `Polish vs. Zeus`, `Debussy vs. SMT`, `Debussy vs. Zeus`, a zero line, paired 95% intervals, and right annotations formatted by:

```python
label = f"{value:+.1f}   {wins}/{ties}/{losses}"
ax.set_xlabel("Event-F1 improvement over baseline (points)")
```

Do not retain the absolute-F1 panel or `Delta`/`W/T/L` column headers.

- [ ] **Step 4: Replace `build_relation_forest` with the six-row relation encoding**

Use two relations for each of Debussy, GrandStaff, and OLiMPiC. Format right annotations as:

```python
label = f"{value:.2f} [{lo:.2f}, {hi:.2f}]"
ax.set_xlabel("Event-F1 drop after removing relation (points)")
```

Remove slur/tie from the plot while retaining the zero reference line.

- [ ] **Step 5: Replace `build_binding_gap` with the grouped point plot**

Use x positions for `RIGOR-OMR`, `SMT`, and `Zeus`, offsets for `Joint`, `Pitch-only`, and `Duration-only`, and annotate every point with one decimal:

```python
ax.annotate(f"{value:.1f}", (x, value), xytext=(0, 5), textcoords="offset points", ha="center")
```

Do not show marginal-minus-joint gap labels.

- [ ] **Step 6: Replace `build_polish_structure_diagnosis` with a diagnostic ladder**

Render four horizontal stages with values `95.96`, `91.75`, `85.26`, and `77.20`, show successive drops `-4.21`, `-6.49`, and `-8.06`, and add separate text:

```python
ax.set_title("Diagnostic lower bounds—not deployable results")
ax.text(..., "Legacy to canonical: -0.18 pp", ...)
```

Use a light gray background to distinguish diagnostic lower bounds.

- [ ] **Step 7: Generate and validate the figure artifacts**

```powershell
python tools\build_paper_evidence_figures.py --out tmp\rigor_omr_user_base_20260723\figure
Get-Item tmp\rigor_omr_user_base_20260723\figure\zero_shot_main_results.pdf,tmp\rigor_omr_user_base_20260723\figure\cross_domain_relation_forest.pdf,tmp\rigor_omr_user_base_20260723\figure\debussy_binding_gap.pdf,tmp\rigor_omr_user_base_20260723\figure\polish_structure_diagnosis.pdf | Select-Object Name,Length
pdffonts tmp\rigor_omr_user_base_20260723\figure\zero_shot_main_results.pdf
```

Expected: four non-empty PDFs; `pdffonts` reports embedded fonts and includes Times New Roman or its system-resolved Times face.

### Task 3: Reorder and simplify the main experiment narrative

**Files:**
- Modify: `D:\PyCharmPojects\Staff\tmp\rigor_omr_user_base_20260723\sections\experiments.tex`

**Interfaces:**
- Consumes: the three redesigned main figures and existing result tables.
- Produces: experiment order comparison → ablation → visualization, with valid labels and no Polish diagnostic float after the conclusion.

- [ ] **Step 1: Fix the two existing cross-reference defects**

Apply these exact source changes:

```tex
\subsection{Component Ablation: Relation Reasoning Drives the Main Gain}
\label{sec:factorial}
```

and:

```tex
Table~\ref{tab:edges}
```

- [ ] **Step 2: Replace the Figure 2 caption and width**

```tex
\begin{figure*}[t]
  \centering
  \includegraphics[width=0.86\textwidth]{figure/zero_shot_main_results.pdf}
  \caption{Paired Event-F1 improvements of RIGOR-OMR over each baseline. Points show corpus differences and bars show paired 95\% bootstrap intervals; right annotations report the mean difference and per-item wins/ties/losses.}
  \label{fig:zero-shot-summary}
\end{figure*}
```

- [ ] **Step 3: Replace the Figure 3 caption and keep it in the ablation block**

```tex
\begin{figure*}[t]
  \centering
  \includegraphics[width=0.86\textwidth]{figure/cross_domain_relation_forest.pdf}
  \caption{Event-F1 decrease after removing each relation family from frozen full-pipeline outputs. Points show corpus differences and bars show paired 95\% bootstrap intervals.}
  \label{fig:edge-forest}
\end{figure*}
```

- [ ] **Step 4: Replace the Figure 4 caption in the visualization block**

```tex
\begin{figure}[t]
  \centering
  \includegraphics[width=\linewidth]{figure/debussy_binding_gap.pdf}
  \caption{Joint, pitch-only, and duration-only Event-F1 on Debussy for the three frozen methods.}
  \label{fig:pitch-duration}
\end{figure}
```

- [ ] **Step 5: Remove the Polish figure environment from the main paper and add an appendix pointer**

Retain the structural-diagnosis prose and replace its figure environment with:

```tex
Appendix Figure~\ref{fig:polish-structure} visualizes this diagnostic ladder and Appendix Table~\ref{tab:polish-diagnostics} provides the complete ledger.
```

- [ ] **Step 6: Verify subsection evidence order**

```powershell
rg -n '^\\subsection|^\\subsubsection|includegraphics' tmp\rigor_omr_user_base_20260723\sections\experiments.tex
```

Expected: protocol/setup; comparison (`Zero-Shot`, `Strict Structured`); ablation (`Component`, `Which Relations`); visualization (`Pitch--Duration`); analysis; limitations. Only three `includegraphics` entries remain in the main experiment file.

### Task 4: Integrate the Polish diagnostic in the appendix

**Files:**
- Modify: `D:\PyCharmPojects\Staff\tmp\rigor_omr_user_base_20260723\appendix.tex`

**Interfaces:**
- Consumes: `figure/polish_structure_diagnosis.pdf`.
- Produces: appendix-only Figure `fig:polish-structure`, placed beside the existing Polish diagnostic ledger.

- [ ] **Step 1: Insert the diagnostic figure after the Polish ledger table**

```tex
\begin{figure*}[t]
  \centering
  \includegraphics[width=0.86\textwidth]{figure/polish_structure_diagnosis.pdf}
  \caption{Polish structural diagnostic ladder. Canonical SER is deployable; subsequent values are page-constrained diagnostic lower bounds and are not deployable system results. The inset reports the legacy-to-canonical serializer change.}
  \label{fig:polish-structure}
\end{figure*}
```

- [ ] **Step 2: Verify unique labels and appendix ordering**

```powershell
rg -n 'fig:polish-structure|includegraphics|setcounter\{figure\}' tmp\rigor_omr_user_base_20260723\appendix.tex tmp\rigor_omr_user_base_20260723\sections\experiments.tex
```

Expected: the label is defined once in `appendix.tex`; the main experiment file contains only a reference; the diagnostic follows the per-item plot and precedes the density visualization.

### Task 5: Compile and visually verify the revised manuscript

**Files:**
- Generate: `D:\PyCharmPojects\Staff\tmp\rigor_omr_user_base_20260723\main.pdf`
- Generate temporarily: rendered PNG pages under `D:\PyCharmPojects\Staff\tmp\rigor_omr_visual_qa\`

**Interfaces:**
- Consumes: the modified LaTeX and four generated PDFs.
- Produces: a clean compiled preview and visual QA evidence.

- [ ] **Step 1: Perform a clean LaTeX build**

```powershell
latexmk -C
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Expected: exit code 0 and a current `main.pdf`.

- [ ] **Step 2: Check the log for broken integration**

```powershell
rg -n 'LaTeX Warning: Reference|undefined references|not found|Emergency stop|Fatal error|Overfull \\hbox' main.log
```

Expected: no undefined/missing/fatal errors; any remaining overfull box must be inspected and corrected if it affects the modified figure/caption regions.

- [ ] **Step 3: Render the PDF and inspect every figure page**

```powershell
pdftoppm -png -r 150 main.pdf D:\PyCharmPojects\Staff\tmp\rigor_omr_visual_qa\page
```

Expected: Figures 2–4 are legible at two-column scale; annotations do not clip or overlap; the Polish diagnostic appears only in the appendix; no result figure floats after the conclusion; the visual order is Performance → Mechanism → Headroom.

- [ ] **Step 4: Inspect embedded fonts in the compiled manuscript**

```powershell
pdffonts main.pdf
```

Expected: all fonts are embedded; regenerated plot text resolves to Times New Roman/Times-compatible embedded faces.

### Task 6: Synchronize locally and package for Overleaf

**Files:**
- Modify matching package files under: `D:\PyCharmPojects\Staff\paper\aaai27\`
- Create: `D:\PyCharmPojects\Staff\paper\aaai27\RIGOR_OMR_Overleaf_figures_revised.zip`
- Create: `D:\PyCharmPojects\Staff\paper\aaai27\RIGOR_OMR_figures_revised_preview.pdf`

**Interfaces:**
- Consumes: the verified authoritative working tree.
- Produces: the updated local manuscript, a clean Overleaf ZIP, and matching preview PDF.

- [ ] **Step 1: Copy only authoritative package files into the local paper tree**

For every file listed by the input ZIP, copy its verified counterpart from the working tree to the same relative path under `paper\aaai27`. Also copy the four regenerated figure PDFs. Do not delete local-only files.

- [ ] **Step 2: Recompile from the synchronized local tree**

```powershell
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Expected: exit code 0 with the same figure placement and no undefined references.

- [ ] **Step 3: Build a clean package manifest**

Include source `.tex`, `reference.bib`, AAAI class/style/bibliography support, checklist, and required files under `sections/`, `table/`, and `figure/`. Exclude `.aux`, `.bbl`, `.blg`, `.fdb_latexmk`, `.fls`, `.log`, `.out`, `.synctex.gz`, temporary directories, previous ZIPs, and preview PDFs.

- [ ] **Step 4: Create the ZIP and preview without overwriting an existing deliverable**

```powershell
Copy-Item -LiteralPath main.pdf -Destination RIGOR_OMR_figures_revised_preview.pdf
$stage = 'D:\PyCharmPojects\Staff\tmp\rigor_omr_overleaf_stage_20260723'
Compress-Archive -Path "$stage\*" -DestinationPath RIGOR_OMR_Overleaf_figures_revised.zip
```

If either destination exists, append `_20260723` before the extension.

- [ ] **Step 5: Verify the deliverable archive independently**

Extract the new ZIP into a fresh temporary directory, run:

```powershell
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Expected: exit code 0, no missing files, no undefined references, and a PDF matching the synchronized preview page count.

- [ ] **Step 6: Report exact changed files and artifact paths**

List the plotting script, modified TeX files, regenerated figure PDFs, synchronized local root, Overleaf ZIP, preview PDF, compile result, page count, and font/visual QA outcome.
