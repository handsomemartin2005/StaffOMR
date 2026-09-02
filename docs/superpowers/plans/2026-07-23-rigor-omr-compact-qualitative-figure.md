# RIGOR-OMR Compact Qualitative Figure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a compact five-panel local evidence chain for Debussy `test_0006`, move the pitch--duration plot to the appendix, verify the AAAI layout, and produce a clean Overleaf ZIP.

**Architecture:** Extend the existing reproducible Matplotlib figure builder with a loader that validates one frozen full-pipeline case and a renderer that reuses one coordinate crop across all five panels. Integrate only the generated PDF into the untracked local paper package, preserve the user-edited manuscript as the source of truth, and verify compilation and packaging from independent staging directories.

**Tech Stack:** Python 3, pytest, NumPy, Pillow, Matplotlib, LaTeX/AAAI 2027, latexmk, Poppler, PowerShell, ZIP archives.

## Global Constraints

- Use only frozen artifacts from `test_0006`; never add or correct a symbol, mask, relation, or reconstruction manually.
- Select `test_0006` by the full A+B+C+D per-system Event-F1 rule and assert its score is 24.472574 within floating-point tolerance.
- Use crop `(965, 390, 1170, 515)` in the 3524-by-732 input coordinate system; expansion is allowed by at most one saved staff space per side without changing musical content.
- Display the actual chain `notehead -> stem -> beam`; never label a direct notehead--beam edge.
- Use Times New Roman for generated text and keep vector annotations in the PDF.
- Keep the five local panels in one horizontal row below approximately one quarter of the text height.
- Treat SAM2 as inspectable intermediate geometry only; do not claim an endpoint-metric improvement.
- Move `debussy_binding_gap.pdf` and its quantitative interpretation to the appendix without changing values.
- Keep figures under `paper/aaai27/figure/`, tables under `paper/aaai27/table/`, and do not create standalone analysis or limitations TeX files.
- Preserve unrelated dirty-worktree changes. Do not stage or commit the untracked `paper/aaai27` package or the already-modified plotting script unless the user separately requests it.

---

### Task 1: Lock the qualitative artifact contract with tests

**Files:**
- Create: `D:\PyCharmPojects\Staff\tests\test_build_paper_qualitative_figure.py`
- Read: `D:\PyCharmPojects\Staff\outputs\debussy_abcd_ablation\summary.json`
- Read: `D:\PyCharmPojects\Staff\outputs\debussy_abcd_ablation\A1B1C1D1\test_0006\symbols\shapes.json`
- Read: `D:\PyCharmPojects\Staff\outputs\ijcv_repro\debussy_staff_transfer24\test_0006\sam2\masks_all.json`

**Interfaces:**
- Consumes: `load_qualitative_case(root: Path = ROOT) -> dict[str, Any]` from Task 2.
- Produces: regression tests for selection score, crop, IDs, mask scores, and typed edges.

- [ ] **Step 1: Create the failing artifact-contract test**

```python
from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from tools import build_paper_evidence_figures as figures


def test_load_qualitative_case_uses_frozen_best_system() -> None:
    case = figures.load_qualitative_case()

    assert case["sample_id"] == "test_0006"
    assert case["event_f1"] == pytest.approx(24.4725738397, abs=1e-9)
    assert case["crop"] == (965, 390, 1170, 515)
    assert Image.open(case["input_path"]).size == (3524, 732)
    assert set(case["notehead_ids"]) == {"deim_000003", "deim_000013"}
    assert set(case["stem_ids"]) == {"synthetic_stem_00012", "synthetic_stem_00013"}
    assert case["beam_id"] == "deim_000069"
    assert case["mask_records"]["deim_000003"]["mask_score"] == pytest.approx(0.8905225992)
    assert case["mask_records"]["deim_000069"]["mask_score"] == pytest.approx(0.8875870109)
    assert set(case["head_stem_edges"]) == {
        ("deim_000003", "synthetic_stem_00012"),
        ("deim_000013", "synthetic_stem_00013"),
    }
    assert set(case["beam_stem_edges"]) == {
        ("deim_000069", "synthetic_stem_00012"),
        ("deim_000069", "synthetic_stem_00013"),
    }
```

- [ ] **Step 2: Run the contract test and confirm the expected failure**

```powershell
python -m pytest tests\test_build_paper_qualitative_figure.py::test_load_qualitative_case_uses_frozen_best_system -v
```

Expected: FAIL with `AttributeError: module 'tools.build_paper_evidence_figures' has no attribute 'load_qualitative_case'`.

---

### Task 2: Implement validated loading and the five-panel renderer

**Files:**
- Modify: `D:\PyCharmPojects\Staff\tools\build_paper_evidence_figures.py`
- Modify: `D:\PyCharmPojects\Staff\tests\test_build_paper_qualitative_figure.py`

**Interfaces:**
- Consumes: frozen summary, input PNG, full-pipeline shapes JSON, and SAM2 mask manifest.
- Produces: `load_qualitative_case(root: Path = ROOT) -> dict[str, Any]` and `build_qualitative_pipeline(out: Path) -> list[str]`.

- [ ] **Step 1: Add imports and immutable case constants**

```python
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch, Rectangle
from PIL import Image

QUALITATIVE_SAMPLE_ID = "test_0006"
QUALITATIVE_CROP = (965, 390, 1170, 515)
QUALITATIVE_NOTEHEAD_IDS = ("deim_000003", "deim_000013")
QUALITATIVE_STEM_IDS = ("synthetic_stem_00012", "synthetic_stem_00013")
QUALITATIVE_BEAM_ID = "deim_000069"
HEAD_COLOR = "#D55E00"
STEM_COLOR = "#7B3294"
BEAM_COLOR = "#0072B2"
STAFF_COLOR = "#009E73"
```

- [ ] **Step 2: Add the validated loader**

```python
def load_qualitative_case(root: Path = ROOT) -> dict[str, Any]:
    summary_path = root / "outputs/debussy_abcd_ablation/summary.json"
    shapes_path = root / "outputs/debussy_abcd_ablation/A1B1C1D1/test_0006/symbols/shapes.json"
    run_root = root / "outputs/ijcv_repro/debussy_staff_transfer24/test_0006"
    input_path = root / "data/ijcv_samples/debussy-omr-transfer24/test_0006.png"
    mask_manifest_path = run_root / "sam2/masks_all.json"

    summary = _json(summary_path)
    rows = summary["variants"]["A1B1C1D1"]["per_page_counts"]
    scored = [(_page_f1(row), row["page_id"]) for row in rows]
    event_f1, sample_id = max(scored, key=lambda item: item[0])
    if sample_id != QUALITATIVE_SAMPLE_ID:
        raise AssertionError(f"best full-pipeline system changed: {sample_id} ({event_f1:.6f})")
    np.testing.assert_allclose(event_f1, 24.4725738397, atol=1e-9)

    shapes = _json(shapes_path)
    symbols = {symbol["id"]: symbol for symbol in shapes["symbols"]}
    required_ids = (*QUALITATIVE_NOTEHEAD_IDS, *QUALITATIVE_STEM_IDS, QUALITATIVE_BEAM_ID)
    missing_symbols = [symbol_id for symbol_id in required_ids if symbol_id not in symbols]
    if missing_symbols:
        raise KeyError(f"missing selected symbols: {missing_symbols}")

    masks = _json(mask_manifest_path)["masks"]
    mask_records = {record["symbol_id"]: record for record in masks}
    missing_masks = [symbol_id for symbol_id in (*QUALITATIVE_NOTEHEAD_IDS, QUALITATIVE_BEAM_ID) if symbol_id not in mask_records]
    if missing_masks:
        raise KeyError(f"missing selected masks: {missing_masks}")

    head_stem_edges = {
        (relation["source"], target)
        for relation in shapes["relations"]
        if relation["type"] == "notehead_stem_attachment"
        for target in relation["targets"]
        if relation["source"] in QUALITATIVE_NOTEHEAD_IDS and target in QUALITATIVE_STEM_IDS
    }
    beam_stem_edges = {
        (relation["source"], target)
        for relation in shapes["relations"]
        if relation["type"] == "beam_stem_group"
        for target in relation["targets"]
        if relation["source"] == QUALITATIVE_BEAM_ID and target in QUALITATIVE_STEM_IDS
    }
    expected_head_edges = set(zip(QUALITATIVE_NOTEHEAD_IDS, QUALITATIVE_STEM_IDS))
    expected_beam_edges = {(QUALITATIVE_BEAM_ID, stem_id) for stem_id in QUALITATIVE_STEM_IDS}
    if head_stem_edges != expected_head_edges or beam_stem_edges != expected_beam_edges:
        raise AssertionError("selected typed relation chain no longer matches the approved case")

    return {
        "sample_id": sample_id,
        "event_f1": event_f1,
        "crop": QUALITATIVE_CROP,
        "input_path": input_path,
        "staves": shapes["staves"],
        "symbols": symbols,
        "notehead_ids": QUALITATIVE_NOTEHEAD_IDS,
        "stem_ids": QUALITATIVE_STEM_IDS,
        "beam_id": QUALITATIVE_BEAM_ID,
        "mask_records": mask_records,
        "head_stem_edges": sorted(head_stem_edges),
        "beam_stem_edges": sorted(beam_stem_edges),
    }
```

- [ ] **Step 3: Add plotting helpers and the exact renderer**

```python
def _bbox_center(symbol: Mapping[str, Any]) -> tuple[float, float]:
    x0, y0, x1, y1 = (float(value) for value in symbol["bbox"])
    return (0.5 * (x0 + x1), 0.5 * (y0 + y1))


def _show_local_background(ax: plt.Axes, image: np.ndarray, crop: tuple[int, int, int, int]) -> None:
    x0, y0, x1, y1 = crop
    ax.imshow(image)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _mask_rgba(mask_path: Path, color: str, alpha: float) -> np.ndarray:
    mask = np.asarray(Image.open(mask_path).convert("L")) > 0
    rgb = np.asarray(to_rgb(color), dtype=float)
    rgba = np.zeros((*mask.shape, 4), dtype=float)
    rgba[mask, :3] = rgb
    rgba[mask, 3] = alpha
    return rgba


def build_qualitative_pipeline(out: Path) -> list[str]:
    case = load_qualitative_case()
    image = np.asarray(Image.open(case["input_path"]).convert("RGB"))
    crop = case["crop"]
    x0, y0, x1, y1 = crop
    symbols = case["symbols"]
    selected_ids = (*case["notehead_ids"], *case["stem_ids"], case["beam_id"])
    object_colors = {
        **{symbol_id: HEAD_COLOR for symbol_id in case["notehead_ids"]},
        **{symbol_id: STEM_COLOR for symbol_id in case["stem_ids"]},
        case["beam_id"]: BEAM_COLOR,
    }

    fig, axes = plt.subplots(1, 5, figsize=(7.0, 1.52))
    titles = ("(a) Staff", "(b) Detections", "(c) Masks", "(d) Relations", "(e) Graph")
    for ax, title in zip(axes, titles):
        _show_local_background(ax, image, crop)
        ax.set_title(title, fontsize=7.2, pad=2.0)

    local_staff = min(
        case["staves"],
        key=lambda staff: abs(np.mean(staff["lines"]) - 0.5 * (y0 + y1)),
    )
    axes[0].add_patch(
        Rectangle(
            (x0, min(local_staff["lines"]) - 3),
            x1 - x0,
            max(local_staff["lines"]) - min(local_staff["lines"]) + 6,
            facecolor=STAFF_COLOR,
            edgecolor=STAFF_COLOR,
            linewidth=0.8,
            alpha=0.10,
        )
    )
    for y in local_staff["lines"]:
        axes[0].plot([x0, x1], [y, y], color=STAFF_COLOR, linewidth=0.9)

    for symbol_id in selected_ids:
        bx0, by0, bx1, by1 = symbols[symbol_id]["bbox"]
        axes[1].add_patch(
            Rectangle(
                (bx0, by0),
                bx1 - bx0,
                by1 - by0,
                facecolor="none",
                edgecolor=object_colors[symbol_id],
                linewidth=1.0,
            )
        )

    for symbol_id in (*case["notehead_ids"], case["beam_id"]):
        record = case["mask_records"][symbol_id]
        mask_path = ROOT / Path(record["mask_path"])
        axes[2].imshow(_mask_rgba(mask_path, object_colors[symbol_id], 0.52))
    axes[2].set_xlim(x0, x1)
    axes[2].set_ylim(y1, y0)

    for head_id, stem_id in case["head_stem_edges"]:
        hx, hy = _bbox_center(symbols[head_id])
        sx, sy = _bbox_center(symbols[stem_id])
        axes[3].plot([hx, sx], [hy, sy], color=HEAD_COLOR, linewidth=1.5, marker="o", markersize=2.2)
    for beam_id, stem_id in case["beam_stem_edges"]:
        bx, by = _bbox_center(symbols[beam_id])
        sx, sy = _bbox_center(symbols[stem_id])
        axes[3].plot([sx, bx], [sy, by], color=BEAM_COLOR, linewidth=1.5, marker="o", markersize=2.2)

    for image_artist in list(axes[4].images):
        image_artist.remove()
    axes[4].set_facecolor("white")
    axes[4].set_xlim(x0, x1)
    axes[4].set_ylim(y1, y0)
    for symbol_id in (*case["notehead_ids"], case["beam_id"]):
        record = case["mask_records"][symbol_id]
        mask_path = ROOT / Path(record["mask_path"])
        axes[4].imshow(_mask_rgba(mask_path, object_colors[symbol_id], 0.78))
    for stem_id in case["stem_ids"]:
        bx0, by0, bx1, by1 = symbols[stem_id]["bbox"]
        axes[4].add_patch(Rectangle((bx0, by0), bx1 - bx0, by1 - by0, color=STEM_COLOR, alpha=0.82))
    for head_id, stem_id in case["head_stem_edges"]:
        hx, hy = _bbox_center(symbols[head_id])
        sx, sy = _bbox_center(symbols[stem_id])
        axes[4].plot([hx, sx], [hy, sy], color=HEAD_COLOR, linewidth=1.4)
    for beam_id, stem_id in case["beam_stem_edges"]:
        bx, by = _bbox_center(symbols[beam_id])
        sx, sy = _bbox_center(symbols[stem_id])
        axes[4].plot([sx, bx], [sy, by], color=BEAM_COLOR, linewidth=1.4)

    handles = [
        Patch(facecolor=HEAD_COLOR, label="notehead"),
        Patch(facecolor=STEM_COLOR, label="stem"),
        Patch(facecolor=BEAM_COLOR, label="beam"),
        Line2D([], [], color=HEAD_COLOR, linewidth=1.5, label="notehead-stem"),
        Line2D([], [], color=BEAM_COLOR, linewidth=1.5, label="beam-stem"),
    ]
    fig.legend(handles=handles, frameon=False, ncol=5, loc="lower center", bbox_to_anchor=(0.5, -0.01), fontsize=6.2)
    fig.subplots_adjust(left=0.01, right=0.995, top=0.86, bottom=0.24, wspace=0.035)
    return _save(fig, out, "debussy_qualitative_pipeline")
```

- [ ] **Step 4: Register the builder in `main()`**

Add this job after `debussy_binding_gap`:

```python
(
    "debussy_qualitative_pipeline",
    build_qualitative_pipeline,
    [
        "debussy_abcd_ablation/summary.json",
        "A1B1C1D1/test_0006/symbols/shapes.json",
        "test_0006/sam2/masks_all.json",
    ],
),
```

- [ ] **Step 5: Add the render-contract test**

```python
def test_build_qualitative_pipeline_writes_compact_strip(tmp_path: Path) -> None:
    paths = [Path(path) for path in figures.build_qualitative_pipeline(tmp_path)]
    assert {path.suffix for path in paths} == {".png", ".pdf"}
    assert all(path.exists() and path.stat().st_size > 0 for path in paths)

    preview = Image.open(next(path for path in paths if path.suffix == ".png"))
    width, height = preview.size
    assert width / height >= 4.5
```

- [ ] **Step 6: Run the focused tests**

```powershell
python -m pytest tests\test_build_paper_qualitative_figure.py -v
```

Expected: 2 tests pass.

- [ ] **Step 7: Preserve dirty-worktree ownership**

```powershell
git diff --check -- tools\build_paper_evidence_figures.py tests\test_build_paper_qualitative_figure.py
git status --short -- tools\build_paper_evidence_figures.py tests\test_build_paper_qualitative_figure.py
```

Expected: only the plotting script and new focused test are listed. Do not stage or commit because the plotting script already contains uncommitted changes from the active paper work.

---

### Task 3: Replace main Figure 4 and move pitch--duration evidence to the appendix

**Files:**
- Modify: `D:\PyCharmPojects\Staff\paper\aaai27\sections\experiments.tex`
- Modify: `D:\PyCharmPojects\Staff\paper\aaai27\appendix.tex`

**Interfaces:**
- Consumes: `figure/debussy_qualitative_pipeline.pdf` and existing `figure/debussy_binding_gap.pdf`.
- Produces: main-paper Figure 4 as the compact local evidence chain and an appendix-only pitch--duration figure.

- [ ] **Step 1: Replace the current pitch--duration subsection in `experiments.tex`**

Replace the block from `\subsection{Result Visualization: Pitch--Duration Attribute Recovery}` through its figure environment with:

```tex
\subsection{Qualitative Evidence: From Staff Geometry to Typed Reconstruction}

Figure~\ref{fig:qualitative-pipeline} follows one local beam group from the highest-scoring full-pipeline Debussy system through the stored intermediate representations. The same crop is used in every panel: staff-normalized geometry localizes the notation, accepted core detections identify two noteheads, two stems, and one beam, and box-prompted masks expose their foreground support. The typed graph then connects each notehead to a stem and each stem to the beam before deterministic reconstruction. This is a composed notehead--stem--beam path; the implementation does not predict a direct notehead--beam edge.

\begin{figure*}[t]
  \centering
  \includegraphics[width=0.98\textwidth]{figure/debussy_qualitative_pipeline.pdf}
  \caption{Qualitative evidence from the Debussy system with the highest full-pipeline Event-F1 (\texttt{test\_0006}, 24.473). The same local beam group is followed through (a) staff localization, (b) accepted notehead, stem, and beam detections, (c) actual box-prompted SAM2 masks, (d) stored notehead--stem and beam--stem relations, and (e) graph-only reconstruction. Masks expose intermediate geometry; the ablation results do not establish an endpoint-metric gain from SAM2.}
  \label{fig:qualitative-pipeline}
\end{figure*}

The mask and graph views are retained for auditability rather than presented as standalone accuracy evidence. In particular, removing SAM2 does not produce a verified endpoint-metric decrease under the current ablation, whereas removing relation reasoning does.
```

- [ ] **Step 2: Add the moved quantitative interpretation and figure to `appendix.tex`**

Insert the following section after `Additional Diagnostic Results` and its oracle table, before `Polish Structural Diagnostic Ledger`:

```tex
\section{Pitch--Duration Attribute Recovery}
On Debussy, RIGOR-OMR achieves a pitch F1 of 26.119 and a duration F1 of 33.901, outperforming Zeus at 20.318 and 29.550 and SMT at 4.553 and 16.530, respectively. These results show that RIGOR-OMR recovers both core musical attributes under substantial domain shift.

We further identify 527 events with correct pitch evidence but unmatched duration and 760 events with correct duration evidence but unmatched pitch. This pool of partially recovered events provides headroom for improving joint Event-F1 through stronger attribute binding and structured decoding.

\begin{figure}[t]
  \centering
  \includegraphics[width=\linewidth]{figure/debussy_binding_gap.pdf}
  \caption{Joint, pitch-only, and duration-only Event-F1 on Debussy for the three frozen methods.}
  \label{fig:pitch-duration}
\end{figure}
```

- [ ] **Step 3: Check unique labels and evidence order**

```powershell
rg -n "qualitative-pipeline|pitch-duration|includegraphics|^\\subsection|^\\section" paper\aaai27\sections\experiments.tex paper\aaai27\appendix.tex
```

Expected: `fig:qualitative-pipeline` is defined once in the main experiment section; `fig:pitch-duration` is defined once in the appendix; the main evidence order remains comparison, ablation, qualitative visualization.

- [ ] **Step 4: Preserve the untracked manuscript package**

```powershell
git status --short -- paper\aaai27\sections\experiments.tex paper\aaai27\appendix.tex
```

Expected: both paths remain untracked package content. Do not stage or commit them.

---

### Task 4: Generate and inspect the qualitative artifact

**Files:**
- Generate temporarily: `D:\PyCharmPojects\Staff\tmp\rigor_omr_qualitative_20260723\debussy_qualitative_pipeline.png`
- Generate temporarily: `D:\PyCharmPojects\Staff\tmp\rigor_omr_qualitative_20260723\debussy_qualitative_pipeline.pdf`
- Create: `D:\PyCharmPojects\Staff\paper\aaai27\figure\debussy_qualitative_pipeline.pdf`

**Interfaces:**
- Consumes: `build_qualitative_pipeline(out: Path)`.
- Produces: the sole new paper figure and a separate PNG visual-QA preview.

- [ ] **Step 1: Generate into a temporary output directory**

```powershell
$qualitativeOut = 'D:\PyCharmPojects\Staff\tmp\rigor_omr_qualitative_20260723'
New-Item -ItemType Directory -Path $qualitativeOut -Force | Out-Null
@'
from pathlib import Path
from tools.build_paper_evidence_figures import build_qualitative_pipeline
print(build_qualitative_pipeline(Path(r"D:\PyCharmPojects\Staff\tmp\rigor_omr_qualitative_20260723")))
'@ | python -
```

Expected: one non-empty PNG and one non-empty PDF.

- [ ] **Step 2: Copy only the PDF into the paper package**

```powershell
Copy-Item -LiteralPath "$qualitativeOut\debussy_qualitative_pipeline.pdf" -Destination 'D:\PyCharmPojects\Staff\paper\aaai27\figure\debussy_qualitative_pipeline.pdf' -Force
```

- [ ] **Step 3: Verify font embedding and dimensions**

```powershell
pdffonts "$qualitativeOut\debussy_qualitative_pipeline.pdf"
pdfinfo "$qualitativeOut\debussy_qualitative_pipeline.pdf" | Select-String 'Page size|Pages'
```

Expected: all fonts are embedded; generated text resolves to Times New Roman or a Times-compatible embedded face; the page is a wide, shallow single-page figure.

- [ ] **Step 4: Visually inspect the PNG preview**

Open `D:\PyCharmPojects\Staff\tmp\rigor_omr_qualitative_20260723\debussy_qualitative_pipeline.png` and verify:

- every panel shows the same local musical group;
- staff lines align with the source image;
- boxes enclose only the selected noteheads, stems, and beam;
- masks follow the intended symbols without swallowing adjacent staff lines;
- red/orange edges are notehead--stem and blue edges are beam--stem;
- the graph panel contains no background image and no direct notehead--beam edge;
- titles and legend remain readable at two-column width.

---

### Task 5: Compile and visually verify the AAAI manuscript

**Files:**
- Generate: `D:\PyCharmPojects\Staff\paper\aaai27\main.pdf`
- Create: `D:\PyCharmPojects\Staff\paper\aaai27\RIGOR_OMR_qualitative_preview.pdf`
- Generate temporarily: `D:\PyCharmPojects\Staff\tmp\rigor_omr_qualitative_pages_20260723\page-*.png`

**Interfaces:**
- Consumes: modified TeX and the seven required paper figures.
- Produces: a verified manuscript preview with the new Figure 4 before analysis/conclusion.

- [ ] **Step 1: Compile from the local paper root**

```powershell
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Run from `D:\PyCharmPojects\Staff\paper\aaai27`.

Expected: exit code 0 and a current `main.pdf`.

- [ ] **Step 2: Check integration warnings**

```powershell
rg -n "undefined references|Reference .* undefined|not found|Emergency stop|Fatal error" main.log
```

Expected: no matches.

- [ ] **Step 3: Render the compiled manuscript**

```powershell
$pageOut = 'D:\PyCharmPojects\Staff\tmp\rigor_omr_qualitative_pages_20260723'
New-Item -ItemType Directory -Path $pageOut -Force | Out-Null
pdftoppm -png -r 170 main.pdf "$pageOut\page"
```

Expected: one PNG per manuscript page.

- [ ] **Step 4: Inspect layout and float order**

Visually inspect the page containing Figure 4 and verify:

- the strip is readable and below approximately one quarter of the text height;
- no panel, legend, or caption is clipped;
- the figure appears before `Analysis`, `Limitations`, and `Conclusion`;
- the pitch--duration plot appears only in the appendix;
- no figure floats to a page after the conclusion.

- [ ] **Step 5: Verify manuscript fonts and save the preview**

```powershell
pdffonts main.pdf
pdfinfo main.pdf | Select-String 'Pages|Page size'
Copy-Item -LiteralPath main.pdf -Destination RIGOR_OMR_qualitative_preview.pdf -Force
```

Expected: all fonts embedded, letter page size, and the preview matches the verified build.

---

### Task 6: Build and independently verify the clean Overleaf ZIP

**Files:**
- Create: `D:\PyCharmPojects\Staff\paper\aaai27\RIGOR_OMR_Overleaf_QUALITATIVE_READY.zip`
- Create temporarily: a unique staging directory under `D:\PyCharmPojects\Staff\tmp\`
- Create temporarily: a unique verification directory under `D:\PyCharmPojects\Staff\tmp\`

**Interfaces:**
- Consumes: verified local source package and required figures.
- Produces: a clean, independently compilable Overleaf archive.

- [ ] **Step 1: Create a fresh staging tree without deleting existing data**

```powershell
$stageName = 'rigor_omr_overleaf_qualitative_' + (Get-Date -Format 'yyyyMMdd_HHmmss')
$stageDir = Join-Path 'D:\PyCharmPojects\Staff\tmp' $stageName
New-Item -ItemType Directory -Path $stageDir | Out-Null
New-Item -ItemType Directory -Path (Join-Path $stageDir 'sections'),(Join-Path $stageDir 'table'),(Join-Path $stageDir 'figure') | Out-Null
```

- [ ] **Step 2: Copy the exact clean manifest**

```powershell
$paperRoot = 'D:\PyCharmPojects\Staff\paper\aaai27'
Copy-Item -LiteralPath "$paperRoot\main.tex","$paperRoot\head.tex","$paperRoot\appendix.tex","$paperRoot\reference.bib","$paperRoot\ReproducibilityChecklist.tex","$paperRoot\aaai2027.sty","$paperRoot\aaai2027.bst" -Destination $stageDir
Copy-Item -LiteralPath "$paperRoot\sections\abstract.tex","$paperRoot\sections\introduction.tex","$paperRoot\sections\related_work.tex","$paperRoot\sections\methodology.tex","$paperRoot\sections\experiments.tex","$paperRoot\sections\conclusion.tex" -Destination (Join-Path $stageDir 'sections')
Get-ChildItem -LiteralPath "$paperRoot\table" -Filter '*.tex' | Copy-Item -Destination (Join-Path $stageDir 'table')
Copy-Item -LiteralPath "$paperRoot\figure\rigor_omr_method.png","$paperRoot\figure\zero_shot_main_results.pdf","$paperRoot\figure\cross_domain_relation_forest.pdf","$paperRoot\figure\debussy_qualitative_pipeline.pdf","$paperRoot\figure\per_item_difference_small_multiples.pdf","$paperRoot\figure\polish_structure_diagnosis.pdf","$paperRoot\figure\debussy_density_stratification.pdf","$paperRoot\figure\debussy_binding_gap.pdf" -Destination (Join-Path $stageDir 'figure')
```

Expected: no ZIPs, build auxiliaries, raw masks, debug overlays, or preview PNGs in the staging tree.

- [ ] **Step 3: Compile the staged package**

```powershell
Push-Location $stageDir
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
Pop-Location
```

Expected: exit code 0 with no missing files or undefined references.

- [ ] **Step 4: Create a non-overwriting archive**

```powershell
$zipPath = 'D:\PyCharmPojects\Staff\paper\aaai27\RIGOR_OMR_Overleaf_QUALITATIVE_READY.zip'
if (Test-Path -LiteralPath $zipPath) {
    $zipPath = 'D:\PyCharmPojects\Staff\paper\aaai27\RIGOR_OMR_Overleaf_QUALITATIVE_READY_' + (Get-Date -Format 'yyyyMMdd_HHmmss') + '.zip'
}
Compress-Archive -Path (Join-Path $stageDir '*') -DestinationPath $zipPath
```

- [ ] **Step 5: Independently extract and compile the archive**

```powershell
$verifyName = 'rigor_omr_overleaf_verify_' + (Get-Date -Format 'yyyyMMdd_HHmmss')
$verifyDir = Join-Path 'D:\PyCharmPojects\Staff\tmp' $verifyName
New-Item -ItemType Directory -Path $verifyDir | Out-Null
Expand-Archive -LiteralPath $zipPath -DestinationPath $verifyDir
Push-Location $verifyDir
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
Pop-Location
```

Expected: exit code 0, no missing files, and the verified PDF has the same page count as `RIGOR_OMR_qualitative_preview.pdf`.

- [ ] **Step 6: Report the exact deliverables and changed files**

Report:

- `tools/build_paper_evidence_figures.py`;
- `tests/test_build_paper_qualitative_figure.py`;
- `paper/aaai27/sections/experiments.tex`;
- `paper/aaai27/appendix.tex`;
- `paper/aaai27/figure/debussy_qualitative_pipeline.pdf`;
- the actual non-overwriting ZIP path;
- `paper/aaai27/RIGOR_OMR_qualitative_preview.pdf`;
- focused pytest result, LaTeX compile result, page count, font check, and visual-QA outcome.
