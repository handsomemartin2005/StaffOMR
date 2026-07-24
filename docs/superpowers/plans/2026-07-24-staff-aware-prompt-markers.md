# Staff-aware Prompt Markers Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Show positive, retained staff-line negative, and suppressed negative prompts in panel (b) of the `test_0006` qualitative pipeline PDF.

**Architecture:** Reconstruct prompt candidates through the same `build_prompt_points` and detector-support filter used by the SAM2 refiner, then expose three marker groups to the figure builder. Use the accepted beam for suppressed points and an overlapping detector hypothesis in the same crop for retained negative points; do not change inference, selection, or downstream panels.

**Tech Stack:** Python 3.11, NumPy, Matplotlib, Pillow, pytest, Poppler.

## Global Constraints

- Keep `test_0006`, crop `(965, 390, 1170, 515)`, and panels (a) and (c)-(f) unchanged.
- Use only genuine prompt coordinates produced by the current staff-aware prompt implementation.
- Green filled circle means `positive prompt`.
- Red filled `X` means `staff-line negative` passed to SAM2.
- Red circled `x` means `suppressed negative` excluded by detector-support filtering.
- Remove the prose annotation about three suppressed beam negatives.
- Do not alter the running full experiment or its safe-residual logic.

---

### Task 1: Reconstruct and verify prompt marker groups

**Files:**
- Modify: `tests/test_build_staff_aware_qualitative_pipeline.py`
- Modify: `tools/build_staff_aware_qualitative_pipeline.py`

**Interfaces:**
- Consumes: `build_prompt_points(symbol, staves, image_size, "box_pos_neg_staff")` and `filter_negative_points_outside_support(coords, labels, detector_box)` from `tools.refine_masks_sam2.py`.
- Produces: `prompt_marker_groups(symbol, staves, image_size) -> dict[str, list[list[float]]]` with keys `positive`, `retained_negative`, and `suppressed_negative`; `load_case()` exposes groups for `deim_000069` and `deim_000278` as `prompt_markers`.

- [ ] **Step 1: Write the failing data-integrity test**

```python
def test_prompt_marker_groups_include_retained_and_suppressed_negatives() -> None:
    case = figure.load_case()
    accepted = case["prompt_markers"]["deim_000069"]
    retained = case["prompt_markers"]["deim_000278"]

    assert len(accepted["positive"]) == 3
    assert accepted["retained_negative"] == []
    assert len(accepted["suppressed_negative"]) == 3
    assert len(retained["positive"]) == 3
    assert len(retained["retained_negative"]) == 3
    assert retained["suppressed_negative"] == []
    x0, y0, x1, y1 = case["crop"]
    assert all(x0 <= x <= x1 and y0 <= y <= y1 for x, y in retained["retained_negative"])
```

- [ ] **Step 2: Run the focused test and verify failure**

Run: `python -m pytest tests/test_build_staff_aware_qualitative_pipeline.py::test_prompt_marker_groups_include_retained_and_suppressed_negatives -q`

Expected: FAIL because `load_case()` does not expose `prompt_markers`.

- [ ] **Step 3: Implement exact marker reconstruction**

In `tools/build_staff_aware_qualitative_pipeline.py`, import the SAM2 refiner in both module and direct-script execution contexts and add:

```python
try:
    from tools import refine_masks_sam2 as sam2_refiner
except ModuleNotFoundError:
    import refine_masks_sam2 as sam2_refiner

PROMPT_BEAM_IDS = (BEAM_ID, "deim_000278")


def prompt_marker_groups(
    symbol: Mapping[str, Any],
    staves: list[dict[str, Any]],
    image_size: tuple[int, int],
) -> dict[str, list[list[float]]]:
    coords, labels, _ = sam2_refiner.build_prompt_points(
        symbol, staves, image_size, "box_pos_neg_staff"
    )
    _, metadata = sam2_refiner.adaptive_prompt_box(symbol, image_size)
    filtered_coords, filtered_labels, removed = sam2_refiner.filter_negative_points_outside_support(
        coords, labels, metadata["detector_box"]
    )
    coords_array = np.asarray(coords, dtype=np.float32)
    labels_array = np.asarray(labels, dtype=np.int32)
    retained = np.asarray(filtered_coords, dtype=np.float32)[np.asarray(filtered_labels) == 0]
    x0, y0, x1, y1 = map(float, metadata["detector_box"])
    suppressed_mask = (
        (labels_array == 0)
        & (coords_array[:, 0] >= x0)
        & (coords_array[:, 0] <= x1)
        & (coords_array[:, 1] >= y0)
        & (coords_array[:, 1] <= y1)
    )
    suppressed = coords_array[suppressed_mask]
    if len(suppressed) != removed:
        raise AssertionError("suppressed prompt reconstruction disagrees with the SAM2 filter")
    return {
        "positive": coords_array[labels_array == 1].tolist(),
        "retained_negative": retained.tolist(),
        "suppressed_negative": suppressed.tolist(),
    }
```

Load the source detector symbols from `masks_payload["symbols_json"]`, build groups for both IDs, and assert their retained/suppressed counts match the mask manifest.

- [ ] **Step 4: Run the focused test and full figure tests**

Run: `python -m pytest tests/test_build_staff_aware_qualitative_pipeline.py -q`

Expected: all tests PASS.

- [ ] **Step 5: Commit the reconstruction and test**

```powershell
git add tests/test_build_staff_aware_qualitative_pipeline.py tools/build_staff_aware_qualitative_pipeline.py
git commit -m "test: verify staff-aware prompt marker groups"
```

---

### Task 2: Draw the three prompt categories and regenerate the PDF

**Files:**
- Modify: `tools/build_staff_aware_qualitative_pipeline.py`
- Verify: `output/pdf/debussy_qualitative_pipeline.pdf`

**Interfaces:**
- Consumes: `case["prompt_markers"]` from Task 1.
- Produces: updated PNG/PDF with an evidence-complete prompt panel and three prompt legend entries.

- [ ] **Step 1: Add a failing source-level rendering assertion**

Add to the existing build test:

```python
source = (figure.ROOT / "tools/build_staff_aware_qualitative_pipeline.py").read_text(encoding="utf-8")
assert "beam: 3 staff-line negatives suppressed" not in source
assert 'label="staff-line negative"' in source
assert 'label="suppressed negative"' in source
```

- [ ] **Step 2: Run the rendering test and verify failure**

Run: `python -m pytest tests/test_build_staff_aware_qualitative_pipeline.py::test_build_writes_single_page_pdf_and_renderable_preview -q`

Expected: FAIL because the old prose annotation remains and the new legend labels are absent.

- [ ] **Step 3: Implement marker drawing and legend entries**

Replace the prose annotation with scatter layers sourced from `case["prompt_markers"]`:

```python
for marker_group in case["prompt_markers"].values():
    if marker_group["positive"]:
        px, py = zip(*marker_group["positive"])
        axes[0].scatter(px, py, s=10, c=POSITIVE_COLOR, edgecolors="white", linewidths=0.35, zorder=6)
    if marker_group["retained_negative"]:
        nx, ny = zip(*marker_group["retained_negative"])
        axes[0].scatter(nx, ny, s=15, c=NEGATIVE_COLOR, marker="X", linewidths=0.45, zorder=7)
    if marker_group["suppressed_negative"]:
        sx, sy = zip(*marker_group["suppressed_negative"])
        axes[0].scatter(sx, sy, s=22, c=NEGATIVE_COLOR, marker=r"$\otimes$", linewidths=0.55, zorder=7)
```

Draw detector and adaptive boxes for both prompt hypotheses. Extend the legend with exact labels `positive prompt`, `staff-line negative`, and `suppressed negative`, using matching `Line2D` markers. Use two legend rows if necessary to avoid clipping.

- [ ] **Step 4: Run tests and regenerate artifacts**

Run:

```powershell
python -m pytest tests/test_build_staff_aware_qualitative_pipeline.py -q
python tools/build_staff_aware_qualitative_pipeline.py --out output/pdf
```

Expected: tests PASS and both `output/pdf/debussy_qualitative_pipeline.png` and `.pdf` are rewritten.

- [ ] **Step 5: Render and visually verify the PDF**

Run:

```powershell
$pdftoppm='C:\Users\A\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\poppler\Library\bin\pdftoppm.exe'
& $pdftoppm -png -f 1 -singlefile -r 200 output\pdf\debussy_qualitative_pipeline.pdf tmp\pdfs\debussy_staff_aware\page_prompt_markers
```

Expected: panel (b) visibly contains green dots, red filled X marks, and red circled-x marks; the legend is legible; no elements overlap or clip; panels (a) and (c)-(f) are unchanged.

- [ ] **Step 6: Replace the delivered PDF and verify identity**

```powershell
Copy-Item -LiteralPath 'D:\PyCharmPojects\Staff\output\pdf\debussy_qualitative_pipeline.pdf' -Destination 'C:\Users\A\Downloads\debussy_qualitative_pipeline.pdf' -Force
Get-FileHash -Algorithm SHA256 'D:\PyCharmPojects\Staff\output\pdf\debussy_qualitative_pipeline.pdf','C:\Users\A\Downloads\debussy_qualitative_pipeline.pdf'
```

Expected: both SHA-256 hashes are identical.

- [ ] **Step 7: Commit the rendering change**

```powershell
git add tests/test_build_staff_aware_qualitative_pipeline.py tools/build_staff_aware_qualitative_pipeline.py
git commit -m "feat: visualize retained and suppressed SAM2 prompts"
```
