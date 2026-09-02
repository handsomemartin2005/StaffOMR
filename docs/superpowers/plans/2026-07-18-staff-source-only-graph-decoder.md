# STAFF Source-Only Graph Decoder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and validate a source-only structured STAFF decoder whose frozen configuration obtains official OLiMPiC test200 SER below the reproduced Zeus zero-shot result of 56.49.

**Architecture:** Add a pure-Python sidecar decoder over existing `symbols_v2_1_shapes.json` and `notes_v2_1.json` artifacts. It extracts detector/rule/SAM2/graph features, aligns source predictions to GrandStaff MusicXML supervision, learns compact candidate and relation scores, performs deterministic constrained beam decoding, and applies only evidence-backed residual edits.

**Tech Stack:** Python 3.11, standard library, NumPy, scikit-learn/joblib, pytest, existing STAFF artifact JSON, existing MusicXML/LMX evaluators.

## Global Constraints

- Do not use OLiMPiC training labels, validation labels, pseudo-label selection, or test200 labels.
- Do not compute a new OLiMPiC test200 metric until the source-domain gate manifest is frozen.
- Train only on GrandStaff, DeepScores, and source-derived synthetic corruption.
- Each learned module must stop at or before epoch 250.
- Peak GPU memory must remain below 15 GB and peak system memory below 15 GB.
- Failed and empty predictions remain in every evaluation.
- Preserve unrelated existing worktree changes.

---

## File Structure

- `tools/staff_graph/features.py`: stable feature schema and artifact-to-node conversion.
- `tools/staff_graph/musicxml_gt.py`: source MusicXML parsing and normalized ground-truth events.
- `tools/staff_graph/alignment.py`: monotonic source prediction-to-ground-truth alignment.
- `tools/staff_graph/relations.py`: relation candidate generation and learned relation score interface.
- `tools/staff_graph/model.py`: compact candidate/event classifiers and persisted model bundle.
- `tools/staff_graph/decoder.py`: constrained event generation and deterministic beam search.
- `tools/staff_graph/corrector.py`: evidence-backed residual reranker and edit ledger.
- `tools/staff_graph/io.py`: JSON manifests, hashing, and atomic result writes.
- `tools/build_staff_graph_training_data.py`: source-only dataset builder CLI.
- `tools/train_staff_graph_decoder.py`: training and source-development gate CLI.
- `tools/run_staff_graph_decoder.py`: frozen artifact inference CLI.
- `tools/freeze_staff_graph_experiment.py`: gate validation and immutable configuration manifest.
- `tests/staff_graph/`: unit and integration tests for every boundary.

### Task 1: Artifact Contract and SAM2 Feature Extraction

**Files:**
- Create: `tools/staff_graph/__init__.py`
- Create: `tools/staff_graph/features.py`
- Create: `tests/staff_graph/test_features.py`

**Interfaces:**
- Consumes: `symbols_v2_1_shapes.json` dictionaries.
- Produces: `FeatureSchema`, `SymbolNode`, `extract_symbol_nodes(payload)`.

- [ ] **Step 1: Write the failing feature-contract tests**

```python
from tools.staff_graph.features import FEATURE_NAMES, extract_symbol_nodes


def test_sam2_features_are_not_silently_zeroed():
    payload = {"staff_space": 10, "symbols": [{
        "id": "n1", "class": "filled_notehead", "bbox": [10, 20, 30, 40],
        "confidence": 0.8, "source": "detector+sam2", "mask_score": 0.9,
        "shape": {"fill_ratio": 0.6, "orientation_angle_degrees": 12,
                  "staff_overlap_ratio": 0.2, "centroid": [21, 31]},
    }]}
    node = extract_symbol_nodes(payload)[0]
    assert node.features[FEATURE_NAMES.index("mask_fill_ratio")] == 0.6
    assert node.missing[FEATURE_NAMES.index("mask_fill_ratio")] is False


def test_missing_mask_has_explicit_missing_indicator():
    payload = {"staff_space": 10, "symbols": [{
        "id": "n1", "class": "filled_notehead", "bbox": [10, 20, 30, 40],
        "confidence": 0.8, "source": "detector",
    }]}
    node = extract_symbol_nodes(payload)[0]
    index = FEATURE_NAMES.index("mask_fill_ratio")
    assert node.features[index] == 0.0
    assert node.missing[index] is True
```

- [ ] **Step 2: Run the tests and verify the import failure**

Run: `python -m pytest tests/staff_graph/test_features.py -v`

Expected: FAIL because `tools.staff_graph.features` does not exist.

- [ ] **Step 3: Implement immutable nodes and the fixed schema**

```python
FEATURE_NAMES = (
    "confidence", "source_detector", "source_rule", "staff_x", "staff_y",
    "width_staff", "height_staff", "mask_score", "mask_fill_ratio",
    "mask_aspect_ratio", "mask_orientation_sin", "mask_orientation_cos",
    "mask_staff_overlap", "mask_centroid_dx", "mask_centroid_dy",
    "skeleton_length_staff", "skeleton_endpoint_count",
)

@dataclass(frozen=True)
class SymbolNode:
    symbol_id: str
    symbol_class: str
    staff: int | None
    bbox: tuple[float, float, float, float]
    features: tuple[float, ...]
    missing: tuple[bool, ...]
```

Implement safe numeric extraction, staff-space normalization, angle encoding, source flags, centroid offsets, skeleton statistics, and explicit missing indicators. Reject duplicate symbol IDs and non-finite values.

- [ ] **Step 4: Run the feature tests**

Run: `python -m pytest tests/staff_graph/test_features.py -v`

Expected: PASS.

- [ ] **Step 5: Commit the artifact contract**

```powershell
git add tools/staff_graph/__init__.py tools/staff_graph/features.py tests/staff_graph/test_features.py
git commit -m "feat: extract STAFF graph and SAM2 features"
```

### Task 2: Ground-Truth Event Parsing and Geometric Alignment

**Files:**
- Create: `tools/staff_graph/musicxml_gt.py`
- Create: `tools/staff_graph/alignment.py`
- Create: `tests/staff_graph/test_musicxml_alignment.py`

**Interfaces:**
- Consumes: GrandStaff MusicXML and extracted notehead candidates.
- Produces: `GroundTruthEvent`, `Alignment`, `parse_musicxml_events(path)`, `align_candidates(nodes, events, staves)`.

- [ ] **Step 1: Write failing parser and alignment tests**

```python
def test_musicxml_parser_preserves_measure_staff_pitch_duration_and_chord(tmp_path):
    path = tmp_path / "score.musicxml"
    path.write_text(MINIMAL_TWO_NOTE_XML, encoding="utf-8")
    events = parse_musicxml_events(path)
    assert [(e.measure, e.staff, e.pitch, e.duration_type, e.is_chord) for e in events] == [
        (1, 1, "C4", "quarter", False),
        (1, 1, "E4", "quarter", True),
    ]


def test_alignment_is_monotonic_and_can_leave_false_candidates_unmatched():
    alignment = align_candidates(CANDIDATES, EVENTS, STAVES)
    matched = [(a.candidate_index, a.event_index) for a in alignment if a.event_index is not None]
    assert matched == sorted(matched)
    assert any(a.event_index is None for a in alignment)
```

- [ ] **Step 2: Verify the tests fail**

Run: `python -m pytest tests/staff_graph/test_musicxml_alignment.py -v`

Expected: FAIL because parser/alignment modules do not exist.

- [ ] **Step 3: Implement MusicXML event parsing**

Parse `measure/@width`, `note/@default-x`, pitch, accidental, type, dot, stem, staff, voice, beam, chord, rest, and print-object. Convert measure-local x into cumulative normalized system x. Preserve missing `default-x` as `None`.

- [ ] **Step 4: Implement deterministic dynamic-programming alignment**

Use a monotonic edit alignment per staff with match cost:

```python
cost = (
    2.0 * normalized_x_distance
    + 1.0 * pitch_step_distance
    + 0.5 * duration_family_mismatch
    - 0.25 * candidate_confidence
)
```

Include explicit candidate-deletion and ground-truth-insertion costs. Resolve ties by fewer unmatched ground-truth events, then fewer unmatched candidates, then lexicographic candidate ID.

- [ ] **Step 5: Run parser/alignment tests on fixtures and one real source page**

Run: `python -m pytest tests/staff_graph/test_musicxml_alignment.py -v`

Expected: PASS, including deterministic repeated output.

- [ ] **Step 6: Commit source alignment**

```powershell
git add tools/staff_graph/musicxml_gt.py tools/staff_graph/alignment.py tests/staff_graph/test_musicxml_alignment.py
git commit -m "feat: align STAFF candidates with source MusicXML"
```

### Task 3: Source-Only Training Dataset Builder

**Files:**
- Create: `tools/staff_graph/io.py`
- Create: `tools/build_staff_graph_training_data.py`
- Create: `tests/staff_graph/test_training_data_builder.py`

**Interfaces:**
- Consumes: `outputs/ijcv_repro/ours_grandstaff_lmx_train1000_calibration_source/*` and `data/ijcv_samples/grandstaff-lmx-train1000/*.musicxml`.
- Produces: JSONL rows, `split_manifest.json`, `feature_schema.json`, corruption metadata.

- [ ] **Step 1: Write failing deterministic-split and target-domain-rejection tests**

```python
def test_split_is_grouped_deterministic_and_disjoint(tmp_path):
    result = build_dataset(SOURCE_ROOT, GT_ROOT, tmp_path, seed=20260718, limit=12)
    assert set(result.train_ids).isdisjoint(result.dev_ids)
    assert result == build_dataset(SOURCE_ROOT, GT_ROOT, tmp_path / "again", seed=20260718, limit=12)


def test_builder_rejects_olimpic_paths(tmp_path):
    with pytest.raises(ValueError, match="target-domain path"):
        build_dataset(Path("data/ijcv_samples/olimpic-test200-flat"), GT_ROOT, tmp_path)
```

- [ ] **Step 2: Verify the tests fail**

Run: `python -m pytest tests/staff_graph/test_training_data_builder.py -v`

Expected: FAIL because the builder does not exist.

- [ ] **Step 3: Implement the builder**

Use an 80/20 page-grouped split with seed `20260718`. Write candidate rows with keep/pitch/duration/accidental/dot targets, relation candidate rows, alignment confidence, and feature missing flags. Add reproducible source-only corruption for box jitter, confidence noise, candidate deletion/insertion, and relation deletion.

- [ ] **Step 4: Run a 12-page smoke build**

Run:

```powershell
python tools/build_staff_graph_training_data.py --artifacts outputs/ijcv_repro/ours_grandstaff_lmx_train1000_calibration_source --ground-truth data/ijcv_samples/grandstaff-lmx-train1000 --out outputs/staff_graph/source_smoke12 --limit 12 --seed 20260718
```

Expected: exit 0; train/dev disjoint; no path containing `olimpic`; every row has the feature-schema hash.

- [ ] **Step 5: Commit the source builder**

```powershell
git add tools/staff_graph/io.py tools/build_staff_graph_training_data.py tests/staff_graph/test_training_data_builder.py
git commit -m "feat: build source-only STAFF graph training data"
```

### Task 4: Learned Relation and Event Models

**Files:**
- Create: `tools/staff_graph/relations.py`
- Create: `tools/staff_graph/model.py`
- Create: `tools/train_staff_graph_decoder.py`
- Create: `tests/staff_graph/test_model_bundle.py`

**Interfaces:**
- Consumes: Task 3 JSONL and manifests.
- Produces: `StaffGraphModelBundle`, calibrated probabilities, `training_report.json`, joblib checkpoint.

- [ ] **Step 1: Write failing model-bundle tests**

```python
def test_model_bundle_refuses_schema_mismatch(tmp_path):
    bundle = train_tiny_bundle(TINY_ROWS, schema_hash="abc", max_epochs=5)
    bundle.save(tmp_path / "model.joblib")
    with pytest.raises(ValueError, match="feature schema"):
        StaffGraphModelBundle.load(tmp_path / "model.joblib", expected_schema_hash="def")


def test_sam2_and_relation_ablations_change_scores():
    bundle = train_tiny_bundle(TINY_ROWS, schema_hash="abc", max_epochs=5)
    full = bundle.score(TINY_ROWS[0])
    assert bundle.score(TINY_ROWS[0], ablate={"sam2"}) != full
    assert bundle.score(TINY_ROWS[0], ablate={"relations"}) != full
```

- [ ] **Step 2: Verify the tests fail**

Run: `python -m pytest tests/staff_graph/test_model_bundle.py -v`

Expected: FAIL because the model modules do not exist.

- [ ] **Step 3: Implement compact calibrated models**

Train class-balanced histogram gradient boosting classifiers for candidate keep, pitch correction, duration, accidental/dot, and each relation type. Use source-dev early stopping and `max_iter <= 250`. Persist schema hash, split hash, seed, library versions, feature importances, and calibration metrics.

- [ ] **Step 4: Run smoke training**

Run:

```powershell
python tools/train_staff_graph_decoder.py --dataset outputs/staff_graph/source_smoke12 --out outputs/staff_graph/model_smoke12 --max-epochs 10 --seed 20260718
```

Expected: exit 0; maximum trained iterations at most 10; memory report below limits; checkpoint reload succeeds.

- [ ] **Step 5: Commit learned scorers**

```powershell
git add tools/staff_graph/relations.py tools/staff_graph/model.py tools/train_staff_graph_decoder.py tests/staff_graph/test_model_bundle.py
git commit -m "feat: train source-only STAFF graph scorers"
```

### Task 5: Constrained Beam Decoder

**Files:**
- Create: `tools/staff_graph/decoder.py`
- Create: `tests/staff_graph/test_decoder.py`

**Interfaces:**
- Consumes: symbol nodes and model probabilities.
- Produces: legal structured events, candidate sequence score, evidence references.

- [ ] **Step 1: Write failing musical-constraint tests**

```python
def test_decoder_never_attaches_one_stem_to_incompatible_staffs():
    result = decode_page(CROSS_STAFF_FIXTURE, bundle=FIXED_BUNDLE, beam_size=8)
    assert all(edge.left_staff == edge.right_staff for edge in result.selected_stem_edges)


def test_duration_uses_mask_and_relation_evidence():
    filled = decode_page(FILLED_BEAMED_FIXTURE, bundle=FIXED_BUNDLE, beam_size=8)
    assert filled.events[0].duration_type == "eighth"
    assert {"mask_fill_ratio", "beam_stem_probability"} <= set(filled.events[0].evidence)


def test_beam_search_is_deterministic():
    assert decode_page(AMBIGUOUS_FIXTURE, FIXED_BUNDLE, 8) == decode_page(AMBIGUOUS_FIXTURE, FIXED_BUNDLE, 8)
```

- [ ] **Step 2: Verify the tests fail**

Run: `python -m pytest tests/staff_graph/test_decoder.py -v`

Expected: FAIL because the decoder does not exist.

- [ ] **Step 3: Implement constrained candidate generation and beam search**

Generate bounded alternatives for keep/drop, pitch correction, duration, accidental, dot, chord, and voice. Enforce staff compatibility, one primary stem per notehead, bounded notes per stem, legal duration vocabulary, accidental proximity/scope, and deterministic tie-breaking. Default `beam_size=16`; reject `beam_size > 64`.

- [ ] **Step 4: Run decoder tests**

Run: `python -m pytest tests/staff_graph/test_decoder.py -v`

Expected: PASS.

- [ ] **Step 5: Commit the decoder**

```powershell
git add tools/staff_graph/decoder.py tests/staff_graph/test_decoder.py
git commit -m "feat: add constrained STAFF graph decoder"
```

### Task 6: Evidence-Backed Residual Corrector

**Files:**
- Create: `tools/staff_graph/corrector.py`
- Create: `tests/staff_graph/test_corrector.py`

**Interfaces:**
- Consumes: beam candidates and source-trained model scores.
- Produces: selected sequence and `EditLedgerEntry` records.

- [ ] **Step 1: Write failing safety tests**

```python
def test_corrector_cannot_invent_unsupported_token():
    with pytest.raises(UnsupportedEdit):
        apply_edit(BASE_SEQUENCE, Edit("replace_pitch", "C4", "F#7"), evidence={})


def test_every_change_has_evidence_ledger_entry():
    result = rerank_candidates(CANDIDATE_SEQUENCES, SOURCE_LM)
    assert len(result.changed_tokens) == len(result.edit_ledger)
    assert all(entry.evidence_ids for entry in result.edit_ledger)
```

- [ ] **Step 2: Verify the tests fail**

Run: `python -m pytest tests/staff_graph/test_corrector.py -v`

Expected: FAIL because the corrector does not exist.

- [ ] **Step 3: Implement residual reranking**

Allow only candidate selection, evidence-backed pitch/duration/accidental substitution, unsupported-candidate deletion, and deterministic measure-balance repair. Enforce corpus token-count ratio in `[0.75, 1.25]`; emit the original structured candidate if the safety check fails.

- [ ] **Step 4: Run safety tests**

Run: `python -m pytest tests/staff_graph/test_corrector.py -v`

Expected: PASS.

- [ ] **Step 5: Commit the constrained corrector**

```powershell
git add tools/staff_graph/corrector.py tests/staff_graph/test_corrector.py
git commit -m "feat: constrain STAFF residual correction"
```

### Task 7: End-to-End Inference, Source Gates, and Freeze Manifest

**Files:**
- Create: `tools/run_staff_graph_decoder.py`
- Create: `tools/freeze_staff_graph_experiment.py`
- Create: `tests/staff_graph/test_end_to_end.py`

**Interfaces:**
- Consumes: artifact roots, model bundle, source ground truth for gate mode.
- Produces: per-page structured JSON, MusicXML/LMX, audit ledger, gate report, frozen manifest.

- [ ] **Step 1: Write failing sealed-evaluation tests**

```python
def test_inference_refuses_target_metric_before_freeze(tmp_path):
    with pytest.raises(PermissionError, match="frozen manifest"):
        run_target_evaluation(TARGET_ARTIFACTS, tmp_path, frozen_manifest=None)


def test_freeze_requires_all_source_gates(tmp_path):
    report = GateReport(sam2_gain=0.2, relation_gain=4.0, decoder_gain=13.0, corrector_gain=2.5)
    with pytest.raises(ValueError, match="SAM2 gate"):
        freeze_experiment(report, tmp_path / "frozen.json")
```

- [ ] **Step 2: Verify the tests fail**

Run: `python -m pytest tests/staff_graph/test_end_to_end.py -v`

Expected: FAIL because end-to-end CLIs do not exist.

- [ ] **Step 3: Implement inference and source-gate reports**

Write per-page outputs atomically. Record failures as empty predictions. Compute full, no-SAM2, no-learned-relations, and no-corrector source-dev variants using one fixed checkpoint. Capture runtime, process memory, optional CUDA peak memory, token-count ratio, feature importance, and error categories.

- [ ] **Step 4: Implement freeze validation**

Require the exact design gates: SAM2 gain at least 1.0 SER or 3.0 direct submetric points; learned relation gain at least 3.0 SER; complete decoder gain at least 12.0 SER; corrector gain at least 2.0 SER; resource and test gates pass. Hash model, config, split, and feature schema into `frozen_manifest.json`.

- [ ] **Step 5: Run all unit/integration tests**

Run: `python -m pytest tests/staff_graph -v`

Expected: PASS.

- [ ] **Step 6: Commit inference and freezing**

```powershell
git add tools/run_staff_graph_decoder.py tools/freeze_staff_graph_experiment.py tests/staff_graph/test_end_to_end.py
git commit -m "feat: gate and freeze STAFF graph experiments"
```

### Task 8: Full Source Build, Training, and One-Time OLiMPiC Evaluation

**Files:**
- Create: `outputs/staff_graph/source_full/` generated artifacts.
- Create: `outputs/staff_graph/model_full/` generated checkpoint and reports.
- Create: `outputs/staff_graph/olimpic_test200_frozen/` generated final outputs.

**Interfaces:**
- Consumes: Tasks 1-7 and existing source/target artifacts.
- Produces: frozen source report and the final official comparison against Zeus.

- [ ] **Step 1: Build the full source dataset**

Run:

```powershell
python tools/build_staff_graph_training_data.py --artifacts outputs/ijcv_repro/ours_grandstaff_lmx_train1000_calibration_source --ground-truth data/ijcv_samples/grandstaff-lmx-train1000 --out outputs/staff_graph/source_full --seed 20260718
```

Expected: no OLiMPiC paths in the manifest; deterministic 80/20 split; feature and split hashes written.

- [ ] **Step 2: Train for at most 250 iterations**

Run:

```powershell
python tools/train_staff_graph_decoder.py --dataset outputs/staff_graph/source_full --out outputs/staff_graph/model_full --max-epochs 250 --seed 20260718
```

Expected: early stopping or completion by iteration 250; resource limits respected.

- [ ] **Step 3: Run source-development gates and fixed ablations**

Run:

```powershell
python tools/run_staff_graph_decoder.py --artifacts outputs/ijcv_repro/ours_grandstaff_lmx_train1000_calibration_source --model outputs/staff_graph/model_full/model.joblib --ground-truth data/ijcv_samples/grandstaff-lmx-train1000 --split-manifest outputs/staff_graph/source_full/split_manifest.json --mode source-gates --out outputs/staff_graph/model_full/source_gates
```

Expected: all Gate A-D checks pass. If any gate fails, stop before target evaluation and return to the single failing component.

- [ ] **Step 4: Freeze the experiment**

Run:

```powershell
python tools/freeze_staff_graph_experiment.py --model-dir outputs/staff_graph/model_full --gate-report outputs/staff_graph/model_full/source_gates/gate_report.json --out outputs/staff_graph/model_full/frozen_manifest.json
```

Expected: immutable hashes and `target_evaluation_allowed: true`.

- [ ] **Step 5: Run OLiMPiC test200 once**

Run:

```powershell
python tools/run_staff_graph_decoder.py --artifacts outputs/ijcv_repro/ours_olimpic_test200_artifact_view --model outputs/staff_graph/model_full/model.joblib --frozen-manifest outputs/staff_graph/model_full/frozen_manifest.json --mode target-inference --out outputs/staff_graph/olimpic_test200_frozen
```

Then use the existing official OLiMPiC evaluator on all 200 outputs.

Expected success: `SER < 56.49`; report STAFF baseline 74.42 and Zeus 56.49 beside the new frozen result. If not reached, preserve the result and return to source-domain error analysis without target-driven parameter selection.

- [ ] **Step 6: Run verification and record the final report**

Run: `python -m pytest tests -v`

Expected: all tests pass. The report contains 200/200 coverage or explicitly counted empty failures, hashes, resources, ablations, and exact commands.

## Plan Self-Review

- Every design requirement maps to a task and source-domain gate.
- All module boundaries define their consumed and produced interfaces.
- No step requires OLiMPiC labels before the frozen final evaluation.
- SAM2, learned relations, and correction each have an explicit ablation threshold.
- Training caps and memory limits are executable gates, not narrative claims.
- The final target evaluation is blocked in code until the source gates pass.
