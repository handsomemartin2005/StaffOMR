# STAFF Source-Only Structured Graph Decoder Design

## 1. Objective

Build a source-only STAFF decoding path that beats the reproduced Zeus GrandStaff-to-OLiMPiC zero-shot result of **56.49 SER** on the fixed OLiMPiC test200 split.

The new path must:

- use no OLiMPiC training labels, validation labels, pseudo-label selection, or test200 labels;
- train only on GrandStaff, DeepScores, and synthetic perturbations derived from source-domain data;
- keep each newly trained module at no more than 250 epochs;
- fit within 16 GB GPU memory and 16 GB system memory;
- make the existing staff prior, detector, SAM2 mask, relation graph, and correction stages carry distinct, measurable responsibilities;
- preserve all failed and empty predictions in evaluation.

The target is not considered reached unless the final fixed configuration obtains `SER < 56.49` with the existing official OLiMPiC evaluator.

## 2. Evidence and Root Cause

The current STAFF result is 74.42 / 73.57 SER on OLiMPiC test200. Existing evidence shows that the primary bottleneck is structured semantic decoding rather than mask boundary quality:

- SAM2 prompt variants change OLiMPiC SER only modestly and have almost no effect on GrandStaff and Polish.
- SAM2 mask geometry is already stored in `symbols_v2_1_shapes.json`, including fill ratio, centroid, orientation, staff overlap, polygon, mask area, and skeleton where applicable, but downstream note assembly uses only a small subset of these fields.
- Four relation scorers have been trained as auxiliary prototypes, but the main pipeline still uses `geometry_heuristic_relation_graph_v2_1`.
- The unconstrained Transformer corrector can catastrophically increase OLiMPiC SER because it may insert, delete, and substitute arbitrary tokens without respecting detected symbols or musical structure.
- Existing error taxonomy identifies pitch, duration, and notation as dominant error categories. These errors require joint symbol-relation decoding, not only better bounding boxes.
- Some OLiMPiC ablations become worse after detector candidates are added, showing that the current fusion stage lacks an uncertainty gate that can reject unsupported detector hypotheses.

## 3. Selected Architecture

The implementation will add a sidecar structured decoder that consumes existing STAFF artifacts. It will not rewrite the compiled V2.1 pipeline during the first iteration.

### 3.1 Candidate Evidence Layer

Each symbol becomes a feature-bearing candidate with:

- class and detector/rule confidence;
- source type and agreement between detector and rule branches;
- staff-relative center, width, height, and pitch offset;
- SAM2 mask score, fill ratio, shape orientation, aspect ratio, centroid offset, staff overlap, and skeleton statistics;
- neighborhood counts and compatible relation candidates.

Rule-only and detector-only candidates remain available, but a low-confidence detector-only candidate cannot reach the final event sequence unless supported by a learned relation or local consistency constraint.

### 3.2 Learned Relation Graph

The graph decoder will score at least these edge types:

- notehead to stem;
- stem to beam;
- ledger line to notehead;
- accidental to notehead;
- dot to note/rest;
- slur/tie to notehead endpoints;
- note-to-note voice and chord grouping.

The first implementation will use compact gradient-boosted or shallow neural scorers on tabular geometric features. This keeps memory low and allows calibrated probabilities. Existing relation artifacts are reused where their feature contract matches; otherwise the source-domain builder regenerates consistent training rows.

### 3.3 Structured Event Decoder

The decoder generates multiple legal event hypotheses per horizontal cluster rather than committing greedily. It jointly selects:

- symbol existence;
- note-to-stem and beam attachments;
- filled versus hollow notehead interpretation;
- pitch from staff coordinate, clef, and ledger evidence;
- duration from notehead state, stem, flag, beam, and dot evidence;
- chord membership, voice, measure placement, and accidental scope.

Candidate sequences are scored by the sum of calibrated evidence scores and penalties for violated constraints. Beam search retains a small configurable number of hypotheses so that uncertainty is preserved without unrestricted autoregressive generation.

### 3.4 Constrained Residual Corrector

The existing free-form token corrector is replaced in the main path by a source-trained reranker. It may choose among structured candidates and apply only edits backed by an observed candidate, relation, or deterministic musical constraint. It cannot invent an arbitrary pitch, duration, accidental, or measure token.

The corrector receives the original structured score, alternative candidate scores, measure-balance features, and source-domain language-model likelihood. Its output includes an edit ledger with the reason and supporting evidence for every change.

## 4. Training Data

All trainable components use source-domain supervision only.

### 4.1 Source Splits

- GrandStaff training pages provide symbol-to-LMX and event-order supervision.
- A fixed GrandStaff development split is held out before feature extraction and is the only labeled set used for model selection and early stopping.
- DeepScores supplies symbol-class and local geometry examples where compatible.
- Existing OLiMPiC artifacts may be used only for unlabeled inference after the configuration is frozen; no OLiMPiC ground truth enters training, threshold selection, or checkpoint selection.

### 4.2 Synthetic Domain Shift

Source pages are perturbed with transformations that do not use OLiMPiC statistics:

- resolution and staff-space scaling;
- blur, compression, contrast, background, and scan noise;
- symbol erosion/dilation and staff-line interference;
- bounding-box jitter, confidence corruption, candidate deletion, and false candidate insertion;
- local class-family confusion;
- controlled relation deletion and edge corruption.

The perturbation labels preserve the original source-domain event graph and provide explicit positive and negative examples for relation and candidate gating.

## 5. Development Gates

OLiMPiC test200 is not used for iteration. The final run is allowed only after all source-domain gates pass.

### Gate A: Feature Usage

- Removing SAM2-derived features must worsen GrandStaff development SER by at least 1.0 absolute point or worsen a directly supervised notehead-fill/duration submetric by at least 3.0 points.
- Removing learned relation scores must worsen GrandStaff development SER by at least 3.0 absolute points.
- Every feature family must have nonzero usage reported by permutation importance or ablation.

### Gate B: Structured Accuracy

- Relation F1 must exceed the existing heuristic for note-stem and beam-stem edges on the fixed GrandStaff development split.
- Pitch error and duration error must each decrease relative to the frozen STAFF baseline.
- The complete structured decoder must improve GrandStaff development SER by at least 12 absolute points relative to the matching frozen artifact baseline.

### Gate C: Corrector Safety

- The constrained corrector must improve aggregate GrandStaff development SER by at least 2 absolute points.
- It must never emit a token unsupported by its edit ledger.
- Predicted token count must remain between 0.75 and 1.25 times the pre-correction candidate sequence count at corpus level.
- No catastrophic aggregate regression is accepted on a second source-domain corruption severity.

### Gate D: Resource and Reproducibility

- Peak GPU memory is below 15 GB and peak system memory below 15 GB.
- Each learned module stops at or before epoch 250.
- Seeds, source splits, feature schema, checkpoints, and commands are written to the result manifest.
- All unit and integration tests pass before the final run.

## 6. OLiMPiC Evaluation Protocol

Historical OLiMPiC test200 results have already been observed in this project. To avoid further selection leakage, the new configuration must be frozen using only the source-domain gates above. No new OLiMPiC metric is computed until that freeze manifest is written.

The final evaluation will:

1. hash the configuration and checkpoints;
2. run inference for all 200 pages;
3. keep failures and empty outputs;
4. run the existing official evaluator once;
5. compare against STAFF 74.42 and Zeus 56.49;
6. run fixed A/B/C/D and learned-component ablations without changing the frozen configuration.

The primary outcome is official SER. SER without tuplets, failure count, runtime, peak memory, token-count ratio, and error-category changes are secondary outcomes.

## 7. Failure Handling

- Missing mask fields fall back to explicit missing indicators rather than fabricated zeros.
- Missing graph candidates produce a legal rule-based fallback event and an audit record.
- Model loading or schema mismatch fails before dataset inference.
- Pages that cannot be linearized are recorded and evaluated as empty predictions.
- If three successive architecture-level attempts fail the GrandStaff source gates, implementation stops for an architecture review instead of continuing parameter sweeps.

## 8. Test Strategy

Unit tests cover feature extraction, missing-mask handling, edge candidate generation, musical constraints, deterministic beam search, edit-ledger legality, and token-count safety.

Integration tests use small source-domain fixtures to verify:

- artifacts to graph rows;
- graph rows to event candidates;
- event candidates to legal LMX;
- ablations actually remove the intended evidence;
- repeated runs with the same seed are identical.

The full GrandStaff development evaluation is the release gate. OLiMPiC test200 is a final sealed evaluation, not a development test.

## 9. Deliverables

- source-only graph training-data builder;
- feature schema and manifest;
- learned relation/candidate scorers;
- structured beam decoder;
- constrained residual reranker and edit ledger;
- source-domain gate report;
- frozen configuration manifest;
- one final OLiMPiC test200 report against Zeus;
- module-utility ablation table showing measurable contributions from SAM2, learned relations, and constrained correction.
