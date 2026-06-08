# V2 Technical Plan: Neural Symbol Recognition for OMR

## 1. Version Positioning

V1 is the current interpretable rule-based OMR baseline. It uses binarization, staff-line projection, geometric filtering, connected components, density rules, and visual debugging outputs to identify staff lines, noteheads, stems, beams, barlines, rests, accidentals, text regions, ledger lines, and simple slur/tie candidates.

V2 does not replace V1 completely. V2 keeps the stable geometric parts of V1 and replaces the unstable symbol-recognition rules with neural image-recognition modules.

The V2 scope is deliberately limited to symbol-level recognition:

```text
Input score image
-> locate staff coordinate system
-> detect basic visual symbols
-> segment symbol masks when useful
-> produce symbol-level JSON
```

V2 does not yet solve full score semantics:

- no complete MusicXML decoding
- no voice separation
- no measure-level rhythm validation
- no cross-staff musical relation inference
- no end-to-end page-to-score transcription

## 2. Core Stack

The first neural version uses:

```text
DEIM-D-FINE + SAM2 + DINOv2
```

Their roles are different:

| Module | Model family | Primary output | V2 role |
|---|---|---|---|
| DEIM-D-FINE | DETR-style object detection | class + bounding box | Main symbol detector |
| SAM2 | promptable segmentation | pixel mask | Mask refinement from detected boxes |
| DINOv2 | self-supervised visual features | embedding vector | Assisted labeling, retrieval, clustering, ReID-style deduplication |
| V1 rules | classical geometry | staff lines, staff space, staff regions | Coordinate system and rule-based constraints |

The main inference direction is:

```text
V1 staff localization
-> page/staff tiling
-> DEIM-D-FINE symbol detection
-> SAM2 mask refinement using DEIM boxes as prompts
-> DINOv2 embedding for label QA, retrieval, clustering, and duplicate handling
-> symbol-level JSON
```

SAM2 is not the first-stage detector. It is used after a candidate box exists, because SAM2 gives masks but does not reliably assign music-symbol classes by itself.

## 3. What V1 Rules Are Kept

The following V1 logic remains in V2 because it is stable, explainable, and useful for reducing the learning problem:

| V1 component | Keep in V2? | Reason |
|---|---:|---|
| Binarization / grayscale normalization | yes | Provides simple alternative input channels and debugging views |
| Staff-line detection | yes | Horizontal projection is currently stable on printed scores |
| Staff grouping | yes | Gives staff index and vertical coordinate system |
| Staff-space estimation | yes | Normalizes symbol sizes across pages and fonts |
| Staff crop generation | yes | Reduces full-page detection into smaller high-resolution regions |
| Pitch-grid coordinates | yes | Used after detection for notehead pitch-step assignment |
| Symbol-to-staff assignment | yes | Geometric post-processing remains simple and reliable |
| Overlay / weight-map / reconstruction visualizations | yes | Still needed for debugging neural errors |

The following V1 rules become weak labels, fallbacks, or evaluation aids rather than the primary recognizer:

| V1 rule | V2 replacement |
|---|---|
| Notehead ellipse-density scoring | DEIM-D-FINE notehead detector |
| Filled/open center-density rule | DEIM class head, optionally crop classifier later |
| Accidental connected-component rules | DEIM-D-FINE accidental detector |
| Rest shape heuristics | DEIM-D-FINE rest detector |
| Stem vertical-run detection | DEIM-D-FINE detection and/or SAM2 mask |
| Beam corridor scoring | DEIM-D-FINE detection and/or SAM2 mask |
| Ledger-line short-run detection | DEIM-D-FINE detection and/or SAM2 mask |
| Text-region connected-component clustering | DEIM-D-FINE text-region detector |

## 4. Target Symbol Classes

V2 starts with basic visual symbols only:

```text
filled_notehead
open_notehead
stem
beam
barline
ledger_line
sharp
flat
natural
rest
treble_clef
bass_clef
text_region
slur_or_tie
```

Optional later subclasses:

```text
quarter_rest
eighth_rest
half_rest
whole_rest
double_barline
repeat_barline
beam_primary
beam_secondary
grace_notehead
ornament
dynamic_text
```

The first version should avoid too many subclasses. A smaller class set is easier to label, train, debug, and evaluate.

## 5. Data Flow

### 5.1 Preprocessing

Inputs may be PNG images or PDF-rendered pages.

V1 preprocessing remains available:

```text
load image
-> grayscale
-> binarize
-> remove tiny components
-> detect staff lines
-> estimate staff_space
-> group staves
```

For neural models, each sample can expose multiple image variants:

```text
RGB original
grayscale image
binary mask
staff-line-removed image
```

The first DEIM-D-FINE experiment should use original RGB or grayscale staff crops. Binary and cleaned variants can be added as ablation inputs later.

### 5.2 Tiling

Full-page music scores contain small symbols. V2 should not train only on heavily resized full pages.

Recommended crop units:

```text
staff crop:
  x = full staff width
  y = staff_y0 - 3.5 * staff_space
      to staff_y1 + 3.5 * staff_space

system crop:
  paired grand-staff region for piano

sliding tile:
  high-resolution page tiles with overlap
```

For V2 first implementation:

```text
primary training/inference unit = staff crop
tile overlap = 10% to 20%
box coordinates are converted back to page coordinates
```

### 5.3 Detection

DEIM-D-FINE is the primary symbol detector.

Input:

```text
staff crop or page tile
```

Output:

```json
{
  "bbox": [x0, y0, x1, y1],
  "class": "filled_notehead",
  "confidence": 0.93,
  "source": "deim_dfine"
}
```

V1 staff geometry is then used to enrich detections:

```json
{
  "staff": 3,
  "pitch_step": 7,
  "normalized_width": 0.92,
  "normalized_height": 0.61
}
```

### 5.4 Segmentation

SAM2 is used after detection, not before detection.

Input:

```text
image crop + DEIM bbox prompt
```

Output:

```json
{
  "mask": "RLE or polygon",
  "mask_score": 0.88,
  "source": "sam2_from_deim_box"
}
```

SAM2 is most useful for:

- stem
- beam
- ledger_line
- slur_or_tie
- barline
- notehead outline refinement

SAM2 masks should not be blindly trusted. V1 geometric checks still apply:

```text
stem mask should be mostly vertical
beam mask should be elongated and near stem terminals
ledger line should cross or sit near a notehead
slur/tie should be curved and not coincide with staff lines
```

### 5.5 DINOv2-Assisted Labeling and ReID

DINOv2 is not the main detector. It provides visual embeddings for symbol crops.

Uses:

1. assisted labeling
   - cluster crop embeddings
   - label one cluster at a time
   - find mislabeled samples

2. weak-label cleanup
   - use V1 outputs as initial labels
   - embed all crops
   - flag outliers inside each class

3. ReID-style duplicate handling
   - when overlapping tiles detect the same symbol twice, compare geometry and DINOv2 embedding similarity
   - merge duplicates when boxes overlap and embeddings are close

4. retrieval
   - user marks one true `natural`
   - system retrieves visually similar candidates across the page or dataset

ReID is therefore treated as a DINOv2 embedding use case, not as a separate first-version model.

## 6. Annotation Strategy

V2 should not start from fully manual labeling.

Recommended path:

```text
V1 rules generate weak boxes
-> manually review a small seed set
-> DINOv2 clusters similar crops
-> human corrects clusters
-> train DEIM-D-FINE
-> use DEIM predictions + SAM2 masks for active learning
-> repeat
```

Annotation format should be COCO-compatible for detection:

```json
{
  "images": [],
  "annotations": [
    {
      "bbox": [x, y, width, height],
      "category_id": 1,
      "area": 123,
      "iscrowd": 0
    }
  ],
  "categories": []
}
```

For segmentation, add either polygons or RLE masks when available. Mask annotation can be sparse at first; detection labels are the priority.

## 7. Training Plan

### Phase 1: Dataset bootstrap

Inputs:

- existing `examples/lg2267728_v18` page
- local PDF/MXL/OMR files
- DeepScores-style dense printed-score images

Steps:

```text
1. Run V1 on selected pages.
2. Export weak symbol boxes.
3. Convert weak boxes to COCO format.
4. Build a review set.
5. Use DINOv2 embeddings to cluster and clean labels.
```

### Phase 2: First detector

Train DEIM-D-FINE on the cleaned symbol boxes.

Initial class grouping:

```text
notehead
stem
beam
barline
ledger_line
accidental
rest
clef
text_region
slur_or_tie
```

If the dataset is clean enough, split classes further:

```text
filled_notehead / open_notehead
sharp / flat / natural
treble_clef / bass_clef
```

### Phase 3: Mask refinement

Use SAM2 after detection:

```text
DEIM bbox -> SAM2 mask -> geometry sanity checks -> JSON mask field
```

Only keep masks that pass class-specific sanity checks.

### Phase 4: Iterative improvement

```text
Train detector
-> run on held-out pages
-> inspect false positives / false negatives
-> use DINOv2 retrieval to find similar failures
-> add corrected labels
-> retrain
```

## 8. Evaluation

V2 evaluation is symbol-level.

Detection metrics:

- AP / AP50 / AP75
- per-class precision
- per-class recall
- false positives per page
- missed symbols per staff

Segmentation metrics where masks exist:

- mask IoU
- skeleton/centerline error for stems, beams, ledger lines, and slurs

V1-vs-V2 comparison:

| Symbol | V1 issue | V2 expected improvement |
|---|---|---|
| notehead | density thresholds fail in dense chords | detector learns local visual context |
| filled/open | staff-line removal can damage centers | class prediction uses raw crop context |
| accidental | font-sensitive connected components | detector learns symbol shape variation |
| rest | heuristic shape rules are brittle | detector learns rest appearance |
| beam | corridor rules over-link or under-link | detector/mask handles local shape |
| ledger line | short horizontal runs confuse noise | detector/mask plus notehead relation |
| text region | rule templates are narrow | detector finds region even without OCR |

## 9. JSON Output Contract

V2 should keep a compatible structured output while adding neural fields.

Example:

```json
{
  "version": "v2",
  "input": "page.png",
  "staff_geometry_source": "v1_rules",
  "symbols": [
    {
      "id": "sym_000001",
      "class": "filled_notehead",
      "bbox": [120, 88, 132, 99],
      "confidence": 0.94,
      "staff": 0,
      "pitch_step": 4,
      "detector": "deim_dfine",
      "mask": null,
      "embedding_id": "dinov2_000001"
    }
  ]
}
```

The output should preserve enough fields to compare against V1:

- staff index
- normalized geometry
- class
- confidence
- bbox
- optional mask
- source module
- post-processing decisions

## 10. Implementation Deliverables

First V2 deliverables:

1. `tools/export_v1_weak_labels.py`
   - runs V1 detection
   - exports weak boxes to COCO

2. `tools/build_symbol_crops.py`
   - crops symbol candidates
   - stores crop metadata

3. `tools/embed_symbol_crops_dinov2.py`
   - creates DINOv2 embeddings
   - exports nearest-neighbor and cluster files

4. `tools/train_deim_dfine_symbols.py`
   - wrapper or config for DEIM-D-FINE training

5. `tools/infer_neural_symbols_v2.py`
   - runs V1 staff localization
   - runs DEIM-D-FINE on staff crops
   - optionally runs SAM2 masks
   - writes V2 symbol JSON and overlay

6. `docs/neural_symbol_recognition_v2_technical_plan.md`
   - this technical plan

## 11. Risks and Controls

| Risk | Control |
|---|---|
| Weak V1 labels contain errors | DINOv2 clustering and manual review before training |
| Small symbols lost during resizing | Train on staff crops or overlapping tiles, not only full pages |
| Dense chords create overlapping boxes | Keep high-resolution crops and evaluate notehead recall separately |
| SAM2 masks include staff lines or neighboring symbols | Use class-specific geometry checks and V1 staff masks |
| Detector learns one font only | Mix local scores with DeepScores-style images and rendered score variants |
| Too many classes too early | Start with grouped classes, split after stable AP/recall |

## 12. References

- DEIM: DETR with Improved Matching for Fast Convergence, CVPR 2025: https://openaccess.thecvf.com/content/CVPR2025/html/Huang_DEIM_DETR_with_Improved_Matching_for_Fast_Convergence_CVPR_2025_paper.html
- D-FINE: Redefine Regression Task in DETRs as Fine-grained Distribution Refinement, arXiv 2024: https://arxiv.org/abs/2410.13842
- DINOv2: Learning Robust Visual Features without Supervision, arXiv 2023: https://arxiv.org/abs/2304.07193
- SAM 2: Segment Anything in Images and Videos, arXiv 2024: https://arxiv.org/abs/2408.00714
