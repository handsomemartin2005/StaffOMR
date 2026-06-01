# Traditional OMR Demo Baseline

This repository contains a compact, reproducible demo for exploring how far traditional image processing can go on dense piano-score OMR before moving to neural-network models.

The long-term goal is not to replace neural OMR with hand-written rules. The current rule-based pipeline is a diagnostic baseline: it exposes which parts of score recognition are geometrically stable, which parts fail, and which errors should become neural detector/classifier/segmentation tasks.

## Repository Contents

```text
tools/traditional_omr_demo.py
docs/traditional_omr_research_proposal.md
docs/traditional-omr-process-walkthrough.pptx
docs/contact-sheet.png
examples/lg2267728_v18/input/lg-2267728-aug-beethoven--page-2.png
examples/lg2267728_v18/results/
```

Large local datasets are intentionally not committed. The demo includes only one input image and the current `v18` outputs.

## Run the Demo

```bash
pip install -r requirements.txt
python tools/traditional_omr_demo.py --out-dir outputs/traditional_omr_demo
```

The default input is:

```text
examples/lg2267728_v18/input/lg-2267728-aug-beethoven--page-2.png
```

The script writes:

- `binary_cleaned_input.png`
- `overlay_detected_staff_notes.png`
- `notehead_weight_regions.png`
- `reconstructed_from_detection.png`
- `demo_visual_result.pdf`
- `detection_result.json`

## Current Baseline Counts

For the included `v18` example:

| Element | Count |
|---|---:|
| staff lines | 50 |
| staves | 10 |
| noteheads | 457 |
| filled noteheads | 415 |
| open noteheads | 42 |
| stems | 357 |
| ledger lines | 288 |
| barlines | 46 |
| rests | 9 |
| accidentals | 21 |
| text regions | 5 |
| beam links | 141 |
| beam groups | 68 |

## Research Direction

The next stage should convert the observed failure modes into neural-network tasks:

- notehead detection and filled/open classification
- accidental/rest/clef/text symbol detection
- beam, slur, tie, and stem relation modeling
- segmentation for staff and fine symbols
- graph or sequence decoding to MusicXML / Linearized MusicXML

The structured OMR output can then support LLM-based music education workflows, including score explanation, practice feedback, rhythm checking, accidental-scope explanation, and student error localization.
