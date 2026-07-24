# Staff-aware prompt marker visualization

## Scope

Revise only panel (b) of `debussy_qualitative_pipeline.pdf`. Keep `test_0006`, the current crop, panels (a) and (c)-(f), and the safe-residual experiment unchanged.

## Evidence source

Panel (b) represents the prompt-generation stage before symbol acceptance, so it may show genuine prompts from overlapping detector hypotheses that are later filtered in panel (c).

- The accepted beam hypothesis `deim_000069` has three positive prompts and three staff-line negative candidates. All three negative candidates overlap its detector support and are suppressed before the SAM2 call.
- The overlapping beam hypothesis `deim_000278`, inside the same crop, has three staff-line negative prompts that survive filtering and are passed to SAM2.
- Retained points come directly from the recorded `negative_points` field.
- Suppressed point coordinates are reconstructed with the same `build_prompt_points` function and classified with the same detector-support predicate used by `filter_negative_points_outside_support`. The figure builder must assert the recorded and reconstructed counts agree.

## Visual encoding

- Green filled circle: `positive prompt`.
- Red filled `X`: `staff-line negative`, actually passed to SAM2.
- Red circled `x`: `suppressed negative`, generated but excluded because it overlaps target support.
- Dashed colored rectangles: adaptive prompt boxes.
- Dotted gray rectangles: detector support boxes.

Remove the sentence `beam: 3 staff-line negatives suppressed`. Add all three point types to the shared legend. Panel (b) remains titled `Staff-aware prompts`; the transition to panel (c), `Accepted symbols`, communicates that panel (b) contains candidate hypotheses.

## Validation

Automated tests must verify that the chosen crop contains all three prompt categories, that `deim_000069` has exactly three reconstructed suppressed negatives, and that `deim_000278` has exactly three retained negatives. Regenerate both PNG and PDF, render the PDF through Poppler, and visually check marker separation, legend legibility, clipping, and unchanged downstream panels.
