# Diagram Brief

## User Goal
- Output: editable overview figure for the AAAI 2027 first draft.
- Audience: AI and document-analysis reviewers unfamiliar with OMR.
- Must communicate: source/target domain shift, two complementary recognition paths, fixed evaluation, and the evidence budget.
- Must not do: imply that released Zeus weights are our architecture or claim state-of-the-art performance.

## Source Inventory
| id | source | type | role | priority | notes |
| --- | --- | --- | --- | --- | --- |
| S1 | `README.md` and V2.1 design | repository text | content and structure | must | Defines StaffOMR-SAM modules. |
| S2 | transfer result ledger | experiment report | content | must | Defines datasets, budgets, and metrics. |
| S3 | AAAI paper conventions | venue | style | should | Compact, readable two-column figure. |

## Requirement Traceability
| id | requirement | source evidence | level | planned encoding |
| --- | --- | --- | --- | --- |
| R1 | Show domain shift | S2 | must | Source and target score cards with a shift arrow. |
| R2 | Separate ownership | S1, S2 | must | Distinct modular and sequence-transfer lanes. |
| R3 | Show target-label budgets | S2 | must | 0, 100, and 1000 page chips. |
| R4 | Show fixed evaluation | S2 | must | LMX SER and error-taxonomy output block. |

## Semantic Model
| id | entity or relationship | direction | encoding | uncertainty |
| --- | --- | --- | --- | --- |
| E1 | source to target domains | left to right | dashed domain-shift arrow | none |
| E2 | images to modular recognition | left to right | blue lane | none |
| E3 | released sequence model to adaptation | left to right | orange lane | none |
| E4 | both paths to common evaluation | fan-in | green evaluation block | none |

## Style Contract
| font | palette | stroke | icon style | density | layout |
| --- | --- | --- | --- | --- | --- |
| Arial | blue, orange, green, neutral gray | 1.5--2 px | simple editable primitives | compact | landscape pipeline |

## Open Assumptions
| assumption | risk | how to verify |
| --- | --- | --- |
| The figure is an overview, not a full architecture diagram. | Readers may want module internals. | Method section and appendix retain module details. |

## Screenshot Review
| issue | observed screenshot | requirement evidence | cells | patch summary | status |
| --- | --- | --- | --- | --- | --- |
