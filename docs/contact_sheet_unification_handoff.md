# Contact Sheet Unification Handoff

## What changed

This pass finished the last contact-sheet and operator-surface cleanup items without changing the accepted IPP2-first measurement architecture.

The workflow is now:

1. analysis measures one IPP2 REDLine-rendered representative frame per clip
2. analysis persists numeric sample values and solve geometry
3. full contact-sheet report reuses those stored measurements
4. report renders corrected review stills only
5. contact sheet overlays are drawn from persisted solve geometry and persisted numeric sample values

## Root cause of the contact-sheet crash

The full contact-sheet path was still capable of deleting the stored `_measurement` preview directory before report generation tried to reuse it. That left the report path with missing original measurement stills and exposed legacy branches that were never supposed to be active in the final workflow.

There was also an over-broad perceptual forcing change in `_build_strategy_payloads()` that broke direct helper/test behavior even though the operational workflow should remain perceptual.

## Fixes

### 1. Preserve analysis measurement stills during report generation

`clear_preview_cache()` in:

- `/Users/sfouasnon/Desktop/R3DMatch/src/r3dmatch/report.py`

now accepts `preserve_measurement_previews`.

Both:

- `build_lightweight_analysis_report()`
- `build_contact_sheet_report()`

now:

1. load analysis records first
2. detect whether reusable IPP2 analysis measurements already exist
3. preserve `_measurement` previews when those reusable measurements are present

This prevents report generation from deleting the exact original stills it needs to reuse.

### 2. Keep operational review perceptual, but restore helper domain behavior

`_build_strategy_payloads()` now honors its `matching_domain` argument again for direct/helper use.

Operational entry points still force the real workflow to perceptual/IP P2:

- `/Users/sfouasnon/Desktop/R3DMatch/src/r3dmatch/workflow.py`
- `/Users/sfouasnon/Desktop/R3DMatch/src/r3dmatch/report.py`

### 3. Operator-facing sample labels are now neutral

Operator-facing output no longer uses Bright / Center / Dark.

It now uses:

- `Sample 1`
- `Sample 2`
- `Sample 3`

Numeric truth is persisted and reused through:

- `sample_1_ire`
- `sample_2_ire`
- `sample_3_ire`

### 4. Contact sheet uses overlay-backed measured stills

The contact sheet now displays:

- original detection overlays
- corrected detection overlays

from:

- `/Users/sfouasnon/Desktop/R3DMatch/src/r3dmatch/report.py`

using persisted:

- sphere center
- sphere radius
- sample region geometry
- sample IRE values

No report-side sphere re-detection or re-measurement is performed.

### 5. Sample Stability Ranking graph removed

The lightweight operator surface no longer emits the older confidence/stability ranking graph as a primary UI element.

## Proof

### Tests

Full CLI suite passed:

- `201 passed`

### Real full-contact-sheet rebuild on existing real analysis run

Completed successfully:

- `/Users/sfouasnon/Desktop/R3DMatch/runs/final_finish_line_contact_sheet/report/contact_sheet.json`
- `/Users/sfouasnon/Desktop/R3DMatch/runs/final_finish_line_contact_sheet/report/contact_sheet.html`
- `/Users/sfouasnon/Desktop/R3DMatch/runs/final_finish_line_contact_sheet/report/preview_contact_sheet.pdf`

### Fresh real 12-clip end-to-end run from source

Completed successfully:

- `/Users/sfouasnon/Desktop/R3DMatch/runs/final_finish_line_fresh/real_063_current_finish/report/contact_sheet.json`
- `/Users/sfouasnon/Desktop/R3DMatch/runs/final_finish_line_fresh/real_063_current_finish/report/contact_sheet.html`
- `/Users/sfouasnon/Desktop/R3DMatch/runs/final_finish_line_fresh/real_063_current_finish/report/preview_contact_sheet.pdf`
- `/Users/sfouasnon/Desktop/R3DMatch/runs/final_finish_line_fresh/real_063_current_finish/report/review_detection_summary.json`

Key proof points from the fresh run:

- `matching_domain = perceptual`
- `preview_transform = REDLine IPP2 / BT.709 / BT.1886 / Medium / Medium`
- `measurement_preview_transform = REDLine IPP2 / BT.709 / BT.1886 / Medium / Medium`
- contact sheet shows `Sample 1 / Sample 2 / Sample 3`
- contact sheet uses `./review_detection_overlays/...`

Most important no-duplicate-render proof:

- `preview_commands.json` for the fresh run contains `original_variants = 0`
- commands are `corrected` only

That means the report stage reused stored analysis originals and did not add a second baseline REDLine render pass.

## Remaining risk

If an older analysis directory predates the persisted rendered-measurement fields, the report path may still need to regenerate baseline measurement originals once in order to recover a valid IPP2 original reference. That is acceptable for legacy runs, but not expected for new runs created by the current code.

## Final state

R3DMatch now has:

- one IPP2-first measurement truth
- no normal Scene-Referred operator path
- no normal user ROI path
- no report-side re-measurement
- no duplicate baseline renders in the current full-contact-sheet workflow
- exact solve overlays on the contact sheet
- neutral `Sample 1 / Sample 2 / Sample 3` operator labeling
