# Contact Sheet Finalization Handoff

## Final render path

The contact sheet now has one authored layout path:

1. `src/r3dmatch/report.py` → `build_contact_sheet_report(...)`
2. `src/r3dmatch/report.py` → `render_contact_sheet_html(...)`
3. `src/r3dmatch/report.py` → `render_contact_sheet_pdf_from_html(...)`

`contact_sheet.html` is the source of truth. `preview_contact_sheet.pdf` is generated only from that HTML.

There is no parallel custom contact-sheet PDF compositor path.

## HTML / PDF layout contract

The final contact sheet uses a deterministic page system built for WeasyPrint:

- one `.page` container per report page
- block layout plus inline-block only for row composition
- no CSS grid or flexbox in the authored contact-sheet page layout
- overview pages render before detailed per-camera pages for larger arrays

Detailed camera pages now use a denser two-panel image row:

- Original + Solve Overlay
- Corrected Frame

The standalone overlay panel was removed. The overlay now does real verification work in the before/after comparison row instead of occupying a separate third panel.

## Large-array review behavior

Large arrays now switch to overview-first review automatically.

- threshold: `18` cameras
- default focus for large arrays: `outliers`
- explicit focus modes:
  - `auto`
  - `full`
  - `outliers`
  - `anchors`
  - `cluster_extremes`

The overview layer now summarizes:

- camera count
- excluded count
- outlier count
- center IRE range
- median residual
- default detail focus
- recommended attention cameras
- outliers / excluded cameras
- anchors / references

Overview tiles are sorted in physical-ish camera-label order and show:

- camera label
- PASS / REVIEW state
- residual
- scalar IRE
- anchor / excluded context

Detailed pages are then filtered according to the effective report focus instead of always exporting every camera first.

## Authoritative displayed values

The report now uses one authoritative source order for displayed sample values:

1. `ipp2_validation`
2. strategy clip payload
3. stored exposure metrics
4. shared original payload

The report does not average or recompute sample values.

Displayed values are:

- `S1`
- `S2`
- `S3`
- `Target Sample`
- `Scalar`
- `exposureAdjust`
- `kelvin`
- `tint`
- validation residual

The detailed metrics are grouped as:

- Gray Exposure
- Original WB Evaluation
- Correction
- Verification

Important semantic rule:

- `S1` / `S2` / `S3` are monitoring-domain IRE values from stored payloads
- `Scalar` is displayed as log2 and is not relabeled as IRE

## Fail-fast behavior

The contact-sheet build now fails before emitting deliverables when required report truth is missing.

Validated before or during HTML emission:

- original still exists
- corrected still exists
- sphere mask overlay exists
- generated `<img src>` references resolve relative to `contact_sheet.html`
- required stored sample values exist in the authoritative payload chain
- report image outputs are successfully written under `report/images/`

Missing assets or missing required sample truth now raise a hard error.

## Report-local image set

The HTML no longer embeds full-resolution report stills directly.

Per report build it writes:

- `report/images/<camera>_original.jpg`
- `report/images/<camera>_corrected.jpg`
- `report/images/<camera>_mask.jpg`

Rules enforced during generation:

- resized before HTML emit
- max width target: 800 px
- JPEG quality: 82
- RGB / 8-bit output
- fail if a generated report image exceeds 1200 px width

This keeps HTML/PDF output smaller and more stable.

## Debug artifact

Each report writes:

- `report/contact_sheet_debug.json`

It includes:

- resolved asset paths per camera
- measurement values per camera
- validation status summary
- fallback-used flags

## Useful retained surfaces

The report keeps only decision-useful surfaces:

- Original + Solve Overlay
- Corrected Frame
- key measurement values
- original WB evaluation when truthfully available
- correction values
- validation residual
- recommended action
- one overview white-balance deviation chart when persisted values are available
- overview exposure summary

Low-value clutter was removed instead of preserved. Large-array exports no longer force an operator to read every full camera page before seeing outliers and anchors.

## White-balance deviation chart

The retained chart is based only on persisted values:

- original path: `original_kelvin`, `original_tint`
- target center: `5600K / Tint 0`

It now uses:

- larger target marker
- larger camera dots
- clearer warm/cool and green/magenta labeling
- collision-reduced label placement

If source values are unavailable, the chart is omitted and the footer notes that it is unavailable from the stored payload.

## UI label cleanup

The report actions now read:

- `Open Report (HTML)`
- `Export PDF`

The web review form now also exposes `Large-Array Export Focus`, which passes `--report-focus` through the review command so the operator can explicitly choose:

- `Auto`
- `All Cameras`
- `Outliers Only`
- `Anchors / References`
- `Cluster Extremes`

## Validation completed

Code validation:

- `py_compile src/r3dmatch/report.py tests/test_cli.py`
- `227 passed` in `tests/test_cli.py`

Real artifact probe:

- rendered from existing real payload under `runs/final_display_scalar_lock/real_063_display_scalar_lock/report/contact_sheet.json`
- output written to `/tmp/r3dmatch_final_contact_sheet_probe/`
- validates asset references relative to the emitted HTML
- uses the same HTML artifact for PDF export

## Integrity guarantees preserved

This pass did not change calibration science.

Still true:

- no report-side remeasurement
- no new sphere detection
- no new sample computation
- no duplicate baseline REDLine render
- HTML remains the report source of truth
