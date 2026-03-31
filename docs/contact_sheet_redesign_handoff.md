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

- one `.page` container per camera
- block layout plus inline-block only for row composition
- no CSS grid or flexbox in the authored contact-sheet page layout
- fixed four-part page structure:
  - `.header`
  - `.image-row`
  - `.metrics`
  - `.footer`

Each camera page renders exactly three report images:

- Original Frame
- Corrected Frame
- Sphere Mask Overlay

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

- Original Frame
- Corrected Frame
- Sphere Mask Overlay
- key measurement values
- correction values
- validation residual
- recommended action
- one white-balance deviation chart when persisted values are available

The first page/footer also carries:

- `Original Array Synopsis`
- `What To Look For`

Low-value clutter was removed instead of preserved.

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

## Validation completed

Code validation:

- `py_compile src/r3dmatch/report.py tests/test_cli.py`
- `221 passed` in `tests/test_cli.py`

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
