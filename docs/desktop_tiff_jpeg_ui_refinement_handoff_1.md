# Desktop TIFF/JPEG UI Refinement Handoff 1

## What changed

This pass focused on four things without changing calibration science:

1. Hardened preview ingestion so TIFF and JPEG previews are not read while REDLine is still finishing the write.
2. Kept TIFF as the canonical measurement still format while allowing JPEG as an operator-selectable review/output preview format.
3. Improved desktop review clarity with stronger web-style result summaries, clearer grouping, and more useful real-time feedback.
4. Completed packaged end-to-end validation for subset `067` in both TIFF and JPEG modes.

## Preview ingestion robustness

`report.py` now uses a retry-safe preview loader instead of a single immediate `Image.open(...).load()` attempt.

Added behavior:

- `_wait_for_file_ready(path)`
  - waits for the file to exist
  - waits for file size to stabilize
- `_load_preview_image_as_normalized_rgb(...)`
  - retries on truncated/partial image read failures
  - logs retry activity for debug visibility
  - does **not** enable `ImageFile.LOAD_TRUNCATED_IMAGES = True`

This keeps the measurement path scientifically honest. We do not silently accept a truncated file; we wait and retry until the file is actually ready or fail clearly.

## TIFF / JPEG decision

The workflow now treats preview formats like this:

- Measurement preview stills: TIFF
- Review preview stills: TIFF or JPEG, operator-selectable

This means:

- TIFF remains the authoritative measurement source
- JPEG can still be used for faster/lighter review output
- scientific provenance remains tied to the TIFF measurement asset

### Confirmed behavior from real packaged runs

TIFF run:

- report preview format: `tiff`
- measurement preview format: `tiff`
- measured preview asset format: TIFF
- measured preview dtype: `uint16`
- normalization denominator: `65535.0`

JPEG run:

- report preview format: `jpeg`
- measurement preview format: `tiff`
- measured preview asset format: TIFF
- measured preview dtype: `uint16`
- normalization denominator: `65535.0`

This confirms the intended split: JPEG is optional for review output, but measurement remains TIFF-first and provenance-safe.

## TIFF compression

TIFF compression was **not enabled** in this pass.

Reason:

- no REDLine TIFF compression flag was safely confirmed from local behavior in this pass
- no invented CLI flags were introduced

Observed TIFF output from the real packaged run:

- format: TIFF
- mode: RGB
- compression: `raw`
- measured source dtype: `uint16`

## Measurement normalization

The preview loader now normalizes according to the actual decoded image dtype:

- `uint8 -> 255.0`
- `uint16 -> 65535.0`

This was validated on real TIFF measurement previews from subset `067`.

## Desktop UI / UX refinement

### Main clarity changes

- removed the weak standalone `Progress` panel from primary emphasis
- promoted `Live Processing Log` as the main real-time execution surface
- kept runtime/config in a secondary settings surface instead of the main workflow
- preserved REDLine persistence and startup reload
- improved result presentation by restoring web-style summary surfaces in the desktop shell

### Results presentation improvements

Desktop results now emphasize:

- calibration recommendation
- WB model summary
- physical/scientific validation summary
- retained cluster vs excluded cameras
- hero / anchor emphasis
- side-by-side visual comparison
- per-camera monitoring log2, offset, and confidence

### Review clarity changes

- clip-group selection remains visible before review launch
- the operator can see the selected group (`067`) explicitly in the packaged workflow
- results use clearer directional exposure cues:
  - `↑ Lift`
  - `↓ Lower`
  - `≈ Hold`

### Report/contact sheet wording cleanup

This pass preserved the earlier truthfulness cleanup and validated that the generated report no longer shows weak placeholder text such as:

- `Profile consistent`
- `Measured RGB:` when that data is not actually available
- `Still stability:` when that data is not actually available

## Production vs debug artifact policy

This pass validated packaged review runs in `artifact-mode production`.

Production output still keeps the artifacts required for:

- operator review
- PDF/HTML report delivery
- scientific validation
- replay/provenance integrity
- commit/apply payload generation

Examples kept in production:

- `analysis/*.analysis.json`
- `report/contact_sheet.html`
- `report/preview_contact_sheet.pdf`
- `report/review_validation.json`
- `report/scientific_validation.json`
- `report/scientific_validation.md`
- `report/calibration_commit_payload.json`
- `review_rmd/*.RMD`

This pass did **not** remove scientific truth artifacts.

## Validation performed

### Automated validation

- `py_compile` passed for modified Python files
- focused regressions passed:
  - retry-safe preview loading
  - TIFF loading / dtype handling
  - desktop results summary surface
  - clip-group selection wiring

### Packaged app validation

Packaged app open:

- opened via desktop smoke mode
- `desktop_window_title=R3DMatch`
- `desktop_minimal_mode=False`

Packaged runtime health:

- `html_pdf_ready=True`
- `red_sdk_runtime_ready=True`
- `redline_ready=True`
- `red_backend_ready=True`

### Real packaged review runs

Completed successfully from the packaged app binary:

- TIFF run:
  - `/Users/sfouasnon/Desktop/R3DMatch/runs/qt_desktop_067_tiff_1/subset_067/report/contact_sheet.html`
  - `/Users/sfouasnon/Desktop/R3DMatch/runs/qt_desktop_067_tiff_1/subset_067/report/preview_contact_sheet.pdf`
- JPEG run:
  - `/Users/sfouasnon/Desktop/R3DMatch/runs/qt_desktop_067_jpeg_1/subset_067/report/contact_sheet.html`
  - `/Users/sfouasnon/Desktop/R3DMatch/runs/qt_desktop_067_jpeg_1/subset_067/report/preview_contact_sheet.pdf`

Both runs reported:

- `review_validation.json -> status = success`
- `scientific_validation.json` present

### Truncation failure validation

Searches across both real run outputs found no truncated-image errors.

## Remaining limitations

- TIFF compression is still unconfirmed and therefore still disabled.
- In the successful subset `067` validation runs there were no excluded cameras, so the report did not naturally render an excluded-camera list for that specific run.
- Production mode still retains a meaningful scientific/report payload set; this pass did not aggressively prune those because reproducibility and scientific traceability take priority over minimalism.
