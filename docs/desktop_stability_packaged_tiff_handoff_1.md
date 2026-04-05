# Desktop Stability and Packaged TIFF Validation Handoff

## Summary

This pass focused on stability, workflow clarity, production artifact hygiene, and proof of real packaged TIFF review execution.

The main runtime failure was not a hidden TIFF parser bug in the final packaged run. The immediate packaged `067` failure came from the measurement-side render path still using one-shot REDLine output handling, and then a second unrecovered failure was traced to low disk space inside the project run tree. Once the measurement render path was hardened and space was restored by removing old generated validation outputs inside `R3DMatch/runs`, the packaged TIFF review workflow completed successfully.

## What Changed

### 1. Measurement-side TIFF render stability

Updated [matching.py](/Users/sfouasnon/Desktop/R3DMatch/src/r3dmatch/matching.py) so rendered-preview measurement no longer trusts a single `render_preview_frame(...)` call plus `exists()`.

It now reuses the report-side validated render helper:

- `_render_preview_frame_with_retries(...)`
- bounded retry count for measurement renders
- explicit output readability validation before measurement
- attempt diagnostics persisted into measurement provenance and diagnostics

New persisted diagnostics include:

- `rendered_measurement_attempt_count`
- `rendered_measurement_recovered_after_retry`
- `rendered_measurement_attempt_diagnostics`
- matching provenance fields inside `measurement_provenance.render_identity`

This preserves honesty while preventing transient half-written TIFFs from crashing analysis immediately.

### 2. Desktop workflow cleanup

Existing desktop changes were preserved and validated:

- tab order remains operator-first:
  - Review
  - Results
  - Apply / Verify
  - Settings
- `Live Processing Log` remains scoped to the review/run surface instead of repeating globally
- successful runs continue to switch operators to the Results tab automatically

### 3. Artifact hygiene policy

Production artifact policy remains explicit and was validated in a real packaged run:

- `sidecars`
  - production-required
  - kept because they remain the canonical per-camera analysis payload for reproducibility, validation, and downstream apply/transcode paths

- `review_rmd`
  - production keeps strategy preview RMDs only
  - this remains necessary because review stills are rendered from these strategy RMDs and they support reproducibility of the rendered review package

- `rmd_compare`
  - debug-only
  - not emitted in default production runs

In the validated production run:

- top-level clip-level `review_rmd/*.RMD` exports were skipped
- strategy-scoped `review_rmd/strategies/...` and closed-loop strategy RMDs remained

### 4. Sphere-analysis reliability

No new sphere-model rewrite was introduced.

The previously hardened full-frame sphere detection path remained in place and was revalidated on the known difficult right-edge cameras in the new packaged run:

- `H007_C067_0403RE_001`
- `H007_D067_0403YU_001`

The overlays in the final packaged run still land cleanly on the gray sphere.

## Root Cause Notes

### Initial packaged TIFF failure

The first current-pass packaged failure occurred because measurement preview rendering in `matching.py` still used:

- one-shot REDLine render
- file existence check only

That path was weaker than the already-hardened preview generation path in `report.py`.

### Second failure after retry hardening

The later unrecovered TIFF write failure on `G007_C067_040399_001` was traced to project disk exhaustion, not a new clip-specific science failure.

Observed evidence:

- REDLine stderr: `TIFFAppendToStrip: Write error at scanline 0.`
- filesystem availability before cleanup: about `153 MiB`
- old generated validation directory under `runs/packaged_detector_validation`: about `23 GiB`

To complete the proof run, old generated validation outputs inside the project were removed:

- `runs/packaged_detector_validation`
- transient probe output
- failed partial packaged stability run

After cleanup, free space returned to about `23 GiB`, and the packaged TIFF run completed successfully.

## Validation Performed

### Focused automated validation

Ran targeted regressions covering:

- rendered-preview measurement retry handling
- report-side preview retry handling
- off-edge sphere detection cases
- artifact policy behavior
- desktop tab ordering

Command:

```bash
PYTHONPATH=src PYTHONPYCACHEPREFIX=/tmp ./.venv/bin/pytest -q tests/test_cli.py -k "far_right_edge_sphere_when_roi_missing or prefers_bright_edge_sphere_over_darker_false_circle or build_review_package_debug_keeps_clip_level_review_rmd_and_rcx_compare or desktop_window_tab_order_matches_operator_workflow or rendered_preview_measurement_retries_missing_tiff_and_records_recovery or uses_rendered_preview_ipp2_for_lightweight_measurement"
```

Result:

- `6 passed`

Also validated earlier focused retry/desktop cases:

- `5 passed`

### Compile validation

`py_compile` passed for:

- [matching.py](/Users/sfouasnon/Desktop/R3DMatch/src/r3dmatch/matching.py)
- [report.py](/Users/sfouasnon/Desktop/R3DMatch/src/r3dmatch/report.py)
- [desktop_app.py](/Users/sfouasnon/Desktop/R3DMatch/src/r3dmatch/desktop_app.py)
- [test_cli.py](/Users/sfouasnon/Desktop/R3DMatch/tests/test_cli.py)

### Packaged app validation

Rebuilt app:

```bash
RED_SDK_ROOT=/Users/sfouasnon/Desktop/R3DSplat_Dependecies/RED_SDK/R3DSDKv9_2_0 /bin/sh scripts/build_macos_app.sh
```

Repackaged zip:

```bash
/bin/sh scripts/package_macos_app.sh
```

Bundle checks:

- `spctl --assess --verbose=4 dist/R3DMatch.app` => `accepted`
- direct packaged smoke launch succeeded
- `open dist/R3DMatch.app` succeeded
- packaged `--check` output confirmed:
  - `html_pdf_ready=True`
  - `red_sdk_runtime_ready=True`
  - `redline_ready=True`
  - `red_backend_ready=True`

### Real packaged TIFF review proof

Real packaged executable used:

- [R3DMatch](/Users/sfouasnon/Desktop/R3DMatch/dist/R3DMatch.app/Contents/MacOS/R3DMatch)

Successful run:

- run dir: [qt_packaged_067_stability_4](/Users/sfouasnon/Desktop/R3DMatch/runs/packaged_stability_validation/qt_packaged_067_stability_4)

Generated artifacts:

- [contact_sheet.html](/Users/sfouasnon/Desktop/R3DMatch/runs/packaged_stability_validation/qt_packaged_067_stability_4/report/contact_sheet.html)
- [preview_contact_sheet.pdf](/Users/sfouasnon/Desktop/R3DMatch/runs/packaged_stability_validation/qt_packaged_067_stability_4/report/preview_contact_sheet.pdf)
- [review_validation.json](/Users/sfouasnon/Desktop/R3DMatch/runs/packaged_stability_validation/qt_packaged_067_stability_4/report/review_validation.json)

Validation outcome:

- `review_validation.status = success`
- TIFF workflow completed end-to-end in packaged context

## Final Artifact Policy Decision

### `review_rmd`

- required in production at the strategy level only
- not required as duplicate top-level clip exports in production
- kept under `review_rmd/strategies/...` and closed-loop strategy folders because preview rendering and reproducibility depend on them

### `rmd_compare`

- debug-only
- not part of default production output

### `sidecars`

- production-required
- retained because they are still the authoritative per-camera analysis artifact used by validation and downstream operations

## Remaining Limitations

- The real blocker that surfaced mid-pass was disk capacity, not just render timing. Very large accumulated run directories can still starve REDLine of space and cause header-only TIFF outputs with write errors.
- Production output is cleaner than before, but strategy-scoped RMDs still remain sizeable because they are functionally required for reproducible review rendering.
- The earlier full test suite had passed before this final narrow measurement retry change, but this pass’s final automated validation was focused on the changed seams plus real packaged execution rather than another full overnight-style suite run.

