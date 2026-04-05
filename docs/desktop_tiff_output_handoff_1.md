# Desktop TIFF / Workflow Refinement Handoff 1

## Scope

This pass combined:

- desktop UI / UX refinement
- TIFF preview-still support as the canonical default with JPEG still available
- workflow safety around subset / clip-group execution
- production-vs-debug output cleanup
- stronger web-style result summary restoration in the desktop app

This pass did **not** change:

- exposure math
- luminance / scalar / IRE semantics
- white-balance solve behavior
- scientific validation semantics
- RED SDK integration

## What Changed

### Desktop UI / workflow

- Removed the weak standalone `Progress` panel from the desktop workflow.
- Promoted `Live Processing Log` as the primary real-time execution surface.
- Moved runtime/environment concerns out of the primary workflow tab and into `Settings`.
- Kept REDLine persistence and runtime-health reuse intact.
- Added explicit clip-group selection after scan for local-folder workflows.
  - operators can now clearly see detected groups before `Run Review`
  - default remains all selected, but the selection is now visible and editable
- `Run Review` is blocked if local groups exist and none are selected.
- Review controls now include:
  - `Preview Still Format`: `TIFF` or `JPEG`
  - `Artifact Mode`: `Production` or `Debug`

### Desktop results summary restoration

- The desktop `Results` surface now reuses the stronger web-style operator summary surfaces instead of presenting artifacts as the primary result.
- Results now emphasize:
  - recommendation / decision summary
  - white-balance model summary
  - physical/scientific validation summary
  - operator-oriented follow-up actions
- Artifact paths remain available, but they are now secondary.

### Apply / Verify

- The desktop `Apply / Verify` tab now renders a clearer operational summary instead of a mostly empty placeholder.
- It surfaces:
  - commit payload presence
  - next-step guidance
  - push / apply / verification surface data when available

### Contact sheet / operator-report cleanup

- Overview tiles now use clearer operator language:
  - `Retained`
  - `Needs Attention`
  - `Excluded / Outlier`
  - `Exposure Anchor`
- Added legend support and directional exposure cues:
  - `↑ Lift`
  - `↓ Lower`
  - `≈ Hold`
- Removed weak or misleading labels from detailed sheets:
  - dropped `Profile consistent`
  - suppressed `Measured RGB` if not actually available
  - suppressed `Still stability` if not actually available
- Kept the surfaces scientifically honest when data is absent.

## TIFF Decision

### Decision

TIFF is now the canonical default preview still format for:

- measurement
- contact-sheet/report image generation
- rendered preview still flows that previously assumed JPEG

JPEG remains available as an explicit option.

### Implementation details

- Default preview-still format: `tiff`
- Optional alternate format: `jpeg`
- The preview format is now passed through CLI and desktop workflow controls.
- Provenance / measurement metadata explicitly records preview format and image characteristics.

### Image normalization

The preview image loader now normalizes based on actual image dtype / bit depth:

- `uint8` -> divide by `255.0`
- `uint16` -> divide by `65535.0`

TIFF loading now preserves higher bit depth during measurement/provenance handling.

### TIFF format confirmation

Confirmed from a real packaged run artifact:

- file format: `TIFF`
- mode: `RGB`
- compression: `raw`
- bit depth tags: `(16, 16, 16)`

### TIFF compression

TIFF compression was **not** enabled in this pass.

Reason:

- no safe locally confirmed REDLine TIFF compression CLI contract was established in this pass
- no invented REDLine flag was introduced

Current state:

- TIFF output is raw / uncompressed

## Production vs Debug Artifact Policy

### Production mode keeps

- operator review artifacts
- report HTML / PDF
- contact sheet assets needed for report delivery
- scientific validation artifacts
- replay / provenance truth needed for reproducibility at the intended level
- core analysis outputs required by downstream report/apply flows

### Production mode suppresses obvious debug-only artifacts

These are not emitted in production mode:

- `rmd_validation.json`
- `preview_semantics.json`
- `debug_exposure_trace/`
- `render_input_state.json`
- `pre_render_log_values.json`
- `post_render_ipp2_values.json`
- `render_trace_comparison.json`

### Debug mode

Debug mode preserves deeper diagnostic traces for engineering/debugging work.

## Validation Performed

### Automated tests

- `PYTHONPATH=src ./.venv/bin/python3.14 -m pytest -q tests/test_cli.py`
- Result: `261 passed`

Focused regression coverage includes:

- TIFF preview handling
- dtype-aware normalization
- clip-group selection behavior
- desktop review command format propagation
- results/apply desktop operator surfaces

### Packaged app rebuild

Build commands:

```bash
cd /Users/sfouasnon/Desktop/R3DMatch
RED_SDK_ROOT=/Users/sfouasnon/Desktop/R3DSplat_Dependecies/RED_SDK/R3DSDKv9_2_0 /bin/sh scripts/build_macos_app.sh

cd /Users/sfouasnon/Desktop/R3DMatch
/bin/sh scripts/package_macos_app.sh
```

Artifacts:

- `dist/R3DMatch.app`
- `dist/R3DMatch-macos-arm64.zip`

### Packaged desktop open

Validated packaged desktop startup with:

```bash
QT_QPA_PLATFORM=offscreen DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib dist/R3DMatch.app/Contents/MacOS/R3DMatch --desktop-smoke --desktop-smoke-ms 1200
```

Observed:

- `desktop_window_title=R3DMatch`
- `desktop_minimal_mode=False`

### Packaged runtime readiness

Validated packaged runtime health with:

```bash
DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib dist/R3DMatch.app/Contents/MacOS/R3DMatch --check
```

Observed:

- `html_pdf_ready=True`
- `red_sdk_runtime_ready=True`
- `redline_ready=True`
- `red_backend_ready=True`

### Real packaged review workflow

Executed a real packaged `.app` review run against clip-group `064`:

- run label: `qt_desktop_tiff_064_3`
- output root: `runs/desktop_tiff_validation/qt_desktop_tiff_064_3`

Observed:

- packaged process exited successfully
- `report/contact_sheet.html` present
- `report/preview_contact_sheet.pdf` present
- `report/review_validation.json` status = `success`

### TIFF end-to-end confirmation

Confirmed from the final packaged `064` run:

- measurement preview asset extension is `.tiff`
- stored diagnostics record:
  - `rendered_measurement_preview_format = tiff`
  - `rendered_measurement_image_dtype = uint16`
  - `rendered_measurement_bit_depth = 16`
- stored measurement provenance records:
  - `preview_format = tiff`
  - `pixel_dtype = uint16`
  - `bit_depth = 16`
  - `normalization_denominator = 65535.0`

## Scientific Validation Truth

No regression was introduced to scientific honesty.

For the final packaged run:

- `review_validation.json` = `success`
- `scientific_validation.json` = `blocked_asset_mismatch`

That status remains an honest replay/provenance truth surface and was not masked.

## Apple Verification / Notarization Feasibility

This was **not feasible to complete in a short local repo-only pass**.

Reason:

- Apple verification / notarization requires Developer ID signing
- notarization credentials / App Store Connect or notarytool setup are needed
- entitlements and hardened runtime signing may need to be finalized
- the build must be signed and submitted to Apple over networked notarization tooling

That cannot be honestly completed as a local code-only refinement without access to:

- the signing identity
- notarization credentials
- final packaging/signing policy
- network submission to Apple

No fake or partial “verified by Apple” claim should be made from this pass.

## Remaining Limitations / Follow-ups

- TIFF compression is still not enabled because no safe REDLine compression flag was confirmed in this pass.
- Scientific validation may still report `blocked_asset_mismatch` for replay-integrity reasons on specific runs; that remains an honest outcome.
- Full interactive GUI validation in this environment remains limited, so packaged desktop validation was performed via offscreen launch plus real packaged workflow execution.
