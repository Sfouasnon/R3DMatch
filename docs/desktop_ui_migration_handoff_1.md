# Desktop UI Migration Handoff 1

## Scope

This pass moved R3DMatch from a web-first operator shell to a PySide6 desktop-first shell while reusing the existing backend logic for:

- runtime health
- RED SDK runtime checks
- REDLine configuration persistence and resolution
- review workflow launch
- FTPS ingest
- report generation
- scientific validation
- apply / verify operations

The Flask/web shell remains in the repo as a secondary path and can still be launched with `--web`, but the packaged app now launches the Qt desktop shell by default.

## Desktop Shell

Primary desktop entrypoint:

- `src/r3dmatch/desktop_app.py`

Packaged launcher:

- `src/r3dmatch/web_launcher.py`

Desktop UI surfaces now include:

- source selection
- runtime/environment health
- REDLine configuration
- review controls
- progress / execution log
- artifacts / result actions
- apply / verify actions

Execution remains responsive through `QProcess`, with backend work delegated to the existing CLI/workflow paths.

## Shared Backend Reuse

Shared runtime/config helpers were consolidated in:

- `src/r3dmatch/runtime_env.py`

The desktop and web shells now both reuse:

- `runtime_cli_prefix()`
- `runtime_subprocess_env(...)`
- `persist_redline_configured_path(...)`

This avoids duplicating launch logic, runtime env handling, or REDLine config persistence.

## Report / Operator Surface Changes

### Exposure chart

The exposure spread chart in `src/r3dmatch/report.py` is now driven from display-domain per-camera values:

- prefers `display_scalar_ire`
- falls back to stored log2 scalar converted to IRE only when needed

Behavior:

- suppresses the chart when spread is analytically flat
- tightens the retained-cluster band
- reduces label clutter for larger camera counts
- uses operator-facing display-domain IRE language

### Original WB evaluation

The per-camera `Original WB Evaluation` block now prefers derived original-still context when it exists and keeps as-shot metadata as context, not as the primary truth.

Current output lines include:

- `Derived from`
- `Original cast`
- `As-shot WB`
- `Measured RGB`
- `Still stability`
- `Array model`

### Overview cards and labels

Operator-facing language was clarified:

- `Anchors / References` -> `Exposure Anchors`

The overview now includes an explicit legend for:

- retained
- borderline
- outlier / excluded
- exposure anchor

### Header cleanup

The contact-sheet HTML header now renders:

- `Exposure Anchor: Median of group`

instead of the duplicated:

- `Exposure Anchor: Exposure Anchor: Median of group`

## Packaged Build / Rebuild

Build command:

```bash
cd /Users/sfouasnon/Desktop/R3DMatch
RED_SDK_ROOT=/Users/sfouasnon/Desktop/R3DSplat_Dependecies/RED_SDK/R3DSDKv9_2_0 /bin/sh scripts/build_macos_app.sh
```

Package command:

```bash
cd /Users/sfouasnon/Desktop/R3DMatch
/bin/sh scripts/package_macos_app.sh
```

Artifacts:

- `dist/R3DMatch.app`
- `dist/R3DMatch-macos-arm64.zip`

## Packaged Validation

### Desktop open proof

The rebuilt packaged app was opened directly with the packaged executable in desktop smoke mode:

```bash
QT_QPA_PLATFORM=offscreen /Users/sfouasnon/Desktop/R3DMatch/dist/R3DMatch.app/Contents/MacOS/R3DMatch --desktop-smoke --desktop-smoke-ms 900
```

Observed output:

- `desktop_window_title=R3DMatch`
- `desktop_minimal_mode=False`

### Real packaged end-to-end review

A fresh numbered packaged run was completed from the rebuilt app:

- run label: `qt_desktop_063_2`
- output: `runs/desktop_qt_validation/qt_desktop_063_2`

Invocation:

```bash
DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib /Users/sfouasnon/Desktop/R3DMatch/dist/R3DMatch.app/Contents/MacOS/R3DMatch --cli review-calibration /Users/sfouasnon/Desktop/R3DMatch_Calibration/Test_Footage --out /Users/sfouasnon/Desktop/R3DMatch/runs/desktop_qt_validation --run-label qt_desktop_063_2 --backend red --target-type gray_sphere --processing-mode both --review-mode full_contact_sheet --report-focus auto --matching-domain perceptual --preview-mode monitoring --clip-group 063 --target-strategy median --target-strategy optimal-exposure
```

Result:

- process exited `0`
- `report/contact_sheet.html` exists
- `report/preview_contact_sheet.pdf` exists
- `report/review_validation.json` status = `success`

Scientific validation for this run remains truthful:

- `scientific_validation.status = blocked_asset_mismatch`

That is expected under the existing replay-integrity model when stored/replayed asset provenance cannot fully reconcile.

## Tests / Validation

Validated in this pass:

- `py_compile` on desktop/runtime/report/test files
- focused desktop/report regression tests
- full CLI suite

Latest results:

- focused tests: `7 passed`
- full suite: `253 passed`

## Remaining Limitations

- The web shell still exists, but only as a secondary/debug path.
- Some clips still only have metadata-context original WB evaluation because richer original-still chromaticity trace was not stored for those payloads.
- Scientific validation may still report `blocked_asset_mismatch` for runs where replay provenance cannot be fully reconciled.
