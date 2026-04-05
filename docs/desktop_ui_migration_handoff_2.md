## Desktop UI Migration Handoff 2

Date: 2026-04-02

### Scope

This pass focused on desktop-shell stability and operator UX recovery, not calibration science.

Goals:
- fix the first-use desktop failure path
- recover the strongest conceptual qualities of the older web UI
- restore branding, hierarchy, and operator readability
- rebuild and reopen the packaged macOS app
- rerun packaged set 063 validation after the fixes

### Crash Root Cause

The first-use review-launch failure was caused by a desktop UI / CLI contract mismatch.

In `src/r3dmatch/desktop_app.py`, the matching-domain control exposed:
- `display`
- `scene`

But the backend CLI accepts:
- `perceptual`
- `scene`

The desktop shell was passing the combo-box display text directly into:

`--matching-domain display`

That caused the packaged review launch to fail with:

`Invalid value: matching domain must be scene or perceptual`

This was not a solve, report, or RED runtime bug. It was a desktop-shell parameter mapping bug.

### Stability Fixes

Implemented in `src/r3dmatch/desktop_app.py`:

- mapped UI label `Display (IPP2 / BT.709 / BT.1886)` to backend value `perceptual`
- preserved `Scene` -> `scene`
- switched command building to use combo-box item data instead of visible label text
- applied the same data-driven approach to preview-mode launch wiring
- added small signal / state guards so minimal or uninitialized UI states do not crash:
  - guarded reference clip wiring in minimal mode
  - guarded progress-surface writes
  - tightened run-button enablement for local vs FTPS source modes

### UX / Layout Recovery

The desktop shell was reworked to feel closer to the stronger web app information architecture:

- branded top header with logo, title, subtitle, workflow copy, and status chips
- clearer top-level flow:
  - Source & Ingest
  - Review Setup
  - Run & Progress
  - Runtime & Configuration
  - Results & Artifacts
  - Apply & Verify
- stronger operator guidance when outputs do not exist yet
- results and apply panels now explain next steps instead of showing empty placeholder surfaces
- runtime health is grouped and readable instead of feeling like a dumped text block

### Branding / Visual Language

The desktop shell now restores visible R3DMatch identity:

- logo restored in the header
- branded title and workflow subtitle restored
- light palette / stylesheet added for a cleaner operator-facing look
- status chips distinguish:
  - RED SDK Runtime
  - REDLine
  - HTML/PDF
  - runtime context

The goal of this pass was to reduce the prior flat gray “utility” feel without introducing decorative noise.

### Report / Surface Continuity

This pass did not change calibration science.

It preserved the previously improved operator/report surfaces and revalidated them in the fresh packaged 063 run, including:
- `Exposure Anchor: Median of group`
- `Exposure Anchors`
- `Original cast`
- clearer legend text in the overview page

### Tests Added

`tests/test_cli.py`

Added desktop-specific regressions:
- desktop matching-domain label maps to CLI value `perceptual`
- results surface explains missing outputs instead of remaining inert

### Validation

#### Python / test validation

- `py_compile` passed
- focused desktop/runtime regressions passed
- full `tests/test_cli.py` suite passed

#### Packaged rebuild

Rebuilt:
- `dist/R3DMatch.app`
- `dist/R3DMatch-macos-arm64.zip`

Timestamp evidence after this pass:
- `build/macos_app/spec/R3DMatch.spec` -> 2026-04-02 19:42:36
- `dist/R3DMatch.app` -> 2026-04-02 19:43:35
- `dist/R3DMatch-macos-arm64.zip` -> 2026-04-02 19:44:06

#### Packaged app open

Opened the rebuilt packaged app in offscreen desktop-smoke mode:

`QT_QPA_PLATFORM=offscreen dist/R3DMatch.app/Contents/MacOS/R3DMatch --desktop-smoke --desktop-smoke-ms 1200`

Observed:
- `desktop_window_title=R3DMatch`
- `desktop_minimal_mode=False`

#### Packaged runtime check

Validated packaged runtime directly:

- `html_pdf_ready=True`
- `red_sdk_runtime_ready=True`
- `redline_ready=True`
- `red_backend_ready=True`

#### Packaged 063 rerun

Fresh packaged validation run:
- run label: `qt_desktop_063_3`
- output root: `runs/desktop_qt_validation/qt_desktop_063_3`

Results:
- process exited `0`
- `report/contact_sheet.html` exists
- `report/preview_contact_sheet.pdf` exists
- `report/review_validation.json` status = `success`

Scientific validation remained truthful:
- `report/scientific_validation.json` status = `blocked_asset_mismatch`

That replay-integrity state is not caused by this desktop pass and was not masked.

### Remaining Limitations

- Full on-screen GUI interaction could not be visually exercised in this sandbox; packaged validation used the strongest feasible combination of:
  - rebuilt packaged desktop open
  - packaged runtime check
  - packaged end-to-end set 063 review run
- REDLine remains an external tool dependency
- scientific validation may still report archived/replay asset provenance issues where appropriate
