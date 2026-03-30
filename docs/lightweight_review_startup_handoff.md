# Lightweight Review Startup Handoff

## Root Cause

The apparent lightweight-run stall was an orchestration visibility failure in the early pure-Python pipeline, not a REDLine crash.

- `review_calibration()` called `analyze_path()` before any report artifact existed.
- `analyze_path()` measured every selected clip before writing the first durable output.
- On slower machines or heavier runs, that created a long CPU-bound window with:
  - no `REDLine` child yet
  - no `.analysis.json` yet
  - no `review_validation.json` yet
  - no progress file

That made a healthy run look dead.

## Fix

Added an explicit startup progress channel:

- new helper: [`src/r3dmatch/progress.py`](/Users/sfouasnon/Desktop/R3DMatch/src/r3dmatch/progress.py)
- workflow emits early review checkpoints before analysis and before report build
- `analyze_path()` emits per-stage and per-clip checkpoints immediately
- lightweight and full report builders emit report-stage checkpoints
- full contact-sheet flow now emits probe/solver progress through the heavy pre-render stage
- web status now reads `review_progress.json` before analysis artifacts exist

## Progress Contract

Runs now create `/review_progress.json` immediately and keep it current.

Important phases include:

- `review_start`
- `analysis_dispatch`
- `analysis_start`
- `source_scan_complete`
- `subset_resolved`
- `measurement_start`
- `clip_measurement_start`
- `clip_measurement_complete`
- `analysis_finalize_start`
- `analysis_complete`
- `report_build_start`

Full contact-sheet runs also emit:

- `contact_report_real_redline`
- `real_redline_validation_start`
- `real_redline_validation_probe`
- `real_redline_validation_complete`
- `closed_loop_*`

## Runtime Evidence

Lightweight real 063 probe:

- run root: [`runs/lightweight_stall_probe_after/after_fix`](/Users/sfouasnon/Desktop/R3DMatch/runs/lightweight_stall_probe_after/after_fix)
- first progress artifact/write: effectively immediate
- first clip start: about `+0.02s`
- first clip complete: about `+0.73s`
- analysis complete: about `+8.36s`
- report complete: about `+8.41s`

Full contact-sheet probe:

- run root: [`runs/full_contact_sheet_probe_final/after_fix_full_progress`](/Users/sfouasnon/Desktop/R3DMatch/runs/full_contact_sheet_probe_final/after_fix_full_progress)
- first progress artifact/write: effectively immediate
- clip-level analysis progress starts at about `+0.02s`
- real REDLine preflight is now visible per clip
- preview generation becomes visibly active instead of appearing frozen

## Safety

No solve math, sphere model, REDLine correction semantics, or validation thresholds were changed in this pass.

Validated:

- `py_compile`
- [`tests/test_cli.py`](/Users/sfouasnon/Desktop/R3DMatch/tests/test_cli.py)
- result: `198 passed`

## Remaining Non-Blocking Risk

The full contact-sheet path is still heavier than lightweight mode because it performs real REDLine validation, closed-loop refinement, and preview rendering. That is expected work, but it is now observable rather than silent.
