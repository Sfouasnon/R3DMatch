# Results Binding + Gray Target Reliability Handoff

## Scope

This pass focused on:

1. fixing desktop Results binding so completed review packages are recognized reliably
2. cleaning the Review / Results responsibility split in the desktop UI
3. preserving the stronger page-1 contact-sheet summary and tighter WB layout
4. keeping chromaticity presentation operator-readable
5. hardening gray-target acquisition with sphere-first, gray-card fallback, and a mixed-target-class guardrail
6. proving a real packaged TIFF review run again

## What Changed

### Results binding / desktop UI

- The desktop app now resolves a selected output root to the latest nested completed review package instead of assuming the selected folder is already the run root.
- The Results loader now merges `review_package.json` with `contact_sheet.json` so older or sparse package manifests do not blank out recommendation, WB, or gray-target summaries.
- Review tab copy was reduced to operational setup guidance: source, clip groups, run controls, and the live run surface.
- Results remains the post-run destination for recommendation, attention list, artifact actions, and gray-target consistency.

### Contact-sheet / report payload flow

- `build_contact_sheet_report()` now returns the same high-level summary fields that were already written into `contact_sheet.json`:
  - `recommended_strategy`
  - `hero_recommendation`
  - `white_balance_model`
  - `run_assessment`
  - `gray_target_consistency`
  - `operator_recommendation`
- `build_review_package()` now carries those values into `review_package.json`, which lets the desktop Results surface bind against the packaged run truth instead of falling back to placeholders.

### Gray-target reliability

- Gray-sample acquisition remains sphere-first.
- If the sphere solve is weak or unresolved, the report path attempts gray-card fallback.
- The system persists:
  - chosen target class
  - detection method
  - confidence
  - whether fallback was used
  - whether extra review is recommended
- Mixed retained target classes are treated as a consistency problem:
  - the dominant class is computed
  - non-dominant retained cameras are excluded from safe commit export
  - run assessment is blocked from safe push
  - the warning is surfaced in run assessment and the desktop/operator summary

## OpenCV

OpenCV was **not** added in this pass.

The gray-card fallback uses the existing image-processing path already in the repo. This was enough to implement the requested sphere-first / gray-card-fallback hierarchy without introducing a second detection stack.

## Validation Performed

### Focused tests

Focused regressions passed:

- Results nested-run discovery
- Results payload merge for sparse `review_package.json`
- gray-card fallback from weak sphere solves
- mixed gray-target consistency guardrail
- desktop operator summary rendering
- review package summary-field persistence

Command used:

```bash
cd /Users/sfouasnon/Desktop/R3DMatch
PYTHONPATH=src /Users/sfouasnon/Desktop/R3DSplat/.venv39validate/bin/python -m pytest -q tests/test_cli.py -k 'load_operator_surface_context_prefers_contact_sheet_summary_when_review_package_is_sparse or results_surface_discovers_latest_completed_run_in_selected_output_root or build_review_package or report_contact_sheet_scaffold or falls_back_to_gray_card_when_sphere_is_unresolved or gray_target_consistency_summary_blocks_mixed_retained_target_classes or desktop_window_results_and_apply_surfaces_use_operator_summary' -o cache_dir=/tmp/r3dmatch_pytest_cache
```

Result: `8 passed`

### Compile check

`py_compile` passed for:

- `src/r3dmatch/report.py`
- `src/r3dmatch/desktop_app.py`
- `src/r3dmatch/workflow.py`
- `src/r3dmatch/matching.py`
- `tests/test_cli.py`

### Packaged app rebuild

Commands used:

```bash
cd /Users/sfouasnon/Desktop/R3DMatch
RED_SDK_ROOT=/Users/sfouasnon/Desktop/R3DSplat_Dependecies/RED_SDK/R3DSDKv9_2_0 /bin/sh scripts/build_macos_app.sh

cd /Users/sfouasnon/Desktop/R3DMatch
/bin/sh scripts/package_macos_app.sh
```

Artifacts:

- `dist/R3DMatch.app`
- `dist/R3DMatch-macos-arm64.zip`

### Packaged desktop validation

Packaged desktop smoke:

```bash
QT_QPA_PLATFORM=offscreen DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib /Users/sfouasnon/Desktop/R3DMatch/dist/R3DMatch.app/Contents/MacOS/R3DMatch --desktop-smoke --desktop-smoke-ms 1000
```

Observed:

- `desktop_window_title=R3DMatch`
- `desktop_minimal_mode=False`

Packaged readiness check:

```bash
DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib /Users/sfouasnon/Desktop/R3DMatch/dist/R3DMatch.app/Contents/MacOS/R3DMatch --check
```

Observed:

- `html_pdf_ready=True`
- `red_sdk_runtime_ready=True`
- `redline_ready=True`
- `red_backend_ready=True`

### Real packaged TIFF review proof

The first rerun attempt failed honestly with `OSError: [Errno 28] No space left on device` while duplicating TIFF previews for the `both` variant. This was an environment/storage failure, not a logic failure.

To recover, old generated validation run folders inside `R3DMatch/runs` were removed:

- `runs/packaged_results_binding_validation`
- `runs/packaged_stability_validation`

That restored about `35 GiB` free, and the packaged TIFF review was rerun successfully.

Successful final packaged run:

- run root: `/Users/sfouasnon/Desktop/R3DMatch/runs/packaged_results_binding_validation/qt_packaged_067_results_1`
- HTML: `/Users/sfouasnon/Desktop/R3DMatch/runs/packaged_results_binding_validation/qt_packaged_067_results_1/report/contact_sheet.html`
- PDF: `/Users/sfouasnon/Desktop/R3DMatch/runs/packaged_results_binding_validation/qt_packaged_067_results_1/report/preview_contact_sheet.pdf`
- review validation: `/Users/sfouasnon/Desktop/R3DMatch/runs/packaged_results_binding_validation/qt_packaged_067_results_1/report/review_validation.json`
- package manifest: `/Users/sfouasnon/Desktop/R3DMatch/runs/packaged_results_binding_validation/qt_packaged_067_results_1/report/review_package.json`

Final outcome:

- packaged run exited successfully
- HTML exists
- PDF exists
- `review_validation.status = success`
- `review_package.json` now carries recommendation / WB / gray-target summary fields correctly
- gray-target consistency for the successful run is:
  - dominant target class: `sphere`
  - mixed target classes: `false`
  - summary: `Retained cameras consistently measured the gray sphere.`

## Remaining Limitations

- `review_validation.json` still keeps most of the higher-level summary data nested under `run_assessment` rather than duplicating every summary field at the top level. That is truthful and currently sufficient, but it is a slightly different shape from `contact_sheet.json` and `review_package.json`.
- The packaged TIFF workflow is now stable in logic, but storage pressure remains an operational risk because preview TIFFs are large.
- OpenCV was not added; if future gray-card stress cases exceed the current fallback path, that would be the next escalation step.
