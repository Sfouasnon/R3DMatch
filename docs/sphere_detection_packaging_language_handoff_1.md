# Sphere Detection, Report Language, and macOS Bundle Pass

## What changed

This pass focused on four linked problems without changing exposure math, scalar/IRE semantics, WB solve behavior, or the TIFF measurement decision:

1. Sphere detection was made less center-biased and more evidence-driven.
2. Contact sheet / PDF wording was shifted toward operator-facing language.
3. The macOS `.app` bundle metadata and post-build signing were hardened so the bundle is recognized more like a real application wrapper.
4. The working TIFF measurement path and packaged runtime flow were preserved and revalidated.

## Sphere detection changes

Updated file:
- `src/r3dmatch/report.py`

### Problem

The previous detection scorer still explicitly rewarded circles near a preferred screen position, and the fallback search window was centered on a narrow interior region. That made off-center gray spheres more likely to fall into `secondary_detected` / low-confidence behavior even when a better candidate existed elsewhere in frame.

### Fixes

- Removed center-position scoring from `_evaluate_detected_sphere_roi(...)`.
- Expanded the secondary search to the full frame instead of a center-weighted sub-window.
- Increased the number of Hough peaks considered.
- Added candidate de-duplication so equivalent circles do not crowd out other real candidates.
- Scored candidates using image evidence instead of position:
  - edge strength
  - edge support / ring continuity
  - interior smoothness
  - neutral RGB balance
  - plausible brightness range for a gray target
  - zone containment
  - sane radius
- Preserved and expanded debug visibility by attaching `candidate_diagnostics` to the chosen detection result.

### Real validation result

On the fresh packaged `067` run:
- `H007_C067_0403RE_001`
  - `sphere_detection_source = primary_detected`
  - `sphere_detection_confidence = 0.5231267213821411`
  - `sphere_detection_label = MEDIUM`
  - `candidate_diagnostics.primary_candidates = 6`

This clip had previously been one of the weak detection examples during audit.

## Report language changes

Updated file:
- `src/r3dmatch/report.py`

### Direction

The goal was not to hide technical truth, but to stop exposing raw implementation language in operator-facing PDF/contact-sheet surfaces.

### Examples of wording improvements

- `Digital correction applied` -> `Exposure adjustment applied`
- `Reference Use` -> `Array role`
- `Validation state` -> `Result`
- `Flags` -> `Notes`
- `Scalar Source` -> `Measurement source`
- `Measured RGB` -> `Original neutral sample`
- `Still stability` -> `Sample consistency`
- `Derived from` -> `Evaluation basis`
- `As-shot metadata context` -> `Using camera capture settings`
- `Derived from stored WB solve` -> `Using stored white-balance analysis`
- `Sphere detection: verified` -> `Sphere check verified`
- `Sphere detection: fallback used` -> `Alternate detection path used`
- `Sphere detection: low-confidence recovery` -> `Alternate detection path used — confirm sphere placement`
- `Profile consistent` -> `Sample profile aligned`

### Preserved truth

- Internal payload `color_preview_status` values such as `disabled_unverified` still exist where they are part of stored machine-readable state.
- Operator surfaces now use `color_preview_operator_status` so the raw internal status is not shown directly in the contact sheet header.
- No measurement values or thresholds were changed to achieve the language cleanup.

## macOS packaging fixes

Updated file:
- `scripts/build_macos_app.sh`

### Fixes

- Added explicit PyInstaller macOS bundle identifier:
  - `com.r3dmatch.desktop`
- Post-processed `Contents/Info.plist` after bundle build to ensure:
  - `CFBundleExecutable = R3DMatch`
  - `CFBundlePackageType = APPL`
  - `CFBundleIdentifier = com.r3dmatch.desktop`
  - `CFBundleShortVersionString`
  - `CFBundleVersion`
  - `NSPrincipalClass = NSApplication`
  - `LSMinimumSystemVersion = 12.0`
  - `LSApplicationCategoryType = public.app-category.photography`
- Re-signed the final assembled app bundle with ad-hoc signing:
  - `codesign --force --deep --sign -`

### Result

The rebuilt bundle now reports:

- `CFBundleIdentifier = com.r3dmatch.desktop`
- `CFBundlePackageType = APPL`
- `Executable = .../Contents/MacOS/R3DMatch`
- `Identifier = com.r3dmatch.desktop`
- `Signature = adhoc`

And Gatekeeper assessment now reports:

- `spctl --assess --verbose=4 dist/R3DMatch.app`
- result: `accepted`

## TIFF / packaged runtime preservation

No changes were made that alter:

- TIFF as the canonical measurement preview still format
- dtype-aware normalization
- retry-safe preview loading
- REDLine path resolution inside the packaged app
- packaged desktop runtime health checks

The packaged runtime still reports:

- `html_pdf_ready=True`
- `red_sdk_runtime_ready=True`
- `redline_ready=True`
- `red_backend_ready=True`

## Validation performed

### Static / automated

- `sh -n scripts/build_macos_app.sh`
- `PYTHONPYCACHEPREFIX=/tmp .venv/bin/python -m py_compile src/r3dmatch/report.py tests/test_cli.py`
- Focused regression subset:
  - `21 passed`
- Full suite:
  - `263 passed`

### Bundle validation

- Rebuilt app:
  - `RED_SDK_ROOT=/Users/sfouasnon/Desktop/R3DSplat_Dependecies/RED_SDK/R3DSDKv9_2_0 /bin/sh scripts/build_macos_app.sh`
- Repackaged zip:
  - `/bin/sh scripts/package_macos_app.sh`
- Direct bundled executable check:
  - `dist/R3DMatch.app/Contents/MacOS/R3DMatch --check`
- Desktop smoke:
  - `QT_QPA_PLATFORM=offscreen dist/R3DMatch.app/Contents/MacOS/R3DMatch --desktop-smoke --desktop-smoke-ms 1200`
- Bundle metadata:
  - `plutil -p dist/R3DMatch.app/Contents/Info.plist`
  - `codesign -dv --verbose=4 dist/R3DMatch.app`
  - `spctl --assess --verbose=4 dist/R3DMatch.app`
- Bundle wrapper launch:
  - `open dist/R3DMatch.app`

### Real packaged review workflow

Successful packaged end-to-end run:
- output root:
  - `runs/packaged_detector_validation/qt_packaged_067_detector_2`
- produced:
  - `report/contact_sheet.html`
  - `report/preview_contact_sheet.pdf`
  - `report/review_validation.json`
  - `report/scientific_validation.json`

Validation result:
- `review_validation.status = success`
- `scientific_validation.status = blocked_asset_mismatch`

The scientific validation status remains honest and was not masked by this pass.

## Production artifact note

This run briefly hit a disk-space failure during a larger multi-strategy validation attempt. I removed only older generated validation outputs inside the project, then reran the packaged validation in a tighter footprint using a single target strategy. That rerun succeeded and produced the final PDF artifact.

## Rebuilt artifact evidence

- `build/macos_app/spec/R3DMatch.spec`
  - `2026-04-04 09:28:47`
- `dist/R3DMatch.app`
  - `2026-04-04 09:30:02`
- `dist/R3DMatch-macos-arm64.zip`
  - `2026-04-04 09:31:00`

## Remaining limitations

- I confirmed that `open dist/R3DMatch.app` works and that `spctl` now accepts the bundle, but I could not directly visually inspect the Finder icon state from within this sandbox. So:
  - bundle recognition is improved and validated by metadata + `open` + `spctl`
  - the prohibited Finder icon was not visually re-checked by eye in this environment
- The scientific validation artifact still reports `blocked_asset_mismatch` for replay-integrity reasons already known from prior work; this pass intentionally did not hide or rewrite that truth.
- The scikit-image deprecation warnings in `calibration.py` remain and were not part of this pass.

## Produced PDF

- `runs/packaged_detector_validation/qt_packaged_067_detector_2/report/preview_contact_sheet.pdf`
