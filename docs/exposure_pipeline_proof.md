# Exposure Pipeline Proof

This note traces the current exposure path from measurement to corrected contact-sheet render and records what is confirmed, what is only approximate, and what was fixed during the March 27, 2026 calibration-proof pass.

## Scope

This proof covers the review path used by `review-calibration`, especially:

- source measurement
- strategy target derivation
- correction computation
- corrected preview rendering
- contact-sheet assembly
- corrected-still residual validation

It does not claim automatic gray-sphere object detection where none exists.

## Code Path Trace

### 1. Frame measurement

Primary entry point:

- `/Users/sfouasnon/Desktop/R3DMatch/src/r3dmatch/matching.py`
  - `analyze_path(...)`
  - `analyze_clip(...)`
  - `measure_frame_color_and_exposure(...)`

`analyze_clip(...)` decodes sample frames, measures each frame, then stores median measurement diagnostics into each `*.analysis.json`.

### 2. ROI extraction

`measure_frame_color_and_exposure(...)` uses:

- `_extract_normalized_roi_region(...)` when a shared normalized ROI is provided
- otherwise `extract_center_region(..., fraction=0.4)`

For the review gray-sphere workflow, the current production path is a shared rectangular ROI, not a detected circular sphere mask.

### 3. Three-sample gray measurement

The exact sampling geometry lives in:

- `_neutral_sample_regions(...)`
- `_measure_three_sample_statistics(...)`

This is always three rectangular windows inside the ROI:

- left
- center
- right

Geometry:

- sample width = 24% of ROI width
- sample height = 28% of ROI height
- centers at 28%, 50%, and 72% of ROI width
- vertical center at 50% of ROI height

For each window:

- luminance is computed with `0.2126 R + 0.7152 G + 0.0722 B`
- valid pixels are trimmed to the 5th to 95th percentile
- the window luminance is reported as `median(log2(trimmed_luminance))`

The final gray exposure value is:

- the median of the three window log2 measurements

This confirms that “three samples” means three spatial ROI windows, not three separate detections, rings, or passes.

### 4. Monitoring vs scene measurement

`measure_frame_color_and_exposure(...)` measures two domains:

- raw/scene-style ROI statistics
- monitoring-domain ROI statistics after `_apply_monitoring_review_transform(...)`

The review payload stores both:

- `measured_log2_luminance_raw`
- `measured_log2_luminance_monitoring`

### 5. Strategy target derivation

Main logic:

- `/Users/sfouasnon/Desktop/R3DMatch/src/r3dmatch/report.py`
  - `_measurement_values_for_record(...)`
  - `_build_strategy_payloads(...)`

For `matching_domain="scene"`:

- strategy solve uses `measured_log2_luminance_raw`

For `matching_domain="perceptual"`:

- strategy solve uses rendered-preview remeasurement

### 6. Corrected render generation

Render path:

- `generate_preview_stills(...)`
- `render_preview_frame(...)`
- `_build_redline_preview_command(...)`

Corrected stills are generated from SDK-authored review RMDs and REDLine, not by simply relabeling originals.

### 7. Contact-sheet placement

Full report assembly:

- `build_contact_sheet_report(...)`
- `render_contact_sheet_html(...)`
- `render_contact_sheet_pdf(...)`

Corrected strategy tiles use `both_corrected` assets from the generated preview set.

## Target Values By Domain

## Log3G10 physical validation target

Defined in:

- `/Users/sfouasnon/Desktop/R3DMatch/src/r3dmatch/workflow.py`

Constant:

- `LOG3G10_MID_GRAY_CODE_VALUE = 0.3333`

This is the project’s explicit physical-validation expectation for encoded Log3G10 gray.

## Strategy solve target

The review/solve target is not a fixed `.33` in all cases.

It is strategy-derived:

- median target for `median`
- trusted best-gray-match camera for `optimal_exposure`
- manual or hero anchor when chosen

So there are two different truths:

- physical validation target in Log3G10 code value space: `0.3333`
- strategy target in the selected matching domain: `target_log2_luminance`

## Monitoring / preview target

The contact sheet is judged in the actual render domain used for the preview assets.

For corrected-still residual proof, the render-domain target is now operationally defined as:

- the median corrected render gray value across trusted corrected stills for that strategy

This avoids pretending that a transformed JPEG preview can be compared directly to the scene-domain target without conversion loss.

## Confirmed Truths

- The production review path really does sample three ROI windows.
- The current shared-ROI gray-sphere review path does not perform automatic sphere-presence verification.
- A successful corrected preview must change pixels when a non-identity correction is requested.
- Corrected stills are now remeasured with the same three-window geometry used by the solver.
- The contact sheet now carries per-image residual status instead of only pixel-diff proof.

## Confirmed Mismatches Found

### 1. Preview settings were not fully honored during render generation

Before this pass, `generate_preview_stills(...)` rebuilt preview settings from the monitoring/display defaults instead of honoring the requested preview mode/output transform.

That meant a requested calibration preview could still be rendered through display defaults.

This is now fixed.

### 2. Corrected-still proof was incomplete

Before this pass, corrected stills were only checked for pixel change.

That proved that a correction was rendered, but not that the corrected result actually converged to the intended target.

This is now fixed with `corrected_residual_validation.json`.

## Unproven Areas

- The shared ROI still relies on operator placement rather than automatic gray-sphere detection.
- Background or stand contamination can still affect results if the ROI includes them.
- The physical-validation `.3333` target and the report strategy target are not the same concept and should not be conflated.

## Fixes Applied In This Pass

- preview rendering now honors requested preview settings
- corrected stills are remeasured with solver-consistent three-window geometry
- contact sheets now surface:
  - exposure correction
  - corrected residual
  - tolerance status
  - trust class / reference use

## Recommended Follow-Up

- If the project needs automatic object-presence verification for gray spheres, add explicit sphere detection or a supplied circular ROI path to the review workflow.
- Keep residual validation mandatory for any future “safe to push” claim.
