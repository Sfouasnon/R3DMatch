# Analysis Layer Scientific Traceability Handoff 2

## What changed

- Added measurement-time provenance persistence in `/Users/sfouasnon/Desktop/R3DMatch/src/r3dmatch/matching.py`
- Added bounded pixel-trace persistence at the zone-stat level in `/Users/sfouasnon/Desktop/R3DMatch/src/r3dmatch/calibration.py`
- Hardened scientific-validation artifact generation in `/Users/sfouasnon/Desktop/R3DMatch/src/r3dmatch/report.py`
- Wired scientific validation into full contact-sheet report generation with hard failure on report drift
- Added regression tests for:
  - provenance hash persistence
  - pixel-trace export
  - `fully_reconciled`
  - `blocked_asset_mismatch`
  - `analysis_drift`

## True measurement code path

### Pixel entry

- `r3dmatch.report._measure_rendered_preview_roi_ipp2(...)`
- `PIL.Image.open(...).convert("RGB")`
- `numpy.asarray(..., dtype=float32) / 255.0`

This is display-referred IPP2 preview data in normalized `0..1` RGB.

### Sphere detection

- `r3dmatch.report._detect_sphere_candidates_in_region_hwc(...)`
- `skimage.feature.canny`
- `skimage.transform.hough_circle`
- `skimage.transform.hough_circle_peaks`

This is not OpenCV in the current implementation.

### Sphere mask and zone sampling

- `r3dmatch.calibration.build_sphere_sampling_mask(...)`
- `r3dmatch.calibration._sphere_gradient_axis(...)`
- `r3dmatch.calibration._sphere_zone_geometries(...)`
- `r3dmatch.calibration._measure_pixel_cloud_statistics(...)`

Zone geometry is three gradient-aligned rectangular sample bands intersected with the refined sphere mask.

### Math

- luminance: `Y = 0.2126 R + 0.7152 G + 0.0722 B`
- trim: 5th to 95th percentile after clipping guard
- scalar: `median(log2(trimmed_luminance))`
- IRE: `(2 ** measured_log2_luminance) * 100`

These are computed display-domain IRE values, not waveform instrument readings.

## Measurement provenance now persisted

Fresh rendered-preview analysis artifacts now persist:

- measurement preview file path
- existence at measurement time
- file size
- SHA-256
- image dimensions
- source clip path
- frame index
- timestamp seconds
- monitoring transform identity
- REDLine command / executable / preview settings
- detected sphere ROI
- dominant gradient axis
- crop bounds / crop size
- per-zone geometry summary

These fields live under:

- `diagnostics.measurement_provenance`

## Pixel-trace data now persisted

Each zone now carries bounded trace data under:

- `diagnostics.zone_measurements[*].pixel_trace`

This includes:

- raw RGB preview values
- valid RGB preview values
- trimmed RGB preview values
- raw luminance preview values
- valid luminance preview values
- trimmed luminance preview values
- trimmed log2 preview values
- median trimmed luminance
- median log2 luminance
- computed IRE

This gives a scientist a stored proof chain:

- `RGB -> luminance -> trim -> log2 -> IRE`

without needing to guess from a later regenerated asset.

## Replay integrity state model

Scientific validation now distinguishes:

- `fully_reconciled`
  - replay asset fingerprint matches
  - replayed values match stored analysis
  - report and IPP2 validation match stored analysis
- `blocked_asset_mismatch`
  - report/analyze truth still reconcile
  - but replay asset fingerprint is missing or mismatched
  - or replayed values from the current asset do not match stored analysis
- `analysis_drift`
  - report truth or validation truth diverges from stored analysis truth

The report builder still fails hard on `analysis_drift`.

## Real run artifact produced

Generated for:

- `/Users/sfouasnon/Desktop/R3DMatch/runs/final_arch_lock/real_063_arch_lock`

Artifacts:

- `/Users/sfouasnon/Desktop/R3DMatch/runs/final_arch_lock/real_063_arch_lock/report/scientific_validation.json`
- `/Users/sfouasnon/Desktop/R3DMatch/runs/final_arch_lock/real_063_arch_lock/report/scientific_validation.md`

## Key result

The archived run now classifies more precisely:

- stored analysis values reconcile with `ipp2_validation.json`
- stored analysis values reconcile with report-side displayed values
- the archived analysis artifact does **not** contain a measurement-time fingerprint
- replaying the current measurement preview asset on disk does **not** reproduce those stored analysis values

Current real-run status:

- `blocked_asset_mismatch`
- provenance status: `missing_measurement_fingerprint`

That means the current archived measurement preview asset is not a reliable replay source for that run’s stored measurements, and the system now says so explicitly instead of collapsing it into a generic mismatch.

## Interpretation

This pass validates the pipeline math and the code path, and it locks fresh-run provenance so future runs can be replayed against the exact measurement-time asset identity.

For the archived `real_063_arch_lock` run:

- explicit math traceability: yes
- report/analyze consistency: yes
- replay integrity from current archived asset: no
- full scientific reconciliation: no

Why not:

- the original measurement-time asset fingerprint was not persisted in that older run
- the current on-disk replay asset does not numerically reproduce the stored analysis measurements

For fresh runs going forward:

- the scientific-validation artifact exposes the measurement-time asset fingerprint
- the analysis JSON carries per-zone pixel-trace previews
- the report build fails clearly on report drift
- archived-asset mismatches are surfaced explicitly instead of silently masked
