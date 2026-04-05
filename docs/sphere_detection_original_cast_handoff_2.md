# Sphere Detection + Original Cast Handoff 2

## What changed

This pass fixed two real issues in the review/report path:

1. Off-edge sphere solves could still miss the actual gray sphere when no explicit calibration ROI was present.
2. The report-level `Original WB Evaluation` block could fall back to metadata wording even when measured original-still chromaticity was available deeper in the run payloads.

## Why the sphere overlay was still wrong

Two problems were interacting:

- When `calibration_roi` was missing, the report measurement path had already been corrected to use the full frame, but the detector still favored a dark neutral circular structure near the chart rig over the true right-edge sphere.
- The Hough candidate scorer had a brightness prior centered too dark for these monitoring-domain TIFF stills, so the actual sphere could be present in the candidate pool and still lose.
- On one failing camera, the detector returned early as soon as the top Hough candidate reached a `MEDIUM/HIGH` confidence bucket, which prevented alternate recovery candidates from competing at all.

## Sphere detection changes

In `src/r3dmatch/report.py`:

- Kept full-frame measurement when no explicit ROI is available.
- Broadened the candidate scorer so it no longer penalizes the real sphere for being brighter than the old center-biased prior expected.
- Reweighted scoring toward a true gray-target signature:
  - edge support
  - contrast to surround
  - neutral balance
  - plausible radius
  - plausible brightness
  - smooth interior
- Added a supplemental `neutral_blob_recovery` path that finds bright, low-saturation, sphere-like components across the frame and evaluates them with the same scoring model.
- Removed the premature “accept top Hough candidate immediately” behavior unless that candidate is already extremely strong.

### Result on the previously failing cameras

- `H007_C067_0403RE_001`
  - chosen source: `primary_detected`
  - chosen ROI: `cx=3510.0, cy=864.0, r=216.0`
- `H007_D067_0403YU_001`
  - chosen source: `neutral_blob_recovery`
  - chosen ROI: `cx=3474.83, cy=902.75, r=166.24`

The rebuilt packaged overlays now land on the actual right-edge gray sphere for both cameras.

## Why Original Cast was missing

The measured values were present in analysis artifacts, but they were not always surviving into the later contact-sheet/report payload.

Specifically:

- `analysis/*.analysis.json` already contained measured original-still chromaticity under diagnostics.
- Later report payload sanitization dropped key WB/measurement fields from the clip payload used by the contact sheet.
- `Original WB Evaluation` then saw an incomplete clip record and fell back to metadata-context wording.

## Original Cast persistence changes

In `src/r3dmatch/report.py`:

- Promoted measured original-still fields into the contact-sheet strategy payload.
- Added defensive fallback reads so the contact-sheet block can resolve measured values from either top-level clip fields or nested metrics payloads.

## Verified Original Cast values

From the rebuilt packaged run:

- `H007_C067_0403RE_001`
  - measured chromaticity: `[0.354374, 0.326538, 0.318054]`
  - report wording: `Evaluation basis: Measured from the original neutral target`
  - report wording: `Original cast: Near neutral in the original still`
- `H007_D067_0403YU_001`
  - measured chromaticity: `[0.354772, 0.324109, 0.321119]`
  - report wording: `Evaluation basis: Measured from the original neutral target`
  - report wording: `Original cast: Near neutral in the original still`

## Tests run

- `PYTHONPYCACHEPREFIX=/tmp ./.venv/bin/python -m py_compile src/r3dmatch/report.py tests/test_cli.py`
- `PYTHONPATH=src ./.venv/bin/pytest -q tests/test_cli.py -k 'off_center_sphere or far_right_edge_sphere or bright_edge_sphere_vs_false_circle or original_wb_block'`
- `PYTHONPATH=src ./.venv/bin/pytest -q tests/test_cli.py`

Results:

- focused regressions: `4 passed`
- full suite: `266 passed`

## Packaged validation

Rebuilt artifacts:

- `dist/R3DMatch.app`
- `dist/R3DMatch-macos-arm64.zip`

Validated with the packaged executable:

- `dist/R3DMatch.app/Contents/MacOS/R3DMatch --cli review-calibration ... --clip-group 067`

Run output:

- run dir: `runs/packaged_detector_validation/qt_packaged_067_detector_6`
- HTML: `runs/packaged_detector_validation/qt_packaged_067_detector_6/report/contact_sheet.html`
- PDF: `runs/packaged_detector_validation/qt_packaged_067_detector_6/report/preview_contact_sheet.pdf`
- review validation: `success`

Overlay checks:

- `runs/packaged_detector_validation/qt_packaged_067_detector_6/report/review_detection_overlays/H007_C067_0403RE_001.original_detection.png`
- `runs/packaged_detector_validation/qt_packaged_067_detector_6/report/review_detection_overlays/H007_D067_0403YU_001.original_detection.png`

Both now place the solve circle on the gray sphere.

## Remaining limitations

- `Measurement source: shared original.display scalar log2` is still more internal than ideal operator wording. This pass focused on the missing cast values and the failing sphere overlays rather than a broader wording cleanup.
- `pre_color_residual` is not populated in the analysis diagnostics for these packaged perceptual runs; the important measured chromaticity values are present and now survive into the report.
- TIFF measurement and scientific validation behavior were preserved. No attempt was made to mask `blocked_asset_mismatch` states where replay provenance is genuinely unresolved.
