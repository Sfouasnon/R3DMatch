# Contact Sheet Diagnostic Redesign Handoff

## Page 1 redesign

The old summary/dashboard surface was replaced with an adaptive contact-sheet page model.

New first-page behavior:

- fixed 3-column camera grid
- row count derived from camera count
- up to 12 cameras per contact-sheet page before pagination
- same structure repeated for larger sets across multiple pages
- no oversized summary cards
- no presentation-style hero blocks
- tight header with logo, title, and compact metadata strip

Each contact-sheet page now contains:

1. original camera grid
2. original array synopsis
3. adjusted camera grid
4. adjusted array synopsis

This makes page 1 read like a calibration sheet instead of a dashboard.

## Adaptive grid and pagination

Grid logic:

- columns = 3
- max rows per contact-sheet page = 4
- page size = 12 cameras

Behavior:

- `<= 12` cameras: all cameras on page 1
- `13–24` cameras: two contact-sheet pages
- `25–36+` cameras: additional contact-sheet pages, still 3-up

Images are not shrunk further just to force more cameras onto a page. Readability stays ahead of single-page completeness.

## Graph scaling changes

The original and adjusted synopsis plots now derive Y ranges from actual persisted values with only small padding.

This applies to:

- exposure trace
- kelvin trace
- tint trace

Result:

- subtle differences remain visible
- graphs no longer flatten small discrepancies with overly wide ranges
- page-level synopsis is useful for anomaly detection instead of decorative summary

## Metadata policy

Original grid metadata line now uses stored metadata only:

- ISO
- shutter
- kelvin
- tint

Rules enforced:

- no T-stop
- no placeholders
- missing values omitted cleanly

Adjusted grid line uses stored correction values only:

- `exposureAdjust`
- kelvin
- tint

## Sphere totals placement

Original-grid cells now place the sphere totals directly under the original still:

- `S1`
- `S2`
- `S3`
- `Scalar`

These are drawn from stored persisted values only:

- `sample_1_ire`
- `sample_2_ire`
- `sample_3_ire`
- stored display scalar

No report-side recomputation was added.

## Per-camera alignment fixes

Per-camera pages were tightened into a diagnostic verification layout:

- top strip: camera, status, recommended action
- locked original/corrected comparison
- prominent stored solve overlay
- sample values immediately adjacent to overlay
- compressed metric strip
- compressed flags strip

This reduces mental joining between image evidence and numeric evidence.

## Persisted truth reused

The redesign still reuses stored truth only:

- original measured stills
- corrected stills
- exact stored solve overlays
- `sample_1_ire`
- `sample_2_ire`
- `sample_3_ire`
- stored display scalar values
- stored correction values
- stored validation residuals
- stored notes / flags

Confirmed:

- no report-side remeasurement
- no new sphere detection
- no new sample computation
- no duplicate baseline REDLine render

## Validation

Validated in this pass:

- `py_compile src/r3dmatch/report.py tests/test_cli.py`
- `PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /Users/sfouasnon/Desktop/R3DSplat/.venv39validate/bin/pytest -q tests/test_cli.py`

Result:

- `203 passed`

## Known limitation

A fresh artifact generation attempt from this session depends on shell sandbox write permissions for the chosen output location inside the project. If that path is blocked, code and test validation remain the source of truth for this pass.
