## White Balance Truth + Chart Usefulness

### Trace

1. Raw neutral / RGB measurement source
   - `src/r3dmatch/report.py`
   - `_measure_rendered_preview_roi_ipp2(...)`
   - writes monitoring-domain sample data into `zone_measurements`, `neutral_samples`, `sample_1_ire`, `sample_2_ire`, `sample_3_ire`

2. WB solve output
   - `src/r3dmatch/report.py`
   - `_build_strategy_payloads(...)`
   - calls `solve_white_balance_model_for_records(...)`
   - per-clip final values are written through `build_commit_values(...)`

3. Commit payload output
   - `src/r3dmatch/report.py`
   - strategy clip payloads receive `commit_values.kelvin`, `commit_values.tint`, `commit_values.exposureAdjust`
   - commit export remains the authoritative flattened per-camera calibration payload

4. UI summary input
   - `src/r3dmatch/web_app.py`
   - `_build_operator_surfaces(...)`
   - operator WB card reads `review_payload["white_balance_model"]`

### Root Cause

The full-contact-sheet payload carried valid per-camera Kelvin/Tint commit values but did not always promote the chosen `white_balance_model` summary to the top-level report payload. The UI therefore rendered `n/a` even though a usable WB result already existed.

### Fix

- `src/r3dmatch/report.py`
  - canonical WB summary is built once by `_canonical_white_balance_model_summary(...)`
  - full-contact-sheet payload now publishes `payload["white_balance_model"]`
  - strategy payloads and strategy comparison payloads carry the same summary object

- `src/r3dmatch/web_app.py`
  - `_resolve_white_balance_model_surface(...)` normalizes the WB summary used by the operator surface
  - if direct solve metadata is absent, the UI derives a truthful fallback from final per-camera Kelvin/Tint payloads
  - fallback rules:
    - shared Kelvin if retained cameras agree within tolerance
    - shared Tint if retained cameras agree within tolerance
    - otherwise `per-camera`
    - `candidate_count >= 1` when a usable final WB solution exists

### Diagnostics Added

The WB surface now exposes:

- retained camera count
- kelvin spread
- tint spread
- source sample count
- mean neutral residual after solve
- summary kind: `direct_solve` or `fallback_derived`

### Chart Usefulness

`src/r3dmatch/report.py` now suppresses low-value charts:

- `strategy_chart_svg` renders only when strategy differences are materially informative
- `trust_chart_svg` renders only when trust classes or trust scores materially vary

This keeps subtle, meaningful plots while removing polished-but-flat clutter.

### Scope Guard

- no calibration science changed
- no report-side remeasurement
- no new sphere detection
- no new sample computation
- no duplicate baseline REDLine render
