## Exposure Analysis Architecture Lock

This pass locks the operational exposure-analysis workflow to the authoritative IPP2 path.

### Ground truth after this pass

- Measurement source for operational review is `rendered_preview_ipp2`.
- Measurement transform is `REDLine IPP2 / BT.709 / BT.1886 / Medium / Medium`.
- The same rendered still is reused for:
  - measurement
  - contact-sheet display
  - solve overlays
- Operator-facing sample labels are:
  - `Sample 1`
  - `Sample 2`
  - `Sample 3`
- Numeric sample values are persisted and reused:
  - `sample_1_ire`
  - `sample_2_ire`
  - `sample_3_ire`

### Scalar solve correction

The exposure scalar used by strategy selection, anchor comparison, and report summaries now comes from the persisted sphere samples themselves.

- Source: stored `zone_measurements`
- Aggregation: median of sample log2 display-luminance values from the IPP2 render
- This replaces any dependence on stale generic luminance fields when a real sphere profile is present.

This keeps exposure solve aligned with the actual sphere measurement rather than frame-level placeholders.

### Display-domain scalar naming

This pass also corrects the scalar naming so the code matches what the system is really measuring.

Old generic naming in the perceptual path included:

- `measured_log2_luminance`
- `measured_log2_luminance_monitoring`
- `log2_luminance`

Those names suggested a generic or scene-style luminance scalar even when the value had already come from an IPP2 render.

The perceptual operational path now uses explicit display-domain naming:

- `sample_scalar_display_log2`
- `sample_scalar_display_ire`
- `display_scalar_log2`
- `display_scalar_ire`
- `derived_display_scalar_log2`
- `derived_display_scalar_ire`

Legacy generic names remain only as compatibility aliases and fallback fields. They are no longer the conceptual source of truth when a sphere sample profile exists.

### What was fixed

The remaining architecture mismatch was that some report/strategy code still read scalar exposure from stored generic `measured_log2_luminance_*` fields even when a full sphere profile was available.

That is now corrected by:

- deriving the scalar from stored `zone_measurements`
- reusing that same scalar in:
  - strategy payload construction
  - anchor-relative offsets
  - reusable rendered-measurement payloads

### Contact sheet reuse

For current analysis runs, report generation reuses the stored analysis originals and does not add a second baseline REDLine render pass.

The fresh full run at:

- `/Users/sfouasnon/Desktop/R3DMatch/runs/final_arch_lock/real_063_arch_lock`

proves this:

- measurement and display transforms match
- `preview_commands.json` contains `0` original commands in the report-stage preview set
- corrected review stills are rendered once for validation/display

### Overlay behavior

Contact-sheet overlays are derived from persisted solve data, not recomputed heuristically.

They reuse stored:

- sphere center
- sphere radius
- gradient axis
- sample-region geometry
- sample IRE values

The report layer still measures corrected validation renders for acceptance. It does **not** re-measure the original baseline truth for the current workflow.

### Operational answers

1. Is any operational review path still using SDK decode for measurement?

- No. `review_calibration()` dispatches analysis with `measurement_source="rendered_preview_ipp2"`.

2. Is exposure solve using sphere measurements exclusively?

- Yes for operational review. The solve scalar now comes from the median of the stored sphere sample measurements.

3. Can perceptual mode still read generic luminance when sphere samples exist?

- No. In the perceptual operational path, `zone_measurements` take priority and the scalar is derived from stored sample measurements. Generic luminance fields are retained only as compatibility fallbacks when sample data is missing.

4. Are duplicate baseline renders happening?

- Not in the current full-contact-sheet workflow for fresh runs.

5. Does `report.py` re-measure anything?

- It re-measures corrected validation renders during closed-loop acceptance.
- It does not re-measure original baseline measurement truth for current fresh runs.

6. Are overlays derived from stored geometry?

- Yes.

7. Are `sample_1_ire` / `sample_2_ire` / `sample_3_ire` stored numerically and reused?

- Yes.

8. Did any numerical outputs change?

- No. This was a semantic and architectural clarification pass, not a math change.

### Fresh validation

Fresh full run:

- `/Users/sfouasnon/Desktop/R3DMatch/runs/final_arch_lock/real_063_arch_lock/report/ipp2_validation.json`

Result:

- `PASS 12`
- `REVIEW 0`
- `FAIL 0`
- `all_within_tolerance = true`

### Remaining non-blocking nuance

Legacy non-operational scene-analysis helpers still exist in the codebase for compatibility and tests. They are no longer the operational truth path for review calibration.
