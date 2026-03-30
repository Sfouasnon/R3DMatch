## Lightweight IPP2 Measurement Optimization Handoff

### Scope

This pass was a surgical optimization of the already-correct IPP2-first lightweight review path.

It did **not** change:

- the IPP2-first architecture
- the 3-band gradient-axis sphere model
- bright / center / dark weighting (`0.3 / 0.5 / 0.2`)
- REDLine direct-parameter semantics
- validation thresholds
- solve math

The goal was only to reduce the remaining cost inside rendered-preview measurement while preserving measurement truth.

### Root Cause

The dominant hot spot was inside `report._measure_rendered_preview_roi_ipp2()`, specifically the profile measurement phase after sphere detection.

Measured before optimization:

- average REDLine render time per clip: about `1.06s`
- average rendered-preview measurement time per clip: about `11.52s`
- average profile measurement time inside that path: about `11.39s`

This proved the remaining bottleneck was **not**:

- REDLine render time
- sphere detection
- gradient-axis solve
- report-side duplication

It was the expensive profile measurement work itself, especially:

- building refined masks over a larger ROI than necessary
- running mask/statistics work on far more pixels than required
- using morphological erosion where a direct interior-circle mask is equivalent for this standardized sphere workflow

### Optimization

Two targeted changes were made.

#### 1. Measure on a tight sphere-local crop

After sphere detection succeeds, the measurement path now crops to a bounded local region around the detected sphere before running the 3-band profile measurement.

This preserves correctness because:

- detection is still performed on the real rendered IPP2 image
- final band statistics are still measured on real rendered IPP2 pixels
- returned zone geometry is remapped back into parent ROI coordinates
- no scene-domain or proxy-monitoring measurement is reintroduced

The render remains `3840x2160`, but the expensive mask/statistics work no longer scans an unnecessarily large region.

#### 2. Replace erosion-based interior mask construction with a direct interior-circle mask

Inside `calibration.build_sphere_sampling_mask()`, the interior region is now built from the intended interior-radius constraint directly instead of computing a binary erosion disk over the full mask.

This preserves the same scientific intent:

- conservative interior sampling
- edge exclusion
- stable many-pixel band measurement

but removes a large amount of avoidable per-clip computation.

### Files Changed

- `src/r3dmatch/report.py`
- `src/r3dmatch/calibration.py`
- `src/r3dmatch/matching.py`

Traceability changes:

- `matching.py` now persists `measurement_crop_bounds` and `measurement_crop_size` into the clip diagnostics so the local-crop optimization is visible in analysis artifacts.

### Timing Evidence

Fresh real UI-style optimized run:

- run root:
  - `runs/ui_ipp2_first_063_opt/ui_ipp2_first_063_opt`
- runtime trace:
  - `runs/ui_ipp2_first_063_opt/ui_ipp2_first_063_opt/lightweight_runtime_trace.json`

Before optimization, from the prior IPP2-first run:

- clip 1 total measurement time: about `13.07s`
- 12-clip measurement phase: about `151.59s`
- full lightweight review: about `152.64s`
- average rendered-preview measurement time per clip: about `11.52s`

After optimization:

- clip 1 complete: about `1.78s`
- clip 12 complete: about `17.34s`
- analysis complete: about `17.38s`
- full lightweight review: about `18.22s`

Average per-clip timing after optimization:

- total per-clip measurement: about `1.41s`
- REDLine render: about `1.01s`
- rendered-preview measurement: about `0.39s`

Representative internal measurement breakdown after optimization:

- image load: about `0.13s`
- sphere detection: effectively negligible
- crop construction: effectively negligible
- mask build: about `0.18s`
- gradient-axis solve: about `0.017s`
- zone geometry: effectively negligible
- zone statistics: about `0.038s`
- full profile measurement: about `0.24s`

### Resolution Decision

The system still renders at `3840x2160`.

That was intentionally preserved for trustworthiness.

The optimization did **not** reduce render resolution. Instead, it reduced unnecessary measurement work after detection by measuring on a sphere-local crop while still using the full-resolution IPP2 render as the authoritative source.

This is the safer choice because:

- detection still sees the real rendered frame
- final band statistics still come from the real rendered frame
- only redundant non-local computation was removed

### Safety Check

The following remain true after this optimization:

- measurement still starts from `rendered_preview_ipp2`
- measurement domain is still real REDLine IPP2 / BT.709 / BT.1886 / Medium / Medium
- no scene-domain or proxy-monitoring operational truth was reintroduced
- `report.py` still reuses stored analysis measurements instead of re-measuring
- the 3-band model and weights are unchanged

Validation after the optimization pass:

- `py_compile` passed
- `tests/test_cli.py` passed

### Remaining Essential Cost

The main remaining per-clip cost is now REDLine rendering itself, at roughly `~1.0s/clip` in the measured run.

That cost is expected and scientifically justified for this workflow because lightweight review is now correctly grounded in real IPP2 pixels from the start of measurement.

### Conclusion

The remaining bottleneck inside IPP2-first measurement was real and has now been reduced substantially without weakening correctness.

The system remains:

- IPP2-first
- single-source-of-truth
- 3-band and gradient-axis aligned
- report-reuse based

but the internal rendered-preview measurement cost is now low enough that lightweight review is operationally responsive again.
