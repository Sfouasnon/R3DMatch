# Lightweight Measurement Domain Handoff

## Scope

This pass fixed two coupled problems in lightweight review mode:

1. the active operator-facing measurement path was still rooted in scene-domain SDK analysis and only labeled as monitoring-like
2. lightweight analysis was doing too much work per clip before the first useful review artifact

The accepted science remains intact:

- gradient-axis 3-band sphere model
- `bright_side` / `center` / `dark_side`
- weights: `0.3 / 0.5 / 0.2`
- direct REDLine monitoring-domain preview path
- authoritative IPP2 validation path

## Root Cause

### Measurement-domain issue

Before this pass, lightweight review commonly started from scene-domain SDK measurements and a local proxy transform:

- analysis decode source: RED SDK frame decode
- analysis domain: scene-domain clip metadata / Log-era semantics
- diagnostics label: monitoring-style fields

That mismatch let the system present monitoring-flavored summaries while still carrying scene-domain assumptions such as:

- `expected_log3g10_gray`
- `pre_log3g10_luminance`
- `post_log3g10_luminance`

Those values are still acceptable as secondary engineering diagnostics, but they are not acceptable as the active operational measurement truth for this workflow.

### Lightweight runtime issue

Before this pass, lightweight analysis still behaved like a mini analysis batch instead of a true representative-frame pass:

- multiple frames per clip
- full-resolution decode
- repeated per-frame measurement work before first durable review artifact

That is why a 12-clip lightweight run could spend roughly 1368 seconds total with per-clip measurement times around 107-117 seconds.

## Active Domain After Fix

The active operator-facing lightweight measurement path now uses:

- one representative frame per clip
- half-resolution decode for the analysis prepass
- real REDLine monitoring previews for the review-facing measurement step
- monitoring-domain pixel measurement from rendered IPP2 previews

Proof artifact:

- `runs/lightweight_domain_fix_063/fresh_lightweight_063/report/contact_sheet.json`

Key fields:

- `matching_domain = "perceptual"`
- `measurement_domain_trace.measurement_source = "rendered_preview_ipp2"`

Important distinction:

- the analysis prepass still uses the SDK decode and is now honestly labeled as `scene_sdk_decode_with_proxy_monitoring`
- the operational lightweight review decision path now uses `rendered_preview_ipp2`

That separation is intentional and now explicit.

## Lightweight Execution Model After Fix

Lightweight mode now enforces:

1. one representative frame per clip
2. half-resolution decode
3. one sphere detection pass per clip in the analysis prepass
4. one gradient-axis computation per clip
5. one region-stat pass per clip
6. strategy reuse from the stored measurement result

Proof artifact:

- `runs/lightweight_domain_fix_063/fresh_lightweight_063/measurement_workload_trace.json`

Representative per-clip values from that trace:

- `frames_analyzed = 1`
- `decode_width = 1920`
- `decode_height = 1080`
- `decode_half_res = true`
- `strategy_reuse = true`

## Changes Made

### `src/r3dmatch/sdk.py`

- added `half_res` decode support to backend decode APIs
- mock backend now downsamples when `half_res=True`
- RED backend threads `half_res` through to the native bridge

### `src/r3dmatch/matching.py`

- lightweight analysis now supports:
  - `half_res_decode`
  - single representative-frame sampling
  - per-run workload trace export
- added workload counters and timings:
  - detection count
  - gradient-axis count
  - region-stat count
  - decode size
  - frame count
  - total measurement time
- analysis measurement domain labels are now honest:
  - `scene_sdk_decode_with_proxy_monitoring`

### `src/r3dmatch/workflow.py`

- lightweight review now resolves to perceptual matching for operator-facing review
- lightweight calls into analysis with:
  - `sample_count=1`
  - `half_res_decode=True`
- review validation now correctly treats perceptual lightweight review as outside scene-domain physical validation

### `src/r3dmatch/report.py`

- lightweight report now always renders representative measurement previews in the monitoring domain
- active measurement source for review decisions is now:
  - `rendered_preview_ipp2`
- trust/quality scoring in perceptual mode now prefers monitoring-derived measurement quality instead of scene-domain array-quality flags
- expected sphere band spread is no longer penalized as if it were flat-patch instability

### `src/r3dmatch/cli.py`

- `review-calibration` default matching domain now favors perceptual review for this workflow

### `tests/test_cli.py`

- added/updated regression coverage for:
  - workload trace output
  - perceptual lightweight review payloads
  - rendered-preview monitoring measurement source
  - physical validation skip behavior in perceptual lightweight review

## Before / After Timing

### Before

User-observed / prior evidence:

- about `1368s` total for a 12-clip lightweight subset
- about `107-117s` per clip during measurement

### After

Fresh run:

- `runs/lightweight_domain_fix_063/fresh_lightweight_063`

Observed timing:

- analysis phase: about `7.7s` total for 12 clips
- per-clip analysis prepass: about `0.6-0.7s`
- representative IPP2 preview render stage: about `12.9s`
- full lightweight end-to-end review: about `78.6s`

This is the correct tradeoff:

- the expensive part is now the intentional real REDLine monitoring render
- the silent CPU-bound analysis explosion has been removed

## Result Quality After Fix

The refreshed lightweight run now aligns with the visible monitoring-domain evidence much better.

Artifacts:

- `runs/lightweight_domain_fix_063/fresh_lightweight_063/report/contact_sheet.json`
- `runs/lightweight_domain_fix_063/fresh_lightweight_063/report/review_package.json`
- `runs/lightweight_domain_fix_063/fresh_lightweight_063/report/review_validation.json`

Key outcomes:

- `review_validation.status = "success"`
- `physical_validation.status = "unsupported"`
- warning explicitly states that this is not a RED scene-domain array calibration
- `run_assessment.status = "READY"`
- `recommendation_strength = "HIGH_CONFIDENCE"`
- trusted camera count resolves coherently across the 12-clip set

## Remaining Non-Blocking Risks

- scene-domain diagnostics such as `expected_log3g10_gray` remain in engineering outputs; they should stay secondary and never be confused with active IPP2 review truth
- lightweight still performs real REDLine monitoring renders, so it is not “instant”; the difference now is that the work is scientifically required, bounded, and progress-reported

## Locked Assumptions

For this workflow, operational lightweight review should be understood as:

> a representative-frame, half-resolution analysis prepass feeding a real monitoring-domain REDLine measurement pass, with operator-facing decisions based on rendered IPP2 preview pixels rather than Log-era scene-domain proxies.
