# UI Lightweight Runtime Handoff

## Goal

Trace the real web-invoked lightweight review path, prove what actually runs between:

- `clip_measurement_start`
- `clip_measurement_complete`

and fix the remaining hot spot without changing the accepted 3-band / IPP2 science.

## What Was Wrong With The Previous Proof

The earlier lightweight-performance proof was based on direct CLI verification, not on a browser-triggered run with durable runtime evidence from the web path itself.

The web route does shell into the same CLI, but that was not previously proven in a trace artifact. That gap made it possible for an older slow run to appear inconsistent with the earlier “fixed” summary.

This pass closes that gap by:

- tagging runs with `R3DMATCH_INVOCATION_SOURCE`
- persisting `lightweight_runtime_trace.json`
- validating the actual UI-style command shape

## Real UI Path

The browser route launches:

- `web_app.build_review_web_command()`
- `/bin/tcsh -c ... python3 -m r3dmatch.cli review-calibration ...`
- `workflow.review_calibration()`
- `matching.analyze_path()`
- `matching.analyze_clip()`

The active clip-measurement path is therefore:

1. `resolve_backend()`
2. `backend.inspect_clip()`
3. `build_sample_plan()`
4. `backend.decode_frames(...)`
5. `analyze_frame(...)`
6. `measure_frame_color_and_exposure(...)`

That path is now explicitly recorded for lightweight runs via:

- `lightweight_runtime_trace.json`
- `measurement_workload_trace.json`

## Domain Proof

### During clip measurement

The clip-measurement phase still uses the SDK decode plus the local proxy monitoring transform.

This is now explicitly labeled as:

- `measurement_domain = scene_sdk_decode_with_proxy_monitoring`
- `measurement_source = scene_sdk_decode_with_proxy_monitoring`

That is acceptable for the fast prepass because it is no longer masquerading as final operator truth.

### During operator-facing lightweight review

The actual review-facing measurement path remains:

- `matching_domain = perceptual`
- `measurement_domain_trace.measurement_source = rendered_preview_ipp2`
- `measurement_preview_transform = REDLine IPP2 / BT.709 / BT.1886 / Medium / Medium`

So:

- fast clip measurement = prepass
- operator-facing review decision = rendered IPP2 preview

The live workflow no longer uses RWG / Log3G10 as operational decision truth.

## Root Cause

The 105s-per-clip behavior did **not** reproduce on the current real UI path once the route was traced directly.

What did reproduce was the true current hot spot inside clip measurement:

- `measure_frame_color_and_exposure()`
- specifically `_measure_gray_sphere_statistics()`

Before the fix in this pass, that function was still doing redundant sphere work:

- legacy raw sphere stats
- refined raw sphere stats
- legacy monitoring sphere stats
- refined monitoring sphere stats
- raw 3-band profile
- monitoring 3-band profile

That was more work than the intended lightweight contract required.

## Fixes Made

### `src/r3dmatch/web_app.py`

- web-launched review commands now set:
  - `R3DMATCH_INVOCATION_SOURCE=web_ui`

### `src/r3dmatch/workflow.py`

- lightweight runs now persist:
  - `lightweight_runtime_trace.json`
- progress start payload now includes `invocation_source`

### `src/r3dmatch/matching.py`

- `analyze_path()` now writes:
  - `lightweight_runtime_trace.json`
- `analyze_clip()` now records:
  - per-function durations
  - per-frame decode/analyze/measure timings
  - actual frame count / resolution / domain / reuse facts
- `measure_frame_color_and_exposure()` now records nested timing for:
  - raw region extraction
  - monitoring transform/extraction
  - gray-sphere stats

### `src/r3dmatch/calibration.py`

- `measure_sphere_zone_profile_statistics()` now accepts an optional gradient-axis override

### Lightweight hot-spot reduction

`_measure_gray_sphere_statistics()` was simplified so it no longer performs unnecessary duplicate refined region passes.

It now:

- keeps the 3-band center-sphere profile
- keeps legacy-vs-refined comparison for diagnostics
- reuses the raw profile’s gradient axis for the monitoring profile
- reduces redundant work inside the lightweight measurement loop

## Workload Proof

### Single-clip UI-style proof run

Run:

- `runs/ui_runtime_trace_063_single/ui_runtime_trace_clip1`

Artifacts:

- `lightweight_runtime_trace.json`
- `measurement_workload_trace.json`

Measured facts for `G007_A063_032563_001`:

- `invocation_source = web_ui`
- `frames_analyzed = 1`
- `decode_width = 1920`
- `decode_height = 1080`
- `decode_half_res = true`
- `measurement_domain = scene_sdk_decode_with_proxy_monitoring`
- `measurement_source = scene_sdk_decode_with_proxy_monitoring`
- `detection_count = 0`
- `gradient_axis_count = 1`
- `region_stat_count = 2`
- `strategy_reuse = true`

Timing:

- `clip_measurement_start` at `+0.02s`
- `clip_measurement_complete` at `+2.12s`

Per-function timing:

- `resolve_backend_seconds ≈ 0.0027`
- `inspect_clip_seconds ≈ 0.0209`
- `decode_and_measure_seconds ≈ 2.0772`

Per-frame timing:

- `decode_wait_seconds ≈ 0.4503`
- `analyze_frame_seconds ≈ 0.0171`
- `measure_frame_seconds ≈ 1.6097`

Nested measurement timing:

- `raw_region_seconds ≈ 0.0075`
- `monitoring_region_seconds ≈ 0.0637`
- `gray_sphere_statistics_seconds ≈ 1.5378`

So the true remaining bottleneck is:

> gray-sphere measurement work inside the lightweight prepass

not REDLine rendering and not a hidden multi-frame decode loop.

### 12-clip UI-style subset proof run

Run:

- `runs/ui_runtime_trace_063_subset/ui_runtime_trace_063_subset`

Observed measurement timings from stdout:

- clip 1 complete at `+2.17s`
- clip 6 complete at `+12.58s`
- clip 12 complete at `+25.19s`
- analysis complete at `+25.23s`

That is the key result:

- the real UI-style lightweight measurement phase for the 12-clip `063` subset is now about `25s`
- not `~105s per clip`

## Before / After

### User-reported bad run

- clip 1 complete around `+105.08s`
- clip 2 starts immediately after
- apparent behavior: `~105s per clip`

### Before this pass, current-code hot spot

From the first UI runtime trace before the inner gray-sphere simplification:

- first clip total measurement ≈ `3.61s`
- dominant time was already `measure_frame_color_and_exposure()`

### After this pass

Single-clip UI-style run:

- clip 1 complete at `+2.12s`

12-clip `063` UI-style run:

- clip 12 complete at `+25.19s`
- analysis complete at `+25.23s`

## Safety Check

The accepted science was not changed:

- 3-band gradient-axis sphere model remains intact
- weights remain `0.3 / 0.5 / 0.2`
- direct REDLine monitoring-domain review remains intact
- operator-facing perceptual review still relies on `rendered_preview_ipp2`
- no operational decision-making was moved back to RWG / Log3G10

## Validation

- `py_compile` passed
- `tests/test_cli.py` passed: `200 passed`

## Remaining Non-Blocking Risk

Lightweight review is now fast in its measurement phase, but it still has a later cost for:

- representative REDLine monitoring preview rendering
- report generation

That later cost is acceptable because it is:

- outside clip measurement
- progress-reported
- scientifically tied to operator-facing IPP2 review

The clip-measurement bottleneck itself is no longer behaving like a hidden 105-second stall.
