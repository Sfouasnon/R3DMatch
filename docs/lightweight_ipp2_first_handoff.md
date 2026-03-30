# Lightweight IPP2-First Handoff

## What was wrong

The real UI-triggered lightweight run still had an architectural split:

1. `workflow.review_calibration()`
2. `matching.analyze_path()`
3. `matching.analyze_clip()`
4. scene-domain SDK decode + proxy monitoring transform
5. `.analysis.json` written from that proxy measurement
6. `report.build_lightweight_analysis_report()`
7. representative REDLine IPP2 previews rendered later
8. strategy payloads partially overridden from rendered IPP2 measurements

That meant the early measurement path and the later operator-facing path were not using the same pixels.

## Root cause

There were two coupled issues:

1. The active lightweight clip measurement still used `scene_sdk_decode_with_proxy_monitoring` in `matching.py`.
2. The later lightweight report rendered real IPP2 measurement previews and reused those, so the run had a hidden two-stage measurement model.

The user-visible runtime symptom matched that split:

- old real UI run: about 52 to 54 seconds per clip during the `clip_measurement_*` window
- preview rendering started later
- the process could still die during later preview/report work

## What changed

### 1. Lightweight analysis now measures real IPP2 pixels from the start

`matching.analyze_clip()` now supports `measurement_source="rendered_preview_ipp2"`.

For lightweight review, `workflow.review_calibration()` now passes:

- `measurement_source="rendered_preview_ipp2"`

That path:

1. inspects the clip
2. selects one representative frame
3. renders that frame through REDLine using the monitoring transform
4. measures the rendered image with the existing sphere detection + gradient-axis 3-band model
5. writes `.analysis.json` from those rendered IPP2 measurements

### 2. Lightweight report now reuses analysis-time IPP2 renders

`report.build_lightweight_analysis_report()` now checks whether the analysis records already carry:

- `exposure_measurement_domain = rendered_preview_ipp2`
- `rendered_measurement_preview_path`

If they do, the report:

- reuses those stored rendered measurements
- does not render a second set of measurement previews
- records `measurement_preview_reused_from_analysis = true`

### 3. UI safety

For lightweight review:

- the Web UI now forces `matching_domain=perceptual`
- the Scene-Referred option is disabled in the form when Lightweight Analysis is selected
- the CLI command emitted by the UI also forces `--matching-domain perceptual`

## Files changed

- `src/r3dmatch/calibration.py`
- `src/r3dmatch/matching.py`
- `src/r3dmatch/report.py`
- `src/r3dmatch/workflow.py`
- `src/r3dmatch/web_app.py`
- `tests/test_cli.py`

## Domain proof

Fresh real UI-style run:

- `runs/ui_ipp2_first_063/ui_ipp2_first_063/lightweight_runtime_trace.json`
- `runs/ui_ipp2_first_063/ui_ipp2_first_063/report/contact_sheet.json`

Those artifacts now show:

- `measurement_source = rendered_preview_ipp2`
- `measurement_domain = rendered_preview_ipp2`
- `measurement_preview_transform = REDLine IPP2 / BT.709 / BT.1886 / Medium / Medium`
- `measurement_preview_reused_from_analysis = true`

This means the operational lightweight measurement path is now grounded in real IPP2 monitoring renders from the first measurement step.

## Real UI-style runtime breakdown

Run:

- subset: `063`
- clip count: `12`
- source: real R3D media
- invocation source: `web_ui`

Observed:

- clip 1 measurement start: `+0.56s`
- clip 1 measurement complete: `+13.07s`
- clip 12 measurement complete: `+151.59s`
- analysis complete: `+151.64s`
- review complete: `+151.68s`
- wall clock: about `152.64s`

Average per clip from `lightweight_runtime_trace.json`:

- total measurement: about `12.58s`
- REDLine render: about `1.06s`
- rendered-preview measurement: about `11.52s`
- sphere detection: about `0.000015s`
- gradient-axis fit: about `0.017s`
- zone-stat loop: about `0.044s`
- profile measurement total: about `11.39s`

## Actual remaining bottleneck

The real remaining lightweight cost is now explicit:

- not SDK decode
- not proxy monitoring
- not a hidden second preview stage
- not REDLine render itself

The dominant cost is the measurement of the 3-band sphere profile on the rendered 3840x2160 IPP2 frame, specifically inside the profile-measurement portion of `_measure_rendered_preview_roi_ipp2()` / `measure_sphere_zone_profile_statistics()`.

So the pipeline is now correct first, and the remaining cost is honestly attributable to the actual IPP2 measurement path.

## Before / after

Before:

- lightweight clip measurement was not IPP2-first
- early decisions came from scene SDK decode + proxy monitoring
- later report logic rendered real IPP2 measurements and partially replaced those earlier values
- real user run showed about `52–54s` per clip in the clip-measurement window
- total 12-clip analysis phase was about `647s` before report stage

After:

- clip measurement itself uses real IPP2 rendered pixels
- no scene-domain measurement drives lightweight operational decisions
- report reuses the analysis-time IPP2 render instead of rendering again for measurement
- real UI-style run shows about `13.07s` for clip 1 measurement
- total 12-clip measurement phase is about `151.59s`

## Scientific safety check

Unchanged:

- gradient-axis 3-band sphere model
- band semantics: `bright_side / center / dark_side`
- weights: `0.3 / 0.5 / 0.2`
- REDLine correction semantics
- thresholds

Also preserved:

- explicit sphere detection confidence / source / recovery behavior
- rendered IPP2 measurement as the operational truth path

## Tests

Passed:

- `py_compile`
- `tests/test_cli.py`

Latest result:

- `201 passed`

## Remaining non-blocking risk

The lightweight path is now architecturally correct, but still not cheap:

- the dominant cost is the full-resolution rendered-preview sphere-profile measurement itself

That cost is now honest and measurable. If future work is needed, it should optimize the real IPP2 profile-measurement implementation without reintroducing scene-domain proxy decisions.
