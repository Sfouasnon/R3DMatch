# R3DMatch v2 — Architecture

## Why a Rebuild

The original system works. Exposure solve is accurate. The core math is sound.
But auto sphere detection failed on all 12 cameras in every test run — operators
are manually placing ROIs every time. The detection code is 35,000 lines because
it accumulated fallback after fallback trying to compensate for a detection model
that doesn't commit hard enough early. The WB closed-loop reports FAIL on every
camera because the gate threshold is physically impossible to satisfy in IPP2
BT.709 output (IPP2 has an inherent ~0.13 warm/cool bias on gray at 5600K).

The rebuild is smaller, harder-gated, and more transparent about what it knows.

## What's Kept Exactly

- The four-step measurement math (luminance → trim → log2 → IRE). Correct, proven.
- The REDLine render command: colorSciVersion=3, colorSpace=13 (BT.709),
  gammaCurve=32 (BT.1886), outputToneMap=1 (medium), rollOff=3 (medium).
- The exposure solve (offset in stops, median anchor, 0.05-stop tolerance). Accurate.
- The `--printMeta 1` metadata extraction (Kelvin, tint, ISO, camera ID, timecode).
- The progress emission contract (JSON on stdout, phase/pct/detail).
- The output schema (contact_sheet.json, analysis/*.analysis.json, per-camera commit values).

## What's Different

### Sphere Detection — Strict Sequential Gates

The original system has ~8 detection strategies with fallback chains. The rebuild
has ONE strategy with hard sequential gates. If a gate fails, detection fails.
No fallbacks. No confidence-weighted averaging. Either it's a sphere or it isn't.

Gate order (each must pass before the next runs):
1. **Geometry**: Hough circle detected with accumulator >= threshold. Circle must
   be within 10-30% of min(frame_height, frame_width) radius. Aspect ratio of
   bounding box <= 1.15. This is the only Hough pass. No multi-scale retry.
2. **Gray material**: Mean chromaticity of interior pixels within 0.04 of (1/3, 1/3).
   The sphere must be neutral. This gates out skin tones, colored objects, etc.
3. **Lambertian falloff**: The sphere must have a radial luminance gradient.
   Divide interior into 4 radial rings. Luminance must decrease monotonically
   outward from the detected center (allowing 10% variance per ring). Flat objects
   fail here.
4. **IRE spread**: The three zones (bright/center/dark) must span at least 1.5 IRE.
   A completely flat sphere (uniform studio lighting from front) fails here.
5. **Interior stddev**: Must be in [0.003, 0.16]. Too low = overlit/clipped.
   Too high = textured surface or bad detection.

On failure: detection_status = FAILED. The UI offers manual ROI placement, which
goes through gates 2-5 (gate 1 bypassed — operator confirmed it's a circle).

### WB Closed-Loop Gate — Recalibrated

Original gate: abs(WC_residual) < 0.015 and abs(GM_residual) < 0.015.
Problem: In IPP2 BT.709 output, a perfect gray sphere at 5600K has inherent
WC_residual ≈ -0.127 due to the pipeline's warm bias. This is not correctable
with --kelvin alone.

New gate: The solve is evaluated on **inter-camera spread improvement**, not
absolute residual. A run PASSES if:
- WC_spread_after < 0.008 (was 0.018 before solve in the test run → 0.0044 after)
- GM_spread_after < 0.005 (was 0.015 before → 0.0015 after)
- All cameras used the same number of iterations

The corrected_sample_warm_cool_residual is reported diagnostically but does not
gate the pass/fail. What matters is that cameras match each other, not that they
match absolute neutral — this is a camera matching tool, not a color correction tool.

### Scientific Validation — Fixed Classifier

The original `blocked_asset_mismatch` was a false positive: SHA-256 matched,
dimensions matched, but the replay used a different code path than the original
measurement (manual ROI vs auto detection). The rebuild separates provenance
tracking from replay validation — they're independent checks.

### Lens Metadata — Opportunistic

`--printMeta 5` runs after `--printMeta 1`. If Aperture or Focal Length are
non-zero, lens fields are added to the clip card. If zero (no /i lens), the
lens section is omitted entirely from the report. No "N/A" labels.

### No RED SDK Dependency

All analysis goes through REDLine. The `_red_sdk_bridge.so` is not used.
The web app checks for REDLine at startup and fails clearly if not found.

## File Map

```
src/r3dmatch3/
  models.py          — dataclasses: ClipMetadata, SphereROI, MeasurementResult,
                        SolveResult, CommitValues, RunResult
  redline.py         — REDLine subprocess: render, printMeta1, printMeta5
  sphere.py          — sphere detection: gated pipeline + LoG supplement
  sphere_profile.py  — per-camera sphere profile store (v2 schema: geometry + photometric priors)
  brdf_gate.py       — BRDF spoke-correlation gate (abs Pearson trimmed mean)
  brdf_verify.py     — standalone BRDF verification tooling
  measure.py         — 4-step measurement math, zone geometry
  solve.py           — exposure solve, WB solve, inter-camera spread gates
  report.py          — contact sheet HTML renderer only
  workflow.py        — orchestration: scan→render→detect→measure→solve→report
  workflow_qc.py     — re-measurement / QC pass
  rcp2.py            — RCP2 WebSocket push: kelvin/tint/exposureAdjust commits
  app.py             — PySide6 desktop UI (main entry point)
  web_app.py         — Flask server, /api routes
  progress.py        — JSON progress emission
  settings.py        — REDLine path resolution, settings persistence
```

## Commit Values Contract

Per camera, the final output for RCP2 push:
```json
{
  "exposureAdjust": -0.153,
  "kelvin": 5609,
  "tint": -1.2
}
```

exposureAdjust: stops offset from camera's native exposure.
kelvin: absolute Kelvin value (not delta) — set this on the camera directly.
tint: absolute tint value.

## REDLine Render Command

Measurement render (full res TIFF, frame 0):
```
REDline --i <clip.R3D> --o <out.tiff>
  --format 1              # TIFF
  --start 0 --frameCount 1
  --colorSciVersion 3     # IPP2
  --colorSpace 13         # BT.709
  --gammaCurve 32         # BT.1886
  --outputToneMap 1       # medium
  --rollOff 3             # medium
  --shadow 0.0
  --useMeta               # use camera's as-shot metadata
  --silent
```

REDLine appends `.000000.tif` to the output filename — always glob for it.

## Sphere Detection — Implementation Notes

Input: HxWx3 float32 normalized [0,1] RGB from PIL load.
Resize to detection plane: max(min_dim, 1080) if original > 2K. Run gates at
detection scale, then map center/radius back to full resolution.

Hough parameters for 5760x3240 KOMODO-X frames:
- Typical sphere: r ≈ 3-7% of min(H,W) = 97-227 pixels at full res.
- Detection scale: downsample to 1080p → sphere is ~32-75 pixels.
- Canny: sigma=2.0, low=0.03, high=0.12.
- Hough radii: 24 to 200 pixels (at detection scale), step=3.

## WB Solve — Implementation Notes

Model: Shared Kelvin / Per-Camera Tint.

Step 1: Collect neutral_rgb for each camera from inner 78% of sphere mask.
Step 2: Compute WC = (R-B)/(R+G+B) and GM = (G - 0.5*(R+B))/(R+G+B).
Step 3: Shared kelvin = round median of as-shot kelvin values (from --printMeta 1).
Step 4: Per-camera tint = solve such that GM matches group median GM.
Step 5: Validate: WC_spread_after < 0.008 and GM_spread_after < 0.005 → PASS.

Response matrix (empirical from test run):
- dWC/dKelvin ≈ 7.19e-05 (per Kelvin unit)
- dGM/dTint ≈ 3.29e-03 (per tint unit)
These are used to estimate correction magnitudes, not to invert analytically.
