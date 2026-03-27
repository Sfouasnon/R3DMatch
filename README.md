# R3DMatch

R3DMatch is a RED multi-camera calibration and review workflow for matching exposure correction, color temperature, and tint across a camera array.

The project is built around an analysis-first, push-later model:

- measure a neutral target from a shared ROI
- compare strategies such as `median` and `optimal-exposure`
- generate operator-facing reports and trust diagnostics
- gate weak runs before any later camera writeback
- support read-only RCP2 camera inspection and verification over WebSocket JSON

The current product state is no longer a small prototype. R3DMatch now includes structured run gating, per-camera trust classes, lightweight and full review outputs, debug exposure traces, and read-only verification flows for connected RED cameras.

## What R3DMatch Does

- analyzes RED clip sets using shared neutral-target sampling
- computes exposure correction plus Kelvin / tint recommendations
- generates full contact-sheet or lightweight review packages
- explains trust with per-camera classifications, stability diagnostics, and run-level gating
- emits commit payloads for later camera writeback workflows
- supports RED RCP2 WebSocket read / verify flows for connected cameras

## Current Capabilities

### Analysis and Sampling

- `analyze` for lower-level scene/view clip analysis
- `review-calibration` for the full operator workflow
- shared ROI support for gray cards and gray spheres
- per-camera debug exposure traces, ROI overlays, and sampling summaries
- strategy comparison for:
  - `median`
  - `optimal-exposure`
  - `manual`
  - `hero-camera`

### Reporting

- full contact-sheet HTML output
- lightweight analysis output for faster operator review
- trust and stability charts
- per-camera trust summaries
- explicit run assessment and recommendation strength

### Trust and Gating

- per-camera trust classes:
  - `TRUSTED`
  - `USE_WITH_CAUTION`
  - `UNTRUSTED`
  - `EXCLUDED`
- run statuses:
  - `READY`
  - `READY_WITH_WARNINGS`
  - `REVIEW_REQUIRED`
  - `DO_NOT_PUSH`
- recommendation strength:
  - `HIGH_CONFIDENCE`
  - `MEDIUM_CONFIDENCE`
  - `LOW_CONFIDENCE`

### Verification

- simulated verification of intended camera targets
- comparison against saved read-only camera state reports
- live read-only verification against connected RED cameras
- explicit verification levels:
  - `VERIFIED`
  - `WITHIN_TOLERANCE`
  - `MISMATCH`
  - `NOT_AVAILABLE`

### RED Camera Integration

- documented RCP2 WebSocket JSON transport on port `9998`
- read current camera state with `read-camera-state`
- compare payload expectations to live camera state with `verify-camera-state --live-read`
- optional live apply commands remain available, but they are intentionally separate from review and should only be used in controlled operator workflows

## Current Limitations

- R3DMatch does not claim that every run is safe to push. `DO_NOT_PUSH` and `REVIEW_REQUIRED` are deliberate guardrails.
- Read-only verification is production-ready; automatic unattended camera pushes are not treated as universally safe.
- Exact replay of historical problem footage is only possible when that source footage is present in the workspace used for analysis.
- Periodic/subscription RCP2 support is scaffolded for future use, but not enabled as the default state manager.
- The RED decode backend still depends on a local RED SDK bridge build when `--backend red` is used.
- Preview generation depends on REDLine availability for real preview renders.

## Installation

### Required

```bash
cd /path/to/R3DMatch
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

### Optional: RED Decode Backend

Use this only when you need real RED decoding with `--backend red`.

```bash
export RED_SDK_ROOT=/path/to/RED_SDK
./scripts/build_red_sdk_bridge.sh
```

Notes:

- if `RED_SDK_ROOT` is unset, the project still installs and works with the deterministic mock backend
- the native bridge is required for `--backend red`
- the WebSocket RCP2 read / verify path does not depend on the legacy raw SDK transport

### Optional: REDLine Preview Rendering

If `REDLine` is not already on `PATH`, point R3DMatch at it explicitly:

```bash
export R3DMATCH_REDLINE_EXECUTABLE=/path/to/REDLine
```

## Quick Start

### 1. Run a Review

```bash
r3dmatch review-calibration /path/to/calibration_r3ds \
  --out ./runs/review_demo \
  --target-type gray_sphere \
  --processing-mode both \
  --backend mock \
  --matching-domain scene \
  --review-mode lightweight_analysis \
  --preview-mode calibration \
  --roi-x 0.20 --roi-y 0.20 --roi-w 0.40 --roi-h 0.40 \
  --clip-group 064 \
  --target-strategy median \
  --target-strategy optimal-exposure
```

This produces a run folder such as:

```text
./runs/review_demo/subset_064/
```

Key files:

- `summary.json`
- `report/review_validation.json`
- `report/contact_sheet.json`
- `report/contact_sheet.html`
- `report/debug_exposure_trace/summary.json`

### 2. Inspect the Trust Result

Look at:

- `report/review_validation.json` for machine-readable status
- `report/contact_sheet.html` for operator review
- `report/debug_exposure_trace/summary.json` for measurement diagnostics

### 3. Read Current Camera State

Read-only RCP2 camera query:

```bash
r3dmatch read-camera-state 10.20.61.191 --camera-label WIFI_RED
```

### 4. Verify a Payload Against Live Camera State

This is read-only. It compares the payload to what the camera currently reports.

```bash
r3dmatch verify-camera-state /path/to/run/report/calibration_commit_payload.json \
  --camera WIFI_RED \
  --live-read
```

## CLI Reference

### Core Analysis

- `r3dmatch analyze`
  - lower-level clip analysis
- `r3dmatch review-calibration`
  - main calibration review workflow
- `r3dmatch report-contact-sheet`
  - build operator review artifacts from an analysis directory
- `r3dmatch clear-preview-cache`
  - remove disposable preview artifacts only

### Calibration Utilities

- `r3dmatch calibrate-sphere`
- `r3dmatch calibrate-exposure`
- `r3dmatch calibrate-color`
- `r3dmatch calibrate-card`

### RMD / Preview / Validation

- `r3dmatch write-rmd`
- `r3dmatch transcode`
- `r3dmatch validate-pipeline`
- `r3dmatch approve-master-rmd`

### Camera State and Verification

- `r3dmatch read-camera-state`
  - read-only current camera values and camera info
- `r3dmatch verify-camera-state`
  - compare expected payload values to simulated or live-read camera state
- `r3dmatch apply-calibration`
  - dry-run by default; `--live` enables real writeback
- `r3dmatch apply-camera-values`
  - intentionally live camera write command; use only in controlled workflows
- `r3dmatch test-rcp2-write`
  - smoke-test write/readback/restore; intentionally invasive and not part of the safe quick start

### UI

- `r3dmatch desktop-ui`
  - launches the desktop wrapper around the operator interface

## Workflow Examples

### Analyze, Review, and Decide

1. Run `review-calibration`
2. Inspect:
   - `report/contact_sheet.html`
   - `report/review_validation.json`
   - `report/debug_exposure_trace/summary.json`
3. Check:
   - per-camera trust classes
   - overall run status
   - recommendation strength
4. Only plan writeback later if the run is strong enough

### Weak-Set Example

A weak set may produce:

- `run_assessment.status = DO_NOT_PUSH`
- `recommendation_strength = LOW_CONFIDENCE`
- several cameras marked `UNTRUSTED`

That means the analysis was useful, but the result is not safe to trust for later push without remeasurement.

### Read-Only Verification Before a Later Push

1. Generate or load a commit payload
2. Read current camera state with `read-camera-state`
3. Run `verify-camera-state --live-read`
4. Review per-parameter comparison before any future live write

## Output Artifacts

Typical review output contains:

- `summary.json`
  - aggregate analysis summary
- `array_calibration.json`
  - solved array-level calibration result
- `report/review_validation.json`
  - final review contract and run assessment
- `report/contact_sheet.json`
  - report payload consumed by HTML / operator surfaces
- `report/contact_sheet.html`
  - human-readable report
- `report/debug_exposure_trace/summary.json`
  - aggregate measurement diagnostics
- `report/debug_exposure_trace/<camera_id>.json`
  - per-camera measurement trace
- `report/debug_exposure_trace/<camera_id>.roi.svg`
  - ROI/sample overlay preview
- `report/calibration_commit_payload.json`
  - camera-target payload for later writeback workflows
- `report/rcp2_camera_state_report.json`
  - saved read-only camera-state snapshot
- `report/rcp2_verification_report.json`
  - structured verification output
- `report/post_apply_verification.json`
  - before/after review comparison when available

## Safety and Trust Model

### Trust Classes

- `TRUSTED`
  - stable reading, reference-eligible
- `USE_WITH_CAUTION`
  - usable but not a strong anchor candidate
- `UNTRUSTED`
  - unstable or low-confidence reading
- `EXCLUDED`
  - intentionally removed from reference formation

### Run Status

- `READY`
  - trustworthy enough to carry forward
- `READY_WITH_WARNINGS`
  - usable, but review the cautions before later push
- `REVIEW_REQUIRED`
  - analysis is informative, but operator judgment is required
- `DO_NOT_PUSH`
  - not safe to trust for later camera writeback

### Recommendation Strength

- `HIGH_CONFIDENCE`
- `MEDIUM_CONFIDENCE`
- `LOW_CONFIDENCE`

### Verification Modes

- `simulated_expected_state`
  - confirms intended targets and tolerance model only
- `camera_state_report_compare`
  - compares against a previously saved read-only camera state report
- `live_read_compare`
  - performs read-only camera queries before comparing

`DO_NOT_PUSH` exists to preserve operator trust. It is better for R3DMatch to say that a weak set is unsafe than to imply confidence it has not earned.

## Supporting Docs

- [Architecture](docs/architecture.md)
- [Matching Domains](docs/matching-modes.md)
- [Workflows](docs/workflows.md)
- [Verification](docs/verification.md)
- [RCP2 Transport](docs/rcp2.md)
- [Validation Model](docs/validation-plan.md)

## Development Notes

- run `python -m r3dmatch.cli --help` for the authoritative command surface
- use the mock backend for deterministic local testing when real RED decoding is not required
- generated runs and debug artifacts belong under `runs/` and are ignored by git by default
