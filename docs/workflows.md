# Workflows

## Standard Review Workflow

1. Run `review-calibration` on a clip set or subset.
2. Inspect:
   - `report/contact_sheet.html`
   - `report/review_validation.json`
   - `report/debug_exposure_trace/summary.json`
3. Check:
   - trust classes
   - run status
   - recommendation strength
4. Only carry a run forward when the assessment supports it.

## Weak-Set Workflow

If a run lands in `REVIEW_REQUIRED` or `DO_NOT_PUSH`:

- inspect the trusted vs untrusted cameras
- review stability and confidence signals
- identify outliers or inconsistent readings
- remeasure or rerun rather than treating the result as push-ready

## Read-Only Camera Verification Workflow

1. Read current camera state:

   ```bash
   r3dmatch read-camera-state 10.20.61.191 --camera-label WIFI_RED
   ```

2. Compare a payload to live state without writing:

   ```bash
   r3dmatch verify-camera-state /path/to/report/calibration_commit_payload.json \
     --camera WIFI_RED \
     --live-read
   ```

3. Review:
   - verification level
   - mismatched fields
   - tolerance notes

## Later Push Workflow

Writeback exists, but it should remain an explicit later step:

- review first
- verify current state
- confirm the run is strong enough
- only then consider live apply commands in a controlled operator workflow
