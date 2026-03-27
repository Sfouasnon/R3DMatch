# Verification

R3DMatch uses verification to answer a specific question:

> Do the expected camera targets match what the system thinks the cameras should or do contain?

## Verification Modes

### `simulated_expected_state`

Uses payload targets only.

Good for:

- checking payload completeness
- documenting expected values
- testing tolerance logic

This mode does not query hardware.

### `camera_state_report_compare`

Compares payload targets against a previously saved read-only camera state report.

Good for:

- offline comparison
- reproducible checks in CI or handoff workflows

### `live_read_compare`

Performs read-only camera queries before comparison.

Good for:

- confirming current camera values before any later push
- validating that a payload still matches the connected camera state

## Verification Levels

- `VERIFIED`
- `WITHIN_TOLERANCE`
- `MISMATCH`
- `NOT_AVAILABLE`

## Current Tolerances

- `exposureAdjust`: tight
- `kelvin`: moderate
- `tint`: tight

The exact tolerance values are defined in the code so the report and CLI use the same comparison contract.

## Output Artifacts

Verification reports are written to:

- `report/rcp2_verification_report.json`

Saved camera-state snapshots are written to:

- `report/rcp2_camera_state_report.json`

## Safety Note

Read-only verification is the safest documented live-camera workflow in the repo. Verification is intentionally separated from camera writeback so operators can inspect state before any future push.
