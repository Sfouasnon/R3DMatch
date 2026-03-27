# Validation Model

R3DMatch validates results at several layers.

## 1. Measurement Validation

Per-camera measurement quality is evaluated through:

- gray sample stability
- chroma stability
- confidence scoring
- primary exposure group membership
- correction magnitude sanity

These signals drive trust classes and anchor eligibility.

## 2. Review Validation

Each review run writes `report/review_validation.json`, which records:

- artifact completeness
- physical validation outcome
- run assessment
- recommendation strength
- commit payload presence

## 3. Verification

Camera-target verification is separated into explicit modes:

- `simulated_expected_state`
- `camera_state_report_compare`
- `live_read_compare`

This keeps “expected values” separate from “camera actually reports these values.”

## 4. Post-Apply Comparison

When reacquired review footage exists, R3DMatch can compare:

- before review output
- after review output

That yields `report/post_apply_verification.json`.

## Validation Philosophy

R3DMatch prefers an honest warning or `DO_NOT_PUSH` status over a false pass. Weak data should be explained, not silently upgraded into confidence.
