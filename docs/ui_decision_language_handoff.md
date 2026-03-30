## UI Decision Language And Workflow Cleanup Handoff

### Scope

This pass cleaned up the operator-facing workflow and decision language without changing the measurement architecture.

It did **not** change:

- the IPP2-first measurement path
- the rendered-preview measurement source
- the 3-band sphere model
- exposure solve math
- validation thresholds
- report reuse of stored analysis measurements

### Workflow Cleanup

The normal Web UI workflow now reflects the actual active system:

- measurement domain is fixed to `Perceptual (IPP2 / BT.709 / BT.1886)`
- Scene-Referred matching is no longer part of the primary UI flow
- manual ROI controls are no longer part of the primary UI flow

Manual ROI still exists only under:

- `Advanced / Debug Controls`

This keeps fallback/debug capability without implying that ROI drawing is part of the standard sphere-calibration workflow.

### Decision Language Changes

The top decision surface no longer uses generic CI/CD-style wording like:

- `SAFE TO COMMIT`
- `DO NOT COMMIT`
- raw confidence-first framing

It now uses calibration language based on stored sphere measurements:

- `Array is within calibration tolerance`
- `Array is near calibration tolerance`
- `Array is out of calibration`
- `Retained gray sphere center values span X IRE to Y IRE`
- `Closest current reference candidate: CAMERA`

The operator notes section was also renamed to be calibration-specific:

- `Calibration Review Notes`

and measurement/trust wording was softened from software confidence language toward:

- `Measurement Stability`
- retained vs excluded camera counts
- reference-candidate language

### Best Reference Candidate Definition

The displayed reference candidate is **not** presented as absolute creative truth.

It is defined as:

- the retained camera with the smallest absolute offset to the current anchor target
- tie-broken by trust score
- then by measurement confidence

This is intended to mean:

- closest current fit to the retained cluster target

not:

- objectively correct exposure for every scene

### Range Definition

The decision banner range is based on:

- retained cameras only
- `Center IRE` extracted from stored `measured_gray_exposure_summary`

So the displayed range is:

- retained-cluster center-band sphere exposure range in the same monitoring domain already used by the solve

No extra measurement pass was introduced.

### Data Fields Used

The new summary language is driven by existing stored review/report fields:

- `per_camera_analysis[*].measured_gray_exposure_summary`
- `per_camera_analysis[*].reference_use`
- `per_camera_analysis[*].camera_offset_from_anchor`
- `per_camera_analysis[*].trust_score`
- `per_camera_analysis[*].confidence`
- `run_assessment.status`
- `run_assessment.operator_note`
- `run_assessment.gating_reasons`

### Files Changed

- `src/r3dmatch/web_app.py`
- `src/r3dmatch/report.py`
- `tests/test_cli.py`

### Validation

Validated with:

- `py_compile`
- full `tests/test_cli.py`

The UI/report wording changes are now aligned with the actual current architecture:

- IPP2-first
- sphere-based
- retained-cluster-aware
- operator-facing rather than CI/CD-facing
