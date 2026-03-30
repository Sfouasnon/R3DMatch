# R3DMatch

**R3DMatch** is an IPP2-first calibration and verification system for multi-camera RED workflows.

It analyzes gray sphere captures across cameras, computes exposure and white balance corrections, and produces a dense, diagnostic contact sheet for validation.

This is not a presentation tool.  
It is an operator-facing calibration instrument.

---

## Core Concept

R3DMatch operates entirely in the **monitoring domain (IPP2-rendered space)**:

- Cameras are evaluated as they are viewed on set
- Gray sphere sampling is used as the measurement anchor
- Corrections are solved against perceptual output, not raw sensor space

This ensures alignment with:
- DIT workflows
- waveform / false color verification
- real-world monitoring conditions

---

## What It Does

### Calibration
- Detects gray sphere in each camera
- Samples multiple regions (S1 / S2 / S3)
- Computes exposure offset and WB adjustments
- Outputs per-camera correction values

### Verification
- Applies corrections via RED pipeline
- Re-renders corrected frames
- Computes residual error per camera
- Flags anomalies and fallback cases

### Reporting
- Generates a **diagnostic contact sheet**
- Shows:
  - original vs corrected frames
  - gray sphere sample values
  - applied corrections
  - residual error
  - solve overlay for verification

---

## Contact Sheet (Key Output)

The contact sheet is the primary deliverable.

It is designed to:

- expose even subtle exposure / color mismatches
- allow fast scan across large camera arrays
- tightly couple imagery and measurement data
- function as a technical verification surface

Not a dashboard. Not a summary.

---

## Workflow

### 1. Capture
- Multi-camera array
- Even lighting
- Gray sphere visible to all cameras
- Verified via IPP2 monitoring (false color / waveform)

### 2. Run calibration

```bash
r3dmatch review-calibration \
  --input path/to/r3d/files \
  --output runs/session_name

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
