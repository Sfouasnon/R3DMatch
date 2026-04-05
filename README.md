# R3DMatch

**R3DMatch** is an IPP2-first calibration and verification system for multi-camera RED workflows. R3DMatch analyzes a gray sphere using sci-kit, samples RGB data in a pixel array in three bands across the middle of the sphere. That data is translated into luminance, then IRE to match on set workflows on display transformed R3Ds. R3DMatch computes exposure, white balance, and tint corrections, and produces a dense, diagnostic contact sheet for validation. These changes can be pushed to RED cameras over network via RCP2 using --exposureAdjust --kelvin --tint for live camera matching. 

---

## Core Concept

R3DMatch operates entirely in the **monitoring domain (IPP2-rendered space)**:

- Cameras are set to an appropriate exposure on set
- Single frame R3Ds are captured of a Gray Sphere in place of a subject
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

### 2. Health check

Recommended operator preflight:

```bash
/bin/sh scripts/run_r3dmatch_web.sh 5000
```

This launcher:

- uses the project runtime
- sets `PYTHONPATH`
- auto-populates `DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib` on macOS when it is unset and Homebrew libs are available there
- preserves externally set `RED_SDK_ROOT` and `RED_SDK_REDISTRIBUTABLE_DIR`
- runs `r3dmatch runtime-health --strict --require-red-backend` before opening the web UI

Direct health check:

```bash
./.venv/bin/python -m r3dmatch.cli runtime-health --strict --require-red-backend
```

### 3. Run review

Recommended web path:

1. Launch the app with `/bin/sh scripts/run_r3dmatch_web.sh 5000`
2. Open `http://127.0.0.1:5000`
3. Scan the calibration folder
4. Run `Full Contact Sheet`

Direct CLI wrapper:

```bash
/bin/sh scripts/run_r3dmatch_review.sh /path/to/r3d/files \
  --out /path/to/run \
  --target-type gray_sphere \
  --processing-mode both \
  --backend red \
  --review-mode full_contact_sheet \
  --preview-mode monitoring \
  --target-strategy median
```

### 4. FTPS ingest workflow

Recommended FTPS operator flow in the web UI:

1. Launch with `/bin/sh scripts/run_r3dmatch_web.sh 5000`
2. Set `Source Mode = FTPS Camera Pull`
3. Enter:
   - `FTPS Reel`
   - `Clip Numbers / Ranges`
   - optional `Camera Subset`
   - optional `Local Ingest Cache Root`
4. Use one of:
   - `Discover`
   - `Download`
   - `Download + Process`
   - `Retry Failed`

Direct CLI ingest wrapper:

```bash
/bin/sh scripts/run_r3dmatch_ingest.sh \
  --action discover \
  --out /path/to/ingest \
  --ftps-reel 007 \
  --ftps-clips 63 \
  --ftps-camera GA
```

Direct CLI download + process:

```bash
./.venv/bin/python -m r3dmatch.cli ftps-download-process \
  --out /path/to/review \
  --ftps-local-root /Volumes/RAID/ingest/063 \
  --ftps-reel 007 \
  --ftps-clips 63 \
  --target-type gray_sphere \
  --processing-mode both \
  --backend red \
  --review-mode full_contact_sheet \
  --preview-mode monitoring \
  --target-strategy median
```

Process existing local ingest without re-downloading:

```bash
./.venv/bin/python -m r3dmatch.cli process-local-ingest \
  /Volumes/RAID/ingest/063 \
  --out /path/to/review \
  --target-type gray_sphere \
  --processing-mode both \
  --backend red \
  --review-mode full_contact_sheet \
  --preview-mode monitoring \
  --target-strategy median
```

Key outputs:

- `report/contact_sheet.html`
- `report/preview_contact_sheet.pdf`
- `report/contact_sheet.json`
- `report/review_package.json`
- `report/ipp2_validation.json`

## RED SDK Setup

The RED SDK is **not** stored in this repo.

For the native RED backend:

1. Install the RED SDK externally.
2. Set `RED_SDK_ROOT` to the SDK root, for example:

```bash
export RED_SDK_ROOT=/Users/sfouasnon/Desktop/R3DSplat_Dependecies/RED_SDK/R3DSDKv9_2_0
```

3. If the redistributable runtime is not under the SDK root, also set:

```bash
export RED_SDK_REDISTRIBUTABLE_DIR=/path/to/Redistributable/mac
```

4. Rebuild the native bridge:

```bash
scripts/build_red_sdk_bridge.sh
```

R3DMatch no longer uses `src/RED_SDK/...` as an operational SDK fallback.

## HTML-Master Contact Sheet

The contact sheet is authored once in HTML and the final PDF is exported from that HTML.

- `contact_sheet.html` is the master artifact
- `preview_contact_sheet.pdf` is generated from the HTML export path
- no parallel custom PDF layout path is used for the contact sheet

## macOS App Build

Build a portable Apple Silicon app bundle:

```bash
/bin/sh scripts/build_macos_app.sh
```

Smoke-check the bundle:

```bash
dist/R3DMatch.app/Contents/MacOS/R3DMatch --check
```

Zip it for transport:

```bash
/bin/sh scripts/package_macos_app.sh
```

Runtime assumptions on the destination Mac:

- Apple Silicon macOS
- external `RED_SDK_ROOT` if RED backend is required
- optional `RED_SDK_REDISTRIBUTABLE_DIR`
- Homebrew cairo / pango / glib libraries available under `/opt/homebrew/lib` for WeasyPrint HTML→PDF export

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
