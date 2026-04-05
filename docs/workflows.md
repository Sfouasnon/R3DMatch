# Workflows

## Recommended Operator Path

### 1. Launch the web UI

```bash
/bin/sh scripts/run_r3dmatch_web.sh 5000
```

What this does:

- uses the project runtime
- exports `PYTHONPATH`
- auto-fills `DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib` on macOS when needed
- preserves `RED_SDK_ROOT`
- preserves `RED_SDK_REDISTRIBUTABLE_DIR`
- runs a strict runtime preflight before starting the app

Open:

- `http://127.0.0.1:5000`

### 2. Run a full review

In the web UI:

1. scan the calibration folder
2. keep `Measurement Domain = Perceptual (IPP2 / BT.709 / BT.1886)`
3. choose `Full Contact Sheet`
4. run review

Primary outputs:

- `report/contact_sheet.html`
- `report/preview_contact_sheet.pdf`
- `report/contact_sheet.json`
- `report/review_package.json`
- `report/ipp2_validation.json`

### 3. Run FTPS ingest as a real workflow

In the web UI:

1. Set `Source Mode = FTPS Camera Pull`
2. Enter `FTPS Reel`, `Clip Numbers / Ranges`, and optional `Camera Subset`
3. Optionally set `Local Ingest Cache Root` to a mounted drive or existing ingest volume
4. Use:
   - `Discover` to verify reachable cameras and matching clips before transfer
   - `Download` to pull matching media only
   - `Download + Process` to ingest and immediately run review
   - `Retry Failed` to reuse the last ingest manifest and retry unsuccessful cameras

Direct CLI ingest:

```bash
/bin/sh scripts/run_r3dmatch_ingest.sh \
  --action download \
  --out /path/to/ingest \
  --ftps-reel 007 \
  --ftps-clips 63 \
  --ftps-camera GA
```

Direct CLI ingest + process:

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

Process a prior ingest without re-downloading:

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

## Recommended CLI Health Check

```bash
./.venv/bin/python -m r3dmatch.cli runtime-health --strict --require-red-backend
```

This reports:

- interpreter path
- active virtual environment
- `DYLD_FALLBACK_LIBRARY_PATH`
- `RED_SDK_ROOT`
- resolved RED SDK redistributable directory
- WeasyPrint importability
- whether HTML→PDF export should work

## Direct CLI Review Path

If you want a single shell command instead of the web UI:

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

## Troubleshooting

- If `runtime-health` reports missing WeasyPrint native libraries, fix the runtime before starting review.
- If `RED_SDK_ROOT` is missing, export it and relaunch.
- If HTML loads but PDF export fails, inspect the `pdf_export_preflight` block in `contact_sheet.json`.
- If FTPS retry says no manifest is available, confirm `ingest_manifest.json` exists under the chosen local ingest root.

## Packaging

Build the portable Apple Silicon app bundle:

```bash
/bin/sh scripts/build_macos_app.sh
```

Package it for transfer:

```bash
/bin/sh scripts/package_macos_app.sh
```

Smoke-check the built app:

```bash
dist/R3DMatch.app/Contents/MacOS/R3DMatch --check
```
