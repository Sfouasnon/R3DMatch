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
