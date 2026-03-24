# R3DMatch

Mac-first internal prototype for automatic exposure matching of RED R3D clips prior to transcoding.

## Prototype status

This standalone project is intentionally separate from `R3DSplat`.

Current prototype capabilities:

- defines a dedicated `r3dmatch` CLI
- supports `scene` and `view` analysis modes
- preserves monitoring/LUT context in emitted manifests
- computes deterministic stop offsets from sampled frames
- supports `full_frame`, `center_crop`, and detected ROI sampling for calibration work
- writes independent exposure and color calibration JSONs for later RMD-side integration
- writes per-clip analysis manifests and structured sidecars
- writes a first supported subset of per-clip `.RMD` files from those sidecars
- emits REDLine command plans for validation workflows
- scaffolds contact-sheet report metadata and HTML output from analysis results

Current prototype limitations:

- the RED SDK bridge currently implements the first decode milestone only: metadata plus single-frame half-res decode
- the bundled backend is a deterministic mock decoder for fast CLI validation
- REDLine flag mapping is left as a wrapper-owned integration point

## CLI

```bash
r3dmatch analyze /path/to/clip.R3D --mode scene --out ./out
r3dmatch analyze /path/to/folder --mode view --lut ./show.cube --out ./out
r3dmatch calibrate-exposure /path/to/folder --target-log2 -2.0 --sampling-mode center_crop --out ./cal/exposure
r3dmatch calibrate-color /path/to/folder --sampling-mode detected_roi --out ./cal/color
r3dmatch validate-pipeline /path/to/folder --analysis-dir ./out --out ./validation
r3dmatch report-contact-sheet ./out --out ./report
r3dmatch review-calibration /path/to/folder --out ./review --target-type gray_sphere --processing-mode both --backend red --roi-x 0.25 --roi-y 0.25 --roi-w 0.5 --roi-h 0.5
r3dmatch approve-master-rmd ./review
r3dmatch clear-preview-cache ./review
r3dmatch write-rmd /path/to/folder --analysis-dir ./out
r3dmatch transcode /path/to/clip.R3D --analysis-dir ./out --use-generated-sidecar --out ./renders
r3dmatch transcode /path/to/clip.R3D --analysis-dir ./out --use-generated-rmd --out ./renders
streamlit run src/r3dmatch/ui.py -- --output-folder ./out
```

## Layout

```text
R3DMatch/
├── README.md
├── docs/
├── scripts/
├── src/r3dmatch/
└── tests/
```

## RED Bridge

The RED SDK integration is isolated behind the Python backend layer in `src/r3dmatch/sdk.py`.
The native bridge lives in `src/r3dmatch/native/` and only exposes low-level metadata/decode calls.

Build note:

```bash
export RED_SDK_ROOT=/path/to/RED_SDK
./scripts/build_red_sdk_bridge.sh
```

Expected native contract:

- `read_metadata(path)` returns a metadata dict only
- `decode_frame(path, ...)` returns an `HxWx3` `float32` image array

If `RED_SDK_ROOT` is unset, the bridge still builds as a stub, but `--backend red` will raise a clear runtime error.
If `RED_SDK_ROOT` is set, the first supported decode path is frame 0 at half resolution with `REDWideGamutRGB` and `Log3G10`.
Set `R3DMATCH_RED_DEBUG=1` to print native decode checkpoints and SDK status codes to stderr while debugging bridge failures.

Why isolated:

- clip identity and grouping must stay in Python
- calibration and luminance logic must stay in Python
- the native layer should only read metadata and decode frames

## Calibration Notes

Edge exclusion matters because lens shading, vignetting, rigs, matte boxes, and partial occlusions can contaminate full-frame statistics.
`center_crop` and detected ROI sampling keep calibration focused on the most stable neutral region instead of averaging edge behavior into the solve.

Exposure and color are separate passes on purpose:

- exposure calibration solves per-group stop offsets from robust log-luminance statistics
- color calibration solves per-group neutral RGB gains from trimmed channel medians

Color matching cannot rely on Kelvin metadata alone. A camera can report the same white-balance metadata while still rendering different neutral-channel balance because of sensor, lens, filtration, flare, or pipeline differences.

The generated `exposure_calibration.json` and `color_calibration.json` files are intended to feed a later RMD-generation step independently:

- exposure only
- color only
- or both together

## Sidecars, RMDs, And REDLine

The current sidecar format is an exact per-clip intermediate JSON named from `clip_id` only:

- `G007_D060_0324M6_001.sidecar.json`
- generated RMD name: `G007_D060_0324M6_001.RMD`

The sidecar remains the canonical intermediate contract. The current RMD writer generates an initial supported subset from it, using exact `clip_id` naming only:

- exposure offsets and applied baseline state
- color gains as pending/applied metadata
- calibration provenance paths
- exact clip identity for downstream REDLine/RMD mapping

The current `.RMD` subset writes:

- exposure offset from `final_offset_stops`
- exposure calibration-loaded state
- applied exposure baseline
- optional neutral RGB gains
- color gain state and basic provenance

This is intentionally not a full camera-metadata round trip yet.

Generate `.RMD` files directly from an analysis folder with:

```bash
r3dmatch write-rmd /path/to/folder --analysis-dir ./out
```

REDLine planning can use either generated sidecars or generated `.RMD` files and keeps the exact clip-to-metadata pairing deterministic:

- original
- exposure
- color
- both

Use generated `.RMD` files in planning with:

```bash
r3dmatch transcode /path/to/folder --analysis-dir ./out --use-generated-rmd --out ./renders
```

Use sidecars instead with:

```bash
r3dmatch transcode /path/to/folder --analysis-dir ./out --use-generated-sidecar --out ./renders
```

## Review UI

The fastest local review app is a small Streamlit page that reads one analyze output folder directly.

It expects:

- `summary.json`
- `array_calibration.json`
- `analysis/*.analysis.json`
- `sidecars/*.sidecar.json`
- optional generated previews under `previews/`

Generate previews plus a real contact-sheet package from an analyze output folder with:

```bash
r3dmatch report-contact-sheet ./out --out ./out/report --preview-mode calibration
```

This writes:

- `./out/previews/<clip_id>.original.review.jpg`
- `./out/previews/<clip_id>.exposure.review.jpg`
- `./out/previews/<clip_id>.color.review.jpg`
- `./out/previews/<clip_id>.both.review.jpg`
- `./out/report/contact_sheet.json`
- `./out/report/preview_contact_sheet.pdf`
- `./out/report/contact_sheet.html`
- `./out/report/review_manifest.json`

Preview rendering now uses `REDLine` and exposes two review modes:

- `calibration` (default): calibration-safe preview using `REDWideGamutRGB / Log3G10 / Medium / Medium`
- `monitoring`: optional operator-facing monitoring preview, with optional `--preview-lut /path/to/show.cube`

The authoritative exposure solve still uses the calibration preview path, even if monitoring previews are requested for the visible review package.

The four preview variants remain:

- `original`
- `exposure`
- `color`
- `both`

`report-contact-sheet` writes those previews under `previews/` and then builds both `preview_contact_sheet.pdf` and `contact_sheet.html`. The preview PDF is now the preferred review artifact for operators because it does not depend on browser rendering. Set `R3DMATCH_REDLINE_EXECUTABLE` if `REDLine` is not on `PATH`.

Each review package also records the detected REDLine capabilities and the exact preview settings used, including whether LUT support was available and whether the chosen preview-space arguments were applied.

## Operator Workflow

1. Run review:

```bash
r3dmatch review-calibration /path/to/calibration_r3ds --out ./review --target-type gray_sphere --processing-mode both --backend red --roi-x 0.25 --roi-y 0.25 --roi-w 0.5 --roi-h 0.5 --target-strategy median --target-strategy brightest-valid --target-strategy manual --reference-clip-id G007_D060_0324M6_001 --preview-mode calibration --preview-output-space REDWideGamutRGB --preview-output-gamma Log3G10 --preview-highlight-rolloff medium --preview-shadow-rolloff medium
```

This generates:

- analysis outputs
- `array_calibration.json`
- temporary review RMDs under `review_rmd/`
- REDLine review previews under `previews/`
- `report/contact_sheet.json`
- `report/contact_sheet.html`
- `report/preview_contact_sheet.pdf`

Original previews are rendered once as shared references, and each requested target strategy gets its own corrected preview set:

- `<clip_id>.original.review.<run_id>.jpg`
- `<clip_id>.exposure.review.<strategy>.<run_id>.jpg`
- `<clip_id>.color.review.<strategy>.<run_id>.jpg`
- `<clip_id>.both.review.<strategy>.<run_id>.jpg`

For gray-card and gray-sphere review, the preferred path is to provide a shared normalized ROI with `--roi-x/--roi-y/--roi-w/--roi-h`. When present, the calibration solve uses only that ROI for luminance, chromaticity, and confidence measurements. If no ROI is provided, the workflow falls back to the broader center-region measurement path.

Exposure matching in the review workflow is now evaluated primarily in monitoring conditions rather than raw-domain luminance alone. The primary exposure metric is the ROI luminance after a shared rendered-preview measurement pass aligned to the calibration preview assumptions:

- REDWideGamutRGB output space
- Log3G10 gamma
- Medium tone map
- Medium highlight roll-off
- Medium shadow roll-off

Raw-domain ROI luminance is still retained in the analysis and report outputs as diagnostics:

- `measured_log2_luminance_monitoring`: primary solve value
- `measured_log2_luminance_raw`: diagnostic only

This keeps the operator-facing exposure solve closer to the same conditions used to judge the preview PDF and review stills, while preserving raw-domain visibility for engineering/debug work.

`--target-strategy` can be repeated to compare multiple array targets in one review package:

- `median`
- `brightest-valid`
- `manual` (requires `--reference-clip-id`)

2. Inspect `report/preview_contact_sheet.pdf` as the primary review artifact, then optionally use `report/contact_sheet.html` or the Streamlit UI for secondary inspection.

3. If the result is not acceptable, rerun or clear disposable review renders safely:

```bash
r3dmatch clear-preview-cache ./review
```

This removes only preview/review artifacts such as review JPGs, preview command logs, and contact-sheet review files. It does not remove analysis JSON, calibration JSON, approved `Master_RMD`, or approval PDFs.

4. Approve the reviewed calibration and generate authoritative master outputs:

```bash
r3dmatch approve-master-rmd ./review --target-strategy manual --reference-clip-id G007_D060_0324M6_001
```

This generates:

- `approval/Master_RMD/<clip_id>.RMD`
- `approval/approval_manifest.json`
- `approval/calibration_report.pdf`

Only the selected review strategy is promoted into `Master_RMD`.

5. Archive `approval/calibration_report.pdf` and the approval manifest with the approved `Master_RMD` folder as the permanent record of the decision.

Launch it from the project root with:

```bash
streamlit run src/r3dmatch/ui.py -- --output-folder /path/to/out
```

If preview stills exist under `previews/` or `stills/` in that output folder, the app will show them automatically. If not, the UI still works without blocking.

For the preferred internal operator UI, launch the local web app:

```bash
cd ~/Desktop/R3DMatch
setenv PYTHONPATH "$PWD/src"
python3 -m r3dmatch.web_app
```

This starts a local server at:

```bash
http://127.0.0.1:5000
```

The web UI is an internal-only Flask wrapper around the existing CLI/workflow. It uses text inputs for paths and server-side validation instead of OS file pickers. It lets an operator:

- choose a calibration folder and scan it recursively for RED `.R3D` clips inside `.RDC` containers or plain folders
- choose backend, target type, processing mode, ROI, strategies, and manual reference clip
- choose `calibration` vs `monitoring` preview mode
- optionally select a `.cube` monitoring LUT
- run review calibration
- inspect logs and the exact command being run
- open the generated report/output folder
- approve a selected strategy into `Master_RMD`
- clear preview cache

The intended source-selection flow is folder-first: operators paste a calibration folder path, the web UI discovers RED clips automatically after scan, and the source summary panel shows clip count plus sample clip IDs before review is run.

Logo behavior:

- the bundled project logo at `src/r3dmatch/static/r3dmatch_logo.png` is served in the header when available
- if the image cannot be loaded, the UI falls back to a text-only header and continues normally

For a quick import check without starting the server:

```bash
cd ~/Desktop/R3DMatch
setenv PYTHONPATH "$PWD/src"
python3 -m r3dmatch.web_app --check
```
