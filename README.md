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

Launch it from the project root with:

```bash
streamlit run src/r3dmatch/ui.py -- --output-folder /path/to/out
```

If preview stills exist under `previews/` or `stills/` in that output folder, the app will show them automatically. If not, the UI still works without blocking.
