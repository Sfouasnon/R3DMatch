# FTPS Ingest + Packaging Handoff

## FTPS audit

Before this pass, FTPS already had real foundations in `src/r3dmatch/ftps_ingest.py`:

- camera/IP map
- reel + clip parsing
- request planning
- recursive remote match discovery
- download execution
- retry loop
- `ingest_manifest.json`

What was still only source-mode plumbing:

- web UI only planned FTPS requests during scan
- `review_calibration(...)` hid ingest internally instead of exposing discovery/download/retry stages
- CLI only offered a single fused `ingest-ftps` download path
- there was no operator-facing distinction between:
  - discover only
  - download only
  - download then process
  - process existing local ingest
  - retry failed cameras

Where ingest now ends and processing begins:

- ingest writes media + `ingest_manifest.json` into the chosen local ingest root
- processing starts from that local ingest root exactly like a local-folder review
- `review_calibration(...)` still owns analysis/report/calibration once local media is available

## New FTPS workflow

Core implementation now lives in `src/r3dmatch/ftps_ingest.py`.

Explicit stages:

- `discover_ftps_batch(...)`
- `download_ftps_batch(...)`
- `retry_failed_ftps_batch(...)`
- `run_ftps_ingest_job(...)`
- `load_ingest_manifest(...)`
- `ingest_manifest_path_for(...)`

The richer ingest manifest now captures:

- requested reel / clip spec / camera subset
- reachable vs unreachable cameras
- cameras with matches vs reachable cameras with no matches
- matched files
- downloaded files
- skipped-existing files
- failed files
- per-camera attempts / errors / status
- estimated bytes
- transferred bytes
- whether processing was requested after ingest

Schema note:

- legacy callers can still use `ingest_ftps_batch(...)`
- it now routes through the explicit download path and writes the richer v2 manifest

## Web/UI changes

`src/r3dmatch/web_app.py` now exposes FTPS as an operator workflow instead of scan-time scaffolding.

New FTPS controls:

- optional `Local Ingest Cache Root`
- `Discover`
- `Download`
- `Download + Process`
- `Retry Failed`

Behavior:

- `Plan Request` still gives a fast non-network summary
- `Discover` runs the real FTPS discovery pass
- `Download` pulls only media
- `Download + Process` reuses the stabilized FTPS review path
- `Retry Failed` reuses the last ingest manifest under the chosen ingest root

Operator visibility:

- execution state now accepts `ingest-ftps` tasks as first-class jobs
- the UI renders an `FTPS Ingest` surface from the current ingest manifest
- per-camera status is visible after discovery/download/retry

## Normalized ingest handoff

`src/r3dmatch/workflow.py` now accepts:

- `ftps_local_root`

For FTPS review runs:

- if `ftps_local_root` is provided, ingest goes there
- otherwise ingest still defaults to `<review output>/ingest`
- analysis then runs against that local ingest root as standard local media

This keeps processing agnostic to whether media came from:

- a local folder
- a new FTPS pull
- an existing ingest volume

## CLI/operator entry points

New/updated commands in `src/r3dmatch/cli.py`:

- `ingest-ftps --action discover|download|retry-failed`
- `ftps-download-process`
- `process-local-ingest`
- `review-calibration --ftps-local-root ...`

Stable shell entry points:

- `scripts/run_r3dmatch_web.sh`
- `scripts/run_r3dmatch_review.sh`
- `scripts/run_r3dmatch_ingest.sh`

## Packaging audit

Remaining portability blockers that were addressed in this pass:

- `WeasyPrint` was part of the real HTML→PDF runtime but missing from `pyproject.toml`
- there was no buildable macOS app path documented in-repo
- there was no packaged entrypoint that launched the stabilized web runtime
- there was no zipped transport artifact path

The desktop packaging target for this pass is the web runtime, not the older tkinter shell.

Why:

- it already carries the truthful review/report/apply state
- it uses the stabilized runtime-health contract
- it is the operator-facing path already validated in recent passes

## New packaged app path

New entrypoint:

- `src/r3dmatch/web_launcher.py`

Build script:

- `scripts/build_macos_app.sh`

Transport packaging:

- `scripts/package_macos_app.sh`

Smoke check:

- `dist/R3DMatch.app/Contents/MacOS/R3DMatch --check`

## Runtime assumptions on another Mac

Still externalized by design:

- `RED_SDK_ROOT`
- optional `RED_SDK_REDISTRIBUTABLE_DIR`

Expected for HTML→PDF export:

- Homebrew cairo / pango / glib libraries under `/opt/homebrew/lib`
- or an equivalent `DYLD_FALLBACK_LIBRARY_PATH`

The packaged launcher itself auto-runs the same runtime-health path used from source.

## Validation

Validated in this pass:

- `py_compile` on modified Python files
- focused FTPS/web tests
- full `tests/test_cli.py`
- PyInstaller build of `dist/R3DMatch.app`
- packaged smoke check:
  - without RED env: HTML/PDF ready, RED backend not ready
  - with `RED_SDK_ROOT` + `DYLD_FALLBACK_LIBRARY_PATH`: HTML/PDF ready, RED backend ready

Build artifact:

- `dist/R3DMatch.app`
- `dist/R3DMatch-macos-arm64.zip` after running `scripts/package_macos_app.sh`

## Known limitations

- the packaged app does not bundle the RED SDK; RED remains external by design
- the packaged app also still relies on the destination Mac having the WeasyPrint native libs available
- the older `desktop_app.py` path remains in the repo, but it is not the packaging target for this workflow
