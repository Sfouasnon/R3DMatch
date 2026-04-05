## Packaged Runtime Hardening Handoff

### Scope

This pass hardened the packaged macOS app runtime without changing calibration science, WB solve behavior, FTPS workflow behavior, or report semantics.

### Packaged vs Dev Runtime Findings

- Packaged subprocesses cannot safely use `sys.executable -m r3dmatch.cli ...` because `sys.executable` points at the bundled app binary, not a Python interpreter.
- Finder-style launches do not inherit shell environment variables such as `RED_SDK_ROOT`.
- Frozen subprocesses should not assume a source checkout working directory or set `PYTHONPATH="$PWD/src"`.
- Optional preview probing must remain soft-fail during `/scan`; packaged launches must not regress that behavior.
- Local localhost route validation required running outside the desktop sandbox because socket bind and localhost curl are restricted there. The packaged app itself launched and served correctly once validated outside the sandbox restriction.

### Code Changes

- `src/r3dmatch/web_app.py`
  - Added `_runtime_cli_prefix()` so packaged subprocesses invoke the embedded CLI as:
    - `R3DMatch --cli ...`
  - Updated review / approve / clear-cache / FTPS ingest command builders to use the embedded CLI prefix when frozen.
  - Updated `_build_tcsh_launch_prefix()` so frozen runtime:
    - `cd`s to the bundled executable directory
    - does not inject `PYTHONPATH`
  - Kept preview generation failures soft in `_ensure_scan_preview(...)`.
  - Added clearer packaged runtime-health messaging for missing `RED_SDK_ROOT`.

- `src/r3dmatch/web_launcher.py`
  - Added `--cli` dispatch so the packaged binary can execute CLI subcommands directly inside the bundle runtime.

- `src/r3dmatch/runtime_env.py`
  - Added `frozen_app` to runtime-health payloads for UI/runtime diagnostics.

- `tests/test_cli.py`
  - Added packaged embedded-CLI coverage.
  - Added frozen `tcsh` launch-prefix coverage.
  - Preserved scan preview soft-failure coverage.

### Rebuild Evidence

Final rebuild/package timestamps after the runtime fixes:

- `dist/R3DMatch.app` -> `2026-04-02 10:43:57`
- `dist/R3DMatch-macos-arm64.zip` -> `2026-04-02 10:44:53`
- `build/macos_app/spec/R3DMatch.spec` -> `2026-04-02 10:43:10`

### Packaged Validation Performed

#### Packaged launcher checks

- Without RED env:
  - `dist/R3DMatch.app/Contents/MacOS/R3DMatch --check`
  - Result:
    - `html_pdf_ready=True`
    - `red_backend_ready=False`

- With RED env:
  - `RED_SDK_ROOT=/Users/sfouasnon/Desktop/R3DSplat_Dependecies/RED_SDK/R3DSDKv9_2_0 DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib dist/R3DMatch.app/Contents/MacOS/R3DMatch --check`
  - Result:
    - `html_pdf_ready=True`
    - `red_backend_ready=True`

#### Live packaged HTTP validation

Validated against the rebuilt bundled executable:

- `GET /` -> `200`
- `POST /scan` -> `200`
- `POST /run-review` -> `200`

Notes:

- The `/run-review` smoke request used a placeholder fake `.R3D`, so the background task failed later during real REDLine render, which is expected for a non-real clip.
- The important packaged-runtime result is that the route returned `200`, the app stayed alive, and the subprocess command used:
  - `.../R3DMatch --cli review-calibration ...`
  rather than `python3 -m r3dmatch.cli ...`

### Remaining External Requirements

- `RED_SDK_ROOT` is still required for RED backend runs.
- `DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib` is still required when native WeasyPrint / Cairo / Pango libraries are not already discoverable.
- Finder launches remain environment-limited by macOS; the UI now calls this out explicitly.
