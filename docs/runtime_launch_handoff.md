# Runtime Launch Handoff

## What changed

This pass finished the runtime/operator layer for the HTML-master contact-sheet workflow.

New pieces:

- `src/r3dmatch/runtime_env.py`
- `r3dmatch cli runtime-health`
- `scripts/_r3dmatch_env.sh`
- `scripts/run_r3dmatch_web.sh`
- `scripts/run_r3dmatch_review.sh`

## Single recommended launch path

```bash
/bin/sh scripts/run_r3dmatch_web.sh 5000
```

This is now the single recommended operator launch path.

It:

- uses the project runtime
- exports `PYTHONPATH`
- auto-fills `DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib` on macOS when unset and available
- preserves `RED_SDK_ROOT`
- preserves `RED_SDK_REDISTRIBUTABLE_DIR`
- runs `runtime-health --strict --require-red-backend` before opening the UI

## Single recommended full-review path

Web path:

1. launch with `scripts/run_r3dmatch_web.sh`
2. open the UI
3. scan source
4. run `Full Contact Sheet`

Direct shell wrapper:

```bash
/bin/sh scripts/run_r3dmatch_review.sh /path/to/r3d/files --out /path/to/run ...
```

## Health check command

```bash
./.venv/bin/python -m r3dmatch.cli runtime-health --strict --require-red-backend
```

Reports:

- interpreter path
- active venv
- `DYLD_FALLBACK_LIBRARY_PATH`
- RED SDK config status
- WeasyPrint import status
- HTML→PDF readiness
- optional HTML asset validation

## Runtime behavior

- Web subprocesses use `sys.executable`, not bare `python3`
- launcher/export env now carries:
  - `PYTHONPATH`
  - `R3DMATCH_INVOCATION_SOURCE`
  - `DYLD_FALLBACK_LIBRARY_PATH`
  - `RED_SDK_ROOT`
  - `RED_SDK_REDISTRIBUTABLE_DIR`
- `contact_sheet_pdf_export_preflight(...)` is stored in report payloads and used for early validation
- `validate_contact_sheet_html_assets(...)` verifies relative `<img src>` references before PDF export

## Current validation boundary

In this session:

- the launch/runtime plumbing is correct
- test coverage passes
- a live preflight still reports missing native WeasyPrint libraries on this host until the required cairo/pango/glib runtime is reachable

That remaining blocker is environmental, not architectural.

## Integrity

This pass did not change:

- calibration science
- report measurement logic
- sphere detection
- sample computation
- REDLine correction semantics
- baseline render count
