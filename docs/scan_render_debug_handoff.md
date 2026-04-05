# Scan / Render Debug Handoff

## Root cause

The Flask 500 on `/scan` was caused by uncaught preview-generation exceptions inside:

- `src/r3dmatch/web_app.py` → `_ensure_scan_preview(...)`

The failure was not in the new large-array `report_focus` form field or the scan template.

The exact broken path was:

1. `/scan` calls `scan_sources(...)`
2. for valid local folders with a representative clip, `/scan` calls `_ensure_scan_preview(...)`
3. `_ensure_scan_preview(...)` calls:
   - `_resolve_redline_executable()`
   - `_detect_redline_capabilities(...)`
   - `_build_redline_preview_command(...)`
   - `subprocess.run(...)`
4. if REDLine cannot be launched or capability probing raises `FileNotFoundError` / similar process errors, that exception propagated out of the request handler and Flask returned HTTP 500

## Fix

`_ensure_scan_preview(...)` now catches preview-generation exceptions and degrades cleanly:

- logs a warning with traceback
- sets `scan["preview_warning"]`
- clears `scan["preview_note"]`
- returns `None`

This keeps `/scan` rendering the page even when preview extraction is unavailable.

## Validation

Validated request states:

- empty path → renders page, no 500
- invalid folder → renders page, no 500
- valid small local folder → renders page, no 500
- valid larger local folder → renders page, no 500
- FTPS source mode with valid planning inputs → renders page, no 500

Focused regression test added:

- `test_web_app_scan_page_survives_preview_generation_failure`

This test forces `_detect_redline_capabilities(...)` to raise `FileNotFoundError` and verifies `/scan` still returns HTTP 200 with a visible preview-fallback message.
