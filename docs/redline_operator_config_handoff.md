# REDLine Operator Configuration Handoff

## Root cause

The packaged app already knew how to resolve REDLine, but it did not give the operator a
real in-app way to configure that resolver. REDLine remained external, while the UI only
reported status.

The real seam was already in `report.py`:

- `REDLINE_CONFIG_PATH`
- `resolve_redline_tool_status()`
- `_resolve_redline_executable()`

The missing operator flow was:

- browse for REDLine
- save the chosen executable using the existing config contract
- re-check the real resolver
- show that resolved truth in runtime health
- persist it across relaunch

## True resolver precedence after this pass

The REDLine resolver remains single-source-of-truth in `report.py`.

Precedence order:

1. `R3DMATCH_REDLINE_EXECUTABLE`
2. `config/redline.json`
   - `redline_executable`
   - `redline_path`
3. `PATH` lookup for `REDLine`

No alternate resolver path was introduced.

## What changed

### REDLine operator control surface

Added a focused REDLine configuration block to the existing Runtime Health section:

- current REDLine status
- current resolved executable
- persisted REDLine executable field
- Browse button
- Save button

### Native browse path

Added a PySide6 native file picker for executable selection:

- `pick_existing_file(...)` in `native_dialogs.py`
- CLI bridge: `pick-file`
- web route: `/browse-file`

### Persistence

Added thin helpers in `runtime_env.py` to read/write the existing config contract:

- `read_redline_config()`
- `write_redline_config()`
- `redline_configured_path()`

These use the existing `REDLINE_CONFIG_PATH` from `report.py`.

### Save-time truthfulness

The save flow does **not** trust filesystem heuristics as final truth.

It:

1. writes the candidate path using the real config contract
2. re-runs `resolve_redline_tool_status()`
3. accepts the save only if the resolver now reports REDLine ready
4. restores the previous config if the new saved path fails

If an environment override is active, the UI says so explicitly.

### Runtime health

Runtime health now exposes the persisted configured path separately from the resolved path:

- `redline_tool.configured_path`
- `redline_tool.configured_config_path`

The actual readiness/source still comes from the real resolver.

## Files changed

- `src/r3dmatch/runtime_env.py`
- `src/r3dmatch/native_dialogs.py`
- `src/r3dmatch/cli.py`
- `src/r3dmatch/web_app.py`
- `tests/test_cli.py`

## Where the REDLine path is persisted

### Dev/source runtime

The config contract remains:

- `config/redline.json`

### Packaged app runtime

For the rebuilt `.app`, the persisted path is stored at:

- `dist/R3DMatch.app/Contents/config/redline.json`

Observed saved payload:

```json
{
  "redline_executable": "/Applications/REDCINE-X Professional/REDCINE-X PRO.app/Contents/MacOS/REDline"
}
```

## Build commands executed

```bash
cd /Users/sfouasnon/Desktop/R3DMatch
RED_SDK_ROOT=/Users/sfouasnon/Desktop/R3DSplat_Dependecies/RED_SDK/R3DSDKv9_2_0 /bin/sh scripts/build_macos_app.sh

cd /Users/sfouasnon/Desktop/R3DMatch
/bin/sh scripts/package_macos_app.sh
```

## Rebuilt artifacts

- `dist/R3DMatch.app`
- `dist/R3DMatch-macos-arm64.zip`

Rebuild evidence:

- spec regenerated: `2026-04-02 14:27:25`
- app rebuilt: `2026-04-02 14:28:26`
- zip rebuilt: `2026-04-02 14:28:53`

## Packaged app open/operability validation

### Opened rebuilt packaged app

The rebuilt packaged app was launched directly from:

- `dist/R3DMatch.app/Contents/MacOS/R3DMatch`

Observed live packaged launcher output:

- `R3DMatch desktop launcher: http://127.0.0.1:5003`
- later relaunch:
  - `R3DMatch desktop launcher: http://127.0.0.1:5001`

This satisfies the “opened after rebuild” requirement using the packaged artifact itself.

### Live packaged validation performed

Validated against the opened packaged app:

- GET `/`
  - Runtime Health rendered
  - `RED SDK Runtime` rendered
  - `REDLine Tool` rendered
  - `REDLine Executable Path` field rendered
  - `Save REDLine Path` rendered
- POST `/browse-file`
  - returned the scripted REDLine executable path
- POST `/configure-redline`
  - saved REDLine successfully
- GET `/` after save
  - `Source: config`
  - `Executable: /Applications/REDCINE-X Professional/REDCINE-X PRO.app/Contents/MacOS/REDline`
  - persisted field value present
- POST `/scan`
  - still rendered successfully
- relaunch validation
  - saved REDLine path persisted across relaunch
  - runtime health still showed `source = config`
- POST `/browse-folder`
  - existing folder browse control still worked in the rebuilt packaged app

### Packaged runtime-health after save/relaunch

Observed:

- `red_sdk_runtime.ready = true`
- `red_sdk_runtime.source = bundled_app`
- `redline_tool.ready = true`
- `redline_tool.source = config`
- `redline_tool.config_path = /Users/sfouasnon/Desktop/R3DMatch/dist/R3DMatch.app/Contents/config/redline.json`

## Tests

- focused REDLine/runtime tests: `12 passed`
- full CLI suite: `244 passed`

## CLAUDE REWRITE EVALUATION

### PySide6 file picker for REDLine

- `KEEP`
- This aligned with the existing PySide6 browse direction and avoided tkinter.
- Implemented as `pick_existing_file(...)` plus `pick-file` and `/browse-file`.

### Writing `config/redline.json` using `redline_executable`

- `KEEP`
- This matches the real resolver contract in `report.py`.
- Implemented through thin helpers in `runtime_env.py`.

### Exposing `configured_path` in runtime health

- `KEEP`
- Helpful for pre-filling the UI field without changing resolver truth.
- Implemented as annotation only; readiness still comes from the real resolver.

### New CLI command `pick-file`

- `KEEP`
- Useful and small.
- Implemented and tested.

### Save-time validation using `Path.exists` / `os.access`

- `MODIFY`
- Pure filesystem checks are not the resolver truth.
- Replaced with “write candidate config → re-run `resolve_redline_tool_status()` → rollback on failure”.

### UI partial updates after save

- `MODIFY`
- Avoided fragile partial synchronization.
- Used a reliable full page rerender so runtime health and the saved field always agree.

### Any env-based override injection

- `REJECT`
- Would create fake state and break precedence truthfulness.
- Not implemented.

### Any alternate config-path logic

- `REJECT`
- The project already had a canonical config contract.
- Not implemented.

## Remaining limitations

- REDLine is still external and is not bundled in this pass.
- Packaged runtime currently resolves REDLine from:
  - saved config path
  - or environment
  - or `PATH`
- The RED SDK redistributable runtime is bundled and ready, but REDLine itself must still exist on the destination Mac.
