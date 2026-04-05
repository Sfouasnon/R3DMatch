# Packaged RED Runtime + Native Browse Handoff

## What changed

This pass made the packaged macOS app usable for real operator testing without requiring
`RED_SDK_ROOT` for ordinary packaged RED runtime readiness.

Implemented:

- bundled RED redistributable runtime inside the `.app`
- separated runtime-health reporting for:
  - RED SDK runtime
  - REDLine tool
- added native folder browsing through PySide6
- rebuilt and repackaged the `.app` after the fixes

## Bundled RED redistributable files

The build now copies only redistributable RED runtime dylibs into the app bundle:

- `REDDecoder.dylib`
- `REDMetal.dylib`
- `REDOpenCL.dylib`
- `REDR3D.dylib`

Bundled locations observed in the rebuilt app:

- `Contents/Resources/red_runtime/redistributable`
- runtime resolved location:
  - `Contents/Frameworks/red_runtime/redistributable`

The packaged runtime now prefers the bundled redistributable directory automatically.

## RED runtime vs REDLine

The app now reports these separately.

### RED SDK Runtime

Reported from `runtime_health_payload()["red_sdk_runtime"]`.

Fields include:

- `ready`
- `error`
- `redistributable_dir`
- `source`
- `root`
- `bundled_dir`

For the rebuilt packaged app, this now reports:

- `ready = true`
- `source = bundled_app`
- no `RED_SDK_ROOT` required for ordinary packaged runtime readiness

### REDLine Tool

Reported from `runtime_health_payload()["redline_tool"]`.

Fields include:

- `ready`
- `configured`
- `resolved_path`
- `source`
- `config_path`
- `error`

REDLine is still external and is **not** bundled in this pass.

Observed packaged runtime resolution:

- `/Applications/REDCINE-X Professional/REDCINE-X PRO.app/Contents/MacOS/REDline`

## Native folder browsing

Folder browsing is now handled through PySide6, not tkinter.

Implementation:

- `src/r3dmatch/native_dialogs.py`
- `src/r3dmatch/cli.py` (`pick-folder`)
- `src/r3dmatch/web_app.py` (`/browse-folder`)

Browse buttons are available for:

- Calibration Folder Path
- Output Folder Path
- Local Ingest Cache Root
- After-Apply Review Folder

Behavior:

- selection populates the field
- cancel leaves the field unchanged
- picker failures fail soft and return a clear error message instead of crashing the page

## Build commands used

Rebuild:

```bash
cd /Users/sfouasnon/Desktop/R3DMatch
RED_SDK_ROOT=/Users/sfouasnon/Desktop/R3DSplat_Dependecies/RED_SDK/R3DSDKv9_2_0 /bin/sh scripts/build_macos_app.sh
```

Repackage:

```bash
cd /Users/sfouasnon/Desktop/R3DMatch
/bin/sh scripts/package_macos_app.sh
```

## Rebuilt artifact paths

- `dist/R3DMatch.app`
- `dist/R3DMatch-macos-arm64.zip`

Final rebuild evidence:

- spec regenerated: `2026-04-02 11:45:50`
- app rebuilt: `2026-04-02 11:46:59`
- zip rebuilt: `2026-04-02 11:47:33`

## Validation performed

### Python / tests

- `py_compile` passed for modified runtime, web, SDK, CLI, and test files
- focused runtime/browse tests: `8 passed`
- full CLI suite: `238 passed`

### Packaged app direct checks

Direct packaged launcher:

```bash
dist/R3DMatch.app/Contents/MacOS/R3DMatch --check
```

Observed:

- `html_pdf_ready=True`
- `red_sdk_runtime_ready=True`
- `redline_ready=True`
- `red_backend_ready=True`

Direct packaged runtime-health:

```bash
dist/R3DMatch.app/Contents/MacOS/R3DMatch --cli runtime-health
```

Observed:

- `red_sdk_runtime.ready = true`
- `red_sdk_runtime.source = bundled_app`
- `red_sdk_root = ""`
- `redline_tool.ready = true`

Direct packaged folder-picker smoke:

```bash
R3DMATCH_PICK_FOLDER_RESPONSE=/tmp/r3dmatch_pkg_smoke_input dist/R3DMatch.app/Contents/MacOS/R3DMatch --cli pick-folder --title "Select Calibration Folder" --directory /tmp
R3DMATCH_PICK_FOLDER_CANCEL=1 dist/R3DMatch.app/Contents/MacOS/R3DMatch --cli pick-folder --title "Select Calibration Folder" --directory /tmp
```

Observed:

- scripted selection returned `/tmp/r3dmatch_pkg_smoke_input`
- scripted cancel returned an empty selection without crashing

## Remaining external requirements

- REDLine remains external and must still be installed/discoverable
- WeasyPrint native libraries still need to be reachable on the destination Mac
  - typically via `/opt/homebrew/lib`
- Finder-style launch environment still may not inherit custom shell env, but packaged RED SDK runtime no longer depends on `RED_SDK_ROOT`

## Known limitation during this session

Direct packaged web-server bind smoke was blocked by this sandbox's local port-binding behavior,
so live browser-route validation against the rebuilt packaged binary could not be completed here.

The packaged CLI/runtime checks above succeeded, and the underlying Flask/web route behavior
remains covered by the automated test suite.
