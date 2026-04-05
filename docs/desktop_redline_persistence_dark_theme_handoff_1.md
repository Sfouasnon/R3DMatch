## Desktop REDLine Persistence + Dark Theme Handoff 1

Date: 2026-04-02

### Scope

Focused desktop-shell pass only:
- persist REDLine executable path across relaunch
- refresh runtime health automatically after save/startup
- apply a fast dark-theme pass without redesigning workflow structure

No calibration, report-science, or RED SDK backend logic was changed.

### REDLine Persistence

Added a desktop user config file:

- `~/.r3dmatch_config.json`

Stored shape:

```json
{
  "redline_path": "/path/to/REDline"
}
```

Implementation lives in:
- `src/r3dmatch/runtime_env.py`

Added helpers:
- `desktop_config_path()`
- `load_config()`
- `save_config()`

Behavior:
- Browse/select in the desktop UI writes `redline_path`
- desktop startup loads the saved path
- startup hydrates:
  - `REDLINE_PATH`
  - `R3DMATCH_REDLINE_EXECUTABLE`
- runtime health is then evaluated through the existing real REDLine resolver

Compatibility:
- the existing project `config/redline.json` contract remains in place
- desktop persistence now mirrors to that project config when possible
- desktop runtime truth still comes from the real resolver in `report.py`

### Desktop Runtime Improvements

`src/r3dmatch/runtime_env.py`
- added desktop config loading / saving
- added persisted REDLine env hydration on startup
- added light self-heal:
  - if no explicit path is configured, common macOS REDLine install locations are checked
- runtime health now exposes:
  - desktop config path
  - desktop config source

`src/r3dmatch/desktop_app.py`
- REDLine save now rehydrates runtime health immediately
- REDLine reload repopulates the saved path
- browse feedback now makes it clear that save persists for next launch
- runtime panel now shows the desktop config location

### Dark Theme

Applied a fast dark-theme pass in `src/r3dmatch/desktop_app.py`:
- darker Fusion palette
- global dark stylesheet
- toned semantic chips:
  - good: `#2e7d32`
  - warn: `#b26a00`
  - bad: `#b00020`
- darker inputs, tabs, cards, and status bar

This was intentionally a restrained readability pass, not a structural redesign.

### Tests Added / Updated

`tests/test_cli.py`

Added / updated:
- desktop REDLine path persists across relaunch
- runtime health can resolve from persisted desktop config
- web REDLine config route also mirrors desktop config safely in tests

### Validation

#### Source / offscreen validation

- `py_compile` passed
- focused REDLine/desktop regressions passed: `6 passed`

The relaunch persistence path was validated in an offscreen desktop test by:
1. opening the desktop window
2. saving a real executable path
3. closing the window
4. reopening the window
5. confirming the path and ready state were still present

#### Packaged validation

Rebuilt and repackaged:
- `dist/R3DMatch.app`
- `dist/R3DMatch-macos-arm64.zip`

Timestamp evidence after this pass:
- `dist/R3DMatch.app` -> 2026-04-02 20:06:10
- `dist/R3DMatch-macos-arm64.zip` -> 2026-04-02 20:09:47

Opened rebuilt packaged app:

`QT_QPA_PLATFORM=offscreen R3DMATCH_USER_CONFIG_PATH=/tmp/r3dmatch_redline_ui_config.json dist/R3DMatch.app/Contents/MacOS/R3DMatch --desktop-smoke --desktop-smoke-ms 1000`

Observed:
- `desktop_window_title=R3DMatch`
- `desktop_minimal_mode=False`

Packaged runtime check with saved REDLine config:

`DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib R3DMATCH_USER_CONFIG_PATH=/tmp/r3dmatch_redline_ui_config.json dist/R3DMatch.app/Contents/MacOS/R3DMatch --check`

Result:
- `html_pdf_ready=True`
- `red_sdk_runtime_ready=True`
- `redline_ready=True`
- `red_backend_ready=True`

### Remaining Limitations

- REDLine remains external; the saved path only removes repeated manual entry
- full on-screen interaction with the rebuilt packaged UI was not automatable in this sandbox, so packaged validation used:
  - offscreen packaged desktop open
  - packaged runtime readiness check
- this pass did not rerun a full packaged 063 review because the requested scope was the focused REDLine persistence + dark-theme fix
