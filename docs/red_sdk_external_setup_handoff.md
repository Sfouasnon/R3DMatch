# RED SDK External Setup Handoff

## What changed

R3DMatch no longer assumes a repo-local RED SDK under `src/RED_SDK/...`.

The RED backend now resolves SDK configuration from environment:

- `RED_SDK_ROOT` is the primary SDK root
- `RED_SDK_REDISTRIBUTABLE_DIR` can override the runtime redistributable directory
- optional `RED_SDK_INCLUDE_DIR` and `RED_SDK_LIBRARY_DIR` remain available for explicit build overrides

## Legacy assumptions removed

Removed as operational assumptions:

- repo-local `src/RED_SDK/R3DSDKv9_2_0/...`
- runtime fallback to a compiled-in `"."` redistributable path
- stale CMake cache reuse of an old SDK root during bridge rebuilds

## Files changed

- `src/r3dmatch/sdk.py`
- `src/r3dmatch/rmd.py`
- `src/r3dmatch/native/CMakeLists.txt`
- `src/r3dmatch/native/red_sdk_bridge.cpp`
- `scripts/build_red_sdk_bridge.sh`
- `README.md`
- `docs/architecture.md`
- `tests/test_cli.py`

## Runtime behavior

When the RED backend is requested:

- Python resolves the external SDK root and redistributable path first
- `RED_SDK_REDISTRIBUTABLE_DIR` is exported before native initialization
- the bridge reports its compiled SDK root/config
- a mismatched compiled SDK root now produces a clear rebuild error

## Web UI launch propagation

Web-launched review commands now explicitly propagate:

- `RED_SDK_ROOT`
- `RED_SDK_REDISTRIBUTABLE_DIR` when present

The tcsh command builder keeps:

- `PYTHONPATH`
- `R3DMATCH_INVOCATION_SOURCE`

and now adds the RED SDK env alongside them so UI-triggered subprocesses use the same external SDK contract as direct CLI runs.

If `backend=red` is selected in the web UI and `RED_SDK_ROOT` is missing from the web app environment, form validation now fails early with a clear actionable message instead of crashing during the first clip measurement.

## Build behavior

`scripts/build_red_sdk_bridge.sh` now:

- requires `RED_SDK_ROOT`
- derives include / library / redistributable paths from it
- validates those directories
- clears the old build directory so stale CMake cache values cannot survive
- passes the resolved SDK paths directly into CMake

## Validation status

- Python syntax validation completed
- focused tests for SDK resolution and RED backend pathing were added
- the bridge rebuild succeeded against:
  - `/Users/sfouasnon/Desktop/R3DSplat_Dependecies/RED_SDK/R3DSDKv9_2_0`
- the rebuilt in-package bridge now reports:
  - compiled root = external `RED_SDK_ROOT`
  - compiled redistributable dir = external `Redistributable/mac`
- runtime validation reached bridge import and `RedSdkDecoder()` initialization successfully
- a project-local `.R3D` fixture metadata inspect reached clip loading, but the fixture itself returned RED clip status `2`, so clip inspection did not complete on that sample
- invalid SDK config now fails early with an actionable message instead of crashing deep in measurement

## Operator guidance

Use:

```bash
export RED_SDK_ROOT=/Users/sfouasnon/Desktop/R3DSplat_Dependecies/RED_SDK/R3DSDKv9_2_0
scripts/build_red_sdk_bridge.sh
```

If the redistributable runtime is elsewhere:

```bash
export RED_SDK_REDISTRIBUTABLE_DIR=/path/to/Redistributable/mac
```
