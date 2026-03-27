# RCP2 Transport

R3DMatch uses the documented RED RCP2 WebSocket JSON transport as the primary live-control path.

## Primary Transport

- protocol: WebSocket JSON
- endpoint: `ws://<camera_ip>:9998`
- default transport kind: `websocket`

The legacy raw TCP path is kept only as a debugging fallback:

- transport kind: `raw-legacy`
- port: `1112`

## Current Supported Live Operations

### Read-Only

- connect and configure RCP2 session
- read camera info
- read current:
  - exposure correction
  - Kelvin
  - tint
- compare live-read state to a payload with `verify-camera-state --live-read`

### Controlled Writeback Paths

The codebase also includes apply commands, but they are intentionally separate from analysis and review:

- `apply-calibration`
- `apply-camera-values`
- `test-rcp2-write`

These commands should only be used in operator-controlled workflows.

## Parameter Mapping

R3DMatch field to RED RCP2 parameter mapping:

- `exposureAdjust` -> `EXPOSURE_ADJUST`
- `kelvin` -> `COLOR_TEMPERATURE`
- `tint` -> `TINT`

Scaling uses camera-provided metadata such as divider, range, and step information whenever available.

## Design Notes

- the primary transport does not use legacy raw footer/session framing
- state handling is async-aware
- stale-message protection is part of the WebSocket read path
- periodic/subscription scaffolding exists but is not enabled by default
