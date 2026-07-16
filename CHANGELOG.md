# Changelog

All notable changes to R3DMatch are recorded here.

## [5.1.0] — 2026-07-16

### Added
- **18% Gray Anchor matching strategy.** A new exposure strategy that drives every
  camera to an *absolute* scene-linear reflectance instead of the array median. The
  target is entered as a Log3G10 IRE (default **33.3 IRE = 0.18 scene-linear = 18%
  gray**) and is editable on the Setup screen. The solve runs entirely in
  scene-linear, so it never touches the display/delivery transform; Log3G10 is only
  the unit the target is named in. The reported anchor is shown in display IRE
  (BT.1886) with the Log3G10 target noted alongside.
- **`--strategy` / `--gray-target-ire` flags** on `tools/golden_regression.py`, so
  both the median and gray-anchor baselines can be captured and gated.
- Fresh golden baselines for the `GraySphere_GrayBackdrop_FocusChart` set (median
  and gray-anchor), replacing the stale display-space median baseline.

### Documentation
- README: document the **Capture** tab (RCP2 synchronized single-frame record +
  FTP-over-TLS ingest) and the two matching strategies; refresh the project layout.

### Changed
- **Exposure now hard-requires a scene-linear measurement for every camera.** The
  previous silent all-or-nothing fallback to a display-space solve (which
  undershoots the stops) has been removed. A missing scene-linear render is retried
  with backoff; if it still can't be produced, the run stops with a per-clip
  diagnostic rather than degrading accuracy invisibly.
- The gray-anchor solve is scored against the measured corrected-array consensus
  (transform-exact), not a slope-1 display assumption.

### Notes
- The exposure/median path is unchanged for runs where every camera has its
  scene-linear render (verified byte-identical against the prior baseline).
- Empirical result on the reference set: an 18% gray (0.18 scene-linear) lands at
  **43.8 IRE** through IPP2/BT.1886 — consistent with the ~41–42 IRE rule of thumb.

## [5.0.0]

- v5 line: single-frame Capture tab (RCP2) + FTP ingest.

## [4.x]

- Scene-linear exposure solve with RWG luminance weights (decouples color from
  exposure); RCP2 push (Detect Cameras, Commit Calibration, Reset to Default);
  step-aware verify tolerance.
