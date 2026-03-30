# R3DMatch Operational Handoff

## Final System Shape

R3DMatch is now treated as a repeatable multi-camera measurement and correction system for RED arrays.

The active production path is:

- sphere detection on the rendered review frame
- gradient-axis aligned 3-band center-sphere measurement
- weighted anchor-relative exposure solve
- direct REDLine preview and iteration controls
- authoritative IPP2-domain validation

Acceptance truth remains:

- REDLine IPP2
- BT.709
- BT.1886
- Medium / Medium

`ipp2_validation.json` is the authoritative acceptance artifact.

## Measurement Model

The gray sphere is measured as a 3-band center-sphere profile, not a flat card.

Band semantics:

- `bright_side`
- `center`
- `dark_side`

Band orientation:

- aligned to the sphere's dominant luminance gradient axis
- not fixed to image vertical
- robust to camera rotation and framing orientation

Per-band statistics:

- sphere detection provides center and radius
- a refined interior sphere mask rejects edge contamination
- conservative interior bands are placed inside the sphere body
- each band uses robust many-pixel statistics

Weights:

- center `0.5`
- bright side `0.3`
- dark side `0.2`

The center band is the primary exposure anchor.

## REDLine Preview Workflow

The canonical preview and iteration path now uses direct REDLine parameters:

- `--useMeta`
- `--exposureAdjust`
- `--kelvin`
- `--tint`

This same parameter model is used for:

- preview generation
- closed-loop review iterations
- authoritative IPP2 validation renders

This removes the earlier semantic mismatch between sidecar-driven preview renders and direct-control validation behavior.

### `--exposure` vs `--exposureAdjust`

On the tested REDLine build, `--exposure` and `--exposureAdjust` behaved identically for the measured preview path.

The important semantic distinction was whether the command used `--useMeta`.

R3DMatch standardizes on `--exposureAdjust` because:

- it matches camera-side parameter naming
- it behaves correctly on this REDLine build
- it aligns preview, solve, validation, and live writeback language

## Preview Mode Policy

Production preview mode is now `monitoring`.

Legacy `calibration` preview mode is retained only as a compatibility alias and resolves to the canonical monitoring transform.

That means:

- no production path should silently render REDWideGamutRGB / Log3G10 operator previews
- no production path should diverge from the authoritative BT.709 / BT.1886 monitoring transform

## Kelvin / Tint Propagation

All preview-command call sites must propagate Kelvin and Tint explicitly.

Defensive defaults are locked as:

- Kelvin: `5600`
- Tint: `0`

The web UI scan-preview path now always provides explicit white-balance values, using:

- form value if present
- otherwise scan state value if present
- otherwise the default

This prevents preview crashes and prevents silent omission of WB controls.

## Detection Policy

Sphere detection remains auditable and recovery-oriented.

The system:

- tries primary detection first
- validates each candidate
- reuses original-frame ROI for corrected-frame measurement
- records detection confidence and source
- exports overlays for inspection

It does not silently fabricate a center sphere.

## Operational Assumptions

The production acquisition protocol assumes:

- a standardized gray sphere
- stable framing intent
- same IPP2 monitoring transform on set
- consistent camera correction semantics

This is a strength, not a weakness. The system is intentionally optimized for repeatable array calibration, not arbitrary image evaluation.

## Non-Blocking Limitations

- RMD sidecars are still exported for downstream interoperability, but they are not the active preview truth path.
- Contact sheets are review artifacts, not acceptance truth. Final pass/fail still comes from `ipp2_validation.json`.
- Color preview remains intentionally conservative in operator-facing review until fully proven.

## Fresh Validation Reference

Fresh source-derived validation artifacts from the finalized direct-monitoring run live at:

- `/Users/sfouasnon/Desktop/R3DMatch/runs/final_direct_monitoring_063/real_063_final_direct_monitoring/report/ipp2_validation.json`
- `/Users/sfouasnon/Desktop/R3DMatch/runs/final_direct_monitoring_063/real_063_final_direct_monitoring/report/contact_sheet.html`
- `/Users/sfouasnon/Desktop/R3DMatch/runs/final_direct_monitoring_063/real_063_final_direct_monitoring/report/preview_contact_sheet.pdf`
- `/Users/sfouasnon/Desktop/R3DMatch/runs/final_direct_monitoring_063/real_063_final_direct_monitoring/report/review_detection_summary.json`

Final authoritative result on the real 063 set:

- `PASS 12 / REVIEW 0 / FAIL 0`
- best residual `0.0038`
- median residual `0.0217`
- worst residual `0.0462`
- preview transform matches validation transform
