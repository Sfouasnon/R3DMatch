# UI Post-Run Status Handoff

## Canonical Validation Artifact

The canonical review validation artifact is:

`<run>/report/review_validation.json`

The Web UI should not expect a root-level `review_validation.json`. A run is considered finalized from the UI's perspective when this canonical report artifact exists and reports `status = success`.

## Post-Run UI Data Sources

After a review completes, the Web UI now reads post-run state from these artifacts:

- Validation status and finalization truth:
  - `<run>/report/review_validation.json`
- Decision/recommendation payload:
  - `<run>/report/contact_sheet.json`
  - if freshness gating would reject it, the UI now falls back to the path recorded in `review_validation.required_artifacts.contact_sheet_json.path`
- Commit / per-camera writeback table:
  - `<run>/report/calibration_commit_payload.json`
  - if freshness gating would reject it, the UI now falls back to the path recorded in `review_validation.commit_payload.aggregate_path`
- Progress / completion hint:
  - `<run>/review_progress.json`

## Freshness / Caching Change

The UI still prefers current artifacts using the run `started_at` timestamp, but completed runs now have a canonical fallback:

- if `review_progress.json` reports `phase = review_complete`
- and `extra.validation_status = success`
- then the UI accepts the canonical `report/review_validation.json` even if its timestamp would otherwise look stale

Once that validated report is accepted, the UI also reloads `contact_sheet.json` and `calibration_commit_payload.json` from the validated report paths instead of leaving the decision panels empty.

This prevents false states such as:

- `Finalization failed`
- `Fresh review_validation.json was not produced for this run.`
- `retained cameras 0`
- empty per-camera commit table

when the run actually succeeded.

## What Changed

Files updated in this pass:

- `src/r3dmatch/web_app.py`
- `tests/test_cli.py`

Behavioral changes:

- completed-run validation uses `review_progress.json` as a success hint when canonical validation exists
- decision/report payloads are anchored to validated artifact paths
- commit payload loading no longer depends solely on per-file mtime once validation success is established

## Integrity

This pass changed only post-run UI state wiring.

It did **not** introduce:

- report-side remeasurement
- new sphere detection
- new sample computation
- duplicate baseline REDLine renders
