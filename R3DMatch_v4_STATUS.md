# R3DMatch v4 — Status & Handoff

**Started:** 2026-06-17
**Baseline:** v3 source copied verbatim (sha256-verified identical, all 17 modules).

---

## Where things stand

**Done**
- Full read-only review of v3 → `R3DMatch_v4_Review_and_Plan.md`.
- Architecture decisions locked (hybrid measure domain; sphere solver untouched; gray card both/auto-detected; face = array median).
- REDLine flags confirmed against the installed build (65.1.3 / R3DSDK 9.2.0): show LUT = `--creativeLut`, baked LUT = `--lut`+`--lutEdgeLength`, `--primaryDev` available for characterization.
- **v4 baseline scaffolded** — v3 copied into this folder verbatim and proven byte-identical.
- Golden-regression harness added → `tools/golden_regression.py`.

**Done — Phase 2: color pipeline engine**
- `src/r3dmatch3/colorpipeline.py` — `ColorPipeline`, `CDL`, frozen `REFERENCE_PIPELINE`.
- `redline.render_measurement_frame()` now takes `pipeline: ColorPipeline = None`, routing color args through it; default = `REFERENCE_PIPELINE`, which emits the exact v3 flags in the exact order. Verified: reference command string is byte-identical to v3; delivery pipelines extend with `--creativeLut` / `--lut` / CDL only when set.
- **Verify on the Mac:** run `golden_regression.py compare` (below) — must PASS on all three sets.

**Done — Phase 3: delivery characterization** (code; characterize on the Mac)
- `src/r3dmatch3/pipeline_profile.py` — `PipelineProfile` (luma weights by output color space, neutral landing, ref→delivery tonal map), `build_profile`, `CharSample`. Math unit-tested in sandbox.
- `src/r3dmatch3/measure_delivery.py` — delivery-domain hero measurement; reuses measure.py zone geometry + 4-step chain, only luma weights parameterized. No edit to measure.py/sphere.py.
- `tools/characterize_pipeline.py` — Mac-run: reference pass (priors off) → render+measure each camera through a delivery pipeline → write `PipelineProfile` JSON. LUT-ready (`--delivery-creative-lut`).
- **Validated on the Mac (GraySphere set):** self-check delivery==reference ⇒ mean |Δlog2| = 0.0000, neutral +0.0434/−0.0157 (matches reference sphere). Tone-map None ⇒ mean |Δlog2| = 0.2062 with neutral shift to +0.0287/−0.0106 — characterization captures both tonal remap and neutral landing. ✓

**Done — Phase 4: hybrid delivery verification** (engine; UI picker still to wire)
- `models.py` — additive delivery fields on `CameraResult` (delivery_corrected_*, delivery_*_match_pct) and `RunResult` (delivery_pipeline_name, delivery_array_match_pct, delivery_min_*, delivery_profile).
- `workflow.py` — `run_analysis`/`verify_run` take explicit `delivery_pipeline: ColorPipeline = None`. Closed loop renders each corrected frame through the delivery pipeline (gated; skipped when None/reference), measures in delivery domain. `_finalize_delivery` scores delivery match % (inter-camera agreement in the delivered look) and records a fresh PipelineProfile from the run's own renders. Reference solve/commits untouched; reference-only path is byte-identical (goldens unaffected).
- `tools/run_delivery.py` — Mac runner enforcing an EXPLICIT per-run choice: `--reference-only` XOR a delivery look. Prints reference vs delivery match %.
- **Validated on the Mac:** golden `compare` all PASS (reference path byte-identical with delivery engine present). `run_delivery` ToneMap=None ⇒ delivery match scored, array 100%, fresh-profile neutral +0.0286/−0.0105 (matches Phase 3 tool). Self-check (delivery==reference) correctly short-circuits. ✓
- Known minor: one run showed n=11 delivery samples (camera I007_D106 dropped its delivery sample); reference unaffected. `log.exception` now reports the cause if it recurs — likely transient REDLine render.

**Done — Phase 5: required per-run delivery picker (desktop app)**
- `app.py` — `prompt_delivery_look()` modal shown at the start of every run (in `_on_run`): operator must pick "Reference only (BT.709)" or browse a project `.cube` show LUT; Cancel aborts the run. No remembered default.
- Choice stored on `AppState` (`delivery_pipeline`/`delivery_label`) and threaded `VerifyWorker → verify_run(delivery_pipeline=…)` (the desktop closed loop is where corrected/delivery renders happen).
- Delivery look + delivery array match % surfaced in the Results hero ("Delivery look" tile) and appended to the report's assessment note.
- All modules compile; reference solve untouched (golden harness drives run_analysis directly — fingerprints unchanged).
- **Verify on the Mac:** launch the app, start a run → the picker must appear and block until chosen; pick a look and confirm delivery match % shows on Results + report. (web_app.py picker still TODO.)

**Done — delivery picker now exposes IPP2 transforms**
- Ingest "DELIVERY LOOK" field: "— Select —" (blocks run) / "Reference" / "Custom IPP2 transform…" / "IPP2 transform + show LUT…". Custom/LUT reveal Color space, Gamma, Tone map, Roll-off dropdowns (default to the reference look) and (for LUT) a Browse. Builds a full ColorPipeline.

**Done — WB modes incl. per-camera Kelvin (corrected approach)**
Field-correct insight (operator): tint = green/magenta only; matching warm/cool needs PER-CAMERA Kelvin. The first "neutral" mode (shared Kelvin slid to ~5010K) was wrong — it detuned WB off scene temp to cancel a *display* cast. Rebuilt:
- `solve.py solve_white_balance(wb_mode=…)`:
  - `match` (default, **byte-identical v3** — golden anchor): shared snapped Kelvin, per-camera tint to group-median GM; WC uncorrected.
  - `scene_match` (**recommended**): per-camera Kelvin trims match WC to the group median, per-camera tint matches GM; array stays anchored at the as-shot **scene Kelvin** (e.g. 5600K). Both axes matched, scene-honest. The uniform IPP2 cast is left to grade/LUT.
  - `neutral`: per-camera Kelvin+tint targeting WC=0/GM=0 (average shifts off scene temp). Special-case only.
- Threaded `run_analysis(wb_mode=…)`; Ingest "WHITE BALANCE" selector (scene_match default in UI; CLI/golden default `match`).
- V4 GrayonGray under scene_match: per-camera Kelvin 5531–5713K, avg 5608K (≈ scene 5600). Matches observed field behavior.

**Not started** (in plan order)
1. web_app.py delivery picker (parity with desktop) + per-camera delivery match column in report/push table.
2. Method abstraction (`SphereMethod` wrapper, zero behavior change) + route `workflow.py`.
3. Gray-card method. 4. Face method. 5. Validation.

---

## Layout

```
R3DMatch_v4/
  R3DMatch_v4_Review_and_Plan.md   ← the plan (read this first)
  R3DMatch_v4_STATUS.md            ← this file
  pyproject.toml, requirements.txt ← copied from v3
  probe_rw_ceiling.py              ← v3 r/w gate probe
  src/r3dmatch3/                   ← VERBATIM v3 source (package name unchanged)
  tools/
    golden_regression.py           ← reference-path identity guard
    golden/                         ← capture baselines here (_106.json, _067.json, _064.json)
  docs/v3_reference/               ← v3 ARCHITECTURE.md, INSTALL.md, Handoff (reference only)
```

### Package name note
The package is still imported as `r3dmatch3` so the baseline is *provably* identical to v3
(v2→v3 renamed the package, which rewrites every import and forces full re-validation). Renaming
to `r3dmatch4` is deliberately deferred to its own tracked step — it should happen once, after the
golden baselines exist, so the rename itself can be regression-checked.

---

## Lock the golden baselines (do this on the Mac, before writing v4 code)

REDLine + footage are required, so this runs on your machine, not in the Cowork sandbox.

```zsh
cd /Users/sfouasnon/Desktop/R3DMatch_v4
# (set up a venv from requirements.txt if not already)
python3 tools/golden_regression.py capture \
  --input /Users/sfouasnon/Desktop/Test_Run/SatJune6_840PM_GrayonGray \
  --out   /tmp/r3dmatch_v4_106 --golden tools/golden/_106.json
# repeat for _067 and _064 sets (paths in docs/v3_reference/R3DMatch_v3_Handoff.md §2)
```

Then after every v4 change, `compare` must PASS:

```zsh
python3 tools/golden_regression.py compare \
  --input /Users/sfouasnon/Desktop/Test_Run/SatJune6_840PM_GrayonGray \
  --out   /tmp/r3dmatch_v4_106 --golden tools/golden/_106.json
```

A non-zero exit = the reference path drifted. That is the tripwire protecting the proven solver.

---

## PARKED — Scene-referred color match via RCP2 CDL (blocked on RED)

Goal: carry a per-camera scene-linear color correction (a diagonal/3x3) in the
clip via **RCP2 ASC CDL** so it travels with the R3D and reproduces downstream.

Confirmed with RED engineering (Dan):
- ASC CDL is applied **pre-output-transform, in the Log3G10 working space** (per
  the IPP2 diagram: after "to Log3G10", before output tone map / transform). So a
  scene-linear gain maps to CDL **offset** (log-domain), not slope.
- RCP2 exposes CDL slope/offset/power/saturation (`RCP_PARAM_CDL_*`); there is
  **no** RGB-gain / WB-RGB handle. WB stays Kelvin/Tint.
- CDL values save to R3D metadata + a sidecar **.cfl** in each .RDC folder.
- Test trick: set CDL **saturation = 0** to confirm pass-through.

**BLOCKER:** `--useMeta` does NOT apply CDL on transcode — REDLine defaults CDL
**off** for transcodes. Today the only way to bake CDL is: enable in RCX → save
RMD → export in REDLine with **RMD enabled**. Dan filed a Jira to add a REDLine
flag to enable CDL on `--useMeta`. **We cannot ship the RCP2-CDL color workflow
until that lands.** Exposure matching (exposureAdjust via --useMeta) is unaffected
and works today.

Validation already done (safe to resume when unblocked): `tools/scene_linear_match_probe.py`
proved scene-linear diagonal matching collapses the array WB; the open work is
only the CDL-in-Log3G10 solve + the RCP2 push, gated on the Jira.
