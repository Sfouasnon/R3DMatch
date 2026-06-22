# R3DMatch v2 — Living Handoff Document
**Last updated:** 2026-06-08  
**Session:** 15  
**Machine:** sfouasnon, macOS Apple Silicon  
**Shell:** zsh (NOT tcsh — always run `zsh` first)  
**For:** Next Claude session (Session 16 / V3 start)

---

## 1. WHAT IS R3DMatch v2

Multi-camera exposure and color calibration tool for RED KOMODO-X arrays. Places a physical 18% gray sphere in frame, renders one TIFF per camera via REDLine in IPP2/BT.709, detects the sphere automatically, measures luminance in IRE, and pushes per-camera `exposureAdjust` / `kelvin` / `tint` commits live to cameras via RCP2 WebSocket before recording.

**Launch V2 from source:**
```zsh
source /Users/sfouasnon/Desktop/R3DMatch_v2/.venv/bin/activate
python3 /Users/sfouasnon/Desktop/R3DMatch_v2/r3dmatch_v2/src/r3dmatch2/app.py
```

**Launch V2 bundle (stable — do not modify):**
```zsh
"/Users/sfouasnon/Desktop/R3DMatch_v2/dist/R3DMatch v2.app/Contents/MacOS/R3DMatch v2"
```

**Launch V3 from source (development):**
```zsh
source /Users/sfouasnon/Desktop/R3DMatch_v3/R3DMatch_v3/.venv/bin/activate
python3 /Users/sfouasnon/Desktop/R3DMatch_v3/R3DMatch_v3/r3dmatch_v3/src/r3dmatch3/app.py
```

---

## 2. INSTALL LOCATIONS

```
/Users/sfouasnon/Desktop/R3DMatch_v2/   ← STABLE — do not touch
  dist/R3DMatch v2.app                   ← tagged v2.0-stable in git
  git tag: v2.0-stable

/Users/sfouasnon/Desktop/R3DMatch_v3/R3DMatch_v3/   ← V3 DEVELOPMENT (note nested dir)
  r3dmatch_v3/src/r3dmatch3/             ← ALL SOURCE FILES (package renamed r3dmatch3)
  .venv/                                 ← Python 3.9.6
  r3dmatch3.spec                         ← PyInstaller spec
  dist/R3DMatch v2.app                   ← STALE v2 bundle copied from V2 — rebuild produces "R3DMatch v3.app"
```

**Test footage — TIFFs live in Test_Run, not Test_Footage:**
```
/Users/sfouasnon/Desktop/Test_Run/*/previews/_measurement/*.tif
```
- `_067` F5.6:    `R3DMatchV2_June3_9AM` / `R3DMatchV2_June12026_Run5`
- `_106` GrayGray: `SatJune6_840PM_GrayonGray`
- `_064` UnderExp: `R3DMatchV2_064_053126`

**GitHub:** https://github.com/Sfouasnon/R3DMatchV2

---

## 3. SOURCE FILES — CURRENT STATE (V3 starting point)

| File | Session | Status | Key changes |
|------|---------|--------|-------------|
| `sphere.py` | 15 | ✓ stable | `photo_prior` added to inner `_detect_sphere_by_path` signature; `_detect_sphere_orig` **kwargs NameError fixed; unified wrapper pops `photo_prior` from kwargs before forwarding |
| `workflow.py` | 15 | ✓ stable | `NEEDS_ASSIST` detection now passes through to measurement (was incorrectly falling into FAIL branch) |
| `report.py` | 15 | ✓ stable | Array comparison page cameras sorted alphabetically by `camera_label` |
| `brdf_gate.py` | 13 | ✓ stable | abs(Pearson) trimmed mean — shadow-side spokes score correctly |
| `sphere_profile.py` | 14 | ✓ stable | v2 schema: ring_lum + brdf_score stored; get_camera_prior returns photometric derived stats |
| `app.py` | 13 | ✓ stable | Settings tab; disable_priors checkbox; Reset to Defaults button |
| `rcp2.py` | 13 | ✓ stable | Fail-closed divider; reset_camera/reset_all_cameras |
| `settings.py` | 13 | ✓ stable | REDLine path + default output dir persistence |
| `redline.py` | 13 | ✓ stable | settings_path param; timeout; error handling |
| `measure.py` | 11 | ✓ stable | Inner mask 0.5r |
| `models.py` | 11 | ✓ stable | best_candidate_roi field |
| `solve.py` | stable | ✓ stable | |

---

## 4. REBUILD COMMAND (V3)

```zsh
source /Users/sfouasnon/Desktop/R3DMatch_v3/R3DMatch_v3/.venv/bin/activate
cd /Users/sfouasnon/Desktop/R3DMatch_v3/R3DMatch_v3
pyinstaller r3dmatch3.spec --clean -y 2>&1 | tail -5
```

**If PyInstaller binary cache error:**
```zsh
rm -rf "/Users/sfouasnon/Library/Application Support/pyinstaller/bincache00py3964bit"
pyinstaller r3dmatch3.spec --clean -y 2>&1 | tail -5
```

**Pre-flight checks before rebuild:**
```zsh
grep -c "photo_prior" /Users/sfouasnon/Desktop/R3DMatch_v3/R3DMatch_v3/r3dmatch_v3/src/r3dmatch3/sphere.py
# Should return 5+

grep -c "NEEDS_ASSIST" /Users/sfouasnon/Desktop/R3DMatch_v3/R3DMatch_v3/r3dmatch_v3/src/r3dmatch3/workflow.py
# Should return 1+

grep -c "camera_label" /Users/sfouasnon/Desktop/R3DMatch_v3/R3DMatch_v3/r3dmatch_v3/src/r3dmatch3/report.py | head -1
# Should return 10+
```

---

## 5. SESSION 15 — WHAT WAS DONE

### r/w ceiling investigation (handoff bug report was wrong)

Session 14 handoff claimed `_RADIUS_MAX_RATIO = 0.15` was rejecting real spheres (observed max 0.256). Probe script run against all 36 TIFFs disproved this:

- True sphere r/w across all 36 clips at detection scale: **0.032–0.058**
- Ceiling of 0.15 has ~2.5× headroom — not blocking anything
- The 0.256 figure in `sphere_profile_v2.json` came from `radius_ratio` stored as `r / min(frame_h, frame_w)` in the profile, vs. `r / w` used in detection — different denominators, same physical sphere
- **No change made to `_RADIUS_MAX_RATIO`**

### sphere.py — photo_prior NameError fix

`photo_prior` was referenced inside the pre-filter loop (line 424) but was not in the `_detect_sphere_by_path` signature. The unified wrapper was calling `_detect_sphere_by_path` with `photo_prior=` as a kwarg, producing `TypeError: unexpected keyword argument`. Fix:
- Added `photo_prior: Optional[dict] = None` to the inner signature
- Wrapper now pops `photo_prior` from `**kwargs` before forwarding, then passes explicitly
- Fixed stray `**kwargs` NameError in `_detect_sphere_orig` (called `detect_sphere` instead of `_detect_sphere_by_path`)

### workflow.py — NEEDS_ASSIST passthrough fix

`detection.success` returns `False` for `NEEDS_ASSIST` status (status is not `SUCCESS`). The failure branch at line 485 was catching `NEEDS_ASSIST` detections and marking them `FAIL` before measurement ran. This caused:
- Report showing `FAIL` instead of `NEEDS ASSIST`
- No render in assessment (measurement never ran → `cr.measurement` was None → `render_path` missing)

Fix: `if not detection.success and detection.status != "NEEDS_ASSIST":` — lets NEEDS_ASSIST detections continue to measurement, commit building, and corrected render.

### report.py — Array comparison alphabetical sort

`_render_array_comparison` received cameras pre-sorted by status/adj. Now sorts by `camera_label` at entry so the array chart reads G007_A → G007_B → ... → I007_D top-to-bottom.

### Validation results Session 15

- `_106` 12-clip workflow: **12/12** — all renders present, correct NEEDS_ASSIST/PASS statuses, array page alphabetical

---

## 5b. SESSION 16 — MATCH PERCENTAGE MODEL (THERE IS NO FAIL)

Design decision (operator: Stephen): this is an on-set tool — binary PASS/FAIL
is useless under time pressure. Replaced with **noise-floor anchored match %**:

- 100% = residual within measurement noise (≤0.05 stop exposure, ≤0.001 GM).
  Linear falloff to 0% at 0.5 stop / 0.02 GM. `solve.exposure_match_pct` /
  `wb_match_pct` / `camera_match_pct` (min of verified axes).
- **WB closed loop is now measured**: WC/GM re-measured from the corrected
  renders (same renders as exposure closed loop). The old solve-time
  `gm_spread_after` was the model inverted against itself — could never fail.
- Camera statuses: `SOLVED` / `NEEDS_ASSIST` / `NO_DATA` / `ERROR`. No FAIL,
  no REVIEW anywhere (workflow, report, app UI, web UI, JSON outputs).
- **Push is never gated.** `rcp2_push_eligible` is always true for solved
  cameras; match % is shown, operator decides.
- Run assessment: `array_match_pct` (mean) + `min_match_pct`/`min_match_clip_id`
  (worst camera). RunResult fields trusted/caution/untrusted_count REMOVED →
  solved_count/scored_count/needs_assist_count/no_data_count.
- `tests/test_solve.py`: 23 pytest cases (run `python3 -m pytest tests/`).
- Fixed pre-existing v2 leftover: `web_app.py` imported nonexistent
  `render_contact_sheet` → now uses `report.build_report`.

NOT yet validated against real footage — run `_106` set before trusting.

### Closed-loop wiring fix (late Session 16 — found on first real run)

First real run (WedJune10_1030_GrayonGray) produced an UNVERIFIED report:
the desktop app calls run_analysis(render_corrected=False) — intentional,
QC can still change ROIs — but nothing ran the closed loop after QC.
Fix: `workflow.verify_run(run_result, redline_path)` (Phase 5 extracted to
`_closed_loop_phase`, shared). App's "Create Assessment" now runs a
VerifyWorker: corrected TIFFs → measure → match % → re-assess → THEN
build_report. On verify error it falls back to the unverified report
(never blocks the operator). Tests: tests/test_verify_run.py (REDLine mocked).

### UI / Report rebuild (late Session 16)

- Design spec: `docs/ui_mockup_v3.html` (interactive, click rail to walk screens)
- app.py: warm graphite + crimson theme constants; topbar (REDLine + IP pills,
  project pill); left workflow rail (1 Ingest / 2 Analyze / 3 Assist / 4 Match
  / 5 Push + System) with R3D boxed logo; Progress has Verify stage mapped to
  corrected_renders/wb_closed_loop/assessing; Results = Match hero (30px %,
  worst-camera line); Push table has Match column (status now col 6)
- report.py: paired light document, crimson accent, R3D mark in headers;
  WB chart re-tooled as static SVG (old canvas/JS was broken — doubled-brace
  JS syntax error, rendered blank); plots measured before→after convergence
- Sample render: `docs/report_sample_paired.html` (synthetic, /tmp/sample_gen.py pattern)
- ⚠ Qt UI NOT visually verified (sandbox has no GL libs) — pyflakes-clean and
  syntax-checked only. First launch on the Mac: check rail spacing, topbar
  pills, Match hero, Push table columns.

---

## 6. KNOWN ISSUES / PENDING WORK

### Active issues

**1. Sphere profile v2 — N=1 for all cameras**  
Photo narrowing won't activate until 3 runs accumulate per camera. Run the pipeline 2 more times on `_067` and `_106` to get there. `build_sphere_profile.py` can be re-run at any time to update `sphere_profile_v2.json`.

**2. Sphere profile v2 — Desktop JSON not integrated into live store**  
`sphere_profile_v2.json` on the Desktop is standalone reference data built by `build_sphere_profile.py`. NOT the same as `~/Library/Application Support/R3DMatch_v2/profiles/`. The v2 schema in `sphere_profile.py` populates the live profile through normal pipeline runs going forward.

**3. Determinism — not confirmed post-Session 15**  
Run same footage set twice and compare with `r3dmatch_determinism_check.py`.

**4. RCP2 hardware test pending**  
Fail-closed divider and Reset to Defaults untested on live KOMODO-X hardware. Target: week of June 16.

**5. `r3dmatch2.spec` has stale skimage hidden imports**  
Non-blocking but adds bundle size. Low priority cleanup.

### Priority queue

**HIGH:**
- Determinism check post-Session 15 changes
- RCP2 hardware test (week of June 16)
- Accumulate 3+ photometric samples per camera to activate narrowing (run pipeline 2 more times)

**MEDIUM:**
- V3 feature development (see §7)
- Persistent push audit log
- `r/w` floor tightening: `_PF_RADIUS_MIN = 0.018` could be raised to ~0.025 — observed minimum is 0.032. Conservative, leaves margin. Probe first.

**LOW:**
- Remove stale skimage hidden imports from `r3dmatch2.spec`
- Camera Network: auto-populate labels from run_result

---

## 7. V3 DEVELOPMENT TARGETS

These are the planned improvements for V3. None should be started until V3 directory is confirmed working (run one full pipeline pass first).

**Detection:**
- LoG supplement tuning — currently only fires when Hough saturates (>200 peaks). Consider lowering threshold for cluttered scenes that don't saturate but still have noise.
- `_PF_RADIUS_MIN` floor tightening from 0.018 → ~0.025 (probe first)
- Photometric narrowing activation — needs 3 runs per camera

**Profile system:**
- Profile migration tool: convert live `sphere_profile_v1` → `sphere_profile_v2` for existing project profiles
- Profile viewer in Settings tab — show per-camera sample count, geometry priors, photo narrowing status

**RCP2 / Push:**
- Push audit log — persistent per-run record of what was committed to which camera
- Hardware test on live KOMODO-X array (week of June 16)

**Report:**
- Per-camera page: show sphere profile sample count and prior confidence
- Export: CSV of all camera commits for post-production metadata ingestion

---

## 8. VALIDATION RESULTS (CUMULATIVE)

| Set | Session | Result | Notes |
|-----|---------|--------|-------|
| `_106` workflow | 9 | 12/12 | Target 60.8 IRE, WB spread 0.0398 |
| `_067` workflow | 11 | 12/12 | WB spread 0.0136 |
| `_064` workflow | 12 | 12/12 | HOUGH_GRADIENT_ALT |
| Mixed 36-clip stress | 14 | 34/36 | Cold, no priors. G007_D064 NEEDS_ASSIST (8.6 IRE, correct). H007_D064 PASS after ring fix |
| `_106` workflow | 15 | 12/12 | NEEDS_ASSIST/PASS statuses correct, renders present, array alphabetical |

---

## 9. MODELS.PY FIELD MAP

```python
cr.measurement.hero_ire            # USE THIS
cr.detection.roi.r                 # (.r NOT .radius)
cr.detection.roi.cx / .cy
cr.detection.source                # "auto_hough"|"prior_assisted"|"manual_operator"
cr.detection.status                # "SUCCESS"|"FAILED"|"MANUAL"|"NEEDS_ASSIST"
cr.metadata.clip_id                # USE THIS (NOT cr.metadata.clip_name)
cr.commit.exposure_adjust / .kelvin / .tint
result.cameras                     # List[CameraResult] (NOT result.clip_results)
result.wb_solve.wc_spread_after
result.anchor_ire
result.created_at                  # ISO-8601 — used for report filename
```

---

## 10. DETECTION PARAMETERS (DO NOT CHANGE WITHOUT DATA)

```python
# HOUGH_GRADIENT_ALT (primary)
_HOUGH_ALT_PARAM2      = 0.9
_HOUGH_ALT_DP          = 1.5
_HOUGH_ALT_BLUR_K      = 5

# Pre-filter gates
_PF_RADIUS_MIN         = 0.018   # observed min 0.032 — safe to raise to 0.025, probe first
_PF_STD_CLEAN_MAX      = 0.020
_PF_STD_HARD_MAX       = 0.130
_PF_BRDF_FALLBACK_MIN  = 0.28    # lowered by profile when photo_narrowing_ready
_PF_STD_FLOOR          = 0.008
_PF_RG_MIN/MAX         = 0.90/1.25
_PF_BG_MIN/MAX         = 0.80/1.20

# 5-gate pipeline
_RADIUS_MIN_RATIO      = 0.02
_RADIUS_MAX_RATIO      = 0.15    # NOT a bug — probe confirmed headroom to 0.058 observed max
_GATE_LAMBERTIAN_TOL   = 0.12
ring_radii             = [0.20, 0.45, 0.68, 0.80]r   # outer tightened from 0.88 in S14
_GATE_IRE_SPREAD_MIN   = 0.8
_GATE_STDDEV_MIN/MAX   = 0.003/0.170   # narrowed by profile when ready
_IRE_CONTEXT_FLOOR     = 18.0
_IRE_CONTEXT_CEIL      = 65.0
_INNER_MASK_RATIO      = 0.5
```

**Observed sphere population (35 clips, 3 sets):**
- r/w at detection scale: 0.032–0.058 (NOT 0.163–0.256 — that was profile radius_ratio using min_dim denominator)
- interior_std: 0.032–0.087
- ire_spread: 0.92–8.77
- chroma_dist: 0.019–0.034

---

## 11. MEASUREMENT MATH

```python
Y = 0.2126*R + 0.7152*G + 0.0722*B           # BT.709
trimmed = pixels[(pixels >= p5) & (pixels <= p95)]
IRE = (2 ** median(log2(clip(trimmed, 1e-6, None)))) * 100
# Inner disk: 0.5r
```

---

## 12. RCP2 PARAMETERS

```python
# Write order: kelvin → tint → exposureAdjust
COLOR_TEMPERATURE  # integer Kelvin, divider=1
TINT               # fixed-point, divider read live (fail-closed — aborts on timeout)
EXPOSURE_ADJUST    # fixed-point, divider read live (fail-closed)
# Neutral defaults: 5600K / 0.0 tint / 0.0 EA
```

---

## 13. TOOLS ON DESKTOP

- `r3dmatch_determinism_check.py` — compare array_calibration.json across runs
- `r3dmatch_stress_test.py` — run detection across all 36 TIFFs from 3 sets
- `build_sphere_profile.py` — mine analysis JSONs into v2 sphere profile
- `probe_rw_ceiling.py` — validates r/w gate changes (updated S15: uses Test_Run path)
- `sphere_profile_v2.json` — current 35-camera photometric population data
- `sphere_profile_v2_report.txt` — cross-camera stats and gate suggestions

---

## 14. SESSION 16 START PROMPT

Upload this document to a new chat, then say:

> "Continue R3DMatch development. Session 16, V3. Working directory is `/Users/sfouasnon/Desktop/R3DMatch_v3/R3DMatch_v3` (nested). Package is `r3dmatch_v3/src/r3dmatch3`, spec is `r3dmatch3.spec`. Source the venv and run one full pipeline pass on `_106` to confirm V3 is clean before making any changes. Session 15 handoff is attached."
