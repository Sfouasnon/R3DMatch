# R3DMatch v4 — v3 Review & v4 Architecture Plan

**Date:** 2026-06-17
**Author:** Cowork session
**Source reviewed:** `/Users/sfouasnon/Desktop/R3DMatch_v3/R3DMatch_v3/r3dmatch_v3` (read-only, no edits)
**Status of v3:** Stable, solving spheres well. Not modified.

---

## 1. Decisions locked for v4

| Decision | Choice |
|---|---|
| Measurement/match domain | **Hybrid** — detect + measure + solve on the fixed IPP2/BT.709 reference render; additionally render through the project transform+LUT and report match %/verification in that delivery domain. |
| Sphere solver | **Untouched.** `sphere.py`, `brdf_gate.py`, and the 4-step measurement math stay byte-identical on the reference path. |
| Gray card method | **Both, auto-detected** — recognize known charts (18% gray / ColorChecker, markers when present); fall back to operator-placed ROI. |
| Face method | **Array median**, mirroring the sphere workflow (anchor exposure + skin chroma to the array median across cameras). |
| First deliverable | This review + plan, for approval before any v4 code. |

---

## 2. What v3 is

A multi-camera exposure + color alignment tool for RED KOMODO-X arrays. An 18% gray sphere is placed in frame; each camera's R3D is rendered to a single TIFF via REDLine in IPP2/BT.709, the sphere is auto-detected, luminance is measured in IRE, and per-camera `exposureAdjust` / `kelvin` / `tint` commits are pushed live to the cameras over RCP2 before recording.

**Pipeline (one run):**

```
scan clips
  → read metadata (REDLine --printMeta 1, optional --printMeta 5 lens)
  → render measurement TIFF   (REDLine, hardcoded IPP2/BT.709/BT.1886/Med/Med, --useMeta)
  → detect sphere             (Hough + 5 sequential gates + LoG supplement + BRDF gate)
  → measure                   (4-step: luminance → trim 5–95 → log2 median → IRE; + interior chroma)
  → solve exposure            (target = robust median of hero_log2; per-camera offset)
  → solve white balance       (shared Kelvin / per-camera tint)
  → closed loop               (render corrected TIFF, re-measure, compute match %)
  → report + RCP2 push
```

**Module map (11.3k LOC, `src/r3dmatch3/`):**

| Module | LOC | Role |
|---|---|---|
| `app.py` | 2505 | PySide6 desktop UI, workflow rail, verify worker |
| `report.py` | 1961 | Contact-sheet HTML report |
| `sphere.py` | 1149 | **Sphere detection — the proven solver. Do not touch.** |
| `workflow.py` | 1031 | Orchestration: scan→render→detect→measure→solve→report; closed loop |
| `rcp2.py` | 994 | RCP2 WebSocket push (kelvin/tint/exposureAdjust), fail-closed |
| `sphere_profile.py` | 621 | Per-camera geometry + photometric priors |
| `redline.py` | 519 | **All REDLine subprocess calls. The single color-science chokepoint.** |
| `brdf_verify.py` | 445 | Standalone BRDF verification tooling |
| `solve.py` | 414 | Exposure + WB solve, match % scoring, run assessment |
| `web_app.py` | 390 | Flask server + /api routes |
| `brdf_gate.py` | 311 | BRDF spoke-correlation gate (inverts BT.1886) |
| `models.py` | 288 | Dataclasses (ClipMetadata, SphereROI, MeasurementResult, etc.) |
| `workflow_qc.py` | 255 | Re-measurement / QC pass |
| `measure.py` | 248 | 4-step measurement math + zone geometry + WC/GM |
| `settings.py` | 67 | REDLine path + output dir persistence |
| `progress.py` | 63 | JSON progress emission |

**What is proven and must be preserved exactly:** the 4-step measurement math, the exposure solve (median anchor, MAD outlier rejection), the sphere detection gate chain and its tuned constants, the BRDF gate, the RCP2 write contract, and the noise-floor-anchored match % model ("there is no FAIL").

---

## 3. The transform/LUT problem (why v4 is non-trivial)

RED color science is applied in exactly **one** place — the REDLine render in `redline.py`:

```python
_COLOR_SPACE_BT709 = "13"      # --colorSpace
_GAMMA_BT1886 = "32"           # --gammaCurve
_OUTPUT_TONEMAP_MEDIUM = "1"   # --outputToneMap
_ROLLOFF_MEDIUM = "3"          # --rollOff
_COLOR_SCI_IPP2 = "3"          # --colorSciVersion
```

Everything downstream reads RGB from the resulting TIFF and **assumes those exact pixels.** Three assumptions are hardcoded to BT.709/BT.1886:

| # | Assumption | Locations | Breaks under… |
|---|---|---|---|
| 1 | Luminance weights `0.2126 / 0.7152 / 0.0722` (BT.709 primaries) | `measure.py:_luminance`, `sphere.py:_to_gray/_to_gray_arr`, `brdf_gate.py:to_lum` | Different output primaries (Rec.2020, P3) |
| 2 | BT.1886 inverse `display^2.4` → scene-linear | `brdf_gate.py:invert_bt1886` | Different gamma curve, or any 3D LUT |
| 3 | Neutral reference `wc − 0.041`, `gm − (−0.016)`; IRE as fixed display code | `solve.py:build_commit_values`; IRE math throughout | Any transform/LUT that moves where neutral lands |

**Key physical facts that make the hybrid approach sound:**

- The same transform+LUT is applied to **every** camera, so a match achieved in one domain is preserved through any smooth monotonic LUT.
- `exposureAdjust` is applied **scene-linear, upstream of the transform** — a stop is a stop in any domain. The exposure solve is inherently domain-independent.
- What genuinely differs by domain: the *reported* match % (a residual looks bigger/smaller through a contrast-y LUT), the *neutral point*, and the sphere gates' tuned pixel statistics.

This is exactly why we solve on the proven reference render and verify through the project LUT.

---

## 4. v4 architecture

### 4.1 Color pipeline engine (new) — `colorpipeline.py`

A `ColorPipeline` describes how a project renders:

```python
@dataclass
class ColorPipeline:
    color_sci_version: str = "3"      # IPP2 (--colorSciVersion 3)
    color_space: str = "13"           # output primaries/gamut (--colorSpace)
    gamma_curve: str = "32"           # transfer (--gammaCurve; BT.1886=32)
    output_tone_map: str = "1"        # --outputToneMap (Low0/Med1/High2/None3)
    roll_off: str = "3"               # --rollOff (None0/Hard1/Def2/Med3/Soft4)
    shadow: str = "0.0"
    creative_lut_path: Optional[str] = None   # IPP2 show LUT  → --creativeLut
    output_lut_path: Optional[str] = None     # pre-baked LUT  → --lut (+ --lutEdgeLength)
    output_lut_edge: Optional[int] = None     # required with --lut
    cdl: Optional[CDL] = None                 # optional project CDL (--cdl* slope/offset/power)
    name: str = "IPP2 BT.709 (reference)"
```

**LUT mechanism (confirmed against REDLine Build 65.1.3 / R3DSDK 9.2.0):** a project show LUT is loaded with **`--creativeLut <file>`** (applied *inside* the IPP2 pipeline — the correct slot for an IPP2 creative look). A pre-baked output 3D LUT uses the generic **`--lut <file>` with `--lutEdgeLength <n>`**. v4 prefers `--creativeLut` and exposes `--lut` for the baked case. Many shows deliver a look as CDL + LUT, so the CDL slots (`--cdlRedSlope/Offset/Power`, `--cdlSaturation`) are wired too.

Two named pipelines exist per run:

- **`REFERENCE_PIPELINE`** — frozen, identical to v3's hardcoded values. The solve domain. Never project-configurable.
- **`delivery_pipeline`** — the project's chosen IPP2 transform + optional LUT. The verification/preview domain.

`redline.render_measurement_frame()` gains a `pipeline: ColorPipeline` parameter that builds the argument list (including the LUT flag) instead of hardcoded constants. **Default = `REFERENCE_PIPELINE`, so existing call sites behave identically.** This is the only change to `redline.py`'s render path, and it is additive.

**Accuracy work — characterizing a delivery pipeline.** For the verification-domain measurement to be correct, v4 derives, per delivery pipeline, at session start (one-time calibration render of synthetic neutral + gray ramps through REDLine):

- output **luminance weights** from the pipeline's primaries,
- the **neutral landing point** (where 18% gray maps in the delivery domain → the WB target),
- the **code-value↔relative-exposure curve** (so IRE and stops are interpreted correctly through the tone-map + LUT).

These become a `PipelineProfile` consumed only by the verification measurement path. The reference path keeps v3's constants verbatim.

**Characterization uses `--primaryDev`.** REDLine's `--primaryDev` exports ungraded RWG / Log3G10 — a true scene-referred image independent of any display transform. v4 uses it to anchor the calibration renders (known scene-linear input → push through the delivery pipeline → observe where neutral/luma land), which makes the `PipelineProfile` derivation exact rather than assumed. It is also the natural basis for a future scene-linear measurement domain if ever wanted.

### 4.2 Dual-render hybrid flow

```
per camera:
  render REFERENCE tiff   ──> detect ──> measure ──> solve   (v3 path, unchanged)
                                                       │ commit (exp/kelvin/tint)
  render DELIVERY tiff (corrected, through transform+LUT) ──> measure_in_domain
                                                       └─> match % / verification (operator-facing)
```

- Detection, measurement, and solve run on the reference render — the sphere solver and its gates see exactly the pixels they were tuned on.
- The closed-loop corrected render is produced in the **delivery** pipeline and measured with the `PipelineProfile`-aware measurement, so match % reflects the show look.
- Cost: one extra render per camera in the closed-loop phase (the phase already renders a corrected frame — in many cases this is a substitution, not a net-new render, when the operator only wants the delivery view).

### 4.3 Method abstraction (new) — `methods/`

Today `workflow.py` calls `detect_sphere` + `measure_render` directly. v4 introduces a thin strategy interface so sphere/gray-card/face share the solve while differing in detect + measure:

```python
class MatchMethod(Protocol):
    name: str
    def detect(self, image_hwc, meta, *, profile=None) -> DetectionResult: ...
    def measure(self, image_hwc, roi, *, clip_id) -> MeasurementResult: ...
    def reference_target(self, measurements) -> Target: ...   # how the array anchor is formed
```

- **`SphereMethod`** wraps the *existing* `sphere.detect_sphere` + `measure.measure_render` with zero behavioral change. It is the default and is selected for every existing project.
- **`GrayCardMethod`** and **`FaceMethod`** are new implementations (below).
- `workflow.py` is refactored only to route through the selected method; the solve, closed loop, report, and RCP2 push are method-agnostic and unchanged.

### 4.4 Gray card method (both, auto-detected) — `methods/graycard.py`

- **Detection:** try chart recognition first (ArUco/known marker layout, or ColorChecker via the classic 24-patch geometry); if found, sample the known neutral patch(es) with known reflectance. If no chart is recognized, accept an operator-placed (or simple quad-detected) flat ROI.
- **Measurement:** reuse the 4-step luminance→trim→log2→IRE chain on the patch (flat region — no zone gradient, no Lambertian/BRDF gate). Interior chroma → WC/GM exactly as the sphere path.
- **Solve:** unchanged. A recognized known chart gives an **absolute** exposure + neutral target; an arbitrary card falls back to the **array-median** anchor.
- Reuses `measure.compute_wc_gm`, `solve.solve_exposure`, `solve.solve_white_balance` verbatim.

### 4.5 Face method (array median) — `methods/face.py`

- **Detection:** face + landmark detection (e.g. mediapipe/cv2 DNN); derive stable sampling regions (forehead, cheeks) and exclude specular highlights, eyes, hair, shadow terminator.
- **Measurement:** skin luminance via the same 4-step chain on the sampled regions; skin chroma via WC/GM. Skin is **not** neutral, so there is no achromatic target — consistency across cameras is the goal.
- **Solve:** anchor exposure + skin chroma to the **array median** across cameras (mirrors the sphere workflow). Per-camera tint drives skin chroma toward the group median; exposure offset drives skin luminance to the group median. Solve math reused unchanged.
- Highest-novelty method; recommend validating against a controlled multi-camera face set before trusting on set.

### 4.6 Data model changes — `models.py` (additive)

- `ClipMetadata`: already carries `color_space`/`gamma_space`/`image_pipeline` (good — used to detect mismatches vs the project pipeline).
- New `ColorPipeline`, `PipelineProfile` dataclasses.
- `MeasurementResult`: add `domain: str` ("reference" | "delivery") and optional delivery-domain fields; keep all existing fields.
- `CameraResult`: add `delivery_match_pct` alongside the existing reference fields; existing fields unchanged.
- `RunResult`: add `method: str` and `delivery_pipeline: ColorPipeline`.

All additions default to v3 behavior so old runs/JSON stay readable.

---

## 5. Guardrails protecting the proven sphere solver

1. **No edits to `sphere.py` or `brdf_gate.py` logic.** Only the render-arg builder in `redline.py` changes, and it defaults to the reference pipeline.
2. **Golden regression test before/after:** run the existing `_106`, `_067`, `_064` sets through v4's reference path and assert detection ROIs, hero IRE, and commits are **bit-for-bit identical** to v3 (the repo already has `tests/`, `r3dmatch_determinism_check.py`, `r3dmatch_stress_test.py`, and `sphere_audit_*` to anchor this).
3. **Reference pipeline is not project-configurable** — it cannot be accidentally retuned by a project's LUT choice.
4. **Determinism check** re-run after the workflow refactor (handoff flags it as not confirmed post-Session 15 regardless).

---

## 6. Build phases (proposed)

1. **Scaffold:** copy v3 source into `R3DMatch_v4` unchanged; lock golden outputs from the three validation sets.
2. **Color pipeline engine:** `ColorPipeline` + parameterized `redline` render builder (default = reference). Prove byte-identical reference renders.
3. **Delivery render + characterization:** `PipelineProfile` (luma weights, neutral, code-value curve) from calibration renders; LUT flag wired and tested with a real `.cube`.
4. **Hybrid verification:** delivery-domain corrected render + domain-aware match %; report shows reference solve and delivery look side by side.
5. **Method abstraction:** extract `SphereMethod` (zero behavior change), route `workflow.py` through it.
6. **Gray card method.**
7. **Face method.**
8. **UI/report/RCP2:** method picker, project pipeline + LUT picker in Settings, delivery match % in the report and push table.
9. **Validation:** golden regression + determinism + new method validation on real footage.

---

## 7. Open items / risks

- ~~**LUT flag syntax**~~ **RESOLVED (REDLine Build 65.1.3 / R3DSDK 9.2.0):** show LUT = `--creativeLut <file>`; baked output LUT = `--lut <file> --lutEdgeLength <n>`. All v3 transform enums confirmed (colorSciVersion 3, colorSpace 13, gammaCurve 32, outputToneMap 1, rollOff 3). CDL slots available. `--primaryDev` available for scene-referred characterization.
- **Code-value↔exposure inversion through a creative LUT** is not analytically invertible; we characterize it numerically via calibration renders. Match % through the LUT is therefore measured, never modeled — consistent with v3's philosophy.
- **Face method** needs a face/landmark dependency added to `pyproject.toml` and on-set validation; it is the least-proven piece.
- **Sphere gates under unusual *reference* footage** are unchanged, so no new risk there — risk is isolated to the new delivery/verification path and the two new methods.
- Pre-existing v3 to-dos (sphere profile N=1, RCP2 hardware test, stale spec hidden imports) are noted but out of scope for v4 color/method work.
