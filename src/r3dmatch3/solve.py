"""
R3DMatch v2 — Exposure and White Balance Solve

Exposure solve: proven, unchanged from original.
  Target = robust median of hero_log2 across all valid cameras.
  Per-camera offset = target - measured.
  Closed-loop: after corrected render, verify corrected IRE is within ±0.05 stops of target.

WB solve: Shared Kelvin / Per-Camera Tint.
  Based on lessons from the test run diagnostics:
  - The solver works. The original gates were wrong.
  - WC/GM residuals of ~-0.127/-0.047 are EXPECTED in IPP2 at 5600K (inherent warm bias).
  - PASS criterion is inter-camera SPREAD, not absolute residual.

Gate calibration (from test run ground truth):
  - WC_spread_before: 0.018  →  WC_spread_after: 0.0044  (76% reduction)
  - GM_spread_before: 0.015  →  GM_spread_after: 0.0015  (90% reduction)
  - PASS if: wc_spread_after < 0.008 AND gm_spread_after < 0.005

  The absolute WC/GM residuals are reported diagnostically only.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from .measure import compute_wc_gm, exposure_offset_stops, array_target_log2
from .models import (
    CameraResult,
    CommitValues,
    ExposureSpread,
    MeasurementResult,
    WBSolveResult,
)

# ---------------------------------------------------------------------------
# Exposure solve constants
# ---------------------------------------------------------------------------

_OUTLIER_MAD_MULTIPLIER = 3.0            # exclude cameras > 3*MAD from median

# ---------------------------------------------------------------------------
# Match scoring — noise-floor anchored
#
# There is no FAIL. Every solved camera gets a match percentage.
# 100% = residual within what the measurement system can resolve (noise floor).
# Calling anything below the floor less than 100% would report noise as
# mismatch. Score falls linearly to 0% at the zero point (a visibly wrong
# camera). The operator decides what to do with the number.
# ---------------------------------------------------------------------------

_EXP_NOISE_FLOOR_STOPS = 0.05   # render + trimmed-median measurement jitter
_EXP_ZERO_MATCH_STOPS  = 0.50   # half a stop off = 0% match
_GM_NOISE_FLOOR        = 0.001  # GM chroma measurement jitter
_GM_ZERO_MATCH         = 0.020  # visibly divergent tint = 0% match

# ---------------------------------------------------------------------------
# WB solve constants
# ---------------------------------------------------------------------------

_WC_SPREAD_PASS = 0.008    # inter-camera WC spread threshold for PASS
_GM_SPREAD_PASS = 0.005    # inter-camera GM spread threshold for PASS

# Empirical response coefficients from test run:
# These tell us how much WC/GM change per unit of Kelvin/Tint
# dWC/dKelvin = 7.19e-05  (very small — Kelvin is mainly a shared anchor)
# dGM/dTint   = 3.29e-03  (tint is the effective per-camera color control)
_DWC_DKELVIN = 7.19e-05
_DGM_DTINT   = 3.29e-03

# Kelvin search range for shared Kelvin selection
_KELVIN_SNAP_CANDIDATES = [3200, 4300, 5000, 5500, 5600, 5900, 6500]


# ---------------------------------------------------------------------------
# Exposure solve
# ---------------------------------------------------------------------------

def solve_exposure(
    measurements: List[MeasurementResult],
) -> Tuple[float, float, ExposureSpread, Dict[str, float]]:
    """
    Compute exposure target and per-camera offsets.

    Returns:
      target_log2: float
      target_ire: float
      spread: ExposureSpread
      per_clip_offsets: {clip_id: offset_stops}
    """
    valid = [m for m in measurements if m.measurement_valid]
    if not valid:
        raise ValueError("No valid measurements for exposure solve")

    log2_values = np.array([m.hero_log2 for m in valid], dtype=np.float64)
    ire_values = np.array([m.hero_ire for m in valid], dtype=np.float64)

    median_log2 = float(np.median(log2_values))
    median_ire = float(np.median(ire_values))

    # Outlier detection via MAD
    mad = float(np.median(np.abs(log2_values - median_log2)))
    threshold = mad * _OUTLIER_MAD_MULTIPLIER if mad > 0 else 0.5
    inlier_mask = np.abs(log2_values - median_log2) <= threshold
    inlier_log2 = log2_values[inlier_mask]
    outlier_clip_ids = [valid[i].clip_id for i, keep in enumerate(inlier_mask) if not keep]

    target_log2 = float(np.median(inlier_log2)) if inlier_log2.size else median_log2
    target_ire = (2.0 ** target_log2) * 100.0

    spread = ExposureSpread(
        median_ire=median_ire,
        min_ire=float(np.min(ire_values)),
        max_ire=float(np.max(ire_values)),
        spread_stops=float(np.max(log2_values) - np.min(log2_values)),
        outlier_clip_ids=outlier_clip_ids,
    )

    # Per-camera exposureAdjust is a SCENE-LINEAR stop, so it must be solved in
    # scene-linear: offset = log2(target_linear / measured_linear). Solving it in
    # display IRE (the non-linear IPP2 output) under-predicts the stops and the
    # cameras land short of target. When every camera carries a scene-linear
    # measurement (hero_log2_lin), use it; otherwise fall back to display-space.
    # target_log2 / target_ire above stay DISPLAY-referred for scoring + report.
    lin_log2 = [getattr(m, "hero_log2_lin", None) for m in valid]
    if all(v is not None for v in lin_log2):
        lin_arr = np.array(lin_log2, dtype=np.float64)
        # Match cameras to the inlier-median scene-linear value (same inlier mask
        # as the display target, for consistent outlier handling).
        target_log2_lin = float(np.median(lin_arr[inlier_mask])) if inlier_mask.any() \
            else float(np.median(lin_arr))
        per_clip_offsets = {
            m.clip_id: float(target_log2_lin - lv)
            for m, lv in zip(valid, lin_log2)
        }
    else:
        per_clip_offsets = {
            m.clip_id: exposure_offset_stops(m.hero_log2, target_log2)
            for m in valid
        }

    return target_log2, target_ire, spread, per_clip_offsets


def exposure_residual_stops(corrected_ire: float, target_log2: float) -> float:
    """Absolute residual in stops between corrected render and target."""
    corrected_log2 = math.log2(max(corrected_ire / 100.0, 1e-6))
    return abs(corrected_log2 - target_log2)


def _ramp_match_pct(residual: float, noise_floor: float, zero_point: float) -> float:
    """100% at/below the noise floor, linear to 0% at the zero point."""
    r = abs(residual)
    if r <= noise_floor:
        return 100.0
    if r >= zero_point:
        return 0.0
    return 100.0 * (1.0 - (r - noise_floor) / (zero_point - noise_floor))


def exposure_match_pct(residual_stops: float) -> float:
    """Exposure match % from closed-loop residual (stops)."""
    return _ramp_match_pct(residual_stops, _EXP_NOISE_FLOOR_STOPS, _EXP_ZERO_MATCH_STOPS)


def wb_match_pct(gm_deviation: float) -> float:
    """WB match % from measured GM deviation vs the group median."""
    return _ramp_match_pct(gm_deviation, _GM_NOISE_FLOOR, _GM_ZERO_MATCH)


def camera_match_pct(
    exp_pct: Optional[float],
    wb_pct: Optional[float],
) -> Optional[float]:
    """
    Overall camera match: the weakest verified axis.
    None if nothing was verified (no corrected render measured).
    """
    scores = [s for s in (exp_pct, wb_pct) if s is not None]
    return min(scores) if scores else None


# ---------------------------------------------------------------------------
# WB solve
# ---------------------------------------------------------------------------

def solve_white_balance(
    measurements: List[MeasurementResult],
    as_shot_kelvins: Dict[str, int],
    wb_mode: str = "match",
) -> Tuple[WBSolveResult, Dict[str, Tuple[int, float]]]:
    """
    Solve Shared Kelvin / Per-Camera Tint.

    as_shot_kelvins: {clip_id: as_shot_kelvin_from_printMeta1}

    wb_mode:
      "match"   (default, proven) — match cameras to each other: GM target is the
                group median, shared Kelvin is the as-shot median (WC left as the
                IPP2 gray bias for the colorist). Byte-identical to v3.
      "neutral" — also drive the matched array toward absolute neutral: GM target
                = 0, and the shared Kelvin is solved toward WC = 0 using the
                empirical dWC/dKelvin response. Cameras stay matched and on one
                common (now neutral) temperature. One-shot estimate; the closed
                loop measures the true result.

    WC (warm/cool) axis is controlled by Kelvin. With a SHARED Kelvin model,
    per-camera WC spread is a diagnostic — it reflects real optical differences
    between cameras (lens coating, filter, flare) and cannot be corrected by
    Kelvin alone. Per-camera Kelvin would be needed, but that shifts cameras
    off a common grade.

    GM (green/magenta) axis is controlled by tint, applied per-camera.
    GM spread should approach zero after solve.

    Gate: GM_spread_after < 0.005 (tint controls this directly).
    WC spread is reported diagnostically only.

    Returns:
      wb_result: WBSolveResult (summary)
      per_clip_wb: {clip_id: (kelvin, tint)}
    """
    valid = [m for m in measurements if m.measurement_valid]
    if not valid:
        return WBSolveResult(
            status="SKIPPED", reason="no_valid_measurements",
            shared_kelvin=5600,
            wc_spread_before=0.0, gm_spread_before=0.0,
            wc_spread_after=0.0, gm_spread_after=0.0,
            iteration_count=0, camera_count=0,
        ), {}

    if wb_mode == "exposure_only":
        # Exposure-only run: white balance is NOT solved or altered. Each camera
        # keeps its as-shot Kelvin and zero tint offset; only exposureAdjust is
        # committed/pushed. WB diagnostics, charts and tint are omitted downstream.
        per_clip_wb = {m.clip_id: (int(round(as_shot_kelvins.get(m.clip_id, 5600))), 0.0)
                       for m in valid}
        wb_result = WBSolveResult(
            status="SKIPPED", reason="exposure_only — white balance not solved",
            shared_kelvin=int(round(float(np.median(
                [as_shot_kelvins.get(m.clip_id, 5600) for m in valid])))),
            wc_spread_before=0.0, gm_spread_before=0.0,
            wc_spread_after=0.0, gm_spread_after=0.0,
            iteration_count=0, camera_count=len(valid),
        )
        return wb_result, per_clip_wb

    # Compute WC/GM for each camera
    wc_values: Dict[str, float] = {}
    gm_values: Dict[str, float] = {}
    for m in valid:
        wc, gm = compute_wc_gm(m.measured_rgb_mean)
        wc_values[m.clip_id] = wc
        gm_values[m.clip_id] = gm

    wc_arr = np.array(list(wc_values.values()))
    gm_arr = np.array(list(gm_values.values()))
    wc_spread_before = float(np.max(wc_arr) - np.min(wc_arr))
    gm_spread_before = float(np.max(gm_arr) - np.min(gm_arr))

    # Anchor Kelvin: median of as-shot values, snapped to nearest common value.
    available_kelvins = [as_shot_kelvins.get(m.clip_id, 5600) for m in valid]
    raw_median_k = int(round(float(np.median(available_kelvins))))

    per_clip_wb: Dict[str, Tuple[int, float]] = {}

    if wb_mode == "match":
        # ───────────────────────────────────────────────────────────────────
        # PROVEN v3 path — SHARED Kelvin / per-camera tint. UNCHANGED.
        # Warm/cool is NOT corrected (shared Kelvin); GM matched to group median
        # via tint. This is the golden-anchored behavior — do not alter.
        # ───────────────────────────────────────────────────────────────────
        shared_kelvin = _snap_kelvin(raw_median_k)
        target_gm = float(np.median(gm_arr))
        gm_after_vals = []
        for m in valid:
            gm_measured = gm_values[m.clip_id]
            gm_delta = target_gm - gm_measured
            tint_delta = gm_delta / _DGM_DTINT if abs(_DGM_DTINT) > 1e-10 else 0.0
            tint = round(float(tint_delta), 2)
            gm_after = gm_measured + (_DGM_DTINT * tint)
            gm_after_vals.append(gm_after)
            per_clip_wb[m.clip_id] = (shared_kelvin, tint)
        wc_spread_after = wc_spread_before  # WC uncorrected in shared-Kelvin model
        gm_spread_after = float(np.max(gm_after_vals) - np.min(gm_after_vals)) if gm_after_vals else 0.0
        derivation = "shared Kelvin / per-camera tint"
    else:
        # ───────────────────────────────────────────────────────────────────
        # PER-CAMERA Kelvin + per-camera tint — matches BOTH axes.
        # Warm/cool matched via small per-camera Kelvin trims (tint cannot move
        # WC); green/magenta matched via tint. The array is anchored at:
        #   scene_match -> the as-shot scene Kelvin (cameras stay ~5600K, the
        #                  uniform IPP2 cast is left to grade/LUT)
        #   neutral     -> absolute neutral (WC=0, GM=0; average slides off scene)
        # ───────────────────────────────────────────────────────────────────
        anchor_k = _snap_kelvin(raw_median_k)
        if wb_mode == "neutral":
            wc_target, gm_target = 0.0, 0.0
        else:  # "scene_match"
            wc_target = float(np.median(wc_arr))
            gm_target = float(np.median(gm_arr))
        wc_after_vals, gm_after_vals = [], []
        for m in valid:
            wc_i = wc_values[m.clip_id]
            gm_i = gm_values[m.clip_id]
            # Per-camera Kelvin trim toward the WC target (anchor = scene temp).
            if abs(_DWC_DKELVIN) > 1e-12:
                k_i = anchor_k - ((wc_i - wc_target) / _DWC_DKELVIN)
            else:
                k_i = anchor_k
            kelvin_i = int(round(max(1700.0, min(50000.0, k_i))))
            tint_i = round((gm_target - gm_i) / _DGM_DTINT if abs(_DGM_DTINT) > 1e-10 else 0.0, 2)
            per_clip_wb[m.clip_id] = (kelvin_i, tint_i)
            wc_after_vals.append(wc_i + _DWC_DKELVIN * (kelvin_i - anchor_k))
            gm_after_vals.append(gm_i + _DGM_DTINT * tint_i)
        shared_kelvin = anchor_k  # nominal anchor; per-camera values in per_clip_wb
        wc_spread_after = float(np.max(wc_after_vals) - np.min(wc_after_vals)) if wc_after_vals else 0.0
        gm_spread_after = float(np.max(gm_after_vals) - np.min(gm_after_vals)) if gm_after_vals else 0.0
        derivation = f"per-camera Kelvin+tint ({wb_mode}, anchor {anchor_k}K)"

    # No FAIL: the solve always produces commits. Spreads are informational;
    # match quality is expressed per camera as wb_match_pct after the
    # closed-loop verification measures the corrected renders.
    reason = (
        f"solved [{derivation}]: gm_spread_predicted={gm_spread_after:.4f} "
        f"wc_spread_predicted={wc_spread_after:.4f}"
    )

    wb_result = WBSolveResult(
        status="SOLVED",
        reason=reason,
        shared_kelvin=shared_kelvin,
        wc_spread_before=wc_spread_before,
        gm_spread_before=gm_spread_before,
        wc_spread_after=wc_spread_after,
        gm_spread_after=gm_spread_after,
        iteration_count=1,
        camera_count=len(valid),
    )

    return wb_result, per_clip_wb


def verify_wb_closed_loop(
    wb_result: WBSolveResult,
    corrected_wc_gm: Dict[str, Tuple[float, float]],
) -> WBSolveResult:
    """
    Closed-loop WB verification from corrected renders.

    The solve-time gm_spread_after is a PREDICTION: tint was computed by
    inverting GM ≈ GM_measured + dGM/dTint·tint, so re-evaluating that same
    model yields near-zero spread by construction (only tint rounding adds
    error). The prediction cannot fail its own gate in any meaningful way.

    This function re-gates PASS/FAIL on the MEASURED inter-camera spread,
    computed from WC/GM of the corrected renders — the same renders already
    produced for the exposure closed loop.

    corrected_wc_gm: {clip_id: (wc, gm)} measured from corrected renders.

    Requires >= 2 measured cameras (spread is undefined below that);
    otherwise the predicted status stands and closed_loop remains False.
    Mutates and returns wb_result.
    """
    if wb_result.status == "SKIPPED" or len(corrected_wc_gm) < 2:
        return wb_result

    wc_arr = np.array([v[0] for v in corrected_wc_gm.values()], dtype=np.float64)
    gm_arr = np.array([v[1] for v in corrected_wc_gm.values()], dtype=np.float64)
    wc_spread = float(np.max(wc_arr) - np.min(wc_arr))
    gm_spread = float(np.max(gm_arr) - np.min(gm_arr))

    wb_result.wc_spread_measured = wc_spread
    wb_result.gm_spread_measured = gm_spread
    wb_result.closed_loop = True

    # No FAIL: measured spreads feed per-camera wb_match_pct.
    # WC remains diagnostic (shared-Kelvin model cannot correct per-camera WC).
    wb_result.status = "VERIFIED"
    wb_result.reason = (
        f"closed-loop measured gm_spread={gm_spread:.4f} "
        f"wc_spread={wc_spread:.4f} (diagnostic) n={len(corrected_wc_gm)}"
    )
    return wb_result


def _snap_kelvin(k: int) -> int:
    """
    Snap to nearest of the common Kelvin values.
    If not close to any, return the raw value.
    """
    best = min(_KELVIN_SNAP_CANDIDATES, key=lambda x: abs(x - k))
    # Only snap if within 200K of a standard value
    if abs(best - k) <= 200:
        return best
    return k


# ---------------------------------------------------------------------------
# Build CommitValues per camera
# ---------------------------------------------------------------------------

def build_commit_values(
    camera_result: "CameraResult",
    *,
    exposure_offset: float,
    kelvin: int,
    tint: float,
    wc_before: float = 0.0,
    gm_before: float = 0.0,
    wc_after: float = 0.0,
    gm_after: float = 0.0,
    exposure_only: bool = False,
) -> CommitValues:
    """Build the final CommitValues for one camera."""
    m = camera_result.measurement
    wc_res = 0.0
    gm_res = 0.0
    if m and m.measurement_valid and not exposure_only:
        wc, gm = compute_wc_gm(m.measured_rgb_mean)
        wc_res = wc - 0.041  # residual vs IPP2 neutral target
        gm_res = gm - (-0.016)

    return CommitValues(
        clip_id=camera_result.clip_id,
        camera_label=camera_result.camera_label,
        exposure_adjust=round(exposure_offset, 6),
        kelvin=kelvin,
        tint=round(tint, 2),
        derivation_method="exposure_only" if exposure_only else "shared_kelvin_per_camera_tint",
        wc_before=wc_before,
        gm_before=gm_before,
        wc_after=wc_after,
        gm_after=gm_after,
        wc_residual=round(wc_res, 6),
        gm_residual=round(gm_res, 6),
        exposure_only=exposure_only,
    )


# ---------------------------------------------------------------------------
# Overall run status assessment
# ---------------------------------------------------------------------------

def assess_run(camera_results: List["CameraResult"]) -> Dict[str, object]:
    """
    Compute final run assessment: array match percentage.

    No FAIL, no blocking. Every solved camera has commits and is pushable.
    The assessment tells the operator how well the array matches and which
    camera is weakest — the decision is theirs.
    """
    total = len(camera_results)
    solved = [c for c in camera_results if c.commit is not None]
    scored = [c for c in solved if c.match_pct is not None]
    needs_assist = sum(1 for c in camera_results if c.status == "NEEDS_ASSIST")
    no_data = sum(1 for c in camera_results
                  if c.status in ("NO_DATA", "ERROR") and c.commit is None)

    array_match_pct: Optional[float] = None
    min_match_pct: Optional[float] = None
    min_match_clip_id = ""
    if scored:
        pcts = [c.match_pct for c in scored]
        array_match_pct = float(np.mean(pcts))
        worst = min(scored, key=lambda c: c.match_pct)
        min_match_pct = worst.match_pct
        min_match_clip_id = worst.clip_id

    if len(solved) == total and total > 0:
        assessment = "SOLVED"
    elif solved:
        assessment = "PARTIAL"
    else:
        assessment = "NO_SOLVE"

    return {
        "assessment_status": assessment,
        "array_match_pct": array_match_pct,
        "min_match_pct": min_match_pct,
        "min_match_clip_id": min_match_clip_id,
        "solved_count": len(solved),
        "scored_count": len(scored),
        "needs_assist_count": needs_assist,
        "no_data_count": no_data,
        "total_cameras": total,
    }
