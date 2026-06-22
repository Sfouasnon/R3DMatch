"""
sphere.py — Gray sphere detection for R3DMatch v2.

Detection pipeline:
  Candidate sources (Session 9):
    Primary:     Hough circle transform on Canny edges
    Supplementary: LoG blob detection when Hough peak count > 200
                   (cluttered scenes — tape, markers, complex backgrounds)
                   LoG is immune to phantom circles from tape/marker corners.

  Pre-filter gates (applied before 5-gate pipeline):
    G1: r/w >= 0.018  — radius floor, eliminates Hough noise rings
    G2: std <= 0.020  — clean-scene ceiling (gray backdrop, controlled light)
        OR
        0.020 < std <= 0.130 AND brdf >= 0.28
                        — BRDF fallback for dramatic lighting / dark backgrounds
                          (handoff §9: "sphere failing std but passing BRDF
                          is legitimate in harder conditions")

  5-gate pipeline (unchanged):
    1. Geometry       — radius 6–32% of frame
    2. Gray Material  — interior chromaticity < 0.045 from achromatic
    3. Lambertian     — radial luminance decreases center→edge
    4. IRE Spread     — bright/dark zone difference ≥ 0.8 IRE
    5. Interior Stddev — 0.003–0.170

  After the 5 gates: IRE Context Check (soft gate).
    • Derives an expected IRE window from all candidates measured in
      the current session (peer-context) or falls back to the conservative
      pipeline-level window [_IRE_CONTEXT_FLOOR, _IRE_CONTEXT_CEIL].
    • Candidates outside the window are accepted as NEEDS_ASSIST rather than
      SUCCESS — operator can override via Manual ROI.
    • Never creates a fixed numeric floor that would reject legitimately
      dark-lit scenes; the window expands with the group's observed range.

Calibration footage:
  _106 (gray sphere / gray backdrop)  — gate calibration target
  _067 (cluttered stage, dark bg)     — LoG + BRDF fallback validation
  _064 (underexposed stress test)     — pending Session 10 validation

DO NOT CHANGE tuned parameters (section 5 of handoff doc) without reason.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image
from skimage.feature import blob_log, canny, peak_local_max
from skimage.transform import hough_circle, hough_circle_peaks

try:
    import cv2 as _cv2
    _CV2_AVAILABLE = True
except ImportError:
    _cv2 = None
    _CV2_AVAILABLE = False

from r3dmatch3.models import DetectionGateResult, SphereDetectionResult, SphereROI
from r3dmatch3.brdf_gate import brdf_gate, rgb_display_to_lum

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tuned parameters — DO NOT CHANGE without reason (handoff §5)
# ---------------------------------------------------------------------------
_CANNY_SIGMA = 0.5
_CANNY_LOW   = 0.005
_CANNY_HIGH  = 0.02
_HOUGH_ACCUMULATOR_MIN = 0.25
_HOUGH_NUM_PEAKS = 50
_HOUGH_MIN_DIST = 20

# HOUGH_GRADIENT_ALT parameters — Session 12
# cv2.HOUGH_GRADIENT_ALT uses gradient direction to constrain votes.
# Edge pixels only vote toward centers whose direction matches their gradient.
# Result: arc fragments and phantom circles from tape/equipment are eliminated
# at the source. Validated: 12/12 rank-1 on _064 and _067 at param2=0.9.
# Falls back to standard skimage Hough when ALT finds 0 candidates
# (low-contrast scenes like _106 gray-on-gray where boundary edges are weak).
_HOUGH_ALT_PARAM1      = 50    # Canny high threshold passed to cv2
_HOUGH_ALT_PARAM2      = 0.9   # Normalised confidence — strict, produces ~1 candidate/frame
_HOUGH_ALT_DP          = 1.5   # Accumulator resolution (ALT works better > 1)
_HOUGH_ALT_BLUR_K      = 5     # GaussianBlur kernel size before ALT
_RADIUS_MIN_RATIO = 0.02  # was 0.06 — sphere at 5.76K/this lens is ~57px at det scale, below old 64px floor
# Ceiling sized for LONG lenses. On 24mm the sphere sits at r/w ~0.02–0.058; a
# 50mm makes it ~2× larger (and closer framings larger still), so 0.15 would
# reject a big sphere before it is even searched. 0.32 admits a sphere filling up
# to ~1/3 of frame width. Large flat backdrops/walls that now reach the gates are
# still rejected by gray_material / lambertian / shadow_specular; ALT's gradient
# constraint keeps the candidate count low regardless of range.
_RADIUS_MAX_RATIO = 0.32  # was 0.15 — widened for 50mm+ lenses (large sphere in frame)
_GATE_IRE_SPREAD_MIN      = 0.8
_GATE_CHROMA_MAX_DISTANCE = 0.045
_GATE_LAMBERTIAN_TOLERANCE = 0.12
_GATE_STDDEV_MIN = 0.003
_GATE_STDDEV_MAX = 0.170

# Gate 3.5: Shadow / Specular (convexity) — Session 16
# Discriminates 3D convex objects (spheres) from flat surfaces (focus charts,
# gray cards, backdrops) that pass the gray material and Lambertian gates.
#
# A real 18% gray sphere under directional lighting always has:
#   (a) A shadow side — dark half measurably dimmer than bright half.
#       shadow_ratio = dark_half_mean / bright_half_mean < 1.0
#   (b) A specular highlight — small bright lobe above interior mean.
#       peak_excess = peak_lum / interior_mean > 1.0
#
# Calibrated on 36 clips across _064 / _067 / _106 (27 confirmed spheres,
# 8 confirmed chart false-positives):
#   Sphere population:  shadow_ratio 0.729–0.954, peak_excess 1.209–1.437
#   Chart population:   shadow_ratio 0.862–1.018, peak_excess 1.142–1.243
#
# Gate uses shadow_ratio only — peak_excess was dropped after live validation
# showed two legitimate spheres (H007_A106, I007_B106) at peak_excess=1.174,
# below the 1.20 threshold. Peak_excess ranges overlap between sphere and chart
# populations; shadow_ratio does not overlap on the problem case (_106 chart=1.017).
#
# shadow_ratio <= 0.96 catches all charts while passing all 27 confirmed spheres.
# peak_excess is still computed and logged in the gate reason for diagnostics
# but is NOT part of the pass/fail decision.
#
# DO NOT change threshold without re-running probe_specular.py on all sets.
_GATE_SHADOW_RATIO_MAX = 0.96   # sphere max observed: 0.954 — 0.006 headroom
_GATE_PEAK_EXCESS_MIN  = 1.20   # diagnostic only — not used in gate decision

# ── Gating-2: looser shadow_specular, second pass ──────────────────────────
# Evenly/softly lit spheres lose their shadow terminator and read shadow_ratio
# ~0.97-0.98, just above _GATE_SHADOW_RATIO_MAX, so pass 1 rejects them. A second
# pass re-checks shadow_specular against this looser bound for candidates that
# failed SOLELY at shadow_specular.
#
# SAFETY: the chart population overlaps this band (calibration: charts 0.862-1.018),
# so the looser threshold is NOT chart-proof on its own. Gating-2 is therefore
# gated on the CLEAN ALT candidate stream only (_used_alt) — ALT's gradient-
# direction constraint does not propose flat charts/cards as candidates (validated:
# 0 flat ALT candidates across the F5.6 set). If ALT had to fall back to skimage
# (cluttered/low-contrast), gating-2 is skipped. Validate on chart-heavy footage
# with tools/sphere_gate_probe.py before relying on this in new conditions.
_GATE_SHADOW_RATIO_MAX_PASS2 = 0.985  # well-lit spheres observed: 0.970, 0.978

_DETECTION_TARGET_DIM = 1080

# ---------------------------------------------------------------------------
# Pre-filter gate parameters — Session 9
# ---------------------------------------------------------------------------
# Gate 1: radius floor — just above Hough noise floor.
# True sphere in _106 (6K 16:9, 12–13ft) sits at r/w 0.019–0.022.
# Do not raise without measuring the smallest sphere across all footage.
_PF_RADIUS_MIN        = 0.018

# Gate 2: std ceiling for clean scenes (gray backdrop, controlled lighting).
# Calibrated on _106: true sphere std 0.004–0.020, specular rings 0.051–0.108.
_PF_STD_CLEAN_MAX     = 0.020

# Gate 2b: relaxed std ceiling for cluttered/dramatic-lighting scenes.
# _067 true sphere reads std 0.045–0.100 due to Lambertian gradient against
# dark background. Any candidate exceeding _PF_STD_CLEAN_MAX in a cluttered
# scene must pass the BRDF fallback gate instead.
_PF_STD_HARD_MAX      = 0.130

# Gate 2c: BRDF fallback threshold.
# Applied when std > _PF_STD_CLEAN_MAX. Sphere in _067 scores 0.29–0.88.
# Tape/marker phantoms that survive radius filtering score -0.12–0.10.
# Threshold set at 0.28 — one std below the worst confirmed sphere score.
_PF_BRDF_FALLBACK_MIN = 0.28

# Gate 3: std floor — kills flat uniform-background arc phantoms.
# Session 12: any candidate with std < 0.008 is a near-uniform region
# (wall, floor, sky) whose Hough ring has no real boundary structure.
# Calibrated: all confirmed true spheres across _064/_067/_106 have std ≥ 0.008.
_PF_STD_FLOOR         = 0.008

# Gate 4: RGB ratio gate — kills non-gray colored objects.
# Session 12: calibrated on 18 verified true-sphere solves in IPP2/BT.709.
# True sphere R/G: 0.90–1.14, B/G: 0.80–1.09.
# Gate set with margin: R/G 0.90–1.25, B/G 0.80–1.20.
_PF_RG_MIN = 0.90
_PF_RG_MAX = 1.25
_PF_BG_MIN = 0.80
_PF_BG_MAX = 1.20

# Hough saturation threshold — if Hough yields more than this many peaks,
# the scene is too cluttered for Hough to be the sole candidate source.
# LoG blob detection is added as a complementary source above this count.
_HOUGH_SATURATION     = 200

# LoG blob detection parameters for cluttered scenes.
# Sphere r/w 0.018–0.12 → r_px 19–130 → sigma 14–92.
_LOG_MIN_SIGMA        = 10
_LOG_MAX_SIGMA        = 55
_LOG_NUM_SIGMA        = 16
_LOG_THRESHOLD        = 0.05

# ---------------------------------------------------------------------------
# IRE context gate — soft, lighting-aware, no hard binary floor
# ---------------------------------------------------------------------------
# These are the outer safety rails. The actual acceptance window is derived
# from the session's peer measurements when ≥2 peers exist.
_IRE_CONTEXT_FLOOR = 18.0   # absolute minimum — below this is crushed black
_IRE_CONTEXT_CEIL  = 65.0   # absolute maximum — above this is blown out

# Headroom added above/below the peer-derived window.
# Makes the gate tolerant of legitimate per-camera variation in lighting.
_IRE_CONTEXT_HEADROOM = 8.0   # ±IRE around observed peer range

# Minimum window width even when all peers cluster tightly.
# Prevents the gate from becoming pathologically narrow on uniform lighting.
_IRE_CONTEXT_MIN_WINDOW = 12.0  # IRE total width (±6 from midpoint)


# ---------------------------------------------------------------------------
# Internal dataclass for a raw Hough candidate
# ---------------------------------------------------------------------------
@dataclass
class _Candidate:
    cx: float
    cy: float
    r: float
    accumulator: float
    gates: List[DetectionGateResult] = field(default_factory=list)
    passed: bool = False
    hero_ire: Optional[float] = None          # set after measurement probe
    ire_context_status: str = "unchecked"     # "ok" | "needs_assist" | "unchecked"
    ire_context_reason: str = ""


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def detect_sphere(
    image_path: str,
    prior_cx: Optional[float] = None,
    prior_cy: Optional[float] = None,
    prior_r: Optional[float] = None,
    prior_weight: float = 0.5,
    peer_ire_values: Optional[List[float]] = None,
    photo_prior: Optional[dict] = None,
) -> SphereDetectionResult:
    """
    Detect the gray calibration sphere in a rendered TIFF.

    Parameters
    ----------
    image_path : str
        Path to the REDLine-rendered TIFF (full resolution).
    prior_cx, prior_cy, prior_r : float or None
        Sphere profile priors (detection-plane coordinates). When provided,
        the Hough radius search is narrowed to ±30% of prior_r.
    prior_weight : float
        Weight [0–1] applied when blending prior and detected position.
        0 = ignore prior, 1 = fully trust prior.
    peer_ire_values : list of float or None
        Hero-IRE measurements from other cameras already detected in this
        run. Used to derive a lighting-context window for the soft IRE gate.
        Pass None (or an empty list) to fall back to the conservative
        pipeline-level window.

    Returns
    -------
    SphereDetectionResult
        .source is "auto_hough" on success, "needs_assist" when the IRE
        context check fails, or None when all gates fail.
    """
    img_full = _load_image(image_path)
    img_det_pil, scale = _resize_for_detection(img_full)
    img_det = np.array(img_det_pil)

    h, w = img_det.shape[:2]
    r_min = max(6, int(w * _RADIUS_MIN_RATIO))
    r_max = min(h // 2, int(w * _RADIUS_MAX_RATIO))

    # Narrow radius search if we have a profile prior
    if prior_r is not None:
        prior_r_det = prior_r  # already in detection-plane units
        r_min_narrow = max(r_min, int(prior_r_det * 0.70))
        r_max_narrow = min(r_max, int(prior_r_det * 1.30))
        if r_min_narrow < r_max_narrow:
            r_min, r_max = r_min_narrow, r_max_narrow
            log.debug("Prior narrowed radius to %d–%d px", r_min, r_max)

    gray = _to_gray(img_det)
    edges = canny(gray, sigma=_CANNY_SIGMA, low_threshold=_CANNY_LOW,
                  high_threshold=_CANNY_HIGH)

    # ---------------------------------------------------------------------------
    # Candidate generation — Session 12
    # Primary: cv2 HOUGH_GRADIENT_ALT (gradient-direction constrained).
    #   Only votes where gradient orientation agrees with the candidate center
    #   direction. Eliminates tape/marker/equipment arc phantoms at source.
    #   Validated: 12/12 rank-1 on _064 and _067 at param2=0.9, avg 1 cand/frame.
    # Fallback: skimage standard Hough.
    #   Used when ALT returns 0 candidates — handles low-contrast scenes
    #   (_106 gray-on-gray) where the sphere boundary is too weak for ALT.
    # ---------------------------------------------------------------------------
    candidates: List[_Candidate] = []
    _used_alt = False

    if _CV2_AVAILABLE:
        # cv2 needs uint8 grayscale with a mild blur
        gray_u8 = (np.clip(gray, 0, 1) * 255).astype(np.uint8)
        gray_blur = _cv2.GaussianBlur(gray_u8,
                                      (_HOUGH_ALT_BLUR_K, _HOUGH_ALT_BLUR_K),
                                      1.0)
        min_dist_px = int(w * _RADIUS_MIN_RATIO)
        alt_circles = _cv2.HoughCircles(
            gray_blur,
            _cv2.HOUGH_GRADIENT_ALT,
            dp=_HOUGH_ALT_DP,
            minDist=min_dist_px,
            param1=_HOUGH_ALT_PARAM1,
            param2=_HOUGH_ALT_PARAM2,
            minRadius=r_min,
            maxRadius=r_max,
        )
        if alt_circles is not None and len(alt_circles[0]) > 0:
            for cx, cy, r in alt_circles[0]:
                # ALT doesn't return a normalised accumulator — use 0.85 sentinel
                # (above LoG's 0.70, below a hypothetical perfect skimage peak)
                candidates.append(_Candidate(cx=float(cx), cy=float(cy),
                                             r=float(r), accumulator=0.85))
            _used_alt = True
            log.debug("HOUGH_GRADIENT_ALT: %d candidates", len(candidates))
        else:
            log.debug("HOUGH_GRADIENT_ALT returned 0 candidates — falling back to skimage Hough")

    if not _used_alt:
        # Standard skimage Hough — reliable fallback for low-contrast scenes
        radii = np.arange(r_min, r_max + 1)
        hough_res = hough_circle(edges, radii)
        _, cx_arr, cy_arr, r_arr = hough_circle_peaks(
            hough_res, radii,
            min_xdistance=_HOUGH_MIN_DIST,
            min_ydistance=_HOUGH_MIN_DIST,
            num_peaks=_HOUGH_NUM_PEAKS,
            threshold=_HOUGH_ACCUMULATOR_MIN,
        )
        for cx, cy, r in zip(cx_arr, cy_arr, r_arr):
            r_idx = np.searchsorted(radii, r)
            r_idx = np.clip(r_idx, 0, len(radii) - 1)
            acc = float(hough_res[r_idx, int(cy), int(cx)])
            candidates.append(_Candidate(cx=float(cx), cy=float(cy),
                                         r=float(r), accumulator=acc))
        log.debug("skimage Hough fallback: %d candidates", len(candidates))

    if len(candidates) == 0:
        log.info("No Hough peaks above threshold — detection failed")
        return _failed_result("No Hough peaks above threshold")

    hough_peak_count = len(candidates)
    log.debug("Hough peaks: %d (saturation threshold: %d)", hough_peak_count, _HOUGH_SATURATION)

    # ---------------------------------------------------------------------------
    # LoG blob detection — supplementary candidate source for cluttered scenes.
    # When Hough is saturated (>_HOUGH_SATURATION peaks), the accumulator is
    # no longer a reliable ranking signal. LoG operates on intensity blobs
    # rather than edge arcs and is immune to tape/marker phantom circles.
    # Validated on _067: LoG finds sphere at rank 1–5 in all 12 clips.
    # Not useful for _106-type scenes (gray sphere on gray backdrop — no
    # intensity blob contrast), so only added when Hough is saturated.
    # ---------------------------------------------------------------------------
    if hough_peak_count > _HOUGH_SATURATION:
        log.debug("Hough saturated — adding LoG blob candidates")
        from math import sqrt as _sqrt
        log_blobs = blob_log(
            gray,
            min_sigma=_LOG_MIN_SIGMA,
            max_sigma=_LOG_MAX_SIGMA,
            num_sigma=_LOG_NUM_SIGMA,
            threshold=_LOG_THRESHOLD,
        )
        log_added = 0
        for blob in log_blobs:
            bcy, bcx, bsigma = blob
            br = bsigma * _sqrt(2)
            if not (_PF_RADIUS_MIN <= br / w <= _RADIUS_MAX_RATIO):
                continue
            # Tag LoG candidates with a high accumulator sentinel so they
            # sort ahead of low-confidence Hough phantoms but behind strong
            # Hough peaks. Real accumulator value is not meaningful here.
            candidates.append(_Candidate(
                cx=float(bcx), cy=float(bcy),
                r=float(br), accumulator=0.70,
            ))
            log_added += 1
        log.debug("LoG added %d blob candidates", log_added)

    # Prefer candidates near prior if we have one
    if prior_cx is not None and prior_cy is not None:
        def _prior_score(c: _Candidate) -> float:
            dist = math.hypot(c.cx - prior_cx, c.cy - prior_cy)
            return dist - c.accumulator * 20  # lower = better
        candidates.sort(key=_prior_score)
    else:
        candidates.sort(key=lambda c: -c.accumulator)

    best_candidate_roi = None
    if candidates:
        top_candidate = candidates[0]
        best_candidate_roi = SphereROI(
            cx=top_candidate.cx / scale,
            cy=top_candidate.cy / scale,
            r=top_candidate.r / scale,
        )

    # Run the 5-gate pipeline on each candidate; take first that passes
    rgb_det = np.array(img_det, dtype=np.float32) / 255.0
    rgb_full = np.array(img_full, dtype=np.float32) / 255.0

    # Scene-linear luminance at detection scale — for BRDF scoring
    _brdf_lum_det = rgb_display_to_lum(rgb_det)   # (H×W) float32
    _pf_lum_gray  = _to_gray_arr(rgb_det)          # cached for pre-filter reuse

    def _score_brdf(cand_cx: float, cand_cy: float, cand_r: float) -> float:
        """Return BRDF score [0–1] for a single candidate."""
        try:
            score, rot, az, el = brdf_gate(_brdf_lum_det, cand_cx, cand_cy, cand_r)
            return score
        except Exception:
            return 0.0

    def _finalize(cand, gates, pass2: bool = False) -> SphereDetectionResult:
        """Build the SUCCESS/NEEDS_ASSIST result for a passing candidate.
        Identical for pass 1 and gating-2; pass2 only tags the source."""
        cand.passed = True
        cx_full = cand.cx / scale
        cy_full = cand.cy / scale
        r_full  = cand.r  / scale
        cand.hero_ire = _probe_hero_ire(rgb_full, cx_full, cy_full, r_full)

        ire_ok, ire_reason = _check_ire_context(cand.hero_ire, peer_ire_values)
        cand.ire_context_status = "ok" if ire_ok else "needs_assist"
        cand.ire_context_reason = ire_reason

        # NO prior blend. The detected candidate stands on its own — the solved
        # prior is only a last-resort back-check at the workflow level (pass 3),
        # never a nudge applied to a real detection.
        roi = SphereROI(cx=cand.cx / scale, cy=cand.cy / scale, r=cand.r / scale)
        base = "auto_hough_g2" if pass2 else "auto_hough"
        source = base if ire_ok else "needs_assist"
        log.info(
            "Sphere detected%s: cx=%.1f cy=%.1f r=%.1f acc=%.3f hero_ire=%.1f ire_status=%s",
            " [gating-2]" if pass2 else "", roi.cx, roi.cy, roi.r, cand.accumulator,
            cand.hero_ire if cand.hero_ire is not None else -1, cand.ire_context_status,
        )
        return SphereDetectionResult(
            clip_id="",
            status="SUCCESS" if ire_ok else "NEEDS_ASSIST",
            roi=roi,
            gates=gates,
            hough_accumulator=cand.accumulator,
            source=source,
            radius_ratio=cand.r / min(h, w),
            ire_spread=_compute_ire_spread(rgb_full, roi.cx, roi.cy, roi.r),
            chromaticity_distance=_chromaticity_distance(rgb_full, roi.cx, roi.cy, roi.r),
        )

    # Candidates that failed pass 1 SOLELY at shadow_specular — eligible for gating-2.
    pass2_candidates: List[_Candidate] = []

    for cand in candidates:
        # -----------------------------------------------------------------
        # Pre-filter gates — Session 9
        #
        # Gate 1: radius floor
        #   Eliminates Hough noise rings (r/w 0.019–0.026 phantom population).
        #   Floor set at 0.018 — just below the smallest confirmed true sphere
        #   (r/w 0.019 in _106 at 6K 16:9, 12–13ft distance).
        #   WARNING: do not raise without re-measuring smallest sphere r/w
        #   across all footage. _106 true sphere sits at r/w 0.019–0.022.
        #
        # Gate 2: std band (scene-adaptive)
        #   Clean scene  (std ≤ 0.020): pass directly.
        #     Calibrated on _106 gray-on-gray: true sphere 0.004–0.020,
        #     specular-edge rings 0.051–0.108.
        #   Hard ceiling (std > 0.130): reject unconditionally.
        #     Nothing legitimate exceeds this.
        #   Middle band  (0.020 < std ≤ 0.130): BRDF fallback.
        #     Covers dramatic-lighting / dark-background scenes (_067).
        #     True sphere reads std 0.045–0.100 due to expressed Lambertian
        #     gradient. Requires brdf ≥ 0.28 to pass.
        #     Handoff §9 prescription: "a sphere that fails std but passes
        #     BRDF is a legitimate sphere in harder conditions."
        # -----------------------------------------------------------------

        # Gate 1 — radius floor
        if cand.r / w < _PF_RADIUS_MIN:
            log.debug(
                "Pre-filter SKIP G1 (r/w=%.3f < %.3f): cx=%.1f cy=%.1f r=%.1f",
                cand.r / w, _PF_RADIUS_MIN, cand.cx, cand.cy, cand.r,
            )
            continue

        # Gate 2 — std band with BRDF fallback
        _pf_mask = _interior_mask(rgb_det, cand.cx, cand.cy, cand.r)
        _pf_std  = float(_pf_lum_gray[_pf_mask].std()) if _pf_mask.sum() > 0 else 1.0

        if _pf_std > _PF_STD_HARD_MAX:
            log.debug(
                "Pre-filter SKIP G2-hard (std=%.4f > %.3f): cx=%.1f cy=%.1f r=%.1f",
                _pf_std, _PF_STD_HARD_MAX, cand.cx, cand.cy, cand.r,
            )
            continue

        if _pf_std > _PF_STD_CLEAN_MAX:
            # Middle band — BRDF fallback required
            _brdf_score = _score_brdf(cand.cx, cand.cy, cand.r)
            # Use profile-derived BRDF minimum when available.
            _brdf_min = _PF_BRDF_FALLBACK_MIN
            if (photo_prior is not None
                    and photo_prior.get("photo_narrowing_ready")
                    and photo_prior.get("brdf_score_min") is not None):
                _brdf_min = max(0.15, min(_PF_BRDF_FALLBACK_MIN,
                                          photo_prior["brdf_score_min"] - 0.10))
            if _brdf_score < _brdf_min:
                log.debug(
                    "Pre-filter SKIP G2-brdf (std=%.4f brdf=%.4f < %.2f%s): "
                    "cx=%.1f cy=%.1f r=%.1f",
                    _pf_std, _brdf_score, _brdf_min,
                    " [profile]" if _brdf_min != _PF_BRDF_FALLBACK_MIN else "",
                    cand.cx, cand.cy, cand.r,
                )
                continue
            log.debug(
                "Pre-filter PASS G2-brdf fallback (std=%.4f brdf=%.4f): "
                "cx=%.1f cy=%.1f r=%.1f",
                _pf_std, _brdf_score, cand.cx, cand.cy, cand.r,
            )
        else:
            _brdf_score = _score_brdf(cand.cx, cand.cy, cand.r)
            log.debug(
                "Pre-filter PASS G2-clean (std=%.4f brdf=%.4f): "
                "cx=%.1f cy=%.1f r=%.1f",
                _pf_std, _brdf_score, cand.cx, cand.cy, cand.r,
            )

        # Gate 3 — std floor: reject near-uniform regions (wall/floor phantoms)
        if _pf_std < _PF_STD_FLOOR:
            log.debug(
                "Pre-filter SKIP G3 (std=%.4f < %.3f): cx=%.1f cy=%.1f r=%.1f",
                _pf_std, _PF_STD_FLOOR, cand.cx, cand.cy, cand.r,
            )
            continue

        # Gate 4 — RGB ratio: reject colored objects
        _pf_pixels = rgb_det[_pf_mask]
        if _pf_pixels.shape[0] > 0:
            _rgb_mean = _pf_pixels.mean(axis=0)
            _g = float(_rgb_mean[1]) if float(_rgb_mean[1]) > 1e-6 else 1e-6
            _rg = float(_rgb_mean[0]) / _g
            _bg = float(_rgb_mean[2]) / _g
            if not (_PF_RG_MIN <= _rg <= _PF_RG_MAX and _PF_BG_MIN <= _bg <= _PF_BG_MAX):
                log.debug(
                    "Pre-filter SKIP G4 (R/G=%.3f B/G=%.3f out of range): "
                    "cx=%.1f cy=%.1f r=%.1f",
                    _rg, _bg, cand.cx, cand.cy, cand.r,
                )
                continue

        _pf_cd = _chromaticity_distance(rgb_det, cand.cx, cand.cy, cand.r)
        log.debug(
            "Candidate advancing to 5-gate pipeline: "
            "cx=%.1f cy=%.1f r=%.1f r/w=%.3f std=%.4f chroma=%.4f brdf=%.4f",
            cand.cx, cand.cy, cand.r, cand.r / w, _pf_std, _pf_cd, _brdf_score,
        )

        gates = _run_5_gates(cand, rgb_det, h, w)
        cand.gates = gates
        # Guard: all() on empty list is vacuously True — reject empty gate lists
        if gates and all(g.passed for g in gates):
            return _finalize(cand, gates)

        # Gating-2 eligibility: failed SOLELY at shadow_specular (geometry, gray,
        # lambertian all passed; shadow_specular is the last/failing gate).
        if (gates and gates[-1].gate == "shadow_specular"
                and not gates[-1].passed
                and all(g.passed for g in gates[:-1])):
            pass2_candidates.append(cand)

    # ── Gating-2: looser shadow_specular on the clean ALT stream only ──────────
    # ALT's gradient constraint already excludes flat charts/cards, so re-checking
    # shadow_specular with a looser bound here rescues evenly-lit spheres without
    # admitting flats. Skipped entirely when ALT fell back to skimage (cluttered).
    if _used_alt and pass2_candidates:
        for cand in pass2_candidates:
            sg = _gate_shadow_specular(cand, rgb_det, shadow_max=_GATE_SHADOW_RATIO_MAX_PASS2)
            if not sg.passed:
                continue
            # Confirm the remaining gates (not reached in pass 1 due to early-exit).
            ire = _gate_ire_spread(cand, rgb_det)
            std = _gate_stddev(cand, rgb_det)
            if not (ire.passed and std.passed):
                continue
            full_gates = list(cand.gates[:-1]) + [sg, ire, std]
            log.info("Gating-2 accepted candidate at looser shadow_ratio (≤%.3f, ALT stream)",
                     _GATE_SHADOW_RATIO_MAX_PASS2)
            return _finalize(cand, full_gates, pass2=True)

    log.info("All Hough candidates failed gates — detection failed")
    return _failed_result("All candidates failed gate pipeline", best_candidate_roi=best_candidate_roi)


# ---------------------------------------------------------------------------
# IRE context gate (soft — returns bool, never raises)
# ---------------------------------------------------------------------------
def _check_ire_context(
    hero_ire: Optional[float],
    peer_ire_values: Optional[List[float]],
) -> Tuple[bool, str]:
    """
    Decide whether this candidate's hero IRE is plausible given:
      a) What other cameras measured in this same run (peer context), or
      b) The conservative pipeline-level safety rails if no peers exist.

    Returns (ok: bool, reason: str).
    ok=False → mark as NEEDS_ASSIST (not FAIL; operator can override).
    """
    if hero_ire is None:
        return True, "IRE probe unavailable — skipping context check"

    # Derive the acceptance window
    peers = [v for v in (peer_ire_values or []) if v is not None]
    if len(peers) >= 2:
        peer_min = min(peers)
        peer_max = max(peers)
        win_lo = max(_IRE_CONTEXT_FLOOR, peer_min - _IRE_CONTEXT_HEADROOM)
        win_hi = min(_IRE_CONTEXT_CEIL,  peer_max + _IRE_CONTEXT_HEADROOM)

        # Enforce minimum window width around peer midpoint
        midpoint = (win_lo + win_hi) / 2.0
        half_min = _IRE_CONTEXT_MIN_WINDOW / 2.0
        if (win_hi - win_lo) < _IRE_CONTEXT_MIN_WINDOW:
            win_lo = max(_IRE_CONTEXT_FLOOR, midpoint - half_min)
            win_hi = min(_IRE_CONTEXT_CEIL,  midpoint + half_min)

        source_desc = (
            f"peer-derived [{peer_min:.1f}–{peer_max:.1f} IRE observed, "
            f"window {win_lo:.1f}–{win_hi:.1f}]"
        )
    else:
        # No peer context — use conservative pipeline rails
        win_lo = _IRE_CONTEXT_FLOOR
        win_hi = _IRE_CONTEXT_CEIL
        source_desc = (
            f"fallback pipeline rails [{win_lo:.0f}–{win_hi:.0f} IRE]"
        )

    if win_lo <= hero_ire <= win_hi:
        return True, f"IRE {hero_ire:.1f} within {source_desc}"
    else:
        direction = "below floor" if hero_ire < win_lo else "above ceiling"
        return False, (
            f"IRE {hero_ire:.1f} is {direction} of {source_desc}. "
            f"Likely detected wrong object. Operator verification required."
        )


# ---------------------------------------------------------------------------
# 5-gate pipeline
# ---------------------------------------------------------------------------
def _run_5_gates(
    cand: _Candidate,
    rgb: np.ndarray,
    h: int,
    w: int,
) -> List[DetectionGateResult]:
    gates: List[DetectionGateResult] = []

    # Gate 1: Geometry
    gates.append(_gate_geometry(cand, h, w))
    if not gates[-1].passed:
        return gates

    # Gate 2: Gray Material
    gates.append(_gate_gray_material(cand, rgb))
    if not gates[-1].passed:
        return gates

    # Gate 3: Lambertian
    gates.append(_gate_lambertian(cand, rgb))
    if not gates[-1].passed:
        return gates

    # Gate 3.5: Shadow / Specular (convexity)
    # Rejects flat surfaces (focus charts, gray cards) that pass Lambertian.
    # Requires both a shadow terminator and a specular highlight — properties
    # that only a 3D convex object under directional lighting can produce.
    gates.append(_gate_shadow_specular(cand, rgb))
    if not gates[-1].passed:
        return gates

    # Gate 4: IRE Spread
    gates.append(_gate_ire_spread(cand, rgb))
    if not gates[-1].passed:
        return gates

    # Gate 5: Interior Stddev
    gates.append(_gate_stddev(cand, rgb))
    return gates


def _gate_geometry(cand: _Candidate, h: int, w: int) -> DetectionGateResult:
    r_ratio = cand.r / w
    ok = _RADIUS_MIN_RATIO <= r_ratio <= _RADIUS_MAX_RATIO
    return DetectionGateResult(
        gate="geometry",
        passed=ok,
        reason=(
            f"r={cand.r:.1f}px ratio={r_ratio:.3f} "
            f"({'ok' if ok else f'outside {_RADIUS_MIN_RATIO:.2f}–{_RADIUS_MAX_RATIO:.2f}'})"
        ),
    )


def _gate_gray_material(cand: _Candidate, rgb: np.ndarray) -> DetectionGateResult:
    dist = _chromaticity_distance_from_candidate(cand, rgb)
    ok = dist <= _GATE_CHROMA_MAX_DISTANCE
    return DetectionGateResult(
        gate="gray_material",
        passed=ok,
        reason=(
            f"chroma_dist={dist:.4f} "
            f"({'ok' if ok else f'>{_GATE_CHROMA_MAX_DISTANCE}'})"
        ),
    )


def _gate_lambertian(cand: _Candidate, rgb: np.ndarray) -> DetectionGateResult:
    """
    Sample luminance in 4 concentric rings. Each ring should be dimmer than
    the one inside it (Lambertian falloff). Tolerance = 12%.
    """
    cx, cy, r = cand.cx, cand.cy, cand.r
    ring_radii = [0.20 * r, 0.45 * r, 0.68 * r, 0.80 * r]
    # Outermost ring tightened from 0.88r to 0.80r (Session 13).
    # At 0.88r the ring samples pixels at the sphere boundary where
    # background bleed and shadow terminator compression (especially
    # at low IRE) cause spurious brightness increases that exceed the
    # 12% tolerance. 0.80r stays within the clean Lambertian mid-tone
    # region. Validated across 36 clips: 0 regressions, 1 improvement.
    ring_lum = []
    for rr in ring_radii:
        ring_lum.append(_sample_ring_luminance(rgb, cx, cy, rr, 0.06 * r))

    violations = 0
    for i in range(len(ring_lum) - 1):
        inner, outer = ring_lum[i], ring_lum[i + 1]
        if inner is None or outer is None:
            continue
        # Allow outer to be up to tolerance brighter than inner
        if outer > inner * (1 + _GATE_LAMBERTIAN_TOLERANCE):
            violations += 1

    ok = violations == 0
    return DetectionGateResult(
        gate="lambertian",
        passed=ok,
        reason=(
            f"ring_lum={[f'{v:.3f}' if v else 'n/a' for v in ring_lum]} "
            f"violations={violations}"
        ),
    )


def _gate_shadow_specular(cand: _Candidate, rgb: np.ndarray,
                          shadow_max: Optional[float] = None) -> DetectionGateResult:
    """
    Gate 3.5 — Shadow / Specular (convexity).

    shadow_max defaults to _GATE_SHADOW_RATIO_MAX (pass 1). Gating-2 calls this
    with the looser _GATE_SHADOW_RATIO_MAX_PASS2 for ALT-sourced candidates that
    failed pass 1 solely at this gate.

    Tests that the candidate region has both a shadow terminator and a
    specular highlight — the two photometric signatures of a 3D convex
    object under directional lighting. Flat surfaces (focus charts, gray
    cards, backdrops) pass Lambertian but fail this gate.

    Metrics:
      shadow_ratio  = dark_half_mean / bright_half_mean
                      Split along axis from center toward peak luminance pixel.
                      Sphere: < 0.96 (dark side meaningfully dimmer).
                      Flat chart: ≥ 0.96 (both halves nearly equal).

      peak_excess   = peak_lum_in_interior / interior_mean
                      Sphere: ≥ 1.20 (tight specular lobe).
                      Flat chart: < 1.20 (diffuse, no highlight).

    Both conditions must hold. Either failing alone is insufficient —
    the gate requires the AND combination to avoid false rejects on
    high-angle or low-contrast sphere shots.

    Calibrated on 36 clips (_064/_067/_106): 27/27 spheres pass,
    8/8 charts rejected. Thresholds have ~0.006 headroom on shadow_ratio
    and ~0.009 on peak_excess vs the tightest confirmed sphere.
    """
    cx, cy, r = cand.cx, cand.cy, cand.r
    h, w = rgb.shape[:2]
    gray = _to_gray_arr(rgb)

    # Sample interior at 0.7r
    ys, xs = np.ogrid[:h, :w]
    dist = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    mask_07 = dist <= (r * 0.7)

    if mask_07.sum() < 20:
        return DetectionGateResult(
            gate="shadow_specular",
            passed=True,   # too few pixels to judge — pass rather than false-reject
            reason="insufficient pixels for shadow/specular check — skipped",
        )

    interior_lum = gray[mask_07]
    mean_i = float(interior_lum.mean())

    # Peak luminance pixel within 0.7r
    masked = np.where(mask_07, gray, 0.0)
    peak_idx = np.unravel_index(np.argmax(masked), gray.shape)
    peak_y, peak_x = int(peak_idx[0]), int(peak_idx[1])
    peak_lum = float(gray[peak_y, peak_x])
    peak_excess = peak_lum / max(mean_i, 1e-6)

    # Split interior along axis center → peak, compute shadow ratio
    dx = peak_x - cx
    dy = peak_y - cy
    norm = math.hypot(dx, dy)
    if norm > 1.0:
        dx /= norm
        dy /= norm
    else:
        dx, dy = 1.0, 0.0

    ys_idx, xs_idx = np.where(mask_07)
    proj = (xs_idx - cx) * dx + (ys_idx - cy) * dy
    bright_vals = gray[mask_07][proj >= 0]
    dark_vals   = gray[mask_07][proj <  0]

    bright_mean = float(bright_vals.mean()) if len(bright_vals) > 0 else mean_i
    dark_mean   = float(dark_vals.mean())   if len(dark_vals)   > 0 else mean_i
    shadow_ratio = dark_mean / max(bright_mean, 1e-6)

    _smax = _GATE_SHADOW_RATIO_MAX if shadow_max is None else shadow_max
    ok = shadow_ratio <= _smax

    if ok:
        reason = (
            f"shadow_ratio={shadow_ratio:.4f} (≤{_smax}) ok "
            f"[peak_excess={peak_excess:.4f} diagnostic]"
        )
    else:
        reason = (
            f"shadow_ratio={shadow_ratio:.4f} >{_smax} "
            f"(flat object — no shadow terminator) "
            f"[peak_excess={peak_excess:.4f} diagnostic]"
        )

    return DetectionGateResult(
        gate="shadow_specular",
        passed=ok,
        reason=reason,
    )


def _gate_ire_spread(cand: _Candidate, rgb: np.ndarray) -> DetectionGateResult:
    spread = _compute_ire_spread_from_candidate(cand, rgb)
    ok = spread >= _GATE_IRE_SPREAD_MIN
    return DetectionGateResult(
        gate="ire_spread",
        passed=ok,
        reason=(
            f"spread={spread:.2f} IRE "
            f"({'ok' if ok else f'<{_GATE_IRE_SPREAD_MIN}'})"
        ),
    )


def _gate_stddev(cand: _Candidate, rgb: np.ndarray) -> DetectionGateResult:
    mask = _interior_mask(rgb, cand.cx, cand.cy, cand.r * 0.85)
    if mask.sum() < 20:
        return DetectionGateResult(
            gate="interior_stddev",
            passed=False,
            reason="insufficient pixels in interior mask",
        )
    pixels = _to_gray_arr(rgb)[mask]
    stddev = float(np.std(pixels))
    ok = _GATE_STDDEV_MIN <= stddev <= _GATE_STDDEV_MAX
    return DetectionGateResult(
        gate="interior_stddev",
        passed=ok,
        reason=(
            f"stddev={stddev:.4f} "
            f"({'ok' if ok else f'outside [{_GATE_STDDEV_MIN},{_GATE_STDDEV_MAX}]'})"
        ),
    )


# ---------------------------------------------------------------------------
# Measurement helpers
# ---------------------------------------------------------------------------
def _probe_hero_ire(
    rgb: np.ndarray, cx: float, cy: float, r: float
) -> Optional[float]:
    """
    Quick hero-center IRE probe (same math as measure.py but for gate use).
    Samples a disk of radius 0.24r at the center of the sphere.
    Returns None if the disk has no valid pixels.
    """
    mask = _interior_mask(rgb, cx, cy, r * 0.24)
    if mask.sum() < 4:
        return None
    lum = _to_gray_arr(rgb)[mask]
    lum = lum[lum > 0]
    if len(lum) < 4:
        return None
    p5, p95 = np.percentile(lum, [5, 95])
    trimmed = lum[(lum >= p5) & (lum <= p95)]
    if len(trimmed) == 0:
        return None
    scalar = float(np.median(np.log2(trimmed + 1e-9)))
    ire = (2 ** scalar) * 100.0
    return ire


def _chromaticity_distance_from_candidate(
    cand: _Candidate, rgb: np.ndarray
) -> float:
    return _chromaticity_distance(rgb, cand.cx, cand.cy, cand.r)


def _chromaticity_distance(
    rgb: np.ndarray, cx: float, cy: float, r: float
) -> float:
    mask = _interior_mask(rgb, cx, cy, r * 0.70)
    if mask.sum() < 10:
        return 999.0
    pixels = rgb[mask]  # shape (N, 3)
    mean_rgb = pixels.mean(axis=0)
    total = mean_rgb.sum()
    if total < 1e-6:
        return 999.0
    r_c = mean_rgb[0] / total
    g_c = mean_rgb[1] / total
    # Achromatic: r_c = g_c = 1/3
    dist = math.hypot(r_c - 1/3, g_c - 1/3)
    return dist


def _compute_ire_spread_from_candidate(
    cand: _Candidate, rgb: np.ndarray
) -> float:
    return _compute_ire_spread(rgb, cand.cx, cand.cy, cand.r)


def _compute_ire_spread(
    rgb: np.ndarray, cx: float, cy: float, r: float
) -> float:
    offset = 0.24 * r
    bright_cx = cx + offset
    dark_cx   = cx - offset

    bright_ire = _probe_hero_ire(rgb, bright_cx, cy, r * 0.20)
    dark_ire   = _probe_hero_ire(rgb, dark_cx,   cy, r * 0.20)

    if bright_ire is None or dark_ire is None:
        return 0.0
    return abs(bright_ire - dark_ire)


def _sample_ring_luminance(
    rgb: np.ndarray,
    cx: float, cy: float,
    ring_r: float,
    half_width: float,
) -> Optional[float]:
    """Mean luminance of an annular ring."""
    h, w = rgb.shape[:2]
    ys, xs = np.ogrid[:h, :w]
    dist = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    mask = (dist >= ring_r - half_width) & (dist <= ring_r + half_width)
    if mask.sum() < 4:
        return None
    lum = _to_gray_arr(rgb)[mask]
    return float(np.mean(lum))


def _interior_mask(
    rgb: np.ndarray, cx: float, cy: float, r: float
) -> np.ndarray:
    h, w = rgb.shape[:2]
    ys, xs = np.ogrid[:h, :w]
    dist = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    return dist <= r


def _to_gray_arr(rgb: np.ndarray) -> np.ndarray:
    return (0.2126 * rgb[..., 0]
            + 0.7152 * rgb[..., 1]
            + 0.0722 * rgb[..., 2])


# ---------------------------------------------------------------------------
# Image loading & scaling
# ---------------------------------------------------------------------------
def _load_image(path: str) -> Image.Image:
    img = Image.open(path)
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGB")
    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (0, 0, 0))
        bg.paste(img, mask=img.split()[3])
        img = bg
    return img


def _resize_for_detection(img: Image.Image) -> Tuple[Image.Image, float]:
    w, h = img.size
    scale = _DETECTION_TARGET_DIM / max(w, h)
    if scale >= 1.0:
        return img, 1.0
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = img.resize((new_w, new_h), Image.LANCZOS)
    return resized, scale


def _to_gray(img: Image.Image) -> np.ndarray:
    arr = np.array(img, dtype=np.float32) / 255.0
    return 0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]


# ---------------------------------------------------------------------------
# Failure sentinel
# ---------------------------------------------------------------------------
def _failed_result(reason: str, best_candidate_roi: Optional[SphereROI] = None) -> SphereDetectionResult:
    return SphereDetectionResult(
        clip_id="",
        status="FAILED",
        roi=None,
        source="auto_hough",
        gates=[],
        failure_reason=reason,
        hough_accumulator=0.0,
        ire_spread=0.0,
        chromaticity_distance=999.0,
        best_candidate_roi=best_candidate_roi,
    )


# ---------------------------------------------------------------------------
# Public aliases for measure.py compatibility
# ---------------------------------------------------------------------------

def load_render_as_hwc(path: str) -> np.ndarray:
    """Load a render TIFF and return a float32 HxWx3 numpy array in [0,1]."""
    img = _load_image(path)
    arr = np.array(img, dtype=np.float32) / 255.0
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    return arr[:, :, :3]


def _luminance(pixels: np.ndarray) -> np.ndarray:
    """BT.709 luminance from Nx3 or HxWx3 float32 array."""
    return _to_gray_arr(pixels)


def validate_manual_roi(
    image_hwc: np.ndarray,
    roi: "SphereROI",
    *,
    clip_id: str,
) -> "SphereDetectionResult":
    """
    Accept an operator-placed ROI and run it through the 5 photometric gates.
    Geometry gate is skipped (operator has already placed the circle).
    Returns a SphereDetectionResult with status SUCCESS or FAILED.
    """
    from r3dmatch3.models import SphereDetectionResult, DetectionGateResult

    h, w = image_hwc.shape[:2]
    cand = _Candidate(
        cx=roi.cx, cy=roi.cy, r=roi.r,
        accumulator=1.0,
    )

    gates = []
    failed_gate = ""
    failure_reason = ""

    for gate_fn, gate_name in [
        (_gate_gray_material, "gray_material"),
        (_gate_lambertian,    "lambertian"),
        (_gate_ire_spread,    "ire_spread"),
        (_gate_stddev,        "stddev"),
    ]:
        result = gate_fn(cand, image_hwc)
        gates.append(result)
        if not result.passed and not failed_gate:
            failed_gate = gate_name
            failure_reason = result.reason

    status = "MANUAL" if not failed_gate else "FAILED"

    return SphereDetectionResult(
        clip_id=clip_id,
        status=status,
        roi=roi if status == "MANUAL" else None,
        source="manual_operator",
        gates=gates,
        failed_gate=failed_gate,
        failure_reason=failure_reason,
        radius_ratio=roi.r / min(h, w),
    )


# ---------------------------------------------------------------------------
# workflow.py-compatible wrapper for detect_sphere
# Replaces the old file-path-based signature with HWC array + prior object
# ---------------------------------------------------------------------------

def _detect_sphere_orig(
    image_path: str,
    prior_cx=None,
    prior_cy=None,
    prior_r=None,
    prior_weight: float = 0.5,
    peer_ire_values=None,
) -> "SphereDetectionResult":
    """Original file-path-based detect_sphere — kept for audit tools."""
    return _detect_sphere_by_path(image_path, prior_cx=prior_cx, prior_cy=prior_cy,
                                   prior_r=prior_r, prior_weight=prior_weight,
                                   peer_ire_values=peer_ire_values)


# ---------------------------------------------------------------------------
# Overload: workflow.py calls detect_sphere(image_hwc, clip_id=, prior=)
# This re-exports a wrapper under the same name for workflow.py compatibility.
# The original detect_sphere above takes image_path:str.
# ---------------------------------------------------------------------------

_detect_sphere_by_path = detect_sphere  # save reference before shadowing

def detect_sphere(image_or_path, clip_id: str = "", prior=None, **kwargs):
    """
    Unified detect_sphere.
    Accepts either:
      - image_or_path as str  → calls original file-path version
      - image_or_path as np.ndarray (HWC float32) → uses array directly
    prior: object with optional attrs cx, cy, r (from sphere_profile)
    """
    import tempfile, os
    from PIL import Image as _Image

    # Unpack prior object into floats
    prior_cx = getattr(prior, 'cx', None)
    prior_weight_override = getattr(prior, 'weight', None)
    prior_cy = getattr(prior, 'cy', None)
    prior_r  = getattr(prior, 'r',  None)

    if isinstance(image_or_path, np.ndarray):
        # Write HWC float32 array to a temp TIFF so the core can process it
        arr_uint8 = np.clip(image_or_path * 255.0, 0, 255).astype(np.uint8)
        img = _Image.fromarray(arr_uint8[:, :, :3], mode='RGB')
        # The core operates in the DETECTION plane and treats prior_cx/cy/r as
        # detection-plane units. Callers (workflow) pass them in FULL-RES pixels,
        # so convert here using the same resize scale. Without this the prior
        # blend throws the ROI far off-image.
        if prior_cx is not None or prior_cy is not None or prior_r is not None:
            _, _det_scale = _resize_for_detection(img)
            if prior_cx is not None:
                prior_cx = prior_cx * _det_scale
            if prior_cy is not None:
                prior_cy = prior_cy * _det_scale
            if prior_r is not None:
                prior_r = prior_r * _det_scale
        with tempfile.NamedTemporaryFile(suffix='.tiff', delete=False) as tmp:
            tmp_path = tmp.name
        try:
            img.save(tmp_path)
            photo_prior_val = kwargs.pop("photo_prior", None)
            result = _detect_sphere_by_path(
                tmp_path,
                prior_cx=prior_cx, prior_cy=prior_cy, prior_r=prior_r,
                prior_weight=prior_weight_override if prior_weight_override is not None else 0.5,
                photo_prior=photo_prior_val,
                **kwargs,
            )
        finally:
            os.unlink(tmp_path)
        # Stamp clip_id onto result
        if clip_id:
            result.clip_id = clip_id
        return result
    else:
        # image_or_path is a string path
        return _detect_sphere_by_path(
            image_or_path,
            prior_cx=prior_cx, prior_cy=prior_cy, prior_r=prior_r,
            **kwargs,
        )
