"""
brdf_gate.py — BRDF radial profile gate for R3DMatch v2.

Extracted from brdf_verify.py (Session 8 reference implementation).
Provides a single public entry point used by sphere.py and brdf_verify.py.

Algorithm (Section 9B of handoff):
  1. BT.1886 gamma inversion (display → scene-linear)
  2. Specular peak detection (brightest connected region inside 0.85r)
  3. Light direction inference from peak offset
  4. Rotation solve — tries 0/90/180/270° of light vector, keeps highest score
     (KOMODO-X has no usable IMU rotation metadata)
  5. Radial spoke scoring — 16 spokes × 24 points, Pearson correlation
  6. Trimmed mean — drop 2 lowest spoke scores before averaging

Public interface:
    from r3dmatch3.brdf_gate import brdf_gate
    score, rotation_deg, light_az, light_el = brdf_gate(lum, cx, cy, r)

    lum  : (H, W) float32 array — BT.1886-inverted scene-linear luminance
    cx, cy, r : float — sphere centre and radius in the same pixel space as lum
    Returns:
        score       float  0–1   (mean Pearson r; higher = better Lambertian match)
        rotation_deg int   0/90/180/270  (solved sensor rotation)
        light_az    float  degrees (inferred key light azimuth)
        light_el    float  degrees (inferred key light elevation)

Score thresholds (provisional — Section 9B):
    BRDF_SCORE_GOOD     = 0.70   → strong Lambertian match
    BRDF_SCORE_MARGINAL = 0.40   → weak but present
    below MARGINAL               → likely not a sphere
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, label

# ---------------------------------------------------------------------------
# Provisional thresholds — calibrate per footage if needed
# ---------------------------------------------------------------------------
BRDF_SCORE_GOOD     = 0.70
BRDF_SCORE_MARGINAL = 0.40

# Number of spokes / sample points (match brdf_verify.py)
_N_SPOKES  = 16
_N_POINTS  = 24
# Number of worst-scoring spokes to trim before averaging
_TRIM_WORST = 2


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _disk_mask(
    h: int, w: int, cx: float, cy: float, r: float, scale: float = 1.0
) -> np.ndarray:
    ys, xs = np.ogrid[:h, :w]
    return (xs - cx) ** 2 + (ys - cy) ** 2 <= (r * scale) ** 2


# ---------------------------------------------------------------------------
# Display-transform inversion
# ---------------------------------------------------------------------------

def invert_bt1886(rgb: np.ndarray) -> np.ndarray:
    """
    BT.1886 gamma inversion: display = linear^(1/2.4) → inverse = display^2.4.
    Input/output: float32 array in [0, 1].
    """
    return np.power(np.clip(rgb, 0.0, 1.0), 2.4).astype(np.float32)


def to_lum(rgb: np.ndarray) -> np.ndarray:
    """BT.709 luminance from (H, W, 3) float32 array → (H, W) float32."""
    return (0.2126 * rgb[:, :, 0]
            + 0.7152 * rgb[:, :, 1]
            + 0.0722 * rgb[:, :, 2])


def rgb_display_to_lum(rgb_display: np.ndarray) -> np.ndarray:
    """
    Convenience: BT.1886 inversion then luminance in one call.
    Input: (H, W, 3) float32 display-space array in [0, 1].
    Output: (H, W) float32 scene-linear luminance.
    """
    return to_lum(invert_bt1886(rgb_display))


# ---------------------------------------------------------------------------
# Specular peak detection
# ---------------------------------------------------------------------------

def _find_specular_peak(
    lum: np.ndarray, cx: float, cy: float, r: float
) -> tuple[float, float]:
    """
    Find the centroid of the brightest connected region inside 0.85r.
    Falls back to (cx, cy) if no region is found.
    """
    h, w = lum.shape
    mask = _disk_mask(h, w, cx, cy, r, scale=0.85)
    interior = lum.copy()
    interior[~mask] = 0.0
    smoothed = gaussian_filter(interior, sigma=3.0)
    smoothed[~mask] = 0.0
    threshold = np.percentile(smoothed[mask], 97)
    bright = (smoothed >= threshold) & mask
    labeled, n = label(bright)
    if n == 0:
        return float(cx), float(cy)
    sizes = [(labeled == i).sum() for i in range(1, n + 1)]
    best = int(np.argmax(sizes)) + 1
    ys, xs = np.where(labeled == best)
    return float(xs.mean()), float(ys.mean())


# ---------------------------------------------------------------------------
# Light direction inference
# ---------------------------------------------------------------------------

def _infer_light_direction(
    cx: float, cy: float, r: float, peak_x: float, peak_y: float
) -> tuple[np.ndarray, float, float]:
    """
    Infer a unit light vector from the specular peak offset.
    Returns (light_vec [lx, ly_3d, lz], azimuth_deg, elevation_deg).
    Note: ly_3d = -dy_image so the vector is in standard 3-D orientation
    (+y upward), consistent with _lambertian_spoke_profile.
    """
    dx = (peak_x - cx) / r
    dy = (peak_y - cy) / r   # image-y convention (+y downward)
    mag = np.sqrt(dx ** 2 + dy ** 2)
    if mag > 0.95:
        dx, dy = dx * 0.95 / mag, dy * 0.95 / mag
    lz = float(np.sqrt(max(0.0, 1.0 - dx ** 2 - dy ** 2)))
    azimuth_deg   = float(np.degrees(np.arctan2(-dy, dx)))
    elevation_deg = float(np.degrees(np.arcsin(lz)))
    return np.array([dx, -dy, lz], dtype=np.float64), azimuth_deg, elevation_deg


# ---------------------------------------------------------------------------
# Radial spoke scoring
# ---------------------------------------------------------------------------

def _sample_spoke(
    lum: np.ndarray,
    cx: float, cy: float, r: float,
    angle_rad: float,
    n_points: int = _N_POINTS,
    max_t: float = 0.92,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample actual luminance along one radial spoke."""
    ts = np.linspace(0.05, max_t, n_points)
    vals = np.full(n_points, np.nan)
    h, w = lum.shape
    for i, t in enumerate(ts):
        px = cx + t * r * np.cos(angle_rad)
        py = cy + t * r * np.sin(angle_rad)
        xi, yi = int(round(px)), int(round(py))
        if 0 <= xi < w and 0 <= yi < h:
            vals[i] = lum[yi, xi]
    return ts, vals


def _lambertian_spoke_profile(
    ts: np.ndarray, angle_rad: float, light_vec: np.ndarray
) -> np.ndarray:
    """
    Predicted Lambertian luminance profile along one radial spoke.
    light_vec is [lx, ly_3d, lz] (ly_3d = -dy_image).
    The dot product is computed in image-y space.
    """
    lx, ly_3d, lz = light_vec
    ly = -ly_3d   # back to image-y convention
    predicted = np.zeros(len(ts))
    for i, t in enumerate(ts):
        t = float(np.clip(t, 0.0, 1.0))
        nx = t * np.cos(angle_rad)
        ny = t * np.sin(angle_rad)
        nz = float(np.sqrt(max(0.0, 1.0 - t ** 2)))
        dot = nx * lx + ny * ly + nz * lz
        predicted[i] = max(0.0, dot)
    return predicted


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson r between two arrays; returns nan if degenerate."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() < 4:
        return float('nan')
    a, b = a[valid], b[valid]
    if a.std() < 1e-9 or b.std() < 1e-9:
        return float('nan')
    return float(np.corrcoef(a, b)[0, 1])


def _radial_brdf_score(
    lum: np.ndarray,
    cx: float, cy: float, r: float,
    light_vec: np.ndarray,
    n_spokes: int = _N_SPOKES,
    n_points: int = _N_POINTS,
    trim_worst: int = _TRIM_WORST,
) -> float | None:
    """
    Compute trimmed-mean Pearson r across all spokes.
    Drops `trim_worst` lowest-magnitude spokes before averaging
    (silhouette edge / specular bleed artefacts).
    Uses abs(Pearson) so shadow-side spokes (strong negative correlation)
    contribute equally to lit-side spokes (strong positive correlation).
    Returns None if fewer than (trim_worst+1) valid spokes exist.
    """
    spoke_corrs = []
    for s in range(n_spokes):
        angle = 2.0 * np.pi * s / n_spokes
        ts, actual = _sample_spoke(lum, cx, cy, r, angle, n_points=n_points)
        predicted  = _lambertian_spoke_profile(ts, angle, light_vec)
        spoke_corrs.append(_pearson(actual, predicted))

    valid = [v for v in spoke_corrs if np.isfinite(v)]
    if len(valid) <= trim_worst:
        return None
    # Sort by absolute value — drop the trim_worst lowest-magnitude spokes
    # (silhouette edge, specular bleed, genuine noise).
    # Shadow-side spokes produce strong *negative* Pearson r because
    # luminance correctly decreases toward the shadow edge but the
    # predicted Lambertian value is near-zero on that side, inverting
    # the correlation sign. Negative magnitude is just as valid as
    # positive — only a real convex Lambertian surface produces
    # high-magnitude correlation (either sign) across most spokes.
    # Flat patches, tape, and monitor bezels produce near-zero
    # correlation on most spokes regardless of sign.
    valid_sorted_by_abs = sorted(valid, key=abs)
    trimmed = valid_sorted_by_abs[trim_worst:]   # drop lowest-magnitude
    return float(np.mean([abs(v) for v in trimmed]))


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def brdf_gate(
    lum: np.ndarray,
    cx: float,
    cy: float,
    r: float,
) -> tuple[float, int, float, float]:
    """
    Compute a BRDF Lambertian confidence score for a sphere candidate.

    Parameters
    ----------
    lum : np.ndarray, shape (H, W), dtype float32
        Scene-linear luminance array (BT.1886-inverted).
        Use ``rgb_display_to_lum(rgb_float32)`` to produce this from a
        display-space TIFF that has already been loaded as float32 in [0, 1].
    cx, cy : float
        Sphere centre in pixel coordinates matching ``lum``.
    r : float
        Sphere radius in pixels.

    Returns
    -------
    score : float
        Mean abs(Pearson) correlation across radial spokes after trimming
        the ``_TRIM_WORST`` lowest-magnitude spokes. Range nominally 0–1.
        Shadow-side spokes (negative r) are treated as equally valid
        evidence of Lambertian geometry. Returns 0.0 on degenerate input.
    rotation_deg : int
        Sensor rotation solved (0 / 90 / 180 / 270 degrees).
        KOMODO-X has no reliable IMU metadata; all four are tried and the
        highest-scoring rotation is returned.
    light_az : float
        Inferred key-light azimuth in degrees (after rotation correction).
    light_el : float
        Inferred key-light elevation in degrees.
    """
    # --- specular peak → raw light direction ---
    peak_x, peak_y = _find_specular_peak(lum, cx, cy, r)
    light_vec_raw, _az_raw, elevation = _infer_light_direction(
        cx, cy, r, peak_x, peak_y
    )

    # --- rotation solve ---
    best_score: float | None = None
    best_vec   = light_vec_raw
    best_rot   = 0

    for rot_deg in [0, 90, 180, 270]:
        a = np.radians(rot_deg)
        lx0, ly0, lz0 = light_vec_raw
        lx_r =  lx0 * np.cos(a) - ly0 * np.sin(a)
        ly_r =  lx0 * np.sin(a) + ly0 * np.cos(a)
        vec_r = np.array([lx_r, ly_r, lz0], dtype=np.float64)
        s = _radial_brdf_score(lum, cx, cy, r, vec_r)
        if s is not None and (best_score is None or s > best_score):
            best_score = s
            best_vec   = vec_r
            best_rot   = rot_deg

    if best_score is None:
        return 0.0, 0, 0.0, float(elevation)

    # --- azimuth from best (rotation-corrected) light vector ---
    light_az = float(np.degrees(np.arctan2(-best_vec[1], best_vec[0])))

    return float(best_score), int(best_rot), light_az, float(elevation)
