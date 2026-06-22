"""
R3DMatch v2 — Measurement Math

The four-step chain, proven accurate and kept exactly from the original:
  1. Luminance: Y = 0.2126R + 0.7152G + 0.0722B
  2. Trim: 5th–95th percentile outlier rejection
  3. Log2: scalar = median(log2(trimmed_luminance))
  4. IRE: (2^scalar) * 100

All measurements are display-referred (IPP2 / BT.709 / BT.1886 / Medium / Medium).

Zone geometry: three rectangular bands aligned with the luminance gradient axis.
  Sample 1 (bright_side): offset = +0.24 * r from center
  Sample 2 (center):      offset = 0.0
  Sample 3 (dark_side):   offset = -0.24 * r from center
Each band: half_width = 0.34 * r, half_height = 0.11 * r
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from PIL import Image

from .models import MeasurementResult, SphereDetectionResult, SphereROI, ZoneMeasurement
from .sphere import load_render_as_hwc, _interior_mask, _luminance, _chromaticity_distance

# ---------------------------------------------------------------------------
# Zone geometry constants (matching original exactly)
# ---------------------------------------------------------------------------

_ZONE_DEFINITIONS = (
    ("bright_side", "Sample 1",  0.24),
    ("center",      "Sample 2",  0.00),
    ("dark_side",   "Sample 3", -0.24),
)
_ZONE_HALF_WIDTH_RATIO  = 0.34
_ZONE_HALF_HEIGHT_RATIO = 0.11
_TRIM_LOW  = 5.0
_TRIM_HIGH = 95.0
_INTERIOR_RADIUS_RATIO = 0.50


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def measure_render(
    render_path: str,
    roi: SphereROI,
    *,
    clip_id: str,
    detection: Optional[SphereDetectionResult] = None,
    roi_source: str = "auto_hough",
) -> MeasurementResult:
    """
    Load a REDLine-rendered TIFF and measure the sphere.

    render_path: path to .tiff.000000.tif (or .tiff)
    roi: accepted sphere ROI in full-resolution pixel coordinates
    """
    image = load_render_as_hwc(render_path)
    height, width = image.shape[:2]
    sha = _sha256_file(render_path)

    # Measure each zone
    zones = []
    for zone_key, zone_label, offset in _ZONE_DEFINITIONS:
        zm = _measure_zone(image, roi, offset=offset)
        zones.append(ZoneMeasurement(
            label=zone_key,
            sample_label=zone_label,
            ire=zm["ire"],
            log2_luminance=zm["log2"],
            rgb_mean=(zm["rgb_mean"][0], zm["rgb_mean"][1], zm["rgb_mean"][2]),
            pixel_count=zm["pixel_count"],
        ))

    zone_bright, zone_center, zone_dark = zones

    # Interior chromaticity (inner 50% of r for WB solve)
    interior_mask = _interior_mask(image, roi.cx, roi.cy, roi.r * _INTERIOR_RADIUS_RATIO)
    interior_pixels = image[interior_mask] if interior_mask.sum() > 0 else np.zeros((1, 3), dtype=np.float32)
    rgb_mean = tuple(float(v) for v in np.mean(interior_pixels, axis=0))
    rgb_sum = sum(rgb_mean)
    if rgb_sum > 1e-6:
        rgb_chroma = tuple(v / rgb_sum for v in rgb_mean)
    else:
        rgb_chroma = (1/3, 1/3, 1/3)
    chroma_dist = _chromaticity_distance(image, roi.cx, roi.cy, roi.r)

    valid = zone_center.pixel_count >= 50
    reason = "" if valid else "insufficient_pixels_in_center_zone"

    return MeasurementResult(
        clip_id=clip_id,
        render_path=render_path,
        render_sha256=sha,
        render_width=width,
        render_height=height,
        roi=roi,

        zone_bright=zone_bright,
        zone_center=zone_center,
        zone_dark=zone_dark,

        hero_ire=zone_center.ire,
        hero_log2=zone_center.log2_luminance,
        hero_rgb=zone_center.rgb_mean,
        hero_pixel_count=zone_center.pixel_count,

        measured_rgb_mean=rgb_mean,
        measured_rgb_chromaticity=rgb_chroma,
        chromaticity_distance=chroma_dist,

        measurement_valid=valid,
        validity_reason=reason,
        roi_source=roi_source,
        detection_result=detection,
    )


def _load_render_full_depth(path: str) -> np.ndarray:
    """Load a render preserving bit depth → float32 HWC RGB in [0,1].

    The main (display) measurement uses the 8-bit PIL loader, which is fine for a
    mid-tone display value. The scene-linear exposure measurement needs full
    precision because a dark sphere sits near the bottom of the linear range,
    where 8-bit quantization is coarse. Falls back to the 8-bit loader if a
    full-depth read isn't available.
    """
    try:
        import cv2
        a = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if a is None:
            raise RuntimeError("imread failed")
        if a.ndim == 2:
            a = np.stack([a, a, a], axis=-1)
        a = a[:, :, :3][:, :, ::-1].astype(np.float32)  # BGR -> RGB
        maxv = 65535.0 if float(a.max()) > 255.0 else 255.0
        return a / maxv
    except Exception:
        return load_render_as_hwc(path)


def measure_center_log2(render_path: str, roi: SphereROI) -> Optional[float]:
    """Median log2 luminance of the sphere CENTER zone for an arbitrary render.

    Given a SCENE-LINEAR render, this returns log2(scene-linear luminance) at the
    sphere center — the value the exposure solve needs (exposureAdjust is a
    scene-linear stop). Read at full bit depth for precision (esp. dark spheres).
    Returns None on any failure so exposure falls back to the display-space solve.
    """
    try:
        image = _load_render_full_depth(render_path)
        zm = _measure_zone(image, roi, offset=0.0)
        if zm["pixel_count"] < 50:
            return None
        return float(zm["log2"])
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Zone measurement
# ---------------------------------------------------------------------------

def _measure_zone(
    image: np.ndarray,
    roi: SphereROI,
    *,
    offset: float,
) -> Dict:
    """
    Measure a single rectangular zone at the given normalized offset from center.

    Returns dict with: ire, log2, rgb_mean, pixel_count
    """
    h, w = image.shape[:2]
    zone_cy = roi.cy + offset * roi.r
    zone_half_h = roi.r * _ZONE_HALF_HEIGHT_RATIO
    zone_half_w = roi.r * _ZONE_HALF_WIDTH_RATIO

    y0 = max(0, int(np.floor(zone_cy - zone_half_h)))
    y1 = min(h, int(np.ceil(zone_cy + zone_half_h)) + 1)
    x0 = max(0, int(np.floor(roi.cx - zone_half_w)))
    x1 = min(w, int(np.ceil(roi.cx + zone_half_w)) + 1)

    if y1 <= y0 or x1 <= x0:
        return {"ire": 0.0, "log2": -10.0, "rgb_mean": [0.0, 0.0, 0.0], "pixel_count": 0}

    strip = image[y0:y1, x0:x1]   # HxWx3
    pixels = strip.reshape(-1, 3)  # Nx3

    lum = _luminance(pixels)
    lum = np.clip(lum, 1e-6, None)

    if lum.size == 0:
        return {"ire": 0.0, "log2": -10.0, "rgb_mean": [0.0, 0.0, 0.0], "pixel_count": 0}

    # Step 2: Trim 5–95 percentile
    p5, p95 = np.percentile(lum, [_TRIM_LOW, _TRIM_HIGH])
    trimmed_mask = (lum >= p5) & (lum <= p95)
    trimmed = lum[trimmed_mask]
    if trimmed.size == 0:
        trimmed = lum

    # Step 3: Log2 median
    log2_val = float(np.median(np.log2(np.maximum(trimmed, 1e-6))))

    # Step 4: IRE
    ire = (2.0 ** log2_val) * 100.0

    # RGB mean (all pixels in zone, not just trimmed)
    rgb_mean = [float(np.mean(pixels[:, c])) for c in range(3)]

    return {
        "ire": ire,
        "log2": log2_val,
        "rgb_mean": rgb_mean,
        "pixel_count": int(pixels.shape[0]),
    }


# ---------------------------------------------------------------------------
# Exposure solve math
# ---------------------------------------------------------------------------

def exposure_offset_stops(measured_log2: float, target_log2: float) -> float:
    """
    Compute stops of exposure correction needed.
    Positive = camera needs more exposure.
    """
    return float(target_log2 - measured_log2)


def array_target_log2(measurements: list) -> float:
    """
    Robust median of measured_log2 values across all valid cameras.
    measurements: list of MeasurementResult
    """
    log2_values = [m.hero_log2 for m in measurements if m.measurement_valid]
    if not log2_values:
        raise ValueError("No valid measurements to compute array target")
    return float(np.median(log2_values))


def array_target_ire(measurements: list) -> float:
    """Median IRE across valid cameras."""
    return (2.0 ** array_target_log2(measurements)) * 100.0


# ---------------------------------------------------------------------------
# WB chromaticity math
# ---------------------------------------------------------------------------

def compute_wc_gm(rgb_mean: Tuple[float, float, float]) -> Tuple[float, float]:
    """
    Warm/Cool and Green/Magenta axes from normalized RGB mean.

    WC = (R - B) / (R + G + B)  — positive is warm, negative is cool
    GM = (G - 0.5*(R+B)) / (R+G+B)  — positive is green, negative is magenta

    These are the axes used by the WB solve.
    """
    r, g, b = rgb_mean
    total = r + g + b
    if total < 1e-6:
        return 0.0, 0.0
    wc = (r - b) / total
    gm = (g - 0.5 * (r + b)) / total
    return float(wc), float(gm)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()
