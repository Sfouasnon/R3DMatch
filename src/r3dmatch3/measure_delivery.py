"""
measure_delivery.py — delivery-domain hero measurement for R3DMatch v4.

The reference measurement (measure.py) is untouched and owns the solve. This
module measures the SAME sphere/patch on a render produced through the project's
delivery pipeline (transform + LUT), so the hybrid verification can score match %
in the delivered look.

It reuses measure.py's exact zone geometry and the proven 4-step chain
(luminance -> trim 5-95 -> log2 median -> IRE); the only parameter is the
output-space luminance weights (BT.709 for a Rec.709 show LUT; wide-gamut weights
for Rec.2020/P3 deliveries). White-balance axes (WC/GM) are RGB ratios and are
weight-independent, so compute_wc_gm is reused as-is.

Nothing here feeds the solve — it only produces the operator-facing delivery
measurement. The reference path and its goldens are unaffected.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from .measure import (
    _ZONE_HALF_WIDTH_RATIO,
    _ZONE_HALF_HEIGHT_RATIO,
    _TRIM_LOW,
    _TRIM_HIGH,
    _INTERIOR_RADIUS_RATIO,
    compute_wc_gm,
)
from .models import SphereROI
from .sphere import load_render_as_hwc, _interior_mask

_DEFAULT_WEIGHTS = (0.2126, 0.7152, 0.0722)


def _luminance_weighted(pixels: np.ndarray, weights: Tuple[float, float, float]) -> np.ndarray:
    wr, wg, wb = weights
    return wr * pixels[:, 0] + wg * pixels[:, 1] + wb * pixels[:, 2]


def measure_delivery_hero(
    render_path: str,
    roi: SphereROI,
    *,
    luma_weights: Tuple[float, float, float] = _DEFAULT_WEIGHTS,
) -> Dict[str, object]:
    """Measure the center (hero) zone + interior chroma on a delivery render.

    Returns dict: hero_ire, hero_log2, rgb_mean, wc, gm, pixel_count, valid.
    Geometry and the 4-step chain match measure.measure_render exactly; only the
    luminance weights differ (to match the delivery output color space).
    """
    image = load_render_as_hwc(render_path)
    h, w = image.shape[:2]

    # Center hero zone — identical geometry to measure._measure_zone(offset=0.0).
    zone_cy = roi.cy
    zone_half_h = roi.r * _ZONE_HALF_HEIGHT_RATIO
    zone_half_w = roi.r * _ZONE_HALF_WIDTH_RATIO
    y0 = max(0, int(np.floor(zone_cy - zone_half_h)))
    y1 = min(h, int(np.ceil(zone_cy + zone_half_h)) + 1)
    x0 = max(0, int(np.floor(roi.cx - zone_half_w)))
    x1 = min(w, int(np.ceil(roi.cx + zone_half_w)) + 1)

    if y1 <= y0 or x1 <= x0:
        return {"hero_ire": 0.0, "hero_log2": -10.0, "rgb_mean": (0.0, 0.0, 0.0),
                "wc": 0.0, "gm": 0.0, "pixel_count": 0, "valid": False}

    pixels = image[y0:y1, x0:x1].reshape(-1, 3)
    lum = np.clip(_luminance_weighted(pixels, luma_weights), 1e-6, None)

    p5, p95 = np.percentile(lum, [_TRIM_LOW, _TRIM_HIGH])
    trimmed = lum[(lum >= p5) & (lum <= p95)]
    if trimmed.size == 0:
        trimmed = lum
    log2_val = float(np.median(np.log2(np.maximum(trimmed, 1e-6))))
    ire = (2.0 ** log2_val) * 100.0

    # Interior chroma (inner 50% of r) — same region measure.py uses for WB.
    interior_mask = _interior_mask(image, roi.cx, roi.cy, roi.r * _INTERIOR_RADIUS_RATIO)
    interior = image[interior_mask] if interior_mask.sum() > 0 else np.zeros((1, 3), dtype=np.float32)
    rgb_mean = tuple(float(v) for v in np.mean(interior, axis=0))
    wc, gm = compute_wc_gm(rgb_mean)

    return {
        "hero_ire": ire,
        "hero_log2": log2_val,
        "rgb_mean": rgb_mean,
        "wc": wc,
        "gm": gm,
        "pixel_count": int(pixels.shape[0]),
        "valid": int(pixels.shape[0]) >= 50,
    }
