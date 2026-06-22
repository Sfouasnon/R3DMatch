"""
brdf_verify.py — BRDF fit diagnostic against known-good ROIs

Usage:
    python3 brdf_verify.py <json_dir> [--tiff-dir <override>] [--log-space]

Inputs:
    - Directory of *_001.json files (manual operator ROIs from R3DMatch)
    - TIFFs at the paths stored in each JSON (or overridden via --tiff-dir)
    - --log-space: TIFFs are RWG/Log3G10 renders, use Log3G10 inversion

What it measures per clip:
    1. Loads full-res TIFF, extracts sphere interior at known cx/cy/r
    2. Converts to scene-linear (BT.1886 inversion or Log3G10 inversion)
    3. Finds specular peak (brightest connected region inside 0.85r)
    4. Infers light direction from peak offset
    5. Solves for sensor rotation (0/90/180/270) by trying all four and
       picking the rotation that produces highest Pearson correlation
    6. Computes radial profile BRDF score: 16 spokes x 24 points, Pearson r
    7. Reports per-clip table with score, rotation, entropy, verdict

Score is mean Pearson r across 16 spokes. 1.0 = perfect Lambertian match.
Thresholds are provisional — calibrate from first run output.
"""

import sys
import os
import json
import glob
import argparse
import numpy as np

try:
    from tifffile import imread as tiff_imread
except ImportError:
    sys.exit("Missing dependency: pip install tifffile --break-system-packages")

try:
    from scipy.ndimage import gaussian_filter, label
except ImportError:
    sys.exit("Missing dependency: pip install scipy --break-system-packages")


# ---------------------------------------------------------------------------
# Thresholds — provisional, calibrate from first run output
# Score is mean Pearson r across 16 radial spokes (1.0 = perfect Lambertian)
# ---------------------------------------------------------------------------
BRDF_SCORE_GOOD     = 0.70
BRDF_SCORE_MARGINAL = 0.40


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _disk_mask(h, w, cx, cy, r, scale=1.0):
    ys, xs = np.ogrid[:h, :w]
    return (xs - cx)**2 + (ys - cy)**2 <= (r * scale)**2


def _to_lum(rgb):
    return 0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]


# ---------------------------------------------------------------------------
# Display transform inversions
# ---------------------------------------------------------------------------

def _invert_display_transform(rgb):
    # BT.1886 gamma inversion: display = linear^(1/2.4), inverse = display^2.4
    return np.power(np.clip(rgb, 0.0, 1.0), 2.4).astype(np.float32)


def _log3g10_to_linear(rgb):
    # Exact inversion of RED Log3G10 per RED white paper 915-0187 Rev-C.
    # Parameters: a=0.224282, b=155.975327, c=0.01, g=15.1927
    # Forward:    log3G10(x) = a * log10((x+c)*b + 1)  for x+c >= 0
    # Inversion:  linear = (10^(log_val/a) - 1) / b - c
    # Key: 18% gray (linear=0.18) encodes to exactly 0.333333
    a = 0.224282
    b = 155.975327
    c = 0.01
    g = 15.1927
    v = np.clip(rgb, 0.0, 1.0).astype(np.float64)
    linear = (np.power(10.0, v / a) - 1.0) / b - c
    neg = v < 0.0
    linear[neg] = v[neg] / g - c
    return np.clip(linear + c, 0.0, None).astype(np.float32)


# ---------------------------------------------------------------------------
# Specular peak detection
# ---------------------------------------------------------------------------

def _find_specular_peak(lum, cx, cy, r):
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
    best = np.argmax(sizes) + 1
    region = labeled == best
    ys, xs = np.where(region)
    return float(xs.mean()), float(ys.mean())


# ---------------------------------------------------------------------------
# Light direction inference
# ---------------------------------------------------------------------------

def _infer_light_direction(cx, cy, r, peak_x, peak_y):
    dx = (peak_x - cx) / r
    dy = (peak_y - cy) / r
    mag = np.sqrt(dx**2 + dy**2)
    if mag > 0.95:
        dx, dy = dx * 0.95 / mag, dy * 0.95 / mag
    lz = np.sqrt(max(0.0, 1.0 - dx**2 - dy**2))
    azimuth_deg  = float(np.degrees(np.arctan2(-dy, dx)))
    elevation_deg = float(np.degrees(np.arcsin(lz)))
    return np.array([dx, -dy, lz], dtype=np.float64), azimuth_deg, elevation_deg


# ---------------------------------------------------------------------------
# Radial profile BRDF scoring
# ---------------------------------------------------------------------------

def _sample_spoke(lum, cx, cy, r, angle_rad, n_points=24, max_t=0.92):
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


def _lambertian_spoke_profile(ts, angle_rad, light_vec):
    # Everything in image space (y increases downward).
    # light_vec from _infer_light_direction is [lx, ly_3d, lz] where ly_3d = -dy_image.
    # Convert back to image-y convention for dot product with spoke normal.
    lx, ly_3d, lz = light_vec
    ly = -ly_3d   # back to image-y (positive = downward)
    predicted = np.zeros(len(ts))
    for i, t in enumerate(ts):
        t = float(np.clip(t, 0.0, 1.0))
        nx = t * np.cos(angle_rad)
        ny = t * np.sin(angle_rad)
        nz = np.sqrt(max(0.0, 1.0 - t**2))
        dot = nx * lx + ny * ly + nz * lz
        predicted[i] = max(0.0, dot)
    return predicted


def _pearson(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 4:
        return np.nan
    a, b = a[mask], b[mask]
    if a.std() < 1e-9 or b.std() < 1e-9:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def _radial_brdf_score(lum, cx, cy, r, light_vec, n_spokes=16, n_points=24):
    spoke_corrs = []
    for s in range(n_spokes):
        angle = 2.0 * np.pi * s / n_spokes
        ts, actual = _sample_spoke(lum, cx, cy, r, angle, n_points=n_points)
        predicted  = _lambertian_spoke_profile(ts, angle, light_vec)
        r_val = _pearson(actual, predicted)
        spoke_corrs.append(r_val)
    valid = [v for v in spoke_corrs if np.isfinite(v)]
    if not valid:
        return None, None, 0, spoke_corrs
    return float(np.mean(valid)), float(np.min(valid)), len(valid), spoke_corrs


# ---------------------------------------------------------------------------
# Texture entropy
# ---------------------------------------------------------------------------

def _texture_entropy(lum, cx, cy, r, h, w):
    mask = _disk_mask(h, w, cx, cy, r, scale=0.7)
    vals = lum[mask]
    if len(vals) < 10:
        return None
    hist, _ = np.histogram(vals, bins=64, range=(0.0, 1.0))
    hist = hist[hist > 0].astype(float)
    hist /= hist.sum()
    return float(-np.sum(hist * np.log2(hist)))


# ---------------------------------------------------------------------------
# Per-clip analysis
# ---------------------------------------------------------------------------

def analyse_clip(json_path, tiff_dir_override=None, log_space=False):
    with open(json_path) as f:
        meta = json.load(f)

    clip_id = meta.get("clip_id") or meta.get("id") or os.path.basename(json_path).replace(".json", "")

    # ROI geometry — resolve to render space, then scale to TIFF space
    render_w = float(meta.get("render_width")  or meta.get("preview_width",  3840))
    render_h = float(meta.get("render_height") or meta.get("preview_height", 2160))

    roi = meta.get("roi_geometry") or {}
    if roi.get("cx") and roi.get("r"):
        cx_render = float(roi["cx"])
        cy_render = float(roi["cy"])
        r_render  = float(roi["r"])
    elif meta.get("image_roi_center") and meta.get("image_roi_radius"):
        cx_render = float(meta["image_roi_center"][0])
        cy_render = float(meta["image_roi_center"][1])
        r_render  = float(meta["image_roi_radius"])
    elif meta.get("center_normalized") and meta.get("radius_preview_px"):
        cx_render = float(meta["center_normalized"][0]) * render_w
        cy_render = float(meta["center_normalized"][1]) * render_h
        r_render  = float(meta["radius_preview_px"])
    else:
        cx_render = cy_render = r_render = 0.0

    r_w_canonical = r_render / render_w

    if r_render < 10:
        return {"clip": clip_id, "error": "invalid ROI geometry"}

    # TIFF path
    tiff_path = meta.get("source_image_for_measurement") or meta.get("source_image")
    if tiff_dir_override:
        candidate = os.path.join(tiff_dir_override, os.path.basename(tiff_path))
        if not os.path.exists(candidate):
            clip_stem = "_".join(clip_id.split("_")[:2])
            import glob as _glob
            matches = _glob.glob(os.path.join(tiff_dir_override, clip_stem + "*.tif*"))
            candidate = matches[0] if matches else candidate
        tiff_path = candidate

    if not tiff_path or not os.path.exists(tiff_path):
        return {"clip": clip_id, "error": f"TIFF not found: {tiff_path}"}

    # Load image
    raw = tiff_imread(tiff_path)
    if raw.dtype == np.uint8:
        rgb = raw.astype(np.float32) / 255.0
    elif raw.dtype == np.uint16:
        rgb = raw.astype(np.float32) / 65535.0
    else:
        rgb = raw.astype(np.float32)
        if rgb.max() > 1.0:
            rgb /= rgb.max()

    if rgb.ndim == 2:
        rgb = np.stack([rgb, rgb, rgb], axis=-1)
    elif rgb.shape[-1] == 4:
        rgb = rgb[:, :, :3]

    # Convert to scene-linear for BRDF fitting
    rgb_display = rgb.copy()
    if log_space:
        rgb = _log3g10_to_linear(rgb)
    else:
        rgb = _invert_display_transform(rgb)

    h, w = rgb.shape[:2]

    # Scale ROI from render space to TIFF space
    scale_x = w / render_w
    scale_y = h / render_h
    cx = cx_render * scale_x
    cy = cy_render * scale_y
    r  = r_render  * scale_x
    r_w = r_w_canonical

    lum = _to_lum(rgb)

    # Zone IREs from JSON
    hero_ire   = meta.get("hero_center_ire") or meta.get("sampled_center_ire")
    spec_ire   = meta.get("sample_1_ire")
    shadow_ire = meta.get("sample_3_ire")
    if spec_ire is None and meta.get("hero_center_measurement"):
        hcm = meta["hero_center_measurement"]
        max_lum = hcm.get("max_luminance")
        min_lum = hcm.get("min_luminance")
        if max_lum: spec_ire   = float(max_lum) * 100
        if min_lum: shadow_ire = float(min_lum) * 100
    gradient_range = None
    if spec_ire is not None and shadow_ire is not None:
        gradient_range = spec_ire - shadow_ire

    # Specular peak
    peak_x, peak_y = _find_specular_peak(lum, cx, cy, r)
    peak_offset_x = (peak_x - cx) / r
    peak_offset_y = (peak_y - cy) / r

    # Light direction + rotation solve
    # Try 0/90/180/270 degree rotations of the inferred light vector.
    # Camera may be physically rotated with no IMU metadata available.
    # The rotation that produces the highest mean Pearson score is kept.
    light_vec_raw, azimuth_raw, elevation = _infer_light_direction(cx, cy, r, peak_x, peak_y)

    best_score = None
    best_vec   = light_vec_raw
    best_rot   = 0
    for rot_deg in [0, 90, 180, 270]:
        a = np.radians(rot_deg)
        lx0, ly0, lz0 = light_vec_raw
        lx_r =  lx0 * np.cos(a) - ly0 * np.sin(a)
        ly_r =  lx0 * np.sin(a) + ly0 * np.cos(a)
        vec_r = np.array([lx_r, ly_r, lz0])
        s, s_min, n_v, _ = _radial_brdf_score(lum, cx, cy, r, vec_r, n_spokes=16, n_points=24)
        if s is not None and (best_score is None or s > best_score):
            best_score = s
            best_vec   = vec_r
            best_rot   = rot_deg

    light_vec = best_vec
    azimuth   = float(np.degrees(np.arctan2(-best_vec[1], best_vec[0])))

    # Final spoke correlation with best rotation
    brdf_score, brdf_score_min, n_valid_spokes, spoke_corrs = _radial_brdf_score(
        lum, cx, cy, r, light_vec, n_spokes=16, n_points=24
    )

    if brdf_score is None:
        return {"clip": clip_id, "error": "insufficient spoke data"}

    # Texture entropy
    entropy = _texture_entropy(lum, cx, cy, r, h, w)

    # Verdict
    if brdf_score >= BRDF_SCORE_GOOD:
        verdict = "GOOD"
    elif brdf_score >= BRDF_SCORE_MARGINAL:
        verdict = "MARGINAL"
    else:
        verdict = "POOR"

    return {
        "clip":            clip_id,
        "r_px":            round(r, 1),
        "r_w":             round(r_w, 4),
        "hero_ire":        round(hero_ire, 1)       if hero_ire        is not None else None,
        "spec_ire":        round(spec_ire, 1)       if spec_ire        is not None else None,
        "shadow_ire":      round(shadow_ire, 1)     if shadow_ire      is not None else None,
        "grad_range":      round(gradient_range, 1) if gradient_range  is not None else None,
        "peak_offset_x":   round(peak_offset_x, 3),
        "peak_offset_y":   round(peak_offset_y, 3),
        "light_az":        round(azimuth, 1),
        "light_el":        round(elevation, 1),
        "sensor_rot":      best_rot,
        "brdf_score":      round(brdf_score, 4),
        "brdf_score_min":  round(brdf_score_min, 4),
        "n_valid_spokes":  n_valid_spokes,
        "entropy":         round(entropy, 3)        if entropy         is not None else None,
        "verdict":         verdict,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="BRDF fit verification against known ROIs")
    parser.add_argument("json_dir", help="Directory containing *_001.json ROI files")
    parser.add_argument("--tiff-dir", default=None,
                        help="Override TIFF directory (if TIFFs have moved)")
    parser.add_argument("--log-space", action="store_true",
                        help="TIFFs are RWG/Log3G10 — use Log3G10 inversion instead of BT.1886")
    args = parser.parse_args()

    json_files = sorted(glob.glob(os.path.join(args.json_dir, "*.json")))
    if not json_files:
        sys.exit(f"No JSON files found in: {args.json_dir}")

    results = []
    for jf in json_files:
        r = analyse_clip(jf, tiff_dir_override=args.tiff_dir, log_space=args.log_space)
        results.append(r)

    header = (
        f"{'Clip':<22} {'r/w':>6} {'heroIRE':>8} {'spec':>6} {'shad':>6} "
        f"{'grad':>5} {'az':>6} {'el':>5} {'rot':>6} {'score':>7} {'min':>7} {'entropy':>7}  verdict"
    )
    print(header)
    print("-" * len(header))

    good = marginal = poor = errors = 0
    for r in results:
        if "error" in r:
            print(f"  {r['clip']:<20}  ERROR: {r['error']}")
            errors += 1
            continue

        def _f(v, fmt):
            return format(v, fmt) if v is not None else "   N/A"

        print(
            f"  {r['clip']:<20}"
            f"  {_f(r['r_w'],           '6.3f')}"
            f"  {_f(r['hero_ire'],      '7.1f')}"
            f"  {_f(r['spec_ire'],      '5.1f')}"
            f"  {_f(r['shadow_ire'],    '5.1f')}"
            f"  {_f(r['grad_range'],    '4.1f')}"
            f"  {_f(r['light_az'],      '5.1f')}"
            f"  {_f(r['light_el'],      '4.1f')}"
            f"  {r.get('sensor_rot', 0):>4}deg"
            f"  {_f(r['brdf_score'],    '7.4f')}"
            f"  {_f(r['brdf_score_min'],'7.4f')}"
            f"  {_f(r['entropy'],       '7.3f')}"
            f"  {r['verdict']}"
        )
        if r["verdict"] == "GOOD":       good += 1
        elif r["verdict"] == "MARGINAL": marginal += 1
        else:                            poor += 1

    print("-" * len(header))
    print(f"  {good} GOOD  {marginal} MARGINAL  {poor} POOR  {errors} ERROR")
    print()
    print("Notes:")
    print("  brdf_score — mean Pearson r across 16 radial spokes (higher = better Lambertian match)")
    print("  brdf_min   — weakest spoke correlation (flags occluded/angled edge spokes)")
    print("  rot        — sensor rotation solved (0/90/180/270 deg) — no IMU metadata available")
    print("  entropy    — luminance histogram entropy of inner 0.7r (lower = more featureless)")
    print("  light_az   — inferred key light azimuth after rotation correction (deg)")
    print("  light_el   — inferred key light elevation above camera plane (deg)")
    print("  spec/shad  — sample_1/sample_3 IRE from JSON (not recomputed)")
    print("  grad_range — spec_ire minus shadow_ire across sphere surface")


if __name__ == "__main__":
    main()
