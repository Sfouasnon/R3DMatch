"""
probe_specular.py — One-off probe: specular highlight detection as sphere discriminator.

Tests whether a localised luminance peak (specular highlight) inside the detected ROI
can distinguish a real 18% gray sphere from a flat focus chart.

A convex diffuse sphere lit from off-axis will always have:
  1. A specular highlight — small bright lobe offset from geometric center
  2. A shadow terminator — dark region on the opposite side
  3. The highlight and shadow are spatially coherent with a single light source

A flat chart has no specular. Its luminance gradient is diffuse and low-contrast.

Usage:
    python3 probe_specular.py

Reads ROIs from the latest MonJune8_UnderExposedSet analysis JSONs.
Runs against the corresponding measurement TIFFs.
Prints specular metrics for all cameras, flagging HB (chart) vs field (sphere).
"""

import json
import sys
import os
import math
import numpy as np
from pathlib import Path
from PIL import Image

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RUN_DIR   = Path("/Users/sfouasnon/Desktop/Test_Run/MonJune8_UnderExposedSet")
ANALYSIS  = RUN_DIR / "analysis"
PREVIEWS  = RUN_DIR / "previews" / "_measurement"
SRC_ROOT  = "/Users/sfouasnon/Desktop/R3DMatch_v3/R3DMatch_v3/r3dmatch_v3/src"

sys.path.insert(0, SRC_ROOT)

# ---------------------------------------------------------------------------
# Image helpers (inline — no source dependency on internal APIs)
# ---------------------------------------------------------------------------

def load_tif(path: str) -> np.ndarray:
    img = Image.open(path)
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGB")
    return np.array(img, dtype=np.float32) / 255.0


def lum(rgb: np.ndarray) -> np.ndarray:
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]


def interior_mask(h, w, cx, cy, r, scale=1.0):
    ys, xs = np.ogrid[:h, :w]
    return np.sqrt((xs - cx)**2 + (ys - cy)**2) <= (r * scale)


def annular_mask(h, w, cx, cy, r_inner, r_outer):
    ys, xs = np.ogrid[:h, :w]
    d = np.sqrt((xs - cx)**2 + (ys - cy)**2)
    return (d >= r_inner) & (d <= r_outer)


# ---------------------------------------------------------------------------
# Specular probe
# ---------------------------------------------------------------------------

def probe_specular(rgb: np.ndarray, cx: float, cy: float, r: float) -> dict:
    """
    Probe for specular highlight signature inside a sphere ROI.

    Metrics returned:
      peak_lum          — max luminance inside 0.6r disk
      mean_lum          — mean luminance inside 0.6r disk
      peak_excess       — peak_lum / mean_lum  (sphere: >1.15, chart: ~1.0)
      peak_offset_norm  — distance of peak pixel from center, normalised by r
                          (sphere: 0.1–0.45, chart: near 0 or near edge)
      highlight_fraction— fraction of interior pixels > (mean + 2*std)
                          (sphere: small tight lobe, chart: diffuse or none)
      shadow_ratio      — mean lum of dark half / mean lum of bright half
                          split along axis from center toward peak
                          (sphere: <0.75 shadow side, chart: ~0.9+)
      specular_score    — composite 0–1  (higher = more sphere-like)
    """
    h, w = rgb.shape[:2]
    gray = lum(rgb)

    # Sample interior disk at 0.7r
    mask_07 = interior_mask(h, w, cx, cy, r, 0.7)
    if mask_07.sum() < 20:
        return {"error": "insufficient pixels"}

    interior = gray[mask_07]
    mean_i = float(interior.mean())
    std_i  = float(interior.std())

    # Find peak luminance pixel within 0.7r
    masked_gray = np.where(mask_07, gray, 0.0)
    peak_idx = np.unravel_index(np.argmax(masked_gray), gray.shape)
    peak_y, peak_x = peak_idx
    peak_lum_val = float(gray[peak_y, peak_x])

    # Peak offset from center (normalised by r)
    offset_px = math.hypot(peak_x - cx, peak_y - cy)
    offset_norm = offset_px / max(r, 1.0)

    # Peak excess — how much brighter is the peak vs the mean
    peak_excess = peak_lum_val / max(mean_i, 1e-6)

    # Highlight fraction — tight bright lobe
    threshold = mean_i + 2.0 * std_i
    highlight_pixels = int((gray[mask_07] > threshold).sum())
    highlight_fraction = highlight_pixels / max(mask_07.sum(), 1)

    # Shadow ratio — split disk along axis center→peak
    # Vector from center to peak
    dx = peak_x - cx
    dy = peak_y - cy
    norm = math.hypot(dx, dy)
    if norm > 1.0:
        dx /= norm
        dy /= norm
    else:
        dx, dy = 1.0, 0.0  # no offset — arbitrary axis

    # Project each interior pixel onto this axis
    ys_idx, xs_idx = np.where(mask_07)
    proj = (xs_idx - cx) * dx + (ys_idx - cy) * dy  # positive = bright side

    bright_mask_vals = gray[mask_07][proj >= 0]
    dark_mask_vals   = gray[mask_07][proj <  0]

    bright_mean = float(bright_mask_vals.mean()) if len(bright_mask_vals) > 0 else mean_i
    dark_mean   = float(dark_mask_vals.mean())   if len(dark_mask_vals)   > 0 else mean_i
    shadow_ratio = dark_mean / max(bright_mean, 1e-6)

    # ---------------------------------------------------------------------------
    # Composite specular score
    # A real sphere scores high on:
    #   - peak_excess > 1.15  (tight bright spot)
    #   - offset_norm 0.10–0.45  (highlight offset from center, not at edge)
    #   - highlight_fraction < 0.15  (small lobe, not diffuse)
    #   - shadow_ratio < 0.80  (real shadow side)
    # ---------------------------------------------------------------------------
    score = 0.0

    # Peak excess contribution (0–0.35)
    if peak_excess >= 1.30:
        score += 0.35
    elif peak_excess >= 1.15:
        score += 0.20
    elif peak_excess >= 1.05:
        score += 0.08

    # Offset contribution (0–0.25) — highlight should be off-center but inside sphere
    if 0.10 <= offset_norm <= 0.45:
        score += 0.25
    elif 0.05 <= offset_norm <= 0.55:
        score += 0.12

    # Highlight fraction contribution (0–0.20) — tight lobe
    if highlight_fraction < 0.06:
        score += 0.20
    elif highlight_fraction < 0.12:
        score += 0.12
    elif highlight_fraction < 0.20:
        score += 0.05

    # Shadow ratio contribution (0–0.20)
    if shadow_ratio < 0.70:
        score += 0.20
    elif shadow_ratio < 0.80:
        score += 0.12
    elif shadow_ratio < 0.88:
        score += 0.05

    return {
        "peak_lum":           round(peak_lum_val, 4),
        "mean_lum":           round(mean_i, 4),
        "std_lum":            round(std_i, 4),
        "peak_excess":        round(peak_excess, 4),
        "offset_norm":        round(offset_norm, 4),
        "highlight_fraction": round(highlight_fraction, 4),
        "shadow_ratio":       round(shadow_ratio, 4),
        "specular_score":     round(score, 3),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    jsons = sorted(ANALYSIS.glob("*.analysis.json"))
    if not jsons:
        print(f"No analysis JSONs found in {ANALYSIS}")
        sys.exit(1)

    print(f"{'camera':<16} {'src':<14} {'IRE':>6}  "
          f"{'peak_ex':>8} {'offset':>7} {'hl_frac':>8} {'shad_r':>7} {'score':>6}  note")
    print("-" * 95)

    results = []
    for jpath in jsons:
        try:
            d = json.loads(jpath.read_text())
        except Exception as e:
            print(f"  skip {jpath.name}: {e}")
            continue

        cam    = d.get("camera_label", "?")
        src    = d.get("detection_source", "?")
        ire    = d.get("hero_ire", 0.0)
        roi    = d.get("sphere_roi") or {}
        cx     = roi.get("cx", 0.0)
        cy     = roi.get("cy", 0.0)
        r      = roi.get("r",  0.0)
        rpath  = d.get("render_path", "")

        if not rpath or not Path(rpath).exists():
            # Try finding the tif in the previews dir
            tif_name = jpath.stem.replace(".analysis", "") + ".original.analysis.measurement.tiff.000000.tif"
            rpath = str(PREVIEWS / tif_name)

        if not Path(rpath).exists():
            print(f"  {cam:<16} no render found — skip")
            continue

        try:
            rgb = load_tif(rpath)
        except Exception as e:
            print(f"  {cam:<16} load error: {e}")
            continue

        metrics = probe_specular(rgb, cx, cy, r)
        if "error" in metrics:
            print(f"  {cam:<16} {metrics['error']}")
            continue

        note = ""
        if ire > 45:
            note = "◄ CHART"
        elif metrics["specular_score"] < 0.35:
            note = "? low score"

        print(f"{cam:<16} {src:<14} {ire:>6.1f}  "
              f"{metrics['peak_excess']:>8.4f} "
              f"{metrics['offset_norm']:>7.4f} "
              f"{metrics['highlight_fraction']:>8.4f} "
              f"{metrics['shadow_ratio']:>7.4f} "
              f"{metrics['specular_score']:>6.3f}  {note}")

        results.append({**metrics, "camera": cam, "ire": ire, "source": src})

    # Summary
    print()
    chart  = [r for r in results if r["ire"] > 45]
    sphere = [r for r in results if r["ire"] <= 45]
    if sphere:
        print(f"Sphere population ({len(sphere)} cameras):")
        print(f"  specular_score : {min(r['specular_score'] for r in sphere):.3f} – {max(r['specular_score'] for r in sphere):.3f}")
        print(f"  peak_excess    : {min(r['peak_excess']    for r in sphere):.4f} – {max(r['peak_excess']    for r in sphere):.4f}")
        print(f"  shadow_ratio   : {min(r['shadow_ratio']   for r in sphere):.4f} – {max(r['shadow_ratio']   for r in sphere):.4f}")
        print(f"  offset_norm    : {min(r['offset_norm']    for r in sphere):.4f} – {max(r['offset_norm']    for r in sphere):.4f}")
    if chart:
        print(f"\nChart population ({len(chart)} cameras):")
        print(f"  specular_score : {min(r['specular_score'] for r in chart):.3f} – {max(r['specular_score'] for r in chart):.3f}")
        print(f"  peak_excess    : {min(r['peak_excess']    for r in chart):.4f} – {max(r['peak_excess']    for r in chart):.4f}")
        print(f"  shadow_ratio   : {min(r['shadow_ratio']   for r in chart):.4f} – {max(r['shadow_ratio']   for r in chart):.4f}")
        print(f"  offset_norm    : {min(r['offset_norm']    for r in chart):.4f} – {max(r['offset_norm']    for r in chart):.4f}")


if __name__ == "__main__":
    main()
