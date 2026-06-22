"""
probe_canny.py — Session 12 diagnostic

Tests _canny_contour_candidates against a real render TIFF without touching
sphere.py. Pass the path to a TIFF on the command line.

Usage:
    python3 probe_canny.py /path/to/clip.tiff

Prints:
  - What Hough finds
  - What contour candidates the new algorithm finds
  - Whether they overlap or are novel
  - A summary of where each candidate lands vs the frame
"""

import sys
import math
import numpy as np
from PIL import Image
from skimage.feature import canny, blob_log
from skimage.measure import find_contours
from skimage.transform import hough_circle, hough_circle_peaks

# ---------------------------------------------------------------------------
# Constants — must match sphere.py exactly
# ---------------------------------------------------------------------------
_CANNY_SIGMA           = 0.5
_CANNY_LOW             = 0.005
_CANNY_HIGH            = 0.02
_HOUGH_ACCUMULATOR_MIN = 0.25
_HOUGH_NUM_PEAKS       = 50
_HOUGH_MIN_DIST        = 20
_RADIUS_MIN_RATIO      = 0.02
_RADIUS_MAX_RATIO      = 0.15
_PF_RADIUS_MIN         = 0.018
_DETECTION_TARGET_DIM  = 1080
_CANNY_CONTOUR_ACCUMULATOR    = 0.65
_CANNY_CONTOUR_MERGE_TOLERANCE = 0.5

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _to_gray(arr):
    return 0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]

def _resize_for_detection(img):
    w, h = img.size
    scale = _DETECTION_TARGET_DIM / max(w, h)
    if scale >= 1.0:
        return img, 1.0
    img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return img, scale

def _min_enclosing_circle(contour):
    pts = contour[:, ::-1].astype(np.float32)
    cx = float(pts[:, 0].mean())
    cy = float(pts[:, 1].mean())
    r  = float(np.sqrt(((pts[:, 0] - cx)**2 + (pts[:, 1] - cy)**2).max()))
    return cx, cy, r

def _gray_contour_candidates(gray, w):
    candidates = []
    seen = []
    p5     = float(np.percentile(gray, 5))
    p99    = float(np.percentile(gray, 99))
    img_max = float(gray.max())
    thr_hi = max(p99, img_max * 0.95)
    span   = thr_hi - p5
    if span < 0.005:
        print("  [contour] image is essentially flat — span < 0.005, skipping")
        return candidates
    thr_lo = p5 + span * 0.02
    thr_hi = thr_hi - span * 0.01
    thresholds = np.linspace(thr_lo, thr_hi, 14)
    print(f"  [contour] gray range: {gray.min():.3f}–{gray.max():.3f}  "
          f"p5={p5:.3f} p99={p99:.3f} img_max={img_max:.3f}")
    print(f"  [contour] threshold sweep: {thresholds[0]:.3f}–{thresholds[-1]:.3f} ({len(thresholds)} steps)")
    for thr in thresholds:
        for contour in find_contours(gray, level=float(thr)):
            if len(contour) < 20:
                continue
            cx, cy, r = _min_enclosing_circle(contour)
            if not (_PF_RADIUS_MIN <= r / w <= _RADIUS_MAX_RATIO):
                continue
            circumference = 2.0 * math.pi * r
            circularity = len(contour) / circumference if circumference > 0 else 0.0
            if not (0.4 <= circularity <= 2.2):
                continue
            dup = any(math.hypot(cx - scx, cy - scy) < 0.5 * r for scx, scy, _ in seen)
            if dup:
                continue
            candidates.append((cx, cy, r))
            seen.append((cx, cy, r))
    return candidates

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(tiff_path):
    print(f"\n=== probe_canny.py ===")
    print(f"TIFF: {tiff_path}\n")

    img = Image.open(tiff_path)
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGB")
    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (0, 0, 0))
        bg.paste(img, mask=img.split()[3])
        img = bg
    img_det, scale = _resize_for_detection(img)
    arr = np.array(img_det, dtype=np.float32) / 255.0
    h, w = arr.shape[:2]
    gray = _to_gray(arr)

    print(f"Detection-scale frame: {w}×{h}  (scale={scale:.3f})")
    r_min = max(6, int(w * _RADIUS_MIN_RATIO))
    r_max = min(h // 2, int(w * _RADIUS_MAX_RATIO))
    print(f"Hough radius search: {r_min}–{r_max} px  "
          f"(r/w {r_min/w:.3f}–{r_max/w:.3f})\n")

    # --- Hough ---
    edges = canny(gray, sigma=_CANNY_SIGMA,
                  low_threshold=_CANNY_LOW, high_threshold=_CANNY_HIGH)
    radii = np.arange(r_min, r_max + 1)
    hough_res = hough_circle(edges, radii)
    _, cx_arr, cy_arr, r_arr = hough_circle_peaks(
        hough_res, radii,
        min_xdistance=_HOUGH_MIN_DIST,
        min_ydistance=_HOUGH_MIN_DIST,
        num_peaks=_HOUGH_NUM_PEAKS,
        threshold=_HOUGH_ACCUMULATOR_MIN,
    )
    hough_peaks = list(zip(cx_arr.tolist(), cy_arr.tolist(), r_arr.tolist()))
    print(f"--- Hough: {len(hough_peaks)} peaks ---")
    for i, (cx, cy, r) in enumerate(hough_peaks[:10]):
        r_idx = np.searchsorted(radii, r)
        r_idx = np.clip(r_idx, 0, len(radii) - 1)
        acc = float(hough_res[r_idx, int(cy), int(cx)])
        print(f"  [{i:2d}] cx={cx/scale:6.0f} cy={cy/scale:6.0f}  "
              f"r={r/scale:5.0f}  r/w={r/w:.3f}  acc={acc:.3f}")
    if len(hough_peaks) > 10:
        print(f"  ... ({len(hough_peaks) - 10} more)")

    # --- Contour candidates ---
    print(f"\n--- Contour candidates ---")
    canny_cands = _gray_contour_candidates(gray, w)
    print(f"  Found: {len(canny_cands)}")
    for i, (cx, cy, r) in enumerate(canny_cands):
        # Check if this is novel vs Hough
        near_hough = any(
            math.hypot(cx - hcx, cy - hcy) < _CANNY_CONTOUR_MERGE_TOLERANCE * r
            for hcx, hcy, _ in hough_peaks
        )
        tag = "DUPLICATE (Hough already has it)" if near_hough else "NOVEL"
        print(f"  [{i:2d}] cx={cx/scale:6.0f} cy={cy/scale:6.0f}  "
              f"r={r/scale:5.0f}  r/w={r/w:.3f}  acc={_CANNY_CONTOUR_ACCUMULATOR}  [{tag}]")

    print(f"\n--- Summary ---")
    novel = [(cx, cy, r) for cx, cy, r in canny_cands
             if not any(math.hypot(cx - hcx, cy - hcy) < _CANNY_CONTOUR_MERGE_TOLERANCE * r
                        for hcx, hcy, _ in hough_peaks)]
    print(f"  Hough peaks:          {len(hough_peaks)}")
    print(f"  Contour candidates:   {len(canny_cands)}")
    print(f"  Novel (not in Hough): {len(novel)}")
    if novel:
        print(f"\n  Novel candidates (full-res coords):")
        for cx, cy, r in novel:
            print(f"    cx={cx/scale:.0f}  cy={cy/scale:.0f}  r={r/scale:.0f}  r/w={r/w:.3f}")
    print()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 probe_canny.py /path/to/clip.tiff")
        sys.exit(1)
    main(sys.argv[1])
