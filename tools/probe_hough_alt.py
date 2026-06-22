"""
probe_hough_alt.py — Session 12

Compares cv2.HoughCircles HOUGH_GRADIENT vs HOUGH_GRADIENT_ALT on real
footage TIFFs against known ground-truth ROIs from the sphere profile.

HOUGH_GRADIENT_ALT differences from standard:
  - Uses gradient direction to constrain votes — edge pixels only vote
    in the direction their gradient points toward the center.
  - param2 is a different scale: it's the accumulator threshold as a
    fraction of the maximum possible votes, roughly analogous to skimage's
    normalized score but computed differently.
  - Requires fewer arc fragments to agree — much stricter about closure.

Reports: candidate count, sphere rank, and what beats the sphere for each.

Usage:
    python3 probe_hough_alt.py <tiff_folder> <profile_json>
"""

import sys, os, json, glob, math
import numpy as np
import cv2
from PIL import Image
import time

# ---------------------------------------------------------------------------
# Detection constants — match sphere.py where applicable
# ---------------------------------------------------------------------------
_DETECTION_TARGET_DIM  = 1080
_RADIUS_MIN_RATIO      = 0.018
_RADIUS_MAX_RATIO      = 0.15

# Standard Hough parameters (cv2 equivalent of current skimage settings)
# param1 = Canny high threshold (low is param1/2)
# param2 = accumulator threshold (votes needed — NOT normalized like skimage)
_STD_PARAM1            = 50     # Canny high threshold
_STD_PARAM2            = 30     # accumulator vote threshold
_STD_MIN_DIST_RATIO    = 0.02   # min distance between centers as fraction of w

# HOUGH_GRADIENT_ALT parameters
# param1 = same Canny threshold
# param2 = 0–1 normalized confidence threshold (closer to 1 = stricter)
_ALT_PARAM1            = 50
_ALT_PARAM2            = 0.7    # start strict, we'll sweep if needed

# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------
def _load_image(path):
    img = Image.open(path)
    if img.mode not in ("RGB","RGBA"): img = img.convert("RGB")
    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (0,0,0))
        bg.paste(img, mask=img.split()[3]); img = bg
    return img

def _resize_for_detection(img):
    w, h = img.size
    scale = _DETECTION_TARGET_DIM / max(w, h)
    if scale >= 1.0: return img, 1.0
    return img.resize((int(w*scale), int(h*scale)), Image.LANCZOS), scale

def _to_gray_uint8(arr_f32):
    """Convert float32 HxWx3 to uint8 grayscale for cv2."""
    gray = 0.2126*arr_f32[...,0] + 0.7152*arr_f32[...,1] + 0.0722*arr_f32[...,2]
    return (np.clip(gray, 0, 1) * 255).astype(np.uint8)

# ---------------------------------------------------------------------------
# Profile helpers
# ---------------------------------------------------------------------------
def load_profile(path):
    with open(os.path.expanduser(path)) as f: return json.load(f)

def get_gt(profile, label):
    cam = profile.get("cameras", {}).get(label, {})
    samples = [s for s in cam.get("samples",[]) if s.get("trust")=="verified_manual"]
    if not samples: samples = cam.get("samples",[])
    if not samples: return None
    g = samples[-1].get("geometry",{})
    return g.get("cx_norm"), g.get("cy_norm"), g.get("radius_ratio")

def camera_label(fname):
    parts = os.path.basename(fname).split("_")
    return f"{parts[0]}_{parts[1]}" if len(parts)>=2 else None

def find_rank(cands, gt):
    """cands: list of (cx,cy,r) tuples. Returns 0-based rank or None."""
    if gt is None: return None
    for i,(cx,cy,r) in enumerate(cands):
        if math.hypot(cx-gt['cx'], cy-gt['cy']) < 0.5*gt['r']: return i
    return None

# ---------------------------------------------------------------------------
# Hough detectors
# ---------------------------------------------------------------------------
def run_standard_hough(gray_u8, w, h, param2=_STD_PARAM2):
    """cv2 HOUGH_GRADIENT — standard accumulator."""
    r_min = int(w * _RADIUS_MIN_RATIO)
    r_max = int(w * _RADIUS_MAX_RATIO)
    min_dist = int(w * _STD_MIN_DIST_RATIO)
    circles = cv2.HoughCircles(
        gray_u8,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=min_dist,
        param1=_STD_PARAM1,
        param2=param2,
        minRadius=r_min,
        maxRadius=r_max,
    )
    if circles is None: return []
    return [(float(x), float(y), float(r))
            for x,y,r in circles[0]]

def run_alt_hough(gray_u8, w, h, param2=_ALT_PARAM2):
    """cv2 HOUGH_GRADIENT_ALT — gradient direction constrained."""
    r_min = int(w * _RADIUS_MIN_RATIO)
    r_max = int(w * _RADIUS_MAX_RATIO)
    min_dist = int(w * _STD_MIN_DIST_RATIO)
    circles = cv2.HoughCircles(
        gray_u8,
        cv2.HOUGH_GRADIENT_ALT,
        dp=1.5,        # ALT works better with dp > 1
        minDist=min_dist,
        param1=_ALT_PARAM1,
        param2=param2,
        minRadius=r_min,
        maxRadius=r_max,
    )
    if circles is None: return []
    return [(float(x), float(y), float(r))
            for x,y,r in circles[0]]

# ---------------------------------------------------------------------------
# Per-TIFF analysis
# ---------------------------------------------------------------------------
def analyze(tiff_path, profile):
    label = camera_label(tiff_path)
    gt_raw = get_gt(profile, label) if label else None

    img = _load_image(tiff_path)
    img_det, scale = _resize_for_detection(img)
    arr = np.array(img_det, dtype=np.float32) / 255.0
    h, w = arr.shape[:2]
    gray_u8 = _to_gray_uint8(arr)

    # Apply mild blur — ALT is sensitive to noise
    gray_blur = cv2.GaussianBlur(gray_u8, (5,5), 1.0)

    gt_det = None
    if gt_raw and all(v is not None for v in gt_raw):
        gt_det = {'cx': gt_raw[0]*w, 'cy': gt_raw[1]*h, 'r': gt_raw[2]*w}

    # Sweep ALT param2 to find a useful operating point
    # Too high = misses sphere; too low = same noise as standard
    alt_results = {}
    for p2 in [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]:
        t0 = time.time()
        cands = run_alt_hough(gray_blur, w, h, param2=p2)
        elapsed = time.time() - t0
        rank = find_rank(cands, gt_det)
        alt_results[p2] = {'cands': cands, 'rank': rank, 'time': elapsed}

    # Standard Hough for comparison
    t0 = time.time()
    std_cands = run_standard_hough(gray_blur, w, h)
    std_time = time.time() - t0
    std_rank = find_rank(std_cands, gt_det)

    return {
        'label': label, 'w': w, 'h': h, 'scale': scale,
        'gt_det': gt_det,
        'std_cands': std_cands, 'std_rank': std_rank, 'std_time': std_time,
        'alt_results': alt_results,
    }

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def report(res):
    cam = res['label'] or '?'
    w, scale = res['w'], res['scale']
    gt = res['gt_det']
    n_std = len(res['std_cands'])
    sr_std = res['std_rank']

    print(f"\n{'='*70}")
    print(f"  {cam}")
    if gt:
        print(f"  GT: cx={gt['cx']/scale:.0f} cy={gt['cy']/scale:.0f} "
              f"r={gt['r']/scale:.0f} r/w={gt['r']/w:.3f}")
    print(f"{'='*70}")

    # Standard
    sr_std_str = f"rank {sr_std+1}/{n_std}" if sr_std is not None else "MISSED"
    print(f"  Standard Hough: {n_std:3d} candidates  sphere {sr_std_str}  "
          f"({res['std_time']*1000:.0f}ms)")
    if sr_std is not None and sr_std > 0:
        above = res['std_cands'][:min(3,sr_std)]
        for i,(cx,cy,r) in enumerate(above):
            print(f"    above [{i+1}] cx={cx/scale:.0f} cy={cy/scale:.0f} "
                  f"r={r/scale:.0f} r/w={r/w:.3f}")

    # ALT sweep
    print(f"\n  HOUGH_GRADIENT_ALT sweep:")
    print(f"  {'param2':>8}  {'cands':>6}  {'sphere_rank':>12}  {'verdict':>12}")
    for p2, ar in sorted(res['alt_results'].items(), reverse=True):
        n = len(ar['cands'])
        rank = ar['rank']
        rank_str = f"{rank+1}/{n}" if rank is not None else "MISSED"
        if rank == 0:       v = "✓ rank 1"
        elif rank is None:  v = "missed"
        elif rank < (sr_std or 999): v = f"better ({rank+1})"
        else:               v = f"rank {rank+1}"
        print(f"  {p2:>8.1f}  {n:>6}  {rank_str:>12}  {v:>12}")

    # Best ALT result
    best_p2 = min(
        res['alt_results'],
        key=lambda p: (res['alt_results'][p]['rank'] if res['alt_results'][p]['rank'] is not None else 9999)
    )
    best = res['alt_results'][best_p2]
    if best['rank'] == 0:
        print(f"\n  Best ALT (param2={best_p2}): sphere rank 1 ✓")
        if best['cands']:
            cx,cy,r = best['cands'][0]
            print(f"    cx={cx/scale:.0f} cy={cy/scale:.0f} r={r/scale:.0f} r/w={r/w:.3f}")

def summary(results):
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Camera':<14} {'Std-rank':>10} {'ALT@0.9':>8} "
          f"{'ALT@0.7':>8} {'ALT@0.5':>8} {'Best-ALT':>10}")
    print(f"  {'-'*14} {'-'*10} {'-'*8} {'-'*8} {'-'*8} {'-'*10}")

    for res in results:
        cam = (res['label'] or '?')[:14]
        n_std = len(res['std_cands'])
        sr = res['std_rank']
        std_s = f"{sr+1}/{n_std}" if sr is not None else "MISS"

        def alt_s(p2):
            ar = res['alt_results'][p2]
            r = ar['rank']; n = len(ar['cands'])
            return f"{r+1}/{n}" if r is not None else "MISS"

        best_rank = min(
            (res['alt_results'][p]['rank'] for p in res['alt_results']
             if res['alt_results'][p]['rank'] is not None),
            default=None
        )
        best_s = f"rank {best_rank+1}" if best_rank is not None else "MISS"

        print(f"  {cam:<14} {std_s:>10} {alt_s(0.9):>8} "
              f"{alt_s(0.7):>8} {alt_s(0.5):>8} {best_s:>10}")

    # Count how many cameras ALT gets to rank 1
    for p2 in [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]:
        rank1 = sum(1 for r in results
                    if r['alt_results'][p2]['rank'] == 0)
        missed = sum(1 for r in results
                     if r['alt_results'][p2]['rank'] is None)
        avg_cands = np.mean([len(r['alt_results'][p2]['cands']) for r in results])
        print(f"  param2={p2:.1f}: rank-1={rank1}/{len(results)}  "
              f"missed={missed}  avg_cands={avg_cands:.0f}")

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main(tiff_folder, profile_path):
    profile = load_profile(profile_path)
    tiffs = sorted(glob.glob(os.path.join(tiff_folder,"*.tif")))
    if not tiffs: print(f"No TIFFs in {tiff_folder}"); sys.exit(1)
    print(f"cv2 version: {cv2.__version__}")
    print(f"Profile: {profile_path}")
    print(f"TIFFs:   {len(tiffs)}")
    results = []
    for tiff in tiffs:
        sys.stdout.write(f"  {os.path.basename(tiff)[:42]}... ")
        sys.stdout.flush()
        res = analyze(tiff, profile)
        sr = res['std_rank']
        best = min((res['alt_results'][p]['rank'] for p in res['alt_results']
                    if res['alt_results'][p]['rank'] is not None), default=None)
        print(f"std={sr+1 if sr is not None else 'X'}  "
              f"alt_best={best+1 if best is not None else 'X'}")
        results.append(res)
    for res in results: report(res)
    summary(results)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 probe_hough_alt.py <tiff_folder> <profile_json>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
