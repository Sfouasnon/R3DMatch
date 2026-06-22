"""
rank_candidates.py — Session 12 diagnostic

For each TIFF in a measurement folder, loads the known ground-truth sphere
position from the sphere profile (verified_manual samples), runs the full
Hough candidate pool, and reports:
  - Total candidates before gates
  - Where the true sphere ranks in the pool (by accumulator)
  - What beat it and why

Usage:
    python3 rank_candidates.py <tiff_folder> <profile_json>

Example:
    python3 rank_candidates.py \\
        /Users/sfouasnon/Desktop/Test_Run/R3DMatchV2_064_053126/previews/_measurement \\
        ~/Library/Application\\ Support/R3DMatch_v2/profiles/Test_Footage_064_UnderExposedGD.json
"""

import sys
import os
import json
import math
import glob
import numpy as np
from PIL import Image
from skimage.feature import canny
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

def _load_image(path):
    img = Image.open(path)
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGB")
    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (0, 0, 0))
        bg.paste(img, mask=img.split()[3])
        img = bg
    return img

def _interior_mask(h, w, cx, cy, r):
    ys, xs = np.ogrid[:h, :w]
    return np.sqrt((xs - cx)**2 + (ys - cy)**2) <= r

def _std_at(gray, cx, cy, r):
    mask = _interior_mask(*gray.shape[:2], cx, cy, r)
    px = gray[mask]
    return float(px.std()) if len(px) > 0 else 1.0

def _edge_ring_continuity(edges, cx, cy, r, n_samples=72):
    """
    Fraction of circumference points that land on an edge pixel.
    A real closed circle scores high. Arc fragments score low.
    """
    h, w = edges.shape
    hits = 0
    for i in range(n_samples):
        angle = 2 * math.pi * i / n_samples
        px = int(round(cx + r * math.cos(angle)))
        py = int(round(cy + r * math.sin(angle)))
        if 0 <= px < w and 0 <= py < h:
            if edges[py, px]:
                hits += 1
    return hits / n_samples

# ---------------------------------------------------------------------------
# Photometric pre-screen — R/G and B/G ratio check
# ---------------------------------------------------------------------------
# Calibrated from analyze_neutrality.py across 18 verified sphere solves
# (_064 and _067, 2.7–25 IRE range):
#   R/G: mean=1.10, min=1.03, max=1.14  → gate: 0.90–1.25 (generous margin)
#   B/G: mean=1.00, min=0.94, max=1.09  → gate: 0.80–1.20
# These bounds are wide enough to tolerate per-camera WB variation while
# rejecting colored tape marks, equipment panels, and LED fixtures.
_PHOTO_RG_MIN = 0.90
_PHOTO_RG_MAX = 1.25
_PHOTO_BG_MIN = 0.80
_PHOTO_BG_MAX = 1.20
_PHOTO_SAMPLE_RADIUS = 0.50  # sample inner 50% of candidate radius

# Std floor — phantoms sampling uniform background have std < 0.008.
# True spheres across _064 and _067 have std > 0.010 at 0.5r sampling.
# This is a pre-filter floor complementing the existing G2 std ceiling.
_STD_FLOOR = 0.008

def _rgb_ratios(rgb, cx, cy, r):
    """
    Sample R/G and B/G ratios from interior disk of radius r*_PHOTO_SAMPLE_RADIUS.
    Returns (rg, bg) or (None, None) if insufficient pixels.
    """
    h, w = rgb.shape[:2]
    ys, xs = np.ogrid[:h, :w]
    dist = np.sqrt((xs - cx)**2 + (ys - cy)**2)
    mask = dist <= r * _PHOTO_SAMPLE_RADIUS
    px = rgb[mask]
    if len(px) < 10:
        return None, None
    r_mean = float(px[:, 0].mean())
    g_mean = float(px[:, 1].mean())
    b_mean = float(px[:, 2].mean())
    if g_mean < 0.005:  # too dark to measure reliably
        return None, None
    return r_mean / g_mean, b_mean / g_mean

def _passes_photometric_prescreen(rg, bg):
    """True if R/G and B/G ratios are within the sphere's expected range."""
    if rg is None or bg is None:
        return True  # too dark to measure — don't reject, let gates decide
    return (_PHOTO_RG_MIN <= rg <= _PHOTO_RG_MAX and
            _PHOTO_BG_MIN <= bg <= _PHOTO_BG_MAX)
def load_profile(profile_path):
    with open(os.path.expanduser(profile_path)) as f:
        return json.load(f)

def get_ground_truth(profile, camera_label):
    """
    Get the best verified_manual sample for a camera.
    Returns (cx_norm, cy_norm, radius_ratio) or None.
    """
    cam = profile.get("cameras", {}).get(camera_label, {})
    samples = [s for s in cam.get("samples", []) if s.get("trust") == "verified_manual"]
    if not samples:
        samples = [s for s in cam.get("samples", [])]
    if not samples:
        return None
    # Take most recent
    s = samples[-1]
    g = s.get("geometry", {})
    return g.get("cx_norm"), g.get("cy_norm"), g.get("radius_ratio")

# ---------------------------------------------------------------------------
# Camera label from filename
# e.g. G007_A064_0325RN_001... -> G007_A064
# ---------------------------------------------------------------------------
def camera_label_from_filename(fname):
    base = os.path.basename(fname)
    parts = base.split("_")
    if len(parts) >= 2:
        return f"{parts[0]}_{parts[1]}"
    return None

# ---------------------------------------------------------------------------
# Main per-TIFF analysis
# ---------------------------------------------------------------------------
def analyze_tiff(tiff_path, profile):
    camera_label = camera_label_from_filename(tiff_path)
    gt = get_ground_truth(profile, camera_label) if camera_label else None

    img = _load_image(tiff_path)
    img_det, scale = _resize_for_detection(img)
    arr = np.array(img_det, dtype=np.float32) / 255.0
    h, w = arr.shape[:2]
    gray = _to_gray(arr)

    r_min = max(6, int(w * _RADIUS_MIN_RATIO))
    r_max = min(h // 2, int(w * _RADIUS_MAX_RATIO))

    edges = canny(gray, sigma=_CANNY_SIGMA,
                  low_threshold=_CANNY_LOW, high_threshold=_CANNY_HIGH)
    rgb = arr  # full float32 HxWx3

    radii = np.arange(r_min, r_max + 1)
    hough_res = hough_circle(edges, radii)
    _, cx_arr, cy_arr, r_arr = hough_circle_peaks(
        hough_res, radii,
        min_xdistance=_HOUGH_MIN_DIST,
        min_ydistance=_HOUGH_MIN_DIST,
        num_peaks=_HOUGH_NUM_PEAKS,
        threshold=_HOUGH_ACCUMULATOR_MIN,
    )

    # Build candidate list with accumulator + ring continuity + RGB ratios
    candidates = []
    for cx, cy, r in zip(cx_arr.tolist(), cy_arr.tolist(), r_arr.tolist()):
        r_idx = np.searchsorted(radii, r)
        r_idx = np.clip(r_idx, 0, len(radii) - 1)
        acc = float(hough_res[r_idx, int(cy), int(cx)])
        continuity = _edge_ring_continuity(edges, cx, cy, r)
        std = _std_at(gray, cx, cy, r * 0.5)
        rg, bg = _rgb_ratios(rgb, cx, cy, r)
        passes_photo = _passes_photometric_prescreen(rg, bg)
        passes_std_floor = (std >= _STD_FLOOR)
        candidates.append({
            "cx": cx, "cy": cy, "r": r,
            "acc": acc,
            "continuity": continuity,
            "std": std,
            "rg": rg,
            "bg": bg,
            "passes_photo": passes_photo,
            "passes_std_floor": passes_std_floor,
            "passes_all": passes_photo and passes_std_floor,
            "r_w": r / w,
        })

    # Sort by accumulator (current pipeline order)
    candidates.sort(key=lambda c: -c["acc"])

    # Rank by combined score: acc * continuity
    candidates_combined = sorted(candidates, key=lambda c: -(c["acc"] * c["continuity"]))

    # Photometric pre-screen: only candidates passing R/G and B/G gate
    candidates_photo = [c for c in candidates if c["passes_photo"]]
    candidates_photo.sort(key=lambda c: -c["acc"])

    # Combined pre-screen: photo + std floor
    candidates_all = [c for c in candidates if c["passes_all"]]
    candidates_all.sort(key=lambda c: -c["acc"])

    # Ground truth in detection-plane coords
    gt_det = None
    if gt and all(v is not None for v in gt):
        cx_norm, cy_norm, radius_ratio = gt
        gt_det = {
            "cx": cx_norm * w,
            "cy": cy_norm * h,
            "r":  radius_ratio * w,
        }

    # Find rank of true sphere — accumulator sort
    sphere_rank = None
    sphere_dist = None
    if gt_det:
        for i, c in enumerate(candidates):
            dist = math.hypot(c["cx"] - gt_det["cx"], c["cy"] - gt_det["cy"])
            if dist < 0.5 * gt_det["r"]:
                sphere_rank = i
                sphere_dist = dist
                break

    # Find rank of true sphere — combined score sort
    sphere_rank_combined = None
    if gt_det:
        for i, c in enumerate(candidates_combined):
            dist = math.hypot(c["cx"] - gt_det["cx"], c["cy"] - gt_det["cy"])
            if dist < 0.5 * gt_det["r"]:
                sphere_rank_combined = i
                break

    # Find rank of true sphere — photometric pre-screen sort
    sphere_rank_photo = None
    if gt_det:
        for i, c in enumerate(candidates_photo):
            dist = math.hypot(c["cx"] - gt_det["cx"], c["cy"] - gt_det["cy"])
            if dist < 0.5 * gt_det["r"]:
                sphere_rank_photo = i
                break

    # Find rank of true sphere — combined filter (photo + std floor)
    sphere_rank_all = None
    if gt_det:
        for i, c in enumerate(candidates_all):
            dist = math.hypot(c["cx"] - gt_det["cx"], c["cy"] - gt_det["cy"])
            if dist < 0.5 * gt_det["r"]:
                sphere_rank_all = i
                break

    return {
        "camera": camera_label,
        "tiff": os.path.basename(tiff_path),
        "n_candidates": len(candidates),
        "n_photo": len(candidates_photo),
        "n_all": len(candidates_all),
        "candidates": candidates,
        "candidates_combined": candidates_combined,
        "candidates_photo": candidates_photo,
        "candidates_all": candidates_all,
        "gt_det": gt_det,
        "sphere_rank": sphere_rank,
        "sphere_rank_combined": sphere_rank_combined,
        "sphere_rank_photo": sphere_rank_photo,
        "sphere_rank_all": sphere_rank_all,
        "sphere_dist": sphere_dist,
        "w": w, "h": h,
    }

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def report(result):
    cam = result["camera"] or "unknown"
    n = result["n_candidates"]
    n_all = result["n_all"]
    rank = result["sphere_rank"]
    rank_a = result["sphere_rank_all"]
    gt = result["gt_det"]
    w = result["w"]

    if gt:
        gt_str = f"GT cx={gt['cx']/1:.0f} cy={gt['cy']/1:.0f} r={gt['r']/1:.0f} r/w={gt['r']/w:.3f}"
    else:
        gt_str = "NO GROUND TRUTH IN PROFILE"

    acc_str  = f"rank {rank+1}/{n}"    if rank  is not None else "MISSING"
    all_str  = f"rank {rank_a+1}/{n_all}" if rank_a is not None else "MISS/FILT"

    tag = " ✓ FIXED" if rank_a == 0 else (
          f" partial (rank {rank_a+1})" if (rank_a is not None and rank is not None and rank_a < rank) else
          " ✗" if rank_a is not None and rank_a > 0 else "")

    print(f"\n{'='*70}")
    print(f"  {cam}  |  {n} total → {n_all} pass photo+std filter")
    print(f"  acc-rank: {acc_str}   filtered-rank: {all_str}{tag}")
    print(f"  {gt_str}")
    print(f"{'='*70}")

    if rank is not None:
        sphere = result["candidates"][rank]
        rg_str = f"{sphere['rg']:.3f}" if sphere['rg'] else "n/a"
        bg_str = f"{sphere['bg']:.3f}" if sphere['bg'] else "n/a"
        print(f"  Sphere:  acc={sphere['acc']:.3f}  cont={sphere['continuity']:.2f}  "
              f"std={sphere['std']:.4f}  R/G={rg_str}  B/G={bg_str}  r/w={sphere['r_w']:.3f}  "
              f"filter={'PASS' if sphere['passes_all'] else 'FAIL(photo='+str(sphere['passes_photo'])+' std='+str(sphere['passes_std_floor'])+')'}")

    if rank_a is not None and rank_a > 0:
        print(f"\n  Still above sphere after photo+std filter (top 5):")
        for i, c in enumerate(result["candidates_all"][:rank_a][:5]):
            rg_str = f"{c['rg']:.3f}" if c['rg'] else "n/a"
            bg_str = f"{c['bg']:.3f}" if c['bg'] else "n/a"
            print(f"    [{i+1:2d}] cx={c['cx']/1:.0f} cy={c['cy']/1:.0f} "
                  f"r={c['r']/1:.0f}  acc={c['acc']:.3f}  std={c['std']:.4f}  "
                  f"R/G={rg_str}  B/G={bg_str}  r/w={c['r_w']:.3f}")
    elif rank is not None and rank > 0 and rank_a == 0:
        print(f"  → photo+std filter moves sphere to rank 1.")

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
def summary(results):
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Camera':<14} {'Total':>6} {'Filt':>6} {'Acc-Rank':>10} {'Filt-Rank':>10} {'Fixed?':>7}")
    print(f"  {'-'*14} {'-'*6} {'-'*6} {'-'*10} {'-'*10} {'-'*7}")
    for r in results:
        cam = (r["camera"] or "?")[:14]
        n = r["n_candidates"]
        na = r["n_all"]
        rank = r["sphere_rank"]
        rank_a = r["sphere_rank_all"]
        acc_str  = f"{rank+1}/{n}"    if rank   is not None else "MISSING"
        all_str  = f"{rank_a+1}/{na}" if rank_a is not None else "MISS/FILT"
        fixed = "✓" if rank_a == 0 else ("—" if rank == 0 else "✗")
        print(f"  {cam:<14} {n:>6} {na:>6} {acc_str:>10} {all_str:>10} {fixed:>7}")

    total_before = sum(r["n_candidates"] for r in results)
    total_after  = sum(r["n_all"]        for r in results)
    print(f"\n  photo+std filter: {total_before} → {total_after} candidates "
          f"({100*total_after/total_before:.0f}% survive)")

    failures_acc  = sum(1 for r in results if r["sphere_rank"]     is not None and r["sphere_rank"]     > 0)
    failures_all  = sum(1 for r in results if r["sphere_rank_all"] is None     or  r["sphere_rank_all"] > 0)
    print(f"  Acc-sort failures:          {failures_acc}/{len(results)}")
    print(f"  photo+std filter failures:  {failures_all}/{len(results)}")

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main(tiff_folder, profile_path):
    profile = load_profile(profile_path)
    tiffs = sorted(glob.glob(os.path.join(tiff_folder, "*.tif")))
    if not tiffs:
        print(f"No TIFFs found in {tiff_folder}")
        sys.exit(1)

    print(f"Profile: {profile_path}")
    print(f"TIFFs:   {len(tiffs)} found in {tiff_folder}")

    results = []
    for tiff in tiffs:
        result = analyze_tiff(tiff, profile)
        report(result)
        results.append(result)

    summary(results)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 rank_candidates.py <tiff_folder> <profile_json>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
