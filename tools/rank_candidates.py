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
# Profile loader
# ---------------------------------------------------------------------------
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
    radii = np.arange(r_min, r_max + 1)
    hough_res = hough_circle(edges, radii)
    _, cx_arr, cy_arr, r_arr = hough_circle_peaks(
        hough_res, radii,
        min_xdistance=_HOUGH_MIN_DIST,
        min_ydistance=_HOUGH_MIN_DIST,
        num_peaks=_HOUGH_NUM_PEAKS,
        threshold=_HOUGH_ACCUMULATOR_MIN,
    )

    # Build candidate list with accumulator + ring continuity
    candidates = []
    for cx, cy, r in zip(cx_arr.tolist(), cy_arr.tolist(), r_arr.tolist()):
        r_idx = np.searchsorted(radii, r)
        r_idx = np.clip(r_idx, 0, len(radii) - 1)
        acc = float(hough_res[r_idx, int(cy), int(cx)])
        continuity = _edge_ring_continuity(edges, cx, cy, r)
        std = _std_at(gray, cx, cy, r * 0.5)
        candidates.append({
            "cx": cx, "cy": cy, "r": r,
            "acc": acc,
            "continuity": continuity,
            "std": std,
            "r_w": r / w,
        })

    # Sort by accumulator (current pipeline order)
    candidates.sort(key=lambda c: -c["acc"])

    # Ground truth in detection-plane coords
    gt_det = None
    if gt and all(v is not None for v in gt):
        cx_norm, cy_norm, radius_ratio = gt
        gt_det = {
            "cx": cx_norm * w,
            "cy": cy_norm * h,
            "r":  radius_ratio * w,
        }

    # Find rank of true sphere
    sphere_rank = None
    sphere_dist = None
    if gt_det:
        for i, c in enumerate(candidates):
            dist = math.hypot(c["cx"] - gt_det["cx"], c["cy"] - gt_det["cy"])
            if dist < 0.5 * gt_det["r"]:
                sphere_rank = i
                sphere_dist = dist
                break

    return {
        "camera": camera_label,
        "tiff": os.path.basename(tiff_path),
        "n_candidates": len(candidates),
        "candidates": candidates,
        "gt_det": gt_det,
        "sphere_rank": sphere_rank,
        "sphere_dist": sphere_dist,
        "w": w, "h": h,
    }

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def report(result):
    cam = result["camera"] or "unknown"
    n = result["n_candidates"]
    rank = result["sphere_rank"]
    gt = result["gt_det"]
    w = result["w"]

    if gt:
        gt_str = f"GT cx={gt['cx']/1:.0f} cy={gt['cy']/1:.0f} r={gt['r']/1:.0f} r/w={gt['r']/w:.3f}"
    else:
        gt_str = "NO GROUND TRUTH IN PROFILE"

    if rank is None:
        rank_str = "SPHERE NOT IN CANDIDATE POOL"
    else:
        rank_str = f"rank {rank+1}/{n}"

    print(f"\n{'='*70}")
    print(f"  {cam}  |  {n} candidates  |  sphere {rank_str}")
    print(f"  {gt_str}")
    print(f"{'='*70}")

    if rank is not None:
        sphere = result["candidates"][rank]
        print(f"  Sphere candidate:  acc={sphere['acc']:.3f}  "
              f"continuity={sphere['continuity']:.2f}  "
              f"std={sphere['std']:.4f}  r/w={sphere['r_w']:.3f}")

        if rank > 0:
            print(f"\n  Candidates ranked above sphere:")
            for i, c in enumerate(result["candidates"][:rank]):
                print(f"    [{i+1:2d}] cx={c['cx']/1:.0f} cy={c['cy']/1:.0f} "
                      f"r={c['r']/1:.0f}  acc={c['acc']:.3f}  "
                      f"continuity={c['continuity']:.2f}  "
                      f"std={c['std']:.4f}  r/w={c['r_w']:.3f}")
    elif gt:
        print(f"\n  Top 5 candidates (sphere not found):")
        for i, c in enumerate(result["candidates"][:5]):
            print(f"    [{i+1:2d}] cx={c['cx']/1:.0f} cy={c['cy']/1:.0f} "
                  f"r={c['r']/1:.0f}  acc={c['acc']:.3f}  "
                  f"continuity={c['continuity']:.2f}  "
                  f"std={c['std']:.4f}  r/w={c['r_w']:.3f}")

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
def summary(results):
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Camera':<14} {'Cands':>6} {'Rank':>8} {'Acc':>6} {'Cont':>6} {'Std':>7}")
    print(f"  {'-'*14} {'-'*6} {'-'*8} {'-'*6} {'-'*6} {'-'*7}")
    for r in results:
        cam = (r["camera"] or "?")[:14]
        n = r["n_candidates"]
        rank = r["sphere_rank"]
        if rank is not None:
            c = r["candidates"][rank]
            rank_str = f"{rank+1}/{n}"
            acc_str = f"{c['acc']:.3f}"
            cont_str = f"{c['continuity']:.2f}"
            std_str = f"{c['std']:.4f}"
        else:
            rank_str = "MISSING"
            acc_str = cont_str = std_str = "—"
        print(f"  {cam:<14} {n:>6} {rank_str:>8} {acc_str:>6} {cont_str:>6} {std_str:>7}")

    # Continuity analysis across all candidates
    all_cands = [c for r in results for c in r["candidates"]]
    sphere_cands = [r["candidates"][r["sphere_rank"]]
                    for r in results if r["sphere_rank"] is not None]
    non_sphere = [c for r in results
                  for i, c in enumerate(r["candidates"])
                  if r["sphere_rank"] is not None and i != r["sphere_rank"]]

    if sphere_cands and non_sphere:
        print(f"\n  Continuity — sphere candidates:     "
              f"mean={np.mean([c['continuity'] for c in sphere_cands]):.3f}  "
              f"min={np.min([c['continuity'] for c in sphere_cands]):.3f}  "
              f"max={np.max([c['continuity'] for c in sphere_cands]):.3f}")
        print(f"  Continuity — non-sphere candidates: "
              f"mean={np.mean([c['continuity'] for c in non_sphere]):.3f}  "
              f"min={np.min([c['continuity'] for c in non_sphere]):.3f}  "
              f"max={np.max([c['continuity'] for c in non_sphere]):.3f}")

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
