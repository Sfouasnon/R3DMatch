#!/usr/bin/env python3
"""
probe_rw_ceiling.py — Session 15
Validates raising _RADIUS_MAX_RATIO from 0.15 → 0.27 across all 36 TIFFs.

What this does:
  - Loads each TIFF, runs HOUGH_GRADIENT_ALT with the same params used in production
  - For every candidate circle, records r/w ratio
  - Compares which candidates pass/fail the ceiling under OLD (0.15) vs NEW (0.27)
  - Reports: newly-admitted candidates, regressions, and full r/w distribution

Usage:
  python3 probe_rw_ceiling.py

Reads from the 3 test footage directories (already rendered TIFFs).
Does NOT write to any pipeline state or profiles.
"""

import os
import sys
import json
import glob
import numpy as np
import cv2

# ---------------------------------------------------------------------------
# TIFF root — all measurement TIFFs live under Test_Run/*/previews/_measurement/
# The probe collects all TIFFs, deduplicates by clip_id (keeps the most recent
# run per clip), then probes the unique set.
# ---------------------------------------------------------------------------
TEST_RUN_ROOT = "/Users/sfouasnon/Desktop/Test_Run"

# Expected "winning" clips from Session 14 stress test (34 SUCCESS + 2 known failures)
KNOWN_NEEDS_ASSIST = {"G007_D064"}   # 8.6 IRE — correct behavior, not a regression

# ---------------------------------------------------------------------------
# Hough parameters (must match production sphere.py exactly)
# ---------------------------------------------------------------------------
_HOUGH_ALT_PARAM2  = 0.9
_HOUGH_ALT_DP      = 1.5
_HOUGH_ALT_BLUR_K  = 5

# r/w ceiling — old vs new
CEILING_OLD = 0.15
CEILING_NEW = 0.27

# r/w floor (the other known bug — NOT changing here, just noting)
FLOOR_CURRENT = 0.018   # observed min is 0.163 — floor is a separate issue

# Other pre-filter constants (kept fixed for this probe)
_PF_STD_CLEAN_MAX  = 0.020
_PF_STD_HARD_MAX   = 0.130

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_tiff_gray(path: str) -> np.ndarray:
    """Load TIFF, convert to float32 [0,1] grayscale."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(f"Cannot read: {path}")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.dtype == np.uint16:
        img = img.astype(np.float32) / 65535.0
    elif img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    else:
        img = img.astype(np.float32)
        if img.max() > 1.0:
            img /= img.max()
    return img


def run_hough_alt(gray: np.ndarray):
    """
    Run HOUGH_GRADIENT_ALT on a float32 [0,1] image.
    Returns list of (cx, cy, r) tuples.
    """
    h, w = gray.shape
    u8 = (np.clip(gray, 0, 1) * 255).astype(np.uint8)
    blurred = cv2.GaussianBlur(u8, (_HOUGH_ALT_BLUR_K, _HOUGH_ALT_BLUR_K), 0)

    min_r = max(5, int(min(h, w) * 0.02))
    max_r = int(min(h, w) * 0.50)   # wide ceiling for candidate collection

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT_ALT,
        dp=_HOUGH_ALT_DP,
        minDist=min_r * 2,
        param1=100,
        param2=_HOUGH_ALT_PARAM2,
        minRadius=min_r,
        maxRadius=max_r,
    )
    if circles is None:
        return []
    return [(float(c[0]), float(c[1]), float(c[2])) for c in circles[0]]


def rw_ratio(r: float, w: int) -> float:
    return r / w


def interior_std(gray: np.ndarray, cx: float, cy: float, r: float) -> float:
    """Compute std of pixels inside 0.5r disk."""
    h, w = gray.shape
    ys, xs = np.ogrid[:h, :w]
    inner_r = 0.5 * r
    mask = (xs - cx)**2 + (ys - cy)**2 <= inner_r**2
    pixels = gray[mask]
    if pixels.size < 10:
        return 999.0
    return float(np.std(pixels))


# ---------------------------------------------------------------------------
# Main probe
# ---------------------------------------------------------------------------

def probe_tiff(tiff_path: str, set_name: str):
    """
    Run detection on one TIFF. Returns dict with probe results.
    """
    try:
        gray = load_tiff_gray(tiff_path)
    except IOError as e:
        return {"path": tiff_path, "error": str(e)}

    h, w = gray.shape
    candidates = run_hough_alt(gray)

    results = []
    for cx, cy, r in candidates:
        ratio = rw_ratio(r, w)
        std   = interior_std(gray, cx, cy, r)

        pass_old_ceiling = ratio <= CEILING_OLD
        pass_new_ceiling = ratio <= CEILING_NEW
        pass_floor       = ratio >= FLOOR_CURRENT
        pass_std_clean   = std <= _PF_STD_CLEAN_MAX
        pass_std_hard    = std <= _PF_STD_HARD_MAX

        results.append({
            "cx": round(cx, 1), "cy": round(cy, 1), "r": round(r, 1),
            "rw": round(ratio, 4),
            "std": round(std, 5),
            "pass_old_ceiling": pass_old_ceiling,
            "pass_new_ceiling": pass_new_ceiling,
            "pass_floor": pass_floor,
            "pass_std_clean": pass_std_clean,
            "pass_std_hard": pass_std_hard,
        })

    # Sort by std (best candidates first)
    results.sort(key=lambda x: x["std"])

    clip_id = os.path.splitext(os.path.basename(tiff_path))[0]

    # Best candidate under each regime
    old_pass = [c for c in results if c["pass_old_ceiling"] and c["pass_floor"] and c["pass_std_hard"]]
    new_pass = [c for c in results if c["pass_new_ceiling"] and c["pass_floor"] and c["pass_std_hard"]]

    newly_admitted  = [c for c in new_pass if not c["pass_old_ceiling"]]   # would have been rejected before
    regression_risk = [c for c in results if c["pass_old_ceiling"] and not c["pass_new_ceiling"]]  # (shouldn't happen)

    return {
        "set": set_name,
        "clip_id": clip_id,
        "path": tiff_path,
        "image_size": f"{w}x{h}",
        "total_candidates": len(candidates),
        "old_pass_count": len(old_pass),
        "new_pass_count": len(new_pass),
        "newly_admitted": newly_admitted,
        "regression_risk": regression_risk,
        "best_old": old_pass[0] if old_pass else None,
        "best_new": new_pass[0] if new_pass else None,
        "all_candidates": results,
    }


def find_tiffs_deduplicated(root: str):
    """
    Find all measurement TIFFs under Test_Run/*/previews/_measurement/.
    Deduplicates by clip_id: for clips appearing in multiple runs, keeps the
    path from the most recently modified run directory.
    Returns: list of (clip_id, tiff_path, run_dir) sorted by clip_id.
    """
    pattern = os.path.join(root, "*", "previews", "_measurement", "*.tif")
    all_tiffs = glob.glob(pattern)
    # Also catch .tiff extension
    all_tiffs += glob.glob(pattern.replace("*.tif", "*.tiff"))

    # clip_id: strip everything after the camera/clip portion
    # Filename format: H007_B067_0403QK_001.original.analysis.measurement.tiff.000000.tif
    # clip_id = first two underscore-separated tokens = e.g. "H007_B067"
    def clip_id_from_path(p):
        fname = os.path.basename(p)
        parts = fname.split("_")
        if len(parts) >= 2:
            return f"{parts[0]}_{parts[1]}"
        return fname

    # Group by clip_id, keep most recently modified
    best = {}  # clip_id -> (mtime, path, run_dir)
    for p in all_tiffs:
        cid = clip_id_from_path(p)
        run_dir = p.split(os.sep)
        # Extract run dir name (3 levels up from file: run/previews/_measurement/file)
        run_name = run_dir[-4] if len(run_dir) >= 4 else "unknown"
        try:
            mtime = os.path.getmtime(p)
        except OSError:
            mtime = 0
        if cid not in best or mtime > best[cid][0]:
            best[cid] = (mtime, p, run_name)

    result = [(cid, info[1], info[2]) for cid, info in best.items()]
    result.sort(key=lambda x: x[0])
    return result


def main():
    print("=" * 70)
    print("probe_rw_ceiling.py — r/w ceiling validation")
    print(f"OLD ceiling: {CEILING_OLD}   NEW ceiling: {CEILING_NEW}")
    print("=" * 70)

    if not os.path.isdir(TEST_RUN_ROOT):
        print(f"\nERROR: TEST_RUN_ROOT not found: {TEST_RUN_ROOT}")
        sys.exit(1)

    tiff_list = find_tiffs_deduplicated(TEST_RUN_ROOT)
    if not tiff_list:
        print(f"\nERROR: No TIFFs found under {TEST_RUN_ROOT}/*/previews/_measurement/")
        sys.exit(1)

    print(f"\nFound {len(tiff_list)} unique clips across all run directories")

    all_results = []
    for clip_id, tiff_path, run_name in tiff_list:
        r = probe_tiff(tiff_path, run_name)
        all_results.append(r)
        nc       = r.get("newly_admitted", [])
        best_new = r.get("best_new")
        rw_str   = f"rw={best_new['rw']:.4f}" if best_new else "no_pass"
        flag     = "  *** NEWLY ADMITTED ***" if nc else ""
        print(f"  {clip_id:20s}  total={r.get('total_candidates',0):3d}  "
              f"old_pass={r.get('old_pass_count',0)}  new_pass={r.get('new_pass_count',0)}  "
              f"best:{rw_str}{flag}")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    total_clips       = len(all_results)
    newly_admitted_clips = [r for r in all_results if r.get("newly_admitted")]
    regression_clips  = [r for r in all_results if r.get("regression_risk")]

    # Collect all r/w ratios seen across all candidates
    all_rw = []
    for r in all_results:
        for c in r.get("all_candidates", []):
            all_rw.append(c["rw"])

    print(f"\nTotal clips probed:            {total_clips}")
    print(f"Clips with newly-admitted cands: {len(newly_admitted_clips)}")
    print(f"Clips with regression risk:      {len(regression_clips)}")

    if all_rw:
        arr = np.array(all_rw)
        print(f"\nAll-candidate r/w distribution:")
        print(f"  min={arr.min():.4f}  p5={np.percentile(arr,5):.4f}  "
              f"p50={np.percentile(arr,50):.4f}  p95={np.percentile(arr,95):.4f}  max={arr.max():.4f}")
        above_old = np.sum(arr > CEILING_OLD)
        above_new = np.sum(arr > CEILING_NEW)
        print(f"  Candidates above OLD ceiling ({CEILING_OLD}): {above_old}")
        print(f"  Candidates above NEW ceiling ({CEILING_NEW}): {above_new}")

    if newly_admitted_clips:
        print(f"\nCLIPS WITH NEWLY-ADMITTED CANDIDATES (review manually):")
        for r in newly_admitted_clips:
            print(f"  {r['clip_id']} ({r['set']})")
            for c in r["newly_admitted"]:
                print(f"    rw={c['rw']}  std={c['std']}  cx={c['cx']}  cy={c['cy']}  r={c['r']}")

    if regression_clips:
        print(f"\n!!! REGRESSION RISK (should be empty — old ceiling < new ceiling):")
        for r in regression_clips:
            print(f"  {r['clip_id']}: {r['regression_risk']}")
    else:
        print(f"\nRegression risk: NONE (expected — raising ceiling cannot reject existing passes)")

    # -----------------------------------------------------------------------
    # Write JSON output for further analysis
    # -----------------------------------------------------------------------
    out_path = os.path.expanduser("~/Desktop/probe_rw_ceiling_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nFull results written to: {out_path}")

    # -----------------------------------------------------------------------
    # Verdict
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("VERDICT")
    if regression_clips:
        print("FAIL — unexpected regression risk detected. Do not patch.")
    elif newly_admitted_clips:
        print("REVIEW — new candidates admitted above old ceiling.")
        print("Check each newly-admitted clip visually to confirm it's a real sphere,")
        print("not a false positive. If all look correct: safe to patch.")
    else:
        print("CLEAN — no newly admitted candidates, no regressions.")
        print("Raising ceiling to 0.27 is safe (but may not help — the spheres are")
        print("being detected by other means, and the ceiling isn't the binding gate).")
    print("=" * 70)


if __name__ == "__main__":
    main()
