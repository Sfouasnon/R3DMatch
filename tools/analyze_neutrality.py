"""
analyze_neutrality.py — Session 12

Loads every verified_manual sphere solve from the sphere profile, samples
RGB values from the known ROI across the full sphere interior, and reports
the neutrality signature: per-channel means, chromaticity, R/G and B/G ratios,
and how these vary across cameras, exposure levels, and footage sets.

Goal: understand whether color neutrality is consistent enough to use as a
stronger pre-filter or ranking signal in sphere.py.

Usage:
    python3 analyze_neutrality.py \
        <tiff_folder_064> <profile_064> \
        <tiff_folder_067> <profile_067>

Example:
    python3 analyze_neutrality.py \\
        /Users/sfouasnon/Desktop/Test_Run/R3DMatchV2_064_053126/previews/_measurement \\
        ~/Library/Application\\ Support/R3DMatch_v2/profiles/Test_Footage_064_UnderExposedGD.json \\
        /Users/sfouasnon/Desktop/Test_Run/R3DMatchV2_067_Run_053126/previews/_measurement \\
        ~/Library/Application\\ Support/R3DMatch_v2/profiles/Test_Footage_F5.6.json
"""

import sys
import os
import json
import glob
import math
import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_DETECTION_TARGET_DIM = 1080
_SAMPLE_RINGS = [0.15, 0.30, 0.45, 0.60, 0.75, 0.90]  # fraction of r
_RING_WIDTH   = 0.07  # half-width of each annular ring as fraction of r

# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------
def _load_image(path):
    img = Image.open(path)
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGB")
    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (0, 0, 0))
        bg.paste(img, mask=img.split()[3])
        img = bg
    return img

def _resize_for_detection(img):
    w, h = img.size
    scale = _DETECTION_TARGET_DIM / max(w, h)
    if scale >= 1.0:
        return img, 1.0
    img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return img, scale

def _interior_mask(h, w, cx, cy, r):
    ys, xs = np.ogrid[:h, :w]
    return np.sqrt((xs - cx)**2 + (ys - cy)**2) <= r

def _ring_mask(h, w, cx, cy, r_inner, r_outer):
    ys, xs = np.ogrid[:h, :w]
    dist = np.sqrt((xs - cx)**2 + (ys - cy)**2)
    return (dist >= r_inner) & (dist <= r_outer)

def _chroma_dist(r_mean, g_mean, b_mean):
    total = r_mean + g_mean + b_mean
    if total < 1e-6:
        return 999.0
    rc = r_mean / total
    gc = g_mean / total
    return math.hypot(rc - 1/3, gc - 1/3)

# ---------------------------------------------------------------------------
# Profile helpers
# ---------------------------------------------------------------------------
def load_profile(path):
    with open(os.path.expanduser(path)) as f:
        return json.load(f)

def get_verified_manual(profile):
    """Return list of (camera_label, cx_norm, cy_norm, radius_ratio)."""
    results = []
    for label, cam in profile.get("cameras", {}).items():
        samples = [s for s in cam.get("samples", [])
                   if s.get("trust") == "verified_manual"]
        if not samples:
            continue
        s = samples[-1]
        g = s.get("geometry", {})
        cx_n = g.get("cx_norm")
        cy_n = g.get("cy_norm")
        rr   = g.get("radius_ratio")
        if None not in (cx_n, cy_n, rr):
            results.append((label, cx_n, cy_n, rr))
    return results

def find_tiff(folder, camera_label):
    pattern = os.path.join(folder, f"{camera_label}_*.tif")
    matches = glob.glob(pattern)
    return matches[0] if matches else None

# ---------------------------------------------------------------------------
# Per-sphere analysis
# ---------------------------------------------------------------------------
def analyze_sphere(tiff_path, cx_norm, cy_norm, radius_ratio, label, set_name):
    img_full = _load_image(tiff_path)
    img_det, scale = _resize_for_detection(img_full)
    arr = np.array(img_det, dtype=np.float32) / 255.0
    h, w = arr.shape[:2]

    cx = cx_norm * w
    cy = cy_norm * h
    r  = radius_ratio * w

    # Full interior sample (0–1r)
    mask_full = _interior_mask(h, w, cx, cy, r)
    px_full   = arr[mask_full]  # Nx3
    if len(px_full) < 10:
        return None

    r_mean = float(px_full[:, 0].mean())
    g_mean = float(px_full[:, 1].mean())
    b_mean = float(px_full[:, 2].mean())
    lum    = 0.2126 * px_full[:, 0] + 0.7152 * px_full[:, 1] + 0.0722 * px_full[:, 2]
    ire    = float(np.median(lum)) * 100.0

    chroma = _chroma_dist(r_mean, g_mean, b_mean)
    rg_ratio = r_mean / g_mean if g_mean > 1e-6 else 0
    bg_ratio = b_mean / g_mean if g_mean > 1e-6 else 0

    # Per-ring breakdown (center → edge)
    rings = []
    for ring_frac in _SAMPLE_RINGS:
        r_inner = max(0, (ring_frac - _RING_WIDTH) * r)
        r_outer = (ring_frac + _RING_WIDTH) * r
        mask_r  = _ring_mask(h, w, cx, cy, r_inner, r_outer)
        px_r    = arr[mask_r]
        if len(px_r) < 4:
            rings.append(None)
            continue
        rm = float(px_r[:, 0].mean())
        gm = float(px_r[:, 1].mean())
        bm = float(px_r[:, 2].mean())
        lm = 0.2126 * rm + 0.7152 * gm + 0.0722 * bm
        cd = _chroma_dist(rm, gm, bm)
        rings.append({
            "frac": ring_frac,
            "r": rm, "g": gm, "b": bm,
            "lum": lm,
            "chroma_dist": cd,
            "rg": rm / gm if gm > 1e-6 else 0,
            "bg": bm / gm if gm > 1e-6 else 0,
        })

    return {
        "label": label,
        "set": set_name,
        "ire": ire,
        "r": r_mean, "g": g_mean, "b": b_mean,
        "chroma_dist": chroma,
        "rg_ratio": rg_ratio,
        "bg_ratio": bg_ratio,
        "r_w": r / w,
        "rings": rings,
    }

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def report_full(all_results):
    for res in all_results:
        print(f"\n  {res['label']} [{res['set']}]  "
              f"IRE={res['ire']:.1f}  r/w={res['r_w']:.3f}")
        print(f"    Full interior:  R={res['r']:.4f}  G={res['g']:.4f}  B={res['b']:.4f}  "
              f"R/G={res['rg_ratio']:.4f}  B/G={res['bg_ratio']:.4f}  "
              f"chroma_dist={res['chroma_dist']:.4f}")
        print(f"    Ring breakdown (center→edge):")
        for ring in res["rings"]:
            if ring is None:
                continue
            print(f"      r={ring['frac']:.2f}:  "
                  f"lum={ring['lum']:.4f}  "
                  f"R={ring['r']:.4f}  G={ring['g']:.4f}  B={ring['b']:.4f}  "
                  f"R/G={ring['rg']:.4f}  B/G={ring['bg']:.4f}  "
                  f"chroma={ring['chroma_dist']:.4f}")

def report_summary(all_results):
    print(f"\n{'='*70}")
    print(f"  NEUTRALITY SUMMARY ACROSS ALL VERIFIED SOLVES")
    print(f"{'='*70}")

    for set_name in ["_064", "_067"]:
        subset = [r for r in all_results if r["set"] == set_name]
        if not subset:
            continue
        chroma  = [r["chroma_dist"] for r in subset]
        rg      = [r["rg_ratio"]    for r in subset]
        bg      = [r["bg_ratio"]    for r in subset]
        ires    = [r["ire"]         for r in subset]
        print(f"\n  {set_name} ({len(subset)} cameras):")
        print(f"    IRE range:        {min(ires):.1f} – {max(ires):.1f}")
        print(f"    Chroma dist:      mean={np.mean(chroma):.4f}  "
              f"min={np.min(chroma):.4f}  max={np.max(chroma):.4f}")
        print(f"    R/G ratio:        mean={np.mean(rg):.4f}  "
              f"min={np.min(rg):.4f}  max={np.max(rg):.4f}")
        print(f"    B/G ratio:        mean={np.mean(bg):.4f}  "
              f"min={np.min(bg):.4f}  max={np.max(bg):.4f}")

    # Combined
    chroma_all = [r["chroma_dist"] for r in all_results]
    rg_all     = [r["rg_ratio"]    for r in all_results]
    bg_all     = [r["bg_ratio"]    for r in all_results]
    print(f"\n  COMBINED ({len(all_results)} cameras across both sets):")
    print(f"    Chroma dist:      mean={np.mean(chroma_all):.4f}  "
          f"min={np.min(chroma_all):.4f}  max={np.max(chroma_all):.4f}")
    print(f"    R/G ratio:        mean={np.mean(rg_all):.4f}  "
          f"min={np.min(rg_all):.4f}  max={np.max(rg_all):.4f}")
    print(f"    B/G ratio:        mean={np.mean(bg_all):.4f}  "
          f"min={np.min(bg_all):.4f}  max={np.max(bg_all):.4f}")

    # Ring consistency — does neutrality hold across the sphere surface?
    print(f"\n  RING CONSISTENCY (chroma_dist center→edge, across all cameras):")
    print(f"  {'Ring':>6}  {'mean':>7}  {'min':>7}  {'max':>7}  {'range':>7}")
    for i, frac in enumerate(_SAMPLE_RINGS):
        ring_chroma = []
        for r in all_results:
            ring = r["rings"][i]
            if ring:
                ring_chroma.append(ring["chroma_dist"])
        if ring_chroma:
            print(f"  {frac:.2f}r  "
                  f"{np.mean(ring_chroma):>7.4f}  "
                  f"{np.min(ring_chroma):>7.4f}  "
                  f"{np.max(ring_chroma):>7.4f}  "
                  f"{np.max(ring_chroma)-np.min(ring_chroma):>7.4f}")

    # Key question: what chroma_dist threshold would cleanly separate sphere from non-sphere?
    print(f"\n  KEY QUESTION — max chroma_dist observed on true sphere: "
          f"{max(chroma_all):.4f}")
    print(f"  Current gate threshold (_GATE_CHROMA_MAX_DISTANCE): 0.0450")
    if max(chroma_all) < 0.045:
        print(f"  → All true spheres pass the current gate. Gate is not the problem.")
    else:
        print(f"  → Some true spheres EXCEED the current gate threshold. Gate too tight.")

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main(tiff_064, profile_064, tiff_067, profile_067):
    all_results = []

    for tiff_folder, profile_path, set_name in [
        (tiff_064, profile_064, "_064"),
        (tiff_067, profile_067, "_067"),
    ]:
        profile = load_profile(profile_path)
        solves  = get_verified_manual(profile)
        print(f"\n{set_name}: {len(solves)} verified_manual solves found in profile")

        for label, cx_n, cy_n, rr in sorted(solves):
            tiff = find_tiff(tiff_folder, label)
            if not tiff:
                print(f"  {label}: TIFF not found in {tiff_folder}")
                continue
            res = analyze_sphere(tiff, cx_n, cy_n, rr, label, set_name)
            if res:
                all_results.append(res)
            else:
                print(f"  {label}: insufficient pixels in ROI")

    print(f"\n{'='*70}")
    print(f"  PER-CAMERA DETAIL")
    print(f"{'='*70}")
    report_full(all_results)
    report_summary(all_results)

if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: python3 analyze_neutrality.py "
              "<tiff_064> <profile_064> <tiff_067> <profile_067>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
