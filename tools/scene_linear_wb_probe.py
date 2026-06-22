#!/usr/bin/env python3
"""
scene_linear_wb_probe.py — should we solve WHITE BALANCE in scene-linear?
==========================================================================

Question
--------
Exposure matching improved when we solved it in SCENE-LINEAR instead of display
space. Does the same hold for white balance? I.e. if we measure each camera's
neutral chroma in scene-linear (REDWideGamutRGB, linear) and solve a per-channel
gain (≈ what Kelvin/Tint do in the sensor domain), does the array's NEUTRAL land
tighter through the IPP2 display transform than the current display-referred WB
solve — and how much residual is left that a 2-DOF WB simply can't fix?

What it measures (per camera, on the gray sphere)
-------------------------------------------------
  * scene-linear neutral chroma:  rg = R/G,  bg = B/G   (the WB-actuator domain)
  * IPP2-display neutral chroma:  WC, GM                (what the eye/grade sees)
  * luminance-band chroma drift across the sphere       (diagonal-can't-fix signal)

Array analysis
--------------
  1. Uncorrected spread of display WC/GM  — the problem size, in the delivered look.
  2. Solve the per-channel gain that drives each camera's linear neutral to the
     array median (the ideal scene-linear WB match) and report its size in stops.
  3. Regress display (WC,GM) on linear (rg,bg) across the array. If display chroma
     is well-explained by linear neutral chroma (high R^2), then collapsing the
     linear neutral to the median collapses the display neutral too — so the
     PREDICTED post-match display spread ≈ the regression residual. That residual
     is the part NOT fixable by any neutral/WB correction (spectral + IPP2 3D
     transform) — the gap a per-camera 3x3 / CDL would have to close.
  4. Luminance-band drift: if neutral chroma drifts with luminance on the sphere,
     a single diagonal gain (and thus Kelvin/Tint) cannot hold it neutral — an
     independent flag that the residual needs a matrix, not WB.

Verdict
-------
  * Large uncorrected display spread + SMALL regression residual + SMALL band
    drift  → scene-linear WB will tighten the neutral; worth wiring into solve.
  * Residual or band drift comparable to the spread → WB can't fix it; the gain
    is in the parked per-camera CDL/3x3 path, not in re-solving WB.

INPUT MODES
-----------
  # A) analyze an existing run's measurement folder (no REDLine, fastest):
  python3 scene_linear_wb_probe.py --measurement-dir \
      "/path/to/<run>/previews/_measurement"

  # B) render fresh from a card folder (needs REDLine + footage):
  python3 scene_linear_wb_probe.py /path/to/CardFolder
  python3 scene_linear_wb_probe.py --reuse /path/to/CardFolder   # reuse renders

Read-only w.r.t. the solver and your profiles. Output: ./scene_linear_wb_out/.
"""
from __future__ import annotations
import argparse
import csv
import math
import os
import sys
from pathlib import Path


def _bootstrap_src(explicit):
    cands = []
    if explicit:
        cands.append(Path(explicit).expanduser())
    if os.environ.get("R3DMATCH_SRC"):
        cands.append(Path(os.environ["R3DMATCH_SRC"]).expanduser())
    here = Path(__file__).resolve()
    for up in [here.parent, *here.parents]:
        cands.append(up / "src")
        if (up / "r3dmatch3").is_dir():
            cands.append(up)
    cands.append(Path.home() / "Desktop" / "R3DMatch_v4" / "src")
    for p in cands:
        if (p / "r3dmatch3" / "sphere.py").is_file():
            return p
    sys.exit("ERROR: could not find r3dmatch3. Pass --src /path/to/R3DMatch_v4/src.")


# ---------------------------------------------------------------------------
def _load_rgb(path, np, cv2):
    """16-bit (or 8-bit) image -> float RGB 0..1, HxWx3."""
    im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if im is None:
        return None
    if im.ndim == 2:
        im = np.stack([im, im, im], axis=-1)
    maxv = 65535.0 if im.dtype != "uint8" else 255.0
    rgb = im[..., ::-1].astype(np.float32) / maxv          # BGR->RGB, normalize
    return rgb


def _interior_samples(rgb, gray, cx, cy, r, np, frac=0.6):
    """Return Nx3 RGB and N luminance for pixels inside frac*r of the sphere."""
    h, w = gray.shape
    y0, y1 = max(0, int(cy - r)), min(h, int(cy + r) + 1)
    x0, x1 = max(0, int(cx - r)), min(w, int(cx + r) + 1)
    yy, xx = np.ogrid[y0:y1, x0:x1]
    m = ((xx - cx) ** 2 + (yy - cy) ** 2) <= (r * frac) ** 2
    sub = rgb[y0:y1, x0:x1]
    px = sub[m]                         # Nx3
    lum = 0.2126 * px[:, 0] + 0.7152 * px[:, 1] + 0.0722 * px[:, 2]
    return px, lum


def _robust_mean_rgb(px, np):
    """Per-channel median (specular-robust) neutral estimate."""
    return np.array([float(np.median(px[:, c])) for c in range(3)], dtype=np.float64)


def _ratios(rgb_mean):
    g = rgb_mean[1] if rgb_mean[1] > 1e-9 else 1e-9
    return rgb_mean[0] / g, rgb_mean[2] / g     # R/G, B/G


# ---------------------------------------------------------------------------
def _gather_from_measurement_dir(mdir, S, np, cv2):
    """Pair *.linear.measurement.* with *.original.analysis.* per camera."""
    mdir = Path(mdir).expanduser()
    disp = {}
    lin = {}
    for f in mdir.iterdir():
        n = f.name
        key = n.split("_001.")[0] if "_001." in n else n.split(".")[0]
        if "linear.measurement" in n:
            lin[key] = f
        elif "original.analysis" in n:
            disp[key] = f
    rows = []
    for key in sorted(set(disp) & set(lin)):
        rows.append((key, disp[key], lin[key]))
    if not rows:
        sys.exit(f"No linear+display TIFF pairs found in {mdir}")
    return rows


def _measure_pair(key, disp_path, lin_path, S, np, cv2, nbands=5):
    rgb_d = _load_rgb(disp_path, np, cv2)
    rgb_l = _load_rgb(lin_path, np, cv2)
    if rgb_d is None or rgb_l is None:
        return None
    # detect the sphere on the display frame (downscaled), scale ROI to full res
    from PIL import Image
    pil = Image.fromarray((np.clip(rgb_d, 0, 1) * 255).astype("uint8"))
    det, scale = S._resize_for_detection(pil)
    drgb = (np.asarray(det, dtype=np.float32)) / 255.0
    r = S.detect_sphere(drgb.astype("float32"), clip_id=key, peer_ire_values=[45, 46, 47])
    if not getattr(r, "roi", None):
        return None
    inv = 1.0 / (scale if scale else 1.0)
    cx, cy, rad = r.roi.cx * inv, r.roi.cy * inv, r.roi.r * inv

    gray_d = 0.2126 * rgb_d[..., 0] + 0.7152 * rgb_d[..., 1] + 0.0722 * rgb_d[..., 2]
    px_d, _ = _interior_samples(rgb_d, gray_d, cx, cy, rad, np)
    px_l, lum_l = _interior_samples(rgb_l, gray_d, cx, cy, rad, np)
    if len(px_d) < 200 or len(px_l) < 200:
        return None

    mean_d = _robust_mean_rgb(px_d, np)
    mean_l = _robust_mean_rgb(px_l, np)
    rg, bg = _ratios(mean_l)

    # luminance-band chroma drift in LINEAR (does neutral chroma move with level?)
    order = np.argsort(lum_l)
    bands = np.array_split(order, nbands)
    band_rg, band_bg = [], []
    for b in bands:
        if len(b) < 30:
            continue
        m = _robust_mean_rgb(px_l[b], np)
        r_, b_ = _ratios(m)
        band_rg.append(r_); band_bg.append(b_)
    drift_rg = (max(band_rg) - min(band_rg)) if band_rg else 0.0
    drift_bg = (max(band_bg) - min(band_bg)) if band_bg else 0.0

    return {
        "camera": key, "rg": rg, "bg": bg,
        "disp_rgb": tuple(mean_d), "lin_rgb": tuple(mean_l),
        "drift_rg": drift_rg, "drift_bg": drift_bg,
    }


def _fit_predict_residual(x1, x2, y, np):
    """OLS y ~ 1 + x1 + x2 ; return R^2 and residual std (predicted post-match spread)."""
    X = np.column_stack([np.ones_like(x1), x1, x2])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ beta
    resid = y - pred
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 1.0
    return r2, float(np.std(resid))


# ---------------------------------------------------------------------------
def _main():
    ap = argparse.ArgumentParser(description="scene-linear WB validation probe")
    ap.add_argument("footage", nargs="?", default="")
    ap.add_argument("--measurement-dir", default="",
                    help="analyze an existing run's previews/_measurement folder")
    ap.add_argument("--src", default="")
    ap.add_argument("--redline", default="")
    ap.add_argument("--reuse", action="store_true")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    src = _bootstrap_src(args.src or None)
    sys.path.insert(0, str(src))
    import numpy as np
    import cv2
    from r3dmatch3 import sphere as S
    from r3dmatch3.measure import compute_wc_gm

    out_dir = Path(args.out).expanduser() if args.out else (Path(__file__).resolve().parent / "scene_linear_wb_out")
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- gather measurements -------------------------------------------------
    if args.measurement_dir:
        pairs = _gather_from_measurement_dir(args.measurement_dir, S, np, cv2)
    else:
        if not args.footage:
            sys.exit("Provide --measurement-dir DIR, or a footage folder to render.")
        pairs = _render_pairs(args.footage, args.redline, args.reuse, out_dir, S, np, cv2)

    cams = []
    print(f"{'camera':20s} {'lin R/G':>8s} {'lin B/G':>8s}   {'disp WC':>8s} {'disp GM':>8s}   {'band drift rg/bg':>18s}")
    print("-" * 86)
    for (key, dpath, lpath) in pairs:
        m = _measure_pair(key, dpath, lpath, S, np, cv2)
        if not m:
            print(f"{key[:20]:20s}  (sphere not found / too few samples)")
            continue
        wc, gm = compute_wc_gm(m["disp_rgb"])
        m["wc"], m["gm"] = wc, gm
        cams.append(m)
        print(f"{m['camera'][:20]:20s} {m['rg']:8.4f} {m['bg']:8.4f}   {wc:8.4f} {gm:8.4f}   "
              f"{m['drift_rg']:7.4f}/{m['drift_bg']:.4f}")

    if len(cams) < 4:
        sys.exit("\nNeed >=4 cameras for the array analysis.")

    rg = np.array([c["rg"] for c in cams]); bg = np.array([c["bg"] for c in cams])
    wc = np.array([c["wc"] for c in cams]); gm = np.array([c["gm"] for c in cams])
    med_rg, med_bg = float(np.median(rg)), float(np.median(bg))

    # ideal scene-linear WB gains to the array median (per channel, in stops)
    gains_R = med_rg / rg            # gain on R relative to G
    gains_B = med_bg / bg            # gain on B relative to G
    gR_stops = np.abs(np.log2(gains_R)); gB_stops = np.abs(np.log2(gains_B))

    # display neutral spread BEFORE any correction (what current "match" mode targets)
    disp_wc_spread = float(np.std(wc)); disp_gm_spread = float(np.std(gm))
    disp_wc_pp = float(wc.max() - wc.min()); disp_gm_pp = float(gm.max() - gm.min())

    # how well does linear neutral chroma EXPLAIN display neutral chroma?
    r2_wc, resid_wc = _fit_predict_residual(rg, bg, wc, np)
    r2_gm, resid_gm = _fit_predict_residual(rg, bg, gm, np)

    lin_rg_spread = float(np.std(rg)); lin_bg_spread = float(np.std(bg))
    drift_rg = float(np.median([c["drift_rg"] for c in cams]))
    drift_bg = float(np.median([c["drift_bg"] for c in cams]))

    print("\nARRAY ANALYSIS")
    print(f"  Linear neutral spread (std):     R/G {lin_rg_spread:.4f}   B/G {lin_bg_spread:.4f}")
    print(f"  Ideal scene-linear WB gain:      R {gR_stops.mean():.3f} stops (max {gR_stops.max():.3f}),"
          f"  B {gB_stops.mean():.3f} stops (max {gB_stops.max():.3f})")
    print(f"  Display neutral spread BEFORE:   WC std {disp_wc_spread:.4f} (pk-pk {disp_wc_pp:.4f}),"
          f"  GM std {disp_gm_spread:.4f} (pk-pk {disp_gm_pp:.4f})")
    print(f"  Display ~ f(linear neutral):     WC  R^2 {r2_wc:.3f},  GM  R^2 {r2_gm:.3f}")
    print(f"  PREDICTED display spread AFTER    "
          f"scene-linear WB match:  WC std {resid_wc:.4f},  GM std {resid_gm:.4f}")
    print(f"  Sphere luminance-band drift:     R/G {drift_rg:.4f}   B/G {drift_bg:.4f}  (median across cams)")

    # verdict heuristics
    wc_gain = (disp_wc_spread - resid_wc) / disp_wc_spread if disp_wc_spread > 1e-6 else 0.0
    gm_gain = (disp_gm_spread - resid_gm) / disp_gm_spread if disp_gm_spread > 1e-6 else 0.0
    drift_big = (drift_rg > 0.02 or drift_bg > 0.02)
    print("\nVERDICT")
    if max(disp_wc_spread, disp_gm_spread) < 0.004:
        print("  • The array is already near-neutral in the delivered look (display spread tiny).")
        print("    Little WB matching headroom on THIS set — re-run on a set with visible cast.")
    elif (wc_gain > 0.5 or gm_gain > 0.5) and not drift_big:
        print(f"  • Scene-linear WB is predicted to remove ~{100*max(wc_gain,gm_gain):.0f}% of the display")
        print("    neutral spread, and band drift is small. → Worth wiring a scene-linear WB solve.")
    else:
        print("  • A neutral/WB correction leaves a large residual (low R^2 or high band drift):")
        print("    the remaining spread is spectral + the IPP2 3D transform, which Kelvin/Tint cannot")
        print("    fix. → The gain is in a per-camera 3x3 / CDL (parked), not in re-solving WB.")
    print("  Note: a sphere can only test the NEUTRAL. Saturated-color residual needs a chart.")

    # CSV
    csv_path = out_dir / "scene_linear_wb.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["camera", "lin_RG", "lin_BG", "disp_WC", "disp_GM",
                    "gainR_stops", "gainB_stops", "band_drift_rg", "band_drift_bg"])
        for i, c in enumerate(cams):
            w.writerow([c["camera"], f"{c['rg']:.5f}", f"{c['bg']:.5f}",
                        f"{c['wc']:.5f}", f"{c['gm']:.5f}",
                        f"{gR_stops[i]:.4f}", f"{gB_stops[i]:.4f}",
                        f"{c['drift_rg']:.5f}", f"{c['drift_bg']:.5f}"])
    print(f"\nWrote {csv_path}")


def _render_pairs(footage, redline_arg, reuse, out_dir, S, np, cv2):
    """Render scene-linear + IPP2-display measurement frames per clip via REDLine."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from r3dmatch3.redline import (resolve_redline_executable, render_measurement_frame)
    from r3dmatch3.colorpipeline import REFERENCE_PIPELINE, SCENE_LINEAR_PIPELINE
    foot = Path(footage).expanduser()
    clips = sorted(foot.rglob("*_001.R3D")) or sorted(foot.rglob("*.R3D"))
    if not clips:
        sys.exit(f"No .R3D under {foot}")
    redline = ""
    pairs = []
    for clip in clips:
        key = clip.stem
        dpath = out_dir / f"{key}__disp.tiff"
        lpath = out_dir / f"{key}__lin.tiff"
        if not (reuse and dpath.exists() and lpath.exists()):
            if not redline:
                redline = resolve_redline_executable(redline_arg or "")
            render_measurement_frame(str(clip), str(dpath), redline=redline,
                                     color_pipeline=REFERENCE_PIPELINE, use_as_shot=True)
            render_measurement_frame(str(clip), str(lpath), redline=redline,
                                     color_pipeline=SCENE_LINEAR_PIPELINE, use_as_shot=True)
        pairs.append((key, dpath, lpath))
    return pairs


if __name__ == "__main__":
    _main()
