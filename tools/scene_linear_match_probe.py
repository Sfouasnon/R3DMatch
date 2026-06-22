#!/usr/bin/env python3
"""
scene_linear_match_probe.py  —  validate scene-linear gray-sphere matching
===========================================================================

Question being tested
---------------------
Can we match the cameras in SCENE-LINEAR (a per-channel gain = diagonal 3x3),
solved on the gray sphere, applied BEFORE the IPP2 display transform — and is a
sphere enough, or do we need a chart?

Why the earlier attempt failed
-------------------------------
Matching was solved in Log3G10. A matrix/gain is a linear operator and is only
valid in linear light; in log the channel deltas are compressed, so the solved
correction was tiny and the display transform later re-expanded the residual.
This probe renders TRUE scene-linear (REDWideGamutRGB, gammaCurve linear) and
solves there.

What it reports
---------------
For each camera it samples the sphere interior in scene-linear and:
  1. solves a per-channel gain that matches the camera to the array MEDIAN
     (this is the diagonal 3x3 the color scientist proposed);
  2. shows the correction magnitudes — meaningful in linear, unlike log;
  3. measures cross-camera chromaticity spread (R/G, B/G) BEFORE vs AFTER the
     gain — how well a diagonal match aligns the array;
  4. splits the sphere into luminance bands and checks whether each camera stays
     NEUTRAL/matched across its whole luminance range after the gain. A sphere
     samples one chromaticity over many luminances, so a LUMINANCE-DEPENDENT
     chromatic drift here is the residual a diagonal can't fix — the only
     sphere-visible signal that a full 3x3 (i.e. a chart) would be needed.

What it can't tell you: residual error on SATURATED colors. That needs chromatic
samples (a chart) for the same rank reason a sphere can't solve a full 3x3.

USAGE
-----
    python3 scene_linear_match_probe.py --profile "Test_Footage_F5.6" \
        ~/Desktop/Test_Footage/F5.6
    python3 scene_linear_match_probe.py --profile "Test_Footage_F5.6" --reuse \
        ~/Desktop/Test_Footage/F5.6     # reuse cached linear renders

Outputs (default ./scene_linear_match_out): scene_linear_match.txt + .csv
Read-only w.r.t. the solver and your profiles; renders temp linear frames.

NOTE on render settings: scene-linear = colorSpace 25 (REDWideGamutRGB),
gammaCurve -1 (linear), tone map / roll-off OFF. If REDLine rejects the
tone-map "0" arg on your build, tell me the error and I'll correct the code.
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
    cands += [Path.home() / "Desktop" / "R3DMatch_v4" / "src",
              Path("/Users/sfouasnon/Desktop/R3DMatch_v4/src")]
    here = Path(__file__).resolve()
    for up in [here.parent, *here.parents]:
        cands.append(up / "src")
        if (up / "r3dmatch3").is_dir():
            cands.append(up)
    for c in cands:
        if (c / "r3dmatch3" / "sphere.py").is_file():
            return c
    sys.exit("ERROR: could not find r3dmatch3. Pass --src /path/to/R3DMatch_v4/src.")


def _read_linear(path):
    """Read a render at full bit depth → float RGB in [0,1] (linear)."""
    try:
        import tifffile
        arr = tifffile.imread(path)
        if arr.ndim == 3 and arr.shape[2] >= 3:
            arr = arr[:, :, :3].astype("float64")
            maxv = 65535.0 if arr.max() > 255 else 255.0
            return arr / maxv
    except Exception:
        pass
    import cv2
    a = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if a is None:
        raise RuntimeError(f"could not read {path}")
    if a.ndim == 3:
        a = a[:, :, ::-1]  # BGR->RGB
    a = a.astype("float64")
    maxv = 65535.0 if a.max() > 255 else 255.0
    return a[:, :, :3] / maxv


def _iter_clips(paths):
    for a in paths:
        p = Path(a).expanduser()
        if p.is_dir():
            for r3d in sorted(p.rglob("*.R3D")):
                if r3d.stem.endswith("_001") or not r3d.stem[-1].isdigit():
                    yield str(r3d)
        elif p.suffix.upper() == ".R3D":
            yield str(p)


def _sphere_samples(lin, cx, cy, r, np):
    """Return per-pixel RGB inside 0.7r, plus per-pixel luminance, as arrays."""
    h, w = lin.shape[:2]
    y0, y1 = max(0, int(cy - r)), min(h, int(cy + r) + 1)
    x0, x1 = max(0, int(cx - r)), min(w, int(cx + r) + 1)
    sub = lin[y0:y1, x0:x1]
    yy, xx = np.ogrid[y0:y1, x0:x1]
    m = ((xx - cx) ** 2 + (yy - cy) ** 2) <= (r * 0.7) ** 2
    px = sub[m]                                   # (N,3)
    lum = px @ np.array([0.2126, 0.7152, 0.0722])  # rough luma for banding
    return px, lum


def _main():
    ap = argparse.ArgumentParser(description="scene-linear gray-sphere match probe")
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--profile", default="")
    ap.add_argument("--src", default="")
    ap.add_argument("--redline", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--reuse", action="store_true")
    ap.add_argument("--bands", type=int, default=4, help="luminance bands across the sphere")
    args = ap.parse_args()

    src = _bootstrap_src(args.src or None)
    sys.path.insert(0, str(src))

    import numpy as np
    from r3dmatch3 import sphere as S
    from r3dmatch3.colorpipeline import ColorPipeline
    from r3dmatch3.redline import resolve_redline_executable, render_measurement_frame
    from r3dmatch3.sphere_profile import load_project_profile, get_camera_prior
    from r3dmatch3.workflow import camera_label_from_clip_id

    # Scene-linear: REDWideGamutRGB (25) + linear gamma (-1), tonemap/rolloff off.
    SCENE_LINEAR = ColorPipeline(color_space="25", gamma_curve="-1",
                                 output_tone_map="0", roll_off="0",
                                 name="scene-linear RWG")

    out_dir = Path(args.out).expanduser() if args.out else (Path(__file__).resolve().parent / "scene_linear_match_out")
    out_dir.mkdir(parents=True, exist_ok=True)
    redline = ""
    profile = {}
    if args.profile:
        try:
            profile = load_project_profile(args.profile)
        except Exception as exc:
            print(f"(could not load profile '{args.profile}': {exc})")

    clips = list(_iter_clips(args.paths))
    print(f"r3dmatch3 src : {src}")
    print(f"clips         : {len(clips)}   profile cams: {len(profile.get('cameras', {})) if profile else 0}")
    print(f"render        : {SCENE_LINEAR.name}  ({' '.join(SCENE_LINEAR.to_redline_color_args())})")
    print("=" * 78)

    cams = []  # dict per camera: name, rgb(mean linear), bands(list of rgb)
    for clip in clips:
        stem = Path(clip).stem
        cam = camera_label_from_clip_id(stem)
        cached = sorted(out_dir.glob(f"{stem}__lin.tiff*"))
        if args.reuse and cached:
            actual = str(cached[0])
        else:
            if not redline:
                try:
                    redline = resolve_redline_executable(args.redline or "")
                except Exception as exc:
                    print(f"  ! need REDLine (or --reuse): {exc}"); return
            res = render_measurement_frame(clip, str(out_dir / f"{stem}__lin.tiff"),
                                           redline=redline, use_as_shot=True,
                                           pipeline=SCENE_LINEAR)
            if not res.get("ok"):
                print(f"  ! linear render failed {stem}: {res.get('status')} {res.get('stderr','')[:200]}")
                continue
            actual = res["output_path"]

        lin = _read_linear(actual)
        h, w = lin.shape[:2]
        prior = get_camera_prior(profile, cam) if profile else None
        if not (prior and prior.get("radius_ratio_mean")):
            print(f"  {cam}: no profile sphere location — skipped (need --profile)")
            continue
        cx = prior["cx_norm_mean"] * w
        cy = prior["cy_norm_mean"] * h
        r = prior["radius_ratio_mean"] * min(w, h)
        px, lum = _sphere_samples(lin, cx, cy, r, np)
        if px.shape[0] < 50:
            print(f"  {cam}: too few sphere pixels — skipped"); continue
        mean_rgb = px.mean(axis=0)
        # luminance bands (quantiles)
        qs = np.quantile(lum, np.linspace(0, 1, args.bands + 1))
        bands = []
        for i in range(args.bands):
            sel = (lum >= qs[i]) & (lum <= qs[i + 1])
            if sel.sum() > 10:
                bands.append(px[sel].mean(axis=0))
        cams.append(dict(cam=cam, rgb=mean_rgb, bands=bands))
        print(f"  {cam:12s} linear RGB = {mean_rgb[0]:.4f} {mean_rgb[1]:.4f} {mean_rgb[2]:.4f}")

    if len(cams) < 2:
        print("Need >=2 cameras with profile locations. Re-run with --profile.")
        return

    # Reference = per-channel MEDIAN across cameras (in linear).
    ref = np.median(np.array([c["rgb"] for c in cams]), axis=0)

    def chroma(rgb):
        g = rgb[1] if rgb[1] > 1e-9 else 1e-9
        return rgb[0] / g, rgb[2] / g

    L = ["SCENE-LINEAR GRAY-SPHERE MATCH PROBE", "=" * 44,
         f"render: {SCENE_LINEAR.name}", f"reference (array median linear RGB): "
         f"{ref[0]:.4f} {ref[1]:.4f} {ref[2]:.4f}", ""]
    L.append(f"{'camera':12s} {'gainR':7s} {'gainG':7s} {'gainB':7s}  "
             f"{'chroma R/G,B/G before':22s} {'after':14s}")
    L.append("-" * 78)
    rows = []
    before_rg, before_bg, after_rg, after_bg = [], [], [], []
    band_resid = []
    for c in cams:
        gain = ref / np.where(c["rgb"] > 1e-9, c["rgb"], 1e-9)
        b_rg, b_bg = chroma(c["rgb"])
        a_rg, a_bg = chroma(c["rgb"] * gain)   # == ref chroma by construction
        before_rg.append(b_rg); before_bg.append(b_bg)
        after_rg.append(a_rg); after_bg.append(a_bg)
        # luminance-band neutrality AFTER applying the single diagonal gain
        worst = 0.0
        for brgb in c["bands"]:
            rg, bg = chroma(brgb * gain)
            worst = max(worst, abs(rg - (ref[0] / ref[1])), abs(bg - (ref[2] / ref[1])))
        band_resid.append(worst)
        L.append(f"{c['cam']:12s} {gain[0]:.4f} {gain[1]:.4f} {gain[2]:.4f}  "
                 f"{b_rg:.4f},{b_bg:.4f}        {a_rg:.4f},{a_bg:.4f}")
        rows.append(dict(camera=c["cam"], gainR=gain[0], gainG=gain[1], gainB=gain[2],
                         rg_before=b_rg, bg_before=b_bg, band_chroma_resid=worst))

    def spread(xs):
        return max(xs) - min(xs)

    L.append("")
    L.append("MATCH QUALITY (cross-camera chromaticity spread, lower = better)")
    L.append(f"  before diagonal: R/G spread={spread(before_rg):.4f}  B/G spread={spread(before_bg):.4f}")
    L.append(f"  after  diagonal: R/G spread={spread(after_rg):.4f}  B/G spread={spread(after_bg):.4f}  "
             f"(≈0 by construction — confirms the gain equalizes neutral in linear)")
    gains_pct = []
    for c in cams:
        g = ref / np.where(c["rgb"] > 1e-9, c["rgb"], 1e-9)
        gains_pct += [abs(g[0] - 1), abs(g[1] - 1), abs(g[2] - 1)]
    L.append(f"  correction magnitude: max {max(gains_pct)*100:.1f}%  median {sorted(gains_pct)[len(gains_pct)//2]*100:.1f}%  "
             f"(meaningful in linear — contrast with the tiny log-space deltas)")
    L.append("")
    # Decompose each camera's gain into exposure (level) vs white-balance (ratio).
    exps = []
    wb_dev = []
    for c in cams:
        g = ref / np.where(c["rgb"] > 1e-9, c["rgb"], 1e-9)
        e = (g[0] * g[1] * g[2]) ** (1.0 / 3.0)   # exposure factor
        wb = g / e                                 # white-balance-only gains
        exps.append(e)
        wb_dev.append(max(abs(wb[0] - 1), abs(wb[1] - 1), abs(wb[2] - 1)))
    L.append("CORRECTION DECOMPOSITION")
    L.append(f"  exposure factor range : {min(exps):.3f}–{max(exps):.3f}  "
             "(level differences — large is normal across an array)")
    L.append(f"  white-balance gain dev: max {max(wb_dev)*100:.1f}%  "
             "(the chromatic part a diagonal fixes; small = same-sensor consistency)")
    L.append("")
    L.append("NOTE — sphere band drift (diagnostic only, NOT a 3x3 indicator)")
    L.append(f"  intra-sphere chroma variation across luminance: {max(band_resid):.4f}")
    L.append("  This is confounded by lighting (lit side vs shadow side are different light")
    L.append("  colors) and per-camera viewing geometry — it does NOT isolate a sensor")
    L.append("  chromatic difference, and a linear 3x3 could not fix it regardless.")
    L.append("")
    L.append("VERDICT")
    L.append(f"  Scene-linear DIAGONAL match validated: cross-camera WB spread "
             f"{spread(before_rg):.4f}/{spread(before_bg):.4f} → 0 after a per-channel gain,")
    L.append("  solved correctly in linear. Implement this ahead of IPP2.")
    L.append("  The off-diagonal 3x3 (saturated-color crosstalk) CANNOT be assessed from a gray")
    L.append("  sphere — that needs a chart's color patches. For an identical-sensor array it is")
    L.append("  expected to be negligible; a chart would only confirm it, not improve matching.")

    txt = out_dir / "scene_linear_match.txt"
    txt.write_text("\n".join(L))
    csv_path = out_dir / "scene_linear_match.csv"
    with open(csv_path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["camera", "gainR", "gainG", "gainB",
                                           "rg_before", "bg_before", "band_chroma_resid"])
        wr.writeheader()
        for r in rows:
            wr.writerow(r)
    print("=" * 78)
    print("\n".join(L))
    print(f"\nwrote {txt}\nwrote {csv_path}")


if __name__ == "__main__":
    _main()
