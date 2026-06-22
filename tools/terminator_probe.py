#!/usr/bin/env python3
"""
terminator_probe.py  —  shadow-terminator: sphere vs chart
==========================================================

Tests the hypothesis that a flat CHART has far less shadow terminator (i.e. a
shadow_ratio much closer to 1.0) than the original gate calibration suggested
(charts 0.862–1.018). If true, gating-2's looser 0.985 bound is even safer than
documented.

For each clip it measures shadow_ratio (= dark-half / bright-half luminance, the
shadow_specular gate metric) at three things:
  • the SPHERE (auto-detected),
  • any CHART / flat targets you MARK with --chart,
  • a baseline scan of smooth neutral-gray patches (backdrop/floor).

Then it reports each population against the Pass-1 (0.96) and gating-2 (0.985)
thresholds, so you can see exactly where real charts land.

USAGE
-----
  # mark the chart(s): full-res cx,cy,r read off the solver / an image viewer.
  python3 terminator_probe.py --chart 1280,720,180 --chart 1600,800,140 \
      CLIP1.R3D CLIP2.R3D
  # reuse cached renders from a previous tool run:
  python3 terminator_probe.py --reuse --chart 1280,720,180 CLIP.R3D

Read-only: renders temp frames, writes a small report; never edits the solver.
"""
from __future__ import annotations
import argparse, math, os, sys
from pathlib import Path


def _bootstrap_src(explicit):
    c = []
    if explicit: c.append(Path(explicit).expanduser())
    if os.environ.get("R3DMATCH_SRC"): c.append(Path(os.environ["R3DMATCH_SRC"]).expanduser())
    c += [Path.home()/"Desktop"/"R3DMatch_v4"/"src", Path("/Users/sfouasnon/Desktop/R3DMatch_v4/src")]
    here = Path(__file__).resolve()
    for up in [here.parent, *here.parents]:
        c.append(up/"src")
        if (up/"r3dmatch3").is_dir(): c.append(up)
    for p in c:
        if (p/"r3dmatch3"/"sphere.py").is_file(): return p
    sys.exit("ERROR: could not find r3dmatch3. Pass --src /path/to/R3DMatch_v4/src.")


def _shadow_ratio(gray, cx, cy, r):
    """The shadow_specular metric: dark-half / bright-half luminance (0.7r interior)."""
    import numpy as np
    h, w = gray.shape
    y0, y1 = max(0, int(cy-r)), min(h, int(cy+r)+1)
    x0, x1 = max(0, int(cx-r)), min(w, int(cx+r)+1)
    sub = gray[y0:y1, x0:x1]
    yy, xx = np.ogrid[y0:y1, x0:x1]
    m = ((xx-cx)**2 + (yy-cy)**2) <= (r*0.7)**2
    if m.sum() < 20: return None
    pidx = np.argmax(np.where(m, sub, -1.0)); py, px = np.unravel_index(pidx, sub.shape); px += x0; py += y0
    dx, dy = px-cx, py-cy; n = math.hypot(dx, dy)
    if n > 1: dx /= n; dy /= n
    else: dx, dy = 1.0, 0.0
    yi, xi = np.where(m); yi = yi+y0; xi = xi+x0; proj = (xi-cx)*dx + (yi-cy)*dy
    bm = float(gray[yi[proj>=0], xi[proj>=0]].mean())
    dm = float(gray[yi[proj<0], xi[proj<0]].mean()) if (proj<0).any() else bm
    return dm / max(bm, 1e-6)


def _stats(name, vals, np):
    if not vals: return f"  {name:26s}: (none)"
    lo, hi, med = min(vals), max(vals), float(np.median(vals))
    below = sum(1 for v in vals if v < 0.985)
    return (f"  {name:26s}: n={len(vals):3d}  median={med:.3f}  range {lo:.3f}–{hi:.3f}  "
            f"<0.985: {below}/{len(vals)}")


def _main():
    ap = argparse.ArgumentParser(description="shadow-terminator sphere vs chart probe")
    ap.add_argument("clips", nargs="+")
    ap.add_argument("--chart", action="append", default=[], help="full-res cx,cy,r of a chart/flat target (repeatable)")
    ap.add_argument("--src", default=""); ap.add_argument("--redline", default="")
    ap.add_argument("--out", default=""); ap.add_argument("--reuse", action="store_true")
    args = ap.parse_args()

    src = _bootstrap_src(args.src or None); sys.path.insert(0, str(src))
    import numpy as np
    from r3dmatch3 import sphere as S
    from r3dmatch3.redline import resolve_redline_executable, render_measurement_frame

    charts = []
    for c in args.chart:
        try: charts.append(tuple(float(x) for x in c.split(",")))
        except Exception: sys.exit(f"--chart must be cx,cy,r — got {c!r}")

    out_dir = Path(args.out).expanduser() if args.out else (Path(__file__).resolve().parent/"terminator_probe_out")
    out_dir.mkdir(parents=True, exist_ok=True)
    redline = ""
    sphere_srs, chart_srs, flat_srs = [], [], []
    print(f"Pass-1 threshold 0.96   ·   gating-2 threshold 0.985")
    print(f"{'camera':16s} {'sphere SR':>9s}   {'chart SR(s)':>16s}")
    print("-"*60)
    for clip in args.clips:
        stem = Path(clip).stem
        cached = sorted(out_dir.glob(f"{stem}__render.tiff*"))
        if args.reuse and cached:
            actual = str(cached[0])
        else:
            if not redline:
                try: redline = resolve_redline_executable(args.redline or "")
                except Exception as exc: print(f"  ! need REDLine (or --reuse): {exc}"); return
            res = render_measurement_frame(clip, str(out_dir/f"{stem}__render.tiff"), redline=redline, use_as_shot=True)
            if not res.get("ok"): print(f"  ! render failed {stem}"); continue
            actual = res["output_path"]

        img = S._load_image(actual); det, scale = S._resize_for_detection(img)
        rgb = np.array(np.array(det), dtype=np.float32)/255.0
        gray = S._to_gray_arr(rgb); h, w = gray.shape
        r = S.detect_sphere(rgb.astype("float32"), clip_id=stem, peer_ire_values=[45, 46, 47])
        ssr = None
        if getattr(r, "roi", None):
            ssr = _shadow_ratio(gray, r.roi.cx*scale, r.roi.cy*scale, r.roi.r*scale)
            if ssr is not None: sphere_srs.append(ssr)
        # charts (marks are full-res → detection plane via scale)
        csr = []
        for (cx, cy, cr) in charts:
            s = _shadow_ratio(gray, cx*scale, cy*scale, cr*scale)
            if s is not None: csr.append(s); chart_srs.append(s)
        # baseline: smooth neutral-gray patches (not ALT candidates, just context)
        for cy in range(60, h-60, 60):
            for cx in range(60, w-60, 60):
                if ssr is not None and math.hypot(cx-r.roi.cx*scale, cy-r.roi.cy*scale) < 90: continue
                patch = gray[cy-56:cy+56, cx-56:cx+56]
                if patch.size == 0 or patch.std() > 0.03: continue
                px = rgb[cy-56:cy+56, cx-56:cx+56].reshape(-1, 3); mn = px.mean(0); g = mn[1] if mn[1] > 1e-6 else 1e-6
                if not (0.9 <= mn[0]/g <= 1.25 and 0.8 <= mn[2]/g <= 1.2): continue
                s = _shadow_ratio(gray, cx, cy, 56)
                if s is not None: flat_srs.append(s)
        cs = " ".join(f"{x:.3f}" for x in csr) if csr else "—"
        print(f"{stem[:16]:16s} {('%.3f'%ssr) if ssr is not None else '—':>9s}   {cs:>16s}")

    print("\nPOPULATIONS")
    print(_stats("SPHERES (auto-detected)", sphere_srs, np))
    print(_stats("CHARTS (marked)", chart_srs, np))
    print(_stats("flat gray patches (context)", flat_srs, np))
    print("\nINTERPRETATION")
    if chart_srs:
        cmed = float(np.median(chart_srs)); cmin = min(chart_srs)
        if cmin >= 0.985:
            print(f"  Marked charts sit at shadow_ratio ≥ {cmin:.3f} (median {cmed:.3f}) — ABOVE gating-2's")
            print("  0.985 bound. They have minimal terminator; gating-2 cannot admit them. Hypothesis CONFIRMED.")
        elif cmin >= 0.96:
            print(f"  Marked charts {cmin:.3f}–{max(chart_srs):.3f}: above Pass-1 0.96 but the low end dips below 0.985.")
            print("  Those low readings are lighting-gradient (angled light across the flat), not convexity.")
        else:
            print(f"  Some marked charts read < 0.96 ({cmin:.3f}) — a real lighting gradient. Note ALT still must")
            print("  PROPOSE the chart as a candidate for gating-2 to matter; run sphere_gate_probe to check that.")
    else:
        print("  No charts marked. Re-run with --chart cx,cy,r on the actual chart(s) in frame for the definitive test.")
    print("\nNote: gating-2 only acts on the ALT candidate stream. Even a low-shadow_ratio flat object is harmless")
    print("unless ALT proposes it as a candidate — verify with tools/sphere_gate_probe.py (flat ALT candidates).")


if __name__ == "__main__":
    _main()
