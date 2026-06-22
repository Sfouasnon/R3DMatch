#!/usr/bin/env python3
"""
sphere_gate_probe.py  —  validate detection on the REAL ALT candidate stream
=============================================================================

Background
----------
The detector's primary candidate generator is cv2 HOUGH_GRADIENT_ALT ("ALT").
ALT uses gradient *direction* to vote, so it rejects tape/grid/equipment
clutter at the source and typically returns ~1 candidate per frame — the ball.
Only if ALT returns 0 does the code fall back to skimage Hough (which spews
hundreds of junk circles).

An earlier version of this probe generated negatives with skimage, which made
the gate look impossible to relax. That was an artifact: the real pipeline never
sees those junk circles. This version measures the REAL ALT stream.

What it tests
-------------
For each clip it runs ALT exactly as detect_sphere does, then for every ALT
candidate records: whether it's the ball (near the profile's confirmed sphere
location), the full gate verdict, the failing gate, and shadow_ratio/peak_excess.

It then answers two questions:
  1. Does ALT cleanly return the ball, and which cameras fail and at which gate?
  2. The proposed fallback — "if the gates fail, accept the ALT candidate" — is
     it SAFE? i.e. when an ALT candidate fails the gates, is it the ball (good to
     accept) or a flat decoy (bad to accept)? And does any flat ALT candidate
     PASS all gates today (an existing false positive)?

USAGE
-----
    python3 sphere_gate_probe.py --profile "Test_Footage_F5.6" \
        ~/Desktop/Test_Footage/F5.6
    # reuse cached renders, sweep ALT confidence:
    python3 sphere_gate_probe.py --profile "Test_Footage_F5.6" --reuse \
        --param2 0.85  ~/Desktop/Test_Footage/F5.6

Outputs (default ./sphere_gate_probe_out):
    alt_probe.csv   — one row per ALT candidate
    alt_probe.txt   — per-clip table + fallback-safety verdict

Read-only: renders temp frames and writes the report folder; never edits the
solver or your profiles.
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


def _shadow_peak(gray, cx, cy, r):
    """Replicates _gate_shadow_specular metrics on a local sub-window (fast)."""
    h, w = gray.shape
    y0, y1 = max(0, int(cy - r)), min(h, int(cy + r) + 1)
    x0, x1 = max(0, int(cx - r)), min(w, int(cx + r) + 1)
    sub = gray[y0:y1, x0:x1]
    yy, xx = np.ogrid[y0:y1, x0:x1]
    m = ((xx - cx) ** 2 + (yy - cy) ** 2) <= (r * 0.7) ** 2
    vals = sub[m]
    if vals.size < 20:
        return None, None
    mean_i = float(vals.mean())
    peak = float(vals.max())
    pe = peak / max(mean_i, 1e-6)
    pidx = np.argmax(np.where(m, sub, -1.0))
    py, px = np.unravel_index(pidx, sub.shape)
    px += x0; py += y0
    dx, dy = px - cx, py - cy
    n = math.hypot(dx, dy)
    if n > 1:
        dx /= n; dy /= n
    else:
        dx, dy = 1.0, 0.0
    yi, xi = np.where(m)
    yi = yi + y0; xi = xi + x0
    proj = (xi - cx) * dx + (yi - cy) * dy
    bm = float(gray[yi[proj >= 0], xi[proj >= 0]].mean())
    dm = float(gray[yi[proj < 0], xi[proj < 0]].mean()) if (proj < 0).any() else bm
    return dm / max(bm, 1e-6), pe


def _iter_clips(paths):
    for a in paths:
        p = Path(a).expanduser()
        if p.is_dir():
            for r3d in sorted(p.rglob("*.R3D")):
                if r3d.stem.endswith("_001") or not r3d.stem[-1].isdigit():
                    yield str(r3d)
        elif p.suffix.upper() == ".R3D":
            yield str(p)


def _alt_candidates(S, cv2, det_arr, param2):
    """Generate candidates the way detect_sphere's primary path does (ALT)."""
    gray = S._to_gray(det_arr)            # float [0,1]
    h, w = gray.shape
    r_min = max(6, int(w * S._RADIUS_MIN_RATIO))
    r_max = min(h // 2, int(w * S._RADIUS_MAX_RATIO))
    g8 = (np.clip(gray, 0, 1) * 255).astype(np.uint8)
    gb = cv2.GaussianBlur(g8, (S._HOUGH_ALT_BLUR_K, S._HOUGH_ALT_BLUR_K), 1.0)
    md = int(w * S._RADIUS_MIN_RATIO)
    ac = cv2.HoughCircles(gb, cv2.HOUGH_GRADIENT_ALT, dp=S._HOUGH_ALT_DP, minDist=md,
                          param1=S._HOUGH_ALT_PARAM1, param2=param2,
                          minRadius=r_min, maxRadius=r_max)
    cands = []
    if ac is not None and len(ac[0]) > 0:
        for cx, cy, r in ac[0]:
            cands.append((float(cx), float(cy), float(r)))
    return cands


def _main():
    ap = argparse.ArgumentParser(description="ALT-stream detection probe")
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--profile", default="")
    ap.add_argument("--src", default="")
    ap.add_argument("--redline", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--reuse", action="store_true")
    ap.add_argument("--param2", type=float, default=None,
                    help="Override ALT param2 (default = solver's _HOUGH_ALT_PARAM2)")
    args = ap.parse_args()

    src = _bootstrap_src(args.src or None)
    sys.path.insert(0, str(src))

    global np
    import numpy as np
    import cv2
    from r3dmatch3 import sphere as S
    from r3dmatch3.redline import resolve_redline_executable, render_measurement_frame
    from r3dmatch3.sphere_profile import load_project_profile, get_camera_prior
    from r3dmatch3.workflow import camera_label_from_clip_id

    param2 = args.param2 if args.param2 is not None else S._HOUGH_ALT_PARAM2
    redline = ""
    out_dir = Path(args.out).expanduser() if args.out else (Path(__file__).resolve().parent / "sphere_gate_probe_out")
    out_dir.mkdir(parents=True, exist_ok=True)

    profile = {}
    if args.profile:
        try:
            profile = load_project_profile(args.profile)
        except Exception as exc:
            print(f"(could not load profile '{args.profile}': {exc})")

    clips = list(_iter_clips(args.paths))
    print(f"r3dmatch3 src : {src}")
    print(f"clips         : {len(clips)}   profile cams: {len(profile.get('cameras', {})) if profile else 0}")
    print(f"ALT param2    : {param2}  (solver default {S._HOUGH_ALT_PARAM2})")
    print("=" * 78)
    print(f"{'camera':12s} {'ALTn':4s} {'ball?':5s} {'ball verdict':22s} {'flatCands':9s} {'flatPASS':8s}")
    print("-" * 78)

    rows = []
    per_clip = []
    for clip in clips:
        stem = Path(clip).stem
        cam = camera_label_from_clip_id(stem)
        cached = sorted(out_dir.glob(f"{stem}__render.tiff*"))
        if args.reuse and cached:
            actual = str(cached[0])
        else:
            if not redline:
                try:
                    redline = resolve_redline_executable(args.redline or "")
                except Exception as exc:
                    print(f"  ! need REDLine (or --reuse): {exc}"); return
            res = render_measurement_frame(clip, str(out_dir / f"{stem}__render.tiff"),
                                           redline=redline, use_as_shot=True)
            if not res.get("ok"):
                print(f"  ! render failed {stem}: {res.get('status')}"); continue
            actual = res["output_path"]

        img = S._load_image(actual)
        det, scale = S._resize_for_detection(img)
        det_arr = np.array(det)
        rgb = np.array(det_arr, dtype=np.float32) / 255.0
        gray = S._to_gray_arr(rgb)
        h, w = gray.shape

        # confirmed sphere location (for labeling)
        sx = sy = None
        prior = get_camera_prior(profile, cam) if profile else None
        if prior and prior.get("radius_ratio_mean"):
            sx = prior["cx_norm_mean"] * w
            sy = prior["cy_norm_mean"] * h

        cands = _alt_candidates(S, cv2, det_arr, param2)
        alt_n = len(cands)
        ball_found = False
        ball_verdict = "(ALT returned 0)" if alt_n == 0 else "no ball candidate"
        flat_cands = 0
        flat_pass = 0
        for (cx, cy, r) in cands:
            is_ball = sx is not None and math.hypot(cx - sx, cy - sy) <= max(25, 0.6 * r)
            gates = S._run_5_gates(S._Candidate(cx=cx, cy=cy, r=r, accumulator=0.85), rgb, h, w)
            passed_all = bool(gates) and all(g.passed for g in gates)
            fail_gate = "" if passed_all else next((g.gate for g in gates if not g.passed), "?")
            sratio, pexc = _shadow_peak(gray, cx, cy, r)
            label = "sphere" if is_ball else "flat"
            rows.append(dict(clip=stem, camera=cam, label=label, cx=round(cx, 1), cy=round(cy, 1),
                             r=round(r, 1), shadow_ratio=round(sratio or -1, 4),
                             peak_excess=round(pexc or -1, 4),
                             passed_all=("yes" if passed_all else "no"),
                             fail_gate=fail_gate))
            if is_ball:
                ball_found = True
                ball_verdict = "PASS (auto)" if passed_all else f"fail@{fail_gate}"
            else:
                flat_cands += 1
                if passed_all:
                    flat_pass += 1
        per_clip.append(dict(cam=cam, alt_n=alt_n, ball=ball_found, verdict=ball_verdict,
                             flat_cands=flat_cands, flat_pass=flat_pass))
        print(f"{cam:12s} {alt_n:<4d} {('y' if ball_found else 'n'):5s} "
              f"{ball_verdict:22s} {flat_cands:<9d} {flat_pass:<8d}")

    # ---- CSV ----
    csv_path = out_dir / "alt_probe.csv"
    with open(csv_path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["clip", "camera", "label", "cx", "cy", "r",
                                           "shadow_ratio", "peak_excess", "passed_all", "fail_gate"])
        wr.writeheader()
        for row in rows:
            wr.writerow(row)

    # ---- summary + fallback-safety verdict ----
    L = ["ALT-STREAM DETECTION PROBE", "=" * 40, f"ALT param2 = {param2}", ""]
    n = len(per_clip)
    alt0 = [c for c in per_clip if c["alt_n"] == 0]
    ball_pass = [c for c in per_clip if c["verdict"].startswith("PASS")]
    ball_failgate = [c for c in per_clip if c["ball"] and c["verdict"].startswith("fail@")]
    no_ball = [c for c in per_clip if not c["ball"] and c["alt_n"] > 0]
    L.append(f"clips                         : {n}")
    L.append(f"ALT returned 0 (would fallback): {len(alt0)}  {[c['cam'] for c in alt0]}")
    L.append(f"ball auto-passes all gates     : {len(ball_pass)}")
    L.append(f"ball found but FAILS a gate    : {len(ball_failgate)}  "
             f"{[(c['cam'], c['verdict']) for c in ball_failgate]}")
    L.append(f"ALT found no ball candidate    : {len(no_ball)}  {[c['cam'] for c in no_ball]}")
    L.append("")

    # Fallback safety: "if gates fail, accept the ALT candidate."
    sphere_rows = [r for r in rows if r["label"] == "sphere"]
    flat_rows = [r for r in rows if r["label"] == "flat"]
    sphere_failing = [r for r in sphere_rows if r["passed_all"] == "no"]
    flat_failing = [r for r in flat_rows if r["passed_all"] == "no"]
    flat_passing = [r for r in flat_rows if r["passed_all"] == "yes"]
    L.append("FALLBACK SAFETY  ('accept the ALT candidate when gates fail')")
    L.append(f"  ALT candidates total          : {len(rows)}  (spheres={len(sphere_rows)} flats={len(flat_rows)})")
    L.append(f"  spheres that FAIL gates        : {len(sphere_failing)}  "
             "→ these are RESCUED by the fallback (good)")
    L.append(f"  flat ALT candidates that FAIL  : {len(flat_failing)}  "
             "→ these would be WRONGLY accepted by a blind fallback (bad)")
    L.append(f"  flat ALT candidates that PASS  : {len(flat_passing)}  "
             "→ existing false positives TODAY (gate already lets them through)")
    if flat_failing:
        L.append("  flats a blind fallback would wrongly accept:")
        for r in flat_failing[:15]:
            L.append(f"    {r['clip']} ({r['cx']},{r['cy']},r{r['r']}) "
                     f"fail@{r['fail_gate']} sr={r['shadow_ratio']} pe={r['peak_excess']}")
    L.append("")
    if not flat_failing and sphere_failing:
        L.append("VERDICT: SAFE. Every gate-failing ALT candidate in this footage is the ball — "
                 "the fallback rescues the well-lit spheres and admits zero flats. (Validate on "
                 "footage containing flat round charts/cards to be thorough.)")
    elif flat_failing:
        L.append("VERDICT: NOT blindly safe — some flat ALT candidates also fail the gates and would "
                 "be wrongly accepted. Prefer accepting the fallback ROI as NEEDS_ASSIST (operator "
                 "review) rather than auto-SUCCESS, or restrict the fallback to when ALT returns "
                 "exactly ONE candidate.")
    else:
        L.append("VERDICT: no gate-failing spheres in this run (nothing to rescue here).")
    L.append("")
    L.append(f"per-candidate detail: {csv_path}")
    txt = out_dir / "alt_probe.txt"
    txt.write_text("\n".join(L))
    print("=" * 78)
    print("\n".join(L))
    print(f"\nwrote {txt}\nwrote {csv_path}")


if __name__ == "__main__":
    _main()
