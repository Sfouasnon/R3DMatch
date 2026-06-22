#!/usr/bin/env python3
"""
sphere_doctor.py  —  one-off sphere-detection diagnostic for R3DMatch v4
=========================================================================

Point it at one or more .R3D clips that FAILED auto-detection and it tells you,
in plain terms, *why* the gray ball was rejected — without touching the proven
solver. It renders each clip exactly the way the app does (REDLine, as-shot,
reference IPP2/BT.709 pipeline), then re-runs the real detection code with full
debug logging and reports:

  1. Every Hough candidate the detector found (position + radius), and which
     gate rejected each one (radius floor, std band, BRDF, RGB ratio, or the
     5-gate geometry/material/Lambertian/specular/size pipeline).
  2. Whether the true ball ever became a candidate at all (if it didn't, the
     problem is upstream — Canny/Hough/contrast or the profile's radius prior).
  3. An annotated PNG so you can SEE where every candidate landed.
  4. (optional) A full gate-by-gate verdict evaluated at the exact spot you
     mark the ball, so you know precisely which gate to look at.

USAGE
-----
    python3 sphere_doctor.py CLIP1.R3D [CLIP2.R3D ...]

    # also test the saved profile's radius prior (replicates the failing run):
    python3 sphere_doctor.py --profile "Test_Footage_F5.6" CLIP.R3D

    # pin the exact ball location (full-res pixels, read off the solver) and get
    # the complete gate stack evaluated right there:
    python3 sphere_doctor.py --mark 1820,690,95 CLIP.R3D

    # custom output folder (default: alongside this script, ./sphere_doctor_out)
    python3 sphere_doctor.py --out ~/Desktop/sphere_report CLIP.R3D

    # override paths if auto-detection fails:
    python3 sphere_doctor.py --src /path/to/R3DMatch_v4/src \
                             --redline /usr/local/bin/REDLine CLIP.R3D

Nothing here writes to your project profiles or the app's output — it only
renders temp frames and writes a report folder. Read-only with respect to the
solver and your data.
"""

from __future__ import annotations
import argparse
import io
import logging
import os
import re
import sys
import tempfile
from pathlib import Path


# ───────────────────────── locate the r3dmatch3 package ─────────────────────
def _bootstrap_src(explicit: str | None) -> Path:
    candidates = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    if os.environ.get("R3DMATCH_SRC"):
        candidates.append(Path(os.environ["R3DMATCH_SRC"]).expanduser())
    # common install locations
    candidates += [
        Path.home() / "Desktop" / "R3DMatch_v4" / "src",
        Path("/Users/sfouasnon/Desktop/R3DMatch_v4/src"),
    ]
    # walk up from this file (in case the script lives inside the repo)
    here = Path(__file__).resolve()
    for up in [here.parent, *here.parents]:
        candidates.append(up / "src")
        if (up / "r3dmatch3").is_dir():
            candidates.append(up)
    for c in candidates:
        if (c / "r3dmatch3" / "sphere.py").is_file():
            return c
    sys.exit(
        "ERROR: could not find the r3dmatch3 package.\n"
        "Pass --src /path/to/R3DMatch_v4/src or set R3DMATCH_SRC."
    )


def _main():
    ap = argparse.ArgumentParser(description="R3DMatch sphere detection doctor")
    ap.add_argument("clips", nargs="+", help="One or more .R3D clips to analyze")
    ap.add_argument("--src", default="", help="Path to R3DMatch_v4/src")
    ap.add_argument("--redline", default="", help="Path to REDLine executable")
    ap.add_argument("--profile", default="", help="Sphere-profile project id to apply radius prior")
    ap.add_argument("--mark", default="", help="Ball location 'cx,cy,r' in FULL-RES pixels")
    ap.add_argument("--out", default="", help="Output folder (default: ./sphere_doctor_out)")
    ap.add_argument("--frame", type=int, default=0, help="Frame index to render (default 0)")
    args = ap.parse_args()

    src = _bootstrap_src(args.src or None)
    sys.path.insert(0, str(src))

    # Imports from the real package (after sys.path set)
    import numpy as np
    from PIL import Image, ImageDraw
    from r3dmatch3 import sphere as S
    from r3dmatch3.redline import resolve_redline_executable, render_measurement_frame
    from r3dmatch3.models import SphereROI
    try:
        from r3dmatch3.sphere_profile import (
            load_project_profile, get_camera_prior, _PROFILE_DIR,
        )
        from r3dmatch3.workflow import camera_label_from_clip_id
    except Exception:
        load_project_profile = get_camera_prior = None
        _PROFILE_DIR = None
        camera_label_from_clip_id = lambda cid: cid  # noqa: E731

    # Resolve REDLine
    try:
        redline = resolve_redline_executable(args.redline or "")
    except Exception as exc:
        sys.exit(f"ERROR: REDLine not found ({exc}). Pass --redline /path/to/REDLine.")

    out_dir = Path(args.out).expanduser() if args.out else (Path(__file__).resolve().parent / "sphere_doctor_out")
    out_dir.mkdir(parents=True, exist_ok=True)

    mark = None
    if args.mark:
        try:
            mx, my, mr = [float(x) for x in args.mark.split(",")]
            mark = (mx, my, mr)
        except Exception:
            sys.exit("ERROR: --mark must be 'cx,cy,r' (full-res pixels), e.g. --mark 1820,690,95")

    print(f"r3dmatch3 src : {src}")
    print(f"REDLine       : {redline}")
    print(f"output folder : {out_dir}")
    print(f"profiles dir  : {_PROFILE_DIR}")
    print("=" * 78)

    for clip in args.clips:
        _analyze_clip(
            clip, redline, out_dir, args.frame, mark, args.profile,
            S, np, Image, ImageDraw, SphereROI,
            load_project_profile, get_camera_prior, camera_label_from_clip_id,
        )


# ───────────────────────── candidate-log parsing ────────────────────────────
# Pull cx/cy/r (detection-plane) and an outcome label out of the debug log.
_CAND_RE = re.compile(r"cx=(-?\d+\.?\d*)\s+cy=(-?\d+\.?\d*)\s+r=(-?\d+\.?\d*)")


def _classify(line: str) -> str | None:
    if "Pre-filter SKIP G1" in line: return "skip_radius"
    if "Pre-filter SKIP G2-hard" in line: return "skip_std_hard"
    if "Pre-filter SKIP G2-brdf" in line: return "skip_brdf"
    if "Pre-filter SKIP G3" in line: return "skip_std_floor"
    if "Pre-filter SKIP G4" in line: return "skip_rgb"
    if "advancing to 5-gate" in line: return "advanced"
    if line.startswith("Sphere detected") or "Sphere detected" in line: return "detected"
    return None


_COLORS = {
    "skip_radius":    (255, 80, 80),    # red    — too small
    "skip_std_hard":  (255, 120, 0),    # orange — texture too high
    "skip_brdf":      (255, 200, 0),    # amber  — failed BRDF fallback
    "skip_std_floor": (120, 120, 255),  # blue   — too uniform (wall/floor)
    "skip_rgb":       (200, 0, 255),    # violet — colored object
    "advanced":       (0, 229, 255),    # cyan   — reached 5-gate pipeline
    "detected":       (0, 255, 100),    # green  — passed everything
}
_LABELS = {
    "skip_radius":    "G1 radius floor (too small)",
    "skip_std_hard":  "G2 std hard-ceiling (too textured)",
    "skip_brdf":      "G2 BRDF fallback (not sphere-like)",
    "skip_std_floor": "G3 std floor (too uniform)",
    "skip_rgb":       "G4 RGB ratio (colored, not gray)",
    "advanced":       "advanced to 5-gate pipeline (failed there)",
    "detected":       "DETECTED (passed all gates)",
}


def _analyze_clip(clip, redline, out_dir, frame_idx, mark, profile_id,
                  S, np, Image, ImageDraw, SphereROI,
                  load_project_profile, get_camera_prior, camera_label_from_clip_id):
    from r3dmatch3.redline import render_measurement_frame

    clip = str(Path(clip).expanduser())
    name = Path(clip).stem
    print(f"\n### {name}")
    if not Path(clip).exists():
        print(f"  ! clip not found: {clip}")
        return

    # 1) Render the frame exactly like the app (as-shot, reference pipeline)
    tiff_path = str(out_dir / f"{name}__render.tiff")
    print("  rendering frame via REDLine …")
    res = render_measurement_frame(clip, tiff_path, redline=redline,
                                   frame_index=frame_idx, use_as_shot=True)
    if not res.get("ok"):
        print(f"  ! render failed: {res.get('status')} — {res.get('stderr','')[:300]}")
        return
    tiff_actual = res["output_path"]
    print(f"  rendered: {tiff_actual}")

    report_lines = [f"SPHERE DOCTOR REPORT — {name}", "=" * 60, f"clip: {clip}", ""]

    # 2) Cold detection (no prior) with full debug capture
    cold_log, cold_result = _run_capture(S, tiff_actual)
    report_lines += _summarize_run("COLD DETECTION (no profile prior)", cold_log, cold_result)

    # 3) Optional: with-profile prior (replicates the real run's radius narrowing)
    prior = None
    if load_project_profile is not None:
        prior = _find_prior(name, profile_id, load_project_profile,
                            get_camera_prior, camera_label_from_clip_id, _profile_dir_of(S))
    if prior is not None:
        # Convert the stored normalized prior → detection-plane pixels, matching
        # how the detector narrows its radius search (±30% of prior_r).
        img_full = S._load_image(tiff_actual)
        img_det_pil, _scale = S._resize_for_detection(img_full)
        w_det, h_det = img_det_pil.size
        base = min(w_det, h_det)
        prior_cx = prior.get("cx_norm_mean", 0.5) * w_det
        prior_cy = prior.get("cy_norm_mean", 0.5) * h_det
        prior_r = prior.get("radius_ratio_mean", 0.05) * base
        report_lines.append("")
        report_lines.append(
            f"Profile prior ({prior.get('sample_tier')}, {prior.get('sample_count')} samples): "
            f"cx_norm={prior.get('cx_norm_mean'):.3f} cy_norm={prior.get('cy_norm_mean'):.3f} "
            f"radius_ratio={prior.get('radius_ratio_mean'):.4f}")
        report_lines.append(
            f"  → detection-plane prior ~ cx={prior_cx:.0f} cy={prior_cy:.0f} r={prior_r:.0f}; "
            f"radius search narrowed to {prior_r*0.7:.0f}-{prior_r*1.3:.0f} px (approx).")
        pr_log, pr_result = _run_capture(
            S, tiff_actual, prior_cx=prior_cx, prior_cy=prior_cy, prior_r=prior_r,
        )
        report_lines += _summarize_run("WITH PROFILE PRIOR (approx reproduction of the run)", pr_log, pr_result)
    else:
        report_lines.append("")
        report_lines.append("No profile prior applied (cold only). Pass --profile <id> to "
                            "replicate the run's radius prior.")

    # 4) Optional: full gate stack at a marked ball location
    if mark is not None:
        report_lines.append("")
        report_lines += _eval_at_mark(S, np, tiff_actual, mark, SphereROI)

    # 5) Annotated image from the cold run's candidates
    png_path = out_dir / f"{name}__candidates.png"
    _draw_candidates(Image, ImageDraw, S, tiff_actual, cold_log, mark, png_path)
    report_lines.append("")
    report_lines.append(f"Annotated candidate map: {png_path}")

    # write report
    rpt = out_dir / f"{name}__report.txt"
    rpt.write_text("\n".join(report_lines))
    print(f"  report : {rpt}")
    print(f"  image  : {png_path}")
    # echo the key verdict to console
    for ln in report_lines:
        if ln.startswith("VERDICT") or ln.startswith("  →"):
            print("  " + ln)


def _profile_dir_of(S):
    try:
        from r3dmatch3.sphere_profile import _PROFILE_DIR
        return _PROFILE_DIR
    except Exception:
        return None


def _run_capture(S, tiff_path, **prior_kwargs):
    """Run the by-path detector with DEBUG capture. Returns (log_text, result)."""
    logger = logging.getLogger("r3dmatch3.sphere")
    buf = io.StringIO()
    h = logging.StreamHandler(buf)
    h.setLevel(logging.DEBUG)
    h.setFormatter(logging.Formatter("%(message)s"))
    old_level = logger.level
    logger.setLevel(logging.DEBUG)
    logger.addHandler(h)
    try:
        # _detect_sphere_by_path is the original path-based detector
        detect = getattr(S, "_detect_sphere_by_path")
        result = detect(tiff_path, **prior_kwargs)
    finally:
        logger.removeHandler(h)
        logger.setLevel(old_level)
    return buf.getvalue(), result


def _summarize_run(title, log_text, result):
    lines = ["", title, "-" * len(title)]
    cands = _parse_candidates(log_text)
    # Hough count line
    m = re.search(r"(HOUGH_GRADIENT_ALT|skimage Hough fallback): (\d+) candidates", log_text)
    if m:
        lines.append(f"Candidate source: {m.group(1)} → {m.group(2)} raw candidates")
    lines.append(f"Candidates evaluated: {len(cands)}")
    for c in cands:
        lines.append(f"  • cx={c['cx']:.0f} cy={c['cy']:.0f} r={c['r']:.0f}  →  {_LABELS.get(c['outcome'], c['outcome'])}")
    status = getattr(result, "status", "?")
    src = getattr(result, "source", None)
    lines.append(f"VERDICT: status={status} source={src}")
    if status not in ("SUCCESS", "NEEDS_ASSIST"):
        # Diagnose the dominant failure mode
        outcomes = [c["outcome"] for c in cands]
        if not cands:
            lines.append("  → No candidates at all — Canny/Hough found no circle. Likely weak "
                         "ball edge (contrast) or the radius band excluded it.")
        elif "advanced" in outcomes:
            lines.append("  → The ball (or something) reached the 5-gate pipeline but failed it. "
                         "Re-run with --mark on the ball to see which of the 5 gates rejects it.")
        else:
            from collections import Counter
            top = Counter(outcomes).most_common(1)[0][0]
            lines.append(f"  → All candidates died in pre-filter; most common: {_LABELS.get(top, top)}. "
                         "If the true ball is among the listed candidates, that gate is the culprit; "
                         "if it's NOT listed, the ball never formed a Hough candidate.")
    return lines


def _parse_candidates(log_text):
    out = []
    for line in log_text.splitlines():
        oc = _classify(line)
        if oc is None:
            continue
        m = _CAND_RE.search(line)
        if not m:
            continue
        out.append({"cx": float(m.group(1)), "cy": float(m.group(2)),
                    "r": float(m.group(3)), "outcome": oc})
    return out


def _find_prior(clip_stem, profile_id, load_project_profile, get_camera_prior,
                camera_label_from_clip_id, profile_dir):
    """Best-effort: find this camera's saved prior in a profile."""
    cam = camera_label_from_clip_id(clip_stem)
    ids = []
    if profile_id:
        ids.append(profile_id)
    elif profile_dir is not None:
        try:
            ids = [p.stem for p in Path(profile_dir).glob("*.json")]
        except Exception:
            ids = []
    for pid in ids:
        try:
            prof = load_project_profile(pid)
            pr = get_camera_prior(prof, cam) if get_camera_prior else None
            if pr and pr.get("radius_ratio_mean"):
                return pr
        except Exception:
            continue
    return None


def _eval_at_mark(S, np, tiff_path, mark, SphereROI):
    """Evaluate the WHOLE gate stack at a marked full-res ball location."""
    lines = ["GATE STACK AT MARKED BALL LOCATION", "-" * 34]
    mx, my, mr = mark
    img_full = S._load_image(tiff_path)
    img_det_pil, scale = S._resize_for_detection(img_full)
    img_det = np.array(img_det_pil)
    h, w = img_det.shape[:2]
    rgb_det = np.array(img_det, dtype=np.float32) / 255.0
    cx, cy, r = mx * scale, my * scale, mr * scale
    lines.append(f"mark (full-res): cx={mx:.0f} cy={my:.0f} r={mr:.0f}  →  "
                 f"detection-plane cx={cx:.0f} cy={cy:.0f} r={r:.0f} (scale {scale:.4f})")

    # Pre-filters (mirror sphere.detect_sphere)
    rw = r / w
    lines.append(f"G1 radius floor : r/w={rw:.4f}  (min {S._PF_RADIUS_MIN})  "
                 f"{'PASS' if rw >= S._PF_RADIUS_MIN else 'FAIL'}")
    mask = S._interior_mask(rgb_det, cx, cy, r)
    gray = S._to_gray_arr(rgb_det)
    std = float(gray[mask].std()) if mask.sum() > 0 else 1.0
    try:
        from r3dmatch3.brdf_gate import brdf_gate, rgb_display_to_lum
        lum = rgb_display_to_lum(rgb_det)
        brdf = float(brdf_gate(lum, cx, cy, r)[0])
    except Exception as exc:
        brdf = float("nan")
        lines.append(f"   (brdf score unavailable: {exc})")
    lines.append(f"G2 std band     : std={std:.4f}  clean≤{S._PF_STD_CLEAN_MAX} "
                 f"hard≤{S._PF_STD_HARD_MAX}  brdf={brdf:.3f} (fallback min {S._PF_BRDF_FALLBACK_MIN})")
    if std > S._PF_STD_HARD_MAX:
        lines.append("   → FAIL G2-hard (texture too high for a sphere interior)")
    elif std > S._PF_STD_CLEAN_MAX:
        lines.append(f"   → middle band; needs BRDF≥{S._PF_BRDF_FALLBACK_MIN} → "
                     f"{'PASS' if brdf >= S._PF_BRDF_FALLBACK_MIN else 'FAIL'}")
    else:
        lines.append("   → PASS G2-clean")
    lines.append(f"G3 std floor    : std={std:.4f}  (floor {S._PF_STD_FLOOR})  "
                 f"{'PASS' if std >= S._PF_STD_FLOOR else 'FAIL (too uniform)'}")
    px = rgb_det[mask]
    if px.shape[0] > 0:
        mean = px.mean(axis=0)
        g = float(mean[1]) if float(mean[1]) > 1e-6 else 1e-6
        rg, bg = float(mean[0]) / g, float(mean[2]) / g
        ok = (S._PF_RG_MIN <= rg <= S._PF_RG_MAX and S._PF_BG_MIN <= bg <= S._PF_BG_MAX)
        lines.append(f"G4 RGB ratio    : R/G={rg:.3f} B/G={bg:.3f}  "
                     f"(R/G {S._PF_RG_MIN}-{S._PF_RG_MAX}, B/G {S._PF_BG_MIN}-{S._PF_BG_MAX})  "
                     f"{'PASS' if ok else 'FAIL (not neutral gray)'}")

    # 5-gate pipeline
    cand = S._Candidate(cx=cx, cy=cy, r=r, accumulator=0.85)
    try:
        gates = S._run_5_gates(cand, rgb_det, h, w)
        lines.append("5-gate pipeline :")
        for g in gates:
            lines.append(f"   - {g.gate:<14} {'PASS' if g.passed else 'FAIL'}  {g.reason}")
        if gates and all(gg.passed for gg in gates):
            lines.append("   → all 5 gates PASS — this location is a valid sphere; the auto-miss "
                         "was upstream (no Hough candidate here, or pre-filter).")
    except Exception as exc:
        lines.append(f"   (5-gate eval error: {exc})")
    return lines


def _draw_candidates(Image, ImageDraw, S, tiff_path, log_text, mark, png_path):
    """Draw every candidate on the detection-scale image, colored by outcome."""
    img_full = S._load_image(tiff_path)
    img_det_pil, scale = S._resize_for_detection(img_full)
    im = img_det_pil.convert("RGB")
    d = ImageDraw.Draw(im)
    for c in _parse_candidates(log_text):
        col = _COLORS.get(c["outcome"], (200, 200, 200))
        x, y, r = c["cx"], c["cy"], c["r"]
        d.ellipse([x - r, y - r, x + r, y + r], outline=col, width=3)
        d.line([x - 6, y, x + 6, y], fill=col, width=1)
        d.line([x, y - 6, x, y + 6], fill=col, width=1)
    if mark is not None:
        mx, my, mr = [v * scale for v in mark]
        d.ellipse([mx - mr, my - mr, mx + mr, my + mr], outline=(255, 0, 200), width=4)
        d.text((mx - mr, my - mr - 12), "MARK", fill=(255, 0, 200))
    # legend
    y0 = 6
    for oc, col in _COLORS.items():
        d.rectangle([6, y0, 18, y0 + 10], fill=col)
        d.text((22, y0), _LABELS.get(oc, oc), fill=(255, 255, 255))
        y0 += 14
    im.save(png_path)


if __name__ == "__main__":
    _main()
