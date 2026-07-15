#!/usr/bin/env python3
"""
macbeth_diag.py  —  Macbeth ColorChecker diagnostic  (R3DMatch V5)

Workflow:
  1.  Find R3D files in a directory  (one per camera)
  2.  Render single frame to TIFF via REDLine  (IPP2 / BT.709 / BT.1886)
  3.  Detect Macbeth ColorChecker  (cv2.mcc)
  4.  Sample all 24 patches at full resolution
  5.  Exposure solve  — neutral patches (bottom row), inter-camera
  6.  WB solve        — neutral patches, inter-camera
  7.  Matrix solve    — all 24 patches, 3x3 least squares  (skip with --no-matrix)
  8.  Diagnostic report to stdout  +  JSON saved to --out

Cross-validates against a sphere solve JSON if --sphere is provided.

Usage:
  python macbeth_diag.py /path/to/R3Ds [options]

Options:
  --redline PATH    Path to REDline binary (auto-detected if omitted)
  --out DIR         Output dir for JSON report  (default: ./macbeth_out)
  --sphere JSON     Sphere solve JSON to cross-validate against
  --no-matrix       Skip 3x3 matrix solve
  --keep-tiffs      Keep rendered TIFFs after analysis
  -v / --verbose    Debug logging

Requirements:
  pip install opencv-contrib-python numpy Pillow
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import math
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

try:
    import cv2
    _MCC_AVAILABLE = hasattr(cv2, "mcc")
except ImportError:
    cv2 = None          # type: ignore
    _MCC_AVAILABLE = False

log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

REDLINE_SEARCH = [
    "/Applications/REDCINE-X Professional/REDCINE-X PRO.app/Contents/MacOS/REDline",
    "/usr/local/bin/REDline",
]

PATCH_ROWS, PATCH_COLS = 4, 6

# Bottom row (row 3) = neutral scale, left→right: White → Black
NEUTRAL_ROW   = 3
NEUTRAL_NAMES = ["White", "Near-White", "Light-Gray", "Med-Gray", "Dark-Gray", "Black"]
NEUTRAL_REFL  = [0.900,   0.591,        0.362,        0.198,      0.090,       0.035]

# Patch layout in normalised chart coordinates.
# _MARGIN leaves room for the white border printed on the chart card.
_MARGIN          = 0.06
_SAMPLE_R_FRAC   = 0.30   # sample radius = this * half-cell size

# WB response matrix — empirical, KOMODO-X / IPP2 / BT.709 (inherited from V3)
_DWC_DKELVIN = 7.19e-05
_DGM_DTINT   = 3.29e-03


# ──────────────────────────────────────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Patch:
    idx: int                               # 0-indexed, row-major (0 = top-left)
    row: int
    col: int
    rgb: Tuple[float, float, float]        # [0, 1], trimmed-mean, display-referred
    ire: float                             # display luma * 100
    wc:  float                             # (R-B) / (R+G+B)
    gm:  float                             # (G - 0.5*(R+B)) / (R+G+B)
    n:   int                               # pixels sampled

@dataclass
class Camera:
    camera_id:    str
    tiff_path:    str
    detected:     bool  = False
    note:         str   = ""
    patches:      List[Patch] = field(default_factory=list)
    # Solve outputs (all relative to anchor; anchor = 0)
    exp_stops:    float = 0.0
    kelvin_delta: float = 0.0
    tint_delta:   float = 0.0
    exp_rms:      float = 0.0   # stop-unit residual across 6 neutral patches
    chroma_rms:   float = 0.0   # chromaticity residual across 6 neutral patches
    matrix:       Optional[List] = None    # 3x3 float as list-of-lists


# ──────────────────────────────────────────────────────────────────────────────
# REDLine
# ──────────────────────────────────────────────────────────────────────────────

def find_redline(override: Optional[str]) -> str:
    if override and os.path.isfile(override):
        return override
    for p in REDLINE_SEARCH:
        if os.path.isfile(p):
            return p
    raise FileNotFoundError("REDline not found — pass --redline PATH")

def render(r3d: str, out_base: str, redline: str) -> str:
    """Render frame 0 of an R3D to TIFF. Returns the produced file path."""
    cmd = [
        redline, "--i", r3d, "--o", out_base,
        "--format",        "1",   # TIFF
        "--start",         "0",
        "--frameCount",    "1",
        "--colorSciVersion", "3", # IPP2
        "--colorSpace",    "13",  # BT.709
        "--gammaCurve",    "32",  # BT.1886
        "--outputToneMap", "1",   # medium
        "--rollOff",       "3",   # medium
        "--shadow",        "0.0",
        "--useMeta",
        "--silent",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.strip())
    hits = sorted(glob.glob(out_base + "*.tif*"))
    if not hits:
        raise RuntimeError("No TIFF produced")
    return hits[0]

def camera_id_from_path(path: str) -> str:
    """Extract camera ID from R3D filename, e.g. G007_A106_... → G007_A."""
    stem = Path(path).stem
    m = re.match(r"([A-Z]\d{3}_[A-Z])", stem)
    return m.group(1) if m else stem[:10]


# ──────────────────────────────────────────────────────────────────────────────
# Chart detection
# ──────────────────────────────────────────────────────────────────────────────

def detect_chart(tiff_path: str) -> Optional[np.ndarray]:
    """
    Detect Macbeth ColorChecker in the TIFF.
    Returns (4, 2) float32 corner array [TL, TR, BR, BL] in full-res pixels,
    or None if detection failed.
    """
    if not _MCC_AVAILABLE:
        log.error("cv2.mcc unavailable — install opencv-contrib-python")
        return None

    img = Image.open(tiff_path).convert("RGB")
    w, h = img.size

    # Downscale for detection — MCC is slow on 6K frames
    scale = min(1.0, 2048 / max(w, h))
    det_w, det_h = int(w * scale), int(h * scale)
    det = img.resize((det_w, det_h), Image.LANCZOS)

    bgr = cv2.cvtColor(np.array(det, dtype=np.uint8), cv2.COLOR_RGB2BGR)
    detector = cv2.mcc.CCheckerDetector.create()
    if not detector.process(bgr, cv2.mcc.MCC24, 1):
        return None
    checkers = detector.getListColorChecker()
    if not checkers:
        return None

    box = np.array(checkers[0].getBox(), dtype=np.float32)
    if box.ndim == 3:           # OpenCV sometimes wraps in extra dim
        box = box.reshape(4, 2)

    box = _sort_corners(box)

    # Scale corners back to full resolution
    box[:, 0] *= w / det_w
    box[:, 1] *= h / det_h

    return box

def _sort_corners(pts: np.ndarray) -> np.ndarray:
    """Sort 4 corner points into [TL, TR, BR, BL] order."""
    s = pts.sum(axis=1)         # x+y: min=TL, max=BR
    d = pts[:, 0] - pts[:, 1]  # x-y: max=TR (large x, small y), min=BL
    return np.array([
        pts[np.argmin(s)],      # TL
        pts[np.argmax(d)],      # TR
        pts[np.argmax(s)],      # BR
        pts[np.argmin(d)],      # BL
    ], dtype=np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Patch sampling
# ──────────────────────────────────────────────────────────────────────────────

def _patch_centers(corners: np.ndarray) -> np.ndarray:
    """
    Map all 24 patch centers from normalised chart coordinates to pixel coords.
    Uses bilinear interpolation on the detected quad to handle perspective.
    Returns (24, 2) float32.
    """
    tl, tr, br, bl = corners
    centers = np.zeros((PATCH_ROWS * PATCH_COLS, 2), dtype=np.float32)
    for row in range(PATCH_ROWS):
        for col in range(PATCH_COLS):
            nx = _MARGIN + (col + 0.5) * (1 - 2 * _MARGIN) / PATCH_COLS
            ny = _MARGIN + (row + 0.5) * (1 - 2 * _MARGIN) / PATCH_ROWS
            top = tl + nx * (tr - tl)
            bot = bl + nx * (br - bl)
            centers[row * PATCH_COLS + col] = top + ny * (bot - top)
    return centers

def _cell_half_size(corners: np.ndarray) -> Tuple[float, float]:
    """Half-cell width and height in pixels, derived from chart corner geometry."""
    chart_w = float(np.linalg.norm(corners[1] - corners[0]))  # TR − TL
    chart_h = float(np.linalg.norm(corners[3] - corners[0]))  # BL − TL
    return (
        chart_w * (1 - 2 * _MARGIN) / PATCH_COLS / 2,
        chart_h * (1 - 2 * _MARGIN) / PATCH_ROWS / 2,
    )

def sample_patches(tiff_path: str, corners: np.ndarray) -> List[Patch]:
    """Load TIFF and sample all 24 patches at full resolution."""
    rgb_arr = np.array(Image.open(tiff_path).convert("RGB"), dtype=np.float32) / 255.0
    H, W = rgb_arr.shape[:2]

    centers      = _patch_centers(corners)
    hcw, hch     = _cell_half_size(corners)
    r_px         = min(hcw, hch) * _SAMPLE_R_FRAC

    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    patches: List[Patch] = []

    for idx, (cx, cy) in enumerate(centers):
        mask = (xs - cx) ** 2 + (ys - cy) ** 2 <= r_px ** 2
        if mask.sum() < 16:
            continue

        pix  = rgb_arr[mask]                                        # (N, 3)
        luma = 0.2126 * pix[:, 0] + 0.7152 * pix[:, 1] + 0.0722 * pix[:, 2]

        # Trim top/bottom 10% by luma — removes specular highlights and shadow edge
        p10, p90 = np.percentile(luma, [10, 90])
        keep = (luma >= p10) & (luma <= p90)
        pix  = pix[keep] if keep.sum() >= 8 else pix

        rgb  = tuple(float(v) for v in pix.mean(axis=0))
        luma_mean = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
        ire  = luma_mean * 100.0

        total = sum(rgb)
        if total > 1e-6:
            wc = (rgb[0] - rgb[2]) / total
            gm = (rgb[1] - 0.5 * (rgb[0] + rgb[2])) / total
        else:
            wc = gm = 0.0

        patches.append(Patch(
            idx=idx, row=idx // PATCH_COLS, col=idx % PATCH_COLS,
            rgb=rgb, ire=ire, wc=wc, gm=gm, n=int(keep.sum()),
        ))

    return patches

def neutral_patches(cam: Camera) -> List[Patch]:
    return sorted([p for p in cam.patches if p.row == NEUTRAL_ROW], key=lambda p: p.col)

def orientation_ok(cam: Camera) -> bool:
    """White patch (col 0) should be brighter than Black (col 5)."""
    nps = neutral_patches(cam)
    if len(nps) < 2:
        return True
    return nps[0].ire > nps[-1].ire


# ──────────────────────────────────────────────────────────────────────────────
# Exposure solve
# ──────────────────────────────────────────────────────────────────────────────

def exposure_solve(cam: Camera, anchor: Camera) -> Tuple[float, float]:
    """
    Median log2-luminance delta across paired neutral patches.
    Returns (correction_stops, rms_residual_stops).
    A non-zero rms indicates non-linearity in the camera's tone response —
    a single exposure offset won't fully fix it.
    """
    cn = {p.col: p for p in neutral_patches(cam)}
    an = {p.col: p for p in neutral_patches(anchor)}
    deltas = []
    for col in sorted(set(cn) & set(an)):
        c_ire, a_ire = cn[col].ire, an[col].ire
        if c_ire < 0.5 or a_ire < 0.5:
            continue
        deltas.append(math.log2(a_ire / c_ire))
    if not deltas:
        return 0.0, 0.0
    corr = float(np.median(deltas))
    rms  = float(np.sqrt(np.mean([(d - corr) ** 2 for d in deltas])))
    return corr, rms


# ──────────────────────────────────────────────────────────────────────────────
# WB solve
# ──────────────────────────────────────────────────────────────────────────────

def wb_solve(cam: Camera, anchor: Camera) -> Tuple[float, float, float]:
    """
    Mean WC/GM delta across paired neutral patches, inverted through response matrix.
    Returns (kelvin_delta, tint_delta, chroma_rms).
    High chroma_rms means the neutral patches disagree on WB (heterogeneous
    lighting on the chart — re-shoot with more even fill).
    """
    cn = {p.col: p for p in neutral_patches(cam)}
    an = {p.col: p for p in neutral_patches(anchor)}
    wc_d, gm_d = [], []
    for col in sorted(set(cn) & set(an)):
        wc_d.append(an[col].wc - cn[col].wc)
        gm_d.append(an[col].gm - cn[col].gm)
    if not wc_d:
        return 0.0, 0.0, 0.0
    mwc = float(np.mean(wc_d))
    mgm = float(np.mean(gm_d))
    rms = float(np.sqrt(np.mean(
        [(w - mwc) ** 2 + (g - mgm) ** 2 for w, g in zip(wc_d, gm_d)]
    )))
    return mwc / _DWC_DKELVIN, mgm / _DGM_DTINT, rms


# ──────────────────────────────────────────────────────────────────────────────
# Matrix solve
# ──────────────────────────────────────────────────────────────────────────────

def matrix_solve(cam: Camera, anchor: Camera) -> Optional[np.ndarray]:
    """
    Least-squares 3x3 M such that M @ cam_rgb ≈ anchor_rgb across all 24 patches.
    Returns (3, 3) float64, or None if fewer than 8 patches are shared.

    Off-diagonal magnitude (od_rms) indicates spectral mismatch:
      < 0.01  — well-matched; kelvin/tint is sufficient
      0.01–0.03 — minor matrix correction would help
      > 0.03  — meaningful spectral difference; V5 should include a matrix path
    """
    cm = {p.idx: p for p in cam.patches}
    am = {p.idx: p for p in anchor.patches}
    shared = sorted(set(cm) & set(am))
    if len(shared) < 8:
        return None
    A = np.array([cm[i].rgb for i in shared], dtype=np.float64)
    B = np.array([am[i].rgb for i in shared], dtype=np.float64)
    M = np.zeros((3, 3), dtype=np.float64)
    for c in range(3):
        M[c], *_ = np.linalg.lstsq(A, B[:, c], rcond=None)
    return M

def _od_rms(M: np.ndarray) -> float:
    od = [M[i, j] for i in range(3) for j in range(3) if i != j]
    return float(np.sqrt(np.mean([x ** 2 for x in od])))


# ──────────────────────────────────────────────────────────────────────────────
# Report
# ──────────────────────────────────────────────────────────────────────────────

def print_report(cameras: List[Camera], anchor: Camera,
                 sphere_json: Optional[str]) -> None:

    nps = neutral_patches(anchor)
    print("\n" + "=" * 74)
    print(f"  MACBETH DIAGNOSTIC  —  anchor: {anchor.camera_id}")
    print("=" * 74)

    print(f"\n  Anchor neutral scale (White -> Black):")
    for name, p, refl in zip(NEUTRAL_NAMES, nps, NEUTRAL_REFL):
        bar = "#" * max(1, int(p.ire / 2))
        ref_ire = refl * 100
        flag = " <-- UNEXPECTED" if abs(p.ire - ref_ire) > 12 else ""
        print(f"    {name:<12}  {p.ire:5.1f} IRE   (ref {ref_ire:4.1f})  {bar}{flag}")

    print(f"\n  {'Camera':<12} {'Exp(stops)':>11} {'dKelvin':>9} {'dTint':>7} "
          f"{'ExpRMS':>8} {'WB-RMS':>8}  Matrix OD-RMS")
    print("  " + "-" * 74)

    for cam in cameras:
        if cam.camera_id == anchor.camera_id:
            print(f"  {cam.camera_id:<12}  <- anchor")
            continue
        if not cam.detected:
            print(f"  {cam.camera_id:<12}  DETECTION FAILED  {cam.note}")
            continue

        mat_str = "--"
        if cam.matrix is not None:
            od = _od_rms(np.array(cam.matrix))
            sev = "low" if od < 0.01 else "med" if od < 0.03 else "HIGH"
            mat_str = f"{od:.4f} ({sev})"

        rms_flag = "  <- non-linear" if cam.exp_rms > 0.05 else ""
        print(f"  {cam.camera_id:<12} {cam.exp_stops:+11.4f} "
              f"{cam.kelvin_delta:+9.0f} {cam.tint_delta:+7.2f} "
              f"{cam.exp_rms:8.4f} {cam.chroma_rms:8.6f}  "
              f"{mat_str}{rms_flag}")

    # Cross-validation against sphere solve
    if sphere_json and os.path.isfile(sphere_json):
        try:
            with open(sphere_json) as f:
                sdata = json.load(f)
            cam_map = {c.camera_id: c for c in cameras if c.detected}
            rows = []
            for cid, cdat in sdata.get("cameras", {}).items():
                if cid not in cam_map:
                    continue
                s_exp = float(cdat.get("exposure_adjust", 0))
                m_exp = cam_map[cid].exp_stops
                diff  = m_exp - s_exp
                flag  = "  <- gap > 0.1 stop" if abs(diff) > 0.1 else ""
                rows.append(f"    {cid:<10}  sphere {s_exp:+.4f}  "
                            f"macbeth {m_exp:+.4f}  delta {diff:+.4f}{flag}")
            if rows:
                print(f"\n  Cross-validation vs sphere solve  [{sphere_json}]:")
                print("\n".join(rows))
            else:
                print("\n  Cross-validation: no matching camera IDs in sphere JSON")
        except Exception as e:
            print(f"\n  Could not load sphere JSON: {e}")

    print("\n" + "=" * 74 + "\n")


# ──────────────────────────────────────────────────────────────────────────────
# Serialisation helper
# ──────────────────────────────────────────────────────────────────────────────

def _serial(obj):
    if isinstance(obj, (list, tuple)):
        return [_serial(x) for x in obj]
    if hasattr(obj, "__dataclass_fields__"):
        return {k: _serial(getattr(obj, k)) for k in obj.__dataclass_fields__}
    return obj


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("r3d_dir",      help="Directory with R3D files (one per camera)")
    ap.add_argument("--redline",    metavar="PATH", help="Path to REDline binary")
    ap.add_argument("--out",        default="./macbeth_out", metavar="DIR")
    ap.add_argument("--sphere",     metavar="JSON",
                    help="Sphere solve JSON for cross-validation")
    ap.add_argument("--no-matrix",  action="store_true", help="Skip 3x3 matrix solve")
    ap.add_argument("--keep-tiffs", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s  %(message)s",
    )

    redline = find_redline(args.redline)
    log.info("REDline: %s", redline)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # ── Discover R3Ds ─────────────────────────────────────────────────────────
    # os.walk is used instead of glob — more reliable inside .RDC containers
    # and deep RED folder structures where glob ** can miss nested directories.
    r3ds = []
    for root, _dirs, files in os.walk(args.r3d_dir):
        for f in files:
            if f.upper().endswith(".R3D"):
                r3ds.append(os.path.join(root, f))
    r3ds.sort()
    if not r3ds:
        sys.exit(f"No R3D files found under {args.r3d_dir}")
    log.info("%d R3D file(s) found", len(r3ds))

    cameras: List[Camera] = []

    # ── Phase 1: Render ───────────────────────────────────────────────────────
    for r3d in r3ds:
        cid  = camera_id_from_path(r3d)
        base = str(out / cid)
        log.info("Rendering  %s ...", cid)
        try:
            tiff = render(r3d, base, redline)
            cameras.append(Camera(camera_id=cid, tiff_path=tiff))
            log.info("  -> %s", tiff)
        except Exception as e:
            log.error("Render failed for %s: %s", cid, e)
            cameras.append(Camera(camera_id=cid, tiff_path="", note=str(e)))

    # ── Phase 2: Detect + sample ──────────────────────────────────────────────
    for cam in cameras:
        if not cam.tiff_path:
            continue
        log.info("Detecting  %s ...", cam.camera_id)
        corners = detect_chart(cam.tiff_path)
        if corners is None:
            cam.note = "MCC detector: chart not found"
            log.warning("  %s: detection failed", cam.camera_id)
            continue
        patches = sample_patches(cam.tiff_path, corners)
        if len(patches) < 18:
            cam.note = f"only {len(patches)}/24 patches sampled"
            log.warning("  %s: %s", cam.camera_id, cam.note)
        else:
            cam.detected = True
            cam.patches  = patches
            if not orientation_ok(cam):
                log.warning("  %s: chart may be horizontally flipped -- verify orientation",
                            cam.camera_id)
            log.info("  %s: %d patches sampled", cam.camera_id, len(patches))

    # ── Phase 3: Anchor ───────────────────────────────────────────────────────
    good = [c for c in cameras if c.detected]
    if not good:
        sys.exit("No cameras with successful chart detection. "
                 "Check TIFF quality, chart size, and framing.")

    def _mean_neutral_ire(c: Camera) -> float:
        nps = neutral_patches(c)
        return float(np.mean([p.ire for p in nps])) if nps else 0.0

    median_ire = float(np.median([_mean_neutral_ire(c) for c in good]))
    anchor     = min(good, key=lambda c: abs(_mean_neutral_ire(c) - median_ire))
    log.info("Anchor: %s  (mean neutral IRE %.1f)", anchor.camera_id,
             _mean_neutral_ire(anchor))

    # ── Phase 4: Solve ────────────────────────────────────────────────────────
    for cam in good:
        cam.exp_stops,    cam.exp_rms    = exposure_solve(cam, anchor)
        cam.kelvin_delta, cam.tint_delta, cam.chroma_rms = wb_solve(cam, anchor)
        if not args.no_matrix:
            M = matrix_solve(cam, anchor)
            cam.matrix = M.tolist() if M is not None else None

    # ── Phase 5: Report ───────────────────────────────────────────────────────
    print_report(cameras, anchor, args.sphere)

    # ── Phase 6: JSON ─────────────────────────────────────────────────────────
    out_json = out / "macbeth_diag.json"
    with open(out_json, "w") as f:
        json.dump({"anchor": anchor.camera_id, "cameras": _serial(cameras)}, f, indent=2)
    log.info("JSON saved -> %s", out_json)

    # ── Cleanup ───────────────────────────────────────────────────────────────
    if not args.keep_tiffs:
        for cam in cameras:
            if cam.tiff_path and os.path.isfile(cam.tiff_path):
                try:
                    os.unlink(cam.tiff_path)
                except OSError:
                    pass


if __name__ == "__main__":
    main()
