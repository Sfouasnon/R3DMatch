#!/usr/bin/env python3
"""
golden_regression.py — Reference-path identity guard for R3DMatch v4.

Purpose
-------
v4 keeps the v3 *reference* render + sphere solver byte-identical. As v4 grows
(color pipeline engine, gray-card / face methods, UI), this harness proves the
reference path still produces the SAME solve outputs it did at the v3 baseline.

It captures the solve-relevant fields for a footage set into a golden JSON,
then later re-runs and diffs against that golden. Renders are reused
(--reuse-renders, the default) so REDLine/GPU pixel jitter is held fixed and
only the *code path* is under test.

Run on the Mac (REDLine + footage required). Example:

    source /Users/sfouasnon/Desktop/R3DMatch_v4/.venv/bin/activate   # once set up
    cd /Users/sfouasnon/Desktop/R3DMatch_v4

    # 1) capture the baseline once, on a clean checkout of the v3-identical copy
    python3 tools/golden_regression.py capture \
        --input /Users/sfouasnon/Desktop/Test_Run/SatJune6_840PM_GrayonGray \
        --out   /tmp/r3dmatch_v4_106_run \
        --golden tools/golden/_106.json

    # 2) after ANY v4 change, re-run and assert the reference path is unchanged
    python3 tools/golden_regression.py compare \
        --input /Users/sfouasnon/Desktop/Test_Run/SatJune6_840PM_GrayonGray \
        --out   /tmp/r3dmatch_v4_106_run \
        --golden tools/golden/_106.json

Exit code is non-zero on any mismatch, so it can gate a build.

Tolerances
----------
Detection ROI, commits (exposureAdjust/kelvin/tint), and hero IRE/log2 are
compared EXACTLY by default — given reused renders, the reference path must be
deterministic. Use --tol to allow a tiny float epsilon if a justified source of
nondeterminism is identified (document it if you do).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

# Make the in-tree package importable without installing.
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from r3dmatch3.workflow import run_analysis  # noqa: E402


# Fields we lock. These are the outputs that drive the on-set decision and the
# RCP2 push; if any of them drifts on the reference path, v4 has regressed.
def _camera_fingerprint(cam) -> dict:
    det = cam.detection
    meas = cam.measurement
    com = cam.commit
    roi = None
    if det is not None and det.roi is not None:
        roi = {"cx": det.roi.cx, "cy": det.roi.cy, "r": det.roi.r}
    elif meas is not None and meas.roi is not None:
        roi = {"cx": meas.roi.cx, "cy": meas.roi.cy, "r": meas.roi.r}
    return {
        "clip_id": cam.clip_id,
        "camera_label": cam.camera_label,
        "status": cam.status,
        "detection_status": det.status if det else None,
        "detection_source": det.source if det else None,
        "roi": roi,
        "hero_ire": meas.hero_ire if meas else None,
        "hero_log2": meas.hero_log2 if meas else None,
        "measured_rgb_mean": list(meas.measured_rgb_mean) if meas else None,
        "exposure_adjust": com.exposure_adjust if com else None,
        "kelvin": com.kelvin if com else None,
        "tint": com.tint if com else None,
        "match_pct": cam.match_pct,
        "exposure_match_pct": cam.exposure_match_pct,
        "wb_match_pct": cam.wb_match_pct,
    }


def _run(input_path: str, out_dir: str, reuse: bool, disable_priors: bool,
         strategy: str = "median", gray_target_ire: float = 33.3) -> dict:
    kwargs = dict(
        out_dir=out_dir,
        reuse_renders=reuse,
        render_corrected=True,
        read_lens_metadata=True,
        disable_priors=disable_priors,
    )
    # Thread strategy only if this build's run_analysis accepts it (keeps the
    # harness usable against older checkouts).
    import inspect
    _params = inspect.signature(run_analysis).parameters
    if "strategy" in _params:
        kwargs["strategy"] = strategy
    if "gray_target_ire" in _params:
        kwargs["gray_target_ire"] = gray_target_ire
    result = run_analysis(input_path, **kwargs)
    cams = sorted(result.cameras, key=lambda c: c.camera_label)
    return {
        "input_path": input_path,
        "priors_disabled": disable_priors,
        "strategy": strategy,
        "assessment_status": result.assessment_status,
        "anchor_ire": result.anchor_ire,
        "anchor_log2": result.anchor_log2,
        "anchor_source": getattr(result, "anchor_source", None),
        "shared_kelvin": result.wb_solve.shared_kelvin if result.wb_solve else None,
        "cameras": [_camera_fingerprint(c) for c in cams],
    }


def _close(a, b, tol: float) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if a is True or a is False or b is True or b is False:
            return a == b
        return math.isclose(float(a), float(b), abs_tol=tol, rel_tol=0.0)
    return a == b


def _diff(golden: dict, fresh: dict, tol: float) -> list[str]:
    problems: list[str] = []
    for key in ("assessment_status", "anchor_ire", "anchor_log2", "shared_kelvin"):
        if not _close(golden.get(key), fresh.get(key), tol):
            problems.append(f"run.{key}: golden={golden.get(key)!r} fresh={fresh.get(key)!r}")

    g_by_id = {c["clip_id"]: c for c in golden["cameras"]}
    f_by_id = {c["clip_id"]: c for c in fresh["cameras"]}
    if set(g_by_id) != set(f_by_id):
        problems.append(
            f"camera set changed: only-in-golden={sorted(set(g_by_id) - set(f_by_id))} "
            f"only-in-fresh={sorted(set(f_by_id) - set(g_by_id))}"
        )

    for clip_id in sorted(set(g_by_id) & set(f_by_id)):
        gc, fc = g_by_id[clip_id], f_by_id[clip_id]
        for field in gc:
            gv, fv = gc[field], fc.get(field)
            if field == "roi":
                if (gv is None) != (fv is None):
                    problems.append(f"{clip_id}.roi presence: golden={gv} fresh={fv}")
                elif gv is not None:
                    for k in ("cx", "cy", "r"):
                        if not _close(gv[k], fv[k], tol):
                            problems.append(f"{clip_id}.roi.{k}: golden={gv[k]} fresh={fv[k]}")
            elif field == "measured_rgb_mean":
                if (gv is None) != (fv is None):
                    problems.append(f"{clip_id}.measured_rgb_mean presence differs")
                elif gv is not None:
                    for i, (g1, f1) in enumerate(zip(gv, fv)):
                        if not _close(g1, f1, tol):
                            problems.append(f"{clip_id}.measured_rgb_mean[{i}]: golden={g1} fresh={f1}")
            elif not _close(gv, fv, tol):
                problems.append(f"{clip_id}.{field}: golden={gv!r} fresh={fv!r}")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["capture", "compare"])
    ap.add_argument("--input", required=True, help="footage set directory (R3D clips)")
    ap.add_argument("--out", required=True, help="working output dir for the run")
    ap.add_argument("--golden", required=True, help="path to the golden JSON")
    ap.add_argument("--tol", type=float, default=0.0, help="abs tolerance for float compares (default 0 = exact)")
    ap.add_argument("--strategy", default="median", choices=["median", "gray_anchor"],
                    help="exposure matching strategy to capture/compare (default: median)")
    ap.add_argument("--gray-target-ire", type=float, default=33.3,
                    help="gray_anchor Log3G10 IRE target (default: 33.3 = 18%% gray)")
    ap.add_argument("--no-reuse", action="store_true", help="force fresh renders (NOT recommended for identity check)")
    ap.add_argument("--use-priors", action="store_true",
                    help="enable the live sphere-profile priors (NOT recommended: the profile store "
                         "mutates across runs, making the golden non-deterministic). Default: priors disabled.")
    args = ap.parse_args()

    fresh = _run(args.input, args.out, reuse=not args.no_reuse, disable_priors=not args.use_priors,
                 strategy=args.strategy, gray_target_ire=args.gray_target_ire)

    golden_path = Path(args.golden)
    if args.mode == "capture":
        golden_path.parent.mkdir(parents=True, exist_ok=True)
        golden_path.write_text(json.dumps(fresh, indent=2, sort_keys=True), encoding="utf-8")
        print(f"Captured golden for {len(fresh['cameras'])} cameras -> {golden_path}")
        return 0

    if not golden_path.exists():
        print(f"ERROR: golden not found: {golden_path}", file=sys.stderr)
        return 2
    golden = json.loads(golden_path.read_text(encoding="utf-8"))
    problems = _diff(golden, fresh, args.tol)
    if not problems:
        print(f"PASS: reference path identical to golden ({len(fresh['cameras'])} cameras, tol={args.tol})")
        return 0
    print(f"FAIL: {len(problems)} difference(s) vs golden:", file=sys.stderr)
    for p in problems:
        print(f"  - {p}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
