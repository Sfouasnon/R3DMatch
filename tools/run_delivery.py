#!/usr/bin/env python3
"""
run_delivery.py — run R3DMatch v4 with an EXPLICIT per-run delivery look.

Models the Phase 4 contract: the operator must explicitly choose the delivery
look before each run — there is no silent default. You must pass exactly one of:

  --reference-only           score only in the BT.709 reference look
  <delivery flags / --lut>   score also in the chosen delivery look

The reference solve (exposureAdjust/kelvin/tint) is identical either way; the
delivery pipeline only adds the operator-facing delivery match %.

Examples (Mac, REDLine + footage):

  # self-check: delivery == reference  -> delivery match % should ~equal reference
  python3 tools/run_delivery.py --input <set> --out ~/Desktop/run_self \
      --delivery-name "Rec.709 (==reference)"

  # transform-only difference
  python3 tools/run_delivery.py --input <set> --out ~/Desktop/run_tm \
      --delivery-name "ToneMap=None" --delivery-tonemap 3

  # Rec.709 + show LUT (when available)
  python3 tools/run_delivery.py --input <set> --out ~/Desktop/run_lut \
      --delivery-name "Show LUT" --delivery-creative-lut luts/show.cube

  # explicit reference-only (no project look)
  python3 tools/run_delivery.py --input <set> --out ~/Desktop/run_ref --reference-only
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from r3dmatch3.workflow import run_analysis          # noqa: E402
from r3dmatch3.colorpipeline import ColorPipeline     # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--reference-only", action="store_true",
                    help="explicitly score only in the reference look (no delivery pipeline)")
    ap.add_argument("--delivery-name", default=None, help="label for the delivery look")
    ap.add_argument("--delivery-color-space", default="13")
    ap.add_argument("--delivery-gamma", default="32")
    ap.add_argument("--delivery-tonemap", default="1")
    ap.add_argument("--delivery-rolloff", default="3")
    ap.add_argument("--delivery-creative-lut", default=None)
    ap.add_argument("--use-priors", action="store_true",
                    help="enable live sphere priors (default off for determinism)")
    args = ap.parse_args()

    # Enforce explicit choice — no silent default look.
    chose_delivery = any([
        args.delivery_name, args.delivery_creative_lut,
        args.delivery_color_space != "13", args.delivery_gamma != "32",
        args.delivery_tonemap != "1", args.delivery_rolloff != "3",
    ])
    if args.reference_only == bool(chose_delivery):
        print("ERROR: choose exactly one — either --reference-only OR a delivery look "
              "(--delivery-name / --delivery-* / --delivery-creative-lut).", file=sys.stderr)
        return 2

    delivery = None
    if not args.reference_only:
        delivery = ColorPipeline(
            color_space=args.delivery_color_space,
            gamma_curve=args.delivery_gamma,
            output_tone_map=args.delivery_tonemap,
            roll_off=args.delivery_rolloff,
            creative_lut_path=args.delivery_creative_lut,
            name=args.delivery_name or "delivery",
        )
        if delivery.is_reference:
            print("NOTE: the selected delivery look is identical to the reference "
                  "pipeline (no transform/LUT difference). Delivery match would equal "
                  "reference, so no separate delivery render is performed.\n")

    result = run_analysis(
        args.input, out_dir=args.out, reuse_renders=True, render_corrected=True,
        disable_priors=not args.use_priors, delivery_pipeline=delivery,
    )

    look = "Reference only" if delivery is None else result.delivery_pipeline_name
    print(f"\n=== Run summary — delivery look: {look} ===")
    print(f"{'camera':<12} {'ref match':>10} {'delivery match':>16}")
    for c in sorted(result.cameras, key=lambda c: c.camera_label):
        if c.match_pct is None:
            continue
        ref = f"{c.match_pct:.0f}%"
        dlv = "—" if c.delivery_match_pct is None else f"{c.delivery_match_pct:.0f}%"
        print(f"{c.camera_label:<12} {ref:>10} {dlv:>16}")
    amp = result.array_match_pct
    print(f"\nreference array match : {amp:.0f}%" if amp is not None else "reference array match : —")
    if result.delivery_array_match_pct is not None:
        print(f"delivery  array match : {result.delivery_array_match_pct:.0f}%  "
              f"(worst {result.delivery_min_match_pct:.0f}% @ {result.delivery_min_match_clip_id})")
        if result.delivery_profile:
            p = result.delivery_profile
            print(f"delivery neutral wc/gm: {p['neutral_wc']:+.4f} / {p['neutral_gm']:+.4f}  "
                  f"(fresh profile, n={p['n_samples']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
