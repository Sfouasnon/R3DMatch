#!/usr/bin/env python3
"""
characterize_pipeline.py — build a delivery PipelineProfile from real footage.

Phase 3 of R3DMatch v4. Runs the proven reference pass (run_analysis, priors
disabled) to get each camera's sphere ROI + reference hero log2, then renders the
same frames through a chosen DELIVERY pipeline (transform + optional show LUT) and
measures them with the delivery-domain hero measurement. The paired samples become
a PipelineProfile JSON used later for hybrid match % in the delivered look.

The reference path is unchanged — this only adds delivery renders + measurement.

Run on the Mac (REDLine + footage required). Examples:

  # Degenerate self-check: delivery == reference. tonal map should be ~identity,
  # neutral_wc/gm should match the reference sphere. Proves the delivery
  # measurement aligns with the reference path.
  python3 tools/characterize_pipeline.py \
      --input /Users/sfouasnon/Desktop/Test_Footage/GraySphere_GrayBackdrop_FocusChart \
      --out   /tmp/char_selfcheck \
      --profile tools/profiles/selfcheck.json \
      --name "Rec.709 (==reference)"

  # Transform-only difference: output tone map None instead of Medium.
  python3 tools/characterize_pipeline.py --input <set> --out /tmp/char_tm \
      --profile tools/profiles/tonemap_none.json --name "Rec.709 ToneMap=None" \
      --delivery-tonemap 3

  # Rec.709 + a project show LUT (when you have one):
  python3 tools/characterize_pipeline.py --input <set> --out /tmp/char_lut \
      --profile tools/profiles/show.json --name "Show LUT" \
      --delivery-creative-lut /Users/sfouasnon/Desktop/R3DMatch_v4/luts/show.cube
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from r3dmatch3.workflow import run_analysis            # noqa: E402
from r3dmatch3.redline import render_measurement_frame, resolve_redline_executable  # noqa: E402
from r3dmatch3.colorpipeline import ColorPipeline      # noqa: E402
from r3dmatch3.measure_delivery import measure_delivery_hero  # noqa: E402
from r3dmatch3.pipeline_profile import build_profile, luma_weights_for, CharSample  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="footage set directory (R3D clips)")
    ap.add_argument("--out", required=True, help="working dir (reference + delivery renders)")
    ap.add_argument("--profile", required=True, help="output PipelineProfile JSON path")
    ap.add_argument("--name", default="delivery", help="human label for this pipeline")
    # Delivery pipeline (defaults == reference, i.e. the self-check)
    ap.add_argument("--delivery-color-space", default="13")
    ap.add_argument("--delivery-gamma", default="32")
    ap.add_argument("--delivery-tonemap", default="1")
    ap.add_argument("--delivery-rolloff", default="3")
    ap.add_argument("--delivery-creative-lut", default=None, help="path to IPP2 show LUT (.cube)")
    args = ap.parse_args()

    delivery = ColorPipeline(
        color_space=args.delivery_color_space,
        gamma_curve=args.delivery_gamma,
        output_tone_map=args.delivery_tonemap,
        roll_off=args.delivery_rolloff,
        creative_lut_path=args.delivery_creative_lut,
        name=args.name,
    )
    weights = luma_weights_for(delivery.color_space)

    # 1) Reference pass — proven path, priors disabled for determinism.
    result = run_analysis(args.input, out_dir=args.out, reuse_renders=True,
                          render_corrected=False, disable_priors=True)

    redline = resolve_redline_executable()
    delivery_dir = Path(args.out) / "delivery_renders"
    delivery_dir.mkdir(parents=True, exist_ok=True)

    samples = []
    for cr in result.cameras:
        if not cr.is_usable() or cr.detection is None or cr.detection.roi is None:
            print(f"  skip {cr.clip_id}: no usable reference measurement")
            continue
        out_tiff = delivery_dir / f"{cr.clip_id}.delivery.tiff"
        rr = render_measurement_frame(
            cr.source_path, str(out_tiff), redline=redline,
            frame_index=0, use_as_shot=True, pipeline=delivery,
        )
        if not rr["ok"]:
            print(f"  skip {cr.clip_id}: delivery render failed ({rr.get('status')})")
            continue
        dm = measure_delivery_hero(rr["output_path"], cr.detection.roi, luma_weights=weights)
        if not dm["valid"]:
            print(f"  skip {cr.clip_id}: delivery measurement invalid")
            continue
        samples.append(CharSample(
            reference_log2=cr.measurement.hero_log2,
            delivery_log2=float(dm["hero_log2"]),
            delivery_wc=float(dm["wc"]),
            delivery_gm=float(dm["gm"]),
            is_neutral=True,   # sphere/gray card; faces would set False
        ))
        print(f"  {cr.camera_label}: ref_log2={cr.measurement.hero_log2:+.4f} "
              f"delivery_log2={dm['hero_log2']:+.4f} wc={dm['wc']:+.4f} gm={dm['gm']:+.4f}")

    if not samples:
        print("ERROR: no usable samples — cannot build profile", file=sys.stderr)
        return 1

    profile = build_profile(args.name, delivery.color_space, samples)
    out = Path(args.profile)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(profile.to_dict(), indent=2, sort_keys=True), encoding="utf-8")

    # Quick self-check signal: mean |delivery - reference| log2 shift.
    import numpy as np
    shift = float(np.mean([abs(s.delivery_log2 - s.reference_log2) for s in samples]))
    print(f"\nProfile written: {out}")
    print(f"  pipeline      : {args.name}  (colorSpace={delivery.color_space}, "
          f"lut={'yes' if delivery.creative_lut_path else 'no'})")
    print(f"  luma weights  : {weights}")
    print(f"  neutral wc/gm : {profile.neutral_wc:+.4f} / {profile.neutral_gm:+.4f}")
    print(f"  samples       : {profile.n_samples}")
    print(f"  mean |Δlog2|  : {shift:.4f}  "
          f"({'~identity (delivery≈reference)' if shift < 0.02 else 'delivery differs from reference'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
