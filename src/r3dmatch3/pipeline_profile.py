"""
pipeline_profile.py — R3DMatch v4 delivery-pipeline characterization.

The reference path (sphere solver + measure.py) is untouched and measures on the
frozen IPP2/BT.709 reference render. This module characterizes a *delivery*
pipeline (the project's transform + show LUT) so the hybrid verification can
report match % in the look the operator actually sees.

A PipelineProfile captures, for one delivery ColorPipeline:
  * luma_weights   — output-space luminance coefficients (BT.709 for a Rec.709
                     show LUT; BT.2020 / P3 for wide-gamut deliveries).
  * neutral_wc/gm  — where an 18% neutral lands in the delivery domain (the LUT
                     can push neutral off equal-RGB; this is the WB target there).
  * tonal_map      — paired (reference_log2, delivery_log2) samples spanning the
                     tonal range, so an exposure residual measured on the
                     reference render can be expressed in the delivery look.

Characterization is empirical (measured from paired reference+delivery renders of
the same frames), never modeled — a creative LUT is not analytically invertible.
This mirrors v3's measurement philosophy: measure, don't assume.

Luma weights are looked up from the output color space (REDLine --colorSpace code).
For a Rec.709 delivery (incl. Rec.709 + creative LUT) the weights equal the
reference, and the LUT'd pixels are measured directly with the proven 4-step math.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# Standard output-space luminance coefficients, keyed by REDLine --colorSpace code.
# BT.709 and BT.2020 are exact (ITU). P3 values are the commonly used D65 luma
# coefficients; RWG is scene-linear (fallback to BT.709 for display measurement).
_BT709 = (0.2126, 0.7152, 0.0722)
STANDARD_LUMA_WEIGHTS: Dict[str, Tuple[float, float, float]] = {
    "1":  _BT709,                       # BT.709
    "13": _BT709,                       # BT.709 (alt code)
    "24": (0.2627, 0.6780, 0.0593),     # BT.2020
    "25": _BT709,                       # REDWideGamutRGB (scene-linear; fallback)
    "26": (0.2096, 0.7215, 0.0690),     # DCI-P3
    "27": (0.2096, 0.7215, 0.0690),     # DCI-P3 D65
}


def luma_weights_for(color_space: str) -> Tuple[float, float, float]:
    """Output-space luminance weights for a REDLine --colorSpace code (BT.709 fallback)."""
    return STANDARD_LUMA_WEIGHTS.get(str(color_space), _BT709)


@dataclass
class PipelineProfile:
    """Empirical characterization of one delivery ColorPipeline."""
    pipeline_name: str
    color_space: str
    luma_weights: Tuple[float, float, float]
    neutral_wc: float = 0.0          # where neutral lands in delivery domain
    neutral_gm: float = 0.0
    # paired tonal samples, sorted ascending by reference_log2
    tonal_ref_log2: List[float] = field(default_factory=list)
    tonal_delivery_log2: List[float] = field(default_factory=list)
    n_samples: int = 0

    def delivery_log2_for(self, reference_log2: float) -> float:
        """Map a reference-domain log2 luminance to the delivery domain.

        Linear interpolation over the measured tonal samples; clamps to the
        sampled range. With <2 samples, returns the input unchanged (identity).
        """
        if len(self.tonal_ref_log2) < 2:
            return float(reference_log2)
        xs = np.asarray(self.tonal_ref_log2, dtype=np.float64)
        ys = np.asarray(self.tonal_delivery_log2, dtype=np.float64)
        order = np.argsort(xs)
        return float(np.interp(float(reference_log2), xs[order], ys[order]))

    def to_dict(self) -> Dict:
        return {
            "pipeline_name": self.pipeline_name,
            "color_space": self.color_space,
            "luma_weights": list(self.luma_weights),
            "neutral_wc": self.neutral_wc,
            "neutral_gm": self.neutral_gm,
            "tonal_ref_log2": self.tonal_ref_log2,
            "tonal_delivery_log2": self.tonal_delivery_log2,
            "n_samples": self.n_samples,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "PipelineProfile":
        return cls(
            pipeline_name=d["pipeline_name"],
            color_space=str(d["color_space"]),
            luma_weights=tuple(d["luma_weights"]),
            neutral_wc=float(d.get("neutral_wc", 0.0)),
            neutral_gm=float(d.get("neutral_gm", 0.0)),
            tonal_ref_log2=list(d.get("tonal_ref_log2", [])),
            tonal_delivery_log2=list(d.get("tonal_delivery_log2", [])),
            n_samples=int(d.get("n_samples", 0)),
        )


@dataclass
class CharSample:
    """One paired sample: a clip measured on both reference and delivery renders."""
    reference_log2: float
    delivery_log2: float
    delivery_wc: float
    delivery_gm: float
    is_neutral: bool = True   # gray sphere/card = neutral; faces would be False


def build_profile(
    pipeline_name: str,
    color_space: str,
    samples: List[CharSample],
) -> PipelineProfile:
    """Build a PipelineProfile from paired reference+delivery measurements.

    neutral_wc/gm = median over neutral samples (where gray lands in delivery).
    tonal map = the (reference_log2, delivery_log2) pairs across all samples.
    """
    weights = luma_weights_for(color_space)
    neutral = [s for s in samples if s.is_neutral]
    neutral_wc = float(np.median([s.delivery_wc for s in neutral])) if neutral else 0.0
    neutral_gm = float(np.median([s.delivery_gm for s in neutral])) if neutral else 0.0

    pairs = sorted(((s.reference_log2, s.delivery_log2) for s in samples), key=lambda p: p[0])
    ref_log2 = [p[0] for p in pairs]
    dlv_log2 = [p[1] for p in pairs]

    return PipelineProfile(
        pipeline_name=pipeline_name,
        color_space=str(color_space),
        luma_weights=weights,
        neutral_wc=neutral_wc,
        neutral_gm=neutral_gm,
        tonal_ref_log2=ref_log2,
        tonal_delivery_log2=dlv_log2,
        n_samples=len(samples),
    )
