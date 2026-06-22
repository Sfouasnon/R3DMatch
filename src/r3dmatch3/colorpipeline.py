"""
colorpipeline.py — R3DMatch v4 color-science pipeline descriptor.

ONE place describes how REDLine renders a frame: output color science, primaries,
transfer, tone-map, roll-off, plus optional creative LUT / output LUT / CDL.

Two roles exist at runtime:

  * REFERENCE_PIPELINE — frozen, identical to v3's hardcoded render
    (IPP2 / BT.709 / BT.1886 / Medium tone-map / Medium roll-off). This is the
    domain the sphere solver + measurement math were proven on. It is NOT
    project-configurable and must never change, or the golden baselines break.

  * a project "delivery" pipeline — the show's chosen IPP2 transform + LUT,
    used (in later phases) for the operator-facing verification / match %.

The reference pipeline emits exactly the same REDLine flags, in the same order,
as v3 did. `to_redline_color_args()` for REFERENCE_PIPELINE returns:

    --colorSciVersion 3 --colorSpace 13 --gammaCurve 32
    --outputToneMap 1 --rollOff 3 --shadow 0.0

LUT / CDL args are appended ONLY when set, so a reference render is byte-identical
to v3 and the golden_regression goldens keep passing.

REDLine flag reference (Build 65.1.3 / R3DSDK 9.2.0, confirmed):
  --colorSciVersion : Current=0, V1=1, FLUT=2, IPP2=3
  --colorSpace      : BT.709=1|13, REDWideGamutRGB=25, BT.2020=24, DCI-P3=26, DCI-P3 D65=27, ...
  --gammaCurve      : BT.1886=32, Log3G10=34, Log3G12=33, ST2084=31, HLG=35, Gamma2.2=36, Gamma2.6=37, linear=-1, ...
  --outputToneMap   : Low=0, Medium=1, High=2, None=3
  --rollOff         : None=0, Hard=1, Default=2, Medium=3, Soft=4
  --creativeLut <f> : IPP2 creative/show LUT (applied inside the IPP2 pipeline)
  --lut <f> --lutEdgeLength <n> : generic pre-baked output 3D LUT
  --cdl* (slope/offset/power), --cdlSaturation : ASC CDL
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass(frozen=True)
class CDL:
    """ASC CDL — per-channel slope/offset/power + saturation."""
    slope: tuple = (1.0, 1.0, 1.0)
    offset: tuple = (0.0, 0.0, 0.0)
    power: tuple = (1.0, 1.0, 1.0)
    saturation: float = 1.0

    def to_redline_args(self) -> List[str]:
        sr, sg, sb = self.slope
        orr, og, ob = self.offset
        pr, pg, pb = self.power
        return [
            "--cdlRedSlope", f"{sr:.6f}", "--cdlGreenSlope", f"{sg:.6f}", "--cdlBlueSlope", f"{sb:.6f}",
            "--cdlRedOffset", f"{orr:.6f}", "--cdlGreenOffset", f"{og:.6f}", "--cdlBlueOffset", f"{ob:.6f}",
            "--cdlRedPower", f"{pr:.6f}", "--cdlGreenPower", f"{pg:.6f}", "--cdlBluePower", f"{pb:.6f}",
            "--cdlSaturation", f"{self.saturation:.6f}",
        ]


@dataclass(frozen=True)
class ColorPipeline:
    """A complete REDLine color-science configuration.

    Defaults reproduce v3's reference render exactly. Only fields a project
    overrides differ; LUT/CDL stay None so reference renders are unchanged.
    """
    color_sci_version: str = "3"      # IPP2
    color_space: str = "13"           # BT.709
    gamma_curve: str = "32"           # BT.1886
    output_tone_map: str = "1"        # Medium
    roll_off: str = "3"               # Medium
    shadow: str = "0.0"
    creative_lut_path: Optional[str] = None   # --creativeLut
    output_lut_path: Optional[str] = None     # --lut
    output_lut_edge: Optional[int] = None     # --lutEdgeLength (with --lut)
    cdl: Optional[CDL] = None
    name: str = "IPP2 BT.709 (reference)"

    def to_redline_color_args(self) -> List[str]:
        """Build the REDLine color-science argument list.

        For the reference pipeline this is byte-for-byte what v3 emitted.
        LUT/CDL flags are appended only when set.
        """
        args: List[str] = [
            "--colorSciVersion", self.color_sci_version,
            "--colorSpace", self.color_space,
            "--gammaCurve", self.gamma_curve,
            "--outputToneMap", self.output_tone_map,
            "--rollOff", self.roll_off,
            "--shadow", self.shadow,
        ]
        if self.creative_lut_path:
            args += ["--creativeLut", str(self.creative_lut_path)]
        if self.output_lut_path:
            args += ["--lut", str(self.output_lut_path)]
            if self.output_lut_edge:
                args += ["--lutEdgeLength", str(int(self.output_lut_edge))]
        if self.cdl is not None:
            args += self.cdl.to_redline_args()
        return args

    @property
    def is_reference(self) -> bool:
        """True if this pipeline is the frozen reference (no LUT/CDL, default transform)."""
        return (
            self.color_sci_version == "3"
            and self.color_space == "13"
            and self.gamma_curve == "32"
            and self.output_tone_map == "1"
            and self.roll_off == "3"
            and self.shadow == "0.0"
            and not self.creative_lut_path
            and not self.output_lut_path
            and self.cdl is None
        )


# The frozen reference. Do not mutate. The sphere solver, measurement math, and
# golden baselines all assume renders produced with exactly this configuration.
REFERENCE_PIPELINE = ColorPipeline()

# Scene-linear measurement pipeline — REDWideGamutRGB, linear gamma, tone map and
# roll-off OFF (REDLine codes: outputToneMap None=3, rollOff None=0). Used ONLY to
# measure the sphere's true scene-linear value for EXPOSURE matching, which must be
# solved in linear (exposureAdjust is a scene-linear stop). Never used for the
# reference measurement, WB, or scoring.
SCENE_LINEAR_PIPELINE = ColorPipeline(
    color_space="25",      # REDWideGamutRGB
    gamma_curve="-1",      # linear
    output_tone_map="3",   # None
    roll_off="0",          # None
    name="scene-linear RWG (exposure)",
)
