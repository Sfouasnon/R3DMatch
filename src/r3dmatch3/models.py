"""
R3DMatch v2 — Core Data Models

All dataclasses used throughout the pipeline. No business logic here.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple
import json


# ---------------------------------------------------------------------------
# Clip identity
# ---------------------------------------------------------------------------

@dataclass
class ClipMetadata:
    """Everything read from REDLine --printMeta 1 (and optionally --printMeta 5)."""
    clip_id: str                        # e.g. "G007_A106_0511R9_001"
    source_path: str                    # absolute path to .R3D
    camera_model: str                   # "KOMODO-X 6K S35"
    camera_pin: str                     # "KXBK1014168"
    camera_position: str                # "A"
    camera_label: str                   # "G007_A106" (reel + position)
    reel: str                           # "007"
    clip_num: str                       # "106"
    kelvin: int                         # as-shot Kelvin from camera
    tint: float                         # as-shot tint from camera
    iso: int                            # ISO
    fps: float                          # record FPS
    frame_width: int                    # 5760
    frame_height: int                   # 3240
    total_frames: int                   # 1 for preflight grabs
    timecode: str                       # "10:57:17:00"
    color_space: int                    # 25 = REDWideGamutRGB
    gamma_space: int                    # 34 = Log3G10
    image_pipeline: str                 # "IPP2"
    # Optional lens data from --printMeta 5
    focal_length_mm: Optional[int] = None
    aperture: Optional[float] = None
    focus_distance_mm: Optional[int] = None
    lens_name: Optional[str] = None     # from mode 1 "Lens:" field

    def has_lens_data(self) -> bool:
        return (self.focal_length_mm or 0) > 0 or (self.aperture or 0) > 0.0


# ---------------------------------------------------------------------------
# Sphere geometry
# ---------------------------------------------------------------------------

@dataclass
class SphereROI:
    """Circle in pixel coordinates (HWC image space)."""
    cx: float       # center x
    cy: float       # center y
    r: float        # radius

    def to_dict(self) -> Dict[str, float]:
        return {"cx": self.cx, "cy": self.cy, "r": self.r}

    @classmethod
    def from_dict(cls, d: Dict) -> "SphereROI":
        return cls(cx=float(d["cx"]), cy=float(d["cy"]), r=float(d["r"]))


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

@dataclass
class DetectionGateResult:
    """Result of a single detection gate."""
    gate: str
    passed: bool
    reason: str = ""
    diagnostics: Dict = field(default_factory=dict)


@dataclass
class SphereDetectionResult:
    """Full result of sphere detection for one clip."""
    clip_id: str
    status: str                         # "SUCCESS" | "FAILED" | "MANUAL"
    roi: Optional[SphereROI]
    source: str                         # "auto_hough" | "manual_operator"
    gates: List[DetectionGateResult] = field(default_factory=list)
    failed_gate: str = ""
    failure_reason: str = ""
    # Diagnostics
    hough_accumulator: float = 0.0
    radius_ratio: float = 0.0          # r / min(H, W)
    interior_luminance_mean: float = 0.0
    interior_luminance_stddev: float = 0.0
    chromaticity_distance: float = 0.0 # distance from (1/3, 1/3)
    ire_spread: float = 0.0            # bright_ire - dark_ire
    lambertian_score: float = 0.0      # 0-1
    best_candidate_roi: Optional[SphereROI] = None

    @property
    def success(self) -> bool:
        return self.status in ("SUCCESS", "MANUAL") and self.roi is not None


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------

@dataclass
class ZoneMeasurement:
    """One of the three sampling zones on the sphere."""
    label: str          # "bright_side" | "center" | "dark_side"
    sample_label: str   # "Sample 1" | "Sample 2" | "Sample 3"
    ire: float
    log2_luminance: float
    rgb_mean: Tuple[float, float, float]
    pixel_count: int


@dataclass
class MeasurementResult:
    """
    Full photometric measurement for one clip, in the IPP2 display domain.

    The authoritative measurement is zone_center (Sample 2).
    All values are display-referred (IPP2 / BT.709 / BT.1886 / Medium / Medium).
    """
    clip_id: str
    render_path: str                    # path to the measurement TIFF
    render_sha256: str
    render_width: int
    render_height: int
    roi: SphereROI                      # the accepted sphere ROI

    # Zone measurements
    zone_bright: ZoneMeasurement        # Sample 1
    zone_center: ZoneMeasurement        # Sample 2 — hero
    zone_dark: ZoneMeasurement          # Sample 3

    # Derived
    hero_ire: float                     # zone_center.ire
    hero_log2: float                    # zone_center.log2_luminance
    hero_rgb: Tuple[float, float, float]
    hero_pixel_count: int

    # Chromaticity (from full interior mask, inner 78% of r)
    measured_rgb_mean: Tuple[float, float, float]
    measured_rgb_chromaticity: Tuple[float, float, float]  # normalized
    chromaticity_distance: float        # distance from achromatic (1/3, 1/3)

    # Quality
    measurement_valid: bool
    validity_reason: str = ""

    # Provenance
    roi_source: str = "auto_hough"      # "auto_hough" | "manual_operator"
    detection_result: Optional[SphereDetectionResult] = None

    # Scene-linear center log2 luminance (from the RWG/linear render), used to
    # solve EXPOSURE in scene-linear. None → fall back to display-space solve.
    hero_log2_lin: Optional[float] = None


# ---------------------------------------------------------------------------
# Solve
# ---------------------------------------------------------------------------

@dataclass
class CommitValues:
    """Final camera corrections to push via RCP2."""
    clip_id: str
    camera_label: str
    exposure_adjust: float              # stops
    kelvin: int                         # absolute Kelvin (not delta)
    tint: float                         # absolute tint
    derivation_method: str              # "shared_kelvin_per_camera_tint"
    # WB solve diagnostics
    wc_before: float = 0.0
    gm_before: float = 0.0
    wc_after: float = 0.0
    gm_after: float = 0.0
    wc_residual: float = 0.0           # absolute residual (diagnostic only)
    gm_residual: float = 0.0
    exposure_only: bool = False        # if True, WB untouched — only exposureAdjust commits/pushes

    def to_dict(self) -> Dict:
        d = {
            "exposureAdjust": round(self.exposure_adjust, 6),
            "derivation_method": self.derivation_method,
        }
        if not self.exposure_only:
            d["kelvin"] = self.kelvin
            d["tint"] = round(self.tint, 2)
        return d


@dataclass
class ExposureSpread:
    median_ire: float
    min_ire: float
    max_ire: float
    spread_stops: float
    outlier_clip_ids: List[str] = field(default_factory=list)


@dataclass
class WBSolveResult:
    """Result of the WB solve across the array."""
    status: str                         # "SOLVED" | "VERIFIED" | "SKIPPED"
    reason: str
    shared_kelvin: int
    wc_spread_before: float
    gm_spread_before: float
    wc_spread_after: float              # PREDICTED at solve time (model inverted against itself)
    gm_spread_after: float              # PREDICTED at solve time
    iteration_count: int
    camera_count: int
    excluded_count: int = 0
    # Closed-loop: MEASURED from corrected renders (None until verified)
    wc_spread_measured: Optional[float] = None
    gm_spread_measured: Optional[float] = None
    closed_loop: bool = False           # True when status was gated on measured spread


@dataclass
class CameraResult:
    """Complete result for one camera in one run."""
    clip_id: str
    camera_label: str
    source_path: str

    # Phases
    metadata: Optional[ClipMetadata]
    detection: Optional[SphereDetectionResult]
    measurement: Optional[MeasurementResult]
    commit: Optional[CommitValues]

    # Summary — there is no FAIL. Quality is expressed as match percentages.
    # status: "SOLVED" (commits ready) | "NEEDS_ASSIST" (operator places ROI)
    #         | "NO_DATA" (measurement invalid) | "ERROR" | "PENDING"
    status: str = "PENDING"
    exposure_closed_loop_status: str = ""   # "VERIFIED" | "ERROR" | ""
    wb_closed_loop_status: str = ""         # "VERIFIED" | "SOLVED" | ""
    # Match percentages (noise-floor anchored; None until closed-loop verified)
    exposure_match_pct: Optional[float] = None
    wb_match_pct: Optional[float] = None
    match_pct: Optional[float] = None       # min of verified axes
    corrected_ire: Optional[float] = None
    corrected_exposure_residual_stops: Optional[float] = None
    corrected_wc: Optional[float] = None   # measured WC from corrected render
    corrected_gm: Optional[float] = None   # measured GM from corrected render

    # Delivery domain (hybrid verification) — measured from the corrected frame
    # rendered through the project's delivery pipeline (transform + LUT).
    # Operator-facing only; the solve never uses these. All None unless a
    # delivery pipeline was explicitly selected for the run.
    delivery_corrected_render_path: Optional[str] = None
    delivery_corrected_ire: Optional[float] = None
    delivery_hero_log2: Optional[float] = None
    delivery_wc: Optional[float] = None
    delivery_gm: Optional[float] = None
    delivery_exposure_match_pct: Optional[float] = None
    delivery_wb_match_pct: Optional[float] = None
    delivery_match_pct: Optional[float] = None

    # Render paths
    original_render_path: Optional[str] = None
    corrected_render_path: Optional[str] = None
    linear_render_path: Optional[str] = None   # scene-linear render (exposure solve / QC re-measure)
    linear_render_error: Optional[str] = None  # populated iff the required scene-linear render failed

    # Failure info
    failed_stage: str = ""
    failure_reason: str = ""

    def is_usable(self) -> bool:
        return self.measurement is not None and self.measurement.measurement_valid


@dataclass
class RunResult:
    """Complete result of one R3DMatch analysis run."""
    run_id: str
    created_at: str
    input_path: str
    out_dir: str

    cameras: List[CameraResult]
    exposure_spread: Optional[ExposureSpread]
    wb_solve: Optional[WBSolveResult]

    # Strategy
    anchor_ire: float = 0.0
    anchor_log2: float = 0.0
    anchor_source: str = "median"
    gray_target_ire: float = 0.0   # gray_anchor: Log3G10 IRE target (e.g. 33.3)

    # Overall assessment — match percentage, never FAIL
    assessment_status: str = "PENDING"      # "SOLVED" | "PARTIAL" | "NO_SOLVE"
    array_match_pct: Optional[float] = None # mean of camera match %
    min_match_pct: Optional[float] = None   # weakest camera
    min_match_clip_id: str = ""
    solved_count: int = 0                   # cameras with commits
    scored_count: int = 0                   # cameras with verified match %
    needs_assist_count: int = 0
    no_data_count: int = 0
    operator_recommendation: str = ""

    # Exposure-only run: white balance was not solved; only exposureAdjust was
    # committed/pushed. Report omits WB metrics, charts, kelvin/tint.
    exposure_only: bool = False

    # Delivery domain (hybrid verification). Empty/None unless a delivery
    # pipeline was explicitly selected for the run. The solve/commits above are
    # always reference-domain and unaffected.
    delivery_pipeline_name: str = ""
    delivery_array_match_pct: Optional[float] = None
    delivery_min_match_pct: Optional[float] = None
    delivery_min_match_clip_id: str = ""
    delivery_profile: Optional[Dict] = None   # fresh PipelineProfile (this run)

    def to_dict(self) -> Dict:
        return asdict(self)
