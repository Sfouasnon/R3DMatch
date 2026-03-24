from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Union


@dataclass
class ClipMetadata:
    clip_id: str
    group_key: str
    original_filename: str
    source_path: str
    fps: float
    width: int
    height: int
    total_frames: int
    reel_id: Optional[str] = None
    source_identifier: Optional[str] = None
    color_space: str = "REDWideGamutRGB"
    gamma_curve: str = "Log3G10"
    iso: Optional[float] = None
    shutter_seconds: Optional[float] = None
    aperture_t_stop: Optional[float] = None
    active_lut_path: Optional[str] = None
    extra_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SamplePlan:
    strategy: str
    sample_count: int
    start_frame: int
    frame_step: int
    max_frames: int


@dataclass
class FrameStat:
    frame_index: int
    timestamp_seconds: float
    log_luminance_median: float
    clipped_fraction: float
    valid_fraction: float
    accepted: bool
    reason: Optional[str] = None


@dataclass
class MonitoringContext:
    mode: str
    ipp2_color_space: Optional[str]
    ipp2_gamma_curve: Optional[str]
    active_lut_path: Optional[str]
    lut_override_path: Optional[str]
    resolved_lut_path: Optional[str]


@dataclass
class SphereROI:
    cx: float
    cy: float
    r: float


@dataclass
class GrayCardROI:
    x: float
    y: float
    width: float
    height: float


@dataclass
class CenterCrop:
    width_ratio: float
    height_ratio: float


@dataclass
class SamplingRegion:
    sampling_mode: str
    roi: Optional[Dict[str, float]] = None
    center_crop: Optional[CenterCrop] = None
    detection_confidence: Optional[float] = None
    fallback_used: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExposureCalibrationEntry:
    clip_id: str
    group_key: str
    source_path: str
    sampling_mode: str
    sampling_region: SamplingRegion
    measured_log2_luminance: float
    camera_baseline_stops: float
    confidence: float
    calibration_source: str
    exposure_stddev_logY: float = 0.0
    frame_index: int = 0


@dataclass
class ExposureCalibration:
    calibration_type: str
    object_type: str
    cameras: List[ExposureCalibrationEntry]
    target_log2_luminance: Optional[float] = None
    reference_camera: Optional[str] = None
    roi_file: Optional[str] = None
    calibration_mode: str = "exposure"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ColorCalibrationEntry:
    clip_id: str
    group_key: str
    source_path: str
    sampling_mode: str
    sampling_region: SamplingRegion
    measured_channel_medians: Dict[str, float]
    rgb_neutral_gains: Dict[str, float]
    confidence: float
    calibration_source: str
    chromaticity_variance: float = 0.0
    frame_index: int = 0


@dataclass
class ColorCalibration:
    calibration_type: str
    object_type: str
    cameras: List[ColorCalibrationEntry]
    roi_file: Optional[str] = None
    calibration_mode: str = "color"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GrayCardDetectionEntry:
    clip_id: str
    group_key: str
    roi: GrayCardROI
    confidence: float
    source_path: str
    method: str
    frame_index: int = 0


@dataclass
class GrayCardDetection:
    object_type: str
    detection_mode: str
    cameras: List[GrayCardDetectionEntry]
    roi_file: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SphereCalibrationCameraEntry:
    clip_id: str
    group_key: str
    roi: SphereROI
    measured_sphere_log2: float
    camera_baseline_stops: float
    confidence: float
    source_path: str
    frame_index: int = 0


@dataclass
class SphereCalibration:
    target_log2_luminance: float
    object_type: str
    cameras: List[SphereCalibrationCameraEntry]
    calibration_mode: str = "sphere"
    roi_file: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GrayCardCalibrationCameraEntry:
    clip_id: str
    group_key: str
    roi: GrayCardROI
    measured_card_log2: float
    camera_baseline_stops: float
    confidence: float
    source_path: str
    frame_index: int = 0


@dataclass
class GrayCardCalibration:
    target_log2_luminance: float
    calibration_mode: str
    object_type: str
    cameras: List[GrayCardCalibrationCameraEntry]
    roi_file: Optional[str] = None
    reference_camera: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


CalibrationLike = Union[SphereCalibration, GrayCardCalibration, ExposureCalibration]
IndependentCalibrationLike = Union[ExposureCalibration, ColorCalibration]


@dataclass
class ClipResult:
    clip_id: str
    group_key: str
    source_path: str
    backend: str
    clip_statistic_log2: float
    group_key_statistic_log2: float
    global_reference_log2: float
    raw_offset_stops: float
    camera_baseline_stops: float
    clip_trim_stops: float
    final_offset_stops: float
    confidence: float
    sample_plan: SamplePlan
    monitoring: MonitoringContext
    clip_metadata: ClipMetadata
    frame_stats: List[FrameStat]
    calibration_provenance: Optional[Dict[str, Any]] = None
    exposure_calibration_provenance: Optional[Dict[str, Any]] = None
    color_calibration_provenance: Optional[Dict[str, Any]] = None
    pending_color_gains: Optional[Dict[str, float]] = None
    exposure_calibration_loaded: bool = False
    exposure_baseline_applied_stops: Optional[float] = None
    color_calibration_loaded: bool = False
    color_gains_state: Optional[str] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
