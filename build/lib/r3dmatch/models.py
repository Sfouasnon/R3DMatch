from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ClipMetadata:
    clip_id: str
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
class ClipResult:
    clip_id: str
    source_path: str
    backend: str
    camera_group: str
    clip_statistic_log2: float
    camera_group_statistic_log2: float
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
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
