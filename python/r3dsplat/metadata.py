from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


class JsonMixin:
    def model_dump(self, mode: str = "python") -> Dict[str, Any]:
        return asdict(self)

    def model_dump_json(self, indent: Optional[int] = None) -> str:
        return json.dumps(self.model_dump(), indent=indent)

    def model_copy(self, update: Optional[Dict[str, Any]] = None):
        return replace(self, **(update or {}))


@dataclass
class ColorInfo(JsonMixin):
    color_space: str = "unknown"
    gamma_curve: str = "unknown"
    iso: Optional[float] = None
    raw_bit_depth: Optional[int] = None
    additional: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ColorInfo":
        return cls(**payload)


@dataclass
class LensInfo(JsonMixin):
    manufacturer: Optional[str] = None
    model: Optional[str] = None
    focal_length_mm: Optional[float] = None
    aperture_t_stop: Optional[float] = None
    focus_distance_m: Optional[float] = None
    additional: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "LensInfo":
        return cls(**payload)


@dataclass
class ExposureInfo(JsonMixin):
    iso: Optional[float] = None
    shutter_seconds: Optional[float] = None
    aperture_t_stop: Optional[float] = None
    nd_filter: Optional[str] = None
    additional: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ExposureInfo":
        return cls(**payload)


@dataclass
class MotionInfo(JsonMixin):
    velocity_hint: Optional[List[float]] = None
    acceleration_hint: Optional[List[float]] = None
    camera_translation: Optional[List[float]] = None
    camera_rotation_quat_wxyz: Optional[List[float]] = None
    additional: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "MotionInfo":
        return cls(**payload)


@dataclass
class CameraInfo(JsonMixin):
    fx: float
    fy: float
    cx: float
    cy: float
    view_matrix: List[List[float]]
    near_plane: float = 0.01
    far_plane: float = 1000.0
    background: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "CameraInfo":
        return cls(**payload)


@dataclass
class ClipRecord(JsonMixin):
    clip_id: str
    source_path: str
    fps: float
    width: int
    height: int
    total_frames: int
    camera_model: str = "pinhole"
    color_info: ColorInfo = field(default_factory=ColorInfo)
    lens_info: LensInfo = field(default_factory=LensInfo)
    reel_id: Optional[str] = None
    source_identifier: Optional[str] = None
    extra_metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ClipRecord":
        return cls(
            clip_id=payload["clip_id"],
            source_path=payload["source_path"],
            fps=payload["fps"],
            width=payload["width"],
            height=payload["height"],
            total_frames=payload["total_frames"],
            camera_model=payload.get("camera_model", "pinhole"),
            color_info=ColorInfo.from_dict(payload.get("color_info", {})),
            lens_info=LensInfo.from_dict(payload.get("lens_info", {})),
            reel_id=payload.get("reel_id"),
            source_identifier=payload.get("source_identifier"),
            extra_metadata=payload.get("extra_metadata", {}),
        )


@dataclass
class CameraRecord(JsonMixin):
    camera_record_id: str
    frame_record_id: str
    intrinsics: Dict[str, float]
    distortion_model: str = "none"
    distortion_params: Dict[str, float] = field(default_factory=dict)
    extrinsics_world_to_camera: List[List[float]] = field(
        default_factory=lambda: [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    extrinsics_camera_to_world: List[List[float]] = field(
        default_factory=lambda: [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    source_of_pose: str = "unsolved"
    pose_confidence: float = 0.0
    alignment_status: str = "unaligned"
    pose_provenance: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "CameraRecord":
        return cls(**payload)


@dataclass
class FiducialSolveRecord(JsonMixin):
    clip_id: str
    frame_index: int
    fiducial_detected: bool
    fiducial_type: str
    object_pose_camera: List[List[float]] = field(
        default_factory=lambda: [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    reprojection_error: Optional[float] = None
    solve_confidence: float = 0.0
    object_scale: float = 1.0
    world_origin_definition: str = "fiducial_center"
    axis_definition: str = "fiducial_axes"
    mask_path: Optional[str] = None

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "FiducialSolveRecord":
        return cls(**payload)


@dataclass
class ColmapSolveRecord(JsonMixin):
    clip_id: str
    colmap_project_dir: str
    database_path: str
    sparse_model_path: str
    camera_model_used: str
    intrinsics_mode: str
    matching_mode: str
    mapper_mode: str
    registered_images: int = 0
    solve_status: str = "unknown"
    notes: str = ""

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ColmapSolveRecord":
        return cls(**payload)


@dataclass
class MaskRecord(JsonMixin):
    clip_id: str
    frame_index: int
    mask_type: str
    mask_path: str
    provenance: str

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "MaskRecord":
        return cls(**payload)


@dataclass
class BackendReport(JsonMixin):
    ingest_backend: str = "unknown"
    fiducial_backend: str = "unset"
    colmap_backend: str = "unset"
    renderer_backend: str = "unset"

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "BackendReport":
        return cls(**payload)


@dataclass
class FrameRecord(JsonMixin):
    clip_id: str
    frame_index: int
    timestamp_seconds: float
    timecode: str
    cache_path: str = ""
    decode_status: str = "decoded"
    camera_record_id: Optional[str] = None
    exposure_info: ExposureInfo = field(default_factory=ExposureInfo)
    white_balance: Optional[float] = None
    orientation: str = "landscape"
    color_info: ColorInfo = field(default_factory=ColorInfo)
    motion_info: MotionInfo = field(default_factory=MotionInfo)
    camera: CameraInfo = field(
        default_factory=lambda: CameraInfo(
            1.0,
            1.0,
            0.0,
            0.0,
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        )
    )
    mask_path: Optional[str] = None

    @property
    def frame_record_id(self) -> str:
        return f"{self.clip_id}:{self.frame_index:06d}"

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "FrameRecord":
        return cls(
            clip_id=payload["clip_id"],
            frame_index=payload["frame_index"],
            timestamp_seconds=payload["timestamp_seconds"],
            timecode=payload["timecode"],
            cache_path=payload.get("cache_path", ""),
            decode_status=payload.get("decode_status", "decoded"),
            camera_record_id=payload.get("camera_record_id"),
            exposure_info=ExposureInfo.from_dict(payload.get("exposure_info", {})),
            white_balance=payload.get("white_balance"),
            orientation=payload.get("orientation", "landscape"),
            color_info=ColorInfo.from_dict(payload.get("color_info", {})),
            motion_info=MotionInfo.from_dict(payload.get("motion_info", {})),
            camera=CameraInfo.from_dict(payload["camera"]),
            mask_path=payload.get("mask_path"),
        )

    def payload(self) -> Dict[str, Any]:
        return self.model_dump()


@dataclass
class DatasetManifest(JsonMixin):
    manifest_version: int
    clip: ClipRecord
    frames: List[FrameRecord]
    camera_records: List[CameraRecord] = field(default_factory=list)
    fiducial_solves: List[FiducialSolveRecord] = field(default_factory=list)
    colmap_solves: List[ColmapSolveRecord] = field(default_factory=list)
    masks: List[MaskRecord] = field(default_factory=list)
    backend_report: BackendReport = field(default_factory=BackendReport)
    transforms_log: List[str] = field(default_factory=list)

    def camera_by_frame_id(self) -> Dict[str, CameraRecord]:
        return {record.frame_record_id: record for record in self.camera_records}

    def fiducials_by_frame_index(self) -> Dict[int, FiducialSolveRecord]:
        return {record.frame_index: record for record in self.fiducial_solves}

    def masks_by_frame_index(self) -> Dict[int, MaskRecord]:
        return {record.frame_index: record for record in self.masks}

    def save(self, dataset_dir: Union[str, Path]) -> None:
        Path(dataset_dir).mkdir(parents=True, exist_ok=True)
        (Path(dataset_dir) / "manifest.json").write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def load(cls, dataset_dir: Union[str, Path]) -> "DatasetManifest":
        payload = json.loads((Path(dataset_dir) / "manifest.json").read_text(encoding="utf-8"))
        return cls(
            manifest_version=payload.get("manifest_version", 1),
            clip=ClipRecord.from_dict(payload["clip"]),
            frames=[FrameRecord.from_dict(frame) for frame in payload["frames"]],
            camera_records=[CameraRecord.from_dict(item) for item in payload.get("camera_records", [])],
            fiducial_solves=[FiducialSolveRecord.from_dict(item) for item in payload.get("fiducial_solves", [])],
            colmap_solves=[ColmapSolveRecord.from_dict(item) for item in payload.get("colmap_solves", [])],
            masks=[MaskRecord.from_dict(item) for item in payload.get("masks", [])],
            backend_report=BackendReport.from_dict(payload.get("backend_report", {})),
            transforms_log=payload.get("transforms_log", []),
        )
