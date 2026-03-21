from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch

from .cache import SequenceCache
from .metadata import BackendReport, DatasetManifest, FiducialSolveRecord, MaskRecord


@dataclass
class FiducialConfig:
    backend: str = "mock"
    fiducial_type: str = "aruco"
    marker_length_m: float = 0.2
    dictionary_name: str = "DICT_4X4_50"
    exclude_from_training: bool = True


class FiducialBackend(ABC):
    name = "unknown"

    @abstractmethod
    def solve(self, manifest: DatasetManifest, cache: SequenceCache, config: FiducialConfig) -> tuple[list[FiducialSolveRecord], list[MaskRecord]]:
        raise NotImplementedError


class MockFiducialBackend(FiducialBackend):
    name = "mock-fiducials"

    def solve(self, manifest: DatasetManifest, cache: SequenceCache, config: FiducialConfig) -> tuple[list[FiducialSolveRecord], list[MaskRecord]]:
        solves: list[FiducialSolveRecord] = []
        masks: list[MaskRecord] = []
        height = manifest.clip.height
        width = manifest.clip.width
        yy, xx = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
        center_x = width / 2.0
        center_y = height / 2.0
        radius = min(height, width) * 0.12
        mask = (((xx - center_x) ** 2 + (yy - center_y) ** 2) < radius**2).to(torch.float32)
        for frame in manifest.frames:
            mask_path = None
            if config.exclude_from_training:
                saved = cache.write_mask(frame.frame_index, mask, mask_type="fiducial")
                mask_path = str(saved)
                masks.append(
                    MaskRecord(
                        clip_id=frame.clip_id,
                        frame_index=frame.frame_index,
                        mask_type="fiducial",
                        mask_path=mask_path,
                        provenance=self.name,
                    )
                )
            solves.append(
                FiducialSolveRecord(
                    clip_id=frame.clip_id,
                    frame_index=frame.frame_index,
                    fiducial_detected=True,
                    fiducial_type=config.fiducial_type,
                    object_pose_camera=[
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 1.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                    reprojection_error=0.0,
                    solve_confidence=1.0,
                    object_scale=config.marker_length_m,
                    world_origin_definition="fiducial_plane_center",
                    axis_definition="x:right,y:down,z:forward",
                    mask_path=mask_path,
                )
            )
        return solves, masks


class OpenCvArucoBackend(FiducialBackend):
    name = "opencv-aruco"

    def solve(self, manifest: DatasetManifest, cache: SequenceCache, config: FiducialConfig) -> tuple[list[FiducialSolveRecord], list[MaskRecord]]:
        try:
            import cv2
        except Exception as exc:
            raise RuntimeError("OpenCV ArUco backend requires cv2 with aruco support") from exc

        solves: list[FiducialSolveRecord] = []
        masks: list[MaskRecord] = []
        dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, config.dictionary_name))
        detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
        for frame in manifest.frames:
            image = cache.read_frame(frame).mul(255.0).clamp(0.0, 255.0).byte().permute(1, 2, 0).numpy()
            corners, ids, _ = detector.detectMarkers(image)
            detected = ids is not None and len(ids) > 0
            mask_path = None
            if detected and config.exclude_from_training:
                mask = torch.zeros((manifest.clip.height, manifest.clip.width), dtype=torch.float32)
                for marker in corners:
                    pts = marker[0]
                    x0 = max(int(pts[:, 0].min()), 0)
                    x1 = min(int(pts[:, 0].max()) + 1, manifest.clip.width)
                    y0 = max(int(pts[:, 1].min()), 0)
                    y1 = min(int(pts[:, 1].max()) + 1, manifest.clip.height)
                    mask[y0:y1, x0:x1] = 1.0
                saved = cache.write_mask(frame.frame_index, mask, mask_type="fiducial")
                mask_path = str(saved)
                masks.append(
                    MaskRecord(
                        clip_id=frame.clip_id,
                        frame_index=frame.frame_index,
                        mask_type="fiducial",
                        mask_path=mask_path,
                        provenance=self.name,
                    )
                )
            solves.append(
                FiducialSolveRecord(
                    clip_id=frame.clip_id,
                    frame_index=frame.frame_index,
                    fiducial_detected=detected,
                    fiducial_type=config.fiducial_type,
                    object_pose_camera=[
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 1.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                    reprojection_error=0.0 if detected else None,
                    solve_confidence=1.0 if detected else 0.0,
                    object_scale=config.marker_length_m,
                    world_origin_definition="fiducial_plane_center",
                    axis_definition="aruco_dictionary_axes",
                    mask_path=mask_path,
                )
            )
        return solves, masks


def resolve_fiducial_backend(name: str) -> FiducialBackend:
    normalized = name.lower()
    if normalized == "mock":
        return MockFiducialBackend()
    if normalized in {"aruco", "opencv-aruco"}:
        return OpenCvArucoBackend()
    raise ValueError("fiducial backend must be one of: mock, aruco, opencv-aruco")


def solve_fiducials(dataset_dir: str, config: FiducialConfig) -> Dict[str, Any]:
    cache = SequenceCache(dataset_dir)
    manifest = cache.load_manifest()
    backend = resolve_fiducial_backend(config.backend)
    solves, masks = backend.solve(manifest, cache, config)
    mask_map = {record.frame_index: record.mask_path for record in masks}
    updated_frames = []
    for frame in manifest.frames:
        updated_frames.append(frame.model_copy(update={"mask_path": mask_map.get(frame.frame_index, frame.mask_path)}))
    updated = manifest.model_copy(
        update={
            "frames": updated_frames,
            "fiducial_solves": solves,
            "masks": masks,
            "backend_report": manifest.backend_report.model_copy(update={"fiducial_backend": backend.name}),
            "transforms_log": manifest.transforms_log + [f"fiducials: solved via {backend.name}"],
        }
    )
    cache.save_manifest(updated)
    return {
        "backend": backend.name,
        "detected_frames": sum(1 for solve in solves if solve.fiducial_detected),
        "total_frames": len(solves),
    }
