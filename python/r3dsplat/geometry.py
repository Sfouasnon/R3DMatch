from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch

from .cache import SequenceCache
from .metadata import CameraRecord


def invert_transform(matrix: List[List[float]]) -> List[List[float]]:
    tensor = torch.tensor(matrix, dtype=torch.float32)
    return torch.linalg.inv(tensor).tolist()


def compose_transform(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
    ta = torch.tensor(a, dtype=torch.float32)
    tb = torch.tensor(b, dtype=torch.float32)
    return (ta @ tb).tolist()


@dataclass
class WorldAlignmentResult:
    aligned_cameras: List[CameraRecord]
    world_from_colmap: List[List[float]]
    aligned_count: int


def align_cameras_to_fiducials(camera_records: List[CameraRecord], fiducial_world_from_camera: Dict[str, List[List[float]]]) -> WorldAlignmentResult:
    aligned: List[CameraRecord] = []
    world_from_colmap = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    for record in camera_records:
        if record.frame_record_id in fiducial_world_from_camera:
            c2w = fiducial_world_from_camera[record.frame_record_id]
            aligned.append(
                record.model_copy(
                    update={
                        "extrinsics_camera_to_world": c2w,
                        "extrinsics_world_to_camera": invert_transform(c2w),
                        "alignment_status": "aligned_to_fiducial_world",
                        "pose_provenance": record.pose_provenance + ["world-aligned"],
                    }
                )
            )
        else:
            aligned.append(record)
    return WorldAlignmentResult(aligned_cameras=aligned, world_from_colmap=world_from_colmap, aligned_count=sum(1 for item in aligned if item.alignment_status == "aligned_to_fiducial_world"))


def align_world(dataset_dir: str) -> Dict[str, object]:
    cache = SequenceCache(dataset_dir)
    manifest = cache.load_manifest()
    fiducials = {f"{manifest.clip.clip_id}:{solve.frame_index:06d}": invert_transform(solve.object_pose_camera) for solve in manifest.fiducial_solves if solve.fiducial_detected}
    result = align_cameras_to_fiducials(manifest.camera_records, fiducials)
    updated = manifest.model_copy(
        update={
            "camera_records": result.aligned_cameras,
            "transforms_log": manifest.transforms_log + ["geometry: aligned world frame from fiducial solves"],
        }
    )
    cache.save_manifest(updated)
    return {
        "aligned_cameras": result.aligned_count,
        "world_from_colmap": result.world_from_colmap,
    }


def summarize_camera_trajectory(camera_records: List[CameraRecord]) -> Dict[str, object]:
    if not camera_records:
        return {
            "camera_count": 0,
            "status": "empty",
            "translation_min": [0.0, 0.0, 0.0],
            "translation_max": [0.0, 0.0, 0.0],
            "sources": {},
        }
    translations = torch.tensor(
        [[record.extrinsics_camera_to_world[row][3] for row in range(3)] for record in camera_records],
        dtype=torch.float32,
    )
    sources: Dict[str, int] = {}
    for record in camera_records:
        sources[record.source_of_pose] = sources.get(record.source_of_pose, 0) + 1
    return {
        "camera_count": len(camera_records),
        "status": "ok",
        "translation_min": translations.min(dim=0).values.tolist(),
        "translation_max": translations.max(dim=0).values.tolist(),
        "sources": sources,
        "alignment_statuses": sorted({record.alignment_status for record in camera_records}),
    }


def debug_poses(dataset_dir: str) -> Dict[str, object]:
    cache = SequenceCache(dataset_dir)
    manifest = cache.load_manifest()
    summary = summarize_camera_trajectory(manifest.camera_records)
    summary["clip_id"] = manifest.clip.clip_id
    summary["frame_count"] = len(manifest.frames)
    return summary
