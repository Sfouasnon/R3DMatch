from __future__ import annotations

from pathlib import Path
from typing import Optional

from .cache import SequenceCache
from .ingest_backends import resolve_ingest_backend
from .metadata import BackendReport, CameraRecord, ClipRecord, FrameRecord


def inspect_clip(source_path: str, backend: str = "auto", sdk_root: Optional[str] = None) -> ClipRecord:
    return resolve_ingest_backend(backend=backend, sdk_root=sdk_root).inspect_clip(source_path)


def ingest_clip(
    source_path: str,
    dataset_dir: str,
    backend: str = "auto",
    sdk_root: Optional[str] = None,
) -> Path:
    decoder = resolve_ingest_backend(backend=backend, sdk_root=sdk_root)
    clip, decoded = decoder.decode_clip(source_path)
    cache = SequenceCache(dataset_dir)
    transforms_log = [
        "decode_backend: {backend}".format(backend=decoder.name),
        "cache: frame tensors serialized as torch .pt files in CHW float32 layout",
        "dataset: temporal ordering preserved exactly as decoded",
    ]

    frame_records: list[FrameRecord] = []
    camera_records: list[CameraRecord] = []
    for frame_record, tensor in decoded:
        cache_path = cache.write_frame(frame_record.frame_index, tensor)
        updated_frame = frame_record.model_copy(update={"cache_path": str(cache_path)})
        frame_records.append(updated_frame)
        camera_records.append(
            CameraRecord(
                camera_record_id=updated_frame.camera_record_id or f"{updated_frame.frame_record_id}:camera",
                frame_record_id=updated_frame.frame_record_id,
                intrinsics={
                    "fx": updated_frame.camera.fx,
                    "fy": updated_frame.camera.fy,
                    "cx": updated_frame.camera.cx,
                    "cy": updated_frame.camera.cy,
                },
                extrinsics_world_to_camera=updated_frame.camera.view_matrix,
                extrinsics_camera_to_world=updated_frame.camera.view_matrix,
                source_of_pose="ingest_prior",
                pose_confidence=0.1,
                alignment_status="unaligned",
                pose_provenance=[decoder.name],
            )
        )

    cache.build_manifest(
        clip=clip,
        frames=frame_records,
        camera_records=camera_records,
        backend_report=BackendReport(ingest_backend=decoder.name),
        transforms_log=transforms_log,
    )
    return Path(dataset_dir)
