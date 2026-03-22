from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Optional

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
    start_frame: int = 0,
    max_frames: Optional[int] = None,
    frame_step: int = 1,
    progress_callback: Optional[Callable[[dict[str, Any]], None]] = None,
) -> Path:
    decoder = resolve_ingest_backend(backend=backend, sdk_root=sdk_root)
    ingest_started = time.perf_counter()
    clip, decoded = decoder.decode_clip(
        source_path,
        start_frame=start_frame,
        max_frames=max_frames,
        frame_step=frame_step,
        progress_callback=progress_callback,
    )
    cache = SequenceCache(dataset_dir)
    selected_frame_indices = list(range(start_frame, clip.total_frames, frame_step))
    if max_frames is not None:
        selected_frame_indices = selected_frame_indices[:max_frames]
    transforms_log = [
        "decode_backend: {backend}".format(backend=decoder.name),
        "cache: frame tensors serialized as torch .pt files in CHW float32 layout",
        "dataset: temporal ordering preserved exactly as decoded",
        "frame_selection: start_frame={start_frame}, max_frames={max_frames}, frame_step={frame_step}, selected_count={selected_count}".format(
            start_frame=start_frame,
            max_frames="all" if max_frames is None else max_frames,
            frame_step=frame_step,
            selected_count=len(selected_frame_indices),
        ),
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
        transforms_log=transforms_log
        + [
            "ingest_elapsed_seconds: {elapsed:.3f}".format(elapsed=time.perf_counter() - ingest_started),
            "dataset_dir: {dataset_dir}".format(dataset_dir=dataset_dir),
        ],
    )
    return Path(dataset_dir)
