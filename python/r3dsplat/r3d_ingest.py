from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Optional

import torch
import torch.nn.functional as F

from .cache import SequenceCache
from .ingest_backends import resolve_ingest_backend
from .metadata import BackendReport, CameraRecord, ClipRecord, FrameRecord


def inspect_clip(
    source_path: str,
    backend: str = "auto",
    sdk_root: Optional[str] = None,
    decode_mode: Optional[str] = None,
) -> ClipRecord:
    return resolve_ingest_backend(backend=backend, sdk_root=sdk_root, decode_mode=decode_mode).inspect_clip(source_path)


def ingest_clip(
    source_path: str,
    dataset_dir: str,
    backend: str = "auto",
    sdk_root: Optional[str] = None,
    start_frame: int = 0,
    max_frames: Optional[int] = None,
    frame_step: int = 1,
    decode_mode: Optional[str] = None,
    resize_scale: Optional[float] = None,
    max_width: Optional[int] = None,
    max_height: Optional[int] = None,
    colmap_image_format: str = "tiff",
    matte_image_format: str = "tiff",
    colmap_max_width: Optional[int] = None,
    colmap_max_height: Optional[int] = None,
    matte_max_width: Optional[int] = None,
    matte_max_height: Optional[int] = None,
    colmap_compression: Optional[str] = None,
    matte_compression: Optional[str] = None,
    resolved_preset: Optional[str] = None,
    progress_callback: Optional[Callable[[dict[str, Any]], None]] = None,
) -> Path:
    dataset_path = Path(dataset_dir).expanduser().resolve()
    resize_settings = _resolve_resize_settings(
        resize_scale=resize_scale,
        max_width=max_width,
        max_height=max_height,
    )
    decoder = resolve_ingest_backend(backend=backend, sdk_root=sdk_root, decode_mode=decode_mode)
    ingest_started = time.perf_counter()
    clip, decoded = decoder.decode_clip(
        source_path,
        start_frame=start_frame,
        max_frames=max_frames,
        frame_step=frame_step,
        progress_callback=progress_callback,
    )
    cache = SequenceCache(dataset_path)
    selected_frame_indices = list(range(start_frame, clip.total_frames, frame_step))
    if max_frames is not None:
        selected_frame_indices = selected_frame_indices[:max_frames]
    decode_mode_name = decode_mode or "full-premium"
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
        "decode_mode: {decode_mode}".format(decode_mode=decode_mode_name),
        "resize_settings: {resize_settings}".format(resize_settings=resize_settings),
        "colmap_export: format={fmt}, path={path}, compression={compression}".format(
            fmt=colmap_image_format,
            path=cache.colmap_images_dir,
            compression=colmap_compression,
        ),
        "matte_export: format={fmt}, path={path}, compression={compression}".format(
            fmt=matte_image_format,
            path=cache.matte_images_dir,
            compression=matte_compression,
        ),
    ]

    frame_records: list[FrameRecord] = []
    camera_records: list[CameraRecord] = []
    for frame_record, tensor in decoded:
        decoded_height = int(tensor.shape[-2])
        decoded_width = int(tensor.shape[-1])
        resized_tensor, cached_width, cached_height = _resize_frame_tensor(tensor, resize_settings)
        scale_x = cached_width / float(decoded_width)
        scale_y = cached_height / float(decoded_height)
        updated_camera = frame_record.camera.model_copy(
            update={
                "fx": frame_record.camera.fx * scale_x,
                "fy": frame_record.camera.fy * scale_y,
                "cx": frame_record.camera.cx * scale_x,
                "cy": frame_record.camera.cy * scale_y,
            }
        )
        cache_path = cache.write_frame(frame_record.frame_index, resized_tensor)
        colmap_image_path = cache.write_exported_image(
            frame_record.frame_index,
            resized_tensor,
            asset_kind="colmap",
            image_format=colmap_image_format,
            compression=colmap_compression,
        )
        matte_image_path = cache.write_exported_image(
            frame_record.frame_index,
            resized_tensor,
            asset_kind="matte",
            image_format=matte_image_format,
            compression=matte_compression,
        )
        updated_frame = frame_record.model_copy(
            update={
                "cache_path": str(cache_path),
                "camera": updated_camera,
                "decoded_width": decoded_width,
                "decoded_height": decoded_height,
                "cached_width": cached_width,
                "cached_height": cached_height,
                "colmap_image_path": str(colmap_image_path),
                "colmap_image_width": cached_width,
                "colmap_image_height": cached_height,
                "colmap_image_format": colmap_image_format,
                "matte_image_path": str(matte_image_path),
                "matte_image_width": cached_width,
                "matte_image_height": cached_height,
                "matte_image_format": matte_image_format,
            }
        )
        frame_records.append(updated_frame)
        camera_records.append(
            CameraRecord(
                camera_record_id=updated_frame.camera_record_id or f"{updated_frame.frame_record_id}:camera",
                frame_record_id=updated_frame.frame_record_id,
                intrinsics={
                    "fx": updated_camera.fx,
                    "fy": updated_camera.fy,
                    "cx": updated_camera.cx,
                    "cy": updated_camera.cy,
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
        backend_report=BackendReport(
            ingest_backend=decoder.name,
            ingest_settings={
                "original_width": clip.width,
                "original_height": clip.height,
                "decoded_width": frame_records[0].decoded_width if frame_records else None,
                "decoded_height": frame_records[0].decoded_height if frame_records else None,
                "cached_width": frame_records[0].cached_width if frame_records else None,
                "cached_height": frame_records[0].cached_height if frame_records else None,
                "frame_selection": {
                    "start_frame": start_frame,
                    "max_frames": max_frames,
                    "frame_step": frame_step,
                    "selected_count": len(frame_records),
                },
                "decode_mode": decode_mode_name,
                "resize_settings": resize_settings,
                "resolved_preset": resolved_preset,
                "colmap_export": {
                    "path": str(cache.colmap_images_dir),
                    "format": colmap_image_format,
                    "max_width": colmap_max_width,
                    "max_height": colmap_max_height,
                    "compression": colmap_compression,
                    "width": frame_records[0].cached_width if frame_records else None,
                    "height": frame_records[0].cached_height if frame_records else None,
                    "alignment_run": False,
                },
                "matte_export": {
                    "path": str(cache.matte_images_dir),
                    "format": matte_image_format,
                    "max_width": matte_max_width,
                    "max_height": matte_max_height,
                    "compression": matte_compression,
                    "width": frame_records[0].cached_width if frame_records else None,
                    "height": frame_records[0].cached_height if frame_records else None,
                },
            },
        ),
        transforms_log=transforms_log
        + [
            "ingest_elapsed_seconds: {elapsed:.3f}".format(elapsed=time.perf_counter() - ingest_started),
            "dataset_dir: {dataset_dir}".format(dataset_dir=dataset_path),
        ],
    )
    _validate_derived_assets(
        cache=cache,
        frame_records=frame_records,
        expected_count=len(selected_frame_indices),
        colmap_image_format=colmap_image_format,
        matte_image_format=matte_image_format,
    )
    return dataset_path


def _resolve_resize_settings(
    *,
    resize_scale: Optional[float],
    max_width: Optional[int],
    max_height: Optional[int],
) -> dict[str, Any]:
    if resize_scale is not None and (max_width is not None or max_height is not None):
        raise ValueError("resize_scale cannot be combined with max_width or max_height")
    if resize_scale is not None:
        if resize_scale <= 0.0:
            raise ValueError("resize_scale must be > 0")
        return {"mode": "scale", "resize_scale": resize_scale}
    if max_width is not None or max_height is not None:
        if max_width is not None and max_width <= 0:
            raise ValueError("max_width must be > 0")
        if max_height is not None and max_height <= 0:
            raise ValueError("max_height must be > 0")
        return {"mode": "bounds", "max_width": max_width, "max_height": max_height}
    return {"mode": "none"}


def _resize_frame_tensor(
    frame: torch.Tensor,
    resize_settings: dict[str, Any],
) -> tuple[torch.Tensor, int, int]:
    height = int(frame.shape[-2])
    width = int(frame.shape[-1])
    target_width, target_height = _target_size(width, height, resize_settings)
    if target_width == width and target_height == height:
        return frame, width, height
    resized = F.interpolate(
        frame.unsqueeze(0),
        size=(target_height, target_width),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)
    return resized, target_width, target_height


def _target_size(width: int, height: int, resize_settings: dict[str, Any]) -> tuple[int, int]:
    mode = resize_settings.get("mode", "none")
    if mode == "none":
        return width, height
    if mode == "scale":
        scale = float(resize_settings["resize_scale"])
        return max(1, int(round(width * scale))), max(1, int(round(height * scale)))
    if mode == "bounds":
        width_scale = 1.0
        height_scale = 1.0
        if resize_settings.get("max_width") is not None:
            width_scale = min(1.0, float(resize_settings["max_width"]) / float(width))
        if resize_settings.get("max_height") is not None:
            height_scale = min(1.0, float(resize_settings["max_height"]) / float(height))
        scale = min(width_scale, height_scale)
        return max(1, int(round(width * scale))), max(1, int(round(height * scale)))
    raise ValueError(f"unsupported resize mode: {mode}")


def _validate_derived_assets(
    *,
    cache: SequenceCache,
    frame_records: list[FrameRecord],
    expected_count: int,
    colmap_image_format: str,
    matte_image_format: str,
) -> None:
    if not cache.colmap_images_dir.exists():
        raise RuntimeError(f"Derived COLMAP image directory missing after ingest: {cache.colmap_images_dir}")
    if not cache.matte_images_dir.exists():
        raise RuntimeError(f"Derived matting image directory missing after ingest: {cache.matte_images_dir}")
    colmap_suffix = f"*.{_normalized_image_suffix(colmap_image_format)}"
    matte_suffix = f"*.{_normalized_image_suffix(matte_image_format)}"
    colmap_files = sorted(cache.colmap_images_dir.glob(colmap_suffix))
    matte_files = sorted(cache.matte_images_dir.glob(matte_suffix))
    if len(colmap_files) != expected_count:
        raise RuntimeError(
            f"Derived COLMAP image count mismatch: expected={expected_count} actual={len(colmap_files)} dir={cache.colmap_images_dir}"
        )
    if len(matte_files) != expected_count:
        raise RuntimeError(
            f"Derived matting image count mismatch: expected={expected_count} actual={len(matte_files)} dir={cache.matte_images_dir}"
        )
    expected_names = [f"{record.frame_index:06d}.{_normalized_image_suffix(colmap_image_format)}" for record in frame_records]
    if [path.name for path in colmap_files] != expected_names:
        raise RuntimeError(f"Derived COLMAP filenames are not deterministic/aligned in {cache.colmap_images_dir}")
    expected_matte_names = [f"{record.frame_index:06d}.{_normalized_image_suffix(matte_image_format)}" for record in frame_records]
    if [path.name for path in matte_files] != expected_matte_names:
        raise RuntimeError(f"Derived matting filenames are not deterministic/aligned in {cache.matte_images_dir}")


def _normalized_image_suffix(image_format: str) -> str:
    normalized = image_format.lower()
    if normalized in {"tif", "tiff"}:
        return "tiff"
    return normalized
