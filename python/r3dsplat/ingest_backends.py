from __future__ import annotations

import importlib
import math
import os
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np
import torch

from .metadata import CameraInfo, CameraRecord, ClipRecord, ColorInfo, ExposureInfo, FrameRecord, MotionInfo


DecodedFrame = Tuple[FrameRecord, torch.Tensor]
DecodedFrames = Iterable[DecodedFrame]
ProgressCallback = Callable[[Dict[str, Any]], None]


class IngestBackend(ABC):
    name = "unknown"

    @abstractmethod
    def inspect_clip(self, source_path: str) -> ClipRecord:
        raise NotImplementedError

    @abstractmethod
    def decode_clip(
        self,
        source_path: str,
        *,
        start_frame: int = 0,
        max_frames: Optional[int] = None,
        frame_step: int = 1,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> Tuple[ClipRecord, DecodedFrames]:
        raise NotImplementedError

    def diagnostics(self) -> Dict[str, Any]:
        return {"backend": self.name, "capability": "unknown"}


class MockIngestBackend(IngestBackend):
    name = "mock"

    def __init__(self, width: int = 64, height: int = 64, total_frames: int = 12, fps: float = 24.0):
        self.width = width
        self.height = height
        self.total_frames = total_frames
        self.fps = fps

    def inspect_clip(self, source_path: str) -> ClipRecord:
        return ClipRecord(
            clip_id=Path(source_path).stem or "synthetic",
            source_path=str(source_path),
            fps=self.fps,
            width=self.width,
            height=self.height,
            total_frames=self.total_frames,
            camera_model="pinhole",
            color_info=ColorInfo(color_space="linear", gamma_curve="linear", raw_bit_depth=16),
        )

    def decode_clip(
        self,
        source_path: str,
        *,
        start_frame: int = 0,
        max_frames: Optional[int] = None,
        frame_step: int = 1,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> Tuple[ClipRecord, DecodedFrames]:
        clip = self.inspect_clip(source_path)
        selected_indices = _select_frame_indices(
            total_frames=clip.total_frames,
            start_frame=start_frame,
            max_frames=max_frames,
            frame_step=frame_step,
        )
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, clip.height),
            torch.linspace(-1.0, 1.0, clip.width),
            indexing="ij",
        )
        start_time = time.perf_counter()

        def iterator() -> Iterator[DecodedFrame]:
            for completed, frame_index in enumerate(selected_indices, start=1):
                t = frame_index / clip.fps
                cx = math.sin(t * 2.0) * 0.35
                cy = math.cos(t * 1.5) * 0.25
                gaussian = torch.exp(-(((xx - cx) ** 2) + ((yy - cy) ** 2)) / 0.12)
                rgb = torch.stack(
                    [
                        gaussian,
                        torch.exp(-(((xx + cx * 0.7) ** 2) + ((yy - cy * 0.2) ** 2)) / 0.18),
                        0.5 * gaussian + 0.3,
                    ],
                    dim=0,
                ).clamp(0.0, 1.0)
                camera = CameraInfo(
                    fx=60.0,
                    fy=60.0,
                    cx=clip.width / 2.0,
                    cy=clip.height / 2.0,
                    view_matrix=[
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, -3.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                )
                record = FrameRecord(
                    clip_id=clip.clip_id,
                    frame_index=frame_index,
                    timestamp_seconds=t,
                    timecode=f"00:00:00:{frame_index:02d}",
                    decode_status="decoded",
                    camera_record_id=f"{clip.clip_id}:{frame_index:06d}:camera",
                    exposure_info=ExposureInfo(iso=800.0, shutter_seconds=1.0 / clip.fps),
                    white_balance=5600.0,
                    orientation="landscape",
                    color_info=clip.color_info,
                    motion_info=MotionInfo(
                        camera_translation=[0.0, 0.0, -3.0],
                        camera_rotation_quat_wxyz=[1.0, 0.0, 0.0, 0.0],
                    ),
                    camera=camera,
                    cache_path="",
                )
                if progress_callback is not None:
                    elapsed = max(time.perf_counter() - start_time, 1e-6)
                    progress_callback(
                        {
                            "frame_index": frame_index,
                            "completed": completed,
                            "total": len(selected_indices),
                            "percent": (completed / len(selected_indices)) * 100.0 if selected_indices else 100.0,
                            "elapsed_seconds": elapsed,
                            "decode_fps": completed / elapsed,
                        }
                    )
                yield record, rgb.to(dtype=torch.float32)

        return clip, iterator()

    def diagnostics(self) -> Dict[str, Any]:
        return {
            "backend": self.name,
            "capability": "mock",
            "width": self.width,
            "height": self.height,
            "total_frames": self.total_frames,
            "fps": self.fps,
        }


class RedSdkIngestBackend(IngestBackend):
    name = "red-sdk"

    def __init__(self, sdk_root: Optional[str] = None, allow_unavailable: bool = False):
        self._native_module = importlib.import_module("r3dsplat._r3d_native")
        self._config = self._native_module.RedSdkConfig(
            sdk_root=sdk_root or "",
            libraries_path=os.environ.get("RED_SDK_REDISTRIBUTABLE_DIR") or "",
        )
        self._decoder = self._native_module.RedDecoderBackend(self._config)
        self._allow_unavailable = allow_unavailable
        if not self._decoder.is_available() and not allow_unavailable:
            raise RuntimeError(self._decoder.sdk_diagnostics()["message"])

    def inspect_clip(self, source_path: str) -> ClipRecord:
        payload = self._decoder.inspect_clip(source_path)
        return ClipRecord.from_dict(payload)

    def decode_clip(
        self,
        source_path: str,
        *,
        start_frame: int = 0,
        max_frames: Optional[int] = None,
        frame_step: int = 1,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> Tuple[ClipRecord, DecodedFrames]:
        clip = self.inspect_clip(source_path)
        frames_payload = self._decoder.list_frames(source_path)
        selected_items = _select_frame_items(
            frames_payload["frames"],
            start_frame=start_frame,
            max_frames=max_frames,
            frame_step=frame_step,
        )
        start_time = time.perf_counter()

        def iterator() -> Iterator[DecodedFrame]:
            for completed, item in enumerate(selected_items, start=1):
                decoded = self._decoder.decode_frame(source_path, item["frame_index"])
                frame_tensor = self._decode_tensor_payload(decoded["frame"])
                frame_record = self._normalize_frame_record(clip, item, decoded.get("frame_metadata", {}))
                if progress_callback is not None:
                    elapsed = max(time.perf_counter() - start_time, 1e-6)
                    progress_callback(
                        {
                            "frame_index": int(item["frame_index"]),
                            "completed": completed,
                            "total": len(selected_items),
                            "percent": (completed / len(selected_items)) * 100.0 if selected_items else 100.0,
                            "elapsed_seconds": elapsed,
                            "decode_fps": completed / elapsed,
                        }
                    )
                yield frame_record, frame_tensor

        return clip, iterator()

    def diagnostics(self) -> Dict[str, Any]:
        diagnostics = dict(self._decoder.sdk_diagnostics())
        diagnostics["backend"] = self.name
        diagnostics["capability"] = self._capability()
        diagnostics["native_module"] = getattr(self._native_module, "__file__", "unknown")
        return diagnostics

    def _capability(self) -> str:
        module_file = str(getattr(self._native_module, "__file__", ""))
        if module_file.endswith(".py"):
            return "native-stub"
        if self._decoder.is_available():
            return "real-sdk"
        return "native-compiled-unavailable"

    @staticmethod
    def _decode_tensor_payload(frame_payload: Dict[str, Any]) -> torch.Tensor:
        data = frame_payload["data"]
        height = int(frame_payload["height"])
        width = int(frame_payload["width"])
        channels = int(frame_payload["channels"])
        bit_depth = int(frame_payload.get("bit_depth", 16))
        dtype = np.uint16 if bit_depth > 8 else np.uint8
        array = np.frombuffer(data, dtype=dtype).reshape(channels, height, width)
        if dtype == np.uint16:
            tensor = torch.from_numpy(array.astype(np.float32) / 65535.0)
        else:
            tensor = torch.from_numpy(array.astype(np.float32) / 255.0)
        return tensor

    @staticmethod
    def _normalize_frame_record(clip: ClipRecord, list_item: Dict[str, Any], frame_metadata: Dict[str, Any]) -> FrameRecord:
        return FrameRecord(
            clip_id=clip.clip_id,
            frame_index=int(list_item["frame_index"]),
            timestamp_seconds=float(list_item["timestamp_seconds"]),
            timecode=str(list_item.get("timecode", "")),
            cache_path="",
            decode_status="decoded",
            camera_record_id=f"{clip.clip_id}:{int(list_item['frame_index']):06d}:camera",
            exposure_info=ExposureInfo(
                iso=frame_metadata.get("iso"),
                shutter_seconds=frame_metadata.get("shutter_seconds"),
                aperture_t_stop=frame_metadata.get("aperture_t_stop"),
            ),
            white_balance=frame_metadata.get("white_balance"),
            orientation=frame_metadata.get("orientation", "landscape"),
            color_info=ColorInfo(
                color_space=frame_metadata.get("color_space", clip.color_info.color_space),
                gamma_curve=frame_metadata.get("gamma_curve", clip.color_info.gamma_curve),
                raw_bit_depth=clip.color_info.raw_bit_depth,
                additional=frame_metadata,
            ),
            motion_info=MotionInfo(additional={"native_frame_metadata": frame_metadata}),
            camera=CameraInfo(
                fx=float(frame_metadata.get("fx", clip.width)),
                fy=float(frame_metadata.get("fy", clip.height)),
                cx=float(frame_metadata.get("cx", clip.width / 2.0)),
                cy=float(frame_metadata.get("cy", clip.height / 2.0)),
                view_matrix=[
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            ),
        )


def resolve_ingest_backend(
    backend: str = "auto",
    sdk_root: Optional[str] = None,
) -> IngestBackend:
    normalized = backend.lower()
    if normalized == "mock":
        return MockIngestBackend()
    if normalized == "red-sdk":
        return RedSdkIngestBackend(sdk_root=sdk_root, allow_unavailable=False)
    if normalized != "auto":
        raise ValueError("backend must be one of: auto, mock, red-sdk")

    try:
        return RedSdkIngestBackend(sdk_root=sdk_root, allow_unavailable=False)
    except Exception:
        return MockIngestBackend()


def ingest_backend_summary(backend: str = "auto", sdk_root: Optional[str] = None) -> Dict[str, Any]:
    normalized = backend.lower()
    if normalized == "mock":
        return MockIngestBackend().diagnostics()
    if normalized == "red-sdk":
        try:
            return RedSdkIngestBackend(sdk_root=sdk_root, allow_unavailable=True).diagnostics()
        except Exception as exc:
            return {
                "backend": "red-sdk",
                "capability": "unavailable",
                "message": str(exc),
            }
    try:
        return RedSdkIngestBackend(sdk_root=sdk_root, allow_unavailable=True).diagnostics()
    except Exception:
        return MockIngestBackend().diagnostics()


def _select_frame_indices(
    *,
    total_frames: int,
    start_frame: int = 0,
    max_frames: Optional[int] = None,
    frame_step: int = 1,
) -> List[int]:
    if start_frame < 0:
        raise ValueError("start_frame must be >= 0")
    if frame_step <= 0:
        raise ValueError("frame_step must be >= 1")
    indices = list(range(start_frame, total_frames, frame_step))
    if max_frames is not None:
        if max_frames <= 0:
            raise ValueError("max_frames must be >= 1 when provided")
        indices = indices[:max_frames]
    if not indices:
        raise ValueError(
            f"frame selection produced no frames (start_frame={start_frame}, max_frames={max_frames}, frame_step={frame_step}, total_frames={total_frames})"
        )
    return indices


def _select_frame_items(
    items: List[Dict[str, Any]],
    *,
    start_frame: int = 0,
    max_frames: Optional[int] = None,
    frame_step: int = 1,
) -> List[Dict[str, Any]]:
    indices = set(
        _select_frame_indices(
            total_frames=len(items),
            start_frame=start_frame,
            max_frames=max_frames,
            frame_step=frame_step,
        )
    )
    return [item for item in items if int(item["frame_index"]) in indices]
