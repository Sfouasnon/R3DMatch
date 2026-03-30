from __future__ import annotations

import hashlib
import importlib
import math
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional, Tuple

import numpy as np

from .identity import clip_id_from_path, group_key_from_clip_id, original_filename_from_path
from .models import ClipMetadata


DecodedFrame = Tuple[int, float, np.ndarray]
# Internal frame layout contract:
# - backends may produce HWC or CHW RGB float32 arrays
# - the Python analysis pipeline always consumes CHW float32 arrays


class R3DBackend(ABC):
    name = "unknown"

    @abstractmethod
    def inspect_clip(self, source_path: str) -> ClipMetadata:
        raise NotImplementedError

    @abstractmethod
    def decode_frames(
        self,
        source_path: str,
        *,
        start_frame: int,
        max_frames: int,
        frame_step: int,
        half_res: bool = False,
    ) -> Iterable[DecodedFrame]:
        raise NotImplementedError


class MockR3DBackend(R3DBackend):
    name = "mock"

    def inspect_clip(self, source_path: str) -> ClipMetadata:
        clip_id = clip_id_from_path(source_path)
        digest = hashlib.sha1(clip_id.encode("utf-8")).digest()
        iso = 400.0 + float(digest[0] % 5) * 160.0
        return ClipMetadata(
            clip_id=clip_id,
            group_key=group_key_from_clip_id(clip_id),
            original_filename=original_filename_from_path(source_path),
            source_path=str(Path(source_path).expanduser().resolve()),
            fps=24.0,
            width=96,
            height=54,
            total_frames=24,
            reel_id=clip_id.split("_")[0] if "_" in clip_id else None,
            source_identifier=clip_id.split("_")[1] if clip_id.count("_") >= 1 else clip_id,
            iso=iso,
            shutter_seconds=1.0 / 48.0,
            aperture_t_stop=2.8,
            active_lut_path=None,
            extra_metadata={"mock_digest": digest.hex()},
        )

    def decode_frames(
        self,
        source_path: str,
        *,
        start_frame: int,
        max_frames: int,
        frame_step: int,
        half_res: bool = False,
    ) -> Iterable[DecodedFrame]:
        clip = self.inspect_clip(source_path)
        selected = list(range(start_frame, clip.total_frames, frame_step))[:max_frames]
        clip_seed = int(hashlib.sha1(clip.clip_id.encode("utf-8")).hexdigest()[:8], 16)
        yy, xx = np.meshgrid(
            np.linspace(-1.0, 1.0, clip.height, dtype=np.float32),
            np.linspace(-1.0, 1.0, clip.width, dtype=np.float32),
            indexing="ij",
        )

        def iterator() -> Iterator[DecodedFrame]:
            base_exposure = 0.7 + (clip_seed % 19) / 40.0
            for frame_index in selected:
                t = frame_index / clip.fps
                cx = math.sin(t * 1.7 + (clip_seed % 13)) * 0.25
                cy = math.cos(t * 1.3 + (clip_seed % 17)) * 0.18
                blob = np.exp(-(((xx - cx) ** 2) + ((yy - cy) ** 2)) / 0.14).astype(np.float32)
                rgb = np.stack(
                    [
                        np.clip(blob * base_exposure, 0.0, 1.0),
                        np.clip((blob * 0.85 + 0.08) * base_exposure, 0.0, 1.0),
                        np.clip((blob * 0.65 + 0.14) * base_exposure, 0.0, 1.0),
                    ],
                    axis=0,
                )
                if frame_index == 0:
                    patch_luma = np.clip(0.18 + ((clip_seed % 7) - 3) * 0.015, 0.08, 0.4)
                    rgb[:, 16:34, 28:68] = patch_luma
                if half_res:
                    rgb = rgb[:, ::2, ::2]
                yield frame_index, frame_index / clip.fps, rgb

        return iterator()


def _load_red_native_module():
    try:
        return importlib.import_module("r3dmatch._red_sdk_bridge")
    except ModuleNotFoundError:
        return None


def normalize_frame_to_chw(array: np.ndarray, *, source_path: str, frame_index: int, backend_name: str) -> np.ndarray:
    frame = np.asarray(array, dtype=np.float32)
    if frame.ndim != 3:
        raise RuntimeError(
            f"{backend_name} backend returned invalid frame for {source_path} frame {frame_index}. Expected 3 dimensions, got {frame.ndim}."
        )
    if frame.shape[0] == 3 and frame.shape[-1] != 3:
        return frame
    if frame.shape[-1] == 3:
        return np.moveaxis(frame, -1, 0)
    raise RuntimeError(
        f"{backend_name} backend returned unexpected frame shape for {source_path} frame {frame_index}: {tuple(frame.shape)}. Expected CHW or HWC RGB."
    )


class RedSdkDecoder(R3DBackend):
    name = "red"

    def __init__(self) -> None:
        self._native = _load_red_native_module()
        if self._native is None:
            raise RuntimeError(
                "RED backend requested but native bridge is not built. Set RED_SDK_ROOT and run scripts/build_red_sdk_bridge.sh."
            )
        if hasattr(self._native, "sdk_available") and not self._native.sdk_available():
            message = getattr(self._native, "unavailable_message", lambda: "RED SDK bridge unavailable.")()
            raise RuntimeError(message)

    def inspect_clip(self, source_path: str) -> ClipMetadata:
        try:
            payload = self._native.read_metadata(source_path)
        except Exception as exc:
            raise RuntimeError(
                f"RED backend failed to read metadata for {source_path}. Native error: {exc}"
            ) from exc
        clip_id = clip_id_from_path(source_path)
        metadata = dict(payload)
        return ClipMetadata(
            clip_id=clip_id,
            group_key=group_key_from_clip_id(clip_id),
            original_filename=original_filename_from_path(source_path),
            source_path=str(Path(source_path).expanduser().resolve()),
            fps=float(metadata.get("fps", 24.0)),
            width=int(metadata.get("width", 0)),
            height=int(metadata.get("height", 0)),
            total_frames=int(metadata.get("total_frames", 1)),
            reel_id=metadata.get("reel_id"),
            source_identifier=metadata.get("source_identifier"),
            color_space=str(metadata.get("color_space", "REDWideGamutRGB")),
            gamma_curve=str(metadata.get("gamma_curve", "Log3G10")),
            iso=float(metadata["iso"]) if metadata.get("iso") is not None else None,
            shutter_seconds=float(metadata["shutter_seconds"]) if metadata.get("shutter_seconds") is not None else None,
            aperture_t_stop=float(metadata["aperture_t_stop"]) if metadata.get("aperture_t_stop") is not None else None,
            active_lut_path=metadata.get("active_lut_path"),
            extra_metadata=metadata,
        )

    def decode_frames(
        self,
        source_path: str,
        *,
        start_frame: int,
        max_frames: int,
        frame_step: int,
        half_res: bool = False,
    ) -> Iterable[DecodedFrame]:
        clip = self.inspect_clip(source_path)
        selected = list(range(start_frame, clip.total_frames, frame_step))[:max_frames]

        def iterator() -> Iterator[DecodedFrame]:
            for frame_index in selected:
                try:
                    frame = self._native.decode_frame(
                        source_path,
                        frame_index,
                        half_res,
                        clip.color_space,
                        clip.gamma_curve,
                    )
                except Exception as exc:
                    raise RuntimeError(
                        f"RED backend failed to decode frame {frame_index} for {source_path}. Native error: {exc}"
                    ) from exc
                array = normalize_frame_to_chw(frame, source_path=source_path, frame_index=frame_index, backend_name="RED")
                yield frame_index, frame_index / max(clip.fps, 1e-6), array

        return iterator()


RedSdkBackend = RedSdkDecoder


def resolve_backend(name: str) -> R3DBackend:
    normalized = name.lower()
    if normalized == "mock":
        return MockR3DBackend()
    if normalized == "red":
        return RedSdkDecoder()
    if normalized == "red-sdk":
        return RedSdkDecoder()
    if normalized == "auto":
        return MockR3DBackend()
    raise ValueError("backend must be one of: auto, mock, red")
