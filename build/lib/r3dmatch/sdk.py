from __future__ import annotations

import hashlib
import math
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Tuple

import numpy as np

from .models import ClipMetadata


DecodedFrame = Tuple[int, float, np.ndarray]


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
    ) -> Iterable[DecodedFrame]:
        raise NotImplementedError


class MockR3DBackend(R3DBackend):
    name = "mock"

    def inspect_clip(self, source_path: str) -> ClipMetadata:
        clip_id = Path(source_path).stem
        digest = hashlib.sha1(clip_id.encode("utf-8")).digest()
        iso = 400.0 + float(digest[0] % 5) * 160.0
        return ClipMetadata(
            clip_id=clip_id,
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
                yield frame_index, frame_index / clip.fps, rgb

        return iterator()


class RedSdkBackend(R3DBackend):
    name = "red-sdk"

    def inspect_clip(self, source_path: str) -> ClipMetadata:
        raise RuntimeError("RED SDK adapter is not wired yet in this standalone prototype.")

    def decode_frames(
        self,
        source_path: str,
        *,
        start_frame: int,
        max_frames: int,
        frame_step: int,
    ) -> Iterable[DecodedFrame]:
        raise RuntimeError("RED SDK adapter is not wired yet in this standalone prototype.")


def resolve_backend(name: str) -> R3DBackend:
    normalized = name.lower()
    if normalized == "mock":
        return MockR3DBackend()
    if normalized == "red-sdk":
        return RedSdkBackend()
    if normalized == "auto":
        return MockR3DBackend()
    raise ValueError("backend must be one of: auto, mock, red-sdk")

