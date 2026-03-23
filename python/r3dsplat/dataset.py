from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Union

import torch
from torch.utils.data import Dataset

from .cache import SequenceCache
from .metadata import CameraRecord, DatasetManifest, FrameRecord


@dataclass
class TemporalWindow:
    frames: torch.Tensor
    timestamps: torch.Tensor
    frame_indices: torch.Tensor
    cameras: dict[str, torch.Tensor]
    metadata: list[dict[str, Any]]
    masks: torch.Tensor | None = None


class TemporalSequenceDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        dataset_dir: Union[str, Path],
        window_size: int = 4,
        stride: int = 1,
    ) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.window_size = window_size
        self.stride = stride
        self.manifest = DatasetManifest.load(self.dataset_dir)
        self.cache = SequenceCache(self.dataset_dir)
        self.frames = sorted(self.manifest.frames, key=lambda frame: frame.frame_index)
        self.camera_records = self.manifest.camera_by_frame_id()
        self.mask_records = self.manifest.masks_by_frame_index()
        self.cached_width = self.frames[0].cached_width or self.frames[0].decoded_width or self.manifest.clip.width
        self.cached_height = self.frames[0].cached_height or self.frames[0].decoded_height or self.manifest.clip.height
        self.validation_report = self._validate_manifest()
        if len(self.frames) < self.window_size:
            raise ValueError("dataset contains fewer frames than requested temporal window")
        self.start_indices = list(range(0, len(self.frames) - self.window_size + 1, self.stride))

    def __len__(self) -> int:
        return len(self.start_indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        start = self.start_indices[index]
        records = self.frames[start : start + self.window_size]
        frames = [self.cache.read_frame(record).to(dtype=torch.float32) for record in records]
        self._validate_frames(records, frames)
        masks = [self._read_mask(record) for record in records]
        self._validate_masks(records, masks)
        timestamps = torch.tensor([record.timestamp_seconds for record in records], dtype=torch.float32)
        frame_indices = torch.tensor([record.frame_index for record in records], dtype=torch.long)
        self._validate_timestamps(timestamps, frame_indices)
        stacked_masks = None
        if any(mask is not None for mask in masks):
            stacked_masks = torch.stack(
                [mask if mask is not None else torch.zeros((self.cached_height, self.cached_width), dtype=torch.float32) for mask in masks],
                dim=0,
            )
        return {
            "frames": torch.stack(frames, dim=0),
            "timestamps": timestamps,
            "frame_indices": frame_indices,
            "cameras": self._stack_cameras(records),
            "metadata": [record.payload() for record in records],
            "clip": self.manifest.clip.model_dump(mode="json"),
            "masks": stacked_masks,
        }

    def _stack_cameras(self, records: list[FrameRecord]) -> dict[str, torch.Tensor]:
        viewmats = []
        intrinsics = []
        backgrounds = []
        near = []
        far = []
        for record in records:
            camera_record = self._camera_record_for_frame(record)
            viewmats.append(camera_record.extrinsics_world_to_camera)
            intrinsics.append(
                [
                    [camera_record.intrinsics.get("fx", record.camera.fx), 0.0, camera_record.intrinsics.get("cx", record.camera.cx)],
                    [0.0, camera_record.intrinsics.get("fy", record.camera.fy), camera_record.intrinsics.get("cy", record.camera.cy)],
                    [0.0, 0.0, 1.0],
                ]
            )
            backgrounds.append(record.camera.background)
            near.append(record.camera.near_plane)
            far.append(record.camera.far_plane)
        return {
            "viewmats": torch.tensor(viewmats, dtype=torch.float32),
            "Ks": torch.tensor(intrinsics, dtype=torch.float32),
            "backgrounds": torch.tensor(backgrounds, dtype=torch.float32),
            "near_plane": torch.tensor(near, dtype=torch.float32),
            "far_plane": torch.tensor(far, dtype=torch.float32),
        }

    def _camera_record_for_frame(self, record: FrameRecord) -> CameraRecord:
        if record.camera_record_id and record.camera_record_id in self.camera_records:
            return self.camera_records[record.camera_record_id]
        return CameraRecord(
            camera_record_id=record.camera_record_id or f"{record.frame_record_id}:camera",
            frame_record_id=record.frame_record_id,
            intrinsics={
                "fx": record.camera.fx,
                "fy": record.camera.fy,
                "cx": record.camera.cx,
                "cy": record.camera.cy,
            },
            extrinsics_world_to_camera=record.camera.view_matrix,
            extrinsics_camera_to_world=record.camera.view_matrix,
            source_of_pose="embedded",
            pose_confidence=0.25,
            alignment_status="raw",
            pose_provenance=["frame.camera"],
        )

    def _read_mask(self, record: FrameRecord) -> torch.Tensor | None:
        mask_path = record.mask_path
        if not mask_path:
            mask_record = self.mask_records.get(record.frame_index)
            mask_path = mask_record.mask_path if mask_record is not None else None
        if not mask_path:
            return None
        return self.cache.read_mask(mask_path).to(dtype=torch.float32)

    def _validate_manifest(self) -> dict[str, Any]:
        if not self.frames:
            raise ValueError("dataset manifest contains no frames")
        frame_indices = [frame.frame_index for frame in self.frames]
        if len(frame_indices) != len(set(frame_indices)):
            raise ValueError("dataset manifest contains duplicate frame indices")
        timestamps = [frame.timestamp_seconds for frame in self.frames]
        if any(timestamps[idx] >= timestamps[idx + 1] for idx in range(len(timestamps) - 1)):
            raise ValueError("dataset manifest timestamps must be strictly increasing")
        return {
            "frame_count": len(self.frames),
            "width": self.cached_width,
            "height": self.cached_height,
            "original_width": self.manifest.clip.width,
            "original_height": self.manifest.clip.height,
            "camera_record_count": len(self.manifest.camera_records),
            "mask_count": len(self.manifest.masks),
        }

    def _validate_frames(self, records: list[FrameRecord], frames: list[torch.Tensor]) -> None:
        for record, frame in zip(records, frames):
            expected_shape = (3, self.cached_height, self.cached_width)
            if tuple(frame.shape) != expected_shape:
                raise ValueError(
                    f"frame shape mismatch for frame_index={record.frame_index}: got={tuple(frame.shape)} expected={expected_shape}"
                )
            if not torch.isfinite(frame).all():
                raise ValueError(f"non-finite values detected in cached frame {record.frame_index}")

    def _validate_masks(self, records: list[FrameRecord], masks: list[torch.Tensor | None]) -> None:
        for record, mask in zip(records, masks):
            if mask is None:
                continue
            expected_shape = (self.cached_height, self.cached_width)
            if tuple(mask.shape) != expected_shape:
                raise ValueError(
                    f"mask shape mismatch for frame_index={record.frame_index}: got={tuple(mask.shape)} expected={expected_shape}"
                )
            if not torch.isfinite(mask).all():
                raise ValueError(f"non-finite values detected in mask for frame {record.frame_index}")
            if mask.min().item() < 0.0 or mask.max().item() > 1.0:
                raise ValueError(f"mask values must lie in [0,1] for frame {record.frame_index}")

    @staticmethod
    def _validate_timestamps(timestamps: torch.Tensor, frame_indices: torch.Tensor) -> None:
        if timestamps.numel() > 1 and not torch.all(timestamps[1:] > timestamps[:-1]):
            raise ValueError(f"window timestamps not strictly increasing for frames {frame_indices.tolist()}")
