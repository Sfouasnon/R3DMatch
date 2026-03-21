from __future__ import annotations

from pathlib import Path
from typing import Union

import torch

from .metadata import (
    BackendReport,
    CameraRecord,
    ClipRecord,
    ColmapSolveRecord,
    DatasetManifest,
    FiducialSolveRecord,
    FrameRecord,
    MaskRecord,
)


class SequenceCache:
    def __init__(self, dataset_dir: Union[str, Path]):
        self.dataset_dir = Path(dataset_dir)
        self.frames_dir = self.dataset_dir / "frames"
        self.masks_dir = self.dataset_dir / "masks"
        self.colmap_dir = self.dataset_dir / "colmap"
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.masks_dir.mkdir(parents=True, exist_ok=True)
        self.colmap_dir.mkdir(parents=True, exist_ok=True)

    def frame_path(self, frame_index: int) -> Path:
        return self.frames_dir / f"{frame_index:06d}.pt"

    def write_frame(self, frame_index: int, frame_tensor: torch.Tensor) -> Path:
        path = self.frame_path(frame_index)
        torch.save(frame_tensor.detach().cpu(), path)
        return path

    def mask_path(self, frame_index: int, mask_type: str = "fiducial") -> Path:
        return self.masks_dir / f"{frame_index:06d}.{mask_type}.pt"

    def write_mask(self, frame_index: int, mask_tensor: torch.Tensor, mask_type: str = "fiducial") -> Path:
        path = self.mask_path(frame_index, mask_type=mask_type)
        torch.save(mask_tensor.detach().cpu(), path)
        return path

    def read_mask(self, path: str) -> torch.Tensor:
        return torch.load(path, map_location="cpu", weights_only=False)

    def load_manifest(self) -> DatasetManifest:
        return DatasetManifest.load(self.dataset_dir)

    def save_manifest(self, manifest: DatasetManifest) -> None:
        manifest.save(self.dataset_dir)

    def read_frame(self, frame_record: FrameRecord) -> torch.Tensor:
        return torch.load(frame_record.cache_path, map_location="cpu", weights_only=False)

    def build_manifest(
        self,
        clip: ClipRecord,
        frames: list[FrameRecord],
        transforms_log: list[str],
        camera_records: list[CameraRecord] | None = None,
        fiducial_solves: list[FiducialSolveRecord] | None = None,
        colmap_solves: list[ColmapSolveRecord] | None = None,
        masks: list[MaskRecord] | None = None,
        backend_report: BackendReport | None = None,
    ) -> DatasetManifest:
        manifest = DatasetManifest(
            manifest_version=2,
            clip=clip,
            frames=frames,
            camera_records=camera_records or [],
            fiducial_solves=fiducial_solves or [],
            colmap_solves=colmap_solves or [],
            masks=masks or [],
            backend_report=backend_report or BackendReport(),
            transforms_log=transforms_log,
        )
        manifest.save(self.dataset_dir)
        return manifest
