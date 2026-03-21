from __future__ import annotations

import json
import hashlib
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from .cache import SequenceCache
from .metadata import CameraRecord, ColmapSolveRecord


@dataclass
class ColmapConfig:
    executable: str = "colmap"
    mode: str = "standard"
    matching_mode: str = "sequential"
    intrinsics_mode: str = "shared"
    camera_model: str = "PINHOLE"


class ColmapCliBridge:
    def __init__(self, config: ColmapConfig):
        self.config = config

    def is_available(self) -> bool:
        return shutil.which(self.config.executable) is not None

    def executable_path(self) -> Optional[str]:
        return shutil.which(self.config.executable)

    def diagnostics(self) -> Dict[str, Any]:
        path = self.executable_path()
        return {
            "backend": "colmap-cli",
            "capability": "real-cli" if path else "missing",
            "executable": self.config.executable,
            "executable_path": path,
            "solve_mode": self.config.mode,
            "matching_mode": self.config.matching_mode,
            "intrinsics_mode": self.config.intrinsics_mode,
            "camera_model": self.config.camera_model,
        }

    def build_commands(self, image_dir: str, project_dir: str) -> List[List[str]]:
        database_path = str(Path(project_dir) / "database.db")
        sparse_dir = str(Path(project_dir) / "sparse")
        cmds = [
            [
                self.config.executable,
                "feature_extractor",
                "--database_path",
                database_path,
                "--image_path",
                image_dir,
                "--ImageReader.camera_model",
                self.config.camera_model,
                "--ImageReader.single_camera",
                "1" if self.config.intrinsics_mode == "shared" else "0",
            ],
        ]
        if self.config.matching_mode == "sequential":
            cmds.append([self.config.executable, "sequential_matcher", "--database_path", database_path])
        else:
            cmds.append([self.config.executable, "exhaustive_matcher", "--database_path", database_path])
        cmds.append(
            [
                self.config.executable,
                "mapper",
                "--database_path",
                database_path,
                "--image_path",
                image_dir,
                "--output_path",
                sparse_dir,
            ]
        )
        return cmds

    def export_frames(self, dataset_dir: str) -> Path:
        cache = SequenceCache(dataset_dir)
        manifest = cache.load_manifest()
        image_dir = cache.colmap_dir / "images"
        image_dir.mkdir(parents=True, exist_ok=True)
        try:
            from PIL import Image
        except Exception as exc:
            raise RuntimeError("COLMAP export requires Pillow to write image files") from exc
        export_manifest: List[Dict[str, Any]] = []
        for frame in manifest.frames:
            tensor = cache.read_frame(frame).clamp(0.0, 1.0)
            array = (tensor.permute(1, 2, 0).mul(255.0).byte().numpy())
            output_path = image_dir / f"{frame.frame_index:06d}.png"
            Image.fromarray(array).save(output_path)
            self._verify_exported_image(output_path, expected_width=manifest.clip.width, expected_height=manifest.clip.height)
            export_manifest.append(
                {
                    "frame_index": frame.frame_index,
                    "path": str(output_path),
                    "sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
                    "size_bytes": output_path.stat().st_size,
                }
            )
        if len(list(image_dir.glob("*.png"))) != len(manifest.frames):
            raise RuntimeError("Frame export count mismatch during COLMAP export")
        (cache.colmap_dir / "export_manifest.json").write_text(json.dumps(export_manifest, indent=2), encoding="utf-8")
        return image_dir

    def run(self, dataset_dir: str) -> Dict[str, Any]:
        cache = SequenceCache(dataset_dir)
        manifest = cache.load_manifest()
        project_dir = cache.colmap_dir / "project"
        project_dir.mkdir(parents=True, exist_ok=True)
        image_dir = self.export_frames(dataset_dir)
        commands = self.build_commands(str(image_dir), str(project_dir))
        if not self.is_available():
            diagnostics = self.diagnostics()
            raise RuntimeError(f"COLMAP executable not found on PATH for requested real solve: {diagnostics}")
        try:
            for command in commands:
                subprocess.run(command, check=True, cwd=project_dir, capture_output=True, text=True)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"COLMAP execution failed with command={exc.cmd!r} returncode={exc.returncode} stderr={exc.stderr.strip()}"
            ) from exc
        solve = ColmapSolveRecord(
            clip_id=manifest.clip.clip_id,
            colmap_project_dir=str(project_dir),
            database_path=str(project_dir / "database.db"),
            sparse_model_path=str(project_dir / "sparse"),
            camera_model_used=self.config.camera_model,
            intrinsics_mode=self.config.intrinsics_mode,
            matching_mode=self.config.matching_mode,
            mapper_mode=self.config.mode,
            registered_images=self._count_registered_images(project_dir / "sparse"),
            solve_status="completed",
            notes="Executed via COLMAP CLI",
        )
        updated = manifest.model_copy(
            update={
                "colmap_solves": manifest.colmap_solves + [solve],
                "backend_report": manifest.backend_report.model_copy(update={"colmap_backend": f"colmap:{self.config.mode}"}),
                "transforms_log": manifest.transforms_log + [f"colmap: executed {self.config.mode} solve"],
            }
        )
        cache.save_manifest(updated)
        return {
            "backend": self.diagnostics(),
            "commands": commands,
            "solve_status": solve.solve_status,
            "registered_images": solve.registered_images,
            "sparse_model_path": solve.sparse_model_path,
            "project_dir": str(project_dir),
        }

    @staticmethod
    def _count_registered_images(sparse_dir: Path) -> int:
        if not sparse_dir.exists():
            return 0
        image_files = list(sparse_dir.glob("*/images.txt")) + list(sparse_dir.glob("images.txt"))
        total = 0
        for image_file in image_files:
            try:
                total += len(parse_images_txt(image_file.read_text(encoding="utf-8")))
            except Exception:
                continue
        return total

    @staticmethod
    def _verify_exported_image(path: Path, expected_width: int, expected_height: int) -> None:
        if not path.exists():
            raise RuntimeError(f"missing exported image: {path}")
        if path.stat().st_size <= 0:
            raise RuntimeError(f"exported image has zero size: {path}")
        from PIL import Image

        with Image.open(path) as image:
            if image.size != (expected_width, expected_height):
                raise RuntimeError(
                    f"exported image resolution mismatch for {path}: got={image.size} expected={(expected_width, expected_height)}"
                )


def parse_images_txt(images_txt: str) -> List[CameraRecord]:
    records: List[CameraRecord] = []
    for line in images_txt.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 10:
            continue
        image_id = parts[0]
        qw, qx, qy, qz = [float(value) for value in parts[1:5]]
        tx, ty, tz = [float(value) for value in parts[5:8]]
        name = parts[9]
        frame_index = int(Path(name).stem)
        records.append(
            CameraRecord(
                camera_record_id=f"colmap:{image_id}",
                frame_record_id=f"unknown:{frame_index:06d}",
                intrinsics={"fx": 0.0, "fy": 0.0, "cx": 0.0, "cy": 0.0},
                extrinsics_world_to_camera=[
                    [1.0, 0.0, 0.0, tx],
                    [0.0, 1.0, 0.0, ty],
                    [0.0, 0.0, 1.0, tz],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                extrinsics_camera_to_world=[
                    [1.0, 0.0, 0.0, -tx],
                    [0.0, 1.0, 0.0, -ty],
                    [0.0, 0.0, 1.0, -tz],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                source_of_pose="colmap",
                pose_confidence=1.0,
                alignment_status="colmap_world",
                pose_provenance=[f"colmap_quat=({qw},{qx},{qy},{qz})"],
            )
        )
    return records


def solve_colmap(dataset_dir: str, config: ColmapConfig) -> Dict[str, Any]:
    bridge = ColmapCliBridge(config)
    return bridge.run(dataset_dir)
