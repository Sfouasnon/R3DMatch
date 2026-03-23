from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

from .cache import SequenceCache
from .metadata import MaskRecord


def apply_masked_l1(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    if mask is None:
        return torch.nn.functional.l1_loss(pred, target)
    while mask.dim() < pred.dim():
        mask = mask.unsqueeze(2)
    mask = mask.expand_as(pred)
    diff = (pred - target).abs() * (1.0 - mask)
    denom = (1.0 - mask).sum().clamp_min(1.0)
    return diff.sum() / denom


@dataclass
class BackgroundMattingConfig:
    repo_dir: str = ""
    background_dir: str = ""
    checkpoint: str = ""
    model_type: str = "mattingrefine"
    model_backbone: str = "resnet50"
    model_backbone_scale: float = 0.25
    model_refine_mode: str = "sampling"
    model_refine_sample_pixels: int = 80000
    device: str = "cuda"
    num_workers: int = 0
    preprocess_alignment: bool = False


def run_background_matting(dataset_dir: str, config: BackgroundMattingConfig) -> Dict[str, Any]:
    repo_dir = Path(config.repo_dir).expanduser()
    if not config.repo_dir:
        raise RuntimeError("BackgroundMattingV2 integration requires --repo-dir")
    script_path = repo_dir / "inference_images.py"
    if not script_path.exists():
        raise RuntimeError(
            f"BackgroundMattingV2 not found at {repo_dir}. Expected inference_images.py there."
        )
    if not config.background_dir:
        raise RuntimeError("BackgroundMattingV2 integration requires --background-dir")
    if not config.checkpoint:
        raise RuntimeError("BackgroundMattingV2 integration requires --checkpoint")

    cache = SequenceCache(dataset_dir)
    manifest = cache.load_manifest()
    matte_dir = cache.matte_images_dir
    if not matte_dir.exists():
        raise RuntimeError(f"Missing matte_images directory: {matte_dir}")

    background_dir = Path(config.background_dir).expanduser()
    if not background_dir.exists():
        raise RuntimeError(f"Background directory does not exist: {background_dir}")
    checkpoint = Path(config.checkpoint).expanduser()
    if not checkpoint.exists():
        raise RuntimeError(f"BackgroundMattingV2 checkpoint does not exist: {checkpoint}")

    output_dir = cache.dataset_dir / ".backgroundmattingv2"
    if output_dir.exists():
        shutil.rmtree(output_dir)

    command = [
        sys.executable,
        str(script_path),
        "--model-type",
        config.model_type,
        "--model-backbone",
        config.model_backbone,
        "--model-backbone-scale",
        str(config.model_backbone_scale),
        "--model-refine-mode",
        config.model_refine_mode,
        "--model-refine-sample-pixels",
        str(config.model_refine_sample_pixels),
        "--model-checkpoint",
        str(checkpoint),
        "--images-src",
        str(matte_dir),
        "--images-bgr",
        str(background_dir),
        "--output-dir",
        str(output_dir),
        "--output-types",
        "pha",
        "--device",
        config.device,
        "--num-workers",
        str(config.num_workers),
        "-y",
    ]
    if config.preprocess_alignment:
        command.append("--preprocess-alignment")

    try:
        subprocess.run(command, check=True, cwd=repo_dir, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"BackgroundMattingV2 execution failed with command={exc.cmd!r} returncode={exc.returncode} stderr={exc.stderr.strip()}"
        ) from exc

    pha_dir = output_dir / "pha"
    if not pha_dir.exists():
        raise RuntimeError(f"BackgroundMattingV2 did not produce pha outputs at {pha_dir}")

    try:
        from PIL import Image
    except Exception as exc:
        raise RuntimeError("Reading BackgroundMattingV2 outputs requires Pillow") from exc

    mask_records: list[MaskRecord] = []
    mask_map: dict[int, str] = {}
    for frame in manifest.frames:
        source = pha_dir / f"{frame.frame_index:06d}.jpg"
        if not source.exists():
            alt = pha_dir / f"{frame.frame_index:06d}.png"
            source = alt if alt.exists() else source
        if not source.exists():
            raise RuntimeError(f"Missing BackgroundMattingV2 matte for frame {frame.frame_index}: {source}")
        with Image.open(source) as image:
            tensor = torch.from_numpy(np.array(image)).to(torch.float32)
        if tensor.dim() == 3:
            tensor = tensor[..., 0]
        tensor = (tensor / 255.0).clamp(0.0, 1.0)
        saved = cache.write_mask(frame.frame_index, tensor, mask_type="matte")
        mask_map[frame.frame_index] = str(saved)
        mask_records.append(
            MaskRecord(
                clip_id=frame.clip_id,
                frame_index=frame.frame_index,
                mask_type="matte",
                mask_path=str(saved),
                provenance="backgroundmattingv2",
            )
        )

    updated_frames = [
        frame.model_copy(update={"mask_path": mask_map.get(frame.frame_index, frame.mask_path)})
        for frame in manifest.frames
    ]
    updated = manifest.model_copy(
        update={
            "frames": updated_frames,
            "masks": [mask for mask in manifest.masks if mask.mask_type != "matte"] + mask_records,
            "transforms_log": manifest.transforms_log + ["masks: generated via BackgroundMattingV2"],
        }
    )
    cache.save_manifest(updated)
    return {
        "backend": "backgroundmattingv2",
        "repo_dir": str(repo_dir),
        "matte_image_dir": str(matte_dir),
        "background_dir": str(background_dir),
        "checkpoint": str(checkpoint),
        "output_dir": str(output_dir),
        "mask_count": len(mask_records),
    }
