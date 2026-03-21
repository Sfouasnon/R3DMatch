from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
import yaml
from torch import nn
from torch.utils.data import DataLoader

from .dataset import TemporalSequenceDataset
from .masking import apply_masked_l1
from .model import CanonicalGaussianModel, GaussianState, TimeConditionedDeformationModel
from .training_bridge import GSplatRendererBridge


@dataclass
class TrainingConfig:
    dataset_dir: str
    output_dir: str
    mode: str = "train-4d"
    window_size: int = 4
    stride: int = 1
    batch_size: int = 1
    epochs: int = 2
    learning_rate: float = 1e-2
    num_gaussians: int = 256
    smoothness_weight: float = 0.1
    regularization_weight: float = 0.01
    production_renderer: bool = False
    seed: int = 7
    grad_clip_norm: float = 5.0
    max_abs_position: float = 10.0
    log_scale_min: float = -6.0
    log_scale_max: float = 2.0
    feature_min: float = -8.0
    feature_max: float = 8.0
    opacity_logit_min: float = -10.0
    opacity_logit_max: float = 10.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def from_yaml(cls, path: Optional[str], **overrides: Any) -> "TrainingConfig":
        payload: dict[str, Any] = {}
        if path:
            payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        payload.update({key: value for key, value in overrides.items() if value is not None})
        return cls(**payload)


class FourDGaussianTrainer(nn.Module):
    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.config = config
        self.gaussians = CanonicalGaussianModel(num_gaussians=config.num_gaussians)
        self.deformation = TimeConditionedDeformationModel(feature_dim=self.gaussians.feature_dim)
        self.renderer = GSplatRendererBridge(production_mode=config.production_renderer)

    def forward(self, batch: dict[str, Any]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        frames = batch["frames"]
        timestamps = batch["timestamps"]
        cameras = batch["cameras"]
        masks = batch.get("masks")
        target = frames
        canonical = self.gaussians.canonical_state()
        image_size = (frames.shape[-2], frames.shape[-1])
        if self.config.mode == "train-static":
            static_pred = self.renderer(canonical, self._select_camera(cameras, 0), image_size)
            pred = static_pred.unsqueeze(1).expand(-1, target.shape[1], -1, -1, -1)
            zero = torch.zeros((), device=target.device)
            losses = self._compose_losses(pred, target, masks, zero, zero)
            return pred, losses

        batch_size, timesteps = timestamps.shape
        predictions = []
        position_deltas = []
        scale_deltas = []
        for timestep in range(timesteps):
            state = self.deformation(canonical, timestamps[:, timestep])
            prediction = self.renderer(state, self._select_camera(cameras, timestep), image_size)
            predictions.append(prediction)
            position_deltas.append(state.means - canonical.means.unsqueeze(0))
            scale_deltas.append(state.scales - canonical.scales.unsqueeze(0))
        pred = torch.stack(predictions, dim=1)
        smoothness = self._temporal_smoothness(position_deltas, scale_deltas)
        regularization = self._deformation_regularization(position_deltas, scale_deltas)
        losses = self._compose_losses(pred, target, masks, smoothness, regularization)
        self._assert_finite_tensor("pred", pred)
        return pred, losses

    @staticmethod
    def _select_camera(cameras: dict[str, torch.Tensor], timestep: int) -> dict[str, torch.Tensor]:
        selected = {}
        for key, value in cameras.items():
            if value.dim() >= 2 and value.shape[1] > timestep:
                selected[key] = value[:, timestep : timestep + 1]
            else:
                selected[key] = value
        return selected

    def _compose_losses(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        masks: Optional[torch.Tensor],
        smoothness: torch.Tensor,
        regularization: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        photometric = apply_masked_l1(pred, target, masks)
        total = (
            photometric
            + self.config.smoothness_weight * smoothness
            + self.config.regularization_weight * regularization
        )
        return {
            "total": total,
            "photometric": photometric,
            "temporal_smoothness": smoothness,
            "deformation_regularization": regularization,
        }

    @staticmethod
    def _temporal_smoothness(
        position_deltas: list[torch.Tensor],
        scale_deltas: list[torch.Tensor],
    ) -> torch.Tensor:
        if len(position_deltas) < 2:
            return torch.zeros((), device=position_deltas[0].device)
        terms = []
        for current, nxt in zip(position_deltas[:-1], position_deltas[1:]):
            terms.append((nxt - current).pow(2).mean())
        for current, nxt in zip(scale_deltas[:-1], scale_deltas[1:]):
            terms.append((nxt - current).pow(2).mean())
        return torch.stack(terms).mean()

    @staticmethod
    def _deformation_regularization(
        position_deltas: list[torch.Tensor],
        scale_deltas: list[torch.Tensor],
    ) -> torch.Tensor:
        terms = [delta.pow(2).mean() for delta in position_deltas]
        terms.extend(delta.pow(2).mean() for delta in scale_deltas)
        return torch.stack(terms).mean()

    @staticmethod
    def _assert_finite_tensor(name: str, tensor: torch.Tensor) -> None:
        if not torch.isfinite(tensor).all():
            raise RuntimeError(f"non-finite tensor detected: {name}")


def _collate(batch: list) -> Dict[str, Any]:
    return {
        "frames": torch.stack([item["frames"] for item in batch], dim=0),
        "timestamps": torch.stack([item["timestamps"] for item in batch], dim=0),
        "frame_indices": torch.stack([item["frame_indices"] for item in batch], dim=0),
        "cameras": {
            key: torch.stack([item["cameras"][key] for item in batch], dim=0)
            for key in batch[0]["cameras"].keys()
        },
        "metadata": [item["metadata"] for item in batch],
        "clip": [item["clip"] for item in batch],
        "masks": None if batch[0].get("masks") is None and all(item.get("masks") is None for item in batch) else torch.stack(
            [
                item["masks"] if item.get("masks") is not None else torch.zeros(
                    (item["frames"].shape[0], item["frames"].shape[-2], item["frames"].shape[-1]),
                    dtype=item["frames"].dtype,
                )
                for item in batch
            ],
            dim=0,
        ),
    }


def train(config: TrainingConfig) -> Path:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(config.seed)
    dataset = TemporalSequenceDataset(
        config.dataset_dir,
        window_size=config.window_size,
        stride=config.stride,
    )
    generator = torch.Generator().manual_seed(config.seed)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=_collate,
        generator=generator,
    )
    trainer = FourDGaussianTrainer(config).to(config.device)
    optimizer = torch.optim.Adam(trainer.parameters(), lr=config.learning_rate)
    history: list[dict[str, float]] = []
    run_summary = {
        "config": asdict(config),
        "dataset_validation": dataset.validation_report,
        "renderer": trainer.renderer.diagnostics(),
        "epochs": [],
    }

    for epoch in range(config.epochs):
        epoch_losses: list[dict[str, float]] = []
        for batch in loader:
            batch = _move_batch_to_device(batch, config.device)
            _, losses = trainer(batch)
            _assert_finite_losses(losses)
            optimizer.zero_grad(set_to_none=True)
            losses["total"].backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(trainer.parameters(), max_norm=config.grad_clip_norm)
            if not torch.isfinite(torch.as_tensor(grad_norm)).all():
                raise RuntimeError("non-finite gradient norm detected")
            optimizer.step()
            _clamp_parameters(trainer, config)
            loss_entry = {key: float(value.detach().cpu()) for key, value in losses.items()}
            loss_entry["grad_norm"] = float(torch.as_tensor(grad_norm).detach().cpu())
            history.append(loss_entry)
            epoch_losses.append(loss_entry)
        run_summary["epochs"].append(
            {
                "epoch": epoch,
                "steps": len(epoch_losses),
                "last": epoch_losses[-1] if epoch_losses else {},
            }
        )

    checkpoint = {
        "config": asdict(config),
        "state_dict": trainer.state_dict(),
        "history": history,
        "renderer_backend": trainer.renderer.backend,
        "renderer_backend_detail": trainer.renderer.backend_detail,
        "renderer_diagnostics": trainer.renderer.diagnostics(),
    }
    checkpoint_path = output_dir / f"{config.mode}.pt"
    torch.save(checkpoint, checkpoint_path)
    (output_dir / "train_summary.yaml").write_text(yaml.safe_dump({"history": history}), encoding="utf-8")
    run_summary["checkpoint"] = str(checkpoint_path)
    run_summary["history_length"] = len(history)
    run_summary["final_losses"] = history[-1] if history else {}
    (output_dir / "run_summary.json").write_text(json.dumps(run_summary, indent=2), encoding="utf-8")
    return checkpoint_path


def evaluate(checkpoint_path: str) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    history = checkpoint.get("history", [])
    last = history[-1] if history else {}
    return {
        "checkpoint": checkpoint_path,
        "renderer_backend": checkpoint.get("renderer_backend"),
        "renderer_backend_detail": checkpoint.get("renderer_backend_detail"),
        "renderer_diagnostics": checkpoint.get("renderer_diagnostics"),
        "final_losses": last,
    }


def render_sequence(checkpoint_path: str, dataset_dir: Optional[str] = None) -> torch.Tensor:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = TrainingConfig(**checkpoint["config"])
    if dataset_dir is not None:
        config.dataset_dir = dataset_dir
    dataset = TemporalSequenceDataset(config.dataset_dir, window_size=config.window_size, stride=config.stride)
    batch = _collate([dataset[0]])
    batch = _move_batch_to_device(batch, config.device)
    trainer = FourDGaussianTrainer(config).to(config.device)
    trainer.load_state_dict(checkpoint["state_dict"])
    trainer.eval()
    with torch.no_grad():
        pred, _ = trainer(batch)
    return pred.detach().cpu()


def _move_batch_to_device(batch: dict[str, Any], device: str) -> dict[str, Any]:
    return {
        "frames": batch["frames"].to(device),
        "timestamps": batch["timestamps"].to(device),
        "frame_indices": batch["frame_indices"].to(device),
        "cameras": {key: value.to(device) for key, value in batch["cameras"].items()},
        "metadata": batch["metadata"],
        "clip": batch["clip"],
        "masks": None if batch.get("masks") is None else batch["masks"].to(device),
    }


def _assert_finite_losses(losses: Dict[str, torch.Tensor]) -> None:
    for name, value in losses.items():
        if not torch.isfinite(value).all():
            raise RuntimeError(f"non-finite loss detected: {name}")


def _clamp_parameters(trainer: FourDGaussianTrainer, config: TrainingConfig) -> None:
    with torch.no_grad():
        trainer.gaussians.position.clamp_(-config.max_abs_position, config.max_abs_position)
        trainer.gaussians.log_scale.clamp_(config.log_scale_min, config.log_scale_max)
        trainer.gaussians.features.clamp_(config.feature_min, config.feature_max)
        trainer.gaussians.logit_opacity.clamp_(config.opacity_logit_min, config.opacity_logit_max)
