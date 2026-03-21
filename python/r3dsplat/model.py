from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn


def _normalize_quaternion(quat: torch.Tensor) -> torch.Tensor:
    return quat / quat.norm(dim=-1, keepdim=True).clamp_min(1e-8)


@dataclass
class GaussianState:
    means: torch.Tensor
    quats: torch.Tensor
    scales: torch.Tensor
    opacities: torch.Tensor
    colors: torch.Tensor


class CanonicalGaussianModel(nn.Module):
    def __init__(self, num_gaussians: int = 256, feature_dim: int = 3) -> None:
        super().__init__()
        self.num_gaussians = num_gaussians
        self.feature_dim = feature_dim
        self.position = nn.Parameter(torch.randn(num_gaussians, 3) * 0.25)
        self.log_scale = nn.Parameter(torch.full((num_gaussians, 3), -1.8))
        self.rotation = nn.Parameter(
            torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32).repeat(num_gaussians, 1)
        )
        self.logit_opacity = nn.Parameter(torch.full((num_gaussians,), -1.5))
        self.features = nn.Parameter(torch.rand(num_gaussians, feature_dim))

    def canonical_state(self) -> GaussianState:
        return GaussianState(
            means=self.position,
            quats=_normalize_quaternion(self.rotation),
            scales=torch.exp(self.log_scale),
            opacities=torch.sigmoid(self.logit_opacity),
            colors=torch.sigmoid(self.features),
        )


class TimeConditionedDeformationModel(nn.Module):
    def __init__(self, feature_dim: int = 3, hidden_dim: int = 64, metadata_dim: int = 0) -> None:
        super().__init__()
        input_dim = 3 + 4 + 3 + feature_dim + 1 + metadata_dim
        output_dim = 3 + 3 + 4 + feature_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.feature_dim = feature_dim
        self.metadata_dim = metadata_dim

    def forward(
        self,
        canonical: GaussianState,
        timestamps: torch.Tensor,
        metadata: Optional[torch.Tensor] = None,
    ) -> GaussianState:
        if timestamps.dim() == 0:
            timestamps = timestamps[None]
        batch_size = timestamps.shape[0]
        num_gaussians = canonical.means.shape[0]
        t = timestamps[:, None, None].expand(batch_size, num_gaussians, 1)
        base = torch.cat(
            [
                canonical.means.unsqueeze(0).expand(batch_size, -1, -1),
                canonical.quats.unsqueeze(0).expand(batch_size, -1, -1),
                canonical.scales.unsqueeze(0).expand(batch_size, -1, -1),
                canonical.colors.unsqueeze(0).expand(batch_size, -1, -1),
                t,
            ],
            dim=-1,
        )
        if metadata is not None:
            metadata_expanded = metadata[:, None, :].expand(batch_size, num_gaussians, -1)
            base = torch.cat([base, metadata_expanded], dim=-1)
        outputs = self.mlp(base)
        pos_delta, scale_delta, quat_delta, color_delta = torch.split(
            outputs,
            [3, 3, 4, self.feature_dim],
            dim=-1,
        )
        deformed_quats = _normalize_quaternion(
            canonical.quats.unsqueeze(0).expand(batch_size, -1, -1) + 0.05 * quat_delta
        )
        return GaussianState(
            means=canonical.means.unsqueeze(0) + 0.05 * pos_delta,
            quats=deformed_quats,
            scales=(canonical.scales.unsqueeze(0) * torch.exp(0.05 * scale_delta)).clamp_min(1e-4),
            opacities=canonical.opacities.unsqueeze(0).expand(batch_size, -1),
            colors=(canonical.colors.unsqueeze(0) + 0.05 * color_delta).sigmoid(),
        )
