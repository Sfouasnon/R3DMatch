from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

import torch
from torch import nn

from .model import GaussianState


def _ensure_local_gsplat_on_path() -> None:
    workspace_root = Path(__file__).resolve().parents[2]
    local_gsplat = workspace_root / "GSplat" / "gsplat-main"
    if local_gsplat.exists():
        gsplat_path = str(local_gsplat)
        if gsplat_path not in sys.path:
            sys.path.insert(0, gsplat_path)


class GSplatRendererBridge(nn.Module):
    def __init__(self, production_mode: bool = False) -> None:
        super().__init__()
        self._gsplat = None
        self.backend = "torch"
        self.backend_detail = "gsplat unavailable"
        self.production_mode = production_mode
        _ensure_local_gsplat_on_path()
        try:
            import gsplat  # type: ignore

            self._gsplat = gsplat
            self.backend = "gsplat"
            self.backend_detail = "gsplat import succeeded"
        except Exception:
            self._gsplat = None
        if self.production_mode and self._gsplat is None:
            raise RuntimeError("Production renderer mode requires real gsplat; no fallback allowed")

    def diagnostics(self) -> Dict[str, Any]:
        return {
            "backend": self.backend,
            "backend_detail": self.backend_detail,
            "production_mode": self.production_mode,
            "gsplat_importable": self._gsplat is not None,
        }

    def forward(
        self,
        state: GaussianState,
        cameras: dict[str, torch.Tensor],
        image_size: tuple[int, int],
    ) -> torch.Tensor:
        if state.means.dim() == 2:
            return self._render_single(state, cameras, image_size)

        rendered = []
        for batch_index in range(state.means.shape[0]):
            rendered.append(
                self._render_single(
                    GaussianState(
                        means=state.means[batch_index],
                        quats=state.quats[batch_index],
                        scales=state.scales[batch_index],
                        opacities=state.opacities[batch_index],
                        colors=state.colors[batch_index],
                    ),
                    {key: value[batch_index : batch_index + 1] for key, value in cameras.items()},
                    image_size,
                )
            )
        return torch.cat(rendered, dim=0)

    def _render_single(
        self,
        state: GaussianState,
        cameras: dict[str, torch.Tensor],
        image_size: tuple[int, int],
    ) -> torch.Tensor:
        if self._gsplat is not None:
            return self._render_with_gsplat(state, cameras, image_size)
        return self._render_with_torch(state, image_size)

    def _render_with_gsplat(
        self,
        state: GaussianState,
        cameras: dict[str, torch.Tensor],
        image_size: tuple[int, int],
    ) -> torch.Tensor:
        width, height = image_size[1], image_size[0]
        if state.colors.dim() != 2 or state.colors.shape[-1] != 3:
            raise RuntimeError(f"gsplat renderer expects colors as [N,3], got shape={tuple(state.colors.shape)}")
        if cameras["viewmats"].dim() != 3 or cameras["viewmats"].shape[-2:] != (4, 4):
            raise RuntimeError(f"gsplat renderer expects viewmats as [C,4,4], got shape={tuple(cameras['viewmats'].shape)}")
        if cameras["Ks"].dim() != 3 or cameras["Ks"].shape[-2:] != (3, 3):
            raise RuntimeError(f"gsplat renderer expects intrinsics as [C,3,3], got shape={tuple(cameras['Ks'].shape)}")
        colors, _, _ = self._gsplat.rasterization(
            state.means,
            state.quats,
            state.scales,
            state.opacities,
            state.colors,
            cameras["viewmats"],
            cameras["Ks"],
            width=width,
            height=height,
            backgrounds=cameras.get("backgrounds"),
            render_mode="RGB",
            packed=False,
        )
        if colors.dim() == 4:
            colors = colors.unsqueeze(0)
        return colors.squeeze(0).permute(0, 3, 1, 2).contiguous()

    def _render_with_torch(self, state: GaussianState, image_size: tuple[int, int]) -> torch.Tensor:
        height, width = image_size
        device = state.means.device
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, height, device=device),
            torch.linspace(-1.0, 1.0, width, device=device),
            indexing="ij",
        )
        means = state.means
        scales = state.scales.mean(dim=-1).clamp_min(1e-4)
        zs = means[:, 2].abs().add(1.0)
        px = torch.tanh(means[:, 0] / zs)
        py = torch.tanh(means[:, 1] / zs)
        sigma = torch.clamp(scales / zs, min=0.02, max=0.5)
        dx = xx.unsqueeze(0) - px[:, None, None]
        dy = yy.unsqueeze(0) - py[:, None, None]
        weights = torch.exp(-0.5 * ((dx / sigma[:, None, None]) ** 2 + (dy / sigma[:, None, None]) ** 2))
        weights = weights * state.opacities[:, None, None]
        numer = (weights[:, None, :, :] * state.colors[:, :, None, None]).sum(dim=0)
        denom = weights.sum(dim=0, keepdim=True).clamp_min(1e-5)
        image = (numer / denom).clamp(0.0, 1.0)
        return image.unsqueeze(0)
