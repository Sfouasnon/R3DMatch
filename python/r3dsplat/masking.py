from __future__ import annotations

from typing import Optional

import torch


def apply_masked_l1(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    if mask is None:
        return torch.nn.functional.l1_loss(pred, target)
    while mask.dim() < pred.dim():
        mask = mask.unsqueeze(2)
    mask = mask.expand_as(pred)
    diff = (pred - target).abs() * (1.0 - mask)
    denom = (1.0 - mask).sum().clamp_min(1.0)
    return diff.sum() / denom
