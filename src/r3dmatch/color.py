from __future__ import annotations

from typing import Dict

import numpy as np


def compute_chromaticity_medians(rgb_pixels: np.ndarray) -> Dict[str, float]:
    luminance = np.clip(
        rgb_pixels[:, 0] * 0.2126 + rgb_pixels[:, 1] * 0.7152 + rgb_pixels[:, 2] * 0.0722,
        1e-6,
        None,
    )
    chroma = np.stack(
        [
            rgb_pixels[:, 0] / luminance,
            rgb_pixels[:, 1] / luminance,
            rgb_pixels[:, 2] / luminance,
        ],
        axis=1,
    )
    return {
        "r": float(np.median(chroma[:, 0])),
        "g": float(np.median(chroma[:, 1])),
        "b": float(np.median(chroma[:, 2])),
    }


def solve_neutral_gains(r_median: float, g_median: float, b_median: float) -> Dict[str, float]:
    return {
        "r": float(g_median / max(r_median, 1e-6)),
        "g": 1.0,
        "b": float(g_median / max(b_median, 1e-6)),
    }


def chromaticity_variance(rgb_pixels: np.ndarray) -> float:
    luminance = np.clip(
        rgb_pixels[:, 0] * 0.2126 + rgb_pixels[:, 1] * 0.7152 + rgb_pixels[:, 2] * 0.0722,
        1e-6,
        None,
    )
    chroma = np.stack(
        [
            rgb_pixels[:, 0] / luminance,
            rgb_pixels[:, 1] / luminance,
            rgb_pixels[:, 2] / luminance,
        ],
        axis=1,
    )
    return float(np.mean(np.var(chroma, axis=0)))
