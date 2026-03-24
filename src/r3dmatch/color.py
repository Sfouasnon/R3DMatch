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


def rgb_gains_to_cdl(rgb_gains: Dict[str, float] | list[float] | tuple[float, float, float]) -> Dict[str, object]:
    if isinstance(rgb_gains, dict):
        slope = [float(rgb_gains["r"]), float(rgb_gains["g"]), float(rgb_gains["b"])]
    else:
        slope = [float(rgb_gains[0]), float(rgb_gains[1]), float(rgb_gains[2])]
    return {
        "slope": slope,
        "offset": [0.0, 0.0, 0.0],
        "power": [1.0, 1.0, 1.0],
        "saturation": 1.0,
    }
