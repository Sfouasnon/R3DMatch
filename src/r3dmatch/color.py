from __future__ import annotations

from typing import Dict, Optional

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


def identity_lggs() -> Dict[str, object]:
    return {
        "lift": [0.0, 0.0, 0.0],
        "gamma": [1.0, 1.0, 1.0],
        "gain": [1.0, 1.0, 1.0],
        "saturation": 1.0,
    }


def lggs_to_cdl(color_model: Dict[str, object]) -> Dict[str, object]:
    gain = [float(value) for value in color_model.get("gain", [1.0, 1.0, 1.0])]
    lift = [float(value) for value in color_model.get("lift", [0.0, 0.0, 0.0])]
    gamma = [max(float(value), 1e-6) for value in color_model.get("gamma", [1.0, 1.0, 1.0])]
    saturation = float(color_model.get("saturation", 1.0))
    return {
        "slope": gain,
        "offset": lift,
        "power": [float(1.0 / value) for value in gamma],
        "saturation": saturation,
    }


def is_identity_cdl_payload(color_cdl: Optional[Dict[str, object]]) -> bool:
    if color_cdl is None:
        return True
    slope = [float(value) for value in color_cdl.get("slope", [1.0, 1.0, 1.0])]
    offset = [float(value) for value in color_cdl.get("offset", [0.0, 0.0, 0.0])]
    power = [float(value) for value in color_cdl.get("power", [1.0, 1.0, 1.0])]
    saturation = float(color_cdl.get("saturation", 1.0))
    return (
        all(abs(value - 1.0) <= 1e-9 for value in slope)
        and all(abs(value) <= 1e-9 for value in offset)
        and all(abs(value - 1.0) <= 1e-9 for value in power)
        and abs(saturation - 1.0) <= 1e-9
    )


def solve_cdl_color_model(
    *,
    measured_rgb_chromaticity: list[float] | tuple[float, float, float],
    target_rgb_chromaticity: list[float] | tuple[float, float, float],
    measured_saturation_fraction: Optional[float] = None,
    target_saturation_fraction: Optional[float] = None,
) -> Dict[str, object]:
    raw_gains = [
        float(target_rgb_chromaticity[0]) / max(float(measured_rgb_chromaticity[0]), 1e-6),
        float(target_rgb_chromaticity[1]) / max(float(measured_rgb_chromaticity[1]), 1e-6),
        float(target_rgb_chromaticity[2]) / max(float(measured_rgb_chromaticity[2]), 1e-6),
    ]
    gain_norm = max(float(np.prod(np.asarray(raw_gains, dtype=np.float32)) ** (1.0 / 3.0)), 1e-6)
    diagnostic_rgb_gains = [float(value / gain_norm) for value in raw_gains]
    if measured_saturation_fraction is None or target_saturation_fraction is None:
        saturation = 1.0
        saturation_source = "identity_fallback"
    else:
        measured_sat = max(float(measured_saturation_fraction), 0.0)
        target_sat = max(float(target_saturation_fraction), 0.0)
        saturation = float(np.clip((target_sat + 0.02) / (measured_sat + 0.02), 0.75, 1.25))
        saturation_source = "roi_saturation_fraction"
    color_model = {
        "lift": [0.0, 0.0, 0.0],
        "gamma": [1.0, 1.0, 1.0],
        "gain": diagnostic_rgb_gains,
        "saturation": saturation,
    }
    return {
        "color_model": color_model,
        "cdl": lggs_to_cdl(color_model),
        "diagnostic_rgb_gains": diagnostic_rgb_gains,
        "saturation": saturation,
        "saturation_source": saturation_source,
    }


def rgb_gains_to_cdl(rgb_gains: Dict[str, float] | list[float] | tuple[float, float, float]) -> Dict[str, object]:
    if isinstance(rgb_gains, dict):
        gain = [float(rgb_gains["r"]), float(rgb_gains["g"]), float(rgb_gains["b"])]
    else:
        gain = [float(rgb_gains[0]), float(rgb_gains[1]), float(rgb_gains[2])]
    return lggs_to_cdl(
        {
            "lift": [0.0, 0.0, 0.0],
            "gamma": [1.0, 1.0, 1.0],
            "gain": gain,
            "saturation": 1.0,
        }
    )
