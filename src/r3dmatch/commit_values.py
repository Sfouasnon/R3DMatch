from __future__ import annotations

import math
from typing import Any, Dict, Iterable, Optional

import numpy as np


DEFAULT_AS_SHOT_KELVIN = 5600.0
DEFAULT_AS_SHOT_TINT = 0.0
KELVIN_MIN = 2000.0
KELVIN_MAX = 12000.0
TINT_MIN = -100.0
TINT_MAX = 100.0
KELVIN_LOG_SCALE = 0.9
TINT_SCALE = 100.0
KELVIN_SHARED_REGULARIZATION = 1.75
WB_MODEL_KEYS = (
    "per_camera_kelvin_per_camera_tint",
    "shared_kelvin_per_camera_tint",
    "shared_kelvin_shared_tint",
    "constrained_kelvin_per_camera_tint",
)
WB_MODEL_LABELS = {
    "per_camera_kelvin_per_camera_tint": "Per-Camera Kelvin / Per-Camera Tint",
    "shared_kelvin_per_camera_tint": "Shared Kelvin / Per-Camera Tint",
    "shared_kelvin_shared_tint": "Shared Kelvin / Shared Tint",
    "constrained_kelvin_per_camera_tint": "Constrained Kelvin / Per-Camera Tint",
}


def _clamp(value: float, lower: float, upper: float) -> float:
    return float(max(lower, min(upper, value)))


def _triplet(values: Optional[Iterable[float]]) -> list[float]:
    if values is None:
        return [1.0, 1.0, 1.0]
    resolved = [float(value) for value in values]
    if len(resolved) != 3:
        return [1.0, 1.0, 1.0]
    return resolved


def _find_nested_value(payload: Any, *keys: str) -> Optional[float]:
    stack = [payload]
    while stack:
        current = stack.pop()
        if not isinstance(current, dict):
            continue
        for key in keys:
            value = current.get(key)
            if value is not None:
                try:
                    return float(value)
                except (TypeError, ValueError):
                    return None
        nested = current.get("extra_metadata")
        if isinstance(nested, dict):
            stack.append(nested)
    return None


def extract_as_shot_white_balance(clip_metadata: Optional[Dict[str, object]]) -> Dict[str, float]:
    kelvin = _find_nested_value(clip_metadata, "white_balance_kelvin", "kelvin")
    tint = _find_nested_value(clip_metadata, "white_balance_tint", "tint")
    return {
        "kelvin": float(kelvin if kelvin is not None else DEFAULT_AS_SHOT_KELVIN),
        "tint": float(tint if tint is not None else DEFAULT_AS_SHOT_TINT),
    }


def neutral_balance_axes_from_rgb_gains(rgb_gains: Optional[Iterable[float]]) -> Dict[str, float]:
    red, green, blue = _triplet(rgb_gains)
    log_red = math.log(max(red, 1e-9))
    log_green = math.log(max(green, 1e-9))
    log_blue = math.log(max(blue, 1e-9))
    amber_blue_axis = log_red - log_blue
    green_magenta_axis = 0.5 * (log_red + log_blue) - log_green
    return {
        "amber_blue_axis": float(amber_blue_axis),
        "green_magenta_axis": float(green_magenta_axis),
    }


def rgb_gains_from_neutral_axes(amber_blue_axis: float, green_magenta_axis: float) -> list[float]:
    log_red = (green_magenta_axis / 3.0) + (amber_blue_axis / 2.0)
    log_green = (-2.0 * green_magenta_axis) / 3.0
    log_blue = (green_magenta_axis / 3.0) - (amber_blue_axis / 2.0)
    return [
        float(math.exp(log_red)),
        float(math.exp(log_green)),
        float(math.exp(log_blue)),
    ]


def _white_balance_from_axes(
    amber_blue_axis: float,
    green_magenta_axis: float,
    *,
    as_shot_kelvin: float,
    as_shot_tint: float,
) -> Dict[str, object]:
    kelvin = _clamp(
        as_shot_kelvin * math.exp(float(amber_blue_axis) * KELVIN_LOG_SCALE),
        KELVIN_MIN,
        KELVIN_MAX,
    )
    tint = _clamp(
        as_shot_tint + float(green_magenta_axis) * TINT_SCALE,
        TINT_MIN,
        TINT_MAX,
    )
    implied_amber_blue_axis = math.log(max(kelvin, 1e-6) / max(as_shot_kelvin, 1e-6)) / KELVIN_LOG_SCALE
    implied_green_magenta_axis = (tint - as_shot_tint) / TINT_SCALE
    implied_rgb_gains = rgb_gains_from_neutral_axes(implied_amber_blue_axis, implied_green_magenta_axis)
    return {
        "kelvin": int(round(kelvin)),
        "tint": round(float(tint), 1),
        "implied_amber_blue_axis": float(implied_amber_blue_axis),
        "implied_green_magenta_axis": float(implied_green_magenta_axis),
        "implied_rgb_gains": [round(float(value), 6) for value in implied_rgb_gains],
    }


def _desired_rgb_gains(
    *,
    measured_rgb_chromaticity: Optional[Iterable[float]],
    target_rgb_chromaticity: Optional[Iterable[float]],
    rgb_gains: Optional[Iterable[float]] = None,
) -> list[float]:
    measured = _triplet(measured_rgb_chromaticity)
    target = _triplet(target_rgb_chromaticity)
    if measured_rgb_chromaticity is None or target_rgb_chromaticity is None:
        return _triplet(rgb_gains)
    raw_gains = [
        float(target[index]) / max(float(measured[index]), 1e-6)
        for index in range(3)
    ]
    gain_norm = max((raw_gains[0] * raw_gains[1] * raw_gains[2]) ** (1.0 / 3.0), 1e-6)
    return [float(value / gain_norm) for value in raw_gains]


def _confidence_weight(
    confidence: Optional[float],
    sample_log2_spread: Optional[float],
    sample_chromaticity_spread: Optional[float],
) -> float:
    base = float(confidence) if confidence is not None else 1.0
    log2_factor = 1.0 - min(max(float(sample_log2_spread or 0.0), 0.0) / 0.35, 0.85)
    chroma_factor = 1.0 - min(max(float(sample_chromaticity_spread or 0.0), 0.0) / 0.03, 0.85)
    return max(0.05, min(1.0, base * log2_factor * chroma_factor))


def _wb_solution_from_request(
    request: Dict[str, object],
    *,
    amber_blue_axis: float,
    green_magenta_axis: float,
    method: str,
    model_key: str,
) -> Dict[str, object]:
    as_shot_kelvin = float(request["as_shot_kelvin"])
    as_shot_tint = float(request["as_shot_tint"])
    predicted = _white_balance_from_axes(
        amber_blue_axis,
        green_magenta_axis,
        as_shot_kelvin=as_shot_kelvin,
        as_shot_tint=as_shot_tint,
    )
    desired_ab = float(request["desired_axes"]["amber_blue_axis"])
    desired_gm = float(request["desired_axes"]["green_magenta_axis"])
    post_residual = math.sqrt(
        (desired_ab - float(predicted["implied_amber_blue_axis"])) ** 2
        + (desired_gm - float(predicted["implied_green_magenta_axis"])) ** 2
    )
    return {
        "kelvin": int(predicted["kelvin"]),
        "tint": float(predicted["tint"]),
        "method": method,
        "model_key": model_key,
        "model_label": WB_MODEL_LABELS[model_key],
        "as_shot_kelvin": int(round(as_shot_kelvin)),
        "as_shot_tint": round(as_shot_tint, 1),
        "white_balance_axes": {
            "amber_blue": round(desired_ab, 6),
            "green_magenta": round(desired_gm, 6),
        },
        "predicted_white_balance_axes": {
            "amber_blue": round(float(predicted["implied_amber_blue_axis"]), 6),
            "green_magenta": round(float(predicted["implied_green_magenta_axis"]), 6),
        },
        "implied_rgb_gains": list(predicted["implied_rgb_gains"]),
        "pre_neutral_residual": round(math.sqrt(desired_ab ** 2 + desired_gm ** 2), 6),
        "post_neutral_residual": round(float(post_residual), 6),
        "confidence_weight": round(float(request["weight"]), 6),
    }


def solve_white_balance_model(
    clip_requests: Iterable[Dict[str, object]],
    *,
    preferred_model: Optional[str] = None,
) -> Dict[str, object]:
    requests = [dict(item) for item in clip_requests]
    if not requests:
        return {
            "model_key": "per_camera_kelvin_per_camera_tint",
            "model_label": WB_MODEL_LABELS["per_camera_kelvin_per_camera_tint"],
            "shared_kelvin": None,
            "shared_tint": None,
            "candidates": [],
            "clips": {},
        }

    weights = np.array([float(item["weight"]) for item in requests], dtype=np.float32)
    if not float(np.sum(weights)):
        weights = np.ones(len(requests), dtype=np.float32)
    desired_ab = np.array([float(item["desired_axes"]["amber_blue_axis"]) for item in requests], dtype=np.float32)
    desired_gm = np.array([float(item["desired_axes"]["green_magenta_axis"]) for item in requests], dtype=np.float32)
    weighted_shared_ab = float(np.average(desired_ab, weights=weights))
    weighted_shared_gm = float(np.average(desired_gm, weights=weights))

    model_axes: Dict[str, tuple[list[float], list[float]]] = {}
    model_axes["per_camera_kelvin_per_camera_tint"] = (
        [float(value) for value in desired_ab],
        [float(value) for value in desired_gm],
    )
    model_axes["shared_kelvin_per_camera_tint"] = (
        [weighted_shared_ab for _ in requests],
        [float(value) for value in desired_gm],
    )
    model_axes["shared_kelvin_shared_tint"] = (
        [weighted_shared_ab for _ in requests],
        [weighted_shared_gm for _ in requests],
    )
    constrained_ab = []
    for desired_value, weight in zip(desired_ab, weights):
        deviation_keep = float(weight) / (float(weight) + KELVIN_SHARED_REGULARIZATION)
        constrained_ab.append(float(weighted_shared_ab + (float(desired_value) - weighted_shared_ab) * deviation_keep))
    model_axes["constrained_kelvin_per_camera_tint"] = (
        constrained_ab,
        [float(value) for value in desired_gm],
    )

    candidates = []
    for model_key in WB_MODEL_KEYS:
        predicted_abs, predicted_gms = model_axes[model_key]
        clip_solutions: Dict[str, Dict[str, object]] = {}
        post_residuals = []
        kelvin_values = []
        kelvin_axis_values = []
        kelvin_deviation = []
        for request, predicted_ab, predicted_gm in zip(requests, predicted_abs, predicted_gms):
            solution = _wb_solution_from_request(
                request,
                amber_blue_axis=float(predicted_ab),
                green_magenta_axis=float(predicted_gm),
                method=f"neutral_axis_{model_key}_v2",
                model_key=model_key,
            )
            clip_solutions[str(request["clip_id"])] = solution
            post_residuals.append(float(solution["post_neutral_residual"]))
            kelvin_values.append(float(solution["kelvin"]))
            kelvin_axis_values.append(float(solution["predicted_white_balance_axes"]["amber_blue"]))
            kelvin_deviation.append(
                abs(math.log(max(float(solution["kelvin"]), 1e-6) / max(float(solution["as_shot_kelvin"]), 1e-6)))
            )

        weighted_mean_post = float(np.average(np.asarray(post_residuals, dtype=np.float32), weights=weights))
        max_post = float(np.max(post_residuals)) if post_residuals else 0.0
        kelvin_axis_std = float(np.std(np.asarray(kelvin_axis_values, dtype=np.float32))) if kelvin_axis_values else 0.0
        mean_kelvin_deviation = float(np.mean(kelvin_deviation)) if kelvin_deviation else 0.0
        mean_confidence_penalty = float(np.mean([max(0.0, 1.0 - float(item["weight"])) for item in requests]))
        score = (
            weighted_mean_post * 8.0
            + max_post * 2.0
            + kelvin_axis_std * 14.0
            + mean_kelvin_deviation * 3.0
            + mean_confidence_penalty * 0.5
        )
        candidates.append(
            {
                "model_key": model_key,
                "model_label": WB_MODEL_LABELS[model_key],
                "score": round(float(score), 6),
                "metrics": {
                    "weighted_mean_post_neutral_residual": round(weighted_mean_post, 6),
                    "max_post_neutral_residual": round(max_post, 6),
                    "kelvin_axis_stddev": round(kelvin_axis_std, 6),
                    "mean_kelvin_log_deviation": round(mean_kelvin_deviation, 6),
                    "mean_confidence_penalty": round(mean_confidence_penalty, 6),
                },
                "shared_kelvin": int(round(np.median(kelvin_values))) if kelvin_values else None,
                "shared_tint": round(float(np.median([float(solution["tint"]) for solution in clip_solutions.values()])), 1)
                if clip_solutions
                else None,
                "clips": clip_solutions,
            }
        )

    if preferred_model:
        preferred_model = str(preferred_model)
    chosen = None
    if preferred_model:
        chosen = next((candidate for candidate in candidates if candidate["model_key"] == preferred_model), None)
    if chosen is None:
        candidates = sorted(candidates, key=lambda item: (float(item["score"]), WB_MODEL_KEYS.index(str(item["model_key"]))))
        chosen = candidates[0]
    return {
        "model_key": str(chosen["model_key"]),
        "model_label": str(chosen["model_label"]),
        "shared_kelvin": chosen["shared_kelvin"],
        "shared_tint": chosen["shared_tint"],
        "metrics": dict(chosen["metrics"]),
        "candidates": [
            {
                "model_key": str(candidate["model_key"]),
                "model_label": str(candidate["model_label"]),
                "score": float(candidate["score"]),
                "metrics": dict(candidate["metrics"]),
                "shared_kelvin": candidate["shared_kelvin"],
                "shared_tint": candidate["shared_tint"],
            }
            for candidate in candidates
        ],
        "clips": dict(chosen["clips"]),
    }


def solve_white_balance_model_for_records(
    clip_records: Iterable[Dict[str, object]],
    *,
    target_rgb_chromaticity: Iterable[float],
    preferred_model: Optional[str] = None,
) -> Dict[str, object]:
    requests = []
    target = _triplet(target_rgb_chromaticity)
    for record in clip_records:
        clip_id = str(record.get("clip_id"))
        diagnostic_gains = _desired_rgb_gains(
            measured_rgb_chromaticity=record.get("measured_rgb_chromaticity"),
            target_rgb_chromaticity=target,
            rgb_gains=record.get("rgb_gains"),
        )
        axes = neutral_balance_axes_from_rgb_gains(diagnostic_gains)
        as_shot = extract_as_shot_white_balance(record.get("clip_metadata"))
        requests.append(
            {
                "clip_id": clip_id,
                "desired_axes": axes,
                "as_shot_kelvin": as_shot["kelvin"],
                "as_shot_tint": as_shot["tint"],
                "weight": _confidence_weight(
                    record.get("confidence"),
                    record.get("sample_log2_spread"),
                    record.get("sample_chromaticity_spread"),
                ),
            }
        )
    return solve_white_balance_model(requests, preferred_model=preferred_model)


def solve_kelvin_tint_from_chromaticity(
    *,
    measured_rgb_chromaticity: Optional[Iterable[float]],
    target_rgb_chromaticity: Optional[Iterable[float]],
    clip_metadata: Optional[Dict[str, object]] = None,
    rgb_gains: Optional[Iterable[float]] = None,
) -> Dict[str, object]:
    diagnostic_rgb_gains = _desired_rgb_gains(
        measured_rgb_chromaticity=measured_rgb_chromaticity,
        target_rgb_chromaticity=target_rgb_chromaticity,
        rgb_gains=rgb_gains,
    )
    axes = neutral_balance_axes_from_rgb_gains(diagnostic_rgb_gains)
    as_shot = extract_as_shot_white_balance(clip_metadata)
    predicted = _white_balance_from_axes(
        float(axes["amber_blue_axis"]),
        float(axes["green_magenta_axis"]),
        as_shot_kelvin=as_shot["kelvin"],
        as_shot_tint=as_shot["tint"],
    )
    pre_residual = math.sqrt(
        float(axes["amber_blue_axis"]) ** 2 + float(axes["green_magenta_axis"]) ** 2
    )
    post_residual = math.sqrt(
        (float(axes["amber_blue_axis"]) - float(predicted["implied_amber_blue_axis"])) ** 2
        + (float(axes["green_magenta_axis"]) - float(predicted["implied_green_magenta_axis"])) ** 2
    )
    return {
        "kelvin": int(predicted["kelvin"]),
        "tint": float(predicted["tint"]),
        "method": "neutral_axis_kelvin_tint_v1",
        "as_shot_kelvin": int(round(as_shot["kelvin"])),
        "as_shot_tint": round(float(as_shot["tint"]), 1),
        "white_balance_axes": {
            "amber_blue": round(float(axes["amber_blue_axis"]), 6),
            "green_magenta": round(float(axes["green_magenta_axis"]), 6),
        },
        "implied_rgb_gains": list(predicted["implied_rgb_gains"]),
        "pre_neutral_residual": round(float(pre_residual), 6),
        "post_neutral_residual": round(float(post_residual), 6),
    }


def derive_kelvin_tint_from_rgb_gains(rgb_gains: Optional[Iterable[float]]) -> Dict[str, object]:
    solved = solve_kelvin_tint_from_chromaticity(
        measured_rgb_chromaticity=None,
        target_rgb_chromaticity=None,
        rgb_gains=rgb_gains,
    )
    return {
        "kelvin": int(solved["kelvin"]),
        "tint": float(solved["tint"]),
        "method": str(solved["method"]),
    }


def build_commit_values(
    *,
    exposure_adjust: float,
    rgb_gains: Optional[Iterable[float]],
    confidence: Optional[float] = None,
    sample_log2_spread: Optional[float] = None,
    sample_chromaticity_spread: Optional[float] = None,
    measured_rgb_chromaticity: Optional[Iterable[float]] = None,
    target_rgb_chromaticity: Optional[Iterable[float]] = None,
    clip_metadata: Optional[Dict[str, object]] = None,
    saturation: Optional[float] = None,
    saturation_supported: bool = False,
    wb_solution: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    wb = dict(
        wb_solution
        or solve_kelvin_tint_from_chromaticity(
            measured_rgb_chromaticity=measured_rgb_chromaticity,
            target_rgb_chromaticity=target_rgb_chromaticity,
            clip_metadata=clip_metadata,
            rgb_gains=rgb_gains,
        )
    )
    notes = []
    if confidence is not None and float(confidence) < 0.75:
        notes.append("Lower-confidence measurement; review before committing.")
    if sample_log2_spread is not None and float(sample_log2_spread) > 0.12:
        notes.append("Three-sample exposure spread is elevated.")
    if sample_chromaticity_spread is not None and float(sample_chromaticity_spread) > 0.02:
        notes.append("Three-sample neutral chromaticity spread is elevated.")
    if not saturation_supported:
        notes.append("Saturation left neutral because a gray target does not support a trustworthy saturation solve.")
    return {
        "exposureAdjust": round(float(exposure_adjust), 6),
        "kelvin": int(wb["kelvin"]),
        "tint": float(wb["tint"]),
        "derivation_method": str(wb["method"]),
        "as_shot_kelvin": int(wb["as_shot_kelvin"]),
        "as_shot_tint": float(wb["as_shot_tint"]),
        "white_balance_axes": dict(wb["white_balance_axes"]),
        "implied_rgb_gains": list(wb["implied_rgb_gains"]),
        "pre_neutral_residual": float(wb["pre_neutral_residual"]),
        "post_neutral_residual": float(wb["post_neutral_residual"]),
        "white_balance_model": wb.get("model_key"),
        "white_balance_model_label": wb.get("model_label"),
        "white_balance_confidence_weight": float(wb.get("confidence_weight", 1.0)),
        "saturation": round(float(saturation if saturation is not None else 1.0), 6),
        "saturation_supported": bool(saturation_supported),
        "notes": notes,
    }
