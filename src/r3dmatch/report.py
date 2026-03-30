from __future__ import annotations

import copy
import html
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import time
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw
from skimage import color, feature, transform

from .calibration import (
    extract_center_region,
    load_color_calibration,
    load_exposure_calibration,
    measure_sphere_region_statistics,
    measure_sphere_zone_profile_statistics,
    percentile_clip,
)
from .color import identity_lggs, is_identity_cdl_payload as _color_is_identity_cdl_payload, rgb_gains_to_cdl, solve_cdl_color_model
from .commit_values import build_commit_values, extract_as_shot_white_balance, solve_white_balance_model_for_records
from .execution import CancellationError, raise_if_cancelled, run_cancellable_subprocess
from .ftps_ingest import source_mode_label
from .matching import _measure_three_sample_statistics
from .models import SphereROI
from .progress import emit_review_progress
from .rmd import write_rmd_for_clip_with_metadata, write_rmds_from_analysis


PREVIEW_VARIANTS = ("original", "exposure", "color", "both")
REVIEW_PREVIEW_TRANSFORM = "REDLine IPP2 / BT.709 / BT.1886 / Medium / Medium"
DEFAULT_REVIEW_TARGET_STRATEGIES = ("median",)
STRATEGY_ORDER = ["median", "optimal_exposure", "manual", "hero_camera", "manual_target"]
# Legacy "calibration" preview mode is now a compatibility alias to the
# canonical monitoring-domain preview path. We preserve the name only so older
# CLI/UI inputs do not break.
DEFAULT_CALIBRATION_PREVIEW = {
    "preview_mode": "monitoring",
    "output_space": "BT.709",
    "output_gamma": "BT.1886",
    "highlight_rolloff": "medium",
    "shadow_rolloff": "medium",
    "lut_path": None,
}
DEFAULT_MONITORING_PREVIEW = {
    "preview_mode": "monitoring",
    "output_space": "BT.709",
    "output_gamma": "BT.1886",
    "highlight_rolloff": "medium",
    "shadow_rolloff": "medium",
    "lut_path": None,
}
DEFAULT_DISPLAY_REVIEW_PREVIEW = {
    "preview_mode": "monitoring",
    "output_space": "BT.709",
    "output_gamma": "BT.1886",
    "highlight_rolloff": "medium",
    "shadow_rolloff": "medium",
    "lut_path": None,
}
COLOR_SPACE_CODES = {"BT.709": 13, "REDWideGamutRGB": 25}
GAMMA_CODES = {"BT.1886": 32, "Log3G10": 34}
ROLLOFF_CODES = {"none": 0, "hard": 1, "default": 2, "medium": 3, "soft": 4}
TONEMAP_CODES = {"low": 0, "medium": 1, "high": 2, "none": 3}
SHADOW_ROLLOFF_VALUES = {"hard": -0.25, "medium": 0.0, "soft": 0.25}
LOGO_PATH = Path(__file__).resolve().parent / "static" / "r3dmatch_logo.png"
REDLINE_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "redline.json"
COLOR_PREVIEW_OVERRIDE_ENV = "R3DMATCH_ENABLE_UNVERIFIED_COLOR_PREVIEW"
REVIEW_MODE_LABELS = {
    "full_contact_sheet": "Full Contact Sheet",
    "lightweight_analysis": "Lightweight Analysis",
}
OPTIMAL_EXPOSURE_MIN_CONFIDENCE = 0.45
OPTIMAL_EXPOSURE_MAX_SAMPLE_LOG2_SPREAD = 0.12
OPTIMAL_EXPOSURE_MAX_SAMPLE_CHROMA_SPREAD = 0.02
OPTIMAL_EXPOSURE_PRIMARY_CLUSTER_GAP = 0.25
CAMERA_TRUST_CAUTION_CONFIDENCE = 0.65
CAMERA_TRUST_CAUTION_LOG2_SPREAD = 0.08
SPHERE_DETECTION_HIGH_CONFIDENCE = 0.72
SPHERE_DETECTION_MEDIUM_CONFIDENCE = 0.50
SPHERE_DETECTION_LOW_CONFIDENCE = 0.30
CAMERA_TRUST_CAUTION_CHROMA_SPREAD = 0.014
CAMERA_TRUST_LARGE_CORRECTION = 0.75
CAMERA_TRUST_EXTREME_CORRECTION = 1.0
CORRECTED_RESIDUAL_PASS_STOPS = 0.10
CORRECTED_RESIDUAL_REVIEW_STOPS = 0.20
IPP2_VALIDATION_PASS_STOPS = 0.05
IPP2_VALIDATION_REVIEW_STOPS = 0.10
IPP2_CLOSED_LOOP_MAX_ITERATIONS = 10
IPP2_CLOSED_LOOP_MAX_CORRECTION_STOPS = 3.0
IPP2_CLOSED_LOOP_MIN_STEP_STOPS = 0.01
IPP2_CLOSED_LOOP_DAMPING = 0.85
IPP2_CLOSED_LOOP_MIN_IMPROVEMENT_STOPS = 0.002
VISIBLE_PREVIEW_EXPOSURE_DELTA_STOPS = 0.02
SAFE_CORRECTION_STOPS = 2.0
WARNING_CORRECTION_STOPS = 3.0
REDLINE_DIRECT_EXPOSURE_PARAMETER = "exposureAdjust"
SPHERE_PROFILE_ZONE_ORDER = ("bright_side", "center", "dark_side")
SPHERE_PROFILE_ZONE_DISPLAY = {"bright_side": "Sample 1", "center": "Sample 2", "dark_side": "Sample 3"}
SPHERE_PROFILE_ZONE_WEIGHTS = {"bright_side": 0.3, "center": 0.5, "dark_side": 0.2}
PROFILE_AUDIT_CONSISTENT_STOPS = 0.10
PROFILE_AUDIT_REVIEW_STOPS = 0.25
PROFILE_AUDIT_CONSISTENT_SPREAD_IRE = 1.5
PROFILE_AUDIT_REVIEW_SPREAD_IRE = 3.0
SPHERE_PROFILE_SAMPLE_FIELD_MAP = {
    "bright_side": "sample_1_ire",
    "center": "sample_2_ire",
    "dark_side": "sample_3_ire",
}


def normalize_review_mode(value: str) -> str:
    normalized = str(value).strip().lower()
    aliases = {
        "full": "full_contact_sheet",
        "full_contact_sheet": "full_contact_sheet",
        "contact_sheet": "full_contact_sheet",
        "full-contact-sheet": "full_contact_sheet",
        "lightweight": "lightweight_analysis",
        "lightweight_analysis": "lightweight_analysis",
        "lightweight-analysis": "lightweight_analysis",
        "analysis": "lightweight_analysis",
    }
    if normalized not in aliases:
        raise ValueError("review mode must be full_contact_sheet or lightweight_analysis")
    return aliases[normalized]


def review_mode_label(value: str) -> str:
    return REVIEW_MODE_LABELS[normalize_review_mode(value)]


def _normalize_matching_domain(value: str) -> str:
    normalized = str(value).strip().lower()
    aliases = {
        "scene": "scene",
        "scene-referred": "scene",
        "scenereferred": "scene",
        "perceptual": "perceptual",
        "monitoring": "perceptual",
        "view": "perceptual",
    }
    if normalized not in aliases:
        raise ValueError("matching domain must be scene or perceptual")
    return aliases[normalized]


def _matching_domain_label(value: str) -> str:
    normalized = _normalize_matching_domain(value)
    if normalized == "scene":
        return "Scene-Referred (REDWideGamutRGB / Log3G10)"
    return "Perceptual (IPP2 / BT.709 / BT.1886)"


def _target_supports_saturation(target_type: Optional[str]) -> bool:
    if target_type is None:
        return True
    normalized = str(target_type).strip().lower().replace("-", "_")
    return normalized not in {"gray_sphere", "gray_card", "neutral_patch"}


def _measurement_values_for_record(
    record: Dict[str, object],
    *,
    matching_domain: str,
    monitoring_measurements_by_clip: Optional[Dict[str, Dict[str, object]]] = None,
) -> Dict[str, object]:
    diagnostics = record.get("diagnostics", {})
    clip_id = str(record["clip_id"])
    monitoring = (monitoring_measurements_by_clip or {}).get(clip_id, {})
    resolved_domain = _normalize_matching_domain(matching_domain)
    diagnostics_are_rendered_ipp2 = str(diagnostics.get("exposure_measurement_domain") or "").strip() == "rendered_preview_ipp2"
    if resolved_domain == "perceptual":
        if not monitoring and diagnostics_are_rendered_ipp2:
            monitoring = {
                "measured_log2_luminance_monitoring": diagnostics.get("measured_log2_luminance_monitoring", diagnostics.get("measured_log2_luminance", 0.0)),
                "measured_rgb_chromaticity_monitoring": diagnostics.get("measured_rgb_chromaticity", [1 / 3, 1 / 3, 1 / 3]),
                "measured_saturation_fraction_monitoring": diagnostics.get("saturation_fraction", 0.0),
                "gray_exposure_summary": diagnostics.get("gray_exposure_summary", diagnostics.get("aggregate_sphere_profile", "n/a")),
                "zone_measurements": diagnostics.get("zone_measurements", diagnostics.get("neutral_samples", [])),
                "neutral_sample_log2_spread": diagnostics.get("neutral_sample_log2_spread", 0.0),
                "neutral_sample_chromaticity_spread": diagnostics.get("neutral_sample_chromaticity_spread", 0.0),
                "sphere_detection_confidence": diagnostics.get("sphere_detection_confidence", 0.0),
                "sphere_detection_label": diagnostics.get("sphere_detection_label", ""),
                "detection_failed": diagnostics.get("detection_failed", False),
            }
        zone_measurements = [dict(item) for item in list(monitoring.get("zone_measurements") or [])]
        if not zone_measurements:
            zone_measurements = [dict(item) for item in list(diagnostics.get("zone_measurements") or diagnostics.get("neutral_samples") or [])]
        sample_display_scalar = _sample_scalar_display_from_profile(zone_measurements)
        resolved_log2 = (
            float(sample_display_scalar.get("sample_scalar_display_log2", 0.0) or 0.0)
            if float(sample_display_scalar.get("sample_count", 0.0) or 0.0) > 0.0
            else float(
                monitoring.get(
                    "measured_log2_luminance_monitoring",
                    diagnostics.get("measured_log2_luminance_monitoring", diagnostics.get("measured_log2_luminance", 0.0)),
                )
                or 0.0
            )
        )
        resolved_ire = (
            float(sample_display_scalar.get("sample_scalar_display_ire", 0.0) or 0.0)
            if float(sample_display_scalar.get("sample_count", 0.0) or 0.0) > 0.0
            else float(_ire_from_log2_luminance(resolved_log2))
        )
        return {
            "display_scalar_log2": resolved_log2,
            "display_scalar_ire": resolved_ire,
            "log2_luminance": resolved_log2,
            "rgb_chromaticity": [
                float(value)
                for value in monitoring.get(
                    "measured_rgb_chromaticity_monitoring",
                    diagnostics.get("measured_rgb_chromaticity", [1 / 3, 1 / 3, 1 / 3]),
                )
            ],
            "saturation_fraction": float(
                monitoring.get(
                    "measured_saturation_fraction_monitoring",
                    diagnostics.get("saturation_fraction", 0.0) or 0.0,
                )
            ),
            "gray_exposure_summary": str(
                monitoring.get(
                    "gray_exposure_summary",
                    diagnostics.get("gray_exposure_summary", diagnostics.get("aggregate_sphere_profile", "n/a")),
                )
                or "n/a"
            ),
            "zone_measurements": zone_measurements,
            "neutral_sample_log2_spread": float(
                monitoring.get(
                    "neutral_sample_log2_spread",
                    diagnostics.get("neutral_sample_log2_spread", 0.0) or 0.0,
                )
            ),
            "neutral_sample_chromaticity_spread": float(
                monitoring.get(
                    "neutral_sample_chromaticity_spread",
                    diagnostics.get("neutral_sample_chromaticity_spread", 0.0) or 0.0,
                )
            ),
            "sphere_detection_confidence": float(monitoring.get("sphere_detection_confidence", 0.0) or 0.0),
            "sphere_detection_label": str(monitoring.get("sphere_detection_label") or ""),
            "detection_failed": bool(monitoring.get("detection_failed")),
            "source": "rendered_preview_ipp2" if monitoring else "analysis_diagnostic",
        }
    return {
        "log2_luminance": float(
            diagnostics.get(
                "measured_log2_luminance_raw",
                diagnostics.get("measured_log2_luminance", 0.0),
            )
        ),
        "rgb_chromaticity": [
            float(value)
            for value in diagnostics.get("measured_rgb_chromaticity", [1 / 3, 1 / 3, 1 / 3])
        ],
        "saturation_fraction": float(
            diagnostics.get("raw_saturation_fraction", diagnostics.get("saturation_fraction", 0.0) or 0.0)
        ),
        "gray_exposure_summary": str(diagnostics.get("gray_exposure_summary", diagnostics.get("aggregate_sphere_profile", "n/a")) or "n/a"),
        "zone_measurements": [dict(item) for item in list(diagnostics.get("neutral_samples") or [])],
        "neutral_sample_log2_spread": float(diagnostics.get("neutral_sample_log2_spread", 0.0) or 0.0),
        "neutral_sample_chromaticity_spread": float(diagnostics.get("neutral_sample_chromaticity_spread", 0.0) or 0.0),
        "sphere_detection_confidence": 0.0,
        "sphere_detection_label": "",
        "detection_failed": False,
        "source": "scene_referred_analysis",
    }


def _sample_ire_fields_from_profile(zone_measurements: List[Dict[str, object]]) -> Dict[str, float]:
    zone_map = _zone_profile_by_label(zone_measurements)
    result: Dict[str, float] = {}
    for label in SPHERE_PROFILE_ZONE_ORDER:
        field_name = SPHERE_PROFILE_SAMPLE_FIELD_MAP[label]
        zone = zone_map.get(label)
        if zone is None:
            result[field_name] = 0.0
            continue
        result[field_name] = float(
            zone.get(
                "measured_ire",
                _ire_from_log2_luminance(float(zone.get("measured_log2_luminance", 0.0) or 0.0)),
            )
            or 0.0
        )
    return result


def _sample_scalar_display_from_profile(zone_measurements: List[Dict[str, object]]) -> Dict[str, float]:
    ordered = _ordered_zone_profile(zone_measurements)
    if not ordered:
        return {
            "sample_scalar_display_log2": 0.0,
            "sample_scalar_display_ire": 0.0,
            "sample_count": 0.0,
            "scalar_domain": "display_ipp2",
        }
    log2_values = np.asarray(
        [float(item.get("measured_log2_luminance", 0.0) or 0.0) for item in ordered],
        dtype=np.float32,
    )
    ire_values = np.asarray(
        [
            float(
                item.get(
                    "measured_ire",
                    _ire_from_log2_luminance(float(item.get("measured_log2_luminance", 0.0) or 0.0)),
                )
                or 0.0
            )
            for item in ordered
        ],
        dtype=np.float32,
    )
    return {
        "sample_scalar_display_log2": float(np.median(log2_values)) if log2_values.size else 0.0,
        "sample_scalar_display_ire": float(np.median(ire_values)) if ire_values.size else 0.0,
        "sample_count": float(log2_values.size),
        "scalar_domain": "display_ipp2",
    }


def _rendered_measurement_from_diagnostics(record: Dict[str, object]) -> Dict[str, object]:
    diagnostics = dict(record.get("diagnostics", {}) or {})
    zone_measurements = [dict(item) for item in list(diagnostics.get("zone_measurements") or diagnostics.get("neutral_samples") or [])]
    sample_ire_fields = _sample_ire_fields_from_profile(zone_measurements)
    sample_display_scalar = _sample_scalar_display_from_profile(zone_measurements)
    resolved_log2 = (
        float(sample_display_scalar.get("sample_scalar_display_log2", 0.0) or 0.0)
        if float(sample_display_scalar.get("sample_count", 0.0) or 0.0) > 0.0
        else float(diagnostics.get("measured_log2_luminance_monitoring", diagnostics.get("measured_log2_luminance", 0.0)) or 0.0)
    )
    return {
        "sample_scalar_display_log2": resolved_log2,
        "sample_scalar_display_ire": (
            float(sample_display_scalar.get("sample_scalar_display_ire", 0.0) or 0.0)
            if float(sample_display_scalar.get("sample_count", 0.0) or 0.0) > 0.0
            else float(_ire_from_log2_luminance(resolved_log2))
        ),
        "sample_scalar_display_domain": "display_ipp2",
        "measured_log2_luminance_monitoring": resolved_log2,
        "measured_rgb_mean": [float(value) for value in diagnostics.get("measured_rgb_mean", [0.0, 0.0, 0.0])],
        "measured_rgb_chromaticity_monitoring": [float(value) for value in diagnostics.get("measured_rgb_chromaticity", [1 / 3, 1 / 3, 1 / 3])],
        "measured_saturation_fraction_monitoring": float(diagnostics.get("saturation_fraction", 0.0) or 0.0),
        "gray_exposure_summary": str(diagnostics.get("gray_exposure_summary") or diagnostics.get("aggregate_sphere_profile") or "n/a"),
        "zone_measurements": zone_measurements,
        "neutral_sample_log2_spread": float(diagnostics.get("neutral_sample_log2_spread", 0.0) or 0.0),
        "neutral_sample_chromaticity_spread": float(diagnostics.get("neutral_sample_chromaticity_spread", 0.0) or 0.0),
        "sphere_detection_confidence": float(diagnostics.get("sphere_detection_confidence", 0.0) or 0.0),
        "sphere_detection_label": str(diagnostics.get("sphere_detection_label") or ""),
        "sphere_roi_source": str(diagnostics.get("sphere_detection_source") or diagnostics.get("sphere_roi_source") or ""),
        "sphere_detection_details": dict(diagnostics.get("sphere_detection_details") or {}),
        "detected_sphere_roi": dict(diagnostics.get("detected_sphere_roi") or {}),
        "measurement_crop_bounds": dict(diagnostics.get("measurement_crop_bounds") or {}),
        "measurement_crop_size": dict(diagnostics.get("measurement_crop_size") or {}),
        "detection_failed": bool(diagnostics.get("detection_failed")),
        "dominant_gradient_axis": dict(diagnostics.get("dominant_gradient_axis") or {}),
        "rendered_preview_path": str(diagnostics.get("rendered_measurement_preview_path") or ""),
        **sample_ire_fields,
    }


def _monitoring_quality_for_measurement(measurement: Dict[str, object]) -> Dict[str, object]:
    if not measurement:
        return {}
    detection_failed = bool(measurement.get("detection_failed"))
    detection_confidence = float(measurement.get("sphere_detection_confidence", 0.0) or 0.0)
    has_sphere_profile = bool(list(measurement.get("zone_measurements") or []))
    sample_log2_spread = 0.0 if has_sphere_profile else float(measurement.get("neutral_sample_log2_spread", 0.0) or 0.0)
    sample_chroma_spread = 0.0 if has_sphere_profile else float(measurement.get("neutral_sample_chromaticity_spread", 0.0) or 0.0)
    spread_penalty = min(sample_log2_spread / max(OPTIMAL_EXPOSURE_MAX_SAMPLE_LOG2_SPREAD, 1e-6), 2.0) * 0.35
    chroma_penalty = min(sample_chroma_spread / max(OPTIMAL_EXPOSURE_MAX_SAMPLE_CHROMA_SPREAD, 1e-6), 2.0) * 0.15
    confidence = 0.0 if detection_failed else max(0.0, min(1.0, detection_confidence * (1.0 - spread_penalty - chroma_penalty)))
    flags: List[str] = []
    if detection_failed:
        flags.append("sphere_detection_failed")
    if sample_log2_spread > OPTIMAL_EXPOSURE_MAX_SAMPLE_LOG2_SPREAD:
        flags.append("neutral_sample_exposure_spread_high")
    if sample_chroma_spread > OPTIMAL_EXPOSURE_MAX_SAMPLE_CHROMA_SPREAD:
        flags.append("neutral_sample_chromaticity_spread_high")
    return {
        "confidence": confidence,
        "neutral_sample_log2_spread": sample_log2_spread,
        "neutral_sample_chromaticity_spread": sample_chroma_spread,
        "flags": flags,
        "sphere_detection_confidence": detection_confidence,
        "sphere_detection_label": str(measurement.get("sphere_detection_label") or ""),
    }


def _measurement_preview_settings_for_domain(matching_domain: str) -> Dict[str, object]:
    _normalize_matching_domain(matching_domain)
    defaults = DEFAULT_DISPLAY_REVIEW_PREVIEW
    return _normalize_preview_settings(
        preview_mode=str(defaults["preview_mode"]),
        preview_output_space=str(defaults["output_space"]),
        preview_output_gamma=str(defaults["output_gamma"]),
        preview_highlight_rolloff=str(defaults["highlight_rolloff"]),
        preview_shadow_rolloff=str(defaults["shadow_rolloff"]),
        preview_lut=None,
    )


def _log2_from_ire(ire_value: float) -> float:
    return float(math.log2(max(float(ire_value), 1e-6) / 100.0))


def normalize_exposure_anchor_mode(value: Optional[str]) -> str:
    normalized = str(value or "").strip().lower().replace("-", "_")
    aliases = {
        "": "auto",
        "auto": "auto",
        "median": "median",
        "hero_camera": "hero_camera",
        "hero_clip": "hero_clip",
        "manual_clip": "manual_clip",
        "manual_target": "manual_target",
    }
    if normalized not in aliases:
        raise ValueError(
            "exposure anchor mode must be one of: median, hero-camera, hero-clip, manual-clip, manual-target"
        )
    return aliases[normalized]


def _strategy_key_for_anchor_mode(anchor_mode: str) -> str:
    normalized = normalize_exposure_anchor_mode(anchor_mode)
    if normalized == "hero_clip":
        return "hero_camera"
    if normalized == "manual_clip":
        return "manual"
    if normalized in {"median", "hero_camera", "manual_target"}:
        return normalized
    return ""


def _anchor_mode_label(anchor_mode: str) -> str:
    normalized = normalize_exposure_anchor_mode(anchor_mode)
    return {
        "auto": "Automatic recommendation",
        "median": "Median of group",
        "hero_camera": "Hero camera",
        "hero_clip": "Hero clip",
        "manual_clip": "Manual clip",
        "manual_target": "Manual target",
    }[normalized]


def _manual_anchor_target_log2(
    *,
    manual_target_stops: Optional[float],
    manual_target_ire: Optional[float],
) -> tuple[Optional[float], Optional[str], Optional[float]]:
    if manual_target_stops is not None:
        return float(manual_target_stops), "stops", float(manual_target_stops)
    if manual_target_ire is not None:
        return _log2_from_ire(float(manual_target_ire)), "ire", float(manual_target_ire)
    return None, None, None


def _format_manual_anchor_ire_summary(
    *,
    anchor_log2: Optional[float],
    manual_target_ire: Optional[float],
) -> str:
    if manual_target_ire is not None:
        return f"{float(manual_target_ire):.0f} IRE (manual scalar target)"
    if anchor_log2 is None:
        return "n/a"
    return f"{_ire_from_log2_luminance(float(anchor_log2)):.0f} IRE (manual scalar target)"


def _anchor_summary_line(
    *,
    anchor_mode: str,
    anchor_source: str,
    anchor_scalar_value: Optional[float],
    anchor_ire_summary: Optional[str],
    manual_target_input_domain: Optional[str] = None,
    manual_target_input_value: Optional[float] = None,
) -> str:
    normalized = normalize_exposure_anchor_mode(anchor_mode)
    if normalized == "median":
        return "Exposure Anchor: Median of group"
    if normalized == "hero_camera":
        return f"Exposure Anchor: Hero camera {anchor_source or 'Unspecified'}"
    if normalized == "hero_clip":
        return f"Exposure Anchor: Hero clip {anchor_source or 'Unspecified'}"
    if normalized == "manual_clip":
        return f"Exposure Anchor: Manual clip {anchor_source or 'Unspecified'}"
    if normalized == "manual_target":
        if manual_target_input_domain == "ire" and manual_target_input_value is not None:
            return (
                f"Exposure Anchor: Manual target {float(manual_target_input_value):.0f} IRE "
                f"(converted to {float(anchor_scalar_value or 0.0):+.3f} stops)"
            )
        if manual_target_input_domain == "stops" and manual_target_input_value is not None:
            suffix = f" | {anchor_ire_summary}" if anchor_ire_summary else ""
            return f"Exposure Anchor: Manual target {float(manual_target_input_value):+.3f} stops{suffix}"
        return "Exposure Anchor: Manual target"
    if anchor_source:
        return f"Exposure Anchor: {anchor_source}"
    return "Exposure Anchor: Automatic recommendation"


def _build_anchor_descriptor(
    *,
    strategy_key: str,
    anchor_mode: str,
    anchor_source: str,
    anchor_scalar_value: float,
    anchor_ire_summary: str,
    manual_target_input_domain: Optional[str] = None,
    manual_target_input_value: Optional[float] = None,
) -> Dict[str, object]:
    return {
        "anchor_mode": anchor_mode,
        "anchor_mode_label": _anchor_mode_label(anchor_mode),
        "anchor_source": anchor_source,
        "anchor_scalar_value": float(anchor_scalar_value),
        "anchor_ire_summary": str(anchor_ire_summary or "n/a"),
        "manual_target_input_domain": manual_target_input_domain,
        "manual_target_input_value": manual_target_input_value,
        "anchor_summary": _anchor_summary_line(
            anchor_mode=anchor_mode,
            anchor_source=anchor_source,
            anchor_scalar_value=anchor_scalar_value,
            anchor_ire_summary=anchor_ire_summary,
            manual_target_input_domain=manual_target_input_domain,
            manual_target_input_value=manual_target_input_value,
        ),
        "strategy_key": strategy_key,
    }


def _load_analysis_records(input_path: str) -> List[Dict[str, object]]:
    root = Path(input_path).expanduser().resolve()
    analysis_dir = root / "analysis" if (root / "analysis").exists() else root
    records = []
    for path in sorted(analysis_dir.glob("*.analysis.json")):
        records.append(json.loads(path.read_text(encoding="utf-8")))
    return records


def _load_summary_payload(input_path: str) -> Optional[Dict[str, object]]:
    root = Path(input_path).expanduser().resolve()
    summary_path = root / "summary.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))
    return None


def _load_array_calibration_payload(input_path: str) -> Optional[Dict[str, object]]:
    root = Path(input_path).expanduser().resolve()
    calibration_path = root / "array_calibration.json"
    if calibration_path.exists():
        return json.loads(calibration_path.read_text(encoding="utf-8"))
    return None


def _quality_by_clip(array_calibration_payload: Optional[Dict[str, object]]) -> Dict[str, Dict[str, object]]:
    quality_by_clip: Dict[str, Dict[str, object]] = {}
    if not array_calibration_payload:
        return quality_by_clip
    for camera in array_calibration_payload.get("cameras", []) or []:
        clip_id = str(camera.get("clip_id") or "").strip()
        if not clip_id:
            continue
        quality_by_clip[clip_id] = dict(camera.get("quality") or {})
    return quality_by_clip


def _anchor_target_log2_from_array_payload(array_calibration_payload: Optional[Dict[str, object]]) -> Optional[float]:
    if not array_calibration_payload:
        return None
    target = dict(array_calibration_payload.get("target") or {})
    exposure = dict(target.get("exposure") or {})
    direct_value = exposure.get("log2_luminance_target")
    if direct_value is not None:
        return float(direct_value)
    fallback_value = array_calibration_payload.get("target_exposure_log2")
    if fallback_value is not None:
        return float(fallback_value)
    return None


def _load_sidecar_map(input_path: str) -> Dict[str, Dict[str, object]]:
    root = Path(input_path).expanduser().resolve()
    sidecar_dir = root / "sidecars"
    payloads: Dict[str, Dict[str, object]] = {}
    if sidecar_dir.exists():
        for path in sorted(sidecar_dir.glob("*.sidecar.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            payloads[str(payload["clip_id"])] = payload
    return payloads


def normalize_target_strategy_name(name: str) -> str:
    normalized = name.strip().lower().replace("-", "_")
    aliases = {
        "optimal_exposure": "optimal_exposure",
        "brightest_valid": "optimal_exposure",
        "best_exposed": "optimal_exposure",
        "hero_camera": "hero_camera",
        "hero_clip": "hero_camera",
        "median": "median",
        "manual": "manual",
        "manual_clip": "manual",
        "manual_target": "manual_target",
    }
    if normalized not in aliases:
        raise ValueError("target strategy must be one of: median, optimal-exposure, manual, hero-camera, manual-target")
    return aliases[normalized]


def strategy_display_name(name: str) -> str:
    return {
        "hero_camera": "Hero Camera",
        "median": "Median",
        "optimal_exposure": "Optimal Exposure (Best Match to Gray)",
        "manual": "Manual Reference",
        "manual_target": "Manual Target",
    }[normalize_target_strategy_name(name)]


def preview_filename_for_clip_id(
    clip_id: str,
    variant: str,
    *,
    strategy: Optional[str] = None,
    run_id: Optional[str] = None,
    extension: str = "jpg",
) -> str:
    suffix = extension.lstrip(".")
    if strategy is None:
        return f"{clip_id}.{variant}.review.{run_id}.{suffix}" if run_id else f"{clip_id}.{variant}.review.{suffix}"
    return (
        f"{clip_id}.{variant}.review.{normalize_target_strategy_name(strategy)}.{run_id}.{suffix}"
        if run_id
        else f"{clip_id}.{variant}.review.{normalize_target_strategy_name(strategy)}.{suffix}"
    )


def clear_preview_cache(
    input_path: str,
    *,
    report_dir: Optional[str] = None,
    preserve_measurement_previews: bool = False,
) -> Dict[str, object]:
    root = Path(input_path).expanduser().resolve()
    preview_root = root / "previews"
    report_root = Path(report_dir).expanduser().resolve() if report_dir else root / "report"
    removed: list[str] = []
    if preview_root.exists():
        for path in sorted(preview_root.glob("*.review.*")):
            path.unlink()
            removed.append(str(path))
        measurement_dir = preview_root / "_measurement"
        if measurement_dir.exists() and not preserve_measurement_previews:
            shutil.rmtree(measurement_dir)
            removed.append(str(measurement_dir))
        ipp2_dir = preview_root / "_ipp2_validation"
        if ipp2_dir.exists():
            shutil.rmtree(ipp2_dir)
            removed.append(str(ipp2_dir))
        commands_path = preview_root / "preview_commands.json"
        if commands_path.exists():
            commands_path.unlink()
            removed.append(str(commands_path))
        validation_path = preview_root / "rmd_validation.json"
        if validation_path.exists():
            validation_path.unlink()
            removed.append(str(validation_path))
    if report_root.exists():
        for name in (
            "contact_sheet.json",
            "contact_sheet.html",
            "preview_contact_sheet.pdf",
            "review_manifest.json",
            "review_package.json",
            "review_validation.json",
            "ipp2_validation.json",
        ):
            path = report_root / name
            if path.exists():
                path.unlink()
                removed.append(str(path))
    return {
        "input_path": str(root),
        "report_dir": str(report_root),
        "removed_count": len(removed),
        "removed_paths": removed,
    }


def _resolve_redline_executable() -> str:
    executable = str(os.environ.get("R3DMATCH_REDLINE_EXECUTABLE", "") or "").strip()
    if not executable and REDLINE_CONFIG_PATH.exists():
        try:
            config_payload = json.loads(REDLINE_CONFIG_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid REDLine config JSON: {REDLINE_CONFIG_PATH}") from exc
        executable = str(config_payload.get("redline_executable") or config_payload.get("redline_path") or "").strip()
    executable = executable or "REDLine"
    resolved = shutil.which(executable)
    if resolved:
        return resolved
    candidate = Path(executable).expanduser()
    if candidate.exists():
        return str(candidate.resolve())
    raise RuntimeError("REDLine not found — real validation required")


def _is_real_red_source_path(source_path: str) -> bool:
    path = Path(source_path).expanduser().resolve()
    suffix = path.suffix.lower()
    if suffix == ".r3d":
        return path.is_file() and path.stat().st_size > 0
    if suffix == ".rdc":
        return path.is_dir()
    return False


def _resolve_probe_output_path(output_path: str | Path) -> Path:
    candidate = Path(output_path).expanduser().resolve()
    if candidate.exists():
        return candidate
    matches = sorted(candidate.parent.glob(f"{candidate.name}.*"))
    return matches[0] if matches else candidate


def _probe_redline_source_renderability(
    source_path: str,
    *,
    redline_executable: str,
    probe_root: Path,
) -> Dict[str, object]:
    clip_name = Path(source_path).stem
    requested_output = probe_root / f"{clip_name}.probe.jpg"
    command = [
        redline_executable,
        "--i",
        str(Path(source_path).expanduser().resolve()),
        "--o",
        str(requested_output),
        "--format",
        "3",
        "--start",
        "0",
        "--frameCount",
        "1",
        "--colorSciVersion",
        "3",
        "--silent",
        "--useMeta",
        "--colorSpace",
        str(COLOR_SPACE_CODES["BT.709"]),
        "--gammaCurve",
        str(GAMMA_CODES["BT.1886"]),
        "--outputToneMap",
        str(TONEMAP_CODES["medium"]),
        "--rollOff",
        str(ROLLOFF_CODES["medium"]),
        "--shadow",
        f"{float(SHADOW_ROLLOFF_VALUES['medium']):.3f}",
    ]
    completed = run_cancellable_subprocess(command)
    actual_output = _resolve_probe_output_path(requested_output)
    output = ((completed.stdout or "") + (completed.stderr or "")).strip()
    return {
        "source_path": str(Path(source_path).expanduser().resolve()),
        "command": command,
        "returncode": int(completed.returncode),
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "output_path": str(actual_output),
        "ok": int(completed.returncode) == 0 and actual_output.exists(),
        "output_preview": output[:400],
    }


def _require_real_redline_validation(
    *,
    input_path: str,
    analysis_records: List[Dict[str, object]],
    redline_executable: str,
    out_root: Path,
    progress_path: Optional[str] = None,
) -> Dict[str, object]:
    validation_started_at = time.perf_counter()
    summary = _load_summary_payload(input_path) or {}
    backend_name = str(summary.get("backend") or "").strip().lower()
    if backend_name == "mock":
        raise RuntimeError(
            "Real REDLine validation requires analysis produced from real media, not the mock backend."
        )
    source_probes: List[Dict[str, object]] = []
    invalid_sources: List[str] = []
    probe_root = out_root / "_real_redline_probe"
    probe_root.mkdir(parents=True, exist_ok=True)
    emit_review_progress(
        progress_path,
        phase="real_redline_validation_start",
        detail="Starting real REDLine source validation.",
        stage_label="Validating REDLine",
        clip_count=len(analysis_records),
        elapsed_seconds=0.0,
        review_mode="full_contact_sheet",
    )
    for probe_index, record in enumerate(analysis_records, start=1):
        source_path = str(record.get("source_path") or "").strip()
        if not source_path or not _is_real_red_source_path(source_path):
            invalid_sources.append(source_path or "<missing>")
            continue
        probe = _probe_redline_source_renderability(
            source_path,
            redline_executable=redline_executable,
            probe_root=probe_root,
        )
        source_probes.append(probe)
        if not bool(probe["ok"]):
            raise RuntimeError(
                f"Real REDLine validation requires decodable source media. Render probe failed for {source_path}. "
                f"STDERR: {str(probe.get('stderr') or '').strip()}"
            )
        emit_review_progress(
            progress_path,
            phase="real_redline_validation_probe",
            detail="Validated REDLine source renderability.",
            stage_label="Validating REDLine",
            clip_index=probe_index,
            clip_count=len(analysis_records),
            current_clip_id=str(record.get("clip_id") or ""),
            elapsed_seconds=time.perf_counter() - validation_started_at,
            review_mode="full_contact_sheet",
        )
    if invalid_sources:
        formatted = ", ".join(sorted(set(invalid_sources)))
        raise RuntimeError(
            "Real REDLine validation requires real source media inside the project. "
            f"Invalid or placeholder sources: {formatted}"
        )
    payload = {
        "required": True,
        "redline_executable": redline_executable,
        "analysis_backend": backend_name or None,
        "validated_source_count": len(source_probes),
        "source_probes": source_probes,
    }
    output_path = out_root / "real_redline_validation.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    emit_review_progress(
        progress_path,
        phase="real_redline_validation_complete",
        detail=f"Validated {len(source_probes)} real REDLine source(s).",
        stage_label="Validating REDLine",
        clip_count=len(source_probes),
        elapsed_seconds=time.perf_counter() - validation_started_at,
        review_mode="full_contact_sheet",
    )
    return {"path": str(output_path), "summary": payload}


def _detect_redline_capabilities(redline_executable: str) -> Dict[str, object]:
    help_result = run_cancellable_subprocess([redline_executable, "--help"])
    help_text = (help_result.stdout or "") + (help_result.stderr or "")
    version_line = next((line.strip() for line in help_text.splitlines() if "REDline Build" in line), None)
    return {
        "redline_path": redline_executable,
        "redline_version": version_line,
        "supports_look_metadata": "--look-metadata" in help_text,
        "supports_load_rmd": "--loadRMD" in help_text and "--useRMD" in help_text,
        "supports_lut": "--lut <filename>" in help_text,
        "supports_output_tonemap": "--outputToneMap" in help_text,
        "supports_rolloff": "--rollOff" in help_text,
        "supports_shadow_control": "--shadow <float>" in help_text,
        "supports_gamma_curve": "--gammaCurve" in help_text,
        "supports_color_space": "--colorSpace" in help_text,
        "help_returncode": help_result.returncode,
    }


def _normalize_preview_settings(
    *,
    preview_mode: str,
    preview_output_space: Optional[str],
    preview_output_gamma: Optional[str],
    preview_highlight_rolloff: Optional[str],
    preview_shadow_rolloff: Optional[str],
    preview_lut: Optional[str],
) -> Dict[str, object]:
    requested_mode = (preview_mode or "monitoring").lower()
    if requested_mode not in {"calibration", "monitoring"}:
        raise ValueError("preview mode must be monitoring (legacy calibration is accepted as an alias)")
    mode = "monitoring"
    defaults = DEFAULT_MONITORING_PREVIEW
    output_space = preview_output_space or str(defaults["output_space"])
    output_gamma = preview_output_gamma or str(defaults["output_gamma"])
    highlight_rolloff = (preview_highlight_rolloff or str(defaults["highlight_rolloff"])).lower()
    shadow_rolloff = (preview_shadow_rolloff or str(defaults["shadow_rolloff"])).lower()
    if output_space not in COLOR_SPACE_CODES:
        raise ValueError(f"unsupported preview output space: {output_space}")
    if output_gamma not in GAMMA_CODES:
        raise ValueError(f"unsupported preview output gamma: {output_gamma}")
    if highlight_rolloff not in ROLLOFF_CODES:
        raise ValueError(f"unsupported preview highlight rolloff: {highlight_rolloff}")
    if shadow_rolloff not in SHADOW_ROLLOFF_VALUES:
        raise ValueError(f"unsupported preview shadow rolloff: {shadow_rolloff}")
    lut_path = str(Path(preview_lut).expanduser().resolve()) if preview_lut else None
    payload = {
        "preview_mode": mode,
        "requested_preview_mode": requested_mode,
        "output_space": output_space,
        "output_gamma": output_gamma,
        "highlight_rolloff": highlight_rolloff,
        "shadow_rolloff": shadow_rolloff,
        "lut_path": lut_path,
        "output_tonemap": "medium",
    }
    if requested_mode == "calibration":
        payload["preview_mode_alias"] = "calibration_compatibility_alias_to_monitoring"
        payload["preview_mode_note"] = (
            "Legacy calibration preview mode now resolves to the canonical monitoring transform "
            "to keep preview and validation semantically aligned."
        )
    return payload


def _preview_transform_label(settings: Dict[str, object]) -> str:
    parts = [
        "REDLine IPP2",
        str(settings["output_space"]),
        str(settings["output_gamma"]),
        "Medium",
        str(settings["highlight_rolloff"]).title(),
    ]
    if settings.get("lut_path"):
        parts.append(f"LUT={Path(str(settings['lut_path'])).name}")
    return " / ".join(parts)


def _color_preview_policy() -> Dict[str, object]:
    enabled = os.environ.get(COLOR_PREVIEW_OVERRIDE_ENV, "").strip().lower() in {"1", "true", "yes", "on"}
    note = (
        None
        if enabled
        else "Color/CDL corrections are still computed and exported, but color preview is disabled for operator review on this REDLine build because the rendered color path has not been proven visually trustworthy."
    )
    return {
        "enabled": enabled,
        "status": "enabled_unverified_override" if enabled else "disabled_unverified",
        "note": note,
    }


def _report_grid_columns(clip_count: int) -> int:
    if clip_count <= 6:
        return 3
    if clip_count <= 16:
        return 4
    if clip_count <= 36:
        return 6
    return 8


def _report_tiles_per_page(columns: int) -> int:
    return max(12, min(18, columns * 2))


def _chunk_tiles(items: List[Dict[str, object]], chunk_size: int) -> List[List[Dict[str, object]]]:
    return [items[index:index + chunk_size] for index in range(0, len(items), chunk_size)] or [[]]


def _format_cdl_gain_saturation(color_metrics: Dict[str, object]) -> str:
    color_model = color_metrics.get("lift_gamma_gain_saturation") or {}
    gain = color_model.get("gain") or color_metrics.get("rgb_gains") or [1.0, 1.0, 1.0]
    saturation = float(color_model.get("saturation", color_metrics.get("saturation", 1.0)))
    return f"{float(gain[0]):.3f}, {float(gain[1]):.3f}, {float(gain[2]):.3f} | sat {saturation:.3f}"


def _extract_preview_corrections(sidecar_payload: Dict[str, object]) -> Dict[str, object]:
    exposure_mapping = dict(sidecar_payload.get("rmd_mapping", {}).get("exposure", {}))
    color_mapping = dict(sidecar_payload.get("rmd_mapping", {}).get("color", {}))
    calibration_state = dict(sidecar_payload.get("calibration_state", {}))
    rgb_gains = color_mapping.get("rgb_neutral_gains")
    if rgb_gains is None and calibration_state.get("rgb_neutral_gains"):
        gains = calibration_state["rgb_neutral_gains"]
        rgb_gains = [gains["r"], gains["g"], gains["b"]]
    return {
        "exposure_stops": float(exposure_mapping.get("final_offset_stops", 0.0) or 0.0),
        "rgb_gains": rgb_gains,
        "cdl": color_mapping.get("cdl") or (rgb_gains_to_cdl(rgb_gains) if rgb_gains is not None else None),
    }


def _extract_normalized_roi_region_hwc(image: np.ndarray, calibration_roi: Optional[Dict[str, float]]) -> np.ndarray:
    if calibration_roi is None:
        chw = np.moveaxis(image, -1, 0)
        centered = extract_center_region(chw, fraction=0.4)
        return np.moveaxis(centered, 0, -1)
    height, width = image.shape[:2]
    x0 = max(0, min(width - 1, int(np.floor(float(calibration_roi["x"]) * width))))
    y0 = max(0, min(height - 1, int(np.floor(float(calibration_roi["y"]) * height))))
    x1 = max(x0 + 1, min(width, int(np.ceil((float(calibration_roi["x"]) + float(calibration_roi["w"])) * width))))
    y1 = max(y0 + 1, min(height, int(np.ceil((float(calibration_roi["y"]) + float(calibration_roi["h"])) * height))))
    return image[y0:y1, x0:x1, :]


def _solver_sampling_measurement_from_hwc_region(region: np.ndarray) -> Dict[str, object]:
    chw = np.moveaxis(region, -1, 0)
    stats = _measure_three_sample_statistics(chw)
    return {
        "measured_log2_luminance_monitoring": float(stats["measured_log2_luminance"]),
        "measured_rgb_mean_monitoring": [float(value) for value in stats["measured_rgb_mean"]],
        "measured_rgb_chromaticity_monitoring": [float(value) for value in stats["measured_rgb_chromaticity"]],
        "valid_pixel_count_monitoring": int(stats["valid_pixel_count"]),
        "monitoring_roi_variance": float(stats["roi_variance"]),
        "neutral_sample_count_monitoring": int(stats["neutral_sample_count"]),
        "neutral_sample_log2_spread_monitoring": float(stats["neutral_sample_log2_spread"]),
        "neutral_sample_chromaticity_spread_monitoring": float(stats["neutral_sample_chromaticity_spread"]),
        "neutral_samples_monitoring": [dict(item) for item in stats["neutral_samples"]],
        "measurement_geometry": "three_rect_windows_within_roi",
    }


def _measure_rendered_preview_roi(preview_path: str, calibration_roi: Optional[Dict[str, float]]) -> Dict[str, object]:
    image = np.asarray(Image.open(preview_path).convert("RGB"), dtype=np.float32) / 255.0
    region = _extract_normalized_roi_region_hwc(image, calibration_roi)
    stats = _solver_sampling_measurement_from_hwc_region(region)
    pixels = region.reshape(-1, 3)
    saturation_fraction = float(np.mean(np.max(pixels, axis=1) >= 0.998)) if pixels.size else 0.0
    return {
        **stats,
        "measured_saturation_fraction_monitoring": saturation_fraction,
    }


def _sphere_roi_for_region_hwc(region: np.ndarray) -> SphereROI:
    height, width = region.shape[:2]
    radius = max(1.0, min(width, height) * 0.46)
    return SphereROI(
        cx=(float(width) - 1.0) / 2.0,
        cy=(float(height) - 1.0) / 2.0,
        r=float(radius),
    )


def _tight_sphere_crop_bounds(region: np.ndarray, roi: SphereROI, *, margin_ratio: float = 1.35) -> Dict[str, int]:
    height, width = region.shape[:2]
    margin = max(float(roi.r) * float(margin_ratio), float(roi.r) + 4.0)
    x0 = max(0, int(math.floor(float(roi.cx) - margin)))
    x1 = min(width, int(math.ceil(float(roi.cx) + margin)))
    y0 = max(0, int(math.floor(float(roi.cy) - margin)))
    y1 = min(height, int(math.ceil(float(roi.cy) + margin)))
    if x1 <= x0:
        x0, x1 = 0, width
    if y1 <= y0:
        y0, y1 = 0, height
    return {"x0": int(x0), "x1": int(x1), "y0": int(y0), "y1": int(y1)}


def _offset_zone_measurements_to_parent(
    zone_measurements: List[Dict[str, object]],
    *,
    offset_x: int,
    offset_y: int,
    parent_width: int,
    parent_height: int,
) -> List[Dict[str, object]]:
    adjusted: List[Dict[str, object]] = []
    for measurement in list(zone_measurements or []):
        item = copy.deepcopy(dict(measurement))
        bounds = dict(item.get("bounds") or {})
        pixel = dict(bounds.get("pixel") or {})
        if pixel:
            pixel = {
                "x0": int(pixel.get("x0", 0) or 0) + int(offset_x),
                "y0": int(pixel.get("y0", 0) or 0) + int(offset_y),
                "x1": int(pixel.get("x1", 0) or 0) + int(offset_x),
                "y1": int(pixel.get("y1", 0) or 0) + int(offset_y),
            }
            bounds["pixel"] = pixel
            bounds["normalized_within_roi"] = {
                "x": float(pixel["x0"]) / float(max(parent_width, 1)),
                "y": float(pixel["y0"]) / float(max(parent_height, 1)),
                "w": float(max(pixel["x1"] - pixel["x0"], 1)) / float(max(parent_width, 1)),
                "h": float(max(pixel["y1"] - pixel["y0"], 1)) / float(max(parent_height, 1)),
            }
        polygon = dict(bounds.get("polygon") or {})
        pixel_polygon = []
        for point in list(polygon.get("pixel") or []):
            pixel_polygon.append(
                {
                    "x": float(point.get("x", 0.0) or 0.0) + float(offset_x),
                    "y": float(point.get("y", 0.0) or 0.0) + float(offset_y),
                }
            )
        if pixel_polygon:
            polygon["pixel"] = pixel_polygon
            polygon["normalized_within_roi"] = [
                {
                    "x": float(point["x"]) / float(max(parent_width, 1)),
                    "y": float(point["y"]) / float(max(parent_height, 1)),
                }
                for point in pixel_polygon
            ]
            bounds["polygon"] = polygon
        item["bounds"] = bounds
        adjusted.append(item)
    return adjusted


def _sphere_detection_label(confidence: float) -> str:
    value = float(confidence)
    if value >= SPHERE_DETECTION_HIGH_CONFIDENCE:
        return "HIGH"
    if value >= SPHERE_DETECTION_MEDIUM_CONFIDENCE:
        return "MEDIUM"
    if value >= SPHERE_DETECTION_LOW_CONFIDENCE:
        return "LOW"
    return "FAILED"


def _zone_containment_score(roi: SphereROI, width: int, height: int) -> float:
    from .calibration import _sphere_zone_geometries

    scores: List[float] = []
    # The containment test is rotationally invariant inside a circle, so a
    # canonical axis is enough here even though measurement uses the detected
    # gradient axis later.
    for zone in _sphere_zone_geometries(width, height, roi, (0.0, -1.0)):
        polygon = dict(zone.get("polygon") or {})
        corners = [
            (float(point.get("x", 0.0) or 0.0), float(point.get("y", 0.0) or 0.0))
            for point in list(polygon.get("pixel") or [])
        ]
        if not corners:
            bounds = dict(zone["pixel"])
            corners = [
                (float(bounds["x0"]), float(bounds["y0"])),
                (float(bounds["x1"]), float(bounds["y0"])),
                (float(bounds["x0"]), float(bounds["y1"])),
                (float(bounds["x1"]), float(bounds["y1"])),
            ]
        inside = 0
        for xx, yy in corners:
            if ((xx - float(roi.cx)) ** 2 + (yy - float(roi.cy)) ** 2) <= float(roi.r) ** 2:
                inside += 1
        scores.append(float(inside) / float(max(len(corners), 1)))
    return float(np.mean(np.asarray(scores, dtype=np.float32))) if scores else 0.0


def _evaluate_detected_sphere_roi(
    region: np.ndarray,
    *,
    roi: SphereROI,
    edge_map: np.ndarray,
    accumulator: float,
    detection_source: str,
) -> Dict[str, object]:
    height, width = region.shape[:2]
    yy, xx = np.meshgrid(np.arange(height, dtype=np.float32), np.arange(width, dtype=np.float32), indexing="ij")
    radius = float(max(roi.r, 1.0))
    distance = np.sqrt((xx - float(roi.cx)) ** 2 + (yy - float(roi.cy)) ** 2)
    interior_mask = distance <= radius * 0.92
    ring_mask = (distance >= radius * 1.02) & (distance <= radius * 1.20)
    luminance = np.clip(region[..., 0] * 0.2126 + region[..., 1] * 0.7152 + region[..., 2] * 0.0722, 1e-6, 1.0)
    interior_values = luminance[interior_mask]
    ring_values = luminance[ring_mask]
    interior_stddev = float(np.std(interior_values)) if interior_values.size else 1.0
    interior_mean = float(np.mean(interior_values)) if interior_values.size else 0.0
    ring_mean = float(np.mean(ring_values)) if ring_values.size else interior_mean
    contrast = abs(interior_mean - ring_mean)
    radius_ratio = radius / float(max(min(height, width), 1))
    radius_score = 1.0 - min(abs(radius_ratio - 0.22) / 0.14, 1.0)
    center_x_ratio = float(roi.cx) / float(max(width, 1))
    center_y_ratio = float(roi.cy) / float(max(height, 1))
    center_x_score = 1.0 - min(abs(center_x_ratio - 0.68) / 0.30, 1.0)
    center_y_score = 1.0 - min(abs(center_y_ratio - 0.50) / 0.35, 1.0)
    edge_ring = np.abs(distance - radius) <= 2.5
    edge_strength = float(np.mean(edge_map[edge_ring])) if np.any(edge_ring) else 0.0
    edge_score = min(max(float(accumulator), edge_strength) / 0.30, 1.0)
    consistency_score = 1.0 - min(interior_stddev / 0.12, 1.0)
    contrast_score = min(contrast / 0.10, 1.0)
    zone_score = _zone_containment_score(roi, width, height)
    confidence = float(
        np.mean(
            np.asarray(
                [
                    radius_score,
                    center_x_score,
                    center_y_score,
                    edge_score,
                    consistency_score,
                    contrast_score,
                    zone_score,
                ],
                dtype=np.float32,
            )
        )
    )
    return {
        "roi": {"cx": float(roi.cx), "cy": float(roi.cy), "r": radius},
        "confidence": confidence,
        "confidence_label": _sphere_detection_label(confidence),
        "source": detection_source,
        "radius_ratio": radius_ratio,
        "center_ratio": {"x": center_x_ratio, "y": center_y_ratio},
        "edge_strength": float(edge_strength),
        "hough_accumulator": float(accumulator),
        "interior_luminance_mean": interior_mean,
        "interior_luminance_stddev": interior_stddev,
        "surround_luminance_mean": ring_mean,
        "contrast_to_surround": contrast,
        "zone_inside_score": zone_score,
        "validation": {
            "radius_sane": 0.10 <= radius_ratio <= 0.30,
            "center_sane": 0.35 <= center_x_ratio <= 0.90 and 0.15 <= center_y_ratio <= 0.85,
            "zones_inside": zone_score >= 0.85,
            "contrast_present": contrast >= 0.015,
        },
    }


def _detect_sphere_candidates_in_region_hwc(
    region: np.ndarray,
    *,
    search_bounds: Optional[Tuple[int, int, int, int]] = None,
    detection_source: str,
    sigma: float = 2.0,
    low_threshold: float = 0.03,
    high_threshold: float = 0.12,
) -> List[Dict[str, object]]:
    height, width = region.shape[:2]
    if search_bounds is None:
        x0, y0, x1, y1 = 0, 0, width, height
    else:
        x0, y0, x1, y1 = search_bounds
    subregion = region[y0:y1, x0:x1, :]
    if subregion.size == 0:
        return []
    min_dimension = min(subregion.shape[:2])
    if min_dimension < 64:
        return []
    grayscale = np.clip(color.rgb2gray(subregion), 0.0, 1.0)
    edge_map = feature.canny(
        grayscale,
        sigma=float(sigma),
        low_threshold=float(low_threshold),
        high_threshold=float(high_threshold),
    )
    if not np.any(edge_map):
        return []
    min_radius = max(24, int(round(float(min_dimension) * 0.10)))
    max_radius = max(min_radius + 4, int(round(float(min_dimension) * 0.30)))
    radii = np.arange(min_radius, max_radius + 1, 4, dtype=np.int32)
    if radii.size == 0:
        return []
    hough_space = transform.hough_circle(edge_map, radii)
    accumulators, centers_x, centers_y, detected_radii = transform.hough_circle_peaks(
        hough_space,
        radii,
        total_num_peaks=8,
    )
    candidates: List[Dict[str, object]] = []
    for accumulator, center_x, center_y, radius in zip(accumulators, centers_x, centers_y, detected_radii):
        roi = SphereROI(
            cx=float(center_x + x0),
            cy=float(center_y + y0),
            r=float(radius),
        )
        candidates.append(
            _evaluate_detected_sphere_roi(
                region,
                roi=roi,
                edge_map=edge_map if search_bounds is None else np.pad(edge_map, ((y0, height - y1), (x0, width - x1)), mode="constant"),
                accumulator=float(accumulator),
                detection_source=detection_source,
            )
        )
    return sorted(candidates, key=lambda item: float(item.get("confidence", 0.0) or 0.0), reverse=True)


def _localized_search_bounds_for_candidate(candidate: Dict[str, object], width: int, height: int) -> Optional[Tuple[int, int, int, int]]:
    roi_payload = dict(candidate.get("roi") or {})
    if not roi_payload:
        return None
    radius = float(roi_payload.get("r", 0.0) or 0.0)
    if radius <= 0.0:
        return None
    cx = float(roi_payload.get("cx", 0.0) or 0.0)
    cy = float(roi_payload.get("cy", 0.0) or 0.0)
    half_width = radius * 1.45
    half_height = radius * 1.35
    x0 = max(0, int(math.floor(cx - half_width)))
    y0 = max(0, int(math.floor(cy - half_height)))
    x1 = min(width, int(math.ceil(cx + half_width)))
    y1 = min(height, int(math.ceil(cy + half_height)))
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def _detect_sphere_roi_in_region_hwc(region: np.ndarray) -> Dict[str, object]:
    height, width = region.shape[:2]
    min_dimension = min(height, width)
    if min_dimension < 64:
        return {"source": "failed", "confidence": 0.0, "confidence_label": "FAILED", "validation": {"reason": "region_too_small"}}
    primary_candidates = _detect_sphere_candidates_in_region_hwc(
        region,
        detection_source="primary_detected",
    )
    if primary_candidates and str(primary_candidates[0].get("confidence_label") or "") in {"HIGH", "MEDIUM"}:
        return dict(primary_candidates[0])
    search_bounds = (
        int(round(width * 0.40)),
        int(round(height * 0.18)),
        int(round(width * 0.92)),
        int(round(height * 0.82)),
    )
    secondary_candidates = _detect_sphere_candidates_in_region_hwc(
        region,
        search_bounds=search_bounds,
        detection_source="secondary_detected",
        sigma=1.6,
        low_threshold=0.02,
        high_threshold=0.10,
    )
    candidate_pool = [*(primary_candidates[:3]), *(secondary_candidates[:3])]
    if candidate_pool:
        best = max(candidate_pool, key=lambda item: float(item.get("confidence", 0.0) or 0.0))
        if float(best.get("confidence", 0.0) or 0.0) >= SPHERE_DETECTION_LOW_CONFIDENCE:
            return dict(best)
    localized_candidates: List[Dict[str, object]] = []
    if candidate_pool:
        seed = max(candidate_pool, key=lambda item: float(item.get("confidence", 0.0) or 0.0))
        localized_bounds = _localized_search_bounds_for_candidate(seed, width, height)
        if localized_bounds is not None:
            localized_candidates = _detect_sphere_candidates_in_region_hwc(
                region,
                search_bounds=localized_bounds,
                detection_source="localized_recovery",
                sigma=1.2,
                low_threshold=0.015,
                high_threshold=0.08,
            )
            if localized_candidates:
                localized_best = max(localized_candidates, key=lambda item: float(item.get("confidence", 0.0) or 0.0))
                if float(localized_best.get("confidence", 0.0) or 0.0) >= SPHERE_DETECTION_LOW_CONFIDENCE:
                    return dict(localized_best)
    forced_pool = [*candidate_pool, *(localized_candidates[:3])]
    if forced_pool:
        forced_best = max(forced_pool, key=lambda item: float(item.get("confidence", 0.0) or 0.0))
        forced_result = dict(forced_best)
        forced_result["source"] = "forced_best_effort"
        forced_result["confidence"] = max(float(forced_result.get("confidence", 0.0) or 0.0), 0.01)
        forced_result["confidence_label"] = "LOW"
        validation = dict(forced_result.get("validation") or {})
        validation["forced_best_effort"] = True
        forced_result["validation"] = validation
        return forced_result
    return {
        "source": "failed",
        "confidence": 0.0,
        "confidence_label": "FAILED",
        "validation": {
            "reason": "no_plausible_circle_candidate",
            "primary_candidate_count": len(primary_candidates),
            "secondary_candidate_count": len(secondary_candidates),
            "localized_candidate_count": len(localized_candidates),
        },
    }


def _coerce_sphere_roi(roi_payload: Optional[Dict[str, float]]) -> Optional[SphereROI]:
    if not roi_payload:
        return None
    if not {"cx", "cy", "r"} <= set(roi_payload):
        return None
    return SphereROI(
        cx=float(roi_payload["cx"]),
        cy=float(roi_payload["cy"]),
        r=float(roi_payload["r"]),
    )


def _measure_rendered_preview_roi_ipp2(
    preview_path: str,
    calibration_roi: Optional[Dict[str, float]],
    *,
    sphere_roi_override: Optional[Dict[str, float]] = None,
) -> Dict[str, object]:
    measurement_started_at = time.perf_counter()
    phase_started_at = measurement_started_at
    image = np.asarray(Image.open(preview_path).convert("RGB"), dtype=np.float32) / 255.0
    image_load_seconds = time.perf_counter() - phase_started_at
    phase_started_at = time.perf_counter()
    region = _extract_normalized_roi_region_hwc(image, calibration_roi)
    roi_extract_seconds = time.perf_counter() - phase_started_at
    if region.size == 0:
        stats = _measure_rendered_preview_roi(preview_path, calibration_roi)
        return {
            **stats,
            "measurement_geometry": "empty_roi_fallback",
            "sampling_method": "empty_roi_fallback",
            "sampling_confidence": 0.0,
            "mask_fraction": 0.0,
            "gray_exposure_summary": "n/a",
            "zone_measurements": [],
            "render_width": int(image.shape[1]),
            "render_height": int(image.shape[0]),
            "measurement_runtime": {
                "image_load_seconds": float(image_load_seconds),
                "roi_extract_seconds": float(roi_extract_seconds),
                "sphere_detection_seconds": 0.0,
                "gradient_axis_seconds": 0.0,
                "zone_stat_seconds": 0.0,
                "profile_measurement_seconds": 0.0,
                "total_measurement_seconds": float(time.perf_counter() - measurement_started_at),
            },
        }
    sphere_roi = _coerce_sphere_roi(sphere_roi_override)
    sphere_roi_source = "reused_from_original" if sphere_roi is not None else "failed"
    detection_confidence = 1.0 if sphere_roi is not None else 0.0
    detection_label = "HIGH" if sphere_roi is not None else "FAILED"
    detection_details: Dict[str, object] = {}
    detection_started_at = time.perf_counter()
    if sphere_roi is None:
        if calibration_roi is None:
            detected = _detect_sphere_roi_in_region_hwc(region)
            detection_details = dict(detected)
            detection_confidence = float(detected.get("confidence", 0.0) or 0.0)
            detection_label = str(detected.get("confidence_label") or "FAILED")
            sphere_roi_source = str(detected.get("source") or "failed")
            sphere_roi = _coerce_sphere_roi(detected.get("roi"))
        else:
            sphere_roi_source = "provided_roi"
            detection_confidence = 1.0
            detection_label = "HIGH"
            sphere_roi = _sphere_roi_for_region_hwc(region)
    sphere_detection_seconds = time.perf_counter() - detection_started_at
    if sphere_roi is None:
        failed_payload = {
            "measured_log2_luminance": 0.0,
            "measured_log2_luminance_monitoring": 0.0,
            "measured_rgb_mean": [0.0, 0.0, 0.0],
            "measured_rgb_chromaticity": [1.0 / 3.0] * 3,
            "valid_pixel_count": 0,
            "roi_variance": 0.0,
            "monitoring_roi_variance": 0.0,
            "measured_saturation_fraction_monitoring": 0.0,
            "measurement_geometry": "sphere_detection_failed",
            "sampling_method": "sphere_detection_failed",
            "sampling_confidence": 0.0,
            "mask_fraction": 0.0,
            "interior_radius_ratio": 0.0,
            "neutral_sample_count": 0,
            "neutral_sample_log2_spread": 0.0,
            "neutral_sample_chromaticity_spread": 0.0,
            "neutral_samples": [],
            "zone_measurements": [],
            "gray_exposure_summary": "Sphere detection failed",
            "bright_ire": 0.0,
            "center_ire": 0.0,
            "dark_ire": 0.0,
            "sample_1_ire": 0.0,
            "sample_2_ire": 0.0,
            "sample_3_ire": 0.0,
            "top_ire": 0.0,
            "mid_ire": 0.0,
            "bottom_ire": 0.0,
            "zone_spread_ire": 0.0,
            "zone_spread_stops": 0.0,
            "sphere_roi_source": "failed",
            "sphere_detection_confidence": 0.0,
            "sphere_detection_label": "FAILED",
            "sphere_detection_details": detection_details,
            "detected_sphere_roi": {},
            "detection_failed": True,
            "render_width": int(image.shape[1]),
            "render_height": int(image.shape[0]),
            "measurement_runtime": {
                "image_load_seconds": float(image_load_seconds),
                "roi_extract_seconds": float(roi_extract_seconds),
                "sphere_detection_seconds": float(sphere_detection_seconds),
                "gradient_axis_seconds": 0.0,
                "zone_stat_seconds": 0.0,
                "profile_measurement_seconds": 0.0,
                "total_measurement_seconds": float(time.perf_counter() - measurement_started_at),
            },
        }
        return failed_payload
    crop_started_at = time.perf_counter()
    crop_bounds = _tight_sphere_crop_bounds(region, sphere_roi)
    cropped_region = region[crop_bounds["y0"]:crop_bounds["y1"], crop_bounds["x0"]:crop_bounds["x1"], :]
    cropped_roi = SphereROI(
        cx=float(sphere_roi.cx) - float(crop_bounds["x0"]),
        cy=float(sphere_roi.cy) - float(crop_bounds["y0"]),
        r=float(sphere_roi.r),
    )
    crop_seconds = time.perf_counter() - crop_started_at
    profile_started_at = time.perf_counter()
    stats = measure_sphere_zone_profile_statistics(np.transpose(cropped_region, (2, 0, 1)), cropped_roi, sampling_variant="refined")
    profile_measurement_seconds = time.perf_counter() - profile_started_at
    pixels = region.reshape(-1, 3)
    saturation_fraction = float(np.mean(np.max(pixels, axis=1) >= 0.998)) if pixels.size else 0.0
    timing = dict(stats.get("timing") or {})
    zone_measurements = _offset_zone_measurements_to_parent(
        [dict(item) for item in (stats.get("zone_measurements") or [])],
        offset_x=int(crop_bounds["x0"]),
        offset_y=int(crop_bounds["y0"]),
        parent_width=int(region.shape[1]),
        parent_height=int(region.shape[0]),
    )
    dominant_gradient_axis = dict(stats.get("dominant_gradient_axis") or {})
    return {
        "measured_log2_luminance": float(stats["measured_log2_luminance"]),
        "measured_log2_luminance_monitoring": float(stats["measured_log2_luminance"]),
        "measured_rgb_mean": [float(value) for value in stats["measured_rgb_mean"]],
        "measured_rgb_chromaticity": [float(value) for value in stats["measured_rgb_chromaticity"]],
        "measured_rgb_chromaticity_monitoring": [float(value) for value in stats["measured_rgb_chromaticity"]],
        "valid_pixel_count": int(stats["valid_pixel_count"]),
        "roi_variance": float(stats["roi_variance"]),
        "monitoring_roi_variance": float(stats["roi_variance"]),
        "measured_saturation_fraction_monitoring": saturation_fraction,
        "measurement_geometry": str(stats.get("measurement_geometry") or "three_band_gradient_aligned_profile_within_refined_sphere_mask"),
        "sampling_method": str(stats["sampling_method"]),
        "sampling_confidence": float(stats["confidence"]),
        "mask_fraction": float(stats["mask_fraction"]),
        "interior_radius_ratio": float(stats.get("interior_radius_ratio", 0.0) or 0.0),
        "neutral_sample_count": int(stats.get("neutral_sample_count", 0) or 0),
        "neutral_sample_log2_spread": float(stats.get("neutral_sample_log2_spread", 0.0) or 0.0),
        "neutral_sample_chromaticity_spread": float(stats.get("neutral_sample_chromaticity_spread", 0.0) or 0.0),
        "neutral_samples": [dict(item) for item in zone_measurements],
        "zone_measurements": [dict(item) for item in zone_measurements],
        "gray_exposure_summary": str(stats.get("aggregate_sphere_profile") or "n/a"),
        "bright_ire": float(stats.get("bright_ire", 0.0) or 0.0),
        "center_ire": float(stats.get("center_ire", 0.0) or 0.0),
        "dark_ire": float(stats.get("dark_ire", 0.0) or 0.0),
        "sample_1_ire": float(stats.get("sample_1_ire", stats.get("bright_ire", 0.0)) or 0.0),
        "sample_2_ire": float(stats.get("sample_2_ire", stats.get("center_ire", 0.0)) or 0.0),
        "sample_3_ire": float(stats.get("sample_3_ire", stats.get("dark_ire", 0.0)) or 0.0),
        "top_ire": float(stats.get("top_ire", 0.0) or 0.0),
        "mid_ire": float(stats.get("mid_ire", 0.0) or 0.0),
        "bottom_ire": float(stats.get("bottom_ire", 0.0) or 0.0),
        "zone_spread_ire": float(stats.get("zone_spread_ire", 0.0) or 0.0),
        "zone_spread_stops": float(stats.get("zone_spread_stops", 0.0) or 0.0),
        "dominant_gradient_axis": dominant_gradient_axis,
        "sphere_roi_source": sphere_roi_source,
        "sphere_detection_confidence": detection_confidence,
        "sphere_detection_label": detection_label,
        "sphere_detection_details": detection_details,
        "detected_sphere_roi": {"cx": float(sphere_roi.cx), "cy": float(sphere_roi.cy), "r": float(sphere_roi.r)},
        "detection_failed": False,
        "render_width": int(image.shape[1]),
        "render_height": int(image.shape[0]),
        "measurement_crop_bounds": dict(crop_bounds),
        "measurement_crop_size": {"width": int(cropped_region.shape[1]), "height": int(cropped_region.shape[0])},
        "measurement_runtime": {
            "image_load_seconds": float(image_load_seconds),
            "roi_extract_seconds": float(roi_extract_seconds),
            "sphere_detection_seconds": float(sphere_detection_seconds),
            "crop_seconds": float(crop_seconds),
            "build_mask_seconds": float(timing.get("build_mask_seconds", 0.0) or 0.0),
            "gradient_axis_seconds": float(timing.get("gradient_axis_seconds", 0.0) or 0.0),
            "zone_geometry_seconds": float(timing.get("zone_geometry_seconds", 0.0) or 0.0),
            "zone_stat_seconds": float(timing.get("zone_measurement_seconds", 0.0) or 0.0),
            "profile_measurement_seconds": float(profile_measurement_seconds),
            "total_measurement_seconds": float(time.perf_counter() - measurement_started_at),
        },
    }


def _ire_from_log2_luminance(log2_value: float) -> float:
    return float(max(0.0, (2.0 ** float(log2_value)) * 100.0))


def _failed_detection_summary(target_zone_profile: List[Dict[str, object]]) -> Dict[str, object]:
    return {
        "zone_residuals": [],
        "weighted_zone_components": [],
        "weighted_residual_stops": 0.0,
        "zone_weights": dict(SPHERE_PROFILE_ZONE_WEIGHTS),
        "aggregate_measurement_log2": 0.0,
        "aggregate_target_log2": float(_derived_profile_scalar(target_zone_profile).get("derived_exposure_value_log2", 0.0) or 0.0),
        "aggregate_residual_stops": 0.0,
        "mean_abs_residual_stops": 0.0,
        "max_abs_residual_stops": 0.0,
        "gray_exposure_summary": "Sphere detection failed",
        "target_gray_exposure_summary": _format_zone_profile_summary(target_zone_profile),
        "measured_profile": [],
        "target_profile": [dict(item) for item in _ordered_zone_profile(target_zone_profile)],
        "legacy_profile_median_measurement_log2": 0.0,
        "legacy_profile_median_target_log2": 0.0,
        "derived_exposure_value": 0.0,
        "derived_exposure_value_log2": 0.0,
        "derived_target_value": float(_derived_profile_scalar(target_zone_profile).get("derived_exposure_value_log2", 0.0) or 0.0),
        "derived_target_value_log2": float(_derived_profile_scalar(target_zone_profile).get("derived_exposure_value_log2", 0.0) or 0.0),
        "derived_exposure_ire": 0.0,
        "derived_target_ire": float(_derived_profile_scalar(target_zone_profile).get("derived_exposure_ire", 0.0) or 0.0),
        "derived_residual_stops": 0.0,
        "derived_residual_abs_stops": IPP2_VALIDATION_REVIEW_STOPS + 1.0,
        "derived_exposure_offset_stops": 0.0,
        "profile_audit_status": "PROFILE SHAPE MISMATCH",
        "profile_audit_label": "Profile mismatch",
        "profile_note": "Sphere detection failed — review needed",
        "profile_audit_tone": "danger",
        "profile_worst_zone_label": "Zone",
        "profile_worst_zone_residual_stops": IPP2_VALIDATION_REVIEW_STOPS + 1.0,
        "profile_zone_spread_ire": 0.0,
        "target_zone_spread_ire": 0.0,
        "profile_spread_delta_ire": 0.0,
        "detection_failed": True,
    }


def _zone_profile_by_label(zone_measurements: List[Dict[str, object]]) -> Dict[str, Dict[str, object]]:
    def normalize_label(label: str) -> str:
        normalized = str(label or "").strip()
        aliases = {
            "upper_mid": "bright_side",
            "center": "center",
            "lower_mid": "dark_side",
            "bright_side": "bright_side",
            "dark_side": "dark_side",
        }
        return aliases.get(normalized, normalized)

    zone_map: Dict[str, Dict[str, object]] = {}
    for item in list(zone_measurements or []):
        label = normalize_label(str(item.get("label") or ""))
        if not label:
            continue
        normalized_item = dict(item)
        normalized_item["label"] = label
        normalized_item["display_label"] = SPHERE_PROFILE_ZONE_DISPLAY.get(
            label,
            str(item.get("display_label") or label),
        )
        zone_map[label] = normalized_item
    return zone_map


def _ordered_zone_profile(zone_measurements: List[Dict[str, object]]) -> List[Dict[str, object]]:
    zone_map = _zone_profile_by_label(zone_measurements)
    return [zone_map[label] for label in SPHERE_PROFILE_ZONE_ORDER if label in zone_map]


def _format_zone_profile_summary(zone_measurements: List[Dict[str, object]]) -> str:
    ordered = _ordered_zone_profile(zone_measurements)
    if not ordered:
        return "n/a"
    parts = []
    for item in ordered:
        label = SPHERE_PROFILE_ZONE_DISPLAY.get(str(item.get("label") or ""), str(item.get("display_label") or str(item.get("label") or "Zone")))
        ire = float(item.get("measured_ire", _ire_from_log2_luminance(float(item.get("measured_log2_luminance", 0.0) or 0.0))) or 0.0)
        parts.append(f"{label} {ire:.0f}")
    return " / ".join(parts) + " IRE"


def _format_zone_residual_summary(zone_residuals: List[Dict[str, object]]) -> str:
    if not zone_residuals:
        return "n/a"
    residual_by_label = {str(item.get("label") or ""): dict(item) for item in list(zone_residuals or []) if str(item.get("label") or "").strip()}
    parts = []
    for label in SPHERE_PROFILE_ZONE_ORDER:
        item = residual_by_label.get(label)
        if item is None:
            continue
        display_label = SPHERE_PROFILE_ZONE_DISPLAY.get(label, str(item.get("display_label") or label))
        residual = float(item.get("residual_stops", 0.0) or 0.0)
        parts.append(f"{display_label} {residual:+.2f}")
    return " / ".join(parts) + " stops" if parts else "n/a"


def _zone_ire_from_profile(zone_measurements: List[Dict[str, object]], label: str) -> float:
    zone = _zone_profile_by_label(zone_measurements).get(str(label))
    if zone is None:
        return 0.0
    measured_log2 = float(zone.get("measured_log2_luminance", 0.0) or 0.0)
    return float(zone.get("measured_ire", _ire_from_log2_luminance(measured_log2)) or 0.0)


def _median_zone_profile(rows: List[Dict[str, object]], *, field_name: str) -> List[Dict[str, object]]:
    profiles = [list(item.get(field_name) or []) for item in rows if list(item.get(field_name) or [])]
    target_profile: List[Dict[str, object]] = []
    for label in SPHERE_PROFILE_ZONE_ORDER:
        matching = []
        for profile in profiles:
            zone = _zone_profile_by_label(profile).get(label)
            if zone is not None:
                matching.append(zone)
        if not matching:
            continue
        measured = float(np.median(np.asarray([float(zone.get("measured_log2_luminance", 0.0) or 0.0) for zone in matching], dtype=np.float32)))
        target_profile.append(
            {
                "label": label,
                "display_label": SPHERE_PROFILE_ZONE_DISPLAY[label],
                "measured_log2_luminance": measured,
                "measured_ire": _ire_from_log2_luminance(measured),
            }
        )
    return target_profile


def _shift_zone_profile(zone_measurements: List[Dict[str, object]], delta_stops: float) -> List[Dict[str, object]]:
    shifted: List[Dict[str, object]] = []
    for item in _ordered_zone_profile(zone_measurements):
        measured = float(item.get("measured_log2_luminance", 0.0) or 0.0) + float(delta_stops)
        shifted.append(
            {
                **dict(item),
                "measured_log2_luminance": measured,
                "measured_ire": _ire_from_log2_luminance(measured),
            }
        )
    return shifted


def _derived_profile_scalar(zone_measurements: List[Dict[str, object]]) -> Dict[str, object]:
    sample_display_scalar = _sample_scalar_display_from_profile(zone_measurements)
    if float(sample_display_scalar.get("sample_count", 0.0) or 0.0) <= 0.0:
        return {
            "derived_display_scalar_log2": 0.0,
            "derived_display_scalar_ire": 0.0,
            "derived_exposure_value": 0.0,
            "derived_exposure_value_log2": 0.0,
            "derived_exposure_ire": 0.0,
            "zone_weights": dict(SPHERE_PROFILE_ZONE_WEIGHTS),
            "weight_total": 0.0,
            "derivation_method": "median_sample_log2",
            "derivation_domain": "display_ipp2",
        }
    derived_log2 = float(sample_display_scalar.get("sample_scalar_display_log2", 0.0) or 0.0)
    derived_ire = float(sample_display_scalar.get("sample_scalar_display_ire", 0.0) or 0.0)
    return {
        "derived_display_scalar_log2": derived_log2,
        "derived_display_scalar_ire": derived_ire,
        "derived_exposure_value": derived_log2,
        "derived_exposure_value_log2": derived_log2,
        "derived_exposure_ire": derived_ire,
        "zone_weights": dict(SPHERE_PROFILE_ZONE_WEIGHTS),
        "weight_total": float(sum(float(SPHERE_PROFILE_ZONE_WEIGHTS.get(label, 0.0)) for label in SPHERE_PROFILE_ZONE_ORDER)),
        "derivation_method": "median_sample_log2",
        "derivation_domain": "display_ipp2",
    }


def _profile_audit_summary(zone_residuals: List[Dict[str, object]], zone_measurements: List[Dict[str, object]], target_zone_profile: List[Dict[str, object]]) -> Dict[str, object]:
    measured_profile = [dict(item) for item in _ordered_zone_profile(zone_measurements)]
    target_profile = [dict(item) for item in _ordered_zone_profile(target_zone_profile)]
    measured_ires = [float(item.get("measured_ire", 0.0) or 0.0) for item in measured_profile]
    target_ires = [float(item.get("measured_ire", 0.0) or 0.0) for item in target_profile]
    measured_spread = float(max(measured_ires) - min(measured_ires)) if measured_ires else 0.0
    target_spread = float(max(target_ires) - min(target_ires)) if target_ires else 0.0
    spread_delta = measured_spread - target_spread
    worst_zone = max(zone_residuals, key=lambda item: float(item.get("residual_abs_stops", 0.0) or 0.0), default={})
    worst_label = str(worst_zone.get("label") or "")
    worst_abs = float(worst_zone.get("residual_abs_stops", 0.0) or 0.0)
    if worst_abs <= PROFILE_AUDIT_CONSISTENT_STOPS and abs(spread_delta) <= PROFILE_AUDIT_CONSISTENT_SPREAD_IRE:
        status = "PROFILE CONSISTENT"
        label = "Profile consistent"
        note = "Profile consistent"
        tone = "good"
    elif worst_label == "dark_side" and worst_abs <= PROFILE_AUDIT_REVIEW_STOPS and abs(spread_delta) <= PROFILE_AUDIT_REVIEW_SPREAD_IRE:
        status = "PROFILE NEEDS REVIEW"
        label = "Profile needs review"
        note = "Sample 3 differs slightly"
        tone = "warning"
    elif worst_abs <= PROFILE_AUDIT_REVIEW_STOPS and abs(spread_delta) <= PROFILE_AUDIT_REVIEW_SPREAD_IRE:
        status = "PROFILE NEEDS REVIEW"
        label = "Profile needs review"
        note = "Profile needs review"
        tone = "warning"
    else:
        status = "PROFILE SHAPE MISMATCH"
        label = "Profile mismatch"
        note = "Profile mismatch — verify T-Stop / lighting / ROI"
        tone = "danger"
    return {
        "profile_audit_status": status,
        "profile_audit_label": label,
        "profile_note": note,
        "profile_audit_tone": tone,
        "profile_worst_zone_label": SPHERE_PROFILE_ZONE_DISPLAY.get(worst_label, worst_label or "Zone"),
        "profile_worst_zone_residual_stops": worst_abs,
        "profile_zone_spread_ire": measured_spread,
        "target_zone_spread_ire": target_spread,
        "profile_spread_delta_ire": spread_delta,
    }


def _zone_profile_residual_summary(zone_measurements: List[Dict[str, object]], target_zone_profile: List[Dict[str, object]]) -> Dict[str, object]:
    measured_map = _zone_profile_by_label(zone_measurements)
    target_map = _zone_profile_by_label(target_zone_profile)
    zone_residuals: List[Dict[str, object]] = []
    weighted_components: List[Dict[str, object]] = []
    for label in SPHERE_PROFILE_ZONE_ORDER:
        measured = measured_map.get(label)
        target = target_map.get(label)
        if measured is None or target is None:
            continue
        measured_log2 = float(measured.get("measured_log2_luminance", 0.0) or 0.0)
        target_log2 = float(target.get("measured_log2_luminance", 0.0) or 0.0)
        residual = measured_log2 - target_log2
        weight = float(SPHERE_PROFILE_ZONE_WEIGHTS.get(label, 0.0))
        zone_residuals.append(
            {
                "label": label,
                "display_label": SPHERE_PROFILE_ZONE_DISPLAY.get(label, label),
                "measured_log2_luminance": measured_log2,
                "target_log2_luminance": target_log2,
                "measured_ire": float(measured.get("measured_ire", _ire_from_log2_luminance(measured_log2)) or 0.0),
                "target_ire": float(target.get("measured_ire", _ire_from_log2_luminance(target_log2)) or 0.0),
                "residual_stops": residual,
                "residual_abs_stops": abs(residual),
            }
        )
        weighted_components.append(
            {
                "label": label,
                "display_label": SPHERE_PROFILE_ZONE_DISPLAY.get(label, label),
                "weight": weight,
                "residual_stops": residual,
                "weighted_residual_stops": residual * weight,
            }
        )
    measured_values = np.asarray([float(item["measured_log2_luminance"]) for item in zone_residuals], dtype=np.float32)
    target_values = np.asarray([float(item["target_log2_luminance"]) for item in zone_residuals], dtype=np.float32)
    residual_values = np.asarray([float(item["residual_stops"]) for item in zone_residuals], dtype=np.float32)
    derived_measure = _derived_profile_scalar(zone_measurements)
    derived_target = _derived_profile_scalar(target_zone_profile)
    derived_residual = float(derived_measure.get("derived_exposure_value_log2", 0.0) or 0.0) - float(derived_target.get("derived_exposure_value_log2", 0.0) or 0.0)
    profile_audit = _profile_audit_summary(zone_residuals, zone_measurements, target_zone_profile)
    return {
        "zone_residuals": zone_residuals,
        "aggregate_measurement_log2": float(derived_measure.get("derived_exposure_value_log2", 0.0) or 0.0),
        "aggregate_target_log2": float(derived_target.get("derived_exposure_value_log2", 0.0) or 0.0),
        "aggregate_residual_stops": derived_residual,
        "mean_abs_residual_stops": float(np.mean(np.abs(residual_values))) if zone_residuals else 0.0,
        "max_abs_residual_stops": float(np.max(np.abs(residual_values))) if zone_residuals else 0.0,
        "weighted_zone_components": weighted_components,
        "weighted_residual_stops": float(sum(float(item.get("weighted_residual_stops", 0.0) or 0.0) for item in weighted_components)),
        "zone_weights": dict(SPHERE_PROFILE_ZONE_WEIGHTS),
        "gray_exposure_summary": _format_zone_profile_summary(zone_measurements),
        "target_gray_exposure_summary": _format_zone_profile_summary(target_zone_profile),
        "measured_profile": [dict(item) for item in _ordered_zone_profile(zone_measurements)],
        "target_profile": [dict(item) for item in _ordered_zone_profile(target_zone_profile)],
        "legacy_profile_median_measurement_log2": float(np.median(measured_values)) if zone_residuals else 0.0,
        "legacy_profile_median_target_log2": float(np.median(target_values)) if zone_residuals else 0.0,
        "derived_exposure_value": float(derived_measure.get("derived_exposure_value_log2", 0.0) or 0.0),
        "derived_exposure_value_log2": float(derived_measure.get("derived_exposure_value_log2", 0.0) or 0.0),
        "derived_target_value": float(derived_target.get("derived_exposure_value_log2", 0.0) or 0.0),
        "derived_target_value_log2": float(derived_target.get("derived_exposure_value_log2", 0.0) or 0.0),
        "derived_exposure_ire": float(derived_measure.get("derived_exposure_ire", 0.0) or 0.0),
        "derived_target_ire": float(derived_target.get("derived_exposure_ire", 0.0) or 0.0),
        "derived_residual_stops": derived_residual,
        "derived_residual_abs_stops": abs(derived_residual),
        "derived_exposure_offset_stops": derived_residual,
        **profile_audit,
    }


def _zone_residual_rank(summary: Dict[str, object]) -> tuple[float, float, float]:
    return (
        abs(float(summary.get("derived_residual_stops", summary.get("aggregate_residual_stops", 0.0)) or 0.0)),
        float(summary.get("max_abs_residual_stops", 0.0) or 0.0),
        float(summary.get("mean_abs_residual_stops", 0.0) or 0.0),
    )


def _zone_residual_improved(candidate: Dict[str, object], best: Dict[str, object]) -> bool:
    cand_rank = _zone_residual_rank(candidate)
    best_rank = _zone_residual_rank(best)
    if cand_rank[0] + IPP2_CLOSED_LOOP_MIN_IMPROVEMENT_STOPS < best_rank[0]:
        return True
    if abs(cand_rank[0] - best_rank[0]) <= IPP2_CLOSED_LOOP_MIN_IMPROVEMENT_STOPS and cand_rank[1] + IPP2_CLOSED_LOOP_MIN_IMPROVEMENT_STOPS < best_rank[1]:
        return True
    return False


def _at_closed_loop_correction_limit(correction_stops: float) -> bool:
    return abs(float(correction_stops)) >= (IPP2_CLOSED_LOOP_MAX_CORRECTION_STOPS - 1e-6)


def _corrected_residual_status(abs_residual: float) -> Dict[str, object]:
    value = float(abs_residual)
    if value <= CORRECTED_RESIDUAL_PASS_STOPS:
        return {
            "status": "within_tolerance",
            "label": "Within tolerance",
            "tone": "good",
        }
    if value <= CORRECTED_RESIDUAL_REVIEW_STOPS:
        return {
            "status": "review",
            "label": "Needs review",
            "tone": "warning",
        }
    return {
        "status": "outside_tolerance",
        "label": "Outside tolerance",
        "tone": "danger",
    }


def _corrected_residual_tolerance_model() -> Dict[str, object]:
    return {
        "metric": "absolute corrected render residual in log2 stops",
        "pass_threshold_stops": CORRECTED_RESIDUAL_PASS_STOPS,
        "review_threshold_stops": CORRECTED_RESIDUAL_REVIEW_STOPS,
        "rationale": (
            "Post-correction residuals should stay below the existing sample-stability caution band. "
            "Residuals above 0.20 stops remain visibly mismatched on the contact sheet and are treated as failures."
        ),
    }


def _ipp2_validation_status(abs_residual: float) -> Dict[str, object]:
    value = float(abs_residual)
    if value <= IPP2_VALIDATION_PASS_STOPS:
        return {"status": "PASS", "tone": "good"}
    if value <= IPP2_VALIDATION_REVIEW_STOPS:
        return {"status": "REVIEW", "tone": "warning"}
    return {"status": "FAIL", "tone": "danger"}


def round_to_standard_stop_fraction(stops: float) -> float:
    magnitude = abs(float(stops))
    step_sizes = (1.0, 0.5, 1.0 / 3.0, 0.25)
    candidates: List[tuple[float, float, int]] = []
    for index, step in enumerate(step_sizes):
        candidate = round(magnitude / step) * step
        candidates.append((abs(candidate - magnitude), candidate, index))
    _, rounded_magnitude, _ = min(candidates, key=lambda item: (item[0], item[2], item[1]))
    return -rounded_magnitude if float(stops) < 0.0 else rounded_magnitude


def format_stop_string(stops: float) -> str:
    rounded = abs(round_to_standard_stop_fraction(stops))
    whole = int(math.floor(rounded + 1e-6))
    fractional_value = rounded - float(whole)
    if fractional_value >= 0.99:
        whole += 1
        fractional_value = 0.0
    fraction_options = (
        (0.0, ""),
        (0.25, "1/4"),
        (1.0 / 3.0, "1/3"),
        (0.5, "1/2"),
        (2.0 / 3.0, "2/3"),
        (0.75, "3/4"),
    )
    _, fraction_label = min(
        ((abs(fractional_value - candidate), label) for candidate, label in fraction_options),
        key=lambda item: item[0],
    )
    if whole and fraction_label:
        text = f"{whole} {fraction_label}"
    elif whole:
        text = str(whole)
    else:
        text = fraction_label or "0"
    suffix = "stop" if (whole == 0 or (whole == 1 and not fraction_label)) else "stops"
    return f"{text} {suffix}"


def _residual_direction(residual_stops: float) -> str:
    if residual_stops > 0.0:
        return "close"
    if residual_stops < 0.0:
        return "open"
    return "hold"


def _correction_direction(correction_stops: float) -> str:
    if correction_stops > 0.0:
        return "open"
    if correction_stops < 0.0:
        return "close"
    return "hold"


def _operator_guidance_for_correction(
    *,
    correction_stops: float,
    residual_stops: float,
    validation_status: str,
) -> Dict[str, object]:
    exact_correction = float(correction_stops)
    rounded_correction = float(round_to_standard_stop_fraction(exact_correction))
    correction_direction = _correction_direction(rounded_correction)
    residual_direction = _residual_direction(float(residual_stops))
    rounded_text = format_stop_string(rounded_correction)
    if correction_direction == "hold":
        suggested_action = "No aperture adjustment required"
    else:
        suggested_action = f"{correction_direction.capitalize()} aperture by {rounded_text}"
    correction_range = "normal"
    operator_status = str(validation_status or "REVIEW")
    notes: List[str] = []
    if abs(exact_correction) >= WARNING_CORRECTION_STOPS:
        correction_range = "outlier"
        operator_status = "OUTLIER"
        notes.append("Camera exceeds safe digital correction range.")
        notes.append("Verify T-Stop and re-run calibration.")
    elif abs(exact_correction) > SAFE_CORRECTION_STOPS:
        correction_range = "warning"
        notes.append("Large exposure correction. Verify T-Stop before final push.")
    else:
        notes.append("Correction stays within the normal digital adjustment range.")
    if operator_status != "OUTLIER":
        if validation_status == "FAIL":
            notes.append(
                "IPP2 residual remains outside tolerance after correction; the image is still "
                f"too {'bright' if residual_direction == 'close' else 'dark' if residual_direction == 'open' else 'close to target'}."
            )
        elif validation_status == "REVIEW":
            notes.append("Residual is close, but verify the gray match on the next run.")
        elif validation_status == "PASS":
            notes.append("Residual is within IPP2 tolerance.")
    return {
        "status": operator_status,
        "validation_status": str(validation_status or "REVIEW"),
        "correction_range": correction_range,
        "correction_stops": exact_correction,
        "rounded_stops": rounded_correction,
        "rounded_stop_string": rounded_text,
        "direction": correction_direction,
        "residual_direction": residual_direction,
        "suggested_action": suggested_action,
        "notes": " ".join(notes).strip(),
    }


def _ipp2_tolerance_model() -> Dict[str, object]:
    return {
        "metric": "absolute corrected IPP2 residual in log2 stops",
        "pass_threshold_stops": IPP2_VALIDATION_PASS_STOPS,
        "review_threshold_stops": IPP2_VALIDATION_REVIEW_STOPS,
        "rationale": (
            "R3DMatch is judged in monitoring space. Corrected cameras must match within 0.05 stops in the "
            "IPP2 Medium / Medium review domain to pass acceptance."
        ),
    }


def _ipp2_validation_preview_settings() -> Dict[str, object]:
    return _normalize_preview_settings(
        preview_mode="monitoring",
        preview_output_space="BT.709",
        preview_output_gamma="BT.1886",
        preview_highlight_rolloff="medium",
        preview_shadow_rolloff="medium",
        preview_lut=None,
    )


def _ipp2_closed_loop_target(
    *,
    strategy_key: str,
    reference_clip_id: Optional[str],
    rows: List[Dict[str, object]],
    anchor_mode: Optional[str] = None,
    anchor_source: Optional[str] = None,
    anchor_scalar_value: Optional[float] = None,
    anchor_ire_summary: Optional[str] = None,
) -> Dict[str, object]:
    trusted_rows = [
        item for item in rows
        if str(item.get("trust_class") or "") in {"TRUSTED", "USE_WITH_CAUTION"}
        and list(item.get("original_ipp2_zone_profile") or [])
    ]
    fallback_rows = [item for item in rows if list(item.get("original_ipp2_zone_profile") or [])]
    if strategy_key == "manual_target" and anchor_scalar_value is not None:
        base_rows = trusted_rows or fallback_rows
        base_profile = _median_zone_profile(base_rows, field_name="original_ipp2_zone_profile")
        base_scalar = _derived_profile_scalar(base_profile)
        delta_stops = float(anchor_scalar_value) - float(base_scalar.get("derived_exposure_value_log2", 0.0) or 0.0)
        target_profile = _shift_zone_profile(base_profile, delta_stops)
        derived_target = _derived_profile_scalar(target_profile)
        return {
            "target_log2": float(anchor_scalar_value),
            "target_zone_profile": target_profile,
            "target_profile_summary": _format_zone_profile_summary(target_profile),
            "target_source": "manual_scalar_target_adjusted_from_original_median_profile",
            "reference_clip_id": str(reference_clip_id or ""),
            "anchor_mode": str(anchor_mode or "manual_target"),
            "anchor_source": str(anchor_source or ""),
            "anchor_scalar_value": float(anchor_scalar_value),
            "anchor_ire_summary": str(anchor_ire_summary or _format_manual_anchor_ire_summary(anchor_log2=float(anchor_scalar_value), manual_target_ire=None)),
            "target_profile_scalar": float(derived_target.get("derived_exposure_value_log2", anchor_scalar_value) or anchor_scalar_value),
        }
    if strategy_key in {"optimal_exposure", "hero_camera", "manual"} and reference_clip_id:
        reference_row = next((item for item in fallback_rows if str(item.get("clip_id") or "") == str(reference_clip_id)), None)
        if reference_row is not None:
            target_profile = [dict(item) for item in list(reference_row.get("original_ipp2_zone_profile") or [])]
            derived_target = _derived_profile_scalar(target_profile)
            return {
                "target_log2": float(derived_target.get("derived_exposure_value_log2", 0.0) or 0.0),
                "target_zone_profile": target_profile,
                "target_profile_summary": _format_zone_profile_summary(target_profile),
                "target_source": "reference_camera_original_ipp2_profile",
                "reference_clip_id": str(reference_row.get("clip_id") or ""),
                "anchor_mode": str(anchor_mode or ("manual_clip" if strategy_key == "manual" else "hero_camera" if strategy_key == "hero_camera" else "hero_clip")),
                "anchor_source": str(anchor_source or reference_row.get("clip_id") or ""),
                "anchor_scalar_value": float(derived_target.get("derived_exposure_value_log2", 0.0) or 0.0),
                "anchor_ire_summary": str(anchor_ire_summary or _format_zone_profile_summary(target_profile)),
            }
    selected_rows = trusted_rows or fallback_rows
    target_profile = _median_zone_profile(selected_rows, field_name="original_ipp2_zone_profile")
    derived_target = _derived_profile_scalar(target_profile)
    return {
        "target_log2": float(derived_target.get("derived_exposure_value_log2", 0.0) or 0.0),
        "target_zone_profile": target_profile,
        "target_profile_summary": _format_zone_profile_summary(target_profile),
        "target_source": "trusted_camera_median_original_ipp2_profile" if trusted_rows else "all_camera_median_original_ipp2_profile",
        "reference_clip_id": str(reference_clip_id or ""),
        "anchor_mode": str(anchor_mode or "median"),
        "anchor_source": str(anchor_source or "Group median"),
        "anchor_scalar_value": float(derived_target.get("derived_exposure_value_log2", 0.0) or 0.0),
        "anchor_ire_summary": str(anchor_ire_summary or _format_zone_profile_summary(target_profile)),
    }


def _ipp2_closed_loop_iteration_path(
    preview_root: Path,
    *,
    clip_id: str,
    strategy_key: str,
    run_id: str,
    iteration: int,
    correction_stops: float,
) -> Path:
    correction_token = f"{float(correction_stops):+.4f}".replace("+", "p").replace("-", "m").replace(".", "_")
    filename = f"{clip_id}.both.review.{normalize_target_strategy_name(strategy_key)}.{run_id}.iter{iteration:02d}.{correction_token}.jpg"
    return preview_root / filename


def _ipp2_closed_loop_next_correction(
    *,
    current_correction: float,
    current_residual: float,
    previous_correction: Optional[float],
    previous_residual: Optional[float],
) -> float:
    candidate = current_correction - (current_residual * IPP2_CLOSED_LOOP_DAMPING)
    if previous_correction is not None and previous_residual is not None:
        denominator = current_residual - previous_residual
        if abs(denominator) > 1e-6 and abs(current_correction - previous_correction) > 1e-6:
            secant_candidate = current_correction - (
                current_residual * (current_correction - previous_correction) / denominator
            )
            candidate = current_correction + ((secant_candidate - current_correction) * IPP2_CLOSED_LOOP_DAMPING)
    if abs(candidate - current_correction) < IPP2_CLOSED_LOOP_MIN_STEP_STOPS:
        step = IPP2_CLOSED_LOOP_MIN_STEP_STOPS if abs(current_residual) < IPP2_CLOSED_LOOP_MIN_STEP_STOPS else min(abs(current_residual), 0.5)
        candidate = current_correction - (np.sign(current_residual) * step)
    return float(np.clip(candidate, -IPP2_CLOSED_LOOP_MAX_CORRECTION_STOPS, IPP2_CLOSED_LOOP_MAX_CORRECTION_STOPS))


def _closed_loop_strategy_metrics(rows: List[Dict[str, object]]) -> Dict[str, object]:
    validation_residuals = [
        float(item.get("ipp2_residual_abs_stops", 0.0) or 0.0)
        for item in rows
    ]
    status_counts = {
        status: sum(1 for item in rows if str(item.get("status") or "") == status)
        for status in ("PASS", "REVIEW", "FAIL")
    }
    return {
        "camera_count": len(rows),
        "best_residual": min(validation_residuals) if validation_residuals else None,
        "median_residual": float(np.median(np.asarray(validation_residuals, dtype=np.float32))) if validation_residuals else None,
        "max_residual": max(validation_residuals) if validation_residuals else None,
        "status_counts": status_counts,
        "all_within_tolerance": all(value <= IPP2_VALIDATION_PASS_STOPS for value in validation_residuals) if validation_residuals else False,
    }


def _recommend_strategy_from_closed_loop(strategy_summaries: List[Dict[str, object]]) -> str:
    if not strategy_summaries:
        return ""
    def rank(item: Dict[str, object]) -> tuple[float, float, float, int]:
        metrics = dict(item.get("metrics") or {})
        counts = dict(metrics.get("status_counts") or {})
        priority = STRATEGY_ORDER.index(str(item.get("strategy_key") or "")) if str(item.get("strategy_key") or "") in STRATEGY_ORDER else len(STRATEGY_ORDER)
        return (
            float(counts.get("FAIL", 0) or 0),
            float(metrics.get("max_residual", 0.0) or 0.0),
            float(metrics.get("median_residual", 0.0) or 0.0),
            priority,
        )
    winner = min(strategy_summaries, key=rank)
    return str(winner.get("strategy_key") or "")




def _run_ipp2_closed_loop_solver(
    *,
    input_path: str,
    out_root: Path,
    analysis_records: List[Dict[str, object]],
    strategy_payloads: List[Dict[str, object]],
    redline_capabilities: Dict[str, object],
    run_id: str,
    explicit_anchor_strategy_key: Optional[str] = None,
    progress_path: Optional[str] = None,
    original_preview_by_clip: Optional[Dict[str, str]] = None,
    original_measurements_by_clip: Optional[Dict[str, Dict[str, object]]] = None,
) -> Dict[str, object]:
    solver_started_at = time.perf_counter()
    ipp2_preview_settings = _ipp2_validation_preview_settings()
    preview_root = Path(input_path).expanduser().resolve() / "previews" / "_ipp2_closed_loop"
    preview_root.mkdir(parents=True, exist_ok=True)
    loop_rmd_root = Path(input_path).expanduser().resolve() / "review_rmd" / "strategies_ipp2_closed_loop_iterations"
    loop_rmd_root.mkdir(parents=True, exist_ok=True)
    initial_preview_paths = generate_preview_stills(
        input_path,
        analysis_records=analysis_records,
        previews_dir=str(preview_root),
        preview_settings=ipp2_preview_settings,
        redline_capabilities=redline_capabilities,
        strategy_payloads=strategy_payloads,
        run_id=f"{run_id}.closed_loop_init",
        strategy_rmd_root=str(Path(input_path).expanduser().resolve() / "review_rmd" / "strategies_ipp2_closed_loop_init"),
        render_originals=False,
        original_preview_by_clip=original_preview_by_clip,
    )
    for clip_id, original_path in dict(original_preview_by_clip or {}).items():
        initial_preview_paths.setdefault(str(clip_id), {}).setdefault("strategies", {})
        initial_preview_paths[str(clip_id)]["original"] = str(original_path)
    emit_review_progress(
        progress_path,
        phase="closed_loop_initial_previews",
        detail="Prepared initial IPP2 closed-loop previews.",
        stage_label="Rendering previews",
        clip_count=len(analysis_records),
        elapsed_seconds=time.perf_counter() - solver_started_at,
        review_mode="full_contact_sheet",
    )
    redline_executable = _resolve_redline_executable()
    record_by_clip = {str(record.get("clip_id") or ""): dict(record) for record in analysis_records}
    original_measure_cache: Dict[str, Dict[str, object]] = {
        str(clip_id): copy.deepcopy(dict(measurement))
        for clip_id, measurement in dict(original_measurements_by_clip or {}).items()
        if measurement
    }
    detected_sphere_roi_cache: Dict[str, Dict[str, float]] = {}
    for clip_id, measurement in original_measure_cache.items():
        detected_roi = dict(measurement.get("detected_sphere_roi") or {})
        if detected_roi:
            detected_sphere_roi_cache[clip_id] = detected_roi
    render_cache: Dict[tuple[str, str, float], Dict[str, object]] = {}
    refined_payloads = copy.deepcopy(strategy_payloads)
    closed_loop_strategy_summaries: List[Dict[str, object]] = []
    preview_paths = copy.deepcopy(initial_preview_paths)

    emit_review_progress(
        progress_path,
        phase="closed_loop_solver_start",
        detail=f"Starting IPP2 closed-loop solver for {len(refined_payloads)} strategy payload(s).",
        stage_label="Closed-loop solve",
        clip_count=len(analysis_records),
        elapsed_seconds=time.perf_counter() - solver_started_at,
        review_mode="full_contact_sheet",
    )
    for strategy_index, strategy_payload in enumerate(refined_payloads, start=1):
        strategy_key = str(strategy_payload.get("strategy_key") or "")
        emit_review_progress(
            progress_path,
            phase="closed_loop_strategy_start",
            detail=f"Solving strategy {strategy_key}.",
            stage_label="Closed-loop solve",
            clip_index=strategy_index,
            clip_count=len(refined_payloads),
            current_clip_id=strategy_key,
            elapsed_seconds=time.perf_counter() - solver_started_at,
            review_mode="full_contact_sheet",
        )
        strategy_exposure_summary = _exposure_summary(list(strategy_payload.get("clips") or []))
        selection_diagnostics = dict(strategy_payload.get("selection_diagnostics") or {})
        primary_cluster_indices = {int(index) for index in ((selection_diagnostics.get("primary_cluster") or {}).get("indices") or [])}
        screened_candidates_by_clip = {
            str(item.get("clip_id") or ""): dict(item)
            for item in list(selection_diagnostics.get("screened_candidates") or [])
            if str(item.get("clip_id") or "").strip()
        }
        initial_rows: List[Dict[str, object]] = []
        for index, clip in enumerate(list(strategy_payload.get("clips") or [])):
            clip_id = str(clip.get("clip_id") or "")
            original_frame = str(initial_preview_paths.get(clip_id, {}).get("original") or "")
            corrected_frame = str(initial_preview_paths.get(clip_id, {}).get("strategies", {}).get(strategy_key, {}).get("both") or "")
            if original_frame and clip_id not in original_measure_cache and Path(original_frame).exists():
                original_measure_cache[clip_id] = _measure_rendered_preview_roi_ipp2(original_frame, clip.get("calibration_roi"))
                detected_roi = dict(original_measure_cache[clip_id].get("detected_sphere_roi") or {})
                if detected_roi:
                    detected_sphere_roi_cache[clip_id] = detected_roi
            original_measure = dict(original_measure_cache.get(clip_id) or {})
            corrected_measure = (
                _measure_rendered_preview_roi_ipp2(
                    corrected_frame,
                    clip.get("calibration_roi"),
                    sphere_roi_override=detected_sphere_roi_cache.get(clip_id),
                )
                if corrected_frame and Path(corrected_frame).exists()
                else {}
            )
            trust_details = _camera_trust_details(
                clip_id=clip_id,
                confidence=float(clip.get("confidence", 0.0) or 0.0),
                sample_log2_spread=float(clip.get("neutral_sample_log2_spread", 0.0) or 0.0),
                sample_chroma_spread=float(clip.get("neutral_sample_chromaticity_spread", 0.0) or 0.0),
                measured_log2=float(clip.get("measured_log2_luminance_monitoring", clip.get("measured_log2_luminance", 0.0)) or 0.0),
                final_offset=float(clip.get("exposure_offset_stops", 0.0) or 0.0),
                exposure_summary=strategy_exposure_summary,
                in_primary_cluster=(not primary_cluster_indices or index in primary_cluster_indices),
                screened_reasons=[str(item) for item in (screened_candidates_by_clip.get(clip_id, {}) or {}).get("reasons", []) if str(item).strip()],
            )
            initial_rows.append(
                {
                    "clip_id": clip_id,
                    "camera_id": _camera_id_from_clip_id(clip_id),
                    "camera_label": _camera_label_for_reporting(clip_id),
                    "source_path": str(clip.get("source_path") or ""),
                    "frame_index": int((record_by_clip.get(clip_id, {}).get("sample_plan") or {}).get("start_frame", 0) or 0),
                    "calibration_roi": clip.get("calibration_roi"),
                    "trust_class": str(trust_details.get("trust_class") or "UNTRUSTED"),
                    "reference_use": str(trust_details.get("reference_use") or "Included"),
                    "trust_reason": str(trust_details.get("trust_reason") or ""),
                    "is_hero_camera": bool(clip.get("is_hero_camera")),
                    "initial_exposure_correction_stops": float(clip.get("exposure_offset_stops", 0.0) or 0.0),
                    "initial_ipp2_value_log2": float(corrected_measure.get("measured_log2_luminance_monitoring", 0.0) or 0.0),
                    "initial_ipp2_zone_profile": [dict(item) for item in list(corrected_measure.get("zone_measurements") or [])],
                    "initial_gray_exposure_summary": str(corrected_measure.get("gray_exposure_summary") or "n/a"),
                    "initial_sphere_detection_source": str(corrected_measure.get("sphere_roi_source") or "failed"),
                    "initial_sphere_detection_confidence": float(corrected_measure.get("sphere_detection_confidence", 0.0) or 0.0),
                    "initial_sphere_detection_label": str(corrected_measure.get("sphere_detection_label") or "FAILED"),
                    "initial_detected_sphere_roi": dict(corrected_measure.get("detected_sphere_roi") or {}),
                    "original_ipp2_value_log2": float(original_measure.get("measured_log2_luminance_monitoring", 0.0) or 0.0),
                    "original_ipp2_zone_profile": [dict(item) for item in list(original_measure.get("zone_measurements") or [])],
                    "original_gray_exposure_summary": str(original_measure.get("gray_exposure_summary") or "n/a"),
                    "original_sphere_detection_source": str(original_measure.get("sphere_roi_source") or "failed"),
                    "original_sphere_detection_confidence": float(original_measure.get("sphere_detection_confidence", 0.0) or 0.0),
                    "original_sphere_detection_label": str(original_measure.get("sphere_detection_label") or "FAILED"),
                    "original_detected_sphere_roi": dict(original_measure.get("detected_sphere_roi") or {}),
                    "detection_failed": bool(original_measure.get("detection_failed")) or bool(corrected_measure.get("detection_failed")),
                    "initial_image_path": corrected_frame,
                    "color_cdl": copy.deepcopy(clip.get("color_cdl")),
                }
            )

        target_summary = _ipp2_closed_loop_target(
            strategy_key=strategy_key,
            reference_clip_id=str(strategy_payload.get("reference_clip_id") or strategy_payload.get("hero_clip_id") or ""),
            rows=initial_rows,
            anchor_mode=str(strategy_payload.get("anchor_mode") or ""),
            anchor_source=str(strategy_payload.get("anchor_source") or ""),
            anchor_scalar_value=(
                float(strategy_payload.get("anchor_scalar_value"))
                if strategy_payload.get("anchor_scalar_value") is not None
                else None
            ),
            anchor_ire_summary=str(strategy_payload.get("anchor_ire_summary") or ""),
        )
        ipp2_target_log2 = float(target_summary["target_log2"])
        target_zone_profile = [dict(item) for item in list(target_summary.get("target_zone_profile") or [])]
        row_results: List[Dict[str, object]] = []
        clip_lookup = {str(item.get("clip_id") or ""): item for item in list(strategy_payload.get("clips") or [])}

        for row_index, row in enumerate(initial_rows, start=1):
            clip_id = str(row["clip_id"])
            emit_review_progress(
                progress_path,
                phase="closed_loop_clip_start",
                detail=f"Evaluating {clip_id} for {strategy_key}.",
                stage_label="Closed-loop solve",
                clip_index=row_index,
                clip_count=len(initial_rows),
                current_clip_id=clip_id,
                elapsed_seconds=time.perf_counter() - solver_started_at,
                review_mode="full_contact_sheet",
                extra={"strategy_key": strategy_key},
            )
            clip_payload = clip_lookup[clip_id]
            current_correction = float(row["initial_exposure_correction_stops"])
            current_path = str(row["initial_image_path"] or "")
            initial_summary = (
                _failed_detection_summary(target_zone_profile)
                if bool(row.get("detection_failed"))
                else _zone_profile_residual_summary(list(row.get("initial_ipp2_zone_profile") or []), target_zone_profile)
            )
            current_measure = float(initial_summary.get("derived_exposure_value_log2", row["initial_ipp2_value_log2"]) or 0.0)
            current_residual = float(initial_summary.get("derived_residual_stops", 0.0) or 0.0)
            best_summary = copy.deepcopy(initial_summary)
            best_measure = current_measure
            best_path = current_path
            best_correction = current_correction
            previous_correction: Optional[float] = None
            previous_residual: Optional[float] = None
            history: List[Dict[str, object]] = [{
                "iteration": 0,
                "exposure_correction_stops": current_correction,
                "ipp2_value_log2": current_measure,
                "ipp2_residual_stops": current_residual,
                "derived_residual_abs_stops": float(initial_summary.get("derived_residual_abs_stops", abs(current_residual)) or abs(current_residual)),
                "profile_max_residual_stops": float(initial_summary.get("max_abs_residual_stops", abs(current_residual)) or abs(current_residual)),
                "gray_exposure_summary": str(initial_summary.get("gray_exposure_summary") or row.get("initial_gray_exposure_summary") or "n/a"),
                "target_gray_exposure_summary": str(initial_summary.get("target_gray_exposure_summary") or target_summary.get("target_profile_summary") or "n/a"),
                "zone_profile": copy.deepcopy(initial_summary.get("measured_profile") or row.get("initial_ipp2_zone_profile") or []),
                "zone_residuals": copy.deepcopy(initial_summary.get("zone_residuals") or []),
                "weighted_zone_components": copy.deepcopy(initial_summary.get("weighted_zone_components") or []),
                "weighted_residual_stops": float(initial_summary.get("weighted_residual_stops", current_residual) or current_residual),
                "image_path": current_path,
                "correction_application_method": "direct_redline_flags",
                "direct_exposure_parameter": REDLINE_DIRECT_EXPOSURE_PARAMETER,
            }]
            if not bool(row.get("is_hero_camera")) and not bool(row.get("detection_failed")):
                for iteration in range(1, IPP2_CLOSED_LOOP_MAX_ITERATIONS + 1):
                    if float(best_summary.get("derived_residual_abs_stops", abs(best_summary.get("derived_residual_stops", 0.0))) or 0.0) <= IPP2_VALIDATION_PASS_STOPS:
                        break
                    next_correction = _ipp2_closed_loop_next_correction(
                        current_correction=current_correction,
                        current_residual=current_residual,
                        previous_correction=previous_correction,
                        previous_residual=previous_residual,
                    )
                    cache_key = (clip_id, strategy_key, round(float(next_correction), 6))
                    cached = render_cache.get(cache_key)
                    if cached is None:
                        iteration_path = _ipp2_closed_loop_iteration_path(
                            preview_root,
                            clip_id=clip_id,
                            strategy_key=strategy_key,
                            run_id=run_id,
                            iteration=iteration,
                            correction_stops=next_correction,
                        )
                        render = render_preview_frame(
                            row["source_path"],
                            str(iteration_path),
                            frame_index=int(row["frame_index"]),
                            redline_executable=redline_executable,
                            redline_capabilities=redline_capabilities,
                            preview_settings=ipp2_preview_settings,
                            use_as_shot_metadata=True,
                            exposure=float(next_correction),
                            color_cdl=None,
                            color_method=None,
                            rmd_path=None,
                            use_rmd_mode=1,
                        )
                        if int(render["returncode"]) != 0:
                            raise RuntimeError(
                                f"Closed-loop IPP2 render failed for {clip_id} ({strategy_key}) at iteration {iteration}. "
                                f"Command: {shlex.join(render['command'])}. STDERR: {str(render['stderr']).strip()}"
                            )
                        measured = _measure_rendered_preview_roi_ipp2(
                            str(render["output_path"]),
                            row.get("calibration_roi"),
                            sphere_roi_override=detected_sphere_roi_cache.get(clip_id),
                        )
                        cached = {
                            "image_path": str(render["output_path"]),
                            "measure": measured,
                            "command": render["command"],
                            "correction_application_method": "direct_redline_flags",
                            "direct_exposure_parameter": REDLINE_DIRECT_EXPOSURE_PARAMETER,
                            "rmd_path": "",
                        }
                        render_cache[cache_key] = cached
                    measured_payload = dict(cached.get("measure") or {})
                    next_summary = (
                        _failed_detection_summary(target_zone_profile)
                        if bool(measured_payload.get("detection_failed"))
                        else _zone_profile_residual_summary(list(measured_payload.get("zone_measurements") or []), target_zone_profile)
                    )
                    next_measure = float(next_summary.get("aggregate_measurement_log2", measured_payload.get("measured_log2_luminance_monitoring", 0.0)) or 0.0)
                    next_residual = float(next_summary.get("aggregate_residual_stops", 0.0) or 0.0)
                    history.append({
                        "iteration": iteration,
                        "exposure_correction_stops": float(next_correction),
                        "ipp2_value_log2": next_measure,
                        "ipp2_residual_stops": next_residual,
                        "derived_residual_abs_stops": float(next_summary.get("derived_residual_abs_stops", abs(next_residual)) or abs(next_residual)),
                        "profile_max_residual_stops": float(next_summary.get("max_abs_residual_stops", abs(next_residual)) or abs(next_residual)),
                        "gray_exposure_summary": str(next_summary.get("gray_exposure_summary") or measured_payload.get("gray_exposure_summary") or "n/a"),
                        "target_gray_exposure_summary": str(next_summary.get("target_gray_exposure_summary") or target_summary.get("target_profile_summary") or "n/a"),
                        "zone_profile": copy.deepcopy(next_summary.get("measured_profile") or measured_payload.get("zone_measurements") or []),
                        "zone_residuals": copy.deepcopy(next_summary.get("zone_residuals") or []),
                        "weighted_zone_components": copy.deepcopy(next_summary.get("weighted_zone_components") or []),
                        "weighted_residual_stops": float(next_summary.get("weighted_residual_stops", next_residual) or next_residual),
                        "image_path": str(cached.get("image_path") or ""),
                        "render_command": copy.deepcopy(cached.get("command") or []),
                        "correction_application_method": str(cached.get("correction_application_method") or "rmd"),
                        "direct_exposure_parameter": str(cached.get("direct_exposure_parameter") or ""),
                        "rmd_path": str(cached.get("rmd_path") or ""),
                    })
                    if _zone_residual_improved(next_summary, best_summary):
                        best_summary = copy.deepcopy(next_summary)
                        best_measure = next_measure
                        best_path = str(cached.get("image_path") or "")
                        best_correction = float(next_correction)
                    if bool(measured_payload.get("detection_failed")):
                        break
                    if float(next_summary.get("derived_residual_abs_stops", abs(next_residual)) or abs(next_residual)) <= IPP2_VALIDATION_PASS_STOPS:
                        best_summary = copy.deepcopy(next_summary)
                        best_measure = next_measure
                        best_path = str(cached.get("image_path") or "")
                        best_correction = float(next_correction)
                        break
                    if _at_closed_loop_correction_limit(next_correction):
                        current_correction = float(next_correction)
                        current_residual = next_residual
                        if _zone_residual_improved(next_summary, best_summary):
                            best_summary = copy.deepcopy(next_summary)
                            best_measure = next_measure
                            best_path = str(cached.get("image_path") or "")
                            best_correction = float(next_correction)
                        break
                    previous_correction = current_correction
                    previous_residual = current_residual
                    current_correction = float(next_correction)
                    current_residual = next_residual

            status = _ipp2_validation_status(float(best_summary.get("derived_residual_abs_stops", abs(best_summary.get("derived_residual_stops", 0.0))) or 0.0))
            best_zone_profile = [dict(item) for item in list(best_summary.get("measured_profile") or row.get("initial_ipp2_zone_profile") or [])]
            clip_payload["initial_exposure_offset_stops"] = float(row["initial_exposure_correction_stops"])
            clip_payload["exposure_offset_stops"] = float(best_correction)
            clip_payload["ipp2_closed_loop_target_log2"] = float(ipp2_target_log2)
            clip_payload["ipp2_closed_loop_target_zone_profile"] = copy.deepcopy(target_zone_profile)
            clip_payload["ipp2_closed_loop_target_gray_exposure_summary"] = str(target_summary.get("target_profile_summary") or "n/a")
            clip_payload["ipp2_closed_loop_initial_residual_stops"] = float(initial_summary.get("derived_residual_stops", 0.0) or 0.0)
            clip_payload["ipp2_closed_loop_final_residual_stops"] = float(best_summary.get("derived_residual_stops", 0.0) or 0.0)
            clip_payload["ipp2_closed_loop_initial_profile_max_residual_stops"] = float(initial_summary.get("max_abs_residual_stops", 0.0) or 0.0)
            clip_payload["ipp2_closed_loop_final_profile_max_residual_stops"] = float(best_summary.get("max_abs_residual_stops", 0.0) or 0.0)
            clip_payload["derived_exposure_value"] = float(best_summary.get("derived_exposure_value_log2", 0.0) or 0.0)
            clip_payload["derived_exposure_offset_stops"] = float(best_summary.get("derived_residual_stops", 0.0) or 0.0)
            clip_payload["profile_audit_status"] = str(best_summary.get("profile_audit_status") or "PROFILE NEEDS REVIEW")
            clip_payload["profile_note"] = str(best_summary.get("profile_note") or "Profile needs review")
            clip_payload["ipp2_closed_loop_iterations"] = copy.deepcopy(history)
            emit_review_progress(
                progress_path,
                phase="closed_loop_clip_complete",
                detail=f"Closed-loop result ready for {clip_id}.",
                stage_label="Closed-loop solve",
                clip_index=row_index,
                clip_count=len(initial_rows),
                current_clip_id=clip_id,
                elapsed_seconds=time.perf_counter() - solver_started_at,
                review_mode="full_contact_sheet",
                extra={
                    "strategy_key": strategy_key,
                    "final_residual_stops": float(best_summary.get("derived_residual_stops", 0.0) or 0.0),
                },
            )
            clip_payload["gray_exposure_summary"] = str(best_summary.get("gray_exposure_summary") or row.get("initial_gray_exposure_summary") or "n/a")
            clip_payload["sphere_zone_profile_monitoring"] = copy.deepcopy(best_zone_profile)
            clip_payload["bright_ire"] = _zone_ire_from_profile(best_zone_profile, "bright_side")
            clip_payload["center_ire"] = _zone_ire_from_profile(best_zone_profile, "center")
            clip_payload["dark_ire"] = _zone_ire_from_profile(best_zone_profile, "dark_side")
            clip_payload["sample_1_ire"] = float(clip_payload["bright_ire"])
            clip_payload["sample_2_ire"] = float(clip_payload["center_ire"])
            clip_payload["sample_3_ire"] = float(clip_payload["dark_ire"])
            clip_payload["top_ire"] = float(clip_payload["bright_ire"])
            clip_payload["mid_ire"] = float(clip_payload["center_ire"])
            clip_payload["bottom_ire"] = float(clip_payload["dark_ire"])
            preview_paths.setdefault(clip_id, {}).setdefault("strategies", {}).setdefault(strategy_key, {})
            preview_paths[clip_id]["strategies"][strategy_key]["both"] = str(best_path)
            if not _color_preview_policy()["enabled"]:
                preview_paths[clip_id]["strategies"][strategy_key]["exposure"] = str(best_path)
            row_results.append(
                {
                    "camera_id": row["camera_id"],
                    "camera_label": row["camera_label"],
                    "clip_id": clip_id,
                    "strategy_key": strategy_key,
                    "strategy_label": str(strategy_payload.get("strategy_label") or strategy_key),
                    "reference_use": str(row["reference_use"]),
                    "trust_class": str(row["trust_class"]),
                    "trust_reason": str(row["trust_reason"]),
                    "ipp2_original_value_log2": float(row["original_ipp2_value_log2"]),
                    "ipp2_original_zone_profile": copy.deepcopy(row.get("original_ipp2_zone_profile") or []),
                    "ipp2_original_gray_exposure_summary": str(row.get("original_gray_exposure_summary") or "n/a"),
                    "ipp2_original_detection_source": str(row.get("original_sphere_detection_source") or "failed"),
                    "ipp2_original_detection_confidence": float(row.get("original_sphere_detection_confidence", 0.0) or 0.0),
                    "ipp2_original_detection_label": str(row.get("original_sphere_detection_label") or "FAILED"),
                    "ipp2_original_detected_sphere_roi": copy.deepcopy(row.get("original_detected_sphere_roi") or {}),
                    "ipp2_initial_value_log2": float(row["initial_ipp2_value_log2"]),
                    "ipp2_initial_zone_profile": copy.deepcopy(row.get("initial_ipp2_zone_profile") or []),
                    "ipp2_initial_gray_exposure_summary": str(row.get("initial_gray_exposure_summary") or "n/a"),
                    "ipp2_initial_detection_source": str(row.get("initial_sphere_detection_source") or "failed"),
                    "ipp2_initial_detection_confidence": float(row.get("initial_sphere_detection_confidence", 0.0) or 0.0),
                    "ipp2_initial_detection_label": str(row.get("initial_sphere_detection_label") or "FAILED"),
                    "ipp2_initial_detected_sphere_roi": copy.deepcopy(row.get("initial_detected_sphere_roi") or {}),
                    "initial_ipp2_residual_stops": float(initial_summary.get("derived_residual_stops", 0.0) or 0.0),
                    "initial_ipp2_profile_max_residual_stops": float(initial_summary.get("max_abs_residual_stops", 0.0) or 0.0),
                    "initial_profile_audit_status": str(initial_summary.get("profile_audit_status") or "PROFILE NEEDS REVIEW"),
                    "initial_profile_note": str(initial_summary.get("profile_note") or "Profile needs review"),
                    "ipp2_value_log2": float(best_measure),
                    "ipp2_zone_profile": copy.deepcopy(best_zone_profile),
                    "ipp2_gray_exposure_summary": str(best_summary.get("gray_exposure_summary") or "n/a"),
                    "ipp2_target_log2": float(ipp2_target_log2),
                    "ipp2_target_zone_profile": copy.deepcopy(target_zone_profile),
                    "ipp2_target_gray_exposure_summary": str(target_summary.get("target_profile_summary") or "n/a"),
                    "ipp2_residual_stops": float(best_summary.get("derived_residual_stops", 0.0) or 0.0),
                    "ipp2_residual_abs_stops": abs(float(best_summary.get("derived_residual_stops", 0.0) or 0.0)),
                    "ipp2_profile_max_residual_stops": float(best_summary.get("max_abs_residual_stops", 0.0) or 0.0),
                    "derived_exposure_value": float(best_summary.get("derived_exposure_value_log2", 0.0) or 0.0),
                    "derived_target_value": float(best_summary.get("derived_target_value_log2", 0.0) or 0.0),
                    "derived_exposure_offset_stops": float(best_summary.get("derived_residual_stops", 0.0) or 0.0),
                    "camera_offset_from_anchor": float(best_summary.get("derived_residual_stops", 0.0) or 0.0),
                    "anchor_mode": str(target_summary.get("anchor_mode") or strategy_payload.get("anchor_mode") or ""),
                    "anchor_source": str(target_summary.get("anchor_source") or strategy_payload.get("anchor_source") or ""),
                    "anchor_scalar_value": float(target_summary.get("anchor_scalar_value", ipp2_target_log2) or ipp2_target_log2),
                    "anchor_ire_summary": str(target_summary.get("anchor_ire_summary") or strategy_payload.get("anchor_ire_summary") or "n/a"),
                    "profile_audit_status": str(best_summary.get("profile_audit_status") or "PROFILE NEEDS REVIEW"),
                    "profile_audit_label": str(best_summary.get("profile_audit_label") or "Profile needs review"),
                    "profile_note": str(best_summary.get("profile_note") or "Profile needs review"),
                    "profile_worst_zone_label": str(best_summary.get("profile_worst_zone_label") or "Zone"),
                    "profile_worst_zone_residual_stops": float(best_summary.get("profile_worst_zone_residual_stops", 0.0) or 0.0),
                    "profile_spread_delta_ire": float(best_summary.get("profile_spread_delta_ire", 0.0) or 0.0),
                    "ipp2_profile_mean_abs_residual_stops": float(best_summary.get("mean_abs_residual_stops", 0.0) or 0.0),
                    "ipp2_zone_residuals": copy.deepcopy(best_summary.get("zone_residuals") or []),
                    "weighted_zone_components": copy.deepcopy(best_summary.get("weighted_zone_components") or []),
                    "weighted_residual_stops": float(best_summary.get("weighted_residual_stops", best_summary.get("derived_residual_stops", 0.0)) or 0.0),
                    "zone_weights": dict(SPHERE_PROFILE_ZONE_WEIGHTS),
                    "sample_1_ire": float(clip_payload.get("sample_1_ire", 0.0) or 0.0),
                    "sample_2_ire": float(clip_payload.get("sample_2_ire", 0.0) or 0.0),
                    "sample_3_ire": float(clip_payload.get("sample_3_ire", 0.0) or 0.0),
                    "top_ire": float(clip_payload.get("top_ire", 0.0) or 0.0),
                    "mid_ire": float(clip_payload.get("mid_ire", 0.0) or 0.0),
                    "bottom_ire": float(clip_payload.get("bottom_ire", 0.0) or 0.0),
                    "zone_spread_ire": float(max((item.get("measured_ire", 0.0) for item in best_zone_profile), default=0.0) - min((item.get("measured_ire", 0.0) for item in best_zone_profile), default=0.0)),
                    "initial_exposure_correction_stops": float(row["initial_exposure_correction_stops"]),
                    "applied_correction_stops": float(best_correction),
                    "hit_correction_limit": bool(_at_closed_loop_correction_limit(best_correction)),
                    "log_original_value_log2": float(clip_payload.get("measured_log2_luminance", 0.0) or 0.0),
                    "log_target_value_log2": float(strategy_payload.get("target_log2_luminance", 0.0) or 0.0),
                    "log_corrected_expected_log2": float(clip_payload.get("measured_log2_luminance", 0.0) or 0.0) + float(best_correction),
                    "log_residual_stops": (
                        float(clip_payload.get("measured_log2_luminance", 0.0) or 0.0)
                        + float(best_correction)
                        - float(strategy_payload.get("target_log2_luminance", 0.0) or 0.0)
                    ),
                    "predicted_ipp2_value_log2": float(row["original_ipp2_value_log2"]) + float(best_correction),
                    "predicted_vs_actual_ipp2_delta_stops": float(best_measure) - (float(row["original_ipp2_value_log2"]) + float(best_correction)),
                    "ipp2_measurement_geometry": "three_band_gradient_aligned_profile_within_refined_sphere_mask",
                    "ipp2_sampling_method": str((render_cache.get((clip_id, strategy_key, round(float(best_correction), 6))) or {}).get("measure", {}).get("sampling_method") or "refined_interior_mask"),
                    "ipp2_sampling_confidence": float((render_cache.get((clip_id, strategy_key, round(float(best_correction), 6))) or {}).get("measure", {}).get("sampling_confidence", 0.0) or 0.0),
                    "ipp2_mask_fraction": float((render_cache.get((clip_id, strategy_key, round(float(best_correction), 6))) or {}).get("measure", {}).get("mask_fraction", 0.0) or 0.0),
                    "ipp2_detection_source": str((render_cache.get((clip_id, strategy_key, round(float(best_correction), 6))) or {}).get("measure", {}).get("sphere_roi_source") or row.get("initial_sphere_detection_source") or "failed"),
                    "ipp2_detection_confidence": float((render_cache.get((clip_id, strategy_key, round(float(best_correction), 6))) or {}).get("measure", {}).get("sphere_detection_confidence", row.get("initial_sphere_detection_confidence", 0.0)) or 0.0),
                    "ipp2_detection_label": str((render_cache.get((clip_id, strategy_key, round(float(best_correction), 6))) or {}).get("measure", {}).get("sphere_detection_label") or row.get("initial_sphere_detection_label") or "FAILED"),
                    "ipp2_detected_sphere_roi": copy.deepcopy(((render_cache.get((clip_id, strategy_key, round(float(best_correction), 6))) or {}).get("measure", {}).get("detected_sphere_roi") or row.get("initial_detected_sphere_roi") or {})),
                    "detection_failed": bool(best_summary.get("detection_failed")),
                    "original_image_path": str(initial_preview_paths.get(clip_id, {}).get("original") or ""),
                    "corrected_image_path": str(best_path),
                    "iteration_history": history,
                    "correction_application_method": str((render_cache.get((clip_id, strategy_key, round(float(best_correction), 6))) or {}).get("correction_application_method") or "direct_redline_flags"),
                    "direct_exposure_parameter": str((render_cache.get((clip_id, strategy_key, round(float(best_correction), 6))) or {}).get("direct_exposure_parameter") or REDLINE_DIRECT_EXPOSURE_PARAMETER),
                    "closed_loop_rmd_path": str((render_cache.get((clip_id, strategy_key, round(float(best_correction), 6))) or {}).get("rmd_path") or ""),
                    "status": str(status["status"]),
                    "tone": str(status["tone"]),
                    "target_source": str(target_summary["target_source"]),
                }
            )

        metrics = _closed_loop_strategy_metrics(row_results)
        closed_loop_strategy_summaries.append(
            {
                "strategy_key": strategy_key,
                "strategy_label": str(strategy_payload.get("strategy_label") or strategy_key),
                "target_log2": float(ipp2_target_log2),
                "target_zone_profile": copy.deepcopy(target_zone_profile),
                "target_profile_summary": str(target_summary.get("target_profile_summary") or "n/a"),
                "target_source": str(target_summary["target_source"]),
                "reference_clip_id": str(target_summary.get("reference_clip_id") or strategy_payload.get("reference_clip_id") or ""),
                "anchor_mode": str(target_summary.get("anchor_mode") or strategy_payload.get("anchor_mode") or ""),
                "anchor_source": str(target_summary.get("anchor_source") or strategy_payload.get("anchor_source") or ""),
                "anchor_scalar_value": float(target_summary.get("anchor_scalar_value", ipp2_target_log2) or ipp2_target_log2),
                "anchor_ire_summary": str(target_summary.get("anchor_ire_summary") or strategy_payload.get("anchor_ire_summary") or "n/a"),
                "rows": row_results,
                "metrics": metrics,
            }
        )
        emit_review_progress(
            progress_path,
            phase="closed_loop_strategy_complete",
            detail=f"Completed strategy {strategy_key}.",
            stage_label="Closed-loop solve",
            clip_index=strategy_index,
            clip_count=len(refined_payloads),
            current_clip_id=strategy_key,
            elapsed_seconds=time.perf_counter() - solver_started_at,
            review_mode="full_contact_sheet",
        )

    recommended_strategy_key = (
        str(explicit_anchor_strategy_key or "")
        if explicit_anchor_strategy_key and any(str(item.get("strategy_key") or "") == str(explicit_anchor_strategy_key) for item in closed_loop_strategy_summaries)
        else _recommend_strategy_from_closed_loop(closed_loop_strategy_summaries)
    )
    for payload in refined_payloads:
        payload["recommended"] = str(payload.get("strategy_key") or "") == recommended_strategy_key
        payload["ipp2_closed_loop"] = next(
            (copy.deepcopy(item) for item in closed_loop_strategy_summaries if str(item.get("strategy_key") or "") == str(payload.get("strategy_key") or "")),
            {},
        )
    summary = {
        "schema_version": "r3dmatch_ipp2_closed_loop_trace_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "validation_transform": _preview_transform_label(ipp2_preview_settings),
        "tolerance_model": _ipp2_tolerance_model(),
        "max_iterations": IPP2_CLOSED_LOOP_MAX_ITERATIONS,
        "recommended_strategy_key": recommended_strategy_key,
        "explicit_anchor_strategy_key": str(explicit_anchor_strategy_key or ""),
        "strategies": closed_loop_strategy_summaries,
    }
    output_path = out_root / "ipp2_closed_loop_trace.json"
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {
        "path": str(output_path),
        "summary": summary,
        "strategy_payloads": refined_payloads,
        "preview_paths": preview_paths,
        "preview_settings": ipp2_preview_settings,
    }


def _preview_settings_match(left: Dict[str, object], right: Dict[str, object]) -> bool:
    keys = (
        "preview_mode",
        "output_space",
        "output_gamma",
        "highlight_rolloff",
        "shadow_rolloff",
        "lut_path",
        "output_tonemap",
    )
    return all(left.get(key) == right.get(key) for key in keys)


def _build_sampling_comparison(analysis_records: List[Dict[str, object]], *, out_root: Path) -> Optional[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for record in analysis_records:
        diagnostics = dict(record.get("diagnostics") or {})
        comparison = dict(diagnostics.get("sphere_sampling_comparison") or {})
        legacy = dict(comparison.get("legacy") or {})
        refined = dict(comparison.get("refined") or {})
        if not legacy or not refined:
            continue
        rows.append(
            {
                "camera_id": _camera_id_from_clip_id(str(record.get("clip_id") or "")),
                "clip_id": str(record.get("clip_id") or ""),
                "legacy_measured_gray_log2": float(legacy.get("raw_log2", 0.0) or 0.0),
                "refined_measured_gray_log2": float(refined.get("raw_log2", 0.0) or 0.0),
                "delta_log2_stops": float(comparison.get("delta_raw_log2", 0.0) or 0.0),
                "legacy_confidence": float(legacy.get("confidence", 0.0) or 0.0),
                "refined_confidence": float(refined.get("confidence", 0.0) or 0.0),
                "legacy_mask_fraction": float(legacy.get("mask_fraction", 1.0) or 0.0),
                "refined_mask_fraction": float(refined.get("mask_fraction", 1.0) or 0.0),
                "rejected_circle_fraction": max(0.0, 1.0 - float(refined.get("mask_fraction", 1.0) or 0.0)),
                "legacy_sampling_method": str(legacy.get("sampling_method") or "legacy_circle_mask"),
                "refined_sampling_method": str(refined.get("sampling_method") or "refined_interior_mask"),
                "window_reference_raw_log2": float(comparison.get("window_reference_raw_log2", 0.0) or 0.0),
            }
        )
    if not rows:
        return None
    delta_values = [abs(float(row["delta_log2_stops"])) for row in rows]
    summary = {
        "schema_version": "r3dmatch_sampling_comparison_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "row_count": len(rows),
        "median_absolute_delta_stops": float(np.median(np.asarray(delta_values, dtype=np.float32))) if delta_values else 0.0,
        "max_absolute_delta_stops": max(delta_values) if delta_values else 0.0,
        "rows": rows,
    }
    output_path = out_root / "sampling_comparison.json"
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {"path": str(output_path), "summary": summary}


def _analysis_records_for_sampling_variant(analysis_records: List[Dict[str, object]], *, variant: str) -> List[Dict[str, object]]:
    updated_records: List[Dict[str, object]] = []
    for record in analysis_records:
        updated = copy.deepcopy(record)
        diagnostics = dict(updated.get("diagnostics") or {})
        comparison = dict(diagnostics.get("sphere_sampling_comparison") or {})
        variant_payload = dict(comparison.get(variant) or {})
        if variant_payload:
            diagnostics["measured_log2_luminance_raw"] = float(variant_payload.get("raw_log2", diagnostics.get("measured_log2_luminance_raw", 0.0)) or 0.0)
            diagnostics["measured_log2_luminance_monitoring"] = float(
                variant_payload.get("monitoring_log2", diagnostics.get("measured_log2_luminance_monitoring", diagnostics.get("measured_log2_luminance", 0.0))) or 0.0
            )
            diagnostics["measured_log2_luminance"] = diagnostics["measured_log2_luminance_monitoring"]
            diagnostics["measured_rgb_mean"] = [float(value) for value in (variant_payload.get("rgb_mean_raw") or diagnostics.get("measured_rgb_mean", [0.0, 0.0, 0.0]))]
            diagnostics["measured_rgb_chromaticity"] = [
                float(value)
                for value in (variant_payload.get("rgb_chromaticity_raw") or diagnostics.get("measured_rgb_chromaticity", [1 / 3, 1 / 3, 1 / 3]))
            ]
            diagnostics["active_sampling_variant"] = variant
        updated["diagnostics"] = diagnostics
        updated_records.append(updated)
    return updated_records


def _strategy_snapshot(strategy_payloads: List[Dict[str, object]]) -> Dict[str, Dict[str, object]]:
    snapshot: Dict[str, Dict[str, object]] = {}
    for payload in strategy_payloads:
        snapshot[str(payload["strategy_key"])] = {
            "strategy_key": str(payload["strategy_key"]),
            "strategy_label": str(payload["strategy_label"]),
            "anchor_mode": str(payload.get("anchor_mode") or ""),
            "anchor_source": str(payload.get("anchor_source") or ""),
            "anchor_scalar_value": float(payload.get("anchor_scalar_value", 0.0) or 0.0),
            "anchor_ire_summary": str(payload.get("anchor_ire_summary") or "n/a"),
            "target_log2_luminance": float(payload["target_log2_luminance"]),
            "reference_clip_id": payload.get("reference_clip_id"),
            "clips": {
                str(clip["clip_id"]): {
                    "measured_log2_luminance": float(clip.get("measured_log2_luminance_raw", clip.get("measured_log2_luminance", 0.0)) or 0.0),
                    "exposure_offset_stops": float(clip["exposure_offset_stops"]),
                    "confidence": float(clip.get("confidence", 0.0) or 0.0),
                }
                for clip in payload.get("clips", [])
            },
        }
    return snapshot


def _build_solve_comparison(
    analysis_records: List[Dict[str, object]],
    *,
    target_strategies: List[str],
    reference_clip_id: Optional[str],
    hero_clip_id: Optional[str],
    target_type: Optional[str],
    matching_domain: str,
    quality_by_clip: Optional[Dict[str, Dict[str, object]]],
    anchor_target_log2: Optional[float],
    exposure_anchor_mode: Optional[str],
    manual_target_stops: Optional[float],
    manual_target_ire: Optional[float],
    out_root: Path,
) -> Optional[Dict[str, object]]:
    if not any((record.get("diagnostics") or {}).get("sphere_sampling_comparison") for record in analysis_records):
        return None
    variants = {
        "legacy": _analysis_records_for_sampling_variant(analysis_records, variant="legacy"),
        "refined": _analysis_records_for_sampling_variant(analysis_records, variant="refined"),
    }
    snapshots: Dict[str, Dict[str, object]] = {}
    for variant_key, variant_records in variants.items():
        payloads = _build_strategy_payloads(
            variant_records,
            target_strategies=target_strategies,
            reference_clip_id=reference_clip_id,
            hero_clip_id=hero_clip_id,
            target_type=target_type,
            monitoring_measurements_by_clip=None,
            matching_domain=matching_domain,
            quality_by_clip=quality_by_clip,
            anchor_target_log2=anchor_target_log2,
            exposure_anchor_mode=exposure_anchor_mode,
            manual_target_stops=manual_target_stops,
            manual_target_ire=manual_target_ire,
        )
        snapshots[variant_key] = _strategy_snapshot(payloads)
    strategies: List[Dict[str, object]] = []
    for strategy_key in sorted(set(snapshots["legacy"].keys()) | set(snapshots["refined"].keys()), key=lambda item: STRATEGY_ORDER.index(item) if item in STRATEGY_ORDER else 999):
        legacy_strategy = dict(snapshots["legacy"].get(strategy_key) or {})
        refined_strategy = dict(snapshots["refined"].get(strategy_key) or {})
        legacy_clips = dict(legacy_strategy.get("clips") or {})
        refined_clips = dict(refined_strategy.get("clips") or {})
        clip_rows = []
        for clip_id in sorted(set(legacy_clips.keys()) | set(refined_clips.keys())):
            legacy_clip = dict(legacy_clips.get(clip_id) or {})
            refined_clip = dict(refined_clips.get(clip_id) or {})
            correction_delta = float(refined_clip.get("exposure_offset_stops", 0.0) or 0.0) - float(legacy_clip.get("exposure_offset_stops", 0.0) or 0.0)
            clip_rows.append(
                {
                    "camera_id": _camera_id_from_clip_id(clip_id),
                    "clip_id": clip_id,
                    "legacy_measured_gray_log2": float(legacy_clip.get("measured_log2_luminance", 0.0) or 0.0),
                    "refined_measured_gray_log2": float(refined_clip.get("measured_log2_luminance", 0.0) or 0.0),
                    "legacy_correction_stops": float(legacy_clip.get("exposure_offset_stops", 0.0) or 0.0),
                    "refined_correction_stops": float(refined_clip.get("exposure_offset_stops", 0.0) or 0.0),
                    "correction_delta_stops": correction_delta,
                    "stronger_brighten_with_refined": correction_delta > 1e-4,
                }
            )
        strategies.append(
            {
                "strategy_key": strategy_key,
                "strategy_label": str(refined_strategy.get("strategy_label") or legacy_strategy.get("strategy_label") or strategy_key),
                "anchor_mode": str(refined_strategy.get("anchor_mode") or legacy_strategy.get("anchor_mode") or ""),
                "anchor_source": str(refined_strategy.get("anchor_source") or legacy_strategy.get("anchor_source") or ""),
                "anchor_scalar_value": float(refined_strategy.get("anchor_scalar_value", legacy_strategy.get("anchor_scalar_value", 0.0)) or 0.0),
                "anchor_ire_summary": str(refined_strategy.get("anchor_ire_summary") or legacy_strategy.get("anchor_ire_summary") or "n/a"),
                "legacy_target_log2": float(legacy_strategy.get("target_log2_luminance", 0.0) or 0.0),
                "refined_target_log2": float(refined_strategy.get("target_log2_luminance", 0.0) or 0.0),
                "target_delta_stops": float(refined_strategy.get("target_log2_luminance", 0.0) or 0.0) - float(legacy_strategy.get("target_log2_luminance", 0.0) or 0.0),
                "clips": clip_rows,
            }
        )
    output_payload = {
        "schema_version": "r3dmatch_solve_comparison_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "strategies": strategies,
    }
    output_path = out_root / "solve_comparison.json"
    output_path.write_text(json.dumps(output_payload, indent=2), encoding="utf-8")
    return {"path": str(output_path), "summary": output_payload}


def _build_corrected_residual_validation(
    *,
    strategies: List[Dict[str, object]],
    out_root: Path,
) -> Dict[str, object]:
    tolerance = _corrected_residual_tolerance_model()
    validation_rows: List[Dict[str, object]] = []
    original_measure_cache: Dict[str, Dict[str, object]] = {}

    for strategy in strategies:
        raw_rows: List[Dict[str, object]] = []
        corrected_values_for_target: List[float] = []
        for clip in strategy.get("clips", []):
            clip_id = str(clip.get("clip_id") or "")
            calibration_roi = clip.get("calibration_roi")
            original_frame = clip.get("original_frame")
            corrected_frame = clip.get("both_corrected")
            if original_frame and clip_id not in original_measure_cache and Path(str(original_frame)).exists():
                original_measure_cache[clip_id] = _measure_rendered_preview_roi(str(original_frame), calibration_roi)
            original_measure = dict(original_measure_cache.get(clip_id) or {})
            corrected_measure = (
                _measure_rendered_preview_roi(str(corrected_frame), calibration_roi)
                if corrected_frame and Path(str(corrected_frame)).exists()
                else {}
            )
            metrics = dict(clip.get("metrics") or {})
            exposure_metrics = dict(metrics.get("exposure") or {})
            trust_class = str(clip.get("trust_class") or "UNTRUSTED")
            corrected_log2 = float(corrected_measure.get("measured_log2_luminance_monitoring", 0.0) or 0.0)
            if trust_class in {"TRUSTED", "USE_WITH_CAUTION"} and corrected_measure:
                corrected_values_for_target.append(corrected_log2)
            raw_rows.append(
                {
                    "camera_id": _camera_id_from_clip_id(clip_id),
                    "camera_label": _camera_label_for_reporting(clip_id),
                    "clip_id": clip_id,
                    "strategy_key": str(strategy.get("strategy_key") or ""),
                    "strategy_label": str(strategy.get("strategy_label") or ""),
                    "recommended": bool(strategy.get("recommended")),
                    "analysis_target_log2": float(strategy.get("target_log2_luminance", 0.0) or 0.0),
                    "original_measured_gray_log2": float(original_measure.get("measured_log2_luminance_monitoring", 0.0) or 0.0),
                    "applied_correction_stops": float(exposure_metrics.get("final_offset_stops", 0.0) or 0.0),
                    "corrected_measured_gray_log2": corrected_log2,
                    "trust_class": trust_class,
                    "reference_use": str(clip.get("reference_use") or "Included"),
                    "trust_reason": str(clip.get("trust_reason") or ""),
                    "measurement_geometry": str(corrected_measure.get("measurement_geometry") or "three_rect_windows_within_roi"),
                    "corrected_image_path": str(corrected_frame or ""),
                    "original_image_path": str(original_frame or ""),
                }
            )

        if corrected_values_for_target:
            render_target_log2 = float(np.median(np.asarray(corrected_values_for_target, dtype=np.float32)))
        else:
            fallback_values = [
                float(item.get("corrected_measured_gray_log2", 0.0) or 0.0)
                for item in raw_rows
                if str(item.get("corrected_image_path") or "").strip()
            ]
            render_target_log2 = float(np.median(np.asarray(fallback_values, dtype=np.float32))) if fallback_values else 0.0

        for row in raw_rows:
            residual = float(row["corrected_measured_gray_log2"]) - render_target_log2
            status = _corrected_residual_status(abs(residual))
            row["render_target_gray_log2"] = render_target_log2
            row["residual_error_stops"] = residual
            row["residual_abs_error_stops"] = abs(residual)
            row["residual_status"] = str(status["status"])
            row["residual_label"] = str(status["label"])
            row["residual_tone"] = str(status["tone"])
            validation_rows.append(row)

    recommended_rows = [row for row in validation_rows if bool(row.get("recommended"))]
    if not recommended_rows:
        recommended_strategy_key = next(
            (str(item.get("strategy_key") or "") for item in strategies if bool(item.get("recommended"))),
            str(strategies[0].get("strategy_key") or "") if strategies else "",
        )
        recommended_rows = [row for row in validation_rows if str(row.get("strategy_key") or "") == recommended_strategy_key]
    abs_residuals = [float(row.get("residual_abs_error_stops", 0.0) or 0.0) for row in recommended_rows]
    outside_rows = [row for row in recommended_rows if str(row.get("residual_status") or "") == "outside_tolerance"]
    summary = {
        "schema_version": "r3dmatch_corrected_residual_validation_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tolerance_model": tolerance,
        "strategy_count": len(strategies),
        "row_count": len(validation_rows),
        "recommended_strategy_key": next(
            (str(item.get("strategy_key") or "") for item in strategies if bool(item.get("recommended"))),
            str(strategies[0].get("strategy_key") or "") if strategies else "",
        ),
        "best_residual_stops": min(abs_residuals) if abs_residuals else None,
        "worst_residual_stops": max(abs_residuals) if abs_residuals else None,
        "median_residual_stops": float(np.median(np.asarray(abs_residuals, dtype=np.float32))) if abs_residuals else None,
        "outside_tolerance_cameras": [
            {
                "camera_id": row["camera_id"],
                "clip_id": row["clip_id"],
                "residual_error_stops": row["residual_error_stops"],
                "trust_class": row["trust_class"],
                "trust_reason": row["trust_reason"],
            }
            for row in outside_rows
        ],
        "rows": validation_rows,
    }
    output_path = out_root / "corrected_residual_validation.json"
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {
        "path": str(output_path),
        "summary": summary,
    }


def _build_ipp2_validation(
    *,
    input_path: str,
    out_root: Path,
    analysis_records: List[Dict[str, object]],
    strategy_payloads: List[Dict[str, object]],
    strategies: List[Dict[str, object]],
    redline_capabilities: Dict[str, object],
    run_id: str,
    display_preview_settings: Dict[str, object],
    display_preview_paths: Dict[str, Dict[str, object]],
    closed_loop_result: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    ipp2_preview_settings = dict((closed_loop_result or {}).get("preview_settings") or _ipp2_validation_preview_settings())
    reuse_display_previews = _preview_settings_match(display_preview_settings, ipp2_preview_settings)
    validation_rows: List[Dict[str, object]] = []
    if closed_loop_result is not None:
        recommended_strategy_key = str((closed_loop_result.get("summary") or {}).get("recommended_strategy_key") or "")
        for strategy_summary in list(((closed_loop_result.get("summary") or {}).get("strategies") or [])):
            strategy_key = str(strategy_summary.get("strategy_key") or "")
            for item in list(strategy_summary.get("rows") or []):
                row = copy.deepcopy(item)
                row["recommended"] = strategy_key == recommended_strategy_key
                row["used_contact_sheet_asset_for_validation"] = False
                row["log_vs_ipp2_residual_delta_stops"] = float(row.get("ipp2_residual_stops", 0.0) or 0.0) - float(row.get("log_residual_stops", 0.0) or 0.0)
                operator_guidance = _operator_guidance_for_correction(
                    correction_stops=float(row.get("applied_correction_stops", 0.0) or 0.0),
                    residual_stops=float(row.get("ipp2_residual_stops", 0.0) or 0.0),
                    validation_status=str(row.get("status") or "REVIEW"),
                )
                row["operator_guidance"] = operator_guidance
                row["correction_stops"] = float(operator_guidance.get("correction_stops", 0.0) or 0.0)
                row["rounded_stops"] = float(operator_guidance.get("rounded_stops", 0.0) or 0.0)
                row["direction"] = str(operator_guidance.get("direction") or "hold")
                row["suggested_action"] = str(operator_guidance.get("suggested_action") or "")
                row["operator_status"] = str(operator_guidance.get("status") or "REVIEW")
                row["operator_notes"] = str(operator_guidance.get("notes") or "")
                row["sphere_detection_note"] = _sphere_detection_note(
                    str(row.get("ipp2_detection_source") or ""),
                    str(row.get("ipp2_detection_label") or ""),
                    detection_failed=bool(row.get("detection_failed")),
                )
                row["camera_offset_from_anchor"] = float(row.get("derived_exposure_offset_stops", 0.0) or 0.0)
                validation_rows.append(row)
    else:
        if reuse_display_previews:
            ipp2_preview_paths = display_preview_paths
        else:
            ipp2_preview_paths = generate_preview_stills(
                input_path,
                analysis_records=analysis_records,
                previews_dir=str(Path(input_path).expanduser().resolve() / "previews" / "_ipp2_validation"),
                preview_settings=ipp2_preview_settings,
                redline_capabilities=redline_capabilities,
                strategy_payloads=strategy_payloads,
                run_id=f"{run_id}.ipp2",
                strategy_rmd_root=str(Path(input_path).expanduser().resolve() / "review_rmd" / "strategies_ipp2_validation"),
                render_originals=True,
            )

        original_measure_cache: Dict[str, Dict[str, object]] = {}
        detected_sphere_roi_cache: Dict[str, Dict[str, float]] = {}

        for strategy in strategies:
            raw_rows: List[Dict[str, object]] = []
            trusted_values: List[float] = []
            fallback_values: List[float] = []
            for clip in strategy.get("clips", []):
                clip_id = str(clip.get("clip_id") or "")
                calibration_roi = clip.get("calibration_roi")
                original_frame = str(ipp2_preview_paths.get(clip_id, {}).get("original") or "")
                corrected_frame = str(
                    ipp2_preview_paths.get(clip_id, {}).get("strategies", {}).get(str(strategy.get("strategy_key") or ""), {}).get("both") or ""
                )
                if original_frame and clip_id not in original_measure_cache and Path(original_frame).exists():
                    original_measure_cache[clip_id] = _measure_rendered_preview_roi_ipp2(original_frame, calibration_roi)
                    detected_roi = dict(original_measure_cache[clip_id].get("detected_sphere_roi") or {})
                    if detected_roi:
                        detected_sphere_roi_cache[clip_id] = detected_roi
                original_measure = dict(original_measure_cache.get(clip_id) or {})
                corrected_measure = (
                    _measure_rendered_preview_roi_ipp2(
                        corrected_frame,
                        calibration_roi,
                        sphere_roi_override=detected_sphere_roi_cache.get(clip_id),
                    )
                    if corrected_frame and Path(corrected_frame).exists()
                    else {}
                )
                ipp2_value = float(corrected_measure.get("measured_log2_luminance_monitoring", 0.0) or 0.0)
                if corrected_measure:
                    fallback_values.append(ipp2_value)
                    if str(clip.get("trust_class") or "") in {"TRUSTED", "USE_WITH_CAUTION"}:
                        trusted_values.append(ipp2_value)
                log_original = float((clip.get("metrics", {}).get("exposure", {}) or {}).get("measured_log2_luminance", 0.0) or 0.0)
                applied_correction = float((clip.get("metrics", {}).get("exposure", {}) or {}).get("final_offset_stops", 0.0) or 0.0)
                log_target = float(strategy.get("target_log2_luminance", 0.0) or 0.0)
                log_corrected_expected = log_original + applied_correction
                log_residual = log_corrected_expected - log_target
                raw_rows.append(
                    {
                        "camera_id": _camera_id_from_clip_id(clip_id),
                        "camera_label": _camera_label_for_reporting(clip_id),
                        "clip_id": clip_id,
                        "strategy_key": str(strategy.get("strategy_key") or ""),
                        "strategy_label": str(strategy.get("strategy_label") or ""),
                        "recommended": bool(strategy.get("recommended")),
                        "reference_use": str(clip.get("reference_use") or "Included"),
                        "trust_class": str(clip.get("trust_class") or "UNTRUSTED"),
                        "trust_reason": str(clip.get("trust_reason") or ""),
                        "ipp2_original_value_log2": float(original_measure.get("measured_log2_luminance_monitoring", 0.0) or 0.0),
                        "ipp2_value_log2": ipp2_value,
                        "applied_correction_stops": applied_correction,
                        "log_original_value_log2": log_original,
                        "log_target_value_log2": log_target,
                        "log_corrected_expected_log2": log_corrected_expected,
                        "log_residual_stops": log_residual,
                        "ipp2_measurement_geometry": str(corrected_measure.get("measurement_geometry") or "refined_sphere_mask_in_rendered_ipp2_roi"),
                        "ipp2_sampling_method": str(corrected_measure.get("sampling_method") or ""),
                        "ipp2_sampling_confidence": float(corrected_measure.get("sampling_confidence", 0.0) or 0.0),
                        "ipp2_mask_fraction": float(corrected_measure.get("mask_fraction", 0.0) or 0.0),
                        "ipp2_original_detection_source": str(original_measure.get("sphere_roi_source") or "failed"),
                        "ipp2_original_detection_confidence": float(original_measure.get("sphere_detection_confidence", 0.0) or 0.0),
                        "ipp2_original_detection_label": str(original_measure.get("sphere_detection_label") or "FAILED"),
                        "ipp2_original_detected_sphere_roi": dict(original_measure.get("detected_sphere_roi") or {}),
                        "ipp2_detection_source": str(corrected_measure.get("sphere_roi_source") or "failed"),
                        "ipp2_detection_confidence": float(corrected_measure.get("sphere_detection_confidence", 0.0) or 0.0),
                        "ipp2_detection_label": str(corrected_measure.get("sphere_detection_label") or "FAILED"),
                        "ipp2_detected_sphere_roi": dict(corrected_measure.get("detected_sphere_roi") or {}),
                        "detection_failed": bool(original_measure.get("detection_failed")) or bool(corrected_measure.get("detection_failed")),
                        "corrected_image_path": corrected_frame,
                        "original_image_path": original_frame,
                        "used_contact_sheet_asset_for_validation": bool(
                            corrected_frame and corrected_frame == str(clip.get("both_corrected") or "")
                        ),
                    }
                )

            ipp2_target = (
                float(np.median(np.asarray(trusted_values, dtype=np.float32)))
                if trusted_values
                else float(np.median(np.asarray(fallback_values, dtype=np.float32)))
                if fallback_values
                else 0.0
            )

            for row in raw_rows:
                ipp2_residual = float(row["ipp2_value_log2"]) - ipp2_target
                status = _ipp2_validation_status(abs(ipp2_residual))
                row["ipp2_target_log2"] = ipp2_target
                row["ipp2_residual_stops"] = ipp2_residual
                row["ipp2_residual_abs_stops"] = abs(ipp2_residual)
                row["status"] = str(status["status"])
                row["tone"] = str(status["tone"])
                row["log_vs_ipp2_residual_delta_stops"] = ipp2_residual - float(row["log_residual_stops"])
                operator_guidance = _operator_guidance_for_correction(
                    correction_stops=float(row.get("applied_correction_stops", 0.0) or 0.0),
                    residual_stops=ipp2_residual,
                    validation_status=str(row["status"]),
                )
                row["operator_guidance"] = operator_guidance
                row["correction_stops"] = float(operator_guidance.get("correction_stops", 0.0) or 0.0)
                row["rounded_stops"] = float(operator_guidance.get("rounded_stops", 0.0) or 0.0)
                row["direction"] = str(operator_guidance.get("direction") or "hold")
                row["suggested_action"] = str(operator_guidance.get("suggested_action") or "")
                row["operator_status"] = str(operator_guidance.get("status") or "REVIEW")
                row["operator_notes"] = str(operator_guidance.get("notes") or "")
                row["sphere_detection_note"] = _sphere_detection_note(
                    str(row.get("ipp2_detection_source") or ""),
                    str(row.get("ipp2_detection_label") or ""),
                    detection_failed=bool(row.get("detection_failed")),
                )
                row["camera_offset_from_anchor"] = float(row.get("derived_exposure_offset_stops", ipp2_residual) or ipp2_residual)
                validation_rows.append(row)

    recommended_rows = [row for row in validation_rows if bool(row.get("recommended"))]
    if not recommended_rows:
        recommended_strategy_key = next(
            (str(item.get("strategy_key") or "") for item in strategies if bool(item.get("recommended"))),
            str(strategies[0].get("strategy_key") or "") if strategies else "",
        )
        recommended_rows = [row for row in validation_rows if str(row.get("strategy_key") or "") == recommended_strategy_key]
    residuals = [float(row.get("ipp2_residual_abs_stops", 0.0) or 0.0) for row in recommended_rows]
    profile_residuals = [float(row.get("ipp2_profile_max_residual_stops", row.get("ipp2_residual_abs_stops", 0.0)) or 0.0) for row in recommended_rows]
    divergence_values = [abs(float(row.get("log_vs_ipp2_residual_delta_stops", 0.0) or 0.0)) for row in recommended_rows]
    status_counts = {
        status: sum(1 for row in recommended_rows if str(row.get("status") or "") == status)
        for status in ("PASS", "REVIEW", "FAIL")
    }
    profile_audit_counts = {
        status: sum(1 for row in recommended_rows if str(row.get("profile_audit_status") or "") == status)
        for status in ("PROFILE CONSISTENT", "PROFILE NEEDS REVIEW", "PROFILE SHAPE MISMATCH")
    }
    summary = {
        "schema_version": "r3dmatch_ipp2_validation_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "validation_transform": _preview_transform_label(ipp2_preview_settings),
        "validation_preview_settings": ipp2_preview_settings,
        "contact_sheet_preview_transform": _preview_transform_label(display_preview_settings),
        "contact_sheet_preview_matches_validation": reuse_display_previews,
        "tolerance_model": _ipp2_tolerance_model(),
        "strategy_count": len(strategies),
        "row_count": len(validation_rows),
        "recommended_strategy_key": next(
            (str(item.get("strategy_key") or "") for item in strategies if bool(item.get("recommended"))),
            str(strategies[0].get("strategy_key") or "") if strategies else "",
        ),
        "anchor_mode": str((recommended_rows[0].get("anchor_mode") if recommended_rows else "") or ""),
        "anchor_source": str((recommended_rows[0].get("anchor_source") if recommended_rows else "") or ""),
        "anchor_scalar_value": (
            float(recommended_rows[0].get("anchor_scalar_value", 0.0) or 0.0)
            if recommended_rows else None
        ),
        "anchor_ire_summary": str((recommended_rows[0].get("anchor_ire_summary") if recommended_rows else "") or ""),
        "max_residual": max(residuals) if residuals else None,
        "median_residual": float(np.median(np.asarray(residuals, dtype=np.float32))) if residuals else None,
        "best_residual": min(residuals) if residuals else None,
        "max_profile_residual": max(profile_residuals) if profile_residuals else None,
        "median_profile_residual": float(np.median(np.asarray(profile_residuals, dtype=np.float32))) if profile_residuals else None,
        "best_profile_residual": min(profile_residuals) if profile_residuals else None,
        "all_within_tolerance": all(float(value) <= IPP2_VALIDATION_PASS_STOPS for value in residuals) if residuals else False,
        "status_counts": status_counts,
        "profile_audit_counts": profile_audit_counts,
        "log_vs_ipp2_max_divergence_stops": max(divergence_values) if divergence_values else None,
        "rows": validation_rows,
    }
    output_path = out_root / "ipp2_validation.json"
    output_path.write_text(json.dumps({"cameras": validation_rows, "summary": summary}, indent=2), encoding="utf-8")
    return {"path": str(output_path), "summary": summary}


def _write_render_trace_artifacts(
    *,
    out_root: Path,
    analysis_records: List[Dict[str, object]],
    strategies: List[Dict[str, object]],
    ipp2_validation_summary: Dict[str, object],
    preview_manifest_payload: Dict[str, object],
) -> Dict[str, str]:
    recommended_strategy_key = str(
        ipp2_validation_summary.get("recommended_strategy_key")
        or next((item.get("strategy_key") for item in strategies if item.get("recommended")), "")
        or ""
    )
    record_by_clip = {
        str(record.get("clip_id") or ""): dict(record)
        for record in analysis_records
        if str(record.get("clip_id") or "").strip()
    }
    strategy_section = next(
        (item for item in strategies if str(item.get("strategy_key") or "") == recommended_strategy_key),
        strategies[0] if strategies else {},
    )
    preview_commands = list(preview_manifest_payload.get("commands") or [])
    baseline_by_clip = {
        str(item.get("clip_id") or ""): dict(item)
        for item in preview_commands
        if isinstance(item, dict) and str(item.get("variant") or "") == "original"
    }
    validation_rows = [
        dict(item)
        for item in list(ipp2_validation_summary.get("rows") or [])
        if str(item.get("strategy_key") or "") == recommended_strategy_key
    ]

    render_input_rows: List[Dict[str, object]] = []
    pre_render_rows: List[Dict[str, object]] = []
    post_render_rows: List[Dict[str, object]] = []
    comparison_rows: List[Dict[str, object]] = []

    for clip in list(strategy_section.get("clips") or []):
        clip_id = str(clip.get("clip_id") or "")
        record = record_by_clip.get(clip_id, {})
        metadata = dict(record.get("clip_metadata") or {})
        baseline_command = dict(baseline_by_clip.get(clip_id) or {})
        ipp2_row = next((item for item in validation_rows if str(item.get("clip_id") or "") == clip_id), {})
        render_input_rows.append(
            {
                "clip_id": clip_id,
                "camera_id": _camera_id_from_clip_id(clip_id),
                "source_path": str(record.get("source_path") or clip.get("source_path") or ""),
                "iso": metadata.get("iso"),
                "kelvin": metadata.get("kelvin"),
                "tint": metadata.get("tint"),
                "color_space": metadata.get("color_space"),
                "gamma_curve": metadata.get("gamma_curve"),
                "original_render_command": baseline_command.get("command"),
                "preview_transform": str((clip.get("metrics") or {}).get("preview_transform") or ""),
                "gray_exposure": str(((clip.get("metrics") or {}).get("exposure") or {}).get("gray_exposure_summary") or "n/a"),
                "reference_profile": str((ipp2_row.get("ipp2_target_gray_exposure_summary") or "n/a")),
                "corrected_state": {
                    "initialExposureAdjust": float(((clip.get("metrics") or {}).get("exposure") or {}).get("initial_offset_stops", 0.0) or 0.0),
                    "exposureAdjust": float(((clip.get("metrics") or {}).get("exposure") or {}).get("final_offset_stops", 0.0) or 0.0),
                    "rgb_gains": (((clip.get("metrics") or {}).get("color") or {}).get("rgb_gains_diagnostic")
                                  or ((clip.get("metrics") or {}).get("color") or {}).get("rgb_gains")),
                    "cdl": (((clip.get("metrics") or {}).get("color") or {}).get("cdl")),
                },
            }
        )
        pre_render_rows.append(
            {
                "clip_id": clip_id,
                "camera_id": _camera_id_from_clip_id(clip_id),
                "log_pre_render": float(((clip.get("metrics") or {}).get("exposure") or {}).get("measured_log2_luminance", 0.0) or 0.0),
                "log_pre_render_monitoring_proxy": float(((clip.get("metrics") or {}).get("exposure") or {}).get("measured_log2_luminance_monitoring", 0.0) or 0.0),
                "gray_exposure": str(((clip.get("metrics") or {}).get("exposure") or {}).get("gray_exposure_summary") or "n/a"),
                "top_ire": float(((clip.get("metrics") or {}).get("exposure") or {}).get("top_ire", 0.0) or 0.0),
                "mid_ire": float(((clip.get("metrics") or {}).get("exposure") or {}).get("mid_ire", 0.0) or 0.0),
                "bottom_ire": float(((clip.get("metrics") or {}).get("exposure") or {}).get("bottom_ire", 0.0) or 0.0),
            }
        )
        post_render_rows.append(
            {
                "clip_id": clip_id,
                "camera_id": _camera_id_from_clip_id(clip_id),
                "ipp2_original_value": float(ipp2_row.get("ipp2_original_value_log2", 0.0) or 0.0),
                "ipp2_corrected_value": float(ipp2_row.get("ipp2_value_log2", 0.0) or 0.0),
                "ipp2_target": float(ipp2_row.get("ipp2_target_log2", 0.0) or 0.0),
                "ipp2_residual": float(ipp2_row.get("ipp2_residual_stops", 0.0) or 0.0),
                "predicted_ipp2_value": float(ipp2_row.get("predicted_ipp2_value_log2", 0.0) or 0.0),
                "predicted_vs_actual_delta": float(ipp2_row.get("predicted_vs_actual_ipp2_delta_stops", 0.0) or 0.0),
                "gray_exposure": str(ipp2_row.get("ipp2_gray_exposure_summary") or "n/a"),
                "original_gray_exposure": str(ipp2_row.get("ipp2_original_gray_exposure_summary") or "n/a"),
                "target_gray_exposure": str(ipp2_row.get("ipp2_target_gray_exposure_summary") or "n/a"),
                "zone_residuals": _format_zone_residual_summary(list(ipp2_row.get("ipp2_zone_residuals") or [])),
                "profile_max_residual": float(ipp2_row.get("ipp2_profile_max_residual_stops", ipp2_row.get("ipp2_residual_abs_stops", 0.0)) or 0.0),
                "status": str(ipp2_row.get("status") or ""),
                "original_image_path": ipp2_row.get("original_image_path"),
                "corrected_image_path": ipp2_row.get("corrected_image_path"),
            }
        )
        comparison_rows.append(
            {
                "clip_id": clip_id,
                "camera_id": _camera_id_from_clip_id(clip_id),
                "log_pre_render": float(((clip.get("metrics") or {}).get("exposure") or {}).get("measured_log2_luminance", 0.0) or 0.0),
                "ipp2_original_value": float(ipp2_row.get("ipp2_original_value_log2", 0.0) or 0.0),
                "ipp2_corrected_value": float(ipp2_row.get("ipp2_value_log2", 0.0) or 0.0),
                "predicted_ipp2_value": float(ipp2_row.get("predicted_ipp2_value_log2", 0.0) or 0.0),
                "applied_correction": float(ipp2_row.get("applied_correction_stops", 0.0) or 0.0),
                "gray_exposure": str(ipp2_row.get("ipp2_gray_exposure_summary") or "n/a"),
                "target_gray_exposure": str(ipp2_row.get("ipp2_target_gray_exposure_summary") or "n/a"),
                "zone_residuals": _format_zone_residual_summary(list(ipp2_row.get("ipp2_zone_residuals") or [])),
                "delta_log_to_ipp2_original": float(ipp2_row.get("ipp2_original_value_log2", 0.0) or 0.0)
                - float(((clip.get("metrics") or {}).get("exposure") or {}).get("measured_log2_luminance", 0.0) or 0.0),
                "delta_log_residual_to_ipp2_residual": float(ipp2_row.get("log_vs_ipp2_residual_delta_stops", 0.0) or 0.0),
                "predicted_vs_actual_ipp2_delta": float(ipp2_row.get("predicted_vs_actual_ipp2_delta_stops", 0.0) or 0.0),
            }
        )

    render_input_path = out_root / "render_input_state.json"
    pre_render_path = out_root / "pre_render_log_values.json"
    post_render_path = out_root / "post_render_ipp2_values.json"
    comparison_path = out_root / "render_trace_comparison.json"
    render_input_path.write_text(json.dumps({"cameras": render_input_rows}, indent=2), encoding="utf-8")
    pre_render_path.write_text(json.dumps({"cameras": pre_render_rows}, indent=2), encoding="utf-8")
    post_render_path.write_text(json.dumps({"cameras": post_render_rows}, indent=2), encoding="utf-8")
    comparison_path.write_text(json.dumps({"cameras": comparison_rows}, indent=2), encoding="utf-8")
    return {
        "render_input_state_path": str(render_input_path),
        "pre_render_log_values_path": str(pre_render_path),
        "post_render_ipp2_values_path": str(post_render_path),
        "render_trace_comparison_path": str(comparison_path),
    }


def _build_strategy_payloads(
    analysis_records: List[Dict[str, object]],
    *,
    target_strategies: List[str],
    reference_clip_id: Optional[str],
    hero_clip_id: Optional[str] = None,
    target_type: Optional[str] = None,
    monitoring_measurements_by_clip: Optional[Dict[str, Dict[str, object]]] = None,
    matching_domain: str = "scene",
    quality_by_clip: Optional[Dict[str, Dict[str, object]]] = None,
    anchor_target_log2: Optional[float] = None,
    exposure_anchor_mode: Optional[str] = None,
    manual_target_stops: Optional[float] = None,
    manual_target_ire: Optional[float] = None,
) -> List[Dict[str, object]]:
    if not analysis_records:
        return []
    requested_matching_domain = _normalize_matching_domain(matching_domain)
    resolved_matching_domain = requested_matching_domain
    resolved_anchor_mode = normalize_exposure_anchor_mode(exposure_anchor_mode)
    manual_target_log2, manual_target_input_domain, manual_target_input_value = _manual_anchor_target_log2(
        manual_target_stops=manual_target_stops,
        manual_target_ire=manual_target_ire,
    )
    saturation_supported = _target_supports_saturation(target_type)
    resolved_quality_by_clip = quality_by_clip or {}
    resolved_measurement_cache: Dict[str, Dict[str, object]] = {}

    def measurement_for_record(record: Dict[str, object]) -> Dict[str, object]:
        clip_id = str(record["clip_id"])
        cached = resolved_measurement_cache.get(clip_id)
        if cached is None:
            cached = _measurement_values_for_record(
                record,
                matching_domain=resolved_matching_domain,
                monitoring_measurements_by_clip=monitoring_measurements_by_clip,
            )
            resolved_measurement_cache[clip_id] = cached
        return cached

    def quality_for_record(record: Dict[str, object]) -> Dict[str, object]:
        clip_id = str(record["clip_id"])
        if resolved_matching_domain == "perceptual":
            monitoring_quality = _monitoring_quality_for_measurement(measurement_for_record(record))
            if monitoring_quality:
                return monitoring_quality
        return dict(resolved_quality_by_clip.get(clip_id, {}) or {})

    def confidence_for_record(record: Dict[str, object]) -> float:
        quality = quality_for_record(record)
        return float(quality.get("confidence", record.get("confidence", 0.0)) or 0.0)

    def sample_log2_spread_for_record(record: Dict[str, object]) -> float:
        measured = measurement_for_record(record) if resolved_matching_domain == "perceptual" else record.get("diagnostics", {})
        quality = quality_for_record(record)
        return float(quality.get("neutral_sample_log2_spread", measured.get("neutral_sample_log2_spread", 0.0)) or 0.0)

    def sample_chroma_spread_for_record(record: Dict[str, object]) -> float:
        measured = measurement_for_record(record) if resolved_matching_domain == "perceptual" else record.get("diagnostics", {})
        quality = quality_for_record(record)
        return float(
            quality.get(
                "neutral_sample_chromaticity_spread",
                measured.get("neutral_sample_chromaticity_spread", 0.0),
            )
            or 0.0
        )

    def merged_flags_for_record(record: Dict[str, object]) -> List[str]:
        quality = quality_for_record(record)
        quality_flags = [str(flag) for flag in (quality.get("flags") or []) if str(flag).strip()]
        return list(dict.fromkeys([*(record.get("flags") or []), *quality_flags]))

    def choose_optimal_exposure_candidate() -> tuple[int, Dict[str, object]]:
        median_value = float(np.median(measured_log2))
        median_abs_dev = float(np.median(np.abs(measured_log2 - median_value))) if measured_log2.size else 0.0
        outlier_threshold = max(0.35, median_abs_dev * 2.0)
        cluster_info = _primary_exposure_cluster(
            measured_log2,
            gap_threshold=max(OPTIMAL_EXPOSURE_PRIMARY_CLUSTER_GAP, min(outlier_threshold, 0.45)),
        )
        primary_cluster_indices = {int(index) for index in cluster_info.get("indices", [])}
        candidate_rows: List[Dict[str, object]] = []
        screened_rows: List[Dict[str, object]] = []
        for index, record in enumerate(analysis_records):
            confidence = confidence_for_record(record)
            sample_log2_spread = sample_log2_spread_for_record(record)
            sample_chroma_spread = sample_chroma_spread_for_record(record)
            measured_value = float(measured_log2[index])
            chroma_penalty = float(np.linalg.norm(measured_chroma[index] - np.asarray(batch_target_rgb, dtype=np.float32))) * 1.5
            exposure_error = abs(measured_value - batch_target_log2)
            stability_penalty = min(sample_log2_spread / OPTIMAL_EXPOSURE_MAX_SAMPLE_LOG2_SPREAD, 2.0) * 0.20
            stability_penalty += min(sample_chroma_spread / OPTIMAL_EXPOSURE_MAX_SAMPLE_CHROMA_SPREAD, 2.0) * 0.10
            confidence_penalty = max(0.0, 1.0 - confidence) * 0.35
            reasons: List[str] = []
            if primary_cluster_indices and index not in primary_cluster_indices:
                reasons.append("outside primary exposure cluster")
            if abs(measured_value - median_value) > outlier_threshold:
                reasons.append("outside central exposure cluster")
            if confidence < OPTIMAL_EXPOSURE_MIN_CONFIDENCE:
                reasons.append("low confidence")
            if (
                "neutral_sample_exposure_spread_high" in merged_flags_for_record(record)
                or sample_log2_spread > OPTIMAL_EXPOSURE_MAX_SAMPLE_LOG2_SPREAD
            ):
                reasons.append("unstable neutral exposure samples")
            if (
                "neutral_sample_chromaticity_spread_high" in merged_flags_for_record(record)
                or sample_chroma_spread > OPTIMAL_EXPOSURE_MAX_SAMPLE_CHROMA_SPREAD
            ):
                reasons.append("unstable neutral chromaticity samples")
            row = {
                "index": index,
                "clip_id": str(record["clip_id"]),
                "measured_log2": measured_value,
                "confidence": confidence,
                "sample_log2_spread": sample_log2_spread,
                "sample_chroma_spread": sample_chroma_spread,
                "exposure_error": exposure_error,
                "stability_penalty": stability_penalty,
                "confidence_penalty": confidence_penalty,
                "chroma_penalty": chroma_penalty,
                "score": exposure_error + stability_penalty + confidence_penalty + chroma_penalty,
                "reasons": reasons,
            }
            if reasons:
                screened_rows.append(row)
            else:
                candidate_rows.append(row)

        fallback_mode = "trusted_anchor"
        if candidate_rows:
            chosen = min(candidate_rows, key=lambda item: (float(item["score"]), -float(item["confidence"]), float(item["exposure_error"])))
        else:
            fallback_candidates = [
                index
                for index in primary_cluster_indices
                if confidence_for_record(analysis_records[index]) >= (OPTIMAL_EXPOSURE_MIN_CONFIDENCE * 0.8)
            ]
            if not fallback_candidates:
                fallback_candidates = list(primary_cluster_indices) or list(range(len(analysis_records)))
            target_index = min(
                fallback_candidates,
                key=lambda item: (
                    abs(float(measured_log2[item]) - median_value),
                    -confidence_for_record(analysis_records[item]),
                    sample_log2_spread_for_record(analysis_records[item]),
                ),
            )
            chosen = {
                "index": target_index,
                "clip_id": str(analysis_records[target_index]["clip_id"]),
                "measured_log2": float(measured_log2[target_index]),
                "confidence": confidence_for_record(analysis_records[target_index]),
                "sample_log2_spread": sample_log2_spread_for_record(analysis_records[target_index]),
                "sample_chroma_spread": sample_chroma_spread_for_record(analysis_records[target_index]),
                "exposure_error": abs(float(measured_log2[target_index]) - batch_target_log2),
                "stability_penalty": 0.0,
                "confidence_penalty": 0.0,
                "chroma_penalty": 0.0,
                "score": abs(float(measured_log2[target_index]) - batch_target_log2),
                "reasons": ["no camera met the minimum trust criteria"],
            }
            fallback_mode = "median_fallback"

        anchor_reason = (
            "Matched to the most trustworthy camera already closest to the gray target."
            if fallback_mode == "trusted_anchor"
            else "No camera met the minimum trust criteria, so the strategy fell back to the most central camera instead of anchoring on an unreliable match."
        )
        diagnostics = {
            "selection_domain": resolved_matching_domain,
            "selection_domain_label": _matching_domain_label(resolved_matching_domain),
            "anchor_target_log2": batch_target_log2,
            "cluster_median_log2": median_value,
            "outlier_threshold_stops": outlier_threshold,
            "primary_cluster": cluster_info,
            "trusted_candidate_count": len(candidate_rows),
            "screened_candidate_count": len(screened_rows),
            "fallback_mode": fallback_mode,
            "anchor_clip_id": str(chosen["clip_id"]),
            "anchor_confidence": float(chosen["confidence"]),
            "anchor_score": float(chosen["score"]),
            "anchor_reason": anchor_reason,
            "scored_candidates": [
                {
                    "clip_id": str(row["clip_id"]),
                    "score": float(row["score"]),
                    "exposure_error": float(row["exposure_error"]),
                    "stability_penalty": float(row["stability_penalty"]),
                    "confidence_penalty": float(row["confidence_penalty"]),
                    "chroma_penalty": float(row["chroma_penalty"]),
                    "confidence": float(row["confidence"]),
                }
                for row in sorted(candidate_rows, key=lambda item: float(item["score"]))
            ],
            "screened_candidates": [
                {
                    "clip_id": str(row["clip_id"]),
                    "measured_log2": float(row["measured_log2"]),
                    "score": float(row["score"]),
                    "confidence": float(row["confidence"]),
                    "reasons": list(row["reasons"]),
                }
                for row in sorted(screened_rows, key=lambda item: float(item["score"]))
            ],
        }
        return int(chosen["index"]), diagnostics

    def measured_log2_for_record(record: Dict[str, object]) -> float:
        measurement = measurement_for_record(record)
        return float(measurement.get("display_scalar_log2", measurement["log2_luminance"]) or 0.0)

    def measured_rgb_for_record(record: Dict[str, object]) -> List[float]:
        return [float(value) for value in measurement_for_record(record)["rgb_chromaticity"]]

    def measured_saturation_for_record(record: Dict[str, object]) -> float:
        return float(measurement_for_record(record)["saturation_fraction"])

    measured_log2 = np.array(
        [measured_log2_for_record(record) for record in analysis_records],
        dtype=np.float32,
    )
    measured_chroma = np.array([measured_rgb_for_record(record) for record in analysis_records], dtype=np.float32)
    measured_saturation = np.array([measured_saturation_for_record(record) for record in analysis_records], dtype=np.float32)
    batch_target_log2 = float(anchor_target_log2) if anchor_target_log2 is not None else float(np.median(measured_log2))
    batch_target_rgb = [float(np.median(measured_chroma[:, index])) for index in range(3)]
    payloads: List[Dict[str, object]] = []
    for requested in target_strategies:
        strategy = normalize_target_strategy_name(requested)
        selection_diagnostics: Dict[str, object] = {}
        anchor_mode_for_strategy = strategy
        anchor_source = ""
        anchor_ire_summary = "n/a"
        if strategy == "median":
            target_log2 = float(np.median(measured_log2))
            target_rgb = [float(np.median(measured_chroma[:, index])) for index in range(3)]
            target_saturation = float(np.median(measured_saturation)) if measured_saturation.size and saturation_supported else 1.0
            resolved_reference = None
            resolved_hero = None
            strategy_summary = "Matched to the batch median target."
            anchor_mode_for_strategy = "median"
            anchor_source = "Group median"
            anchor_ire_summary = f"{_ire_from_log2_luminance(target_log2):.0f} IRE (derived scalar anchor)"
        elif strategy == "optimal_exposure":
            target_index, selection_diagnostics = choose_optimal_exposure_candidate()
            target_log2 = float(measured_log2[target_index])
            target_rgb = [float(value) for value in measured_chroma[target_index]]
            target_saturation = float(measured_saturation[target_index]) if measured_saturation.size and saturation_supported else 1.0
            resolved_reference = str(analysis_records[target_index]["clip_id"])
            resolved_hero = None
            screened_candidates = list(selection_diagnostics.get("screened_candidates") or [])
            if str(selection_diagnostics.get("fallback_mode")) == "median_fallback":
                strategy_summary = (
                    f"No camera met the minimum trust criteria, so the strategy fell back to the most central camera {resolved_reference}."
                )
            elif screened_candidates:
                screened_names = ", ".join(str(item["clip_id"]) for item in screened_candidates[:3])
                strategy_summary = (
                    f"Matched to camera {resolved_reference}, which was closest to the gray target while staying stable enough to trust. "
                    f"Screened out {len(screened_candidates)} less trustworthy candidate(s) such as {screened_names}."
                )
            else:
                strategy_summary = f"Matched to camera {resolved_reference}, which was already closest to the gray target."
            anchor_mode_for_strategy = "hero_clip"
            anchor_source = str(resolved_reference or "")
            anchor_ire_summary = f"{_ire_from_log2_luminance(target_log2):.0f} IRE (strategy-selected clip anchor)"
        elif strategy == "hero_camera":
            if not hero_clip_id:
                raise ValueError("hero-camera target strategy requires --hero-clip-id")
            matches = [record for record in analysis_records if str(record["clip_id"]) == hero_clip_id]
            if not matches:
                raise ValueError(f"hero clip not found in analysis records: {hero_clip_id}")
            hero_record = matches[0]
            target_log2 = measured_log2_for_record(hero_record)
            target_rgb = measured_rgb_for_record(hero_record)
            target_saturation = measured_saturation_for_record(hero_record) if saturation_supported else 1.0
            resolved_reference = hero_clip_id
            resolved_hero = hero_clip_id
            strategy_summary = f"Matched to hero camera {hero_clip_id}."
            anchor_mode_for_strategy = resolved_anchor_mode if resolved_anchor_mode in {"hero_camera", "hero_clip"} else "hero_camera"
            anchor_source = str(hero_clip_id)
            anchor_ire_summary = f"{_ire_from_log2_luminance(target_log2):.0f} IRE (hero anchor)"
        elif strategy == "manual_target":
            if manual_target_log2 is None:
                raise ValueError("manual-target anchor requires --manual-target-stops or --manual-target-ire")
            target_log2 = float(manual_target_log2)
            target_rgb = [float(np.median(measured_chroma[:, index])) for index in range(3)]
            target_saturation = float(np.median(measured_saturation)) if measured_saturation.size and saturation_supported else 1.0
            resolved_reference = None
            resolved_hero = None
            anchor_mode_for_strategy = "manual_target"
            anchor_source = (
                f"{float(manual_target_input_value):.0f} IRE"
                if manual_target_input_domain == "ire" and manual_target_input_value is not None
                else f"{float(manual_target_input_value or target_log2):+.3f} stops"
            )
            anchor_ire_summary = _format_manual_anchor_ire_summary(
                anchor_log2=target_log2,
                manual_target_ire=manual_target_ire,
            )
            strategy_summary = (
                f"Matched to manual target {anchor_source}."
                if anchor_source
                else "Matched to manual scalar target."
            )
        else:
            if not reference_clip_id:
                raise ValueError("manual target strategy requires --reference-clip-id")
            matches = [record for record in analysis_records if str(record["clip_id"]) == reference_clip_id]
            if not matches:
                raise ValueError(f"manual reference clip not found in analysis records: {reference_clip_id}")
            reference_record = matches[0]
            target_log2 = measured_log2_for_record(reference_record)
            target_rgb = measured_rgb_for_record(reference_record)
            target_saturation = measured_saturation_for_record(reference_record) if saturation_supported else 1.0
            resolved_reference = reference_clip_id
            resolved_hero = None
            strategy_summary = f"Matched to manual reference clip {reference_clip_id}."
            anchor_mode_for_strategy = "manual_clip"
            anchor_source = str(reference_clip_id)
            anchor_ire_summary = f"{_ire_from_log2_luminance(target_log2):.0f} IRE (manual clip anchor)"

        anchor_descriptor = _build_anchor_descriptor(
            strategy_key=strategy,
            anchor_mode=anchor_mode_for_strategy,
            anchor_source=anchor_source,
            anchor_scalar_value=float(target_log2),
            anchor_ire_summary=anchor_ire_summary,
            manual_target_input_domain=manual_target_input_domain if strategy == "manual_target" else None,
            manual_target_input_value=manual_target_input_value if strategy == "manual_target" else None,
        )

        strategy_clips = []
        wb_requests = []
        for record in analysis_records:
            measured = record.get("diagnostics", {})
            resolved_measurement = measurement_for_record(record)
            measured_rgb = measured_rgb_for_record(record)
            measured_monitoring_log2 = measured_log2_for_record(record)
            measured_saturation_fraction = measured_saturation_for_record(record)
            color_solution = solve_cdl_color_model(
                measured_rgb_chromaticity=measured_rgb,
                target_rgb_chromaticity=target_rgb,
                measured_saturation_fraction=measured_saturation_fraction,
                target_saturation_fraction=target_saturation,
                allow_saturation_adjustment=saturation_supported,
            )
            rgb_gains = [float(value) for value in color_solution["diagnostic_rgb_gains"]]
            color_lggs = dict(color_solution["color_model"])
            color_cdl = dict(color_solution["cdl"])
            is_hero_camera = bool(strategy == "hero_camera" and str(record["clip_id"]) == hero_clip_id)
            if is_hero_camera:
                rgb_gains = [1.0, 1.0, 1.0]
                color_lggs = identity_lggs()
                color_cdl = rgb_gains_to_cdl(rgb_gains)
            pre_exposure_residual = abs(0.0 if is_hero_camera else float(target_log2 - measured_monitoring_log2))
            wb_requests.append(
                {
                    "clip_id": str(record["clip_id"]),
                    "measured_rgb_chromaticity": measured_rgb,
                    "target_rgb_chromaticity": target_rgb,
                    "clip_metadata": record.get("clip_metadata"),
                    "rgb_gains": rgb_gains,
                    "confidence": confidence_for_record(record),
                    "sample_log2_spread": sample_log2_spread_for_record(record),
                    "sample_chromaticity_spread": sample_chroma_spread_for_record(record),
                    "is_hero_camera": is_hero_camera,
                }
            )
            strategy_clips.append(
                {
                    "clip_id": str(record["clip_id"]),
                    "group_key": str(record["group_key"]),
                    "source_path": record.get("source_path"),
                    "clip_metadata": record.get("clip_metadata"),
                    "display_scalar_log2": measured_monitoring_log2,
                    "display_scalar_domain": "display_ipp2" if resolved_matching_domain == "perceptual" else "scene_analysis",
                    "measured_log2_luminance": measured_monitoring_log2,
                    "measured_log2_luminance_monitoring": measured_monitoring_log2,
                    "measured_log2_luminance_raw": float(measured.get("measured_log2_luminance_raw", measured.get("measured_log2_luminance", 0.0))),
                    "measured_rgb_chromaticity": [float(value) for value in measured_rgb],
                    "monitoring_measurement_source": str(resolved_measurement["source"]),
                    "gray_exposure_summary": str(resolved_measurement.get("gray_exposure_summary") or measured.get("gray_exposure_summary") or "n/a"),
                    "zone_measurements": [dict(item) for item in list(resolved_measurement.get("zone_measurements") or [])],
                    "exposure_offset_stops": 0.0 if is_hero_camera else float(target_log2 - measured_monitoring_log2),
                    "camera_offset_from_anchor": 0.0 if is_hero_camera else float(target_log2 - measured_monitoring_log2),
                    "pre_exposure_residual_stops": pre_exposure_residual,
                    "post_exposure_residual_stops": 0.0,
                    "rgb_gains": rgb_gains,
                    "color_lggs": color_lggs,
                    "color_cdl": color_cdl,
                    "measured_saturation_fraction": measured_saturation_fraction,
                    "target_saturation_fraction": target_saturation,
                    "saturation_source": str(color_solution["saturation_source"]),
                    "saturation_supported": saturation_supported,
                    "confidence": confidence_for_record(record),
                    "flags": merged_flags_for_record(record),
                    "calibration_roi": measured.get("calibration_roi"),
                    "measurement_mode": measured.get("calibration_measurement_mode"),
                    "exposure_measurement_domain": resolved_matching_domain,
                    "color_measurement_domain": resolved_matching_domain,
                    "neutral_sample_count": int(measured.get("neutral_sample_count", 0) or 0),
                    "neutral_sample_log2_spread": sample_log2_spread_for_record(record),
                    "neutral_sample_chromaticity_spread": sample_chroma_spread_for_record(record),
                    "neutral_samples": list(measured.get("neutral_samples", []) or []),
                    "is_hero_camera": is_hero_camera,
                    "anchor_mode": str(anchor_descriptor["anchor_mode"]),
                    "anchor_source": str(anchor_descriptor["anchor_source"]),
                    "anchor_scalar_value": float(anchor_descriptor["anchor_scalar_value"]),
                    "anchor_ire_summary": str(anchor_descriptor["anchor_ire_summary"]),
                }
            )

        non_hero_wb_requests = [item for item in wb_requests if not bool(item.get("is_hero_camera"))]
        wb_model_solution = solve_white_balance_model_for_records(
            non_hero_wb_requests,
            target_rgb_chromaticity=target_rgb,
        )
        wb_by_clip = dict(wb_model_solution.get("clips", {}))
        for clip in strategy_clips:
            clip_id = str(clip["clip_id"])
            record = next(item for item in analysis_records if str(item["clip_id"]) == clip_id)
            measured = record.get("diagnostics", {})
            is_hero_camera = bool(clip.get("is_hero_camera"))
            rgb_gains = clip["rgb_gains"] if not is_hero_camera else [1.0, 1.0, 1.0]
            if is_hero_camera:
                as_shot_wb = extract_as_shot_white_balance(record.get("clip_metadata"))
                wb_solution = {
                    "kelvin": int(round(as_shot_wb["kelvin"])),
                    "tint": round(float(as_shot_wb["tint"]), 1),
                    "method": f"neutral_axis_{wb_model_solution.get('model_key', 'per_camera_kelvin_per_camera_tint')}_v2",
                    "model_key": wb_model_solution.get("model_key"),
                    "model_label": wb_model_solution.get("model_label"),
                    "as_shot_kelvin": int(round(as_shot_wb["kelvin"])),
                    "as_shot_tint": round(float(as_shot_wb["tint"]), 1),
                    "white_balance_axes": {"amber_blue": 0.0, "green_magenta": 0.0},
                    "predicted_white_balance_axes": {"amber_blue": 0.0, "green_magenta": 0.0},
                    "implied_rgb_gains": [1.0, 1.0, 1.0],
                    "pre_neutral_residual": 0.0,
                    "post_neutral_residual": 0.0,
                    "confidence_weight": 1.0,
                }
            else:
                wb_solution = wb_by_clip.get(clip_id)
            commit_values = build_commit_values(
                exposure_adjust=float(clip["exposure_offset_stops"]),
                rgb_gains=rgb_gains,
                confidence=float(clip.get("confidence", 0.0) or 0.0),
                sample_log2_spread=float(measured.get("neutral_sample_log2_spread", 0.0) or 0.0),
                sample_chromaticity_spread=float(measured.get("neutral_sample_chromaticity_spread", 0.0) or 0.0),
                measured_rgb_chromaticity=clip["measured_rgb_chromaticity"],
                target_rgb_chromaticity=target_rgb,
                clip_metadata=record.get("clip_metadata"),
                saturation=float(clip.get("color_lggs", {}).get("saturation", 1.0)),
                saturation_supported=saturation_supported,
                wb_solution=wb_solution,
            )
            clip["commit_values"] = commit_values
            clip["pre_color_residual"] = float(commit_values.get("pre_neutral_residual", 0.0) or 0.0)
            clip["post_color_residual"] = float(commit_values.get("post_neutral_residual", 0.0) or 0.0)
            clip["white_balance_model"] = str(commit_values.get("white_balance_model") or wb_model_solution.get("model_key"))
            clip["white_balance_model_label"] = str(commit_values.get("white_balance_model_label") or wb_model_solution.get("model_label"))
            clip["shared_kelvin"] = wb_model_solution.get("shared_kelvin")
            clip["shared_tint"] = wb_model_solution.get("shared_tint")

        payloads.append(
            {
                "strategy_key": strategy,
                "strategy_label": strategy_display_name(strategy),
                "reference_clip_id": resolved_reference,
                "hero_clip_id": resolved_hero,
                "target_log2_luminance": target_log2,
                "target_rgb_chromaticity": target_rgb,
                "target_saturation_fraction": target_saturation,
                "strategy_summary": strategy_summary,
                "anchor_mode": str(anchor_descriptor["anchor_mode"]),
                "anchor_mode_label": str(anchor_descriptor["anchor_mode_label"]),
                "anchor_source": str(anchor_descriptor["anchor_source"]),
                "anchor_scalar_value": float(anchor_descriptor["anchor_scalar_value"]),
                "anchor_ire_summary": str(anchor_descriptor["anchor_ire_summary"]),
                "anchor_summary": str(anchor_descriptor["anchor_summary"]),
                "manual_target_input_domain": anchor_descriptor.get("manual_target_input_domain"),
                "manual_target_input_value": anchor_descriptor.get("manual_target_input_value"),
                "selection_diagnostics": selection_diagnostics,
                "matching_domain": resolved_matching_domain,
                "matching_domain_label": _matching_domain_label(resolved_matching_domain),
                "white_balance_model": {
                    "model_key": wb_model_solution.get("model_key"),
                    "model_label": wb_model_solution.get("model_label"),
                    "shared_kelvin": wb_model_solution.get("shared_kelvin"),
                    "shared_tint": wb_model_solution.get("shared_tint"),
                    "metrics": wb_model_solution.get("metrics"),
                    "candidates": wb_model_solution.get("candidates", []),
                },
                "clips": strategy_clips,
            }
        )
        print(
            f"[r3dmatch] review strategy={strategy} target_monitoring_log2={target_log2:.6f} "
            f"reference={resolved_reference or 'median'}"
        )
        for clip in strategy_clips:
            print(
                f"[r3dmatch] review clip={clip['clip_id']} strategy={strategy} "
                f"monitoring_log2={clip['measured_log2_luminance_monitoring']:.6f} "
                f"offset={clip['exposure_offset_stops']:.6f}"
            )
    return payloads


def _camera_label_for_reporting(clip_id: str) -> str:
    parts = str(clip_id).split("_")
    return "_".join(parts[:2]) if len(parts) >= 2 else str(clip_id)


def _strategy_distribution_metrics(strategy_payload: Dict[str, object]) -> Dict[str, float]:
    offsets = np.array(
        [abs(float(clip.get("exposure_offset_stops", 0.0) or 0.0)) for clip in strategy_payload.get("clips", [])],
        dtype=np.float32,
    )
    color_residuals = np.array(
        [float(clip.get("pre_color_residual", 0.0) or 0.0) for clip in strategy_payload.get("clips", [])],
        dtype=np.float32,
    )
    confidence_penalty = np.array(
        [max(0.0, 1.0 - float(clip.get("confidence", 0.0) or 0.0)) for clip in strategy_payload.get("clips", [])],
        dtype=np.float32,
    )
    confidence_values = np.array(
        [float(clip.get("confidence", 0.0) or 0.0) for clip in strategy_payload.get("clips", [])],
        dtype=np.float32,
    )
    if offsets.size == 0:
        return {
            "mean_abs_offset": 0.0,
            "median_abs_offset": 0.0,
            "max_abs_offset": 0.0,
            "p90_abs_offset": 0.0,
            "mean_color_residual": 0.0,
            "max_color_residual": 0.0,
            "mean_confidence_penalty": 0.0,
            "mean_confidence": 0.0,
        }
    return {
        "mean_abs_offset": float(np.mean(offsets)),
        "median_abs_offset": float(np.median(offsets)),
        "max_abs_offset": float(np.max(offsets)),
        "p90_abs_offset": float(np.percentile(offsets, 90)),
        "mean_color_residual": float(np.mean(color_residuals)) if color_residuals.size else 0.0,
        "max_color_residual": float(np.max(color_residuals)) if color_residuals.size else 0.0,
        "mean_confidence_penalty": float(np.mean(confidence_penalty)) if confidence_penalty.size else 0.0,
        "mean_confidence": float(np.mean(confidence_values)) if confidence_values.size else 0.0,
    }


def _recommend_strategy(strategy_payloads: List[Dict[str, object]]) -> Dict[str, object]:
    scored: List[tuple[tuple[float, float, int], Dict[str, object], Dict[str, float]]] = []
    for payload in strategy_payloads:
        metrics = _strategy_distribution_metrics(payload)
        priority = STRATEGY_ORDER.index(str(payload["strategy_key"])) if str(payload["strategy_key"]) in STRATEGY_ORDER else len(STRATEGY_ORDER)
        scored.append(
            (
                (
                    metrics["mean_abs_offset"],
                    metrics["mean_color_residual"],
                    metrics["max_abs_offset"] + metrics["mean_confidence_penalty"],
                    priority,
                ),
                payload,
                metrics,
            )
        )
    _, winner, winner_metrics = min(scored, key=lambda item: item[0])
    reason = (
        f"{winner['strategy_label']} is recommended because it minimizes the average correction spread "
        f"({winner_metrics['mean_abs_offset']:.2f} stops mean absolute adjustment), keeps the pre-correction neutral residual "
        f"low ({winner_metrics['mean_color_residual']:.4f}), and avoids excessive worst-case corrections "
        f"({winner_metrics['max_abs_offset']:.2f} stops max)."
    )
    return {
        "strategy_key": winner["strategy_key"],
        "strategy_label": winner["strategy_label"],
        "metrics": winner_metrics,
        "reason": reason,
    }


def _ensure_anchor_strategy_list(
    target_strategies: List[str],
    *,
    exposure_anchor_mode: Optional[str],
) -> List[str]:
    normalized = [normalize_target_strategy_name(item) for item in (target_strategies or list(DEFAULT_REVIEW_TARGET_STRATEGIES))]
    anchor_mode = normalize_exposure_anchor_mode(exposure_anchor_mode)
    explicit_strategy = _strategy_key_for_anchor_mode(anchor_mode)
    if explicit_strategy and explicit_strategy not in normalized:
        normalized.append(explicit_strategy)
    ordered: List[str] = []
    for item in normalized:
        if item not in ordered:
            ordered.append(item)
    return ordered


def _choose_strategy_with_anchor(
    strategy_payloads: List[Dict[str, object]],
    *,
    explicit_anchor_strategy_key: Optional[str],
) -> Dict[str, object]:
    if explicit_anchor_strategy_key:
        chosen = next(
            (dict(item) for item in strategy_payloads if str(item.get("strategy_key") or "") == str(explicit_anchor_strategy_key)),
            None,
        )
        if chosen is not None:
            return {
                "strategy_key": str(chosen.get("strategy_key") or ""),
                "strategy_label": str(chosen.get("strategy_label") or strategy_display_name(str(chosen.get("strategy_key") or "median"))),
                "metrics": _strategy_distribution_metrics(chosen),
                "reason": (
                    f"{str(chosen.get('strategy_label') or 'Chosen anchor')} is being used because the exposure anchor was set explicitly "
                    f"to {str(chosen.get('anchor_summary') or chosen.get('anchor_source') or 'the selected reference')}."
                ),
                "anchor_mode": str(chosen.get("anchor_mode") or ""),
                "anchor_source": str(chosen.get("anchor_source") or ""),
                "anchor_scalar_value": chosen.get("anchor_scalar_value"),
                "anchor_ire_summary": chosen.get("anchor_ire_summary"),
                "anchor_summary": chosen.get("anchor_summary"),
                "anchor_explicit": True,
            }
    recommended = _recommend_strategy(strategy_payloads)
    strategy_match = next(
        (item for item in strategy_payloads if str(item.get("strategy_key") or "") == str(recommended.get("strategy_key") or "")),
        {},
    )
    return {
        **recommended,
        "anchor_mode": str(strategy_match.get("anchor_mode") or ""),
        "anchor_source": str(strategy_match.get("anchor_source") or ""),
        "anchor_scalar_value": strategy_match.get("anchor_scalar_value"),
        "anchor_ire_summary": strategy_match.get("anchor_ire_summary"),
        "anchor_summary": strategy_match.get("anchor_summary"),
        "anchor_explicit": False,
    }


def _hero_candidate_summary(strategy_payloads: List[Dict[str, object]]) -> Dict[str, object]:
    if not strategy_payloads:
        return {
            "candidate_clip_id": None,
            "confidence": "low",
            "reason": "No strategy payloads were available to nominate a hero candidate.",
        }
    clips = strategy_payloads[0].get("clips", [])
    exposures = np.array([float(clip.get("measured_log2_luminance_monitoring", 0.0) or 0.0) for clip in clips], dtype=np.float32)
    if exposures.size == 0:
        return {
            "candidate_clip_id": None,
            "confidence": "low",
            "reason": "No clip measurements were available to nominate a hero candidate.",
        }
    median_exposure = float(np.median(exposures))
    median_abs_dev = float(np.median(np.abs(exposures - median_exposure))) if exposures.size else 0.0
    tolerance = max(0.20, median_abs_dev * 2.0)
    best: Optional[Dict[str, object]] = None
    for clip in clips:
        measured = float(clip.get("measured_log2_luminance_monitoring", 0.0) or 0.0)
        chroma = np.array([float(value) for value in clip.get("measured_rgb_chromaticity", [1 / 3, 1 / 3, 1 / 3])], dtype=np.float32)
        chroma_center = np.median(
            np.array([item.get("measured_rgb_chromaticity", [1 / 3, 1 / 3, 1 / 3]) for item in clips], dtype=np.float32),
            axis=0,
        )
        confidence = float(clip.get("confidence", 0.0) or 0.0)
        outlier = abs(measured - median_exposure) > tolerance
        score = abs(measured - median_exposure) + float(np.linalg.norm(chroma - chroma_center)) * 2.5 + max(0.0, 0.8 - confidence)
        candidate = {
            "clip_id": str(clip["clip_id"]),
            "camera_label": _camera_label_for_reporting(str(clip["clip_id"])),
            "score": score,
            "confidence_value": confidence,
            "outlier": outlier,
            "exposure_distance": abs(measured - median_exposure),
        }
        if outlier:
            continue
        if best is None or float(candidate["score"]) < float(best["score"]):
            best = candidate
    if best is None:
        return {
            "candidate_clip_id": None,
            "confidence": "low",
            "reason": "No clear hero candidate emerged because all measured cameras behaved like exposure outliers or confidence was too weak.",
        }
    confidence_label = "high" if float(best["confidence_value"]) >= 0.95 else "medium" if float(best["confidence_value"]) >= 0.75 else "low"
    return {
        "candidate_clip_id": best["clip_id"],
        "camera_label": best["camera_label"],
        "confidence": confidence_label,
        "reason": (
            f"{best['clip_id']} is closest to the central exposure cluster "
            f"({float(best['exposure_distance']):.2f} stops from median) and is not flagged as an outlier."
        ),
    }


def _exposure_summary(clips: List[Dict[str, object]]) -> Dict[str, object]:
    exposures = np.array([float(clip.get("measured_log2_luminance_monitoring", 0.0) or 0.0) for clip in clips], dtype=np.float32)
    if exposures.size == 0:
        return {"median": 0.0, "minimum": 0.0, "maximum": 0.0, "spread": 0.0, "outlier_count": 0}
    median_value = float(np.median(exposures))
    median_abs_dev = float(np.median(np.abs(exposures - median_value))) if exposures.size else 0.0
    outlier_threshold = max(0.35, median_abs_dev * 2.0)
    outlier_count = int(np.sum(np.abs(exposures - median_value) > outlier_threshold))
    return {
        "median": median_value,
        "minimum": float(np.min(exposures)),
        "maximum": float(np.max(exposures)),
        "spread": float(np.max(exposures) - np.min(exposures)),
        "outlier_count": outlier_count,
        "outlier_threshold": outlier_threshold,
    }


def _primary_exposure_cluster(exposures: np.ndarray, *, gap_threshold: float) -> Dict[str, object]:
    values = np.asarray(exposures, dtype=np.float32).reshape(-1)
    if values.size == 0:
        return {"indices": [], "groups": [], "gap_threshold": float(gap_threshold)}
    ordered = sorted(enumerate(float(item) for item in values.tolist()), key=lambda item: item[1])
    groups: List[List[tuple[int, float]]] = []
    current: List[tuple[int, float]] = [ordered[0]]
    for previous, current_item in zip(ordered, ordered[1:]):
        if abs(float(current_item[1]) - float(previous[1])) > float(gap_threshold):
            groups.append(current)
            current = [current_item]
        else:
            current.append(current_item)
    groups.append(current)
    median_value = float(np.median(values))
    largest_group_size = max((len(group) for group in groups), default=0)
    if len(groups) <= 1 or largest_group_size < 2:
        return {
            "indices": [int(index) for index, _value in ordered],
            "groups": [
                {
                    "indices": [int(index) for index, _value in group],
                    "minimum": float(min(value for _, value in group)),
                    "maximum": float(max(value for _, value in group)),
                    "size": len(group),
                }
                for group in groups
            ],
            "gap_threshold": float(gap_threshold),
        }
    primary_group = max(
        groups,
        key=lambda group: (
            len(group),
            -min(abs(float(value) - median_value) for _, value in group),
        ),
    )
    return {
        "indices": [int(index) for index, _value in primary_group],
        "groups": [
            {
                "indices": [int(index) for index, _value in group],
                "minimum": float(min(value for _, value in group)),
                "maximum": float(max(value for _, value in group)),
                "size": len(group),
            }
            for group in groups
        ],
        "gap_threshold": float(gap_threshold),
    }


def _svg_truncate(text: str, limit: int) -> str:
    cleaned = " ".join(str(text).split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: max(limit - 1, 1)].rstrip() + "…"


def _svg_wrapped_text(
    text: str,
    *,
    x: float,
    y: float,
    width_chars: int,
    max_lines: int,
    font_size: int = 11,
    fill: str = "#64748b",
    anchor: str = "start",
    line_height: int = 14,
) -> str:
    words = " ".join(str(text).split()).split()
    if not words:
        return ""
    lines: List[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) <= width_chars or not current:
            current = candidate
            continue
        lines.append(current)
        current = word
        if len(lines) >= max_lines - 1:
            break
    if current and len(lines) < max_lines:
        lines.append(current)
    remaining_words = words[len(" ".join(lines).split()):]
    if remaining_words and lines:
        lines[-1] = _svg_truncate(lines[-1], max(width_chars - 1, 1))
    tspans = []
    for index, line in enumerate(lines[:max_lines]):
        tspans.append(
            f"<tspan x=\"{x:.1f}\" dy=\"{0 if index == 0 else line_height}\" text-anchor=\"{anchor}\">{html.escape(line)}</tspan>"
        )
    return f"<text x=\"{x:.1f}\" y=\"{y:.1f}\" font-size=\"{font_size}\" fill=\"{fill}\">{''.join(tspans)}</text>"


def _build_exposure_plot_svg(
    clips: List[Dict[str, object]],
    *,
    target_log2: Optional[float] = None,
    reference_clip_id: Optional[str] = None,
    outlier_threshold: Optional[float] = None,
) -> str:
    if not clips:
        return ""
    ordered = sorted(clips, key=lambda clip: float(clip.get("measured_log2_luminance_monitoring", 0.0) or 0.0), reverse=True)
    values = [float(clip.get("measured_log2_luminance_monitoring", 0.0) or 0.0) for clip in ordered]
    minimum = min(values)
    maximum = max(values)
    median_value = float(np.median(np.asarray(values, dtype=np.float32)))
    threshold = float(outlier_threshold or max(0.35, float(np.median(np.abs(np.asarray(values, dtype=np.float32) - median_value))) * 2.0))
    lower_cluster = median_value - threshold
    upper_cluster = median_value + threshold
    width = 1180
    height = 500
    pad_left = 96
    pad_right = 68
    pad_top = 42
    pad_bottom = 96
    inner_width = width - pad_left - pad_right
    inner_height = height - pad_top - pad_bottom
    scale = (maximum - minimum) or 1.0
    step = inner_width / max(len(ordered) - 1, 1)
    def y_for(value: float) -> float:
        return pad_top + inner_height - ((value - minimum) / scale) * inner_height

    points: List[str] = []
    labels: List[str] = []
    stems: List[str] = []
    for index, clip in enumerate(ordered):
        x = pad_left + step * index
        value = float(clip.get("measured_log2_luminance_monitoring", 0.0) or 0.0)
        y = y_for(value)
        points.append(f"{x:.1f},{y:.1f}")
        if target_log2 is not None:
            target_y = y_for(float(target_log2))
            stems.append(f"<line x1=\"{x:.1f}\" y1=\"{y:.1f}\" x2=\"{x:.1f}\" y2=\"{target_y:.1f}\" stroke=\"#cbd5e1\" stroke-width=\"2\" stroke-dasharray=\"4 4\"/>")
        labels.append(
            _svg_wrapped_text(
                _camera_label_for_reporting(str(clip["clip_id"])),
                x=x,
                y=height - 44,
                width_chars=12,
                max_lines=2,
                font_size=11,
                fill="#475569",
                anchor="middle",
                line_height=13,
            )
        )
    polyline = " ".join(points)
    grid_lines = []
    for tick in range(4):
        y = pad_top + (inner_height / 3.0) * tick
        value = maximum - ((maximum - minimum) / 3.0) * tick
        grid_lines.append(f"<line x1=\"{pad_left}\" y1=\"{y:.1f}\" x2=\"{width - pad_right}\" y2=\"{y:.1f}\" stroke=\"#e2e8f0\" stroke-width=\"1\"/>")
        grid_lines.append(f"<text x=\"{pad_left - 12}\" y=\"{y + 4:.1f}\" text-anchor=\"end\" font-size=\"11\" fill=\"#64748b\">{value:.2f}</text>")
    cluster_band = (
        f"<rect x=\"{pad_left}\" y=\"{y_for(upper_cluster):.1f}\" width=\"{inner_width:.1f}\" height=\"{max(y_for(lower_cluster) - y_for(upper_cluster), 6):.1f}\" fill=\"#dbeafe\" opacity=\"0.45\" rx=\"14\"/>"
    )
    cluster_line = f"<line x1=\"{pad_left}\" y1=\"{y_for(median_value):.1f}\" x2=\"{width - pad_right}\" y2=\"{y_for(median_value):.1f}\" stroke=\"#1d4ed8\" stroke-width=\"2\" stroke-dasharray=\"6 5\"/>"
    target_line = (
        f"<line x1=\"{pad_left}\" y1=\"{y_for(float(target_log2)):.1f}\" x2=\"{width - pad_right}\" y2=\"{y_for(float(target_log2)):.1f}\" stroke=\"#0f766e\" stroke-width=\"2\"/>"
        if target_log2 is not None
        else ""
    )
    legend = (
        f"<text x=\"{pad_left}\" y=\"24\" font-size=\"13\" font-weight=\"700\" fill=\"#1d4ed8\">Blue band: stable cluster</text>"
        f"<text x=\"{pad_left + 210}\" y=\"24\" font-size=\"13\" font-weight=\"700\" fill=\"#0f766e\">Green line: target</text>"
        f"<text x=\"{pad_left + 390}\" y=\"24\" font-size=\"13\" font-weight=\"700\" fill=\"#dc2626\">Red point: outlier</text>"
        f"<text x=\"{pad_left}\" y=\"{max(y_for(median_value) - 10, 40):.1f}\" font-size=\"11\" fill=\"#1d4ed8\">stable cluster band</text>"
        + (
            f"<text x=\"{width - pad_right - 110}\" y=\"{max(y_for(float(target_log2)) - 10, 40):.1f}\" font-size=\"11\" fill=\"#0f766e\">target exposure</text>"
            if target_log2 is not None
            else ""
        )
    )
    circles = []
    for clip, point in zip(ordered, points):
        x, y = point.split(",")
        value = float(clip.get("measured_log2_luminance_monitoring", 0.0) or 0.0)
        is_outlier = abs(value - median_value) > threshold
        is_anchor = reference_clip_id and str(clip.get("clip_id")) == str(reference_clip_id)
        fill = "#f59e0b" if is_anchor else "#dc2626" if is_outlier else "#0f172a"
        radius = 6 if is_anchor else 5
        circles.append(f"<circle cx=\"{x}\" cy=\"{y}\" r=\"{radius}\" fill=\"{fill}\" stroke=\"#ffffff\" stroke-width=\"2\"/>")
    return (
        f"<svg viewBox=\"0 0 {width} {height}\" role=\"img\" aria-label=\"Per-camera exposure plot\">"
        f"{''.join(grid_lines)}"
        f"{cluster_band}"
        f"{cluster_line}"
        f"{target_line}"
        f"{legend}"
        f"{''.join(stems)}"
        f"<polyline fill=\"none\" stroke=\"#2563eb\" stroke-width=\"3\" points=\"{polyline}\"/>"
        f"{''.join(circles)}"
        f"{''.join(labels)}"
        "</svg>"
    )


def _build_before_after_exposure_svg(clips: List[Dict[str, object]], *, target_log2: float) -> str:
    if not clips:
        return ""
    ordered = sorted(clips, key=lambda clip: float(abs(clip.get("exposure_offset_stops", 0.0) or 0.0)), reverse=True)
    values = [float(clip.get("measured_log2_luminance_monitoring", 0.0) or 0.0) for clip in ordered] + [float(target_log2)]
    minimum = min(values)
    maximum = max(values)
    width = 1180
    row_height = 54
    height = 130 + row_height * len(ordered)
    pad_left = 176
    pad_right = 76
    pad_top = 40
    inner_width = width - pad_left - pad_right
    scale = (maximum - minimum) or 1.0

    def x_for(value: float) -> float:
        return pad_left + ((value - minimum) / scale) * inner_width

    rows: List[str] = []
    for index, clip in enumerate(ordered):
        y = pad_top + index * row_height + 12
        measured = float(clip.get("measured_log2_luminance_monitoring", 0.0) or 0.0)
        measured_x = x_for(measured)
        target_x = x_for(float(target_log2))
        rows.append(
            _svg_wrapped_text(
                _camera_label_for_reporting(str(clip["clip_id"])),
                x=pad_left - 14,
                y=y - 6,
                width_chars=18,
                max_lines=2,
                font_size=11,
                fill="#475569",
                anchor="end",
                line_height=13,
            )
            +
            f"<line x1=\"{measured_x:.1f}\" y1=\"{y:.1f}\" x2=\"{target_x:.1f}\" y2=\"{y:.1f}\" stroke=\"#94a3b8\" stroke-width=\"3\"/>"
            f"<circle cx=\"{measured_x:.1f}\" cy=\"{y:.1f}\" r=\"5\" fill=\"#2563eb\"/>"
            f"<circle cx=\"{target_x:.1f}\" cy=\"{y:.1f}\" r=\"5\" fill=\"#0f766e\"/>"
        )
    axis = []
    for tick in range(4):
        value = minimum + ((maximum - minimum) / 3.0) * tick
        x = x_for(value)
        axis.append(f"<line x1=\"{x:.1f}\" y1=\"{pad_top - 10}\" x2=\"{x:.1f}\" y2=\"{height - 18}\" stroke=\"#e2e8f0\" stroke-width=\"1\"/>")
        axis.append(f"<text x=\"{x:.1f}\" y=\"{height - 4}\" text-anchor=\"middle\" font-size=\"11\" fill=\"#64748b\">{value:.2f}</text>")
    legend = (
        f"<text x=\"{pad_left}\" y=\"24\" font-size=\"13\" font-weight=\"700\" fill=\"#2563eb\">Blue dot: measured</text>"
        f"<text x=\"{pad_left + 176}\" y=\"24\" font-size=\"13\" font-weight=\"700\" fill=\"#0f766e\">Green dot: target</text>"
    )
    return f"<svg viewBox=\"0 0 {width} {height}\" role=\"img\" aria-label=\"Before and after exposure correction chart\">{legend}{''.join(axis)}{''.join(rows)}</svg>"


def _build_confidence_chart_svg(clips: List[Dict[str, object]]) -> str:
    if not clips:
        return ""
    ordered = sorted(clips, key=lambda clip: str(clip.get("clip_id") or ""))
    width = 1180
    height = 500
    pad_left = 74
    pad_right = 36
    pad_top = 32
    pad_bottom = 92
    inner_width = width - pad_left - pad_right
    inner_height = height - pad_top - pad_bottom
    step = inner_width / max(len(ordered), 1)
    bars: List[str] = []
    labels: List[str] = []
    guides: List[str] = []
    for value, label, color in [(0.75, "trusted", "#94a3b8"), (0.5, "review", "#cbd5e1"), (0.35, "minimum", "#dc2626")]:
        y = pad_top + inner_height - (inner_height * value)
        stroke_width = "2" if value == 0.35 else "1"
        guides.append(f"<line x1=\"{pad_left}\" y1=\"{y:.1f}\" x2=\"{width - pad_right}\" y2=\"{y:.1f}\" stroke=\"{color}\" stroke-width=\"{stroke_width}\" stroke-dasharray=\"6 4\"/>")
        guides.append(f"<text x=\"{width - pad_right}\" y=\"{y - 6:.1f}\" text-anchor=\"end\" font-size=\"11\" fill=\"{color}\">{label} {value:.2f}</text>")
    for index, clip in enumerate(ordered):
        x = pad_left + index * step + step * 0.15
        bar_width = step * 0.7
        confidence = max(0.0, min(1.0, float(clip.get("confidence", 0.0) or 0.0)))
        bar_height = inner_height * confidence
        y = pad_top + inner_height - bar_height
        fill = "#16a34a" if confidence >= 0.75 else "#f59e0b" if confidence >= 0.5 else "#dc2626"
        labels.append(
            _svg_wrapped_text(
                _camera_label_for_reporting(str(clip["clip_id"])),
                x=x + bar_width / 2,
                y=height - 46,
                width_chars=12,
                max_lines=2,
                font_size=11,
                fill="#475569",
                anchor="middle",
                line_height=13,
            )
            +
            f"<text x=\"{x + bar_width / 2:.1f}\" y=\"{y - 6:.1f}\" text-anchor=\"middle\" font-size=\"10\" fill=\"#475569\">{confidence:.2f}</text>"
        )
        bars.append(f"<rect x=\"{x:.1f}\" y=\"{y:.1f}\" width=\"{bar_width:.1f}\" height=\"{max(bar_height, 4):.1f}\" rx=\"8\" fill=\"{fill}\"/>")
    baseline = f"<line x1=\"{pad_left}\" y1=\"{pad_top + inner_height:.1f}\" x2=\"{width - pad_right}\" y2=\"{pad_top + inner_height:.1f}\" stroke=\"#cbd5e1\" stroke-width=\"1\"/>"
    return f"<svg viewBox=\"0 0 {width} {height}\" role=\"img\" aria-label=\"Per-camera confidence chart\">{baseline}{''.join(guides)}{''.join(bars)}{''.join(labels)}</svg>"


def _build_strategy_chart_svg(strategy_summaries: List[Dict[str, object]]) -> str:
    if not strategy_summaries:
        return ""
    width = 1180
    row_height = 86
    height = 92 + row_height * len(strategy_summaries)
    max_value = max(float(item["correction_metrics"]["max_abs_offset"]) for item in strategy_summaries) or 1.0
    bars: List[str] = []
    for index, item in enumerate(strategy_summaries):
        y = 32 + index * row_height
        width_value = 520 * (float(item["correction_metrics"]["mean_abs_offset"]) / max_value)
        confidence_text = float(item["correction_metrics"].get("mean_confidence", 0.0) or 0.0)
        anchor_reason = str(((item.get("selection_diagnostics") or {}).get("anchor_reason")) or "")
        strategy_role = "stability anchor" if str(item.get("strategy_key") or "") == "median" else "gray-match anchor" if str(item.get("strategy_key") or "") == "optimal_exposure" else "review anchor"
        recommended_fill = "#0f766e" if item.get("recommended") else "#2563eb"
        bars.append(
            _svg_wrapped_text(
                str(item["strategy_label"]),
                x=28,
                y=y + 10,
                width_chars=18,
                max_lines=2,
                font_size=13,
                fill="#0f172a",
                anchor="start",
                line_height=14,
            )
            +
            f"<rect x=\"280\" y=\"{y}\" width=\"560\" height=\"22\" rx=\"10\" fill=\"#e2e8f0\"/>"
            f"<rect x=\"280\" y=\"{y}\" width=\"{max(width_value, 4):.1f}\" height=\"22\" rx=\"10\" fill=\"{recommended_fill}\"/>"
            f"<text x=\"860\" y=\"{y + 16}\" font-size=\"12\" fill=\"#475569\">mean {float(item['correction_metrics']['mean_abs_offset']):.2f} / max {float(item['correction_metrics']['max_abs_offset']):.2f} | trust {confidence_text:.2f}</text>"
            f"<text x=\"280\" y=\"{y + 48}\" font-size=\"11\" fill=\"#334155\">{html.escape(strategy_role)}</text>"
            f"{_svg_wrapped_text(anchor_reason, x=400, y=y + 48, width_chars=66, max_lines=2, font_size=11, fill='#64748b', anchor='start', line_height=13)}"
        )
    return f"<svg viewBox=\"0 0 {width} {height}\" role=\"img\" aria-label=\"Strategy correction spread chart\">{''.join(bars)}</svg>"


def _trust_tone_for_class(trust_class: str) -> str:
    normalized = str(trust_class or "").strip().upper()
    if normalized == "TRUSTED":
        return "#16a34a"
    if normalized == "USE_WITH_CAUTION":
        return "#f59e0b"
    if normalized == "UNTRUSTED":
        return "#ef4444"
    if normalized == "EXCLUDED":
        return "#7c2d12"
    return "#64748b"


def _camera_trust_details(
    *,
    clip_id: str,
    confidence: float,
    sample_log2_spread: float,
    sample_chroma_spread: float,
    measured_log2: float,
    final_offset: float,
    exposure_summary: Dict[str, object],
    in_primary_cluster: bool,
    screened_reasons: List[str],
) -> Dict[str, object]:
    median_value = float(exposure_summary.get("median", measured_log2) or measured_log2)
    outlier_threshold = float(exposure_summary.get("outlier_threshold", 0.35) or 0.35)
    outside_central_cluster = abs(measured_log2 - median_value) > outlier_threshold
    low_confidence = confidence < OPTIMAL_EXPOSURE_MIN_CONFIDENCE
    caution_confidence = confidence < CAMERA_TRUST_CAUTION_CONFIDENCE
    unstable_log2 = sample_log2_spread > OPTIMAL_EXPOSURE_MAX_SAMPLE_LOG2_SPREAD
    unstable_chroma = sample_chroma_spread > OPTIMAL_EXPOSURE_MAX_SAMPLE_CHROMA_SPREAD
    elevated_log2 = sample_log2_spread > CAMERA_TRUST_CAUTION_LOG2_SPREAD
    elevated_chroma = sample_chroma_spread > CAMERA_TRUST_CAUTION_CHROMA_SPREAD
    large_correction = abs(final_offset) >= CAMERA_TRUST_LARGE_CORRECTION
    extreme_correction = abs(final_offset) >= CAMERA_TRUST_EXTREME_CORRECTION

    reasons: List[str] = []
    if not in_primary_cluster:
        reasons.append("Outside primary exposure group")
    if outside_central_cluster and "Outside primary exposure group" not in reasons:
        reasons.append("Outside stable exposure cluster")
    if unstable_log2:
        reasons.append("Unstable gray sample")
    elif elevated_log2:
        reasons.append("Gray sample spread is elevated")
    if unstable_chroma:
        reasons.append("Neutral color sample is unstable")
    elif elevated_chroma:
        reasons.append("Neutral color spread is elevated")
    if low_confidence:
        reasons.append("Low confidence")
    elif caution_confidence:
        reasons.append("Confidence is moderate")
    if extreme_correction:
        reasons.append("Large correction needed")
    elif large_correction:
        reasons.append("Correction is larger than the main cluster")
    for reason in screened_reasons:
        normalized_reason = str(reason).strip()
        if normalized_reason and normalized_reason not in reasons:
            reasons.append(normalized_reason)

    trust_penalty = max(0.0, 1.0 - confidence) * 0.45
    trust_penalty += min(sample_log2_spread / OPTIMAL_EXPOSURE_MAX_SAMPLE_LOG2_SPREAD, 2.0) * 0.22
    trust_penalty += min(sample_chroma_spread / OPTIMAL_EXPOSURE_MAX_SAMPLE_CHROMA_SPREAD, 2.0) * 0.14
    trust_penalty += min(abs(final_offset) / 1.25, 1.5) * 0.12
    if not in_primary_cluster:
        trust_penalty += 0.22
    if outside_central_cluster:
        trust_penalty += 0.14
    trust_score = max(0.0, min(1.0, 1.0 - trust_penalty))

    if (
        (not in_primary_cluster)
        or outside_central_cluster
        or (extreme_correction and (low_confidence or unstable_log2 or unstable_chroma))
    ):
        trust_class = "EXCLUDED"
    elif low_confidence or unstable_log2 or unstable_chroma or extreme_correction:
        trust_class = "UNTRUSTED"
    elif caution_confidence or elevated_log2 or elevated_chroma or large_correction:
        trust_class = "USE_WITH_CAUTION"
    else:
        trust_class = "TRUSTED"

    if trust_class == "TRUSTED":
        trust_reason = "Stable gray sample"
        stability_label = "Stable gray sample"
        correction_confidence = "HIGH"
    elif trust_class == "USE_WITH_CAUTION":
        trust_reason = reasons[0] if reasons else "Use with caution"
        stability_label = "Slightly unstable reading"
        correction_confidence = "MEDIUM"
    elif trust_class == "UNTRUSTED":
        trust_reason = reasons[0] if reasons else "Unstable reading"
        stability_label = "Unstable reading"
        correction_confidence = "LOW"
    else:
        trust_reason = reasons[0] if reasons else "Excluded due to inconsistent measurement"
        stability_label = "Excluded from reference"
        correction_confidence = "LOW"

    return {
        "trust_class": trust_class,
        "trust_score": round(trust_score, 4),
        "trust_reason": trust_reason,
        "stability_label": stability_label,
        "correction_confidence": correction_confidence,
        "reference_use": "Excluded" if trust_class == "EXCLUDED" else "Included",
        "screened_reasons": reasons,
        "outside_central_cluster": outside_central_cluster,
        "outside_primary_cluster": not in_primary_cluster,
    }


def _build_trust_chart_svg(rows: List[Dict[str, object]]) -> str:
    if not rows:
        return ""
    ordered = sorted(rows, key=lambda item: (str(item.get("trust_class") or ""), -float(item.get("trust_score", 0.0) or 0.0), str(item.get("camera_label") or "")))
    width = 1180
    height = 500
    pad_left = 74
    pad_right = 36
    pad_top = 34
    pad_bottom = 96
    inner_width = width - pad_left - pad_right
    inner_height = height - pad_top - pad_bottom
    step = inner_width / max(len(ordered), 1)
    bars: List[str] = []
    labels: List[str] = []
    guides: List[str] = []
    for value, label, color in [(0.75, "trusted", "#16a34a"), (0.5, "caution", "#f59e0b"), (0.35, "review", "#dc2626")]:
        y = pad_top + inner_height - (inner_height * value)
        guides.append(f"<line x1=\"{pad_left}\" y1=\"{y:.1f}\" x2=\"{width - pad_right}\" y2=\"{y:.1f}\" stroke=\"{color}\" stroke-width=\"1\" stroke-dasharray=\"6 4\"/>")
        guides.append(f"<text x=\"{width - pad_right}\" y=\"{y - 6:.1f}\" text-anchor=\"end\" font-size=\"11\" fill=\"{color}\">{label} {value:.2f}</text>")
    for index, row in enumerate(ordered):
        x = pad_left + index * step + step * 0.15
        bar_width = step * 0.7
        trust_score = max(0.0, min(1.0, float(row.get("trust_score", 0.0) or 0.0)))
        bar_height = inner_height * trust_score
        y = pad_top + inner_height - bar_height
        fill = _trust_tone_for_class(str(row.get("trust_class") or ""))
        labels.append(
            _svg_wrapped_text(
                str(row.get("camera_label") or row.get("clip_id") or ""),
                x=x + bar_width / 2,
                y=height - 50,
                width_chars=12,
                max_lines=2,
                font_size=11,
                fill="#475569",
                anchor="middle",
                line_height=13,
            )
            + f"<text x=\"{x + bar_width / 2:.1f}\" y=\"{y - 8:.1f}\" text-anchor=\"middle\" font-size=\"10\" fill=\"#334155\">{html.escape(str(row.get('trust_class') or ''))}</text>"
        )
        bars.append(f"<rect x=\"{x:.1f}\" y=\"{y:.1f}\" width=\"{bar_width:.1f}\" height=\"{max(bar_height, 4):.1f}\" rx=\"8\" fill=\"{fill}\"/>")
    baseline = f"<line x1=\"{pad_left}\" y1=\"{pad_top + inner_height:.1f}\" x2=\"{width - pad_right}\" y2=\"{pad_top + inner_height:.1f}\" stroke=\"#cbd5e1\" stroke-width=\"1\"/>"
    return f"<svg viewBox=\"0 0 {width} {height}\" role=\"img\" aria-label=\"Per-camera trust chart\">{baseline}{''.join(guides)}{''.join(bars)}{''.join(labels)}</svg>"


def _build_stability_chart_svg(rows: List[Dict[str, object]]) -> str:
    if not rows:
        return ""
    ordered = sorted(
        rows,
        key=lambda item: (
            float(item.get("neutral_sample_log2_spread", 0.0) or 0.0),
            float(item.get("neutral_sample_chromaticity_spread", 0.0) or 0.0),
            -float(item.get("confidence", 0.0) or 0.0),
        ),
    )
    width = 1180
    row_height = 62
    height = 94 + row_height * len(ordered)
    max_value = max(float(item.get("neutral_sample_log2_spread", 0.0) or 0.0) for item in ordered) or OPTIMAL_EXPOSURE_MAX_SAMPLE_LOG2_SPREAD
    rows_svg: List[str] = []
    threshold_x = 280 + 520 * min(OPTIMAL_EXPOSURE_MAX_SAMPLE_LOG2_SPREAD / max_value, 1.0)
    rows_svg.append(f"<line x1=\"{threshold_x:.1f}\" y1=\"22\" x2=\"{threshold_x:.1f}\" y2=\"{height - 20}\" stroke=\"#dc2626\" stroke-width=\"2\" stroke-dasharray=\"6 4\"/>")
    rows_svg.append(f"<text x=\"{threshold_x + 8:.1f}\" y=\"24\" font-size=\"11\" fill=\"#dc2626\">stability threshold</text>")
    for index, item in enumerate(ordered):
        y = 40 + index * row_height
        spread = float(item.get("neutral_sample_log2_spread", 0.0) or 0.0)
        width_value = 520 * min(spread / max_value, 1.0)
        fill = _trust_tone_for_class(str(item.get("trust_class") or ""))
        rows_svg.append(
            _svg_wrapped_text(
                str(item.get("camera_label") or item.get("clip_id") or ""),
                x=28,
                y=y + 8,
                width_chars=18,
                max_lines=2,
                font_size=13,
                fill="#0f172a",
                anchor="start",
                line_height=14,
            )
            + f"<rect x=\"280\" y=\"{y}\" width=\"560\" height=\"20\" rx=\"10\" fill=\"#e2e8f0\"/>"
            + f"<rect x=\"280\" y=\"{y}\" width=\"{max(width_value, 4):.1f}\" height=\"20\" rx=\"10\" fill=\"{fill}\"/>"
            + f"<text x=\"860\" y=\"{y + 14}\" font-size=\"12\" fill=\"#475569\">spread {spread:.3f} | {html.escape(str(item.get('stability_label') or ''))}</text>"
        )
    return f"<svg viewBox=\"0 0 {width} {height}\" role=\"img\" aria-label=\"Per-camera stability ranking chart\">{''.join(rows_svg)}</svg>"


def _build_run_assessment(
    *,
    per_camera_rows: List[Dict[str, object]],
    recommended_payload: Dict[str, object],
    strategy_summaries: List[Dict[str, object]],
    exposure_summary: Dict[str, object],
) -> Dict[str, object]:
    total = len(per_camera_rows)
    trusted = [row for row in per_camera_rows if str(row.get("trust_class") or "") == "TRUSTED"]
    caution = [row for row in per_camera_rows if str(row.get("trust_class") or "") == "USE_WITH_CAUTION"]
    untrusted = [row for row in per_camera_rows if str(row.get("trust_class") or "") == "UNTRUSTED"]
    excluded = [row for row in per_camera_rows if str(row.get("trust_class") or "") == "EXCLUDED"]
    reference_eligible = trusted + caution
    trusted_count = len(trusted)
    excluded_count = len(excluded)
    reference_eligible_count = len(reference_eligible)
    average_trust_score = round(
        float(np.mean([float(row.get("trust_score", 0.0) or 0.0) for row in per_camera_rows])) if per_camera_rows else 0.0,
        4,
    )
    minimum_reference_count = max(3 if total >= 3 else total, int(np.ceil(total * 0.5)) if total >= 6 else 2 if total >= 2 else total)
    recommended_key = str(recommended_payload.get("strategy_key") or "")
    anchor_clip_id = str(recommended_payload.get("reference_clip_id") or "")
    anchor_row = next((row for row in per_camera_rows if str(row.get("clip_id") or "") == anchor_clip_id), None)
    anchor_trust_class = str(anchor_row.get("trust_class") or "") if anchor_row else "DERIVED_CLUSTER_TARGET"
    fragmented = bool(int(exposure_summary.get("outlier_count", 0) or 0) > max(1, total // 4))
    gating_reasons: List[str] = []
    if reference_eligible_count < minimum_reference_count:
        gating_reasons.append("Too few cameras are trustworthy enough to represent the set.")
    if anchor_row and anchor_trust_class in {"UNTRUSTED", "EXCLUDED"}:
        gating_reasons.append("The chosen anchor does not meet the trust requirements.")
    if excluded_count > max(1, total // 4):
        gating_reasons.append("Too many cameras fell outside the safe reference group.")
    if fragmented:
        gating_reasons.append("The exposure set is fragmented into a stable cluster plus outliers.")
    if average_trust_score < 0.5:
        gating_reasons.append("Overall measurement quality is weak across the set.")

    if reference_eligible_count < minimum_reference_count or (anchor_row and anchor_trust_class in {"UNTRUSTED", "EXCLUDED"}):
        run_status = "DO_NOT_PUSH"
        recommendation_strength = "LOW_CONFIDENCE"
    elif trusted_count < max(2, min(total, 3)) or fragmented or average_trust_score < 0.58:
        run_status = "REVIEW_REQUIRED"
        recommendation_strength = "LOW_CONFIDENCE"
    elif caution or untrusted or excluded or average_trust_score < 0.8:
        run_status = "READY_WITH_WARNINGS"
        recommendation_strength = "MEDIUM_CONFIDENCE"
    else:
        run_status = "READY"
        recommendation_strength = "HIGH_CONFIDENCE"

    if recommended_key == "median":
        anchor_summary = "Median used the center of the trusted camera group instead of a single-camera anchor."
    else:
        selected_strategy_summary = next(
            (item for item in strategy_summaries if str(item.get("strategy_key") or "") == recommended_key),
            {},
        )
        selected_strategy_diagnostics = dict((selected_strategy_summary or {}).get("selection_diagnostics") or {})
        anchor_summary = str(selected_strategy_diagnostics.get("anchor_reason") or "A single trustworthy camera anchored the group.")

    if run_status == "READY":
        operator_note = "This run is strong enough to trust later for push if the camera readback still matches."
    elif run_status == "READY_WITH_WARNINGS":
        operator_note = "This run is usable, but at least one camera should be reviewed before any later push."
    elif run_status == "REVIEW_REQUIRED":
        operator_note = "Review is required before trusting this run for later camera push."
    else:
        operator_note = "Do not push these corrections later without remeasuring the set."

    return {
        "status": run_status,
        "recommendation_strength": recommendation_strength,
        "safe_to_push_later": run_status in {"READY", "READY_WITH_WARNINGS"},
        "trusted_camera_count": trusted_count,
        "caution_camera_count": len(caution),
        "untrusted_camera_count": len(untrusted),
        "excluded_camera_count": excluded_count,
        "reference_eligible_count": reference_eligible_count,
        "camera_count": total,
        "average_trust_score": average_trust_score,
        "anchor_camera": anchor_clip_id or None,
        "anchor_trust_class": anchor_trust_class,
        "anchor_summary": anchor_summary,
        "gating_reasons": gating_reasons,
        "operator_note": operator_note,
    }


def _lightweight_summary_sentence(
    *,
    exposure_summary: Dict[str, object],
    recommended_strategy: Dict[str, object],
) -> str:
    outlier_count = int(exposure_summary.get("outlier_count", 0) or 0)
    strategy_label = str(recommended_strategy.get("strategy_label") or "Pending")
    if outlier_count <= 0:
        return f"Most cameras are consistent, and {strategy_label} keeps the array balanced."
    if outlier_count == 1:
        return "Most cameras are consistent. 1 camera is significantly out of family and should be reviewed separately."
    return f"Most cameras are consistent, but {outlier_count} cameras sit outside the stable cluster and need review."


def _lightweight_summary_points(
    *,
    exposure_summary: Dict[str, object],
    recommended_strategy: Dict[str, object],
    strategy_summaries: List[Dict[str, object]],
) -> List[str]:
    outlier_count = int(exposure_summary.get("outlier_count", 0) or 0)
    points = [
        f"Core cluster spread: {float(exposure_summary['spread']):.2f} across the measured gray-target brightness range.",
        (
            f"Outlier deviation: {outlier_count} camera requires separate review."
            if outlier_count == 1
            else f"Outlier deviation: {outlier_count} cameras require separate review."
            if outlier_count > 1
            else "Outlier deviation: no cameras currently sit outside the stable cluster."
        ),
        f"Chosen strategy: {str(recommended_strategy['strategy_label'])}.",
        str(recommended_strategy.get("reason") or ""),
    ]
    optimal_summary = next((item for item in strategy_summaries if str(item.get("strategy_key") or "") == "optimal_exposure"), None)
    if str(recommended_strategy.get("strategy_key") or "") == "median":
        optimal_reason = str(((optimal_summary or {}).get("selection_diagnostics") or {}).get("anchor_reason") or "")
        points.append(
            "Median was chosen as a fallback because no single camera was trustworthy enough to anchor the array."
            if "fell back" in optimal_reason.lower()
            else "Median was chosen because it keeps the group stable instead of hinging the solve on one camera."
        )
        if optimal_summary:
            points.append(
                f"Optimal Exposure checked {str(optimal_summary.get('reference_clip_id') or 'the closest gray-match camera')}, but it was rejected because {optimal_reason.lower() or 'it would have increased corrections across the group'}."
            )
    return [item for item in points if str(item).strip()]


def _build_lightweight_synopsis(
    *,
    exposure_summary: Dict[str, object],
    strategy_summaries: List[Dict[str, object]],
    recommended_strategy: Dict[str, object],
    hero_summary: Dict[str, object],
) -> str:
    summary_sentence = _lightweight_summary_sentence(
        exposure_summary=exposure_summary,
        recommended_strategy=recommended_strategy,
    )
    spread = float(exposure_summary["spread"])
    minimum = float(exposure_summary["minimum"])
    maximum = float(exposure_summary["maximum"])
    synopsis = [
        summary_sentence,
        f"Core cluster spread was {spread:.2f} in the derived gray-target brightness metric, from {minimum:.2f} to {maximum:.2f}.",
    ]
    if strategy_summaries:
        median_summary = next((item for item in strategy_summaries if item["strategy_key"] == "median"), None)
        optimal_summary = next((item for item in strategy_summaries if item["strategy_key"] == "optimal_exposure"), None)
        if median_summary:
            synopsis.append(
                f"The median strategy would center the array with an average exposure correction of {float(median_summary['correction_metrics']['mean_abs_offset']):.2f}."
            )
        if optimal_summary:
            optimal_diagnostics = dict(optimal_summary.get("selection_diagnostics") or {})
            screened_candidates = list(optimal_diagnostics.get("screened_candidates") or [])
            if str(optimal_diagnostics.get("fallback_mode")) == "median_fallback":
                synopsis.append(
                    f"Optimal Exposure could not find a trustworthy single-camera anchor, so it fell back to the most central camera {optimal_summary.get('reference_clip_id') or 'in the set'}."
                )
            else:
                synopsis.append(
                    f"Optimal Exposure would anchor to {optimal_summary.get('reference_clip_id') or 'the best gray-match camera'}, asking for at most {float(optimal_summary['correction_metrics']['max_abs_offset']):.2f} of exposure correction."
                )
            if screened_candidates:
                screened_names = ", ".join(str(item.get("clip_id") or "") for item in screened_candidates[:3] if str(item.get("clip_id") or "").strip())
                synopsis.append(
                    f"{len(screened_candidates)} camera(s) such as {screened_names} were screened out because they did not meet the stability and trust requirements for an anchor."
                )
            if recommended_strategy.get("strategy_key") != "optimal_exposure":
                synopsis.append(
                    f"A single camera was closest to correct gray, but using it as the anchor would require larger corrections across the group than the chosen {recommended_strategy['strategy_label']} strategy."
                )
    synopsis.append(recommended_strategy["reason"])
    if hero_summary.get("candidate_clip_id"):
        synopsis.append(f"{hero_summary['candidate_clip_id']} appears to be the strongest hero candidate. {hero_summary['reason']}")
    else:
        synopsis.append(hero_summary["reason"])
    return " ".join(synopsis)


def _camera_id_from_clip_id(clip_id: str) -> str:
    parts = str(clip_id).split("_")
    return "_".join(parts[:2]) if len(parts) >= 2 else str(clip_id)


def _roi_overlay_svg(
    *,
    clip_id: str,
    calibration_roi: Optional[Dict[str, float]],
    sample_sets: List[Dict[str, object]],
) -> str:
    width = 960
    height = 540
    roi = dict(calibration_roi or {"x": 0.3, "y": 0.3, "w": 0.4, "h": 0.4})
    roi_x = float(roi.get("x", 0.0) or 0.0) * width
    roi_y = float(roi.get("y", 0.0) or 0.0) * height
    roi_w = float(roi.get("w", 1.0) or 1.0) * width
    roi_h = float(roi.get("h", 1.0) or 1.0) * height
    sample_boxes: List[str] = []
    colors = {"bright_side": "#f59e0b", "center": "#22c55e", "dark_side": "#3b82f6", "left": "#22c55e", "right": "#ef4444"}
    for sample in sample_sets:
        bounds = dict((sample.get("bounds") or {}).get("normalized_within_roi") or {})
        sample_x = roi_x + float(bounds.get("x", 0.0) or 0.0) * roi_w
        sample_y = roi_y + float(bounds.get("y", 0.0) or 0.0) * roi_h
        sample_w = float(bounds.get("w", 0.0) or 0.0) * roi_w
        sample_h = float(bounds.get("h", 0.0) or 0.0) * roi_h
        label = str(sample.get("label") or "")
        stroke = colors.get(label, "#38bdf8")
        sample_boxes.append(
            f"<rect x=\"{sample_x:.1f}\" y=\"{sample_y:.1f}\" width=\"{sample_w:.1f}\" height=\"{sample_h:.1f}\" "
            f"fill=\"none\" stroke=\"{stroke}\" stroke-width=\"4\" rx=\"10\"/>"
            f"<text x=\"{sample_x + 8:.1f}\" y=\"{sample_y + 22:.1f}\" font-size=\"16\" fill=\"{stroke}\">{html.escape(SPHERE_PROFILE_ZONE_DISPLAY.get(label, label.title()))}</text>"
        )
    return (
        f"<svg viewBox=\"0 0 {width} {height}\" role=\"img\" aria-label=\"ROI overlay for {html.escape(str(clip_id))}\">"
        f"<rect x=\"0\" y=\"0\" width=\"{width}\" height=\"{height}\" fill=\"#0f172a\"/>"
        f"<rect x=\"18\" y=\"18\" width=\"{width - 36}\" height=\"{height - 36}\" fill=\"#111827\" stroke=\"#334155\" stroke-width=\"2\" rx=\"20\"/>"
        f"<text x=\"28\" y=\"42\" font-size=\"20\" fill=\"#e2e8f0\">{html.escape(str(clip_id))}</text>"
        f"<text x=\"28\" y=\"68\" font-size=\"14\" fill=\"#94a3b8\">Shared ROI and gradient-aligned sphere bands</text>"
        f"<rect x=\"{roi_x:.1f}\" y=\"{roi_y:.1f}\" width=\"{roi_w:.1f}\" height=\"{roi_h:.1f}\" fill=\"#38bdf8\" fill-opacity=\"0.08\" stroke=\"#38bdf8\" stroke-width=\"5\" rx=\"16\"/>"
        f"<text x=\"{roi_x + 10:.1f}\" y=\"{max(roi_y - 12, 22):.1f}\" font-size=\"16\" fill=\"#38bdf8\">Calibration ROI</text>"
        f"{''.join(sample_boxes)}"
        "</svg>"
    )


def _sphere_detection_note(detection_source: str, detection_label: str, *, detection_failed: bool = False) -> str:
    if detection_failed or str(detection_source or "") == "failed" or str(detection_label or "") == "FAILED":
        return "Sphere detection: review needed"
    if str(detection_source or "") == "forced_best_effort":
        return "Sphere detection: low-confidence recovery"
    if str(detection_source or "") in {"secondary_detected", "localized_recovery", "reused_from_original"} or str(detection_label or "") == "LOW":
        return "Sphere detection: fallback used"
    return "Sphere detection: verified"


def _measurement_region_origin(image_size: Tuple[int, int], calibration_roi: Optional[Dict[str, float]]) -> Tuple[int, int]:
    width, height = image_size
    if calibration_roi is None:
        region_width = int(round(width * 0.4))
        region_height = int(round(height * 0.4))
        return (max(0, (width - region_width) // 2), max(0, (height - region_height) // 2))
    x0 = max(0, min(width - 1, int(np.floor(float(calibration_roi["x"]) * width))))
    y0 = max(0, min(height - 1, int(np.floor(float(calibration_roi["y"]) * height))))
    return (x0, y0)


def _draw_detection_overlay(
    *,
    image_path: str,
    output_path: Path,
    clip_id: str,
    calibration_roi: Optional[Dict[str, float]],
    detected_sphere_roi: Optional[Dict[str, float]],
    zone_measurements: List[Dict[str, object]],
    detection_source: str,
    detection_label: str,
    detection_confidence: float,
) -> None:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size
    origin_x, origin_y = _measurement_region_origin((width, height), calibration_roi)
    if calibration_roi is None:
        region_width = int(round(width * 0.4))
        region_height = int(round(height * 0.4))
    else:
        region_width = max(1, int(np.ceil(float(calibration_roi["w"]) * width)))
        region_height = max(1, int(np.ceil(float(calibration_roi["h"]) * height)))
    draw.rectangle(
        (origin_x, origin_y, origin_x + region_width, origin_y + region_height),
        outline="#facc15",
        width=4,
    )
    if detected_sphere_roi:
        cx = origin_x + float(detected_sphere_roi.get("cx", 0.0) or 0.0)
        cy = origin_y + float(detected_sphere_roi.get("cy", 0.0) or 0.0)
        radius = float(detected_sphere_roi.get("r", 0.0) or 0.0)
        draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), outline="#22d3ee", width=5)
        draw.ellipse((cx - 4, cy - 4, cx + 4, cy + 4), fill="#22d3ee")
    colors = {"bright_side": "#f59e0b", "center": "#22c55e", "dark_side": "#3b82f6", "upper_mid": "#f59e0b", "lower_mid": "#3b82f6"}
    for zone in list(zone_measurements or []):
        bounds = dict((zone.get("bounds") or {}).get("pixel") or {})
        polygon = list(((zone.get("bounds") or {}).get("polygon") or {}).get("pixel") or [])
        if not bounds and not polygon:
            continue
        x0 = origin_x + int(bounds.get("x0", 0) or 0)
        y0 = origin_y + int(bounds.get("y0", 0) or 0)
        x1 = origin_x + int(bounds.get("x1", 0) or 0)
        y1 = origin_y + int(bounds.get("y1", 0) or 0)
        stroke = colors.get(str(zone.get("label") or ""), "#f8fafc")
        if polygon:
            polygon_points = [
                (origin_x + float(point.get("x", 0.0) or 0.0), origin_y + float(point.get("y", 0.0) or 0.0))
                for point in polygon
            ]
            draw.line(polygon_points + [polygon_points[0]], fill=stroke, width=4)
        else:
            draw.rectangle((x0, y0, x1, y1), outline=stroke, width=4)
        label_x = x0 if bounds else int(min(point[0] for point in polygon_points))
        label_y = y0 if bounds else int(min(point[1] for point in polygon_points))
        draw.text((label_x + 4, max(4, label_y - 16)), f"{zone.get('display_label', zone.get('label', 'Zone'))} {float(zone.get('measured_ire', 0.0) or 0.0):.1f}", fill=stroke)
    draw.rectangle((16, 16, min(width - 16, 840), 140), fill=(15, 23, 42))
    draw.text((28, 28), str(clip_id), fill="#f8fafc")
    draw.text((28, 52), f"Detection: {detection_source}", fill="#cbd5e1")
    draw.text((28, 76), f"Confidence: {detection_label} ({float(detection_confidence):.2f})", fill="#cbd5e1")
    draw.text((28, 100), _sphere_detection_note(detection_source, detection_label, detection_failed=(detection_label == 'FAILED')), fill="#e2e8f0")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def _build_sphere_detection_artifacts(
    *,
    out_root: Path,
    validation_summary: Dict[str, object],
    recommended_strategy_key: str,
) -> Dict[str, object]:
    overlay_root = out_root / "review_detection_overlays"
    overlay_root.mkdir(parents=True, exist_ok=True)
    rows = [
        dict(item)
        for item in list(validation_summary.get("rows") or [])
        if str(item.get("strategy_key") or "") == str(recommended_strategy_key)
    ]
    summary_rows: List[Dict[str, object]] = []
    for row in rows:
        clip_id = str(row.get("clip_id") or "")
        calibration_roi = row.get("calibration_roi")
        original_overlay_path = overlay_root / f"{clip_id}.original_detection.png"
        corrected_overlay_path = overlay_root / f"{clip_id}.corrected_detection.png"
        original_detection_source = str(row.get("ipp2_original_detection_source") or "failed")
        original_detection_label = str(row.get("ipp2_original_detection_label") or "FAILED")
        original_detection_confidence = float(row.get("ipp2_original_detection_confidence", 0.0) or 0.0)
        original_detected_roi = dict(row.get("ipp2_original_detected_sphere_roi") or {})
        corrected_detection_source = str(row.get("ipp2_detection_source") or "failed")
        corrected_detection_label = str(row.get("ipp2_detection_label") or "FAILED")
        corrected_detection_confidence = float(row.get("ipp2_detection_confidence", 0.0) or 0.0)
        corrected_detected_roi = dict(row.get("ipp2_detected_sphere_roi") or {})
        if str(row.get("original_image_path") or "").strip():
            _draw_detection_overlay(
                image_path=str(row.get("original_image_path") or ""),
                output_path=original_overlay_path,
                clip_id=clip_id,
                calibration_roi=calibration_roi,
                detected_sphere_roi=original_detected_roi,
                zone_measurements=list(row.get("ipp2_original_zone_profile") or []),
                detection_source=original_detection_source,
                detection_label=original_detection_label,
                detection_confidence=original_detection_confidence,
            )
        if str(row.get("corrected_image_path") or "").strip():
            _draw_detection_overlay(
                image_path=str(row.get("corrected_image_path") or ""),
                output_path=corrected_overlay_path,
                clip_id=clip_id,
                calibration_roi=calibration_roi,
                detected_sphere_roi=corrected_detected_roi or original_detected_roi,
                zone_measurements=list(row.get("ipp2_zone_profile") or []),
                detection_source=corrected_detection_source,
                detection_label=corrected_detection_label,
                detection_confidence=corrected_detection_confidence,
            )
        summary_rows.append(
            {
                "clip_id": clip_id,
                "camera_id": str(row.get("camera_id") or ""),
                "original_detection_source": original_detection_source,
                "original_detection_confidence": original_detection_confidence,
                "original_detection_label": original_detection_label,
                "original_detected_sphere_roi": original_detected_roi,
                "corrected_detection_source": corrected_detection_source,
                "corrected_detection_confidence": corrected_detection_confidence,
                "corrected_detection_label": corrected_detection_label,
                "corrected_detected_sphere_roi": corrected_detected_roi or original_detected_roi,
                "corrected_frame_reused_original_roi": corrected_detection_source == "reused_from_original",
                "fallback_used": original_detection_source in {"secondary_detected", "localized_recovery", "forced_best_effort"}
                or corrected_detection_source in {"secondary_detected", "localized_recovery", "forced_best_effort", "reused_from_original"},
                "detection_failed": bool(row.get("detection_failed")),
                "original_overlay_path": str(original_overlay_path),
                "corrected_overlay_path": str(corrected_overlay_path),
            }
        )
    confidence_counts = {
        label: sum(1 for row in summary_rows if str(row.get("original_detection_label") or "") == label)
        for label in ("HIGH", "MEDIUM", "LOW", "FAILED")
    }
    fallback_count = sum(1 for row in summary_rows if bool(row.get("fallback_used")))
    failed_count = sum(1 for row in summary_rows if bool(row.get("detection_failed")) or str(row.get("original_detection_label") or "") == "FAILED")
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "recommended_strategy_key": recommended_strategy_key,
        "camera_count": len(summary_rows),
        "confidence_counts": confidence_counts,
        "fallback_count": fallback_count,
        "failed_count": failed_count,
        "rows": summary_rows,
    }
    summary_path = out_root / "review_detection_summary.json"
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {
        "path": str(summary_path),
        "overlay_root": str(overlay_root),
        "summary": payload,
    }


def _build_exposure_trace_artifacts(
    *,
    analysis_records: List[Dict[str, object]],
    recommended_payload: Dict[str, object],
    strategy_summaries: List[Dict[str, object]],
    exposure_summary: Dict[str, object],
    out_root: Path,
) -> Dict[str, object]:
    debug_root = out_root / "debug_exposure_trace"
    overlay_root = debug_root / "roi_overlays"
    debug_root.mkdir(parents=True, exist_ok=True)
    overlay_root.mkdir(parents=True, exist_ok=True)
    recommended_by_clip = {str(item.get("clip_id") or ""): dict(item) for item in (recommended_payload.get("clips") or [])}
    optimal_summary = next(
        (item for item in strategy_summaries if str(item.get("strategy_key") or "") == "optimal_exposure"),
        None,
    )
    optimal_diag = dict((optimal_summary or {}).get("selection_diagnostics") or {})
    trusted_candidates = {
        str(item.get("clip_id") or ""): dict(item)
        for item in (optimal_diag.get("scored_candidates") or [])
        if str(item.get("clip_id") or "").strip()
    }
    screened_candidates = {
        str(item.get("clip_id") or ""): dict(item)
        for item in (optimal_diag.get("screened_candidates") or [])
        if str(item.get("clip_id") or "").strip()
    }

    rows: List[Dict[str, object]] = []
    for record in analysis_records:
        clip_id = str(record.get("clip_id") or "")
        strategy_clip = recommended_by_clip.get(clip_id) or {}
        diagnostics = dict(record.get("diagnostics") or {})
        confidence = float(strategy_clip.get("confidence", record.get("confidence", 0.0)) or 0.0)
        sample_spread = float(strategy_clip.get("neutral_sample_log2_spread", diagnostics.get("neutral_sample_log2_spread", 0.0)) or 0.0)
        sample_chroma_spread = float(
            strategy_clip.get("neutral_sample_chromaticity_spread", diagnostics.get("neutral_sample_chromaticity_spread", 0.0)) or 0.0
        )
        measured_monitoring = float(strategy_clip.get("measured_log2_luminance_monitoring", diagnostics.get("measured_log2_luminance_monitoring", diagnostics.get("measured_log2_luminance", 0.0))) or 0.0)
        measured_raw = float(strategy_clip.get("measured_log2_luminance_raw", diagnostics.get("measured_log2_luminance_raw", measured_monitoring)) or 0.0)
        final_offset = float(strategy_clip.get("exposure_offset_stops", 0.0) or 0.0)
        target_log2 = float(recommended_payload.get("target_log2_luminance", 0.0) or 0.0)
        trusted_row = trusted_candidates.get(clip_id)
        screened_row = screened_candidates.get(clip_id)
        should_be_trusted = (
            confidence >= OPTIMAL_EXPOSURE_MIN_CONFIDENCE
            and sample_spread <= OPTIMAL_EXPOSURE_MAX_SAMPLE_LOG2_SPREAD
            and sample_chroma_spread <= OPTIMAL_EXPOSURE_MAX_SAMPLE_CHROMA_SPREAD
            and abs(measured_monitoring - float(exposure_summary.get("median", 0.0) or 0.0)) <= float(exposure_summary.get("outlier_threshold", 0.35) or 0.35)
        )
        trust_status = "trusted_candidate" if trusted_row else "screened_out" if screened_row else "not_evaluated"
        directional_bias = "within_cluster"
        if final_offset >= float(exposure_summary.get("outlier_threshold", 0.35) or 0.35):
            directional_bias = "darker_than_cluster"
        elif final_offset <= -float(exposure_summary.get("outlier_threshold", 0.35) or 0.35):
            directional_bias = "brighter_than_cluster"
        overlay_path = overlay_root / f"{_camera_id_from_clip_id(clip_id)}.svg"
        overlay_path.write_text(
            _roi_overlay_svg(
                clip_id=clip_id,
                calibration_roi=diagnostics.get("calibration_roi"),
                sample_sets=[dict(item) for item in (diagnostics.get("neutral_samples") or [])],
            ),
            encoding="utf-8",
        )
        row = {
            "camera_id": _camera_id_from_clip_id(clip_id),
            "camera_label": _camera_label_for_reporting(clip_id),
            "clip_id": clip_id,
            "source_path": str(record.get("source_path") or ""),
            "measurement_mode": str(diagnostics.get("calibration_measurement_mode") or ""),
            "calibration_roi": dict(diagnostics.get("calibration_roi") or {}),
            "gray_exposure_summary": str(diagnostics.get("gray_exposure_summary") or diagnostics.get("aggregate_sphere_profile") or "n/a"),
            "top_ire": float(diagnostics.get("top_ire", 0.0) or 0.0),
            "mid_ire": float(diagnostics.get("mid_ire", 0.0) or 0.0),
            "bottom_ire": float(diagnostics.get("bottom_ire", 0.0) or 0.0),
            "final_gray_value_used": {
                "monitoring_log2": measured_monitoring,
                "raw_log2": measured_raw,
                "target_log2": target_log2,
            },
            "correction": {
                "exposure_adjust": final_offset,
                "deviation_from_anchor": round(measured_monitoring - target_log2, 6),
                "directional_bias": directional_bias,
            },
            "quality": {
                "confidence": confidence,
                "flags": [str(item) for item in (strategy_clip.get("flags") or []) if str(item).strip()],
                "neutral_sample_log2_spread": sample_spread,
                "neutral_sample_chromaticity_spread": sample_chroma_spread,
            },
            "trust": {
                "status": trust_status,
                "trusted_for_anchor_selection": bool(trusted_row),
                "should_have_been_trusted": bool(should_be_trusted),
                "screening_reasons": list((screened_row or {}).get("reasons") or []),
            },
            "sampling": {
                "sampled_frames": int(diagnostics.get("sampled_frames", 0) or 0),
                "accepted_frames": int(diagnostics.get("accepted_frames", 0) or 0),
                "rejected_frame_count": max(
                    int(diagnostics.get("sampled_frames", 0) or 0) - int(diagnostics.get("accepted_frames", 0) or 0),
                    0,
                ),
                "neutral_samples_monitoring": [dict(item) for item in (diagnostics.get("neutral_samples") or [])],
                "neutral_samples_raw": [dict(item) for item in (diagnostics.get("neutral_samples_raw") or [])],
                "roi_overlay_path": str(overlay_path),
            },
            "failure_mode_flags": {
                "high_variance_gray_samples": sample_spread > OPTIMAL_EXPOSURE_MAX_SAMPLE_LOG2_SPREAD,
                "low_confidence_used": confidence < OPTIMAL_EXPOSURE_MIN_CONFIDENCE and abs(final_offset) > 0.0,
                "anchor_chosen_from_unstable_camera": bool(clip_id == optimal_diag.get("anchor_clip_id") and screened_row),
                "directional_bias": directional_bias,
            },
            "strategy_inclusion": {
                "recommended_strategy": str(recommended_payload.get("strategy_key") or ""),
                "optimal_exposure_reference": str((optimal_summary or {}).get("reference_clip_id") or ""),
            },
        }
        rows.append(row)
        (debug_root / f"{row['camera_id']}.json").write_text(json.dumps(row, indent=2), encoding="utf-8")

    brightness_order = sorted(rows, key=lambda item: float(item["final_gray_value_used"]["monitoring_log2"]), reverse=True)
    confidence_order = sorted(rows, key=lambda item: float(item["quality"]["confidence"]), reverse=True)
    stability_order = sorted(
        rows,
        key=lambda item: (
            float(item["quality"]["neutral_sample_log2_spread"]),
            float(item["quality"]["neutral_sample_chromaticity_spread"]),
        ),
    )
    for index, row in enumerate(brightness_order, start=1):
        row.setdefault("rankings", {})["brightness_rank"] = index
    for index, row in enumerate(confidence_order, start=1):
        row.setdefault("rankings", {})["confidence_rank"] = index
    for index, row in enumerate(stability_order, start=1):
        row.setdefault("rankings", {})["stability_rank"] = index
    for row in rows:
        (debug_root / f"{row['camera_id']}.json").write_text(json.dumps(row, indent=2), encoding="utf-8")

    summary = {
        "schema_version": "r3dmatch_exposure_debug_trace_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "recommended_strategy": str(recommended_payload.get("strategy_key") or ""),
        "anchor_clip_id": str(recommended_payload.get("reference_clip_id") or ""),
        "target_gray_log2": float(recommended_payload.get("target_log2_luminance", 0.0) or 0.0),
        "exposure_summary": dict(exposure_summary),
        "optimal_exposure_diagnostics": optimal_diag,
        "roi_alignment": {
            "measurement_modes": sorted({str(item.get("measurement_mode") or "") for item in rows}),
            "shared_roi": len({json.dumps(item.get("calibration_roi") or {}, sort_keys=True) for item in rows}) <= 1,
            "shared_sample_geometry": len(
                {
                    json.dumps(
                        [sample.get("bounds") for sample in (item.get("sampling") or {}).get("neutral_samples_monitoring", [])],
                        sort_keys=True,
                    )
                    for item in rows
                }
            )
            <= 1,
        },
        "cameras": sorted(rows, key=lambda item: str(item["camera_id"])),
    }
    summary_path = debug_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {
        "directory": str(debug_root),
        "summary_path": str(summary_path),
        "camera_count": len(rows),
    }


def _render_lightweight_analysis_html(payload: Dict[str, object]) -> str:
    exposure_summary = payload["exposure_summary"]
    hero_summary = payload["hero_recommendation"]
    run_assessment = dict(payload.get("run_assessment") or {})
    anchor_summary = str(payload.get("exposure_anchor_summary") or "Exposure Anchor: Automatic recommendation")
    per_camera_rows = [dict(item) for item in list(payload.get("per_camera_analysis") or [])]
    retained_rows = [row for row in per_camera_rows if str(row.get("reference_use") or "").strip().lower() != "excluded"]

    retained_sample_2_values = [
        float(row.get("sample_2_ire"))
        for row in retained_rows
        if row.get("sample_2_ire") is not None
    ]
    best_reference = None
    best_reference_profile_text = ""
    if retained_rows:
        best_reference = min(
            retained_rows,
            key=lambda row: (
                abs(float(row.get("camera_offset_from_anchor", row.get("final_offset_stops", 0.0)) or 0.0)),
                -float(row.get("trust_score", 0.0) or 0.0),
                -float(row.get("confidence", 0.0) or 0.0),
                str(row.get("camera_label") or row.get("clip_id") or ""),
            ),
        )
        best_reference_profile_text = f"({str(best_reference.get('measured_gray_exposure_summary') or 'n/a')})"
    run_state = str(run_assessment.get("status") or "")
    readiness_label = {
        "READY": "ARRAY WITHIN CALIBRATION TOLERANCE",
        "READY_WITH_WARNINGS": "ARRAY NEEDS CALIBRATION REVIEW",
        "REVIEW_REQUIRED": "ARRAY OUT OF CALIBRATION",
        "DO_NOT_PUSH": "ARRAY OUT OF CALIBRATION",
    }.get(run_state, "ARRAY NEEDS CALIBRATION REVIEW")
    readiness_tone = "success" if str(run_assessment.get("status") or "") == "READY" else "danger" if str(run_assessment.get("status") or "") == "DO_NOT_PUSH" else "warning"
    summary_sentence = _lightweight_summary_sentence(
        exposure_summary=exposure_summary,
        recommended_strategy=payload["recommended_strategy"],
    )
    summary_points = _lightweight_summary_points(
        exposure_summary=exposure_summary,
        recommended_strategy=payload["recommended_strategy"],
        strategy_summaries=payload["strategy_comparison"],
    )
    banner_subline = (
        f"Retained gray sphere Sample 2 values span {min(retained_sample_2_values):.0f} IRE to {max(retained_sample_2_values):.0f} IRE."
        if retained_sample_2_values
        else "Retained gray sphere Sample 2 range is not available yet."
    )
    strategy_cards = []
    for strategy in payload["strategy_comparison"]:
        recommended_badge = "<span class='recommended-badge'>Recommended</span>" if strategy.get("recommended") else ""
        selection_diagnostics = dict(strategy.get("selection_diagnostics") or {})
        strategy_role = "Stability anchor" if str(strategy.get("strategy_key") or "") == "median" else "Accuracy to gray" if str(strategy.get("strategy_key") or "") == "optimal_exposure" else "Reference anchor"
        strategy_cards.append(
            "<article class='strategy-card'>"
            f"<div class='strategy-card-top'><h3>{html.escape(str(strategy['strategy_label']))}</h3>{recommended_badge}</div>"
            f"<p>{html.escape(str(strategy['summary']))}</p>"
            f"<dl class='metric-pairs'><div><dt>Gray Target</dt><dd>{float(strategy['target_log2_luminance']):.2f} derived brightness</dd></div>"
            f"<div><dt>Mean | Max Correction</dt><dd>{float(strategy['correction_metrics']['mean_abs_offset']):.2f} | {float(strategy['correction_metrics']['max_abs_offset']):.2f}</dd></div>"
            f"<div><dt>Confidence</dt><dd>{float(strategy['correction_metrics'].get('mean_confidence', 0.0)):.2f}</dd></div>"
            f"<div><dt>Neutral Residual</dt><dd>{float(strategy['correction_metrics'].get('mean_color_residual', 0.0)):.4f}</dd></div>"
            f"<div><dt>WB Model</dt><dd>{html.escape(str((strategy.get('white_balance_model') or {}).get('model_label') or 'n/a'))}</dd></div>"
            f"<div><dt>Shared Kelvin</dt><dd>{html.escape(str((strategy.get('white_balance_model') or {}).get('shared_kelvin') or 'per-camera'))}</dd></div>"
            f"<div><dt>Reference</dt><dd>{html.escape(str(strategy.get('reference_clip_id') or 'Derived'))}</dd></div>"
            "</dl>"
            f"<p class='subtle' style='margin-top: 10px;'><strong>Role:</strong> {html.escape(strategy_role)}</p>"
            f"<p class='subtle' style='margin-top: 10px;'><strong>Anchor guard:</strong> {html.escape(str(selection_diagnostics.get('anchor_reason') or 'Derived target.'))}</p>"
            "</article>"
        )
    def severity_key(row: Dict[str, object]) -> tuple[int, float, float]:
        trust_order = {"EXCLUDED": 0, "UNTRUSTED": 1, "USE_WITH_CAUTION": 2, "TRUSTED": 3}
        return (
            trust_order.get(str(row.get("trust_class") or ""), 4),
            float(row.get("trust_score", 0.0) or 0.0),
            -abs(float(row.get("final_offset_stops", 0.0) or 0.0)),
        )

    table_rows = []
    for row in sorted(payload["per_camera_analysis"], key=severity_key):
        offset_value = float(row.get("camera_offset_from_anchor", row.get("final_offset_stops", 0.0)) or 0.0)
        if abs(offset_value) <= IPP2_VALIDATION_PASS_STOPS:
            result_label = "In Range"
            status_class = "good-pill"
            validation_status = "PASS"
        elif abs(offset_value) <= IPP2_VALIDATION_REVIEW_STOPS:
            result_label = "Needs adjustment"
            status_class = "warning-pill"
            validation_status = "REVIEW"
        else:
            result_label = "Outside exposure tolerance"
            status_class = "outlier-pill"
            validation_status = "FAIL"
        action = _operator_guidance_for_correction(
            correction_stops=offset_value,
            residual_stops=offset_value,
            validation_status=validation_status,
        )
        profile_note = str(row.get("trust_reason") or "Profile consistent")
        table_rows.append(
            "<tr>"
            f"<td><div class='camera-cell'><strong>{html.escape(str(row['camera_label']))}</strong><span>{html.escape(str(row['clip_id']))}</span></div></td>"
            f"<td>{html.escape(str(row.get('measured_gray_exposure_summary') or 'n/a'))}</td>"
            f"<td>{offset_value:+.2f} stops</td>"
            f"<td><span class='{status_class}'>{html.escape(result_label)}</span><div class='subtle'>{html.escape(profile_note)}</div></td>"
            f"<td><strong>{html.escape(str(action.get('suggested_action') or 'No adjustment needed'))}</strong><div class='subtle'>{html.escape(str(action.get('notes') or ''))}</div></td>"
            "</tr>"
        )
    color_preview_line = (
        f"Color preview: {html.escape(str(payload.get('color_preview_status')))} | {html.escape(str(payload.get('color_preview_note')))}"
        if payload.get("color_preview_note")
        else f"Color preview: {html.escape(str(payload.get('color_preview_status') or 'unknown'))}"
    )
    def chart_frame(title: str, svg: str) -> str:
        return (
            f"<div class='chart-launch' role='button' tabindex='0' data-chart-title='{html.escape(title)}'>"
            f"<div class='chart-frame'>{svg}</div>"
            "<span class='chart-hint'>Click to enlarge</span>"
            "</div>"
        )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>R3DMatch Lightweight Analysis</title>
  <style>
    body {{ margin: 0; background: #edf2f7; color: #0f172a; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    .page {{ max-width: 1320px; margin: 0 auto; padding: 32px; }}
    .hero {{ background: linear-gradient(135deg, rgba(255,255,255,0.98) 0%, rgba(241,245,249,0.98) 100%); border: 1px solid #dbe4ef; border-radius: 24px; padding: 28px; box-shadow: 0 24px 60px rgba(15,23,42,0.08); }}
    .decision-banner {{ margin-top: 22px; border-radius: 22px; padding: 24px; color: white; box-shadow: 0 18px 40px rgba(15,23,42,0.16); }}
    .decision-banner.success {{ background: linear-gradient(135deg, #14532d 0%, #166534 100%); }}
    .decision-banner.warning {{ background: linear-gradient(135deg, #92400e 0%, #b45309 100%); }}
    .decision-banner.danger {{ background: linear-gradient(135deg, #7f1d1d 0%, #b91c1c 100%); }}
    .decision-banner-kicker {{ font-size: 12px; font-weight: 800; letter-spacing: 0.12em; text-transform: uppercase; opacity: 0.88; }}
    .decision-banner-title {{ margin: 8px 0 6px 0; font-size: 38px; line-height: 1.02; font-weight: 900; letter-spacing: -0.03em; }}
    .decision-banner-copy {{ margin: 0; font-size: 17px; line-height: 1.7; color: rgba(255,255,255,0.92); max-width: 72ch; }}
    .decision-metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; margin-top: 16px; }}
    .decision-metrics > div {{ padding: 12px 14px; border-radius: 14px; background: rgba(255,255,255,0.12); border: 1px solid rgba(255,255,255,0.18); }}
    .hero-top {{ display: flex; align-items: flex-start; gap: 18px; }}
    .hero-top img {{ width: 150px; max-width: 30vw; object-fit: contain; }}
    h1 {{ font-size: 38px; line-height: 1.08; margin: 0; }}
    .eyebrow {{ font-size: 12px; letter-spacing: 0.12em; text-transform: uppercase; color: #475569; font-weight: 700; }}
    .synopsis {{ margin-top: 18px; font-size: 19px; line-height: 1.8; color: #1e293b; max-width: 82ch; }}
    .meta-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; margin-top: 20px; }}
    .meta-card, .section {{ background: rgba(255,255,255,0.96); border: 1px solid #d9e1ec; border-radius: 20px; box-shadow: 0 16px 36px rgba(15,23,42,0.05); }}
    .meta-card {{ padding: 14px 16px; }}
    .meta-card dt {{ font-size: 11px; text-transform: uppercase; letter-spacing: 0.08em; color: #64748b; font-weight: 700; }}
    .meta-card dd {{ margin: 8px 0 0 0; font-size: 16px; font-weight: 700; }}
    .grid {{ display: grid; gap: 18px; margin-top: 18px; }}
    .two-up {{ grid-template-columns: 1.3fr 1fr; }}
    .section {{ padding: 24px; }}
    .section h2 {{ margin: 0 0 10px 0; font-size: 26px; }}
    .section p.lead {{ margin: 0 0 12px 0; color: #475569; line-height: 1.75; font-size: 17px; }}
    .stats {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; }}
    .stat {{ padding: 14px; border-radius: 16px; background: #f8fafc; border: 1px solid #e2e8f0; }}
    .stat .label {{ font-size: 11px; text-transform: uppercase; letter-spacing: 0.08em; color: #64748b; font-weight: 700; }}
    .stat .value {{ margin-top: 8px; font-size: 24px; font-weight: 800; }}
    .strategy-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 14px; }}
    .strategy-card {{ padding: 16px; border-radius: 18px; background: #f8fafc; border: 1px solid #dbe4ef; }}
    .strategy-card-top {{ display: flex; justify-content: space-between; align-items: center; gap: 12px; }}
    .strategy-card h3 {{ margin: 0; font-size: 20px; }}
    .recommended-badge, .hero-pill, .outlier-pill, .warning-pill, .good-pill {{ display: inline-flex; align-items: center; border-radius: 999px; padding: 4px 8px; font-size: 11px; font-weight: 800; letter-spacing: 0.06em; text-transform: uppercase; }}
    .recommended-badge {{ background: #dbeafe; color: #1d4ed8; }}
    .hero-pill {{ background: #dcfce7; color: #166534; margin-left: 6px; }}
    .outlier-pill {{ background: #fee2e2; color: #991b1b; margin-right: 6px; }}
    .warning-pill {{ background: #fef3c7; color: #92400e; margin-right: 6px; }}
    .good-pill {{ background: #dcfce7; color: #166534; margin-right: 6px; }}
    .metric-pairs {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; margin-top: 12px; }}
    .metric-pairs dt {{ font-size: 11px; text-transform: uppercase; color: #64748b; font-weight: 700; }}
    .metric-pairs dd {{ margin: 6px 0 0 0; font-size: 15px; font-weight: 700; }}
    .chart-frame {{ padding: 22px; border-radius: 18px; background: #f8fafc; border: 1px solid #dbe4ef; overflow: auto; }}
    .chart-frame svg {{ display: block; width: 100%; height: auto; min-height: 560px; }}
    .chart-launch {{ display: block; width: 100%; border: 0; padding: 0; background: transparent; text-align: left; cursor: zoom-in; }}
    .chart-launch:hover .chart-frame {{ border-color: #94a3b8; box-shadow: 0 10px 20px rgba(15,23,42,0.08); }}
    .chart-hint {{ display: inline-flex; margin-top: 10px; font-size: 13px; font-weight: 700; color: #475569; }}
    .chart-modal[hidden] {{ display: none; }}
    .chart-modal {{ position: fixed; inset: 0; background: rgba(15, 23, 42, 0.7); z-index: 999; display: flex; align-items: center; justify-content: center; padding: 16px; }}
    .chart-modal-card {{ width: min(1440px, 98vw); max-height: 96vh; overflow: auto; background: white; border-radius: 20px; padding: 24px; box-shadow: 0 24px 60px rgba(15,23,42,0.28); }}
    .chart-modal-top {{ display: flex; align-items: center; justify-content: space-between; gap: 12px; margin-bottom: 12px; }}
    .chart-modal-top h3 {{ margin: 0; font-size: 20px; }}
    .chart-modal-close {{ border: 0; border-radius: 10px; background: #0f172a; color: white; padding: 8px 12px; cursor: pointer; }}
    .chart-modal-body .chart-frame svg {{ min-height: 980px; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 14px; }}
    th, td {{ padding: 14px 12px; border-bottom: 1px solid #e2e8f0; vertical-align: top; font-size: 15px; }}
    th {{ text-align: left; font-size: 12px; letter-spacing: 0.08em; text-transform: uppercase; color: #64748b; }}
    .camera-cell {{ display: flex; flex-direction: column; gap: 4px; }}
    .camera-cell span {{ color: #64748b; font-size: 12px; }}
    .recommendation {{ font-size: 20px; line-height: 1.8; }}
    .subtle {{ color: #64748b; font-size: 14px; line-height: 1.7; }}
    .bullet-list {{ margin: 0; padding-left: 20px; color: #334155; font-size: 15px; line-height: 1.8; }}
    @media (max-width: 920px) {{ .two-up {{ grid-template-columns: 1fr; }} .stats {{ grid-template-columns: repeat(2, minmax(0,1fr)); }} }}
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <div class="hero-top">
        {f'<img src="{LOGO_PATH}" alt="R3DMatch logo">' if LOGO_PATH.exists() else ''}
        <div>
          <div class="eyebrow">Lightweight Analysis</div>
          <h1>R3DMatch Diagnostic Review</h1>
          <p class="subtle">Run label: {html.escape(str(payload['run_label']))} | Review mode: {html.escape(str(payload['review_mode_label']))}</p>
        </div>
      </div>
      <p class="synopsis">{html.escape(summary_sentence)}</p>
      <div class="decision-banner {readiness_tone}">
        <div class="decision-banner-kicker">Array Calibration Review</div>
        <div class="decision-banner-title">{html.escape(readiness_label)}</div>
        <p class="decision-banner-copy">{html.escape(str(payload['operator_recommendation']))}</p>
        <p class="decision-banner-copy" style="margin-top:8px;font-size:15px;font-weight:700;">{html.escape(banner_subline)}</p>
        <div class="decision-metrics">
          <div><div class="eyebrow">Exposure Anchor</div><div class="recommendation">{html.escape(anchor_summary.replace('Exposure Anchor: ', ''))}</div></div>
          <div><div class="eyebrow">Strategy</div><div class="recommendation">{html.escape(str(payload['recommended_strategy']['strategy_label']))}</div></div>
          <div><div class="eyebrow">Retained Cameras</div><div class="recommendation">{len(retained_rows)}</div></div>
          <div><div class="eyebrow">Excluded Cameras</div><div class="recommendation">{len(per_camera_rows) - len(retained_rows)}</div></div>
          <div><div class="eyebrow">Reference Candidate</div><div class="recommendation">{html.escape(str((best_reference or {}).get('camera_label') or 'n/a'))}</div></div>
        </div>
      </div>
      <dl class="meta-grid">
          <div class="meta-card"><dt>Created</dt><dd>{html.escape(str(payload['created_at']))}</dd></div>
          <div class="meta-card"><dt>Source Mode</dt><dd>{html.escape(str(payload['source_mode_label']))}</dd></div>
          <div class="meta-card"><dt>Target Type</dt><dd>{html.escape(str(payload['target_type']))}</dd></div>
          <div class="meta-card"><dt>Matching Domain</dt><dd>{html.escape(str(payload['matching_domain_label']))}</dd></div>
        <div class="meta-card"><dt>Strategies</dt><dd>{html.escape(str(payload['selected_strategy_labels']))}</dd></div>
        <div class="meta-card"><dt>Subset</dt><dd>{html.escape(str(payload['subset_label']))}</dd></div>
        <div class="meta-card"><dt>Exposure Anchor</dt><dd>{html.escape(anchor_summary.replace('Exposure Anchor: ', ''))}</dd></div>
        <div class="meta-card"><dt>Recommendation</dt><dd>{html.escape(str(payload['recommended_strategy']['strategy_label']))}</dd></div>
        <div class="meta-card"><dt>Calibration State</dt><dd>{html.escape(readiness_label.title())}</dd></div>
        <div class="meta-card"><dt>WB Model</dt><dd>{html.escape(str((payload.get('white_balance_model') or {}).get('model_label') or 'n/a'))}</dd></div>
        <div class="meta-card"><dt>Shared Kelvin</dt><dd>{html.escape(str((payload.get('white_balance_model') or {}).get('shared_kelvin') or 'per-camera'))}</dd></div>
      </dl>
    </section>

    <div class="grid two-up">
      <section class="section">
        <h2>Exposure Consistency Summary</h2>
        <p class="lead">A fast exposure-first view of how tightly the array is grouped before approval.</p>
        <div class="stats">
          <div class="stat"><div class="label">Median</div><div class="value">{float(exposure_summary['median']):.2f}</div></div>
          <div class="stat"><div class="label">Range</div><div class="value">{float(exposure_summary['spread']):.2f}</div></div>
          <div class="stat"><div class="label">Min / Max</div><div class="value">{float(exposure_summary['minimum']):.2f} / {float(exposure_summary['maximum']):.2f}</div></div>
          <div class="stat"><div class="label">Outliers</div><div class="value">{int(exposure_summary['outlier_count'])}</div></div>
        </div>
        <div style="margin-top:16px;">{chart_frame('Exposure Spread', payload['visuals']['exposure_plot_svg'])}</div>
      </section>
      <section class="section">
        <h2>Calibration Recommendation</h2>
        <p class="eyebrow">At A Glance</p>
        <p class="recommendation"><strong>{html.escape(summary_sentence)}</strong></p>
        <p class="subtle"><strong>Closest current reference candidate:</strong> {html.escape(str((best_reference or {}).get('camera_label') or 'n/a'))} {html.escape(best_reference_profile_text)}</p>
        <p class="eyebrow">Why This Was Chosen</p>
        <ul class="bullet-list">
          {''.join(f"<li>{html.escape(point)}</li>" for point in summary_points)}
        </ul>
        <div>{chart_frame('Strategy Comparison', payload['visuals']['strategy_chart_svg'])}</div>
        <p class="subtle" style="margin-top:12px;"><strong>Hero recommendation:</strong> {html.escape(str(hero_summary['candidate_clip_id'] or 'No clear hero'))}. {html.escape(str(hero_summary['reason']))}</p>
      </section>
    </div>

    <div class="grid two-up">
      <section class="section">
        <h2>Calibration Review Notes</h2>
        <p class="lead">{html.escape(str(run_assessment.get('operator_note') or 'Review the retained camera group before normalizing the array.'))}</p>
        <ul class="bullet-list">
          <li>{html.escape(str(run_assessment.get('anchor_summary') or 'No anchor summary available.'))}</li>
          <li>{html.escape(f"Retained cameras: {len(retained_rows)} / {int(run_assessment.get('camera_count', 0) or 0)}")}</li>
          <li>{html.escape(f"Excluded cameras: {len(per_camera_rows) - len(retained_rows)}")}</li>
          <li>{html.escape('Calibration payload is ready for review.' if bool(run_assessment.get('safe_to_push_later')) else 'Review retained cameras before normalizing the array.')}</li>
          {''.join(f"<li>{html.escape(str(reason))}</li>" for reason in list(run_assessment.get('gating_reasons') or []))}
        </ul>
      </section>
      <section class="section">
        <h2>Measurement Stability</h2>
        <p class="lead">Trust classes summarize sample stability, cluster membership, and correction size using the current sphere measurements.</p>
        <div>{chart_frame('Camera Trust', payload['visuals']['trust_chart_svg'])}</div>
      </section>
    </div>

    <div class="grid two-up">
      <section class="section">
        <h2>Before / After Exposure</h2>
        <p class="lead">Each line shows where a camera measured before correction and where the chosen strategy aims to land it.</p>
        <div>{chart_frame('Before / After Exposure', payload['visuals']['before_after_exposure_svg'])}</div>
      </section>
    </div>

    <section class="section" style="margin-top:18px;">
      <h2>Strategy Comparison</h2>
      <p class="lead">Strategies are ranked by correction size, anchor trustworthiness, and how safely they keep the array together.</p>
      <div class="strategy-grid">{''.join(strategy_cards)}</div>
    </section>

    <section class="section" style="margin-top:18px;">
      <h2>Per-Camera Analysis</h2>
      <p class="lead">Use this table first on set. It shows gray exposure, offset to anchor, the current exposure result, and the immediate operator action.</p>
      <table>
        <thead>
          <tr>
            <th>Camera</th>
            <th>Gray Exposure</th>
            <th>Offset to Anchor</th>
            <th>Result</th>
            <th>Recommended Action</th>
          </tr>
        </thead>
        <tbody>{''.join(table_rows)}</tbody>
      </table>
    </section>
  </div>
  <div id="chart-modal" class="chart-modal" hidden>
    <div class="chart-modal-card">
      <div class="chart-modal-top">
        <h3 id="chart-modal-title">Chart</h3>
        <button type="button" class="chart-modal-close" id="chart-modal-close">Close</button>
      </div>
      <div class="chart-modal-body" id="chart-modal-body"></div>
    </div>
  </div>
  <script>
    (function() {{
      const modal = document.getElementById('chart-modal');
      const body = document.getElementById('chart-modal-body');
      const title = document.getElementById('chart-modal-title');
      const close = document.getElementById('chart-modal-close');
      if (!modal || !body || !title || !close) return;
      document.addEventListener('click', (event) => {{
        const trigger = event.target.closest('.chart-launch');
        if (trigger) {{
          const frame = trigger.querySelector('.chart-frame');
          title.textContent = trigger.dataset.chartTitle || 'Chart';
          body.innerHTML = frame ? frame.outerHTML : '';
          modal.hidden = false;
          return;
        }}
        if (event.target === modal || event.target === close) {{
          modal.hidden = true;
          body.innerHTML = '';
        }}
      }});
      document.addEventListener('keydown', (event) => {{
        const trigger = event.target.closest ? event.target.closest('.chart-launch') : null;
        if (trigger && (event.key === 'Enter' || event.key === ' ')) {{
          event.preventDefault();
          const frame = trigger.querySelector('.chart-frame');
          title.textContent = trigger.dataset.chartTitle || 'Chart';
          body.innerHTML = frame ? frame.outerHTML : '';
          modal.hidden = false;
          return;
        }}
        if (event.key === 'Escape') {{
          modal.hidden = true;
          body.innerHTML = '';
        }}
      }});
    }})();
  </script>
</body>
</html>"""


def build_lightweight_analysis_report(
    input_path: str,
    *,
    out_dir: str,
    source_mode: str = "local_folder",
    source_mode_label_value: Optional[str] = None,
    source_input_path: Optional[str] = None,
    ingest_manifest: Optional[Dict[str, object]] = None,
    target_type: Optional[str] = None,
    processing_mode: Optional[str] = None,
    run_label: Optional[str] = None,
    matching_domain: str = "scene",
    selected_clip_ids: Optional[List[str]] = None,
    selected_clip_groups: Optional[List[str]] = None,
    calibration_roi: Optional[Dict[str, float]] = None,
    target_strategies: Optional[List[str]] = None,
    reference_clip_id: Optional[str] = None,
    hero_clip_id: Optional[str] = None,
    exposure_anchor_mode: Optional[str] = None,
    manual_target_stops: Optional[float] = None,
    manual_target_ire: Optional[float] = None,
    clear_cache: bool = True,
    progress_path: Optional[str] = None,
) -> Dict[str, object]:
    report_started_at = time.perf_counter()
    raise_if_cancelled("Run cancelled before lightweight report generation.")
    root = Path(input_path).expanduser().resolve()
    out_root = Path(out_dir).expanduser().resolve()
    analysis_records = _load_analysis_records(input_path)
    requested_matching_domain = _normalize_matching_domain(matching_domain)
    resolved_matching_domain = "perceptual"
    reusable_analysis_measurements = all(
        str((record.get("diagnostics", {}) or {}).get("exposure_measurement_domain") or "") == "rendered_preview_ipp2"
        and str((record.get("diagnostics", {}) or {}).get("rendered_measurement_preview_path") or "").strip()
        for record in analysis_records
    )
    out_root.mkdir(parents=True, exist_ok=True)
    if clear_cache:
        clear_preview_cache(
            str(root),
            report_dir=str(out_root),
            preserve_measurement_previews=reusable_analysis_measurements,
        )
    emit_review_progress(
        progress_path,
        phase="lightweight_report_loaded",
        detail=f"Loaded {len(analysis_records)} analysis record(s) for lightweight report.",
        stage_label="Building report",
        clip_count=len(analysis_records),
        elapsed_seconds=time.perf_counter() - report_started_at,
        review_mode="lightweight_analysis",
    )
    resolved_source_mode = source_mode
    resolved_source_mode_label = source_mode_label_value or source_mode_label(resolved_source_mode)
    resolved_strategies = _ensure_anchor_strategy_list(
        list(target_strategies or list(DEFAULT_REVIEW_TARGET_STRATEGIES)),
        exposure_anchor_mode=exposure_anchor_mode,
    )
    explicit_anchor_strategy_key = _strategy_key_for_anchor_mode(normalize_exposure_anchor_mode(exposure_anchor_mode))
    resolved_run_label = run_label or root.name or "review"
    monitoring_measurements_by_clip: Dict[str, Dict[str, object]] = {}
    measurement_preview_rendered = 0
    reused_measurement_renders = 0
    array_calibration_payload = _load_array_calibration_payload(str(root))
    measurement_preview_settings = _measurement_preview_settings_for_domain(resolved_matching_domain)
    if reusable_analysis_measurements:
        emit_review_progress(
            progress_path,
            phase="lightweight_report_measurement_previews_reused",
            detail="Reusing representative IPP2 monitoring previews from analysis.",
            stage_label="Reusing measurement previews",
            clip_count=len(analysis_records),
            elapsed_seconds=time.perf_counter() - report_started_at,
            review_mode="lightweight_analysis",
        )
        for record in analysis_records:
            clip_id = str(record["clip_id"])
            reusable_measurement = _rendered_measurement_from_diagnostics(record)
            monitoring_measurements_by_clip[clip_id] = reusable_measurement
            reused_measurement_renders += 1
        measurement_preview_rendered = reused_measurement_renders
    else:
        redline_executable = _resolve_redline_executable()
        redline_capabilities = _detect_redline_capabilities(redline_executable)
        emit_review_progress(
            progress_path,
            phase="lightweight_report_measurement_previews_start",
            detail="Rendering representative IPP2 monitoring previews for lightweight measurement.",
            stage_label="Rendering measurement previews",
            clip_count=len(analysis_records),
            elapsed_seconds=time.perf_counter() - report_started_at,
            review_mode="lightweight_analysis",
        )
        measurement_preview_paths = generate_preview_stills(
            input_path,
            analysis_records=analysis_records,
            previews_dir=str(root / "previews" / "_measurement"),
            preview_settings=measurement_preview_settings,
            redline_capabilities=redline_capabilities,
            strategy_payloads=[],
            run_id=resolved_run_label,
        )
        measurement_preview_rendered = len(measurement_preview_paths)
        emit_review_progress(
            progress_path,
            phase="lightweight_report_measurement_previews_complete",
            detail=f"Rendered {measurement_preview_rendered} representative monitoring preview(s).",
            stage_label="Rendering measurement previews",
            clip_count=len(analysis_records),
            elapsed_seconds=time.perf_counter() - report_started_at,
            review_mode="lightweight_analysis",
        )
        for record in analysis_records:
            clip_id = str(record["clip_id"])
            original_frame = measurement_preview_paths.get(clip_id, {}).get("original")
            if original_frame:
                monitoring_measurements_by_clip[clip_id] = _measure_rendered_preview_roi_ipp2(
                    str(original_frame),
                    record.get("diagnostics", {}).get("calibration_roi") or calibration_roi,
                )
    quality_by_clip = {
        **_quality_by_clip(array_calibration_payload),
        **{
            clip_id: _monitoring_quality_for_measurement(measurement)
            for clip_id, measurement in monitoring_measurements_by_clip.items()
        },
    }
    strategy_payloads = _build_strategy_payloads(
        analysis_records,
        target_strategies=resolved_strategies,
        reference_clip_id=reference_clip_id,
        hero_clip_id=hero_clip_id,
        target_type=target_type,
        monitoring_measurements_by_clip=monitoring_measurements_by_clip,
        matching_domain=resolved_matching_domain,
        quality_by_clip=quality_by_clip,
        anchor_target_log2=_anchor_target_log2_from_array_payload(array_calibration_payload),
        exposure_anchor_mode=exposure_anchor_mode,
        manual_target_stops=manual_target_stops,
        manual_target_ire=manual_target_ire,
    )
    emit_review_progress(
        progress_path,
        phase="lightweight_report_strategies",
        detail=f"Built {len(strategy_payloads)} lightweight strategy payload(s).",
        stage_label="Building report",
        clip_count=len(analysis_records),
        elapsed_seconds=time.perf_counter() - report_started_at,
        review_mode="lightweight_analysis",
    )
    if quality_by_clip:
        for payload in strategy_payloads:
            for clip in payload.get("clips", []):
                quality = quality_by_clip.get(str(clip.get("clip_id")))
                if not quality:
                    continue
                clip["confidence"] = float(quality.get("confidence", clip.get("confidence", 0.0)) or 0.0)
                clip["neutral_sample_log2_spread"] = float(
                    quality.get("neutral_sample_log2_spread", clip.get("neutral_sample_log2_spread", 0.0)) or 0.0
                )
                clip["neutral_sample_chromaticity_spread"] = float(
                    quality.get(
                        "neutral_sample_chromaticity_spread",
                        clip.get("neutral_sample_chromaticity_spread", 0.0),
                    )
                    or 0.0
                )
                clip["post_color_residual"] = float(
                    quality.get("post_color_residual", clip.get("post_color_residual", 0.0)) or 0.0
                )
                clip["post_exposure_residual_stops"] = float(
                    quality.get(
                        "post_exposure_residual_stops",
                        clip.get("post_exposure_residual_stops", 0.0),
                    )
                    or 0.0
                )
                quality_flags = [str(flag) for flag in (quality.get("flags") or []) if str(flag).strip()]
                clip["flags"] = list(dict.fromkeys([*(clip.get("flags") or []), *quality_flags]))
    recommended_strategy = _choose_strategy_with_anchor(
        strategy_payloads,
        explicit_anchor_strategy_key=explicit_anchor_strategy_key,
    )
    recommended_payload = next(payload for payload in strategy_payloads if payload["strategy_key"] == recommended_strategy["strategy_key"])
    exposure_summary = _exposure_summary(recommended_payload["clips"])
    hero_summary = _hero_candidate_summary(strategy_payloads)
    outlier_threshold = float(exposure_summary.get("outlier_threshold", 0.35))
    optimal_payload_for_rows = next((item for item in strategy_payloads if str(item.get("strategy_key") or "") == "optimal_exposure"), None)
    optimal_diag_for_rows = dict((optimal_payload_for_rows or {}).get("selection_diagnostics") or {})
    primary_cluster_indices = {int(index) for index in ((optimal_diag_for_rows.get("primary_cluster") or {}).get("indices") or [])}
    screened_candidates_by_clip = {
        str(item.get("clip_id") or ""): dict(item)
        for item in list(optimal_diag_for_rows.get("screened_candidates") or [])
        if str(item.get("clip_id") or "").strip()
    }
    per_camera_rows = []
    for index, record in enumerate(analysis_records):
        clip_id = str(record["clip_id"])
        strategy_clip = next(item for item in recommended_payload["clips"] if item["clip_id"] == clip_id)
        measured_log2 = float(strategy_clip.get("display_scalar_log2", strategy_clip["measured_log2_luminance"]) or 0.0)
        is_outlier = abs(measured_log2 - float(exposure_summary["median"])) > outlier_threshold
        trust_details = _camera_trust_details(
            clip_id=clip_id,
            confidence=float(strategy_clip.get("confidence", 0.0) or 0.0),
            sample_log2_spread=float(strategy_clip.get("neutral_sample_log2_spread", 0.0) or 0.0),
            sample_chroma_spread=float(strategy_clip.get("neutral_sample_chromaticity_spread", 0.0) or 0.0),
            measured_log2=measured_log2,
            final_offset=float(strategy_clip["exposure_offset_stops"]),
            exposure_summary=exposure_summary,
            in_primary_cluster=(not primary_cluster_indices or index in primary_cluster_indices),
            screened_reasons=[str(item) for item in (screened_candidates_by_clip.get(clip_id, {}) or {}).get("reasons", []) if str(item).strip()],
        )
        note_bits: List[str] = []
        if is_outlier:
            note_bits.append("Exposure outlier against the central cluster.")
        if strategy_clip.get("flags"):
            note_bits.append(", ".join(str(flag) for flag in strategy_clip["flags"]))
        if strategy_clip.get("is_hero_camera"):
            note_bits.append("Hero camera receives identity correction.")
        if not note_bits:
            note_bits.append("Within normal correction range for this subset.")
        per_camera_rows.append(
            {
                "camera_label": _camera_label_for_reporting(clip_id),
                "clip_id": clip_id,
                "display_scalar_log2": measured_log2,
                "display_scalar_domain": str(strategy_clip.get("display_scalar_domain") or "display_ipp2"),
                "measured_log2_luminance": measured_log2,
                "anchor_mode": str(strategy_clip.get("anchor_mode") or recommended_payload.get("anchor_mode") or ""),
                "anchor_source": str(strategy_clip.get("anchor_source") or recommended_payload.get("anchor_source") or ""),
                "anchor_scalar_value": float(strategy_clip.get("anchor_scalar_value", recommended_payload.get("anchor_scalar_value", 0.0)) or 0.0),
                "anchor_ire_summary": str(strategy_clip.get("anchor_ire_summary") or recommended_payload.get("anchor_ire_summary") or "n/a"),
                "measured_scalar_value": measured_log2,
                "measured_gray_exposure_summary": str(strategy_clip.get("gray_exposure_summary") or strategy_clip.get("aggregate_sphere_profile") or "n/a"),
                "sample_1_ire": float(strategy_clip.get("sample_1_ire", strategy_clip.get("bright_ire", 0.0)) or 0.0),
                "sample_2_ire": float(strategy_clip.get("sample_2_ire", strategy_clip.get("center_ire", 0.0)) or 0.0),
                "sample_3_ire": float(strategy_clip.get("sample_3_ire", strategy_clip.get("dark_ire", 0.0)) or 0.0),
                "monitoring_measurement_source": str(strategy_clip.get("monitoring_measurement_source") or ""),
                "raw_offset_stops": float(record.get("raw_offset_stops", 0.0) or 0.0),
                "final_offset_stops": float(strategy_clip["exposure_offset_stops"]),
                "camera_offset_from_anchor": float(strategy_clip.get("camera_offset_from_anchor", strategy_clip["exposure_offset_stops"]) or 0.0),
                "derived_display_scalar_log2": float(strategy_clip.get("display_scalar_log2", strategy_clip.get("measured_log2_luminance_monitoring", strategy_clip.get("measured_log2_luminance", 0.0))) or 0.0),
                "derived_exposure_value": float(strategy_clip.get("display_scalar_log2", strategy_clip.get("measured_log2_luminance_monitoring", strategy_clip.get("measured_log2_luminance", 0.0))) or 0.0),
                "derived_exposure_offset_stops": float(strategy_clip.get("camera_offset_from_anchor", strategy_clip["exposure_offset_stops"]) or 0.0),
                "commit_values": strategy_clip["commit_values"],
                "pre_color_residual": float(strategy_clip.get("pre_color_residual", 0.0) or 0.0),
                "post_color_residual": float(strategy_clip.get("post_color_residual", 0.0) or 0.0),
                "white_balance_model": strategy_clip.get("white_balance_model"),
                "white_balance_model_label": strategy_clip.get("white_balance_model_label"),
                "shared_kelvin": strategy_clip.get("shared_kelvin"),
                "shared_tint": strategy_clip.get("shared_tint"),
                "rgb_gain_summary": ", ".join(f"{float(value):.3f}" for value in strategy_clip["rgb_gains"]),
                "confidence": float(strategy_clip.get("confidence", 0.0) or 0.0),
                "neutral_sample_log2_spread": float(strategy_clip.get("neutral_sample_log2_spread", 0.0) or 0.0),
                "neutral_sample_chromaticity_spread": float(strategy_clip.get("neutral_sample_chromaticity_spread", 0.0) or 0.0),
                "is_outlier": is_outlier,
                "is_hero_camera": bool(strategy_clip.get("is_hero_camera")),
                "trust_class": str(trust_details["trust_class"]),
                "trust_score": float(trust_details["trust_score"]),
                "trust_reason": str(trust_details["trust_reason"]),
                "stability_label": str(trust_details["stability_label"]),
                "correction_confidence": str(trust_details["correction_confidence"]),
                "reference_use": str(trust_details["reference_use"]),
                "screening_reasons": list(trust_details.get("screened_reasons") or []),
                "operator_summary": (
                    f"{str(trust_details['trust_class']).replace('_', ' ')} / {str(trust_details['reference_use'])} / "
                    f"{str(trust_details['trust_reason'])}. Offset to anchor {float(strategy_clip.get('camera_offset_from_anchor', strategy_clip['exposure_offset_stops'])):+.2f}."
                ),
                "note": " ".join(note_bits),
            }
        )
    strategy_summaries = []
    for payload in strategy_payloads:
        metrics = _strategy_distribution_metrics(payload)
        strategy_summaries.append(
            {
                "strategy_key": payload["strategy_key"],
                "strategy_label": payload["strategy_label"],
                "reference_clip_id": payload.get("reference_clip_id"),
                "hero_clip_id": payload.get("hero_clip_id"),
                "anchor_mode": payload.get("anchor_mode"),
                "anchor_source": payload.get("anchor_source"),
                "anchor_scalar_value": payload.get("anchor_scalar_value"),
                "anchor_ire_summary": payload.get("anchor_ire_summary"),
                "anchor_summary": payload.get("anchor_summary"),
                "target_log2_luminance": float(payload["target_log2_luminance"]),
                "summary": payload.get("strategy_summary"),
                "selection_diagnostics": dict(payload.get("selection_diagnostics") or {}),
                "white_balance_model": payload.get("white_balance_model"),
                "correction_metrics": metrics,
                "recommended": payload["strategy_key"] == recommended_strategy["strategy_key"],
            }
        )
    run_assessment = _build_run_assessment(
        per_camera_rows=per_camera_rows,
        recommended_payload=recommended_payload,
        strategy_summaries=strategy_summaries,
        exposure_summary=exposure_summary,
    )
    subset_bits = []
    if selected_clip_groups:
        subset_bits.append(f"groups {', '.join(str(item) for item in selected_clip_groups)}")
    if selected_clip_ids:
        subset_bits.append(f"{len(selected_clip_ids)} explicit clips")
    subset_label = " / ".join(subset_bits) if subset_bits else "full discovered subset"
    synopsis = _build_lightweight_synopsis(
        exposure_summary=exposure_summary,
        strategy_summaries=strategy_summaries,
        recommended_strategy=recommended_strategy,
        hero_summary=hero_summary,
    )
    operator_recommendation = str(run_assessment.get("operator_note") or "")
    payload = {
        "report_kind": "lightweight_analysis",
        "review_mode": "lightweight_analysis",
        "review_mode_label": review_mode_label("lightweight_analysis"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_path": str(root),
        "source_mode": resolved_source_mode,
        "source_mode_label": resolved_source_mode_label,
        "source_input_path": source_input_path or str(root),
        "ingest_manifest": ingest_manifest,
        "run_label": resolved_run_label,
        "target_type": target_type or "unspecified",
        "processing_mode": processing_mode or "both",
        "matching_domain": resolved_matching_domain,
        "matching_domain_label": _matching_domain_label(resolved_matching_domain),
        "requested_matching_domain": requested_matching_domain,
        "requested_matching_domain_label": _matching_domain_label(requested_matching_domain),
        "requested_matching_domain": requested_matching_domain,
        "requested_matching_domain_label": _matching_domain_label(requested_matching_domain),
        "selected_clip_ids": [str(item) for item in (selected_clip_ids or []) if str(item).strip()],
        "selected_clip_groups": [str(item) for item in (selected_clip_groups or []) if str(item).strip()],
        "selected_strategy_labels": ", ".join(str(item["strategy_label"]) for item in strategy_summaries),
        "reference_clip_id": reference_clip_id,
        "hero_clip_id": hero_clip_id,
        "exposure_anchor_mode": normalize_exposure_anchor_mode(exposure_anchor_mode),
        "exposure_anchor_strategy_key": explicit_anchor_strategy_key or str(recommended_payload.get("strategy_key") or ""),
        "exposure_anchor_summary": str(recommended_payload.get("anchor_summary") or recommended_strategy.get("anchor_summary") or ""),
        "anchor_mode": str(recommended_payload.get("anchor_mode") or recommended_strategy.get("anchor_mode") or ""),
        "anchor_source": str(recommended_payload.get("anchor_source") or recommended_strategy.get("anchor_source") or ""),
        "anchor_scalar_value": float(recommended_payload.get("anchor_scalar_value", 0.0) or 0.0),
        "anchor_ire_summary": str(recommended_payload.get("anchor_ire_summary") or recommended_strategy.get("anchor_ire_summary") or "n/a"),
        "clip_count": len(analysis_records),
        "calibration_roi": calibration_roi or (analysis_records[0].get("diagnostics", {}).get("calibration_roi") if analysis_records else None),
        "color_preview_enabled": False,
        "color_preview_status": _color_preview_policy()["status"],
        "color_preview_note": _color_preview_policy()["note"],
        "executive_synopsis": synopsis,
        "subset_label": subset_label,
        "exposure_summary": exposure_summary,
        "strategy_comparison": strategy_summaries,
        "white_balance_model": dict(recommended_payload.get("white_balance_model", {})),
        "recommended_strategy": recommended_strategy,
        "run_assessment": run_assessment,
        "hero_recommendation": hero_summary,
        "operator_recommendation": operator_recommendation,
        "per_camera_analysis": per_camera_rows,
        "measurement_render_count": measurement_preview_rendered,
        "measurement_domain_trace": {
            "requested_matching_domain": requested_matching_domain,
            "effective_matching_domain": resolved_matching_domain,
            "measurement_source": "rendered_preview_ipp2",
            "measurement_preview_transform": _preview_transform_label(measurement_preview_settings),
            "measurement_preview_reused_from_analysis": bool(reusable_analysis_measurements),
        },
        "shared_originals": [],
        "strategies": [],
        "visuals": {
            "exposure_plot_svg": _build_exposure_plot_svg(
                recommended_payload["clips"],
                target_log2=float(recommended_payload["target_log2_luminance"]),
                reference_clip_id=str(recommended_payload.get("reference_clip_id") or ""),
                outlier_threshold=float(exposure_summary.get("outlier_threshold", 0.35) or 0.35),
            ),
            "before_after_exposure_svg": _build_before_after_exposure_svg(
                recommended_payload["clips"],
                target_log2=float(recommended_payload["target_log2_luminance"]),
            ),
            "strategy_chart_svg": _build_strategy_chart_svg(strategy_summaries),
            "trust_chart_svg": _build_trust_chart_svg(per_camera_rows),
        },
    }
    payload["debug_exposure_trace"] = _build_exposure_trace_artifacts(
        analysis_records=analysis_records,
        recommended_payload=recommended_payload,
        strategy_summaries=strategy_summaries,
        exposure_summary=exposure_summary,
        out_root=out_root,
    )
    report_json_path = out_root / "contact_sheet.json"
    report_html_path = out_root / "contact_sheet.html"
    review_manifest_path = out_root / "review_manifest.json"
    previews_root = root / "previews"
    previews_root.mkdir(parents=True, exist_ok=True)
    preview_manifest_path = previews_root / "preview_commands.json"
    try:
        raise_if_cancelled("Run cancelled before writing lightweight report artifacts.")
        report_json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        report_html_path.write_text(_render_lightweight_analysis_html(payload), encoding="utf-8")
        review_manifest_path.write_text(
            json.dumps(
                {
                    "review_mode": "lightweight_analysis",
                    "review_mode_label": review_mode_label("lightweight_analysis"),
                    "input_path": str(root),
                    "source_mode": payload["source_mode"],
                    "source_mode_label": payload["source_mode_label"],
                    "source_input_path": payload["source_input_path"],
                    "run_label": resolved_run_label,
                    "target_type": payload["target_type"],
                    "processing_mode": payload["processing_mode"],
                    "matching_domain": payload["matching_domain"],
                    "matching_domain_label": payload["matching_domain_label"],
                    "selected_clip_ids": payload["selected_clip_ids"],
                    "selected_clip_groups": payload["selected_clip_groups"],
                    "target_strategies": resolved_strategies,
                    "reference_clip_id": reference_clip_id,
                    "hero_clip_id": hero_clip_id,
                    "clip_count": payload["clip_count"],
                    "report_json": str(report_json_path),
                    "report_html": str(report_html_path),
                    "preview_report_pdf": None,
                    "bulk_preview_rendering": "skipped",
                    "measurement_render_count": measurement_preview_rendered,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        preview_manifest_path.write_text(
            json.dumps(
                {
                    "review_mode": "lightweight_analysis",
                    "commands": [],
                    "skipped_bulk_preview_rendering": True,
                    "measurement_render_count": measurement_preview_rendered,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    except CancellationError:
        for artifact in (report_json_path, report_html_path, review_manifest_path, preview_manifest_path):
            if artifact.exists():
                artifact.unlink()
        raise
    return {
        "report_json": str(report_json_path),
        "report_html": str(report_html_path),
        "preview_report_pdf": None,
        "previews_dir": str(previews_root),
        "review_manifest": str(review_manifest_path),
        "clip_count": len(analysis_records),
        "preview_transform": None,
        "measurement_preview_transform": _preview_transform_label(_measurement_preview_settings_for_domain(resolved_matching_domain)),
        "run_label": resolved_run_label,
        "matching_domain": resolved_matching_domain,
        "matching_domain_label": _matching_domain_label(resolved_matching_domain),
        "preview_mode": "analysis",
        "preview_settings": None,
        "measurement_preview_settings": _measurement_preview_settings_for_domain(resolved_matching_domain),
        "redline_capabilities": None,
        "review_mode": "lightweight_analysis",
        "review_mode_label": review_mode_label("lightweight_analysis"),
    }


def _build_redline_preview_command(
    source_path: str,
    *,
    output_path: str,
    frame_index: int,
    exposure_stops: Optional[float],
    color_cdl: Optional[Dict[str, object]],
    color_method: Optional[str],
    redline_executable: str,
    preview_settings: Dict[str, object],
    redline_capabilities: Dict[str, object],
    use_as_shot_metadata: bool,
    kelvin: Optional[int] = 5600,
    tint: Optional[float] = 0,
    rmd_path: Optional[str] = None,
    use_rmd_mode: int = 1,
) -> List[str]:
    if kelvin is None:
        kelvin = 5600
    if tint is None:
        tint = 0
    command = [
        redline_executable,
        "--i",
        str(Path(source_path).expanduser().resolve()),
        "--o",
        str(Path(output_path).expanduser().resolve()),
        "--format",
        "3",
        "--start",
        str(frame_index),
        "--frameCount",
        "1",
        "--colorSciVersion",
        "3",
        "--silent",
    ]
    if use_as_shot_metadata:
        command.append("--useMeta")
    if redline_capabilities.get("supports_color_space"):
        command.extend(["--colorSpace", str(COLOR_SPACE_CODES[str(preview_settings["output_space"])])])
    if redline_capabilities.get("supports_gamma_curve"):
        command.extend(["--gammaCurve", str(GAMMA_CODES[str(preview_settings["output_gamma"])])])
    if redline_capabilities.get("supports_output_tonemap"):
        command.extend(["--outputToneMap", str(TONEMAP_CODES[str(preview_settings["output_tonemap"])])])
    if redline_capabilities.get("supports_rolloff"):
        command.extend(["--rollOff", str(ROLLOFF_CODES[str(preview_settings["highlight_rolloff"])])])
    if redline_capabilities.get("supports_shadow_control"):
        command.extend(["--shadow", f"{float(SHADOW_ROLLOFF_VALUES[str(preview_settings['shadow_rolloff'])]):.3f}"])
    if preview_settings.get("lut_path") and redline_capabilities.get("supports_lut"):
        command.extend(["--lut", str(preview_settings["lut_path"])])
    if rmd_path and redline_capabilities.get("supports_load_rmd"):
        command.extend(["--loadRMD", str(Path(rmd_path).expanduser().resolve()), "--useRMD", str(use_rmd_mode)])
    else:
        if kelvin is not None:
            command.extend(["--kelvin", str(int(round(float(kelvin))))])
        if tint is not None:
            command.extend(["--tint", f"{float(tint):.6f}"])
        if exposure_stops is not None:
            command.extend([f"--{REDLINE_DIRECT_EXPOSURE_PARAMETER}", f"{float(exposure_stops):.6f}"])
        if color_cdl is not None and color_method:
            if color_method == "cdl":
                slope = [float(value) for value in color_cdl.get("slope", [1.0, 1.0, 1.0])]
                offset = [float(value) for value in color_cdl.get("offset", [0.0, 0.0, 0.0])]
                power = [float(value) for value in color_cdl.get("power", [1.0, 1.0, 1.0])]
                saturation = float(color_cdl.get("saturation", 1.0))
                command.extend(
                    [
                        "--cdlRedSlope",
                        f"{slope[0]:.6f}",
                        "--cdlGreenSlope",
                        f"{slope[1]:.6f}",
                        "--cdlBlueSlope",
                        f"{slope[2]:.6f}",
                        "--cdlRedOffset",
                        f"{offset[0]:.6f}",
                        "--cdlGreenOffset",
                        f"{offset[1]:.6f}",
                        "--cdlBlueOffset",
                        f"{offset[2]:.6f}",
                        "--cdlRedPower",
                        f"{power[0]:.6f}",
                        "--cdlGreenPower",
                        f"{power[1]:.6f}",
                        "--cdlBluePower",
                        f"{power[2]:.6f}",
                        "--cdlSaturation",
                        f"{saturation:.6f}",
                    ]
                )
            elif color_method == "rgb_gain":
                slope = [float(value) for value in color_cdl.get("slope", [1.0, 1.0, 1.0])]
                command.extend(
                    [
                        "--redGain",
                        f"{slope[0]:.6f}",
                        "--greenGain",
                        f"{slope[1]:.6f}",
                        "--blueGain",
                        f"{slope[2]:.6f}",
                    ]
                )
    return command


def _build_redline_metadata_probe_command(
    source_path: str,
    *,
    redline_executable: str,
    rmd_path: Optional[str],
    exposure_stops: Optional[float],
    kelvin: Optional[int],
    tint: Optional[float],
    color_cdl: Optional[Dict[str, object]],
    color_method: Optional[str],
    redline_capabilities: Dict[str, object],
    use_rmd_mode: int,
) -> List[str]:
    command = [
        redline_executable,
        "--i",
        str(Path(source_path).expanduser().resolve()),
        "--printMeta",
        "1",
        "--noRender",
    ]
    if rmd_path and redline_capabilities.get("supports_load_rmd"):
        command.extend(["--loadRMD", str(Path(rmd_path).expanduser().resolve()), "--useRMD", str(use_rmd_mode)])
    else:
        if kelvin is not None:
            command.extend(["--kelvin", str(int(round(float(kelvin))))])
        if tint is not None:
            command.extend(["--tint", f"{float(tint):.6f}"])
        if exposure_stops is not None:
            command.extend([f"--{REDLINE_DIRECT_EXPOSURE_PARAMETER}", f"{float(exposure_stops):.6f}"])
        if color_cdl is not None and color_method:
            slope = [float(value) for value in color_cdl.get("slope", [1.0, 1.0, 1.0])]
            if color_method == "cdl":
                command.extend(
                    [
                        "--cdlRedSlope",
                        f"{slope[0]:.6f}",
                        "--cdlGreenSlope",
                        f"{slope[1]:.6f}",
                        "--cdlBlueSlope",
                        f"{slope[2]:.6f}",
                    ]
                )
            elif color_method == "rgb_gain":
                command.extend(
                    [
                        "--redGain",
                        f"{slope[0]:.6f}",
                        "--greenGain",
                        f"{slope[1]:.6f}",
                        "--blueGain",
                        f"{slope[2]:.6f}",
                    ]
                )
    return command


def _probe_redline_application(
    source_path: str,
    *,
    redline_executable: str,
    rmd_path: Optional[str],
    exposure_stops: Optional[float],
    kelvin: Optional[int],
    tint: Optional[float],
    color_cdl: Optional[Dict[str, object]],
    color_method: Optional[str],
    redline_capabilities: Dict[str, object],
    use_rmd_mode: int = 1,
) -> Dict[str, object]:
    raise_if_cancelled("Run cancelled before REDLine metadata probe.")
    command = _build_redline_metadata_probe_command(
        source_path,
        redline_executable=redline_executable,
        rmd_path=rmd_path,
        exposure_stops=exposure_stops,
        kelvin=kelvin,
        tint=tint,
        color_cdl=color_cdl,
        color_method=color_method,
        redline_capabilities=redline_capabilities,
        use_rmd_mode=use_rmd_mode,
    )
    completed = run_cancellable_subprocess(command)
    output = (completed.stdout or "") + (completed.stderr or "")
    if rmd_path:
        applied = completed.returncode == 0 and "Error parsing RMD file" not in output
        method = "rmd" if applied else "cli_flags"
    else:
        applied = completed.returncode == 0
        method = "cli_flags"
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "applied": applied,
        "method": method,
    }


def _resolve_rendered_output_path(output_path: str | Path) -> Path:
    candidate = Path(output_path).expanduser().resolve()
    if candidate.exists():
        return candidate
    matches = sorted(candidate.parent.glob(f"{candidate.name}.*"))
    if matches:
        matches[0].replace(candidate)
    return candidate


def _compute_image_difference_metrics(reference_path: str | Path, candidate_path: str | Path) -> Dict[str, object]:
    reference = np.asarray(Image.open(Path(reference_path)).convert("RGB"), dtype=np.float32)
    candidate = np.asarray(Image.open(Path(candidate_path)).convert("RGB"), dtype=np.float32)
    if reference.shape != candidate.shape:
        return {
            "mean_absolute_difference": None,
            "max_absolute_difference": None,
            "shape_mismatch": True,
            "pixel_output_changed": False,
        }
    diff = np.abs(reference - candidate)
    return {
        "mean_absolute_difference": float(np.mean(diff)),
        "max_absolute_difference": float(np.max(diff)),
        "shape_mismatch": False,
        "pixel_output_changed": bool(float(np.mean(diff)) >= 1e-3),
    }


def _is_identity_rgb_gains(rgb_gains: Optional[List[float]]) -> bool:
    if rgb_gains is None:
        return True
    return all(abs(float(value) - 1.0) <= 1e-9 for value in rgb_gains)


def _is_identity_cdl_payload(color_cdl: Optional[Dict[str, object]]) -> bool:
    return _color_is_identity_cdl_payload(color_cdl)


def render_preview_frame(
    input_r3d: str,
    output_path: str,
    *,
    frame_index: int,
    redline_executable: str,
    redline_capabilities: Dict[str, object],
    preview_settings: Dict[str, object],
    use_as_shot_metadata: bool,
    exposure: Optional[float] = None,
    kelvin: Optional[int] = None,
    tint: Optional[float] = None,
    red_gain: Optional[float] = None,
    green_gain: Optional[float] = None,
    blue_gain: Optional[float] = None,
    color_cdl: Optional[Dict[str, object]] = None,
    rmd_path: Optional[str] = None,
    use_rmd_mode: int = 1,
    color_method: Optional[str] = None,
) -> Dict[str, object]:
    raise_if_cancelled("Run cancelled before preview render.")
    resolved_color_cdl = copy.deepcopy(color_cdl) if color_cdl is not None else None
    resolved_color_method = color_method
    if red_gain is not None or green_gain is not None or blue_gain is not None:
        resolved_color_cdl = {
            "slope": [
                float(red_gain if red_gain is not None else 1.0),
                float(green_gain if green_gain is not None else 1.0),
                float(blue_gain if blue_gain is not None else 1.0),
            ],
            "offset": [0.0, 0.0, 0.0],
            "power": [1.0, 1.0, 1.0],
            "saturation": 1.0,
        }
        if resolved_color_method is None:
            resolved_color_method = "rgb_gain"
    command = _build_redline_preview_command(
        input_r3d,
        output_path=output_path,
        frame_index=frame_index,
        exposure_stops=exposure,
        kelvin=kelvin,
        tint=tint,
        color_cdl=resolved_color_cdl,
        color_method=resolved_color_method,
        redline_executable=redline_executable,
        preview_settings=preview_settings,
        redline_capabilities=redline_capabilities,
        use_as_shot_metadata=use_as_shot_metadata,
        rmd_path=rmd_path,
        use_rmd_mode=use_rmd_mode,
    )
    completed = run_cancellable_subprocess(command)
    actual_output = _resolve_rendered_output_path(output_path)
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "output_path": str(actual_output),
    }


def _validate_rmd_render(
    *,
    clip_id: str,
    variant: str,
    strategy_key: str,
    source_path: str,
    frame_index: int,
    baseline_path: Path,
    preview_root: Path,
    redline_executable: str,
    redline_capabilities: Dict[str, object],
    preview_settings: Dict[str, object],
    exposure_stops: float,
    rgb_gains: Optional[List[float]],
    rmd_path: str,
    use_rmd_mode: int,
) -> Dict[str, object]:
    validation_root = preview_root / "_rmd_validation"
    validation_root.mkdir(parents=True, exist_ok=True)
    run_id = preview_root.parent.name
    direct_output = validation_root / f"{clip_id}.{variant}.validation.flags.{strategy_key}.{run_id}.jpg"
    direct_render = render_preview_frame(
        source_path,
        str(direct_output),
        frame_index=frame_index,
        redline_executable=redline_executable,
        redline_capabilities=redline_capabilities,
        preview_settings=preview_settings,
        use_as_shot_metadata=True,
        exposure=exposure_stops,
        red_gain=float(rgb_gains[0]) if rgb_gains is not None else None,
        green_gain=float(rgb_gains[1]) if rgb_gains is not None else None,
        blue_gain=float(rgb_gains[2]) if rgb_gains is not None else None,
        color_method="cdl" if rgb_gains is not None else None,
    )
    if int(direct_render["returncode"]) != 0:
        raise RuntimeError(
            f"REDLine direct-flag validation render failed for {clip_id} ({variant}). "
            f"Command: {shlex.join(direct_render['command'])}. STDERR: {str(direct_render['stderr']).strip()}"
        )
    direct_path = Path(str(direct_render["output_path"]))
    if not direct_path.exists():
        raise RuntimeError(f"REDLine direct-flag validation render did not create expected file: {direct_path}")

    rmd_output = validation_root / f"{clip_id}.{variant}.validation.rmd.{strategy_key}.{run_id}.jpg"
    rmd_render = render_preview_frame(
        source_path,
        str(rmd_output),
        frame_index=frame_index,
        redline_executable=redline_executable,
        redline_capabilities=redline_capabilities,
        preview_settings=preview_settings,
        use_as_shot_metadata=False,
        rmd_path=rmd_path,
        use_rmd_mode=use_rmd_mode,
    )
    if int(rmd_render["returncode"]) != 0:
        raise RuntimeError(
            f"REDLine RMD validation render failed for {clip_id} ({variant}). "
            f"Command: {shlex.join(rmd_render['command'])}. STDERR: {str(rmd_render['stderr']).strip()}"
        )
    rmd_path_resolved = Path(str(rmd_render["output_path"]))
    if not rmd_path_resolved.exists():
        raise RuntimeError(f"REDLine RMD validation render did not create expected file: {rmd_path_resolved}")

    diff_baseline_vs_flags = _compute_image_difference_metrics(baseline_path, direct_path)
    diff_baseline_vs_rmd = _compute_image_difference_metrics(baseline_path, rmd_path_resolved)
    diff_flag_vs_rmd = _compute_image_difference_metrics(direct_path, rmd_path_resolved)
    return {
        "clip_id": clip_id,
        "strategy": strategy_key,
        "variant": variant,
        "exposure_offset": float(exposure_stops),
        "rgb_gains": rgb_gains,
        "rmd_path": rmd_path,
        "direct_flag_command": direct_render["command"],
        "rmd_command": rmd_render["command"],
        "direct_flag_output": str(direct_path),
        "rmd_output": str(rmd_path_resolved),
        "pixel_diff_baseline_vs_flags": diff_baseline_vs_flags["mean_absolute_difference"],
        "pixel_diff_baseline_vs_rmd": diff_baseline_vs_rmd["mean_absolute_difference"],
        "pixel_diff_flag_vs_rmd": diff_flag_vs_rmd["mean_absolute_difference"],
        "max_diff_flag_vs_rmd": diff_flag_vs_rmd["max_absolute_difference"],
        "rmd_pipeline_valid": bool(
            diff_flag_vs_rmd["mean_absolute_difference"] is not None
            and float(diff_flag_vs_rmd["mean_absolute_difference"]) < 1e-3
        ),
        "direct_flag_returncode": int(direct_render["returncode"]),
        "rmd_returncode": int(rmd_render["returncode"]),
    }


def _write_closed_loop_exposure_rmd(
    *,
    clip_id: str,
    source_path: str,
    exposure_stops: float,
    target_dir: Path,
) -> Dict[str, object]:
    review_sidecar = {
        "clip_id": clip_id,
        "source_path": source_path,
        "schema": "r3dmatch_v2",
        "calibration_state": {
            "exposure_calibration_loaded": True,
            "exposure_baseline_applied_stops": float(exposure_stops),
            "color_calibration_loaded": False,
            "rgb_neutral_gains": None,
            "color_gains_state": "review",
        },
        "rmd_mapping": {
            "exposure": {
                "final_offset_stops": float(exposure_stops),
            },
            "color": {
                "rgb_neutral_gains": None,
                "cdl": None,
                "cdl_enabled": False,
            },
        },
    }
    rmd_path, metadata = write_rmd_for_clip_with_metadata(clip_id, review_sidecar, target_dir)
    return {
        "rmd_path": str(rmd_path),
        "metadata": metadata,
    }


def generate_preview_stills(
    input_path: str,
    *,
    analysis_records: List[Dict[str, object]],
    previews_dir: str,
    preview_settings: Dict[str, object],
    redline_capabilities: Dict[str, object],
    strategy_payloads: Optional[List[Dict[str, object]]] = None,
    run_id: Optional[str] = None,
    strategy_rmd_root: Optional[str] = None,
    render_originals: bool = True,
    original_preview_by_clip: Optional[Dict[str, str]] = None,
) -> Dict[str, Dict[str, object]]:
    raise_if_cancelled("Run cancelled before preview generation.")
    preview_root = Path(previews_dir).expanduser().resolve()
    preview_root.mkdir(parents=True, exist_ok=True)
    sidecars = _load_sidecar_map(input_path)
    redline_executable = _resolve_redline_executable()
    preview_paths: Dict[str, Dict[str, object]] = {}
    command_records: List[Dict[str, object]] = []
    rmd_validation_records: List[Dict[str, object]] = []
    selected_rmd_use_mode = 1
    color_preview_policy = _color_preview_policy()
    render_settings = _normalize_preview_settings(
        preview_mode=str(preview_settings.get("preview_mode") or DEFAULT_DISPLAY_REVIEW_PREVIEW["preview_mode"]),
        preview_output_space=str(preview_settings.get("output_space") or DEFAULT_DISPLAY_REVIEW_PREVIEW["output_space"]),
        preview_output_gamma=str(preview_settings.get("output_gamma") or DEFAULT_DISPLAY_REVIEW_PREVIEW["output_gamma"]),
        preview_highlight_rolloff=str(preview_settings.get("highlight_rolloff") or DEFAULT_DISPLAY_REVIEW_PREVIEW["highlight_rolloff"]),
        preview_shadow_rolloff=str(preview_settings.get("shadow_rolloff") or DEFAULT_DISPLAY_REVIEW_PREVIEW["shadow_rolloff"]),
        preview_lut=str(preview_settings.get("lut_path")) if preview_settings.get("lut_path") else None,
    )

    resolved_strategy_payloads = strategy_payloads
    if resolved_strategy_payloads is None:
        resolved_strategy_payloads = []
        for record in analysis_records:
            clip_id = str(record["clip_id"])
            sidecar_payload = sidecars.get(clip_id, {})
            corrections = _extract_preview_corrections(sidecar_payload)
            resolved_strategy_payloads.append(
                {
                    "strategy_key": "median",
                    "strategy_label": strategy_display_name("median"),
                    "reference_clip_id": None,
                    "target_log2_luminance": None,
                    "target_rgb_chromaticity": None,
                    "clips": [
                        {
                            "clip_id": clip_id,
                            "exposure_offset_stops": float(corrections["exposure_stops"]),
                            "rgb_gains": corrections["rgb_gains"],
                            "color_cdl": corrections["cdl"],
                        }
                    ],
                }
            )

    for record in analysis_records:
        raise_if_cancelled("Run cancelled while rendering preview stills.")
        clip_id = str(record["clip_id"])
        source_path = str(record["source_path"])
        sample_plan = dict(record.get("sample_plan", {}))
        frame_index = int(sample_plan.get("start_frame", 0))
        preview_paths[clip_id] = {"strategies": {}}

        original_path = preview_root / preview_filename_for_clip_id(clip_id, "original", run_id=run_id)
        if render_originals:
            baseline_render = render_preview_frame(
                source_path,
                str(original_path),
                frame_index=frame_index,
                redline_executable=redline_executable,
                redline_capabilities=redline_capabilities,
                preview_settings=render_settings,
                use_as_shot_metadata=True,
            )
            if int(baseline_render["returncode"]) != 0:
                raise RuntimeError(
                    f"REDLine preview render failed for {clip_id} (original). "
                    f"Command: {shlex.join(baseline_render['command'])}. STDERR: {str(baseline_render['stderr']).strip()}"
                )
            original_path = Path(str(baseline_render["output_path"]))
            if not original_path.exists():
                raise RuntimeError(
                    f"REDLine preview render did not create expected file for {clip_id} (original): {original_path}"
                )
            command_records.append(
                {
                    "clip_id": clip_id,
                    "variant": "original",
                    "mode": "baseline",
                    "strategy": None,
                    "exposure": None,
                    "redGain": None,
                    "greenGain": None,
                    "blueGain": None,
                    "command": baseline_render["command"],
                    "output": str(original_path),
                    "returncode": baseline_render["returncode"],
                    "stdout": baseline_render["stdout"],
                    "stderr": baseline_render["stderr"],
                    "pixel_diff_from_baseline": 0.0,
                    "pixel_output_changed": False,
                    "as_shot_metadata_used": True,
                    "explicit_transform_used": True,
                    "explicit_correction_flags_used": False,
                }
            )
            print(f"[r3dmatch] preview render clip={clip_id} mode=baseline output={original_path}")
        else:
            reused_original = str((original_preview_by_clip or {}).get(clip_id) or "")
            if reused_original:
                original_path = Path(reused_original)
        preview_paths[clip_id]["original"] = str(original_path) if original_path.exists() else None

        for strategy_payload in resolved_strategy_payloads:
            raise_if_cancelled("Run cancelled while rendering corrected previews.")
            strategy_key = str(strategy_payload["strategy_key"])
            clip_entry = next((item for item in strategy_payload["clips"] if item["clip_id"] == clip_id), None)
            if clip_entry is None:
                continue
            exposure_stops = float(clip_entry["exposure_offset_stops"])
            rgb_gains = clip_entry.get("rgb_gains")
            variants = {
                "exposure": {"exposure": exposure_stops, "gains": None},
                "color": {"exposure": 0.0, "gains": rgb_gains},
                "both": {"exposure": exposure_stops, "gains": rgb_gains},
            }
            preview_paths[clip_id]["strategies"][strategy_key] = {}
            for variant, variant_settings in variants.items():
                raise_if_cancelled("Run cancelled while rendering corrected previews.")
                preview_path = preview_root / preview_filename_for_clip_id(clip_id, variant, strategy=strategy_key, run_id=run_id)
                look_metadata_path = None
                rmd_metadata = None
                if strategy_rmd_root is not None:
                    review_sidecar = {
                        "clip_id": clip_id,
                        "source_path": source_path,
                        "schema": "r3dmatch_v2",
                        "calibration_state": {
                            "exposure_calibration_loaded": variant in {"exposure", "both"},
                            "exposure_baseline_applied_stops": float(variant_settings["exposure"]) if variant in {"exposure", "both"} else 0.0,
                            "color_calibration_loaded": variant in {"color", "both"} and variant_settings["gains"] is not None,
                            "rgb_neutral_gains": (
                                {
                                    "r": float(variant_settings["gains"][0]),
                                    "g": float(variant_settings["gains"][1]),
                                    "b": float(variant_settings["gains"][2]),
                                }
                                if variant_settings["gains"] is not None and variant in {"color", "both"}
                                else None
                            ),
                            "color_gains_state": "review",
                        },
                        "rmd_mapping": {
                            "exposure": {
                                "final_offset_stops": float(variant_settings["exposure"]) if variant in {"exposure", "both"} else 0.0,
                            },
                            "color": {
                                "rgb_neutral_gains": variant_settings["gains"] if variant in {"color", "both"} else None,
                                "cdl": clip_entry.get("color_cdl") if variant in {"color", "both"} else None,
                                "cdl_enabled": bool(
                                    variant in {"color", "both"} and not _is_identity_cdl_payload(clip_entry.get("color_cdl"))
                                ),
                            },
                        },
                    }
                    review_rmd_dir = Path(strategy_rmd_root).expanduser().resolve() / strategy_key / variant
                    rmd_path_obj, rmd_metadata = write_rmd_for_clip_with_metadata(clip_id, review_sidecar, review_rmd_dir)
                    look_metadata_path = str(rmd_path_obj)
                    if str(rmd_metadata.get("rmd_kind")) != "red_sdk":
                        raise RuntimeError(
                            f"RMD pipeline not valid for {clip_id} ({variant}): generated fallback XML instead of RED SDK RMD. "
                            f"Error: {rmd_metadata.get('error')}"
                        )
                    if not redline_capabilities.get("supports_load_rmd"):
                        raise RuntimeError(
                            f"RMD pipeline not valid for {clip_id} ({variant}): this REDLine build does not support --loadRMD/--useRMD."
                        )

                red_gain = green_gain = blue_gain = None
                if variant_settings["gains"] is not None:
                    red_gain = float(variant_settings["gains"][0])
                    green_gain = float(variant_settings["gains"][1])
                    blue_gain = float(variant_settings["gains"][2])
                color_cdl = clip_entry.get("color_cdl") if variant in {"color", "both"} else None
                cdl_enabled = bool(
                    (isinstance(rmd_metadata, dict) and rmd_metadata.get("cdl_enabled"))
                    or (
                        isinstance(rmd_metadata, dict)
                        and isinstance(rmd_metadata.get("settings"), dict)
                        and rmd_metadata["settings"].get("cdl_enabled")
                    )
                )
                color_preview_disabled = variant in {"color", "both"} and not bool(color_preview_policy["enabled"])
                output_reused_from_variant = None
                corrected_render = None
                commit_values = dict(clip_entry.get("commit_values") or {})
                direct_kelvin = None
                direct_tint = None
                if variant in {"color", "both"} and not color_preview_disabled:
                    if commit_values.get("kelvin") is not None:
                        direct_kelvin = int(commit_values.get("kelvin"))
                    if commit_values.get("tint") is not None:
                        direct_tint = float(commit_values.get("tint"))
                if color_preview_disabled:
                    if variant == "color":
                        source_preview_path = original_path
                        output_reused_from_variant = "original"
                    else:
                        exposure_preview = preview_paths[clip_id]["strategies"][strategy_key].get("exposure")
                        source_preview_path = Path(str(exposure_preview)) if exposure_preview else original_path
                        output_reused_from_variant = "exposure"
                    shutil.copyfile(source_preview_path, preview_path)
                    preview_path = preview_path.resolve()
                else:
                    corrected_render = render_preview_frame(
                        source_path,
                        str(preview_path),
                        frame_index=frame_index,
                        redline_executable=redline_executable,
                        redline_capabilities=redline_capabilities,
                        preview_settings=render_settings,
                        use_as_shot_metadata=True,
                        exposure=float(variant_settings["exposure"]) if variant in {"exposure", "both"} else None,
                        kelvin=direct_kelvin,
                        tint=direct_tint,
                        red_gain=None,
                        green_gain=None,
                        blue_gain=None,
                        rmd_path=None,
                        use_rmd_mode=selected_rmd_use_mode,
                    )
                    if int(corrected_render["returncode"]) != 0:
                        raise RuntimeError(
                            f"REDLine preview render failed for {clip_id} ({variant}). "
                            f"Command: {shlex.join(corrected_render['command'])}. STDERR: {str(corrected_render['stderr']).strip()}"
                        )
                    preview_path = Path(str(corrected_render["output_path"]))
                    if not preview_path.exists():
                        raise RuntimeError(
                            f"REDLine preview render did not create expected file for {clip_id} ({variant}): {preview_path}"
                        )

                diff_metrics = _compute_image_difference_metrics(original_path, preview_path)
                mean_diff = diff_metrics["mean_absolute_difference"]
                correction_payload_identity = bool(
                    abs(float(variant_settings["exposure"])) <= 1e-6
                    and _is_identity_rgb_gains(variant_settings["gains"])
                    and _is_identity_cdl_payload(color_cdl)
                )
                requires_change = abs(float(variant_settings["exposure"])) > VISIBLE_PREVIEW_EXPOSURE_DELTA_STOPS or (
                    variant_settings["gains"] is not None
                    and any(abs(float(value) - 1.0) > 1e-6 for value in variant_settings["gains"])
                ) or not _is_identity_cdl_payload(color_cdl)
                error_message = None
                if requires_change and not color_preview_disabled and (mean_diff is None or float(mean_diff) < 1e-3):
                    error_message = "Direct REDLine correction did not change rendered pixels"
                    print(
                        f"[r3dmatch] ERROR: {error_message} clip={clip_id} strategy={strategy_key} "
                        f"variant={variant} command={shlex.join(corrected_render['command']) if corrected_render else 'preview-copy'}"
                    )
                    raise RuntimeError(
                        f"Corrected preview render did not visibly change pixels for {clip_id} "
                        f"({strategy_key}/{variant}) even though non-identity corrections were requested. "
                        f"Command: {shlex.join(corrected_render['command']) if corrected_render else 'preview-copy'}"
                    )
                if color_preview_disabled:
                    error_message = None
                validation_record = {
                    "clip_id": clip_id,
                    "strategy": strategy_key,
                    "variant": variant,
                    "application_method": "preview_color_disabled" if color_preview_disabled else "direct_redline_flags",
                    "rmd_path": look_metadata_path,
                    "use_rmd_mode": selected_rmd_use_mode,
                    "exposure_offset": float(variant_settings["exposure"]),
                    "kelvin": direct_kelvin,
                    "tint": direct_tint,
                    "direct_exposure_parameter": REDLINE_DIRECT_EXPOSURE_PARAMETER if variant in {"exposure", "both"} else None,
                    "rgb_gains": [red_gain, green_gain, blue_gain] if red_gain is not None else None,
                    "color_cdl": color_cdl,
                    "cdl_enabled": cdl_enabled,
                    "preview_color_applied": not color_preview_disabled,
                    "preview_disabled_reason": color_preview_policy["note"] if color_preview_disabled else None,
                    "output_reused_from_variant": output_reused_from_variant,
                    "correction_payload_identity": correction_payload_identity,
                    "pixel_diff_from_baseline": mean_diff,
                    "max_pixel_diff_from_baseline": diff_metrics["max_absolute_difference"],
                    "pixel_output_changed": diff_metrics["pixel_output_changed"],
                    "validation_method": "pixel_diff_from_baseline" if not color_preview_disabled else "preview_fallback_copy",
                    "error": error_message,
                }
                rmd_validation_records.append(validation_record)
                print(
                    f"[r3dmatch] preview render clip={clip_id} mode=corrected strategy={strategy_key} "
                    f"variant={variant} exposure={float(variant_settings['exposure']):.6f} "
                    f"redGain={red_gain} greenGain={green_gain} blueGain={blue_gain} pixel_diff={mean_diff} "
                    f"application_method={'preview_color_disabled' if color_preview_disabled else 'direct_redline_flags'}"
                )
                command_records.append(
                    {
                        "clip_id": clip_id,
                        "variant": variant,
                        "mode": "corrected",
                        "strategy": strategy_key,
                        "exposure": float(variant_settings["exposure"]),
                        "redGain": red_gain,
                        "greenGain": green_gain,
                        "blueGain": blue_gain,
                        "command": corrected_render["command"] if corrected_render else None,
                        "output": str(preview_path),
                        "look_metadata_path": look_metadata_path,
                        "rmd_path": look_metadata_path,
                        "rmd_metadata": rmd_metadata,
                        "returncode": corrected_render["returncode"] if corrected_render else 0,
                        "stdout": corrected_render["stdout"] if corrected_render else "",
                        "stderr": corrected_render["stderr"] if corrected_render else "",
                        "application_method": "preview_color_disabled" if color_preview_disabled else "direct_redline_flags",
                        "kelvin": direct_kelvin,
                        "tint": direct_tint,
                        "direct_exposure_parameter": REDLINE_DIRECT_EXPOSURE_PARAMETER if variant in {"exposure", "both"} else None,
                        "pixel_diff_from_baseline": mean_diff,
                        "max_pixel_diff_from_baseline": diff_metrics["max_absolute_difference"],
                        "pixel_output_changed": diff_metrics["pixel_output_changed"],
                        "correction_payload_identity": correction_payload_identity,
                        "cdl_enabled": cdl_enabled,
                        "preview_color_applied": not color_preview_disabled,
                        "preview_disabled_reason": color_preview_policy["note"] if color_preview_disabled else None,
                        "output_reused_from_variant": output_reused_from_variant,
                        "color_payload_summary": color_cdl,
                        "color_model_summary": clip_entry.get("color_lggs") if variant in {"color", "both"} else None,
                        "error": error_message,
                        "as_shot_metadata_used": not color_preview_disabled,
                        "explicit_transform_used": True,
                        "explicit_correction_flags_used": not color_preview_disabled,
                        "correction_application_method": "preview_color_disabled" if color_preview_disabled else "direct_redline_flags",
                        "use_rmd_mode": selected_rmd_use_mode,
                        "validation_method": "pixel_diff_from_baseline" if not color_preview_disabled else "preview_fallback_copy",
                    }
                )
                preview_paths[clip_id]["strategies"][strategy_key][variant] = str(preview_path)
    (preview_root / "preview_commands.json").write_text(
        json.dumps({"color_preview_policy": color_preview_policy, "commands": command_records}, indent=2),
        encoding="utf-8",
    )
    semantics_payload = {
        "selected_use_rmd_mode": selected_rmd_use_mode,
        "chosen_preview_correction_path": "direct_redline_flags",
        "direct_exposure_parameter": REDLINE_DIRECT_EXPOSURE_PARAMETER,
        "color_preview_policy": color_preview_policy,
        "pipeline_valid": all(
            bool(item["correction_payload_identity"])
            or bool(item["pixel_output_changed"])
            or not bool(item.get("preview_color_applied", True))
            for item in rmd_validation_records
        )
        if rmd_validation_records
        else None,
        "validation_method": "pixel_diff_from_baseline",
        "validations": rmd_validation_records,
    }
    (preview_root / "rmd_validation.json").write_text(
        json.dumps(semantics_payload, indent=2),
        encoding="utf-8",
    )
    (preview_root / "preview_semantics.json").write_text(
        json.dumps(semantics_payload, indent=2),
        encoding="utf-8",
    )
    return preview_paths


def build_contact_sheet_report(
    input_path: str,
    *,
    out_dir: str,
    source_mode: str = "local_folder",
    source_mode_label_value: Optional[str] = None,
    source_input_path: Optional[str] = None,
    ingest_manifest: Optional[Dict[str, object]] = None,
    exposure_calibration_path: Optional[str] = None,
    color_calibration_path: Optional[str] = None,
    target_type: Optional[str] = None,
    processing_mode: Optional[str] = None,
    run_label: Optional[str] = None,
    matching_domain: str = "scene",
    selected_clip_ids: Optional[List[str]] = None,
    selected_clip_groups: Optional[List[str]] = None,
    preview_mode: str = "monitoring",
    preview_output_space: Optional[str] = None,
    preview_output_gamma: Optional[str] = None,
    preview_highlight_rolloff: Optional[str] = None,
    preview_shadow_rolloff: Optional[str] = None,
    preview_lut: Optional[str] = None,
    clear_cache: bool = True,
    calibration_roi: Optional[Dict[str, float]] = None,
    target_strategies: Optional[List[str]] = None,
    reference_clip_id: Optional[str] = None,
    hero_clip_id: Optional[str] = None,
    exposure_anchor_mode: Optional[str] = None,
    manual_target_stops: Optional[float] = None,
    manual_target_ire: Optional[float] = None,
    require_real_redline: bool = False,
    progress_path: Optional[str] = None,
) -> Dict[str, object]:
    report_started_at = time.perf_counter()
    raise_if_cancelled("Run cancelled before report generation.")
    root = Path(input_path).expanduser().resolve()
    out_root = Path(out_dir).expanduser().resolve()
    analysis_records = _load_analysis_records(input_path)
    reusable_analysis_measurements = all(
        str((record.get("diagnostics", {}) or {}).get("exposure_measurement_domain") or "") == "rendered_preview_ipp2"
        and str((record.get("diagnostics", {}) or {}).get("rendered_measurement_preview_path") or "").strip()
        for record in analysis_records
    )
    out_root.mkdir(parents=True, exist_ok=True)
    if clear_cache:
        clear_preview_cache(
            str(root),
            report_dir=str(out_root),
            preserve_measurement_previews=reusable_analysis_measurements,
        )
    emit_review_progress(
        progress_path,
        phase="contact_report_loaded",
        detail=f"Loaded {len(analysis_records)} analysis record(s) for full contact sheet.",
        stage_label="Building report",
        clip_count=len(analysis_records),
        elapsed_seconds=time.perf_counter() - report_started_at,
        review_mode="full_contact_sheet",
    )
    requested_matching_domain = _normalize_matching_domain(matching_domain)
    resolved_matching_domain = "perceptual"
    resolved_source_mode = source_mode
    resolved_source_mode_label = source_mode_label_value or source_mode_label(resolved_source_mode)
    resolved_strategies = _ensure_anchor_strategy_list(
        list(target_strategies or list(DEFAULT_REVIEW_TARGET_STRATEGIES)),
        exposure_anchor_mode=exposure_anchor_mode,
    )
    explicit_anchor_strategy_key = _strategy_key_for_anchor_mode(normalize_exposure_anchor_mode(exposure_anchor_mode))
    resolved_run_label = run_label or root.name or "review"
    run_id = resolved_run_label
    redline_executable = _resolve_redline_executable()
    redline_capabilities = _detect_redline_capabilities(redline_executable)
    real_redline_validation = None
    if require_real_redline:
        emit_review_progress(
            progress_path,
            phase="contact_report_real_redline",
            detail="Checking real REDLine availability and source media.",
            stage_label="Validating REDLine",
            clip_count=len(analysis_records),
            elapsed_seconds=time.perf_counter() - report_started_at,
            review_mode="full_contact_sheet",
        )
        real_redline_validation = _require_real_redline_validation(
            input_path=input_path,
            analysis_records=analysis_records,
            redline_executable=redline_executable,
            out_root=out_root,
            progress_path=progress_path,
        )
    color_preview_policy = _color_preview_policy()
    measurement_preview_settings = _measurement_preview_settings_for_domain(resolved_matching_domain)
    display_preview_settings = _normalize_preview_settings(
        preview_mode=preview_mode,
        preview_output_space=preview_output_space,
        preview_output_gamma=preview_output_gamma,
        preview_highlight_rolloff=preview_highlight_rolloff,
        preview_shadow_rolloff=preview_shadow_rolloff,
        preview_lut=preview_lut,
    )
    exposure = load_exposure_calibration(exposure_calibration_path) if exposure_calibration_path else None
    color = load_color_calibration(color_calibration_path) if color_calibration_path else None
    exposure_by_group = {entry.group_key: entry for entry in exposure.cameras} if exposure else {}
    color_by_group = {entry.group_key: entry for entry in color.cameras} if color else {}
    array_calibration_payload = _load_array_calibration_payload(str(root))
    quality_by_clip = _quality_by_clip(array_calibration_payload)
    monitoring_measurements_by_clip: Dict[str, Dict[str, object]] = {}
    analysis_original_preview_by_clip: Dict[str, str] = {}
    if resolved_matching_domain == "perceptual":
        if reusable_analysis_measurements:
            for record in analysis_records:
                clip_id = str(record["clip_id"])
                reusable_measurement = _rendered_measurement_from_diagnostics(record)
                monitoring_measurements_by_clip[clip_id] = reusable_measurement
                rendered_path = str(reusable_measurement.get("rendered_preview_path") or "")
                if rendered_path:
                    analysis_original_preview_by_clip[clip_id] = rendered_path
        else:
            measurement_preview_paths = generate_preview_stills(
                input_path,
                analysis_records=analysis_records,
                previews_dir=str(root / "previews" / "_measurement"),
                preview_settings=measurement_preview_settings,
                redline_capabilities=redline_capabilities,
                strategy_payloads=[],
                run_id=run_id,
            )
            for record in analysis_records:
                raise_if_cancelled("Run cancelled while measuring rendered originals.")
                clip_id = str(record["clip_id"])
                original_frame = measurement_preview_paths.get(clip_id, {}).get("original")
                if original_frame:
                    measured = _measure_rendered_preview_roi_ipp2(
                        str(original_frame),
                        record.get("diagnostics", {}).get("calibration_roi") or calibration_roi,
                    )
                    monitoring_measurements_by_clip[clip_id] = measured
                    analysis_original_preview_by_clip[clip_id] = str(original_frame)
    strategy_payloads = _build_strategy_payloads(
        analysis_records,
        target_strategies=resolved_strategies,
        reference_clip_id=reference_clip_id,
        hero_clip_id=hero_clip_id,
        target_type=target_type,
        monitoring_measurements_by_clip=monitoring_measurements_by_clip,
        matching_domain=resolved_matching_domain,
        quality_by_clip=quality_by_clip,
        anchor_target_log2=_anchor_target_log2_from_array_payload(array_calibration_payload),
        exposure_anchor_mode=exposure_anchor_mode,
        manual_target_stops=manual_target_stops,
        manual_target_ire=manual_target_ire,
    )
    emit_review_progress(
        progress_path,
        phase="contact_report_strategies",
        detail=f"Built {len(strategy_payloads)} strategy payload(s).",
        stage_label="Building report",
        clip_count=len(analysis_records),
        elapsed_seconds=time.perf_counter() - report_started_at,
        review_mode="full_contact_sheet",
    )
    ipp2_closed_loop = _run_ipp2_closed_loop_solver(
        input_path=input_path,
        out_root=out_root,
        analysis_records=analysis_records,
        strategy_payloads=strategy_payloads,
        redline_capabilities=redline_capabilities,
        run_id=run_id,
        explicit_anchor_strategy_key=explicit_anchor_strategy_key,
        progress_path=progress_path,
        original_preview_by_clip=analysis_original_preview_by_clip,
        original_measurements_by_clip=monitoring_measurements_by_clip,
    )
    strategy_payloads = list(ipp2_closed_loop.get("strategy_payloads") or strategy_payloads)
    emit_review_progress(
        progress_path,
        phase="contact_report_closed_loop",
        detail="Closed-loop IPP2 refinement complete.",
        stage_label="Building report",
        clip_count=len(analysis_records),
        elapsed_seconds=time.perf_counter() - report_started_at,
        review_mode="full_contact_sheet",
    )
    sampling_comparison = _build_sampling_comparison(analysis_records, out_root=out_root)
    solve_comparison = _build_solve_comparison(
        analysis_records,
        target_strategies=resolved_strategies,
        reference_clip_id=reference_clip_id,
        hero_clip_id=hero_clip_id,
        target_type=target_type,
        matching_domain=resolved_matching_domain,
        quality_by_clip=quality_by_clip,
        anchor_target_log2=_anchor_target_log2_from_array_payload(array_calibration_payload),
        exposure_anchor_mode=exposure_anchor_mode,
        manual_target_stops=manual_target_stops,
        manual_target_ire=manual_target_ire,
        out_root=out_root,
    )
    recommended_strategy_key = str((ipp2_closed_loop.get("summary") or {}).get("recommended_strategy_key") or "")
    recommended_strategy = next(
        (item for item in strategy_payloads if str(item.get("strategy_key") or "") == recommended_strategy_key),
        _choose_strategy_with_anchor(strategy_payloads, explicit_anchor_strategy_key=explicit_anchor_strategy_key) if strategy_payloads else None,
    )
    if recommended_strategy is not None:
        closed_loop_strategy = next(
            (item for item in list((ipp2_closed_loop.get("summary") or {}).get("strategies") or []) if str(item.get("strategy_key") or "") == str(recommended_strategy.get("strategy_key") or "")),
            {},
        )
        closed_loop_metrics = dict(closed_loop_strategy.get("metrics") or {})
        recommended_reason = (
            f"{strategy_display_name(str(recommended_strategy.get('strategy_key') or 'median'))} is recommended because it produced the "
            f"lowest IPP2 residual outcome after closed-loop refinement "
            f"({float(closed_loop_metrics.get('median_residual', 0.0) or 0.0):.2f} median / "
            f"{float(closed_loop_metrics.get('max_residual', 0.0) or 0.0):.2f} worst, "
            f"{int((closed_loop_metrics.get('status_counts') or {}).get('PASS', 0) or 0)} pass)."
        )
        if explicit_anchor_strategy_key and str(recommended_strategy.get("strategy_key") or "") == str(explicit_anchor_strategy_key):
            recommended_reason = (
                f"{strategy_display_name(str(recommended_strategy.get('strategy_key') or 'median'))} is being used because the exposure anchor was set explicitly to "
                f"{str(recommended_strategy.get('anchor_summary') or recommended_strategy.get('anchor_source') or 'the selected reference')}."
            )
        recommended_strategy = {
            **dict(recommended_strategy),
            "ipp2_closed_loop": copy.deepcopy(closed_loop_strategy),
            "reason": recommended_reason,
        }
    recommended_strategy_key = str((recommended_strategy or {}).get("strategy_key") or "")
    preview_paths = generate_preview_stills(
        input_path,
        analysis_records=analysis_records,
        previews_dir=str(root / "previews"),
        preview_settings=display_preview_settings,
        redline_capabilities=redline_capabilities,
        strategy_payloads=strategy_payloads,
        run_id=run_id,
        strategy_rmd_root=str(root / "review_rmd" / "strategies"),
        render_originals=False,
        original_preview_by_clip=analysis_original_preview_by_clip,
    )
    for clip_id, original_path in analysis_original_preview_by_clip.items():
        preview_paths.setdefault(str(clip_id), {}).setdefault("strategies", {})
        preview_paths[str(clip_id)]["original"] = str(original_path)
    emit_review_progress(
        progress_path,
        phase="contact_report_previews",
        detail="Preview generation complete.",
        stage_label="Rendering previews",
        clip_count=len(analysis_records),
        elapsed_seconds=time.perf_counter() - report_started_at,
        review_mode="full_contact_sheet",
    )
    preview_manifest_payload = json.loads((root / "previews" / "preview_commands.json").read_text(encoding="utf-8"))
    preview_command_lookup = {
        (
            str(item.get("clip_id") or ""),
            str(item.get("strategy") or ""),
            str(item.get("variant") or ""),
        ): item
        for item in list(preview_manifest_payload.get("commands") or [])
        if isinstance(item, dict)
    }

    shared_originals = []
    for record in analysis_records:
        raise_if_cancelled("Run cancelled while collecting review metadata.")
        clip_id = str(record["clip_id"])
        measurement = _measurement_values_for_record(
            record,
            matching_domain=resolved_matching_domain,
            monitoring_measurements_by_clip=monitoring_measurements_by_clip,
        )
        shared_originals.append(
            {
                "clip_id": clip_id,
                "group_key": str(record["group_key"]),
                "source_path": record.get("source_path"),
                "clip_metadata": record.get("clip_metadata"),
                "original_frame": preview_paths.get(clip_id, {}).get("original"),
                "display_scalar_log2": float(measurement.get("display_scalar_log2", measurement["log2_luminance"]) or 0.0),
                "display_scalar_domain": "display_ipp2" if resolved_matching_domain == "perceptual" else "scene_analysis",
                "measured_log2_luminance": measurement["log2_luminance"],
                "measured_log2_luminance_monitoring": monitoring_measurements_by_clip.get(clip_id, {}).get(
                    "measured_log2_luminance_monitoring",
                    record.get("diagnostics", {}).get("measured_log2_luminance_monitoring"),
                ),
                "measured_log2_luminance_raw": record.get("diagnostics", {}).get("measured_log2_luminance_raw"),
                "gray_exposure_summary": str(record.get("diagnostics", {}).get("gray_exposure_summary") or record.get("diagnostics", {}).get("aggregate_sphere_profile") or "n/a"),
                "sample_1_ire": float(record.get("diagnostics", {}).get("sample_1_ire", record.get("diagnostics", {}).get("bright_ire", 0.0)) or 0.0),
                "sample_2_ire": float(record.get("diagnostics", {}).get("sample_2_ire", record.get("diagnostics", {}).get("center_ire", 0.0)) or 0.0),
                "sample_3_ire": float(record.get("diagnostics", {}).get("sample_3_ire", record.get("diagnostics", {}).get("dark_ire", 0.0)) or 0.0),
                "top_ire": float(record.get("diagnostics", {}).get("top_ire", 0.0) or 0.0),
                "mid_ire": float(record.get("diagnostics", {}).get("mid_ire", 0.0) or 0.0),
                "bottom_ire": float(record.get("diagnostics", {}).get("bottom_ire", 0.0) or 0.0),
                "confidence": record.get("confidence"),
            }
        )

    strategies = []
    for strategy_payload in strategy_payloads:
        raise_if_cancelled("Run cancelled while assembling strategy report payloads.")
        strategy_exposure_summary = _exposure_summary(strategy_payload["clips"])
        selection_diagnostics = dict(strategy_payload.get("selection_diagnostics") or {})
        primary_cluster_indices = {
            int(index) for index in ((selection_diagnostics.get("primary_cluster") or {}).get("indices") or [])
        }
        screened_candidates_by_clip = {
            str(item.get("clip_id") or ""): dict(item)
            for item in list(selection_diagnostics.get("screened_candidates") or [])
            if str(item.get("clip_id") or "").strip()
        }
        strategy_clips = []
        for index, record in enumerate(analysis_records):
            raise_if_cancelled("Run cancelled while assembling strategy report payloads.")
            group_key = str(record["group_key"])
            clip_id = str(record["clip_id"])
            exposure_entry = exposure_by_group.get(group_key)
            color_entry = color_by_group.get(group_key)
            strategy_clip = next(item for item in strategy_payload["clips"] if item["clip_id"] == clip_id)
            trust_details = _camera_trust_details(
                clip_id=clip_id,
                confidence=float(strategy_clip.get("confidence", 0.0) or 0.0),
                sample_log2_spread=float(strategy_clip.get("neutral_sample_log2_spread", 0.0) or 0.0),
                sample_chroma_spread=float(strategy_clip.get("neutral_sample_chromaticity_spread", 0.0) or 0.0),
                measured_log2=float(strategy_clip.get("measured_log2_luminance_monitoring", strategy_clip.get("measured_log2_luminance", 0.0)) or 0.0),
                final_offset=float(strategy_clip.get("exposure_offset_stops", 0.0) or 0.0),
                exposure_summary=strategy_exposure_summary,
                in_primary_cluster=(not primary_cluster_indices or index in primary_cluster_indices),
                screened_reasons=[str(item) for item in (screened_candidates_by_clip.get(clip_id, {}) or {}).get("reasons", []) if str(item).strip()],
            )
            strategy_clips.append(
                {
                    "clip_id": clip_id,
                    "group_key": group_key,
                    "source_path": record.get("source_path"),
                    "original_frame": preview_paths.get(clip_id, {}).get("original"),
                    "exposure_corrected": preview_paths.get(clip_id, {}).get("strategies", {}).get(strategy_payload["strategy_key"], {}).get("exposure"),
                    "color_corrected": preview_paths.get(clip_id, {}).get("strategies", {}).get(strategy_payload["strategy_key"], {}).get("color"),
                    "both_corrected": preview_paths.get(clip_id, {}).get("strategies", {}).get(strategy_payload["strategy_key"], {}).get("both"),
                    "preview_variants": {
                        "original": preview_paths.get(clip_id, {}).get("original"),
                        "exposure": preview_paths.get(clip_id, {}).get("strategies", {}).get(strategy_payload["strategy_key"], {}).get("exposure"),
                        "color": preview_paths.get(clip_id, {}).get("strategies", {}).get(strategy_payload["strategy_key"], {}).get("color"),
                        "both": preview_paths.get(clip_id, {}).get("strategies", {}).get(strategy_payload["strategy_key"], {}).get("both"),
                    },
                    "render_validation": dict(
                        preview_command_lookup.get((clip_id, str(strategy_payload["strategy_key"]), "both")) or {}
                    ),
                    "calibration_roi": strategy_clip["calibration_roi"],
                    "is_hero_camera": strategy_clip["is_hero_camera"],
                    "trust_class": str(trust_details["trust_class"]),
                    "trust_score": float(trust_details["trust_score"]),
                    "trust_reason": str(trust_details["trust_reason"]),
                    "stability_label": str(trust_details["stability_label"]),
                    "correction_confidence": str(trust_details["correction_confidence"]),
                    "reference_use": str(trust_details["reference_use"]),
                    "recommended": str(strategy_payload["strategy_key"]) == recommended_strategy_key,
                    "anchor_mode": str(strategy_payload.get("anchor_mode") or ""),
                    "anchor_source": str(strategy_payload.get("anchor_source") or ""),
                    "anchor_scalar_value": float(strategy_payload.get("anchor_scalar_value", 0.0) or 0.0),
                    "anchor_ire_summary": str(strategy_payload.get("anchor_ire_summary") or "n/a"),
                    "metrics": {
                        "exposure": {
                            "raw_offset_stops": record.get("raw_offset_stops"),
                            "initial_offset_stops": strategy_clip.get("initial_exposure_offset_stops", strategy_clip["exposure_offset_stops"]),
                            "final_offset_stops": strategy_clip["exposure_offset_stops"],
                            "pre_residual_stops": strategy_clip.get("pre_exposure_residual_stops"),
                            "post_residual_stops": strategy_clip.get("post_exposure_residual_stops"),
                            "measured_log2_luminance": strategy_clip["measured_log2_luminance"],
                            "measured_log2_luminance_monitoring": strategy_clip["measured_log2_luminance_monitoring"],
                            "measured_log2_luminance_raw": strategy_clip["measured_log2_luminance_raw"],
                            "gray_exposure_summary": str(strategy_clip.get("gray_exposure_summary") or strategy_clip.get("aggregate_sphere_profile") or "n/a"),
                            "sample_1_ire": float(strategy_clip.get("sample_1_ire", strategy_clip.get("bright_ire", 0.0)) or 0.0),
                            "sample_2_ire": float(strategy_clip.get("sample_2_ire", strategy_clip.get("center_ire", 0.0)) or 0.0),
                            "sample_3_ire": float(strategy_clip.get("sample_3_ire", strategy_clip.get("dark_ire", 0.0)) or 0.0),
                            "top_ire": float(strategy_clip.get("top_ire", 0.0) or 0.0),
                            "mid_ire": float(strategy_clip.get("mid_ire", 0.0) or 0.0),
                            "bottom_ire": float(strategy_clip.get("bottom_ire", 0.0) or 0.0),
                            "zone_spread_ire": float(strategy_clip.get("zone_spread_ire", 0.0) or 0.0),
                            "zone_spread_stops": float(strategy_clip.get("zone_spread_stops", 0.0) or 0.0),
                            "sphere_zone_profile_monitoring": copy.deepcopy(strategy_clip.get("sphere_zone_profile_monitoring") or []),
                            "measurement_domain": strategy_clip["exposure_measurement_domain"],
                            "ipp2_closed_loop_target_log2": strategy_clip.get("ipp2_closed_loop_target_log2"),
                            "ipp2_closed_loop_initial_residual_stops": strategy_clip.get("ipp2_closed_loop_initial_residual_stops"),
                            "ipp2_closed_loop_final_residual_stops": strategy_clip.get("ipp2_closed_loop_final_residual_stops"),
                            "ipp2_closed_loop_iterations": strategy_clip.get("ipp2_closed_loop_iterations"),
                            "camera_offset_from_anchor": float(strategy_clip.get("camera_offset_from_anchor", strategy_clip.get("exposure_offset_stops", 0.0)) or 0.0),
                            "anchor_mode": str(strategy_clip.get("anchor_mode") or strategy_payload.get("anchor_mode") or ""),
                            "anchor_source": str(strategy_clip.get("anchor_source") or strategy_payload.get("anchor_source") or ""),
                            "anchor_scalar_value": float(strategy_clip.get("anchor_scalar_value", strategy_payload.get("anchor_scalar_value", 0.0)) or 0.0),
                            "anchor_ire_summary": str(strategy_clip.get("anchor_ire_summary") or strategy_payload.get("anchor_ire_summary") or "n/a"),
                        },
                        "color": {
                            "rgb_gains": strategy_clip["rgb_gains"],
                            "rgb_gains_diagnostic": strategy_clip["rgb_gains"],
                            "cdl": strategy_clip["color_cdl"],
                            "lift_gamma_gain_saturation": strategy_clip["color_lggs"],
                            "saturation": strategy_clip["color_lggs"]["saturation"],
                            "saturation_source": strategy_clip["saturation_source"],
                            "measured_saturation_fraction": strategy_clip["measured_saturation_fraction"],
                            "target_saturation_fraction": strategy_clip["target_saturation_fraction"],
                            "measurement_domain": strategy_clip["color_measurement_domain"],
                            "measured_channel_medians": color_entry.measured_channel_medians if color_entry else None,
                            "pre_residual": strategy_clip.get("pre_color_residual"),
                            "post_residual": strategy_clip.get("post_color_residual"),
                            "saturation_supported": strategy_clip.get("saturation_supported"),
                        },
                        "sampling_mode": (
                            exposure_entry.sampling_mode if exposure_entry
                            else color_entry.sampling_mode if color_entry
                            else None
                        ),
                        "confidence": strategy_clip["confidence"],
                        "flags": strategy_clip["flags"],
                        "measurement_mode": strategy_clip["measurement_mode"],
                        "neutral_sample_count": strategy_clip.get("neutral_sample_count"),
                        "neutral_sample_log2_spread": strategy_clip.get("neutral_sample_log2_spread"),
                        "neutral_sample_chromaticity_spread": strategy_clip.get("neutral_sample_chromaticity_spread"),
                        "neutral_samples": strategy_clip.get("neutral_samples"),
                        "commit_values": strategy_clip.get("commit_values"),
                        "preview_transform": _preview_transform_label(display_preview_settings),
                        "color_preview_applied": bool(color_preview_policy["enabled"]),
                        "color_preview_status": str(color_preview_policy["status"]),
                        "color_preview_note": color_preview_policy["note"],
                        "is_hero_camera": strategy_clip["is_hero_camera"],
                    },
                }
            )
        strategies.append(
            {
                "strategy_key": strategy_payload["strategy_key"],
                "strategy_label": strategy_payload["strategy_label"],
                "reference_clip_id": strategy_payload["reference_clip_id"],
                "hero_clip_id": strategy_payload.get("hero_clip_id"),
                "target_log2_luminance": strategy_payload["target_log2_luminance"],
                "target_rgb_chromaticity": strategy_payload["target_rgb_chromaticity"],
                "target_saturation_fraction": strategy_payload.get("target_saturation_fraction"),
                "strategy_summary": strategy_payload.get("strategy_summary"),
                "target_gray_exposure_summary": str(strategy_payload.get("ipp2_closed_loop_target_gray_exposure_summary") or strategy_payload.get("target_gray_exposure_summary") or "n/a"),
                "anchor_mode": str(strategy_payload.get("anchor_mode") or ""),
                "anchor_source": str(strategy_payload.get("anchor_source") or ""),
                "anchor_scalar_value": float(strategy_payload.get("anchor_scalar_value", 0.0) or 0.0),
                "anchor_ire_summary": str(strategy_payload.get("anchor_ire_summary") or "n/a"),
                "anchor_summary": str(strategy_payload.get("anchor_summary") or ""),
                "calibration_roi": calibration_roi or (analysis_records[0].get("diagnostics", {}).get("calibration_roi") if analysis_records else None),
                "recommended": str(strategy_payload["strategy_key"]) == recommended_strategy_key,
                "clips": strategy_clips,
            }
        )
    corrected_validation_rows = [
        item
        for item in list(preview_manifest_payload.get("commands") or [])
        if isinstance(item, dict) and str(item.get("mode") or "") == "corrected" and str(item.get("variant") or "") == "both"
    ]
    payload_render_truth = {
        "corrected_render_count": len(corrected_validation_rows),
        "changed_render_count": sum(1 for item in corrected_validation_rows if bool(item.get("pixel_output_changed"))),
        "unchanged_non_identity_count": sum(
            1
            for item in corrected_validation_rows
            if not bool(item.get("pixel_output_changed")) and not bool(item.get("correction_payload_identity"))
        ),
        "color_preview_policy": preview_manifest_payload.get("color_preview_policy"),
    }
    corrected_residual_validation = _build_corrected_residual_validation(
        strategies=strategies,
        out_root=out_root,
    )
    ipp2_validation = _build_ipp2_validation(
        input_path=input_path,
        out_root=out_root,
        analysis_records=analysis_records,
        strategy_payloads=strategy_payloads,
        strategies=strategies,
        redline_capabilities=redline_capabilities,
        run_id=run_id,
        display_preview_settings=display_preview_settings,
        display_preview_paths=preview_paths,
        closed_loop_result=ipp2_closed_loop,
    )
    render_trace_artifacts = _write_render_trace_artifacts(
        out_root=out_root,
        analysis_records=analysis_records,
        strategies=strategies,
        ipp2_validation_summary=ipp2_validation["summary"],
        preview_manifest_payload=preview_manifest_payload,
    )
    sphere_detection_artifacts = _build_sphere_detection_artifacts(
        out_root=out_root,
        validation_summary=ipp2_validation["summary"],
        recommended_strategy_key=recommended_strategy_key,
    )
    residual_lookup = {
        (str(item.get("strategy_key") or ""), str(item.get("clip_id") or "")): dict(item)
        for item in list((corrected_residual_validation.get("summary") or {}).get("rows") or [])
        if str(item.get("strategy_key") or "").strip() and str(item.get("clip_id") or "").strip()
    }
    ipp2_lookup = {
        (str(item.get("strategy_key") or ""), str(item.get("clip_id") or "")): dict(item)
        for item in list((ipp2_validation.get("summary") or {}).get("rows") or [])
        if str(item.get("strategy_key") or "").strip() and str(item.get("clip_id") or "").strip()
    }
    for strategy in strategies:
        for clip in strategy.get("clips", []):
            clip["corrected_residual_validation"] = dict(
                residual_lookup.get((str(strategy.get("strategy_key") or ""), str(clip.get("clip_id") or ""))) or {}
            )
            clip["ipp2_validation"] = dict(
                ipp2_lookup.get((str(strategy.get("strategy_key") or ""), str(clip.get("clip_id") or ""))) or {}
            )
        strategy_target_profile = next(
            (
                str((clip.get("ipp2_validation") or {}).get("ipp2_target_gray_exposure_summary") or "")
                for clip in strategy.get("clips", [])
                if str((clip.get("ipp2_validation") or {}).get("ipp2_target_gray_exposure_summary") or "").strip()
            ),
            str(strategy.get("target_gray_exposure_summary") or "n/a"),
        )
        strategy["target_gray_exposure_summary"] = strategy_target_profile

    payload = {
        "report_kind": "full_contact_sheet",
        "review_mode": "full_contact_sheet",
        "review_mode_label": review_mode_label("full_contact_sheet"),
        "input_path": str(root),
        "source_mode": resolved_source_mode,
        "source_mode_label": resolved_source_mode_label,
        "source_input_path": source_input_path or str(root),
        "ingest_manifest": ingest_manifest,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "summary_path": str(root / "summary.json") if (root / "summary.json").exists() else None,
        "array_calibration_path": str(root / "array_calibration.json") if (root / "array_calibration.json").exists() else None,
        "exposure_calibration_path": str(Path(exposure_calibration_path).expanduser().resolve()) if exposure_calibration_path else None,
        "color_calibration_path": str(Path(color_calibration_path).expanduser().resolve()) if color_calibration_path else None,
        "target_exposure_log2": strategies[0]["target_log2_luminance"] if strategies else None,
        "target_rgb_chromaticity": strategies[0]["target_rgb_chromaticity"] if strategies else None,
        "previews_dir": str(root / "previews"),
        "target_type": target_type or "unspecified",
        "processing_mode": processing_mode or "both",
        "run_label": resolved_run_label,
        "matching_domain": resolved_matching_domain,
        "matching_domain_label": _matching_domain_label(resolved_matching_domain),
        "preview_transform": _preview_transform_label(display_preview_settings),
        "measurement_preview_transform": _preview_transform_label(measurement_preview_settings),
        "color_preview_enabled": bool(color_preview_policy["enabled"]),
        "color_preview_status": str(color_preview_policy["status"]),
        "color_preview_note": color_preview_policy["note"],
        "exposure_measurement_domain": resolved_matching_domain,
        "preview_mode": display_preview_settings["preview_mode"],
        "preview_settings": display_preview_settings,
        "measurement_preview_settings": measurement_preview_settings,
        "redline_capabilities": redline_capabilities,
        "real_redline_validation_required": bool(require_real_redline),
        "real_redline_validation": real_redline_validation["summary"] if real_redline_validation else None,
        "real_redline_validation_path": real_redline_validation["path"] if real_redline_validation else None,
        "calibration_roi": calibration_roi or (analysis_records[0].get("diagnostics", {}).get("calibration_roi") if analysis_records else None),
        "run_id": run_id,
        "selected_clip_ids": [str(item) for item in (selected_clip_ids or []) if str(item).strip()],
        "selected_clip_groups": [str(item) for item in (selected_clip_groups or []) if str(item).strip()],
        "target_strategies": resolved_strategies,
        "reference_clip_id": reference_clip_id,
        "hero_clip_id": hero_clip_id,
        "exposure_anchor_mode": normalize_exposure_anchor_mode(exposure_anchor_mode),
        "exposure_anchor_strategy_key": explicit_anchor_strategy_key or recommended_strategy_key,
        "clip_count": len(analysis_records),
        "shared_originals": shared_originals,
        "strategies": strategies,
        "clips": next((item["clips"] for item in strategies if str(item.get("strategy_key") or "") == recommended_strategy_key), strategies[0]["clips"] if strategies else []),
        "strategy_review_rmd_root": str((root / "review_rmd" / "strategies").resolve()),
        "render_truth_summary": payload_render_truth,
        "sampling_comparison": sampling_comparison["summary"] if sampling_comparison else None,
        "sampling_comparison_path": sampling_comparison["path"] if sampling_comparison else None,
        "solve_comparison": solve_comparison["summary"] if solve_comparison else None,
        "solve_comparison_path": solve_comparison["path"] if solve_comparison else None,
        "corrected_residual_validation": corrected_residual_validation["summary"],
        "corrected_residual_validation_path": corrected_residual_validation["path"],
        "ipp2_closed_loop": ipp2_closed_loop["summary"],
        "ipp2_closed_loop_trace_path": ipp2_closed_loop["path"],
        "ipp2_validation": ipp2_validation["summary"],
        "ipp2_validation_path": ipp2_validation["path"],
        "sphere_detection_summary": sphere_detection_artifacts["summary"],
        "sphere_detection_summary_path": sphere_detection_artifacts["path"],
        "sphere_detection_overlay_root": sphere_detection_artifacts["overlay_root"],
        **render_trace_artifacts,
    }
    strategy_summaries = [
        {
            "strategy_key": item["strategy_key"],
            "strategy_label": item["strategy_label"],
            "reference_clip_id": item.get("reference_clip_id"),
            "hero_clip_id": item.get("hero_clip_id"),
            "anchor_mode": item.get("anchor_mode"),
            "anchor_source": item.get("anchor_source"),
            "anchor_scalar_value": item.get("anchor_scalar_value"),
            "anchor_ire_summary": item.get("anchor_ire_summary"),
            "anchor_summary": item.get("anchor_summary"),
            "target_log2_luminance": float(item["target_log2_luminance"]),
            "target_gray_exposure_summary": next(
                (
                    str(strategy.get("target_gray_exposure_summary") or "n/a")
                    for strategy in strategies
                    if str(strategy.get("strategy_key") or "") == str(item.get("strategy_key") or "")
                ),
                "n/a",
            ),
            "summary": item.get("strategy_summary"),
            "correction_metrics": _strategy_distribution_metrics(item),
        }
        for item in strategy_payloads
    ]
    payload["recommended_strategy"] = recommended_strategy
    payload["hero_recommendation"] = _hero_candidate_summary(strategy_payloads)
    recommended_strategy_section = next(
        (item for item in strategies if str(item.get("strategy_key") or "") == recommended_strategy_key),
        strategies[0] if strategies else None,
    )
    payload["exposure_summary"] = _exposure_summary((recommended_strategy_section or {}).get("clips", []))
    payload["anchor_mode"] = str((recommended_strategy_section or {}).get("anchor_mode") or (recommended_strategy or {}).get("anchor_mode") or "")
    payload["anchor_source"] = str((recommended_strategy_section or {}).get("anchor_source") or (recommended_strategy or {}).get("anchor_source") or "")
    payload["anchor_scalar_value"] = float(
        ((recommended_strategy_section or {}).get("anchor_scalar_value"))
        if (recommended_strategy_section or {}).get("anchor_scalar_value") is not None
        else ((recommended_strategy or {}).get("anchor_scalar_value") or 0.0)
    )
    payload["anchor_ire_summary"] = str((recommended_strategy_section or {}).get("anchor_ire_summary") or (recommended_strategy or {}).get("anchor_ire_summary") or "n/a")
    payload["exposure_anchor_summary"] = str((recommended_strategy_section or {}).get("anchor_summary") or (recommended_strategy or {}).get("anchor_summary") or "")
    payload["strategy_comparison"] = strategy_summaries
    payload["visuals"] = {
        "exposure_plot_svg": _build_exposure_plot_svg((recommended_strategy_section or {}).get("clips", [])),
        "strategy_chart_svg": _build_strategy_chart_svg(strategy_summaries),
    }
    if strategy_payloads:
        recommended_payload = next(
            item for item in strategy_payloads if str(item.get("strategy_key") or "") == str((payload["recommended_strategy"] or {}).get("strategy_key") or "")
        )
        payload["debug_exposure_trace"] = _build_exposure_trace_artifacts(
            analysis_records=analysis_records,
            recommended_payload=recommended_payload,
            strategy_summaries=strategy_summaries,
            exposure_summary=payload["exposure_summary"],
            out_root=out_root,
        )
    payload["executive_synopsis"] = _build_lightweight_synopsis(
        exposure_summary=payload["exposure_summary"],
        strategy_summaries=strategy_summaries,
        recommended_strategy=payload["recommended_strategy"] or {"reason": "No recommendation available."},
        hero_summary=payload["hero_recommendation"],
    ) if strategy_payloads else ""
    json_path = out_root / "contact_sheet.json"
    html_path = out_root / "contact_sheet.html"
    pdf_path = out_root / "preview_contact_sheet.pdf"
    review_manifest_path = out_root / "review_manifest.json"
    try:
        raise_if_cancelled("Run cancelled before writing review report artifacts.")
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        raise_if_cancelled("Run cancelled before writing review HTML.")
        html_path.write_text(render_contact_sheet_html(payload), encoding="utf-8")
        raise_if_cancelled("Run cancelled before rendering review PDF.")
        render_contact_sheet_pdf(payload, output_path=str(pdf_path), title="R3DMatch Review Contact Sheet")
        raise_if_cancelled("Run cancelled before writing review manifest.")
        review_manifest_path.write_text(
            json.dumps(
                {
                    "input_path": str(root),
                    "source_mode": payload["source_mode"],
                    "source_mode_label": payload["source_mode_label"],
                    "source_input_path": payload["source_input_path"],
                    "review_mode": "full_contact_sheet",
                    "review_mode_label": review_mode_label("full_contact_sheet"),
                    "target_type": payload["target_type"],
                    "processing_mode": payload["processing_mode"],
                    "run_label": payload["run_label"],
                    "matching_domain": payload["matching_domain"],
                    "matching_domain_label": payload["matching_domain_label"],
                    "preview_transform": payload["preview_transform"],
                    "measurement_preview_transform": payload["measurement_preview_transform"],
                    "color_preview_enabled": payload["color_preview_enabled"],
                    "color_preview_status": payload["color_preview_status"],
                    "color_preview_note": payload["color_preview_note"],
                    "preview_mode": payload["preview_mode"],
                    "preview_settings": payload["preview_settings"],
                    "measurement_preview_settings": payload["measurement_preview_settings"],
                    "redline_capabilities": payload["redline_capabilities"],
                    "exposure_measurement_domain": payload["exposure_measurement_domain"],
                    "calibration_roi": payload["calibration_roi"],
                    "selected_clip_ids": payload["selected_clip_ids"],
                    "selected_clip_groups": payload["selected_clip_groups"],
                    "target_strategies": payload["target_strategies"],
                    "reference_clip_id": payload["reference_clip_id"],
                    "hero_clip_id": payload["hero_clip_id"],
                    "clip_count": payload["clip_count"],
                    "sphere_detection_summary_path": payload["sphere_detection_summary_path"],
                    "sphere_detection_overlay_root": payload["sphere_detection_overlay_root"],
                    "report_json": str(json_path),
                    "report_html": str(html_path),
                    "preview_report_pdf": str(pdf_path),
                    "previews_dir": payload["previews_dir"],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    except CancellationError:
        for artifact in (json_path, html_path, pdf_path, review_manifest_path):
            if artifact.exists():
                artifact.unlink()
        raise
    return {
        "report_json": str(json_path),
        "report_html": str(html_path),
        "preview_report_pdf": str(pdf_path),
        "previews_dir": str(root / "previews"),
        "review_manifest": str(review_manifest_path),
        "clip_count": len(analysis_records),
        "preview_transform": payload["preview_transform"],
        "measurement_preview_transform": payload["measurement_preview_transform"],
        "run_label": payload["run_label"],
        "matching_domain": payload["matching_domain"],
        "matching_domain_label": payload["matching_domain_label"],
        "review_mode": "full_contact_sheet",
        "review_mode_label": review_mode_label("full_contact_sheet"),
        "preview_mode": payload["preview_mode"],
        "preview_settings": payload["preview_settings"],
        "measurement_preview_settings": payload["measurement_preview_settings"],
        "redline_capabilities": payload["redline_capabilities"],
        "real_redline_validation_required": payload["real_redline_validation_required"],
        "real_redline_validation_path": payload["real_redline_validation_path"],
        "render_input_state_path": payload["render_input_state_path"],
        "pre_render_log_values_path": payload["pre_render_log_values_path"],
        "post_render_ipp2_values_path": payload["post_render_ipp2_values_path"],
        "render_trace_comparison_path": payload["render_trace_comparison_path"],
        "sphere_detection_summary_path": payload["sphere_detection_summary_path"],
        "sphere_detection_overlay_root": payload["sphere_detection_overlay_root"],
    }


def _write_rcx_comparison_placeholder(root: Path) -> str:
    compare_root = root / "rmd_compare"
    compare_root.mkdir(parents=True, exist_ok=True)
    manual_sample_root = compare_root / "manual_rcx_samples"
    manual_sample_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "open_question",
        "question": "SDK-authored RMD vs RCX-authored RMD parity remains an open comparison point.",
        "manual_sample_root": str(manual_sample_root),
        "manual_sample_count": len(list(manual_sample_root.glob("*"))),
        "notes": [
            "Drop any manually exported RCX-authored RMD samples into manual_rcx_samples for side-by-side inspection.",
            "Canonical preview/render correction now uses direct REDLine flags with --useMeta, --exposureAdjust, --kelvin, and --tint.",
            "SDK-authored RMDs are still exported for downstream interoperability, not as the active preview truth path.",
        ],
    }
    note_path = compare_root / "rcx_parity_notes.json"
    note_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return str(note_path)


def build_review_package(
    input_path: str,
    *,
    out_dir: str,
    source_mode: str = "local_folder",
    source_mode_label_value: Optional[str] = None,
    source_input_path: Optional[str] = None,
    ingest_manifest: Optional[Dict[str, object]] = None,
    exposure_calibration_path: Optional[str] = None,
    color_calibration_path: Optional[str] = None,
    target_type: str = "gray_sphere",
    processing_mode: str = "both",
    run_label: Optional[str] = None,
    matching_domain: str = "scene",
    review_mode: str = "full_contact_sheet",
    selected_clip_ids: Optional[List[str]] = None,
    selected_clip_groups: Optional[List[str]] = None,
    preview_mode: str = "monitoring",
    preview_output_space: Optional[str] = None,
    preview_output_gamma: Optional[str] = None,
    preview_highlight_rolloff: Optional[str] = None,
    preview_shadow_rolloff: Optional[str] = None,
    preview_lut: Optional[str] = None,
    calibration_roi: Optional[Dict[str, float]] = None,
    target_strategies: Optional[List[str]] = None,
    reference_clip_id: Optional[str] = None,
    hero_clip_id: Optional[str] = None,
    exposure_anchor_mode: Optional[str] = None,
    manual_target_stops: Optional[float] = None,
    manual_target_ire: Optional[float] = None,
    require_real_redline: bool = False,
    progress_path: Optional[str] = None,
) -> Dict[str, object]:
    raise_if_cancelled("Run cancelled before review package assembly.")
    root = Path(out_dir).expanduser().resolve()
    report_dir = root / "report"
    review_rmd_dir = root / "review_rmd"
    rcx_compare_note = _write_rcx_comparison_placeholder(root)
    resolved_review_mode = normalize_review_mode(review_mode)
    if resolved_review_mode == "lightweight_analysis":
        report_payload = build_lightweight_analysis_report(
            out_dir,
            out_dir=str(report_dir),
            source_mode=source_mode,
            source_mode_label_value=source_mode_label_value,
            source_input_path=source_input_path,
            ingest_manifest=ingest_manifest,
            target_type=target_type,
            processing_mode=processing_mode,
            run_label=run_label,
            matching_domain=matching_domain,
            selected_clip_ids=selected_clip_ids,
            selected_clip_groups=selected_clip_groups,
            calibration_roi=calibration_roi,
            target_strategies=target_strategies,
            reference_clip_id=reference_clip_id,
            hero_clip_id=hero_clip_id,
            exposure_anchor_mode=exposure_anchor_mode,
            manual_target_stops=manual_target_stops,
            manual_target_ire=manual_target_ire,
            clear_cache=False,
            progress_path=progress_path,
        )
    else:
        report_payload = build_contact_sheet_report(
            out_dir,
            out_dir=str(report_dir),
            source_mode=source_mode,
            source_mode_label_value=source_mode_label_value,
            source_input_path=source_input_path,
            ingest_manifest=ingest_manifest,
            exposure_calibration_path=exposure_calibration_path,
            color_calibration_path=color_calibration_path,
            target_type=target_type,
            processing_mode=processing_mode,
            run_label=run_label,
            matching_domain=matching_domain,
            selected_clip_ids=selected_clip_ids,
            selected_clip_groups=selected_clip_groups,
            preview_mode=preview_mode,
            preview_output_space=preview_output_space,
            preview_output_gamma=preview_output_gamma,
            preview_highlight_rolloff=preview_highlight_rolloff,
            preview_shadow_rolloff=preview_shadow_rolloff,
            preview_lut=preview_lut,
            calibration_roi=calibration_roi,
            target_strategies=target_strategies,
            reference_clip_id=reference_clip_id,
            hero_clip_id=hero_clip_id,
            exposure_anchor_mode=exposure_anchor_mode,
            manual_target_stops=manual_target_stops,
            manual_target_ire=manual_target_ire,
            clear_cache=True,
            require_real_redline=require_real_redline,
            progress_path=progress_path,
        )
    if resolved_review_mode == "lightweight_analysis":
        rmd_manifest = {
            "skipped": True,
            "reason": "Lightweight analysis does not require temporary review RMD authoring.",
            "review_rmd_dir": str(review_rmd_dir),
        }
    else:
        raise_if_cancelled("Run cancelled before writing temporary review RMDs.")
        rmd_manifest = write_rmds_from_analysis(out_dir, out_dir=str(review_rmd_dir))
    package_manifest = {
        "workflow_phase": "review",
        "review_mode": resolved_review_mode,
        "review_mode_label": review_mode_label(resolved_review_mode),
        "analysis_dir": str(root),
        "source_mode": source_mode,
        "source_mode_label": source_mode_label_value or source_mode_label(source_mode),
        "source_input_path": source_input_path or str(root),
        "ingest_manifest": ingest_manifest,
        "run_label": run_label or root.name,
        "matching_domain": _normalize_matching_domain(matching_domain),
        "matching_domain_label": _matching_domain_label(matching_domain),
        "selected_clip_ids": [str(item) for item in (selected_clip_ids or []) if str(item).strip()],
        "selected_clip_groups": [str(item) for item in (selected_clip_groups or []) if str(item).strip()],
        "target_type": target_type,
        "processing_mode": processing_mode,
        "preview_transform": report_payload.get("preview_transform"),
        "measurement_preview_transform": report_payload.get("measurement_preview_transform"),
        "preview_mode": preview_mode,
        "preview_settings": report_payload.get("preview_settings"),
        "measurement_preview_settings": report_payload.get("measurement_preview_settings"),
        "redline_capabilities": report_payload.get("redline_capabilities"),
        "exposure_measurement_domain": _normalize_matching_domain(matching_domain),
        "calibration_roi": calibration_roi,
        "target_strategies": target_strategies or list(DEFAULT_REVIEW_TARGET_STRATEGIES),
        "reference_clip_id": reference_clip_id,
        "hero_clip_id": hero_clip_id,
        "exposure_anchor_mode": normalize_exposure_anchor_mode(exposure_anchor_mode),
        "manual_target_stops": manual_target_stops,
        "manual_target_ire": manual_target_ire,
        "review_rmd_dir": str(review_rmd_dir),
        "rcx_compare_note": rcx_compare_note,
        "report_json": report_payload["report_json"],
        "report_html": report_payload["report_html"],
        "preview_report_pdf": report_payload["preview_report_pdf"],
        "review_manifest": report_payload["review_manifest"],
        "clip_count": report_payload["clip_count"],
        "temporary_rmd_manifest": rmd_manifest,
    }
    manifest_path = report_dir / "review_package.json"
    manifest_path.write_text(json.dumps(package_manifest, indent=2), encoding="utf-8")
    package_manifest["package_manifest"] = str(manifest_path)
    return package_manifest


def _contact_sheet_image_src(path: object) -> str:
    path_text = str(path or "").strip()
    if not path_text:
        return ""
    path_obj = Path(path_text)
    if "review_detection_overlays" in path_obj.parts:
        return f"./review_detection_overlays/{path_obj.name}"
    if "_measurement" in path_obj.parts:
        return f"../previews/_measurement/{path_obj.name}"
    if "_ipp2_closed_loop" in path_obj.parts:
        return f"../previews/_ipp2_closed_loop/{path_obj.name}"
    return f"../previews/{path_obj.name}"


def _contact_sheet_display_scalar_ire(log2_value: object, fallback_ire: object = None) -> float:
    if fallback_ire is not None:
        try:
            return float(fallback_ire)
        except (TypeError, ValueError):
            pass
    try:
        return float(_ire_from_log2_luminance(float(log2_value or 0.0)))
    except (TypeError, ValueError):
        return 0.0


def _contact_sheet_status_tone(status: str) -> str:
    return {"PASS": "good", "REVIEW": "warning", "FAIL": "danger"}.get(str(status or "REVIEW"), "warning")


def _contact_sheet_reference_candidate(entries: List[Dict[str, object]]) -> Optional[Dict[str, object]]:
    retained = [item for item in entries if str(item.get("reference_use") or "Included") != "Excluded"]
    if not retained:
        retained = list(entries)
    if not retained:
        return None
    return min(
        retained,
        key=lambda item: (
            abs(float(item.get("offset_to_anchor", 0.0) or 0.0)),
            -float(item.get("trust_score", 0.0) or 0.0),
            str(item.get("camera_label") or ""),
        ),
    )


def _contact_sheet_goal_achieved(entries: List[Dict[str, object]], goal_stops: float = 0.02) -> bool:
    if not entries:
        return False
    return all(float(item.get("residual_abs_stops", 0.0) or 0.0) <= goal_stops for item in entries)


def _contact_sheet_chunks(entries: List[Dict[str, object]], *, columns: int = 3, max_rows: int = 4) -> List[List[Dict[str, object]]]:
    page_size = max(1, int(columns) * int(max_rows))
    if not entries:
        return [[]]
    return [entries[index:index + page_size] for index in range(0, len(entries), page_size)]


def _contact_sheet_format_tint(value: object) -> str:
    try:
        tint = float(value)
    except (TypeError, ValueError):
        return ""
    rounded = round(tint, 1)
    if abs(rounded) < 1e-6:
        rounded = 0.0
    text = f"Tint {rounded:+.1f}"
    return text.replace("+0.0", "0.0").replace("-0.0", "0.0")


def _contact_sheet_format_kelvin(value: object) -> str:
    try:
        kelvin = float(value)
    except (TypeError, ValueError):
        return ""
    if kelvin <= 0:
        return ""
    return f"{int(round(kelvin))}K"


def _contact_sheet_format_iso(value: object) -> str:
    try:
        iso = float(value)
    except (TypeError, ValueError):
        return ""
    if iso <= 0:
        return ""
    return f"ISO {int(round(iso))}"


def _contact_sheet_format_shutter(value: object) -> str:
    try:
        shutter_seconds = float(value)
    except (TypeError, ValueError):
        return ""
    if shutter_seconds <= 0:
        return ""
    denominator = round(1.0 / shutter_seconds)
    if denominator > 0 and abs((1.0 / denominator) - shutter_seconds) <= 0.001:
        return f"1/{int(denominator)}"
    return f"{shutter_seconds:.4f}s"


def _contact_sheet_join_bits(parts: List[str]) -> str:
    return " | ".join(part for part in parts if str(part).strip())


def _contact_sheet_metric_range(values: List[float], *, fallback_span: float = 1.0) -> Tuple[float, float]:
    if not values:
        return (0.0, fallback_span)
    low = min(values)
    high = max(values)
    if math.isclose(low, high, abs_tol=1e-9):
        pad = max(abs(low) * 0.05, fallback_span * 0.5, 0.5)
        return low - pad, high + pad
    span = high - low
    pad = max(span * 0.08, span * 0.02)
    return low - pad, high + pad


def _contact_sheet_svg_single_series(
    title: str,
    labels: List[str],
    values: List[float],
    *,
    stroke: str,
    units: str,
    goal: Optional[float] = None,
) -> str:
    if not labels or not values:
        return ""
    width = 880
    height = 180
    margin_left = 58
    margin_right = 18
    margin_top = 18
    margin_bottom = 36
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    y_min, y_max = _contact_sheet_metric_range(values, fallback_span=0.5)

    def x_for(index: int) -> float:
        if len(values) == 1:
            return margin_left + plot_width / 2
        return margin_left + (index / max(len(values) - 1, 1)) * plot_width

    def y_for(value: float) -> float:
        if math.isclose(y_min, y_max, abs_tol=1e-9):
            return margin_top + plot_height / 2
        return margin_top + (1.0 - ((value - y_min) / (y_max - y_min))) * plot_height

    points = " ".join(f"{x_for(index):.1f},{y_for(value):.1f}" for index, value in enumerate(values))
    y_ticks = [y_min, (y_min + y_max) / 2.0, y_max]
    goal_markup = ""
    if goal is not None and y_min <= goal <= y_max:
        goal_y = y_for(goal)
        goal_markup = (
            f"<line x1='{margin_left:.1f}' y1='{goal_y:.1f}' x2='{width - margin_right:.1f}' y2='{goal_y:.1f}' "
            "stroke='#9ca3af' stroke-width='1.5' stroke-dasharray='6 5' />"
            f"<text x='{width - margin_right:.1f}' y='{goal_y - 6:.1f}' text-anchor='end' fill='#6b7280' font-size='12'>goal {goal:+.2f}</text>"
        )

    x_labels = []
    for index, label in enumerate(labels):
        x = x_for(index)
        x_labels.append(
            f"<text x='{x:.1f}' y='{height - 10:.1f}' text-anchor='middle' fill='#64748b' font-size='11'>{html.escape(label)}</text>"
        )
    y_labels = []
    for tick in y_ticks:
        y = y_for(tick)
        y_labels.append(
            f"<line x1='{margin_left:.1f}' y1='{y:.1f}' x2='{width - margin_right:.1f}' y2='{y:.1f}' stroke='#e2e8f0' stroke-width='1' />"
            f"<text x='{margin_left - 8:.1f}' y='{y + 4:.1f}' text-anchor='end' fill='#64748b' font-size='11'>{tick:.2f}{html.escape(units)}</text>"
        )
    point_markup = "".join(
        f"<circle cx='{x_for(index):.1f}' cy='{y_for(value):.1f}' r='4.5' fill='{stroke}' />"
        for index, value in enumerate(values)
    )
    return (
        f"<svg viewBox='0 0 {width} {height}' role='img' aria-label='{html.escape(title)}'>"
        f"<rect x='0' y='0' width='{width}' height='{height}' fill='white' />"
        f"<text x='0' y='14' fill='#0f172a' font-size='14' font-weight='700'>{html.escape(title)}</text>"
        + "".join(y_labels)
        + goal_markup
        + f"<polyline fill='none' stroke='{stroke}' stroke-width='3' points='{points}' />"
        + point_markup
        + "".join(x_labels)
        + "</svg>"
    )


def _contact_sheet_svg_color_chart(title: str, labels: List[str], kelvin_values: List[float], tint_values: List[float]) -> str:
    if not labels:
        return ""
    width = 880
    height = 220
    margin_left = 58
    margin_right = 18
    margin_top = 18
    margin_bottom = 36
    track_gap = 20
    track_height = (height - margin_top - margin_bottom - track_gap) / 2.0
    plot_width = width - margin_left - margin_right
    kelvin_min, kelvin_max = _contact_sheet_metric_range(kelvin_values, fallback_span=100.0)
    tint_min, tint_max = _contact_sheet_metric_range(tint_values, fallback_span=1.0)

    def x_for(index: int) -> float:
        if len(labels) == 1:
            return margin_left + plot_width / 2
        return margin_left + (index / max(len(labels) - 1, 1)) * plot_width

    def track_y_for(value: float, low: float, high: float, top: float) -> float:
        if math.isclose(low, high, abs_tol=1e-9):
            return top + track_height / 2
        return top + (1.0 - ((value - low) / (high - low))) * track_height

    def series_markup(values: List[float], low: float, high: float, top: float, stroke: str, unit_label: str) -> str:
        if not values:
            return ""
        points = " ".join(
            f"{x_for(index):.1f},{track_y_for(value, low, high, top):.1f}"
            for index, value in enumerate(values)
        )
        ticks = [low, (low + high) / 2.0, high]
        markup = "".join(
            f"<line x1='{margin_left:.1f}' y1='{track_y_for(tick, low, high, top):.1f}' x2='{width - margin_right:.1f}' y2='{track_y_for(tick, low, high, top):.1f}' stroke='#e2e8f0' stroke-width='1' />"
            f"<text x='{margin_left - 8:.1f}' y='{track_y_for(tick, low, high, top) + 4:.1f}' text-anchor='end' fill='#64748b' font-size='11'>{tick:.1f}{html.escape(unit_label)}</text>"
            for tick in ticks
        )
        markup += f"<polyline fill='none' stroke='{stroke}' stroke-width='3' points='{points}' />"
        markup += "".join(
            f"<circle cx='{x_for(index):.1f}' cy='{track_y_for(value, low, high, top):.1f}' r='4.5' fill='{stroke}' />"
            for index, value in enumerate(values)
        )
        return markup

    x_labels = "".join(
        f"<text x='{x_for(index):.1f}' y='{height - 10:.1f}' text-anchor='middle' fill='#64748b' font-size='11'>{html.escape(label)}</text>"
        for index, label in enumerate(labels)
    )
    return (
        f"<svg viewBox='0 0 {width} {height}' role='img' aria-label='{html.escape(title)}'>"
        f"<rect x='0' y='0' width='{width}' height='{height}' fill='white' />"
        f"<text x='0' y='14' fill='#0f172a' font-size='14' font-weight='700'>{html.escape(title)}</text>"
        f"<text x='{margin_left:.1f}' y='{margin_top - 4:.1f}' fill='#64748b' font-size='11' font-weight='700'>Kelvin</text>"
        + series_markup(kelvin_values, kelvin_min, kelvin_max, margin_top, "#0f766e", "K")
        + f"<text x='{margin_left:.1f}' y='{margin_top + track_height + track_gap - 4:.1f}' fill='#64748b' font-size='11' font-weight='700'>Tint</text>"
        + series_markup(tint_values, tint_min, tint_max, margin_top + track_height + track_gap, "#7c3aed", "")
        + x_labels
        + "</svg>"
    )


def _contact_sheet_view_model(payload: Dict[str, object]) -> Dict[str, object]:
    recommended_strategy = dict(payload.get("recommended_strategy") or {})
    recommended_key = str(recommended_strategy.get("strategy_key") or "")
    strategies = list(payload.get("strategies") or [])
    strategy_payload = next((item for item in strategies if str(item.get("strategy_key") or "") == recommended_key), None)
    if strategy_payload is None and strategies:
        strategy_payload = strategies[0]
        recommended_key = str(strategy_payload.get("strategy_key") or "")
        recommended_strategy = dict(strategy_payload)

    ipp2_validation = dict(payload.get("ipp2_validation") or {})
    status_counts = dict(ipp2_validation.get("status_counts") or {})
    shared_originals = {
        str(item.get("clip_id") or ""): dict(item)
        for item in list(payload.get("shared_originals") or [])
        if str(item.get("clip_id") or "").strip()
    }
    overlay_by_clip = {
        str(item.get("clip_id") or ""): dict(item)
        for item in list(((payload.get("sphere_detection_summary") or {}).get("rows") or []))
        if str(item.get("clip_id") or "").strip()
    }
    validation_by_clip = {
        str(item.get("clip_id") or ""): dict(item)
        for item in list(ipp2_validation.get("rows") or [])
        if str(item.get("clip_id") or "").strip() and (
            not recommended_key or str(item.get("strategy_key") or "") == recommended_key
        )
    }

    entries: List[Dict[str, object]] = []
    for clip in list((strategy_payload or {}).get("clips") or []):
        clip_id = str(clip.get("clip_id") or "")
        if not clip_id:
            continue
        shared = shared_originals.get(clip_id, {})
        clip_metadata = dict(shared.get("clip_metadata") or clip.get("clip_metadata") or {})
        ipp2_row = dict(clip.get("ipp2_validation") or validation_by_clip.get(clip_id) or {})
        exposure_metrics = dict(((clip.get("metrics") or {}).get("exposure") or {}))
        commit_values = dict(((clip.get("metrics") or {}).get("commit_values") or {}))
        overlay = overlay_by_clip.get(clip_id, {})
        original_image = str(
            shared.get("original_frame")
            or clip.get("original_frame")
            or ipp2_row.get("original_image_path")
            or ""
        )
        corrected_image = str(
            ipp2_row.get("corrected_image_path")
            or clip.get("both_corrected")
            or ""
        )
        overlay_image = str(
            overlay.get("original_overlay_path")
            or overlay.get("corrected_overlay_path")
            or original_image
        )
        corrected_overlay_image = str(overlay.get("corrected_overlay_path") or "")
        sample_1_ire = float(
            ipp2_row.get("sample_1_ire", exposure_metrics.get("sample_1_ire", shared.get("sample_1_ire", 0.0))) or 0.0
        )
        sample_2_ire = float(
            ipp2_row.get("sample_2_ire", exposure_metrics.get("sample_2_ire", shared.get("sample_2_ire", 0.0))) or 0.0
        )
        sample_3_ire = float(
            ipp2_row.get("sample_3_ire", exposure_metrics.get("sample_3_ire", shared.get("sample_3_ire", 0.0))) or 0.0
        )
        display_scalar_log2 = float(
            shared.get(
                "display_scalar_log2",
                exposure_metrics.get(
                    "display_scalar_log2",
                    exposure_metrics.get("measured_log2_luminance_monitoring", 0.0),
                ),
            )
            or 0.0
        )
        display_scalar_ire = _contact_sheet_display_scalar_ire(
            display_scalar_log2,
            shared.get("display_scalar_ire", exposure_metrics.get("display_scalar_ire")),
        )
        adjusted_display_scalar_log2 = float(ipp2_row.get("ipp2_value_log2", display_scalar_log2) or display_scalar_log2)
        adjusted_display_scalar_ire = _contact_sheet_display_scalar_ire(
            adjusted_display_scalar_log2,
            ipp2_row.get("ipp2_value_ire"),
        )
        status = str(ipp2_row.get("status") or "REVIEW")
        tone = _contact_sheet_status_tone(status)
        residual_stops = float(ipp2_row.get("ipp2_residual_stops", 0.0) or 0.0)
        residual_abs_stops = float(ipp2_row.get("ipp2_residual_abs_stops", abs(residual_stops)) or abs(residual_stops))
        exposure_adjust_stops = float(
            commit_values.get("exposureAdjust", ipp2_row.get("correction_stops", exposure_metrics.get("final_offset_stops", 0.0)))
            or 0.0
        )
        operator_guidance = dict(ipp2_row.get("operator_guidance") or {})
        result_statement = {
            "PASS": "Validation landed within the IPP2 tolerance band.",
            "REVIEW": "Correction is close, but should be reviewed before commit.",
            "FAIL": "Correction remains outside the accepted IPP2 tolerance.",
        }.get(status, "Review the corrected result before commit.")
        operator_result_label = {
            "PASS": "In range",
            "REVIEW": "Needs adjustment",
            "FAIL": "Outside tolerance",
        }.get(status, "Needs adjustment")
        iso_value = clip_metadata.get("iso")
        shutter_seconds = clip_metadata.get("shutter_seconds")
        original_kelvin = clip_metadata.get("kelvin")
        original_tint = clip_metadata.get("tint")
        original_metadata_line = _contact_sheet_join_bits(
            [
                _contact_sheet_format_iso(iso_value),
                _contact_sheet_format_shutter(shutter_seconds),
                _contact_sheet_format_kelvin(original_kelvin),
                _contact_sheet_format_tint(original_tint),
            ]
        )
        adjusted_metadata_line = _contact_sheet_join_bits(
            [
                f"Exp {exposure_adjust_stops:+.2f}",
                _contact_sheet_format_kelvin(commit_values.get("kelvin")),
                _contact_sheet_format_tint(commit_values.get("tint")),
            ]
        )
        entries.append(
            {
                "clip_id": clip_id,
                "camera_label": str(ipp2_row.get("camera_label") or _camera_label_for_reporting(clip_id)),
                "grid_label": str(ipp2_row.get("camera_label") or _camera_label_for_reporting(clip_id)),
                "is_hero_camera": bool(clip.get("is_hero_camera") or clip_id == str(payload.get("hero_clip_id") or "")),
                "status": status,
                "tone": tone,
                "result_label": "PASS" if status == "PASS" else "REVIEW" if status == "REVIEW" else "FAIL",
                "operator_result_label": operator_result_label,
                "result_statement": result_statement,
                "recommended_action": str(ipp2_row.get("suggested_action") or operator_guidance.get("suggested_action") or "No adjustment needed"),
                "action_note": str(ipp2_row.get("operator_notes") or operator_guidance.get("notes") or "Stored IPP2 validation result."),
                "original_image": original_image,
                "corrected_image": corrected_image,
                "overlay_image": overlay_image,
                "corrected_overlay_image": corrected_overlay_image,
                "original_profile": str(ipp2_row.get("ipp2_original_gray_exposure_summary") or shared.get("gray_exposure_summary") or "n/a"),
                "corrected_profile": str(ipp2_row.get("ipp2_gray_exposure_summary") or exposure_metrics.get("gray_exposure_summary") or "n/a"),
                "sample_1_ire": sample_1_ire,
                "sample_2_ire": sample_2_ire,
                "sample_3_ire": sample_3_ire,
                "display_scalar_log2": display_scalar_log2,
                "display_scalar_ire": display_scalar_ire,
                "adjusted_display_scalar_log2": adjusted_display_scalar_log2,
                "adjusted_display_scalar_ire": adjusted_display_scalar_ire,
                "exposure_adjust_stops": exposure_adjust_stops,
                "kelvin": float(commit_values.get("kelvin", 0.0) or 0.0),
                "tint": float(commit_values.get("tint", 0.0) or 0.0),
                "original_kelvin": float(original_kelvin or 0.0) if original_kelvin is not None else 0.0,
                "original_tint": float(original_tint or 0.0) if original_tint is not None else 0.0,
                "iso": float(iso_value or 0.0) if iso_value is not None else 0.0,
                "shutter_seconds": float(shutter_seconds or 0.0) if shutter_seconds is not None else 0.0,
                "original_metadata_line": original_metadata_line,
                "adjusted_metadata_line": adjusted_metadata_line,
                "residual_stops": residual_stops,
                "residual_abs_stops": residual_abs_stops,
                "offset_to_anchor": float(ipp2_row.get("camera_offset_from_anchor", ipp2_row.get("derived_exposure_offset_stops", 0.0)) or 0.0),
                "reference_use": str(ipp2_row.get("reference_use") or clip.get("reference_use") or "Included"),
                "trust_score": float(clip.get("trust_score", ipp2_row.get("trust_score", 0.0)) or 0.0),
                "trust_class": str(clip.get("trust_class") or ipp2_row.get("trust_class") or "TRUSTED"),
                "sphere_detection_note": str(ipp2_row.get("sphere_detection_note") or "Sphere detection: verified"),
                "profile_note": str(ipp2_row.get("profile_note") or "Profile consistent with stored solve."),
                "measurement_domain": str(payload.get("measurement_preview_transform") or REVIEW_PREVIEW_TRANSFORM),
            }
        )

    before_values = [float(item.get("display_scalar_log2", 0.0) or 0.0) for item in entries]
    after_values = [
        float(
            item.get("adjusted_display_scalar_log2", item.get("display_scalar_log2", 0.0))
            or 0.0
        )
        for item in entries
    ]
    center_values = [float(item.get("sample_2_ire", 0.0) or 0.0) for item in entries if float(item.get("sample_2_ire", 0.0) or 0.0) > 0.0]
    reference_candidate = _contact_sheet_reference_candidate(entries)
    contact_pages = []
    for page_index, page_entries in enumerate(_contact_sheet_chunks(entries), start=1):
        labels = [str(item.get("camera_label") or item.get("clip_id") or "") for item in page_entries]
        original_exposure_values = [float(item.get("display_scalar_ire", 0.0) or 0.0) for item in page_entries]
        adjusted_exposure_values = [float(item.get("adjusted_display_scalar_ire", item.get("display_scalar_ire", 0.0)) or 0.0) for item in page_entries]
        original_kelvin_values = [float(item.get("original_kelvin", 0.0) or 0.0) for item in page_entries]
        original_tint_values = [float(item.get("original_tint", 0.0) or 0.0) for item in page_entries]
        adjusted_kelvin_values = [float(item.get("kelvin", 0.0) or 0.0) for item in page_entries]
        adjusted_tint_values = [float(item.get("tint", 0.0) or 0.0) for item in page_entries]
        contact_pages.append(
            {
                "page_index": page_index,
                "entries": page_entries,
                "original_exposure_chart_svg": _contact_sheet_svg_single_series(
                    "Original exposure scalar (IRE)",
                    labels,
                    original_exposure_values,
                    stroke="#0f172a",
                    units="",
                ),
                "adjusted_exposure_chart_svg": _contact_sheet_svg_single_series(
                    "Adjusted exposure scalar (IRE)",
                    labels,
                    adjusted_exposure_values,
                    stroke="#15803d",
                    units="",
                    goal=None,
                ),
                "original_color_chart_svg": _contact_sheet_svg_color_chart(
                    "Original white balance trace",
                    labels,
                    original_kelvin_values,
                    original_tint_values,
                ),
                "adjusted_color_chart_svg": _contact_sheet_svg_color_chart(
                    "Adjusted white balance trace",
                    labels,
                    adjusted_kelvin_values,
                    adjusted_tint_values,
                ),
            }
        )
    return {
        "title": "R3DMatch Calibration Assessment",
        "batch_label": ", ".join(str(item) for item in list(payload.get("selected_clip_groups") or []) if str(item).strip()) or str(payload.get("run_label") or "Calibration run"),
        "camera_count": len(entries),
        "strategy_label": str((strategy_payload or {}).get("strategy_label") or recommended_strategy.get("strategy_label") or "Median"),
        "anchor_summary": str(
            payload.get("exposure_anchor_summary")
            or recommended_strategy.get("anchor_summary")
            or "Exposure Anchor: Derived from retained cluster"
        ),
        "domain_label": str(payload.get("measurement_preview_transform") or REVIEW_PREVIEW_TRANSFORM),
        "all_within_tolerance": bool(ipp2_validation.get("all_within_tolerance")),
        "status_counts": {
            "PASS": int(status_counts.get("PASS", 0) or 0),
            "REVIEW": int(status_counts.get("REVIEW", 0) or 0),
            "FAIL": int(status_counts.get("FAIL", 0) or 0),
        },
        "before_spread_stops": (max(before_values) - min(before_values)) if before_values else 0.0,
        "after_spread_stops": (max(after_values) - min(after_values)) if after_values else 0.0,
        "center_ire_min": min(center_values) if center_values else 0.0,
        "center_ire_max": max(center_values) if center_values else 0.0,
        "best_residual": float(ipp2_validation.get("best_residual", 0.0) or 0.0),
        "median_residual": float(ipp2_validation.get("median_residual", 0.0) or 0.0),
        "worst_residual": float(ipp2_validation.get("max_residual", 0.0) or 0.0),
        "goal_threshold_stops": 0.02,
        "goal_achieved": _contact_sheet_goal_achieved(entries, 0.02),
        "reference_candidate": reference_candidate,
        "visuals": dict(payload.get("visuals") or {}),
        "contact_pages": contact_pages,
        "entries": entries,
    }


def render_contact_sheet_pdf(
    payload: Dict[str, object],
    *,
    output_path: str,
    title: str,
    timestamp_label: Optional[str] = None,
) -> str:
    raise_if_cancelled("Run cancelled before PDF rendering.")
    from PIL import Image, ImageDraw, ImageFont

    view_model = _contact_sheet_view_model(payload)

    def load_font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
        preferred = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
        try:
            return ImageFont.truetype(preferred, size=size)
        except OSError:
            return ImageFont.load_default()

    def wrapped(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, width: int) -> List[str]:
        words = str(text or "").split()
        if not words:
            return [""]
        lines: List[str] = []
        current = words[0]
        for word in words[1:]:
            trial = f"{current} {word}"
            if draw.textlength(trial, font=font) <= width:
                current = trial
            else:
                lines.append(current)
                current = word
        lines.append(current)
        return lines

    def fit_image(path: str, width: int, height: int) -> Image.Image:
        canvas = Image.new("RGB", (width, height), "#f4f7fb")
        if path and Path(path).exists():
            preview = Image.open(path).convert("RGB")
            preview.thumbnail((width, height))
            canvas.paste(preview, ((width - preview.width) // 2, (height - preview.height) // 2))
        return canvas

    page_width = 1700
    page_height = 2520
    margin = 56
    title_font = load_font(42, bold=True)
    subtitle_font = load_font(20, bold=False)
    h2_font = load_font(24, bold=True)
    h3_font = load_font(20, bold=True)
    body_font = load_font(16)
    small_font = load_font(13)
    small_bold = load_font(13, bold=True)
    metric_font = load_font(24, bold=True)
    logo_image = None
    if LOGO_PATH.exists():
        try:
            logo_image = Image.open(LOGO_PATH).convert("RGBA")
            logo_image.thumbnail((170, 80))
        except OSError:
            logo_image = None

    def draw_header(page: Image.Image, *, first_page: bool, page_label: str) -> int:
        draw = ImageDraw.Draw(page)
        header_x = margin
        if first_page and logo_image is not None:
            page.paste(logo_image, (margin, margin), logo_image)
            header_x += logo_image.width + 18
        draw.text((header_x, margin), title, fill="#0f172a", font=title_font)
        draw.text((header_x, margin + 48), page_label, fill="#475569", font=subtitle_font)
        if timestamp_label:
            draw.text((page_width - margin - 300, margin + 6), timestamp_label, fill="#64748b", font=small_font)
        draw.line((margin, margin + 104, page_width - margin, margin + 104), fill="#0f172a", width=3)
        return margin + 116

    def draw_meta_line(draw: ImageDraw.ImageDraw, y: int, items: List[str]) -> int:
        x = margin
        line_y = y
        for item in items:
            text = str(item)
            width = int(draw.textlength(text, font=small_font))
            if x + width > page_width - margin:
                x = margin
                line_y += 22
            draw.text((x, line_y), text, fill="#475569", font=small_font)
            x += width + 20
        return line_y + 20

    def draw_chart(draw: ImageDraw.ImageDraw, rect: Tuple[int, int, int, int], labels: List[str], values: List[float], *, stroke: str) -> None:
        x0, y0, x1, y1 = rect
        draw.rectangle(rect, outline="#d7dee8", width=1)
        chart_left = x0 + 48
        chart_right = x1 - 14
        chart_top = y0 + 18
        chart_bottom = y1 - 26
        y_min, y_max = _contact_sheet_metric_range(values, fallback_span=0.5)
        if not values:
            return
        draw.line((chart_left, chart_bottom, chart_right, chart_bottom), fill="#94a3b8", width=1)
        draw.line((chart_left, chart_top, chart_left, chart_bottom), fill="#94a3b8", width=1)
        ticks = [y_min, (y_min + y_max) / 2.0, y_max]
        for tick in ticks:
            y = chart_bottom if math.isclose(y_min, y_max, abs_tol=1e-9) else chart_top + (1.0 - ((tick - y_min) / (y_max - y_min))) * (chart_bottom - chart_top)
            draw.line((chart_left, int(y), chart_right, int(y)), fill="#e7edf4", width=1)
            draw.text((x0 + 4, int(y) - 7), f"{tick:.2f}", fill="#64748b", font=small_font)
        points = []
        for index, value in enumerate(values):
            x = chart_left + ((chart_right - chart_left) / max(len(values) - 1, 1)) * index if len(values) > 1 else (chart_left + chart_right) / 2
            y = chart_bottom if math.isclose(y_min, y_max, abs_tol=1e-9) else chart_top + (1.0 - ((value - y_min) / (y_max - y_min))) * (chart_bottom - chart_top)
            points.append((int(x), int(y)))
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=stroke)
            draw.text((x - 18, chart_bottom + 6), labels[index], fill="#64748b", font=small_font)
        if len(points) > 1:
            draw.line(points, fill=stroke, width=3)

    def draw_color_chart(draw: ImageDraw.ImageDraw, rect: Tuple[int, int, int, int], labels: List[str], kelvin_values: List[float], tint_values: List[float]) -> None:
        x0, y0, x1, y1 = rect
        draw.rectangle(rect, outline="#d7dee8", width=1)
        mid = y0 + (y1 - y0) // 2
        draw.text((x0 + 12, y0 + 8), "Kelvin", fill="#475569", font=small_bold)
        draw_chart(draw, (x0, y0 + 18, x1, mid - 4), labels, kelvin_values, stroke="#0f766e")
        draw.text((x0 + 12, mid + 8), "Tint", fill="#475569", font=small_bold)
        draw_chart(draw, (x0, mid + 18, x1, y1), labels, tint_values, stroke="#7c3aed")

    def draw_contact_cell(page: Image.Image, draw: ImageDraw.ImageDraw, entry: Dict[str, object], *, x: int, y: int, width: int, adjusted: bool, image_height: int) -> None:
        image_path = str(entry["corrected_image"] if adjusted else entry["original_image"])
        preview = fit_image(image_path, width, image_height)
        page.paste(preview, (x, y))
        line_y = y + image_height + 8
        draw.text((x, line_y), f"{entry['camera_label']} / {entry['clip_id']}", fill="#0f172a", font=small_bold)
        line_y += 20
        meta_line = str(entry["adjusted_metadata_line"] if adjusted else entry["original_metadata_line"])
        if meta_line:
            draw.text((x, line_y), meta_line, fill="#475569", font=small_font)
            line_y += 18
        if not adjusted:
            draw.text((x, line_y), "Gray Sphere Totals", fill="#64748b", font=small_bold)
            line_y += 16
            totals_line = (
                f"S1 {float(entry['sample_1_ire']):.1f}   "
                f"S2 {float(entry['sample_2_ire']):.1f}   "
                f"S3 {float(entry['sample_3_ire']):.1f}   "
                f"Scalar {float(entry['display_scalar_ire']):.1f}"
            )
            draw.text((x, line_y), totals_line, fill="#0f172a", font=small_font)

    pages: List[Image.Image] = []
    snapshot_items = [
        f"Batch {view_model['batch_label']}",
        f"Cameras {int(view_model['camera_count'])}",
        f"Strategy {view_model['strategy_label']}",
        f"Domain {view_model['domain_label']}",
        f"Exposure Anchor {view_model['anchor_summary']}",
        f"Hero clip {payload.get('hero_clip_id') or 'None selected'}",
        f"PASS / REVIEW / FAIL {view_model['status_counts']['PASS']} / {view_model['status_counts']['REVIEW']} / {view_model['status_counts']['FAIL']}",
        f"Spread {float(view_model['before_spread_stops']):.2f} → {float(view_model['after_spread_stops']):.2f} stops",
        f"Residual median {float(view_model['median_residual']):.3f} | worst {float(view_model['worst_residual']):.3f}",
        f"±0.02 goal {'met' if view_model['goal_achieved'] else 'open'}",
    ]

    contact_pages = list(view_model.get("contact_pages") or [])
    for contact_page in contact_pages:
        raise_if_cancelled("Run cancelled during contact sheet PDF rendering.")
        page = Image.new("RGB", (page_width, page_height), "white")
        draw = ImageDraw.Draw(page)
        page_index = int(contact_page.get("page_index", 1) or 1)
        y = draw_header(page, first_page=page_index == 1, page_label=f"Calibration Contact Sheet · Page {page_index}")
        y = draw_meta_line(draw, y, snapshot_items)
        y += 10
        entries = list(contact_page.get("entries") or [])
        draw.text((margin, y), "Original Camera Grid", fill="#0f172a", font=h2_font)
        y += 26
        gutter = 16
        cell_width = int((page_width - (2 * margin) - (2 * gutter)) / 3)
        rows = max(1, math.ceil(len(entries) / 3))
        original_image_height = 138 if rows >= 4 else 156
        original_cell_height = original_image_height + 78
        for index, entry in enumerate(entries):
            row = index // 3
            col = index % 3
            x = margin + col * (cell_width + gutter)
            cell_y = y + row * original_cell_height
            draw_contact_cell(page, draw, entry, x=x, y=cell_y, width=cell_width, adjusted=False, image_height=original_image_height)
        y += rows * original_cell_height + 18
        draw.line((margin, y, page_width - margin, y), fill="#d7dee8", width=1)
        y += 10
        draw.text((margin, y), "Original Array Synopsis", fill="#0f172a", font=h2_font)
        y += 28
        chart_height = 208
        chart_width = int((page_width - (2 * margin) - gutter) / 2)
        labels = [str(item.get("camera_label") or item.get("clip_id") or "") for item in entries]
        draw_chart(draw, (margin, y, margin + chart_width, y + chart_height), labels, [float(item.get("display_scalar_ire", 0.0) or 0.0) for item in entries], stroke="#0f172a")
        draw_color_chart(draw, (margin + chart_width + gutter, y, page_width - margin, y + chart_height), labels, [float(item.get("original_kelvin", 0.0) or 0.0) for item in entries], [float(item.get("original_tint", 0.0) or 0.0) for item in entries])
        y += chart_height + 18
        draw.text((margin, y), "Adjusted Camera Grid", fill="#0f172a", font=h2_font)
        y += 26
        adjusted_image_height = 138 if rows >= 4 else 156
        adjusted_cell_height = adjusted_image_height + 52
        for index, entry in enumerate(entries):
            row = index // 3
            col = index % 3
            x = margin + col * (cell_width + gutter)
            cell_y = y + row * adjusted_cell_height
            draw_contact_cell(page, draw, entry, x=x, y=cell_y, width=cell_width, adjusted=True, image_height=adjusted_image_height)
        y += rows * adjusted_cell_height + 18
        draw.line((margin, y, page_width - margin, y), fill="#d7dee8", width=1)
        y += 10
        draw.text((margin, y), "Adjusted Array Synopsis", fill="#0f172a", font=h2_font)
        y += 28
        draw_chart(draw, (margin, y, margin + chart_width, y + chart_height), labels, [float(item.get("adjusted_display_scalar_ire", item.get("display_scalar_ire", 0.0)) or 0.0) for item in entries], stroke="#15803d")
        draw_color_chart(draw, (margin + chart_width + gutter, y, page_width - margin, y + chart_height), labels, [float(item.get("kelvin", 0.0) or 0.0) for item in entries], [float(item.get("tint", 0.0) or 0.0) for item in entries])
        pages.append(page)

    for entry in view_model["entries"]:
        raise_if_cancelled("Run cancelled during contact sheet PDF rendering.")
        page = Image.new("RGB", (page_width, page_height), "white")
        draw = ImageDraw.Draw(page)
        y = draw_header(page, first_page=False, page_label=f"{entry['camera_label']} · {entry['clip_id']}")
        tone = str(entry["tone"])
        status_color = {"good": "#166534", "warning": "#92400e", "danger": "#991b1b"}[tone]
        draw.text((page_width - margin - 170, margin + 8), str(entry["result_label"]), fill=status_color, font=h2_font)
        draw.text((margin, y), "Recommended Action", fill="#64748b", font=small_bold)
        draw.text((margin, y + 18), str(entry["recommended_action"]), fill="#0f172a", font=h2_font)
        y += 64
        draw.line((margin, y, page_width - margin, y), fill="#d7dee8", width=1)
        y += 14
        gutter = 16
        comparison_width = 420
        overlay_width = page_width - (2 * margin) - (2 * comparison_width) - (2 * gutter)
        image_height = 300
        original = fit_image(str(entry["original_image"]), comparison_width, image_height)
        corrected = fit_image(str(entry["corrected_image"]), comparison_width, image_height)
        overlay = fit_image(str(entry["overlay_image"]), overlay_width, image_height)
        page.paste(original, (margin, y))
        page.paste(corrected, (margin + comparison_width + gutter, y))
        page.paste(overlay, (margin + (2 * (comparison_width + gutter)), y))
        draw.text((margin, y - 20), "Original", fill="#64748b", font=small_bold)
        draw.text((margin + comparison_width + gutter, y - 20), "Corrected", fill="#64748b", font=small_bold)
        draw.text((margin + (2 * (comparison_width + gutter)), y - 20), "Solve Overlay", fill="#64748b", font=small_bold)
        y += image_height + 18
        sample_lines = [
            f"Sample 1 {float(entry['sample_1_ire']):.1f} IRE",
            f"Sample 2 {float(entry['sample_2_ire']):.1f} IRE",
            f"Sample 3 {float(entry['sample_3_ire']):.1f} IRE",
        ]
        for index, line in enumerate(sample_lines):
            draw.text((margin + (2 * (comparison_width + gutter)), y + (index * 18)), line, fill="#0f172a", font=body_font)
        metric_y = y + 70
        draw.line((margin, metric_y - 10, page_width - margin, metric_y - 10), fill="#d7dee8", width=1)
        metrics = [
            ("Samples", f"S1 {float(entry['sample_1_ire']):.1f} | S2 {float(entry['sample_2_ire']):.1f} | S3 {float(entry['sample_3_ire']):.1f}"),
            ("Scalar", f"{float(entry['display_scalar_ire']):.1f} IRE"),
            ("Correction", f"{float(entry['exposure_adjust_stops']):+.2f} exp | {int(round(float(entry['kelvin']) or 0.0))}K | {float(entry['tint']):+.1f} tint"),
            ("Residual", f"{float(entry['residual_abs_stops']):.3f} stops"),
        ]
        col_width = int((page_width - (2 * margin) - (3 * gutter)) / 4)
        metric_block_height = 0
        for index, (label, value) in enumerate(metrics):
            x = margin + index * (col_width + gutter)
            draw.text((x, metric_y), label, fill="#64748b", font=small_bold)
            wrapped_lines = wrapped(draw, value, body_font, col_width)
            for line_index, line in enumerate(wrapped_lines):
                draw.text((x, metric_y + 18 + (line_index * 18)), line, fill="#0f172a", font=body_font)
            metric_block_height = max(metric_block_height, 18 + (len(wrapped_lines) * 18))
        flags_y = metric_y + metric_block_height + 24
        draw.line((margin, flags_y - 12, page_width - margin, flags_y - 12), fill="#d7dee8", width=1)
        flag_texts = [
            f"Reference use: {entry['reference_use']}",
            str(entry["sphere_detection_note"]),
            str(entry["profile_note"]),
            f"Offset to anchor {float(entry['offset_to_anchor']):+.3f} stops",
        ]
        for index, text in enumerate(flag_texts):
            draw.text((margin, flags_y + (index * 20)), text, fill="#334155", font=body_font)
        pages.append(page)

    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    raise_if_cancelled("Run cancelled before final PDF write.")
    pages[0].save(output, format="PDF", save_all=True, append_images=pages[1:])
    return str(output)


def build_ui_calibration_table(input_path: str, *, out_dir: str) -> Dict[str, object]:
    analysis_records = _load_analysis_records(input_path)
    rows = [
        {
            "clip_id": record["clip_id"],
            "group_key": record["group_key"],
            "measured_log2_luminance": record.get("diagnostics", {}).get("measured_log2_luminance"),
            "exposure_offset_stops": record.get("exposure_baseline_applied_stops", record.get("final_offset_stops")),
            "rgb_gains": record.get("pending_color_gains"),
            "confidence": record.get("confidence"),
        }
        for record in analysis_records
    ]
    payload = {"rows": rows}
    out_root = Path(out_dir).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    path = out_root / "ui_calibration_table.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {"table_json": str(path), "row_count": len(rows)}


def _exposure_correction_presentation(offset_stops: float) -> Dict[str, str]:
    offset = float(offset_stops or 0.0)
    if offset >= 0.05:
        return {
            "label": f"Exposure Correction: +{offset:.2f}",
            "tone": "lift",
            "short_reason": "Lifted",
        }
    if offset <= -0.05:
        return {
            "label": f"Exposure Correction: {offset:.2f}",
            "tone": "lower",
            "short_reason": "Lowered",
        }
    return {
        "label": f"Exposure Correction: {offset:.2f}",
        "tone": "neutral",
        "short_reason": "Near neutral",
    }


def _corrected_residual_presentation(validation: Dict[str, object]) -> Dict[str, str]:
    residual = float(validation.get("residual_error_stops", 0.0) or 0.0)
    label = f"Residual after correction: {residual:+.2f} stops"
    tone = {
        "within_tolerance": "good",
        "review": "warning",
        "outside_tolerance": "danger",
    }.get(str(validation.get("residual_status") or ""), "warning")
    summary = str(validation.get("residual_label") or "Needs review")
    return {
        "label": label,
        "summary": summary,
        "tone": tone,
    }


def _ipp2_validation_presentation(validation: Dict[str, object]) -> Dict[str, str]:
    residual = float(validation.get("ipp2_residual_stops", 0.0) or 0.0)
    status = str(validation.get("status") or "REVIEW")
    guidance = dict(validation.get("operator_guidance") or {})
    if not guidance:
        guidance = _operator_guidance_for_correction(
            correction_stops=float(
                validation.get("correction_stops", validation.get("applied_correction_stops", 0.0)) or 0.0
            ),
            residual_stops=residual,
            validation_status=status,
        )
    operator_status = str(guidance.get("status") or status or "REVIEW")
    correction_stops = float(guidance.get("correction_stops", validation.get("correction_stops", 0.0)) or 0.0)
    abs_correction = abs(correction_stops)
    scalar_abs_residual = abs(residual)
    profile_max_residual = float(validation.get("ipp2_profile_max_residual_stops", validation.get("ipp2_residual_abs_stops", abs(residual))) or abs(residual))
    profile_status = str(validation.get("profile_audit_status") or "PROFILE NEEDS REVIEW")
    profile_note = str(validation.get("profile_note") or "Profile needs review")
    if abs_correction >= WARNING_CORRECTION_STOPS:
        if scalar_abs_residual <= IPP2_VALIDATION_PASS_STOPS:
            presentation_state = "Outlier corrected successfully"
            presentation_note = "Verify T-Stop if needed, but final result is within tolerance."
            tone_key = "outlier_good"
        elif scalar_abs_residual <= IPP2_VALIDATION_REVIEW_STOPS:
            presentation_state = "Outlier needs review"
            presentation_note = "Large correction was required. Verify T-Stop and confirm result."
            tone_key = "outlier_warning"
        else:
            presentation_state = "Outlier still outside tolerance"
            presentation_note = "Camera exceeds safe correction range and did not converge. Verify T-Stop and re-run calibration."
            tone_key = "outlier_danger"
    else:
        presentation_state = {
            "PASS": "Within tolerance",
            "REVIEW": "Needs review",
            "FAIL": "Outside tolerance",
        }.get(status, "Needs review")
        presentation_note = {
            "PASS": "Final result is within the IPP2 acceptance tolerance.",
            "REVIEW": "Result is close, but should be checked before commit.",
            "FAIL": "Result remains outside the IPP2 acceptance tolerance.",
        }.get(status, "Check the result before commit.")
        tone_key = status
    tone = {
        "PASS": "good",
        "REVIEW": "warning",
        "FAIL": "danger",
        "OUTLIER": "danger",
        "outlier_good": "outlier-good",
        "outlier_warning": "outlier-warning",
        "outlier_danger": "outlier-danger",
    }.get(tone_key, "warning")
    summary = operator_status if operator_status == "OUTLIER" else status
    suggested_action = str(guidance.get("suggested_action") or "No aperture adjustment required")
    notes = str(guidance.get("notes") or "")
    rounded_stops = float(guidance.get("rounded_stops", validation.get("rounded_stops", 0.0)) or 0.0)
    physical_label = "Suggested physical adjustment" if operator_status == "OUTLIER" else "Suggested lens action"
    validation_label = {
        "PASS": "In range",
        "REVIEW": "Needs review",
        "FAIL": "Outside exposure tolerance",
        "OUTLIER": "Verify T-Stop",
    }.get(operator_status, summary)
    result_label = (
        "Outlier" if operator_status == "OUTLIER"
        else "In range" if status == "PASS"
        else "Needs adjustment" if status == "REVIEW"
        else "Outside tolerance"
    )
    exact_correction_text = f"Digital correction applied: {correction_stops:+.2f} stops"
    rounded_correction_text = f"Rounded operator target: {rounded_stops:+.2f} stops"
    residual_text = f"Exposure residual after validation: {scalar_abs_residual:.2f} stops"
    profile_residual_text = f"Worst zone residual after validation: {profile_max_residual:.2f} stops"
    gray_exposure_summary = str(validation.get("ipp2_gray_exposure_summary", validation.get("gray_exposure_summary", validation.get("ipp2_original_gray_exposure_summary", "n/a"))) or "n/a")
    target_profile_summary = str(validation.get("ipp2_target_gray_exposure_summary", validation.get("target_gray_exposure_summary", "n/a")) or "n/a")
    zone_residual_text = _format_zone_residual_summary(list(validation.get("ipp2_zone_residuals") or []))
    return {
        "label": f"IPP2 exposure residual after correction: {scalar_abs_residual:.2f} stops",
        "summary": summary,
        "presentation_state": presentation_state,
        "presentation_note": presentation_note,
        "validation_label": validation_label,
        "result_label": result_label,
        "profile_state": profile_status.replace("PROFILE ", "Profile ").title().replace("Shape Mismatch", "Shape mismatch"),
        "profile_note": profile_note,
        "action_label": physical_label,
        "action": suggested_action,
        "notes": notes,
        "exact_correction_text": exact_correction_text,
        "rounded_correction_text": rounded_correction_text,
        "residual_text": residual_text,
        "profile_residual_text": profile_residual_text,
        "gray_exposure_text": f"Gray Exposure: {gray_exposure_summary}",
        "target_profile_text": f"Reference profile: {target_profile_summary}",
        "profile_note_text": f"Profile note: {profile_note}",
        "zone_residual_text": f"Zone residuals: {zone_residual_text}" if zone_residual_text != "n/a" else "Zone residuals: n/a",
        "sphere_detection_note": str(validation.get("sphere_detection_note") or "Sphere detection: review needed"),
        "tone": tone,
    }


def render_contact_sheet_html(payload: Dict[str, object]) -> str:
    view_model = _contact_sheet_view_model(payload)
    logo_markup = (
        f"<img class='brand-logo' src='{LOGO_PATH.as_posix()}' alt='R3DMatch logo' />"
        if LOGO_PATH.exists()
        else "<div class='brand-logo-text'>R3DMatch</div>"
    )
    reference_candidate = dict(view_model.get("reference_candidate") or {})
    snapshot_bits = [
        f"Batch {html.escape(str(view_model['batch_label']))}",
        f"Cameras {int(view_model['camera_count'])}",
        f"Strategy {html.escape(str(view_model['strategy_label']))}",
        f"Domain {html.escape(str(view_model['domain_label']))}",
        f"Color preview: {html.escape(str(payload.get('color_preview_status') or 'unknown'))}",
        f"Exposure Anchor {html.escape(str(view_model['anchor_summary']))}",
        f"Hero clip: {html.escape(str(payload.get('hero_clip_id') or 'None selected'))}",
        f"PASS / REVIEW / FAIL {view_model['status_counts']['PASS']} / {view_model['status_counts']['REVIEW']} / {view_model['status_counts']['FAIL']}",
        f"Spread {float(view_model['before_spread_stops']):.2f} → {float(view_model['after_spread_stops']):.2f} stops",
        f"Residual median {float(view_model['median_residual']):.3f} | worst {float(view_model['worst_residual']):.3f}",
        f"Reference {html.escape(str(reference_candidate.get('camera_label') or 'n/a'))}",
        f"±0.02 goal {'met' if view_model['goal_achieved'] else 'open'}",
    ]

    def render_grid_cell(entry: Dict[str, object], *, adjusted: bool) -> str:
        image_path = _contact_sheet_image_src(entry.get("corrected_image" if adjusted else "original_image"))
        meta_line = str(entry.get("adjusted_metadata_line" if adjusted else "original_metadata_line") or "")
        header_line = f"{html.escape(str(entry.get('camera_label') or ''))} / {html.escape(str(entry.get('clip_id') or ''))}"
        totals_line = (
            f"S1 {float(entry.get('sample_1_ire', 0.0) or 0.0):.1f}   "
            f"S2 {float(entry.get('sample_2_ire', 0.0) or 0.0):.1f}   "
            f"S3 {float(entry.get('sample_3_ire', 0.0) or 0.0):.1f}   "
            f"Scalar {float(entry.get('display_scalar_ire', 0.0) or 0.0):.1f}"
        )
        footer = (
            f"<div class='cell-totals-label'>Gray Sphere Totals</div><div class='cell-totals'>{html.escape(totals_line)}</div>"
            if not adjusted
            else ""
        )
        return (
            "<article class='camera-cell'>"
            + (f"<img class='cell-image' src='{image_path}' alt='{html.escape(str(entry.get('clip_id') or ''))} {'adjusted' if adjusted else 'original'}' />" if image_path else "<div class='cell-image missing-image'>Image unavailable</div>")
            + f"<div class='cell-line cell-clip'>{header_line}</div>"
            + (f"<div class='cell-line cell-meta'>{html.escape(meta_line)}</div>" if meta_line else "")
            + footer
            + "</article>"
        )

    contact_pages_markup = []
    for page in list(view_model.get("contact_pages") or []):
        page_entries = list(page.get("entries") or [])
        first_page = int(page.get("page_index", 1) or 1) == 1
        contact_pages_markup.append(
            "<section class='contact-page'>"
            "<header class='sheet-header'>"
            + (f"<div class='sheet-brand'>{logo_markup}</div>" if first_page else "<div class='sheet-brand sheet-brand-placeholder'></div>")
            + "<div class='sheet-head-block'>"
            + (f"<div class='sheet-kicker'>Calibration Contact Sheet</div><h1>{html.escape(str(view_model.get('title') or 'R3DMatch Calibration Assessment'))}</h1>" if first_page else f"<div class='sheet-kicker'>Contact Sheet Page {int(page.get('page_index', 1) or 1)}</div><h1>{html.escape(str(view_model.get('batch_label') or 'Calibration run'))}</h1>")
            + f"<div class='sheet-meta'>{''.join(f'<span>{item}</span>' for item in snapshot_bits)}</div>"
            + (
                "<div class='sheet-guide'>What To Look For: Original / Corrected / Solve Overlay / Sample 1 / Sample 2 / Sample 3 / Validation Residual</div>"
                "<div class='sheet-guide'>Exposure Summary: original grid, adjusted grid, exposure trace, white-balance trace.</div>"
                if first_page
                else ""
            )
            + "</div>"
            "</header>"
            "<section class='grid-section'>"
            "<div class='section-strip'><h2>Original Camera Grid</h2><div class='section-note'>Stored measurement stills</div></div>"
            f"<div class='camera-grid'>{''.join(render_grid_cell(entry, adjusted=False) for entry in page_entries)}</div>"
            "</section>"
            "<section class='synopsis-section'>"
            "<div class='section-strip'><h2>Original Array Synopsis</h2><div class='section-note'>Camera order left to right</div></div>"
            "<div class='synopsis-grid'>"
            + (
                f"<article class='chart-panel'><div class='chart-frame'>{page.get('original_exposure_chart_svg') or ''}</div></article>"
                if str(page.get("original_exposure_chart_svg") or "").strip() else ""
            )
            + (
                f"<article class='chart-panel'><div class='chart-frame'>{page.get('original_color_chart_svg') or ''}</div></article>"
                if str(page.get("original_color_chart_svg") or "").strip() else ""
            )
            + "</div>"
            "</section>"
            "<section class='grid-section'>"
            "<div class='section-strip'><h2>Adjusted Camera Grid</h2><div class='section-note'>Stored corrected stills</div></div>"
            f"<div class='camera-grid'>{''.join(render_grid_cell(entry, adjusted=True) for entry in page_entries)}</div>"
            "</section>"
            "<section class='synopsis-section'>"
            "<div class='section-strip'><h2>Adjusted Array Synopsis</h2><div class='section-note'>Convergence check</div></div>"
            "<div class='synopsis-grid'>"
            + (
                f"<article class='chart-panel'><div class='chart-frame'>{page.get('adjusted_exposure_chart_svg') or ''}</div></article>"
                if str(page.get("adjusted_exposure_chart_svg") or "").strip() else ""
            )
            + (
                f"<article class='chart-panel'><div class='chart-frame'>{page.get('adjusted_color_chart_svg') or ''}</div></article>"
                if str(page.get("adjusted_color_chart_svg") or "").strip() else ""
            )
            + "</div>"
            "</section>"
            "</section>"
        )

    camera_sections = []
    for entry in list(view_model.get("entries") or []):
        original_image = _contact_sheet_image_src(entry.get("original_image"))
        corrected_image = _contact_sheet_image_src(entry.get("corrected_image"))
        overlay_image = _contact_sheet_image_src(entry.get("overlay_image"))
        corrected_overlay_image = _contact_sheet_image_src(entry.get("corrected_overlay_image"))
        flags = []
        if str(entry.get("reference_use") or "Included") != "Included":
            flags.append(str(entry.get("reference_use")))
        if "fallback" in str(entry.get("sphere_detection_note") or "").lower():
            flags.append("Fallback used")
        if float(entry.get("residual_abs_stops", 0.0) or 0.0) > 0.02:
            flags.append("Residual > ±0.02 goal")
        if str(entry.get("status") or "PASS") != "PASS":
            flags.append(str(entry.get("operator_result_label") or entry.get("status") or "Review"))
        sample_range = max(
            float(entry.get("sample_1_ire", 0.0) or 0.0),
            float(entry.get("sample_2_ire", 0.0) or 0.0),
            float(entry.get("sample_3_ire", 0.0) or 0.0),
        ) - min(
            float(entry.get("sample_1_ire", 0.0) or 0.0),
            float(entry.get("sample_2_ire", 0.0) or 0.0),
            float(entry.get("sample_3_ire", 0.0) or 0.0),
        )
        if sample_range > 3.0:
            flags.append("Uneven sample profile")
        if not flags:
            flags.append("No anomaly flags")
        hero_badge_html = "<div class='hero-badge hero-tag'>Hero Camera</div>" if entry.get("is_hero_camera") else ""
        comparison_figures = []
        for label, src in [
            ("Original", original_image),
            ("Corrected", corrected_image),
        ]:
            comparison_figures.append(
                "<figure class='comparison-figure'>"
                f"<div class='figure-label'>{html.escape(label)}</div>"
                + (f"<img src='{src}' alt='{html.escape(str(entry.get('clip_id') or 'camera'))} {html.escape(label)}' />" if src else "<div class='missing-image'>Image unavailable</div>")
                + 
                "</figure>"
            )
        overlay_markup = (
            "<figure class='overlay-figure'>"
            "<div class='figure-label'>Solve Overlay</div>"
            + (f"<img src='{overlay_image}' alt='{html.escape(str(entry.get('clip_id') or 'camera'))} Solve Overlay' />" if overlay_image else "<div class='missing-image'>Image unavailable</div>")
            + "</figure>"
        )
        camera_sections.append(
            f"<section class='camera-page' data-corrected-overlay='{html.escape(corrected_overlay_image)}'>"
            "<div class='camera-topline'>"
            f"<div class='camera-title'><div class='camera-kicker'>{html.escape(str(entry.get('clip_id') or ''))}</div><h2>{html.escape(str(entry.get('camera_label') or 'Camera'))}</h2>{hero_badge_html}</div>"
            f"<div class='camera-status-line'><span class='status-inline {html.escape(str(entry.get('tone') or 'warning'))}'>{html.escape(str(entry.get('status') or 'REVIEW'))}</span><div class='status-text'>{html.escape(str(entry.get('operator_result_label') or entry.get('status') or 'Review'))}</div></div>"
            f"<div class='camera-action-line'><span>Recommended Action</span><strong>{html.escape(str(entry.get('recommended_action') or 'Review'))}</strong></div>"
            "</div>"
            "<div class='camera-main'>"
            f"<div class='camera-images'>{''.join(comparison_figures)}</div>"
            "<div class='verify-column'>"
            f"{overlay_markup}"
            "<div class='sample-coupling'>"
            "<div class='sample-coupling-title'>Gray Exposure Sample Map</div>"
            f"<div class='sample-coupling-row'><span>Sample 1</span><strong>{float(entry.get('sample_1_ire', 0.0) or 0.0):.1f} IRE</strong></div>"
            f"<div class='sample-coupling-row'><span>Sample 2</span><strong>{float(entry.get('sample_2_ire', 0.0) or 0.0):.1f} IRE</strong></div>"
            f"<div class='sample-coupling-row'><span>Sample 3</span><strong>{float(entry.get('sample_3_ire', 0.0) or 0.0):.1f} IRE</strong></div>"
            "</div>"
            "</div>"
            "</div>"
            "<div class='metric-block'>"
            f"<div><span>Samples</span><strong>S1 {float(entry.get('sample_1_ire', 0.0) or 0.0):.1f} | S2 {float(entry.get('sample_2_ire', 0.0) or 0.0):.1f} | S3 {float(entry.get('sample_3_ire', 0.0) or 0.0):.1f}</strong></div>"
            f"<div><span>Scalar</span><strong>{float(entry.get('display_scalar_ire', 0.0) or 0.0):.1f} IRE</strong></div>"
            f"<div><span>Digital correction applied</span><strong>{float(entry.get('exposure_adjust_stops', 0.0) or 0.0):+.2f} exp | {int(round(float(entry.get('kelvin', 0.0) or 0.0)))}K | {float(entry.get('tint', 0.0) or 0.0):+.1f} tint</strong></div>"
            f"<div><span>Validation Residual</span><strong>{float(entry.get('residual_abs_stops', 0.0) or 0.0):.3f} stops</strong></div>"
            "</div>"
            "<div class='flag-block'>"
            f"<div><span>Reference</span><strong>{html.escape(str(entry.get('reference_use') or 'Included'))}</strong></div>"
            f"<div><span>Sphere Detection</span><strong>{html.escape(str(entry.get('sphere_detection_note') or 'Verified'))}</strong></div>"
            f"<div><span>Profile</span><strong>{html.escape(str(entry.get('profile_note') or 'Profile consistent with stored solve.'))}</strong></div>"
            f"<div><span>Flags</span><strong>{html.escape(' | '.join(flags))}</strong></div>"
            "</div>"
            "</section>"
        )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>R3DMatch Calibration Assessment</title>
  <style>
    :root {{ --ink:#0f172a; --muted:#475569; --soft:#64748b; --paper:#ffffff; --ground:#edf2f7; --line:#d7dee8; --softline:#e7edf4; --pass:#166534; --review:#92400e; --fail:#991b1b; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; background:var(--ground); color:var(--ink); font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }}
    .page {{ max-width:1660px; margin:0 auto; padding:18px; }}
    .shell {{ display:grid; gap:16px; }}
    .contact-page, .camera-page {{ background:var(--paper); border:1px solid var(--line); }}
    .contact-page {{ padding:16px 18px 18px; display:grid; gap:14px; page-break-after:always; }}
    .camera-page {{ padding:16px 18px 18px; display:grid; gap:12px; page-break-before:always; }}
    .sheet-header {{ display:grid; grid-template-columns:auto 1fr; gap:18px; align-items:start; padding-bottom:10px; border-bottom:2px solid var(--ink); }}
    .sheet-brand {{ min-width:120px; }}
    .sheet-brand-placeholder {{ min-width:0; }}
    .brand-logo {{ width:132px; height:auto; object-fit:contain; display:block; }}
    .brand-logo-text {{ font-size:34px; font-weight:800; }}
    .sheet-kicker, .section-note, .camera-kicker, .metric-block span, .flag-block span, .camera-action-line span, .sample-coupling-title, .figure-label {{ font-size:11px; font-weight:800; letter-spacing:.08em; text-transform:uppercase; color:var(--soft); }}
    h1 {{ margin:0; font-size:34px; line-height:1; letter-spacing:-.03em; }}
    h2 {{ margin:0; font-size:22px; line-height:1.1; }}
    .sheet-meta {{ margin-top:8px; display:flex; flex-wrap:wrap; gap:10px 14px; font-size:14px; line-height:1.45; color:var(--muted); }}
    .sheet-meta span {{ white-space:nowrap; }}
    .sheet-guide {{ margin-top:8px; font-size:13px; line-height:1.4; color:var(--muted); }}
    .grid-section, .synopsis-section {{ display:grid; gap:8px; }}
    .section-strip {{ display:flex; justify-content:space-between; align-items:flex-end; gap:16px; padding-bottom:6px; border-bottom:1px solid var(--line); }}
    .camera-grid {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:12px; }}
    .camera-cell {{ display:grid; gap:5px; align-content:start; }}
    .cell-image, .comparison-figure img, .overlay-figure img, .missing-image {{ width:100%; display:block; aspect-ratio:16/10; object-fit:cover; background:#dce3ec; border:1px solid var(--line); }}
    .missing-image {{ display:grid; place-items:center; color:var(--soft); font-size:16px; }}
    .cell-line {{ font-size:13px; line-height:1.35; }}
    .cell-clip {{ font-weight:700; }}
    .cell-meta {{ color:var(--muted); white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
    .cell-totals-label {{ font-size:11px; font-weight:800; letter-spacing:.08em; text-transform:uppercase; color:var(--soft); }}
    .cell-totals {{ font-size:14px; font-weight:700; letter-spacing:.01em; }}
    .synopsis-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:12px; }}
    .chart-panel {{ border:1px solid var(--line); padding:8px; }}
    .chart-frame svg {{ display:block; width:100%; height:auto; }}
    .camera-topline {{ display:grid; grid-template-columns:1.2fr auto 1fr; gap:16px; align-items:end; padding-bottom:10px; border-bottom:2px solid var(--ink); }}
    .camera-title h2 {{ margin-top:4px; font-size:30px; }}
    .hero-tag, .hero-badge {{ margin-top:6px; font-size:11px; font-weight:800; letter-spacing:.08em; text-transform:uppercase; color:#991b1b; }}
    .status-inline {{ display:inline-block; padding:6px 12px; border:1px solid currentColor; font-size:13px; font-weight:800; letter-spacing:.08em; text-transform:uppercase; }}
    .status-inline.good {{ color:var(--pass); }}
    .status-inline.warning {{ color:var(--review); }}
    .status-inline.danger {{ color:var(--fail); }}
    .camera-status-line {{ justify-self:center; text-align:center; }}
    .status-text {{ margin-top:6px; font-size:14px; color:var(--muted); }}
    .camera-action-line {{ justify-self:end; text-align:right; }}
    .camera-action-line strong {{ display:block; margin-top:4px; font-size:20px; line-height:1.2; }}
    .camera-main {{ display:grid; grid-template-columns:1.55fr .85fr; gap:14px; align-items:start; }}
    .camera-images {{ display:grid; grid-template-columns:1fr 1fr; gap:12px; }}
    .comparison-figure, .overlay-figure {{ margin:0; display:grid; gap:8px; }}
    .verify-column {{ display:grid; gap:8px; }}
    .sample-coupling {{ border-top:1px solid var(--ink); border-bottom:1px solid var(--line); }}
    .sample-coupling-row {{ display:flex; justify-content:space-between; gap:12px; padding:8px 0; border-top:1px solid var(--softline); font-size:15px; }}
    .sample-coupling-row:first-of-type {{ border-top:0; }}
    .sample-coupling-row strong {{ font-size:18px; }}
    .metric-block, .flag-block {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:0; border-top:1px solid var(--ink); border-bottom:1px solid var(--line); }}
    .metric-block > div, .flag-block > div {{ padding:10px 10px 10px 0; border-right:1px solid var(--softline); }}
    .metric-block > div:last-child, .flag-block > div:last-child {{ border-right:0; padding-right:0; }}
    .metric-block strong, .flag-block strong {{ display:block; margin-top:5px; font-size:18px; line-height:1.35; }}
    .flag-block strong {{ font-size:15px; }}
    @media (max-width:1280px) {{ .synopsis-grid,.camera-main,.metric-block,.flag-block {{ grid-template-columns:1fr; }} .camera-grid {{ grid-template-columns:repeat(2,minmax(0,1fr)); }} .camera-topline {{ grid-template-columns:1fr; align-items:start; }} .camera-action-line {{ justify-self:start; text-align:left; }} }}
    @media (max-width:860px) {{ .page {{ padding:10px; }} .camera-grid,.camera-images,.synopsis-grid,.camera-main,.metric-block,.flag-block,.sheet-header {{ grid-template-columns:1fr; }} .sheet-meta span,.cell-meta {{ white-space:normal; }} h1 {{ font-size:28px; }} .camera-title h2 {{ font-size:24px; }} }}
    @media print {{ body {{ background:white; }} .page {{ padding:6px; }} .contact-page,.camera-page {{ break-inside:avoid; }} }}
  </style>
</head>
<body>
  <div class="page">
    <div class="shell">
      {''.join(contact_pages_markup)}
      {''.join(camera_sections)}
    </div>
  </div>
</body>
</html>"""
