from __future__ import annotations

import copy
import hashlib
import html
import json
import logging
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
import textwrap
from urllib.parse import unquote, urlparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw
from scipy.optimize import least_squares
from skimage import color, feature, measure, morphology, transform

try:
    import cv2
except Exception:  # pragma: no cover - optional runtime dependency
    cv2 = None

try:
    import tifffile
except Exception:  # pragma: no cover - optional runtime dependency
    tifffile = None

from .calibration import (
    _measure_pixel_cloud_statistics,
    _sphere_zone_geometries,
    build_sphere_sampling_mask,
    compute_luminance,
    detect_gray_card_roi,
    extract_rect_pixels,
    load_color_calibration,
    load_exposure_calibration,
    measure_sphere_region_statistics,
    measure_sphere_zone_profile_statistics,
    percentile_clip,
    trimmed_luminance_measurement,
)
from .color import identity_lggs, is_identity_cdl_payload as _color_is_identity_cdl_payload, rgb_gains_to_cdl, solve_cdl_color_model
from .commit_values import build_commit_values, extract_as_shot_white_balance, solve_white_balance_model_for_records
from .execution import CancellationError, raise_if_cancelled, run_cancellable_subprocess
from .ftps_ingest import source_mode_label
from .matching import _measure_three_sample_statistics
from .models import GrayCardROI, SphereROI
from .progress import emit_review_progress
from .rmd import write_rmd_for_clip_with_metadata, write_rmds_from_analysis
from .sdk import resolve_backend

LOGGER = logging.getLogger(__name__)

PREVIEW_VARIANTS = ("original", "exposure", "color", "both")
REVIEW_PREVIEW_TRANSFORM = "REDLine IPP2 / BT.709 / BT.1886 / Medium / Medium"
DEFAULT_REVIEW_TARGET_STRATEGIES = ("median",)
STRATEGY_ORDER = ["median", "optimal_exposure", "manual", "hero_camera", "manual_target"]
SPHERE_DETECTION_MAX_DIMENSION = 1280
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
    "preview_still_format": "tiff",
}
DEFAULT_MONITORING_PREVIEW = {
    "preview_mode": "monitoring",
    "output_space": "BT.709",
    "output_gamma": "BT.1886",
    "highlight_rolloff": "medium",
    "shadow_rolloff": "medium",
    "lut_path": None,
    "preview_still_format": "tiff",
}
DEFAULT_DISPLAY_REVIEW_PREVIEW = {
    "preview_mode": "monitoring",
    "output_space": "BT.709",
    "output_gamma": "BT.1886",
    "highlight_rolloff": "medium",
    "shadow_rolloff": "medium",
    "lut_path": None,
    "preview_still_format": "tiff",
}
PREVIEW_STILL_FORMAT_CODES = {"tiff": 1, "jpeg": 3}
PREVIEW_STILL_FORMAT_EXTENSIONS = {"tiff": "tiff", "jpeg": "jpg"}
DEFAULT_PREVIEW_STILL_FORMAT = "tiff"
DEFAULT_ARTIFACT_MODE = "production"
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
REPORT_FOCUS_LABELS = {
    "auto": "Auto",
    "full": "All Cameras",
    "outliers": "Outliers Only",
    "anchors": "Exposure Anchors",
    "cluster_extremes": "Cluster Extremes",
}
ARTIFACT_MODE_LABELS = {
    "production": "Production",
    "debug": "Debug",
}
LARGE_ARRAY_OVERVIEW_THRESHOLD = 18
DEFAULT_LARGE_ARRAY_AUTO_FOCUS = "outliers"
OPTIMAL_EXPOSURE_MIN_CONFIDENCE = 0.45
OPTIMAL_EXPOSURE_MAX_SAMPLE_LOG2_SPREAD = 0.12
OPTIMAL_EXPOSURE_MAX_SAMPLE_CHROMA_SPREAD = 0.02
OPTIMAL_EXPOSURE_PRIMARY_CLUSTER_GAP = 0.25
CAMERA_TRUST_CAUTION_CONFIDENCE = 0.65
CAMERA_TRUST_CAUTION_LOG2_SPREAD = 0.08
SPHERE_DETECTION_HIGH_CONFIDENCE = 0.72
SPHERE_DETECTION_MEDIUM_CONFIDENCE = 0.50
SPHERE_DETECTION_LOW_CONFIDENCE = 0.30
SPHERE_DETECTION_PLAUSIBILITY_MIN = 0.46
SPHERE_DETECTION_PROFILE_REVIEW = 0.60
SPHERE_DETECTION_PROFILE_HIGH = 0.78
SPHERE_NEUTRAL_SEED_MAX_CHROMA_DISTANCE = 0.072
SPHERE_NEUTRAL_SEED_MIN_COMPONENT_AREA_FRACTION = 0.018
SPHERE_NEUTRAL_EXPANSION_MIN_RING_FRACTION = 0.28
SPHERE_NEUTRAL_SEED_MIN_LUMINANCE = 0.085
SPHERE_NEUTRAL_SEED_MIN_CHANNEL_SUM = 0.18
SPHERE_NEUTRAL_SEED_MAX_FRAGMENT_COUNT = 3
SPHERE_NEUTRAL_REGION_MIN_SOLIDITY = 0.82
SPHERE_SHAPE_MIN_CIRCULARITY = 0.52
SPHERE_SHAPE_MAX_ASPECT_RATIO = 1.24
SPHERE_SHAPE_MAX_RADIAL_CV = 0.42
SPHERE_SHAPE_MAX_CENTROID_DISTANCE_NORM = 0.34
SPHERE_SHAPE_PROXY_MIN_GEOMETRY_CIRCULARITY = 0.55
SPHERE_SHAPE_PROXY_MAX_ASPECT_RATIO = 1.30
SPHERE_SHAPE_PROXY_MIN_SOLIDITY = 0.70
SPHERE_SHAPE_PROXY_MAX_CENTROID_DISTANCE_NORM = 0.24
SPHERE_SHAPE_PROXY_MIN_REGION_EXPANSION = 0.55
SPHERE_SHAPE_PROXY_MIN_NEUTRAL_CONSISTENCY = 0.55
SPHERE_HARD_GATE_MIN_ZONE_SPREAD_IRE = 1.4
SPHERE_HARD_GATE_MAX_CHROMA_RANGE = 0.085
SPHERE_HARD_GATE_MIN_VALID_PIXEL_COUNT = 600
SPHERE_HARD_GATE_MIN_VALID_PIXEL_FRACTION = 0.075
SPHERE_HARD_GATE_MIN_RADIUS_RATIO = 0.08
SPHERE_HARD_GATE_CENTER_IRE_MIN = 8.0
SPHERE_HARD_GATE_CENTER_IRE_MAX = 78.0
SPHERE_HARD_GATE_MIN_INTERIOR_STDDEV = 0.002
SPHERE_HARD_GATE_MAX_INTERIOR_STDDEV = 0.170
SPHERE_HARD_GATE_CENTER_EXTREMUM_MARGIN_IRE = 0.75
SPHERE_HARD_GATE_MIN_MEASUREMENT_CONFIDENCE = 0.46
SPHERE_HARD_GATE_MIN_RADIAL_COHERENCE = 0.56
SPHERE_HARD_GATE_MIN_NEUTRAL_CONSISTENCY = 0.42
SPHERE_HARD_GATE_MIN_REGION_EXPANSION = 0.34
SPHERE_HARD_GATE_MIN_SHAPE_SCORE = 0.48
SPHERE_HARD_GATE_MIN_EXPOSURE_VALIDITY = 0.44
SPHERE_HARD_GATE_MIN_RADIAL_PROFILE_SCORE = 0.26
SPHERE_HARD_GATE_MIN_ROI_AREA_FRACTION = 0.010
SPHERE_HARD_GATE_MIN_GEOMETRY_SCORE = 0.40
SPHERE_HARD_GATE_MAX_SAMPLE_EQUALITY_IRE = 0.55
SPHERE_HARD_GATE_HIGH_FLAT_IRE = 65.0
SPHERE_HARD_GATE_LOW_FLAT_IRE = 4.5
SPHERE_HARD_GATE_MIN_EDGE_SUPPORT = 0.28
SPHERE_HARD_GATE_MAX_FIT_RESIDUAL_RATIO = 0.060
SPHERE_EDGE_SUPPORT_MIN_GRADIENT = 0.004
SPHERE_EDGE_SUPPORT_GRADIENT_QUANTILE = 0.60
SPHERE_REFINED_GEOMETRY_OVERRIDE_MIN_EDGE_SUPPORT = 0.34
SPHERE_REFINED_GEOMETRY_OVERRIDE_MAX_FIT_RESIDUAL_RATIO = 0.012
SPHERE_REFINED_GEOMETRY_OVERRIDE_MIN_RADIAL_COHERENCE = 0.70
SPHERE_REFINED_GEOMETRY_OVERRIDE_MIN_RADIAL_PROFILE = 0.65
SPHERE_REFINED_GEOMETRY_OVERRIDE_MIN_NEUTRAL_CONSISTENCY = 0.58
SPHERE_REFINED_GEOMETRY_OVERRIDE_MIN_REGION_EXPANSION = 0.38
SPHERE_MEASUREMENT_RADIUS_RATIO = 0.80
SPHERE_HERO_PATCH_RADIUS_RATIO = 0.08
SPHERE_HERO_PATCH_MIN_PIXEL_COUNT = 64
SPHERE_HERO_PATCH_TRIM_FRACTION = 0.10
SPHERE_CANDIDATE_REFINEMENT_TOP_N = 10
SPHERE_CANDIDATE_REFINEMENT_MAX_EDGE_POINTS = 320
SPHERE_CANDIDATE_REFINEMENT_EDGE_BINS = 48
SPHERE_CANDIDATE_REFINEMENT_MAX_NFEV = 40
SPHERE_CANDIDATE_REFINEMENT_RANSAC_TRIALS = 12
SPHERE_CANDIDATE_REFINEMENT_INLIER_TOLERANCE = 3.2
SPHERE_CANDIDATE_DEDUP_DISTANCE_FACTOR = 0.34
SPHERE_CANDIDATE_DEDUP_RADIUS_FACTOR = 0.20
MANUAL_ASSIST_LOCAL_SHIFT_FACTORS = (-0.30, -0.15, 0.0, 0.15, 0.30)
MANUAL_ASSIST_LOCAL_RADIUS_FACTORS = (0.85, 1.0, 1.15)
PREVIEW_RENDER_MIN_FILE_SIZE_BYTES = 1024
PREVIEW_RENDER_MIN_TIFF_SIZE_BYTES = 4096
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
SPHERE_DETECTION_FALLBACK_SOURCES = {
    "secondary_detected",
    "localized_recovery",
    "neutral_blob_recovery",
    "opencv_hough_recovery",
    "opencv_hough_alt_recovery",
    "opencv_contour_recovery",
    "card_adjacent_recovery",
    "manual_operator_assist",
    "manual_operator_assist_refined",
    "reused_from_original",
}
PROFILE_AUDIT_CONSISTENT_STOPS = 0.10
PROFILE_AUDIT_REVIEW_STOPS = 0.25
PROFILE_AUDIT_CONSISTENT_SPREAD_IRE = 1.5
PROFILE_AUDIT_REVIEW_SPREAD_IRE = 3.0
CONTACT_SHEET_TARGET_SAMPLE_LABEL = SPHERE_PROFILE_ZONE_DISPLAY["center"]
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


def normalize_report_focus(value: Optional[str]) -> str:
    normalized = str(value or "auto").strip().lower()
    aliases = {
        "": "auto",
        "auto": "auto",
        "default": "auto",
        "full": "full",
        "all": "full",
        "all_cameras": "full",
        "all-cameras": "full",
        "outliers": "outliers",
        "outliers_only": "outliers",
        "outliers-only": "outliers",
        "anchors": "anchors",
        "references": "anchors",
        "anchors_references": "anchors",
        "anchors-references": "anchors",
        "cluster_extremes": "cluster_extremes",
        "cluster-extremes": "cluster_extremes",
        "extremes": "cluster_extremes",
    }
    if normalized not in aliases:
        raise ValueError("report focus must be auto, full, outliers, anchors, or cluster_extremes")
    return aliases[normalized]


def report_focus_label(value: Optional[str]) -> str:
    return REPORT_FOCUS_LABELS[normalize_report_focus(value)]


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


def _numeric_value_or_none(value: object) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _measurement_valid_flag(payload: Optional[Dict[str, object]]) -> bool:
    if not payload:
        return False
    explicit = payload.get("measurement_valid")
    if explicit is None:
        explicit = payload.get("gray_target_measurement_valid")
    if explicit is not None:
        return bool(explicit)
    if bool(payload.get("detection_failed")) or bool(payload.get("sphere_detection_unresolved")):
        return False
    gray_target_class = str(payload.get("gray_target_class") or "").strip().lower()
    if gray_target_class == "unresolved":
        return False
    if payload.get("sphere_detection_success") is False and gray_target_class != "gray_card":
        return False
    resolved_log2 = _numeric_value_or_none(
        payload.get(
            "measured_log2_luminance_monitoring",
            payload.get("measured_log2_luminance"),
        )
    )
    return resolved_log2 is not None


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
                "measurement_valid": diagnostics.get("measurement_valid", diagnostics.get("gray_target_measurement_valid")),
                "measured_log2_luminance_monitoring": diagnostics.get("measured_log2_luminance_monitoring", diagnostics.get("measured_log2_luminance")),
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
        measurement_valid = _measurement_valid_flag(monitoring or diagnostics)
        zone_measurements = [dict(item) for item in list(monitoring.get("zone_measurements") or [])]
        if not zone_measurements:
            zone_measurements = [dict(item) for item in list(diagnostics.get("zone_measurements") or diagnostics.get("neutral_samples") or [])]
        sample_display_scalar = _sample_scalar_display_from_profile(zone_measurements)
        resolved_log2 = None
        resolved_ire = None
        if measurement_valid:
            resolved_log2 = (
                float(sample_display_scalar.get("sample_scalar_display_log2", 0.0) or 0.0)
                if float(sample_display_scalar.get("sample_count", 0.0) or 0.0) > 0.0
                else float(
                    monitoring.get(
                        "measured_log2_luminance_monitoring",
                        diagnostics.get("measured_log2_luminance_monitoring", diagnostics.get("measured_log2_luminance")),
                    )
                )
            )
            resolved_ire = (
                float(sample_display_scalar.get("sample_scalar_display_ire", 0.0) or 0.0)
                if float(sample_display_scalar.get("sample_count", 0.0) or 0.0) > 0.0
                else float(_ire_from_log2_luminance(float(resolved_log2)))
            )
        return {
            "measurement_valid": measurement_valid,
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
        "measurement_valid": True,
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
    measurement_valid = _measurement_valid_flag(diagnostics)
    resolved_log2 = (
        float(sample_display_scalar.get("sample_scalar_display_log2", 0.0) or 0.0)
        if measurement_valid and float(sample_display_scalar.get("sample_count", 0.0) or 0.0) > 0.0
        else _numeric_value_or_none(diagnostics.get("measured_log2_luminance_monitoring", diagnostics.get("measured_log2_luminance")))
        if measurement_valid
        else None
    )
    return {
        "measurement_valid": measurement_valid,
        "sample_scalar_display_log2": resolved_log2,
        "sample_scalar_display_ire": (
            float(sample_display_scalar.get("sample_scalar_display_ire", 0.0) or 0.0)
            if measurement_valid and float(sample_display_scalar.get("sample_count", 0.0) or 0.0) > 0.0
            else float(_ire_from_log2_luminance(float(resolved_log2)))
            if measurement_valid and resolved_log2 is not None
            else None
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
    if not _measurement_valid_flag(measurement):
        return {
            "confidence": 0.0,
            "neutral_sample_log2_spread": 0.0,
            "neutral_sample_chromaticity_spread": 0.0,
            "flags": ["measurement_invalid"],
        }
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
        preview_still_format=str(defaults.get("preview_still_format") or DEFAULT_PREVIEW_STILL_FORMAT),
    )


def normalize_preview_still_format(value: Optional[str]) -> str:
    normalized = str(value or DEFAULT_PREVIEW_STILL_FORMAT).strip().lower().replace(".", "")
    aliases = {
        "tif": "tiff",
        "tiff": "tiff",
        "jpg": "jpeg",
        "jpeg": "jpeg",
    }
    if normalized not in aliases:
        raise ValueError("preview still format must be tiff or jpeg")
    return aliases[normalized]


def preview_still_extension(preview_still_format: Optional[str]) -> str:
    return PREVIEW_STILL_FORMAT_EXTENSIONS[normalize_preview_still_format(preview_still_format)]


def prepare_manual_sphere_assist_bundle(
    analysis_root: str,
    *,
    out_dir: Optional[str] = None,
    clip_ids: Optional[List[str]] = None,
) -> Dict[str, object]:
    root = Path(analysis_root).expanduser().resolve()
    report_root = root / "report"
    analysis_dir = root / "analysis"
    if not analysis_dir.exists():
        raise FileNotFoundError(f"Analysis directory is missing: {analysis_dir}")
    report_payload = {}
    contact_sheet_path = report_root / "contact_sheet.json"
    if contact_sheet_path.exists():
        report_payload = json.loads(contact_sheet_path.read_text(encoding="utf-8"))
    per_camera_map = {
        str(item.get("clip_id") or "").strip(): dict(item)
        for item in list(report_payload.get("per_camera_analysis") or [])
        if str(item.get("clip_id") or "").strip()
    }
    requested_clip_ids = {str(item).strip() for item in (clip_ids or []) if str(item).strip()}
    assist_root = Path(out_dir).expanduser().resolve() if str(out_dir or "").strip() else report_root / "manual_sphere_assist"
    previews_dir = assist_root / "previews"
    previews_dir.mkdir(parents=True, exist_ok=True)
    resample = getattr(Image, "Resampling", Image).LANCZOS
    entries: List[Dict[str, object]] = []
    for analysis_path in sorted(analysis_dir.glob("*.analysis.json")):
        record = json.loads(analysis_path.read_text(encoding="utf-8"))
        clip_id = str(record.get("clip_id") or "").strip()
        if not clip_id:
            continue
        if requested_clip_ids and clip_id not in requested_clip_ids:
            continue
        per_camera = dict(per_camera_map.get(clip_id) or {})
        confidence = float(
            per_camera.get("gray_target_confidence")
            or per_camera.get("confidence")
            or record.get("confidence")
            or 0.0
        )
        trust_class = str(per_camera.get("trust_class") or "").strip().upper()
        reference_use = str(per_camera.get("reference_use") or "").strip()
        review_required = bool(
            per_camera.get("gray_target_review_recommended")
            or record.get("diagnostics", {}).get("gray_target_review_recommended")
        )
        selected_for_assist = (
            not per_camera_map
            or reference_use == "Excluded"
            or trust_class in {"EXCLUDED", "UNTRUSTED"}
            or review_required
            or confidence < 0.90
        )
        if not selected_for_assist:
            continue
        diagnostics = dict(record.get("diagnostics") or {})
        measurement_source_image = Path(
            str(
                diagnostics.get("rendered_measurement_preview_path")
                or ((diagnostics.get("runtime_trace") or {}).get("rendered_image_path"))
                or ""
            )
        ).expanduser().resolve()
        if not measurement_source_image.exists():
            continue
        preview_output = previews_dir / f"{clip_id}.manual_sphere_assist.jpg"
        with Image.open(measurement_source_image) as image:
            render_width, render_height = int(image.width), int(image.height)
            preview = image.convert("RGB")
            preview.thumbnail((1280, 1280), resample)
            preview.save(preview_output, format="JPEG", quality=72, optimize=True, progressive=True)
            preview_width, preview_height = int(preview.width), int(preview.height)
        detected_roi = dict(diagnostics.get("detected_sphere_roi") or {})
        estimated_radius_normalized = None
        estimated_radius_preview_px = None
        if detected_roi and min(render_width, render_height) > 0:
            estimated_radius_normalized = float(detected_roi.get("r", 0.0) or 0.0) / float(min(render_width, render_height))
            estimated_radius_preview_px = float(estimated_radius_normalized) * float(min(preview_width, preview_height))
        entries.append(
            {
                "clip_id": clip_id,
                "source_path": str(record.get("source_path") or ""),
                "source_image": str(preview_output),
                "source_image_hash": _manual_assist_preview_hash(preview_output),
                "measurement_source_image": str(measurement_source_image),
                "measurement_source_hash": _manual_assist_preview_hash(measurement_source_image),
                "preview_width": preview_width,
                "preview_height": preview_height,
                "render_width": render_width,
                "render_height": render_height,
                "center_preview_px": None,
                "radius_preview_px": None,
                "center_normalized": None,
                "radius_normalized": None,
                "estimated_radius_preview_px": estimated_radius_preview_px,
                "estimated_radius_normalized": estimated_radius_normalized,
                "auto_detected_sphere_roi": detected_roi,
                "trust_class": trust_class or "UNKNOWN",
                "reference_use": reference_use or "Included",
                "gray_target_confidence": confidence,
                "reason": str(per_camera.get("trust_reason") or "Manual sphere assist requested"),
            }
        )
    manifest = {
        "schema_version": "r3dmatch_sphere_assist_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "analysis_root": str(root),
        "entry_count": len(entries),
        "entries": entries,
    }
    manifest_path = assist_root / "sphere_assist_template.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {
        "analysis_root": str(root),
        "assist_root": str(assist_root),
        "manifest_path": str(manifest_path),
        "preview_dir": str(previews_dir),
        "entry_count": len(entries),
        "clip_ids": [str(item.get("clip_id") or "") for item in entries],
    }


def normalize_artifact_mode(value: Optional[str]) -> str:
    normalized = str(value or DEFAULT_ARTIFACT_MODE).strip().lower().replace("-", "_")
    if normalized not in ARTIFACT_MODE_LABELS:
        raise ValueError("artifact mode must be production or debug")
    return normalized


def artifact_mode_label(value: Optional[str]) -> str:
    return ARTIFACT_MODE_LABELS[normalize_artifact_mode(value)]


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


def _focus_overlay_path(overlay_root: Path, clip_id: str) -> Path:
    safe_clip_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(clip_id or "camera"))
    return overlay_root / f"{safe_clip_id}.focus_validation.jpg"


def _write_focus_validation_overlay(
    *,
    source_image_path: str,
    output_path: Path,
    detection: Dict[str, object],
) -> str:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source_image_path) as image:
        overlay = image.convert("RGB")
        draw = ImageDraw.Draw(overlay)
        for box in list(detection.get("cell_boxes") or []):
            draw.rectangle(
                [
                    int(box.get("x0", 0) or 0),
                    int(box.get("y0", 0) or 0),
                    int(box.get("x1", 0) or 0),
                    int(box.get("y1", 0) or 0),
                ],
                outline=(48, 196, 106),
                width=8,
            )
        chart_box = dict(detection.get("colorchecker_bbox") or {})
        if chart_box:
            draw.rectangle(
                [
                    int(chart_box.get("x0", 0) or 0),
                    int(chart_box.get("y0", 0) or 0),
                    int(chart_box.get("x1", 0) or 0),
                    int(chart_box.get("y1", 0) or 0),
                ],
                outline=(59, 130, 246),
                width=10,
            )
        roi = dict(detection.get("roi") or {})
        if roi:
            draw.rectangle(
                [
                    int(roi.get("x0", 0) or 0),
                    int(roi.get("y0", 0) or 0),
                    int(roi.get("x1", 0) or 0),
                    int(roi.get("y1", 0) or 0),
                ],
                outline=(245, 158, 11),
                width=12,
            )
        overlay.save(output_path, format="JPEG", quality=88, optimize=True, progressive=True)
    return str(output_path)


def _crop_focus_roi(region: np.ndarray, roi_payload: Dict[str, object]) -> np.ndarray:
    height, width = region.shape[:2]
    roi = dict(roi_payload or {})
    x0 = max(0, min(width - 1, int(roi.get("x0", 0) or 0)))
    y0 = max(0, min(height - 1, int(roi.get("y0", 0) or 0)))
    x1 = max(x0 + 1, min(width, int(roi.get("x1", width) or width)))
    y1 = max(y0 + 1, min(height, int(roi.get("y1", height) or height)))
    return np.asarray(region[y0:y1, x0:x1, :], dtype=np.float32)


def _crop_focus_roi_from_normalized(region: np.ndarray, normalized_roi: Dict[str, object]) -> np.ndarray:
    height, width = region.shape[:2]
    roi = dict(normalized_roi or {})
    x0 = max(0, min(width - 1, int(round(float(roi.get("x", 0.0) or 0.0) * width))))
    y0 = max(0, min(height - 1, int(round(float(roi.get("y", 0.0) or 0.0) * height))))
    x1 = max(x0 + 1, min(width, int(round((float(roi.get("x", 0.0) or 0.0) + float(roi.get("w", 1.0) or 1.0)) * width))))
    y1 = max(y0 + 1, min(height, int(round((float(roi.get("y", 0.0) or 0.0) + float(roi.get("h", 1.0) or 1.0)) * height))))
    return np.asarray(region[y0:y1, x0:x1, :], dtype=np.float32)


def _decode_focus_reference_region(
    *,
    source_path: str,
    normalized_roi: Dict[str, object],
) -> Tuple[np.ndarray, Dict[str, object]]:
    backend = resolve_backend("red")
    decoded = list(backend.decode_frames(str(source_path), start_frame=0, max_frames=1, frame_step=1, half_res=True))
    if not decoded:
        raise RuntimeError(f"Unable to decode focus reference frame for {source_path}")
    _frame_index, _timecode, image = decoded[0]
    hwc = np.moveaxis(np.asarray(image, dtype=np.float32), 0, -1)
    cropped = _crop_focus_roi_from_normalized(hwc, normalized_roi)
    return cropped, {
        "backend": "red_sdk_half_res",
        "width": int(hwc.shape[1]),
        "height": int(hwc.shape[0]),
    }


def _focus_classification_from_ratio(ratio_to_best: float) -> str:
    value = float(ratio_to_best)
    if value >= 0.88:
        return "Sharp"
    if value >= 0.68:
        return "Review"
    return "Soft"


def _focus_validation_summary(
    *,
    rows: List[Dict[str, object]],
    reference_rows: List[Dict[str, object]],
) -> Dict[str, object]:
    if not rows:
        return {
            "enabled": True,
            "status": "no_rows",
            "tiff_is_sufficient": False,
            "reason": "No focus rows were generated.",
            "confidence": "high",
            "rows": [],
        }
    tiff_scores = [float(row.get("composite_focus_score", 0.0) or 0.0) for row in rows]
    reference_scores = [float(row.get("composite_focus_score", 0.0) or 0.0) for row in reference_rows] if reference_rows else []
    tiff_best = max(tiff_scores) if tiff_scores else 1.0
    reference_best = max(reference_scores) if reference_scores else 1.0
    reference_map = {str(item.get("clip_id") or ""): dict(item) for item in reference_rows}
    classification_matches = 0
    comparable_rows = 0
    for row in rows:
        ratio_to_best = float(row.get("composite_focus_score", 0.0) or 0.0) / max(float(tiff_best), 1e-6)
        row["normalized_focus_ratio"] = float(ratio_to_best)
        row["focus_classification"] = _focus_classification_from_ratio(ratio_to_best)
        reference = reference_map.get(str(row.get("clip_id") or ""))
        if reference:
            ref_ratio = float(reference.get("composite_focus_score", 0.0) or 0.0) / max(float(reference_best), 1e-6)
            reference["normalized_focus_ratio"] = float(ref_ratio)
            reference["focus_classification"] = _focus_classification_from_ratio(ref_ratio)
            comparable_rows += 1
            if str(reference["focus_classification"]) == str(row["focus_classification"]):
                classification_matches += 1
    rank_corr = _spearman_rank_correlation(tiff_scores, reference_scores) if len(reference_scores) == len(tiff_scores) and len(tiff_scores) >= 2 else 1.0
    metric_delta_rows = []
    for row in rows:
        reference = reference_map.get(str(row.get("clip_id") or ""))
        if not reference:
            continue
        metric_delta_rows.append(
            {
                "laplacian_variance": abs(float(row.get("laplacian_variance", 0.0) or 0.0) - float(reference.get("laplacian_variance", 0.0) or 0.0)) / max(abs(float(reference.get("laplacian_variance", 0.0) or 0.0)), 1e-6) * 100.0,
                "tenengrad": abs(float(row.get("tenengrad", 0.0) or 0.0) - float(reference.get("tenengrad", 0.0) or 0.0)) / max(abs(float(reference.get("tenengrad", 0.0) or 0.0)), 1e-6) * 100.0,
                "fft_high_frequency_energy": abs(float(row.get("fft_high_frequency_energy", 0.0) or 0.0) - float(reference.get("fft_high_frequency_energy", 0.0) or 0.0)) / max(abs(float(reference.get("fft_high_frequency_energy", 0.0) or 0.0)), 1e-6) * 100.0,
            }
        )
    median_abs_delta_percent = {
        "laplacian_variance": float(np.median([item["laplacian_variance"] for item in metric_delta_rows])) if metric_delta_rows else 0.0,
        "tenengrad": float(np.median([item["tenengrad"] for item in metric_delta_rows])) if metric_delta_rows else 0.0,
        "fft_high_frequency_energy": float(np.median([item["fft_high_frequency_energy"] for item in metric_delta_rows])) if metric_delta_rows else 0.0,
    }
    classification_agreement = float(classification_matches) / float(max(comparable_rows, 1))
    tiff_is_sufficient = bool(rank_corr >= 0.95 and classification_agreement >= 0.99)
    status = "sufficient" if tiff_is_sufficient else "insufficient"
    reason = (
        f"TIFF preserved focus ordering with rank correlation {rank_corr:.3f} and class agreement {classification_agreement:.2%} "
        "against the available RED SDK half-resolution reference."
    )
    return {
        "enabled": True,
        "status": status,
        "rows": rows,
        "reference_rows": reference_rows,
        "rank_correlation": float(rank_corr),
        "classification_agreement": float(classification_agreement),
        "median_abs_delta_percent": median_abs_delta_percent,
        "tiff_is_sufficient": bool(tiff_is_sufficient),
        "reason": reason,
        "measured_error": (
            f"rank_correlation={rank_corr:.3f}; class_agreement={classification_agreement:.2%}; "
            f"median_abs_delta_percent(lap={median_abs_delta_percent['laplacian_variance']:.2f}, "
            f"ten={median_abs_delta_percent['tenengrad']:.2f}, fft={median_abs_delta_percent['fft_high_frequency_energy']:.2f})"
        ),
        "confidence": "high",
        "reference_basis": "red_sdk_half_res",
    }


def _build_focus_validation_artifacts(
    *,
    out_root: Path,
    per_camera_rows: List[Dict[str, object]],
    shared_originals: List[Dict[str, object]],
    progress_path: Optional[str] = None,
) -> Dict[str, object]:
    overlay_root = out_root / "focus_validation_overlays"
    overlay_root.mkdir(parents=True, exist_ok=True)
    shared_map = {
        str(item.get("clip_id") or ""): dict(item)
        for item in list(shared_originals or [])
        if str(item.get("clip_id") or "").strip()
    }
    focus_rows: List[Dict[str, object]] = []
    reference_rows: List[Dict[str, object]] = []
    for index, row in enumerate(list(per_camera_rows or []), start=1):
        clip_id = str(row.get("clip_id") or "")
        if not clip_id:
            continue
        shared = shared_map.get(clip_id) or {}
        source_image_path = str(shared.get("original_frame") or row.get("original_frame") or "")
        source_path = str(row.get("source_path") or shared.get("source_path") or "")
        if not source_image_path or not Path(source_image_path).exists():
            continue
        emit_review_progress(
            progress_path,
            phase="focus_validation_clip",
            detail=f"Evaluating focus target for {clip_id}.",
            stage_label="Focus validation",
            clip_index=index,
            clip_count=len(per_camera_rows),
        )
        image, metadata = _load_preview_image_as_normalized_rgb(source_image_path)
        detection = _detect_focus_chart_roi_hwc(image)
        if not bool(detection.get("found")):
            focus_rows.append(
                {
                    "clip_id": clip_id,
                    "camera_label": _camera_label_for_reporting(clip_id),
                    "status": "unresolved",
                    "focus_classification": "Review",
                    "detection_method": str(detection.get("method") or "opencv_colorchecker_cluster"),
                    "detection_confidence": float(detection.get("confidence", 0.0) or 0.0),
                    "reason": str(detection.get("reason") or "Chart was not detected."),
                    "source_image": source_image_path,
                }
            )
            continue
        focus_region = _crop_focus_roi(image, dict(detection.get("roi") or {}))
        metrics = _focus_metric_bundle(focus_region)
        overlay_path = _write_focus_validation_overlay(
            source_image_path=source_image_path,
            output_path=_focus_overlay_path(overlay_root, clip_id),
            detection=detection,
        )
        row_payload = {
            "clip_id": clip_id,
            "camera_label": _camera_label_for_reporting(clip_id),
            "status": "resolved",
            "source_image": source_image_path,
            "source_path": source_path,
            "overlay_image": overlay_path,
            "detection_method": str(detection.get("method") or "opencv_colorchecker_cluster"),
            "detection_confidence": float(detection.get("confidence", 0.0) or 0.0),
            "focus_roi": dict(detection.get("roi") or {}),
            "reference_basis": "tiff_preview",
            "image_width": int(metadata.get("width", image.shape[1])),
            "image_height": int(metadata.get("height", image.shape[0])),
            **metrics,
        }
        focus_rows.append(row_payload)
        if source_path:
            try:
                reference_region, reference_meta = _decode_focus_reference_region(
                    source_path=source_path,
                    normalized_roi=dict((detection.get("roi") or {}).get("normalized") or {}),
                )
                reference_metrics = _focus_metric_bundle(reference_region)
                reference_rows.append(
                    {
                        "clip_id": clip_id,
                        "camera_label": _camera_label_for_reporting(clip_id),
                        "status": "resolved",
                        "reference_basis": str(reference_meta.get("backend") or "red_sdk_half_res"),
                        **reference_metrics,
                    }
                )
            except Exception as exc:
                row_payload["reference_error"] = str(exc)
    if focus_rows:
        for metric_name in ("laplacian_variance", "tenengrad", "fft_high_frequency_energy"):
            metric_values = [float(item.get(metric_name, 0.0) or 0.0) for item in focus_rows]
            metric_best = max(metric_values) if metric_values else 1.0
            for item in focus_rows:
                item[f"{metric_name}_normalized"] = float(item.get(metric_name, 0.0) or 0.0) / max(float(metric_best), 1e-6)
        for item in focus_rows:
            item["composite_focus_score"] = float(
                np.mean(
                    [
                        float(item.get("laplacian_variance_normalized", 0.0) or 0.0),
                        float(item.get("tenengrad_normalized", 0.0) or 0.0),
                        float(item.get("fft_high_frequency_energy_normalized", 0.0) or 0.0),
                    ]
                )
            )
    if reference_rows:
        for metric_name in ("laplacian_variance", "tenengrad", "fft_high_frequency_energy"):
            metric_values = [float(item.get(metric_name, 0.0) or 0.0) for item in reference_rows]
            metric_best = max(metric_values) if metric_values else 1.0
            for item in reference_rows:
                item[f"{metric_name}_normalized"] = float(item.get(metric_name, 0.0) or 0.0) / max(float(metric_best), 1e-6)
        for item in reference_rows:
            item["composite_focus_score"] = float(
                np.mean(
                    [
                        float(item.get("laplacian_variance_normalized", 0.0) or 0.0),
                        float(item.get("tenengrad_normalized", 0.0) or 0.0),
                        float(item.get("fft_high_frequency_energy_normalized", 0.0) or 0.0),
                    ]
                )
            )
    summary = _focus_validation_summary(rows=focus_rows, reference_rows=reference_rows)
    summary["overlay_root"] = str(overlay_root)
    summary["resolved_camera_count"] = int(sum(1 for item in focus_rows if str(item.get("status") or "") == "resolved"))
    summary["soft_camera_count"] = int(sum(1 for item in focus_rows if str(item.get("focus_classification") or "") == "Soft"))
    summary["review_camera_count"] = int(sum(1 for item in focus_rows if str(item.get("focus_classification") or "") == "Review"))
    summary["sharp_camera_count"] = int(sum(1 for item in focus_rows if str(item.get("focus_classification") or "") == "Sharp"))
    path = out_root / "focus_validation.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {"summary": summary, "path": str(path), "overlay_root": str(overlay_root)}


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


def _scientific_validation_pixel_preview(pixel_array: np.ndarray, *, limit: int = 24) -> Dict[str, object]:
    values = np.asarray(pixel_array, dtype=np.float32).reshape(-1, 3)
    preview = values[: min(limit, values.shape[0])]
    return {
        "count": int(values.shape[0]),
        "normalized_rgb_preview": [[float(channel) for channel in row] for row in preview.tolist()],
        "uint8_rgb_preview": [
            [int(round(float(np.clip(channel, 0.0, 1.0)) * 255.0)) for channel in row]
            for row in preview.tolist()
        ],
    }


def _normalize_preview_image_array(image_array: np.ndarray) -> Tuple[np.ndarray, Dict[str, object]]:
    array = np.asarray(image_array)
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=2)
    elif array.ndim == 3 and array.shape[2] >= 3:
        array = array[..., :3]
    else:
        raise RuntimeError(f"Unsupported preview image shape for measurement: {array.shape!r}")
    dtype_name = str(array.dtype)
    denominator: Optional[float] = None
    if array.dtype == np.uint8:
        denominator = 255.0
    elif array.dtype == np.uint16:
        denominator = 65535.0
    if denominator is not None:
        normalized = array.astype(np.float32) / float(denominator)
    elif np.issubdtype(array.dtype, np.floating):
        normalized = array.astype(np.float32)
    else:
        normalized = array.astype(np.float32)
    normalized = np.clip(normalized, 0.0, 1.0)
    bit_depth = 16 if array.dtype == np.uint16 else 8 if array.dtype == np.uint8 else None
    return normalized, {
        "source_dtype": dtype_name,
        "bit_depth": bit_depth,
        "normalization_denominator": denominator,
    }


def _load_preview_image_as_normalized_rgb(preview_path: str | Path) -> Tuple[np.ndarray, Dict[str, object]]:
    path = Path(preview_path).expanduser().resolve()
    _wait_for_file_ready(path)
    last_error: Optional[BaseException] = None
    max_attempts = 8
    for attempt in range(1, max_attempts + 1):
        try:
            return _load_preview_image_once(path)
        except OSError as exc:
            last_error = exc
            if not _is_retryable_preview_load_error(exc) or attempt >= max_attempts:
                raise
            LOGGER.debug(
                "Preview load retry %s/%s for %s after OSError: %s",
                attempt,
                max_attempts,
                path,
                exc,
            )
            time.sleep(0.25)
            _wait_for_file_ready(path, max_attempts=6, delay_seconds=0.20)
        except Exception as exc:
            last_error = exc
            if attempt >= max_attempts:
                raise
            message = str(exc).lower()
            if "truncated" not in message and "failed to read" not in message and "cannot identify image file" not in message:
                raise
            LOGGER.debug(
                "Preview load retry %s/%s for %s after decode error: %s",
                attempt,
                max_attempts,
                path,
                exc,
            )
            time.sleep(0.25)
            _wait_for_file_ready(path, max_attempts=6, delay_seconds=0.20)
    if last_error is not None:
        raise last_error
    raise RuntimeError(f"Failed to load preview image: {path}")


def _minimum_render_output_size_bytes(path: Path) -> int:
    suffix = str(path.suffix or "").strip().lower()
    if suffix in {".tif", ".tiff"}:
        return PREVIEW_RENDER_MIN_TIFF_SIZE_BYTES
    return PREVIEW_RENDER_MIN_FILE_SIZE_BYTES


def _wait_for_file_ready(
    path: Path,
    *,
    max_attempts: int = 8,
    delay_seconds: float = 0.20,
    stable_polls_required: int = 2,
    min_size_bytes: Optional[int] = None,
) -> None:
    required_size = int(min_size_bytes if min_size_bytes is not None else 1)
    last_size: Optional[int] = None
    stable_polls = 0
    for attempt in range(1, max_attempts + 1):
        if path.exists():
            size = int(path.stat().st_size)
            if size >= required_size and last_size is not None and size == last_size:
                stable_polls += 1
            else:
                stable_polls = 0
            if size >= required_size and stable_polls >= stable_polls_required:
                if attempt > 1:
                    LOGGER.debug("Preview file became ready after %s poll(s): %s", attempt, path)
                return
            last_size = size
        else:
            last_size = None
            stable_polls = 0
        if attempt < max_attempts:
            LOGGER.debug("Waiting for preview file readiness %s/%s: %s", attempt, max_attempts, path)
            time.sleep(delay_seconds)
    if not path.exists():
        raise FileNotFoundError(f"Preview image does not exist yet: {path}")
    final_size = int(path.stat().st_size)
    if final_size <= 0:
        raise OSError(f"Preview image is empty: {path}")
    if final_size < required_size:
        raise OSError(
            f"Preview image below sane minimum size ({final_size} < {required_size} bytes): {path}"
        )


def _is_retryable_preview_load_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    return (
        "truncated" in message
        or "cannot identify image file" in message
        or "failed to read" in message
        or "image file is incomplete" in message
        or "below sane minimum size" in message
        or "invalid render stub" in message
    )


def _load_preview_image_once(path: Path) -> Tuple[np.ndarray, Dict[str, object]]:
    with Image.open(path) as image:
        mode = str(image.mode or "")
        width = int(image.width)
        height = int(image.height)
        raw_array: Optional[np.ndarray] = None
        if path.suffix.lower() in {".tif", ".tiff"}:
            try:
                if tifffile is not None:
                    raw_array = np.asarray(tifffile.imread(path))
                else:
                    import imageio.v3 as iio

                    raw_array = np.asarray(iio.imread(path))
            except Exception as exc:
                if _is_retryable_preview_load_error(exc):
                    raise OSError(str(exc)) from exc
                raw_array = None
        if raw_array is None:
            image.load()
            raw_array = np.asarray(image)
        supports_direct_normalization = (
            raw_array.ndim == 3
            and raw_array.shape[2] >= 3
            and (
                raw_array.dtype == np.uint8
                or raw_array.dtype == np.uint16
                or np.issubdtype(raw_array.dtype, np.floating)
            )
        )
        if supports_direct_normalization:
            normalized, metadata = _normalize_preview_image_array(raw_array[:, :, :3])
        else:
            converted = np.asarray(image.convert("RGB"))
            normalized, metadata = _normalize_preview_image_array(converted)
        metadata.update(
            {
                "path": str(path),
                "image_mode": mode,
                "width": width,
                "height": height,
                "preview_format": path.suffix.lstrip(".").lower(),
            }
        )
        return normalized, metadata


def _scientific_validation_file_fingerprint(path: Path) -> Dict[str, object]:
    fingerprint = {
        "path": str(path.resolve()),
        "exists": bool(path.exists()),
        "file_size_bytes": 0,
        "sha256": "",
        "image_dimensions": {"width": 0, "height": 0},
        "image_mode": "",
        "pixel_dtype": "",
        "bit_depth": None,
    }
    if not path.exists():
        return fingerprint
    stat = path.stat()
    fingerprint["file_size_bytes"] = int(stat.st_size)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    fingerprint["sha256"] = digest.hexdigest()
    try:
        _wait_for_file_ready(path, max_attempts=4, delay_seconds=0.12)
        with Image.open(path) as image:
            fingerprint["image_dimensions"] = {"width": int(image.width), "height": int(image.height)}
            fingerprint["image_mode"] = str(image.mode or "")
            raw_array: Optional[np.ndarray] = None
            if path.suffix.lower() in {".tif", ".tiff"}:
                try:
                    if tifffile is not None:
                        raw_array = np.asarray(tifffile.imread(path))
                    else:
                        import imageio.v3 as iio

                        raw_array = np.asarray(iio.imread(path))
                except Exception:
                    raw_array = None
            if raw_array is None:
                raw_array = np.asarray(image)
            fingerprint["pixel_dtype"] = str(raw_array.dtype)
            if raw_array.dtype == np.uint16:
                fingerprint["bit_depth"] = 16
            elif raw_array.dtype == np.uint8:
                fingerprint["bit_depth"] = 8
    except OSError:
        pass
    return fingerprint


def _scientific_validation_measurement_provenance(diagnostics: Dict[str, object], preview_path: Optional[str]) -> Dict[str, object]:
    provenance = dict(diagnostics.get("measurement_provenance") or {})
    stored_asset = dict(provenance.get("measurement_source_asset") or {})
    resolved_preview_path = Path(preview_path).expanduser().resolve() if preview_path else None
    current_asset = _scientific_validation_file_fingerprint(resolved_preview_path) if resolved_preview_path else {
        "path": str(preview_path or ""),
        "exists": False,
        "file_size_bytes": 0,
        "sha256": "",
        "image_dimensions": {"width": 0, "height": 0},
    }
    stored_dimensions = dict(stored_asset.get("image_dimensions") or {})
    stored_width = int(stored_dimensions.get("width", 0) or 0)
    stored_height = int(stored_dimensions.get("height", 0) or 0)
    fingerprint_present = bool(stored_asset.get("sha256")) and int(stored_asset.get("file_size_bytes", 0) or 0) > 0
    fingerprint_matches = (
        fingerprint_present
        and bool(current_asset.get("exists"))
        and str(stored_asset.get("sha256") or "") == str(current_asset.get("sha256") or "")
        and int(stored_asset.get("file_size_bytes", 0) or 0) == int(current_asset.get("file_size_bytes", 0) or 0)
        and stored_width == int(dict(current_asset.get("image_dimensions") or {}).get("width", 0) or 0)
        and stored_height == int(dict(current_asset.get("image_dimensions") or {}).get("height", 0) or 0)
    )
    status = "missing_measurement_fingerprint"
    reason = "Analysis artifact does not contain a measurement-time asset fingerprint."
    if fingerprint_present and not bool(current_asset.get("exists")):
        status = "missing_replay_asset"
        reason = "The original measurement-time replay asset is no longer present on disk."
    elif fingerprint_present and fingerprint_matches:
        status = "fingerprint_match"
        reason = "Current replay asset matches the measurement-time asset fingerprint."
    elif fingerprint_present:
        status = "fingerprint_mismatch"
        reason = "Current replay asset does not match the measurement-time asset fingerprint."
    return {
        "stored": provenance,
        "stored_asset": stored_asset,
        "current_asset": current_asset,
        "fingerprint_present": fingerprint_present,
        "fingerprint_matches": fingerprint_matches,
        "status": status,
        "reason": reason,
    }


def _scientific_validation_crop_bounds(
    diagnostics: Dict[str, object],
    *,
    width: int,
    height: int,
) -> Dict[str, int]:
    stored_width, stored_height = _scientific_validation_stored_frame_size(diagnostics)
    scale_x = float(width) / float(stored_width) if stored_width and stored_width > 0 else 1.0
    scale_y = float(height) / float(stored_height) if stored_height and stored_height > 0 else 1.0
    bounds = dict(diagnostics.get("measurement_crop_bounds") or {})
    if not bounds:
        return {"x0": 0, "x1": int(width), "y0": 0, "y1": int(height)}
    x0 = max(0, min(int(width - 1), int(round(float(bounds.get("x0", 0) or 0) * scale_x))))
    x1 = max(x0 + 1, min(int(width), int(round(float(bounds.get("x1", width) or width) * scale_x))))
    y0 = max(0, min(int(height - 1), int(round(float(bounds.get("y0", 0) or 0) * scale_y))))
    y1 = max(y0 + 1, min(int(height), int(round(float(bounds.get("y1", height) or height) * scale_y))))
    return {"x0": x0, "x1": x1, "y0": y0, "y1": y1}


def _scientific_validation_axis(diagnostics: Dict[str, object]) -> Tuple[float, float]:
    axis = dict(diagnostics.get("dominant_gradient_axis") or {})
    axis_x = float(axis.get("x", 0.0) or 0.0)
    axis_y = float(axis.get("y", -1.0) or -1.0)
    magnitude = math.hypot(axis_x, axis_y)
    if magnitude <= 1e-8:
        return (0.0, -1.0)
    return (axis_x / magnitude, axis_y / magnitude)


def _scientific_validation_stored_frame_size(diagnostics: Dict[str, object]) -> Tuple[Optional[float], Optional[float]]:
    provenance = dict((dict(diagnostics.get("measurement_provenance") or {})).get("measurement_source_asset") or {})
    dimensions = dict(provenance.get("image_dimensions") or {})
    if int(dimensions.get("width", 0) or 0) > 0 and int(dimensions.get("height", 0) or 0) > 0:
        return float(dimensions["width"]), float(dimensions["height"])
    width_candidates: List[float] = []
    height_candidates: List[float] = []
    for zone in list(diagnostics.get("zone_measurements") or diagnostics.get("neutral_samples") or []):
        bounds = dict(zone.get("bounds") or {})
        pixel = dict(bounds.get("pixel") or {})
        normalized = dict(bounds.get("normalized_within_roi") or {})
        x0 = float(pixel.get("x0", 0.0) or 0.0)
        x1 = float(pixel.get("x1", 0.0) or 0.0)
        y0 = float(pixel.get("y0", 0.0) or 0.0)
        y1 = float(pixel.get("y1", 0.0) or 0.0)
        w = max(x1 - x0, 0.0)
        h = max(y1 - y0, 0.0)
        nx = float(normalized.get("x", 0.0) or 0.0)
        ny = float(normalized.get("y", 0.0) or 0.0)
        nw = float(normalized.get("w", 0.0) or 0.0)
        nh = float(normalized.get("h", 0.0) or 0.0)
        if nw > 1e-8:
            width_candidates.append(w / nw)
        if nh > 1e-8:
            height_candidates.append(h / nh)
        if nx > 1e-8:
            width_candidates.append(x0 / nx)
        if ny > 1e-8:
            height_candidates.append(y0 / ny)
    stored_width = float(np.median(np.asarray(width_candidates, dtype=np.float32))) if width_candidates else None
    stored_height = float(np.median(np.asarray(height_candidates, dtype=np.float32))) if height_candidates else None
    return stored_width, stored_height


def _scientific_validation_resolve_report_clip(
    payload: Dict[str, object],
    *,
    strategy_key: str,
    clip_id: str,
) -> Dict[str, object]:
    for strategy in list(payload.get("strategies") or []):
        if str(strategy.get("strategy_key") or "") != str(strategy_key):
            continue
        for clip in list(strategy.get("clips") or []):
            if str(clip.get("clip_id") or "") == str(clip_id):
                return dict(clip)
    for clip in list(payload.get("clips") or []):
        if str(clip.get("clip_id") or "") == str(clip_id):
            return dict(clip)
    return {}


def _scientific_validation_report_measurement_reference(
    payload: Dict[str, object],
    *,
    strategy_key: str,
    clip_id: str,
) -> Dict[str, object]:
    report_json_path = str(payload.get("report_json") or "").strip()
    if report_json_path:
        debug_path = Path(report_json_path).with_name("contact_sheet_debug.json")
        if debug_path.exists():
            try:
                debug_payload = json.loads(debug_path.read_text(encoding="utf-8"))
                for row in list(debug_payload.get("measurement_values_per_camera") or []):
                    if str(row.get("clip_id") or "") != str(clip_id):
                        continue
                    measurement_values = dict(row.get("measurement_values") or {})
                    if measurement_values:
                        return {
                            "sample_1_ire": float(measurement_values.get("sample_1_ire", 0.0) or 0.0),
                            "sample_2_ire": float(measurement_values.get("sample_2_ire", 0.0) or 0.0),
                            "sample_3_ire": float(measurement_values.get("sample_3_ire", 0.0) or 0.0),
                            "display_scalar_log2": float(measurement_values.get("display_scalar_log2", 0.0) or 0.0),
                            "measurement_domain": "display-referred IPP2",
                            "source": "contact_sheet_debug",
                        }
            except (json.JSONDecodeError, OSError, ValueError):
                pass
    for item in list(payload.get("shared_originals") or []):
        if str(item.get("clip_id") or "") != str(clip_id):
            continue
        return {
            "sample_1_ire": float(item.get("sample_1_ire", 0.0) or 0.0),
            "sample_2_ire": float(item.get("sample_2_ire", 0.0) or 0.0),
            "sample_3_ire": float(item.get("sample_3_ire", 0.0) or 0.0),
            "display_scalar_log2": float(
                item.get("display_scalar_log2", item.get("measured_log2_luminance", 0.0)) or 0.0
            ),
            "measurement_domain": str(item.get("display_scalar_domain") or item.get("measurement_domain") or "display-referred IPP2"),
            "source": "shared_originals",
        }
    report_clip = _scientific_validation_resolve_report_clip(
        payload,
        strategy_key=strategy_key,
        clip_id=clip_id,
    )
    report_exposure_metrics = dict((report_clip.get("metrics") or {}).get("exposure") or {})
    return {
        "sample_1_ire": float(report_exposure_metrics.get("sample_1_ire", 0.0) or 0.0),
        "sample_2_ire": float(report_exposure_metrics.get("sample_2_ire", 0.0) or 0.0),
        "sample_3_ire": float(report_exposure_metrics.get("sample_3_ire", 0.0) or 0.0),
        "display_scalar_log2": float(
            report_exposure_metrics.get(
                "display_scalar_log2",
                report_exposure_metrics.get(
                    "measured_log2_luminance_monitoring",
                    report_exposure_metrics.get("measured_log2_luminance", 0.0),
                ),
            ) or 0.0
        ),
        "measurement_domain": str(report_exposure_metrics.get("measurement_domain") or ""),
        "source": "strategy_clip_metrics",
    }


def _scientific_validation_resolve_preview_path(
    diagnostics: Dict[str, object],
    payload: Dict[str, object],
    *,
    strategy_key: str,
    clip_id: str,
) -> str:
    provenance = dict((dict(diagnostics.get("measurement_provenance") or {})).get("measurement_source_asset") or {})
    provenance_path = str(provenance.get("path") or "").strip()
    if provenance_path:
        return provenance_path
    direct_path = str(diagnostics.get("rendered_measurement_preview_path") or "").strip()
    if direct_path:
        return direct_path
    report_clip = _scientific_validation_resolve_report_clip(
        payload,
        strategy_key=strategy_key,
        clip_id=clip_id,
    )
    fallback_candidates = [
        str(report_clip.get("original_frame") or "").strip(),
        str((report_clip.get("preview_variants") or {}).get("original") or "").strip(),
    ]
    for item in list(payload.get("shared_originals") or []):
        if str(item.get("clip_id") or "") != str(clip_id):
            continue
        fallback_candidates.append(str(item.get("original_frame") or "").strip())
    for candidate in fallback_candidates:
        if candidate and Path(candidate).exists():
            return candidate
    return ""


def _scientific_validation_recompute_samples(
    preview_path: str,
    diagnostics: Dict[str, object],
) -> Dict[str, object]:
    image, image_metadata = _load_preview_image_as_normalized_rgb(preview_path)
    image_height, image_width = image.shape[:2]
    stored_width, stored_height = _scientific_validation_stored_frame_size(diagnostics)
    scale_x = float(image_width) / float(stored_width) if stored_width and stored_width > 0 else 1.0
    scale_y = float(image_height) / float(stored_height) if stored_height and stored_height > 0 else 1.0
    crop_bounds = _scientific_validation_crop_bounds(
        diagnostics,
        width=image_width,
        height=image_height,
    )
    cropped_region = image[crop_bounds["y0"]:crop_bounds["y1"], crop_bounds["x0"]:crop_bounds["x1"], :]
    if cropped_region.size == 0:
        raise RuntimeError(f"Scientific validation crop is empty for {preview_path}")
    detected_roi = _coerce_sphere_roi(dict(diagnostics.get("detected_sphere_roi") or {}))
    if detected_roi is None:
        raise RuntimeError(f"Scientific validation requires detected_sphere_roi for {preview_path}")
    local_roi = SphereROI(
        cx=(float(detected_roi.cx) * scale_x) - float(crop_bounds["x0"]),
        cy=(float(detected_roi.cy) * scale_y) - float(crop_bounds["y0"]),
        r=float(detected_roi.r) * float((scale_x + scale_y) * 0.5),
    )
    pixels_chw = np.transpose(cropped_region, (2, 0, 1))
    sphere_mask, mask_metadata = build_sphere_sampling_mask(
        pixels_chw,
        local_roi,
        sampling_variant="refined",
    )
    axis_x, axis_y = _scientific_validation_axis(diagnostics)
    perpendicular_x = float(-axis_y)
    perpendicular_y = float(axis_x)
    height, width = sphere_mask.shape
    yy, xx = np.meshgrid(np.arange(height, dtype=np.float32), np.arange(width, dtype=np.float32), indexing="ij")
    interior_radius = max(float(local_roi.r) * 0.78, 1.0)
    interior_circle = ((xx - local_roi.cx) ** 2 + (yy - local_roi.cy) ** 2) <= interior_radius ** 2
    zone_results: List[Dict[str, object]] = []
    for zone_bounds in _sphere_zone_geometries(width, height, local_roi, (axis_x, axis_y)):
        zone_center = dict(zone_bounds.get("center") or {})
        center_x = float(zone_center.get("x", local_roi.cx) or local_roi.cx)
        center_y = float(zone_center.get("y", local_roi.cy) or local_roi.cy)
        along = ((xx - center_x) * axis_x) + ((yy - center_y) * axis_y)
        across = ((xx - center_x) * perpendicular_x) + ((yy - center_y) * perpendicular_y)
        rect_half_width = max(float(local_roi.r) * 0.34, 1.0)
        rect_half_height = max(float(local_roi.r) * 0.11, 1.0)
        rect_mask = (np.abs(along) <= rect_half_height) & (np.abs(across) <= rect_half_width)
        zone_mask = sphere_mask & rect_mask
        sampling_method = str(mask_metadata.get("sampling_method", "circle_mask"))
        if not np.any(zone_mask):
            zone_mask = interior_circle & rect_mask
            sampling_method = f"{sampling_method}_zone_fallback"
        if not np.any(zone_mask):
            zone_mask = interior_circle & (
                (np.abs(((xx - center_x) * axis_x) + ((yy - center_y) * axis_y)) <= (rect_half_height * 1.25))
                & (np.abs(((xx - center_x) * perpendicular_x) + ((yy - center_y) * perpendicular_y)) <= (rect_half_width * 1.25))
            )
            sampling_method = f"{sampling_method}_expanded_fallback"
        zone_pixels = cropped_region[zone_mask]
        rect_area = float(max(np.sum(rect_mask), 1))
        zone_fraction = float(np.sum(zone_mask)) / rect_area
        stats = _measure_pixel_cloud_statistics(
            zone_pixels,
            sampling_confidence=float(zone_fraction),
            sampling_method=sampling_method,
            mask_fraction=float(mask_metadata.get("mask_fraction", 1.0) or 1.0),
            interior_fraction=float(mask_metadata.get("interior_fraction", 1.0) or 1.0),
            interior_radius_ratio=float(mask_metadata.get("interior_radius_ratio", 1.0) or 1.0),
        )
        raw_pixels = np.asarray(zone_pixels, dtype=np.float32).reshape(-1, 3)
        valid_mask = np.all((raw_pixels > 0.002) & (raw_pixels < 0.998), axis=1)
        valid_pixels = raw_pixels[valid_mask]
        if valid_pixels.size == 0:
            valid_pixels = raw_pixels
        luminance = np.clip(
            valid_pixels[:, 0] * 0.2126 + valid_pixels[:, 1] * 0.7152 + valid_pixels[:, 2] * 0.0722,
            1e-6,
            1.0,
        )
        trimmed_luminance = percentile_clip(luminance, 5.0, 95.0)
        low = float(trimmed_luminance.min())
        high = float(trimmed_luminance.max())
        trimmed_pixels = valid_pixels[(luminance >= low) & (luminance <= high)]
        if trimmed_pixels.size == 0:
            trimmed_pixels = valid_pixels
        log_values = np.log2(np.clip(trimmed_luminance, 1e-6, 1.0))
        zone_results.append(
            {
                "label": str(zone_bounds.get("label") or ""),
                "display_label": str(zone_bounds.get("display_label") or ""),
                "zone_fraction": zone_fraction,
                "sampling_method": sampling_method,
                "pixel_preview": _scientific_validation_pixel_preview(raw_pixels),
                "valid_pixel_preview": _scientific_validation_pixel_preview(valid_pixels),
                "trimmed_pixel_preview": _scientific_validation_pixel_preview(trimmed_pixels),
                "luminance_preview": [float(value) for value in luminance[: min(24, luminance.size)].tolist()],
                "trimmed_luminance_preview": [float(value) for value in trimmed_luminance[: min(24, trimmed_luminance.size)].tolist()],
                "log2_luminance_preview": [float(value) for value in log_values[: min(24, log_values.size)].tolist()],
                "computed": {
                    "luminance_formula": "Y = 0.2126 R + 0.7152 G + 0.0722 B",
                    "luminance_domain": "display-referred IPP2 RGB normalized to 0-1",
                    "trim_low_percentile": 5.0,
                    "trim_high_percentile": 95.0,
                    "median_trimmed_luminance": float(np.median(trimmed_luminance)),
                    "median_log2_luminance": float(np.median(log_values)),
                    "computed_ire": float((2.0 ** float(np.median(log_values))) * 100.0),
                    "trimmed_luminance_count": int(trimmed_luminance.size),
                    "retained_fraction": float(trimmed_luminance.size) / float(max(raw_pixels.shape[0], 1)),
                },
                "recomputed_zone_measurement": {
                    "measured_log2_luminance": float(stats["measured_log2_luminance"]),
                    "measured_ire": float(stats["measured_ire"]),
                    "measured_rgb_mean": [float(value) for value in list(stats["measured_rgb_mean"])],
                    "measured_rgb_chromaticity": [float(value) for value in list(stats["measured_rgb_chromaticity"])],
                    "gray_luminance_distribution": dict(stats["gray_luminance_distribution"]),
                    "gray_log2_distribution": dict(stats["gray_log2_distribution"]),
                },
            }
        )
    ordered = _ordered_zone_profile(zone_results)
    aggregate_log2 = float(np.median(np.asarray([float(item["recomputed_zone_measurement"]["measured_log2_luminance"]) for item in ordered], dtype=np.float32)))
    aggregate_ire = _ire_from_log2_luminance(aggregate_log2)
    return {
        "preview_path": str(preview_path),
        "image_metadata": image_metadata,
        "image_width": int(image_width),
        "image_height": int(image_height),
        "stored_coordinate_space": {
            "width": stored_width,
            "height": stored_height,
            "scale_x": scale_x,
            "scale_y": scale_y,
        },
        "crop_bounds": crop_bounds,
        "crop_size": {"width": int(cropped_region.shape[1]), "height": int(cropped_region.shape[0])},
        "local_sphere_roi": {"cx": float(local_roi.cx), "cy": float(local_roi.cy), "r": float(local_roi.r)},
        "mask_metadata": {
            key: float(value) if isinstance(value, (float, int)) else value
            for key, value in dict(mask_metadata).items()
        },
        "axis_vector": {
            "x": float(axis_x),
            "y": float(axis_y),
            "perpendicular_x": float(perpendicular_x),
            "perpendicular_y": float(perpendicular_y),
        },
        "zone_samples": ordered,
        "aggregate": {
            "measured_log2_luminance": aggregate_log2,
            "measured_ire": aggregate_ire,
            "sample_scalar_display_log2": aggregate_log2,
            "sample_scalar_display_ire": aggregate_ire,
        },
    }


def _scientific_validation_report_markdown(summary: Dict[str, object]) -> str:
    pipeline = dict(summary.get("pipeline") or {})
    example = dict(summary.get("worked_example") or {})
    provenance = dict(summary.get("measurement_provenance") or {})
    stored_asset = dict(provenance.get("stored_asset") or {})
    current_asset = dict(provenance.get("current_asset") or {})
    zone_map = _zone_profile_by_label(list((example.get("zone_samples") or [])))
    sample_2 = dict(zone_map.get("center") or {})
    sample_2_computed = dict(sample_2.get("computed") or {})
    sample_2_recomputed = dict(sample_2.get("recomputed_zone_measurement") or {})
    crop_bounds = dict(example.get("crop_bounds") or {})
    consistency = dict(summary.get("consistency_validation") or {})
    lines = [
        "# Scientific Validation",
        "",
        f"- Status: `{summary.get('status')}`",
        f"- Reason: `{summary.get('reason')}`",
        "",
        "## Measurement Provenance",
        "",
        f"- Provenance status: `{provenance.get('status')}`",
        f"- Provenance reason: `{provenance.get('reason')}`",
        f"- Measurement-time asset path: `{stored_asset.get('path')}`",
        f"- Measurement-time SHA-256: `{stored_asset.get('sha256')}`",
        f"- Measurement-time size: `{stored_asset.get('file_size_bytes')}` bytes",
        f"- Measurement-time dimensions: `{stored_asset.get('image_dimensions')}`",
        f"- Current replay asset path: `{current_asset.get('path')}`",
        f"- Current replay SHA-256: `{current_asset.get('sha256')}`",
        f"- Current replay size: `{current_asset.get('file_size_bytes')}` bytes",
        f"- Current replay dimensions: `{current_asset.get('image_dimensions')}`",
        "",
        "## Measurement Entry Point",
        "",
        f"- Pixel loader: `{pipeline.get('image_reader_function')}`",
        f"- Library: `{pipeline.get('image_reader_library')}`",
        f"- Input format: `{pipeline.get('pixel_input_format')}`",
        f"- Normalized format: `{pipeline.get('pixel_normalized_format')}`",
        f"- Measurement domain: `{pipeline.get('measurement_domain')}`",
        "",
        "## Sphere Detection + Sampling Geometry",
        "",
        f"- Detection library: `{pipeline.get('sphere_detection_library')}`",
        f"- Detection function: `{pipeline.get('sphere_detection_function')}`",
        f"- Edge detector: `{pipeline.get('edge_detector')}`",
        f"- Circle estimator: `{pipeline.get('circle_estimator')}`",
        f"- Mask refinement: `{pipeline.get('mask_refinement')}`",
        f"- Sample geometry: `{pipeline.get('sample_geometry')}`",
        "",
        "## Measurement Formulas",
        "",
        "- Luminance: `Y = 0.2126 R + 0.7152 G + 0.0722 B`",
        "- Trim: retain 5th to 95th percentile luminance values after clipping guard",
        "- Log2 scalar: `median(log2(trimmed_luminance))`",
        "- IRE mapping: `IRE = (2 ** measured_log2_luminance) * 100`",
        "- Note: these are computed display-domain IRE values, not waveform instrument readings",
        "",
        "## Worked Example",
        "",
        f"- Clip: `{example.get('clip_id')}`",
        f"- Preview image: `{example.get('preview_path')}`",
        f"- Crop bounds: `{crop_bounds}`",
        f"- Sphere ROI (local crop): `{example.get('local_sphere_roi')}`",
        "",
        "### Sample 2 Proof Block",
        "",
        f"- Raw RGB preview (normalized): `{(sample_2.get('pixel_preview') or {}).get('normalized_rgb_preview', [])[:10]}`",
        f"- Luminance preview: `{list((sample_2.get('luminance_preview') or [])[:10])}`",
        f"- Trimmed luminance preview: `{list((sample_2.get('trimmed_luminance_preview') or [])[:10])}`",
        f"- Log2 preview: `{list((sample_2.get('log2_luminance_preview') or [])[:10])}`",
        f"- Median trimmed luminance: `{sample_2_computed.get('median_trimmed_luminance')}`",
        f"- Median log2 luminance: `{sample_2_computed.get('median_log2_luminance')}`",
        f"- Computed IRE: `{sample_2_computed.get('computed_ire')}`",
        f"- Recomputed stored log2: `{sample_2_recomputed.get('measured_log2_luminance')}`",
        f"- Recomputed stored IRE: `{sample_2_recomputed.get('measured_ire')}`",
        "",
        "## Consistency Validation",
        "",
        f"- Analysis match: `{consistency.get('analysis_matches_recomputed')}`",
        f"- IPP2 validation match: `{consistency.get('ipp2_validation_matches_analysis')}`",
        f"- Report metrics match: `{consistency.get('report_matches_analysis')}`",
        f"- Validation status: `{consistency.get('status')}`",
        f"- Replay asset fingerprint match: `{provenance.get('fingerprint_matches')}`",
        "",
    ]
    return "\n".join(lines)


def _build_scientific_validation_artifacts(
    *,
    out_root: Path,
    analysis_records: List[Dict[str, object]],
    payload: Dict[str, object],
    fail_on_analysis_drift: bool = False,
) -> Dict[str, object]:
    out_root.mkdir(parents=True, exist_ok=True)
    if not analysis_records:
        raise RuntimeError("Scientific validation requires at least one analysis record.")
    recommended_strategy = dict(payload.get("recommended_strategy") or {})
    recommended_strategy_key = str(recommended_strategy.get("strategy_key") or "median")
    ipp2_rows = list((payload.get("ipp2_validation") or {}).get("rows") or [])
    example_row = next(
        (
            dict(row)
            for row in ipp2_rows
            if str(row.get("strategy_key") or "") == recommended_strategy_key and str(row.get("clip_id") or "").strip()
        ),
        dict(ipp2_rows[0]) if ipp2_rows else {},
    )
    example_clip_id = str(example_row.get("clip_id") or analysis_records[0].get("clip_id") or "")
    analysis_record = next(
        (dict(record) for record in analysis_records if str(record.get("clip_id") or "") == example_clip_id),
        dict(analysis_records[0]),
    )
    diagnostics = dict(analysis_record.get("diagnostics") or {})
    preview_path = _scientific_validation_resolve_preview_path(
        diagnostics,
        payload,
        strategy_key=recommended_strategy_key,
        clip_id=example_clip_id,
    )
    if (
        not preview_path
        or not Path(preview_path).exists()
        or not dict(diagnostics.get("detected_sphere_roi") or {})
        or not list(diagnostics.get("zone_measurements") or diagnostics.get("neutral_samples") or [])
    ):
        summary = {
            "status": "unavailable",
            "reason": "stored run data does not contain full sphere replay prerequisites",
            "clip_id": example_clip_id,
            "recommended_strategy_key": recommended_strategy_key,
            "available_fields": {
                "preview_path": preview_path,
                "preview_exists": bool(preview_path and Path(preview_path).exists()),
                "detected_sphere_roi": bool(dict(diagnostics.get("detected_sphere_roi") or {})),
                "zone_measurements": bool(list(diagnostics.get("zone_measurements") or diagnostics.get("neutral_samples") or [])),
            },
        }
        json_path = out_root / "scientific_validation.json"
        markdown_path = out_root / "scientific_validation.md"
        json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        markdown_path.write_text(_scientific_validation_report_markdown(summary), encoding="utf-8")
        return {
            "path": str(json_path),
            "markdown_path": str(markdown_path),
            "summary": summary,
        }
    provenance = _scientific_validation_measurement_provenance(diagnostics, preview_path)
    recomputed = _scientific_validation_recompute_samples(preview_path, diagnostics)
    analysis_zone_profile = _ordered_zone_profile(list(diagnostics.get("zone_measurements") or diagnostics.get("neutral_samples") or []))
    if not analysis_zone_profile:
        raise RuntimeError(f"Scientific validation requires zone_measurements for {example_clip_id}")
    report_reference = _scientific_validation_report_measurement_reference(
        payload,
        strategy_key=recommended_strategy_key,
        clip_id=example_clip_id,
    )
    ipp2_original_profile = _ordered_zone_profile(list(example_row.get("ipp2_original_zone_profile") or []))
    if not ipp2_original_profile:
        ipp2_original_profile = analysis_zone_profile

    analysis_sample_ires = _sample_ire_fields_from_profile(analysis_zone_profile)
    analysis_scalar = _sample_scalar_display_from_profile(analysis_zone_profile)
    recomputed_zone_map = _zone_profile_by_label(list(recomputed.get("zone_samples") or []))
    analysis_zone_map = _zone_profile_by_label(analysis_zone_profile)
    ipp2_zone_map = _zone_profile_by_label(ipp2_original_profile)

    per_zone_checks: List[Dict[str, object]] = []
    for label in SPHERE_PROFILE_ZONE_ORDER:
        recomputed_zone = dict(recomputed_zone_map.get(label) or {})
        analysis_zone = dict(analysis_zone_map.get(label) or {})
        ipp2_zone = dict(ipp2_zone_map.get(label) or {})
        recomputed_measure = dict(recomputed_zone.get("recomputed_zone_measurement") or {})
        if not recomputed_measure or not analysis_zone:
            raise RuntimeError(f"Scientific validation missing zone trace for {example_clip_id} / {label}")
        analysis_delta_log2 = abs(
            float(recomputed_measure.get("measured_log2_luminance", 0.0) or 0.0)
            - float(analysis_zone.get("measured_log2_luminance", 0.0) or 0.0)
        )
        analysis_delta_ire = abs(
            float(recomputed_measure.get("measured_ire", 0.0) or 0.0)
            - float(analysis_zone.get("measured_ire", 0.0) or 0.0)
        )
        ipp2_delta_ire = abs(
            float(analysis_zone.get("measured_ire", 0.0) or 0.0)
            - float(ipp2_zone.get("measured_ire", analysis_zone.get("measured_ire", 0.0)) or 0.0)
        )
        per_zone_checks.append(
            {
                "label": label,
                "display_label": SPHERE_PROFILE_ZONE_DISPLAY[label],
                "analysis_delta_log2": analysis_delta_log2,
                "analysis_delta_ire": analysis_delta_ire,
                "ipp2_delta_ire": ipp2_delta_ire,
                "analysis_match": analysis_delta_log2 <= 1e-6 and analysis_delta_ire <= 1e-6,
                "ipp2_match": ipp2_delta_ire <= 1e-6,
            }
        )
    report_sample_1 = float(report_reference.get("sample_1_ire", analysis_sample_ires["sample_1_ire"]) or 0.0)
    report_sample_2 = float(report_reference.get("sample_2_ire", analysis_sample_ires["sample_2_ire"]) or 0.0)
    report_sample_3 = float(report_reference.get("sample_3_ire", analysis_sample_ires["sample_3_ire"]) or 0.0)
    report_scalar = float(report_reference.get("display_scalar_log2", analysis_scalar["sample_scalar_display_log2"]) or 0.0)
    report_deltas = {
        "sample_1_ire": abs(report_sample_1 - analysis_sample_ires["sample_1_ire"]),
        "sample_2_ire": abs(report_sample_2 - analysis_sample_ires["sample_2_ire"]),
        "sample_3_ire": abs(report_sample_3 - analysis_sample_ires["sample_3_ire"]),
        "scalar_log2": abs(report_scalar - analysis_scalar["sample_scalar_display_log2"]),
    }
    report_matches_analysis = all(value <= 1e-6 for value in report_deltas.values())

    analysis_matches_recomputed = all(bool(item.get("analysis_match")) for item in per_zone_checks)
    ipp2_matches_analysis = all(bool(item.get("ipp2_match")) for item in per_zone_checks)
    validation_status = "fully_reconciled"
    validation_reason = "Stored analysis, report values, and replayed measurement asset fully reconcile."
    if not report_matches_analysis or not ipp2_matches_analysis:
        validation_status = "analysis_drift"
        validation_reason = "Stored analysis truth diverges from report-visible or validation-visible values."
    elif not bool(provenance.get("fingerprint_matches")) or not analysis_matches_recomputed:
        validation_status = "blocked_asset_mismatch"
        if str(provenance.get("status") or "") == "fingerprint_mismatch":
            validation_reason = "Stored analysis/report values reconcile, but the current replay asset does not match the measurement-time fingerprint."
        elif str(provenance.get("status") or "") == "missing_replay_asset":
            validation_reason = "Stored analysis/report values reconcile, but the original replay asset is missing."
        elif str(provenance.get("status") or "") == "missing_measurement_fingerprint":
            validation_reason = "Stored analysis/report values reconcile, but the archived analysis artifact does not contain a measurement-time fingerprint and current replay values do not reconcile."
        else:
            validation_reason = "Stored analysis/report values reconcile, but replaying the current measurement preview asset does not reproduce the stored analysis measurements."
    summary = {
        "status": validation_status,
        "reason": validation_reason,
        "clip_id": example_clip_id,
        "recommended_strategy_key": recommended_strategy_key,
        "pipeline": {
            "image_reader_function": "r3dmatch.report._measure_rendered_preview_roi_ipp2",
            "image_reader_library": "Pillow -> numpy float32",
            "pixel_input_format": "uint8 RGB JPEG/PNG preview",
            "pixel_normalized_format": "float32 RGB normalized to 0-1",
            "measurement_domain": "display-referred IPP2 / BT.709 / BT.1886",
            "sphere_detection_library": "scikit-image",
            "sphere_detection_function": "r3dmatch.report._detect_sphere_candidates_in_region_hwc",
            "edge_detector": "skimage.feature.canny",
            "circle_estimator": "skimage.transform.hough_circle + hough_circle_peaks",
            "mask_refinement": "build_sphere_sampling_mask -> remove_small_objects -> binary_opening -> binary_closing -> regionprops selection",
            "sample_geometry": "three gradient-aligned rectangular bands intersected with refined sphere mask",
            "luminance_formula": "Y = 0.2126 R + 0.7152 G + 0.0722 B",
            "statistical_reduction": "clip 0.002-0.998 -> 5th-95th percentile trim -> median(log2(trimmed_luminance))",
            "ire_mapping": "IRE = (2 ** measured_log2_luminance) * 100",
            "ire_note": "Computed display-domain IRE values, not waveform readings",
        },
        "worked_example": {
            "clip_id": example_clip_id,
            **recomputed,
        },
        "measurement_provenance": provenance,
        "stored_analysis_reference": {
            "analysis_path": str(analysis_record.get("analysis_path") or ""),
            "preview_path": preview_path,
            "sample_1_ire": analysis_sample_ires["sample_1_ire"],
            "sample_2_ire": analysis_sample_ires["sample_2_ire"],
            "sample_3_ire": analysis_sample_ires["sample_3_ire"],
            "display_scalar_log2": float(analysis_scalar["sample_scalar_display_log2"]),
            "gray_exposure_summary": str(diagnostics.get("gray_exposure_summary") or ""),
            "zone_measurements": analysis_zone_profile,
        },
        "report_reference": {
            "report_clip_id": example_clip_id,
            "source": str(report_reference.get("source") or ""),
            "report_sample_1_ire": report_sample_1,
            "report_sample_2_ire": report_sample_2,
            "report_sample_3_ire": report_sample_3,
            "report_scalar_log2": report_scalar,
            "report_measurement_domain": str(report_reference.get("measurement_domain") or ""),
        },
        "ipp2_validation_reference": {
            "clip_id": example_clip_id,
            "ipp2_original_value_log2": float(
                example_row.get("ipp2_original_value_log2", diagnostics.get("measured_log2_luminance", 0.0)) or 0.0
            ),
            "ipp2_original_zone_profile": ipp2_original_profile,
        },
        "consistency_validation": {
            "status": validation_status,
            "per_zone_checks": per_zone_checks,
            "analysis_matches_recomputed": analysis_matches_recomputed,
            "ipp2_validation_matches_analysis": ipp2_matches_analysis,
            "report_matches_analysis": report_matches_analysis,
            "report_deltas": report_deltas,
        },
    }
    if fail_on_analysis_drift and validation_status == "analysis_drift":
        raise RuntimeError(
            "Scientific validation detected report IRE/scalar drift from stored measurement payload "
            f"for {example_clip_id}: {report_deltas}"
        )
    json_path = out_root / "scientific_validation.json"
    markdown_path = out_root / "scientific_validation.md"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    markdown_path.write_text(_scientific_validation_report_markdown(summary), encoding="utf-8")
    return {
        "path": str(json_path),
        "markdown_path": str(markdown_path),
        "summary": summary,
    }


def build_scientific_validation_report(
    input_path: str,
    *,
    out_dir: Optional[str] = None,
) -> Dict[str, object]:
    analysis_records = _load_analysis_records(input_path)
    root = Path(input_path).expanduser().resolve()
    report_root = Path(out_dir).expanduser().resolve() if out_dir else (root / "report" if (root / "report").exists() else root)
    contact_sheet_path = report_root / "contact_sheet.json"
    payload = json.loads(contact_sheet_path.read_text(encoding="utf-8")) if contact_sheet_path.exists() else {}
    return _build_scientific_validation_artifacts(
        out_root=report_root,
        analysis_records=analysis_records,
        payload=payload,
    )


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
    extension: Optional[str] = None,
) -> str:
    suffix = (extension or preview_still_extension(DEFAULT_PREVIEW_STILL_FORMAT)).lstrip(".")
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
    status = resolve_redline_tool_status()
    if status["ready"]:
        return str(status["resolved_path"])
    raise RuntimeError(str(status["error"] or "REDLine not found — real validation required"))


def resolve_redline_tool_status() -> Dict[str, object]:
    configured = str(os.environ.get("R3DMATCH_REDLINE_EXECUTABLE", "") or "").strip()
    source = "environment" if configured else "path"
    config_path = ""
    if not configured and REDLINE_CONFIG_PATH.exists():
        try:
            config_payload = json.loads(REDLINE_CONFIG_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            return {
                "ready": False,
                "configured": "",
                "resolved_path": "",
                "source": "config_error",
                "config_path": str(REDLINE_CONFIG_PATH),
                "error": f"Invalid REDLine config JSON: {REDLINE_CONFIG_PATH}: {exc}",
            }
        configured = str(config_payload.get("redline_executable") or config_payload.get("redline_path") or "").strip()
        if configured:
            source = "config"
            config_path = str(REDLINE_CONFIG_PATH)
    executable = configured or "REDLine"
    resolved = shutil.which(executable)
    if resolved:
        return {
            "ready": True,
            "configured": executable,
            "resolved_path": str(Path(resolved).resolve()),
            "source": source if configured else "path",
            "config_path": config_path,
            "error": "",
        }
    candidate = Path(executable).expanduser()
    if candidate.exists():
        return {
            "ready": True,
            "configured": executable,
            "resolved_path": str(candidate.resolve()),
            "source": "explicit_path" if configured else "path",
            "config_path": config_path,
            "error": "",
        }
    return {
        "ready": False,
        "configured": executable,
        "resolved_path": "",
        "source": source,
        "config_path": config_path,
        "error": "REDLine executable is not available. Configure R3DMATCH_REDLINE_EXECUTABLE, add REDLine to PATH, or provide a valid config/redline.json entry.",
    }


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
    artifact_mode: str = DEFAULT_ARTIFACT_MODE,
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
    if normalize_artifact_mode(artifact_mode) == "production":
        for probe in source_probes:
            probe_path = Path(str(probe.get("output_path") or "")).expanduser()
            if probe_path.exists():
                try:
                    probe_path.unlink()
                    probe["output_deleted_after_validation"] = True
                except OSError:
                    probe["output_deleted_after_validation"] = False
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
    preview_still_format: Optional[str] = None,
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
    resolved_preview_still_format = normalize_preview_still_format(
        preview_still_format or str(defaults.get("preview_still_format") or DEFAULT_PREVIEW_STILL_FORMAT)
    )
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
        "preview_still_format": resolved_preview_still_format,
        "preview_still_extension": preview_still_extension(resolved_preview_still_format),
        "preview_still_redline_format_code": int(PREVIEW_STILL_FORMAT_CODES[resolved_preview_still_format]),
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
        "operator_status": "Enabled for operator review" if enabled else "Not shown in operator review",
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
        return image
    height, width = image.shape[:2]
    x0 = max(0, min(width - 1, int(np.floor(float(calibration_roi["x"]) * width))))
    y0 = max(0, min(height - 1, int(np.floor(float(calibration_roi["y"]) * height))))
    x1 = max(x0 + 1, min(width, int(np.ceil((float(calibration_roi["x"]) + float(calibration_roi["w"])) * width))))
    y1 = max(y0 + 1, min(height, int(np.ceil((float(calibration_roi["y"]) + float(calibration_roi["h"])) * height))))
    return image[y0:y1, x0:x1, :]


def _normalized_roi_origin(image: np.ndarray, calibration_roi: Optional[Dict[str, float]]) -> Tuple[int, int]:
    if calibration_roi is None:
        return (0, 0)
    height, width = image.shape[:2]
    x0 = max(0, min(width - 1, int(np.floor(float(calibration_roi["x"]) * width))))
    y0 = max(0, min(height - 1, int(np.floor(float(calibration_roi["y"]) * height))))
    return (x0, y0)


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
    image, image_metadata = _load_preview_image_as_normalized_rgb(preview_path)
    region = _extract_normalized_roi_region_hwc(image, calibration_roi)
    stats = _solver_sampling_measurement_from_hwc_region(region)
    pixels = region.reshape(-1, 3)
    saturation_fraction = float(np.mean(np.max(pixels, axis=1) >= 0.998)) if pixels.size else 0.0
    return {
        **stats,
        "measured_saturation_fraction_monitoring": saturation_fraction,
        "render_width": int(image_metadata.get("width", image.shape[1])),
        "render_height": int(image_metadata.get("height", image.shape[0])),
        "rendered_image_dtype": str(image_metadata.get("source_dtype") or ""),
        "rendered_image_mode": str(image_metadata.get("image_mode") or ""),
        "rendered_image_bit_depth": image_metadata.get("bit_depth"),
        "rendered_image_normalization_denominator": image_metadata.get("normalization_denominator"),
        "rendered_preview_format": str(image_metadata.get("preview_format") or ""),
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


def _dedupe_sphere_candidates(candidates: List[Dict[str, object]], *, distance_factor: float = 0.40) -> List[Dict[str, object]]:
    if not candidates:
        return []
    kept: List[Dict[str, object]] = []
    for candidate in sorted(candidates, key=lambda item: float(item.get("confidence", 0.0) or 0.0), reverse=True):
        roi_payload = dict(candidate.get("roi") or {})
        radius = float(roi_payload.get("r", 0.0) or 0.0)
        cx = float(roi_payload.get("cx", 0.0) or 0.0)
        cy = float(roi_payload.get("cy", 0.0) or 0.0)
        if radius <= 0.0:
            continue
        duplicate = False
        for existing in kept:
            existing_roi = dict(existing.get("roi") or {})
            existing_radius = float(existing_roi.get("r", 0.0) or 0.0)
            distance = math.hypot(
                cx - float(existing_roi.get("cx", 0.0) or 0.0),
                cy - float(existing_roi.get("cy", 0.0) or 0.0),
            )
            if distance <= max(radius, existing_radius) * float(distance_factor) and abs(radius - existing_radius) <= max(radius, existing_radius) * 0.25:
                duplicate = True
                break
        if not duplicate:
            kept.append(candidate)
    return kept


def _resize_region_for_sphere_detection(region: np.ndarray) -> Tuple[np.ndarray, float]:
    height, width = region.shape[:2]
    max_dimension = max(height, width)
    if max_dimension <= SPHERE_DETECTION_MAX_DIMENSION:
        return region, 1.0
    scale = float(SPHERE_DETECTION_MAX_DIMENSION) / float(max_dimension)
    resized_height = max(64, int(round(float(height) * scale)))
    resized_width = max(64, int(round(float(width) * scale)))
    resized = transform.resize(
        region,
        (resized_height, resized_width, region.shape[2]),
        order=1,
        mode="reflect",
        anti_aliasing=True,
        preserve_range=True,
    ).astype(np.float32)
    return np.clip(resized, 0.0, 1.0), float(scale)


def _rescale_sphere_candidates(candidates: List[Dict[str, object]], scale: float) -> List[Dict[str, object]]:
    if not candidates or math.isclose(float(scale), 1.0, rel_tol=1e-9, abs_tol=1e-9):
        return [copy.deepcopy(dict(candidate)) for candidate in candidates]
    factor = 1.0 / float(scale)
    rescaled: List[Dict[str, object]] = []
    for candidate in candidates:
        item = copy.deepcopy(dict(candidate))
        roi_payload = dict(item.get("roi") or {})
        if roi_payload:
            roi_payload["cx"] = float(roi_payload.get("cx", 0.0) or 0.0) * factor
            roi_payload["cy"] = float(roi_payload.get("cy", 0.0) or 0.0) * factor
            roi_payload["r"] = float(roi_payload.get("r", 0.0) or 0.0) * factor
            item["roi"] = roi_payload
        item["detection_resize_scale"] = float(scale)
        rescaled.append(item)
    return rescaled


def _plateau_score(value: float, low: float, high: float, falloff: float) -> float:
    numeric_value = float(value)
    numeric_low = float(low)
    numeric_high = float(high)
    numeric_falloff = max(float(falloff), 1e-6)
    if numeric_value < numeric_low:
        return max(0.0, 1.0 - ((numeric_low - numeric_value) / numeric_falloff))
    if numeric_value > numeric_high:
        return max(0.0, 1.0 - ((numeric_value - numeric_high) / numeric_falloff))
    return 1.0


def _evaluate_detected_sphere_roi(
    region: np.ndarray,
    *,
    roi: SphereROI,
    edge_map: np.ndarray,
    accumulator: float,
    detection_source: str,
    shape_circularity: Optional[float] = None,
    aspect_ratio: Optional[float] = None,
    region_state: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    if region_state is None:
        region_state = _build_sphere_region_state(region)
    height = int(region_state.get("height", region.shape[0]))
    width = int(region_state.get("width", region.shape[1]))
    yy = np.asarray(region_state.get("yy"), dtype=np.float32)
    xx = np.asarray(region_state.get("xx"), dtype=np.float32)
    radius = float(max(roi.r, 1.0))
    distance = np.sqrt((xx - float(roi.cx)) ** 2 + (yy - float(roi.cy)) ** 2)
    interior_mask = distance <= radius * 0.92
    ring_mask = (distance >= radius * 1.02) & (distance <= radius * 1.20)
    edge_band = np.abs(distance - radius) <= 2.5
    center_core_mask = distance <= radius * 0.35
    inner_shell_mask = (distance >= radius * 0.52) & (distance <= radius * 0.82)
    edge_band_pixels = int(np.count_nonzero(edge_band))
    gradient_magnitude = np.asarray(region_state.get("gradient_magnitude"), dtype=np.float32)
    edge_support = _circle_edge_support(edge_map, roi, gradient_magnitude=gradient_magnitude)
    luminance = np.maximum(np.asarray(region_state.get("luminance"), dtype=np.float32), 1e-6)
    interior_values = luminance[interior_mask]
    ring_values = luminance[ring_mask]
    center_core_values = luminance[center_core_mask]
    inner_shell_values = luminance[inner_shell_mask]
    interior_stddev = float(np.std(interior_values)) if interior_values.size else 1.0
    interior_mean = float(np.mean(interior_values)) if interior_values.size else 0.0
    ring_mean = float(np.mean(ring_values)) if ring_values.size else interior_mean
    center_core_mean = float(np.mean(center_core_values)) if center_core_values.size else interior_mean
    inner_shell_mean = float(np.mean(inner_shell_values)) if inner_shell_values.size else interior_mean
    contrast = abs(interior_mean - ring_mean)
    internal_shading = abs(center_core_mean - inner_shell_mean)
    radius_ratio = radius / float(max(min(height, width), 1))
    radius_score = 1.0 - min(abs(radius_ratio - 0.22) / 0.14, 1.0)
    center_x_ratio = float(roi.cx) / float(max(width, 1))
    center_y_ratio = float(roi.cy) / float(max(height, 1))
    edge_strength = float(np.mean(edge_map[edge_band])) if np.any(edge_band) else 0.0
    edge_score = min(max(float(accumulator), edge_strength, edge_support) / 0.30, 1.0)
    texture_score = _plateau_score(interior_stddev, 0.012, 0.16, 0.08)
    shading_score = _plateau_score(internal_shading, 0.015, 0.22, 0.10)
    contrast_score = min(contrast / 0.12, 1.0)
    zone_score = _zone_containment_score(roi, width, height)
    interior_rgb = np.asarray(region_state.get("rgb"), dtype=np.float32)[interior_mask]
    interior_rgb_mean = np.mean(interior_rgb, axis=0) if interior_rgb.size else np.asarray([0.0, 0.0, 0.0], dtype=np.float32)
    if interior_rgb.size:
        interior_sum = np.sum(interior_rgb, axis=1)
        valid_chrom = interior_sum > 1e-6
        safe_sum = np.where(valid_chrom, interior_sum, 1.0)
        r_chrom = np.divide(interior_rgb[:, 0], safe_sum, out=np.zeros_like(interior_sum), where=valid_chrom)
        g_chrom = np.divide(interior_rgb[:, 1], safe_sum, out=np.zeros_like(interior_sum), where=valid_chrom)
        chroma_distance = np.sqrt((r_chrom - (1.0 / 3.0)) ** 2 + (g_chrom - (1.0 / 3.0)) ** 2)
        chroma_range = float(np.max(chroma_distance)) if chroma_distance.size else 1.0
        chroma_mean_distance = float(np.mean(chroma_distance[valid_chrom])) if np.any(valid_chrom) else 1.0
    else:
        chroma_range = 1.0
        chroma_mean_distance = 1.0
    neutral_score = _plateau_score(chroma_mean_distance, 0.0, SPHERE_NEUTRAL_SEED_MAX_CHROMA_DISTANCE, 0.04)
    brightness_score = _plateau_score(interior_mean, 0.10, 0.72, 0.22)
    support_score = min(edge_support / 0.22, 1.0)
    radius_score = _plateau_score(radius_ratio, 0.08, 0.18, 0.08)
    shape_score = _plateau_score(float(shape_circularity if shape_circularity is not None else 0.70), 0.28, 1.0, 0.25)
    aspect_score = _plateau_score(float(aspect_ratio if aspect_ratio is not None else 1.0), 0.0, 1.45, 0.60)
    target_signature_score = min((contrast * max(interior_mean, 0.0)) / 0.08, 1.0)
    weighted_terms = np.asarray(
        [
            radius_score * 1.10,
            edge_score * 1.00,
            support_score * 1.00,
            texture_score * 0.95,
            shading_score * 1.15,
            contrast_score * 1.50,
            target_signature_score * 1.30,
            neutral_score * 1.00,
            brightness_score * 1.00,
            zone_score * 0.80,
            shape_score * 0.90,
            aspect_score * 0.60,
        ],
        dtype=np.float32,
    )
    confidence = float(weighted_terms.sum() / 11.30)
    radius_sane = radius_ratio >= SPHERE_HARD_GATE_MIN_RADIUS_RATIO and radius_ratio <= 0.30
    shape_sane = (
        float(shape_circularity if shape_circularity is not None else 0.70) >= SPHERE_SHAPE_MIN_CIRCULARITY
        and float(aspect_ratio if aspect_ratio is not None else 1.0) <= SPHERE_SHAPE_MAX_ASPECT_RATIO
    )
    contrast_present = contrast >= 0.015 or internal_shading >= 0.010
    interior_neutral = chroma_mean_distance <= SPHERE_NEUTRAL_SEED_MAX_CHROMA_DISTANCE
    brightness_sane = interior_mean >= SPHERE_NEUTRAL_SEED_MIN_LUMINANCE and interior_mean <= 0.60
    critical_veto_reasons: List[str] = []
    if not radius_sane:
        critical_veto_reasons.append("radius_outside_expected_range")
    if not shape_sane:
        critical_veto_reasons.append("shape_not_sane")
    if not interior_neutral:
        critical_veto_reasons.append("neutral_material_not_present")
    if not brightness_sane:
        critical_veto_reasons.append("brightness_not_sane")
    if critical_veto_reasons:
        confidence = 0.0
    return {
        "roi": {"cx": float(roi.cx), "cy": float(roi.cy), "r": radius},
        "confidence": confidence,
        "confidence_label": _sphere_detection_label(confidence),
        "source": detection_source,
        "radius_ratio": radius_ratio,
        "center_ratio": {"x": center_x_ratio, "y": center_y_ratio},
        "edge_strength": float(edge_strength),
        "edge_support": float(edge_support),
        "hough_accumulator": float(accumulator),
        "interior_luminance_mean": interior_mean,
        "interior_luminance_stddev": interior_stddev,
        "surround_luminance_mean": ring_mean,
        "contrast_to_surround": contrast,
        "internal_shading": float(internal_shading),
        "neutral_rgb_mean": [float(value) for value in interior_rgb_mean.tolist()],
        "neutral_rgb_range": float(chroma_range),
        "neutral_chromaticity_distance": float(chroma_mean_distance),
        "zone_inside_score": zone_score,
        "fit_residual_pixels": float(max(radius * 0.08, 2.5)),
        "fit_residual_ratio": float(max(0.08, 2.5 / max(radius, 1.0))),
        "target_signature_score": float(target_signature_score),
        "shape_circularity": None if shape_circularity is None else float(shape_circularity),
        "aspect_ratio": None if aspect_ratio is None else float(aspect_ratio),
        "validation": {
            "radius_sane": bool(radius_sane),
            "placement_free": True,
            "zones_inside": zone_score >= 0.85,
            "contrast_present": bool(contrast_present),
            "interior_neutral": bool(interior_neutral),
            "brightness_sane": bool(brightness_sane),
            "critical_veto_reasons": list(critical_veto_reasons),
        },
    }


def _sphere_detection_uses_fallback(
    detection_source: object,
    *,
    manual_assist_used: bool = False,
    reused_from_original: bool = False,
) -> bool:
    normalized = str(detection_source or "").strip().lower()
    return bool(manual_assist_used or reused_from_original or normalized in SPHERE_DETECTION_FALLBACK_SOURCES)


def _display_domain_luminance(region: np.ndarray) -> np.ndarray:
    rgb = np.asarray(region[..., :3], dtype=np.float32)
    return (rgb[..., 0] * 0.2126) + (rgb[..., 1] * 0.7152) + (rgb[..., 2] * 0.0722)


def _sphere_measurement_radius(detected_radius: float) -> float:
    return float(max(float(detected_radius) * SPHERE_MEASUREMENT_RADIUS_RATIO, 1.0))


def _sphere_hero_patch_radius(detected_radius: float) -> float:
    return float(max(float(detected_radius) * SPHERE_HERO_PATCH_RADIUS_RATIO, 1.0))


def _weighted_trimmed_mean(values: np.ndarray, weights: np.ndarray, *, trim_fraction: float) -> Optional[float]:
    sample_values = np.asarray(values, dtype=np.float32).reshape(-1)
    sample_weights = np.asarray(weights, dtype=np.float32).reshape(-1)
    finite_mask = np.isfinite(sample_values) & np.isfinite(sample_weights) & (sample_weights > 0.0)
    if not np.any(finite_mask):
        return None
    sample_values = sample_values[finite_mask]
    sample_weights = sample_weights[finite_mask]
    if sample_values.size == 0:
        return None
    order = np.argsort(sample_values, kind="mergesort")
    ordered_values = sample_values[order]
    ordered_weights = sample_weights[order]
    trim = float(np.clip(trim_fraction, 0.0, 0.49))
    if trim > 0.0:
        cumulative = np.cumsum(ordered_weights)
        total_weight = float(cumulative[-1])
        if total_weight <= 0.0:
            return None
        lower = total_weight * trim
        upper = total_weight * (1.0 - trim)
        keep_mask = (cumulative >= lower) & (cumulative <= upper)
        if not np.any(keep_mask):
            keep_mask = np.ones_like(ordered_values, dtype=bool)
        ordered_values = ordered_values[keep_mask]
        ordered_weights = ordered_weights[keep_mask]
    weight_total = float(np.sum(ordered_weights))
    if weight_total <= 0.0:
        return None
    return float(np.sum(ordered_values * ordered_weights) / weight_total)


def _compute_sphere_center_patch(
    region: np.ndarray,
    cx: float,
    cy: float,
    measurement_radius: float,
    *,
    luminance: Optional[np.ndarray] = None,
) -> Optional[Dict[str, object]]:
    rgb = np.asarray(region[..., :3], dtype=np.float32)
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        return None
    height, width = rgb.shape[:2]
    detected_radius = float(measurement_radius) / SPHERE_MEASUREMENT_RADIUS_RATIO
    hero_radius = _sphere_hero_patch_radius(detected_radius)
    yy, xx = np.meshgrid(np.arange(height, dtype=np.float32), np.arange(width, dtype=np.float32), indexing="ij")
    distance = np.sqrt(((xx - float(cx)) ** 2) + ((yy - float(cy)) ** 2))
    inside_measurement = distance <= float(measurement_radius)
    inside_patch = distance <= float(hero_radius)
    mask = inside_measurement & inside_patch
    if not np.any(mask):
        return None
    patch_luminance = np.asarray(_display_domain_luminance(rgb) if luminance is None else luminance, dtype=np.float32)
    radial_weight = np.clip(1.0 - (distance / max(float(hero_radius), 1e-6)), 0.0, 1.0)
    mask &= np.isfinite(patch_luminance)
    mask &= np.all(np.isfinite(rgb), axis=2)
    if not np.any(mask):
        return None
    luminance_values = patch_luminance[mask]
    rgb_values = rgb[mask]
    weights = radial_weight[mask]
    center_luminance = _weighted_trimmed_mean(luminance_values, weights, trim_fraction=SPHERE_HERO_PATCH_TRIM_FRACTION)
    center_rgb = [
        _weighted_trimmed_mean(rgb_values[:, channel], weights, trim_fraction=SPHERE_HERO_PATCH_TRIM_FRACTION)
        for channel in range(3)
    ]
    if center_luminance is None or any(value is None for value in center_rgb):
        return None
    if not np.isfinite(center_luminance):
        return None
    center_rgb_values = np.asarray(center_rgb, dtype=np.float32)
    if not np.all(np.isfinite(center_rgb_values)):
        return None
    sample_count = int(np.count_nonzero(mask))
    if sample_count < SPHERE_HERO_PATCH_MIN_PIXEL_COUNT:
        return None
    return {
        "center_luminance": float(center_luminance),
        "center_rgb": (float(center_rgb_values[0]), float(center_rgb_values[1]), float(center_rgb_values[2])),
        "sample_count": sample_count,
        "radius_px": float(hero_radius),
        "measurement_radius_px": float(measurement_radius),
        "center_x": float(cx),
        "center_y": float(cy),
    }


def _display_domain_neutral_chromaticity(region: np.ndarray) -> Dict[str, np.ndarray]:
    rgb = np.asarray(region[..., :3], dtype=np.float32)
    channel_sum = np.sum(rgb, axis=2)
    safe_sum = np.where(channel_sum > 1e-6, channel_sum, 1.0)
    r_chrom = np.divide(rgb[..., 0], safe_sum, out=np.zeros_like(channel_sum), where=channel_sum > 1e-6)
    g_chrom = np.divide(rgb[..., 1], safe_sum, out=np.zeros_like(channel_sum), where=channel_sum > 1e-6)
    chroma_distance = np.sqrt((r_chrom - (1.0 / 3.0)) ** 2 + (g_chrom - (1.0 / 3.0)) ** 2)
    return {
        "channel_sum": channel_sum.astype(np.float32),
        "r_chrom": r_chrom.astype(np.float32),
        "g_chrom": g_chrom.astype(np.float32),
        "distance": chroma_distance.astype(np.float32),
    }


def _build_sphere_region_state(
    region: np.ndarray,
    *,
    sigma: float = 2.0,
    low_threshold: float = 0.03,
    high_threshold: float = 0.12,
) -> Dict[str, object]:
    rgb = np.asarray(region[..., :3], dtype=np.float32)
    height, width = rgb.shape[:2]
    yy, xx = np.meshgrid(np.arange(height, dtype=np.float32), np.arange(width, dtype=np.float32), indexing="ij")
    grayscale = np.clip(color.rgb2gray(rgb), 0.0, 1.0).astype(np.float32)
    luminance = np.asarray(_display_domain_luminance(rgb), dtype=np.float32)
    chromaticity = _display_domain_neutral_chromaticity(rgb)
    grad_y, grad_x = np.gradient(luminance)
    gradient_magnitude = np.sqrt((grad_x * grad_x) + (grad_y * grad_y)).astype(np.float32)
    gray_u8 = None
    if cv2 is not None:
        gray_u8 = cv2.cvtColor(_opencv_gray_u8(rgb), cv2.COLOR_RGB2GRAY)
    edge_map = feature.canny(
        grayscale,
        sigma=float(sigma),
        low_threshold=float(low_threshold),
        high_threshold=float(high_threshold),
    )
    return {
        "rgb": rgb,
        "height": int(height),
        "width": int(width),
        "yy": yy,
        "xx": xx,
        "grayscale": grayscale,
        "gray_u8": gray_u8,
        "luminance": luminance,
        "chromaticity": chromaticity,
        "grad_x": np.asarray(grad_x, dtype=np.float32),
        "grad_y": np.asarray(grad_y, dtype=np.float32),
        "gradient_magnitude": gradient_magnitude,
        "edge_maps": {
            (round(float(sigma), 3), round(float(low_threshold), 4), round(float(high_threshold), 4)): edge_map
        },
    }


def _sphere_region_edge_map(
    region: np.ndarray,
    *,
    region_state: Optional[Dict[str, object]] = None,
    sigma: float = 2.0,
    low_threshold: float = 0.03,
    high_threshold: float = 0.12,
) -> np.ndarray:
    if region_state is None:
        region_state = _build_sphere_region_state(
            region,
            sigma=sigma,
            low_threshold=low_threshold,
            high_threshold=high_threshold,
        )
    edge_maps = dict(region_state.get("edge_maps") or {})
    key = (round(float(sigma), 3), round(float(low_threshold), 4), round(float(high_threshold), 4))
    if key not in edge_maps:
        edge_maps[key] = feature.canny(
            np.asarray(region_state.get("grayscale"), dtype=np.float32),
            sigma=float(sigma),
            low_threshold=float(low_threshold),
            high_threshold=float(high_threshold),
        )
        region_state["edge_maps"] = edge_maps
    return np.asarray(edge_maps[key], dtype=bool)


def _circle_residuals(
    params: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
) -> np.ndarray:
    cx, cy, radius = [float(value) for value in np.asarray(params, dtype=np.float64)]
    return np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2) - radius


def _fit_circle_from_three_points(points: np.ndarray) -> Optional[Tuple[float, float, float]]:
    if points.shape[0] < 3:
        return None
    x1, y1 = [float(value) for value in points[0]]
    x2, y2 = [float(value) for value in points[1]]
    x3, y3 = [float(value) for value in points[2]]
    determinant = 2.0 * ((x1 * (y2 - y3)) + (x2 * (y3 - y1)) + (x3 * (y1 - y2)))
    if abs(determinant) <= 1e-6:
        return None
    squared_1 = (x1 * x1) + (y1 * y1)
    squared_2 = (x2 * x2) + (y2 * y2)
    squared_3 = (x3 * x3) + (y3 * y3)
    cx = (
        (squared_1 * (y2 - y3))
        + (squared_2 * (y3 - y1))
        + (squared_3 * (y1 - y2))
    ) / determinant
    cy = (
        (squared_1 * (x3 - x2))
        + (squared_2 * (x1 - x3))
        + (squared_3 * (x2 - x1))
    ) / determinant
    radius = math.hypot(x1 - cx, y1 - cy)
    if radius <= 0.0 or not np.isfinite(radius):
        return None
    return (float(cx), float(cy), float(radius))


def _circle_edge_support(
    edge_map: np.ndarray,
    roi: SphereROI,
    *,
    gradient_magnitude: Optional[np.ndarray] = None,
    samples: int = 48,
    tolerance_pixels: int = 2,
) -> float:
    if samples <= 0:
        return 0.0
    height, width = edge_map.shape[:2]
    gradient_floor = SPHERE_EDGE_SUPPORT_MIN_GRADIENT
    if gradient_magnitude is not None and gradient_magnitude.shape[:2] == edge_map.shape[:2]:
        yy, xx = np.indices(edge_map.shape[:2], dtype=np.float32)
        distance = np.sqrt((xx - float(roi.cx)) ** 2 + (yy - float(roi.cy)) ** 2)
        band_mask = np.abs(distance - float(roi.r)) <= float(max(tolerance_pixels + 1, 3))
        band_values = np.asarray(gradient_magnitude[band_mask], dtype=np.float32)
        band_values = band_values[np.isfinite(band_values)]
        band_values = band_values[band_values > 0.0]
        if band_values.size:
            gradient_floor = max(
                float(np.quantile(band_values, SPHERE_EDGE_SUPPORT_GRADIENT_QUANTILE)),
                SPHERE_EDGE_SUPPORT_MIN_GRADIENT,
            )
    supported = 0
    for index in range(samples):
        angle = (2.0 * math.pi * float(index)) / float(samples)
        sample_x = int(round(float(roi.cx) + (float(roi.r) * math.cos(angle))))
        sample_y = int(round(float(roi.cy) + (float(roi.r) * math.sin(angle))))
        x0 = max(0, sample_x - tolerance_pixels)
        x1 = min(width, sample_x + tolerance_pixels + 1)
        y0 = max(0, sample_y - tolerance_pixels)
        y1 = min(height, sample_y + tolerance_pixels + 1)
        if x1 <= x0 or y1 <= y0:
            continue
        local_edge = bool(np.any(edge_map[y0:y1, x0:x1]))
        local_gradient = False
        if gradient_magnitude is not None and gradient_magnitude.shape[:2] == edge_map.shape[:2]:
            local_gradient = bool(np.max(np.asarray(gradient_magnitude[y0:y1, x0:x1], dtype=np.float32)) >= gradient_floor)
        if local_edge or local_gradient:
            supported += 1
    return float(supported) / float(max(samples, 1))


def _merge_sphere_candidate_clusters(
    candidates: List[Dict[str, object]],
) -> List[Dict[str, object]]:
    if not candidates:
        return []
    clusters: List[List[Dict[str, object]]] = []
    for candidate in sorted(candidates, key=lambda item: float(item.get("confidence", 0.0) or 0.0), reverse=True):
        roi = _coerce_sphere_roi(candidate.get("roi"))
        if roi is None or float(roi.r) <= 0.0:
            continue
        matched_cluster: Optional[List[Dict[str, object]]] = None
        for cluster in clusters:
            cluster_roi = _coerce_sphere_roi(cluster[0].get("roi"))
            if cluster_roi is None:
                continue
            distance = math.hypot(float(roi.cx) - float(cluster_roi.cx), float(roi.cy) - float(cluster_roi.cy))
            radius_delta = abs(float(roi.r) - float(cluster_roi.r))
            max_radius = max(float(roi.r), float(cluster_roi.r), 1.0)
            if (
                distance <= max_radius * SPHERE_CANDIDATE_DEDUP_DISTANCE_FACTOR
                and radius_delta <= max_radius * SPHERE_CANDIDATE_DEDUP_RADIUS_FACTOR
            ):
                matched_cluster = cluster
                break
        if matched_cluster is None:
            clusters.append([copy.deepcopy(dict(candidate))])
        else:
            matched_cluster.append(copy.deepcopy(dict(candidate)))
    merged: List[Dict[str, object]] = []
    for cluster in clusters:
        best = max(cluster, key=lambda item: float(item.get("confidence", 0.0) or 0.0))
        merged_item = copy.deepcopy(dict(best))
        sources = sorted({str(item.get("source") or "") for item in cluster if str(item.get("source") or "").strip()})
        weights = np.asarray(
            [max(float(item.get("confidence", 0.0) or 0.0), 0.05) for item in cluster],
            dtype=np.float32,
        )
        if float(np.sum(weights)) <= 1e-6:
            weights = np.ones((len(cluster),), dtype=np.float32)
        roi_values = []
        for item in cluster:
            roi = _coerce_sphere_roi(item.get("roi"))
            if roi is not None:
                roi_values.append((float(roi.cx), float(roi.cy), float(roi.r)))
        if roi_values:
            roi_array = np.asarray(roi_values, dtype=np.float32)
            merged_item["roi"] = {
                "cx": float(np.average(roi_array[:, 0], weights=weights[: roi_array.shape[0]])),
                "cy": float(np.average(roi_array[:, 1], weights=weights[: roi_array.shape[0]])),
                "r": float(np.average(roi_array[:, 2], weights=weights[: roi_array.shape[0]])),
            }
        merged_item["proposal_sources"] = sources
        merged_item["proposal_count"] = int(len(cluster))
        merged_item["proposal_cluster"] = [
            {
                "source": str(item.get("source") or ""),
                "confidence": float(item.get("confidence", 0.0) or 0.0),
                "roi": copy.deepcopy(dict(item.get("roi") or {})),
            }
            for item in cluster
        ]
        merged_item["confidence"] = float(
            max(float(merged_item.get("confidence", 0.0) or 0.0), min(1.0, float(np.max(weights)) + (0.04 * float(len(sources) - 1))))
        )
        merged.append(merged_item)
    return sorted(merged, key=lambda item: float(item.get("confidence", 0.0) or 0.0), reverse=True)


def _rank_sphere_candidates_for_refinement(candidates: List[Dict[str, object]]) -> List[Dict[str, object]]:
    ranked: List[Dict[str, object]] = []
    for candidate in list(candidates or []):
        item = copy.deepcopy(dict(candidate))
        ranking_score = (
            float(item.get("confidence", 0.0) or 0.0)
            + (0.05 * float(item.get("proposal_count", 1) or 1))
            + (0.08 * float(item.get("edge_support", 0.0) or 0.0))
            + (0.04 * float(item.get("zone_inside_score", 0.0) or 0.0))
        )
        item["_refinement_rank"] = float(ranking_score)
        ranked.append(item)
    return sorted(ranked, key=lambda item: float(item.get("_refinement_rank", 0.0) or 0.0), reverse=True)


def _refine_sphere_candidate_geometry(
    region: np.ndarray,
    candidate: Dict[str, object],
    *,
    region_state: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    item = copy.deepcopy(dict(candidate))
    roi = _coerce_sphere_roi(item.get("roi"))
    if roi is None or float(roi.r) <= 0.0:
        return item
    if region_state is None:
        region_state = _build_sphere_region_state(region)
    edge_map = _sphere_region_edge_map(region, region_state=region_state)
    yy = np.asarray(region_state.get("yy"), dtype=np.float32)
    xx = np.asarray(region_state.get("xx"), dtype=np.float32)
    gradient_magnitude = np.asarray(region_state.get("gradient_magnitude"), dtype=np.float32)
    distance = np.sqrt((xx - float(roi.cx)) ** 2 + (yy - float(roi.cy)) ** 2)
    tolerance = max(2.0, float(roi.r) * 0.05)
    band_mask = np.abs(distance - float(roi.r)) <= tolerance
    candidate_mask = band_mask & edge_map
    if np.count_nonzero(candidate_mask) < 16:
        item["refined_geometry"] = {
            "performed": False,
            "reason": "insufficient_edge_support",
            "edge_support": float(_circle_edge_support(edge_map, roi, gradient_magnitude=gradient_magnitude)),
            "fit_residual_pixels": float(max(float(roi.r) * 0.12, tolerance)),
        }
        return item
    gradient_floor = max(float(np.quantile(gradient_magnitude[candidate_mask], 0.40)), 0.002)
    support_mask = candidate_mask & (gradient_magnitude >= gradient_floor)
    if np.count_nonzero(support_mask) < 16:
        support_mask = candidate_mask
    points_y, points_x = np.nonzero(support_mask)
    residuals = np.abs(distance[support_mask] - float(roi.r))
    angles = np.arctan2(points_y.astype(np.float32) - float(roi.cy), points_x.astype(np.float32) - float(roi.cx))
    selected_indices: List[int] = []
    bin_edges = np.linspace(-math.pi, math.pi, SPHERE_CANDIDATE_REFINEMENT_EDGE_BINS + 1, dtype=np.float32)
    for bin_index in range(SPHERE_CANDIDATE_REFINEMENT_EDGE_BINS):
        left = float(bin_edges[bin_index])
        right = float(bin_edges[bin_index + 1])
        mask = (angles >= left) & (angles < right)
        if not np.any(mask):
            continue
        local_indices = np.nonzero(mask)[0]
        best_local = local_indices[int(np.argmin(residuals[local_indices]))]
        selected_indices.append(int(best_local))
    if not selected_indices:
        selected_indices = list(range(min(points_x.size, SPHERE_CANDIDATE_REFINEMENT_MAX_EDGE_POINTS)))
    if len(selected_indices) > SPHERE_CANDIDATE_REFINEMENT_MAX_EDGE_POINTS:
        selected_indices = selected_indices[:SPHERE_CANDIDATE_REFINEMENT_MAX_EDGE_POINTS]
    xs = points_x[np.asarray(selected_indices, dtype=np.int32)].astype(np.float64)
    ys = points_y[np.asarray(selected_indices, dtype=np.int32)].astype(np.float64)
    inlier_mask = np.ones(xs.shape[0], dtype=bool)
    if xs.shape[0] >= 24:
        best_inliers = inlier_mask
        best_inlier_count = int(np.count_nonzero(best_inliers))
        points = np.column_stack((xs, ys))
        sample_span = max(points.shape[0] - 2, 1)
        for trial in range(SPHERE_CANDIDATE_REFINEMENT_RANSAC_TRIALS):
            start = (trial * 7) % sample_span
            sample = np.asarray(
                [
                    points[start % points.shape[0]],
                    points[(start + max(points.shape[0] // 3, 1)) % points.shape[0]],
                    points[(start + max((2 * points.shape[0]) // 3, 2)) % points.shape[0]],
                ],
                dtype=np.float64,
            )
            circle = _fit_circle_from_three_points(sample)
            if circle is None:
                continue
            residual_vector = np.abs(
                _circle_residuals(np.asarray(circle, dtype=np.float64), xs, ys)
            )
            trial_inliers = residual_vector <= SPHERE_CANDIDATE_REFINEMENT_INLIER_TOLERANCE
            trial_count = int(np.count_nonzero(trial_inliers))
            if trial_count > best_inlier_count:
                best_inliers = trial_inliers
                best_inlier_count = trial_count
        if int(np.count_nonzero(best_inliers)) >= 12:
            inlier_mask = best_inliers
    xs_fit = xs[inlier_mask]
    ys_fit = ys[inlier_mask]
    if xs_fit.shape[0] < 8:
        xs_fit = xs
        ys_fit = ys
    initial = np.asarray([float(roi.cx), float(roi.cy), float(roi.r)], dtype=np.float64)
    refined_roi = roi
    fit_residual_pixels = float(max(np.median(np.abs(_circle_residuals(initial, xs_fit, ys_fit))), tolerance))
    try:
        optimized = least_squares(
            lambda params: _circle_residuals(params, xs_fit, ys_fit),
            initial,
            bounds=(
                np.asarray([float(roi.cx) - (float(roi.r) * 0.35), float(roi.cy) - (float(roi.r) * 0.35), max(float(roi.r) * 0.60, 1.0)], dtype=np.float64),
                np.asarray([float(roi.cx) + (float(roi.r) * 0.35), float(roi.cy) + (float(roi.r) * 0.35), float(roi.r) * 1.40], dtype=np.float64),
            ),
            loss="huber",
            f_scale=max(1.6, tolerance),
            max_nfev=SPHERE_CANDIDATE_REFINEMENT_MAX_NFEV,
        )
        if optimized.success:
            refined_candidate = SphereROI(
                cx=float(optimized.x[0]),
                cy=float(optimized.x[1]),
                r=float(optimized.x[2]),
            )
            if refined_candidate.r > 0.0:
                refined_roi = refined_candidate
                fit_residual_pixels = float(np.median(np.abs(_circle_residuals(optimized.x, xs_fit, ys_fit))))
    except Exception:
        refined_roi = roi
    edge_support = _circle_edge_support(edge_map, refined_roi, gradient_magnitude=gradient_magnitude)
    item["roi"] = {"cx": float(refined_roi.cx), "cy": float(refined_roi.cy), "r": float(refined_roi.r)}
    item["edge_support"] = float(edge_support)
    item["fit_residual_pixels"] = float(fit_residual_pixels)
    item["fit_residual_ratio"] = float(fit_residual_pixels / max(float(refined_roi.r), 1.0))
    item["refined_geometry"] = {
        "performed": True,
        "edge_sample_count": int(xs_fit.shape[0]),
        "edge_support": float(edge_support),
        "fit_residual_pixels": float(fit_residual_pixels),
        "fit_residual_ratio": float(fit_residual_pixels / max(float(refined_roi.r), 1.0)),
        "ransac_prefilter_used": bool(np.count_nonzero(inlier_mask) != xs.shape[0]),
        "initial_roi": {"cx": float(roi.cx), "cy": float(roi.cy), "r": float(roi.r)},
    }
    return item


def _candidate_geometry_shape_proxy_ok(
    candidate: Dict[str, object],
    *,
    neutral_region: Dict[str, object],
    neutral_consistency_score: float,
) -> bool:
    validation = dict(candidate.get("validation") or {})
    if validation.get("radius_sane") is False:
        return False
    fragmented = bool(neutral_region.get("fragmented"))
    if fragmented:
        return False
    region_expansion_score = float(neutral_region.get("region_expansion_score", 0.0) or 0.0)
    centroid_distance_norm = float(neutral_region.get("centroid_distance_norm", 1.0) or 1.0)
    neutral_aspect_ratio = float(neutral_region.get("aspect_ratio", 99.0) or 99.0)
    solidity = float(neutral_region.get("solidity", 0.0) or 0.0)
    if (
        region_expansion_score < SPHERE_SHAPE_PROXY_MIN_REGION_EXPANSION
        or neutral_consistency_score < SPHERE_SHAPE_PROXY_MIN_NEUTRAL_CONSISTENCY
        or centroid_distance_norm > SPHERE_SHAPE_PROXY_MAX_CENTROID_DISTANCE_NORM
        or neutral_aspect_ratio > SPHERE_SHAPE_PROXY_MAX_ASPECT_RATIO
        or solidity < SPHERE_SHAPE_PROXY_MIN_SOLIDITY
    ):
        return False
    candidate_aspect_ratio = float(candidate.get("aspect_ratio", neutral_aspect_ratio) or neutral_aspect_ratio)
    candidate_circularity = candidate.get("shape_circularity")
    if candidate_circularity is not None:
        return (
            float(candidate_circularity) >= SPHERE_SHAPE_PROXY_MIN_GEOMETRY_CIRCULARITY
            and candidate_aspect_ratio <= SPHERE_SHAPE_PROXY_MAX_ASPECT_RATIO
        )
    return bool(candidate.get("opencv_hough_circle"))


def _refined_geometry_authoritative(
    candidate: Dict[str, object],
    *,
    metrics: Dict[str, float | int | bool],
) -> bool:
    validation = dict(candidate.get("validation") or {})
    if validation.get("radius_sane") is False:
        return False
    if bool(metrics.get("edge_aligned")) or bool(metrics.get("fragmented")):
        return False
    return (
        float(metrics.get("edge_support", 0.0) or 0.0) >= SPHERE_REFINED_GEOMETRY_OVERRIDE_MIN_EDGE_SUPPORT
        and float(metrics.get("fit_residual_ratio", 1.0) or 1.0) <= SPHERE_REFINED_GEOMETRY_OVERRIDE_MAX_FIT_RESIDUAL_RATIO
        and float(metrics.get("radial_gradient_coherence", 0.0) or 0.0) >= SPHERE_REFINED_GEOMETRY_OVERRIDE_MIN_RADIAL_COHERENCE
        and float(metrics.get("radial_profile_score", 0.0) or 0.0) >= SPHERE_REFINED_GEOMETRY_OVERRIDE_MIN_RADIAL_PROFILE
        and float(metrics.get("neutral_consistency_score", 0.0) or 0.0) >= SPHERE_REFINED_GEOMETRY_OVERRIDE_MIN_NEUTRAL_CONSISTENCY
        and float(metrics.get("region_expansion_score", 0.0) or 0.0) >= SPHERE_REFINED_GEOMETRY_OVERRIDE_MIN_REGION_EXPANSION
    )


def _sphere_radial_gradient_coherence(
    region: np.ndarray,
    roi: SphereROI,
    *,
    region_state: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    if region_state is None:
        region_state = _build_sphere_region_state(region)
    height = int(region_state.get("height", region.shape[0]))
    width = int(region_state.get("width", region.shape[1]))
    if min(height, width) < 8:
        return {"score": 0.0, "support": 0, "mean_alignment": 0.0}
    yy = np.asarray(region_state.get("yy"), dtype=np.float32)
    xx = np.asarray(region_state.get("xx"), dtype=np.float32)
    dx = xx - float(roi.cx)
    dy = yy - float(roi.cy)
    distance = np.sqrt((dx * dx) + (dy * dy))
    annulus = (distance >= float(roi.r) * 0.62) & (distance <= float(roi.r) * 1.06)
    if not np.any(annulus):
        return {"score": 0.0, "support": 0, "mean_alignment": 0.0}
    grad_y = np.asarray(region_state.get("grad_y"), dtype=np.float32)
    grad_x = np.asarray(region_state.get("grad_x"), dtype=np.float32)
    gradient_magnitude = np.asarray(region_state.get("gradient_magnitude"), dtype=np.float32)
    annulus_gradients = gradient_magnitude[annulus]
    if annulus_gradients.size == 0:
        return {"score": 0.0, "support": 0, "mean_alignment": 0.0}
    gradient_floor = max(float(np.quantile(annulus_gradients, 0.58)), 0.006)
    radial_norm = np.sqrt((dx * dx) + (dy * dy))
    valid = annulus & (gradient_magnitude >= gradient_floor) & (radial_norm >= 1.0)
    support = int(np.count_nonzero(valid))
    if support < 72:
        return {"score": 0.0, "support": support, "mean_alignment": 0.0}
    radial_x = np.divide(dx, radial_norm, out=np.zeros_like(dx), where=radial_norm > 1e-6)
    radial_y = np.divide(dy, radial_norm, out=np.zeros_like(dy), where=radial_norm > 1e-6)
    alignment = np.abs(
        np.divide(
            (grad_x * radial_x) + (grad_y * radial_y),
            gradient_magnitude,
            out=np.zeros_like(gradient_magnitude),
            where=gradient_magnitude > 1e-6,
        )
    )
    weights = gradient_magnitude[valid]
    mean_alignment = (
        float(np.average(alignment[valid], weights=weights))
        if np.any(weights > 0.0)
        else float(np.mean(alignment[valid]))
    )
    return {
        "score": float(mean_alignment),
        "support": support,
        "mean_alignment": float(mean_alignment),
        "gradient_floor": float(gradient_floor),
    }


def _sphere_candidate_neutral_region_probe(
    region: np.ndarray,
    roi: SphereROI,
    *,
    region_state: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    if region_state is None:
        region_state = _build_sphere_region_state(region)
    height = int(region_state.get("height", region.shape[0]))
    width = int(region_state.get("width", region.shape[1]))
    if min(height, width) < 12:
        return {
            "score": 0.0,
            "region_expansion_score": 0.0,
            "shape_score": 0.0,
            "seed_area_fraction": 0.0,
            "seed_bbox": {},
            "ring_neutral_fractions": [],
            "edge_aligned": False,
        }
    yy = np.asarray(region_state.get("yy"), dtype=np.float32)
    xx = np.asarray(region_state.get("xx"), dtype=np.float32)
    distance = np.sqrt((xx - float(roi.cx)) ** 2 + (yy - float(roi.cy)) ** 2)
    interior_mask = distance <= float(roi.r) * 0.96
    if np.count_nonzero(interior_mask) < 64:
        return {
            "score": 0.0,
            "region_expansion_score": 0.0,
            "shape_score": 0.0,
            "seed_area_fraction": 0.0,
            "seed_bbox": {},
            "ring_neutral_fractions": [],
            "edge_aligned": False,
        }
    luminance = np.asarray(region_state.get("luminance"), dtype=np.float32)
    chromaticity = dict(region_state.get("chromaticity") or _display_domain_neutral_chromaticity(region))
    chroma_distance = np.asarray(chromaticity["distance"], dtype=np.float32)
    channel_sum = np.asarray(chromaticity["channel_sum"], dtype=np.float32)
    interior_chroma = chroma_distance[interior_mask]
    interior_luminance = luminance[interior_mask]
    if interior_chroma.size == 0 or interior_luminance.size == 0:
        return {
            "score": 0.0,
            "region_expansion_score": 0.0,
            "shape_score": 0.0,
            "seed_area_fraction": 0.0,
            "seed_bbox": {},
            "ring_neutral_fractions": [],
            "edge_aligned": False,
        }
    chroma_threshold = min(
        SPHERE_NEUTRAL_SEED_MAX_CHROMA_DISTANCE,
        max(0.018, float(np.quantile(interior_chroma, 0.35)) + 0.008),
    )
    luminance_lo = max(SPHERE_NEUTRAL_SEED_MIN_LUMINANCE, float(np.quantile(interior_luminance, 0.10)) - 0.02)
    luminance_hi = float(np.quantile(interior_luminance, 0.94)) + 0.03
    neutral_mask = (
        interior_mask
        & (chroma_distance <= chroma_threshold)
        & (luminance >= luminance_lo)
        & (luminance <= luminance_hi)
        & (channel_sum >= SPHERE_NEUTRAL_SEED_MIN_CHANNEL_SUM)
    )
    minimum_component_size = max(24, int(round(float(np.count_nonzero(interior_mask)) * SPHERE_NEUTRAL_SEED_MIN_COMPONENT_AREA_FRACTION)))
    neutral_mask = morphology.binary_opening(neutral_mask, morphology.disk(2))
    neutral_mask = morphology.binary_closing(neutral_mask, morphology.disk(3))
    neutral_mask = morphology.remove_small_objects(neutral_mask, min_size=minimum_component_size)
    if not np.any(neutral_mask):
        return {
            "score": 0.0,
            "region_expansion_score": 0.0,
            "shape_score": 0.0,
            "seed_area_fraction": 0.0,
            "seed_bbox": {},
            "ring_neutral_fractions": [],
            "edge_aligned": False,
            "chroma_threshold": float(chroma_threshold),
        }
    labeled = measure.label(neutral_mask)
    component_count = int(labeled.max())
    expected_area = float(np.count_nonzero(interior_mask))
    best_component = None
    best_component_score = -1.0
    for component in measure.regionprops(labeled):
        centroid_y, centroid_x = component.centroid
        centroid_distance_norm = math.hypot(float(centroid_x) - float(roi.cx), float(centroid_y) - float(roi.cy)) / max(float(roi.r), 1.0)
        component_area_fraction = float(component.area) / max(expected_area, 1.0)
        perimeter = float(component.perimeter or 0.0)
        circularity = 0.0 if perimeter <= 1e-6 else float((4.0 * math.pi * float(component.area)) / float(perimeter ** 2))
        component_score = (
            min(component_area_fraction / 0.22, 1.0) * 1.30
            + max(0.0, 1.0 - (centroid_distance_norm / 0.45)) * 1.10
            + min(max(circularity, 0.0) / 0.72, 1.0) * 0.85
        )
        if component_score > best_component_score:
            best_component_score = component_score
            best_component = component
    if best_component is None:
        return {
            "score": 0.0,
            "region_expansion_score": 0.0,
            "shape_score": 0.0,
            "seed_area_fraction": 0.0,
            "seed_bbox": {},
            "ring_neutral_fractions": [],
            "edge_aligned": False,
            "chroma_threshold": float(chroma_threshold),
        }
    component_mask = labeled == int(best_component.label)
    min_row, min_col, max_row, max_col = best_component.bbox
    bbox_width = float(max_col - min_col)
    bbox_height = float(max_row - min_row)
    aspect_ratio = max(bbox_width / max(bbox_height, 1.0), bbox_height / max(bbox_width, 1.0))
    perimeter = float(best_component.perimeter or 0.0)
    circularity = 0.0 if perimeter <= 1e-6 else float((4.0 * math.pi * float(best_component.area)) / float(perimeter ** 2))
    solidity = float(best_component.solidity or 0.0)
    coords = np.argwhere(component_mask)
    radial_distances = np.sqrt((coords[:, 1].astype(np.float32) - float(roi.cx)) ** 2 + (coords[:, 0].astype(np.float32) - float(roi.cy)) ** 2)
    radial_cv = float(np.std(radial_distances) / max(float(np.mean(radial_distances)), 1.0)) if radial_distances.size else 1.0
    centroid_distance_norm = math.hypot(float(best_component.centroid[1]) - float(roi.cx), float(best_component.centroid[0]) - float(roi.cy)) / max(float(roi.r), 1.0)
    touches_crop_edge = bool(min_col <= 1 or min_row <= 1 or max_col >= width - 1 or max_row >= height - 1)
    ring_neutral_fractions: List[float] = []
    ring_edges = np.linspace(0.18, 0.92, 6, dtype=np.float32)
    previous_edge = 0.0
    for edge in ring_edges:
        annulus = (distance >= float(roi.r) * previous_edge) & (distance <= float(roi.r) * float(edge))
        annulus &= interior_mask
        if np.count_nonzero(annulus) < 24:
            ring_neutral_fractions.append(0.0)
        else:
            ring_neutral_fractions.append(float(np.count_nonzero(component_mask & annulus)) / float(np.count_nonzero(annulus)))
        previous_edge = float(edge)
    largest_drop = 0.0
    for left, right in zip(ring_neutral_fractions, ring_neutral_fractions[1:]):
        largest_drop = max(largest_drop, float(left) - float(right))
    region_expansion_score = float(
        np.clip(
            np.mean(np.asarray(ring_neutral_fractions, dtype=np.float32)) * max(0.0, 1.0 - max(0.0, largest_drop - 0.38)),
            0.0,
            1.0,
        )
    )
    circularity_score = min(max((circularity - SPHERE_SHAPE_MIN_CIRCULARITY) / 0.24, 0.0), 1.0)
    aspect_score = max(0.0, 1.0 - max(0.0, aspect_ratio - 1.0) / max(SPHERE_SHAPE_MAX_ASPECT_RATIO - 1.0, 1e-6))
    centroid_score = max(0.0, 1.0 - (centroid_distance_norm / SPHERE_SHAPE_MAX_CENTROID_DISTANCE_NORM))
    radial_variance_score = max(0.0, 1.0 - (radial_cv / SPHERE_SHAPE_MAX_RADIAL_CV))
    solidity_score = max(0.0, min((solidity - SPHERE_NEUTRAL_REGION_MIN_SOLIDITY) / 0.12, 1.0))
    edge_score = 0.0 if touches_crop_edge else 1.0
    shape_score = float(
        np.mean(
            np.asarray(
                [circularity_score, aspect_score, centroid_score, radial_variance_score, solidity_score, edge_score],
                dtype=np.float32,
            )
        )
    )
    chroma_consistency = max(
        0.0,
        1.0 - (float(np.std(chroma_distance[component_mask])) / 0.05 if np.any(component_mask) else 1.0),
    )
    fragmented = bool(component_count > SPHERE_NEUTRAL_SEED_MAX_FRAGMENT_COUNT)
    if fragmented or circularity < SPHERE_SHAPE_MIN_CIRCULARITY or aspect_ratio > SPHERE_SHAPE_MAX_ASPECT_RATIO or radial_cv > SPHERE_SHAPE_MAX_RADIAL_CV or solidity < SPHERE_NEUTRAL_REGION_MIN_SOLIDITY or centroid_distance_norm > SPHERE_SHAPE_MAX_CENTROID_DISTANCE_NORM:
        shape_score = 0.0
    if fragmented or largest_drop > 0.52 or max(ring_neutral_fractions or [0.0]) < SPHERE_NEUTRAL_EXPANSION_MIN_RING_FRACTION:
        region_expansion_score = 0.0
    score = float(
        np.clip(
            np.mean(
                np.asarray(
                    [
                        region_expansion_score * 1.25,
                        shape_score * 1.10,
                        chroma_consistency * 0.90,
                    ],
                    dtype=np.float32,
                )
            ),
            0.0,
            1.0,
        )
    )
    return {
        "score": score,
        "region_expansion_score": float(region_expansion_score),
        "shape_score": float(shape_score),
        "seed_area_fraction": float(best_component.area) / max(expected_area, 1.0),
        "seed_bbox": {
            "x0": int(min_col),
            "y0": int(min_row),
            "x1": int(max_col),
            "y1": int(max_row),
        },
        "ring_neutral_fractions": [float(value) for value in ring_neutral_fractions],
        "centroid_distance_norm": float(centroid_distance_norm),
        "radial_cv": float(radial_cv),
        "circularity": float(circularity),
        "aspect_ratio": float(aspect_ratio),
        "solidity": float(solidity),
        "component_count": int(component_count),
        "fragmented": fragmented,
        "largest_ring_drop": float(largest_drop),
        "edge_aligned": bool(touches_crop_edge),
        "chroma_threshold": float(chroma_threshold),
        "chroma_consistency": float(chroma_consistency),
    }


def _sphere_candidate_radial_luminance_profile(
    region: np.ndarray,
    roi: SphereROI,
    *,
    region_state: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    if region_state is None:
        region_state = _build_sphere_region_state(region)
    height = int(region_state.get("height", region.shape[0]))
    width = int(region_state.get("width", region.shape[1]))
    if min(height, width) < 12:
        return {"score": 0.0, "ring_ire_values": [], "flat_profile": True}
    yy = np.asarray(region_state.get("yy"), dtype=np.float32)
    xx = np.asarray(region_state.get("xx"), dtype=np.float32)
    distance = np.sqrt((xx - float(roi.cx)) ** 2 + (yy - float(roi.cy)) ** 2)
    interior_mask = distance <= float(roi.r) * 0.96
    if np.count_nonzero(interior_mask) < 48:
        return {"score": 0.0, "ring_ire_values": [], "flat_profile": True}
    luminance = np.asarray(region_state.get("luminance"), dtype=np.float32)
    ring_edges = np.linspace(0.12, 0.92, 6, dtype=np.float32)
    previous_edge = 0.0
    ring_values: List[float] = []
    ring_stddevs: List[float] = []
    for edge in ring_edges:
        annulus = (distance >= float(roi.r) * previous_edge) & (distance <= float(roi.r) * float(edge))
        annulus &= interior_mask
        if np.count_nonzero(annulus) < 24:
            ring_values.append(0.0)
            ring_stddevs.append(0.0)
        else:
            ring_pixels = luminance[annulus]
            ring_values.append(float(np.mean(ring_pixels)))
            ring_stddevs.append(float(np.std(ring_pixels)))
        previous_edge = float(edge)
    ring_range = float(max(ring_values) - min(ring_values)) if ring_values else 0.0
    angular_variation = float(np.mean(ring_stddevs)) if ring_stddevs else 0.0
    ring_deltas = [abs(float(right) - float(left)) for left, right in zip(ring_values, ring_values[1:])]
    second_deltas = [abs(float(right) - float(left)) for left, right in zip(ring_deltas, ring_deltas[1:])]
    sign_changes = 0
    previous_sign = 0
    for delta in [float(right) - float(left) for left, right in zip(ring_values, ring_values[1:])]:
        sign = 1 if delta > 1e-4 else -1 if delta < -1e-4 else 0
        if previous_sign and sign and sign != previous_sign:
            sign_changes += 1
        if sign:
            previous_sign = sign
    structure_score = min(max(ring_range / 0.055, angular_variation / 0.020), 1.0)
    smoothness_score = max(0.0, 1.0 - (float(np.mean(second_deltas)) / 0.028 if second_deltas else 0.0))
    monotonicity_score = 1.0 if sign_changes <= 1 else max(0.0, 1.0 - (float(sign_changes - 1) / 3.0))
    score = float(np.clip((structure_score * 0.50) + (smoothness_score * 0.25) + (monotonicity_score * 0.25), 0.0, 1.0))
    if ring_range < 0.012 and angular_variation < 0.010:
        score = 0.0
    return {
        "score": score,
        "ring_luminance_values": [float(value) for value in ring_values],
        "ring_ire_values": [float(_ire_from_log2_luminance(math.log2(max(value, 1e-6)))) for value in ring_values],
        "ring_range": float(ring_range),
        "ring_stddevs": [float(value) for value in ring_stddevs],
        "angular_variation": float(angular_variation),
        "flat_profile": bool(ring_range < 0.012 and angular_variation < 0.010),
        "smoothness_score": float(smoothness_score),
        "monotonicity_score": float(monotonicity_score),
        "sign_changes": int(sign_changes),
    }


def _detect_sphere_candidates_in_region_hwc(
    region: np.ndarray,
    *,
    search_bounds: Optional[Tuple[int, int, int, int]] = None,
    detection_source: str,
    sigma: float = 2.0,
    low_threshold: float = 0.03,
    high_threshold: float = 0.12,
    region_state: Optional[Dict[str, object]] = None,
) -> List[Dict[str, object]]:
    if region_state is None:
        region_state = _build_sphere_region_state(
            region,
            sigma=sigma,
            low_threshold=low_threshold,
            high_threshold=high_threshold,
        )
    height = int(region_state.get("height", region.shape[0]))
    width = int(region_state.get("width", region.shape[1]))
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
    grayscale = np.asarray(region_state.get("grayscale"), dtype=np.float32)[y0:y1, x0:x1]
    edge_map_local = _sphere_region_edge_map(
        region,
        region_state=region_state,
        sigma=sigma,
        low_threshold=low_threshold,
        high_threshold=high_threshold,
    )[y0:y1, x0:x1]
    if not np.any(edge_map_local):
        return []
    min_radius = max(24, int(round(float(min_dimension) * 0.10)))
    max_radius = max(min_radius + 4, int(round(float(min_dimension) * 0.30)))
    radii = np.arange(min_radius, max_radius + 1, 4, dtype=np.int32)
    if radii.size == 0:
        return []
    hough_space = transform.hough_circle(edge_map_local, radii)
    accumulators, centers_x, centers_y, detected_radii = transform.hough_circle_peaks(
        hough_space,
        radii,
        total_num_peaks=12,
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
                edge_map=edge_map_local if search_bounds is None else np.pad(edge_map_local, ((y0, height - y1), (x0, width - x1)), mode="constant"),
                accumulator=float(accumulator),
                detection_source=detection_source,
                region_state=region_state,
            )
        )
    return _merge_sphere_candidate_clusters(candidates)


def _detect_neutral_blob_candidates_in_region_hwc(
    region: np.ndarray,
    *,
    detection_source: str,
    region_state: Optional[Dict[str, object]] = None,
) -> List[Dict[str, object]]:
    if region_state is None:
        region_state = _build_sphere_region_state(region)
    height = int(region_state.get("height", region.shape[0]))
    width = int(region_state.get("width", region.shape[1]))
    min_dimension = min(height, width)
    if min_dimension < 64:
        return []
    luminance = np.asarray(region_state.get("luminance"), dtype=np.float32)
    chromaticity = dict(region_state.get("chromaticity") or _display_domain_neutral_chromaticity(region))
    chroma_distance = np.asarray(chromaticity["distance"], dtype=np.float32)
    channel_sum = np.asarray(chromaticity["channel_sum"], dtype=np.float32)
    luminance_threshold = max(SPHERE_NEUTRAL_SEED_MIN_LUMINANCE, float(np.quantile(luminance, 0.16)))
    chroma_threshold = min(
        SPHERE_NEUTRAL_SEED_MAX_CHROMA_DISTANCE,
        max(0.018, float(np.quantile(chroma_distance, 0.32)) + 0.008),
    )
    mask = (
        (luminance >= luminance_threshold)
        & (chroma_distance <= chroma_threshold)
        & (channel_sum >= SPHERE_NEUTRAL_SEED_MIN_CHANNEL_SUM)
    )
    open_radius = max(2, int(round(float(min_dimension) * 0.004)))
    close_radius = max(4, int(round(float(min_dimension) * 0.007)))
    mask = morphology.binary_opening(mask, morphology.disk(open_radius))
    mask = morphology.binary_closing(mask, morphology.disk(close_radius))
    mask = morphology.remove_small_objects(mask, min_size=max(256, int(round(float(min_dimension * min_dimension) * 0.004))))
    if not np.any(mask):
        return []
    edge_map = _sphere_region_edge_map(region, region_state=region_state)
    labeled = measure.label(mask)
    candidates: List[Dict[str, object]] = []
    image_area = float(max(height * width, 1))
    for component in measure.regionprops(labeled, intensity_image=luminance):
        min_row, min_col, max_row, max_col = component.bbox
        box_width = float(max_col - min_col)
        box_height = float(max_row - min_row)
        if box_width <= 0.0 or box_height <= 0.0:
            continue
        area_ratio = float(component.area) / image_area
        if area_ratio >= 0.12:
            continue
        touches_borders = int(min_col <= 0) + int(min_row <= 0) + int(max_col >= width) + int(max_row >= height)
        if touches_borders > 1:
            continue
        aspect_ratio = max(box_width / max(box_height, 1.0), box_height / max(box_width, 1.0))
        if aspect_ratio > 1.7:
            continue
        circularity = 0.0
        if component.perimeter > 1e-6:
            circularity = float((4.0 * math.pi * float(component.area)) / float(component.perimeter ** 2))
        if circularity < 0.18:
            continue
        equivalent_radius = float(component.equivalent_diameter) * 0.5
        radius = max(equivalent_radius, min(box_width, box_height) * 0.42)
        radius_ratio = radius / float(max(min_dimension, 1))
        if not (0.07 <= radius_ratio <= 0.30):
            continue
        roi = SphereROI(cx=float(component.centroid[1]), cy=float(component.centroid[0]), r=float(radius))
        candidate = _evaluate_detected_sphere_roi(
            region,
            roi=roi,
            edge_map=edge_map,
            accumulator=0.0,
            detection_source=detection_source,
            shape_circularity=circularity,
            aspect_ratio=aspect_ratio,
            region_state=region_state,
        )
        candidate["component_area_ratio"] = float(area_ratio)
        candidate["component_bbox"] = {
            "x0": int(min_col),
            "y0": int(min_row),
            "x1": int(max_col),
            "y1": int(max_row),
        }
        candidates.append(candidate)
    return _merge_sphere_candidate_clusters(sorted(candidates, key=lambda item: float(item.get("confidence", 0.0) or 0.0), reverse=True))


def _opencv_gray_u8(region: np.ndarray) -> np.ndarray:
    clipped = np.clip(region[..., :3], 0.0, 1.0)
    return np.clip(np.round(clipped * 255.0), 0.0, 255.0).astype(np.uint8)


def _focus_gray_from_rgb(region: np.ndarray) -> np.ndarray:
    rgb = np.clip(np.asarray(region, dtype=np.float32)[..., :3], 0.0, 1.0)
    if rgb.size == 0:
        return np.zeros((1, 1), dtype=np.float32)
    if cv2 is not None:
        gray_u8 = cv2.cvtColor(_opencv_gray_u8(rgb), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    else:
        gray_u8 = np.clip(color.rgb2gray(rgb), 0.0, 1.0).astype(np.float32)
    lo = float(np.quantile(gray_u8, 0.01))
    hi = float(np.quantile(gray_u8, 0.99))
    if hi > lo + 1e-6:
        return np.clip((gray_u8 - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
    return gray_u8


def _focus_metric_bundle(region: np.ndarray) -> Dict[str, float]:
    gray = _focus_gray_from_rgb(region)
    gray64 = np.asarray(gray, dtype=np.float64)
    if cv2 is not None:
        lap = cv2.Laplacian(gray64, cv2.CV_64F)
        laplacian_variance = float(np.var(lap))
        sobel_x = cv2.Sobel(gray64, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray64, cv2.CV_64F, 0, 1, ksize=3)
    else:
        sobel_x = np.gradient(gray64, axis=1)
        sobel_y = np.gradient(gray64, axis=0)
        laplacian_variance = float(np.var(np.gradient(sobel_x, axis=1) + np.gradient(sobel_y, axis=0)))
    tenengrad = float(np.mean((sobel_x * sobel_x) + (sobel_y * sobel_y)))
    centered = gray64 - float(np.mean(gray64))
    spectrum = np.fft.fftshift(np.fft.fft2(centered))
    power = np.abs(spectrum) ** 2
    total_power = float(np.sum(power))
    if total_power <= 1e-12:
        fft_high_frequency_energy = 0.0
    else:
        height, width = gray64.shape
        yy, xx = np.indices((height, width), dtype=np.float32)
        cy = (float(height) - 1.0) * 0.5
        cx = (float(width) - 1.0) * 0.5
        radius = np.sqrt(((yy - cy) / max(float(height), 1.0)) ** 2 + ((xx - cx) / max(float(width), 1.0)) ** 2)
        high_mask = radius >= 0.18
        fft_high_frequency_energy = float(np.sum(power[high_mask]) / total_power)
    return {
        "laplacian_variance": laplacian_variance,
        "tenengrad": tenengrad,
        "fft_high_frequency_energy": fft_high_frequency_energy,
    }


def _rank_values(values: List[float]) -> List[float]:
    indexed = sorted(enumerate(values), key=lambda item: float(item[1]))
    ranks = [0.0] * len(values)
    index = 0
    while index < len(indexed):
        end = index + 1
        while end < len(indexed) and math.isclose(float(indexed[end][1]), float(indexed[index][1]), abs_tol=1e-9):
            end += 1
        average_rank = ((index + 1) + end) / 2.0
        for item_index in range(index, end):
            ranks[indexed[item_index][0]] = float(average_rank)
        index = end
    return ranks


def _spearman_rank_correlation(left: List[float], right: List[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        return 1.0
    left_rank = np.asarray(_rank_values([float(value) for value in left]), dtype=np.float64)
    right_rank = np.asarray(_rank_values([float(value) for value in right]), dtype=np.float64)
    left_centered = left_rank - float(np.mean(left_rank))
    right_centered = right_rank - float(np.mean(right_rank))
    denominator = float(np.linalg.norm(left_centered) * np.linalg.norm(right_centered))
    if denominator <= 1e-12:
        return 1.0
    return float(np.clip(np.dot(left_centered, right_centered) / denominator, -1.0, 1.0))


def _detect_focus_chart_roi_hwc(image: np.ndarray) -> Dict[str, object]:
    if cv2 is None:
        return {"found": False, "reason": "opencv_unavailable", "confidence": 0.0}
    rgb_u8 = _opencv_gray_u8(image)
    height, width = rgb_u8.shape[:2]
    if min(height, width) < 128:
        return {"found": False, "reason": "image_too_small", "confidence": 0.0}
    gray = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0.0)
    edges = cv2.Canny(blurred, 80, 180)
    contours, _hierarchy = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cell_candidates: List[Dict[str, float]] = []
    image_area = float(max(height * width, 1))
    for contour in contours[:2048]:
        area = float(cv2.contourArea(contour))
        if area <= 24.0 or area >= image_area * 0.02:
            continue
        perimeter = float(cv2.arcLength(contour, True))
        if perimeter <= 1e-6:
            continue
        approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
        if len(approx) not in {4, 5, 6}:
            continue
        x, y, w_box, h_box = cv2.boundingRect(contour)
        if min(w_box, h_box) < 6:
            continue
        aspect = float(w_box) / max(float(h_box), 1.0)
        fill_ratio = area / max(float(w_box * h_box), 1.0)
        if not (0.72 <= aspect <= 1.35 and 0.48 <= fill_ratio <= 0.98):
            continue
        patch = rgb_u8[y:y + h_box, x:x + w_box, :]
        if patch.size == 0:
            continue
        hsv = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV)
        saturation_mean = float(np.mean(hsv[..., 1])) / 255.0
        value_std = float(np.std(hsv[..., 2])) / 255.0
        cell_candidates.append(
            {
                "x": float(x),
                "y": float(y),
                "w": float(w_box),
                "h": float(h_box),
                "cx": float(x + (w_box * 0.5)),
                "cy": float(y + (h_box * 0.5)),
                "area": area,
                "aspect": aspect,
                "fill_ratio": fill_ratio,
                "saturation_mean": saturation_mean,
                "value_std": value_std,
            }
        )
    if len(cell_candidates) < 8:
        return {"found": False, "reason": "insufficient_square_cells", "confidence": 0.0, "cell_count": len(cell_candidates)}
    median_size = float(np.median([max(item["w"], item["h"]) for item in cell_candidates]))
    distance_limit = max(median_size * 3.6, 18.0)
    used = [False] * len(cell_candidates)
    clusters: List[List[int]] = []
    for start_index in range(len(cell_candidates)):
        if used[start_index]:
            continue
        stack = [start_index]
        used[start_index] = True
        component: List[int] = []
        while stack:
            current = stack.pop()
            component.append(current)
            current_candidate = cell_candidates[current]
            for neighbor in range(len(cell_candidates)):
                if used[neighbor]:
                    continue
                candidate = cell_candidates[neighbor]
                distance = math.hypot(float(current_candidate["cx"]) - float(candidate["cx"]), float(current_candidate["cy"]) - float(candidate["cy"]))
                if distance <= distance_limit:
                    used[neighbor] = True
                    stack.append(neighbor)
        clusters.append(component)
    best_cluster: Optional[Dict[str, object]] = None
    for cluster_indices in clusters:
        if len(cluster_indices) < 8:
            continue
        members = [cell_candidates[index] for index in cluster_indices]
        xs = [float(item["x"]) for item in members]
        ys = [float(item["y"]) for item in members]
        x2 = [float(item["x"] + item["w"]) for item in members]
        y2 = [float(item["y"] + item["h"]) for item in members]
        bbox_x0 = min(xs)
        bbox_y0 = min(ys)
        bbox_x1 = max(x2)
        bbox_y1 = max(y2)
        bbox_w = max(bbox_x1 - bbox_x0, 1.0)
        bbox_h = max(bbox_y1 - bbox_y0, 1.0)
        bbox_area = bbox_w * bbox_h
        mean_sat = float(np.mean([float(item["saturation_mean"]) for item in members]))
        mean_value_std = float(np.mean([float(item["value_std"]) for item in members]))
        density = float(sum(float(item["area"]) for item in members) / max(bbox_area, 1.0))
        cluster_score = (
            float(len(members)) * 1.6
            + min(density / 0.45, 1.0) * 4.0
            + min(mean_sat / 0.18, 1.0) * 2.0
            + min(mean_value_std / 0.10, 1.0) * 1.5
        )
        cluster_payload = {
            "members": members,
            "bbox": {
                "x0": int(round(bbox_x0)),
                "y0": int(round(bbox_y0)),
                "x1": int(round(bbox_x1)),
                "y1": int(round(bbox_y1)),
            },
            "cluster_score": float(cluster_score),
            "cell_count": int(len(members)),
            "density": float(density),
        }
        if best_cluster is None or float(cluster_payload["cluster_score"]) > float(best_cluster["cluster_score"]):
            best_cluster = cluster_payload
    if best_cluster is None:
        return {"found": False, "reason": "no_chart_cluster", "confidence": 0.0, "cell_count": len(cell_candidates)}
    bbox = dict(best_cluster["bbox"])
    color_w = max(int(bbox["x1"]) - int(bbox["x0"]), 1)
    color_h = max(int(bbox["y1"]) - int(bbox["y0"]), 1)
    roi_x0 = max(0, int(round(int(bbox["x0"]) - (color_w * 1.45))))
    roi_x1 = min(width, int(round(int(bbox["x1"]) + (color_w * 0.18))))
    roi_y0 = max(0, int(round(int(bbox["y0"]) - (color_h * 0.18))))
    roi_y1 = min(height, int(round(int(bbox["y1"]) + (color_h * 0.20))))
    roi_w = max(roi_x1 - roi_x0, 1)
    roi_h = max(roi_y1 - roi_y0, 1)
    confidence = min(
        0.99,
        0.35
        + min(float(best_cluster["cell_count"]) / 18.0, 1.0) * 0.35
        + min(float(best_cluster["density"]) / 0.45, 1.0) * 0.15
        + min(float(best_cluster["cluster_score"]) / 18.0, 1.0) * 0.15,
    )
    return {
        "found": True,
        "method": "opencv_colorchecker_cluster",
        "confidence": float(confidence),
        "cell_count": int(best_cluster["cell_count"]),
        "cell_boxes": [
            {
                "x0": int(round(float(item["x"]))),
                "y0": int(round(float(item["y"]))),
                "x1": int(round(float(item["x"] + item["w"]))),
                "y1": int(round(float(item["y"] + item["h"]))),
            }
            for item in list(best_cluster["members"])
        ],
        "colorchecker_bbox": bbox,
        "roi": {
            "x0": int(roi_x0),
            "y0": int(roi_y0),
            "x1": int(roi_x1),
            "y1": int(roi_y1),
            "w": int(roi_w),
            "h": int(roi_h),
            "normalized": {
                "x": float(roi_x0) / float(max(width, 1)),
                "y": float(roi_y0) / float(max(height, 1)),
                "w": float(roi_w) / float(max(width, 1)),
                "h": float(roi_h) / float(max(height, 1)),
            },
        },
    }


def _detect_sphere_candidates_opencv_hough_hwc(
    region: np.ndarray,
    *,
    detection_source: str,
    search_bounds: Optional[Tuple[int, int, int, int]] = None,
    region_state: Optional[Dict[str, object]] = None,
) -> List[Dict[str, object]]:
    if cv2 is None:
        return []
    if region_state is None:
        region_state = _build_sphere_region_state(region)
    height = int(region_state.get("height", region.shape[0]))
    width = int(region_state.get("width", region.shape[1]))
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
    gray_source = np.asarray(region_state.get("gray_u8")) if region_state.get("gray_u8") is not None else cv2.cvtColor(_opencv_gray_u8(region), cv2.COLOR_RGB2GRAY)
    gray = gray_source[y0:y1, x0:x1]
    blurred = cv2.GaussianBlur(gray, (9, 9), 1.8)
    edge_map_local = cv2.Canny(blurred, 40, 120) > 0
    edge_map = np.pad(edge_map_local, ((y0, height - y1), (x0, width - x1)), mode="constant")
    min_radius = max(24, int(round(float(min_dimension) * 0.10)))
    max_radius = max(min_radius + 4, int(round(float(min_dimension) * 0.30)))
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.15,
        minDist=max(float(min_radius) * 1.3, 24.0),
        param1=100.0,
        param2=18.0,
        minRadius=min_radius,
        maxRadius=max_radius,
    )
    if circles is None:
        return []
    candidates: List[Dict[str, object]] = []
    for circle in np.asarray(circles[0], dtype=np.float32)[:16]:
        center_x, center_y, radius = [float(value) for value in circle.tolist()]
        roi = SphereROI(cx=center_x + float(x0), cy=center_y + float(y0), r=radius)
        accumulator = max(0.0, min(1.0, radius / float(max(max_radius, 1))))
        candidate = _evaluate_detected_sphere_roi(
            region,
            roi=roi,
            edge_map=edge_map,
            accumulator=accumulator,
            detection_source=detection_source,
            region_state=region_state,
        )
        candidate["opencv_hough_circle"] = {
            "cx": float(roi.cx),
            "cy": float(roi.cy),
            "r": float(roi.r),
        }
        candidates.append(candidate)
    return _merge_sphere_candidate_clusters(sorted(candidates, key=lambda item: float(item.get("confidence", 0.0) or 0.0), reverse=True))


def _detect_sphere_candidates_opencv_hough_alt_hwc(
    region: np.ndarray,
    *,
    detection_source: str,
    search_bounds: Optional[Tuple[int, int, int, int]] = None,
    region_state: Optional[Dict[str, object]] = None,
) -> List[Dict[str, object]]:
    if cv2 is None or not hasattr(cv2, "HOUGH_GRADIENT_ALT"):
        return []
    if region_state is None:
        region_state = _build_sphere_region_state(region)
    height = int(region_state.get("height", region.shape[0]))
    width = int(region_state.get("width", region.shape[1]))
    if search_bounds is None:
        x0, y0, x1, y1 = 0, 0, width, height
    else:
        x0, y0, x1, y1 = search_bounds
    if x1 <= x0 or y1 <= y0:
        return []
    gray_source = np.asarray(region_state.get("gray_u8")) if region_state.get("gray_u8") is not None else cv2.cvtColor(_opencv_gray_u8(region), cv2.COLOR_RGB2GRAY)
    gray = gray_source[y0:y1, x0:x1]
    min_dimension = min(gray.shape[:2])
    if min_dimension < 64:
        return []
    blurred = cv2.GaussianBlur(gray, (7, 7), 1.4)
    edge_map_local = cv2.Canny(blurred, 32, 96) > 0
    edge_map = np.pad(edge_map_local, ((y0, height - y1), (x0, width - x1)), mode="constant")
    min_radius = max(24, int(round(float(min_dimension) * 0.10)))
    max_radius = max(min_radius + 4, int(round(float(min_dimension) * 0.30)))
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT_ALT,
        dp=1.35,
        minDist=max(float(min_radius) * 1.2, 20.0),
        param1=220.0,
        param2=0.82,
        minRadius=min_radius,
        maxRadius=max_radius,
    )
    if circles is None:
        return []
    candidates: List[Dict[str, object]] = []
    for circle in np.asarray(circles[0], dtype=np.float32)[:16]:
        center_x, center_y, radius = [float(value) for value in circle.tolist()]
        roi = SphereROI(cx=center_x + float(x0), cy=center_y + float(y0), r=radius)
        candidate = _evaluate_detected_sphere_roi(
            region,
            roi=roi,
            edge_map=edge_map,
            accumulator=min(1.0, max(0.0, float(radius) / float(max(max_radius, 1)))),
            detection_source=detection_source,
            region_state=region_state,
        )
        candidate["opencv_hough_alt_circle"] = {
            "cx": float(roi.cx),
            "cy": float(roi.cy),
            "r": float(roi.r),
        }
        candidates.append(candidate)
    return _merge_sphere_candidate_clusters(sorted(candidates, key=lambda item: float(item.get("confidence", 0.0) or 0.0), reverse=True))


def _detect_sphere_candidates_opencv_contours_hwc(
    region: np.ndarray,
    *,
    detection_source: str,
    region_state: Optional[Dict[str, object]] = None,
) -> List[Dict[str, object]]:
    if cv2 is None:
        return []
    if region_state is None:
        region_state = _build_sphere_region_state(region)
    height = int(region_state.get("height", region.shape[0]))
    width = int(region_state.get("width", region.shape[1]))
    min_dimension = min(height, width)
    if min_dimension < 64:
        return []
    gray = np.asarray(region_state.get("gray_u8")) if region_state.get("gray_u8") is not None else cv2.cvtColor(_opencv_gray_u8(region), cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 1.4)
    luminance = np.clip(np.asarray(region_state.get("luminance"), dtype=np.float32), 0.0, 1.0)
    saturation = np.clip(np.max(region[..., :3], axis=2) - np.min(region[..., :3], axis=2), 0.0, 1.0)
    bright_threshold = int(round(np.quantile(gray, 0.70)))
    bright_threshold = max(72, min(bright_threshold, 208))
    neutral_threshold = min(0.14, max(0.06, float(np.quantile(saturation, 0.55) + 0.03)))
    mask = ((gray >= bright_threshold) & (saturation <= neutral_threshold) & (luminance <= 0.85)).astype(np.uint8) * 255
    kernel_open = max(3, int(round(float(min_dimension) * 0.01)))
    if kernel_open % 2 == 0:
        kernel_open += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_open, kernel_open))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    edge_map = cv2.Canny(blurred, 40, 120) > 0
    image_area = float(max(height * width, 1))
    candidates: List[Dict[str, object]] = []
    for contour in contours[:48]:
        area = float(cv2.contourArea(contour))
        if area <= 0.0:
            continue
        area_ratio = area / image_area
        if not (0.004 <= area_ratio <= 0.12):
            continue
        perimeter = float(cv2.arcLength(contour, True))
        if perimeter <= 1e-6:
            continue
        circularity = float((4.0 * math.pi * area) / float(perimeter * perimeter))
        if circularity < 0.18:
            continue
        (center_x, center_y), radius = cv2.minEnclosingCircle(contour)
        radius_ratio = float(radius) / float(max(min_dimension, 1))
        if not (0.07 <= radius_ratio <= 0.30):
            continue
        x, y, w_box, h_box = cv2.boundingRect(contour)
        aspect_ratio = float(max(float(w_box), float(h_box)) / max(min(float(w_box), float(h_box)), 1.0))
        if aspect_ratio > 1.75:
            continue
        roi = SphereROI(cx=float(center_x), cy=float(center_y), r=float(radius))
        candidate = _evaluate_detected_sphere_roi(
            region,
            roi=roi,
            edge_map=edge_map,
            accumulator=min(area_ratio / 0.08, 1.0),
            detection_source=detection_source,
            shape_circularity=circularity,
            aspect_ratio=aspect_ratio,
            region_state=region_state,
        )
        candidate["opencv_contour"] = {
            "area_ratio": float(area_ratio),
            "circularity": float(circularity),
            "bbox": {"x": int(x), "y": int(y), "w": int(w_box), "h": int(h_box)},
        }
        candidates.append(candidate)
    return _merge_sphere_candidate_clusters(sorted(candidates, key=lambda item: float(item.get("confidence", 0.0) or 0.0), reverse=True))


def _sphere_candidate_profile_plausibility(
    region: np.ndarray,
    candidate: Dict[str, object],
    *,
    region_state: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    roi = _coerce_sphere_roi(candidate.get("roi"))
    if roi is None:
        return {
            "score": 0.0,
            "label": "IMPLAUSIBLE",
            "review_required": True,
            "valid": False,
            "reason": "missing_roi",
        }
    try:
        crop_bounds = _tight_sphere_crop_bounds(region, roi)
        cropped_region = region[crop_bounds["y0"]:crop_bounds["y1"], crop_bounds["x0"]:crop_bounds["x1"], :]
        if cropped_region.size == 0:
            raise ValueError("empty_crop")
        cropped_roi = SphereROI(
            cx=float(roi.cx) - float(crop_bounds["x0"]),
            cy=float(roi.cy) - float(crop_bounds["y0"]),
            r=float(roi.r),
        )
        stats = measure_sphere_zone_profile_statistics(np.transpose(cropped_region, (2, 0, 1)), cropped_roi, sampling_variant="refined")
    except Exception as exc:
        return {
            "score": 0.0,
            "label": "IMPLAUSIBLE",
            "review_required": True,
            "valid": False,
            "reason": f"profile_probe_failed:{exc}",
        }
    sample_1 = float(stats.get("sample_1_ire", stats.get("bright_ire", 0.0)) or 0.0)
    sample_2 = float(stats.get("sample_2_ire", stats.get("center_ire", 0.0)) or 0.0)
    sample_3 = float(stats.get("sample_3_ire", stats.get("dark_ire", 0.0)) or 0.0)
    center_ire = float(stats.get("center_ire", sample_2) or sample_2)
    zone_spread_ire = float(stats.get("zone_spread_ire", 0.0) or 0.0)
    valid_pixel_count = int(stats.get("valid_pixel_count", 0) or 0)
    expected_area = float(math.pi * max(float(cropped_roi.r), 1.0) * max(float(cropped_roi.r), 1.0))
    valid_pixel_fraction = float(valid_pixel_count / max(expected_area, 1.0))
    measured_rgb_payload = stats.get("measured_rgb_chromaticity")
    if measured_rgb_payload is None:
        measured_rgb_payload = [1.0 / 3.0] * 3
    measured_rgb = np.asarray(measured_rgb_payload, dtype=np.float32)
    chroma_range = float(np.max(measured_rgb) - np.min(measured_rgb)) if measured_rgb.size else 1.0
    chroma_distance_from_neutral = float(
        math.sqrt(
            (float(measured_rgb[0]) - (1.0 / 3.0)) ** 2
            + (float(measured_rgb[1]) - (1.0 / 3.0)) ** 2
        )
    ) if measured_rgb.size >= 2 else 1.0
    interior_stddev = float(math.sqrt(max(float(stats.get("roi_variance", 0.0) or 0.0), 0.0)))
    center_margin_ire = float(min(abs(sample_2 - sample_1), abs(sample_2 - sample_3)))
    cropped_region_state = _build_sphere_region_state(cropped_region) if region_state is None else _build_sphere_region_state(cropped_region)
    radial_coherence = _sphere_radial_gradient_coherence(cropped_region, cropped_roi, region_state=cropped_region_state)
    neutral_region = _sphere_candidate_neutral_region_probe(cropped_region, cropped_roi, region_state=cropped_region_state)
    radial_profile = _sphere_candidate_radial_luminance_profile(cropped_region, cropped_roi, region_state=cropped_region_state)
    radius_ratio = float(candidate.get("radius_ratio", 0.0) or 0.0)
    if radius_ratio <= 0.0:
        radius_ratio = float(roi.r) / float(max(min(region.shape[:2]), 1))
    center_score = _plateau_score(center_ire, 12.0, 62.0, 11.0)
    spread_score = _plateau_score(zone_spread_ire, 1.8, 34.0, 9.0)
    neutral_score = _plateau_score(chroma_distance_from_neutral, 0.0, SPHERE_NEUTRAL_SEED_MAX_CHROMA_DISTANCE, 0.04)
    count_score = min(float(valid_pixel_count) / 3200.0, 1.0)
    measurement_confidence = float(stats.get("confidence", 0.0) or 0.0)
    center_between = min(sample_1, sample_3) - 1.0 <= sample_2 <= max(sample_1, sample_3) + 1.0
    monotonic_score = 1.0 if center_between else 0.20
    sample_range_ire = float(max(sample_1, sample_2, sample_3) - min(sample_1, sample_2, sample_3))
    samples_equal = sample_range_ire <= SPHERE_HARD_GATE_MAX_SAMPLE_EQUALITY_IRE
    flat_high = center_ire >= SPHERE_HARD_GATE_HIGH_FLAT_IRE and sample_range_ire < 2.0
    flat_low = center_ire <= SPHERE_HARD_GATE_LOW_FLAT_IRE and sample_range_ire < 1.5
    plausible_samples = center_ire >= 10.0 and (
        valid_pixel_count >= SPHERE_HARD_GATE_MIN_VALID_PIXEL_COUNT
        or valid_pixel_fraction >= SPHERE_HARD_GATE_MIN_VALID_PIXEL_FRACTION
    )
    neutral_consistency_score = float(
        np.clip(
            np.mean(
                np.asarray(
                    [
                        neutral_score,
                        float(neutral_region.get("score", 0.0) or 0.0),
                        float(neutral_region.get("region_expansion_score", 0.0) or 0.0),
                    ],
                    dtype=np.float32,
                )
            ),
            0.0,
            1.0,
        )
    )
    shape_score = float(np.clip(neutral_region.get("shape_score", 0.0) or 0.0, 0.0, 1.0))
    radial_score = float(
        np.clip(
            np.mean(
                np.asarray(
                    [
                        float((radial_coherence.get("score") or 0.0)),
                        float(radial_profile.get("score", 0.0) or 0.0),
                    ],
                    dtype=np.float32,
                )
            ),
            0.0,
            1.0,
        )
    )
    exposure_validity_score = float(
        np.clip(
            np.mean(
                np.asarray(
                    [
                        center_score,
                        spread_score,
                        monotonic_score,
                        min(sample_range_ire / 4.0, 1.0),
                        0.0 if (samples_equal or flat_high or flat_low) else 1.0,
                    ],
                    dtype=np.float32,
                )
            ),
            0.0,
            1.0,
        )
    )
    edge_support = float(candidate.get("edge_support", 0.0) or 0.0)
    fit_residual_ratio = float(candidate.get("fit_residual_ratio", 1.0) or 1.0)
    edge_support_score = _plateau_score(edge_support, SPHERE_HARD_GATE_MIN_EDGE_SUPPORT, 1.0, 0.20)
    fit_residual_score = max(
        0.0,
        1.0 - (max(0.0, fit_residual_ratio - SPHERE_HARD_GATE_MAX_FIT_RESIDUAL_RATIO) / max(SPHERE_HARD_GATE_MAX_FIT_RESIDUAL_RATIO, 1e-6)),
    )
    geometry_support_score = float(
        np.clip(
            np.mean(
                np.asarray(
                    [
                        edge_support_score,
                        fit_residual_score,
                    ],
                    dtype=np.float32,
                )
            ),
            0.0,
            1.0,
        )
    )
    neutral_shape_proxy_ok = _candidate_geometry_shape_proxy_ok(
        candidate,
        neutral_region=neutral_region,
        neutral_consistency_score=neutral_consistency_score,
    ) or (
        float(neutral_region.get("region_expansion_score", 0.0) or 0.0) >= SPHERE_SHAPE_PROXY_MIN_REGION_EXPANSION
        and neutral_consistency_score >= SPHERE_SHAPE_PROXY_MIN_NEUTRAL_CONSISTENCY
        and int(neutral_region.get("component_count", 0) or 0) <= 1
        and not bool(neutral_region.get("fragmented"))
        and float(neutral_region.get("solidity", 0.0) or 0.0) >= SPHERE_SHAPE_PROXY_MIN_SOLIDITY
        and float(neutral_region.get("aspect_ratio", 99.0) or 99.0) <= SPHERE_SHAPE_PROXY_MAX_ASPECT_RATIO
        and float(neutral_region.get("centroid_distance_norm", 1.0) or 1.0) <= SPHERE_SHAPE_PROXY_MAX_CENTROID_DISTANCE_NORM
    )
    effective_shape_score = max(shape_score, 0.55 if neutral_shape_proxy_ok else 0.0)
    critical_failures: List[str] = []
    if neutral_consistency_score < SPHERE_HARD_GATE_MIN_NEUTRAL_CONSISTENCY:
        critical_failures.append("neutral_consistency_below_min")
    if float(neutral_region.get("region_expansion_score", 0.0) or 0.0) < SPHERE_HARD_GATE_MIN_REGION_EXPANSION:
        critical_failures.append("neutrality_collapses_on_expansion")
    if shape_score < SPHERE_HARD_GATE_MIN_SHAPE_SCORE and not neutral_shape_proxy_ok:
        critical_failures.append("non_circular_region")
    if radius_ratio < SPHERE_HARD_GATE_MIN_RADIUS_RATIO:
        critical_failures.append("radius_outside_expected_range")
    if radial_score < SPHERE_HARD_GATE_MIN_RADIAL_COHERENCE:
        critical_failures.append("radial_coherence_below_min")
    if float(radial_profile.get("score", 0.0) or 0.0) < SPHERE_HARD_GATE_MIN_RADIAL_PROFILE_SCORE:
        critical_failures.append("flat_luminance_profile")
    if edge_support < SPHERE_HARD_GATE_MIN_EDGE_SUPPORT:
        critical_failures.append("edge_support_below_min")
    if fit_residual_ratio > SPHERE_HARD_GATE_MAX_FIT_RESIDUAL_RATIO:
        critical_failures.append("fit_residual_above_max")
    if exposure_validity_score < SPHERE_HARD_GATE_MIN_EXPOSURE_VALIDITY or samples_equal or flat_high or flat_low:
        critical_failures.append("exposure_distribution_invalid")
    if float(neutral_region.get("seed_area_fraction", 0.0) or 0.0) < SPHERE_HARD_GATE_MIN_ROI_AREA_FRACTION:
        critical_failures.append("minimum_size_not_met")
    if not plausible_samples:
        critical_failures.append("insufficient_sample_support")
    component_product = float(
        np.clip(
            neutral_consistency_score * radial_score * effective_shape_score * exposure_validity_score * geometry_support_score,
            0.0,
            1.0,
        )
    )
    score = float(np.clip(math.sqrt(component_product), 0.0, 1.0))
    if critical_failures:
        score = 0.0
    else:
        if measurement_confidence < SPHERE_HARD_GATE_MIN_MEASUREMENT_CONFIDENCE:
            score *= max(0.35, measurement_confidence)
        if count_score < 0.25:
            score *= 0.5
    label = (
        "HIGH"
        if score >= SPHERE_DETECTION_PROFILE_HIGH
        else "MEDIUM"
        if score >= SPHERE_DETECTION_PROFILE_REVIEW
        else "LOW"
        if score >= SPHERE_DETECTION_PLAUSIBILITY_MIN
        else "IMPLAUSIBLE"
    )
    return {
        "score": score,
        "label": label,
        "review_required": score < SPHERE_DETECTION_PROFILE_REVIEW,
        "valid": score >= SPHERE_DETECTION_PLAUSIBILITY_MIN and plausible_samples and not critical_failures,
        "sample_1_ire": sample_1,
        "sample_2_ire": sample_2,
        "sample_3_ire": sample_3,
        "center_ire": center_ire,
        "zone_spread_ire": zone_spread_ire,
        "valid_pixel_count": valid_pixel_count,
        "valid_pixel_fraction": valid_pixel_fraction,
        "chroma_range": chroma_range,
        "chroma_distance_from_neutral": float(chroma_distance_from_neutral),
        "interior_stddev": interior_stddev,
        "center_margin_ire": center_margin_ire,
        "center_between_extremes": bool(center_between),
        "sample_range_ire": float(sample_range_ire),
        "samples_equal": bool(samples_equal),
        "flat_high": bool(flat_high),
        "flat_low": bool(flat_low),
        "measurement_confidence": measurement_confidence,
        "radius_ratio": float(radius_ratio),
        "plausible_samples": bool(plausible_samples),
        "radial_gradient_coherence": radial_coherence,
        "neutral_region": neutral_region,
        "radial_luminance_profile": radial_profile,
        "neutral_consistency_score": float(neutral_consistency_score),
        "shape_score": float(shape_score),
        "effective_shape_score": float(effective_shape_score),
        "radial_score": float(radial_score),
        "geometry_support_score": float(geometry_support_score),
        "exposure_validity_score": float(exposure_validity_score),
        "edge_support": float(edge_support),
        "fit_residual_ratio": float(fit_residual_ratio),
        "critical_failures": list(dict.fromkeys(critical_failures)),
    }


def _sphere_candidate_hard_gate(candidate: Dict[str, object], profile_probe: Dict[str, object]) -> Dict[str, object]:
    chroma_range_value = profile_probe.get("chroma_range")
    roi_payload = dict(candidate.get("roi") or {})
    reasons: List[str] = []
    validation = dict(candidate.get("validation") or {})
    radius_ratio = float(profile_probe.get("radius_ratio", candidate.get("radius_ratio", 0.0)) or 0.0)
    if radius_ratio <= 0.0 and {"r"} <= set(roi_payload):
        radius_ratio = float(roi_payload.get("r", 0.0) or 0.0) / float(max(float(min(candidate.get("region_height", 0.0) or 0.0, candidate.get("region_width", 0.0) or 0.0)), 1.0))
    metrics = {
        "zone_spread_ire": float(profile_probe.get("zone_spread_ire", 0.0) or 0.0),
        "chroma_range": float(chroma_range_value) if chroma_range_value is not None else 1.0,
        "valid_pixel_count": int(profile_probe.get("valid_pixel_count", 0) or 0),
        "valid_pixel_fraction": float(profile_probe.get("valid_pixel_fraction", 0.0) or 0.0),
        "center_ire": float(profile_probe.get("center_ire", 0.0) or 0.0),
        "interior_stddev": float(
            profile_probe.get("interior_stddev", candidate.get("interior_luminance_stddev", 0.0))
            or candidate.get("interior_luminance_stddev", 0.0)
            or 0.0
        ),
        "measurement_confidence": float(profile_probe.get("measurement_confidence", 0.0) or 0.0),
        "center_margin_ire": float(profile_probe.get("center_margin_ire", 0.0) or 0.0),
        "center_between_extremes": bool(profile_probe.get("center_between_extremes")),
        "sample_range_ire": float(profile_probe.get("sample_range_ire", 0.0) or 0.0),
        "samples_equal": bool(profile_probe.get("samples_equal")),
        "flat_high": bool(profile_probe.get("flat_high")),
        "flat_low": bool(profile_probe.get("flat_low")),
        "radial_gradient_coherence": float(
            ((profile_probe.get("radial_gradient_coherence") or {}).get("score"))
            or 0.0
        ),
        "radial_gradient_support": int(
            ((profile_probe.get("radial_gradient_coherence") or {}).get("support"))
            or 0
        ),
        "neutral_consistency_score": float(profile_probe.get("neutral_consistency_score", 0.0) or 0.0),
        "region_expansion_score": float(((profile_probe.get("neutral_region") or {}).get("region_expansion_score")) or 0.0),
        "shape_score": float(profile_probe.get("shape_score", 0.0) or 0.0),
        "exposure_validity_score": float(profile_probe.get("exposure_validity_score", 0.0) or 0.0),
        "radial_profile_score": float(((profile_probe.get("radial_luminance_profile") or {}).get("score")) or 0.0),
        "roi_area_fraction": float(((profile_probe.get("neutral_region") or {}).get("seed_area_fraction")) or 0.0),
        "edge_support": float(profile_probe.get("edge_support", candidate.get("edge_support", 0.0)) or 0.0),
        "fit_residual_ratio": float(profile_probe.get("fit_residual_ratio", candidate.get("fit_residual_ratio", 1.0)) or 1.0),
        "edge_aligned": bool(((profile_probe.get("neutral_region") or {}).get("edge_aligned"))),
        "circularity": float(((profile_probe.get("neutral_region") or {}).get("circularity")) or 0.0),
        "aspect_ratio": float(((profile_probe.get("neutral_region") or {}).get("aspect_ratio")) or 99.0),
        "radial_cv": float(((profile_probe.get("neutral_region") or {}).get("radial_cv")) or 1.0),
        "solidity": float(((profile_probe.get("neutral_region") or {}).get("solidity")) or 0.0),
        "component_count": int(((profile_probe.get("neutral_region") or {}).get("component_count")) or 0),
        "fragmented": bool(((profile_probe.get("neutral_region") or {}).get("fragmented"))),
        "centroid_distance_norm": float(((profile_probe.get("neutral_region") or {}).get("centroid_distance_norm")) or 1.0),
        "radius_ratio": float(radius_ratio),
    }
    neutral_shape_proxy_ok = _candidate_geometry_shape_proxy_ok(
        candidate,
        neutral_region=dict(profile_probe.get("neutral_region") or {}),
        neutral_consistency_score=metrics["neutral_consistency_score"],
    ) or (
        validation.get("radius_sane") is not False
        and metrics["region_expansion_score"] >= SPHERE_SHAPE_PROXY_MIN_REGION_EXPANSION
        and metrics["neutral_consistency_score"] >= SPHERE_SHAPE_PROXY_MIN_NEUTRAL_CONSISTENCY
        and metrics["component_count"] <= 1
        and not metrics["fragmented"]
        and metrics["solidity"] >= SPHERE_SHAPE_PROXY_MIN_SOLIDITY
        and metrics["aspect_ratio"] <= SPHERE_SHAPE_PROXY_MAX_ASPECT_RATIO
        and metrics["centroid_distance_norm"] <= SPHERE_SHAPE_PROXY_MAX_CENTROID_DISTANCE_NORM
    )
    refined_geometry_authoritative = _refined_geometry_authoritative(candidate, metrics=metrics)
    if validation.get("radius_sane") is False:
        reasons.append("radius_outside_expected_range")
    if not bool(profile_probe.get("plausible_samples")):
        reasons.append("insufficient_sample_support")
    if metrics["zone_spread_ire"] < SPHERE_HARD_GATE_MIN_ZONE_SPREAD_IRE:
        reasons.append("zone_spread_below_min")
    if metrics["chroma_range"] > SPHERE_HARD_GATE_MAX_CHROMA_RANGE:
        reasons.append("chroma_range_above_max")
    if metrics["valid_pixel_count"] < SPHERE_HARD_GATE_MIN_VALID_PIXEL_COUNT:
        reasons.append("valid_pixel_count_below_min")
    if metrics["valid_pixel_fraction"] < SPHERE_HARD_GATE_MIN_VALID_PIXEL_FRACTION:
        reasons.append("valid_pixel_fraction_below_min")
    if metrics["center_ire"] < SPHERE_HARD_GATE_CENTER_IRE_MIN or metrics["center_ire"] > SPHERE_HARD_GATE_CENTER_IRE_MAX:
        reasons.append("center_ire_out_of_range")
    if metrics["interior_stddev"] < SPHERE_HARD_GATE_MIN_INTERIOR_STDDEV:
        reasons.append("interior_too_flat")
    if metrics["interior_stddev"] > SPHERE_HARD_GATE_MAX_INTERIOR_STDDEV:
        reasons.append("interior_too_noisy")
    if not metrics["center_between_extremes"]:
        reasons.append("center_not_between_extremes")
    center_extremum_override = (
        refined_geometry_authoritative
        and metrics["edge_support"] >= 0.50
        and metrics["radial_profile_score"] >= 0.90
        and metrics["neutral_consistency_score"] >= 0.80
        and metrics["center_between_extremes"]
        and metrics["zone_spread_ire"] >= 1.20
    )
    if metrics["center_margin_ire"] < SPHERE_HARD_GATE_CENTER_EXTREMUM_MARGIN_IRE and not center_extremum_override:
        reasons.append("center_too_close_to_extremum")
    if metrics["measurement_confidence"] < SPHERE_HARD_GATE_MIN_MEASUREMENT_CONFIDENCE:
        reasons.append("measurement_confidence_below_min")
    if metrics["radial_gradient_coherence"] < SPHERE_HARD_GATE_MIN_RADIAL_COHERENCE:
        reasons.append("radial_coherence_below_min")
    if metrics["neutral_consistency_score"] < SPHERE_HARD_GATE_MIN_NEUTRAL_CONSISTENCY:
        reasons.append("neutral_consistency_below_min")
    if metrics["region_expansion_score"] < SPHERE_HARD_GATE_MIN_REGION_EXPANSION:
        reasons.append("neutrality_collapses_on_expansion")
    if metrics["shape_score"] < SPHERE_HARD_GATE_MIN_SHAPE_SCORE and not neutral_shape_proxy_ok and not refined_geometry_authoritative:
        reasons.append("non_circular_region")
    if metrics["circularity"] < SPHERE_SHAPE_MIN_CIRCULARITY and not neutral_shape_proxy_ok and not refined_geometry_authoritative:
        reasons.append("circularity_below_min")
    if metrics["aspect_ratio"] > SPHERE_SHAPE_MAX_ASPECT_RATIO and not neutral_shape_proxy_ok and not refined_geometry_authoritative:
        reasons.append("aspect_ratio_above_max")
    if metrics["radial_cv"] > SPHERE_SHAPE_MAX_RADIAL_CV:
        reasons.append("radial_variance_above_max")
    if metrics["solidity"] < SPHERE_NEUTRAL_REGION_MIN_SOLIDITY and not neutral_shape_proxy_ok and not refined_geometry_authoritative:
        reasons.append("shape_solidity_below_min")
    if metrics["fragmented"] or metrics["component_count"] > SPHERE_NEUTRAL_SEED_MAX_FRAGMENT_COUNT:
        reasons.append("fragmented_neutral_region")
    if metrics["edge_aligned"]:
        reasons.append("edge_aligned_region")
    if metrics["radial_profile_score"] < SPHERE_HARD_GATE_MIN_RADIAL_PROFILE_SCORE:
        reasons.append("flat_luminance_profile")
    if metrics["edge_support"] < SPHERE_HARD_GATE_MIN_EDGE_SUPPORT:
        reasons.append("edge_support_below_min")
    if metrics["fit_residual_ratio"] > SPHERE_HARD_GATE_MAX_FIT_RESIDUAL_RATIO:
        reasons.append("fit_residual_above_max")
    if (
        metrics["exposure_validity_score"] < SPHERE_HARD_GATE_MIN_EXPOSURE_VALIDITY
        or metrics["samples_equal"]
        or metrics["flat_high"]
        or metrics["flat_low"]
    ):
        reasons.append("exposure_distribution_invalid")
    if metrics["samples_equal"]:
        reasons.append("samples_too_equal")
    if metrics["flat_high"]:
        reasons.append("flat_high_ire_region")
    if metrics["flat_low"]:
        reasons.append("flat_low_ire_region")
    if metrics["roi_area_fraction"] < SPHERE_HARD_GATE_MIN_ROI_AREA_FRACTION:
        reasons.append("minimum_size_not_met")
    if metrics["radius_ratio"] < SPHERE_HARD_GATE_MIN_RADIUS_RATIO:
        reasons.append("minimum_size_not_met")
    reasons.extend([str(reason) for reason in list(profile_probe.get("critical_failures") or [])])
    return {
        "passed": not reasons,
        "reasons": list(dict.fromkeys(reasons)),
        "metrics": metrics,
    }


def _rescore_sphere_candidates_with_profile(
    region: np.ndarray,
    candidates: List[Dict[str, object]],
    *,
    region_state: Optional[Dict[str, object]] = None,
) -> List[Dict[str, object]]:
    rescored: List[Dict[str, object]] = []
    for candidate in list(candidates or [])[:16]:
        item = copy.deepcopy(dict(candidate))
        profile_probe = _sphere_candidate_profile_plausibility(region, item, region_state=region_state)
        hard_gate = _sphere_candidate_hard_gate(item, profile_probe)
        base_confidence = float(item.get("confidence", 0.0) or 0.0)
        plausibility_score = float(profile_probe.get("score", 0.0) or 0.0)
        combined_confidence = float(max(0.0, min(1.0, plausibility_score)))
        if not bool(profile_probe.get("valid")):
            combined_confidence *= 0.55
        if not bool(hard_gate.get("passed")):
            combined_confidence *= 0.35
        item["geometry_confidence"] = base_confidence
        item["sample_plausibility"] = str(profile_probe.get("label") or "IMPLAUSIBLE")
        item["sample_plausibility_score"] = plausibility_score
        item["sample_review_required"] = bool(profile_probe.get("review_required"))
        item["profile_probe"] = profile_probe
        item["hard_gate"] = hard_gate
        item["confidence"] = combined_confidence
        item["confidence_label"] = _sphere_detection_label(combined_confidence)
        rescored.append(item)
    return sorted(rescored, key=lambda item: float(item.get("confidence", 0.0) or 0.0), reverse=True)


def _validate_known_sphere_roi(
    region: np.ndarray,
    *,
    roi: SphereROI,
    detection_source: str,
    confidence_hint: Optional[float] = None,
    manual_assist_metadata: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    region_state = _build_sphere_region_state(region)
    payload: Dict[str, object] = {
        "source": str(detection_source or "unresolved"),
        "roi": {"cx": float(roi.cx), "cy": float(roi.cy), "r": float(roi.r)},
    }
    if manual_assist_metadata:
        payload["manual_assist"] = dict(manual_assist_metadata)
    payload = _refine_sphere_candidate_geometry(region, payload, region_state=region_state)
    profile_probe = _sphere_candidate_profile_plausibility(region, payload, region_state=region_state)
    hard_gate = _sphere_candidate_hard_gate(payload, profile_probe)
    plausibility_score = float(profile_probe.get("score", 0.0) or 0.0)
    base_confidence = float(confidence_hint if confidence_hint is not None else plausibility_score)
    confidence = float(max(0.0, min(1.0, (base_confidence * 0.55) + (plausibility_score * 0.45))))
    if not bool(hard_gate.get("passed")):
        confidence = 0.0
    return {
        **payload,
        "profile_probe": profile_probe,
        "hard_gate": hard_gate,
        "sample_plausibility": str(profile_probe.get("label") or "IMPLAUSIBLE"),
        "sample_review_required": bool(profile_probe.get("review_required")) or not bool(hard_gate.get("passed")),
        "confidence": float(confidence),
        "confidence_label": _sphere_detection_label(confidence) if bool(hard_gate.get("passed")) else "UNRESOLVED",
        "sphere_detection_success": bool(hard_gate.get("passed")),
        "sphere_detection_unresolved": not bool(hard_gate.get("passed")),
    }


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


def _circle_candidate_bbox(roi: SphereROI) -> Tuple[float, float, float, float]:
    return (
        float(roi.cx) - float(roi.r),
        float(roi.cy) - float(roi.r),
        float(roi.cx) + float(roi.r),
        float(roi.cy) + float(roi.r),
    )


def _bbox_intersection_area(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    x0 = max(float(a[0]), float(b[0]))
    y0 = max(float(a[1]), float(b[1]))
    x1 = min(float(a[2]), float(b[2]))
    y1 = min(float(a[3]), float(b[3]))
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return float((x1 - x0) * (y1 - y0))


def _rescore_sphere_candidates_with_gray_card_context(region: np.ndarray, candidates: List[Dict[str, object]]) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    gray_card = _detect_gray_card_in_region_hwc(region)
    card_roi = dict(gray_card.get("roi") or {})
    card_confidence = float(gray_card.get("confidence", 0.0) or 0.0)
    card_area_fraction = float(gray_card.get("roi_area_fraction", 0.0) or 0.0)
    card_aspect_ratio = float(gray_card.get("aspect_ratio", 0.0) or 0.0)
    if (
        not card_roi
        or card_confidence < 0.62
        or card_area_fraction <= 0.02
        or card_aspect_ratio < 0.78
    ):
        return candidates, {"available": False}
    card_x = float(card_roi.get("x", 0.0) or 0.0)
    card_y = float(card_roi.get("y", 0.0) or 0.0)
    card_w = float(card_roi.get("width", 0.0) or 0.0)
    card_h = float(card_roi.get("height", 0.0) or 0.0)
    card_scale = max(card_w, card_h, 1.0)
    card_center_x = card_x + (card_w * 0.5)
    card_center_y = card_y + (card_h * 0.5)
    card_bbox = (card_x, card_y, card_x + card_w, card_y + card_h)
    rescored: List[Dict[str, object]] = []
    diagnostics: List[Dict[str, object]] = []
    for candidate in list(candidates or []):
        item = copy.deepcopy(dict(candidate))
        roi = _coerce_sphere_roi(item.get("roi"))
        if roi is None:
            rescored.append(item)
            continue
        candidate_bbox = _circle_candidate_bbox(roi)
        candidate_area = max(float((candidate_bbox[2] - candidate_bbox[0]) * (candidate_bbox[3] - candidate_bbox[1])), 1.0)
        overlap_area = _bbox_intersection_area(candidate_bbox, card_bbox)
        overlap_fraction = float(overlap_area / candidate_area)
        center_inside_card = card_x <= float(roi.cx) <= (card_x + card_w) and card_y <= float(roi.cy) <= (card_y + card_h)
        dx_norm = abs(float(roi.cx) - card_center_x) / card_scale
        dy_norm = abs(float(roi.cy) - card_center_y) / card_scale
        axis_pair_score = max(
            _plateau_score(dx_norm, 0.55, 1.75, 0.65) * _plateau_score(dy_norm, 0.0, 0.42, 0.28),
            _plateau_score(dy_norm, 0.55, 1.75, 0.65) * _plateau_score(dx_norm, 0.0, 0.42, 0.28),
        )
        radius_relation_score = _plateau_score(float(roi.r) / card_scale, 0.32, 0.85, 0.20)
        separation_score = max(0.0, 1.0 - min(overlap_fraction / 0.12, 1.0))
        context_score = float(np.mean(np.asarray([axis_pair_score * 1.15, radius_relation_score, separation_score], dtype=np.float32)))
        base_confidence = float(item.get("confidence", 0.0) or 0.0)
        if center_inside_card or overlap_fraction >= 0.10:
            adjusted_confidence = base_confidence * 0.18
        else:
            adjusted_confidence = float(max(0.0, min(1.0, (base_confidence * 0.78) + (context_score * 0.22))))
        item["confidence"] = adjusted_confidence
        item["confidence_label"] = _sphere_detection_label(adjusted_confidence)
        item["gray_card_context"] = {
            "card_confidence": card_confidence,
            "overlap_fraction": overlap_fraction,
            "center_inside_card": bool(center_inside_card),
            "axis_pair_score": float(axis_pair_score),
            "radius_relation_score": float(radius_relation_score),
            "context_score": float(context_score),
        }
        rescored.append(item)
        diagnostics.append(
            {
                "source": str(item.get("source") or ""),
                "confidence": adjusted_confidence,
                "roi": dict(item.get("roi") or {}),
                "overlap_fraction": overlap_fraction,
                "center_inside_card": bool(center_inside_card),
                "context_score": float(context_score),
            }
        )
    return sorted(rescored, key=lambda item: float(item.get("confidence", 0.0) or 0.0), reverse=True), {
        "available": True,
        "confidence": float(card_confidence),
        "roi": card_roi,
        "rescored_candidates": diagnostics[:8],
    }


def _card_adjacent_search_bounds(card_roi: Dict[str, object], width: int, height: int) -> List[Tuple[int, int, int, int]]:
    card_x = float(card_roi.get("x", 0.0) or 0.0)
    card_y = float(card_roi.get("y", 0.0) or 0.0)
    card_w = float(card_roi.get("width", 0.0) or 0.0)
    card_h = float(card_roi.get("height", 0.0) or 0.0)
    y0 = max(0, int(math.floor(card_y - (card_h * 1.25))))
    y1 = min(height, int(math.ceil(card_y + (card_h * 2.0))))
    bounds: List[Tuple[int, int, int, int]] = []
    right_x0 = max(0, int(math.floor(card_x + (card_w * 0.45))))
    right_x1 = min(width, int(math.ceil(card_x + (card_w * 3.10))))
    left_x0 = max(0, int(math.floor(card_x - (card_w * 2.60))))
    left_x1 = min(width, int(math.ceil(card_x + (card_w * 0.55))))
    for x0, x1 in ((right_x0, right_x1), (left_x0, left_x1)):
        if x1 > x0 and y1 > y0:
            bounds.append((x0, y0, x1, y1))
    return bounds


def _detect_sphere_roi_in_region_hwc(region: np.ndarray) -> Dict[str, object]:
    height, width = region.shape[:2]
    min_dimension = min(height, width)
    if min_dimension < 64:
        return {
            "source": "failed",
            "confidence": 0.0,
            "confidence_label": "FAILED",
            "sphere_detection_success": False,
            "sphere_detection_unresolved": False,
            "validation": {"reason": "region_too_small"},
        }
    detection_region, detection_scale = _resize_region_for_sphere_detection(region)
    full_region_state = _build_sphere_region_state(region)
    detection_region_state = _build_sphere_region_state(detection_region)
    detection_height, detection_width = detection_region.shape[:2]
    candidate_summary = {
        "primary_candidates": [],
        "secondary_candidates": [],
        "blob_candidates": [],
        "opencv_hough_candidates": [],
        "opencv_hough_alt_candidates": [],
        "opencv_contour_candidates": [],
        "localized_candidates": [],
        "fallback_activated": False,
        "detection_resize_scale": float(detection_scale),
        "detection_input_size": {"width": int(detection_width), "height": int(detection_height)},
        "original_region_size": {"width": int(width), "height": int(height)},
    }
    gray_card_context = _detect_gray_card_in_region_hwc(region)
    card_roi = dict(gray_card_context.get("roi") or {})
    candidate_summary["gray_card_context"] = {
        "available": bool(card_roi),
        "confidence": float(gray_card_context.get("confidence", 0.0) or 0.0),
        "roi": card_roi,
    }
    raw_primary_candidates = _merge_sphere_candidate_clusters(
        [
            *_rescale_sphere_candidates(
                _detect_sphere_candidates_in_region_hwc(
                    detection_region,
                    detection_source="primary_detected",
                    region_state=detection_region_state,
                ),
                detection_scale,
            ),
            *_rescale_sphere_candidates(
                _detect_sphere_candidates_opencv_hough_hwc(
                    detection_region,
                    detection_source="opencv_hough_detected",
                    region_state=detection_region_state,
                ),
                detection_scale,
            ),
            *_rescale_sphere_candidates(
                _detect_sphere_candidates_opencv_hough_alt_hwc(
                    detection_region,
                    detection_source="opencv_hough_alt_detected",
                    region_state=detection_region_state,
                ),
                detection_scale,
            ),
            *_rescale_sphere_candidates(
                _detect_sphere_candidates_opencv_contours_hwc(
                    detection_region,
                    detection_source="opencv_contour_detected",
                    region_state=detection_region_state,
                ),
                detection_scale,
            ),
            *_rescale_sphere_candidates(
                _detect_neutral_blob_candidates_in_region_hwc(
                    detection_region,
                    detection_source="neutral_blob_detected",
                    region_state=detection_region_state,
                ),
                detection_scale,
            ),
        ]
    )
    candidate_summary["primary_candidates"] = raw_primary_candidates[:8]
    candidate_summary["opencv_hough_candidates"] = [
        item for item in raw_primary_candidates if "opencv_hough_circle" in item
    ][:6]
    candidate_summary["opencv_hough_alt_candidates"] = [
        item for item in raw_primary_candidates if "opencv_hough_alt_circle" in item
    ][:6]
    candidate_summary["opencv_contour_candidates"] = [
        item for item in raw_primary_candidates if "opencv_contour" in item
    ][:6]
    candidate_summary["blob_candidates"] = [
        item for item in raw_primary_candidates if str(item.get("source") or "").startswith("neutral_blob")
    ][:6]
    ranked_primary = _rank_sphere_candidates_for_refinement(raw_primary_candidates)
    refined_primary = [
        _refine_sphere_candidate_geometry(region, candidate, region_state=full_region_state)
        for candidate in ranked_primary[:SPHERE_CANDIDATE_REFINEMENT_TOP_N]
    ]
    primary_rescored = _rescore_sphere_candidates_with_profile(
        region,
        refined_primary,
        region_state=full_region_state,
    )
    candidate_summary["rescored_candidates"] = primary_rescored[:8]
    viable_primary = [dict(item) for item in primary_rescored if bool((item.get("hard_gate") or {}).get("passed"))]
    candidate_summary["viable_candidates"] = viable_primary[:8]
    if viable_primary:
        selected = dict(viable_primary[0])
        selected["candidate_diagnostics"] = candidate_summary
        selected["sphere_detection_success"] = True
        selected["sphere_detection_unresolved"] = False
        return selected
    best_primary_rejected = dict(primary_rescored[0]) if primary_rescored else {}
    if best_primary_rejected:
        candidate_summary["best_primary_rejected_candidate"] = best_primary_rejected
    candidate_summary["fallback_activated"] = True
    candidate_summary["fallback_trigger_reason"] = (
        "primary_candidates_rejected_by_hard_gate" if raw_primary_candidates else "no_primary_candidates"
    )
    search_bounds = (0, 0, detection_width, detection_height)
    secondary_candidates = _merge_sphere_candidate_clusters(
        [
            *_rescale_sphere_candidates(
                _detect_sphere_candidates_in_region_hwc(
                    detection_region,
                    search_bounds=search_bounds,
                    detection_source="secondary_detected",
                    sigma=1.6,
                    low_threshold=0.02,
                    high_threshold=0.10,
                    region_state=detection_region_state,
                ),
                detection_scale,
            ),
            *_rescale_sphere_candidates(
                _detect_sphere_candidates_opencv_hough_hwc(
                    detection_region,
                    search_bounds=search_bounds,
                    detection_source="opencv_hough_recovery",
                    region_state=detection_region_state,
                ),
                detection_scale,
            ),
            *_rescale_sphere_candidates(
                _detect_sphere_candidates_opencv_hough_alt_hwc(
                    detection_region,
                    search_bounds=search_bounds,
                    detection_source="opencv_hough_alt_recovery",
                    region_state=detection_region_state,
                ),
                detection_scale,
            ),
            *_rescale_sphere_candidates(
                _detect_sphere_candidates_opencv_contours_hwc(
                    detection_region,
                    detection_source="opencv_contour_recovery",
                    region_state=detection_region_state,
                ),
                detection_scale,
            ),
            *_rescale_sphere_candidates(
                _detect_neutral_blob_candidates_in_region_hwc(
                    detection_region,
                    detection_source="neutral_blob_recovery",
                    region_state=detection_region_state,
                ),
                detection_scale,
            ),
        ]
    )
    candidate_summary["secondary_candidates"] = secondary_candidates[:8]
    fallback_pool = list(secondary_candidates[:SPHERE_CANDIDATE_REFINEMENT_TOP_N])
    localized_candidates: List[Dict[str, object]] = []
    if fallback_pool:
        seed = max(fallback_pool, key=lambda item: float(item.get("confidence", 0.0) or 0.0))
        localized_bounds = _localized_search_bounds_for_candidate(seed, width, height)
        if localized_bounds is not None:
            scaled_bounds = (
                max(0, int(math.floor(float(localized_bounds[0]) * detection_scale))),
                max(0, int(math.floor(float(localized_bounds[1]) * detection_scale))),
                min(detection_width, int(math.ceil(float(localized_bounds[2]) * detection_scale))),
                min(detection_height, int(math.ceil(float(localized_bounds[3]) * detection_scale))),
            )
            localized_candidates = _rescale_sphere_candidates(
                _detect_sphere_candidates_in_region_hwc(
                    detection_region,
                    search_bounds=scaled_bounds,
                    detection_source="localized_recovery",
                    sigma=1.2,
                    low_threshold=0.015,
                    high_threshold=0.08,
                    region_state=detection_region_state,
                ),
                detection_scale,
            )
            candidate_summary["localized_candidates"] = localized_candidates[:6]
    forced_pool = _merge_sphere_candidate_clusters([*fallback_pool, *(localized_candidates[:8])])
    if forced_pool:
        ranked_pool = _rank_sphere_candidates_for_refinement(forced_pool)
        refined_pool = [
            _refine_sphere_candidate_geometry(region, candidate, region_state=full_region_state)
            for candidate in ranked_pool[:SPHERE_CANDIDATE_REFINEMENT_TOP_N]
        ]
        rescored_pool = _rescore_sphere_candidates_with_profile(
            region,
            refined_pool,
            region_state=full_region_state,
        )
        rescored_pool, gray_card_context = _rescore_sphere_candidates_with_gray_card_context(region, rescored_pool)
        candidate_summary["rescored_candidates"] = rescored_pool[:8]
        candidate_summary["gray_card_context"] = gray_card_context
        viable_pool = [dict(item) for item in rescored_pool if bool((item.get("hard_gate") or {}).get("passed"))]
        candidate_summary["viable_candidates"] = viable_pool[:8]
        if viable_pool:
            selected = dict(viable_pool[0])
            selected["candidate_diagnostics"] = candidate_summary
            selected["sphere_detection_success"] = True
            selected["sphere_detection_unresolved"] = False
            return selected
        best_rejected = dict(rescored_pool[0])
        candidate_summary["best_rejected_candidate"] = best_rejected
        return {
            "source": "unresolved",
            "confidence": max(float(best_rejected.get("confidence", 0.0) or 0.0), 0.01),
            "confidence_label": "UNRESOLVED",
            "sample_plausibility": str(best_rejected.get("sample_plausibility") or "IMPLAUSIBLE"),
            "profile_probe": dict(best_rejected.get("profile_probe") or {}),
            "hard_gate": dict(best_rejected.get("hard_gate") or {}),
            "best_rejected_candidate": best_rejected,
            "candidate_diagnostics": candidate_summary,
            "sphere_detection_success": False,
            "sphere_detection_unresolved": True,
            "validation": {
                "reason": "candidates_rejected_by_hard_gate",
                "best_rejected_source": str(best_rejected.get("source") or ""),
                "hard_gate_reasons": list((best_rejected.get("hard_gate") or {}).get("reasons") or []),
                "fallback_trigger_reason": str(candidate_summary.get("fallback_trigger_reason") or ""),
            },
        }
    if raw_primary_candidates:
        candidate_summary["best_rejected_candidate"] = best_primary_rejected
        return {
            "source": "unresolved",
            "confidence": max(float(best_primary_rejected.get("confidence", 0.0) or 0.0), 0.01),
            "confidence_label": "UNRESOLVED",
            "sample_plausibility": str(best_primary_rejected.get("sample_plausibility") or "IMPLAUSIBLE"),
            "profile_probe": dict(best_primary_rejected.get("profile_probe") or {}),
            "hard_gate": dict(best_primary_rejected.get("hard_gate") or {}),
            "best_rejected_candidate": best_primary_rejected,
            "candidate_diagnostics": candidate_summary,
            "sphere_detection_success": False,
            "sphere_detection_unresolved": True,
            "validation": {
                "reason": "primary_candidates_rejected_by_hard_gate",
                "best_rejected_source": str(best_primary_rejected.get("source") or ""),
                "hard_gate_reasons": list((best_primary_rejected.get("hard_gate") or {}).get("reasons") or []),
                "fallback_trigger_reason": str(candidate_summary.get("fallback_trigger_reason") or ""),
            },
        }
    return {
        "source": "failed",
        "confidence": 0.0,
        "confidence_label": "FAILED",
        "candidate_diagnostics": candidate_summary,
        "sphere_detection_success": False,
        "sphere_detection_unresolved": False,
        "validation": {
            "reason": "no_plausible_circle_candidate",
            "primary_candidate_count": len(raw_primary_candidates),
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


def _detect_gray_card_in_region_hwc(region: np.ndarray) -> Dict[str, object]:
    chw_region = np.transpose(region, (2, 0, 1))
    detected = dict(detect_gray_card_roi(chw_region))
    roi = detected.get("roi")
    if not isinstance(roi, GrayCardROI):
        return {
            "roi": None,
            "confidence": 0.0,
            "confidence_label": "FAILED",
            "source": "gray_card_failed",
            "review_recommended": True,
        }
    pixels, info = extract_rect_pixels(chw_region, roi)
    stats = _measure_pixel_cloud_statistics(
        pixels,
        sampling_confidence=float(detected.get("confidence", 0.0) or 0.0),
        sampling_method="gray_card_detected_roi",
        mask_fraction=float(info.get("area_fraction", 0.0) or 0.0),
        interior_fraction=float(info.get("area_fraction", 0.0) or 0.0),
        interior_radius_ratio=1.0,
    )
    measured_rgb = np.asarray(stats.get("measured_rgb_chromaticity") or [1.0 / 3.0] * 3, dtype=np.float32)
    chroma_range = float(np.max(measured_rgb) - np.min(measured_rgb)) if measured_rgb.size else 1.0
    brightness = float((2.0 ** float(stats.get("measured_log2_luminance", 0.0) or 0.0)))
    neutrality_score = _plateau_score(chroma_range, 0.0, 0.06, 0.05)
    brightness_score = _plateau_score(brightness, 0.10, 0.55, 0.18)
    detection_confidence = float(detected.get("confidence", 0.0) or 0.0)
    measurement_confidence = float(stats.get("confidence", 0.0) or 0.0)
    combined_confidence = float(
        max(
            0.0,
            min(
                1.0,
                (detection_confidence * 0.45) + (measurement_confidence * 0.35) + (neutrality_score * 0.10) + (brightness_score * 0.10),
            ),
        )
    )
    roi_dict = {
        "x": float(roi.x),
        "y": float(roi.y),
        "width": float(roi.width),
        "height": float(roi.height),
    }
    region_height = max(int(region.shape[0]), 1)
    region_width = max(int(region.shape[1]), 1)
    roi_area_fraction = float((float(roi.width) * float(roi.height)) / float(region_width * region_height))
    aspect_ratio = float(min(float(roi.width), float(roi.height)) / max(float(roi.width), float(roi.height), 1.0))
    border_margin_fraction = float(
        min(
            float(roi.x),
            float(roi.y),
            max(float(region_width) - (float(roi.x) + float(roi.width)), 0.0),
            max(float(region_height) - (float(roi.y) + float(roi.height)), 0.0),
        )
        / float(max(min(region_width, region_height), 1))
    )
    area_score = _plateau_score(roi_area_fraction, 0.025, 0.22, 0.08)
    aspect_score = _plateau_score(aspect_ratio, 0.78, 1.0, 0.18)
    margin_score = _plateau_score(border_margin_fraction, 0.01, 0.25, 0.05)
    final_confidence = float(
        0.0
        if (
            detection_confidence < 0.25
            or roi_area_fraction >= 0.35
            or roi_area_fraction <= 0.012
            or border_margin_fraction <= 0.005
            or aspect_ratio < 0.72
        )
        else max(
            0.0,
            min(
                1.0,
                (combined_confidence * 0.70) + (area_score * 0.15) + (aspect_score * 0.10) + (margin_score * 0.05),
            ),
        )
    )
    confidence_label = (
        "HIGH"
        if final_confidence >= 0.75
        else "MEDIUM"
        if final_confidence >= 0.5
        else "LOW"
        if final_confidence >= 0.3
        else "FAILED"
    )
    return {
        "roi": roi_dict,
        "stats": stats,
        "confidence": final_confidence,
        "confidence_label": confidence_label,
        "source": "gray_card_detected",
        "review_recommended": final_confidence < 0.5,
        "neutrality_score": float(neutrality_score),
        "brightness_score": float(brightness_score),
        "detection_confidence": detection_confidence,
        "measurement_confidence": measurement_confidence,
        "roi_area_fraction": roi_area_fraction,
        "aspect_ratio": aspect_ratio,
        "border_margin_fraction": border_margin_fraction,
        "measured_rgb_chromaticity": [float(value) for value in measured_rgb.tolist()],
    }


def _should_attempt_gray_card_fallback(
    *,
    sphere_roi: Optional[SphereROI],
    detection_source: str,
    detection_confidence: float,
    detection_label: str,
) -> bool:
    return False


def _gray_target_note(
    *,
    target_class: str,
    detection_source: str,
    detection_label: str,
    detection_failed: bool = False,
    fallback_used: bool = False,
    sphere_detection_success: Optional[bool] = None,
    sphere_detection_unresolved: bool = False,
) -> str:
    normalized_class = str(target_class or "").strip().lower()
    if sphere_detection_success is False and sphere_detection_unresolved:
        return "Gray target needs review"
    if detection_failed or normalized_class == "unresolved" or str(detection_source or "") == "failed" or str(detection_label or "") == "FAILED":
        return "Gray target needs review"
    if normalized_class == "gray_card":
        return "Gray card fallback used" if fallback_used else "Gray card verified"
    if fallback_used or str(detection_label or "") == "LOW":
        return "Alternate detection path used"
    return "Sphere check verified"


def _sphere_detection_status(success: bool, unresolved: bool) -> str:
    if success:
        return "SUCCESS"
    if unresolved:
        return "UNRESOLVED"
    return "FAILED"


def _sphere_detection_rejection_reason_label(reason: str) -> str:
    mapping = {
        "radius_outside_expected_range": "radius outside expected range",
        "insufficient_sample_support": "insufficient valid pixel support",
        "zone_spread_below_min": "insufficient exposure variation",
        "chroma_range_above_max": "material is not neutral enough",
        "valid_pixel_count_below_min": "insufficient valid pixel count",
        "valid_pixel_fraction_below_min": "insufficient valid pixel fraction",
        "center_ire_out_of_range": "center IRE is outside the valid range",
        "interior_too_flat": "interior is too flat",
        "interior_too_noisy": "interior is too noisy",
        "center_not_between_extremes": "center sample is not between the bright and dark sides",
        "center_too_close_to_extremum": "center sample behaves like an extreme",
        "measurement_confidence_below_min": "measurement confidence is too low",
        "radial_coherence_below_min": "insufficient radial coherence",
        "neutral_consistency_below_min": "neutrality is not spatially consistent",
        "neutrality_collapses_on_expansion": "neutrality collapses during region expansion",
        "non_circular_region": "non-circular region",
        "edge_aligned_region": "region is edge-aligned",
        "flat_luminance_profile": "flat luminance profile",
        "exposure_distribution_invalid": "implausible exposure distribution",
        "samples_too_equal": "sample zones are too equal",
        "flat_high_ire_region": "clipped or high-IRE flat region",
        "flat_low_ire_region": "near-black flat region",
        "minimum_size_not_met": "minimum size constraint not met",
    }
    return mapping.get(str(reason or "").strip(), str(reason or "").replace("_", " "))


def _sphere_detection_rejection_reasons_from_details(details: Optional[Dict[str, object]]) -> List[str]:
    hard_gate = dict((details or {}).get("hard_gate") or {})
    reasons = list(hard_gate.get("reasons") or [])
    validation = dict((details or {}).get("validation") or {})
    if not reasons:
        reasons = list(validation.get("hard_gate_reasons") or [])
    return [str(reason) for reason in reasons if str(reason or "").strip()]


def _manual_assist_preview_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _manual_assist_sanitized_entry(entry: Optional[Dict[str, object]]) -> Dict[str, object]:
    payload = dict(entry or {})
    if not payload:
        return {}
    sanitized: Dict[str, object] = {
        "clip_id": str(payload.get("clip_id") or "").strip(),
        "source_path": str(payload.get("source_path") or "").strip(),
        "source_image": str(payload.get("source_image") or payload.get("preview_image") or "").strip(),
        "source_image_hash": str(payload.get("source_image_hash") or payload.get("preview_image_hash") or "").strip(),
        "measurement_source_image": str(payload.get("measurement_source_image") or "").strip(),
        "measurement_source_hash": str(payload.get("measurement_source_hash") or "").strip(),
        "center_preview_px": payload.get("center_preview_px"),
        "radius_preview_px": payload.get("radius_preview_px"),
        "center_normalized": payload.get("center_normalized"),
        "radius_normalized": payload.get("radius_normalized"),
        "estimated_radius_preview_px": payload.get("estimated_radius_preview_px"),
        "estimated_radius_normalized": payload.get("estimated_radius_normalized"),
        "preview_width": payload.get("preview_width"),
        "preview_height": payload.get("preview_height"),
        "render_width": payload.get("render_width"),
        "render_height": payload.get("render_height"),
        "operator_note": str(payload.get("operator_note") or "").strip(),
    }
    filtered: Dict[str, object] = {}
    for key, value in sanitized.items():
        if value is None:
            continue
        if isinstance(value, str) and not value:
            continue
        if isinstance(value, (list, tuple, dict)) and len(value) == 0:
            continue
        filtered[key] = value
    return filtered


def _manual_assist_roi_for_image(
    entry: Optional[Dict[str, object]],
    *,
    image_width: int,
    image_height: int,
) -> Optional[Dict[str, float]]:
    payload = dict(entry or {})
    if not payload or image_width <= 0 or image_height <= 0:
        return None

    def _pair(value: object) -> Optional[Tuple[float, float]]:
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            try:
                return float(value[0]), float(value[1])
            except (TypeError, ValueError):
                return None
        return None

    preview_width = int(payload.get("preview_width", 0) or 0)
    preview_height = int(payload.get("preview_height", 0) or 0)
    center_normalized = _pair(payload.get("center_normalized"))
    center_preview_px = _pair(payload.get("center_preview_px"))
    if center_normalized is None and center_preview_px is not None and preview_width > 0 and preview_height > 0:
        center_normalized = (
            float(center_preview_px[0]) / float(max(preview_width, 1)),
            float(center_preview_px[1]) / float(max(preview_height, 1)),
        )
    if center_normalized is None:
        return None

    radius_normalized: Optional[float]
    try:
        radius_normalized = float(payload.get("radius_normalized")) if payload.get("radius_normalized") is not None else None
    except (TypeError, ValueError):
        radius_normalized = None
    if radius_normalized is None:
        try:
            radius_preview_px = float(payload.get("radius_preview_px")) if payload.get("radius_preview_px") is not None else None
        except (TypeError, ValueError):
            radius_preview_px = None
        if radius_preview_px is not None and preview_width > 0 and preview_height > 0:
            radius_normalized = radius_preview_px / float(max(min(preview_width, preview_height), 1))
    if radius_normalized is None:
        try:
            radius_normalized = float(payload.get("estimated_radius_normalized")) if payload.get("estimated_radius_normalized") is not None else None
        except (TypeError, ValueError):
            radius_normalized = None
    if radius_normalized is None:
        try:
            estimated_radius_preview_px = (
                float(payload.get("estimated_radius_preview_px"))
                if payload.get("estimated_radius_preview_px") is not None
                else None
            )
        except (TypeError, ValueError):
            estimated_radius_preview_px = None
        if estimated_radius_preview_px is not None and preview_width > 0 and preview_height > 0:
            radius_normalized = estimated_radius_preview_px / float(max(min(preview_width, preview_height), 1))
    if radius_normalized is None:
        return None

    cx = float(center_normalized[0]) * float(image_width)
    cy = float(center_normalized[1]) * float(image_height)
    radius = float(radius_normalized) * float(min(image_width, image_height))
    if not (0.0 <= cx <= float(image_width) and 0.0 <= cy <= float(image_height) and radius > 1.0):
        return None
    return {
        "cx": float(cx),
        "cy": float(cy),
        "r": float(radius),
    }


def _refine_manual_assist_sphere_roi(
    image: np.ndarray,
    manual_roi_payload: Optional[Dict[str, float]],
) -> Dict[str, object]:
    manual_roi = _coerce_sphere_roi(manual_roi_payload)
    if manual_roi is None:
        return {"refined": False}
    height, width = image.shape[:2]
    pad = max(float(manual_roi.r) * 2.3, 180.0)
    x0 = max(0, int(math.floor(float(manual_roi.cx) - pad)))
    y0 = max(0, int(math.floor(float(manual_roi.cy) - pad)))
    x1 = min(width, int(math.ceil(float(manual_roi.cx) + pad)))
    y1 = min(height, int(math.ceil(float(manual_roi.cy) + pad)))
    if x1 <= x0 or y1 <= y0:
        return {"refined": False, "reason": "empty_refinement_crop"}
    crop = image[y0:y1, x0:x1]
    detected = dict(_detect_sphere_roi_in_region_hwc(crop))
    candidate_pool: List[Dict[str, object]] = []
    for item in list(((detected.get("candidate_diagnostics") or {}).get("rescored_candidates") or [])):
        candidate = copy.deepcopy(dict(item))
        roi = _coerce_sphere_roi(candidate.get("roi"))
        if roi is None:
            continue
        global_roi = {
            "cx": float(roi.cx) + float(x0),
            "cy": float(roi.cy) + float(y0),
            "r": float(roi.r),
        }
        center_distance = math.hypot(float(global_roi["cx"]) - float(manual_roi.cx), float(global_roi["cy"]) - float(manual_roi.cy))
        radius_ratio = float(global_roi["r"]) / max(float(manual_roi.r), 1e-6)
        plausibility = float(candidate.get("sample_plausibility_score", 0.0) or 0.0)
        confidence = float(candidate.get("confidence", 0.0) or 0.0)
        if center_distance > max(float(manual_roi.r) * 1.10, 90.0):
            continue
        if not (0.45 <= radius_ratio <= 1.70):
            continue
        candidate["roi"] = global_roi
        candidate["_manual_center_distance"] = float(center_distance)
        candidate["_manual_radius_ratio"] = float(radius_ratio)
        candidate["_manual_rank"] = float((center_distance / max(float(manual_roi.r), 1.0)) - (confidence * 0.35) - (plausibility * 0.30))
        candidate_pool.append(candidate)
    if candidate_pool:
        detected = min(candidate_pool, key=lambda item: float(item.get("_manual_rank", 0.0) or 0.0))
    detected_roi = _coerce_sphere_roi(detected.get("roi"))
    accept = False
    candidate = copy.deepcopy(detected)
    if detected_roi is not None:
        global_roi = {
            "cx": float(detected_roi.cx) + float(x0),
            "cy": float(detected_roi.cy) + float(y0),
            "r": float(detected_roi.r),
        }
        center_distance = math.hypot(float(global_roi["cx"]) - float(manual_roi.cx), float(global_roi["cy"]) - float(manual_roi.cy))
        radius_ratio = float(global_roi["r"]) / max(float(manual_roi.r), 1e-6)
        candidate["roi"] = global_roi
        candidate["local_refinement"] = {
            "base_roi": {"cx": float(manual_roi.cx), "cy": float(manual_roi.cy), "r": float(manual_roi.r)},
            "local_bounds": {"x": x0, "y": y0, "width": x1 - x0, "height": y1 - y0},
            "center_distance": float(center_distance),
            "radius_ratio": float(radius_ratio),
        }
        accept = (
            float(detected.get("confidence", 0.0) or 0.0) >= SPHERE_DETECTION_LOW_CONFIDENCE
            and center_distance <= max(float(manual_roi.r) * 0.85, 55.0)
            and 0.55 <= radius_ratio <= 1.45
        )
    if not accept:
        search_results: List[Dict[str, object]] = []
        image_height, image_width = image.shape[:2]
        for dx_factor in MANUAL_ASSIST_LOCAL_SHIFT_FACTORS:
            for dy_factor in MANUAL_ASSIST_LOCAL_SHIFT_FACTORS:
                for radius_factor in MANUAL_ASSIST_LOCAL_RADIUS_FACTORS:
                    probe_roi = SphereROI(
                        cx=float(manual_roi.cx) + (float(manual_roi.r) * float(dx_factor)),
                        cy=float(manual_roi.cy) + (float(manual_roi.r) * float(dy_factor)),
                        r=float(manual_roi.r) * float(radius_factor),
                    )
                    if (
                        probe_roi.cx - probe_roi.r < 0.0
                        or probe_roi.cy - probe_roi.r < 0.0
                        or probe_roi.cx + probe_roi.r > float(image_width)
                        or probe_roi.cy + probe_roi.r > float(image_height)
                    ):
                        continue
                    validated = _validate_known_sphere_roi(
                        image,
                        roi=probe_roi,
                        detection_source="manual_operator_assist_local_search",
                        confidence_hint=0.92,
                    )
                    hard_gate = dict(validated.get("hard_gate") or {})
                    metrics = dict(hard_gate.get("metrics") or {})
                    search_results.append(
                        {
                            "validated": validated,
                            "rank": float(validated.get("confidence", 0.0) or 0.0)
                            + float((validated.get("profile_probe") or {}).get("score", 0.0) or 0.0)
                            + float(metrics.get("center_margin_ire", 0.0) or 0.0) * 0.01,
                        }
                    )
        passed_candidates = [
            item for item in search_results if bool((item.get("validated") or {}).get("sphere_detection_success"))
        ]
        if passed_candidates:
            best = max(passed_candidates, key=lambda item: float(item.get("rank", 0.0) or 0.0))
            validated = dict(best.get("validated") or {})
            accepted_roi = _coerce_sphere_roi(validated.get("roi"))
            if accepted_roi is not None:
                return {
                    "refined": True,
                    "roi": {"cx": float(accepted_roi.cx), "cy": float(accepted_roi.cy), "r": float(accepted_roi.r)},
                    "source": "manual_operator_assist_local_search",
                    "confidence": float(validated.get("confidence", 0.0) or 0.0),
                    "confidence_label": str(validated.get("confidence_label") or "HIGH"),
                    "candidate": validated,
                    "reason": "accepted_manual_local_search",
                }
        if search_results:
            best_rejected = max(search_results, key=lambda item: float(item.get("rank", 0.0) or 0.0))
            candidate["manual_local_search"] = dict(best_rejected.get("validated") or {})
    return {
        "refined": bool(accept),
        "roi": global_roi if accept and detected_roi is not None else {"cx": float(manual_roi.cx), "cy": float(manual_roi.cy), "r": float(manual_roi.r)},
        "source": str(detected.get("source") or "manual_operator_assist_refinement"),
        "confidence": float(detected.get("confidence", 0.0) or 0.0),
        "confidence_label": str(detected.get("confidence_label") or "LOW"),
        "candidate": candidate,
        "reason": "accepted_local_refinement" if accept else ("no_local_circle" if detected_roi is None else "manual_seed_retained"),
    }


def _measure_rendered_preview_roi_ipp2(
    preview_path: str,
    calibration_roi: Optional[Dict[str, float]],
    *,
    sphere_roi_override: Optional[Dict[str, float]] = None,
    sphere_roi_source_override: Optional[str] = None,
    manual_assist_entry: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    measurement_started_at = time.perf_counter()
    phase_started_at = measurement_started_at
    image, image_metadata = _load_preview_image_as_normalized_rgb(preview_path)
    image_load_seconds = time.perf_counter() - phase_started_at
    phase_started_at = time.perf_counter()
    region = _extract_normalized_roi_region_hwc(image, calibration_roi)
    region_origin_x, region_origin_y = _normalized_roi_origin(image, calibration_roi)
    measurement_region_scope = "calibration_roi" if calibration_roi is not None else "full_frame"
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
            "render_width": int(image_metadata.get("width", image.shape[1])),
            "render_height": int(image_metadata.get("height", image.shape[0])),
            "rendered_image_dtype": str(image_metadata.get("source_dtype") or ""),
            "rendered_image_mode": str(image_metadata.get("image_mode") or ""),
            "rendered_image_bit_depth": image_metadata.get("bit_depth"),
            "rendered_image_normalization_denominator": image_metadata.get("normalization_denominator"),
            "rendered_preview_format": str(image_metadata.get("preview_format") or ""),
            "measurement_region_scope": measurement_region_scope,
            "measurement_region_origin": {"x": int(region_origin_x), "y": int(region_origin_y)},
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
    manual_assist_metadata = _manual_assist_sanitized_entry(manual_assist_entry)
    sphere_roi = _coerce_sphere_roi(sphere_roi_override)
    sphere_roi_source = ""
    detection_confidence = 1.0 if sphere_roi is not None else 0.0
    detection_label = "HIGH" if sphere_roi is not None else "FAILED"
    detection_details: Dict[str, object] = {}
    gray_card_details: Dict[str, object] = {}
    gray_card_detection_seconds = 0.0
    sphere_detection_success = False
    sphere_detection_unresolved = False
    if sphere_roi is not None:
        sphere_roi_source = str(sphere_roi_source_override or "reused_from_original")
        detection_details = _validate_known_sphere_roi(
            region,
            roi=sphere_roi,
            detection_source=sphere_roi_source,
            confidence_hint=1.0,
        )
        detection_confidence = float(detection_details.get("confidence", 0.0) or 0.0)
        detection_label = str(detection_details.get("confidence_label") or "FAILED")
        sphere_detection_success = bool(detection_details.get("sphere_detection_success"))
        sphere_detection_unresolved = bool(detection_details.get("sphere_detection_unresolved"))
        if not sphere_detection_success:
            sphere_roi = None
    detection_started_at = time.perf_counter()
    if sphere_roi is None and not sphere_detection_success:
        detected = _detect_sphere_roi_in_region_hwc(region)
        detection_details = dict(detected)
        detection_confidence = float(detected.get("confidence", 0.0) or 0.0)
        detection_label = str(detected.get("confidence_label") or "FAILED")
        sphere_roi_source = str(detected.get("source") or "failed")
        sphere_roi = _coerce_sphere_roi(detected.get("roi"))
        sphere_detection_success = bool(detected.get("sphere_detection_success"))
        sphere_detection_unresolved = bool(detected.get("sphere_detection_unresolved"))
        region_candidate_valid = bool(sphere_detection_success) and sphere_roi is not None
        region_plausibility_score = float((detected.get("profile_probe") or {}).get("score", 0.0) or 0.0)
        if calibration_roi is not None:
            full_detected = _detect_sphere_roi_in_region_hwc(image)
            full_confidence = float(full_detected.get("confidence", 0.0) or 0.0)
            full_candidate_valid = bool(full_detected.get("sphere_detection_success")) and _coerce_sphere_roi(full_detected.get("roi")) is not None
            full_plausibility_score = float((full_detected.get("profile_probe") or {}).get("score", 0.0) or 0.0)
            if full_candidate_valid and (
                not region_candidate_valid
                or _sphere_detection_uses_fallback(sphere_roi_source)
                or detection_confidence < 0.90
                or full_confidence >= detection_confidence
                or full_plausibility_score >= region_plausibility_score + 0.03
            ):
                detection_details = dict(full_detected)
                detection_confidence = full_confidence
                detection_label = str(full_detected.get("confidence_label") or "FAILED")
                sphere_roi_source = str(full_detected.get("source") or "failed")
                sphere_roi = _coerce_sphere_roi(full_detected.get("roi"))
                sphere_detection_success = bool(full_detected.get("sphere_detection_success"))
                sphere_detection_unresolved = bool(full_detected.get("sphere_detection_unresolved"))
                region = image
                region_origin_x = 0
                region_origin_y = 0
                measurement_region_scope = "full_frame"
        if not sphere_detection_success:
            sphere_roi = None
    if sphere_roi is None and manual_assist_entry:
        manual_roi_payload = _manual_assist_roi_for_image(
            manual_assist_entry,
            image_width=int(image_metadata.get("width", image.shape[1])),
            image_height=int(image_metadata.get("height", image.shape[0])),
        )
        refinement = _refine_manual_assist_sphere_roi(image, manual_roi_payload)
        manual_roi = _coerce_sphere_roi(refinement.get("roi") or manual_roi_payload)
        if manual_roi is not None:
            region = image
            region_origin_x = 0
            region_origin_y = 0
            measurement_region_scope = "manual_assist_full_frame"
            sphere_roi_source = (
                "manual_operator_assist_refined"
                if bool(refinement.get("refined"))
                else "manual_operator_assist"
            )
            manual_assist_metadata["refinement_used"] = bool(refinement.get("refined"))
            manual_assist_metadata["refinement_reason"] = str(refinement.get("reason") or "")
            detection_details = _validate_known_sphere_roi(
                region,
                roi=manual_roi,
                detection_source=sphere_roi_source,
                confidence_hint=float(refinement.get("confidence", 0.92) or 0.92),
                manual_assist_metadata=manual_assist_metadata,
            )
            detection_details["manual_refinement"] = dict(refinement.get("candidate") or {})
            detection_details["manual_refinement_reason"] = str(refinement.get("reason") or "")
            detection_confidence = float(detection_details.get("confidence", 0.0) or 0.0)
            detection_label = str(detection_details.get("confidence_label") or "UNRESOLVED")
            sphere_detection_success = bool(detection_details.get("sphere_detection_success"))
            sphere_detection_unresolved = bool(detection_details.get("sphere_detection_unresolved"))
            if sphere_detection_success:
                sphere_roi = manual_roi
    sphere_detection_seconds = time.perf_counter() - detection_started_at
    gray_target_class = "sphere" if sphere_detection_success and sphere_roi is not None else "unresolved"
    gray_target_detection_method = str(sphere_roi_source or "failed")
    gray_target_confidence = float(detection_confidence)
    manual_assist_used = str(sphere_roi_source or "").startswith("manual_") and sphere_detection_success
    gray_target_fallback_used = _sphere_detection_uses_fallback(
        sphere_roi_source,
        manual_assist_used=manual_assist_used,
        reused_from_original=bool(sphere_roi_override),
    ) if sphere_detection_success else False
    sample_plausibility = str((detection_details.get("sample_plausibility") or "IMPLAUSIBLE") if detection_details else ("HIGH" if sphere_detection_success else "IMPLAUSIBLE"))
    gray_target_review_recommended = (not sphere_detection_success) or gray_target_confidence < SPHERE_DETECTION_MEDIUM_CONFIDENCE
    if _should_attempt_gray_card_fallback(
        sphere_roi=sphere_roi,
        detection_source=sphere_roi_source,
        detection_confidence=detection_confidence,
        detection_label=detection_label,
    ):
        gray_card_started_at = time.perf_counter()
        gray_card_details = _detect_gray_card_in_region_hwc(region)
        gray_card_detection_seconds = time.perf_counter() - gray_card_started_at
        gray_card_confidence = float(gray_card_details.get("confidence", 0.0) or 0.0)
        if gray_card_confidence >= max(0.82, float(detection_confidence or 0.0) + 0.15):
            gray_target_class = "gray_card"
            gray_target_detection_method = str(gray_card_details.get("source") or "gray_card_detected")
            gray_target_confidence = gray_card_confidence
            gray_target_fallback_used = True
            gray_target_review_recommended = bool(gray_card_details.get("review_recommended"))
            sphere_roi = None
            sphere_roi_source = "gray_card_fallback"
            detection_label = str(gray_card_details.get("confidence_label") or "MEDIUM")
            detection_confidence = gray_card_confidence
    if sphere_roi is None:
        if gray_target_class == "gray_card" and dict(gray_card_details.get("roi") or {}):
            gray_card_roi = dict(gray_card_details.get("roi") or {})
            gray_card_stats = dict(gray_card_details.get("stats") or {})
            measured_log2 = float(gray_card_stats.get("measured_log2_luminance", 0.0) or 0.0)
            measured_ire = float(gray_card_stats.get("measured_ire", _ire_from_log2_luminance(measured_log2)) or _ire_from_log2_luminance(measured_log2))
            gray_card_summary = (
                f"Gray card sample {measured_ire:.1f} IRE"
                if measured_ire > 0.0
                else "Gray card sample"
            )
            return {
                "measurement_valid": True,
                "gray_target_measurement_valid": True,
                "measured_log2_luminance": measured_log2,
                "measured_log2_luminance_monitoring": measured_log2,
                "measured_rgb_mean": [float(value) for value in list(gray_card_stats.get("measured_rgb_mean") or [0.0, 0.0, 0.0])[:3]],
                "measured_rgb_chromaticity": [float(value) for value in list(gray_card_stats.get("measured_rgb_chromaticity") or [1.0 / 3.0] * 3)[:3]],
                "measured_rgb_chromaticity_monitoring": [float(value) for value in list(gray_card_stats.get("measured_rgb_chromaticity") or [1.0 / 3.0] * 3)[:3]],
                "valid_pixel_count": int(gray_card_stats.get("valid_pixel_count", 0) or 0),
                "roi_variance": float(gray_card_stats.get("roi_variance", 0.0) or 0.0),
                "monitoring_roi_variance": float(gray_card_stats.get("roi_variance", 0.0) or 0.0),
                "measured_saturation_fraction_monitoring": float(gray_card_stats.get("saturation_fraction", 0.0) or 0.0),
                "measurement_geometry": "gray_card_rectangular_sample_fallback",
                "sampling_method": "gray_card_detected_roi",
                "sampling_confidence": float(gray_target_confidence),
                "mask_fraction": float(gray_card_stats.get("mask_fraction", 0.0) or 0.0),
                "interior_radius_ratio": 1.0,
                "neutral_sample_count": int(gray_card_stats.get("valid_pixel_count", 0) or 0),
                "neutral_sample_log2_spread": float(gray_card_stats.get("gray_log2_stddev", 0.0) or 0.0),
                "neutral_sample_chromaticity_spread": float(gray_card_details.get("neutrality_score", 0.0) or 0.0),
                "neutral_samples": [],
                "zone_measurements": [],
                "gray_exposure_summary": gray_card_summary,
                "bright_ire": measured_ire,
                "center_ire": measured_ire,
                "dark_ire": measured_ire,
                "sample_1_ire": measured_ire,
                "sample_2_ire": measured_ire,
                "sample_3_ire": measured_ire,
                "top_ire": measured_ire,
                "mid_ire": measured_ire,
                "bottom_ire": measured_ire,
                "zone_spread_ire": 0.0,
                "zone_spread_stops": 0.0,
                "dominant_gradient_axis": {},
                "sphere_roi_source": "gray_card_fallback",
                "sphere_detection_confidence": float(detection_confidence),
                "sphere_detection_label": str(detection_label or "MEDIUM"),
                "sphere_detection_success": False,
                "sphere_detection_unresolved": False,
                "sphere_detection_details": detection_details,
                "detected_sphere_roi": {},
                "detected_gray_card_roi": gray_card_roi,
                "gray_card_detection_details": gray_card_details,
                "gray_target_class": gray_target_class,
                "gray_target_detection_method": gray_target_detection_method,
                "gray_target_confidence": float(gray_target_confidence),
                "gray_target_fallback_used": bool(gray_target_fallback_used),
                "gray_target_review_recommended": bool(gray_target_review_recommended),
                "manual_assist_used": bool(manual_assist_used),
                "manual_assist_metadata": manual_assist_metadata,
                "sample_plausibility": sample_plausibility,
                "detection_failed": False,
                "render_width": int(image_metadata.get("width", image.shape[1])),
                "render_height": int(image_metadata.get("height", image.shape[0])),
                "rendered_image_dtype": str(image_metadata.get("source_dtype") or ""),
                "rendered_image_mode": str(image_metadata.get("image_mode") or ""),
                "rendered_image_bit_depth": image_metadata.get("bit_depth"),
                "rendered_image_normalization_denominator": image_metadata.get("normalization_denominator"),
                "rendered_preview_format": str(image_metadata.get("preview_format") or ""),
                "measurement_region_scope": measurement_region_scope,
                "measurement_region_origin": {"x": int(region_origin_x), "y": int(region_origin_y)},
                "measurement_crop_bounds": {},
                "measurement_crop_size": {},
                "measurement_radius_px": None,
                "measurement_patch_radius_px": None,
                "measurement_patch_center": {},
                "measurement_source": "gray_card_detected_roi",
                "measurement_fallback_used": False,
                "hero_center_measurement": None,
                "measurement_runtime": {
                    "image_load_seconds": float(image_load_seconds),
                    "roi_extract_seconds": float(roi_extract_seconds),
                    "sphere_detection_seconds": float(sphere_detection_seconds),
                    "gray_card_detection_seconds": float(gray_card_detection_seconds),
                    "gradient_axis_seconds": 0.0,
                    "zone_stat_seconds": 0.0,
                    "profile_measurement_seconds": 0.0,
                    "total_measurement_seconds": float(time.perf_counter() - measurement_started_at),
                },
            }
        failed_payload = {
            "measurement_valid": False,
            "gray_target_measurement_valid": False,
            "measured_log2_luminance": None,
            "measured_log2_luminance_monitoring": None,
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
            "bright_ire": None,
            "center_ire": None,
            "dark_ire": None,
            "sample_1_ire": None,
            "sample_2_ire": None,
            "sample_3_ire": None,
            "top_ire": None,
            "mid_ire": None,
            "bottom_ire": None,
            "zone_spread_ire": None,
            "zone_spread_stops": None,
            "sphere_roi_source": str(sphere_roi_source or ("unresolved" if sphere_detection_unresolved else "failed")),
            "sphere_detection_confidence": float(detection_confidence),
            "sphere_detection_label": "UNRESOLVED" if sphere_detection_unresolved else "FAILED",
            "sphere_detection_success": False,
            "sphere_detection_unresolved": bool(sphere_detection_unresolved),
            "sphere_detection_details": detection_details,
            "detected_sphere_roi": {},
            "detected_gray_card_roi": dict(gray_card_details.get("roi") or {}),
            "gray_card_detection_details": gray_card_details,
            "gray_target_class": "unresolved",
            "gray_target_detection_method": str(sphere_roi_source or ("unresolved" if sphere_detection_unresolved else "failed")),
            "gray_target_confidence": float(detection_confidence),
            "gray_target_fallback_used": False,
            "gray_target_review_recommended": True,
            "manual_assist_used": bool(str(sphere_roi_source or "").startswith("manual_")),
            "manual_assist_metadata": manual_assist_metadata,
            "sample_plausibility": str(sample_plausibility or "IMPLAUSIBLE"),
            "detection_failed": not bool(sphere_detection_unresolved),
            "render_width": int(image_metadata.get("width", image.shape[1])),
            "render_height": int(image_metadata.get("height", image.shape[0])),
            "rendered_image_dtype": str(image_metadata.get("source_dtype") or ""),
            "rendered_image_mode": str(image_metadata.get("image_mode") or ""),
            "rendered_image_bit_depth": image_metadata.get("bit_depth"),
            "rendered_image_normalization_denominator": image_metadata.get("normalization_denominator"),
            "rendered_preview_format": str(image_metadata.get("preview_format") or ""),
            "measurement_region_scope": measurement_region_scope,
            "measurement_region_origin": {"x": int(region_origin_x), "y": int(region_origin_y)},
            "measurement_radius_px": None,
            "measurement_patch_radius_px": None,
            "measurement_patch_center": {},
            "measurement_source": "sphere_detection_failed",
            "measurement_fallback_used": False,
            "hero_center_measurement": None,
            "measurement_runtime": {
                "image_load_seconds": float(image_load_seconds),
                "roi_extract_seconds": float(roi_extract_seconds),
                "sphere_detection_seconds": float(sphere_detection_seconds),
                "gray_card_detection_seconds": float(gray_card_detection_seconds),
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
    luminance = _display_domain_luminance(cropped_region)
    measurement_radius_px = _sphere_measurement_radius(cropped_roi.r)
    hero_patch = _compute_sphere_center_patch(
        cropped_region,
        float(cropped_roi.cx),
        float(cropped_roi.cy),
        measurement_radius_px,
        luminance=luminance,
    )
    hero_measurement_valid = hero_patch is not None
    measurement_source = "hero_center_patch" if hero_measurement_valid else "legacy_center_fallback"
    measurement_fallback_used = not hero_measurement_valid
    if hero_measurement_valid:
        hero_center_luminance = float(hero_patch["center_luminance"])
        hero_center_log2 = float(math.log2(max(hero_center_luminance, 1e-6)))
        hero_center_ire = float(hero_center_luminance * 100.0)
        hero_center_rgb = [float(value) for value in hero_patch["center_rgb"]]
        hero_rgb_sum = max(float(sum(hero_center_rgb)), 1e-6)
        hero_center_chromaticity = [float(value / hero_rgb_sum) for value in hero_center_rgb]
    else:
        hero_center_log2 = float(stats["measured_log2_luminance"])
        hero_center_ire = float(stats.get("center_ire", stats.get("sample_2_ire", 0.0)) or 0.0)
        hero_center_rgb = [float(value) for value in stats["measured_rgb_mean"]]
        hero_center_chromaticity = [float(value) for value in stats["measured_rgb_chromaticity"]]
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
    for zone in zone_measurements:
        if str(zone.get("label") or "") != "center":
            continue
        zone["display_label"] = "Hero center"
        zone["measured_log2_luminance"] = float(hero_center_log2)
        zone["measured_ire"] = float(hero_center_ire)
        zone["measured_rgb_mean"] = [float(value) for value in hero_center_rgb]
        zone["measured_rgb_chromaticity"] = [float(value) for value in hero_center_chromaticity]
        zone["valid_pixel_count"] = int(hero_patch["sample_count"]) if hero_measurement_valid else int(zone.get("valid_pixel_count", 0) or 0)
        zone["sampling_method"] = str(measurement_source)
        break
    dominant_gradient_axis = dict(stats.get("dominant_gradient_axis") or {})
    bright_ire = float(stats.get("bright_ire", 0.0) or 0.0)
    dark_ire = float(stats.get("dark_ire", 0.0) or 0.0)
    hero_summary = f"Sample 1 {bright_ire:.0f} / Hero center {hero_center_ire:.0f} / Sample 3 {dark_ire:.0f} IRE"
    return {
        "measurement_valid": True,
        "gray_target_measurement_valid": True,
        "measured_log2_luminance": float(hero_center_log2),
        "measured_log2_luminance_monitoring": float(hero_center_log2),
        "measured_rgb_mean": [float(value) for value in hero_center_rgb],
        "measured_rgb_chromaticity": [float(value) for value in hero_center_chromaticity],
        "measured_rgb_chromaticity_monitoring": [float(value) for value in hero_center_chromaticity],
        "valid_pixel_count": int(hero_patch["sample_count"]) if hero_measurement_valid else int(stats["valid_pixel_count"]),
        "roi_variance": float(stats["roi_variance"]),
        "monitoring_roi_variance": float(stats["roi_variance"]),
        "measured_saturation_fraction_monitoring": saturation_fraction,
        "measurement_geometry": "hero_center_patch_within_measurement_radius",
        "sampling_method": str(measurement_source),
        "sampling_confidence": float(stats["confidence"]),
        "mask_fraction": float(stats["mask_fraction"]),
        "interior_radius_ratio": float(stats.get("interior_radius_ratio", 0.0) or 0.0),
        "neutral_sample_count": int(stats.get("neutral_sample_count", 0) or 0),
        "neutral_sample_log2_spread": float(stats.get("neutral_sample_log2_spread", 0.0) or 0.0),
        "neutral_sample_chromaticity_spread": float(stats.get("neutral_sample_chromaticity_spread", 0.0) or 0.0),
        "neutral_samples": [dict(item) for item in zone_measurements],
        "zone_measurements": [dict(item) for item in zone_measurements],
        "gray_exposure_summary": hero_summary,
        "bright_ire": bright_ire,
        "center_ire": float(hero_center_ire),
        "dark_ire": dark_ire,
        "sample_1_ire": float(stats.get("sample_1_ire", bright_ire) or bright_ire),
        "sample_2_ire": float(hero_center_ire),
        "sample_3_ire": float(stats.get("sample_3_ire", dark_ire) or dark_ire),
        "top_ire": bright_ire,
        "mid_ire": float(hero_center_ire),
        "bottom_ire": dark_ire,
        "zone_spread_ire": float(stats.get("zone_spread_ire", 0.0) or 0.0),
        "zone_spread_stops": float(stats.get("zone_spread_stops", 0.0) or 0.0),
        "dominant_gradient_axis": dominant_gradient_axis,
        "sphere_roi_source": sphere_roi_source,
        "sphere_detection_confidence": detection_confidence,
        "sphere_detection_label": detection_label,
        "sphere_detection_success": True,
        "sphere_detection_unresolved": False,
        "sphere_detection_details": detection_details,
        "detected_sphere_roi": {"cx": float(sphere_roi.cx), "cy": float(sphere_roi.cy), "r": float(sphere_roi.r)},
        "detected_gray_card_roi": dict(gray_card_details.get("roi") or {}),
        "gray_card_detection_details": gray_card_details,
        "gray_target_class": gray_target_class,
        "gray_target_detection_method": gray_target_detection_method,
        "gray_target_confidence": float(gray_target_confidence),
        "gray_target_fallback_used": bool(gray_target_fallback_used),
        "gray_target_review_recommended": bool(gray_target_review_recommended),
        "manual_assist_used": bool(manual_assist_used),
        "manual_assist_metadata": manual_assist_metadata,
        "sample_plausibility": sample_plausibility,
        "detection_failed": False,
        "render_width": int(image_metadata.get("width", image.shape[1])),
        "render_height": int(image_metadata.get("height", image.shape[0])),
        "rendered_image_dtype": str(image_metadata.get("source_dtype") or ""),
        "rendered_image_mode": str(image_metadata.get("image_mode") or ""),
        "rendered_image_bit_depth": image_metadata.get("bit_depth"),
        "rendered_image_normalization_denominator": image_metadata.get("normalization_denominator"),
        "rendered_preview_format": str(image_metadata.get("preview_format") or ""),
        "measurement_region_scope": measurement_region_scope,
        "measurement_region_origin": {"x": int(region_origin_x), "y": int(region_origin_y)},
        "measurement_crop_bounds": dict(crop_bounds),
        "measurement_crop_size": {"width": int(cropped_region.shape[1]), "height": int(cropped_region.shape[0])},
        "measurement_radius_px": float(measurement_radius_px),
        "measurement_patch_radius_px": float(hero_patch["radius_px"]) if hero_measurement_valid else float(_sphere_hero_patch_radius(cropped_roi.r)),
        "measurement_patch_center": {"x": float(cropped_roi.cx), "y": float(cropped_roi.cy)},
        "measurement_source": str(measurement_source),
        "measurement_fallback_used": bool(measurement_fallback_used),
        "hero_center_measurement": copy.deepcopy(hero_patch)
        if hero_measurement_valid
        else {
            "center_luminance": float(2.0 ** hero_center_log2),
            "center_rgb": [float(value) for value in hero_center_rgb],
            "sample_count": int(stats["valid_pixel_count"]),
            "radius_px": float(_sphere_hero_patch_radius(cropped_roi.r)),
            "measurement_radius_px": float(measurement_radius_px),
            "center_x": float(cropped_roi.cx),
            "center_y": float(cropped_roi.cy),
        },
        "measurement_runtime": {
            "image_load_seconds": float(image_load_seconds),
            "roi_extract_seconds": float(roi_extract_seconds),
            "sphere_detection_seconds": float(sphere_detection_seconds),
            "gray_card_detection_seconds": float(gray_card_detection_seconds),
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
        label = "Profile aligned"
        note = "Sample profile aligned"
        tone = "good"
    elif worst_label == "dark_side" and worst_abs <= PROFILE_AUDIT_REVIEW_STOPS and abs(spread_delta) <= PROFILE_AUDIT_REVIEW_SPREAD_IRE:
        status = "PROFILE NEEDS REVIEW"
        label = "Profile needs attention"
        note = "Shadow-side sample differs slightly"
        tone = "warning"
    elif worst_abs <= PROFILE_AUDIT_REVIEW_STOPS and abs(spread_delta) <= PROFILE_AUDIT_REVIEW_SPREAD_IRE:
        status = "PROFILE NEEDS REVIEW"
        label = "Profile needs attention"
        note = "Sample profile needs review"
        tone = "warning"
    else:
        status = "PROFILE SHAPE MISMATCH"
        label = "Profile mismatch"
        note = "Sample profile does not match expectation — verify stop, lighting, or ROI"
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
    artifact_mode: str = DEFAULT_ARTIFACT_MODE,
) -> Dict[str, object]:
    solver_started_at = time.perf_counter()
    ipp2_preview_settings = _ipp2_validation_preview_settings()
    preview_root = Path(input_path).expanduser().resolve() / "previews" / "_ipp2_closed_loop"
    preview_root.mkdir(parents=True, exist_ok=True)
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
        artifact_mode=artifact_mode,
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
                    "initial_sphere_detection_success": bool(corrected_measure.get("sphere_detection_success")),
                    "initial_sphere_detection_unresolved": bool(corrected_measure.get("sphere_detection_unresolved")),
                    "initial_detected_sphere_roi": dict(corrected_measure.get("detected_sphere_roi") or {}),
                    "initial_measurement_radius_px": _numeric_value_or_none(corrected_measure.get("measurement_radius_px")),
                    "initial_measurement_patch_radius_px": _numeric_value_or_none(corrected_measure.get("measurement_patch_radius_px")),
                    "initial_measurement_patch_center": dict(corrected_measure.get("measurement_patch_center") or {}),
                    "original_ipp2_value_log2": float(original_measure.get("measured_log2_luminance_monitoring", 0.0) or 0.0),
                    "original_ipp2_zone_profile": [dict(item) for item in list(original_measure.get("zone_measurements") or [])],
                    "original_gray_exposure_summary": str(original_measure.get("gray_exposure_summary") or "n/a"),
                    "original_sphere_detection_source": str(original_measure.get("sphere_roi_source") or "failed"),
                    "original_sphere_detection_confidence": float(original_measure.get("sphere_detection_confidence", 0.0) or 0.0),
                    "original_sphere_detection_label": str(original_measure.get("sphere_detection_label") or "FAILED"),
                    "original_sphere_detection_success": bool(original_measure.get("sphere_detection_success")),
                    "original_sphere_detection_unresolved": bool(original_measure.get("sphere_detection_unresolved")),
                    "original_detected_sphere_roi": dict(original_measure.get("detected_sphere_roi") or {}),
                    "original_measurement_radius_px": _numeric_value_or_none(original_measure.get("measurement_radius_px")),
                    "original_measurement_patch_radius_px": _numeric_value_or_none(original_measure.get("measurement_patch_radius_px")),
                    "original_measurement_patch_center": dict(original_measure.get("measurement_patch_center") or {}),
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
                        render = _render_preview_frame_with_retries(
                            row["source_path"],
                            str(iteration_path),
                            clip_id=clip_id,
                            variant=f"closed_loop_{strategy_key}_iter_{iteration}",
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
                            max_attempts=4,
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
            final_best_path = str(best_path)
            if normalize_artifact_mode(artifact_mode) == "production" and str(best_path or "").strip():
                best_path_obj = Path(str(best_path)).expanduser().resolve()
                if best_path_obj.exists() and preview_root in best_path_obj.parents:
                    relocated_path = preview_root.parent / best_path_obj.name
                    relocated_path.parent.mkdir(parents=True, exist_ok=True)
                    if relocated_path.exists():
                        relocated_path.unlink()
                    shutil.move(str(best_path_obj), str(relocated_path))
                    final_best_path = str(relocated_path)
                    best_path = final_best_path
            preview_paths[clip_id]["strategies"][strategy_key]["both"] = str(final_best_path)
            if normalize_artifact_mode(artifact_mode) == "debug" and not _color_preview_policy()["enabled"]:
                preview_paths[clip_id]["strategies"][strategy_key]["exposure"] = str(final_best_path)
            if normalize_artifact_mode(artifact_mode) == "production":
                cleanup_candidates = {
                    str(row.get("initial_image_path") or ""),
                    *(str(item.get("image_path") or "") for item in history),
                }
                cleanup_candidates.discard(str(final_best_path or ""))
                for candidate_path in cleanup_candidates:
                    candidate_text = str(candidate_path or "").strip()
                    if not candidate_text:
                        continue
                    candidate_obj = Path(candidate_text).expanduser().resolve()
                    if not candidate_obj.exists():
                        continue
                    try:
                        candidate_obj.unlink()
                    except OSError:
                        pass
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
                    "ipp2_original_measurement_radius_px": _numeric_value_or_none(row.get("original_measurement_radius_px")),
                    "ipp2_original_measurement_patch_radius_px": _numeric_value_or_none(row.get("original_measurement_patch_radius_px")),
                    "ipp2_original_measurement_patch_center": copy.deepcopy(row.get("original_measurement_patch_center") or {}),
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
                    "ipp2_sphere_detection_success": bool((render_cache.get((clip_id, strategy_key, round(float(best_correction), 6))) or {}).get("measure", {}).get("sphere_detection_success", row.get("initial_sphere_detection_success", False))),
                    "ipp2_sphere_detection_unresolved": bool((render_cache.get((clip_id, strategy_key, round(float(best_correction), 6))) or {}).get("measure", {}).get("sphere_detection_unresolved", row.get("initial_sphere_detection_unresolved", False))),
                    "ipp2_detected_sphere_roi": copy.deepcopy(((render_cache.get((clip_id, strategy_key, round(float(best_correction), 6))) or {}).get("measure", {}).get("detected_sphere_roi") or row.get("initial_detected_sphere_roi") or {})),
                    "ipp2_measurement_radius_px": _numeric_value_or_none(
                        ((render_cache.get((clip_id, strategy_key, round(float(best_correction), 6))) or {}).get("measure", {})).get(
                            "measurement_radius_px",
                            row.get("initial_measurement_radius_px"),
                        )
                    ),
                    "ipp2_measurement_patch_radius_px": _numeric_value_or_none(
                        ((render_cache.get((clip_id, strategy_key, round(float(best_correction), 6))) or {}).get("measure", {})).get(
                            "measurement_patch_radius_px",
                            row.get("initial_measurement_patch_radius_px"),
                        )
                    ),
                    "ipp2_measurement_patch_center": copy.deepcopy(
                        ((render_cache.get((clip_id, strategy_key, round(float(best_correction), 6))) or {}).get("measure", {})).get(
                            "measurement_patch_center"
                        )
                        or row.get("initial_measurement_patch_center")
                        or {}
                    ),
                    "detection_failed": bool(best_summary.get("detection_failed")),
                    "original_image_path": str(initial_preview_paths.get(clip_id, {}).get("original") or ""),
                    "corrected_image_path": str(final_best_path),
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
    if normalize_artifact_mode(artifact_mode) == "production" and preview_root.exists():
        shutil.rmtree(preview_root, ignore_errors=True)
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
                    sphere_detection_success=bool(row.get("ipp2_sphere_detection_success")),
                    sphere_detection_unresolved=bool(row.get("ipp2_sphere_detection_unresolved")),
                    fallback_used=bool(row.get("gray_target_fallback_used")),
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
                        "ipp2_original_sphere_detection_success": bool(original_measure.get("sphere_detection_success")),
                        "ipp2_original_sphere_detection_unresolved": bool(original_measure.get("sphere_detection_unresolved")),
                        "ipp2_original_detected_sphere_roi": dict(original_measure.get("detected_sphere_roi") or {}),
                        "ipp2_original_measurement_region_scope": str(original_measure.get("measurement_region_scope") or "calibration_roi"),
                        "ipp2_original_measurement_region_origin": dict(original_measure.get("measurement_region_origin") or {}),
                        "ipp2_original_measurement_radius_px": _numeric_value_or_none(original_measure.get("measurement_radius_px")),
                        "ipp2_original_measurement_patch_radius_px": _numeric_value_or_none(original_measure.get("measurement_patch_radius_px")),
                        "ipp2_original_measurement_patch_center": dict(original_measure.get("measurement_patch_center") or {}),
                        "ipp2_original_sample_plausibility": str(original_measure.get("sample_plausibility") or ""),
                        "ipp2_original_detection_details": dict(original_measure.get("sphere_detection_details") or {}),
                        "ipp2_detection_source": str(corrected_measure.get("sphere_roi_source") or "failed"),
                        "ipp2_detection_confidence": float(corrected_measure.get("sphere_detection_confidence", 0.0) or 0.0),
                        "ipp2_detection_label": str(corrected_measure.get("sphere_detection_label") or "FAILED"),
                        "ipp2_sphere_detection_success": bool(corrected_measure.get("sphere_detection_success")),
                        "ipp2_sphere_detection_unresolved": bool(corrected_measure.get("sphere_detection_unresolved")),
                        "ipp2_detected_sphere_roi": dict(corrected_measure.get("detected_sphere_roi") or {}),
                        "ipp2_measurement_region_scope": str(corrected_measure.get("measurement_region_scope") or "calibration_roi"),
                        "ipp2_measurement_region_origin": dict(corrected_measure.get("measurement_region_origin") or {}),
                        "ipp2_measurement_radius_px": _numeric_value_or_none(corrected_measure.get("measurement_radius_px")),
                        "ipp2_measurement_patch_radius_px": _numeric_value_or_none(corrected_measure.get("measurement_patch_radius_px")),
                        "ipp2_measurement_patch_center": dict(corrected_measure.get("measurement_patch_center") or {}),
                        "ipp2_sample_plausibility": str(corrected_measure.get("sample_plausibility") or ""),
                        "ipp2_detection_details": dict(corrected_measure.get("sphere_detection_details") or {}),
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
                    sphere_detection_success=bool(row.get("ipp2_sphere_detection_success")),
                    sphere_detection_unresolved=bool(row.get("ipp2_sphere_detection_unresolved")),
                    fallback_used=bool(row.get("gray_target_fallback_used")),
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

    def measurement_valid_for_record(record: Dict[str, object]) -> bool:
        return bool(measurement_for_record(record).get("measurement_valid"))

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
            if not measurement_valid_for_record(record):
                screened_rows.append(
                    {
                        "index": index,
                        "clip_id": str(record["clip_id"]),
                        "measured_log2": float("nan"),
                        "confidence": 0.0,
                        "sample_log2_spread": 0.0,
                        "sample_chroma_spread": 0.0,
                        "exposure_error": float("inf"),
                        "stability_penalty": 0.0,
                        "confidence_penalty": 1.0,
                        "chroma_penalty": 0.0,
                        "score": float("inf"),
                        "reasons": ["measurement invalid"],
                    }
                )
                continue
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
                if measurement_valid_for_record(analysis_records[index])
                and confidence_for_record(analysis_records[index]) >= (OPTIMAL_EXPOSURE_MIN_CONFIDENCE * 0.8)
            ]
            if not fallback_candidates:
                fallback_candidates = [index for index in list(primary_cluster_indices) if measurement_valid_for_record(analysis_records[index])]
            if not fallback_candidates:
                fallback_candidates = [index for index, record in enumerate(analysis_records) if measurement_valid_for_record(record)]
            if not fallback_candidates:
                fallback_candidates = [0]
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
        if not bool(measurement.get("measurement_valid")):
            return float("nan")
        return float(measurement.get("display_scalar_log2", measurement["log2_luminance"]))

    def measured_rgb_for_record(record: Dict[str, object]) -> List[float]:
        return [float(value) for value in measurement_for_record(record)["rgb_chromaticity"]]

    def measured_saturation_for_record(record: Dict[str, object]) -> float:
        return float(measurement_for_record(record)["saturation_fraction"])

    measured_log2 = np.array([measured_log2_for_record(record) for record in analysis_records], dtype=np.float32)
    measured_chroma = np.array([measured_rgb_for_record(record) for record in analysis_records], dtype=np.float32)
    measured_saturation = np.array([measured_saturation_for_record(record) for record in analysis_records], dtype=np.float32)
    valid_indices = [index for index, record in enumerate(analysis_records) if measurement_valid_for_record(record) and np.isfinite(measured_log2[index])]
    valid_measured_log2 = np.asarray([float(measured_log2[index]) for index in valid_indices], dtype=np.float32)
    valid_measured_chroma = np.asarray([measured_chroma[index] for index in valid_indices], dtype=np.float32) if valid_indices else np.empty((0, 3), dtype=np.float32)
    valid_measured_saturation = np.asarray([float(measured_saturation[index]) for index in valid_indices], dtype=np.float32) if valid_indices else np.empty((0,), dtype=np.float32)
    batch_target_log2 = float(anchor_target_log2) if anchor_target_log2 is not None else float(np.median(valid_measured_log2)) if valid_measured_log2.size else 0.0
    batch_target_rgb = [float(np.median(valid_measured_chroma[:, index])) for index in range(3)] if valid_measured_chroma.size else [1 / 3, 1 / 3, 1 / 3]
    payloads: List[Dict[str, object]] = []
    for requested in target_strategies:
        strategy = normalize_target_strategy_name(requested)
        selection_diagnostics: Dict[str, object] = {}
        anchor_mode_for_strategy = strategy
        anchor_source = ""
        anchor_ire_summary = "n/a"
        if strategy == "median":
            target_log2 = float(np.median(valid_measured_log2)) if valid_measured_log2.size else 0.0
            target_rgb = [float(np.median(valid_measured_chroma[:, index])) for index in range(3)] if valid_measured_chroma.size else [1 / 3, 1 / 3, 1 / 3]
            target_saturation = float(np.median(valid_measured_saturation)) if valid_measured_saturation.size and saturation_supported else 1.0
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
            target_rgb = [float(np.median(valid_measured_chroma[:, index])) for index in range(3)] if valid_measured_chroma.size else [1 / 3, 1 / 3, 1 / 3]
            target_saturation = float(np.median(valid_measured_saturation)) if valid_measured_saturation.size and saturation_supported else 1.0
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
            measurement_valid = bool(resolved_measurement.get("measurement_valid"))
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
            pre_exposure_residual = abs(0.0 if is_hero_camera or not measurement_valid else float(target_log2 - measured_monitoring_log2))
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
                    "measurement_valid": measurement_valid,
                    "gray_target_measurement_valid": measurement_valid,
                    "display_scalar_log2": (None if not measurement_valid else measured_monitoring_log2),
                    "display_scalar_domain": "display_ipp2" if resolved_matching_domain == "perceptual" else "scene_analysis",
                    "measured_log2_luminance": (None if not measurement_valid else measured_monitoring_log2),
                    "measured_log2_luminance_monitoring": (None if not measurement_valid else measured_monitoring_log2),
                    "measured_log2_luminance_raw": (_numeric_value_or_none(measured.get("measured_log2_luminance_raw", measured.get("measured_log2_luminance"))) if measurement_valid else None),
                    "measured_rgb_chromaticity": [float(value) for value in measured_rgb],
                    "monitoring_measurement_source": str(resolved_measurement["source"]),
                    "gray_exposure_summary": str(resolved_measurement.get("gray_exposure_summary") or measured.get("gray_exposure_summary") or "n/a"),
                    "zone_measurements": [dict(item) for item in list(resolved_measurement.get("zone_measurements") or [])],
                    "exposure_offset_stops": 0.0 if is_hero_camera or not measurement_valid else float(target_log2 - measured_monitoring_log2),
                    "camera_offset_from_anchor": 0.0 if is_hero_camera or not measurement_valid else float(target_log2 - measured_monitoring_log2),
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

        white_balance_summary = _canonical_white_balance_model_summary(
            strategy_clips,
            {
                "model_key": wb_model_solution.get("model_key"),
                "model_label": wb_model_solution.get("model_label"),
                "shared_kelvin": wb_model_solution.get("shared_kelvin"),
                "shared_tint": wb_model_solution.get("shared_tint"),
                "metrics": wb_model_solution.get("metrics"),
                "candidates": wb_model_solution.get("candidates", []),
            },
        )

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
                "white_balance_model": white_balance_summary,
                "clips": strategy_clips,
            }
        )
        print(
            f"[r3dmatch] review strategy={strategy} target_monitoring_log2="
            f"{'unresolved' if target_log2 is None else f'{float(target_log2):.6f}'} "
            f"reference={resolved_reference or 'median'}"
        )
        for clip in strategy_clips:
            clip_log2 = clip.get("measured_log2_luminance_monitoring")
            print(
                f"[r3dmatch] review clip={clip['clip_id']} strategy={strategy} "
                f"monitoring_log2={'unresolved' if clip_log2 is None else f'{float(clip_log2):.6f}'} "
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
    clips = [clip for clip in strategy_payloads[0].get("clips", []) if bool(clip.get("measurement_valid"))]
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
    valid_clips = [clip for clip in clips if bool(clip.get("measurement_valid", True)) and clip.get("measured_log2_luminance_monitoring") is not None]
    exposures = np.array([float(clip.get("measured_log2_luminance_monitoring", 0.0) or 0.0) for clip in valid_clips], dtype=np.float32)
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
    valid_clips = [clip for clip in clips if bool(clip.get("measurement_valid", True))]
    if not valid_clips:
        return ""

    def _display_ire_value(clip: Dict[str, object]) -> float:
        raw_ire = clip.get("display_scalar_ire")
        if isinstance(raw_ire, (int, float)):
            return float(raw_ire)
        raw_log2 = clip.get("display_scalar_log2", clip.get("measured_log2_luminance_monitoring", clip.get("measured_log2_luminance")))
        if raw_log2 is None:
            return 0.0
        try:
            return float((2.0 ** float(raw_log2)) * 100.0)
        except Exception:
            return 0.0

    ordered = sorted(valid_clips, key=_display_ire_value, reverse=True)
    values = [_display_ire_value(clip) for clip in ordered]
    if len(values) < 2:
        return ""
    minimum = min(values)
    maximum = max(values)
    spread = maximum - minimum
    if spread < 0.5:
        return ""
    median_value = float(np.median(np.asarray(values, dtype=np.float32)))
    median_abs_deviation = float(np.median(np.abs(np.asarray(values, dtype=np.float32) - median_value)))
    threshold = float(outlier_threshold or max(0.75, median_abs_deviation * 1.75, spread * 0.14))
    lower_cluster = median_value - threshold
    upper_cluster = median_value + threshold
    width = 1180
    height = 440
    pad_left = 96
    pad_right = 68
    pad_top = 42
    pad_bottom = 88
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
        value = _display_ire_value(clip)
        y = y_for(value)
        points.append(f"{x:.1f},{y:.1f}")
        if target_log2 is not None:
            target_y = y_for(float((2.0 ** float(target_log2)) * 100.0))
            stems.append(f"<line x1=\"{x:.1f}\" y1=\"{y:.1f}\" x2=\"{x:.1f}\" y2=\"{target_y:.1f}\" stroke=\"#cbd5e1\" stroke-width=\"2\" stroke-dasharray=\"4 4\"/>")
        is_anchor = reference_clip_id and str(clip.get("clip_id")) == str(reference_clip_id)
        is_outlier = abs(value - median_value) > threshold
        should_label = len(ordered) <= 14 or is_anchor or is_outlier or (len(ordered) <= 24 and index % 2 == 0)
        if should_label:
            labels.append(
                _svg_wrapped_text(
                    _camera_label_for_reporting(str(clip["clip_id"])),
                    x=x,
                    y=height - 44,
                    width_chars=11,
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
        grid_lines.append(f"<text x=\"{pad_left - 12}\" y=\"{y + 4:.1f}\" text-anchor=\"end\" font-size=\"11\" fill=\"#64748b\">{value:.1f}</text>")
    cluster_band = (
        f"<rect x=\"{pad_left}\" y=\"{y_for(upper_cluster):.1f}\" width=\"{inner_width:.1f}\" height=\"{max(y_for(lower_cluster) - y_for(upper_cluster), 6):.1f}\" fill=\"#dbeafe\" opacity=\"0.45\" rx=\"14\"/>"
    )
    cluster_line = f"<line x1=\"{pad_left}\" y1=\"{y_for(median_value):.1f}\" x2=\"{width - pad_right}\" y2=\"{y_for(median_value):.1f}\" stroke=\"#1d4ed8\" stroke-width=\"2\" stroke-dasharray=\"6 5\"/>"
    target_line = (
        f"<line x1=\"{pad_left}\" y1=\"{y_for(float((2.0 ** float(target_log2)) * 100.0)):.1f}\" x2=\"{width - pad_right}\" y2=\"{y_for(float((2.0 ** float(target_log2)) * 100.0)):.1f}\" stroke=\"#0f766e\" stroke-width=\"2\"/>"
        if target_log2 is not None
        else ""
    )
    legend = (
        f"<text x=\"{pad_left}\" y=\"24\" font-size=\"13\" font-weight=\"700\" fill=\"#1d4ed8\">Blue band: retained cluster</text>"
        f"<text x=\"{pad_left + 220}\" y=\"24\" font-size=\"13\" font-weight=\"700\" fill=\"#0f766e\">Green line: target center IRE</text>"
        f"<text x=\"{pad_left + 470}\" y=\"24\" font-size=\"13\" font-weight=\"700\" fill=\"#dc2626\">Red point: attention / outlier</text>"
        f"<text x=\"{pad_left}\" y=\"{max(y_for(median_value) - 10, 40):.1f}\" font-size=\"11\" fill=\"#1d4ed8\">retained cluster range</text>"
        + (
            f"<text x=\"{width - pad_right - 170}\" y=\"{max(y_for(float((2.0 ** float(target_log2)) * 100.0)) - 10, 40):.1f}\" font-size=\"11\" fill=\"#0f766e\">target center IRE</text>"
            if target_log2 is not None
            else ""
        )
        + f"<text x=\"{width - pad_right}\" y=\"24\" font-size=\"12\" text-anchor=\"end\" fill=\"#334155\">display-domain center scalar (IRE)</text>"
    )
    circles = []
    for clip, point in zip(ordered, points):
        x, y = point.split(",")
        value = _display_ire_value(clip)
        is_outlier = abs(value - median_value) > threshold
        is_anchor = reference_clip_id and str(clip.get("clip_id")) == str(reference_clip_id)
        fill = "#f59e0b" if is_anchor else "#dc2626" if is_outlier else "#0f172a"
        radius = 6 if is_anchor else 5
        circles.append(f"<circle cx=\"{x}\" cy=\"{y}\" r=\"{radius}\" fill=\"{fill}\" stroke=\"#ffffff\" stroke-width=\"2\"/>")
    return (
        f"<svg viewBox=\"0 0 {width} {height}\" role=\"img\" aria-label=\"Per-camera exposure plot in display-domain IRE\">"
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
    ordered = [
        clip
        for clip in sorted(clips, key=lambda clip: float(abs(clip.get("exposure_offset_stops", 0.0) or 0.0)), reverse=True)
        if bool(clip.get("measurement_valid", True)) and clip.get("measured_log2_luminance_monitoring") is not None
    ]
    if not ordered:
        return ""
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


def _upgrade_trust_from_ipp2_validation(
    trust_details: Dict[str, object],
    *,
    ipp2_validation: Dict[str, object],
    gray_target_class: str,
    sample_plausibility: str,
) -> Dict[str, object]:
    upgraded = dict(trust_details or {})
    normalized_target = str(gray_target_class or "").strip().lower()
    if normalized_target != "sphere":
        return upgraded
    status = str(ipp2_validation.get("status") or "").strip().upper()
    if status != "PASS":
        return upgraded
    current_class = str(upgraded.get("trust_class") or "").strip().upper()
    if current_class not in {"EXCLUDED", "UNTRUSTED"}:
        return upgraded
    profile_label = str(ipp2_validation.get("profile_audit_label") or "").strip().lower()
    validation_residual = float(
        ipp2_validation.get(
            "derived_residual_abs_stops",
            ipp2_validation.get("validation_residual_stops", ipp2_validation.get("residual_stops", 0.0)),
        )
        or 0.0
    )
    plausibility = str(sample_plausibility or "").strip().upper()
    screened_reasons = [
        str(item).strip()
        for item in list(upgraded.get("screened_reasons") or [])
        if str(item).strip()
    ]
    cluster_only_reasons = {
        "Outside primary exposure group",
        "Outside stable exposure cluster",
        "outside primary exposure cluster",
        "outside central exposure cluster",
    }
    has_non_cluster_reason = any(reason not in cluster_only_reasons for reason in screened_reasons)
    if has_non_cluster_reason and validation_residual > CORRECTED_RESIDUAL_PASS_STOPS:
        return upgraded
    if profile_label == "profile aligned" and plausibility in {"HIGH", "MEDIUM"}:
        upgraded["trust_class"] = "TRUSTED"
        upgraded["trust_score"] = max(float(upgraded.get("trust_score", 0.0) or 0.0), 0.82)
        upgraded["trust_reason"] = "Recovered on the sphere after corrected-frame validation"
        upgraded["stability_label"] = "Validated on corrected frame"
        upgraded["correction_confidence"] = "MEDIUM"
        upgraded["reference_use"] = "Included"
    else:
        upgraded["trust_class"] = "USE_WITH_CAUTION"
        upgraded["trust_score"] = max(float(upgraded.get("trust_score", 0.0) or 0.0), 0.64)
        upgraded["trust_reason"] = "Recovered on the sphere after corrected-frame validation"
        upgraded["stability_label"] = "Validated after correction"
        upgraded["correction_confidence"] = "MEDIUM"
        upgraded["reference_use"] = "Included"
    if "Recovered on corrected-frame validation" not in screened_reasons:
        screened_reasons.append("Recovered on corrected-frame validation")
    upgraded["screened_reasons"] = screened_reasons
    return upgraded


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


def _white_balance_mode_from_values(values: List[float], *, tolerance: float) -> Tuple[str, Optional[float]]:
    if not values:
        return ("per_camera", None)
    baseline = float(values[0])
    if all(math.isclose(float(value), baseline, abs_tol=tolerance) for value in values[1:]):
        return ("shared", baseline)
    return ("per_camera", None)


def _fallback_white_balance_label(*, kelvin_mode: str, tint_mode: str) -> str:
    if kelvin_mode == "shared" and tint_mode == "shared":
        return "Shared Kelvin / Shared Tint"
    if kelvin_mode == "shared" and tint_mode == "per_camera":
        return "Shared Kelvin / Per-Camera Tint"
    if kelvin_mode == "per_camera" and tint_mode == "shared":
        return "Per-Camera Kelvin / Shared Tint"
    return "Per-Camera Kelvin / Per-Camera Tint"


def _canonical_white_balance_model_summary(
    strategy_clips: List[Dict[str, object]],
    raw_summary: Optional[Dict[str, object]],
) -> Dict[str, object]:
    summary = dict(raw_summary or {})
    clip_rows = [dict(item) for item in strategy_clips if str(item.get("clip_id") or "").strip()]
    commit_rows = [dict(item.get("commit_values") or {}) for item in clip_rows if isinstance(item.get("commit_values"), dict)]
    kelvin_values = [int(row.get("kelvin")) for row in commit_rows if row.get("kelvin") is not None]
    tint_values = [float(row.get("tint")) for row in commit_rows if row.get("tint") is not None]
    pre_residuals = [float(row.get("pre_neutral_residual")) for row in commit_rows if row.get("pre_neutral_residual") is not None]
    post_residuals = [float(row.get("post_neutral_residual")) for row in commit_rows if row.get("post_neutral_residual") is not None]
    sample_counts = [int(item.get("neutral_sample_count", 0) or 0) for item in clip_rows]
    candidate_rows = [dict(item) for item in list(summary.get("candidates") or [])]
    chosen_model_key = str(summary.get("model_key") or "").strip()
    direct_summary = bool(chosen_model_key or candidate_rows)
    kelvin_mode, shared_kelvin_value = _white_balance_mode_from_values([float(value) for value in kelvin_values], tolerance=50.0)
    tint_mode, shared_tint_value = _white_balance_mode_from_values(tint_values, tolerance=0.1)
    fallback_label = _fallback_white_balance_label(kelvin_mode=kelvin_mode, tint_mode=tint_mode)
    if direct_summary:
        model_label = str(summary.get("model_label") or fallback_label)
        model_key = chosen_model_key or "white_balance_model"
        selection_reason = str(summary.get("selection_reason") or "")
        derivation_kind = "direct_solve"
        source_measurement_type = "neutral_rgb_samples"
    else:
        model_label = "Derived from Final WB Values"
        model_key = "fallback_final_commit_values"
        selection_reason = (
            "Derived from final per-camera Kelvin/Tint payloads because no white-balance model comparison summary was serialized."
            if commit_rows
            else "No white-balance solution is available."
        )
        derivation_kind = "fallback_derived"
        source_measurement_type = "final_commit_values"
        if commit_rows:
            candidate_rows = [
                {
                    "model_key": f"fallback_{kelvin_mode}_{tint_mode}",
                    "model_label": fallback_label,
                    "score": 0.0,
                    "metrics": {},
                    "shared_kelvin": int(round(shared_kelvin_value)) if shared_kelvin_value is not None else None,
                    "shared_tint": round(float(shared_tint_value), 1) if shared_tint_value is not None else None,
                }
            ]
    if not selection_reason and direct_summary:
        selection_reason = (
            f"{model_label} was chosen from {len(candidate_rows)} white-balance candidate model(s) using stored neutral RGB sample residuals."
            if candidate_rows
            else f"{model_label} was chosen from the stored white-balance solve."
        )
    metrics = dict(summary.get("metrics") or {})
    diagnostics = {
        "retained_camera_count": len(clip_rows),
        "mean_neutral_residual_before_solve": float(np.mean(pre_residuals)) if pre_residuals else 0.0,
        "mean_neutral_residual_after_solve": float(np.mean(post_residuals)) if post_residuals else 0.0,
        "max_neutral_residual": float(max(post_residuals)) if post_residuals else 0.0,
        "kelvin_spread": int(round(max(kelvin_values) - min(kelvin_values))) if len(kelvin_values) >= 2 else 0,
        "tint_spread": round(float(max(tint_values) - min(tint_values)), 3) if len(tint_values) >= 2 else 0.0,
        "source_sample_count": int(sum(sample_counts)),
        "summary_kind": derivation_kind,
    }
    diagnostics.update({
        key: value for key, value in metrics.items()
        if key in {"weighted_mean_post_neutral_residual", "max_post_neutral_residual", "kelvin_axis_stddev"}
    })
    return {
        "status": "available" if commit_rows else "unavailable",
        "model_key": model_key,
        "model_label": model_label,
        "candidate_count": int(len(candidate_rows) or (1 if commit_rows else 0)),
        "shared_kelvin_mode": kelvin_mode,
        "shared_kelvin": int(round(shared_kelvin_value)) if shared_kelvin_value is not None else None,
        "shared_tint_mode": tint_mode,
        "shared_tint": round(float(shared_tint_value), 1) if shared_tint_value is not None else None,
        "selection_reason": selection_reason,
        "source_measurement_type": source_measurement_type,
        "candidates": candidate_rows,
        "diagnostics": diagnostics,
    }


def _strategy_chart_is_informative(strategy_summaries: List[Dict[str, object]]) -> bool:
    if len(strategy_summaries) < 2:
        return False
    mean_offsets = [float((item.get("correction_metrics") or {}).get("mean_abs_offset", 0.0) or 0.0) for item in strategy_summaries]
    max_offsets = [float((item.get("correction_metrics") or {}).get("max_abs_offset", 0.0) or 0.0) for item in strategy_summaries]
    mean_confidences = [float((item.get("correction_metrics") or {}).get("mean_confidence", 0.0) or 0.0) for item in strategy_summaries]
    return (
        (max(mean_offsets) - min(mean_offsets) >= 0.03)
        or (max(max_offsets) - min(max_offsets) >= 0.05)
        or (max(mean_confidences) - min(mean_confidences) >= 0.08)
    )


def _trust_chart_is_informative(rows: List[Dict[str, object]]) -> bool:
    if len(rows) < 2:
        return False
    trust_classes = {str(item.get("trust_class") or "") for item in rows if str(item.get("trust_class") or "").strip()}
    trust_scores = [float(item.get("trust_score", 0.0) or 0.0) for item in rows]
    return len(trust_classes) > 1 or ((max(trust_scores) - min(trust_scores)) >= 0.08 if trust_scores else False)


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
    gray_target_consistency = _gray_target_consistency_summary(per_camera_rows)
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
    if bool(gray_target_consistency.get("mixed_target_classes")):
        gating_reasons.append("Retained cameras mix gray-sphere and gray-card solves, so a shared safe commit set is blocked.")

    if bool(gray_target_consistency.get("mixed_target_classes")):
        run_status = "DO_NOT_PUSH"
        recommendation_strength = "LOW_CONFIDENCE"
    elif reference_eligible_count < minimum_reference_count or (anchor_row and anchor_trust_class in {"UNTRUSTED", "EXCLUDED"}):
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

    if bool(gray_target_consistency.get("mixed_target_classes")):
        operator_note = "Review is required before commit because retained cameras were measured from mixed gray target classes."
    elif run_status == "READY":
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
        "gray_target_consistency": gray_target_consistency,
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


def _sphere_detection_note(
    detection_source: str,
    detection_label: str,
    *,
    detection_failed: bool = False,
    sphere_detection_success: Optional[bool] = None,
    sphere_detection_unresolved: bool = False,
    fallback_used: bool = False,
) -> str:
    return _gray_target_note(
        target_class="sphere",
        detection_source=detection_source,
        detection_label=detection_label,
        detection_failed=detection_failed,
        fallback_used=fallback_used or _sphere_detection_uses_fallback(detection_source),
        sphere_detection_success=sphere_detection_success,
        sphere_detection_unresolved=sphere_detection_unresolved,
    )


def _measurement_region_origin(
    image_size: Tuple[int, int],
    calibration_roi: Optional[Dict[str, float]],
    *,
    measurement_region_scope: str = "calibration_roi",
) -> Tuple[int, int]:
    width, height = image_size
    if calibration_roi is None or str(measurement_region_scope or "calibration_roi") == "full_frame":
        return (0, 0)
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
    measurement_region_scope: str = "calibration_roi",
    sample_plausibility: str = "",
    detection_details: Optional[Dict[str, object]] = None,
    measurement_radius_px: Optional[float] = None,
    measurement_patch_radius_px: Optional[float] = None,
    measurement_patch_center: Optional[Dict[str, float]] = None,
) -> None:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size
    origin_x, origin_y = _measurement_region_origin((width, height), calibration_roi, measurement_region_scope=measurement_region_scope)
    if calibration_roi is None:
        region_width = width
        region_height = height
    else:
        region_width = max(1, int(np.ceil(float(calibration_roi["w"]) * width)))
        region_height = max(1, int(np.ceil(float(calibration_roi["h"]) * height)))
    draw.rectangle(
        (origin_x, origin_y, origin_x + region_width, origin_y + region_height),
        outline="#facc15",
        width=4,
    )
    detection_roi = dict(detected_sphere_roi or {})
    if not detection_roi and isinstance(detection_details, dict):
        detection_roi = dict(detection_details.get("roi") or {})
    if detection_roi:
        cx = origin_x + float(detection_roi.get("cx", 0.0) or 0.0)
        cy = origin_y + float(detection_roi.get("cy", 0.0) or 0.0)
        radius = float(detection_roi.get("r", 0.0) or 0.0)
        draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), outline="#22d3ee", width=5)
        measurement_radius = float(measurement_radius_px if measurement_radius_px is not None else _sphere_measurement_radius(radius))
        draw.ellipse((cx - measurement_radius, cy - measurement_radius, cx + measurement_radius, cy + measurement_radius), outline="#34d399", width=4)
        patch_center = dict(measurement_patch_center or {})
        patch_cx = origin_x + float(patch_center.get("x", float(detection_roi.get("cx", 0.0) or 0.0)) or 0.0)
        patch_cy = origin_y + float(patch_center.get("y", float(detection_roi.get("cy", 0.0) or 0.0)) or 0.0)
        patch_radius = float(measurement_patch_radius_px if measurement_patch_radius_px is not None else _sphere_hero_patch_radius(radius))
        draw.ellipse((patch_cx - patch_radius, patch_cy - patch_radius, patch_cx + patch_radius, patch_cy + patch_radius), outline="#a3e635", width=3)
        draw.line((patch_cx - 8, patch_cy, patch_cx + 8, patch_cy), fill="#a3e635", width=2)
        draw.line((patch_cx, patch_cy - 8, patch_cx, patch_cy + 8), fill="#a3e635", width=2)
    profile_probe = dict((detection_details or {}).get("profile_probe") or {})
    neutral_region = dict(profile_probe.get("neutral_region") or {})
    seed_bbox = dict(neutral_region.get("seed_bbox") or {})
    if seed_bbox:
        draw.rectangle(
            (
                origin_x + int(seed_bbox.get("x0", 0) or 0),
                origin_y + int(seed_bbox.get("y0", 0) or 0),
                origin_x + int(seed_bbox.get("x1", 0) or 0),
                origin_y + int(seed_bbox.get("y1", 0) or 0),
            ),
            outline="#f97316",
            width=3,
        )
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
    draw.rectangle((16, 16, min(width - 16, 980), 174), fill=(15, 23, 42))
    detection_status = _sphere_detection_status(
        bool((detection_details or {}).get("sphere_detection_success")),
        bool((detection_details or {}).get("sphere_detection_unresolved")),
    )
    rejection_reasons = _sphere_detection_rejection_reasons_from_details(detection_details)
    draw.text((28, 28), str(clip_id), fill="#f8fafc")
    draw.text((28, 52), f"Detection status: {detection_status}", fill="#e2e8f0")
    draw.text((28, 76), f"Detection method: {detection_source}", fill="#cbd5e1")
    draw.text((28, 100), f"Confidence: {detection_label} ({float(detection_confidence):.2f})", fill="#cbd5e1")
    draw.text((28, 124), _sphere_detection_note(detection_source, detection_label, detection_failed=(detection_label == 'FAILED')), fill="#e2e8f0")
    if str(sample_plausibility or "").strip():
        draw.text((28, 148), f"Sample plausibility: {sample_plausibility}", fill="#e2e8f0")
    if rejection_reasons:
        draw.text((360, 28), f"Rejection reasons: {', '.join(_sphere_detection_rejection_reason_label(item) for item in rejection_reasons[:3])}", fill="#fda4af")
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
        original_detection_success = bool(row.get("ipp2_original_sphere_detection_success", row.get("original_sphere_detection_success")))
        original_detection_unresolved = bool(row.get("ipp2_original_sphere_detection_unresolved", row.get("original_sphere_detection_unresolved")))
        original_detected_roi = dict(row.get("ipp2_original_detected_sphere_roi") or {})
        original_detection_details = dict(row.get("ipp2_original_detection_details") or {})
        original_measurement_region_scope = str(row.get("ipp2_original_measurement_region_scope") or "calibration_roi")
        original_sample_plausibility = str(row.get("ipp2_original_sample_plausibility") or "")
        original_measurement_radius_px = _numeric_value_or_none(row.get("ipp2_original_measurement_radius_px"))
        original_measurement_patch_radius_px = _numeric_value_or_none(row.get("ipp2_original_measurement_patch_radius_px"))
        original_measurement_patch_center = dict(row.get("ipp2_original_measurement_patch_center") or {})
        corrected_detection_source = str(row.get("ipp2_detection_source") or "failed")
        corrected_detection_label = str(row.get("ipp2_detection_label") or "FAILED")
        corrected_detection_confidence = float(row.get("ipp2_detection_confidence", 0.0) or 0.0)
        corrected_detection_success = bool(row.get("ipp2_sphere_detection_success"))
        corrected_detection_unresolved = bool(row.get("ipp2_sphere_detection_unresolved"))
        corrected_detected_roi = dict(row.get("ipp2_detected_sphere_roi") or {})
        corrected_detection_details = dict(row.get("ipp2_detection_details") or {})
        corrected_measurement_region_scope = str(row.get("ipp2_measurement_region_scope") or "calibration_roi")
        corrected_sample_plausibility = str(row.get("ipp2_sample_plausibility") or "")
        corrected_measurement_radius_px = _numeric_value_or_none(row.get("ipp2_measurement_radius_px"))
        corrected_measurement_patch_radius_px = _numeric_value_or_none(row.get("ipp2_measurement_patch_radius_px"))
        corrected_measurement_patch_center = dict(row.get("ipp2_measurement_patch_center") or {})
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
                measurement_region_scope=original_measurement_region_scope,
                sample_plausibility=original_sample_plausibility,
                detection_details=original_detection_details,
                measurement_radius_px=original_measurement_radius_px,
                measurement_patch_radius_px=original_measurement_patch_radius_px,
                measurement_patch_center=original_measurement_patch_center,
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
                measurement_region_scope=corrected_measurement_region_scope,
                sample_plausibility=corrected_sample_plausibility,
                detection_details=corrected_detection_details,
                measurement_radius_px=corrected_measurement_radius_px,
                measurement_patch_radius_px=corrected_measurement_patch_radius_px,
                measurement_patch_center=corrected_measurement_patch_center,
            )
        corrected_rejection_reasons = _sphere_detection_rejection_reasons_from_details(corrected_detection_details)
        summary_rows.append(
            {
                "clip_id": clip_id,
                "camera_id": str(row.get("camera_id") or ""),
                "source": corrected_detection_source or original_detection_source,
                "confidence": corrected_detection_confidence or original_detection_confidence,
                "confidence_label": corrected_detection_label or original_detection_label,
                "original_detection_source": original_detection_source,
                "original_detection_confidence": original_detection_confidence,
                "original_detection_label": original_detection_label,
                "original_detection_success": original_detection_success,
                "original_detection_unresolved": original_detection_unresolved,
                "original_detected_sphere_roi": original_detected_roi,
                "corrected_detection_source": corrected_detection_source,
                "corrected_detection_confidence": corrected_detection_confidence,
                "corrected_detection_label": corrected_detection_label,
                "corrected_detection_success": corrected_detection_success,
                "corrected_detection_unresolved": corrected_detection_unresolved,
                "corrected_detected_sphere_roi": corrected_detected_roi or original_detected_roi,
                "corrected_frame_reused_original_roi": corrected_detection_source == "reused_from_original",
                "fallback_used": (
                    (original_detection_success and _sphere_detection_uses_fallback(original_detection_source))
                    or (
                        corrected_detection_success
                        and _sphere_detection_uses_fallback(
                            corrected_detection_source,
                            reused_from_original=corrected_detection_source == "reused_from_original",
                        )
                    )
                ),
                "detection_failed": bool(row.get("detection_failed")) or (not corrected_detection_success and not corrected_detection_unresolved),
                "review_recommended": bool(row.get("detection_failed")) or corrected_detection_unresolved or str(corrected_detection_label or original_detection_label) in {"LOW", "FAILED"},
                "target_class": str(row.get("ipp2_target_class") or "sphere"),
                "measurement_region_scope": corrected_measurement_region_scope or original_measurement_region_scope,
                "sample_plausibility": corrected_sample_plausibility or original_sample_plausibility,
                "detection_status": _sphere_detection_status(corrected_detection_success, corrected_detection_unresolved),
                "detection_method": corrected_detection_source or original_detection_source,
                "rejection_reasons": corrected_rejection_reasons,
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
        measured_monitoring = _numeric_value_or_none(
            strategy_clip.get(
                "measured_log2_luminance_monitoring",
                diagnostics.get("measured_log2_luminance_monitoring", diagnostics.get("measured_log2_luminance")),
            )
        )
        measured_raw = _numeric_value_or_none(
            strategy_clip.get("measured_log2_luminance_raw", diagnostics.get("measured_log2_luminance_raw", measured_monitoring))
        )
        final_offset = float(strategy_clip.get("exposure_offset_stops", 0.0) or 0.0)
        target_log2 = float(recommended_payload.get("target_log2_luminance", 0.0) or 0.0)
        trusted_row = trusted_candidates.get(clip_id)
        screened_row = screened_candidates.get(clip_id)
        should_be_trusted = (
            confidence >= OPTIMAL_EXPOSURE_MIN_CONFIDENCE
            and sample_spread <= OPTIMAL_EXPOSURE_MAX_SAMPLE_LOG2_SPREAD
            and sample_chroma_spread <= OPTIMAL_EXPOSURE_MAX_SAMPLE_CHROMA_SPREAD
            and measured_monitoring is not None
            and abs(float(measured_monitoring) - float(exposure_summary.get("median", 0.0) or 0.0)) <= float(exposure_summary.get("outlier_threshold", 0.35) or 0.35)
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
                "deviation_from_anchor": (None if measured_monitoring is None else round(float(measured_monitoring) - target_log2, 6)),
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

    brightness_order = sorted(
        rows,
        key=lambda item: (
            float("-inf")
            if item["final_gray_value_used"]["monitoring_log2"] is None
            else float(item["final_gray_value_used"]["monitoring_log2"])
        ),
        reverse=True,
    )
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
        profile_note = str(row.get("trust_reason") or "Sample profile aligned")
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
        f"Color preview: {html.escape(str(payload.get('color_preview_operator_status') or payload.get('color_preview_status')))} | {html.escape(str(payload.get('color_preview_note')))}"
        if payload.get("color_preview_note")
        else f"Color preview: {html.escape(str(payload.get('color_preview_operator_status') or payload.get('color_preview_status') or 'unknown'))}"
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
    preview_still_format: str = DEFAULT_PREVIEW_STILL_FORMAT,
    artifact_mode: str = DEFAULT_ARTIFACT_MODE,
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
    resolved_artifact_mode = normalize_artifact_mode(artifact_mode)
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
    measurement_preview_settings["preview_still_format"] = normalize_preview_still_format(preview_still_format)
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
            artifact_mode=resolved_artifact_mode,
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
                "sample_1_ire": _numeric_value_or_none(strategy_clip.get("sample_1_ire", strategy_clip.get("bright_ire"))),
                "sample_2_ire": _numeric_value_or_none(strategy_clip.get("sample_2_ire", strategy_clip.get("center_ire"))),
                "sample_3_ire": _numeric_value_or_none(strategy_clip.get("sample_3_ire", strategy_clip.get("dark_ire"))),
                "monitoring_measurement_source": str(strategy_clip.get("monitoring_measurement_source") or ""),
                "raw_offset_stops": float(record.get("raw_offset_stops", 0.0) or 0.0),
                "final_offset_stops": float(strategy_clip["exposure_offset_stops"]),
                "camera_offset_from_anchor": float(strategy_clip.get("camera_offset_from_anchor", strategy_clip["exposure_offset_stops"]) or 0.0),
                "derived_display_scalar_log2": _numeric_value_or_none(
                    strategy_clip.get(
                        "display_scalar_log2",
                        strategy_clip.get("measured_log2_luminance_monitoring", strategy_clip.get("measured_log2_luminance")),
                    )
                ),
                "derived_exposure_value": _numeric_value_or_none(
                    strategy_clip.get(
                        "display_scalar_log2",
                        strategy_clip.get("measured_log2_luminance_monitoring", strategy_clip.get("measured_log2_luminance")),
                    )
                ),
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
        "color_preview_operator_status": _color_preview_policy()["operator_status"],
        "color_preview_note": _color_preview_policy()["note"],
        "executive_synopsis": synopsis,
        "subset_label": subset_label,
        "exposure_summary": exposure_summary,
        "strategy_comparison": strategy_summaries,
        "white_balance_model": dict(recommended_payload.get("white_balance_model", {})),
        "recommended_strategy": recommended_strategy,
        "run_assessment": run_assessment,
        "gray_target_consistency": dict(run_assessment.get("gray_target_consistency") or {}),
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
            "strategy_chart_svg": _build_strategy_chart_svg(strategy_summaries) if _strategy_chart_is_informative(strategy_summaries) else "",
            "trust_chart_svg": _build_trust_chart_svg(per_camera_rows) if _trust_chart_is_informative(per_camera_rows) else "",
        },
    }
    payload["debug_exposure_trace"] = (
        _build_exposure_trace_artifacts(
            analysis_records=analysis_records,
            recommended_payload=recommended_payload,
            strategy_summaries=strategy_summaries,
            exposure_summary=exposure_summary,
            out_root=out_root,
        )
        if resolved_artifact_mode == "debug"
        else None
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
        str(int(preview_settings.get("preview_still_redline_format_code", PREVIEW_STILL_FORMAT_CODES[DEFAULT_PREVIEW_STILL_FORMAT]))),
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


def _remove_stale_render_targets(output_path: str | Path) -> None:
    candidate = Path(output_path).expanduser().resolve()
    stale_paths = {candidate}
    for path in candidate.parent.glob(f"{candidate.name}.*"):
        stale_paths.add(path.resolve())
    for path in stale_paths:
        try:
            if path.exists() and path.is_file():
                path.unlink()
        except OSError:
            continue


def _validate_rendered_preview_output(path: Path) -> Dict[str, object]:
    min_size_bytes = _minimum_render_output_size_bytes(path)
    _wait_for_file_ready(path, max_attempts=6, delay_seconds=0.20, min_size_bytes=min_size_bytes)
    size_bytes = int(path.stat().st_size)
    if size_bytes < min_size_bytes:
        raise OSError(
            f"Rendered output looks like an invalid render stub ({size_bytes} bytes < {min_size_bytes} bytes): {path}"
        )
    _, image_metadata = _load_preview_image_as_normalized_rgb(path)
    width = int(image_metadata.get("width", 0) or 0)
    height = int(image_metadata.get("height", 0) or 0)
    if width <= 0 or height <= 0:
        raise OSError(f"Rendered output decoded without valid dimensions: {path}")
    return {
        "output_path": str(path),
        "size_bytes": size_bytes,
        "image_metadata": image_metadata,
    }


def _summarize_preview_render_attempts(attempts: List[Dict[str, object]]) -> str:
    parts: List[str] = []
    for item in attempts:
        summary_bits = [
            f"attempt={int(item.get('attempt', 0) or 0)}",
            f"returncode={int(item.get('returncode', 0) or 0)}",
            f"reason={str(item.get('reason') or 'unknown')}",
        ]
        output_path = str(item.get("resolved_output_path") or "")
        if output_path:
            summary_bits.append(f"output={output_path}")
        size_bytes = item.get("output_size_bytes")
        if size_bytes is not None:
            summary_bits.append(f"size={size_bytes}")
        error_text = str(item.get("error") or "").strip()
        if error_text:
            summary_bits.append(f"error={error_text}")
        parts.append("[" + ", ".join(summary_bits) + "]")
    return " ".join(parts)


def _render_preview_frame_with_retries(
    input_r3d: str,
    output_path: str,
    *,
    clip_id: str,
    variant: str,
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
    max_attempts: int = 3,
) -> Dict[str, object]:
    attempts: List[Dict[str, object]] = []
    requested_output = Path(output_path).expanduser().resolve()
    min_size_bytes = _minimum_render_output_size_bytes(requested_output)
    for attempt in range(1, max_attempts + 1):
        _remove_stale_render_targets(requested_output)
        rendered = render_preview_frame(
            input_r3d,
            str(requested_output),
            frame_index=frame_index,
            redline_executable=redline_executable,
            redline_capabilities=redline_capabilities,
            preview_settings=preview_settings,
            use_as_shot_metadata=use_as_shot_metadata,
            exposure=exposure,
            kelvin=kelvin,
            tint=tint,
            red_gain=red_gain,
            green_gain=green_gain,
            blue_gain=blue_gain,
            color_cdl=color_cdl,
            rmd_path=rmd_path,
            use_rmd_mode=use_rmd_mode,
            color_method=color_method,
        )
        resolved_output = Path(str(rendered.get("output_path") or requested_output)).expanduser().resolve()
        attempt_record: Dict[str, object] = {
            "attempt": attempt,
            "clip_id": clip_id,
            "variant": variant,
            "command": rendered.get("command"),
            "returncode": int(rendered.get("returncode", 0) or 0),
            "stdout": str(rendered.get("stdout") or ""),
            "stderr": str(rendered.get("stderr") or ""),
            "requested_output_path": str(requested_output),
            "resolved_output_path": str(resolved_output),
            "output_exists": bool(resolved_output.exists()),
            "minimum_size_bytes": int(min_size_bytes),
            "reason": "",
        }
        if resolved_output.exists():
            attempt_record["first_observed_size_bytes"] = int(resolved_output.stat().st_size)
        try:
            if int(rendered.get("returncode", 0) or 0) != 0:
                raise RuntimeError(
                    f"REDLine exited with code {int(rendered.get('returncode', 0) or 0)}. "
                    f"STDERR: {str(rendered.get('stderr') or '').strip()}"
                )
            output_validation = _validate_rendered_preview_output(resolved_output)
            attempt_record["output_exists"] = True
            attempt_record["output_size_bytes"] = int(output_validation["size_bytes"])
            attempt_record["reason"] = "success"
            attempts.append(attempt_record)
            rendered["output_path"] = str(resolved_output)
            rendered["attempt_count"] = attempt
            rendered["recovered_after_retry"] = attempt > 1
            rendered["attempt_diagnostics"] = attempts
            rendered["output_size_bytes"] = int(output_validation["size_bytes"])
            rendered["image_metadata"] = output_validation["image_metadata"]
            if attempt > 1:
                LOGGER.warning(
                    "Recovered REDLine preview render for %s (%s) after %s attempts",
                    clip_id,
                    variant,
                    attempt,
                )
                print(
                    f"[r3dmatch] recovered preview render clip={clip_id} variant={variant} attempts={attempt} output={resolved_output}"
                )
            return rendered
        except Exception as exc:
            if (
                int(rendered.get("returncode", 0) or 0) == 0
                and resolved_output.exists()
                and _is_retryable_preview_load_error(exc)
            ):
                settle_error: BaseException = exc
                settle_sizes: List[int] = []
                for settle_attempt in range(1, 4):
                    time.sleep(0.5 * settle_attempt)
                    if resolved_output.exists():
                        settle_sizes.append(int(resolved_output.stat().st_size))
                    try:
                        output_validation = _validate_rendered_preview_output(resolved_output)
                        attempt_record["output_exists"] = True
                        attempt_record["output_size_bytes"] = int(output_validation["size_bytes"])
                        attempt_record["reason"] = "success_after_settle"
                        if settle_sizes:
                            attempt_record["settle_size_bytes"] = list(settle_sizes)
                        attempts.append(attempt_record)
                        rendered["output_path"] = str(resolved_output)
                        rendered["attempt_count"] = attempt
                        rendered["recovered_after_retry"] = attempt > 1
                        rendered["attempt_diagnostics"] = attempts
                        rendered["output_size_bytes"] = int(output_validation["size_bytes"])
                        rendered["image_metadata"] = output_validation["image_metadata"]
                        rendered["settle_attempts"] = settle_attempt
                        return rendered
                    except Exception as settle_exc:
                        settle_error = settle_exc
                        if not _is_retryable_preview_load_error(settle_exc):
                            break
                if settle_sizes:
                    attempt_record["settle_size_bytes"] = list(settle_sizes)
                exc = settle_error
            attempt_record["output_exists"] = bool(resolved_output.exists())
            attempt_record["output_size_bytes"] = int(resolved_output.stat().st_size) if resolved_output.exists() else 0
            attempt_record["reason"] = (
                "nonzero_returncode"
                if int(rendered.get("returncode", 0) or 0) != 0
                else "output_missing_or_unreadable"
            )
            attempt_record["error"] = str(exc)
            attempts.append(attempt_record)
            if attempt >= max_attempts:
                raise RuntimeError(
                    f"REDLine preview render failed for {clip_id} ({variant}) after {max_attempts} attempt(s). "
                    f"Command: {shlex.join(list(rendered.get('command') or []))}. "
                    f"Attempt diagnostics: {_summarize_preview_render_attempts(attempts)}"
                ) from exc
            LOGGER.warning(
                "Retrying REDLine preview render for %s (%s), attempt %s/%s: %s",
                clip_id,
                variant,
                attempt,
                max_attempts,
                exc,
            )
            print(
                f"[r3dmatch] preview render retry clip={clip_id} variant={variant} attempt={attempt}/{max_attempts} reason={exc}"
            )
            time.sleep(0.25 * attempt)
    raise RuntimeError(f"Preview render retry loop exhausted unexpectedly for {clip_id} ({variant})")


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
    artifact_mode: str = DEFAULT_ARTIFACT_MODE,
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
    resolved_artifact_mode = normalize_artifact_mode(artifact_mode)
    render_settings = _normalize_preview_settings(
        preview_mode=str(preview_settings.get("preview_mode") or DEFAULT_DISPLAY_REVIEW_PREVIEW["preview_mode"]),
        preview_output_space=str(preview_settings.get("output_space") or DEFAULT_DISPLAY_REVIEW_PREVIEW["output_space"]),
        preview_output_gamma=str(preview_settings.get("output_gamma") or DEFAULT_DISPLAY_REVIEW_PREVIEW["output_gamma"]),
        preview_highlight_rolloff=str(preview_settings.get("highlight_rolloff") or DEFAULT_DISPLAY_REVIEW_PREVIEW["highlight_rolloff"]),
        preview_shadow_rolloff=str(preview_settings.get("shadow_rolloff") or DEFAULT_DISPLAY_REVIEW_PREVIEW["shadow_rolloff"]),
        preview_lut=str(preview_settings.get("lut_path")) if preview_settings.get("lut_path") else None,
        preview_still_format=str(preview_settings.get("preview_still_format") or DEFAULT_DISPLAY_REVIEW_PREVIEW.get("preview_still_format") or DEFAULT_PREVIEW_STILL_FORMAT),
    )
    preview_extension = str(render_settings.get("preview_still_extension") or preview_still_extension(DEFAULT_PREVIEW_STILL_FORMAT))

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

        original_path = preview_root / preview_filename_for_clip_id(clip_id, "original", run_id=run_id, extension=preview_extension)
        if render_originals:
            baseline_render = _render_preview_frame_with_retries(
                source_path,
                str(original_path),
                clip_id=clip_id,
                variant="original",
                frame_index=frame_index,
                redline_executable=redline_executable,
                redline_capabilities=redline_capabilities,
                preview_settings=render_settings,
                use_as_shot_metadata=True,
            )
            original_path = Path(str(baseline_render["output_path"]))
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
                    "attempt_count": int(baseline_render.get("attempt_count", 1) or 1),
                    "recovered_after_retry": bool(baseline_render.get("recovered_after_retry")),
                    "attempt_diagnostics": list(baseline_render.get("attempt_diagnostics") or []),
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
            variants = (
                {
                    "exposure": {"exposure": exposure_stops, "gains": None},
                    "color": {"exposure": 0.0, "gains": rgb_gains},
                    "both": {"exposure": exposure_stops, "gains": rgb_gains},
                }
                if resolved_artifact_mode == "debug"
                else {
                    "both": {"exposure": exposure_stops, "gains": rgb_gains},
                }
            )
            preview_paths[clip_id]["strategies"][strategy_key] = {}
            for variant, variant_settings in variants.items():
                raise_if_cancelled("Run cancelled while rendering corrected previews.")
                preview_path = preview_root / preview_filename_for_clip_id(clip_id, variant, strategy=strategy_key, run_id=run_id, extension=preview_extension)
                look_metadata_path = None
                rmd_metadata = None
                if strategy_rmd_root is not None and resolved_artifact_mode == "debug":
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
                color_preview_disabled = variant == "color" and not bool(color_preview_policy["enabled"])
                preview_color_applied = variant in {"color", "both"} and bool(color_preview_policy["enabled"])
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
                    source_preview_path = Path(source_preview_path).expanduser().resolve()
                    try:
                        if preview_path.exists() or preview_path.is_symlink():
                            preview_path.unlink()
                        os.link(source_preview_path, preview_path)
                        preview_path = preview_path.resolve()
                    except OSError:
                        try:
                            if preview_path.exists() or preview_path.is_symlink():
                                preview_path.unlink()
                            preview_path.symlink_to(source_preview_path)
                            preview_path = preview_path.resolve()
                        except OSError:
                            preview_path = source_preview_path
                else:
                    corrected_render = _render_preview_frame_with_retries(
                        source_path,
                        str(preview_path),
                        clip_id=clip_id,
                        variant=variant,
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
                    preview_path = Path(str(corrected_render["output_path"]))
                _validate_rendered_preview_output(preview_path)

                diff_metrics = _compute_image_difference_metrics(original_path, preview_path)
                mean_diff = diff_metrics["mean_absolute_difference"]
                correction_payload_identity = bool(
                    abs(float(variant_settings["exposure"])) <= 1e-6
                    and _is_identity_rgb_gains(variant_settings["gains"])
                    and _is_identity_cdl_payload(color_cdl)
                )
                measurement_valid = bool(
                    clip_entry.get(
                        "measurement_valid",
                        clip_entry.get("gray_target_measurement_valid", True),
                    )
                )
                requires_change = abs(float(variant_settings["exposure"])) > VISIBLE_PREVIEW_EXPOSURE_DELTA_STOPS or (
                    variant_settings["gains"] is not None
                    and any(abs(float(value) - 1.0) > 1e-6 for value in variant_settings["gains"])
                ) or not _is_identity_cdl_payload(color_cdl)
                error_message = None
                if (
                    measurement_valid
                    and requires_change
                    and not color_preview_disabled
                    and (mean_diff is None or float(mean_diff) < 1e-3)
                ):
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
                    "preview_color_applied": preview_color_applied,
                    "measurement_valid": measurement_valid,
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
                        "attempt_count": int(corrected_render.get("attempt_count", 1) or 1) if corrected_render else 1,
                        "recovered_after_retry": bool(corrected_render.get("recovered_after_retry")) if corrected_render else False,
                        "attempt_diagnostics": list(corrected_render.get("attempt_diagnostics") or []) if corrected_render else [],
                        "correction_payload_identity": correction_payload_identity,
                        "cdl_enabled": cdl_enabled,
                        "preview_color_applied": preview_color_applied,
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
        "preview_still_format": str(render_settings.get("preview_still_format") or DEFAULT_PREVIEW_STILL_FORMAT),
        "preview_still_extension": preview_extension,
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
    if resolved_artifact_mode == "debug":
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
    preview_still_format: str = DEFAULT_PREVIEW_STILL_FORMAT,
    report_focus: str = "auto",
    artifact_mode: str = DEFAULT_ARTIFACT_MODE,
    clear_cache: bool = True,
    calibration_roi: Optional[Dict[str, float]] = None,
    target_strategies: Optional[List[str]] = None,
    reference_clip_id: Optional[str] = None,
    hero_clip_id: Optional[str] = None,
    exposure_anchor_mode: Optional[str] = None,
    manual_target_stops: Optional[float] = None,
    manual_target_ire: Optional[float] = None,
    require_real_redline: bool = False,
    focus_validation: bool = False,
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
    resolved_report_focus = normalize_report_focus(report_focus)
    resolved_artifact_mode = normalize_artifact_mode(artifact_mode)
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
            artifact_mode=resolved_artifact_mode,
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
        preview_still_format=preview_still_format,
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
                artifact_mode=resolved_artifact_mode,
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
        artifact_mode=resolved_artifact_mode,
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
        artifact_mode=resolved_artifact_mode,
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
                "measurement_valid": bool(measurement.get("measurement_valid")),
                "gray_target_measurement_valid": bool(measurement.get("measurement_valid")),
                "display_scalar_log2": _numeric_value_or_none(measurement.get("display_scalar_log2", measurement["log2_luminance"])),
                "display_scalar_domain": "display_ipp2" if resolved_matching_domain == "perceptual" else "scene_analysis",
                "measured_log2_luminance": measurement["log2_luminance"],
                "measured_rgb_chromaticity": [float(value) for value in list(measurement.get("measured_rgb_chromaticity") or [])[:3]],
                "measured_log2_luminance_monitoring": monitoring_measurements_by_clip.get(clip_id, {}).get(
                    "measured_log2_luminance_monitoring",
                    record.get("diagnostics", {}).get("measured_log2_luminance_monitoring"),
                ),
                "measured_log2_luminance_raw": record.get("diagnostics", {}).get("measured_log2_luminance_raw"),
                "gray_exposure_summary": str(record.get("diagnostics", {}).get("gray_exposure_summary") or record.get("diagnostics", {}).get("aggregate_sphere_profile") or "n/a"),
                "sample_1_ire": _numeric_value_or_none(record.get("diagnostics", {}).get("sample_1_ire", record.get("diagnostics", {}).get("bright_ire"))),
                "sample_2_ire": _numeric_value_or_none(record.get("diagnostics", {}).get("sample_2_ire", record.get("diagnostics", {}).get("center_ire"))),
                "sample_3_ire": _numeric_value_or_none(record.get("diagnostics", {}).get("sample_3_ire", record.get("diagnostics", {}).get("dark_ire"))),
                "top_ire": _numeric_value_or_none(record.get("diagnostics", {}).get("top_ire")),
                "mid_ire": _numeric_value_or_none(record.get("diagnostics", {}).get("mid_ire")),
                "bottom_ire": _numeric_value_or_none(record.get("diagnostics", {}).get("bottom_ire")),
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
            diagnostics = dict(record.get("diagnostics", {}) or {})
            gray_target_detection_method = str(
                strategy_clip.get("gray_target_detection_method")
                or diagnostics.get("gray_target_detection_method")
                or ""
            )
            gray_target_fallback_used = bool(
                strategy_clip.get("gray_target_fallback_used", diagnostics.get("gray_target_fallback_used", False))
            )
            manual_assist_used = bool(
                strategy_clip.get("manual_assist_used", diagnostics.get("manual_assist_used", False))
            ) or gray_target_detection_method.startswith("manual_")
            measurement_valid = bool(
                strategy_clip.get(
                    "measurement_valid",
                    diagnostics.get("measurement_valid", diagnostics.get("gray_target_measurement_valid", False)),
                )
            )
            if measurement_valid:
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
            else:
                trust_details = {
                    "trust_class": "EXCLUDED",
                    "trust_score": 0.0,
                    "trust_reason": "Gray-sphere measurement unresolved",
                    "stability_label": "Measurement unresolved",
                    "correction_confidence": "LOW",
                    "reference_use": "Excluded",
                    "screened_reasons": ["measurement invalid"],
                    "outside_central_cluster": True,
                    "outside_primary_cluster": True,
                }
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
                    "measurement_valid": measurement_valid,
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
                            "measurement_valid": measurement_valid,
                            "initial_offset_stops": strategy_clip.get("initial_exposure_offset_stops", strategy_clip["exposure_offset_stops"]),
                            "final_offset_stops": strategy_clip["exposure_offset_stops"],
                            "pre_residual_stops": strategy_clip.get("pre_exposure_residual_stops"),
                            "post_residual_stops": strategy_clip.get("post_exposure_residual_stops"),
                            "measured_log2_luminance": strategy_clip["measured_log2_luminance"],
                            "measured_log2_luminance_monitoring": strategy_clip["measured_log2_luminance_monitoring"],
                            "measured_log2_luminance_raw": strategy_clip["measured_log2_luminance_raw"],
                            "gray_exposure_summary": str(strategy_clip.get("gray_exposure_summary") or strategy_clip.get("aggregate_sphere_profile") or "n/a"),
                            "sample_1_ire": _numeric_value_or_none(strategy_clip.get("sample_1_ire", strategy_clip.get("bright_ire"))),
                            "sample_2_ire": _numeric_value_or_none(strategy_clip.get("sample_2_ire", strategy_clip.get("center_ire"))),
                            "sample_3_ire": _numeric_value_or_none(strategy_clip.get("sample_3_ire", strategy_clip.get("dark_ire"))),
                            "top_ire": _numeric_value_or_none(strategy_clip.get("top_ire")),
                            "mid_ire": _numeric_value_or_none(strategy_clip.get("mid_ire")),
                            "bottom_ire": _numeric_value_or_none(strategy_clip.get("bottom_ire")),
                            "zone_spread_ire": _numeric_value_or_none(strategy_clip.get("zone_spread_ire")),
                            "zone_spread_stops": _numeric_value_or_none(strategy_clip.get("zone_spread_stops")),
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
                    "gray_target_class": str(
                        strategy_clip.get("gray_target_class")
                        or diagnostics.get("gray_target_class")
                        or "sphere"
                    ),
                    "gray_target_detection_method": gray_target_detection_method,
                    "gray_target_confidence": float(
                        strategy_clip.get("gray_target_confidence", diagnostics.get("gray_target_confidence", strategy_clip["confidence"]))
                        or 0.0
                    ),
                    "gray_target_fallback_used": gray_target_fallback_used,
                    "sample_plausibility": str(
                        strategy_clip.get("sample_plausibility")
                        or diagnostics.get("sample_plausibility")
                        or ""
                    ),
                    "manual_assist_used": manual_assist_used,
                    "neutral_sample_count": strategy_clip.get("neutral_sample_count"),
                    "neutral_sample_log2_spread": strategy_clip.get("neutral_sample_log2_spread"),
                    "neutral_sample_chromaticity_spread": strategy_clip.get("neutral_sample_chromaticity_spread"),
                    "neutral_samples": strategy_clip.get("neutral_samples"),
                    "commit_values": strategy_clip.get("commit_values"),
                        "preview_transform": _preview_transform_label(display_preview_settings),
                        "color_preview_applied": bool(color_preview_policy["enabled"]),
                        "color_preview_status": str(color_preview_policy["status"]),
                        "color_preview_operator_status": str(color_preview_policy["operator_status"]),
                        "color_preview_note": color_preview_policy["note"],
                        "is_hero_camera": strategy_clip["is_hero_camera"],
                    },
                    "clip_metadata": record.get("clip_metadata"),
                    "confidence": strategy_clip.get("confidence"),
                    "measured_rgb_chromaticity": [float(value) for value in list(strategy_clip.get("measured_rgb_chromaticity") or [])[:3]],
                    "gray_target_class": str(
                        strategy_clip.get("gray_target_class")
                        or diagnostics.get("gray_target_class")
                        or "sphere"
                    ),
                    "gray_target_detection_method": gray_target_detection_method,
                    "gray_target_confidence": float(
                        strategy_clip.get("gray_target_confidence", diagnostics.get("gray_target_confidence", strategy_clip.get("confidence", 0.0)))
                        or 0.0
                    ),
                    "gray_target_fallback_used": gray_target_fallback_used,
                    "sample_plausibility": str(
                        strategy_clip.get("sample_plausibility")
                        or diagnostics.get("sample_plausibility")
                        or ""
                    ),
                    "manual_assist_used": manual_assist_used,
                    "neutral_sample_count": strategy_clip.get("neutral_sample_count"),
                    "neutral_sample_log2_spread": strategy_clip.get("neutral_sample_log2_spread"),
                    "neutral_sample_chromaticity_spread": strategy_clip.get("neutral_sample_chromaticity_spread"),
                    "pre_color_residual": strategy_clip.get("pre_color_residual"),
                    "post_color_residual": strategy_clip.get("post_color_residual"),
                    "white_balance_model_label": strategy_clip.get("white_balance_model_label"),
                    "white_balance_model": strategy_clip.get("white_balance_model"),
                    "commit_values": copy.deepcopy(strategy_clip.get("commit_values") or {}),
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
                "white_balance_model": dict(strategy_payload.get("white_balance_model") or {}),
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
    corrected_residual_validation = _build_corrected_residual_validation(
        strategies=strategies,
        out_root=out_root,
    )
    render_trace_artifacts = (
        _write_render_trace_artifacts(
            out_root=out_root,
            analysis_records=analysis_records,
            strategies=strategies,
            ipp2_validation_summary=ipp2_validation["summary"],
            preview_manifest_payload=preview_manifest_payload,
        )
        if resolved_artifact_mode == "debug"
        else {
            "render_input_state_path": "",
            "pre_render_log_values_path": "",
            "post_render_ipp2_values_path": "",
            "render_trace_comparison_path": "",
        }
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
            upgraded_trust = _upgrade_trust_from_ipp2_validation(
                {
                    "trust_class": clip.get("trust_class"),
                    "trust_score": clip.get("trust_score"),
                    "trust_reason": clip.get("trust_reason"),
                    "stability_label": clip.get("stability_label"),
                    "correction_confidence": clip.get("correction_confidence"),
                    "reference_use": clip.get("reference_use"),
                    "screened_reasons": clip.get("screened_reasons"),
                },
                ipp2_validation=dict(clip.get("ipp2_validation") or {}),
                gray_target_class=str(clip.get("gray_target_class") or ""),
                sample_plausibility=str(clip.get("sample_plausibility") or ""),
            )
            clip["trust_class"] = str(upgraded_trust.get("trust_class") or clip.get("trust_class") or "UNTRUSTED")
            clip["trust_score"] = float(upgraded_trust.get("trust_score", clip.get("trust_score", 0.0)) or 0.0)
            clip["trust_reason"] = str(upgraded_trust.get("trust_reason") or clip.get("trust_reason") or "")
            clip["stability_label"] = str(upgraded_trust.get("stability_label") or clip.get("stability_label") or "")
            clip["correction_confidence"] = str(upgraded_trust.get("correction_confidence") or clip.get("correction_confidence") or "")
            clip["reference_use"] = str(upgraded_trust.get("reference_use") or clip.get("reference_use") or "Included")
            clip["screened_reasons"] = list(upgraded_trust.get("screened_reasons") or clip.get("screened_reasons") or [])
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
        "color_preview_operator_status": str(color_preview_policy["operator_status"]),
        "color_preview_note": color_preview_policy["note"],
        "exposure_measurement_domain": resolved_matching_domain,
        "preview_mode": display_preview_settings["preview_mode"],
        "preview_still_format": str(display_preview_settings.get("preview_still_format") or DEFAULT_PREVIEW_STILL_FORMAT),
        "report_focus": resolved_report_focus,
        "report_focus_label": report_focus_label(resolved_report_focus),
        "artifact_mode": resolved_artifact_mode,
        "artifact_mode_label": artifact_mode_label(resolved_artifact_mode),
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
        "strategy_review_rmd_root": (
            str((root / "review_rmd" / "strategies").resolve())
            if resolved_artifact_mode == "debug"
            else ""
        ),
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
            "white_balance_model": dict(item.get("white_balance_model") or {}),
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
    payload["white_balance_model"] = dict((recommended_strategy_section or {}).get("white_balance_model") or {})
    payload["strategy_comparison"] = strategy_summaries
    payload["visuals"] = {
        "exposure_plot_svg": _build_exposure_plot_svg((recommended_strategy_section or {}).get("clips", [])),
        "strategy_chart_svg": _build_strategy_chart_svg(strategy_summaries) if _strategy_chart_is_informative(strategy_summaries) else "",
    }
    if strategy_payloads and resolved_artifact_mode == "debug":
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
    else:
        payload["debug_exposure_trace"] = None
    payload["executive_synopsis"] = _build_lightweight_synopsis(
        exposure_summary=payload["exposure_summary"],
        strategy_summaries=strategy_summaries,
        recommended_strategy=payload["recommended_strategy"] or {"reason": "No recommendation available."},
        hero_summary=payload["hero_recommendation"],
    ) if strategy_payloads else ""
    recommended_strategy_clips = [
        copy.deepcopy(item)
        for item in list((recommended_strategy_section or {}).get("clips") or [])
    ]
    payload["per_camera_analysis"] = recommended_strategy_clips
    focus_validation_artifacts = (
        _build_focus_validation_artifacts(
            out_root=out_root,
            per_camera_rows=recommended_strategy_clips,
            shared_originals=shared_originals,
            progress_path=progress_path,
        )
        if bool(focus_validation)
        else {
            "summary": {
                "enabled": False,
                "status": "disabled",
                "reason": "Focus validation is disabled for this run.",
                "rows": [],
                "tiff_is_sufficient": False,
                "confidence": "high",
            },
            "path": "",
            "overlay_root": "",
        }
    )
    payload["focus_validation"] = dict(focus_validation_artifacts.get("summary") or {})
    payload["focus_validation_path"] = str(focus_validation_artifacts.get("path") or "")
    payload["focus_validation_overlay_root"] = str(focus_validation_artifacts.get("overlay_root") or "")
    focus_rows_by_clip = {
        str(item.get("clip_id") or ""): dict(item)
        for item in list((payload.get("focus_validation") or {}).get("rows") or [])
        if str(item.get("clip_id") or "").strip()
    }
    for item in recommended_strategy_clips:
        clip_id = str(item.get("clip_id") or "")
        item["focus_validation"] = dict(focus_rows_by_clip.get(clip_id) or {})
    run_assessment = _build_run_assessment(
        per_camera_rows=recommended_strategy_clips,
        recommended_payload=payload["recommended_strategy"] or {},
        strategy_summaries=strategy_summaries,
        exposure_summary=payload["exposure_summary"],
    )
    payload["run_assessment"] = run_assessment
    payload["gray_target_consistency"] = dict(run_assessment.get("gray_target_consistency") or {})
    payload["operator_recommendation"] = str(run_assessment.get("operator_note") or "")
    scientific_validation = _build_scientific_validation_artifacts(
        out_root=out_root,
        analysis_records=analysis_records,
        payload=payload,
        fail_on_analysis_drift=True,
    )
    payload["scientific_validation"] = scientific_validation["summary"]
    payload["scientific_validation_path"] = scientific_validation["path"]
    payload["scientific_validation_markdown_path"] = scientific_validation["markdown_path"]
    json_path = out_root / "contact_sheet.json"
    html_path = out_root / "contact_sheet.html"
    debug_json_path = out_root / "contact_sheet_debug.json"
    pdf_path = out_root / "preview_contact_sheet.pdf"
    review_manifest_path = out_root / "review_manifest.json"
    try:
        raise_if_cancelled("Run cancelled before writing review report artifacts.")
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        payload["report_json"] = str(json_path)
        payload["report_html"] = str(html_path)
        payload["preview_report_pdf"] = str(pdf_path)
        raise_if_cancelled("Run cancelled before writing review HTML.")
        html_path.write_text(render_contact_sheet_html(payload, html_path=str(html_path)), encoding="utf-8")
        payload["contact_sheet_debug"] = str(debug_json_path)
        payload["html_asset_validation"] = _assert_contact_sheet_html_assets(str(html_path))
        payload["pdf_export_preflight"] = contact_sheet_pdf_export_preflight(str(html_path))
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        raise_if_cancelled("Run cancelled before rendering review PDF.")
        render_contact_sheet_pdf_from_html(str(html_path), output_path=str(pdf_path))
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
                    "preview_still_format": payload["preview_still_format"],
                    "report_focus": payload["report_focus"],
                    "report_focus_label": payload["report_focus_label"],
                    "artifact_mode": payload["artifact_mode"],
                    "artifact_mode_label": payload["artifact_mode_label"],
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
                    "scientific_validation_path": payload["scientific_validation_path"],
                    "scientific_validation_markdown_path": payload["scientific_validation_markdown_path"],
                    "report_json": str(json_path),
                    "report_html": str(html_path),
                    "contact_sheet_debug": str(debug_json_path),
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
        "preview_still_format": payload["preview_still_format"],
        "report_focus": payload["report_focus"],
        "report_focus_label": payload["report_focus_label"],
        "artifact_mode": payload["artifact_mode"],
        "artifact_mode_label": payload["artifact_mode_label"],
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
        "scientific_validation_path": payload["scientific_validation_path"],
        "scientific_validation_markdown_path": payload["scientific_validation_markdown_path"],
        "recommended_strategy": payload.get("recommended_strategy"),
        "hero_recommendation": payload.get("hero_recommendation"),
        "white_balance_model": payload.get("white_balance_model"),
        "run_assessment": payload.get("run_assessment"),
        "gray_target_consistency": payload.get("gray_target_consistency"),
        "operator_recommendation": payload.get("operator_recommendation"),
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
    preview_still_format: str = DEFAULT_PREVIEW_STILL_FORMAT,
    report_focus: str = "auto",
    artifact_mode: str = DEFAULT_ARTIFACT_MODE,
    calibration_roi: Optional[Dict[str, float]] = None,
    target_strategies: Optional[List[str]] = None,
    reference_clip_id: Optional[str] = None,
    hero_clip_id: Optional[str] = None,
    exposure_anchor_mode: Optional[str] = None,
    manual_target_stops: Optional[float] = None,
    manual_target_ire: Optional[float] = None,
    require_real_redline: bool = False,
    focus_validation: bool = False,
    progress_path: Optional[str] = None,
) -> Dict[str, object]:
    raise_if_cancelled("Run cancelled before review package assembly.")
    root = Path(out_dir).expanduser().resolve()
    report_dir = root / "report"
    review_rmd_dir = root / "review_rmd"
    resolved_artifact_mode = normalize_artifact_mode(artifact_mode)
    rcx_compare_note = _write_rcx_comparison_placeholder(root) if resolved_artifact_mode == "debug" else None
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
            preview_still_format=preview_still_format,
            artifact_mode=artifact_mode,
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
            preview_still_format=preview_still_format,
            report_focus=report_focus,
            artifact_mode=artifact_mode,
            calibration_roi=calibration_roi,
            target_strategies=target_strategies,
            reference_clip_id=reference_clip_id,
            hero_clip_id=hero_clip_id,
            exposure_anchor_mode=exposure_anchor_mode,
            manual_target_stops=manual_target_stops,
            manual_target_ire=manual_target_ire,
            clear_cache=True,
            require_real_redline=require_real_redline,
            focus_validation=focus_validation,
            progress_path=progress_path,
        )
    if resolved_review_mode == "lightweight_analysis":
        rmd_manifest = {
            "skipped": True,
            "reason": "Lightweight analysis does not require temporary review RMD authoring.",
            "review_rmd_dir": str(review_rmd_dir),
        }
    else:
        if resolved_artifact_mode == "debug":
            raise_if_cancelled("Run cancelled before writing temporary review RMDs.")
            rmd_manifest = write_rmds_from_analysis(out_dir, out_dir=str(review_rmd_dir))
        else:
            rmd_manifest = {
                "skipped": True,
                "reason": "Production mode skips temporary review RMD exports and relies on direct REDLine flags for the lean review preview path.",
                "review_rmd_dir": str(review_rmd_dir),
                "strategy_review_rmd_root": "",
            }
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
        "preview_still_format": normalize_preview_still_format(preview_still_format),
        "focus_validation_enabled": bool(focus_validation),
        "report_focus": normalize_report_focus(report_focus),
        "report_focus_label": report_focus_label(report_focus),
        "artifact_mode": resolved_artifact_mode,
        "artifact_mode_label": artifact_mode_label(artifact_mode),
        "artifact_policy": {
            "sidecars": {
                "default_mode": "production_required",
                "reason": "Generated sidecars remain the canonical per-camera analysis payload for reproducibility, validation, and downstream apply/transcode paths.",
            },
            "review_rmd": {
                "default_mode": "debug_only",
                "reason": "Temporary review RMD exports are reserved for debug/forensic runs. Production review uses direct REDLine flags and no longer writes strategy preview RMDs by default.",
            },
            "rmd_compare": {
                "default_mode": "debug_only",
                "reason": "RCX parity scratch space is diagnostic-only and no longer clutters default production runs.",
            },
        },
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
        "recommended_strategy": report_payload.get("recommended_strategy"),
        "hero_recommendation": report_payload.get("hero_recommendation"),
        "white_balance_model": report_payload.get("white_balance_model"),
        "run_assessment": report_payload.get("run_assessment"),
        "gray_target_consistency": report_payload.get("gray_target_consistency"),
        "operator_recommendation": report_payload.get("operator_recommendation"),
    }
    manifest_path = report_dir / "review_package.json"
    manifest_path.write_text(json.dumps(package_manifest, indent=2), encoding="utf-8")
    package_manifest["package_manifest"] = str(manifest_path)
    return package_manifest


def _contact_sheet_image_src(path: object, *, html_path: Optional[object] = None) -> str:
    path_text = str(path or "").strip()
    if not path_text:
        return ""
    path_obj = Path(path_text)
    if html_path:
        html_dir = Path(html_path).expanduser().resolve().parent
        try:
            return Path(os.path.relpath(path_obj.expanduser().resolve(), start=html_dir)).as_posix()
        except (OSError, ValueError):
            return path_obj.expanduser().resolve().as_uri()
    if "review_detection_overlays" in path_obj.parts:
        return f"./review_detection_overlays/{path_obj.name}"
    if "_measurement" in path_obj.parts:
        return f"../previews/_measurement/{path_obj.name}"
    if "_ipp2_closed_loop" in path_obj.parts:
        return f"../previews/_ipp2_closed_loop/{path_obj.name}"
    return f"../previews/{path_obj.name}"


_CONTACT_SHEET_IMG_SRC_RE = re.compile(r"<img\b[^>]*\bsrc=(['\"])(.*?)\1", re.IGNORECASE)


def _validate_contact_sheet_html_assets_markup(html_text: str, *, html_path: str) -> Dict[str, object]:
    html_file = Path(html_path).expanduser().resolve()
    resolved_assets: List[str] = []
    missing_assets: List[str] = []
    sources = [match[1] for match in _CONTACT_SHEET_IMG_SRC_RE.findall(html_text)]
    for src in sources:
        asset_path = _resolve_contact_sheet_html_asset(src, html_path=str(html_file))
        if asset_path is None:
            continue
        resolved_assets.append(str(asset_path))
        if not asset_path.exists():
            missing_assets.append(f"{src} -> {asset_path}")
    return {
        "html_path": str(html_file),
        "image_count": len(sources),
        "resolved_asset_count": len(resolved_assets),
        "resolved_assets": resolved_assets,
        "missing_assets": missing_assets,
        "all_assets_exist": not missing_assets,
    }


def _resolve_contact_sheet_html_asset(src: str, *, html_path: str) -> Optional[Path]:
    source = str(src or "").strip()
    if not source or source.startswith(("http://", "https://", "data:", "about:", "#")):
        return None
    if source.startswith("file://"):
        parsed = urlparse(source)
        return Path(unquote(parsed.path))
    source_path = Path(source)
    if source_path.is_absolute():
        return source_path
    return Path(html_path).expanduser().resolve().parent / source_path


def validate_contact_sheet_html_assets(html_path: str) -> Dict[str, object]:
    html_file = Path(html_path).expanduser().resolve()
    html_text = html_file.read_text(encoding="utf-8")
    return _validate_contact_sheet_html_assets_markup(html_text, html_path=str(html_file))


def _assert_contact_sheet_html_assets(html_path: str) -> Dict[str, object]:
    validation = validate_contact_sheet_html_assets(html_path)
    missing_assets = list(validation.get("missing_assets") or [])
    if missing_assets:
        preview = "; ".join(missing_assets[:6])
        if len(missing_assets) > 6:
            preview += f"; ... and {len(missing_assets) - 6} more"
        raise RuntimeError(
            "Generated contact_sheet.html references missing image assets. "
            f"HTML: {html_path}. Missing: {preview}"
        )
    return validation


def _contact_sheet_sanitized_stem(value: object) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "").strip()).strip("._")
    return text or "camera"


def _contact_sheet_resample_filter() -> int:
    resampling = getattr(Image, "Resampling", Image)
    return int(getattr(resampling, "LANCZOS"))


def _contact_sheet_write_report_image(source_path: str, *, output_path: Path) -> Dict[str, object]:
    source = Path(source_path).expanduser().resolve()
    if not source.exists():
        raise RuntimeError(f"Contact-sheet asset is missing: {source}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        converted = image.convert("RGB")
        width, height = converted.size
        if width <= 0 or height <= 0:
            raise RuntimeError(f"Contact-sheet asset has invalid size: {source}")
        target_width = min(width, 800)
        if width != target_width:
            target_height = max(1, int(round(height * (target_width / width))))
            converted = converted.resize((target_width, target_height), _contact_sheet_resample_filter())
        if converted.size[0] > 1200:
            raise RuntimeError(
                f"Contact-sheet report image exceeds the 1200px width limit after resize: {output_path} ({converted.size[0]}px)"
            )
        converted.save(output_path, format="JPEG", quality=82, optimize=True, progressive=True)
    return {
        "source_path": str(source),
        "output_path": str(output_path.resolve()),
        "width": int(converted.size[0]),
        "height": int(converted.size[1]),
    }


def _contact_sheet_collect_consistent_numeric(
    field_name: str,
    *,
    required: bool = False,
    tolerance: float = 1e-4,
    treat_zero_as_missing: bool = False,
    sources: List[Tuple[str, Dict[str, object]]],
) -> Tuple[Optional[float], str]:
    found: List[Tuple[str, float]] = []
    for source_name, source in sources:
        raw_value = source.get(field_name)
        if raw_value in (None, ""):
            continue
        try:
            value = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"Contact-sheet field {field_name} is not numeric in {source_name}: {raw_value!r}") from exc
        if treat_zero_as_missing and math.isclose(value, 0.0, abs_tol=tolerance):
            continue
        found.append((source_name, value))
    if required and not found:
        raise RuntimeError(f"Contact-sheet field {field_name} is missing from the stored measurement payload.")
    if len(found) > 1:
        baseline = found[0][1]
        for source_name, value in found[1:]:
            if not math.isclose(value, baseline, abs_tol=tolerance):
                source_summary = ", ".join(f"{name}={val:.6f}" for name, val in found)
                raise RuntimeError(
                    f"Stored measurement payload disagrees for {field_name}: {source_summary}. "
                    "Contact-sheet build stopped to avoid displaying inconsistent values."
                )
    return (found[0][1], found[0][0]) if found else (None, "")


def _contact_sheet_collect_sample_numeric(
    sample_field: str,
    *,
    required: bool = False,
    tolerance: float = 1e-4,
    sources: List[Tuple[str, Dict[str, object]]],
) -> Tuple[Optional[float], str]:
    alias_map = {
        "sample_1_ire": ("sample_1_ire", "bright_ire"),
        "sample_2_ire": ("sample_2_ire", "center_ire"),
        "sample_3_ire": ("sample_3_ire", "dark_ire"),
    }
    aliases = alias_map.get(sample_field, (sample_field,))
    for source_name, source in sources:
        for alias in aliases:
            raw_value = source.get(alias)
            if raw_value in (None, ""):
                continue
            try:
                candidate = float(raw_value)
            except (TypeError, ValueError) as exc:
                raise RuntimeError(f"Contact-sheet field {alias} is not numeric in {source_name}: {raw_value!r}") from exc
            if math.isclose(candidate, 0.0, abs_tol=tolerance):
                continue
            return candidate, f"{source_name}.{alias}"
    if required:
        raise RuntimeError(f"Contact-sheet field {sample_field} is missing from the stored measurement payload.")
    return None, ""


def _contact_sheet_resolve_measurement_fields(
    *,
    clip_id: str,
    clip_row: Dict[str, object],
    shared: Dict[str, object],
    exposure_metrics: Dict[str, object],
    ipp2_row: Dict[str, object],
) -> Dict[str, object]:
    measurement_valid = bool(
        shared.get(
            "measurement_valid",
            clip_row.get(
                "measurement_valid",
                exposure_metrics.get("measurement_valid", clip_row.get("gray_target_measurement_valid", False)),
            ),
        )
    )
    if not measurement_valid:
        return {
            "measurement_valid": False,
            "sample_1_ire": None,
            "sample_2_ire": None,
            "sample_3_ire": None,
            "sample_sources": {},
            "target_sample_label": "Gray target unresolved",
            "display_scalar_log2": None,
            "display_scalar_source": "measurement_unresolved",
        }
    # -------------------------------------------------------------------------
    # AUTHORITATIVE MEASUREMENT SOURCE: shared_original ONLY
    # -------------------------------------------------------------------------
    # The displayed "original" sample values MUST represent the true measurement
    # pipeline: R3D → REDLine (IPP2 render, useMeta) → JPEG → OpenCV sphere
    # detection → Masked pixel sampling → Median per region → Luminance → IRE
    # These values are stored in shared_original. NO FALLBACKS. NO EXCEPTIONS.
    # -------------------------------------------------------------------------

    # Extract sample IRE values from shared_original ONLY
    sample_aliases = {
        "sample_1_ire": ("sample_1_ire", "bright_ire"),
        "sample_2_ire": ("sample_2_ire", "center_ire"),
        "sample_3_ire": ("sample_3_ire", "dark_ire"),
    }

    def _extract_authoritative_sample(field_name: str) -> float:
        """Extract sample value from shared_original. Fail hard if missing or invalid."""
        aliases = sample_aliases.get(field_name, (field_name,))
        for alias in aliases:
            raw_value = shared.get(alias)
            if raw_value in (None, ""):
                continue
            try:
                value = float(raw_value)
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"Contact-sheet field {alias} is not numeric in shared_original for {clip_id}: {raw_value!r}"
                ) from exc
            return value
        raise RuntimeError(
            f"Contact-sheet field {field_name} is missing from shared_original for {clip_id}. "
            f"This is a data integrity error: the authoritative measurement source must contain valid sample values."
        )

    sample_1_ire = _extract_authoritative_sample("sample_1_ire")
    sample_2_ire = _extract_authoritative_sample("sample_2_ire")
    sample_3_ire = _extract_authoritative_sample("sample_3_ire")

    # -------------------------------------------------------------------------
    # SCALAR: Must be derived from shared_original ONLY
    # -------------------------------------------------------------------------
    scalar_value = None
    scalar_source = ""

    # First try display_scalar_log2 from shared_original
    raw_scalar = shared.get("display_scalar_log2")
    if raw_scalar not in (None, ""):
        try:
            scalar_value = float(raw_scalar)
            scalar_source = "shared_original.display_scalar_log2"
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"Contact-sheet scalar display_scalar_log2 is not numeric in shared_original for {clip_id}: {raw_scalar!r}"
            ) from exc

    # Fallback to measured_log2_luminance_monitoring from shared_original
    if scalar_value is None:
        raw_scalar = shared.get("measured_log2_luminance_monitoring")
        if raw_scalar not in (None, ""):
            try:
                scalar_value = float(raw_scalar)
                scalar_source = "shared_original.measured_log2_luminance_monitoring"
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"Contact-sheet scalar measured_log2_luminance_monitoring is not numeric in shared_original for {clip_id}: {raw_scalar!r}"
                ) from exc

    if scalar_value is None:
        raise RuntimeError(
            f"Contact-sheet scalar is missing from shared_original for {clip_id}. "
            f"This is a data integrity error: the authoritative measurement source must contain a valid scalar."
        )

    return {
        "measurement_valid": True,
        "sample_1_ire": sample_1_ire,
        "sample_2_ire": sample_2_ire,
        "sample_3_ire": sample_3_ire,
        "sample_sources": {
            "sample_1_ire": "shared_original",
            "sample_2_ire": "shared_original",
            "sample_3_ire": "shared_original",
        },
        "target_sample_label": CONTACT_SHEET_TARGET_SAMPLE_LABEL,
        "display_scalar_log2": scalar_value,
        "display_scalar_source": scalar_source,
    }


def _contact_sheet_required_asset(path_value: object, *, label: str, clip_id: str) -> str:
    path_text = str(path_value or "").strip()
    if not path_text:
        raise RuntimeError(f"Contact-sheet build requires {label} for {clip_id}, but no stored asset path was present.")
    resolved = Path(path_text).expanduser().resolve()
    if not resolved.exists():
        raise RuntimeError(f"Contact-sheet build requires {label} for {clip_id}, but the stored asset is missing: {resolved}")
    return str(resolved)


def contact_sheet_pdf_export_preflight(html_path: Optional[str] = None) -> Dict[str, object]:
    preflight: Dict[str, object] = {
        "interpreter": sys.executable,
        "dyld_fallback_library_path": str(os.environ.get("DYLD_FALLBACK_LIBRARY_PATH") or ""),
        "red_sdk_root": str(os.environ.get("RED_SDK_ROOT") or ""),
        "red_sdk_redistributable_dir": str(os.environ.get("RED_SDK_REDISTRIBUTABLE_DIR") or ""),
        "weasyprint_importable": False,
        "weasyprint_error": "",
        "asset_validation": None,
    }
    try:
        from weasyprint import HTML  # noqa: F401
    except Exception as exc:  # pragma: no cover - host dependency boundary
        preflight["weasyprint_error"] = str(exc)
        return preflight
    preflight["weasyprint_importable"] = True
    if html_path:
        preflight["asset_validation"] = validate_contact_sheet_html_assets(html_path)
    return preflight


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
    retained = [
        item
        for item in entries
        if bool(item.get("measurement_valid")) and str(item.get("reference_use") or "Included") != "Excluded"
    ]
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
    valid_entries = [item for item in entries if bool(item.get("measurement_valid"))]
    if not valid_entries:
        return False
    return all(float(item.get("residual_abs_stops", 0.0) or 0.0) <= goal_stops for item in valid_entries)


def _camera_layout_sort_key(label: str) -> tuple[str, str]:
    text = str(label or "").strip().upper()
    match = re.match(r"([A-Z]+)(\d+)?", text)
    if not match:
        return (text, "")
    prefix = str(match.group(1) or "")
    suffix = str(match.group(2) or "")
    suffix_key = f"{int(suffix):04d}" if suffix.isdigit() else ""
    return (prefix, suffix_key)


def _contact_sheet_attention_class(entry: Dict[str, object]) -> str:
    if not bool(entry.get("measurement_valid")):
        return "outlier"
    if str(entry.get("reference_use") or "Included") == "Excluded":
        return "outlier"
    if str(entry.get("trust_class") or "") in {"EXCLUDED", "UNTRUSTED"}:
        return "outlier"
    if float(entry.get("residual_abs_stops", 0.0) or 0.0) > IPP2_VALIDATION_REVIEW_STOPS:
        return "outlier"
    if bool(entry.get("fallback_used")):
        return "borderline"
    if float(entry.get("residual_abs_stops", 0.0) or 0.0) > IPP2_VALIDATION_PASS_STOPS:
        return "borderline"
    if str(entry.get("trust_class") or "") == "USE_WITH_CAUTION":
        return "borderline"
    return "safe"


def _contact_sheet_original_cast_label(amber_blue: float, green_magenta: float) -> str:
    temperature = "near-neutral"
    tint = "neutral"
    if amber_blue > 0.03:
        temperature = "cool / blue"
    elif amber_blue < -0.03:
        temperature = "warm / amber"
    if green_magenta > 0.02:
        tint = "green"
    elif green_magenta < -0.02:
        tint = "magenta"
    if temperature == "near-neutral" and tint == "neutral":
        return "Near neutral in the original still"
    if tint == "neutral":
        return f"{temperature.title()} bias"
    if temperature == "near-neutral":
        return f"{tint.title()} bias"
    return f"{temperature.title()} with {tint} bias"


def _contact_sheet_stability_summary(
    *,
    confidence: Optional[float],
    log2_spread: Optional[float],
    chroma_spread: Optional[float],
    sample_count: Optional[int],
) -> str:
    notes: List[str] = []
    if sample_count is not None and int(sample_count) > 0:
        notes.append(f"{int(sample_count)} neutral samples")
    if confidence is not None:
        notes.append(f"confidence {float(confidence):.2f}")
    if log2_spread is not None:
        notes.append(f"log2 spread {float(log2_spread):.3f}")
    if chroma_spread is not None:
        notes.append(f"chroma spread {float(chroma_spread):.4f}")
    return " | ".join(notes) if notes else "Stored original-still stability summary is not available."


def _contact_sheet_has_meaningful_text(value: object, *, blocked_prefixes: Tuple[str, ...] = ()) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    lowered = text.lower()
    if "not available" in lowered or "not stored" in lowered or "no stored" in lowered or lowered == "n/a":
        return False
    return not any(lowered.startswith(prefix.lower()) for prefix in blocked_prefixes)


def _contact_sheet_exposure_direction_label(offset_stops: float) -> str:
    offset = float(offset_stops or 0.0)
    if offset >= 0.05:
        return f"↑ Lift {offset:+.2f} stops"
    if offset <= -0.05:
        return f"↓ Lower {offset:+.2f} stops"
    return f"≈ Hold {offset:+.2f} stops"


def _contact_sheet_overview_status(entry: Dict[str, object]) -> Tuple[str, str]:
    if not bool(entry.get("measurement_valid")):
        return ("Measurement Unresolved", "danger")
    attention = str(entry.get("attention_class") or "safe")
    if str(entry.get("reference_use") or "Included") == "Excluded" or attention == "outlier":
        return ("Excluded / Outlier", "danger")
    if attention == "borderline":
        return ("Needs Attention", "warning")
    if bool(entry.get("is_anchor_reference")):
        return ("Exposure Anchor", "info")
    return ("Retained", "good")


def _contact_sheet_reference_role_label(reference_use: object) -> str:
    normalized = str(reference_use or "Included").strip() or "Included"
    mapping = {
        "Included": "Retained in the solve",
        "Excluded": "Excluded from the solve",
        "Anchor": "Exposure anchor",
    }
    return mapping.get(normalized, normalized)


def _contact_sheet_measurement_source_label(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return "Hero-center patch"
    lowered = text.lower().replace("_", " ")
    mapping = {
        "measurement unresolved": "No valid gray measurement",
        "shared original.display scalar log2": "Hero-center patch",
        "shared original.measured log2 luminance monitoring": "Hero-center patch",
        "shared original": "Hero-center patch",
        "stored measurement payload": "Hero-center patch",
        "rendered preview ipp2": "Hero-center patch",
        "median sample log2": "Hero-center patch",
    }
    return mapping.get(lowered, text.replace("_", " "))


def _contact_sheet_target_class_label(value: object) -> str:
    normalized = str(value or "").strip().lower()
    return {
        "sphere": "Gray sphere",
        "gray_card": "Gray card",
        "unresolved": "Gray target unresolved",
    }.get(normalized, "Gray target")


def _contact_sheet_chromaticity_visual(measured_rgb: List[float]) -> str:
    if len(measured_rgb) != 3:
        return ""
    red, green, blue = [float(value) for value in measured_rgb[:3]]
    warm_cool = max(-0.08, min(0.08, red - blue))
    green_magenta = max(-0.08, min(0.08, ((red + blue) * 0.5) - green))
    width = 236
    height = 84
    pad_x = 28
    bar_width = width - (pad_x * 2)
    warm_cool_y = 28
    green_magenta_y = 56
    center_x = width / 2.0
    scale = bar_width / 0.16
    warm_cool_x = center_x + (warm_cool * scale)
    green_magenta_x = center_x + (green_magenta * scale)
    return (
        f"<svg viewBox='0 0 {width} {height}' role='img' aria-label='Neutral chromaticity placement'>"
        f"<rect x='0' y='0' width='{width}' height='{height}' rx='12' fill='#f8fafc' stroke='#d7dee8'/>"
        f"<text x='{pad_x:.1f}' y='15' fill='#64748b' font-size='10' font-weight='700'>Warm ←→ Cool</text>"
        f"<text x='{center_x:.1f}' y='15' text-anchor='middle' fill='#0f172a' font-size='9' font-weight='700'>Neutral</text>"
        f"<line x1='{pad_x}' y1='{warm_cool_y}' x2='{width - pad_x}' y2='{warm_cool_y}' stroke='#cbd5e1' stroke-width='5' stroke-linecap='round'/>"
        f"<line x1='{center_x:.1f}' y1='{warm_cool_y - 8}' x2='{center_x:.1f}' y2='{warm_cool_y + 8}' stroke='#0f172a' stroke-width='2'/>"
        f"<circle cx='{warm_cool_x:.1f}' cy='{warm_cool_y}' r='6.5' fill='#2563eb' stroke='white' stroke-width='2'/>"
        f"<text x='{pad_x:.1f}' y='{green_magenta_y - 13}' fill='#64748b' font-size='10' font-weight='700'>Green ←→ Magenta</text>"
        f"<line x1='{pad_x}' y1='{green_magenta_y}' x2='{width - pad_x}' y2='{green_magenta_y}' stroke='#cbd5e1' stroke-width='5' stroke-linecap='round'/>"
        f"<line x1='{center_x:.1f}' y1='{green_magenta_y - 8}' x2='{center_x:.1f}' y2='{green_magenta_y + 8}' stroke='#0f172a' stroke-width='2'/>"
        f"<circle cx='{green_magenta_x:.1f}' cy='{green_magenta_y}' r='6.5' fill='#0f766e' stroke='white' stroke-width='2'/>"
        f"<text x='{center_x:.1f}' y='{height - 8}' text-anchor='middle' fill='#0f172a' font-size='10' font-weight='700'>Neutral placement from measured sample RGB</text>"
        "</svg>"
    )


def _gray_target_consistency_summary(rows: List[Dict[str, object]]) -> Dict[str, object]:
    tracked_rows = []
    for row in list(rows or []):
        reference_use = str(row.get("reference_use") or "Included")
        trust_class = str(row.get("trust_class") or "")
        if not bool(row.get("measurement_valid")) or reference_use == "Excluded" or trust_class in {"EXCLUDED", "UNTRUSTED"}:
            continue
        target_class = str(row.get("gray_target_class") or "unresolved").strip().lower()
        tracked_rows.append(
            {
                "clip_id": str(row.get("clip_id") or ""),
                "camera_label": str(row.get("camera_label") or row.get("clip_id") or ""),
                "target_class": target_class if target_class in {"sphere", "gray_card"} else "unresolved",
            }
        )
    counts = {"sphere": 0, "gray_card": 0, "unresolved": 0}
    for item in tracked_rows:
        counts[str(item["target_class"])] = counts.get(str(item["target_class"]), 0) + 1
    dominant = "unresolved"
    dominant_count = -1
    for target_class in ("sphere", "gray_card", "unresolved"):
        if counts[target_class] > dominant_count:
            dominant = target_class
            dominant_count = counts[target_class]
    mixed = counts["sphere"] > 0 and counts["gray_card"] > 0
    non_dominant = [
        item
        for item in tracked_rows
        if str(item.get("target_class") or "") in {"sphere", "gray_card"} and str(item.get("target_class") or "") != dominant
    ]
    summary = (
        "Retained cameras consistently measured the gray sphere."
        if dominant == "sphere" and not mixed
        else "Retained cameras consistently used gray-card fallback."
        if dominant == "gray_card" and not mixed
        else "Retained cameras mix gray-sphere and gray-card solves. Review is required before commit."
        if mixed
        else "Gray target class could not be established for the retained set."
    )
    return {
        "dominant_target_class": dominant,
        "counts": counts,
        "mixed_target_classes": mixed,
        "non_dominant_clip_ids": [str(item.get("clip_id") or "") for item in non_dominant if str(item.get("clip_id") or "").strip()],
        "non_dominant_camera_labels": [str(item.get("camera_label") or "") for item in non_dominant if str(item.get("camera_label") or "").strip()],
        "summary": summary,
        "commit_blocked": mixed,
    }


def _contact_sheet_flag_label(flag: str) -> str:
    normalized = str(flag or "").strip()
    mapping = {
        "Fallback used": "Alternate detection path used",
        "Residual > ±0.02": "Residual exceeds ±0.02 stops",
        "Uneven sample profile": "Sample profile varies across the sphere",
        "No anomaly flags": "No anomaly flags",
        "Excluded": "Excluded from the solve",
        "Included": "Retained in the solve",
    }
    return mapping.get(normalized, normalized)


def _contact_sheet_original_wb_block(
    *,
    clip: Dict[str, object],
    clip_metadata: Dict[str, object],
    top_level_wb_model: Dict[str, object],
) -> Dict[str, object]:
    metrics = dict(clip.get("metrics") or {})
    as_shot = extract_as_shot_white_balance(clip_metadata if clip_metadata else None)
    pre_residual = clip.get("pre_color_residual")
    if pre_residual is None:
        pre_residual = metrics.get("color", {}).get("pre_residual") if isinstance(metrics.get("color"), dict) else None
    source_kind = "metadata_context"
    source_label = "Using camera capture settings"
    measured_rgb_payload = clip.get("measured_rgb_chromaticity")
    if not measured_rgb_payload and isinstance(metrics.get("measurement"), dict):
        measured_rgb_payload = metrics.get("measurement", {}).get("measured_rgb_chromaticity")
    measured_rgb = [float(value) for value in list(measured_rgb_payload or [])[:3]]
    commit_values = dict(clip.get("commit_values") or metrics.get("commit_values") or {})
    axes = dict(commit_values.get("white_balance_axes") or clip.get("white_balance_axes") or {})
    amber_blue = float(axes.get("amber_blue", 0.0) or 0.0)
    green_magenta = float(axes.get("green_magenta", 0.0) or 0.0)
    if measured_rgb:
        source_kind = "measured_original_still"
        source_label = "Measured from the original neutral target"
    elif pre_residual is not None or str(clip.get("white_balance_model_label") or metrics.get("white_balance_model_label") or "").strip():
        source_kind = "measured_or_solved"
        source_label = "Using stored white-balance analysis"
    shared_kelvin_mode = str((top_level_wb_model or {}).get("shared_kelvin_mode") or "")
    shared_tint_mode = str((top_level_wb_model or {}).get("shared_tint_mode") or "")
    shared_kelvin = (top_level_wb_model or {}).get("shared_kelvin")
    shared_tint = (top_level_wb_model or {}).get("shared_tint")
    context_bits: List[str] = []
    model_label = str(
        clip.get("white_balance_model_label")
        or metrics.get("white_balance_model_label")
        or (top_level_wb_model or {}).get("model_label")
        or ""
    ).strip()
    if model_label and model_label != "n/a":
        context_bits.append(model_label)
    if shared_kelvin_mode == "shared" and shared_kelvin is not None:
        context_bits.append(f"Shared Kelvin {int(round(float(shared_kelvin)))}K")
    elif shared_kelvin_mode == "per_camera":
        context_bits.append("Kelvin varies per camera")
    if shared_tint_mode == "shared" and shared_tint is not None:
        context_bits.append(f"Shared Tint {float(shared_tint):+.1f}")
    elif shared_tint_mode == "per_camera":
        context_bits.append("Tint varies per camera")
    as_shot_kelvin = as_shot.get("kelvin")
    as_shot_tint = as_shot.get("tint")
    chroma_summary = (
        f"R {measured_rgb[0]:.3f} / G {measured_rgb[1]:.3f} / B {measured_rgb[2]:.3f}"
        if len(measured_rgb) == 3
        else "Stored original neutral sample is not available."
    )
    gray_target_class = str(
        clip.get("gray_target_class")
        or metrics.get("gray_target_class")
        or ((metrics.get("exposure") or {}).get("gray_target_class") if isinstance(metrics.get("exposure"), dict) else "")
        or "sphere"
    )
    gray_target_confidence = float(
        clip.get("gray_target_confidence")
        or metrics.get("gray_target_confidence")
        or ((metrics.get("exposure") or {}).get("gray_target_confidence") if isinstance(metrics.get("exposure"), dict) else 0.0)
        or clip.get("confidence")
        or metrics.get("confidence")
        or 0.0
    )
    stability_summary = _contact_sheet_stability_summary(
        confidence=(clip.get("confidence") or metrics.get("confidence")),
        log2_spread=(clip.get("neutral_sample_log2_spread") or metrics.get("neutral_sample_log2_spread")),
        chroma_spread=(clip.get("neutral_sample_chromaticity_spread") or metrics.get("neutral_sample_chromaticity_spread")),
        sample_count=(clip.get("neutral_sample_count") or metrics.get("neutral_sample_count")),
    )
    wb_context = " | ".join(context_bits) if context_bits else "Stored array white-balance context is limited for this camera."
    return {
        "source_kind": source_kind,
        "source_label": source_label,
        "as_shot_kelvin": (int(round(float(as_shot_kelvin))) if as_shot_kelvin is not None else None),
        "as_shot_tint": (float(as_shot_tint) if as_shot_tint is not None else None),
        "pre_neutral_residual": (float(pre_residual) if pre_residual is not None else None),
        "derived_cast": _contact_sheet_original_cast_label(amber_blue, green_magenta) if measured_rgb or axes else "A derived original-still cast was not stored for this camera.",
        "derived_axes": {
            "amber_blue": amber_blue,
            "green_magenta": green_magenta,
        },
        "chroma_summary": chroma_summary,
        "context": wb_context,
        "gray_target_class": gray_target_class,
        "gray_target_label": _contact_sheet_target_class_label(gray_target_class),
        "gray_target_confidence": gray_target_confidence,
        "stability_summary": stability_summary,
        "chromaticity_chart_svg": _contact_sheet_chromaticity_visual(measured_rgb),
        "has_chroma_summary": len(measured_rgb) == 3,
        "has_context_summary": _contact_sheet_has_meaningful_text(wb_context),
        "has_stability_summary": _contact_sheet_has_meaningful_text(stability_summary),
        "has_derived_cast": _contact_sheet_has_meaningful_text(
            _contact_sheet_original_cast_label(amber_blue, green_magenta) if measured_rgb or axes else ""
        ),
    }


def _contact_sheet_array_focus_entries(
    entries: List[Dict[str, object]],
    *,
    focus_mode: str,
    reference_candidate: Optional[Dict[str, object]],
) -> List[Dict[str, object]]:
    normalized = normalize_report_focus(focus_mode)
    if normalized == "full":
        return list(entries)
    anchors: List[Dict[str, object]] = []
    if reference_candidate is not None:
        anchors.append(reference_candidate)
    anchors.extend(item for item in entries if bool(item.get("is_hero_camera")))
    unique_by_clip = {str(item.get("clip_id") or ""): item for item in anchors if str(item.get("clip_id") or "").strip()}
    anchors = list(unique_by_clip.values())
    retained = [item for item in entries if str(item.get("reference_use") or "Included") != "Excluded"]
    cluster_extremes: List[Dict[str, object]] = []
    if retained:
        low = min(retained, key=lambda item: float(item.get("offset_to_anchor", 0.0) or 0.0))
        high = max(retained, key=lambda item: float(item.get("offset_to_anchor", 0.0) or 0.0))
        cluster_extremes = [low] if str(low.get("clip_id") or "") == str(high.get("clip_id") or "") else [low, high]
    outliers = [item for item in entries if _contact_sheet_attention_class(item) == "outlier"]
    if normalized == "outliers":
        selected = outliers
    elif normalized == "anchors":
        selected = anchors
    elif normalized == "cluster_extremes":
        selected = cluster_extremes
    else:
        selected = [*outliers, *anchors, *cluster_extremes]
    if not selected:
        selected = sorted(entries, key=lambda item: float(item.get("residual_abs_stops", 0.0) or 0.0), reverse=True)[: min(6, len(entries))]
    ordered: Dict[str, Dict[str, object]] = {}
    for item in selected:
        clip_id = str(item.get("clip_id") or "")
        if clip_id:
            ordered[clip_id] = item
    return list(ordered.values())


def _contact_sheet_chunks(entries: List[Dict[str, object]], *, columns: int = 2, max_rows: int = 2) -> List[List[Dict[str, object]]]:
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
    width = 1040
    height = 268
    margin_left = 64
    margin_right = 22
    margin_top = 22
    margin_bottom = 46
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
            "stroke='#9ca3af' stroke-width='2' stroke-dasharray='8 6' />"
            f"<text x='{width - margin_right:.1f}' y='{goal_y - 8:.1f}' text-anchor='end' fill='#6b7280' font-size='14'>goal {goal:+.2f}</text>"
        )

    x_labels = []
    for index, label in enumerate(labels):
        x = x_for(index)
        x_labels.append(
            f"<text x='{x:.1f}' y='{height - 12:.1f}' text-anchor='middle' fill='#64748b' font-size='14' font-weight='700'>{html.escape(label)}</text>"
        )
    y_labels = []
    for tick in y_ticks:
        y = y_for(tick)
        y_labels.append(
            f"<line x1='{margin_left:.1f}' y1='{y:.1f}' x2='{width - margin_right:.1f}' y2='{y:.1f}' stroke='#d6dee8' stroke-width='1.5' />"
            f"<text x='{margin_left - 10:.1f}' y='{y + 5:.1f}' text-anchor='end' fill='#64748b' font-size='13'>{tick:.2f}{html.escape(units)}</text>"
        )
    point_markup = "".join(
        f"<circle cx='{x_for(index):.1f}' cy='{y_for(value):.1f}' r='6' fill='{stroke}' />"
        for index, value in enumerate(values)
    )
    return (
        f"<svg viewBox='0 0 {width} {height}' role='img' aria-label='{html.escape(title)}'>"
        f"<rect x='0' y='0' width='{width}' height='{height}' fill='white' />"
        f"<text x='0' y='18' fill='#0f172a' font-size='18' font-weight='700'>{html.escape(title)}</text>"
        + "".join(y_labels)
        + goal_markup
        + f"<polyline fill='none' stroke='{stroke}' stroke-width='4.5' points='{points}' />"
        + point_markup
        + "".join(x_labels)
        + "</svg>"
    )


def _contact_sheet_svg_color_deviation_chart(
    title: str,
    labels: List[str],
    kelvin_values: List[float],
    tint_values: List[float],
    *,
    target_kelvin: float = 5600.0,
    target_tint: float = 0.0,
    stroke: str = "#0f766e",
) -> str:
    if not labels:
        return ""
    width = 1080
    height = 396
    margin_left = 94
    margin_right = 34
    margin_top = 44
    margin_bottom = 72
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    kelvin_deltas = [float(value or 0.0) - target_kelvin for value in kelvin_values]
    tint_deltas = [float(value or 0.0) - target_tint for value in tint_values]
    tint_min, tint_max = _contact_sheet_metric_range(tint_deltas + [0.0], fallback_span=2.0)
    kelvin_min, kelvin_max = _contact_sheet_metric_range(kelvin_deltas + [0.0], fallback_span=300.0)

    def x_for(delta_tint: float) -> float:
        if math.isclose(tint_min, tint_max, abs_tol=1e-9):
            return margin_left + plot_width / 2
        return margin_left + ((delta_tint - tint_min) / (tint_max - tint_min)) * plot_width

    def y_for(delta_kelvin: float) -> float:
        if math.isclose(kelvin_min, kelvin_max, abs_tol=1e-9):
            return margin_top + plot_height / 2
        return margin_top + (1.0 - ((delta_kelvin - kelvin_min) / (kelvin_max - kelvin_min))) * plot_height

    zero_x = x_for(0.0)
    zero_y = y_for(0.0)
    x_ticks = [tint_min, 0.0, tint_max]
    y_ticks = [kelvin_min, 0.0, kelvin_max]
    guides = "".join(
        f"<line x1='{margin_left:.1f}' y1='{y_for(tick):.1f}' x2='{width - margin_right:.1f}' y2='{y_for(tick):.1f}' stroke='#d6dee8' stroke-width='2.5' />"
        f"<text x='{margin_left - 14:.1f}' y='{y_for(tick) + 7:.1f}' text-anchor='end' fill='#64748b' font-size='18'>{tick:+.0f}K</text>"
        for tick in y_ticks
    )
    guides += "".join(
        f"<line x1='{x_for(tick):.1f}' y1='{margin_top:.1f}' x2='{x_for(tick):.1f}' y2='{height - margin_bottom:.1f}' stroke='#e2e8f0' stroke-width='2.5' />"
        f"<text x='{x_for(tick):.1f}' y='{height - 18:.1f}' text-anchor='middle' fill='#64748b' font-size='18'>{tick:+.1f}</text>"
        for tick in x_ticks
    )
    target_markup = (
        f"<line x1='{margin_left:.1f}' y1='{zero_y:.1f}' x2='{width - margin_right:.1f}' y2='{zero_y:.1f}' stroke='#94a3b8' stroke-width='3' />"
        f"<line x1='{zero_x:.1f}' y1='{margin_top:.1f}' x2='{zero_x:.1f}' y2='{height - margin_bottom:.1f}' stroke='#94a3b8' stroke-width='3' />"
        f"<circle cx='{zero_x:.1f}' cy='{zero_y:.1f}' r='11' fill='white' stroke='#0f172a' stroke-width='3.5' />"
        f"<text x='{zero_x + 18:.1f}' y='{zero_y - 18:.1f}' fill='#475569' font-size='18' font-weight='700'>Target 5600K / Tint 0</text>"
    )
    axis_notes = [
        f"<text x='{margin_left + 10:.1f}' y='{margin_top + 18:.1f}' fill='#64748b' font-size='17' font-weight='700'>Cooler</text>",
        f"<text x='{margin_left + 10:.1f}' y='{height - margin_bottom - 10:.1f}' fill='#64748b' font-size='17' font-weight='700'>Warmer</text>",
        f"<text x='{margin_left + 6:.1f}' y='{zero_y - 14:.1f}' fill='#64748b' font-size='17' font-weight='700'>Green</text>",
        f"<text x='{width - margin_right - 6:.1f}' y='{zero_y - 14:.1f}' text-anchor='end' fill='#64748b' font-size='17' font-weight='700'>Magenta</text>",
    ]
    placed_labels: List[Tuple[float, float]] = []
    points = []
    for index, label in enumerate(labels):
        x = x_for(tint_deltas[index])
        y = y_for(kelvin_deltas[index])
        text_anchor = "start" if x < (margin_left + plot_width * 0.75) else "end"
        label_x = x + 16 if text_anchor == "start" else x - 16
        label_y = y - 16 if index % 2 == 0 else y + 28
        for placed_x, placed_y in placed_labels:
            if abs(label_y - placed_y) < 20 and abs(label_x - placed_x) < 72:
                label_y += 20 if label_y <= y else -20
        placed_labels.append((label_x, label_y))
        points.append(
            f"<circle cx='{x:.1f}' cy='{y:.1f}' r='11' fill='{stroke}' stroke='white' stroke-width='3' />"
            f"<line x1='{x:.1f}' y1='{y:.1f}' x2='{label_x:.1f}' y2='{label_y - 6:.1f}' stroke='{stroke}' stroke-width='2.25' stroke-opacity='0.7' />"
            f"<text x='{label_x:.1f}' y='{label_y:.1f}' text-anchor='{text_anchor}' fill='#0f172a' font-size='18' font-weight='700'>{html.escape(label)}</text>"
        )
    return (
        f"<svg viewBox='0 0 {width} {height}' role='img' aria-label='{html.escape(title)}'>"
        f"<rect x='0' y='0' width='{width}' height='{height}' fill='white' />"
        f"<text x='0' y='26' fill='#0f172a' font-size='24' font-weight='700'>{html.escape(title)}</text>"
        f"<text x='{margin_left:.1f}' y='{height - 22:.1f}' fill='#64748b' font-size='18' font-weight='700'>Tint delta from target</text>"
        f"<text x='{margin_left - 2:.1f}' y='{margin_top - 14:.1f}' fill='#64748b' font-size='18' font-weight='700'>Kelvin delta from target</text>"
        + guides
        + target_markup
        + "".join(axis_notes)
        + "".join(points)
        + "</svg>"
    )


def _contact_sheet_svg_exposure_offset_lollipop(entries: List[Dict[str, object]]) -> str:
    if not entries:
        return ""
    ordered = sorted(
        list(entries),
        key=lambda item: abs(float(item.get("offset_to_anchor", 0.0) or 0.0)),
        reverse=True,
    )
    width = 1080
    row_height = 44
    height = max(220, 120 + row_height * len(ordered))
    margin_left = 220
    margin_right = 46
    margin_top = 42
    margin_bottom = 36
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    offsets = [abs(float(item.get("offset_to_anchor", 0.0) or 0.0)) for item in ordered]
    max_abs_offset = max(offsets) if offsets else 0.0
    focus_abs_offset = float(np.quantile(np.asarray(offsets, dtype=np.float32), 0.85)) if offsets else 0.0
    axis_max = max(0.20, focus_abs_offset + 0.05)
    if max_abs_offset <= 0.60:
        axis_max = max(axis_max, max_abs_offset + 0.05)
    axis_max = min(0.75, axis_max)
    axis_max = max(0.20, round(axis_max / 0.05) * 0.05)

    def x_for(value: float) -> float:
        clipped = max(-axis_max, min(axis_max, float(value)))
        normalized = (clipped + axis_max) / (axis_max * 2.0)
        return margin_left + (normalized * plot_width)

    zero_x = x_for(0.0)
    guides = [
        f"<line x1='{zero_x:.1f}' y1='{margin_top - 8:.1f}' x2='{zero_x:.1f}' y2='{height - margin_bottom + 6:.1f}' stroke='#475569' stroke-width='2.5' />"
    ]
    tick = -axis_max
    while tick <= axis_max + 1e-9:
        x = x_for(tick)
        is_major = math.isclose((tick / 0.10) - round(tick / 0.10), 0.0, abs_tol=1e-6) or math.isclose(tick, 0.0, abs_tol=1e-6)
        guides.append(
            f"<line x1='{x:.1f}' y1='{margin_top - 4:.1f}' x2='{x:.1f}' y2='{height - margin_bottom + 4:.1f}' "
            f"stroke='{'#cbd5e1' if is_major else '#e2e8f0'}' stroke-width='{'1.8' if is_major else '0.9'}' />"
        )
        if is_major:
            guides.append(f"<text x='{x:.1f}' y='{height - 6:.1f}' text-anchor='middle' fill='#64748b' font-size='14'>{tick:+.2f}</text>")
        tick = round(tick + 0.05, 4)
    guides.append(
        f"<rect x='{x_for(-0.25):.1f}' y='{margin_top:.1f}' width='{max(x_for(0.25) - x_for(-0.25), 4):.1f}' height='{plot_height:.1f}' fill='#dcfce7' opacity='0.65' rx='12' />"
    )
    if axis_max > 0.25:
        guides.append(
            f"<rect x='{x_for(-axis_max):.1f}' y='{margin_top:.1f}' width='{max(x_for(axis_max) - x_for(-axis_max), 4):.1f}' height='{plot_height:.1f}' fill='#fef3c7' opacity='0.22' rx='12' />"
        )
    rows = []
    for index, item in enumerate(ordered):
        y = margin_top + (index * row_height) + 18
        offset = float(item.get("offset_to_anchor", 0.0) or 0.0)
        plotted_offset = max(-axis_max, min(axis_max, offset))
        camera_label = str(item.get("camera_label") or item.get("clip_id") or f"Camera {index + 1}")
        attention = str(item.get("attention_class") or "safe")
        is_anchor = bool(item.get("is_anchor_reference"))
        fill = "#2563eb" if is_anchor else "#dc2626" if attention == "outlier" else "#f59e0b" if attention == "borderline" else "#0f172a"
        label_suffix = " | Anchor" if is_anchor else " | Excluded" if attention == "outlier" else " | Review" if attention == "borderline" else ""
        clipped = not math.isclose(plotted_offset, offset, abs_tol=1e-6)
        value_label = f"{offset:+.2f} stops" + (" (clipped)" if clipped else "")
        rows.append(
            f"<text x='{margin_left - 12:.1f}' y='{y + 5:.1f}' text-anchor='end' fill='#0f172a' font-size='15' font-weight='700'>{html.escape(camera_label + label_suffix)}</text>"
            f"<line x1='{zero_x:.1f}' y1='{y:.1f}' x2='{x_for(plotted_offset):.1f}' y2='{y:.1f}' stroke='#94a3b8' stroke-width='4' />"
            f"<circle cx='{x_for(plotted_offset):.1f}' cy='{y:.1f}' r='7' fill='{fill}' stroke='white' stroke-width='2.5' />"
            + (
                f"<polygon points='{x_for(plotted_offset):.1f},{y - 8:.1f} {x_for(plotted_offset) + (10 if offset > 0 else -10):.1f},{y:.1f} {x_for(plotted_offset):.1f},{y + 8:.1f}' fill='{fill}' opacity='0.85' />"
                if clipped
                else ""
            )
            + f"<text x='{x_for(plotted_offset) + 14 if plotted_offset >= 0 else x_for(plotted_offset) - 14:.1f}' y='{y + 5:.1f}' text-anchor='{'start' if plotted_offset >= 0 else 'end'}' fill='#334155' font-size='14' font-weight='700'>{html.escape(value_label)}</text>"
        )
    legend = (
        f"<text x='{margin_left:.1f}' y='20' fill='#0f172a' font-size='18' font-weight='700'>Exposure offset from the anchor (sorted by magnitude)</text>"
        f"<text x='{margin_left:.1f}' y='40' fill='#64748b' font-size='13'>Fine grid = 0.05 stops | Green band = within ±0.25 | Blue = anchor | Red = excluded / outlier | Clipped labels mark larger offsets outside the focused range</text>"
    )
    return (
        f"<svg viewBox='0 0 {width} {height}' role='img' aria-label='Exposure offsets from the anchor'>"
        f"<rect x='0' y='0' width='{width}' height='{height}' fill='white' />"
        + "".join(guides)
        + legend
        + "".join(rows)
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
    top_level_wb_model = dict(payload.get("white_balance_model") or {})
    focus_validation = dict(payload.get("focus_validation") or {})
    focus_rows_by_clip = {
        str(item.get("clip_id") or ""): dict(item)
        for item in list(focus_validation.get("rows") or [])
        if str(item.get("clip_id") or "").strip()
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
        resolved_measurements = _contact_sheet_resolve_measurement_fields(
            clip_id=clip_id,
            clip_row=dict(clip),
            shared=shared,
            exposure_metrics=exposure_metrics,
            ipp2_row=ipp2_row,
        )
        measurement_valid = bool(resolved_measurements.get("measurement_valid"))
        sample_1_ire = _numeric_value_or_none(resolved_measurements.get("sample_1_ire"))
        sample_2_ire = _numeric_value_or_none(resolved_measurements.get("sample_2_ire"))
        sample_3_ire = _numeric_value_or_none(resolved_measurements.get("sample_3_ire"))
        display_scalar_log2 = _numeric_value_or_none(resolved_measurements.get("display_scalar_log2"))
        display_scalar_ire = (
            _contact_sheet_display_scalar_ire(
                display_scalar_log2,
                shared.get("display_scalar_ire", exposure_metrics.get("display_scalar_ire")),
            )
            if measurement_valid and display_scalar_log2 is not None
            else None
        )
        adjusted_display_scalar_log2 = _numeric_value_or_none(ipp2_row.get("ipp2_value_log2"))
        if adjusted_display_scalar_log2 is None:
            adjusted_display_scalar_log2 = display_scalar_log2
        adjusted_display_scalar_ire = (
            _contact_sheet_display_scalar_ire(
                adjusted_display_scalar_log2,
                ipp2_row.get("ipp2_value_ire"),
            )
            if measurement_valid and adjusted_display_scalar_log2 is not None
            else None
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
        original_wb = _contact_sheet_original_wb_block(
            clip=dict(clip),
            clip_metadata=clip_metadata,
            top_level_wb_model=top_level_wb_model,
        )
        focus_row = dict(focus_rows_by_clip.get(clip_id) or clip.get("focus_validation") or {})
        detection_details = dict(ipp2_row.get("ipp2_detection_details") or {})
        detection_status = _sphere_detection_status(
            bool(ipp2_row.get("ipp2_sphere_detection_success")),
            bool(ipp2_row.get("ipp2_sphere_detection_unresolved")),
        )
        rejection_reasons = _sphere_detection_rejection_reasons_from_details(detection_details)
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
                "measurement_valid": measurement_valid,
                "sample_1_ire": sample_1_ire,
                "sample_2_ire": sample_2_ire,
                "sample_3_ire": sample_3_ire,
                "target_sample_label": str(resolved_measurements.get("target_sample_label") or CONTACT_SHEET_TARGET_SAMPLE_LABEL),
                "sample_sources": dict(resolved_measurements.get("sample_sources") or {}),
                "display_scalar_log2": display_scalar_log2,
                "display_scalar_ire": display_scalar_ire,
                "display_scalar_source": str(resolved_measurements.get("display_scalar_source") or ""),
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
                "original_wb": original_wb,
                "residual_stops": residual_stops,
                "residual_abs_stops": residual_abs_stops,
                "offset_to_anchor": float(ipp2_row.get("camera_offset_from_anchor", ipp2_row.get("derived_exposure_offset_stops", 0.0)) or 0.0),
                "reference_use": (
                    "Excluded"
                    if not measurement_valid
                    else str(ipp2_row.get("reference_use") or clip.get("reference_use") or "Included")
                ),
                "trust_score": float(clip.get("trust_score", ipp2_row.get("trust_score", 0.0)) or 0.0),
                "trust_class": str(clip.get("trust_class") or ipp2_row.get("trust_class") or "TRUSTED"),
                "sphere_detection_note": str(ipp2_row.get("sphere_detection_note") or "Sphere check verified"),
                "detection_status": detection_status,
                "detection_method": str(ipp2_row.get("ipp2_detection_source") or clip.get("gray_target_detection_method") or ""),
                "rejection_reasons": rejection_reasons,
                "profile_note": str(ipp2_row.get("profile_note") or "Sample profile aligned with stored solve."),
                "fallback_used": bool(
                    clip.get("gray_target_fallback_used")
                    or ipp2_row.get("gray_target_fallback_used")
                ),
                "sample_plausibility": str(ipp2_row.get("ipp2_sample_plausibility") or clip.get("sample_plausibility") or ""),
                "gray_target_class": str(
                    ipp2_row.get("ipp2_target_class")
                    or clip.get("gray_target_class")
                    or ((clip.get("metrics") or {}).get("exposure", {}) or {}).get("gray_target_class")
                    or "sphere"
                ),
                "gray_target_detection_method": str(
                    ipp2_row.get("ipp2_target_detection_method")
                    or clip.get("gray_target_detection_method")
                    or ((clip.get("metrics") or {}).get("exposure", {}) or {}).get("gray_target_detection_method")
                    or ""
                ),
                "gray_target_confidence": float(
                    ipp2_row.get("ipp2_target_confidence")
                    or clip.get("gray_target_confidence")
                    or ((clip.get("metrics") or {}).get("exposure", {}) or {}).get("gray_target_confidence")
                    or clip.get("confidence")
                    or 0.0
                ),
                "measurement_domain": str(payload.get("measurement_preview_transform") or REVIEW_PREVIEW_TRANSFORM),
                "focus_validation": focus_row,
                "focus_classification": str(focus_row.get("focus_classification") or ""),
                "focus_composite_score": float(focus_row.get("composite_focus_score", 0.0) or 0.0),
            }
        )

    entries = sorted(
        entries,
        key=lambda item: _camera_layout_sort_key(str(item.get("camera_label") or item.get("clip_id") or "")),
    )
    before_values = [float(item["display_scalar_log2"]) for item in entries if item.get("measurement_valid") and item.get("display_scalar_log2") is not None]
    after_values = [
        float(item.get("adjusted_display_scalar_log2"))
        for item in entries
        if item.get("measurement_valid") and item.get("adjusted_display_scalar_log2") is not None
    ]
    center_values = [
        float(item["sample_2_ire"])
        for item in entries
        if item.get("measurement_valid") and item.get("sample_2_ire") is not None
    ]
    reference_candidate = _contact_sheet_reference_candidate(entries)
    reference_clip_id = str((reference_candidate or {}).get("clip_id") or "")
    for entry in entries:
        entry["attention_class"] = _contact_sheet_attention_class(entry)
        entry["is_reference_candidate"] = str(entry.get("clip_id") or "") == reference_clip_id
        entry["is_anchor_reference"] = bool(entry.get("is_reference_candidate")) or bool(entry.get("is_hero_camera"))
    is_large_array = len(entries) >= LARGE_ARRAY_OVERVIEW_THRESHOLD
    requested_focus = normalize_report_focus(payload.get("report_focus"))
    effective_focus = DEFAULT_LARGE_ARRAY_AUTO_FOCUS if requested_focus == "auto" and is_large_array else "full" if requested_focus == "auto" else requested_focus
    detail_entries = _contact_sheet_array_focus_entries(entries, focus_mode=effective_focus, reference_candidate=reference_candidate)
    outlier_entries = [item for item in entries if str(item.get("attention_class") or "") == "outlier"]
    borderline_entries = [item for item in entries if str(item.get("attention_class") or "") == "borderline"]
    anchor_entries = [item for item in entries if bool(item.get("is_anchor_reference"))]
    retained_entries = [item for item in entries if str(item.get("reference_use") or "Included") != "Excluded"]
    gray_target_consistency = _gray_target_consistency_summary(entries)
    focus_rows = [dict(item) for item in list(focus_validation.get("rows") or [])]
    cluster_extreme_entries: List[Dict[str, object]] = []
    if retained_entries:
        low = min(retained_entries, key=lambda item: float(item.get("offset_to_anchor", 0.0) or 0.0))
        high = max(retained_entries, key=lambda item: float(item.get("offset_to_anchor", 0.0) or 0.0))
        cluster_extreme_entries = [low] if str(low.get("clip_id") or "") == str(high.get("clip_id") or "") else [low, high]
    recommended_attention = list(dict.fromkeys(
        [str(item.get("camera_label") or item.get("clip_id") or "") for item in [*outlier_entries, *anchor_entries, *cluster_extreme_entries] if str(item.get("camera_label") or item.get("clip_id") or "").strip()]
    ))
    overview_pages = _chunk_tiles(
        [
            {
                "clip_id": str(item.get("clip_id") or ""),
                "camera_label": str(item.get("camera_label") or ""),
                "status": str(item.get("status") or "REVIEW"),
                "attention_class": str(item.get("attention_class") or "safe"),
                "residual_abs_stops": float(item.get("residual_abs_stops", 0.0) or 0.0),
                "display_scalar_ire": _numeric_value_or_none(item.get("display_scalar_ire")),
                "measurement_valid": bool(item.get("measurement_valid")),
                "reference_use": str(item.get("reference_use") or "Included"),
                "is_anchor_reference": bool(item.get("is_anchor_reference")),
            }
            for item in entries
        ],
        24,
    )
    contact_pages = []
    for page_index, page_entries in enumerate(_contact_sheet_chunks(detail_entries, columns=2, max_rows=2), start=1):
        labels = [str(item.get("camera_label") or item.get("clip_id") or "") for item in page_entries]
        original_exposure_values = [float(item.get("display_scalar_ire", 0.0) or 0.0) for item in page_entries]
        adjusted_exposure_values = [float(item.get("adjusted_display_scalar_ire", item.get("display_scalar_ire", 0.0)) or 0.0) for item in page_entries]
        adjusted_kelvin_values = [float(item.get("kelvin", 0.0) or 0.0) for item in page_entries]
        adjusted_tint_values = [float(item.get("tint", 0.0) or 0.0) for item in page_entries]
        original_kelvin_values = [float(item.get("original_kelvin", 0.0) or 0.0) for item in page_entries]
        original_tint_values = [float(item.get("original_tint", 0.0) or 0.0) for item in page_entries]
        residual_values = [float(item.get("residual_stops", 0.0) or 0.0) for item in page_entries]
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
                "original_color_deviation_chart_svg": _contact_sheet_svg_color_deviation_chart(
                    "Original white-balance deviation",
                    labels,
                    original_kelvin_values,
                    original_tint_values,
                    stroke="#0f766e",
                ),
                "adjusted_color_deviation_chart_svg": _contact_sheet_svg_color_deviation_chart(
                    "Adjusted white-balance deviation",
                    labels,
                    adjusted_kelvin_values,
                    adjusted_tint_values,
                    stroke="#7c3aed",
                ),
                "adjusted_residual_chart_svg": _contact_sheet_svg_single_series(
                    "Validation residual (stops)",
                    labels,
                    residual_values,
                    stroke="#991b1b",
                    units="",
                    goal=0.0,
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
        "is_large_array": is_large_array,
        "requested_report_focus": requested_focus,
        "effective_report_focus": effective_focus,
        "effective_report_focus_label": report_focus_label(effective_focus),
        "detail_entry_count": len(detail_entries),
        "overview_pages": overview_pages,
        "outlier_entries": outlier_entries,
        "anchor_entries": anchor_entries,
        "cluster_extreme_entries": cluster_extreme_entries,
        "recommended_attention": recommended_attention,
        "gray_target_consistency": gray_target_consistency,
        "focus_validation": focus_validation,
        "focus_rows": focus_rows,
        "summary_blocks": {
            "excluded_count": sum(1 for item in entries if str(item.get("reference_use") or "Included") == "Excluded"),
            "outlier_count": len(outlier_entries),
            "borderline_count": len(borderline_entries),
            "anchor_count": len(anchor_entries),
            "tint_spread": (
                max(float(item.get("tint", 0.0) or 0.0) for item in entries) - min(float(item.get("tint", 0.0) or 0.0) for item in entries)
            ) if len(entries) >= 2 else 0.0,
            "kelvin_spread": (
                max(float(item.get("kelvin", 0.0) or 0.0) for item in entries) - min(float(item.get("kelvin", 0.0) or 0.0) for item in entries)
            ) if len(entries) >= 2 else 0.0,
        },
        "visuals": dict(payload.get("visuals") or {}),
        "contact_pages": contact_pages,
        "entries": entries,
        "detail_entries": detail_entries,
    }


def render_contact_sheet_pdf_from_html(html_path: str, *, output_path: str) -> str:
    raise_if_cancelled("Run cancelled before PDF rendering.")
    preflight = contact_sheet_pdf_export_preflight(html_path)
    asset_validation = dict(preflight.get("asset_validation") or {})
    missing_assets = list(asset_validation.get("missing_assets") or [])
    if missing_assets:
        missing_preview = "; ".join(missing_assets[:6])
        if len(missing_assets) > 6:
            missing_preview += f"; ... and {len(missing_assets) - 6} more"
        raise RuntimeError(
            "Contact-sheet HTML references missing image assets. "
            f"HTML: {html_path}. Missing: {missing_preview}"
        )
    if not bool(preflight.get("weasyprint_importable")):
        dyld = str(preflight.get("dyld_fallback_library_path") or "")
        interpreter = str(preflight.get("interpreter") or sys.executable)
        error_text = str(preflight.get("weasyprint_error") or "unknown import error")
        raise RuntimeError(
            "HTML-to-PDF rendering requires WeasyPrint plus its native cairo/pango/glib libraries. "
            f"Interpreter: {interpreter}. "
            f"DYLD_FALLBACK_LIBRARY_PATH: {dyld or '<unset>'}. "
            f"Import error: {error_text}. "
            "Install the missing system libraries for WeasyPrint and re-run the report export."
        )
    try:
        from weasyprint import HTML
    except Exception as exc:  # pragma: no cover - preflight should already catch this
        raise RuntimeError(str(exc)) from exc

    source_html = Path(html_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    HTML(filename=str(source_html), base_url=str(source_html.parent)).write_pdf(str(output))
    return str(output)


def render_contact_sheet_pdf(
    payload: Dict[str, object],
    *,
    output_path: str,
    title: str,
    timestamp_label: Optional[str] = None,
) -> str:
    output = Path(output_path).expanduser().resolve()
    html_path = output.with_suffix(".html")
    html_markup = render_contact_sheet_html(
        payload,
        html_path=str(html_path),
        title_override=title,
        timestamp_label=timestamp_label,
    )
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(html_markup, encoding="utf-8")
    return render_contact_sheet_pdf_from_html(str(html_path), output_path=str(output))


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
    exact_correction_text = f"Exposure adjustment applied: {correction_stops:+.2f} stops"
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
        "sphere_detection_note": str(validation.get("sphere_detection_note") or "Sphere check needs review"),
        "tone": tone,
    }


def render_contact_sheet_html(
    payload: Dict[str, object],
    *,
    html_path: Optional[str] = None,
    title_override: Optional[str] = None,
    timestamp_label: Optional[str] = None,
) -> str:
    view_model = _contact_sheet_view_model(payload)
    final_title = str(title_override or view_model.get("title") or "R3DMatch Calibration Assessment")
    html_file = Path(html_path).expanduser().resolve() if html_path else None
    report_dir = html_file.parent if html_file else None
    images_dir = report_dir / "images" if report_dir else None
    logo_markup = (
        f"<img class='brand-logo' src='{LOGO_PATH.as_posix()}' alt='R3DMatch logo' />"
        if LOGO_PATH.exists()
        else "<div class='brand-logo-text'>R3DMatch</div>"
    )
    chart_entries = [
        entry for entry in list(view_model.get("entries") or [])
        if entry.get("original_kelvin") not in (None, 0.0) or entry.get("original_tint") not in (None, 0.0)
    ]
    chart_svg = ""
    if chart_entries:
        chart_svg = _contact_sheet_svg_color_deviation_chart(
            "White-balance deviation (ΔK / ΔTint from 5600K / Tint 0)",
            [str(item.get("camera_label") or item.get("clip_id") or "") for item in chart_entries],
            [float(item.get("original_kelvin", 0.0) or 0.0) for item in chart_entries],
            [float(item.get("original_tint", 0.0) or 0.0) for item in chart_entries],
            stroke="#0f766e",
        )
    exposure_chart_svg = _contact_sheet_svg_exposure_offset_lollipop([dict(item) for item in list(view_model.get("entries") or [])])
    summary_blocks = dict(view_model.get("summary_blocks") or {})
    anchor_summary_text = str(view_model.get("anchor_summary") or "Exposure Anchor: Derived from retained cluster")
    anchor_summary_display = anchor_summary_text.replace("Exposure Anchor: ", "", 1)
    gray_target_consistency = dict(view_model.get("gray_target_consistency") or {})
    focus_validation = dict(view_model.get("focus_validation") or {})
    focus_rows = [dict(item) for item in list(view_model.get("focus_rows") or [])]
    focus_chart_svg = ""
    if focus_rows:
        sorted_focus_rows = sorted(focus_rows, key=lambda item: float(item.get("composite_focus_score", 0.0) or 0.0), reverse=True)
        focus_chart_svg = _contact_sheet_svg_single_series(
            "Focus score",
            [str(item.get("camera_label") or item.get("clip_id") or "") for item in sorted_focus_rows],
            [float(item.get("composite_focus_score", 0.0) or 0.0) for item in sorted_focus_rows],
            stroke="#7c3aed",
            units="score",
            goal=None,
        )
    entries = [dict(item) for item in list(view_model.get("entries") or [])]
    detail_entries = [dict(item) for item in list(view_model.get("detail_entries") or [])]
    reference_candidate = dict(view_model.get("reference_candidate") or {})
    run_assessment = dict(payload.get("run_assessment") or {})

    def _fmt_ire(value: object, *, unresolved: str = "Unresolved") -> str:
        numeric = _numeric_value_or_none(value)
        return f"{numeric:.1f} IRE" if numeric is not None else unresolved

    def _fmt_ire_number(value: object, *, unresolved: str = "Unresolved") -> str:
        numeric = _numeric_value_or_none(value)
        return f"{numeric:.1f}" if numeric is not None else unresolved

    def _fmt_log2(value: object, *, unresolved: str = "Unresolved") -> str:
        numeric = _numeric_value_or_none(value)
        return f"{numeric:+.3f} log2" if numeric is not None else unresolved

    def _fmt_confidence(value: object) -> str:
        numeric = _numeric_value_or_none(value)
        return f"{numeric:.2f}" if numeric is not None else "n/a"

    def _fmt_stops(value: object, *, unresolved: str = "Unresolved") -> str:
        numeric = _numeric_value_or_none(value)
        return f"{numeric:+.2f}" if numeric is not None else unresolved

    def _fmt_residual(value: object, *, unresolved: str = "Unresolved") -> str:
        numeric = _numeric_value_or_none(value)
        return f"{numeric:.3f} stops" if numeric is not None else unresolved

    def _fmt_method(value: object) -> str:
        text = str(value or "").strip()
        return text.replace("_", " ").title() if text else "Unavailable"

    def _delta_class(value: Optional[float]) -> str:
        if value is None:
            return ""
        if value > 0:
            return "delta-pos"
        if value < 0:
            return "delta-neg"
        return "delta-neutral"

    def _status_dot_class(entry: Dict[str, object]) -> str:
        if bool(entry.get("is_anchor_reference")):
            return "anchor"
        if not bool(entry.get("measurement_valid")):
            return "alert"
        if str(entry.get("attention_class") or "") == "outlier":
            return "alert"
        if str(entry.get("attention_class") or "") == "borderline":
            return "review"
        return "good"

    def _overview_role(entry: Dict[str, object]) -> str:
        if bool(entry.get("is_anchor_reference")):
            return "Exposure Anchor"
        return _contact_sheet_reference_role_label(entry.get("reference_use") or "Included")

    valid_display_ires = [
        float(item.get("display_scalar_ire"))
        for item in entries
        if bool(item.get("measurement_valid")) and item.get("display_scalar_ire") is not None
    ]
    target_ire = _numeric_value_or_none(reference_candidate.get("display_scalar_ire"))
    if target_ire is None and valid_display_ires:
        target_ire = float(np.median(np.asarray(valid_display_ires, dtype=np.float32)))
    anchor_camera_label = str(reference_candidate.get("camera_label") or reference_candidate.get("clip_id") or "Derived cluster center")

    valid_count = sum(1 for item in entries if bool(item.get("measurement_valid")))
    largest_offset_entry = max(
        entries,
        key=lambda item: abs(float(item.get("exposure_adjust_stops", 0.0) or 0.0)),
        default={},
    )
    wb_model_label = str((payload.get("white_balance_model") or {}).get("model_label") or "n/a")
    anchor_original_wb = dict(reference_candidate.get("original_wb") or {})
    as_shot_kelvin = anchor_original_wb.get("as_shot_kelvin")
    as_shot_tint = anchor_original_wb.get("as_shot_tint")
    as_shot_summary = (
        "Unavailable"
        if as_shot_kelvin is None
        else f"{int(as_shot_kelvin)}K / {float(as_shot_tint or 0.0):+.1f}"
    )
    result_summary = str(run_assessment.get("run_status") or ("READY" if bool(view_model.get("goal_achieved")) else "REVIEW")).replace("_", " ").upper()
    result_note = str(run_assessment.get("operator_recommendation") or gray_target_consistency.get("summary") or "Review the retained cluster before commit.")
    hero_model_line = "Hero-center measurement model"
    metadata_run_label = str(timestamp_label or payload.get("run_label") or view_model.get("batch_label") or "Calibration run")

    page_markup: List[str] = []
    debug_rows: List[Dict[str, object]] = []
    table_rows: List[str] = []
    for entry in entries:
        measurement_valid = bool(entry.get("measurement_valid"))
        camera_label = str(entry.get("camera_label") or entry.get("clip_id") or "Camera")
        hero_center_ire = _numeric_value_or_none(entry.get("display_scalar_ire"))
        exposure_delta_ire = (
            hero_center_ire - target_ire
            if measurement_valid and hero_center_ire is not None and target_ire is not None
            else None
        )
        neutral_rgb = str((entry.get("original_wb") or {}).get("chroma_summary") or "Unavailable")
        solved_wb_text = f"{int(round(float(entry.get('kelvin', 0.0) or 0.0)))}K / {float(entry.get('tint', 0.0) or 0.0):+.1f}"
        row_class = "anchor-row" if bool(entry.get("is_anchor_reference")) else ""
        row_class = f"{row_class} unresolved-row".strip() if not measurement_valid else row_class
        table_rows.append(
            "<tr class='{row_class}'>"
            "<td>"
            f"<span class='dot {_status_dot_class(entry)}'></span><span class='cam'>{html.escape(camera_label)}</span>"
            "</td>"
            f"<td>{html.escape(_overview_role(entry))}</td>"
            f"<td class='col-exposure mono'>{html.escape(_fmt_ire(entry.get('display_scalar_ire')))}</td>"
            f"<td class='col-exposure mono'>{html.escape(_fmt_ire(target_ire))}</td>"
            f"<td class='col-exposure mono {_delta_class(exposure_delta_ire)}'>{html.escape(_fmt_ire(exposure_delta_ire) if exposure_delta_ire is not None else 'Unresolved')}</td>"
            f"<td class='col-wb mono'>{html.escape(neutral_rgb)}</td>"
            f"<td class='col-correction mono {_delta_class(_numeric_value_or_none(entry.get('exposure_adjust_stops')))}'>{html.escape(_fmt_stops(entry.get('exposure_adjust_stops')))}</td>"
            f"<td class='col-correction mono'>{html.escape(solved_wb_text)}</td>"
            f"<td class='col-correction mono'>{html.escape(_fmt_residual(entry.get('residual_abs_stops')))}</td>"
            "</tr>".format(row_class=html.escape(row_class))
        )

    overview_panel_bits = [
        "<div>All cameras are being compared against one exposure anchor camera.</div>",
        "<div>The cameras farthest from the anchor need exposure adjustment first.</div>",
        "<div>Each camera page shows the original frame, corrected frame, measured exposure, target exposure, and the correction needed.</div>",
        "<div>If a value feels visually wrong, inspect the solve overlay and the hero-vs-ring context on that camera page.</div>",
    ]
    overview_markup = (
        "<section class='landscape-page'>"
        "<div class='topbar'>"
        "<div>"
        f"<div class='eyebrow'>{html.escape(final_title)}</div>"
        f"<h1>{html.escape(final_title)}</h1>"
        f"<div class='run-mark'>{html.escape(str(payload.get('run_label') or view_model.get('batch_label') or 'Calibration run'))}</div>"
        f"<div class='sub'>Exposure Anchor: {html.escape(anchor_summary_display)}</div>"
        "</div>"
        "<div class='meta'>"
        "<div class='meta-grid'>"
        f"<div class='k'>Run</div><div class='v'>{html.escape(metadata_run_label)}</div>"
        f"<div class='k'>Target</div><div class='v'>{html.escape(_contact_sheet_target_class_label(gray_target_consistency.get('dominant_target_class') or 'sphere'))} (conf. {_fmt_confidence(reference_candidate.get('gray_target_confidence'))})</div>"
        f"<div class='k'>Domain</div><div class='v'>{html.escape(str(view_model.get('domain_label') or REVIEW_PREVIEW_TRANSFORM))}</div>"
        f"<div class='k'>Included</div><div class='v'>{valid_count} / {int(view_model.get('camera_count', 0) or 0)} cameras</div>"
        f"<div class='k'>Anchor</div><div class='v'>{html.escape(anchor_camera_label)}</div>"
        "</div>"
        "</div>"
        "</div>"
        "<div class='summary-cards'>"
        f"<div class='card accent-anchor'><div class='label'>Exposure Anchor</div><div class='value mono'>{html.escape(_fmt_ire(target_ire))}</div><div class='note'>Anchor camera {html.escape(anchor_camera_label)}.</div></div>"
        f"<div class='card accent-offset'><div class='label'>Largest Offset</div><div class='value mono'>{html.escape(_fmt_stops(largest_offset_entry.get('exposure_adjust_stops')))}</div><div class='note'>{html.escape(str(largest_offset_entry.get('camera_label') or 'n/a'))} is farthest from the anchor.</div></div>"
        f"<div class='card accent-model'><div class='label'>White Balance Model</div><div class='value wb-value'>{html.escape(wb_model_label)}</div><div class='note'>{html.escape(str((payload.get('white_balance_model') or {}).get('selection_reason') or 'Stored white-balance model for the retained array.'))}</div></div>"
        f"<div class='card accent-shot'><div class='label'>As-shot</div><div class='value mono'>{html.escape(as_shot_summary)}</div><div class='note'>Reference camera as-shot white balance.</div></div>"
        f"<div class='card accent-result'><div class='label'>Result</div><div class='value'>{html.escape(result_summary)}</div><div class='note'>{html.escape(result_note)}</div></div>"
        "</div>"
        "<div class='section-title'>Array Calibration Summary</div>"
        "<div class='table-shell'>"
        "<table class='summary-table'>"
        "<thead>"
        "<tr class='group'>"
        "<th colspan='2'>Camera</th>"
        "<th colspan='3'>Exposure</th>"
        "<th colspan='1'>Original Neutral RGB</th>"
        "<th colspan='3'>Correction to Target</th>"
        "</tr>"
        "<tr>"
        "<th>Camera</th>"
        "<th>Role</th>"
        "<th class='col-exposure'>Hero Center IRE</th>"
        "<th class='col-exposure'>Target IRE</th>"
        "<th class='col-exposure'>Delta</th>"
        "<th class='col-wb'>Neutral RGB</th>"
        "<th class='col-correction'>Exposure Adj.</th>"
        "<th class='col-correction'>Kelvin / Tint</th>"
        "<th class='col-correction'>Residual</th>"
        "</tr>"
        "</thead>"
        "<tbody>"
        + "".join(table_rows)
        + "</tbody>"
        "</table>"
        "<div class='legend'>"
        "<span><span class='dot anchor'></span>Anchor</span>"
        "<span><span class='dot good'></span>Retained</span>"
        "<span><span class='dot review'></span>Review</span>"
        "<span><span class='dot alert'></span>Unresolved / Excluded</span>"
        "</div>"
        "</div>"
        "<div class='two-col'>"
        f"<div class='panel chart-panel'><div class='section-title subtle'>Exposure Spread</div>{exposure_chart_svg}</div>"
        "<div class='panel side-panel'>"
        "<div class='section-title subtle'>HOW TO READ THIS REPORT</div>"
        + "".join(overview_panel_bits)
        + "</div>"
        "</div>"
        "</section>"
    )
    page_markup.append(overview_markup)

    for index, entry in enumerate(detail_entries, start=1):
        clip_id = str(entry.get("clip_id") or "")
        camera_label = str(entry.get("camera_label") or clip_id or f"Camera {index}")
        original_source = _contact_sheet_required_asset(entry.get("original_image"), label="original frame", clip_id=clip_id)
        corrected_source = _contact_sheet_required_asset(entry.get("corrected_image"), label="corrected frame", clip_id=clip_id)
        overlay_source = _contact_sheet_required_asset(entry.get("overlay_image"), label="sphere mask overlay", clip_id=clip_id)
        original_src = _contact_sheet_image_src(original_source, html_path=html_path)
        corrected_src = _contact_sheet_image_src(corrected_source, html_path=html_path)
        overlay_src = _contact_sheet_image_src(overlay_source, html_path=html_path)
        if images_dir is not None:
            asset_stem = _contact_sheet_sanitized_stem(camera_label or clip_id or f"camera_{index}")
            original_info = _contact_sheet_write_report_image(original_source, output_path=images_dir / f"{asset_stem}_original.jpg")
            corrected_info = _contact_sheet_write_report_image(corrected_source, output_path=images_dir / f"{asset_stem}_corrected.jpg")
            overlay_info = _contact_sheet_write_report_image(overlay_source, output_path=images_dir / f"{asset_stem}_mask.jpg")
            original_src = _contact_sheet_image_src(original_info["output_path"], html_path=str(html_file))
            corrected_src = _contact_sheet_image_src(corrected_info["output_path"], html_path=str(html_file))
            overlay_src = _contact_sheet_image_src(overlay_info["output_path"], html_path=str(html_file))
        else:
            original_info = {"source_path": original_source, "output_path": original_source}
            corrected_info = {"source_path": corrected_source, "output_path": corrected_source}
            overlay_info = {"source_path": overlay_source, "output_path": overlay_source}
        measurement_valid = bool(entry.get("measurement_valid"))
        sample_1_ire = _numeric_value_or_none(entry.get("sample_1_ire"))
        sample_2_ire = _numeric_value_or_none(entry.get("sample_2_ire"))
        sample_3_ire = _numeric_value_or_none(entry.get("sample_3_ire"))
        scalar_log2 = _numeric_value_or_none(entry.get("display_scalar_log2"))
        hero_center_ire = _numeric_value_or_none(entry.get("display_scalar_ire"))
        ring_values = [value for value in (sample_1_ire, sample_3_ire) if value is not None]
        ring_average_ire = float(sum(ring_values) / len(ring_values)) if ring_values else None
        ring_low_high = (
            f"{min(ring_values):.1f}–{max(ring_values):.1f} IRE"
            if len(ring_values) == 2
            else "Unavailable"
        )
        flags: List[str] = []
        if bool(entry.get("fallback_used")):
            flags.append("Alternate detection path used")
        if str(entry.get("reference_use") or "Included") != "Included":
            flags.append(_contact_sheet_reference_role_label(str(entry.get("reference_use") or "Excluded")))
        if float(entry.get("residual_abs_stops", 0.0) or 0.0) > 0.02:
            flags.append("Residual exceeds ±0.02 stops")
        sample_range = (
            max(sample_1_ire, sample_2_ire, sample_3_ire) - min(sample_1_ire, sample_2_ire, sample_3_ire)
            if measurement_valid and None not in (sample_1_ire, sample_2_ire, sample_3_ire)
            else 0.0
        )
        if measurement_valid and sample_range > 3.0:
            flags.append("Sample profile varies across the sphere")
        if not measurement_valid:
            flags.append("Gray target unresolved")
        if not flags:
            flags.append("No anomaly flags")
        original_wb = dict(entry.get("original_wb") or {})
        as_shot_kelvin = original_wb.get("as_shot_kelvin")
        as_shot_tint = original_wb.get("as_shot_tint")
        as_shot_kelvin_text = (
            "Unavailable"
            if as_shot_kelvin is None
            else f"{int(as_shot_kelvin)}K"
        )
        as_shot_tint_text = (
            "Unavailable"
            if as_shot_tint is None
            else f"{float(as_shot_tint or 0.0):+.1f}"
        )
        header_right = _contact_sheet_join_bits(
            [
                str(entry.get("status") or "REVIEW"),
                str(entry.get("operator_result_label") or ""),
                f"Residual {float(entry.get('residual_abs_stops', 0.0) or 0.0):.3f} stops",
            ]
        )
        rejection_reasons = list(entry.get("rejection_reasons") or [])
        offset_for_anchor = float(entry.get("offset_to_anchor", 0.0) or 0.0)
        offset_percent = max(0.0, min(100.0, 50.0 + ((offset_for_anchor / 0.25) * 40.0)))
        offset_width = min(40.0, abs(offset_for_anchor / 0.25) * 40.0)
        exp_bar_left = 50.0 if offset_for_anchor >= 0 else max(10.0, 50.0 - offset_width)
        exp_bar_color = "#d4880e" if abs(offset_for_anchor) > 0.10 else "#25a05a"
        wb_chart_svg = str(original_wb.get("chromaticity_chart_svg") or "").strip()
        detection_summary = (
            f"{str(entry.get('detection_status') or 'FAILED')} / {_fmt_method(entry.get('detection_method') or 'failed')}"
        )
        rejection_summary = (
            " | ".join(_sphere_detection_rejection_reason_label(item) for item in rejection_reasons[:3])
            if rejection_reasons
            else str(entry.get("sphere_detection_note") or "Sphere check verified")
        )
        notes_items = [
            ("Result", str(entry.get("operator_result_label") or entry.get("status") or "Review")),
            ("Detection", detection_summary),
            ("Measurement source", _contact_sheet_measurement_source_label(entry.get("display_scalar_source") or "stored measurement payload")),
            ("Notes", " | ".join(_contact_sheet_flag_label(flag) for flag in flags)),
        ]
        page_markup.append(
            "<section class='landscape-page camera-page'>"
            "<div class='hd'>"
            "<div>"
            f"<div class='hd-brand'>{html.escape(final_title)} · {html.escape(str(payload.get('run_label') or view_model.get('batch_label') or 'Calibration run'))}</div>"
            + (
                "<div class='anchor-flag hero-badge'>Anchor / Hero</div>"
                if bool(entry.get("is_anchor_reference")) and bool(entry.get("is_hero_camera"))
                else "<div class='anchor-flag hero-badge'>Exposure Anchor</div>"
                if bool(entry.get("is_anchor_reference"))
                else "<div class='anchor-flag hero-badge'>Hero Camera</div>"
                if bool(entry.get("is_hero_camera"))
                else ""
            )
            + f"<div class='hd-cam'>{html.escape(camera_label)}</div>"
            + f"<div class='hd-clip'>{html.escape(clip_id)}</div>"
            + f"<div class='hd-project'>{html.escape(str(view_model.get('domain_label') or REVIEW_PREVIEW_TRANSFORM))}<br>Exposure anchor: {html.escape(anchor_summary_display)}</div>"
            + "</div>"
            + "<div class='hd-right'>"
            + f"<div class='hd-run mono'>{html.escape(str(entry.get('operator_result_label') or entry.get('status') or 'Review'))}</div>"
            + (f"<div class='hd-meta'>Hero Camera</div>" if bool(entry.get("is_hero_camera")) else "")
            + f"<div class='hd-meta'>{html.escape(header_right)}</div>"
            + f"<div class='hd-meta'>Recommended action: {html.escape(str(entry.get('recommended_action') or 'Review'))}</div>"
            + "</div>"
            + "</div>"
            + "<div class='img-row'>"
            + f"<div class='img-panel'><div class='img-label'>ORIGINAL + SOLVE OVERLAY</div><div class='img-frame'><img class='img-box' src='{html.escape(overlay_src)}' alt='{html.escape(clip_id)} original with solve overlay'></div></div>"
            + f"<div class='img-panel'><div class='img-label'>CORRECTED FRAME</div><div class='img-frame'><img class='img-box' src='{html.escape(corrected_src)}' alt='{html.escape(clip_id)} corrected'></div></div>"
            + "</div>"
            + "<div class='metrics-band'>"
            + "<div class='metrics'>"
            + "<div class='metric-col section-exposure'>"
            + "<div class='mc-title'>Gray Exposure</div>"
            + f"<div class='mc-row'><span class='mc-key'>Hero Center IRE</span><span class='mc-val large {_delta_class(_numeric_value_or_none(entry.get('exposure_adjust_stops')))} mono'>{html.escape(_fmt_ire_number(hero_center_ire))}</span></div>"
            + f"<div class='mc-row'><span class='mc-key'>Target IRE</span><span class='mc-val mono'>{html.escape(_fmt_ire_number(target_ire))}</span></div>"
            + f"<div class='mc-row'><span class='mc-key'>Original → Target</span><span class='mc-val mono'>{html.escape(f'{_fmt_ire_number(hero_center_ire)} → {_fmt_ire_number(target_ire)}')}</span></div>"
            + "<div class='mc-spacer'></div>"
            + "<div class='exp-vis'>"
            + "<div class='exp-label'><span>Offset from anchor</span><span>±0.25 stops</span></div>"
            + "<div class='exp-track'>"
            + "<div class='exp-zone' style='left:40%; width:20%;'></div>"
            + "<div class='exp-center-line'></div>"
            + f"<div class='exp-bar' style='left:{exp_bar_left:.1f}%; width:{max(offset_width, 1.2):.1f}%; background:{exp_bar_color};'></div>"
            + "</div>"
            + "<div class='exp-tick-labels'><span>−0.25</span><span>−0.10</span><span>0</span><span>+0.10</span><span>+0.25</span></div>"
            + "</div>"
            + f"<div class='metric-foot mono'>Scalar {html.escape(_fmt_log2(scalar_log2))} · authoritative hero-center measurement</div>"
            + "</div>"
            + "<div class='metric-col section-wb'>"
            + "<div class='mc-title'>White Balance</div>"
            + f"<div class='mc-row'><span class='mc-key'>As-shot</span><span class='mc-val mono'>{html.escape(as_shot_kelvin_text)} / {html.escape(as_shot_tint_text)}</span></div>"
            + f"<div class='mc-row'><span class='mc-key'>Solved Kelvin</span><span class='mc-val mono'>{int(round(float(entry.get('kelvin', 0.0) or 0.0)))}K</span></div>"
            + f"<div class='mc-row'><span class='mc-key'>Tint</span><span class='mc-val mono'>{float(entry.get('tint', 0.0) or 0.0):+.1f}</span></div>"
            + f"<div class='mc-row'><span class='mc-key'>Gray target</span><span class='mc-val'>{html.escape(str(original_wb.get('gray_target_label') or 'Gray sphere'))}</span></div>"
            + f"<div class='mc-row'><span class='mc-key'>Confidence</span><span class='mc-val mono'>{html.escape(_fmt_confidence(entry.get('gray_target_confidence')))}</span></div>"
            + (f"<div class='wb-vis'>{wb_chart_svg}</div>" if wb_chart_svg else "")
            + f"<div class='metric-foot mono'>Neutral RGB · {html.escape(str(original_wb.get('chroma_summary') or 'Unavailable'))}</div>"
            + "</div>"
            + "<div class='metric-col section-correction'>"
            + "<div class='mc-title'>Correction Applied</div>"
            + f"<div class='mc-row'><span class='mc-key'>Exposure adj.</span><span class='mc-val mono {_delta_class(_numeric_value_or_none(entry.get('exposure_adjust_stops')))}'>{html.escape(_contact_sheet_exposure_direction_label(float(entry.get('exposure_adjust_stops', 0.0) or 0.0)))}</span></div>"
            + f"<div class='mc-row'><span class='mc-key'>Original → Target</span><span class='mc-val mono'>{html.escape(f'{_fmt_ire_number(hero_center_ire)} → {_fmt_ire_number(target_ire)}')}</span></div>"
            + f"<div class='mc-row'><span class='mc-key'>Residual</span><span class='mc-val mono'>{html.escape(_fmt_residual(entry.get('residual_abs_stops')))}</span></div>"
            + f"<div class='mc-row'><span class='mc-key'>Array role</span><span class='mc-val'>{html.escape(_contact_sheet_reference_role_label(entry.get('reference_use') or 'Included'))}</span></div>"
            + f"<div class='mc-row'><span class='mc-key'>White balance model</span><span class='mc-val'>{html.escape(wb_model_label)}</span></div>"
            + "</div>"
            + "<div class='metric-col section-solve'>"
            + "<div class='mc-title'>Sphere Solve</div>"
            + f"<div class='mc-row'><span class='mc-key'>Result</span><span class='mc-val'>{html.escape(str(entry.get('operator_result_label') or entry.get('status') or 'Review'))}</span></div>"
            + f"<div class='mc-row'><span class='mc-key'>Detection</span><span class='mc-val'>{html.escape(str(entry.get('detection_status') or 'FAILED'))}</span></div>"
            + f"<div class='mc-row'><span class='mc-key'>Method</span><span class='mc-val'>{html.escape(_fmt_method(entry.get('detection_method') or 'failed'))}</span></div>"
            + f"<div class='mc-row'><span class='mc-key'>Measurement source</span><span class='mc-val'>Hero-center patch</span></div>"
            + f"<div class='mc-row'><span class='mc-key'>Hero vs Ring</span><span class='mc-val mono'>{html.escape('Unavailable' if hero_center_ire is None or ring_average_ire is None else f'{hero_center_ire:.1f} vs {ring_average_ire:.1f} IRE')}</span></div>"
            + f"<div class='mc-row'><span class='mc-key'>Ring low–high</span><span class='mc-val mono'>{html.escape(ring_low_high)}</span></div>"
            + "</div>"
            + "</div>"
            + "</div>"
            + "<div class='notes-strip'><div class='notes-inner'>"
            + "".join(
                f"<div class='note-item'><div class='note-key'>{html.escape(key)}</div><div class='note-val'>{html.escape(value)}</div></div>"
                for key, value in notes_items
            )
            + "</div></div>"
            + "<div class='pg-footer'>"
            + f"<span>{html.escape(str(entry.get('measurement_domain') or REVIEW_PREVIEW_TRANSFORM))}</span>"
            + f"<span>{html.escape(rejection_summary)}</span>"
            + f"<span>Page {index + 1}</span>"
            + "</div>"
            + "</section>"
        )
        debug_rows.append(
            {
                "clip_id": clip_id,
                "camera_label": camera_label,
                "validation_status": str(entry.get("status") or "REVIEW"),
                "resolved_asset_paths": {
                    "original": str(original_info.get("output_path") or original_source),
                    "corrected": str(corrected_info.get("output_path") or corrected_source),
                    "mask": str(overlay_info.get("output_path") or overlay_source),
                },
                "measurement_values": {
                    "measurement_valid": measurement_valid,
                    "hero_center_ire": hero_center_ire,
                    "target_ire": target_ire,
                    "sample_1_ire": sample_1_ire,
                    "sample_2_ire": sample_2_ire,
                    "sample_3_ire": sample_3_ire,
                    "target_sample_label": str(entry.get("target_sample_label") or CONTACT_SHEET_TARGET_SAMPLE_LABEL),
                    "display_scalar_log2": scalar_log2,
                    "exposure_adjust_stops": float(entry.get("exposure_adjust_stops", 0.0) or 0.0),
                },
                "original_wb": original_wb,
                "fallback_used": bool(entry.get("fallback_used")),
            }
        )
    markup = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(final_title)}</title>
  <style>
    :root {{
      --bg:#d2d8df;
      --paper:#f3f4f6;
      --card:#ffffff;
      --ink:#101828;
      --muted:#667085;
      --muted-2:#475467;
      --line:#d8dde3;
      --line-2:#d0d5dd;
      --blue-weak:#f3f7fc;
      --green-weak:#f4faf6;
      --amber-weak:#fdf8f1;
      --violet-weak:#f7f5fb;
      --good:#16a34a;
      --review:#f59e0b;
      --alert:#dc2626;
      --anchor:#2563eb;
      --surface:#f5f5f3;
      --surface-2:#ebebea;
      --good-bg:#eef9f1;
      --review-bg:#fff8eb;
      --anchor-bg:#eef4ff;
      --mono: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      --sans: Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
    }}
    * {{ box-sizing:border-box; }}
    body {{
      margin:0;
      background:var(--bg);
      color:var(--ink);
      font-family:var(--sans);
      -webkit-print-color-adjust: exact;
      print-color-adjust: exact;
    }}
    .mono {{
      font-variant-numeric: tabular-nums;
      font-feature-settings:"tnum" 1;
      font-family:var(--mono);
    }}
    .landscape-page {{
      width:11in;
      min-height:8.5in;
      margin:16px auto;
      background:var(--paper);
      padding:22px 24px 18px;
      box-shadow:0 8px 30px rgba(16,24,40,.10);
      page-break-after:always;
      position:relative;
    }}
    .eyebrow {{
      font-size:12px;
      font-weight:800;
      letter-spacing:.08em;
      text-transform:uppercase;
      color:var(--muted);
      margin-bottom:10px;
    }}
    h1 {{
      margin:0;
      font-size:36px;
      line-height:1;
      letter-spacing:-.03em;
    }}
    .run-mark {{
      margin-top:18px;
      font-size:24px;
      font-weight:800;
      letter-spacing:-.02em;
      color:var(--ink);
    }}
    .sub {{
      margin-top:8px;
      font-size:15px;
      font-weight:600;
      color:var(--muted-2);
      line-height:1.45;
    }}
    .topbar {{
      display:flex;
      justify-content:space-between;
      gap:20px;
      align-items:flex-start;
      margin-bottom:20px;
    }}
    .meta {{
      min-width:300px;
      background:var(--card);
      border:1px solid var(--line);
      border-radius:16px;
      padding:14px 16px;
      box-shadow:0 4px 14px rgba(19,35,63,.06);
    }}
    .meta-grid {{
      display:grid;
      grid-template-columns:120px 1fr;
      gap:7px 12px;
      font-size:13px;
    }}
    .meta-grid .k {{ color:var(--muted); font-weight:600; }}
    .meta-grid .v {{ font-weight:700; }}
    .summary-cards {{
      display:grid;
      grid-template-columns:1.15fr 1fr 1fr 1fr 1fr;
      gap:12px;
      margin-bottom:18px;
    }}
    .card {{
      background:var(--card);
      border:1px solid var(--line);
      border-radius:16px;
      padding:14px 16px 16px;
      position:relative;
      overflow:hidden;
      box-shadow:0 4px 14px rgba(19,35,63,.06);
    }}
    .card::before {{
      content:"";
      position:absolute;
      top:0;
      left:0;
      right:0;
      height:4px;
      background:#d6deea;
    }}
    .card.accent-anchor::before {{ background:linear-gradient(90deg,#2563eb,#8cb6ff); }}
    .card.accent-offset::before {{ background:linear-gradient(90deg,#f59e0b,#ffd48a); }}
    .card.accent-model::before {{ background:linear-gradient(90deg,#0f766e,#7ed1c4); }}
    .card.accent-shot::before {{ background:linear-gradient(90deg,#64748b,#c4ceda); }}
    .card.accent-result::before {{ background:linear-gradient(90deg,#16a34a,#9be7ae); }}
    .card .label {{
      font-size:11px;
      letter-spacing:.08em;
      text-transform:uppercase;
      color:var(--muted);
      font-weight:800;
      margin-bottom:10px;
    }}
    .card .value {{
      font-size:32px;
      font-weight:800;
      letter-spacing:-.03em;
      line-height:1;
      color:var(--ink);
    }}
    .card .value.wb-value {{
      font-size:22px;
      line-height:1.15;
    }}
    .card .note {{
      margin-top:8px;
      font-size:13px;
      color:var(--muted-2);
      line-height:1.4;
    }}
    .section-title {{
      margin:18px 0 10px;
      font-size:14px;
      font-weight:900;
      letter-spacing:.06em;
      text-transform:uppercase;
      color:#344054;
    }}
    .section-title.subtle {{
      margin:0 0 10px;
      font-size:12px;
    }}
    .table-shell {{
      background:var(--card);
      border:1px solid var(--line);
      border-radius:16px;
      overflow:hidden;
      box-shadow:0 4px 14px rgba(19,35,63,.06);
    }}
    table {{
      width:100%;
      border-collapse:collapse;
      font-size:13px;
    }}
    thead th {{
      padding:10px 10px;
      border-bottom:1px solid var(--line);
      color:#344054;
      font-size:11px;
      font-weight:900;
      letter-spacing:.05em;
      text-transform:uppercase;
      text-align:left;
      vertical-align:bottom;
    }}
    thead tr.group th {{
      font-size:10px;
      color:#667085;
      background:#f9fafb;
    }}
    tbody td {{
      padding:10px 10px;
      border-top:1px solid #e8ebef;
      vertical-align:middle;
    }}
    tbody tr:nth-child(even) {{ background:#fbfcfd; }}
    tbody tr.anchor-row {{ box-shadow: inset 4px 0 0 var(--anchor); }}
    tbody tr.unresolved-row {{ opacity:0.92; }}
    .col-exposure {{ background:var(--blue-weak); }}
    .col-wb {{ background:var(--green-weak); }}
    .col-correction {{ background:var(--amber-weak); }}
    .cam {{
      font-weight:800;
      letter-spacing:-.01em;
    }}
    .dot {{
      display:inline-block;
      width:9px;
      height:9px;
      border-radius:50%;
      margin-right:8px;
      vertical-align:middle;
    }}
    .good {{ background:var(--good); }}
    .review {{ background:var(--review); }}
    .alert {{ background:var(--alert); }}
    .anchor {{ background:var(--anchor); }}
    .delta-pos {{ color:var(--alert); font-weight:800; }}
    .delta-neg {{ color:var(--good); font-weight:800; }}
    .delta-neutral {{ color:var(--ink); font-weight:800; }}
    .legend {{
      display:flex;
      gap:24px;
      flex-wrap:wrap;
      padding:12px 16px;
      border-top:1px solid var(--line);
      font-size:12px;
      color:var(--muted-2);
      background:#fafbfc;
    }}
    .two-col {{
      display:grid;
      grid-template-columns:2.3fr 1fr;
      gap:12px;
      margin-top:14px;
      align-items:start;
    }}
    .panel {{
      background:var(--card);
      border:1px solid var(--line);
      border-radius:16px;
      padding:14px 15px;
      box-shadow:0 4px 14px rgba(19,35,63,.06);
    }}
    .chart-panel svg {{
      width:100%;
      height:auto;
      display:block;
    }}
    .side-panel {{
      font-size:13px;
      color:var(--muted-2);
      line-height:1.55;
    }}
    .side-panel div {{
      margin-bottom:10px;
    }}
    .camera-page {{
      min-height:8.5in;
    }}
    .hd {{
      display:grid;
      grid-template-columns:1fr auto;
      gap:0 2rem;
      align-items:start;
      padding-bottom:12px;
      border-bottom:1px solid var(--line-2);
      margin-bottom:16px;
    }}
    .hd-brand {{
      font-family:var(--mono);
      font-size:10px;
      font-weight:500;
      letter-spacing:.12em;
      text-transform:uppercase;
      color:var(--muted);
      margin-bottom:6px;
    }}
    .hd-cam {{
      font-family:var(--mono);
      font-size:26px;
      font-weight:500;
      color:var(--ink);
      letter-spacing:-.01em;
      line-height:1;
      margin-bottom:4px;
    }}
    .hd-clip {{
      font-family:var(--mono);
      font-size:11px;
      color:var(--muted);
      letter-spacing:.02em;
      margin-bottom:10px;
    }}
    .hd-project {{
      font-size:11px;
      color:var(--muted-2);
      line-height:1.45;
    }}
    .hd-right {{
      text-align:right;
      padding-top:6px;
    }}
    .hd-run {{
      font-size:18px;
      font-weight:700;
      color:var(--ink);
      margin-bottom:6px;
    }}
    .hd-meta {{
      font-size:11px;
      color:var(--muted-2);
      line-height:1.5;
    }}
    .anchor-flag {{
      display:inline-flex;
      align-items:center;
      gap:5px;
      font-family:var(--mono);
      font-size:9px;
      font-weight:500;
      letter-spacing:.08em;
      text-transform:uppercase;
      color:#1a3d8b;
      background:#eff5ff;
      border:1px solid #93b8f5;
      border-radius:3px;
      padding:3px 8px;
      margin-bottom:6px;
    }}
    .img-row {{
      display:grid;
      grid-template-columns:1fr 1fr;
      gap:16px;
      margin-bottom:12px;
    }}
    .img-panel {{
      display:flex;
      flex-direction:column;
      gap:6px;
    }}
    .img-label {{
      font-family:var(--mono);
      font-size:10px;
      font-weight:500;
      letter-spacing:.12em;
      text-transform:uppercase;
      color:var(--muted);
    }}
    .img-frame {{
      width:100%;
      aspect-ratio:16 / 9;
      overflow:hidden;
      border:1px solid var(--line);
      border-radius:10px;
      background:#dce3ec;
    }}
    .img-box {{
      width:100%;
      height:100%;
      object-fit:cover;
      object-position:center;
      background:#dce3ec;
      display:block;
    }}
    .metrics-band {{
      background:rgba(255,255,255,.82);
      border:1px solid var(--line);
      border-radius:16px;
      padding:8px 12px;
      margin-bottom:14px;
      box-shadow:0 4px 14px rgba(19,35,63,.06);
    }}
    .metrics {{
      display:grid;
      grid-template-columns:repeat(4, 1fr);
      gap:0;
      margin-bottom:0;
    }}
    .metric-col {{
      background:transparent;
      border-right:1px solid var(--line);
      padding:8px 14px 6px;
      min-height:176px;
    }}
    .metric-col:last-child {{ border-right:none; }}
    .mc-title {{
      font-family:var(--mono);
      font-size:10px;
      font-weight:500;
      letter-spacing:.12em;
      text-transform:uppercase;
      color:var(--muted);
      margin-bottom:8px;
    }}
    .mc-row {{
      display:flex;
      justify-content:space-between;
      gap:10px;
      margin-bottom:5px;
      align-items:baseline;
    }}
    .mc-key {{
      font-size:11px;
      color:var(--muted-2);
      line-height:1.35;
    }}
    .mc-val {{
      font-size:13px;
      font-weight:700;
      color:var(--ink);
      text-align:right;
      line-height:1.35;
    }}
    .mc-val.large {{
      font-size:28px;
      line-height:1;
      letter-spacing:-.02em;
    }}
    .section-correction .mc-row:first-of-type .mc-val {{
      font-size:15px;
      font-weight:800;
    }}
    .section-solve .mc-title,
    .section-solve .mc-val,
    .section-solve .mc-key {{
      opacity:.92;
    }}
    .mc-spacer {{
      height:6px;
    }}
    .metric-foot {{
      margin-top:6px;
      font-size:9px;
      color:#9b9fa9;
      line-height:1.5;
    }}
    .exp-vis {{
      margin-top:2px;
    }}
    .exp-label {{
      display:flex;
      justify-content:space-between;
      gap:8px;
      font-size:9px;
      color:#9b9fa9;
      margin-bottom:4px;
    }}
    .exp-track {{
      position:relative;
      height:20px;
      background:#fff;
      border:1px solid #e2e2df;
      border-radius:2px;
    }}
    .exp-zone {{
      position:absolute;
      top:0;
      bottom:0;
      background:rgba(37,160,90,.12);
    }}
    .exp-center-line {{
      position:absolute;
      left:50%;
      top:-3px;
      bottom:-3px;
      width:1px;
      background:#6b6f7c;
    }}
    .exp-bar {{
      position:absolute;
      top:3px;
      bottom:3px;
      border-radius:2px;
    }}
    .exp-tick-labels {{
      display:flex;
      justify-content:space-between;
      margin-top:3px;
      font-size:7.5px;
      color:#9b9fa9;
    }}
    .wb-vis {{
      margin-top:6px;
      border:1px solid var(--line);
      border-radius:4px;
      padding:5px;
      background:#fff;
    }}
    .wb-vis svg {{
      width:100%;
      height:auto;
      display:block;
    }}
    .notes-strip {{
      background:var(--surface);
      border:1px solid var(--line);
      border-radius:12px;
      padding:8px 12px;
      margin-bottom:14px;
    }}
    .notes-inner {{
      display:flex;
      gap:28px;
      flex-wrap:wrap;
    }}
    .note-key {{
      font-family:var(--mono);
      font-size:9px;
      font-weight:500;
      letter-spacing:.08em;
      text-transform:uppercase;
      color:#9b9fa9;
      margin-bottom:2px;
    }}
    .note-val {{
      font-size:11px;
      color:#3a3d47;
      line-height:1.4;
      max-width:220px;
    }}
    .pg-footer {{
      position:absolute;
      bottom:0.2in;
      left:26px;
      right:26px;
      display:flex;
      justify-content:space-between;
      align-items:center;
      border-top:1px solid var(--line);
      padding-top:8px;
      font-family:var(--mono);
      font-size:9px;
      color:#9b9fa9;
      letter-spacing:.05em;
      gap:10px;
    }}
    @media print {{
      body {{ background:#fff; }}
      .landscape-page {{
        box-shadow:none;
        margin:0;
      }}
    }}
    @page {{ size: letter landscape; margin:0; }}
  </style>
</head>
<body>
  {''.join(page_markup)}
</body>
</html>"""
    if html_file is not None:
        asset_validation = _validate_contact_sheet_html_assets_markup(markup, html_path=str(html_file))
        missing_assets = list(asset_validation.get("missing_assets") or [])
        if missing_assets:
            preview = "; ".join(missing_assets[:6])
            if len(missing_assets) > 6:
                preview += f"; ... and {len(missing_assets) - 6} more"
            raise RuntimeError(
                "Generated contact_sheet.html references missing image assets. "
                f"HTML: {html_file}. Missing: {preview}"
            )
        debug_payload = {
            "html_path": str(html_file),
            "resolved_asset_paths": list(asset_validation.get("resolved_assets") or []),
            "scientific_validation_path": str(payload.get("scientific_validation_path") or ""),
            "scientific_validation_markdown_path": str(payload.get("scientific_validation_markdown_path") or ""),
            "measurement_values_per_camera": debug_rows,
            "validation_status": {
                "status_counts": dict(view_model.get("status_counts") or {}),
                "all_within_tolerance": bool(view_model.get("all_within_tolerance")),
                "median_residual": float(view_model.get("median_residual", 0.0) or 0.0),
                "worst_residual": float(view_model.get("worst_residual", 0.0) or 0.0),
            },
        }
        debug_path = html_file.parent / "contact_sheet_debug.json"
        debug_path.write_text(json.dumps(debug_payload, indent=2), encoding="utf-8")
        payload["contact_sheet_debug_path"] = str(debug_path)
        payload["html_asset_validation"] = asset_validation
    return markup
