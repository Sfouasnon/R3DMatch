from __future__ import annotations

import json
import math
import os
import re
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from PIL import Image, UnidentifiedImageError

from .color import is_identity_cdl_payload
from .commit_values import build_commit_values
from .execution import raise_if_cancelled
from .ftps_ingest import DEFAULT_CAMERA_IP_MAP, normalize_source_mode, run_ftps_ingest_job, source_mode_label
from .identity import group_key_from_clip_id, inventory_camera_label_from_clip_id, inventory_camera_label_from_source_path
from .matching import analyze_path
from .progress import emit_review_progress, review_progress_path_for
from .report import (
    build_contact_sheet_report,
    build_review_package,
    clear_preview_cache,
    normalize_review_mode,
    render_contact_sheet_pdf,
    review_mode_label,
)
from .rmd import write_rmd_for_clip_with_metadata


class ReviewValidationError(RuntimeError):
    pass


LOG3G10_MID_GRAY_CODE_VALUE = 0.3333
LOG3G10_LIN_SLOPE = 155.975327
LOG3G10_LIN_OFFSET = 0.01
LOG3G10_LOG_SLOPE = 0.224282

PHYSICAL_MEAN_EXPOSURE_ERROR_THRESHOLD = 0.02
PHYSICAL_MAX_NEUTRAL_ERROR_THRESHOLD = 0.02
PHYSICAL_EXTREME_EXPOSURE_ERROR_THRESHOLD = 0.03
PHYSICAL_EXTREME_NEUTRAL_ERROR_THRESHOLD = 0.025
PHYSICAL_MIN_CONFIDENCE_THRESHOLD = 0.35
PHYSICAL_KELVIN_STDDEV_THRESHOLD = 150.0
PHYSICAL_TINT_VARIATION_THRESHOLD = 0.1
PHYSICAL_LOG2_SPREAD_REFERENCE = 0.25
PHYSICAL_CHROMA_SPREAD_REFERENCE = 0.01
PHYSICAL_ROI_VARIANCE_REFERENCE = 0.001
PHYSICAL_MAX_EXCLUDED_CAMERA_FRACTION = 0.25
PHYSICAL_MIN_USABLE_CLUSTER_CAMERAS = 3


def review_report_root(analysis_dir: str | Path) -> Path:
    return Path(analysis_dir).expanduser().resolve() / "report"


def review_payload_path_for(analysis_dir: str | Path) -> Path:
    return review_report_root(analysis_dir) / "contact_sheet.json"


def review_html_path_for(analysis_dir: str | Path) -> Path:
    return review_report_root(analysis_dir) / "contact_sheet.html"


def review_pdf_path_for(analysis_dir: str | Path) -> Path:
    return review_report_root(analysis_dir) / "preview_contact_sheet.pdf"


def review_manifest_path_for(analysis_dir: str | Path) -> Path:
    return review_report_root(analysis_dir) / "review_manifest.json"


def review_package_path_for(analysis_dir: str | Path) -> Path:
    return review_report_root(analysis_dir) / "review_package.json"


def review_validation_path_for(analysis_dir: str | Path) -> Path:
    return review_report_root(analysis_dir) / "review_validation.json"


def review_preview_commands_path_for(analysis_dir: str | Path) -> Path:
    return Path(analysis_dir).expanduser().resolve() / "previews" / "preview_commands.json"


def review_commit_payload_path_for(analysis_dir: str | Path) -> Path:
    return review_report_root(analysis_dir) / "calibration_commit_payload.json"


def review_commit_payloads_dir_for(analysis_dir: str | Path) -> Path:
    return review_report_root(analysis_dir) / "calibration_payloads"


def post_apply_verification_path_for(analysis_dir: str | Path) -> Path:
    return review_report_root(analysis_dir) / "post_apply_verification.json"


def normalize_matching_domain(value: str) -> str:
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


def matching_domain_label(value: str) -> str:
    normalized = normalize_matching_domain(value)
    if normalized == "scene":
        return "Scene-Referred (REDWideGamutRGB / Log3G10)"
    return "Perceptual (IPP2 / BT.709 / BT.1886)"


def _sanitize_run_label(label: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", label.strip()).strip("._-")
    return cleaned or "calibration_run"


def resolve_run_label(
    *,
    run_label: Optional[str],
    selected_clip_ids: Optional[List[str]],
    selected_clip_groups: Optional[List[str]],
) -> Optional[str]:
    if run_label and str(run_label).strip():
        return _sanitize_run_label(str(run_label))
    clip_ids = [str(item).strip() for item in (selected_clip_ids or []) if str(item).strip()]
    clip_groups = [str(item).strip() for item in (selected_clip_groups or []) if str(item).strip()]
    if clip_groups and len(clip_groups) == 1 and not clip_ids:
        return _sanitize_run_label(f"subset_{clip_groups[0]}")
    if len(clip_ids) == 1 and not clip_groups:
        return _sanitize_run_label(clip_ids[0])
    if clip_ids or clip_groups:
        return _sanitize_run_label(f"subset_{len(clip_ids) or len(clip_groups)}")
    return None


def resolve_review_output_dir(
    out_dir: str,
    *,
    run_label: Optional[str],
    selected_clip_ids: Optional[List[str]],
    selected_clip_groups: Optional[List[str]],
) -> str:
    root = Path(out_dir).expanduser().resolve()
    resolved_label = resolve_run_label(
        run_label=run_label,
        selected_clip_ids=selected_clip_ids,
        selected_clip_groups=selected_clip_groups,
    )
    return str((root / resolved_label).resolve()) if resolved_label else str(root)


def _load_json_file(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_existing_file(path: Path, *, kind: str) -> Optional[str]:
    if not path.exists():
        return f"Missing required artifact: {kind} ({path})"
    try:
        size_bytes = path.stat().st_size
    except OSError as exc:
        return f"Failed to stat required artifact {kind}: {exc}"
    if size_bytes <= 0:
        return f"Required artifact is empty: {kind} ({path})"
    return None


def _review_preview_paths_from_payload(payload: Dict[str, object]) -> List[str]:
    paths: List[str] = []
    for item in payload.get("shared_originals", []):
        original = item.get("original_frame")
        if isinstance(original, str) and original.strip():
            paths.append(original)
    for strategy in payload.get("strategies", []):
        for clip in strategy.get("clips", []):
            for key in ("original_frame", "exposure_corrected", "color_corrected", "both_corrected"):
                candidate = clip.get(key)
                if isinstance(candidate, str) and candidate.strip():
                    paths.append(candidate)
            for candidate in (clip.get("preview_variants") or {}).values():
                if isinstance(candidate, str) and candidate.strip():
                    paths.append(candidate)
    return sorted(set(paths))


def _load_optional_json(path: Path) -> Optional[Dict[str, object]]:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _confidence_bucket(value: float) -> str:
    if value >= 0.8:
        return "high"
    if value >= 0.55:
        return "medium"
    return "low"


def _log3g10_encode(linear_value: float) -> float:
    clamped = max(float(linear_value), -LOG3G10_LIN_OFFSET)
    return math.log10((clamped + LOG3G10_LIN_OFFSET) * LOG3G10_LIN_SLOPE + 1.0) * LOG3G10_LOG_SLOPE


def _luminance_from_rgb(rgb: List[float]) -> float:
    return float(rgb[0]) * 0.2126 + float(rgb[1]) * 0.7152 + float(rgb[2]) * 0.0722


def _chromaticity(rgb: List[float]) -> List[float]:
    total = max(sum(float(value) for value in rgb), 1e-6)
    return [float(value) / total for value in rgb]


def _neutrality_error_from_chromaticity(chromaticity: List[float]) -> float:
    neutral = 1.0 / 3.0
    return max(abs(float(chromaticity[0]) - neutral), abs(float(chromaticity[1]) - neutral), abs(float(chromaticity[2]) - neutral))


def _per_camera_validation_confidence(
    *,
    exposure_error: float,
    neutrality_error: float,
    sample_log2_spread: float,
    sample_chromaticity_spread: float,
    roi_variance: float,
) -> float:
    exposure_component = max(0.0, 1.0 - (float(exposure_error) / PHYSICAL_EXTREME_EXPOSURE_ERROR_THRESHOLD))
    neutrality_component = max(0.0, 1.0 - (float(neutrality_error) / PHYSICAL_MAX_NEUTRAL_ERROR_THRESHOLD))
    spread_component = 0.5 * max(0.0, 1.0 - (float(sample_log2_spread) / PHYSICAL_LOG2_SPREAD_REFERENCE))
    spread_component += 0.5 * max(0.0, 1.0 - (float(sample_chromaticity_spread) / PHYSICAL_CHROMA_SPREAD_REFERENCE))
    roi_component = max(0.0, 1.0 - (float(roi_variance) / PHYSICAL_ROI_VARIANCE_REFERENCE))
    return (
        0.40 * exposure_component
        + 0.30 * neutrality_component
        + 0.20 * spread_component
        + 0.10 * roi_component
    )


def _build_physical_validation(analysis_root: Path) -> Dict[str, object]:
    array_path = analysis_root / "array_calibration.json"
    array_payload = _load_optional_json(array_path)
    if not isinstance(array_payload, dict):
        raise ReviewValidationError(f"Missing required artifact for physical validation: {array_path}")

    analysis_dir = analysis_root / "analysis"
    analysis_by_clip: Dict[str, Dict[str, object]] = {}
    if analysis_dir.exists():
        for path in analysis_dir.glob("*.analysis.json"):
            try:
                payload = _load_optional_json(path)
            except Exception:
                payload = None
            if isinstance(payload, dict) and isinstance(payload.get("clip_id"), str):
                analysis_by_clip[str(payload["clip_id"])] = payload

    exposure_rows: List[Dict[str, object]] = []
    neutrality_rows: List[Dict[str, object]] = []
    confidence_rows: List[Dict[str, object]] = []
    outliers: List[Dict[str, object]] = []
    validation_warnings: List[str] = []
    kelvin_values: List[float] = []
    tint_values: List[float] = []

    for camera in array_payload.get("cameras", []):
        clip_id = str(camera.get("clip_id") or "")
        camera_id = str(camera.get("camera_id") or clip_id)
        measurement = camera.get("measurement") or {}
        solution = camera.get("solution") or {}
        quality = camera.get("quality") or {}
        analysis_payload = analysis_by_clip.get(clip_id, {})
        diagnostics = analysis_payload.get("diagnostics") or {}

        measured_rgb = [float(value) for value in measurement.get("measured_rgb_mean", [0.0, 0.0, 0.0])]
        gains = [float(value) for value in solution.get("rgb_gains", [1.0, 1.0, 1.0])]
        measured_chromaticity = [float(value) for value in measurement.get("measured_rgb_chromaticity", _chromaticity(measured_rgb))]

        pre_linear_luminance = _luminance_from_rgb(measured_rgb)
        post_linear_luminance = pre_linear_luminance * (2.0 ** float(solution.get("exposure_offset_stops", 0.0) or 0.0))
        pre_log3g10_luminance = _log3g10_encode(pre_linear_luminance)
        post_log3g10_luminance = _log3g10_encode(post_linear_luminance)
        exposure_error = abs(post_log3g10_luminance - LOG3G10_MID_GRAY_CODE_VALUE)
        pre_exposure_error = abs(pre_log3g10_luminance - LOG3G10_MID_GRAY_CODE_VALUE)

        post_rgb = [measured_rgb[index] * gains[index] for index in range(3)]
        post_chromaticity = _chromaticity(post_rgb)
        pre_neutral_error = _neutrality_error_from_chromaticity(measured_chromaticity)
        post_neutral_error = _neutrality_error_from_chromaticity(post_chromaticity)

        neutral_samples = diagnostics.get("neutral_samples") or []
        roi_variances = [float(sample.get("roi_variance", 0.0) or 0.0) for sample in neutral_samples if isinstance(sample, dict)]
        roi_variance = float(statistics.median(roi_variances)) if roi_variances else 0.0
        sample_log2_spread = float(quality.get("neutral_sample_log2_spread", measurement.get("neutral_sample_log2_spread", 0.0)) or 0.0)
        sample_chromaticity_spread = float(
            quality.get("neutral_sample_chromaticity_spread", measurement.get("neutral_sample_chromaticity_spread", 0.0)) or 0.0
        )
        confidence = _per_camera_validation_confidence(
            exposure_error=exposure_error,
            neutrality_error=post_neutral_error,
            sample_log2_spread=sample_log2_spread,
            sample_chromaticity_spread=sample_chromaticity_spread,
            roi_variance=roi_variance,
        )

        outlier_reasons: List[str] = []
        if exposure_error > PHYSICAL_EXTREME_EXPOSURE_ERROR_THRESHOLD:
            outlier_reasons.append("exposure_error_extreme")
        if post_neutral_error > PHYSICAL_EXTREME_NEUTRAL_ERROR_THRESHOLD:
            outlier_reasons.append("neutrality_error_extreme")
        if confidence < PHYSICAL_MIN_CONFIDENCE_THRESHOLD:
            outlier_reasons.append("validation_confidence_low")
        if outlier_reasons:
            outliers.append({"clip_id": clip_id, "camera_id": camera_id, "reasons": outlier_reasons})

        exposure_rows.append(
            {
                "clip_id": clip_id,
                "camera_id": camera_id,
                "pre_log3g10_luminance": pre_log3g10_luminance,
                "post_log3g10_luminance": post_log3g10_luminance,
                "pre_exposure_error": pre_exposure_error,
                "exposure_error": exposure_error,
                "exposure_adjust": float(solution.get("exposure_offset_stops", 0.0) or 0.0),
            }
        )
        neutrality_rows.append(
            {
                "clip_id": clip_id,
                "camera_id": camera_id,
                "pre_neutral_error": pre_neutral_error,
                "post_neutral_error": post_neutral_error,
                "pre_rgb_chromaticity": measured_chromaticity,
                "post_rgb_chromaticity": post_chromaticity,
            }
        )
        confidence_rows.append(
            {
                "clip_id": clip_id,
                "camera_id": camera_id,
                "confidence": confidence,
                "roi_variance": roi_variance,
                "sample_log2_spread": sample_log2_spread,
                "sample_chromaticity_spread": sample_chromaticity_spread,
            }
        )
        if solution.get("kelvin") is not None:
            kelvin_values.append(float(solution.get("kelvin")))
        if solution.get("tint") is not None:
            tint_values.append(float(solution.get("tint")))

    mean_exposure_error = float(statistics.mean(row["exposure_error"] for row in exposure_rows)) if exposure_rows else 0.0
    max_exposure_error = float(max((row["exposure_error"] for row in exposure_rows), default=0.0))
    mean_pre_neutral_error = float(statistics.mean(row["pre_neutral_error"] for row in neutrality_rows)) if neutrality_rows else 0.0
    mean_post_neutral_error = float(statistics.mean(row["post_neutral_error"] for row in neutrality_rows)) if neutrality_rows else 0.0
    max_post_neutral_error = float(max((row["post_neutral_error"] for row in neutrality_rows), default=0.0))
    mean_confidence = float(statistics.mean(row["confidence"] for row in confidence_rows)) if confidence_rows else 0.0
    min_confidence = float(min((row["confidence"] for row in confidence_rows), default=1.0))
    kelvin_stddev = float(statistics.pstdev(kelvin_values)) if len(kelvin_values) > 1 else 0.0
    tint_stddev = float(statistics.pstdev(tint_values)) if len(tint_values) > 1 else 0.0
    excluded_clip_ids = {str(item.get("clip_id") or "") for item in outliers}
    included_confidence_rows = [row for row in confidence_rows if str(row.get("clip_id") or "") not in excluded_clip_ids]
    included_camera_count = max(len(confidence_rows) - len(outliers), 0)
    excluded_camera_count = len(outliers)
    excluded_camera_fraction = (float(excluded_camera_count) / float(len(confidence_rows))) if confidence_rows else 0.0
    cluster_mean_confidence = (
        float(statistics.mean(row["confidence"] for row in included_confidence_rows))
        if included_confidence_rows
        else 0.0
    )
    cluster_min_confidence = (
        float(min((row["confidence"] for row in included_confidence_rows), default=0.0))
        if included_confidence_rows
        else 0.0
    )
    cluster_is_usable_after_exclusions = (
        included_camera_count >= PHYSICAL_MIN_USABLE_CLUSTER_CAMERAS
        and excluded_camera_fraction <= PHYSICAL_MAX_EXCLUDED_CAMERA_FRACTION
    )

    validation_errors: List[str] = []
    if mean_exposure_error >= PHYSICAL_MEAN_EXPOSURE_ERROR_THRESHOLD:
        validation_errors.append(
            f"Physical exposure validation failed: mean exposure error {mean_exposure_error:.4f} exceeds threshold {PHYSICAL_MEAN_EXPOSURE_ERROR_THRESHOLD:.4f}."
        )
    if max_post_neutral_error >= PHYSICAL_MAX_NEUTRAL_ERROR_THRESHOLD:
        validation_errors.append(
            f"Physical neutrality validation failed: max post neutral error {max_post_neutral_error:.4f} exceeds threshold {PHYSICAL_MAX_NEUTRAL_ERROR_THRESHOLD:.4f}."
        )
    if outliers and not cluster_is_usable_after_exclusions:
        validation_errors.append(f"Physical validation failed: {len(outliers)} extreme outlier camera(s) detected.")
    elif outliers:
        validation_warnings.append(
            f"Excluded {len(outliers)} outlier camera(s) from commit readiness; the remaining {included_camera_count}-camera cluster is still usable."
        )
    if kelvin_stddev > PHYSICAL_KELVIN_STDDEV_THRESHOLD:
        validation_errors.append(
            f"Physical Kelvin validation failed: Kelvin stddev {kelvin_stddev:.2f} exceeds threshold {PHYSICAL_KELVIN_STDDEV_THRESHOLD:.2f}."
        )

    return {
        "status": "failed" if validation_errors else ("warning" if validation_warnings else "success"),
        "errors": validation_errors,
        "warnings": validation_warnings,
        "thresholds": {
            "expected_log3g10_gray": LOG3G10_MID_GRAY_CODE_VALUE,
            "mean_exposure_error_threshold": PHYSICAL_MEAN_EXPOSURE_ERROR_THRESHOLD,
            "max_neutral_error_threshold": PHYSICAL_MAX_NEUTRAL_ERROR_THRESHOLD,
            "extreme_exposure_error_threshold": PHYSICAL_EXTREME_EXPOSURE_ERROR_THRESHOLD,
            "extreme_neutral_error_threshold": PHYSICAL_EXTREME_NEUTRAL_ERROR_THRESHOLD,
            "minimum_confidence_threshold": PHYSICAL_MIN_CONFIDENCE_THRESHOLD,
            "kelvin_stddev_threshold": PHYSICAL_KELVIN_STDDEV_THRESHOLD,
            "max_excluded_camera_fraction": PHYSICAL_MAX_EXCLUDED_CAMERA_FRACTION,
            "minimum_usable_cluster_cameras": PHYSICAL_MIN_USABLE_CLUSTER_CAMERAS,
        },
        "exposure": {
            "expected_log3g10_gray": LOG3G10_MID_GRAY_CODE_VALUE,
            "mean_exposure_error": mean_exposure_error,
            "max_exposure_error": max_exposure_error,
            "per_camera": exposure_rows,
        },
        "neutrality": {
            "target_rgb_chromaticity": [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
            "mean_pre_neutral_error": mean_pre_neutral_error,
            "mean_post_neutral_error": mean_post_neutral_error,
            "max_post_neutral_error": max_post_neutral_error,
            "per_camera": neutrality_rows,
        },
        "kelvin_tint_analysis": {
            "kelvin_stddev": kelvin_stddev,
            "kelvin_variance": float(statistics.pvariance(kelvin_values)) if len(kelvin_values) > 1 else 0.0,
            "kelvin_min": min(kelvin_values) if kelvin_values else None,
            "kelvin_max": max(kelvin_values) if kelvin_values else None,
            "kelvin_is_stable": kelvin_stddev <= PHYSICAL_KELVIN_STDDEV_THRESHOLD,
            "tint_stddev": tint_stddev,
            "tint_variance": float(statistics.pvariance(tint_values)) if len(tint_values) > 1 else 0.0,
            "tint_min": min(tint_values) if tint_values else None,
            "tint_max": max(tint_values) if tint_values else None,
            "tint_carries_variation": tint_stddev >= PHYSICAL_TINT_VARIATION_THRESHOLD,
        },
        "confidence": {
            "mean_confidence": mean_confidence,
            "min_confidence": min_confidence,
            "cluster_mean_confidence": cluster_mean_confidence,
            "cluster_min_confidence": cluster_min_confidence,
            "per_camera": confidence_rows,
        },
        "outliers": outliers,
        "excluded_cameras": [
            {
                "clip_id": str(item.get("clip_id") or ""),
                "camera_id": str(item.get("camera_id") or ""),
                "reasons": [str(reason) for reason in item.get("reasons", []) if str(reason).strip()],
                "excluded_from_anchor_selection": True,
                "excluded_from_commit": True,
            }
            for item in outliers
        ],
        "excluded_camera_count": excluded_camera_count,
        "included_camera_count": included_camera_count,
        "cluster_is_usable_after_exclusions": cluster_is_usable_after_exclusions,
    }


def _physical_validation_capability(
    analysis_root: Path,
    *,
    summary_payload: Optional[Dict[str, object]],
    review_payload: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    array_path = analysis_root / "array_calibration.json"
    array_payload = _load_optional_json(array_path)
    backend = str((array_payload or {}).get("backend") or (summary_payload or {}).get("backend") or "").strip().lower()
    measurement_domain = str(
        (review_payload or {}).get("exposure_measurement_domain")
        or (review_payload or {}).get("matching_domain")
        or
        (array_payload or {}).get("measurement_domain")
        or (summary_payload or {}).get("mode")
        or ""
    ).strip().lower()

    if isinstance(array_payload, dict) and backend == "red" and measurement_domain == "scene":
        return {"applicable": True, "reason": "physical_scene_validation_available"}
    if isinstance(array_payload, dict):
        return {
            "applicable": False,
            "status": "unsupported",
            "warnings": [
                "Physical validation skipped because this run is not a RED scene-domain array calibration."
            ],
            "reason": f"backend={backend or 'unknown'} domain={measurement_domain or 'unknown'}",
        }
    if backend == "red" and measurement_domain == "scene":
        return {
            "applicable": False,
            "status": "failed",
            "errors": [f"Missing required artifact for physical validation: {array_path}"],
            "reason": "physical_scene_validation_missing_array_calibration",
        }
    return {
        "applicable": False,
        "status": "unsupported",
        "warnings": [
            "Physical validation skipped because this run is not a RED scene-domain array calibration."
        ],
        "reason": "physical_scene_validation_not_applicable",
    }


def _strategy_label_from_key(value: str) -> str:
    mapping = {
        "median": "Median",
        "optimal_exposure": "Optimal Exposure (Best Match to Gray)",
        "optimal-exposure": "Optimal Exposure (Best Match to Gray)",
        "brightest_valid": "Optimal Exposure (Best Match to Gray)",
        "brightest-valid": "Optimal Exposure (Best Match to Gray)",
        "hero_camera": "Hero Camera",
        "hero-camera": "Hero Camera",
        "manual": "Manual Reference",
    }
    return mapping.get(str(value), str(value).replace("_", " ").title())


def _build_failure_modes(physical_validation: Dict[str, object]) -> List[Dict[str, object]]:
    thresholds = physical_validation.get("thresholds") or {}
    exposure = physical_validation.get("exposure") or {}
    neutrality = physical_validation.get("neutrality") or {}
    kelvin_tint = physical_validation.get("kelvin_tint_analysis") or {}
    confidence = physical_validation.get("confidence") or {}
    failure_modes: List[Dict[str, object]] = []
    if float(exposure.get("mean_exposure_error", 0.0) or 0.0) >= float(
        thresholds.get("mean_exposure_error_threshold", PHYSICAL_MEAN_EXPOSURE_ERROR_THRESHOLD)
    ):
        failure_modes.append(
            {
                "code": "exposure_out_of_range",
                "severity": "error",
                "message": (
                    f"Mean exposure error {float(exposure.get('mean_exposure_error', 0.0) or 0.0):.4f} "
                    f"exceeds threshold {float(thresholds.get('mean_exposure_error_threshold', PHYSICAL_MEAN_EXPOSURE_ERROR_THRESHOLD)):.4f}."
                ),
            }
        )
    if float(neutrality.get("max_post_neutral_error", 0.0) or 0.0) >= float(
        thresholds.get("max_neutral_error_threshold", PHYSICAL_MAX_NEUTRAL_ERROR_THRESHOLD)
    ):
        failure_modes.append(
            {
                "code": "neutrality_failure",
                "severity": "error",
                "message": (
                    f"Max post-neutral error {float(neutrality.get('max_post_neutral_error', 0.0) or 0.0):.4f} "
                    f"exceeds threshold {float(thresholds.get('max_neutral_error_threshold', PHYSICAL_MAX_NEUTRAL_ERROR_THRESHOLD)):.4f}."
                ),
            }
        )
    if not bool(kelvin_tint.get("kelvin_is_stable", True)):
        failure_modes.append(
            {
                "code": "unstable_kelvin",
                "severity": "error",
                "message": (
                    f"Kelvin spread is unstable across cameras "
                    f"(stddev {float(kelvin_tint.get('kelvin_stddev', 0.0) or 0.0):.2f} K)."
                ),
            }
        )
    if float(confidence.get("min_confidence", 1.0) or 1.0) < float(
        thresholds.get("minimum_confidence_threshold", PHYSICAL_MIN_CONFIDENCE_THRESHOLD)
    ):
        failure_modes.append(
            {
                "code": "low_confidence",
                "severity": "warning",
                "message": (
                    f"Lowest per-camera confidence {float(confidence.get('min_confidence', 0.0) or 0.0):.3f} "
                    f"is below threshold {float(thresholds.get('minimum_confidence_threshold', PHYSICAL_MIN_CONFIDENCE_THRESHOLD)):.3f}."
                ),
            }
        )
    excluded_cameras = list(physical_validation.get("excluded_cameras") or [])
    if excluded_cameras:
        failure_modes.append(
            {
                "code": "excluded_cameras_present",
                "severity": "warning",
                "message": (
                    f"{len(excluded_cameras)} camera(s) were excluded from anchor selection and safe commit export; "
                    f"review them individually before applying calibration."
                ),
            }
        )
    return failure_modes


def _camera_rows_for_commit_payload(
    *,
    report_payload: Dict[str, object],
    array_payload: Optional[Dict[str, object]],
    physical_validation: Optional[Dict[str, object]] = None,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    array_by_clip_id: Dict[str, Dict[str, object]] = {}
    excluded_by_clip_id: Dict[str, Dict[str, object]] = {}
    gray_target_consistency = dict(report_payload.get("gray_target_consistency") or {})
    dominant_target_class = str(gray_target_consistency.get("dominant_target_class") or "").strip().lower()
    non_dominant_clip_ids = {
        str(item).strip()
        for item in list(gray_target_consistency.get("non_dominant_clip_ids") or [])
        if str(item).strip()
    }
    mixed_target_classes = bool(gray_target_consistency.get("mixed_target_classes"))
    for excluded in list((physical_validation or {}).get("excluded_cameras") or []):
        clip_id = str(excluded.get("clip_id") or "").strip()
        if clip_id:
            excluded_by_clip_id[clip_id] = dict(excluded)
    if isinstance(array_payload, dict):
        for camera in array_payload.get("cameras", []):
            clip_id = str(camera.get("clip_id") or "").strip()
            if clip_id:
                array_by_clip_id[clip_id] = dict(camera)
    for item in report_payload.get("per_camera_analysis", []):
        commit_values = dict(item.get("commit_values") or {})
        camera_id = str(item.get("camera_label") or item.get("camera_id") or item.get("clip_id") or "").strip()
        clip_id = str(item.get("clip_id") or "").strip()
        source_path = str((array_by_clip_id.get(clip_id) or {}).get("source_path") or "")
        inventory_camera_label = (
            inventory_camera_label_from_source_path(source_path)
            or inventory_camera_label_from_clip_id(clip_id)
            or inventory_camera_label_from_clip_id(camera_id)
        )
        if not camera_id or not commit_values:
            continue
        exclusion = excluded_by_clip_id.get(clip_id, {})
        exclusion_reasons = [str(reason) for reason in exclusion.get("reasons", []) if str(reason).strip()]
        gray_target_class = str(item.get("gray_target_class") or "unresolved").strip().lower()
        if mixed_target_classes and clip_id in non_dominant_clip_ids:
            exclusion_reasons.append("Mixed gray-target classes across retained cameras")
            exclusion = {
                **exclusion,
                "reasons": exclusion_reasons,
            }
        confidence = float(item.get("confidence", 0.0) or 0.0)
        trust_class = str(item.get("trust_class") or "").strip()
        if not trust_class:
            if exclusion:
                trust_class = "EXCLUDED"
            elif confidence < PHYSICAL_MIN_CONFIDENCE_THRESHOLD:
                trust_class = "UNTRUSTED"
            else:
                trust_class = "TRUSTED"
        trust_reason = str(item.get("trust_reason") or "").strip()
        if not trust_reason:
            if exclusion:
                trust_reason = ", ".join(exclusion_reasons) or "Excluded due to inconsistent measurement"
            elif confidence < PHYSICAL_MIN_CONFIDENCE_THRESHOLD:
                trust_reason = "Low confidence"
            else:
                trust_reason = "Stable gray sample"
        reference_use = str(item.get("reference_use") or "").strip() or ("Excluded" if exclusion else "Included")
        correction_confidence = str(item.get("correction_confidence") or "").strip() or (
            "LOW" if trust_class in {"UNTRUSTED", "EXCLUDED"} else "HIGH"
        )
        rows.append(
            {
                "camera_id": camera_id,
                "clip_id": clip_id,
                "inventory_camera_label": inventory_camera_label,
                "inventory_camera_ip": DEFAULT_CAMERA_IP_MAP.get(str(inventory_camera_label or "").upper(), ""),
                "source_path": source_path,
                "commit_values": commit_values,
                "confidence": confidence,
                "note": str(item.get("note") or ""),
                "trust_class": trust_class,
                "trust_reason": trust_reason,
                "reference_use": reference_use,
                "correction_confidence": correction_confidence,
                "is_hero_camera": bool(item.get("is_hero_camera")),
                "excluded_from_commit": bool(exclusion),
                "exclusion_reasons": exclusion_reasons,
                "gray_target_class": gray_target_class,
                "gray_target_is_dominant": bool(gray_target_class and gray_target_class == dominant_target_class),
            }
        )
    if rows or not isinstance(array_payload, dict):
        return rows
    for camera in array_payload.get("cameras", []):
        commit_values = build_commit_values(
            exposure_adjust=float((camera.get("solution") or {}).get("exposure_offset_stops", 0.0) or 0.0),
            rgb_gains=(camera.get("solution") or {}).get("rgb_gains"),
            confidence=float((camera.get("quality") or {}).get("confidence", 0.0) or 0.0),
            sample_log2_spread=float((camera.get("quality") or {}).get("neutral_sample_log2_spread", 0.0) or 0.0),
            sample_chromaticity_spread=float((camera.get("quality") or {}).get("neutral_sample_chromaticity_spread", 0.0) or 0.0),
            measured_rgb_chromaticity=(camera.get("measurement") or {}).get("measured_rgb_chromaticity"),
            target_rgb_chromaticity=((array_payload.get("target") or {}).get("color") or {}).get("target_rgb_chromaticity"),
            saturation=1.0,
            saturation_supported=False,
            wb_solution={
                "kelvin": int((camera.get("solution") or {}).get("kelvin") or 5600),
                "tint": float((camera.get("solution") or {}).get("tint", 0.0) or 0.0),
                "method": str((camera.get("solution") or {}).get("derivation_method") or "unknown"),
                "model_key": (((array_payload.get("global_scene_intent") or {}).get("white_balance_model") or {}).get("model_key")),
                "model_label": (((array_payload.get("global_scene_intent") or {}).get("white_balance_model") or {}).get("model_label")),
                "as_shot_kelvin": int((camera.get("measurement") or {}).get("as_shot_kelvin") or 5600),
                "as_shot_tint": float((camera.get("measurement") or {}).get("as_shot_tint", 0.0) or 0.0),
                "white_balance_axes": {"amber_blue": 0.0, "green_magenta": 0.0},
                "predicted_white_balance_axes": {"amber_blue": 0.0, "green_magenta": 0.0},
                "implied_rgb_gains": list((camera.get("solution") or {}).get("rgb_gains") or [1.0, 1.0, 1.0]),
                "pre_neutral_residual": float((camera.get("quality") or {}).get("color_residual", 0.0) or 0.0),
                "post_neutral_residual": float((camera.get("quality") or {}).get("post_color_residual", 0.0) or 0.0),
                "confidence_weight": float((camera.get("quality") or {}).get("confidence", 1.0) or 1.0),
            },
        )
        rows.append(
            {
                "camera_id": str(camera.get("camera_id") or camera.get("clip_id") or "").strip(),
                "clip_id": str(camera.get("clip_id") or ""),
                "inventory_camera_label": (
                    inventory_camera_label_from_source_path(str(camera.get("source_path") or ""))
                    or inventory_camera_label_from_clip_id(str(camera.get("clip_id") or ""))
                ),
                "inventory_camera_ip": DEFAULT_CAMERA_IP_MAP.get(
                    str(
                        inventory_camera_label_from_source_path(str(camera.get("source_path") or ""))
                        or inventory_camera_label_from_clip_id(str(camera.get("clip_id") or ""))
                        or ""
                    ).upper(),
                    "",
                ),
                "source_path": str(camera.get("source_path") or ""),
                "commit_values": commit_values,
                "confidence": float((camera.get("quality") or {}).get("confidence", 0.0) or 0.0),
                "note": "",
                "is_hero_camera": False,
                "excluded_from_commit": False,
                "exclusion_reasons": [],
            }
        )
    return rows


def _write_commit_payload_artifacts(
    *,
    analysis_root: Path,
    report_payload: Dict[str, object],
    array_payload: Optional[Dict[str, object]],
    physical_validation: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    gray_target_consistency = dict(report_payload.get("gray_target_consistency") or (report_payload.get("run_assessment") or {}).get("gray_target_consistency") or {})
    rows = _camera_rows_for_commit_payload(
        report_payload=report_payload,
        array_payload=array_payload,
        physical_validation=physical_validation,
    )
    payload_dir = review_commit_payloads_dir_for(analysis_root)
    payload_dir.mkdir(parents=True, exist_ok=True)
    mapping: Dict[str, Dict[str, object]] = {}
    per_camera_paths: List[Dict[str, object]] = []
    for row in rows:
        camera_id = str(row["camera_id"])
        inventory_camera_label = str(row.get("inventory_camera_label") or "").upper()
        commit_values = dict(row["commit_values"])
        calibration_values = {
            "exposureAdjust": float(commit_values.get("exposureAdjust", 0.0) or 0.0),
            "kelvin": int(commit_values.get("kelvin", 5600) or 5600),
            "tint": float(commit_values.get("tint", 0.0) or 0.0),
        }
        payload = {
            "schema_version": "r3dmatch_rcp2_ready_v1",
            "camera_id": camera_id,
            "clip_id": row["clip_id"],
            "inventory_camera_label": inventory_camera_label,
            "inventory_camera_ip": str(row.get("inventory_camera_ip") or ""),
            "source_path": str(row.get("source_path") or ""),
            "format": "rcp2_ready",
            "calibration": calibration_values,
            "confidence": float(row["confidence"]),
            "trust_class": str(row.get("trust_class") or ""),
            "trust_reason": str(row.get("trust_reason") or ""),
            "reference_use": str(row.get("reference_use") or ""),
            "correction_confidence": str(row.get("correction_confidence") or ""),
            "is_hero_camera": bool(row["is_hero_camera"]),
            "excluded_from_commit": bool(row.get("excluded_from_commit")),
            "exclusion_reasons": [str(reason) for reason in row.get("exclusion_reasons", []) if str(reason).strip()],
            "safe_to_commit": not bool(row.get("excluded_from_commit")),
            "gray_target_class": str(row.get("gray_target_class") or ""),
            "gray_target_is_dominant": bool(row.get("gray_target_is_dominant")),
            "notes": [
                *([str(row["note"])] if str(row["note"]).strip() else []),
                *(
                    [f"Excluded from safe commit set: {', '.join(str(reason) for reason in row.get('exclusion_reasons', []) if str(reason).strip())}"]
                    if row.get("excluded_from_commit")
                    else []
                ),
            ],
        }
        payload_path = payload_dir / f"{camera_id}.json"
        payload_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        mapping[camera_id] = calibration_values
        per_camera_paths.append(
            {
                "camera_id": camera_id,
                "inventory_camera_label": inventory_camera_label,
                "inventory_camera_ip": str(row.get("inventory_camera_ip") or ""),
                "path": str(payload_path),
            }
        )

    aggregate = {
        "schema_version": "r3dmatch_calibration_commit_payload_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_label": report_payload.get("run_label"),
        "recommended_strategy": ((report_payload.get("recommended_strategy") or {}).get("strategy_key")),
        "hero_camera": ((report_payload.get("hero_recommendation") or {}).get("candidate_clip_id")),
        "cameras": mapping,
        "camera_targets": [
            {
                "camera_id": str(row["camera_id"]),
                "clip_id": str(row["clip_id"]),
                "inventory_camera_label": str(row.get("inventory_camera_label") or ""),
                "inventory_camera_ip": str(row.get("inventory_camera_ip") or ""),
                "calibration": {
                    "exposureAdjust": float(dict(row["commit_values"]).get("exposureAdjust", 0.0) or 0.0),
                    "kelvin": int(dict(row["commit_values"]).get("kelvin", 5600) or 5600),
                    "tint": float(dict(row["commit_values"]).get("tint", 0.0) or 0.0),
                },
                "confidence": float(row["confidence"]),
                "trust_class": str(row.get("trust_class") or ""),
                "trust_reason": str(row.get("trust_reason") or ""),
                "reference_use": str(row.get("reference_use") or ""),
                "correction_confidence": str(row.get("correction_confidence") or ""),
                "is_hero_camera": bool(row["is_hero_camera"]),
                "excluded_from_commit": bool(row.get("excluded_from_commit")),
                "exclusion_reasons": [str(reason) for reason in row.get("exclusion_reasons", []) if str(reason).strip()],
                "gray_target_class": str(row.get("gray_target_class") or ""),
                "gray_target_is_dominant": bool(row.get("gray_target_is_dominant")),
            }
            for row in rows
        ],
        "safe_camera_targets": [
            {
                "camera_id": str(row["camera_id"]),
                "clip_id": str(row["clip_id"]),
                "inventory_camera_label": str(row.get("inventory_camera_label") or ""),
                "inventory_camera_ip": str(row.get("inventory_camera_ip") or ""),
                "calibration": {
                    "exposureAdjust": float(dict(row["commit_values"]).get("exposureAdjust", 0.0) or 0.0),
                    "kelvin": int(dict(row["commit_values"]).get("kelvin", 5600) or 5600),
                    "tint": float(dict(row["commit_values"]).get("tint", 0.0) or 0.0),
                },
                "confidence": float(row["confidence"]),
                "trust_class": str(row.get("trust_class") or ""),
                "trust_reason": str(row.get("trust_reason") or ""),
                "reference_use": str(row.get("reference_use") or ""),
                "correction_confidence": str(row.get("correction_confidence") or ""),
                "is_hero_camera": bool(row["is_hero_camera"]),
                "gray_target_class": str(row.get("gray_target_class") or ""),
            }
            for row in rows
            if not bool(row.get("excluded_from_commit"))
        ],
        "excluded_cameras": [
            {
                "camera_id": str(row["camera_id"]),
                "clip_id": str(row["clip_id"]),
                "inventory_camera_label": str(row.get("inventory_camera_label") or ""),
                "inventory_camera_ip": str(row.get("inventory_camera_ip") or ""),
                "trust_class": str(row.get("trust_class") or ""),
                "reasons": [str(reason) for reason in row.get("exclusion_reasons", []) if str(reason).strip()],
                "gray_target_class": str(row.get("gray_target_class") or ""),
            }
            for row in rows
            if bool(row.get("excluded_from_commit"))
        ],
        "gray_target_consistency": gray_target_consistency,
        "per_camera_payloads": per_camera_paths,
    }
    aggregate_path = review_commit_payload_path_for(analysis_root)
    aggregate_path.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    return {
        "aggregate_path": str(aggregate_path),
        "per_camera_dir": str(payload_dir),
        "camera_count": len(mapping),
        "mapping": mapping,
        "safe_camera_targets": aggregate["safe_camera_targets"],
        "excluded_cameras": aggregate["excluded_cameras"],
        "per_camera_payloads": per_camera_paths,
    }


def _build_recommendation_layer(
    *,
    report_payload: Dict[str, object],
    physical_validation: Dict[str, object],
) -> Dict[str, object]:
    recommended = dict(report_payload.get("recommended_strategy") or {})
    hero = dict(report_payload.get("hero_recommendation") or {})
    physical_confidence = float(((physical_validation.get("confidence") or {}).get("mean_confidence", 0.0)) or 0.0)
    strategy_confidence = max(
        0.0,
        1.0 - float(((recommended.get("metrics") or {}).get("mean_confidence_penalty", 1.0)) or 1.0),
    )
    confidence_score = round((physical_confidence + strategy_confidence) / 2.0, 4) if recommended else round(physical_confidence, 4)
    notes: List[str] = []
    if recommended.get("reason"):
        notes.append(str(recommended["reason"]))
    if hero.get("reason"):
        notes.append(str(hero["reason"]))
    operator_note = str(report_payload.get("operator_recommendation") or "").strip()
    if operator_note:
        notes.append(operator_note)
    run_assessment = dict(report_payload.get("run_assessment") or {})
    return {
        "recommended_strategy": {
            "strategy_key": str(recommended.get("strategy_key") or ""),
            "strategy_label": str(recommended.get("strategy_label") or _strategy_label_from_key(str(recommended.get("strategy_key") or ""))),
            "reason": str(recommended.get("reason") or ""),
            "metrics": dict(recommended.get("metrics") or {}),
        },
        "hero_camera": str(hero.get("candidate_clip_id") or "") or None,
        "hero_camera_confidence": str(hero.get("confidence") or "") or None,
        "confidence_score": confidence_score,
        "recommendation_strength": str(run_assessment.get("recommendation_strength") or "MEDIUM_CONFIDENCE"),
        "run_status": str(run_assessment.get("status") or ""),
        "summary_notes": notes,
    }


def _resolve_run_assessment(
    *,
    report_payload: Dict[str, object],
    recommendation: Dict[str, object],
    physical_validation: Dict[str, object],
) -> Dict[str, object]:
    payload_assessment = dict(report_payload.get("run_assessment") or {})
    if str(payload_assessment.get("status") or "").strip():
        return payload_assessment
    gray_target_consistency = dict(report_payload.get("gray_target_consistency") or {})
    excluded_camera_count = int(physical_validation.get("excluded_camera_count", 0) or 0)
    camera_count = max(int(physical_validation.get("included_camera_count", 0) or 0) + excluded_camera_count, 0)
    physical_status = str(physical_validation.get("status") or "")
    confidence_score = float(recommendation.get("confidence_score", 0.0) or 0.0)
    cluster_usable = bool(physical_validation.get("cluster_is_usable_after_exclusions"))
    if bool(gray_target_consistency.get("mixed_target_classes")):
        status = "DO_NOT_PUSH"
        recommendation_strength = "LOW_CONFIDENCE"
        operator_note = "Review is required before commit because retained cameras were measured from mixed gray target classes."
    elif physical_status == "failed" or not cluster_usable and excluded_camera_count:
        status = "DO_NOT_PUSH"
        recommendation_strength = "LOW_CONFIDENCE"
        operator_note = "Do not push these corrections later without remeasuring the set."
    elif excluded_camera_count or physical_status == "warning" or confidence_score < 0.7:
        status = "READY_WITH_WARNINGS"
        recommendation_strength = "MEDIUM_CONFIDENCE"
        operator_note = "This run is usable, but at least one camera should be reviewed before any later push."
    else:
        status = "READY"
        recommendation_strength = "HIGH_CONFIDENCE"
        operator_note = "This run is strong enough to trust later for push if the camera readback still matches."
    return {
        "status": status,
        "recommendation_strength": recommendation_strength,
        "safe_to_push_later": status in {"READY", "READY_WITH_WARNINGS"},
        "trusted_camera_count": max(int(physical_validation.get("included_camera_count", 0) or 0), 0),
        "caution_camera_count": 0,
        "untrusted_camera_count": 0,
        "excluded_camera_count": excluded_camera_count,
        "reference_eligible_count": max(int(physical_validation.get("included_camera_count", 0) or 0), 0),
        "camera_count": camera_count,
        "average_trust_score": confidence_score,
        "anchor_camera": None,
        "anchor_trust_class": "",
        "anchor_summary": "Legacy report payload did not include anchor trust details.",
        "gating_reasons": [
            *list(physical_validation.get("warnings") or []),
            *(
                ["Retained cameras mix gray-sphere and gray-card solves, so safe commit export is blocked."]
                if bool(gray_target_consistency.get("mixed_target_classes"))
                else []
            ),
        ],
        "operator_note": operator_note,
        "gray_target_consistency": gray_target_consistency,
    }


def _build_human_summary(
    *,
    report_payload: Dict[str, object],
    recommendation: Dict[str, object],
    physical_validation: Dict[str, object],
    failure_modes: List[Dict[str, object]],
    run_assessment: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    recommended = recommendation.get("recommended_strategy") or {}
    exposure = physical_validation.get("exposure") or {}
    neutrality = physical_validation.get("neutrality") or {}
    resolved_run_assessment = dict(run_assessment or report_payload.get("run_assessment") or {})
    exec_summary = str(report_payload.get("executive_synopsis") or "").strip()
    if not exec_summary:
        exec_summary = (
            f"{recommended.get('strategy_label') or 'No strategy'} is the current recommendation. "
            f"Mean exposure error is {float(exposure.get('mean_exposure_error', 0.0) or 0.0):.4f} and "
            f"max post-neutral error is {float(neutrality.get('max_post_neutral_error', 0.0) or 0.0):.4f}."
        )
    if physical_validation.get("status") == "success":
        pass_fail = "Physical validation passed and the calibration recommendation is ready for commit export."
    elif physical_validation.get("status") == "warning" and physical_validation.get("cluster_is_usable_after_exclusions"):
        pass_fail = (
            "The main camera cluster remains usable, but one or more cameras were excluded from the safe commit set. "
            "Proceed only with the non-excluded cameras and review the flagged outliers separately."
        )
    elif physical_validation.get("status") == "unsupported" and str(report_payload.get("matching_domain") or "") == "perceptual":
        pass_fail = (
            "Perceptual lightweight review completed in the authoritative IPP2 monitoring domain. "
            "Use the rendered monitoring measurements and operator guidance for commit decisions."
        )
    else:
        pass_fail = "Physical validation did not fully pass; review the flagged failure modes before committing calibration values."
    per_camera_summary = [
        {
            "camera_id": str(item.get("camera_label") or item.get("camera_id") or item.get("clip_id") or ""),
            "clip_id": str(item.get("clip_id") or ""),
            "summary": str(
                item.get("operator_summary")
                or item.get("trust_reason")
                or item.get("note")
                or "No additional note."
            ),
            "confidence": float(item.get("confidence", 0.0) or 0.0),
            "trust_class": str(item.get("trust_class") or ""),
            "reference_use": str(item.get("reference_use") or ""),
        }
        for item in report_payload.get("per_camera_analysis", [])
    ]
    return {
        "executive_summary": exec_summary,
        "pass_fail_explanation": pass_fail,
        "run_status": str(resolved_run_assessment.get("status") or ""),
        "recommendation_strength": str(resolved_run_assessment.get("recommendation_strength") or ""),
        "safe_to_push_later": bool(resolved_run_assessment.get("safe_to_push_later")),
        "failure_modes_explanation": [str(item.get("message") or "") for item in failure_modes],
        "per_camera_summary": per_camera_summary,
    }


def _build_post_apply_validation(
    *,
    report_payload: Dict[str, object],
    physical_validation: Dict[str, object],
    recommendation: Dict[str, object],
) -> Dict[str, object]:
    exposure = physical_validation.get("exposure") or {}
    neutrality = physical_validation.get("neutrality") or {}
    if not exposure and not neutrality:
        return {
            "status": "unsupported",
            "verification_mode": "modeled_from_recommended_commit_values",
            "recommended_strategy": (recommendation.get("recommended_strategy") or {}).get("strategy_key"),
            "summary": {
                "pre_mean_exposure_error": 0.0,
                "post_mean_exposure_error": 0.0,
                "pre_mean_neutral_error": 0.0,
                "post_mean_neutral_error": 0.0,
                "pre_combined_variance": 0.0,
                "post_combined_variance": 0.0,
                "exposure_error_reduced": False,
                "neutrality_improved": False,
                "variance_reduced": False,
            },
            "notes": [
                "Scene-domain physical validation is not applicable to this perceptual lightweight review.",
                "No modeled post-apply comparison was produced because no scene-domain physical rows were available.",
            ],
            "per_camera": [],
        }
    exp_rows = {str(item.get("clip_id") or ""): item for item in exposure.get("per_camera", [])}
    neu_rows = {str(item.get("clip_id") or ""): item for item in neutrality.get("per_camera", [])}
    per_camera: List[Dict[str, object]] = []
    for item in report_payload.get("per_camera_analysis", []):
        clip_id = str(item.get("clip_id") or "")
        exp_row = exp_rows.get(clip_id, {})
        neu_row = neu_rows.get(clip_id, {})
        pre_exp = float(exp_row.get("pre_exposure_error", 0.0) or 0.0)
        post_exp = float(exp_row.get("exposure_error", 0.0) or 0.0)
        pre_neu = float(neu_row.get("pre_neutral_error", 0.0) or 0.0)
        post_neu = float(neu_row.get("post_neutral_error", 0.0) or 0.0)
        per_camera.append(
            {
                "camera_id": str(item.get("camera_label") or clip_id),
                "clip_id": clip_id,
                "pre_exposure_error": pre_exp,
                "post_exposure_error": post_exp,
                "pre_neutral_error": pre_neu,
                "post_neutral_error": post_neu,
                "exposure_error_reduced": post_exp <= pre_exp,
                "neutrality_improved": post_neu <= pre_neu,
                "variance_reduced": (post_exp + post_neu) <= (pre_exp + pre_neu),
            }
        )
    pre_mean_exp = float(statistics.mean(item["pre_exposure_error"] for item in per_camera)) if per_camera else 0.0
    post_mean_exp = float(statistics.mean(item["post_exposure_error"] for item in per_camera)) if per_camera else 0.0
    pre_mean_neu = float(statistics.mean(item["pre_neutral_error"] for item in per_camera)) if per_camera else 0.0
    post_mean_neu = float(statistics.mean(item["post_neutral_error"] for item in per_camera)) if per_camera else 0.0
    pre_combined = [item["pre_exposure_error"] + item["pre_neutral_error"] for item in per_camera]
    post_combined = [item["post_exposure_error"] + item["post_neutral_error"] for item in per_camera]
    pre_var = float(statistics.pvariance(pre_combined)) if len(pre_combined) > 1 else 0.0
    post_var = float(statistics.pvariance(post_combined)) if len(post_combined) > 1 else 0.0
    improved = post_mean_exp <= pre_mean_exp and post_mean_neu <= pre_mean_neu and post_var <= pre_var
    return {
        "status": "success" if improved else "warning",
        "verification_mode": "modeled_from_recommended_commit_values",
        "recommended_strategy": (recommendation.get("recommended_strategy") or {}).get("strategy_key"),
        "summary": {
            "pre_mean_exposure_error": pre_mean_exp,
            "post_mean_exposure_error": post_mean_exp,
            "pre_mean_neutral_error": pre_mean_neu,
            "post_mean_neutral_error": post_mean_neu,
            "pre_combined_variance": pre_var,
            "post_combined_variance": post_var,
            "exposure_error_reduced": post_mean_exp <= pre_mean_exp,
            "neutrality_improved": post_mean_neu <= pre_mean_neu,
            "variance_reduced": post_var <= pre_var,
        },
        "notes": [
            "This verification pass is modeled from the solved correction residuals and physical validation outputs.",
            "Automated camera re-measurement after apply is not yet implemented in this loop.",
        ],
        "per_camera": per_camera,
    }


def _comparison_variance_from_validation(validation: Dict[str, object]) -> float:
    physical = validation.get("physical_validation") or {}
    exposure_rows = list(((physical.get("exposure") or {}).get("per_camera") or []))
    neutrality_rows = list(((physical.get("neutrality") or {}).get("per_camera") or []))
    neutrality_by_clip = {str(row.get("clip_id") or ""): row for row in neutrality_rows}
    combined: List[float] = []
    for exp_row in exposure_rows:
        clip_id = str(exp_row.get("clip_id") or "")
        neu_row = neutrality_by_clip.get(clip_id, {})
        combined.append(
            float(exp_row.get("exposure_error", 0.0) or 0.0)
            + float(neu_row.get("post_neutral_error", 0.0) or 0.0)
        )
    if len(combined) <= 1:
        return 0.0
    return float(statistics.pvariance(combined))


def build_post_apply_verification_from_reviews(
    before_review_dir: str,
    after_review_dir: str,
    *,
    out_path: Optional[str] = None,
) -> Dict[str, object]:
    before_root = Path(before_review_dir).expanduser().resolve()
    after_root = Path(after_review_dir).expanduser().resolve()
    before_validation_path = review_validation_path_for(before_root)
    after_validation_path = review_validation_path_for(after_root)
    if not before_validation_path.exists():
        raise ReviewValidationError(f"Before-review validation is missing: {before_validation_path}")
    if not after_validation_path.exists():
        raise ReviewValidationError(f"After-review validation is missing: {after_validation_path}")

    before_validation = json.loads(before_validation_path.read_text(encoding="utf-8"))
    after_validation = json.loads(after_validation_path.read_text(encoding="utf-8"))

    before_physical = dict(before_validation.get("physical_validation") or {})
    after_physical = dict(after_validation.get("physical_validation") or {})
    before_exposure = dict(before_physical.get("exposure") or {})
    after_exposure = dict(after_physical.get("exposure") or {})
    before_neutrality = dict(before_physical.get("neutrality") or {})
    after_neutrality = dict(after_physical.get("neutrality") or {})

    before_mean_exposure = float(before_exposure.get("mean_exposure_error", 0.0) or 0.0)
    after_mean_exposure = float(after_exposure.get("mean_exposure_error", 0.0) or 0.0)
    before_mean_neutral = float(before_neutrality.get("mean_post_neutral_error", 0.0) or 0.0)
    after_mean_neutral = float(after_neutrality.get("mean_post_neutral_error", 0.0) or 0.0)
    before_variance = _comparison_variance_from_validation(before_validation)
    after_variance = _comparison_variance_from_validation(after_validation)

    exposure_improved = after_mean_exposure <= before_mean_exposure
    neutrality_improved = after_mean_neutral <= before_mean_neutral
    variance_reduced = after_variance <= before_variance
    improvement_count = sum(1 for flag in (exposure_improved, neutrality_improved, variance_reduced) if flag)

    if after_validation.get("status") != "success" or after_physical.get("status") not in {"success", "warning"}:
        status = "fail"
    elif improvement_count == 3:
        status = "success"
    elif improvement_count >= 1:
        status = "warning"
    else:
        status = "fail"

    result = {
        "schema_version": "r3dmatch_post_apply_verification_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "before_review_dir": str(before_root),
        "after_review_dir": str(after_root),
        "before_validation_path": str(before_validation_path),
        "after_validation_path": str(after_validation_path),
        "before_strategy": ((before_validation.get("recommendation") or {}).get("recommended_strategy") or {}).get("strategy_key"),
        "after_strategy": ((after_validation.get("recommendation") or {}).get("recommended_strategy") or {}).get("strategy_key"),
        "status": status,
        "summary": {
            "before_mean_exposure_error": before_mean_exposure,
            "after_mean_exposure_error": after_mean_exposure,
            "before_mean_neutral_error": before_mean_neutral,
            "after_mean_neutral_error": after_mean_neutral,
            "before_combined_variance": before_variance,
            "after_combined_variance": after_variance,
            "exposure_improved": exposure_improved,
            "neutrality_improved": neutrality_improved,
            "variance_reduced": variance_reduced,
        },
        "notes": [
            "This comparison uses the physical validation summaries from two completed review runs.",
            "It does not automate clip reacquisition; it compares the before and after review artifacts you provide.",
        ],
    }
    if out_path:
        output_path = Path(out_path).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        result["report_path"] = str(output_path)
    return result


def validate_review_run_contract(analysis_dir: str) -> Dict[str, object]:
    analysis_root = Path(analysis_dir).expanduser().resolve()
    report_root = review_report_root(analysis_root)
    result: Dict[str, object] = {
        "analysis_dir": str(analysis_root),
        "report_dir": str(report_root),
        "status": "success",
        "errors": [],
        "warnings": [],
        "required_artifacts": {},
        "optional_artifacts": {},
        "preview_reference_count": 0,
        "preview_existing_count": 0,
        "missing_preview_paths": [],
    }

    required_paths = {
        "summary_json": analysis_root / "summary.json",
        "contact_sheet_json": review_payload_path_for(analysis_root),
        "review_manifest_json": review_manifest_path_for(analysis_root),
        "review_package_json": review_package_path_for(analysis_root),
        "preview_commands_json": review_preview_commands_path_for(analysis_root),
    }
    optional_paths = {
        "contact_sheet_html": review_html_path_for(analysis_root),
        "preview_contact_sheet_pdf": review_pdf_path_for(analysis_root),
    }

    parsed_payloads: Dict[str, Dict[str, object]] = {}
    for key, path in required_paths.items():
        error = _validate_existing_file(path, kind=key)
        exists = path.exists()
        entry: Dict[str, object] = {"path": str(path), "exists": exists, "parsed": False, "size_bytes": path.stat().st_size if exists else 0}
        if error:
            result["errors"].append(error)
        else:
            try:
                if path.suffix == ".json":
                    parsed_payloads[key] = _load_json_file(path)
                    entry["parsed"] = True
                else:
                    entry["parsed"] = True
            except Exception as exc:
                result["errors"].append(f"Failed to parse required artifact {key}: {exc}")
        result["required_artifacts"][key] = entry

    for key, path in optional_paths.items():
        exists = path.exists()
        entry = {"path": str(path), "exists": exists, "size_bytes": path.stat().st_size if exists else 0}
        result["optional_artifacts"][key] = entry
        if not exists:
            result["warnings"].append(f"Optional artifact missing: {key} ({path})")
        elif entry["size_bytes"] <= 0:
            result["errors"].append(f"Optional report artifact is empty: {key} ({path})")

    review_manifest_payload = parsed_payloads.get("review_manifest_json") or {}
    contact_sheet_payload = parsed_payloads.get("contact_sheet_json") or {}
    resolved_review_mode = normalize_review_mode(
        str(
            review_manifest_payload.get("review_mode")
            or contact_sheet_payload.get("review_mode")
            or "full_contact_sheet"
        )
    )
    result["review_mode"] = resolved_review_mode
    result["review_mode_label"] = review_mode_label(resolved_review_mode)
    if contact_sheet_payload:
        preview_paths = _review_preview_paths_from_payload(contact_sheet_payload)
        missing_preview_paths = [path for path in preview_paths if not Path(path).exists()]
        unreadable_preview_paths: List[str] = []
        empty_preview_paths: List[str] = []
        for path_str in preview_paths:
            preview_path = Path(path_str)
            if not preview_path.exists():
                continue
            try:
                if preview_path.stat().st_size <= 0:
                    empty_preview_paths.append(path_str)
                    continue
            except OSError:
                empty_preview_paths.append(path_str)
                continue
            try:
                with Image.open(preview_path) as image:
                    image.verify()
            except (OSError, UnidentifiedImageError):
                unreadable_preview_paths.append(path_str)
        result["preview_reference_count"] = len(preview_paths)
        result["preview_existing_count"] = len(preview_paths) - len(missing_preview_paths) - len(empty_preview_paths) - len(unreadable_preview_paths)
        result["missing_preview_paths"] = missing_preview_paths
        result["empty_preview_paths"] = empty_preview_paths
        result["unreadable_preview_paths"] = unreadable_preview_paths
        if resolved_review_mode == "full_contact_sheet" and not preview_paths:
            result["errors"].append("Review report did not reference any preview images.")
        if resolved_review_mode == "full_contact_sheet" and missing_preview_paths:
            result["errors"].append(f"Missing {len(missing_preview_paths)} preview image(s) referenced by contact_sheet.json.")
        if resolved_review_mode == "full_contact_sheet" and empty_preview_paths:
            result["errors"].append(f"Found {len(empty_preview_paths)} empty preview image(s) referenced by contact_sheet.json.")
        if resolved_review_mode == "full_contact_sheet" and unreadable_preview_paths:
            result["errors"].append(f"Found {len(unreadable_preview_paths)} unreadable preview image(s) referenced by contact_sheet.json.")
        result["clip_count"] = int(contact_sheet_payload.get("clip_count", 0) or 0)
        result["color_preview_status"] = contact_sheet_payload.get("color_preview_status")
        result["color_preview_note"] = contact_sheet_payload.get("color_preview_note")

    if not result["optional_artifacts"]["contact_sheet_html"]["exists"] and not result["optional_artifacts"]["preview_contact_sheet_pdf"]["exists"]:
        result["errors"].append("No human-readable report artifact was produced: both HTML and PDF are missing.")

    capability = _physical_validation_capability(
        analysis_root,
        summary_payload=parsed_payloads.get("summary_json"),
        review_payload=parsed_payloads.get("review_package_json") or parsed_payloads.get("contact_sheet_json"),
    )
    array_payload = _load_optional_json(analysis_root / "array_calibration.json")
    if capability.get("applicable"):
        try:
            physical_validation = _build_physical_validation(analysis_root)
        except Exception as exc:
            physical_validation = {
                "status": "failed",
                "errors": [f"Failed to compute physical validation: {exc}"],
                "reason": "physical_validation_exception",
            }
    else:
        physical_validation = {
            "status": capability.get("status", "unsupported"),
            "errors": list(capability.get("errors", [])),
            "warnings": list(capability.get("warnings", [])),
            "reason": capability.get("reason"),
        }
    result["physical_validation"] = physical_validation
    if physical_validation.get("status") == "failed":
        result["errors"].extend(str(item) for item in physical_validation.get("errors", []) if str(item).strip())
    result["warnings"].extend(str(item) for item in physical_validation.get("warnings", []) if str(item).strip())

    report_payload = parsed_payloads.get("contact_sheet_json") or {}
    failure_modes = _build_failure_modes(physical_validation)
    recommendation = _build_recommendation_layer(
        report_payload=report_payload,
        physical_validation=physical_validation,
    )
    run_assessment = _resolve_run_assessment(
        report_payload=report_payload,
        recommendation=recommendation,
        physical_validation=physical_validation,
    )
    human_summary = _build_human_summary(
        report_payload=report_payload,
        recommendation=recommendation,
        physical_validation=physical_validation,
        failure_modes=failure_modes,
        run_assessment=run_assessment,
    )
    commit_payload = _write_commit_payload_artifacts(
        analysis_root=analysis_root,
        report_payload=report_payload,
        array_payload=array_payload,
        physical_validation=physical_validation,
    )
    post_apply_validation = _build_post_apply_validation(
        report_payload=report_payload,
        physical_validation=physical_validation,
        recommendation=recommendation,
    )
    result["recommendation"] = recommendation
    result["run_assessment"] = run_assessment
    result["failure_modes"] = failure_modes
    result["human_summary"] = human_summary
    result["commit_payload"] = commit_payload
    result["post_apply_validation"] = post_apply_validation
    gray_target_consistency = dict(report_payload.get("gray_target_consistency") or {})
    dominant_target_class = str(gray_target_consistency.get("dominant_target_class") or "").strip().lower()
    if bool(gray_target_consistency.get("mixed_target_classes")):
        result["errors"].append("Target class mismatch — run invalid")
    elif str(report_payload.get("target_type") or "").strip().lower() == "gray_sphere" and dominant_target_class and dominant_target_class != "sphere":
        result["errors"].append("Gray-sphere review resolved to a non-sphere retained target class — run invalid")

    if result["errors"]:
        result["status"] = "failed"

    validation_path = review_validation_path_for(analysis_root)
    validation_path.parent.mkdir(parents=True, exist_ok=True)
    result["validation_path"] = str(validation_path)
    result["validated_at"] = time.time()
    validation_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _format_review_validation_failure(validation: Dict[str, object]) -> str:
    errors = [str(item) for item in validation.get("errors", []) if str(item).strip()]
    if not errors:
        return "Review run failed validation for an unspecified reason."
    return "Review run failed validation: " + "; ".join(errors)


def _load_clip_subset_definition(path: str) -> Dict[str, object]:
    subset_path = Path(path).expanduser().resolve()
    payload = json.loads(subset_path.read_text(encoding="utf-8"))
    return {
        "path": str(subset_path),
        "run_label": payload.get("run_label"),
        "clip_ids": [str(item) for item in payload.get("clip_ids", []) if str(item).strip()],
        "clip_groups": [str(item) for item in payload.get("clip_groups", []) if str(item).strip()],
    }


def review_calibration(
    input_path: str,
    *,
    out_dir: str,
    source_mode: str = "local_folder",
    ftps_reel: Optional[str] = None,
    ftps_clip_spec: Optional[str] = None,
    ftps_cameras: Optional[List[str]] = None,
    ftps_local_root: Optional[str] = None,
    ftps_username: str = "ftp1",
    ftps_password: str = "12345678",
    target_type: str,
    processing_mode: str,
    mode: str,
    backend: str,
    lut_override: Optional[str],
    calibration_path: Optional[str],
    exposure_calibration_path: Optional[str],
    color_calibration_path: Optional[str],
    calibration_mode: Optional[str],
    sample_count: int,
    sampling_strategy: str,
    calibration_roi: Optional[Dict[str, float]],
    target_strategies: List[str],
    reference_clip_id: Optional[str],
    hero_clip_id: Optional[str] = None,
    exposure_anchor_mode: Optional[str] = None,
    manual_target_stops: Optional[float] = None,
    manual_target_ire: Optional[float] = None,
    selected_clip_ids: Optional[List[str]] = None,
    selected_clip_groups: Optional[List[str]] = None,
    clip_subset_file: Optional[str] = None,
    run_label: Optional[str] = None,
    matching_domain: str = "scene",
    review_mode: str = "full_contact_sheet",
    report_focus: str = "auto",
    preview_mode: str = "monitoring",
    preview_output_space: Optional[str] = None,
    preview_output_gamma: Optional[str] = None,
    preview_highlight_rolloff: Optional[str] = None,
    preview_shadow_rolloff: Optional[str] = None,
    preview_lut: Optional[str] = None,
    preview_still_format: str = "tiff",
    require_real_redline: bool = False,
    artifact_mode: str = "production",
    sphere_assist_file: Optional[str] = None,
    focus_validation: bool = False,
) -> Dict[str, object]:
    workflow_started_at = time.perf_counter()
    invocation_source = str(os.getenv("R3DMATCH_INVOCATION_SOURCE", "direct_cli") or "direct_cli")
    resolved_source_mode = normalize_source_mode(source_mode)
    subset_definition = _load_clip_subset_definition(clip_subset_file) if clip_subset_file else None
    merged_clip_ids = [str(item) for item in (selected_clip_ids or []) if str(item).strip()]
    merged_clip_groups = [str(item) for item in (selected_clip_groups or []) if str(item).strip()]
    if subset_definition:
        merged_clip_ids = list(dict.fromkeys([*subset_definition["clip_ids"], *merged_clip_ids]))
        merged_clip_groups = list(dict.fromkeys([*subset_definition["clip_groups"], *merged_clip_groups]))
        run_label = run_label or subset_definition.get("run_label")
    resolved_matching_domain = "perceptual"
    resolved_run_label = resolve_run_label(
        run_label=run_label,
        selected_clip_ids=merged_clip_ids,
        selected_clip_groups=merged_clip_groups,
    )
    resolved_out_dir = resolve_review_output_dir(
        out_dir,
        run_label=resolved_run_label,
        selected_clip_ids=merged_clip_ids,
        selected_clip_groups=merged_clip_groups,
    )
    progress_path = review_progress_path_for(resolved_out_dir)
    emit_review_progress(
        progress_path,
        phase="review_start",
        detail="Starting review calibration.",
        stage_label="Preparing review",
        review_mode=review_mode,
        elapsed_seconds=0.0,
        extra={
            "source_mode": resolved_source_mode,
            "run_label": resolved_run_label,
            "invocation_source": invocation_source,
        },
    )
    source_input_path = input_path
    ingest_manifest = None
    if resolved_source_mode == "ftps_camera_pull":
        if not ftps_reel:
            raise ValueError("FTPS source mode requires --ftps-reel.")
        if not ftps_clip_spec:
            raise ValueError("FTPS source mode requires --ftps-clips.")
        ingest_root = Path(ftps_local_root).expanduser().resolve() if str(ftps_local_root or "").strip() else Path(resolved_out_dir).expanduser().resolve() / "ingest"
        raise_if_cancelled("Run cancelled before FTPS ingest.")
        ingest_manifest = run_ftps_ingest_job(
            action="download",
            out_dir=str(ingest_root),
            reel_identifier=ftps_reel,
            clip_spec=ftps_clip_spec,
            requested_cameras=ftps_cameras,
            username=ftps_username,
            password=ftps_password,
            processing_requested_after_ingest=True,
        )
        source_input_path = str(ingest_root)
    raise_if_cancelled("Run cancelled before analysis.")
    emit_review_progress(
        progress_path,
        phase="analysis_dispatch",
        detail="Dispatching source analysis.",
        stage_label="Analyzing sources",
        review_mode=review_mode,
        elapsed_seconds=time.perf_counter() - workflow_started_at,
    )
    analyze_summary = analyze_path(
        source_input_path,
        out_dir=resolved_out_dir,
        mode=mode,
        backend=backend,
        lut_override=lut_override,
        calibration_path=calibration_path,
        exposure_calibration_path=exposure_calibration_path,
        color_calibration_path=color_calibration_path,
        calibration_mode=calibration_mode,
        sample_count=1 if normalize_review_mode(review_mode) == "lightweight_analysis" else sample_count,
        sampling_strategy=sampling_strategy,
        calibration_roi=calibration_roi,
        target_type=target_type,
        selected_clip_ids=merged_clip_ids,
        selected_clip_groups=merged_clip_groups,
        progress_path=str(progress_path),
        half_res_decode=normalize_review_mode(review_mode) == "lightweight_analysis",
        workload_trace_path=str(Path(resolved_out_dir).expanduser().resolve() / "measurement_workload_trace.json"),
        runtime_trace_path=str(Path(resolved_out_dir).expanduser().resolve() / "lightweight_runtime_trace.json"),
        invocation_source=invocation_source,
        measurement_source="rendered_preview_ipp2",
        sphere_assist_path=sphere_assist_file,
    )
    raise_if_cancelled("Run cancelled before review package generation.")
    emit_review_progress(
        progress_path,
        phase="report_build_start",
        detail="Building review package.",
        stage_label="Building report",
        review_mode=review_mode,
        elapsed_seconds=time.perf_counter() - workflow_started_at,
        extra={"clip_count": int(analyze_summary.get("clip_count", 0) or 0)},
    )
    package = build_review_package(
        source_input_path,
        out_dir=resolved_out_dir,
        exposure_calibration_path=exposure_calibration_path or calibration_path,
        color_calibration_path=color_calibration_path,
        target_type=target_type,
        processing_mode=processing_mode,
        run_label=resolved_run_label,
        matching_domain=resolved_matching_domain,
        review_mode=review_mode,
        report_focus=report_focus,
        selected_clip_ids=merged_clip_ids,
        selected_clip_groups=merged_clip_groups,
        preview_mode=preview_mode,
        preview_output_space=preview_output_space,
        preview_output_gamma=preview_output_gamma,
        preview_highlight_rolloff=preview_highlight_rolloff,
        preview_shadow_rolloff=preview_shadow_rolloff,
        preview_lut=preview_lut,
        preview_still_format=preview_still_format,
        calibration_roi=calibration_roi,
        target_strategies=target_strategies,
        reference_clip_id=reference_clip_id,
        hero_clip_id=hero_clip_id,
        exposure_anchor_mode=exposure_anchor_mode,
        manual_target_stops=manual_target_stops,
        manual_target_ire=manual_target_ire,
        require_real_redline=require_real_redline,
        focus_validation=focus_validation,
        artifact_mode=artifact_mode,
        source_mode=resolved_source_mode,
        source_mode_label_value=source_mode_label(resolved_source_mode),
        source_input_path=source_input_path,
        ingest_manifest=ingest_manifest,
        progress_path=str(progress_path),
    )
    package["analyze_summary"] = analyze_summary
    package["analysis_dir"] = resolved_out_dir
    package["run_label"] = resolved_run_label
    package["selected_clip_ids"] = merged_clip_ids
    package["selected_clip_groups"] = merged_clip_groups
    package["matching_domain"] = resolved_matching_domain
    package["matching_domain_label"] = matching_domain_label(resolved_matching_domain)
    package["source_mode"] = resolved_source_mode
    package["source_mode_label"] = source_mode_label(resolved_source_mode)
    package["source_input_path"] = source_input_path
    package["ingest_manifest"] = ingest_manifest
    package["clip_subset_file"] = subset_definition["path"] if subset_definition else None
    package["sphere_assist_file"] = str(Path(sphere_assist_file).expanduser().resolve()) if str(sphere_assist_file or "").strip() else None
    validation = validate_review_run_contract(resolved_out_dir)
    package["review_validation"] = validation
    package_manifest_path = package.get("package_manifest")
    if package_manifest_path:
        Path(str(package_manifest_path)).write_text(json.dumps(package, indent=2), encoding="utf-8")
    emit_review_progress(
        progress_path,
        phase="review_complete",
        detail="Review package complete.",
        stage_label="Complete",
        review_mode=review_mode,
        elapsed_seconds=time.perf_counter() - workflow_started_at,
        extra={"validation_status": package.get("review_validation", {}).get("status")},
    )
    if validation["status"] != "success":
        raise ReviewValidationError(_format_review_validation_failure(validation))
    return package


def _identity_cdl_payload() -> Dict[str, object]:
    return {
        "slope": [1.0, 1.0, 1.0],
        "offset": [0.0, 0.0, 0.0],
        "power": [1.0, 1.0, 1.0],
        "saturation": 1.0,
    }


def _normalize_cdl_payload(payload: Optional[Dict[str, object]]) -> Dict[str, object]:
    if not isinstance(payload, dict):
        return _identity_cdl_payload()
    return {
        "slope": [float(value) for value in payload.get("slope", [1.0, 1.0, 1.0])],
        "offset": [float(value) for value in payload.get("offset", [0.0, 0.0, 0.0])],
        "power": [float(value) for value in payload.get("power", [1.0, 1.0, 1.0])],
        "saturation": float(payload.get("saturation", 1.0)),
    }


def _approved_sidecar_payload_for_clip(clip: Dict[str, object]) -> Dict[str, object]:
    color_metrics = clip["metrics"]["color"]
    gains = color_metrics.get("rgb_gains_diagnostic") or color_metrics.get("rgb_gains")
    color_cdl = _normalize_cdl_payload(color_metrics.get("cdl"))
    cdl_enabled = not is_identity_cdl_payload(color_cdl)
    return {
        "clip_id": clip["clip_id"],
        "source_path": clip["source_path"],
        "schema": "r3dmatch_v2",
        "calibration_state": {
            "exposure_calibration_loaded": True,
            "exposure_baseline_applied_stops": clip["metrics"]["exposure"]["final_offset_stops"],
            "color_calibration_loaded": gains is not None,
            "rgb_neutral_gains": {"r": gains[0], "g": gains[1], "b": gains[2]} if gains else None,
            "color_gains_state": "approved",
        },
        "rmd_mapping": {
            "exposure": {"final_offset_stops": clip["metrics"]["exposure"]["final_offset_stops"]},
            "color": {
                "rgb_neutral_gains": gains,
                "cdl": color_cdl,
                "cdl_enabled": cdl_enabled,
            },
        },
    }


def _correction_key_for_clip(clip: Dict[str, object]) -> str:
    return group_key_from_clip_id(str(clip["clip_id"]))


def _correction_signature(clip: Dict[str, object]) -> tuple:
    color_metrics = clip["metrics"]["color"]
    cdl = _normalize_cdl_payload(color_metrics.get("cdl"))
    return (
        round(float(clip["metrics"]["exposure"]["final_offset_stops"]), 6),
        tuple(round(float(value), 6) for value in cdl["slope"]),
        tuple(round(float(value), 6) for value in cdl["offset"]),
        tuple(round(float(value), 6) for value in cdl["power"]),
        round(float(cdl["saturation"]), 6),
        bool(not is_identity_cdl_payload(cdl)),
    )


def _detect_source_root(source_paths: List[str]) -> Optional[str]:
    if not source_paths:
        return None
    common_path = Path(os.path.commonpath(source_paths))
    if common_path.suffix.lower() == ".r3d":
        common_path = common_path.parent
    return str(common_path)


def _relative_to_root(path: str, root: Optional[str]) -> Optional[str]:
    if root is None:
        return None
    try:
        return str(Path(path).resolve().relative_to(Path(root).resolve()))
    except ValueError:
        return os.path.relpath(path, root)


def _write_master_rmds_from_strategy(strategy_payload: Dict[str, object], *, out_dir: str) -> Dict[str, object]:
    raise_if_cancelled("Run cancelled before MasterRMD export.")
    target_dir = Path(out_dir).expanduser().resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    grouped: Dict[str, List[Dict[str, object]]] = {}
    for clip in strategy_payload["clips"]:
        grouped.setdefault(_correction_key_for_clip(clip), []).append(clip)

    exports: List[Dict[str, object]] = []
    clip_mappings: List[Dict[str, object]] = []
    source_paths = [str(clip["source_path"]) for clip in strategy_payload["clips"]]
    source_root = _detect_source_root(source_paths)

    for correction_key, clips in sorted(grouped.items()):
        raise_if_cancelled("Run cancelled while exporting MasterRMD files.")
        representative = clips[0]
        representative_signature = _correction_signature(representative)
        for candidate in clips[1:]:
            if _correction_signature(candidate) != representative_signature:
                raise ValueError(
                    f"Cannot export a single MasterRMD for correction key {correction_key}: "
                    f"approved clips in this group do not share identical corrections."
                )

        sidecar_like = _approved_sidecar_payload_for_clip(representative)
        path, metadata = write_rmd_for_clip_with_metadata(correction_key, sidecar_like, target_dir)
        exposure_offset = float(representative["metrics"]["exposure"]["final_offset_stops"])
        color_metrics = representative["metrics"]["color"]
        color_cdl = _normalize_cdl_payload(color_metrics.get("cdl"))
        cdl_enabled = bool(not is_identity_cdl_payload(color_cdl))
        export_record = {
            "correction_key": correction_key,
            "camera_identity": correction_key,
            "approved_strategy": strategy_payload["strategy_key"],
            "hero_clip_id": strategy_payload.get("hero_clip_id"),
            "reference_clip_id": strategy_payload.get("reference_clip_id"),
            "camera_group_key": representative["group_key"],
            "representative_clip_id": representative["clip_id"],
            "source_clip_ids": [clip["clip_id"] for clip in clips],
            "source_r3d_paths": [clip["source_path"] for clip in clips],
            "source_r3d_relative_paths": [_relative_to_root(str(clip["source_path"]), source_root) for clip in clips],
            "master_rmd_path": str(path),
            "master_rmd_name": path.name,
            "rmd_kind": metadata.get("rmd_kind"),
            "exposure_correction_stops": exposure_offset,
            "cdl_enabled": cdl_enabled,
            "cdl": color_cdl,
            "rgb_gains_diagnostic": color_metrics.get("rgb_gains_diagnostic") or color_metrics.get("rgb_gains"),
            "commit_values": representative.get("commit_values")
            or build_commit_values(
                exposure_adjust=exposure_offset,
                rgb_gains=color_metrics.get("rgb_gains_diagnostic") or color_metrics.get("rgb_gains"),
                confidence=representative["metrics"].get("confidence"),
                sample_log2_spread=representative["metrics"]["exposure"].get("neutral_sample_log2_spread"),
                sample_chromaticity_spread=representative["metrics"]["color"].get("neutral_sample_chromaticity_spread"),
            ),
        }
        exports.append(export_record)

        for clip in sorted(clips, key=lambda item: str(item["clip_id"])):
            raise_if_cancelled("Run cancelled while building approval clip mappings.")
            clip_cdl = _normalize_cdl_payload(clip["metrics"]["color"].get("cdl"))
            clip_mappings.append(
                {
                    "clip_id": str(clip["clip_id"]),
                    "source_r3d_path": str(clip["source_path"]),
                    "source_r3d_relative_path": _relative_to_root(str(clip["source_path"]), source_root),
                    "camera_group_key": str(clip["group_key"]),
                    "correction_key": correction_key,
                    "approved_strategy": strategy_payload["strategy_key"],
                    "hero_clip_id": strategy_payload.get("hero_clip_id"),
                    "reference_clip_id": strategy_payload.get("reference_clip_id"),
                    "master_rmd_path": str(path),
                    "master_rmd_name": path.name,
                    "exposure_correction_stops": float(clip["metrics"]["exposure"]["final_offset_stops"]),
                    "authored_cdl_summary": clip_cdl,
                    "cdl_enabled": bool(not is_identity_cdl_payload(clip_cdl)),
                    "is_hero_camera": bool(clip.get("is_hero_camera")),
                    "commit_values": clip.get("commit_values")
                    or build_commit_values(
                        exposure_adjust=float(clip["metrics"]["exposure"]["final_offset_stops"]),
                        rgb_gains=clip["metrics"]["color"].get("rgb_gains_diagnostic") or clip["metrics"]["color"].get("rgb_gains"),
                        confidence=clip["metrics"].get("confidence"),
                        sample_log2_spread=clip["metrics"]["exposure"].get("neutral_sample_log2_spread"),
                        sample_chromaticity_spread=clip["metrics"]["color"].get("neutral_sample_chromaticity_spread"),
                    ),
                }
            )

    return {
        "rmd_dir": str(target_dir),
        "folder_name": target_dir.name,
        "clip_count": len(strategy_payload["clips"]),
        "correction_key_count": len(exports),
        "correction_key_model": {
            "name": "group_key_from_clip_id",
            "description": "Uses the first two underscore-delimited clip ID tokens as the stable per-camera correction key.",
        },
        "source_root": source_root,
        "master_rmds": exports,
        "clip_mappings": clip_mappings,
    }


def _build_batch_manifest(
    *,
    analysis_root: Path,
    approval_root: Path,
    strategy_payload: Dict[str, object],
    master_rmd_manifest: Dict[str, object],
    run_label: Optional[str] = None,
    matching_domain: Optional[str] = None,
    source_mode: str = "local_folder",
    source_mode_label_value: Optional[str] = None,
    selected_clip_ids: Optional[List[str]] = None,
    selected_clip_groups: Optional[List[str]] = None,
) -> Dict[str, object]:
    raise_if_cancelled("Run cancelled before batch manifest generation.")
    batch_root = approval_root / "batch"
    batch_root.mkdir(parents=True, exist_ok=True)
    source_root = master_rmd_manifest.get("source_root")
    entries: List[Dict[str, object]] = []
    for item in master_rmd_manifest["clip_mappings"]:
        raise_if_cancelled("Run cancelled while building batch manifest.")
        master_rmd_path = Path(item["master_rmd_path"]).resolve()
        entries.append(
            {
                "clip_id": item["clip_id"],
                "source_r3d_path": item["source_r3d_path"],
                "source_r3d_relative_path": item.get("source_r3d_relative_path"),
                "camera_group_key": item["camera_group_key"],
                "correction_key": item["correction_key"],
                "approved_strategy": item["approved_strategy"],
                "approved_run_label": run_label,
                "hero_clip_id": item.get("hero_clip_id"),
                "reference_clip_id": item.get("reference_clip_id"),
                "master_rmd_path": str(master_rmd_path),
                "master_rmd_name": master_rmd_path.name,
                "master_rmd_relative_path": str(Path("..") / "MasterRMD" / master_rmd_path.name),
                "exposure_correction_stops": item["exposure_correction_stops"],
                "authored_cdl_summary": item["authored_cdl_summary"],
                "cdl_enabled": item["cdl_enabled"],
                "is_hero_camera": item["is_hero_camera"],
                "commit_values": item.get("commit_values"),
            }
        )

    manifest = {
        "analysis_dir": str(analysis_root),
        "approval_dir": str(approval_root),
        "batch_dir": str(batch_root),
        "run_label": run_label,
        "matching_domain": matching_domain,
        "matching_domain_label": matching_domain_label(matching_domain or "scene"),
        "source_mode": source_mode,
        "source_mode_label": source_mode_label_value or source_mode_label(source_mode),
        "selected_clip_ids": [str(item) for item in (selected_clip_ids or []) if str(item).strip()],
        "selected_clip_groups": [str(item) for item in (selected_clip_groups or []) if str(item).strip()],
        "approved_strategy": strategy_payload["strategy_key"],
        "hero_clip_id": strategy_payload.get("hero_clip_id"),
        "reference_clip_id": strategy_payload.get("reference_clip_id"),
        "source_root_detected_from_run": source_root,
        "master_rmd_dir": master_rmd_manifest["rmd_dir"],
        "correction_key_model": master_rmd_manifest["correction_key_model"],
        "entries": entries,
    }
    manifest_path = batch_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def _build_batch_readme(*, batch_root: Path, batch_manifest: Dict[str, object]) -> str:
    readme_path = batch_root / "README.txt"
    lines = [
        "R3DMatch MasterRMD Batch Handoff",
        "",
        "This batch folder maps each approved source clip to a production MasterRMD.",
        "Set SOURCE_ROOT in the generated shell script to the directory containing the source R3D media.",
        "MASTER_RMD_DIR defaults to ../MasterRMD relative to the batch folder.",
        "OUTPUT_ROOT controls where REDLine renders are written.",
        "",
        "Mapping model:",
        "- correction_key uses the first two underscore-delimited tokens from clip_id.",
        "- every clip listed in manifest.json is paired with the matching MasterRMD for that correction_key.",
        "",
        f"Approved strategy: {batch_manifest['approved_strategy']}",
        f"Hero clip: {batch_manifest.get('hero_clip_id')}",
        f"Detected source root from this run: {batch_manifest.get('source_root_detected_from_run')}",
        "",
        "Review manifest.json for exact clip-to-RMD mappings and authored correction payload summaries.",
        f"Run label: {batch_manifest.get('run_label')}",
        f"Matching domain: {batch_manifest.get('matching_domain_label')}",
        f"Source mode: {batch_manifest.get('source_mode_label')}",
    ]
    readme_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(readme_path)


def _render_batch_script_sh(*, batch_root: Path, batch_manifest: Dict[str, object]) -> str:
    script_path = batch_root / "transcode_with_master_rmd.sh"
    lines = [
        "#!/bin/sh",
        "set -eu",
        "",
        'SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"',
        'REDLINE_BIN="${REDLINE_BIN:-REDLine}"',
        'SOURCE_ROOT="${SOURCE_ROOT:-__SET_SOURCE_ROOT__}"',
        'MASTER_RMD_DIR="${MASTER_RMD_DIR:-$SCRIPT_DIR/../MasterRMD}"',
        'OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRIPT_DIR/output}"',
        'OUTPUT_EXT="${OUTPUT_EXT:-mov}"',
        "",
        'if [ "$SOURCE_ROOT" = "__SET_SOURCE_ROOT__" ]; then',
        '  echo "Set SOURCE_ROOT to the root directory containing the source R3D clips before running this script." >&2',
        "  exit 1",
        "fi",
        "",
        'mkdir -p "$OUTPUT_ROOT"',
        "",
        "# Add any production-specific REDLine output options directly to the commands below if needed.",
        "",
    ]
    for entry in batch_manifest["entries"]:
        source_relative = entry.get("source_r3d_relative_path") or entry["source_r3d_path"]
        lines.extend(
            [
                f"# clip_id: {entry['clip_id']}",
                f"# correction_key: {entry['correction_key']}",
                f"# master_rmd: {entry['master_rmd_name']}",
                f'"$REDLINE_BIN" --i "$SOURCE_ROOT/{source_relative}" --o "$OUTPUT_ROOT/{entry["clip_id"]}.$OUTPUT_EXT" --loadRMD "$MASTER_RMD_DIR/{entry["master_rmd_name"]}" --useRMD 1',
                "",
            ]
        )
    script_path.write_text("\n".join(lines), encoding="utf-8")
    script_path.chmod(0o755)
    return str(script_path)


def _render_batch_script_tcsh(*, batch_root: Path, batch_manifest: Dict[str, object]) -> str:
    script_path = batch_root / "transcode_with_master_rmd.tcsh"
    lines = [
        "#!/bin/tcsh",
        "set SCRIPT_DIR = `cd \"`dirname \"$0\"`\" && pwd`",
        "if ( ! $?REDLINE_BIN ) set REDLINE_BIN = REDLine",
        "if ( ! $?SOURCE_ROOT ) set SOURCE_ROOT = __SET_SOURCE_ROOT__",
        "if ( ! $?MASTER_RMD_DIR ) set MASTER_RMD_DIR = \"$SCRIPT_DIR/../MasterRMD\"",
        "if ( ! $?OUTPUT_ROOT ) set OUTPUT_ROOT = \"$SCRIPT_DIR/output\"",
        "if ( ! $?OUTPUT_EXT ) set OUTPUT_EXT = mov",
        "",
        "if ( \"$SOURCE_ROOT\" == \"__SET_SOURCE_ROOT__\" ) then",
        "  echo \"Set SOURCE_ROOT to the root directory containing the source R3D clips before running this script.\" >&2",
        "  exit 1",
        "endif",
        "",
        "mkdir -p \"$OUTPUT_ROOT\"",
        "",
        "# Add any production-specific REDLine output options directly to the commands below if needed.",
        "",
    ]
    for entry in batch_manifest["entries"]:
        source_relative = entry.get("source_r3d_relative_path") or entry["source_r3d_path"]
        lines.extend(
            [
                f"# clip_id: {entry['clip_id']}",
                f"# correction_key: {entry['correction_key']}",
                f"# master_rmd: {entry['master_rmd_name']}",
                f'"$REDLINE_BIN" --i "$SOURCE_ROOT/{source_relative}" --o "$OUTPUT_ROOT/{entry["clip_id"]}.$OUTPUT_EXT" --loadRMD "$MASTER_RMD_DIR/{entry["master_rmd_name"]}" --useRMD 1',
                "",
            ]
        )
    script_path.write_text("\n".join(lines), encoding="utf-8")
    script_path.chmod(0o755)
    return str(script_path)


def _load_review_package_payload(analysis_root: Path) -> Dict[str, object]:
    package_path = analysis_root / "report" / "review_package.json"
    if package_path.exists():
        return json.loads(package_path.read_text(encoding="utf-8"))
    return {}


def approve_master_rmd(
    analysis_dir: str,
    *,
    out_dir: Optional[str] = None,
    target_strategy: str = "median",
    reference_clip_id: Optional[str] = None,
    hero_clip_id: Optional[str] = None,
) -> Dict[str, object]:
    raise_if_cancelled("Run cancelled before approval export.")
    analysis_root = Path(analysis_dir).expanduser().resolve()
    approval_root = Path(out_dir).expanduser().resolve() if out_dir else analysis_root / "approval"
    approval_root.mkdir(parents=True, exist_ok=True)
    master_rmd_dir = approval_root / "MasterRMD"
    report_dir = analysis_root / "report"
    review_package = _load_review_package_payload(analysis_root)
    preview_settings = dict(review_package.get("preview_settings") or {})
    raise_if_cancelled("Run cancelled before rebuilding approval report.")
    report_payload = build_contact_sheet_report(
        str(analysis_root),
        out_dir=str(report_dir),
        clear_cache=False,
        source_mode=str(review_package.get("source_mode", "local_folder")),
        source_mode_label_value=str(review_package.get("source_mode_label", source_mode_label("local_folder"))),
        source_input_path=str(review_package.get("source_input_path", analysis_root)),
        ingest_manifest=review_package.get("ingest_manifest"),
        target_type=review_package.get("target_type"),
        processing_mode=review_package.get("processing_mode"),
        run_label=review_package.get("run_label"),
        matching_domain=str(review_package.get("matching_domain", "scene")),
        selected_clip_ids=review_package.get("selected_clip_ids"),
        selected_clip_groups=review_package.get("selected_clip_groups"),
        preview_mode=str(review_package.get("preview_mode", "monitoring")),
        preview_output_space=preview_settings.get("output_space"),
        preview_output_gamma=preview_settings.get("output_gamma"),
        preview_highlight_rolloff=preview_settings.get("highlight_rolloff"),
        preview_shadow_rolloff=preview_settings.get("shadow_rolloff"),
        preview_lut=preview_settings.get("lut_path"),
        calibration_roi=review_package.get("calibration_roi"),
        target_strategies=[target_strategy],
        reference_clip_id=reference_clip_id,
        hero_clip_id=hero_clip_id,
    )
    review_payload = json.loads(Path(report_payload["report_json"]).read_text(encoding="utf-8"))
    chosen_strategy = review_payload["strategies"][0]
    raise_if_cancelled("Run cancelled before MasterRMD export.")
    rmd_manifest = _write_master_rmds_from_strategy(chosen_strategy, out_dir=str(master_rmd_dir))
    batch_manifest = _build_batch_manifest(
        analysis_root=analysis_root,
        approval_root=approval_root,
        strategy_payload=chosen_strategy,
        master_rmd_manifest=rmd_manifest,
        run_label=review_package.get("run_label"),
        matching_domain=review_package.get("matching_domain"),
        source_mode=str(review_package.get("source_mode", "local_folder")),
        source_mode_label_value=str(review_package.get("source_mode_label", source_mode_label("local_folder"))),
        selected_clip_ids=review_package.get("selected_clip_ids"),
        selected_clip_groups=review_package.get("selected_clip_groups"),
    )
    batch_root = approval_root / "batch"
    batch_readme = _build_batch_readme(batch_root=batch_root, batch_manifest=batch_manifest)
    batch_script_sh = _render_batch_script_sh(batch_root=batch_root, batch_manifest=batch_manifest)
    batch_script_tcsh = _render_batch_script_tcsh(batch_root=batch_root, batch_manifest=batch_manifest)

    approval_timestamp = datetime.now(timezone.utc).isoformat()
    approval_pdf_path = render_contact_sheet_pdf(
        review_payload,
        output_path=str(approval_root / "calibration_report.pdf"),
        title="R3DMatch Approval Report",
        timestamp_label=f"Approved at: {approval_timestamp}",
    )
    commit_package = {
        "schema_version": "r3dmatch_commit_package_v1",
        "generated_at": approval_timestamp,
        "analysis_dir": str(analysis_root),
        "approval_dir": str(approval_root),
        "run_label": review_package.get("run_label"),
        "source_mode": review_package.get("source_mode", "local_folder"),
        "source_mode_label": review_package.get("source_mode_label", source_mode_label("local_folder")),
        "matching_domain": review_package.get("matching_domain"),
        "matching_domain_label": review_package.get("matching_domain_label"),
        "selected_strategy": chosen_strategy["strategy_key"],
        "selected_reference_clip_id": chosen_strategy.get("reference_clip_id"),
        "selected_hero_clip_id": chosen_strategy.get("hero_clip_id"),
        "per_camera_values": [
            {
                "clip_id": item["clip_id"],
                "correction_key": item["correction_key"],
                "camera_group_key": item["camera_group_key"],
                "inventory_camera_label": inventory_camera_label_from_clip_id(str(item["clip_id"])),
                "inventory_camera_ip": DEFAULT_CAMERA_IP_MAP.get(str(inventory_camera_label_from_clip_id(str(item["clip_id"])) or "").upper(), ""),
                "master_rmd_path": item["master_rmd_path"],
                "commit_values": item.get("commit_values"),
                "is_hero_camera": item.get("is_hero_camera"),
                "notes": ["Hero camera receives identity correction."] if item.get("is_hero_camera") else [],
            }
            for item in rmd_manifest["clip_mappings"]
        ],
    }
    commit_package_path = approval_root / "calibration_commit_package.json"
    commit_package_path.write_text(json.dumps(commit_package, indent=2), encoding="utf-8")
    manifest = {
        "workflow_phase": "approved_master",
        "approved_at": approval_timestamp,
        "analysis_dir": str(analysis_root),
        "master_rmd_dir": str(master_rmd_dir),
        "master_rmd_folder_name": master_rmd_dir.name,
        "report_json": report_payload["report_json"],
        "report_html": report_payload["report_html"],
        "calibration_report_pdf": approval_pdf_path,
        "selected_target_strategy": chosen_strategy["strategy_key"],
        "selected_reference_clip_id": chosen_strategy.get("reference_clip_id"),
        "selected_hero_clip_id": chosen_strategy.get("hero_clip_id"),
        "run_label": review_package.get("run_label"),
        "matching_domain": review_package.get("matching_domain"),
        "matching_domain_label": review_package.get("matching_domain_label"),
        "source_mode": review_package.get("source_mode", "local_folder"),
        "source_mode_label": review_package.get("source_mode_label", source_mode_label("local_folder")),
        "source_input_path": review_package.get("source_input_path", str(analysis_root)),
        "ingest_manifest": review_package.get("ingest_manifest"),
        "selected_clip_ids": review_package.get("selected_clip_ids"),
        "selected_clip_groups": review_package.get("selected_clip_groups"),
        "target_type": review_payload.get("target_type"),
        "processing_mode": review_payload.get("processing_mode"),
        "calibration_roi": review_payload.get("calibration_roi"),
        "preview_transform": review_payload.get("preview_transform"),
        "measurement_preview_transform": review_payload.get("measurement_preview_transform"),
        "exposure_measurement_domain": review_payload.get("exposure_measurement_domain"),
        "clip_count": rmd_manifest["clip_count"],
        "correction_key_count": rmd_manifest["correction_key_count"],
        "correction_key_model": rmd_manifest["correction_key_model"],
        "master_rmd_exports": rmd_manifest["master_rmds"],
        "clip_mappings": rmd_manifest["clip_mappings"],
        "batch_dir": str(batch_root),
        "batch_manifest": batch_manifest["manifest_path"],
        "batch_scripts": {
            "sh": batch_script_sh,
            "tcsh": batch_script_tcsh,
        },
        "batch_readme": batch_readme,
        "calibration_commit_package": str(commit_package_path),
    }
    manifest_path = approval_root / "approval_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest["approval_manifest"] = str(manifest_path)
    return manifest


__all__ = [
    "approve_master_rmd",
    "clear_preview_cache",
    "matching_domain_label",
    "normalize_matching_domain",
    "review_calibration",
    "resolve_review_output_dir",
    "validate_review_run_contract",
]
