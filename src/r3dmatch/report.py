from __future__ import annotations

import html
import json
import os
import shlex
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image

from .calibration import extract_center_region, load_color_calibration, load_exposure_calibration, percentile_clip
from .color import identity_lggs, is_identity_cdl_payload as _color_is_identity_cdl_payload, rgb_gains_to_cdl, solve_cdl_color_model
from .commit_values import build_commit_values, extract_as_shot_white_balance, solve_white_balance_model_for_records
from .execution import CancellationError, raise_if_cancelled, run_cancellable_subprocess
from .ftps_ingest import source_mode_label
from .rmd import write_rmd_for_clip_with_metadata, write_rmds_from_analysis


PREVIEW_VARIANTS = ("original", "exposure", "color", "both")
REVIEW_PREVIEW_TRANSFORM = "REDLine IPP2 Log3G10 / Medium / Medium"
DEFAULT_REVIEW_TARGET_STRATEGIES = ("median",)
STRATEGY_ORDER = ["median", "optimal_exposure", "manual", "hero_camera"]
DEFAULT_CALIBRATION_PREVIEW = {
    "preview_mode": "calibration",
    "output_space": "REDWideGamutRGB",
    "output_gamma": "Log3G10",
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
CAMERA_TRUST_CAUTION_CHROMA_SPREAD = 0.014
CAMERA_TRUST_LARGE_CORRECTION = 0.75
CAMERA_TRUST_EXTREME_CORRECTION = 1.0


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
    if resolved_domain == "perceptual":
        return {
            "log2_luminance": float(
                monitoring.get(
                    "measured_log2_luminance_monitoring",
                    diagnostics.get("measured_log2_luminance_monitoring", diagnostics.get("measured_log2_luminance", 0.0)),
                )
            ),
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
            "source": "rendered_preview" if monitoring else "analysis_diagnostic",
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
        "source": "scene_referred_analysis",
    }


def _measurement_preview_settings_for_domain(matching_domain: str) -> Dict[str, object]:
    normalized = _normalize_matching_domain(matching_domain)
    defaults = DEFAULT_CALIBRATION_PREVIEW if normalized == "scene" else DEFAULT_DISPLAY_REVIEW_PREVIEW
    return _normalize_preview_settings(
        preview_mode=str(defaults["preview_mode"]),
        preview_output_space=str(defaults["output_space"]),
        preview_output_gamma=str(defaults["output_gamma"]),
        preview_highlight_rolloff=str(defaults["highlight_rolloff"]),
        preview_shadow_rolloff=str(defaults["shadow_rolloff"]),
        preview_lut=None,
    )


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
        "median": "median",
        "manual": "manual",
    }
    if normalized not in aliases:
        raise ValueError("target strategy must be one of: median, optimal-exposure, manual, hero-camera")
    return aliases[normalized]


def strategy_display_name(name: str) -> str:
    return {
        "hero_camera": "Hero Camera",
        "median": "Median",
        "optimal_exposure": "Optimal Exposure (Best Match to Gray)",
        "manual": "Manual Reference",
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


def clear_preview_cache(input_path: str, *, report_dir: Optional[str] = None) -> Dict[str, object]:
    root = Path(input_path).expanduser().resolve()
    preview_root = root / "previews"
    report_root = Path(report_dir).expanduser().resolve() if report_dir else root / "report"
    removed: list[str] = []
    if preview_root.exists():
        for path in sorted(preview_root.glob("*.review.*")):
            path.unlink()
            removed.append(str(path))
        measurement_dir = preview_root / "_measurement"
        if measurement_dir.exists():
            shutil.rmtree(measurement_dir)
            removed.append(str(measurement_dir))
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
    executable = os.environ.get("R3DMATCH_REDLINE_EXECUTABLE", "REDLine")
    resolved = shutil.which(executable)
    if resolved:
        return resolved
    if executable != "REDLine":
        return executable
    return executable


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
    mode = preview_mode.lower()
    if mode not in {"calibration", "monitoring"}:
        raise ValueError("preview mode must be calibration or monitoring")
    defaults = DEFAULT_CALIBRATION_PREVIEW if mode == "calibration" else DEFAULT_MONITORING_PREVIEW
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
    return {
        "preview_mode": mode,
        "output_space": output_space,
        "output_gamma": output_gamma,
        "highlight_rolloff": highlight_rolloff,
        "shadow_rolloff": shadow_rolloff,
        "lut_path": lut_path,
        "output_tonemap": "medium",
    }


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


def _measure_rendered_preview_roi(preview_path: str, calibration_roi: Optional[Dict[str, float]]) -> Dict[str, object]:
    image = np.asarray(Image.open(preview_path).convert("RGB"), dtype=np.float32) / 255.0
    region = _extract_normalized_roi_region_hwc(image, calibration_roi)
    luminance = np.clip(region[..., 0] * 0.2126 + region[..., 1] * 0.7152 + region[..., 2] * 0.0722, 1e-6, 1.0)
    valid_mask = (luminance > 0.002) & (luminance < 0.998)
    pixels = region[valid_mask]
    if pixels.size == 0:
        pixels = region.reshape(-1, 3)
        luminance_values = luminance.reshape(-1)
    else:
        luminance_values = luminance[valid_mask]
    trimmed_luma = percentile_clip(luminance_values, 5.0, 95.0)
    measured_log2 = float(np.median(np.log2(trimmed_luma)))
    rgb_mean = np.median(pixels, axis=0)
    chroma = rgb_mean / max(float(np.sum(rgb_mean)), 1e-6)
    saturation_fraction = float(np.mean(np.max(pixels, axis=1) >= 0.998)) if pixels.size else 0.0
    return {
        "measured_log2_luminance_monitoring": measured_log2,
        "measured_rgb_mean_monitoring": [float(rgb_mean[0]), float(rgb_mean[1]), float(rgb_mean[2])],
        "measured_rgb_chromaticity_monitoring": [float(chroma[0]), float(chroma[1]), float(chroma[2])],
        "measured_saturation_fraction_monitoring": saturation_fraction,
        "valid_pixel_count_monitoring": int(pixels.shape[0]),
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
) -> List[Dict[str, object]]:
    if not analysis_records:
        return []
    resolved_matching_domain = _normalize_matching_domain(matching_domain)
    saturation_supported = _target_supports_saturation(target_type)
    resolved_quality_by_clip = quality_by_clip or {}

    def quality_for_record(record: Dict[str, object]) -> Dict[str, object]:
        return dict(resolved_quality_by_clip.get(str(record["clip_id"]), {}) or {})

    def confidence_for_record(record: Dict[str, object]) -> float:
        quality = quality_for_record(record)
        return float(quality.get("confidence", record.get("confidence", 0.0)) or 0.0)

    def sample_log2_spread_for_record(record: Dict[str, object]) -> float:
        measured = record.get("diagnostics", {})
        quality = quality_for_record(record)
        return float(quality.get("neutral_sample_log2_spread", measured.get("neutral_sample_log2_spread", 0.0)) or 0.0)

    def sample_chroma_spread_for_record(record: Dict[str, object]) -> float:
        measured = record.get("diagnostics", {})
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
        return float(
            _measurement_values_for_record(
                record,
                matching_domain=resolved_matching_domain,
                monitoring_measurements_by_clip=monitoring_measurements_by_clip,
            )["log2_luminance"]
        )

    def measured_rgb_for_record(record: Dict[str, object]) -> List[float]:
        return [
            float(value)
            for value in _measurement_values_for_record(
                record,
                matching_domain=resolved_matching_domain,
                monitoring_measurements_by_clip=monitoring_measurements_by_clip,
            )["rgb_chromaticity"]
        ]

    def measured_saturation_for_record(record: Dict[str, object]) -> float:
        return float(
            _measurement_values_for_record(
                record,
                matching_domain=resolved_matching_domain,
                monitoring_measurements_by_clip=monitoring_measurements_by_clip,
            )["saturation_fraction"]
        )

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
        if strategy == "median":
            target_log2 = float(np.median(measured_log2))
            target_rgb = [float(np.median(measured_chroma[:, index])) for index in range(3)]
            target_saturation = float(np.median(measured_saturation)) if measured_saturation.size and saturation_supported else 1.0
            resolved_reference = None
            resolved_hero = None
            strategy_summary = "Matched to the batch median target."
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

        strategy_clips = []
        wb_requests = []
        for record in analysis_records:
            measured = record.get("diagnostics", {})
            resolved_measurement = _measurement_values_for_record(
                record,
                matching_domain=resolved_matching_domain,
                monitoring_measurements_by_clip=monitoring_measurements_by_clip,
            )
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
                    "measured_log2_luminance": measured_monitoring_log2,
                    "measured_log2_luminance_monitoring": measured_monitoring_log2,
                    "measured_log2_luminance_raw": float(measured.get("measured_log2_luminance_raw", measured.get("measured_log2_luminance", 0.0))),
                    "measured_rgb_chromaticity": [float(value) for value in measured_rgb],
                    "monitoring_measurement_source": str(resolved_measurement["source"]),
                    "exposure_offset_stops": 0.0 if is_hero_camera else float(target_log2 - measured_monitoring_log2),
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
    colors = {"left": "#22c55e", "center": "#f59e0b", "right": "#ef4444"}
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
            f"<text x=\"{sample_x + 8:.1f}\" y=\"{sample_y + 22:.1f}\" font-size=\"16\" fill=\"{stroke}\">{html.escape(label.title())}</text>"
        )
    return (
        f"<svg viewBox=\"0 0 {width} {height}\" role=\"img\" aria-label=\"ROI overlay for {html.escape(str(clip_id))}\">"
        f"<rect x=\"0\" y=\"0\" width=\"{width}\" height=\"{height}\" fill=\"#0f172a\"/>"
        f"<rect x=\"18\" y=\"18\" width=\"{width - 36}\" height=\"{height - 36}\" fill=\"#111827\" stroke=\"#334155\" stroke-width=\"2\" rx=\"20\"/>"
        f"<text x=\"28\" y=\"42\" font-size=\"20\" fill=\"#e2e8f0\">{html.escape(str(clip_id))}</text>"
        f"<text x=\"28\" y=\"68\" font-size=\"14\" fill=\"#94a3b8\">Shared ROI and neutral-sample windows</text>"
        f"<rect x=\"{roi_x:.1f}\" y=\"{roi_y:.1f}\" width=\"{roi_w:.1f}\" height=\"{roi_h:.1f}\" fill=\"#38bdf8\" fill-opacity=\"0.08\" stroke=\"#38bdf8\" stroke-width=\"5\" rx=\"16\"/>"
        f"<text x=\"{roi_x + 10:.1f}\" y=\"{max(roi_y - 12, 22):.1f}\" font-size=\"16\" fill=\"#38bdf8\">Calibration ROI</text>"
        f"{''.join(sample_boxes)}"
        "</svg>"
    )


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
    readiness_label = {
        "READY": "SAFE TO COMMIT",
        "READY_WITH_WARNINGS": "COMMIT WITH WARNINGS",
        "REVIEW_REQUIRED": "REVIEW REQUIRED",
        "DO_NOT_PUSH": "DO NOT PUSH",
    }.get(str(run_assessment.get("status") or ""), "COMMIT WITH WARNINGS")
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
        f"{int(exposure_summary.get('outlier_count', 0) or 0)} camera excluded from reference, corrections still applied."
        if int(exposure_summary.get("outlier_count", 0) or 0) == 1
        else f"{int(exposure_summary.get('outlier_count', 0) or 0)} cameras excluded from reference, corrections still applied."
        if int(exposure_summary.get("outlier_count", 0) or 0) > 1
        else "No cameras excluded from reference in this subset."
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
        outlier_badge = "<span class='outlier-pill'>Outlier</span>" if row.get("is_outlier") else ""
        hero_badge = "<span class='hero-pill'>Hero</span>" if row.get("is_hero_camera") else ""
        status_label = str(row.get("trust_class") or "TRUSTED").replace("_", " ")
        status_class = "outlier-pill" if status_label == "EXCLUDED" else "warning-pill" if status_label in {"UNTRUSTED", "USE WITH CAUTION"} else "good-pill"
        reason_label = str(row.get("trust_reason") or "Stable gray sample")
        table_rows.append(
            "<tr>"
            f"<td><div class='camera-cell'><span class='{status_class}'>{html.escape(status_label)}</span><strong>{html.escape(str(row['camera_label']))}</strong><span>{html.escape(str(row['clip_id']))}</span><span>{html.escape(reason_label)}</span></div></td>"
            f"<td>{float(row['measured_log2_luminance']):.2f}</td>"
            f"<td>{float(row['raw_offset_stops']):.2f}</td>"
            f"<td>{float(row['final_offset_stops']):.2f}</td>"
            f"<td>{int(row['commit_values'].get('kelvin', 5600))} / {float(row['commit_values'].get('tint', 0.0)):.1f}</td>"
            f"<td>{float(row['pre_color_residual']):.4f}</td>"
            f"<td>{float(row['confidence']):.2f}<div class='subtle'>dL {float(row['neutral_sample_log2_spread']):.3f} | dC {float(row['neutral_sample_chromaticity_spread']):.4f}</div></td>"
            f"<td>{html.escape(str(row.get('reference_use') or 'Included'))}<div class='subtle'>{html.escape(str(row.get('correction_confidence') or ''))} correction confidence</div></td>"
            f"<td>{outlier_badge}{hero_badge}<div class='subtle'>{html.escape(str(row['note']))}</div></td>"
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
        <div class="decision-banner-kicker">Calibration Decision</div>
        <div class="decision-banner-title">{html.escape(readiness_label)}</div>
        <p class="decision-banner-copy">{html.escape(str(payload['operator_recommendation']))}</p>
        <p class="decision-banner-copy" style="margin-top:8px;font-size:15px;font-weight:700;">{html.escape(banner_subline)}</p>
        <div class="decision-metrics">
          <div><div class="eyebrow">Strategy</div><div class="recommendation">{html.escape(str(payload['recommended_strategy']['strategy_label']))}</div></div>
          <div><div class="eyebrow">Outliers</div><div class="recommendation">{int(exposure_summary['outlier_count'])}</div></div>
          <div><div class="eyebrow">Trusted Cameras</div><div class="recommendation">{int(run_assessment.get('trusted_camera_count', 0) or 0)}</div></div>
          <div><div class="eyebrow">Recommendation Strength</div><div class="recommendation">{html.escape(str(run_assessment.get('recommendation_strength') or 'MEDIUM_CONFIDENCE').replace('_', ' '))}</div></div>
        </div>
      </div>
      <dl class="meta-grid">
          <div class="meta-card"><dt>Created</dt><dd>{html.escape(str(payload['created_at']))}</dd></div>
          <div class="meta-card"><dt>Source Mode</dt><dd>{html.escape(str(payload['source_mode_label']))}</dd></div>
          <div class="meta-card"><dt>Target Type</dt><dd>{html.escape(str(payload['target_type']))}</dd></div>
          <div class="meta-card"><dt>Matching Domain</dt><dd>{html.escape(str(payload['matching_domain_label']))}</dd></div>
        <div class="meta-card"><dt>Strategies</dt><dd>{html.escape(str(payload['selected_strategy_labels']))}</dd></div>
        <div class="meta-card"><dt>Subset</dt><dd>{html.escape(str(payload['subset_label']))}</dd></div>
        <div class="meta-card"><dt>Recommendation</dt><dd>{html.escape(str(payload['recommended_strategy']['strategy_label']))}</dd></div>
        <div class="meta-card"><dt>Run Status</dt><dd>{html.escape(str(run_assessment.get('status') or 'READY_WITH_WARNINGS').replace('_', ' '))}</dd></div>
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
        <h2>Recommendation</h2>
        <p class="eyebrow">At A Glance</p>
        <p class="recommendation"><strong>{html.escape(summary_sentence)}</strong></p>
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
        <h2>Should I Trust This Run?</h2>
        <p class="lead">{html.escape(str(run_assessment.get('operator_note') or 'Review the trusted camera group before any later push.'))}</p>
        <ul class="bullet-list">
          <li>{html.escape(str(run_assessment.get('anchor_summary') or 'No anchor summary available.'))}</li>
          <li>{html.escape(f"Trusted cameras: {int(run_assessment.get('trusted_camera_count', 0) or 0)} / {int(run_assessment.get('camera_count', 0) or 0)}")}</li>
          <li>{html.escape(f"Excluded cameras: {int(run_assessment.get('excluded_camera_count', 0) or 0)}")}</li>
          <li>{html.escape('Safe to push later: yes' if bool(run_assessment.get('safe_to_push_later')) else 'Safe to push later: no')}</li>
          {''.join(f"<li>{html.escape(str(reason))}</li>" for reason in list(run_assessment.get('gating_reasons') or []))}
        </ul>
      </section>
      <section class="section">
        <h2>Trust & Eligibility</h2>
        <p class="lead">Trust classes summarize confidence, stability, cluster membership, and correction size using the current measured signals.</p>
        <div>{chart_frame('Camera Trust', payload['visuals']['trust_chart_svg'])}</div>
      </section>
    </div>

    <div class="grid two-up">
      <section class="section">
        <h2>Before / After Exposure</h2>
        <p class="lead">Each line shows where a camera measured before correction and where the chosen strategy aims to land it.</p>
        <div>{chart_frame('Before / After Exposure', payload['visuals']['before_after_exposure_svg'])}</div>
      </section>
      <section class="section">
        <h2>Confidence / Reliability</h2>
        <p class="lead">Higher bars indicate cameras with steadier neutral samples and more trustworthy measurements.</p>
        <div>{chart_frame('Confidence / Reliability', payload['visuals']['confidence_chart_svg'])}</div>
      </section>
    </div>

    <section class="section" style="margin-top:18px;">
      <h2>Sample Stability Ranking</h2>
      <p class="lead">Lower spread means a steadier gray reading. Cameras to the right of the threshold need more caution.</p>
      <div>{chart_frame('Sample Stability Ranking', payload['visuals']['stability_chart_svg'])}</div>
    </section>

    <section class="section" style="margin-top:18px;">
      <h2>Strategy Comparison</h2>
      <p class="lead">Strategies are ranked by correction size, anchor trustworthiness, and how safely they keep the array together.</p>
      <div class="strategy-grid">{''.join(strategy_cards)}</div>
    </section>

    <section class="section" style="margin-top:18px;">
      <h2>Per-Camera Analysis</h2>
      <p class="lead">Rows are sorted by severity so the riskiest cameras appear first.</p>
      <table>
        <thead>
          <tr>
            <th>Status / Camera / Clip</th>
            <th>Measured</th>
            <th>Raw Offset</th>
            <th>Recommended Offset</th>
            <th>Commit (K / Tint)</th>
            <th>Neutral Residual</th>
            <th>Confidence / Spread</th>
            <th>Reference Use</th>
            <th>Summary</th>
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
    clear_cache: bool = True,
) -> Dict[str, object]:
    raise_if_cancelled("Run cancelled before lightweight report generation.")
    root = Path(input_path).expanduser().resolve()
    out_root = Path(out_dir).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    if clear_cache:
        clear_preview_cache(str(root), report_dir=str(out_root))
    analysis_records = _load_analysis_records(input_path)
    resolved_matching_domain = _normalize_matching_domain(matching_domain)
    resolved_source_mode = source_mode
    resolved_source_mode_label = source_mode_label_value or source_mode_label(resolved_source_mode)
    resolved_strategies = [normalize_target_strategy_name(item) for item in (target_strategies or list(DEFAULT_REVIEW_TARGET_STRATEGIES))]
    resolved_run_label = run_label or root.name or "review"
    monitoring_measurements_by_clip: Dict[str, Dict[str, object]] = {}
    measurement_preview_rendered = 0
    array_calibration_payload = _load_array_calibration_payload(str(root))
    quality_by_clip = _quality_by_clip(array_calibration_payload)
    if resolved_matching_domain == "perceptual":
        measurement_preview_settings = _measurement_preview_settings_for_domain(resolved_matching_domain)
        redline_executable = _resolve_redline_executable()
        redline_capabilities = _detect_redline_capabilities(redline_executable)
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
        for record in analysis_records:
            clip_id = str(record["clip_id"])
            original_frame = measurement_preview_paths.get(clip_id, {}).get("original")
            if original_frame:
                monitoring_measurements_by_clip[clip_id] = _measure_rendered_preview_roi(
                    str(original_frame),
                    record.get("diagnostics", {}).get("calibration_roi") or calibration_roi,
                )
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
    recommended_strategy = _recommend_strategy(strategy_payloads)
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
        measured_log2 = float(strategy_clip["measured_log2_luminance"])
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
                "measured_log2_luminance": measured_log2,
                "raw_offset_stops": float(record.get("raw_offset_stops", 0.0) or 0.0),
                "final_offset_stops": float(strategy_clip["exposure_offset_stops"]),
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
                    f"{str(trust_details['trust_reason'])}. Exposure correction {float(strategy_clip['exposure_offset_stops']):+.2f}."
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
        "selected_clip_ids": [str(item) for item in (selected_clip_ids or []) if str(item).strip()],
        "selected_clip_groups": [str(item) for item in (selected_clip_groups or []) if str(item).strip()],
        "selected_strategy_labels": ", ".join(str(item["strategy_label"]) for item in strategy_summaries),
        "reference_clip_id": reference_clip_id,
        "hero_clip_id": hero_clip_id,
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
            "confidence_chart_svg": _build_confidence_chart_svg(recommended_payload["clips"]),
            "strategy_chart_svg": _build_strategy_chart_svg(strategy_summaries),
            "trust_chart_svg": _build_trust_chart_svg(per_camera_rows),
            "stability_chart_svg": _build_stability_chart_svg(per_camera_rows),
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
    rmd_path: Optional[str] = None,
    use_rmd_mode: int = 1,
) -> List[str]:
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
        if exposure_stops is not None:
            command.extend(["--exposure", f"{float(exposure_stops):.6f}"])
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
        if exposure_stops is not None:
            command.extend(["--exposure", f"{float(exposure_stops):.6f}"])
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
    red_gain: Optional[float] = None,
    green_gain: Optional[float] = None,
    blue_gain: Optional[float] = None,
    rmd_path: Optional[str] = None,
    use_rmd_mode: int = 1,
    color_method: Optional[str] = None,
) -> Dict[str, object]:
    raise_if_cancelled("Run cancelled before preview render.")
    color_cdl = None
    resolved_color_method = color_method
    if red_gain is not None or green_gain is not None or blue_gain is not None:
        color_cdl = {
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
        color_cdl=color_cdl,
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
        use_as_shot_metadata=False,
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
        preview_mode=str(DEFAULT_DISPLAY_REVIEW_PREVIEW["preview_mode"]),
        preview_output_space=str(DEFAULT_DISPLAY_REVIEW_PREVIEW["output_space"]),
        preview_output_gamma=str(DEFAULT_DISPLAY_REVIEW_PREVIEW["output_gamma"]),
        preview_highlight_rolloff=str(DEFAULT_DISPLAY_REVIEW_PREVIEW["highlight_rolloff"]),
        preview_shadow_rolloff=str(DEFAULT_DISPLAY_REVIEW_PREVIEW["shadow_rolloff"]),
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
                        use_as_shot_metadata=False,
                        exposure=None,
                        red_gain=None,
                        green_gain=None,
                        blue_gain=None,
                        rmd_path=look_metadata_path,
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
                requires_change = abs(float(variant_settings["exposure"])) > 1e-6 or (
                    variant_settings["gains"] is not None
                    and any(abs(float(value) - 1.0) > 1e-6 for value in variant_settings["gains"])
                )
                error_message = None
                if requires_change and not color_preview_disabled and (mean_diff is None or float(mean_diff) < 1e-3):
                    error_message = "RMD correction did not change rendered pixels"
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
                    "application_method": "preview_color_disabled" if color_preview_disabled else "rmd",
                    "rmd_path": look_metadata_path,
                    "use_rmd_mode": selected_rmd_use_mode,
                    "exposure_offset": float(variant_settings["exposure"]),
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
                    f"application_method={'preview_color_disabled' if color_preview_disabled else 'rmd'}"
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
                        "application_method": "preview_color_disabled" if color_preview_disabled else "rmd",
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
                        "as_shot_metadata_used": False,
                        "explicit_transform_used": True,
                        "explicit_correction_flags_used": False,
                        "correction_application_method": "rmd",
                        "use_rmd_mode": selected_rmd_use_mode,
                        "validation_method": "pixel_diff_from_baseline" if not color_preview_disabled else "preview_fallback_copy",
                    }
                )
                preview_paths[clip_id]["strategies"][strategy_key][variant] = str(preview_path)
    (preview_root / "preview_commands.json").write_text(
        json.dumps({"color_preview_policy": color_preview_policy, "commands": command_records}, indent=2),
        encoding="utf-8",
    )
    (preview_root / "rmd_validation.json").write_text(
        json.dumps(
            {
                "selected_use_rmd_mode": selected_rmd_use_mode,
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
            },
            indent=2,
        ),
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
    preview_mode: str = "calibration",
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
) -> Dict[str, object]:
    raise_if_cancelled("Run cancelled before report generation.")
    root = Path(input_path).expanduser().resolve()
    out_root = Path(out_dir).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    if clear_cache:
        clear_preview_cache(str(root), report_dir=str(out_root))
    analysis_records = _load_analysis_records(input_path)
    resolved_matching_domain = _normalize_matching_domain(matching_domain)
    resolved_source_mode = source_mode
    resolved_source_mode_label = source_mode_label_value or source_mode_label(resolved_source_mode)
    resolved_strategies = [normalize_target_strategy_name(item) for item in (target_strategies or list(DEFAULT_REVIEW_TARGET_STRATEGIES))]
    resolved_run_label = run_label or root.name or "review"
    run_id = resolved_run_label
    redline_executable = _resolve_redline_executable()
    redline_capabilities = _detect_redline_capabilities(redline_executable)
    color_preview_policy = _color_preview_policy()
    measurement_preview_settings = _measurement_preview_settings_for_domain(resolved_matching_domain)
    display_preview_settings = _normalize_preview_settings(
        preview_mode=str(DEFAULT_DISPLAY_REVIEW_PREVIEW["preview_mode"]),
        preview_output_space=preview_output_space or str(DEFAULT_DISPLAY_REVIEW_PREVIEW["output_space"]),
        preview_output_gamma=preview_output_gamma or str(DEFAULT_DISPLAY_REVIEW_PREVIEW["output_gamma"]),
        preview_highlight_rolloff=preview_highlight_rolloff or str(DEFAULT_DISPLAY_REVIEW_PREVIEW["highlight_rolloff"]),
        preview_shadow_rolloff=preview_shadow_rolloff or str(DEFAULT_DISPLAY_REVIEW_PREVIEW["shadow_rolloff"]),
        preview_lut=preview_lut,
    )
    exposure = load_exposure_calibration(exposure_calibration_path) if exposure_calibration_path else None
    color = load_color_calibration(color_calibration_path) if color_calibration_path else None
    exposure_by_group = {entry.group_key: entry for entry in exposure.cameras} if exposure else {}
    color_by_group = {entry.group_key: entry for entry in color.cameras} if color else {}
    array_calibration_payload = _load_array_calibration_payload(str(root))
    quality_by_clip = _quality_by_clip(array_calibration_payload)
    monitoring_measurements_by_clip = {}
    if resolved_matching_domain == "perceptual":
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
                monitoring_measurements_by_clip[clip_id] = _measure_rendered_preview_roi(
                    str(original_frame),
                    record.get("diagnostics", {}).get("calibration_roi") or calibration_roi,
                )
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
    )
    preview_paths = generate_preview_stills(
        input_path,
        analysis_records=analysis_records,
        previews_dir=str(root / "previews"),
        preview_settings=display_preview_settings,
        redline_capabilities=redline_capabilities,
        strategy_payloads=strategy_payloads,
        run_id=run_id,
        strategy_rmd_root=str(root / "review_rmd" / "strategies"),
        render_originals=True,
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
        shared_originals.append(
            {
                "clip_id": clip_id,
                "group_key": str(record["group_key"]),
                "source_path": record.get("source_path"),
                "original_frame": preview_paths.get(clip_id, {}).get("original"),
                "measured_log2_luminance": _measurement_values_for_record(
                    record,
                    matching_domain=resolved_matching_domain,
                    monitoring_measurements_by_clip=monitoring_measurements_by_clip,
                )["log2_luminance"],
                "measured_log2_luminance_monitoring": monitoring_measurements_by_clip.get(clip_id, {}).get(
                    "measured_log2_luminance_monitoring",
                    record.get("diagnostics", {}).get("measured_log2_luminance_monitoring"),
                ),
                "measured_log2_luminance_raw": record.get("diagnostics", {}).get("measured_log2_luminance_raw"),
                "confidence": record.get("confidence"),
            }
        )

    strategies = []
    for strategy_payload in strategy_payloads:
        raise_if_cancelled("Run cancelled while assembling strategy report payloads.")
        strategy_clips = []
        for record in analysis_records:
            raise_if_cancelled("Run cancelled while assembling strategy report payloads.")
            group_key = str(record["group_key"])
            clip_id = str(record["clip_id"])
            exposure_entry = exposure_by_group.get(group_key)
            color_entry = color_by_group.get(group_key)
            strategy_clip = next(item for item in strategy_payload["clips"] if item["clip_id"] == clip_id)
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
                    "metrics": {
                        "exposure": {
                            "raw_offset_stops": record.get("raw_offset_stops"),
                            "final_offset_stops": strategy_clip["exposure_offset_stops"],
                            "pre_residual_stops": strategy_clip.get("pre_exposure_residual_stops"),
                            "post_residual_stops": strategy_clip.get("post_exposure_residual_stops"),
                            "measured_log2_luminance": strategy_clip["measured_log2_luminance"],
                            "measured_log2_luminance_monitoring": strategy_clip["measured_log2_luminance_monitoring"],
                            "measured_log2_luminance_raw": strategy_clip["measured_log2_luminance_raw"],
                            "measurement_domain": strategy_clip["exposure_measurement_domain"],
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
                "calibration_roi": calibration_roi or (analysis_records[0].get("diagnostics", {}).get("calibration_roi") if analysis_records else None),
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
        "calibration_roi": calibration_roi or (analysis_records[0].get("diagnostics", {}).get("calibration_roi") if analysis_records else None),
        "run_id": run_id,
        "selected_clip_ids": [str(item) for item in (selected_clip_ids or []) if str(item).strip()],
        "selected_clip_groups": [str(item) for item in (selected_clip_groups or []) if str(item).strip()],
        "target_strategies": resolved_strategies,
        "reference_clip_id": reference_clip_id,
        "hero_clip_id": hero_clip_id,
        "clip_count": len(analysis_records),
        "shared_originals": shared_originals,
        "strategies": strategies,
        "clips": strategies[0]["clips"] if strategies else [],
        "strategy_review_rmd_root": str((root / "review_rmd" / "strategies").resolve()),
        "render_truth_summary": payload_render_truth,
    }
    strategy_summaries = [
        {
            "strategy_key": item["strategy_key"],
            "strategy_label": item["strategy_label"],
            "reference_clip_id": item.get("reference_clip_id"),
            "hero_clip_id": item.get("hero_clip_id"),
            "target_log2_luminance": float(item["target_log2_luminance"]),
            "summary": item.get("strategy_summary"),
            "correction_metrics": _strategy_distribution_metrics(item),
        }
        for item in strategy_payloads
    ]
    payload["recommended_strategy"] = _recommend_strategy(strategy_payloads) if strategy_payloads else None
    payload["hero_recommendation"] = _hero_candidate_summary(strategy_payloads)
    payload["exposure_summary"] = _exposure_summary(strategies[0]["clips"] if strategies else [])
    payload["strategy_comparison"] = strategy_summaries
    payload["visuals"] = {
        "exposure_plot_svg": _build_exposure_plot_svg(strategies[0]["clips"] if strategies else []),
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
            "Canonical preview/render correction currently uses the validated SDK-authored RMD path.",
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
    preview_mode: str = "calibration",
    preview_output_space: Optional[str] = None,
    preview_output_gamma: Optional[str] = None,
    preview_highlight_rolloff: Optional[str] = None,
    preview_shadow_rolloff: Optional[str] = None,
    preview_lut: Optional[str] = None,
    calibration_roi: Optional[Dict[str, float]] = None,
    target_strategies: Optional[List[str]] = None,
    reference_clip_id: Optional[str] = None,
    hero_clip_id: Optional[str] = None,
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
            clear_cache=True,
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
            clear_cache=True,
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


def render_contact_sheet_pdf(
    payload: Dict[str, object],
    *,
    output_path: str,
    title: str,
    timestamp_label: Optional[str] = None,
) -> str:
    raise_if_cancelled("Run cancelled before PDF rendering.")
    from PIL import Image, ImageDraw, ImageFont

    def load_font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
        preferred = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
        try:
            return ImageFont.truetype(preferred, size=size)
        except OSError:
            return ImageFont.load_default()

    title_font = load_font(38, bold=True)
    section_font = load_font(24, bold=True)
    clip_font = load_font(18, bold=True)
    body_font = load_font(15, bold=False)
    caption_font = load_font(13, bold=False)
    pages = []
    page_width = 1700
    page_height = 2200
    margin_x = 60
    margin_y = 60
    logo_max_width = 150
    logo_max_height = 84
    tile_gap_x = 20
    tile_gap_y = 36
    metadata_line_height = 22
    section_gap = 38

    logo_image = None
    if LOGO_PATH.exists():
        try:
            loaded_logo = Image.open(LOGO_PATH).convert("RGBA")
            loaded_logo.thumbnail((logo_max_width, logo_max_height))
            logo_image = loaded_logo
        except OSError:
            logo_image = None

    def new_page() -> tuple[Image.Image, ImageDraw.ImageDraw, int]:
        page = Image.new("RGB", (page_width, page_height), "white")
        draw = ImageDraw.Draw(page)
        y = margin_y
        text_x = margin_x
        if logo_image is not None:
            page.paste(logo_image, (margin_x, y), logo_image)
            text_x += logo_image.width + 28
        draw.text((text_x, y), title, fill="black", font=title_font)
        draw.text((text_x, y + 42), "Internal Review", fill="black", font=section_font)
        summary_lines = [
            f"Run label: {payload.get('run_label')}",
            f"Target type: {payload.get('target_type')}",
            f"Processing mode: {payload.get('processing_mode')}",
            f"Matching domain: {payload.get('matching_domain_label')}",
            f"Preview transform: {payload.get('preview_transform')}",
            f"Measurement transform: {payload.get('measurement_preview_transform')}",
            f"Calibration ROI: {payload.get('calibration_roi')}",
            f"Selected clip groups: {payload.get('selected_clip_groups') or 'all'}",
            f"Selected clip count: {len(payload.get('selected_clip_ids') or []) or payload.get('clip_count')}",
            f"Target strategies: {', '.join(strategy_display_name(item) for item in payload.get('target_strategies', []))}",
        ]
        summary_y = y + 74
        for line in summary_lines:
            draw.text((text_x, summary_y), line, fill="black", font=body_font)
            summary_y += 20
        if timestamp_label:
            draw.text((page_width - margin_x - 420, y + 8), timestamp_label, fill="black", font=body_font)
        header_bottom = max(summary_y, y + (logo_image.height if logo_image is not None else 96))
        draw.line((margin_x, header_bottom + 16, page_width - margin_x, header_bottom + 16), fill="#d7d7d7", width=2)
        return page, draw, header_bottom + 34

    def ensure_room(current_y: int, needed_height: int) -> tuple[Image.Image, ImageDraw.ImageDraw, int]:
        nonlocal page, draw
        if current_y + needed_height <= page_height - margin_y:
            return page, draw, current_y
        pages.append(page)
        page, draw, new_y = new_page()
        return page, draw, new_y

    def fit_image(path: str, tile_width: int, tile_height: int) -> Image.Image:
        preview = Image.open(path).convert("RGB")
        preview.thumbnail((tile_width, tile_height))
        canvas = Image.new("RGB", (tile_width, tile_height), "#f0ece3")
        paste_x = (tile_width - preview.width) // 2
        paste_y = (tile_height - preview.height) // 2
        canvas.paste(preview, (paste_x, paste_y))
        return canvas

    def draw_tiles(
        *,
        section_title: str,
        tiles: List[Dict[str, object]],
        section_meta: Optional[List[str]] = None,
    ) -> int:
        nonlocal page, draw, y
        columns = _report_grid_columns(len(tiles))
        tile_width = int((page_width - (2 * margin_x) - (tile_gap_x * (columns - 1))) / max(columns, 1))
        tile_width = max(180, tile_width)
        image_height = max(120, int(tile_width * 0.62))
        metadata_block_height = 128
        tile_block_height = image_height + metadata_block_height
        tiles_per_page = _report_tiles_per_page(columns)
        paged_tiles = _chunk_tiles(tiles, tiles_per_page)

        for page_index, page_tiles in enumerate(paged_tiles):
            raise_if_cancelled("Run cancelled during PDF pagination.")
            _, _, y = ensure_room(y, 140)
            heading = section_title if len(paged_tiles) == 1 else f"{section_title} ({page_index + 1}/{len(paged_tiles)})"
            draw.text((margin_x, y), heading, fill="black", font=section_font)
            y += 34
            local_meta = list(section_meta or [])
            if len(paged_tiles) > 1:
                local_meta.append(f"Page {page_index + 1} of {len(paged_tiles)}")
            for meta_line in local_meta:
                draw.text((margin_x, y), meta_line, fill="black", font=body_font)
                y += 20
            if local_meta:
                y += 8

            rows = (len(page_tiles) + columns - 1) // columns
            needed_height = max(1, rows) * (tile_block_height + tile_gap_y)
            _, _, y = ensure_room(y, needed_height + 20)

            for index, tile in enumerate(page_tiles):
                raise_if_cancelled("Run cancelled during PDF tile rendering.")
                row = index // columns
                col = index % columns
                tile_x = margin_x + col * (tile_width + tile_gap_x)
                tile_y = y + row * (tile_block_height + tile_gap_y)
                image_path = tile.get("image_path")
                if image_path and Path(str(image_path)).exists():
                    page.paste(fit_image(str(image_path), tile_width, image_height), (tile_x, tile_y))
                else:
                    draw.rounded_rectangle((tile_x, tile_y, tile_x + tile_width, tile_y + image_height), radius=14, fill="#f0ece3", outline="#d7d7d7")
                    draw.text((tile_x + 12, tile_y + 12), "No preview", fill="#666666", font=body_font)
                meta_y = tile_y + image_height + 8
                draw.text((tile_x, meta_y), str(tile.get("clip_id", "")), fill="black", font=clip_font)
                correction_text = str(tile.get("correction_text", "Exposure: n/a"))
                correction_fill = {"lift": "#15803d", "lower": "#b91c1c", "neutral": "#475569"}.get(str(tile.get("correction_tone") or ""), "#475569")
                draw.text((tile_x, meta_y + metadata_line_height), correction_text, fill=correction_fill, font=body_font)
                draw.text((tile_x, meta_y + metadata_line_height * 2), f"Commit: {tile.get('rgb_gains_text', 'n/a')}", fill="black", font=body_font)
                draw.text((tile_x, meta_y + metadata_line_height * 3), f"Confidence: {tile.get('confidence', 'n/a')} | log2: {tile.get('raw_log2', 'n/a')}", fill="#475569", font=caption_font)
                render_note = str(tile.get("render_truth_note") or "")
                if render_note:
                    draw.text((tile_x, meta_y + metadata_line_height * 4), render_note, fill="#1e293b", font=caption_font)
            y += needed_height + section_gap
            if page_index < len(paged_tiles) - 1:
                pages.append(page)
                page, draw, y = new_page()
        return y

    page, draw, y = new_page()

    original_tiles = [
        {
            "clip_id": f"{clip['clip_id']} [HERO]" if str(clip["clip_id"]) == str(payload.get("hero_clip_id") or "") else clip["clip_id"],
            "image_path": clip.get("original_frame"),
            "exposure_offset_stops": clip.get("measured_log2_luminance_monitoring"),
            "rgb_gains_text": "original",
            "confidence": clip.get("confidence"),
            "raw_log2": clip.get("measured_log2_luminance_raw"),
            "correction_text": "Exposure: original",
            "correction_tone": "neutral",
        }
        for clip in payload.get("shared_originals", [])
    ]
    draw_tiles(section_title="Original", tiles=original_tiles)

    strategies_by_key = {str(item["strategy_key"]): item for item in payload.get("strategies", [])}
    for strategy_key in STRATEGY_ORDER:
        raise_if_cancelled("Run cancelled during PDF strategy rendering.")
        strategy = strategies_by_key.get(strategy_key)
        if not strategy:
            continue
        tiles = []
        for clip in strategy.get("clips", []):
            gains_text = _format_cdl_gain_saturation(clip["metrics"]["color"])
            exposure_present = _exposure_correction_presentation(float(clip["metrics"]["exposure"]["final_offset_stops"]))
            render_validation = dict(clip.get("render_validation") or {})
            render_note = (
                f"Render proof: changed pixels ({float(render_validation.get('pixel_diff_from_baseline', 0.0) or 0.0):.2f} mean delta)"
                if render_validation.get("pixel_output_changed")
                else "Render proof: no visible change expected"
            )
            tiles.append(
                {
                    "clip_id": f"{clip['clip_id']} [HERO]" if clip.get("is_hero_camera") else clip["clip_id"],
                    "image_path": clip.get("both_corrected"),
                    "exposure_offset_stops": clip["metrics"]["exposure"]["final_offset_stops"],
                    "rgb_gains_text": f"K {int((clip['metrics'].get('commit_values') or {}).get('kelvin', 5600) or 5600)} | Tint {float((clip['metrics'].get('commit_values') or {}).get('tint', 0.0) or 0.0):.1f}",
                    "confidence": clip["metrics"]["confidence"],
                    "raw_log2": clip["metrics"]["exposure"]["measured_log2_luminance_raw"],
                    "correction_text": exposure_present["label"],
                    "correction_tone": exposure_present["tone"],
                    "render_truth_note": render_note,
                }
            )
        section_meta = [
            f"Summary: {strategy.get('strategy_summary')}",
            f"Reference clip: {strategy.get('reference_clip_id')}",
            f"Hero clip: {strategy.get('hero_clip_id')}",
            f"Target log2 luminance: {strategy.get('target_log2_luminance')}",
            f"Matching domain: {payload.get('matching_domain_label')} | Preview transform: {payload.get('preview_transform')}",
            f"Calibration ROI: {strategy.get('calibration_roi')}",
        ]
        if payload.get("color_preview_note"):
            section_meta.append(f"Color preview note: {payload['color_preview_note']}")
        draw_tiles(section_title=strategy["strategy_label"], tiles=tiles, section_meta=section_meta)
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


def render_contact_sheet_html(payload: Dict[str, object]) -> str:
    recommended_strategy = payload.get("recommended_strategy") or {}
    recommended_key = str(recommended_strategy.get("strategy_key") or "")
    recommended_payload = next((item for item in payload.get("strategies", []) if str(item.get("strategy_key")) == recommended_key), None)
    exposure_summary = payload.get("exposure_summary") or {}
    hero_summary = payload.get("hero_recommendation") or {}
    strategy_comparison = payload.get("strategy_comparison") or []
    logo_markup = (
        f"<img class='brand-logo' src='{LOGO_PATH.as_posix()}' alt='R3DMatch logo' />"
        if LOGO_PATH.exists()
        else "<div class='brand-logo-text'>R3DMatch</div>"
    )
    strategy_cards = []
    for strategy in strategy_comparison:
        recommended_badge = "<span class='recommended-badge'>Recommended</span>" if strategy.get("strategy_key") == recommended_key else ""
        reference_label = strategy.get("hero_clip_id") or strategy.get("reference_clip_id") or "Derived"
        strategy_cards.append(
            "<article class='strategy-card'>"
            f"<div class='strategy-card-top'><h3>{html.escape(str(strategy['strategy_label']))}</h3>{recommended_badge}</div>"
            f"<p>{html.escape(str(strategy.get('summary') or ''))}</p>"
            f"<div class='strategy-metrics'>Target {float(strategy['target_log2_luminance']):.2f} log2</div>"
            f"<div class='strategy-metrics'>Mean {float(strategy['correction_metrics']['mean_abs_offset']):.2f} / Max {float(strategy['correction_metrics']['max_abs_offset']):.2f} stops</div>"
            f"<div class='strategy-metrics subtle'>Anchor: {html.escape(str(reference_label))}</div>"
            "</article>"
        )
    calibration_rows = []
    for clip in (recommended_payload or {}).get("clips", []):
        metrics = clip["metrics"]
        commit_values = metrics.get("commit_values") or {}
        sample_spread = float(metrics["exposure"].get("neutral_sample_log2_spread", 0.0) or 0.0)
        outlier_marker = "Outlier" if sample_spread > 0.12 else ""
        hero_marker = "Hero" if clip.get("is_hero_camera") else ""
        flags = " | ".join(item for item in [hero_marker, outlier_marker] if item)
        calibration_rows.append(
            "<tr>"
            f"<td><div class='camera-cell'><strong>{html.escape(_camera_label_for_reporting(str(clip['clip_id'])))}</strong><span>{html.escape(str(clip['clip_id']))}</span></div></td>"
            f"<td>{float(metrics['exposure']['measured_log2_luminance_monitoring']):.2f}</td>"
            f"<td>{float(commit_values.get('exposureAdjust', 0.0) or 0.0):.3f}</td>"
            f"<td>{int(commit_values.get('kelvin', 5600) or 5600)}</td>"
            f"<td>{float(commit_values.get('tint', 0.0) or 0.0):.1f}</td>"
            f"<td>{float(metrics.get('confidence', 0.0) or 0.0):.2f}</td>"
            f"<td>{sample_spread:.3f}</td>"
            f"<td>{html.escape(flags or 'Normal')}</td>"
            "</tr>"
        )
    render_truth_summary = dict(payload.get("render_truth_summary") or {})
    total_originals = len(payload.get("shared_originals", []))
    original_columns = _report_grid_columns(total_originals)
    original_cards = []
    for clip in payload.get("shared_originals", []):
        path = clip.get("original_frame")
        is_hero_camera = str(clip["clip_id"]) == str(payload.get("hero_clip_id") or "")
        hero_badge = "<div class='hero-badge'>Hero</div>" if is_hero_camera else ""
        image_html = (
            f"<figure><img src='../previews/{Path(path).name}' alt='{clip['clip_id']} Original' /><figcaption>Original baseline</figcaption></figure>"
            if path else "<p class='subtle'>No original preview.</p>"
        )
        original_cards.append(
            f"<article class='clip-card{' hero' if is_hero_camera else ''}'>"
            f"{hero_badge}"
            f"{image_html}"
            f"<div class='clip-label'>{clip['clip_id']}</div>"
            f"<div class='clip-meta'>Measured log2: {float(clip.get('measured_log2_luminance_monitoring', 0.0) or 0.0):.2f}</div>"
            f"<div class='clip-meta subtle'>Group: {clip['group_key']}</div>"
            "</article>"
        )
    strategy_sections = []
    for strategy in payload.get("strategies", []):
        columns = _report_grid_columns(len(strategy.get("clips", [])))
        cards = []
        for clip in strategy["clips"]:
            metrics = clip["metrics"]
            image_path = clip.get("both_corrected")
            render_validation = dict(clip.get("render_validation") or {})
            exposure_present = _exposure_correction_presentation(float((metrics.get("exposure") or {}).get("final_offset_stops", 0.0) or 0.0))
            hero_badge_html = "<div class='hero-badge'>Hero Camera</div>" if clip.get("is_hero_camera") else ""
            preview_note_html = (
                f"<div class='clip-meta subtle'>Preview note: {html.escape(str(metrics.get('color_preview_note')))}</div>"
                if metrics.get("color_preview_note")
                else ""
            )
            thumbs_html = (
                f"<figure><img src='../previews/{Path(image_path).name}' alt='{clip['clip_id']} Corrected' /><figcaption>Proposed corrected still</figcaption></figure>"
                if image_path else "<p class='subtle'>No preview still available.</p>"
            )
            commit_values = metrics.get("commit_values") or {}
            cards.append(
                f"<article class='clip-card{' hero' if clip.get('is_hero_camera') else ''}'>"
                f"{hero_badge_html}"
                f"<div class='thumb-single'>{thumbs_html}</div>"
                f"<div class='clip-label'>{clip['clip_id']}</div>"
                f"<div class='correction-cue {html.escape(exposure_present['tone'])}'>{html.escape(exposure_present['short_reason'])}</div>"
                f"<div class='clip-meta'><strong>{html.escape(exposure_present['label'])}</strong></div>"
                f"<div class='clip-meta'>Commit: K {int(commit_values.get('kelvin', 5600) or 5600)} | Tint {float(commit_values.get('tint', 0.0) or 0.0):.1f}</div>"
                f"<div class='clip-meta'>Confidence: {float(metrics.get('confidence', 0.0) or 0.0):.2f} | sample spread {float(metrics['exposure'].get('neutral_sample_log2_spread', 0.0) or 0.0):.3f}</div>"
                f"<div class='clip-meta advanced subtle'>CDL gain/sat: {html.escape(_format_cdl_gain_saturation(metrics['color']))}</div>"
                f"<div class='render-proof'><strong>Render proof:</strong> "
                f"{'Corrected preview changed pixels relative to the original.' if render_validation.get('pixel_output_changed') else 'No visible pixel change was expected.'} "
                f"Mean delta: {float(render_validation.get('pixel_diff_from_baseline', 0.0) or 0.0):.2f}. "
                f"Asset: {html.escape(Path(str(image_path)).name if image_path else 'n/a')}</div>"
                f"{preview_note_html}"
                "</article>"
            )
        strategy_sections.append(
            f"<section class='report-section'>"
            f"<div class='section-top'><h2>{strategy['strategy_label']}</h2><p class='section-meta'>{html.escape(str(strategy.get('strategy_summary') or ''))}</p></div>"
            f"<div class='grid cols-{columns}'>{''.join(cards)}</div></section>"
        )
    color_preview_line = (
        f"Color preview: {html.escape(str(payload.get('color_preview_status')))} | {html.escape(str(payload.get('color_preview_note')))}"
        if payload.get("color_preview_note")
        else f"Color preview: {html.escape(str(payload.get('color_preview_status') or 'unknown'))}"
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>R3DMatch Review Contact Sheet</title>
  <style>
    body {{ margin: 0; background: #eef2f7; color: #111827; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    .page {{ max-width: 1560px; margin: 0 auto; padding: 34px; }}
    .shell {{ display: grid; gap: 20px; }}
    .hero, .panel, .report-section {{ background: rgba(255,255,255,0.98); border: 1px solid #d8dee8; border-radius: 24px; box-shadow: 0 18px 44px rgba(15,23,42,0.06); }}
    .hero {{ padding: 34px; }}
    .hero-top {{ display: flex; align-items: flex-start; gap: 20px; }}
    .brand-logo {{ width: 150px; height: auto; object-fit: contain; }}
    .brand-logo-text {{ font-size:30px; font-weight: 800; }}
    .eyebrow {{ font-size: 12px; font-weight: 800; letter-spacing: 0.12em; text-transform: uppercase; color: #64748b; }}
    h1 {{ margin: 6px 0 10px 0; font-size: 38px; line-height: 1.05; }}
    .hero-subtitle {{ margin: 0; font-size: 20px; color: #334155; }}
    .synopsis {{ margin-top: 20px; font-size: 22px; line-height: 1.7; color: #1e293b; max-width: 74ch; font-weight: 700; }}
    .meta-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin-top: 22px; }}
    .meta-card {{ background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 18px; padding: 16px; }}
    .meta-card dt {{ font-size: 11px; font-weight: 800; letter-spacing: 0.08em; text-transform: uppercase; color: #64748b; }}
    .meta-card dd {{ margin: 8px 0 0 0; font-size: 17px; font-weight: 800; }}
    .two-up {{ display: grid; grid-template-columns: 1.25fr 1fr; gap: 20px; }}
    .panel {{ padding: 22px; }}
    .panel h2, .report-section h2 {{ margin: 0 0 10px 0; font-size: 30px; line-height: 1.08; }}
    .panel p.lead, .section-meta {{ margin: 0 0 16px 0; color: #475569; font-size: 18px; line-height: 1.75; }}
    .stats {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin-bottom: 14px; }}
    .stat {{ border-radius: 18px; background: #f8fafc; border: 1px solid #e2e8f0; padding: 16px; }}
    .stat .label {{ font-size: 11px; font-weight: 800; letter-spacing: 0.08em; text-transform: uppercase; color: #64748b; }}
    .stat .value {{ margin-top: 8px; font-size: 26px; font-weight: 800; }}
    .strategy-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 14px; }}
    .strategy-card {{ border-radius: 18px; background: #f8fafc; border: 1px solid #dbe3ee; padding: 16px; }}
    .strategy-card-top {{ display: flex; justify-content: space-between; gap: 12px; align-items: center; }}
    .strategy-card h3 {{ margin: 0; font-size:20px; }}
    .strategy-metrics {{ font-size: 15px; margin-top: 8px; color: #1e293b; }}
    .recommended-badge {{ display: inline-flex; align-items: center; padding: 4px 8px; border-radius: 999px; background: #dbeafe; color: #1d4ed8; font-size: 11px; font-weight: 800; letter-spacing: 0.06em; text-transform: uppercase; }}
    .chart-frame {{ border-radius: 18px; background: #f8fafc; border: 1px solid #dbe3ee; padding: 18px; margin-top: 16px; overflow: auto; }}
    .chart-frame svg {{ display: block; width: 100%; height: auto; min-height: 520px; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 14px; }}
    th, td {{ padding: 16px 12px; border-bottom: 1px solid #e5e7eb; text-align: left; vertical-align: top; font-size: 16px; }}
    th {{ font-size: 12px; font-weight: 800; letter-spacing: 0.08em; text-transform: uppercase; color: #64748b; }}
    .camera-cell {{ display: flex; flex-direction: column; gap: 4px; }}
    .camera-cell span {{ font-size: 12px; color: #64748b; }}
    .report-section {{ padding: 22px; }}
    .section-top {{ display: flex; justify-content: space-between; gap: 18px; align-items: baseline; margin-bottom: 14px; flex-wrap: wrap; }}
    .grid {{ display: grid; gap: 18px; align-items: start; }}
    .grid.cols-3 {{ grid-template-columns: repeat(3, minmax(0,1fr)); }}
    .grid.cols-4 {{ grid-template-columns: repeat(4, minmax(0,1fr)); }}
    .grid.cols-6 {{ grid-template-columns: repeat(6, minmax(0,1fr)); }}
    .grid.cols-8 {{ grid-template-columns: repeat(8, minmax(0,1fr)); }}
    .clip-card {{ background: white; border-radius: 20px; padding: 18px; border: 1px solid #e2e8f0; }}
    .clip-card.hero {{ border: 2px solid #b91c1c; }}
    .hero-badge {{ display: inline-flex; align-items: center; padding: 5px 9px; border-radius: 999px; background: #b91c1c; color: white; font-size: 11px; font-weight: 800; letter-spacing: 0.06em; text-transform: uppercase; margin-bottom: 10px; }}
    .thumb-single figure, figure {{ margin: 0; }}
    .thumb-single img, figure img {{ width: 100%; display: block; border-radius: 12px; background: #d7dce4; aspect-ratio: 16 / 10; object-fit: cover; }}
    figcaption {{ margin-top: 8px; font-size: 13px; color: #64748b; }}
    .clip-label {{ font-size: 20px; font-weight: 800; margin-top: 14px; line-height: 1.2; }}
    .clip-meta {{ font-size: 15px; line-height: 1.65; margin-top: 8px; }}
    .clip-meta.advanced {{ font-size: 13px; }}
    .correction-cue {{ margin-top: 10px; display: inline-flex; align-items: center; padding: 6px 10px; border-radius: 999px; font-size: 13px; font-weight: 800; letter-spacing: 0.05em; text-transform: uppercase; }}
    .correction-cue.lift {{ background: #dcfce7; color: #166534; }}
    .correction-cue.lower {{ background: #fee2e2; color: #991b1b; }}
    .correction-cue.neutral {{ background: #e2e8f0; color: #475569; }}
    .render-proof {{ margin-top: 10px; padding-top: 10px; border-top: 1px solid #e2e8f0; font-size: 13px; line-height: 1.6; color: #334155; }}
    .truth-panel {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin-top: 18px; }}
    .truth-card {{ background: #f8fafc; border: 1px solid #dbe3ee; border-radius: 16px; padding: 14px 16px; }}
    .truth-card strong {{ display: block; font-size: 24px; margin-top: 6px; }}
    .subtle {{ color: #64748b; }}
    @media (max-width: 980px) {{ .two-up {{ grid-template-columns: 1fr; }} .stats {{ grid-template-columns: repeat(2, minmax(0,1fr)); }} }}
    @media print {{ body {{ background: white; }} .page {{ padding: 12px; }} .hero, .panel, .report-section {{ box-shadow: none; }} }}
  </style>
</head>
<body>
  <div class="page">
    <div class="shell">
      <section class="hero">
        <div class="hero-top">
          {logo_markup}
          <div>
            <div class="eyebrow">Multi-Camera RED Calibration Review</div>
            <h1>R3DMatch Decision Report</h1>
            <p class="hero-subtitle">A customer-facing review of exposure consistency, strategy behavior, and proposed per-camera commit values.</p>
            <p class="section-meta">Hero clip: {html.escape(str(payload.get('hero_clip_id') or 'None selected'))} | {color_preview_line}</p>
          </div>
        </div>
        <p class="synopsis">{html.escape(str(payload.get('executive_synopsis') or ''))}</p>
        <div class="truth-panel">
          <div class="truth-card"><div class="eyebrow">Corrected Renders</div><strong>{int(render_truth_summary.get('corrected_render_count', 0) or 0)}</strong></div>
          <div class="truth-card"><div class="eyebrow">Changed Pixels</div><strong>{int(render_truth_summary.get('changed_render_count', 0) or 0)}</strong></div>
          <div class="truth-card"><div class="eyebrow">Unexpected No-Change</div><strong>{int(render_truth_summary.get('unchanged_non_identity_count', 0) or 0)}</strong></div>
        </div>
        <dl class="meta-grid">
          <div class="meta-card"><dt>Run Label</dt><dd>{html.escape(str(payload.get('run_label')))}</dd></div>
          <div class="meta-card"><dt>Source Mode</dt><dd>{html.escape(str(payload.get('source_mode_label')))}</dd></div>
          <div class="meta-card"><dt>Target Type</dt><dd>{html.escape(str(payload.get('target_type')))}</dd></div>
          <div class="meta-card"><dt>Matching Domain</dt><dd>{html.escape(str(payload.get('matching_domain_label')))}</dd></div>
          <div class="meta-card"><dt>Chosen Method</dt><dd>{html.escape(str((recommended_strategy or {}).get('strategy_label') or 'Pending'))}</dd></div>
          <div class="meta-card"><dt>Created</dt><dd>{html.escape(str(payload.get('created_at')))}</dd></div>
        </dl>
      </section>

      <div class="two-up">
        <section class="panel">
          <h2>Exposure Consistency Summary</h2>
          <p class="lead">Exposure remains the most trustworthy review layer on this build. This panel shows how tightly the array is grouped before approval.</p>
          <div class="stats">
            <div class="stat"><div class="label">Median</div><div class="value">{float(exposure_summary.get('median', 0.0) or 0.0):.2f}</div></div>
            <div class="stat"><div class="label">Range</div><div class="value">{float(exposure_summary.get('spread', 0.0) or 0.0):.2f}</div></div>
            <div class="stat"><div class="label">Min / Max</div><div class="value">{float(exposure_summary.get('minimum', 0.0) or 0.0):.2f} / {float(exposure_summary.get('maximum', 0.0) or 0.0):.2f}</div></div>
            <div class="stat"><div class="label">Outliers</div><div class="value">{int(exposure_summary.get('outlier_count', 0) or 0)}</div></div>
          </div>
          <div class="chart-frame">{payload.get('visuals', {}).get('exposure_plot_svg', '')}</div>
        </section>

        <section class="panel">
          <h2>Strategy Decision Panel</h2>
          <p class="lead">{html.escape(str((recommended_strategy or {}).get('reason') or 'No recommendation available.'))}</p>
          <div class="strategy-grid">{''.join(strategy_cards)}</div>
          <div class="chart-frame">{payload.get('visuals', {}).get('strategy_chart_svg', '')}</div>
          <p class="lead" style="margin-top:12px;"><strong>Hero recommendation:</strong> {html.escape(str(hero_summary.get('candidate_clip_id') or 'No clear hero'))}. {html.escape(str(hero_summary.get('reason') or ''))}</p>
          <p class="section-meta">Preview policy: exposure previews are active; color/CDL is computed and exported, but preview color remains conservatively disabled on this build.</p>
        </section>
      </div>

      <section class="panel">
        <h2>Per-Camera Calibration Table</h2>
        <p class="lead">Commit values are centered on exposureAdjust, kelvin, and tint so the operator can review the exact proposed calibration package at a glance.</p>
        <table>
          <thead>
            <tr>
              <th>Camera / Clip</th>
              <th>Measured</th>
              <th>exposureAdjust</th>
              <th>Kelvin</th>
              <th>Tint</th>
              <th>Confidence</th>
              <th>Sample Spread</th>
              <th>Note</th>
            </tr>
          </thead>
          <tbody>{''.join(calibration_rows)}</tbody>
        </table>
      </section>

      <section class="report-section">
        <div class="section-top"><h2>Visual Confirmation Stills</h2><p class="section-meta">Original frames are shown once as the baseline reference. The strategy sections below use strategy-specific corrected assets.</p></div>
        <div class="grid cols-{original_columns}">{''.join(original_cards)}</div>
      </section>

      {''.join(strategy_sections)}
    </div>
  </div>
</body>
</html>"""
