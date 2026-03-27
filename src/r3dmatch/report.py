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
STRATEGY_ORDER = ["median", "brightest_valid", "manual", "hero_camera"]
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
        "brightest_valid": "brightest_valid",
        "best_exposed": "brightest_valid",
        "hero_camera": "hero_camera",
        "median": "median",
        "manual": "manual",
    }
    if normalized not in aliases:
        raise ValueError("target strategy must be one of: median, brightest-valid, manual, hero-camera")
    return aliases[normalized]


def strategy_display_name(name: str) -> str:
    return {
        "hero_camera": "Hero Camera",
        "median": "Median",
        "brightest_valid": "Brightest Valid",
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
) -> List[Dict[str, object]]:
    if not analysis_records:
        return []
    resolved_matching_domain = _normalize_matching_domain(matching_domain)
    saturation_supported = _target_supports_saturation(target_type)

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
    payloads: List[Dict[str, object]] = []
    for requested in target_strategies:
        strategy = normalize_target_strategy_name(requested)
        if strategy == "median":
            target_log2 = float(np.median(measured_log2))
            target_rgb = [float(np.median(measured_chroma[:, index])) for index in range(3)]
            target_saturation = float(np.median(measured_saturation)) if measured_saturation.size and saturation_supported else 1.0
            resolved_reference = None
            resolved_hero = None
            strategy_summary = "Matched to the batch median target."
        elif strategy == "brightest_valid":
            target_index = int(np.argmax(measured_log2))
            target_log2 = float(measured_log2[target_index])
            target_rgb = [float(value) for value in measured_chroma[target_index]]
            target_saturation = float(measured_saturation[target_index]) if measured_saturation.size and saturation_supported else 1.0
            resolved_reference = str(analysis_records[target_index]["clip_id"])
            resolved_hero = None
            strategy_summary = f"Matched to brightest valid clip {resolved_reference}."
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
                    "confidence": float(record.get("confidence", 0.0) or 0.0),
                    "sample_log2_spread": float(measured.get("neutral_sample_log2_spread", 0.0) or 0.0),
                    "sample_chromaticity_spread": float(measured.get("neutral_sample_chromaticity_spread", 0.0) or 0.0),
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
                    "confidence": float(record.get("confidence", 0.0)),
                    "flags": list(record.get("flags", [])),
                    "calibration_roi": measured.get("calibration_roi"),
                    "measurement_mode": measured.get("calibration_measurement_mode"),
                    "exposure_measurement_domain": resolved_matching_domain,
                    "color_measurement_domain": resolved_matching_domain,
                    "neutral_sample_count": int(measured.get("neutral_sample_count", 0) or 0),
                    "neutral_sample_log2_spread": float(measured.get("neutral_sample_log2_spread", 0.0) or 0.0),
                    "neutral_sample_chromaticity_spread": float(measured.get("neutral_sample_chromaticity_spread", 0.0) or 0.0),
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
    if offsets.size == 0:
        return {
            "mean_abs_offset": 0.0,
            "median_abs_offset": 0.0,
            "max_abs_offset": 0.0,
            "p90_abs_offset": 0.0,
            "mean_color_residual": 0.0,
            "max_color_residual": 0.0,
            "mean_confidence_penalty": 0.0,
        }
    return {
        "mean_abs_offset": float(np.mean(offsets)),
        "median_abs_offset": float(np.median(offsets)),
        "max_abs_offset": float(np.max(offsets)),
        "p90_abs_offset": float(np.percentile(offsets, 90)),
        "mean_color_residual": float(np.mean(color_residuals)) if color_residuals.size else 0.0,
        "max_color_residual": float(np.max(color_residuals)) if color_residuals.size else 0.0,
        "mean_confidence_penalty": float(np.mean(confidence_penalty)) if confidence_penalty.size else 0.0,
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


def _build_exposure_plot_svg(clips: List[Dict[str, object]]) -> str:
    if not clips:
        return ""
    ordered = sorted(clips, key=lambda clip: float(clip.get("measured_log2_luminance_monitoring", 0.0) or 0.0))
    values = [float(clip.get("measured_log2_luminance_monitoring", 0.0) or 0.0) for clip in ordered]
    minimum = min(values)
    maximum = max(values)
    width = 920
    height = 220
    pad_left = 80
    pad_right = 36
    pad_top = 28
    pad_bottom = 54
    inner_width = width - pad_left - pad_right
    inner_height = height - pad_top - pad_bottom
    scale = (maximum - minimum) or 1.0
    step = inner_width / max(len(ordered) - 1, 1)
    points: List[str] = []
    labels: List[str] = []
    for index, clip in enumerate(ordered):
        x = pad_left + step * index
        value = float(clip.get("measured_log2_luminance_monitoring", 0.0) or 0.0)
        y = pad_top + inner_height - ((value - minimum) / scale) * inner_height
        points.append(f"{x:.1f},{y:.1f}")
        labels.append(
            f"<text x=\"{x:.1f}\" y=\"{height - 20}\" text-anchor=\"middle\" font-size=\"11\" fill=\"#64748b\">{html.escape(_camera_label_for_reporting(str(clip['clip_id'])))}</text>"
        )
    polyline = " ".join(points)
    grid_lines = []
    for tick in range(4):
        y = pad_top + (inner_height / 3.0) * tick
        value = maximum - ((maximum - minimum) / 3.0) * tick
        grid_lines.append(f"<line x1=\"{pad_left}\" y1=\"{y:.1f}\" x2=\"{width - pad_right}\" y2=\"{y:.1f}\" stroke=\"#e2e8f0\" stroke-width=\"1\"/>")
        grid_lines.append(f"<text x=\"{pad_left - 12}\" y=\"{y + 4:.1f}\" text-anchor=\"end\" font-size=\"11\" fill=\"#64748b\">{value:.2f}</text>")
    circles = []
    for point in points:
        x, y = point.split(",")
        circles.append(f"<circle cx=\"{x}\" cy=\"{y}\" r=\"5\" fill=\"#0f172a\"/>")
    return (
        f"<svg viewBox=\"0 0 {width} {height}\" role=\"img\" aria-label=\"Per-camera exposure plot\">"
        f"{''.join(grid_lines)}"
        f"<polyline fill=\"none\" stroke=\"#2563eb\" stroke-width=\"3\" points=\"{polyline}\"/>"
        f"{''.join(circles)}"
        f"{''.join(labels)}"
        "</svg>"
    )


def _build_strategy_chart_svg(strategy_summaries: List[Dict[str, object]]) -> str:
    if not strategy_summaries:
        return ""
    width = 680
    row_height = 42
    height = 70 + row_height * len(strategy_summaries)
    max_value = max(float(item["correction_metrics"]["max_abs_offset"]) for item in strategy_summaries) or 1.0
    bars: List[str] = []
    for index, item in enumerate(strategy_summaries):
        y = 32 + index * row_height
        width_value = 420 * (float(item["correction_metrics"]["mean_abs_offset"]) / max_value)
        bars.append(
            f"<text x=\"24\" y=\"{y + 16}\" font-size=\"13\" fill=\"#0f172a\">{html.escape(str(item['strategy_label']))}</text>"
            f"<rect x=\"190\" y=\"{y}\" width=\"420\" height=\"16\" rx=\"8\" fill=\"#e2e8f0\"/>"
            f"<rect x=\"190\" y=\"{y}\" width=\"{max(width_value, 4):.1f}\" height=\"16\" rx=\"8\" fill=\"#2563eb\"/>"
            f"<text x=\"624\" y=\"{y + 13}\" font-size=\"12\" fill=\"#475569\">mean {float(item['correction_metrics']['mean_abs_offset']):.2f} / max {float(item['correction_metrics']['max_abs_offset']):.2f}</text>"
        )
    return f"<svg viewBox=\"0 0 {width} {height}\" role=\"img\" aria-label=\"Strategy correction spread chart\">{''.join(bars)}</svg>"


def _build_lightweight_synopsis(
    *,
    exposure_summary: Dict[str, object],
    strategy_summaries: List[Dict[str, object]],
    recommended_strategy: Dict[str, object],
    hero_summary: Dict[str, object],
) -> str:
    spread = float(exposure_summary["spread"])
    minimum = float(exposure_summary["minimum"])
    maximum = float(exposure_summary["maximum"])
    synopsis = [
        f"Cameras were measured across a {spread:.2f} stop exposure span, from {minimum:.2f} to {maximum:.2f} log2 stops.",
    ]
    if strategy_summaries:
        median_summary = next((item for item in strategy_summaries if item["strategy_key"] == "median"), None)
        brightest_summary = next((item for item in strategy_summaries if item["strategy_key"] == "brightest_valid"), None)
        if median_summary:
            synopsis.append(
                f"The median strategy would center the array with an average absolute correction of {float(median_summary['correction_metrics']['mean_abs_offset']):.2f} stops."
            )
        if brightest_summary:
            synopsis.append(
                f"Brightest-valid would anchor to {brightest_summary.get('reference_clip_id') or 'the brightest valid clip'}, increasing the maximum correction to {float(brightest_summary['correction_metrics']['max_abs_offset']):.2f} stops."
            )
    synopsis.append(recommended_strategy["reason"])
    if hero_summary.get("candidate_clip_id"):
        synopsis.append(f"{hero_summary['candidate_clip_id']} appears to be the strongest hero candidate. {hero_summary['reason']}")
    else:
        synopsis.append(hero_summary["reason"])
    return " ".join(synopsis)


def _render_lightweight_analysis_html(payload: Dict[str, object]) -> str:
    exposure_summary = payload["exposure_summary"]
    hero_summary = payload["hero_recommendation"]
    strategy_cards = []
    for strategy in payload["strategy_comparison"]:
        recommended_badge = "<span class='recommended-badge'>Recommended</span>" if strategy.get("recommended") else ""
        strategy_cards.append(
            "<article class='strategy-card'>"
            f"<div class='strategy-card-top'><h3>{html.escape(str(strategy['strategy_label']))}</h3>{recommended_badge}</div>"
            f"<p>{html.escape(str(strategy['summary']))}</p>"
            f"<dl class='metric-pairs'><div><dt>Target</dt><dd>{float(strategy['target_log2_luminance']):.2f} log2</dd></div>"
            f"<div><dt>Mean | Max</dt><dd>{float(strategy['correction_metrics']['mean_abs_offset']):.2f} | {float(strategy['correction_metrics']['max_abs_offset']):.2f} stops</dd></div>"
            f"<div><dt>Neutral Residual</dt><dd>{float(strategy['correction_metrics'].get('mean_color_residual', 0.0)):.4f}</dd></div>"
            f"<div><dt>WB Model</dt><dd>{html.escape(str((strategy.get('white_balance_model') or {}).get('model_label') or 'n/a'))}</dd></div>"
            f"<div><dt>Shared Kelvin</dt><dd>{html.escape(str((strategy.get('white_balance_model') or {}).get('shared_kelvin') or 'per-camera'))}</dd></div>"
            f"<div><dt>Reference</dt><dd>{html.escape(str(strategy.get('reference_clip_id') or 'Derived'))}</dd></div>"
            "</dl>"
            "</article>"
        )
    table_rows = []
    for row in payload["per_camera_analysis"]:
        outlier_badge = "<span class='outlier-pill'>Outlier</span>" if row.get("is_outlier") else ""
        hero_badge = "<span class='hero-pill'>Hero</span>" if row.get("is_hero_camera") else ""
        table_rows.append(
            "<tr>"
            f"<td><div class='camera-cell'><strong>{html.escape(str(row['camera_label']))}</strong><span>{html.escape(str(row['clip_id']))}</span></div></td>"
            f"<td>{float(row['measured_log2_luminance']):.2f}</td>"
            f"<td>{float(row['raw_offset_stops']):.2f}</td>"
            f"<td>{float(row['final_offset_stops']):.2f}</td>"
            f"<td>{int(row['commit_values'].get('kelvin', 5600))} / {float(row['commit_values'].get('tint', 0.0)):.1f}</td>"
            f"<td>{float(row['pre_color_residual']):.4f}</td>"
            f"<td>{float(row['confidence']):.2f}<div class='subtle'>dL {float(row['neutral_sample_log2_spread']):.3f} | dC {float(row['neutral_sample_chromaticity_spread']):.4f}</div></td>"
            f"<td>{outlier_badge}{hero_badge}</td>"
            f"<td>{html.escape(str(row['note']))}</td>"
            "</tr>"
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
  <title>R3DMatch Lightweight Analysis</title>
  <style>
    body {{ margin: 0; background: #edf2f7; color: #0f172a; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    .page {{ max-width: 1180px; margin: 0 auto; padding: 28px; }}
    .hero {{ background: linear-gradient(135deg, rgba(255,255,255,0.98) 0%, rgba(241,245,249,0.98) 100%); border: 1px solid #dbe4ef; border-radius: 24px; padding: 28px; box-shadow: 0 24px 60px rgba(15,23,42,0.08); }}
    .hero-top {{ display: flex; align-items: flex-start; gap: 18px; }}
    .hero-top img {{ width: 150px; max-width: 30vw; object-fit: contain; }}
    h1 {{ font-size: 32px; line-height: 1.1; margin: 0; }}
    .eyebrow {{ font-size: 12px; letter-spacing: 0.12em; text-transform: uppercase; color: #475569; font-weight: 700; }}
    .synopsis {{ margin-top: 18px; font-size: 17px; line-height: 1.7; color: #1e293b; max-width: 82ch; }}
    .meta-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; margin-top: 20px; }}
    .meta-card, .section {{ background: rgba(255,255,255,0.96); border: 1px solid #d9e1ec; border-radius: 20px; box-shadow: 0 16px 36px rgba(15,23,42,0.05); }}
    .meta-card {{ padding: 14px 16px; }}
    .meta-card dt {{ font-size: 11px; text-transform: uppercase; letter-spacing: 0.08em; color: #64748b; font-weight: 700; }}
    .meta-card dd {{ margin: 8px 0 0 0; font-size: 16px; font-weight: 700; }}
    .grid {{ display: grid; gap: 18px; margin-top: 18px; }}
    .two-up {{ grid-template-columns: 1.3fr 1fr; }}
    .section {{ padding: 22px; }}
    .section h2 {{ margin: 0 0 10px 0; font-size: 22px; }}
    .section p.lead {{ margin: 0 0 12px 0; color: #475569; line-height: 1.6; }}
    .stats {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; }}
    .stat {{ padding: 14px; border-radius: 16px; background: #f8fafc; border: 1px solid #e2e8f0; }}
    .stat .label {{ font-size: 11px; text-transform: uppercase; letter-spacing: 0.08em; color: #64748b; font-weight: 700; }}
    .stat .value {{ margin-top: 8px; font-size: 24px; font-weight: 800; }}
    .strategy-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 14px; }}
    .strategy-card {{ padding: 16px; border-radius: 18px; background: #f8fafc; border: 1px solid #dbe4ef; }}
    .strategy-card-top {{ display: flex; justify-content: space-between; align-items: center; gap: 12px; }}
    .strategy-card h3 {{ margin: 0; font-size: 18px; }}
    .recommended-badge, .hero-pill, .outlier-pill {{ display: inline-flex; align-items: center; border-radius: 999px; padding: 4px 8px; font-size: 11px; font-weight: 800; letter-spacing: 0.06em; text-transform: uppercase; }}
    .recommended-badge {{ background: #dbeafe; color: #1d4ed8; }}
    .hero-pill {{ background: #dcfce7; color: #166534; margin-left: 6px; }}
    .outlier-pill {{ background: #fee2e2; color: #991b1b; margin-right: 6px; }}
    .metric-pairs {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; margin-top: 12px; }}
    .metric-pairs dt {{ font-size: 11px; text-transform: uppercase; color: #64748b; font-weight: 700; }}
    .metric-pairs dd {{ margin: 6px 0 0 0; font-size: 14px; font-weight: 700; }}
    .chart-frame {{ padding: 12px; border-radius: 18px; background: #f8fafc; border: 1px solid #dbe4ef; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 14px; }}
    th, td {{ padding: 12px 10px; border-bottom: 1px solid #e2e8f0; vertical-align: top; font-size: 14px; }}
    th {{ text-align: left; font-size: 12px; letter-spacing: 0.08em; text-transform: uppercase; color: #64748b; }}
    .camera-cell {{ display: flex; flex-direction: column; gap: 4px; }}
    .camera-cell span {{ color: #64748b; font-size: 12px; }}
    .recommendation {{ font-size: 17px; line-height: 1.7; }}
    .subtle {{ color: #64748b; font-size: 13px; line-height: 1.6; }}
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
      <p class="synopsis">{html.escape(str(payload['executive_synopsis']))}</p>
      <dl class="meta-grid">
          <div class="meta-card"><dt>Created</dt><dd>{html.escape(str(payload['created_at']))}</dd></div>
          <div class="meta-card"><dt>Source Mode</dt><dd>{html.escape(str(payload['source_mode_label']))}</dd></div>
          <div class="meta-card"><dt>Target Type</dt><dd>{html.escape(str(payload['target_type']))}</dd></div>
          <div class="meta-card"><dt>Matching Domain</dt><dd>{html.escape(str(payload['matching_domain_label']))}</dd></div>
        <div class="meta-card"><dt>Strategies</dt><dd>{html.escape(str(payload['selected_strategy_labels']))}</dd></div>
        <div class="meta-card"><dt>Subset</dt><dd>{html.escape(str(payload['subset_label']))}</dd></div>
        <div class="meta-card"><dt>Recommendation</dt><dd>{html.escape(str(payload['recommended_strategy']['strategy_label']))}</dd></div>
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
        <div class="chart-frame" style="margin-top:16px;">{payload['visuals']['exposure_plot_svg']}</div>
      </section>
      <section class="section">
        <h2>Recommendation</h2>
        <p class="recommendation"><strong>{html.escape(str(payload['recommended_strategy']['strategy_label']))}</strong> is the preferred strategy for this run.</p>
        <p class="lead">{html.escape(str(payload['recommended_strategy']['reason']))}</p>
        <div class="chart-frame">{payload['visuals']['strategy_chart_svg']}</div>
        <p class="subtle" style="margin-top:12px;"><strong>Hero recommendation:</strong> {html.escape(str(hero_summary['candidate_clip_id'] or 'No clear hero'))}. {html.escape(str(hero_summary['reason']))}</p>
        <p class="subtle"><strong>Next step:</strong> {html.escape(str(payload['operator_recommendation']))}</p>
      </section>
    </div>

    <section class="section" style="margin-top:18px;">
      <h2>Strategy Comparison</h2>
      <p class="lead">Strategies are ranked by how much correction they ask the operator to apply across the set.</p>
      <div class="strategy-grid">{''.join(strategy_cards)}</div>
    </section>

    <section class="section" style="margin-top:18px;">
      <h2>Per-Camera Analysis</h2>
      <p class="lead">Rows are anchored to the recommended strategy so the operator can see practical offsets at a glance.</p>
      <table>
        <thead>
          <tr>
            <th>Camera / Clip</th>
            <th>Measured</th>
            <th>Raw Offset</th>
            <th>Recommended Offset</th>
            <th>Commit (K / Tint)</th>
            <th>Neutral Residual</th>
            <th>Confidence / Spread</th>
            <th>Flags</th>
            <th>Note</th>
          </tr>
        </thead>
        <tbody>{''.join(table_rows)}</tbody>
      </table>
    </section>
  </div>
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
    )
    array_calibration_payload = _load_array_calibration_payload(str(root))
    quality_by_clip: Dict[str, Dict[str, object]] = {}
    if array_calibration_payload:
        for camera in array_calibration_payload.get("cameras", []):
            clip_id = str(camera.get("clip_id") or "").strip()
            if not clip_id:
                continue
            quality_by_clip[clip_id] = dict(camera.get("quality", {}) or {})
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
    per_camera_rows = []
    for record in analysis_records:
        clip_id = str(record["clip_id"])
        strategy_clip = next(item for item in recommended_payload["clips"] if item["clip_id"] == clip_id)
        measured_log2 = float(strategy_clip["measured_log2_luminance"])
        is_outlier = abs(measured_log2 - float(exposure_summary["median"])) > outlier_threshold
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
                "white_balance_model": payload.get("white_balance_model"),
                "correction_metrics": metrics,
                "recommended": payload["strategy_key"] == recommended_strategy["strategy_key"],
            }
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
    operator_recommendation = (
        "Proceed with the recommended strategy and export approval if the customer only needs exposure consistency confirmation. "
        "Generate the full contact-sheet mode only when visual per-camera still comparison is needed."
        if float(exposure_summary["outlier_count"]) <= 1
        else "Inspect the identified outliers before approval. A full contact-sheet review is warranted for visual confirmation."
    )
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
        "hero_recommendation": hero_summary,
        "operator_recommendation": operator_recommendation,
        "per_camera_analysis": per_camera_rows,
        "measurement_render_count": measurement_preview_rendered,
        "shared_originals": [],
        "strategies": [],
        "visuals": {
            "exposure_plot_svg": _build_exposure_plot_svg(recommended_payload["clips"]),
            "strategy_chart_svg": _build_strategy_chart_svg(strategy_summaries),
        },
    }
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
                    abs(float(variant_settings["exposure"])) <= 1e-9
                    and _is_identity_rgb_gains(variant_settings["gains"])
                    and _is_identity_cdl_payload(color_cdl)
                )
                requires_change = abs(float(variant_settings["exposure"])) > 1e-9 or (
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

    title_font = load_font(30, bold=True)
    section_font = load_font(20, bold=True)
    clip_font = load_font(15, bold=True)
    body_font = load_font(13, bold=False)
    caption_font = load_font(12, bold=False)
    pages = []
    page_width = 1700
    page_height = 2200
    margin_x = 60
    margin_y = 60
    logo_max_width = 150
    logo_max_height = 84
    tile_gap_x = 20
    tile_gap_y = 36
    metadata_line_height = 18
    section_gap = 30

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
        metadata_block_height = 88
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
                draw.text((tile_x, meta_y + metadata_line_height), f"Exposure: {tile.get('exposure_offset_stops', 'n/a')}", fill="black", font=body_font)
                draw.text((tile_x, meta_y + metadata_line_height * 2), f"CDL gain/sat: {tile.get('rgb_gains_text', 'n/a')}", fill="black", font=body_font)
                draw.text((tile_x, meta_y + metadata_line_height * 3), f"Confidence: {tile.get('confidence', 'n/a')} | log2: {tile.get('raw_log2', 'n/a')}", fill="black", font=caption_font)
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
            tiles.append(
                {
                    "clip_id": f"{clip['clip_id']} [HERO]" if clip.get("is_hero_camera") else clip["clip_id"],
                    "image_path": clip.get("both_corrected"),
                    "exposure_offset_stops": clip["metrics"]["exposure"]["final_offset_stops"],
                    "rgb_gains_text": gains_text,
                    "confidence": clip["metrics"]["confidence"],
                    "raw_log2": clip["metrics"]["exposure"]["measured_log2_luminance_raw"],
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
                f"<div class='clip-meta'>Commit: exp {float(commit_values.get('exposureAdjust', 0.0) or 0.0):.3f} | K {int(commit_values.get('kelvin', 5600) or 5600)} | tint {float(commit_values.get('tint', 0.0) or 0.0):.1f}</div>"
                f"<div class='clip-meta'>Confidence: {float(metrics.get('confidence', 0.0) or 0.0):.2f} | sample spread {float(metrics['exposure'].get('neutral_sample_log2_spread', 0.0) or 0.0):.3f}</div>"
                f"<div class='clip-meta subtle'>CDL gain/sat: {html.escape(_format_cdl_gain_saturation(metrics['color']))}</div>"
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
    .page {{ max-width: 1440px; margin: 0 auto; padding: 28px; }}
    .shell {{ display: grid; gap: 20px; }}
    .hero, .panel, .report-section {{ background: rgba(255,255,255,0.98); border: 1px solid #d8dee8; border-radius: 24px; box-shadow: 0 18px 44px rgba(15,23,42,0.06); }}
    .hero {{ padding: 28px; }}
    .hero-top {{ display: flex; align-items: flex-start; gap: 20px; }}
    .brand-logo {{ width: 150px; height: auto; object-fit: contain; }}
    .brand-logo-text {{ font-size:30px; font-weight: 800; }}
    .eyebrow {{ font-size: 12px; font-weight: 800; letter-spacing: 0.12em; text-transform: uppercase; color: #64748b; }}
    h1 {{ margin: 6px 0 10px 0; font-size: 38px; line-height: 1.05; }}
    .hero-subtitle {{ margin: 0; font-size: 18px; color: #334155; }}
    .synopsis {{ margin-top: 18px; font-size: 18px; line-height: 1.8; color: #1e293b; max-width: 86ch; }}
    .meta-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin-top: 22px; }}
    .meta-card {{ background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 18px; padding: 16px; }}
    .meta-card dt {{ font-size: 11px; font-weight: 800; letter-spacing: 0.08em; text-transform: uppercase; color: #64748b; }}
    .meta-card dd {{ margin: 8px 0 0 0; font-size: 17px; font-weight: 800; }}
    .two-up {{ display: grid; grid-template-columns: 1.25fr 1fr; gap: 20px; }}
    .panel {{ padding: 22px; }}
    .panel h2, .report-section h2 {{ margin: 0 0 10px 0; font-size: 26px; line-height: 1.1; }}
    .panel p.lead, .section-meta {{ margin: 0 0 14px 0; color: #475569; font-size: 16px; line-height: 1.7; }}
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
    .chart-frame {{ border-radius: 18px; background: #f8fafc; border: 1px solid #dbe3ee; padding: 12px; margin-top: 12px; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 14px; }}
    th, td {{ padding: 14px 10px; border-bottom: 1px solid #e5e7eb; text-align: left; vertical-align: top; font-size: 15px; }}
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
    .clip-card {{ background: white; border-radius: 18px; padding: 14px; border: 1px solid #e2e8f0; }}
    .clip-card.hero {{ border: 2px solid #b91c1c; }}
    .hero-badge {{ display: inline-flex; align-items: center; padding: 5px 9px; border-radius: 999px; background: #b91c1c; color: white; font-size: 11px; font-weight: 800; letter-spacing: 0.06em; text-transform: uppercase; margin-bottom: 10px; }}
    .thumb-single figure, figure {{ margin: 0; }}
    .thumb-single img, figure img {{ width: 100%; display: block; border-radius: 12px; background: #d7dce4; aspect-ratio: 16 / 10; object-fit: cover; }}
    figcaption {{ margin-top: 8px; font-size: 13px; color: #64748b; }}
    .clip-label {{ font-size: 18px; font-weight: 800; margin-top: 12px; }}
    .clip-meta {{ font-size: 14px; line-height: 1.55; margin-top: 6px; }}
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
        <div class="section-top"><h2>Visual Confirmation Stills</h2><p class="section-meta">Original stills remain in the report as the confirmation layer, not the only decision input.</p></div>
        <div class="grid cols-{original_columns}">{''.join(original_cards)}</div>
      </section>

      {''.join(strategy_sections)}
    </div>
  </div>
</body>
</html>"""
