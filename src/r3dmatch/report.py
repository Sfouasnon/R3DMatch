from __future__ import annotations

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
from .color import rgb_gains_to_cdl
from .rmd import write_rmd_for_clip, write_rmds_from_analysis


PREVIEW_VARIANTS = ("original", "exposure", "color", "both")
REVIEW_PREVIEW_TRANSFORM = "REDLine IPP2 Log3G10 / Medium / Medium"
DEFAULT_REVIEW_TARGET_STRATEGIES = ("median",)
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
        "median": "median",
        "manual": "manual",
    }
    if normalized not in aliases:
        raise ValueError("target strategy must be one of: median, brightest-valid, manual")
    return aliases[normalized]


def strategy_display_name(name: str) -> str:
    return {
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
    if report_root.exists():
        for name in ("contact_sheet.json", "contact_sheet.html", "preview_contact_sheet.pdf", "review_manifest.json"):
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
    help_result = subprocess.run([redline_executable, "--help"], capture_output=True, text=True, check=False)
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
    return {
        "measured_log2_luminance_monitoring": measured_log2,
        "measured_rgb_mean_monitoring": [float(rgb_mean[0]), float(rgb_mean[1]), float(rgb_mean[2])],
        "measured_rgb_chromaticity_monitoring": [float(chroma[0]), float(chroma[1]), float(chroma[2])],
        "valid_pixel_count_monitoring": int(pixels.shape[0]),
    }


def _build_strategy_payloads(
    analysis_records: List[Dict[str, object]],
    *,
    target_strategies: List[str],
    reference_clip_id: Optional[str],
    monitoring_measurements_by_clip: Optional[Dict[str, Dict[str, object]]] = None,
) -> List[Dict[str, object]]:
    if not analysis_records:
        return []
    measured_log2 = np.array(
        [
            float(
                (monitoring_measurements_by_clip or {}).get(str(record["clip_id"]), {}).get(
                    "measured_log2_luminance_monitoring",
                    record.get("diagnostics", {}).get("measured_log2_luminance_monitoring", record.get("diagnostics", {}).get("measured_log2_luminance", 0.0)),
                )
            )
            for record in analysis_records
        ],
        dtype=np.float32,
    )
    measured_chroma = np.array([record.get("diagnostics", {}).get("measured_rgb_chromaticity", [1 / 3, 1 / 3, 1 / 3]) for record in analysis_records], dtype=np.float32)
    payloads: List[Dict[str, object]] = []
    for requested in target_strategies:
        strategy = normalize_target_strategy_name(requested)
        if strategy == "median":
            target_log2 = float(np.median(measured_log2))
            target_rgb = [float(np.median(measured_chroma[:, index])) for index in range(3)]
            resolved_reference = None
        elif strategy == "brightest_valid":
            target_index = int(np.argmax(measured_log2))
            target_log2 = float(measured_log2[target_index])
            target_rgb = [float(value) for value in measured_chroma[target_index]]
            resolved_reference = str(analysis_records[target_index]["clip_id"])
        else:
            if not reference_clip_id:
                raise ValueError("manual target strategy requires --reference-clip-id")
            matches = [record for record in analysis_records if str(record["clip_id"]) == reference_clip_id]
            if not matches:
                raise ValueError(f"manual reference clip not found in analysis records: {reference_clip_id}")
            reference_record = matches[0]
            target_log2 = float(
                (monitoring_measurements_by_clip or {}).get(str(reference_record["clip_id"]), {}).get(
                    "measured_log2_luminance_monitoring",
                    reference_record.get("diagnostics", {}).get("measured_log2_luminance_monitoring", reference_record.get("diagnostics", {}).get("measured_log2_luminance", 0.0)),
                )
            )
            target_rgb = [float(value) for value in reference_record.get("diagnostics", {}).get("measured_rgb_chromaticity", [1 / 3, 1 / 3, 1 / 3])]
            resolved_reference = reference_clip_id

        strategy_clips = []
        for record in analysis_records:
            measured = record.get("diagnostics", {})
            monitoring_measured = (monitoring_measurements_by_clip or {}).get(str(record["clip_id"]), {})
            measured_rgb = [float(value) for value in measured.get("measured_rgb_chromaticity", [1 / 3, 1 / 3, 1 / 3])]
            measured_monitoring_log2 = float(
                monitoring_measured.get(
                    "measured_log2_luminance_monitoring",
                    measured.get("measured_log2_luminance_monitoring", measured.get("measured_log2_luminance", 0.0)),
                )
            )
            raw_gains = [
                target_rgb[0] / max(measured_rgb[0], 1e-6),
                target_rgb[1] / max(measured_rgb[1], 1e-6),
                target_rgb[2] / max(measured_rgb[2], 1e-6),
            ]
            gain_norm = (raw_gains[0] * raw_gains[1] * raw_gains[2]) ** (1.0 / 3.0)
            rgb_gains = [float(value / max(gain_norm, 1e-6)) for value in raw_gains]
            color_cdl = rgb_gains_to_cdl(rgb_gains)
            strategy_clips.append(
                {
                    "clip_id": str(record["clip_id"]),
                    "group_key": str(record["group_key"]),
                    "source_path": record.get("source_path"),
                    "measured_log2_luminance": measured_monitoring_log2,
                    "measured_log2_luminance_monitoring": measured_monitoring_log2,
                    "measured_log2_luminance_raw": float(measured.get("measured_log2_luminance_raw", measured.get("measured_log2_luminance", 0.0))),
                    "measured_rgb_chromaticity": [float(value) for value in measured_rgb],
                    "monitoring_measurement_source": "rendered_preview" if monitoring_measured else "analysis_diagnostic",
                    "exposure_offset_stops": float(target_log2 - measured_monitoring_log2),
                    "rgb_gains": rgb_gains,
                    "color_cdl": color_cdl,
                    "confidence": float(record.get("confidence", 0.0)),
                    "flags": list(record.get("flags", [])),
                    "calibration_roi": measured.get("calibration_roi"),
                    "measurement_mode": measured.get("calibration_measurement_mode"),
                    "exposure_measurement_domain": measured.get("exposure_measurement_domain", "monitoring"),
                }
            )

        payloads.append(
            {
                "strategy_key": strategy,
                "strategy_label": strategy_display_name(strategy),
                "reference_clip_id": resolved_reference,
                "target_log2_luminance": target_log2,
                "target_rgb_chromaticity": target_rgb,
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
        command.extend(["--loadRMD", str(Path(rmd_path).expanduser().resolve()), "--useRMD", "2"])
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
) -> List[str]:
    command = [
        redline_executable,
        "--i",
        str(Path(source_path).expanduser().resolve()),
        "--useMeta",
        "--printMeta",
        "1",
        "--noRender",
    ]
    if rmd_path and redline_capabilities.get("supports_load_rmd"):
        command.extend(["--loadRMD", str(Path(rmd_path).expanduser().resolve()), "--useRMD", "2"])
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
) -> Dict[str, object]:
    command = _build_redline_metadata_probe_command(
        source_path,
        redline_executable=redline_executable,
        rmd_path=rmd_path,
        exposure_stops=exposure_stops,
        color_cdl=color_cdl,
        color_method=color_method,
        redline_capabilities=redline_capabilities,
    )
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
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
) -> Dict[str, object]:
    color_cdl = None
    color_method = None
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
        color_method = "rgb_gain"
    command = _build_redline_preview_command(
        input_r3d,
        output_path=output_path,
        frame_index=frame_index,
        exposure_stops=exposure,
        color_cdl=color_cdl,
        color_method=color_method,
        redline_executable=redline_executable,
        preview_settings=preview_settings,
        redline_capabilities=redline_capabilities,
        use_as_shot_metadata=use_as_shot_metadata,
        rmd_path=None,
    )
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    actual_output = _resolve_rendered_output_path(output_path)
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "output_path": str(actual_output),
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
    preview_root = Path(previews_dir).expanduser().resolve()
    preview_root.mkdir(parents=True, exist_ok=True)
    sidecars = _load_sidecar_map(input_path)
    redline_executable = _resolve_redline_executable()
    preview_paths: Dict[str, Dict[str, object]] = {}
    command_records: List[Dict[str, object]] = []
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
                preview_path = preview_root / preview_filename_for_clip_id(clip_id, variant, strategy=strategy_key, run_id=run_id)
                look_metadata_path = None
                probe_result = None
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
                            },
                        },
                    }
                    review_rmd_dir = Path(strategy_rmd_root).expanduser().resolve() / strategy_key / variant
                    look_metadata_path = str(write_rmd_for_clip(clip_id, review_sidecar, review_rmd_dir))
                    probe_result = _probe_redline_application(
                        source_path,
                        redline_executable=redline_executable,
                        rmd_path=look_metadata_path,
                        exposure_stops=float(variant_settings["exposure"]),
                        color_cdl=clip_entry.get("color_cdl"),
                        color_method="rgb_gain" if variant_settings["gains"] is not None else None,
                        redline_capabilities=redline_capabilities,
                    )

                red_gain = green_gain = blue_gain = None
                if variant_settings["gains"] is not None:
                    red_gain = float(variant_settings["gains"][0])
                    green_gain = float(variant_settings["gains"][1])
                    blue_gain = float(variant_settings["gains"][2])
                corrected_render = render_preview_frame(
                    source_path,
                    str(preview_path),
                    frame_index=frame_index,
                    redline_executable=redline_executable,
                    redline_capabilities=redline_capabilities,
                    preview_settings=render_settings,
                    use_as_shot_metadata=False,
                    exposure=float(variant_settings["exposure"]),
                    red_gain=red_gain,
                    green_gain=green_gain,
                    blue_gain=blue_gain,
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
                requires_change = abs(float(variant_settings["exposure"])) > 0.05 or (
                    variant_settings["gains"] is not None
                    and any(abs(float(value) - 1.0) > 1e-6 for value in variant_settings["gains"])
                )
                error_message = None
                if requires_change and (mean_diff is None or float(mean_diff) < 1e-3):
                    error_message = "REDLine correction not applied"
                    print(
                        f"[r3dmatch] ERROR: {error_message} clip={clip_id} strategy={strategy_key} "
                        f"variant={variant} command={shlex.join(corrected_render['command'])}"
                    )
                print(
                    f"[r3dmatch] preview render clip={clip_id} mode=corrected strategy={strategy_key} "
                    f"variant={variant} exposure={float(variant_settings['exposure']):.6f} "
                    f"redGain={red_gain} greenGain={green_gain} blueGain={blue_gain} pixel_diff={mean_diff}"
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
                        "command": corrected_render["command"],
                        "output": str(preview_path),
                        "look_metadata_path": look_metadata_path,
                        "rmd_metadata_probe": probe_result,
                        "returncode": corrected_render["returncode"],
                        "stdout": corrected_render["stdout"],
                        "stderr": corrected_render["stderr"],
                        "pixel_diff_from_baseline": mean_diff,
                        "pixel_output_changed": diff_metrics["pixel_output_changed"],
                        "error": error_message,
                        "as_shot_metadata_used": False,
                        "explicit_transform_used": True,
                        "explicit_correction_flags_used": bool(
                            abs(float(variant_settings["exposure"])) > 1e-9
                            or (red_gain is not None and abs(float(red_gain) - 1.0) > 1e-9)
                            or (green_gain is not None and abs(float(green_gain) - 1.0) > 1e-9)
                            or (blue_gain is not None and abs(float(blue_gain) - 1.0) > 1e-9)
                        ),
                    }
                )
                preview_paths[clip_id]["strategies"][strategy_key][variant] = str(preview_path)
    (preview_root / "preview_commands.json").write_text(json.dumps({"commands": command_records}, indent=2), encoding="utf-8")
    return preview_paths


def build_contact_sheet_report(
    input_path: str,
    *,
    out_dir: str,
    exposure_calibration_path: Optional[str] = None,
    color_calibration_path: Optional[str] = None,
    target_type: Optional[str] = None,
    processing_mode: Optional[str] = None,
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
) -> Dict[str, object]:
    root = Path(input_path).expanduser().resolve()
    out_root = Path(out_dir).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    if clear_cache:
        clear_preview_cache(str(root), report_dir=str(out_root))
    analysis_records = _load_analysis_records(input_path)
    resolved_strategies = [normalize_target_strategy_name(item) for item in (target_strategies or list(DEFAULT_REVIEW_TARGET_STRATEGIES))]
    run_id = root.name or "review"
    redline_executable = _resolve_redline_executable()
    redline_capabilities = _detect_redline_capabilities(redline_executable)
    measurement_preview_settings = _normalize_preview_settings(
        preview_mode="calibration",
        preview_output_space=None,
        preview_output_gamma=None,
        preview_highlight_rolloff=None,
        preview_shadow_rolloff=None,
        preview_lut=None,
    )
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
    measurement_preview_paths = generate_preview_stills(
        input_path,
        analysis_records=analysis_records,
        previews_dir=str(root / "previews" / "_measurement"),
        preview_settings=measurement_preview_settings,
        redline_capabilities=redline_capabilities,
        strategy_payloads=[],
        run_id=run_id,
    )
    monitoring_measurements_by_clip = {}
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
        monitoring_measurements_by_clip=monitoring_measurements_by_clip,
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
        clip_id = str(record["clip_id"])
        shared_originals.append(
            {
                "clip_id": clip_id,
                "group_key": str(record["group_key"]),
                "source_path": record.get("source_path"),
                "original_frame": preview_paths.get(clip_id, {}).get("original"),
                "measured_log2_luminance": monitoring_measurements_by_clip.get(clip_id, {}).get("measured_log2_luminance_monitoring", record.get("diagnostics", {}).get("measured_log2_luminance")),
                "measured_log2_luminance_monitoring": monitoring_measurements_by_clip.get(clip_id, {}).get("measured_log2_luminance_monitoring", record.get("diagnostics", {}).get("measured_log2_luminance_monitoring")),
                "measured_log2_luminance_raw": record.get("diagnostics", {}).get("measured_log2_luminance_raw"),
                "confidence": record.get("confidence"),
            }
        )

    strategies = []
    for strategy_payload in strategy_payloads:
        strategy_clips = []
        for record in analysis_records:
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
                    "metrics": {
                        "exposure": {
                            "raw_offset_stops": record.get("raw_offset_stops"),
                            "final_offset_stops": strategy_clip["exposure_offset_stops"],
                            "measured_log2_luminance": strategy_clip["measured_log2_luminance"],
                            "measured_log2_luminance_monitoring": strategy_clip["measured_log2_luminance_monitoring"],
                            "measured_log2_luminance_raw": strategy_clip["measured_log2_luminance_raw"],
                            "measurement_domain": strategy_clip["exposure_measurement_domain"],
                        },
                        "color": {
                            "rgb_gains": strategy_clip["rgb_gains"],
                            "measured_channel_medians": color_entry.measured_channel_medians if color_entry else None,
                        },
                        "sampling_mode": (
                            exposure_entry.sampling_mode if exposure_entry
                            else color_entry.sampling_mode if color_entry
                            else None
                        ),
                        "confidence": strategy_clip["confidence"],
                        "flags": strategy_clip["flags"],
                        "measurement_mode": strategy_clip["measurement_mode"],
                        "preview_transform": _preview_transform_label(display_preview_settings),
                    },
                }
            )
        strategies.append(
            {
                "strategy_key": strategy_payload["strategy_key"],
                "strategy_label": strategy_payload["strategy_label"],
                "reference_clip_id": strategy_payload["reference_clip_id"],
                "target_log2_luminance": strategy_payload["target_log2_luminance"],
                "target_rgb_chromaticity": strategy_payload["target_rgb_chromaticity"],
                "calibration_roi": calibration_roi or (analysis_records[0].get("diagnostics", {}).get("calibration_roi") if analysis_records else None),
                "clips": strategy_clips,
            }
        )

    payload = {
        "input_path": str(root),
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
        "preview_transform": _preview_transform_label(display_preview_settings),
        "measurement_preview_transform": _preview_transform_label(measurement_preview_settings),
        "exposure_measurement_domain": "monitoring",
        "preview_mode": display_preview_settings["preview_mode"],
        "preview_settings": display_preview_settings,
        "measurement_preview_settings": measurement_preview_settings,
        "redline_capabilities": redline_capabilities,
        "calibration_roi": calibration_roi or (analysis_records[0].get("diagnostics", {}).get("calibration_roi") if analysis_records else None),
        "run_id": run_id,
        "target_strategies": resolved_strategies,
        "reference_clip_id": reference_clip_id,
        "clip_count": len(analysis_records),
        "shared_originals": shared_originals,
        "strategies": strategies,
        "clips": strategies[0]["clips"] if strategies else [],
        "strategy_review_rmd_root": str((root / "review_rmd" / "strategies").resolve()),
    }
    json_path = out_root / "contact_sheet.json"
    html_path = out_root / "contact_sheet.html"
    pdf_path = out_root / "preview_contact_sheet.pdf"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    html_path.write_text(render_contact_sheet_html(payload), encoding="utf-8")
    render_contact_sheet_pdf(payload, output_path=str(pdf_path), title="R3DMatch Review Contact Sheet")
    review_manifest_path = out_root / "review_manifest.json"
    review_manifest_path.write_text(
        json.dumps(
            {
                "input_path": str(root),
                "target_type": payload["target_type"],
                "processing_mode": payload["processing_mode"],
                "preview_transform": payload["preview_transform"],
                "measurement_preview_transform": payload["measurement_preview_transform"],
                "preview_mode": payload["preview_mode"],
                "preview_settings": payload["preview_settings"],
                "measurement_preview_settings": payload["measurement_preview_settings"],
                "redline_capabilities": payload["redline_capabilities"],
                "exposure_measurement_domain": payload["exposure_measurement_domain"],
                "calibration_roi": payload["calibration_roi"],
                "target_strategies": payload["target_strategies"],
                "reference_clip_id": payload["reference_clip_id"],
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
    return {
        "report_json": str(json_path),
        "report_html": str(html_path),
        "preview_report_pdf": str(pdf_path),
        "previews_dir": str(root / "previews"),
        "review_manifest": str(review_manifest_path),
        "clip_count": len(analysis_records),
        "preview_transform": payload["preview_transform"],
        "measurement_preview_transform": payload["measurement_preview_transform"],
        "preview_mode": payload["preview_mode"],
        "preview_settings": payload["preview_settings"],
        "measurement_preview_settings": payload["measurement_preview_settings"],
        "redline_capabilities": payload["redline_capabilities"],
    }


def build_review_package(
    input_path: str,
    *,
    out_dir: str,
    exposure_calibration_path: Optional[str] = None,
    color_calibration_path: Optional[str] = None,
    target_type: str = "gray_sphere",
    processing_mode: str = "both",
    preview_mode: str = "calibration",
    preview_output_space: Optional[str] = None,
    preview_output_gamma: Optional[str] = None,
    preview_highlight_rolloff: Optional[str] = None,
    preview_shadow_rolloff: Optional[str] = None,
    preview_lut: Optional[str] = None,
    calibration_roi: Optional[Dict[str, float]] = None,
    target_strategies: Optional[List[str]] = None,
    reference_clip_id: Optional[str] = None,
) -> Dict[str, object]:
    root = Path(out_dir).expanduser().resolve()
    report_dir = root / "report"
    review_rmd_dir = root / "review_rmd"
    report_payload = build_contact_sheet_report(
        out_dir,
        out_dir=str(report_dir),
        exposure_calibration_path=exposure_calibration_path,
        color_calibration_path=color_calibration_path,
        target_type=target_type,
        processing_mode=processing_mode,
        preview_mode=preview_mode,
        preview_output_space=preview_output_space,
        preview_output_gamma=preview_output_gamma,
        preview_highlight_rolloff=preview_highlight_rolloff,
        preview_shadow_rolloff=preview_shadow_rolloff,
        preview_lut=preview_lut,
        calibration_roi=calibration_roi,
        target_strategies=target_strategies,
        reference_clip_id=reference_clip_id,
        clear_cache=True,
    )
    rmd_manifest = write_rmds_from_analysis(out_dir, out_dir=str(review_rmd_dir))
    package_manifest = {
        "workflow_phase": "review",
        "analysis_dir": str(root),
        "target_type": target_type,
        "processing_mode": processing_mode,
        "preview_transform": report_payload.get("preview_transform"),
        "measurement_preview_transform": report_payload.get("measurement_preview_transform"),
        "preview_mode": preview_mode,
        "preview_settings": report_payload.get("preview_settings"),
        "measurement_preview_settings": report_payload.get("measurement_preview_settings"),
        "redline_capabilities": report_payload.get("redline_capabilities"),
        "exposure_measurement_domain": "monitoring",
        "calibration_roi": calibration_roi,
        "target_strategies": target_strategies or list(DEFAULT_REVIEW_TARGET_STRATEGIES),
        "reference_clip_id": reference_clip_id,
        "review_rmd_dir": str(review_rmd_dir),
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
    from PIL import Image, ImageDraw, ImageFont

    def load_font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
        preferred = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
        try:
            return ImageFont.truetype(preferred, size=size)
        except OSError:
            return ImageFont.load_default()

    title_font = load_font(14, bold=True)
    section_font = load_font(14, bold=True)
    body_font = load_font(12, bold=False)
    caption_font = load_font(12, bold=False)
    pages = []
    page_width = 1700
    page_height = 2200
    margin_x = 60
    margin_y = 60
    tile_width = 320
    tile_height = 220
    tile_gap_x = 24
    tile_gap_y = 110
    metadata_line_height = 18
    section_gap = 28

    def new_page() -> tuple[Image.Image, ImageDraw.ImageDraw, int]:
        page = Image.new("RGB", (page_width, page_height), "white")
        draw = ImageDraw.Draw(page)
        y = margin_y
        draw.text((margin_x, y), title, fill="black", font=title_font)
        y += 40
        if timestamp_label:
            draw.text((margin_x, y), timestamp_label, fill="black", font=body_font)
            y += 30
        summary_lines = [
            f"Target type: {payload.get('target_type')}",
            f"Processing mode: {payload.get('processing_mode')}",
            f"Preview transform: {payload.get('preview_transform')}",
            f"Exposure measurement domain: {payload.get('exposure_measurement_domain')}",
            f"Calibration ROI: {payload.get('calibration_roi')}",
            f"Selected target strategies: {payload.get('target_strategies')}",
        ]
        for line in summary_lines:
            draw.text((margin_x, y), line, fill="black", font=body_font)
            y += 24
        return page, draw, y + section_gap

    def ensure_room(current_y: int, needed_height: int) -> tuple[Image.Image, ImageDraw.ImageDraw, int]:
        nonlocal page, draw
        if current_y + needed_height <= page_height - margin_y:
            return page, draw, current_y
        pages.append(page)
        page, draw, new_y = new_page()
        return page, draw, new_y

    def fit_image(path: str) -> Image.Image:
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
        _, _, y = ensure_room(y, 120)
        draw.text((margin_x, y), section_title, fill="black", font=section_font)
        y += 28
        for meta_line in section_meta or []:
            draw.text((margin_x, y), meta_line, fill="black", font=body_font)
            y += 20
        if section_meta:
            y += 8

        columns = max(1, (page_width - (2 * margin_x) + tile_gap_x) // (tile_width + tile_gap_x))
        for index, tile in enumerate(tiles):
            row = index // columns
            col = index % columns
            tile_x = margin_x + col * (tile_width + tile_gap_x)
            tile_y = y + row * (tile_height + tile_gap_y)
            if tile_y + tile_height + tile_gap_y > page_height - margin_y:
                pages.append(page)
                page, draw, y = new_page()
                return draw_tiles(section_title=section_title, tiles=tiles[index:], section_meta=section_meta)

            image_path = tile.get("image_path")
            if image_path and Path(str(image_path)).exists():
                page.paste(fit_image(str(image_path)), (tile_x, tile_y))
            draw.text((tile_x, tile_y + tile_height + 8), str(tile.get("clip_id", "")), fill="black", font=caption_font)
            draw.text(
                (tile_x, tile_y + tile_height + 8 + metadata_line_height),
                f"Exposure: {tile.get('exposure_offset_stops', 'n/a')}",
                fill="black",
                font=body_font,
            )
            draw.text(
                (tile_x, tile_y + tile_height + 8 + metadata_line_height * 2),
                f"RGB gains: {tile.get('rgb_gains_text', 'n/a')}",
                fill="black",
                font=body_font,
            )
            draw.text(
                (tile_x, tile_y + tile_height + 8 + metadata_line_height * 3),
                f"Confidence: {tile.get('confidence', 'n/a')} | Raw log2: {tile.get('raw_log2', 'n/a')}",
                fill="black",
                font=body_font,
            )
        rows = (len(tiles) + columns - 1) // columns
        y += rows * (tile_height + tile_gap_y) + section_gap
        return y

    page, draw, y = new_page()

    original_tiles = [
        {
            "clip_id": clip["clip_id"],
            "image_path": clip.get("original_frame"),
            "exposure_offset_stops": clip.get("measured_log2_luminance_monitoring"),
            "rgb_gains_text": "original",
            "confidence": clip.get("confidence"),
            "raw_log2": clip.get("measured_log2_luminance_raw"),
        }
        for clip in payload.get("shared_originals", [])
    ]
    draw_tiles(section_title="Original", tiles=original_tiles)

    strategy_order = ["median", "brightest_valid", "manual"]
    strategies_by_key = {str(item["strategy_key"]): item for item in payload.get("strategies", [])}
    for strategy_key in strategy_order:
        strategy = strategies_by_key.get(strategy_key)
        if not strategy:
            continue
        tiles = []
        for clip in strategy.get("clips", []):
            gains = clip["metrics"]["color"]["rgb_gains"]
            gains_text = ", ".join(f"{float(value):.3f}" for value in gains) if gains else "n/a"
            tiles.append(
                {
                    "clip_id": clip["clip_id"],
                    "image_path": clip.get("both_corrected"),
                    "exposure_offset_stops": clip["metrics"]["exposure"]["final_offset_stops"],
                    "rgb_gains_text": gains_text,
                    "confidence": clip["metrics"]["confidence"],
                    "raw_log2": clip["metrics"]["exposure"]["measured_log2_luminance_raw"],
                }
            )
        section_meta = [
            f"Reference clip: {strategy.get('reference_clip_id')}",
            f"Target log2 luminance: {strategy.get('target_log2_luminance')}",
            f"Exposure solve domain: monitoring | Preview transform: {payload.get('preview_transform')}",
            f"Calibration ROI: {strategy.get('calibration_roi')}",
        ]
        draw_tiles(section_title=strategy["strategy_label"], tiles=tiles, section_meta=section_meta)
    pages.append(page)

    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
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
    original_cards = []
    for clip in payload.get("shared_originals", []):
        path = clip.get("original_frame")
        image_html = (
            f"<figure><img src='../previews/{Path(path).name}' alt='{clip['clip_id']} Original' /><figcaption>Original</figcaption></figure>"
            if path else "<p class='subtle'>No original preview.</p>"
        )
        original_cards.append(
            "<article class='clip-card'>"
            f"<h2>{clip['clip_id']}</h2>"
            f"<p class='subtle'>Group: {clip['group_key']}</p>"
            f"{image_html}"
            "</article>"
        )
    strategy_sections = []
    for strategy in payload.get("strategies", []):
        cards = []
        for clip in strategy["clips"]:
            metrics = clip["metrics"]
            images = []
            for label, path_key in (("Exposure", "exposure_corrected"), ("Color", "color_corrected"), ("Both", "both_corrected")):
                path = clip.get(path_key)
                if path:
                    images.append(
                        f"<figure><img src='../previews/{Path(path).name}' alt='{clip['clip_id']} {label}' /><figcaption>{label}</figcaption></figure>"
                    )
            gains = metrics["color"]["rgb_gains"]
            gains_text = ", ".join(f"{float(value):.3f}" for value in gains) if gains else "n/a"
            flags = ", ".join(metrics.get("flags", [])) if metrics.get("flags") else "none"
            thumbs_html = "".join(images) if images else "<p class='subtle'>No previews generated.</p>"
            cards.append(
                "<article class='clip-card'>"
                f"<h2>{clip['clip_id']}</h2>"
                f"<p class='subtle'>Group: {clip['group_key']}</p>"
                f"<div class='thumb-grid'>{thumbs_html}</div>"
                "<dl>"
                f"<dt>Target type</dt><dd>{payload.get('target_type')}</dd>"
                f"<dt>Preview transform</dt><dd>{payload.get('preview_transform')}</dd>"
                f"<dt>Calibration ROI</dt><dd>{clip.get('calibration_roi') or payload.get('calibration_roi')}</dd>"
                f"<dt>Monitoring log2 luminance</dt><dd>{metrics['exposure']['measured_log2_luminance_monitoring']}</dd>"
                f"<dt>Raw log2 luminance</dt><dd>{metrics['exposure']['measured_log2_luminance_raw']}</dd>"
                f"<dt>Exposure solve domain</dt><dd>{metrics['exposure']['measurement_domain']}</dd>"
                f"<dt>Exposure offset</dt><dd>{metrics['exposure']['final_offset_stops']}</dd>"
                f"<dt>RGB gains</dt><dd>{gains_text}</dd>"
                f"<dt>Confidence</dt><dd>{metrics['confidence']}</dd>"
                f"<dt>Flags</dt><dd>{flags}</dd>"
                f"<dt>Measurement mode</dt><dd>{metrics.get('measurement_mode')}</dd>"
                f"<dt>Source path</dt><dd>{clip['source_path']}</dd>"
                "</dl>"
                "</article>"
            )
        strategy_sections.append(
            f"<section><h2>Target Strategy: {strategy['strategy_label']}</h2>"
            f"<p class='subtle'>Reference clip: {strategy.get('reference_clip_id')} | Target log2: {strategy.get('target_log2_luminance')} | Target chromaticity: {strategy.get('target_rgb_chromaticity')}</p>"
            f"<div class='grid'>{''.join(cards)}</div></section>"
        )
    return (
        "<!doctype html><html><head><meta charset='utf-8'><title>R3DMatch Contact Sheet</title>"
        "<style>"
        "body{font-family:-apple-system,BlinkMacSystemFont,sans-serif;background:#f5f1e8;color:#181513;margin:0;padding:24px;}"
        "header{margin-bottom:24px;padding:20px;background:white;border-radius:16px;box-shadow:0 10px 30px rgba(0,0,0,.06);}"
        ".grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:18px;}"
        ".clip-card{background:white;border-radius:16px;padding:16px;box-shadow:0 10px 30px rgba(0,0,0,.06);}"
        ".thumb-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px;margin:12px 0 16px;}"
        "figure{margin:0;}img{width:100%;display:block;border-radius:10px;background:#d7d2c7;}figcaption{font-size:12px;color:#5b554e;margin-top:4px;}"
        "dl{display:grid;grid-template-columns:max-content 1fr;gap:6px 12px;margin:0;}dt{font-weight:600;}dd{margin:0;}"
        ".subtle{color:#5b554e;}"
        "</style></head>"
        "<body><header><h1>R3DMatch Contact Sheet</h1>"
        f"<p>Clip count: {payload['clip_count']}</p>"
        f"<p>Target type: {payload.get('target_type')}</p>"
        f"<p>Processing mode: {payload.get('processing_mode')}</p>"
        f"<p>Preview transform: {payload.get('preview_transform')}</p>"
        f"<p>Exposure measurement domain: {payload.get('exposure_measurement_domain')}</p>"
        f"<p>Calibration ROI: {payload.get('calibration_roi')}</p>"
        f"<p>Target strategies: {payload.get('target_strategies')}</p>"
        "</header>"
        f"<section><h2>Shared Originals</h2><div class='grid'>{''.join(original_cards)}</div></section>"
        f"<main>{''.join(strategy_sections)}</main>"
        "</body></html>"
    )
