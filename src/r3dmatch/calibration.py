from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .color import chromaticity_variance, compute_chromaticity_medians, solve_neutral_gains
from .commit_values import solve_white_balance_model_for_records
from .models import (
    CalibrationLike,
    CenterCrop,
    ColorCalibration,
    ColorCalibrationEntry,
    ExposureCalibration,
    ExposureCalibrationEntry,
    GrayCardCalibration,
    GrayCardCalibrationCameraEntry,
    GrayCardDetection,
    GrayCardDetectionEntry,
    GrayCardROI,
    IndependentCalibrationLike,
    SamplingRegion,
    SphereCalibration,
    SphereCalibrationCameraEntry,
    SphereROI,
)
from .sdk import resolve_backend


DEFAULT_CENTER_CROP_WIDTH_RATIO = 0.4
DEFAULT_CENTER_CROP_HEIGHT_RATIO = 0.4
LOW_CONFIDENCE_THRESHOLD = 0.35


@dataclass
class ExposureTarget:
    log2_luminance_target: float
    estimator: str
    included_camera_count: int
    excluded_camera_ids: List[str]


@dataclass
class ColorTarget:
    target_rgb_chromaticity: List[float]
    estimator: str
    included_camera_count: int
    excluded_camera_ids: List[str]


@dataclass
class ArrayTarget:
    method: str
    exposure: ExposureTarget
    color: ColorTarget


@dataclass
class CameraMeasurement:
    gray_sample_count: int
    valid_pixel_count: int
    measured_log2_luminance: float
    measured_rgb_mean: List[float]
    measured_rgb_chromaticity: List[float]
    saturation_fraction: float
    black_fraction: float
    neutral_sample_log2_spread: float = 0.0
    neutral_sample_chromaticity_spread: float = 0.0
    as_shot_kelvin: Optional[float] = None
    as_shot_tint: Optional[float] = None


@dataclass
class CameraSolution:
    exposure_offset_stops: float
    rgb_gains: List[float]
    luminance_preserving_gain_normalization: bool
    final_exposure_offset_with_global_intent: float
    kelvin: Optional[int] = None
    tint: Optional[float] = None
    saturation: float = 1.0
    derivation_method: Optional[str] = None


@dataclass
class CameraQuality:
    confidence: float
    exposure_residual_stops: float
    color_residual: float
    flags: List[str]
    post_exposure_residual_stops: float = 0.0
    post_color_residual: float = 0.0
    neutral_sample_log2_spread: float = 0.0
    neutral_sample_chromaticity_spread: float = 0.0


@dataclass
class CameraCalibrationEntry:
    clip_id: str
    source_path: str
    camera_id: str
    group_key: str
    measurement: CameraMeasurement
    solution: CameraSolution
    quality: CameraQuality


@dataclass
class ArrayCalibration:
    schema: str
    capture_id: str
    created_at: str
    input_path: str
    mode: str
    backend: Optional[str]
    measurement_domain: str
    group_key: str
    target: ArrayTarget
    global_scene_intent: Dict[str, object]
    cameras: List[CameraCalibrationEntry]

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def extract_center_region(frame: np.ndarray, fraction: float = 0.4) -> np.ndarray:
    if frame.ndim != 3:
        raise ValueError("extract_center_region expects frame shape (C, H, W)")
    channels, height, width = frame.shape
    crop_height = max(1, int(round(height * fraction)))
    crop_width = max(1, int(round(width * fraction)))
    y0 = max(0, (height - crop_height) // 2)
    x0 = max(0, (width - crop_width) // 2)
    y1 = min(height, y0 + crop_height)
    x1 = min(width, x0 + crop_width)
    return frame[:, y0:y1, x0:x1]


def percentile_clip(values: np.ndarray, low_percentile: float = 5.0, high_percentile: float = 95.0) -> np.ndarray:
    if values.size == 0:
        return values
    low = np.percentile(values, low_percentile)
    high = np.percentile(values, high_percentile)
    clipped = values[(values >= low) & (values <= high)]
    return clipped if clipped.size else values


def compute_luminance(frame: np.ndarray) -> np.ndarray:
    return np.clip(frame[0] * 0.2126 + frame[1] * 0.7152 + frame[2] * 0.0722, 1e-6, None)


def calibrate_exposure_path(
    input_path: str,
    *,
    out_dir: str,
    source: str = "gray_card",
    sampling_mode: str = "detected_roi",
    roi_file: Optional[str] = None,
    target_log2_luminance: Optional[float] = None,
    reference_camera: Optional[str] = None,
    center_crop_width_ratio: float = DEFAULT_CENTER_CROP_WIDTH_RATIO,
    center_crop_height_ratio: float = DEFAULT_CENTER_CROP_HEIGHT_RATIO,
    backend: str = "auto",
) -> Dict[str, object]:
    if target_log2_luminance is None and reference_camera is None:
        raise ValueError("exposure calibration requires either --target-log2 or --reference-camera")
    if target_log2_luminance is not None and reference_camera is not None:
        raise ValueError("exposure calibration accepts either --target-log2 or --reference-camera, not both")
    normalized_source = normalize_exposure_source(source)
    normalized_sampling_mode = normalize_sampling_mode(sampling_mode)
    clips = discover_clips(input_path)
    if not clips:
        raise ValueError(f"No .R3D clips found under {input_path}")

    backend_impl = resolve_backend(backend)
    card_roi_map = load_card_roi_file(roi_file) if roi_file is not None and normalized_source != "gray_sphere" else {}
    sphere_roi_map = load_roi_file(roi_file) if roi_file is not None and normalized_source == "gray_sphere" else {}
    entries: list[ExposureCalibrationEntry] = []
    sampling_details: list[Dict[str, object]] = []
    seen_groups: set[str] = set()

    for clip_path in clips:
        clip = backend_impl.inspect_clip(str(clip_path))
        if normalized_source == "gray_sphere" and clip.group_key in seen_groups:
            continue
        decoded = list(backend_impl.decode_frames(str(clip_path), start_frame=0, max_frames=1, frame_step=1))
        if not decoded:
            continue
        seen_groups.add(clip.group_key)
        frame_index, _, image = decoded[0]
        region, detail = resolve_sampling_region(
            image,
            clip_id=clip.clip_id,
            group_key=clip.group_key,
            sampling_mode=normalized_sampling_mode,
            roi_map=sphere_roi_map if normalized_source == "gray_sphere" else card_roi_map,
            roi_shape="circle" if normalized_source == "gray_sphere" else "rect",
            center_crop_width_ratio=center_crop_width_ratio,
            center_crop_height_ratio=center_crop_height_ratio,
        )
        measurement = measure_exposure_region(image, region)
        entries.append(
            ExposureCalibrationEntry(
                clip_id=clip.clip_id,
                group_key=clip.group_key,
                source_path=str(Path(clip_path).expanduser().resolve()),
                sampling_mode=region.sampling_mode,
                sampling_region=region,
                measured_log2_luminance=measurement["measured_log2_luminance"],
                camera_baseline_stops=0.0,
                confidence=combine_confidence(
                    measurement["confidence"],
                    detail.get("detection_confidence"),
                ),
                calibration_source=normalized_source,
                exposure_stddev_logY=measurement["stddev_logY"],
                frame_index=frame_index,
            )
        )
        sampling_details.append(
            {
                "clip_id": clip.clip_id,
                "group_key": clip.group_key,
                "sampling_mode": region.sampling_mode,
                "sampling_region": region.to_dict(),
                "confidence": detail.get("detection_confidence"),
                "source_path": str(Path(clip_path).expanduser().resolve()),
            }
        )

    if not entries:
        raise ValueError("No exposure calibration entries were produced.")

    resolved_target = target_log2_luminance
    if reference_camera is not None:
        resolved_target = _find_reference_exposure_entry(entries, reference_camera).measured_log2_luminance
    assert resolved_target is not None
    grouped_offsets: Dict[str, List[float]] = {}
    for entry in entries:
        grouped_offsets.setdefault(entry.group_key, []).append(resolved_target - entry.measured_log2_luminance)
    normalized_offsets = {group_key: float(np.median(values)) for group_key, values in grouped_offsets.items()}
    for entry in entries:
        entry.camera_baseline_stops = normalized_offsets[entry.group_key]

    calibration = ExposureCalibration(
        calibration_type="exposure",
        object_type=normalized_source,
        cameras=sorted(entries, key=lambda item: item.group_key),
        target_log2_luminance=resolved_target,
        reference_camera=reference_camera,
        roi_file=str(Path(roi_file).expanduser().resolve()) if roi_file is not None else None,
    )
    out_root = Path(out_dir).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    detail_path = out_root / "exposure_sampling.json"
    calibration_path = out_root / "exposure_calibration.json"
    write_json(detail_path, {"calibration_type": "exposure", "sampling_mode": normalized_sampling_mode, "cameras": sampling_details})
    write_json(
        calibration_path,
        {
            **calibration.to_dict(),
            "clips": [
                {
                    "clip_id": entry.clip_id,
                    "group_key": entry.group_key,
                    "exposure_offset_stops": entry.camera_baseline_stops,
                    "measured_log2_luminance": entry.measured_log2_luminance,
                    "calibration_source": entry.calibration_source,
                    "sampling_mode": entry.sampling_mode,
                    "sampling_region": entry.sampling_region.to_dict(),
                    "confidence": {"exposure": entry.confidence},
                    "exposure_stddev_logY": entry.exposure_stddev_logY,
                    "provenance": {"source_path": entry.source_path},
                }
                for entry in calibration.cameras
            ],
        },
    )
    return {
        "input_path": str(Path(input_path).expanduser().resolve()),
        "calibration_type": "exposure",
        "object_type": normalized_source,
        "sampling_mode": normalized_sampling_mode,
        "camera_count": len(calibration.cameras),
        "detail_path": str(detail_path),
        "calibration_path": str(calibration_path),
    }


def calibrate_color_path(
    input_path: str,
    *,
    out_dir: str,
    sampling_mode: str = "detected_roi",
    roi_file: Optional[str] = None,
    center_crop_width_ratio: float = DEFAULT_CENTER_CROP_WIDTH_RATIO,
    center_crop_height_ratio: float = DEFAULT_CENTER_CROP_HEIGHT_RATIO,
    backend: str = "auto",
) -> Dict[str, object]:
    normalized_sampling_mode = normalize_sampling_mode(sampling_mode)
    clips = discover_clips(input_path)
    if not clips:
        raise ValueError(f"No .R3D clips found under {input_path}")
    backend_impl = resolve_backend(backend)
    card_roi_map = load_card_roi_file(roi_file) if roi_file is not None else {}
    entries: list[ColorCalibrationEntry] = []
    sampling_details: list[Dict[str, object]] = []

    for clip_path in clips:
        clip = backend_impl.inspect_clip(str(clip_path))
        decoded = list(backend_impl.decode_frames(str(clip_path), start_frame=0, max_frames=1, frame_step=1))
        if not decoded:
            continue
        frame_index, _, image = decoded[0]
        region, detail = resolve_sampling_region(
            image,
            clip_id=clip.clip_id,
            group_key=clip.group_key,
            sampling_mode=normalized_sampling_mode,
            roi_map=card_roi_map,
            roi_shape="rect",
            center_crop_width_ratio=center_crop_width_ratio,
            center_crop_height_ratio=center_crop_height_ratio,
        )
        measurement = measure_color_region(image, region)
        entries.append(
            ColorCalibrationEntry(
                clip_id=clip.clip_id,
                group_key=clip.group_key,
                source_path=str(Path(clip_path).expanduser().resolve()),
                sampling_mode=region.sampling_mode,
                sampling_region=region,
                measured_channel_medians=measurement["measured_channel_medians"],
                rgb_neutral_gains=measurement["rgb_neutral_gains"],
                confidence=combine_confidence(measurement["confidence"], detail.get("detection_confidence")),
                calibration_source="neutral_region",
                chromaticity_variance=measurement["chromaticity_variance"],
                frame_index=frame_index,
            )
        )
        sampling_details.append(
            {
                "clip_id": clip.clip_id,
                "group_key": clip.group_key,
                "sampling_mode": region.sampling_mode,
                "sampling_region": region.to_dict(),
                "confidence": detail.get("detection_confidence"),
                "source_path": str(Path(clip_path).expanduser().resolve()),
            }
        )

    reference_entries: Dict[str, ColorCalibrationEntry] = {}
    for entry in sorted(entries, key=lambda item: item.clip_id):
        reference_entries.setdefault(entry.group_key, entry)
    for entry in entries:
        reference = reference_entries[entry.group_key]
        entry.rgb_neutral_gains = {
            "r": reference.measured_channel_medians["r"] / max(entry.measured_channel_medians["r"], 1e-6),
            "g": reference.measured_channel_medians["g"] / max(entry.measured_channel_medians["g"], 1e-6),
            "b": reference.measured_channel_medians["b"] / max(entry.measured_channel_medians["b"], 1e-6),
        }

    calibration = ColorCalibration(
        calibration_type="color",
        object_type="neutral_patch",
        cameras=sorted(entries, key=lambda item: item.group_key),
        roi_file=str(Path(roi_file).expanduser().resolve()) if roi_file is not None else None,
    )
    out_root = Path(out_dir).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    detail_path = out_root / "color_sampling.json"
    calibration_path = out_root / "color_calibration.json"
    write_json(detail_path, {"calibration_type": "color", "sampling_mode": normalized_sampling_mode, "cameras": sampling_details})
    write_json(
        calibration_path,
        {
            **calibration.to_dict(),
            "clips": [
                {
                    "clip_id": entry.clip_id,
                    "group_key": entry.group_key,
                    "rgb_gains": [entry.rgb_neutral_gains["r"], entry.rgb_neutral_gains["g"], entry.rgb_neutral_gains["b"]],
                    "measured_channel_medians": entry.measured_channel_medians,
                    "calibration_source": entry.calibration_source,
                    "sampling_mode": entry.sampling_mode,
                    "sampling_region": entry.sampling_region.to_dict(),
                    "confidence": {"color": entry.confidence},
                    "chromaticity_variance": entry.chromaticity_variance,
                    "provenance": {"source_path": entry.source_path},
                }
                for entry in calibration.cameras
            ],
        },
    )
    return {
        "input_path": str(Path(input_path).expanduser().resolve()),
        "calibration_type": "color",
        "sampling_mode": normalized_sampling_mode,
        "camera_count": len(calibration.cameras),
        "detail_path": str(detail_path),
        "calibration_path": str(calibration_path),
    }


def calibrate_sphere_path(
    input_path: str,
    *,
    target_log2_luminance: float,
    out_dir: str,
    roi_file: str,
    backend: str = "auto",
) -> Dict[str, object]:
    payload = calibrate_exposure_path(
        input_path,
        out_dir=out_dir,
        source="gray_sphere",
        sampling_mode="detected_roi",
        roi_file=roi_file,
        target_log2_luminance=target_log2_luminance,
        backend=backend,
    )
    exposure = load_exposure_calibration(payload["calibration_path"])
    legacy = SphereCalibration(
        target_log2_luminance=float(exposure.target_log2_luminance),
        object_type="gray_sphere",
        roi_file=exposure.roi_file,
        cameras=[
            SphereCalibrationCameraEntry(
                clip_id=entry.clip_id,
                group_key=entry.group_key,
                roi=SphereROI(
                    cx=float(entry.sampling_region.roi["cx"]),
                    cy=float(entry.sampling_region.roi["cy"]),
                    r=float(entry.sampling_region.roi["r"]),
                ),
                measured_sphere_log2=entry.measured_log2_luminance,
                camera_baseline_stops=entry.camera_baseline_stops,
                confidence=entry.confidence,
                source_path=entry.source_path,
                frame_index=entry.frame_index,
            )
            for entry in exposure.cameras
        ],
    )
    legacy_path = Path(out_dir).expanduser().resolve() / "sphere_calibration.json"
    write_json(legacy_path, legacy.to_dict())
    payload["legacy_calibration_path"] = str(legacy_path)
    return payload


def calibrate_card_path(
    input_path: str,
    *,
    out_dir: str,
    roi_file: Optional[str] = None,
    target_log2_luminance: Optional[float] = None,
    reference_camera: Optional[str] = None,
    backend: str = "auto",
) -> Dict[str, object]:
    payload = calibrate_exposure_path(
        input_path,
        out_dir=out_dir,
        source="gray_card",
        sampling_mode="detected_roi",
        roi_file=roi_file,
        target_log2_luminance=target_log2_luminance,
        reference_camera=reference_camera,
        backend=backend,
    )
    exposure = load_exposure_calibration(payload["calibration_path"])
    detection = GrayCardDetection(
        object_type="gray_card",
        detection_mode="manual_override" if roi_file is not None else "auto",
        cameras=[
            GrayCardDetectionEntry(
                clip_id=entry.clip_id,
                group_key=entry.group_key,
                roi=GrayCardROI(
                    x=float(entry.sampling_region.roi["x"]),
                    y=float(entry.sampling_region.roi["y"]),
                    width=float(entry.sampling_region.roi["width"]),
                    height=float(entry.sampling_region.roi["height"]),
                ),
                confidence=entry.sampling_region.detection_confidence or entry.confidence,
                source_path=entry.source_path,
                method="manual" if roi_file is not None else "auto",
                frame_index=entry.frame_index,
            )
            for entry in exposure.cameras
        ],
        roi_file=exposure.roi_file,
    )
    legacy = GrayCardCalibration(
        target_log2_luminance=float(exposure.target_log2_luminance),
        calibration_mode="gray_card",
        object_type="gray_card",
        roi_file=exposure.roi_file,
        reference_camera=reference_camera,
        cameras=[
            GrayCardCalibrationCameraEntry(
                clip_id=entry.clip_id,
                group_key=entry.group_key,
                roi=GrayCardROI(
                    x=float(entry.sampling_region.roi["x"]),
                    y=float(entry.sampling_region.roi["y"]),
                    width=float(entry.sampling_region.roi["width"]),
                    height=float(entry.sampling_region.roi["height"]),
                ),
                measured_card_log2=entry.measured_log2_luminance,
                camera_baseline_stops=entry.camera_baseline_stops,
                confidence=entry.confidence,
                source_path=entry.source_path,
                frame_index=entry.frame_index,
            )
            for entry in exposure.cameras
        ],
    )
    out_root = Path(out_dir).expanduser().resolve()
    detection_path = out_root / "gray_card_detection.json"
    legacy_path = out_root / "gray_card_calibration.json"
    write_json(detection_path, detection.to_dict())
    write_json(legacy_path, legacy.to_dict())
    payload["detection_path"] = str(detection_path)
    payload["legacy_calibration_path"] = str(legacy_path)
    payload["calibration_mode"] = "gray_card"
    return payload


def normalize_sampling_mode(sampling_mode: str) -> str:
    normalized = sampling_mode.lower()
    if normalized not in {"full_frame", "center_crop", "detected_roi"}:
        raise ValueError("sampling mode must be full_frame, center_crop, or detected_roi")
    return normalized


def normalize_exposure_source(source: str) -> str:
    normalized = source.lower()
    if normalized not in {"gray_card", "gray_sphere", "center_crop", "roi"}:
        raise ValueError("exposure source must be gray_card, gray_sphere, center_crop, or roi")
    return normalized


def center_crop_roi(image: np.ndarray, width_ratio: float = DEFAULT_CENTER_CROP_WIDTH_RATIO, height_ratio: float = DEFAULT_CENTER_CROP_HEIGHT_RATIO) -> GrayCardROI:
    height = image.shape[1]
    width = image.shape[2]
    crop_width = max(1, int(round(width * width_ratio)))
    crop_height = max(1, int(round(height * height_ratio)))
    x = max(0, (width - crop_width) // 2)
    y = max(0, (height - crop_height) // 2)
    return GrayCardROI(x=float(x), y=float(y), width=float(crop_width), height=float(crop_height))


def resolve_sampling_region(
    image: np.ndarray,
    *,
    clip_id: str,
    group_key: str,
    sampling_mode: str,
    roi_map: Dict[str, object],
    roi_shape: str,
    center_crop_width_ratio: float,
    center_crop_height_ratio: float,
) -> Tuple[SamplingRegion, Dict[str, object]]:
    if sampling_mode == "full_frame":
        return SamplingRegion(sampling_mode="full_frame"), {"detection_confidence": None}
    if sampling_mode == "center_crop":
        crop = CenterCrop(width_ratio=center_crop_width_ratio, height_ratio=center_crop_height_ratio)
        return SamplingRegion(sampling_mode="center_crop", center_crop=crop), {"detection_confidence": None}

    roi = roi_map.get(clip_id) or roi_map.get(group_key)
    if roi is not None:
        return SamplingRegion(sampling_mode="detected_roi", roi=roi_to_dict(roi), detection_confidence=1.0), {"detection_confidence": 1.0}
    if roi_shape == "circle":
        crop = CenterCrop(width_ratio=center_crop_width_ratio, height_ratio=center_crop_height_ratio)
        return SamplingRegion(sampling_mode="center_crop", center_crop=crop, fallback_used=True), {"detection_confidence": 0.0}

    detected = detect_gray_card_roi(image)
    detection_confidence = float(detected["confidence"])
    if detection_confidence < LOW_CONFIDENCE_THRESHOLD:
        crop = CenterCrop(width_ratio=center_crop_width_ratio, height_ratio=center_crop_height_ratio)
        return SamplingRegion(
            sampling_mode="center_crop",
            center_crop=crop,
            detection_confidence=detection_confidence,
            fallback_used=True,
        ), {"detection_confidence": detection_confidence}
    return SamplingRegion(
        sampling_mode="detected_roi",
        roi=roi_to_dict(detected["roi"]),
        detection_confidence=detection_confidence,
    ), {"detection_confidence": detection_confidence}


def extract_region_pixels(image: np.ndarray, region: SamplingRegion) -> Tuple[np.ndarray, Dict[str, float]]:
    if region.sampling_mode == "full_frame":
        pixels = np.moveaxis(image, 0, -1).reshape(-1, 3)
        return pixels, {"area_fraction": 1.0}
    if region.sampling_mode == "center_crop":
        assert region.center_crop is not None
        roi = center_crop_roi(image, region.center_crop.width_ratio, region.center_crop.height_ratio)
        return extract_rect_pixels(image, roi)
    if region.roi is None:
        raise ValueError("detected_roi sampling requires ROI data")
    if {"cx", "cy", "r"} <= set(region.roi):
        return extract_circle_pixels(
            image,
            SphereROI(cx=float(region.roi["cx"]), cy=float(region.roi["cy"]), r=float(region.roi["r"])),
        )
    return extract_rect_pixels(
        image,
        GrayCardROI(
            x=float(region.roi["x"]),
            y=float(region.roi["y"]),
            width=float(region.roi["width"]),
            height=float(region.roi["height"]),
        ),
    )


def extract_rect_pixels(image: np.ndarray, roi: GrayCardROI) -> Tuple[np.ndarray, Dict[str, float]]:
    x0 = max(0, int(np.floor(roi.x)))
    y0 = max(0, int(np.floor(roi.y)))
    x1 = min(image.shape[2], int(np.ceil(roi.x + roi.width)))
    y1 = min(image.shape[1], int(np.ceil(roi.y + roi.height)))
    pixels = np.moveaxis(image[:, y0:y1, x0:x1], 0, -1).reshape(-1, 3)
    area_fraction = float(max((x1 - x0) * (y1 - y0), 0)) / float(max(image.shape[1] * image.shape[2], 1))
    return pixels, {"area_fraction": area_fraction}


def extract_circle_pixels(image: np.ndarray, roi: SphereROI) -> Tuple[np.ndarray, Dict[str, float]]:
    yy, xx = np.meshgrid(np.arange(image.shape[1], dtype=np.float32), np.arange(image.shape[2], dtype=np.float32), indexing="ij")
    mask = ((xx - roi.cx) ** 2 + (yy - roi.cy) ** 2) <= roi.r ** 2
    pixels = np.moveaxis(image, 0, -1)[mask]
    area_fraction = float(mask.sum()) / float(max(mask.size, 1))
    return pixels, {"area_fraction": area_fraction}


def measure_exposure_region(image: np.ndarray, region: SamplingRegion, *, low_percentile: float = 5.0, high_percentile: float = 95.0) -> Dict[str, float]:
    centered = image if region.sampling_mode == "detected_roi" else extract_center_region(image, fraction=0.4)
    pixels, info = extract_region_pixels(centered, region)
    luminance = np.clip(pixels[:, 0] * 0.2126 + pixels[:, 1] * 0.7152 + pixels[:, 2] * 0.0722, 1e-6, 1.0)
    measurement = trimmed_luminance_measurement(luminance, low_percentile=low_percentile, high_percentile=high_percentile)
    log_values = np.log2(np.clip(percentile_clip(luminance, low_percentile, high_percentile), 1e-6, 1.0))
    variance_penalty = 1.0 - min(float(np.std(log_values)) / 1.5, 1.0)
    measurement["confidence"] = combine_confidence(
        measurement["confidence"],
        variance_penalty,
        info["area_fraction"],
    )
    measurement["stddev_logY"] = float(np.std(log_values))
    return measurement


def measure_color_region(image: np.ndarray, region: SamplingRegion, *, low_percentile: float = 5.0, high_percentile: float = 95.0) -> Dict[str, object]:
    centered = image if region.sampling_mode == "detected_roi" else extract_center_region(image, fraction=0.4)
    pixels, info = extract_region_pixels(centered, region)
    clipped = np.any((pixels <= 0.002) | (pixels >= 0.998), axis=1)
    valid_pixels = pixels[~clipped]
    if valid_pixels.size == 0:
        valid_pixels = pixels
    luminance = np.clip(valid_pixels[:, 0] * 0.2126 + valid_pixels[:, 1] * 0.7152 + valid_pixels[:, 2] * 0.0722, 1e-6, 1.0)
    trimmed_luminance = percentile_clip(luminance, low_percentile, high_percentile)
    low = float(trimmed_luminance.min())
    high = float(trimmed_luminance.max())
    trimmed_pixels = valid_pixels[(luminance >= low) & (luminance <= high)]
    if trimmed_pixels.size == 0:
        trimmed_pixels = valid_pixels
    medians = compute_chromaticity_medians(trimmed_pixels)
    gains = solve_neutral_gains(medians["r"], medians["g"], medians["b"])
    clipped_fraction = 1.0 - (float(valid_pixels.shape[0]) / float(max(pixels.shape[0], 1)))
    channel_variance = chromaticity_variance(trimmed_pixels) if trimmed_pixels.size else 1.0
    confidence = combine_confidence(
        max(0.0, min(1.0, float(trimmed_pixels.shape[0]) / float(max(pixels.shape[0], 1)))),
        1.0 - clipped_fraction,
        1.0 - min(channel_variance / 0.05, 1.0),
        info["area_fraction"],
    )
    return {
        "measured_channel_medians": medians,
        "rgb_neutral_gains": gains,
        "rgb_gains": [gains["r"], gains["g"], gains["b"]],
        "clipped_fraction": clipped_fraction,
        "confidence": confidence,
        "chromaticity_variance": channel_variance,
    }


def trimmed_luminance_measurement(luminance: np.ndarray, *, low_percentile: float = 5.0, high_percentile: float = 95.0) -> Dict[str, float]:
    if luminance.size == 0:
        raise ValueError("sample region produced no pixels")
    unclipped = luminance[(luminance > 0.002) & (luminance < 0.998)]
    if unclipped.size == 0:
        unclipped = luminance
    trimmed = percentile_clip(unclipped, low_percentile, high_percentile)
    measured = float(np.median(np.log2(trimmed)))
    clipped_fraction = 1.0 - (float(unclipped.size) / float(max(luminance.size, 1)))
    retained_fraction = float(trimmed.size) / float(max(luminance.size, 1))
    confidence = combine_confidence(retained_fraction, 1.0 - clipped_fraction)
    return {
        "measured_log2_luminance": measured,
        "clipped_fraction": clipped_fraction,
        "retained_fraction": retained_fraction,
        "confidence": confidence,
    }


def combine_confidence(*values: Optional[float]) -> float:
    filtered = [max(0.0, min(1.0, float(value))) for value in values if value is not None]
    if not filtered:
        return 0.0
    return float(np.prod(filtered) ** (1.0 / len(filtered)))


def measure_sphere_from_roi(image: np.ndarray, roi: SphereROI, *, low_percentile: float = 5.0, high_percentile: float = 95.0) -> Dict[str, float]:
    result = measure_exposure_region(
        image,
        SamplingRegion(sampling_mode="detected_roi", roi=roi_to_dict(roi), detection_confidence=1.0),
        low_percentile=low_percentile,
        high_percentile=high_percentile,
    )
    return {
        "measured_sphere_log2": result["measured_log2_luminance"],
        "clipped_fraction": result["clipped_fraction"],
        "confidence": result["confidence"],
        "retained_fraction": result["retained_fraction"],
    }


def measure_card_from_roi(image: np.ndarray, roi: GrayCardROI, *, low_percentile: float = 5.0, high_percentile: float = 95.0) -> Dict[str, float]:
    result = measure_exposure_region(
        image,
        SamplingRegion(sampling_mode="detected_roi", roi=roi_to_dict(roi), detection_confidence=1.0),
        low_percentile=low_percentile,
        high_percentile=high_percentile,
    )
    return {
        "measured_card_log2": result["measured_log2_luminance"],
        "clipped_fraction": result["clipped_fraction"],
        "confidence": result["confidence"],
        "retained_fraction": result["retained_fraction"],
    }


def detect_gray_card_roi(image: np.ndarray) -> Dict[str, object]:
    luminance = np.clip(image[0] * 0.2126 + image[1] * 0.7152 + image[2] * 0.0722, 1e-6, 1.0)
    saturation = image.max(axis=0) - image.min(axis=0)
    candidate_mask = (saturation < 0.06) & (luminance > 0.08) & (luminance < 0.75)
    components = _connected_components(candidate_mask)
    if not components:
        fallback = GrayCardROI(x=image.shape[2] * 0.35, y=image.shape[1] * 0.35, width=image.shape[2] * 0.3, height=image.shape[1] * 0.3)
        return {"roi": fallback, "confidence": 0.1}

    best_score = -1.0
    best_roi: Optional[GrayCardROI] = None
    for coords in components:
        ys = np.array([coord[0] for coord in coords], dtype=np.int32)
        xs = np.array([coord[1] for coord in coords], dtype=np.int32)
        x0 = int(xs.min())
        x1 = int(xs.max()) + 1
        y0 = int(ys.min())
        y1 = int(ys.max()) + 1
        width = x1 - x0
        height = y1 - y0
        area = float(len(coords))
        box_area = float(max(width * height, 1))
        fill_ratio = area / box_area
        mean_luma = float(luminance[y0:y1, x0:x1].mean())
        mean_sat = float(saturation[y0:y1, x0:x1].mean())
        compactness = min(width, height) / float(max(width, height, 1))
        score = min(area / float(image.shape[1] * image.shape[2]), 1.0) * 2.0 + fill_ratio + compactness + (1.0 - min(mean_sat / 0.06, 1.0)) + (1.0 - abs(mean_luma - 0.18) / 0.18)
        if score > best_score:
            best_score = score
            best_roi = GrayCardROI(x=float(x0), y=float(y0), width=float(width), height=float(height))

    assert best_roi is not None
    return {"roi": best_roi, "confidence": max(0.0, min(1.0, best_score / 5.0))}


def load_roi_file(path: str) -> Dict[str, SphereROI]:
    payload = json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))
    return {group_key: SphereROI(cx=float(entry["cx"]), cy=float(entry["cy"]), r=float(entry["r"])) for group_key, entry in payload.items()}


def load_card_roi_file(path: str) -> Dict[str, GrayCardROI]:
    payload = json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))
    return {key: GrayCardROI(x=float(entry["x"]), y=float(entry["y"]), width=float(entry["width"]), height=float(entry["height"])) for key, entry in payload.items()}


def load_exposure_calibration(path: str) -> ExposureCalibration:
    payload = json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))
    if payload.get("calibration_type") == "exposure":
        entries_payload = payload.get("cameras", payload.get("clips", []))
        return ExposureCalibration(
            calibration_type="exposure",
            object_type=str(payload["object_type"]),
            target_log2_luminance=float(payload["target_log2_luminance"]) if payload.get("target_log2_luminance") is not None else None,
            reference_camera=payload.get("reference_camera"),
            roi_file=payload.get("roi_file"),
            cameras=[
                ExposureCalibrationEntry(
                    clip_id=str(entry["clip_id"]),
                    group_key=str(entry["group_key"]),
                    source_path=str(entry.get("source_path", entry.get("provenance", {}).get("source_path", ""))),
                    sampling_mode=str(entry["sampling_mode"]),
                    sampling_region=_sampling_region_from_payload(entry.get("sampling_region", {"sampling_mode": entry["sampling_mode"], "roi": None, "center_crop": None})),
                    measured_log2_luminance=float(entry["measured_log2_luminance"]),
                    camera_baseline_stops=float(entry.get("camera_baseline_stops", entry.get("exposure_offset_stops", 0.0))),
                    confidence=float(entry["confidence"]["exposure"] if isinstance(entry.get("confidence"), dict) else entry["confidence"]),
                    calibration_source=str(entry["calibration_source"]),
                    exposure_stddev_logY=float(entry.get("exposure_stddev_logY", 0.0)),
                    frame_index=int(entry.get("frame_index", 0)),
                )
                for entry in entries_payload
            ],
        )
    legacy = load_calibration(path)
    return ExposureCalibration(
        calibration_type="exposure",
        object_type=str(legacy.object_type),
        target_log2_luminance=float(legacy.target_log2_luminance),
        reference_camera=getattr(legacy, "reference_camera", None),
        roi_file=getattr(legacy, "roi_file", None),
        cameras=_legacy_to_exposure_entries(legacy),
    )


def load_color_calibration(path: str) -> ColorCalibration:
    payload = json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))
    entries_payload = payload.get("cameras", payload.get("clips", []))
    return ColorCalibration(
        calibration_type="color",
        object_type=str(payload["object_type"]),
        roi_file=payload.get("roi_file"),
        cameras=[
            ColorCalibrationEntry(
                clip_id=str(entry["clip_id"]),
                group_key=str(entry["group_key"]),
                source_path=str(entry.get("source_path", entry.get("provenance", {}).get("source_path", ""))),
                sampling_mode=str(entry["sampling_mode"]),
                sampling_region=_sampling_region_from_payload(entry.get("sampling_region", {"sampling_mode": entry["sampling_mode"], "roi": None, "center_crop": None})),
                measured_channel_medians={key: float(value) for key, value in entry["measured_channel_medians"].items()},
                rgb_neutral_gains=(
                    {key: float(value) for key, value in entry["rgb_neutral_gains"].items()}
                    if entry.get("rgb_neutral_gains") is not None
                    else {"r": float(entry["rgb_gains"][0]), "g": float(entry["rgb_gains"][1]), "b": float(entry["rgb_gains"][2])}
                ),
                confidence=float(entry["confidence"]["color"] if isinstance(entry.get("confidence"), dict) else entry["confidence"]),
                calibration_source=str(entry["calibration_source"]),
                chromaticity_variance=float(entry.get("chromaticity_variance", 0.0)),
                frame_index=int(entry.get("frame_index", 0)),
            )
            for entry in entries_payload
        ],
    )


def load_calibration(path: str) -> CalibrationLike:
    payload = json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))
    calibration_type = str(payload.get("calibration_type", ""))
    calibration_mode = str(payload.get("calibration_mode", "sphere"))
    if calibration_type == "exposure":
        return load_exposure_calibration(path)
    if calibration_type == "color":
        raise ValueError("Use load_color_calibration() for color calibration JSON.")
    if calibration_mode == "gray_card":
        return GrayCardCalibration(
            target_log2_luminance=float(payload["target_log2_luminance"]),
            calibration_mode=calibration_mode,
            object_type=str(payload["object_type"]),
            roi_file=payload.get("roi_file"),
            reference_camera=payload.get("reference_camera"),
            cameras=[
                GrayCardCalibrationCameraEntry(
                    clip_id=str(entry["clip_id"]),
                    group_key=str(entry.get("group_key", entry.get("camera_group"))),
                    roi=GrayCardROI(x=float(entry["roi"]["x"]), y=float(entry["roi"]["y"]), width=float(entry["roi"]["width"]), height=float(entry["roi"]["height"])),
                    measured_card_log2=float(entry["measured_card_log2"]),
                    camera_baseline_stops=float(entry["camera_baseline_stops"]),
                    confidence=float(entry["confidence"]),
                    source_path=str(entry["source_path"]),
                    frame_index=int(entry.get("frame_index", 0)),
                )
                for entry in payload["cameras"]
            ],
        )
    return SphereCalibration(
        target_log2_luminance=float(payload["target_log2_luminance"]),
        object_type=str(payload["object_type"]),
        calibration_mode=calibration_mode,
        roi_file=payload.get("roi_file"),
        cameras=[
            SphereCalibrationCameraEntry(
                clip_id=str(entry.get("clip_id", "")),
                group_key=str(entry.get("group_key", entry.get("camera_group"))),
                roi=SphereROI(cx=float(entry["roi"]["cx"]), cy=float(entry["roi"]["cy"]), r=float(entry["roi"]["r"])),
                measured_sphere_log2=float(entry["measured_sphere_log2"]),
                camera_baseline_stops=float(entry["camera_baseline_stops"]),
                confidence=float(entry["confidence"]),
                source_path=str(entry["source_path"]),
                frame_index=int(entry.get("frame_index", 0)),
            )
            for entry in payload["cameras"]
        ],
    )


def calibration_baselines(calibration: CalibrationLike | ExposureCalibration) -> Dict[str, float]:
    return {entry.group_key: entry.camera_baseline_stops for entry in calibration.cameras}


def color_calibration_gains(calibration: ColorCalibration) -> Dict[str, Dict[str, float]]:
    return {entry.group_key: entry.rgb_neutral_gains for entry in calibration.cameras}


def derive_array_group_key(input_path: str) -> str:
    root = Path(input_path).expanduser().resolve()
    return f"array_{root.name or 'batch'}"


def build_array_calibration_from_analysis(
    results: List[object],
    *,
    input_path: str,
    capture_id: Optional[str] = None,
    group_key: Optional[str] = None,
) -> ArrayCalibration:
    if not results:
        raise ValueError("array calibration requires at least one clip result")
    resolved_group_key = group_key or derive_array_group_key(input_path)
    resolved_capture_id = capture_id or Path(input_path).expanduser().resolve().name or resolved_group_key
    measured_log2 = np.array(
        [
            float(
                item.diagnostics.get(
                    "measured_log2_luminance_raw",
                    item.diagnostics["measured_log2_luminance_monitoring"],
                )
            )
            for item in results
        ],
        dtype=np.float32,
    )
    measured_chroma = np.array([item.diagnostics["measured_rgb_chromaticity"] for item in results], dtype=np.float32)
    target_log2 = float(np.median(percentile_clip(measured_log2)))
    target_chroma = percentile_clip(measured_chroma[:, 0]), percentile_clip(measured_chroma[:, 1]), percentile_clip(measured_chroma[:, 2])
    target_rgb = [
        float(np.median(target_chroma[0])),
        float(np.median(target_chroma[1])),
        float(np.median(target_chroma[2])),
    ]
    print(f"[r3dmatch] array calibration clips={len(results)}")
    print(f"[r3dmatch] shared target exposure monitoring log2={target_log2:.6f}")
    print(f"[r3dmatch] shared target chromaticity={target_rgb}")

    wb_model_solution = solve_white_balance_model_for_records(
        [
            {
                "clip_id": item.clip_id,
                "measured_rgb_chromaticity": item.diagnostics["measured_rgb_chromaticity"],
                "target_rgb_chromaticity": target_rgb,
                "clip_metadata": item.clip_metadata.to_dict() if hasattr(item.clip_metadata, "to_dict") else None,
                "rgb_gains": None,
                "confidence": float(item.confidence) if float(item.confidence) > 0.0 else None,
                "sample_log2_spread": float(item.diagnostics.get("neutral_sample_log2_spread", 0.0) or 0.0),
                "sample_chromaticity_spread": float(item.diagnostics.get("neutral_sample_chromaticity_spread", 0.0) or 0.0),
            }
            for item in results
        ],
        target_rgb_chromaticity=target_rgb,
    )

    cameras: list[CameraCalibrationEntry] = []
    for item in results:
        measured_rgb = list(item.diagnostics["measured_rgb_mean"])
        measured_chromaticity = list(item.diagnostics["measured_rgb_chromaticity"])
        raw_gains = [
            target_rgb[0] / max(measured_chromaticity[0], 1e-6),
            target_rgb[1] / max(measured_chromaticity[1], 1e-6),
            target_rgb[2] / max(measured_chromaticity[2], 1e-6),
        ]
        gain_norm = (raw_gains[0] * raw_gains[1] * raw_gains[2]) ** (1.0 / 3.0)
        normalized_gains = [float(value / max(gain_norm, 1e-6)) for value in raw_gains]
        measured_log2_value = float(
            item.diagnostics.get(
                "measured_log2_luminance_raw",
                item.diagnostics["measured_log2_luminance_monitoring"],
            )
        )
        exposure_offset = float(target_log2 - measured_log2_value)
        wb_solution = dict(wb_model_solution["clips"][item.clip_id])
        color_residual = float(wb_solution["pre_neutral_residual"])
        post_color_residual = float(wb_solution["post_neutral_residual"])
        sample_log2_spread = float(item.diagnostics.get("neutral_sample_log2_spread", 0.0) or 0.0)
        sample_chroma_spread = float(item.diagnostics.get("neutral_sample_chromaticity_spread", 0.0) or 0.0)
        accepted_ratio = float(item.diagnostics.get("accepted_frames", 0)) / max(float(item.diagnostics.get("sampled_frames", 1)), 1.0)
        quality_confidence = combine_confidence(
            float(item.confidence) if float(item.confidence) > 0.0 else accepted_ratio,
            1.0 - min(sample_log2_spread / 0.35, 1.0),
            1.0 - min(sample_chroma_spread / 0.03, 1.0),
            1.0 - min(post_color_residual / 0.05, 1.0),
        )
        flags: list[str] = []
        if sample_log2_spread > 0.12:
            flags.append("neutral_sample_exposure_spread_high")
        if sample_chroma_spread > 0.02:
            flags.append("neutral_sample_chromaticity_spread_high")
        entry = CameraCalibrationEntry(
            clip_id=item.clip_id,
            source_path=item.source_path,
            camera_id=item.clip_id.split("_", 2)[0] + "_" + item.clip_id.split("_", 2)[1] if item.clip_id.count("_") >= 1 else item.clip_id,
            group_key=resolved_group_key,
            measurement=CameraMeasurement(
                gray_sample_count=int(item.diagnostics.get("neutral_sample_count", 0) or 0),
                valid_pixel_count=int(item.diagnostics.get("valid_pixel_count", 0)),
                measured_log2_luminance=measured_log2_value,
                measured_rgb_mean=[float(value) for value in measured_rgb],
                measured_rgb_chromaticity=[float(value) for value in measured_chromaticity],
                saturation_fraction=float(item.diagnostics.get("raw_saturation_fraction", item.diagnostics.get("saturation_fraction", 0.0))),
                black_fraction=float(item.diagnostics.get("black_fraction", 0.0)),
                neutral_sample_log2_spread=sample_log2_spread,
                neutral_sample_chromaticity_spread=sample_chroma_spread,
                as_shot_kelvin=float(wb_solution["as_shot_kelvin"]),
                as_shot_tint=float(wb_solution["as_shot_tint"]),
            ),
            solution=CameraSolution(
                exposure_offset_stops=exposure_offset,
                rgb_gains=normalized_gains,
                luminance_preserving_gain_normalization=True,
                final_exposure_offset_with_global_intent=exposure_offset,
                kelvin=int(wb_solution["kelvin"]),
                tint=float(wb_solution["tint"]),
                saturation=1.0,
                derivation_method=str(wb_solution["method"]),
            ),
            quality=CameraQuality(
                confidence=quality_confidence,
                exposure_residual_stops=abs(exposure_offset),
                color_residual=color_residual,
                flags=flags,
                post_exposure_residual_stops=0.0,
                post_color_residual=post_color_residual,
                neutral_sample_log2_spread=sample_log2_spread,
                neutral_sample_chromaticity_spread=sample_chroma_spread,
            ),
        )
        print(
            f"[r3dmatch] camera={entry.camera_id} offset={entry.solution.exposure_offset_stops:.6f} "
            f"gains={entry.solution.rgb_gains} kelvin={entry.solution.kelvin} tint={entry.solution.tint}"
        )
        cameras.append(entry)

    return ArrayCalibration(
        schema="r3dmatch_array_calibration_v1",
        capture_id=resolved_capture_id,
        created_at=datetime.now(timezone.utc).isoformat(),
        input_path=str(Path(input_path).expanduser().resolve()),
        mode="array_gray_sphere",
        backend=str(getattr(results[0], "backend", "") or ""),
        measurement_domain="scene",
        group_key=resolved_group_key,
        target=ArrayTarget(
            method="robust_array_center",
            exposure=ExposureTarget(
                log2_luminance_target=target_log2,
                estimator="median",
                included_camera_count=len(results),
                excluded_camera_ids=[],
            ),
            color=ColorTarget(
                target_rgb_chromaticity=target_rgb,
                estimator="median",
                included_camera_count=len(results),
                excluded_camera_ids=[],
            ),
        ),
        global_scene_intent={
            "enabled": False,
            "global_exposure_offset_stops": 0.0,
            "notes": None,
            "white_balance_model": {
                "model_key": wb_model_solution.get("model_key"),
                "model_label": wb_model_solution.get("model_label"),
                "shared_kelvin": wb_model_solution.get("shared_kelvin"),
                "shared_tint": wb_model_solution.get("shared_tint"),
                "metrics": wb_model_solution.get("metrics"),
                "candidates": wb_model_solution.get("candidates", []),
            },
        },
        cameras=cameras,
    )


def write_array_calibration_json(calibration: ArrayCalibration, path: str | Path) -> str:
    target_path = Path(path).expanduser().resolve()
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(json.dumps(calibration.to_dict(), indent=2), encoding="utf-8")
    return str(target_path)


def discover_clips(input_path: str) -> List[Path]:
    path = Path(input_path).expanduser().resolve()
    if path.is_file():
        return [path] if is_valid_clip_file(path) else []
    clips: list[Path] = []
    for candidate in path.rglob("*"):
        if candidate.is_file() and is_valid_clip_file(candidate):
            clips.append(candidate)
    return sorted(clips)


def is_valid_clip_file(path: Path) -> bool:
    if path.suffix.lower() != ".r3d":
        return False
    if any(part.startswith(".") or part.startswith("._") for part in path.parts):
        return False
    return True


def camera_group_from_clip_id(clip_id: str) -> str:
    return clip_id.split("_", 1)[0] if "_" in clip_id else clip_id


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def roi_to_dict(roi: object) -> Dict[str, float]:
    if isinstance(roi, GrayCardROI):
        return {"x": roi.x, "y": roi.y, "width": roi.width, "height": roi.height}
    if isinstance(roi, SphereROI):
        return {"cx": roi.cx, "cy": roi.cy, "r": roi.r}
    if isinstance(roi, dict):
        return {str(key): float(value) for key, value in roi.items()}
    raise TypeError(f"Unsupported ROI type: {type(roi)!r}")


def _sampling_region_from_payload(payload: Dict[str, object]) -> SamplingRegion:
    center_crop_payload = payload.get("center_crop")
    return SamplingRegion(
        sampling_mode=str(payload["sampling_mode"]),
        roi={str(key): float(value) for key, value in payload["roi"].items()} if payload.get("roi") else None,
        center_crop=CenterCrop(
            width_ratio=float(center_crop_payload["width_ratio"]),
            height_ratio=float(center_crop_payload["height_ratio"]),
        ) if center_crop_payload else None,
        detection_confidence=float(payload["detection_confidence"]) if payload.get("detection_confidence") is not None else None,
        fallback_used=bool(payload.get("fallback_used", False)),
    )


def _legacy_to_exposure_entries(legacy: CalibrationLike) -> List[ExposureCalibrationEntry]:
    if isinstance(legacy, GrayCardCalibration):
        return [
            ExposureCalibrationEntry(
                clip_id=entry.clip_id,
                group_key=entry.group_key,
                source_path=entry.source_path,
                sampling_mode="detected_roi",
                sampling_region=SamplingRegion(sampling_mode="detected_roi", roi=roi_to_dict(entry.roi), detection_confidence=entry.confidence),
                measured_log2_luminance=entry.measured_card_log2,
                camera_baseline_stops=entry.camera_baseline_stops,
                confidence=entry.confidence,
                calibration_source="gray_card",
                frame_index=entry.frame_index,
            )
            for entry in legacy.cameras
        ]
    return [
        ExposureCalibrationEntry(
            clip_id=entry.clip_id,
            group_key=entry.group_key,
            source_path=entry.source_path,
            sampling_mode="detected_roi",
            sampling_region=SamplingRegion(sampling_mode="detected_roi", roi=roi_to_dict(entry.roi), detection_confidence=entry.confidence),
            measured_log2_luminance=entry.measured_sphere_log2,
            camera_baseline_stops=entry.camera_baseline_stops,
            confidence=entry.confidence,
            calibration_source="gray_sphere",
            frame_index=entry.frame_index,
        )
        for entry in legacy.cameras
    ]


def _find_reference_exposure_entry(entries: List[ExposureCalibrationEntry], reference_camera: str) -> ExposureCalibrationEntry:
    for entry in entries:
        if entry.group_key == reference_camera or entry.clip_id == reference_camera:
            return entry
    raise ValueError(f"reference camera/clip not found in calibration set: {reference_camera}")


def _connected_components(mask: np.ndarray) -> List[List[Tuple[int, int]]]:
    visited = np.zeros_like(mask, dtype=bool)
    height, width = mask.shape
    components: list[list[tuple[int, int]]] = []
    for y in range(height):
        for x in range(width):
            if not mask[y, x] or visited[y, x]:
                continue
            stack = [(y, x)]
            visited[y, x] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cy, cx = stack.pop()
                coords.append((cy, cx))
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
            components.append(coords)
    return components
