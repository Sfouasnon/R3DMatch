from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np

from .calibration import (
    build_array_calibration_from_analysis,
    calibration_baselines,
    color_calibration_gains,
    derive_array_group_key,
    extract_center_region,
    load_color_calibration,
    load_exposure_calibration,
    measure_sphere_region_statistics,
    measure_sphere_zone_profile_statistics,
    percentile_clip,
    write_array_calibration_json,
)
from .execution import raise_if_cancelled
from .identity import clip_id_from_path, legacy_camera_group_from_clip_id, subset_key_from_clip_id
from .luts import CubeLut, apply_lut, load_cube_lut
from .models import ClipResult, FrameStat, MonitoringContext, SamplePlan, SphereROI
from .progress import emit_review_progress
from .sdk import resolve_backend
from .sidecar import write_sidecar_file


def analyze_path(
    input_path: str,
    *,
    out_dir: str,
    mode: str,
    backend: str,
    lut_override: Optional[str],
    calibration_path: Optional[str],
    exposure_calibration_path: Optional[str] = None,
    color_calibration_path: Optional[str] = None,
    calibration_mode: Optional[str] = None,
    sample_count: int = 8,
    sampling_strategy: str = "uniform",
    clamp_stops: float = 3.0,
    calibration_roi: Optional[Dict[str, float]] = None,
    target_type: Optional[str] = None,
    selected_clip_ids: Optional[List[str]] = None,
    selected_clip_groups: Optional[List[str]] = None,
    progress_path: Optional[str] = None,
    half_res_decode: bool = False,
    workload_trace_path: Optional[str] = None,
    runtime_trace_path: Optional[str] = None,
    invocation_source: str = "direct_cli",
    measurement_source: str = "scene_sdk_decode_with_proxy_monitoring",
) -> Dict[str, object]:
    analysis_started_at = time.perf_counter()
    emit_review_progress(
        progress_path,
        phase="analysis_start",
        detail="Starting source discovery.",
        stage_label="Scanning sources",
    )
    raise_if_cancelled("Run cancelled before source discovery.")
    clips = discover_clips(input_path)
    if not clips:
        raise ValueError(f"No .R3D clips found under {input_path}")
    emit_review_progress(
        progress_path,
        phase="source_scan_complete",
        detail=f"Discovered {len(clips)} clip(s).",
        stage_label="Scanning sources",
        clip_count=len(clips),
        elapsed_seconds=time.perf_counter() - analysis_started_at,
    )

    requested_clip_ids = {str(item).strip() for item in (selected_clip_ids or []) if str(item).strip()}
    requested_clip_groups = {str(item).strip() for item in (selected_clip_groups or []) if str(item).strip()}
    if requested_clip_ids or requested_clip_groups:
        filtered = []
        for clip in clips:
            raise_if_cancelled("Run cancelled while filtering selected clips.")
            clip_id = clip_id_from_path(str(clip))
            subset_key = subset_key_from_clip_id(clip_id)
            if clip_id in requested_clip_ids or subset_key in requested_clip_groups:
                filtered.append(clip)
        clips = filtered
        if not clips:
            raise ValueError(
                "No .R3D clips matched the requested subset selection "
                f"(clip_ids={sorted(requested_clip_ids)}, clip_groups={sorted(requested_clip_groups)})"
            )
    emit_review_progress(
        progress_path,
        phase="subset_resolved",
        detail=f"Resolved subset to {len(clips)} clip(s).",
        stage_label="Resolving subset",
        clip_count=len(clips),
        elapsed_seconds=time.perf_counter() - analysis_started_at,
        extra={
            "selected_clip_ids": sorted(requested_clip_ids),
            "selected_clip_groups": sorted(requested_clip_groups),
        },
    )

    out_root = Path(out_dir).expanduser().resolve()
    analysis_dir = out_root / "analysis"
    sidecar_dir = out_root / "sidecars"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    sidecar_dir.mkdir(parents=True, exist_ok=True)
    rendered_preview_context = None
    if str(measurement_source or "").strip() == "rendered_preview_ipp2":
        rendered_preview_context = _resolve_rendered_preview_context()
    emit_review_progress(
        progress_path,
        phase="measurement_start",
        detail="Starting measurement phase." if str(measurement_source or "").strip() != "rendered_preview_ipp2" else "Starting IPP2 render-and-measure phase.",
        stage_label="Measuring clips",
        clip_count=len(clips),
        elapsed_seconds=time.perf_counter() - analysis_started_at,
    )

    provisional: list[ClipResult] = []
    for clip_index, clip in enumerate(clips, start=1):
        raise_if_cancelled("Run cancelled during clip measurement.")
        clip_id = clip_id_from_path(str(clip))
        clip_started_at = time.perf_counter()
        emit_review_progress(
            progress_path,
            phase="clip_measurement_start",
            detail="Measuring clip." if str(measurement_source or "").strip() != "rendered_preview_ipp2" else "Rendering and measuring clip in IPP2.",
            stage_label="Measuring clips",
            clip_index=clip_index,
            clip_count=len(clips),
            current_clip_id=clip_id,
            elapsed_seconds=time.perf_counter() - analysis_started_at,
        )
        provisional.append(
            analyze_clip(
                str(clip),
                mode=mode,
                backend=backend,
                lut_override=lut_override,
                sample_count=sample_count,
                sampling_strategy=sampling_strategy,
                calibration_roi=calibration_roi,
                target_type=target_type,
                half_res_decode=half_res_decode,
                measurement_source=measurement_source,
                measurement_output_dir=str(out_root),
                rendered_preview_context=rendered_preview_context,
            )
        )
        emit_review_progress(
            progress_path,
            phase="clip_measurement_complete",
            detail=f"Finished measuring {clip_id}.",
            stage_label="Measuring clips",
            clip_index=clip_index,
            clip_count=len(clips),
            current_clip_id=clip_id,
            elapsed_seconds=time.perf_counter() - analysis_started_at,
            extra={"clip_elapsed_seconds": time.perf_counter() - clip_started_at},
        )

    should_use_array_batch = calibration_mode == "array-gray-sphere" or (
        Path(input_path).expanduser().resolve().is_dir()
        and calibration_path is None
        and exposure_calibration_path is None
        and color_calibration_path is None
    )
    shared_batch_group = derive_array_group_key(input_path) if should_use_array_batch else None
    if shared_batch_group is not None:
        for item in provisional:
            item.group_key = shared_batch_group
            item.clip_metadata.group_key = shared_batch_group

    resolved_exposure_path = exposure_calibration_path or calibration_path
    loaded_exposure_calibration = load_exposure_calibration(resolved_exposure_path) if resolved_exposure_path else None
    loaded_color_calibration = load_color_calibration(color_calibration_path) if color_calibration_path else None
    if shared_batch_group is None and Path(input_path).expanduser().resolve().is_dir():
        calibration_group_keys = set()
        if loaded_exposure_calibration:
            calibration_group_keys.update(entry.group_key for entry in loaded_exposure_calibration.cameras)
        if loaded_color_calibration:
            calibration_group_keys.update(entry.group_key for entry in loaded_color_calibration.cameras)
        if len(calibration_group_keys) == 1:
            forced_group_key = next(iter(calibration_group_keys))
            for item in provisional:
                item.group_key = forced_group_key
                item.clip_metadata.group_key = forced_group_key

    by_group: Dict[str, List[float]] = {}
    for item in provisional:
        by_group.setdefault(item.group_key, []).append(item.raw_offset_stops)
    group_baselines = {group_key: median(values) for group_key, values in by_group.items()}

    calibrated_baselines = calibration_baselines(loaded_exposure_calibration) if loaded_exposure_calibration else {}
    applied_color_gains = color_calibration_gains(loaded_color_calibration) if loaded_color_calibration else {}
    array_calibration = None
    if shared_batch_group is not None:
        array_calibration = build_array_calibration_from_analysis(
            provisional,
            input_path=input_path,
            group_key=shared_batch_group,
        )
        write_array_calibration_json(array_calibration, out_root / "array_calibration.json")
        calibrated_baselines = {entry.clip_id: entry.solution.exposure_offset_stops for entry in array_calibration.cameras}
        applied_color_gains = {
            entry.clip_id: {
                "r": entry.solution.rgb_gains[0],
                "g": entry.solution.rgb_gains[1],
                "b": entry.solution.rgb_gains[2],
            }
            for entry in array_calibration.cameras
        }

    final_results: list[ClipResult] = []
    emit_review_progress(
        progress_path,
        phase="analysis_finalize_start",
        detail="Writing analysis outputs.",
        stage_label="Writing analysis",
        clip_count=len(provisional),
        elapsed_seconds=time.perf_counter() - analysis_started_at,
    )
    for item in provisional:
        raise_if_cancelled("Run cancelled while writing analysis outputs.")
        derived_group_baseline = group_baselines[item.group_key]
        clip_trim = 0.0
        applied_baseline = calibrated_baselines.get(item.clip_id, calibrated_baselines.get(item.group_key, derived_group_baseline))
        final_offset = max(-clamp_stops, min(clamp_stops, applied_baseline))
        confidence = compute_confidence(item.frame_stats, unclamped_offset=item.raw_offset_stops, clamp_stops=clamp_stops)
        exposure_provenance = (
            {
                "source": str(Path(resolved_exposure_path).expanduser().resolve()),
                "object_type": loaded_exposure_calibration.object_type,
                "calibration_mode": loaded_exposure_calibration.calibration_mode,
                "calibration_type": loaded_exposure_calibration.calibration_type,
                "target_log2_luminance": loaded_exposure_calibration.target_log2_luminance,
                "applied_camera_baseline_stops": applied_baseline,
                "derived_group_baseline_stops": derived_group_baseline,
            }
            if loaded_exposure_calibration and (item.group_key in calibrated_baselines or item.clip_id in calibrated_baselines)
            else None
        )
        color_provenance = (
            {
                "source": str(Path(color_calibration_path).expanduser().resolve()),
                "object_type": loaded_color_calibration.object_type,
                "calibration_mode": loaded_color_calibration.calibration_mode,
                "calibration_type": loaded_color_calibration.calibration_type,
                "pending_rgb_neutral_gains": applied_color_gains.get(item.clip_id, applied_color_gains.get(item.group_key)),
            }
            if loaded_color_calibration and (item.group_key in applied_color_gains or item.clip_id in applied_color_gains)
            else None
        )
        array_entry = next((entry for entry in array_calibration.cameras if entry.clip_id == item.clip_id), None) if array_calibration else None
        if array_entry is not None:
            exposure_provenance = {
                "source": str((out_root / "array_calibration.json").resolve()),
                "calibration_type": "array_exposure",
                "group_key": array_calibration.group_key,
                "applied_camera_baseline_stops": array_entry.solution.exposure_offset_stops,
            }
            color_provenance = {
                "source": str((out_root / "array_calibration.json").resolve()),
                "calibration_type": "array_color",
                "group_key": array_calibration.group_key,
                "pending_rgb_neutral_gains": array_entry.solution.rgb_gains,
            }
        resolved_color_gains = (
            {"r": array_entry.solution.rgb_gains[0], "g": array_entry.solution.rgb_gains[1], "b": array_entry.solution.rgb_gains[2]}
            if array_entry is not None
            else applied_color_gains.get(item.clip_id, applied_color_gains.get(item.group_key))
        )
        result = ClipResult(
            clip_id=item.clip_id,
            group_key=item.group_key,
            source_path=item.source_path,
            backend=item.backend,
            clip_statistic_log2=item.clip_statistic_log2,
            group_key_statistic_log2=item.group_key_statistic_log2,
            global_reference_log2=item.global_reference_log2,
            raw_offset_stops=item.raw_offset_stops,
            camera_baseline_stops=applied_baseline,
            clip_trim_stops=clip_trim,
            final_offset_stops=final_offset,
            confidence=confidence,
            sample_plan=item.sample_plan,
            monitoring=item.monitoring,
            clip_metadata=item.clip_metadata,
            frame_stats=item.frame_stats,
            calibration_provenance=exposure_provenance,
            exposure_calibration_provenance=exposure_provenance,
            color_calibration_provenance=color_provenance,
            pending_color_gains=resolved_color_gains,
            exposure_calibration_loaded=exposure_provenance is not None or array_entry is not None,
            exposure_baseline_applied_stops=applied_baseline if (exposure_provenance is not None or array_entry is not None) else None,
            color_calibration_loaded=color_provenance is not None or array_entry is not None,
            color_gains_state="pending" if (color_provenance is not None or array_entry is not None) else None,
            diagnostics={
                **item.diagnostics,
                "pending_color_gains": resolved_color_gains,
                "runtime_trace": {
                    **dict((item.diagnostics or {}).get("runtime_trace") or {}),
                    "invocation_source": invocation_source,
                },
                "workload_trace": {
                    **dict((item.diagnostics or {}).get("workload_trace") or {}),
                    "invocation_source": invocation_source,
                },
            },
        )
        final_results.append(result)
        write_json(analysis_dir / f"{result.clip_id}.analysis.json", result.to_dict())
        write_sidecar_file(sidecar_dir, result)

    summary = {
        "input_path": str(Path(input_path).expanduser().resolve()),
        "mode": mode,
        "backend": backend,
        "clip_count": len(final_results),
        "analysis_dir": str(analysis_dir),
        "sidecar_dir": str(sidecar_dir),
        "group_baselines": group_baselines,
        "array_calibration": (
            {
                "path": str((out_root / "array_calibration.json").resolve()),
                "group_key": array_calibration.group_key,
                "camera_count": len(array_calibration.cameras),
                "target_exposure_log2": array_calibration.target.exposure.log2_luminance_target,
                "target_rgb_chromaticity": array_calibration.target.color.target_rgb_chromaticity,
            }
            if array_calibration
            else None
        ),
        "exposure_calibration": (
            {
                "path": str(Path(resolved_exposure_path).expanduser().resolve()),
                "object_type": loaded_exposure_calibration.object_type,
                "calibration_mode": loaded_exposure_calibration.calibration_mode,
                "calibration_type": loaded_exposure_calibration.calibration_type,
                "target_log2_luminance": loaded_exposure_calibration.target_log2_luminance,
                "reference_camera": getattr(loaded_exposure_calibration, "reference_camera", None),
                "group_baselines": calibrated_baselines,
            }
            if loaded_exposure_calibration
            else None
        ),
        "color_calibration": (
            {
                "path": str(Path(color_calibration_path).expanduser().resolve()),
                "object_type": loaded_color_calibration.object_type,
                "calibration_mode": loaded_color_calibration.calibration_mode,
                "calibration_type": loaded_color_calibration.calibration_type,
                "group_gains": applied_color_gains,
            }
            if loaded_color_calibration
            else None
        ),
        "calibration": (
            {
                "path": str(Path(resolved_exposure_path).expanduser().resolve()),
                "object_type": loaded_exposure_calibration.object_type,
                "calibration_mode": loaded_exposure_calibration.calibration_mode,
                "target_log2_luminance": loaded_exposure_calibration.target_log2_luminance,
                "reference_camera": getattr(loaded_exposure_calibration, "reference_camera", None),
                "group_baselines": calibrated_baselines,
            }
            if loaded_exposure_calibration
            else None
        ),
        "clips": [
            {
                "clip_id": item.clip_id,
                "group_key": item.group_key,
                "raw_offset_stops": item.raw_offset_stops,
                "camera_baseline_stops": item.camera_baseline_stops,
                "clip_trim_stops": item.clip_trim_stops,
                "final_offset_stops": item.final_offset_stops,
                "confidence": item.confidence,
                "pending_color_gains": item.pending_color_gains,
                "exposure_calibration_loaded": item.exposure_calibration_loaded,
                "exposure_baseline_applied_stops": item.exposure_baseline_applied_stops,
                "color_calibration_loaded": item.color_calibration_loaded,
                "color_gains_state": item.color_gains_state,
            }
            for item in final_results
        ],
        "calibration_roi": calibration_roi,
        "selected_clip_ids": sorted(requested_clip_ids) if requested_clip_ids else None,
        "selected_clip_groups": sorted(requested_clip_groups) if requested_clip_groups else None,
    }
    runtime_rows = []
    if workload_trace_path or runtime_trace_path:
        workload_rows = []
        for item in final_results:
            workload = dict((item.diagnostics or {}).get("workload_trace") or {})
            runtime = dict((item.diagnostics or {}).get("runtime_trace") or {})
            workload_rows.append(
                {
                    "clip_id": item.clip_id,
                    "group_key": item.group_key,
                    "representative_frame_index": int(workload.get("representative_frame_index", item.sample_plan.start_frame) or 0),
                    "frames_analyzed": int(workload.get("frames_analyzed", len(item.frame_stats)) or 0),
                    "decode_width": int(workload.get("decode_width", 0) or 0),
                    "decode_height": int(workload.get("decode_height", 0) or 0),
                    "decode_half_res": bool(workload.get("decode_half_res", half_res_decode)),
                    "measurement_domain": str(workload.get("measurement_domain") or "scene_sdk_decode_with_proxy_monitoring"),
                    "detection_count": int(workload.get("detection_count", 0) or 0),
                    "gradient_axis_count": int(workload.get("gradient_axis_count", 0) or 0),
                    "region_stat_count": int(workload.get("region_stat_count", 0) or 0),
                    "strategy_reuse": bool(workload.get("strategy_reuse", True)),
                    "total_measurement_time_seconds": float(workload.get("total_measurement_time_seconds", 0.0) or 0.0),
                }
            )
            runtime_rows.append(
                {
                    "clip_id": item.clip_id,
                    "group_key": item.group_key,
                    "invocation_source": str(runtime.get("invocation_source") or invocation_source),
                    "backend": str(runtime.get("backend") or item.backend),
                    "measurement_domain": str(runtime.get("measurement_domain") or workload.get("measurement_domain") or "scene_sdk_decode_with_proxy_monitoring"),
                    "measurement_source": str(runtime.get("measurement_source") or "scene_sdk_decode_with_proxy_monitoring"),
                    "representative_frame_index": int(runtime.get("representative_frame_index", item.sample_plan.start_frame) or 0),
                    "frames_analyzed": int(runtime.get("frames_analyzed", len(item.frame_stats)) or 0),
                    "decode_width": int(runtime.get("decode_width", workload.get("decode_width", 0)) or 0),
                    "decode_height": int(runtime.get("decode_height", workload.get("decode_height", 0)) or 0),
                    "decode_half_res": bool(runtime.get("decode_half_res", workload.get("decode_half_res", half_res_decode))),
                    "detection_count": int(runtime.get("detection_count", workload.get("detection_count", 0)) or 0),
                    "gradient_axis_count": int(runtime.get("gradient_axis_count", workload.get("gradient_axis_count", 0)) or 0),
                    "region_stat_count": int(runtime.get("region_stat_count", workload.get("region_stat_count", 0)) or 0),
                    "strategy_reuse": bool(runtime.get("strategy_reuse", workload.get("strategy_reuse", True))),
                    "durations_seconds": dict(runtime.get("durations_seconds") or {}),
                    "frame_runtime": list(runtime.get("frame_runtime") or []),
                    "total_measurement_time_seconds": float(runtime.get("total_measurement_time_seconds", workload.get("total_measurement_time_seconds", 0.0)) or 0.0),
                    "rendered_image_path": str(
                        runtime.get("rendered_image_path")
                        or workload.get("rendered_image_path")
                        or ""
                    ),
                }
            )
    if workload_trace_path:
        write_json(
            Path(workload_trace_path).expanduser().resolve(),
            {
                "measurement_domain": "scene_sdk_decode_with_proxy_monitoring",
                "measurement_source": str(measurement_source or "scene_sdk_decode_with_proxy_monitoring"),
                "half_res_decode_enabled": bool(half_res_decode),
                "invocation_source": invocation_source,
                "clip_count": len(workload_rows),
                "clips": workload_rows,
            },
        )
    if runtime_trace_path:
        write_json(
            Path(runtime_trace_path).expanduser().resolve(),
            {
                "measurement_domain": str(measurement_source or "scene_sdk_decode_with_proxy_monitoring"),
                "measurement_source": str(measurement_source or "scene_sdk_decode_with_proxy_monitoring"),
                "half_res_decode_enabled": bool(half_res_decode),
                "invocation_source": invocation_source,
                "clip_count": len(runtime_rows),
                "clips": runtime_rows,
            },
        )
    write_json(out_root / "summary.json", summary)
    emit_review_progress(
        progress_path,
        phase="analysis_complete",
        detail=f"Analysis complete for {len(final_results)} clip(s).",
        stage_label="Analysis complete",
        clip_count=len(final_results),
        elapsed_seconds=time.perf_counter() - analysis_started_at,
    )
    return summary


def analyze_clip(
    source_path: str,
    *,
    mode: str,
    backend: str,
    lut_override: Optional[str],
    sample_count: int,
    sampling_strategy: str,
    calibration_roi: Optional[Dict[str, float]] = None,
    target_type: Optional[str] = None,
    half_res_decode: bool = False,
    measurement_source: str = "scene_sdk_decode_with_proxy_monitoring",
    measurement_output_dir: Optional[str] = None,
    rendered_preview_context: Optional[Dict[str, object]] = None,
) -> ClipResult:
    raise_if_cancelled("Run cancelled before clip decode.")
    clip_started_at = time.perf_counter()
    runtime_trace: Dict[str, object] = {
        "measurement_domain": str(measurement_source or "scene_sdk_decode_with_proxy_monitoring"),
        "measurement_source": str(measurement_source or "scene_sdk_decode_with_proxy_monitoring"),
        "frame_runtime": [],
    }
    function_durations: Dict[str, float] = {}
    phase_started_at = time.perf_counter()
    backend_impl = resolve_backend(backend)
    function_durations["resolve_backend_seconds"] = time.perf_counter() - phase_started_at
    phase_started_at = time.perf_counter()
    clip = backend_impl.inspect_clip(source_path)
    function_durations["inspect_clip_seconds"] = time.perf_counter() - phase_started_at
    phase_started_at = time.perf_counter()
    sample_plan = build_sample_plan(clip.total_frames, sample_count=sample_count, strategy=sampling_strategy)
    function_durations["build_sample_plan_seconds"] = time.perf_counter() - phase_started_at
    monitoring = MonitoringContext(
        mode=mode,
        ipp2_color_space=clip.color_space,
        ipp2_gamma_curve=clip.gamma_curve,
        active_lut_path=clip.active_lut_path,
        lut_override_path=str(Path(lut_override).expanduser().resolve()) if lut_override else None,
        resolved_lut_path=None,
    )
    monitoring.resolved_lut_path = monitoring.lut_override_path or monitoring.active_lut_path
    lut = resolve_lut(monitoring.resolved_lut_path) if mode == "view" else None

    if str(measurement_source or "").strip() == "rendered_preview_ipp2":
        return _analyze_clip_via_rendered_preview(
            clip=clip,
            sample_plan=sample_plan,
            source_path=source_path,
            backend_name=backend_impl.name,
            global_reference=math.log2(0.18),
            calibration_roi=calibration_roi,
            target_type=target_type,
            measurement_output_dir=measurement_output_dir,
            rendered_preview_context=rendered_preview_context or _resolve_rendered_preview_context(),
            clip_started_at=clip_started_at,
            function_durations=function_durations,
        )

    frame_stats: list[FrameStat] = []
    accepted_values: list[float] = []
    measured_log_values: list[float] = []
    measured_log_values_raw: list[float] = []
    measured_rgb_means: list[list[float]] = []
    measured_rgb_chroma: list[list[float]] = []
    valid_pixel_counts: list[int] = []
    saturation_fractions: list[float] = []
    black_fractions: list[float] = []
    neutral_sample_spreads: list[float] = []
    neutral_sample_chroma_spreads: list[float] = []
    representative_neutral_samples: list[dict[str, object]] = []
    sphere_sampling_measurements: list[dict[str, object]] = []
    sphere_sampling_confidences: list[float] = []
    decode_width = 0
    decode_height = 0
    detection_count = 0
    gradient_axis_count = 0
    region_stat_count = 0
    phase_started_at = time.perf_counter()
    decode_wait_started_at = phase_started_at
    for frame_index, timestamp_seconds, image in backend_impl.decode_frames(
        source_path,
        start_frame=sample_plan.start_frame,
        max_frames=sample_plan.max_frames,
        frame_step=sample_plan.frame_step,
        half_res=half_res_decode,
    ):
        raise_if_cancelled("Run cancelled during frame measurement.")
        frame_decode_seconds = time.perf_counter() - decode_wait_started_at
        _, decode_height, decode_width = image.shape
        analyze_started_at = time.perf_counter()
        stat = analyze_frame(frame_index, timestamp_seconds, image, mode=mode, lut=lut)
        analyze_seconds = time.perf_counter() - analyze_started_at
        frame_stats.append(stat)
        measurement_started_at = time.perf_counter()
        measurement = measure_frame_color_and_exposure(
            image,
            mode=mode,
            lut=lut,
            calibration_roi=calibration_roi,
            target_type=target_type,
        )
        measurement_seconds = time.perf_counter() - measurement_started_at
        workload = dict(measurement.get("workload_trace") or {})
        detection_count += int(workload.get("detection_count", 0) or 0)
        gradient_axis_count += int(workload.get("gradient_axis_count", 0) or 0)
        region_stat_count += int(workload.get("region_stat_count", 0) or 0)
        runtime_trace["frame_runtime"].append(
            {
                "frame_index": int(frame_index),
                "timestamp_seconds": float(timestamp_seconds),
                "decode_wait_seconds": float(frame_decode_seconds),
                "analyze_frame_seconds": float(analyze_seconds),
                "measure_frame_seconds": float(measurement_seconds),
                "measurement_runtime": dict(measurement.get("measurement_runtime") or {}),
            }
        )
        measured_log_values.append(measurement["measured_log2_luminance"])
        measured_log_values_raw.append(measurement["measured_log2_luminance_raw"])
        measured_rgb_means.append(measurement["measured_rgb_mean"])
        measured_rgb_chroma.append(measurement["measured_rgb_chromaticity"])
        valid_pixel_counts.append(measurement["valid_pixel_count"])
        saturation_fractions.append(measurement["saturation_fraction"])
        black_fractions.append(measurement["black_fraction"])
        neutral_sample_spreads.append(float(measurement.get("neutral_sample_log2_spread", 0.0) or 0.0))
        neutral_sample_chroma_spreads.append(float(measurement.get("neutral_sample_chromaticity_spread", 0.0) or 0.0))
        if not representative_neutral_samples and measurement.get("neutral_samples"):
            representative_neutral_samples = list(measurement["neutral_samples"])
        if measurement.get("sphere_sampling_comparison"):
            sphere_sampling_measurements.append(dict(measurement["sphere_sampling_comparison"]))
            sphere_sampling_confidences.append(float(measurement.get("gray_sphere_sampling_confidence", 0.0) or 0.0))
        if stat.accepted:
            accepted_values.append(stat.log_luminance_median)
        decode_wait_started_at = time.perf_counter()
    function_durations["decode_and_measure_seconds"] = time.perf_counter() - phase_started_at
    if not accepted_values:
        accepted_values = [item.log_luminance_median for item in frame_stats]

    clip_statistic = median(accepted_values)
    global_reference = math.log2(0.18)
    raw_offset = global_reference - clip_statistic
    diagnostics = {
        "sampled_frames": len(frame_stats),
        "accepted_frames": sum(1 for item in frame_stats if item.accepted),
        "measured_log2_luminance": median(measured_log_values),
        "measured_log2_luminance_monitoring": median(measured_log_values),
        "measured_log2_luminance_raw": median(measured_log_values_raw),
        "measured_rgb_mean": np.median(np.asarray(measured_rgb_means, dtype=np.float32), axis=0).tolist() if measured_rgb_means else [0.0, 0.0, 0.0],
        "measured_rgb_chromaticity": np.median(np.asarray(measured_rgb_chroma, dtype=np.float32), axis=0).tolist() if measured_rgb_chroma else [1 / 3, 1 / 3, 1 / 3],
        "valid_pixel_count": int(np.median(valid_pixel_counts)) if valid_pixel_counts else 0,
        "saturation_fraction": float(np.median(saturation_fractions)) if saturation_fractions else 0.0,
        "black_fraction": float(np.median(black_fractions)) if black_fractions else 0.0,
        "neutral_sample_count": 3,
        "neutral_sample_log2_spread": float(np.median(neutral_sample_spreads)) if neutral_sample_spreads else 0.0,
        "neutral_sample_chromaticity_spread": float(np.median(neutral_sample_chroma_spreads)) if neutral_sample_chroma_spreads else 0.0,
        "neutral_samples": representative_neutral_samples,
        "calibration_roi": calibration_roi,
        "calibration_measurement_mode": "shared_roi" if calibration_roi is not None else "center_region_fallback",
        "exposure_measurement_domain": "scene_sdk_decode_with_proxy_monitoring",
        "monitoring_preview_transform": "Local monitoring proxy over SDK decode (not REDLine IPP2 render)",
        "workload_trace": {
            "representative_frame_index": int(sample_plan.start_frame),
            "frames_analyzed": len(frame_stats),
            "decode_width": int(decode_width),
            "decode_height": int(decode_height),
            "decode_half_res": bool(half_res_decode),
            "measurement_domain": "scene_sdk_decode_with_proxy_monitoring",
            "detection_count": int(detection_count),
            "gradient_axis_count": int(gradient_axis_count),
            "region_stat_count": int(region_stat_count),
            "strategy_reuse": True,
            "total_measurement_time_seconds": time.perf_counter() - clip_started_at,
        },
        "runtime_trace": {
            "invocation_source": "",
            "backend": backend_impl.name,
            "measurement_domain": "scene_sdk_decode_with_proxy_monitoring",
            "measurement_source": "scene_sdk_decode_with_proxy_monitoring",
            "representative_frame_index": int(sample_plan.start_frame),
            "frames_analyzed": len(frame_stats),
            "decode_width": int(decode_width),
            "decode_height": int(decode_height),
            "decode_half_res": bool(half_res_decode),
            "detection_count": int(detection_count),
            "gradient_axis_count": int(gradient_axis_count),
            "region_stat_count": int(region_stat_count),
            "strategy_reuse": True,
            "durations_seconds": function_durations,
            "frame_runtime": list(runtime_trace["frame_runtime"]),
            "total_measurement_time_seconds": time.perf_counter() - clip_started_at,
        },
    }
    if sphere_sampling_measurements:
        diagnostics.update(
            {
                "sphere_sampling_comparison": _aggregate_sphere_sampling_comparisons(sphere_sampling_measurements),
                "gray_sphere_sampling_confidence": float(np.median(np.asarray(sphere_sampling_confidences, dtype=np.float32))) if sphere_sampling_confidences else 0.0,
                "calibration_measurement_mode": str(measurement.get("calibration_measurement_mode") or diagnostics["calibration_measurement_mode"]),
            }
        )
    return ClipResult(
        clip_id=clip.clip_id,
        group_key=clip.group_key,
        source_path=clip.source_path,
        backend=backend_impl.name,
        clip_statistic_log2=clip_statistic,
        group_key_statistic_log2=clip_statistic,
        global_reference_log2=global_reference,
        raw_offset_stops=raw_offset,
        camera_baseline_stops=0.0,
        clip_trim_stops=raw_offset,
        final_offset_stops=raw_offset,
        confidence=0.0,
        sample_plan=sample_plan,
        monitoring=monitoring,
        clip_metadata=clip,
        frame_stats=frame_stats,
        diagnostics=diagnostics,
    )


def _resolve_rendered_preview_context() -> Dict[str, object]:
    from .report import (
        _detect_redline_capabilities,
        _measurement_preview_settings_for_domain,
        _resolve_redline_executable,
    )

    redline_executable = _resolve_redline_executable()
    return {
        "redline_executable": redline_executable,
        "redline_capabilities": _detect_redline_capabilities(redline_executable),
        "preview_settings": _measurement_preview_settings_for_domain("perceptual"),
    }


def _analyze_clip_via_rendered_preview(
    *,
    clip,
    sample_plan: SamplePlan,
    source_path: str,
    backend_name: str,
    global_reference: float,
    calibration_roi: Optional[Dict[str, float]],
    target_type: Optional[str],
    measurement_output_dir: Optional[str],
    rendered_preview_context: Dict[str, object],
    clip_started_at: float,
    function_durations: Dict[str, float],
) -> ClipResult:
    from .report import _measure_rendered_preview_roi_ipp2, _preview_transform_label, render_preview_frame

    if measurement_output_dir is None:
        raise RuntimeError("Rendered-preview measurement requires a measurement output directory.")
    measurement_root = Path(measurement_output_dir).expanduser().resolve() / "previews" / "_measurement"
    measurement_root.mkdir(parents=True, exist_ok=True)
    preview_path = measurement_root / f"{clip.clip_id}.original.analysis.measurement.jpg"

    render_started_at = time.perf_counter()
    render = render_preview_frame(
        source_path,
        str(preview_path),
        frame_index=int(sample_plan.start_frame),
        redline_executable=str(rendered_preview_context["redline_executable"]),
        redline_capabilities=dict(rendered_preview_context["redline_capabilities"]),
        preview_settings=dict(rendered_preview_context["preview_settings"]),
        use_as_shot_metadata=True,
    )
    render_seconds = time.perf_counter() - render_started_at
    function_durations["render_preview_seconds"] = float(render_seconds)
    if int(render["returncode"]) != 0:
        raise RuntimeError(
            f"REDLine preview render failed for {clip.clip_id} during lightweight measurement. "
            f"Command: {' '.join(render['command'])}. STDERR: {str(render['stderr']).strip()}"
        )
    actual_preview_path = Path(str(render["output_path"])).expanduser().resolve()
    if not actual_preview_path.exists():
        raise RuntimeError(
            f"REDLine preview render did not create expected file for lightweight measurement: {actual_preview_path}"
        )

    measure_started_at = time.perf_counter()
    measured = _measure_rendered_preview_roi_ipp2(
        str(actual_preview_path),
        calibration_roi if str(target_type or "").strip().lower().replace("-", "_") == "gray_sphere" else calibration_roi,
    )
    measure_seconds = time.perf_counter() - measure_started_at
    function_durations["measure_rendered_preview_seconds"] = float(measure_seconds)

    measured_log2 = float(
        measured.get(
            "measured_log2_luminance_monitoring",
            measured.get("measured_log2_luminance", 0.0),
        )
        or 0.0
    )
    raw_offset = float(global_reference - measured_log2)
    measurement_runtime = dict(measured.get("measurement_runtime") or {})
    frame_runtime = {
        "frame_index": int(sample_plan.start_frame),
        "timestamp_seconds": float(sample_plan.start_frame / max(float(clip.fps or 24.0), 1e-6)),
        "render_preview_seconds": float(render_seconds),
        "measure_frame_seconds": float(measure_seconds),
        "measurement_runtime": measurement_runtime,
        "rendered_image_path": str(actual_preview_path),
    }
    diagnostics = {
        "sampled_frames": 1,
        "accepted_frames": 1,
        "measured_log2_luminance": measured_log2,
        "measured_log2_luminance_monitoring": measured_log2,
        "measured_log2_luminance_raw": measured_log2,
        "measured_rgb_mean": [float(value) for value in measured.get("measured_rgb_mean", [0.0, 0.0, 0.0])],
        "measured_rgb_chromaticity": [float(value) for value in measured.get("measured_rgb_chromaticity", [1 / 3, 1 / 3, 1 / 3])],
        "valid_pixel_count": int(measured.get("valid_pixel_count", 0) or 0),
        "saturation_fraction": float(measured.get("measured_saturation_fraction_monitoring", 0.0) or 0.0),
        "black_fraction": 0.0,
        "neutral_sample_count": int(measured.get("neutral_sample_count", 0) or 0),
        "neutral_sample_log2_spread": float(measured.get("neutral_sample_log2_spread", 0.0) or 0.0),
        "neutral_sample_chromaticity_spread": float(measured.get("neutral_sample_chromaticity_spread", 0.0) or 0.0),
        "neutral_samples": [dict(item) for item in list(measured.get("neutral_samples") or [])],
        "calibration_roi": calibration_roi,
        "calibration_measurement_mode": str(
            "gray_sphere_three_zone_profile" if str(target_type or "").strip().lower().replace("-", "_") == "gray_sphere" else "rendered_preview_sampling"
        ),
        "exposure_measurement_domain": "rendered_preview_ipp2",
        "monitoring_preview_transform": _preview_transform_label(dict(rendered_preview_context["preview_settings"])),
        "rendered_measurement_preview_path": str(actual_preview_path),
        "rendered_measurement_command": list(render.get("command") or []),
        "rendered_measurement_returncode": int(render.get("returncode", 0) or 0),
        "rendered_measurement_stdout": str(render.get("stdout") or ""),
        "rendered_measurement_stderr": str(render.get("stderr") or ""),
        "rendered_measurement_source": "rendered_preview_ipp2",
        "rendered_measurement_domain": "rendered_preview_ipp2",
        "measurement_render_resolution": {
            "width": int(measured.get("render_width", 0) or 0),
            "height": int(measured.get("render_height", 0) or 0),
        },
        "workload_trace": {
            "representative_frame_index": int(sample_plan.start_frame),
            "frames_analyzed": 1,
            "decode_width": int(measured.get("render_width", 0) or 0),
            "decode_height": int(measured.get("render_height", 0) or 0),
            "decode_half_res": False,
            "measurement_domain": "rendered_preview_ipp2",
            "measurement_source": "rendered_preview_ipp2",
            "detection_count": 1,
            "gradient_axis_count": 1 if list(measured.get("zone_measurements") or []) else 0,
            "region_stat_count": 1,
            "strategy_reuse": True,
            "render_time_seconds": float(render_seconds),
            "detection_time_seconds": float(measurement_runtime.get("sphere_detection_seconds", 0.0) or 0.0),
            "gradient_time_seconds": float(measurement_runtime.get("gradient_axis_seconds", 0.0) or 0.0),
            "stat_time_seconds": float(measurement_runtime.get("zone_stat_seconds", 0.0) or 0.0),
            "total_measurement_time_seconds": float(time.perf_counter() - clip_started_at),
            "rendered_image_path": str(actual_preview_path),
        },
        "runtime_trace": {
            "invocation_source": "",
            "backend": backend_name,
            "measurement_domain": "rendered_preview_ipp2",
            "measurement_source": "rendered_preview_ipp2",
            "representative_frame_index": int(sample_plan.start_frame),
            "frames_analyzed": 1,
            "decode_width": int(measured.get("render_width", 0) or 0),
            "decode_height": int(measured.get("render_height", 0) or 0),
            "decode_half_res": False,
            "detection_count": 1,
            "gradient_axis_count": 1 if list(measured.get("zone_measurements") or []) else 0,
            "region_stat_count": 1,
            "strategy_reuse": True,
            "durations_seconds": dict(function_durations),
            "frame_runtime": [frame_runtime],
            "total_measurement_time_seconds": float(time.perf_counter() - clip_started_at),
            "render_time_seconds": float(render_seconds),
            "detection_time_seconds": float(measurement_runtime.get("sphere_detection_seconds", 0.0) or 0.0),
            "gradient_time_seconds": float(measurement_runtime.get("gradient_axis_seconds", 0.0) or 0.0),
            "stat_time_seconds": float(measurement_runtime.get("zone_stat_seconds", 0.0) or 0.0),
            "rendered_image_path": str(actual_preview_path),
        },
    }
    diagnostics.update(
        {
            "gray_exposure_summary": str(measured.get("gray_exposure_summary") or "n/a"),
            "bright_ire": float(measured.get("bright_ire", 0.0) or 0.0),
            "center_ire": float(measured.get("center_ire", 0.0) or 0.0),
            "dark_ire": float(measured.get("dark_ire", 0.0) or 0.0),
            "sample_1_ire": float(measured.get("sample_1_ire", measured.get("bright_ire", 0.0)) or 0.0),
            "sample_2_ire": float(measured.get("sample_2_ire", measured.get("center_ire", 0.0)) or 0.0),
            "sample_3_ire": float(measured.get("sample_3_ire", measured.get("dark_ire", 0.0)) or 0.0),
            "top_ire": float(measured.get("top_ire", 0.0) or 0.0),
            "mid_ire": float(measured.get("mid_ire", 0.0) or 0.0),
            "bottom_ire": float(measured.get("bottom_ire", 0.0) or 0.0),
            "zone_spread_ire": float(measured.get("zone_spread_ire", 0.0) or 0.0),
            "zone_spread_stops": float(measured.get("zone_spread_stops", 0.0) or 0.0),
            "zone_measurements": [dict(item) for item in list(measured.get("zone_measurements") or [])],
            "sphere_detection_confidence": float(measured.get("sphere_detection_confidence", 0.0) or 0.0),
            "sphere_detection_label": str(measured.get("sphere_detection_label") or ""),
            "sphere_detection_source": str(measured.get("sphere_roi_source") or ""),
            "sphere_detection_details": dict(measured.get("sphere_detection_details") or {}),
            "detected_sphere_roi": dict(measured.get("detected_sphere_roi") or {}),
            "measurement_crop_bounds": dict(measured.get("measurement_crop_bounds") or {}),
            "measurement_crop_size": dict(measured.get("measurement_crop_size") or {}),
            "detection_failed": bool(measured.get("detection_failed")),
            "dominant_gradient_axis": dict(measured.get("dominant_gradient_axis") or {}),
        }
    )
    frame_stat = FrameStat(
        frame_index=int(sample_plan.start_frame),
        timestamp_seconds=float(sample_plan.start_frame / max(float(clip.fps or 24.0), 1e-6)),
        log_luminance_median=measured_log2,
        clipped_fraction=0.0,
        valid_fraction=1.0,
        accepted=not bool(measured.get("detection_failed")),
        reason="sphere_detection_failed" if bool(measured.get("detection_failed")) else None,
    )
    return ClipResult(
        clip_id=clip.clip_id,
        group_key=clip.group_key,
        source_path=clip.source_path,
        backend=backend_name,
        clip_statistic_log2=measured_log2,
        group_key_statistic_log2=measured_log2,
        global_reference_log2=global_reference,
        raw_offset_stops=raw_offset,
        camera_baseline_stops=0.0,
        clip_trim_stops=raw_offset,
        final_offset_stops=raw_offset,
        confidence=float(measured.get("sphere_detection_confidence", 0.0) or 0.0),
        sample_plan=sample_plan,
        monitoring=MonitoringContext(
            mode="view",
            ipp2_color_space="BT.709",
            ipp2_gamma_curve="BT.1886",
            active_lut_path=None,
            lut_override_path=None,
            resolved_lut_path=None,
        ),
        clip_metadata=clip,
        frame_stats=[frame_stat],
        diagnostics=diagnostics,
    )


def analyze_frame(frame_index: int, timestamp_seconds: float, image: np.ndarray, *, mode: str, lut: Optional[CubeLut]) -> FrameStat:
    transformed = np.clip(image, 0.0, 1.0)
    if mode == "view":
        transformed = apply_lut(transformed, lut) if lut is not None else np.power(transformed, 1.0 / 2.4)
    transformed = extract_center_region(transformed, fraction=0.4)
    luminance = np.clip(transformed[0] * 0.2126 + transformed[1] * 0.7152 + transformed[2] * 0.0722, 1e-6, 1.0)
    clipped = (luminance <= 0.002) | (luminance >= 0.998)
    valid = ~clipped
    clipped_fraction = float(clipped.mean())
    valid_fraction = float(valid.mean())
    selected = luminance[valid if valid.any() else np.ones_like(valid, dtype=bool)]
    log_values = np.log2(percentile_clip(selected, 5.0, 95.0))
    accepted = valid_fraction >= 0.25 and clipped_fraction < 0.2
    return FrameStat(
        frame_index=frame_index,
        timestamp_seconds=timestamp_seconds,
        log_luminance_median=float(np.median(log_values)),
        clipped_fraction=clipped_fraction,
        valid_fraction=valid_fraction,
        accepted=accepted,
        reason=None if accepted else "frame_rejected_for_extremes",
    )


def _extract_normalized_roi_region(image: np.ndarray, calibration_roi: Dict[str, float]) -> np.ndarray:
    _, height, width = image.shape
    x0 = max(0, min(width - 1, int(np.floor(float(calibration_roi["x"]) * width))))
    y0 = max(0, min(height - 1, int(np.floor(float(calibration_roi["y"]) * height))))
    x1 = max(x0 + 1, min(width, int(np.ceil((float(calibration_roi["x"]) + float(calibration_roi["w"])) * width))))
    y1 = max(y0 + 1, min(height, int(np.ceil((float(calibration_roi["y"]) + float(calibration_roi["h"])) * height))))
    return image[:, y0:y1, x0:x1]


def _apply_monitoring_review_transform(image: np.ndarray) -> np.ndarray:
    transformed = np.clip(image, 0.0, 1.0).astype(np.float32, copy=False)
    # Shared monitoring-domain proxy for the REDLine IPP2 review path:
    # soft shoulder + medium contrast + Rec709/BT1886 style display encoding.
    shoulder = transformed / (transformed + 0.6)
    contrast = np.clip((shoulder - 0.18) * 1.15 + 0.18, 0.0, 1.0)
    return np.power(contrast, 1.0 / 2.4)


def _sample_distribution(values: np.ndarray) -> Dict[str, object]:
    array = np.asarray(values, dtype=np.float32).reshape(-1)
    if array.size == 0:
        return {
            "count": 0,
            "mean": 0.0,
            "median": 0.0,
            "stddev": 0.0,
            "minimum": 0.0,
            "maximum": 0.0,
            "p05": 0.0,
            "p95": 0.0,
            "preview_values": [],
        }
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "stddev": float(np.std(array)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
        "p05": float(np.percentile(array, 5.0)),
        "p95": float(np.percentile(array, 95.0)),
        "preview_values": [float(value) for value in array[: min(32, array.size)].tolist()],
    }


def _measure_region_statistics(region: np.ndarray) -> Dict[str, object]:
    luminance = np.clip(region[0] * 0.2126 + region[1] * 0.7152 + region[2] * 0.0722, 1e-6, 1.0)
    valid_mask = (luminance > 0.002) & (luminance < 0.998)
    pixels = np.moveaxis(region, 0, -1)[valid_mask]
    if pixels.size == 0:
        pixels = np.moveaxis(region, 0, -1).reshape(-1, 3)
        luminance_values = luminance.reshape(-1)
    else:
        luminance_values = luminance[valid_mask]
    trimmed_luma = percentile_clip(luminance_values, 5.0, 95.0)
    measured_log2 = float(np.median(np.log2(trimmed_luma)))
    rgb_mean = np.median(pixels, axis=0)
    chroma = rgb_mean / max(float(np.sum(rgb_mean)), 1e-6)
    saturation_fraction = float(np.mean(np.max(pixels, axis=1) >= 0.998)) if pixels.size else 0.0
    black_fraction = float(np.mean(np.min(pixels, axis=1) <= 0.002)) if pixels.size else 0.0
    roi_variance = float(np.var(trimmed_luma)) if trimmed_luma.size else 0.0
    log2_values = np.log2(trimmed_luma) if trimmed_luma.size else np.asarray([], dtype=np.float32)
    return {
        "measured_log2_luminance": measured_log2,
        "measured_rgb_mean": [float(rgb_mean[0]), float(rgb_mean[1]), float(rgb_mean[2])],
        "measured_rgb_chromaticity": [float(chroma[0]), float(chroma[1]), float(chroma[2])],
        "valid_pixel_count": int(pixels.shape[0]),
        "saturation_fraction": saturation_fraction,
        "black_fraction": black_fraction,
        "roi_variance": roi_variance,
        "gray_luminance_distribution": _sample_distribution(trimmed_luma),
        "gray_log2_distribution": _sample_distribution(log2_values),
    }


def _neutral_sample_regions(region: np.ndarray) -> list[tuple[str, Dict[str, object], np.ndarray]]:
    _, height, width = region.shape
    sample_width = max(1, int(round(width * 0.24)))
    sample_height = max(1, int(round(height * 0.28)))
    center_y = height * 0.5
    center_x_positions = [width * 0.28, width * 0.5, width * 0.72]
    labels = ["left", "center", "right"]
    samples: list[tuple[str, Dict[str, object], np.ndarray]] = []
    for label, center_x in zip(labels, center_x_positions):
        x0 = max(0, min(width - sample_width, int(round(center_x - sample_width / 2.0))))
        y0 = max(0, min(height - sample_height, int(round(center_y - sample_height / 2.0))))
        samples.append(
            (
                label,
                {
                    "pixel": {"x0": int(x0), "y0": int(y0), "x1": int(x0 + sample_width), "y1": int(y0 + sample_height)},
                    "normalized_within_roi": {
                        "x": float(x0) / float(max(width, 1)),
                        "y": float(y0) / float(max(height, 1)),
                        "w": float(sample_width) / float(max(width, 1)),
                        "h": float(sample_height) / float(max(height, 1)),
                    },
                },
                region[:, y0:y0 + sample_height, x0:x0 + sample_width],
            )
        )
    return samples


def _measure_three_sample_statistics(region: np.ndarray) -> Dict[str, object]:
    sample_measurements = []
    for label, sample_bounds, sample_region in _neutral_sample_regions(region):
        stats = _measure_region_statistics(sample_region)
        sample_measurements.append(
            {
                "label": label,
                "bounds": sample_bounds,
                "measured_log2_luminance": float(stats["measured_log2_luminance"]),
                "measured_rgb_mean": [float(value) for value in stats["measured_rgb_mean"]],
                "measured_rgb_chromaticity": [float(value) for value in stats["measured_rgb_chromaticity"]],
                "valid_pixel_count": int(stats["valid_pixel_count"]),
                "roi_variance": float(stats["roi_variance"]),
                "gray_luminance_distribution": dict(stats.get("gray_luminance_distribution") or {}),
                "gray_log2_distribution": dict(stats.get("gray_log2_distribution") or {}),
            }
        )
    log2_values = np.asarray([item["measured_log2_luminance"] for item in sample_measurements], dtype=np.float32)
    rgb_means = np.asarray([item["measured_rgb_mean"] for item in sample_measurements], dtype=np.float32)
    chroma = np.asarray([item["measured_rgb_chromaticity"] for item in sample_measurements], dtype=np.float32)
    rgb_mean = np.median(rgb_means, axis=0)
    chroma_mean = np.median(chroma, axis=0)
    return {
        "measured_log2_luminance": float(np.median(log2_values)),
        "measured_rgb_mean": [float(rgb_mean[0]), float(rgb_mean[1]), float(rgb_mean[2])],
        "measured_rgb_chromaticity": [float(chroma_mean[0]), float(chroma_mean[1]), float(chroma_mean[2])],
        "valid_pixel_count": int(np.median([item["valid_pixel_count"] for item in sample_measurements])) if sample_measurements else 0,
        "roi_variance": float(np.median([item["roi_variance"] for item in sample_measurements])) if sample_measurements else 0.0,
        "neutral_sample_count": len(sample_measurements),
        "neutral_sample_log2_spread": float(np.max(log2_values) - np.min(log2_values)) if sample_measurements else 0.0,
        "neutral_sample_chromaticity_spread": float(np.max(np.linalg.norm(chroma - chroma_mean, axis=1))) if sample_measurements else 0.0,
        "neutral_samples": sample_measurements,
    }


def _sphere_roi_for_region(region: np.ndarray) -> SphereROI:
    _, height, width = region.shape
    radius = max(min(width, height) * 0.46, 1.0)
    return SphereROI(
        cx=(float(width) - 1.0) / 2.0,
        cy=(float(height) - 1.0) / 2.0,
        r=float(radius),
    )


def _measure_gray_sphere_statistics(raw_region: np.ndarray, monitoring_region: np.ndarray) -> Dict[str, object]:
    sphere_roi = _sphere_roi_for_region(raw_region)
    legacy_raw = measure_sphere_region_statistics(raw_region, sphere_roi, sampling_variant="legacy")
    legacy_monitoring = measure_sphere_region_statistics(monitoring_region, sphere_roi, sampling_variant="legacy")
    raw_profile = measure_sphere_zone_profile_statistics(raw_region, sphere_roi, sampling_variant="refined")
    raw_axis = dict(raw_profile.get("dominant_gradient_axis") or {})
    raw_vector = list(raw_axis.get("vector") or [0.0, -1.0])
    monitoring_profile = measure_sphere_zone_profile_statistics(
        monitoring_region,
        sphere_roi,
        sampling_variant="refined",
        gradient_axis_override=(float(raw_vector[0]), float(raw_vector[1])),
    )
    return {
        "measured_log2_luminance": float(monitoring_profile["measured_log2_luminance"]),
        "measured_log2_luminance_monitoring": float(monitoring_profile["measured_log2_luminance"]),
        "measured_log2_luminance_raw": float(raw_profile["measured_log2_luminance"]),
        "measured_rgb_mean": [float(value) for value in raw_profile["measured_rgb_mean"]],
        "measured_rgb_chromaticity": [float(value) for value in raw_profile["measured_rgb_chromaticity"]],
        "valid_pixel_count": int(monitoring_profile["valid_pixel_count"]),
        "saturation_fraction": float(monitoring_profile["saturation_fraction"]),
        "black_fraction": float(monitoring_profile["black_fraction"]),
        "roi_variance": float(monitoring_profile["roi_variance"]),
        "monitoring_roi_variance": float(monitoring_profile["roi_variance"]),
        "raw_roi_variance": float(raw_profile["roi_variance"]),
        "neutral_sample_count": int(monitoring_profile["neutral_sample_count"]),
        "neutral_sample_log2_spread": float(monitoring_profile["neutral_sample_log2_spread"]),
        "neutral_sample_chromaticity_spread": float(raw_profile["neutral_sample_chromaticity_spread"]),
        "neutral_samples": [dict(item) for item in monitoring_profile["zone_measurements"]],
        "neutral_samples_raw": [dict(item) for item in raw_profile["zone_measurements"]],
        "gray_exposure_summary": str(monitoring_profile["aggregate_sphere_profile"]),
        "gray_exposure_summary_raw": str(raw_profile["aggregate_sphere_profile"]),
        "top_ire": float(monitoring_profile["top_ire"]),
        "mid_ire": float(monitoring_profile["mid_ire"]),
        "bottom_ire": float(monitoring_profile["bottom_ire"]),
        "zone_spread_ire": float(monitoring_profile["zone_spread_ire"]),
        "zone_spread_stops": float(monitoring_profile["zone_spread_stops"]),
        "sphere_zone_profile_monitoring": [dict(item) for item in monitoring_profile["zone_measurements"]],
        "sphere_zone_profile_raw": [dict(item) for item in raw_profile["zone_measurements"]],
        "raw_saturation_fraction": float(raw_profile["saturation_fraction"]),
        "sphere_sampling_comparison": {
            "measurement_geometry": "three_band_gradient_aligned_profile_within_refined_sphere_mask",
            "sphere_roi_within_roi": {"cx": sphere_roi.cx, "cy": sphere_roi.cy, "r": sphere_roi.r},
            "legacy": {
                "monitoring_log2": float(legacy_monitoring["measured_log2_luminance"]),
                "raw_log2": float(legacy_raw["measured_log2_luminance"]),
                "confidence": float(legacy_raw["confidence"]),
                "mask_fraction": float(legacy_raw["mask_fraction"]),
                "sampling_method": str(legacy_raw["sampling_method"]),
                "rgb_mean_raw": [float(value) for value in legacy_raw["measured_rgb_mean"]],
                "rgb_chromaticity_raw": [float(value) for value in legacy_raw["measured_rgb_chromaticity"]],
            },
            "refined": {
                "monitoring_log2": float(monitoring_profile["measured_log2_luminance"]),
                "raw_log2": float(raw_profile["measured_log2_luminance"]),
                "confidence": float(raw_profile["confidence"]),
                "mask_fraction": float(raw_profile["mask_fraction"]),
                "sampling_method": str(raw_profile["sampling_method"]),
                "rgb_mean_raw": [float(value) for value in raw_profile["measured_rgb_mean"]],
                "rgb_chromaticity_raw": [float(value) for value in raw_profile["measured_rgb_chromaticity"]],
                "top_ire": float(raw_profile["top_ire"]),
                "mid_ire": float(raw_profile["mid_ire"]),
                "bottom_ire": float(raw_profile["bottom_ire"]),
                "profile_summary": str(raw_profile["aggregate_sphere_profile"]),
            },
            "delta_raw_log2": float(raw_profile["measured_log2_luminance"] - legacy_raw["measured_log2_luminance"]),
            "delta_monitoring_log2": float(monitoring_profile["measured_log2_luminance"] - legacy_monitoring["measured_log2_luminance"]),
            "window_reference_raw_log2": float(raw_profile["measured_log2_luminance"]),
            "window_reference_monitoring_log2": float(monitoring_profile["measured_log2_luminance"]),
            "window_reference_profile": str(monitoring_profile["aggregate_sphere_profile"]),
        },
        "gray_sphere_sampling_confidence": float(raw_profile["confidence"]),
        "calibration_measurement_mode": "gray_sphere_three_zone_profile",
        "workload_trace": {
            "measurement_domain": "scene_sdk_decode_with_proxy_monitoring",
            "detection_count": 0,
            "gradient_axis_count": 1,
            "region_stat_count": 2,
        },
    }


def _aggregate_sampling_variant(values: List[Dict[str, object]]) -> Dict[str, object]:
    return {
        "monitoring_log2": float(np.median(np.asarray([float(item.get("monitoring_log2", 0.0) or 0.0) for item in values], dtype=np.float32))),
        "raw_log2": float(np.median(np.asarray([float(item.get("raw_log2", 0.0) or 0.0) for item in values], dtype=np.float32))),
        "confidence": float(np.median(np.asarray([float(item.get("confidence", 0.0) or 0.0) for item in values], dtype=np.float32))),
        "mask_fraction": float(np.median(np.asarray([float(item.get("mask_fraction", 0.0) or 0.0) for item in values], dtype=np.float32))),
        "sampling_method": str(values[0].get("sampling_method") or ""),
        "rgb_mean_raw": np.median(np.asarray([item.get("rgb_mean_raw", [0.0, 0.0, 0.0]) for item in values], dtype=np.float32), axis=0).tolist(),
        "rgb_chromaticity_raw": np.median(
            np.asarray([item.get("rgb_chromaticity_raw", [1 / 3, 1 / 3, 1 / 3]) for item in values], dtype=np.float32),
            axis=0,
        ).tolist(),
    }


def _aggregate_sphere_sampling_comparisons(values: List[Dict[str, object]]) -> Dict[str, object]:
    legacy = _aggregate_sampling_variant([dict(item.get("legacy") or {}) for item in values])
    refined = _aggregate_sampling_variant([dict(item.get("refined") or {}) for item in values])
    return {
        "measurement_geometry": str(values[0].get("measurement_geometry") or "inscribed_circle_with_refined_interior_mask"),
        "sphere_roi_within_roi": dict(values[0].get("sphere_roi_within_roi") or {}),
        "legacy": legacy,
        "refined": refined,
        "delta_raw_log2": float(refined["raw_log2"]) - float(legacy["raw_log2"]),
        "delta_monitoring_log2": float(refined["monitoring_log2"]) - float(legacy["monitoring_log2"]),
        "window_reference_raw_log2": float(np.median(np.asarray([float(item.get("window_reference_raw_log2", 0.0) or 0.0) for item in values], dtype=np.float32))),
        "window_reference_monitoring_log2": float(
            np.median(np.asarray([float(item.get("window_reference_monitoring_log2", 0.0) or 0.0) for item in values], dtype=np.float32))
        ),
    }


def measure_frame_color_and_exposure(
    image: np.ndarray,
    *,
    mode: str,
    lut: Optional[CubeLut],
    calibration_roi: Optional[Dict[str, float]] = None,
    target_type: Optional[str] = None,
) -> Dict[str, object]:
    measurement_started_at = time.perf_counter()
    phase_started_at = time.perf_counter()
    raw_region = np.clip(image, 0.0, 1.0)
    raw_region = _extract_normalized_roi_region(raw_region, calibration_roi) if calibration_roi is not None else extract_center_region(raw_region, fraction=0.4)
    raw_region_seconds = time.perf_counter() - phase_started_at
    phase_started_at = time.perf_counter()
    monitoring_region = np.clip(image, 0.0, 1.0)
    if mode == "view":
        monitoring_region = apply_lut(monitoring_region, lut) if lut is not None else np.power(monitoring_region, 1.0 / 2.4)
    monitoring_region = _apply_monitoring_review_transform(monitoring_region)
    monitoring_region = _extract_normalized_roi_region(monitoring_region, calibration_roi) if calibration_roi is not None else extract_center_region(monitoring_region, fraction=0.4)
    monitoring_region_seconds = time.perf_counter() - phase_started_at

    measurement_runtime = {
        "raw_region_seconds": float(raw_region_seconds),
        "monitoring_region_seconds": float(monitoring_region_seconds),
    }

    if str(target_type or "").strip().lower().replace("-", "_") == "gray_sphere" and calibration_roi is not None:
        phase_started_at = time.perf_counter()
        result = _measure_gray_sphere_statistics(raw_region, monitoring_region)
        measurement_runtime["gray_sphere_statistics_seconds"] = float(time.perf_counter() - phase_started_at)
        measurement_runtime["total_measurement_seconds"] = float(time.perf_counter() - measurement_started_at)
        result["measurement_runtime"] = measurement_runtime
        return result

    phase_started_at = time.perf_counter()
    raw_stats = _measure_three_sample_statistics(raw_region)
    measurement_runtime["raw_three_sample_seconds"] = float(time.perf_counter() - phase_started_at)
    phase_started_at = time.perf_counter()
    monitoring_stats = _measure_three_sample_statistics(monitoring_region)
    measurement_runtime["monitoring_three_sample_seconds"] = float(time.perf_counter() - phase_started_at)
    phase_started_at = time.perf_counter()
    saturation_monitoring = _measure_region_statistics(monitoring_region)
    measurement_runtime["saturation_monitoring_seconds"] = float(time.perf_counter() - phase_started_at)
    phase_started_at = time.perf_counter()
    saturation_raw = _measure_region_statistics(raw_region)
    measurement_runtime["saturation_raw_seconds"] = float(time.perf_counter() - phase_started_at)
    measurement_runtime["total_measurement_seconds"] = float(time.perf_counter() - measurement_started_at)
    return {
        "measured_log2_luminance": monitoring_stats["measured_log2_luminance"],
        "measured_log2_luminance_monitoring": monitoring_stats["measured_log2_luminance"],
        "measured_log2_luminance_raw": raw_stats["measured_log2_luminance"],
        "measured_rgb_mean": raw_stats["measured_rgb_mean"],
        "measured_rgb_chromaticity": raw_stats["measured_rgb_chromaticity"],
        "valid_pixel_count": monitoring_stats["valid_pixel_count"],
        "saturation_fraction": saturation_monitoring["saturation_fraction"],
        "black_fraction": saturation_monitoring["black_fraction"],
        "roi_variance": monitoring_stats["roi_variance"],
        "monitoring_roi_variance": monitoring_stats["roi_variance"],
        "raw_roi_variance": raw_stats["roi_variance"],
        "neutral_sample_count": monitoring_stats["neutral_sample_count"],
        "neutral_sample_log2_spread": monitoring_stats["neutral_sample_log2_spread"],
        "neutral_sample_chromaticity_spread": raw_stats["neutral_sample_chromaticity_spread"],
        "neutral_samples": monitoring_stats["neutral_samples"],
        "neutral_samples_raw": raw_stats["neutral_samples"],
        "raw_saturation_fraction": saturation_raw["saturation_fraction"],
        "workload_trace": {
            "measurement_domain": "scene_sdk_decode_with_proxy_monitoring",
            "detection_count": 0,
            "gradient_axis_count": 0,
            "region_stat_count": 4,
        },
        "measurement_runtime": measurement_runtime,
    }


def build_sample_plan(total_frames: int, *, sample_count: int, strategy: str) -> SamplePlan:
    if strategy not in {"uniform", "head"}:
        raise ValueError("sampling strategy must be uniform or head")
    if sample_count <= 1:
        representative_frame = max(0, min(total_frames - 1, total_frames // 2))
        return SamplePlan(strategy="representative", sample_count=1, start_frame=representative_frame, frame_step=1, max_frames=1)
    if strategy == "head":
        return SamplePlan(strategy=strategy, sample_count=sample_count, start_frame=0, frame_step=1, max_frames=min(total_frames, sample_count))
    frame_step = max(total_frames // max(sample_count, 1), 1)
    max_frames = min(sample_count, len(range(0, total_frames, frame_step)))
    return SamplePlan(strategy=strategy, sample_count=sample_count, start_frame=0, frame_step=frame_step, max_frames=max_frames)


def compute_confidence(frame_stats: List[FrameStat], *, unclamped_offset: float, clamp_stops: float) -> float:
    accepted_ratio = sum(1 for item in frame_stats if item.accepted) / max(len(frame_stats), 1)
    mean_clipping = sum(item.clipped_fraction for item in frame_stats) / max(len(frame_stats), 1)
    clamp_penalty = 1.0 if abs(unclamped_offset) <= clamp_stops else 0.65
    return max(0.0, min(1.0, accepted_ratio * (1.0 - mean_clipping) * clamp_penalty))


def resolve_lut(path: Optional[str]) -> Optional[CubeLut]:
    if path is None:
        return None
    candidate = Path(path).expanduser().resolve()
    return load_cube_lut(str(candidate)) if candidate.exists() else None


def discover_clips(input_path: str) -> List[Path]:
    raise_if_cancelled("Run cancelled before source scan.")
    path = Path(input_path).expanduser().resolve()
    if path.is_file():
        return [path] if is_valid_clip_file(path) else []

    clips: list[Path] = []
    for candidate in path.rglob("*"):
        raise_if_cancelled("Run cancelled during source scan.")
        if not candidate.is_file():
            continue
        if is_valid_clip_file(candidate):
            clips.append(candidate)
    return sorted(clips)


def is_valid_clip_file(path: Path) -> bool:
    if path.suffix.lower() != ".r3d":
        return False
    if any(part.startswith(".") or part.startswith("._") for part in path.parts):
        return False
    return True


def camera_group_from_clip_id(clip_id: str) -> str:
    return legacy_camera_group_from_clip_id(clip_id)


def median(values: Iterable[float]) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return math.log2(0.18)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
