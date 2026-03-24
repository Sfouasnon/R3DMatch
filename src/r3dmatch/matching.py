from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np

from .calibration import calibration_baselines, color_calibration_gains, load_calibration, load_color_calibration, load_exposure_calibration
from .identity import group_key_from_clip_id, legacy_camera_group_from_clip_id, rmd_name_for_clip_id
from .luts import CubeLut, apply_lut, load_cube_lut
from .models import ClipResult, FrameStat, MonitoringContext, SamplePlan
from .sdk import resolve_backend
from .sidecar import build_sidecar_payload, write_sidecar_file


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
    sample_count: int = 8,
    sampling_strategy: str = "uniform",
    clamp_stops: float = 3.0,
) -> Dict[str, object]:
    clips = discover_clips(input_path)
    if not clips:
        raise ValueError(f"No .R3D clips found under {input_path}")

    out_root = Path(out_dir).expanduser().resolve()
    analysis_dir = out_root / "analysis"
    sidecar_dir = out_root / "sidecars"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    sidecar_dir.mkdir(parents=True, exist_ok=True)

    provisional = [
        analyze_clip(
            str(clip),
            mode=mode,
            backend=backend,
            lut_override=lut_override,
            sample_count=sample_count,
            sampling_strategy=sampling_strategy,
        )
        for clip in clips
    ]

    by_group: Dict[str, List[float]] = {}
    for item in provisional:
        by_group.setdefault(item.group_key, []).append(item.raw_offset_stops)
    group_baselines = {group_key: median(values) for group_key, values in by_group.items()}

    resolved_exposure_path = exposure_calibration_path or calibration_path
    loaded_exposure_calibration = load_exposure_calibration(resolved_exposure_path) if resolved_exposure_path else None
    loaded_color_calibration = load_color_calibration(color_calibration_path) if color_calibration_path else None
    calibrated_baselines = calibration_baselines(loaded_exposure_calibration) if loaded_exposure_calibration else {}
    applied_color_gains = color_calibration_gains(loaded_color_calibration) if loaded_color_calibration else {}

    final_results: list[ClipResult] = []
    for item in provisional:
        derived_group_baseline = group_baselines[item.group_key]
        clip_trim = item.raw_offset_stops - derived_group_baseline
        applied_baseline = calibrated_baselines.get(item.group_key, derived_group_baseline)
        final_offset = max(-clamp_stops, min(clamp_stops, applied_baseline + clip_trim))
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
            if loaded_exposure_calibration and item.group_key in calibrated_baselines
            else None
        )
        color_provenance = (
            {
                "source": str(Path(color_calibration_path).expanduser().resolve()),
                "object_type": loaded_color_calibration.object_type,
                "calibration_mode": loaded_color_calibration.calibration_mode,
                "calibration_type": loaded_color_calibration.calibration_type,
                "pending_rgb_neutral_gains": applied_color_gains[item.group_key],
            }
            if loaded_color_calibration and item.group_key in applied_color_gains
            else None
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
            pending_color_gains=applied_color_gains.get(item.group_key),
            exposure_calibration_loaded=exposure_provenance is not None,
            exposure_baseline_applied_stops=applied_baseline if exposure_provenance is not None else None,
            color_calibration_loaded=color_provenance is not None,
            color_gains_state="pending" if color_provenance is not None else None,
            diagnostics={
                **item.diagnostics,
                "pending_color_gains": applied_color_gains.get(item.group_key),
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
    }
    write_json(out_root / "summary.json", summary)
    return summary


def analyze_clip(
    source_path: str,
    *,
    mode: str,
    backend: str,
    lut_override: Optional[str],
    sample_count: int,
    sampling_strategy: str,
) -> ClipResult:
    backend_impl = resolve_backend(backend)
    clip = backend_impl.inspect_clip(source_path)
    sample_plan = build_sample_plan(clip.total_frames, sample_count=sample_count, strategy=sampling_strategy)
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

    frame_stats: list[FrameStat] = []
    accepted_values: list[float] = []
    for frame_index, timestamp_seconds, image in backend_impl.decode_frames(
        source_path,
        start_frame=sample_plan.start_frame,
        max_frames=sample_plan.max_frames,
        frame_step=sample_plan.frame_step,
    ):
        stat = analyze_frame(frame_index, timestamp_seconds, image, mode=mode, lut=lut)
        frame_stats.append(stat)
        if stat.accepted:
            accepted_values.append(stat.log_luminance_median)
    if not accepted_values:
        accepted_values = [item.log_luminance_median for item in frame_stats]

    clip_statistic = median(accepted_values)
    global_reference = math.log2(0.18)
    raw_offset = global_reference - clip_statistic
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
        diagnostics={"sampled_frames": len(frame_stats), "accepted_frames": sum(1 for item in frame_stats if item.accepted)},
    )


def analyze_frame(frame_index: int, timestamp_seconds: float, image: np.ndarray, *, mode: str, lut: Optional[CubeLut]) -> FrameStat:
    transformed = np.clip(image, 0.0, 1.0)
    if mode == "view":
        transformed = apply_lut(transformed, lut) if lut is not None else np.power(transformed, 1.0 / 2.4)
    luminance = np.clip(transformed[0] * 0.2126 + transformed[1] * 0.7152 + transformed[2] * 0.0722, 1e-6, 1.0)
    clipped = (luminance <= 0.002) | (luminance >= 0.998)
    valid = ~clipped
    clipped_fraction = float(clipped.mean())
    valid_fraction = float(valid.mean())
    log_values = np.log2(luminance[valid if valid.any() else np.ones_like(valid, dtype=bool)])
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


def build_sample_plan(total_frames: int, *, sample_count: int, strategy: str) -> SamplePlan:
    if strategy not in {"uniform", "head"}:
        raise ValueError("sampling strategy must be uniform or head")
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
    path = Path(input_path).expanduser().resolve()
    if path.is_file():
        return [path] if is_valid_clip_file(path) else []

    clips: list[Path] = []
    for candidate in path.rglob("*"):
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
