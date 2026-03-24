from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np

from .luts import CubeLut, apply_lut, load_cube_lut
from .models import ClipMetadata, ClipResult, FrameStat, MonitoringContext, SamplePlan
from .sdk import resolve_backend


def analyze_path(
    input_path: str,
    *,
    out_dir: str,
    mode: str,
    backend: str,
    lut_override: Optional[str],
    sample_count: int,
    sampling_strategy: str,
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
        raw_offset = item.global_reference_log2 - item.clip_statistic_log2
        by_group.setdefault(item.camera_group, []).append(raw_offset)

    camera_baselines = {group: median(values) for group, values in by_group.items()}

    final_results: list[ClipResult] = []
    for item in provisional:
        raw_offset = item.raw_offset_stops
        camera_baseline = camera_baselines[item.camera_group]
        clip_trim = raw_offset - camera_baseline
        final_offset = max(-clamp_stops, min(clamp_stops, clip_trim))
        confidence = compute_confidence(item.frame_stats, unclamped_offset=raw_offset, clamp_stops=clamp_stops)
        result = ClipResult(
            clip_id=item.clip_id,
            source_path=item.source_path,
            backend=item.backend,
            camera_group=item.camera_group,
            clip_statistic_log2=item.clip_statistic_log2,
            camera_group_statistic_log2=item.camera_group_statistic_log2,
            global_reference_log2=item.global_reference_log2,
            raw_offset_stops=raw_offset,
            camera_baseline_stops=camera_baseline,
            clip_trim_stops=clip_trim,
            final_offset_stops=final_offset,
            confidence=confidence,
            sample_plan=item.sample_plan,
            monitoring=item.monitoring,
            clip_metadata=item.clip_metadata,
            frame_stats=item.frame_stats,
            diagnostics=item.diagnostics,
        )
        final_results.append(result)
        write_json(analysis_dir / f"{result.clip_id}.analysis.json", result.to_dict())
        write_json(sidecar_dir / f"{result.clip_id}.sidecar.json", build_sidecar(result))

    summary = {
        "input_path": str(Path(input_path).expanduser().resolve()),
        "mode": mode,
        "backend": backend,
        "clip_count": len(final_results),
        "analysis_dir": str(analysis_dir),
        "sidecar_dir": str(sidecar_dir),
        "camera_baselines": camera_baselines,
        "clips": [
            {
                "clip_id": item.clip_id,
                "camera_group": item.camera_group,
                "raw_offset_stops": item.raw_offset_stops,
                "camera_baseline_stops": item.camera_baseline_stops,
                "final_offset_stops": item.final_offset_stops,
                "confidence": item.confidence,
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
    camera_group = camera_group_from_clip_id(clip.clip_id)
    raw_offset = global_reference - clip_statistic
    return ClipResult(
        clip_id=clip.clip_id,
        source_path=clip.source_path,
        backend=backend_impl.name,
        camera_group=camera_group,
        clip_statistic_log2=clip_statistic,
        camera_group_statistic_log2=clip_statistic,
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
    luminance = np.clip(
        transformed[0] * 0.2126 + transformed[1] * 0.7152 + transformed[2] * 0.0722,
        1e-6,
        1.0,
    )
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


def build_sidecar(result: ClipResult) -> Dict[str, object]:
    return {
        "clip_id": result.clip_id,
        "source_path": result.source_path,
        "camera_group": result.camera_group,
        "mode": result.monitoring.mode,
        "raw_offset_stops": result.raw_offset_stops,
        "camera_baseline_stops": result.camera_baseline_stops,
        "final_offset_stops": result.final_offset_stops,
        "confidence": result.confidence,
        "monitoring": asdict(result.monitoring),
        "redline_mapping": {
            "prototype_note": "Map final_offset_stops into the local REDLine exposure/grade flag set during validation.",
            "raw_offset_stops": result.raw_offset_stops,
            "camera_baseline_stops": result.camera_baseline_stops,
            "final_offset_stops": result.final_offset_stops,
        },
    }


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
    return clip_id.split("_", 1)[0] if "_" in clip_id else clip_id


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
