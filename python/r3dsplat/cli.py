from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Any, Optional

import typer

from .colmap_bridge import ColmapCliBridge, ColmapConfig, solve_colmap
from .fiducials import FiducialConfig, solve_fiducials
from .geometry import align_world, debug_poses
from .ingest_backends import ingest_backend_summary
from .masking import BackgroundMattingConfig, run_background_matting
from .metadata import DatasetManifest
from .r3d_ingest import ingest_clip, inspect_clip
from .training import TrainingConfig, evaluate, render_sequence, train
from .training_bridge import GSplatRendererBridge

app = typer.Typer(no_args_is_help=True, help="R3DSplat internal CLI")
LOW_DISK_WARNING_INGEST_BYTES = 20 * 1024 * 1024 * 1024
LOW_DISK_WARNING_TRAIN_BYTES = 10 * 1024 * 1024 * 1024
INGEST_PRESETS = {
    "quick-test": {
        "start_frame": 0,
        "max_frames": 24,
        "frame_step": 2,
        "decode_mode": "half-good",
        "resize_mode": "bounds",
        "max_width": 960,
    },
    "desktop-review": {
        "start_frame": 0,
        "max_frames": 96,
        "frame_step": 2,
        "decode_mode": "half-premium",
        "resize_mode": "bounds",
        "max_width": 1920,
    },
    "full-quality": {
        "start_frame": 0,
        "max_frames": None,
        "frame_step": 1,
        "decode_mode": "full-premium",
    },
}


def _warn_if_tmp_path(path: str, *, purpose: str) -> None:
    resolved = Path(path).expanduser()
    if str(resolved).startswith("/tmp") or str(resolved).startswith("/var/tmp"):
        typer.secho(
            f"Warning: {purpose} is pointed at {resolved}. For real RED runs, prefer a user-owned path like ~/Desktop/r3d_real_test or ~/Desktop/r3d_real_run.",
            fg=typer.colors.YELLOW,
        )


def _warn_if_low_disk_space(path: str, *, purpose: str, threshold_bytes: int) -> None:
    resolved = Path(path).expanduser()
    target = resolved if resolved.exists() else resolved.parent
    target.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(target).free
    if free_bytes < threshold_bytes:
        typer.secho(
            "Warning: low free space on target filesystem for {purpose}: {free_gb:.1f} GiB free at {path}. Large real-data runs may fail or destabilize remote sessions.".format(
                purpose=purpose,
                free_gb=free_bytes / (1024 ** 3),
                path=target,
            ),
            fg=typer.colors.YELLOW,
        )


def _make_ingest_progress_callback(output_path: str, *, decode_mode: str, cached_size: tuple[int, int]):
    last_emit = {"time": 0.0, "completed": 0}

    def emit(progress: dict[str, object]) -> None:
        completed = int(progress["completed"])
        total = int(progress["total"])
        now = time.perf_counter()
        should_emit = (
            completed == 1
            or completed == total
            or completed - last_emit["completed"] >= 10
            or now - last_emit["time"] >= 1.0
        )
        if not should_emit:
            return
        last_emit["time"] = now
        last_emit["completed"] = completed
        typer.echo(
            "[ingest] frame {frame_index} ({completed}/{total}, {percent:.1f}%) -> {output_path} | decode_mode={decode_mode} | cache={width}x{height} | elapsed {elapsed:.1f}s | decode {fps:.2f} fps".format(
                frame_index=int(progress["frame_index"]),
                completed=completed,
                total=total,
                percent=float(progress["percent"]),
                output_path=output_path,
                decode_mode=decode_mode,
                width=cached_size[0],
                height=cached_size[1],
                elapsed=float(progress["elapsed_seconds"]),
                fps=float(progress["decode_fps"]),
            )
        )

    return emit


def _format_size(value_bytes: float) -> str:
    gib = value_bytes / (1024 ** 3)
    if gib >= 1.0:
        return f"{gib:.1f} GiB"
    mib = value_bytes / (1024 ** 2)
    return f"{mib:.0f} MiB"


def _estimate_range_bytes(value_bytes: float) -> tuple[float, float]:
    return value_bytes * 0.9, value_bytes * 1.1


def _estimate_output_bytes(width: int, height: int, frame_count: int) -> float:
    return float(width * height * 3 * 4 * frame_count)


def _decode_dimensions(width: int, height: int, decode_mode: str) -> tuple[int, int]:
    if decode_mode == "full-premium":
        return width, height
    if decode_mode in {"half-premium", "half-good"}:
        return max(1, (width + 1) // 2), max(1, (height + 1) // 2)
    raise ValueError(f"unsupported decode mode: {decode_mode}")


def _resolve_cached_size(decoded_width: int, decoded_height: int, settings: dict[str, Any]) -> tuple[int, int]:
    mode = settings["resize_mode"]
    if mode == "none":
        return decoded_width, decoded_height
    if mode == "scale":
        scale = float(settings["resize_scale"])
        return max(1, int(round(decoded_width * scale))), max(1, int(round(decoded_height * scale)))
    width_scale = 1.0
    height_scale = 1.0
    if settings.get("max_width") is not None:
        width_scale = min(1.0, float(settings["max_width"]) / float(decoded_width))
    if settings.get("max_height") is not None:
        height_scale = min(1.0, float(settings["max_height"]) / float(decoded_height))
    scale = min(width_scale, height_scale)
    return max(1, int(round(decoded_width * scale))), max(1, int(round(decoded_height * scale)))


def _resolved_ingest_settings(
    *,
    preset: Optional[str],
    start_frame: Optional[int],
    max_frames: Optional[int],
    frame_step: Optional[int],
    decode_mode: Optional[str],
    resize_scale: Optional[float],
    max_width: Optional[int],
    max_height: Optional[int],
) -> dict[str, Any]:
    if preset is not None and preset not in INGEST_PRESETS:
        raise typer.BadParameter(f"unknown preset: {preset}")
    if resize_scale is not None and (max_width is not None or max_height is not None):
        raise typer.BadParameter("resize-scale cannot be combined with max-width or max-height")
    resolved = dict(INGEST_PRESETS.get(preset, {}))
    if start_frame is not None:
        resolved["start_frame"] = start_frame
    if max_frames is not None:
        resolved["max_frames"] = max_frames
    if frame_step is not None:
        resolved["frame_step"] = frame_step
    if decode_mode is not None:
        resolved["decode_mode"] = decode_mode
    if resize_scale is not None:
        resolved["resize_mode"] = "scale"
        resolved["resize_scale"] = resize_scale
        resolved.pop("max_width", None)
        resolved.pop("max_height", None)
    else:
        if max_width is not None:
            resolved["max_width"] = max_width
        if max_height is not None:
            resolved["max_height"] = max_height
        if max_width is not None or max_height is not None:
            resolved["resize_mode"] = "bounds"
    resolved.setdefault("start_frame", 0)
    resolved.setdefault("max_frames", None)
    resolved.setdefault("frame_step", 1)
    resolved.setdefault("decode_mode", "full-premium")
    resolved.setdefault("resize_mode", "none")
    if resolved["resize_mode"] == "none":
        resolved.pop("resize_scale", None)
        resolved.pop("max_width", None)
        resolved.pop("max_height", None)
    return resolved


def _selected_frame_count(total_frames: int, start_frame: int, max_frames: Optional[int], frame_step: int) -> int:
    if start_frame >= total_frames:
        return 0
    count = len(range(start_frame, total_frames, frame_step))
    return count if max_frames is None else min(count, max_frames)


def _print_ingest_estimate(
    *,
    backend: str,
    clip_width: int,
    clip_height: int,
    settings: dict[str, Any],
    selected_frames: int,
) -> tuple[int, int]:
    full_width, full_height = clip_width, clip_height
    decode_mode_note = settings["decode_mode"]
    if backend == "red-sdk":
        decoded_width, decoded_height = _decode_dimensions(full_width, full_height, settings["decode_mode"])
    else:
        decoded_width, decoded_height = full_width, full_height
        decode_mode_note = f"{settings['decode_mode']} (ignored for {backend})"
    cached_width, cached_height = _resolve_cached_size(decoded_width, decoded_height, settings)
    final_bytes = _estimate_output_bytes(cached_width, cached_height, selected_frames)
    final_low, final_high = _estimate_range_bytes(final_bytes)
    per_frame_low, per_frame_high = _estimate_range_bytes(_estimate_output_bytes(cached_width, cached_height, 1))

    typer.echo(
        "[ingest-settings] preset={preset} frame_selection=start:{start}, max:{max_frames}, step:{step} decode_mode={decode_mode} resize_mode={resize_mode} resize_scale={resize_scale} max_width={max_width} max_height={max_height}".format(
            preset=settings.get("preset") or "none",
            start=settings["start_frame"],
            max_frames=settings["max_frames"],
            step=settings["frame_step"],
            decode_mode=decode_mode_note,
            resize_mode=settings["resize_mode"],
            resize_scale=settings.get("resize_scale"),
            max_width=settings.get("max_width"),
            max_height=settings.get("max_height"),
        )
    )
    typer.echo(
        "[ingest-resolution] original={ow}x{oh} decoded={dw}x{dh} final_cached={cw}x{ch}".format(
            ow=full_width,
            oh=full_height,
            dw=decoded_width,
            dh=decoded_height,
            cw=cached_width,
            ch=cached_height,
        )
    )
    typer.echo(
        "[ingest-estimate] selected_frames={frames} cached_resolution={cw}x{ch} per_frame~{pfl}-{pfh} total~{tl}-{th}".format(
            frames=selected_frames,
            cw=cached_width,
            ch=cached_height,
            pfl=_format_size(per_frame_low),
            pfh=_format_size(per_frame_high),
            tl=_format_size(final_low),
            th=_format_size(final_high),
        )
    )
    if full_width >= 6000 and settings["decode_mode"] == "full-premium" and selected_frames >= 48:
        typer.secho(
            "Warning: high-risk ingest configuration detected for a large clip. Consider --max-frames, --frame-step, --decode-mode half-premium, or --max-width/--resize-scale, and prefer a path like ~/Desktop/r3d_real_test.",
            fg=typer.colors.YELLOW,
        )
    return cached_width, cached_height


@app.command("inspect")
def inspect_command(
    clip: str,
    backend: str = typer.Option("auto", help="Ingest backend: auto, mock, red-sdk"),
    sdk_root: Optional[str] = typer.Option(None, help="Optional RED SDK root override"),
    decode_mode: Optional[str] = typer.Option(None, help="Optional RED decode mode override"),
) -> None:
    typer.echo(str({"ingest_backend": ingest_backend_summary(backend=backend, sdk_root=sdk_root, decode_mode=decode_mode)}))
    record = inspect_clip(clip, backend=backend, sdk_root=sdk_root, decode_mode=decode_mode)
    typer.echo(record.model_dump_json(indent=2))


@app.command("ingest")
def ingest_command(
    input_path: str,
    out: str = typer.Option(..., "--out", help="Dataset output directory"),
    backend: str = typer.Option("auto", help="Ingest backend: auto, mock, red-sdk"),
    sdk_root: Optional[str] = typer.Option(None, help="Optional RED SDK root override"),
    start_frame: Optional[int] = typer.Option(None, "--start-frame", help="First source frame index to include"),
    max_frames: Optional[int] = typer.Option(None, "--max-frames", help="Maximum number of frames to ingest"),
    frame_step: Optional[int] = typer.Option(None, "--frame-step", help="Source frame subsampling interval"),
    resize_scale: Optional[float] = typer.Option(None, "--resize-scale", help="Uniform resize scale before cache write"),
    max_width: Optional[int] = typer.Option(None, "--max-width", help="Maximum cached width with aspect preserved"),
    max_height: Optional[int] = typer.Option(None, "--max-height", help="Maximum cached height with aspect preserved"),
    decode_mode: Optional[str] = typer.Option(None, "--decode-mode", help="RED decode mode: full-premium, half-premium, half-good"),
    preset: Optional[str] = typer.Option(None, "--preset", help="Ingest preset: quick-test, desktop-review, full-quality"),
    colmap_image_format: str = typer.Option("tiff", "--colmap-image-format", help="Derived image format for COLMAP exports"),
    matte_image_format: str = typer.Option("tiff", "--matte-image-format", help="Derived image format for BackgroundMattingV2 exports"),
    colmap_max_width: Optional[int] = typer.Option(None, "--colmap-max-width", help="Maximum width for COLMAP image exports"),
    colmap_max_height: Optional[int] = typer.Option(None, "--colmap-max-height", help="Maximum height for COLMAP image exports"),
    matte_max_width: Optional[int] = typer.Option(None, "--matte-max-width", help="Maximum width for matting image exports"),
    matte_max_height: Optional[int] = typer.Option(None, "--matte-max-height", help="Maximum height for matting image exports"),
    colmap_compression: Optional[str] = typer.Option(None, "--colmap-compression", help="Optional TIFF compression mode for COLMAP exports"),
    matte_compression: Optional[str] = typer.Option(None, "--matte-compression", help="Optional TIFF compression mode for matting exports"),
    dry_run: bool = typer.Option(False, "--dry-run", "--estimate-only", help="Resolve settings and estimate output size without decoding"),
) -> None:
    started = time.perf_counter()
    settings = _resolved_ingest_settings(
        preset=preset,
        start_frame=start_frame,
        max_frames=max_frames,
        frame_step=frame_step,
        decode_mode=decode_mode,
        resize_scale=resize_scale,
        max_width=max_width,
        max_height=max_height,
    )
    settings["preset"] = preset
    typer.echo(str({"ingest_backend": ingest_backend_summary(backend=backend, sdk_root=sdk_root, decode_mode=settings["decode_mode"])}))
    _warn_if_tmp_path(out, purpose="dataset output")
    _warn_if_low_disk_space(out, purpose="ingest", threshold_bytes=LOW_DISK_WARNING_INGEST_BYTES)
    resolved_dataset_dir = Path(out).expanduser().resolve()
    typer.echo(
        "[ingest-paths] dataset_dir={dataset_dir} colmap_images={colmap_dir} matte_images={matte_dir}".format(
            dataset_dir=resolved_dataset_dir,
            colmap_dir=resolved_dataset_dir / "colmap_images",
            matte_dir=resolved_dataset_dir / "matte_images",
        )
    )
    typer.echo(
        "[ingest-derived-assets] colmap={fmt} max_width={cw} max_height={ch} compression={cc} | matte={mfmt} max_width={mw} max_height={mh} compression={mc}".format(
            fmt=colmap_image_format,
            cw=colmap_max_width,
            ch=colmap_max_height,
            cc=colmap_compression,
            mfmt=matte_image_format,
            mw=matte_max_width,
            mh=matte_max_height,
            mc=matte_compression,
        )
    )
    clip_record = inspect_clip(input_path, backend=backend, sdk_root=sdk_root, decode_mode=settings["decode_mode"])
    selected_frames = _selected_frame_count(
        clip_record.total_frames,
        settings["start_frame"],
        settings["max_frames"],
        settings["frame_step"],
    )
    cached_width, cached_height = _print_ingest_estimate(
        backend=backend,
        clip_width=clip_record.width,
        clip_height=clip_record.height,
        settings=settings,
        selected_frames=selected_frames,
    )
    if dry_run:
        typer.echo("[ingest-dry-run] estimate only; no frames decoded and no dataset written")
        return
    dataset_dir = ingest_clip(
        input_path,
        out,
        backend=backend,
        sdk_root=sdk_root,
        start_frame=settings["start_frame"],
        max_frames=settings["max_frames"],
        frame_step=settings["frame_step"],
        decode_mode=settings["decode_mode"],
        resize_scale=settings.get("resize_scale"),
        max_width=settings.get("max_width"),
        max_height=settings.get("max_height"),
        colmap_image_format=colmap_image_format,
        matte_image_format=matte_image_format,
        colmap_max_width=colmap_max_width,
        colmap_max_height=colmap_max_height,
        matte_max_width=matte_max_width,
        matte_max_height=matte_max_height,
        colmap_compression=colmap_compression,
        matte_compression=matte_compression,
        resolved_preset=preset,
        progress_callback=_make_ingest_progress_callback(
            out,
            decode_mode=settings["decode_mode"],
            cached_size=(cached_width, cached_height),
        ),
    )
    manifest = DatasetManifest.load(dataset_dir)
    frame_indices = [frame.frame_index for frame in manifest.frames]
    manifest_path = Path(dataset_dir) / "manifest.json"
    if frame_indices:
        range_summary = f"{frame_indices[0]}..{frame_indices[-1]} (step={settings['frame_step']})"
    else:
        range_summary = "empty"
    typer.echo(
        "[ingest-summary] clip={clip} backend={backend} preset={preset} frame_range={frame_range} frames_written={frames_written} exported_colmap_images={exported_colmap} exported_matte_images={exported_matte} original={original_width}x{original_height} decoded={decoded_width}x{decoded_height} cached={cached_width}x{cached_height} decode_mode={decode_mode} colmap_images={colmap_dir} matte_images={matte_dir} out={out_dir} manifest={manifest} elapsed={elapsed:.1f}s".format(
            clip=input_path,
            backend=manifest.backend_report.ingest_backend,
            preset=preset or "none",
            original_width=manifest.backend_report.ingest_settings.get("original_width", clip_record.width),
            original_height=manifest.backend_report.ingest_settings.get("original_height", clip_record.height),
            decoded_width=manifest.backend_report.ingest_settings.get("decoded_width", clip_record.width),
            decoded_height=manifest.backend_report.ingest_settings.get("decoded_height", clip_record.height),
            decode_mode=(
                manifest.backend_report.ingest_settings.get("decode_mode", settings["decode_mode"])
                if manifest.backend_report.ingest_backend == "red-sdk"
                else f"{manifest.backend_report.ingest_settings.get('decode_mode', settings['decode_mode'])} (ignored for {manifest.backend_report.ingest_backend})"
            ),
            frame_range=range_summary,
            frames_written=len(manifest.frames),
            exported_colmap=len(list(Path(manifest.backend_report.ingest_settings.get("colmap_export", {}).get("path", Path(dataset_dir) / "colmap_images")).glob(f"*.{colmap_image_format.lower() if colmap_image_format.lower() != 'tif' else 'tiff'}"))),
            exported_matte=len(list(Path(manifest.backend_report.ingest_settings.get("matte_export", {}).get("path", Path(dataset_dir) / "matte_images")).glob(f"*.{matte_image_format.lower() if matte_image_format.lower() != 'tif' else 'tiff'}"))),
            cached_width=manifest.backend_report.ingest_settings.get("cached_width", cached_width),
            cached_height=manifest.backend_report.ingest_settings.get("cached_height", cached_height),
            colmap_dir=manifest.backend_report.ingest_settings.get("colmap_export", {}).get("path", Path(dataset_dir) / "colmap_images"),
            matte_dir=manifest.backend_report.ingest_settings.get("matte_export", {}).get("path", Path(dataset_dir) / "matte_images"),
            out_dir=dataset_dir,
            manifest=manifest_path,
            elapsed=time.perf_counter() - started,
        )
    )
    typer.echo("Derived assets written successfully")


@app.command("solve-fiducials")
def solve_fiducials_command(
    dataset_dir: str,
    backend: str = typer.Option("mock", help="Fiducial backend: mock, aruco"),
    marker_length_m: float = typer.Option(0.2, help="Physical fiducial size in meters"),
    exclude_from_training: bool = typer.Option(True, help="Create masks to exclude fiducials from training"),
) -> None:
    result = solve_fiducials(
        dataset_dir,
        FiducialConfig(
            backend=backend,
            marker_length_m=marker_length_m,
            exclude_from_training=exclude_from_training,
        ),
    )
    typer.echo(str(result))


@app.command("run-matting")
def run_matting_command(
    dataset_dir: str,
    background_dir: str = typer.Option(..., "--background-dir", help="Background image directory for BackgroundMattingV2"),
    checkpoint: str = typer.Option(..., "--checkpoint", help="BackgroundMattingV2 model checkpoint"),
    repo_dir: str = typer.Option(
        ...,
        "--repo-dir",
        help="BackgroundMattingV2 checkout directory",
    ),
    model_type: str = typer.Option("mattingrefine", help="BackgroundMattingV2 model type"),
    model_backbone: str = typer.Option("resnet50", help="BackgroundMattingV2 backbone"),
    model_backbone_scale: float = typer.Option(0.25, help="BackgroundMattingV2 backbone scale"),
    model_refine_mode: str = typer.Option("sampling", help="BackgroundMattingV2 refine mode"),
    model_refine_sample_pixels: int = typer.Option(80000, help="BackgroundMattingV2 refine sample pixels"),
    device: str = typer.Option("cuda", help="BackgroundMattingV2 device"),
    num_workers: int = typer.Option(0, help="BackgroundMattingV2 dataloader workers"),
    preprocess_alignment: bool = typer.Option(False, help="Enable BackgroundMattingV2 homographic alignment"),
) -> None:
    result = run_background_matting(
        dataset_dir,
        BackgroundMattingConfig(
            repo_dir=repo_dir,
            background_dir=background_dir,
            checkpoint=checkpoint,
            model_type=model_type,
            model_backbone=model_backbone,
            model_backbone_scale=model_backbone_scale,
            model_refine_mode=model_refine_mode,
            model_refine_sample_pixels=model_refine_sample_pixels,
            device=device,
            num_workers=num_workers,
            preprocess_alignment=preprocess_alignment,
        ),
    )
    typer.echo(str(result))


@app.command("solve-colmap")
def solve_colmap_command(
    dataset_dir: str,
    executable: str = typer.Option("colmap", help="COLMAP executable"),
    mode: str = typer.Option("standard", help="COLMAP solve mode"),
    matching_mode: str = typer.Option("sequential", help="COLMAP matching mode"),
    intrinsics_mode: str = typer.Option("shared", help="COLMAP intrinsics mode"),
    camera_model: str = typer.Option("PINHOLE", help="COLMAP camera model"),
) -> None:
    typer.echo(str({"colmap_backend": ColmapCliBridge(ColmapConfig(executable=executable, mode=mode, matching_mode=matching_mode, intrinsics_mode=intrinsics_mode, camera_model=camera_model)).diagnostics()}))
    result = solve_colmap(
        dataset_dir,
        ColmapConfig(
            executable=executable,
            mode=mode,
            matching_mode=matching_mode,
            intrinsics_mode=intrinsics_mode,
            camera_model=camera_model,
        ),
    )
    typer.echo(str(result))


@app.command("align-world")
def align_world_command(dataset_dir: str) -> None:
    typer.echo(str(align_world(dataset_dir)))


@app.command("debug-poses")
def debug_poses_command(dataset_dir: str) -> None:
    typer.echo(str(debug_poses(dataset_dir)))


@app.command("train-4d")
def train_4d_command(
    dataset_dir: str,
    output_dir: str = typer.Option(..., "--out", help="Checkpoint output directory"),
    config: Optional[str] = typer.Option(None, "--config", help="Optional YAML config"),
    epochs: Optional[int] = typer.Option(None, help="Override epochs"),
    window_size: Optional[int] = typer.Option(None, help="Override temporal window size"),
    num_gaussians: Optional[int] = typer.Option(None, help="Override Gaussian count"),
    production_renderer: bool = typer.Option(False, help="Require real gsplat backend and disable fallback"),
) -> None:
    typer.echo(str({"renderer_backend": GSplatRendererBridge(production_mode=production_renderer).diagnostics()}))
    _warn_if_tmp_path(output_dir, purpose="training output")
    _warn_if_low_disk_space(output_dir, purpose="training", threshold_bytes=LOW_DISK_WARNING_TRAIN_BYTES)
    cfg = TrainingConfig.from_yaml(
        config,
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        mode="train-4d",
        epochs=epochs,
        window_size=window_size,
        num_gaussians=num_gaussians,
        production_renderer=production_renderer,
    )
    checkpoint = train(cfg)
    typer.echo(f"4D training complete: {checkpoint}")
    typer.echo(f"Run summary: {Path(output_dir) / 'run_summary.json'}")


@app.command("train-static")
def train_static_command(
    dataset_dir: str,
    output_dir: str = typer.Option(..., "--out", help="Checkpoint output directory"),
    config: Optional[str] = typer.Option(None, "--config", help="Optional YAML config"),
    production_renderer: bool = typer.Option(False, help="Require real gsplat backend and disable fallback"),
) -> None:
    typer.echo(str({"renderer_backend": GSplatRendererBridge(production_mode=production_renderer).diagnostics()}))
    _warn_if_tmp_path(output_dir, purpose="training output")
    _warn_if_low_disk_space(output_dir, purpose="training", threshold_bytes=LOW_DISK_WARNING_TRAIN_BYTES)
    cfg = TrainingConfig.from_yaml(
        config,
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        mode="train-static",
        production_renderer=production_renderer,
    )
    checkpoint = train(cfg)
    typer.echo(f"Static training complete: {checkpoint}")
    typer.echo(f"Run summary: {Path(output_dir) / 'run_summary.json'}")


@app.command("eval-4d")
def eval_4d_command(checkpoint: str, dataset_dir: Optional[str] = None) -> None:
    payload = evaluate(checkpoint)
    if dataset_dir is not None:
        payload["dataset_dir"] = dataset_dir
    typer.echo(str(payload))


@app.command("render-sequence")
def render_sequence_command(
    checkpoint: str,
    dataset_dir: Optional[str] = typer.Option(None, help="Optional dataset override"),
    out: Optional[str] = typer.Option(None, "--out", help="Optional torch tensor output path"),
) -> None:
    rendered = render_sequence(checkpoint, dataset_dir=dataset_dir)
    if out:
        import torch

        out_path = Path(out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(rendered, out_path)
        typer.echo(f"Rendered sequence saved to {out_path}")
        return
    typer.echo(str(tuple(rendered.shape)))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
