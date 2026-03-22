from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Optional

import typer

from .colmap_bridge import ColmapCliBridge, ColmapConfig, solve_colmap
from .fiducials import FiducialConfig, solve_fiducials
from .geometry import align_world, debug_poses
from .ingest_backends import ingest_backend_summary
from .metadata import DatasetManifest
from .r3d_ingest import ingest_clip, inspect_clip
from .training import TrainingConfig, evaluate, render_sequence, train
from .training_bridge import GSplatRendererBridge

app = typer.Typer(no_args_is_help=True, help="R3DSplat internal CLI")
LOW_DISK_WARNING_INGEST_BYTES = 20 * 1024 * 1024 * 1024
LOW_DISK_WARNING_TRAIN_BYTES = 10 * 1024 * 1024 * 1024


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


def _make_ingest_progress_callback(output_path: str):
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
            "[ingest] frame {frame_index} ({completed}/{total}, {percent:.1f}%) -> {output_path} | elapsed {elapsed:.1f}s | decode {fps:.2f} fps".format(
                frame_index=int(progress["frame_index"]),
                completed=completed,
                total=total,
                percent=float(progress["percent"]),
                output_path=output_path,
                elapsed=float(progress["elapsed_seconds"]),
                fps=float(progress["decode_fps"]),
            )
        )

    return emit


@app.command("inspect")
def inspect_command(
    clip: str,
    backend: str = typer.Option("auto", help="Ingest backend: auto, mock, red-sdk"),
    sdk_root: Optional[str] = typer.Option(None, help="Optional RED SDK root override"),
) -> None:
    typer.echo(str({"ingest_backend": ingest_backend_summary(backend=backend, sdk_root=sdk_root)}))
    record = inspect_clip(clip, backend=backend, sdk_root=sdk_root)
    typer.echo(record.model_dump_json(indent=2))


@app.command("ingest")
def ingest_command(
    input_path: str,
    out: str = typer.Option(..., "--out", help="Dataset output directory"),
    backend: str = typer.Option("auto", help="Ingest backend: auto, mock, red-sdk"),
    sdk_root: Optional[str] = typer.Option(None, help="Optional RED SDK root override"),
    start_frame: int = typer.Option(0, "--start-frame", help="First source frame index to include"),
    max_frames: Optional[int] = typer.Option(None, "--max-frames", help="Maximum number of frames to ingest"),
    frame_step: int = typer.Option(1, "--frame-step", help="Source frame subsampling interval"),
) -> None:
    started = time.perf_counter()
    typer.echo(str({"ingest_backend": ingest_backend_summary(backend=backend, sdk_root=sdk_root)}))
    _warn_if_tmp_path(out, purpose="dataset output")
    _warn_if_low_disk_space(out, purpose="ingest", threshold_bytes=LOW_DISK_WARNING_INGEST_BYTES)
    dataset_dir = ingest_clip(
        input_path,
        out,
        backend=backend,
        sdk_root=sdk_root,
        start_frame=start_frame,
        max_frames=max_frames,
        frame_step=frame_step,
        progress_callback=_make_ingest_progress_callback(out),
    )
    manifest = DatasetManifest.load(dataset_dir)
    frame_indices = [frame.frame_index for frame in manifest.frames]
    manifest_path = Path(dataset_dir) / "manifest.json"
    if frame_indices:
        range_summary = f"{frame_indices[0]}..{frame_indices[-1]} (step={frame_step})"
    else:
        range_summary = "empty"
    typer.echo(
        "[ingest-summary] clip={clip} backend={backend} frame_range={frame_range} frames_written={frames_written} out={out_dir} manifest={manifest} elapsed={elapsed:.1f}s".format(
            clip=input_path,
            backend=manifest.backend_report.ingest_backend,
            frame_range=range_summary,
            frames_written=len(manifest.frames),
            out_dir=dataset_dir,
            manifest=manifest_path,
            elapsed=time.perf_counter() - started,
        )
    )


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
