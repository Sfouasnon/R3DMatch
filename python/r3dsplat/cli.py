from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from .colmap_bridge import ColmapCliBridge, ColmapConfig, solve_colmap
from .fiducials import FiducialConfig, solve_fiducials
from .geometry import align_world, debug_poses
from .ingest_backends import ingest_backend_summary
from .r3d_ingest import ingest_clip, inspect_clip
from .training import TrainingConfig, evaluate, render_sequence, train
from .training_bridge import GSplatRendererBridge

app = typer.Typer(no_args_is_help=True, help="R3DSplat internal CLI")


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
) -> None:
    typer.echo(str({"ingest_backend": ingest_backend_summary(backend=backend, sdk_root=sdk_root)}))
    dataset_dir = ingest_clip(input_path, out, backend=backend, sdk_root=sdk_root)
    typer.echo(f"Ingest complete: {dataset_dir}")


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


@app.command("train-static")
def train_static_command(
    dataset_dir: str,
    output_dir: str = typer.Option(..., "--out", help="Checkpoint output directory"),
    config: Optional[str] = typer.Option(None, "--config", help="Optional YAML config"),
    production_renderer: bool = typer.Option(False, help="Require real gsplat backend and disable fallback"),
) -> None:
    typer.echo(str({"renderer_backend": GSplatRendererBridge(production_mode=production_renderer).diagnostics()}))
    cfg = TrainingConfig.from_yaml(
        config,
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        mode="train-static",
        production_renderer=production_renderer,
    )
    checkpoint = train(cfg)
    typer.echo(f"Static training complete: {checkpoint}")


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
