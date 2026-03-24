from __future__ import annotations

from typing import Optional

import typer

from .matching import analyze_path
from .transcode import write_transcode_plan

app = typer.Typer(no_args_is_help=True, help="R3DMatch CLI")


@app.command("analyze")
def analyze_command(
    input_path: str,
    out: str = typer.Option(..., "--out", help="Analysis output directory"),
    mode: str = typer.Option("scene", "--mode", help="Matching mode: scene or view"),
    lut: Optional[str] = typer.Option(None, "--lut", help="Optional LUT override (.cube)"),
    backend: str = typer.Option("auto", "--backend", help="Backend: auto, mock, red-sdk"),
    sample_count: int = typer.Option(8, "--sample-count", help="Number of sampled frames per clip"),
    sampling_strategy: str = typer.Option("uniform", "--sampling-strategy", help="Sampling strategy: uniform or head"),
) -> None:
    normalized_mode = mode.lower()
    if normalized_mode not in {"scene", "view"}:
        raise typer.BadParameter("mode must be scene or view")
    payload = analyze_path(
        input_path,
        out_dir=out,
        mode=normalized_mode,
        backend=backend,
        lut_override=lut,
        sample_count=sample_count,
        sampling_strategy=sampling_strategy,
    )
    typer.echo(str(payload))


@app.command("transcode")
def transcode_command(
    input_path: str,
    out: str = typer.Option(..., "--out", help="Transcode plan output directory"),
    analysis_dir: Optional[str] = typer.Option(None, "--analysis-dir", help="Analysis directory that contains generated sidecars"),
    use_generated_sidecar: bool = typer.Option(False, "--use-generated-sidecar", help="Load sidecars from the analysis output"),
    redline_executable: str = typer.Option("REDLine", "--redline-executable", help="REDLine executable"),
    output_ext: str = typer.Option("mov", "--output-ext", help="Rendered output extension"),
    execute: bool = typer.Option(False, "--execute", help="Execute commands instead of just writing plans"),
) -> None:
    payload = write_transcode_plan(
        input_path,
        out_dir=out,
        analysis_dir=analysis_dir,
        use_generated_sidecar=use_generated_sidecar,
        redline_executable=redline_executable,
        output_ext=output_ext,
        execute=execute,
    )
    typer.echo(str(payload))


def main() -> None:
    app()


if __name__ == "__main__":
    main()

