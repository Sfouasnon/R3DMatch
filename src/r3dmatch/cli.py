from __future__ import annotations

from typing import Optional

import typer

from .calibration import calibrate_card_path, calibrate_color_path, calibrate_exposure_path, calibrate_sphere_path
from .matching import analyze_path
from .report import build_contact_sheet_report
from .transcode import write_transcode_plan
from .validation import validate_pipeline

app = typer.Typer(no_args_is_help=True, help="R3DMatch CLI")


@app.command("analyze")
def analyze_command(
    input_path: str,
    out: str = typer.Option(..., "--out", help="Analysis output directory"),
    mode: str = typer.Option("scene", "--mode", help="Matching mode: scene or view"),
    lut: Optional[str] = typer.Option(None, "--lut", help="Optional LUT override (.cube)"),
    calibration: Optional[str] = typer.Option(None, "--calibration", help="Optional legacy/single exposure calibration JSON"),
    exposure_calibration: Optional[str] = typer.Option(None, "--exposure-calibration", help="Optional exposure calibration JSON"),
    color_calibration: Optional[str] = typer.Option(None, "--color-calibration", help="Optional color calibration JSON"),
    backend: str = typer.Option("mock", "--backend", help="Backend: mock or red"),
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
        calibration_path=calibration,
        exposure_calibration_path=exposure_calibration,
        color_calibration_path=color_calibration,
        sample_count=sample_count,
        sampling_strategy=sampling_strategy,
    )
    typer.echo(str(payload))


@app.command("calibrate-sphere")
def calibrate_sphere_command(
    input_path: str,
    target_log2: float = typer.Option(..., "--target-log2", help="Target log2 luminance for the gray sphere"),
    out: str = typer.Option(..., "--out", help="Calibration output directory"),
    roi_file: str = typer.Option(..., "--roi-file", help="JSON file mapping camera groups to {cx, cy, r}"),
    backend: str = typer.Option("mock", "--backend", help="Backend: mock or red"),
) -> None:
    payload = calibrate_sphere_path(
        input_path,
        target_log2_luminance=target_log2,
        out_dir=out,
        roi_file=roi_file,
        backend=backend,
    )
    typer.echo(str(payload))


@app.command("calibrate-exposure")
def calibrate_exposure_command(
    input_path: str,
    out: str = typer.Option(..., "--out", help="Exposure calibration output directory"),
    source: str = typer.Option("gray_card", "--source", help="Exposure source: gray_card, gray_sphere, center_crop, or roi"),
    roi_file: Optional[str] = typer.Option(None, "--roi-file", help="Optional manual ROI override"),
    target_log2: Optional[float] = typer.Option(None, "--target-log2", help="Target log2 luminance"),
    reference_camera: Optional[str] = typer.Option(None, "--reference-camera", help="Reference camera group or clip ID"),
    sampling_mode: str = typer.Option("detected_roi", "--sampling-mode", help="Sampling mode: full_frame, center_crop, or detected_roi"),
    center_crop_width_ratio: float = typer.Option(0.3, "--center-crop-width-ratio", help="Normalized center-crop width ratio"),
    center_crop_height_ratio: float = typer.Option(0.3, "--center-crop-height-ratio", help="Normalized center-crop height ratio"),
    backend: str = typer.Option("mock", "--backend", help="Backend: mock or red"),
) -> None:
    payload = calibrate_exposure_path(
        input_path,
        out_dir=out,
        source=source,
        roi_file=roi_file,
        target_log2_luminance=target_log2,
        reference_camera=reference_camera,
        sampling_mode=sampling_mode,
        center_crop_width_ratio=center_crop_width_ratio,
        center_crop_height_ratio=center_crop_height_ratio,
        backend=backend,
    )
    typer.echo(str(payload))


@app.command("calibrate-color")
def calibrate_color_command(
    input_path: str,
    out: str = typer.Option(..., "--out", help="Color calibration output directory"),
    roi_file: Optional[str] = typer.Option(None, "--roi-file", help="Optional manual ROI override"),
    sampling_mode: str = typer.Option("detected_roi", "--sampling-mode", help="Sampling mode: full_frame, center_crop, or detected_roi"),
    center_crop_width_ratio: float = typer.Option(0.3, "--center-crop-width-ratio", help="Normalized center-crop width ratio"),
    center_crop_height_ratio: float = typer.Option(0.3, "--center-crop-height-ratio", help="Normalized center-crop height ratio"),
    backend: str = typer.Option("mock", "--backend", help="Backend: mock or red"),
) -> None:
    payload = calibrate_color_path(
        input_path,
        out_dir=out,
        roi_file=roi_file,
        sampling_mode=sampling_mode,
        center_crop_width_ratio=center_crop_width_ratio,
        center_crop_height_ratio=center_crop_height_ratio,
        backend=backend,
    )
    typer.echo(str(payload))


@app.command("calibrate-card")
def calibrate_card_command(
    input_path: str,
    out: str = typer.Option(..., "--out", help="Calibration output directory"),
    roi_file: Optional[str] = typer.Option(None, "--roi-file", help="Optional manual ROI override mapping clip IDs or camera groups to {x, y, width, height}"),
    target_log2: Optional[float] = typer.Option(None, "--target-log2", help="Target log2 luminance for the gray card"),
    reference_camera: Optional[str] = typer.Option(None, "--reference-camera", help="Reference camera group or clip ID to define the target"),
    backend: str = typer.Option("mock", "--backend", help="Backend: mock or red"),
) -> None:
    payload = calibrate_card_path(
        input_path,
        out_dir=out,
        roi_file=roi_file,
        target_log2_luminance=target_log2,
        reference_camera=reference_camera,
        backend=backend,
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


@app.command("validate-pipeline")
def validate_pipeline_command(
    input_path: str,
    analysis_dir: str = typer.Option(..., "--analysis-dir", help="Analysis output directory"),
    out: str = typer.Option(..., "--out", help="Validation output directory"),
    exposure_calibration: Optional[str] = typer.Option(None, "--exposure-calibration", help="Optional exposure calibration JSON"),
    color_calibration: Optional[str] = typer.Option(None, "--color-calibration", help="Optional color calibration JSON"),
    redline_executable: str = typer.Option("REDLine", "--redline-executable", help="REDLine executable"),
    output_ext: str = typer.Option("mov", "--output-ext", help="Rendered output extension"),
) -> None:
    payload = validate_pipeline(
        input_path,
        analysis_dir=analysis_dir,
        exposure_calibration_path=exposure_calibration,
        color_calibration_path=color_calibration,
        out_dir=out,
        redline_executable=redline_executable,
        output_ext=output_ext,
    )
    typer.echo(str(payload))


@app.command("report-contact-sheet")
def report_contact_sheet_command(
    input_path: str,
    out: str = typer.Option(..., "--out", help="Report output directory"),
    exposure_calibration: Optional[str] = typer.Option(None, "--exposure-calibration", help="Optional exposure calibration JSON"),
    color_calibration: Optional[str] = typer.Option(None, "--color-calibration", help="Optional color calibration JSON"),
) -> None:
    payload = build_contact_sheet_report(
        input_path,
        out_dir=out,
        exposure_calibration_path=exposure_calibration,
        color_calibration_path=color_calibration,
    )
    typer.echo(str(payload))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
