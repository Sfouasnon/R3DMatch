from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from .calibration import calibrate_card_path, calibrate_color_path, calibrate_exposure_path, calibrate_sphere_path
from .execution import CancellationError
from .matching import analyze_path
from .report import build_contact_sheet_report, normalize_target_strategy_name
from .rmd import write_rmds_from_analysis
from .transcode import write_transcode_plan
from .validation import validate_pipeline
from .workflow import approve_master_rmd, clear_preview_cache, normalize_matching_domain, review_calibration

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
    calibration_mode: Optional[str] = typer.Option(None, "--calibration-mode", help="Optional calibration mode, e.g. array-gray-sphere"),
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
        calibration_mode=calibration_mode,
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
    use_generated_rmd: bool = typer.Option(False, "--use-generated-rmd", help="Generate and use .RMD files from the analysis output"),
    redline_executable: str = typer.Option("REDLine", "--redline-executable", help="REDLine executable"),
    output_ext: str = typer.Option("mov", "--output-ext", help="Rendered output extension"),
    execute: bool = typer.Option(False, "--execute", help="Execute commands instead of just writing plans"),
) -> None:
    payload = write_transcode_plan(
        input_path,
        out_dir=out,
        analysis_dir=analysis_dir,
        use_generated_sidecar=use_generated_sidecar,
        use_generated_rmd=use_generated_rmd,
        redline_executable=redline_executable,
        output_ext=output_ext,
        execute=execute,
    )
    typer.echo(str(payload))


@app.command("write-rmd")
def write_rmd_command(
    input_path: str,
    analysis_dir: str = typer.Option(..., "--analysis-dir", help="Analysis directory that contains generated sidecars"),
    out: Optional[str] = typer.Option(None, "--out", help="Optional RMD output directory; defaults to analysis_dir/rmd"),
) -> None:
    del input_path
    payload = write_rmds_from_analysis(analysis_dir, out_dir=out)
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
    target_type: Optional[str] = typer.Option(None, "--target-type", help="Optional target type: gray_card, gray_sphere, or color_chart"),
    processing_mode: Optional[str] = typer.Option(None, "--processing-mode", help="Optional processing mode: exposure, color, or both"),
    preview_mode: str = typer.Option("calibration", "--preview-mode", help="Preview mode: calibration or monitoring"),
    preview_output_space: Optional[str] = typer.Option(None, "--preview-output-space", help="Preview output color space"),
    preview_output_gamma: Optional[str] = typer.Option(None, "--preview-output-gamma", help="Preview output gamma"),
    preview_highlight_rolloff: Optional[str] = typer.Option(None, "--preview-highlight-rolloff", help="Preview highlight rolloff"),
    preview_shadow_rolloff: Optional[str] = typer.Option(None, "--preview-shadow-rolloff", help="Preview shadow rolloff"),
    preview_lut: Optional[str] = typer.Option(None, "--preview-lut", help="Optional monitoring LUT (.cube)"),
) -> None:
    payload = build_contact_sheet_report(
        input_path,
        out_dir=out,
        exposure_calibration_path=exposure_calibration,
        color_calibration_path=color_calibration,
        target_type=target_type,
        processing_mode=processing_mode,
        preview_mode=preview_mode,
        preview_output_space=preview_output_space,
        preview_output_gamma=preview_output_gamma,
        preview_highlight_rolloff=preview_highlight_rolloff,
        preview_shadow_rolloff=preview_shadow_rolloff,
        preview_lut=preview_lut,
    )
    typer.echo(str(payload))


@app.command("review-calibration")
def review_calibration_command(
    input_path: str,
    out: str = typer.Option(..., "--out", help="Review output directory"),
    run_label: Optional[str] = typer.Option(None, "--run-label", help="Optional subset/run label; creates a nested run folder under --out when provided"),
    target_type: str = typer.Option(..., "--target-type", help="Target type: gray_card, gray_sphere, or color_chart"),
    processing_mode: str = typer.Option("both", "--processing-mode", help="Processing mode: exposure, color, or both"),
    mode: str = typer.Option("scene", "--mode", help="Matching mode: scene or view"),
    matching_domain: str = typer.Option("scene", "--matching-domain", help="Matching domain: scene or perceptual"),
    lut: Optional[str] = typer.Option(None, "--lut", help="Optional LUT override (.cube)"),
    calibration: Optional[str] = typer.Option(None, "--calibration", help="Optional legacy/single exposure calibration JSON"),
    exposure_calibration: Optional[str] = typer.Option(None, "--exposure-calibration", help="Optional exposure calibration JSON"),
    color_calibration: Optional[str] = typer.Option(None, "--color-calibration", help="Optional color calibration JSON"),
    calibration_mode: Optional[str] = typer.Option(None, "--calibration-mode", help="Optional calibration mode, e.g. array-gray-sphere"),
    backend: str = typer.Option("mock", "--backend", help="Backend: mock or red"),
    sample_count: int = typer.Option(8, "--sample-count", help="Number of sampled frames per clip"),
    sampling_strategy: str = typer.Option("uniform", "--sampling-strategy", help="Sampling strategy: uniform or head"),
    roi_x: Optional[float] = typer.Option(None, "--roi-x", help="Shared normalized ROI X origin"),
    roi_y: Optional[float] = typer.Option(None, "--roi-y", help="Shared normalized ROI Y origin"),
    roi_w: Optional[float] = typer.Option(None, "--roi-w", help="Shared normalized ROI width"),
    roi_h: Optional[float] = typer.Option(None, "--roi-h", help="Shared normalized ROI height"),
    clip_id: list[str] = typer.Option([], "--clip-id", help="Repeat to include only specific clip IDs in this calibration subset"),
    clip_group: list[str] = typer.Option([], "--clip-group", help="Repeat to include all clips from a discovered subset group (for example take number)"),
    clip_subset_file: Optional[str] = typer.Option(None, "--clip-subset-file", help="Optional JSON file describing clip_ids / clip_groups / run_label for this subset run"),
    target_strategy: list[str] = typer.Option(["median"], "--target-strategy", help="Target strategy: median, brightest-valid, manual, or hero-camera; repeat to compare multiple"),
    reference_clip_id: Optional[str] = typer.Option(None, "--reference-clip-id", help="Reference clip ID for manual target strategy"),
    hero_clip_id: Optional[str] = typer.Option(None, "--hero-clip-id", help="Hero clip ID for hero-camera target strategy"),
    preview_mode: str = typer.Option("calibration", "--preview-mode", help="Preview mode: calibration or monitoring"),
    preview_output_space: Optional[str] = typer.Option(None, "--preview-output-space", help="Preview output color space"),
    preview_output_gamma: Optional[str] = typer.Option(None, "--preview-output-gamma", help="Preview output gamma"),
    preview_highlight_rolloff: Optional[str] = typer.Option(None, "--preview-highlight-rolloff", help="Preview highlight rolloff"),
    preview_shadow_rolloff: Optional[str] = typer.Option(None, "--preview-shadow-rolloff", help="Preview shadow rolloff"),
    preview_lut: Optional[str] = typer.Option(None, "--preview-lut", help="Optional monitoring LUT (.cube)"),
) -> None:
    calibration_roi = None
    roi_values = [roi_x, roi_y, roi_w, roi_h]
    if any(value is not None for value in roi_values):
        if not all(value is not None for value in roi_values):
            raise typer.BadParameter("roi-x, roi-y, roi-w, and roi-h must be provided together")
        if not all(0.0 <= float(value) <= 1.0 for value in roi_values):
            raise typer.BadParameter("ROI values must be normalized between 0.0 and 1.0")
        if float(roi_x) + float(roi_w) > 1.0 or float(roi_y) + float(roi_h) > 1.0:
            raise typer.BadParameter("Normalized ROI must remain inside the image bounds")
        calibration_roi = {"x": float(roi_x), "y": float(roi_y), "w": float(roi_w), "h": float(roi_h)}
    normalized_strategies = [normalize_target_strategy_name(item) for item in target_strategy]
    if "manual" in normalized_strategies and not reference_clip_id:
        raise typer.BadParameter("manual target strategy requires --reference-clip-id")
    if "hero_camera" in normalized_strategies and not hero_clip_id:
        raise typer.BadParameter("hero-camera target strategy requires --hero-clip-id")
    try:
        resolved_matching_domain = normalize_matching_domain(matching_domain)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    payload = review_calibration(
        input_path,
        out_dir=out,
        run_label=run_label,
        target_type=target_type,
        processing_mode=processing_mode,
        mode=mode,
        matching_domain=resolved_matching_domain,
        backend=backend,
        lut_override=lut,
        calibration_path=calibration,
        exposure_calibration_path=exposure_calibration,
        color_calibration_path=color_calibration,
        calibration_mode=calibration_mode,
        sample_count=sample_count,
        sampling_strategy=sampling_strategy,
        calibration_roi=calibration_roi,
        selected_clip_ids=clip_id,
        selected_clip_groups=clip_group,
        clip_subset_file=clip_subset_file,
        target_strategies=target_strategy,
        reference_clip_id=reference_clip_id,
        hero_clip_id=hero_clip_id,
        preview_mode=preview_mode,
        preview_output_space=preview_output_space,
        preview_output_gamma=preview_output_gamma,
        preview_highlight_rolloff=preview_highlight_rolloff,
        preview_shadow_rolloff=preview_shadow_rolloff,
        preview_lut=preview_lut,
    )
    typer.echo(str(payload))


@app.command("approve-master-rmd")
def approve_master_rmd_command(
    analysis_dir: str,
    out: Optional[str] = typer.Option(None, "--out", help="Approval output directory; defaults to <analysis_dir>/approval"),
    target_strategy: str = typer.Option("median", "--target-strategy", help="Chosen target strategy: median, brightest-valid, manual, or hero-camera"),
    reference_clip_id: Optional[str] = typer.Option(None, "--reference-clip-id", help="Reference clip ID for manual target strategy"),
    hero_clip_id: Optional[str] = typer.Option(None, "--hero-clip-id", help="Hero clip ID for hero-camera target strategy"),
) -> None:
    normalized_strategy = normalize_target_strategy_name(target_strategy)
    if normalized_strategy == "manual" and not reference_clip_id:
        raise typer.BadParameter("manual target strategy requires --reference-clip-id")
    if normalized_strategy == "hero_camera" and not hero_clip_id:
        raise typer.BadParameter("hero-camera target strategy requires --hero-clip-id")
    payload = approve_master_rmd(
        analysis_dir,
        out_dir=out,
        target_strategy=target_strategy,
        reference_clip_id=reference_clip_id,
        hero_clip_id=hero_clip_id,
    )
    typer.echo(str(payload))


@app.command("clear-preview-cache")
def clear_preview_cache_command(
    input_path: str,
    report_dir: Optional[str] = typer.Option(None, "--report-dir", help="Optional report directory; defaults to <input_path>/report"),
) -> None:
    payload = clear_preview_cache(input_path, report_dir=report_dir)
    typer.echo(str(payload))


@app.command("desktop-ui")
def desktop_ui_command() -> None:
    from .desktop_app import launch_desktop_ui

    launch_desktop_ui(str(Path.cwd()))


def main() -> None:
    try:
        app()
    except CancellationError as exc:
        typer.echo(str(exc))
        raise typer.Exit(code=130) from exc
    except RuntimeError as exc:
        typer.echo(f"ERROR: {exc}")
        raise typer.Exit(code=1) from exc


if __name__ == "__main__":
    main()
