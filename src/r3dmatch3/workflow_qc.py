"""
workflow_qc.py — Sphere QC re-measurement
==========================================
Called after the initial run_analysis() completes, when the operator
has corrected sphere positions in the Sphere QC screen.

remeasure_cameras() takes the existing RunResult and a dict of corrected
SphereROIs, re-runs detection validation + measurement for only those
cameras, then re-solves exposure and WB across the full array.

The RunResult is mutated in-place so the Results screen can just re-render.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional

from .models import CameraResult, RunResult, SphereROI
from .sphere import validate_manual_roi, load_render_as_hwc
from .measure import measure_render
from .solve import (
    assess_run,
    build_commit_values,
    solve_exposure,
    solve_white_balance,
    GRAY_ANCHOR_IRE_LOG3G10,
)
from .sphere_profile import (
    load_project_profile,
    save_project_profile,
    manual_sample_photometrics,
    record_detection,
    project_id_from_path,
)
from . import progress as prog


def remeasure_cameras(
    run_result: RunResult,
    corrected_rois: Dict[str, SphereROI],
    *,
    progress_callback=None,
) -> RunResult:
    """
    Re-measure specific cameras with operator-corrected sphere ROIs,
    then re-solve the full array.

    Parameters
    ----------
    run_result : RunResult
        The existing result from run_analysis(). Mutated in-place.
    corrected_rois : dict[clip_id → SphereROI]
        Operator-confirmed ROIs from the Sphere QC screen.
    progress_callback : optional callable(phase, detail, clip_id=None)
        Called with progress updates for the UI.

    Returns
    -------
    RunResult (same object, mutated)
    """
    def _cb(phase: str, detail: str, clip_id: str = ""):
        if progress_callback:
            try:
                progress_callback(phase, detail, clip_id)
            except Exception:
                pass
        prog.emit(phase, pct=0, detail=detail,
                  **{"clip_id": clip_id} if clip_id else {})

    if not corrected_rois:
        return run_result

    project_id   = project_id_from_path(run_result.input_path)
    sphere_profile = load_project_profile(project_id)

    # Build a lookup of existing CameraResults by clip_id
    by_clip: Dict[str, CameraResult] = {cr.clip_id: cr for cr in run_result.cameras}

    # --- Re-detect and re-measure each corrected camera ---
    for clip_id, roi in corrected_rois.items():
        cr = by_clip.get(clip_id)
        if cr is None:
            _cb("qc_skip", f"Unknown clip_id {clip_id}", clip_id)
            continue

        render_path = (
            cr.measurement.render_path if cr.measurement
            else cr.original_render_path
        )
        if not render_path or not Path(render_path).exists():
            _cb("qc_error", f"Render not found for {clip_id}", clip_id)
            cr.failed_stage = "qc_render_missing"
            cr.failure_reason = f"Render not found: {render_path}"
            cr.status = "ERROR"
            continue

        _cb("qc_detect", f"Validating manual ROI for {clip_id}", clip_id)
        try:
            image_hwc = load_render_as_hwc(render_path)
        except Exception as exc:
            _cb("qc_error", f"Could not load render for {clip_id}: {exc}", clip_id)
            cr.failed_stage = "qc_load_error"
            cr.failure_reason = str(exc)
            cr.status = "ERROR"
            continue

        # Validate manual ROI through gates 2-5
        detection = validate_manual_roi(image_hwc, roi, clip_id=clip_id)
        cr.detection = detection

        if not detection.success:
            # Manual ROIs get through gate 1 automatically — failure here
            # means gates 2-5 failed (not gray, not Lambertian, etc).
            # Still accept it — operator override — but mark as REVIEW.
            # Use MANUAL status so the report shows operator source.
            from .models import SphereDetectionResult
            detection = SphereDetectionResult(
                clip_id=clip_id,
                status="MANUAL",
                roi=roi,
                source="manual_operator",
                gates=detection.gates,
                failed_gate=detection.failed_gate,
                failure_reason=f"operator_override: {detection.failure_reason}",
            )
            cr.detection = detection

        _cb("qc_measure", f"Measuring {clip_id}", clip_id)
        try:
            measurement = measure_render(
                render_path,
                roi,
                clip_id=clip_id,
                detection=detection,
                roi_source="manual_operator",
            )
        except Exception as exc:
            _cb("qc_error", f"Measurement failed for {clip_id}: {exc}", clip_id)
            cr.failed_stage = "qc_measurement_error"
            cr.failure_reason = str(exc)
            cr.status = "ERROR"
            continue

        # Scene-linear measurement at the NEW ROI — without this the re-measured
        # camera loses hero_log2_lin and the array re-solve silently falls back to
        # the display-space exposure solve (which undershoots). Reuses the linear
        # render produced in the main run; no re-render needed.
        lin_path = getattr(cr, "linear_render_path", None)
        if lin_path and measurement.measurement_valid:
            from .measure import measure_center_log2
            measurement.hero_log2_lin = measure_center_log2(lin_path, roi)

        cr.measurement = measurement
        cr.failed_stage = ""
        cr.failure_reason = ""

        if not measurement.measurement_valid:
            cr.status = "NO_DATA"
            cr.failed_stage = "qc_measurement_invalid"
            cr.failure_reason = measurement.validity_reason
        else:
            cr.status = "SOLVED"

        # Record into sphere profile
        if cr.metadata:
            photometrics = manual_sample_photometrics(
                detection=detection,
                measurement=measurement,
            )
            sphere_profile = record_detection(
                sphere_profile,
                clip_id=clip_id,
                camera_label=cr.camera_label,
                run_id=run_result.run_id,
                roi_cx=roi.cx,
                roi_cy=roi.cy,
                roi_r=roi.r,
                frame_width=cr.metadata.frame_width or measurement.render_width,
                frame_height=cr.metadata.frame_height or measurement.render_height,
                ire_spread=photometrics["ire_spread"],
                chroma_distance=photometrics["chroma_distance"],
                lambertian_score=photometrics["lambertian_score"],
                interior_lum_mean=photometrics["interior_lum_mean"],
                interior_lum_stddev=photometrics["interior_lum_stddev"],
                hero_ire=photometrics["hero_ire"],
                trust="verified_manual",
            )

        _cb("qc_done", f"{clip_id}: {measurement.hero_ire:.1f} IRE", clip_id)

    # --- Re-solve the full array ---
    _cb("qc_solve", "Re-solving exposure and WB across array")

    valid = [cr.measurement for cr in run_result.cameras
             if cr.measurement and cr.measurement.measurement_valid]

    if not valid:
        _cb("qc_error", "No valid measurements — cannot re-solve")
        return run_result

    # Preserve the run's exposure strategy across the QC re-solve (else the
    # absolute-gray anchor would silently revert to median).
    _strategy = getattr(run_result, "anchor_source", "median") or "median"
    _gray_ire = getattr(run_result, "gray_target_ire", 0.0) or GRAY_ANCHOR_IRE_LOG3G10
    # Exposure requires a scene-linear measurement per camera. The QC re-measure
    # reuses the main-run linear render, so this only trips if a re-measured ROI
    # couldn't produce a linear value — surface it rather than crashing the UI.
    try:
        if _strategy == "gray_anchor":
            target_log2, target_ire, spread, per_clip_exp = solve_exposure(
                valid, strategy="gray_anchor", gray_target_ire=_gray_ire)
        else:
            target_log2, target_ire, spread, per_clip_exp = solve_exposure(valid)
    except ValueError as exc:
        _cb("qc_error", f"Exposure re-solve failed: {exc}")
        return run_result
    _cb("qc_solve", "Exposure re-solved — scene-linear")

    as_shot_kelvins = {
        cr.clip_id: cr.metadata.kelvin
        for cr in run_result.cameras
        if cr.metadata
    }
    wb_result, per_clip_wb = solve_white_balance(valid, as_shot_kelvins)

    # Patch commit values on all usable cameras
    for cr in run_result.cameras:
        if not cr.is_usable():
            continue
        exp_offset = per_clip_exp.get(cr.clip_id, 0.0)
        kelvin, tint = per_clip_wb.get(cr.clip_id, (5600, 0.0))
        m = cr.measurement
        wc_before = gm_before = 0.0
        if m:
            from .measure import compute_wc_gm
            wc_before, gm_before = compute_wc_gm(m.measured_rgb_mean)
        cr.commit = build_commit_values(
            cr,
            exposure_offset=exp_offset,
            kelvin=kelvin,
            tint=tint,
            wc_before=wc_before,
            gm_before=gm_before,
        )
        # Re-solve invalidates previous closed-loop verification: commits
        # changed, so match % is unknown until corrected renders are re-measured.
        cr.exposure_closed_loop_status = ""
        cr.wb_closed_loop_status = wb_result.status
        cr.exposure_match_pct = None
        cr.wb_match_pct = None
        cr.match_pct = None

    # Mark corrected cameras SOLVED if usable
    for clip_id in corrected_rois:
        cr = by_clip.get(clip_id)
        if cr and cr.is_usable():
            cr.status = "SOLVED"

    # Re-assess
    assessment = assess_run(run_result.cameras)
    run_result.exposure_spread = spread
    run_result.wb_solve        = wb_result
    run_result.anchor_ire      = target_ire
    run_result.anchor_log2     = target_log2
    run_result.assessment_status   = assessment["assessment_status"]
    run_result.array_match_pct     = assessment["array_match_pct"]
    run_result.min_match_pct       = assessment["min_match_pct"]
    run_result.min_match_clip_id   = assessment["min_match_clip_id"]
    run_result.solved_count        = assessment["solved_count"]
    run_result.scored_count        = assessment["scored_count"]
    run_result.needs_assist_count  = assessment["needs_assist_count"]
    run_result.no_data_count       = assessment["no_data_count"]

    # Save sphere profile
    save_project_profile(project_id, sphere_profile)
    _cb("qc_complete",
        f"Re-solve done. {assessment['solved_count']}/{len(run_result.cameras)} solved "
        f"(match % pending corrected-render verification)")

    return run_result
