"""
R3DMatch v2 — Workflow Orchestration

Coordinates the full pipeline:
  1. Scan — find R3D clips under input_path
  2. Metadata — read each clip via --printMeta 1 (+ optional --printMeta 5)
  3. Render — REDLine renders measurement TIFF per clip
  4. Detect — auto sphere detection (hard gates) OR validate manual ROI
  5. Measure — four-step measurement math
  6. Solve — exposure target + WB model
  7. Render corrected — re-render with commit values for closed-loop
  8. Verify — check corrected render against targets
  9. Report — generate contact sheet JSON + HTML

Each phase emits structured JSON progress on stdout.
"""
from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import progress as prog
from .measure import (measure_render, measure_center_log2, array_target_log2,
                      exposure_offset_stops, compute_wc_gm)
from .models import (
    CameraResult,
    ClipMetadata,
    CommitValues,
    ExposureSpread,
    RunResult,
    SphereDetectionResult,
    SphereROI,
)
from .redline import (
    read_clip_metadata,
    render_measurement_frame,
    render_measurement_frame_retried,
    resolve_redline_executable,
    check_redline_available,
    _resolve_output_path,
)
from .solve import (
    assess_run,
    build_commit_values,
    camera_match_pct,
    exposure_match_pct,
    exposure_residual_stops,
    solve_exposure,
    solve_white_balance,
    verify_wb_closed_loop,
    wb_match_pct,
    GRAY_ANCHOR_IRE_LOG3G10,
)
from .colorpipeline import ColorPipeline, SCENE_LINEAR_PIPELINE
from .match_export import write_match_export
from .measure_delivery import measure_delivery_hero
from .pipeline_profile import build_profile, luma_weights_for, CharSample
from .sphere import detect_sphere, validate_manual_roi, load_render_as_hwc
from .sphere_profile import (
    load_project_profile,
    save_project_profile,
    record_detection,
    get_camera_prior,
    manual_sample_photometrics,
    apply_prior_bonus_detect_scale,
    project_id_from_path,
    profile_summary,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# R3D clip discovery
# ---------------------------------------------------------------------------

def discover_clips(input_path: str) -> List[Path]:
    """
    Find all valid R3D clips under input_path.
    Only first-span (*_001.R3D) clips, deduplicated by clip_id.
    Fixes the duplicate bug where the same clip appeared twice.
    """
    root = Path(input_path).expanduser().resolve()
    seen_ids: set = set()
    clips: List[Path] = []
    for candidate in sorted(root.rglob("*.R3D")):
        name = candidate.name
        if name.startswith(".") or name.startswith("._"):
            continue
        if "_001.R3D" not in name.upper():
            continue
        clip_id = candidate.stem
        if clip_id in seen_ids:
            continue
        seen_ids.add(clip_id)
        clips.append(candidate)
    return clips

def clip_id_from_path(path: str) -> str:
    """Extract clip_id from R3D path. e.g. G007_A106_0511R9_001"""
    stem = Path(path).stem  # G007_A106_0511R9_001
    return stem


def camera_label_from_clip_id(clip_id: str) -> str:
    """e.g. G007_A106_0511R9_001 → G007_A106"""
    parts = clip_id.split("_")
    if len(parts) >= 2:
        return f"{parts[0]}_{parts[1]}"
    return parts[0]


# ---------------------------------------------------------------------------
# Manual ROI loading
# ---------------------------------------------------------------------------

def load_manual_rois(roi_file: Optional[str]) -> Dict[str, SphereROI]:
    """
    Load manual ROIs from a JSON file.
    Format: {"clip_id": {"cx": ..., "cy": ..., "r": ...}, ...}
    """
    if not roi_file:
        return {}
    path = Path(roi_file).expanduser().resolve()
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return {
            str(k): SphereROI.from_dict(v)
            for k, v in payload.items()
            if all(key in v for key in ("cx", "cy", "r"))
        }
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Main run
# ---------------------------------------------------------------------------

def run_analysis(
    input_path: str,
    *,
    out_dir: str,
    run_id: Optional[str] = None,
    manual_roi_file: Optional[str] = None,
    reuse_renders: bool = True,
    render_corrected: bool = True,
    read_lens_metadata: bool = True,
    disable_priors: bool = False,
    delivery_pipeline: Optional[ColorPipeline] = None,
    wb_mode: str = "match",
    strategy: str = "median",
    gray_target_ire: float = GRAY_ANCHOR_IRE_LOG3G10,
) -> RunResult:
    """
    Full R3DMatch analysis run.

    Returns RunResult with all per-camera results and the overall solve.
    Also writes JSON files to out_dir.
    """
    started_at = time.perf_counter()
    created_at = datetime.now(timezone.utc).isoformat()
    run_id = run_id or f"run_{int(time.time())}"
    if disable_priors:
        log.warning(
            "disable_priors=True — prior injection suppressed for all cameras (validation mode)"
        )
    project_id = project_id_from_path(input_path)
    out_root = Path(out_dir).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    analysis_dir = out_root / "analysis"
    previews_dir = out_root / "previews"
    measurement_dir = previews_dir / "_measurement"
    analysis_dir.mkdir(exist_ok=True)
    previews_dir.mkdir(exist_ok=True)
    measurement_dir.mkdir(exist_ok=True)

    # --- Phase 1: Startup ---
    prog.emit("startup", pct=0, detail="Checking REDLine availability")
    try:
        redline = resolve_redline_executable()
        rl_check = check_redline_available(redline)
        if not rl_check["ready"]:
            raise RuntimeError(f"REDLine not ready: {rl_check['error']}")
    except Exception as exc:
        prog.emit("error", pct=0, detail=str(exc), error=True)
        raise

    # --- Load sphere profile ---
    sphere_profile = load_project_profile(project_id)
    prog.emit("profile_loaded", pct=1,
              detail=f"Sphere profile: {sum(len(e.get('samples',[])) for e in sphere_profile.get('cameras',{}).values())} samples across {len(sphere_profile.get('cameras',{}))} cameras")

    # --- Phase 2: Discover clips ---
    prog.emit("scan", pct=2, detail="Scanning for R3D clips")
    clips = discover_clips(input_path)
    if not clips:
        raise ValueError(f"No R3D clips found under {input_path}")
    prog.emit("scan_complete", pct=5, detail=f"Found {len(clips)} clip(s)", clip_count=len(clips))

    # --- Load manual ROIs ---
    manual_rois = load_manual_rois(manual_roi_file)

    # --- Phase 3: Parallel renders, then serial detect + measure ---
    camera_results: List[CameraResult] = []
    n = len(clips)

    # ── 3a+3b: Metadata + Render in parallel ─────────────────────────────────
    # Build per-clip stubs and kick off all renders concurrently.
    # REDLine is CPU-bound and independent per clip — parallel renders cut
    # wall-clock time by ~(N-1)/N for N cameras.
    # MAX_RENDER_WORKERS: conservative — each REDLine process is heavy.
    MAX_RENDER_WORKERS = min(4, n)

    # Per-clip state accumulated during parallel phase
    RenderSlot = Dict  # keys: clip_id, result, meta, render_path, error

    def _render_one_clip(clip_path: Path, index: int) -> RenderSlot:
        """Run metadata + render for one clip. Returns a slot dict."""
        clip_id = clip_id_from_path(str(clip_path))
        camera_label = camera_label_from_clip_id(clip_id)
        slot: RenderSlot = {
            "clip_path": clip_path,
            "clip_id": clip_id,
            "camera_label": camera_label,
            "index": index,
            "meta": None,
            "render_path": None,
            "linear_render_path": None,   # scene-linear render for exposure solve
            "render_status": "OK",
            "error_stage": None,
            "error_msg": None,
        }

        # Metadata
        try:
            meta = read_clip_metadata(str(clip_path), redline=redline,
                                      read_lens=read_lens_metadata)
            slot["meta"] = meta
        except Exception as exc:
            slot["error_stage"] = "metadata"
            slot["error_msg"] = str(exc)
            return slot

        # Render
        render_out = measurement_dir / f"{clip_id}.original.analysis.measurement.tiff"
        render_actual = _resolve_output_path(str(render_out))
        if reuse_renders and render_actual.exists() and render_actual.stat().st_size > 10_000:
            slot["render_path"] = str(render_actual)
            slot["reused"] = True
        else:
            slot["reused"] = False
            render_result = render_measurement_frame(
                str(clip_path), str(render_out), redline=redline,
                frame_index=0, use_as_shot=True,
            )
            if not render_result["ok"]:
                slot["error_stage"] = "render"
                slot["error_msg"] = render_result["stderr"]
                slot["render_status"] = render_result.get("status", "RENDER_ERROR")
                return slot
            slot["render_path"] = render_result["output_path"]
            slot["render_status"] = render_result.get("status", "OK")

        # Scene-linear render for the EXPOSURE solve (separate from the display
        # reference render above). REQUIRED: the exposure solve runs in
        # scene-linear for accuracy, so a persistent failure here is recorded and
        # the run hard-fails at the solve boundary rather than silently degrading.
        # Retried to absorb transient GPU/IO contention under parallel rendering.
        try:
            lin_out = measurement_dir / f"{clip_id}.linear.measurement.tiff"
            lin_actual = _resolve_output_path(str(lin_out))
            if reuse_renders and lin_actual.exists() and lin_actual.stat().st_size > 10_000:
                slot["linear_render_path"] = str(lin_actual)
            else:
                lin_result = render_measurement_frame_retried(
                    str(clip_path), str(lin_out), redline=redline,
                    frame_index=0, use_as_shot=True, pipeline=SCENE_LINEAR_PIPELINE,
                )
                if lin_result.get("ok"):
                    slot["linear_render_path"] = lin_result["output_path"]
                else:
                    slot["linear_render_error"] = (
                        f"{lin_result.get('status', 'RENDER_ERROR')}: "
                        f"{lin_result.get('stderr', '')}"[:400]
                    )
        except Exception as exc:
            slot["linear_render_error"] = f"exception: {exc}"[:400]
            log.warning("Scene-linear render failed for %s: %s", clip_id, exc)

        return slot

    # Emit scan-start progress, then fire all renders
    prog.emit("render_start", pct=5,
              detail=f"Rendering {n} clips ({MAX_RENDER_WORKERS} parallel)")

    # Ordered results — preserve clip order for deterministic solve
    slots_by_id: Dict[str, RenderSlot] = {}
    futures_map = {}

    with ThreadPoolExecutor(max_workers=MAX_RENDER_WORKERS) as executor:
        for i, clip_path in enumerate(clips):
            clip_id = clip_id_from_path(str(clip_path))
            pct = prog.emit_phase_pct(i, n, 5, 40)
            prog.emit("clip_start", pct=pct,
                      detail=f"Processing {clip_id}",
                      clip_id=clip_id, clip_index=i + 1, clip_count=n)
            fut = executor.submit(_render_one_clip, clip_path, i)
            futures_map[fut] = {
                "clip_id": clip_id,
                "index": i,
                "clip_path": clip_path,
                "camera_label": camera_label_from_clip_id(clip_id),
            }

        for fut in as_completed(futures_map):
            ctx = futures_map[fut]
            try:
                slot = fut.result()
            except Exception:
                clip_id = ctx["clip_id"]
                log.exception("Render worker crashed for clip %s", clip_id)
                slot = {
                    "clip_path": ctx["clip_path"],
                    "clip_id": clip_id,
                    "camera_label": ctx["camera_label"],
                    "index": ctx["index"],
                    "meta": None,
                    "render_path": None,
                    "render_status": "RENDER_ERROR",
                    "error_stage": "render",
                    "error_msg": "Render worker crashed; see log for traceback.",
                }
            if slot is None:
                clip_id = ctx["clip_id"]
                log.error("Render worker returned no result for clip %s", clip_id)
                slot = {
                    "clip_path": ctx["clip_path"],
                    "clip_id": clip_id,
                    "camera_label": ctx["camera_label"],
                    "index": ctx["index"],
                    "meta": None,
                    "render_path": None,
                    "render_status": "RENDER_ERROR",
                    "error_stage": "render",
                    "error_msg": "Render worker returned no result.",
                }
            clip_id = slot["clip_id"]
            slots_by_id[clip_id] = slot
            pct = prog.emit_phase_pct(slot["index"], n, 5, 40)
            # Metadata is already read at this point — ship it to the UI
            # so FPS/resolution populate live instead of after the run.
            meta = slot.get("meta")
            meta_kw = {}
            if meta is not None:
                if getattr(meta, "fps", None):
                    meta_kw["clip_fps"] = f"{meta.fps:.2f}".rstrip("0").rstrip(".")
                if getattr(meta, "frame_width", 0) and getattr(meta, "frame_height", 0):
                    meta_kw["clip_res"] = f"{meta.frame_width}×{meta.frame_height}"
            if slot["error_stage"]:
                prog.emit("clip_error", pct=pct,
                          detail=f"{slot['error_stage']} failed: {slot['error_msg'][:120]}",
                          clip_id=clip_id, error=True, **meta_kw)
            elif slot.get("reused"):
                prog.emit("clip_render_reused", pct=pct,
                          detail=f"Render reused for {clip_id}", clip_id=clip_id, **meta_kw)
            else:
                prog.emit("clip_render", pct=pct,
                          detail=f"Rendered {clip_id}", clip_id=clip_id, **meta_kw)

    # ── 3c+3d: Detect + Measure (serial, in original clip order) ─────────────
    # Accumulate hero IRE from confirmed detections as cameras complete.
    # Passed into detect_sphere so the IRE context gate uses the run's own
    # observed lighting range rather than the wide pipeline fallback rails.
    peer_ire_values: List[float] = []

    for i, clip_path in enumerate(clips):
        clip_id = clip_id_from_path(str(clip_path))
        camera_label = camera_label_from_clip_id(clip_id)
        pct_base = prog.emit_phase_pct(i, n, 40, 75)
        slot = slots_by_id.get(clip_id)

        result = CameraResult(
            clip_id=clip_id,
            camera_label=camera_label,
            source_path=str(clip_path),
            metadata=slot.get("meta") if slot else None,
            detection=None,
            measurement=None,
            commit=None,
        )

        if slot is None:
            log.warning("Skipping detect for %s: render worker produced no slot", clip_id)
            result.detection = SphereDetectionResult(
                clip_id=clip_id,
                status="FAILED",
                roi=None,
                source="render_error",
                failed_gate="render_error",
                failure_reason="Render worker produced no slot.",
            )
            result.failed_stage = "render"
            result.failure_reason = result.detection.failure_reason
            result.status = "ERROR"
            camera_results.append(result)
            prog.emit("clip_detection_failed", pct=pct_base + 1,
                      detail="Render failed before detection",
                      clip_id=clip_id, error=True)
            continue

        # Propagate metadata errors
        if slot.get("error_stage") == "metadata":
            result.failed_stage = slot["error_stage"]
            result.failure_reason = slot["error_msg"]
            result.status = "ERROR"
            camera_results.append(result)
            continue

        render_status = slot.get("render_status", "")
        if slot.get("render_path") is None or render_status in ("RENDER_ERROR", "RENDER_TIMEOUT"):
            log.warning("Skipping detect for %s due to render failure status=%s", clip_id, render_status or "UNKNOWN")
            result.detection = SphereDetectionResult(
                clip_id=clip_id,
                status="FAILED",
                roi=None,
                source="render_error",
                failed_gate="render_error",
                failure_reason=slot.get("error_msg") or f"Render failed with status {render_status or 'UNKNOWN'}.",
            )
            result.failed_stage = "render"
            result.failure_reason = result.detection.failure_reason
            result.status = "ERROR"
            camera_results.append(result)
            prog.emit("clip_detection_failed", pct=pct_base + 1,
                      detail=f"Render failed: {result.failure_reason[:80]}",
                      clip_id=clip_id, error=True)
            continue

        render_actual = Path(slot["render_path"])
        result.original_render_path = str(render_actual)
        result.linear_render_path = slot.get("linear_render_path")
        result.linear_render_error = slot.get("linear_render_error")

        # --- 3c: Sphere detection ---
        prog.emit("clip_detect", pct=pct_base + 1,
                  detail=f"Detecting sphere in {clip_id}", clip_id=clip_id)
        try:
            image_hwc = load_render_as_hwc(str(render_actual))
        except Exception as exc:
            result.failed_stage = "detection"
            result.failure_reason = f"Could not load render: {exc}"
            result.status = "ERROR"
            camera_results.append(result)
            continue

        # Load prior for this camera unless validation mode disables it.
        camera_prior = None if disable_priors else get_camera_prior(
            sphere_profile, camera_label
        )

        # Prior injection: when a camera has verified_manual sample(s), synthesize
        # an ROI directly from the prior geometry and treat it as operator-confirmed.
        # This bypasses Hough entirely for cameras that have been manually solved before.
        # Gate validation is SKIPPED — this is a trusted human-confirmed position.
        prior_roi_injected = False
        prior_roi = None
        if camera_prior and camera_prior.get("sample_count", 0) >= 1:
            cam_entry = sphere_profile.get("cameras", {}).get(camera_label, {})
            manual_samples = [s for s in cam_entry.get("samples", [])
                              if s.get("trust") == "verified_manual"]
            if len(manual_samples) >= 1:
                try:
                    from .sphere import _load_image
                    img_full = _load_image(str(render_actual))
                    fw = img_full.width
                    fh = img_full.height
                    cx = camera_prior["cx_norm_mean"] * fw
                    cy = camera_prior["cy_norm_mean"] * fh
                    r  = camera_prior["radius_ratio_mean"] * min(fw, fh)
                    prior_roi = SphereROI(cx=cx, cy=cy, r=r)
                    prior_roi_injected = True
                except Exception:
                    prior_roi_injected = False

        # Detection priority — the solved prior is a LAST-RESORT back-check only,
        # never a leading feature (it can't help field users unless the same clip
        # is calibrated repeatedly), and is NEVER blended into a real detection.
        #   0. Manual ROI — operator placed, unconditional trust.
        #   Pass 1: cold Hough/ALT + standard gates (no prior of any kind).
        #   Pass 2: gating-2 — ALT's candidate revalidated at a looser shadow_specular
        #           (handled inside detect_sphere on the clean ALT stream).
        #   Pass 3: ONLY if pass 1+2 return FAILED or NEEDS_ASSIST — back-check the
        #           stored prior ROI with validate_manual_roi. Pass → prior_assisted.
        if clip_id in manual_rois:
            detection = validate_manual_roi(image_hwc, manual_rois[clip_id],
                                            clip_id=clip_id)
        else:
            # --- Pass 1 + 2: cold detection — NO position prior (no blend/ranking
            # hint). photo_prior (BRDF-threshold stats) is retained as it is a gate
            # tuning detail, not a sphere-position prior, and is part of the proven
            # cold path. ---
            detection = detect_sphere(image_hwc, clip_id=clip_id,
                                      photo_prior=camera_prior,
                                      peer_ire_values=peer_ire_values)

            # --- Pass 3: prior back-check, only on FAILED or NEEDS_ASSIST. ---
            if (detection.status in ("FAILED", "NEEDS_ASSIST")
                    and prior_roi_injected and prior_roi is not None):
                validated_prior = validate_manual_roi(image_hwc, prior_roi,
                                                      clip_id=clip_id)
                all_gates_passed = (
                    validated_prior.gates
                    and all(g.passed for g in validated_prior.gates)
                )
                if all_gates_passed:
                    # Prior ROI passed photometric gates — accept as prior_assisted.
                    detection = SphereDetectionResult(
                        clip_id=clip_id,
                        status="SUCCESS",
                        roi=prior_roi,
                        source="prior_assisted",
                        gates=validated_prior.gates,
                        failed_gate="",
                        failure_reason="",
                        hough_accumulator=validated_prior.hough_accumulator,
                        radius_ratio=validated_prior.radius_ratio,
                        interior_luminance_mean=validated_prior.interior_luminance_mean,
                        interior_luminance_stddev=validated_prior.interior_luminance_stddev,
                        chromaticity_distance=validated_prior.chromaticity_distance,
                        ire_spread=validated_prior.ire_spread,
                        lambertian_score=validated_prior.lambertian_score,
                        best_candidate_roi=validated_prior.best_candidate_roi,
                    )
                else:
                    # Prior ROI failed gates — something unexpected is at that
                    # position (wrong object, scene change). Flag NEEDS_ASSIST so
                    # the operator sees the candidate and can place a manual ROI.
                    # Measurement still runs so IRE is visible in the terminal and
                    # the report shows the bad solve for operator review.
                    failed_gate_name = next(
                        (g.gate for g in (validated_prior.gates or []) if not g.passed),
                        "unknown",
                    )
                    detection = SphereDetectionResult(
                        clip_id=clip_id,
                        status="NEEDS_ASSIST",
                        roi=prior_roi,
                        source="prior_assisted",
                        gates=validated_prior.gates,
                        failed_gate=failed_gate_name,
                        failure_reason=(
                            f"Prior-assisted ROI failed gate '{failed_gate_name}'. "
                            "Scene may have changed or wrong object at prior position. "
                            "Operator verification required."
                        ),
                        hough_accumulator=validated_prior.hough_accumulator,
                        radius_ratio=validated_prior.radius_ratio,
                        interior_luminance_mean=validated_prior.interior_luminance_mean,
                        interior_luminance_stddev=validated_prior.interior_luminance_stddev,
                        chromaticity_distance=validated_prior.chromaticity_distance,
                        ire_spread=validated_prior.ire_spread,
                        lambertian_score=validated_prior.lambertian_score,
                        best_candidate_roi=validated_prior.best_candidate_roi,
                    )

        result.detection = detection

        if not detection.success and detection.status != "NEEDS_ASSIST":
            result.failed_stage = "detection"
            result.failure_reason = f"{detection.failed_gate}: {detection.failure_reason}"
            result.status = "NEEDS_ASSIST"  # operator places ROI — not a failure
            camera_results.append(result)
            prog.emit("clip_detection_failed", pct=pct_base + 1,
                      detail=f"Sphere not found: {detection.failure_reason}",
                      clip_id=clip_id)
            continue

        # --- 3d: Measure ---
        prog.emit("clip_measure", pct=pct_base + 3,
                  detail=f"Measuring {clip_id}", clip_id=clip_id)
        try:
            measurement = measure_render(
                str(render_actual),
                detection.roi,
                clip_id=clip_id,
                detection=detection,
                roi_source=detection.source,
            )
        except Exception as exc:
            result.failed_stage = "measurement"
            result.failure_reason = str(exc)
            result.status = "ERROR"
            camera_results.append(result)
            continue

        # Scene-linear measurement at the SAME ROI — for the exposure solve.
        lin_path = slot.get("linear_render_path")
        if lin_path and measurement.measurement_valid:
            measurement.hero_log2_lin = measure_center_log2(lin_path, detection.roi)

        result.measurement = measurement

        if not measurement.measurement_valid:
            result.failed_stage = "measurement"
            result.failure_reason = measurement.validity_reason
            result.status = "NO_DATA"
            camera_results.append(result)
            continue

        # Add this camera's hero IRE to the peer pool for subsequent cameras.
        # Only confirmed measurements contribute — failed/invalid cameras don't
        # influence the context window.
        if measurement.hero_ire is not None:
            peer_ire_values.append(measurement.hero_ire)

        # Record detection into sphere profile
        if detection.success and measurement.measurement_valid and result.metadata:
            trust = "verified_manual" if detection.source in (
                "manual_operator", "prior_assisted") else "verified_auto"
            photometrics = (
                manual_sample_photometrics(detection=detection, measurement=measurement)
                if trust == "verified_manual"
                else {
                    "ire_spread": detection.ire_spread,
                    "chroma_distance": detection.chromaticity_distance,
                    "lambertian_score": detection.lambertian_score,
                    "interior_lum_mean": detection.interior_luminance_mean,
                    "interior_lum_stddev": detection.interior_luminance_stddev,
                    "hero_ire": measurement.hero_ire,
                }
            )
            sphere_profile = record_detection(
                sphere_profile,
                clip_id=clip_id,
                camera_label=camera_label,
                run_id=run_id,
                roi_cx=detection.roi.cx,
                roi_cy=detection.roi.cy,
                roi_r=detection.roi.r,
                frame_width=result.metadata.frame_width or measurement.render_width,
                frame_height=result.metadata.frame_height or measurement.render_height,
                ire_spread=photometrics["ire_spread"],
                chroma_distance=photometrics["chroma_distance"],
                lambertian_score=photometrics["lambertian_score"],
                interior_lum_mean=photometrics["interior_lum_mean"],
                interior_lum_stddev=photometrics["interior_lum_stddev"],
                hero_ire=photometrics["hero_ire"],
                trust=trust,
            )

        # Write per-clip analysis JSON
        _write_analysis_json(analysis_dir, clip_id, result, detection, measurement)

        camera_results.append(result)
        prog.emit("clip_complete", pct=pct_base + 5,
                  detail=f"{clip_id}: {measurement.hero_ire:.1f} IRE",
                  clip_id=clip_id)

    # --- Phase 4: Array solve ---
    prog.emit("solve", pct=76, detail="Computing array exposure target")
    valid_measurements = [r.measurement for r in camera_results if r.measurement and r.measurement.measurement_valid]

    if not valid_measurements:
        prog.emit("error", pct=76, detail="No valid measurements — cannot solve", error=True)
        raise ValueError("No valid measurements produced. Check sphere detection.")

    # Hard requirement: exposure is solved in scene-linear for accuracy, so every
    # valid camera must carry a scene-linear measurement. A display render that
    # succeeded but whose (retried) linear render did not is an abort, not a
    # silent downgrade to the display-space solve.
    _missing_lin = [
        r for r in camera_results
        if r.measurement and r.measurement.measurement_valid
        and getattr(r.measurement, "hero_log2_lin", None) is None
    ]
    if _missing_lin:
        _lines = []
        for r in _missing_lin:
            why = r.linear_render_error or "linear measurement unavailable"
            _lines.append(f"{r.clip_id}: {why}")
        detail = ("Scene-linear render missing for "
                  f"{len(_missing_lin)} camera(s); cannot solve exposure accurately.")
        prog.emit("error", pct=76, detail=detail, error=True)
        raise RuntimeError(detail + " Details:\n  " + "\n  ".join(_lines))

    target_log2, target_ire, spread, per_clip_exp_offsets = solve_exposure(
        valid_measurements, strategy=strategy, gray_target_ire=gray_target_ire)
    if strategy == "gray_anchor":
        prog.emit("solve_exposure", pct=78,
                  detail=f"Anchor: {gray_target_ire:.1f} IRE Log3G10 (18% gray) — scene-linear")
    else:
        prog.emit("solve_exposure", pct=78, detail=f"Target: {target_ire:.1f} IRE — scene-linear")

    # WB solve
    prog.emit("solve_wb", pct=79, detail="Computing white balance model")
    as_shot_kelvins = {}
    for r in camera_results:
        if r.metadata:
            as_shot_kelvins[r.clip_id] = r.metadata.kelvin
    wb_result, per_clip_wb = solve_white_balance(valid_measurements, as_shot_kelvins,
                                                 wb_mode=wb_mode)
    prog.emit("solve_wb_complete", pct=81, detail=f"WB: {wb_result.status} spread_after={wb_result.wc_spread_after:.4f}")

    # Build commit values per camera
    exposure_only = (wb_mode == "exposure_only")
    for cr in camera_results:
        if not cr.is_usable():
            continue
        exp_offset = per_clip_exp_offsets.get(cr.clip_id, 0.0)
        kelvin, tint = per_clip_wb.get(cr.clip_id, (5600, 0.0))
        m = cr.measurement
        wc_before, gm_before, wc_after, gm_after = 0.0, 0.0, 0.0, 0.0
        if m and not exposure_only:
            from .measure import compute_wc_gm
            wc_before, gm_before = compute_wc_gm(m.measured_rgb_mean)
        cr.commit = build_commit_values(
            cr,
            exposure_offset=exp_offset,
            kelvin=kelvin,
            tint=tint,
            wc_before=wc_before,
            gm_before=gm_before,
            exposure_only=exposure_only,
        )

    # --- Phase 5: Corrected renders + verification ---
    _consensus_log2 = None
    if render_corrected:
        wb_result, _consensus_log2 = _closed_loop_phase(
            camera_results,
            target_log2=target_log2,
            wb_result=wb_result,
            previews_dir=previews_dir,
            redline=redline,
            delivery_pipeline=delivery_pipeline,
            score_vs_consensus=(strategy == "gray_anchor"),
        )
    # For the absolute-gray anchor, the true DISPLAY anchor is only knowable from
    # the corrected renders (it depends on the display transform). Adopt the
    # measured corrected consensus as the reported anchor when available.
    if _consensus_log2 is not None:
        target_log2 = float(_consensus_log2)
        target_ire = float((2.0 ** target_log2) * 100.0)

    # --- Phase 6: Match scoring (there is no FAIL) ---
    prog.emit("assessing", pct=94, detail="Computing match percentages")
    for cr in camera_results:
        if not cr.is_usable():
            continue
        cr.status = "SOLVED"
        cr.match_pct = camera_match_pct(cr.exposure_match_pct, cr.wb_match_pct)

    assessment = assess_run(camera_results)

    run_result = RunResult(
        run_id=run_id,
        created_at=created_at,
        input_path=str(Path(input_path).expanduser().resolve()),
        out_dir=str(out_root),
        cameras=camera_results,
        exposure_spread=spread,
        wb_solve=wb_result,
        anchor_ire=target_ire,
        anchor_log2=target_log2,
        anchor_source=strategy,
        gray_target_ire=gray_target_ire,
        assessment_status=assessment["assessment_status"],
        array_match_pct=assessment["array_match_pct"],
        min_match_pct=assessment["min_match_pct"],
        min_match_clip_id=assessment["min_match_clip_id"],
        solved_count=assessment["solved_count"],
        scored_count=assessment["scored_count"],
        needs_assist_count=assessment["needs_assist_count"],
        no_data_count=assessment["no_data_count"],
        operator_recommendation=_operator_recommendation(assessment),
        exposure_only=exposure_only,
    )

    # Delivery-domain finalize (no-op unless a delivery pipeline was selected)
    _finalize_delivery(run_result, delivery_pipeline)

    # --- Save sphere profile ---
    save_project_profile(project_id, sphere_profile)
    prog.emit("profile_saved", pct=95,
              detail=f"Sphere profile saved: {project_id}")

    # --- Phase 7: Write outputs ---
    prog.emit("writing", pct=96, detail="Writing report")
    _write_summary_json(out_root, run_result, rl_check)
    _write_array_calibration_json(out_root, run_result)
    # Offline fallback — always written so post has the match even if RCP2 was
    # unreachable on set (generic JSON + REDCINE-X develop CSV).
    try:
        exp = write_match_export(out_root, run_result)
        prog.emit("match_export", pct=97,
                  detail=f"Match export written ({exp['camera_count']} cameras)")
    except Exception:
        log.exception("match_export failed (non-fatal)")

    amp = assessment["array_match_pct"]
    prog.emit("complete", pct=100, detail=(
        f"Done. Array match {amp:.0f}% ({assessment['solved_count']}/{len(camera_results)} solved)"
        if amp is not None
        else f"Done. {assessment['solved_count']}/{len(camera_results)} solved (unverified)"
    ))

    return run_result


# ---------------------------------------------------------------------------
# Closed-loop verification — shared by run_analysis (render_corrected=True)
# and verify_run (post-QC path used by the desktop app)
# ---------------------------------------------------------------------------

def _closed_loop_phase(
    camera_results: List[CameraResult],
    *,
    target_log2: float,
    wb_result,
    previews_dir: Path,
    redline: str,
    delivery_pipeline: Optional[ColorPipeline] = None,
    score_vs_consensus: bool = False,
):
    """Render corrected frames, measure them, score per-camera match axes.
    Mutates camera_results in place.

    Returns (wb_result, consensus_log2):
      wb_result      — the (mutated) WB solve result.
      consensus_log2 — display-referred median log2 of the corrected renders when
                       score_vs_consensus is set, else None.

    Exposure scoring:
      Default — each corrected camera is scored against the fixed target_log2
                (the median strategy; unchanged, golden-anchored).
      score_vs_consensus — exposure is scored against the corrected-array median
                (transform-exact), mirroring the WB and delivery consensus paths.
                Used by the absolute-gray anchor, where every camera is driven to
                the same scene-linear value and the true display anchor is only
                knowable from the corrected renders.

    If delivery_pipeline is provided (and not the reference pipeline), each
    corrected frame is ALSO rendered through that pipeline and measured in the
    delivery domain — operator-facing only, never feeding the reference solve.
    """
    use_delivery = delivery_pipeline is not None and not delivery_pipeline.is_reference
    delivery_weights = (
        luma_weights_for(delivery_pipeline.color_space) if use_delivery else None
    )
    prog.emit("corrected_renders", pct=82, detail="Rendering corrected frames")
    for i, cr in enumerate(camera_results):
        if not cr.is_usable() or cr.commit is None:
            continue
        pct = prog.emit_phase_pct(i, len(camera_results), 82, 93)
        corrected_out = previews_dir / f"{cr.clip_id}.both.review.corrected.tiff"
        # Exposure-only: keep as-shot white balance (don't override kelvin/tint).
        _eo = getattr(cr.commit, "exposure_only", False)
        render_res = render_measurement_frame(
            cr.source_path,
            str(corrected_out),
            redline=redline,
            frame_index=0,
            use_as_shot=True,
            exposure_adjust=cr.commit.exposure_adjust,
            kelvin=None if _eo else cr.commit.kelvin,
            tint=None if _eo else cr.commit.tint,
        )
        roi = None
        if render_res["ok"]:
            corrected_actual = Path(render_res["output_path"])
            cr.corrected_render_path = str(corrected_actual)
            # Verify corrected render
            try:
                roi = cr.detection.roi if cr.detection else cr.measurement.roi
                corrected_meas = measure_render(
                    str(corrected_actual), roi, clip_id=cr.clip_id
                )
                cr.exposure_closed_loop_status = "VERIFIED"
                cr.corrected_ire = corrected_meas.hero_ire
                if not score_vs_consensus:
                    # Fixed-target scoring (median strategy) — unchanged.
                    residual = exposure_residual_stops(corrected_meas.hero_ire, target_log2)
                    cr.corrected_exposure_residual_stops = residual
                    cr.exposure_match_pct = exposure_match_pct(residual)
                # score_vs_consensus: exposure residual/match set after the loop,
                # once the corrected-array median is known.
                # WB closed loop: capture measured WC/GM from the same render
                if corrected_meas.measurement_valid:
                    wc_c, gm_c = compute_wc_gm(corrected_meas.measured_rgb_mean)
                    cr.corrected_wc = wc_c
                    cr.corrected_gm = gm_c
            except Exception:
                cr.exposure_closed_loop_status = "ERROR"

        # Delivery-domain corrected render (same commits, project pipeline).
        if use_delivery and render_res["ok"] and roi is not None:
            try:
                delivery_out = previews_dir / f"{cr.clip_id}.delivery.review.corrected.tiff"
                drr = render_measurement_frame(
                    cr.source_path,
                    str(delivery_out),
                    redline=redline,
                    frame_index=0,
                    use_as_shot=True,
                    exposure_adjust=cr.commit.exposure_adjust,
                    kelvin=cr.commit.kelvin,
                    tint=cr.commit.tint,
                    pipeline=delivery_pipeline,
                )
                if drr["ok"]:
                    dm = measure_delivery_hero(
                        drr["output_path"], roi, luma_weights=delivery_weights
                    )
                    if dm["valid"]:
                        cr.delivery_corrected_render_path = drr["output_path"]
                        cr.delivery_corrected_ire = float(dm["hero_ire"])
                        cr.delivery_hero_log2 = float(dm["hero_log2"])
                        cr.delivery_wc = float(dm["wc"])
                        cr.delivery_gm = float(dm["gm"])
            except Exception:
                log.exception("Delivery-domain render/measure failed for %s", cr.clip_id)
        prog.emit("corrected_render_done", pct=pct, detail=f"{cr.clip_id} corrected",
                  clip_id=cr.clip_id)

    # Exposure closed-loop assessment for the absolute-gray anchor: score each
    # camera against the MEASURED corrected-array median (transform-exact). All
    # cameras were driven to the same scene-linear value, so this median is the
    # true display anchor; residuals reflect real convergence + render jitter.
    consensus_log2 = None
    if score_vs_consensus:
        corr_log2 = [
            float(np.log2(max(cr.corrected_ire / 100.0, 1e-6)))
            for cr in camera_results
            if cr.is_usable() and cr.corrected_ire and cr.corrected_ire > 0
        ]
        if corr_log2:
            consensus_log2 = float(np.median(corr_log2))
            for cr in camera_results:
                if not (cr.is_usable() and cr.corrected_ire and cr.corrected_ire > 0):
                    continue
                resid = abs(float(np.log2(max(cr.corrected_ire / 100.0, 1e-6))) - consensus_log2)
                cr.corrected_exposure_residual_stops = resid
                cr.exposure_match_pct = exposure_match_pct(resid)

    # WB closed-loop assessment: re-gate on MEASURED spread from the
    # corrected renders (solve-time spread_after is only a prediction).
    corrected_wc_gm = {
        cr.clip_id: (cr.corrected_wc, cr.corrected_gm)
        for cr in camera_results
        if cr.is_usable() and cr.corrected_wc is not None and cr.corrected_gm is not None
    }
    wb_result = verify_wb_closed_loop(wb_result, corrected_wc_gm)
    prog.emit(
        "wb_closed_loop",
        pct=93,
        detail=(
            f"WB closed-loop: {wb_result.status} "
            f"gm_spread_measured={wb_result.gm_spread_measured:.4f}"
            if wb_result.closed_loop
            else f"WB closed-loop skipped (n={len(corrected_wc_gm)}<2) — unverified"
        ),
    )
    # Per-camera WB match: measured GM deviation vs group median
    if wb_result.closed_loop:
        group_gm_median = float(np.median([v[1] for v in corrected_wc_gm.values()]))
        for cr in camera_results:
            if cr.corrected_gm is not None:
                cr.wb_match_pct = wb_match_pct(cr.corrected_gm - group_gm_median)
    for cr in camera_results:
        if not cr.is_usable():
            continue
        cr.wb_closed_loop_status = wb_result.status
    return wb_result, consensus_log2


def _finalize_delivery(
    run_result: RunResult,
    delivery_pipeline: Optional[ColorPipeline],
) -> None:
    """Score delivery-domain match % and build a fresh PipelineProfile.

    No-op unless a delivery pipeline was selected and delivery measurements were
    produced. Scores inter-camera agreement in the delivery look (mirrors the
    reference match model) and records a profile characterized from THIS run's
    own corrected renders. Never touches the reference solve/commits.
    """
    if delivery_pipeline is None or delivery_pipeline.is_reference:
        return
    cams = [c for c in run_result.cameras
            if c.is_usable() and c.delivery_hero_log2 is not None]
    if not cams:
        return

    run_result.delivery_pipeline_name = delivery_pipeline.name

    # Exposure: residual vs the array's delivery-domain median (delivery target).
    delivery_target_log2 = float(np.median([c.delivery_hero_log2 for c in cams]))
    # WB: GM deviation vs the array's delivery-domain median.
    gm_vals = [c.delivery_gm for c in cams if c.delivery_gm is not None]
    delivery_gm_median = float(np.median(gm_vals)) if gm_vals else 0.0

    samples = []
    for c in cams:
        exp_residual = abs(c.delivery_hero_log2 - delivery_target_log2)
        c.delivery_exposure_match_pct = exposure_match_pct(exp_residual)
        if c.delivery_gm is not None:
            c.delivery_wb_match_pct = wb_match_pct(c.delivery_gm - delivery_gm_median)
        c.delivery_match_pct = camera_match_pct(
            c.delivery_exposure_match_pct, c.delivery_wb_match_pct
        )
        # Pair with the reference-domain corrected log2 for the fresh tonal map.
        if c.corrected_ire and c.corrected_ire > 0:
            ref_log2 = float(np.log2(max(c.corrected_ire / 100.0, 1e-6)))
            samples.append(CharSample(
                reference_log2=ref_log2,
                delivery_log2=c.delivery_hero_log2,
                delivery_wc=c.delivery_wc if c.delivery_wc is not None else 0.0,
                delivery_gm=c.delivery_gm if c.delivery_gm is not None else 0.0,
                is_neutral=True,
            ))

    scored = [c.delivery_match_pct for c in cams if c.delivery_match_pct is not None]
    if scored:
        run_result.delivery_array_match_pct = float(np.mean(scored))
        worst = min((c for c in cams if c.delivery_match_pct is not None),
                    key=lambda c: c.delivery_match_pct)
        run_result.delivery_min_match_pct = worst.delivery_match_pct
        run_result.delivery_min_match_clip_id = worst.clip_id

    if samples:
        profile = build_profile(
            delivery_pipeline.name, delivery_pipeline.color_space, samples
        )
        run_result.delivery_profile = profile.to_dict()


def verify_run(run_result: RunResult, redline_path: str = "",
               delivery_pipeline: Optional[ColorPipeline] = None) -> RunResult:
    """
    Closed-loop verification for an existing run — the post-QC path.

    The desktop app runs analysis with render_corrected=False (QC may still
    change ROIs, so corrected renders would be wasted). Call this AFTER the
    operator accepts QC and BEFORE building the report: renders corrected
    frames with the final commits, measures them, scores match percentages,
    re-assesses, and refreshes summary/calibration JSON.

    Mutates and returns run_result.
    """
    if run_result.wb_solve is None or not run_result.cameras:
        return run_result
    redline = resolve_redline_executable(redline_path) if redline_path \
        else resolve_redline_executable()
    out_root = Path(run_result.out_dir)
    previews_dir = out_root / "previews"
    previews_dir.mkdir(parents=True, exist_ok=True)

    _gray_anchor = (run_result.anchor_source == "gray_anchor")
    run_result.wb_solve, _consensus_log2 = _closed_loop_phase(
        run_result.cameras,
        target_log2=run_result.anchor_log2,
        wb_result=run_result.wb_solve,
        previews_dir=previews_dir,
        redline=redline,
        delivery_pipeline=delivery_pipeline,
        score_vs_consensus=_gray_anchor,
    )
    if _consensus_log2 is not None:
        run_result.anchor_log2 = float(_consensus_log2)
        run_result.anchor_ire = float((2.0 ** _consensus_log2) * 100.0)

    _finalize_delivery(run_result, delivery_pipeline)

    prog.emit("assessing", pct=94, detail="Computing match percentages")
    for cr in run_result.cameras:
        if not cr.is_usable():
            continue
        cr.status = "SOLVED"
        cr.match_pct = camera_match_pct(cr.exposure_match_pct, cr.wb_match_pct)

    assessment = assess_run(run_result.cameras)
    run_result.assessment_status      = assessment["assessment_status"]
    run_result.array_match_pct        = assessment["array_match_pct"]
    run_result.min_match_pct          = assessment["min_match_pct"]
    run_result.min_match_clip_id      = assessment["min_match_clip_id"]
    run_result.solved_count           = assessment["solved_count"]
    run_result.scored_count           = assessment["scored_count"]
    run_result.needs_assist_count     = assessment["needs_assist_count"]
    run_result.no_data_count          = assessment["no_data_count"]
    run_result.operator_recommendation = _operator_recommendation(assessment)

    # Refresh JSON outputs (preserve redline build info from the original run)
    rl_check: Dict = {}
    try:
        existing = json.loads((out_root / "summary.json").read_text(encoding="utf-8"))
        rl_check = {"build": existing.get("redline_build", ""),
                    "sdk_version": existing.get("redline_sdk_version", "")}
    except Exception:
        pass
    _write_summary_json(out_root, run_result, rl_check)
    _write_array_calibration_json(out_root, run_result)
    # Offline fallback — always written (generic JSON + REDCINE-X develop CSV).
    try:
        exp = write_match_export(out_root, run_result)
        prog.emit("match_export", pct=97,
                  detail=f"Match export written ({exp['camera_count']} cameras)")
    except Exception:
        log.exception("match_export failed (non-fatal)")

    amp = assessment["array_match_pct"]
    prog.emit("verify_complete", pct=100, detail=(
        f"Verified. Array match {amp:.0f}%" if amp is not None
        else "Verification incomplete — no corrected renders measured"
    ))
    return run_result


# ---------------------------------------------------------------------------
# JSON output writers
# ---------------------------------------------------------------------------

def _write_analysis_json(
    analysis_dir: Path,
    clip_id: str,
    cr: CameraResult,
    detection: SphereDetectionResult,
    measurement,
) -> None:
    path = analysis_dir / f"{clip_id}.analysis.json"
    payload = {
        "schema_version": "r3dmatch3_analysis_v1",
        "clip_id": clip_id,
        "camera_label": cr.camera_label,
        "source_path": cr.source_path,
        "detection_source": detection.source,
        "sphere_roi": detection.roi.to_dict() if detection.roi else None,
        "detection_gates": [
            {"gate": g.gate, "passed": g.passed, "reason": g.reason}
            for g in detection.gates
        ],
        "measurement_valid": measurement.measurement_valid,
        "hero_ire": measurement.hero_ire,
        "hero_log2": measurement.hero_log2,
        "zone_bright_ire": measurement.zone_bright.ire,
        "zone_center_ire": measurement.zone_center.ire,
        "zone_dark_ire": measurement.zone_dark.ire,
        "measured_rgb_mean": list(measurement.measured_rgb_mean),
        "measured_rgb_chromaticity": list(measurement.measured_rgb_chromaticity),
        "chromaticity_distance": measurement.chromaticity_distance,
        "render_path": measurement.render_path,
        "render_sha256": measurement.render_sha256,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_summary_json(out_root: Path, run: RunResult, rl_check: Dict) -> None:
    summary = {
        "schema_version": "r3dmatch3_summary_v1",
        "run_id": run.run_id,
        "created_at": run.created_at,
        "input_path": run.input_path,
        "assessment_status": run.assessment_status,
        "camera_count": len(run.cameras),
        "array_match_pct": run.array_match_pct,
        "min_match_pct": run.min_match_pct,
        "min_match_clip_id": run.min_match_clip_id,
        "solved_count": run.solved_count,
        "scored_count": run.scored_count,
        "needs_assist_count": run.needs_assist_count,
        "no_data_count": run.no_data_count,
        "anchor_ire": run.anchor_ire,
        "anchor_log2": run.anchor_log2,
        "wb_status": run.wb_solve.status if run.wb_solve else "SKIPPED",
        "shared_kelvin": run.wb_solve.shared_kelvin if run.wb_solve else None,
        "wc_spread_before": run.wb_solve.wc_spread_before if run.wb_solve else None,
        "wc_spread_after": run.wb_solve.wc_spread_after if run.wb_solve else None,
        "gm_spread_before": run.wb_solve.gm_spread_before if run.wb_solve else None,
        "gm_spread_after": run.wb_solve.gm_spread_after if run.wb_solve else None,
        "wc_spread_measured": run.wb_solve.wc_spread_measured if run.wb_solve else None,
        "gm_spread_measured": run.wb_solve.gm_spread_measured if run.wb_solve else None,
        "wb_closed_loop": run.wb_solve.closed_loop if run.wb_solve else False,
        "redline_build": rl_check.get("build", ""),
        "redline_sdk_version": rl_check.get("sdk_version", ""),
        "operator_recommendation": run.operator_recommendation,
        "cameras": [
            {
                "clip_id": cr.clip_id,
                "camera_label": cr.camera_label,
                "status": cr.status,
                "hero_ire": cr.measurement.hero_ire if cr.measurement else None,
                "exposure_adjust": cr.commit.exposure_adjust if cr.commit else None,
                "kelvin": cr.commit.kelvin if cr.commit else None,
                "tint": cr.commit.tint if cr.commit else None,
                "corrected_ire": cr.corrected_ire,
                "corrected_exposure_residual_stops": cr.corrected_exposure_residual_stops,
                "corrected_wc": cr.corrected_wc,
                "corrected_gm": cr.corrected_gm,
                "exposure_match_pct": cr.exposure_match_pct,
                "wb_match_pct": cr.wb_match_pct,
                "match_pct": cr.match_pct,
                "exposure_closed_loop_status": cr.exposure_closed_loop_status,
                "wb_closed_loop_status": cr.wb_closed_loop_status,
                "detection_source": cr.detection.source if cr.detection else "failed",
                "failed_stage": cr.failed_stage,
                "failure_reason": cr.failure_reason,
            }
            for cr in run.cameras
        ],
    }
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _write_array_calibration_json(out_root: Path, run: RunResult) -> None:
    """Write the RCP2-ready calibration payload."""
    cameras = []
    for cr in run.cameras:
        if cr.commit:
            cameras.append({
                "clip_id": cr.clip_id,
                "camera_label": cr.camera_label,
                "commit_values": cr.commit.to_dict(),
                "rcp2_push_eligible": True,  # no software veto — operator decides
                "match_pct": cr.match_pct,
            })
    payload = {
        "schema_version": "r3dmatch3_array_calibration_v1",
        "run_id": run.run_id,
        "created_at": run.created_at,
        "shared_kelvin": run.wb_solve.shared_kelvin if run.wb_solve else None,
        "wb_status": run.wb_solve.status if run.wb_solve else "SKIPPED",
        "cameras": cameras,
    }
    (out_root / "array_calibration.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _operator_recommendation(assessment: Dict) -> str:
    """Informational only — never blocks. The operator owns the push decision."""
    solved = assessment["solved_count"]
    total = assessment["total_cameras"]
    amp = assessment["array_match_pct"]
    mmp = assessment["min_match_pct"]
    worst = assessment["min_match_clip_id"]
    assist = assessment["needs_assist_count"]

    parts = []
    if amp is not None:
        parts.append(f"Array match {amp:.0f}%.")
        if mmp is not None and mmp < amp - 0.5:
            parts.append(f"Lowest: {worst} at {mmp:.0f}%.")
    elif solved:
        parts.append(f"{solved}/{total} cameras solved (match unverified — no corrected renders).")
    if assist:
        parts.append(f"{assist} camera(s) need ROI assist.")
    if solved:
        parts.append(f"{solved}/{total} ready to push via RCP2.")
    else:
        parts.append("No solves yet — place ROIs and re-run.")
    return " ".join(parts)
