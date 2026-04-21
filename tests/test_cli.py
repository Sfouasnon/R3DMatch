import hashlib
import json
import os
import subprocess
import sys
import time
import types
from pathlib import Path
from typing import Optional, cast

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFilter
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication
from typer.testing import CliRunner

from r3dmatch.calibration import build_array_calibration_from_analysis, calibrate_card_path, calibrate_color_path, calibrate_exposure_path, calibrate_sphere_path, center_crop_roi, derive_array_group_key, detect_gray_card_roi, load_calibration, load_card_roi_file, load_color_calibration, load_exposure_calibration, measure_card_from_roi, measure_color_region, measure_sphere_from_roi, solve_neutral_gains
from r3dmatch.cli import app
from r3dmatch.commit_values import build_commit_values, solve_kelvin_tint_from_chromaticity, solve_white_balance_model_for_records
from r3dmatch.desktop_app import R3DMatchDesktopWindow, _build_desktop_results_summary, build_review_command, run_ui_self_check, scan_calibration_sources
from r3dmatch.execution import CANCEL_FILE_ENV, CancellationError, run_cancellable_subprocess
from r3dmatch.ftps_ingest import discover_ftps_batch, ingest_ftps_batch, ingest_manifest_path_for, plan_ftps_request, retry_failed_ftps_batch, run_ftps_ingest_job
from r3dmatch.identity import group_key_from_clip_id, rmd_name_for_clip_id, subset_key_from_clip_id
from r3dmatch.matching import _resolve_measurement_worker_count, analyze_path, camera_group_from_clip_id, discover_clips, measure_frame_color_and_exposure
from r3dmatch.models import ClipMetadata, ClipResult, FrameStat, GrayCardROI, MonitoringContext, SamplePlan, SamplingRegion, SphereROI
from r3dmatch.rcp2_apply import (
    MemoryRcp2Transport,
    Rcp2TimeoutError,
    WebSocketRcp2Transport,
    apply_calibration_payload,
    build_camera_verification_report,
    summarize_apply_report,
)
from r3dmatch.rcp2_websocket import (
    PARAMETER_SPECS,
    JsonWebSocketClient,
    Rcp2ParameterState,
    Rcp2WebSocketSession,
    Rcp2WebSocketTimeout,
    normalize_parameter_id,
    build_periodic_get_payload,
    build_websocket_text_frame,
    build_websocket_upgrade_request,
    expected_websocket_accept,
    extract_parameter_state,
)
from r3dmatch.report import _build_exposure_plot_svg, _build_scientific_validation_artifacts, _build_sphere_detection_artifacts, _build_strategy_payloads, _compute_image_difference_metrics, _compute_sphere_center_patch, _contact_sheet_chromaticity_visual, _contact_sheet_original_wb_block, _contact_sheet_resolve_measurement_fields, _contact_sheet_svg_color_deviation_chart, _derived_profile_scalar, _detect_focus_chart_roi_hwc, _detect_sphere_roi_in_region_hwc, _focus_metric_bundle, _focus_validation_summary, _gray_target_consistency_summary, _ipp2_closed_loop_next_correction, _ipp2_closed_loop_target, _ipp2_validation_presentation, _load_preview_image_as_normalized_rgb, _manual_assist_sanitized_entry, _measure_rendered_preview_roi_ipp2, _measurement_values_for_record, _normalize_preview_image_array, _operator_guidance_for_correction, _refine_manual_assist_sphere_roi, _render_preview_frame_with_retries, _report_grid_columns, _report_tiles_per_page, _resolve_redline_executable, _sphere_detection_label, _sphere_detection_note, _upgrade_trust_from_ipp2_validation, _validate_known_sphere_roi, _validate_rendered_preview_output, build_contact_sheet_report, build_lightweight_analysis_report, build_review_package, contact_sheet_pdf_export_preflight, format_stop_string, generate_preview_stills, preview_filename_for_clip_id, render_contact_sheet_html, render_contact_sheet_pdf, render_contact_sheet_pdf_from_html, render_preview_frame, round_to_standard_stop_fraction, validate_contact_sheet_html_assets
from r3dmatch.rmd import render_rmd_xml, rmd_filename_for_clip_id, write_rmd_for_clip_with_metadata, write_rmds_from_analysis
from r3dmatch.runtime_env import (
    DESKTOP_CONFIG_ENV,
    ensure_runtime_environment,
    persist_redline_configured_path,
    redline_configured_path,
    runtime_health_payload,
)
from r3dmatch.ui import build_table_rows, load_review_bundle
from r3dmatch.sdk import MockR3DBackend, RedSdkConfiguration, RedSdkDecoder, load_configured_red_native_module, red_sdk_configuration_error, resolve_backend, resolve_red_sdk_configuration
from r3dmatch.sidecar import build_sidecar_payload, sidecar_filename_for_clip_id
from r3dmatch.transcode import build_redline_command, build_redline_command_variants, write_transcode_plan
from r3dmatch.validation import validate_pipeline
from r3dmatch.web_app import _build_tcsh_launch_prefix, _calibration_overview_from_payload, _ensure_scan_preview, _normalize_subset_form, _preferred_roi_preview_record, _resolve_selected_clip_ids, _resolved_ingest_root_for_form, _resolved_output_path_for_form, _scan_preview_path, _subset_selection_ui, _validate_form, _validate_ftps_ingest_form, build_ftps_ingest_web_command, build_review_web_command, create_app, scan_sources
from r3dmatch.workflow import approve_master_rmd, build_post_apply_verification_from_reviews, clear_preview_cache, resolve_review_output_dir, review_calibration, validate_review_run_contract

runner = CliRunner()


def _write_fake_red_redistributables(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for filename in ("REDDecoder.dylib", "REDMetal.dylib", "REDOpenCL.dylib", "REDR3D.dylib"):
        (path / filename).write_bytes(b"runtime")


def _qt_test_app() -> QApplication:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture(autouse=True)
def _stub_contact_sheet_pdf_when_weasyprint_native_libs_are_missing(
    monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    if request.node.name in {
        "test_render_contact_sheet_pdf_paginates_large_camera_counts",
        "test_render_contact_sheet_pdf_uses_final_titles_and_skips_original_color_trace",
        "test_render_contact_sheet_pdf_from_html_reports_actionable_dependency_error",
        "test_validate_form_reports_weasyprint_preflight_failure_for_full_contact_sheet",
    }:
        return
    try:
        import weasyprint  # type: ignore # noqa: F401
    except Exception:
        def _fake_contact_sheet_pdf_export_preflight(html_path: Optional[str] = None) -> dict[str, object]:
            asset_validation = None
            if html_path:
                source = Path(html_path)
                asset_validation = {
                    "html_path": str(source.resolve()),
                    "image_count": 0,
                    "resolved_asset_count": 0,
                    "missing_assets": [],
                    "all_assets_exist": True,
                }
            return {
                "interpreter": sys.executable,
                "dyld_fallback_library_path": str(os.environ.get("DYLD_FALLBACK_LIBRARY_PATH") or ""),
                "red_sdk_root": str(os.environ.get("RED_SDK_ROOT") or ""),
                "red_sdk_redistributable_dir": str(os.environ.get("RED_SDK_REDISTRIBUTABLE_DIR") or ""),
                "weasyprint_importable": True,
                "weasyprint_error": "",
                "asset_validation": asset_validation,
            }

        def _fake_runtime_health_payload(html_path: Optional[str] = None) -> dict[str, object]:
            preflight = _fake_contact_sheet_pdf_export_preflight(html_path)
            return {
                "interpreter": sys.executable,
                "frozen_app": False,
                "virtual_env": "",
                "dyld_fallback_library_path": str(os.environ.get("DYLD_FALLBACK_LIBRARY_PATH") or ""),
                "dyld_fallback_source": "environment",
                "red_sdk_root": str(os.environ.get("RED_SDK_ROOT") or ""),
                "red_sdk_redistributable_dir": str(os.environ.get("RED_SDK_REDISTRIBUTABLE_DIR") or ""),
                "resolved_red_sdk_redistributable_dir": str(os.environ.get("RED_SDK_REDISTRIBUTABLE_DIR") or ""),
                "red_backend_ready": True,
                "red_backend_error": "",
                "red_sdk_runtime": {
                    "ready": True,
                    "error": "",
                    "redistributable_dir": str(os.environ.get("RED_SDK_REDISTRIBUTABLE_DIR") or ""),
                    "source": "environment_override",
                    "root": str(os.environ.get("RED_SDK_ROOT") or ""),
                    "bundled_dir": "",
                },
                "redline_tool": {
                    "ready": True,
                    "configured": "REDLine",
                    "resolved_path": "/usr/local/bin/REDLine",
                    "source": "path",
                    "config_path": "",
                    "configured_path": "",
                    "configured_config_path": str(Path("/tmp/redline.json")),
                    "error": "",
                },
                "weasyprint_importable": True,
                "weasyprint_error": "",
                "asset_validation": preflight.get("asset_validation"),
                "html_pdf_ready": True,
            }

        def _fake_render_contact_sheet_pdf_from_html(html_path: str, *, output_path: str) -> str:
            source = Path(html_path)
            output = Path(output_path)
            assert source.exists()
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(b"%PDF-1.4\n%r3dmatch test html-pdf stub\n")
            return str(output.resolve())

        monkeypatch.setattr("r3dmatch.report.contact_sheet_pdf_export_preflight", _fake_contact_sheet_pdf_export_preflight)
        monkeypatch.setattr("r3dmatch.web_app.runtime_health_payload", _fake_runtime_health_payload)
        monkeypatch.setattr("r3dmatch.report.render_contact_sheet_pdf_from_html", _fake_render_contact_sheet_pdf_from_html)


def test_build_websocket_upgrade_request_contains_expected_headers() -> None:
    request = build_websocket_upgrade_request(host="10.20.61.191", port=9998, key="dGhlIHNhbXBsZSBub25jZQ==").decode("utf-8")
    assert request.startswith("GET / HTTP/1.1\r\n")
    assert "Host: 10.20.61.191:9998\r\n" in request
    assert "Upgrade: websocket\r\n" in request
    assert "Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n" in request
    assert expected_websocket_accept("dGhlIHNhbXBsZSBub25jZQ==") == "s3pPLMBiTxaQ9kYGzzhZRbK+xOo="


def test_build_websocket_text_frame_masks_payload() -> None:
    frame = build_websocket_text_frame("hello", mask_key=b"\x01\x02\x03\x04")
    assert frame[:2] == bytes([0x81, 0x80 | 5])
    assert frame[2:6] == b"\x01\x02\x03\x04"
    assert frame[6:] == bytes([
        ord("h") ^ 0x01,
        ord("e") ^ 0x02,
        ord("l") ^ 0x03,
        ord("l") ^ 0x04,
        ord("o") ^ 0x01,
    ])


def test_extract_parameter_state_uses_edit_info_divider_and_display() -> None:
    state = extract_parameter_state(
        [
            {"type": "rcp_cur_int_edit_info", "id": "EXPOSURE_ADJUST", "divider": 1000, "cur": 12},
            {"type": "rcp_cur_int", "id": "EXPOSURE_ADJUST", "cur": {"val": 12}},
            {"type": "rcp_cur_str", "id": "EXPOSURE_ADJUST", "display": {"str": "0.012", "abbr": "0.012"}},
        ],
        parameter_id="EXPOSURE_ADJUST",
        fallback_divider=1000,
    )
    assert state.parameter_id == "EXPOSURE_ADJUST"
    assert state.raw_value == 12
    assert state.divider == 1000
    assert state.value == pytest.approx(0.012)
    assert state.display == "0.012"


def test_ipp2_closed_loop_next_correction_moves_against_residual() -> None:
    assert _ipp2_closed_loop_next_correction(
        current_correction=0.5,
        current_residual=-0.4,
        previous_correction=None,
        previous_residual=None,
    ) > 0.5
    assert _ipp2_closed_loop_next_correction(
        current_correction=0.5,
        current_residual=0.4,
        previous_correction=None,
        previous_residual=None,
    ) < 0.5


def test_measure_rendered_preview_roi_ipp2_detects_off_center_sphere_when_roi_missing(tmp_path: Path) -> None:
    height = 400
    width = 400
    image = np.zeros((height, width, 3), dtype=np.float32)
    yy, xx = np.meshgrid(np.arange(height, dtype=np.float32), np.arange(width, dtype=np.float32), indexing="ij")
    cx = 245.0
    cy = 180.0
    radius = 42.0
    distance = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    sphere_mask = distance <= radius
    vertical = np.clip((yy - (cy - radius)) / max(2.0 * radius, 1.0), 0.0, 1.0)
    sphere_luma = 0.42 - (vertical * 0.14)
    for channel in range(3):
        image[..., channel] = np.where(sphere_mask, sphere_luma, image[..., channel])
    path = tmp_path / "off_center_sphere.jpg"
    Image.fromarray(np.clip(image * 255.0, 0, 255).astype(np.uint8), mode="RGB").save(path, format="JPEG")

    measured = _measure_rendered_preview_roi_ipp2(str(path), None)

    assert measured["sphere_detection_success"] is True
    assert measured["sphere_roi_source"] in {
        "primary_detected",
        "secondary_detected",
        "neutral_blob_recovery",
        "opencv_hough_recovery",
        "opencv_hough_alt_detected",
        "opencv_hough_alt_recovery",
        "opencv_contour_recovery",
    }
    assert measured["detected_sphere_roi"]["cx"] == pytest.approx(cx, abs=18.0)
    assert measured["detected_sphere_roi"]["cy"] == pytest.approx(cy, abs=18.0)
    assert measured["top_ire"] > measured["mid_ire"] > measured["bottom_ire"]


def test_measure_rendered_preview_roi_ipp2_detects_far_right_edge_sphere_when_roi_missing(tmp_path: Path) -> None:
    tifffile = pytest.importorskip("tifffile")
    height = 420
    width = 520
    image = np.full((height, width, 3), 0.05, dtype=np.float32)
    yy, xx = np.meshgrid(np.arange(height, dtype=np.float32), np.arange(width, dtype=np.float32), indexing="ij")
    cx = 442.0
    cy = 214.0
    radius = 44.0
    distance = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    sphere_mask = distance <= radius
    vertical = np.clip((yy - (cy - radius)) / max(2.0 * radius, 1.0), 0.0, 1.0)
    sphere_luma = 0.30 - (vertical * 0.08)
    image[sphere_mask] = np.stack([sphere_luma[sphere_mask]] * 3, axis=-1)
    path = tmp_path / "far_right_edge_sphere.tiff"
    tifffile.imwrite(path, np.clip(image * 65535.0, 0, 65535).astype(np.uint16), photometric="rgb")

    measured = _measure_rendered_preview_roi_ipp2(str(path), None)

    assert measured["detection_failed"] is False
    assert measured["sphere_detection_success"] is True
    assert measured["sphere_roi_source"] in {
        "primary_detected",
        "secondary_detected",
        "localized_recovery",
        "neutral_blob_recovery",
        "opencv_hough_recovery",
        "opencv_hough_alt_detected",
        "opencv_hough_alt_recovery",
        "opencv_contour_recovery",
    }
    assert measured["detected_sphere_roi"]["cx"] == pytest.approx(cx, abs=22.0)
    assert measured["detected_sphere_roi"]["cy"] == pytest.approx(cy, abs=22.0)
    assert measured["rendered_preview_format"] == "tiff"
    assert measured["rendered_image_bit_depth"] == 16


def test_measure_rendered_preview_roi_ipp2_prefers_bright_edge_sphere_over_darker_false_circle(tmp_path: Path) -> None:
    tifffile = pytest.importorskip("tifffile")
    height = 420
    width = 640
    image = np.full((height, width, 3), 0.06, dtype=np.float32)
    yy, xx = np.meshgrid(np.arange(height, dtype=np.float32), np.arange(width, dtype=np.float32), indexing="ij")

    false_cx = 210.0
    false_cy = 205.0
    false_radius = 46.0
    false_distance = np.sqrt((xx - false_cx) ** 2 + (yy - false_cy) ** 2)
    false_mask = false_distance <= false_radius
    false_vertical = np.clip((yy - (false_cy - false_radius)) / max(2.0 * false_radius, 1.0), 0.0, 1.0)
    false_luma = 0.19 - (false_vertical * 0.03)
    image[false_mask] = np.stack([false_luma[false_mask]] * 3, axis=-1)

    sphere_cx = 548.0
    sphere_cy = 208.0
    sphere_radius = 50.0
    sphere_distance = np.sqrt((xx - sphere_cx) ** 2 + (yy - sphere_cy) ** 2)
    sphere_mask = sphere_distance <= sphere_radius
    sphere_vertical = np.clip((yy - (sphere_cy - sphere_radius)) / max(2.0 * sphere_radius, 1.0), 0.0, 1.0)
    sphere_luma = 0.48 - (sphere_vertical * 0.07)
    image[sphere_mask] = np.stack([sphere_luma[sphere_mask]] * 3, axis=-1)

    path = tmp_path / "bright_edge_sphere_vs_false_circle.tiff"
    tifffile.imwrite(path, np.clip(image * 65535.0, 0, 65535).astype(np.uint16), photometric="rgb")

    measured = _measure_rendered_preview_roi_ipp2(str(path), None)

    assert measured["detection_failed"] is False
    assert measured["sphere_detection_success"] is True
    assert measured["detected_sphere_roi"]["cx"] == pytest.approx(sphere_cx, abs=24.0)
    assert measured["detected_sphere_roi"]["cy"] == pytest.approx(sphere_cy, abs=24.0)


def test_measure_rendered_preview_roi_ipp2_fails_explicitly_when_no_sphere_candidate_exists(tmp_path: Path) -> None:
    image = np.zeros((320, 320, 3), dtype=np.float32)
    image[..., :] = 0.18
    path = tmp_path / "no_sphere.jpg"
    Image.fromarray(np.clip(image * 255.0, 0, 255).astype(np.uint8), mode="RGB").save(path, format="JPEG")

    measured = _measure_rendered_preview_roi_ipp2(str(path), None)

    assert measured["detection_failed"] is True
    assert measured["sphere_detection_success"] is False
    assert measured["sphere_detection_unresolved"] is False
    assert measured["sphere_roi_source"] == "failed"
    assert measured["gray_exposure_summary"] == "Sphere detection failed"


def test_measure_rendered_preview_roi_ipp2_reuses_original_roi_for_corrected_frames(tmp_path: Path) -> None:
    height = 320
    width = 320
    image = np.zeros((height, width, 3), dtype=np.float32)
    yy, xx = np.meshgrid(np.arange(height, dtype=np.float32), np.arange(width, dtype=np.float32), indexing="ij")
    cx = 220.0
    cy = 150.0
    radius = 50.0
    distance = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    sphere_mask = distance <= radius
    vertical = np.clip((yy - (cy - radius)) / max(2.0 * radius, 1.0), 0.0, 1.0)
    sphere_luma = 0.40 - (vertical * 0.15)
    for channel in range(3):
        image[..., channel] = np.where(sphere_mask, sphere_luma, image[..., channel])
    path = tmp_path / "corrected_like.jpg"
    Image.fromarray(np.clip(image * 255.0, 0, 255).astype(np.uint8), mode="RGB").save(path, format="JPEG")

    measured = _measure_rendered_preview_roi_ipp2(
        str(path),
        {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0},
        sphere_roi_override={"cx": cx, "cy": cy, "r": radius},
    )

    assert measured["detection_failed"] is False
    assert measured["sphere_detection_success"] is True
    assert measured["sphere_roi_source"] == "reused_from_original"
    assert measured["detected_sphere_roi"]["cx"] == pytest.approx(cx)


def test_measure_rendered_preview_roi_ipp2_returns_unresolved_when_sphere_candidates_fail_hard_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = np.full((96, 128, 3), 0.42, dtype=np.float32)
    path = tmp_path / "sphere_unresolved.jpg"
    Image.fromarray(np.clip(image * 255.0, 0, 255).astype(np.uint8), mode="RGB").save(path, format="JPEG")

    monkeypatch.setattr(
        "r3dmatch.report._detect_sphere_roi_in_region_hwc",
        lambda _region: {
            "source": "unresolved",
            "confidence": 0.24,
            "confidence_label": "UNRESOLVED",
            "roi": None,
            "sample_plausibility": "LOW",
            "sphere_detection_success": False,
            "sphere_detection_unresolved": True,
            "validation": {"reason": "candidates_rejected_by_hard_gate", "hard_gate_reasons": ["radial_coherence_below_min"]},
        },
    )

    measured = _measure_rendered_preview_roi_ipp2(str(path), None)

    assert measured["detection_failed"] is False
    assert measured["sphere_detection_success"] is False
    assert measured["sphere_detection_unresolved"] is True
    assert measured["gray_target_class"] == "unresolved"
    assert measured["gray_target_fallback_used"] is False
    assert measured["sphere_roi_source"] == "unresolved"
    assert measured["sample_plausibility"] == "LOW"


def test_compute_sphere_center_patch_uses_only_hero_center_pixels() -> None:
    region = np.full((240, 240, 3), np.asarray([0.70, 0.12, 0.12], dtype=np.float32), dtype=np.float32)
    yy, xx = np.meshgrid(np.arange(240, dtype=np.float32), np.arange(240, dtype=np.float32), indexing="ij")
    cx = 120.0
    cy = 118.0
    measurement_radius = 80.0
    hero_mask = ((xx - cx) ** 2 + (yy - cy) ** 2) <= (8.0 ** 2)
    region[hero_mask] = np.asarray([0.24, 0.24, 0.24], dtype=np.float32)

    patch = _compute_sphere_center_patch(region, cx, cy, measurement_radius)

    assert patch is not None
    assert patch["sample_count"] >= 64
    assert patch["center_luminance"] == pytest.approx(0.24, abs=0.02)
    assert tuple(patch["center_rgb"]) == pytest.approx((0.24, 0.24, 0.24), abs=0.02)


def test_measure_rendered_preview_roi_ipp2_uses_hero_center_as_authoritative_scalar(tmp_path: Path) -> None:
    height = 320
    width = 320
    image = np.full((height, width, 3), 0.05, dtype=np.float32)
    yy, xx = np.meshgrid(np.arange(height, dtype=np.float32), np.arange(width, dtype=np.float32), indexing="ij")
    cx = 168.0
    cy = 154.0
    radius = 72.0
    distance = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    sphere_mask = distance <= radius
    vertical = np.clip((yy - (cy - radius)) / max(2.0 * radius, 1.0), 0.0, 1.0)
    sphere_luma = 0.34 - (vertical * 0.10)
    image[sphere_mask] = np.stack([sphere_luma[sphere_mask]] * 3, axis=-1)
    outer_ring = sphere_mask & (distance >= (radius * 0.82))
    image[outer_ring] = np.asarray([0.62, 0.62, 0.62], dtype=np.float32)
    hero_center = distance <= (radius * 0.08)
    image[hero_center] = np.asarray([0.22, 0.22, 0.22], dtype=np.float32)
    path = tmp_path / "hero_authoritative.jpg"
    Image.fromarray(np.clip(image * 255.0, 0, 255).astype(np.uint8), mode="RGB").save(path, format="JPEG")

    measured = _measure_rendered_preview_roi_ipp2(
        str(path),
        {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0},
        sphere_roi_override={"cx": cx, "cy": cy, "r": radius},
    )

    assert measured["measurement_source"] == "hero_center_patch"
    assert measured["measurement_fallback_used"] is False
    assert measured["sample_2_ire"] == pytest.approx(22.0, abs=2.0)
    assert measured["center_ire"] == pytest.approx(22.0, abs=2.0)
    assert measured["measured_log2_luminance_monitoring"] == pytest.approx(np.log2(0.22), abs=0.15)


def test_measure_rendered_preview_roi_ipp2_uses_legacy_center_fallback_only_when_hero_patch_invalid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    height = 320
    width = 320
    image = np.full((height, width, 3), 0.05, dtype=np.float32)
    yy, xx = np.meshgrid(np.arange(height, dtype=np.float32), np.arange(width, dtype=np.float32), indexing="ij")
    cx = 160.0
    cy = 150.0
    radius = 70.0
    distance = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    sphere_mask = distance <= radius
    vertical = np.clip((yy - (cy - radius)) / max(2.0 * radius, 1.0), 0.0, 1.0)
    sphere_luma = 0.30 - (vertical * 0.08)
    image[sphere_mask] = np.stack([sphere_luma[sphere_mask]] * 3, axis=-1)
    path = tmp_path / "hero_fallback.jpg"
    Image.fromarray(np.clip(image * 255.0, 0, 255).astype(np.uint8), mode="RGB").save(path, format="JPEG")
    monkeypatch.setattr("r3dmatch.report._compute_sphere_center_patch", lambda *args, **kwargs: None)

    measured = _measure_rendered_preview_roi_ipp2(
        str(path),
        {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0},
        sphere_roi_override={"cx": cx, "cy": cy, "r": radius},
    )

    assert measured["measurement_source"] == "legacy_center_fallback"
    assert measured["measurement_fallback_used"] is True
    assert measured["sphere_detection_success"] is True


def test_validate_known_sphere_roi_rejects_flat_neutral_surface() -> None:
    image = np.full((180, 180, 3), 0.42, dtype=np.float32)

    validated = _validate_known_sphere_roi(
        image,
        roi=SphereROI(cx=90.0, cy=90.0, r=48.0),
        detection_source="primary_detected",
        confidence_hint=0.92,
    )

    assert validated["sphere_detection_success"] is False
    assert validated["sphere_detection_unresolved"] is True
    reasons = list((validated.get("hard_gate") or {}).get("reasons") or [])
    assert "flat_luminance_profile" in reasons or "samples_too_equal" in reasons
    assert "exposure_distribution_invalid" in reasons


def test_detect_sphere_roi_in_region_hwc_attempts_fallback_when_primary_candidates_are_all_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = np.full((220, 220, 3), 0.35, dtype=np.float32)
    primary_candidate = {
        "source": "primary_detected",
        "confidence": 0.76,
        "roi": {"cx": 110.0, "cy": 110.0, "r": 46.0},
        "validation": {"radius_sane": True},
    }
    fallback_calls = {"secondary": 0, "opencv_hough": 0, "opencv_hough_alt": 0, "opencv_contour": 0, "blob": 0}

    monkeypatch.setattr("r3dmatch.report._resize_region_for_sphere_detection", lambda region: (region, 1.0))
    monkeypatch.setattr("r3dmatch.report._detect_gray_card_in_region_hwc", lambda _region: {"roi": {}, "confidence": 0.0})

    def fake_detect(region: np.ndarray, *, search_bounds=None, detection_source: str, sigma=2.0, low_threshold=0.03, high_threshold=0.12, **_kwargs: object):  # type: ignore[no-untyped-def]
        if detection_source == "primary_detected":
            return [dict(primary_candidate)]
        fallback_calls["secondary"] += 1
        return []

    monkeypatch.setattr("r3dmatch.report._detect_sphere_candidates_in_region_hwc", fake_detect)
    monkeypatch.setattr(
        "r3dmatch.report._detect_sphere_candidates_opencv_hough_hwc",
        lambda *args, **kwargs: fallback_calls.__setitem__("opencv_hough", fallback_calls["opencv_hough"] + 1) or [],
    )
    monkeypatch.setattr(
        "r3dmatch.report._detect_sphere_candidates_opencv_hough_alt_hwc",
        lambda *args, **kwargs: fallback_calls.__setitem__("opencv_hough_alt", fallback_calls["opencv_hough_alt"] + 1) or [],
    )
    monkeypatch.setattr(
        "r3dmatch.report._detect_sphere_candidates_opencv_contours_hwc",
        lambda *args, **kwargs: fallback_calls.__setitem__("opencv_contour", fallback_calls["opencv_contour"] + 1) or [],
    )
    monkeypatch.setattr(
        "r3dmatch.report._detect_neutral_blob_candidates_in_region_hwc",
        lambda *args, **kwargs: fallback_calls.__setitem__("blob", fallback_calls["blob"] + 1) or [],
    )
    monkeypatch.setattr(
        "r3dmatch.report._rescore_sphere_candidates_with_profile",
        lambda _region, candidates, **_kwargs: [
            {
                **dict(candidates[0]),
                "hard_gate": {"passed": False, "reasons": ["flat_luminance_profile"], "metrics": {}},
                "profile_probe": {"score": 0.18},
                "sample_plausibility": "IMPLAUSIBLE",
                "confidence": 0.18,
                "confidence_label": "UNRESOLVED",
            }
        ],
    )

    detected = _detect_sphere_roi_in_region_hwc(image)

    assert detected["source"] == "unresolved"
    assert detected["sphere_detection_success"] is False
    assert detected["sphere_detection_unresolved"] is True
    assert detected["validation"]["reason"] == "primary_candidates_rejected_by_hard_gate"
    assert detected["validation"]["fallback_trigger_reason"] == "primary_candidates_rejected_by_hard_gate"
    assert fallback_calls == {"secondary": 1, "opencv_hough": 2, "opencv_hough_alt": 2, "opencv_contour": 2, "blob": 2}


def _write_scientific_validation_preview(tmp_path: Path) -> tuple[str, dict[str, object]]:
    height = 320
    width = 320
    image = np.zeros((height, width, 3), dtype=np.float32)
    yy, xx = np.meshgrid(np.arange(height, dtype=np.float32), np.arange(width, dtype=np.float32), indexing="ij")
    cx = 220.0
    cy = 150.0
    radius = 50.0
    distance = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    sphere_mask = distance <= radius
    vertical = np.clip((yy - (cy - radius)) / max(2.0 * radius, 1.0), 0.0, 1.0)
    sphere_luma = 0.40 - (vertical * 0.15)
    for channel in range(3):
        image[..., channel] = np.where(sphere_mask, sphere_luma, image[..., channel])
    path = tmp_path / "scientific_validation.jpg"
    Image.fromarray(np.clip(image * 255.0, 0, 255).astype(np.uint8), mode="RGB").save(path, format="JPEG")
    measured = _measure_rendered_preview_roi_ipp2(
        str(path),
        {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0},
        sphere_roi_override={"cx": cx, "cy": cy, "r": radius},
    )
    return str(path), measured


def _measurement_fingerprint_for_path(path: Path) -> dict[str, object]:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with Image.open(path) as image:
        return {
            "measurement_source_asset": {
                "path": str(path.resolve()),
                "exists_at_measurement_time": True,
                "file_size_bytes": int(path.stat().st_size),
                "sha256": digest,
                "image_dimensions": {"width": int(image.width), "height": int(image.height)},
            }
        }


def test_build_scientific_validation_artifacts_exports_traceable_example(tmp_path: Path) -> None:
    preview_path, measured = _write_scientific_validation_preview(tmp_path)
    diagnostics = dict(measured)
    diagnostics["rendered_measurement_preview_path"] = preview_path
    diagnostics["measurement_provenance"] = _measurement_fingerprint_for_path(Path(preview_path))
    analysis_record = {
        "clip_id": "CAM001",
        "analysis_path": str(tmp_path / "analysis" / "CAM001.analysis.json"),
        "diagnostics": diagnostics,
    }
    payload = {
        "recommended_strategy": {"strategy_key": "median"},
        "ipp2_validation": {
            "rows": [
                {
                    "clip_id": "CAM001",
                    "strategy_key": "median",
                    "ipp2_original_value_log2": float(diagnostics["measured_log2_luminance"]),
                    "ipp2_original_zone_profile": list(diagnostics["zone_measurements"]),
                }
            ]
        },
        "strategies": [
            {
                "strategy_key": "median",
                "clips": [
                    {
                        "clip_id": "CAM001",
                        "metrics": {
                            "exposure": {
                                "sample_1_ire": float(diagnostics["sample_1_ire"]),
                                "sample_2_ire": float(diagnostics["sample_2_ire"]),
                                "sample_3_ire": float(diagnostics["sample_3_ire"]),
                                "measured_log2_luminance_monitoring": float(diagnostics["measured_log2_luminance"]),
                                "measurement_domain": "perceptual",
                            }
                        },
                    }
                ],
            }
        ],
    }

    artifacts = _build_scientific_validation_artifacts(
        out_root=tmp_path / "report",
        analysis_records=[analysis_record],
        payload=payload,
    )

    assert Path(artifacts["path"]).exists()
    assert Path(artifacts["markdown_path"]).exists()
    summary = json.loads(Path(artifacts["path"]).read_text(encoding="utf-8"))
    assert summary["status"] == "fully_reconciled"
    assert summary["worked_example"]["clip_id"] == "CAM001"
    assert summary["consistency_validation"]["analysis_matches_recomputed"] is True
    assert summary["consistency_validation"]["ipp2_validation_matches_analysis"] is True
    assert summary["consistency_validation"]["report_matches_analysis"] is True
    assert summary["measurement_provenance"]["fingerprint_matches"] is True
    center_zone = next(item for item in summary["worked_example"]["zone_samples"] if item["label"] == "center")
    assert len(center_zone["pixel_preview"]["normalized_rgb_preview"]) > 0
    assert len(center_zone["luminance_preview"]) > 0


def test_build_scientific_validation_artifacts_fails_on_report_ire_drift(tmp_path: Path) -> None:
    preview_path, measured = _write_scientific_validation_preview(tmp_path)
    diagnostics = dict(measured)
    diagnostics["rendered_measurement_preview_path"] = preview_path
    diagnostics["measurement_provenance"] = _measurement_fingerprint_for_path(Path(preview_path))
    analysis_record = {
        "clip_id": "CAM001",
        "analysis_path": str(tmp_path / "analysis" / "CAM001.analysis.json"),
        "diagnostics": diagnostics,
    }
    payload = {
        "recommended_strategy": {"strategy_key": "median"},
        "ipp2_validation": {
            "rows": [
                {
                    "clip_id": "CAM001",
                    "strategy_key": "median",
                    "ipp2_original_value_log2": float(diagnostics["measured_log2_luminance"]),
                    "ipp2_original_zone_profile": list(diagnostics["zone_measurements"]),
                }
            ]
        },
        "strategies": [
            {
                "strategy_key": "median",
                "clips": [
                    {
                        "clip_id": "CAM001",
                        "metrics": {
                            "exposure": {
                                "sample_1_ire": float(diagnostics["sample_1_ire"]) + 0.25,
                                "sample_2_ire": float(diagnostics["sample_2_ire"]),
                                "sample_3_ire": float(diagnostics["sample_3_ire"]),
                                "measured_log2_luminance_monitoring": float(diagnostics["measured_log2_luminance"]),
                                "measurement_domain": "perceptual",
                            }
                        },
                    }
                ],
            }
        ],
    }

    with pytest.raises(RuntimeError, match="report IRE/scalar drift"):
        _build_scientific_validation_artifacts(
            out_root=tmp_path / "report",
            analysis_records=[analysis_record],
            payload=payload,
            fail_on_analysis_drift=True,
        )


def test_build_scientific_validation_artifacts_reports_analysis_drift_state(tmp_path: Path) -> None:
    preview_path, measured = _write_scientific_validation_preview(tmp_path)
    diagnostics = dict(measured)
    diagnostics["rendered_measurement_preview_path"] = preview_path
    diagnostics["measurement_provenance"] = _measurement_fingerprint_for_path(Path(preview_path))
    analysis_record = {
        "clip_id": "CAM001",
        "analysis_path": str(tmp_path / "analysis" / "CAM001.analysis.json"),
        "diagnostics": diagnostics,
    }
    payload = {
        "recommended_strategy": {"strategy_key": "median"},
        "ipp2_validation": {
            "rows": [
                {
                    "clip_id": "CAM001",
                    "strategy_key": "median",
                    "ipp2_original_value_log2": float(diagnostics["measured_log2_luminance"]),
                    "ipp2_original_zone_profile": list(diagnostics["zone_measurements"]),
                }
            ]
        },
        "strategies": [
            {
                "strategy_key": "median",
                "clips": [
                    {
                        "clip_id": "CAM001",
                        "metrics": {
                            "exposure": {
                                "sample_1_ire": float(diagnostics["sample_1_ire"]) + 0.25,
                                "sample_2_ire": float(diagnostics["sample_2_ire"]),
                                "sample_3_ire": float(diagnostics["sample_3_ire"]),
                                "measured_log2_luminance_monitoring": float(diagnostics["measured_log2_luminance"]),
                                "measurement_domain": "perceptual",
                            }
                        },
                    }
                ],
            }
        ],
    }

    artifacts = _build_scientific_validation_artifacts(
        out_root=tmp_path / "report",
        analysis_records=[analysis_record],
        payload=payload,
    )
    summary = json.loads(Path(artifacts["path"]).read_text(encoding="utf-8"))
    assert summary["status"] == "analysis_drift"
    assert summary["consistency_validation"]["report_matches_analysis"] is False


def test_build_scientific_validation_artifacts_reports_blocked_asset_mismatch_state(tmp_path: Path) -> None:
    preview_path, measured = _write_scientific_validation_preview(tmp_path)
    diagnostics = dict(measured)
    diagnostics["rendered_measurement_preview_path"] = preview_path
    diagnostics["measurement_provenance"] = _measurement_fingerprint_for_path(Path(preview_path))
    diagnostics["measurement_provenance"]["measurement_source_asset"]["sha256"] = "0" * 64
    analysis_record = {
        "clip_id": "CAM001",
        "analysis_path": str(tmp_path / "analysis" / "CAM001.analysis.json"),
        "diagnostics": diagnostics,
    }
    payload = {
        "recommended_strategy": {"strategy_key": "median"},
        "ipp2_validation": {
            "rows": [
                {
                    "clip_id": "CAM001",
                    "strategy_key": "median",
                    "ipp2_original_value_log2": float(diagnostics["measured_log2_luminance"]),
                    "ipp2_original_zone_profile": list(diagnostics["zone_measurements"]),
                }
            ]
        },
        "strategies": [
            {
                "strategy_key": "median",
                "clips": [
                    {
                        "clip_id": "CAM001",
                        "metrics": {
                            "exposure": {
                                "sample_1_ire": float(diagnostics["sample_1_ire"]),
                                "sample_2_ire": float(diagnostics["sample_2_ire"]),
                                "sample_3_ire": float(diagnostics["sample_3_ire"]),
                                "measured_log2_luminance_monitoring": float(diagnostics["measured_log2_luminance"]),
                                "measurement_domain": "perceptual",
                            }
                        },
                    }
                ],
            }
        ],
    }

    artifacts = _build_scientific_validation_artifacts(
        out_root=tmp_path / "report",
        analysis_records=[analysis_record],
        payload=payload,
    )
    summary = json.loads(Path(artifacts["path"]).read_text(encoding="utf-8"))
    assert summary["status"] == "blocked_asset_mismatch"
    assert summary["measurement_provenance"]["status"] == "fingerprint_mismatch"


def test_measurement_values_for_record_uses_sphere_sample_median_for_perceptual_domain() -> None:
    record = {
        "clip_id": "G007_A063_032563_001",
        "diagnostics": {
            "exposure_measurement_domain": "rendered_preview_ipp2",
            "measured_log2_luminance_monitoring": 9.0,
            "zone_measurements": [
                {"label": "bright_side", "measured_log2_luminance": -1.8, "measured_ire": 28.717458874925873},
                {"label": "center", "measured_log2_luminance": -1.2, "measured_ire": 43.5275281648062},
                {"label": "dark_side", "measured_log2_luminance": -2.4, "measured_ire": 18.94645708137998},
            ],
        },
    }

    values = _measurement_values_for_record(record, matching_domain="perceptual")

    assert values["display_scalar_log2"] == pytest.approx(-1.8, abs=1e-6)
    assert values["display_scalar_ire"] == pytest.approx(28.717458874925873, abs=1e-6)
    assert values["log2_luminance"] == pytest.approx(-1.8, abs=1e-6)
    assert len(values["zone_measurements"]) == 3


def test_measurement_values_for_record_keeps_unresolved_measurement_non_numeric() -> None:
    record = {
        "clip_id": "G007_A067_0403RE_001",
        "diagnostics": {
            "exposure_measurement_domain": "rendered_preview_ipp2",
            "measurement_valid": False,
            "gray_target_measurement_valid": False,
            "gray_target_class": "unresolved",
            "measured_log2_luminance_monitoring": None,
            "measured_log2_luminance": None,
            "sample_1_ire": None,
            "sample_2_ire": None,
            "sample_3_ire": None,
            "zone_measurements": [],
        },
    }

    values = _measurement_values_for_record(record, matching_domain="perceptual")

    assert values["measurement_valid"] is False
    assert values["display_scalar_log2"] is None
    assert values["display_scalar_ire"] is None
    assert values["log2_luminance"] is None


def test_contact_sheet_resolve_measurement_fields_skips_unresolved_scalar_placeholders() -> None:
    resolved = _contact_sheet_resolve_measurement_fields(
        clip_id="G007_A067_0403RE_001",
        clip_row={"measurement_valid": False},
        shared={"measurement_valid": False, "display_scalar_log2": None},
        exposure_metrics={"measurement_valid": False},
        ipp2_row={},
    )

    assert resolved["measurement_valid"] is False
    assert resolved["sample_1_ire"] is None
    assert resolved["display_scalar_log2"] is None
    assert resolved["target_sample_label"] == "Gray target unresolved"


def test_derived_profile_scalar_uses_median_sample_value() -> None:
    zone_measurements = [
        {"label": "bright_side", "measured_log2_luminance": -2.0, "measured_ire": 25.0},
        {"label": "center", "measured_log2_luminance": -1.0, "measured_ire": 50.0},
        {"label": "dark_side", "measured_log2_luminance": 0.0, "measured_ire": 100.0},
    ]

    derived = _derived_profile_scalar(zone_measurements)

    assert derived["derived_display_scalar_log2"] == pytest.approx(-1.0, abs=1e-6)
    assert derived["derived_display_scalar_ire"] == pytest.approx(50.0, abs=1e-6)
    assert derived["derived_exposure_value_log2"] == pytest.approx(-1.0, abs=1e-6)
    assert derived["derived_exposure_ire"] == pytest.approx(50.0, abs=1e-6)
    assert derived["derivation_method"] == "median_sample_log2"
    assert derived["derivation_domain"] == "display_ipp2"


def test_sphere_detection_label_thresholds() -> None:
    assert _sphere_detection_label(0.85) == "HIGH"
    assert _sphere_detection_label(0.60) == "MEDIUM"
    assert _sphere_detection_label(0.35) == "LOW"
    assert _sphere_detection_label(0.10) == "FAILED"


def test_sphere_detection_note_includes_recovery_labels() -> None:
    assert _sphere_detection_note("primary_detected", "HIGH") == "Sphere check verified"
    assert _sphere_detection_note("localized_recovery", "MEDIUM", sphere_detection_success=True, fallback_used=True) == "Alternate detection path used"
    assert _sphere_detection_note("unresolved", "UNRESOLVED", sphere_detection_success=False, sphere_detection_unresolved=True) == "Gray target needs review"


def test_sphere_detection_prefers_off_center_gray_sphere_over_center_distractor() -> None:
    height, width = 320, 520
    image = np.full((height, width, 3), 0.12, dtype=np.float32)
    yy, xx = np.meshgrid(np.arange(height, dtype=np.float32), np.arange(width, dtype=np.float32), indexing="ij")

    true_cx, true_cy, true_r = 118.0, 174.0, 54.0
    distractor_cx, distractor_cy, distractor_r = 300.0, 160.0, 54.0

    true_distance = np.sqrt((xx - true_cx) ** 2 + (yy - true_cy) ** 2)
    distractor_distance = np.sqrt((xx - distractor_cx) ** 2 + (yy - distractor_cy) ** 2)

    true_mask = true_distance <= true_r
    true_ring = np.abs(true_distance - true_r) <= 2.0
    true_vertical = np.clip((yy - (true_cy - true_r)) / max(2.0 * true_r, 1.0), 0.0, 1.0)
    true_luma = 0.31 - (true_vertical * 0.08)
    image[true_mask] = np.stack([true_luma[true_mask]] * 3, axis=-1)
    image[true_ring] = np.asarray([0.82, 0.82, 0.82], dtype=np.float32)

    distractor_mask = distractor_distance <= distractor_r
    distractor_ring = np.abs(distractor_distance - distractor_r) <= 2.0
    image[distractor_mask] = np.asarray([0.24, 0.11, 0.32], dtype=np.float32)
    image[distractor_ring] = np.asarray([0.86, 0.34, 0.18], dtype=np.float32)
    checker = ((np.floor((xx - distractor_cx + distractor_r) / 8.0) + np.floor((yy - distractor_cy + distractor_r) / 8.0)) % 2) == 0
    noisy_mask = distractor_mask & checker
    image[noisy_mask] = np.asarray([0.46, 0.07, 0.55], dtype=np.float32)

    detected = _detect_sphere_roi_in_region_hwc(np.clip(image, 0.0, 1.0))
    roi = dict(detected.get("roi") or {})

    assert detected["source"] != "failed"
    assert roi
    assert abs(float(roi["cx"]) - true_cx) < 24.0
    assert abs(float(roi["cy"]) - true_cy) < 24.0
    assert detected["confidence_label"] in {"HIGH", "MEDIUM", "LOW"}


def test_measure_rendered_preview_recovers_sphere_outside_calibration_roi(tmp_path: Path) -> None:
    tifffile = pytest.importorskip("tifffile")
    height, width = 320, 520
    image = np.full((height, width, 3), 0.08, dtype=np.float32)
    yy, xx = np.meshgrid(np.arange(height, dtype=np.float32), np.arange(width, dtype=np.float32), indexing="ij")
    sphere_cx, sphere_cy, sphere_r = 430.0, 170.0, 48.0
    sphere_distance = np.sqrt((xx - sphere_cx) ** 2 + (yy - sphere_cy) ** 2)
    sphere_mask = sphere_distance <= sphere_r
    sphere_vertical = np.clip((yy - (sphere_cy - sphere_r)) / max(2.0 * sphere_r, 1.0), 0.0, 1.0)
    sphere_luma = 0.29 - (sphere_vertical * 0.08)
    image[sphere_mask] = np.stack([sphere_luma[sphere_mask]] * 3, axis=-1)
    image[np.abs(sphere_distance - sphere_r) <= 2.0] = np.asarray([0.78, 0.78, 0.78], dtype=np.float32)
    image[:, :220, :] = np.asarray([0.18, 0.18, 0.18], dtype=np.float32)
    preview_path = tmp_path / "preview.tiff"
    tifffile.imwrite(preview_path, np.clip(image * 65535.0, 0, 65535).astype(np.uint16), photometric="rgb")

    measured = _measure_rendered_preview_roi_ipp2(
        str(preview_path),
        {"x": 0.00, "y": 0.00, "w": 0.46, "h": 1.00},
    )

    detected_roi = dict(measured.get("detected_sphere_roi") or {})
    assert measured["gray_target_class"] == "sphere"
    assert measured["measurement_region_scope"] == "full_frame"
    assert measured["sample_plausibility"] in {"HIGH", "MEDIUM"}
    assert detected_roi
    assert abs(float(detected_roi["cx"]) - sphere_cx) < 28.0
    assert abs(float(detected_roi["cy"]) - sphere_cy) < 28.0


def test_ipp2_closed_loop_target_uses_original_zone_profiles_for_median_anchor() -> None:
    rows = [
        {
            "clip_id": "A",
            "trust_class": "TRUSTED",
            "original_ipp2_zone_profile": [
                {"label": "upper_mid", "measured_log2_luminance": -2.0, "measured_ire": 25.0},
                {"label": "center", "measured_log2_luminance": -2.2, "measured_ire": 22.0},
                {"label": "lower_mid", "measured_log2_luminance": -2.4, "measured_ire": 19.0},
            ],
            "initial_ipp2_zone_profile": [
                {"label": "upper_mid", "measured_log2_luminance": -4.0, "measured_ire": 6.0},
                {"label": "center", "measured_log2_luminance": -4.0, "measured_ire": 6.0},
                {"label": "lower_mid", "measured_log2_luminance": -4.0, "measured_ire": 6.0},
            ],
        },
        {
            "clip_id": "B",
            "trust_class": "TRUSTED",
            "original_ipp2_zone_profile": [
                {"label": "upper_mid", "measured_log2_luminance": -1.8, "measured_ire": 29.0},
                {"label": "center", "measured_log2_luminance": -2.0, "measured_ire": 25.0},
                {"label": "lower_mid", "measured_log2_luminance": -2.2, "measured_ire": 22.0},
            ],
            "initial_ipp2_zone_profile": [
                {"label": "upper_mid", "measured_log2_luminance": -4.0, "measured_ire": 6.0},
                {"label": "center", "measured_log2_luminance": -4.0, "measured_ire": 6.0},
                {"label": "lower_mid", "measured_log2_luminance": -4.0, "measured_ire": 6.0},
            ],
        },
    ]

    target = _ipp2_closed_loop_target(strategy_key="median", reference_clip_id=None, rows=rows, anchor_mode="median")

    assert target["target_source"] == "trusted_camera_median_original_ipp2_profile"
    assert target["target_profile_summary"] != "Top 6 / Mid 6 / Bottom 6 IRE"
    assert target["target_zone_profile"][0]["measured_log2_luminance"] == pytest.approx(-1.9)
    assert target["target_zone_profile"][1]["measured_log2_luminance"] == pytest.approx(-2.1)
    assert target["target_zone_profile"][2]["measured_log2_luminance"] == pytest.approx(-2.3)


def test_round_to_standard_stop_fraction_prefers_standard_lens_steps() -> None:
    assert round_to_standard_stop_fraction(1.48) == pytest.approx(1.5)
    assert round_to_standard_stop_fraction(-1.48) == pytest.approx(-1.5)
    assert round_to_standard_stop_fraction(0.62) == pytest.approx(2.0 / 3.0)
    assert round_to_standard_stop_fraction(0.74) == pytest.approx(0.75)


def test_format_stop_string_formats_human_readable_stop_labels() -> None:
    assert format_stop_string(1.0) == "1 stop"
    assert format_stop_string(0.5) == "1/2 stop"
    assert format_stop_string(1.0 / 3.0) == "1/3 stop"
    assert format_stop_string(0.25) == "1/4 stop"
    assert format_stop_string(2.5) == "2 1/2 stops"


def test_operator_guidance_marks_large_corrections_as_outliers() -> None:
    guidance = _operator_guidance_for_correction(
        correction_stops=-3.2,
        residual_stops=0.18,
        validation_status="FAIL",
    )
    assert guidance["status"] == "OUTLIER"
    assert guidance["direction"] == "close"
    assert guidance["suggested_action"] == "Close aperture by 3 1/4 stops"
    assert "Verify T-Stop" in guidance["notes"]


def test_operator_guidance_handles_zero_correction_without_direction_bias() -> None:
    guidance = _operator_guidance_for_correction(
        correction_stops=0.0,
        residual_stops=0.0,
        validation_status="PASS",
    )
    assert guidance["status"] == "PASS"
    assert guidance["direction"] == "hold"
    assert guidance["suggested_action"] == "No aperture adjustment required"


def test_operator_guidance_rounds_tiny_corrections_down_to_no_adjustment() -> None:
    guidance = _operator_guidance_for_correction(
        correction_stops=0.02,
        residual_stops=0.0,
        validation_status="PASS",
    )
    assert guidance["direction"] == "hold"
    assert guidance["suggested_action"] == "No aperture adjustment required"


def test_ipp2_validation_presentation_distinguishes_outlier_outcomes() -> None:
    landed = _ipp2_validation_presentation(
        {
            "status": "PASS",
            "ipp2_residual_stops": 0.03,
            "ipp2_profile_max_residual_stops": 0.28,
            "applied_correction_stops": 3.0,
            "ipp2_gray_exposure_summary": "Sample 1 40 / Sample 2 31 / Sample 3 26 IRE",
            "ipp2_target_gray_exposure_summary": "Sample 1 41 / Sample 2 31 / Sample 3 26 IRE",
            "ipp2_zone_residuals": [
                {"label": "bright_side", "residual_stops": 0.03},
                {"label": "center", "residual_stops": -0.01},
                {"label": "dark_side", "residual_stops": 0.02},
            ],
        }
    )
    assert landed["presentation_state"] == "Outlier corrected successfully"
    assert landed["validation_label"] == "Verify T-Stop"
    assert landed["gray_exposure_text"] == "Gray Exposure: Sample 1 40 / Sample 2 31 / Sample 3 26 IRE"
    assert landed["target_profile_text"] == "Reference profile: Sample 1 41 / Sample 2 31 / Sample 3 26 IRE"
    assert "Sample 1 +0.03" in landed["zone_residual_text"]
    assert landed["residual_text"] == "Exposure residual after validation: 0.03 stops"
    assert landed["profile_residual_text"] == "Worst zone residual after validation: 0.28 stops"
    assert landed["result_label"] == "Outlier"

    review = _ipp2_validation_presentation(
        {
            "status": "REVIEW",
            "ipp2_residual_stops": 0.08,
            "ipp2_profile_max_residual_stops": 0.08,
            "applied_correction_stops": -3.0,
        }
    )
    assert review["presentation_state"] == "Outlier needs review"

    failed = _ipp2_validation_presentation(
        {
            "status": "FAIL",
            "ipp2_residual_stops": 0.18,
            "ipp2_profile_max_residual_stops": 0.18,
            "applied_correction_stops": -3.0,
        }
    )
    assert failed["presentation_state"] == "Outlier still outside tolerance"


def test_build_sphere_detection_artifacts_writes_overlay_images_and_summary(tmp_path: Path) -> None:
    image = np.zeros((240, 320, 3), dtype=np.float32)
    yy, xx = np.meshgrid(np.arange(240, dtype=np.float32), np.arange(320, dtype=np.float32), indexing="ij")
    cx = 220.0
    cy = 120.0
    radius = 44.0
    distance = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    mask = distance <= radius
    image[..., :] = 0.05
    image[mask] = 0.30
    original_path = tmp_path / "CAM001.original.jpg"
    corrected_path = tmp_path / "CAM001.corrected.jpg"
    Image.fromarray(np.clip(image * 255.0, 0, 255).astype(np.uint8), mode="RGB").save(original_path, format="JPEG")
    Image.fromarray(np.clip(image * 255.0, 0, 255).astype(np.uint8), mode="RGB").save(corrected_path, format="JPEG")

    summary = {
        "rows": [
            {
                "strategy_key": "median",
                "clip_id": "CAM001",
                "camera_id": "CAM001",
                "calibration_roi": None,
                "original_image_path": str(original_path),
                "corrected_image_path": str(corrected_path),
                "ipp2_original_detection_source": "primary_detected",
                "ipp2_original_detection_confidence": 0.81,
                "ipp2_original_detection_label": "HIGH",
                "ipp2_original_detected_sphere_roi": {"cx": cx, "cy": cy, "r": radius},
                "ipp2_detection_source": "reused_from_original",
                "ipp2_detection_confidence": 1.0,
                "ipp2_detection_label": "HIGH",
                "ipp2_detected_sphere_roi": {"cx": cx, "cy": cy, "r": radius},
                "ipp2_original_zone_profile": [
                    {"label": "upper_mid", "display_label": "Top", "measured_ire": 39.0, "bounds": {"pixel": {"x0": 196, "y0": 96, "x1": 244, "y1": 114}}},
                    {"label": "center", "display_label": "Mid", "measured_ire": 31.0, "bounds": {"pixel": {"x0": 196, "y0": 114, "x1": 244, "y1": 132}}},
                    {"label": "lower_mid", "display_label": "Bottom", "measured_ire": 26.0, "bounds": {"pixel": {"x0": 196, "y0": 132, "x1": 244, "y1": 150}}},
                ],
                "ipp2_zone_profile": [
                    {"label": "upper_mid", "display_label": "Top", "measured_ire": 39.0, "bounds": {"pixel": {"x0": 196, "y0": 96, "x1": 244, "y1": 114}}},
                    {"label": "center", "display_label": "Mid", "measured_ire": 31.0, "bounds": {"pixel": {"x0": 196, "y0": 114, "x1": 244, "y1": 132}}},
                    {"label": "lower_mid", "display_label": "Bottom", "measured_ire": 26.0, "bounds": {"pixel": {"x0": 196, "y0": 132, "x1": 244, "y1": 150}}},
                ],
                "detection_failed": False,
            }
        ]
    }

    artifacts = _build_sphere_detection_artifacts(
        out_root=tmp_path,
        validation_summary=summary,
        recommended_strategy_key="median",
    )

    assert Path(artifacts["path"]).exists()
    assert Path(artifacts["overlay_root"]).joinpath("CAM001.original_detection.png").exists()
    assert Path(artifacts["overlay_root"]).joinpath("CAM001.corrected_detection.png").exists()
    assert artifacts["summary"]["confidence_counts"]["HIGH"] == 1


def test_desktop_results_summary_highlights_target_basis_and_plausibility(tmp_path: Path) -> None:
    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    (analysis_dir / "CAM001.analysis.json").write_text(
        json.dumps(
            {
                "clip_id": "CAM001",
                "confidence": 0.88,
                "diagnostics": {
                    "gray_target_class": "sphere",
                    "gray_target_detection_method": "opencv_contour_recovery",
                    "sample_plausibility": "HIGH",
                    "gray_target_review_recommended": False,
                    "measured_log2_luminance_monitoring": -1.18,
                },
            }
        ),
        encoding="utf-8",
    )
    html = _build_desktop_results_summary(
        {
            "contact_sheet_payload": {
                "recommended_strategy": {
                    "strategy_label": "Optimal Exposure",
                    "strategy_summary": "Use the best match to the measured sphere.",
                    "reference_clip_id": "CAM001",
                },
                "hero_recommendation": {"candidate_clip_id": "CAM001"},
                "run_assessment": {"status": "READY", "operator_note": "Run is ready."},
                "white_balance_model": {"model_label": "Shared Kelvin / Per-Camera Tint", "candidate_count": 4, "diagnostics": {"mean_neutral_residual_after_solve": 0.002, "source_sample_count": 12}},
                "gray_target_consistency": {"dominant_target_class": "sphere", "summary": "Retained cameras consistently measured the gray sphere.", "mixed_target_classes": False},
                "recommended_attention": ["CAM001"],
                "selected_clip_groups": ["068"],
                "executive_synopsis": "Optimal Exposure is the recommended strategy.",
            },
            "corrected_validation": {
                "recommended_strategy_key": "",
                "best_residual_stops": 0.001,
                "worst_residual_stops": 0.004,
                "outside_tolerance_cameras": [],
                "rows": [
                    {
                        "clip_id": "CAM001",
                        "camera_label": "CAM001",
                        "applied_correction_stops": 0.08,
                        "reference_use": "Included",
                        "trust_class": "TRUSTED",
                        "trust_reason": "Sphere solve is stable.",
                    }
                ],
            },
            "scientific_validation": {"status": "fully_reconciled", "reason": "Replay matched the archived report."},
        },
        tmp_path,
    )
    assert "Target:" in html
    assert "Solve:" in html
    assert "Plausibility:" in html
    assert "Opencv Contour Recovery" in html


def test_parameter_specs_use_documented_fallback_dividers() -> None:
    assert PARAMETER_SPECS["exposureAdjust"].default_divider == 1000
    assert PARAMETER_SPECS["kelvin"].default_divider == 1
    assert PARAMETER_SPECS["tint"].default_divider == 1000


def test_build_periodic_get_payload_normalizes_parameter_id() -> None:
    payload = build_periodic_get_payload(parameter_id="RCP_PARAM_TINT", interval_ms=500)
    assert payload == {"type": "rcp_get_periodic_on", "id": "TINT", "interval_ms": 500}


def test_collect_matching_messages_ignores_stale_pending_matches(monkeypatch: pytest.MonkeyPatch) -> None:
    client = JsonWebSocketClient(
        host="10.20.61.191",
        port=9998,
        connect_timeout_ms=100,
        operation_timeout_ms=100,
        settle_timeout_ms=10,
    )
    session = Rcp2WebSocketSession(sock=types.SimpleNamespace(), host="10.20.61.191", port=9998)
    session.message_counter = 5
    session.pending_messages.append(
        {
            "type": "rcp_cur_int",
            "id": "EXPOSURE_ADJUST",
            "cur": {"val": 12},
            "_r3dmatch_message_index": 4,
        }
    )
    calls = {"count": 0}

    def fake_recv_json(_session: object, _timeout_s: float) -> dict[str, object]:
        calls["count"] += 1
        if calls["count"] == 1:
            return {
                "type": "rcp_cur_int",
                "id": "EXPOSURE_ADJUST",
                "cur": {"val": 15},
                "_r3dmatch_message_index": 6,
            }
        raise Rcp2WebSocketTimeout("done")

    monkeypatch.setattr(client, "recv_json", fake_recv_json)
    messages = client.collect_matching_messages(
        session,
        matcher=lambda message: str(message.get("id") or "") == "EXPOSURE_ADJUST",
        timeout_ms=100,
        settle_timeout_ms=10,
        minimum_message_index=5,
    )
    assert len(messages) == 1
    assert messages[0]["cur"]["val"] == 15


def _write_test_preview(path: Path, color: tuple[int, int, int] = (128, 128, 128)) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (320, 200), color).save(path, format="JPEG")


def _write_minimal_array_calibration(path: Path, *, clip_id: str = "G007_B057_0324YT_001") -> None:
    payload = {
        "schema": "r3dmatch_array_calibration_v1",
        "capture_id": "test_capture",
        "created_at": "2026-03-26T00:00:00Z",
        "input_path": str(path.parent),
        "mode": "array_gray_sphere",
        "backend": "red",
        "measurement_domain": "scene",
        "group_key": "array_test",
        "target": {
            "method": "robust_array_center",
            "exposure": {
                "log2_luminance_target": -2.473931188332412,
                "estimator": "median",
                "included_camera_count": 1,
                "excluded_camera_count": [],
            },
            "color": {
                "target_rgb_chromaticity": [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
                "estimator": "median",
                "included_camera_count": 1,
                "excluded_camera_count": [],
            },
        },
        "global_scene_intent": {
            "enabled": False,
            "global_exposure_offset_stops": 0.0,
            "notes": None,
            "white_balance_model": {
                "model_key": "shared_kelvin_per_camera_tint",
                "model_label": "Shared Kelvin / Per-Camera Tint",
                "shared_kelvin": 5600,
                "shared_tint": 0.0,
            },
        },
        "cameras": [
            {
                "clip_id": clip_id,
                "source_path": str(path.parent / f"{clip_id}.R3D"),
                "camera_id": "G007_B057",
                "group_key": "array_test",
                "measurement": {
                    "gray_sample_count": 3,
                    "valid_pixel_count": 3000,
                    "measured_log2_luminance": -2.473931188332412,
                    "measured_rgb_mean": [0.18, 0.18, 0.18],
                    "measured_rgb_chromaticity": [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
                    "saturation_fraction": 0.0,
                    "black_fraction": 0.0,
                    "neutral_sample_log2_spread": 0.01,
                    "neutral_sample_chromaticity_spread": 0.001,
                    "as_shot_kelvin": 5600.0,
                    "as_shot_tint": 0.0,
                },
                "solution": {
                    "exposure_offset_stops": 0.0,
                    "rgb_gains": [1.0, 1.0, 1.0],
                    "luminance_preserving_gain_normalization": True,
                    "final_exposure_offset_with_global_intent": 0.0,
                    "kelvin": 5600,
                    "tint": 0.0,
                    "saturation": 1.0,
                    "derivation_method": "neutral_axis_shared_kelvin_per_camera_tint_v2",
                },
                "quality": {
                    "confidence": 0.95,
                    "exposure_residual_stops": 0.0,
                    "color_residual": 0.0,
                    "flags": [],
                    "post_exposure_residual_stops": 0.0,
                    "post_color_residual": 0.0,
                    "neutral_sample_log2_spread": 0.01,
                    "neutral_sample_chromaticity_spread": 0.001,
                },
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _apply_fake_cdl(image: np.ndarray, cdl: Optional[dict], *, enabled: bool) -> np.ndarray:
    if not enabled or not isinstance(cdl, dict):
        return image
    slope = np.asarray(cdl.get("slope", [1.0, 1.0, 1.0]), dtype=np.float32)
    offset = np.asarray(cdl.get("offset", [0.0, 0.0, 0.0]), dtype=np.float32)
    power = np.asarray(cdl.get("power", [1.0, 1.0, 1.0]), dtype=np.float32)
    saturation = float(cdl.get("saturation", 1.0))
    corrected = np.clip(image * slope.reshape((1, 1, 3)) + offset.reshape((1, 1, 3)), 0.0, 1.0)
    corrected = np.power(np.clip(corrected, 0.0, 1.0), power.reshape((1, 1, 3)))
    luma = corrected[..., 0] * 0.2126 + corrected[..., 1] * 0.7152 + corrected[..., 2] * 0.0722
    return np.clip(luma[..., None] + (corrected - luma[..., None]) * saturation, 0.0, 1.0)


def _fake_redline_base_image(command: list[str]) -> np.ndarray:
    base = np.zeros((18, 32, 3), dtype=np.float32)
    if "--useMeta" in command:
        base[..., 0] = 80.0 / 255.0
        base[..., 1] = 90.0 / 255.0
        base[..., 2] = 100.0 / 255.0
    else:
        base[..., 0] = 70.0 / 255.0
        base[..., 1] = 80.0 / 255.0
        base[..., 2] = 90.0 / 255.0
    return base


def _fake_redline_command_state(command: list[str]) -> dict:
    exposure = 0.0
    kelvin = None
    tint = 0.0
    red_gain = 1.0
    green_gain = 1.0
    blue_gain = 1.0
    rmd_payload = None
    if "--loadRMD" in command:
        rmd_path = Path(command[command.index("--loadRMD") + 1])
        rmd_payload = json.loads(rmd_path.read_text(encoding="utf-8"))
        exposure = float(rmd_payload.get("exposure", 0.0) or 0.0)
        gains = rmd_payload.get("rgb_gains")
        if gains:
            red_gain = float(gains[0])
            green_gain = float(gains[1])
            blue_gain = float(gains[2])
    else:
        if "--exposureAdjust" in command:
            exposure = float(command[command.index("--exposureAdjust") + 1])
        elif "--exposure" in command:
            exposure = float(command[command.index("--exposure") + 1])
        if "--kelvin" in command:
            kelvin = int(float(command[command.index("--kelvin") + 1]))
        if "--tint" in command:
            tint = float(command[command.index("--tint") + 1])
        red_gain = float(command[command.index("--redGain") + 1]) if "--redGain" in command else red_gain
        green_gain = float(command[command.index("--greenGain") + 1]) if "--greenGain" in command else green_gain
        blue_gain = float(command[command.index("--blueGain") + 1]) if "--blueGain" in command else blue_gain
    return {
        "exposure": exposure,
        "kelvin": kelvin,
        "tint": tint,
        "red_gain": red_gain,
        "green_gain": green_gain,
        "blue_gain": blue_gain,
        "rmd_payload": rmd_payload,
    }


def _apply_fake_redline_direct_controls(image: np.ndarray, *, kelvin: Optional[int], tint: float) -> np.ndarray:
    corrected = np.asarray(image, dtype=np.float32).copy()
    if kelvin is not None:
        warmth = float(np.clip((float(kelvin) - 5600.0) / 4000.0, -0.5, 0.5))
        corrected[..., 0] *= 1.0 + (warmth * 0.20)
        corrected[..., 2] *= 1.0 - (warmth * 0.20)
    if abs(float(tint)) > 1e-9:
        tint_shift = float(np.clip(float(tint) / 10.0, -0.25, 0.25))
        corrected[..., 1] *= 1.0 - (tint_shift * 0.12)
        corrected[..., 0] *= 1.0 + (tint_shift * 0.06)
        corrected[..., 2] *= 1.0 + (tint_shift * 0.06)
    return np.clip(corrected, 0.0, 1.0)


def _install_fake_redline(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_write_rmd_for_clip_with_metadata(clip_id: str, payload: dict, out_dir):  # type: ignore[no-untyped-def]
        out_path = Path(out_dir) / f"{clip_id}.RMD"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cdl_payload = payload.get("rmd_mapping", {}).get("color", {}).get("cdl")
        cdl_enabled = bool(payload.get("rmd_mapping", {}).get("color", {}).get("cdl_enabled"))
        rmd_payload = {
            "clip_id": clip_id,
            "exposure": float(payload.get("rmd_mapping", {}).get("exposure", {}).get("final_offset_stops", 0.0) or 0.0),
            "rgb_gains": payload.get("rmd_mapping", {}).get("color", {}).get("rgb_neutral_gains"),
            "cdl": cdl_payload,
            "cdl_enabled": cdl_enabled,
        }
        out_path.write_text(json.dumps(rmd_payload), encoding="utf-8")
        return out_path, {
            "rmd_kind": "red_sdk",
            "path": str(out_path),
            "settings": {
                "cdl_enabled": cdl_enabled,
                "exposure_adjust": rmd_payload["exposure"],
            },
        }

    def fake_run(command, cwd=None, env=None, text=True):  # type: ignore[no-untyped-def]
        from PIL import Image

        if "--help" in command:
            help_text = "\n".join(
                [
                    "REDline Build 65.1.3   64bit Public Release",
                    "--loadRMD <filename>",
                    "--useRMD <int>",
                    "--exposure <float>",
                    "--exposureAdjust <float>",
                    "--kelvin <int>",
                    "--tint <float>",
                    "--redGain <float>",
                    "--greenGain <float>",
                    "--blueGain <float>",
                    "--gammaCurve <int>",
                    "--colorSpace <int>",
                    "--outputToneMap <int>",
                    "--rollOff <int>",
                    "--shadow <float>",
                    "--lut <filename>",
                ]
            )
            return types.SimpleNamespace(returncode=0, stdout=help_text, stderr="")
        if "--printMeta" in command:
            if "--loadRMD" in command:
                rmd_path = Path(command[command.index("--loadRMD") + 1])
                rmd_payload = json.loads(rmd_path.read_text(encoding="utf-8"))
                return types.SimpleNamespace(
                    returncode=0,
                    stdout=(
                        "RMD metadata loaded\n"
                        f"Exposure={rmd_payload.get('exposure', 0.0)}\n"
                        f"RGBGains={rmd_payload.get('rgb_gains')}\n"
                    ),
                    stderr="",
                )
            return types.SimpleNamespace(returncode=0, stdout="CLI metadata loaded\n", stderr="")
        output_path = Path(command[command.index("--o") + 1])
        generated_path = output_path.with_name(f"{output_path.name}.000000.jpg")
        generated_path.parent.mkdir(parents=True, exist_ok=True)
        state = _fake_redline_command_state(command)
        exposure = float(state["exposure"])
        red_gain = float(state["red_gain"])
        green_gain = float(state["green_gain"])
        blue_gain = float(state["blue_gain"])
        base = _fake_redline_base_image(command)
        base = _apply_fake_redline_direct_controls(
            base,
            kelvin=cast(Optional[int], state["kelvin"]),
            tint=float(state["tint"]),
        )
        base[..., 0] = np.clip(base[..., 0] * (2.0**exposure) * red_gain, 0, 1)
        base[..., 1] = np.clip(base[..., 1] * (2.0**exposure) * green_gain, 0, 1)
        base[..., 2] = np.clip(base[..., 2] * (2.0**exposure) * blue_gain, 0, 1)
        if "--loadRMD" in command:
            rmd_payload = cast(Optional[dict], state["rmd_payload"]) or {}
            base = _apply_fake_cdl(base, rmd_payload.get("cdl"), enabled=bool(rmd_payload.get("cdl_enabled")))
        output_array = np.clip(base * 255.0, 0, 255).astype(np.uint8)
        if output_path.suffix.lower() in {".tif", ".tiff"}:
            Image.fromarray(output_array, mode="RGB").resize((96, 96), resample=Image.Resampling.NEAREST).save(
                generated_path,
                format="TIFF",
            )
        else:
            Image.fromarray(output_array, mode="RGB").save(generated_path, format="JPEG")
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("r3dmatch.report.write_rmd_for_clip_with_metadata", fake_write_rmd_for_clip_with_metadata)
    monkeypatch.setattr("r3dmatch.report.run_cancellable_subprocess", fake_run)


def test_backend_selection_defaults_to_mock() -> None:
    backend = resolve_backend("mock")
    assert isinstance(backend, MockR3DBackend)


def test_backend_selection_auto_falls_back_to_mock() -> None:
    backend = resolve_backend("auto")
    assert isinstance(backend, MockR3DBackend)


def test_red_backend_raises_clear_error_when_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RED_SDK_ROOT", raising=False)
    monkeypatch.setattr("r3dmatch.sdk._load_red_native_module", lambda: None)
    with pytest.raises(RuntimeError, match="RED_SDK_ROOT is not set"):
        RedSdkDecoder()


def test_resolve_red_sdk_configuration_uses_external_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    sdk_root = tmp_path / "R3DSDKv9_2_0"
    (sdk_root / "Include").mkdir(parents=True)
    (sdk_root / "Lib" / "mac64").mkdir(parents=True)
    _write_fake_red_redistributables(sdk_root / "Redistributable" / "mac")
    monkeypatch.setenv("RED_SDK_ROOT", str(sdk_root))
    monkeypatch.delenv("RED_SDK_REDISTRIBUTABLE_DIR", raising=False)

    config = resolve_red_sdk_configuration()

    assert config.root == sdk_root
    assert config.include_dir == sdk_root / "Include"
    assert config.library_dir == sdk_root / "Lib" / "mac64"
    assert config.redistributable_dir == sdk_root / "Redistributable" / "mac"
    assert config.redistributable_source == "sdk_root"
    assert config.errors == ()


def test_resolve_red_sdk_configuration_uses_bundled_redistributable_when_root_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    bundled = tmp_path / "bundle" / "red_runtime" / "redistributable"
    bundled.mkdir(parents=True)
    for filename in ("REDDecoder.dylib", "REDMetal.dylib", "REDOpenCL.dylib", "REDR3D.dylib"):
        (bundled / filename).write_bytes(b"runtime")
    monkeypatch.delenv("RED_SDK_ROOT", raising=False)
    monkeypatch.delenv("RED_SDK_REDISTRIBUTABLE_DIR", raising=False)
    monkeypatch.setattr("r3dmatch.sdk.bundled_red_redistributable_dir", lambda: bundled)

    config = resolve_red_sdk_configuration()

    assert config.root is None
    assert config.redistributable_dir == bundled
    assert config.redistributable_source == "bundled_app"
    assert config.errors == ()


def test_red_sdk_configuration_error_is_actionable() -> None:
    config = RedSdkConfiguration(
        root=None,
        include_dir=None,
        library_dir=None,
        redistributable_dir=None,
        errors=("RED_SDK_ROOT is not set.",),
    )
    message = red_sdk_configuration_error(config)
    assert "RED_SDK_ROOT is not set." in message
    assert "scripts/build_red_sdk_bridge.sh" in message
    assert "RED_SDK_REDISTRIBUTABLE_DIR" in message


def test_load_configured_red_native_module_rejects_bridge_root_mismatch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    sdk_root = tmp_path / "R3DSDKv9_2_0"
    (sdk_root / "Include").mkdir(parents=True)
    (sdk_root / "Lib" / "mac64").mkdir(parents=True)
    _write_fake_red_redistributables(sdk_root / "Redistributable" / "mac")
    monkeypatch.setenv("RED_SDK_ROOT", str(sdk_root))
    fake_module = types.SimpleNamespace(
        bridge_configuration=lambda: {
            "compiled_red_sdk_root": "/Users/sfouasnon/Desktop/R3DMatch/src/RED_SDK/R3DSDKv9_2_0"
        }
    )
    monkeypatch.setattr("r3dmatch.sdk._load_red_native_module", lambda: fake_module)

    with pytest.raises(RuntimeError, match="built against a different SDK root"):
        load_configured_red_native_module()


def test_red_backend_uses_native_module_when_available(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    sdk_root = tmp_path / "R3DSDKv9_2_0"
    (sdk_root / "Include").mkdir(parents=True)
    (sdk_root / "Lib" / "mac64").mkdir(parents=True)
    _write_fake_red_redistributables(sdk_root / "Redistributable" / "mac")
    monkeypatch.setenv("RED_SDK_ROOT", str(sdk_root))
    fake_module = types.SimpleNamespace(
        sdk_available=lambda: True,
        bridge_configuration=lambda: {"compiled_red_sdk_root": str(sdk_root)},
        read_metadata=lambda path: {
            "source_path": path,
            "original_filename": Path(path).name,
            "fps": 23.976,
            "width": 2048,
            "height": 1152,
            "total_frames": 2,
            "color_space": "REDWideGamutRGB",
            "gamma_curve": "Log3G10",
        },
        decode_frame=lambda path, frame_index, half_res, colorspace, gamma: np.zeros((4, 5, 3), dtype=np.float32),
    )
    monkeypatch.setattr("r3dmatch.sdk._load_red_native_module", lambda: fake_module)
    decoder = RedSdkDecoder()
    clip_path = tmp_path / "G007_D060_0324M6_001.R3D"
    clip_path.write_bytes(b"")
    clip = decoder.inspect_clip(str(clip_path))
    frame_index, _, frame = list(decoder.decode_frames(str(clip_path), start_frame=0, max_frames=1, frame_step=1))[0]
    assert clip.clip_id == "G007_D060_0324M6_001"
    assert clip.group_key == "G007_D060"
    assert clip.original_filename == "G007_D060_0324M6_001.R3D"
    assert frame_index == 0
    assert frame.shape == (3, 4, 5)


def test_red_backend_wraps_decode_errors_with_clip_context(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    sdk_root = tmp_path / "R3DSDKv9_2_0"
    (sdk_root / "Include").mkdir(parents=True)
    (sdk_root / "Lib" / "mac64").mkdir(parents=True)
    _write_fake_red_redistributables(sdk_root / "Redistributable" / "mac")
    monkeypatch.setenv("RED_SDK_ROOT", str(sdk_root))
    native_detail = (
        "RED SDK frame decode failed. "
        "clip_path=/tmp/test/G007_D060_0324M6_001.R3D frame_index=0 "
        "decode_width=1024 decode_height=576 half_res=true "
        "colorspace=REDWideGamutRGB gamma=Log3G10 decode_attempts=[DECODE_HALF_RES_GOOD=7]"
    )
    fake_module = types.SimpleNamespace(
        sdk_available=lambda: True,
        bridge_configuration=lambda: {"compiled_red_sdk_root": str(sdk_root)},
        read_metadata=lambda path: {
            "fps": 24.0,
            "width": 2048,
            "height": 1152,
            "total_frames": 1,
            "color_space": "REDWideGamutRGB",
            "gamma_curve": "Log3G10",
        },
        decode_frame=lambda path, frame_index, half_res, colorspace, gamma: (_ for _ in ()).throw(RuntimeError(native_detail)),
    )
    monkeypatch.setattr("r3dmatch.sdk._load_red_native_module", lambda: fake_module)
    decoder = RedSdkDecoder()
    clip_path = tmp_path / "G007_D060_0324M6_001.R3D"
    clip_path.write_bytes(b"")
    with pytest.raises(RuntimeError, match="decode frame 0") as excinfo:
        list(decoder.decode_frames(str(clip_path), start_frame=0, max_frames=1, frame_step=1))
    message = str(excinfo.value)
    assert "frame_index=0" in message
    assert "decode_attempts=[DECODE_HALF_RES_GOOD=7]" in message
    assert native_detail in message


def test_clip_id_and_group_key_extraction_for_red_name() -> None:
    filename = "G007_D060_0324M6_001.R3D"
    assert Path(filename).stem == "G007_D060_0324M6_001"
    assert group_key_from_clip_id(Path(filename).stem) == "G007_D060"


def test_rmd_name_uses_exact_clip_id() -> None:
    assert rmd_name_for_clip_id("G007_D060_0324M6_001") == "G007_D060_0324M6_001.RMD"


def test_grouping_does_not_affect_rmd_name() -> None:
    clip_id = "G007_D060_0324M6_001"
    assert group_key_from_clip_id(clip_id) == "G007_D060"
    assert rmd_name_for_clip_id(clip_id) == "G007_D060_0324M6_001.RMD"


def test_exact_sidecar_naming_from_clip_id() -> None:
    assert sidecar_filename_for_clip_id("G007_D060_0324M6_001") == "G007_D060_0324M6_001.sidecar.json"


def test_exact_rmd_naming_from_clip_id() -> None:
    assert rmd_filename_for_clip_id("G007_D060_0324M6_001") == "G007_D060_0324M6_001.RMD"


def test_preview_path_generation_from_clip_id() -> None:
    assert preview_filename_for_clip_id("G007_D060_0324M6_001", "both") == "G007_D060_0324M6_001.both.review.tiff"
    assert (
        preview_filename_for_clip_id("G007_D060_0324M6_001", "both", strategy="median", run_id="batch01")
        == "G007_D060_0324M6_001.both.review.median.batch01.tiff"
    )


def test_desktop_ui_review_command_uses_tcsh_and_preview_options() -> None:
    command = build_review_command(
        repo_root="/Users/sfouasnon/Desktop/R3DMatch",
        input_path="/tmp/in",
        output_path="/tmp/out",
        backend="red",
        target_type="gray_sphere",
        processing_mode="both",
        roi_x="0.1",
        roi_y="0.2",
        roi_w="0.3",
        roi_h="0.4",
        target_strategies=["median", "manual"],
        reference_clip_id="G007_D060_0324M6_001",
        preview_mode="monitoring",
        preview_lut="/tmp/show.cube",
    )
    assert command[1:4] == ["-m", "r3dmatch.cli", "review-calibration"]
    assert "--preview-mode" in command
    assert "monitoring" in command
    assert "--preview-still-format" in command
    assert "tiff" in command
    assert "--artifact-mode" in command
    assert "production" in command
    assert "/tmp/show.cube" in command
    assert "G007_D060_0324M6_001" in command


def test_scan_calibration_sources_discovers_rdc_media(tmp_path: Path) -> None:
    rdc = tmp_path / "A001_C001_0001AB.RDC"
    rdc.mkdir()
    clip = rdc / "A001_C001_0001AB_001.R3D"
    clip.write_bytes(b"")
    hidden = rdc / "._A001_C001_0001AB_002.R3D"
    hidden.write_bytes(b"")
    summary = scan_calibration_sources(str(tmp_path))
    assert summary["clip_count"] == 1
    assert summary["rdc_count"] == 1
    assert summary["r3d_count"] == 1
    assert summary["clip_ids"] == ["A001_C001_0001AB_001"]
    assert [group["group_id"] for group in summary["clip_groups"]] == ["001"]
    assert summary["warning"] is None


def test_normalize_preview_image_array_scales_uint8_and_uint16() -> None:
    uint8 = np.array([[[0, 128, 255]]], dtype=np.uint8)
    uint16 = np.array([[[0, 32768, 65535]]], dtype=np.uint16)
    normalized8, meta8 = _normalize_preview_image_array(uint8)
    normalized16, meta16 = _normalize_preview_image_array(uint16)
    assert normalized8.dtype == np.float32
    assert normalized16.dtype == np.float32
    assert float(normalized8[0, 0, 2]) == pytest.approx(1.0)
    assert float(normalized8[0, 0, 1]) == pytest.approx(128.0 / 255.0, rel=1e-6)
    assert float(normalized16[0, 0, 2]) == pytest.approx(1.0)
    assert float(normalized16[0, 0, 1]) == pytest.approx(32768.0 / 65535.0, rel=1e-6)
    assert meta8["normalization_denominator"] == 255.0
    assert meta16["normalization_denominator"] == 65535.0


def test_load_preview_image_as_normalized_rgb_preserves_uint16_tiff_metadata(tmp_path: Path) -> None:
    tifffile = pytest.importorskip("tifffile")
    tiff_path = tmp_path / "preview.tiff"
    source = np.array([[[0, 32768, 65535]]], dtype=np.uint16)
    tifffile.imwrite(tiff_path, source, photometric="rgb")
    normalized, metadata = _load_preview_image_as_normalized_rgb(tiff_path)
    assert normalized.dtype == np.float32
    assert float(normalized[0, 0, 2]) == pytest.approx(1.0)
    assert float(normalized[0, 0, 1]) == pytest.approx(32768.0 / 65535.0, rel=1e-6)
    assert metadata["bit_depth"] == 16
    assert metadata["normalization_denominator"] == 65535.0
    assert metadata["preview_format"] == "tiff"


def test_load_preview_image_as_normalized_rgb_retries_truncated_jpeg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    jpeg_path = tmp_path / "preview.jpg"
    Image.fromarray(np.full((4, 4, 3), 160, dtype=np.uint8), mode="RGB").save(jpeg_path, format="JPEG")
    real_open = Image.open
    attempts = {"count": 0}

    def flaky_open(path, *args, **kwargs):  # type: ignore[no-untyped-def]
        if str(path) == str(jpeg_path) and attempts["count"] < 2:
            attempts["count"] += 1
            raise OSError("image file is truncated")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr("r3dmatch.report.Image.open", flaky_open)
    monkeypatch.setattr("r3dmatch.report.time.sleep", lambda _seconds: None)
    monkeypatch.setattr("r3dmatch.report._wait_for_file_ready", lambda _path, **_kwargs: None)

    normalized, metadata = _load_preview_image_as_normalized_rgb(jpeg_path)

    assert attempts["count"] == 2
    assert normalized.shape == (4, 4, 3)
    assert metadata["preview_format"] == "jpg"
    assert metadata["bit_depth"] == 8


def test_load_preview_image_as_normalized_rgb_retries_truncated_tiff_until_ready(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tiff_path = tmp_path / "preview.tiff"
    Image.fromarray(np.full((4, 4, 3), 220, dtype=np.uint8), mode="RGB").save(tiff_path, format="TIFF")
    attempts = {"count": 0}
    real_loader = _load_preview_image_as_normalized_rgb

    def flaky_once(path):  # type: ignore[no-untyped-def]
        if str(path) == str(tiff_path) and attempts["count"] < 3:
            attempts["count"] += 1
            raise OSError("image file is truncated (156 bytes not processed)")
        return original_once(path)

    original_once = __import__("r3dmatch.report", fromlist=["_load_preview_image_once"])._load_preview_image_once
    monkeypatch.setattr("r3dmatch.report._load_preview_image_once", flaky_once)
    monkeypatch.setattr("r3dmatch.report.time.sleep", lambda _seconds: None)
    monkeypatch.setattr("r3dmatch.report._wait_for_file_ready", lambda _path, **_kwargs: None)

    normalized, metadata = real_loader(tiff_path)

    assert attempts["count"] == 3
    assert normalized.shape == (4, 4, 3)
    assert metadata["preview_format"] == "tiff"


def test_validate_rendered_preview_output_rejects_tiny_tiff_stub(tmp_path: Path) -> None:
    stub_path = tmp_path / "preview.tiff"
    stub_path.write_bytes(
        bytes.fromhex(
            "49492a00080000000c000001030001000000000f0000010103000100000070080000"
            "02010300030000009e00000003010300010000000100000006010300010000000200"
            "00000a01030001000000010000001101040001000000080000001201030001000000"
            "01000000150103000100000003000000160103000100000070080000170104000100"
            "0000000000001c010300010000000100000000000000100010001000"
        )
    )

    with pytest.raises(OSError, match="below sane minimum size|invalid render stub"):
        _validate_rendered_preview_output(stub_path)


def test_manual_assist_sanitized_entry_keeps_coordinate_lists() -> None:
    payload = _manual_assist_sanitized_entry(
        {
            "clip_id": "CAM001",
            "center_preview_px": [1004, 329],
            "radius_preview_px": 72,
            "center_normalized": [],
            "operator_note": "Manual sphere center placed from low-bandwidth preview.",
        }
    )

    assert payload["clip_id"] == "CAM001"
    assert payload["center_preview_px"] == [1004, 329]
    assert payload["radius_preview_px"] == 72
    assert "center_normalized" not in payload


def test_refine_manual_assist_sphere_roi_runs_local_search_until_valid(monkeypatch: pytest.MonkeyPatch) -> None:
    image = np.ones((120, 120, 3), dtype=np.float32) * 0.4

    def fake_validate(region: np.ndarray, *, roi, detection_source: str, confidence_hint=None, manual_assist_metadata=None):  # type: ignore[no-untyped-def]
        passed = abs(float(roi.cx) - 66.0) < 1.0 and abs(float(roi.cy) - 54.0) < 1.0 and abs(float(roi.r) - 19.55) < 0.6
        margin = 7.5 if passed else 0.2
        return {
            "source": detection_source,
            "roi": {"cx": float(roi.cx), "cy": float(roi.cy), "r": float(roi.r)},
            "profile_probe": {
                "score": 0.96 if passed else 0.52,
                "review_required": not passed,
            },
            "hard_gate": {
                "passed": passed,
                "reasons": [] if passed else ["center_too_close_to_extremum"],
                "metrics": {
                    "center_margin_ire": margin,
                },
            },
            "sample_plausibility": "HIGH" if passed else "LOW",
            "confidence": 0.94 if passed else 0.31,
            "confidence_label": "HIGH" if passed else "UNRESOLVED",
            "sphere_detection_success": passed,
            "sphere_detection_unresolved": not passed,
        }

    def fake_detect(region: np.ndarray) -> dict[str, object]:
        return {
            "source": "unresolved",
            "confidence": 0.2,
            "confidence_label": "UNRESOLVED",
            "sphere_detection_success": False,
            "sphere_detection_unresolved": True,
            "candidate_diagnostics": {"rescored_candidates": []},
        }

    monkeypatch.setattr("r3dmatch.report._validate_known_sphere_roi", fake_validate)
    monkeypatch.setattr("r3dmatch.report._detect_sphere_roi_in_region_hwc", fake_detect)

    refinement = _refine_manual_assist_sphere_roi(
        image,
        {
            "cx": 72.0,
            "cy": 48.0,
            "r": 23.0,
        },
    )

    assert refinement["refined"] is True
    assert refinement["reason"] == "accepted_manual_local_search"
    assert refinement["source"] == "manual_operator_assist_local_search"


def test_upgrade_trust_from_ipp2_validation_recovers_sphere_camera() -> None:
    upgraded = _upgrade_trust_from_ipp2_validation(
        {
            "trust_class": "EXCLUDED",
            "trust_score": 0.31,
            "trust_reason": "Outside primary exposure group",
            "stability_label": "Excluded from reference",
            "correction_confidence": "LOW",
            "reference_use": "Excluded",
            "screened_reasons": ["Outside primary exposure group", "outside central exposure cluster"],
        },
        ipp2_validation={
            "status": "PASS",
            "profile_audit_label": "Profile mismatch",
            "validation_residual_stops": 0.03,
        },
        gray_target_class="sphere",
        sample_plausibility="MEDIUM",
    )

    assert upgraded["trust_class"] == "USE_WITH_CAUTION"
    assert upgraded["reference_use"] == "Included"
    assert "Recovered on corrected-frame validation" in upgraded["screened_reasons"]


def test_upgrade_trust_from_ipp2_validation_preserves_non_sphere_target() -> None:
    upgraded = _upgrade_trust_from_ipp2_validation(
        {
            "trust_class": "EXCLUDED",
            "trust_score": 0.2,
            "trust_reason": "Outside primary exposure group",
            "reference_use": "Excluded",
            "screened_reasons": ["Outside primary exposure group"],
        },
        ipp2_validation={
            "status": "PASS",
            "profile_audit_label": "Profile aligned",
            "validation_residual_stops": 0.01,
        },
        gray_target_class="gray_card",
        sample_plausibility="HIGH",
    )

    assert upgraded["trust_class"] == "EXCLUDED"
    assert upgraded["reference_use"] == "Excluded"


def test_detect_focus_chart_roi_hwc_finds_colorchecker_cluster() -> None:
    pytest.importorskip("cv2")
    image = np.ones((420, 820, 3), dtype=np.float32) * 0.72
    rows, cols = 4, 6
    start_x, start_y = 620, 120
    cell_w, cell_h = 24, 28
    gap = 4
    palette = [
        (0.82, 0.28, 0.26),
        (0.27, 0.53, 0.82),
        (0.18, 0.68, 0.36),
        (0.86, 0.72, 0.22),
        (0.76, 0.34, 0.78),
        (0.32, 0.79, 0.78),
    ]
    for row in range(rows):
        for col in range(cols):
            x0 = start_x + (col * (cell_w + gap))
            y0 = start_y + (row * (cell_h + gap))
            image[y0:y0 + cell_h, x0:x0 + cell_w, :] = palette[(row + col) % len(palette)]
            image[y0:y0 + cell_h, x0:x0 + 2, :] = 0.08
            image[y0:y0 + cell_h, x0 + cell_w - 2:x0 + cell_w, :] = 0.08
            image[y0:y0 + 2, x0:x0 + cell_w, :] = 0.08
            image[y0 + cell_h - 2:y0 + cell_h, x0:x0 + cell_w, :] = 0.08
    detection = _detect_focus_chart_roi_hwc(image)
    assert detection["found"] is True
    roi = dict(detection["roi"])
    assert roi["x0"] < start_x
    assert roi["x1"] > start_x + (cols * (cell_w + gap)) - gap
    assert detection["confidence"] > 0.5


def test_focus_metric_bundle_prefers_sharp_image() -> None:
    image = np.zeros((240, 240, 3), dtype=np.float32)
    image[:, ::12, :] = 1.0
    image[::12, :, :] = 1.0
    sharp_metrics = _focus_metric_bundle(image)
    blurred = np.asarray(Image.fromarray(np.clip(image * 255.0, 0.0, 255.0).astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius=3)), dtype=np.float32) / 255.0
    blur_metrics = _focus_metric_bundle(blurred)
    assert sharp_metrics["laplacian_variance"] > blur_metrics["laplacian_variance"]
    assert sharp_metrics["tenengrad"] > blur_metrics["tenengrad"]
    assert sharp_metrics["fft_high_frequency_energy"] > blur_metrics["fft_high_frequency_energy"]


def test_focus_validation_summary_marks_tiff_sufficient_for_matching_ranks() -> None:
    rows = [
        {"clip_id": "A", "laplacian_variance": 12.0, "tenengrad": 9.0, "fft_high_frequency_energy": 0.6, "composite_focus_score": 1.0},
        {"clip_id": "B", "laplacian_variance": 8.0, "tenengrad": 6.0, "fft_high_frequency_energy": 0.4, "composite_focus_score": 0.75},
        {"clip_id": "C", "laplacian_variance": 4.0, "tenengrad": 3.0, "fft_high_frequency_energy": 0.2, "composite_focus_score": 0.42},
    ]
    reference_rows = [
        {"clip_id": "A", "laplacian_variance": 11.5, "tenengrad": 8.7, "fft_high_frequency_energy": 0.58, "composite_focus_score": 0.98},
        {"clip_id": "B", "laplacian_variance": 7.8, "tenengrad": 5.8, "fft_high_frequency_energy": 0.39, "composite_focus_score": 0.73},
        {"clip_id": "C", "laplacian_variance": 4.2, "tenengrad": 3.1, "fft_high_frequency_energy": 0.22, "composite_focus_score": 0.44},
    ]
    summary = _focus_validation_summary(rows=rows, reference_rows=reference_rows)
    assert summary["tiff_is_sufficient"] is True
    assert summary["confidence"] == "high"
    assert summary["rank_correlation"] > 0.95


def test_build_review_command_includes_focus_validation_flag() -> None:
    command = build_review_command(
        repo_root="/Users/sfouasnon/Desktop/R3DMatch",
        input_path="/tmp/input",
        output_path="/tmp/output",
        backend="red",
        target_type="gray_card",
        processing_mode="both",
        roi_x=None,
        roi_y=None,
        roi_w=None,
        roi_h=None,
        target_strategies=["median"],
        reference_clip_id=None,
        preview_mode="monitoring",
        preview_lut=None,
        preview_still_format="tiff",
        artifact_mode="production",
        focus_validation=True,
    )
    assert "--focus-validation" in command


def test_scan_calibration_sources_warns_on_empty_folder(tmp_path: Path) -> None:
    summary = scan_calibration_sources(str(tmp_path))
    assert summary["clip_count"] == 0
    assert "No valid RED" in str(summary["warning"])


def test_web_scan_sources_discovers_rdc_media(tmp_path: Path) -> None:
    rdc = tmp_path / "A001_C001_0001AB.RDC"
    rdc.mkdir()
    clip = rdc / "A001_C001_0001AB_001.R3D"
    clip.write_bytes(b"")
    summary = scan_sources(str(tmp_path))
    assert summary["clip_count"] == 1
    assert summary["sample_clip_ids"] == ["A001_C001_0001AB_001"]
    assert summary["warning"] is None


def test_web_scan_sources_discovers_subset_groups(tmp_path: Path) -> None:
    for name in [
        "G007_A063_0325EV_001.R3D",
        "G007_B063_0325EV_001.R3D",
        "G007_A064_0325EV_001.R3D",
    ]:
        (tmp_path / name).write_bytes(b"")
    summary = scan_sources(str(tmp_path))
    assert summary["clip_count"] == 3
    assert [group["group_id"] for group in summary["clip_groups"]] == ["063", "064"]
    assert summary["clip_groups"][0]["clip_count"] == 2
    assert subset_key_from_clip_id("G007_B063_0325EV_001") == "063"
    assert summary["clip_records"][0]["camera_label"] == "A"


def test_web_subset_selection_ui_switches_between_group_and_manual_modes(tmp_path: Path) -> None:
    for name in [
        "G007_A063_0325EV_001.R3D",
        "G007_B063_0325EV_001.R3D",
        "G007_A064_0325EV_001.R3D",
    ]:
        (tmp_path / name).write_bytes(b"")
    scan = scan_sources(str(tmp_path))

    group_form = {
        "selected_clip_ids": ["G007_A064_0325EV_001"],
        "selected_clip_groups": ["063"],
        "advanced_clip_selection": False,
    }
    _normalize_subset_form(group_form, scan)
    assert group_form["selected_clip_ids"] == []
    assert _resolve_selected_clip_ids(group_form, scan) == ["G007_A063_0325EV_001", "G007_B063_0325EV_001"]
    group_ui = _subset_selection_ui(group_form, scan)
    assert group_ui["summary_text"] == "2 clips selected from group 063"
    assert group_ui["mode_label"] == "Group Mode"

    manual_form = {
        "selected_clip_ids": ["G007_A064_0325EV_001"],
        "selected_clip_groups": ["063"],
        "advanced_clip_selection": True,
    }
    _normalize_subset_form(manual_form, scan)
    assert manual_form["selected_clip_groups"] == []
    assert _resolve_selected_clip_ids(manual_form, scan) == ["G007_A064_0325EV_001"]
    manual_ui = _subset_selection_ui(manual_form, scan)
    assert manual_ui["summary_text"] == "Manual selection: 1 clip"
    assert manual_ui["group_panel_note"] == "Clip groups are disabled while manual clip selection is active."


def test_build_review_web_command_uses_tcsh_and_preview_options(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("r3dmatch.web_app.sys.executable", "/tmp/venv/bin/python")
    monkeypatch.setenv("RED_SDK_ROOT", "/external/red/sdk")
    monkeypatch.setenv("RED_SDK_REDISTRIBUTABLE_DIR", "/external/red/sdk/Redistributable/mac")
    monkeypatch.setenv("DYLD_FALLBACK_LIBRARY_PATH", "/opt/homebrew/lib:/opt/homebrew/opt/glib/lib")
    command = build_review_web_command(
        "/Users/sfouasnon/Desktop/R3DMatch",
        {
            "input_path": "/tmp/in",
            "output_path": "/tmp/out",
            "run_label": "",
            "backend": "red",
            "target_type": "gray_sphere",
            "processing_mode": "both",
            "matching_domain": "scene",
            "roi_x": "0.1",
            "roi_y": "0.2",
            "roi_w": "0.3",
            "roi_h": "0.4",
            "selected_clip_ids": [],
            "selected_clip_groups": [],
            "target_strategies": ["median", "manual"],
            "reference_clip_id": "G007_D060_0324M6_001",
            "hero_clip_id": "",
            "review_mode": "full_contact_sheet",
            "report_focus": "outliers",
            "preview_mode": "monitoring",
            "preview_lut": "/tmp/show.cube",
        },
    )
    joined = " ".join(command)
    assert command[0] == "/bin/tcsh"
    assert "setenv PYTHONPATH" in joined
    assert "/tmp/venv/bin/python -m r3dmatch.cli review-calibration" in joined
    assert "setenv DYLD_FALLBACK_LIBRARY_PATH /opt/homebrew/lib:/opt/homebrew/opt/glib/lib" in joined
    assert "setenv RED_SDK_ROOT /external/red/sdk" in joined
    assert "setenv RED_SDK_REDISTRIBUTABLE_DIR /external/red/sdk/Redistributable/mac" in joined
    assert "setenv R3DMATCH_INVOCATION_SOURCE web_ui" in joined
    assert "--review-mode full_contact_sheet" in joined
    assert "--report-focus outliers" in joined
    assert "--preview-mode monitoring" in joined
    assert "--preview-lut /tmp/show.cube" in joined
    assert "--reference-clip-id G007_D060_0324M6_001" in joined


def test_build_review_web_command_uses_embedded_cli_prefix_when_frozen(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("r3dmatch.web_app.sys.executable", "/tmp/R3DMatch.app/Contents/MacOS/R3DMatch")
    monkeypatch.setattr("r3dmatch.web_app.sys.frozen", True, raising=False)
    command = build_review_web_command(
        "/Users/sfouasnon/Desktop/R3DMatch",
        {
            "input_path": "/tmp/in",
            "output_path": "/tmp/out",
            "run_label": "",
            "backend": "mock",
            "target_type": "gray_sphere",
            "processing_mode": "both",
            "matching_domain": "perceptual",
            "roi_x": "",
            "roi_y": "",
            "roi_w": "",
            "roi_h": "",
            "selected_clip_ids": [],
            "selected_clip_groups": [],
            "target_strategies": ["median"],
            "reference_clip_id": "",
            "hero_clip_id": "",
            "review_mode": "full_contact_sheet",
            "report_focus": "auto",
            "preview_mode": "monitoring",
            "preview_lut": "",
        },
    )
    joined = " ".join(command)
    assert "/tmp/R3DMatch.app/Contents/MacOS/R3DMatch --cli review-calibration" in joined
    assert "-m r3dmatch.cli" not in joined


def test_build_tcsh_launch_prefix_uses_bundle_runtime_when_frozen(monkeypatch: pytest.MonkeyPatch) -> None:
    executable = str((Path("/tmp/R3DMatch.app/Contents/MacOS/R3DMatch")).resolve())
    monkeypatch.setattr("r3dmatch.web_app.sys.executable", executable)
    monkeypatch.setattr("r3dmatch.web_app.sys.frozen", True, raising=False)
    monkeypatch.setenv("DYLD_FALLBACK_LIBRARY_PATH", "/opt/homebrew/lib")
    monkeypatch.setenv("RED_SDK_ROOT", "/external/red/sdk")
    prefix = _build_tcsh_launch_prefix("/Users/sfouasnon/Desktop/R3DMatch", invocation_source="web_ui")
    assert f'cd "{Path(executable).parent}"' in prefix
    assert 'setenv PYTHONPATH "$PWD/src"' not in prefix
    assert "setenv DYLD_FALLBACK_LIBRARY_PATH /opt/homebrew/lib" in prefix
    assert "setenv RED_SDK_ROOT /external/red/sdk" in prefix
    assert "setenv R3DMATCH_INVOCATION_SOURCE web_ui" in prefix


def test_web_launcher_dispatches_embedded_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    from r3dmatch import web_launcher

    captured: dict[str, object] = {}

    def fake_cli_app() -> None:
        captured["argv"] = list(web_launcher.sys.argv)

    monkeypatch.setattr("r3dmatch.web_launcher.sys.argv", ["R3DMatch", "--cli", "runtime-health", "--strict"])
    monkeypatch.setattr("r3dmatch.cli.app", fake_cli_app)

    web_launcher.main()

    assert captured["argv"] == ["R3DMatch", "runtime-health", "--strict"]


def test_web_launcher_defaults_to_desktop_ui(monkeypatch: pytest.MonkeyPatch) -> None:
    from r3dmatch import web_launcher

    called: dict[str, object] = {}

    def fake_launch(repo_root: str, *, minimal_mode: bool = False, smoke_exit_ms: int = 0) -> None:
        called["repo_root"] = repo_root
        called["minimal_mode"] = minimal_mode
        called["smoke_exit_ms"] = smoke_exit_ms

    monkeypatch.setattr("r3dmatch.web_launcher.launch_desktop_ui", fake_launch)
    monkeypatch.setattr("r3dmatch.web_launcher.sys.argv", ["R3DMatch"])

    web_launcher.main()

    assert str(called["repo_root"]).endswith("/R3DMatch")
    assert called["minimal_mode"] is False
    assert called["smoke_exit_ms"] == 0


def test_web_launcher_desktop_smoke_forwards_auto_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    from r3dmatch import web_launcher

    called: dict[str, object] = {}

    def fake_launch(repo_root: str, *, minimal_mode: bool = False, smoke_exit_ms: int = 0) -> None:
        called["repo_root"] = repo_root
        called["smoke_exit_ms"] = smoke_exit_ms

    monkeypatch.setattr("r3dmatch.web_launcher.launch_desktop_ui", fake_launch)
    monkeypatch.setattr("r3dmatch.web_launcher.sys.argv", ["R3DMatch", "--desktop-smoke", "--desktop-smoke-ms", "900"])

    web_launcher.main()

    assert called["smoke_exit_ms"] == 900


def test_desktop_window_matching_domain_maps_display_label_to_perceptual(tmp_path: Path) -> None:
    app = _qt_test_app()
    window = R3DMatchDesktopWindow(repo_root=str(tmp_path))
    try:
        window.input_path.setText(str(tmp_path))
        window.output_path.setText(str(tmp_path / "out"))
        assert window.matching_domain.currentText() == "Display (IPP2 / BT.709 / BT.1886)"
        assert window.matching_domain.currentData() == "perceptual"
        command = window._review_command_from_form()
        index = command.index("--matching-domain")
        assert command[index + 1] == "perceptual"
    finally:
        window.close()
        if QApplication.instance() is app:
            app.processEvents()


def test_desktop_window_results_surface_explains_missing_outputs(tmp_path: Path) -> None:
    app = _qt_test_app()
    window = R3DMatchDesktopWindow(repo_root=str(tmp_path))
    try:
        window.output_path.setText(str(tmp_path / "missing-run"))
        window._refresh_artifacts()
        assert "No output folder is selected yet" in (window.results_banner.text() if window.results_banner else "")
        assert "No artifacts yet" not in (window.results_summary.toPlainText() if window.results_summary else "")
    finally:
        window.close()
        if QApplication.instance() is app:
            app.processEvents()


def test_desktop_window_results_surface_discovers_latest_completed_run_in_selected_output_root(tmp_path: Path) -> None:
    app = _qt_test_app()
    output_root = tmp_path / "runs"
    older_report = output_root / "subset_066" / "report"
    newer_report = output_root / "subset_067" / "report"
    older_report.mkdir(parents=True)
    newer_report.mkdir(parents=True)
    for report_dir, stamp in [(older_report, 1000), (newer_report, 2000)]:
        (report_dir / "contact_sheet.html").write_text("<html></html>", encoding="utf-8")
        (report_dir / "preview_contact_sheet.pdf").write_text("pdf", encoding="utf-8")
        (report_dir / "review_validation.json").write_text(
            json.dumps({"status": "success", "review_complete": True}),
            encoding="utf-8",
        )
        os.utime(report_dir / "review_validation.json", (stamp, stamp))
        os.utime(report_dir / "contact_sheet.html", (stamp, stamp))
        os.utime(report_dir / "preview_contact_sheet.pdf", (stamp, stamp))
    window = R3DMatchDesktopWindow(repo_root=str(tmp_path))
    try:
        window.output_path.setText(str(output_root))
        resolved_root = window._maybe_output_root()
        assert resolved_root == output_root / "subset_067"
    finally:
        window.close()
        if QApplication.instance() is app:
            app.processEvents()


def test_desktop_window_load_operator_surface_context_prefers_contact_sheet_summary_when_review_package_is_sparse(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    app = _qt_test_app()
    out_dir = tmp_path / "run"
    report_dir = out_dir / "report"
    report_dir.mkdir(parents=True)
    (report_dir / "contact_sheet.html").write_text("<html></html>", encoding="utf-8")
    (report_dir / "preview_contact_sheet.pdf").write_text("pdf", encoding="utf-8")
    (report_dir / "review_validation.json").write_text(json.dumps({"status": "success"}), encoding="utf-8")
    (report_dir / "contact_sheet.json").write_text(
        json.dumps(
            {
                "recommended_strategy": {"strategy_label": "Optimal Exposure", "reference_clip_id": "H007_C067_0403RE_001"},
                "hero_recommendation": {"candidate_clip_id": "H007_C067_0403RE_001"},
                "white_balance_model": {"model_label": "Shared Kelvin / Per-Camera Tint"},
                "run_assessment": {"status": "READY", "operator_note": "Ready to export."},
                "gray_target_consistency": {"dominant_target_class": "sphere", "mixed_target_classes": False, "summary": "Retained cameras consistently measured the gray sphere."},
                "operator_recommendation": "Ready to export.",
            }
        ),
        encoding="utf-8",
    )
    (report_dir / "review_package.json").write_text(
        json.dumps({"review_mode": "full_contact_sheet", "run_assessment": None, "gray_target_consistency": None}),
        encoding="utf-8",
    )
    monkeypatch.setattr("r3dmatch.web_app._build_operator_surfaces", lambda **kwargs: {})
    monkeypatch.setattr("r3dmatch.web_app._render_apply_surface", lambda *_args, **_kwargs: "")
    monkeypatch.setattr("r3dmatch.web_app._render_commit_table", lambda *_args, **_kwargs: "")
    monkeypatch.setattr("r3dmatch.web_app._render_decision_surface", lambda *_args, **_kwargs: "")
    monkeypatch.setattr("r3dmatch.web_app._render_push_surface", lambda *_args, **_kwargs: "")
    monkeypatch.setattr("r3dmatch.web_app._render_verification_surface", lambda *_args, **_kwargs: "")
    window = R3DMatchDesktopWindow(repo_root=str(tmp_path))
    try:
        context = window._load_operator_surface_context(out_dir)
        review_payload = dict(context["review_payload"])
        assert review_payload["run_assessment"]["status"] == "READY"
        assert review_payload["gray_target_consistency"]["dominant_target_class"] == "sphere"
        assert review_payload["recommended_strategy"]["strategy_label"] == "Optimal Exposure"
    finally:
        window.close()
        app.processEvents()


def test_desktop_window_tab_order_matches_operator_workflow(tmp_path: Path) -> None:
    app = _qt_test_app()
    window = R3DMatchDesktopWindow(repo_root=str(tmp_path))
    try:
        assert [window.tabs.tabText(index) for index in range(window.tabs.count())] == [
            "Review",
            "Results",
            "Apply / Verify",
            "Settings",
        ]
        assert window.log_output.parent() is not None
        assert window.tabs.currentWidget() is window.review_tab
    finally:
        window.close()
        app.processEvents()


def test_desktop_window_scan_populates_clip_groups_and_review_command_uses_selection(tmp_path: Path) -> None:
    app = _qt_test_app()
    for name in [
        "G007_A063_0325EV_001.R3D",
        "G007_B063_0325EV_001.R3D",
        "G007_A064_0325EV_001.R3D",
    ]:
        (tmp_path / name).write_bytes(b"")
    window = R3DMatchDesktopWindow(repo_root=str(tmp_path))
    try:
        window.input_path.setText(str(tmp_path))
        window.output_path.setText(str(tmp_path / "out"))
        window._scan_source()
        assert window.clip_group_list is not None
        assert window.clip_group_list.count() == 2
        second = window.clip_group_list.item(1)
        second.setCheckState(Qt.CheckState.Unchecked)
        command = window._review_command_from_form()
        group_indices = [index for index, token in enumerate(command) if token == "--clip-group"]
        selected_groups = [command[index + 1] for index in group_indices]
        assert selected_groups == ["063"]
        assert "Only these groups will be included" in (window.clip_group_summary.text() if window.clip_group_summary else "")
    finally:
        window.close()
        app.processEvents()


def test_desktop_window_results_and_apply_surfaces_use_operator_summary(tmp_path: Path) -> None:
    app = _qt_test_app()
    out_dir = tmp_path / "run"
    report_dir = out_dir / "report"
    report_dir.mkdir(parents=True)
    (report_dir / "contact_sheet.html").write_text("<html></html>", encoding="utf-8")
    (report_dir / "preview_contact_sheet.pdf").write_text("pdf", encoding="utf-8")
    (report_dir / "review_validation.json").write_text(json.dumps({"status": "success"}), encoding="utf-8")
    preview_dir = out_dir / "previews"
    preview_dir.mkdir(parents=True)
    original = preview_dir / "original.jpg"
    corrected = preview_dir / "corrected.jpg"
    Image.fromarray(np.full((8, 8, 3), 120, dtype=np.uint8), mode="RGB").save(original, format="JPEG")
    Image.fromarray(np.full((8, 8, 3), 180, dtype=np.uint8), mode="RGB").save(corrected, format="JPEG")
    window = R3DMatchDesktopWindow(repo_root=str(tmp_path))
    try:
        window.output_path.setText(str(out_dir))
        window._load_operator_surface_context = lambda _output_root: {  # type: ignore[method-assign]
            "review_validation": {"status": "success"},
            "scientific_validation": {"status": "blocked_asset_mismatch", "replay_integrity_state": "blocked_asset_mismatch"},
            "report_root": report_dir,
            "contact_sheet_payload": {
                "recommended_strategy": {
                    "strategy_label": "Optimal Exposure (Best Match to Gray)",
                    "strategy_summary": "Matched to the best gray target.",
                    "reference_clip_id": "G007_A063_001",
                },
                "run_assessment": {
                    "status": "READY_WITH_WARNINGS",
                    "operator_note": "Review one flagged camera before commit.",
                },
                "gray_target_consistency": {
                    "dominant_target_class": "sphere",
                    "mixed_target_classes": False,
                    "summary": "Retained cameras consistently measured the gray sphere.",
                },
                "white_balance_model": {
                    "model_label": "Shared Kelvin / Per-Camera Tint",
                    "candidate_count": 3,
                    "selection_reason": "Chosen from stored neutral RGB residuals.",
                    "shared_kelvin": 5520,
                    "shared_tint_mode": "per_camera",
                    "diagnostics": {
                        "mean_neutral_residual_after_solve": 0.0123,
                        "source_sample_count": 36,
                    },
                },
                "hero_recommendation": {"candidate_clip_id": "G007_A063_001"},
                "selected_clip_groups": ["063"],
                "report_focus_label": "Auto",
            },
            "corrected_validation": {
                "recommended_strategy_key": "optimal_exposure",
                "best_residual_stops": 0.01,
                "worst_residual_stops": 0.42,
                "outside_tolerance_cameras": ["H007_D063_001"],
                "rows": [
                    {
                        "clip_id": "G007_A063_001",
                        "camera_label": "G007_A063",
                        "camera_id": "G007_A063",
                        "strategy_key": "optimal_exposure",
                        "applied_correction_stops": 0.12,
                        "trust_class": "TRUSTED",
                        "reference_use": "Included",
                        "trust_reason": "Stable gray sample",
                        "original_measured_gray_log2": -2.11,
                        "original_image_path": str(original),
                        "corrected_image_path": str(corrected),
                    },
                    {
                        "clip_id": "H007_D063_001",
                        "camera_label": "H007_D063",
                        "camera_id": "H007_D063",
                        "strategy_key": "optimal_exposure",
                        "applied_correction_stops": 1.35,
                        "trust_class": "UNTRUSTED",
                        "reference_use": "Excluded",
                        "trust_reason": "Offset exceeds safe range",
                        "original_measured_gray_log2": -3.20,
                        "original_image_path": str(original),
                        "corrected_image_path": str(corrected),
                    },
                ],
            },
            "decision_surface_html": "<div class='summary-panel'><h3>Calibration Recommendation</h3></div>",
            "commit_table_html": "<div>Commit Table</div>",
            "camera_state_report": {},
            "verification_report": {},
            "post_apply_report": {},
            "apply_report": {},
            "push_surface_html": "<div>Camera Push Plan</div>",
            "apply_surface_html": "",
            "verification_surface_html": "",
        }
        window._refresh_artifacts()
        window._refresh_apply_surface()
        assert "Calibration Recommendation" in (window.results_summary.toPlainText() if window.results_summary else "")
        assert "Retained Cluster" in (window.results_summary.toPlainText() if window.results_summary else "")
        assert "Excluded Cameras" in (window.results_summary.toPlainText() if window.results_summary else "")
        assert "Gray Sample Basis" in (window.results_summary.toPlainText() if window.results_summary else "")
        assert "Visual Comparison" in (window.results_summary.toPlainText() if window.results_summary else "")
        assert "Retained cameras consistently measured the gray sphere" in (window.results_summary.toPlainText() if window.results_summary else "")
        assert "Camera Push Plan" in (window.apply_summary.toPlainText() if window.apply_summary else "")
    finally:
        window.close()
        app.processEvents()


def test_desktop_redline_path_persists_across_relaunch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    app = _qt_test_app()
    desktop_config = tmp_path / ".r3dmatch_config.json"
    project_config = tmp_path / "config" / "redline.json"
    redline_executable = tmp_path / "REDline"
    redline_executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    redline_executable.chmod(0o755)
    monkeypatch.setenv(DESKTOP_CONFIG_ENV, str(desktop_config))
    monkeypatch.setattr("r3dmatch.report.REDLINE_CONFIG_PATH", project_config)
    monkeypatch.setattr("r3dmatch.runtime_env.REDLINE_CONFIG_PATH", project_config)
    monkeypatch.delenv("R3DMATCH_REDLINE_EXECUTABLE", raising=False)
    monkeypatch.delenv("REDLINE_PATH", raising=False)

    first_window = R3DMatchDesktopWindow(repo_root=str(tmp_path))
    try:
        first_window.redline_path.setText(str(redline_executable))
        first_window._save_redline()
        assert "Ready" in first_window.redline_feedback.text()
        assert redline_configured_path() == str(redline_executable)
        assert desktop_config.exists()
        persisted = json.loads(desktop_config.read_text(encoding="utf-8"))
        assert persisted["redline_path"] == str(redline_executable)
    finally:
        first_window.close()
        app.processEvents()

    second_window = R3DMatchDesktopWindow(repo_root=str(tmp_path))
    try:
        assert second_window.redline_path.text() == str(redline_executable)
        assert "Ready" in second_window.redline_feedback.text()
    finally:
        second_window.close()
        app.processEvents()


def test_web_app_routes_and_validation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    started: dict[str, object] = {}

    def fake_start(state, command, output_path, **kwargs):  # type: ignore[no-untyped-def]
        started["command"] = command
        started["output_path"] = output_path
        started["kwargs"] = kwargs
        state.task.command = " ".join(command)
        state.task.output_path = output_path
        state.task.status = "running"
        state.task.logs = ["started\n"]
        state.task.clip_count = int(kwargs.get("clip_count", 0) or 0)
        state.task.strategies_text = str(kwargs.get("strategies_text", ""))
        state.task.review_mode = str(kwargs.get("review_mode", ""))
        state.task.preview_mode = str(kwargs.get("preview_mode", ""))

    monkeypatch.setattr("r3dmatch.web_app._start_command_task", fake_start)
    monkeypatch.setattr("r3dmatch.web_app._ensure_scan_preview", lambda input_path, scan, form=None: None)
    app = create_app()
    client = app.test_client()

    index = client.get("/")
    assert index.status_code == 200
    assert b"R3DMatch" in index.data

    invalid = client.post("/run-review", data={"input_path": "", "output_path": ""})
    assert invalid.status_code == 200
    assert b"Calibration folder path is required" in invalid.data

    rdc = tmp_path / "A001_C001_0001AB.RDC"
    rdc.mkdir()
    clip = rdc / "A001_C001_0001AB_001.R3D"
    clip.write_bytes(b"")
    out_dir = tmp_path / "out"
    scan_response = client.post("/scan", data={"input_path": str(tmp_path), "output_path": str(out_dir)})
    assert scan_response.status_code == 200
    assert b"found 1 RED clips" in scan_response.data
    assert b"A001_C001_0001AB_001" in scan_response.data

    run_response = client.post(
        "/run-review",
        data={
            "input_path": str(tmp_path),
            "output_path": str(out_dir),
            "backend": "mock",
            "target_type": "gray_sphere",
            "processing_mode": "both",
            "target_strategies": ["median"],
            "review_mode": "lightweight_analysis",
            "preview_mode": "monitoring",
            "roi_mode": "center",
        },
    )
    assert run_response.status_code == 200
    assert started["output_path"] == str(out_dir)
    assert "review-calibration" in " ".join(started["command"])
    assert started["kwargs"]["clip_count"] == 1
    assert started["kwargs"]["strategies_text"] == "median"
    assert started["kwargs"]["review_mode"] == "lightweight_analysis"
    assert started["kwargs"]["preview_mode"] == "monitoring"


def test_resolved_output_path_for_form_matches_workflow_group_run_label(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    scan = {
        "clip_ids": [f"G007_{camera}065_0325XX_001" for camera in ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"]],
    }
    form = {
        "output_path": str(out_dir),
        "run_label": "",
        "selected_clip_ids": [],
        "selected_clip_groups": ["065"],
    }
    assert _resolved_output_path_for_form(form, scan) == str(out_dir / "subset_065")


def test_web_run_review_uses_canonical_group_output_path_not_subset_count(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    started: dict[str, object] = {}

    def fake_start(state, command, output_path, **kwargs):  # type: ignore[no-untyped-def]
        started["command"] = command
        started["output_path"] = output_path
        state.task.command = " ".join(command)
        state.task.output_path = output_path
        state.task.canonical_output_path = output_path
        state.task.status = "running"

    monkeypatch.setattr("r3dmatch.web_app._start_command_task", fake_start)
    monkeypatch.setattr("r3dmatch.web_app._ensure_scan_preview", lambda input_path, scan, form=None: None)
    app = create_app()
    client = app.test_client()
    input_root = tmp_path / "input"
    out_dir = tmp_path / "out"
    for camera in ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"]:
        rdc = input_root / f"G007_{camera}065_0325XX.RDC"
        rdc.mkdir(parents=True, exist_ok=True)
        (rdc / f"G007_{camera}065_0325XX_001.R3D").write_bytes(b"")

    scan_response = client.post("/scan", data={"input_path": str(input_root), "output_path": str(out_dir)})
    assert scan_response.status_code == 200

    run_response = client.post(
        "/run-review",
        data={
            "input_path": str(input_root),
            "output_path": str(out_dir),
            "backend": "mock",
            "target_type": "gray_sphere",
            "processing_mode": "both",
            "matching_domain": "scene",
            "review_mode": "lightweight_analysis",
            "preview_mode": "monitoring",
            "roi_x": "0.43",
            "roi_y": "0.42",
            "roi_w": "0.14",
            "roi_h": "0.16",
            "selected_clip_groups": ["065"],
            "target_strategies": ["median"],
        },
    )
    assert run_response.status_code == 200
    assert started["output_path"] == str(out_dir / "subset_065")
    assert "subset_12" not in str(started["output_path"])


def test_preferred_roi_preview_record_prefers_group_063_when_available() -> None:
    scan = {
        "clip_records": [
            {
                "clip_id": "G007_A065_0325F6_001",
                "subset_group": "065",
                "camera_label": "Cam A",
                "source_path": "/tmp/G007_A065_0325F6_001.R3D",
            },
            {
                "clip_id": "G007_A063_032563_001",
                "subset_group": "063",
                "camera_label": "Cam A",
                "source_path": "/tmp/G007_A063_032563_001.R3D",
            },
        ],
        "clip_ids": ["G007_A065_0325F6_001", "G007_A063_032563_001"],
        "clip_groups": [
            {"group_id": "063", "clip_ids": ["G007_A063_032563_001"], "clip_count": 1},
            {"group_id": "065", "clip_ids": ["G007_A065_0325F6_001"], "clip_count": 1},
        ],
    }
    form = {
        "selected_clip_groups": ["065"],
        "selected_clip_ids": [],
        "advanced_clip_selection": False,
    }
    record = _preferred_roi_preview_record(form, scan)
    assert record is not None
    assert record["clip_id"] == "G007_A063_032563_001"


def test_scan_preview_path_changes_with_selected_roi_clip() -> None:
    first = _scan_preview_path("/tmp/calibration", clip_id="G007_A063_032563_001", preview_mode="monitoring")
    second = _scan_preview_path("/tmp/calibration", clip_id="G007_A065_0325F6_001", preview_mode="monitoring")
    assert first != second


def test_ensure_scan_preview_propagates_kelvin_and_tint_defaults(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clip_path = tmp_path / "G007_A063_032563_001.R3D"
    clip_path.write_bytes(b"")
    captured: dict[str, object] = {}

    def fake_build(*args, **kwargs):  # type: ignore[no-untyped-def]
        captured["kelvin"] = kwargs.get("kelvin")
        captured["tint"] = kwargs.get("tint")
        output_index = args[0] if False else None
        return ["REDLine", "--o", str(kwargs["output_path"])]

    def fake_run(command, capture_output, text, check):  # type: ignore[no-untyped-def]
        output_path = Path(command[command.index("--o") + 1])
        output_path.write_bytes(b"preview")
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("r3dmatch.web_app._resolve_redline_executable", lambda: "/usr/local/bin/REDLine")
    monkeypatch.setattr(
        "r3dmatch.web_app._detect_redline_capabilities",
        lambda executable: {
            "supports_color_space": True,
            "supports_gamma_curve": True,
            "supports_output_tonemap": True,
            "supports_rolloff": True,
            "supports_shadow_control": True,
            "supports_lut": False,
            "supports_load_rmd": True,
        },
    )
    monkeypatch.setattr("r3dmatch.web_app._build_redline_preview_command", fake_build)
    monkeypatch.setattr("r3dmatch.web_app.subprocess.run", fake_run)
    scan = {
        "first_clip_path": str(clip_path),
        "kelvin": 6100,
        "tint": -3,
    }
    preview_path = _ensure_scan_preview(str(tmp_path), scan, form={"preview_mode": "monitoring"})
    assert preview_path is not None
    assert captured["kelvin"] == 6100
    assert captured["tint"] == -3.0


def test_build_review_web_command_supports_ftps_source_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("r3dmatch.web_app.sys.executable", "/tmp/venv/bin/python")
    monkeypatch.setenv("RED_SDK_ROOT", "/external/red/sdk")
    command = build_review_web_command(
        "/Users/sfouasnon/Desktop/R3DMatch",
        {
            "source_mode": "ftps_camera_pull",
            "input_path": "",
            "output_path": "/tmp/out",
            "run_label": "ftps_pull",
            "backend": "mock",
            "target_type": "gray_sphere",
            "processing_mode": "both",
            "matching_domain": "scene",
            "roi_x": "0.1",
            "roi_y": "0.2",
            "roi_w": "0.3",
            "roi_h": "0.4",
            "selected_clip_ids": [],
            "selected_clip_groups": [],
            "target_strategies": ["median"],
            "reference_clip_id": "",
            "hero_clip_id": "",
            "review_mode": "lightweight_analysis",
            "report_focus": "cluster_extremes",
            "preview_mode": "monitoring",
            "preview_lut": "",
            "local_ingest_root": "/Volumes/RAID/ingest/063",
            "ftps_reel": "007",
            "ftps_clips": "63,64-65",
            "ftps_cameras": ["AA", "AB"],
        },
    )
    joined = " ".join(command)
    assert "/tmp/venv/bin/python -m r3dmatch.cli review-calibration" in joined
    assert "--matching-domain perceptual" in joined
    assert "--source-mode ftps_camera_pull" in joined
    assert "--report-focus cluster_extremes" in joined
    assert "--ftps-reel 007" in joined
    assert "--ftps-clips 63,64-65" in joined
    assert "--ftps-local-root /Volumes/RAID/ingest/063" in joined
    assert joined.count("--ftps-camera") == 2
    assert "/Volumes/RAID/ingest/063" in joined
    assert "setenv RED_SDK_ROOT /external/red/sdk" in joined


def test_build_ftps_ingest_web_command_supports_discover_and_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("r3dmatch.web_app.sys.executable", "/tmp/venv/bin/python")
    form = {
        "output_path": "/tmp/review",
        "local_ingest_root": "/Volumes/RAID/ingest/063",
        "ftps_reel": "007",
        "ftps_clips": "63",
        "ftps_cameras": "AA,AB",
    }
    discover = " ".join(build_ftps_ingest_web_command("/Users/sfouasnon/Desktop/R3DMatch", form, action="discover"))
    retry = " ".join(build_ftps_ingest_web_command("/Users/sfouasnon/Desktop/R3DMatch", form, action="retry-failed"))
    assert "ingest-ftps --action discover --out /Volumes/RAID/ingest/063" in discover
    assert "--ftps-reel 007" in discover
    assert "--ftps-clips 63" in discover
    assert discover.count("--ftps-camera") == 2
    assert "ingest-ftps --action retry-failed --out /Volumes/RAID/ingest/063" in retry
    assert f"--manifest-path {ingest_manifest_path_for('/Volumes/RAID/ingest/063')}" in retry


def test_validate_ftps_ingest_form_accepts_local_ingest_root_retry(tmp_path: Path) -> None:
    ingest_root = tmp_path / "ingest"
    ingest_root.mkdir()
    ingest_manifest_path_for(ingest_root).write_text("{}", encoding="utf-8")
    error = _validate_ftps_ingest_form(
        {
            "source_mode": "ftps_camera_pull",
            "output_path": "",
            "local_ingest_root": str(ingest_root),
            "ftps_reel": "",
            "ftps_clips": "",
            "ftps_cameras": "",
        },
        action="retry-failed",
    )
    assert error is None


def test_validate_form_requires_red_sdk_runtime_for_red_backend(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    clip_path = tmp_path / "G007_A063_0325EV_001.R3D"
    clip_path.write_bytes(b"")
    monkeypatch.setattr(
        "r3dmatch.web_app.runtime_health_payload",
        lambda html_path=None: {
            "html_pdf_ready": True,
            "red_sdk_runtime": {"ready": False, "error": "RED SDK runtime is unavailable."},
            "redline_tool": {"ready": True, "error": ""},
        },
    )

    error = _validate_form(
        {
            "input_path": str(tmp_path),
            "output_path": str(tmp_path / "out"),
            "backend": "red",
            "source_mode": "local_folder",
            "target_type": "gray_sphere",
            "processing_mode": "both",
            "matching_domain": "perceptual",
            "review_mode": "lightweight_analysis",
            "preview_mode": "monitoring",
            "target_strategies": ["median"],
            "selected_clip_groups": ["063"],
            "selected_clip_ids": [],
            "reference_clip_id": "",
            "hero_clip_id": "",
            "preview_lut": "",
            "roi_mode": "sphere_auto",
        }
    )

    assert error is not None
    assert "RED SDK runtime is unavailable." in error


def test_validate_form_requires_redline_for_red_backend(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    clip_path = tmp_path / "G007_A063_0325EV_001.R3D"
    clip_path.write_bytes(b"")
    monkeypatch.setattr(
        "r3dmatch.web_app.runtime_health_payload",
        lambda html_path=None: {
            "html_pdf_ready": True,
            "red_sdk_runtime": {"ready": True, "error": ""},
            "redline_tool": {"ready": False, "error": "REDLine executable is not available."},
        },
    )

    error = _validate_form(
        {
            "input_path": str(tmp_path),
            "output_path": str(tmp_path / "out"),
            "backend": "red",
            "source_mode": "local_folder",
            "target_type": "gray_sphere",
            "processing_mode": "both",
            "matching_domain": "perceptual",
            "review_mode": "lightweight_analysis",
            "preview_mode": "monitoring",
            "target_strategies": ["median"],
            "selected_clip_groups": ["063"],
            "selected_clip_ids": [],
            "reference_clip_id": "",
            "hero_clip_id": "",
            "preview_lut": "",
            "roi_mode": "sphere_auto",
        }
    )

    assert error == "REDLine executable is not available."


def test_validate_form_reports_weasyprint_preflight_failure_for_full_contact_sheet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clip_path = tmp_path / "G007_A063_0325EV_001.R3D"
    clip_path.write_bytes(b"")
    monkeypatch.setattr(
        "r3dmatch.web_app.runtime_health_payload",
        lambda html_path=None: {
            "interpreter": "/tmp/venv/bin/python",
            "virtual_env": "/tmp/venv",
            "dyld_fallback_library_path": "/opt/homebrew/lib",
            "html_pdf_ready": False,
            "weasyprint_error": "cannot load library 'libgobject-2.0-0'",
        },
    )

    error = _validate_form(
        {
            "input_path": str(tmp_path),
            "output_path": str(tmp_path / "out"),
            "backend": "mock",
            "source_mode": "local_folder",
            "target_type": "gray_sphere",
            "processing_mode": "both",
            "matching_domain": "perceptual",
            "review_mode": "full_contact_sheet",
            "roi_mode": "sphere_auto",
            "target_strategies": ["median"],
            "reference_clip_id": "",
            "hero_clip_id": "",
            "preview_lut": "",
        }
    )

    assert error is not None
    assert "Full Contact Sheet PDF export is unavailable" in error
    assert "/tmp/venv/bin/python" in error
    assert "/opt/homebrew/lib" in error
    assert "libgobject-2.0-0" in error


def test_ensure_runtime_environment_auto_populates_homebrew_dyld(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DYLD_FALLBACK_LIBRARY_PATH", raising=False)
    monkeypatch.setattr("r3dmatch.runtime_env.sys.platform", "darwin")
    monkeypatch.setattr(
        "r3dmatch.runtime_env.Path.exists",
        lambda self: str(self) == "/opt/homebrew/lib",
    )

    payload = ensure_runtime_environment()

    assert payload["source"] == "auto_homebrew"
    assert os.environ["DYLD_FALLBACK_LIBRARY_PATH"] == "/opt/homebrew/lib"


def test_runtime_health_command_strict_fails_when_html_pdf_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "r3dmatch.cli.runtime_health_payload",
        lambda html_path=None: {
            "html_pdf_ready": False,
            "red_backend_ready": True,
            "interpreter": "/tmp/venv/bin/python",
            "virtual_env": "/tmp/venv",
        },
    )

    result = runner.invoke(app, ["runtime-health", "--strict"])

    assert result.exit_code == 1
    assert "\"html_pdf_ready\": false" in result.stdout.lower()


def test_pick_folder_command_returns_selected_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "r3dmatch.native_dialogs.pick_existing_directory",
        lambda title, directory="": "/Volumes/RAID/calibration",
    )
    result = runner.invoke(app, ["pick-folder", "--title", "Select Calibration Folder", "--directory", "/tmp"])
    assert result.exit_code == 0
    assert "/Volumes/RAID/calibration" in result.stdout


def test_pick_file_command_returns_selected_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "r3dmatch.native_dialogs.pick_existing_file",
        lambda title, directory="", filter="": "/Applications/REDCINE-X Professional/REDCINE-X PRO.app/Contents/MacOS/REDline",
    )
    result = runner.invoke(app, ["pick-file", "--title", "Select REDLine", "--directory", "/Applications", "--filter", "Executables (*)"])
    assert result.exit_code == 0
    assert "REDline" in result.stdout


def test_runtime_health_payload_reports_red_sdk_and_html_pdf_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "r3dmatch.runtime_env.ensure_runtime_environment",
        lambda: {"value": "/opt/homebrew/lib", "source": "auto_homebrew"},
    )
    monkeypatch.setattr(
        "r3dmatch.runtime_env.contact_sheet_pdf_export_preflight",
        lambda html_path=None: {
            "weasyprint_importable": True,
            "weasyprint_error": "",
            "asset_validation": {"all_assets_exist": True},
        },
    )
    monkeypatch.setattr(
        "r3dmatch.runtime_env.resolve_red_sdk_configuration",
        lambda: types.SimpleNamespace(
            redistributable_dir=Path("/external/red/sdk/Redistributable/mac"),
            redistributable_source="environment_override",
            root=Path("/external/red/sdk"),
            errors=(),
        ),
    )
    monkeypatch.setattr("r3dmatch.runtime_env.load_configured_red_native_module", lambda: object())
    monkeypatch.setattr(
        "r3dmatch.runtime_env.resolve_redline_tool_status",
        lambda: {
            "ready": False,
            "configured": "REDLine",
            "resolved_path": "",
            "source": "path",
            "config_path": "",
            "error": "REDLine executable is not available.",
        },
    )
    monkeypatch.setattr("r3dmatch.runtime_env.redline_configured_path", lambda: "/configured/REDline")
    monkeypatch.setenv("RED_SDK_ROOT", "/external/red/sdk")

    payload = runtime_health_payload()

    assert payload["html_pdf_ready"] is True
    assert payload["red_backend_ready"] is True
    assert payload["resolved_red_sdk_redistributable_dir"].endswith("Redistributable/mac")
    assert payload["red_sdk_runtime"]["source"] == "environment_override"
    assert payload["redline_tool"]["ready"] is False
    assert payload["redline_tool"]["configured_path"] == "/configured/REDline"


def test_runtime_health_payload_reports_bundled_red_runtime(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    bundled = tmp_path / "Resources" / "red_runtime" / "redistributable"
    bundled.mkdir(parents=True)
    for filename in ("REDDecoder.dylib", "REDMetal.dylib", "REDOpenCL.dylib", "REDR3D.dylib"):
        (bundled / filename).write_bytes(b"runtime")
    monkeypatch.setattr(
        "r3dmatch.runtime_env.ensure_runtime_environment",
        lambda: {"value": "/opt/homebrew/lib", "source": "auto_homebrew"},
    )
    monkeypatch.setattr(
        "r3dmatch.runtime_env.contact_sheet_pdf_export_preflight",
        lambda html_path=None: {
            "weasyprint_importable": True,
            "weasyprint_error": "",
            "asset_validation": {"all_assets_exist": True},
        },
    )
    monkeypatch.setattr("r3dmatch.sdk.bundled_red_redistributable_dir", lambda: bundled)
    monkeypatch.setattr("r3dmatch.runtime_env.load_configured_red_native_module", lambda: object())
    monkeypatch.setattr(
        "r3dmatch.runtime_env.resolve_redline_tool_status",
        lambda: {
            "ready": True,
            "configured": "REDLine",
            "resolved_path": "/usr/local/bin/REDLine",
            "source": "path",
            "config_path": "",
            "error": "",
        },
    )
    monkeypatch.setattr("r3dmatch.runtime_env.redline_configured_path", lambda: "")
    monkeypatch.delenv("RED_SDK_ROOT", raising=False)
    monkeypatch.delenv("RED_SDK_REDISTRIBUTABLE_DIR", raising=False)

    payload = runtime_health_payload()

    assert payload["red_backend_ready"] is True
    assert payload["red_sdk_runtime"]["source"] == "bundled_app"
    assert payload["red_sdk_runtime"]["redistributable_dir"] == str(bundled)


def test_runtime_health_payload_uses_persisted_desktop_redline_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    desktop_config = tmp_path / ".r3dmatch_config.json"
    redline_executable = tmp_path / "REDline"
    redline_executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    redline_executable.chmod(0o755)
    desktop_config.write_text(json.dumps({"redline_path": str(redline_executable)}, indent=2), encoding="utf-8")
    monkeypatch.setenv(DESKTOP_CONFIG_ENV, str(desktop_config))
    monkeypatch.delenv("R3DMATCH_REDLINE_EXECUTABLE", raising=False)
    monkeypatch.delenv("REDLINE_PATH", raising=False)
    monkeypatch.setattr(
        "r3dmatch.runtime_env.contact_sheet_pdf_export_preflight",
        lambda html_path=None: {
            "weasyprint_importable": True,
            "weasyprint_error": "",
            "asset_validation": {"all_assets_exist": True},
        },
    )
    monkeypatch.setattr(
        "r3dmatch.runtime_env.resolve_red_sdk_configuration",
        lambda: types.SimpleNamespace(
            redistributable_dir=Path("/external/red/sdk/Redistributable/mac"),
            redistributable_source="environment_override",
            root=Path("/external/red/sdk"),
            errors=(),
        ),
    )
    monkeypatch.setattr("r3dmatch.runtime_env.load_configured_red_native_module", lambda: object())

    payload = runtime_health_payload()

    assert os.environ["REDLINE_PATH"] == str(redline_executable)
    assert os.environ["R3DMATCH_REDLINE_EXECUTABLE"] == str(redline_executable)
    assert payload["redline_tool"]["configured_path"] == str(redline_executable)
    assert payload["redline_tool"]["desktop_config_path"] == str(desktop_config)
    assert payload["redline_tool"]["ready"] is True


def test_calibration_overview_from_full_contact_payload_uses_clips_fallback() -> None:
    overview = _calibration_overview_from_payload(
        {
            "clips": [
                {
                    "clip_id": "G007_A063_0325AA_001",
                    "camera_label": "A",
                    "reference_use": "Included",
                    "trust_score": 0.95,
                    "confidence": 0.93,
                    "metrics": {
                        "exposure": {
                            "sample_2_ire": 22.0,
                            "camera_offset_from_anchor": 0.08,
                            "final_offset_stops": 0.08,
                            "gray_exposure_summary": "Sample 1 24 / Sample 2 22 / Sample 3 21 IRE",
                        }
                    },
                    "ipp2_validation": {
                        "camera_label": "A",
                        "sample_2_ire": 22.0,
                        "camera_offset_from_anchor": 0.08,
                        "ipp2_gray_exposure_summary": "Sample 1 24 / Sample 2 22 / Sample 3 21 IRE",
                    },
                },
                {
                    "clip_id": "G007_B063_0325AA_001",
                    "camera_label": "B",
                    "reference_use": "Excluded",
                    "trust_score": 0.4,
                    "confidence": 0.4,
                    "metrics": {"exposure": {"sample_2_ire": 28.0, "camera_offset_from_anchor": 0.9}},
                    "ipp2_validation": {"camera_label": "B", "sample_2_ire": 28.0, "camera_offset_from_anchor": 0.9},
                },
            ]
        },
        {"status": "READY"},
    )
    assert overview["retained_count"] == 1
    assert overview["excluded_count"] == 1
    assert overview["sample_2_ire_range_text"] == "22 IRE to 22 IRE"
    assert overview["best_reference_camera"] == "A"


def test_web_app_subset_panel_defaults_to_group_control_and_read_only_clips(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("r3dmatch.web_app._ensure_scan_preview", lambda input_path, scan, form=None: None)
    app = create_app()
    client = app.test_client()
    for name in [
        "G007_A063_0325EV_001.R3D",
        "G007_B063_0325EV_001.R3D",
        "G007_A064_0325EV_001.R3D",
    ]:
        (tmp_path / name).write_bytes(b"")

    response = client.post(
        "/scan",
        data={
            "input_path": str(tmp_path),
            "output_path": str(tmp_path / "out"),
            "selected_clip_groups": ["063"],
        },
    )

    assert response.status_code == 200
    assert b"Clip Groups" in response.data
    assert b"Selected Clips" in response.data
    assert b"Enable advanced clip selection" in response.data
    assert b"2 clips selected from group 063" in response.data
    assert b"Read-only preview of the current group-driven subset" in response.data
    assert b"class=\"clip-selector clip-checkbox\"" not in response.data


def test_web_app_advanced_clip_mode_disables_group_submission(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    started: dict[str, object] = {}

    def fake_start(state, command, output_path, **kwargs):  # type: ignore[no-untyped-def]
        started["command"] = command
        started["output_path"] = output_path
        started["kwargs"] = kwargs
        state.task.command = " ".join(command)
        state.task.output_path = output_path
        state.task.status = "running"
        state.task.logs = ["started\n"]

    monkeypatch.setattr("r3dmatch.web_app._start_command_task", fake_start)
    monkeypatch.setattr("r3dmatch.web_app._ensure_scan_preview", lambda input_path, scan, form=None: None)
    app = create_app()
    client = app.test_client()
    for name in [
        "G007_A063_0325EV_001.R3D",
        "G007_B063_0325EV_001.R3D",
        "G007_A064_0325EV_001.R3D",
    ]:
        (tmp_path / name).write_bytes(b"")

    response = client.post(
        "/run-review",
        data={
            "input_path": str(tmp_path),
            "output_path": str(tmp_path / "out"),
            "backend": "mock",
            "target_type": "gray_sphere",
            "processing_mode": "both",
            "target_strategies": ["median"],
            "preview_mode": "monitoring",
            "roi_mode": "center",
            "advanced_clip_selection": "1",
            "selected_clip_groups": ["063"],
            "selected_clip_ids": ["G007_A064_0325EV_001"],
        },
    )

    assert response.status_code == 200
    joined = " ".join(started["command"])
    assert "--clip-group" not in joined
    assert "--clip-id G007_A064_0325EV_001" in joined
    assert started["kwargs"]["clip_count"] == 1
    assert b"Manual selection: 1 clip" in response.data
    assert b"Manual clip selection active" in response.data
    assert b"class=\"clip-selector clip-checkbox\"" in response.data


def test_web_app_status_progress_and_completion_links(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("r3dmatch.web_app._ensure_scan_preview", lambda input_path, scan, form=None: None)
    app = create_app()
    client = app.test_client()
    out_dir = tmp_path / "out"
    report_dir = out_dir / "report"
    report_dir.mkdir(parents=True)
    pdf_path = report_dir / "preview_contact_sheet.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")
    html_path = report_dir / "contact_sheet.html"
    html_path.write_text("<html></html>", encoding="utf-8")
    (report_dir / "review_validation.json").write_text(
        json.dumps({"status": "success", "errors": [], "warnings": [], "preview_reference_count": 0, "preview_existing_count": 0}),
        encoding="utf-8",
    )
    state = app.config["UI_STATE"]
    state.task.command = "python3 -m r3dmatch.cli review-calibration /tmp/in --out /tmp/out"
    state.task.output_path = str(out_dir)
    state.task.status = "running"
    state.task.returncode = 0
    state.task.clip_count = 3
    state.task.strategies_text = "median, optimal-exposure"
    state.task.preview_mode = "monitoring"
    status_response = client.get("/status")
    assert status_response.status_code == 200
    payload = status_response.get_json()
    assert payload["stage"] == "Complete"
    assert payload["report_ready"] is True
    assert payload["preview_pdf_url"]
    assert payload["preview_html_url"]
    assert payload["output_folder"] == str(out_dir)
    assert payload["progress_percent"] == 100
    assert payload["can_cancel"] is False


def test_web_app_status_prefers_live_process_over_stale_completed_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("r3dmatch.web_app._ensure_scan_preview", lambda input_path, scan, form=None: None)
    app = create_app()
    client = app.test_client()
    out_dir = tmp_path / "out"
    (out_dir / "analysis").mkdir(parents=True)
    state = app.config["UI_STATE"]

    class FakeProcess:
        def poll(self):  # type: ignore[no-untyped-def]
            return None

    state.task.command = "python3 -m r3dmatch.cli review-calibration /tmp/in --out /tmp/out"
    state.task.output_path = str(out_dir)
    state.task.status = "completed"
    state.task.returncode = 0
    state.task.stage = "Scanning sources"
    state.task.stage_index = 0
    state.task.items_completed = 0
    state.task.items_total = 12
    state.task.clip_count = 12
    state.task.process = FakeProcess()  # type: ignore[assignment]
    response = client.get("/status")
    payload = response.get_json()
    assert payload["status"] == "running"
    assert payload["stage"] == "Scanning sources"
    assert payload["can_cancel"] is True
    assert payload["process_alive"] is True


def test_web_app_status_marks_finishing_and_stalled_with_live_process(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("r3dmatch.web_app._ensure_scan_preview", lambda input_path, scan, form=None: None)
    app = create_app()
    client = app.test_client()
    out_dir = tmp_path / "out"
    report_dir = out_dir / "report"
    previews_dir = out_dir / "previews"
    analysis_dir = out_dir / "analysis"
    report_dir.mkdir(parents=True)
    previews_dir.mkdir(parents=True)
    analysis_dir.mkdir(parents=True)
    for index in range(12):
        (analysis_dir / f"G007_A{index:03d}_0325EV_001.analysis.json").write_text("{}", encoding="utf-8")
    for index in range(48):
        (previews_dir / f"preview_{index:03d}.jpg").write_bytes(b"jpg")
    state = app.config["UI_STATE"]

    class FakeProcess:
        def poll(self):  # type: ignore[no-untyped-def]
            return None

    state.task.command = "python3 -m r3dmatch.cli review-calibration /tmp/in --out /tmp/out"
    state.task.output_path = str(out_dir)
    state.task.status = "running"
    state.task.clip_count = 12
    state.task.items_total = 48
    state.task.strategies_text = "median"
    state.task.process = FakeProcess()  # type: ignore[assignment]
    state.task.started_at = 100.0
    state.task.last_output_at = 100.0
    state.task.last_progress_at = 100.0

    monkeypatch.setattr("r3dmatch.web_app.time.time", lambda: 105.0)
    finishing_payload = client.get("/status").get_json()
    assert finishing_payload["status"] == "finishing"
    assert finishing_payload["stage"] == "Building report"
    assert finishing_payload["can_cancel"] is True

    monkeypatch.setattr("r3dmatch.web_app.time.time", lambda: 130.0)
    stalled_payload = client.get("/status").get_json()
    assert stalled_payload["status"] == "stalled"
    assert stalled_payload["stage"] == "Building report"
    assert stalled_payload["can_cancel"] is True


def test_validate_review_run_contract_success(tmp_path: Path) -> None:
    report_dir = tmp_path / "report"
    previews_dir = tmp_path / "previews"
    report_dir.mkdir(parents=True)
    previews_dir.mkdir(parents=True)
    baseline = previews_dir / "G007_B057_0324YT_001.original.review.jpg"
    corrected = previews_dir / "G007_B057_0324YT_001.exposure.review.median.jpg"
    _write_test_preview(baseline)
    _write_test_preview(corrected, color=(140, 140, 140))
    (tmp_path / "summary.json").write_text("{}", encoding="utf-8")
    (previews_dir / "preview_commands.json").write_text("{}", encoding="utf-8")
    payload = {
        "clip_count": 1,
        "shared_originals": [{"clip_id": "G007_B057_0324YT_001", "original_frame": str(baseline)}],
        "strategies": [
            {
                "strategy": "median",
                "clips": [
                    {
                        "clip_id": "G007_B057_0324YT_001",
                        "original_frame": str(baseline),
                        "exposure_corrected": str(corrected),
                        "color_corrected": str(baseline),
                        "both_corrected": str(corrected),
                        "preview_variants": {
                            "original": str(baseline),
                            "exposure": str(corrected),
                            "color": str(baseline),
                            "both": str(corrected),
                        },
                    }
                ],
            }
        ],
    }
    (report_dir / "contact_sheet.json").write_text(json.dumps(payload), encoding="utf-8")
    (report_dir / "review_manifest.json").write_text("{}", encoding="utf-8")
    (report_dir / "review_package.json").write_text("{}", encoding="utf-8")
    (report_dir / "contact_sheet.html").write_text("<html></html>", encoding="utf-8")
    _write_minimal_array_calibration(tmp_path / "array_calibration.json")
    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir(parents=True)
    (analysis_dir / "G007_B057_0324YT_001.analysis.json").write_text(
        json.dumps(
            {
                "clip_id": "G007_B057_0324YT_001",
                "diagnostics": {
                    "neutral_samples": [
                        {"label": "left", "roi_variance": 0.0001},
                        {"label": "center", "roi_variance": 0.0001},
                        {"label": "right", "roi_variance": 0.0001},
                    ]
                },
            }
        ),
        encoding="utf-8",
    )

    validation = validate_review_run_contract(str(tmp_path))

    assert validation["status"] == "success"
    assert validation["physical_validation"]["status"] == "success"
    assert validation["preview_reference_count"] == 2
    assert validation["preview_existing_count"] == 2
    assert Path(validation["validation_path"]).exists()


def test_validate_review_run_contract_fails_when_required_preview_missing(tmp_path: Path) -> None:
    report_dir = tmp_path / "report"
    previews_dir = tmp_path / "previews"
    report_dir.mkdir(parents=True)
    previews_dir.mkdir(parents=True)
    existing = previews_dir / "G007_B057_0324YT_001.original.review.jpg"
    missing = previews_dir / "G007_B057_0324YT_001.exposure.review.median.jpg"
    _write_test_preview(existing)
    (tmp_path / "summary.json").write_text("{}", encoding="utf-8")
    (previews_dir / "preview_commands.json").write_text("{}", encoding="utf-8")
    payload = {
        "clip_count": 1,
        "shared_originals": [{"clip_id": "G007_B057_0324YT_001", "original_frame": str(existing)}],
        "strategies": [
            {
                "strategy": "median",
                "clips": [
                    {
                        "clip_id": "G007_B057_0324YT_001",
                        "original_frame": str(existing),
                        "exposure_corrected": str(missing),
                        "preview_variants": {"original": str(existing), "exposure": str(missing)},
                    }
                ],
            }
        ],
    }
    (report_dir / "contact_sheet.json").write_text(json.dumps(payload), encoding="utf-8")
    (report_dir / "review_manifest.json").write_text("{}", encoding="utf-8")
    (report_dir / "review_package.json").write_text("{}", encoding="utf-8")
    _write_minimal_array_calibration(tmp_path / "array_calibration.json")

    validation = validate_review_run_contract(str(tmp_path))

    assert validation["status"] == "failed"
    assert validation["preview_reference_count"] == 2
    assert validation["preview_existing_count"] == 1
    assert validation["missing_preview_paths"] == [str(missing)]
    assert any("Missing 1 preview image" in error for error in validation["errors"])
    assert any("No human-readable report artifact was produced" in error for error in validation["errors"])


def test_validate_review_run_contract_fails_when_preview_is_unreadable(tmp_path: Path) -> None:
    report_dir = tmp_path / "report"
    previews_dir = tmp_path / "previews"
    report_dir.mkdir(parents=True)
    previews_dir.mkdir(parents=True)
    unreadable = previews_dir / "G007_B057_0324YT_001.original.review.jpg"
    unreadable.write_bytes(b"not-a-real-jpeg")
    (tmp_path / "summary.json").write_text("{}", encoding="utf-8")
    (previews_dir / "preview_commands.json").write_text("{}", encoding="utf-8")
    payload = {
        "clip_count": 1,
        "shared_originals": [{"clip_id": "G007_B057_0324YT_001", "original_frame": str(unreadable)}],
        "strategies": [],
    }
    (report_dir / "contact_sheet.json").write_text(json.dumps(payload), encoding="utf-8")
    (report_dir / "review_manifest.json").write_text("{}", encoding="utf-8")
    (report_dir / "review_package.json").write_text("{}", encoding="utf-8")
    (report_dir / "contact_sheet.html").write_text("<html></html>", encoding="utf-8")
    _write_minimal_array_calibration(tmp_path / "array_calibration.json")

    validation = validate_review_run_contract(str(tmp_path))

    assert validation["status"] == "failed"
    assert validation["unreadable_preview_paths"] == [str(unreadable)]
    assert any("unreadable preview image" in error for error in validation["errors"])


def test_validate_review_run_contract_adds_physical_validation_metrics(tmp_path: Path) -> None:
    report_dir = tmp_path / "report"
    previews_dir = tmp_path / "previews"
    analysis_dir = tmp_path / "analysis"
    report_dir.mkdir(parents=True)
    previews_dir.mkdir(parents=True)
    analysis_dir.mkdir(parents=True)
    (tmp_path / "summary.json").write_text("{}", encoding="utf-8")
    (previews_dir / "preview_commands.json").write_text(
        json.dumps({"review_mode": "lightweight_analysis", "commands": [], "skipped_bulk_preview_rendering": True}),
        encoding="utf-8",
    )
    (report_dir / "contact_sheet.json").write_text(
        json.dumps({"clip_count": 1, "review_mode": "lightweight_analysis", "shared_originals": [], "strategies": []}),
        encoding="utf-8",
    )
    (report_dir / "review_manifest.json").write_text(json.dumps({"review_mode": "lightweight_analysis"}), encoding="utf-8")
    (report_dir / "review_package.json").write_text(json.dumps({"review_mode": "lightweight_analysis"}), encoding="utf-8")
    (report_dir / "contact_sheet.html").write_text("<html></html>", encoding="utf-8")
    _write_minimal_array_calibration(tmp_path / "array_calibration.json")
    (analysis_dir / "G007_B057_0324YT_001.analysis.json").write_text(
        json.dumps(
            {
                "clip_id": "G007_B057_0324YT_001",
                "diagnostics": {
                    "neutral_samples": [
                        {"label": "left", "roi_variance": 0.0002},
                        {"label": "center", "roi_variance": 0.0003},
                        {"label": "right", "roi_variance": 0.0002},
                    ]
                },
            }
        ),
        encoding="utf-8",
    )

    validation = validate_review_run_contract(str(tmp_path))

    assert validation["status"] == "success"
    assert validation["physical_validation"]["status"] == "success"
    assert validation["physical_validation"]["exposure"]["mean_exposure_error"] == pytest.approx(0.0, abs=1e-3)
    assert validation["physical_validation"]["neutrality"]["max_post_neutral_error"] == pytest.approx(0.0, abs=1e-6)
    assert validation["physical_validation"]["kelvin_tint_analysis"]["kelvin_is_stable"] is True
    assert validation["physical_validation"]["confidence"]["per_camera"][0]["roi_variance"] == pytest.approx(0.0002)


def test_validate_review_run_contract_adds_recommendation_commit_and_post_apply_outputs(tmp_path: Path) -> None:
    report_dir = tmp_path / "report"
    previews_dir = tmp_path / "previews"
    analysis_dir = tmp_path / "analysis"
    report_dir.mkdir(parents=True)
    previews_dir.mkdir(parents=True)
    analysis_dir.mkdir(parents=True)
    (tmp_path / "summary.json").write_text(json.dumps({"backend": "red", "mode": "scene"}), encoding="utf-8")
    (previews_dir / "preview_commands.json").write_text(
        json.dumps({"review_mode": "lightweight_analysis", "commands": [], "skipped_bulk_preview_rendering": True}),
        encoding="utf-8",
    )
    payload = {
        "clip_count": 1,
        "review_mode": "lightweight_analysis",
        "run_label": "subset_065",
        "executive_synopsis": "The array is tightly grouped and ready for a median-based commit.",
        "recommended_strategy": {
            "strategy_key": "median",
            "strategy_label": "Median",
            "reason": "Median keeps the array centered with the smallest overall correction spread.",
            "metrics": {"mean_confidence_penalty": 0.1},
        },
        "hero_recommendation": {
            "candidate_clip_id": "G007_B057_0324YT_001",
            "confidence": "medium",
            "reason": "The clip sits near the robust center and is not an outlier.",
        },
        "operator_recommendation": "Proceed with the recommended strategy and export commit values.",
        "shared_originals": [],
        "strategies": [],
        "per_camera_analysis": [
            {
                "camera_label": "G007_B057",
                "clip_id": "G007_B057_0324YT_001",
                "confidence": 0.92,
                "note": "Within normal correction range for this subset.",
                "is_hero_camera": False,
                "commit_values": {
                    "exposureAdjust": 0.125,
                    "kelvin": 5600,
                    "tint": -0.3,
                },
            }
        ],
    }
    (report_dir / "contact_sheet.json").write_text(json.dumps(payload), encoding="utf-8")
    (report_dir / "review_manifest.json").write_text(json.dumps({"review_mode": "lightweight_analysis"}), encoding="utf-8")
    (report_dir / "review_package.json").write_text(json.dumps({"review_mode": "lightweight_analysis"}), encoding="utf-8")
    (report_dir / "contact_sheet.html").write_text("<html></html>", encoding="utf-8")
    _write_minimal_array_calibration(tmp_path / "array_calibration.json")
    (analysis_dir / "G007_B057_0324YT_001.analysis.json").write_text(
        json.dumps(
            {
                "clip_id": "G007_B057_0324YT_001",
                "diagnostics": {
                    "neutral_samples": [
                        {"label": "left", "roi_variance": 0.0002},
                        {"label": "center", "roi_variance": 0.0002},
                        {"label": "right", "roi_variance": 0.0002},
                    ]
                },
            }
        ),
        encoding="utf-8",
    )

    validation = validate_review_run_contract(str(tmp_path))

    assert validation["status"] == "success"
    assert validation["recommendation"]["recommended_strategy"]["strategy_key"] == "median"
    assert validation["recommendation"]["hero_camera"] == "G007_B057_0324YT_001"
    assert validation["failure_modes"] == []
    assert validation["human_summary"]["executive_summary"].startswith("The array is tightly grouped")
    assert validation["commit_payload"]["camera_count"] == 1
    assert Path(validation["commit_payload"]["aggregate_path"]).exists()
    per_camera_payload = json.loads(Path(validation["commit_payload"]["per_camera_payloads"][0]["path"]).read_text(encoding="utf-8"))
    assert per_camera_payload["camera_id"] == "G007_B057"
    assert per_camera_payload["calibration"]["exposureAdjust"] == pytest.approx(0.125)
    assert validation["post_apply_validation"]["status"] == "success"
    assert validation["post_apply_validation"]["summary"]["exposure_error_reduced"] is True


def test_validate_review_run_contract_fails_when_physical_validation_thresholds_are_exceeded(tmp_path: Path) -> None:
    report_dir = tmp_path / "report"
    previews_dir = tmp_path / "previews"
    report_dir.mkdir(parents=True)
    previews_dir.mkdir(parents=True)
    (tmp_path / "summary.json").write_text("{}", encoding="utf-8")
    (previews_dir / "preview_commands.json").write_text(
        json.dumps({"review_mode": "lightweight_analysis", "commands": [], "skipped_bulk_preview_rendering": True}),
        encoding="utf-8",
    )
    (report_dir / "contact_sheet.json").write_text(
        json.dumps({"clip_count": 1, "review_mode": "lightweight_analysis", "shared_originals": [], "strategies": []}),
        encoding="utf-8",
    )
    (report_dir / "review_manifest.json").write_text(json.dumps({"review_mode": "lightweight_analysis"}), encoding="utf-8")
    (report_dir / "review_package.json").write_text(json.dumps({"review_mode": "lightweight_analysis"}), encoding="utf-8")
    (report_dir / "contact_sheet.html").write_text("<html></html>", encoding="utf-8")
    _write_minimal_array_calibration(tmp_path / "array_calibration.json")
    payload = json.loads((tmp_path / "array_calibration.json").read_text(encoding="utf-8"))
    camera = payload["cameras"][0]
    camera["measurement"]["measured_rgb_mean"] = [0.4, 0.2, 0.1]
    camera["measurement"]["measured_rgb_chromaticity"] = [0.5714, 0.2857, 0.1429]
    camera["solution"]["rgb_gains"] = [1.0, 1.0, 1.0]
    camera["solution"]["exposure_offset_stops"] = 0.0
    (tmp_path / "array_calibration.json").write_text(json.dumps(payload), encoding="utf-8")

    validation = validate_review_run_contract(str(tmp_path))

    assert validation["status"] == "failed"
    assert validation["physical_validation"]["status"] == "failed"
    assert any("Physical exposure validation failed" in error or "Physical neutrality validation failed" in error for error in validation["errors"])
    assert {item["code"] for item in validation["failure_modes"]} >= {"exposure_out_of_range", "neutrality_failure"}


def test_validate_review_run_contract_warns_and_excludes_single_outlier_when_cluster_is_still_usable(tmp_path: Path) -> None:
    report_dir = tmp_path / "report"
    previews_dir = tmp_path / "previews"
    analysis_dir = tmp_path / "analysis"
    report_dir.mkdir(parents=True)
    previews_dir.mkdir(parents=True)
    analysis_dir.mkdir(parents=True)
    (tmp_path / "summary.json").write_text(json.dumps({"backend": "red", "mode": "scene"}), encoding="utf-8")
    (previews_dir / "preview_commands.json").write_text(
        json.dumps({"review_mode": "lightweight_analysis", "commands": [], "skipped_bulk_preview_rendering": True}),
        encoding="utf-8",
    )
    clip_ids = [
        "G007_A064_0325RN_001",
        "H007_A064_0325PK_001",
        "I007_A064_0325XY_001",
        "G007_B064_032526_001",
    ]
    (report_dir / "contact_sheet.json").write_text(
        json.dumps(
            {
                "clip_count": 4,
                "review_mode": "lightweight_analysis",
                "run_label": "subset_064",
                "executive_synopsis": "The subset has one intentional low-exposure outlier but the remaining cluster is coherent.",
                "recommended_strategy": {
                    "strategy_key": "median",
                    "strategy_label": "Median",
                    "reason": "Median keeps the main cluster balanced.",
                    "metrics": {"mean_confidence_penalty": 0.22},
                },
                "hero_recommendation": {
                    "candidate_clip_id": "G007_B064_032526_001",
                    "confidence": "medium",
                    "reason": "The clip sits near the center of the stable cluster.",
                },
                "operator_recommendation": "Proceed with the non-excluded cameras and review the intentional outlier separately.",
                "shared_originals": [],
                "strategies": [],
                "per_camera_analysis": [
                    {
                        "camera_label": "G007_A064",
                        "clip_id": clip_ids[0],
                        "confidence": 0.81,
                        "note": "Within normal correction range for this subset.",
                        "is_hero_camera": False,
                        "commit_values": {"exposureAdjust": -0.12, "kelvin": 5664, "tint": 0.0},
                    },
                    {
                        "camera_label": "H007_A064",
                        "clip_id": clip_ids[1],
                        "confidence": 0.79,
                        "note": "Within normal correction range for this subset.",
                        "is_hero_camera": False,
                        "commit_values": {"exposureAdjust": -0.05, "kelvin": 5664, "tint": 0.1},
                    },
                    {
                        "camera_label": "I007_A064",
                        "clip_id": clip_ids[2],
                        "confidence": 0.30,
                        "note": "Intentional low-exposure camera. Review required.",
                        "is_hero_camera": False,
                        "commit_values": {"exposureAdjust": -1.219, "kelvin": 5664, "tint": 0.2},
                    },
                    {
                        "camera_label": "G007_B064",
                        "clip_id": clip_ids[3],
                        "confidence": 0.84,
                        "note": "Within normal correction range for this subset.",
                        "is_hero_camera": False,
                        "commit_values": {"exposureAdjust": 0.02, "kelvin": 5664, "tint": -0.1},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    (report_dir / "review_manifest.json").write_text(json.dumps({"review_mode": "lightweight_analysis"}), encoding="utf-8")
    (report_dir / "review_package.json").write_text(json.dumps({"review_mode": "lightweight_analysis"}), encoding="utf-8")
    (report_dir / "contact_sheet.html").write_text("<html></html>", encoding="utf-8")

    array_payload = {
        "schema": "r3dmatch_array_calibration_v1",
        "backend": "red",
        "measurement_domain": "scene",
        "target": {
            "exposure": {"log2_luminance_target": -2.473931188332412},
            "color": {"target_rgb_chromaticity": [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]},
        },
        "global_scene_intent": {
            "white_balance_model": {
                "model_key": "shared_kelvin_per_camera_tint",
                "model_label": "Shared Kelvin / Per-Camera Tint",
                "shared_kelvin": 5664,
                "shared_tint": 0.0,
            }
        },
        "cameras": [],
    }
    camera_specs = [
        (clip_ids[0], "G007_A064", [0.18, 0.18, 0.18], 0.0, 0.82, 0.01, 0.001),
        (clip_ids[1], "H007_A064", [0.18, 0.18, 0.18], 0.0, 0.79, 0.01, 0.001),
        (clip_ids[2], "I007_A064", [0.4, 0.4, 0.4], 0.0, 0.30, 0.35, 0.02),
        (clip_ids[3], "G007_B064", [0.18, 0.18, 0.18], 0.0, 0.84, 0.01, 0.001),
    ]
    for clip_id, camera_id, rgb_mean, exposure_offset, quality_confidence, sample_log2_spread, sample_chroma_spread in camera_specs:
        array_payload["cameras"].append(
            {
                "clip_id": clip_id,
                "camera_id": camera_id,
                "source_path": str(tmp_path / f"{clip_id}.R3D"),
                "measurement": {
                    "measured_rgb_mean": rgb_mean,
                    "measured_rgb_chromaticity": [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
                    "neutral_sample_log2_spread": sample_log2_spread,
                    "neutral_sample_chromaticity_spread": sample_chroma_spread,
                },
                "solution": {
                    "exposure_offset_stops": exposure_offset,
                    "rgb_gains": [1.0, 1.0, 1.0],
                    "kelvin": 5664,
                    "tint": 0.0,
                },
                "quality": {
                    "confidence": quality_confidence,
                    "neutral_sample_log2_spread": sample_log2_spread,
                    "neutral_sample_chromaticity_spread": sample_chroma_spread,
                    "post_color_residual": 0.0,
                },
            }
        )
        (analysis_dir / f"{clip_id}.analysis.json").write_text(
            json.dumps({"clip_id": clip_id, "diagnostics": {"neutral_samples": [{"roi_variance": 0.0002}]}}),
            encoding="utf-8",
        )
    (tmp_path / "array_calibration.json").write_text(json.dumps(array_payload), encoding="utf-8")

    validation = validate_review_run_contract(str(tmp_path))

    assert validation["status"] == "success"
    assert validation["physical_validation"]["status"] == "warning"
    assert validation["physical_validation"]["cluster_is_usable_after_exclusions"] is True
    assert validation["physical_validation"]["excluded_camera_count"] == 1
    assert validation["physical_validation"]["included_camera_count"] == 3
    assert validation["physical_validation"]["excluded_cameras"][0]["clip_id"] == "I007_A064_0325XY_001"
    assert any("Excluded 1 outlier camera" in warning for warning in validation["warnings"])
    assert {item["code"] for item in validation["failure_modes"]} >= {"excluded_cameras_present"}
    assert validation["human_summary"]["pass_fail_explanation"].startswith("The main camera cluster remains usable")
    assert validation["run_assessment"]["status"] == "READY_WITH_WARNINGS"
    assert validation["run_assessment"]["recommendation_strength"] == "MEDIUM_CONFIDENCE"
    assert validation["run_assessment"]["safe_to_push_later"] is True
    assert validation["human_summary"]["run_status"] == "READY_WITH_WARNINGS"
    assert len(validation["commit_payload"]["safe_camera_targets"]) == 3
    excluded_payload = next(item for item in validation["commit_payload"]["per_camera_payloads"] if item["camera_id"] == "I007_A064")
    excluded_payload_json = json.loads(Path(excluded_payload["path"]).read_text(encoding="utf-8"))
    assert excluded_payload_json["excluded_from_commit"] is True
    assert excluded_payload_json["safe_to_commit"] is False
    assert excluded_payload_json["trust_class"] == "EXCLUDED"


def test_validate_review_run_contract_marks_mock_runs_as_physical_validation_unsupported(tmp_path: Path) -> None:
    report_dir = tmp_path / "report"
    previews_dir = tmp_path / "previews"
    report_dir.mkdir(parents=True)
    previews_dir.mkdir(parents=True)
    (tmp_path / "summary.json").write_text(json.dumps({"backend": "mock", "mode": "scene"}), encoding="utf-8")
    (previews_dir / "preview_commands.json").write_text(
        json.dumps({"review_mode": "lightweight_analysis", "commands": [], "skipped_bulk_preview_rendering": True}),
        encoding="utf-8",
    )
    (report_dir / "contact_sheet.json").write_text(
        json.dumps({"clip_count": 0, "review_mode": "lightweight_analysis", "shared_originals": [], "strategies": []}),
        encoding="utf-8",
    )
    (report_dir / "review_manifest.json").write_text(json.dumps({"review_mode": "lightweight_analysis"}), encoding="utf-8")
    (report_dir / "review_package.json").write_text(json.dumps({"review_mode": "lightweight_analysis"}), encoding="utf-8")
    (report_dir / "contact_sheet.html").write_text("<html></html>", encoding="utf-8")
    payload = json.loads((tmp_path / "array_calibration.json").read_text(encoding="utf-8")) if (tmp_path / "array_calibration.json").exists() else None
    _write_minimal_array_calibration(tmp_path / "array_calibration.json")
    mock_payload = json.loads((tmp_path / "array_calibration.json").read_text(encoding="utf-8"))
    mock_payload["backend"] = "mock"
    (tmp_path / "array_calibration.json").write_text(json.dumps(mock_payload), encoding="utf-8")

    validation = validate_review_run_contract(str(tmp_path))

    assert validation["status"] == "success"
    assert validation["physical_validation"]["status"] == "unsupported"


def test_web_app_status_reports_review_validation_failure_detail(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("r3dmatch.web_app._ensure_scan_preview", lambda input_path, scan, form=None: None)
    app = create_app()
    client = app.test_client()
    out_dir = tmp_path / "out"
    report_dir = out_dir / "report"
    report_dir.mkdir(parents=True)
    (report_dir / "review_validation.json").write_text(
        json.dumps(
            {
                "status": "failed",
                "errors": ["Missing required artifact: review_manifest_json"],
                "warnings": [],
                "preview_reference_count": 48,
                "preview_existing_count": 48,
            }
        ),
        encoding="utf-8",
    )
    state = app.config["UI_STATE"]
    state.task.command = "python3 -m r3dmatch.cli review-calibration /tmp/in --out /tmp/out"
    state.task.output_path = str(out_dir)
    state.task.status = "running"
    state.task.returncode = 0
    state.task.stage = "Building report"
    state.task.stage_index = 4
    state.task.items_completed = 48
    state.task.items_total = 48

    payload = client.get("/status").get_json()

    assert payload["status"] == "failed"
    assert payload["stage"] == "Finalization failed"
    assert payload["status_detail"] == "Missing required artifact: review_manifest_json"
    assert payload["validation_status"] == "failed"
    assert payload["can_cancel"] is False


def test_web_app_status_fails_when_review_validation_exists_only_in_wrong_location(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("r3dmatch.web_app._ensure_scan_preview", lambda input_path, scan, form=None: None)
    app = create_app()
    client = app.test_client()
    out_dir = tmp_path / "out"
    report_dir = out_dir / "report"
    report_dir.mkdir(parents=True)
    (out_dir / "review_validation.json").write_text(
        json.dumps({"status": "success", "errors": [], "warnings": []}),
        encoding="utf-8",
    )
    state = app.config["UI_STATE"]
    state.task.command = "python3 -m r3dmatch.cli review-calibration /tmp/in --out /tmp/out"
    state.task.output_path = str(out_dir)
    state.task.status = "running"
    state.task.returncode = 0
    state.task.stage = "Building report"
    state.task.stage_index = 4

    payload = client.get("/status").get_json()

    assert payload["status"] == "failed"
    assert payload["stage"] == "Finalization failed"
    assert payload["status_detail"] == "Fresh review_validation.json was not produced for this run."
    assert payload["validation_status"] is None


def test_web_app_status_ignores_stale_review_artifacts_from_previous_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("r3dmatch.web_app._ensure_scan_preview", lambda input_path, scan, form=None: None)
    app = create_app()
    client = app.test_client()
    out_dir = tmp_path / "out"
    report_dir = out_dir / "report"
    previews_dir = out_dir / "previews"
    analysis_dir = out_dir / "analysis"
    report_dir.mkdir(parents=True)
    previews_dir.mkdir(parents=True)
    analysis_dir.mkdir(parents=True)
    old_time = 100.0
    for path in [
        report_dir / "contact_sheet.html",
        report_dir / "preview_contact_sheet.pdf",
        report_dir / "review_validation.json",
        analysis_dir / "G007_A063_032563_001.analysis.json",
        previews_dir / "G007_A063_032563_001.original.review.jpg",
    ]:
        path.write_text("{}" if path.suffix == ".json" else "stale", encoding="utf-8")
        os.utime(path, (old_time, old_time))
    (report_dir / "review_validation.json").write_text(json.dumps({"status": "success", "errors": [], "warnings": []}), encoding="utf-8")
    os.utime(report_dir / "review_validation.json", (old_time, old_time))

    state = app.config["UI_STATE"]

    class FakeProcess:
        def poll(self):  # type: ignore[no-untyped-def]
            return None

    state.task.command = "python3 -m r3dmatch.cli review-calibration /tmp/in --out /tmp/out"
    state.task.output_path = str(out_dir)
    state.task.status = "running"
    state.task.returncode = None
    state.task.stage = "Scanning sources"
    state.task.stage_index = 0
    state.task.items_completed = 0
    state.task.items_total = 12
    state.task.clip_count = 12
    state.task.process = FakeProcess()  # type: ignore[assignment]
    state.task.started_at = 200.0
    state.task.last_output_at = 200.0
    state.task.last_progress_at = 200.0

    monkeypatch.setattr("r3dmatch.web_app.time.time", lambda: 205.0)
    payload = client.get("/status").get_json()

    assert payload["status"] == "running"
    assert payload["stage"] == "Scanning sources"
    assert payload["report_ready"] is False
    assert payload["preview_html_path"] is None
    assert payload["preview_pdf_path"] is None
    assert payload["validation_status"] is None
    assert payload["items_completed"] == 0


def test_web_app_status_prefers_completed_over_late_cancel_when_validation_succeeds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("r3dmatch.web_app._ensure_scan_preview", lambda input_path, scan, form=None: None)
    app = create_app()
    client = app.test_client()
    out_dir = tmp_path / "out"
    report_dir = out_dir / "report"
    report_dir.mkdir(parents=True)
    (report_dir / "review_validation.json").write_text(
        json.dumps({"status": "success", "errors": [], "warnings": []}),
        encoding="utf-8",
    )
    state = app.config["UI_STATE"]
    state.task.command = "python3 -m r3dmatch.cli review-calibration /tmp/in --out /tmp/out"
    state.task.output_path = str(out_dir)
    state.task.status = "cancelling"
    state.task.returncode = 0
    state.task.cancellation_requested = True
    state.task.stage = "Cancelling..."
    state.task.stage_index = 4

    payload = client.get("/status").get_json()

    assert payload["status"] == "completed"
    assert payload["stage"] == "Complete"
    assert payload["validation_status"] == "success"


def test_web_app_status_hides_optional_validation_warning_from_completed_detail(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("r3dmatch.web_app._ensure_scan_preview", lambda input_path, scan, form=None: None)
    app = create_app()
    client = app.test_client()
    out_dir = tmp_path / "out"
    report_dir = out_dir / "report"
    report_dir.mkdir(parents=True)
    (report_dir / "review_validation.json").write_text(
        json.dumps({"status": "success", "errors": [], "warnings": ["Optional artifact missing: preview_contact_sheet_pdf"]}),
        encoding="utf-8",
    )
    state = app.config["UI_STATE"]
    state.task.command = "python3 -m r3dmatch.cli review-calibration /tmp/in --out /tmp/out"
    state.task.output_path = str(out_dir)
    state.task.status = "running"
    state.task.returncode = 0
    state.task.stage = "Building report"
    state.task.stage_index = 4

    payload = client.get("/status").get_json()

    assert payload["status"] == "completed"
    assert payload["status_detail"] == ""
    assert payload["validation_warnings"] == ["Optional artifact missing: preview_contact_sheet_pdf"]


def test_web_app_status_accepts_canonical_review_validation_with_validated_at_even_if_mtime_is_old(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("r3dmatch.web_app._ensure_scan_preview", lambda input_path, scan, form=None: None)
    app = create_app()
    client = app.test_client()
    out_dir = tmp_path / "subset_065"
    report_dir = out_dir / "report"
    report_dir.mkdir(parents=True)
    validation_path = report_dir / "review_validation.json"
    validation_path.write_text(
        json.dumps(
            {
                "status": "success",
                "errors": [],
                "warnings": ["Optional artifact missing: preview_contact_sheet_pdf"],
                "review_mode": "lightweight_analysis",
                "clip_count": 12,
                "validated_at": 200.5,
            }
        ),
        encoding="utf-8",
    )
    os.utime(validation_path, (150.0, 150.0))

    state = app.config["UI_STATE"]
    state.task.command = "python3 -m r3dmatch.cli review-calibration /tmp/in --out /tmp/out --review-mode lightweight_analysis"
    state.task.output_path = str(out_dir)
    state.task.review_mode = "lightweight_analysis"
    state.task.status = "running"
    state.task.returncode = 0
    state.task.stage = "Building report"
    state.task.stage_index = 4
    state.task.started_at = 200.0
    state.task.last_output_at = 200.0
    state.task.last_progress_at = 200.0

    payload = client.get("/status").get_json()

    assert payload["status"] == "completed"
    assert payload["stage"] == "Complete"
    assert payload["validation_status"] == "success"
    assert payload["status_detail"] == ""
    assert payload["output_folder"] == str(out_dir)
    assert payload["validation_path"] == str(validation_path)


def test_web_app_status_accepts_completed_review_validation_when_progress_reports_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("r3dmatch.web_app._ensure_scan_preview", lambda input_path, scan, form=None: None)
    app = create_app()
    client = app.test_client()
    out_dir = tmp_path / "subset_064"
    report_dir = out_dir / "report"
    report_dir.mkdir(parents=True)
    validation_path = report_dir / "review_validation.json"
    validation_path.write_text(
        json.dumps(
            {
                "status": "success",
                "errors": [],
                "warnings": [],
                "review_mode": "full_contact_sheet",
                "validated_at": 450.0,
                "validation_path": str(validation_path),
                "physical_validation": {"status": "success"},
            }
        ),
        encoding="utf-8",
    )
    os.utime(validation_path, (450.0, 450.0))
    (out_dir / "review_progress.json").write_text(
        json.dumps(
            {
                "phase": "review_complete",
                "detail": "Review package complete.",
                "stage_label": "Complete",
                "updated_at": 500.5,
                "extra": {"validation_status": "success"},
            }
        ),
        encoding="utf-8",
    )

    state = app.config["UI_STATE"]
    state.task.command = "python3 -m r3dmatch.cli review-calibration /tmp/in --out /tmp/out --review-mode full_contact_sheet"
    state.task.output_path = str(out_dir)
    state.task.review_mode = "full_contact_sheet"
    state.task.status = "failed"
    state.task.returncode = 0
    state.task.stage = "Finalization failed"
    state.task.stage_index = 4
    state.task.started_at = 500.0
    state.task.last_output_at = 500.0
    state.task.last_progress_at = 500.5

    payload = client.get("/status").get_json()

    assert payload["status"] == "completed"
    assert payload["stage"] == "Complete"
    assert payload["validation_status"] == "success"
    assert payload["validation_path"] == str(validation_path)


def test_web_app_status_uses_review_progress_before_analysis_artifacts_exist(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("r3dmatch.web_app._ensure_scan_preview", lambda input_path, scan, form=None: None)
    monkeypatch.setattr("r3dmatch.web_app._process_is_alive", lambda process: True)
    app = create_app()
    client = app.test_client()
    out_dir = tmp_path / "subset_065"
    out_dir.mkdir(parents=True)
    (out_dir / "review_progress.json").write_text(
        json.dumps(
            {
                "phase": "clip_measurement_start",
                "detail": "Measuring clip.",
                "stage_label": "Measuring clips",
                "clip_index": 1,
                "clip_count": 12,
                "current_clip_id": "G007_A063_032563_001",
                "updated_at": 200.5,
            }
        ),
        encoding="utf-8",
    )

    state = app.config["UI_STATE"]
    state.task.command = "python3 -m r3dmatch.cli review-calibration /tmp/in --out /tmp/out --review-mode lightweight_analysis"
    state.task.output_path = str(out_dir)
    state.task.review_mode = "lightweight_analysis"
    state.task.status = "running"
    state.task.returncode = None
    state.task.started_at = 200.0
    state.task.last_output_at = 200.0
    state.task.last_progress_at = 200.0

    payload = client.get("/status").get_json()

    assert payload["status"] in {"running", "stalled", "finishing"}
    assert payload["stage"] == "Measuring clips"
    assert payload["status_detail"] == "Measuring clip."
    assert payload["items_total"] == 12
    assert payload["items_completed"] == 1


def test_web_app_status_uses_canonical_task_output_path_over_stale_guessed_folder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("r3dmatch.web_app._ensure_scan_preview", lambda input_path, scan, form=None: None)
    app = create_app()
    client = app.test_client()
    parent_dir = tmp_path / "runs"
    stale_out_dir = parent_dir / "subset_12"
    real_out_dir = parent_dir / "subset_065"
    stale_out_dir.mkdir(parents=True)
    report_dir = real_out_dir / "report"
    report_dir.mkdir(parents=True)
    validation_path = report_dir / "review_validation.json"
    validation_path.write_text(
        json.dumps(
            {
                "status": "success",
                "errors": [],
                "warnings": [],
                "review_mode": "lightweight_analysis",
                "validated_at": 301.0,
                "validation_path": str(validation_path),
                "physical_validation": {"status": "success"},
            }
        ),
        encoding="utf-8",
    )

    state = app.config["UI_STATE"]
    state.form.update({"output_path": str(parent_dir), "resolved_output_path": str(stale_out_dir)})
    state.task.command = "python3 -m r3dmatch.cli review-calibration /tmp/in --out /tmp/out --review-mode lightweight_analysis"
    state.task.output_path = str(stale_out_dir)
    state.task.canonical_output_path = str(real_out_dir)
    state.task.status = "running"
    state.task.returncode = 0
    state.task.stage = "Building report"
    state.task.stage_index = 4
    state.task.started_at = 300.0
    state.task.last_output_at = 300.0
    state.task.last_progress_at = 300.0

    payload = client.get("/status").get_json()

    assert payload["status"] == "completed"
    assert payload["stage"] == "Complete"
    assert payload["validation_status"] == "success"
    assert payload["status_detail"] == ""
    assert payload["output_folder"] == str(real_out_dir)
    assert payload["validation_path"] == str(validation_path)
    assert payload["finalization_debug"]["resolved_run_directory"] == str(real_out_dir)
    assert payload["finalization_debug"]["resolved_validation_path"] == str(validation_path)


def test_web_app_status_populates_decision_surfaces_from_validated_report_artifacts_even_when_payload_mtimes_are_old(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("r3dmatch.web_app._ensure_scan_preview", lambda input_path, scan, form=None: None)
    app = create_app()
    client = app.test_client()
    out_dir = tmp_path / "subset_064"
    report_dir = out_dir / "report"
    report_dir.mkdir(parents=True)
    contact_sheet_path = report_dir / "contact_sheet.json"
    commit_payload_path = report_dir / "calibration_commit_payload.json"
    validation_path = report_dir / "review_validation.json"

    contact_sheet_path.write_text(
        json.dumps(
            {
                "review_mode": "full_contact_sheet",
                "matching_domain": "perceptual",
                "matching_domain_label": "Perceptual (IPP2 / BT.709 / BT.1886)",
                "clip_count": 2,
                "clips": [
                    {
                        "clip_id": "G007_A064_0001",
                        "camera_label": "GA",
                        "reference_use": "Included",
                        "trust_score": 0.91,
                        "confidence": 0.88,
                        "offset_to_anchor": 0.01,
                        "metrics": {
                            "exposure": {
                                "sample_2_ire": 31.2,
                                "camera_offset_from_anchor": 0.01,
                                "gray_exposure_summary": "Sample 1 34 / Sample 2 31 / Sample 3 28 IRE",
                            }
                        },
                        "ipp2_validation": {
                            "camera_label": "GA",
                            "reference_use": "Included",
                            "sample_2_ire": 31.2,
                            "camera_offset_from_anchor": 0.01,
                            "ipp2_gray_exposure_summary": "Sample 1 34 / Sample 2 31 / Sample 3 28 IRE",
                            "trust_score": 0.91,
                        },
                    },
                    {
                        "clip_id": "G007_B064_0001",
                        "camera_label": "GB",
                        "reference_use": "Excluded",
                        "trust_score": 0.22,
                        "confidence": 0.31,
                        "offset_to_anchor": -1.2,
                        "metrics": {
                            "exposure": {
                                "sample_2_ire": 36.8,
                                "camera_offset_from_anchor": -1.2,
                                "gray_exposure_summary": "Sample 1 40 / Sample 2 37 / Sample 3 33 IRE",
                            }
                        },
                        "ipp2_validation": {
                            "camera_label": "GB",
                            "reference_use": "Excluded",
                            "sample_2_ire": 36.8,
                            "camera_offset_from_anchor": -1.2,
                            "ipp2_gray_exposure_summary": "Sample 1 40 / Sample 2 37 / Sample 3 33 IRE",
                            "trust_score": 0.22,
                        },
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    commit_payload_path.write_text(
        json.dumps(
            {
                "schema_version": "r3dmatch_calibration_commit_payload_v1",
                "camera_targets": [
                    {
                        "camera_id": "G007_A064",
                        "clip_id": "G007_A064_0001",
                        "inventory_camera_label": "GA",
                        "inventory_camera_ip": "10.0.0.1",
                        "calibration": {"exposureAdjust": 0.11, "kelvin": 5600, "tint": 2.0},
                        "confidence": 0.88,
                        "excluded_from_commit": False,
                        "exclusion_reasons": [],
                        "reference_use": "Included",
                    }
                ],
                "per_camera_payloads": [],
            }
        ),
        encoding="utf-8",
    )
    os.utime(contact_sheet_path, (450.0, 450.0))
    os.utime(commit_payload_path, (450.0, 450.0))
    validation_path.write_text(
        json.dumps(
            {
                "status": "success",
                "errors": [],
                "warnings": [],
                "review_mode": "full_contact_sheet",
                "validated_at": 500.5,
                "validation_path": str(validation_path),
                "physical_validation": {"status": "success"},
                "recommendation": {
                    "confidence_score": 0.88,
                },
                "run_assessment": {
                    "status": "READY",
                    "operator_note": "Calibration payload is ready for review and camera sync.",
                },
                "required_artifacts": {
                    "contact_sheet_json": {"path": str(contact_sheet_path)},
                },
                "commit_payload": {
                    "aggregate_path": str(commit_payload_path),
                },
            }
        ),
        encoding="utf-8",
    )
    (out_dir / "review_progress.json").write_text(
        json.dumps(
            {
                "phase": "review_complete",
                "detail": "Review package complete.",
                "stage_label": "Complete",
                "updated_at": 500.6,
                "extra": {"validation_status": "success"},
            }
        ),
        encoding="utf-8",
    )

    state = app.config["UI_STATE"]
    state.task.command = "python3 -m r3dmatch.cli review-calibration /tmp/in --out /tmp/out --review-mode full_contact_sheet"
    state.task.output_path = str(out_dir)
    state.task.review_mode = "full_contact_sheet"
    state.task.status = "running"
    state.task.returncode = 0
    state.task.stage = "Building report"
    state.task.stage_index = 4
    state.task.started_at = 500.0
    state.task.last_output_at = 500.0
    state.task.last_progress_at = 500.6

    payload = client.get("/status").get_json()

    assert payload["status"] == "completed"
    assert payload["decision_surface"]["commit_readiness"]["state"] == "Ready for Commit"
    assert payload["decision_surface"]["commit_table"]["camera_count"] == 1
    assert "GA" in payload["decision_surface_html"]
    assert "31 IRE to 31 IRE" in payload["decision_surface_html"]
    assert "GA" in payload["commit_table_html"]
    assert payload["validation_path"] == str(validation_path)


def test_web_app_status_mismatched_form_path_cannot_override_successful_validation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("r3dmatch.web_app._ensure_scan_preview", lambda input_path, scan, form=None: None)
    app = create_app()
    client = app.test_client()
    parent_dir = tmp_path / "runs"
    stale_out_dir = parent_dir / "subset_12"
    real_out_dir = parent_dir / "subset_065"
    stale_out_dir.mkdir(parents=True)
    report_dir = real_out_dir / "report"
    report_dir.mkdir(parents=True)
    validation_path = report_dir / "review_validation.json"
    validation_path.write_text(
        json.dumps(
            {
                "status": "success",
                "errors": [],
                "warnings": ["Optional artifact missing: preview_contact_sheet_pdf"],
                "review_mode": "lightweight_analysis",
                "clip_count": 12,
                "validated_at": 500.5,
                "validation_path": str(validation_path),
                "physical_validation": {"status": "success"},
            }
        ),
        encoding="utf-8",
    )
    os.utime(validation_path, (450.0, 450.0))

    state = app.config["UI_STATE"]
    state.form.update({"output_path": str(parent_dir), "resolved_output_path": str(stale_out_dir)})
    state.task.command = "python3 -m r3dmatch.cli review-calibration /tmp/in --out /tmp/out --review-mode lightweight_analysis"
    state.task.output_path = str(stale_out_dir)
    state.task.canonical_output_path = str(real_out_dir)
    state.task.status = "running"
    state.task.returncode = 0
    state.task.stage = "Building report"
    state.task.stage_index = 4
    state.task.started_at = 500.0
    state.task.last_output_at = 500.0
    state.task.last_progress_at = 500.0

    payload = client.get("/status").get_json()

    assert payload["status"] == "completed"
    assert payload["stage"] == "Complete"
    assert payload["validation_status"] == "success"
    assert payload["status_detail"] == ""
    assert payload["output_folder"] == str(real_out_dir)
    assert payload["validation_path"] == str(validation_path)


def test_validate_review_run_contract_writes_validation_path_and_timestamp(tmp_path: Path) -> None:
    report_dir = tmp_path / "report"
    previews_dir = tmp_path / "previews"
    report_dir.mkdir(parents=True)
    previews_dir.mkdir(parents=True)
    (tmp_path / "summary.json").write_text("{}", encoding="utf-8")
    (report_dir / "contact_sheet.json").write_text(
        json.dumps({"clip_count": 1, "review_mode": "lightweight_analysis", "shared_originals": [], "strategies": []}),
        encoding="utf-8",
    )
    (report_dir / "review_manifest.json").write_text(json.dumps({"review_mode": "lightweight_analysis"}), encoding="utf-8")
    (report_dir / "review_package.json").write_text(json.dumps({"review_mode": "lightweight_analysis"}), encoding="utf-8")
    (report_dir / "contact_sheet.html").write_text("<html></html>", encoding="utf-8")
    (previews_dir / "preview_commands.json").write_text(
        json.dumps({"review_mode": "lightweight_analysis", "commands": [], "skipped_bulk_preview_rendering": True}),
        encoding="utf-8",
    )
    _write_minimal_array_calibration(tmp_path / "array_calibration.json")

    validation = validate_review_run_contract(str(tmp_path))
    persisted = json.loads((report_dir / "review_validation.json").read_text(encoding="utf-8"))

    assert validation["status"] == "success"
    assert persisted["validation_path"] == str(report_dir / "review_validation.json")
    assert isinstance(persisted["validated_at"], float)
    assert persisted["validated_at"] >= 0.0


def test_web_app_cancel_run_marks_task_and_requests_termination(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    terminated: list[object] = []

    def fake_terminate(process, *, grace_period=2.0):  # type: ignore[no-untyped-def]
        terminated.append(process)

    monkeypatch.setattr("r3dmatch.web_app.terminate_process_group", fake_terminate)
    app = create_app()
    client = app.test_client()
    state = app.config["UI_STATE"]
    cancel_file = tmp_path / "cancel.flag"
    class FakeProcess:
        def poll(self):  # type: ignore[no-untyped-def]
            return None

    fake_process = FakeProcess()
    state.task.status = "running"
    state.task.command = "python3 -m r3dmatch.cli review-calibration /tmp/in --out /tmp/out"
    state.task.cancel_file_path = str(cancel_file)
    state.task.process = fake_process  # type: ignore[assignment]

    response = client.post("/cancel-run")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["ok"] is True
    assert cancel_file.exists()
    assert state.task.cancellation_requested is True
    assert state.task.status == "cancelling"
    assert terminated == [fake_process]


def test_run_cancellable_subprocess_raises_and_terminates_when_cancelled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cancel_file = tmp_path / "cancel.flag"
    monkeypatch.setenv(CANCEL_FILE_ENV, str(cancel_file))

    class FakeProcess:
        def __init__(self) -> None:
            self.returncode = None
            self.terminated = False
            self.communicate_calls = 0

        def communicate(self, timeout=None):  # type: ignore[no-untyped-def]
            self.communicate_calls += 1
            if self.communicate_calls == 1:
                cancel_file.write_text("cancel\n", encoding="utf-8")
                raise subprocess.TimeoutExpired(cmd=["fake"], timeout=timeout)
            self.returncode = -15
            return ("", "")

        def poll(self):  # type: ignore[no-untyped-def]
            return self.returncode

        def terminate(self):  # type: ignore[no-untyped-def]
            self.terminated = True
            self.returncode = -15

        def kill(self):  # type: ignore[no-untyped-def]
            self.returncode = -9

    fake_process = FakeProcess()
    monkeypatch.setattr("r3dmatch.execution.subprocess.Popen", lambda *args, **kwargs: fake_process)

    with pytest.raises(CancellationError):
        run_cancellable_subprocess(["/bin/sleep", "5"])

    assert fake_process.terminated is True


def test_web_review_command_includes_hero_clip() -> None:
    command = build_review_web_command(
        "/Users/sfouasnon/Desktop/R3DMatch",
        {
            "input_path": "/tmp/in",
            "output_path": "/tmp/out",
            "run_label": "",
            "backend": "red",
            "target_type": "gray_sphere",
            "processing_mode": "both",
            "matching_domain": "scene",
            "roi_x": "0.2",
            "roi_y": "0.2",
            "roi_w": "0.4",
            "roi_h": "0.4",
            "selected_clip_ids": [],
            "selected_clip_groups": [],
            "target_strategies": ["hero-camera"],
            "reference_clip_id": "",
            "hero_clip_id": "G007_D060_0324M6_001",
            "review_mode": "full_contact_sheet",
            "report_focus": "anchors",
            "preview_mode": "monitoring",
            "preview_lut": "",
        },
    )
    joined = " ".join(command)
    assert "--target-strategy hero-camera" in joined
    assert "--hero-clip-id G007_D060_0324M6_001" in joined
    assert "--report-focus anchors" in joined


def test_web_review_command_includes_subset_and_matching_domain() -> None:
    command = build_review_web_command(
        "/Users/sfouasnon/Desktop/R3DMatch",
        {
            "input_path": "/tmp/in",
            "output_path": "/tmp/out",
            "run_label": "clip63_even",
            "backend": "red",
            "target_type": "gray_sphere",
            "processing_mode": "both",
            "matching_domain": "perceptual",
            "roi_x": "0.2",
            "roi_y": "0.2",
            "roi_w": "0.4",
            "roi_h": "0.4",
            "selected_clip_ids": ["G007_B057_0324YT_063", "G007_C057_0324YT_063"],
            "selected_clip_groups": ["063"],
            "target_strategies": ["median"],
            "reference_clip_id": "",
            "hero_clip_id": "",
            "review_mode": "lightweight_analysis",
            "report_focus": "auto",
            "preview_mode": "monitoring",
            "preview_lut": "",
        },
    )
    joined = " ".join(command)
    assert "--run-label clip63_even" in joined
    assert "--matching-domain perceptual" in joined
    assert "--review-mode lightweight_analysis" in joined
    assert "--report-focus auto" in joined
    assert "--clip-group 063" in joined
    assert "--clip-id G007_B057_0324YT_063" in joined


def test_web_app_scan_page_uses_fixed_measurement_domain_and_advanced_roi_controls(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("r3dmatch.web_app._ensure_scan_preview", lambda input_path, scan, form=None: None)
    app = create_app()
    client = app.test_client()
    response = client.post("/scan", data={"input_path": str(tmp_path), "output_path": str(tmp_path / "out")})
    assert response.status_code == 200
    assert b"Measurement Domain" in response.data
    assert b"Perceptual (IPP2 / BT.709 / BT.1886)" in response.data
    assert b"Scene-Referred (REDWideGamutRGB / Log3G10)" not in response.data
    assert b"Advanced / Debug Controls" in response.data
    assert b"Preview available after scan or first render" in response.data
    assert b"sphere_auto" in response.data
    assert b"center" in response.data
    assert b"full" in response.data
    assert b"manual" in response.data
    assert b"Hero Camera" in response.data
    assert b"Browse" in response.data
    assert b"RED SDK Runtime" in response.data
    assert b"REDLine Tool" in response.data
    assert b"REDLine Executable Path" in response.data
    assert b"Save REDLine Path" in response.data


def test_web_app_scan_page_shows_roi_guidance_and_preview_note(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    clip_dir = tmp_path / "G007_A063_032563.RDC"
    clip_dir.mkdir()
    (clip_dir / "G007_A063_032563_001.R3D").write_bytes(b"")

    def fake_preview(input_path, scan, form=None):  # type: ignore[no-untyped-def]
        scan["preview_note"] = "ROI preview uses G007_A063_032563_001 from group 063 with the active monitoring preview transform."
        scan["preview_warning"] = None
        return "/tmp/roi-preview.jpg"

    monkeypatch.setattr("r3dmatch.web_app._ensure_scan_preview", fake_preview)
    app = create_app()
    client = app.test_client()
    response = client.post(
        "/scan",
        data={"input_path": str(tmp_path), "output_path": str(tmp_path / "out"), "selected_clip_groups": ["063"]},
    )
    assert response.status_code == 200
    assert b"Draw region around gray target" in response.data
    assert b"ROI preview uses G007_A063_032563_001 from group 063" in response.data


def test_web_app_scan_page_survives_preview_generation_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    clip_dir = tmp_path / "G007_A063_032563.RDC"
    clip_dir.mkdir()
    (clip_dir / "G007_A063_032563_001.R3D").write_bytes(b"")

    monkeypatch.setattr(
        "r3dmatch.web_app._detect_redline_capabilities",
        lambda executable: (_ for _ in ()).throw(FileNotFoundError("missing REDLine binary")),
    )
    app = create_app()
    client = app.test_client()
    response = client.post("/scan", data={"input_path": str(tmp_path), "output_path": str(tmp_path / "out")})
    assert response.status_code == 200
    assert b"ROI preview unavailable because preview extraction failed" in response.data
    assert b"missing REDLine binary" in response.data


def test_browse_folder_route_returns_selected_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "r3dmatch.web_app._browse_folder_via_native_dialog",
        lambda repo_root, field, current_path, title: {
            "ok": True,
            "cancelled": False,
            "path": "/Volumes/RAID/calibration",
            "error": "",
        },
    )
    app = create_app()
    client = app.test_client()
    response = client.post("/browse-folder", data={"field": "input_path", "current_path": "", "title": "Select Calibration Folder"})
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["ok"] is True
    assert payload["path"] == "/Volumes/RAID/calibration"


def test_browse_folder_route_reports_cancel(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "r3dmatch.web_app._browse_folder_via_native_dialog",
        lambda repo_root, field, current_path, title: {
            "ok": True,
            "cancelled": True,
            "path": "",
            "error": "",
        },
    )
    app = create_app()
    client = app.test_client()
    response = client.post("/browse-folder", data={"field": "output_path", "current_path": "/tmp/out", "title": "Select Output Folder"})
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["cancelled"] is True
    assert payload["path"] == ""


def test_browse_folder_route_fails_soft_when_picker_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "r3dmatch.web_app._browse_folder_via_native_dialog",
        lambda repo_root, field, current_path, title: {
            "ok": False,
            "cancelled": False,
            "path": "",
            "error": "PySide6 folder picker failed",
        },
    )
    app = create_app()
    client = app.test_client()
    response = client.post("/browse-folder", data={"field": "local_ingest_root", "current_path": "", "title": "Select Local Ingest Cache Root"})
    assert response.status_code == 400
    payload = response.get_json()
    assert payload["ok"] is False
    assert "PySide6 folder picker failed" in payload["error"]


def test_browse_file_route_returns_selected_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "r3dmatch.web_app._browse_file_via_native_dialog",
        lambda repo_root, field, current_path, title: {
            "ok": True,
            "cancelled": False,
            "path": "/Applications/REDCINE-X Professional/REDCINE-X PRO.app/Contents/MacOS/REDline",
            "error": "",
        },
    )
    app = create_app()
    client = app.test_client()
    response = client.post("/browse-file", data={"field": "redline_executable", "current_path": "", "title": "Select REDLine Executable"})
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["ok"] is True
    assert payload["path"].endswith("/REDline")


def test_browse_file_route_reports_cancel(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "r3dmatch.web_app._browse_file_via_native_dialog",
        lambda repo_root, field, current_path, title: {
            "ok": True,
            "cancelled": True,
            "path": "",
            "error": "",
        },
    )
    app = create_app()
    client = app.test_client()
    response = client.post("/browse-file", data={"field": "redline_executable", "current_path": "", "title": "Select REDLine Executable"})
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["cancelled"] is True


def test_browse_file_route_fails_soft_when_picker_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "r3dmatch.web_app._browse_file_via_native_dialog",
        lambda repo_root, field, current_path, title: {
            "ok": False,
            "cancelled": False,
            "path": "",
            "error": "PySide6 file picker failed",
        },
    )
    app = create_app()
    client = app.test_client()
    response = client.post("/browse-file", data={"field": "redline_executable", "current_path": "", "title": "Select REDLine Executable"})
    assert response.status_code == 400
    payload = response.get_json()
    assert payload["ok"] is False
    assert "PySide6 file picker failed" in payload["error"]


def test_configure_redline_route_persists_path_and_runtime_health(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = tmp_path / "config" / "redline.json"
    desktop_config = tmp_path / ".r3dmatch_config.json"
    redline_executable = tmp_path / "REDline"
    redline_executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    redline_executable.chmod(0o755)
    monkeypatch.setenv(DESKTOP_CONFIG_ENV, str(desktop_config))
    monkeypatch.setattr("r3dmatch.report.REDLINE_CONFIG_PATH", config_path)
    monkeypatch.setattr("r3dmatch.runtime_env.REDLINE_CONFIG_PATH", config_path)

    app = create_app()
    client = app.test_client()
    response = client.post("/configure-redline", data={"redline_executable": str(redline_executable)})
    assert response.status_code == 200
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    assert payload["redline_executable"] == str(redline_executable)
    desktop_payload = json.loads(desktop_config.read_text(encoding="utf-8"))
    assert desktop_payload["redline_path"] == str(redline_executable)
    assert b"Saved REDLine executable" in response.data
    assert runtime_health_payload()["redline_tool"]["configured_path"] == str(redline_executable)


def test_configure_redline_route_rejects_invalid_path_and_restores_previous_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = tmp_path / "config" / "redline.json"
    desktop_config = tmp_path / ".r3dmatch_config.json"
    previous = tmp_path / "REDline"
    previous.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    previous.chmod(0o755)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps({"redline_executable": str(previous)}, indent=2), encoding="utf-8")
    desktop_config.write_text(json.dumps({"redline_path": str(previous)}, indent=2), encoding="utf-8")
    monkeypatch.setenv(DESKTOP_CONFIG_ENV, str(desktop_config))
    monkeypatch.setattr("r3dmatch.report.REDLINE_CONFIG_PATH", config_path)
    monkeypatch.setattr("r3dmatch.runtime_env.REDLINE_CONFIG_PATH", config_path)

    app = create_app()
    client = app.test_client()
    response = client.post("/configure-redline", data={"redline_executable": str(tmp_path / 'missing-redline')})
    assert response.status_code == 200
    persisted = json.loads(config_path.read_text(encoding="utf-8"))
    assert persisted["redline_executable"] == str(previous)
    desktop_persisted = json.loads(desktop_config.read_text(encoding="utf-8"))
    assert desktop_persisted["redline_path"] == str(previous)
    assert b"REDLine executable is not available" in response.data


def test_ui_self_check_reports_visible_sections() -> None:
    lines = run_ui_self_check("/Users/sfouasnon/Desktop/R3DMatch")
    assert "header section created" in lines
    assert "calibration folder section created" in lines
    assert "output folder section created" in lines
    assert "basic settings section created" in lines
    assert "source summary section created" in lines
    assert "actions section created" in lines
    assert "log section created" in lines


def test_ui_self_check_minimal_mode_reports_minimal_sections() -> None:
    lines = run_ui_self_check("/Users/sfouasnon/Desktop/R3DMatch", minimal_mode=True)
    assert "minimal_mode=True" in lines


def test_original_wb_block_prefers_measured_original_still_context() -> None:
    block = _contact_sheet_original_wb_block(
        clip={
            "measured_rgb_chromaticity": [0.31, 0.36, 0.33],
            "confidence": 0.88,
            "neutral_sample_log2_spread": 0.012,
            "neutral_sample_chromaticity_spread": 0.004,
            "neutral_sample_count": 24,
            "commit_values": {"white_balance_axes": {"amber_blue": 0.06, "green_magenta": 0.03}},
            "white_balance_model_label": "Shared Kelvin / Per-Camera Tint",
            "pre_color_residual": 0.012,
        },
        clip_metadata={"kelvin": 5600, "tint": 2.0},
        top_level_wb_model={"shared_kelvin_mode": "shared", "shared_kelvin": 5625, "shared_tint_mode": "per_camera"},
    )
    assert block["source_label"] == "Measured from the original neutral target"
    assert "cool / blue" in str(block["derived_cast"]).lower()
    assert "neutral samples" in str(block["stability_summary"])
    assert "Shared Kelvin" in str(block["context"])


def test_original_wb_block_uses_nested_strategy_payload_when_top_level_fields_are_missing() -> None:
    block = _contact_sheet_original_wb_block(
        clip={
            "metrics": {
                "confidence": 0.82,
                "neutral_sample_log2_spread": 0.014,
                "neutral_sample_chromaticity_spread": 0.006,
                "neutral_sample_count": 18,
                "commit_values": {"white_balance_axes": {"amber_blue": -0.07, "green_magenta": 0.02}},
                "white_balance_model_label": "Shared Kelvin / Per-Camera Tint",
                "color": {"pre_residual": 0.011},
            },
            "commit_values": {"white_balance_axes": {"amber_blue": -0.07, "green_magenta": 0.02}},
            "measured_rgb_chromaticity": [0.361, 0.317, 0.322],
        },
        clip_metadata={"kelvin": 5600, "tint": 0.0},
        top_level_wb_model={"shared_kelvin_mode": "shared", "shared_kelvin": 5799, "shared_tint_mode": "per_camera"},
    )

    assert block["source_label"] == "Measured from the original neutral target"
    assert block["has_derived_cast"] is True
    assert "warm" in str(block["derived_cast"]).lower() or "magenta" in str(block["derived_cast"]).lower()
    assert "Shared Kelvin" in str(block["context"])


def test_exposure_plot_svg_suppresses_flat_operator_chart() -> None:
    svg = _build_exposure_plot_svg(
        [
            {"clip_id": "CAM001", "display_scalar_ire": 23.2},
            {"clip_id": "CAM002", "display_scalar_ire": 23.4},
            {"clip_id": "CAM003", "display_scalar_ire": 23.5},
        ],
        target_log2=np.log2(0.233),
    )
    assert svg == ""


def test_exposure_plot_svg_uses_display_domain_ire_when_informative() -> None:
    svg = _build_exposure_plot_svg(
        [
            {"clip_id": "CAM001", "display_scalar_ire": 19.5},
            {"clip_id": "CAM002", "display_scalar_ire": 23.5},
            {"clip_id": "CAM003", "display_scalar_ire": 27.0},
        ],
        target_log2=np.log2(0.235),
        reference_clip_id="CAM002",
    )
    assert "target center IRE" in svg
    assert "display-domain center scalar (IRE)" in svg


def test_measurement_uses_shared_roi_only() -> None:
    image = np.full((3, 20, 20), 0.8, dtype=np.float32)
    image[:, 8:12, 8:12] = 0.18
    full_measurement = measure_frame_color_and_exposure(image, mode="scene", lut=None, calibration_roi=None)
    roi_measurement = measure_frame_color_and_exposure(
        image,
        mode="scene",
        lut=None,
        calibration_roi={"x": 0.4, "y": 0.4, "w": 0.2, "h": 0.2},
    )
    assert full_measurement["measured_log2_luminance"] != pytest.approx(roi_measurement["measured_log2_luminance"])
    assert roi_measurement["measured_log2_luminance"] == roi_measurement["measured_log2_luminance_monitoring"]
    assert roi_measurement["measured_log2_luminance_raw"] == pytest.approx(np.log2(0.18), abs=0.2)
    assert roi_measurement["measured_log2_luminance_monitoring"] != pytest.approx(roi_measurement["measured_log2_luminance_raw"])


def test_monitoring_domain_measurement_is_primary_and_raw_is_preserved() -> None:
    image = np.full((3, 20, 20), 0.18, dtype=np.float32)
    measurement = measure_frame_color_and_exposure(
        image,
        mode="scene",
        lut=None,
        calibration_roi={"x": 0.25, "y": 0.25, "w": 0.5, "h": 0.5},
    )
    assert measurement["measured_log2_luminance"] == measurement["measured_log2_luminance_monitoring"]
    assert "measured_log2_luminance_raw" in measurement
    assert measurement["measured_log2_luminance_raw"] != measurement["measured_log2_luminance_monitoring"]


def test_measurement_reports_three_neutral_samples_and_spread() -> None:
    image = np.full((3, 90, 90), 0.18, dtype=np.float32)
    image[:, 32:58, 18:38] = np.asarray([0.24, 0.20, 0.16], dtype=np.float32).reshape(3, 1, 1)
    image[:, 32:58, 35:55] = np.asarray([0.18, 0.18, 0.18], dtype=np.float32).reshape(3, 1, 1)
    image[:, 32:58, 52:72] = np.asarray([0.14, 0.17, 0.23], dtype=np.float32).reshape(3, 1, 1)
    measurement = measure_frame_color_and_exposure(
        image,
        mode="scene",
        lut=None,
        calibration_roi={"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0},
    )
    assert measurement["neutral_sample_count"] == 3
    assert len(measurement["neutral_samples"]) == 3
    assert len(measurement["neutral_samples_raw"]) == 3
    assert measurement["neutral_sample_log2_spread"] > 0.0
    assert measurement["neutral_sample_chromaticity_spread"] > 0.0


def test_target_strategies_use_scene_domain_values_by_default() -> None:
    records = [
        {
            "clip_id": "G007_B057_0324YT_001",
            "group_key": "array_batch",
            "source_path": "/tmp/G007_B057_0324YT_001.R3D",
            "confidence": 0.9,
            "flags": [],
            "diagnostics": {
                "measured_log2_luminance": -1.0,
                "measured_log2_luminance_monitoring": -3.0,
                "measured_log2_luminance_raw": -1.0,
                "measured_rgb_chromaticity": [0.34, 0.33, 0.33],
                "calibration_roi": {"x": 0.2, "y": 0.2, "w": 0.4, "h": 0.4},
                "calibration_measurement_mode": "shared_roi",
                "exposure_measurement_domain": "monitoring",
            },
        },
        {
            "clip_id": "G007_C057_0324YT_001",
            "group_key": "array_batch",
            "source_path": "/tmp/G007_C057_0324YT_001.R3D",
            "confidence": 0.9,
            "flags": [],
            "diagnostics": {
                "measured_log2_luminance": -2.0,
                "measured_log2_luminance_monitoring": -1.5,
                "measured_log2_luminance_raw": -2.0,
                "measured_rgb_chromaticity": [0.33, 0.33, 0.34],
                "calibration_roi": {"x": 0.2, "y": 0.2, "w": 0.4, "h": 0.4},
                "calibration_measurement_mode": "shared_roi",
                "exposure_measurement_domain": "monitoring",
            },
        },
    ]
    strategies = _build_strategy_payloads(
        records,
        target_strategies=["optimal-exposure", "manual"],
        reference_clip_id="G007_B057_0324YT_001",
        anchor_target_log2=-1.1,
    )
    assert strategies[0]["reference_clip_id"] == "G007_B057_0324YT_001"
    assert strategies[0]["target_log2_luminance"] == pytest.approx(-1.0)
    assert strategies[1]["reference_clip_id"] == "G007_B057_0324YT_001"
    assert strategies[1]["target_log2_luminance"] == pytest.approx(-1.0)


def test_target_strategies_use_perceptual_domain_values_when_requested() -> None:
    records = [
        {
            "clip_id": "G007_B057_0324YT_001",
            "group_key": "array_batch",
            "source_path": "/tmp/G007_B057_0324YT_001.R3D",
            "confidence": 0.9,
            "flags": [],
            "diagnostics": {
                "measured_log2_luminance": -1.0,
                "measured_log2_luminance_monitoring": -3.0,
                "measured_log2_luminance_raw": -1.0,
                "measured_rgb_chromaticity": [0.34, 0.33, 0.33],
                "sphere_detection_confidence": 0.95,
            },
        },
        {
            "clip_id": "G007_C057_0324YT_001",
            "group_key": "array_batch",
            "source_path": "/tmp/G007_C057_0324YT_001.R3D",
            "confidence": 0.9,
            "flags": [],
            "diagnostics": {
                "measured_log2_luminance": -2.0,
                "measured_log2_luminance_monitoring": -1.5,
                "measured_log2_luminance_raw": -2.0,
                "measured_rgb_chromaticity": [0.33, 0.33, 0.34],
                "sphere_detection_confidence": 0.95,
            },
        },
    ]
    strategies = _build_strategy_payloads(
        records,
        target_strategies=["optimal-exposure", "manual"],
        reference_clip_id="G007_B057_0324YT_001",
        matching_domain="perceptual",
        anchor_target_log2=-1.6,
    )
    assert strategies[0]["matching_domain"] == "perceptual"
    assert strategies[0]["selection_diagnostics"]["selection_domain"] == "perceptual"
    assert strategies[0]["reference_clip_id"] == "G007_B057_0324YT_001"
    assert strategies[0]["target_log2_luminance"] == pytest.approx(-3.0)
    assert strategies[1]["reference_clip_id"] == "G007_B057_0324YT_001"
    assert strategies[1]["target_log2_luminance"] == pytest.approx(-3.0)


def test_optimal_exposure_prefers_best_gray_match_and_excludes_untrusted_outlier() -> None:
    records = [
        {
            "clip_id": "I007_A065_0325D0_001",
            "group_key": "array_batch",
            "source_path": "/tmp/I007_A065_0325D0_001.R3D",
            "confidence": 0.95,
            "flags": [],
            "diagnostics": {
                "measured_log2_luminance": -1.52,
                "measured_log2_luminance_monitoring": -1.52,
                "measured_log2_luminance_raw": -1.52,
                "measured_rgb_chromaticity": [0.343, 0.331, 0.326],
                "neutral_sample_log2_spread": 0.03,
                "neutral_sample_chromaticity_spread": 0.002,
            },
        },
        {
            "clip_id": "G007_D065_0325BJ_001",
            "group_key": "array_batch",
            "source_path": "/tmp/G007_D065_0325BJ_001.R3D",
            "confidence": 0.95,
            "flags": [],
            "diagnostics": {
                "measured_log2_luminance": -2.84,
                "measured_log2_luminance_monitoring": -2.84,
                "measured_log2_luminance_raw": -2.84,
                "measured_rgb_chromaticity": [0.338, 0.329, 0.333],
                "neutral_sample_log2_spread": 0.03,
                "neutral_sample_chromaticity_spread": 0.002,
            },
        },
        {
            "clip_id": "H007_D065_0325CS_001",
            "group_key": "array_batch",
            "source_path": "/tmp/H007_D065_0325CS_001.R3D",
            "confidence": 0.92,
            "flags": [],
            "diagnostics": {
                "measured_log2_luminance": -2.95,
                "measured_log2_luminance_monitoring": -2.95,
                "measured_log2_luminance_raw": -2.95,
                "measured_rgb_chromaticity": [0.337, 0.33, 0.333],
                "neutral_sample_log2_spread": 0.02,
                "neutral_sample_chromaticity_spread": 0.002,
            },
        },
        {
            "clip_id": "I007_D065_0325BP_001",
            "group_key": "array_batch",
            "source_path": "/tmp/I007_D065_0325BP_001.R3D",
            "confidence": 0.89,
            "flags": [],
            "diagnostics": {
                "measured_log2_luminance": -2.85,
                "measured_log2_luminance_monitoring": -2.85,
                "measured_log2_luminance_raw": -2.85,
                "measured_rgb_chromaticity": [0.342, 0.331, 0.327],
                "neutral_sample_log2_spread": 0.03,
                "neutral_sample_chromaticity_spread": 0.003,
            },
        },
    ]
    quality_by_clip = {
        "I007_A065_0325D0_001": {
            "confidence": 0.58,
            "neutral_sample_log2_spread": 0.34,
            "neutral_sample_chromaticity_spread": 0.002,
            "flags": ["neutral_sample_exposure_spread_high"],
        },
        "G007_D065_0325BJ_001": {
            "confidence": 0.85,
            "neutral_sample_log2_spread": 0.03,
            "neutral_sample_chromaticity_spread": 0.003,
            "flags": [],
        },
        "H007_D065_0325CS_001": {
            "confidence": 0.74,
            "neutral_sample_log2_spread": 0.04,
            "neutral_sample_chromaticity_spread": 0.004,
            "flags": [],
        },
        "I007_D065_0325BP_001": {
            "confidence": 0.79,
            "neutral_sample_log2_spread": 0.05,
            "neutral_sample_chromaticity_spread": 0.004,
            "flags": [],
        },
    }
    strategies = _build_strategy_payloads(
        records,
        target_strategies=["optimal-exposure"],
        reference_clip_id=None,
        matching_domain="scene",
        quality_by_clip=quality_by_clip,
        anchor_target_log2=-2.84,
    )
    strategy = strategies[0]
    assert strategy["reference_clip_id"] == "G007_D065_0325BJ_001"
    assert strategy["selection_diagnostics"]["fallback_mode"] == "trusted_anchor"
    assert strategy["selection_diagnostics"]["scored_candidates"][0]["clip_id"] == "G007_D065_0325BJ_001"
    assert strategy["selection_diagnostics"]["screened_candidates"][0]["clip_id"] == "I007_A065_0325D0_001"
    assert "closest to the gray target" in strategy["strategy_summary"]


def test_optimal_exposure_screens_out_secondary_exposure_cluster() -> None:
    records = [
        {
            "clip_id": "G007_A064_0325AA_001",
            "group_key": "array_batch",
            "source_path": "/tmp/G007_A064_0325AA_001.R3D",
            "confidence": 0.82,
            "flags": [],
            "diagnostics": {
                "measured_log2_luminance": -2.82,
                "measured_log2_luminance_monitoring": -2.82,
                "measured_log2_luminance_raw": -2.82,
                "measured_rgb_chromaticity": [0.342, 0.331, 0.327],
                "neutral_sample_log2_spread": 0.03,
                "neutral_sample_chromaticity_spread": 0.003,
            },
        },
        {
            "clip_id": "H007_A064_0325BB_001",
            "group_key": "array_batch",
            "source_path": "/tmp/H007_A064_0325BB_001.R3D",
            "confidence": 0.79,
            "flags": [],
            "diagnostics": {
                "measured_log2_luminance": -2.78,
                "measured_log2_luminance_monitoring": -2.78,
                "measured_log2_luminance_raw": -2.78,
                "measured_rgb_chromaticity": [0.342, 0.331, 0.327],
                "neutral_sample_log2_spread": 0.04,
                "neutral_sample_chromaticity_spread": 0.003,
            },
        },
        {
            "clip_id": "I007_A064_0325CC_001",
            "group_key": "array_batch",
            "source_path": "/tmp/I007_A064_0325CC_001.R3D",
            "confidence": 0.77,
            "flags": [],
            "diagnostics": {
                "measured_log2_luminance": -2.75,
                "measured_log2_luminance_monitoring": -2.75,
                "measured_log2_luminance_raw": -2.75,
                "measured_rgb_chromaticity": [0.341, 0.332, 0.327],
                "neutral_sample_log2_spread": 0.03,
                "neutral_sample_chromaticity_spread": 0.004,
            },
        },
        {
            "clip_id": "J007_A064_0325DD_001",
            "group_key": "array_batch",
            "source_path": "/tmp/J007_A064_0325DD_001.R3D",
            "confidence": 0.84,
            "flags": [],
            "diagnostics": {
                "measured_log2_luminance": -1.92,
                "measured_log2_luminance_monitoring": -1.92,
                "measured_log2_luminance_raw": -1.92,
                "measured_rgb_chromaticity": [0.342, 0.331, 0.327],
                "neutral_sample_log2_spread": 0.02,
                "neutral_sample_chromaticity_spread": 0.003,
            },
        },
    ]
    quality_by_clip = {
        item["clip_id"]: {
            "confidence": float(item["confidence"]),
            "neutral_sample_log2_spread": float(item["diagnostics"]["neutral_sample_log2_spread"]),
            "neutral_sample_chromaticity_spread": float(item["diagnostics"]["neutral_sample_chromaticity_spread"]),
            "flags": [],
        }
        for item in records
    }
    strategy = _build_strategy_payloads(
        records,
        target_strategies=["optimal-exposure"],
        reference_clip_id=None,
        matching_domain="scene",
        quality_by_clip=quality_by_clip,
        anchor_target_log2=-2.79,
    )[0]
    assert strategy["reference_clip_id"] in {"G007_A064_0325AA_001", "H007_A064_0325BB_001", "I007_A064_0325CC_001"}
    screened = {row["clip_id"]: row for row in strategy["selection_diagnostics"]["screened_candidates"]}
    assert "J007_A064_0325DD_001" in screened
    assert "outside primary exposure cluster" in screened["J007_A064_0325DD_001"]["reasons"]


def test_gray_target_keeps_saturation_neutral() -> None:
    records = [
        {
            "clip_id": "G007_B057_0324YT_001",
            "group_key": "array_batch",
            "source_path": "/tmp/G007_B057_0324YT_001.R3D",
            "confidence": 0.95,
            "flags": [],
            "diagnostics": {
                "measured_log2_luminance_raw": -1.0,
                "measured_log2_luminance_monitoring": -1.0,
                "measured_rgb_chromaticity": [0.34, 0.33, 0.33],
                "saturation_fraction": 0.25,
                "raw_saturation_fraction": 0.25,
            },
        },
        {
            "clip_id": "G007_C057_0324YT_001",
            "group_key": "array_batch",
            "source_path": "/tmp/G007_C057_0324YT_001.R3D",
            "confidence": 0.95,
            "flags": [],
            "diagnostics": {
                "measured_log2_luminance_raw": -1.2,
                "measured_log2_luminance_monitoring": -1.2,
                "measured_rgb_chromaticity": [0.32, 0.33, 0.35],
                "saturation_fraction": 0.05,
                "raw_saturation_fraction": 0.05,
            },
        },
    ]
    strategy = _build_strategy_payloads(records, target_strategies=["median"], reference_clip_id=None, target_type="gray_sphere")[0]
    for clip in strategy["clips"]:
        assert clip["color_cdl"]["saturation"] == pytest.approx(1.0)
        assert clip["commit_values"]["saturation_supported"] is False


def test_commit_values_use_as_shot_white_balance_and_axis_residuals() -> None:
    commit = build_commit_values(
        exposure_adjust=0.25,
        rgb_gains=[1.02, 0.99, 0.99],
        measured_rgb_chromaticity=[0.335, 0.332, 0.333],
        target_rgb_chromaticity=[0.342, 0.331, 0.327],
        clip_metadata={"extra_metadata": {"extra_metadata": {"white_balance_kelvin": 5600.0, "white_balance_tint": 0.0}}},
        saturation=1.0,
        saturation_supported=False,
    )
    assert commit["kelvin"] != 5600 or commit["tint"] != 0.0
    assert commit["derivation_method"] == "neutral_axis_kelvin_tint_v1"
    assert commit["pre_neutral_residual"] >= 0.0
    assert "white_balance_axes" in commit


def test_white_balance_model_prefers_shared_kelvin_with_per_camera_tint() -> None:
    records = [
        {
            "clip_id": "A",
            "measured_rgb_chromaticity": [0.341, 0.333, 0.326],
            "clip_metadata": {"extra_metadata": {"extra_metadata": {"white_balance_kelvin": 5600.0, "white_balance_tint": 0.0}}},
            "confidence": 0.95,
            "sample_log2_spread": 0.02,
            "sample_chromaticity_spread": 0.002,
        },
        {
            "clip_id": "B",
            "measured_rgb_chromaticity": [0.343, 0.328, 0.329],
            "clip_metadata": {"extra_metadata": {"extra_metadata": {"white_balance_kelvin": 5600.0, "white_balance_tint": 0.0}}},
            "confidence": 0.95,
            "sample_log2_spread": 0.02,
            "sample_chromaticity_spread": 0.002,
        },
        {
            "clip_id": "C",
            "measured_rgb_chromaticity": [0.344, 0.330, 0.326],
            "clip_metadata": {"extra_metadata": {"extra_metadata": {"white_balance_kelvin": 5600.0, "white_balance_tint": 0.0}}},
            "confidence": 0.95,
            "sample_log2_spread": 0.02,
            "sample_chromaticity_spread": 0.002,
        },
    ]
    solved = solve_white_balance_model_for_records(records, target_rgb_chromaticity=[0.342, 0.331, 0.327])
    assert solved["model_key"] == "shared_kelvin_per_camera_tint"
    assert solved["shared_kelvin"] is not None
    tints = [float(solved["clips"][clip_id]["tint"]) for clip_id in ["A", "B", "C"]]
    assert len({round(value, 1) for value in tints}) > 1


def test_array_calibration_uses_scene_measurements_and_preserves_confidence() -> None:
    result = types.SimpleNamespace(
        clip_id="G007_B057_0324YT_001",
        source_path="/tmp/G007_B057_0324YT_001.R3D",
        confidence=0.0,
        clip_metadata=types.SimpleNamespace(
            to_dict=lambda: {"extra_metadata": {"extra_metadata": {"white_balance_kelvin": 5600.0, "white_balance_tint": 0.0}}}
        ),
        diagnostics={
            "measured_log2_luminance_monitoring": -2.5,
            "measured_log2_luminance_raw": -1.5,
            "measured_rgb_mean": [0.18, 0.18, 0.18],
            "measured_rgb_chromaticity": [0.34, 0.33, 0.33],
            "valid_pixel_count": 100,
            "raw_saturation_fraction": 0.0,
            "black_fraction": 0.0,
            "neutral_sample_count": 3,
            "neutral_sample_log2_spread": 0.03,
            "neutral_sample_chromaticity_spread": 0.002,
            "accepted_frames": 1,
            "sampled_frames": 1,
        },
    )
    calibration = build_array_calibration_from_analysis([result], input_path="/tmp/batch")
    camera = calibration.cameras[0]
    assert camera.measurement.measured_log2_luminance == pytest.approx(-1.5)
    assert camera.quality.confidence > 0.5
    assert camera.solution.kelvin is not None
    assert camera.solution.derivation_method in {
        "neutral_axis_shared_kelvin_per_camera_tint_v2",
        "neutral_axis_constrained_kelvin_per_camera_tint_v2",
        "neutral_axis_per_camera_kelvin_per_camera_tint_v2",
        "neutral_axis_shared_kelvin_shared_tint_v2",
    }


def test_array_calibration_excludes_invalid_measurements_from_targets() -> None:
    valid = types.SimpleNamespace(
        clip_id="G007_B057_0324YT_001",
        source_path="/tmp/G007_B057_0324YT_001.R3D",
        confidence=0.9,
        clip_metadata=types.SimpleNamespace(
            to_dict=lambda: {"extra_metadata": {"extra_metadata": {"white_balance_kelvin": 5600.0, "white_balance_tint": 0.0}}}
        ),
        diagnostics={
            "measurement_valid": True,
            "gray_target_measurement_valid": True,
            "measured_log2_luminance_monitoring": -2.5,
            "measured_log2_luminance_raw": -1.5,
            "measured_rgb_mean": [0.18, 0.18, 0.18],
            "measured_rgb_chromaticity": [0.34, 0.33, 0.33],
            "valid_pixel_count": 100,
            "raw_saturation_fraction": 0.0,
            "black_fraction": 0.0,
            "neutral_sample_count": 3,
            "neutral_sample_log2_spread": 0.03,
            "neutral_sample_chromaticity_spread": 0.002,
            "accepted_frames": 1,
            "sampled_frames": 1,
        },
    )
    invalid = types.SimpleNamespace(
        clip_id="G007_C057_0324YT_001",
        source_path="/tmp/G007_C057_0324YT_001.R3D",
        confidence=0.1,
        clip_metadata=types.SimpleNamespace(
            to_dict=lambda: {"extra_metadata": {"extra_metadata": {"white_balance_kelvin": 5600.0, "white_balance_tint": 0.0}}}
        ),
        diagnostics={
            "measurement_valid": False,
            "gray_target_measurement_valid": False,
            "measured_log2_luminance_monitoring": None,
            "measured_log2_luminance_raw": None,
            "measured_rgb_mean": [0.0, 0.0, 0.0],
            "measured_rgb_chromaticity": [1 / 3, 1 / 3, 1 / 3],
            "valid_pixel_count": 0,
            "raw_saturation_fraction": 0.0,
            "black_fraction": 0.0,
            "neutral_sample_count": 0,
            "neutral_sample_log2_spread": 0.0,
            "neutral_sample_chromaticity_spread": 0.0,
            "accepted_frames": 1,
            "sampled_frames": 1,
        },
    )
    calibration = build_array_calibration_from_analysis([valid, invalid], input_path="/tmp/batch")
    assert [camera.clip_id for camera in calibration.cameras] == ["G007_B057_0324YT_001"]
    assert calibration.target.exposure.included_camera_count == 1
    assert calibration.target.exposure.excluded_camera_ids == ["G007_C057_0324YT_001"]
    assert calibration.target.color.excluded_camera_ids == ["G007_C057_0324YT_001"]


def test_hero_camera_strategy_matches_to_selected_hero() -> None:
    records = [
        {
            "clip_id": "G007_B057_0324YT_001",
            "group_key": "array_batch",
            "source_path": "/tmp/G007_B057_0324YT_001.R3D",
            "confidence": 0.9,
            "flags": [],
            "diagnostics": {
                "measured_log2_luminance_monitoring": -2.8,
                "measured_log2_luminance_raw": -2.2,
                "measured_rgb_chromaticity": [0.35, 0.33, 0.32],
                "saturation_fraction": 0.08,
                "calibration_roi": {"x": 0.2, "y": 0.2, "w": 0.4, "h": 0.4},
                "calibration_measurement_mode": "shared_roi",
                "exposure_measurement_domain": "monitoring",
            },
        },
        {
            "clip_id": "G007_C057_0324YT_001",
            "group_key": "array_batch",
            "source_path": "/tmp/G007_C057_0324YT_001.R3D",
            "confidence": 0.9,
            "flags": [],
            "diagnostics": {
                "measured_log2_luminance_monitoring": -1.4,
                "measured_log2_luminance_raw": -1.1,
                "measured_rgb_chromaticity": [0.31, 0.33, 0.36],
                "saturation_fraction": 0.16,
                "calibration_roi": {"x": 0.2, "y": 0.2, "w": 0.4, "h": 0.4},
                "calibration_measurement_mode": "shared_roi",
                "exposure_measurement_domain": "monitoring",
            },
        },
    ]
    strategies = _build_strategy_payloads(records, target_strategies=["hero-camera"], reference_clip_id=None, hero_clip_id="G007_C057_0324YT_001")
    hero_strategy = strategies[0]
    assert hero_strategy["hero_clip_id"] == "G007_C057_0324YT_001"
    assert "hero camera" in hero_strategy["strategy_summary"].lower()
    hero_clip = next(item for item in hero_strategy["clips"] if item["clip_id"] == "G007_C057_0324YT_001")
    other_clip = next(item for item in hero_strategy["clips"] if item["clip_id"] == "G007_B057_0324YT_001")
    assert hero_clip["is_hero_camera"] is True
    assert hero_clip["exposure_offset_stops"] == pytest.approx(0.0)
    assert hero_clip["rgb_gains"] == pytest.approx([1.0, 1.0, 1.0])
    assert hero_clip["color_lggs"]["gain"] == pytest.approx([1.0, 1.0, 1.0])
    assert hero_clip["color_cdl"]["saturation"] == pytest.approx(1.0)
    assert other_clip["exposure_offset_stops"] == pytest.approx(1.1)
    assert other_clip["color_cdl"]["saturation"] != pytest.approx(1.0)


def test_manual_target_strategy_records_explicit_anchor_metadata() -> None:
    records = [
        {
            "clip_id": "G007_A063_032563_001",
            "group_key": "array_batch",
            "source_path": "/tmp/G007_A063_032563_001.R3D",
            "confidence": 0.9,
            "flags": [],
            "diagnostics": {
                "measured_log2_luminance_monitoring": -1.3,
                "measured_log2_luminance_raw": -1.3,
                "measured_rgb_chromaticity": [0.34, 0.33, 0.33],
            },
        },
        {
            "clip_id": "I007_D063_0325EZ_001",
            "group_key": "array_batch",
            "source_path": "/tmp/I007_D063_0325EZ_001.R3D",
            "confidence": 0.9,
            "flags": [],
            "diagnostics": {
                "measured_log2_luminance_monitoring": -1.7,
                "measured_log2_luminance_raw": -1.7,
                "measured_rgb_chromaticity": [0.33, 0.33, 0.34],
            },
        },
    ]
    strategy = _build_strategy_payloads(
        records,
        target_strategies=["manual-target"],
        reference_clip_id=None,
        manual_target_ire=39.0,
        exposure_anchor_mode="manual_target",
    )[0]
    assert strategy["strategy_key"] == "manual_target"
    assert strategy["anchor_mode"] == "manual_target"
    assert strategy["anchor_source"] == "39 IRE"
    assert strategy["anchor_scalar_value"] == pytest.approx(np.log2(39.0 / 100.0))
    assert "Manual target 39 IRE" in strategy["anchor_summary"]
    assert strategy["clips"][0]["camera_offset_from_anchor"] == pytest.approx(strategy["clips"][0]["exposure_offset_stops"])


def test_analyze_path_writes_manifest_and_sidecar(tmp_path: Path) -> None:
    clip_a = tmp_path / "G007_D060_0324M6_001.R3D"
    clip_b = tmp_path / "G007_D061_0324M6_002.R3D"
    clip_a.write_bytes(b"")
    clip_b.write_bytes(b"")
    summary = analyze_path(str(tmp_path), out_dir=str(tmp_path / "out"), mode="scene", backend="mock", lut_override=None, calibration_path=None, sample_count=4, sampling_strategy="uniform")
    clip_summary = summary["clips"][0]
    sidecar_payload = json.loads((tmp_path / "out" / "sidecars" / f"{clip_summary['clip_id']}.sidecar.json").read_text(encoding="utf-8"))
    assert clip_summary["group_key"] == derive_array_group_key(str(tmp_path))
    assert sidecar_payload["rmd_name"] == f"{clip_summary['clip_id']}.RMD"
    assert (tmp_path / "out" / "array_calibration.json").exists()


def test_analyze_path_writes_review_progress_early(tmp_path: Path) -> None:
    clip_a = tmp_path / "G007_D060_0324M6_001.R3D"
    clip_b = tmp_path / "G007_D061_0324M6_002.R3D"
    clip_a.write_bytes(b"")
    clip_b.write_bytes(b"")
    progress_path = tmp_path / "out" / "review_progress.json"

    analyze_path(
        str(tmp_path),
        out_dir=str(tmp_path / "out"),
        mode="scene",
        backend="mock",
        lut_override=None,
        calibration_path=None,
        sample_count=4,
        sampling_strategy="uniform",
        progress_path=str(progress_path),
    )

    payload = json.loads(progress_path.read_text(encoding="utf-8"))
    assert payload["phase"] == "analysis_complete"
    assert payload["clip_count"] == 2
    assert payload["stage_label"] == "Analysis complete"


def test_analyze_path_writes_workload_trace_for_half_res_single_frame(tmp_path: Path) -> None:
    clip_a = tmp_path / "G007_D060_0324M6_001.R3D"
    clip_a.write_bytes(b"")
    workload_path = tmp_path / "out" / "measurement_workload_trace.json"

    analyze_path(
        str(tmp_path),
        out_dir=str(tmp_path / "out"),
        mode="scene",
        backend="mock",
        lut_override=None,
        calibration_path=None,
        sample_count=1,
        sampling_strategy="uniform",
        half_res_decode=True,
        workload_trace_path=str(workload_path),
    )

    payload = json.loads(workload_path.read_text(encoding="utf-8"))
    assert payload["half_res_decode_enabled"] is True
    assert payload["clip_count"] == 1
    row = payload["clips"][0]
    assert row["frames_analyzed"] == 1
    assert row["decode_half_res"] is True
    assert row["decode_width"] == 48
    assert row["decode_height"] == 27
    assert row["representative_frame_index"] == 12
    assert row["strategy_reuse"] is True
    assert row["measurement_domain"] == "scene_sdk_decode_with_proxy_monitoring"


def test_resolve_measurement_worker_count_defaults_conservatively() -> None:
    assert _resolve_measurement_worker_count(requested_workers=None, measurement_source="scene_sdk_decode_with_proxy_monitoring", clip_count=12) == 1
    assert _resolve_measurement_worker_count(requested_workers=None, measurement_source="rendered_preview_ipp2", clip_count=1) == 1
    assert _resolve_measurement_worker_count(requested_workers=None, measurement_source="rendered_preview_ipp2", clip_count=12) in {1, 2}
    assert _resolve_measurement_worker_count(requested_workers=3, measurement_source="rendered_preview_ipp2", clip_count=2) == 2


def test_analyze_path_preserves_deterministic_clip_order_with_parallel_measurement(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    clip_names = [
        "G007_A068_0403IW_001.R3D",
        "H007_B068_0403YH_001.R3D",
        "H007_D068_0403AA_001.R3D",
    ]
    clip_paths = []
    for name in clip_names:
        path = tmp_path / name
        path.write_bytes(b"")
        clip_paths.append(path)

    monkeypatch.setattr("r3dmatch.matching.discover_clips", lambda _input: list(clip_paths))

    delays = {
        "G007_A068_0403IW_001": 0.10,
        "H007_B068_0403YH_001": 0.02,
        "H007_D068_0403AA_001": 0.06,
    }

    def fake_analyze_clip(  # type: ignore[no-untyped-def]
        source_path,
        *,
        mode,
        backend,
        lut_override,
        sample_count,
        sampling_strategy,
        calibration_roi=None,
        target_type=None,
        half_res_decode=False,
        measurement_source="scene_sdk_decode_with_proxy_monitoring",
        measurement_output_dir=None,
        rendered_preview_context=None,
        sphere_assist_entry=None,
    ):
        clip_id = Path(source_path).stem
        time.sleep(delays[clip_id])
        return ClipResult(
            clip_id=clip_id,
            group_key="subset_068",
            source_path=str(source_path),
            backend="red",
            clip_statistic_log2=-2.473931188332412,
            group_key_statistic_log2=-2.473931188332412,
            global_reference_log2=-2.473931188332412,
            raw_offset_stops=0.0,
            camera_baseline_stops=0.0,
            clip_trim_stops=0.0,
            final_offset_stops=0.0,
            confidence=0.91,
            sample_plan=SamplePlan(strategy="uniform", sample_count=1, start_frame=0, frame_step=1, max_frames=1),
            monitoring=MonitoringContext(
                mode="scene",
                ipp2_color_space="REDWideGamutRGB",
                ipp2_gamma_curve="Log3G10",
                active_lut_path=None,
                lut_override_path=None,
                resolved_lut_path=None,
            ),
            clip_metadata=ClipMetadata(
                clip_id=clip_id,
                group_key="subset_068",
                original_filename=f"{clip_id}.R3D",
                source_path=str(source_path),
                fps=24.0,
                width=4096,
                height=2160,
                total_frames=100,
            ),
            frame_stats=[
                FrameStat(
                    frame_index=0,
                    timestamp_seconds=0.0,
                    log_luminance_median=-2.473931188332412,
                    clipped_fraction=0.0,
                    valid_fraction=1.0,
                    accepted=True,
                )
            ],
            diagnostics={
                "measured_log2_luminance_monitoring": -2.473931188332412,
                "measured_log2_luminance_raw": -2.473931188332412,
                "measured_rgb_mean": [0.18, 0.18, 0.18],
                "measured_rgb_chromaticity": [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
                "neutral_sample_log2_spread": 0.0,
                "neutral_sample_chromaticity_spread": 0.0,
                "neutral_samples": [],
                "runtime_trace": {},
                "workload_trace": {},
            },
        )

    monkeypatch.setattr("r3dmatch.matching.analyze_clip", fake_analyze_clip)

    summary = analyze_path(
        str(tmp_path),
        out_dir=str(tmp_path / "out"),
        mode="scene",
        backend="mock",
        lut_override=None,
        calibration_path=None,
        sample_count=1,
        sampling_strategy="uniform",
        target_type="gray_sphere",
        measurement_source="rendered_preview_ipp2",
        measurement_workers=3,
    )

    assert [row["clip_id"] for row in summary["clips"]] == [Path(name).stem for name in clip_names]


def test_analyze_path_uses_rendered_preview_ipp2_for_lightweight_measurement(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_redline(monkeypatch)
    clip_a = tmp_path / "G007_D060_0324M6_001.R3D"
    clip_a.write_bytes(b"")
    workload_path = tmp_path / "out" / "measurement_workload_trace.json"
    runtime_path = tmp_path / "out" / "lightweight_runtime_trace.json"

    analyze_path(
        str(tmp_path),
        out_dir=str(tmp_path / "out"),
        mode="scene",
        backend="mock",
        lut_override=None,
        calibration_path=None,
        sample_count=1,
        sampling_strategy="uniform",
        calibration_roi={"x": "0.0", "y": "0.0", "w": "1.0", "h": "1.0"},
        target_type="gray_sphere",
        half_res_decode=True,
        workload_trace_path=str(workload_path),
        runtime_trace_path=str(runtime_path),
        measurement_source="rendered_preview_ipp2",
    )

    workload = json.loads(workload_path.read_text(encoding="utf-8"))
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    workload_row = workload["clips"][0]
    runtime_row = runtime["clips"][0]
    assert workload["measurement_source"] == "rendered_preview_ipp2"
    assert workload_row["measurement_domain"] == "rendered_preview_ipp2"
    assert workload_row["decode_half_res"] is False
    assert workload_row["frames_analyzed"] == 1
    assert runtime_row["measurement_source"] == "rendered_preview_ipp2"
    assert runtime_row["decode_half_res"] is False
    assert runtime_row["detection_count"] == 1
    assert runtime_row["gradient_axis_count"] in {0, 1}
    assert runtime_row["region_stat_count"] == 1
    assert runtime_row["durations_seconds"]["render_preview_seconds"] > 0.0
    assert runtime_row["rendered_image_path"].endswith(".tiff")
    analysis_payload = json.loads((tmp_path / "out" / "analysis" / "G007_D060_0324M6_001.analysis.json").read_text(encoding="utf-8"))
    diagnostics = dict(analysis_payload["diagnostics"])
    provenance = dict(diagnostics.get("measurement_provenance") or {})
    source_asset = dict(provenance.get("measurement_source_asset") or {})
    assert source_asset["exists_at_measurement_time"] is True
    assert int(source_asset["file_size_bytes"]) > 0
    assert len(str(source_asset["sha256"])) == 64
    assert int(dict(source_asset["image_dimensions"]).get("width", 0)) > 0
    zone_measurements = list(diagnostics.get("zone_measurements") or [])
    if zone_measurements:
        center_zone = next(item for item in zone_measurements if item["label"] == "center")
        pixel_trace = dict(center_zone.get("pixel_trace") or {})
        assert len(dict(pixel_trace.get("raw_rgb_preview") or {}).get("normalized_rgb_preview", [])) > 0
    else:
        measured_rgb = list(diagnostics.get("measured_rgb_chromaticity") or [])
        assert len(measured_rgb) == 3


def test_analyze_path_rendered_preview_measurement_retries_missing_tiff_and_records_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clip_a = tmp_path / "H007_C067_0403RE_001.R3D"
    clip_a.write_bytes(b"")
    attempts: dict[str, int] = {}

    def fake_render_preview_frame(  # type: ignore[no-untyped-def]
        input_r3d,
        output_path,
        *,
        frame_index,
        redline_executable,
        redline_capabilities,
        preview_settings,
        use_as_shot_metadata,
        exposure=None,
        kelvin=None,
        tint=None,
        red_gain=None,
        green_gain=None,
        blue_gain=None,
        color_cdl=None,
        rmd_path=None,
        use_rmd_mode=1,
        color_method=None,
    ):
        output = Path(output_path)
        attempts[str(output)] = attempts.get(str(output), 0) + 1
        if attempts[str(output)] == 1:
            return {
                "command": ["REDLine", "--i", input_r3d, "--o", output_path],
                "returncode": 0,
                "stdout": "render started",
                "stderr": "",
                "output_path": output_path,
            }
        output.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.full((96, 96, 3), 180, dtype=np.uint8), mode="RGB").save(output, format="TIFF")
        return {
            "command": ["REDLine", "--i", input_r3d, "--o", output_path],
            "returncode": 0,
            "stdout": "render recovered",
            "stderr": "",
            "output_path": output_path,
        }

    def fake_measure_rendered_preview_roi_ipp2(  # type: ignore[no-untyped-def]
        preview_path: str,
        calibration_roi,
        *,
        sphere_roi_override=None,
        manual_assist_entry=None,
    ):
        return {
            "measured_log2_luminance": -2.45,
            "measured_log2_luminance_monitoring": -2.45,
            "measured_rgb_mean": [0.34, 0.33, 0.33],
            "measured_rgb_chromaticity": [0.34, 0.33, 0.33],
            "valid_pixel_count": 512,
            "measured_saturation_fraction_monitoring": 0.0,
            "neutral_sample_count": 48,
            "neutral_sample_log2_spread": 0.05,
            "neutral_sample_chromaticity_spread": 0.01,
            "neutral_samples": [],
            "gray_exposure_summary": "S1 24 / S2 23 / S3 22 IRE",
            "bright_ire": 24.0,
            "center_ire": 23.0,
            "dark_ire": 22.0,
            "sample_1_ire": 24.0,
            "sample_2_ire": 23.0,
            "sample_3_ire": 22.0,
            "top_ire": 24.0,
            "mid_ire": 23.0,
            "bottom_ire": 22.0,
            "zone_spread_ire": 2.0,
            "zone_spread_stops": 0.1,
            "zone_measurements": [],
            "sphere_detection_confidence": 0.78,
            "sphere_detection_label": "MEDIUM",
            "sphere_detection_success": True,
            "sphere_detection_unresolved": False,
            "sphere_roi_source": "neutral_blob_recovery",
            "sphere_detection_details": {"reason": "recovered_after_render_retry"},
            "detected_sphere_roi": {"cx": 320.0, "cy": 210.0, "r": 88.0},
            "measurement_crop_bounds": {"x0": 0, "y0": 0, "x1": 16, "y1": 16},
            "measurement_crop_size": {"width": 16, "height": 16},
            "dominant_gradient_axis": {"axis": "vertical"},
            "measurement_runtime": {
                "sphere_detection_seconds": 0.01,
                "gradient_axis_seconds": 0.01,
                "zone_stat_seconds": 0.01,
            },
            "render_width": 16,
            "render_height": 16,
            "rendered_image_dtype": "uint16",
            "rendered_image_bit_depth": 16,
            "rendered_preview_format": "tiff",
        }

    monkeypatch.setattr("r3dmatch.report.render_preview_frame", fake_render_preview_frame)
    monkeypatch.setattr("r3dmatch.report._measure_rendered_preview_roi_ipp2", fake_measure_rendered_preview_roi_ipp2)
    monkeypatch.setattr(
        "r3dmatch.matching._resolve_rendered_preview_context",
        lambda: {
            "redline_executable": "REDLine",
            "redline_capabilities": {},
            "preview_settings": {
                "preview_mode": "monitoring",
                "preview_still_format": "tiff",
                "output_space": "BT.709",
                "output_gamma": "BT.1886",
                "highlight_rolloff": "medium",
            },
        },
    )

    analyze_path(
        str(tmp_path),
        out_dir=str(tmp_path / "out"),
        mode="scene",
        backend="mock",
        lut_override=None,
        calibration_path=None,
        sample_count=1,
        sampling_strategy="uniform",
        calibration_roi={"x": "0.0", "y": "0.0", "w": "1.0", "h": "1.0"},
        target_type="gray_sphere",
        half_res_decode=True,
        measurement_source="rendered_preview_ipp2",
    )

    analysis_payload = json.loads((tmp_path / "out" / "analysis" / "H007_C067_0403RE_001.analysis.json").read_text(encoding="utf-8"))
    diagnostics = dict(analysis_payload["diagnostics"])
    assert diagnostics["rendered_measurement_attempt_count"] == 2
    assert diagnostics["rendered_measurement_recovered_after_retry"] is True
    assert len(diagnostics["rendered_measurement_attempt_diagnostics"]) == 2
    provenance = dict(diagnostics["measurement_provenance"])
    render_identity = dict(provenance["render_identity"])
    assert render_identity["render_attempt_count"] == 2
    assert render_identity["render_recovered_after_retry"] is True
    assert len(render_identity["render_attempt_diagnostics"]) == 2


def test_review_calibration_command_passes_explicit_anchor_options(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def fake_review_calibration(input_path: str, **kwargs: object) -> dict[str, object]:
        captured["input_path"] = input_path
        captured.update(kwargs)
        return {"status": "ok"}

    monkeypatch.setattr("r3dmatch.cli.review_calibration", fake_review_calibration)
    result = runner.invoke(
        app,
        [
            "review-calibration",
            str(tmp_path),
            "--out",
            str(tmp_path / "out"),
            "--target-type",
            "gray_sphere",
            "--review-mode",
            "lightweight_analysis",
            "--exposure-anchor-mode",
            "manual-target",
            "--manual-target-ire",
            "39",
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert captured["exposure_anchor_mode"] == "manual_target"
    assert captured["manual_target_ire"] == pytest.approx(39.0)
    assert "manual_target" in list(captured["target_strategies"])


def test_sidecar_to_rmd_conversion_exposure_only() -> None:
    payload = {
        "schema": "r3dmatch_v2",
        "clip_id": "G007_D060_0324M6_001",
        "source_path": "/tmp/G007_D060_0324M6_001.R3D",
        "calibration_state": {
            "exposure_calibration_loaded": True,
            "exposure_baseline_applied_stops": 0.4,
            "color_calibration_loaded": False,
            "rgb_neutral_gains": None,
            "color_gains_state": None,
        },
        "rmd_mapping": {
            "exposure": {"final_offset_stops": 0.4},
            "color": {"rgb_neutral_gains": None},
        },
    }
    xml = render_rmd_xml(payload)
    assert "G007_D060_0324M6_001.RMD" in xml
    assert 'final_offset_stops="0.4"' in xml
    assert 'calibration_loaded="true"' in xml


def test_sidecar_to_rmd_conversion_with_color() -> None:
    payload = {
        "schema": "r3dmatch_v2",
        "clip_id": "G007_D060_0324M6_001",
        "source_path": "/tmp/G007_D060_0324M6_001.R3D",
        "calibration_state": {
            "exposure_calibration_loaded": True,
            "exposure_baseline_applied_stops": 0.4,
            "color_calibration_loaded": True,
            "rgb_neutral_gains": {"r": 1.1, "g": 1.0, "b": 0.9},
            "color_gains_state": "pending",
        },
        "rmd_mapping": {
            "exposure": {"final_offset_stops": 0.4},
            "color": {"rgb_neutral_gains": [1.1, 1.0, 0.9]},
        },
    }
    xml = render_rmd_xml(payload)
    assert 'rgb_neutral_gains=' in xml
    assert "1.1" in xml
    assert 'state="pending"' in xml


def test_sidecar_generation_with_exposure_only(tmp_path: Path) -> None:
    clip_path = tmp_path / "G007_D060_0324M6_001.R3D"
    clip_path.write_bytes(b"")
    exposure_path = tmp_path / "exposure_calibration.json"
    exposure_path.write_text(
        json.dumps(
            {
                "calibration_type": "exposure",
                "calibration_mode": "exposure",
                "object_type": "gray_card",
                "target_log2_luminance": -2.0,
                "reference_camera": None,
                "roi_file": None,
                "cameras": [
                    {
                        "clip_id": "G007_D060_0324M6_001",
                        "group_key": "G007_D060",
                        "source_path": str(clip_path),
                        "sampling_mode": "center_crop",
                        "sampling_region": {"sampling_mode": "center_crop", "center_crop": {"width_ratio": 0.3, "height_ratio": 0.3}, "roi": None, "detection_confidence": None, "fallback_used": False},
                        "measured_log2_luminance": -1.0,
                        "camera_baseline_stops": 0.4,
                        "confidence": 0.9,
                        "calibration_source": "gray_card",
                        "frame_index": 0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    analyze_path(str(tmp_path), out_dir=str(tmp_path / "out"), mode="scene", backend="mock", lut_override=None, calibration_path=None, exposure_calibration_path=str(exposure_path), color_calibration_path=None, sample_count=4, sampling_strategy="uniform")
    sidecar = json.loads((tmp_path / "out" / "sidecars" / "G007_D060_0324M6_001.sidecar.json").read_text(encoding="utf-8"))
    assert sidecar["calibration_state"]["exposure_calibration_loaded"] is True
    assert sidecar["calibration_state"]["color_calibration_loaded"] is False


def test_sidecar_generation_with_color_only(tmp_path: Path) -> None:
    clip_path = tmp_path / "G007_D060_0324M6_001.R3D"
    clip_path.write_bytes(b"")
    color_path = tmp_path / "color_calibration.json"
    color_path.write_text(
        json.dumps(
            {
                "calibration_type": "color",
                "calibration_mode": "color",
                "object_type": "neutral_patch",
                "roi_file": None,
                "cameras": [
                    {
                        "clip_id": "G007_D060_0324M6_001",
                        "group_key": "G007_D060",
                        "source_path": str(clip_path),
                        "sampling_mode": "center_crop",
                        "sampling_region": {"sampling_mode": "center_crop", "center_crop": {"width_ratio": 0.3, "height_ratio": 0.3}, "roi": None, "detection_confidence": None, "fallback_used": False},
                        "measured_channel_medians": {"r": 0.3, "g": 0.4, "b": 0.5},
                        "rgb_neutral_gains": {"r": 1.33, "g": 1.0, "b": 0.8},
                        "confidence": 0.9,
                        "calibration_source": "neutral_region",
                        "frame_index": 0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    analyze_path(str(tmp_path), out_dir=str(tmp_path / "out"), mode="scene", backend="mock", lut_override=None, calibration_path=None, exposure_calibration_path=None, color_calibration_path=str(color_path), sample_count=4, sampling_strategy="uniform")
    sidecar = json.loads((tmp_path / "out" / "sidecars" / "G007_D060_0324M6_001.sidecar.json").read_text(encoding="utf-8"))
    assert sidecar["calibration_state"]["exposure_calibration_loaded"] is False
    assert sidecar["calibration_state"]["color_calibration_loaded"] is True
    assert sidecar["calibration_state"]["rgb_neutral_gains"]["b"] == pytest.approx(0.8)


def test_sidecar_generation_with_both(tmp_path: Path) -> None:
    clip_path = tmp_path / "G007_D060_0324M6_001.R3D"
    clip_path.write_bytes(b"")
    exposure_path = tmp_path / "exposure.json"
    color_path = tmp_path / "color.json"
    exposure_path.write_text(
        json.dumps(
            {
                "calibration_type": "exposure",
                "calibration_mode": "exposure",
                "object_type": "gray_card",
                "target_log2_luminance": -2.0,
                "reference_camera": None,
                "roi_file": None,
                "cameras": [
                    {
                        "clip_id": "G007_D060_0324M6_001",
                        "group_key": "G007_D060",
                        "source_path": str(clip_path),
                        "sampling_mode": "center_crop",
                        "sampling_region": {"sampling_mode": "center_crop", "center_crop": {"width_ratio": 0.3, "height_ratio": 0.3}, "roi": None, "detection_confidence": None, "fallback_used": False},
                        "measured_log2_luminance": -1.2,
                        "camera_baseline_stops": 0.4,
                        "confidence": 0.9,
                        "calibration_source": "gray_card",
                        "frame_index": 0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    color_path.write_text(
        json.dumps(
            {
                "calibration_type": "color",
                "calibration_mode": "color",
                "object_type": "neutral_patch",
                "roi_file": None,
                "cameras": [
                    {
                        "clip_id": "G007_D060_0324M6_001",
                        "group_key": "G007_D060",
                        "source_path": str(clip_path),
                        "sampling_mode": "center_crop",
                        "sampling_region": {"sampling_mode": "center_crop", "center_crop": {"width_ratio": 0.3, "height_ratio": 0.3}, "roi": None, "detection_confidence": None, "fallback_used": False},
                        "measured_channel_medians": {"r": 0.3, "g": 0.4, "b": 0.5},
                        "rgb_neutral_gains": {"r": 1.33, "g": 1.0, "b": 0.8},
                        "confidence": 0.9,
                        "calibration_source": "neutral_region",
                        "frame_index": 0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    summary = analyze_path(str(tmp_path), out_dir=str(tmp_path / "out"), mode="scene", backend="mock", lut_override=None, calibration_path=None, exposure_calibration_path=str(exposure_path), color_calibration_path=str(color_path), sample_count=4, sampling_strategy="uniform")
    sidecar = json.loads((tmp_path / "out" / "sidecars" / "G007_D060_0324M6_001.sidecar.json").read_text(encoding="utf-8"))
    assert sidecar["calibration_state"]["exposure_calibration_loaded"] is True
    assert sidecar["calibration_state"]["color_calibration_loaded"] is True
    assert summary["clips"][0]["exposure_baseline_applied_stops"] == pytest.approx(0.4)


def test_discover_clips_skips_hidden_resource_forks_and_non_r3d(tmp_path: Path) -> None:
    clip_dir = tmp_path / "cards" / "A001_001.RDC"
    clip_dir.mkdir(parents=True)
    valid = clip_dir / "G007_D060_0324M6_001.R3D"
    (clip_dir / "._G007_D060_0324M6_001.R3D").write_bytes(b"")
    (clip_dir / "notes.txt").write_text("ignore", encoding="utf-8")
    valid.write_bytes(b"")
    assert discover_clips(str(tmp_path)) == [valid]


def test_camera_group_uses_first_token_for_a007_pattern() -> None:
    assert camera_group_from_clip_id("A007_A033_0202CB_001") == "A007"


def test_analysis_uses_group_key_not_legacy_camera_group(tmp_path: Path) -> None:
    clip_a = tmp_path / "G007_D060_0324M6_001.R3D"
    clip_b = tmp_path / "G007_D061_0324M6_002.R3D"
    clip_a.write_bytes(b"")
    clip_b.write_bytes(b"")
    summary = analyze_path(str(tmp_path), out_dir=str(tmp_path / "out"), mode="scene", backend="mock", lut_override=None, calibration_path=None, sample_count=4, sampling_strategy="uniform")
    by_clip = {item["clip_id"]: item for item in summary["clips"]}
    expected_group = derive_array_group_key(str(tmp_path))
    assert by_clip["G007_D060_0324M6_001"]["group_key"] == expected_group
    assert by_clip["G007_D061_0324M6_002"]["group_key"] == expected_group


def test_measure_sphere_roi_applies_masking_and_percentile_trimming() -> None:
    image = np.full((3, 10, 10), 0.1, dtype=np.float32)
    image[:, 3:7, 3:7] = 0.25
    image[:, 4, 4] = 1.0
    image[:, 5, 5] = 0.01
    measurement = measure_sphere_from_roi(image, SphereROI(cx=4.5, cy=4.5, r=2.5), low_percentile=10.0, high_percentile=90.0)
    assert measurement["measured_sphere_log2"] == pytest.approx(np.log2(0.25), abs=0.15)
    assert measurement["sampling_method"] in {"refined_interior_mask", "interior_circle_fallback"}
    assert measurement["interior_radius_ratio"] < 1.0


def test_measure_sphere_roi_prefers_interior_pixels_over_edge_contamination() -> None:
    image = np.full((3, 32, 32), 0.05, dtype=np.float32)
    yy, xx = np.meshgrid(np.arange(32, dtype=np.float32), np.arange(32, dtype=np.float32), indexing="ij")
    circle = ((xx - 16.0) ** 2 + (yy - 16.0) ** 2) <= 10.0 ** 2
    interior = ((xx - 16.0) ** 2 + (yy - 16.0) ** 2) <= 7.5 ** 2
    edge_ring = circle & ~interior
    image[:, circle] = 0.25
    image[:, edge_ring] = 0.55
    measurement = measure_sphere_from_roi(image, SphereROI(cx=16.0, cy=16.0, r=10.0), low_percentile=5.0, high_percentile=95.0)
    assert measurement["measured_sphere_log2"] == pytest.approx(np.log2(0.25), abs=0.12)
    assert measurement["sampling_method"] == "refined_interior_mask"
    assert measurement["mask_fraction"] < 0.9


def test_measure_frame_color_and_exposure_uses_refined_sphere_sampling_for_gray_sphere() -> None:
    image = np.full((3, 48, 48), 0.05, dtype=np.float32)
    yy, xx = np.meshgrid(np.arange(48, dtype=np.float32), np.arange(48, dtype=np.float32), indexing="ij")
    cx = cy = 24.0
    outer = ((xx - cx) ** 2 + (yy - cy) ** 2) <= 11.0 ** 2
    inner = ((xx - cx) ** 2 + (yy - cy) ** 2) <= 7.0 ** 2
    edge_ring = outer & ~inner
    image[:, 12:36, 12:36] = 0.2
    image[:, outer] = 0.25
    image[0, edge_ring] = 0.70
    image[1, edge_ring] = 0.18
    image[2, edge_ring] = 0.12
    measurement = measure_frame_color_and_exposure(
        image,
        mode="scene",
        lut=None,
        calibration_roi={"x": 12 / 48, "y": 12 / 48, "w": 24 / 48, "h": 24 / 48},
        target_type="gray_sphere",
    )
    assert measurement["measured_log2_luminance_raw"] == pytest.approx(np.log2(0.25), abs=0.12)
    comparison = measurement["sphere_sampling_comparison"]
    assert comparison["legacy"]["raw_log2"] > comparison["refined"]["raw_log2"]
    assert measurement["calibration_measurement_mode"] == "gray_sphere_three_zone_profile"


def test_sphere_calibration_grouping_uses_group_key(tmp_path: Path) -> None:
    clip_a = tmp_path / "G007_D060_0324M6_001.R3D"
    clip_b = tmp_path / "G008_D061_0324M6_002.R3D"
    clip_a.write_bytes(b"")
    clip_b.write_bytes(b"")
    roi_path = tmp_path / "sphere-roi.json"
    roi_path.write_text(json.dumps({"G007_D060": {"cx": 48, "cy": 27, "r": 12}, "G008_D061": {"cx": 48, "cy": 27, "r": 12}}), encoding="utf-8")
    calibrate_sphere_path(str(tmp_path), target_log2_luminance=-2.0, out_dir=str(tmp_path / "sphere-cal"), roi_file=str(roi_path), backend="mock")
    calibration = load_calibration(str(tmp_path / "sphere-cal" / "sphere_calibration.json"))
    assert {entry.group_key for entry in calibration.cameras} == {"G007_D060", "G008_D061"}


def test_load_card_roi_file_reads_rectangles() -> None:
    rois = load_card_roi_file.__call__  # smoke that import exists
    assert callable(rois)


def test_measure_card_roi_applies_rectangular_masking_and_trimming() -> None:
    image = np.full((3, 12, 12), 0.05, dtype=np.float32)
    image[:, 2:8, 3:9] = 0.2
    image[:, 3, 4] = 1.0
    image[:, 4, 5] = 0.001
    measurement = measure_card_from_roi(image, GrayCardROI(x=3, y=2, width=6, height=6), low_percentile=10.0, high_percentile=90.0)
    assert measurement["measured_card_log2"] == pytest.approx(np.log2(0.2), abs=0.2)


def test_center_crop_sampling_extraction() -> None:
    image = np.zeros((3, 100, 200), dtype=np.float32)
    roi = center_crop_roi(image, width_ratio=0.3, height_ratio=0.3)
    assert roi.width == pytest.approx(60.0)
    assert roi.height == pytest.approx(30.0)
    assert roi.x == pytest.approx(70.0)
    assert roi.y == pytest.approx(35.0)


def test_neutral_gain_solve_from_synthetic_rgb_values() -> None:
    gains = solve_neutral_gains(0.25, 0.5, 1.0)
    assert gains["r"] == pytest.approx(2.0)
    assert gains["g"] == pytest.approx(1.0)
    assert gains["b"] == pytest.approx(0.5)


def test_measure_color_region_reports_neutral_gains() -> None:
    image = np.zeros((3, 20, 20), dtype=np.float32)
    image[0, 5:15, 5:15] = 0.25
    image[1, 5:15, 5:15] = 0.5
    image[2, 5:15, 5:15] = 1.0
    measurement = measure_color_region(
        image,
        SamplingRegion(sampling_mode="detected_roi", roi={"x": 5.0, "y": 5.0, "width": 10.0, "height": 10.0}),
    )
    assert measurement["rgb_neutral_gains"]["r"] == pytest.approx(2.0)
    assert measurement["rgb_neutral_gains"]["g"] == pytest.approx(1.0)
    assert measurement["rgb_neutral_gains"]["b"] == pytest.approx(0.5)


def test_gray_card_target_log2_mode_writes_calibration(tmp_path: Path) -> None:
    clip_path = tmp_path / "G007_D060_0324M6_001.R3D"
    clip_path.write_bytes(b"")
    roi_path = tmp_path / "card-roi.json"
    roi_path.write_text(json.dumps({"G007_D060": {"x": 20, "y": 12, "width": 24, "height": 18}}), encoding="utf-8")
    payload = calibrate_card_path(str(tmp_path), out_dir=str(tmp_path / "card-cal"), roi_file=str(roi_path), target_log2_luminance=-2.0, backend="mock")
    calibration = load_calibration(str(tmp_path / "card-cal" / "gray_card_calibration.json"))
    assert payload["calibration_mode"] == "gray_card"
    assert calibration.cameras[0].clip_id == "G007_D060_0324M6_001"
    assert calibration.cameras[0].group_key == "G007_D060"


def test_exposure_calibration_json_writing_and_sampling_mode(tmp_path: Path) -> None:
    clip_path = tmp_path / "G007_D060_0324M6_001.R3D"
    clip_path.write_bytes(b"")
    payload = calibrate_exposure_path(
        str(tmp_path),
        out_dir=str(tmp_path / "exposure-cal"),
        source="center_crop",
        sampling_mode="center_crop",
        target_log2_luminance=-2.0,
        backend="mock",
    )
    calibration = load_exposure_calibration(str(tmp_path / "exposure-cal" / "exposure_calibration.json"))
    assert payload["calibration_type"] == "exposure"
    assert calibration.cameras[0].sampling_mode == "center_crop"
    assert calibration.cameras[0].sampling_region.center_crop is not None


def test_color_calibration_json_writing(tmp_path: Path) -> None:
    clip_path = tmp_path / "G007_D060_0324M6_001.R3D"
    clip_path.write_bytes(b"")
    payload = calibrate_color_path(
        str(tmp_path),
        out_dir=str(tmp_path / "color-cal"),
        sampling_mode="center_crop",
        backend="mock",
    )
    calibration = load_color_calibration(str(tmp_path / "color-cal" / "color_calibration.json"))
    assert payload["calibration_type"] == "color"
    assert calibration.cameras[0].rgb_neutral_gains["g"] == pytest.approx(1.0)
    assert calibration.cameras[0].sampling_mode == "center_crop"


def test_detect_gray_card_roi_on_mock_like_frame() -> None:
    image = np.full((3, 54, 96), 0.06, dtype=np.float32)
    image[:, 16:34, 28:68] = 0.18
    detected = detect_gray_card_roi(image)
    assert isinstance(detected["roi"], GrayCardROI)


def test_gray_card_auto_detection_writes_detection_and_calibration_json(tmp_path: Path) -> None:
    clip_path = tmp_path / "G007_D060_0324M6_001.R3D"
    clip_path.write_bytes(b"")
    payload = calibrate_card_path(str(tmp_path), out_dir=str(tmp_path / "card-cal"), target_log2_luminance=-2.0, backend="mock")
    detection_payload = json.loads((tmp_path / "card-cal" / "gray_card_detection.json").read_text(encoding="utf-8"))
    assert Path(payload["detection_path"]).exists()
    assert detection_payload["cameras"][0]["group_key"] == "G007_D060"


def test_gray_card_manual_roi_file_overrides_auto_detection(tmp_path: Path) -> None:
    clip_path = tmp_path / "G007_D060_0324M6_001.R3D"
    clip_path.write_bytes(b"")
    roi_path = tmp_path / "card-roi.json"
    roi_path.write_text(json.dumps({"G007_D060": {"x": 5, "y": 6, "width": 7, "height": 8}}), encoding="utf-8")
    calibrate_card_path(str(tmp_path), out_dir=str(tmp_path / "card-cal"), roi_file=str(roi_path), target_log2_luminance=-2.0, backend="mock")
    detection_payload = json.loads((tmp_path / "card-cal" / "gray_card_detection.json").read_text(encoding="utf-8"))
    assert detection_payload["detection_mode"] == "manual_override"


def test_gray_card_reference_camera_mode_uses_group_key(tmp_path: Path) -> None:
    clip_a = tmp_path / "G007_D060_0324M6_001.R3D"
    clip_b = tmp_path / "G008_D061_0324M6_002.R3D"
    clip_a.write_bytes(b"")
    clip_b.write_bytes(b"")
    calibrate_card_path(str(tmp_path), out_dir=str(tmp_path / "card-cal"), reference_camera="G007_D060", backend="mock")
    calibration = load_calibration(str(tmp_path / "card-cal" / "gray_card_calibration.json"))
    by_group = {entry.group_key: entry for entry in calibration.cameras}
    assert by_group["G007_D060"].camera_baseline_stops == pytest.approx(0.0)


def test_analyze_uses_gray_card_calibration(tmp_path: Path) -> None:
    clip_path = tmp_path / "G007_D060_0324M6_001.R3D"
    clip_path.write_bytes(b"")
    calibration_path = tmp_path / "gray_card_calibration.json"
    calibration_path.write_text(
        json.dumps(
            {
                "target_log2_luminance": -2.0,
                "calibration_mode": "gray_card",
                "object_type": "gray_card",
                "roi_file": None,
                "reference_camera": "G007_D060",
                "cameras": [
                    {
                        "clip_id": "G007_D060_0324M6_001",
                        "group_key": "G007_D060",
                        "roi": {"x": 20, "y": 12, "width": 24, "height": 18},
                        "measured_card_log2": -1.2,
                        "camera_baseline_stops": 0.4,
                        "confidence": 0.9,
                        "source_path": str(clip_path),
                        "frame_index": 0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    summary = analyze_path(str(tmp_path), out_dir=str(tmp_path / "out"), mode="scene", backend="mock", lut_override=None, calibration_path=str(calibration_path), sample_count=4, sampling_strategy="uniform")
    assert summary["calibration"]["group_baselines"]["G007_D060"] == pytest.approx(0.4)


def test_analyze_uses_exposure_calibration_only(tmp_path: Path) -> None:
    clip_path = tmp_path / "G007_D060_0324M6_001.R3D"
    clip_path.write_bytes(b"")
    exposure_payload = {
        "calibration_type": "exposure",
        "calibration_mode": "exposure",
        "object_type": "center_crop",
        "target_log2_luminance": -2.0,
        "reference_camera": None,
        "roi_file": None,
        "cameras": [
            {
                "clip_id": "G007_D060_0324M6_001",
                "group_key": derive_array_group_key(str(tmp_path)),
                "source_path": str(clip_path),
                "sampling_mode": "center_crop",
                "sampling_region": {"sampling_mode": "center_crop", "center_crop": {"width_ratio": 0.3, "height_ratio": 0.3}, "roi": None, "detection_confidence": None, "fallback_used": False},
                "measured_log2_luminance": -1.0,
                "camera_baseline_stops": 0.4,
                "confidence": 0.9,
                "calibration_source": "center_crop",
                "frame_index": 0,
            }
        ],
    }
    exposure_path = tmp_path / "exposure_calibration.json"
    exposure_path.write_text(json.dumps(exposure_payload), encoding="utf-8")
    summary = analyze_path(str(tmp_path), out_dir=str(tmp_path / "out"), mode="scene", backend="mock", lut_override=None, calibration_path=None, exposure_calibration_path=str(exposure_path), sample_count=4, sampling_strategy="uniform")
    assert summary["exposure_calibration"]["group_baselines"][derive_array_group_key(str(tmp_path))] == pytest.approx(0.4)
    assert summary["color_calibration"] is None


def test_analyze_uses_color_calibration_only(tmp_path: Path) -> None:
    clip_path = tmp_path / "G007_D060_0324M6_001.R3D"
    clip_path.write_bytes(b"")
    color_payload = {
        "calibration_type": "color",
        "calibration_mode": "color",
        "object_type": "neutral_patch",
        "roi_file": None,
        "cameras": [
            {
                "clip_id": "G007_D060_0324M6_001",
                "group_key": derive_array_group_key(str(tmp_path)),
                "source_path": str(clip_path),
                "sampling_mode": "center_crop",
                "sampling_region": {"sampling_mode": "center_crop", "center_crop": {"width_ratio": 0.3, "height_ratio": 0.3}, "roi": None, "detection_confidence": None, "fallback_used": False},
                "measured_channel_medians": {"r": 0.3, "g": 0.4, "b": 0.5},
                "rgb_neutral_gains": {"r": 1.3333333, "g": 1.0, "b": 0.8},
                "confidence": 0.9,
                "calibration_source": "neutral_region",
                "frame_index": 0,
            }
        ],
    }
    color_path = tmp_path / "color_calibration.json"
    color_path.write_text(json.dumps(color_payload), encoding="utf-8")
    summary = analyze_path(str(tmp_path), out_dir=str(tmp_path / "out"), mode="scene", backend="mock", lut_override=None, calibration_path=None, color_calibration_path=str(color_path), sample_count=4, sampling_strategy="uniform")
    assert summary["color_calibration"]["group_gains"][derive_array_group_key(str(tmp_path))]["g"] == pytest.approx(1.0)
    assert summary["exposure_calibration"] is None
    assert summary["clips"][0]["pending_color_gains"]["b"] == pytest.approx(0.8)


def test_analyze_uses_both_exposure_and_color_calibration(tmp_path: Path) -> None:
    clip_path = tmp_path / "G007_D060_0324M6_001.R3D"
    clip_path.write_bytes(b"")
    exposure_path = tmp_path / "exposure_calibration.json"
    color_path = tmp_path / "color_calibration.json"
    exposure_path.write_text(
        json.dumps(
            {
                "calibration_type": "exposure",
                "calibration_mode": "exposure",
                "object_type": "gray_card",
                "target_log2_luminance": -2.0,
                "reference_camera": None,
                "roi_file": None,
                "cameras": [
                    {
                        "clip_id": "G007_D060_0324M6_001",
                        "group_key": derive_array_group_key(str(tmp_path)),
                        "source_path": str(clip_path),
                        "sampling_mode": "detected_roi",
                        "sampling_region": {"sampling_mode": "detected_roi", "center_crop": None, "roi": {"x": 20, "y": 12, "width": 24, "height": 18}, "detection_confidence": 0.9, "fallback_used": False},
                        "measured_log2_luminance": -1.2,
                        "camera_baseline_stops": 0.4,
                        "confidence": 0.9,
                        "calibration_source": "gray_card",
                        "frame_index": 0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    color_path.write_text(
        json.dumps(
            {
                "calibration_type": "color",
                "calibration_mode": "color",
                "object_type": "neutral_patch",
                "roi_file": None,
                "cameras": [
                    {
                        "clip_id": "G007_D060_0324M6_001",
                        "group_key": derive_array_group_key(str(tmp_path)),
                        "source_path": str(clip_path),
                        "sampling_mode": "detected_roi",
                        "sampling_region": {"sampling_mode": "detected_roi", "center_crop": None, "roi": {"x": 20, "y": 12, "width": 24, "height": 18}, "detection_confidence": 0.9, "fallback_used": False},
                        "measured_channel_medians": {"r": 0.3, "g": 0.4, "b": 0.5},
                        "rgb_neutral_gains": {"r": 1.3333333, "g": 1.0, "b": 0.8},
                        "confidence": 0.9,
                        "calibration_source": "neutral_region",
                        "frame_index": 0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    summary = analyze_path(
        str(tmp_path),
        out_dir=str(tmp_path / "out"),
        mode="scene",
        backend="mock",
        lut_override=None,
        calibration_path=None,
        exposure_calibration_path=str(exposure_path),
        color_calibration_path=str(color_path),
        sample_count=4,
        sampling_strategy="uniform",
    )
    assert summary["exposure_calibration"]["group_baselines"][derive_array_group_key(str(tmp_path))] == pytest.approx(0.4)
    assert summary["color_calibration"]["group_gains"][derive_array_group_key(str(tmp_path))]["r"] == pytest.approx(1.3333333)
    assert summary["clips"][0]["pending_color_gains"]["r"] == pytest.approx(1.3333333)


def test_array_calibration_json_written_and_shared_group(tmp_path: Path) -> None:
    clip_a = tmp_path / "G007_B057_0324YT_001.R3D"
    clip_b = tmp_path / "G007_C057_0324YT_001.R3D"
    clip_a.write_bytes(b"")
    clip_b.write_bytes(b"")
    summary = analyze_path(str(tmp_path), out_dir=str(tmp_path / "out"), mode="scene", backend="mock", lut_override=None, calibration_path=None, sample_count=4, sampling_strategy="uniform")
    calibration = json.loads((tmp_path / "out" / "array_calibration.json").read_text(encoding="utf-8"))
    expected_group = derive_array_group_key(str(tmp_path))
    assert calibration["schema"] == "r3dmatch_array_calibration_v1"
    assert calibration["group_key"] == expected_group
    assert {clip["group_key"] for clip in summary["clips"]} == {expected_group}


def test_build_sidecar_payload_uses_exact_clip_id() -> None:
    from r3dmatch.models import ClipMetadata, ClipResult, MonitoringContext, SamplePlan
    result = ClipResult(
        clip_id="G007_D060_0324M6_001",
        group_key="G007_D060",
        source_path="/tmp/G007_D060_0324M6_001.R3D",
        backend="mock",
        clip_statistic_log2=-2.0,
        group_key_statistic_log2=-2.0,
        global_reference_log2=-2.47,
        raw_offset_stops=0.4,
        camera_baseline_stops=0.2,
        clip_trim_stops=0.2,
        final_offset_stops=0.4,
        confidence=0.9,
        sample_plan=SamplePlan(strategy="uniform", sample_count=1, start_frame=0, frame_step=1, max_frames=1),
        monitoring=MonitoringContext(mode="scene", ipp2_color_space=None, ipp2_gamma_curve=None, active_lut_path=None, lut_override_path=None, resolved_lut_path=None),
        clip_metadata=ClipMetadata(clip_id="G007_D060_0324M6_001", group_key="G007_D060", original_filename="G007_D060_0324M6_001.R3D", source_path="/tmp/G007_D060_0324M6_001.R3D", fps=24.0, width=96, height=54, total_frames=1),
        frame_stats=[],
    )
    payload = build_sidecar_payload(result)
    assert payload["rmd_name"] == "G007_D060_0324M6_001.RMD"
    assert payload["sidecar_filename"] == "G007_D060_0324M6_001.sidecar.json"


def test_analyze_uses_sphere_calibration_group_key(tmp_path: Path) -> None:
    clip_path = tmp_path / "G007_D060_0324M6_001.R3D"
    clip_path.write_bytes(b"")
    calibration_path = tmp_path / "sphere_calibration.json"
    calibration_path.write_text(
        json.dumps(
            {
                "target_log2_luminance": -2.0,
                "object_type": "gray_sphere",
                "calibration_mode": "sphere",
                "roi_file": None,
                "cameras": [
                    {
                        "clip_id": "G007_D060_0324M6_001",
                        "group_key": "G007_D060",
                        "roi": {"cx": 48, "cy": 27, "r": 12},
                        "measured_sphere_log2": -1.0,
                        "camera_baseline_stops": 0.5,
                        "confidence": 0.9,
                        "source_path": str(clip_path),
                        "frame_index": 0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    summary = analyze_path(str(tmp_path), out_dir=str(tmp_path / "out"), mode="scene", backend="mock", lut_override=None, calibration_path=str(calibration_path), sample_count=4, sampling_strategy="uniform")
    assert summary["calibration"]["group_baselines"]["G007_D060"] == pytest.approx(0.5)


def test_cli_analyze_and_transcode(tmp_path: Path) -> None:
    clip_path = tmp_path / "G007_D060_0324M6_001.R3D"
    clip_path.write_bytes(b"")
    analysis_dir = tmp_path / "analysis"
    transcode_dir = tmp_path / "transcode"
    analyze_result = runner.invoke(app, ["analyze", str(clip_path), "--out", str(analysis_dir), "--backend", "mock", "--mode", "view"])
    assert analyze_result.exit_code == 0
    auto_card_calibration_result = runner.invoke(app, ["calibrate-card", str(clip_path), "--target-log2", "-2.0", "--out", str(tmp_path / "card-calibration-auto"), "--backend", "mock"])
    assert auto_card_calibration_result.exit_code == 0
    transcode_result = runner.invoke(app, ["transcode", str(clip_path), "--analysis-dir", str(analysis_dir), "--use-generated-sidecar", "--out", str(transcode_dir), "--redline-executable", "echo"])
    assert transcode_result.exit_code == 0


def test_analyze_path_filters_selected_clip_ids_and_groups(tmp_path: Path) -> None:
    for name in [
        "G007_A063_0325EV_001.R3D",
        "G007_B063_0325EV_001.R3D",
        "G007_A064_0325EV_001.R3D",
    ]:
        (tmp_path / name).write_bytes(b"")
    by_clip_id = analyze_path(
        str(tmp_path),
        out_dir=str(tmp_path / "analysis-clip"),
        mode="scene",
        backend="mock",
        lut_override=None,
        calibration_path=None,
        sample_count=4,
        sampling_strategy="uniform",
        selected_clip_ids=["G007_A063_0325EV_001"],
    )
    assert by_clip_id["clip_count"] == 1
    assert by_clip_id["selected_clip_ids"] == ["G007_A063_0325EV_001"]
    by_group = analyze_path(
        str(tmp_path),
        out_dir=str(tmp_path / "analysis-group"),
        mode="scene",
        backend="mock",
        lut_override=None,
        calibration_path=None,
        sample_count=4,
        sampling_strategy="uniform",
        selected_clip_groups=["063"],
    )
    assert by_group["clip_count"] == 2
    assert by_group["selected_clip_groups"] == ["063"]
    analysis_files = sorted((tmp_path / "analysis-group" / "analysis").glob("*.analysis.json"))
    assert [path.stem.replace(".analysis", "") for path in analysis_files] == [
        "G007_A063_0325EV_001",
        "G007_B063_0325EV_001",
    ]


def test_cli_write_rmd(tmp_path: Path) -> None:
    clip_path = tmp_path / "G007_D060_0324M6_001.R3D"
    clip_path.write_bytes(b"")
    analysis_dir = tmp_path / "analysis"
    runner.invoke(app, ["analyze", str(clip_path), "--out", str(analysis_dir), "--backend", "mock", "--mode", "scene"], catch_exceptions=False)
    write_result = runner.invoke(app, ["write-rmd", str(clip_path), "--analysis-dir", str(analysis_dir)])
    assert write_result.exit_code == 0
    assert (analysis_dir / "rmd" / "G007_D060_0324M6_001.RMD").exists()


def test_cli_validate_pipeline_and_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_redline(monkeypatch)
    clip_path = tmp_path / "G007_D060_0324M6_001.R3D"
    clip_path.write_bytes(b"")
    analysis_dir = tmp_path / "analysis"
    analyze_path(
        str(clip_path),
        out_dir=str(analysis_dir),
        mode="scene",
        backend="mock",
        lut_override=None,
        calibration_path=None,
        sample_count=4,
        sampling_strategy="uniform",
        calibration_roi={"x": 0.2, "y": 0.2, "w": 0.4, "h": 0.4},
    )
    validate_result = runner.invoke(app, ["validate-pipeline", str(clip_path), "--analysis-dir", str(analysis_dir), "--out", str(tmp_path / "validation")])
    report_result = runner.invoke(
        app,
        ["report-contact-sheet", str(analysis_dir), "--out", str(tmp_path / "report"), "--target-type", "gray_sphere"],
    )
    assert validate_result.exit_code == 0
    assert report_result.exit_code == 0


def test_cli_review_and_approve_workflow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_redline(monkeypatch)
    clip_path = tmp_path / "G007_D060_0324M6_001.R3D"
    clip_path.write_bytes(b"")
    review_dir = tmp_path / "review"
    review_result = runner.invoke(
        app,
        [
            "review-calibration",
            str(clip_path),
            "--out",
            str(review_dir),
            "--target-type",
            "gray_sphere",
            "--processing-mode",
            "both",
            "--backend",
            "mock",
            "--roi-x",
            "0.25",
            "--roi-y",
            "0.25",
            "--roi-w",
            "0.5",
            "--roi-h",
            "0.5",
            "--target-strategy",
            "median",
            "--target-strategy",
            "optimal-exposure",
        ],
    )
    assert review_result.exit_code == 0
    assert (review_dir / "report" / "contact_sheet.html").exists()
    assert (review_dir / "report" / "preview_contact_sheet.pdf").exists()
    assert not (review_dir / "review_rmd" / "strategies").exists()
    review_manifest = json.loads((review_dir / "report" / "review_manifest.json").read_text(encoding="utf-8"))
    assert review_manifest["calibration_roi"] == {"x": 0.25, "y": 0.25, "w": 0.5, "h": 0.5}
    assert review_manifest["target_strategies"] == ["median", "optimal_exposure"]
    review_package = json.loads((review_dir / "report" / "review_package.json").read_text(encoding="utf-8"))
    assert review_package["artifact_mode"] == "production"
    assert review_package["temporary_rmd_manifest"]["skipped"] is True
    assert review_package["temporary_rmd_manifest"]["strategy_review_rmd_root"] == ""
    assert review_package["rcx_compare_note"] is None
    assert review_package["recommended_strategy"]["strategy_key"] == "median"
    assert review_package["hero_recommendation"]["candidate_clip_id"] is None
    assert review_package["run_assessment"]["status"] in {"READY", "READY_WITH_WARNINGS", "REVIEW_REQUIRED", "DO_NOT_PUSH"}
    assert review_package["gray_target_consistency"]["dominant_target_class"] in {"sphere", "gray_card", "unresolved"}
    approve_result = runner.invoke(app, ["approve-master-rmd", str(review_dir), "--target-strategy", "optimal-exposure"])
    assert approve_result.exit_code == 0
    assert (review_dir / "approval" / "MasterRMD" / "G007_D060.RMD").exists()
    assert (review_dir / "approval" / "batch" / "manifest.json").exists()
    assert (review_dir / "approval" / "approval_manifest.json").exists()
    assert (review_dir / "approval" / "calibration_report.pdf").exists()
    approval_manifest = json.loads((review_dir / "approval" / "approval_manifest.json").read_text(encoding="utf-8"))
    assert approval_manifest["selected_target_strategy"] == "optimal_exposure"
    assert approval_manifest["master_rmd_folder_name"] == "MasterRMD"


def test_build_review_package_debug_keeps_clip_level_review_rmd_and_rcx_compare(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_redline(monkeypatch)
    clip_path = tmp_path / "G007_D060_0324M6_001.R3D"
    clip_path.write_bytes(b"")
    package = review_calibration(
        str(tmp_path),
        out_dir=str(tmp_path / "runs"),
        target_type="gray_sphere",
        processing_mode="both",
        mode="scene",
        backend="mock",
        lut_override=None,
        calibration_path=None,
        exposure_calibration_path=None,
        color_calibration_path=None,
        calibration_mode=None,
        sample_count=4,
        sampling_strategy="uniform",
        calibration_roi=None,
        selected_clip_ids=None,
        selected_clip_groups=None,
        clip_subset_file=None,
        run_label="debug_review",
        matching_domain="perceptual",
        artifact_mode="debug",
        target_strategies=["median"],
        reference_clip_id=None,
        hero_clip_id=None,
    )
    assert package["artifact_mode"] == "debug"
    assert Path(str(package["rcx_compare_note"])).exists()
    assert (tmp_path / "runs" / "debug_review" / "review_rmd" / "G007_D060_0324M6_001.RMD").exists()
    assert package["temporary_rmd_manifest"]["clip_count"] == 1


def test_review_calibration_subset_run_label_and_matching_domain(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_redline(monkeypatch)
    for name in [
        "G007_A063_0325EV_001.R3D",
        "G007_B063_0325EV_001.R3D",
        "G007_A064_0325EV_001.R3D",
        "G007_B064_0325EV_001.R3D",
    ]:
        (tmp_path / name).write_bytes(b"")
    parent_out = tmp_path / "runs"
    payload = review_calibration(
        str(tmp_path),
        out_dir=str(parent_out),
        run_label="clip63_even",
        target_type="gray_sphere",
        processing_mode="both",
        mode="scene",
        matching_domain="perceptual",
        backend="mock",
        lut_override=None,
        calibration_path=None,
        exposure_calibration_path=None,
        color_calibration_path=None,
        calibration_mode=None,
        sample_count=4,
        sampling_strategy="uniform",
        calibration_roi={"x": 0.25, "y": 0.25, "w": 0.5, "h": 0.5},
        selected_clip_ids=None,
        selected_clip_groups=["063"],
        clip_subset_file=None,
        target_strategies=["median", "hero-camera"],
        reference_clip_id=None,
        hero_clip_id="G007_A063_0325EV_001",
    )
    run_root = parent_out / "clip63_even"
    assert payload["analysis_dir"] == str(run_root)
    assert payload["run_label"] == "clip63_even"
    assert payload["matching_domain"] == "perceptual"
    report_json = json.loads((run_root / "report" / "contact_sheet.json").read_text(encoding="utf-8"))
    assert report_json["run_label"] == "clip63_even"
    assert report_json["matching_domain"] == "perceptual"
    assert report_json["matching_domain_label"] == "Perceptual (IPP2 / BT.709 / BT.1886)"
    assert report_json["measurement_preview_transform"] == "REDLine IPP2 / BT.709 / BT.1886 / Medium / Medium"
    assert report_json["selected_clip_groups"] == ["063"]
    assert report_json["clip_count"] == 2
    assert {clip["clip_id"] for clip in report_json["shared_originals"]} == {
        "G007_A063_0325EV_001",
        "G007_B063_0325EV_001",
    }
    hero_strategy = next(item for item in report_json["strategies"] if item["strategy_key"] == "hero_camera")
    hero_clip = next(item for item in hero_strategy["clips"] if item["clip_id"] == "G007_A063_0325EV_001")
    assert hero_clip["metrics"]["exposure"]["final_offset_stops"] == pytest.approx(0.0)


def test_build_redline_command_includes_sidecar(tmp_path: Path) -> None:
    clip_path = tmp_path / "G007_D060_0324M6_001.R3D"
    clip_path.write_bytes(b"")
    sidecar_path = tmp_path / "clip.sidecar.json"
    sidecar_path.write_text("{}", encoding="utf-8")
    command = build_redline_command(str(clip_path), render_dir=str(tmp_path / "renders"), sidecar_path=str(sidecar_path), redline_executable="REDLine", output_ext="mov")
    assert command[0] == "REDLine"


def test_write_rmds_from_analysis_creates_exact_clip_rmd(tmp_path: Path) -> None:
    clip_path = tmp_path / "G007_D060_0324M6_001.R3D"
    clip_path.write_bytes(b"")
    analyze_path(str(tmp_path), out_dir=str(tmp_path / "analysis-out"), mode="scene", backend="mock", lut_override=None, calibration_path=None, sample_count=4, sampling_strategy="uniform")
    manifest = write_rmds_from_analysis(str(tmp_path / "analysis-out"))
    assert manifest["clip_count"] == 1
    assert (tmp_path / "analysis-out" / "rmd" / "G007_D060_0324M6_001.RMD").exists()


def test_redline_command_generation_variants(tmp_path: Path) -> None:
    clip_path = tmp_path / "G007_D060_0324M6_001.R3D"
    clip_path.write_bytes(b"")
    sidecar_payload = {
        "calibration_state": {
            "exposure_calibration_loaded": True,
            "color_calibration_loaded": True,
        }
    }
    variants = build_redline_command_variants(
        str(clip_path),
        render_dir=str(tmp_path / "renders"),
        sidecar_path=str(tmp_path / "G007_D060_0324M6_001.sidecar.json"),
        redline_executable="REDLine",
        output_ext="mov",
        sidecar_payload=sidecar_payload,
    )
    assert [variant["variant"] for variant in variants] == ["original", "exposure", "color", "both"]


def test_transcode_plan_with_generated_rmds(tmp_path: Path) -> None:
    clip_path = tmp_path / "G007_D060_0324M6_001.R3D"
    clip_path.write_bytes(b"")
    analyze_path(str(tmp_path), out_dir=str(tmp_path / "analysis-out"), mode="scene", backend="mock", lut_override=None, calibration_path=None, sample_count=4, sampling_strategy="uniform")
    payload = write_transcode_plan(
        str(clip_path),
        out_dir=str(tmp_path / "transcode"),
        analysis_dir=str(tmp_path / "analysis-out"),
        use_generated_sidecar=False,
        use_generated_rmd=True,
        redline_executable="REDLine",
        output_ext="mov",
        execute=False,
    )
    clip_payload = payload["clips"][0]
    assert clip_payload["rmd_name_matches_clip_id"] is True
    assert clip_payload["rmd_path"].endswith("G007_D060_0324M6_001.RMD")
    assert any(variant["uses_rmd"] for variant in clip_payload["variants"] if variant["variant"] != "original")


def test_transcode_execute_prints_progress_bar(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    clip_path = tmp_path / "G007_D060_0324M6_001.R3D"
    clip_path.write_bytes(b"")

    def fake_run(command, capture_output, text, check):  # type: ignore[no-untyped-def]
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("r3dmatch.transcode.subprocess.run", fake_run)
    monotonic_values = iter([100.0, 142.0])
    monkeypatch.setattr("r3dmatch.transcode.time.monotonic", lambda: next(monotonic_values))
    payload = write_transcode_plan(
        str(clip_path),
        out_dir=str(tmp_path / "transcode"),
        analysis_dir=None,
        use_generated_sidecar=False,
        use_generated_rmd=False,
        redline_executable="REDLine",
        output_ext="mov",
        execute=True,
    )
    captured = capsys.readouterr()
    assert payload["clips"][0]["variants"][0]["executed"] is True
    assert "[transcode]" in captured.err
    assert "1/1" in captured.err
    assert "00:42 elapsed" in captured.err
    assert "ETA 00:00" in captured.err
    assert "G007_D060_0324M6_001 [orig]" in captured.err


def test_transcode_execute_prints_failure_message(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    clip_path = tmp_path / "G007_D060_0324M6_001.R3D"
    clip_path.write_bytes(b"")

    def fake_run(command, capture_output, text, check):  # type: ignore[no-untyped-def]
        return types.SimpleNamespace(returncode=7, stdout="", stderr="boom")

    monkeypatch.setattr("r3dmatch.transcode.subprocess.run", fake_run)
    monotonic_values = iter([100.0, 112.0])
    monkeypatch.setattr("r3dmatch.transcode.time.monotonic", lambda: next(monotonic_values))
    payload = write_transcode_plan(
        str(clip_path),
        out_dir=str(tmp_path / "transcode"),
        analysis_dir=None,
        use_generated_sidecar=False,
        use_generated_rmd=False,
        redline_executable="REDLine",
        output_ext="mov",
        execute=True,
    )
    captured = capsys.readouterr()
    assert payload["clips"][0]["variants"][0]["returncode"] == 7
    assert "FAILED G007_D060_0324M6_001 [orig] (returncode=7)" in captured.err


def test_transcode_requires_analysis_dir_for_generated_sidecars(tmp_path: Path) -> None:
    clip_path = tmp_path / "G007_D060_0324M6_001.R3D"
    clip_path.write_bytes(b"")
    with pytest.raises(ValueError):
        write_transcode_plan(str(clip_path), out_dir=str(tmp_path / "transcode"), analysis_dir=None, use_generated_sidecar=True, use_generated_rmd=False, redline_executable="REDLine", output_ext="mov", execute=False)


def test_transcode_requires_analysis_dir_for_generated_rmds(tmp_path: Path) -> None:
    clip_path = tmp_path / "G007_D060_0324M6_001.R3D"
    clip_path.write_bytes(b"")
    with pytest.raises(ValueError):
        write_transcode_plan(str(clip_path), out_dir=str(tmp_path / "transcode"), analysis_dir=None, use_generated_sidecar=False, use_generated_rmd=True, redline_executable="REDLine", output_ext="mov", execute=False)


def test_validate_pipeline_behavior(tmp_path: Path) -> None:
    clip_path = tmp_path / "G007_D060_0324M6_001.R3D"
    clip_path.write_bytes(b"")
    analyze_path(str(tmp_path), out_dir=str(tmp_path / "analysis-out"), mode="scene", backend="mock", lut_override=None, calibration_path=None, sample_count=4, sampling_strategy="uniform")
    payload = validate_pipeline(
        str(tmp_path),
        analysis_dir=str(tmp_path / "analysis-out"),
        exposure_calibration_path=None,
        color_calibration_path=None,
        out_dir=str(tmp_path / "validation"),
        redline_executable="REDLine",
        output_ext="mov",
    )
    assert payload["clip_count"] == 1
    assert payload["clips"][0]["sidecar_name_matches_clip_id"] is True
    assert payload["clips"][0]["rmd_name_matches_clip_id"] is True
    assert all(record["command_uses_exact_rmd"] for record in payload["clips"][0]["redline_variant_records"] if record["uses_rmd"])


def test_report_contact_sheet_scaffold(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_redline(monkeypatch)
    clip_path = tmp_path / "G007_D060_0324M6_001.R3D"
    clip_path.write_bytes(b"")
    analyze_path(
        str(tmp_path),
        out_dir=str(tmp_path / "analysis-out"),
        mode="scene",
        backend="mock",
        lut_override=None,
        calibration_path=None,
        sample_count=4,
        sampling_strategy="uniform",
        calibration_roi={"x": 0.2, "y": 0.2, "w": 0.4, "h": 0.4},
    )
    (tmp_path / "show.cube").write_text("LUT_3D_SIZE 2\n0 0 0\n0 0 1\n0 1 0\n0 1 1\n1 0 0\n1 0 1\n1 1 0\n1 1 1\n", encoding="utf-8")
    payload = build_contact_sheet_report(
        str(tmp_path / "analysis-out"),
        out_dir=str(tmp_path / "report"),
        target_type="gray_sphere",
        processing_mode="both",
        target_strategies=["median", "optimal-exposure"],
        preview_mode="monitoring",
        preview_lut=str(tmp_path / "show.cube"),
    )
    assert Path(payload["report_json"]).exists()
    assert Path(payload["report_html"]).exists()
    assert Path(payload["preview_report_pdf"]).exists()
    assert Path(payload["previews_dir"]).exists()
    assert payload["recommended_strategy"]["strategy_key"] == "median"
    assert payload["hero_recommendation"]["candidate_clip_id"] is None
    assert payload["run_assessment"]["status"] in {"READY", "READY_WITH_WARNINGS", "REVIEW_REQUIRED", "DO_NOT_PUSH"}
    assert payload["gray_target_consistency"]["dominant_target_class"] in {"sphere", "gray_card", "unresolved"}
    report_json = json.loads(Path(payload["report_json"]).read_text(encoding="utf-8"))
    assert report_json["clips"][0]["clip_id"] == "G007_D060_0324M6_001"
    assert report_json["target_type"] == "gray_sphere"
    assert report_json["preview_transform"]
    assert report_json["exposure_measurement_domain"] == "perceptual"
    assert report_json["preview_mode"] == "monitoring"
    assert report_json["preview_settings"]["lut_path"].endswith("show.cube")
    assert report_json["measurement_preview_settings"]["preview_mode"] == "monitoring"
    assert report_json["redline_capabilities"]["supports_lut"] is True
    assert report_json["calibration_roi"] == {"x": 0.2, "y": 0.2, "w": 0.4, "h": 0.4}
    assert report_json["target_strategies"] == ["median", "optimal_exposure"]
    assert report_json["shared_originals"][0]["original_frame"].endswith("G007_D060_0324M6_001.original.review.analysis-out.tiff")
    assert report_json["strategies"][0]["clips"][0]["both_corrected"].endswith("G007_D060_0324M6_001.both.review.median.analysis-out.tiff")
    assert report_json["white_balance_model"]["candidate_count"] >= 1
    assert isinstance(report_json.get("per_camera_analysis"), list)
    assert report_json["per_camera_analysis"][0]["clip_id"] == "G007_D060_0324M6_001"
    assert report_json["run_assessment"]["status"] in {"READY", "READY_WITH_WARNINGS", "REVIEW_REQUIRED", "DO_NOT_PUSH"}
    assert "gray_target_consistency" in report_json["run_assessment"]
    assert report_json["gray_target_consistency"]["dominant_target_class"] in {"sphere", "gray_card", "unresolved"}
    assert "render_truth_summary" in report_json
    assert Path(report_json["ipp2_validation_path"]).exists()
    assert Path(report_json["ipp2_closed_loop_trace_path"]).exists()
    assert report_json["ipp2_validation"]["contact_sheet_preview_matches_validation"] is False
    assert report_json["strategies"][0]["clips"][0]["metrics"]["preview_transform"] == report_json["preview_transform"]
    assert report_json["strategies"][0]["clips"][0]["ipp2_validation"]["status"] in {"PASS", "REVIEW", "FAIL"}
    assert "ipp2_gray_exposure_summary" in report_json["strategies"][0]["clips"][0]["ipp2_validation"]
    assert "ipp2_target_gray_exposure_summary" in report_json["strategies"][0]["clips"][0]["ipp2_validation"]
    assert isinstance(report_json["strategies"][0]["clips"][0]["ipp2_validation"].get("ipp2_zone_residuals"), list)
    assert "operator_guidance" in report_json["strategies"][0]["clips"][0]["ipp2_validation"]
    assert report_json["strategies"][0]["clips"][0]["ipp2_validation"]["operator_guidance"]["direction"] in {"open", "close", "hold"}
    assert "initial_offset_stops" in report_json["strategies"][0]["clips"][0]["metrics"]["exposure"]
    assert "ipp2_closed_loop_iterations" in report_json["strategies"][0]["clips"][0]["metrics"]["exposure"]
    assert "measured_log2_luminance_monitoring" in report_json["strategies"][0]["clips"][0]["metrics"]["exposure"]
    assert "measured_log2_luminance_raw" in report_json["strategies"][0]["clips"][0]["metrics"]["exposure"]
    assert Path(report_json["clips"][0]["original_frame"]).exists()
    assert not list((tmp_path / "analysis-out" / "previews").glob("*.000000.jpg"))
    assert not (tmp_path / "analysis-out" / "previews" / "_ipp2_closed_loop").exists()
    preview_commands = json.loads((tmp_path / "analysis-out" / "previews" / "preview_commands.json").read_text(encoding="utf-8"))
    strategy_command = next(command for command in preview_commands["commands"] if command["variant"] == "both" and command["strategy"] == "median")
    assert strategy_command["application_method"] == "direct_redline_flags"
    assert strategy_command["preview_color_applied"] is False
    assert strategy_command["validation_method"] == "pixel_diff_from_baseline"
    assert strategy_command["correction_application_method"] == "direct_redline_flags"
    assert strategy_command["command"] is not None
    assert strategy_command["rmd_path"] in {"", None}
    assert "--useMeta" in strategy_command["command"]
    assert "--exposureAdjust" in strategy_command["command"]
    assert "--loadRMD" not in strategy_command["command"]
    assert "--lut" in strategy_command["command"]
    assert report_json["color_preview_enabled"] is False
    assert report_json["color_preview_status"] == "disabled_unverified"
    assert report_json["color_preview_operator_status"] == "Not shown in operator review"
    html = Path(payload["report_html"]).read_text(encoding="utf-8")
    assert "Array Calibration Summary" in html
    assert "Hero Center IRE" in html
    assert "Original → Target" in html
    assert "Measurement source" in html
    assert "Hero-center patch" in html
    assert "HOW TO READ THIS REPORT" in html
    assert "<th>Detection</th>" not in html
    assert "<th>Result</th>" not in html
    assert "Digital correction applied" not in html
    assert "Reference Use" not in html
    assert "Validation state" not in html
    assert "Scalar Source" not in html
    assert "White Balance Model" in html or "White Balance" in html
    assert "summary-table" in html
    assert "Recommended action:" in html
    assert "G007_D060_0324M6_001" in html
    assert "images/G007_D060_corrected.jpg" in html
    assert "images/G007_D060_mask.jpg" in html
    assert Path(payload["report_html"]).with_name("contact_sheet_debug.json").exists()
    assert Path(payload["scientific_validation_path"]).exists()
    assert Path(payload["scientific_validation_markdown_path"]).exists()
    assert payload["preview_mode"] == "monitoring"
    assert payload["preview_settings"]["lut_path"].endswith("show.cube")
    assert payload["measurement_preview_settings"]["preview_mode"] == "monitoring"
    assert payload["redline_capabilities"]["supports_lut"] is True


def test_resolve_redline_executable_uses_project_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "r3dmatch.report.resolve_redline_tool_status",
        lambda: {
            "ready": True,
            "configured": "/usr/local/bin/REDLine",
            "resolved_path": "/usr/local/bin/REDLine",
            "source": "config",
            "config_path": str(tmp_path / "redline.json"),
            "error": "",
        },
    )
    assert _resolve_redline_executable() == "/usr/local/bin/REDLine"


def test_report_contact_sheet_requires_real_redline_for_strict_validation(tmp_path: Path) -> None:
    clip_path = tmp_path / "G007_D060_0324M6_001.R3D"
    clip_path.write_bytes(b"")
    analyze_path(
        str(tmp_path),
        out_dir=str(tmp_path / "analysis-out"),
        mode="scene",
        backend="mock",
        lut_override=None,
        calibration_path=None,
        sample_count=4,
        sampling_strategy="uniform",
        calibration_roi={"x": 0.2, "y": 0.2, "w": 0.4, "h": 0.4},
    )
    with pytest.raises(RuntimeError, match="real media, not the mock backend"):
        build_contact_sheet_report(
            str(tmp_path / "analysis-out"),
            out_dir=str(tmp_path / "report"),
            target_type="gray_sphere",
            processing_mode="both",
            target_strategies=["median"],
            require_real_redline=True,
        )


def test_contact_sheet_report_treats_calibration_preview_mode_as_monitoring_alias(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_redline(monkeypatch)
    clip_path = tmp_path / "G007_D060_0324M6_001.R3D"
    clip_path.write_bytes(b"")
    analyze_path(
        str(tmp_path),
        out_dir=str(tmp_path / "analysis-out"),
        mode="scene",
        backend="mock",
        lut_override=None,
        calibration_path=None,
        sample_count=4,
        sampling_strategy="uniform",
        calibration_roi={"x": 0.2, "y": 0.2, "w": 0.4, "h": 0.4},
    )
    payload = build_contact_sheet_report(
        str(tmp_path / "analysis-out"),
        out_dir=str(tmp_path / "report"),
        target_type="gray_sphere",
        processing_mode="both",
        target_strategies=["median"],
        preview_mode="calibration",
    )
    report_json = json.loads(Path(payload["report_json"]).read_text(encoding="utf-8"))
    preview_commands = json.loads((tmp_path / "analysis-out" / "previews" / "preview_commands.json").read_text(encoding="utf-8"))
    corrected_command = next(command for command in preview_commands["commands"] if command["variant"] == "both")
    assert report_json["preview_mode"] == "monitoring"
    assert report_json["preview_settings"]["requested_preview_mode"] == "calibration"
    assert report_json["preview_settings"]["preview_mode_alias"] == "calibration_compatibility_alias_to_monitoring"
    assert report_json["preview_settings"]["output_space"] == "BT.709"
    assert report_json["preview_settings"]["output_gamma"] == "BT.1886"
    assert report_json["ipp2_validation"]["contact_sheet_preview_matches_validation"] is True
    assert Path(report_json["ipp2_validation_path"]).exists()
    assert not any(command["variant"] == "original" for command in preview_commands["commands"])
    assert "--colorSpace" in corrected_command["command"]
    assert corrected_command["command"][corrected_command["command"].index("--colorSpace") + 1] == "13"
    assert "--gammaCurve" in corrected_command["command"]
    assert corrected_command["command"][corrected_command["command"].index("--gammaCurve") + 1] == "32"


def test_lightweight_analysis_report_generates_customer_facing_summary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_redline(monkeypatch)
    for name in [
        "G007_A063_032563_001.R3D",
        "G007_B063_0325V8_001.R3D",
        "G007_C063_032589_001.R3D",
    ]:
        (tmp_path / name).write_bytes(b"")
    analyze_path(
        str(tmp_path),
        out_dir=str(tmp_path / "analysis-out"),
        mode="scene",
        backend="mock",
        lut_override=None,
        calibration_path=None,
        sample_count=4,
        sampling_strategy="uniform",
        calibration_roi={"x": 0.2, "y": 0.2, "w": 0.4, "h": 0.4},
    )
    payload = build_lightweight_analysis_report(
        str(tmp_path / "analysis-out"),
        out_dir=str(tmp_path / "report"),
        target_type="gray_sphere",
        processing_mode="both",
        matching_domain="scene",
        target_strategies=["median", "optimal-exposure"],
        artifact_mode="debug",
    )
    assert Path(payload["report_json"]).exists()
    assert Path(payload["report_html"]).exists()
    assert payload["preview_report_pdf"] is None
    report_json = json.loads(Path(payload["report_json"]).read_text(encoding="utf-8"))
    assert report_json["review_mode"] == "lightweight_analysis"
    assert report_json["report_kind"] == "lightweight_analysis"
    assert report_json["matching_domain"] == "perceptual"
    assert report_json["requested_matching_domain"] == "scene"
    assert report_json["measurement_domain_trace"]["measurement_source"] == "rendered_preview_ipp2"
    assert report_json["executive_synopsis"]
    assert "median strategy" in report_json["executive_synopsis"].lower()
    assert report_json["recommended_strategy"]["strategy_key"] in {"median", "optimal_exposure"}
    assert "exposure_plot_svg" in report_json["visuals"]
    assert report_json["visuals"]["before_after_exposure_svg"] in {"", None}
    assert "strategy_chart_svg" in report_json["visuals"]
    assert "trust_chart_svg" in report_json["visuals"]
    assert report_json["run_assessment"]["status"] in {"READY", "READY_WITH_WARNINGS", "REVIEW_REQUIRED", "DO_NOT_PUSH"}
    assert report_json["run_assessment"]["recommendation_strength"] in {"HIGH_CONFIDENCE", "MEDIUM_CONFIDENCE", "LOW_CONFIDENCE"}
    assert report_json["per_camera_analysis"][0]["trust_class"] in {"TRUSTED", "USE_WITH_CAUTION", "UNTRUSTED", "EXCLUDED"}
    assert "selection_diagnostics" in report_json["strategy_comparison"][0]
    assert report_json["white_balance_model"]["model_key"] in {
        "shared_kelvin_per_camera_tint",
        "constrained_kelvin_per_camera_tint",
        "per_camera_kelvin_per_camera_tint",
        "shared_kelvin_shared_tint",
    }
    assert report_json["measurement_render_count"] == 3
    preview_commands = json.loads((tmp_path / "analysis-out" / "previews" / "preview_commands.json").read_text(encoding="utf-8"))
    assert preview_commands["skipped_bulk_preview_rendering"] is True
    assert preview_commands["commands"] == []
    html = Path(payload["report_html"]).read_text(encoding="utf-8")
    assert "R3DMatch Diagnostic Review" in html
    assert any(label in html for label in ["ARRAY WITHIN CALIBRATION TOLERANCE", "ARRAY NEEDS CALIBRATION REVIEW", "ARRAY OUT OF CALIBRATION"])
    if report_json["visuals"]["strategy_chart_svg"]:
        assert "Strategy Comparison" in html
    assert "Before / After Exposure" in html
    assert "Measurement Stability" in html
    assert "Calibration Review Notes" in html
    assert "Per-Camera Analysis" in html
    assert "Lightweight Analysis" in html
    assert "Shared Kelvin" in html
    assert "Click to enlarge" in html
    assert "chart-modal" in html
    assert "chart-launch" in html
    assert "Why This Was Chosen" in html
    assert "Closest current reference candidate" in html


def test_lightweight_analysis_report_prefers_monitoring_measurement_quality_over_scene_array_quality(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_redline(monkeypatch)
    for name in [
        "G007_A063_032563_001.R3D",
        "G007_B063_0325V8_001.R3D",
        "G007_C063_032589_001.R3D",
    ]:
        (tmp_path / name).write_bytes(b"")
    analyze_path(
        str(tmp_path),
        out_dir=str(tmp_path / "analysis-out"),
        mode="scene",
        backend="mock",
        lut_override=None,
        calibration_path=None,
        sample_count=4,
        sampling_strategy="uniform",
        calibration_roi={"x": 0.2, "y": 0.2, "w": 0.4, "h": 0.4},
    )
    array_path = tmp_path / "analysis-out" / "array_calibration.json"
    array_payload = json.loads(array_path.read_text(encoding="utf-8"))
    camera = array_payload["cameras"][0]
    camera["quality"]["confidence"] = 0.42
    camera["quality"]["neutral_sample_log2_spread"] = 0.123
    camera["quality"]["neutral_sample_chromaticity_spread"] = 0.0045
    camera["quality"]["post_color_residual"] = 0.007
    camera["quality"]["flags"] = ["quality_override_test"]
    array_path.write_text(json.dumps(array_payload, indent=2), encoding="utf-8")

    payload = build_lightweight_analysis_report(
        str(tmp_path / "analysis-out"),
        out_dir=str(tmp_path / "report"),
        target_type="gray_sphere",
        processing_mode="both",
        matching_domain="scene",
        target_strategies=["median", "optimal-exposure"],
        artifact_mode="debug",
    )
    report_json = json.loads(Path(payload["report_json"]).read_text(encoding="utf-8"))
    row = next(item for item in report_json["per_camera_analysis"] if item["clip_id"] == camera["clip_id"])
    assert report_json["matching_domain"] == "perceptual"
    assert row["monitoring_measurement_source"] == "rendered_preview_ipp2"
    assert row["measured_gray_exposure_summary"] != "n/a"
    assert row["confidence"] != pytest.approx(0.42)
    assert row["neutral_sample_log2_spread"] != pytest.approx(0.123)
    assert row["neutral_sample_chromaticity_spread"] != pytest.approx(0.0045)
    assert "quality_override_test" not in row["note"]
    assert row["trust_class"] in {"TRUSTED", "USE_WITH_CAUTION", "UNTRUSTED", "EXCLUDED"}


def test_lightweight_analysis_report_writes_exposure_debug_trace_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_redline(monkeypatch)
    for name in [
        "G007_A063_032563_001.R3D",
        "G007_B063_0325V8_001.R3D",
        "G007_C063_032589_001.R3D",
    ]:
        (tmp_path / name).write_bytes(b"")
    analyze_path(
        str(tmp_path),
        out_dir=str(tmp_path / "analysis-out"),
        mode="scene",
        backend="mock",
        lut_override=None,
        calibration_path=None,
        sample_count=4,
        sampling_strategy="uniform",
        calibration_roi={"x": 0.2, "y": 0.2, "w": 0.4, "h": 0.4},
    )
    payload = build_lightweight_analysis_report(
        str(tmp_path / "analysis-out"),
        out_dir=str(tmp_path / "report"),
        target_type="gray_sphere",
        processing_mode="both",
        matching_domain="scene",
        target_strategies=["median", "optimal-exposure"],
        artifact_mode="debug",
    )
    report_json = json.loads(Path(payload["report_json"]).read_text(encoding="utf-8"))
    debug_payload = dict(report_json.get("debug_exposure_trace") or {})
    assert Path(debug_payload["summary_path"]).exists()
    summary = json.loads(Path(debug_payload["summary_path"]).read_text(encoding="utf-8"))
    assert summary["roi_alignment"]["shared_roi"] is True
    first_camera = summary["cameras"][0]
    camera_trace_path = Path(debug_payload["directory"]) / f"{first_camera['camera_id']}.json"
    assert camera_trace_path.exists()
    trace = json.loads(camera_trace_path.read_text(encoding="utf-8"))
    assert trace["sampling"]["neutral_samples_monitoring"][0]["bounds"]["normalized_within_roi"]["w"] > 0.0
    assert "gray_log2_distribution" in trace["sampling"]["neutral_samples_monitoring"][0]
    assert Path(trace["sampling"]["roi_overlay_path"]).exists()


def test_lightweight_analysis_report_run_assessment_is_deterministic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_redline(monkeypatch)
    for name in [
        "G007_A064_0325RN_001.R3D",
        "H007_A064_0325PK_001.R3D",
        "I007_A064_0325XY_001.R3D",
        "G007_B064_032526_001.R3D",
    ]:
        (tmp_path / name).write_bytes(b"")
    analyze_path(
        str(tmp_path),
        out_dir=str(tmp_path / "analysis-out"),
        mode="scene",
        backend="mock",
        lut_override=None,
        calibration_path=None,
        sample_count=4,
        sampling_strategy="uniform",
        calibration_roi={"x": 0.2, "y": 0.2, "w": 0.4, "h": 0.4},
    )

    first = build_lightweight_analysis_report(
        str(tmp_path / "analysis-out"),
        out_dir=str(tmp_path / "report-a"),
        target_type="gray_sphere",
        processing_mode="both",
        matching_domain="scene",
        target_strategies=["median", "optimal-exposure"],
    )
    second = build_lightweight_analysis_report(
        str(tmp_path / "analysis-out"),
        out_dir=str(tmp_path / "report-b"),
        target_type="gray_sphere",
        processing_mode="both",
        matching_domain="scene",
        target_strategies=["median", "optimal-exposure"],
    )

    first_json = json.loads(Path(first["report_json"]).read_text(encoding="utf-8"))
    second_json = json.loads(Path(second["report_json"]).read_text(encoding="utf-8"))
    assert first_json["run_assessment"]["status"] == second_json["run_assessment"]["status"]
    assert first_json["run_assessment"]["recommendation_strength"] == second_json["run_assessment"]["recommendation_strength"]
    assert [item["trust_class"] for item in first_json["per_camera_analysis"]] == [item["trust_class"] for item in second_json["per_camera_analysis"]]


def test_validate_review_run_contract_accepts_lightweight_report_without_preview_images(tmp_path: Path) -> None:
    report_dir = tmp_path / "report"
    report_dir.mkdir(parents=True)
    (tmp_path / "summary.json").write_text(json.dumps({"clips": []}), encoding="utf-8")
    (tmp_path / "previews").mkdir(parents=True)
    (tmp_path / "previews" / "preview_commands.json").write_text(
        json.dumps({"review_mode": "lightweight_analysis", "commands": [], "skipped_bulk_preview_rendering": True}),
        encoding="utf-8",
    )
    (report_dir / "contact_sheet.json").write_text(
        json.dumps(
            {
                "report_kind": "lightweight_analysis",
                "review_mode": "lightweight_analysis",
                "clip_count": 2,
                "shared_originals": [],
                "strategies": [],
                "color_preview_status": "disabled_unverified",
                "color_preview_note": "Color preview disabled.",
            }
        ),
        encoding="utf-8",
    )
    (report_dir / "contact_sheet.html").write_text("<html><body>lightweight</body></html>", encoding="utf-8")
    (report_dir / "review_manifest.json").write_text(
        json.dumps({"review_mode": "lightweight_analysis", "report_html": str(report_dir / "contact_sheet.html")}),
        encoding="utf-8",
    )
    (report_dir / "review_package.json").write_text(json.dumps({"review_mode": "lightweight_analysis"}), encoding="utf-8")
    _write_minimal_array_calibration(tmp_path / "array_calibration.json")

    validation = validate_review_run_contract(str(tmp_path))

    assert validation["status"] == "success"
    assert validation["review_mode"] == "lightweight_analysis"


def test_validate_review_run_contract_skips_scene_physical_validation_for_perceptual_lightweight(tmp_path: Path) -> None:
    report_dir = tmp_path / "report"
    report_dir.mkdir(parents=True)
    (tmp_path / "summary.json").write_text(json.dumps({"backend": "red", "mode": "scene"}), encoding="utf-8")
    (tmp_path / "array_calibration.json").write_text(json.dumps({"backend": "red", "measurement_domain": "scene", "cameras": []}), encoding="utf-8")
    (tmp_path / "previews").mkdir(parents=True)
    (tmp_path / "previews" / "preview_commands.json").write_text(
        json.dumps({"review_mode": "lightweight_analysis", "commands": [], "skipped_bulk_preview_rendering": True}),
        encoding="utf-8",
    )
    (report_dir / "contact_sheet.json").write_text(
        json.dumps({"clip_count": 1, "review_mode": "lightweight_analysis", "matching_domain": "perceptual", "shared_originals": [], "strategies": []}),
        encoding="utf-8",
    )
    (report_dir / "contact_sheet.html").write_text("<html></html>", encoding="utf-8")
    (report_dir / "review_manifest.json").write_text(json.dumps({"review_mode": "lightweight_analysis"}), encoding="utf-8")
    (report_dir / "review_package.json").write_text(
        json.dumps({"review_mode": "lightweight_analysis", "matching_domain": "perceptual", "exposure_measurement_domain": "perceptual"}),
        encoding="utf-8",
    )

    validation = validate_review_run_contract(str(tmp_path))

    assert validation["status"] == "success"
    assert validation["physical_validation"]["status"] == "unsupported"
    assert "not a RED scene-domain array calibration" in " ".join(validation["warnings"])
    assert validation["preview_reference_count"] == 0
    assert validation["preview_existing_count"] == 0


def test_build_review_package_skips_temporary_rmds_for_lightweight_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_redline(monkeypatch)
    clip_path = tmp_path / "G007_D060_0324M6_001.R3D"
    clip_path.write_bytes(b"")
    analyze_path(
        str(tmp_path),
        out_dir=str(tmp_path / "analysis-out"),
        mode="scene",
        backend="mock",
        lut_override=None,
        calibration_path=None,
        sample_count=4,
        sampling_strategy="uniform",
        calibration_roi={"x": 0.2, "y": 0.2, "w": 0.4, "h": 0.4},
    )
    called = {"value": False}

    def _fake_write_rmds_from_analysis(*args, **kwargs):  # type: ignore[no-untyped-def]
        called["value"] = True
        return {"unexpected": True}

    monkeypatch.setattr("r3dmatch.report.write_rmds_from_analysis", _fake_write_rmds_from_analysis)
    package = build_review_package(
        str(tmp_path / "analysis-out"),
        out_dir=str(tmp_path / "analysis-out"),
        review_mode="lightweight_analysis",
        target_type="gray_sphere",
        processing_mode="both",
        matching_domain="scene",
        target_strategies=["median"],
    )
    assert called["value"] is False
    assert package["temporary_rmd_manifest"]["skipped"] is True


def test_build_contact_sheet_report_cancellation_removes_partial_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_redline(monkeypatch)
    clip_path = tmp_path / "G007_D060_0324M6_001.R3D"
    clip_path.write_bytes(b"")
    analyze_path(
        str(tmp_path),
        out_dir=str(tmp_path / "analysis-out"),
        mode="scene",
        backend="mock",
        lut_override=None,
        calibration_path=None,
        sample_count=4,
        sampling_strategy="uniform",
        calibration_roi={"x": 0.2, "y": 0.2, "w": 0.4, "h": 0.4},
    )

    def fake_render_contact_sheet_pdf_from_html(html_path, *, output_path):  # type: ignore[no-untyped-def]
        assert Path(html_path).name == "contact_sheet.html"
        output = Path(output_path)
        output.write_bytes(b"%PDF-1.4 partial\n")
        raise CancellationError("Run cancelled during report build.")

    monkeypatch.setattr("r3dmatch.report.render_contact_sheet_pdf_from_html", fake_render_contact_sheet_pdf_from_html)

    with pytest.raises(CancellationError):
        build_contact_sheet_report(
            str(tmp_path / "analysis-out"),
            out_dir=str(tmp_path / "report"),
            target_type="gray_sphere",
            processing_mode="both",
            target_strategies=["median"],
        )

    assert not (tmp_path / "report" / "contact_sheet.json").exists()
    assert not (tmp_path / "report" / "contact_sheet.html").exists()
    assert not (tmp_path / "report" / "preview_contact_sheet.pdf").exists()
    assert not (tmp_path / "report" / "review_manifest.json").exists()


def test_report_layout_helpers_scale_columns_and_tiles() -> None:
    assert _report_grid_columns(3) == 3
    assert _report_grid_columns(12) == 4
    assert _report_grid_columns(24) == 6
    assert _report_grid_columns(41) == 8
    assert _report_tiles_per_page(3) == 12
    assert _report_tiles_per_page(4) == 12
    assert _report_tiles_per_page(6) == 12
    assert _report_tiles_per_page(8) == 16


def test_write_rmd_for_clip_rejects_non_identity_cdl_when_disabled(tmp_path: Path) -> None:
    clip_path = tmp_path / "G007_D060_0324M6_001.R3D"
    clip_path.write_bytes(b"")
    sidecar_payload = {
        "clip_id": "G007_D060_0324M6_001",
        "source_path": str(clip_path),
        "rmd_mapping": {
            "exposure": {"final_offset_stops": 0.0},
            "color": {
                "cdl": {
                    "slope": [1.2, 0.8, 1.0],
                    "offset": [0.0, 0.0, 0.0],
                    "power": [1.0, 1.0, 1.0],
                    "saturation": 1.0,
                },
                "cdl_enabled": False,
            },
        },
    }
    with pytest.raises(ValueError, match="CDL payload is non-identity but cdl_enabled is false"):
        write_rmd_for_clip_with_metadata("G007_D060_0324M6_001", sidecar_payload, tmp_path / "rmd-out")


def test_render_contact_sheet_html_uses_logo_and_dynamic_grid(tmp_path: Path) -> None:
    preview_dir = tmp_path / "previews"
    original_path = preview_dir / "CAM001.original.review.run01.jpg"
    corrected_path = preview_dir / "CAM001.both.review.median.run01.jpg"
    overlay_dir = tmp_path / "report" / "review_detection_overlays"
    overlay_path = overlay_dir / "CAM001.original_detection.png"
    _write_test_preview(original_path, (90, 90, 90))
    _write_test_preview(corrected_path, (140, 120, 100))
    _write_test_preview(overlay_path, (180, 180, 180))
    payload = {
        "target_type": "gray_sphere",
        "processing_mode": "both",
        "preview_transform": "REDLine IPP2 / BT.709 / BT.1886 / Medium / Medium",
        "exposure_anchor_summary": "Exposure Anchor: Hero clip CAM001",
        "clip_count": 3,
        "calibration_roi": {"x": 0.2, "y": 0.2, "w": 0.4, "h": 0.4},
        "shared_originals": [
                {
                    "clip_id": f"CAM{i:03d}",
                    "group_key": f"CAM{i:03d}",
                    "original_frame": str(original_path),
                    "measurement_valid": True,
                    "gray_target_measurement_valid": True,
                    "display_scalar_log2": -2.5,
                    "sample_1_ire": 24.0,
                    "sample_2_ire": 23.0,
                "sample_3_ire": 22.0,
                "clip_metadata": {"iso": 800, "shutter_seconds": 1 / 48, "kelvin": 5600, "tint": 2.0},
            }
            for i in range(1, 4)
        ],
        "sphere_detection_summary": {
            "rows": [
                {
                    "clip_id": f"CAM{i:03d}",
                    "original_overlay_path": str(overlay_path),
                    "corrected_overlay_path": str(overlay_path),
                }
                for i in range(1, 4)
            ]
        },
        "ipp2_validation": {
            "status_counts": {"PASS": 3, "REVIEW": 0, "FAIL": 0},
            "all_within_tolerance": True,
            "best_residual": 0.01,
            "median_residual": 0.02,
            "max_residual": 0.03,
        },
        "visuals": {"exposure_plot_svg": "<svg viewBox='0 0 100 40'></svg>"},
        "strategies": [
            {
                "strategy_key": "median",
                "strategy_label": "Median",
                "reference_clip_id": None,
                "anchor_summary": "Exposure Anchor: Hero clip CAM001",
                "target_log2_luminance": -2.8,
                "calibration_roi": {"x": 0.2, "y": 0.2, "w": 0.4, "h": 0.4},
                "clips": [
                        {
                            "clip_id": f"CAM{i:03d}",
                            "group_key": f"CAM{i:03d}",
                            "measurement_valid": True,
                            "gray_target_measurement_valid": True,
                            "both_corrected": str(corrected_path),
                            "metrics": {
                                "exposure": {
                                    "measurement_valid": True,
                                    "final_offset_stops": 0.1 * i,
                                    "measured_log2_luminance_monitoring": -2.5,
                                "sample_1_ire": 24.0,
                                "sample_2_ire": 23.0,
                                "sample_3_ire": 22.0,
                            },
                            "commit_values": {"exposureAdjust": 0.1 * i, "kelvin": 5600, "tint": 0.0},
                            "color": {"rgb_gains": [1.0, 1.0, 1.0]},
                            "confidence": 0.95,
                            "flags": [],
                        },
                        "ipp2_validation": {
                            "camera_label": f"CAM{i:03d}",
                            "clip_id": f"CAM{i:03d}",
                            "status": "PASS",
                            "sample_1_ire": 24.0,
                            "sample_2_ire": 23.0,
                            "sample_3_ire": 22.0,
                            "ipp2_original_gray_exposure_summary": "Sample 1 24 / Sample 2 23 / Sample 3 22 IRE",
                            "ipp2_gray_exposure_summary": "Sample 1 24 / Sample 2 23 / Sample 3 22 IRE",
                            "ipp2_residual_abs_stops": 0.02,
                            "ipp2_residual_stops": 0.02,
                            "camera_offset_from_anchor": 0.1 * i,
                            "suggested_action": "No adjustment needed",
                            "profile_note": "Profile consistent",
                            "sphere_detection_note": "Sphere detection: verified",
                            "corrected_image_path": str(corrected_path),
                            "original_image_path": str(original_path),
                        },
                        "render_validation": {"pixel_output_changed": True, "pixel_diff_from_baseline": 12.5},
                    }
                    for i in range(1, 4)
                ],
            }
        ],
    }
    html_path = tmp_path / "report" / "contact_sheet.html"
    html = render_contact_sheet_html(payload, html_path=str(html_path))
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(html, encoding="utf-8")
    asset_validation = validate_contact_sheet_html_assets(str(html_path))
    debug_payload = json.loads((html_path.parent / "contact_sheet_debug.json").read_text(encoding="utf-8"))
    assert "R3DMATCH DEBUG — NEW RENDER PATH ACTIVE" not in html
    assert "R3DMatch Calibration Assessment" in html
    assert "landscape-page" in html
    assert "summary-table" in html
    assert "Array Calibration Summary" in html
    assert "ORIGINAL + SOLVE OVERLAY" in html
    assert "CORRECTED FRAME" in html
    assert "images/CAM001_corrected.jpg" in html
    assert "images/CAM001_mask.jpg" in html
    assert "White Balance" in html
    assert "Basis / cast" not in html
    assert "S1:" not in html
    assert "S2:" not in html
    assert "S3:" not in html
    assert "Target Sample" not in html
    assert "Hero Center IRE" in html
    assert "Original → Target" in html
    assert "Recommended action:" in html
    assert "Exposure Anchor" in html
    assert "Exposure Anchor: Exposure Anchor:" not in html
    assert "Hero-center patch" in html
    assert "Scalar -2.500 log2" in html
    assert "Original white balance trace" not in html
    assert "HOW TO READ THIS REPORT" in html
    assert "Confidence:" not in html
    assert "Zone residuals" not in html
    assert "<th>Detection</th>" not in html
    assert "<th>Result</th>" not in html
    assert asset_validation["all_assets_exist"] is True
    assert debug_payload["validation_status"]["all_within_tolerance"] is True
    assert debug_payload["measurement_values_per_camera"][0]["measurement_values"]["hero_center_ire"] == pytest.approx(17.68, abs=0.02)
    assert debug_payload["measurement_values_per_camera"][0]["measurement_values"]["sample_2_ire"] == pytest.approx(23.0)


def test_gray_target_consistency_summary_blocks_mixed_retained_target_classes() -> None:
    summary = _gray_target_consistency_summary(
        [
            {"clip_id": "A", "camera_label": "A", "reference_use": "Included", "trust_class": "TRUSTED", "gray_target_class": "sphere", "measurement_valid": True},
            {"clip_id": "B", "camera_label": "B", "reference_use": "Included", "trust_class": "USE_WITH_CAUTION", "gray_target_class": "gray_card", "measurement_valid": True},
            {"clip_id": "C", "camera_label": "C", "reference_use": "Excluded", "trust_class": "EXCLUDED", "gray_target_class": "gray_card", "measurement_valid": False},
        ]
    )

    assert summary["dominant_target_class"] == "sphere"
    assert summary["mixed_target_classes"] is True
    assert summary["commit_blocked"] is True
    assert summary["non_dominant_clip_ids"] == ["B"]


def test_render_contact_sheet_html_large_arrays_default_to_overview_and_outliers(tmp_path: Path) -> None:
    preview_dir = tmp_path / "previews"
    overlay_dir = tmp_path / "report" / "review_detection_overlays"
    shared_originals = []
    strategy_clips = []
    overlay_rows = []
    validation_rows = []
    for index in range(24):
        clip_id = f"CAM{index + 1:03d}"
        original_path = preview_dir / f"{clip_id}.original.jpg"
        corrected_path = preview_dir / f"{clip_id}.corrected.jpg"
        overlay_path = overlay_dir / f"{clip_id}.overlay.png"
        _write_test_preview(original_path, (80 + (index % 10), 90, 100))
        _write_test_preview(corrected_path, (110 + (index % 10), 120, 130))
        _write_test_preview(overlay_path, (150, 150, 150))
        shared_originals.append(
                {
                    "clip_id": clip_id,
                    "group_key": clip_id,
                    "original_frame": str(original_path),
                    "measurement_valid": True,
                    "gray_target_measurement_valid": True,
                    "display_scalar_log2": -2.5 + index * 0.01,
                    "sample_1_ire": 24.0,
                    "sample_2_ire": 23.0,
                "sample_3_ire": 22.0,
                "clip_metadata": {"kelvin": 5600 + index, "tint": (index % 5) - 2},
            }
        )
        strategy_clips.append(
                {
                    "clip_id": clip_id,
                    "group_key": clip_id,
                    "measurement_valid": True,
                    "gray_target_measurement_valid": True,
                    "both_corrected": str(corrected_path),
                    "metrics": {
                        "exposure": {
                            "measurement_valid": True,
                            "final_offset_stops": 0.05 * (index % 4),
                            "measured_log2_luminance_monitoring": -2.5 + index * 0.01,
                        "sample_1_ire": 24.0,
                        "sample_2_ire": 23.0,
                        "sample_3_ire": 22.0,
                    },
                    "commit_values": {"exposureAdjust": 0.05 * (index % 4), "kelvin": 5600 + index, "tint": (index % 5) - 2},
                },
                "pre_color_residual": 0.01 * (index % 3),
                "ipp2_validation": {
                    "camera_label": clip_id,
                    "clip_id": clip_id,
                    "status": "REVIEW" if index in {3, 11, 19} else "PASS",
                    "sample_1_ire": 24.0,
                    "sample_2_ire": 23.0,
                    "sample_3_ire": 22.0,
                    "ipp2_residual_abs_stops": 0.11 if index in {3, 11, 19} else 0.02,
                    "ipp2_residual_stops": 0.11 if index in {3, 11, 19} else 0.02,
                    "corrected_image_path": str(corrected_path),
                    "original_image_path": str(original_path),
                    "reference_use": "Excluded" if index == 19 else "Included",
                    "sphere_detection_note": "Sphere detection: verified",
                    "profile_note": "Profile consistent",
                },
            }
        )
        overlay_rows.append({"clip_id": clip_id, "original_overlay_path": str(overlay_path), "corrected_overlay_path": str(overlay_path)})
        validation_rows.append(
            {
                "clip_id": clip_id,
                "strategy_key": "median",
                "status": "REVIEW" if index in {3, 11, 19} else "PASS",
                "sample_1_ire": 24.0,
                "sample_2_ire": 23.0,
                "sample_3_ire": 22.0,
                "ipp2_residual_abs_stops": 0.11 if index in {3, 11, 19} else 0.02,
                "ipp2_residual_stops": 0.11 if index in {3, 11, 19} else 0.02,
                "corrected_image_path": str(corrected_path),
                "original_image_path": str(original_path),
                "reference_use": "Excluded" if index == 19 else "Included",
            }
        )
    payload = {
        "target_type": "gray_sphere",
        "processing_mode": "both",
        "report_focus": "auto",
        "shared_originals": shared_originals,
        "sphere_detection_summary": {"rows": overlay_rows},
        "ipp2_validation": {
            "status_counts": {"PASS": 21, "REVIEW": 3, "FAIL": 0},
            "all_within_tolerance": False,
            "best_residual": 0.01,
            "median_residual": 0.02,
            "max_residual": 0.11,
            "rows": validation_rows,
        },
        "strategies": [{"strategy_key": "median", "strategy_label": "Median", "clips": strategy_clips}],
    }

    html = render_contact_sheet_html(payload, html_path=str(tmp_path / "report" / "contact_sheet.html"))

    assert "Array Calibration Summary" in html
    assert "summary-table" in html
    assert "Exposure Spread" in html
    assert "Hero-center patch" in html
    assert "HOW TO READ THIS REPORT" in html
    assert "<th>Detection</th>" not in html
    assert "<th>Result</th>" not in html
    assert html.count("ORIGINAL + SOLVE OVERLAY") < 24


def test_validate_contact_sheet_html_assets_reports_missing_relative_images(tmp_path: Path) -> None:
    report_dir = tmp_path / "report"
    report_dir.mkdir()
    existing_dir = tmp_path / "previews" / "_measurement"
    existing_dir.mkdir(parents=True)
    existing_image = existing_dir / "CAM001.jpg"
    existing_image.write_bytes(b"jpg")
    html_path = report_dir / "contact_sheet.html"
    html_path.write_text(
        (
            "<!doctype html><html><body>"
            "<img src='../previews/_measurement/CAM001.jpg' alt='ok'>"
            "<img src='review_detection_overlays/CAM001.png' alt='missing'>"
            "</body></html>"
        ),
        encoding="utf-8",
    )

    validation = validate_contact_sheet_html_assets(str(html_path))

    assert validation["all_assets_exist"] is False
    assert validation["image_count"] == 2
    assert "../previews/_measurement/CAM001.jpg" not in " ".join(validation["missing_assets"])
    assert "review_detection_overlays/CAM001.png" in validation["missing_assets"][0]


def test_render_contact_sheet_html_fails_when_overlay_asset_is_missing(tmp_path: Path) -> None:
    preview_dir = tmp_path / "previews"
    original_path = preview_dir / "CAM001.original.review.run01.jpg"
    corrected_path = preview_dir / "CAM001.both.review.median.run01.jpg"
    _write_test_preview(original_path, (90, 90, 90))
    _write_test_preview(corrected_path, (120, 120, 120))
    payload = {
        "shared_originals": [
                {
                    "clip_id": "CAM001",
                    "group_key": "CAM001",
                    "original_frame": str(original_path),
                    "measurement_valid": True,
                    "gray_target_measurement_valid": True,
                    "display_scalar_log2": -2.5,
                    "sample_1_ire": 24.0,
                    "sample_2_ire": 23.0,
                "sample_3_ire": 22.0,
            }
        ],
        "sphere_detection_summary": {
            "rows": [{"clip_id": "CAM001", "original_overlay_path": str(tmp_path / "report" / "review_detection_overlays" / "missing.png")}]
        },
        "ipp2_validation": {
            "status_counts": {"PASS": 1, "REVIEW": 0, "FAIL": 0},
            "all_within_tolerance": True,
            "rows": [
                {
                    "clip_id": "CAM001",
                    "strategy_key": "median",
                    "status": "PASS",
                    "sample_1_ire": 24.0,
                    "sample_2_ire": 23.0,
                    "sample_3_ire": 22.0,
                    "corrected_image_path": str(corrected_path),
                    "original_image_path": str(original_path),
                }
            ],
        },
        "strategies": [
            {
                "strategy_key": "median",
                "strategy_label": "Median",
                "clips": [
                        {
                            "clip_id": "CAM001",
                            "measurement_valid": True,
                            "gray_target_measurement_valid": True,
                            "both_corrected": str(corrected_path),
                            "metrics": {
                                "exposure": {
                                    "measurement_valid": True,
                                    "sample_1_ire": 24.0,
                                    "sample_2_ire": 23.0,
                                    "sample_3_ire": 22.0,
                                "measured_log2_luminance_monitoring": -2.5,
                            },
                            "commit_values": {"exposureAdjust": 0.1, "kelvin": 5600, "tint": 0.0},
                        },
                        "ipp2_validation": {
                            "clip_id": "CAM001",
                            "camera_label": "CAM001",
                            "status": "PASS",
                            "sample_1_ire": 24.0,
                            "sample_2_ire": 23.0,
                            "sample_3_ire": 22.0,
                            "corrected_image_path": str(corrected_path),
                            "original_image_path": str(original_path),
                        },
                    }
                ],
            }
        ],
    }

    with pytest.raises(RuntimeError, match="sphere mask overlay"):
        render_contact_sheet_html(payload, html_path=str(tmp_path / "report" / "contact_sheet.html"))


def test_render_contact_sheet_html_prefers_authoritative_sample_payload(tmp_path: Path) -> None:
    preview_dir = tmp_path / "previews"
    original_path = preview_dir / "CAM001.original.review.run01.jpg"
    corrected_path = preview_dir / "CAM001.both.review.median.run01.jpg"
    overlay_dir = tmp_path / "report" / "review_detection_overlays"
    overlay_path = overlay_dir / "CAM001.original_detection.png"
    _write_test_preview(original_path, (90, 90, 90))
    _write_test_preview(corrected_path, (120, 120, 120))
    _write_test_preview(overlay_path, (160, 160, 160))
    payload = {
        "shared_originals": [
            {
                "clip_id": "CAM001",
                "group_key": "CAM001",
                "original_frame": str(original_path),
                "measurement_valid": True,
                "gray_target_measurement_valid": True,
                "display_scalar_log2": -2.5,
                "sample_1_ire": 24.0,
                "sample_2_ire": 23.0,
                "sample_3_ire": 22.0,
            }
        ],
        "sphere_detection_summary": {
            "rows": [{"clip_id": "CAM001", "original_overlay_path": str(overlay_path), "corrected_overlay_path": str(overlay_path)}]
        },
        "ipp2_validation": {
            "status_counts": {"PASS": 1, "REVIEW": 0, "FAIL": 0},
            "all_within_tolerance": True,
            "rows": [
                {
                    "clip_id": "CAM001",
                    "strategy_key": "median",
                    "status": "PASS",
                    "sample_1_ire": 25.0,
                    "sample_2_ire": 23.0,
                    "sample_3_ire": 22.0,
                    "corrected_image_path": str(corrected_path),
                    "original_image_path": str(original_path),
                }
            ],
        },
        "strategies": [
            {
                "strategy_key": "median",
                "strategy_label": "Median",
                "clips": [
                    {
                        "clip_id": "CAM001",
                        "measurement_valid": True,
                        "gray_target_measurement_valid": True,
                        "both_corrected": str(corrected_path),
                        "metrics": {
                            "exposure": {
                                "measurement_valid": True,
                                "sample_1_ire": 24.0,
                                "sample_2_ire": 23.0,
                                "sample_3_ire": 22.0,
                                "measured_log2_luminance_monitoring": -2.5,
                            },
                            "commit_values": {"exposureAdjust": 0.1, "kelvin": 5600, "tint": 0.0},
                        },
                        "ipp2_validation": {
                            "clip_id": "CAM001",
                            "camera_label": "CAM001",
                            "status": "PASS",
                            "sample_1_ire": 25.0,
                            "sample_2_ire": 23.0,
                            "sample_3_ire": 22.0,
                            "corrected_image_path": str(corrected_path),
                            "original_image_path": str(original_path),
                        },
                    }
                ],
            }
        ],
    }

    html = render_contact_sheet_html(payload, html_path=str(tmp_path / "report" / "contact_sheet.html"))
    assert "Hero Center IRE" in html
    assert "Original → Target" in html
    debug_payload = json.loads((tmp_path / "report" / "contact_sheet_debug.json").read_text(encoding="utf-8"))
    assert debug_payload["measurement_values_per_camera"][0]["measurement_values"]["sample_1_ire"] == pytest.approx(24.0)
    assert debug_payload["measurement_values_per_camera"][0]["measurement_values"]["target_sample_label"] == "Sample 2"


def test_render_contact_sheet_pdf_paginates_large_camera_counts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    preview_dir = tmp_path / "previews"
    overlay_dir = tmp_path / "report" / "review_detection_overlays"
    shared_originals = []
    strategy_clips = []
    overlay_rows = []
    validation_rows = []
    for index in range(40):
        clip_id = f"CAM{index + 1:03d}"
        original_path = preview_dir / f"{clip_id}.original.review.large.jpg"
        corrected_path = preview_dir / f"{clip_id}.both.review.median.large.jpg"
        overlay_path = overlay_dir / f"{clip_id}.original_detection.png"
        _write_test_preview(original_path, (80 + (index % 10), 90, 100))
        _write_test_preview(corrected_path, (110 + (index % 10), 120, 130))
        _write_test_preview(overlay_path, (150, 150, 150))
        shared_originals.append(
            {
                "clip_id": clip_id,
                "group_key": clip_id,
                "original_frame": str(original_path),
                "measurement_valid": True,
                "gray_target_measurement_valid": True,
                "confidence": 0.9,
                "sample_1_ire": 24.0,
                "sample_2_ire": 23.0,
                "sample_3_ire": 22.0,
                "measured_log2_luminance_monitoring": -2.5,
                "measured_log2_luminance_raw": -2.9,
            }
        )
        strategy_clips.append(
            {
                "clip_id": clip_id,
                "group_key": clip_id,
                "measurement_valid": True,
                "gray_target_measurement_valid": True,
                "both_corrected": str(corrected_path),
                "metrics": {
                    "exposure": {"measurement_valid": True, "final_offset_stops": 0.25, "measured_log2_luminance_raw": -2.9},
                    "color": {"rgb_gains": [1.01, 0.99, 1.0]},
                    "confidence": 0.91,
                    "commit_values": {"exposureAdjust": 0.25, "kelvin": 5600, "tint": 0.0},
                },
                "ipp2_validation": {
                    "camera_label": clip_id,
                    "clip_id": clip_id,
                    "status": "PASS",
                    "sample_1_ire": 24.0,
                    "sample_2_ire": 23.0,
                    "sample_3_ire": 22.0,
                    "ipp2_residual_abs_stops": 0.02,
                    "ipp2_residual_stops": 0.02,
                    "corrected_image_path": str(corrected_path),
                    "original_image_path": str(original_path),
                    "sphere_detection_note": "Sphere detection: verified",
                    "profile_note": "Profile consistent",
                },
            }
        )
        overlay_rows.append(
            {
                "clip_id": clip_id,
                "original_overlay_path": str(overlay_path),
                "corrected_overlay_path": str(overlay_path),
            }
        )
        validation_rows.append(
            {
                "clip_id": clip_id,
                "strategy_key": "median",
                "status": "PASS",
                "sample_1_ire": 24.0,
                "sample_2_ire": 23.0,
                "sample_3_ire": 22.0,
                "ipp2_residual_abs_stops": 0.02,
                "ipp2_residual_stops": 0.02,
                "corrected_image_path": str(corrected_path),
                "original_image_path": str(original_path),
            }
        )
    payload = {
        "target_type": "gray_sphere",
        "processing_mode": "both",
        "preview_transform": "REDLine IPP2 / BT.709 / BT.1886 / Medium / Medium",
        "calibration_roi": {"x": 0.2, "y": 0.2, "w": 0.4, "h": 0.4},
        "target_strategies": ["median"],
        "shared_originals": shared_originals,
        "sphere_detection_summary": {"rows": overlay_rows},
        "ipp2_validation": {
            "status_counts": {"PASS": 40, "REVIEW": 0, "FAIL": 0},
            "all_within_tolerance": True,
            "best_residual": 0.01,
            "median_residual": 0.02,
            "max_residual": 0.03,
            "rows": validation_rows,
        },
        "strategies": [
            {
                "strategy_key": "median",
                "strategy_label": "Median",
                "reference_clip_id": None,
                "target_log2_luminance": -2.7,
                "calibration_roi": {"x": 0.2, "y": 0.2, "w": 0.4, "h": 0.4},
                "clips": strategy_clips,
            }
        ],
    }
    captured: dict[str, object] = {}

    class FakeHTML:
        def __init__(self, *, filename, base_url):  # type: ignore[no-untyped-def]
            captured["filename"] = filename
            captured["base_url"] = base_url

        def write_pdf(self, target):  # type: ignore[no-untyped-def]
            captured["target"] = target
            Path(target).write_bytes(b"%PDF-1.4\n%fake weasyprint output\n")

    monkeypatch.setitem(sys.modules, "weasyprint", types.SimpleNamespace(HTML=FakeHTML))
    output_path = tmp_path / "report.pdf"
    result = render_contact_sheet_pdf(payload, output_path=str(output_path), title="R3DMatch Review Contact Sheet")
    assert result == str(output_path.resolve())
    assert output_path.exists()
    assert output_path.stat().st_size > 0
    html_path = output_path.with_suffix(".html")
    assert Path(captured["filename"]) == html_path.resolve()
    assert Path(captured["target"]) == output_path.resolve()
    assert Path(captured["base_url"]) == html_path.resolve().parent
    html = html_path.read_text(encoding="utf-8")
    assert "Array Calibration Summary" in html
    assert html.count("class='landscape-page'") >= 1
    assert html.count("class='landscape-page camera-page'") >= 5
    assert html.count("ORIGINAL + SOLVE OVERLAY") < 40


def test_render_contact_sheet_pdf_uses_final_titles_and_skips_original_color_trace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    preview_dir = tmp_path / "previews"
    original_path = preview_dir / "CAM001.original.review.run01.jpg"
    corrected_path = preview_dir / "CAM001.both.review.median.run01.jpg"
    overlay_dir = tmp_path / "report" / "review_detection_overlays"
    overlay_path = overlay_dir / "CAM001.original_detection.png"
    _write_test_preview(original_path, (90, 90, 90))
    _write_test_preview(corrected_path, (140, 120, 100))
    _write_test_preview(overlay_path, (180, 180, 180))
    payload = {
        "target_type": "gray_sphere",
        "processing_mode": "both",
        "preview_transform": "REDLine IPP2 / BT.709 / BT.1886 / Medium / Medium",
        "clip_count": 1,
        "shared_originals": [
            {
                "clip_id": "CAM001",
                "group_key": "CAM001",
                "original_frame": str(original_path),
                "measurement_valid": True,
                "gray_target_measurement_valid": True,
                "display_scalar_log2": -2.5,
                "sample_1_ire": 24.0,
                "sample_2_ire": 23.0,
                "sample_3_ire": 22.0,
                "clip_metadata": {"iso": 800, "shutter_seconds": 1 / 48, "kelvin": 5600, "tint": 2.0},
            }
        ],
        "sphere_detection_summary": {
            "rows": [
                {
                    "clip_id": "CAM001",
                    "original_overlay_path": str(overlay_path),
                    "corrected_overlay_path": str(overlay_path),
                }
            ]
        },
        "ipp2_validation": {
            "status_counts": {"PASS": 1, "REVIEW": 0, "FAIL": 0},
            "all_within_tolerance": True,
            "best_residual": 0.01,
            "median_residual": 0.02,
            "max_residual": 0.03,
        },
        "strategies": [
            {
                "strategy_key": "median",
                "strategy_label": "Median",
                "clips": [
                    {
                        "clip_id": "CAM001",
                        "group_key": "CAM001",
                        "measurement_valid": True,
                        "gray_target_measurement_valid": True,
                        "both_corrected": str(corrected_path),
                        "metrics": {
                            "exposure": {
                                "measurement_valid": True,
                                "final_offset_stops": 0.1,
                                "measured_log2_luminance_monitoring": -2.5,
                                "sample_1_ire": 24.0,
                                "sample_2_ire": 23.0,
                                "sample_3_ire": 22.0,
                            },
                            "commit_values": {"exposureAdjust": 0.1, "kelvin": 5600, "tint": 0.0},
                            "color": {"rgb_gains": [1.0, 1.0, 1.0]},
                            "confidence": 0.95,
                        },
                        "ipp2_validation": {
                            "camera_label": "CAM001",
                            "clip_id": "CAM001",
                            "status": "PASS",
                            "sample_1_ire": 24.0,
                            "sample_2_ire": 23.0,
                            "sample_3_ire": 22.0,
                            "ipp2_original_gray_exposure_summary": "Sample 1 24 / Sample 2 23 / Sample 3 22 IRE",
                            "ipp2_gray_exposure_summary": "Sample 1 24 / Sample 2 23 / Sample 3 22 IRE",
                            "ipp2_residual_abs_stops": 0.02,
                            "ipp2_residual_stops": 0.02,
                            "camera_offset_from_anchor": 0.1,
                            "suggested_action": "No adjustment needed",
                            "profile_note": "Profile consistent",
                            "sphere_detection_note": "Sphere detection: verified",
                            "corrected_image_path": str(corrected_path),
                            "original_image_path": str(original_path),
                        },
                    }
                ],
            }
        ],
    }
    output_path = tmp_path / "report.pdf"

    class FakeHTML:
        def __init__(self, *, filename, base_url):  # type: ignore[no-untyped-def]
            self.filename = filename
            self.base_url = base_url

        def write_pdf(self, target):  # type: ignore[no-untyped-def]
            Path(target).write_bytes(b"%PDF-1.4\n%fake weasyprint output\n")

    monkeypatch.setitem(sys.modules, "weasyprint", types.SimpleNamespace(HTML=FakeHTML))
    render_contact_sheet_pdf(payload, output_path=str(output_path), title="R3DMatch Review Contact Sheet")
    assert output_path.exists()
    html = output_path.with_suffix(".html").read_text(encoding="utf-8")
    assert "R3DMATCH DEBUG — NEW RENDER PATH ACTIVE" not in html
    assert "Original white balance trace" not in html
    assert "HOW TO READ THIS REPORT" in html
    assert "White-balance deviation (ΔK / ΔTint from 5600K / Tint 0)" not in html
    assert "Adjusted white-balance deviation" not in html
    assert "Target 5600K / Tint 0" not in html
    assert "<title>R3DMatch Review Contact Sheet</title>" in html


def test_render_contact_sheet_pdf_from_html_reports_actionable_dependency_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    html_path = tmp_path / "contact_sheet.html"
    html_path.write_text("<!doctype html><html><body><h1>probe</h1></body></html>", encoding="utf-8")
    monkeypatch.setitem(sys.modules, "weasyprint", None)

    with pytest.raises(RuntimeError, match="WeasyPrint plus its native cairo/pango/glib libraries"):
        render_contact_sheet_pdf_from_html(str(html_path), output_path=str(tmp_path / "contact_sheet.pdf"))


def test_contact_sheet_pdf_export_preflight_reports_interpreter_and_asset_validation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    report_dir = tmp_path / "report"
    overlay_dir = report_dir / "review_detection_overlays"
    overlay_dir.mkdir(parents=True)
    overlay_path = overlay_dir / "CAM001.png"
    overlay_path.write_bytes(b"png")
    html_path = report_dir / "contact_sheet.html"
    html_path.write_text(
        "<!doctype html><html><body><img src='review_detection_overlays/CAM001.png'></body></html>",
        encoding="utf-8",
    )
    monkeypatch.setattr("r3dmatch.report.sys.executable", "/tmp/venv/bin/python")
    monkeypatch.setenv("DYLD_FALLBACK_LIBRARY_PATH", "/opt/homebrew/lib")
    monkeypatch.setitem(sys.modules, "weasyprint", types.SimpleNamespace(HTML=object))

    preflight = contact_sheet_pdf_export_preflight(str(html_path))

    assert preflight["interpreter"] == "/tmp/venv/bin/python"
    assert preflight["dyld_fallback_library_path"] == "/opt/homebrew/lib"
    assert preflight["weasyprint_importable"] is True
    assert preflight["asset_validation"]["all_assets_exist"] is True


def test_color_deviation_chart_uses_deviation_axes_and_prominent_markers() -> None:
    svg = _contact_sheet_svg_color_deviation_chart(
        "Original white-balance deviation",
        ["CAM001", "CAM002", "CAM003"],
        [5600.0, 5900.0, 5300.0],
        [0.0, 1.5, -1.2],
    )

    assert "Target 5600K / Tint 0" in svg
    assert "Tint delta from target" in svg
    assert "Kelvin delta from target" in svg
    assert "Cooler" in svg
    assert "Warmer" in svg
    assert "Green" in svg
    assert "Magenta" in svg
    assert "stroke-width='3'" in svg


def test_contact_sheet_chromaticity_visual_uses_explicit_operator_labels() -> None:
    svg = _contact_sheet_chromaticity_visual([0.345, 0.333, 0.322])

    assert "Warm ←→ Cool" in svg
    assert "Green ←→ Magenta" in svg
    assert "Neutral" in svg


def test_report_preview_keeps_rmd_as_canonical_path_without_printmeta_validation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(command, cwd=None, env=None, text=True):  # type: ignore[no-untyped-def]
        from PIL import Image

        if "--help" in command:
            help_text = "\n".join(
                [
                    "REDline Build 65.1.3   64bit Public Release",
                    "--loadRMD <filename>",
                    "--useRMD <int>",
                    "--exposure <float>",
                    "--exposureAdjust <float>",
                    "--kelvin <int>",
                    "--tint <float>",
                    "--redGain <float>",
                    "--greenGain <float>",
                    "--blueGain <float>",
                    "--gammaCurve <int>",
                    "--colorSpace <int>",
                    "--outputToneMap <int>",
                    "--rollOff <int>",
                    "--shadow <float>",
                    "--lut <filename>",
                ]
            )
            return types.SimpleNamespace(returncode=0, stdout=help_text, stderr="")
        if "--printMeta" in command:
            return types.SimpleNamespace(returncode=1, stdout="Baseline-looking metadata only\n", stderr="")
        output_path = Path(command[command.index("--o") + 1])
        generated_path = output_path.with_name(f"{output_path.name}.000000.jpg")
        generated_path.parent.mkdir(parents=True, exist_ok=True)
        state = _fake_redline_command_state(command)
        image = _fake_redline_base_image(command)
        image = _apply_fake_redline_direct_controls(
            image,
            kelvin=cast(Optional[int], state["kelvin"]),
            tint=float(state["tint"]),
        )
        image[..., 0] = np.clip(image[..., 0] * (2.0 ** float(state["exposure"])) * float(state["red_gain"]), 0.0, 1.0)
        image[..., 1] = np.clip(image[..., 1] * (2.0 ** float(state["exposure"])) * float(state["green_gain"]), 0.0, 1.0)
        image[..., 2] = np.clip(image[..., 2] * (2.0 ** float(state["exposure"])) * float(state["blue_gain"]), 0.0, 1.0)
        image_u8 = np.clip(image * 255.0, 0.0, 255.0).astype(np.uint8)
        if output_path.suffix.lower() in {".tif", ".tiff"}:
            Image.fromarray(image_u8, mode="RGB").resize((96, 96), resample=Image.Resampling.NEAREST).save(
                generated_path,
                format="TIFF",
            )
        else:
            Image.fromarray(image_u8, mode="RGB").save(generated_path, format="JPEG")
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("r3dmatch.report.run_cancellable_subprocess", fake_run)
    def fake_write_rmd_for_clip_with_metadata(clip_id: str, payload: dict, out_dir):  # type: ignore[no-untyped-def]
        out_path = Path(out_dir) / f"{clip_id}.RMD"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        gains = payload.get("rmd_mapping", {}).get("color", {}).get("rgb_neutral_gains")
        out_path.write_text(
            json.dumps(
                {
                    "exposure": float(payload.get("rmd_mapping", {}).get("exposure", {}).get("final_offset_stops", 0.0) or 0.0),
                    "rgb_gains": gains,
                }
            ),
            encoding="utf-8",
        )
        return out_path, {"rmd_kind": "red_sdk", "settings": {"cdl_enabled": bool(gains)}}

    monkeypatch.setattr("r3dmatch.report.write_rmd_for_clip_with_metadata", fake_write_rmd_for_clip_with_metadata)
    clip_path = tmp_path / "G007_D060_0324M6_001.R3D"
    clip_path.write_bytes(b"")
    analyze_path(
        str(tmp_path),
        out_dir=str(tmp_path / "analysis-out"),
        mode="scene",
        backend="mock",
        lut_override=None,
        calibration_path=None,
        sample_count=4,
        sampling_strategy="uniform",
        calibration_roi={"x": 0.2, "y": 0.2, "w": 0.4, "h": 0.4},
    )
    build_contact_sheet_report(
        str(tmp_path / "analysis-out"),
        out_dir=str(tmp_path / "report"),
        target_type="gray_sphere",
        processing_mode="both",
        target_strategies=["median"],
    )
    preview_commands = json.loads((tmp_path / "analysis-out" / "previews" / "preview_commands.json").read_text(encoding="utf-8"))
    assert not any(command["variant"] == "exposure" and command["strategy"] == "median" for command in preview_commands["commands"])
    strategy_command = next(command for command in preview_commands["commands"] if command["variant"] == "both" and command["strategy"] == "median")
    assert strategy_command["mode"] == "corrected"
    assert strategy_command["command"] is not None
    assert "--useMeta" in strategy_command["command"]
    assert "--loadRMD" not in strategy_command["command"]
    assert strategy_command["pixel_diff_from_baseline"] > 0.0
    assert strategy_command["error"] is None
    assert strategy_command["application_method"] == "direct_redline_flags"
    assert strategy_command["validation_method"] == "pixel_diff_from_baseline"


def test_render_preview_frame_corrected_image_differs_from_baseline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_redline(monkeypatch)
    clip_path = tmp_path / "G007_D060_0324M6_001.R3D"
    clip_path.write_bytes(b"")
    settings = {
        "preview_mode": "monitoring",
        "output_space": "BT.709",
        "output_gamma": "BT.1886",
        "highlight_rolloff": "medium",
        "shadow_rolloff": "medium",
        "output_tonemap": "medium",
        "lut_path": None,
    }
    capabilities = {
        "supports_color_space": True,
        "supports_gamma_curve": True,
        "supports_output_tonemap": True,
        "supports_rolloff": True,
        "supports_shadow_control": True,
        "supports_lut": False,
    }
    baseline = render_preview_frame(
        str(clip_path),
        str(tmp_path / "baseline.jpg"),
        frame_index=0,
        redline_executable="REDLine",
        redline_capabilities=capabilities,
        preview_settings=settings,
        use_as_shot_metadata=True,
    )
    corrected = render_preview_frame(
        str(clip_path),
        str(tmp_path / "corrected.jpg"),
        frame_index=0,
        redline_executable="REDLine",
        redline_capabilities=capabilities,
        preview_settings=settings,
        use_as_shot_metadata=False,
        exposure=1.0,
        red_gain=1.2,
        green_gain=1.0,
        blue_gain=0.8,
    )
    metrics = _compute_image_difference_metrics(baseline["output_path"], corrected["output_path"])
    assert metrics["mean_absolute_difference"] is not None
    assert metrics["mean_absolute_difference"] > 0.0
    assert metrics["pixel_output_changed"] is True
    assert "--useMeta" in baseline["command"]
    assert "--useMeta" not in corrected["command"]


def test_generate_preview_stills_retries_missing_output_and_records_recovery(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    clip_path = tmp_path / "G007_D067_0403YU_001.R3D"
    clip_path.write_bytes(b"")
    attempts: dict[str, int] = {}

    def fake_render_preview_frame(  # type: ignore[no-untyped-def]
        input_r3d,
        output_path,
        *,
        frame_index,
        redline_executable,
        redline_capabilities,
        preview_settings,
        use_as_shot_metadata,
        exposure=None,
        kelvin=None,
        tint=None,
        red_gain=None,
        green_gain=None,
        blue_gain=None,
        color_cdl=None,
        rmd_path=None,
        use_rmd_mode=1,
        color_method=None,
    ):
        key = str(Path(output_path).name)
        attempts[key] = attempts.get(key, 0) + 1
        if "original" in key and attempts[key] == 1:
            return {
                "command": ["REDLine", "--i", input_r3d, "--o", output_path],
                "returncode": 0,
                "stdout": "",
                "stderr": "",
                "output_path": output_path,
            }
        pixel_value = 180 if ("exposure" in key or "both" in key) else 140
        Image.fromarray(np.full((96, 96, 3), pixel_value, dtype=np.uint8), mode="RGB").save(output_path, format="TIFF")
        return {
            "command": ["REDLine", "--i", input_r3d, "--o", output_path],
            "returncode": 0,
            "stdout": "",
            "stderr": "",
            "output_path": output_path,
        }

    monkeypatch.setattr("r3dmatch.report._resolve_redline_executable", lambda: "REDLine")
    monkeypatch.setattr("r3dmatch.report.render_preview_frame", fake_render_preview_frame)
    preview_paths = generate_preview_stills(
        str(tmp_path),
        analysis_records=[
            {
                "clip_id": "G007_D067_0403YU_001",
                "source_path": str(clip_path),
                "sample_plan": {"start_frame": 0},
            }
        ],
        previews_dir=str(tmp_path / "previews"),
        preview_settings={"preview_mode": "monitoring", "preview_still_format": "tiff"},
        redline_capabilities={},
        strategy_payloads=[
            {
                "strategy_key": "median",
                "strategy_label": "Median",
                "clips": [
                    {
                        "clip_id": "G007_D067_0403YU_001",
                        "exposure_offset_stops": 0.25,
                        "rgb_gains": None,
                        "color_cdl": None,
                        "commit_values": {},
                    }
                ],
            }
        ],
    )
    original_preview = Path(str(preview_paths["G007_D067_0403YU_001"]["original"]))
    assert original_preview.exists()
    preview_commands = json.loads((tmp_path / "previews" / "preview_commands.json").read_text(encoding="utf-8"))
    original_command = next(command for command in preview_commands["commands"] if command["variant"] == "original")
    assert original_command["attempt_count"] == 2
    assert original_command["recovered_after_retry"] is True
    assert len(original_command["attempt_diagnostics"]) == 2


def test_render_preview_frame_with_retries_rerenders_after_tiny_tiff_stub(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_path = tmp_path / "preview.tiff"
    attempts = {"count": 0}

    def fake_render_preview_frame(*args, **kwargs):  # type: ignore[no-untyped-def]
        attempts["count"] += 1
        if attempts["count"] == 1:
            output_path.write_bytes(b"II*\x00" + (b"\x00" * 160))
        else:
            Image.fromarray(np.full((64, 64, 3), 180, dtype=np.uint8), mode="RGB").save(output_path, format="TIFF")
        return {
            "command": ["REDLine", "--o", str(output_path)],
            "returncode": 0,
            "stdout": "",
            "stderr": "",
            "output_path": str(output_path),
        }

    monkeypatch.setattr("r3dmatch.report.render_preview_frame", fake_render_preview_frame)
    monkeypatch.setattr("r3dmatch.report.time.sleep", lambda _seconds: None)

    rendered = _render_preview_frame_with_retries(
        "clip.R3D",
        str(output_path),
        clip_id="CAM001",
        variant="measurement_original",
        frame_index=0,
        redline_executable="REDLine",
        redline_capabilities={},
        preview_settings={"preview_still_format": "tiff"},
        use_as_shot_metadata=True,
        max_attempts=3,
    )

    assert attempts["count"] == 2
    assert rendered["attempt_count"] == 2
    assert rendered["recovered_after_retry"] is True
    assert int(rendered["output_size_bytes"]) > 4096
    diagnostics = list(rendered["attempt_diagnostics"])
    assert diagnostics[0]["output_size_bytes"] < 4096
    assert "minimum_size_bytes" in diagnostics[0]
    assert diagnostics[1]["reason"] == "success"


def test_generate_preview_stills_raises_after_repeated_missing_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    clip_path = tmp_path / "G007_D067_0403YU_001.R3D"
    clip_path.write_bytes(b"")

    def fake_render_preview_frame(*args, **kwargs):  # type: ignore[no-untyped-def]
        output_path = str(args[1])
        return {
            "command": ["REDLine", "--o", output_path],
            "returncode": 0,
            "stdout": "",
            "stderr": "",
            "output_path": output_path,
        }

    monkeypatch.setattr("r3dmatch.report._resolve_redline_executable", lambda: "REDLine")
    monkeypatch.setattr("r3dmatch.report.render_preview_frame", fake_render_preview_frame)
    with pytest.raises(RuntimeError, match="after 3 attempt\\(s\\)"):
        generate_preview_stills(
            str(tmp_path),
            analysis_records=[
                {
                    "clip_id": "G007_D067_0403YU_001",
                    "source_path": str(clip_path),
                    "sample_plan": {"start_frame": 0},
                }
            ],
            previews_dir=str(tmp_path / "previews"),
            preview_settings={"preview_mode": "monitoring", "preview_still_format": "tiff"},
            redline_capabilities={},
            strategy_payloads=[
                {
                    "strategy_key": "median",
                    "strategy_label": "Median",
                    "clips": [
                        {
                            "clip_id": "G007_D067_0403YU_001",
                            "exposure_offset_stops": 0.25,
                            "rgb_gains": None,
                            "color_cdl": None,
                            "commit_values": {},
                        }
                    ],
                }
            ],
        )


def test_contact_sheet_strategy_clip_uses_corrected_asset_and_render_truth(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_redline(monkeypatch)
    clip_a = tmp_path / "G007_B057_0324YT_001.R3D"
    clip_b = tmp_path / "G007_C057_0324YT_001.R3D"
    clip_a.write_bytes(b"")
    clip_b.write_bytes(b"")
    analyze_path(
        str(tmp_path),
        out_dir=str(tmp_path / "analysis-out"),
        mode="scene",
        backend="mock",
        lut_override=None,
        calibration_path=None,
        sample_count=4,
        sampling_strategy="uniform",
        calibration_roi={"x": 0.2, "y": 0.2, "w": 0.4, "h": 0.4},
    )
    payload = build_contact_sheet_report(
        str(tmp_path / "analysis-out"),
        out_dir=str(tmp_path / "report"),
        target_type="gray_sphere",
        processing_mode="both",
        target_strategies=["median"],
    )
    report_json = json.loads(Path(payload["report_json"]).read_text(encoding="utf-8"))
    strategy_clip = report_json["strategies"][0]["clips"][0]
    assert strategy_clip["both_corrected"] != strategy_clip["original_frame"]
    assert Path(strategy_clip["both_corrected"]).name == Path(str(strategy_clip["render_validation"]["output"])).name
    assert strategy_clip["render_validation"]["variant"] == "both"


def test_contact_sheet_report_writes_corrected_residual_validation_from_corrected_assets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_redline(monkeypatch)
    clip_a = tmp_path / "G007_B057_0324YT_001.R3D"
    clip_b = tmp_path / "G007_C057_0324YT_001.R3D"
    clip_a.write_bytes(b"")
    clip_b.write_bytes(b"")
    analyze_path(
        str(tmp_path),
        out_dir=str(tmp_path / "analysis-out"),
        mode="scene",
        backend="mock",
        lut_override=None,
        calibration_path=None,
        sample_count=4,
        sampling_strategy="uniform",
        calibration_roi={"x": 0.2, "y": 0.2, "w": 0.4, "h": 0.4},
    )
    payload = build_contact_sheet_report(
        str(tmp_path / "analysis-out"),
        out_dir=str(tmp_path / "report"),
        target_type="gray_sphere",
        processing_mode="both",
        target_strategies=["median"],
    )
    report_json = json.loads(Path(payload["report_json"]).read_text(encoding="utf-8"))
    residual_path = Path(report_json["corrected_residual_validation_path"])
    ipp2_path = Path(report_json["ipp2_validation_path"])
    assert residual_path.exists()
    assert ipp2_path.exists()
    residual_payload = json.loads(residual_path.read_text(encoding="utf-8"))
    ipp2_payload = json.loads(ipp2_path.read_text(encoding="utf-8"))
    assert residual_payload["tolerance_model"]["pass_threshold_stops"] == pytest.approx(0.10)
    assert ipp2_payload["summary"]["tolerance_model"]["pass_threshold_stops"] == pytest.approx(0.05)
    assert residual_payload["rows"]
    row = residual_payload["rows"][0]
    assert row["measurement_geometry"] == "three_rect_windows_within_roi"
    assert Path(row["corrected_image_path"]).exists()
    strategy_clip = next(
        clip
        for clip in report_json["strategies"][0]["clips"]
        if clip["clip_id"] == row["clip_id"]
    )
    assert strategy_clip["corrected_residual_validation"]["clip_id"] == row["clip_id"]
    assert strategy_clip["ipp2_validation"]["clip_id"] == row["clip_id"]
    assert "log_vs_ipp2_residual_delta_stops" in strategy_clip["ipp2_validation"]
    assert strategy_clip["ipp2_validation"]["operator_guidance"]["status"] in {"PASS", "REVIEW", "FAIL", "OUTLIER"}
    assert "suggested_action" in strategy_clip["ipp2_validation"]
    assert "operator_status" in strategy_clip["ipp2_validation"]
    html = Path(payload["report_html"]).read_text(encoding="utf-8")
    assert "Gray Exposure" in html
    assert "aperture" in html
    assert "Recommended action:" in html
    assert "In range" in html or "Needs adjustment" in html or "Outside tolerance" in html


def test_contact_sheet_report_writes_sampling_and_solve_comparison_for_gray_sphere(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_redline(monkeypatch)
    for name in [
        "G007_A064_0325AA_001.R3D",
        "G007_B064_0325BB_001.R3D",
        "G007_C064_0325CC_001.R3D",
        "I007_A064_0325XY_001.R3D",
    ]:
        (tmp_path / name).write_bytes(b"")
    analyze_path(
        str(tmp_path),
        out_dir=str(tmp_path / "analysis-out"),
        mode="scene",
        backend="mock",
        lut_override=None,
        calibration_path=None,
        target_type="gray_sphere",
        sample_count=4,
        sampling_strategy="uniform",
        calibration_roi={"x": 0.2, "y": 0.2, "w": 0.4, "h": 0.4},
    )
    payload = build_contact_sheet_report(
        str(tmp_path / "analysis-out"),
        out_dir=str(tmp_path / "report"),
        target_type="gray_sphere",
        processing_mode="both",
        target_strategies=["median", "optimal_exposure"],
    )
    report_json = json.loads(Path(payload["report_json"]).read_text(encoding="utf-8"))
    sampling_path = Path(report_json["sampling_comparison_path"])
    solve_path = Path(report_json["solve_comparison_path"])
    assert sampling_path.exists()
    assert solve_path.exists()
    sampling_payload = json.loads(sampling_path.read_text(encoding="utf-8"))
    solve_payload = json.loads(solve_path.read_text(encoding="utf-8"))
    assert sampling_payload["row_count"] == 4
    assert any(abs(float(row["delta_log2_stops"])) > 1e-6 for row in sampling_payload["rows"])
    assert any(strategy["strategy_key"] == "median" for strategy in solve_payload["strategies"])


def test_multi_strategy_review_includes_manual_reference_and_shared_originals(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_redline(monkeypatch)
    clip_a = tmp_path / "G007_B057_0324YT_001.R3D"
    clip_b = tmp_path / "G007_C057_0324YT_001.R3D"
    clip_a.write_bytes(b"")
    clip_b.write_bytes(b"")
    analyze_path(
        str(tmp_path),
        out_dir=str(tmp_path / "analysis-out"),
        mode="scene",
        backend="mock",
        lut_override=None,
        calibration_path=None,
        sample_count=4,
        sampling_strategy="uniform",
        calibration_roi={"x": 0.2, "y": 0.2, "w": 0.4, "h": 0.4},
    )
    payload = build_contact_sheet_report(
        str(tmp_path / "analysis-out"),
        out_dir=str(tmp_path / "report"),
        target_type="gray_sphere",
        processing_mode="both",
        target_strategies=["median", "manual"],
        reference_clip_id="G007_B057_0324YT_001",
    )
    report_json = json.loads(Path(payload["report_json"]).read_text(encoding="utf-8"))
    assert len(report_json["shared_originals"]) == 2
    assert [item["strategy_key"] for item in report_json["strategies"]] == ["median", "manual"]
    manual_strategy = report_json["strategies"][1]
    assert manual_strategy["reference_clip_id"] == "G007_B057_0324YT_001"
    assert manual_strategy["clips"][0]["preview_variants"]["both"].endswith(".both.review.manual.analysis-out.tiff")
    assert report_json["shared_originals"][0]["original_frame"].endswith(".original.review.analysis-out.tiff")


def test_hero_camera_review_marks_identity_correction_and_cdl_payload(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_redline(monkeypatch)
    clip_a = tmp_path / "G007_B057_0324YT_001.R3D"
    clip_b = tmp_path / "G007_C057_0324YT_001.R3D"
    clip_a.write_bytes(b"")
    clip_b.write_bytes(b"")
    analyze_path(
        str(tmp_path),
        out_dir=str(tmp_path / "analysis-out"),
        mode="scene",
        backend="mock",
        lut_override=None,
        calibration_path=None,
        sample_count=4,
        sampling_strategy="uniform",
        calibration_roi={"x": 0.2, "y": 0.2, "w": 0.4, "h": 0.4},
    )
    payload = build_contact_sheet_report(
        str(tmp_path / "analysis-out"),
        out_dir=str(tmp_path / "report"),
        target_type="gray_sphere",
        processing_mode="both",
        target_strategies=["hero-camera"],
        hero_clip_id="G007_B057_0324YT_001",
    )
    report_json = json.loads(Path(payload["report_json"]).read_text(encoding="utf-8"))
    hero_strategy = report_json["strategies"][0]
    assert hero_strategy["strategy_key"] == "hero_camera"
    assert hero_strategy["hero_clip_id"] == "G007_B057_0324YT_001"
    hero_clip = next(item for item in hero_strategy["clips"] if item["clip_id"] == "G007_B057_0324YT_001")
    assert hero_clip["is_hero_camera"] is True
    assert hero_clip["metrics"]["exposure"]["final_offset_stops"] == pytest.approx(0.0)
    assert hero_clip["metrics"]["color"]["cdl"] is not None
    assert hero_clip["metrics"]["color"]["lift_gamma_gain_saturation"]["gain"] == pytest.approx([1.0, 1.0, 1.0])
    preview_commands = json.loads((tmp_path / "analysis-out" / "previews" / "preview_commands.json").read_text(encoding="utf-8"))
    hero_both = next(command for command in preview_commands["commands"] if command["variant"] == "both" and command["strategy"] == "hero_camera" and command["clip_id"] == "G007_B057_0324YT_001")
    assert hero_both["application_method"] == "direct_redline_flags"
    assert hero_both["preview_color_applied"] is False
    assert hero_both["cdl_enabled"] is False
    assert hero_both["rmd_path"] in {"", None}
    html = Path(payload["report_html"]).read_text(encoding="utf-8")
    assert "Hero Camera" in html
    assert "Anchor / Hero" in html
    assert "hero-badge" in html


def test_approve_master_rmd_uses_selected_manual_strategy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_redline(monkeypatch)
    clip_a = tmp_path / "G007_B057_0324YT_001.R3D"
    clip_b = tmp_path / "G007_C057_0324YT_001.R3D"
    clip_a.write_bytes(b"")
    clip_b.write_bytes(b"")
    review_dir = tmp_path / "review"
    review_calibration(
        str(tmp_path),
        out_dir=str(review_dir),
        target_type="gray_sphere",
        processing_mode="both",
        mode="scene",
        backend="mock",
        lut_override=None,
        calibration_path=None,
        exposure_calibration_path=None,
        color_calibration_path=None,
        calibration_mode=None,
        sample_count=4,
        sampling_strategy="uniform",
        calibration_roi={"x": 0.2, "y": 0.2, "w": 0.4, "h": 0.4},
        target_strategies=["median", "manual"],
        reference_clip_id="G007_B057_0324YT_001",
    )
    payload = approve_master_rmd(str(review_dir), target_strategy="manual", reference_clip_id="G007_B057_0324YT_001")
    manifest = json.loads(Path(payload["approval_manifest"]).read_text(encoding="utf-8"))
    assert manifest["selected_target_strategy"] == "manual"
    assert manifest["selected_reference_clip_id"] == "G007_B057_0324YT_001"
    assert (Path(payload["master_rmd_dir"]) / "G007_B057.RMD").exists()
    assert Path(manifest["batch_manifest"]).exists()


def test_approve_master_rmd_uses_selected_hero_strategy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_redline(monkeypatch)
    clip_a = tmp_path / "G007_B057_0324YT_001.R3D"
    clip_b = tmp_path / "G007_C057_0324YT_001.R3D"
    clip_a.write_bytes(b"")
    clip_b.write_bytes(b"")
    review_dir = tmp_path / "review-hero"
    review_calibration(
        str(tmp_path),
        out_dir=str(review_dir),
        target_type="gray_sphere",
        processing_mode="both",
        mode="scene",
        backend="mock",
        lut_override=None,
        calibration_path=None,
        exposure_calibration_path=None,
        color_calibration_path=None,
        calibration_mode=None,
        sample_count=4,
        sampling_strategy="uniform",
        calibration_roi={"x": 0.2, "y": 0.2, "w": 0.4, "h": 0.4},
        target_strategies=["hero-camera"],
        reference_clip_id=None,
        hero_clip_id="G007_B057_0324YT_001",
    )
    payload = approve_master_rmd(str(review_dir), target_strategy="hero-camera", hero_clip_id="G007_B057_0324YT_001")
    manifest = json.loads(Path(payload["approval_manifest"]).read_text(encoding="utf-8"))
    assert manifest["selected_target_strategy"] == "hero_camera"
    assert manifest["selected_hero_clip_id"] == "G007_B057_0324YT_001"
    hero_mapping = next(item for item in manifest["clip_mappings"] if item["clip_id"] == "G007_B057_0324YT_001")
    assert hero_mapping["correction_key"] == "G007_B057"
    assert hero_mapping["is_hero_camera"] is True
    assert hero_mapping["exposure_correction_stops"] == pytest.approx(0.0)
    assert Path(hero_mapping["master_rmd_path"]).name == "G007_B057.RMD"


def test_approval_manifest_preserves_subset_run_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_redline(monkeypatch)
    for name in [
        "G007_B057_0324YT_065.R3D",
        "G007_C057_0324YT_065.R3D",
    ]:
        (tmp_path / name).write_bytes(b"")
    parent_out = tmp_path / "subset-parent"
    review_payload = review_calibration(
        str(tmp_path),
        out_dir=str(parent_out),
        run_label="clip65_red_contamination",
        target_type="gray_sphere",
        processing_mode="both",
        mode="scene",
        matching_domain="scene",
        backend="mock",
        lut_override=None,
        calibration_path=None,
        exposure_calibration_path=None,
        color_calibration_path=None,
        calibration_mode=None,
        sample_count=4,
        sampling_strategy="uniform",
        calibration_roi={"x": 0.2, "y": 0.2, "w": 0.4, "h": 0.4},
        selected_clip_ids=["G007_B057_0324YT_065", "G007_C057_0324YT_065"],
        selected_clip_groups=None,
        clip_subset_file=None,
        target_strategies=["hero-camera"],
        reference_clip_id=None,
        hero_clip_id="G007_B057_0324YT_065",
    )
    approval_payload = approve_master_rmd(
        review_payload["analysis_dir"],
        target_strategy="hero-camera",
        hero_clip_id="G007_B057_0324YT_065",
    )
    approval_manifest = json.loads(Path(approval_payload["approval_manifest"]).read_text(encoding="utf-8"))
    batch_manifest = json.loads(Path(approval_payload["batch_manifest"]).read_text(encoding="utf-8"))
    assert approval_manifest["run_label"] == "clip65_red_contamination"
    assert approval_manifest["matching_domain"] == "perceptual"
    assert approval_manifest["selected_clip_ids"] == ["G007_B057_0324YT_065", "G007_C057_0324YT_065"]
    assert batch_manifest["run_label"] == "clip65_red_contamination"
    assert batch_manifest["matching_domain"] == "perceptual"
    assert all(entry["approved_run_label"] == "clip65_red_contamination" for entry in batch_manifest["entries"])


def test_clear_preview_cache_only_removes_review_artifacts(tmp_path: Path) -> None:
    analysis_root = tmp_path / "analysis-out"
    previews = analysis_root / "previews"
    report_dir = analysis_root / "report"
    approval_dir = analysis_root / "approval"
    previews.mkdir(parents=True)
    report_dir.mkdir(parents=True)
    approval_dir.mkdir(parents=True)
    (analysis_root / "summary.json").write_text("{}", encoding="utf-8")
    (analysis_root / "array_calibration.json").write_text("{}", encoding="utf-8")
    (previews / "G007_D060_0324M6_001.original.review.jpg").write_text("x", encoding="utf-8")
    (previews / "preview_commands.json").write_text("{}", encoding="utf-8")
    (report_dir / "contact_sheet.json").write_text("{}", encoding="utf-8")
    (report_dir / "contact_sheet.html").write_text("{}", encoding="utf-8")
    (approval_dir / "calibration_report.pdf").write_text("pdf", encoding="utf-8")
    (approval_dir / "MasterRMD").mkdir()
    payload = clear_preview_cache(str(analysis_root))
    assert payload["removed_count"] >= 3
    assert not (previews / "G007_D060_0324M6_001.original.review.jpg").exists()
    assert not (report_dir / "contact_sheet.html").exists()
    assert (analysis_root / "summary.json").exists()
    assert (analysis_root / "array_calibration.json").exists()
    assert (approval_dir / "calibration_report.pdf").exists()
    assert (approval_dir / "MasterRMD").exists()


def test_approve_master_rmd_creates_manifest_pdf_and_exact_names(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_redline(monkeypatch)
    clip_path = tmp_path / "G007_D060_0324M6_001.R3D"
    clip_path.write_bytes(b"")
    analyze_path(
        str(tmp_path),
        out_dir=str(tmp_path / "analysis-out"),
        mode="scene",
        backend="mock",
        lut_override=None,
        calibration_path=None,
        sample_count=4,
        sampling_strategy="uniform",
        calibration_roi={"x": 0.2, "y": 0.2, "w": 0.4, "h": 0.4},
    )
    build_contact_sheet_report(
        str(tmp_path / "analysis-out"),
        out_dir=str(tmp_path / "analysis-out" / "report"),
        target_type="gray_sphere",
        processing_mode="both",
    )
    payload = approve_master_rmd(str(tmp_path / "analysis-out"))
    assert Path(payload["approval_manifest"]).exists()
    assert Path(payload["calibration_report_pdf"]).exists()
    assert (Path(payload["master_rmd_dir"]) / "G007_D060.RMD").exists()
    assert Path(payload["batch_manifest"]).exists()
    assert Path(payload["batch_scripts"]["sh"]).exists()
    assert Path(payload["batch_scripts"]["tcsh"]).exists()
    assert Path(payload["batch_readme"]).exists()


def test_approve_master_rmd_writes_commit_package_with_commit_values(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_redline(monkeypatch)
    for name in [
        "G007_B057_0324YT_001.R3D",
        "G007_C057_0324GR_001.R3D",
    ]:
        (tmp_path / name).write_bytes(b"")
    review_dir = tmp_path / "review-commit"
    review_calibration(
        str(tmp_path),
        out_dir=str(review_dir),
        target_type="gray_sphere",
        processing_mode="both",
        mode="scene",
        backend="mock",
        lut_override=None,
        calibration_path=None,
        exposure_calibration_path=None,
        color_calibration_path=None,
        calibration_mode=None,
        sample_count=4,
        sampling_strategy="uniform",
        calibration_roi={"x": 0.2, "y": 0.2, "w": 0.4, "h": 0.4},
        target_strategies=["median"],
        reference_clip_id=None,
        source_mode="local_folder",
    )
    payload = approve_master_rmd(str(review_dir))
    commit_package_path = Path(payload["calibration_commit_package"])
    assert commit_package_path.exists()
    commit_package = json.loads(commit_package_path.read_text(encoding="utf-8"))
    assert commit_package["schema_version"] == "r3dmatch_commit_package_v1"
    assert commit_package["source_mode"] == "local_folder"
    first = commit_package["per_camera_values"][0]
    assert set(first["commit_values"]) >= {"exposureAdjust", "kelvin", "tint"}
    assert Path(first["master_rmd_path"]).name.endswith(".RMD")


def test_review_commit_payload_includes_inventory_camera_targets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_redline(monkeypatch)
    for path in [
        tmp_path / "GA" / "G007_A065_0325F6.RDC" / "G007_A065_0325F6_001.R3D",
        tmp_path / "GB" / "G007_B065_0325PN.RDC" / "G007_B065_0325PN_001.R3D",
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")
    review_dir = tmp_path / "review-rcp2"
    review_calibration(
        str(tmp_path),
        out_dir=str(review_dir),
        target_type="gray_sphere",
        processing_mode="both",
        mode="scene",
        backend="mock",
        lut_override=None,
        calibration_path=None,
        exposure_calibration_path=None,
        color_calibration_path=None,
        calibration_mode=None,
        sample_count=4,
        sampling_strategy="uniform",
        calibration_roi={"x": 0.2, "y": 0.2, "w": 0.4, "h": 0.4},
        target_strategies=["median"],
        reference_clip_id=None,
        source_mode="local_folder",
    )
    validation = json.loads((review_dir / "report" / "review_validation.json").read_text(encoding="utf-8"))
    aggregate = json.loads(Path(validation["commit_payload"]["aggregate_path"]).read_text(encoding="utf-8"))
    assert aggregate["camera_targets"][0]["inventory_camera_label"] == "GA"
    assert aggregate["camera_targets"][0]["inventory_camera_ip"].startswith("172.20.114.")
    per_camera = json.loads(Path(validation["commit_payload"]["per_camera_payloads"][0]["path"]).read_text(encoding="utf-8"))
    assert per_camera["inventory_camera_label"] == "GA"
    assert per_camera["inventory_camera_ip"].startswith("172.20.114.")


def test_plan_ftps_request_normalizes_reel_clips_and_camera_subset() -> None:
    plan = plan_ftps_request(reel_identifier="7", clip_spec="63,64-65", requested_cameras=["aa", "KB"])
    assert plan["reel_identifier"] == "007"
    assert plan["clip_numbers"] == [63, 64, 65]
    assert plan["clip_spec"] == "63-65"
    assert plan["requested_cameras"] == ["AA", "KB"]


def test_ingest_ftps_batch_writes_manifest_and_partial_status(tmp_path: Path) -> None:
    class FakeFTP:
        trees = {
            "172.20.114.141": {
                "/": [("media", "dir")],
                "/media": [("G007_AA63_032563.RDC", "dir")],
                "/media/G007_AA63_032563.RDC": [("G007_AA63_032563_001.R3D", "file")],
            },
            "172.20.114.142": {
                "/": [],
            },
        }

        def __init__(self) -> None:
            self.host = ""

        def connect(self, host: str, port: int, timeout: float) -> None:
            self.host = host

        def login(self, user: str, passwd: str) -> None:
            return None

        def prot_p(self) -> None:
            return None

        def mlsd(self, remote_dir: str):
            entries = self.trees[self.host].get(remote_dir, [])
            return [(name, {"type": entry_type}) for name, entry_type in entries]

        def retrbinary(self, command: str, callback) -> None:
            callback(f"mock:{self.host}:{command}".encode("utf-8"))

        def quit(self) -> None:
            return None

        def close(self) -> None:
            return None

    manifest = ingest_ftps_batch(
        out_dir=str(tmp_path / "ingest"),
        reel_identifier="007",
        clip_spec="63",
        requested_cameras=["AA", "AB"],
        ftp_factory=FakeFTP,
    )
    manifest_path = tmp_path / "ingest" / "ingest_manifest.json"
    assert manifest["status"] == "partial"
    assert manifest_path.exists()
    saved = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert saved["successful_cameras"] == ["AA"]
    assert saved["failed_cameras"] == ["AB"]
    assert saved["clips_found"] == 1
    assert saved["status"] == "partial"
    assert saved["manifest_path"] == str(manifest_path)
    assert saved["requested_camera_ips"]["AA"] == "172.20.114.141"
    assert saved["per_camera_status"][1]["failure_code"] == "no_matching_clips"
    downloaded = Path(saved["per_camera_status"][0]["downloaded_files"][0]["local_path"])
    assert downloaded.exists()
    assert downloaded.read_bytes().startswith(b"mock:172.20.114.141")


def test_discover_ftps_batch_reports_reachable_unreachable_and_matches(tmp_path: Path) -> None:
    class FakeFTP:
        trees = {
            "172.20.114.141": {
                "/": [("media", "dir")],
                "/media": [("G007_AA63_032563.RDC", "dir")],
                "/media/G007_AA63_032563.RDC": [("G007_AA63_032563_001.R3D", "file")],
            },
        }

        def __init__(self) -> None:
            self.host = ""

        def connect(self, host: str, port: int, timeout: float) -> None:
            if host == "172.20.114.142":
                raise OSError("camera offline")
            self.host = host

        def login(self, user: str, passwd: str) -> None:
            return None

        def prot_p(self) -> None:
            return None

        def mlsd(self, remote_dir: str):
            entries = self.trees[self.host].get(remote_dir, [])
            return [(name, {"type": entry_type}) for name, entry_type in entries]

        def size(self, remote_path: str) -> int:
            return 1024

        def quit(self) -> None:
            return None

        def close(self) -> None:
            return None

    manifest = discover_ftps_batch(
        out_dir=str(tmp_path / "ingest"),
        reel_identifier="007",
        clip_spec="63",
        requested_cameras=["AA", "AB"],
        ftp_factory=FakeFTP,
    )
    assert manifest["action"] == "discover"
    assert manifest["reachable_cameras"] == ["AA"]
    assert manifest["unreachable_cameras"] == ["AB"]
    assert manifest["matched_file_count"] == 1
    assert manifest["estimated_bytes"] == 1024
    assert Path(manifest["manifest_path"]).exists()


def test_retry_failed_ftps_batch_reuses_previous_manifest_camera_subset(tmp_path: Path) -> None:
    ingest_root = tmp_path / "ingest"
    ingest_root.mkdir()
    previous = {
        "schema_version": "r3dmatch_ftps_ingest_v2",
        "manifest_path": str(ingest_manifest_path_for(ingest_root)),
        "reel_identifier": "007",
        "clip_spec": "63",
        "failed_cameras": ["AB"],
        "unreachable_cameras": ["AB"],
        "per_camera_status": [{"camera_label": "AB", "status": "unreachable"}],
    }
    ingest_manifest_path_for(ingest_root).write_text(json.dumps(previous), encoding="utf-8")
    captured: dict[str, object] = {}

    def fake_download(**kwargs):  # type: ignore[no-untyped-def]
        captured.update(kwargs)
        return {"status": "success"}

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr("r3dmatch.ftps_ingest.download_ftps_batch", fake_download)
    try:
        payload = retry_failed_ftps_batch(out_dir=str(ingest_root))
    finally:
        monkeypatch.undo()
    assert payload["status"] == "success"
    assert captured["requested_cameras"] == ["AB"]
    assert captured["reel_identifier"] == "007"
    assert captured["clip_spec"] == "63"


def test_apply_calibration_payload_dry_run_uses_inventory_targets(tmp_path: Path) -> None:
    payload = {
        "schema_version": "r3dmatch_rcp2_ready_v1",
        "camera_id": "G007_A065",
        "clip_id": "G007_A065_0325F6_001",
        "inventory_camera_label": "GA",
        "inventory_camera_ip": "172.20.114.165",
        "format": "rcp2_ready",
        "calibration": {"exposureAdjust": -0.821145, "kelvin": 5663, "tint": -0.2},
        "confidence": 0.68,
        "is_hero_camera": False,
        "notes": [],
    }
    payload_path = tmp_path / "GA.json"
    payload_path.write_text(json.dumps(payload), encoding="utf-8")
    result = apply_calibration_payload(str(payload_path), out_path=str(tmp_path / "apply.json"))
    assert result["status"] == "dry_run"
    assert result["transport_mode"] == "dry_run"
    assert result["results"][0]["inventory_camera_label"] == "GA"
    assert result["results"][0]["status"] == "dry_run"
    assert result["results"][0]["readback"]["kelvin"] == 5663
    assert Path(result["report_path"]).exists()
    assert Path(result["post_apply_verification_path"]).exists()


def test_apply_calibration_payload_enforces_kelvin_tint_exposure_order(tmp_path: Path) -> None:
    payload = {
        "schema_version": "r3dmatch_rcp2_ready_v1",
        "camera_id": "G007_A065",
        "clip_id": "G007_A065_0325F6_001",
        "inventory_camera_label": "GA",
        "inventory_camera_ip": "172.20.114.165",
        "format": "rcp2_ready",
        "calibration": {"exposureAdjust": -0.75, "kelvin": 5663, "tint": -0.2},
        "confidence": 0.68,
        "is_hero_camera": False,
        "notes": [],
    }
    payload_path = tmp_path / "GA-order.json"
    payload_path.write_text(json.dumps(payload), encoding="utf-8")

    class OrderedTransport:
        mode = "live"

        def __init__(self) -> None:
            self.writes = []

        def open(self, *, host: str, port: int, camera_label: str) -> dict:
            return {"host": host, "port": port, "camera_label": camera_label, "state": {"exposureAdjust": 0.0, "kelvin": 5600, "tint": 0.0}}

        def close(self, session: object) -> None:
            return None

        def read_state(self, session: object) -> dict:
            return dict((session or {}).get("state") or {})

        def write_state(self, session: object, values: dict) -> None:
            self.writes.append(dict(values))
            session["state"] = dict(values)

    transport = OrderedTransport()
    result = apply_calibration_payload(str(payload_path), transport=transport)
    row = result["results"][0]
    assert row["status"] == "applied_successfully"
    assert [step["field_name"] for step in row["camera_verification"]["write_sequence_trace"]] == ["kelvin", "tint", "exposureAdjust"]
    assert transport.writes[0] == {"exposureAdjust": 0.0, "kelvin": 5663, "tint": 0.0}
    assert transport.writes[1] == {"exposureAdjust": 0.0, "kelvin": 5663, "tint": -0.2}
    assert transport.writes[2] == {"exposureAdjust": -0.75, "kelvin": 5663, "tint": -0.2}


def test_apply_calibration_payload_stops_on_first_field_verification_failure(tmp_path: Path) -> None:
    payload = {
        "schema_version": "r3dmatch_rcp2_ready_v1",
        "camera_id": "H007_D065",
        "clip_id": "H007_D065_0325CS_001",
        "inventory_camera_label": "HD",
        "inventory_camera_ip": "172.20.114.172",
        "format": "rcp2_ready",
        "calibration": {"exposureAdjust": 0.9, "kelvin": 5663, "tint": 0.7},
        "confidence": 0.71,
        "is_hero_camera": False,
        "notes": [],
    }
    payload_path = tmp_path / "HD-stop.json"
    payload_path.write_text(json.dumps(payload), encoding="utf-8")
    transport = MemoryRcp2Transport(
        initial_states={"172.20.114.172": {"exposureAdjust": 0.0, "kelvin": 5600, "tint": 0.0}},
        readback_overrides={"172.20.114.172": {"exposureAdjust": 0.0, "kelvin": 5900, "tint": 0.0}},
    )
    result = apply_calibration_payload(str(payload_path), transport=transport)
    row = result["results"][0]
    assert row["status"] == "mismatch_after_writeback"
    assert row["camera_verification"]["verification_summary"]["mismatched_fields"] == ["kelvin"]
    assert [step["field_name"] for step in row["camera_verification"]["write_sequence_trace"]] == ["kelvin"]
    assert row["verification"]["mismatched_fields"] == ["kelvin"]


def test_apply_calibration_payload_reports_readback_mismatch(tmp_path: Path) -> None:
    payload = {
        "schema_version": "r3dmatch_rcp2_ready_v1",
        "camera_id": "H007_D065",
        "clip_id": "H007_D065_0325CS_001",
        "inventory_camera_label": "HD",
        "inventory_camera_ip": "172.20.114.172",
        "format": "rcp2_ready",
        "calibration": {"exposureAdjust": 0.9, "kelvin": 5663, "tint": 0.7},
        "confidence": 0.71,
        "is_hero_camera": False,
        "notes": [],
    }
    payload_path = tmp_path / "HD.json"
    payload_path.write_text(json.dumps(payload), encoding="utf-8")
    transport = MemoryRcp2Transport(readback_overrides={"172.20.114.172": {"exposureAdjust": 0.9, "kelvin": 5900, "tint": 0.7}})
    result = apply_calibration_payload(str(payload_path), transport=transport)
    assert result["status"] == "failed"
    assert result["results"][0]["status"] == "mismatch_after_writeback"
    assert result["results"][0]["verification"]["mismatched_fields"] == ["kelvin"]
    assert result["results"][0]["camera_verification"]["verification_summary"]["mismatched_fields"] == ["kelvin"]


def test_apply_calibration_payload_retries_once_after_timeout_then_succeeds(tmp_path: Path) -> None:
    payload = {
        "schema_version": "r3dmatch_rcp2_ready_v1",
        "camera_id": "G007_A065",
        "clip_id": "G007_A065_0325F6_001",
        "inventory_camera_label": "GA",
        "inventory_camera_ip": "172.20.114.165",
        "format": "rcp2_ready",
        "calibration": {"exposureAdjust": -0.821145, "kelvin": 5663, "tint": -0.2},
        "confidence": 0.68,
        "is_hero_camera": False,
        "notes": [],
    }
    payload_path = tmp_path / "GA.json"
    payload_path.write_text(json.dumps(payload), encoding="utf-8")

    class RetryOnceTransport:
        mode = "live"

        def __init__(self) -> None:
            self.open_count = 0

        def open(self, *, host: str, port: int, camera_label: str) -> dict:
            self.open_count += 1
            if self.open_count == 1:
                raise Rcp2TimeoutError(f"timeout talking to {camera_label}")
            return {"host": host, "port": port, "camera_label": camera_label, "state": {}}

        def close(self, session: object) -> None:
            return None

        def read_state(self, session: object) -> dict:
            return dict((session or {}).get("state") or {})

        def write_state(self, session: object, values: dict) -> None:
            session["state"] = dict(values)

    result = apply_calibration_payload(str(payload_path), transport=RetryOnceTransport(), retry_count=1)
    assert result["status"] == "success"
    assert result["results"][0]["status"] == "applied_successfully"
    assert len(result["results"][0]["attempts"]) == 2
    assert result["results"][0]["attempts"][0]["failure_reason"] == "timeout"
    assert result["results"][0]["attempts"][1]["status"] == "applied_successfully"


def test_apply_calibration_payload_records_timeout_field_verification(tmp_path: Path) -> None:
    payload = {
        "schema_version": "r3dmatch_rcp2_ready_v1",
        "camera_id": "G007_A065",
        "clip_id": "G007_A065_0325F6_001",
        "inventory_camera_label": "GA",
        "inventory_camera_ip": "172.20.114.165",
        "format": "rcp2_ready",
        "calibration": {"exposureAdjust": -0.821145, "kelvin": 5663, "tint": -0.2},
        "confidence": 0.68,
        "is_hero_camera": False,
        "notes": [],
    }
    payload_path = tmp_path / "GA-timeout-field.json"
    payload_path.write_text(json.dumps(payload), encoding="utf-8")

    class TimeoutFieldTransport:
        mode = "live"

        def open(self, *, host: str, port: int, camera_label: str) -> dict:
            return {"host": host, "port": port, "camera_label": camera_label}

        def close(self, session: object) -> None:
            return None

        def read_state(self, session: object) -> dict:
            return {"exposureAdjust": 0.0, "kelvin": 5600, "tint": 0.0}

        def write_state(self, session: object, values: dict) -> None:
            return None

        def apply_state_transactionally(self, session: object, values: dict, **kwargs) -> dict:
            del session, values, kwargs
            return {
                "before_state": {"exposureAdjust": 0.0, "kelvin": 5600, "tint": 0.0},
                "final_readback": {"exposureAdjust": 0.0, "kelvin": 5600, "tint": 0.0},
                "per_field": {
                    "kelvin": {
                        "field_name": "kelvin",
                        "requested_value": 5663,
                        "applied_value": None,
                        "delta": None,
                        "verification_status": "TIMEOUT",
                    }
                },
                "write_sequence_trace": [{"field_name": "kelvin", "verification_status": "TIMEOUT"}],
                "verification_summary": {
                    "mismatched_fields": [],
                    "timeout_fields": ["kelvin"],
                    "unavailable_fields": [],
                    "exact_match_fields": [],
                    "within_tolerance_fields": [],
                    "all_verified": False,
                    "final_status": "TIMEOUT",
                },
                "final_status": "TIMEOUT",
                "histogram_guard": {"enabled": False, "status": "not_run"},
                "clip_metadata_cross_check": {"status": "not_run"},
            }

    result = apply_calibration_payload(str(payload_path), transport=TimeoutFieldTransport(), retry_count=0)
    row = result["results"][0]
    assert row["status"] == "timeout"
    assert row["failure_reason"] == "timeout"
    assert row["camera_verification"]["verification_summary"]["timeout_fields"] == ["kelvin"]


def test_apply_calibration_payload_reports_timeout_and_generates_full_report(tmp_path: Path) -> None:
    payload = {
        "schema_version": "r3dmatch_rcp2_ready_v1",
        "camera_id": "G007_A065",
        "clip_id": "G007_A065_0325F6_001",
        "inventory_camera_label": "GA",
        "inventory_camera_ip": "172.20.114.165",
        "format": "rcp2_ready",
        "calibration": {"exposureAdjust": -0.821145, "kelvin": 5663, "tint": -0.2},
        "confidence": 0.68,
        "is_hero_camera": False,
        "notes": [],
    }
    payload_path = tmp_path / "GA.json"
    payload_path.write_text(json.dumps(payload), encoding="utf-8")

    class TimeoutTransport:
        mode = "live"

        def open(self, *, host: str, port: int, camera_label: str) -> dict:
            raise Rcp2TimeoutError(f"connect timeout to {host}:{port}")

        def close(self, session: object) -> None:
            return None

        def read_state(self, session: object) -> dict:
            raise AssertionError("read_state should not be called after a failed open")

        def write_state(self, session: object, values: dict) -> None:
            raise AssertionError("write_state should not be called after a failed open")

    out_path = tmp_path / "apply-timeout.json"
    result = apply_calibration_payload(str(payload_path), transport=TimeoutTransport(), retry_count=1, out_path=str(out_path))
    assert result["status"] == "failed"
    assert result["results"][0]["status"] == "timeout"
    assert result["results"][0]["failure_reason"] == "timeout"
    assert len(result["results"][0]["attempts"]) == 2
    assert result["operator_status"] == "failed"
    assert out_path.exists()


def test_apply_calibration_payload_rejects_invalid_payload_before_transport(tmp_path: Path) -> None:
    payload = {
        "schema_version": "r3dmatch_rcp2_ready_v1",
        "camera_id": "G007_A065",
        "clip_id": "G007_A065_0325F6_001",
        "inventory_camera_label": "GA",
        "inventory_camera_ip": "172.20.114.165",
        "format": "rcp2_ready",
        "calibration": {"exposureAdjust": 9.5, "kelvin": 90000, "tint": 99.0},
        "confidence": 0.68,
        "is_hero_camera": False,
        "notes": [],
    }
    payload_path = tmp_path / "GA-invalid.json"
    payload_path.write_text(json.dumps(payload), encoding="utf-8")

    class NoTouchTransport:
        mode = "live"

        def open(self, *, host: str, port: int, camera_label: str) -> dict:
            raise AssertionError("transport should not be touched for invalid payloads")

        def close(self, session: object) -> None:
            return None

        def read_state(self, session: object) -> dict:
            raise AssertionError("transport should not be touched for invalid payloads")

        def write_state(self, session: object, values: dict) -> None:
            raise AssertionError("transport should not be touched for invalid payloads")

    result = apply_calibration_payload(str(payload_path), transport=NoTouchTransport())
    assert result["status"] == "failed"
    assert result["results"][0]["status"] == "invalid_payload"
    assert result["results"][0]["failure_reason"] == "invalid_payload"
    assert "outside the safe range" in result["results"][0]["error"]


def test_apply_calibration_payload_isolates_partial_success_by_camera(tmp_path: Path) -> None:
    payloads_dir = tmp_path / "payloads"
    payloads_dir.mkdir(parents=True, exist_ok=True)
    ga_path = payloads_dir / "GA.json"
    ga_path.write_text(
        json.dumps(
            {
                "schema_version": "r3dmatch_rcp2_ready_v1",
                "camera_id": "G007_A065",
                "clip_id": "G007_A065_0325F6_001",
                "inventory_camera_label": "GA",
                "inventory_camera_ip": "172.20.114.165",
                "format": "rcp2_ready",
                "calibration": {"exposureAdjust": -0.82, "kelvin": 5663, "tint": -0.2},
                "confidence": 0.68,
                "is_hero_camera": False,
                "notes": [],
            }
        ),
        encoding="utf-8",
    )
    ic_path = payloads_dir / "IC.json"
    ic_path.write_text(
        json.dumps(
            {
                "schema_version": "r3dmatch_rcp2_ready_v1",
                "camera_id": "I007_C065",
                "clip_id": "I007_C065_0325WK_001",
                "inventory_camera_label": "IC",
                "inventory_camera_ip": "172.20.114.175",
                "format": "rcp2_ready",
                "calibration": {"exposureAdjust": -0.15, "kelvin": 5663, "tint": -0.8},
                "confidence": 0.91,
                "is_hero_camera": True,
                "notes": [],
            }
        ),
        encoding="utf-8",
    )
    aggregate_path = tmp_path / "commit.json"
    aggregate_path.write_text(
        json.dumps(
            {
                "schema_version": "r3dmatch_calibration_commit_payload_v1",
                "per_camera_payloads": [
                    {"path": str(ga_path)},
                    {"path": str(ic_path)},
                ],
            }
        ),
        encoding="utf-8",
    )

    class PartialTransport:
        mode = "live"

        def open(self, *, host: str, port: int, camera_label: str) -> dict:
            if host == "172.20.114.175":
                raise Rcp2TimeoutError(f"connect timeout to {host}:{port}")
            return {"host": host, "port": port, "camera_label": camera_label, "state": {}}

        def close(self, session: object) -> None:
            return None

        def read_state(self, session: object) -> dict:
            return dict((session or {}).get("state") or {})

        def write_state(self, session: object, values: dict) -> None:
            session["state"] = dict(values)

    result = apply_calibration_payload(str(aggregate_path), transport=PartialTransport(), retry_count=0)
    assert result["status"] == "partial"
    assert result["status_counts"]["applied_successfully"] == 1
    assert result["status_counts"]["timeout"] == 1
    assert {row["inventory_camera_label"]: row["status"] for row in result["results"]} == {
        "GA": "applied_successfully",
        "IC": "timeout",
    }


def test_cli_apply_calibration_dry_run_writes_report(tmp_path: Path) -> None:
    payload = {
        "schema_version": "r3dmatch_rcp2_ready_v1",
        "camera_id": "I007_C065",
        "clip_id": "I007_C065_0325WK_001",
        "inventory_camera_label": "IC",
        "inventory_camera_ip": "172.20.114.175",
        "format": "rcp2_ready",
        "calibration": {"exposureAdjust": 0.0, "kelvin": 5663, "tint": -0.1},
        "confidence": 0.92,
        "is_hero_camera": True,
        "notes": [],
    }
    payload_path = tmp_path / "IC.json"
    payload_path.write_text(json.dumps(payload), encoding="utf-8")
    report_path = tmp_path / "apply-report.json"
    result = runner.invoke(app, ["apply-calibration", str(payload_path), "--out", str(report_path)])
    assert result.exit_code == 0
    saved = json.loads(report_path.read_text(encoding="utf-8"))
    assert saved["status"] == "dry_run"
    assert saved["results"][0]["inventory_camera_label"] == "IC"


def test_cli_apply_calibration_live_forwards_live_flag_and_sdk_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = {
        "schema_version": "r3dmatch_rcp2_ready_v1",
        "camera_id": "G007_A065",
        "clip_id": "G007_A065_0325F6_001",
        "inventory_camera_label": "GA",
        "inventory_camera_ip": "172.20.114.165",
        "format": "rcp2_ready",
        "calibration": {"exposureAdjust": -0.1, "kelvin": 5663, "tint": -0.2},
        "confidence": 0.68,
        "is_hero_camera": False,
        "notes": [],
    }
    payload_path = tmp_path / "GA.json"
    payload_path.write_text(json.dumps(payload), encoding="utf-8")
    report_path = tmp_path / "apply-live-report.json"
    calls = {}

    def fake_apply_calibration_payload(payload_path: str, **kwargs):
        calls["payload_path"] = payload_path
        calls["kwargs"] = dict(kwargs)
        report = {
            "status": "failed",
            "transport_mode": "live",
            "results": [
                {
                    "inventory_camera_label": "GA",
                    "status": "failed_to_connect",
                    "error": "connect timeout",
                }
            ],
            "report_path": kwargs.get("out_path"),
        }
        Path(str(kwargs["out_path"])).write_text(json.dumps(report), encoding="utf-8")
        return report

    monkeypatch.setattr("r3dmatch.cli.apply_calibration_payload", fake_apply_calibration_payload)
    result = runner.invoke(
        app,
        [
            "apply-calibration",
            str(payload_path),
            "--live",
            "--sdk-root",
            "/tmp/fake_sdk",
            "--out",
            str(report_path),
        ],
    )
    assert result.exit_code == 0
    assert calls["payload_path"] == str(payload_path)
    assert calls["kwargs"]["live"] is True
    assert calls["kwargs"]["sdk_root"] == "/tmp/fake_sdk"
    assert calls["kwargs"]["transport_kind"] == "websocket"
    saved = json.loads(report_path.read_text(encoding="utf-8"))
    assert saved["transport_mode"] == "live"
    assert saved["results"][0]["status"] == "failed_to_connect"


def test_cli_read_camera_state_uses_helper(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    del tmp_path
    calls = {}

    def fake_read_camera_state(**kwargs):
        calls["kwargs"] = dict(kwargs)
        return {
            "schema_version": "r3dmatch_rcp2_camera_state_v1",
            "transport_mode": "live_websocket",
            "state": {"exposureAdjust": 0.0, "kelvin": 5600, "tint": 0.0},
        }

    monkeypatch.setattr("r3dmatch.cli.read_camera_state", fake_read_camera_state)
    result = runner.invoke(
        app,
        [
            "read-camera-state",
            "10.20.61.191",
            "--camera-label",
            "KOMODO",
        ],
    )
    assert result.exit_code == 0
    assert calls["kwargs"]["host"] == "10.20.61.191"
    assert calls["kwargs"]["port"] == 9998
    assert calls["kwargs"]["transport_kind"] == "websocket"


def test_cli_test_rcp2_write_uses_helper(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {}

    def fake_test_rcp2_write_smoke(**kwargs):
        calls["kwargs"] = dict(kwargs)
        return {
            "schema_version": "r3dmatch_rcp2_write_smoke_v1",
            "transport_mode": "live_websocket",
            "restored_matches_original": True,
        }

    monkeypatch.setattr("r3dmatch.cli.test_rcp2_write_smoke", fake_test_rcp2_write_smoke)
    result = runner.invoke(app, ["test-rcp2-write", "10.20.61.191"])
    assert result.exit_code == 0
    assert calls["kwargs"]["host"] == "10.20.61.191"
    assert calls["kwargs"]["port"] == 9998
    assert calls["kwargs"]["field_name"] == "exposureAdjust"
    assert calls["kwargs"]["transport_kind"] == "websocket"


def test_websocket_transaction_uses_ordered_set_get_verify_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    transport = WebSocketRcp2Transport()
    session = Rcp2WebSocketSession(sock=object(), host="10.0.0.1", port=9998)
    current = {
        "kelvin": Rcp2ParameterState("COLOR_TEMPERATURE", 5600, 1, 5600.0, "5600", {"divider": 1, "min": 2000, "max": 10000}, []),
        "tint": Rcp2ParameterState("TINT", 0, 1000, 0.0, "0.0", {"divider": 1000, "min": -10000, "max": 10000}, []),
        "exposureAdjust": Rcp2ParameterState("EXPOSURE_ADJUST", 0, 1000, 0.0, "0.0", {"divider": 1000, "min": -3000, "max": 3000}, []),
    }
    call_order = []

    def fake_read_state(_session: object) -> dict:
        return {
            "exposureAdjust": current["exposureAdjust"].value,
            "kelvin": int(round(current["kelvin"].value)),
            "tint": current["tint"].value,
        }

    def fake_read_parameter_state(_session: object, field_name: str) -> Rcp2ParameterState:
        call_order.append(("get", field_name))
        return current[field_name]

    def fake_write_parameter_state(_session: object, field_name: str, raw_value: int) -> Rcp2ParameterState:
        call_order.append(("set", field_name, raw_value))
        state = current[field_name]
        current[field_name] = Rcp2ParameterState(
            parameter_id=normalize_parameter_id(state.parameter_id),
            raw_value=raw_value,
            divider=state.divider,
            value=float(raw_value) / float(state.divider),
            display=str(float(raw_value) / float(state.divider)),
            edit_info=state.edit_info,
            messages=[],
        )
        return current[field_name]

    monkeypatch.setattr(transport, "read_state", fake_read_state)
    monkeypatch.setattr(transport, "_read_parameter_state", fake_read_parameter_state)
    monkeypatch.setattr(transport, "_write_parameter_state", fake_write_parameter_state)

    result = transport.apply_state_transactionally(
        session,
        {"exposureAdjust": -0.25, "kelvin": 5663, "tint": -0.2},
    )
    assert [step["field_name"] for step in result["write_sequence_trace"]] == ["kelvin", "tint", "exposureAdjust"]
    assert call_order == [
        ("get", "kelvin"),
        ("set", "kelvin", 5663),
        ("get", "kelvin"),
        ("get", "tint"),
        ("set", "tint", -200),
        ("get", "tint"),
        ("get", "exposureAdjust"),
        ("set", "exposureAdjust", -250),
        ("get", "exposureAdjust"),
    ]
    assert result["verification_summary"]["final_status"] == "VERIFIED"


def test_websocket_transaction_marks_invalid_input_when_histogram_clipping_detected(monkeypatch: pytest.MonkeyPatch) -> None:
    transport = WebSocketRcp2Transport()
    session = Rcp2WebSocketSession(sock=object(), host="10.0.0.1", port=9998)
    monkeypatch.setattr(transport, "read_state", lambda _session: {"exposureAdjust": 0.0, "kelvin": 5600, "tint": 0.0})
    monkeypatch.setattr(
        transport,
        "_best_effort_histogram_guard",
        lambda _session: {"enabled": True, "status": "success", "clipping_detected": True, "note": "Clipping detected on gray reference — measurement invalid"},
    )
    result = transport.apply_state_transactionally(
        session,
        {"exposureAdjust": 0.0, "kelvin": 5600, "tint": 0.0},
        enable_histogram_guard=True,
    )
    assert result["verification_summary"]["final_status"] == "INVALID_INPUT"
    assert result["histogram_guard"]["clipping_detected"] is True


def test_websocket_transaction_includes_clip_metadata_cross_check(monkeypatch: pytest.MonkeyPatch) -> None:
    transport = WebSocketRcp2Transport()
    session = Rcp2WebSocketSession(sock=object(), host="10.0.0.1", port=9998)
    current = {
        "kelvin": Rcp2ParameterState("COLOR_TEMPERATURE", 5600, 1, 5600.0, "5600", {"divider": 1, "min": 2000, "max": 10000}, []),
        "tint": Rcp2ParameterState("TINT", 0, 1000, 0.0, "0.0", {"divider": 1000, "min": -10000, "max": 10000}, []),
        "exposureAdjust": Rcp2ParameterState("EXPOSURE_ADJUST", 0, 1000, 0.0, "0.0", {"divider": 1000, "min": -3000, "max": 3000}, []),
    }

    def fake_read_state(_session: object) -> dict:
        return {
            "exposureAdjust": current["exposureAdjust"].value,
            "kelvin": int(round(current["kelvin"].value)),
            "tint": current["tint"].value,
        }

    def fake_read_parameter_state(_session: object, field_name: str) -> Rcp2ParameterState:
        return current[field_name]

    def fake_write_parameter_state(_session: object, field_name: str, raw_value: int) -> Rcp2ParameterState:
        state = current[field_name]
        current[field_name] = Rcp2ParameterState(
            parameter_id=normalize_parameter_id(state.parameter_id),
            raw_value=raw_value,
            divider=state.divider,
            value=float(raw_value) / float(state.divider),
            display=str(float(raw_value) / float(state.divider)),
            edit_info=state.edit_info,
            messages=[],
        )
        return current[field_name]

    monkeypatch.setattr(transport, "read_state", fake_read_state)
    monkeypatch.setattr(transport, "_read_parameter_state", fake_read_parameter_state)
    monkeypatch.setattr(transport, "_write_parameter_state", fake_write_parameter_state)
    monkeypatch.setattr(
        transport,
        "_best_effort_clip_metadata",
        lambda _session, _state: {"status": "success", "metadata_match_status": "WITHIN_TOLERANCE", "fields": {"kelvin": {"status": "WITHIN_TOLERANCE"}}},
    )
    result = transport.apply_state_transactionally(
        session,
        {"exposureAdjust": 0.0, "kelvin": 5663, "tint": 0.0},
        include_clip_metadata_cross_check=True,
    )
    assert result["clip_metadata_cross_check"]["metadata_match_status"] == "WITHIN_TOLERANCE"


def test_summarize_apply_report_uses_operator_tolerances() -> None:
    report = {
        "results": [
            {
                "inventory_camera_label": "GA",
                "status": "applied_successfully",
                "requested": {"exposureAdjust": 0.5, "kelvin": 5663, "tint": 0.0},
                "readback": {"exposureAdjust": 0.505, "kelvin": 5670, "tint": 0.005},
            },
            {
                "inventory_camera_label": "GB",
                "status": "applied_successfully",
                "requested": {"exposureAdjust": 0.5, "kelvin": 5663, "tint": 0.0},
                "readback": {"exposureAdjust": 0.52, "kelvin": 5800, "tint": 0.2},
            },
        ]
    }
    summary = summarize_apply_report(report)
    assert summary["operator_status"] == "partial"
    assert summary["operator_status_counts"]["applied_with_deviation"] == 1
    assert summary["operator_status_counts"]["mismatch_after_writeback"] == 1
    assert summary["results"][0]["operator_result"]["display_status"] == "applied_with_deviation"
    assert summary["results"][1]["operator_result"]["display_status"] == "mismatch_after_writeback"


def test_summarize_apply_report_accepts_boundary_tolerances_and_flags_timeout() -> None:
    report = {
        "results": [
            {
                "inventory_camera_label": "GA",
                "status": "applied_successfully",
                "requested": {"exposureAdjust": 0.5, "kelvin": 5663, "tint": 0.0},
                "readback": {"exposureAdjust": 0.51, "kelvin": 5713, "tint": 0.01},
            },
            {
                "inventory_camera_label": "GB",
                "status": "timeout",
                "requested": {"exposureAdjust": 0.1, "kelvin": 5663, "tint": 0.0},
                "error": "connect timeout",
                "failure_reason": "timeout",
            },
        ]
    }
    summary = summarize_apply_report(report)
    assert summary["operator_status"] == "partial"
    assert summary["operator_status_counts"]["applied_with_deviation"] == 1
    assert summary["operator_status_counts"]["timeout"] == 1
    assert summary["results"][0]["operator_result"]["display_status"] == "applied_with_deviation"
    assert summary["results"][1]["operator_result"]["display_status"] == "timeout"


def test_build_post_apply_verification_from_reviews_compares_real_review_metrics(tmp_path: Path) -> None:
    before_dir = tmp_path / "before"
    after_dir = tmp_path / "after"
    for root, exposure_error, neutral_error, variance_bump in [
        (before_dir, 0.02, 0.015, 0.02),
        (after_dir, 0.01, 0.01, 0.005),
    ]:
        report_dir = root / "report"
        report_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "status": "success",
            "recommendation": {"recommended_strategy": {"strategy_key": "median"}},
            "physical_validation": {
                "status": "success",
                "exposure": {
                    "mean_exposure_error": exposure_error,
                    "per_camera": [
                        {"clip_id": "A", "exposure_error": exposure_error + variance_bump},
                        {"clip_id": "B", "exposure_error": exposure_error - variance_bump / 2.0},
                    ],
                },
                "neutrality": {
                    "mean_post_neutral_error": neutral_error,
                    "per_camera": [
                        {"clip_id": "A", "post_neutral_error": neutral_error + variance_bump / 4.0},
                        {"clip_id": "B", "post_neutral_error": neutral_error - variance_bump / 8.0},
                    ],
                },
            },
        }
        (report_dir / "review_validation.json").write_text(json.dumps(payload), encoding="utf-8")
    comparison = build_post_apply_verification_from_reviews(str(before_dir), str(after_dir), out_path=str(before_dir / "report" / "post_apply_verification.json"))
    assert comparison["status"] == "success"
    assert comparison["summary"]["exposure_improved"] is True
    assert comparison["summary"]["neutrality_improved"] is True
    assert comparison["summary"]["variance_reduced"] is True
    assert Path(comparison["report_path"]).exists()


def test_build_camera_verification_report_distinguishes_simulated_and_actual_compare(tmp_path: Path) -> None:
    payload_path = tmp_path / "GA.json"
    payload_path.write_text(
        json.dumps(
            {
                "schema_version": "r3dmatch_rcp2_ready_v1",
                "camera_id": "G007_A065",
                "clip_id": "G007_A065_0325F6_001",
                "inventory_camera_label": "GA",
                "inventory_camera_ip": "10.20.61.191",
                "source_path": "/captures/GA/G007_A065_0325F6_001.R3D",
                "calibration": {"exposureAdjust": 0.2, "kelvin": 5650, "tint": 0.4},
                "confidence": 0.8,
                "notes": ["Stable"],
            }
        ),
        encoding="utf-8",
    )
    simulated = build_camera_verification_report(str(payload_path))
    assert simulated["verification_mode"] == "simulated_expected_state"
    assert simulated["status"] == "simulated"
    assert simulated["results"][0]["verification_status"] == "simulated_expected"
    assert simulated["results"][0]["verification_level"] == "NOT_AVAILABLE"

    camera_state_report = {
        "schema_version": "r3dmatch_rcp2_camera_state_report_v1",
        "camera_count": 1,
        "connected_camera_count": 1,
        "results": [
            {
                "inventory_camera_label": "GA",
                "status": "success",
                "state": {"exposureAdjust": 0.205, "kelvin": 5655, "tint": 0.405},
            }
        ],
    }
    compared = build_camera_verification_report(str(payload_path), camera_state_report=camera_state_report)
    assert compared["verification_mode"] == "camera_state_report_compare"
    assert compared["status"] == "success"
    assert compared["within_tolerance_camera_count"] == 1
    assert compared["results"][0]["verification_status"] == "within_tolerance"
    assert compared["results"][0]["verification_level"] == "WITHIN_TOLERANCE"
    assert compared["results"][0]["verification_confidence_score"] > 0.0
    assert compared["results"][0]["comparison"]["matches_within_tolerance"] is True


def test_verify_camera_state_cli_forwards_to_structured_report(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def fake_build_camera_verification_report(payload_path: str, **kwargs: object) -> dict[str, object]:
        captured["payload_path"] = payload_path
        captured["kwargs"] = kwargs
        return {"schema_version": "r3dmatch_rcp2_verification_report_v1", "status": "simulated"}

    monkeypatch.setattr("r3dmatch.cli.build_camera_verification_report", fake_build_camera_verification_report)
    payload_path = tmp_path / "GA.json"
    payload_path.write_text("{}", encoding="utf-8")
    result = runner.invoke(app, ["verify-camera-state", str(payload_path), "--camera", "GA"])
    assert result.exit_code == 0
    assert captured["payload_path"] == str(payload_path)
    assert captured["kwargs"]["requested_cameras"] == ["GA"]


def test_web_app_status_surfaces_decision_commit_apply_and_verification(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out_dir = tmp_path / "review-output"
    report_dir = out_dir / "report"
    payload_dir = report_dir / "calibration_payloads"
    payload_dir.mkdir(parents=True, exist_ok=True)
    started_at = time.time()
    review_validation = {
        "status": "success",
        "validated_at": started_at + 1.0,
        "warnings": [],
        "errors": [],
        "recommendation": {
            "recommended_strategy": {"strategy_key": "median", "strategy_label": "Median", "reason": "Median minimizes correction spread."},
            "hero_camera": "I007_C065_0325WK_001",
            "hero_camera_confidence": "medium",
            "confidence_score": 0.82,
            "summary_notes": ["Array is stable enough to commit."],
        },
        "physical_validation": {
            "status": "success",
            "exposure": {"mean_exposure_error": 0.008, "max_exposure_error": 0.01, "per_camera": []},
            "neutrality": {"mean_post_neutral_error": 0.009, "max_post_neutral_error": 0.011, "per_camera": []},
            "kelvin_tint_analysis": {"kelvin_is_stable": True, "tint_carries_variation": True},
            "confidence": {"mean_confidence": 0.77},
            "outliers": [],
        },
        "failure_modes": [],
        "post_apply_validation": {
            "status": "warning",
            "verification_mode": "modeled_from_recommended_commit_values",
            "summary": {"exposure_error_reduced": True, "neutrality_improved": False, "variance_reduced": True},
            "notes": ["Modeled verification only."],
        },
    }
    (report_dir / "review_validation.json").write_text(json.dumps(review_validation), encoding="utf-8")
    report_payload = {
        "white_balance_model": {
            "model_key": "shared_kelvin_per_camera_tint",
            "model_label": "Shared Kelvin / Per-Camera Tint",
            "shared_kelvin": 5663,
            "shared_tint": 0.0,
            "candidates": [
                {
                    "model_key": "shared_kelvin_per_camera_tint",
                    "model_label": "Shared Kelvin / Per-Camera Tint",
                    "metrics": {"weighted_mean_post_neutral_residual": 0.002, "kelvin_axis_stddev": 0.0004},
                }
            ],
        },
        "visuals": {
            "exposure_plot_svg": "<svg viewBox='0 0 10 10'></svg>",
            "before_after_exposure_svg": "<svg viewBox='0 0 10 10'></svg>",
            "strategy_chart_svg": "<svg viewBox='0 0 10 10'></svg>",
        },
        "per_camera_analysis": [
            {
                "camera_label": "GA",
                "clip_id": "G007_A065_0325F6_001",
                "reference_use": "Included",
                "measured_gray_exposure_summary": "Sample 1 40 / Sample 2 30 / Sample 3 21 IRE",
                "sample_1_ire": 40.0,
                "sample_2_ire": 30.0,
                "sample_3_ire": 21.0,
                "camera_offset_from_anchor": 0.0,
                "final_offset_stops": 0.0,
                "trust_score": 0.88,
                "confidence": 0.82,
            }
        ],
    }
    (report_dir / "contact_sheet.json").write_text(json.dumps(report_payload), encoding="utf-8")
    per_camera_payload = {
        "schema_version": "r3dmatch_rcp2_ready_v1",
        "camera_id": "G007_A065",
        "clip_id": "G007_A065_0325F6_001",
        "inventory_camera_label": "GA",
        "inventory_camera_ip": "172.20.114.165",
        "source_path": "/captures/GA/G007_A065_0325F6_001.R3D",
        "calibration": {"exposureAdjust": -0.821145, "kelvin": 5663, "tint": -0.2},
        "confidence": 0.68,
        "notes": ["Exposure outlier against the central cluster."],
    }
    per_camera_path = payload_dir / "G007_A065.json"
    per_camera_path.write_text(json.dumps(per_camera_payload), encoding="utf-8")
    commit_payload = {
        "schema_version": "r3dmatch_calibration_commit_payload_v1",
        "camera_targets": [
            {
                "camera_id": "G007_A065",
                "clip_id": "G007_A065_0325F6_001",
                "inventory_camera_label": "GA",
                "inventory_camera_ip": "172.20.114.165",
                "calibration": {"exposureAdjust": -0.821145, "kelvin": 5663, "tint": -0.2},
                "confidence": 0.68,
                "is_hero_camera": False,
            }
        ],
        "per_camera_payloads": [{"camera_id": "G007_A065", "path": str(per_camera_path)}],
    }
    (report_dir / "calibration_commit_payload.json").write_text(json.dumps(commit_payload), encoding="utf-8")
    apply_report = {
        "schema_version": "r3dmatch_rcp2_apply_report_v1",
        "transport_mode": "live",
        "operator_status": "applied_with_deviation",
        "operator_status_counts": {"applied_with_deviation": 1},
        "results": [
            {
                "inventory_camera_label": "GA",
                "status": "applied_successfully",
                "operator_result": {"display_status": "applied_with_deviation"},
                "requested": {"exposureAdjust": -0.82, "kelvin": 5663, "tint": -0.2},
                "readback": {"exposureAdjust": -0.815, "kelvin": 5670, "tint": -0.195},
            }
        ],
    }
    (report_dir / "rcp2_live_apply_report.json").write_text(json.dumps(apply_report), encoding="utf-8")
    camera_state_report = {
        "schema_version": "r3dmatch_rcp2_camera_state_report_v1",
        "camera_count": 1,
        "connected_camera_count": 1,
        "results": [
            {
                "inventory_camera_label": "GA",
                "inventory_camera_ip": "172.20.114.165",
                "camera_id": "G007_A065",
                "clip_id": "G007_A065_0325F6_001",
                "status": "success",
                "state": {"exposureAdjust": -0.500, "kelvin": 5600, "tint": -0.1},
            }
        ],
    }
    (report_dir / "rcp2_camera_state_report.json").write_text(json.dumps(camera_state_report), encoding="utf-8")
    post_apply_report = {
        "schema_version": "r3dmatch_post_apply_verification_v1",
        "status": "success",
        "summary": {"exposure_improved": True, "neutrality_improved": True, "variance_reduced": True},
        "notes": ["Compared before/after review runs."],
    }
    (report_dir / "post_apply_verification.json").write_text(json.dumps(post_apply_report), encoding="utf-8")

    app_instance = create_app()
    state = app_instance.config["UI_STATE"]
    state.task.command = "python3 -m r3dmatch.cli review-calibration"
    state.task.output_path = str(out_dir)
    state.task.source_mode = "local_folder"
    state.task.status = "completed"
    state.task.returncode = 0
    state.task.started_at = started_at
    state.task.last_output_at = started_at + 1.0
    state.task.last_progress_at = started_at + 1.0
    monkeypatch.setattr("r3dmatch.web_app._process_is_alive", lambda process: False)
    with app_instance.test_client() as client:
        payload = client.get("/status").get_json()
    assert payload["decision_surface"]["commit_readiness"]["state"] == "Ready for Commit"
    assert "Median" in payload["decision_surface_html"]
    assert "Array is within calibration tolerance" in payload["decision_surface_html"]
    assert "Shared Kelvin / Per-Camera Tint" in payload["decision_surface_html"]
    assert "Exposure Spread" in payload["decision_surface_html"]
    assert "Measurement Stability" in payload["decision_surface_html"]
    assert "Calibration Review Notes" in payload["decision_surface_html"]
    assert "Reference Candidate" in payload["decision_surface_html"]
    assert "chart-launch" in payload["decision_surface_html"]
    assert "Click to enlarge" in payload["decision_surface_html"]
    assert "Closest current reference candidate" in payload["decision_surface_html"]
    assert "GA" in payload["commit_table_html"]
    assert "Exposure Correction" in payload["commit_table_html"]
    assert "+0.00" not in payload["decision_surface_html"]
    assert "Exposure span" not in payload["decision_surface_html"]
    assert "Current Exposure" in payload["push_surface_html"]
    assert "New Exposure" in payload["push_surface_html"]
    assert "Exposure Change" in payload["push_surface_html"]
    assert "Reference Use" in payload["push_surface_html"]
    assert "Writeback Status" in payload["push_surface_html"]
    assert "Included" in payload["push_surface_html"]
    assert "Verified" in payload["push_surface_html"]
    assert "-0.500" in payload["push_surface_html"]
    assert "-0.821" in payload["push_surface_html"]
    assert "<span class='delta-negative'>-0.321</span>" in payload["push_surface_html"]
    assert "All exposure values shown in this push section are relative corrections applied to match the calibration target." in payload["push_surface_html"]
    assert "applied_with_deviation" in payload["apply_surface_html"]
    assert "Compared before/after review runs." in payload["verification_surface_html"]


def test_web_app_surfaces_excluded_cameras_and_warning_commit_readiness() -> None:
    review_validation = {
        "status": "success",
        "warnings": ["Excluded 1 outlier camera from commit readiness; the remaining 11-camera cluster is still usable."],
        "run_assessment": {
            "status": "READY_WITH_WARNINGS",
            "recommendation_strength": "MEDIUM_CONFIDENCE",
            "trusted_camera_count": 11,
            "camera_count": 12,
            "excluded_camera_count": 1,
            "safe_to_push_later": True,
            "anchor_summary": "Median used the center of the trusted camera group instead of a single-camera anchor.",
            "gating_reasons": ["The exposure set is fragmented into a stable cluster plus outliers."],
            "operator_note": "This run is usable, but at least one camera should be reviewed before any later push.",
        },
        "recommendation": {
            "recommended_strategy": {"strategy_key": "median", "strategy_label": "Median", "reason": "Median keeps the main cluster balanced."},
            "confidence_score": 0.62,
            "recommendation_strength": "MEDIUM_CONFIDENCE",
            "run_status": "READY_WITH_WARNINGS",
            "summary_notes": [],
        },
        "physical_validation": {
            "status": "warning",
            "exposure": {"mean_exposure_error": 0.008},
            "neutrality": {"max_post_neutral_error": 0.011},
            "kelvin_tint_analysis": {"kelvin_is_stable": True, "tint_carries_variation": True},
            "confidence": {"mean_confidence": 0.71},
            "outliers": [{"clip_id": "I007_A064_0325XY_001"}],
            "excluded_cameras": [
                {
                    "clip_id": "I007_A064_0325XY_001",
                    "camera_id": "I007_A064",
                    "reasons": ["validation_confidence_low"],
                }
            ],
            "excluded_camera_count": 1,
            "included_camera_count": 11,
            "cluster_is_usable_after_exclusions": True,
            "warnings": ["Excluded 1 outlier camera from commit readiness; the remaining 11-camera cluster is still usable."],
        },
        "failure_modes": [{"code": "excluded_cameras_present", "severity": "warning", "message": "1 camera was excluded from safe commit export."}],
    }
    report_payload = {
        "visuals": {"exposure_plot_svg": "<svg viewBox='0 0 10 10'></svg>"},
        "per_camera_analysis": [
            {
                "camera_label": "H007_B064",
                "clip_id": "H007_B064_0325XY_001",
                "reference_use": "Included",
                "measured_gray_exposure_summary": "Sample 1 41 / Sample 2 31 / Sample 3 21 IRE",
                "sample_1_ire": 41.0,
                "sample_2_ire": 31.0,
                "sample_3_ire": 21.0,
                "camera_offset_from_anchor": 0.08,
                "final_offset_stops": 0.08,
                "trust_score": 0.91,
                "confidence": 0.94,
            },
            {
                "camera_label": "I007_A064",
                "clip_id": "I007_A064_0325XY_001",
                "reference_use": "Excluded",
                "measured_gray_exposure_summary": "Sample 1 48 / Sample 2 36 / Sample 3 25 IRE",
                "sample_1_ire": 48.0,
                "sample_2_ire": 36.0,
                "sample_3_ire": 25.0,
                "camera_offset_from_anchor": -1.22,
                "final_offset_stops": -1.22,
                "trust_score": 0.22,
                "confidence": 0.30,
            },
        ],
    }
    commit_payload = {
        "camera_targets": [
            {
                "camera_id": "I007_A064",
                "clip_id": "I007_A064_0325XY_001",
                "inventory_camera_label": "IA",
                "inventory_camera_ip": "172.20.114.173",
                "calibration": {"exposureAdjust": -1.219, "kelvin": 5664, "tint": 0.2},
                "confidence": 0.30,
                "excluded_from_commit": True,
                "exclusion_reasons": ["validation_confidence_low"],
            }
        ],
        "per_camera_payloads": [],
    }
    from r3dmatch.web_app import _build_operator_surfaces, _render_commit_table, _render_decision_surface

    surfaces = _build_operator_surfaces(
        review_validation=review_validation,
        review_payload=report_payload,
        commit_payload=commit_payload,
        camera_state_report=None,
        apply_report=None,
        writeback_verification_report=None,
        post_apply_report=None,
        source_mode_label_text="Local Folder",
    )

    assert surfaces["commit_readiness"]["state"] == "Commit with Warnings"
    assert surfaces["physical_validation"]["cluster_is_usable_after_exclusions"] is True
    decision_html = _render_decision_surface(surfaces)
    commit_html = _render_commit_table(surfaces)
    assert "Array is near calibration tolerance" in decision_html
    assert "Retained gray sphere Sample 2 values span" in decision_html
    assert "Closest current reference candidate" in decision_html
    assert "I007_A064" in decision_html
    assert "Calibration Review Notes" in decision_html
    assert "Excluded" in commit_html


def test_web_apply_and_verify_routes_use_existing_backend_truth(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out_dir = tmp_path / "workflow-output"
    report_dir = out_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    commit_payload_path = report_dir / "calibration_commit_payload.json"
    commit_payload_path.write_text(json.dumps({"schema_version": "r3dmatch_calibration_commit_payload_v1", "camera_targets": [], "per_camera_payloads": []}), encoding="utf-8")

    app_instance = create_app()
    state = app_instance.config["UI_STATE"]
    state.scan = {"clip_count": 0, "clip_ids": [], "sample_clip_ids": [], "remaining_count": 0, "clip_records": [], "clip_groups": [], "warning": None, "preview_available": False}
    state.form.update({"output_path": str(out_dir), "resolved_output_path": str(out_dir), "verification_after_path": str(tmp_path / 'after-review')})

    def fake_apply_calibration_payload(*args, **kwargs):
        report = {
            "schema_version": "r3dmatch_rcp2_apply_report_v1",
            "transport_mode": "dry_run",
            "operator_status": "dry_run",
            "operator_status_counts": {"dry_run": 1},
            "results": [],
        }
        Path(str(kwargs["out_path"])).write_text(json.dumps(report), encoding="utf-8")
        return report

    def fake_build_post_apply_verification(before_review_dir: str, after_review_dir: str, *, out_path: Optional[str] = None):
        report = {"schema_version": "r3dmatch_post_apply_verification_v1", "status": "warning", "summary": {"exposure_improved": True, "neutrality_improved": False, "variance_reduced": True}}
        Path(str(out_path)).write_text(json.dumps(report), encoding="utf-8")
        report["report_path"] = str(out_path)
        return report

    monkeypatch.setattr("r3dmatch.web_app.apply_calibration_payload", fake_apply_calibration_payload)
    monkeypatch.setattr("r3dmatch.web_app.build_post_apply_verification_from_reviews", fake_build_post_apply_verification)

    with app_instance.test_client() as client:
        apply_response = client.post("/apply-calibration", data={"output_path": str(out_dir), "apply_cameras": "GA"})
        verify_response = client.post("/verify-apply", data={"output_path": str(out_dir), "verification_after_path": str(tmp_path / 'after-review')})
    assert apply_response.status_code == 200
    assert verify_response.status_code == 200
    assert (report_dir / "rcp2_apply_report.json").exists()
    assert (report_dir / "post_apply_verification.json").exists()


def test_build_operator_surfaces_derives_white_balance_summary_from_commit_payload() -> None:
    from r3dmatch.web_app import _build_operator_surfaces, _render_decision_surface

    surfaces = _build_operator_surfaces(
        review_validation={
            "recommendation": {
                "recommended_strategy": {
                    "strategy_key": "median",
                    "strategy_label": "Median",
                    "reason": "Median keeps the cluster stable.",
                }
            },
            "physical_validation": {"status": "success"},
        },
        review_payload={
            "run_assessment": {"status": "READY_WITH_WARNINGS"},
            "per_camera_analysis": [
                {
                    "camera_label": "GA",
                    "clip_id": "G007_A064_001",
                    "reference_use": "Included",
                    "sample_2_ire": 23.0,
                    "measured_gray_exposure_summary": "Sample 1 24 / Sample 2 23 / Sample 3 21 IRE",
                }
            ],
            "white_balance_model": {},
            "visuals": {},
        },
        commit_payload={
            "camera_targets": [
                {
                    "camera_id": "G007_A064",
                    "clip_id": "G007_A064_001",
                    "inventory_camera_label": "GA",
                    "inventory_camera_ip": "172.20.114.165",
                    "calibration": {"exposureAdjust": 0.11, "kelvin": 5626, "tint": -0.2},
                    "confidence": 0.8,
                },
                {
                    "camera_id": "G007_B064",
                    "clip_id": "G007_B064_001",
                    "inventory_camera_label": "GB",
                    "inventory_camera_ip": "172.20.114.166",
                    "calibration": {"exposureAdjust": -0.08, "kelvin": 5624, "tint": 0.1},
                    "confidence": 0.78,
                },
            ],
            "per_camera_payloads": [],
        },
        camera_state_report=None,
        apply_report=None,
        writeback_verification_report=None,
        post_apply_report=None,
        source_mode_label_text="Local Folder",
    )

    wb_model = dict(surfaces["white_balance_model"])
    assert wb_model["model_label"] == "Derived from Final WB Values"
    assert wb_model["shared_kelvin_mode"] == "shared"
    assert wb_model["shared_kelvin"] == 5625
    assert wb_model["shared_tint_mode"] == "per_camera"
    assert wb_model["candidate_count"] >= 1
    assert wb_model["source_measurement_type"] == "final_commit_values"
    assert wb_model["diagnostics"]["summary_kind"] == "fallback_derived"

    decision_html = _render_decision_surface(surfaces)
    assert "Derived from Final WB Values" in decision_html
    assert "Kelvin Spread" in decision_html
    assert "Tint Spread" in decision_html
    assert "Source Samples" in decision_html


def test_web_push_routes_load_payload_and_read_current_values(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out_dir = tmp_path / "workflow-output"
    report_dir = out_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    payload_dir = report_dir / "calibration_payloads"
    payload_dir.mkdir(parents=True, exist_ok=True)
    per_camera_path = payload_dir / "GA.json"
    per_camera_path.write_text(
        json.dumps(
            {
                "schema_version": "r3dmatch_rcp2_ready_v1",
                "camera_id": "G007_A065",
                "clip_id": "G007_A065_0325F6_001",
                "inventory_camera_label": "GA",
                "inventory_camera_ip": "10.20.61.191",
                "source_path": "/captures/GA/G007_A065_0325F6_001.R3D",
                "calibration": {"exposureAdjust": 0.2, "kelvin": 5650, "tint": 0.4},
                "confidence": 0.8,
                "notes": ["Stable"],
            }
        ),
        encoding="utf-8",
    )
    commit_payload_path = report_dir / "calibration_commit_payload.json"
    commit_payload_path.write_text(
        json.dumps(
            {
                "schema_version": "r3dmatch_calibration_commit_payload_v1",
                "camera_targets": [
                    {
                        "camera_id": "G007_A065",
                        "clip_id": "G007_A065_0325F6_001",
                        "inventory_camera_label": "GA",
                        "inventory_camera_ip": "10.20.61.191",
                        "calibration": {"exposureAdjust": 0.2, "kelvin": 5650, "tint": 0.4},
                        "confidence": 0.8,
                        "excluded_from_commit": False,
                        "exclusion_reasons": [],
                    }
                ],
                "per_camera_payloads": [{"camera_id": "G007_A065", "path": str(per_camera_path)}],
            }
        ),
        encoding="utf-8",
    )

    def fake_read_camera_state(*, host: str, port: int = 9998, camera_label: str = "LIVE", **_: object) -> dict[str, object]:
        assert host == "10.20.61.191"
        assert camera_label == "GA"
        return {
            "schema_version": "r3dmatch_rcp2_camera_state_v1",
            "transport_mode": "live_websocket",
            "host": host,
            "port": port,
            "camera_label": camera_label,
            "state": {"exposureAdjust": 0.0, "kelvin": 5600, "tint": 0.0},
        }

    monkeypatch.setattr("r3dmatch.web_app.read_camera_state", fake_read_camera_state)

    app_instance = create_app()
    state = app_instance.config["UI_STATE"]
    state.scan = {"clip_count": 0, "clip_ids": [], "sample_clip_ids": [], "remaining_count": 0, "clip_records": [], "clip_groups": [], "warning": None, "preview_available": False}
    state.form.update({"output_path": str(out_dir), "resolved_output_path": str(out_dir)})

    with app_instance.test_client() as client:
        load_response = client.post("/load-recommended-payload", data={"output_path": str(out_dir), "apply_cameras": "GA"})
        read_response = client.post("/read-current-camera-values", data={"output_path": str(out_dir), "apply_cameras": "GA"})
        status_payload = client.get("/status").get_json()

    assert load_response.status_code == 200
    assert read_response.status_code == 200
    camera_state_report = json.loads((report_dir / "rcp2_camera_state_report.json").read_text(encoding="utf-8"))
    assert camera_state_report["connected_camera_count"] == 1
    assert camera_state_report["results"][0]["state"]["kelvin"] == 5600
    assert "Ready" in status_payload["push_surface_html"]
    assert "+0.200" in status_payload["push_surface_html"]
    assert "<span class='delta-positive'>+0.200</span>" in status_payload["push_surface_html"]


def test_verify_last_push_route_builds_structured_verification_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out_dir = tmp_path / "workflow-output"
    report_dir = out_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    payload_dir = report_dir / "calibration_payloads"
    payload_dir.mkdir(parents=True, exist_ok=True)
    per_camera_path = payload_dir / "GA.json"
    per_camera_path.write_text(
        json.dumps(
            {
                "schema_version": "r3dmatch_rcp2_ready_v1",
                "camera_id": "G007_A065",
                "clip_id": "G007_A065_0325F6_001",
                "inventory_camera_label": "GA",
                "inventory_camera_ip": "10.20.61.191",
                "source_path": "/captures/GA/G007_A065_0325F6_001.R3D",
                "calibration": {"exposureAdjust": 0.2, "kelvin": 5650, "tint": 0.4},
                "confidence": 0.8,
                "notes": ["Stable"],
            }
        ),
        encoding="utf-8",
    )
    commit_payload_path = report_dir / "calibration_commit_payload.json"
    commit_payload_path.write_text(
        json.dumps(
            {
                "schema_version": "r3dmatch_calibration_commit_payload_v1",
                "camera_targets": [
                    {
                        "camera_id": "G007_A065",
                        "clip_id": "G007_A065_0325F6_001",
                        "inventory_camera_label": "GA",
                        "inventory_camera_ip": "10.20.61.191",
                        "calibration": {"exposureAdjust": 0.2, "kelvin": 5650, "tint": 0.4},
                        "confidence": 0.8,
                        "excluded_from_commit": False,
                        "exclusion_reasons": [],
                    }
                ],
                "per_camera_payloads": [{"camera_id": "G007_A065", "path": str(per_camera_path)}],
            }
        ),
        encoding="utf-8",
    )
    (report_dir / "rcp2_camera_state_report.json").write_text(
        json.dumps(
            {
                "schema_version": "r3dmatch_rcp2_camera_state_report_v1",
                "camera_count": 1,
                "connected_camera_count": 1,
                "results": [
                    {
                        "inventory_camera_label": "GA",
                        "status": "success",
                        "state": {"exposureAdjust": 0.205, "kelvin": 5655, "tint": 0.405},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    app_instance = create_app()
    state = app_instance.config["UI_STATE"]
    state.scan = {"clip_count": 0, "clip_ids": [], "sample_clip_ids": [], "remaining_count": 0, "clip_records": [], "clip_groups": [], "warning": None, "preview_available": False}
    state.form.update({"output_path": str(out_dir), "resolved_output_path": str(out_dir)})

    with app_instance.test_client() as client:
        response = client.post("/verify-last-push", data={"output_path": str(out_dir), "apply_cameras": "GA"})
        status_payload = client.get("/status").get_json()

    assert response.status_code == 200
    verification_report = json.loads((report_dir / "rcp2_verification_report.json").read_text(encoding="utf-8"))
    assert verification_report["verification_mode"] == "camera_state_report_compare"
    assert verification_report["status"] == "success"
    assert verification_report["results"][0]["verification_status"] == "within_tolerance"
    assert verification_report["results"][0]["verification_level"] == "WITHIN_TOLERANCE"
    assert "Verification mode: camera state report compare." in status_payload["push_surface_html"]


def test_review_calibration_ftps_source_mode_threads_ingest_manifest_into_review(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_redline(monkeypatch)

    def fake_run_ftps_ingest_job(**kwargs):
        ingest_root = Path(kwargs["out_dir"])
        clip_path = ingest_root / "GA" / "G007_A063_032563.RDC" / "G007_A063_032563_001.R3D"
        clip_path.parent.mkdir(parents=True, exist_ok=True)
        clip_path.write_bytes(b"")
        manifest = {
            "schema_version": "r3dmatch_ftps_ingest_v1",
            "source_mode": "ftps_camera_pull",
            "source_mode_label": "FTPS Camera Pull",
            "local_ingest_root": str(ingest_root),
            "reel_identifier": "007",
            "clip_spec": "63",
            "clips_requested": [63],
            "requested_cameras": ["GA"],
            "requested_camera_ips": {"GA": "172.20.114.165"},
            "successful_cameras": ["GA"],
            "failed_cameras": [],
            "requested_camera_count": 1,
            "successful_camera_count": 1,
            "failed_camera_count": 0,
            "clips_found": 1,
            "bytes_pulled": 1,
            "per_camera_status": [],
            "status": "success",
            "manifest_path": str(ingest_root / "ingest_manifest.json"),
        }
        Path(manifest["manifest_path"]).write_text(json.dumps(manifest), encoding="utf-8")
        return manifest

    monkeypatch.setattr("r3dmatch.workflow.run_ftps_ingest_job", fake_run_ftps_ingest_job)
    review_dir = tmp_path / "review-ftps"
    payload = review_calibration(
        None,
        out_dir=str(review_dir),
        source_mode="ftps_camera_pull",
        ftps_reel="007",
        ftps_clip_spec="63",
        ftps_cameras=["GA"],
        target_type="gray_sphere",
        processing_mode="both",
        mode="scene",
        backend="mock",
        lut_override=None,
        calibration_path=None,
        exposure_calibration_path=None,
        color_calibration_path=None,
        calibration_mode=None,
        sample_count=4,
        sampling_strategy="uniform",
        calibration_roi={"x": 0.2, "y": 0.2, "w": 0.4, "h": 0.4},
        target_strategies=["median"],
        reference_clip_id=None,
        review_mode="lightweight_analysis",
    )
    review_package = json.loads((Path(payload["analysis_dir"]) / "report" / "review_package.json").read_text(encoding="utf-8"))
    validation = json.loads((Path(payload["analysis_dir"]) / "report" / "review_validation.json").read_text(encoding="utf-8"))
    assert review_package["source_mode"] == "ftps_camera_pull"
    assert review_package["ingest_manifest"]["status"] == "success"
    assert review_package["ingest_manifest"]["requested_camera_ips"]["GA"] == "172.20.114.165"
    assert validation["status"] == "success"


def test_approval_export_writes_batch_mapping_and_portable_scripts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_redline(monkeypatch)
    clip_a = tmp_path / "G007_B057_0324YT_001.R3D"
    clip_b = tmp_path / "G007_C057_0324GR_001.R3D"
    clip_c = tmp_path / "G007_D060_0324M6_001.R3D"
    clip_a.write_bytes(b"")
    clip_b.write_bytes(b"")
    clip_c.write_bytes(b"")
    review_dir = tmp_path / "review-export"
    review_calibration(
        str(tmp_path),
        out_dir=str(review_dir),
        target_type="gray_sphere",
        processing_mode="both",
        mode="scene",
        backend="mock",
        lut_override=None,
        calibration_path=None,
        exposure_calibration_path=None,
        color_calibration_path=None,
        calibration_mode=None,
        sample_count=4,
        sampling_strategy="uniform",
        calibration_roi={"x": 0.2, "y": 0.2, "w": 0.4, "h": 0.4},
        target_strategies=["median"],
        reference_clip_id=None,
    )
    payload = approve_master_rmd(str(review_dir))
    approval_manifest = json.loads(Path(payload["approval_manifest"]).read_text(encoding="utf-8"))
    batch_manifest = json.loads(Path(payload["batch_manifest"]).read_text(encoding="utf-8"))
    assert approval_manifest["correction_key_model"]["name"] == "group_key_from_clip_id"
    assert Path(payload["master_rmd_dir"]).name == "MasterRMD"
    assert any(item["correction_key"] == "G007_B057" for item in approval_manifest["master_rmd_exports"])
    first_entry = next(item for item in batch_manifest["entries"] if item["clip_id"] == "G007_B057_0324YT_001")
    assert first_entry["master_rmd_name"] == "G007_B057.RMD"
    assert first_entry["correction_key"] == "G007_B057"
    assert first_entry["approved_strategy"] == "median"
    sh_script = Path(payload["batch_scripts"]["sh"]).read_text(encoding="utf-8")
    tcsh_script = Path(payload["batch_scripts"]["tcsh"]).read_text(encoding="utf-8")
    assert "SOURCE_ROOT" in sh_script
    assert "--loadRMD \"$MASTER_RMD_DIR/G007_B057.RMD\" --useRMD 1" in sh_script
    assert "correction_key: G007_B057" in sh_script
    assert "--loadRMD \"$MASTER_RMD_DIR/G007_B057.RMD\" --useRMD 1" in tcsh_script


def test_load_review_bundle_for_ui(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    (out_dir / "analysis").mkdir(parents=True)
    (out_dir / "sidecars").mkdir(parents=True)
    (out_dir / "previews").mkdir(parents=True)
    (out_dir / "summary.json").write_text(json.dumps({"clip_count": 1}), encoding="utf-8")
    (out_dir / "array_calibration.json").write_text(
        json.dumps(
            {
                "target": {
                    "exposure": {"log2_luminance_target": -3.2},
                    "color": {"target_rgb_chromaticity": [0.33, 0.33, 0.34]},
                },
                "cameras": [
                    {
                        "clip_id": "G007_D060_0324M6_001",
                        "camera_id": "G007_D060",
                        "group_key": "array_batch",
                        "source_path": "/tmp/G007_D060_0324M6_001.R3D",
                        "measurement": {
                            "measured_log2_luminance": -3.3,
                            "measured_rgb_mean": [0.2, 0.21, 0.22],
                            "measured_rgb_chromaticity": [0.31, 0.33, 0.36],
                            "gray_sample_count": 8,
                            "valid_pixel_count": 100,
                            "saturation_fraction": 0.0,
                            "black_fraction": 0.0,
                        },
                        "solution": {"exposure_offset_stops": 0.1, "rgb_gains": [1.1, 1.0, 0.9]},
                        "quality": {"confidence": 0.98, "flags": [], "color_residual": 0.02, "exposure_residual_stops": 0.01},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (out_dir / "analysis" / "G007_D060_0324M6_001.analysis.json").write_text(
        json.dumps({"clip_id": "G007_D060_0324M6_001", "group_key": "array_batch", "confidence": 0.98}),
        encoding="utf-8",
    )
    (out_dir / "sidecars" / "G007_D060_0324M6_001.sidecar.json").write_text(
        json.dumps({"clip_id": "G007_D060_0324M6_001"}),
        encoding="utf-8",
    )
    (out_dir / "previews" / "G007_D060_0324M6_001.original.review.jpg").write_bytes(b"jpg")
    (out_dir / "previews" / "G007_D060_0324M6_001.both.review.jpg").write_bytes(b"jpg")
    bundle = load_review_bundle(str(out_dir))
    rows = build_table_rows(bundle["rows"])
    assert bundle["summary_text"]["clip_count"] == 1
    assert rows[0]["clip_id"] == "G007_D060_0324M6_001"
    assert bundle["rows"][0]["preview_variants"]["original"].endswith("G007_D060_0324M6_001.original.review.jpg")
    assert bundle["rows"][0]["preview_path"].endswith("G007_D060_0324M6_001.both.review.jpg")
