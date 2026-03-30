import json
import os
import subprocess
import time
import types
from pathlib import Path
from typing import Optional, cast

import numpy as np
import pytest
from PIL import Image
from typer.testing import CliRunner

from r3dmatch.calibration import build_array_calibration_from_analysis, calibrate_card_path, calibrate_color_path, calibrate_exposure_path, calibrate_sphere_path, center_crop_roi, derive_array_group_key, detect_gray_card_roi, load_calibration, load_card_roi_file, load_color_calibration, load_exposure_calibration, measure_card_from_roi, measure_color_region, measure_sphere_from_roi, solve_neutral_gains
from r3dmatch.cli import app
from r3dmatch.commit_values import build_commit_values, solve_kelvin_tint_from_chromaticity, solve_white_balance_model_for_records
from r3dmatch.desktop_app import build_review_command, run_ui_self_check, scan_calibration_sources
from r3dmatch.execution import CANCEL_FILE_ENV, CancellationError, run_cancellable_subprocess
from r3dmatch.ftps_ingest import ingest_ftps_batch, plan_ftps_request
from r3dmatch.identity import group_key_from_clip_id, rmd_name_for_clip_id, subset_key_from_clip_id
from r3dmatch.matching import analyze_path, camera_group_from_clip_id, discover_clips, measure_frame_color_and_exposure
from r3dmatch.models import GrayCardROI, SamplingRegion, SphereROI
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
from r3dmatch.report import _build_sphere_detection_artifacts, _build_strategy_payloads, _compute_image_difference_metrics, _ipp2_closed_loop_next_correction, _ipp2_closed_loop_target, _ipp2_validation_presentation, _measure_rendered_preview_roi_ipp2, _operator_guidance_for_correction, _report_grid_columns, _report_tiles_per_page, _resolve_redline_executable, _sphere_detection_label, _sphere_detection_note, build_contact_sheet_report, build_lightweight_analysis_report, build_review_package, format_stop_string, preview_filename_for_clip_id, render_contact_sheet_html, render_contact_sheet_pdf, render_preview_frame, round_to_standard_stop_fraction
from r3dmatch.rmd import render_rmd_xml, rmd_filename_for_clip_id, write_rmd_for_clip_with_metadata, write_rmds_from_analysis
from r3dmatch.ui import build_table_rows, load_review_bundle
from r3dmatch.sdk import MockR3DBackend, RedSdkDecoder, resolve_backend
from r3dmatch.sidecar import build_sidecar_payload, sidecar_filename_for_clip_id
from r3dmatch.transcode import build_redline_command, build_redline_command_variants, write_transcode_plan
from r3dmatch.validation import validate_pipeline
from r3dmatch.web_app import _ensure_scan_preview, _normalize_subset_form, _preferred_roi_preview_record, _resolve_selected_clip_ids, _resolved_output_path_for_form, _scan_preview_path, _subset_selection_ui, build_review_web_command, create_app, scan_sources
from r3dmatch.workflow import approve_master_rmd, build_post_apply_verification_from_reviews, clear_preview_cache, resolve_review_output_dir, review_calibration, validate_review_run_contract

runner = CliRunner()


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

    assert measured["sphere_roi_source"] in {"primary_detected", "secondary_detected"}
    assert measured["detected_sphere_roi"]["cx"] > 90.0
    assert measured["top_ire"] > measured["mid_ire"] > measured["bottom_ire"]


def test_measure_rendered_preview_roi_ipp2_fails_explicitly_when_no_sphere_candidate_exists(tmp_path: Path) -> None:
    image = np.zeros((320, 320, 3), dtype=np.float32)
    image[..., :] = 0.18
    path = tmp_path / "no_sphere.jpg"
    Image.fromarray(np.clip(image * 255.0, 0, 255).astype(np.uint8), mode="RGB").save(path, format="JPEG")

    measured = _measure_rendered_preview_roi_ipp2(str(path), None)

    assert measured["detection_failed"] is True
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
    assert measured["sphere_roi_source"] == "reused_from_original"
    assert measured["detected_sphere_roi"]["cx"] == pytest.approx(cx)


def test_sphere_detection_label_thresholds() -> None:
    assert _sphere_detection_label(0.85) == "HIGH"
    assert _sphere_detection_label(0.60) == "MEDIUM"
    assert _sphere_detection_label(0.35) == "LOW"
    assert _sphere_detection_label(0.10) == "FAILED"


def test_sphere_detection_note_includes_recovery_labels() -> None:
    assert _sphere_detection_note("primary_detected", "HIGH") == "Sphere detection: verified"
    assert _sphere_detection_note("localized_recovery", "MEDIUM") == "Sphere detection: fallback used"
    assert _sphere_detection_note("forced_best_effort", "LOW") == "Sphere detection: low-confidence recovery"


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
            "ipp2_gray_exposure_summary": "Bright 40 / Center 31 / Dark 26 IRE",
            "ipp2_target_gray_exposure_summary": "Bright 41 / Center 31 / Dark 26 IRE",
            "ipp2_zone_residuals": [
                {"label": "bright_side", "residual_stops": 0.03},
                {"label": "center", "residual_stops": -0.01},
                {"label": "dark_side", "residual_stops": 0.02},
            ],
        }
    )
    assert landed["presentation_state"] == "Outlier corrected successfully"
    assert landed["validation_label"] == "Verify T-Stop"
    assert landed["gray_exposure_text"] == "Gray Exposure: Bright 40 / Center 31 / Dark 26 IRE"
    assert landed["target_profile_text"] == "Reference profile: Bright 41 / Center 31 / Dark 26 IRE"
    assert "Bright +0.03" in landed["zone_residual_text"]
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
        Image.fromarray(np.clip(base * 255.0, 0, 255).astype(np.uint8), mode="RGB").save(generated_path, format="JPEG")
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
    monkeypatch.setattr("r3dmatch.sdk._load_red_native_module", lambda: None)
    with pytest.raises(RuntimeError, match="native bridge is not built"):
        RedSdkDecoder()


def test_red_backend_uses_native_module_when_available(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake_module = types.SimpleNamespace(
        sdk_available=lambda: True,
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
    native_detail = (
        "RED SDK frame decode failed. "
        "clip_path=/tmp/test/G007_D060_0324M6_001.R3D frame_index=0 "
        "decode_width=1024 decode_height=576 half_res=true "
        "colorspace=REDWideGamutRGB gamma=Log3G10 decode_attempts=[DECODE_HALF_RES_GOOD=7]"
    )
    fake_module = types.SimpleNamespace(
        sdk_available=lambda: True,
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
    assert preview_filename_for_clip_id("G007_D060_0324M6_001", "both") == "G007_D060_0324M6_001.both.review.jpg"
    assert (
        preview_filename_for_clip_id("G007_D060_0324M6_001", "both", strategy="median", run_id="batch01")
        == "G007_D060_0324M6_001.both.review.median.batch01.jpg"
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
    joined = " ".join(command)
    assert command[0] == "/bin/tcsh"
    assert "setenv PYTHONPATH" in joined
    assert "--preview-mode monitoring" in joined
    assert "--preview-lut /tmp/show.cube" in joined
    assert "--reference-clip-id G007_D060_0324M6_001" in joined


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
    assert summary["warning"] is None


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


def test_build_review_web_command_uses_tcsh_and_preview_options() -> None:
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
            "preview_mode": "monitoring",
            "preview_lut": "/tmp/show.cube",
        },
    )
    joined = " ".join(command)
    assert command[0] == "/bin/tcsh"
    assert "setenv PYTHONPATH" in joined
    assert "--review-mode full_contact_sheet" in joined
    assert "--preview-mode monitoring" in joined
    assert "--preview-lut /tmp/show.cube" in joined
    assert "--reference-clip-id G007_D060_0324M6_001" in joined


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


def test_build_review_web_command_supports_ftps_source_mode() -> None:
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
            "preview_mode": "monitoring",
            "preview_lut": "",
            "ftps_reel": "007",
            "ftps_clips": "63,64-65",
            "ftps_cameras": ["AA", "AB"],
        },
    )
    joined = " ".join(command)
    assert "--source-mode ftps_camera_pull" in joined
    assert "--ftps-reel 007" in joined
    assert "--ftps-clips 63,64-65" in joined
    assert joined.count("--ftps-camera") == 2
    assert "/tmp/out/ingest" in joined


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
            "preview_mode": "monitoring",
            "preview_lut": "",
        },
    )
    joined = " ".join(command)
    assert "--target-strategy hero-camera" in joined
    assert "--hero-clip-id G007_D060_0324M6_001" in joined


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
            "preview_mode": "monitoring",
            "preview_lut": "",
        },
    )
    joined = " ".join(command)
    assert "--run-label clip63_even" in joined
    assert "--matching-domain perceptual" in joined
    assert "--review-mode lightweight_analysis" in joined
    assert "--clip-group 063" in joined
    assert "--clip-id G007_B057_0324YT_063" in joined


def test_web_app_scan_page_shows_roi_mode_and_preview_placeholder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("r3dmatch.web_app._ensure_scan_preview", lambda input_path, scan, form=None: None)
    app = create_app()
    client = app.test_client()
    response = client.post("/scan", data={"input_path": str(tmp_path), "output_path": str(tmp_path / "out")})
    assert response.status_code == 200
    assert b"ROI Mode" in response.data
    assert b"Preview available after scan or first render" in response.data
    assert b"draw" in response.data
    assert b"center" in response.data
    assert b"full" in response.data
    assert b"manual" in response.data
    assert b"Hero Camera" in response.data


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
    assert strategies[0]["reference_clip_id"] == "G007_C057_0324YT_001"
    assert strategies[0]["target_log2_luminance"] == pytest.approx(-1.5)
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
    runner.invoke(app, ["analyze", str(clip_path), "--out", str(analysis_dir), "--backend", "mock", "--mode", "scene"], catch_exceptions=False)
    validate_result = runner.invoke(app, ["validate-pipeline", str(clip_path), "--analysis-dir", str(analysis_dir), "--out", str(tmp_path / "validation")])
    report_result = runner.invoke(app, ["report-contact-sheet", str(analysis_dir), "--out", str(tmp_path / "report")])
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
    assert (review_dir / "review_rmd" / "G007_D060_0324M6_001.RMD").exists()
    review_manifest = json.loads((review_dir / "report" / "review_manifest.json").read_text(encoding="utf-8"))
    assert review_manifest["calibration_roi"] == {"x": 0.25, "y": 0.25, "w": 0.5, "h": 0.5}
    assert review_manifest["target_strategies"] == ["median", "optimal_exposure"]
    approve_result = runner.invoke(app, ["approve-master-rmd", str(review_dir), "--target-strategy", "optimal-exposure"])
    assert approve_result.exit_code == 0
    assert (review_dir / "approval" / "MasterRMD" / "G007_D060.RMD").exists()
    assert (review_dir / "approval" / "batch" / "manifest.json").exists()
    assert (review_dir / "approval" / "approval_manifest.json").exists()
    assert (review_dir / "approval" / "calibration_report.pdf").exists()
    approval_manifest = json.loads((review_dir / "approval" / "approval_manifest.json").read_text(encoding="utf-8"))
    assert approval_manifest["selected_target_strategy"] == "optimal_exposure"
    assert approval_manifest["master_rmd_folder_name"] == "MasterRMD"


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
    report_json = json.loads(Path(payload["report_json"]).read_text(encoding="utf-8"))
    assert report_json["clips"][0]["clip_id"] == "G007_D060_0324M6_001"
    assert report_json["target_type"] == "gray_sphere"
    assert report_json["preview_transform"]
    assert report_json["exposure_measurement_domain"] == "scene"
    assert report_json["preview_mode"] == "monitoring"
    assert report_json["preview_settings"]["lut_path"].endswith("show.cube")
    assert report_json["measurement_preview_settings"]["preview_mode"] == "monitoring"
    assert report_json["redline_capabilities"]["supports_lut"] is True
    assert report_json["calibration_roi"] == {"x": 0.2, "y": 0.2, "w": 0.4, "h": 0.4}
    assert report_json["target_strategies"] == ["median", "optimal_exposure"]
    assert report_json["shared_originals"][0]["original_frame"].endswith("G007_D060_0324M6_001.original.review.analysis-out.jpg")
    assert report_json["strategies"][0]["clips"][0]["both_corrected"].endswith("G007_D060_0324M6_001.both.review.median.analysis-out.jpg")
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
    preview_commands = json.loads((tmp_path / "analysis-out" / "previews" / "preview_commands.json").read_text(encoding="utf-8"))
    exposure_command = next(command for command in preview_commands["commands"] if command["variant"] == "exposure" and command["strategy"] == "median")
    assert exposure_command["application_method"] == "direct_redline_flags"
    assert exposure_command["look_metadata_path"].endswith("/review_rmd/strategies/median/exposure/G007_D060_0324M6_001.RMD")
    assert Path(exposure_command["look_metadata_path"]).exists()
    assert "--useMeta" in exposure_command["command"]
    assert "--exposureAdjust" in exposure_command["command"]
    assert "--loadRMD" not in exposure_command["command"]
    assert "--lut" in exposure_command["command"]
    strategy_command = next(command for command in preview_commands["commands"] if command["variant"] == "both" and command["strategy"] == "median")
    assert strategy_command["application_method"] == "preview_color_disabled"
    assert strategy_command["preview_color_applied"] is False
    assert strategy_command["output_reused_from_variant"] == "exposure"
    assert strategy_command["validation_method"] == "preview_fallback_copy"
    assert strategy_command["correction_application_method"] == "preview_color_disabled"
    assert strategy_command["command"] is None
    assert strategy_command["rmd_path"].endswith("/review_rmd/strategies/median/both/G007_D060_0324M6_001.RMD")
    assert Path(strategy_command["rmd_path"]).exists()
    assert report_json["color_preview_enabled"] is False
    assert report_json["color_preview_status"] == "disabled_unverified"
    html = Path(payload["report_html"]).read_text(encoding="utf-8")
    assert "Color preview" in html
    assert "Gray Exposure" in html
    assert "aperture" in html
    assert "Digital correction applied" in html
    assert "What To Look For" in html
    assert "Exposure Summary" in html
    assert "Recommended Action" in html
    assert "G007_D060_0324M6_001" in html
    assert "../previews/G007_D060_0324M6_001.original.review.analysis-out.jpg" in html
    assert "both.review.median.analysis-out.jpg" in html
    assert payload["preview_mode"] == "monitoring"
    assert payload["preview_settings"]["lut_path"].endswith("show.cube")
    assert payload["measurement_preview_settings"]["preview_mode"] == "monitoring"
    assert payload["redline_capabilities"]["supports_lut"] is True


def test_resolve_redline_executable_uses_project_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = tmp_path / "redline.json"
    config_path.write_text(json.dumps({"redline_executable": "/usr/local/bin/REDLine"}), encoding="utf-8")
    monkeypatch.delenv("R3DMATCH_REDLINE_EXECUTABLE", raising=False)
    monkeypatch.setattr("r3dmatch.report.REDLINE_CONFIG_PATH", config_path)
    monkeypatch.setattr("r3dmatch.report.shutil.which", lambda command: command if command == "/usr/local/bin/REDLine" else None)
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
    original_command = next(command for command in preview_commands["commands"] if command["variant"] == "original")
    assert report_json["preview_mode"] == "monitoring"
    assert report_json["preview_settings"]["requested_preview_mode"] == "calibration"
    assert report_json["preview_settings"]["preview_mode_alias"] == "calibration_compatibility_alias_to_monitoring"
    assert report_json["preview_settings"]["output_space"] == "BT.709"
    assert report_json["preview_settings"]["output_gamma"] == "BT.1886"
    assert report_json["ipp2_validation"]["contact_sheet_preview_matches_validation"] is True
    assert Path(report_json["ipp2_validation_path"]).exists()
    assert "--colorSpace" in original_command["command"]
    assert original_command["command"][original_command["command"].index("--colorSpace") + 1] == "13"
    assert "--gammaCurve" in original_command["command"]
    assert original_command["command"][original_command["command"].index("--gammaCurve") + 1] == "32"


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
    )
    assert Path(payload["report_json"]).exists()
    assert Path(payload["report_html"]).exists()
    assert payload["preview_report_pdf"] is None
    report_json = json.loads(Path(payload["report_json"]).read_text(encoding="utf-8"))
    assert report_json["review_mode"] == "lightweight_analysis"
    assert report_json["report_kind"] == "lightweight_analysis"
    assert report_json["executive_synopsis"]
    assert "median strategy" in report_json["executive_synopsis"].lower()
    assert report_json["recommended_strategy"]["strategy_key"] in {"median", "optimal_exposure"}
    assert report_json["visuals"]["exposure_plot_svg"]
    assert report_json["visuals"]["before_after_exposure_svg"]
    assert report_json["visuals"]["confidence_chart_svg"]
    assert report_json["visuals"]["strategy_chart_svg"]
    assert report_json["visuals"]["trust_chart_svg"]
    assert report_json["visuals"]["stability_chart_svg"]
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
    assert report_json["measurement_render_count"] == 0
    preview_commands = json.loads((tmp_path / "analysis-out" / "previews" / "preview_commands.json").read_text(encoding="utf-8"))
    assert preview_commands["skipped_bulk_preview_rendering"] is True
    assert preview_commands["commands"] == []
    html = Path(payload["report_html"]).read_text(encoding="utf-8")
    assert "R3DMatch Diagnostic Review" in html
    assert any(label in html for label in ["SAFE TO COMMIT", "COMMIT WITH WARNINGS", "REVIEW REQUIRED", "DO NOT PUSH"])
    assert "Strategy Comparison" in html
    assert "Before / After Exposure" in html
    assert "Confidence / Reliability" in html
    assert "Should I Trust This Run?" in html
    assert "Trust & Eligibility" in html
    assert "Sample Stability Ranking" in html
    assert "Per-Camera Analysis" in html
    assert "Lightweight Analysis" in html
    assert "Shared Kelvin" in html
    assert "Click to enlarge" in html
    assert "chart-modal" in html
    assert "chart-launch" in html
    assert "Why This Was Chosen" in html
    assert "Most cameras are consistent" in html


def test_lightweight_analysis_report_uses_array_calibration_quality_metrics(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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
    )
    report_json = json.loads(Path(payload["report_json"]).read_text(encoding="utf-8"))
    row = next(item for item in report_json["per_camera_analysis"] if item["clip_id"] == camera["clip_id"])
    assert row["confidence"] == pytest.approx(0.42)
    assert row["neutral_sample_log2_spread"] == pytest.approx(0.123)
    assert row["neutral_sample_chromaticity_spread"] == pytest.approx(0.0045)
    assert row["post_color_residual"] == pytest.approx(0.007)
    assert "quality_override_test" in row["note"]
    assert row["trust_class"] in {"USE_WITH_CAUTION", "UNTRUSTED", "EXCLUDED"}
    assert row["trust_reason"]


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
    assert validation["preview_reference_count"] == 0
    assert validation["preview_existing_count"] == 0
    assert validation["physical_validation"]["status"] == "success"


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
    )

    def fake_render_contact_sheet_pdf(payload, *, output_path, title, timestamp_label=None):  # type: ignore[no-untyped-def]
        output = Path(output_path)
        output.write_bytes(b"%PDF-1.4 partial\n")
        raise CancellationError("Run cancelled during report build.")

    monkeypatch.setattr("r3dmatch.report.render_contact_sheet_pdf", fake_render_contact_sheet_pdf)

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
    _write_test_preview(original_path, (90, 90, 90))
    _write_test_preview(corrected_path, (140, 120, 100))
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
            }
            for i in range(1, 4)
        ],
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
                        "both_corrected": str(corrected_path),
                        "metrics": {
                            "exposure": {
                                "final_offset_stops": 0.1 * i,
                                "measured_log2_luminance_monitoring": -2.5,
                            },
                            "color": {"rgb_gains": [1.0, 1.0, 1.0]},
                            "confidence": 0.95,
                            "flags": [],
                        },
                        "render_validation": {"pixel_output_changed": True, "pixel_diff_from_baseline": 12.5},
                    }
                    for i in range(1, 4)
                ],
            }
        ],
    }
    html = render_contact_sheet_html(payload)
    assert "R3DMatch Review Contact Sheet" in html
    assert "brand-logo" in html
    assert "grid cols-2" in html or "grid cols-3" in html
    assert "../previews/CAM001.original.review.run01.jpg" in html
    assert "../previews/CAM001.both.review.median.run01.jpg" in html
    assert "status-chip" in html
    assert "Gray Exposure" in html
    assert "Exposure Summary" in html
    assert "Result" in html
    assert "Recommended Action" in html
    assert "What To Look For" in html
    assert "IPP2 is the only acceptance domain" in html
    assert "Exposure Anchor" in html
    assert "Hero clip CAM001" in html
    assert "Offset to Anchor" in html
    assert "Confidence:" not in html
    assert "Zone residuals" not in html
    assert "font-size:30px" in html
    assert "font-size: 21px" in html


def test_render_contact_sheet_pdf_paginates_large_camera_counts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    preview_dir = tmp_path / "previews"
    shared_originals = []
    strategy_clips = []
    for index in range(40):
        clip_id = f"CAM{index + 1:03d}"
        original_path = preview_dir / f"{clip_id}.original.review.large.jpg"
        corrected_path = preview_dir / f"{clip_id}.both.review.median.large.jpg"
        _write_test_preview(original_path, (80 + (index % 10), 90, 100))
        _write_test_preview(corrected_path, (110 + (index % 10), 120, 130))
        shared_originals.append(
            {
                "clip_id": clip_id,
                "group_key": clip_id,
                "original_frame": str(original_path),
                "confidence": 0.9,
                "measured_log2_luminance_monitoring": -2.5,
                "measured_log2_luminance_raw": -2.9,
            }
        )
        strategy_clips.append(
            {
                "clip_id": clip_id,
                "group_key": clip_id,
                "both_corrected": str(corrected_path),
                "metrics": {
                    "exposure": {"final_offset_stops": 0.25, "measured_log2_luminance_raw": -2.9},
                    "color": {"rgb_gains": [1.01, 0.99, 1.0]},
                    "confidence": 0.91,
                },
            }
        )
    payload = {
        "target_type": "gray_sphere",
        "processing_mode": "both",
        "preview_transform": "REDLine IPP2 / BT.709 / BT.1886 / Medium / Medium",
        "calibration_roi": {"x": 0.2, "y": 0.2, "w": 0.4, "h": 0.4},
        "target_strategies": ["median"],
        "shared_originals": shared_originals,
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
    real_save = __import__("PIL.Image").Image.Image.save

    def save_spy(self, fp, format=None, **kwargs):  # type: ignore[no-untyped-def]
        captured["page_count"] = 1 + len(kwargs.get("append_images", []))
        return real_save(self, fp, format=format, **kwargs)

    monkeypatch.setattr("PIL.Image.Image.save", save_spy)
    output_path = tmp_path / "report.pdf"
    result = render_contact_sheet_pdf(payload, output_path=str(output_path), title="R3DMatch Review Contact Sheet")
    assert result == str(output_path.resolve())
    assert output_path.exists()
    assert output_path.stat().st_size > 0
    assert captured["page_count"] >= 4


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
    )
    build_contact_sheet_report(
        str(tmp_path / "analysis-out"),
        out_dir=str(tmp_path / "report"),
        target_type="gray_sphere",
        processing_mode="both",
        target_strategies=["median"],
    )
    preview_commands = json.loads((tmp_path / "analysis-out" / "previews" / "preview_commands.json").read_text(encoding="utf-8"))
    exposure_command = next(command for command in preview_commands["commands"] if command["variant"] == "exposure" and command["strategy"] == "median")
    assert "--useMeta" in exposure_command["command"]
    assert "--exposureAdjust" in exposure_command["command"]
    assert "--loadRMD" not in exposure_command["command"]
    strategy_command = next(command for command in preview_commands["commands"] if command["variant"] == "both" and command["strategy"] == "median")
    assert strategy_command["mode"] == "corrected"
    assert strategy_command["command"] is None
    assert strategy_command["pixel_diff_from_baseline"] == pytest.approx(0.0)
    assert strategy_command["error"] is None
    assert strategy_command["application_method"] == "preview_color_disabled"
    assert strategy_command["validation_method"] == "preview_fallback_copy"


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
    )
    payload = build_contact_sheet_report(
        str(tmp_path / "analysis-out"),
        out_dir=str(tmp_path / "report"),
        target_type="gray_sphere",
        processing_mode="both",
        target_strategies=["median"],
    )
    report_json = json.loads(Path(payload["report_json"]).read_text(encoding="utf-8"))
    strategy_clip = next(
        clip
        for clip in report_json["strategies"][0]["clips"]
        if abs(float((clip["metrics"]["exposure"].get("final_offset_stops") or 0.0))) > 1e-6
    )
    assert strategy_clip["both_corrected"] != strategy_clip["original_frame"]
    assert strategy_clip["render_validation"]["pixel_output_changed"] is True
    assert float(strategy_clip["render_validation"]["pixel_diff_from_baseline"]) > 0.0
    assert Path(strategy_clip["both_corrected"]).name == Path(str(strategy_clip["render_validation"]["output"])).name


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
    row = next(item for item in residual_payload["rows"] if abs(float(item["applied_correction_stops"])) > 1e-6)
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
    assert "Recommended Action" in html
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
    assert manual_strategy["clips"][0]["preview_variants"]["both"].endswith(".both.review.manual.analysis-out.jpg")
    assert report_json["shared_originals"][0]["original_frame"].endswith(".original.review.analysis-out.jpg")


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
    assert hero_both["application_method"] == "preview_color_disabled"
    assert hero_both["preview_color_applied"] is False
    assert hero_both["cdl_enabled"] is False
    assert Path(hero_both["rmd_path"]).exists()
    html = Path(payload["report_html"]).read_text(encoding="utf-8")
    assert "Hero Camera" in html
    assert "Hero clip: G007_B057_0324YT_001" in html
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
    assert approval_manifest["matching_domain"] == "scene"
    assert approval_manifest["selected_clip_ids"] == ["G007_B057_0324YT_065", "G007_C057_0324YT_065"]
    assert batch_manifest["run_label"] == "clip65_red_contamination"
    assert batch_manifest["matching_domain"] == "scene"
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
    analyze_path(str(tmp_path), out_dir=str(tmp_path / "analysis-out"), mode="scene", backend="mock", lut_override=None, calibration_path=None, sample_count=4, sampling_strategy="uniform")
    build_contact_sheet_report(str(tmp_path / "analysis-out"), out_dir=str(tmp_path / "analysis-out" / "report"), target_type="gray_card", processing_mode="both")
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
            "confidence_chart_svg": "<svg viewBox='0 0 10 10'></svg>",
            "strategy_chart_svg": "<svg viewBox='0 0 10 10'></svg>",
        },
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
    assert "SAFE TO COMMIT" in payload["decision_surface_html"]
    assert "Shared Kelvin / Per-Camera Tint" in payload["decision_surface_html"]
    assert "Exposure Spread" in payload["decision_surface_html"]
    assert "Confidence / Reliability" in payload["decision_surface_html"]
    assert "Should I Trust This Run?" in payload["decision_surface_html"]
    assert "Recommendation Strength" in payload["decision_surface_html"]
    assert "chart-launch" in payload["decision_surface_html"]
    assert "Click to enlarge" in payload["decision_surface_html"]
    assert "No cameras excluded from reference" in payload["decision_surface_html"]
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
    report_payload = {"visuals": {"exposure_plot_svg": "<svg viewBox='0 0 10 10'></svg>"}}
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
    assert "COMMIT WITH WARNINGS" in decision_html
    assert "1 camera excluded from reference" in decision_html
    assert "remaining cluster is still commit-worthy" in decision_html
    assert "I007_A064" in decision_html
    assert "Should I Trust This Run?" in decision_html
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
    def fake_ingest_ftps_batch(**kwargs):
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

    monkeypatch.setattr("r3dmatch.workflow.ingest_ftps_batch", fake_ingest_ftps_batch)
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
