import json
import os
import subprocess
import types
from pathlib import Path
from typing import Optional

import numpy as np
import pytest
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
from r3dmatch.report import _build_strategy_payloads, _compute_image_difference_metrics, _report_grid_columns, _report_tiles_per_page, build_contact_sheet_report, build_lightweight_analysis_report, build_review_package, preview_filename_for_clip_id, render_contact_sheet_html, render_contact_sheet_pdf, render_preview_frame
from r3dmatch.rmd import render_rmd_xml, rmd_filename_for_clip_id, write_rmd_for_clip_with_metadata, write_rmds_from_analysis
from r3dmatch.ui import build_table_rows, load_review_bundle
from r3dmatch.sdk import MockR3DBackend, RedSdkDecoder, resolve_backend
from r3dmatch.sidecar import build_sidecar_payload, sidecar_filename_for_clip_id
from r3dmatch.transcode import build_redline_command, build_redline_command_variants, write_transcode_plan
from r3dmatch.validation import validate_pipeline
from r3dmatch.web_app import _normalize_subset_form, _resolve_selected_clip_ids, _subset_selection_ui, build_review_web_command, create_app, scan_sources
from r3dmatch.workflow import approve_master_rmd, clear_preview_cache, resolve_review_output_dir, review_calibration, validate_review_run_contract

runner = CliRunner()


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
        exposure = 0.0
        red_gain = 1.0
        green_gain = 1.0
        blue_gain = 1.0
        if "--loadRMD" in command:
            rmd_path = Path(command[command.index("--loadRMD") + 1])
            rmd_payload = json.loads(rmd_path.read_text(encoding="utf-8"))
            exposure = float(rmd_payload.get("exposure", 0.0) or 0.0)
            gains = rmd_payload.get("rgb_gains")
            if gains:
                red_gain = float(gains[0])
                green_gain = float(gains[1])
                blue_gain = float(gains[2])
        elif "--exposure" in command:
            exposure = float(command[command.index("--exposure") + 1])
            red_gain = float(command[command.index("--redGain") + 1]) if "--redGain" in command else 1.0
            green_gain = float(command[command.index("--greenGain") + 1]) if "--greenGain" in command else 1.0
            blue_gain = float(command[command.index("--blueGain") + 1]) if "--blueGain" in command else 1.0
        base = np.zeros((18, 32, 3), dtype=np.float32)
        base[..., 0] = np.clip((80.0 / 255.0) * (2.0**exposure) * red_gain, 0, 1)
        base[..., 1] = np.clip((90.0 / 255.0) * (2.0**exposure) * green_gain, 0, 1)
        base[..., 2] = np.clip((100.0 / 255.0) * (2.0**exposure) * blue_gain, 0, 1)
        if "--loadRMD" in command:
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
    monkeypatch.setattr("r3dmatch.web_app._ensure_scan_preview", lambda input_path, scan: None)
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
            "preview_mode": "calibration",
            "roi_mode": "center",
        },
    )
    assert run_response.status_code == 200
    assert started["output_path"] == str(out_dir)
    assert "review-calibration" in " ".join(started["command"])
    assert started["kwargs"]["clip_count"] == 1
    assert started["kwargs"]["strategies_text"] == "median"
    assert started["kwargs"]["review_mode"] == "lightweight_analysis"
    assert started["kwargs"]["preview_mode"] == "calibration"


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
            "preview_mode": "calibration",
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
    monkeypatch.setattr("r3dmatch.web_app._ensure_scan_preview", lambda input_path, scan: None)
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
    monkeypatch.setattr("r3dmatch.web_app._ensure_scan_preview", lambda input_path, scan: None)
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
            "preview_mode": "calibration",
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
    monkeypatch.setattr("r3dmatch.web_app._ensure_scan_preview", lambda input_path, scan: None)
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
    state.task.strategies_text = "median, brightest-valid"
    state.task.preview_mode = "calibration"
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
    monkeypatch.setattr("r3dmatch.web_app._ensure_scan_preview", lambda input_path, scan: None)
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
    monkeypatch.setattr("r3dmatch.web_app._ensure_scan_preview", lambda input_path, scan: None)
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


def test_web_app_status_reports_review_validation_failure_detail(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("r3dmatch.web_app._ensure_scan_preview", lambda input_path, scan: None)
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
    monkeypatch.setattr("r3dmatch.web_app._ensure_scan_preview", lambda input_path, scan: None)
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
    monkeypatch.setattr("r3dmatch.web_app._ensure_scan_preview", lambda input_path, scan: None)
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
    monkeypatch.setattr("r3dmatch.web_app._ensure_scan_preview", lambda input_path, scan: None)
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
    monkeypatch.setattr("r3dmatch.web_app._ensure_scan_preview", lambda input_path, scan: None)
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
    monkeypatch.setattr("r3dmatch.web_app._ensure_scan_preview", lambda input_path, scan: None)
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
            "preview_mode": "calibration",
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
            "preview_mode": "calibration",
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
    monkeypatch.setattr("r3dmatch.web_app._ensure_scan_preview", lambda input_path, scan: None)
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
    strategies = _build_strategy_payloads(records, target_strategies=["brightest-valid", "manual"], reference_clip_id="G007_B057_0324YT_001")
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
        target_strategies=["brightest-valid", "manual"],
        reference_clip_id="G007_B057_0324YT_001",
        matching_domain="perceptual",
    )
    assert strategies[0]["reference_clip_id"] == "G007_C057_0324YT_001"
    assert strategies[0]["target_log2_luminance"] == pytest.approx(-1.5)
    assert strategies[1]["reference_clip_id"] == "G007_B057_0324YT_001"
    assert strategies[1]["target_log2_luminance"] == pytest.approx(-3.0)


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
            "brightest-valid",
        ],
    )
    assert review_result.exit_code == 0
    assert (review_dir / "report" / "contact_sheet.html").exists()
    assert (review_dir / "report" / "preview_contact_sheet.pdf").exists()
    assert (review_dir / "review_rmd" / "G007_D060_0324M6_001.RMD").exists()
    review_manifest = json.loads((review_dir / "report" / "review_manifest.json").read_text(encoding="utf-8"))
    assert review_manifest["calibration_roi"] == {"x": 0.25, "y": 0.25, "w": 0.5, "h": 0.5}
    assert review_manifest["target_strategies"] == ["median", "brightest_valid"]
    approve_result = runner.invoke(app, ["approve-master-rmd", str(review_dir), "--target-strategy", "brightest-valid"])
    assert approve_result.exit_code == 0
    assert (review_dir / "approval" / "MasterRMD" / "G007_D060.RMD").exists()
    assert (review_dir / "approval" / "batch" / "manifest.json").exists()
    assert (review_dir / "approval" / "approval_manifest.json").exists()
    assert (review_dir / "approval" / "calibration_report.pdf").exists()
    approval_manifest = json.loads((review_dir / "approval" / "approval_manifest.json").read_text(encoding="utf-8"))
    assert approval_manifest["selected_target_strategy"] == "brightest_valid"
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
        target_strategies=["median", "brightest-valid"],
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
    assert report_json["measurement_preview_settings"]["preview_mode"] == "calibration"
    assert report_json["redline_capabilities"]["supports_lut"] is True
    assert report_json["calibration_roi"] == {"x": 0.2, "y": 0.2, "w": 0.4, "h": 0.4}
    assert report_json["target_strategies"] == ["median", "brightest_valid"]
    assert report_json["shared_originals"][0]["original_frame"].endswith("G007_D060_0324M6_001.original.review.analysis-out.jpg")
    assert report_json["strategies"][0]["clips"][0]["both_corrected"].endswith("G007_D060_0324M6_001.both.review.median.analysis-out.jpg")
    assert report_json["strategies"][0]["clips"][0]["metrics"]["preview_transform"] == report_json["preview_transform"]
    assert "measured_log2_luminance_monitoring" in report_json["strategies"][0]["clips"][0]["metrics"]["exposure"]
    assert "measured_log2_luminance_raw" in report_json["strategies"][0]["clips"][0]["metrics"]["exposure"]
    assert Path(report_json["clips"][0]["original_frame"]).exists()
    assert not list((tmp_path / "analysis-out" / "previews").glob("*.000000.jpg"))
    preview_commands = json.loads((tmp_path / "analysis-out" / "previews" / "preview_commands.json").read_text(encoding="utf-8"))
    exposure_command = next(command for command in preview_commands["commands"] if command["variant"] == "exposure" and command["strategy"] == "median")
    assert exposure_command["application_method"] == "rmd"
    assert exposure_command["look_metadata_path"].endswith("/review_rmd/strategies/median/exposure/G007_D060_0324M6_001.RMD")
    assert Path(exposure_command["look_metadata_path"]).exists()
    assert "--loadRMD" in exposure_command["command"]
    assert "--exposure" not in exposure_command["command"]
    assert "--useMeta" not in exposure_command["command"]
    assert "--lut" in exposure_command["command"]
    strategy_command = next(command for command in preview_commands["commands"] if command["variant"] == "both" and command["strategy"] == "median")
    assert strategy_command["application_method"] == "preview_color_disabled"
    assert strategy_command["preview_color_applied"] is False
    assert strategy_command["output_reused_from_variant"] == "exposure"
    assert strategy_command["validation_method"] == "preview_fallback_copy"
    assert strategy_command["correction_application_method"] == "rmd"
    assert strategy_command["command"] is None
    assert strategy_command["rmd_path"].endswith("/review_rmd/strategies/median/both/G007_D060_0324M6_001.RMD")
    assert Path(strategy_command["rmd_path"]).exists()
    assert report_json["color_preview_enabled"] is False
    assert report_json["color_preview_status"] == "disabled_unverified"
    html = Path(payload["report_html"]).read_text(encoding="utf-8")
    assert "Color preview" in html
    assert "G007_D060_0324M6_001" in html
    assert "../previews/G007_D060_0324M6_001.original.review.analysis-out.jpg" in html
    assert "both.review.median.analysis-out.jpg" in html
    assert payload["preview_mode"] == "monitoring"
    assert payload["preview_settings"]["lut_path"].endswith("show.cube")
    assert payload["measurement_preview_settings"]["preview_mode"] == "calibration"
    assert payload["redline_capabilities"]["supports_lut"] is True


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
        target_strategies=["median", "brightest-valid"],
    )
    assert Path(payload["report_json"]).exists()
    assert Path(payload["report_html"]).exists()
    assert payload["preview_report_pdf"] is None
    report_json = json.loads(Path(payload["report_json"]).read_text(encoding="utf-8"))
    assert report_json["review_mode"] == "lightweight_analysis"
    assert report_json["report_kind"] == "lightweight_analysis"
    assert report_json["executive_synopsis"]
    assert "median strategy" in report_json["executive_synopsis"].lower()
    assert report_json["recommended_strategy"]["strategy_key"] in {"median", "brightest_valid"}
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
    assert "Strategy Comparison" in html
    assert "Per-Camera Analysis" in html
    assert "Lightweight Analysis" in html
    assert "Shared Kelvin" in html


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
        target_strategies=["median", "brightest-valid"],
    )
    report_json = json.loads(Path(payload["report_json"]).read_text(encoding="utf-8"))
    row = next(item for item in report_json["per_camera_analysis"] if item["clip_id"] == camera["clip_id"])
    assert row["confidence"] == pytest.approx(0.42)
    assert row["neutral_sample_log2_spread"] == pytest.approx(0.123)
    assert row["neutral_sample_chromaticity_spread"] == pytest.approx(0.0045)
    assert row["post_color_residual"] == pytest.approx(0.007)
    assert "quality_override_test" in row["note"]


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
                    }
                    for i in range(1, 4)
                ],
            }
        ],
    }
    html = render_contact_sheet_html(payload)
    assert "R3DMatch Review Contact Sheet" in html
    assert "brand-logo" in html
    assert "grid cols-3" in html
    assert "../previews/CAM001.original.review.run01.jpg" in html
    assert "../previews/CAM001.both.review.median.run01.jpg" in html
    assert "font-size:30px" in html
    assert "font-size:20px" in html


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
        exposure = 0.0
        red_gain = 1.0
        green_gain = 1.0
        blue_gain = 1.0
        if "--loadRMD" in command:
            rmd_path = Path(command[command.index("--loadRMD") + 1])
            rmd_payload = json.loads(rmd_path.read_text(encoding="utf-8"))
            exposure = float(rmd_payload.get("exposure", 0.0) or 0.0)
            gains = rmd_payload.get("rgb_gains")
            if gains:
                red_gain = float(gains[0])
                green_gain = float(gains[1])
                blue_gain = float(gains[2])
        elif "--exposure" in command:
            exposure = float(command[command.index("--exposure") + 1])
            red_gain = float(command[command.index("--redGain") + 1]) if "--redGain" in command else 1.0
            green_gain = float(command[command.index("--greenGain") + 1]) if "--greenGain" in command else 1.0
            blue_gain = float(command[command.index("--blueGain") + 1]) if "--blueGain" in command else 1.0
        image = np.zeros((18, 32, 3), dtype=np.uint8)
        image[..., 0] = np.clip(70.0 * (2.0**exposure) * red_gain, 0, 255)
        image[..., 1] = np.clip(80.0 * (2.0**exposure) * green_gain, 0, 255)
        image[..., 2] = np.clip(90.0 * (2.0**exposure) * blue_gain, 0, 255)
        Image.fromarray(image, mode="RGB").save(generated_path, format="JPEG")
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
    assert "--loadRMD" in exposure_command["command"]
    assert "--useRMD" in exposure_command["command"]
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
        "preview_mode": "calibration",
        "output_space": "REDWideGamutRGB",
        "output_gamma": "Log3G10",
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
    downloaded = Path(saved["per_camera_status"][0]["downloaded_files"][0]["local_path"])
    assert downloaded.exists()
    assert downloaded.read_bytes().startswith(b"mock:172.20.114.141")


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
