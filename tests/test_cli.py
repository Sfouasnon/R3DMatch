import json
import types
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from r3dmatch.calibration import build_array_calibration_from_analysis, calibrate_card_path, calibrate_color_path, calibrate_exposure_path, calibrate_sphere_path, center_crop_roi, derive_array_group_key, detect_gray_card_roi, load_calibration, load_card_roi_file, load_color_calibration, load_exposure_calibration, measure_card_from_roi, measure_color_region, measure_sphere_from_roi, solve_neutral_gains
from r3dmatch.cli import app
from r3dmatch.desktop_app import build_review_command, run_ui_self_check, scan_calibration_sources
from r3dmatch.identity import group_key_from_clip_id, rmd_name_for_clip_id
from r3dmatch.matching import analyze_path, camera_group_from_clip_id, discover_clips, measure_frame_color_and_exposure
from r3dmatch.models import GrayCardROI, SamplingRegion, SphereROI
from r3dmatch.report import _build_strategy_payloads, _compute_image_difference_metrics, build_contact_sheet_report, preview_filename_for_clip_id, render_preview_frame
from r3dmatch.rmd import render_rmd_xml, rmd_filename_for_clip_id, write_rmds_from_analysis
from r3dmatch.ui import build_table_rows, load_review_bundle
from r3dmatch.sdk import MockR3DBackend, RedSdkDecoder, resolve_backend
from r3dmatch.sidecar import build_sidecar_payload, sidecar_filename_for_clip_id
from r3dmatch.transcode import build_redline_command, build_redline_command_variants, write_transcode_plan
from r3dmatch.validation import validate_pipeline
from r3dmatch.web_app import build_review_web_command, create_app, scan_sources
from r3dmatch.workflow import approve_master_rmd, clear_preview_cache, review_calibration

runner = CliRunner()


def _install_fake_redline(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(command, capture_output, text, check):  # type: ignore[no-untyped-def]
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
                return types.SimpleNamespace(returncode=0, stdout="RMD metadata loaded\n", stderr="")
            return types.SimpleNamespace(returncode=0, stdout="CLI metadata loaded\n", stderr="")
        output_path = Path(command[command.index("--o") + 1])
        generated_path = output_path.with_name(f"{output_path.name}.000000.jpg")
        generated_path.parent.mkdir(parents=True, exist_ok=True)
        exposure = 0.0
        if "--exposure" in command:
            exposure = float(command[command.index("--exposure") + 1])
        red_gain = float(command[command.index("--redGain") + 1]) if "--redGain" in command else 1.0
        green_gain = float(command[command.index("--greenGain") + 1]) if "--greenGain" in command else 1.0
        blue_gain = float(command[command.index("--blueGain") + 1]) if "--blueGain" in command else 1.0
        base = np.zeros((18, 32, 3), dtype=np.uint8)
        base[..., 0] = np.clip(80.0 * (2.0**exposure) * red_gain, 0, 255)
        base[..., 1] = np.clip(90.0 * (2.0**exposure) * green_gain, 0, 255)
        base[..., 2] = np.clip(100.0 * (2.0**exposure) * blue_gain, 0, 255)
        Image.fromarray(base, mode="RGB").save(generated_path, format="JPEG")
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("r3dmatch.report.subprocess.run", fake_run)


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


def test_build_review_web_command_uses_tcsh_and_preview_options() -> None:
    command = build_review_web_command(
        "/Users/sfouasnon/Desktop/R3DMatch",
        {
            "input_path": "/tmp/in",
            "output_path": "/tmp/out",
            "backend": "red",
            "target_type": "gray_sphere",
            "processing_mode": "both",
            "roi_x": "0.1",
            "roi_y": "0.2",
            "roi_w": "0.3",
            "roi_h": "0.4",
            "target_strategies": ["median", "manual"],
            "reference_clip_id": "G007_D060_0324M6_001",
            "preview_mode": "monitoring",
            "preview_lut": "/tmp/show.cube",
        },
    )
    joined = " ".join(command)
    assert command[0] == "/bin/tcsh"
    assert "setenv PYTHONPATH" in joined
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
            "preview_mode": "calibration",
            "roi_mode": "center",
        },
    )
    assert run_response.status_code == 200
    assert started["output_path"] == str(out_dir)
    assert "review-calibration" in " ".join(started["command"])
    assert started["kwargs"]["clip_count"] == 1
    assert started["kwargs"]["strategies_text"] == "median"
    assert started["kwargs"]["preview_mode"] == "calibration"


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


def test_target_strategies_use_monitoring_domain_values() -> None:
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
    assert strategies[0]["reference_clip_id"] == "G007_C057_0324YT_001"
    assert strategies[0]["target_log2_luminance"] == pytest.approx(-1.5)
    assert strategies[1]["reference_clip_id"] == "G007_B057_0324YT_001"
    assert strategies[1]["target_log2_luminance"] == pytest.approx(-3.0)


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
    assert (review_dir / "approval" / "Master_RMD" / "G007_D060_0324M6_001.RMD").exists()
    assert (review_dir / "approval" / "approval_manifest.json").exists()
    assert (review_dir / "approval" / "calibration_report.pdf").exists()
    approval_manifest = json.loads((review_dir / "approval" / "approval_manifest.json").read_text(encoding="utf-8"))
    assert approval_manifest["selected_target_strategy"] == "brightest_valid"


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
    assert report_json["exposure_measurement_domain"] == "monitoring"
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
    strategy_command = next(command for command in preview_commands["commands"] if command["variant"] == "both" and command["strategy"] == "median")
    assert strategy_command["look_metadata_path"].endswith("/review_rmd/strategies/median/both/G007_D060_0324M6_001.RMD")
    assert Path(strategy_command["look_metadata_path"]).exists()
    assert strategy_command["mode"] == "corrected"
    assert strategy_command["rmd_metadata_probe"]["returncode"] == 0
    assert "--loadRMD" not in strategy_command["command"]
    assert "--exposure" in strategy_command["command"]
    assert "--useMeta" not in strategy_command["command"]
    assert strategy_command["pixel_diff_from_baseline"] == pytest.approx(0.0)
    assert strategy_command["pixel_output_changed"] is False
    assert strategy_command["error"] is None
    assert strategy_command["as_shot_metadata_used"] is False
    assert strategy_command["explicit_transform_used"] is True
    assert strategy_command["explicit_correction_flags_used"] is False
    assert "--lut" in strategy_command["command"]
    html = Path(payload["report_html"]).read_text(encoding="utf-8")
    assert "G007_D060_0324M6_001" in html
    assert "../previews/G007_D060_0324M6_001.original.review.analysis-out.jpg" in html
    assert "both.review.median.analysis-out.jpg" in html
    assert payload["preview_mode"] == "monitoring"
    assert payload["preview_settings"]["lut_path"].endswith("show.cube")
    assert payload["measurement_preview_settings"]["preview_mode"] == "calibration"
    assert payload["redline_capabilities"]["supports_lut"] is True


def test_report_preview_falls_back_to_cli_flags_when_rmd_probe_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(command, capture_output, text, check):  # type: ignore[no-untyped-def]
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
        if "--printMeta" in command and "--loadRMD" in command:
            return types.SimpleNamespace(returncode=1, stdout="", stderr="Error parsing RMD file")
        if "--printMeta" in command:
            return types.SimpleNamespace(returncode=0, stdout="CLI metadata loaded\n", stderr="")
        output_path = Path(command[command.index("--o") + 1])
        generated_path = output_path.with_name(f"{output_path.name}.000000.jpg")
        generated_path.parent.mkdir(parents=True, exist_ok=True)
        exposure = float(command[command.index("--exposure") + 1]) if "--exposure" in command else 0.0
        red_gain = float(command[command.index("--redGain") + 1]) if "--redGain" in command else 1.0
        green_gain = float(command[command.index("--greenGain") + 1]) if "--greenGain" in command else 1.0
        blue_gain = float(command[command.index("--blueGain") + 1]) if "--blueGain" in command else 1.0
        image = np.zeros((18, 32, 3), dtype=np.uint8)
        image[..., 0] = np.clip(70.0 * (2.0**exposure) * red_gain, 0, 255)
        image[..., 1] = np.clip(80.0 * (2.0**exposure) * green_gain, 0, 255)
        image[..., 2] = np.clip(90.0 * (2.0**exposure) * blue_gain, 0, 255)
        Image.fromarray(image, mode="RGB").save(generated_path, format="JPEG")
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("r3dmatch.report.subprocess.run", fake_run)
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
    strategy_command = next(command for command in preview_commands["commands"] if command["variant"] == "both" and command["strategy"] == "median")
    assert strategy_command["mode"] == "corrected"
    assert strategy_command["rmd_metadata_probe"]["returncode"] == 1
    assert "--loadRMD" not in strategy_command["command"]
    assert "--exposure" in strategy_command["command"]
    assert "--useMeta" not in strategy_command["command"]
    assert strategy_command["pixel_diff_from_baseline"] == pytest.approx(0.0)
    assert strategy_command["error"] is None


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
    assert (Path(payload["master_rmd_dir"]) / "G007_B057_0324YT_001.RMD").exists()


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
    (approval_dir / "Master_RMD").mkdir()
    payload = clear_preview_cache(str(analysis_root))
    assert payload["removed_count"] >= 3
    assert not (previews / "G007_D060_0324M6_001.original.review.jpg").exists()
    assert not (report_dir / "contact_sheet.html").exists()
    assert (analysis_root / "summary.json").exists()
    assert (analysis_root / "array_calibration.json").exists()
    assert (approval_dir / "calibration_report.pdf").exists()
    assert (approval_dir / "Master_RMD").exists()


def test_approve_master_rmd_creates_manifest_pdf_and_exact_names(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_redline(monkeypatch)
    clip_path = tmp_path / "G007_D060_0324M6_001.R3D"
    clip_path.write_bytes(b"")
    analyze_path(str(tmp_path), out_dir=str(tmp_path / "analysis-out"), mode="scene", backend="mock", lut_override=None, calibration_path=None, sample_count=4, sampling_strategy="uniform")
    build_contact_sheet_report(str(tmp_path / "analysis-out"), out_dir=str(tmp_path / "analysis-out" / "report"), target_type="gray_card", processing_mode="both")
    payload = approve_master_rmd(str(tmp_path / "analysis-out"))
    assert Path(payload["approval_manifest"]).exists()
    assert Path(payload["calibration_report_pdf"]).exists()
    assert (Path(payload["master_rmd_dir"]) / "G007_D060_0324M6_001.RMD").exists()


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
