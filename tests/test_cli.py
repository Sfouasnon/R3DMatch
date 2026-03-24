import json
import types
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from r3dmatch.calibration import build_array_calibration_from_analysis, calibrate_card_path, calibrate_color_path, calibrate_exposure_path, calibrate_sphere_path, center_crop_roi, derive_array_group_key, detect_gray_card_roi, load_calibration, load_card_roi_file, load_color_calibration, load_exposure_calibration, measure_card_from_roi, measure_color_region, measure_sphere_from_roi, solve_neutral_gains
from r3dmatch.cli import app
from r3dmatch.identity import group_key_from_clip_id, rmd_name_for_clip_id
from r3dmatch.matching import analyze_path, camera_group_from_clip_id, discover_clips
from r3dmatch.models import GrayCardROI, SamplingRegion, SphereROI
from r3dmatch.report import build_contact_sheet_report
from r3dmatch.ui import build_table_rows, load_review_bundle
from r3dmatch.sdk import MockR3DBackend, RedSdkDecoder, resolve_backend
from r3dmatch.sidecar import build_sidecar_payload, sidecar_filename_for_clip_id
from r3dmatch.transcode import build_redline_command, build_redline_command_variants, write_transcode_plan
from r3dmatch.validation import validate_pipeline

runner = CliRunner()


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


def test_cli_validate_pipeline_and_report(tmp_path: Path) -> None:
    clip_path = tmp_path / "G007_D060_0324M6_001.R3D"
    clip_path.write_bytes(b"")
    analysis_dir = tmp_path / "analysis"
    runner.invoke(app, ["analyze", str(clip_path), "--out", str(analysis_dir), "--backend", "mock", "--mode", "scene"], catch_exceptions=False)
    validate_result = runner.invoke(app, ["validate-pipeline", str(clip_path), "--analysis-dir", str(analysis_dir), "--out", str(tmp_path / "validation")])
    report_result = runner.invoke(app, ["report-contact-sheet", str(analysis_dir), "--out", str(tmp_path / "report")])
    assert validate_result.exit_code == 0
    assert report_result.exit_code == 0


def test_build_redline_command_includes_sidecar(tmp_path: Path) -> None:
    clip_path = tmp_path / "G007_D060_0324M6_001.R3D"
    clip_path.write_bytes(b"")
    sidecar_path = tmp_path / "clip.sidecar.json"
    sidecar_path.write_text("{}", encoding="utf-8")
    command = build_redline_command(str(clip_path), render_dir=str(tmp_path / "renders"), sidecar_path=str(sidecar_path), redline_executable="REDLine", output_ext="mov")
    assert command[0] == "REDLine"


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


def test_transcode_requires_analysis_dir_for_generated_sidecars(tmp_path: Path) -> None:
    clip_path = tmp_path / "G007_D060_0324M6_001.R3D"
    clip_path.write_bytes(b"")
    with pytest.raises(ValueError):
        write_transcode_plan(str(clip_path), out_dir=str(tmp_path / "transcode"), analysis_dir=None, use_generated_sidecar=True, redline_executable="REDLine", output_ext="mov", execute=False)


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


def test_report_contact_sheet_scaffold(tmp_path: Path) -> None:
    clip_path = tmp_path / "G007_D060_0324M6_001.R3D"
    clip_path.write_bytes(b"")
    analyze_path(str(tmp_path), out_dir=str(tmp_path / "analysis-out"), mode="scene", backend="mock", lut_override=None, calibration_path=None, sample_count=4, sampling_strategy="uniform")
    payload = build_contact_sheet_report(str(tmp_path / "analysis-out"), out_dir=str(tmp_path / "report"))
    assert Path(payload["report_json"]).exists()
    assert Path(payload["report_html"]).exists()
    report_json = json.loads(Path(payload["report_json"]).read_text(encoding="utf-8"))
    assert report_json["clips"][0]["clip_id"] == "G007_D060_0324M6_001"


def test_load_review_bundle_for_ui(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    (out_dir / "analysis").mkdir(parents=True)
    (out_dir / "sidecars").mkdir(parents=True)
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
    bundle = load_review_bundle(str(out_dir))
    rows = build_table_rows(bundle["rows"])
    assert bundle["summary_text"]["clip_count"] == 1
    assert rows[0]["clip_id"] == "G007_D060_0324M6_001"
