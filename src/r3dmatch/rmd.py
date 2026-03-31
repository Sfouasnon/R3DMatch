from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Optional
from xml.etree import ElementTree as ET

from . import sdk
from .color import rgb_gains_to_cdl
from .identity import rmd_name_for_clip_id


SDK_COLOR_SPACE_REC709 = 1
SDK_GAMMA_BT1886 = 15
SDK_TONEMAP_MEDIUM = 1
SDK_ROLLOFF_MEDIUM = 3
SDK_IMAGE_PIPELINE_FULL_GRADED = 1


def rmd_filename_for_clip_id(clip_id: str) -> str:
    return rmd_name_for_clip_id(clip_id)


def build_rmd_tree_from_sidecar(sidecar_payload: Dict[str, object]) -> ET.Element:
    clip_id = str(sidecar_payload["clip_id"])
    root = ET.Element("RMD", {"schema": "r3dmatch_rmd_subset_v1"})
    ET.SubElement(root, "Clip", {"id": clip_id})
    look = ET.SubElement(root, "Look")
    calibration_state = dict(sidecar_payload.get("calibration_state", {}))
    rmd_mapping = dict(sidecar_payload.get("rmd_mapping", {}))
    exposure = ET.SubElement(look, "Exposure")
    exposure.set("final_offset_stops", str(dict(rmd_mapping.get("exposure", {})).get("final_offset_stops", 0.0)))
    exposure.set("calibration_loaded", str(bool(calibration_state.get("exposure_calibration_loaded"))).lower())
    exposure.set("baseline_applied_stops", str(calibration_state.get("exposure_baseline_applied_stops")))
    color_gains = calibration_state.get("rgb_neutral_gains")
    color = ET.SubElement(look, "Color")
    if color_gains:
        color.set("rgb_neutral_gains", json.dumps(color_gains, separators=(",", ":")))
    color.set("calibration_loaded", str(bool(calibration_state.get("color_calibration_loaded"))).lower())
    color.set("state", str(calibration_state.get("color_gains_state")))
    provenance = ET.SubElement(root, "Provenance")
    provenance.set("clip_id", clip_id)
    provenance.set("source_path", str(sidecar_payload.get("source_path", "")))
    provenance.set("sidecar_schema", str(sidecar_payload.get("schema", "")))
    provenance.set("rmd_name", rmd_filename_for_clip_id(clip_id))
    note = ET.SubElement(root, "Note")
    note.text = "Fallback subset only: this is not a RED-readable RMD."
    return root


def render_rmd_xml(sidecar_payload: Dict[str, object]) -> str:
    root = build_rmd_tree_from_sidecar(sidecar_payload)
    xml = ET.tostring(root, encoding="unicode")
    return "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n" + xml + "\n"


def _clamp(value: float, lower: float, upper: float) -> float:
    return float(max(lower, min(upper, value)))


def _normalize_rgb_gains(sidecar_payload: Dict[str, object]) -> Optional[list[float]]:
    color_mapping = dict(dict(sidecar_payload.get("rmd_mapping", {})).get("color", {}))
    rgb_gains = color_mapping.get("rgb_neutral_gains")
    if rgb_gains is None:
        calibration_state = dict(sidecar_payload.get("calibration_state", {}))
        gains_dict = calibration_state.get("rgb_neutral_gains")
        if isinstance(gains_dict, dict):
            rgb_gains = [gains_dict.get("r", 1.0), gains_dict.get("g", 1.0), gains_dict.get("b", 1.0)]
    if rgb_gains is None:
        return None
    return [_clamp(float(rgb_gains[0]), 0.0, 4.0), _clamp(float(rgb_gains[1]), 0.0, 4.0), _clamp(float(rgb_gains[2]), 0.0, 4.0)]


def _is_identity_cdl(cdl_payload: Dict[str, object]) -> bool:
    slope = [float(value) for value in cdl_payload.get("slope", [1.0, 1.0, 1.0])]
    offset = [float(value) for value in cdl_payload.get("offset", [0.0, 0.0, 0.0])]
    power = [float(value) for value in cdl_payload.get("power", [1.0, 1.0, 1.0])]
    saturation = float(cdl_payload.get("saturation", 1.0))
    return (
        all(abs(value - 1.0) <= 1e-9 for value in slope)
        and all(abs(value) <= 1e-9 for value in offset)
        and all(abs(value - 1.0) <= 1e-9 for value in power)
        and abs(saturation - 1.0) <= 1e-9
    )


def _resolve_cdl_payload(sidecar_payload: Dict[str, object], rgb_gains: Optional[list[float]]) -> Dict[str, object]:
    color_mapping = dict(dict(sidecar_payload.get("rmd_mapping", {})).get("color", {}))
    explicit_cdl = color_mapping.get("cdl")
    if isinstance(explicit_cdl, dict):
        return {
            "slope": [float(value) for value in explicit_cdl.get("slope", [1.0, 1.0, 1.0])],
            "offset": [float(value) for value in explicit_cdl.get("offset", [0.0, 0.0, 0.0])],
            "power": [float(value) for value in explicit_cdl.get("power", [1.0, 1.0, 1.0])],
            "saturation": float(explicit_cdl.get("saturation", 1.0)),
        }
    return rgb_gains_to_cdl(rgb_gains or [1.0, 1.0, 1.0])


def _sdk_settings_from_sidecar(sidecar_payload: Dict[str, object]) -> Dict[str, object]:
    exposure_mapping = dict(dict(sidecar_payload.get("rmd_mapping", {})).get("exposure", {}))
    exposure_adjust = _clamp(float(exposure_mapping.get("final_offset_stops", 0.0) or 0.0), -12.0, 12.0)
    rgb_gains = _normalize_rgb_gains(sidecar_payload)
    cdl = _resolve_cdl_payload(sidecar_payload, rgb_gains)
    cdl_identity = _is_identity_cdl(cdl)
    color_mapping = dict(dict(sidecar_payload.get("rmd_mapping", {})).get("color", {}))
    requested_cdl_enabled = color_mapping.get("cdl_enabled")
    cdl_enabled = bool(not cdl_identity if requested_cdl_enabled is None else requested_cdl_enabled)
    if not cdl_identity and not cdl_enabled:
        raise ValueError("CDL payload is non-identity but cdl_enabled is false; refusing to author an inactive color correction RMD.")
    return {
        "exposure_adjust": exposure_adjust,
        "rgb_gains": rgb_gains,
        "cdl_slope": [float(value) for value in cdl["slope"]],
        "cdl_offset": [float(value) for value in cdl["offset"]],
        "cdl_power": [float(value) for value in cdl["power"]],
        "cdl_saturation": float(cdl["saturation"]),
        "cdl_enabled": cdl_enabled,
        "cdl_identity": cdl_identity,
        "output_tonemap": SDK_TONEMAP_MEDIUM,
        "highlight_rolloff": SDK_ROLLOFF_MEDIUM,
        "color_space": SDK_COLOR_SPACE_REC709,
        "gamma_curve": SDK_GAMMA_BT1886,
        "image_pipeline_mode": SDK_IMAGE_PIPELINE_FULL_GRADED,
    }


def _write_real_rmd_for_clip(clip_id: str, sidecar_payload: Dict[str, object], out_dir: str | Path) -> tuple[Path, Dict[str, object]]:
    native = sdk.load_configured_red_native_module()
    if not hasattr(native, "create_rmd_from_settings"):
        raise RuntimeError("RED SDK bridge does not expose create_rmd_from_settings. Rebuild the native bridge with scripts/build_red_sdk_bridge.sh.")
    if hasattr(native, "sdk_available") and not native.sdk_available():
        raise RuntimeError(getattr(native, "unavailable_message", lambda: "RED SDK bridge unavailable.")())

    source_path = Path(str(sidecar_payload.get("source_path", ""))).expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"Cannot create real RMD because source clip does not exist: {source_path}")

    target_dir = Path(out_dir).expanduser().resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / rmd_filename_for_clip_id(clip_id)
    settings = _sdk_settings_from_sidecar(sidecar_payload)
    with tempfile.TemporaryDirectory(prefix="r3dmatch_rmd_") as temp_dir_raw:
        temp_dir = Path(temp_dir_raw)
        temp_clip_path = temp_dir / source_path.name
        os.symlink(source_path, temp_clip_path)
        result = dict(
            native.create_rmd_from_settings(
                str(temp_clip_path),
                settings["exposure_adjust"],
                settings["cdl_slope"],
                settings["cdl_offset"],
                settings["cdl_power"],
                settings["cdl_saturation"],
                settings["cdl_enabled"],
                settings["output_tonemap"],
                settings["highlight_rolloff"],
                settings["color_space"],
                settings["gamma_curve"],
                settings["image_pipeline_mode"],
            )
        )
        generated_path = Path(str(result.get("rmd_path", ""))).expanduser().resolve()
        if not generated_path.exists():
            raise FileNotFoundError(f"RED SDK reported RMD path but file does not exist: {generated_path}")
        shutil.copy2(generated_path, target_path)
        result["rmd_path"] = str(target_path)
        result["generated_rmd_path"] = str(generated_path)
        result["source_path"] = str(source_path)
        result["settings"] = settings
        result["rmd_kind"] = "red_sdk"
        result["clip_id"] = clip_id
        result["used_temp_symlink"] = True
        result["temp_clip_path"] = str(temp_clip_path)
        return target_path, result


def write_rmd_for_clip(clip_id: str, sidecar_payload: Dict[str, object], out_dir: str | Path) -> Path:
    path, _ = write_rmd_for_clip_with_metadata(clip_id, sidecar_payload, out_dir)
    return path


def write_rmd_for_clip_with_metadata(clip_id: str, sidecar_payload: Dict[str, object], out_dir: str | Path) -> tuple[Path, Dict[str, object]]:
    try:
        return _write_real_rmd_for_clip(clip_id, sidecar_payload, out_dir)
    except ValueError:
        raise
    except Exception as exc:
        target_dir = Path(out_dir).expanduser().resolve()
        target_dir.mkdir(parents=True, exist_ok=True)
        path = target_dir / rmd_filename_for_clip_id(clip_id)
        path.write_text(render_rmd_xml(sidecar_payload), encoding="utf-8")
        return path, {
            "clip_id": clip_id,
            "rmd_path": str(path),
            "rmd_kind": "fallback_xml",
            "error": str(exc),
            "source_path": str(sidecar_payload.get("source_path", "")),
            "settings": _sdk_settings_from_sidecar(sidecar_payload),
        }


def write_rmds_from_analysis(analysis_dir: str, *, out_dir: Optional[str] = None) -> Dict[str, object]:
    analysis_root = Path(analysis_dir).expanduser().resolve()
    sidecar_dir = analysis_root / "sidecars"
    if not sidecar_dir.exists():
        raise FileNotFoundError(f"Missing sidecar directory: {sidecar_dir}")
    rmd_dir = Path(out_dir).expanduser().resolve() if out_dir else analysis_root / "rmd"
    records = []
    for sidecar_path in sorted(sidecar_dir.glob("*.sidecar.json")):
        sidecar_payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
        clip_id = str(sidecar_payload["clip_id"])
        rmd_path, metadata = write_rmd_for_clip_with_metadata(clip_id, sidecar_payload, rmd_dir)
        records.append(
            {
                "clip_id": clip_id,
                "sidecar_path": str(sidecar_path),
                "rmd_path": str(rmd_path),
                "rmd_name": rmd_path.name,
                "rmd_kind": metadata.get("rmd_kind"),
                "metadata": metadata,
            }
        )
    manifest = {
        "analysis_dir": str(analysis_root),
        "rmd_dir": str(rmd_dir),
        "clip_count": len(records),
        "clips": records,
    }
    (rmd_dir / "rmd_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
