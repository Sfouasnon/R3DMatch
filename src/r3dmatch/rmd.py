from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional
from xml.etree import ElementTree as ET

from .identity import rmd_name_for_clip_id


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
    note.text = "Initial supported subset only: exposure and optional color gains."
    return root


def render_rmd_xml(sidecar_payload: Dict[str, object]) -> str:
    root = build_rmd_tree_from_sidecar(sidecar_payload)
    xml = ET.tostring(root, encoding="unicode")
    return "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n" + xml + "\n"


def write_rmd_for_clip(clip_id: str, sidecar_payload: Dict[str, object], out_dir: str | Path) -> Path:
    target_dir = Path(out_dir).expanduser().resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / rmd_filename_for_clip_id(clip_id)
    path.write_text(render_rmd_xml(sidecar_payload), encoding="utf-8")
    return path


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
        rmd_path = write_rmd_for_clip(clip_id, sidecar_payload, rmd_dir)
        records.append(
            {
                "clip_id": clip_id,
                "sidecar_path": str(sidecar_path),
                "rmd_path": str(rmd_path),
                "rmd_name": rmd_path.name,
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
