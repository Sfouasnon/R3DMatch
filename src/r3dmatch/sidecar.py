from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict

from .identity import rmd_name_for_clip_id
from .models import ClipResult


def sidecar_filename_for_clip_id(clip_id: str) -> str:
    return f"{clip_id}.sidecar.json"


def build_sidecar_payload(result: ClipResult) -> Dict[str, object]:
    return {
        "sidecar_format": "r3dmatch_intermediate_v1",
        "clip_id": result.clip_id,
        "group_key": result.group_key,
        "source_path": result.source_path,
        "sidecar_filename": sidecar_filename_for_clip_id(result.clip_id),
        "rmd_name": rmd_name_for_clip_id(result.clip_id),
        "identity": {
            "clip_id_exact": result.clip_id,
            "group_key": result.group_key,
            "sidecar_name_uses_clip_id_only": True,
        },
        "monitoring": asdict(result.monitoring),
        "analysis_state": {
            "mode": result.monitoring.mode,
            "raw_offset_stops": result.raw_offset_stops,
            "camera_baseline_stops": result.camera_baseline_stops,
            "clip_trim_stops": result.clip_trim_stops,
            "final_offset_stops": result.final_offset_stops,
            "confidence": result.confidence,
        },
        "calibration_state": {
            "exposure_calibration_loaded": result.exposure_calibration_loaded,
            "exposure_baseline_applied_stops": result.exposure_baseline_applied_stops,
            "exposure_calibration_provenance": result.exposure_calibration_provenance,
            "color_calibration_loaded": result.color_calibration_loaded,
            "color_gains_state": result.color_gains_state,
            "rgb_neutral_gains": result.pending_color_gains,
            "color_calibration_provenance": result.color_calibration_provenance,
        },
        "rmd_mapping": {
            "status": "intermediate_sidecar_only",
            "exposure": {
                "raw_offset_stops": result.raw_offset_stops,
                "camera_baseline_stops": result.camera_baseline_stops,
                "clip_trim_stops": result.clip_trim_stops,
                "final_offset_stops": result.final_offset_stops,
            },
            "color": {
                "rgb_neutral_gains": result.pending_color_gains,
                "state": result.color_gains_state,
            },
        },
    }


def write_sidecar_file(sidecar_dir: Path, result: ClipResult) -> Path:
    sidecar_dir.mkdir(parents=True, exist_ok=True)
    path = sidecar_dir / sidecar_filename_for_clip_id(result.clip_id)
    path.write_text(json.dumps(build_sidecar_payload(result), indent=2), encoding="utf-8")
    return path
