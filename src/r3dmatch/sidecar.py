from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict

from .color import is_identity_cdl_payload, rgb_gains_to_cdl
from .identity import rmd_name_for_clip_id
from .models import ClipResult


def sidecar_filename_for_clip_id(clip_id: str) -> str:
    return f"{clip_id}.sidecar.json"


def build_sidecar_payload(result: ClipResult) -> Dict[str, object]:
    rgb_gains = (
        [
            result.pending_color_gains["r"],
            result.pending_color_gains["g"],
            result.pending_color_gains["b"],
        ]
        if result.pending_color_gains
        else None
    )
    cdl_payload = rgb_gains_to_cdl(rgb_gains) if rgb_gains else None
    return {
        "schema": "r3dmatch_v2",
        "clip_id": result.clip_id,
        "group_key": result.group_key,
        "exposure": {
            "offset_stops": result.final_offset_stops,
        },
        "color": {
            "rgb_gains": rgb_gains,
            "cdl": cdl_payload,
            "cdl_enabled": bool(cdl_payload is not None and not is_identity_cdl_payload(cdl_payload)),
        },
        "confidence": {
            "exposure": result.confidence,
            "color": 1.0 if result.pending_color_gains else 0.0,
        },
        "source_path": result.source_path,
        "sidecar_filename": sidecar_filename_for_clip_id(result.clip_id),
        "rmd_name": rmd_name_for_clip_id(result.clip_id),
        "calibration_state": {
            "exposure_calibration_loaded": result.exposure_calibration_loaded,
            "exposure_baseline_applied_stops": result.exposure_baseline_applied_stops,
            "color_calibration_loaded": result.color_calibration_loaded,
            "rgb_neutral_gains": result.pending_color_gains,
            "exposure_calibration_provenance": result.exposure_calibration_provenance,
            "color_calibration_provenance": result.color_calibration_provenance,
            "color_gains_state": result.color_gains_state,
        },
        "rmd_mapping": {
            "exposure": {
                "final_offset_stops": result.final_offset_stops,
            },
            "color": {
                "rgb_neutral_gains": rgb_gains,
                "cdl": cdl_payload,
                "cdl_enabled": bool(cdl_payload is not None and not is_identity_cdl_payload(cdl_payload)),
            },
        },
        "monitoring": asdict(result.monitoring),
    }


def write_sidecar_file(sidecar_dir: Path, result: ClipResult) -> Path:
    sidecar_dir.mkdir(parents=True, exist_ok=True)
    path = sidecar_dir / sidecar_filename_for_clip_id(result.clip_id)
    path.write_text(json.dumps(build_sidecar_payload(result), indent=2), encoding="utf-8")
    return path
