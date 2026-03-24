from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from .calibration import discover_clips, load_color_calibration, load_exposure_calibration
from .identity import clip_id_from_path, group_key_from_clip_id
from .rmd import rmd_filename_for_clip_id, write_rmds_from_analysis
from .sidecar import sidecar_filename_for_clip_id
from .transcode import build_redline_command_variants, load_sidecar


def validate_pipeline(
    input_path: str,
    *,
    analysis_dir: str,
    exposure_calibration_path: Optional[str],
    color_calibration_path: Optional[str],
    out_dir: str,
    redline_executable: str,
    output_ext: str,
) -> Dict[str, object]:
    clips = discover_clips(input_path)
    if not clips:
        raise ValueError(f"No .R3D clips found under {input_path}")
    clip_ids = [clip_id_from_path(str(path)) for path in clips]
    group_keys: list[str] = []
    if len(set(clip_ids)) != len(clip_ids):
        raise ValueError("clip_id collision detected in discovered clips")

    analysis_root = Path(analysis_dir).expanduser().resolve()
    sidecar_dir = analysis_root / "sidecars"
    analysis_json_dir = analysis_root / "analysis"
    if not sidecar_dir.exists() or not analysis_json_dir.exists():
        raise FileNotFoundError("analysis_dir must contain analysis/ and sidecars/ directories")
    rmd_manifest = write_rmds_from_analysis(str(analysis_root))
    rmd_dir = Path(rmd_manifest["rmd_dir"])

    exposure = load_exposure_calibration(exposure_calibration_path) if exposure_calibration_path else None
    color = load_color_calibration(color_calibration_path) if color_calibration_path else None
    clip_records: list[dict[str, object]] = []
    for clip in clips:
        clip_id = clip_id_from_path(str(clip))
        sidecar_path = sidecar_dir / sidecar_filename_for_clip_id(clip_id)
        analysis_path = analysis_json_dir / f"{clip_id}.analysis.json"
        if not sidecar_path.exists():
            raise FileNotFoundError(f"Missing sidecar for {clip_id}: {sidecar_path}")
        if not analysis_path.exists():
            raise FileNotFoundError(f"Missing analysis JSON for {clip_id}: {analysis_path}")
        analysis_payload = json.loads(analysis_path.read_text(encoding="utf-8"))
        analysis_group_key = str(analysis_payload.get("group_key", group_key_from_clip_id(clip_id)))
        group_keys.append(analysis_group_key)
        rmd_path = rmd_dir / rmd_filename_for_clip_id(clip_id)
        if not rmd_path.exists():
            raise FileNotFoundError(f"Missing RMD for {clip_id}: {rmd_path}")
        sidecar_payload = load_sidecar(str(sidecar_path))
        if sidecar_payload.get("clip_id") != clip_id:
            raise ValueError(f"Sidecar clip_id mismatch for {clip_id}")
        if Path(sidecar_path).name != sidecar_filename_for_clip_id(clip_id):
            raise ValueError(f"Sidecar filename mismatch for {clip_id}")
        if Path(rmd_path).name != rmd_filename_for_clip_id(clip_id):
            raise ValueError(f"RMD filename mismatch for {clip_id}")
        variants = build_redline_command_variants(
            str(clip),
            render_dir=str(Path(out_dir).expanduser().resolve() / "renders"),
            sidecar_path=None,
            rmd_path=str(rmd_path),
            redline_executable=redline_executable,
            output_ext=output_ext,
            sidecar_payload=sidecar_payload,
        )
        variant_records = []
        for variant in variants:
            uses_rmd = bool(variant.get("uses_rmd"))
            command = variant["command"]
            command_uses_exact_rmd = str(rmd_path) in command if uses_rmd else False
            if uses_rmd and not command_uses_exact_rmd:
                raise ValueError(f"REDLine command does not reference expected RMD for {clip_id}")
            variant_records.append(
                {
                    "variant": variant["variant"],
                    "uses_rmd": uses_rmd,
                    "command_uses_exact_rmd": command_uses_exact_rmd,
                }
            )
        clip_records.append(
            {
                "clip_id": clip_id,
                "group_key": analysis_group_key,
                "analysis_path": str(analysis_path),
                "sidecar_path": str(sidecar_path),
                "rmd_path": str(rmd_path),
                "sidecar_name_matches_clip_id": True,
                "rmd_name_matches_clip_id": True,
                "redline_variant_count": len(variants),
                "redline_variants": [variant["variant"] for variant in variants],
                "redline_variant_records": variant_records,
            }
        )

    payload = {
        "input_path": str(Path(input_path).expanduser().resolve()),
        "analysis_dir": str(analysis_root),
        "exposure_calibration_path": str(Path(exposure_calibration_path).expanduser().resolve()) if exposure_calibration_path else None,
        "color_calibration_path": str(Path(color_calibration_path).expanduser().resolve()) if color_calibration_path else None,
        "clip_count": len(clips),
        "clip_ids": clip_ids,
        "group_keys": group_keys,
        "identity_collisions": {
            "clip_id_collision": len(set(clip_ids)) != len(clip_ids),
        },
        "group_key_summary": {
            "unique_group_count": len(set(group_keys)),
            "shared_group_keys_present": len(set(group_keys)) != len(group_keys),
        },
        "calibrations": {
            "exposure_loaded": exposure is not None,
            "color_loaded": color is not None,
        },
        "rmd": {
            "rmd_dir": str(rmd_dir),
            "clip_count": rmd_manifest["clip_count"],
        },
        "clips": clip_records,
    }
    out_root = Path(out_dir).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    output_path = out_root / "pipeline_validation.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
