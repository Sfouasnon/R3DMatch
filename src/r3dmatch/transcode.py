from __future__ import annotations

import json
import shlex
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

from .identity import clip_id_from_path, rmd_name_for_clip_id
from .matching import discover_clips
from .sidecar import sidecar_filename_for_clip_id


def load_sidecar(path: str) -> Dict[str, object]:
    return json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))


def build_redline_mapping(sidecar_payload: Dict[str, object]) -> Dict[str, object]:
    exposure = dict(sidecar_payload.get("exposure", {}))
    color = dict(sidecar_payload.get("color", {}))
    rgb_gains = color.get("rgb_gains")
    return {
        "exposure_adjustment": exposure.get("offset_stops"),
        "cdl_slope": rgb_gains,
    }


def build_redline_command(
    clip_path: str,
    *,
    render_dir: str,
    sidecar_path: Optional[str],
    redline_executable: str,
    output_ext: str,
    variant: str = "original",
) -> List[str]:
    clip = Path(clip_path).expanduser().resolve()
    clip_id = clip_id_from_path(str(clip))
    render_root = Path(render_dir).expanduser().resolve()
    render_root.mkdir(parents=True, exist_ok=True)
    output_path = render_root / f"{clip_id}.{variant}.{output_ext.lstrip('.')}"
    command = [redline_executable, "--input", str(clip), "--output", str(output_path)]
    if sidecar_path:
        command.extend(["--look-metadata", str(Path(sidecar_path).expanduser().resolve())])
    return command


def build_redline_command_variants(
    clip_path: str,
    *,
    render_dir: str,
    sidecar_path: Optional[str],
    redline_executable: str,
    output_ext: str,
    sidecar_payload: Optional[Dict[str, object]] = None,
) -> List[Dict[str, object]]:
    variants: list[dict[str, object]] = []
    variants.append(
        {
            "variant": "original",
            "uses_sidecar": False,
            "command": build_redline_command(
                clip_path,
                render_dir=render_dir,
                sidecar_path=None,
                redline_executable=redline_executable,
                output_ext=output_ext,
                variant="original",
            ),
        }
    )
    if sidecar_path and sidecar_payload:
        calibration_state = dict(sidecar_payload.get("calibration_state", {}))
        exposure_loaded = bool(calibration_state.get("exposure_calibration_loaded"))
        color_loaded = bool(calibration_state.get("color_calibration_loaded"))
        if exposure_loaded:
            variants.append(
                {
                    "variant": "exposure",
                    "uses_sidecar": True,
                    "command": build_redline_command(
                        clip_path,
                        render_dir=render_dir,
                        sidecar_path=sidecar_path,
                        redline_executable=redline_executable,
                        output_ext=output_ext,
                        variant="exposure",
                    ),
                }
            )
        if color_loaded:
            variants.append(
                {
                    "variant": "color",
                    "uses_sidecar": True,
                    "command": build_redline_command(
                        clip_path,
                        render_dir=render_dir,
                        sidecar_path=sidecar_path,
                        redline_executable=redline_executable,
                        output_ext=output_ext,
                        variant="color",
                    ),
                }
            )
        if exposure_loaded and color_loaded:
            variants.append(
                {
                    "variant": "both",
                    "uses_sidecar": True,
                    "command": build_redline_command(
                        clip_path,
                        render_dir=render_dir,
                        sidecar_path=sidecar_path,
                        redline_executable=redline_executable,
                        output_ext=output_ext,
                        variant="both",
                    ),
                }
            )
    return variants


def write_transcode_plan(
    input_path: str,
    *,
    out_dir: str,
    analysis_dir: Optional[str],
    use_generated_sidecar: bool,
    redline_executable: str,
    output_ext: str,
    execute: bool,
) -> Dict[str, object]:
    clips = discover_clips(input_path)
    out_root = Path(out_dir).expanduser().resolve()
    commands_dir = out_root / "commands"
    renders_dir = out_root / "renders"
    commands_dir.mkdir(parents=True, exist_ok=True)
    renders_dir.mkdir(parents=True, exist_ok=True)

    resolved_analysis_dir = Path(analysis_dir).expanduser().resolve() if analysis_dir else None
    manifests = []
    shell_lines = ["#!/bin/sh", "set -eu", ""]
    for clip in clips:
        clip_id = clip_id_from_path(str(clip))
        sidecar_path = None
        sidecar_payload = None
        if use_generated_sidecar:
            if resolved_analysis_dir is None:
                raise ValueError("--use-generated-sidecar requires --analysis-dir")
            candidate = resolved_analysis_dir / "sidecars" / sidecar_filename_for_clip_id(clip_id)
            if not candidate.exists():
                raise FileNotFoundError(f"Missing generated sidecar: {candidate}")
            sidecar_path = str(candidate)
            sidecar_payload = load_sidecar(str(candidate))

        variants = build_redline_command_variants(
            str(clip),
            render_dir=str(renders_dir),
            sidecar_path=sidecar_path,
            redline_executable=redline_executable,
            output_ext=output_ext,
            sidecar_payload=sidecar_payload,
        )
        record = {
            "clip_id": clip_id,
            "rmd_name": rmd_name_for_clip_id(clip_id),
            "source_path": str(clip.resolve()),
            "sidecar_path": sidecar_path,
            "sidecar_name_matches_clip_id": (Path(sidecar_path).name == sidecar_filename_for_clip_id(clip_id)) if sidecar_path else True,
            "variants": [],
        }
        for variant in variants:
            redline_mapping = build_redline_mapping(sidecar_payload) if sidecar_payload and variant["uses_sidecar"] else {
                "exposure_adjustment": None,
                "cdl_slope": None,
            }
            variant_record = {
                "variant": variant["variant"],
                "uses_sidecar": variant["uses_sidecar"],
                "command": variant["command"],
                "redline_mapping": redline_mapping,
                "executed": False,
                "returncode": None,
            }
            if execute:
                completed = subprocess.run(variant["command"], capture_output=True, text=True, check=False)
                variant_record["executed"] = True
                variant_record["returncode"] = completed.returncode
                variant_record["stdout"] = completed.stdout
                variant_record["stderr"] = completed.stderr
            record["variants"].append(variant_record)
            shell_lines.append(shlex.join(variant["command"]))
            (commands_dir / f"{clip_id}.{variant['variant']}.command.sh").write_text(shlex.join(variant["command"]) + "\n", encoding="utf-8")
            (commands_dir / f"{clip_id}.{variant['variant']}.command.json").write_text(json.dumps(variant_record, indent=2), encoding="utf-8")
        manifests.append(record)

    shell_path = commands_dir / "run_redline_commands.sh"
    shell_path.write_text("\n".join(shell_lines) + "\n", encoding="utf-8")
    plan = {
        "input_path": str(Path(input_path).expanduser().resolve()),
        "analysis_dir": str(resolved_analysis_dir) if resolved_analysis_dir else None,
        "use_generated_sidecar": use_generated_sidecar,
        "commands_dir": str(commands_dir),
        "renders_dir": str(renders_dir),
        "shell_script": str(shell_path),
        "clips": manifests,
    }
    (out_root / "transcode_plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    return plan
