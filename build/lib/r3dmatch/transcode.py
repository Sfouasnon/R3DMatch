from __future__ import annotations

import json
import shlex
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

from .matching import discover_clips


def build_redline_command(
    clip_path: str,
    *,
    render_dir: str,
    sidecar_path: Optional[str],
    redline_executable: str,
    output_ext: str,
) -> List[str]:
    clip = Path(clip_path).expanduser().resolve()
    render_root = Path(render_dir).expanduser().resolve()
    render_root.mkdir(parents=True, exist_ok=True)
    output_path = render_root / f"{clip.stem}.{output_ext.lstrip('.')}"
    command = [redline_executable, "--input", str(clip), "--output", str(output_path)]
    if sidecar_path:
        command.extend(["--look-metadata", str(Path(sidecar_path).expanduser().resolve())])
    return command


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
    for clip in clips:
        sidecar_path = None
        if use_generated_sidecar:
            if resolved_analysis_dir is None:
                raise ValueError("--use-generated-sidecar requires --analysis-dir")
            candidate = resolved_analysis_dir / "sidecars" / f"{clip.stem}.sidecar.json"
            if not candidate.exists():
                raise FileNotFoundError(f"Missing generated sidecar: {candidate}")
            sidecar_path = str(candidate)

        command = build_redline_command(
            str(clip),
            render_dir=str(renders_dir),
            sidecar_path=sidecar_path,
            redline_executable=redline_executable,
            output_ext=output_ext,
        )
        record = {
            "clip_id": clip.stem,
            "source_path": str(clip.resolve()),
            "sidecar_path": sidecar_path,
            "command": command,
            "executed": False,
            "returncode": None,
        }
        if execute:
            completed = subprocess.run(command, capture_output=True, text=True, check=False)
            record["executed"] = True
            record["returncode"] = completed.returncode
            record["stdout"] = completed.stdout
            record["stderr"] = completed.stderr
        manifests.append(record)
        (commands_dir / f"{clip.stem}.command.sh").write_text(shlex.join(command) + "\n", encoding="utf-8")
        (commands_dir / f"{clip.stem}.command.json").write_text(json.dumps(record, indent=2), encoding="utf-8")

    plan = {
        "input_path": str(Path(input_path).expanduser().resolve()),
        "analysis_dir": str(resolved_analysis_dir) if resolved_analysis_dir else None,
        "use_generated_sidecar": use_generated_sidecar,
        "commands_dir": str(commands_dir),
        "renders_dir": str(renders_dir),
        "clips": manifests,
    }
    (out_root / "transcode_plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    return plan

