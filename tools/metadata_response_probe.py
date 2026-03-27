from __future__ import annotations

import argparse
import json
import math
import shlex
import subprocess
from pathlib import Path
from typing import Dict, List

import numpy as np

from r3dmatch.report import (
    _detect_redline_capabilities,
    _load_analysis_records,
    _load_array_calibration_payload,
    _measure_rendered_preview_roi,
    _measurement_preview_settings_for_domain,
    _resolve_rendered_output_path,
    COLOR_SPACE_CODES,
    GAMMA_CODES,
    ROLLOFF_CODES,
    SHADOW_ROLLOFF_VALUES,
    TONEMAP_CODES,
)


REPRESENTATIVE_CLIPS = (
    "G007_A063_032563_001",
    "H007_D063_0325EZ_001",
    "I007_C063_0325L3_001",
)


def _norm3(values: List[float]) -> float:
    return float(math.sqrt(sum(float(v) * float(v) for v in values)))


def _delta3(a: List[float], b: List[float]) -> List[float]:
    return [float(x) - float(y) for x, y in zip(a, b)]


def _norm_delta(a: List[float], b: List[float]) -> float:
    return _norm3(_delta3(a, b))


def _find_nested(meta: Dict[str, object], key: str, default: float) -> float:
    current: object = meta
    while isinstance(current, dict):
        if key in current and current[key] is not None:
            return float(current[key])
        current = current.get("extra_metadata")
    return float(default)


def _build_command(
    *,
    source_path: str,
    output_path: str,
    frame_index: int,
    use_meta: bool,
    kelvin: int | None,
    tint: float | None,
) -> List[str]:
    settings = _measurement_preview_settings_for_domain("scene")
    command = [
        "/usr/local/bin/REDLine",
        "--i",
        str(Path(source_path).expanduser().resolve()),
        "--o",
        str(Path(output_path).expanduser().resolve()),
        "--format",
        "3",
        "--start",
        str(frame_index),
        "--frameCount",
        "1",
        "--colorSciVersion",
        "3",
        "--silent",
        "--colorSpace",
        str(COLOR_SPACE_CODES[str(settings["output_space"])]),
        "--gammaCurve",
        str(GAMMA_CODES[str(settings["output_gamma"])]),
        "--outputToneMap",
        str(TONEMAP_CODES[str(settings["output_tonemap"])]),
        "--rollOff",
        str(ROLLOFF_CODES[str(settings["highlight_rolloff"])]),
        "--shadow",
        f"{float(SHADOW_ROLLOFF_VALUES[str(settings['shadow_rolloff'])]):.3f}",
    ]
    if use_meta:
        command.append("--useMeta")
    if kelvin is not None:
        command.extend(["--kelvin", str(int(kelvin))])
    if tint is not None:
        command.extend(["--tint", f"{float(tint):.1f}"])
    return command


def run_probe(analysis_root: Path, out_dir: Path, clip_ids: List[str]) -> Dict[str, object]:
    analysis_records = {str(item["clip_id"]): item for item in _load_analysis_records(str(analysis_root))}
    array_payload = _load_array_calibration_payload(str(analysis_root))
    if not array_payload:
        raise RuntimeError(f"array_calibration.json not found under {analysis_root}")
    camera_map = {str(camera["clip_id"]): camera for camera in array_payload.get("cameras", [])}
    target_chroma = [
        float(value)
        for value in array_payload.get("target", {}).get("color", {}).get("target_rgb_chromaticity", [1 / 3, 1 / 3, 1 / 3])
    ]
    out_dir.mkdir(parents=True, exist_ok=True)

    experiment: Dict[str, object] = {
        "analysis_root": str(analysis_root),
        "out_dir": str(out_dir),
        "target_rgb_chromaticity": target_chroma,
        "clips": [],
    }

    for clip_id in clip_ids:
        record = analysis_records[clip_id]
        camera = camera_map[clip_id]
        diagnostics = dict(record.get("diagnostics", {}) or {})
        clip_meta = dict(record.get("clip_metadata", {}) or {})
        source_path = str(record["source_path"])
        roi = diagnostics.get("calibration_roi")
        as_shot_kelvin = int(round(_find_nested(clip_meta, "white_balance_kelvin", 5600.0)))
        as_shot_tint = float(_find_nested(clip_meta, "white_balance_tint", 0.0))
        solved_kelvin = int(camera["solution"]["kelvin"])
        solved_tint = float(camera["solution"]["tint"])
        scene_raw_chroma = [float(v) for v in diagnostics.get("measured_rgb_chromaticity", [1 / 3, 1 / 3, 1 / 3])]
        scene_raw_log2 = float(diagnostics.get("measured_log2_luminance_raw", diagnostics.get("measured_log2_luminance", 0.0)))
        predicted_post_residual = float(camera["quality"].get("post_color_residual", 0.0) or 0.0)
        predicted_pre_residual = float(camera["quality"].get("color_residual", 0.0) or 0.0)

        variants = [
            {"name": "scene_nometa_default", "use_meta": False, "kelvin": None, "tint": None},
            {"name": "scene_nometa_explicit_as_shot", "use_meta": False, "kelvin": as_shot_kelvin, "tint": as_shot_tint},
            {"name": "scene_usemeta_default", "use_meta": True, "kelvin": None, "tint": None},
            {"name": "scene_usemeta_kelvin_minus500", "use_meta": True, "kelvin": as_shot_kelvin - 500, "tint": None},
            {"name": "scene_usemeta_kelvin_plus500", "use_meta": True, "kelvin": as_shot_kelvin + 500, "tint": None},
            {"name": "scene_usemeta_tint_minus10", "use_meta": True, "kelvin": None, "tint": as_shot_tint - 10.0},
            {"name": "scene_usemeta_tint_plus10", "use_meta": True, "kelvin": None, "tint": as_shot_tint + 10.0},
            {"name": "scene_usemeta_solver_commit", "use_meta": True, "kelvin": solved_kelvin, "tint": solved_tint},
        ]

        rendered: Dict[str, Dict[str, object]] = {}
        clip_out_dir = out_dir / clip_id
        clip_out_dir.mkdir(parents=True, exist_ok=True)
        for variant in variants:
            nominal_output = clip_out_dir / f"{variant['name']}.jpg"
            command = _build_command(
                source_path=source_path,
                output_path=str(nominal_output),
                frame_index=0,
                use_meta=bool(variant["use_meta"]),
                kelvin=variant["kelvin"],
                tint=variant["tint"],
            )
            completed = subprocess.run(command, capture_output=True, text=True, check=False)
            resolved_output = _resolve_rendered_output_path(nominal_output)
            if completed.returncode != 0 or not resolved_output.exists():
                raise RuntimeError(
                    f"REDLine metadata probe failed for {clip_id} {variant['name']} "
                    f"returncode={completed.returncode} stderr={completed.stderr}"
                )
            measurement = _measure_rendered_preview_roi(str(resolved_output), roi)
            chroma = [float(v) for v in measurement["measured_rgb_chromaticity_monitoring"]]
            rendered[variant["name"]] = {
                "command": command,
                "command_text": " ".join(shlex.quote(part) for part in command),
                "output_path": str(resolved_output),
                "measured_log2_luminance": float(measurement["measured_log2_luminance_monitoring"]),
                "measured_rgb_chromaticity": chroma,
                "delta_from_scene_raw_chroma": _delta3(chroma, scene_raw_chroma),
                "norm_delta_from_scene_raw_chroma": _norm_delta(chroma, scene_raw_chroma),
                "post_residual_to_target": _norm_delta(chroma, target_chroma),
            }

        usemeta = rendered["scene_usemeta_default"]
        nometa = rendered["scene_nometa_default"]
        nometa_explicit = rendered["scene_nometa_explicit_as_shot"]
        kelvin_minus = rendered["scene_usemeta_kelvin_minus500"]
        kelvin_plus = rendered["scene_usemeta_kelvin_plus500"]
        tint_minus = rendered["scene_usemeta_tint_minus10"]
        tint_plus = rendered["scene_usemeta_tint_plus10"]
        solver = rendered["scene_usemeta_solver_commit"]

        clip_result = {
            "clip_id": clip_id,
            "source_path": source_path,
            "as_shot_kelvin": as_shot_kelvin,
            "as_shot_tint": as_shot_tint,
            "scene_raw_measurement": {
                "measured_log2_luminance": scene_raw_log2,
                "measured_rgb_chromaticity": scene_raw_chroma,
                "pre_residual_to_target": _norm_delta(scene_raw_chroma, target_chroma),
            },
            "current_solver": {
                "recommended_kelvin": solved_kelvin,
                "recommended_tint": solved_tint,
                "predicted_pre_residual": predicted_pre_residual,
                "predicted_post_residual": predicted_post_residual,
                "observed_post_residual_from_metadata_decode": solver["post_residual_to_target"],
                "prediction_error": solver["post_residual_to_target"] - predicted_post_residual,
            },
            "metadata_effects": {
                "nometa_vs_usemeta_chroma_delta": _norm_delta(
                    nometa["measured_rgb_chromaticity"], usemeta["measured_rgb_chromaticity"]
                ),
                "nometa_explicit_vs_usemeta_chroma_delta": _norm_delta(
                    nometa_explicit["measured_rgb_chromaticity"], usemeta["measured_rgb_chromaticity"]
                ),
                "nometa_vs_usemeta_log2_delta": abs(
                    float(nometa["measured_log2_luminance"]) - float(usemeta["measured_log2_luminance"])
                ),
                "nometa_explicit_vs_usemeta_log2_delta": abs(
                    float(nometa_explicit["measured_log2_luminance"]) - float(usemeta["measured_log2_luminance"])
                ),
                "kelvin_response_vector_per_1000K": [
                    float(a - b) / 1000.0
                    for a, b in zip(
                        kelvin_plus["measured_rgb_chromaticity"],
                        kelvin_minus["measured_rgb_chromaticity"],
                    )
                ],
                "tint_response_vector_per_20": [
                    float(a - b) / 20.0
                    for a, b in zip(
                        tint_plus["measured_rgb_chromaticity"],
                        tint_minus["measured_rgb_chromaticity"],
                    )
                ],
                "kelvin_response_norm_per_1000K": _norm_delta(
                    kelvin_plus["measured_rgb_chromaticity"],
                    kelvin_minus["measured_rgb_chromaticity"],
                )
                / 1000.0,
                "tint_response_norm_per_20": _norm_delta(
                    tint_plus["measured_rgb_chromaticity"],
                    tint_minus["measured_rgb_chromaticity"],
                )
                / 20.0,
            },
            "rendered_variants": rendered,
        }
        experiment["clips"].append(clip_result)

    return experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe REDLine kelvin/tint metadata response on real clips.")
    parser.add_argument("--analysis-root", required=True, help="Completed analysis/review root containing analysis JSONs.")
    parser.add_argument("--out", required=True, help="Directory for rendered variants and summary JSON.")
    parser.add_argument("--clip-id", action="append", dest="clip_ids", help="Clip ID to probe. May be repeated.")
    args = parser.parse_args()

    analysis_root = Path(args.analysis_root).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()
    clip_ids = list(args.clip_ids or REPRESENTATIVE_CLIPS)
    payload = run_probe(analysis_root, out_dir, clip_ids)
    output_path = out_dir / "metadata_response_probe.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
