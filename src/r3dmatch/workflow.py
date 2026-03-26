from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from PIL import Image, UnidentifiedImageError

from .color import is_identity_cdl_payload
from .execution import raise_if_cancelled
from .identity import group_key_from_clip_id
from .matching import analyze_path
from .report import build_contact_sheet_report, build_review_package, clear_preview_cache, render_contact_sheet_pdf
from .rmd import write_rmd_for_clip_with_metadata


class ReviewValidationError(RuntimeError):
    pass


def normalize_matching_domain(value: str) -> str:
    normalized = str(value).strip().lower()
    aliases = {
        "scene": "scene",
        "scene-referred": "scene",
        "scenereferred": "scene",
        "perceptual": "perceptual",
        "monitoring": "perceptual",
        "view": "perceptual",
    }
    if normalized not in aliases:
        raise ValueError("matching domain must be scene or perceptual")
    return aliases[normalized]


def matching_domain_label(value: str) -> str:
    normalized = normalize_matching_domain(value)
    if normalized == "scene":
        return "Scene-Referred (REDWideGamutRGB / Log3G10)"
    return "Perceptual (IPP2 / BT.709 / BT.1886)"


def _sanitize_run_label(label: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", label.strip()).strip("._-")
    return cleaned or "calibration_run"


def resolve_run_label(
    *,
    run_label: Optional[str],
    selected_clip_ids: Optional[List[str]],
    selected_clip_groups: Optional[List[str]],
) -> Optional[str]:
    if run_label and str(run_label).strip():
        return _sanitize_run_label(str(run_label))
    clip_ids = [str(item).strip() for item in (selected_clip_ids or []) if str(item).strip()]
    clip_groups = [str(item).strip() for item in (selected_clip_groups or []) if str(item).strip()]
    if clip_groups and len(clip_groups) == 1 and not clip_ids:
        return _sanitize_run_label(f"subset_{clip_groups[0]}")
    if len(clip_ids) == 1 and not clip_groups:
        return _sanitize_run_label(clip_ids[0])
    if clip_ids or clip_groups:
        return _sanitize_run_label(f"subset_{len(clip_ids) or len(clip_groups)}")
    return None


def resolve_review_output_dir(
    out_dir: str,
    *,
    run_label: Optional[str],
    selected_clip_ids: Optional[List[str]],
    selected_clip_groups: Optional[List[str]],
) -> str:
    root = Path(out_dir).expanduser().resolve()
    resolved_label = resolve_run_label(
        run_label=run_label,
        selected_clip_ids=selected_clip_ids,
        selected_clip_groups=selected_clip_groups,
    )
    return str((root / resolved_label).resolve()) if resolved_label else str(root)


def _load_json_file(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_existing_file(path: Path, *, kind: str) -> Optional[str]:
    if not path.exists():
        return f"Missing required artifact: {kind} ({path})"
    try:
        size_bytes = path.stat().st_size
    except OSError as exc:
        return f"Failed to stat required artifact {kind}: {exc}"
    if size_bytes <= 0:
        return f"Required artifact is empty: {kind} ({path})"
    return None


def _review_preview_paths_from_payload(payload: Dict[str, object]) -> List[str]:
    paths: List[str] = []
    for item in payload.get("shared_originals", []):
        original = item.get("original_frame")
        if isinstance(original, str) and original.strip():
            paths.append(original)
    for strategy in payload.get("strategies", []):
        for clip in strategy.get("clips", []):
            for key in ("original_frame", "exposure_corrected", "color_corrected", "both_corrected"):
                candidate = clip.get(key)
                if isinstance(candidate, str) and candidate.strip():
                    paths.append(candidate)
            for candidate in (clip.get("preview_variants") or {}).values():
                if isinstance(candidate, str) and candidate.strip():
                    paths.append(candidate)
    return sorted(set(paths))


def validate_review_run_contract(analysis_dir: str) -> Dict[str, object]:
    analysis_root = Path(analysis_dir).expanduser().resolve()
    report_root = analysis_root / "report"
    result: Dict[str, object] = {
        "analysis_dir": str(analysis_root),
        "report_dir": str(report_root),
        "status": "success",
        "errors": [],
        "warnings": [],
        "required_artifacts": {},
        "optional_artifacts": {},
        "preview_reference_count": 0,
        "preview_existing_count": 0,
        "missing_preview_paths": [],
    }

    required_paths = {
        "summary_json": analysis_root / "summary.json",
        "contact_sheet_json": report_root / "contact_sheet.json",
        "review_manifest_json": report_root / "review_manifest.json",
        "review_package_json": report_root / "review_package.json",
        "preview_commands_json": analysis_root / "previews" / "preview_commands.json",
    }
    optional_paths = {
        "contact_sheet_html": report_root / "contact_sheet.html",
        "preview_contact_sheet_pdf": report_root / "preview_contact_sheet.pdf",
    }

    parsed_payloads: Dict[str, Dict[str, object]] = {}
    for key, path in required_paths.items():
        error = _validate_existing_file(path, kind=key)
        exists = path.exists()
        entry: Dict[str, object] = {"path": str(path), "exists": exists, "parsed": False, "size_bytes": path.stat().st_size if exists else 0}
        if error:
            result["errors"].append(error)
        else:
            try:
                if path.suffix == ".json":
                    parsed_payloads[key] = _load_json_file(path)
                    entry["parsed"] = True
                else:
                    entry["parsed"] = True
            except Exception as exc:
                result["errors"].append(f"Failed to parse required artifact {key}: {exc}")
        result["required_artifacts"][key] = entry

    for key, path in optional_paths.items():
        exists = path.exists()
        entry = {"path": str(path), "exists": exists, "size_bytes": path.stat().st_size if exists else 0}
        result["optional_artifacts"][key] = entry
        if not exists:
            result["warnings"].append(f"Optional artifact missing: {key} ({path})")
        elif entry["size_bytes"] <= 0:
            result["errors"].append(f"Optional report artifact is empty: {key} ({path})")

    contact_sheet_payload = parsed_payloads.get("contact_sheet_json")
    if contact_sheet_payload:
        preview_paths = _review_preview_paths_from_payload(contact_sheet_payload)
        missing_preview_paths = [path for path in preview_paths if not Path(path).exists()]
        unreadable_preview_paths: List[str] = []
        empty_preview_paths: List[str] = []
        for path_str in preview_paths:
            preview_path = Path(path_str)
            if not preview_path.exists():
                continue
            try:
                if preview_path.stat().st_size <= 0:
                    empty_preview_paths.append(path_str)
                    continue
            except OSError:
                empty_preview_paths.append(path_str)
                continue
            try:
                with Image.open(preview_path) as image:
                    image.verify()
            except (OSError, UnidentifiedImageError):
                unreadable_preview_paths.append(path_str)
        result["preview_reference_count"] = len(preview_paths)
        result["preview_existing_count"] = len(preview_paths) - len(missing_preview_paths) - len(empty_preview_paths) - len(unreadable_preview_paths)
        result["missing_preview_paths"] = missing_preview_paths
        result["empty_preview_paths"] = empty_preview_paths
        result["unreadable_preview_paths"] = unreadable_preview_paths
        if not preview_paths:
            result["errors"].append("Review report did not reference any preview images.")
        if missing_preview_paths:
            result["errors"].append(f"Missing {len(missing_preview_paths)} preview image(s) referenced by contact_sheet.json.")
        if empty_preview_paths:
            result["errors"].append(f"Found {len(empty_preview_paths)} empty preview image(s) referenced by contact_sheet.json.")
        if unreadable_preview_paths:
            result["errors"].append(f"Found {len(unreadable_preview_paths)} unreadable preview image(s) referenced by contact_sheet.json.")
        result["clip_count"] = int(contact_sheet_payload.get("clip_count", 0) or 0)
        result["color_preview_status"] = contact_sheet_payload.get("color_preview_status")
        result["color_preview_note"] = contact_sheet_payload.get("color_preview_note")

    if not result["optional_artifacts"]["contact_sheet_html"]["exists"] and not result["optional_artifacts"]["preview_contact_sheet_pdf"]["exists"]:
        result["errors"].append("No human-readable report artifact was produced: both HTML and PDF are missing.")

    if result["errors"]:
        result["status"] = "failed"

    validation_path = report_root / "review_validation.json"
    validation_path.parent.mkdir(parents=True, exist_ok=True)
    validation_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    result["validation_path"] = str(validation_path)
    return result


def _format_review_validation_failure(validation: Dict[str, object]) -> str:
    errors = [str(item) for item in validation.get("errors", []) if str(item).strip()]
    if not errors:
        return "Review run failed validation for an unspecified reason."
    return "Review run failed validation: " + "; ".join(errors)


def _load_clip_subset_definition(path: str) -> Dict[str, object]:
    subset_path = Path(path).expanduser().resolve()
    payload = json.loads(subset_path.read_text(encoding="utf-8"))
    return {
        "path": str(subset_path),
        "run_label": payload.get("run_label"),
        "clip_ids": [str(item) for item in payload.get("clip_ids", []) if str(item).strip()],
        "clip_groups": [str(item) for item in payload.get("clip_groups", []) if str(item).strip()],
    }


def review_calibration(
    input_path: str,
    *,
    out_dir: str,
    target_type: str,
    processing_mode: str,
    mode: str,
    backend: str,
    lut_override: Optional[str],
    calibration_path: Optional[str],
    exposure_calibration_path: Optional[str],
    color_calibration_path: Optional[str],
    calibration_mode: Optional[str],
    sample_count: int,
    sampling_strategy: str,
    calibration_roi: Optional[Dict[str, float]],
    target_strategies: List[str],
    reference_clip_id: Optional[str],
    hero_clip_id: Optional[str] = None,
    selected_clip_ids: Optional[List[str]] = None,
    selected_clip_groups: Optional[List[str]] = None,
    clip_subset_file: Optional[str] = None,
    run_label: Optional[str] = None,
    matching_domain: str = "scene",
    preview_mode: str = "calibration",
    preview_output_space: Optional[str] = None,
    preview_output_gamma: Optional[str] = None,
    preview_highlight_rolloff: Optional[str] = None,
    preview_shadow_rolloff: Optional[str] = None,
    preview_lut: Optional[str] = None,
) -> Dict[str, object]:
    subset_definition = _load_clip_subset_definition(clip_subset_file) if clip_subset_file else None
    merged_clip_ids = [str(item) for item in (selected_clip_ids or []) if str(item).strip()]
    merged_clip_groups = [str(item) for item in (selected_clip_groups or []) if str(item).strip()]
    if subset_definition:
        merged_clip_ids = list(dict.fromkeys([*subset_definition["clip_ids"], *merged_clip_ids]))
        merged_clip_groups = list(dict.fromkeys([*subset_definition["clip_groups"], *merged_clip_groups]))
        run_label = run_label or subset_definition.get("run_label")
    resolved_matching_domain = normalize_matching_domain(matching_domain)
    resolved_run_label = resolve_run_label(
        run_label=run_label,
        selected_clip_ids=merged_clip_ids,
        selected_clip_groups=merged_clip_groups,
    )
    resolved_out_dir = resolve_review_output_dir(
        out_dir,
        run_label=resolved_run_label,
        selected_clip_ids=merged_clip_ids,
        selected_clip_groups=merged_clip_groups,
    )
    raise_if_cancelled("Run cancelled before analysis.")
    analyze_summary = analyze_path(
        input_path,
        out_dir=resolved_out_dir,
        mode=mode,
        backend=backend,
        lut_override=lut_override,
        calibration_path=calibration_path,
        exposure_calibration_path=exposure_calibration_path,
        color_calibration_path=color_calibration_path,
        calibration_mode=calibration_mode,
        sample_count=sample_count,
        sampling_strategy=sampling_strategy,
        calibration_roi=calibration_roi,
        selected_clip_ids=merged_clip_ids,
        selected_clip_groups=merged_clip_groups,
    )
    raise_if_cancelled("Run cancelled before review package generation.")
    package = build_review_package(
        input_path,
        out_dir=resolved_out_dir,
        exposure_calibration_path=exposure_calibration_path or calibration_path,
        color_calibration_path=color_calibration_path,
        target_type=target_type,
        processing_mode=processing_mode,
        run_label=resolved_run_label,
        matching_domain=resolved_matching_domain,
        selected_clip_ids=merged_clip_ids,
        selected_clip_groups=merged_clip_groups,
        preview_mode=preview_mode,
        preview_output_space=preview_output_space,
        preview_output_gamma=preview_output_gamma,
        preview_highlight_rolloff=preview_highlight_rolloff,
        preview_shadow_rolloff=preview_shadow_rolloff,
        preview_lut=preview_lut,
        calibration_roi=calibration_roi,
        target_strategies=target_strategies,
        reference_clip_id=reference_clip_id,
        hero_clip_id=hero_clip_id,
    )
    package["analyze_summary"] = analyze_summary
    package["analysis_dir"] = resolved_out_dir
    package["run_label"] = resolved_run_label
    package["selected_clip_ids"] = merged_clip_ids
    package["selected_clip_groups"] = merged_clip_groups
    package["matching_domain"] = resolved_matching_domain
    package["matching_domain_label"] = matching_domain_label(resolved_matching_domain)
    package["clip_subset_file"] = subset_definition["path"] if subset_definition else None
    validation = validate_review_run_contract(resolved_out_dir)
    package["review_validation"] = validation
    package_manifest_path = package.get("package_manifest")
    if package_manifest_path:
        Path(str(package_manifest_path)).write_text(json.dumps(package, indent=2), encoding="utf-8")
    if validation["status"] != "success":
        raise ReviewValidationError(_format_review_validation_failure(validation))
    return package


def _identity_cdl_payload() -> Dict[str, object]:
    return {
        "slope": [1.0, 1.0, 1.0],
        "offset": [0.0, 0.0, 0.0],
        "power": [1.0, 1.0, 1.0],
        "saturation": 1.0,
    }


def _normalize_cdl_payload(payload: Optional[Dict[str, object]]) -> Dict[str, object]:
    if not isinstance(payload, dict):
        return _identity_cdl_payload()
    return {
        "slope": [float(value) for value in payload.get("slope", [1.0, 1.0, 1.0])],
        "offset": [float(value) for value in payload.get("offset", [0.0, 0.0, 0.0])],
        "power": [float(value) for value in payload.get("power", [1.0, 1.0, 1.0])],
        "saturation": float(payload.get("saturation", 1.0)),
    }


def _approved_sidecar_payload_for_clip(clip: Dict[str, object]) -> Dict[str, object]:
    color_metrics = clip["metrics"]["color"]
    gains = color_metrics.get("rgb_gains_diagnostic") or color_metrics.get("rgb_gains")
    color_cdl = _normalize_cdl_payload(color_metrics.get("cdl"))
    cdl_enabled = not is_identity_cdl_payload(color_cdl)
    return {
        "clip_id": clip["clip_id"],
        "source_path": clip["source_path"],
        "schema": "r3dmatch_v2",
        "calibration_state": {
            "exposure_calibration_loaded": True,
            "exposure_baseline_applied_stops": clip["metrics"]["exposure"]["final_offset_stops"],
            "color_calibration_loaded": gains is not None,
            "rgb_neutral_gains": {"r": gains[0], "g": gains[1], "b": gains[2]} if gains else None,
            "color_gains_state": "approved",
        },
        "rmd_mapping": {
            "exposure": {"final_offset_stops": clip["metrics"]["exposure"]["final_offset_stops"]},
            "color": {
                "rgb_neutral_gains": gains,
                "cdl": color_cdl,
                "cdl_enabled": cdl_enabled,
            },
        },
    }


def _correction_key_for_clip(clip: Dict[str, object]) -> str:
    return group_key_from_clip_id(str(clip["clip_id"]))


def _correction_signature(clip: Dict[str, object]) -> tuple:
    color_metrics = clip["metrics"]["color"]
    cdl = _normalize_cdl_payload(color_metrics.get("cdl"))
    return (
        round(float(clip["metrics"]["exposure"]["final_offset_stops"]), 6),
        tuple(round(float(value), 6) for value in cdl["slope"]),
        tuple(round(float(value), 6) for value in cdl["offset"]),
        tuple(round(float(value), 6) for value in cdl["power"]),
        round(float(cdl["saturation"]), 6),
        bool(not is_identity_cdl_payload(cdl)),
    )


def _detect_source_root(source_paths: List[str]) -> Optional[str]:
    if not source_paths:
        return None
    common_path = Path(os.path.commonpath(source_paths))
    if common_path.suffix.lower() == ".r3d":
        common_path = common_path.parent
    return str(common_path)


def _relative_to_root(path: str, root: Optional[str]) -> Optional[str]:
    if root is None:
        return None
    try:
        return str(Path(path).resolve().relative_to(Path(root).resolve()))
    except ValueError:
        return os.path.relpath(path, root)


def _write_master_rmds_from_strategy(strategy_payload: Dict[str, object], *, out_dir: str) -> Dict[str, object]:
    raise_if_cancelled("Run cancelled before MasterRMD export.")
    target_dir = Path(out_dir).expanduser().resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    grouped: Dict[str, List[Dict[str, object]]] = {}
    for clip in strategy_payload["clips"]:
        grouped.setdefault(_correction_key_for_clip(clip), []).append(clip)

    exports: List[Dict[str, object]] = []
    clip_mappings: List[Dict[str, object]] = []
    source_paths = [str(clip["source_path"]) for clip in strategy_payload["clips"]]
    source_root = _detect_source_root(source_paths)

    for correction_key, clips in sorted(grouped.items()):
        raise_if_cancelled("Run cancelled while exporting MasterRMD files.")
        representative = clips[0]
        representative_signature = _correction_signature(representative)
        for candidate in clips[1:]:
            if _correction_signature(candidate) != representative_signature:
                raise ValueError(
                    f"Cannot export a single MasterRMD for correction key {correction_key}: "
                    f"approved clips in this group do not share identical corrections."
                )

        sidecar_like = _approved_sidecar_payload_for_clip(representative)
        path, metadata = write_rmd_for_clip_with_metadata(correction_key, sidecar_like, target_dir)
        exposure_offset = float(representative["metrics"]["exposure"]["final_offset_stops"])
        color_metrics = representative["metrics"]["color"]
        color_cdl = _normalize_cdl_payload(color_metrics.get("cdl"))
        cdl_enabled = bool(not is_identity_cdl_payload(color_cdl))
        export_record = {
            "correction_key": correction_key,
            "camera_identity": correction_key,
            "approved_strategy": strategy_payload["strategy_key"],
            "hero_clip_id": strategy_payload.get("hero_clip_id"),
            "reference_clip_id": strategy_payload.get("reference_clip_id"),
            "camera_group_key": representative["group_key"],
            "representative_clip_id": representative["clip_id"],
            "source_clip_ids": [clip["clip_id"] for clip in clips],
            "source_r3d_paths": [clip["source_path"] for clip in clips],
            "source_r3d_relative_paths": [_relative_to_root(str(clip["source_path"]), source_root) for clip in clips],
            "master_rmd_path": str(path),
            "master_rmd_name": path.name,
            "rmd_kind": metadata.get("rmd_kind"),
            "exposure_correction_stops": exposure_offset,
            "cdl_enabled": cdl_enabled,
            "cdl": color_cdl,
            "rgb_gains_diagnostic": color_metrics.get("rgb_gains_diagnostic") or color_metrics.get("rgb_gains"),
        }
        exports.append(export_record)

        for clip in sorted(clips, key=lambda item: str(item["clip_id"])):
            raise_if_cancelled("Run cancelled while building approval clip mappings.")
            clip_cdl = _normalize_cdl_payload(clip["metrics"]["color"].get("cdl"))
            clip_mappings.append(
                {
                    "clip_id": str(clip["clip_id"]),
                    "source_r3d_path": str(clip["source_path"]),
                    "source_r3d_relative_path": _relative_to_root(str(clip["source_path"]), source_root),
                    "camera_group_key": str(clip["group_key"]),
                    "correction_key": correction_key,
                    "approved_strategy": strategy_payload["strategy_key"],
                    "hero_clip_id": strategy_payload.get("hero_clip_id"),
                    "reference_clip_id": strategy_payload.get("reference_clip_id"),
                    "master_rmd_path": str(path),
                    "master_rmd_name": path.name,
                    "exposure_correction_stops": float(clip["metrics"]["exposure"]["final_offset_stops"]),
                    "authored_cdl_summary": clip_cdl,
                    "cdl_enabled": bool(not is_identity_cdl_payload(clip_cdl)),
                    "is_hero_camera": bool(clip.get("is_hero_camera")),
                }
            )

    return {
        "rmd_dir": str(target_dir),
        "folder_name": target_dir.name,
        "clip_count": len(strategy_payload["clips"]),
        "correction_key_count": len(exports),
        "correction_key_model": {
            "name": "group_key_from_clip_id",
            "description": "Uses the first two underscore-delimited clip ID tokens as the stable per-camera correction key.",
        },
        "source_root": source_root,
        "master_rmds": exports,
        "clip_mappings": clip_mappings,
    }


def _build_batch_manifest(
    *,
    analysis_root: Path,
    approval_root: Path,
    strategy_payload: Dict[str, object],
    master_rmd_manifest: Dict[str, object],
    run_label: Optional[str] = None,
    matching_domain: Optional[str] = None,
    selected_clip_ids: Optional[List[str]] = None,
    selected_clip_groups: Optional[List[str]] = None,
) -> Dict[str, object]:
    raise_if_cancelled("Run cancelled before batch manifest generation.")
    batch_root = approval_root / "batch"
    batch_root.mkdir(parents=True, exist_ok=True)
    source_root = master_rmd_manifest.get("source_root")
    entries: List[Dict[str, object]] = []
    for item in master_rmd_manifest["clip_mappings"]:
        raise_if_cancelled("Run cancelled while building batch manifest.")
        master_rmd_path = Path(item["master_rmd_path"]).resolve()
        entries.append(
            {
                "clip_id": item["clip_id"],
                "source_r3d_path": item["source_r3d_path"],
                "source_r3d_relative_path": item.get("source_r3d_relative_path"),
                "camera_group_key": item["camera_group_key"],
                "correction_key": item["correction_key"],
                "approved_strategy": item["approved_strategy"],
                "approved_run_label": run_label,
                "hero_clip_id": item.get("hero_clip_id"),
                "reference_clip_id": item.get("reference_clip_id"),
                "master_rmd_path": str(master_rmd_path),
                "master_rmd_name": master_rmd_path.name,
                "master_rmd_relative_path": str(Path("..") / "MasterRMD" / master_rmd_path.name),
                "exposure_correction_stops": item["exposure_correction_stops"],
                "authored_cdl_summary": item["authored_cdl_summary"],
                "cdl_enabled": item["cdl_enabled"],
                "is_hero_camera": item["is_hero_camera"],
            }
        )

    manifest = {
        "analysis_dir": str(analysis_root),
        "approval_dir": str(approval_root),
        "batch_dir": str(batch_root),
        "run_label": run_label,
        "matching_domain": matching_domain,
        "matching_domain_label": matching_domain_label(matching_domain or "scene"),
        "selected_clip_ids": [str(item) for item in (selected_clip_ids or []) if str(item).strip()],
        "selected_clip_groups": [str(item) for item in (selected_clip_groups or []) if str(item).strip()],
        "approved_strategy": strategy_payload["strategy_key"],
        "hero_clip_id": strategy_payload.get("hero_clip_id"),
        "reference_clip_id": strategy_payload.get("reference_clip_id"),
        "source_root_detected_from_run": source_root,
        "master_rmd_dir": master_rmd_manifest["rmd_dir"],
        "correction_key_model": master_rmd_manifest["correction_key_model"],
        "entries": entries,
    }
    manifest_path = batch_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def _build_batch_readme(*, batch_root: Path, batch_manifest: Dict[str, object]) -> str:
    readme_path = batch_root / "README.txt"
    lines = [
        "R3DMatch MasterRMD Batch Handoff",
        "",
        "This batch folder maps each approved source clip to a production MasterRMD.",
        "Set SOURCE_ROOT in the generated shell script to the directory containing the source R3D media.",
        "MASTER_RMD_DIR defaults to ../MasterRMD relative to the batch folder.",
        "OUTPUT_ROOT controls where REDLine renders are written.",
        "",
        "Mapping model:",
        "- correction_key uses the first two underscore-delimited tokens from clip_id.",
        "- every clip listed in manifest.json is paired with the matching MasterRMD for that correction_key.",
        "",
        f"Approved strategy: {batch_manifest['approved_strategy']}",
        f"Hero clip: {batch_manifest.get('hero_clip_id')}",
        f"Detected source root from this run: {batch_manifest.get('source_root_detected_from_run')}",
        "",
        "Review manifest.json for exact clip-to-RMD mappings and authored correction payload summaries.",
        f"Run label: {batch_manifest.get('run_label')}",
        f"Matching domain: {batch_manifest.get('matching_domain_label')}",
    ]
    readme_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(readme_path)


def _render_batch_script_sh(*, batch_root: Path, batch_manifest: Dict[str, object]) -> str:
    script_path = batch_root / "transcode_with_master_rmd.sh"
    lines = [
        "#!/bin/sh",
        "set -eu",
        "",
        'SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"',
        'REDLINE_BIN="${REDLINE_BIN:-REDLine}"',
        'SOURCE_ROOT="${SOURCE_ROOT:-__SET_SOURCE_ROOT__}"',
        'MASTER_RMD_DIR="${MASTER_RMD_DIR:-$SCRIPT_DIR/../MasterRMD}"',
        'OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRIPT_DIR/output}"',
        'OUTPUT_EXT="${OUTPUT_EXT:-mov}"',
        "",
        'if [ "$SOURCE_ROOT" = "__SET_SOURCE_ROOT__" ]; then',
        '  echo "Set SOURCE_ROOT to the root directory containing the source R3D clips before running this script." >&2',
        "  exit 1",
        "fi",
        "",
        'mkdir -p "$OUTPUT_ROOT"',
        "",
        "# Add any production-specific REDLine output options directly to the commands below if needed.",
        "",
    ]
    for entry in batch_manifest["entries"]:
        source_relative = entry.get("source_r3d_relative_path") or entry["source_r3d_path"]
        lines.extend(
            [
                f"# clip_id: {entry['clip_id']}",
                f"# correction_key: {entry['correction_key']}",
                f"# master_rmd: {entry['master_rmd_name']}",
                f'"$REDLINE_BIN" --i "$SOURCE_ROOT/{source_relative}" --o "$OUTPUT_ROOT/{entry["clip_id"]}.$OUTPUT_EXT" --loadRMD "$MASTER_RMD_DIR/{entry["master_rmd_name"]}" --useRMD 1',
                "",
            ]
        )
    script_path.write_text("\n".join(lines), encoding="utf-8")
    script_path.chmod(0o755)
    return str(script_path)


def _render_batch_script_tcsh(*, batch_root: Path, batch_manifest: Dict[str, object]) -> str:
    script_path = batch_root / "transcode_with_master_rmd.tcsh"
    lines = [
        "#!/bin/tcsh",
        "set SCRIPT_DIR = `cd \"`dirname \"$0\"`\" && pwd`",
        "if ( ! $?REDLINE_BIN ) set REDLINE_BIN = REDLine",
        "if ( ! $?SOURCE_ROOT ) set SOURCE_ROOT = __SET_SOURCE_ROOT__",
        "if ( ! $?MASTER_RMD_DIR ) set MASTER_RMD_DIR = \"$SCRIPT_DIR/../MasterRMD\"",
        "if ( ! $?OUTPUT_ROOT ) set OUTPUT_ROOT = \"$SCRIPT_DIR/output\"",
        "if ( ! $?OUTPUT_EXT ) set OUTPUT_EXT = mov",
        "",
        "if ( \"$SOURCE_ROOT\" == \"__SET_SOURCE_ROOT__\" ) then",
        "  echo \"Set SOURCE_ROOT to the root directory containing the source R3D clips before running this script.\" >&2",
        "  exit 1",
        "endif",
        "",
        "mkdir -p \"$OUTPUT_ROOT\"",
        "",
        "# Add any production-specific REDLine output options directly to the commands below if needed.",
        "",
    ]
    for entry in batch_manifest["entries"]:
        source_relative = entry.get("source_r3d_relative_path") or entry["source_r3d_path"]
        lines.extend(
            [
                f"# clip_id: {entry['clip_id']}",
                f"# correction_key: {entry['correction_key']}",
                f"# master_rmd: {entry['master_rmd_name']}",
                f'"$REDLINE_BIN" --i "$SOURCE_ROOT/{source_relative}" --o "$OUTPUT_ROOT/{entry["clip_id"]}.$OUTPUT_EXT" --loadRMD "$MASTER_RMD_DIR/{entry["master_rmd_name"]}" --useRMD 1',
                "",
            ]
        )
    script_path.write_text("\n".join(lines), encoding="utf-8")
    script_path.chmod(0o755)
    return str(script_path)


def _load_review_package_payload(analysis_root: Path) -> Dict[str, object]:
    package_path = analysis_root / "report" / "review_package.json"
    if package_path.exists():
        return json.loads(package_path.read_text(encoding="utf-8"))
    return {}


def approve_master_rmd(
    analysis_dir: str,
    *,
    out_dir: Optional[str] = None,
    target_strategy: str = "median",
    reference_clip_id: Optional[str] = None,
    hero_clip_id: Optional[str] = None,
) -> Dict[str, object]:
    raise_if_cancelled("Run cancelled before approval export.")
    analysis_root = Path(analysis_dir).expanduser().resolve()
    approval_root = Path(out_dir).expanduser().resolve() if out_dir else analysis_root / "approval"
    approval_root.mkdir(parents=True, exist_ok=True)
    master_rmd_dir = approval_root / "MasterRMD"
    report_dir = analysis_root / "report"
    review_package = _load_review_package_payload(analysis_root)
    preview_settings = dict(review_package.get("preview_settings") or {})
    raise_if_cancelled("Run cancelled before rebuilding approval report.")
    report_payload = build_contact_sheet_report(
        str(analysis_root),
        out_dir=str(report_dir),
        clear_cache=False,
        target_type=review_package.get("target_type"),
        processing_mode=review_package.get("processing_mode"),
        run_label=review_package.get("run_label"),
        matching_domain=str(review_package.get("matching_domain", "scene")),
        selected_clip_ids=review_package.get("selected_clip_ids"),
        selected_clip_groups=review_package.get("selected_clip_groups"),
        preview_mode=str(review_package.get("preview_mode", "calibration")),
        preview_output_space=preview_settings.get("output_space"),
        preview_output_gamma=preview_settings.get("output_gamma"),
        preview_highlight_rolloff=preview_settings.get("highlight_rolloff"),
        preview_shadow_rolloff=preview_settings.get("shadow_rolloff"),
        preview_lut=preview_settings.get("lut_path"),
        calibration_roi=review_package.get("calibration_roi"),
        target_strategies=[target_strategy],
        reference_clip_id=reference_clip_id,
        hero_clip_id=hero_clip_id,
    )
    review_payload = json.loads(Path(report_payload["report_json"]).read_text(encoding="utf-8"))
    chosen_strategy = review_payload["strategies"][0]
    raise_if_cancelled("Run cancelled before MasterRMD export.")
    rmd_manifest = _write_master_rmds_from_strategy(chosen_strategy, out_dir=str(master_rmd_dir))
    batch_manifest = _build_batch_manifest(
        analysis_root=analysis_root,
        approval_root=approval_root,
        strategy_payload=chosen_strategy,
        master_rmd_manifest=rmd_manifest,
        run_label=review_package.get("run_label"),
        matching_domain=review_package.get("matching_domain"),
        selected_clip_ids=review_package.get("selected_clip_ids"),
        selected_clip_groups=review_package.get("selected_clip_groups"),
    )
    batch_root = approval_root / "batch"
    batch_readme = _build_batch_readme(batch_root=batch_root, batch_manifest=batch_manifest)
    batch_script_sh = _render_batch_script_sh(batch_root=batch_root, batch_manifest=batch_manifest)
    batch_script_tcsh = _render_batch_script_tcsh(batch_root=batch_root, batch_manifest=batch_manifest)

    approval_timestamp = datetime.now(timezone.utc).isoformat()
    approval_pdf_path = render_contact_sheet_pdf(
        review_payload,
        output_path=str(approval_root / "calibration_report.pdf"),
        title="R3DMatch Approval Report",
        timestamp_label=f"Approved at: {approval_timestamp}",
    )
    manifest = {
        "workflow_phase": "approved_master",
        "approved_at": approval_timestamp,
        "analysis_dir": str(analysis_root),
        "master_rmd_dir": str(master_rmd_dir),
        "master_rmd_folder_name": master_rmd_dir.name,
        "report_json": report_payload["report_json"],
        "report_html": report_payload["report_html"],
        "calibration_report_pdf": approval_pdf_path,
        "selected_target_strategy": chosen_strategy["strategy_key"],
        "selected_reference_clip_id": chosen_strategy.get("reference_clip_id"),
        "selected_hero_clip_id": chosen_strategy.get("hero_clip_id"),
        "run_label": review_package.get("run_label"),
        "matching_domain": review_package.get("matching_domain"),
        "matching_domain_label": review_package.get("matching_domain_label"),
        "selected_clip_ids": review_package.get("selected_clip_ids"),
        "selected_clip_groups": review_package.get("selected_clip_groups"),
        "target_type": review_payload.get("target_type"),
        "processing_mode": review_payload.get("processing_mode"),
        "calibration_roi": review_payload.get("calibration_roi"),
        "preview_transform": review_payload.get("preview_transform"),
        "measurement_preview_transform": review_payload.get("measurement_preview_transform"),
        "exposure_measurement_domain": review_payload.get("exposure_measurement_domain"),
        "clip_count": rmd_manifest["clip_count"],
        "correction_key_count": rmd_manifest["correction_key_count"],
        "correction_key_model": rmd_manifest["correction_key_model"],
        "master_rmd_exports": rmd_manifest["master_rmds"],
        "clip_mappings": rmd_manifest["clip_mappings"],
        "batch_dir": str(batch_root),
        "batch_manifest": batch_manifest["manifest_path"],
        "batch_scripts": {
            "sh": batch_script_sh,
            "tcsh": batch_script_tcsh,
        },
        "batch_readme": batch_readme,
    }
    manifest_path = approval_root / "approval_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest["approval_manifest"] = str(manifest_path)
    return manifest


__all__ = [
    "approve_master_rmd",
    "clear_preview_cache",
    "matching_domain_label",
    "normalize_matching_domain",
    "review_calibration",
    "resolve_review_output_dir",
    "validate_review_run_contract",
]
