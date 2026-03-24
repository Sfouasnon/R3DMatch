from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from .matching import analyze_path
from .report import REVIEW_PREVIEW_TRANSFORM, build_contact_sheet_report, build_review_package, clear_preview_cache, render_contact_sheet_pdf
from .rmd import write_rmd_for_clip


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
    preview_mode: str = "calibration",
    preview_output_space: Optional[str] = None,
    preview_output_gamma: Optional[str] = None,
    preview_highlight_rolloff: Optional[str] = None,
    preview_shadow_rolloff: Optional[str] = None,
    preview_lut: Optional[str] = None,
) -> Dict[str, object]:
    analyze_summary = analyze_path(
        input_path,
        out_dir=out_dir,
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
    )
    package = build_review_package(
        input_path,
        out_dir=out_dir,
        exposure_calibration_path=exposure_calibration_path or calibration_path,
        color_calibration_path=color_calibration_path,
        target_type=target_type,
        processing_mode=processing_mode,
        preview_mode=preview_mode,
        preview_output_space=preview_output_space,
        preview_output_gamma=preview_output_gamma,
        preview_highlight_rolloff=preview_highlight_rolloff,
        preview_shadow_rolloff=preview_shadow_rolloff,
        preview_lut=preview_lut,
        calibration_roi=calibration_roi,
        target_strategies=target_strategies,
        reference_clip_id=reference_clip_id,
    )
    package["analyze_summary"] = analyze_summary
    return package


def _write_master_rmds_from_strategy(strategy_payload: Dict[str, object], *, out_dir: str) -> Dict[str, object]:
    from pathlib import Path

    records = []
    target_dir = Path(out_dir).expanduser().resolve()
    for clip in strategy_payload["clips"]:
        gains = clip["metrics"]["color"]["rgb_gains"]
        sidecar_like = {
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
                "color": {"rgb_neutral_gains": gains},
            },
        }
        path = write_rmd_for_clip(str(clip["clip_id"]), sidecar_like, target_dir)
        records.append({"clip_id": clip["clip_id"], "rmd_path": str(path), "rmd_name": path.name, "source_path": clip["source_path"]})
    return {"rmd_dir": str(target_dir), "clip_count": len(records), "clips": records}

def approve_master_rmd(
    analysis_dir: str,
    *,
    out_dir: Optional[str] = None,
    target_strategy: str = "median",
    reference_clip_id: Optional[str] = None,
) -> Dict[str, object]:
    analysis_root = Path(analysis_dir).expanduser().resolve()
    approval_root = Path(out_dir).expanduser().resolve() if out_dir else analysis_root / "approval"
    approval_root.mkdir(parents=True, exist_ok=True)
    master_rmd_dir = approval_root / "Master_RMD"
    report_dir = analysis_root / "report"
    report_payload = build_contact_sheet_report(
        str(analysis_root),
        out_dir=str(report_dir),
        clear_cache=False,
        target_strategies=[target_strategy],
        reference_clip_id=reference_clip_id,
    )
    review_payload = json.loads(Path(report_payload["report_json"]).read_text(encoding="utf-8"))
    chosen_strategy = review_payload["strategies"][0]
    rmd_manifest = _write_master_rmds_from_strategy(chosen_strategy, out_dir=str(master_rmd_dir))
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
        "report_json": report_payload["report_json"],
        "report_html": report_payload["report_html"],
        "calibration_report_pdf": approval_pdf_path,
        "selected_target_strategy": chosen_strategy["strategy_key"],
        "selected_reference_clip_id": chosen_strategy.get("reference_clip_id"),
        "preview_transform": REVIEW_PREVIEW_TRANSFORM,
        "clip_count": rmd_manifest["clip_count"],
        "clips": rmd_manifest["clips"],
    }
    manifest_path = approval_root / "approval_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest["approval_manifest"] = str(manifest_path)
    return manifest


__all__ = [
    "approve_master_rmd",
    "clear_preview_cache",
    "review_calibration",
]
