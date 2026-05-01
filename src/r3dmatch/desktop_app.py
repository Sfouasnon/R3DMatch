from __future__ import annotations

import argparse
import html
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

from PySide6.QtCore import QProcess, QProcessEnvironment, QTimer, Qt, QUrl
from PySide6.QtGui import QAction, QColor, QDesktopServices, QFont, QPalette, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QScrollArea,
    QStatusBar,
    QTabWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
    QListWidgetItem,
    QFrame,
    QTableWidget,
    QTableWidgetItem,
    QProgressBar,
    QToolButton,
)

from .ftps_ingest import (
    DEFAULT_FTPS_PASSWORD,
    DEFAULT_FTPS_USERNAME,
    normalize_source_mode,
    plan_ftps_request,
    source_mode_label,
)
from .identity import clip_id_from_path, subset_key_from_clip_id
from .matching import discover_clips
from .native_dialogs import pick_existing_directory, pick_existing_file
from .progress import load_review_progress, review_progress_path_for
from .report import (
    normalize_report_focus,
    normalize_review_mode,
    normalize_target_strategy_name,
    report_focus_label,
    review_mode_label,
)
from .runtime_env import (
    ensure_runtime_environment,
    persist_redline_configured_path,
    redline_configured_path,
    runtime_cli_prefix,
    runtime_health_payload,
    runtime_subprocess_env,
)
from .workflow import (
    post_apply_verification_path_for,
    review_commit_payload_path_for,
    review_html_path_for,
    review_pdf_path_for,
    review_validation_path_for,
)


DEFAULT_TARGET_STRATEGIES = ["median", "optimal-exposure"]
DEFAULT_LOGO_PATH = Path(__file__).resolve().parent / "static" / "r3dmatch_logo.png"
DEFAULT_PREVIEW_STILL_FORMAT = "tiff"
DEFAULT_ARTIFACT_MODE = "production"
APP_STYLESHEET = """
QMainWindow, QWidget { background: #d7dde3; color: #0f172a; font-size: 13px; }
QScrollArea { border: none; background: transparent; }
QFrame#card { background: #fbfcfd; border: 1px solid #d8dde3; border-radius: 10px; }
QFrame#hero { background: #111827; border: 1px solid #111827; border-radius: 18px; }
QLabel { color: #0f172a; }
QFrame#hero QLabel { color: #d7dee8; }
QPushButton { background: #ffffff; color: #344054; border: 1px solid #d8dee6; border-radius: 9px; padding: 10px 14px; font-weight: 800; }
QPushButton:hover { background: #f8fafc; }
QPushButton:disabled { background: #eef2f7; color: #98a2b3; }
QLineEdit, QComboBox, QPlainTextEdit, QTextBrowser, QListWidget, QProgressBar {
  background: #fbfcfd; color: #0f172a; border: 1px solid #d8dee6; border-radius: 9px; padding: 8px 10px;
}
QTabWidget::pane { border: none; }
QTabBar::tab { background: #ffffff; border: 1px solid #d8dee6; border-radius: 10px; padding: 12px 14px; margin-right: 8px; min-width: 170px; font-weight: 800; color: #344054; }
QTabBar::tab:selected { background: #111827; border-color: #111827; color: #ffffff; }
QLabel[badgeRole="status"] { border-radius: 10px; padding: 6px 10px; font-size: 11px; font-weight: 800; border: 1px solid #d8dde3; background: #fbfcfd; }
QLabel[badgeState="good"] { background: #e7f4ea; color: #166534; border-color: #b8e5c5; }
QLabel[badgeState="warn"] { background: #fbf5e6; color: #92400e; border-color: #f5dda2; }
QLabel[badgeState="bad"] { background: #fef2f2; color: #991b1b; border-color: #fecaca; }
QLabel[badgeState="info"] { background: #eef4ff; color: #1d4ed8; border-color: #bfdbfe; }
QStatusBar { background: #f4f6f8; color: #475467; }
"""


DESKTOP_SURFACE_STYLES = """
<style>
body { font-family: "Avenir Next", "Helvetica Neue", Helvetica, Arial, sans-serif; color: #e5e7eb; background: #202225; margin: 0; }
.surface-root { padding: 4px 2px 10px; }
.surface-banner { border: 1px solid #3a3d42; border-radius: 16px; background: #262a31; padding: 16px 18px; margin-bottom: 14px; }
.surface-banner h2 { margin: 0 0 6px; font-size: 22px; color: #f8fafc; }
.surface-banner p { margin: 0; color: #cdd5df; line-height: 1.5; }
.surface-grid { display: block; }
.summary-grid { width: 100%; }
.summary-panel { display: inline-block; vertical-align: top; width: 32.1%; margin: 0 1.3% 14px 0; padding: 16px; border: 1px solid #3a3d42; border-radius: 16px; background: #262a31; }
.summary-panel:nth-child(3n) { margin-right: 0; }
.surface-heading-row { display: flex; justify-content: space-between; align-items: center; gap: 8px; margin-bottom: 10px; }
.surface-heading-row h3 { margin: 0; font-size: 20px; color: #f8fafc; }
.surface-kicker { margin: 0 0 8px; color: #9aa4b2; font-size: 12px; font-weight: 800; letter-spacing: 0.08em; text-transform: uppercase; }
.surface-lead { margin: 0 0 12px; color: #f3f4f6; font-size: 18px; font-weight: 800; line-height: 1.4; }
.surface-copy, .surface-note { color: #cdd5df; line-height: 1.5; }
.surface-note { font-size: 13px; }
.surface-metrics { width: 100%; margin: 10px 0 14px; }
.surface-metrics > div { display: inline-block; vertical-align: top; width: 48%; margin: 0 2% 10px 0; padding: 10px 12px; border: 1px solid #3a3d42; border-radius: 14px; background: #1f2329; }
.surface-metrics > div:nth-child(2n) { margin-right: 0; }
.metric-label { display: block; color: #9aa4b2; font-size: 11px; font-weight: 800; letter-spacing: 0.06em; text-transform: uppercase; }
.metric-value { display: block; margin-top: 5px; color: #f8fafc; font-size: 18px; font-weight: 800; line-height: 1.35; }
.surface-list, .compact-summary-list { margin: 8px 0 0 18px; color: #dbe4ef; line-height: 1.5; padding: 0; }
.surface-list li, .compact-summary-list li { margin-bottom: 6px; }
.surface-badge { display: inline-block; border-radius: 999px; padding: 6px 10px; font-size: 11px; font-weight: 800; letter-spacing: 0.04em; text-transform: uppercase; }
.surface-badge.success, .surface-badge.good { background: #1f3a23; color: #d8f0db; border: 1px solid #2e7d32; }
.surface-badge.warning { background: #3f2c10; color: #ffe0b3; border: 1px solid #b26a00; }
.surface-badge.pending, .surface-badge.info { background: #223348; color: #d7e7ff; border: 1px solid #4d7ea8; }
.surface-badge.danger, .surface-badge.outlier { background: #402127; color: #ffd6dc; border: 1px solid #b00020; }
.decision-banner { border: 1px solid #3a3d42; border-radius: 18px; background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); padding: 18px; margin-bottom: 14px; }
.decision-banner.success { background: linear-gradient(135deg, #17381f 0%, #1f3a23 100%); }
.decision-banner.warning { background: linear-gradient(135deg, #3b280b 0%, #4b320d 100%); }
.decision-banner.danger { background: linear-gradient(135deg, #401b24 0%, #541a22 100%); }
.decision-banner.pending, .decision-banner.info { background: linear-gradient(135deg, #203246 0%, #1d2635 100%); }
.decision-banner-kicker { color: #cbd5e1; font-size: 12px; font-weight: 800; letter-spacing: 0.08em; text-transform: uppercase; }
.decision-banner-title { margin-top: 8px; color: #f8fafc; font-size: 28px; font-weight: 900; }
.decision-banner-subtitle { margin: 8px 0 0; color: #dbe4ef; font-size: 15px; line-height: 1.5; }
.decision-banner-metrics { margin-top: 14px; }
.decision-banner-metrics > div { display: inline-block; vertical-align: top; width: 30%; margin: 0 3% 10px 0; }
.decision-banner-metrics > div:nth-child(3n) { margin-right: 0; }
.table-wrap { margin-top: 14px; overflow: auto; border: 1px solid #3a3d42; border-radius: 14px; background: #1f2329; }
.data-table { width: 100%; border-collapse: collapse; }
.data-table th, .data-table td { padding: 10px 12px; border-bottom: 1px solid #343942; text-align: left; vertical-align: top; color: #e5e7eb; }
.data-table th { color: #9aa4b2; font-size: 11px; font-weight: 800; letter-spacing: 0.06em; text-transform: uppercase; background: #191c21; }
.empty-state { padding: 18px; border: 1px dashed #3a3d42; border-radius: 14px; color: #cbd5e1; background: #23272f; }
.delta-positive { color: #90e0a0; font-weight: 800; }
.delta-negative { color: #f3b1bc; font-weight: 800; }
.delta-neutral { color: #cbd5e1; font-weight: 700; }
.push-callout { margin-bottom: 12px; color: #dbe4ef; font-weight: 700; }
.legend-bar { margin: 0 0 12px; padding: 12px 14px; border: 1px solid #3a3d42; border-radius: 14px; background: #23272f; color: #dbe4ef; }
.legend-pill { display: inline-block; margin: 0 8px 8px 0; padding: 6px 10px; border-radius: 999px; background: #1f2329; border: 1px solid #3a3d42; font-size: 12px; font-weight: 700; color: #dbe4ef; }
.results-actions { margin-top: 14px; color: #aeb7c5; font-size: 13px; }
.results-actions strong { color: #f8fafc; }
.camera-grid { margin-top: 10px; }
.camera-card { display: inline-block; vertical-align: top; width: 31.8%; margin: 0 1.8% 14px 0; padding: 14px; border-radius: 16px; border: 1px solid #3a3d42; background: #20242a; }
.camera-card:nth-child(3n) { margin-right: 0; }
.camera-card.good { border-color: #2e7d32; background: #18241b; }
.camera-card.warning { border-color: #b26a00; background: #2c2516; }
.camera-card.danger, .camera-card.outlier { border-color: #b00020; background: #2f191f; }
.camera-card.info { border-color: #4d7ea8; background: #1a2633; }
.camera-card h4 { margin: 0 0 8px; font-size: 18px; color: #f8fafc; }
.camera-subtitle { color: #aeb7c5; font-size: 13px; font-weight: 700; margin-bottom: 10px; }
.camera-metrics { margin: 0; padding: 0; list-style: none; }
.camera-metrics li { margin-bottom: 6px; color: #e5e7eb; font-size: 14px; line-height: 1.45; }
.camera-pair-grid { display: block; margin-top: 10px; }
.camera-frame-panel { display: inline-block; vertical-align: top; width: 48.5%; margin-right: 3%; }
.camera-frame-panel:last-child { margin-right: 0; }
.camera-frame-label { display: block; margin-bottom: 6px; color: #aeb7c5; font-size: 12px; font-weight: 800; letter-spacing: 0.06em; text-transform: uppercase; }
.camera-frame-panel img { width: 100%; border-radius: 12px; border: 1px solid #3a3d42; background: #16181d; }
</style>
"""


def build_review_command(
    *,
    repo_root: str,
    input_path: str,
    output_path: str,
    backend: str,
    target_type: str,
    processing_mode: str,
    roi_x: Optional[str],
    roi_y: Optional[str],
    roi_w: Optional[str],
    roi_h: Optional[str],
    target_strategies: List[str],
    reference_clip_id: Optional[str],
    preview_mode: str,
    preview_lut: Optional[str],
    preview_still_format: str = DEFAULT_PREVIEW_STILL_FORMAT,
    artifact_mode: str = DEFAULT_ARTIFACT_MODE,
    focus_validation: bool = False,
) -> List[str]:
    args = [
        *runtime_cli_prefix(),
        "review-calibration",
        input_path,
        "--out",
        output_path,
        "--backend",
        backend,
        "--target-type",
        target_type,
        "--processing-mode",
        processing_mode,
        "--review-mode",
        "full_contact_sheet",
        "--report-focus",
        "auto",
        "--preview-mode",
        preview_mode,
        "--preview-still-format",
        preview_still_format,
        "--artifact-mode",
        artifact_mode,
    ]
    if focus_validation:
        args.append("--focus-validation")
    if all(value not in (None, "") for value in (roi_x, roi_y, roi_w, roi_h)):
        args.extend(["--roi-x", str(roi_x), "--roi-y", str(roi_y), "--roi-w", str(roi_w), "--roi-h", str(roi_h)])
    for strategy in target_strategies:
        args.extend(["--target-strategy", normalize_target_strategy_name(strategy)])
    if reference_clip_id:
        args.extend(["--reference-clip-id", reference_clip_id])
    if preview_lut:
        args.extend(["--preview-lut", preview_lut])
    return args


def build_approve_command(*, repo_root: str, analysis_dir: str, target_strategy: str, reference_clip_id: Optional[str]) -> List[str]:
    args = [
        *runtime_cli_prefix(),
        "approve-master-rmd",
        analysis_dir,
        "--target-strategy",
        normalize_target_strategy_name(target_strategy),
    ]
    if reference_clip_id:
        args.extend(["--reference-clip-id", reference_clip_id])
    return args


def build_clear_cache_command(*, repo_root: str, analysis_dir: str) -> List[str]:
    return [*runtime_cli_prefix(), "clear-preview-cache", analysis_dir]


def scan_calibration_sources(input_path: str) -> Dict[str, object]:
    root = Path(input_path).expanduser().resolve()
    if not root.exists():
        return {
            "input_path": str(root),
            "exists": False,
            "clip_count": 0,
            "clip_ids": [],
            "sample_clip_ids": [],
            "clip_records": [],
            "clip_groups": [],
            "rdc_count": 0,
            "r3d_count": 0,
            "warning": "Selected calibration folder does not exist.",
        }
    clips = discover_clips(str(root))
    clip_ids = [clip_id_from_path(str(path)) for path in clips]
    clip_records = [
        {
            "clip_id": clip_id,
            "subset_group": subset_key_from_clip_id(clip_id),
            "camera_label": clip_id.split("_", 1)[0] if "_" in clip_id else clip_id,
            "source_path": str(path),
        }
        for clip_id, path in zip(clip_ids, clips)
    ]
    grouped: Dict[str, List[str]] = {}
    for clip_id in clip_ids:
        grouped.setdefault(subset_key_from_clip_id(clip_id), []).append(clip_id)
    clip_groups = [
        {"group_id": group_id, "clip_ids": clip_list, "clip_count": len(clip_list)}
        for group_id, clip_list in sorted(grouped.items())
    ]
    rdc_count = len({path.parent for path in clips if path.parent.suffix.lower() == ".rdc"})
    return {
        "input_path": str(root),
        "exists": True,
        "clip_count": len(clips),
        "clip_ids": clip_ids,
        "sample_clip_ids": clip_ids[:12],
        "clip_records": clip_records,
        "clip_groups": clip_groups,
        "rdc_count": rdc_count,
        "r3d_count": len(clips),
        "warning": None if clips else "No valid RED .R3D clips were found in the selected folder.",
    }


def run_ui_self_check(repo_root: str, *, minimal_mode: bool = False) -> List[str]:
    repo_root = str(Path(repo_root).expanduser().resolve())
    lines = [
        f"repo_root={repo_root}",
        "header section created",
        "calibration folder section created",
        "output folder section created",
        "basic settings section created",
        "source summary section created",
        "runtime health section created",
        "redline section created",
        "actions section created",
        "log section created",
        f"minimal_mode={minimal_mode}",
    ]
    return lines


def _artifact_json(path: Path) -> Optional[Dict[str, object]]:
    try:
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            return payload
    except Exception:
        return None
    return None


def _completed_review_package_score(root: Path) -> Optional[tuple[float, Path]]:
    validation_path = review_validation_path_for(root)
    html_path = review_html_path_for(root)
    pdf_path = review_pdf_path_for(root)
    if not validation_path.exists():
        return None
    payload = _artifact_json(validation_path) or {}
    if not isinstance(payload, dict):
        return None
    review_complete = bool(payload.get("review_complete"))
    status = str(payload.get("status") or "").strip().lower()
    if not review_complete and status not in {"success", "warning"}:
        return None
    if not html_path.exists() or not pdf_path.exists():
        return None
    try:
        mtime = max(
            validation_path.stat().st_mtime,
            html_path.stat().st_mtime,
            pdf_path.stat().st_mtime,
        )
    except OSError:
        return None
    return (mtime, root)


def _discover_review_output_root(candidate: Path) -> Optional[Path]:
    resolved = candidate.expanduser().resolve()
    direct = _completed_review_package_score(resolved)
    if direct is not None:
        return direct[1]
    if not resolved.exists() or not resolved.is_dir():
        return None
    best: Optional[tuple[float, Path]] = None
    for validation_path in resolved.glob("**/report/review_validation.json"):
        try:
            relative_depth = len(validation_path.relative_to(resolved).parts)
        except ValueError:
            continue
        if relative_depth > 6:
            continue
        review_root = validation_path.parent.parent
        candidate_score = _completed_review_package_score(review_root)
        if candidate_score is None:
            continue
        if best is None or candidate_score[0] > best[0]:
            best = candidate_score
    return best[1] if best is not None else None


def _open_local_path(path: Path) -> bool:
    return QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))


def _configure_application_palette(app: QApplication) -> None:
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor("#1e1f22"))
    palette.setColor(QPalette.ColorRole.WindowText, QColor("#e6e6e6"))
    palette.setColor(QPalette.ColorRole.Base, QColor("#202225"))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor("#2a2c30"))
    palette.setColor(QPalette.ColorRole.Text, QColor("#f3f4f6"))
    palette.setColor(QPalette.ColorRole.Button, QColor("#2a2c30"))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor("#f8fafc"))
    palette.setColor(QPalette.ColorRole.Highlight, QColor("#4d7ea8"))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
    app.setPalette(palette)
    app.setStyleSheet(APP_STYLESHEET)


def _desktop_surface_html(*sections: str) -> str:
    body = "".join(section for section in sections if section)
    return f"{DESKTOP_SURFACE_STYLES}<div class='surface-root'>{body}</div>"


def _desktop_exposure_tone(offset_stops: float) -> str:
    absolute = abs(float(offset_stops or 0.0))
    if absolute <= 0.25:
        return "good"
    if absolute <= 1.0:
        return "warning"
    return "danger"


def _desktop_confidence_tone(confidence: float) -> str:
    value = float(confidence or 0.0)
    if value >= 0.8:
        return "good"
    if value >= 0.55:
        return "warning"
    return "danger"


def _load_analysis_result_map(output_root: Path) -> Dict[str, Dict[str, object]]:
    analysis_dir = output_root / "analysis"
    results: Dict[str, Dict[str, object]] = {}
    if not analysis_dir.exists():
        return results
    for path in sorted(analysis_dir.glob("*.analysis.json")):
        payload = _artifact_json(path)
        if payload:
            results[str(payload.get("clip_id") or path.stem.replace(".analysis", ""))] = payload
    return results


def _desktop_camera_pair_html(row: Dict[str, object], *, title: str) -> str:
    original_path = str(row.get("original_image_path") or "").strip()
    corrected_path = str(row.get("corrected_image_path") or "").strip()
    if not original_path or not corrected_path:
        return ""
    original = Path(original_path)
    corrected = Path(corrected_path)
    if not original.exists() or not corrected.exists():
        return ""
    return (
        "<div class='summary-panel' style='width:100%;margin-right:0;'>"
        f"<div class='surface-heading-row'><h3>{html.escape(title)}</h3></div>"
        "<p class='surface-copy'>Original and corrected frames are shown side-by-side so the operator can visually sanity-check the recommended correction against the measured still.</p>"
        "<div class='camera-pair-grid'>"
        f"<div class='camera-frame-panel'><span class='camera-frame-label'>Original</span><img src='{html.escape(original.as_uri())}' alt='Original preview'></div>"
        f"<div class='camera-frame-panel'><span class='camera-frame-label'>Corrected</span><img src='{html.escape(corrected.as_uri())}' alt='Corrected preview'></div>"
        "</div>"
        "</div>"
    )


def _build_desktop_results_summary(context: Dict[str, object], output_root: Path) -> str:
    report_payload = dict(context.get("contact_sheet_payload") or {})
    corrected_validation = dict(context.get("corrected_validation") or {})
    ipp2_validation = dict(report_payload.get("ipp2_validation") or {})
    validation_summary = corrected_validation if list(corrected_validation.get("rows") or []) else ipp2_validation
    run_assessment = dict(report_payload.get("run_assessment") or {})
    recommended_strategy = dict(report_payload.get("recommended_strategy") or {})
    wb_model = dict(report_payload.get("white_balance_model") or {})
    scientific = dict(context.get("scientific_validation") or report_payload.get("scientific_validation") or {})
    focus_validation = dict(report_payload.get("focus_validation") or {})
    gray_target_consistency = dict(report_payload.get("gray_target_consistency") or run_assessment.get("gray_target_consistency") or {})
    strategy_key = str(validation_summary.get("recommended_strategy_key") or recommended_strategy.get("strategy_key") or "").strip()
    all_rows = [dict(item) for item in list(validation_summary.get("rows") or [])]
    rows = [row for row in all_rows if str(row.get("strategy_key") or "") == strategy_key] if strategy_key else all_rows
    analysis_map = _load_analysis_result_map(output_root)
    per_camera_analysis_map = {
        str(item.get("clip_id") or "").strip(): dict(item)
        for item in list(report_payload.get("per_camera_analysis") or [])
        if str(item.get("clip_id") or "").strip()
    }
    hero_clip_id = str((report_payload.get("hero_recommendation") or {}).get("candidate_clip_id") or report_payload.get("hero_clip_id") or "").strip()
    anchor_clip_id = str(recommended_strategy.get("reference_clip_id") or "").strip()

    enriched_rows: List[Dict[str, object]] = []
    for row in rows:
        clip_id = str(row.get("clip_id") or "").strip()
        analysis = dict(analysis_map.get(clip_id) or {})
        per_camera = dict(per_camera_analysis_map.get(clip_id) or {})
        diagnostics = dict(analysis.get("diagnostics") or {})
        exposure_metrics = dict((per_camera.get("metrics") or {}).get("exposure") or {})
        confidence = float(analysis.get("confidence", 0.0) or 0.0)
        offset = float(row.get("applied_correction_stops", 0.0) or 0.0)
        reference_use = str(row.get("reference_use") or "Included")
        trust_class = str(row.get("trust_class") or "")
        excluded = reference_use == "Excluded" or trust_class in {"EXCLUDED", "UNTRUSTED"}
        tone = "danger" if excluded else _desktop_exposure_tone(offset)
        target_class = str(
            per_camera.get("gray_target_class")
            or exposure_metrics.get("gray_target_class")
            or diagnostics.get("gray_target_class")
            or row.get("gray_target_class")
            or "sphere"
        ).replace("_", " ").title()
        detection_method = str(
            per_camera.get("gray_target_detection_method")
            or exposure_metrics.get("gray_target_detection_method")
            or diagnostics.get("gray_target_detection_method")
            or row.get("gray_target_detection_method")
            or ""
        ).replace("_", " ")
        sample_plausibility = str(
            per_camera.get("sample_plausibility")
            or exposure_metrics.get("sample_plausibility")
            or diagnostics.get("sample_plausibility")
            or row.get("sample_plausibility")
            or ""
        )
        review_required = bool(
            per_camera.get("gray_target_review_recommended")
            or exposure_metrics.get("gray_target_review_recommended")
            or diagnostics.get("gray_target_review_recommended")
        )
        enriched_rows.append(
            {
                **row,
                "monitoring_log2": float(
                    diagnostics.get(
                        "measured_log2_luminance_monitoring",
                        diagnostics.get("measured_log2_luminance", row.get("original_measured_gray_log2", 0.0)),
                    )
                    or 0.0
                ),
                "confidence_value": confidence,
                "tone": tone,
                "excluded": excluded,
                "is_hero": clip_id == hero_clip_id,
                "is_anchor": clip_id == anchor_clip_id,
                "target_class_label": target_class,
                "detection_method_label": detection_method.title() if detection_method else "Sphere solve",
                "sample_plausibility": sample_plausibility.title() if sample_plausibility else "n/a",
                "review_required": review_required,
            }
        )
    retained = [row for row in enriched_rows if not bool(row.get("excluded"))]
    excluded = [row for row in enriched_rows if bool(row.get("excluded"))]
    retained_sorted = sorted(retained, key=lambda item: abs(float(item.get("applied_correction_stops", 0.0) or 0.0)))
    excluded_sorted = sorted(excluded, key=lambda item: abs(float(item.get("applied_correction_stops", 0.0) or 0.0)), reverse=True)
    hero_row = next((row for row in enriched_rows if bool(row.get("is_hero"))), None)
    if hero_row is None and retained_sorted:
        hero_row = retained_sorted[0]
    comparison_rows: List[Dict[str, object]] = []
    if hero_row is not None:
        comparison_rows.append(hero_row)
    if excluded_sorted:
        worst = excluded_sorted[0]
        if worst is not hero_row:
            comparison_rows.append(worst)
    elif len(retained_sorted) >= 2:
        widest = max(retained_sorted, key=lambda item: abs(float(item.get("applied_correction_stops", 0.0) or 0.0)))
        if widest is not hero_row:
            comparison_rows.append(widest)

    retained_cards = "".join(
        (
            f"<div class='camera-card {html.escape(str(row.get('tone') or 'good'))}'>"
            f"<div class='surface-badge {html.escape(str(row.get('tone') or 'good'))}'>{html.escape('Exposure Anchor' if row.get('is_anchor') else 'Retained Cluster')}</div>"
            f"<h4>{html.escape(str(row.get('camera_label') or row.get('camera_id') or row.get('clip_id') or 'Camera'))}</h4>"
            f"<div class='camera-subtitle'>{html.escape(str(row.get('clip_id') or ''))}</div>"
            "<ul class='camera-metrics'>"
            f"<li><strong>Adjustment:</strong> {float(row.get('applied_correction_stops', 0.0) or 0.0):+.2f} stops</li>"
            f"<li><strong>Target:</strong> {html.escape(str(row.get('target_class_label') or 'Gray sphere'))}</li>"
            f"<li><strong>Solve:</strong> {html.escape(str(row.get('detection_method_label') or 'Sphere solve'))}</li>"
            f"<li><strong>Plausibility:</strong> {html.escape(str(row.get('sample_plausibility') or 'n/a'))} | confidence <span class='surface-badge {html.escape(_desktop_confidence_tone(float(row.get('confidence_value', 0.0) or 0.0)))}'>{float(row.get('confidence_value', 0.0) or 0.0):.2f}</span></li>"
            f"<li><strong>Notes:</strong> {html.escape(str(row.get('trust_reason') or 'Retained in cluster'))}</li>"
            "</ul>"
            "</div>"
        )
        for row in retained_sorted[:6]
    )
    excluded_cards = "".join(
        (
            f"<div class='camera-card danger'>"
            "<div class='surface-badge danger'>Excluded Camera</div>"
            f"<h4>{html.escape(str(row.get('camera_label') or row.get('camera_id') or row.get('clip_id') or 'Camera'))}</h4>"
            f"<div class='camera-subtitle'>{html.escape(str(row.get('clip_id') or ''))}</div>"
            "<ul class='camera-metrics'>"
            f"<li><strong>Adjustment:</strong> {float(row.get('applied_correction_stops', 0.0) or 0.0):+.2f} stops</li>"
            f"<li><strong>Target:</strong> {html.escape(str(row.get('target_class_label') or 'Gray sphere'))}</li>"
            f"<li><strong>Solve:</strong> {html.escape(str(row.get('detection_method_label') or 'Sphere solve'))}</li>"
            f"<li><strong>Plausibility:</strong> {html.escape(str(row.get('sample_plausibility') or 'n/a'))} | confidence <span class='surface-badge {html.escape(_desktop_confidence_tone(float(row.get('confidence_value', 0.0) or 0.0)))}'>{float(row.get('confidence_value', 0.0) or 0.0):.2f}</span></li>"
            f"<li><strong>Reason:</strong> {html.escape(str(row.get('trust_reason') or row.get('residual_label') or 'Flagged for operator review'))}</li>"
            "</ul>"
            "</div>"
        )
        for row in excluded_sorted[:6]
    )
    retained_empty_html = "<div class='empty-state'>No retained cameras were found for the recommended strategy.</div>"
    excluded_empty_html = "<div class='empty-state'>No cameras were excluded for the recommended strategy.</div>"
    scientific_state = str(scientific.get("status") or "pending").replace("_", " ")
    scientific_tone = "success" if str(scientific.get("status") or "") == "fully_reconciled" else "warning" if scientific else "pending"
    readiness_status = str(run_assessment.get("status") or "pending").replace("_", " ").title()
    readiness_tone = (
        "good"
        if str(run_assessment.get("status") or "") == "READY"
        else "danger"
        if str(run_assessment.get("status") or "") == "DO_NOT_PUSH"
        else "warning"
    )
    attention_list = [
        str(item).strip()
        for item in list(report_payload.get("recommended_attention") or [])
        if str(item).strip()
    ]
    attention_text = ", ".join(attention_list[:8]) or "No special attention cameras."
    selected_groups_text = ", ".join(report_payload.get("selected_clip_groups") or []) or "all"
    gray_target_summary = str(gray_target_consistency.get("summary") or "Gray target basis is not available.")
    best_residual = corrected_validation.get("best_residual_stops")
    if best_residual is None:
        best_residual = validation_summary.get("best_residual")
    worst_residual = corrected_validation.get("worst_residual_stops")
    if worst_residual is None:
        worst_residual = validation_summary.get("max_residual")
    outside_tolerance_count = len(list(corrected_validation.get("outside_tolerance_cameras") or []))
    if outside_tolerance_count <= 0:
        outside_tolerance_count = int(((validation_summary.get("status_counts") or {}).get("FAIL", 0) or 0))
    consistency_warning = ""
    if bool(gray_target_consistency.get("mixed_target_classes")):
        consistency_warning = (
            "<div class='surface-banner'>"
            "<h2>Gray-target consistency warning</h2>"
            f"<p><strong>Target basis:</strong> {html.escape(gray_target_summary)}<br>"
            f"<strong>Action:</strong> Safe commit export is blocked until the run is measured consistently from one target class.</p>"
            "</div>"
        )
    return "".join(
        [
            consistency_warning,
            "<div class='surface-grid'>",
            (
                "<div class='summary-panel'>"
                "<div class='surface-kicker'>Calibration Recommendation</div>"
                f"<div class='surface-heading-row'><h3>{html.escape(str(recommended_strategy.get('strategy_label') or 'Recommendation pending'))}</h3>"
                f"<span class='surface-badge {html.escape(readiness_tone)}'>{html.escape(readiness_status)}</span></div>"
                f"<p class='surface-lead'>{html.escape(str(recommended_strategy.get('strategy_summary') or report_payload.get('executive_synopsis') or 'Run review to populate recommendation framing.'))}</p>"
                "<div class='surface-metrics'>"
                f"<div><span class='metric-label'>Exposure Anchor</span><span class='metric-value'>{html.escape(str(recommended_strategy.get('reference_clip_id') or recommended_strategy.get('anchor_source') or 'Pending'))}</span></div>"
                f"<div><span class='metric-label'>Hero Camera</span><span class='metric-value'>{html.escape(hero_clip_id or str((report_payload.get('hero_recommendation') or {}).get('candidate_clip_id') or 'Not set'))}</span></div>"
                f"<div><span class='metric-label'>Retained Cluster</span><span class='metric-value'>{len(retained)}</span></div>"
                f"<div><span class='metric-label'>Excluded Cameras</span><span class='metric-value'>{len(excluded)}</span></div>"
                "</div>"
                f"<ul class='compact-summary-list'><li>{html.escape(str(run_assessment.get('operator_note') or 'Review the retained cluster before apply.'))}</li><li>{html.escape(f'Attention cameras: {attention_text}')}</li></ul>"
                "</div>"
            ),
            (
                "<div class='summary-panel'>"
                "<div class='surface-kicker'>White-Balance Model</div>"
                f"<div class='surface-heading-row'><h3>{html.escape(str(wb_model.get('model_label') or 'n/a'))}</h3>"
                f"<span class='surface-badge info'>{int(wb_model.get('candidate_count', 0) or 0)} models</span></div>"
                f"<p class='surface-copy'>{html.escape(str(wb_model.get('selection_reason') or 'No white-balance model summary is available yet.'))}</p>"
                "<div class='surface-metrics'>"
                f"<div><span class='metric-label'>Shared Kelvin</span><span class='metric-value'>{html.escape(str(wb_model.get('shared_kelvin') or wb_model.get('shared_kelvin_mode') or 'per-camera'))}</span></div>"
                f"<div><span class='metric-label'>Tint Mode</span><span class='metric-value'>{html.escape(str(wb_model.get('shared_tint_mode') or 'per-camera'))}</span></div>"
                f"<div><span class='metric-label'>Mean Residual After Solve</span><span class='metric-value'>{float(((wb_model.get('diagnostics') or {}).get('mean_neutral_residual_after_solve', 0.0) or 0.0)):.4f}</span></div>"
                f"<div><span class='metric-label'>Source Samples</span><span class='metric-value'>{int(((wb_model.get('diagnostics') or {}).get('source_sample_count', 0) or 0))}</span></div>"
                "</div>"
                "</div>"
            ),
            (
                "<div class='summary-panel'>"
                "<div class='surface-kicker'>Physical / Scientific Validation</div>"
                f"<div class='surface-heading-row'><h3>{html.escape(scientific_state.title())}</h3>"
                f"<span class='surface-badge {html.escape(scientific_tone)}'>{html.escape(scientific_state)}</span></div>"
                f"<p class='surface-copy'>{html.escape(str(scientific.get('reason') or 'Scientific validation will appear here when artifacts are available.'))}</p>"
                "<div class='surface-metrics'>"
                f"<div><span class='metric-label'>Best Residual</span><span class='metric-value'>{float(best_residual or 0.0):.3f}</span></div>"
                f"<div><span class='metric-label'>Worst Residual</span><span class='metric-value'>{float(worst_residual or 0.0):.3f}</span></div>"
                f"<div><span class='metric-label'>Outside Tolerance</span><span class='metric-value'>{outside_tolerance_count}</span></div>"
                f"<div><span class='metric-label'>Gray Sample Basis</span><span class='metric-value'>{html.escape(str(gray_target_consistency.get('dominant_target_class') or 'sphere').replace('_', ' ').title())}</span></div>"
                "</div>"
                f"<ul class='compact-summary-list'><li>{html.escape(gray_target_summary)}</li><li>{html.escape('Selected groups: ' + selected_groups_text)}</li></ul>"
                "</div>"
            ),
            (
                "<div class='summary-panel'>"
                "<div class='surface-kicker'>Focus Validation</div>"
                f"<div class='surface-heading-row'><h3>{html.escape(str(focus_validation.get('status') or 'Disabled').replace('_', ' ').title())}</h3>"
                f"<span class='surface-badge {'success' if focus_validation.get('tiff_is_sufficient') else 'warning' if focus_validation.get('enabled') else 'info'}'>{html.escape('TIFF sufficient' if focus_validation.get('tiff_is_sufficient') else 'Review required' if focus_validation.get('enabled') else 'Disabled')}</span></div>"
                f"<p class='surface-copy'>{html.escape(str(focus_validation.get('reason') or 'Focus validation is disabled for this run.'))}</p>"
                "<div class='surface-metrics'>"
                f"<div><span class='metric-label'>Sharp</span><span class='metric-value'>{int(focus_validation.get('sharp_camera_count', 0) or 0)}</span></div>"
                f"<div><span class='metric-label'>Review</span><span class='metric-value'>{int(focus_validation.get('review_camera_count', 0) or 0)}</span></div>"
                f"<div><span class='metric-label'>Soft</span><span class='metric-value'>{int(focus_validation.get('soft_camera_count', 0) or 0)}</span></div>"
                f"<div><span class='metric-label'>Rank Correlation</span><span class='metric-value'>{float(focus_validation.get('rank_correlation', 0.0) or 0.0):.3f}</span></div>"
                "</div>"
                f"<ul class='compact-summary-list'><li>{html.escape(str(focus_validation.get('measured_error') or 'No focus metrics were recorded.'))}</li></ul>"
                "</div>"
            ),
            "</div>",
            (
                "<div class='summary-panel' style='width:100%;margin-right:0;'>"
                "<div class='surface-heading-row'><h3>Retained Cluster</h3></div>"
                "<p class='surface-copy'>These cards are intentionally compact: adjustment, target basis, solve path, plausibility, and the one note the operator needs next.</p>"
                f"<div class='camera-grid'>{retained_cards or retained_empty_html}</div>"
                "</div>"
            ),
            (
                "<div class='summary-panel' style='width:100%;margin-right:0;'>"
                "<div class='surface-heading-row'><h3>Excluded Cameras</h3></div>"
                "<p class='surface-copy'>This section is explicit on purpose. Excluded cameras should never be buried inside the retained cluster summary.</p>"
                f"<div class='camera-grid'>{excluded_cards or excluded_empty_html}</div>"
                "</div>"
            ),
            "".join(
                _desktop_camera_pair_html(
                    row,
                    title=f"Visual Comparison: {str(row.get('camera_label') or row.get('clip_id') or 'Camera')}",
                )
                for row in comparison_rows
            ),
        ]
    )


class R3DMatchDesktopWindow(QMainWindow):
    def __init__(self, *, repo_root: str, minimal_mode: bool = False) -> None:
        super().__init__()
        self.repo_root = str(Path(repo_root).expanduser().resolve())
        self.minimal_mode = minimal_mode
        self.active_output_root: Optional[Path] = None
        self.last_task_kind = ""
        self.last_task_output_root: Optional[Path] = None
        self.last_manifest_path: Optional[Path] = None
        self.run_review_button: Optional[QPushButton] = None
        self.approve_button: Optional[QPushButton] = None
        self.clear_preview_button: Optional[QPushButton] = None
        self.open_html_button: Optional[QPushButton] = None
        self.open_pdf_button: Optional[QPushButton] = None
        self.open_output_button: Optional[QPushButton] = None
        self.read_current_values_button: Optional[QPushButton] = None
        self.dry_run_push_button: Optional[QPushButton] = None
        self.live_push_button: Optional[QPushButton] = None
        self.verify_push_button: Optional[QPushButton] = None
        self.progress_surface: Optional[QPlainTextEdit] = None
        self.runtime_chip: Optional[QLabel] = None
        self.redline_chip: Optional[QLabel] = None
        self.pdf_chip: Optional[QLabel] = None
        self.context_chip: Optional[QLabel] = None
        self.source_summary: Optional[QTextBrowser] = None
        self.runtime_health: Optional[QTextBrowser] = None
        self.results_summary: Optional[QTextBrowser] = None
        self.apply_summary: Optional[QTextBrowser] = None
        self.results_banner: Optional[QLabel] = None
        self.apply_banner: Optional[QLabel] = None
        self.workflow_banner: Optional[QLabel] = None
        self.scan_hint: Optional[QLabel] = None
        self.clip_group_list: Optional[QListWidget] = None
        self.clip_group_summary: Optional[QLabel] = None
        self.last_scan_summary: Dict[str, object] = {}

        self.process = QProcess(self)
        self.process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._drain_process_output)
        self.process.finished.connect(self._task_finished)
        self.progress_timer = QTimer(self)
        self.progress_timer.setInterval(1200)
        self.progress_timer.timeout.connect(self._refresh_progress_state)

        self.setWindowTitle("R3DMatch")
        self.resize(1460, 980)
        self.setObjectName("mainWindow")

        self.status_bar = QStatusBar(self)
        self.setStatusBar(self.status_bar)

        self._build_ui()
        self._bind_signals()
        self._refresh_runtime_health()
        self._refresh_source_mode_visibility()
        self._refresh_artifacts()
        self._refresh_button_state()

    def _build_ui(self) -> None:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        root = QWidget()
        scroll.setWidget(root)
        self.setCentralWidget(scroll)

        outer = QVBoxLayout(root)
        outer.setContentsMargins(18, 18, 18, 18)
        outer.setSpacing(14)

        header_card = self.make_card()
        header = QVBoxLayout(header_card)
        header.setContentsMargins(14, 12, 14, 12)
        header.setSpacing(6)

        top_row = QHBoxLayout()
        top_row.setSpacing(12)

        left = QVBoxLayout()
        left.setSpacing(2)
        left.addWidget(QLabel("<span style='font-size:11px;font-weight:900;letter-spacing:1.6px;color:#667085;'>R3DMATCH OPERATOR CONSOLE</span>"))
        left.addWidget(QLabel("<span style='font-size:40px;font-weight:900;color:#101828;'>R3DMatch</span>"))
        left.addWidget(QLabel("<span style='font-size:15px;color:#475467;'>Ingest / Scan / Calibrate / Review / Push Looks</span>"))
        left.addWidget(QLabel("<span style='font-size:14px;color:#475467;'>Operator desktop for ingest, camera analysis, calibration review, manual sphere assist, and camera writeback.</span>"))
        top_row.addLayout(left, 1)

        right = QVBoxLayout()
        right.setSpacing(6)
        self.runtime_chip = self._status_badge('RED SDK · Checking', 'info')
        self.redline_chip = self._status_badge('REDLine · Checking', 'info')
        self.pdf_chip = self._status_badge('HTML/PDF · Checking', 'info')
        self.context_chip = self._status_badge('Packaged App · Checking', 'info')
        for w in [self.runtime_chip, self.redline_chip, self.pdf_chip, self.context_chip]:
            right.addWidget(w)
        refresh_health = QPushButton('Refresh Runtime Health')
        refresh_health.clicked.connect(self._refresh_runtime_health)
        refresh_health.setMaximumWidth(180)
        right.addWidget(refresh_health, alignment=Qt.AlignmentFlag.AlignRight)
        top_row.addLayout(right)

        header.addLayout(top_row)
        self.workflow_banner = QLabel()
        self.workflow_banner.setWordWrap(True)
        self.workflow_banner.hide()
        header.addWidget(self.workflow_banner)
        outer.addWidget(header_card)

        self.log_output = QPlainTextEdit(); self.log_output.setReadOnly(True); self.log_output.setMinimumHeight(200)
        self.tabs = QTabWidget(); outer.addWidget(self.tabs,1)

        ingest_tab=QWidget(); calibrate_tab=QWidget(); review_tab=QWidget(); push_tab=QWidget(); settings_tab=QWidget()
        runtime_tab=QWidget()
        self.review_tab=ingest_tab; self.results_tab=review_tab; self.apply_tab=push_tab; self.settings_tab=runtime_tab
        self.tabs.addTab(ingest_tab,'Ingest'); self.tabs.addTab(calibrate_tab,'Scan'); self.tabs.addTab(review_tab,'Calibrate'); self.tabs.addTab(push_tab,'Review'); self.tabs.addTab(settings_tab,'Push Looks To Camera'); self.tabs.addTab(runtime_tab,'Settings')
        self._build_ingest_tab(ingest_tab)
        self._build_scan_tab(calibrate_tab)
        self._build_calibrate_tab(review_tab)
        self._build_results_tab(push_tab)
        self._build_apply_tab(settings_tab)
        self._build_runtime_tab(runtime_tab)

    def make_card(self) -> QFrame:
        c=QFrame(); c.setObjectName('card'); return c

    def make_hero(self, kicker:str, title:str, subtitle:str) -> QWidget:
        card=self.make_card(); card.setObjectName('hero')
        lay=QVBoxLayout(card)
        lay.addWidget(QLabel(f"<span style='font-size:13px;letter-spacing:1px;color:#93c5fd;font-weight:700'>{kicker.upper()}</span>"))
        lay.addWidget(QLabel(f"<span style='font-size:44px;font-weight:800;color:#fff'>{title}</span>"))
        sub=QLabel(subtitle); sub.setWordWrap(True); lay.addWidget(sub)
        return card

    def _build_ingest_tab(self, tab: QWidget) -> None:
        layout = QVBoxLayout(tab); layout.setSpacing(12)
        layout.addWidget(self.make_hero('Ingest','R3DMatch Ingest','Set source/destination, scan source, then run.'))
        top=QHBoxLayout();
        source=self.make_card(); sg=QGridLayout(source)
        self.source_mode=QComboBox(); self.source_mode.addItem('Local Folder','local_folder'); self.source_mode.addItem('FTPS Camera Pull','ftps_camera_pull')
        self.input_path=QLineEdit(); self.output_path=QLineEdit(); self.local_ingest_root=QLineEdit(); self.verification_after_path=QLineEdit(); self.ftps_reel=QLineEdit(); self.ftps_clips=QLineEdit(); self.ftps_cameras=QLineEdit()
        self.backend=QComboBox(); self.backend.addItems(['red','mock'])
        self.target_type=QComboBox(); self.target_type.addItems(['gray_sphere','gray_card','color_chart'])
        self.processing_mode=QComboBox(); self.processing_mode.addItems(['both','exposure','color'])
        self.review_mode=QComboBox(); [self.review_mode.addItem(review_mode_label(v),v) for v in ['full_contact_sheet','lightweight_analysis']]
        self.report_focus=QComboBox(); [self.report_focus.addItem(report_focus_label(v),v) for v in ['auto','full','outliers','anchors','cluster_extremes']]
        self.preview_mode=QComboBox(); self.preview_mode.addItem('Monitoring (IPP2 / BT.709 / BT.1886)','monitoring')
        self.preview_lut=QLineEdit(); self.preview_still_format=QComboBox(); self.preview_still_format.addItem('TIFF','tiff'); self.preview_still_format.addItem('JPEG','jpeg')
        self.focus_validation_check=QCheckBox('Enable focus validation'); self.artifact_mode=QComboBox(); self.artifact_mode.addItem('Production','production'); self.artifact_mode.addItem('Debug','debug')
        self.matching_domain=QComboBox(); self.matching_domain.addItem('Display (IPP2 / BT.709 / BT.1886)','perceptual'); self.matching_domain.addItem('Scene','scene')
        self.reference_clip_id=QLineEdit(); self.hero_clip_id=QLineEdit(); self.roi_x=QLineEdit(); self.roi_y=QLineEdit(); self.roi_w=QLineEdit(); self.roi_h=QLineEdit()
        self.strategy_checks={}
        rr=QHBoxLayout();
        for n in ['median','optimal-exposure','manual','hero-camera']:
            b=QCheckBox(n); b.setChecked(n in DEFAULT_TARGET_STRATEGIES); self.strategy_checks[n]=b; rr.addWidget(b)
        rr.addStretch(1)
        fields=[('Source Mode',self.source_mode),('Calibration Folder',self._wrap_layout(self._path_row(self.input_path,self._browse_input))),('Output Folder',self._wrap_layout(self._path_row(self.output_path,self._browse_output))),('Camera Subset',self.ftps_cameras),('Target',self.target_type),('Strategy',self._wrap_layout(rr)),('Processing',self.processing_mode),('Report',self.review_mode)]
        for i,(k,w) in enumerate(fields): sg.addWidget(self.make_field_row(k,w),i,0)
        adv=QToolButton(); adv.setText('Advanced'); adv.setCheckable(True); adv.setChecked(False)
        adv_panel=self.make_card(); adv_l=QGridLayout(adv_panel)
        for i,(k,w) in enumerate([('Local Ingest Cache Root',self._wrap_layout(self._path_row(self.local_ingest_root,self._browse_ingest_root))),('After-Apply Review Folder',self._wrap_layout(self._path_row(self.verification_after_path,self._browse_after_apply))),('FTPS Reel',self.ftps_reel),('FTPS Clips / Ranges',self.ftps_clips),('normalized ROI',self.roi_x),('reference clip',self.reference_clip_id),('hero clip',self.hero_clip_id),('preview LUT',self.preview_lut),('matching internals',self.matching_domain)]): adv_l.addWidget(self.make_field_row(k,w),i,0)
        adv_panel.setVisible(False); adv.toggled.connect(adv_panel.setVisible)
        left=QVBoxLayout(); left.addWidget(source); left.addWidget(adv); left.addWidget(adv_panel)
        layout.addLayout(left)
        self.scan_button=QPushButton('Scan Source'); self.scan_button.clicked.connect(self._scan_source)
        self.discover_button=QPushButton('Discover'); self.discover_button.clicked.connect(self._discover_ftps)
        self.download_button=QPushButton('Download'); self.download_button.clicked.connect(self._download_ftps)
        self.download_process_button=QPushButton('Download + Process'); self.download_process_button.clicked.connect(self._download_then_process)
        self.retry_failed_button=QPushButton('Retry Failed'); self.retry_failed_button.clicked.connect(self._retry_failed_ftps)
        row=QHBoxLayout(); [row.addWidget(w) for w in [self.scan_button,self.discover_button,self.download_button,self.download_process_button,self.retry_failed_button]]; row.addStretch(1); layout.addLayout(row)
        self.source_summary=QTextBrowser(); self.source_summary.setMinimumHeight(90); self.source_summary.setHtml('<h3>Awaiting source scan</h3><p>Use Continue to Scan to run source scan.</p>')
        self.clip_group_summary=QLabel('No clip groups yet. Scan source to populate groups.')
        self.clip_group_list=QListWidget(); self.clip_group_list.setMinimumHeight(90)
        self.run_review_button=QPushButton('Run Review'); self.run_review_button.clicked.connect(self._run_review)
        self.approve_button=QPushButton('Approve Master RMD'); self.approve_button.clicked.connect(self._approve_master)
        self.clear_preview_button=QPushButton('Clear Preview Cache'); self.clear_preview_button.clicked.connect(self._clear_preview_cache)
        go_scan=QPushButton('Continue to Scan / Scan Source'); go_scan.clicked.connect(lambda: self.tabs.setCurrentIndex(1)); layout.addWidget(go_scan)


    def _build_scan_tab(self, tab: QWidget) -> None:
        layout = QVBoxLayout(tab); layout.setSpacing(12)
        layout.addWidget(self.make_hero('Scan','Source Scan','Scan source, review groups, then run review.'))
        row=QHBoxLayout(); [row.addWidget(w) for w in [self.scan_button,self.discover_button,self.download_button,self.download_process_button,self.retry_failed_button]]; row.addStretch(1); layout.addLayout(row)
        self.source_summary.setMinimumHeight(120); layout.addWidget(self.source_summary)
        layout.addWidget(self.clip_group_summary)
        self.clip_group_list.setMinimumHeight(90); layout.addWidget(self.clip_group_list)
        r2=QHBoxLayout(); [r2.addWidget(w) for w in [self.run_review_button,self.approve_button,self.clear_preview_button]]; r2.addStretch(1); layout.addLayout(r2)
        self.log_output.setMinimumHeight(140); layout.addWidget(self.log_output)
    def _build_calibrate_tab(self, tab: QWidget) -> None:
        l=QVBoxLayout(tab); l.addWidget(self.make_hero('Calibrate','Calibrate','Track run status, ETA, and terminal output.'))
        self.calibrate_progress=QProgressBar(); self.calibrate_progress.setValue(0); l.addWidget(self.calibrate_progress)
        self.progress_surface=QPlainTextEdit(); self.progress_surface.setReadOnly(True); self.progress_surface.setMinimumHeight(80); l.addWidget(self.progress_surface)

    def _build_runtime_tab(self, tab: QWidget) -> None:
        layout=QVBoxLayout(tab); layout.addWidget(self.make_hero('Settings','Settings','Runtime health, REDLine configuration, diagnostics.'))
        self.runtime_health=QTextBrowser(); self.runtime_health.setMinimumHeight(260); layout.addWidget(self.runtime_health)
        self.redline_path=QLineEdit(redline_configured_path())
        self.save_redline_button=QPushButton('Save / Apply'); self.save_redline_button.clicked.connect(self._save_redline)
        reload=QPushButton('Reload'); reload.clicked.connect(self._reload_redline)
        self.redline_feedback=QLabel('')
        layout.addWidget(self.make_field_row('REDLine path',self._wrap_layout(self._path_row(self.redline_path,self._browse_redline))))
        rr=QHBoxLayout(); rr.addWidget(self.save_redline_button); rr.addWidget(reload); rr.addStretch(1); layout.addLayout(rr); layout.addWidget(self.redline_feedback)

    def _build_results_tab(self, tab: QWidget) -> None:
        layout=QVBoxLayout(tab); layout.addWidget(self.make_hero('Review','Review','Summary, camera results, and manual sphere assist.'))
        self.results_banner=QLabel(); self.results_banner.setWordWrap(True); layout.addWidget(self.results_banner)
        self.results_summary=QTextBrowser(); self.results_summary.setMinimumHeight(260); layout.addWidget(self.results_summary)
        buttons=QHBoxLayout();
        self.open_html_button=QPushButton('Open HTML'); self.open_html_button.clicked.connect(self._open_report_html)
        self.open_pdf_button=QPushButton('Open PDF'); self.open_pdf_button.clicked.connect(self._open_report_pdf)
        self.open_output_button=QPushButton('Open Output Folder'); self.open_output_button.clicked.connect(self._open_output_folder)
        self.refresh_results_button=QPushButton('Open Diagnostics'); self.refresh_results_button.clicked.connect(self._refresh_artifacts)
        [buttons.addWidget(w) for w in [self.open_html_button,self.open_pdf_button,self.open_output_button,self.refresh_results_button]]; buttons.addStretch(1); layout.addLayout(buttons)

    def _build_apply_tab(self, tab: QWidget) -> None:
        layout=QVBoxLayout(tab); layout.addWidget(self.make_hero('Push Looks To Camera','Push Looks To Camera','Network readiness, payload readiness, and push actions.'))
        self.apply_banner=QLabel(); self.apply_banner.setWordWrap(True); layout.addWidget(self.apply_banner)
        self.apply_summary=QTextBrowser(); self.apply_summary.setMinimumHeight(260); layout.addWidget(self.apply_summary)
        row=QHBoxLayout()
        self.read_current_values_button=QPushButton('Read Current Values'); self.read_current_values_button.clicked.connect(self._read_current_camera_values)
        self.dry_run_push_button=QPushButton('Dry Run Push'); self.dry_run_push_button.clicked.connect(self._apply_dry_run)
        self.live_push_button=QPushButton('Push Looks'); self.live_push_button.clicked.connect(self._apply_live)
        self.verify_push_button=QPushButton('Verify Last Push'); self.verify_push_button.clicked.connect(self._verify_last_push)
        [row.addWidget(w) for w in [self.read_current_values_button,self.dry_run_push_button,self.live_push_button,self.verify_push_button]]; row.addStretch(1); layout.addLayout(row)

    def make_field_row(self, label:str, widget:QWidget) -> QWidget:
        w=self.make_card(); l=QVBoxLayout(w); l.addWidget(QLabel(f"<span style='font-size:11px;font-weight:700;letter-spacing:1px'>{label.upper()}</span>")); l.addWidget(widget); return w

    def _wrap_layout(self, layout) -> QWidget:  # type: ignore[no-untyped-def]
        widget=QWidget(); widget.setLayout(layout); return widget

    def _path_row(self, line_edit: QLineEdit, callback, *, button_text: str = 'Browse') -> QHBoxLayout:  # type: ignore[no-untyped-def]
        row=QHBoxLayout(); row.setContentsMargins(0,0,0,0); row.addWidget(line_edit,1); b=QPushButton(button_text); b.clicked.connect(callback); row.addWidget(b); return row

    def _bind_signals(self) -> None:
        self.source_mode.currentIndexChanged.connect(self._refresh_source_mode_visibility)
        self.input_path.textChanged.connect(self._refresh_button_state)
        self.output_path.textChanged.connect(self._refresh_button_state)
        self.ftps_reel.textChanged.connect(self._refresh_button_state)
        self.ftps_clips.textChanged.connect(self._refresh_button_state)
        if hasattr(self, "reference_clip_id"):
            self.reference_clip_id.textChanged.connect(self._refresh_button_state)
        if self.clip_group_list is not None:
            self.clip_group_list.itemChanged.connect(self._clip_group_selection_changed)

    def _clip_group_selection_changed(self, _item: QListWidgetItem) -> None:
        self._refresh_clip_group_summary()
        self._refresh_button_state()

    def _populate_clip_groups(self, scan: Dict[str, object]) -> None:
        self.last_scan_summary = dict(scan or {})
        if self.clip_group_list is None:
            return
        self.clip_group_list.blockSignals(True)
        self.clip_group_list.clear()
        for group in list(scan.get("clip_groups") or []):
            label = str(group.get("group_id") or "").strip()
            if not label:
                continue
            item = QListWidgetItem(f"{label}  ({int(group.get('clip_count', 0) or 0)} clips)")
            item.setData(Qt.ItemDataRole.UserRole, label)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
            item.setCheckState(Qt.CheckState.Checked)
            self.clip_group_list.addItem(item)
        self.clip_group_list.blockSignals(False)
        self._refresh_clip_group_summary()

    def _selected_clip_groups(self) -> List[str]:
        if self.clip_group_list is None or self.clip_group_list.count() <= 0:
            return []
        selected: List[str] = []
        for index in range(self.clip_group_list.count()):
            item = self.clip_group_list.item(index)
            if item.checkState() == Qt.CheckState.Checked:
                label = str(item.data(Qt.ItemDataRole.UserRole) or "").strip()
                if label:
                    selected.append(label)
        return selected

    def _refresh_clip_group_summary(self) -> None:
        if self.clip_group_summary is None:
            return
        groups = list((self.last_scan_summary or {}).get("clip_groups") or [])
        if not groups:
            if self.current_source_mode == "ftps_camera_pull":
                self.clip_group_summary.setText("Subset groups are determined after local media scan. FTPS planning stays camera-driven until clips are ingested locally.")
            else:
                self.clip_group_summary.setText("Scan a local folder to populate clip groups.")
            return
        selected = self._selected_clip_groups()
        total = len(groups)
        if not selected:
            self.clip_group_summary.setText("No clip groups are currently selected. Choose at least one group before running review.")
            return
        if len(selected) == total:
            self.clip_group_summary.setText(f"All {total} discovered clip groups are selected for review. Deselect groups here to avoid accidental multi-group runs.")
            return
        self.clip_group_summary.setText(f"Selected groups: {', '.join(selected)}. Only these groups will be included when review starts.")

    def _refresh_source_mode_visibility(self) -> None:
        is_ftps = self.current_source_mode == "ftps_camera_pull"
        self.input_path.setEnabled(not is_ftps)
        self.ftps_reel.setEnabled(is_ftps)
        self.ftps_clips.setEnabled(is_ftps)
        self.ftps_cameras.setEnabled(is_ftps)
        self.discover_button.setEnabled(is_ftps)
        self.download_button.setEnabled(is_ftps)
        self.download_process_button.setEnabled(is_ftps)
        self.retry_failed_button.setEnabled(is_ftps)
        self.scan_button.setText("Plan Request" if is_ftps else "Scan Folder")
        if self.clip_group_list is not None:
            self.clip_group_list.setEnabled(not is_ftps)
        self._refresh_clip_group_summary()
        self._refresh_button_state()

    @property
    def current_source_mode(self) -> str:
        return normalize_source_mode(str(self.source_mode.currentData() or "local_folder"))

    def _selected_strategies(self) -> List[str]:
        return [name for name, box in self.strategy_checks.items() if box.isChecked()]

    def _append_log(self, text: str) -> None:
        if not text:
            return
        self.log_output.appendPlainText(text.rstrip("\n"))
        scrollbar = self.log_output.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _show_error(self, title: str, message: str) -> None:
        QMessageBox.critical(self, title, message)

    def _show_info(self, title: str, message: str) -> None:
        QMessageBox.information(self, title, message)

    def _browse_input(self) -> None:
        selected = pick_existing_directory(title="Select Calibration Folder", directory=self.input_path.text().strip())
        if selected:
            self.input_path.setText(selected)

    def _browse_output(self) -> None:
        selected = pick_existing_directory(title="Select Output Folder", directory=self.output_path.text().strip())
        if selected:
            self.output_path.setText(selected)

    def _browse_ingest_root(self) -> None:
        selected = pick_existing_directory(title="Select Local Ingest Cache Root", directory=self.local_ingest_root.text().strip())
        if selected:
            self.local_ingest_root.setText(selected)

    def _browse_after_apply(self) -> None:
        selected = pick_existing_directory(title="Select After-Apply Review Folder", directory=self.verification_after_path.text().strip())
        if selected:
            self.verification_after_path.setText(selected)

    def _browse_preview_lut(self) -> None:
        selected = pick_existing_file(
            title="Select Monitoring LUT",
            directory=str(Path(self.preview_lut.text().strip() or "~").expanduser()),
            filter="Cube LUT (*.cube);;All Files (*)",
        )
        if selected:
            self.preview_lut.setText(selected)

    def _browse_redline(self) -> None:
        try:
            selected = pick_existing_file(
                title="Select REDLine Executable",
                directory=str(Path(self.redline_path.text().strip() or "/Applications").expanduser()),
                filter="Executables (*)",
            )
        except Exception as exc:
            self.redline_feedback.setText(f"REDLine picker failed: {exc}")
            return
        if selected:
            self.redline_path.setText(selected)
            self.redline_feedback.setText("Selected path is ready to save and will persist for the next launch.")

    def _reload_redline(self) -> None:
        self.redline_path.setText(redline_configured_path())
        self._refresh_runtime_health()

    def _save_redline(self) -> None:
        payload = persist_redline_configured_path(self.redline_path.text().strip())
        if payload.get("ok"):
            self.redline_path.setText(redline_configured_path())
            self.redline_feedback.setText(str(payload.get("message") or "Saved REDLine configuration."))
        else:
            self.redline_feedback.setText(str(payload.get("error") or "Unable to save REDLine configuration."))
        self._refresh_runtime_health()

    def _scan_source(self) -> None:
        if self.current_source_mode == "ftps_camera_pull":
            self._populate_clip_groups({})
            try:
                planned = plan_ftps_request(
                    reel_identifier=self.ftps_reel.text().strip(),
                    clip_spec=self.ftps_clips.text().strip(),
                    requested_cameras=[item.strip() for item in self.ftps_cameras.text().split(",") if item.strip()],
                )
            except Exception as exc:
                if self.source_summary is not None:
                    self.source_summary.setHtml(
                        f"<h3>FTPS planning failed</h3><p>{html.escape(str(exc))}</p>"
                    )
                self.status_bar.showMessage("FTPS planning failed", 5000)
                return
            camera_rows = "".join(
                f"<li><strong>{html.escape(str(camera))}</strong> · {html.escape(str(ip))}</li>"
                for camera, ip in dict(planned.get("camera_ips") or {}).items()
            )
            if self.source_summary is not None:
                self.source_summary.setHtml(
                    "<h3>FTPS request planned</h3>"
                    f"<p><strong>Source mode:</strong> {html.escape(source_mode_label('ftps_camera_pull'))}<br>"
                    f"<strong>Reel:</strong> {html.escape(str(planned.get('reel_identifier') or ''))}<br>"
                    f"<strong>Clip spec:</strong> {html.escape(str(planned.get('clip_spec') or ''))}<br>"
                    f"<strong>Camera count:</strong> {int(planned.get('camera_count') or 0)}</p>"
                    f"<p><strong>Planned cameras</strong></p><ul>{camera_rows or '<li>No camera/IP pairs resolved yet.</li>'}</ul>"
                )
            self.status_bar.showMessage("FTPS request planned", 4000)
            return

        summary = scan_calibration_sources(self.input_path.text().strip())
        self._populate_clip_groups(summary)
        warning = str(summary.get("warning") or "").strip()
        clip_ids = list(summary.get("sample_clip_ids") or [])
        clip_groups = list(summary.get("clip_groups") or [])
        clips_markup = "".join(f"<li>{html.escape(str(clip_id))}</li>" for clip_id in clip_ids)
        groups_markup = "".join(
            f"<li>Group <strong>{html.escape(str(group.get('group_id') or ''))}</strong> · {int(group.get('clip_count', 0) or 0)} clip(s)</li>"
            for group in clip_groups[:10]
        )
        if self.source_summary is not None:
            self.source_summary.setHtml(
                "<h3>Local source summary</h3>"
                f"<p><strong>Input path:</strong> {html.escape(str(summary.get('input_path') or ''))}<br>"
                f"<strong>Exists:</strong> {'Yes' if summary.get('exists') else 'No'}<br>"
                f"<strong>Clip count:</strong> {int(summary.get('clip_count') or 0)}<br>"
                f"<strong>Clip groups:</strong> {len(clip_groups)}<br>"
                f"<strong>RDC count:</strong> {int(summary.get('rdc_count') or 0)}<br>"
                f"<strong>R3D count:</strong> {int(summary.get('r3d_count') or 0)}</p>"
                + (f"<p><strong>Warning:</strong> {html.escape(warning)}</p>" if warning else "")
                + (
                    f"<p><strong>Detected groups</strong></p><ul>{groups_markup}</ul>"
                    if groups_markup
                    else ""
                )
                + (
                    f"<p><strong>Sample clips</strong></p><ul>{clips_markup}</ul>"
                    if clips_markup
                    else "<p>No clips are currently available for review from this path.</p>"
                )
            )
        self.status_bar.showMessage("Local folder scanned", 3000)

    def _require_output_path(self) -> Optional[Path]:
        output_text = self.output_path.text().strip()
        if not output_text:
            self._show_error("Missing Output Folder", "Select an output folder first.")
            return None
        return Path(output_text).expanduser().resolve()

    def _review_command_from_form(self) -> List[str]:
        args = [*runtime_cli_prefix(), "review-calibration"]
        if self.current_source_mode == "local_folder":
            args.append(self.input_path.text().strip())
        args.extend(
            [
                "--out",
                str(self._require_output_path()),
                "--source-mode",
                self.current_source_mode,
                "--backend",
                self.backend.currentText(),
                "--target-type",
                self.target_type.currentText(),
                "--processing-mode",
                self.processing_mode.currentText(),
                "--matching-domain",
                str(self.matching_domain.currentData() or "perceptual"),
                "--review-mode",
                str(self.review_mode.currentData()),
                "--report-focus",
                str(self.report_focus.currentData()),
                "--preview-mode",
                str(self.preview_mode.currentData() or "monitoring"),
                "--preview-still-format",
                str(self.preview_still_format.currentData() or DEFAULT_PREVIEW_STILL_FORMAT),
                "--artifact-mode",
                str(self.artifact_mode.currentData() or DEFAULT_ARTIFACT_MODE),
            ]
        )
        if self.focus_validation_check.isChecked():
            args.append("--focus-validation")
        if self.current_source_mode == "ftps_camera_pull":
            args.extend(["--ftps-reel", self.ftps_reel.text().strip(), "--ftps-clips", self.ftps_clips.text().strip()])
            for token in [item.strip() for item in self.ftps_cameras.text().split(",") if item.strip()]:
                args.extend(["--ftps-camera", token])
            if self.local_ingest_root.text().strip():
                args.extend(["--ftps-local-root", self.local_ingest_root.text().strip()])
        if self.preview_lut.text().strip():
            args.extend(["--preview-lut", self.preview_lut.text().strip()])
        if self.reference_clip_id.text().strip():
            args.extend(["--reference-clip-id", self.reference_clip_id.text().strip()])
        if self.hero_clip_id.text().strip():
            args.extend(["--hero-clip-id", self.hero_clip_id.text().strip()])
        for name in self._selected_strategies():
            args.extend(["--target-strategy", normalize_target_strategy_name(name)])
        if self.current_source_mode == "local_folder":
            for group_id in self._selected_clip_groups():
                args.extend(["--clip-group", group_id])
        roi_values = [self.roi_x.text().strip(), self.roi_y.text().strip(), self.roi_w.text().strip(), self.roi_h.text().strip()]
        if any(roi_values):
            if not all(roi_values):
                raise ValueError("ROI x/y/w/h must all be provided together.")
            args.extend(["--roi-x", roi_values[0], "--roi-y", roi_values[1], "--roi-w", roi_values[2], "--roi-h", roi_values[3]])
        return args

    def _ftps_ingest_root(self) -> Path:
        if self.local_ingest_root.text().strip():
            return Path(self.local_ingest_root.text().strip()).expanduser().resolve()
        output_root = self._require_output_path()
        assert output_root is not None
        return output_root / "ingest"

    def _start_process(self, command: List[str], *, task_kind: str, output_root: Optional[Path] = None) -> None:
        if self.process.state() != QProcess.ProcessState.NotRunning:
            self._show_error("Task Already Running", "Wait for the current task to finish before starting another one.")
            return
        env = runtime_subprocess_env(self.repo_root, invocation_source="desktop_ui")
        process_env = QProcessEnvironment()
        for key, value in env.items():
            process_env.insert(str(key), str(value))
        self.process.setProcessEnvironment(process_env)
        self.process.setProgram(command[0])
        self.process.setArguments(command[1:])
        self.last_task_kind = task_kind
        self.last_task_output_root = output_root
        self.active_output_root = output_root or self.active_output_root
        self.log_output.clear()
        self._append_log(f"$ {' '.join(command)}")
        self.process.start()
        if not self.process.waitForStarted(3000):
            self._show_error("Failed to Start Task", f"Unable to start command:\n{' '.join(command)}")
            return
        self.status_bar.showMessage(f"Running {task_kind}…")
        self.progress_timer.start()
        self._refresh_button_state()

    def _run_review(self) -> None:
        if not self._selected_strategies():
            self._show_error("Missing Strategy", "Select at least one target strategy.")
            return
        if self.current_source_mode == "local_folder":
            if not self.input_path.text().strip():
                self._show_error("Missing Calibration Folder", "Select a calibration folder.")
                return
            summary = scan_calibration_sources(self.input_path.text().strip())
            if int(summary.get("clip_count", 0) or 0) <= 0:
                self._show_error("No RED Media Found", str(summary.get("warning") or "No RED clips were found."))
                return
            if list(summary.get("clip_groups") or []) and not self._selected_clip_groups():
                self._show_error("No Clip Group Selected", "Choose at least one discovered clip group before running review.")
                return
        output_root = self._require_output_path()
        if output_root is None:
            return
        try:
            command = self._review_command_from_form()
        except Exception as exc:
            self._show_error("Invalid Review Settings", str(exc))
            return
        self._start_process(command, task_kind="review", output_root=output_root)

    def _build_ingest_command(self, action: str) -> List[str]:
        ingest_root = self._ftps_ingest_root()
        args = [
            *runtime_cli_prefix(),
            "ingest-ftps",
            "--action",
            action,
            "--out",
            str(ingest_root),
            "--ftps-reel",
            self.ftps_reel.text().strip(),
            "--ftps-clips",
            self.ftps_clips.text().strip(),
        ]
        for token in [item.strip() for item in self.ftps_cameras.text().split(",") if item.strip()]:
            args.extend(["--ftps-camera", token])
        if action == "retry-failed" and self.last_manifest_path is not None:
            args.extend(["--manifest-path", str(self.last_manifest_path)])
        return args

    def _discover_ftps(self) -> None:
        if self.current_source_mode != "ftps_camera_pull":
            return
        self._start_process(self._build_ingest_command("discover"), task_kind="ftps_discover", output_root=self._ftps_ingest_root())

    def _download_ftps(self) -> None:
        if self.current_source_mode != "ftps_camera_pull":
            return
        self._start_process(self._build_ingest_command("download"), task_kind="ftps_download", output_root=self._ftps_ingest_root())

    def _download_then_process(self) -> None:
        output_root = self._require_output_path()
        if output_root is None:
            return
        args = [
            *runtime_cli_prefix(),
            "ftps-download-process",
            "--out",
            str(output_root),
            "--ftps-reel",
            self.ftps_reel.text().strip(),
            "--ftps-clips",
            self.ftps_clips.text().strip(),
            "--backend",
            self.backend.currentText(),
            "--target-type",
            self.target_type.currentText(),
            "--processing-mode",
            self.processing_mode.currentText(),
            "--matching-domain",
            str(self.matching_domain.currentData() or "perceptual"),
            "--review-mode",
            str(self.review_mode.currentData()),
            "--report-focus",
            str(self.report_focus.currentData()),
            "--preview-mode",
            str(self.preview_mode.currentData() or "monitoring"),
            "--preview-still-format",
            str(self.preview_still_format.currentData() or DEFAULT_PREVIEW_STILL_FORMAT),
            "--artifact-mode",
            str(self.artifact_mode.currentData() or DEFAULT_ARTIFACT_MODE),
        ]
        if self.focus_validation_check.isChecked():
            args.append("--focus-validation")
        for token in [item.strip() for item in self.ftps_cameras.text().split(",") if item.strip()]:
            args.extend(["--ftps-camera", token])
        if self.local_ingest_root.text().strip():
            args.extend(["--ftps-local-root", self.local_ingest_root.text().strip()])
        for name in self._selected_strategies():
            args.extend(["--target-strategy", normalize_target_strategy_name(name)])
        if self.reference_clip_id.text().strip():
            args.extend(["--reference-clip-id", self.reference_clip_id.text().strip()])
        if self.hero_clip_id.text().strip():
            args.extend(["--hero-clip-id", self.hero_clip_id.text().strip()])
        if self.preview_lut.text().strip():
            args.extend(["--preview-lut", self.preview_lut.text().strip()])
        if self.current_source_mode == "local_folder":
            for group_id in self._selected_clip_groups():
                args.extend(["--clip-group", group_id])
        self._start_process(args, task_kind="ftps_download_process", output_root=output_root)

    def _retry_failed_ftps(self) -> None:
        if self.last_manifest_path is None:
            self._show_error("No Ingest Manifest", "Run discover or download first so there is a manifest to retry.")
            return
        self._start_process(self._build_ingest_command("retry-failed"), task_kind="ftps_retry", output_root=self._ftps_ingest_root())

    def _approve_master(self) -> None:
        output_root = self._require_output_path()
        if output_root is None:
            return
        selected = self._selected_strategies()
        strategy = selected[0] if selected else "median"
        command = build_approve_command(
            repo_root=self.repo_root,
            analysis_dir=str(output_root),
            target_strategy=strategy,
            reference_clip_id=self.reference_clip_id.text().strip() or None,
        )
        self._start_process(command, task_kind="approve_master", output_root=output_root)

    def _clear_preview_cache(self) -> None:
        output_root = self._require_output_path()
        if output_root is None:
            return
        command = build_clear_cache_command(repo_root=self.repo_root, analysis_dir=str(output_root))
        self._start_process(command, task_kind="clear_preview_cache", output_root=output_root)

    def _read_current_camera_values(self) -> None:
        output_root = self._require_output_path()
        if output_root is None:
            return
        commit_payload = review_commit_payload_path_for(output_root)
        if not commit_payload.exists():
            self._show_error("Missing Commit Payload", f"No commit payload found at:\n{commit_payload}")
            return
        report_path = review_validation_path_for(output_root).parent / "rcp2_camera_state_report.json"
        command = [*runtime_cli_prefix(), "verify-camera-state", str(commit_payload), "--live-read", "--out", str(report_path)]
        self._start_process(command, task_kind="read_camera_state", output_root=output_root)

    def _apply_dry_run(self) -> None:
        output_root = self._require_output_path()
        if output_root is None:
            return
        payload_path = review_commit_payload_path_for(output_root)
        if not payload_path.exists():
            self._show_error("Missing Commit Payload", f"No commit payload found at:\n{payload_path}")
            return
        report_path = review_validation_path_for(output_root).parent / "rcp2_apply_report.json"
        self._start_process(
            [*runtime_cli_prefix(), "apply-calibration", str(payload_path), "--dry-run", "--out", str(report_path)],
            task_kind="apply_dry_run",
            output_root=output_root,
        )

    def _apply_live(self) -> None:
        output_root = self._require_output_path()
        if output_root is None:
            return
        payload_path = review_commit_payload_path_for(output_root)
        if not payload_path.exists():
            self._show_error("Missing Commit Payload", f"No commit payload found at:\n{payload_path}")
            return
        report_path = review_validation_path_for(output_root).parent / "rcp2_apply_report_live.json"
        self._start_process(
            [*runtime_cli_prefix(), "apply-calibration", str(payload_path), "--live", "--out", str(report_path)],
            task_kind="apply_live",
            output_root=output_root,
        )

    def _verify_last_push(self) -> None:
        output_root = self._require_output_path()
        if output_root is None:
            return
        payload_path = review_commit_payload_path_for(output_root)
        if not payload_path.exists():
            self._show_error("Missing Commit Payload", f"No commit payload found at:\n{payload_path}")
            return
        report_path = review_validation_path_for(output_root).parent / "rcp2_verification_report.json"
        self._start_process(
            [*runtime_cli_prefix(), "verify-camera-state", str(payload_path), "--out", str(report_path)],
            task_kind="verify_push",
            output_root=output_root,
        )

    def _drain_process_output(self) -> None:
        data = bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace")
        if data:
            self._append_log(data.rstrip("\n"))

    def _task_finished(self, *_args) -> None:
        self.progress_timer.stop()
        exit_code = self.process.exitCode()
        self.status_bar.showMessage(f"{self.last_task_kind or 'task'} finished with exit code {exit_code}", 7000)
        if self.last_task_output_root is not None:
            self.active_output_root = self.last_task_output_root
        self._refresh_runtime_health()
        self._refresh_artifacts()
        self._refresh_apply_surface()
        if exit_code == 0 and self.last_task_kind in {"review", "ftps_download_process"}:
            self.tabs.setCurrentWidget(self.results_tab)
        self._refresh_button_state()

    def _refresh_progress_state(self) -> None:
        output_root = self.last_task_output_root or self._maybe_output_root()
        if output_root is None:
            return
        progress_path = review_progress_path_for(output_root)
        if progress_path.exists():
            payload = load_review_progress(progress_path)
            extra = dict(payload.get("extra") or {})
            summary_bits = [
                f"Phase: {payload.get('phase') or 'unknown'}",
                f"Status: {payload.get('status') or 'unknown'}",
            ]
            if extra.get("processed_clips") is not None and extra.get("clip_count") is not None:
                summary_bits.append(f"Clips: {extra.get('processed_clips')}/{extra.get('clip_count')}")
            if extra.get("validation_status"):
                summary_bits.append(f"Validation: {extra.get('validation_status')}")
            message = str(payload.get("message") or "").strip()
            if message:
                summary_bits.append(message)
            self.status_bar.showMessage(" | ".join(summary_bits))
            if self.workflow_banner is not None:
                self.workflow_banner.setText(" | ".join(summary_bits))

    def _maybe_output_root(self) -> Optional[Path]:
        output_text = self.output_path.text().strip()
        if output_text:
            candidate = Path(output_text).expanduser().resolve()
            discovered = _discover_review_output_root(candidate)
            if discovered is not None:
                return discovered
            return candidate if candidate.exists() else self.active_output_root
        if self.active_output_root is None:
            return None
        discovered = _discover_review_output_root(self.active_output_root)
        return discovered or self.active_output_root

    def _refresh_runtime_health(self) -> None:
        payload = runtime_health_payload()
        red_sdk = dict(payload.get("red_sdk_runtime") or {})
        redline = dict(payload.get("redline_tool") or {})
        self._set_badge_state(
            self.runtime_chip,
            f"RED SDK Runtime: {'Ready' if red_sdk.get('ready') else 'Check setup'}",
            "good" if red_sdk.get("ready") else "bad",
        )
        self._set_badge_state(
            self.redline_chip,
            f"REDLine: {'Ready' if redline.get('ready') else 'Needs path'}",
            "good" if redline.get("ready") else "warn",
        )
        self._set_badge_state(
            self.pdf_chip,
            f"HTML/PDF: {'Ready' if payload.get('html_pdf_ready') else 'Blocked'}",
            "good" if payload.get("html_pdf_ready") else "warn",
        )
        self._set_badge_state(
            self.context_chip,
            "Packaged App" if payload.get("frozen_app") else "Dev Runtime",
            "info",
        )
        if self.workflow_banner is not None:
            self.workflow_banner.setText(
                "Review: source, subset, run. Results: recommendation, artifacts, and gray-target consistency. Apply / Verify: staged payloads only after a valid completed run exists."
            )
        runtime_html = (
            "<h3>Runtime status</h3>"
            f"<p><strong>Interpreter:</strong> {html.escape(str(payload.get('interpreter') or ''))}<br>"
            f"<strong>Runtime context:</strong> {'Packaged app' if payload.get('frozen_app') else 'Dev / Python'}<br>"
            f"<strong>Virtual env:</strong> {html.escape(str(payload.get('virtual_env') or 'n/a'))}<br>"
            f"<strong>DYLD fallback:</strong> {html.escape(str(payload.get('dyld_fallback_library_path') or '<unset>'))} ({html.escape(str(payload.get('dyld_fallback_source') or 'unknown'))})</p>"
            "<h3>RED SDK runtime</h3>"
            f"<p><strong>Status:</strong> {'Ready' if red_sdk.get('ready') else 'Missing'}<br>"
            f"<strong>Source:</strong> {html.escape(str(red_sdk.get('source') or 'unknown'))}<br>"
            f"<strong>Redistributable dir:</strong> {html.escape(str(red_sdk.get('redistributable_dir') or '<unset>'))}<br>"
            f"<strong>Error:</strong> {html.escape(str(red_sdk.get('error') or 'none'))}</p>"
            "<h3>REDLine tool</h3>"
            f"<p><strong>Status:</strong> {'Ready' if redline.get('ready') else 'Missing'}<br>"
            f"<strong>Source:</strong> {html.escape(str(redline.get('source') or 'unknown'))}<br>"
            f"<strong>Desktop config:</strong> {html.escape(str(redline.get('desktop_config_path') or '<unset>'))}<br>"
            f"<strong>Configured path:</strong> {html.escape(str(redline.get('configured_path') or '<unset>'))}<br>"
            f"<strong>Resolved path:</strong> {html.escape(str(redline.get('resolved_path') or '<unset>'))}<br>"
            f"<strong>Error:</strong> {html.escape(str(redline.get('error') or 'none'))}</p>"
            "<h3>HTML / PDF export</h3>"
            f"<p><strong>Ready:</strong> {html.escape(str(payload.get('html_pdf_ready')))}<br>"
            f"<strong>WeasyPrint importable:</strong> {html.escape(str(payload.get('weasyprint_importable')))}<br>"
            f"<strong>WeasyPrint error:</strong> {html.escape(str(payload.get('weasyprint_error') or 'none'))}</p>"
        )
        if self.runtime_health is not None:
            self.runtime_health.setHtml(runtime_html)
        self.redline_feedback.setText(
            f"{'Ready' if redline.get('ready') else 'Missing'} | source={redline.get('source') or 'unknown'} | path={redline.get('resolved_path') or redline.get('configured_path') or '<unset>'}"
        )

    def _load_operator_surface_context(self, output_root: Path) -> Dict[str, object]:
        report_root = review_validation_path_for(output_root).parent
        review_validation = _artifact_json(review_validation_path_for(output_root))
        contact_sheet_payload = _artifact_json(report_root / "contact_sheet.json")
        review_package_payload = _artifact_json(report_root / "review_package.json")
        review_payload = dict(contact_sheet_payload or {})
        review_payload.update(dict(review_package_payload or {}))
        for key in (
            "recommended_strategy",
            "hero_recommendation",
            "white_balance_model",
            "run_assessment",
            "gray_target_consistency",
            "operator_recommendation",
        ):
            if review_payload.get(key) is None and contact_sheet_payload is not None:
                review_payload[key] = contact_sheet_payload.get(key)
        commit_payload = _artifact_json(review_commit_payload_path_for(output_root))
        camera_state_report = _artifact_json(report_root / "rcp2_camera_state_report.json")
        verification_report = _artifact_json(report_root / "rcp2_verification_report.json")
        scientific_validation = _artifact_json(report_root / "scientific_validation.json")
        corrected_validation = _artifact_json(report_root / "corrected_residual_validation.json")
        apply_reports = sorted(report_root.glob("rcp2_apply_report*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
        apply_report = _artifact_json(apply_reports[0]) if apply_reports else None
        post_apply_report = _artifact_json(post_apply_verification_path_for(output_root))
        from .web_app import (
            _build_operator_surfaces,
            _render_apply_surface,
            _render_commit_table,
            _render_decision_surface,
            _render_push_surface,
            _render_verification_surface,
        )

        source_mode_value = str((review_payload or {}).get("source_mode") or self.current_source_mode)
        surfaces = _build_operator_surfaces(
            review_validation=review_validation,
            review_payload=review_payload,
            commit_payload=commit_payload,
            camera_state_report=camera_state_report,
            apply_report=apply_report,
            writeback_verification_report=verification_report,
            post_apply_report=post_apply_report,
            source_mode_label_text=source_mode_label(source_mode_value),
        )
        return {
            "report_root": report_root,
            "review_validation": review_validation,
            "review_payload": review_payload,
            "contact_sheet_payload": contact_sheet_payload,
            "commit_payload": commit_payload,
            "camera_state_report": camera_state_report,
            "apply_report": apply_report,
            "verification_report": verification_report,
            "post_apply_report": post_apply_report,
            "scientific_validation": scientific_validation,
            "corrected_validation": corrected_validation,
            "operator_surfaces": surfaces,
            "decision_surface_html": _render_decision_surface(surfaces),
            "commit_table_html": _render_commit_table(surfaces),
            "push_surface_html": _render_push_surface(surfaces),
            "apply_surface_html": _render_apply_surface(dict(surfaces.get("apply_surface") or {})),
            "verification_surface_html": _render_verification_surface(dict(surfaces.get("verification_surface") or {})),
        }

    def _refresh_artifacts(self) -> None:
        output_root = self._maybe_output_root()
        if output_root is None:
            if self.results_banner is not None:
                self.results_banner.setText("No output folder is selected yet. Choose an output path in the Review tab, then run a review or point the app at an existing run folder to inspect artifacts.")
            if self.results_summary is not None:
                self.results_summary.setHtml(
                    _desktop_surface_html(
                        "<div class='surface-banner'><h2>No results yet</h2><p>Select an output folder, scan media, choose clip groups, then run review to populate calibration recommendation, white-balance model, scientific validation, and delivery artifacts.</p></div>"
                    )
                )
            self._refresh_apply_surface()
            self._refresh_button_state()
            return
        html_path = review_html_path_for(output_root)
        pdf_path = review_pdf_path_for(output_root)
        validation_path = review_validation_path_for(output_root)
        commit_payload_path = review_commit_payload_path_for(output_root)
        context = self._load_operator_surface_context(output_root)
        validation = dict(context.get("review_validation") or {})
        scientific = dict(context.get("scientific_validation") or {})
        scientific_state = str(scientific.get("replay_integrity_state") or scientific.get("status") or "pending")
        scientific_tone = "success" if scientific_state == "fully_reconciled" else "warning" if scientific_state == "blocked_asset_mismatch" else "danger" if scientific else "pending"
        if self.results_banner is not None:
            if validation:
                self.results_banner.setText(
                    f"Latest review status: {validation.get('status') or 'unknown'}. Bound run root: {output_root}. Results below prioritize recommendation, attention cameras, and artifact actions."
                )
            else:
                self.results_banner.setText(
                    "This folder does not currently resolve to a completed review package. Run Review first, or point the app at a folder that contains a finished run."
                )
        details: List[str] = [
            "<div class='legend-bar'>"
            "<span class='legend-pill'>Green = ready / stable</span>"
            "<span class='legend-pill'>Amber = review / caution</span>"
            "<span class='legend-pill'>Red = outlier / blocked</span>"
            "<span class='legend-pill'>↑ exposure lift / increase</span>"
            "<span class='legend-pill'>↓ exposure reduction</span>"
            "</div>",
            _build_desktop_results_summary(context, output_root),
            str(context.get("decision_surface_html") or ""),
            "<div class='surface-grid'>",
            "<div class='summary-panel' style='width:100%;margin-right:0;'>"
            "<div class='surface-heading-row'><h3>Per-Camera Commit Table</h3></div>"
            "<p class='surface-copy'>These rows are the staged camera targets that back the report recommendation and later Apply / Verify workflow.</p>"
            f"{str(context.get('commit_table_html') or '')}"
            "</div>",
            "</div>",
            (
                "<div class='surface-banner'>"
                "<h2>Scientific Validation</h2>"
                f"<p><strong>Status:</strong> <span class='surface-badge {html.escape(scientific_tone)}'>{html.escape(scientific_state.replace('_', ' '))}</span><br>"
                f"<strong>Artifact:</strong> {html.escape(str((context.get('report_root') or Path('.')) / 'scientific_validation.json'))}</p>"
                "</div>"
                if scientific
                else ""
            ),
            "<div class='results-actions'>"
            f"<strong>Output root:</strong> {html.escape(str(output_root))}<br>"
            f"<strong>Report HTML:</strong> {html.escape(str(html_path if html_path.exists() else '<missing>'))}<br>"
            f"<strong>Report PDF:</strong> {html.escape(str(pdf_path if pdf_path.exists() else '<missing>'))}<br>"
            f"<strong>Review validation:</strong> {html.escape(str(validation_path if validation_path.exists() else '<missing>'))}<br>"
            f"<strong>Commit payload:</strong> {html.escape(str(commit_payload_path if commit_payload_path.exists() else '<missing>'))}"
            "</div>",
        ]
        if self.results_summary is not None:
            self.results_summary.setHtml(_desktop_surface_html(*details))
        self._refresh_apply_surface()
        self._refresh_button_state()

    def _refresh_apply_surface(self) -> None:
        output_root = self._maybe_output_root()
        if output_root is None:
            if self.apply_banner is not None:
                self.apply_banner.setText("Apply / Verify becomes meaningful only after a review package has produced a commit payload. Select an output folder first, then run review or open an existing completed run.")
            if self.apply_summary is not None:
                self.apply_summary.setHtml(
                    _desktop_surface_html(
                        "<div class='surface-banner'><h2>No apply state yet</h2><p>A completed review run will populate commit payload, current camera reads, writeback reports, and post-apply verification. Start with Results, then come back here when the payload is ready.</p></div>"
                    )
                )
            return
        commit_payload_path = review_commit_payload_path_for(output_root)
        context = self._load_operator_surface_context(output_root)
        camera_state = dict(context.get("camera_state_report") or {})
        verification = dict(context.get("verification_report") or {})
        post_apply = dict(context.get("post_apply_report") or {})
        if self.apply_banner is not None:
            if commit_payload_path.exists():
                self.apply_banner.setText("A commit payload is available. Start with Read Current Camera Values or Dry Run Push, then use Live Push only when the verification path is understood and operator-approved.")
            else:
                self.apply_banner.setText("No commit payload is available yet. Run a review first so the app can stage safe camera targets before any apply or verification action.")
        details: List[str] = [
            "<div class='legend-bar'>"
            "<span class='legend-pill'>1. Read current camera values</span>"
            "<span class='legend-pill'>2. Dry run push</span>"
            "<span class='legend-pill'>3. Live push only when approved</span>"
            "<span class='legend-pill'>4. Verify last push</span>"
            "</div>",
            "<div class='summary-panel' style='width:100%;margin-right:0;'>"
            "<div class='surface-heading-row'><h3>Camera Push Plan</h3></div>"
            "<p class='surface-copy'>This surface compares the staged review payload to current camera state and later writeback verification. It is intentionally operational, not decorative.</p>"
            f"{str(context.get('push_surface_html') or '')}"
            "</div>",
            (
                "<div class='summary-panel' style='width:100%;margin-right:0;'>"
                "<div class='surface-heading-row'><h3>Apply Report</h3></div>"
                f"{str(context.get('apply_surface_html') or '')}"
                "</div>"
                if context.get("apply_report")
                else ""
            ),
            (
                "<div class='summary-panel' style='width:100%;margin-right:0;'>"
                "<div class='surface-heading-row'><h3>Verification Surface</h3></div>"
                f"{str(context.get('verification_surface_html') or '')}"
                "</div>"
                if verification or post_apply
                else ""
            ),
        ]
        if camera_state and not verification and not post_apply:
            details.append(
                f"<div class='surface-banner'><h2>Current camera state loaded</h2><p>{int(camera_state.get('connected_camera_count') or 0)} of {int(camera_state.get('camera_count') or 0)} cameras responded. Dry Run Push and Verify Last Push will populate the next sections.</p></div>"
            )
        if self.apply_summary is not None:
            self.apply_summary.setHtml(_desktop_surface_html(*details))

    def _refresh_button_state(self) -> None:
        running = self.process.state() != QProcess.ProcessState.NotRunning
        output_root = self._maybe_output_root()
        html_path = review_html_path_for(output_root) if output_root is not None else None
        pdf_path = review_pdf_path_for(output_root) if output_root is not None else None
        commit_path = review_commit_payload_path_for(output_root) if output_root is not None else None
        has_output = bool(self.output_path.text().strip())
        has_local_input = bool(self.input_path.text().strip())
        has_ftps_request = bool(self.ftps_reel.text().strip() and self.ftps_clips.text().strip())
        has_group_selection = True
        if self.current_source_mode == "local_folder" and list((self.last_scan_summary or {}).get("clip_groups") or []):
            has_group_selection = bool(self._selected_clip_groups())
        can_run_review = has_output and (has_ftps_request if self.current_source_mode == "ftps_camera_pull" else has_local_input) and has_group_selection

        if self.run_review_button is not None:
            self.run_review_button.setEnabled(not running and can_run_review)
        if self.approve_button is not None:
            self.approve_button.setEnabled(not running and output_root is not None)
        if self.clear_preview_button is not None:
            self.clear_preview_button.setEnabled(not running and output_root is not None)
        if self.open_html_button is not None:
            self.open_html_button.setEnabled(bool(html_path and html_path.exists()))
        if self.open_pdf_button is not None:
            self.open_pdf_button.setEnabled(bool(pdf_path and pdf_path.exists()))
        if self.open_output_button is not None:
            self.open_output_button.setEnabled(output_root is not None and output_root.exists())
        has_commit = bool(commit_path and commit_path.exists())
        if self.read_current_values_button is not None:
            self.read_current_values_button.setEnabled(not running and has_commit)
        if self.dry_run_push_button is not None:
            self.dry_run_push_button.setEnabled(not running and has_commit)
        if self.live_push_button is not None:
            self.live_push_button.setEnabled(not running and has_commit)
        if self.verify_push_button is not None:
            self.verify_push_button.setEnabled(not running and has_commit)

    def _open_report_html(self) -> None:
        output_root = self._maybe_output_root()
        if output_root is None:
            return
        path = review_html_path_for(output_root)
        if not path.exists():
            self._show_error("Missing HTML Report", f"No report HTML found at:\n{path}")
            return
        _open_local_path(path)

    def _open_report_pdf(self) -> None:
        output_root = self._maybe_output_root()
        if output_root is None:
            return
        path = review_pdf_path_for(output_root)
        if not path.exists():
            self._show_error("Missing PDF Report", f"No report PDF found at:\n{path}")
            return
        _open_local_path(path)

    def _open_output_folder(self) -> None:
        output_root = self._maybe_output_root()
        if output_root is None or not output_root.exists():
            self._show_error("Missing Output Folder", "Select an existing output folder first.")
            return
        _open_local_path(output_root)


def launch_desktop_ui(repo_root: str, *, minimal_mode: bool = False, smoke_exit_ms: int = 0) -> None:
    ensure_runtime_environment()
    app = QApplication.instance()
    owns_app = app is None
    if app is None:
        app = QApplication([sys.argv[0]])
        app.setApplicationName("R3DMatch")
        app.setStyle("Fusion")
        font = app.font()
        if font.pointSize() < 11:
            font.setPointSize(11)
            app.setFont(font)
    _configure_application_palette(app)
    window = R3DMatchDesktopWindow(repo_root=repo_root, minimal_mode=minimal_mode)
    window.show()
    window.raise_()
    window.activateWindow()
    if smoke_exit_ms > 0:
        print(f"desktop_window_title={window.windowTitle()}")
        print(f"desktop_minimal_mode={minimal_mode}")
        QTimer.singleShot(smoke_exit_ms, app.quit)
    if owns_app:
        app.exec()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--check", action="store_true", help="Validate imports without launching the desktop UI")
    parser.add_argument("--self-check", action="store_true", help="Print structural UI section checks")
    parser.add_argument("--minimal", action="store_true", help="Launch the minimal desktop UI")
    args = parser.parse_args()
    if args.check:
        print("R3DMatch desktop UI imports OK")
        return
    if args.self_check:
        for line in run_ui_self_check(args.repo_root, minimal_mode=args.minimal):
            print(line)
        return
    launch_desktop_ui(args.repo_root, minimal_mode=args.minimal)


if __name__ == "__main__":
    main()
