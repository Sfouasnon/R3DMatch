from __future__ import annotations

import argparse
import html
import hashlib
import json
import logging
import os
import re
import shlex
import subprocess
import threading
import time
import uuid
import webbrowser
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from flask import Flask, Response, jsonify, redirect, render_template_string, request, send_file, url_for

from .ftps_ingest import normalize_source_mode, plan_ftps_request, source_mode_label
from .identity import clip_id_from_path, subset_key_from_clip_id
from .matching import discover_clips
from .progress import load_review_progress, review_progress_path_for
from .rcp2_apply import (
    apply_calibration_payload,
    build_camera_verification_report,
    load_apply_targets,
    operator_apply_tolerances,
    read_camera_state,
    summarize_apply_report,
)
from .execution import CANCEL_FILE_ENV, terminate_process_group
from .report import (
    _build_redline_preview_command,
    _detect_redline_capabilities,
    _normalize_preview_settings,
    _resolve_redline_executable,
    normalize_review_mode,
    review_mode_label,
)
from .workflow import (
    matching_domain_label,
    post_apply_verification_path_for,
    resolve_review_output_dir,
    review_payload_path_for,
    review_commit_payload_path_for,
    review_html_path_for,
    review_manifest_path_for,
    review_pdf_path_for,
    review_validation_path_for,
    build_post_apply_verification_from_reviews,
)


DEFAULT_LOGO_PATH = Path(__file__).resolve().parent / "static" / "r3dmatch_logo.png"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 5000
STALL_THRESHOLD_SECONDS = 20.0
ARTIFACT_FRESHNESS_TOLERANCE_SECONDS = 1.0
LOGGER = logging.getLogger(__name__)


PAGE_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>R3DMatch Internal Review</title>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: linear-gradient(180deg, #eef2f8 0%, #f8fafc 100%); color: #111827; margin: 0; }
    .page { max-width: 1440px; margin: 0 auto; padding: 28px 28px 40px 28px; }
    .header { background: linear-gradient(135deg, rgba(255,255,255,0.98) 0%, rgba(241,245,249,0.98) 100%); border: 1px solid #cfd7e3; border-radius: 18px; padding: 22px; margin-bottom: 16px; box-shadow: 0 18px 40px rgba(15, 23, 42, 0.08); }
    .header-row { display: flex; align-items: center; gap: 18px; }
    .header img { max-width: 320px; max-height: 120px; object-fit: contain; }
    .title { font-size: 34px; font-weight: 750; margin: 0; }
    .subtitle { font-size: 14px; color: #475569; margin: 4px 0 0 0; text-transform: uppercase; letter-spacing: 0.08em; }
    .instructions { margin-top: 12px; font-size: 16px; color: #334155; max-width: 720px; }
    .card { background: rgba(255,255,255,0.96); border: 1px solid #d3d9e4; border-radius: 18px; padding: 20px; margin-bottom: 16px; box-shadow: 0 12px 28px rgba(15, 23, 42, 0.05); }
    .section-title { font-size: 22px; font-weight: 800; margin: 0 0 14px 0; letter-spacing: -0.01em; }
    .section-subtitle { margin: -6px 0 14px 0; font-size: 14px; color: #475569; line-height: 1.6; }
    .field { margin-bottom: 12px; min-width: 0; }
    .field label { display: block; font-weight: 700; margin-bottom: 6px; font-size: 13px; text-transform: uppercase; letter-spacing: 0.08em; color: #475569; }
    .field input[type="text"], .field select { width: 100%; box-sizing: border-box; padding: 12px 14px; border: 1px solid #c7ccd4; border-radius: 12px; background: white; color: #0f172a; }
    .row { display: flex; gap: 12px; flex-wrap: wrap; }
    .row > * { flex: 1 1 220px; }
    .checkbox-row { display: flex; gap: 18px; align-items: center; flex-wrap: wrap; }
    .actions { display: flex; gap: 10px; flex-wrap: wrap; }
    button, .link-button { background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); color: white; border: 0; border-radius: 10px; padding: 10px 14px; font-weight: 700; cursor: pointer; text-decoration: none; display: inline-block; box-shadow: 0 8px 16px rgba(15, 23, 42, 0.16); }
    button.secondary, .link-button.secondary { background: linear-gradient(135deg, #475569 0%, #64748b 100%); }
    button:disabled { opacity: 0.5; cursor: not-allowed; }
    .status { padding: 10px 12px; border-radius: 8px; margin-bottom: 12px; }
    .status.error { background: #fef2f2; color: #991b1b; border: 1px solid #fecaca; }
    .status.info { background: #eff6ff; color: #1d4ed8; border: 1px solid #bfdbfe; }
    .summary-box { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 12px; padding: 14px; }
    .clip-list { margin-top: 8px; padding-left: 20px; max-height: 180px; overflow: auto; }
    .selector-list { display: grid; gap: 8px; max-height: 220px; overflow: auto; padding: 8px 0; }
    .selector-item { display: flex; align-items: center; gap: 8px; font-size: 14px; }
    pre.log { background: #0f172a; color: #e5e7eb; border-radius: 8px; padding: 14px; min-height: 220px; overflow: auto; white-space: pre-wrap; }
    .meta { color: #64748b; font-size: 13px; line-height: 1.5; margin-top: 8px; }
    .roi-preview-wrap { margin-top: 14px; }
    .roi-preview-guidance { margin: 0 0 10px 0; font-size: 14px; font-weight: 700; color: #334155; }
    .roi-preview-note { margin-top: 10px; font-size: 13px; color: #64748b; line-height: 1.6; }
    .roi-preview-stage { position: relative; display: inline-block; max-width: 100%; border: 1px solid #cbd5e1; border-radius: 12px; overflow: hidden; background: linear-gradient(180deg, #020617 0%, #111827 100%); box-shadow: inset 0 0 0 1px rgba(255,255,255,0.06); }
    .roi-preview-stage::after { content: "Draw region around gray target"; position: absolute; left: 12px; top: 12px; z-index: 4; padding: 6px 10px; border-radius: 999px; background: rgba(15,23,42,0.72); color: #f8fafc; font-size: 12px; font-weight: 800; letter-spacing: 0.04em; pointer-events: none; box-shadow: 0 4px 14px rgba(15,23,42,0.22); }
    .roi-preview-stage img { display: block; max-width: 100%; height: auto; filter: brightness(1.12) contrast(1.08) saturate(1.02); }
    .roi-overlay { position: absolute; border: 3px solid #f8fafc; box-shadow: 0 0 0 2px rgba(37,99,235,0.95), 0 0 0 9999px rgba(15,23,42,0.14); background: rgba(37,99,235,0.12); pointer-events: none; display: none; }
    .roi-canvas { position: absolute; inset: 0; cursor: crosshair; }
    .progress-track { background: #e5e7eb; height: 14px; border-radius: 999px; overflow: hidden; margin: 10px 0; }
    .progress-fill { background: #2563eb; height: 100%; width: 0%; transition: width 0.2s ease; }
    .stage-list { display: flex; flex-wrap: wrap; gap: 8px; margin: 8px 0 12px 0; padding: 0; list-style: none; }
    .stage-pill { padding: 6px 10px; border-radius: 999px; background: #e5e7eb; color: #374151; font-size: 13px; }
    .stage-pill.active { background: #2563eb; color: white; }
    .summary-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 14px; }
    .summary-panel { border: 1px solid #e2e8f0; border-radius: 14px; padding: 16px; background: #f8fafc; }
    .push-card { border: 1px solid #c7d2fe; background: linear-gradient(180deg, #f8fbff 0%, #eef4ff 100%); box-shadow: 0 14px 30px rgba(37, 99, 235, 0.08); }
    .push-callout { padding: 14px 16px; border-radius: 14px; background: rgba(37, 99, 235, 0.08); border: 1px solid rgba(37, 99, 235, 0.14); color: #1e3a8a; font-size: 14px; line-height: 1.7; margin-bottom: 14px; }
    .compact-summary-list { display: grid; gap: 8px; margin: 0 0 12px 0; padding: 0; list-style: none; }
    .compact-summary-list li { padding: 10px 12px; border-radius: 12px; background: white; border: 1px solid #dbe4ef; font-size: 14px; color: #334155; }
    .delta-positive { color: #166534; font-weight: 800; }
    .delta-negative { color: #b91c1c; font-weight: 800; }
    .delta-neutral { color: #475569; font-weight: 700; }
    .decision-banner { border-radius: 20px; padding: 22px; margin-bottom: 18px; color: white; box-shadow: 0 18px 36px rgba(15, 23, 42, 0.16); }
    .decision-banner.success { background: linear-gradient(135deg, #14532d 0%, #166534 100%); }
    .decision-banner.warning { background: linear-gradient(135deg, #92400e 0%, #b45309 100%); }
    .decision-banner.danger { background: linear-gradient(135deg, #7f1d1d 0%, #b91c1c 100%); }
    .decision-banner.pending { background: linear-gradient(135deg, #1d4ed8 0%, #2563eb 100%); }
    .decision-banner-kicker { font-size: 12px; font-weight: 800; letter-spacing: 0.12em; text-transform: uppercase; opacity: 0.88; }
    .decision-banner-title { margin: 8px 0 6px 0; font-size: 36px; line-height: 1.02; font-weight: 900; letter-spacing: -0.03em; }
    .decision-banner-subtitle { margin: 0; font-size: 16px; line-height: 1.6; max-width: 72ch; color: rgba(255,255,255,0.92); }
    .decision-banner-metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin-top: 16px; }
    .decision-banner-metrics > div { padding: 12px 14px; border-radius: 14px; background: rgba(255,255,255,0.12); border: 1px solid rgba(255,255,255,0.18); }
    .decision-banner .metric-label { color: rgba(255,255,255,0.75); }
    .decision-banner .metric-value { color: white; font-size: 22px; }
    .surface-heading-row { display: flex; justify-content: space-between; gap: 12px; align-items: center; margin-bottom: 10px; flex-wrap: wrap; }
    .surface-heading-row h3 { margin: 0; font-size: 22px; }
    .surface-badge { display: inline-flex; align-items: center; padding: 6px 10px; border-radius: 999px; font-size: 12px; font-weight: 800; letter-spacing: 0.06em; text-transform: uppercase; }
    .surface-badge.success { background: #dcfce7; color: #166534; }
    .surface-badge.warning { background: #fef3c7; color: #92400e; }
    .surface-badge.danger { background: #fee2e2; color: #991b1b; }
    .surface-badge.pending, .surface-badge.info { background: #dbeafe; color: #1d4ed8; }
    .surface-kicker { font-size: 12px; font-weight: 800; letter-spacing: 0.12em; text-transform: uppercase; color: #64748b; margin: 0 0 8px 0; }
    .surface-lead { font-size: 20px; line-height: 1.55; margin: 0 0 10px 0; font-weight: 700; }
    .surface-copy, .surface-note { font-size: 16px; line-height: 1.7; color: #475569; margin: 0 0 12px 0; }
    .surface-metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 10px; margin: 12px 0; }
    .surface-metrics > div { background: white; border: 1px solid #e2e8f0; border-radius: 12px; padding: 10px 12px; }
    .metric-label { display: block; font-size: 11px; font-weight: 800; letter-spacing: 0.08em; text-transform: uppercase; color: #64748b; margin-bottom: 4px; }
    .metric-value { display: block; font-size: 18px; font-weight: 800; color: #0f172a; }
    .surface-list { margin: 0; padding-left: 20px; color: #334155; font-size: 15px; line-height: 1.8; }
    .chart-launch { display: block; width: 100%; padding: 0; border: 0; background: transparent; text-align: left; cursor: zoom-in; box-shadow: none; }
    .chart-launch:hover .chart-frame { border-color: #94a3b8; box-shadow: 0 10px 22px rgba(15, 23, 42, 0.08); }
    .chart-launch-hint { display: inline-flex; margin-top: 12px; font-size: 13px; font-weight: 700; color: #475569; }
    .chart-frame { padding: 22px; border-radius: 18px; border: 1px solid #dbe4ef; background: white; overflow: auto; }
    .chart-frame svg { display: block; width: 100%; height: auto; min-height: 560px; }
    .chart-modal[hidden] { display: none; }
    .chart-modal { position: fixed; inset: 0; z-index: 999; background: rgba(15, 23, 42, 0.72); display: flex; align-items: center; justify-content: center; padding: 16px; }
    .chart-modal-card { width: min(1440px, 98vw); max-height: 96vh; overflow: auto; background: #ffffff; border-radius: 18px; padding: 24px; box-shadow: 0 24px 60px rgba(15, 23, 42, 0.3); }
    .chart-modal-top { display: flex; align-items: center; justify-content: space-between; gap: 12px; margin-bottom: 12px; }
    .chart-modal-top h3 { margin: 0; font-size: 20px; }
    .chart-modal-close { background: #0f172a; color: white; border: 0; border-radius: 10px; padding: 8px 12px; cursor: pointer; }
    .chart-modal-body .chart-frame { padding: 22px; }
    .chart-modal-body .chart-frame svg { min-height: 980px; }
    .table-wrap { overflow: auto; border: 1px solid #e2e8f0; border-radius: 14px; background: white; }
    .data-table { width: 100%; border-collapse: collapse; font-size: 15px; }
    .data-table th, .data-table td { padding: 14px 16px; border-bottom: 1px solid #e2e8f0; text-align: left; vertical-align: top; }
    .data-table thead th { background: #f8fafc; font-size: 12px; font-weight: 800; letter-spacing: 0.08em; text-transform: uppercase; color: #475569; position: sticky; top: 0; }
    .table-status { display: inline-flex; align-items: center; gap: 6px; }
    .status-dot { width: 10px; height: 10px; border-radius: 999px; display: inline-block; }
    .status-dot.good { background: #16a34a; }
    .status-dot.warning { background: #f59e0b; }
    .status-dot.outlier { background: #dc2626; }
    details { margin-top: 10px; }
    summary { cursor: pointer; font-weight: 600; }
    .subset-panel { overflow: hidden; }
    .subset-toolbar { display: flex; align-items: flex-end; justify-content: space-between; gap: 16px; flex-wrap: wrap; margin-bottom: 16px; }
    .subset-heading-block { display: flex; flex-direction: column; gap: 8px; }
    .subset-heading { font-size: 28px; font-weight: 800; letter-spacing: -0.02em; margin: 0; line-height: 1.1; }
    .subset-run-field { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
    .subset-run-label { font-size: 12px; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; color: #475569; }
    .subset-run-input { width: min(320px, 100%); }
    .subset-mode-toggle { display: inline-flex; align-items: center; gap: 10px; padding: 12px 14px; border-radius: 14px; background: #eff6ff; border: 1px solid #bfdbfe; color: #1d4ed8; font-weight: 700; }
    .subset-mode-toggle input { width: 18px; height: 18px; accent-color: #2563eb; }
    .subset-summary-bar { display: flex; justify-content: space-between; align-items: center; gap: 16px; flex-wrap: wrap; padding: 14px 16px; border-radius: 16px; background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); color: white; margin-bottom: 18px; }
    .subset-summary-text { font-size: 18px; font-weight: 800; line-height: 1.4; }
    .subset-summary-hint { color: #cbd5e1; font-size: 13px; margin-top: 4px; }
    .mode-pill { display: inline-flex; align-items: center; padding: 6px 10px; border-radius: 999px; font-size: 12px; font-weight: 800; text-transform: uppercase; letter-spacing: 0.08em; background: #1e293b; color: #e2e8f0; }
    .mode-pill.manual { background: #dbeafe; color: #1d4ed8; }
    .mode-pill.group { background: #dcfce7; color: #166534; }
    .mode-pill.full { background: #fef3c7; color: #92400e; }
    .subset-layout { display: grid; grid-template-columns: minmax(300px, 0.95fr) minmax(460px, 1.4fr); gap: 18px; align-items: start; }
    .subset-column { min-height: 430px; display: flex; flex-direction: column; padding: 16px; border: 1px solid #dbe2ea; border-radius: 16px; background: linear-gradient(180deg, rgba(255,255,255,0.98) 0%, rgba(248,250,252,0.98) 100%); }
    .subset-column.secondary-mode { opacity: 0.7; }
    .panel-header { display: flex; justify-content: space-between; gap: 16px; align-items: flex-start; margin-bottom: 12px; }
    .panel-title-wrap { display: flex; flex-direction: column; gap: 6px; }
    .panel-title { font-size: 18px; font-weight: 800; margin: 0; }
    .panel-note { margin: 0; color: #64748b; font-size: 13px; line-height: 1.5; max-width: 44ch; }
    .group-list { display: grid; gap: 10px; overflow: auto; padding-right: 4px; }
    .group-card { display: flex; align-items: center; justify-content: space-between; gap: 12px; padding: 14px 16px; border: 1px solid #d3dbe7; border-radius: 14px; background: white; cursor: pointer; transition: border-color 0.18s ease, background 0.18s ease, box-shadow 0.18s ease, transform 0.18s ease; }
    .group-card:hover { border-color: #94a3b8; box-shadow: 0 10px 20px rgba(15, 23, 42, 0.08); transform: translateY(-1px); }
    .group-card.is-selected { border-color: #2563eb; background: linear-gradient(135deg, rgba(219,234,254,0.7) 0%, rgba(255,255,255,1) 100%); box-shadow: 0 10px 22px rgba(37, 99, 235, 0.14); }
    .group-card.is-disabled { opacity: 0.58; cursor: not-allowed; }
    .group-card input { width: 18px; height: 18px; accent-color: #2563eb; flex: 0 0 auto; }
    .group-main { display: flex; flex-direction: column; gap: 4px; min-width: 0; flex: 1 1 auto; }
    .group-kicker { font-size: 11px; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; color: #64748b; }
    .group-id { font-size: 20px; font-weight: 800; letter-spacing: -0.01em; color: #0f172a; }
    .group-meta { display: flex; flex-direction: column; align-items: flex-end; gap: 6px; text-align: right; }
    .group-count { font-size: 14px; font-weight: 700; color: #1f2937; }
    .group-state { display: inline-flex; align-items: center; padding: 5px 8px; border-radius: 999px; background: #e2e8f0; color: #475569; font-size: 11px; font-weight: 800; letter-spacing: 0.08em; text-transform: uppercase; }
    .group-card.is-selected .group-state { background: #dbeafe; color: #1d4ed8; }
    .group-panel-footnote { margin-top: 12px; padding-top: 10px; border-top: 1px solid #e2e8f0; }
    .clips-panel.manual-mode { border-color: #bfdbfe; box-shadow: inset 0 0 0 1px rgba(37, 99, 235, 0.12); }
    .selected-clips-topline { display: flex; align-items: center; justify-content: space-between; gap: 12px; flex-wrap: wrap; margin-bottom: 12px; }
    .selected-clips-count { font-size: 15px; font-weight: 800; color: #0f172a; }
    .manual-banner { display: none; align-items: center; padding: 6px 10px; border-radius: 999px; background: #dbeafe; color: #1d4ed8; font-size: 12px; font-weight: 800; letter-spacing: 0.06em; text-transform: uppercase; }
    .manual-banner.visible { display: inline-flex; }
    .clip-list-shell { display: flex; flex-direction: column; min-height: 320px; border: 1px solid #dde4ee; border-radius: 14px; background: white; overflow: hidden; }
    .clip-list-header { display: grid; grid-template-columns: 92px minmax(0, 1fr) 74px; gap: 12px; padding: 12px 14px; border-bottom: 1px solid #e5e7eb; background: #f8fafc; color: #64748b; font-size: 12px; font-weight: 800; letter-spacing: 0.08em; text-transform: uppercase; }
    .clip-list-header.manual { grid-template-columns: 52px 92px minmax(0, 1fr) 74px; }
    .clip-scroll { display: grid; max-height: 360px; overflow: auto; }
    .clip-row { display: grid; grid-template-columns: 92px minmax(0, 1fr) 74px; gap: 12px; align-items: center; padding: 12px 14px; border-bottom: 1px solid #edf2f7; background: white; font-family: "SFMono-Regular", ui-monospace, Menlo, monospace; font-size: 13px; line-height: 1.45; }
    .clip-row.manual { grid-template-columns: 52px 92px minmax(0, 1fr) 74px; cursor: pointer; }
    .clip-row.selected { background: #f8fbff; }
    .clip-row:last-child { border-bottom: 0; }
    .clip-checkbox { width: 16px; height: 16px; accent-color: #2563eb; }
    .clip-camera { display: inline-flex; align-items: center; justify-content: center; padding: 4px 8px; border-radius: 999px; background: #e2e8f0; color: #334155; font-weight: 800; }
    .clip-id { min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: #0f172a; font-weight: 800; }
    .clip-group { text-align: right; color: #64748b; font-weight: 700; }
    .empty-state { padding: 20px 16px; color: #64748b; font-size: 14px; line-height: 1.6; }
    @media (max-width: 980px) {
      .subset-layout { grid-template-columns: 1fr; }
      .subset-column { min-height: 0; }
      .subset-summary-bar { align-items: flex-start; }
      .header-row { align-items: flex-start; }
    }
    @media (max-width: 640px) {
      .page { padding: 18px; }
      .title { font-size: 28px; }
      .subset-toolbar { align-items: stretch; }
      .subset-run-field { align-items: flex-start; }
      .subset-run-input { width: 100%; }
      .clip-list-header { display: none; }
      .clip-list-header.manual { display: none; }
      .clip-row, .clip-row.manual { grid-template-columns: 1fr; gap: 8px; }
      .clip-group { text-align: left; }
      .group-meta { align-items: flex-start; text-align: left; }
    }
  </style>
</head>
<body>
  <div class="page">
    <div class="header">
      <div class="header-row">
        {% if has_logo %}
          <img src="{{ url_for('logo') }}" alt="R3DMatch logo">
        {% endif %}
        <div>
          <h1 class="title">R3DMatch</h1>
          <p class="subtitle">Internal Review</p>
        </div>
      </div>
      <p class="instructions">Enter a calibration folder path, configure settings, then run review.</p>
    </div>

    {% if error %}
      <div class="status error">{{ error }}</div>
    {% endif %}
    {% if message %}
      <div class="status info">{{ message }}</div>
    {% endif %}

    <form method="post" action="{{ url_for('scan') }}">
      <div class="card">
        <h2 class="section-title">Calibration Folder</h2>
        <div class="row">
          <div class="field">
            <label for="source_mode">Source Mode</label>
            <select id="source_mode" name="source_mode">
              <option value="local_folder" {% if form.source_mode == 'local_folder' %}selected{% endif %}>Local Folder</option>
              <option value="ftps_camera_pull" {% if form.source_mode == 'ftps_camera_pull' %}selected{% endif %}>FTPS Camera Pull</option>
            </select>
          </div>
        </div>
        <div class="field">
          <label for="input_path">Calibration Folder Path</label>
          <input id="input_path" name="input_path" type="text" value="{{ form.input_path }}">
        </div>
        <div id="ftps-fields" {% if form.source_mode != 'ftps_camera_pull' %}style="display:none"{% endif %}>
          <div class="row">
            <div class="field">
              <label for="ftps_reel">FTPS Reel</label>
              <input id="ftps_reel" name="ftps_reel" type="text" value="{{ form.ftps_reel }}" placeholder="007">
            </div>
            <div class="field">
              <label for="ftps_clips">Clip Numbers / Ranges</label>
              <input id="ftps_clips" name="ftps_clips" type="text" value="{{ form.ftps_clips }}" placeholder="63,64-65">
            </div>
            <div class="field">
              <label for="ftps_cameras">Camera Subset</label>
              <input id="ftps_cameras" name="ftps_cameras" type="text" value="{{ form.ftps_cameras }}" placeholder="AA,AB,AC or leave blank for all">
            </div>
          </div>
          <div class="meta">FTPS ingest uses the built-in camera map and default RED camera credentials unless overridden in the CLI.</div>
        </div>
        <div class="actions">
          <button type="submit">Scan Folder</button>
        </div>
      </div>

      <div class="card">
        <h2 class="section-title">Source Summary</h2>
        <div class="summary-box">
          <div>{{ scan_summary_text }}</div>
          {% if scan.sample_clip_ids %}
            <ul class="clip-list">
              {% for clip_id in scan.sample_clip_ids %}
                <li>{{ clip_id }}</li>
              {% endfor %}
              {% if scan.remaining_count > 0 %}
                <li>... and {{ scan.remaining_count }} more</li>
              {% endif %}
            </ul>
          {% endif %}
        </div>
      </div>

      <div class="card subset-panel" id="subset-panel">
        <div class="subset-toolbar">
          <div class="subset-heading-block">
            <h2 class="subset-heading">Calibration Subset</h2>
            <div class="subset-run-field">
              <span class="subset-run-label">Run Label</span>
              <input class="subset-run-input" id="run_label" name="run_label" type="text" value="{{ form.run_label }}" placeholder="clip63_even">
              <span class="meta">Use a short label to keep subset runs separate under the parent output folder.</span>
            </div>
          </div>
          <label class="subset-mode-toggle" for="advanced_clip_selection">
            <input id="advanced_clip_selection" name="advanced_clip_selection" type="checkbox" value="1" {% if form.advanced_clip_selection %}checked{% endif %}>
            <span>Enable advanced clip selection</span>
          </label>
        </div>
        <div class="subset-summary-bar">
          <div>
            <div class="subset-summary-text" id="subset-selection-summary">{{ subset_ui.summary_text }}</div>
            <div class="subset-summary-hint" id="subset-selection-hint">{{ subset_ui.summary_hint }}</div>
          </div>
          <span class="mode-pill {{ subset_ui.mode_class }}" id="subset-selection-mode">{{ subset_ui.mode_label }}</span>
        </div>
        <div class="subset-layout">
          <section class="subset-column {% if form.advanced_clip_selection %}secondary-mode{% endif %}" id="subset-groups-panel">
            <div class="panel-header">
              <div class="panel-title-wrap">
                <h3 class="panel-title">Clip Groups</h3>
                <p class="panel-note">Primary control for fast subset selection. Pick one or more groups to define the calibration subset.</p>
              </div>
            </div>
            {% if scan.clip_groups %}
              <div class="group-list" id="group-list">
                {% for group in scan.clip_groups %}
                  <label class="group-card {% if group.group_id in form.selected_clip_groups %}is-selected{% endif %} {% if form.advanced_clip_selection %}is-disabled{% endif %}">
                    <input class="group-selector" type="checkbox" name="selected_clip_groups" value="{{ group.group_id }}" {% if group.group_id in form.selected_clip_groups %}checked{% endif %} {% if form.advanced_clip_selection %}disabled{% endif %}>
                    <div class="group-main">
                      <span class="group-kicker">Group</span>
                      <span class="group-id">{{ group.group_id }}</span>
                    </div>
                    <div class="group-meta">
                      <span class="group-count">{{ group.clip_count }} clip{% if group.clip_count != 1 %}s{% endif %}</span>
                      <span class="group-state">{% if group.group_id in form.selected_clip_groups %}Selected{% else %}Available{% endif %}</span>
                    </div>
                  </label>
                {% endfor %}
              </div>
              <div class="meta group-panel-footnote" id="group-panel-footnote">{{ subset_ui.group_panel_note }}</div>
            {% else %}
              <div class="meta">Scan a calibration folder to discover clip groups.</div>
            {% endif %}
          </section>
          <section class="subset-column clips-panel {% if form.advanced_clip_selection %}manual-mode{% else %}read-only{% endif %}" id="subset-clips-panel">
            <div class="panel-header">
              <div class="panel-title-wrap">
                <h3 class="panel-title">Selected Clips</h3>
                <p class="panel-note" id="clip-panel-note">{{ subset_ui.clip_panel_note }}</p>
              </div>
            </div>
            <div class="selected-clips-topline">
              <div class="selected-clips-count"><span id="selected-clip-count">{{ selected_clip_ids|length }}</span> clips in view</div>
              <div class="manual-banner {% if form.advanced_clip_selection %}visible{% endif %}" id="manual-selection-banner">Manual clip selection active</div>
            </div>
            {% if scan.clip_records %}
              <div class="clip-list-shell">
                <div class="clip-list-header {% if form.advanced_clip_selection %}manual{% endif %}" id="subset-clip-header">
                  {% if form.advanced_clip_selection %}
                    <span>Select</span>
                  {% endif %}
                  <span>Camera</span>
                  <span>Clip ID</span>
                  <span>Group</span>
                </div>
                <div class="clip-scroll" id="subset-clip-list">
                  {% if form.advanced_clip_selection %}
                    {% for clip in scan.clip_records %}
                      <label class="clip-row manual {% if clip.clip_id in selected_clip_ids %}selected{% endif %}">
                        <input class="clip-selector clip-checkbox" data-group="{{ clip.subset_group }}" type="checkbox" name="selected_clip_ids" value="{{ clip.clip_id }}" {% if clip.clip_id in selected_clip_ids %}checked{% endif %}>
                        <span class="clip-camera">Cam {{ clip.camera_label }}</span>
                        <span class="clip-id">{{ clip.clip_id }}</span>
                        <span class="clip-group">{{ clip.subset_group }}</span>
                      </label>
                    {% endfor %}
                  {% else %}
                    {% for clip in scan.clip_records if clip.clip_id in selected_clip_ids %}
                      <div class="clip-row selected">
                        <span class="clip-camera">Cam {{ clip.camera_label }}</span>
                        <span class="clip-id">{{ clip.clip_id }}</span>
                        <span class="clip-group">{{ clip.subset_group }}</span>
                      </div>
                    {% endfor %}
                  {% endif %}
                </div>
              </div>
            {% else %}
              <div class="meta">Scan a calibration folder to preview the clips in this subset.</div>
            {% endif %}
          </section>
        </div>
        {% if scan.clip_records %}
          <script id="subset-data" type="application/json">{{ subset_ui_data|tojson }}</script>
        {% endif %}
      </div>

      <div class="card">
        <h2 class="section-title">Output Folder</h2>
        <div class="field">
          <label for="output_path">Output Folder Path</label>
          <input id="output_path" name="output_path" type="text" value="{{ form.output_path }}">
        </div>
      </div>

      <div class="card">
        <h2 class="section-title">Basic Settings</h2>
        <div class="row">
          <div class="field">
            <label for="backend">Backend</label>
            <select id="backend" name="backend">
              {% for value in ['red', 'mock'] %}
                <option value="{{ value }}" {% if form.backend == value %}selected{% endif %}>{{ value }}</option>
              {% endfor %}
            </select>
          </div>
          <div class="field">
            <label for="target_type">Target Type</label>
            <select id="target_type" name="target_type">
              {% for value in ['gray_sphere', 'gray_card', 'color_chart'] %}
                <option value="{{ value }}" {% if form.target_type == value %}selected{% endif %}>{{ value }}</option>
              {% endfor %}
            </select>
          </div>
          <div class="field">
            <label for="processing_mode">Processing Mode</label>
            <select id="processing_mode" name="processing_mode">
              {% for value in ['exposure', 'color', 'both'] %}
                <option value="{{ value }}" {% if form.processing_mode == value %}selected{% endif %}>{{ value }}</option>
              {% endfor %}
            </select>
          </div>
          <div class="field">
            <label for="matching_domain_display">Measurement Domain</label>
            <input id="matching_domain" name="matching_domain" type="hidden" value="perceptual">
            <input id="matching_domain_display" type="text" value="Perceptual (IPP2 / BT.709 / BT.1886)" readonly>
            <div id="matching-domain-note" class="meta">Lightweight review is locked to the perceptual IPP2 monitoring domain so sphere measurement, operator review, and recommended corrections all use the same rendered pixels.</div>
          </div>
          <div class="field">
            <label for="review_mode">Review Mode</label>
            <select id="review_mode" name="review_mode">
              <option value="full_contact_sheet" {% if form.review_mode == 'full_contact_sheet' %}selected{% endif %}>Full Contact Sheet</option>
              <option value="lightweight_analysis" {% if form.review_mode == 'lightweight_analysis' %}selected{% endif %}>Lightweight Analysis</option>
            </select>
            <div class="meta">Lightweight Analysis skips bulk still generation and produces a fast exposure-first diagnostic brief.</div>
          </div>
        </div>
      </div>

      <details class="card">
        <summary>Advanced / Debug Controls</summary>
        <div class="meta" style="margin-top:12px;">Sphere localization is the default spatial basis for gray-sphere calibration. Manual ROI is available here only for fallback debugging.</div>
        <div class="field" style="margin-top:14px;">
          <label for="roi_mode">ROI Mode</label>
          <select id="roi_mode" name="roi_mode">
            {% for value, label in [('sphere_auto', 'sphere_auto'), ('draw', 'draw'), ('center', 'center'), ('full', 'full'), ('manual', 'manual')] %}
              <option value="{{ value }}" {% if form.roi_mode == value %}selected{% endif %}>{{ label }}</option>
            {% endfor %}
          </select>
        </div>
        <div id="roi-description" class="meta">{{ roi_description }}</div>
        <div class="row" id="manual-roi-row">
          <div class="field"><label for="roi_x">x</label><input id="roi_x" name="roi_x" type="text" value="{{ form.roi_x }}"></div>
          <div class="field"><label for="roi_y">y</label><input id="roi_y" name="roi_y" type="text" value="{{ form.roi_y }}"></div>
          <div class="field"><label for="roi_w">w</label><input id="roi_w" name="roi_w" type="text" value="{{ form.roi_w }}"></div>
          <div class="field"><label for="roi_h">h</label><input id="roi_h" name="roi_h" type="text" value="{{ form.roi_h }}"></div>
        </div>
        {% if scan.preview_available %}
          <div class="roi-preview-wrap">
            <p class="roi-preview-guidance">Draw region around gray target.</p>
            <div class="roi-preview-stage" id="roi-stage">
              <img id="roi-image" src="{{ url_for('scan_preview') }}" alt="ROI preview">
              <div id="roi-overlay" class="roi-overlay"></div>
              <div id="roi-canvas" class="roi-canvas"></div>
            </div>
            {% if scan.preview_note %}
              <div class="roi-preview-note">{{ scan.preview_note }}</div>
            {% endif %}
          </div>
        {% else %}
          <div class="meta">{{ scan.preview_warning or 'Preview available after scan or first render.' }}</div>
        {% endif %}
      </details>

      <div class="card">
        <h2 class="section-title">Strategies</h2>
        <div class="checkbox-row">
          <label><input type="checkbox" name="target_strategies" value="median" {% if 'median' in form.target_strategies %}checked{% endif %}> median</label>
          <label><input type="checkbox" name="target_strategies" value="optimal-exposure" {% if 'optimal-exposure' in form.target_strategies or 'brightest-valid' in form.target_strategies %}checked{% endif %}> Optimal Exposure (Best Match to Gray)</label>
          <label><input id="manual_strategy" type="checkbox" name="target_strategies" value="manual" {% if 'manual' in form.target_strategies %}checked{% endif %}> manual</label>
          <label><input id="hero_strategy" type="checkbox" name="target_strategies" value="hero-camera" {% if 'hero-camera' in form.target_strategies %}checked{% endif %}> hero-camera</label>
        </div>
        <div class="field" style="margin-top: 12px;">
          <label for="reference_clip_id">Manual Reference Clip ID</label>
          <input id="reference_clip_id" name="reference_clip_id" type="text" value="{{ form.reference_clip_id }}">
        </div>
        <div class="field" style="margin-top: 12px;">
          <label for="hero_clip_id">Hero Camera</label>
          <select id="hero_clip_id" name="hero_clip_id">
            <option value="">Select hero camera</option>
            {% for clip_id in selected_clip_ids or [] %}
              <option value="{{ clip_id }}" {% if form.hero_clip_id == clip_id %}selected{% endif %}>{{ clip_id }}</option>
            {% endfor %}
          </select>
          <div class="meta">Hero-camera strategy matches every non-hero camera to this selected clip.</div>
        </div>
      </div>

      <div class="card">
        <h2 class="section-title">Preview</h2>
        <div class="field">
          <label for="preview_mode">Preview Mode</label>
          <select id="preview_mode" name="preview_mode">
            {% for value in ['monitoring'] %}
              <option value="{{ value }}" {% if form.preview_mode == value %}selected{% endif %}>{{ value }}</option>
            {% endfor %}
          </select>
        </div>
        <div class="field">
          <label for="preview_lut">LUT Path (.cube, optional)</label>
          <input id="preview_lut" name="preview_lut" type="text" value="{{ form.preview_lut }}">
        </div>
      </div>

      <div class="card">
        <h2 class="section-title">Actions</h2>
        <div class="actions">
          <button type="submit" formaction="{{ url_for('run_review') }}">Run Review</button>
          <button type="submit" formaction="{{ url_for('approve_master') }}" class="secondary">Approve Master RMD</button>
          <button type="submit" formaction="{{ url_for('clear_preview') }}" class="secondary">Clear Preview Cache</button>
          {% if report_url %}
            <a class="link-button secondary" href="{{ report_url }}" target="_blank" rel="noopener">Open Report</a>
          {% endif %}
        </div>
      </div>

      <div class="card push-card">
        <h2 class="section-title">Push Looks to Cameras</h2>
        <p class="section-subtitle">Write calibration corrections to connected RED cameras via RCP2.</p>
        <div class="row">
          <div class="field">
            <label for="apply_cameras">Apply Camera Subset</label>
            <input id="apply_cameras" name="apply_cameras" type="text" value="{{ form.apply_cameras }}" placeholder="GA,IC or leave blank for all payload cameras">
          </div>
          <div class="field">
            <label for="verification_after_path">After-Apply Review Folder</label>
            <input id="verification_after_path" name="verification_after_path" type="text" value="{{ form.verification_after_path }}" placeholder="/path/to/reacquired/review/output">
          </div>
        </div>
        <div class="actions">
          <button type="submit" formaction="{{ url_for('load_recommended_payload_route') }}" class="secondary">Load Recommended Payload</button>
          <button type="submit" formaction="{{ url_for('read_current_camera_values_route') }}" class="secondary">Read Current Camera Values</button>
          <button type="submit" formaction="{{ url_for('apply_calibration_route') }}" class="secondary">Dry Run Push</button>
          <button type="submit" formaction="{{ url_for('apply_calibration_live_route') }}" class="secondary">Live Push</button>
          <button type="submit" formaction="{{ url_for('verify_last_push_route') }}" class="secondary">Verify Last Push</button>
          <button type="submit" formaction="{{ url_for('verify_apply_route') }}" class="secondary">Compare Before / After Review</button>
        </div>
        <div class="meta">Load Recommended Payload previews the current commit package. Read Current Camera Values queries the connected cameras in read-only mode. Push controls use the generated commit payload for the current output folder. Compare Before / After Review compares the current review output against a later reacquired review run.</div>
      </div>
    </form>

    <div class="card">
      <h2 class="section-title">Execution</h2>
      <div><strong>Status:</strong> <span id="task-status">{{ task.status }}</span></div>
      <div class="meta"><strong>Current Stage:</strong> <span id="task-stage">{{ task.stage }}</span></div>
      <div class="meta"><strong>Detail:</strong> <span id="task-detail">{{ task.status_detail or 'None' }}</span></div>
      <div class="meta"><strong>Watchdog:</strong> <span id="task-watchdog">{{ task.watchdog_status }}</span> | Process alive: <span id="task-process-alive">{{ 'yes' if task.process_alive else 'no' }}</span> | Last activity: <span id="task-last-activity">{{ task.last_activity_seconds }}</span>s ago</div>
      <div class="meta"><strong>Clips Found:</strong> <span id="task-clip-count">{{ task.clip_count }}</span></div>
      <div class="meta"><strong>Strategies:</strong> <span id="task-strategies">{{ task.strategies_text }}</span></div>
      <div class="meta"><strong>Review Mode:</strong> <span id="task-review-mode">{{ task.review_mode_label or task.review_mode }}</span></div>
      <div class="meta"><strong>Source Mode:</strong> <span id="task-source-mode">{{ task.source_mode_label or task.source_mode }}</span></div>
      <div class="meta"><strong>Preview Mode:</strong> <span id="task-preview-mode">{{ task.preview_mode }}</span></div>
      <div class="meta"><strong>Output Folder:</strong> <span id="task-output">{{ task.output_path or form.output_path }}</span></div>
      <div class="progress-track"><div id="task-progress-fill" class="progress-fill" style="width: {{ task.progress_percent }}%;"></div></div>
      <div class="meta"><span id="task-counts">{{ task.items_completed }} / {{ task.items_total }}</span></div>
      <div class="actions" style="margin-top: 12px;">
        <button
          type="button"
          class="secondary"
          id="cancel-run-button"
          {% if not task.can_cancel %}style="display: none;"{% endif %}
          {% if task.status == 'cancelling' %}disabled{% endif %}
          onclick="cancelRun()"
        >{% if task.status == 'cancelling' %}Cancelling...{% else %}Stop Run{% endif %}</button>
      </div>
      <ul class="stage-list" id="stage-list">
        {% for stage_name in task.stage_names %}
          <li class="stage-pill {% if loop.index0 == task.stage_index %}active{% endif %}">{{ stage_name }}</li>
        {% endfor %}
      </ul>
      <div class="actions" id="completion-actions">
        {% if task.report_ready and task.preview_pdf_url %}
          <a class="link-button" id="preview-pdf-link" href="{{ task.preview_pdf_url }}" target="_blank" rel="noopener">Open Preview PDF</a>
        {% endif %}
        {% if task.report_ready and task.preview_html_url %}
          <a class="link-button secondary" id="preview-html-link" href="{{ task.preview_html_url }}" target="_blank" rel="noopener">Open HTML Preview</a>
        {% endif %}
        {% if task.output_folder and task.report_ready %}
          <button type="button" class="secondary" id="open-output-button" onclick="openOutputFolder()">Open Output Folder</button>
        {% endif %}
      </div>
      <details>
        <summary>Technical Details</summary>
        <div class="meta"><strong>Command:</strong> <span id="task-command">{{ task.command }}</span></div>
        <pre class="log" id="task-log">{{ task.log_text }}</pre>
      </details>
    </div>

    <div class="card">
      <h2 class="section-title">Calibration Decision</h2>
      <div id="decision-surface">{{ task.decision_surface_html | safe }}</div>
    </div>

    <div class="card">
      <h2 class="section-title">Per-Camera Commit Table</h2>
      <div id="commit-table-surface">{{ task.commit_table_html | safe }}</div>
    </div>

    <div class="card push-card">
      <h2 class="section-title">Push Looks to Cameras</h2>
      <p class="section-subtitle">Write calibration corrections to connected RED cameras via RCP2.</p>
      <div id="push-surface">{{ task.push_surface_html | safe }}</div>
    </div>

    <div class="card">
      <h2 class="section-title">Apply Results</h2>
      <div id="apply-surface">{{ task.apply_surface_html | safe }}</div>
    </div>

    <div class="card">
      <h2 class="section-title">Post-Apply Verification</h2>
      <div id="verification-surface">{{ task.verification_surface_html | safe }}</div>
    </div>
  </div>

  <div id="chart-modal" class="chart-modal" hidden>
    <div class="chart-modal-card">
      <div class="chart-modal-top">
        <h3 id="chart-modal-title">Chart</h3>
        <button type="button" class="chart-modal-close" id="chart-modal-close">Close</button>
      </div>
      <div class="chart-modal-body" id="chart-modal-body"></div>
    </div>
  </div>

  <script>
    function clamp(value, min, max) {
      return Math.max(min, Math.min(max, value));
    }

    function wireChartModal() {
      const modal = document.getElementById('chart-modal');
      const body = document.getElementById('chart-modal-body');
      const title = document.getElementById('chart-modal-title');
      const close = document.getElementById('chart-modal-close');
      if (!modal || !body || !title || !close) return;
      document.addEventListener('click', (event) => {
        const trigger = event.target.closest('.chart-launch');
        if (trigger) {
          const frame = trigger.querySelector('.chart-frame');
          title.textContent = trigger.dataset.chartTitle || 'Chart';
          body.innerHTML = frame ? frame.outerHTML : '';
          modal.hidden = false;
          return;
        }
        if (event.target === modal || event.target === close) {
          modal.hidden = true;
          body.innerHTML = '';
        }
      });
      document.addEventListener('keydown', (event) => {
        const trigger = event.target.closest ? event.target.closest('.chart-launch') : null;
        if (trigger && (event.key === 'Enter' || event.key === ' ')) {
          event.preventDefault();
          const frame = trigger.querySelector('.chart-frame');
          title.textContent = trigger.dataset.chartTitle || 'Chart';
          body.innerHTML = frame ? frame.outerHTML : '';
          modal.hidden = false;
          return;
        }
        if (event.key === 'Escape') {
          modal.hidden = true;
          body.innerHTML = '';
        }
      });
    }
    wireChartModal();

    function toggleManualReference() {
      const manual = document.getElementById('manual_strategy');
      const ref = document.getElementById('reference_clip_id');
      ref.disabled = !manual.checked;
    }
    toggleManualReference();
    document.getElementById('manual_strategy').addEventListener('change', toggleManualReference);

    function toggleHeroSelector() {
      const hero = document.getElementById('hero_strategy');
      const heroSelect = document.getElementById('hero_clip_id');
      heroSelect.disabled = !hero.checked;
    }
    toggleHeroSelector();
    document.getElementById('hero_strategy').addEventListener('change', toggleHeroSelector);

    function wireSubsetPanel() {
      const subsetDataElement = document.getElementById('subset-data');
      if (!subsetDataElement) return;
      const subsetData = JSON.parse(subsetDataElement.textContent);
      const advancedToggle = document.getElementById('advanced_clip_selection');
      const groupSelectors = Array.from(document.querySelectorAll('.group-selector'));
      const groupPanel = document.getElementById('subset-groups-panel');
      const clipsPanel = document.getElementById('subset-clips-panel');
      const clipHeader = document.getElementById('subset-clip-header');
      const clipList = document.getElementById('subset-clip-list');
      const summaryText = document.getElementById('subset-selection-summary');
      const summaryHint = document.getElementById('subset-selection-hint');
      const modePill = document.getElementById('subset-selection-mode');
      const groupPanelNote = document.getElementById('group-panel-footnote');
      const clipPanelNote = document.getElementById('clip-panel-note');
      const clipCount = document.getElementById('selected-clip-count');
      const manualBanner = document.getElementById('manual-selection-banner');
      const allClipIds = (subsetData.clip_records || []).map((record) => record.clip_id);
      const groupMap = new Map((subsetData.clip_groups || []).map((group) => [group.group_id, group.clip_ids || []]));
      let manualSelection = new Set((subsetData.selected_clip_ids || []).filter((clipId) => allClipIds.includes(clipId)));

      function pluralize(count, singular, plural) {
        return count === 1 ? singular : (plural || singular + 's');
      }

      function selectedGroupIds() {
        return groupSelectors.filter((groupBox) => groupBox.checked).map((groupBox) => groupBox.value);
      }

      function groupDrivenSelection(groupIds) {
        if (!groupIds.length) return allClipIds.slice();
        const selected = new Set();
        groupIds.forEach((groupId) => {
          (groupMap.get(groupId) || []).forEach((clipId) => selected.add(clipId));
        });
        return allClipIds.filter((clipId) => selected.has(clipId));
      }

      function effectiveSelectedClipIds() {
        if (advancedToggle.checked) {
          return allClipIds.filter((clipId) => manualSelection.has(clipId));
        }
        return groupDrivenSelection(selectedGroupIds());
      }

      function selectionCopy(groupIds, selectedIds) {
        if (advancedToggle.checked) {
          return {
            summary: 'Manual selection: ' + selectedIds.length + ' ' + pluralize(selectedIds.length, 'clip'),
            hint: selectedIds.length
              ? 'Manual clip selection is active. Group selection is locked until you switch back.'
              : 'Manual clip selection is active. Choose one or more clips before running review.',
            mode: 'Manual Mode',
            modeClass: 'manual',
            groupNote: 'Clip groups are disabled while manual clip selection is active.',
            clipNote: 'Interactive clip list. Choose the exact clips to include in this run.',
          };
        }
        if (groupIds.length === 1) {
          return {
            summary: selectedIds.length + ' clips selected from group ' + groupIds[0],
            hint: 'Group selection drives the subset. The clip list on the right is a read-only preview.',
            mode: 'Group Mode',
            modeClass: 'group',
            groupNote: 'One group selected. Add more groups to expand the subset.',
            clipNote: 'Read-only preview of the current group-driven subset.',
          };
        }
        if (groupIds.length > 1) {
          return {
            summary: selectedIds.length + ' clips selected from groups ' + groupIds.join(', '),
            hint: 'Multiple groups are combined into one subset. The clip list is a read-only preview.',
            mode: 'Group Mode',
            modeClass: 'group',
            groupNote: 'Multiple groups selected. The run will include the combined clip set shown on the right.',
            clipNote: 'Read-only preview of the combined group selection.',
          };
        }
        if (allClipIds.length) {
          return {
            summary: 'Full calibration set selected: ' + selectedIds.length + ' clips across ' + (subsetData.clip_groups || []).length + ' ' + pluralize((subsetData.clip_groups || []).length, 'group'),
            hint: 'No group filter is active yet. Choose one or more groups to narrow the run.',
            mode: 'Full Set',
            modeClass: 'full',
            groupNote: 'No group filter selected. Review will use the full discovered set.',
            clipNote: 'Read-only preview of the full calibration set.',
          };
        }
        return {
          summary: 'No clips selected yet.',
          hint: 'Scan a calibration folder to populate the subset controls.',
          mode: 'Full Set',
          modeClass: 'full',
          groupNote: 'Scan a calibration folder to discover clip groups.',
          clipNote: 'Scan a calibration folder to preview clips for this subset.',
        };
      }

      function renderClipHeader() {
        clipHeader.className = 'clip-list-header' + (advancedToggle.checked ? ' manual' : '');
        clipHeader.innerHTML = '';
        const labels = advancedToggle.checked ? ['Select', 'Camera', 'Clip ID', 'Group'] : ['Camera', 'Clip ID', 'Group'];
        labels.forEach((label) => {
          const cell = document.createElement('span');
          cell.textContent = label;
          clipHeader.appendChild(cell);
        });
      }

      function renderClipList(selectedIds) {
        clipList.innerHTML = '';
        const readOnlyRecords = (subsetData.clip_records || []).filter((record) => selectedIds.includes(record.clip_id));
        const records = advancedToggle.checked ? (subsetData.clip_records || []) : readOnlyRecords;
        if (!records.length) {
          const emptyState = document.createElement('div');
          emptyState.className = 'empty-state';
          emptyState.textContent = advancedToggle.checked
            ? 'No clips are selected yet. Use the checkboxes above to build a manual subset.'
            : 'No clips are selected yet. Choose one or more groups to populate this preview.';
          clipList.appendChild(emptyState);
          return;
        }
        records.forEach((record) => {
          const selected = selectedIds.includes(record.clip_id);
          const row = document.createElement(advancedToggle.checked ? 'label' : 'div');
          row.className = 'clip-row' + (advancedToggle.checked ? ' manual' : '') + (selected ? ' selected' : '');
          if (advancedToggle.checked) {
            const checkbox = document.createElement('input');
            checkbox.type = 'checkbox';
            checkbox.name = 'selected_clip_ids';
            checkbox.value = record.clip_id;
            checkbox.checked = manualSelection.has(record.clip_id);
            checkbox.className = 'clip-selector clip-checkbox';
            checkbox.dataset.group = record.subset_group;
            checkbox.addEventListener('change', () => {
              if (checkbox.checked) {
                manualSelection.add(record.clip_id);
              } else {
                manualSelection.delete(record.clip_id);
              }
              syncSubsetPanel();
            });
            row.appendChild(checkbox);
          }
          const camera = document.createElement('span');
          camera.className = 'clip-camera';
          camera.textContent = 'Cam ' + record.camera_label;
          row.appendChild(camera);
          const clipId = document.createElement('span');
          clipId.className = 'clip-id';
          clipId.textContent = record.clip_id;
          row.appendChild(clipId);
          const group = document.createElement('span');
          group.className = 'clip-group';
          group.textContent = record.subset_group;
          row.appendChild(group);
          clipList.appendChild(row);
        });
      }

      function updateGroupCards() {
        groupSelectors.forEach((groupBox) => {
          const card = groupBox.closest('.group-card');
          const state = card ? card.querySelector('.group-state') : null;
          groupBox.disabled = advancedToggle.checked;
          if (card) {
            card.classList.toggle('is-selected', groupBox.checked);
            card.classList.toggle('is-disabled', advancedToggle.checked);
          }
          if (state) {
            state.textContent = groupBox.checked ? 'Selected' : 'Available';
          }
        });
        groupPanel.classList.toggle('secondary-mode', advancedToggle.checked);
        clipsPanel.classList.toggle('manual-mode', advancedToggle.checked);
      }

      function syncSubsetPanel() {
        const groups = selectedGroupIds();
        const selectedIds = effectiveSelectedClipIds();
        const copy = selectionCopy(groups, selectedIds);
        updateGroupCards();
        renderClipHeader();
        renderClipList(selectedIds);
        summaryText.textContent = copy.summary;
        summaryHint.textContent = copy.hint;
        modePill.textContent = copy.mode;
        modePill.className = 'mode-pill ' + copy.modeClass;
        clipPanelNote.textContent = copy.clipNote;
        if (groupPanelNote) {
          groupPanelNote.textContent = copy.groupNote;
        }
        clipCount.textContent = String(selectedIds.length);
        manualBanner.classList.toggle('visible', advancedToggle.checked);
      }

      groupSelectors.forEach((groupBox) => groupBox.addEventListener('change', syncSubsetPanel));
      if (advancedToggle) {
        advancedToggle.addEventListener('change', () => {
          if (advancedToggle.checked) {
            manualSelection = new Set(groupDrivenSelection(selectedGroupIds()));
          }
          syncSubsetPanel();
        });
      }
      syncSubsetPanel();
    }

    function applyRoiMode() {
      const roiModeField = document.getElementById('roi_mode');
      const manualRow = document.getElementById('manual-roi-row');
      const desc = document.getElementById('roi-description');
      const x = document.getElementById('roi_x');
      const y = document.getElementById('roi_y');
      const w = document.getElementById('roi_w');
      const h = document.getElementById('roi_h');
      if (!roiModeField || !manualRow || !desc || !x || !y || !w || !h) return;
      const mode = roiModeField.value;
      if (mode === 'sphere_auto') {
        desc.textContent = 'Automatic sphere detection drives the gray-sphere measurement workflow.';
        manualRow.style.display = 'none';
      } else if (mode === 'center') {
        x.value = '0.35'; y.value = '0.35'; w.value = '0.30'; h.value = '0.30';
        desc.textContent = 'Center crop (30% of frame)';
        manualRow.style.display = 'none';
      } else if (mode === 'full') {
        x.value = '0.0'; y.value = '0.0'; w.value = '1.0'; h.value = '1.0';
        desc.textContent = 'Full frame';
        manualRow.style.display = 'none';
      } else if (mode === 'draw') {
        desc.textContent = 'Draw a custom ROI on the preview image.';
        manualRow.style.display = 'none';
      } else {
        desc.textContent = 'Manual ROI values';
        manualRow.style.display = '';
      }
      updateOverlayFromInputs();
    }
    const roiModeField = document.getElementById('roi_mode');
    if (roiModeField) {
      roiModeField.addEventListener('change', applyRoiMode);
    }

    function applySourceMode() {
      const mode = document.getElementById('source_mode').value;
      const ftpsFields = document.getElementById('ftps-fields');
      const inputField = document.getElementById('input_path');
      const inputLabel = document.querySelector('label[for="input_path"]');
      if (ftpsFields) {
        ftpsFields.style.display = mode === 'ftps_camera_pull' ? '' : 'none';
      }
      if (inputLabel) {
        inputLabel.textContent = mode === 'ftps_camera_pull' ? 'Local Ingest Cache Root (optional)' : 'Calibration Folder Path';
      }
      if (inputField && mode === 'ftps_camera_pull' && !inputField.value) {
        inputField.placeholder = 'Leave blank to ingest under the review output folder';
      } else if (inputField) {
        inputField.placeholder = '';
      }
    }
    document.getElementById('source_mode').addEventListener('change', applySourceMode);

    function applyReviewMode() {
      const reviewMode = document.getElementById('review_mode').value;
      const matchingDomain = document.getElementById('matching_domain');
      const note = document.getElementById('matching-domain-note');
      const lightweight = reviewMode === 'lightweight_analysis';
      if (matchingDomain) {
        matchingDomain.value = 'perceptual';
      }
      if (note) {
        note.textContent = lightweight
          ? 'Lightweight Analysis is locked to the perceptual IPP2 domain so sphere measurement, operator review, and recommended corrections all use the same rendered pixels.'
          : 'Review runs use the perceptual IPP2 monitoring domain so sphere measurement and operator review stay aligned.';
      }
    }
    document.getElementById('review_mode').addEventListener('change', applyReviewMode);

    function updateOverlayFromInputs() {
      const stage = document.getElementById('roi-stage');
      const overlay = document.getElementById('roi-overlay');
      const desc = document.getElementById('roi-description');
      const xField = document.getElementById('roi_x');
      const yField = document.getElementById('roi_y');
      const wField = document.getElementById('roi_w');
      const hField = document.getElementById('roi_h');
      if (!stage || !overlay || !desc || !xField || !yField || !wField || !hField) return;
      const x = parseFloat(xField.value || '0');
      const y = parseFloat(yField.value || '0');
      const w = parseFloat(wField.value || '0');
      const h = parseFloat(hField.value || '0');
      if (w <= 0 || h <= 0) {
        overlay.style.display = 'none';
        return;
      }
      overlay.style.display = 'block';
      overlay.style.left = (x * stage.clientWidth) + 'px';
      overlay.style.top = (y * stage.clientHeight) + 'px';
      overlay.style.width = (w * stage.clientWidth) + 'px';
      overlay.style.height = (h * stage.clientHeight) + 'px';
      const px = Math.round(x * stage.clientWidth);
      const py = Math.round(y * stage.clientHeight);
      const pw = Math.round(w * stage.clientWidth);
      const ph = Math.round(h * stage.clientHeight);
      desc.textContent = 'Custom ROI: ' + pw + ' x ' + ph + ' at (' + px + ', ' + py + ')';
    }

    function wireDrawRoi() {
      const stage = document.getElementById('roi-stage');
      const canvas = document.getElementById('roi-canvas');
      const overlay = document.getElementById('roi-overlay');
      if (!stage || !canvas || !overlay) return;
      let dragging = false;
      let startX = 0;
      let startY = 0;
      canvas.addEventListener('mousedown', (event) => {
        const roiModeField = document.getElementById('roi_mode');
        if (!roiModeField || roiModeField.value !== 'draw') return;
        dragging = true;
        const rect = stage.getBoundingClientRect();
        startX = clamp(event.clientX - rect.left, 0, rect.width);
        startY = clamp(event.clientY - rect.top, 0, rect.height);
        overlay.style.display = 'block';
      });
      window.addEventListener('mousemove', (event) => {
        if (!dragging) return;
        const rect = stage.getBoundingClientRect();
        const currentX = clamp(event.clientX - rect.left, 0, rect.width);
        const currentY = clamp(event.clientY - rect.top, 0, rect.height);
        const left = Math.min(startX, currentX);
        const top = Math.min(startY, currentY);
        const width = Math.abs(currentX - startX);
        const height = Math.abs(currentY - startY);
        overlay.style.left = left + 'px';
        overlay.style.top = top + 'px';
        overlay.style.width = width + 'px';
        overlay.style.height = height + 'px';
      });
      window.addEventListener('mouseup', (event) => {
        if (!dragging) return;
        dragging = false;
        const rect = stage.getBoundingClientRect();
        const currentX = clamp(event.clientX - rect.left, 0, rect.width);
        const currentY = clamp(event.clientY - rect.top, 0, rect.height);
        const left = Math.min(startX, currentX);
        const top = Math.min(startY, currentY);
        const width = Math.abs(currentX - startX);
        const height = Math.abs(currentY - startY);
        if (width < 4 || height < 4) return;
        document.getElementById('roi_x').value = (left / rect.width).toFixed(4);
        document.getElementById('roi_y').value = (top / rect.height).toFixed(4);
        document.getElementById('roi_w').value = (width / rect.width).toFixed(4);
        document.getElementById('roi_h').value = (height / rect.height).toFixed(4);
        updateOverlayFromInputs();
      });
    }

    function openOutputFolder() {
      fetch('{{ url_for("open_output") }}', {method: 'POST'});
    }

    async function cancelRun() {
      const button = document.getElementById('cancel-run-button');
      if (button) {
        button.disabled = true;
        button.textContent = 'Cancelling...';
      }
      await fetch('{{ url_for("cancel_run") }}', {method: 'POST'});
      await refreshStatus();
    }

    async function refreshStatus() {
      const response = await fetch('{{ url_for("status") }}');
      const data = await response.json();
      document.getElementById('task-status').textContent = data.status;
      document.getElementById('task-stage').textContent = data.stage || '';
      document.getElementById('task-detail').textContent = data.status_detail || 'None';
      document.getElementById('task-watchdog').textContent = data.watchdog_status || data.status || '';
      document.getElementById('task-process-alive').textContent = data.process_alive ? 'yes' : 'no';
      document.getElementById('task-last-activity').textContent = String(data.last_activity_seconds ?? 0);
      document.getElementById('task-clip-count').textContent = data.clip_count ?? '';
      document.getElementById('task-strategies').textContent = data.strategies_text || '';
      document.getElementById('task-review-mode').textContent = data.review_mode_label || data.review_mode || '';
      document.getElementById('task-source-mode').textContent = data.source_mode_label || data.source_mode || '';
      document.getElementById('task-preview-mode').textContent = data.preview_mode || '';
      document.getElementById('task-output').textContent = data.output_path || '';
      document.getElementById('task-command').textContent = data.command || '';
      document.getElementById('task-log').textContent = data.log_text || '';
      document.getElementById('task-progress-fill').style.width = (data.progress_percent || 0) + '%';
      document.getElementById('task-counts').textContent = (data.items_completed || 0) + ' / ' + (data.items_total || 0);
      const cancelButton = document.getElementById('cancel-run-button');
      if (cancelButton) {
        const canCancel = !!data.can_cancel || data.status === 'cancelling';
        cancelButton.style.display = canCancel ? 'inline-block' : 'none';
        cancelButton.disabled = data.status === 'cancelling';
        cancelButton.textContent = data.status === 'cancelling' ? 'Cancelling...' : 'Stop Run';
      }
      const stageList = document.getElementById('stage-list');
      stageList.innerHTML = '';
      (data.stage_names || []).forEach((name, index) => {
        const li = document.createElement('li');
        li.className = 'stage-pill' + (index === data.stage_index ? ' active' : '');
        li.textContent = name;
        stageList.appendChild(li);
      });
      const actions = document.getElementById('completion-actions');
      actions.innerHTML = '';
      if (data.report_ready && data.preview_pdf_url) {
        const a = document.createElement('a');
        a.className = 'link-button';
        a.id = 'preview-pdf-link';
        a.href = data.preview_pdf_url;
        a.target = '_blank';
        a.rel = 'noopener';
        a.textContent = 'Open Preview PDF';
        actions.appendChild(a);
      }
      if (data.report_ready && data.preview_html_url) {
        const a = document.createElement('a');
        a.className = 'link-button secondary';
        a.id = 'preview-html-link';
        a.href = data.preview_html_url;
        a.target = '_blank';
        a.rel = 'noopener';
        a.textContent = 'Open HTML Preview';
        actions.appendChild(a);
      }
      if (data.report_ready && data.output_folder) {
        const button = document.createElement('button');
        button.type = 'button';
        button.className = 'secondary';
        button.textContent = 'Open Output Folder';
        button.onclick = openOutputFolder;
        actions.appendChild(button);
      }
      document.getElementById('decision-surface').innerHTML = data.decision_surface_html || '';
      document.getElementById('commit-table-surface').innerHTML = data.commit_table_html || '';
      document.getElementById('push-surface').innerHTML = data.push_surface_html || '';
      document.getElementById('apply-surface').innerHTML = data.apply_surface_html || '';
      document.getElementById('verification-surface').innerHTML = data.verification_surface_html || '';
    }
    applyRoiMode();
    applySourceMode();
    applyReviewMode();
    wireSubsetPanel();
    wireDrawRoi();
    updateOverlayFromInputs();
    setInterval(refreshStatus, 1000);
  </script>
</body>
</html>
"""


def build_review_web_command(repo_root: str, form: Dict[str, object]) -> List[str]:
    source_mode = normalize_source_mode(str(form.get("source_mode", "local_folder")))
    review_mode = normalize_review_mode(str(form.get("review_mode", "full_contact_sheet")))
    matching_domain_value = "perceptual"
    input_path = str(form["input_path"]) if source_mode == "local_folder" else str(Path(str(form["output_path"])).expanduser().resolve() / "ingest")
    args = [
        "python3",
        "-m",
        "r3dmatch.cli",
        "review-calibration",
        input_path,
        "--out",
        str(form["output_path"]),
        "--source-mode",
        source_mode,
        "--backend",
        str(form["backend"]),
        "--target-type",
        str(form["target_type"]),
        "--processing-mode",
        str(form["processing_mode"]),
        "--matching-domain",
        matching_domain_value,
        "--review-mode",
        review_mode,
        "--preview-mode",
        str(form["preview_mode"]),
    ]
    if source_mode == "ftps_camera_pull":
        args.extend(["--ftps-reel", str(form["ftps_reel"]), "--ftps-clips", str(form["ftps_clips"])])
        for camera in [item.strip() for item in str(form.get("ftps_cameras", "")).split(",") if item.strip()]:
            args.extend(["--ftps-camera", camera])
    roi_mode = str(form.get("roi_mode", "sphere_auto"))
    roi_values = [form.get("roi_x"), form.get("roi_y"), form.get("roi_w"), form.get("roi_h")]
    if roi_mode in {"draw", "manual", "center", "full"} and all(value not in (None, "") for value in roi_values):
        args.extend(["--roi-x", str(form["roi_x"]), "--roi-y", str(form["roi_y"]), "--roi-w", str(form["roi_w"]), "--roi-h", str(form["roi_h"])])
    if form.get("run_label"):
        args.extend(["--run-label", str(form["run_label"])])
    for group in form.get("selected_clip_groups", []):
        args.extend(["--clip-group", str(group)])
    for clip in form.get("selected_clip_ids", []):
        args.extend(["--clip-id", str(clip)])
    for strategy in form["target_strategies"]:
        args.extend(["--target-strategy", str(strategy)])
    if form.get("reference_clip_id"):
        args.extend(["--reference-clip-id", str(form["reference_clip_id"])])
    if form.get("hero_clip_id"):
        args.extend(["--hero-clip-id", str(form["hero_clip_id"])])
    if form.get("preview_lut"):
        args.extend(["--preview-lut", str(form["preview_lut"])])
    shell_command = (
        f'cd "{repo_root}"; setenv PYTHONPATH "$PWD/src"; setenv R3DMATCH_INVOCATION_SOURCE web_ui; '
        + " ".join(shlex.quote(item) for item in args)
    )
    return ["/bin/tcsh", "-c", shell_command]


def build_approve_web_command(repo_root: str, form: Dict[str, object]) -> List[str]:
    strategy = form["target_strategies"][0] if form["target_strategies"] else "median"
    output_path = str(form.get("resolved_output_path") or form["output_path"])
    args = [
        "python3",
        "-m",
        "r3dmatch.cli",
        "approve-master-rmd",
        output_path,
        "--target-strategy",
        str(strategy),
    ]
    if form.get("reference_clip_id"):
        args.extend(["--reference-clip-id", str(form["reference_clip_id"])])
    if form.get("hero_clip_id"):
        args.extend(["--hero-clip-id", str(form["hero_clip_id"])])
    shell_command = f'cd "{repo_root}"; setenv PYTHONPATH "$PWD/src"; ' + " ".join(shlex.quote(item) for item in args)
    return ["/bin/tcsh", "-c", shell_command]


def build_clear_cache_web_command(repo_root: str, form: Dict[str, object]) -> List[str]:
    output_path = str(form.get("resolved_output_path") or form["output_path"])
    args = ["python3", "-m", "r3dmatch.cli", "clear-preview-cache", output_path]
    shell_command = f'cd "{repo_root}"; setenv PYTHONPATH "$PWD/src"; ' + " ".join(shlex.quote(item) for item in args)
    return ["/bin/tcsh", "-c", shell_command]


def _camera_label_from_clip_id(clip_id: str) -> str:
    parts = str(clip_id).split("_")
    if len(parts) > 1 and parts[1]:
        return parts[1][0]
    return str(clip_id)[:1] if clip_id else "?"


def scan_sources(input_path: str) -> Dict[str, object]:
    root = Path(input_path).expanduser().resolve()
    if not input_path.strip():
        return {
            "clip_count": 0,
            "clip_ids": [],
            "sample_clip_ids": [],
            "remaining_count": 0,
            "clip_records": [],
            "clip_groups": [],
            "warning": None,
            "preview_note": None,
            "preview_warning": None,
        }
    if not root.exists() or not root.is_dir():
        return {
            "clip_count": 0,
            "clip_ids": [],
            "sample_clip_ids": [],
            "remaining_count": 0,
            "clip_records": [],
            "clip_groups": [],
            "warning": "Calibration folder does not exist.",
            "preview_note": None,
            "preview_warning": None,
        }
    clips = discover_clips(str(root))
    clip_ids = [clip_id_from_path(str(path)) for path in clips]
    clip_records = [
        {
            "clip_id": clip_id,
            "subset_group": subset_key_from_clip_id(clip_id),
            "camera_label": _camera_label_from_clip_id(clip_id),
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
    return {
        "clip_count": len(clips),
        "clip_ids": clip_ids,
        "sample_clip_ids": clip_ids[:12],
        "remaining_count": max(0, len(clip_ids) - 12),
        "clip_records": clip_records,
        "clip_groups": clip_groups,
        "first_clip_path": str(clips[0]) if clips else None,
        "preview_note": None,
        "preview_warning": None,
        "warning": None if clips else "No valid RED .R3D clips were found in the calibration folder.",
    }


def _plan_ftps_scan(form: Dict[str, object]) -> Dict[str, object]:
    try:
        plan = plan_ftps_request(
            reel_identifier=str(form.get("ftps_reel", "")),
            clip_spec=str(form.get("ftps_clips", "")),
            requested_cameras=[item.strip() for item in str(form.get("ftps_cameras", "")).split(",") if item.strip()],
        )
    except Exception as exc:
        return {
            "clip_count": 0,
            "clip_ids": [],
            "sample_clip_ids": [],
            "remaining_count": 0,
            "clip_records": [],
            "clip_groups": [],
            "warning": str(exc),
            "source_mode": "ftps_camera_pull",
            "summary": None,
        }
    return {
        "clip_count": 0,
        "clip_ids": [],
        "sample_clip_ids": [f"{camera} reel {plan['reel_identifier']} clip(s) {plan['clip_spec']}" for camera in plan["requested_cameras"][:12]],
        "remaining_count": max(0, len(plan["requested_cameras"]) - 12),
        "clip_records": [],
        "clip_groups": [],
        "first_clip_path": None,
        "warning": None,
        "source_mode": "ftps_camera_pull",
        "summary": plan,
    }


def _default_form() -> Dict[str, object]:
    return {
        "source_mode": "local_folder",
        "input_path": "",
        "output_path": "",
        "run_label": "",
        "ftps_reel": "",
        "ftps_clips": "",
        "ftps_cameras": "",
        "backend": "red",
        "target_type": "gray_sphere",
        "processing_mode": "both",
        "matching_domain": "perceptual",
        "roi_x": "",
        "roi_y": "",
        "roi_w": "",
        "roi_h": "",
        "selected_clip_ids": [],
        "selected_clip_groups": [],
        "target_strategies": ["median", "optimal-exposure"],
        "reference_clip_id": "",
        "hero_clip_id": "",
        "review_mode": "full_contact_sheet",
        "preview_mode": "monitoring",
        "preview_lut": "",
        "apply_cameras": "",
        "verification_after_path": "",
        "roi_mode": "sphere_auto",
        "advanced_clip_selection": False,
    }


def _parse_form(post_data) -> Dict[str, object]:
    form = _default_form()
    for key in ["source_mode", "input_path", "output_path", "run_label", "ftps_reel", "ftps_clips", "ftps_cameras", "backend", "target_type", "processing_mode", "matching_domain", "roi_x", "roi_y", "roi_w", "roi_h", "reference_clip_id", "hero_clip_id", "review_mode", "preview_mode", "preview_lut", "apply_cameras", "verification_after_path", "roi_mode"]:
        form[key] = post_data.get(key, form[key]).strip()
    strategies = post_data.getlist("target_strategies")
    form["target_strategies"] = strategies or []
    form["selected_clip_ids"] = [str(item).strip() for item in post_data.getlist("selected_clip_ids") if str(item).strip()]
    form["selected_clip_groups"] = [str(item).strip() for item in post_data.getlist("selected_clip_groups") if str(item).strip()]
    form["advanced_clip_selection"] = post_data.get("advanced_clip_selection", "").strip().lower() in {"1", "true", "on", "yes"}
    form["matching_domain"] = "perceptual"
    return form


def _validate_form(form: Dict[str, object], *, require_output: bool = True, require_source: bool = True) -> Optional[str]:
    scan = None
    try:
        resolved_source_mode = normalize_source_mode(str(form.get("source_mode", "local_folder")))
    except ValueError as exc:
        return str(exc)
    if require_source:
        if resolved_source_mode == "local_folder":
            input_path = str(form["input_path"]).strip()
            if not input_path:
                return "Calibration folder path is required."
            scan = scan_sources(input_path)
            if scan["warning"]:
                return str(scan["warning"])
            if not _resolve_selected_clip_ids(form, scan):
                return "Select at least one clip for this calibration run."
        else:
            if not str(form.get("ftps_reel", "")).strip():
                return "FTPS source mode requires a reel identifier."
            if not str(form.get("ftps_clips", "")).strip():
                return "FTPS source mode requires clip numbers or ranges."
            planned = _plan_ftps_scan(form)
            if planned.get("warning"):
                return str(planned["warning"])
    if require_output and not str(form["output_path"]).strip():
        return "Output folder path is required."
    try:
        matching_domain_label(str(form.get("matching_domain", "scene")))
    except ValueError as exc:
        return str(exc)
    try:
        normalize_review_mode(str(form.get("review_mode", "full_contact_sheet")))
    except ValueError as exc:
        return str(exc)
    roi_mode = str(form.get("roi_mode", "sphere_auto"))
    if roi_mode in {"draw", "manual"}:
        roi_values = [str(form.get("roi_x", "")).strip(), str(form.get("roi_y", "")).strip(), str(form.get("roi_w", "")).strip(), str(form.get("roi_h", "")).strip()]
        if not all(roi_values):
            return "ROI is required for draw/manual mode."
    if not form["target_strategies"]:
        return "At least one target strategy is required."
    if "manual" in form["target_strategies"] and not str(form["reference_clip_id"]).strip():
        return "Manual strategy requires a reference clip ID."
    if "hero-camera" in form["target_strategies"] and not str(form["hero_clip_id"]).strip():
        return "Hero-camera strategy requires a hero clip ID."
    if scan is not None:
        selected_clip_ids = _resolve_selected_clip_ids(form, scan)
        if str(form["reference_clip_id"]).strip() and str(form["reference_clip_id"]).strip() not in selected_clip_ids:
            return "Manual reference clip must be included in the selected calibration subset."
        if str(form["hero_clip_id"]).strip() and str(form["hero_clip_id"]).strip() not in selected_clip_ids:
            return "Hero camera must be included in the selected calibration subset."
    preview_lut = str(form["preview_lut"]).strip()
    if preview_lut:
        lut_path = Path(preview_lut).expanduser().resolve()
        if not lut_path.exists():
            return "Preview LUT path does not exist."
    return None


@dataclass
class TaskState:
    status: str = "idle"
    command: str = ""
    output_path: str = ""
    canonical_output_path: str = ""
    source_mode: str = ""
    logs: List[str] = field(default_factory=lambda: ["Ready.\n"])
    returncode: Optional[int] = None
    stage: str = "Idle"
    stage_index: int = 0
    total_stages: int = 0
    items_completed: int = 0
    items_total: int = 0
    clip_count: int = 0
    strategies_text: str = ""
    review_mode: str = ""
    preview_mode: str = ""
    preview_pdf_path: Optional[str] = None
    preview_html_path: Optional[str] = None
    report_ready: bool = False
    cancellation_requested: bool = False
    cancel_file_path: Optional[str] = None
    started_at: float = field(default_factory=time.time)
    last_output_at: float = field(default_factory=time.time)
    last_progress_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    process: Optional[subprocess.Popen[str]] = field(default=None, repr=False, compare=False)
    lock: threading.RLock = field(default_factory=threading.RLock)
    last_finalization_debug_signature: str = ""

    def append(self, text: str) -> None:
        with self.lock:
            self.logs.append(text)
            self.last_output_at = time.time()

    def snapshot(self) -> Dict[str, object]:
        with self.lock:
            process_alive = _process_is_alive(self.process)
            return {
                "status": self.status,
                "command": self.command,
                "output_path": self.output_path,
                "canonical_output_path": self.canonical_output_path,
                "source_mode": self.source_mode,
                "returncode": self.returncode,
                "stage": self.stage,
                "stage_index": self.stage_index,
                "total_stages": self.total_stages,
                "items_completed": self.items_completed,
                "items_total": self.items_total,
                "clip_count": self.clip_count,
                "strategies_text": self.strategies_text,
                "review_mode": self.review_mode,
                "preview_mode": self.preview_mode,
                "preview_pdf_path": self.preview_pdf_path,
                "preview_html_path": self.preview_html_path,
                "report_ready": self.report_ready,
                "cancellation_requested": self.cancellation_requested,
                "cancel_file_path": self.cancel_file_path,
                "process_alive": process_alive,
                "started_at": self.started_at,
                "last_output_at": self.last_output_at,
                "last_progress_at": self.last_progress_at,
                "finished_at": self.finished_at,
                "log_text": "".join(self.logs)[-50000:],
            }


@dataclass
class UiState:
    form: Dict[str, object] = field(default_factory=_default_form)
    scan: Dict[str, object] = field(
        default_factory=lambda: {
            "clip_count": 0,
            "clip_ids": [],
            "sample_clip_ids": [],
            "remaining_count": 0,
            "clip_records": [],
            "clip_groups": [],
            "warning": None,
            "preview_available": False,
            "preview_note": None,
            "preview_warning": None,
        }
    )
    task: TaskState = field(default_factory=TaskState)
    error: Optional[str] = None
    message: Optional[str] = None
    lock: threading.Lock = field(default_factory=threading.Lock)

    def update_form(self, form: Dict[str, object]) -> None:
        with self.lock:
            self.form = form


def _path_is_current(path: Path, *, started_at: Optional[float] = None) -> bool:
    if not path.exists():
        return False
    if started_at is None:
        return True
    try:
        return path.stat().st_mtime >= (float(started_at) - ARTIFACT_FRESHNESS_TOLERANCE_SECONDS)
    except OSError:
        return False


def _validation_timestamp_is_current(
    payload: Dict[str, object],
    path: Path,
    *,
    started_at: Optional[float] = None,
) -> bool:
    if started_at is None:
        return True
    threshold = float(started_at) - ARTIFACT_FRESHNESS_TOLERANCE_SECONDS
    candidate_times: list[float] = []
    validated_at = payload.get("validated_at")
    if isinstance(validated_at, (int, float)):
        candidate_times.append(float(validated_at))
    try:
        candidate_times.append(path.stat().st_mtime)
    except OSError:
        pass
    return bool(candidate_times) and max(candidate_times) >= threshold


def _report_url_for_output(output_path: str, *, started_at: Optional[float] = None) -> Optional[str]:
    if not output_path.strip():
        return None
    report_pdf = review_pdf_path_for(output_path)
    if _path_is_current(report_pdf, started_at=started_at):
        return url_for("artifact", path=str(report_pdf))
    report_html = review_html_path_for(output_path)
    if _path_is_current(report_html, started_at=started_at):
        return url_for("artifact", path=str(report_html))
    return None


def _load_json_path(path: Path) -> Optional[Dict[str, object]]:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _load_review_payload(output_root: Optional[Path], *, started_at: Optional[float] = None) -> Optional[Dict[str, object]]:
    if output_root is None:
        return None
    path = review_payload_path_for(output_root)
    if not _path_is_current(path, started_at=started_at):
        return None
    payload = _load_json_path(path)
    if payload is not None:
        payload["report_json_path"] = str(path)
    return payload


def _load_commit_payload(output_root: Optional[Path], *, started_at: Optional[float] = None) -> Optional[Dict[str, object]]:
    if output_root is None:
        return None
    path = review_commit_payload_path_for(output_root)
    if not _path_is_current(path, started_at=started_at):
        return None
    payload = _load_json_path(path)
    if payload is not None:
        payload["commit_payload_path"] = str(path)
    return payload


def _load_latest_apply_report(output_root: Optional[Path]) -> Optional[Dict[str, object]]:
    if output_root is None:
        return None
    report_root = review_validation_path_for(output_root).parent
    candidates = sorted(report_root.glob("*apply_report.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    for path in candidates:
        payload = _load_json_path(path)
        if payload is None:
            continue
        payload.update(summarize_apply_report(payload, tolerances=operator_apply_tolerances()))
        payload["report_path"] = str(path)
        return payload
    return None


def _camera_state_report_path_for(output_root: Optional[Path]) -> Optional[Path]:
    if output_root is None:
        return None
    return review_validation_path_for(output_root).parent / "rcp2_camera_state_report.json"


def _load_latest_camera_state_report(output_root: Optional[Path]) -> Optional[Dict[str, object]]:
    path = _camera_state_report_path_for(output_root)
    if path is None or not path.exists():
        return None
    payload = _load_json_path(path)
    if payload is not None:
        payload["report_path"] = str(path)
    return payload


def _writeback_verification_report_path_for(output_root: Optional[Path]) -> Optional[Path]:
    if output_root is None:
        return None
    return review_validation_path_for(output_root).parent / "rcp2_verification_report.json"


def _load_latest_writeback_verification_report(output_root: Optional[Path]) -> Optional[Dict[str, object]]:
    path = _writeback_verification_report_path_for(output_root)
    if path is None or not path.exists():
        return None
    payload = _load_json_path(path)
    if payload is not None:
        payload["report_path"] = str(path)
    return payload


def _requested_camera_tokens(form: Dict[str, object]) -> List[str]:
    return [item.strip() for item in str(form.get("apply_cameras", "")).split(",") if item.strip()]


def _read_current_camera_values_report(
    *,
    commit_payload_path: Path,
    report_path: Path,
    requested_cameras: Optional[List[str]] = None,
) -> Dict[str, object]:
    started_at = datetime.now(timezone.utc).isoformat()
    results: List[Dict[str, object]] = []
    targets = load_apply_targets(str(commit_payload_path), requested_cameras=requested_cameras or None)
    for target in targets:
        if not str(target.inventory_camera_ip or "").strip():
            results.append(
                {
                    "inventory_camera_label": target.inventory_camera_label,
                    "inventory_camera_ip": target.inventory_camera_ip,
                    "camera_id": target.camera_id,
                    "clip_id": target.clip_id,
                    "status": "disabled",
                    "error": "No inventory camera IP is mapped for this target.",
                    "state": {},
                }
            )
            continue
        try:
            snapshot = read_camera_state(
                host=target.inventory_camera_ip,
                camera_label=target.inventory_camera_label or target.camera_id or "LIVE",
            )
        except Exception as exc:
            results.append(
                {
                    "inventory_camera_label": target.inventory_camera_label,
                    "inventory_camera_ip": target.inventory_camera_ip,
                    "camera_id": target.camera_id,
                    "clip_id": target.clip_id,
                    "status": "failed_to_connect",
                    "error": str(exc),
                    "state": {},
                }
            )
            continue
        results.append(
            {
                "inventory_camera_label": target.inventory_camera_label,
                "inventory_camera_ip": target.inventory_camera_ip,
                "camera_id": target.camera_id,
                "clip_id": target.clip_id,
                "status": "success",
                "error": "",
                "state": dict(snapshot.get("state") or {}),
                "transport_mode": snapshot.get("transport_mode"),
                "camera_info": dict(((snapshot.get("state") or {}).get("camera_info")) or {}),
            }
        )
    payload = {
        "schema_version": "r3dmatch_rcp2_camera_state_report_v1",
        "started_at": started_at,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "payload_path": str(commit_payload_path),
        "requested_cameras": list(requested_cameras or []),
        "camera_count": len(results),
        "connected_camera_count": sum(1 for row in results if str(row.get("status") or "") == "success"),
        "results": results,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    payload["report_path"] = str(report_path)
    return payload


def _load_post_apply_verification(output_root: Optional[Path]) -> Optional[Dict[str, object]]:
    if output_root is None:
        return None
    payload = _load_json_path(post_apply_verification_path_for(output_root))
    if payload is not None:
        payload["report_path"] = str(post_apply_verification_path_for(output_root))
    return payload


def _commit_readiness_surface(review_validation: Optional[Dict[str, object]]) -> Dict[str, object]:
    if not isinstance(review_validation, dict):
        return {
            "state": "Ready for Review",
            "tone": "pending",
            "reason": "Run review to generate a recommendation, commit payload, and physical validation summary.",
        }
    physical_status = str(((review_validation.get("physical_validation") or {}).get("status")) or "")
    failure_modes = list(review_validation.get("failure_modes") or [])
    warnings = list(review_validation.get("warnings") or [])
    recommendation = dict(review_validation.get("recommendation") or {})
    run_assessment = dict(review_validation.get("run_assessment") or {})
    confidence_score = float(recommendation.get("confidence_score", 0.0) or 0.0)
    failure_codes = {str(item.get("code") or "") for item in failure_modes}
    blocking_codes = {"exposure_out_of_range", "neutrality_failure", "unstable_kelvin"}
    run_status = str(run_assessment.get("status") or "")
    if review_validation.get("status") != "success" or physical_status not in {"success", "warning"}:
        return {
            "state": "Do Not Commit",
            "tone": "danger",
            "reason": "Review validation is not successful enough to commit calibration values safely.",
        }
    if run_status == "DO_NOT_PUSH":
        return {
            "state": "Do Not Commit",
            "tone": "danger",
            "reason": str(run_assessment.get("operator_note") or "The run is too weak to trust for later push."),
        }
    if run_status == "REVIEW_REQUIRED":
        return {
            "state": "Do Not Commit",
            "tone": "danger",
            "reason": str(run_assessment.get("operator_note") or "Review is required before trusting this run."),
        }
    if failure_codes & blocking_codes:
        return {
            "state": "Do Not Commit",
            "tone": "danger",
            "reason": "Blocking physical validation failures are still present in the review results.",
        }
    if run_status == "READY_WITH_WARNINGS":
        return {
            "state": "Commit with Warnings",
            "tone": "warning",
            "reason": str(run_assessment.get("operator_note") or "The run is usable, but some cameras need caution."),
        }
    if "excluded_cameras_present" in failure_codes:
        return {
            "state": "Commit with Warnings",
            "tone": "warning",
            "reason": "The main cluster is still usable, but one or more cameras were excluded from the safe commit set and need separate review.",
        }
    if warnings or "low_confidence" in failure_codes or confidence_score < 0.7:
        return {
            "state": "Commit with Warnings",
            "tone": "warning",
            "reason": "The recommendation is usable, but warnings or lower-confidence measurements should be reviewed before pushing values.",
        }
    return {
        "state": "Ready for Commit",
        "tone": "success",
        "reason": "Physical validation passed and the current recommendation is ready to export or apply.",
    }


def _build_operator_surfaces(
    *,
    review_validation: Optional[Dict[str, object]],
    review_payload: Optional[Dict[str, object]],
    commit_payload: Optional[Dict[str, object]],
    camera_state_report: Optional[Dict[str, object]],
    apply_report: Optional[Dict[str, object]],
    writeback_verification_report: Optional[Dict[str, object]],
    post_apply_report: Optional[Dict[str, object]],
    source_mode_label_text: str,
) -> Dict[str, object]:
    recommendation = dict((review_validation or {}).get("recommendation") or {})
    physical = dict((review_validation or {}).get("physical_validation") or {})
    run_assessment = dict((review_validation or {}).get("run_assessment") or (review_payload or {}).get("run_assessment") or {})
    wb_model = dict((review_payload or {}).get("white_balance_model") or {})
    wb_candidates = list(wb_model.get("candidates") or [])
    chosen_model_key = str(wb_model.get("model_key") or "")
    chosen_candidate = next((item for item in wb_candidates if str(item.get("model_key") or "") == chosen_model_key), {})
    commit_rows: List[Dict[str, object]] = []
    if isinstance(commit_payload, dict):
        per_camera_note_map: Dict[str, Dict[str, object]] = {}
        for item in commit_payload.get("per_camera_payloads", []) or []:
            payload = _load_json_path(Path(str(item.get("path") or "")).expanduser())
            if isinstance(payload, dict):
                per_camera_note_map[str(payload.get("camera_id") or "")] = payload
        for row in commit_payload.get("camera_targets", []) or []:
            payload = per_camera_note_map.get(str(row.get("camera_id") or ""), {})
            commit_rows.append(
                {
                    "inventory_camera_label": str(row.get("inventory_camera_label") or ""),
                    "inventory_camera_ip": str(row.get("inventory_camera_ip") or ""),
                    "camera_id": str(row.get("camera_id") or ""),
                    "clip_id": str(row.get("clip_id") or ""),
                    "source_path": str(payload.get("source_path") or ""),
                    "calibration": dict(row.get("calibration") or {}),
                    "confidence": float(row.get("confidence", 0.0) or 0.0),
                    "trust_class": str(row.get("trust_class") or payload.get("trust_class") or ""),
                    "trust_reason": str(row.get("trust_reason") or payload.get("trust_reason") or ""),
                    "reference_use": str(row.get("reference_use") or payload.get("reference_use") or ""),
                    "correction_confidence": str(row.get("correction_confidence") or payload.get("correction_confidence") or ""),
                    "notes": [str(note) for note in payload.get("notes", []) if str(note).strip()],
                    "excluded_from_commit": bool(row.get("excluded_from_commit")),
                    "exclusion_reasons": [str(reason) for reason in row.get("exclusion_reasons", []) if str(reason).strip()],
                    "safe_to_commit": not bool(row.get("excluded_from_commit")),
                }
            )
        commit_rows.sort(key=_severity_for_commit_row)
    verification_surface = post_apply_report or dict((review_validation or {}).get("post_apply_validation") or {})
    camera_state_lookup = _current_state_lookup(camera_state_report)
    apply_lookup = _apply_result_lookup(apply_report)
    push_rows: List[Dict[str, object]] = []
    for row in commit_rows:
        calibration = dict(row.get("calibration") or {})
        current_row = dict(camera_state_lookup.get(str(row.get("inventory_camera_label") or "")) or {})
        current_state = dict(current_row.get("state") or {})
        apply_row = dict(apply_lookup.get(str(row.get("inventory_camera_label") or "")) or {})
        current_exposure = current_state.get("exposureAdjust")
        current_kelvin = current_state.get("kelvin")
        current_tint = current_state.get("tint")
        new_exposure = float(calibration.get("exposureAdjust", 0.0) or 0.0)
        new_kelvin = int(calibration.get("kelvin", 0) or 0)
        new_tint = float(calibration.get("tint", 0.0) or 0.0)
        if not str(row.get("inventory_camera_ip") or "").strip():
            writeback_status = "Disabled"
            writeback_tone = "pending"
        elif str((apply_row.get("operator_result") or {}).get("display_status") or "") in {"applied_successfully", "applied_with_deviation"}:
            writeback_status = "Verified"
            writeback_tone = "success"
        elif str(apply_row.get("status") or "") in {"failed_to_connect", "timeout"}:
            writeback_status = "Not Reachable"
            writeback_tone = "danger"
        elif str(apply_row.get("status") or "") in {"rejected_by_camera", "mismatch_after_writeback"}:
            writeback_status = "Failed"
            writeback_tone = "danger"
        elif current_row:
            writeback_status = "Warning" if row.get("excluded_from_commit") else "Ready"
            writeback_tone = "warning" if row.get("excluded_from_commit") else "success"
        else:
            writeback_status = "Ready" if not row.get("excluded_from_commit") else "Warning"
            writeback_tone = "success" if not row.get("excluded_from_commit") else "warning"
        note_bits = [
            *[str(item) for item in row.get("exclusion_reasons", []) if str(item).strip()],
            *[str(item) for item in row.get("notes", []) if str(item).strip()],
            str(current_row.get("error") or "").strip(),
            str(apply_row.get("error") or "").strip(),
        ]
        push_rows.append(
            {
                **row,
                "current_state": current_state,
                "current_exposure": float(current_exposure) if current_exposure is not None else None,
                "current_kelvin": int(current_kelvin) if current_kelvin is not None else None,
                "current_tint": float(current_tint) if current_tint is not None else None,
                "new_exposure": new_exposure,
                "new_kelvin": new_kelvin,
                "new_tint": new_tint,
                "exposure_delta": (new_exposure - float(current_exposure)) if current_exposure is not None else None,
                "tint_delta": (new_tint - float(current_tint)) if current_tint is not None else None,
                "reference_use": "Excluded" if row.get("excluded_from_commit") else "Included",
                "writeback_status": writeback_status,
                "writeback_tone": writeback_tone,
                "notes_text": "; ".join(item for item in note_bits if item) or "Ready for camera sync.",
                "compact_summary": f"{str(row.get('inventory_camera_label') or '')} → Exposure {_format_signed_value(new_exposure, decimals=2)}, Kelvin {new_kelvin}, Tint {_format_signed_value(new_tint, decimals=1)}",
            }
        )
    return {
        "review_payload": review_payload or {},
        "recommendation": {
            "strategy_label": str(((recommendation.get("recommended_strategy") or {}).get("strategy_label")) or "Pending"),
            "strategy_key": str(((recommendation.get("recommended_strategy") or {}).get("strategy_key")) or ""),
            "reason": str(((recommendation.get("recommended_strategy") or {}).get("reason")) or "No recommendation available yet."),
            "hero_camera": recommendation.get("hero_camera"),
            "hero_camera_confidence": recommendation.get("hero_camera_confidence"),
            "confidence_score": float(recommendation.get("confidence_score", 0.0) or 0.0),
            "recommendation_strength": str(recommendation.get("recommendation_strength") or run_assessment.get("recommendation_strength") or ""),
            "run_status": str(recommendation.get("run_status") or run_assessment.get("status") or ""),
            "summary_notes": [str(item) for item in recommendation.get("summary_notes", []) if str(item).strip()],
            "source_mode_label": source_mode_label_text,
        },
        "run_assessment": run_assessment,
        "white_balance_model": {
            "model_label": str(wb_model.get("model_label") or "n/a"),
            "model_key": chosen_model_key,
            "shared_kelvin": wb_model.get("shared_kelvin"),
            "shared_tint": wb_model.get("shared_tint"),
            "candidate_count": len(wb_candidates),
            "selection_reason": (
                f"{str(wb_model.get('model_label') or 'Selected model')} was chosen because it kept the weighted post-neutral residual at "
                f"{float((chosen_candidate.get('metrics') or {}).get('weighted_mean_post_neutral_residual', 0.0) or 0.0):.4f} "
                f"while holding Kelvin scatter to {float((chosen_candidate.get('metrics') or {}).get('kelvin_axis_stddev', 0.0) or 0.0):.4f}."
                if chosen_candidate
                else "No white-balance model summary is available yet."
            ),
            "candidates": wb_candidates,
        },
        "physical_validation": {
            "status": str(physical.get("status") or "pending"),
            "exposure": dict(physical.get("exposure") or {}),
            "neutrality": dict(physical.get("neutrality") or {}),
            "kelvin_tint_analysis": dict(physical.get("kelvin_tint_analysis") or {}),
            "confidence": dict(physical.get("confidence") or {}),
            "outliers": list(physical.get("outliers") or []),
            "excluded_cameras": list(physical.get("excluded_cameras") or []),
            "excluded_camera_count": int(physical.get("excluded_camera_count", 0) or 0),
            "included_camera_count": int(physical.get("included_camera_count", 0) or 0),
            "cluster_is_usable_after_exclusions": bool(physical.get("cluster_is_usable_after_exclusions")),
            "warnings": [str(item) for item in physical.get("warnings", []) if str(item).strip()],
        },
        "commit_readiness": _commit_readiness_surface(review_validation),
        "commit_table": {
            "camera_count": len(commit_rows),
            "rows": commit_rows,
        },
        "push_surface": {
            "payload_path": str((commit_payload or {}).get("commit_payload_path") or ""),
            "strategy_label": str(((recommendation.get("recommended_strategy") or {}).get("strategy_label")) or "Pending"),
            "strategy_key": str(((recommendation.get("recommended_strategy") or {}).get("strategy_key")) or ""),
            "camera_count": len(push_rows),
            "connected_camera_count": sum(1 for row in push_rows if row.get("current_state")),
            "readback_verification_state": str(
                (writeback_verification_report or {}).get("status")
                or (apply_report or {}).get("operator_status")
                or "not_run"
            ),
            "verification_mode": str((writeback_verification_report or {}).get("verification_mode") or ""),
            "rows": push_rows,
            "compact_summaries": [str(row.get("compact_summary") or "") for row in push_rows if str(row.get("compact_summary") or "")],
            "current_report_path": str((camera_state_report or {}).get("report_path") or ""),
            "verification_report_path": str((writeback_verification_report or {}).get("report_path") or ""),
        },
        "lightweight_visuals": dict((review_payload or {}).get("visuals") or {}),
        "strategy_comparison": list((review_payload or {}).get("strategy_comparison") or []),
        "camera_state_report": camera_state_report or {},
        "writeback_verification_report": writeback_verification_report or {},
        "apply_surface": apply_report or {},
        "verification_surface": verification_surface or {},
        "failure_modes": list((review_validation or {}).get("failure_modes") or []),
    }


def _badge_html(label: str, tone: str) -> str:
    return f'<span class="surface-badge {html.escape(tone)}">{html.escape(label)}</span>'


def _chart_frame_html(title: str, subtitle: str, svg: str) -> str:
    return (
        f"<div class='chart-launch' role='button' tabindex='0' data-chart-title='{html.escape(title)}'>"
        f"<div class='chart-frame'>{svg}</div>"
        f"<span class='chart-launch-hint'>{html.escape(subtitle)}</span>"
        "</div>"
    )


def _format_signed_value(value: Optional[float], *, decimals: int = 2, suffix: str = "") -> str:
    if value is None:
        return "—"
    numeric = float(value or 0.0)
    sign = "+" if numeric > 0 else ""
    return f"{sign}{numeric:.{decimals}f}{suffix}"


def _format_delta_html(value: Optional[float], *, decimals: int = 2, suffix: str = "") -> str:
    if value is None:
        return "<span class='delta-neutral'>—</span>"
    numeric = float(value or 0.0)
    tone = "delta-positive" if numeric > 0 else "delta-negative" if numeric < 0 else "delta-neutral"
    return f"<span class='{tone}'>{html.escape(_format_signed_value(numeric, decimals=decimals, suffix=suffix))}</span>"


def _current_state_lookup(camera_state_report: Optional[Dict[str, object]]) -> Dict[str, Dict[str, object]]:
    lookup: Dict[str, Dict[str, object]] = {}
    for row in list((camera_state_report or {}).get("results") or []):
        key = str(row.get("inventory_camera_label") or "")
        if key:
            lookup[key] = dict(row)
    return lookup


def _apply_result_lookup(apply_report: Optional[Dict[str, object]]) -> Dict[str, Dict[str, object]]:
    lookup: Dict[str, Dict[str, object]] = {}
    for row in list((apply_report or {}).get("results") or []):
        key = str(row.get("inventory_camera_label") or "")
        if key:
            lookup[key] = dict(row)
    return lookup


def _severity_for_commit_row(row: Dict[str, object]) -> tuple[int, float, float]:
    notes = " ".join(str(item) for item in row.get("notes", []) if str(item).strip()).lower()
    calibration = dict(row.get("calibration") or {})
    confidence = float(row.get("confidence", 0.0) or 0.0)
    exposure_adjust = abs(float(calibration.get("exposureAdjust", 0.0) or 0.0))
    trust_class = str(row.get("trust_class") or "")
    if trust_class == "EXCLUDED":
        return (0, -confidence, -exposure_adjust)
    if trust_class == "UNTRUSTED":
        return (1, -confidence, -exposure_adjust)
    if trust_class == "USE_WITH_CAUTION":
        return (2, -confidence, -exposure_adjust)
    if row.get("excluded_from_commit"):
        return (0, -confidence, -exposure_adjust)
    if "outlier" in notes or confidence < 0.5 or exposure_adjust >= 1.0:
        return (1, -confidence, -exposure_adjust)
    if confidence < 0.75 or exposure_adjust >= 0.5:
        return (2, -confidence, -exposure_adjust)
    return (3, -confidence, -exposure_adjust)


def _status_meta_for_commit_row(row: Dict[str, object]) -> tuple[str, str]:
    trust_class = str(row.get("trust_class") or "")
    if trust_class == "EXCLUDED":
        return ("Excluded", "outlier")
    if trust_class == "UNTRUSTED":
        return ("Untrusted", "warning")
    if trust_class == "USE_WITH_CAUTION":
        return ("Caution", "warning")
    if trust_class == "TRUSTED":
        return ("Trusted", "good")
    notes = " ".join(str(item) for item in row.get("notes", []) if str(item).strip()).lower()
    calibration = dict(row.get("calibration") or {})
    confidence = float(row.get("confidence", 0.0) or 0.0)
    exposure_adjust = abs(float(calibration.get("exposureAdjust", 0.0) or 0.0))
    if row.get("excluded_from_commit"):
        return ("Excluded", "outlier")
    if "outlier" in notes or confidence < 0.5 or exposure_adjust >= 1.0:
        return ("Warning", "warning")
    return ("Good", "good")


def _reason_label_for_commit_row(row: Dict[str, object]) -> str:
    if str(row.get("trust_reason") or "").strip():
        return str(row.get("trust_reason") or "")
    notes = " ".join(str(item) for item in row.get("notes", []) if str(item).strip()).lower()
    confidence = float(row.get("confidence", 0.0) or 0.0)
    calibration = dict(row.get("calibration") or {})
    exposure_adjust = float(calibration.get("exposureAdjust", 0.0) or 0.0)
    if row.get("excluded_from_commit"):
        return "Excluded"
    if exposure_adjust <= -0.75:
        return "Underexposed"
    if confidence < 0.5:
        return "Low confidence"
    return "Stable"


def _bullet_list_html(items: List[str]) -> str:
    cleaned = [str(item).strip() for item in items if str(item).strip()]
    if not cleaned:
        return "<ul class='surface-list'><li>No additional notes.</li></ul>"
    return "<ul class='surface-list'>" + "".join(f"<li>{html.escape(item)}</li>" for item in cleaned) + "</ul>"


def _calibration_overview_from_payload(
    review_payload: Optional[Dict[str, object]],
    run_assessment: Dict[str, object],
    readiness: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    rows = [dict(item) for item in list((review_payload or {}).get("per_camera_analysis") or [])]
    retained_rows = [row for row in rows if str(row.get("reference_use") or "").strip().lower() != "excluded"]
    sample_2_values = [
        float(row.get("sample_2_ire"))
        for row in retained_rows
        if row.get("sample_2_ire") is not None
    ]
    best_reference = None
    if retained_rows:
        best_reference = min(
            retained_rows,
            key=lambda row: (
                abs(float(row.get("camera_offset_from_anchor", row.get("final_offset_stops", 0.0)) or 0.0)),
                -float(row.get("trust_score", 0.0) or 0.0),
                -float(row.get("confidence", 0.0) or 0.0),
                str(row.get("camera_label") or row.get("clip_id") or ""),
            ),
        )
    status = str(run_assessment.get("status") or "")
    readiness_state = str((readiness or {}).get("state") or "")
    if not status:
        status = {
            "Ready for Commit": "READY",
            "Commit with Warnings": "READY_WITH_WARNINGS",
            "Do Not Commit": "DO_NOT_PUSH",
            "Ready for Review": "PENDING",
        }.get(readiness_state, "")
    if status == "READY":
        state_title = "Array is within calibration tolerance"
        state_tone = "success"
        action_line = "Calibration payload is ready for review and camera sync."
    elif status == "READY_WITH_WARNINGS":
        state_title = "Array is near calibration tolerance"
        state_tone = "warning"
        action_line = "Review excluded or unstable cameras before normalizing the array."
    elif status in {"DO_NOT_PUSH", "REVIEW_REQUIRED"}:
        state_title = "Array is out of calibration"
        state_tone = "danger"
        action_line = "Review the retained cluster before normalizing the array."
    else:
        state_title = "Array calibration review pending"
        state_tone = "pending"
        action_line = "Run review to measure the current gray-sphere alignment."
    range_text = "n/a"
    if sample_2_values:
        range_text = f"{min(sample_2_values):.0f} IRE to {max(sample_2_values):.0f} IRE"
    return {
        "state_title": state_title,
        "state_tone": state_tone,
        "action_line": action_line,
        "retained_count": len(retained_rows),
        "excluded_count": max(0, len(rows) - len(retained_rows)),
        "sample_2_ire_min": min(sample_2_values) if sample_2_values else None,
        "sample_2_ire_max": max(sample_2_values) if sample_2_values else None,
        "sample_2_ire_range_text": range_text,
        "best_reference_camera": str((best_reference or {}).get("camera_label") or "n/a"),
        "best_reference_clip_id": str((best_reference or {}).get("clip_id") or ""),
        "best_reference_summary": str((best_reference or {}).get("measured_gray_exposure_summary") or "n/a"),
        "best_reference_offset": float((best_reference or {}).get("camera_offset_from_anchor", (best_reference or {}).get("final_offset_stops", 0.0)) or 0.0),
        "best_reference_reason": (
            "Closest fit to the retained cluster target."
            if best_reference
            else "No retained camera candidate is available yet."
        ),
    }


def _render_decision_surface(surface: Dict[str, object]) -> str:
    recommendation = dict(surface.get("recommendation") or {})
    run_assessment = dict(surface.get("run_assessment") or {})
    wb_model = dict(surface.get("white_balance_model") or {})
    readiness = dict(surface.get("commit_readiness") or {})
    physical = dict(surface.get("physical_validation") or {})
    exposure = dict(physical.get("exposure") or {})
    neutrality = dict(physical.get("neutrality") or {})
    kelvin_tint = dict(physical.get("kelvin_tint_analysis") or {})
    confidence = dict(physical.get("confidence") or {})
    failure_modes = list(surface.get("failure_modes") or [])
    visuals = dict(surface.get("lightweight_visuals") or {})
    excluded_cameras = list(physical.get("excluded_cameras") or [])
    overview = _calibration_overview_from_payload(dict(surface.get("review_payload") or {}), run_assessment, readiness)
    readiness_tone = str(overview.get("state_tone") or readiness.get("tone") or "pending")
    shared_kelvin_text = "per-camera" if wb_model.get("shared_kelvin") is None else str(wb_model.get("shared_kelvin"))
    shared_tint_text = "per-camera" if wb_model.get("shared_tint") is None else str(wb_model.get("shared_tint"))
    failure_html = "".join(
        f"<li><strong>{html.escape(str(item.get('code') or 'warning'))}</strong>: {html.escape(str(item.get('message') or ''))}</li>"
        for item in failure_modes
    ) or "<li>No blocking failure modes reported.</li>"
    strategy_label = str(recommendation.get("strategy_label") or "Pending")
    strategy_key = str(recommendation.get("strategy_key") or "")
    recommendation_strength = str(recommendation.get("recommendation_strength") or run_assessment.get("recommendation_strength") or "MEDIUM_CONFIDENCE")
    run_status = str(recommendation.get("run_status") or run_assessment.get("status") or "")
    optimal_exposure_summary = dict(
        next(
            (
                item
                for item in list(surface.get("strategy_comparison") or [])
                if str(item.get("strategy_key") or "") == "optimal_exposure"
            ),
            {},
        )
        or {}
    )
    optimal_reference = str(optimal_exposure_summary.get("reference_clip_id") or "No single camera")
    optimal_anchor_reason = str((optimal_exposure_summary.get("selection_diagnostics") or {}).get("anchor_reason") or "")
    excluded_count = int(overview.get("excluded_count", 0) or 0)
    decision_subline = (
        f"Retained gray sphere Sample 2 values span {overview.get('sample_2_ire_range_text', 'n/a')} across {int(overview.get('retained_count', 0) or 0)} retained cameras."
        if int(overview.get("retained_count", 0) or 0) > 0
        else "No retained cameras are available for a calibration summary yet."
    )
    summary_sentence = (
        f"Closest current reference candidate: {overview.get('best_reference_camera', 'n/a')}."
        if str(overview.get("best_reference_camera") or "").strip() and str(overview.get("best_reference_camera")) != "n/a"
        else "Reference candidate will appear after review measures the retained camera cluster."
    )
    why_chosen_points = [
        "Minimizes correction spread across the stable cameras." if strategy_key == "median" else "Matches the camera already closest to correct gray exposure.",
        "Keeps neutral error low across the array.",
        "Avoids extreme adjustments on the remaining cluster." if strategy_key == "median" else "Uses the most trustworthy gray-match camera instead of the brightest image.",
        str(recommendation.get("reason") or ""),
        *[str(note) for note in recommendation.get("summary_notes", []) if str(note).strip()],
    ]
    if strategy_key == "median":
        why_chosen_points.insert(
            0,
            "Median was chosen as a fallback because no single camera was trustworthy enough to anchor the array."
            if optimal_anchor_reason and "fell back" in optimal_anchor_reason.lower()
            else "Median was chosen because it kept the group more stable than a single-camera anchor.",
        )
    elif optimal_anchor_reason:
        why_chosen_points.insert(0, optimal_anchor_reason)
    if strategy_key != "optimal_exposure" and optimal_reference:
        why_chosen_points.append(
            f"Optimal Exposure checked {optimal_reference} as the closest gray match, but it was not selected because {optimal_anchor_reason.lower() or 'it would have pushed the rest of the group harder'}."
        )
    wb_points = [
        str(wb_model.get("selection_reason") or ""),
        f"Shared Kelvin: {shared_kelvin_text}.",
        f"Shared Tint Anchor: {shared_tint_text}.",
    ]
    validation_points = [
        f"Retained gray sphere Sample 2 range: {overview.get('sample_2_ire_range_text', 'n/a')}.",
        (
            f"Closest current reference candidate: {overview.get('best_reference_camera')} ({overview.get('best_reference_summary')})."
            if str(overview.get("best_reference_camera") or "").strip() and str(overview.get("best_reference_camera")) != "n/a"
            else "No retained reference candidate is available yet."
        ),
        f"Mean exposure error: {float(exposure.get('mean_exposure_error', 0.0) or 0.0):.4f}.",
        f"Max neutral error: {float(neutrality.get('max_post_neutral_error', 0.0) or 0.0):.4f}.",
        f"Kelvin stability: {'stable' if kelvin_tint.get('kelvin_is_stable') else 'review required'}.",
        f"Tint variation: {'present' if kelvin_tint.get('tint_carries_variation') else 'flat'}.",
        (
            "The remaining cluster is still commit-worthy after exclusions."
            if physical.get("cluster_is_usable_after_exclusions")
            else "No exclusion-based cluster rescue is currently available."
        ),
    ]
    trust_points = [
        str(run_assessment.get("anchor_summary") or "No anchor summary available."),
        f"Retained cameras: {int(overview.get('retained_count', 0) or 0)} / {int(run_assessment.get('camera_count', 0) or 0)}.",
        f"Excluded cameras: {excluded_count}.",
        str(overview.get("best_reference_reason") or ""),
        *[str(item) for item in run_assessment.get("gating_reasons", []) if str(item).strip()],
    ]
    decision_title = str(overview.get("state_title") or "Array calibration review pending")
    banner_subtitle = str(overview.get("action_line") or readiness.get("reason") or "Run review to generate a calibration summary.")
    visuals_html = ""
    if visuals:
        chart_cards = []
        if visuals.get("exposure_plot_svg"):
            chart_cards.append(
                "<div class='summary-panel'>"
                "<div class='surface-heading-row'><h3>Exposure Spread</h3></div>"
                "<p class='surface-copy'>Brighter cameras appear higher. The central blue band marks the stable cluster.</p>"
                f"{_chart_frame_html('Exposure Spread', 'Click to enlarge', str(visuals.get('exposure_plot_svg') or ''))}"
                "</div>"
            )
        if visuals.get("before_after_exposure_svg"):
            chart_cards.append(
                "<div class='summary-panel'>"
                "<div class='surface-heading-row'><h3>Before / After Exposure</h3></div>"
                "<p class='surface-copy'>Each line shows the measured camera exposure and the target the recommendation pulls it toward.</p>"
                f"{_chart_frame_html('Before / After Exposure', 'Click to enlarge', str(visuals.get('before_after_exposure_svg') or ''))}"
                "</div>"
            )
        if visuals.get("confidence_chart_svg"):
            chart_cards.append(
                "<div class='summary-panel'>"
                "<div class='surface-heading-row'><h3>Confidence / Reliability</h3></div>"
                "<p class='surface-copy'>Higher bars indicate steadier neutral samples and more trustworthy measurements.</p>"
                f"{_chart_frame_html('Confidence / Reliability', 'Click to enlarge', str(visuals.get('confidence_chart_svg') or ''))}"
                "</div>"
            )
        if visuals.get("strategy_chart_svg"):
            chart_cards.append(
                "<div class='summary-panel'>"
                "<div class='surface-heading-row'><h3>Strategy Comparison</h3></div>"
                "<p class='surface-copy'>This chart shows why the system favored the winning strategy over brighter but riskier anchors.</p>"
                f"{_chart_frame_html('Strategy Comparison', 'Click to enlarge', str(visuals.get('strategy_chart_svg') or ''))}"
                "</div>"
            )
        if visuals.get("trust_chart_svg"):
            chart_cards.append(
                "<div class='summary-panel'>"
                "<div class='surface-heading-row'><h3>Camera Trust</h3></div>"
                "<p class='surface-copy'>Trust classes summarize confidence, stability, cluster membership, and correction size.</p>"
                f"{_chart_frame_html('Camera Trust', 'Click to enlarge', str(visuals.get('trust_chart_svg') or ''))}"
                "</div>"
            )
        if chart_cards:
            visuals_html = f'<div class="summary-grid" style="margin-top: 16px;">{"".join(chart_cards)}</div>'
    return f"""
    <div class="decision-banner {html.escape(readiness_tone)}">
      <div class="decision-banner-kicker">Array Calibration Review</div>
      <div class="decision-banner-title">{html.escape(decision_title)}</div>
      <p class="decision-banner-subtitle">{html.escape(banner_subtitle)}</p>
      <p class="decision-banner-subtitle" style="margin-top:8px;font-size:15px;font-weight:700;">{html.escape(decision_subline)}</p>
      <div class="decision-banner-metrics">
        <div><span class="metric-label">Sample 2 IRE Range</span><span class="metric-value">{html.escape(str(overview.get('sample_2_ire_range_text') or 'n/a'))}</span></div>
        <div><span class="metric-label">Retained Cameras</span><span class="metric-value">{int(overview.get('retained_count', 0) or 0)}</span></div>
        <div><span class="metric-label">Excluded Cameras</span><span class="metric-value">{excluded_count}</span></div>
        <div><span class="metric-label">Reference Candidate</span><span class="metric-value">{html.escape(str(overview.get('best_reference_camera') or 'n/a'))}</span></div>
        <div><span class="metric-label">Strategy</span><span class="metric-value">{html.escape(strategy_label)}</span></div>
        <div><span class="metric-label">Review State</span><span class="metric-value">{html.escape(run_status.replace('_', ' ') or 'pending')}</span></div>
      </div>
    </div>
    <div class="summary-grid">
      <div class="summary-panel">
        <div class="surface-heading-row">
          <h3>Calibration Recommendation</h3>
          {_badge_html(str(overview.get("state_title") or "Review pending"), readiness_tone)}
        </div>
        <p class="surface-kicker">At A Glance</p>
        <p class="surface-lead">{html.escape(summary_sentence)}</p>
        <div class="surface-metrics">
          <div><span class="metric-label">Reference Candidate</span><span class="metric-value">{html.escape(str(overview.get('best_reference_camera') or 'n/a'))}</span></div>
          <div><span class="metric-label">Candidate Profile</span><span class="metric-value">{html.escape(str(overview.get('best_reference_summary') or 'n/a'))}</span></div>
          <div><span class="metric-label">Source Mode</span><span class="metric-value">{html.escape(str(recommendation.get('source_mode_label') or ''))}</span></div>
        </div>
        <p class="surface-kicker">Why This Was Chosen</p>
        {_bullet_list_html(why_chosen_points)}
      </div>
      <div class="summary-panel">
        <div class="surface-heading-row">
          <h3>White Balance Model</h3>
          {_badge_html(html.escape(str(wb_model.get('model_label') or 'n/a')), 'info')}
        </div>
        <p class="surface-kicker">Model Selection</p>
        <p class="surface-lead">{html.escape(str(wb_model.get('model_label') or 'n/a'))}</p>
        <div class="surface-metrics">
          <div><span class="metric-label">Shared Kelvin</span><span class="metric-value">{html.escape(shared_kelvin_text)}</span></div>
          <div><span class="metric-label">Shared Tint Anchor</span><span class="metric-value">{html.escape(shared_tint_text)}</span></div>
          <div><span class="metric-label">Models Compared</span><span class="metric-value">{int(wb_model.get('candidate_count', 0) or 0)}</span></div>
        </div>
        {_bullet_list_html(wb_points)}
      </div>
      <div class="summary-panel">
        <div class="surface-heading-row">
          <h3>Physical Validation</h3>
          {_badge_html(str(physical.get('status') or 'pending'), 'success' if str(physical.get('status') or '') == 'success' else 'warning')}
        </div>
        <p class="surface-kicker">Subset Health</p>
        <p class="surface-lead">{'Retained cluster is usable' if physical.get('cluster_is_usable_after_exclusions') else 'Retained cluster needs review'}</p>
        <div class="surface-metrics">
          <div><span class="metric-label">Mean Exposure Error</span><span class="metric-value">{float(exposure.get('mean_exposure_error', 0.0) or 0.0):.4f}</span></div>
          <div><span class="metric-label">Max Neutral Error</span><span class="metric-value">{float(neutrality.get('max_post_neutral_error', 0.0) or 0.0):.4f}</span></div>
          <div><span class="metric-label">Kelvin Stability</span><span class="metric-value">{'stable' if kelvin_tint.get('kelvin_is_stable') else 'review'}</span></div>
          <div><span class="metric-label">Tint Variation</span><span class="metric-value">{'present' if kelvin_tint.get('tint_carries_variation') else 'flat'}</span></div>
          <div><span class="metric-label">Measurement Stability</span><span class="metric-value">{float(confidence.get('mean_confidence', 0.0) or 0.0):.2f}</span></div>
          <div><span class="metric-label">Outliers</span><span class="metric-value">{len(list(physical.get('outliers') or []))}</span></div>
          <div><span class="metric-label">Excluded Cameras</span><span class="metric-value">{excluded_count}</span></div>
        </div>
        {_bullet_list_html(validation_points)}
        <p class="surface-kicker" style="margin-top:14px;">Warnings and Exclusions</p>
        <ul class="surface-list">{failure_html}</ul>
        {_bullet_list_html([f"{str(item.get('camera_id') or item.get('clip_id') or 'camera')}: {', '.join(str(reason) for reason in item.get('reasons', []) if str(reason).strip()) or 'review required'}" for item in excluded_cameras] or ['No cameras are currently excluded from safe commit export.'])}
      </div>
      <div class="summary-panel">
        <div class="surface-heading-row">
          <h3>Calibration Review Notes</h3>
          {_badge_html(run_status.replace('_', ' ') or 'Pending', readiness_tone)}
        </div>
        <p class="surface-kicker">Operator Guidance</p>
        <p class="surface-lead">{html.escape(str(run_assessment.get('operator_note') or 'Review the retained camera group before normalizing the array.'))}</p>
        {_bullet_list_html(trust_points)}
      </div>
    </div>
    {visuals_html}
    """


def _render_commit_table(surface: Dict[str, object]) -> str:
    rows = list((surface.get("commit_table") or {}).get("rows") or [])
    if not rows:
        return '<div class="empty-state">Run review to generate the per-camera commit table.</div>'
    body = []
    for row in rows:
        calibration = dict(row.get("calibration") or {})
        notes = "; ".join(str(item) for item in row.get("notes", []) if str(item).strip()) or "None"
        status_label, status_tone = _status_meta_for_commit_row(row)
        reason_label = _reason_label_for_commit_row(row)
        camera_badges = _badge_html(status_label, "danger" if status_tone == "outlier" else status_tone)
        body.append(
            "<tr>"
            f"<td><span class='table-status'><span class='status-dot {html.escape(status_tone)}'></span>{camera_badges}</span><div style='margin-top:8px;font-weight:800;'>{html.escape(str(row.get('inventory_camera_label') or ''))}</div><div style='margin-top:6px;color:#64748b;font-size:12px;font-weight:700;text-transform:uppercase;letter-spacing:0.05em;'>{html.escape(reason_label)}</div></td>"
            f"<td>{html.escape(str(row.get('inventory_camera_ip') or ''))}</td>"
            f"<td>{html.escape(str(row.get('clip_id') or row.get('source_path') or ''))}</td>"
            f"<td>{html.escape(_format_signed_value(float(calibration.get('exposureAdjust', 0.0) or 0.0), decimals=3))}</td>"
            f"<td>{int(calibration.get('kelvin', 0) or 0)}</td>"
            f"<td>{html.escape(_format_signed_value(float(calibration.get('tint', 0.0) or 0.0), decimals=2))}</td>"
            f"<td>{float(row.get('confidence', 0.0) or 0.0):.2f}</td>"
            f"<td>{html.escape(notes)}</td>"
            "</tr>"
        )
    return (
        '<div class="table-wrap"><table class="data-table"><thead><tr>'
        "<th>Status / Camera</th><th>IP</th><th>Clip / Source</th><th>Exposure Correction</th><th>Kelvin</th><th>Tint</th><th>Confidence</th><th>Warnings / Notes</th>"
        "</tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table></div>"
    )


def _render_push_surface(surface: Dict[str, object]) -> str:
    push = dict(surface.get("push_surface") or {})
    rows = list(push.get("rows") or [])
    if not rows:
        return '<div class="empty-state">Load or generate a calibration commit payload to preview the camera writeback plan.</div>'
    metrics_html = "".join(
        [
            f"<div><span class='metric-label'>Payload Source</span><span class='metric-value'>{html.escape(str(Path(str(push.get('payload_path') or '')).name or 'current review payload'))}</span></div>",
            f"<div><span class='metric-label'>Strategy</span><span class='metric-value'>{html.escape(str(push.get('strategy_label') or 'Pending'))}</span></div>",
            f"<div><span class='metric-label'>Payload Cameras</span><span class='metric-value'>{int(push.get('camera_count', 0) or 0)}</span></div>",
            f"<div><span class='metric-label'>Connected Cameras Detected</span><span class='metric-value'>{int(push.get('connected_camera_count', 0) or 0)}</span></div>",
            f"<div><span class='metric-label'>Readback Verification</span><span class='metric-value'>{html.escape(str(push.get('readback_verification_state') or 'not_run').replace('_', ' '))}</span></div>",
        ]
    )
    verification_mode = str(push.get("verification_mode") or "").strip()
    compact_items = "".join(f"<li>{html.escape(str(item))}</li>" for item in list(push.get("compact_summaries") or []))
    body = []
    for row in rows:
        body.append(
            "<tr>"
            f"<td>{html.escape(str(row.get('inventory_camera_label') or ''))}</td>"
            f"<td>{_format_signed_value(row.get('current_exposure'), decimals=3) if row.get('current_exposure') is not None else '—'}</td>"
            f"<td>{_format_signed_value(row.get('new_exposure'), decimals=3)}</td>"
            f"<td>{_format_delta_html(row.get('exposure_delta'), decimals=3)}</td>"
            f"<td>{int(row.get('current_kelvin')) if row.get('current_kelvin') is not None else '—'}</td>"
            f"<td>{int(row.get('new_kelvin', 0) or 0)}</td>"
            f"<td>{_format_signed_value(row.get('current_tint'), decimals=3) if row.get('current_tint') is not None else '—'}</td>"
            f"<td>{_format_signed_value(row.get('new_tint'), decimals=3)}</td>"
            f"<td>{html.escape(str(row.get('reference_use') or 'Included'))}</td>"
            f"<td>{_badge_html(str(row.get('writeback_status') or 'Ready'), str(row.get('writeback_tone') or 'pending'))}</td>"
            f"<td>{html.escape(str(row.get('notes_text') or ''))}</td>"
            "</tr>"
        )
    return (
        "<div class='push-callout'>All exposure values shown in this push section are relative corrections applied to match the calibration target.</div>"
        f"<div class='surface-metrics'>{metrics_html}</div>"
        "<p class='surface-copy'>Current Exposure / New Exposure / Exposure Change describe the exposure correction value on the camera, not the gray-target measurement used internally during review.</p>"
        + (
            f"<p class='surface-note'>Verification mode: {html.escape(verification_mode.replace('_', ' '))}.</p>"
            if verification_mode
            else ""
        )
        +
        f"<ul class='compact-summary-list'>{compact_items}</ul>"
        '<div class="table-wrap"><table class="data-table"><thead><tr>'
        "<th>Camera</th><th>Current Exposure</th><th>New Exposure</th><th>Exposure Change</th><th>Current Kelvin</th><th>New Kelvin</th><th>Current Tint</th><th>New Tint</th><th>Reference Use</th><th>Writeback Status</th><th>Notes</th>"
        "</tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table></div>"
    )


def _render_apply_surface(surface: Dict[str, object]) -> str:
    if not surface:
        return '<div class="empty-state">No apply report has been generated for this run yet.</div>'
    counts = dict(surface.get("operator_status_counts") or {})
    rows = list(surface.get("results") or [])
    counts_html = "".join(
        f"<div><span class='metric-label'>{html.escape(key)}</span><span class='metric-value'>{int(value)}</span></div>"
        for key, value in sorted(counts.items())
    ) or "<div><span class='metric-label'>Status</span><span class='metric-value'>No rows</span></div>"
    body = []
    for row in rows:
        operator_result = dict(row.get("operator_result") or {})
        requested = dict(row.get("requested") or {})
        readback = dict(row.get("readback") or {})
        camera_verification = dict(row.get("camera_verification") or {})
        verification_summary = dict(camera_verification.get("verification_summary") or {})
        per_field = dict(camera_verification.get("per_field") or {})
        deviations = dict(operator_result.get("deviations") or {})
        failure_reason = str(row.get("failure_reason") or "")
        verified = bool((row.get("verification") or {}).get("matches")) if isinstance(row.get("verification"), dict) else False
        exposure_requested = float(requested.get("exposureAdjust", 0.0) or 0.0) if requested else None
        kelvin_requested = int(requested.get("kelvin", 0) or 0) if requested else None
        tint_requested = float(requested.get("tint", 0.0) or 0.0) if requested else None
        exposure_applied = float(readback.get("exposureAdjust", 0.0) or 0.0) if readback else None
        kelvin_applied = int(readback.get("kelvin", 0) or 0) if readback else None
        tint_applied = float(readback.get("tint", 0.0) or 0.0) if readback else None
        quantization_notes = []
        if exposure_requested is not None and exposure_applied is not None and abs(exposure_applied - exposure_requested) > 1e-6:
            quantization_notes.append(f"Exposure quantized by {_format_signed_value(exposure_applied - exposure_requested, decimals=3)}")
        if kelvin_requested is not None and kelvin_applied is not None and kelvin_applied != kelvin_requested:
            quantization_notes.append(f"Kelvin quantized by {kelvin_applied - kelvin_requested:+d}")
        if tint_requested is not None and tint_applied is not None and abs(tint_applied - tint_requested) > 1e-6:
            quantization_notes.append(f"Tint quantized by {_format_signed_value(tint_applied - tint_requested, decimals=3)}")
        for field_name in ("kelvin", "tint", "exposureAdjust"):
            field_result = dict(per_field.get(field_name) or {})
            field_status = str(field_result.get("verification_status") or "").strip()
            if field_status and field_status != "EXACT_MATCH":
                quantization_notes.append(f"{field_name}: {field_status.lower().replace('_', ' ')}")
        histogram_guard = dict(camera_verification.get("histogram_guard") or {})
        if bool(histogram_guard.get("clipping_detected")):
            quantization_notes.append("Gray reference clipping detected")
        clip_metadata = dict(camera_verification.get("clip_metadata_cross_check") or {})
        metadata_status = str(clip_metadata.get("metadata_match_status") or "").strip()
        if metadata_status and metadata_status != "NOT_AVAILABLE":
            quantization_notes.append(f"clip metadata: {metadata_status.lower().replace('_', ' ')}")
        verification_label = (
            str(verification_summary.get("final_status") or "").replace("_", " ").title()
            or ("Verified" if verified else ("Not verified" if readback else "n/a"))
        )
        body.append(
            "<tr>"
            f"<td>{html.escape(str(row.get('inventory_camera_label') or row.get('camera_id') or ''))}</td>"
            f"<td>{html.escape(str(operator_result.get('display_status') or row.get('status') or ''))}</td>"
            f"<td>{html.escape(failure_reason) if failure_reason else 'n/a'}</td>"
            f"<td>{_format_signed_value(exposure_requested, decimals=3) if exposure_requested is not None else '—'}</td>"
            f"<td>{_format_signed_value(exposure_applied, decimals=3) if exposure_applied is not None else '—'}</td>"
            f"<td>{kelvin_requested if kelvin_requested is not None else '—'}</td>"
            f"<td>{kelvin_applied if kelvin_applied is not None else '—'}</td>"
            f"<td>{_format_signed_value(tint_requested, decimals=3) if tint_requested is not None else '—'}</td>"
            f"<td>{_format_signed_value(tint_applied, decimals=3) if tint_applied is not None else '—'}</td>"
            f"<td>{html.escape(verification_label)}</td>"
            f"<td>{html.escape('; '.join(quantization_notes) if quantization_notes else (json.dumps(deviations, separators=(', ', ': ')) if deviations else 'Exact readback'))}</td>"
            f"<td>{html.escape(str(row.get('error') or ''))}</td>"
            "</tr>"
        )
    return (
        f"<div class='surface-metrics'>{counts_html}</div>"
        f"<p class='surface-note'>Transport mode: {html.escape(str(surface.get('transport_mode') or ''))}. "
        f"Operator status: {html.escape(str(surface.get('operator_status') or surface.get('status') or ''))}. "
        f"Tolerances: {html.escape(json.dumps(surface.get('operator_tolerances') or {}, separators=(', ', ': ')))}.</p>"
        "<p class='surface-copy'>Requested values are the intended writeback targets. Applied values come from camera readback, so any quantization is visible immediately.</p>"
        '<div class="table-wrap"><table class="data-table"><thead><tr>'
        "<th>Camera</th><th>Display Status</th><th>Failure Reason</th><th>Requested Exposure</th><th>Applied Exposure</th><th>Requested Kelvin</th><th>Applied Kelvin</th><th>Requested Tint</th><th>Applied Tint</th><th>Readback Verified</th><th>Quantization / Deviation</th><th>Error</th>"
        "</tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table></div>"
    )


def _render_verification_surface(surface: Dict[str, object]) -> str:
    if not surface:
        return '<div class="empty-state">No post-apply verification summary is available yet.</div>'
    summary = dict(surface.get("summary") or {})
    notes = "".join(f"<li>{html.escape(str(note))}</li>" for note in surface.get("notes", []) or []) or "<li>No verification notes.</li>"
    mode = str(surface.get("verification_mode") or "before_after_review_comparison")
    return f"""
    <div class="surface-heading-row">
      <h3>Verification Summary</h3>
      {_badge_html(str(surface.get('status') or 'pending'), 'success' if str(surface.get('status') or '') == 'success' else 'warning')}
    </div>
    <div class="surface-metrics">
      <div><span class="metric-label">Mode</span><span class="metric-value">{html.escape(mode)}</span></div>
      <div><span class="metric-label">Exposure Improved</span><span class="metric-value">{'yes' if summary.get('exposure_improved') or summary.get('exposure_error_reduced') else 'no'}</span></div>
      <div><span class="metric-label">Neutrality Improved</span><span class="metric-value">{'yes' if summary.get('neutrality_improved') else 'no'}</span></div>
      <div><span class="metric-label">Variance Reduced</span><span class="metric-value">{'yes' if summary.get('variance_reduced') else 'no'}</span></div>
    </div>
    <ul class="surface-list">{notes}</ul>
    """


def _scan_summary_text(scan: Dict[str, object]) -> str:
    if scan.get("warning"):
        return str(scan["warning"])
    if scan.get("source_mode") == "ftps_camera_pull" and scan.get("summary"):
        summary = scan["summary"]
        return (
            f"Planned FTPS pull: reel {summary['reel_identifier']} | clips {summary['clip_spec']} | "
            f"{summary['camera_count']} cameras"
        )
    return f"Found {scan.get('clip_count', 0)} RED clips"


def _advanced_clip_selection_enabled(form: Dict[str, object]) -> bool:
    return bool(form.get("advanced_clip_selection"))


def _selected_group_ids(form: Dict[str, object], scan: Dict[str, object]) -> List[str]:
    selected_groups = {str(item) for item in form.get("selected_clip_groups", []) if str(item).strip()}
    return [str(group.get("group_id")) for group in scan.get("clip_groups", []) if str(group.get("group_id")) in selected_groups]


def _normalize_subset_form(form: Dict[str, object], scan: Dict[str, object]) -> Dict[str, object]:
    advanced_mode = _advanced_clip_selection_enabled(form)
    discovered_clip_ids = [str(item) for item in scan.get("clip_ids", [])]
    form["selected_clip_ids"] = [clip_id for clip_id in form.get("selected_clip_ids", []) if clip_id in discovered_clip_ids]
    form["selected_clip_groups"] = _selected_group_ids(form, scan)
    if advanced_mode:
        form["selected_clip_groups"] = []
    else:
        form["selected_clip_ids"] = []
    return form


def _resolve_selected_clip_ids(form: Dict[str, object], scan: Dict[str, object]) -> List[str]:
    discovered_clip_ids = [str(item) for item in scan.get("clip_ids", [])]
    selected = {str(item) for item in form.get("selected_clip_ids", []) if str(item).strip()}
    selected_groups = set() if _advanced_clip_selection_enabled(form) else {str(item) for item in form.get("selected_clip_groups", []) if str(item).strip()}
    if selected_groups:
        for group in scan.get("clip_groups", []):
            if str(group.get("group_id")) in selected_groups:
                selected.update(str(item) for item in group.get("clip_ids", []))
    if not discovered_clip_ids:
        return []
    if not selected:
        if _advanced_clip_selection_enabled(form):
            return []
        return list(discovered_clip_ids)
    return [clip_id for clip_id in discovered_clip_ids if clip_id in selected]


def _pluralize(count: int, singular: str, plural: Optional[str] = None) -> str:
    return singular if count == 1 else (plural or f"{singular}s")


def _subset_selection_ui(form: Dict[str, object], scan: Dict[str, object]) -> Dict[str, str]:
    selected_clip_ids = _resolve_selected_clip_ids(form, scan)
    selected_groups = _selected_group_ids(form, scan)
    clip_count = len(selected_clip_ids)
    if _advanced_clip_selection_enabled(form):
        return {
            "summary_text": f"Manual selection: {clip_count} {_pluralize(clip_count, 'clip')}",
            "summary_hint": (
                "Manual clip selection is active. Group selection is locked until you switch back."
                if clip_count
                else "Manual clip selection is active. Choose one or more clips before running review."
            ),
            "mode_label": "Manual Mode",
            "mode_class": "manual",
            "group_panel_note": "Clip groups are disabled while manual clip selection is active.",
            "clip_panel_note": "Interactive clip list. Choose the exact clips to include in this run.",
        }
    if len(selected_groups) == 1:
        group_id = selected_groups[0]
        return {
            "summary_text": f"{clip_count} clips selected from group {group_id}",
            "summary_hint": "Group selection drives the subset. The clip list on the right is a read-only preview.",
            "mode_label": "Group Mode",
            "mode_class": "group",
            "group_panel_note": "One group selected. Add more groups to expand the subset.",
            "clip_panel_note": "Read-only preview of the current group-driven subset.",
        }
    if len(selected_groups) > 1:
        group_text = ", ".join(selected_groups)
        return {
            "summary_text": f"{clip_count} clips selected from groups {group_text}",
            "summary_hint": "Multiple groups are combined into one subset. The clip list is a read-only preview.",
            "mode_label": "Group Mode",
            "mode_class": "group",
            "group_panel_note": "Multiple groups selected. The run will include the combined clip set shown on the right.",
            "clip_panel_note": "Read-only preview of the combined group selection.",
        }
    if scan.get("clip_ids"):
        group_count = len(scan.get("clip_groups", []))
        return {
            "summary_text": f"Full calibration set selected: {clip_count} clips across {group_count} {_pluralize(group_count, 'group')}",
            "summary_hint": "No group filter is active yet. Choose one or more groups to narrow the run.",
            "mode_label": "Full Set",
            "mode_class": "full",
            "group_panel_note": "No group filter selected. Review will use the full discovered set.",
            "clip_panel_note": "Read-only preview of the full calibration set.",
        }
    return {
        "summary_text": "No clips selected yet.",
        "summary_hint": "Scan a calibration folder to populate the subset controls.",
        "mode_label": "Full Set",
        "mode_class": "full",
        "group_panel_note": "Scan a calibration folder to discover clip groups.",
        "clip_panel_note": "Scan a calibration folder to preview clips for this subset.",
    }


def _selected_summary_text(form: Dict[str, object], scan: Dict[str, object]) -> str:
    return _subset_selection_ui(form, scan)["summary_text"]


def _resolved_output_path_for_form(form: Dict[str, object], scan: Dict[str, object]) -> str:
    if not str(form.get("output_path", "")).strip():
        return ""
    selected_clip_ids = [str(item).strip() for item in form.get("selected_clip_ids", []) if str(item).strip()]
    discovered_clip_ids = [str(item) for item in scan.get("clip_ids", [])]
    selected_clip_groups = [str(item) for item in form.get("selected_clip_groups", []) if str(item).strip()]
    if (
        not str(form.get("run_label") or "").strip()
        and not selected_clip_groups
        and not selected_clip_ids
        and discovered_clip_ids
    ):
        return str(form["output_path"])
    return resolve_review_output_dir(
        str(form["output_path"]),
        run_label=str(form.get("run_label") or "").strip() or None,
        selected_clip_ids=selected_clip_ids,
        selected_clip_groups=selected_clip_groups,
    )


def _roi_description(form: Dict[str, object]) -> str:
    mode = str(form.get("roi_mode", "sphere_auto"))
    if mode == "sphere_auto":
        return "Automatic sphere detection drives the gray-sphere measurement workflow."
    if mode == "center":
        return "Center crop (30% of frame)"
    if mode == "full":
        return "Full frame"
    if mode == "manual":
        return "Manual ROI values"
    roi_values = [form.get("roi_x"), form.get("roi_y"), form.get("roi_w"), form.get("roi_h")]
    if all(str(v).strip() for v in roi_values):
        return f"Custom ROI: x={form['roi_x']} y={form['roi_y']} w={form['roi_w']} h={form['roi_h']}"
    return "Draw a custom ROI on the preview image."


def _preferred_roi_preview_record(form: Dict[str, object], scan: Dict[str, object]) -> Optional[Dict[str, object]]:
    clip_records = [dict(item) for item in scan.get("clip_records", []) if isinstance(item, dict)]
    if not clip_records:
        return None
    selected_clip_ids = set(_resolve_selected_clip_ids(form, scan))
    preferred_groups = [str(item).strip() for item in form.get("selected_clip_groups", []) if str(item).strip()]
    available_groups = {str(item.get("subset_group") or "") for item in clip_records}
    prioritized_groups: List[str] = []
    if "063" in available_groups:
        prioritized_groups.append("063")
    for group_id in preferred_groups:
        if group_id in available_groups and group_id not in prioritized_groups:
            prioritized_groups.append(group_id)
    if not prioritized_groups:
        prioritized_groups = sorted(group_id for group_id in available_groups if group_id)

    selected_records = [
        record for record in clip_records if (not selected_clip_ids or str(record.get("clip_id") or "") in selected_clip_ids)
    ]
    candidates = selected_records or clip_records

    def choose_by_group(group_id: str, records: List[Dict[str, object]]) -> Optional[Dict[str, object]]:
        matching = [record for record in records if str(record.get("subset_group") or "") == group_id]
        if matching:
            return sorted(matching, key=lambda item: str(item.get("clip_id") or ""))[0]
        return None

    for group_id in prioritized_groups:
        source_records = clip_records if group_id == "063" else candidates
        chosen = choose_by_group(group_id, source_records)
        if chosen is not None:
            return chosen
    return sorted(candidates, key=lambda item: str(item.get("clip_id") or ""))[0]


def _scan_preview_path(input_path: str, *, clip_id: str = "", preview_mode: str = "monitoring") -> Path:
    normalized_preview_mode = "monitoring" if (preview_mode.strip().lower() if preview_mode else "monitoring") == "calibration" else (preview_mode.strip() or "monitoring")
    token_source = "|".join(
        [
            str(Path(input_path).expanduser().resolve()),
            clip_id.strip(),
            normalized_preview_mode,
        ]
    )
    token = hashlib.sha1(token_source.encode("utf-8")).hexdigest()[:12]
    return Path("/tmp") / f"r3dmatch_web_scan_preview_{token}.jpg"


def _ensure_scan_preview(input_path: str, scan: Dict[str, object], form: Optional[Dict[str, object]] = None) -> Optional[str]:
    resolved_form = form or {}
    record = _preferred_roi_preview_record(resolved_form, scan)
    clip_path = str((record or {}).get("source_path") or scan.get("first_clip_path") or "").strip()
    clip_id = str((record or {}).get("clip_id") or "")
    if not clip_path:
        scan["preview_warning"] = "No representative clip frame was available for ROI preview."
        scan["preview_note"] = None
        return None
    preview_mode = str(resolved_form.get("preview_mode") or "monitoring")
    output_path = _scan_preview_path(str(input_path), clip_id=clip_id, preview_mode=preview_mode)
    actual_preview_mode = preview_mode
    if output_path.exists():
        subset_group = str((record or {}).get("subset_group") or "")
        scan["preview_note"] = (
            f"ROI preview uses {clip_id} from group {subset_group} with the active {actual_preview_mode} preview transform."
            if clip_id and subset_group
            else f"ROI preview uses {clip_id} with the active {actual_preview_mode} preview transform."
        )
        scan["preview_warning"] = None
        return str(output_path)
    redline_executable = _resolve_redline_executable()
    capabilities = _detect_redline_capabilities(redline_executable)
    settings = _normalize_preview_settings(
        preview_mode=preview_mode,
        preview_output_space=None,
        preview_output_gamma=None,
        preview_highlight_rolloff=None,
        preview_shadow_rolloff=None,
        preview_lut=None,
    )
    actual_preview_mode = str(settings.get("preview_mode") or "monitoring")
    kelvin_raw = resolved_form.get("kelvin", scan.get("kelvin", 5600))
    tint_raw = resolved_form.get("tint", scan.get("tint", 0))
    try:
        kelvin = int(round(float(kelvin_raw if kelvin_raw not in (None, "") else 5600)))
    except (TypeError, ValueError):
        kelvin = 5600
    try:
        tint = float(tint_raw if tint_raw not in (None, "") else 0)
    except (TypeError, ValueError):
        tint = 0.0
    command = _build_redline_preview_command(
        clip_path,
        output_path=str(output_path),
        frame_index=0,
        exposure_stops=None,
        kelvin=kelvin,
        tint=tint,
        color_cdl=None,
        color_method=None,
        redline_executable=redline_executable,
        preview_settings=settings,
        redline_capabilities=capabilities,
        use_as_shot_metadata=True,
        rmd_path=None,
    )
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        scan["preview_warning"] = f"ROI preview fallback in use because preview extraction failed for {clip_id or 'the selected clip'}."
        scan["preview_note"] = None
        return None
    if not output_path.exists():
        generated_matches = sorted(output_path.parent.glob(f"{output_path.name}.*"))
        if generated_matches:
            generated_matches[0].replace(output_path)
    if output_path.exists():
        subset_group = str((record or {}).get("subset_group") or "")
        scan["preview_note"] = (
            f"ROI preview uses {clip_id} from group {subset_group} with the active {actual_preview_mode} preview transform."
            if clip_id and subset_group
            else f"ROI preview uses {clip_id} with the active {actual_preview_mode} preview transform."
        )
        scan["preview_warning"] = None
        return str(output_path)
    scan["preview_warning"] = "ROI preview fallback in use because no rendered frame was produced."
    scan["preview_note"] = None
    return None


REVIEW_STAGES = [
    "Scanning sources",
    "Measuring originals",
    "Solving strategies",
    "Rendering previews",
    "Building report",
    "Complete",
]

APPROVAL_STAGES = [
    "Loading review package",
    "Writing Master RMDs",
    "Complete",
]


def _make_cancel_file_path() -> str:
    return str((Path("/tmp") / f"r3dmatch_cancel_{uuid.uuid4().hex}.flag").resolve())


def _process_is_alive(process: Optional[subprocess.Popen[str]]) -> bool:
    if process is None:
        return False
    try:
        return process.poll() is None
    except Exception:
        return False


def _artifacts_ready_for_task(task: TaskState, output_root: Optional[Path], *, is_review: bool, is_approve: bool) -> bool:
    if output_root is None or not output_root.exists():
        return False
    if is_review:
        return _path_is_current(review_pdf_path_for(output_root), started_at=task.started_at) or _path_is_current(review_html_path_for(output_root), started_at=task.started_at)
    if is_approve:
        approval_root = output_root / "approval"
        return _path_is_current(approval_root / "calibration_report.pdf", started_at=task.started_at) or _path_is_current(approval_root / "approval_manifest.json", started_at=task.started_at)
    return False


def _load_review_validation(output_root: Optional[Path], *, started_at: Optional[float] = None) -> Optional[Dict[str, object]]:
    if output_root is None:
        return None
    validation_path = review_validation_path_for(output_root)
    if not validation_path.exists():
        return None
    try:
        payload = json.loads(validation_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            "status": "failed",
            "errors": [f"Failed to parse review_validation.json: {exc}"],
            "warnings": [],
            "validation_path": str(validation_path),
        }
    if not _validation_timestamp_is_current(payload, validation_path, started_at=started_at):
        return None
    payload["validation_path"] = str(validation_path)
    return payload


def _output_root_from_validation_payload(payload: Optional[Dict[str, object]]) -> Optional[Path]:
    if not isinstance(payload, dict):
        return None
    validation_path = str(payload.get("validation_path") or "").strip()
    if not validation_path:
        return None
    candidate = Path(validation_path).expanduser().resolve()
    if not candidate.exists() or candidate.name != "review_validation.json":
        return None
    if candidate.parent.name != "report":
        return None
    return candidate.parent.parent


def _canonical_output_root_from_base(base: Dict[str, object]) -> Optional[Path]:
    output_text = str(base.get("canonical_output_path") or base.get("output_path") or "").strip()
    if not output_text:
        return None
    return Path(output_text).expanduser().resolve()


def _finalization_debug_payload(
    *,
    base: Dict[str, object],
    output_root: Optional[Path],
    review_validation: Optional[Dict[str, object]],
) -> Dict[str, object]:
    validation_path = ""
    validation_exists = False
    parsed_status = None
    validated_at = None
    if isinstance(review_validation, dict):
        validation_path = str(review_validation.get("validation_path") or "")
        parsed_status = review_validation.get("status")
        validated_at = review_validation.get("validated_at")
        if validation_path:
            validation_exists = Path(validation_path).expanduser().resolve().exists()
    return {
        "resolved_run_directory": str(output_root) if output_root else "",
        "resolved_validation_path": validation_path,
        "validation_exists": validation_exists,
        "parsed_validation_status": parsed_status,
        "validated_at": validated_at,
        "returncode": base.get("returncode"),
        "final_ui_status": base.get("status"),
        "final_ui_stage": base.get("stage"),
    }


def _log_finalization_debug(task: TaskState, debug_payload: Dict[str, object]) -> None:
    signature = json.dumps(debug_payload, sort_keys=True, default=str)
    with task.lock:
        if signature == task.last_finalization_debug_signature:
            return
        task.last_finalization_debug_signature = signature
    LOGGER.info("review finalization debug: %s", signature)


def _load_review_manifest(output_root: Optional[Path], *, started_at: Optional[float] = None) -> Optional[Dict[str, object]]:
    if output_root is None:
        return None
    manifest_path = review_manifest_path_for(output_root)
    if not _path_is_current(manifest_path, started_at=started_at):
        return None
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    payload["manifest_path"] = str(manifest_path)
    return payload


def _load_review_progress(output_root: Optional[Path], *, started_at: Optional[float] = None) -> Optional[Dict[str, object]]:
    if output_root is None:
        return None
    return load_review_progress(review_progress_path_for(output_root), started_at=started_at)


def _update_progress_timestamp(task: TaskState, previous_state: tuple[object, ...]) -> None:
    current_state = (
        task.stage,
        task.stage_index,
        task.total_stages,
        task.items_completed,
        task.items_total,
        task.preview_pdf_path,
        task.preview_html_path,
        task.report_ready,
    )
    if current_state != previous_state:
        task.last_progress_at = time.time()


def _request_task_cancellation(task: TaskState) -> bool:
    with task.lock:
        process = task.process
        if not _process_is_alive(process) and task.status not in {"running", "finishing", "stalled", "cancelling"}:
            return False
        task.cancellation_requested = True
        task.status = "cancelling"
        task.stage = "Cancelling..."
        cancel_file_path = task.cancel_file_path
        task.logs.append("[cancel] Operator requested cancellation.\n")
    if cancel_file_path:
        Path(cancel_file_path).write_text("cancelled\n", encoding="utf-8")
    if process is not None:
        threading.Thread(target=terminate_process_group, args=(process,), kwargs={"grace_period": 2.0}, daemon=True).start()
    return True


def _sync_task_output_path(task: TaskState, resolved_output_path: str) -> None:
    resolved = str(resolved_output_path or "").strip()
    if not resolved:
        return
    with task.lock:
        task.output_path = resolved
        task.canonical_output_path = resolved


def _infer_progress(task: TaskState) -> None:
    now = time.time()
    process_alive = _process_is_alive(task.process)
    resolved_output = str(task.canonical_output_path or task.output_path).strip()
    output_root = Path(resolved_output).expanduser().resolve() if resolved_output else None
    is_review = "review-calibration" in task.command
    is_approve = "approve-master-rmd" in task.command
    review_validation = _load_review_validation(output_root, started_at=task.started_at) if is_review else None
    review_manifest = _load_review_manifest(output_root, started_at=task.started_at) if is_review else None
    review_progress = _load_review_progress(output_root, started_at=task.started_at) if is_review else None
    resolved_review_mode = normalize_review_mode(
        str(
            task.review_mode
            or (review_manifest or {}).get("review_mode")
            or (review_validation or {}).get("review_mode")
            or "full_contact_sheet"
        )
    ) if is_review else ""
    if is_review:
        task.review_mode = resolved_review_mode
    previous_progress_state = (
        task.stage,
        task.stage_index,
        task.total_stages,
        task.items_completed,
        task.items_total,
        task.preview_pdf_path,
        task.preview_html_path,
        task.report_ready,
    )
    if is_review:
        task.total_stages = len(REVIEW_STAGES)
        if output_root and output_root.exists():
            preview_pdf = review_pdf_path_for(output_root)
            preview_html = review_html_path_for(output_root)
            analysis_dir = output_root / "analysis"
            array_cal = output_root / "array_calibration.json"
            previews_dir = output_root / "previews"
            visible_previews = [
                p for p in previews_dir.glob("*.jpg")
                if ".000000." not in p.name and _path_is_current(p, started_at=task.started_at)
            ] if previews_dir.exists() else []
            clip_count = len([p for p in analysis_dir.glob("*.analysis.json") if _path_is_current(p, started_at=task.started_at)]) if analysis_dir.exists() else 0
            progress_clip_count = int((review_progress or {}).get("clip_count", 0) or 0)
            progress_clip_index = int((review_progress or {}).get("clip_index", 0) or 0)
            strategies_count = max(1, len([item for item in task.strategies_text.split(", ") if item])) if task.strategies_text else 1
            expected_previews = clip_count * (1 + strategies_count * 3) if clip_count and resolved_review_mode == "full_contact_sheet" else 0
            task.clip_count = clip_count or progress_clip_count or task.clip_count
            if resolved_review_mode == "lightweight_analysis":
                task.items_total = clip_count or progress_clip_count or task.items_total
                if progress_clip_index:
                    task.items_completed = max(task.items_completed, min(progress_clip_index, task.items_total or progress_clip_index))
                else:
                    task.items_completed = min(task.items_completed, task.items_total) if task.items_total else task.items_completed
            else:
                task.items_total = expected_previews or task.items_total
                task.items_completed = min(len(visible_previews), task.items_total or len(visible_previews))
            preview_pdf_ready = _path_is_current(preview_pdf, started_at=task.started_at)
            preview_html_ready = _path_is_current(preview_html, started_at=task.started_at)
            if preview_pdf_ready or preview_html_ready:
                if process_alive:
                    task.stage_index = 4
                    task.stage = REVIEW_STAGES[4]
                else:
                    task.stage_index = len(REVIEW_STAGES) - 1
                    task.stage = REVIEW_STAGES[-1]
            elif expected_previews and len(visible_previews) >= expected_previews:
                task.stage_index = 4
                task.stage = REVIEW_STAGES[4]
            elif visible_previews:
                task.stage_index = 3
                task.stage = REVIEW_STAGES[3]
            elif _path_is_current(array_cal, started_at=task.started_at):
                task.stage_index = 2
                task.stage = REVIEW_STAGES[2]
            elif clip_count:
                task.stage_index = 1
                task.stage = REVIEW_STAGES[1]
            elif review_progress:
                progress_stage = str(review_progress.get("stage_label") or "").strip()
                if progress_stage:
                    task.stage = progress_stage
            else:
                task.stage_index = 0
                task.stage = REVIEW_STAGES[0]
            task.preview_pdf_path = str(preview_pdf) if preview_pdf_ready else None
            task.preview_html_path = str(preview_html) if preview_html_ready else None
            task.report_ready = preview_pdf_ready or preview_html_ready
            if review_validation:
                preview_reference_count = int(review_validation.get("preview_reference_count", 0) or 0)
                preview_existing_count = int(review_validation.get("preview_existing_count", preview_reference_count) or 0)
                if preview_reference_count:
                    task.items_total = max(task.items_total, preview_reference_count)
                    task.items_completed = max(task.items_completed, preview_existing_count)
                elif resolved_review_mode == "lightweight_analysis":
                    clip_count_from_validation = int(review_validation.get("clip_count", task.clip_count) or task.clip_count)
                    task.items_total = max(task.items_total, clip_count_from_validation)
                    if task.report_ready or not process_alive:
                        task.items_completed = task.items_total
    elif is_approve:
        task.total_stages = len(APPROVAL_STAGES)
        if output_root and output_root.exists():
            approval_root = output_root / "approval"
            master_dir = approval_root / "MasterRMD"
            if not master_dir.exists():
                master_dir = approval_root / "Master_RMD"
            approval_pdf = approval_root / "calibration_report.pdf"
            approval_manifest = approval_root / "approval_manifest.json"
            task.items_completed = len(list(master_dir.glob("*.RMD"))) if master_dir.exists() else 0
            task.items_total = task.items_completed or task.items_total
            approval_pdf_ready = _path_is_current(approval_pdf, started_at=task.started_at)
            approval_manifest_ready = _path_is_current(approval_manifest, started_at=task.started_at)
            if approval_pdf_ready or approval_manifest_ready:
                if process_alive:
                    task.stage_index = 1
                    task.stage = APPROVAL_STAGES[1]
                else:
                    task.stage_index = len(APPROVAL_STAGES) - 1
                    task.stage = APPROVAL_STAGES[-1]
            elif master_dir.exists() and task.items_completed:
                task.stage_index = 1
                task.stage = APPROVAL_STAGES[1]
            else:
                task.stage_index = 0
                task.stage = APPROVAL_STAGES[0]
            task.preview_pdf_path = str(approval_pdf) if approval_pdf_ready else None
            task.preview_html_path = None
            task.report_ready = approval_pdf_ready or approval_manifest_ready

    _update_progress_timestamp(task, previous_progress_state)

    artifacts_ready = _artifacts_ready_for_task(task, output_root, is_review=is_review, is_approve=is_approve)
    last_activity_at = max(task.started_at, task.last_output_at, task.last_progress_at)
    stalled = process_alive and (now - last_activity_at) >= STALL_THRESHOLD_SECONDS

    if task.cancellation_requested and process_alive:
        task.status = "cancelling"
        task.stage = "Cancelling..."
        return

    if process_alive:
        if stalled:
            task.status = "stalled"
        elif task.report_ready or (task.items_total and task.items_completed >= task.items_total):
            task.status = "finishing"
        else:
            task.status = "running"
        return

    if task.returncode is None:
        if task.status not in {"idle", "completed", "failed", "cancelled"}:
            task.status = "idle"
        return

    if task.returncode == 0 and is_review:
        if review_validation is not None and review_validation.get("status") == "success":
            task.status = "completed"
            task.stage_index = len(REVIEW_STAGES) - 1
            task.stage = REVIEW_STAGES[-1]
        else:
            task.status = "failed"
            task.stage_index = 4
            task.stage = "Finalization failed"
        return

    if task.returncode == 0 and artifacts_ready:
        task.status = "completed"
        if is_review:
            task.stage_index = len(REVIEW_STAGES) - 1
            task.stage = REVIEW_STAGES[-1]
        elif is_approve:
            task.stage_index = len(APPROVAL_STAGES) - 1
            task.stage = APPROVAL_STAGES[-1]
        return

    if task.returncode in {130, -15, -9}:
        task.status = "cancelled"
        task.stage = "Cancelled"
        return

    if task.returncode == 0 and not artifacts_ready:
        task.status = "failed"
        task.stage = "Finalization incomplete"
        return

    task.status = "failed"


def _status_payload(task: TaskState) -> Dict[str, object]:
    with task.lock:
        _infer_progress(task)
        base = task.snapshot()
    output_root = _canonical_output_root_from_base(base)
    review_validation = _load_review_validation(output_root, started_at=float(base.get("started_at") or 0.0)) if "review-calibration" in base["command"] else None
    review_progress = _load_review_progress(output_root, started_at=float(base.get("started_at") or 0.0)) if "review-calibration" in base["command"] else None
    canonical_output_root = _output_root_from_validation_payload(review_validation)
    if canonical_output_root is not None and canonical_output_root != output_root:
        output_root = canonical_output_root
        base["canonical_output_path"] = str(canonical_output_root)
        base["output_path"] = str(canonical_output_root)
        with task.lock:
            task.canonical_output_path = str(canonical_output_root)
            task.output_path = str(canonical_output_root)
    review_payload = _load_review_payload(output_root, started_at=float(base.get("started_at") or 0.0)) if output_root else None
    commit_payload = _load_commit_payload(output_root, started_at=float(base.get("started_at") or 0.0)) if output_root else None
    camera_state_report = _load_latest_camera_state_report(output_root) if output_root else None
    apply_report = _load_latest_apply_report(output_root) if output_root else None
    writeback_verification_report = _load_latest_writeback_verification_report(output_root) if output_root else None
    post_apply_report = _load_post_apply_verification(output_root) if output_root else None
    resolved_review_mode = normalize_review_mode(str(base.get("review_mode") or "full_contact_sheet")) if "review-calibration" in base["command"] else ""
    stage_names = REVIEW_STAGES if "review-calibration" in base["command"] else APPROVAL_STAGES if "approve-master-rmd" in base["command"] else []
    progress_percent = int(((base["stage_index"] + (1 if base["status"] == "completed" else 0)) / max(len(stage_names), 1)) * 100) if stage_names else 0
    can_cancel = bool(base.get("process_alive")) or base["status"] == "cancelling"
    last_activity_seconds = int(max(0.0, time.time() - max(float(base.get("last_output_at") or 0.0), float(base.get("last_progress_at") or 0.0), float(base.get("started_at") or 0.0))))
    validation_errors = list(review_validation.get("errors", [])) if isinstance(review_validation, dict) else []
    validation_warnings = list(review_validation.get("warnings", [])) if isinstance(review_validation, dict) else []
    status_detail = validation_errors[0] if validation_errors else ""
    if not status_detail and base.get("status") not in {"completed"} and validation_warnings:
        status_detail = validation_warnings[0]
    if not status_detail and isinstance(review_progress, dict):
        status_detail = str(review_progress.get("detail") or "")
    if not status_detail and "review-calibration" in base["command"] and base.get("status") == "failed" and base.get("returncode") == 0 and review_validation is None:
        status_detail = "Fresh review_validation.json was not produced for this run."
    operator_surfaces = _build_operator_surfaces(
        review_validation=review_validation,
        review_payload=review_payload,
        commit_payload=commit_payload,
        camera_state_report=camera_state_report,
        apply_report=apply_report,
        writeback_verification_report=writeback_verification_report,
        post_apply_report=post_apply_report,
        source_mode_label_text=source_mode_label(str(base.get("source_mode") or "local_folder")) if base.get("source_mode") else "",
    )
    canonical_output_text = str(base.get("canonical_output_path") or base.get("output_path") or "")
    base.update(
        {
            "output_path": canonical_output_text,
            "canonical_output_path": canonical_output_text,
            "stage_names": stage_names,
            "progress_percent": progress_percent,
            "preview_pdf_url": url_for("artifact", path=base["preview_pdf_path"]) if base.get("preview_pdf_path") else None,
            "preview_html_url": url_for("artifact", path=base["preview_html_path"]) if base.get("preview_html_path") else None,
            "output_folder": canonical_output_text,
            "can_cancel": can_cancel,
            "watchdog_status": base["status"],
            "last_activity_seconds": last_activity_seconds,
            "review_mode_label": review_mode_label(resolved_review_mode) if resolved_review_mode else "",
            "source_mode_label": source_mode_label(str(base.get("source_mode") or "local_folder")) if base.get("source_mode") else "",
            "status_detail": status_detail,
            "review_validation": review_validation,
            "review_progress": review_progress,
            "validation_status": review_validation.get("status") if isinstance(review_validation, dict) else None,
            "validation_path": review_validation.get("validation_path") if isinstance(review_validation, dict) else None,
            "validation_errors": validation_errors,
            "validation_warnings": validation_warnings,
            "decision_surface": operator_surfaces,
            "decision_surface_html": _render_decision_surface(operator_surfaces),
            "commit_table_html": _render_commit_table(operator_surfaces),
            "push_surface_html": _render_push_surface(operator_surfaces),
            "apply_surface_html": _render_apply_surface(operator_surfaces.get("apply_surface") or {}),
            "verification_surface_html": _render_verification_surface(operator_surfaces.get("verification_surface") or {}),
        }
    )
    if "review-calibration" in str(base.get("command") or "") and base.get("returncode") is not None:
        debug_payload = _finalization_debug_payload(base=base, output_root=output_root, review_validation=review_validation)
        base["finalization_debug"] = debug_payload
        _log_finalization_debug(task, debug_payload)
    return base


def _start_command_task(
    state: UiState,
    command: List[str],
    output_path: str,
    *,
    clip_count: int = 0,
    strategies_text: str = "",
    source_mode: str = "",
    review_mode: str = "",
    preview_mode: str = "",
) -> None:
    task = state.task
    snapshot = task.snapshot()
    if snapshot["process_alive"] or snapshot["status"] in {"running", "finishing", "stalled", "cancelling"}:
        state.error = "Another command is already running."
        return
    cancel_file_path = _make_cancel_file_path()
    with task.lock:
        task.status = "running"
        task.command = " ".join(command)
        task.output_path = output_path
        task.canonical_output_path = output_path
        task.source_mode = source_mode
        task.logs = [f"$ {' '.join(command)}\n"]
        task.returncode = None
        task.clip_count = clip_count
        task.strategies_text = strategies_text
        task.review_mode = review_mode
        task.preview_mode = preview_mode
        task.preview_pdf_path = None
        task.preview_html_path = None
        task.report_ready = False
        task.cancellation_requested = False
        task.cancel_file_path = cancel_file_path
        task.started_at = time.time()
        task.last_output_at = task.started_at
        task.last_progress_at = task.started_at
        task.finished_at = None
        task.process = None
        task.last_finalization_debug_signature = ""
        if "review-calibration" in task.command:
            task.total_stages = len(REVIEW_STAGES)
            task.stage_index = 0
            task.stage = REVIEW_STAGES[0]
            task.items_completed = 0
            task.items_total = clip_count
        elif "approve-master-rmd" in task.command:
            task.total_stages = len(APPROVAL_STAGES)
            task.stage_index = 0
            task.stage = APPROVAL_STAGES[0]
            task.items_completed = 0
            task.items_total = clip_count

    def worker() -> None:
        env = os.environ.copy()
        env[CANCEL_FILE_ENV] = cancel_file_path
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
            env=env,
        )
        with task.lock:
            task.process = process
        assert process.stdout is not None
        try:
            for line in process.stdout:
                task.append(line)
            returncode = process.wait()
            with task.lock:
                task.returncode = returncode
                task.finished_at = time.time()
                if task.cancellation_requested and returncode in {130, -15, -9}:
                    task.status = "cancelled"
                    task.stage = "Cancelled"
                elif returncode != 0:
                    task.status = "failed"
                task.logs.append("\n[cancelled]\n" if task.cancellation_requested and returncode in {130, -15, -9} else f"\n[exit] {returncode}\n")
        finally:
            Path(cancel_file_path).unlink(missing_ok=True)
            with task.lock:
                task.process = None

    threading.Thread(target=worker, daemon=True).start()


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["REPO_ROOT"] = str(Path(__file__).resolve().parents[2])
    app.config["UI_STATE"] = UiState()

    def render_page() -> str:
        state: UiState = app.config["UI_STATE"]
        selected_clip_ids = _resolve_selected_clip_ids(state.form, state.scan)
        subset_ui = _subset_selection_ui(state.form, state.scan)
        task_payload = _status_payload(state.task)
        current_output = str(task_payload.get("output_folder") or _resolved_output_path_for_form(state.form, state.scan) or state.form.get("output_path", ""))
        report_url = _report_url_for_output(
            current_output,
            started_at=float(task_payload.get("started_at") or 0.0) if task_payload.get("process_alive") else None,
        )
        return render_template_string(
            PAGE_TEMPLATE,
            has_logo=DEFAULT_LOGO_PATH.exists(),
            form=state.form,
            scan=state.scan,
            scan_summary_text=_scan_summary_text(state.scan),
            subset_ui=subset_ui,
            selected_clip_ids=selected_clip_ids,
            subset_ui_data={
                "clip_groups": state.scan.get("clip_groups", []),
                "clip_records": state.scan.get("clip_records", []),
                "selected_clip_ids": selected_clip_ids,
            },
            task=task_payload,
            roi_description=_roi_description(state.form),
            error=state.error,
            message=state.message,
            report_url=report_url,
        )

    @app.get("/")
    def index() -> str:
        return render_page()

    @app.post("/scan")
    def scan() -> str:
        state: UiState = app.config["UI_STATE"]
        form = _parse_form(request.form)
        if normalize_source_mode(str(form.get("source_mode", "local_folder"))) == "ftps_camera_pull":
            state.scan = _plan_ftps_scan(form)
        else:
            state.scan = scan_sources(str(form["input_path"]))
            _normalize_subset_form(form, state.scan)
        state.update_form(form)
        preview_path = _ensure_scan_preview(str(form["input_path"]), state.scan, form) if state.scan.get("first_clip_path") else None
        state.scan["preview_available"] = bool(preview_path)
        state.error = state.scan["warning"] if state.scan.get("warning") else None
        state.message = None if state.error else (
            f"Prepared FTPS ingest request for reel {state.scan['summary']['reel_identifier']}."
            if state.scan.get("source_mode") == "ftps_camera_pull" and state.scan.get("summary")
            else f"Scanned calibration folder: found {state.scan['clip_count']} RED clips."
        )
        return render_page()

    @app.post("/run-review")
    def run_review() -> str:
        state: UiState = app.config["UI_STATE"]
        form = _parse_form(request.form)
        if normalize_source_mode(str(form.get("source_mode", "local_folder"))) == "ftps_camera_pull":
            state.scan = _plan_ftps_scan(form)
        else:
            state.scan = scan_sources(str(form["input_path"]))
            _normalize_subset_form(form, state.scan)
        form["resolved_output_path"] = _resolved_output_path_for_form(form, state.scan)
        state.update_form(form)
        preview_path = _ensure_scan_preview(str(form["input_path"]), state.scan, form) if state.scan.get("first_clip_path") else None
        state.scan["preview_available"] = bool(preview_path)
        error = _validate_form(form)
        state.error = error
        state.message = None
        if error is None:
            command = build_review_web_command(app.config["REPO_ROOT"], form)
            selected_clip_ids = _resolve_selected_clip_ids(form, state.scan) if state.scan.get("clip_ids") else []
            _start_command_task(
                state,
                command,
                str(form["resolved_output_path"]),
                clip_count=len(selected_clip_ids),
                strategies_text=", ".join(str(value) for value in form.get("target_strategies", [])),
                source_mode=str(form.get("source_mode", "")),
                review_mode=str(form.get("review_mode", "")),
                preview_mode=str(form.get("preview_mode", "")),
            )
            state.message = "Review command started."
        return render_page()

    @app.post("/approve-master-rmd")
    def approve_master() -> str:
        state: UiState = app.config["UI_STATE"]
        form = _parse_form(request.form)
        _normalize_subset_form(form, state.scan)
        form["resolved_output_path"] = _resolved_output_path_for_form(form, state.scan)
        state.update_form(form)
        error = _validate_form(form, require_source=False, require_output=True)
        state.error = error
        state.message = None
        if error is None:
            command = build_approve_web_command(app.config["REPO_ROOT"], form)
            _start_command_task(
                state,
                command,
                str(form["resolved_output_path"]),
                clip_count=len(_resolve_selected_clip_ids(form, state.scan)),
                strategies_text=", ".join(str(value) for value in form.get("target_strategies", [])),
                source_mode=str(form.get("source_mode", "")),
                review_mode=str(form.get("review_mode", "")),
                preview_mode=str(form.get("preview_mode", "")),
            )
            state.message = "Approve Master RMD command started."
        return render_page()

    @app.post("/clear-preview-cache")
    def clear_preview() -> str:
        state: UiState = app.config["UI_STATE"]
        form = _parse_form(request.form)
        _normalize_subset_form(form, state.scan)
        form["resolved_output_path"] = _resolved_output_path_for_form(form, state.scan)
        state.update_form(form)
        if not str(form["output_path"]).strip():
            state.error = "Output folder path is required."
            state.message = None
            return render_page()
        state.error = None
        state.message = "Clear preview cache command started."
        command = build_clear_cache_web_command(app.config["REPO_ROOT"], form)
        _start_command_task(
            state,
            command,
            str(form["resolved_output_path"]),
            clip_count=len(_resolve_selected_clip_ids(form, state.scan)),
            strategies_text=", ".join(str(value) for value in form.get("target_strategies", [])),
            source_mode=str(form.get("source_mode", "")),
            review_mode=str(form.get("review_mode", "")),
            preview_mode=str(form.get("preview_mode", "")),
        )
        return render_page()

    @app.post("/apply-calibration")
    def apply_calibration_route() -> str:
        state: UiState = app.config["UI_STATE"]
        form = _parse_form(request.form)
        _normalize_subset_form(form, state.scan)
        form["resolved_output_path"] = _resolved_output_path_for_form(form, state.scan)
        state.update_form(form)
        _sync_task_output_path(state.task, str(form.get("resolved_output_path") or ""))
        if _process_is_alive(state.task.process):
            state.error = "Wait for the current command to finish before applying calibration."
            state.message = None
            return render_page()
        output_root = Path(str(form.get("resolved_output_path") or "")).expanduser().resolve()
        commit_payload_path = review_commit_payload_path_for(output_root)
        if not commit_payload_path.exists():
            state.error = "No calibration commit payload exists for the current review output yet."
            state.message = None
            return render_page()
        requested_cameras = [item.strip() for item in str(form.get("apply_cameras", "")).split(",") if item.strip()]
        report_path = output_root / "report" / "rcp2_apply_report.json"
        try:
            result = apply_calibration_payload(
                str(commit_payload_path),
                out_path=str(report_path),
                requested_cameras=requested_cameras or None,
                live=False,
            )
        except Exception as exc:
            state.error = str(exc)
            state.message = None
            return render_page()
        state.error = None
        state.message = f"Dry-run apply finished with status: {result.get('operator_status') or result.get('status')}."
        return render_page()

    @app.post("/load-recommended-payload")
    def load_recommended_payload_route() -> str:
        state: UiState = app.config["UI_STATE"]
        form = _parse_form(request.form)
        _normalize_subset_form(form, state.scan)
        form["resolved_output_path"] = _resolved_output_path_for_form(form, state.scan)
        state.update_form(form)
        _sync_task_output_path(state.task, str(form.get("resolved_output_path") or ""))
        output_root = Path(str(form.get("resolved_output_path") or "")).expanduser().resolve()
        commit_payload_path = review_commit_payload_path_for(output_root)
        if not commit_payload_path.exists():
            state.error = "No calibration commit payload exists for the current review output yet."
            state.message = None
            return render_page()
        requested_cameras = _requested_camera_tokens(form)
        try:
            targets = load_apply_targets(str(commit_payload_path), requested_cameras=requested_cameras or None)
        except Exception as exc:
            state.error = str(exc)
            state.message = None
            return render_page()
        state.error = None
        camera_text = f" for {len(targets)} camera{'s' if len(targets) != 1 else ''}" if targets else ""
        state.message = f"Loaded recommended payload{camera_text} from {commit_payload_path.name}."
        return render_page()

    @app.post("/read-current-camera-values")
    def read_current_camera_values_route() -> str:
        state: UiState = app.config["UI_STATE"]
        form = _parse_form(request.form)
        _normalize_subset_form(form, state.scan)
        form["resolved_output_path"] = _resolved_output_path_for_form(form, state.scan)
        state.update_form(form)
        _sync_task_output_path(state.task, str(form.get("resolved_output_path") or ""))
        if _process_is_alive(state.task.process):
            state.error = "Wait for the current command to finish before reading current camera values."
            state.message = None
            return render_page()
        output_root = Path(str(form.get("resolved_output_path") or "")).expanduser().resolve()
        commit_payload_path = review_commit_payload_path_for(output_root)
        if not commit_payload_path.exists():
            state.error = "No calibration commit payload exists for the current review output yet."
            state.message = None
            return render_page()
        requested_cameras = _requested_camera_tokens(form)
        report_path = _camera_state_report_path_for(output_root)
        assert report_path is not None
        try:
            report = _read_current_camera_values_report(
                commit_payload_path=commit_payload_path,
                report_path=report_path,
                requested_cameras=requested_cameras or None,
            )
        except Exception as exc:
            state.error = str(exc)
            state.message = None
            return render_page()
        state.error = None
        state.message = (
            f"Read current camera values for {int(report.get('connected_camera_count', 0) or 0)} of "
            f"{int(report.get('camera_count', 0) or 0)} payload cameras."
        )
        return render_page()

    @app.post("/apply-calibration-live")
    def apply_calibration_live_route() -> str:
        state: UiState = app.config["UI_STATE"]
        form = _parse_form(request.form)
        _normalize_subset_form(form, state.scan)
        form["resolved_output_path"] = _resolved_output_path_for_form(form, state.scan)
        state.update_form(form)
        _sync_task_output_path(state.task, str(form.get("resolved_output_path") or ""))
        if _process_is_alive(state.task.process):
            state.error = "Wait for the current command to finish before applying calibration."
            state.message = None
            return render_page()
        output_root = Path(str(form.get("resolved_output_path") or "")).expanduser().resolve()
        commit_payload_path = review_commit_payload_path_for(output_root)
        if not commit_payload_path.exists():
            state.error = "No calibration commit payload exists for the current review output yet."
            state.message = None
            return render_page()
        requested_cameras = [item.strip() for item in str(form.get("apply_cameras", "")).split(",") if item.strip()]
        report_path = output_root / "report" / "rcp2_live_apply_report.json"
        try:
            result = apply_calibration_payload(
                str(commit_payload_path),
                out_path=str(report_path),
                requested_cameras=requested_cameras or None,
                live=True,
            )
        except Exception as exc:
            state.error = str(exc)
            state.message = None
            return render_page()
        state.error = None
        state.message = f"Live apply finished with status: {result.get('operator_status') or result.get('status')}."
        return render_page()

    @app.post("/verify-last-push")
    def verify_last_push_route() -> str:
        state: UiState = app.config["UI_STATE"]
        form = _parse_form(request.form)
        _normalize_subset_form(form, state.scan)
        form["resolved_output_path"] = _resolved_output_path_for_form(form, state.scan)
        state.update_form(form)
        _sync_task_output_path(state.task, str(form.get("resolved_output_path") or ""))
        output_root = Path(str(form.get("resolved_output_path") or "")).expanduser().resolve()
        commit_payload_path = review_commit_payload_path_for(output_root)
        if not commit_payload_path.exists():
            state.error = "No calibration commit payload exists for the current review output yet."
            state.message = None
            return render_page()
        requested_cameras = _requested_camera_tokens(form)
        verification_report_path = _writeback_verification_report_path_for(output_root)
        assert verification_report_path is not None
        camera_state_report = _load_latest_camera_state_report(output_root)
        try:
            verification_report = build_camera_verification_report(
                str(commit_payload_path),
                out_path=str(verification_report_path),
                requested_cameras=requested_cameras or None,
                camera_state_report=camera_state_report,
            )
        except Exception as exc:
            state.error = str(exc)
            state.message = None
            return render_page()
        state.error = None
        mode = str(verification_report.get("verification_mode") or "unknown").replace("_", " ")
        state.message = f"Push verification refreshed: {verification_report.get('status')} via {mode}."
        return render_page()

    @app.post("/verify-apply")
    def verify_apply_route() -> str:
        state: UiState = app.config["UI_STATE"]
        form = _parse_form(request.form)
        _normalize_subset_form(form, state.scan)
        form["resolved_output_path"] = _resolved_output_path_for_form(form, state.scan)
        state.update_form(form)
        _sync_task_output_path(state.task, str(form.get("resolved_output_path") or ""))
        output_root = str(form.get("resolved_output_path") or "").strip()
        after_path = str(form.get("verification_after_path") or "").strip()
        if not output_root:
            state.error = "Output folder path is required before building a verification comparison."
            state.message = None
            return render_page()
        if not after_path:
            state.error = "After-apply review folder is required for verification comparison."
            state.message = None
            return render_page()
        try:
            result = build_post_apply_verification_from_reviews(
                output_root,
                after_path,
                out_path=str(post_apply_verification_path_for(output_root)),
            )
        except Exception as exc:
            state.error = str(exc)
            state.message = None
            return render_page()
        state.error = None
        state.message = f"Verification comparison finished with status: {result.get('status')}."
        return render_page()

    @app.post("/cancel-run")
    def cancel_run() -> Response:
        state: UiState = app.config["UI_STATE"]
        cancelled = _request_task_cancellation(state.task)
        state.error = None
        state.message = "Cancellation requested." if cancelled else "No active run to cancel."
        return jsonify({"ok": cancelled, "status": _status_payload(state.task)})

    @app.get("/status")
    def status() -> Response:
        state: UiState = app.config["UI_STATE"]
        return jsonify(_status_payload(state.task))

    @app.get("/scan-preview")
    def scan_preview() -> Response:
        state: UiState = app.config["UI_STATE"]
        record = _preferred_roi_preview_record(state.form, state.scan)
        candidate = _scan_preview_path(
            str(state.form.get("input_path", "")),
            clip_id=str((record or {}).get("clip_id") or ""),
            preview_mode=str(state.form.get("preview_mode") or "monitoring"),
        )
        if not candidate.exists():
            return redirect(url_for("index"))
        return send_file(candidate)

    @app.get("/artifact")
    def artifact() -> Response:
        path = request.args.get("path", "").strip()
        if not path:
            return redirect(url_for("index"))
        candidate = Path(path).expanduser().resolve()
        if not candidate.exists():
            return redirect(url_for("index"))
        return send_file(candidate)

    @app.get("/logo")
    def logo() -> Response:
        if not DEFAULT_LOGO_PATH.exists():
            return redirect(url_for("index"))
        return send_file(DEFAULT_LOGO_PATH)

    @app.post("/open-output")
    def open_output() -> Response:
        state: UiState = app.config["UI_STATE"]
        payload = _status_payload(state.task)
        output_path = str(payload.get("output_folder") or state.form.get("resolved_output_path") or state.form.get("output_path", "")).strip()
        if output_path:
            subprocess.run(["open", output_path], check=False)
        return jsonify({"ok": True})

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--check", action="store_true", help="Validate imports without starting the server")
    parser.add_argument("--open-browser", action="store_true", help="Open the UI in a browser after starting")
    args = parser.parse_args()
    if args.check:
        print("R3DMatch web UI imports OK")
        return
    app = create_app()
    url = f"http://{args.host}:{args.port}"
    print(f"R3DMatch web UI: {url}")
    if args.open_browser and os.environ.get("DISPLAY", "") != "":
        try:
            webbrowser.open(url)
        except Exception:
            pass
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
