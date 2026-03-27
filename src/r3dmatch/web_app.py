from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import threading
import time
import uuid
import webbrowser
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from flask import Flask, Response, jsonify, redirect, render_template_string, request, send_file, url_for

from .ftps_ingest import normalize_source_mode, plan_ftps_request, source_mode_label
from .identity import clip_id_from_path, subset_key_from_clip_id
from .matching import discover_clips
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
    resolve_review_output_dir,
    review_html_path_for,
    review_manifest_path_for,
    review_pdf_path_for,
    review_validation_path_for,
)


DEFAULT_LOGO_PATH = Path(__file__).resolve().parent / "static" / "r3dmatch_logo.png"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 5000
STALL_THRESHOLD_SECONDS = 20.0
ARTIFACT_FRESHNESS_TOLERANCE_SECONDS = 1.0


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
    .section-title { font-size: 20px; font-weight: 700; margin: 0 0 12px 0; }
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
    .roi-preview-wrap { margin-top: 12px; }
    .roi-preview-stage { position: relative; display: inline-block; max-width: 100%; border: 1px solid #d1d5db; border-radius: 8px; overflow: hidden; background: #000; }
    .roi-preview-stage img { display: block; max-width: 100%; height: auto; }
    .roi-overlay { position: absolute; border: 2px solid #2563eb; background: rgba(37,99,235,0.12); pointer-events: none; display: none; }
    .roi-canvas { position: absolute; inset: 0; cursor: crosshair; }
    .progress-track { background: #e5e7eb; height: 14px; border-radius: 999px; overflow: hidden; margin: 10px 0; }
    .progress-fill { background: #2563eb; height: 100%; width: 0%; transition: width 0.2s ease; }
    .stage-list { display: flex; flex-wrap: wrap; gap: 8px; margin: 8px 0 12px 0; padding: 0; list-style: none; }
    .stage-pill { padding: 6px 10px; border-radius: 999px; background: #e5e7eb; color: #374151; font-size: 13px; }
    .stage-pill.active { background: #2563eb; color: white; }
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
            <label for="matching_domain">Matching Domain</label>
            <select id="matching_domain" name="matching_domain">
              <option value="scene" {% if form.matching_domain == 'scene' %}selected{% endif %}>Scene-Referred (REDWideGamutRGB / Log3G10)</option>
              <option value="perceptual" {% if form.matching_domain == 'perceptual' %}selected{% endif %}>Perceptual (IPP2 / BT.709 / BT.1886)</option>
            </select>
            <div class="meta">Choose whether matching is solved in a camera-space style domain or the human review display domain.</div>
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

      <div class="card">
        <h2 class="section-title">ROI</h2>
        <div class="field">
          <label for="roi_mode">ROI Mode</label>
          <select id="roi_mode" name="roi_mode">
            {% for value, label in [('draw', 'draw'), ('center', 'center'), ('full', 'full'), ('manual', 'manual')] %}
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
            <div class="roi-preview-stage" id="roi-stage">
              <img id="roi-image" src="{{ url_for('scan_preview') }}" alt="ROI preview">
              <div id="roi-overlay" class="roi-overlay"></div>
              <div id="roi-canvas" class="roi-canvas"></div>
            </div>
          </div>
        {% else %}
          <div class="meta">Preview available after scan or first render.</div>
        {% endif %}
      </div>

      <div class="card">
        <h2 class="section-title">Strategies</h2>
        <div class="checkbox-row">
          <label><input type="checkbox" name="target_strategies" value="median" {% if 'median' in form.target_strategies %}checked{% endif %}> median</label>
          <label><input type="checkbox" name="target_strategies" value="brightest-valid" {% if 'brightest-valid' in form.target_strategies %}checked{% endif %}> brightest-valid</label>
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
            {% for value in ['calibration', 'monitoring'] %}
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
  </div>

  <script>
    function clamp(value, min, max) {
      return Math.max(min, Math.min(max, value));
    }

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
      const mode = document.getElementById('roi_mode').value;
      const manualRow = document.getElementById('manual-roi-row');
      const desc = document.getElementById('roi-description');
      const x = document.getElementById('roi_x');
      const y = document.getElementById('roi_y');
      const w = document.getElementById('roi_w');
      const h = document.getElementById('roi_h');
      if (mode === 'center') {
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
    document.getElementById('roi_mode').addEventListener('change', applyRoiMode);

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

    function updateOverlayFromInputs() {
      const stage = document.getElementById('roi-stage');
      const overlay = document.getElementById('roi-overlay');
      if (!stage || !overlay) return;
      const x = parseFloat(document.getElementById('roi_x').value || '0');
      const y = parseFloat(document.getElementById('roi_y').value || '0');
      const w = parseFloat(document.getElementById('roi_w').value || '0');
      const h = parseFloat(document.getElementById('roi_h').value || '0');
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
      document.getElementById('roi-description').textContent = 'Custom ROI: ' + pw + ' x ' + ph + ' at (' + px + ', ' + py + ')';
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
        if (document.getElementById('roi_mode').value !== 'draw') return;
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
    }
    applyRoiMode();
    applySourceMode();
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
        str(form["matching_domain"]),
        "--review-mode",
        str(form["review_mode"]),
        "--preview-mode",
        str(form["preview_mode"]),
    ]
    if source_mode == "ftps_camera_pull":
        args.extend(["--ftps-reel", str(form["ftps_reel"]), "--ftps-clips", str(form["ftps_clips"])])
        for camera in [item.strip() for item in str(form.get("ftps_cameras", "")).split(",") if item.strip()]:
            args.extend(["--ftps-camera", camera])
    roi_values = [form.get("roi_x"), form.get("roi_y"), form.get("roi_w"), form.get("roi_h")]
    if all(value not in (None, "") for value in roi_values):
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
    shell_command = f'cd "{repo_root}"; setenv PYTHONPATH "$PWD/src"; ' + " ".join(shlex.quote(item) for item in args)
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
        return {"clip_count": 0, "clip_ids": [], "sample_clip_ids": [], "remaining_count": 0, "clip_records": [], "clip_groups": [], "warning": None}
    if not root.exists() or not root.is_dir():
        return {"clip_count": 0, "clip_ids": [], "sample_clip_ids": [], "remaining_count": 0, "clip_records": [], "clip_groups": [], "warning": "Calibration folder does not exist."}
    clips = discover_clips(str(root))
    clip_ids = [clip_id_from_path(str(path)) for path in clips]
    clip_records = [
        {
            "clip_id": clip_id,
            "subset_group": subset_key_from_clip_id(clip_id),
            "camera_label": _camera_label_from_clip_id(clip_id),
        }
        for clip_id in clip_ids
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
        "matching_domain": "scene",
        "roi_x": "",
        "roi_y": "",
        "roi_w": "",
        "roi_h": "",
        "selected_clip_ids": [],
        "selected_clip_groups": [],
        "target_strategies": ["median", "brightest-valid"],
        "reference_clip_id": "",
        "hero_clip_id": "",
        "review_mode": "full_contact_sheet",
        "preview_mode": "calibration",
        "preview_lut": "",
        "roi_mode": "draw",
        "advanced_clip_selection": False,
    }


def _parse_form(post_data) -> Dict[str, object]:
    form = _default_form()
    for key in ["source_mode", "input_path", "output_path", "run_label", "ftps_reel", "ftps_clips", "ftps_cameras", "backend", "target_type", "processing_mode", "matching_domain", "roi_x", "roi_y", "roi_w", "roi_h", "reference_clip_id", "hero_clip_id", "review_mode", "preview_mode", "preview_lut", "roi_mode"]:
        form[key] = post_data.get(key, form[key]).strip()
    strategies = post_data.getlist("target_strategies")
    form["target_strategies"] = strategies or []
    form["selected_clip_ids"] = [str(item).strip() for item in post_data.getlist("selected_clip_ids") if str(item).strip()]
    form["selected_clip_groups"] = [str(item).strip() for item in post_data.getlist("selected_clip_groups") if str(item).strip()]
    form["advanced_clip_selection"] = post_data.get("advanced_clip_selection", "").strip().lower() in {"1", "true", "on", "yes"}
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
    roi_mode = str(form.get("roi_mode", "draw"))
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
    scan: Dict[str, object] = field(default_factory=lambda: {"clip_count": 0, "clip_ids": [], "sample_clip_ids": [], "remaining_count": 0, "clip_records": [], "clip_groups": [], "warning": None, "preview_available": False})
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
    selected_clip_ids = _resolve_selected_clip_ids(form, scan)
    discovered_clip_ids = [str(item) for item in scan.get("clip_ids", [])]
    selected_clip_groups = [str(item) for item in form.get("selected_clip_groups", []) if str(item).strip()]
    if (
        not str(form.get("run_label") or "").strip()
        and not selected_clip_groups
        and selected_clip_ids
        and selected_clip_ids == discovered_clip_ids
    ):
        return str(form["output_path"])
    return resolve_review_output_dir(
        str(form["output_path"]),
        run_label=str(form.get("run_label") or "").strip() or None,
        selected_clip_ids=selected_clip_ids,
        selected_clip_groups=selected_clip_groups,
    )


def _roi_description(form: Dict[str, object]) -> str:
    mode = str(form.get("roi_mode", "draw"))
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


def _scan_preview_path(input_path: str) -> Path:
    token = hashlib.sha1(str(Path(input_path).expanduser().resolve()).encode("utf-8")).hexdigest()[:12]
    return Path("/tmp") / f"r3dmatch_web_scan_preview_{token}.jpg"


def _ensure_scan_preview(input_path: str, scan: Dict[str, object]) -> Optional[str]:
    first_clip = scan.get("first_clip_path")
    if not first_clip:
        return None
    output_path = _scan_preview_path(str(input_path))
    if output_path.exists():
        return str(output_path)
    redline_executable = _resolve_redline_executable()
    capabilities = _detect_redline_capabilities(redline_executable)
    settings = _normalize_preview_settings(
        preview_mode="calibration",
        preview_output_space=None,
        preview_output_gamma=None,
        preview_highlight_rolloff=None,
        preview_shadow_rolloff=None,
        preview_lut=None,
    )
    command = _build_redline_preview_command(
        str(first_clip),
        output_path=str(output_path),
        frame_index=0,
        exposure_stops=None,
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
        return None
    if not output_path.exists():
        generated_matches = sorted(output_path.parent.glob(f"{output_path.name}.*"))
        if generated_matches:
            generated_matches[0].replace(output_path)
    return str(output_path) if output_path.exists() else None


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


def _infer_progress(task: TaskState) -> None:
    now = time.time()
    process_alive = _process_is_alive(task.process)
    output_root = Path(task.output_path).expanduser().resolve() if task.output_path else None
    is_review = "review-calibration" in task.command
    is_approve = "approve-master-rmd" in task.command
    review_validation = _load_review_validation(output_root, started_at=task.started_at) if is_review else None
    review_manifest = _load_review_manifest(output_root, started_at=task.started_at) if is_review else None
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
            strategies_count = max(1, len([item for item in task.strategies_text.split(", ") if item])) if task.strategies_text else 1
            expected_previews = clip_count * (1 + strategies_count * 3) if clip_count and resolved_review_mode == "full_contact_sheet" else 0
            task.clip_count = clip_count or task.clip_count
            if resolved_review_mode == "lightweight_analysis":
                task.items_total = clip_count or task.items_total
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
    output_root = Path(str(base["output_path"])).expanduser().resolve() if base.get("output_path") else None
    review_validation = _load_review_validation(output_root, started_at=float(base.get("started_at") or 0.0)) if "review-calibration" in base["command"] else None
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
    if not status_detail and "review-calibration" in base["command"] and base.get("status") == "failed" and base.get("returncode") == 0 and review_validation is None:
        status_detail = "Fresh review_validation.json was not produced for this run."
    base.update(
        {
            "stage_names": stage_names,
            "progress_percent": progress_percent,
            "preview_pdf_url": url_for("artifact", path=base["preview_pdf_path"]) if base.get("preview_pdf_path") else None,
            "preview_html_url": url_for("artifact", path=base["preview_html_path"]) if base.get("preview_html_path") else None,
            "output_folder": base.get("output_path"),
            "can_cancel": can_cancel,
            "watchdog_status": base["status"],
            "last_activity_seconds": last_activity_seconds,
            "review_mode_label": review_mode_label(resolved_review_mode) if resolved_review_mode else "",
            "source_mode_label": source_mode_label(str(base.get("source_mode") or "local_folder")) if base.get("source_mode") else "",
            "status_detail": status_detail,
            "review_validation": review_validation,
            "validation_status": review_validation.get("status") if isinstance(review_validation, dict) else None,
            "validation_errors": validation_errors,
            "validation_warnings": validation_warnings,
        }
    )
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
        current_output = str(state.task.output_path or _resolved_output_path_for_form(state.form, state.scan) or state.form.get("output_path", ""))
        task_payload = _status_payload(state.task)
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
        preview_path = _ensure_scan_preview(str(form["input_path"]), state.scan) if state.scan.get("first_clip_path") else None
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
        preview_path = _ensure_scan_preview(str(form["input_path"]), state.scan) if state.scan.get("first_clip_path") else None
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
        candidate = _scan_preview_path(str(state.form.get("input_path", "")))
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
        output_path = str(state.task.output_path or state.form.get("resolved_output_path") or state.form.get("output_path", "")).strip()
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
