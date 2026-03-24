from __future__ import annotations

import argparse
import hashlib
import os
import shlex
import subprocess
import threading
import webbrowser
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from flask import Flask, Response, jsonify, redirect, render_template_string, request, send_file, url_for

from .identity import clip_id_from_path
from .matching import discover_clips
from .report import (
    _build_redline_preview_command,
    _detect_redline_capabilities,
    _normalize_preview_settings,
    _resolve_redline_executable,
)


DEFAULT_LOGO_PATH = Path(__file__).resolve().parent / "static" / "r3dmatch_logo.png"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 5000


PAGE_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>R3DMatch Internal Review</title>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #f6f7f9; color: #111827; margin: 0; }
    .page { max-width: 1100px; margin: 0 auto; padding: 24px; }
    .header { background: white; border: 1px solid #d1d5db; border-radius: 10px; padding: 18px; margin-bottom: 16px; }
    .header-row { display: flex; align-items: center; gap: 16px; }
    .header img { max-width: 300px; max-height: 120px; object-fit: contain; }
    .title { font-size: 28px; font-weight: 700; margin: 0; }
    .subtitle { font-size: 15px; color: #4b5563; margin: 4px 0 0 0; }
    .instructions { margin-top: 12px; font-size: 15px; }
    .card { background: white; border: 1px solid #d1d5db; border-radius: 10px; padding: 16px; margin-bottom: 14px; }
    .section-title { font-size: 18px; font-weight: 600; margin: 0 0 12px 0; }
    .field { margin-bottom: 12px; }
    .field label { display: block; font-weight: 600; margin-bottom: 6px; }
    .field input[type="text"], .field select { width: 100%; box-sizing: border-box; padding: 10px; border: 1px solid #c7ccd4; border-radius: 8px; background: white; }
    .row { display: flex; gap: 12px; }
    .row > * { flex: 1; }
    .checkbox-row { display: flex; gap: 18px; align-items: center; flex-wrap: wrap; }
    .actions { display: flex; gap: 10px; flex-wrap: wrap; }
    button, .link-button { background: #111827; color: white; border: 0; border-radius: 8px; padding: 10px 14px; font-weight: 600; cursor: pointer; text-decoration: none; display: inline-block; }
    button.secondary, .link-button.secondary { background: #4b5563; }
    button:disabled { opacity: 0.5; cursor: not-allowed; }
    .status { padding: 10px 12px; border-radius: 8px; margin-bottom: 12px; }
    .status.error { background: #fef2f2; color: #991b1b; border: 1px solid #fecaca; }
    .status.info { background: #eff6ff; color: #1d4ed8; border: 1px solid #bfdbfe; }
    .summary-box { background: #f9fafb; border: 1px solid #e5e7eb; border-radius: 8px; padding: 12px; }
    .clip-list { margin-top: 8px; padding-left: 20px; max-height: 180px; overflow: auto; }
    pre.log { background: #0f172a; color: #e5e7eb; border-radius: 8px; padding: 14px; min-height: 220px; overflow: auto; white-space: pre-wrap; }
    .meta { color: #4b5563; font-size: 14px; margin-top: 8px; }
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
        <div class="field">
          <label for="input_path">Calibration Folder Path</label>
          <input id="input_path" name="input_path" type="text" value="{{ form.input_path }}">
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
        </div>
        <div class="field" style="margin-top: 12px;">
          <label for="reference_clip_id">Manual Reference Clip ID</label>
          <input id="reference_clip_id" name="reference_clip_id" type="text" value="{{ form.reference_clip_id }}">
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
      <div class="meta"><strong>Clips Found:</strong> <span id="task-clip-count">{{ task.clip_count }}</span></div>
      <div class="meta"><strong>Strategies:</strong> <span id="task-strategies">{{ task.strategies_text }}</span></div>
      <div class="meta"><strong>Preview Mode:</strong> <span id="task-preview-mode">{{ task.preview_mode }}</span></div>
      <div class="meta"><strong>Output Folder:</strong> <span id="task-output">{{ task.output_path or form.output_path }}</span></div>
      <div class="progress-track"><div id="task-progress-fill" class="progress-fill" style="width: {{ task.progress_percent }}%;"></div></div>
      <div class="meta"><span id="task-counts">{{ task.items_completed }} / {{ task.items_total }}</span></div>
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

    async function refreshStatus() {
      const response = await fetch('{{ url_for("status") }}');
      const data = await response.json();
      document.getElementById('task-status').textContent = data.status;
      document.getElementById('task-stage').textContent = data.stage || '';
      document.getElementById('task-clip-count').textContent = data.clip_count ?? '';
      document.getElementById('task-strategies').textContent = data.strategies_text || '';
      document.getElementById('task-preview-mode').textContent = data.preview_mode || '';
      document.getElementById('task-output').textContent = data.output_path || '';
      document.getElementById('task-command').textContent = data.command || '';
      document.getElementById('task-log').textContent = data.log_text || '';
      document.getElementById('task-progress-fill').style.width = (data.progress_percent || 0) + '%';
      document.getElementById('task-counts').textContent = (data.items_completed || 0) + ' / ' + (data.items_total || 0);
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
    wireDrawRoi();
    updateOverlayFromInputs();
    setInterval(refreshStatus, 1000);
  </script>
</body>
</html>
"""


def build_review_web_command(repo_root: str, form: Dict[str, object]) -> List[str]:
    args = [
        "python3",
        "-m",
        "r3dmatch.cli",
        "review-calibration",
        str(form["input_path"]),
        "--out",
        str(form["output_path"]),
        "--backend",
        str(form["backend"]),
        "--target-type",
        str(form["target_type"]),
        "--processing-mode",
        str(form["processing_mode"]),
        "--preview-mode",
        str(form["preview_mode"]),
    ]
    roi_values = [form.get("roi_x"), form.get("roi_y"), form.get("roi_w"), form.get("roi_h")]
    if all(value not in (None, "") for value in roi_values):
        args.extend(["--roi-x", str(form["roi_x"]), "--roi-y", str(form["roi_y"]), "--roi-w", str(form["roi_w"]), "--roi-h", str(form["roi_h"])])
    for strategy in form["target_strategies"]:
        args.extend(["--target-strategy", str(strategy)])
    if form.get("reference_clip_id"):
        args.extend(["--reference-clip-id", str(form["reference_clip_id"])])
    if form.get("preview_lut"):
        args.extend(["--preview-lut", str(form["preview_lut"])])
    shell_command = f'cd "{repo_root}"; setenv PYTHONPATH "$PWD/src"; ' + " ".join(shlex.quote(item) for item in args)
    return ["/bin/tcsh", "-c", shell_command]


def build_approve_web_command(repo_root: str, form: Dict[str, object]) -> List[str]:
    strategy = form["target_strategies"][0] if form["target_strategies"] else "median"
    args = [
        "python3",
        "-m",
        "r3dmatch.cli",
        "approve-master-rmd",
        str(form["output_path"]),
        "--target-strategy",
        str(strategy),
    ]
    if form.get("reference_clip_id"):
        args.extend(["--reference-clip-id", str(form["reference_clip_id"])])
    shell_command = f'cd "{repo_root}"; setenv PYTHONPATH "$PWD/src"; ' + " ".join(shlex.quote(item) for item in args)
    return ["/bin/tcsh", "-c", shell_command]


def build_clear_cache_web_command(repo_root: str, form: Dict[str, object]) -> List[str]:
    args = ["python3", "-m", "r3dmatch.cli", "clear-preview-cache", str(form["output_path"])]
    shell_command = f'cd "{repo_root}"; setenv PYTHONPATH "$PWD/src"; ' + " ".join(shlex.quote(item) for item in args)
    return ["/bin/tcsh", "-c", shell_command]


def scan_sources(input_path: str) -> Dict[str, object]:
    root = Path(input_path).expanduser().resolve()
    if not input_path.strip():
        return {"clip_count": 0, "sample_clip_ids": [], "remaining_count": 0, "warning": None}
    if not root.exists() or not root.is_dir():
        return {"clip_count": 0, "sample_clip_ids": [], "remaining_count": 0, "warning": "Calibration folder does not exist."}
    clips = discover_clips(str(root))
    clip_ids = [clip_id_from_path(str(path)) for path in clips]
    return {
        "clip_count": len(clips),
        "sample_clip_ids": clip_ids[:12],
        "remaining_count": max(0, len(clip_ids) - 12),
        "first_clip_path": str(clips[0]) if clips else None,
        "warning": None if clips else "No valid RED .R3D clips were found in the calibration folder.",
    }


def _default_form() -> Dict[str, object]:
    return {
        "input_path": "",
        "output_path": "",
        "backend": "red",
        "target_type": "gray_sphere",
        "processing_mode": "both",
        "roi_x": "",
        "roi_y": "",
        "roi_w": "",
        "roi_h": "",
        "target_strategies": ["median", "brightest-valid"],
        "reference_clip_id": "",
        "preview_mode": "calibration",
        "preview_lut": "",
        "roi_mode": "draw",
    }


def _parse_form(post_data) -> Dict[str, object]:
    form = _default_form()
    for key in ["input_path", "output_path", "backend", "target_type", "processing_mode", "roi_x", "roi_y", "roi_w", "roi_h", "reference_clip_id", "preview_mode", "preview_lut", "roi_mode"]:
        form[key] = post_data.get(key, form[key]).strip()
    strategies = post_data.getlist("target_strategies")
    form["target_strategies"] = strategies or []
    return form


def _validate_form(form: Dict[str, object], *, require_output: bool = True, require_source: bool = True) -> Optional[str]:
    if require_source:
        input_path = str(form["input_path"]).strip()
        if not input_path:
            return "Calibration folder path is required."
        scan = scan_sources(input_path)
        if scan["warning"]:
            return str(scan["warning"])
    if require_output and not str(form["output_path"]).strip():
        return "Output folder path is required."
    roi_mode = str(form.get("roi_mode", "draw"))
    if roi_mode in {"draw", "manual"}:
        roi_values = [str(form.get("roi_x", "")).strip(), str(form.get("roi_y", "")).strip(), str(form.get("roi_w", "")).strip(), str(form.get("roi_h", "")).strip()]
        if not all(roi_values):
            return "ROI is required for draw/manual mode."
    if not form["target_strategies"]:
        return "At least one target strategy is required."
    if "manual" in form["target_strategies"] and not str(form["reference_clip_id"]).strip():
        return "Manual strategy requires a reference clip ID."
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
    logs: List[str] = field(default_factory=lambda: ["Ready.\n"])
    returncode: Optional[int] = None
    stage: str = "Idle"
    stage_index: int = 0
    total_stages: int = 0
    items_completed: int = 0
    items_total: int = 0
    clip_count: int = 0
    strategies_text: str = ""
    preview_mode: str = ""
    preview_pdf_path: Optional[str] = None
    preview_html_path: Optional[str] = None
    report_ready: bool = False
    lock: threading.RLock = field(default_factory=threading.RLock)

    def append(self, text: str) -> None:
        with self.lock:
            self.logs.append(text)

    def snapshot(self) -> Dict[str, object]:
        with self.lock:
            return {
                "status": self.status,
                "command": self.command,
                "output_path": self.output_path,
                "returncode": self.returncode,
                "stage": self.stage,
                "stage_index": self.stage_index,
                "total_stages": self.total_stages,
                "items_completed": self.items_completed,
                "items_total": self.items_total,
                "clip_count": self.clip_count,
                "strategies_text": self.strategies_text,
                "preview_mode": self.preview_mode,
                "preview_pdf_path": self.preview_pdf_path,
                "preview_html_path": self.preview_html_path,
                "report_ready": self.report_ready,
                "log_text": "".join(self.logs)[-50000:],
            }


@dataclass
class UiState:
    form: Dict[str, object] = field(default_factory=_default_form)
    scan: Dict[str, object] = field(default_factory=lambda: {"clip_count": 0, "sample_clip_ids": [], "remaining_count": 0, "warning": None, "preview_available": False})
    task: TaskState = field(default_factory=TaskState)
    error: Optional[str] = None
    message: Optional[str] = None
    lock: threading.Lock = field(default_factory=threading.Lock)

    def update_form(self, form: Dict[str, object]) -> None:
        with self.lock:
            self.form = form


def _report_url_for_output(output_path: str) -> Optional[str]:
    if not output_path.strip():
        return None
    report_pdf = Path(output_path).expanduser().resolve() / "report" / "preview_contact_sheet.pdf"
    if report_pdf.exists():
        return url_for("artifact", path=str(report_pdf))
    report_html = Path(output_path).expanduser().resolve() / "report" / "contact_sheet.html"
    if report_html.exists():
        return url_for("artifact", path=str(report_html))
    return None


def _scan_summary_text(scan: Dict[str, object]) -> str:
    if scan.get("warning"):
        return str(scan["warning"])
    return f"Found {scan.get('clip_count', 0)} RED clips"


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
        exposure_stops=0.0,
        rgb_gains=None,
        redline_executable=redline_executable,
        preview_settings=settings,
        redline_capabilities=capabilities,
        look_metadata_path=None,
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


def _infer_progress(task: TaskState) -> None:
    output_root = Path(task.output_path).expanduser().resolve() if task.output_path else None
    if not output_root or not output_root.exists():
        return
    is_review = "review-calibration" in task.command
    is_approve = "approve-master-rmd" in task.command
    if is_review:
        task.total_stages = len(REVIEW_STAGES)
        preview_pdf = output_root / "report" / "preview_contact_sheet.pdf"
        preview_html = output_root / "report" / "contact_sheet.html"
        analysis_dir = output_root / "analysis"
        array_cal = output_root / "array_calibration.json"
        previews_dir = output_root / "previews"
        visible_previews = [p for p in previews_dir.glob("*.jpg") if ".000000." not in p.name] if previews_dir.exists() else []
        clip_count = len(list(analysis_dir.glob("*.analysis.json"))) if analysis_dir.exists() else 0
        strategies_count = max(1, len(task.strategies_text.split(", "))) if task.strategies_text else 1
        expected_previews = clip_count * (1 + strategies_count * 3) if clip_count else 0
        task.clip_count = clip_count or task.clip_count
        task.items_total = expected_previews or task.items_total
        task.items_completed = len(visible_previews) if expected_previews else clip_count
        if preview_pdf.exists():
            task.stage_index = len(REVIEW_STAGES) - 1
            task.stage = REVIEW_STAGES[-1]
        elif (output_root / "report" / "contact_sheet.json").exists():
            task.stage_index = 4
            task.stage = REVIEW_STAGES[4]
        elif visible_previews:
            task.stage_index = 3
            task.stage = REVIEW_STAGES[3]
        elif array_cal.exists():
            task.stage_index = 2
            task.stage = REVIEW_STAGES[2]
        elif clip_count:
            task.stage_index = 1
            task.stage = REVIEW_STAGES[1]
        else:
            task.stage_index = 0
            task.stage = REVIEW_STAGES[0]
        task.preview_pdf_path = str(preview_pdf) if preview_pdf.exists() else None
        task.preview_html_path = str(preview_html) if preview_html.exists() else None
        task.report_ready = preview_pdf.exists() or preview_html.exists()
    elif is_approve:
        task.total_stages = len(APPROVAL_STAGES)
        approval_root = output_root / "approval"
        master_dir = approval_root / "Master_RMD"
        approval_pdf = approval_root / "calibration_report.pdf"
        task.items_completed = len(list(master_dir.glob("*.RMD"))) if master_dir.exists() else 0
        task.items_total = task.items_completed or task.items_total
        if approval_pdf.exists():
            task.stage_index = len(APPROVAL_STAGES) - 1
            task.stage = APPROVAL_STAGES[-1]
        elif master_dir.exists() and task.items_completed:
            task.stage_index = 1
            task.stage = APPROVAL_STAGES[1]
        else:
            task.stage_index = 0
            task.stage = APPROVAL_STAGES[0]
        task.preview_pdf_path = str(approval_pdf) if approval_pdf.exists() else None
        task.preview_html_path = None
        task.report_ready = approval_pdf.exists()
    if task.returncode == 0:
        task.status = "completed"
    if task.returncode not in (None, 0):
        task.status = "failed"


def _status_payload(task: TaskState) -> Dict[str, object]:
    with task.lock:
        _infer_progress(task)
        base = task.snapshot()
    stage_names = REVIEW_STAGES if "review-calibration" in base["command"] else APPROVAL_STAGES if "approve-master-rmd" in base["command"] else []
    progress_percent = int(((base["stage_index"] + (1 if base["status"] == "completed" else 0)) / max(len(stage_names), 1)) * 100) if stage_names else 0
    base.update(
        {
            "stage_names": stage_names,
            "progress_percent": progress_percent,
            "preview_pdf_url": url_for("artifact", path=base["preview_pdf_path"]) if base.get("preview_pdf_path") else None,
            "preview_html_url": url_for("artifact", path=base["preview_html_path"]) if base.get("preview_html_path") else None,
            "output_folder": base.get("output_path"),
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
    preview_mode: str = "",
) -> None:
    task = state.task
    snapshot = task.snapshot()
    if snapshot["status"] == "running":
        state.error = "Another command is already running."
        return
    with task.lock:
        task.status = "running"
        task.command = " ".join(command)
        task.output_path = output_path
        task.logs = [f"$ {' '.join(command)}\n"]
        task.returncode = None
        task.clip_count = clip_count
        task.strategies_text = strategies_text
        task.preview_mode = preview_mode
        task.preview_pdf_path = None
        task.preview_html_path = None
        task.report_ready = False
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
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        assert process.stdout is not None
        for line in process.stdout:
            task.append(line)
        returncode = process.wait()
        with task.lock:
            task.returncode = returncode
            task.status = "completed" if returncode == 0 else "failed"
            task.logs.append(f"\n[exit] {returncode}\n")

    threading.Thread(target=worker, daemon=True).start()


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["REPO_ROOT"] = str(Path(__file__).resolve().parents[2])
    app.config["UI_STATE"] = UiState()

    def render_page() -> str:
        state: UiState = app.config["UI_STATE"]
        report_url = _report_url_for_output(str(state.form.get("output_path", "")))
        return render_template_string(
            PAGE_TEMPLATE,
            has_logo=DEFAULT_LOGO_PATH.exists(),
            form=state.form,
            scan=state.scan,
            scan_summary_text=_scan_summary_text(state.scan),
            task=_status_payload(state.task),
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
        state.update_form(form)
        state.scan = scan_sources(str(form["input_path"]))
        preview_path = _ensure_scan_preview(str(form["input_path"]), state.scan)
        state.scan["preview_available"] = bool(preview_path)
        state.error = state.scan["warning"] if state.scan.get("warning") else None
        state.message = None if state.error else f"Scanned calibration folder: found {state.scan['clip_count']} RED clips."
        return render_page()

    @app.post("/run-review")
    def run_review() -> str:
        state: UiState = app.config["UI_STATE"]
        form = _parse_form(request.form)
        state.update_form(form)
        state.scan = scan_sources(str(form["input_path"]))
        preview_path = _ensure_scan_preview(str(form["input_path"]), state.scan)
        state.scan["preview_available"] = bool(preview_path)
        error = _validate_form(form)
        state.error = error
        state.message = None
        if error is None:
            command = build_review_web_command(app.config["REPO_ROOT"], form)
            _start_command_task(
                state,
                command,
                str(form["output_path"]),
                clip_count=int(state.scan.get("clip_count", 0) or 0),
                strategies_text=", ".join(str(value) for value in form.get("target_strategies", [])),
                preview_mode=str(form.get("preview_mode", "")),
            )
            state.message = "Review command started."
        return render_page()

    @app.post("/approve-master-rmd")
    def approve_master() -> str:
        state: UiState = app.config["UI_STATE"]
        form = _parse_form(request.form)
        state.update_form(form)
        error = _validate_form(form, require_source=False, require_output=True)
        state.error = error
        state.message = None
        if error is None:
            command = build_approve_web_command(app.config["REPO_ROOT"], form)
            _start_command_task(
                state,
                command,
                str(form["output_path"]),
                clip_count=int(state.scan.get("clip_count", 0) or 0),
                strategies_text=", ".join(str(value) for value in form.get("target_strategies", [])),
                preview_mode=str(form.get("preview_mode", "")),
            )
            state.message = "Approve Master RMD command started."
        return render_page()

    @app.post("/clear-preview-cache")
    def clear_preview() -> str:
        state: UiState = app.config["UI_STATE"]
        form = _parse_form(request.form)
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
            str(form["output_path"]),
            clip_count=int(state.scan.get("clip_count", 0) or 0),
            strategies_text=", ".join(str(value) for value in form.get("target_strategies", [])),
            preview_mode=str(form.get("preview_mode", "")),
        )
        return render_page()

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
        output_path = str(state.form.get("output_path", "")).strip()
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
