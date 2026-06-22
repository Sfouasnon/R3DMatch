"""
R3DMatch v2 — Web Application

Flask-based UI server. Serves the single-page web interface and
exposes a REST API that the frontend calls.

Routes:
  GET  /                    — main UI
  GET  /api/health          — runtime health check
  POST /api/run             — start a new analysis run
  GET  /api/run/<run_id>    — get run status/results
  POST /api/roi             — commit a manual ROI for a clip
  GET  /api/report/<run_id> — get the contact sheet HTML path
  POST /api/push/<run_id>   — (stub) trigger RCP2 push
"""
from __future__ import annotations

import json
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Dict, Optional

from flask import Flask, jsonify, request, send_file, send_from_directory

from .redline import check_redline_available, resolve_redline_executable
from .report import build_report
from .workflow import run_analysis

app = Flask(__name__, static_folder=None)

# In-memory run registry
_runs: Dict[str, Dict] = {}
_runs_lock = threading.Lock()

DEFAULT_OUT_BASE = Path.home() / "Desktop" / "R3DMatch_Runs"


# ---------------------------------------------------------------------------
# Static UI
# ---------------------------------------------------------------------------

_UI_HTML = None


def _get_ui_html() -> str:
    global _UI_HTML
    if _UI_HTML is None:
        ui_path = Path(__file__).parent / "ui" / "index.html"
        if ui_path.exists():
            _UI_HTML = ui_path.read_text(encoding="utf-8")
        else:
            _UI_HTML = _fallback_ui_html()
    return _UI_HTML


@app.route("/")
def index():
    from flask import Response
    return Response(_get_ui_html(), mimetype="text/html")


@app.route("/report/<run_id>")
def serve_report(run_id: str):
    with _runs_lock:
        run_info = _runs.get(run_id)
    if not run_info:
        return jsonify({"error": "run not found"}), 404
    report_path = run_info.get("report_html")
    if not report_path or not Path(report_path).exists():
        return jsonify({"error": "report not yet generated"}), 404
    return send_file(report_path)


# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------

@app.route("/api/health")
def api_health():
    try:
        redline = resolve_redline_executable()
        rl = check_redline_available(redline)
    except Exception as exc:
        rl = {"ready": False, "error": str(exc), "build": "", "sdk_version": ""}
    return jsonify({
        "redline_ready": rl["ready"],
        "redline_build": rl.get("build", ""),
        "redline_sdk_version": rl.get("sdk_version", ""),
        "redline_error": rl.get("error", ""),
        "status": "ok" if rl["ready"] else "redline_not_ready",
    })


@app.route("/api/run", methods=["POST"])
def api_start_run():
    payload = request.get_json(force=True) or {}
    input_path = str(payload.get("input_path") or "").strip()
    if not input_path:
        return jsonify({"error": "input_path is required"}), 400
    if not Path(input_path).exists():
        return jsonify({"error": f"input_path does not exist: {input_path}"}), 400

    run_id = str(uuid.uuid4())[:8]
    out_dir = str(Path(payload.get("out_dir") or DEFAULT_OUT_BASE) / run_id)
    manual_roi_file = payload.get("manual_roi_file") or None
    reuse_renders = bool(payload.get("reuse_renders", True))
    render_corrected = bool(payload.get("render_corrected", True))

    run_info = {
        "run_id": run_id,
        "status": "running",
        "input_path": input_path,
        "out_dir": out_dir,
        "started_at": time.time(),
        "progress": [],
        "result": None,
        "error": None,
        "report_html": None,
    }
    with _runs_lock:
        _runs[run_id] = run_info

    def _run_thread():
        try:
            result = run_analysis(
                input_path,
                out_dir=out_dir,
                run_id=run_id,
                manual_roi_file=manual_roi_file,
                reuse_renders=reuse_renders,
                render_corrected=render_corrected,
            )
            # Generate contact sheet (build_report returns the written path)
            report_dir = Path(out_dir) / "report"
            report_dir.mkdir(parents=True, exist_ok=True)
            report_path = build_report(result, str(report_dir))

            with _runs_lock:
                run_info["status"] = "complete"
                run_info["result"] = _summarize_result(result)
                run_info["report_html"] = report_path
        except Exception as exc:
            with _runs_lock:
                run_info["status"] = "error"
                run_info["error"] = str(exc)

    t = threading.Thread(target=_run_thread, daemon=True)
    t.start()

    return jsonify({"run_id": run_id, "status": "running", "out_dir": out_dir})


@app.route("/api/run/<run_id>")
def api_get_run(run_id: str):
    with _runs_lock:
        run_info = _runs.get(run_id)
    if not run_info:
        return jsonify({"error": "run not found"}), 404
    return jsonify({
        "run_id": run_id,
        "status": run_info["status"],
        "error": run_info.get("error"),
        "result": run_info.get("result"),
        "report_url": f"/report/{run_id}" if run_info.get("report_html") else None,
    })


@app.route("/api/runs")
def api_list_runs():
    with _runs_lock:
        runs = [
            {
                "run_id": rid,
                "status": info["status"],
                "input_path": info["input_path"],
                "started_at": info["started_at"],
            }
            for rid, info in _runs.items()
        ]
    return jsonify({"runs": sorted(runs, key=lambda x: x["started_at"], reverse=True)})


@app.route("/api/roi/<run_id>", methods=["POST"])
def api_commit_roi(run_id: str):
    """
    Commit a manual ROI for a clip. Saves to a JSON file the workflow can load.
    """
    payload = request.get_json(force=True) or {}
    clip_id = str(payload.get("clip_id") or "").strip()
    cx = payload.get("cx")
    cy = payload.get("cy")
    r = payload.get("r")
    if not clip_id or cx is None or cy is None or r is None:
        return jsonify({"error": "clip_id, cx, cy, r are required"}), 400

    with _runs_lock:
        run_info = _runs.get(run_id)
    if not run_info:
        return jsonify({"error": "run not found"}), 404

    roi_file = Path(run_info["out_dir"]) / "manual_rois.json"
    roi_file.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if roi_file.exists():
        try:
            existing = json.loads(roi_file.read_text(encoding="utf-8"))
        except Exception:
            existing = {}
    existing[clip_id] = {"cx": float(cx), "cy": float(cy), "r": float(r)}
    roi_file.write_text(json.dumps(existing, indent=2), encoding="utf-8")

    return jsonify({"ok": True, "clip_id": clip_id, "roi_file": str(roi_file)})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _summarize_result(result) -> Dict:
    """Convert RunResult to a JSON-safe summary dict."""
    cameras = []
    for cr in result.cameras:
        cameras.append({
            "clip_id": cr.clip_id,
            "camera_label": cr.camera_label,
            "status": cr.status,
            "hero_ire": cr.measurement.hero_ire if cr.measurement else None,
            "corrected_ire": cr.corrected_ire,
            "exposure_adjust": cr.commit.exposure_adjust if cr.commit else None,
            "kelvin": cr.commit.kelvin if cr.commit else None,
            "tint": cr.commit.tint if cr.commit else None,
            "detection_source": cr.detection.source if cr.detection else "failed",
            "exposure_status": cr.exposure_closed_loop_status,
            "wb_status": cr.wb_closed_loop_status,
            "match_pct": cr.match_pct,
            "failed_stage": cr.failed_stage,
            "failure_reason": cr.failure_reason,
        })
    wb = result.wb_solve
    return {
        "run_id": result.run_id,
        "assessment_status": result.assessment_status,
        "array_match_pct": result.array_match_pct,
        "min_match_pct": result.min_match_pct,
        "min_match_clip_id": result.min_match_clip_id,
        "solved_count": result.solved_count,
        "camera_count": len(result.cameras),
        "anchor_ire": result.anchor_ire,
        "shared_kelvin": wb.shared_kelvin if wb else None,
        "wb_status": wb.status if wb else "SKIPPED",
        "wc_spread_before": wb.wc_spread_before if wb else None,
        "wc_spread_after": wb.wc_spread_after if wb else None,
        "gm_spread_before": wb.gm_spread_before if wb else None,
        "gm_spread_after": wb.gm_spread_after if wb else None,
        "operator_recommendation": result.operator_recommendation,
        "cameras": cameras,
    }


def _fallback_ui_html() -> str:
    """Minimal UI when the ui/index.html is not present."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>R3DMatch v2</title>
<style>
body { font-family: -apple-system, sans-serif; background: #0f172a; color: #e2e8f0;
       display: flex; flex-direction: column; align-items: center; padding: 40px 20px; min-height: 100vh; }
h1 { font-size: 28px; font-weight: 700; margin-bottom: 8px; }
.sub { color: #64748b; margin-bottom: 40px; }
.card { background: #1e293b; border-radius: 10px; padding: 28px; width: 100%; max-width: 560px;
        border: 1px solid #334155; margin-bottom: 20px; }
label { display: block; font-size: 12px; color: #94a3b8; margin-bottom: 6px; text-transform: uppercase; }
input { width: 100%; padding: 10px 12px; background: #0f172a; border: 1px solid #334155;
        border-radius: 6px; color: #f1f5f9; font-size: 14px; margin-bottom: 16px; }
button { background: #3b82f6; color: #fff; border: none; border-radius: 6px;
         padding: 10px 24px; font-size: 14px; cursor: pointer; font-weight: 600; }
button:hover { background: #2563eb; }
#status { margin-top: 20px; color: #94a3b8; font-size: 13px; min-height: 24px; }
#result { margin-top: 16px; }
a { color: #60a5fa; }
</style>
</head>
<body>
<h1>R3DMatch v2</h1>
<p class="sub">Multi-Camera Exposure & Color Alignment</p>

<div class="card">
  <label>Source Folder (containing R3D files)</label>
  <input type="text" id="input-path" placeholder="/path/to/media/folder">
  <label>Output Folder</label>
  <input type="text" id="out-dir" placeholder="/path/to/output">
  <button onclick="startRun()">Run Analysis</button>
</div>

<div id="status"></div>
<div id="result"></div>

<script>
let currentRunId = null;
let pollInterval = null;

async function startRun() {
  const inputPath = document.getElementById('input-path').value.trim();
  const outDir = document.getElementById('out-dir').value.trim();
  if (!inputPath) { setStatus('Enter a source folder path.'); return; }
  setStatus('Starting analysis...');
  try {
    const res = await fetch('/api/run', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({input_path: inputPath, out_dir: outDir || undefined}),
    });
    const data = await res.json();
    if (data.error) { setStatus('Error: ' + data.error); return; }
    currentRunId = data.run_id;
    setStatus(`Run started: ${currentRunId}`);
    pollInterval = setInterval(pollRun, 2000);
  } catch(e) { setStatus('Request failed: ' + e.message); }
}

async function pollRun() {
  if (!currentRunId) return;
  try {
    const res = await fetch('/api/run/' + currentRunId);
    const data = await res.json();
    if (data.status === 'running') {
      setStatus(`Running... (${currentRunId})`);
    } else if (data.status === 'complete') {
      clearInterval(pollInterval);
      setStatus('Complete!');
      showResult(data);
    } else if (data.status === 'error') {
      clearInterval(pollInterval);
      setStatus('Error: ' + data.error);
    }
  } catch(e) {}
}

function showResult(data) {
  const r = data.result;
  if (!r) return;
  const matchStr = r.array_match_pct != null ? `Array match ${r.array_match_pct.toFixed(0)}%` : 'Match unverified';
  let html = `<div class="card">
    <strong>${matchStr}</strong> · ${r.solved_count}/${r.camera_count} cameras solved<br>
    Target: ${r.anchor_ire ? r.anchor_ire.toFixed(1) : '?'} IRE &nbsp;
    Kelvin: ${r.shared_kelvin || '?'}K<br>
    WB spread: WC ${r.wc_spread_after ? r.wc_spread_after.toFixed(4) : '?'} GM ${r.gm_spread_after ? r.gm_spread_after.toFixed(4) : '?'}<br><br>`;
  if (data.report_url) {
    html += `<a href="${data.report_url}" target="_blank">Open Contact Sheet →</a><br><br>`;
  }
  r.cameras.forEach(c => {
    html += `<div style="margin:4px 0;font-size:13px">
      <strong>${c.camera_label}</strong> ${c.match_pct != null ? c.match_pct.toFixed(0) + '%' : c.status}
      · ${c.hero_ire ? c.hero_ire.toFixed(1) : '?'} IRE
      · ExpAdj ${c.exposure_adjust !== null ? (c.exposure_adjust >= 0 ? '+' : '') + c.exposure_adjust.toFixed(3) : '?'}
      · ${c.kelvin || '?'}K T${c.tint !== null ? (c.tint >= 0 ? '+' : '') + c.tint.toFixed(1) : '?'}
    </div>`;
  });
  html += '</div>';
  document.getElementById('result').innerHTML = html;
}

function setStatus(msg) { document.getElementById('status').textContent = msg; }
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="R3DMatch v2 web server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5001)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    print(f"R3DMatch v2 starting at http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
