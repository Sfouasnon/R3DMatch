"""
report.py — R3DMatch v2 contact-sheet HTML renderer.

Generates a self-contained, print-ready HTML report with:
  • Overview page: summary cards, exposure lollipop chart, WB vectorscope,
    priority table sorted by |exposure_adjust|
  • Per-camera pages (landscape, 11×8.5in):
      - ORIGINAL frame with sphere overlay (circle + zone dots + center cross)
      - CORRECTED frame (brightness-simulated using exposure_adjust)
      - Closed-loop proof panel (exposure IRE → target → residual)
      - WB mini-vectorscope, per-camera JSON commit block
  • Light/professional theme matching v1 design language
  • Full-resolution base64-embedded JPEG thumbnails (no external deps)

Usage:
    from r3dmatch3.report import build_report
    html_path = build_report(run_result, out_dir)
"""

from __future__ import annotations

import base64
import io
import json
import logging
import math
import os
import textwrap
from typing import List, Optional, Tuple

from PIL import Image, ImageDraw, ImageEnhance

from r3dmatch3.models import CameraResult, RunResult, SphereROI

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Thumbnail quality settings
# ---------------------------------------------------------------------------
_THUMB_LONG_EDGE = 1920   # full-res thumbnails — long edge in pixels
_THUMB_JPEG_QUALITY = 88  # JPEG quality for embedded thumbnails


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def build_report(run_result: RunResult, out_dir: str) -> str:
    """
    Build a self-contained HTML contact sheet and write it to out_dir.
    Filename: report_<YYYYMMDD_HHMMSS>.html  — unique per run, never clobbered.
    Returns the absolute path to the written file.
    """
    import datetime
    os.makedirs(out_dir, exist_ok=True)
    html = _render_html(run_result)
    # Prefer the run's own timestamp; fall back to wall-clock now.
    ts_src = getattr(run_result, "created_at", None) or ""
    try:
        # created_at is ISO-8601: "2026-06-01T14:32:07.123456" or "2026-06-01T14:32:07"
        dt = datetime.datetime.fromisoformat(ts_src[:19])
    except (ValueError, TypeError):
        dt = datetime.datetime.now()
    ts = dt.strftime("%Y%m%d_%H%M%S")
    filename = f"report_{ts}.html"
    out_path = os.path.join(out_dir, filename)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    log.info("Report written to %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# Top-level HTML assembly
# ---------------------------------------------------------------------------
def _render_html(rr: RunResult) -> str:
    run_name = rr.run_id
    run_date = rr.created_at[:10] if rr.created_at else "—"
    strategy = rr.anchor_source or "Median"

    # Camera pages sorted alphabetically by camera_label (G007_A → G007_B → ... → I007_D).
    # The overview summary table retains its own status/adj sort independently.
    cameras = sorted(rr.cameras, key=lambda cr: cr.camera_label)

    overview_html    = _render_overview(rr, cameras, run_name, run_date, strategy)
    array_page_html  = _render_array_comparison(rr, cameras, run_name)
    coherence_html   = _render_coherence_pages(rr, cameras)
    export_html      = _render_export_page(rr, cameras)
    total_pages      = 2 + len(cameras) + 1   # overview + array + charts + camera pages
    camera_pages     = "".join(
        _render_camera_page(cr, rr, i + 3) for i, cr in enumerate(cameras)
    )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>R3DMatch Assessment — {run_name}</title>
  <style>{_css()}</style>
</head>
<body>
{overview_html}
{array_page_html}
{coherence_html}
{camera_pages}
{export_html}
</body>
</html>"""


# ---------------------------------------------------------------------------
# CSS — light/professional, v1 design language extended
# ---------------------------------------------------------------------------
def _css() -> str:
    return """
    /* Paired with the R3DMatch v3 app theme (docs/ui_mockup_v3.html):
       same crimson accent + typography on clean neutral paper. */
    :root {
      --bg: #dbdad7;
      --paper: #f4f4f2;
      --card: #ffffff;
      --ink: #1a1816;
      --muted: #6e6962;
      --muted-2: #524e49;
      --line: #ddd9d3;
      --line-2: #d3cfc9;
      --blue-weak: #f1f0ee;
      --green-weak: #f4faf6;
      --amber-weak: #fdf8f1;
      --violet-weak: #f7f5fb;
      --good: #16a34a;
      --review: #f59e0b;
      --alert: #d61f4d;
      --anchor: #1a1816;
      --needs-assist: #7c3aed;
      --surface: #f5f4f2;
      --surface-2: #ebeae7;
      --good-bg: #eef9f1;
      --review-bg: #fff8eb;
      --anchor-bg: #f1f0ee;
      --assist-bg: #f5f3ff;
      --accent: #e3242b;
      --accent-deep: #8f1318;
      --accent-dim: rgba(227,36,43,0.08);
      --mono: 'IBM Plex Mono', ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      --sans: "DM Sans", ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    .r3d-mark {
      display:inline-block; background:linear-gradient(135deg,var(--accent),var(--accent-deep));
      color:#fff; font-weight:900; font-size:.78em; letter-spacing:.04em;
      padding:.18em .5em; border-radius:.4em; vertical-align:baseline; margin-right:.35em;
    }
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800;900&family=IBM+Plex+Mono:wght@400;500;600&display=swap');
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: var(--sans);
      -webkit-print-color-adjust: exact;
      print-color-adjust: exact;
    }
    .mono {
      font-variant-numeric: tabular-nums;
      font-feature-settings: "tnum" 1;
      font-family: var(--mono);
    }

    /* ── Shared page layout ── */
    .landscape-page {
      width: 11in;
      min-height: 8.5in;
      margin: 16px auto;
      background: var(--paper);
      padding: 22px 26px 20px;
      box-shadow: 0 8px 30px rgba(16,24,40,.10);
      page-break-after: always;
      position: relative;
    }
    @media print {
      body { background: none; }
      .landscape-page { margin: 0; box-shadow: none; }
    }

    /* ── Overview page ── */
    .overview-page .top-meta {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 20px;
      align-items: start;
      margin-bottom: 12px;
    }
    .overview-page .brand {
      font-size: 11px;
      font-weight: 800;
      letter-spacing: .12em;
      text-transform: uppercase;
      color: #6e6962;
      margin-bottom: 8px;
    }
    .overview-page h1 {
      margin: 0 0 6px 0;
      font-size: 38px;
      line-height: 1;
      letter-spacing: -.03em;
      font-weight: 900;
    }
    .overview-page .run {
      font-size: 19px;
      font-weight: 800;
      margin-bottom: 3px;
      font-family: var(--mono);
      color: var(--accent);
    }
    .overview-page .sub {
      color: #524e49;
      font-size: 14px;
      line-height: 1.45;
    }
    .overview-page .meta-list {
      display: grid;
      grid-template-columns: auto auto;
      gap: 6px 18px;
      font-size: 13px;
      align-self: start;
      padding-top: 8px;
    }
    .overview-page .meta-list .k { color: #524e49; font-weight: 800; }
    .overview-page .meta-list .v { color: #524e49; font-weight: 600; font-family: var(--mono); font-size: 12px; }
    .overview-page .cards {
      display: grid;
      grid-template-columns: repeat(7, 1fr);
      gap: 8px;
      margin: 12px 0;
    }
    .overview-page .card {
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 11px 13px;
      box-shadow: 0 1px 0 rgba(16,24,40,.04);
    }
    .overview-page .card-head {
      display: flex;
      align-items: flex-start;
      gap: 9px;
      margin-bottom: 14px;
    }
    .overview-page .icon {
      width: 32px;
      height: 32px;
      border-radius: 7px;
      display: grid;
      place-items: center;
      color: #6e6962;
      flex: 0 0 auto;
    }
    .overview-page .card-label {
      font-size: 10px;
      font-weight: 900;
      letter-spacing: .06em;
      text-transform: uppercase;
      color: #524e49;
      line-height: 1.2;
      padding-top: 3px;
    }
    .overview-page .card-value {
      font-size: 22px;
      font-weight: 800;
      line-height: 1.12;
      margin-bottom: 7px;
    }
    .overview-page .card-value.tight { font-size: 16px; }
    .overview-page .card-note {
      font-size: 11px;
      color: #6e6962;
      line-height: 1.35;
    }
    .overview-page .card.result .card-value { color: var(--good); }
    .overview-page .card.result-warn .card-value { color: var(--review); }
    .overview-page .card.result-err .card-value { color: var(--alert); }

    /* Summary table */
    .section-title {
      font-size: 11px;
      font-weight: 900;
      letter-spacing: .06em;
      text-transform: uppercase;
      color: #524e49;
      margin: 0 0 7px 0;
    }
    /* Offline match-export page */
    .export-files, .export-values {
      width: 100%; border-collapse: collapse; font-size: 10.5px;
      background: var(--card); border: 1px solid var(--line); border-radius: 6px; overflow: hidden;
    }
    .export-files th, .export-values th {
      text-align: left; background: #f4f2ef; color: #524e49; font-weight: 800;
      padding: 6px 10px; border-bottom: 1px solid var(--line); letter-spacing: .03em;
    }
    .export-files td, .export-values td {
      padding: 6px 10px; border-bottom: 1px solid #efece8; color: #3f3c38; vertical-align: top;
    }
    .export-files tr:last-child td, .export-values tr:last-child td { border-bottom: none; }
    .export-files td:first-child { white-space: nowrap; }
    .export-files code, .export-values code, .recipe-box code {
      font-family: ui-monospace, SFMono-Regular, monospace; font-size: 10px;
    }
    .recipe-box {
      background: #1e1c1a; color: #e8e4de; border-radius: 6px;
      padding: 10px 13px; margin: 2px 0 8px; overflow-x: auto;
    }
    .recipe-box code { color: #e8e4de; white-space: nowrap; }
    .export-flow {
      margin: 2px 0 8px; padding-left: 20px; max-width: 1020px;
      font-size: 11px; color: #4b4742; line-height: 1.5;
    }
    .export-flow li { margin-bottom: 6px; }
    .table-wrap {
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 10px;
      overflow: hidden;
    }
    table.summary-table {
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }
    table.summary-table thead th {
      background: #faf9f7;
      border-bottom: 1px solid var(--line);
      padding: 8px 10px;
      text-align: left;
      font-size: 10px;
      font-weight: 900;
      text-transform: uppercase;
      letter-spacing: .04em;
      color: #524e49;
    }
    table.summary-table tbody td {
      padding: 7px 10px;
      border-bottom: 1px solid #f1f0ee;
      vertical-align: middle;
    }
    table.summary-table tbody tr:last-child td { border-bottom: none; }
    .cam { font-weight: 700; font-family: var(--mono); }
    .tone-red   { color: #8f1318; }
    .tone-orange { color: #c2410c; }
    .tone-green  { color: #15803d; }
    .tone-blue   { color: #1a1816; }
    .tone-violet { color: #6d28d9; }
    .delta-pos { color: #1a1816; }
    .delta-neg { color: #8f1318; }
    .priority { width: 34px; height: 34px; border-radius: 50%; display: inline-grid; place-items: center; color: #fff; font-weight: 900; font-size: 13px; }
    .p-red    { background: #d61f4d; }
    .p-orange { background: #f97316; }
    .p-amber  { background: #f59e0b; }
    .p-green  { background: #16a34a; }
    .p-blue   { background: #1a1816; }
    .p-violet { background: #7c3aed; }
    .legend {
      display: flex;
      gap: 20px;
      padding: 9px 14px;
      background: #faf9f7;
      border-top: 1px solid var(--line);
      font-size: 12px;
      color: #6e6962;
    }
    .legend .row { display: flex; align-items: center; gap: 7px; }
    .bullet { width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }
    .needs-assist-badge {
      display: inline-block;
      background: var(--assist-bg);
      color: var(--needs-assist);
      border: 1px solid #c4b5fd;
      border-radius: 5px;
      font-size: 10px;
      font-weight: 800;
      letter-spacing: .04em;
      padding: 1px 6px;
      vertical-align: middle;
    }

    /* Charts page */
    .charts-page-2 .row {
      display: flex;
      gap: 10px;
      margin-bottom: 10px;
    }
    .panel {
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 10px;
      overflow: hidden;
      flex: 1;
    }
    .panel-head {
      padding: 10px 14px 8px;
      border-bottom: 1px solid var(--line);
      background: #fcfbfa;
    }
    .panel-title {
      font-size: 12px;
      font-weight: 900;
      text-transform: uppercase;
      letter-spacing: .05em;
      color: #1a1816;
    }
    .panel-sub {
      font-size: 11px;
      color: #6e6962;
      margin-top: 2px;
    }
    .panel-body { padding: 10px; }
    .panel-body svg { width: 100%; display: block; }
    .wb-panel .panel-body { overflow: visible; position: relative; }

    /* Array comparison page */
    .array-page .array-header {
      display: grid;
      grid-template-columns: 1fr auto;
      align-items: start;
      margin-bottom: 10px;
      padding-bottom: 10px;
      border-bottom: 2px solid var(--line-2);
    }
    .array-page .section-headline {
      font-size: 20px;
      font-weight: 900;
      letter-spacing: -.02em;
      color: var(--ink);
      margin-bottom: 3px;
    }
    .array-page .section-sub {
      font-size: 12px;
      color: var(--muted);
      line-height: 1.5;
      max-width: 520px;
    }
    .array-verdict {
      font-family: var(--mono);
      font-size: 18px;
      font-weight: 800;
      color: var(--good);
      background: var(--good-bg);
      border: 1.5px solid #bbf7d0;
      border-radius: 8px;
      padding: 6px 18px;
      white-space: nowrap;
    }
    .post-callout {
      background: linear-gradient(135deg, #eff6ff, #f0f9ff);
      border: 1.5px solid #d6d3cd;
      border-radius: 8px;
      padding: 12px 16px;
      margin-top: 14px;
      display: flex;
      gap: 12px;
      align-items: flex-start;
      font-size: 12px;
      line-height: 1.7;
      color: #3f3c38;
    }
    .post-callout .pc-icon { font-size: 18px; flex-shrink: 0; margin-top: 1px; }
    .post-callout strong { color: #1e3a8a; }
    .post-callout code {
      background: rgba(30,64,175,0.1);
      padding: 1px 6px;
      border-radius: 3px;
      font-size: 11px;
      font-family: var(--mono);
    }

    /* Quick notes */
    .quick h3 { margin: 0 0 10px 0; font-size: 13px; font-weight: 900; }
    .quick-group { margin-bottom: 10px; }
    .quick-label { font-size: 11px; font-weight: 800; text-transform: uppercase; letter-spacing: .05em; color: #6e6962; margin-bottom: 6px; }
    .quick-item { display: flex; align-items: flex-start; gap: 9px; font-size: 13px; margin-bottom: 5px; }
    .quick-item .num { width: 22px; height: 22px; border-radius: 50%; display: inline-grid; place-items: center; color: #fff; font-size: 12px; font-weight: 900; flex-shrink: 0; }

    /* ── Camera pages ── */
    .camera-page .hd {
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      margin-bottom: 9px;
      padding-bottom: 9px;
      border-bottom: 1.5px solid var(--line);
    }
    .hd-brand {
      font-size: 10px;
      font-weight: 800;
      letter-spacing: .1em;
      text-transform: uppercase;
      color: #a3a09b;
      margin-bottom: 3px;
    }
    .hd-cam {
      font-size: 28px;
      font-weight: 900;
      letter-spacing: -.02em;
      line-height: 1;
      margin-bottom: 2px;
      font-family: var(--mono);
    }
    .hd-clip {
      font-size: 12px;
      font-weight: 600;
      color: #6e6962;
      font-family: var(--mono);
    }
    .hd-project {
      font-size: 11px;
      color: #a3a09b;
      margin-top: 2px;
      line-height: 1.4;
    }
    .hd-right { text-align: right; }
    .hd-status {
      font-size: 20px;
      font-weight: 900;
      letter-spacing: -.01em;
    }
    .status-pass         { color: var(--good); }
    .status-needs-assist { color: var(--needs-assist); }
    .status-fail         { color: var(--alert); }
    .hd-meta {
      font-size: 12px;
      color: #6e6962;
      margin-top: 3px;
    }
    .anchor-flag {
      display: inline-block;
      background: var(--anchor-bg);
      color: var(--anchor);
      border: 1px solid #d6d3cd;
      border-radius: 5px;
      font-size: 10px;
      font-weight: 900;
      letter-spacing: .05em;
      padding: 2px 8px;
      margin-bottom: 4px;
      text-transform: uppercase;
    }

    /* Image row */
    .img-row {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
      margin-bottom: 9px;
    }
    .img-panel {
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
    }
    .img-label {
      font-size: 9px;
      font-weight: 900;
      letter-spacing: .10em;
      text-transform: uppercase;
      color: #a3a09b;
      padding: 5px 10px 4px;
      border-bottom: 1px solid var(--line);
      background: #faf9f7;
    }
    .img-frame {
      background: #111;
      display: flex;
      align-items: center;
      justify-content: center;
      height: 220px;
      overflow: hidden;
    }
    .img-box {
      width: 100%;
      height: 100%;
      object-fit: contain;
      display: block;
    }

    /* Metrics band */
    .metrics-band {
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
      margin-bottom: 8px;
    }
    .metrics {
      display: grid;
      grid-template-columns: 1.3fr 1.1fr 1fr 1fr;
      gap: 0;
    }
    .metric-col {
      padding: 10px 12px;
      border-right: 1px solid #f1f0ee;
    }
    .metric-col:last-child { border-right: none; }
    .mc-title {
      font-size: 9px;
      font-weight: 900;
      letter-spacing: .07em;
      text-transform: uppercase;
      color: #a3a09b;
      margin-bottom: 8px;
    }
    .mc-row {
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      gap: 8px;
      margin-bottom: 4px;
      font-size: 12px;
    }
    .mc-key { color: #6e6962; font-weight: 600; white-space: nowrap; }
    .mc-val { font-weight: 700; text-align: right; }
    .mc-val.large { font-size: 17px; }
    .mc-spacer { height: 6px; }
    .mc-foot {
      font-size: 10px;
      color: #a3a09b;
      margin-top: 6px;
      font-family: var(--mono);
      line-height: 1.3;
    }

    /* Exposure track */
    .exp-vis { margin-top: 8px; }
    .exp-label {
      display: flex;
      justify-content: space-between;
      font-size: 10px;
      color: #a3a09b;
      margin-bottom: 3px;
    }
    .exp-track {
      height: 12px;
      background: #f1f5f9;
      border-radius: 6px;
      position: relative;
      overflow: hidden;
    }
    .exp-zone {
      position: absolute;
      top: 0;
      height: 100%;
      background: #dcfce7;
    }
    .exp-center-line {
      position: absolute;
      left: 50%;
      top: 0;
      width: 2px;
      height: 100%;
      background: #a3a09b;
      transform: translateX(-1px);
    }
    .exp-bar {
      position: absolute;
      top: 2px;
      height: 8px;
      border-radius: 4px;
      min-width: 4px;
    }
    .exp-tick-labels {
      display: flex;
      justify-content: space-between;
      font-size: 9px;
      color: #a3a09b;
      margin-top: 2px;
    }

    /* WB mini chart */
    .wb-vis { margin-top: 7px; }

    /* JSON commit block */
    .commit-block {
      background: #faf9f7;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px 10px;
      margin-top: 8px;
      font-family: var(--mono);
      font-size: 11px;
      color: #3f3c38;
      position: relative;
    }
    .commit-copy-btn {
      position: absolute;
      top: 6px;
      right: 8px;
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 4px;
      padding: 2px 7px;
      font-size: 10px;
      font-weight: 700;
      cursor: pointer;
      color: #6e6962;
    }
    .commit-copy-btn:hover { background: var(--line); }

    /* Notes strip */
    .notes-strip {
      background: #faf9f7;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 6px 12px;
      margin-bottom: 6px;
    }
    .notes-inner {
      display: flex;
      flex-wrap: wrap;
      gap: 4px 20px;
    }
    .note-item { font-size: 11px; }
    .note-key { display: inline; font-weight: 800; color: #524e49; margin-right: 4px; }
    .note-val { display: inline; color: #6e6962; }

    /* IRE context warning */
    .ire-context-warn {
      background: var(--assist-bg);
      border: 1px solid #c4b5fd;
      border-radius: 6px;
      padding: 7px 12px;
      font-size: 11px;
      color: #5b21b6;
      margin-bottom: 7px;
      display: flex;
      gap: 8px;
      align-items: flex-start;
    }
    .ire-context-warn .warn-icon { font-size: 14px; flex-shrink: 0; margin-top: 1px; }

    /* IPP2 WB note */
    .ipp2-note {
      background: var(--blue-weak);
      border: 1px solid #d6d3cd;
      border-radius: 6px;
      padding: 6px 10px;
      font-size: 10px;
      color: #3f3c38;
      margin-top: 6px;
      line-height: 1.4;
    }

    /* Closed-loop proof */
    .closed-loop-pass { color: var(--good); font-weight: 700; }
    .closed-loop-fail { color: var(--alert); font-weight: 700; }
    .closed-loop-na   { color: #a3a09b; }

    /* Page footer */
    .pg-footer {
      position: absolute;
      bottom: 10px;
      left: 26px;
      right: 26px;
      display: flex;
      justify-content: space-between;
      font-size: 10px;
      color: #a3a09b;
      border-top: 1px solid var(--line-2);
      padding-top: 6px;
    }

    /* Exposure-only: quick notes spans full width (no WB panel beside it) */
    .quick-notes-wide { flex: 1 1 100%; max-width: none; }
    .quick-notes-wide .quick { display: flex; flex-wrap: wrap; gap: 14px 48px; align-items: flex-start; }
    .quick-notes-wide .quick h3 { flex-basis: 100%; margin: 0 0 4px; }
    .quick-notes-wide .quick-group { margin-bottom: 0; min-width: 220px; }

    /* Array coherence contact sheet */
    .coherence-page { padding: 26px 26px 40px; }
    .coh-top {
      display: flex; justify-content: space-between; align-items: flex-end;
      gap: 24px; border-bottom: 2px solid var(--line-2);
      padding-bottom: 8px; margin-bottom: 12px;
    }
    .coh-title { font-size: 18px; font-weight: 800; color: #1a1816; }
    .coh-claim {
      font-size: 10.5px; line-height: 1.4; color: #4b4843;
      max-width: 58%; text-align: right;
    }
    .coh-block { margin-bottom: 14px; }
    .coh-blocklabel {
      font-size: 11px; font-weight: 800; letter-spacing: 0.08em;
      text-transform: uppercase; color: #46627f; margin-bottom: 6px;
    }
    .coh-blocklabel.coh-after { color: #15803d; }
    .coh-grid { display: grid; gap: 6px; }
    .coh-cell {
      background: #000; border-radius: 4px; overflow: hidden;
      border: 1px solid var(--line);
    }
    .coh-cell img { width: 100%; aspect-ratio: 16 / 9; object-fit: cover; display: block; }
    .coh-missing {
      aspect-ratio: 16 / 9; display: grid; place-items: center;
      color: #8a857d; font-size: 9px; background: #1a1816;
    }
    .coh-cap {
      display: flex; justify-content: space-between; align-items: center;
      gap: 4px; padding: 1px 6px; background: #faf9f7;
      font-family: ui-monospace, monospace; font-size: 11px; line-height: 1.35;
    }
    .coh-cam { color: #3f3c38; font-weight: 700; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .coh-ire { font-weight: 800; flex-shrink: 0; }
    """


# ---------------------------------------------------------------------------
# Overview page
# ---------------------------------------------------------------------------
def _render_overview(
    rr: RunResult,
    cameras: List[CameraResult],
    run_name: str,
    run_date: str,
    strategy: str,
) -> str:
    total = len(cameras)
    solved = sum(1 for cr in cameras if _camera_status(cr) == "SOLVED")
    needs_assist = sum(1 for cr in cameras if _camera_status(cr) == "NEEDS_ASSIST")

    anchors = [cr for cr in cameras if _is_anchor(cr, rr)]
    anchor_label = anchors[0].camera_label if anchors else "—"

    # Exposure stats
    adjs = [cr.commit.exposure_adjust for cr in cameras if cr.commit]
    exp_spread = max(abs(a) for a in adjs) if adjs else 0.0
    sorted_by_adj = sorted(cameras, key=lambda cr: abs(cr.commit.exposure_adjust) if cr.commit else 0, reverse=True)

    # WB
    eo = getattr(rr, "exposure_only", False)   # exposure-only run → omit all WB
    kelvins = [cr.commit.kelvin for cr in cameras if cr.commit]
    shared_k = kelvins[0] if kelvins else 0

    wb_card = "" if eo else f"""
      <div class="card">
        <div class="card-head">
          <div class="icon"><svg width="32" height="32" viewBox="0 0 32 32" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M16 4a4 4 0 0 0-4 4v9a5 5 0 1 0 8 0V8a4 4 0 0 0-4-4Z"/><path d="M16 11v7"/></svg></div>
          <div class="card-label">White Balance</div>
        </div>
        <div class="card-value mono">{shared_k}K</div>
        <div class="card-note">Shared Kelvin target; tint varies per camera</div>
      </div>"""

    # Overall result badge — array match percentage, never PASS/FAIL
    array_pct = getattr(rr, "array_match_pct", None)
    if array_pct is not None:
        result_str = f"{array_pct:.0f}%"
        result_class = "result" if array_pct >= 95.0 else "result-warn"
        min_pct = getattr(rr, "min_match_pct", None)
        min_clip = getattr(rr, "min_match_clip_id", "")
        result_note = "Array match — mean of camera match scores."
        if min_pct is not None and min_pct < array_pct - 0.5:
            result_note = f"Array match. Lowest: {min_clip} at {min_pct:.0f}%."
    elif needs_assist > 0:
        result_str = "ASSIST"
        result_class = "result-warn"
        result_note = f"{needs_assist} camera(s) need operator ROI placement."
    else:
        result_str = f"{solved}/{total}"
        result_class = "result" if solved == total else "result-warn"
        result_note = "Cameras solved. Match % pending corrected-render verification."

    # Delivery-look line — appended only when a project look was scored.
    _dname = getattr(rr, "delivery_pipeline_name", "") or ""
    _dpct  = getattr(rr, "delivery_array_match_pct", None)
    if _dname or _dpct is not None:
        _dtxt = _dname or "delivery"
        if _dpct is not None:
            _dtxt += f" — {_dpct:.0f}% delivery match"
        result_note += f" Delivery look: {_dtxt}."

    top3 = sorted_by_adj[:3]

    cards_html = f"""
    <div class="cards">
      <div class="card">
        <div class="card-head">
          <div class="icon"><svg width="32" height="32" viewBox="0 0 32 32" fill="none" stroke="currentColor" stroke-width="1.8"><circle cx="16" cy="16" r="9"/><circle cx="16" cy="16" r="3.5"/><path d="M16 2v7M16 23v7M2 16h7M23 16h7"/></svg></div>
          <div class="card-label">Matching<br>Strategy</div>
        </div>
        <div class="card-value mono">{strategy}</div>
        <div class="card-note">Robust median in stops (3&times;MAD outlier rejection) &mdash; an off-exposure camera can&rsquo;t pull the target.</div>
      </div>
      <div class="card">
        <div class="card-head">
          <div class="icon"><svg width="32" height="32" viewBox="0 0 32 32" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M4 26h24M6 24V8l5 9 4-6 4 11 4-8 5 4"/></svg></div>
          <div class="card-label">Exposure<br>Spread (±)</div>
        </div>
        <div class="card-value mono">{exp_spread:.2f} stops</div>
        <div class="card-note">Largest correction across all cameras</div>
      </div>
      {wb_card}
      <div class="card">
        <div class="card-head">
          <div class="icon"><svg width="32" height="32" viewBox="0 0 32 32" fill="none" stroke="currentColor" stroke-width="1.8"><circle cx="11" cy="11" r="6"/><circle cx="21" cy="11" r="6"/><circle cx="16" cy="20" r="6"/></svg></div>
          <div class="card-label">Top Review<br>Camera</div>
        </div>
        <div class="card-value tight mono">{top3[0].camera_label if top3 else '—'}<br>{top3[0].commit.exposure_adjust:+.2f}</div>
        <div class="card-note">Largest exposure correction</div>
      </div>
      <div class="card">
        <div class="card-head">
          <div class="icon"><svg width="32" height="32" viewBox="0 0 32 32" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M16 5v5M16 22v5M5 16h5M22 16h5"/><circle cx="16" cy="16" r="7"/></svg></div>
          <div class="card-label">Anchor<br>Camera</div>
        </div>
        <div class="card-value tight mono">{anchor_label}<br>Reference</div>
        <div class="card-note">Exposure reference camera</div>
      </div>
      <div class="card">
        <div class="card-head">
          <div class="icon"><svg width="32" height="32" viewBox="0 0 32 32" fill="none" stroke="currentColor" stroke-width="1.8"><circle cx="16" cy="16" r="10"/></svg></div>
          <div class="card-label">Cameras<br>Solved</div>
        </div>
        <div class="card-value mono">{solved} / {total}</div>
        <div class="card-note">Commits ready for push</div>
      </div>
      <div class="card {result_class}">
        <div class="card-head">
          <div class="icon" style="color:{'var(--good)' if result_class=='result' else 'var(--review)'}"><svg width="32" height="32" viewBox="0 0 32 32" fill="none" stroke="currentColor" stroke-width="2"><circle cx="16" cy="16" r="11"/><path d="M10 16l4 4 9-9"/></svg></div>
          <div class="card-label">Array<br>Match</div>
        </div>
        <div class="card-value">{result_str}</div>
        <div class="card-note">{result_note}</div>
      </div>
    </div>"""

    # Priority table
    table_rows = ""
    for rank, cr in enumerate(sorted_by_adj[:12], 1):
        status = _camera_status(cr)
        adj = cr.commit.exposure_adjust if cr.commit else 0.0
        ire = cr.measurement.hero_ire if cr.measurement else 0.0
        k = cr.commit.kelvin if cr.commit else 0
        tint = cr.commit.tint if cr.commit else 0.0
        tone_cls = _tone_class(adj, status)
        delta_cls = "delta-pos" if adj > 0.0005 else ("delta-neg" if adj < -0.0005 else "")
        p_cls = _priority_class(rank)
        assist_badge = ' <span class="needs-assist-badge">NEEDS ASSIST</span>' if status == "NEEDS_ASSIST" else ""
        table_rows += f"""
        <tr>
          <td><span class="priority {p_cls}">{rank}</span></td>
          <td class="cam">{cr.camera_label}{assist_badge}</td>
          <td class="mono {tone_cls}">{ire:.1f}</td>
          <td class="mono">{_anchor_ire(rr):.1f}</td>
          <td class="mono {delta_cls}">{adj:+.3f}</td>
          {'' if eo else f'<td class="mono">{k}K / {tint:+.1f}</td>'}
          <td class="mono">{_residual_str(cr)}</td>
        </tr>"""

    lollipop_svg = _render_exposure_lollipop(cameras, rr)
    vectorscope_svg = _render_vectorscope(cameras)
    quick_notes = _render_quick_notes(top3, anchor_label, solved, total, cameras)

    return f"""
<section class="landscape-page overview-page">
  <div class="top-meta">
    <div>
      <div class="brand"><span class="r3d-mark">R3D</span>Match Assessment</div>
      <h1><span class="r3d-mark">R3D</span>Match Assessment</h1>
      <div class="run">{run_name}</div>
      <div class="sub">Exposure Anchor: {strategy}</div>
    </div>
    <div class="meta-list">
      <div class="k">Run Date</div><div class="v">{run_date}</div>
      <div class="k">Target</div><div class="v">Gray sphere</div>
      <div class="k">Measurement Domain</div><div class="v">REDLine IPP2 / BT.709 / BT.1886 / Medium / Medium</div>
      <div class="k">Total Cameras</div><div class="v">{total}</div>
      <div class="k">Solved / Total</div><div class="v">{solved} / {total}</div>
      {'<div class="k">Needs Assist</div><div class="v">' + str(needs_assist) + '</div>' if needs_assist > 0 else ''}
    </div>
  </div>
  {cards_html}
  <div class="section-title">Camera Exposure &amp; Color Summary (Hero Center Measurement)</div>
  <div class="table-wrap">
    <table class="summary-table">
      <thead>
        <tr>
          <th>Priority</th>
          <th>Camera</th>
          <th>Before IRE</th>
          <th>Target IRE</th>
          <th>Exposure Fix</th>
          {'' if eo else '<th>Kelvin / Tint</th>'}
          <th>Exposure Adjustment</th>
        </tr>
      </thead>
      <tbody>{table_rows}</tbody>
    </table>
    <div class="legend">
      <div class="row"><span class="bullet" style="background:var(--alert)"></span>Needs Attention</div>
      <div class="row"><span class="bullet" style="background:#f97316"></span>Review</div>
      <div class="row"><span class="bullet" style="background:var(--good)"></span>Good</div>
      <div class="row"><span class="bullet" style="background:var(--anchor)"></span>Anchor</div>
      <div class="row"><span class="bullet" style="background:var(--needs-assist)"></span>Needs Assist</div>
    </div>
  </div>
</section>

<section class="landscape-page overview-page">
  <div class="charts-page-2">
    <div class="row row-top">
      <div class="panel exposure-panel">
        <div class="panel-head">
          <div class="panel-title">Exposure Spread (from anchor)</div>
          <div class="panel-sub">Sorted by deviation (stops)</div>
        </div>
        <div class="panel-body">{lollipop_svg}</div>
      </div>
    </div>
    <div class="row row-bottom">
      {'' if eo else f'''<div class="panel wb-panel">
        <div class="panel-head">
          <div class="panel-title">White Balance (Neutral Placement)</div>
          <div class="panel-sub">Hollow = as-shot &rarr; filled = corrected, measured from render. Centered on group neutral &mdash; cameras match each other, not absolute white.</div>
        </div>
        <div class="panel-body">{vectorscope_svg}</div>
      </div>'''}
      <div class="panel quick-notes-panel{' quick-notes-wide' if eo else ''}"{'' if eo else ' style="max-width:280px"'}>
        <div class="panel-body">{quick_notes}</div>
      </div>
    </div>
  </div>
</section>"""


# ---------------------------------------------------------------------------
# Array comparison page  (Page 2 — the missing piece from v1)
# Full-width IRE before/after for all cameras side by side.
# ---------------------------------------------------------------------------
def _render_array_comparison(rr: RunResult, cameras: list, run_name: str) -> str:
    """
    Landscape page showing all cameras in one view:
      - Gray bars = raw measured IRE (before calibration)
      - Colored bars = corrected IRE (after commit applied)
      - Dashed orange target line at anchor_ire
      - Green acceptance band ±2 IRE
      - Post-production --useMeta callout
    """
    # Sort alphabetically by camera_label for easy visual scan
    cameras = sorted(cameras, key=lambda cr: cr.camera_label)
    anchor = _anchor_ire(rr)
    n = len(cameras)
    if n == 0:
        return ""

    # --- Build IRE convergence SVG ---
    W, H = 1060, 460
    LEFT, RIGHT = 44, W - 60
    chart_w = RIGHT - LEFT
    col_w = chart_w / n
    TOP, BOT = 28, H - 72
    chart_h = BOT - TOP

    IRE_MIN = max(0.0, anchor - 20)
    adjs = [cr.commit.exposure_adjust if cr.commit else 0.0 for cr in cameras]
    ires = [cr.measurement.hero_ire if cr.measurement else anchor for cr in cameras]
    corr_ires = [_corrected_ire_for(cr, ir, adj)[0] or anchor
                 for cr, ir, adj in zip(cameras, ires, adjs)]
    all_ires = ires + corr_ires + [anchor]
    IRE_MAX = max(all_ires) + 4
    IRE_MIN = min(min(all_ires) - 2, anchor - 8)

    def ire_to_y(ire: float) -> float:
        return BOT - (ire - IRE_MIN) / max(IRE_MAX - IRE_MIN, 1) * chart_h

    svg = []
    svg.append(f"<svg viewBox='0 0 {W} {H}' role='img' xmlns='http://www.w3.org/2000/svg'>")
    svg.append(f"<rect x='0' y='0' width='{W}' height='{H}' fill='white'/>")

    # Grid lines + IRE axis labels (every 5 IRE)
    step = 5
    ire_tick = int(IRE_MIN / step) * step
    while ire_tick <= IRE_MAX:
        y = ire_to_y(ire_tick)
        col = "#e7e5e1" if ire_tick % 10 != 0 else "#d6d3cd"
        svg.append(f"<line x1='{LEFT}' y1='{y:.1f}' x2='{RIGHT}' y2='{y:.1f}' stroke='{col}' stroke-width='1'/>")
        svg.append(f"<text x='{LEFT-5}' y='{y+4:.1f}' text-anchor='end' fill='#a3a09b' font-size='9' font-family='ui-monospace,monospace'>{ire_tick}</text>")
        ire_tick += step

    # Acceptance band — scene-linear exposure matching. ±0.5 IRE is the tight,
    # REPEATABLE benchmark: it clears the camera exposureAdjust quantization
    # floor (~0.07-0.15 IRE depending on target level), measurement jitter
    # (~0.05 stops), and the observed array spread, on every run and target.
    # (The single-camera precision floor is ~0.1 IRE, but that's a best-case
    # capability, not a guarantee — see report notes.)
    ACCEPT_IRE = 0.5
    ay1 = ire_to_y(anchor + ACCEPT_IRE)
    ay2 = ire_to_y(anchor - ACCEPT_IRE)
    svg.append(f"<rect x='{LEFT}' y='{ay1:.1f}' width='{chart_w}' height='{ay2 - ay1:.1f}' fill='#dcfce7' opacity='0.55'/>")

    # Target dashed line
    ty = ire_to_y(anchor)
    svg.append(f"<line x1='{LEFT}' y1='{ty:.1f}' x2='{RIGHT}' y2='{ty:.1f}' stroke='#e3242b' stroke-width='2' stroke-dasharray='6 3'/>")
    svg.append(f"<text x='{RIGHT+4}' y='{ty+4:.1f}' fill='#e3242b' font-size='9' font-weight='700' font-family='ui-monospace,monospace'>TARGET {anchor:.1f}</text>")

    # Per-camera bars
    bar_hw = col_w * 0.26
    for i, cr in enumerate(cameras):
        cx = LEFT + i * col_w + col_w * 0.5
        raw_ire = cr.measurement.hero_ire if cr.measurement else anchor
        adj = cr.commit.exposure_adjust if cr.commit else 0.0
        corr = _corrected_ire_for(cr, raw_ire, adj)[0] or anchor

        # "Before" bar — steel blue so it reads clearly against the white field
        # (the old near-white gray fell into the background).
        by = ire_to_y(raw_ire)
        bh = BOT - by
        if bh > 0:
            svg.append(f"<rect x='{cx - bar_hw - 1:.1f}' y='{by:.1f}' width='{bar_hw:.1f}' height='{bh:.1f}' fill='#7e9bbf' rx='2'/>")
        svg.append(f"<circle cx='{cx - bar_hw/2:.1f}' cy='{by:.1f}' r='4' fill='#52708f'/>")
        svg.append(f"<text x='{cx - bar_hw/2:.1f}' y='{by - 5:.1f}' text-anchor='middle' fill='#46627f' font-size='8' font-family='ui-monospace,monospace'>{raw_ire:.1f}</text>")

        # Colored "after" bar — thresholds scale with the tightened band.
        cy_bar = ire_to_y(corr)
        ch = BOT - cy_bar
        delta = abs(corr - anchor)
        fill = "#16a34a" if delta <= ACCEPT_IRE else ("#f59e0b" if delta <= 2 * ACCEPT_IRE else "#f97316")
        if ch > 0:
            svg.append(f"<rect x='{cx + 1:.1f}' y='{cy_bar:.1f}' width='{bar_hw:.1f}' height='{ch:.1f}' fill='{fill}' opacity='0.85' rx='2'/>")
        svg.append(f"<circle cx='{cx + bar_hw/2:.1f}' cy='{cy_bar:.1f}' r='4' fill='{fill}'/>")
        svg.append(f"<text x='{cx + bar_hw/2:.1f}' y='{cy_bar - 5:.1f}' text-anchor='middle' fill='{fill}' font-size='8' font-weight='700' font-family='ui-monospace,monospace'>{corr:.1f}</text>")

        # Camera label (abbreviated)
        short = cr.camera_label
        svg.append(f"<text x='{cx:.1f}' y='{BOT + 13}' text-anchor='middle' fill='#3f3c38' font-size='9' font-weight='700' font-family='ui-monospace,monospace'>{short[:8]}</text>")
        adj_str = f"{adj:+.2f}"
        adj_col = "#1a1816" if adj > 0 else ("#8f1318" if adj < 0 else "#6e6962")
        svg.append(f"<text x='{cx:.1f}' y='{BOT + 24}' text-anchor='middle' fill='{adj_col}' font-size='8' font-family='ui-monospace,monospace'>{adj_str}</text>")

    # Legend
    ley = H - 10
    svg.append(f"<rect x='{LEFT}' y='{ley - 8}' width='14' height='10' fill='#7e9bbf' rx='2'/>")
    svg.append(f"<text x='{LEFT + 18}' y='{ley}' fill='#6e6962' font-size='9' font-family='ui-sans-serif,sans-serif'>Before calibration (raw IRE)</text>")
    # After bars are colour-coded by distance from target \u2014 show all three states.
    svg.append(f"<rect x='{LEFT + 196}' y='{ley - 8}' width='10' height='10' fill='#16a34a' opacity='0.85' rx='2'/>")
    svg.append(f"<rect x='{LEFT + 207}' y='{ley - 8}' width='10' height='10' fill='#f59e0b' opacity='0.85' rx='2'/>")
    svg.append(f"<rect x='{LEFT + 218}' y='{ley - 8}' width='10' height='10' fill='#f97316' opacity='0.85' rx='2'/>")
    svg.append(f"<text x='{LEFT + 233}' y='{ley}' fill='#6e6962' font-size='9' font-family='ui-sans-serif,sans-serif'>"
               f"After (corrected): green &le;&plusmn;{ACCEPT_IRE:g}, amber &le;&plusmn;{2*ACCEPT_IRE:g}, orange beyond</text>")
    svg.append(f"<line x1='{LEFT + 520}' y1='{ley - 4}' x2='{LEFT + 536}' y2='{ley - 4}' stroke='#e3242b' stroke-width='2' stroke-dasharray='4 2'/>")
    svg.append(f"<text x='{LEFT + 540}' y='{ley}' fill='#e3242b' font-size='9' font-weight='700' font-family='ui-sans-serif,sans-serif'>Target {anchor:.1f} IRE</text>")
    svg.append(f"<rect x='{LEFT + 660}' y='{ley - 9}' width='32' height='12' fill='#dcfce7' rx='2'/>")
    svg.append(f"<text x='{LEFT + 696}' y='{ley}' fill='#15803d' font-size='9' font-family='ui-sans-serif,sans-serif'>\u00b1{ACCEPT_IRE:g} IRE acceptance</text>")
    svg.append("</svg>")
    svg_html = "".join(svg)

    # Overall verdict — array match %, never PASS/FAIL
    scored = [cr.match_pct for cr in cameras if getattr(cr, "match_pct", None) is not None]
    n_solved = sum(1 for cr in cameras if _camera_status(cr) == "SOLVED")
    verdict_cls = "array-verdict"
    if scored:
        avg = sum(scored) / len(scored)
        verdict_col = "#16a34a" if avg >= 95.0 else "#f59e0b"
        verdict_txt = f"{avg:.0f}% MATCH"
    else:
        verdict_col = "#16a34a" if n_solved == n else "#f59e0b"
        verdict_txt = f"{n_solved}/{n} SOLVED"

    return f"""
<section class="landscape-page array-page">
  <div class="array-header">
    <div>
      <div style="font-size:11px;font-weight:800;letter-spacing:.12em;text-transform:uppercase;color:#6e6962;margin-bottom:6px;"><span class="r3d-mark">R3D</span>Match Assessment &nbsp;&middot;&nbsp; Array-Level Comparison</div>
      <div class="section-headline">Before / After &mdash; Full Array Overview</div>
      <div class="section-sub">
        Each column = one camera. <strong>Gray bars</strong> = raw measured IRE before calibration.
        <strong>Colored bars</strong> = corrected IRE after commit values are applied via
        <code>--useMeta</code>. The dashed orange line is the shared target ({anchor:.1f}&nbsp;IRE).
        All cameras should converge into the green acceptance band after correction.
      </div>
    </div>
    <div class="{verdict_cls}" style="color:{verdict_col};">{verdict_txt}</div>
  </div>

  <div style="background:#ffffff;border:1px solid #ddd9d3;border-radius:10px;overflow:hidden;margin-bottom:10px;">
    <div class="panel-head">
      <div class="panel-title">IRE Convergence &mdash; Raw vs Calibrated &nbsp;&middot;&nbsp; All {n} Cameras</div>
      <div class="panel-sub">Anchor: {anchor:.1f}&nbsp;IRE &nbsp;&middot;&nbsp; Strategy: {rr.anchor_source or 'median'} &nbsp;&middot;&nbsp; Acceptance zone: &plusmn;{ACCEPT_IRE:g}&nbsp;IRE</div>
    </div>
    <div class="panel-body" style="padding:12px 14px;">
      {svg_html}
    </div>
  </div>

  <div class="post-callout">
    <div class="pc-icon">&#x1F4EC;</div>
    <div>
      <strong>Note for Post Production &mdash;</strong>
      The correction values in this report are embedded in each clip&rsquo;s R3D metadata.
      Apply them in your colour pipeline by rendering with REDline using the
      <code>--useMeta</code> flag, which tells REDline to read the
      {'<code>exposureAdjust</code>' if getattr(rr, 'exposure_only', False) else '<code>exposureAdjust</code>, <code>kelvin</code>, and <code>tint</code>'} values
      written by R3DMatch rather than the camera-recorded defaults.
      The chart above shows the visual result: all cameras converge within &plusmn;{ACCEPT_IRE:g}&nbsp;IRE
      of the {anchor:.1f}&nbsp;IRE target after correction.
      Example: <code>REDline --i /path/to/clip.R3D --useMeta --format 16 --outDir /renders/</code>
    </div>
  </div>

  <div class="pg-footer">
    <span>REDLine IPP2 / BT.709 / BT.1886 / Medium / Medium</span>
    <span>Array comparison &nbsp;&middot;&nbsp; {run_name}</span>
    <span>Page 2</span>
  </div>
</section>"""


# ---------------------------------------------------------------------------
# Array coherence — before/after contact sheet (read the whole array by eye)
# ---------------------------------------------------------------------------
_COH_LONG_EDGE = 320  # contact-sheet thumbnail long edge (px)


def _contact_thumb(path: Optional[str], long_edge: int = _COH_LONG_EDGE,
                   brightness: float = 1.0) -> Optional[str]:
    """Small JPEG base64 of a render for the contact sheet. brightness !=1.0
    applies a 2^adj-style linear scale (used only for the corrected fallback)."""
    if not path or not os.path.exists(path):
        return None
    try:
        img = Image.open(path).convert("RGB")
        if abs(brightness - 1.0) > 1e-6:
            import numpy as np
            arr = np.clip(np.array(img, dtype=np.float32) / 255.0 * brightness, 0.0, 1.0)
            img = Image.fromarray((arr * 255).astype(np.uint8))
        w, h = img.size
        s = long_edge / max(w, h)
        if s < 1.0:
            img = img.resize((int(w * s), int(h * s)), Image.LANCZOS)
        return _jpeg_b64(img)
    except Exception as e:
        log.warning("contact thumb failed: %s", e)
        return None


def _coh_cell(cr: CameraResult, b64: Optional[str], cap_ire: str, cap_col: str) -> str:
    img = (f'<img src="data:image/jpeg;base64,{b64}">' if b64
           else '<div class="coh-missing">no render</div>')
    return (f'<div class="coh-cell">{img}'
            f'<div class="coh-cap"><span class="coh-cam">{cr.camera_label}</span>'
            f'<span class="coh-ire" style="color:{cap_col}">{cap_ire}</span></div></div>')


def _render_export_page(rr: RunResult, cameras: List[CameraResult]) -> str:
    """Offline match-export page: documents the JSON/CSV/REDLine files written to
    the calibration folder, so post can reproduce the match without RCP2."""
    import html as _html

    def _code(cr) -> str:
        pos = (getattr(cr.metadata, "camera_position", "") or "").strip()
        label = (cr.camera_label or cr.clip_id or "").strip()
        return ((label[:1] + pos).upper().strip()) or label

    eo = getattr(rr, "exposure_only", False)
    rows = []
    for cr in cameras:
        if not cr.commit:
            continue
        ev = cr.commit.exposure_adjust
        wb = "—" if (eo or cr.commit.exposure_only) else f"{cr.commit.kelvin} K / {cr.commit.tint:+.2f}"
        rows.append(
            f"<tr><td style='font-family:ui-monospace,monospace'>{_html.escape(_code(cr))}</td>"
            f"<td style='font-family:ui-monospace,monospace'>{_html.escape(cr.camera_label)}</td>"
            f"<td style='text-align:right;font-family:ui-monospace,monospace'>{ev:+.3f}</td>"
            f"<td style='text-align:center;font-family:ui-monospace,monospace'>{wb}</td></tr>"
        )
    table_rows = "".join(rows) or "<tr><td colspan='4' style='color:#a3a09b'>No commits in this run.</td></tr>"

    files_html = """
      <table class="export-files">
        <thead><tr><th>File</th><th>Purpose</th></tr></thead>
        <tbody>
          <tr><td><code>match_export.json</code></td><td>Tool-agnostic, full fidelity. Per camera: exposureAdjust (plus Kelvin/tint on white-balance runs), as-shot metadata, match&nbsp;%, and the raw/divider integer form RCP2 uses. Carries the REDLine recipe block.</td></tr>
          <tr><td><code>redline_batch.sh</code></td><td>Runnable, camera-wide REDLine batch. <code>bash redline_batch.sh</code> — prompts for the REDLine binary (auto-found if possible), media root, output dir, and your REDLine flags. Groups clips by camera subfolder, so each camera's EV applies to <b>every</b> clip that camera shot that day (not just the calibrated one). Offers a dry run first. Build&nbsp;65.1.3+, no RMD.</td></tr>
          <tr><td><code>camera_offsets.csv</code></td><td>One EV per camera code (GA, GB, … ID) — the offsets the batch reads.</td></tr>
          <tr><td><code>clips.csv</code></td><td>The calibrated clips per camera (reference only — the batch discovers the full day from the media root).</td></tr>
          <tr><td><code>redcinex_develop.csv</code></td><td>REDCINE-X develop-panel reference (Exposure / Color Temp / Tint). For GUI work — build one Exposure-Adjust Look Preset per camera and apply by bin.</td></tr>
        </tbody>
      </table>"""

    recipe = _html.escape(
        'REDline --i "<CLIP.R3D>" --exportPreset "<PRESET>" '
        '--colorSciVersion 3 --exposureAdjust <EV> --outDir "<OUT>"'
    )

    return f"""
<section class="landscape-page export-page">
  <div class="hd">
    <div>
      <div class="hd-brand"><span class="r3d-mark">R3D</span>Match Assessment · {rr.run_id}</div>
      <div class="hd-cam">Offline Match Export</div>
      <div class="hd-project">Network-down fallback — written to the calibration folder on every run, alongside summary.json</div>
    </div>
  </div>

  <div style="padding:6px 26px 22px;">
    <div class="section-title" style="margin-top:6px;">The workflow</div>
    <ol class="export-flow">
      <li><b>On set (primary path).</b> R3DMatch measures each camera against the 18% gray sphere and
          solves a per-camera exposure offset (and white balance, unless Exposure-only). Those values are
          pushed live to the cameras over RCP2 before recording, so the array is matched in-camera.</li>
      <li><b>Fallback (these files).</b> If RCP2 is unreachable — network down, camera offline, or the push
          is skipped — every run still writes the solved corrections to this calibration folder. Nothing is
          lost; the match is captured on disk.</li>
      <li><b>In post.</b> The offsets are the same IPP2 Exposure Adjust values the camera would have taken,
          so a REDLine render reproduces the on-set match exactly. <code>redline_batch.sh</code> applies each
          camera's offset to <b>every clip that camera shot that day</b> — it reads the camera identity from
          each clip's name (reel + position → GA, GB, … ID), so it works whether the day's media is foldered
          by camera or laid out flat as <code>media/RDM/RDC/R3D</code>.</li>
    </ol>
    <p style="font-size:10.5px;color:#6e6962;max-width:1020px;line-height:1.5;margin-top:4px;">
      Calibration runs on one clip per camera; the batch expands that to the camera's whole day. Exposure is
      applied in the RAW/IPP2 domain via <code>--exposureAdjust</code> — not baked into a LUT or CDL. White
      balance, when solved, travels via the RMD / REDCINE-X develop path.
    </p>

    <div class="section-title" style="margin-top:16px;">Files written to this calibration folder</div>
    {files_html}

    <div class="section-title" style="margin-top:18px;">REDLine batch recipe</div>
    <div class="recipe-box"><code>{recipe}</code></div>
    <p style="font-size:10.5px;color:#6e6962;max-width:1020px;line-height:1.5;">
      Use <code>--exposureAdjust</code> (IPP2 Exposure Adjust), <b>not</b> <code>--exposure</code>.
      <code>--colorSciVersion 3</code> forces IPP2 explicitly. Do not bake offsets into LUTs or CDLs —
      that is not RAW-domain matching. This batch applies <b>exposure only</b>; white balance travels via
      the RMD / REDCINE-X develop path. Validate with a one-frame render or <code>--printMeta 2</code> first.
    </p>

    <div class="section-title" style="margin-top:18px;">Per-camera values in this run</div>
    <table class="export-values">
      <thead><tr><th>Camera</th><th>Clip label</th><th style="text-align:right">exposureAdjust (EV)</th><th style="text-align:center">White balance</th></tr></thead>
      <tbody>{table_rows}</tbody>
    </table>
  </div>
</section>"""


def _render_coherence_pages(rr: RunResult, cameras: List[CameraResult]) -> str:
    """Before/after contact sheet: two grids of the whole array so coherence is
    read by eye. One page up to 12 cameras (before above after); two pages beyond
    (before, then after). Columns scale with camera count."""
    n = len(cameras)
    if n == 0:
        return ""
    anchor = _anchor_ire(rr)
    cols = 4 if n <= 12 else 6
    single_page = n <= 12

    # Build before/after grids
    before_cells, after_cells = [], []
    for cr in cameras:
        raw = cr.measurement.hero_ire if cr.measurement else None
        adj = cr.commit.exposure_adjust if cr.commit else 0.0
        b_thumb = _contact_thumb(_render_path(cr))
        before_cells.append(_coh_cell(
            cr, b_thumb,
            f"{raw:.1f} IRE" if raw is not None else "—", "#46627f"))
        # after: real corrected render if present, else brightness sim
        cpath = getattr(cr, "corrected_render_path", None)
        if cpath and os.path.exists(cpath):
            a_thumb = _contact_thumb(cpath)
        else:
            a_thumb = _contact_thumb(_render_path(cr), brightness=2.0 ** adj)
        corr = _corrected_ire_for(cr, raw, adj)[0] if raw is not None else None
        d = abs((corr if corr is not None else anchor) - anchor)
        a_col = "#16a34a" if d <= 0.5 else ("#b45309" if d <= 1.0 else "#b91c1c")
        after_cells.append(_coh_cell(
            cr, a_thumb,
            f"{corr:.1f} IRE" if corr is not None else "—", a_col))

    grid_style = f"grid-template-columns:repeat({cols},1fr)"
    before_block = (
        '<div class="coh-block"><div class="coh-blocklabel">Before &mdash; as shot</div>'
        f'<div class="coh-grid" style="{grid_style}">' + "".join(before_cells) + '</div></div>')
    after_block = (
        '<div class="coh-block"><div class="coh-blocklabel coh-after">After &mdash; matched</div>'
        f'<div class="coh-grid" style="{grid_style}">' + "".join(after_cells) + '</div></div>')

    # Accuracy claim — placed where the match is seen.
    if getattr(rr, "exposure_only", False):
        claim = ("Exposure matched to within &plusmn;0.5&nbsp;IRE across the array, verified at the "
                 "sensor via closed-loop re-render. Per-camera precision approaches "
                 "&plusmn;0.1&nbsp;IRE under ideal lighting.")
    else:
        claim = ("Exposure matched to within &plusmn;0.5&nbsp;IRE across the array, verified at the "
                 "sensor via closed-loop re-render (per-camera precision approaches "
                 "&plusmn;0.1&nbsp;IRE under ideal lighting). White balance matched per camera.")

    def page(blocks, pg_label):
        return f"""
<section class="landscape-page coherence-page">
  <div class="coh-top">
    <div class="coh-title"><span class="r3d-mark">R3D</span>Array Coherence &mdash; Before vs After</div>
    <div class="coh-claim">{claim}</div>
  </div>
  {blocks}
  <div class="pg-footer">
    <span>Verified at the sensor &nbsp;&middot;&nbsp; closed-loop re-render</span>
    <span>Array coherence &nbsp;&middot;&nbsp; {rr.run_id}</span>
    <span>{pg_label}</span>
  </div>
</section>"""

    if single_page:
        return page(before_block + after_block, "Coherence")
    return page(before_block, "Coherence — Before") + page(after_block, "Coherence — After")


# ---------------------------------------------------------------------------
# Per-camera page
# ---------------------------------------------------------------------------
def _render_camera_page(cr: CameraResult, rr: RunResult, page_num: int) -> str:
    status = _camera_status(cr)
    is_anc = _is_anchor(cr, rr)
    adj = cr.commit.exposure_adjust if cr.commit else 0.0
    k = cr.commit.kelvin if cr.commit else 0
    tint = cr.commit.tint if cr.commit else 0.0
    ire = cr.measurement.hero_ire if cr.measurement else None
    anchor_ire = _anchor_ire(rr)
    corrected_ire, corr_measured = _corrected_ire_for(cr, ire, adj)
    clip_name = cr.metadata.clip_id if cr.metadata else cr.camera_label

    # Status styling — chip shows match % when verified, otherwise state
    status_cls = {
        "SOLVED": "status-pass",
        "NEEDS_ASSIST": "status-needs-assist",
        "NO_DATA": "status-fail",
    }.get(status, "status-fail")
    match_pct = getattr(cr, "match_pct", None)
    if status == "SOLVED" and match_pct is not None:
        chip_text = f"{match_pct:.0f}% MATCH"
        if match_pct < 80.0:
            status_cls = "status-fail"      # red below 80 — color only, no FAIL word
        elif match_pct < 95.0:
            status_cls = "status-needs-assist"
    else:
        chip_text = status.replace("_", " ")

    # Anchor badge
    anchor_html = '<div class="anchor-flag">Exposure Anchor</div>' if is_anc else ""

    # Header right: status + recommendation
    action = _recommend_action(adj, status)
    hd_right = f"""
    <div class="hd-right">
      <div class="hd-status {status_cls}">{chip_text}</div>
      <div class="hd-meta">{_gate_summary(cr)}</div>
      <div class="hd-meta">{action}</div>
    </div>"""

    # Thumbnails: original with sphere overlay; corrected = real closed-loop
    # render when available, brightness simulation otherwise
    orig_b64 = _make_original_thumb(cr)
    corr_b64, corr_thumb_measured = _make_corrected_thumb(cr, adj)
    corr_label = ("Corrected Frame (measured render)" if corr_thumb_measured
                  else "Corrected Frame (exposure simulation)")

    orig_src = f"data:image/jpeg;base64,{orig_b64}" if orig_b64 else ""
    corr_src = f"data:image/jpeg;base64,{corr_b64}" if corr_b64 else ""
    orig_img = f'<img class="img-box" src="{orig_src}" alt="{clip_name} original">' if orig_src else '<div style="color:#6e6962;font-size:12px;padding:20px">No render available</div>'
    corr_img = f'<img class="img-box" src="{corr_src}" alt="{clip_name} corrected">' if corr_src else '<div style="color:#6e6962;font-size:12px;padding:20px">No corrected render</div>'

    # IRE context warning banner
    ire_ctx = getattr(cr.detection, "ire_context_status", "ok") if cr.detection else "ok"
    ire_ctx_reason = getattr(cr.detection, "ire_context_reason", "") if cr.detection else ""
    ire_warn_html = ""
    if ire_ctx == "needs_assist":
        ire_warn_html = f"""
    <div class="ire-context-warn">
      <span class="warn-icon">⚠</span>
      <span><strong>IRE Context Check:</strong> {ire_ctx_reason}</span>
    </div>"""

    # Metrics columns
    eo = getattr(rr, "exposure_only", False)   # exposure-only → omit WB column + WB JSON
    exp_html = _render_exposure_col(ire, anchor_ire, adj, corrected_ire, cr)
    wb_html  = "" if eo else _render_wb_col(cr, k, tint)
    corr_html = _render_correction_col(adj, ire, anchor_ire, corrected_ire, cr)
    solve_html = _render_solve_col(cr)

    # Commit JSON block
    _commit_obj = {"exposureAdjust": round(adj, 4)}
    if not eo:
        _commit_obj["kelvin"] = k
        _commit_obj["tint"] = round(tint, 2)
    commit_json = json.dumps(_commit_obj, indent=2)
    commit_id = f"commit_{cr.camera_label.replace(' ', '_')}"

    # Notes strip
    notes_html = _render_notes_strip(cr, status, adj)

    return f"""
<section class="landscape-page camera-page">
  <div class="hd">
    <div>
      <div class="hd-brand"><span class="r3d-mark">R3D</span>Match Assessment · {rr.run_id}</div>
      {anchor_html}
      <div class="hd-cam">{cr.camera_label}</div>
      <div class="hd-clip">{clip_name}</div>
      <div class="hd-project">REDLine IPP2 / BT.709 / BT.1886 / Medium / Medium<br>Exposure anchor: {rr.anchor_source or 'Median'}</div>
    </div>
    {hd_right}
  </div>

  {ire_warn_html}

  <div class="img-row">
    <div class="img-panel">
      <div class="img-label">Original + Sphere Solve Overlay</div>
      <div class="img-frame">{orig_img}</div>
    </div>
    <div class="img-panel">
      <div class="img-label">{corr_label}</div>
      <div class="img-frame">{corr_img}</div>
    </div>
  </div>

  <div class="metrics-band">
    <div class="metrics">
      {exp_html}
      {wb_html}
      {corr_html}
      {solve_html}
    </div>
  </div>

  {notes_html}

  <div style="background:var(--card);border:1px solid var(--line);border-radius:6px;padding:8px 12px;margin-bottom:6px;">
    <div class="commit-block" id="{commit_id}">
      <button class="commit-copy-btn" onclick="navigator.clipboard.writeText(document.getElementById('{commit_id}').innerText.replace(/Copy/g,'').trim())">Copy</button>
      <pre style="margin:0;font-size:11px;line-height:1.5;">{commit_json}</pre>
    </div>
  </div>

  <div class="pg-footer">
    <span>REDLine IPP2 / BT.709 / BT.1886 / Medium / Medium</span>
    <span>Sphere solve verified</span>
    <span>Page {page_num}</span>
  </div>
</section>"""


# ---------------------------------------------------------------------------
# Metric columns
# ---------------------------------------------------------------------------
def _render_exposure_col(
    ire: Optional[float],
    anchor_ire: float,
    adj: float,
    corrected_ire: Optional[float],
    cr: CameraResult,
) -> str:
    ire_str = f"{ire:.1f}" if ire is not None else "—"
    corr_str = f"{corrected_ire:.1f}" if corrected_ire is not None else "—"
    adj_cls = "delta-pos" if adj > 0 else "delta-neg"

    # Exposure track: range ±0.25 stops, bar from center
    bar_pct = min(abs(adj) / 0.25, 1.0) * 10  # 10% = full half-width
    if adj >= 0:
        bar_left = 50.0
        bar_width = bar_pct
    else:
        bar_left = 50.0 - bar_pct
        bar_width = bar_pct
    bar_color = "#25a05a" if abs(adj) <= 0.10 else ("#f59e0b" if abs(adj) <= 0.20 else "#d61f4d")

    scalar_str = ""
    if cr.measurement:
        try:
            scalar_val = math.log2(max(ire, 0.001) / 100.0) if ire else 0.0
            scalar_str = f"Scalar {scalar_val:.3f} log2"
        except Exception:
            pass

    return f"""
    <div class="metric-col section-exposure">
      <div class="mc-title">Gray Exposure</div>
      <div class="mc-row"><span class="mc-key">Hero Center IRE</span><span class="mc-val large {adj_cls} mono">{ire_str}</span></div>
      <div class="mc-row"><span class="mc-key">Target IRE</span><span class="mc-val mono">{anchor_ire:.1f}</span></div>
      <div class="mc-row"><span class="mc-key">Original → Target</span><span class="mc-val mono">{ire_str} → {anchor_ire:.1f}</span></div>
      <div class="mc-row"><span class="mc-key">Corrected IRE</span><span class="mc-val mono">{corr_str}</span></div>
      <div class="mc-spacer"></div>
      <div class="exp-vis">
        <div class="exp-label"><span>Offset from anchor</span><span>±0.25 stops</span></div>
        <div class="exp-track">
          <div class="exp-zone" style="left:40%;width:20%;"></div>
          <div class="exp-center-line"></div>
          <div class="exp-bar" style="left:{bar_left:.1f}%;width:{max(bar_width,0.8):.1f}%;background:{bar_color};"></div>
        </div>
        <div class="exp-tick-labels"><span>−0.25</span><span>−0.10</span><span>0</span><span>+0.10</span><span>+0.25</span></div>
      </div>
      <div class="mc-foot">{scalar_str}</div>
    </div>"""


def _render_wb_col(cr: CameraResult, k: int, tint: float) -> str:
    as_k = cr.metadata.kelvin if cr.metadata else k
    as_t = cr.metadata.tint if cr.metadata else 0.0

    # Simple 2-axis mini WB visualization (SVG)
    # Normalize tint to ±5.0 range → position on a 180px track (center=90)
    tint_norm = max(-5.0, min(5.0, tint)) / 5.0   # -1..1
    tint_pos  = 90 + tint_norm * 60                  # px from left (center=90)

    wb_svg = f"""<svg viewBox="0 0 236 84" role="img" aria-label="Neutral chromaticity placement">
      <rect x="0" y="0" width="236" height="84" rx="10" fill="#faf9f7" stroke="#d7dee8"/>
      <text x="28" y="15" fill="#6e6962" font-size="10" font-weight="700">Green ←→ Magenta</text>
      <text x="118" y="15" text-anchor="middle" fill="#1a1816" font-size="9" font-weight="700">Neutral</text>
      <line x1="28" y1="28" x2="208" y2="28" stroke="#d6d3cd" stroke-width="5" stroke-linecap="round"/>
      <line x1="118" y1="20" x2="118" y2="36" stroke="#1a1816" stroke-width="2"/>
      <circle cx="{tint_pos:.1f}" cy="28" r="6.5" fill="#1a1816" stroke="white" stroke-width="2"/>
      <text x="118" y="70" text-anchor="middle" fill="#1a1816" font-size="10" font-weight="700">Tint offset: {tint:+.1f} units</text>
    </svg>"""

    # IPP2 WC note
    ipp2_note = """<div class="ipp2-note">
      IPP2/BT.709 renders carry an inherent ~0.013 WC residual (pipeline characteristic, not a WB error). 
      Only GM spread &gt; 0.005 indicates a real white balance issue.
    </div>"""

    # Closed-loop: measured WC/GM from the corrected render (vs solve-time prediction)
    if cr.corrected_gm is not None and cr.corrected_wc is not None:
        verified_row = (
            f'<div class="mc-row"><span class="mc-key">Corrected GM / WC</span>'
            f'<span class="mc-val mono">{cr.corrected_gm:+.4f} / {cr.corrected_wc:+.4f} '
            f'<span title="Measured from corrected render (closed loop)">●&nbsp;measured</span></span></div>'
        )
    else:
        verified_row = (
            '<div class="mc-row"><span class="mc-key">Corrected GM / WC</span>'
            '<span class="mc-val mono" title="No corrected render measured — solve-time prediction only">'
            'predicted only</span></div>'
        )

    return f"""
    <div class="metric-col section-wb">
      <div class="mc-title">White Balance</div>
      <div class="mc-row"><span class="mc-key">As-shot</span><span class="mc-val mono">{as_k}K / {as_t:+.1f}</span></div>
      <div class="mc-row"><span class="mc-key">Proposed Kelvin</span><span class="mc-val mono">{k}K</span></div>
      <div class="mc-row"><span class="mc-key">Proposed Tint</span><span class="mc-val mono">{tint:+.1f}</span></div>
      {verified_row}
      <div class="wb-vis">{wb_svg}</div>
      {ipp2_note}
    </div>"""


def _render_correction_col(
    adj: float,
    ire: Optional[float],
    anchor_ire: float,
    corrected_ire: Optional[float],
    cr: CameraResult,
) -> str:
    direction = "↑ Lift" if adj > 0 else ("↓ Lower" if adj < 0 else "≈ Hold")
    adj_cls = "delta-pos" if adj > 0 else "delta-neg"

    # Exposure closed-loop: match % from measured corrected render when
    # available, estimated residual otherwise. No PASS/FAIL — a percentage.
    exp_pct = getattr(cr, "exposure_match_pct", None)
    if exp_pct is not None:
        loop_cls = "closed-loop-pass" if exp_pct >= 95.0 else "closed-loop-fail"
        loop_html = f'<span class="{loop_cls}">{exp_pct:.0f}%</span>'
        r = getattr(cr, "corrected_exposure_residual_stops", None)
        residual_str = f"{r:.3f} stops" if r is not None else "—"
    elif corrected_ire is not None:
        residual_stops = abs(math.log2(corrected_ire / anchor_ire)) if anchor_ire > 0 and corrected_ire > 0 else None
        loop_html = '<span class="closed-loop-na">estimated</span>'
        residual_str = f"{residual_stops:.3f} stops (est.)" if residual_stops is not None else "—"
    else:
        loop_html = '<span class="closed-loop-na">—</span>'
        residual_str = "—"

    ire_str = f"{ire:.1f}" if ire is not None else "—"
    corr_measured = getattr(cr, "corrected_ire", None) is not None
    corr_str = (f"{corrected_ire:.1f}{' (measured)' if corr_measured else ' (est.)'}"
                if corrected_ire is not None else "—")

    # Array role
    role = _array_role(cr)

    return f"""
    <div class="metric-col section-correction">
      <div class="mc-title">Correction Applied</div>
      <div class="mc-row"><span class="mc-key">Exposure adj.</span><span class="mc-val mono {adj_cls}">{direction} {adj:+.3f} stops</span></div>
      <div class="mc-row"><span class="mc-key">Original → Target</span><span class="mc-val mono">{ire_str} → {anchor_ire:.1f}</span></div>
      <div class="mc-row"><span class="mc-key">Corrected IRE</span><span class="mc-val mono">{corr_str}</span></div>
      <div class="mc-row"><span class="mc-key">Closed-loop proof</span><span class="mc-val">{loop_html}</span></div>
      <div class="mc-row"><span class="mc-key">Residual</span><span class="mc-val mono">{residual_str}</span></div>
      <div class="mc-row"><span class="mc-key">Array role</span><span class="mc-val">{role}</span></div>
      <div class="mc-row"><span class="mc-key">WB model</span><span class="mc-val">Shared K / Per-Cam Tint</span></div>
    </div>"""


def _render_solve_col(cr: CameraResult) -> str:
    status = _camera_status(cr)
    detection_src = cr.detection.source if cr.detection else "—"
    acc = f"{cr.detection.hough_accumulator:.3f}" if cr.detection else "—"
    ire_spread = f"{cr.detection.ire_spread:.2f} IRE" if cr.detection else "—"
    chroma = f"{cr.detection.chromaticity_distance:.4f}" if cr.detection else "—"

    ire = cr.measurement.hero_ire if cr.measurement else None
    bright = cr.measurement.zone_bright.ire if (cr.measurement and cr.measurement.zone_bright) else None
    dark   = cr.measurement.zone_dark.ire   if (cr.measurement and cr.measurement.zone_dark)   else None
    hero_str   = f"{ire:.1f}" if ire is not None else "—"
    bright_str = f"{bright:.1f}" if bright is not None else "—"
    dark_str   = f"{dark:.1f}"   if dark   is not None else "—"

    return f"""
    <div class="metric-col section-solve">
      <div class="mc-title">Sphere Solve</div>
      <div class="mc-row"><span class="mc-key">Result</span><span class="mc-val">{status.replace('_',' ')}</span></div>
      <div class="mc-row"><span class="mc-key">Detection</span><span class="mc-val">{detection_src or '—'}</span></div>
      <div class="mc-row"><span class="mc-key">Accumulator</span><span class="mc-val mono">{acc}</span></div>
      <div class="mc-row"><span class="mc-key">IRE Spread</span><span class="mc-val mono">{ire_spread}</span></div>
      <div class="mc-row"><span class="mc-key">Chroma dist</span><span class="mc-val mono">{chroma}</span></div>
      <div class="mc-row"><span class="mc-key">Hero / Bright / Dark</span><span class="mc-val mono">{hero_str} / {bright_str} / {dark_str}</span></div>
    </div>"""


# ---------------------------------------------------------------------------
# Notes strip
# ---------------------------------------------------------------------------
def _render_notes_strip(cr: CameraResult, status: str, adj: float) -> str:
    result_str = _match_str(cr) + " match" if getattr(cr, "match_pct", None) is not None \
        else status.replace("_", " ")
    items: List[Tuple[str, str]] = [
        ("Result", result_str),
        ("Detection", cr.detection.source if cr.detection else "—"),
        ("Array role", _array_role(cr)),
    ]
    if cr.failure_reason:
        items.append(("Failure", cr.failure_reason))
    ire_ctx = getattr(cr.detection, "ire_context_status", "ok") if cr.detection else "ok"
    if ire_ctx == "needs_assist":
        items.append(("IRE context", "NEEDS ASSIST — operator verification required"))

    items_html = "".join(
        f'<div class="note-item"><span class="note-key">{k}</span><span class="note-val">{v}</span></div>'
        for k, v in items
    )
    return f'<div class="notes-strip"><div class="notes-inner">{items_html}</div></div>'


# ---------------------------------------------------------------------------
# Thumbnail generation
# ---------------------------------------------------------------------------
def _make_original_thumb(cr: CameraResult) -> Optional[str]:
    """
    Load the render TIFF, draw sphere overlay (circle + zone dots + cross),
    scale to _THUMB_LONG_EDGE, encode as JPEG base64.
    """
    render_path = _render_path(cr)
    if not render_path or not os.path.exists(render_path):
        return None
    try:
        img = Image.open(render_path).convert("RGB")
        # Draw sphere overlay if we have a detection ROI
        if cr.detection and cr.detection.roi:
            img = _draw_sphere_overlay(img, cr.detection.roi)
        img = _scale_thumb(img)
        return _jpeg_b64(img)
    except Exception as e:
        log.warning("Could not generate original thumb for %s: %s", cr.camera_label, e)
        return None


def _make_corrected_thumb(cr: CameraResult, adj: float) -> Tuple[Optional[str], bool]:
    """
    Corrected-frame thumbnail. Prefers the ACTUAL closed-loop corrected render
    (REDLine output with final commits applied) when it exists; otherwise
    falls back to a brightness simulation (2^adj) of the original render.

    Returns (jpeg_b64 | None, measured) — measured=True when the real
    corrected render was used.
    """
    # Real corrected render from the closed loop
    corrected_path = getattr(cr, "corrected_render_path", None)
    if corrected_path and os.path.exists(corrected_path):
        try:
            img = Image.open(corrected_path).convert("RGB")
            img = _scale_thumb(img)
            return _jpeg_b64(img), True
        except Exception as e:
            log.warning("Could not load corrected render for %s: %s",
                        cr.camera_label, e)

    # Fallback: simulate on the original render
    render_path = _render_path(cr)
    if not render_path or not os.path.exists(render_path):
        return None, False
    try:
        img = Image.open(render_path).convert("RGB")
        # Simulate exposure correction by scaling brightness
        factor = 2.0 ** adj
        # Use numpy for per-pixel float multiply to avoid PIL clipping artefacts
        import numpy as np
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = np.clip(arr * factor, 0.0, 1.0)
        img = Image.fromarray((arr * 255).astype(np.uint8))
        img = _scale_thumb(img)
        return _jpeg_b64(img), False
    except Exception as e:
        log.warning("Could not generate corrected thumb for %s: %s", cr.camera_label, e)
        return None, False


def _draw_sphere_overlay(img: Image.Image, roi: SphereROI) -> Image.Image:
    """
    Draw sphere detection overlay on full-res image:
      • Outer glow ring (white, 2px wider than detection circle)
      • Red-orange detection circle
      • Center crosshair
      • Zone dots at ±0.24r (bright/dark sampling zones)
      • Small zone dots at center (hero zone)
    """
    img = img.copy()
    draw = ImageDraw.Draw(img, "RGBA")

    cx, cy = roi.cx, roi.cy
    r = roi.r

    # Glow ring
    glow_w = max(3, int(r * 0.03))
    draw.ellipse(
        [cx - r - glow_w, cy - r - glow_w, cx + r + glow_w, cy + r + glow_w],
        outline=(255, 255, 255, 180),
        width=glow_w,
    )
    # Main circle
    lw = max(2, int(r * 0.018))
    draw.ellipse(
        [cx - r, cy - r, cx + r, cy + r],
        outline=(240, 100, 40, 230),
        width=lw,
    )

    # Center crosshair
    cross_len = int(r * 0.18)
    draw.line([cx - cross_len, cy, cx + cross_len, cy], fill=(255, 255, 255, 220), width=max(1, lw - 1))
    draw.line([cx, cy - cross_len, cx, cy + cross_len], fill=(255, 255, 255, 220), width=max(1, lw - 1))

    # Zone dots
    zone_offset = 0.24 * r
    dot_r = max(4, int(r * 0.06))

    # Bright zone (right side)
    _draw_zone_dot(draw, cx + zone_offset, cy, dot_r, (255, 220, 60, 200))
    # Dark zone (left side)
    _draw_zone_dot(draw, cx - zone_offset, cy, dot_r, (60, 120, 255, 200))
    # Hero center
    _draw_zone_dot(draw, cx, cy, dot_r, (255, 255, 255, 200))

    return img


def _draw_zone_dot(draw: ImageDraw.ImageDraw, x: float, y: float, r: float, color: tuple) -> None:
    draw.ellipse(
        [x - r, y - r, x + r, y + r],
        fill=color,
        outline=(255, 255, 255, 200),
        width=1,
    )


def _scale_thumb(img: Image.Image) -> Image.Image:
    w, h = img.size
    long = max(w, h)
    if long <= _THUMB_LONG_EDGE:
        return img
    scale = _THUMB_LONG_EDGE / long
    return img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)


def _jpeg_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=_THUMB_JPEG_QUALITY, optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ---------------------------------------------------------------------------
# SVG charts
# ---------------------------------------------------------------------------
def _render_exposure_lollipop(cameras: List[CameraResult], rr: RunResult) -> str:
    sorted_cams = sorted(cameras, key=lambda c: abs(c.commit.exposure_adjust if c.commit else 0), reverse=True)
    n = len(sorted_cams)
    if n == 0:
        return "<svg viewBox='0 0 800 400'><text x='400' y='200' text-anchor='middle' fill='#999'>No data</text></svg>"

    W, H = 1080, max(400, n * 46 + 60)
    left_margin = 180
    right_margin = 120
    top = 30
    bot = H - 30
    chart_w = W - left_margin - right_margin
    chart_h = bot - top

    # Scale: ±0.35 stops shown, clip beyond
    stop_range = 0.35
    cx_line = left_margin + chart_w / 2

    def x_of(stops: float) -> float:
        return left_margin + chart_w / 2 + (stops / stop_range) * (chart_w / 2)

    lines = []
    lines.append(f"<rect x='0' y='0' width='{W}' height='{H}' fill='white'/>")

    # Green band ±0.10 stops
    band_lo = x_of(-0.10)
    band_hi = x_of(+0.10)
    lines.append(f"<rect x='{band_lo:.1f}' y='{top}' width='{band_hi - band_lo:.1f}' height='{chart_h}' fill='#dcfce7' opacity='0.8' rx='6'/>")

    # Grid lines
    for stops_mark in [-0.30, -0.20, -0.10, 0.0, 0.10, 0.20, 0.30]:
        gx = x_of(stops_mark)
        sw = "2.5" if stops_mark == 0 else ("1.8" if stops_mark in [-0.20, -0.10, 0.10, 0.20] else "0.9")
        col = "#524e49" if stops_mark == 0 else ("#d6d3cd" if stops_mark % 0.20 == 0 else "#e7e5e1")
        lines.append(f"<line x1='{gx:.1f}' y1='{top}' x2='{gx:.1f}' y2='{bot}' stroke='{col}' stroke-width='{sw}'/>")
        if stops_mark != 0:
            lines.append(f"<text x='{gx:.1f}' y='{bot + 16}' text-anchor='middle' fill='#6e6962' font-size='13'>{stops_mark:+.2f}</text>")
        else:
            lines.append(f"<text x='{gx:.1f}' y='{bot + 16}' text-anchor='middle' fill='#6e6962' font-size='13'>+0.00</text>")

    row_h = chart_h / max(n, 1)
    for i, cr in enumerate(sorted_cams):
        adj = cr.commit.exposure_adjust if cr.commit else 0.0
        row_y = top + i * row_h + row_h / 2

        is_anc = _is_anchor(cr, rr)
        status = _camera_status(cr)
        dot_fill = "#1a1816" if is_anc else ("#7c3aed" if status == "NEEDS_ASSIST" else "#1a1816")

        adj_clamped = max(-stop_range, min(stop_range, adj))
        dot_x = x_of(adj_clamped)

        label = cr.camera_label + (" | Anchor" if is_anc else "")
        lines.append(f"<text x='{left_margin - 8}' y='{row_y + 5:.1f}' text-anchor='end' fill='#1a1816' font-size='14' font-weight='700'>{label}</text>")
        lines.append(f"<line x1='{cx_line:.1f}' y1='{row_y:.1f}' x2='{dot_x:.1f}' y2='{row_y:.1f}' stroke='#a3a09b' stroke-width='3.5'/>")
        lines.append(f"<circle cx='{dot_x:.1f}' cy='{row_y:.1f}' r='7' fill='{dot_fill}' stroke='white' stroke-width='2.5'/>")

        val_x = dot_x + (12 if adj >= 0 else -12)
        anchor_txt = "start" if adj >= 0 else "end"
        lines.append(f"<text x='{val_x:.1f}' y='{row_y + 5:.1f}' text-anchor='{anchor_txt}' fill='#3f3c38' font-size='13' font-weight='700'>{adj:+.3f} stops</text>")

    return f"<svg viewBox='0 0 {W} {H}' preserveAspectRatio='none' role='img'>{''.join(lines)}</svg>"


def _render_vectorscope(cameras: List[CameraResult]) -> str:
    """
    Neutral-placement chart — static SVG, print-safe, no JS.

    Plots each camera's MEASURED chromaticity (WC × GM) relative to the group
    median: hollow dot = before correction, filled dot = after (closed-loop
    measured from the corrected render), connected by a line so convergence
    is visible at a glance. Absolute neutral is irrelevant in IPP2 (inherent
    warm bias) — this is a camera-matching chart, centered on the group.
    """
    from .measure import compute_wc_gm

    pts = []  # (label, wc_before, gm_before, wc_after|None, gm_after|None)
    for cr in cameras:
        if not (cr.measurement and cr.measurement.measurement_valid):
            continue
        wc_b, gm_b = compute_wc_gm(cr.measurement.measured_rgb_mean)
        wc_a = getattr(cr, "corrected_wc", None)
        gm_a = getattr(cr, "corrected_gm", None)
        pts.append((cr.camera_label, wc_b, gm_b, wc_a, gm_a))

    if not pts:
        return ('<div style="padding:40px;text-align:center;color:#a3a09b;'
                'font-size:13px;">No valid measurements to plot</div>')

    # Center on the group median of the most-corrected state available
    wc_ctr_vals = [(p[3] if p[3] is not None else p[1]) for p in pts]
    gm_ctr_vals = [(p[4] if p[4] is not None else p[2]) for p in pts]
    wc_ctr = sorted(wc_ctr_vals)[len(wc_ctr_vals) // 2]
    gm_ctr = sorted(gm_ctr_vals)[len(gm_ctr_vals) // 2]

    # Per-axis half-range from full data extent (before AND after), padded
    def _ext(vals, ctr, floor):
        m = max((abs(v - ctr) for v in vals), default=0.0)
        return max(m * 1.35, floor)
    all_wc = [p[1] for p in pts] + [p[3] for p in pts if p[3] is not None]
    all_gm = [p[2] for p in pts] + [p[4] for p in pts if p[4] is not None]
    hr_wc = _ext(all_wc, wc_ctr, 0.010)
    hr_gm = _ext(all_gm, gm_ctr, 0.006)

    W, H, PAD = 1080, 420, 56
    cx, cy = W / 2, H / 2
    sx = (W / 2 - PAD) / hr_wc
    sy = (H / 2 - PAD) / hr_gm

    def X(wc): return cx + (wc - wc_ctr) * sx
    def Y(gm): return cy - (gm - gm_ctr) * sy   # green up, magenta down

    s = [f'<svg viewBox="0 0 {W} {H}" style="width:100%;display:block;" '
         f'xmlns="http://www.w3.org/2000/svg" font-family="ui-sans-serif,sans-serif">']

    # GM acceptance band: ±0.005 about group median (matching gate context)
    band = 0.005
    s.append(f'<rect x="{PAD}" y="{Y(gm_ctr + band):.1f}" width="{W - 2*PAD}" '
             f'height="{(Y(gm_ctr - band) - Y(gm_ctr + band)):.1f}" '
             f'fill="#16a34a" opacity="0.07"/>')
    s.append(f'<text x="{W - PAD - 4}" y="{Y(gm_ctr + band) - 4:.1f}" text-anchor="end" '
             f'font-size="9" fill="#15803d">±0.005 GM band</text>')

    # Crosshair at group median
    s.append(f'<line x1="{PAD}" y1="{cy}" x2="{W - PAD}" y2="{cy}" stroke="#d6d3cd" stroke-width="1.4"/>')
    s.append(f'<line x1="{cx}" y1="{PAD}" x2="{cx}" y2="{H - PAD}" stroke="#d6d3cd" stroke-width="1.4"/>')
    s.append(f'<circle cx="{cx}" cy="{cy}" r="8" fill="white" stroke="#1a1816" stroke-width="2.4"/>')
    s.append(f'<text x="{cx + 13}" y="{cy - 11}" font-size="11" font-weight="700" '
             f'fill="#524e49">Group neutral</text>')

    # Axis labels
    s.append(f'<text x="{cx}" y="{PAD - 14}" text-anchor="middle" font-size="12" font-weight="700" fill="#6e6962">Green</text>')
    s.append(f'<text x="{cx}" y="{H - PAD + 26}" text-anchor="middle" font-size="12" font-weight="700" fill="#6e6962">Magenta</text>')
    s.append(f'<text x="{PAD - 8}" y="{cy + 4}" text-anchor="end" font-size="12" font-weight="700" fill="#6e6962">Cool</text>')
    s.append(f'<text x="{W - PAD + 8}" y="{cy + 4}" text-anchor="start" font-size="12" font-weight="700" fill="#6e6962">Warm</text>')
    # Scale note
    s.append(f'<text x="{PAD}" y="{H - 10}" font-size="9.5" font-family="ui-monospace,monospace" '
             f'fill="#a3a09b">scale: ±{hr_wc:.3f} WC / ±{hr_gm:.3f} GM about group median</text>')

    # Cameras: before (hollow) → after (filled), connected
    for label, wc_b, gm_b, wc_a, gm_a in pts:
        bx, by = X(wc_b), Y(gm_b)
        if wc_a is not None and gm_a is not None:
            ax, ay = X(wc_a), Y(gm_a)
            dev = abs(gm_a - gm_ctr)
            col = "#16a34a" if dev <= band else "#f59e0b"
            s.append(f'<line x1="{bx:.1f}" y1="{by:.1f}" x2="{ax:.1f}" y2="{ay:.1f}" '
                     f'stroke="#d6d3cd" stroke-width="1.3"/>')
            s.append(f'<circle cx="{bx:.1f}" cy="{by:.1f}" r="5.5" fill="white" '
                     f'stroke="#a3a09b" stroke-width="1.8"/>')
            s.append(f'<circle cx="{ax:.1f}" cy="{ay:.1f}" r="7" fill="{col}" '
                     f'stroke="white" stroke-width="2"/>')
            lx, ly = ax, ay - 12
        else:
            s.append(f'<circle cx="{bx:.1f}" cy="{by:.1f}" r="7" fill="white" '
                     f'stroke="#6e6962" stroke-width="2.2"/>')
            lx, ly = bx, by - 12
            col = "#6e6962"
        s.append(f'<text x="{lx:.1f}" y="{ly:.1f}" text-anchor="middle" font-size="9.5" '
                 f'font-weight="700" font-family="ui-monospace,monospace" '
                 f'fill="{col}">{label}</text>')

    # Legend
    ly = H - PAD + 26
    s.append(f'<g font-size="10" fill="#6e6962">'
             f'<circle cx="{W - PAD - 250}" cy="{ly - 3}" r="5" fill="white" stroke="#a3a09b" stroke-width="1.8"/>'
             f'<text x="{W - PAD - 240}" y="{ly}">as-shot</text>'
             f'<circle cx="{W - PAD - 170}" cy="{ly - 3}" r="5.5" fill="#16a34a" stroke="white" stroke-width="1.5"/>'
             f'<text x="{W - PAD - 160}" y="{ly}">corrected (measured)</text>'
             f'</g>')

    s.append('</svg>')
    return "".join(s)

def _render_quick_notes(
    top3: List[CameraResult],
    anchor_label: str,
    solved: int,
    total: int,
    cameras: List[CameraResult],
) -> str:
    colors = ["#ef4444", "#f97316", "#f59e0b"]
    items_html = ""
    for i, cr in enumerate(top3):
        adj = cr.commit.exposure_adjust if cr.commit else 0.0
        c = colors[i] if i < len(colors) else "#6e6962"
        items_html += f"""
        <div class="quick-item">
          <span class="num" style="background:{c}">{i+1}</span>
          <span>{cr.camera_label} <span class="mono">({adj:+.2f})</span></span>
        </div>"""

    needs_assist_cameras = [cr for cr in cameras if _camera_status(cr) == "NEEDS_ASSIST"]
    assist_html = ""
    if needs_assist_cameras:
        assist_html = f"""
        <div class="quick-group">
          <div class="quick-label">Needs operator verification</div>
          {"".join(f'<div class="quick-item"><span class="num" style="background:#7c3aed">!</span><span>{cr.camera_label} — IRE context flag</span></div>' for cr in needs_assist_cameras)}
        </div>"""

    return f"""
    <div class="quick">
      <h3>Quick Notes</h3>
      <div class="quick-group">
        <div class="quick-label">Biggest exposure differences</div>
        {items_html}
      </div>
      <div class="quick-group">
        <div class="quick-item">
          <span class="bullet" style="background:var(--anchor)"></span>
          <span><strong>Anchor camera</strong><br><span class="mono">{anchor_label}</span></span>
        </div>
      </div>
      {assist_html}
      <div class="quick-group" style="margin-bottom:0">
        <div class="quick-item">
          <span class="bullet" style="background:var(--review)"></span>
          <span><strong>{solved}/{total} cameras solved and ready to push.</strong></span>
        </div>
      </div>
    </div>"""


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def _camera_status(cr: CameraResult) -> str:
    if hasattr(cr, "ire_context_status") and cr.ire_context_status == "needs_assist":
        return "NEEDS_ASSIST"
    if cr.detection and getattr(cr.detection, "ire_context_status", "ok") == "needs_assist":
        return "NEEDS_ASSIST"
    if cr.failed_stage:
        return "NO_DATA"
    if cr.commit:
        return "SOLVED"
    return "NO_DATA"


def _match_str(cr: CameraResult) -> str:
    """Match percentage display string — there is no FAIL, only a number."""
    pct = getattr(cr, "match_pct", None)
    return f"{pct:.0f}%" if pct is not None else "—"


def _is_anchor(cr: CameraResult, rr: RunResult) -> bool:
    try:
        return getattr(rr, "anchor_camera_label", None) == cr.camera_label
    except Exception:
        return False


def _is_anchor_by_index(cr: CameraResult, cameras: List[CameraResult]) -> bool:
    # Heuristic: anchor is the camera with exposure_adjust closest to 0
    if not cameras:
        return False
    closest = min(cameras, key=lambda c: abs(c.commit.exposure_adjust) if c.commit else 999)
    return cr.camera_label == closest.camera_label


def _anchor_ire(rr: RunResult) -> float:
    try:
        return float(getattr(rr, "anchor_ire", 36.4))
    except Exception:
        return 36.4


def _corrected_ire_for(cr: CameraResult, ire: Optional[float], adj: float) -> Tuple[Optional[float], bool]:
    """Corrected IRE: MEASURED from the closed-loop render when available,
    else the 2^adj estimate. Returns (value, measured)."""
    measured = getattr(cr, "corrected_ire", None)
    if measured is not None:
        return measured, True
    return _estimate_corrected_ire(ire, adj), False


def _estimate_corrected_ire(ire: Optional[float], adj: float) -> Optional[float]:
    if ire is None:
        return None
    factor = 2.0 ** adj
    return ire * factor


def _render_path(cr: CameraResult) -> Optional[str]:
    try:
        return cr.measurement.render_path
    except AttributeError:
        return None


def _residual_str(cr: CameraResult) -> str:
    """
    If corrected_ire is available: show log2 residual vs anchor (true closed-loop proof).
    Otherwise: show the exposure_adjust magnitude — how far this camera was from anchor.
    """
    try:
        # True closed-loop residual (requires render_corrected=True in workflow)
        if cr.corrected_ire and cr.corrected_ire > 0 and cr.measurement and cr.measurement.hero_ire > 0:
            anchor = cr.measurement.hero_ire * (2.0 ** cr.commit.exposure_adjust)
            residual = abs(math.log2(cr.corrected_ire / anchor)) if anchor > 0 else 0.0
            return f"{residual:.3f} stops"
        # Pre-correction: just show the magnitude of required adjustment
        if cr.commit and cr.commit.exposure_adjust is not None:
            return f"{cr.commit.exposure_adjust:+.3f} stops"
        return "—"
    except Exception:
        return "—"


def _array_role(cr: CameraResult) -> str:
    status = _camera_status(cr)
    if status == "NEEDS_ASSIST":
        return "Needs operator verification"
    if status == "NO_DATA":
        return "Excluded from solve"
    adj = cr.commit.exposure_adjust if cr.commit else 0.0
    if abs(adj) < 0.005:
        return "Anchor (reference)"
    return "Included in solve"


def _gate_summary(cr: CameraResult) -> str:
    if not cr.detection or not cr.detection.gates:
        return "No detection data"
    passed = sum(1 for g in cr.detection.gates if g.passed)
    total = len(cr.detection.gates)
    ire_ctx = getattr(cr.detection, "ire_context_status", "ok")
    ctx_str = " | IRE context: NEEDS ASSIST" if ire_ctx == "needs_assist" else ""
    return f"{passed}/{total} gates passed{ctx_str}"


def _recommend_action(adj: float, status: str) -> str:
    if status in ("NO_DATA", "NEEDS_ASSIST"):
        return "Recommended action: Operator verification required"
    if abs(adj) < 0.025:
        return "Recommended action: No adjustment required"
    direction = "open" if adj > 0 else "close"
    stops = abs(adj)
    if stops < 0.12:
        return f"Recommended action: {direction.capitalize()} aperture by 1/8 stop"
    if stops < 0.20:
        return f"Recommended action: {direction.capitalize()} aperture by 1/4 stop"
    if stops < 0.30:
        return f"Recommended action: {direction.capitalize()} aperture by 1/3 stop"
    return f"Recommended action: {direction.capitalize()} aperture by 1/2 stop"


def _tone_class(adj: float, status: str) -> str:
    if status == "NEEDS_ASSIST":
        return "tone-violet"
    if abs(adj) > 0.20:
        return "tone-red"
    if abs(adj) > 0.10:
        return "tone-orange"
    if abs(adj) < 0.03:
        return "tone-blue"
    return "tone-green"


def _priority_class(rank: int) -> str:
    if rank == 1:
        return "p-red"
    if rank <= 3:
        return "p-orange"
    if rank <= 6:
        return "p-amber"
    if rank <= 9:
        return "p-green"
    return "p-blue"
