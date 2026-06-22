"""
R3DMatch v2 — Progress Emission

Structured JSON progress on stdout. One line per event.
The web app reads these lines to update the UI.

Format:
  {"type": "progress", "phase": "...", "pct": 0-100, "detail": "...", "clip_id": "..."}
"""
from __future__ import annotations

import json
import sys
import time
from typing import Optional


def emit(
    phase: str,
    *,
    pct: int = 0,
    detail: str = "",
    clip_id: Optional[str] = None,
    clip_index: Optional[int] = None,
    clip_count: Optional[int] = None,
    error: bool = False,
    **extra,
) -> None:
    """Emit one progress line to stdout.

    Extra keyword args (JSON-serializable) are merged into the payload —
    e.g. clip_fps / clip_res so the UI can fill metadata live.
    """
    payload = {
        "type": "progress",
        "phase": phase,
        "pct": max(0, min(100, pct)),
        "detail": detail,
        "ts": time.time(),
    }
    if clip_id:
        payload["clip_id"] = clip_id
    if clip_index is not None:
        payload["clip_index"] = clip_index
    if clip_count is not None:
        payload["clip_count"] = clip_count
    if error:
        payload["error"] = True
    for k, v in extra.items():
        if v is not None:
            payload[k] = v
    try:
        print(json.dumps(payload), flush=True)
    except (BrokenPipeError, OSError):
        pass


def emit_phase_pct(clip_index: int, clip_count: int, phase_start_pct: int, phase_end_pct: int) -> int:
    """Compute current percentage within a phase given clip progress."""
    if clip_count <= 0:
        return phase_start_pct
    span = phase_end_pct - phase_start_pct
    return phase_start_pct + int(span * (clip_index / clip_count))
