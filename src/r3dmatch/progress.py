from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional


def review_progress_path_for(output_root: str | Path) -> Path:
    return Path(output_root).expanduser().resolve() / "review_progress.json"


def emit_review_progress(
    progress_path: str | Path | None,
    *,
    phase: str,
    detail: str,
    stage_label: Optional[str] = None,
    clip_index: Optional[int] = None,
    clip_count: Optional[int] = None,
    current_clip_id: Optional[str] = None,
    elapsed_seconds: Optional[float] = None,
    review_mode: Optional[str] = None,
    extra: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    now = time.time()
    payload: Dict[str, object] = {
        "phase": str(phase),
        "detail": str(detail),
        "stage_label": str(stage_label or ""),
        "clip_index": int(clip_index) if clip_index is not None else None,
        "clip_count": int(clip_count) if clip_count is not None else None,
        "current_clip_id": str(current_clip_id) if current_clip_id else None,
        "elapsed_seconds": float(elapsed_seconds) if elapsed_seconds is not None else None,
        "review_mode": str(review_mode) if review_mode else None,
        "updated_at": now,
        "updated_at_iso": datetime.now(timezone.utc).isoformat(),
    }
    if extra:
        payload["extra"] = dict(extra)
    message = f"[PROGRESS {payload['updated_at_iso']}] {phase}: {detail}"
    if clip_index is not None and clip_count is not None:
        message += f" ({clip_index}/{clip_count})"
    if current_clip_id:
        message += f" [{current_clip_id}]"
    if elapsed_seconds is not None:
        message += f" +{elapsed_seconds:.2f}s"
    print(message, flush=True)
    if progress_path:
        path = Path(progress_path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def load_review_progress(progress_path: str | Path | None, *, started_at: Optional[float] = None) -> Optional[Dict[str, object]]:
    if not progress_path:
        return None
    path = Path(progress_path).expanduser().resolve()
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    updated_at = float(payload.get("updated_at", 0.0) or 0.0)
    if started_at is not None and updated_at and updated_at < float(started_at):
        return None
    payload["progress_path"] = str(path)
    return payload
