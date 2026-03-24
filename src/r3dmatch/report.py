from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from .calibration import load_color_calibration, load_exposure_calibration


def _load_analysis_records(input_path: str) -> List[Dict[str, object]]:
    root = Path(input_path).expanduser().resolve()
    analysis_dir = root / "analysis" if (root / "analysis").exists() else root
    records = []
    for path in sorted(analysis_dir.glob("*.analysis.json")):
        records.append(json.loads(path.read_text(encoding="utf-8")))
    return records


def build_contact_sheet_report(
    input_path: str,
    *,
    out_dir: str,
    exposure_calibration_path: Optional[str] = None,
    color_calibration_path: Optional[str] = None,
) -> Dict[str, object]:
    analysis_records = _load_analysis_records(input_path)
    exposure = load_exposure_calibration(exposure_calibration_path) if exposure_calibration_path else None
    color = load_color_calibration(color_calibration_path) if color_calibration_path else None
    exposure_by_group = {entry.group_key: entry for entry in exposure.cameras} if exposure else {}
    color_by_group = {entry.group_key: entry for entry in color.cameras} if color else {}

    clips = []
    for record in analysis_records:
        group_key = str(record["group_key"])
        clip_id = str(record["clip_id"])
        exposure_entry = exposure_by_group.get(group_key)
        color_entry = color_by_group.get(group_key)
        clips.append(
            {
                "clip_id": clip_id,
                "group_key": group_key,
                "exposure_metrics": {
                    "raw_offset_stops": record.get("raw_offset_stops"),
                    "camera_baseline_stops": record.get("camera_baseline_stops"),
                    "final_offset_stops": record.get("final_offset_stops"),
                    "calibration_measured_log2_luminance": exposure_entry.measured_log2_luminance if exposure_entry else None,
                },
                "color_metrics": {
                    "rgb_neutral_gains": color_entry.rgb_neutral_gains if color_entry else record.get("pending_color_gains"),
                    "measured_channel_medians": color_entry.measured_channel_medians if color_entry else None,
                },
                "sampling_mode": (
                    exposure_entry.sampling_mode if exposure_entry
                    else color_entry.sampling_mode if color_entry
                    else None
                ),
                "confidence": record.get("confidence"),
                "placeholder_images": {
                    "original_image": None,
                    "exposure_image": None,
                    "color_image": None,
                },
            }
        )

    payload = {
        "input_path": str(Path(input_path).expanduser().resolve()),
        "exposure_calibration_path": str(Path(exposure_calibration_path).expanduser().resolve()) if exposure_calibration_path else None,
        "color_calibration_path": str(Path(color_calibration_path).expanduser().resolve()) if color_calibration_path else None,
        "clip_count": len(clips),
        "clips": clips,
    }
    out_root = Path(out_dir).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    json_path = out_root / "contact_sheet.json"
    html_path = out_root / "contact_sheet.html"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    html_path.write_text(render_contact_sheet_html(payload), encoding="utf-8")
    return {
        "report_json": str(json_path),
        "report_html": str(html_path),
        "clip_count": len(clips),
    }


def render_contact_sheet_html(payload: Dict[str, object]) -> str:
    rows = []
    for clip in payload["clips"]:
        rows.append(
            "<section>"
            f"<h2>{clip['clip_id']}</h2>"
            f"<p>Group: {clip['group_key']}</p>"
            f"<pre>{json.dumps(clip, indent=2)}</pre>"
            "</section>"
        )
    body = "\n".join(rows)
    return (
        "<!doctype html><html><head><meta charset='utf-8'><title>R3DMatch Contact Sheet</title></head>"
        "<body><h1>R3DMatch Contact Sheet</h1>"
        f"<p>Clip count: {payload['clip_count']}</p>"
        f"{body}"
        "</body></html>"
    )
