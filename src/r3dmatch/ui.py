from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional


def _load_json(path: Path) -> Optional[Dict[str, object]]:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_review_bundle(output_dir: str) -> Dict[str, object]:
    root = Path(output_dir).expanduser().resolve()
    summary = _load_json(root / "summary.json")
    array_calibration = _load_json(root / "array_calibration.json")
    analysis_dir = root / "analysis"
    sidecar_dir = root / "sidecars"

    analysis_records = []
    if analysis_dir.exists():
        for path in sorted(analysis_dir.glob("*.analysis.json")):
            analysis_records.append(json.loads(path.read_text(encoding="utf-8")))

    sidecars: Dict[str, Dict[str, object]] = {}
    if sidecar_dir.exists():
        for path in sorted(sidecar_dir.glob("*.sidecar.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            sidecars[str(payload["clip_id"])] = payload

    camera_rows = []
    if array_calibration and array_calibration.get("cameras"):
        for camera in array_calibration["cameras"]:
            clip_id = str(camera["clip_id"])
            solution = dict(camera.get("solution", {}))
            measurement = dict(camera.get("measurement", {}))
            quality = dict(camera.get("quality", {}))
            sidecar = sidecars.get(clip_id, {})
            camera_rows.append(
                {
                    "clip_id": clip_id,
                    "camera_id": camera.get("camera_id"),
                    "group_key": camera.get("group_key"),
                    "measured_log2_luminance": measurement.get("measured_log2_luminance"),
                    "exposure_offset_stops": solution.get("exposure_offset_stops"),
                    "rgb_gains": solution.get("rgb_gains"),
                    "confidence": quality.get("confidence"),
                    "flags": quality.get("flags", []),
                    "color_residual": quality.get("color_residual"),
                    "exposure_residual_stops": quality.get("exposure_residual_stops"),
                    "source_path": camera.get("source_path"),
                    "sidecar_path": str(root / "sidecars" / f"{clip_id}.sidecar.json") if clip_id in sidecars else None,
                    "preview_path": _find_preview_path(root, clip_id),
                    "sidecar_loaded": bool(sidecar),
                }
            )
    else:
        for record in analysis_records:
            clip_id = str(record["clip_id"])
            camera_rows.append(
                {
                    "clip_id": clip_id,
                    "camera_id": clip_id,
                    "group_key": record.get("group_key"),
                    "measured_log2_luminance": record.get("diagnostics", {}).get("measured_log2_luminance"),
                    "exposure_offset_stops": record.get("final_offset_stops"),
                    "rgb_gains": record.get("pending_color_gains"),
                    "confidence": record.get("confidence"),
                    "flags": [],
                    "color_residual": None,
                    "exposure_residual_stops": None,
                    "source_path": record.get("source_path"),
                    "sidecar_path": str(root / "sidecars" / f"{clip_id}.sidecar.json") if clip_id in sidecars else None,
                    "preview_path": _find_preview_path(root, clip_id),
                    "sidecar_loaded": clip_id in sidecars,
                }
            )

    rows = sorted(camera_rows, key=lambda row: row["clip_id"])
    exposure_offsets = [float(row["exposure_offset_stops"]) for row in rows if row["exposure_offset_stops"] is not None]
    color_residuals = [float(row["color_residual"]) for row in rows if row["color_residual"] is not None]
    summary_text = {
        "clip_count": len(rows),
        "min_exposure_offset": min(exposure_offsets) if exposure_offsets else None,
        "max_exposure_offset": max(exposure_offsets) if exposure_offsets else None,
        "largest_color_deviation": max(color_residuals) if color_residuals else None,
    }
    return {
        "root": str(root),
        "summary": summary,
        "array_calibration": array_calibration,
        "analysis_records": analysis_records,
        "sidecars": sidecars,
        "rows": rows,
        "summary_text": summary_text,
    }


def _find_preview_path(root: Path, clip_id: str) -> Optional[str]:
    for candidate in (
        root / "previews" / f"{clip_id}.jpg",
        root / "previews" / f"{clip_id}.png",
        root / "stills" / f"{clip_id}.jpg",
        root / "stills" / f"{clip_id}.png",
    ):
        if candidate.exists():
            return str(candidate)
    return None


def _exposure_outlier_threshold(rows: List[Dict[str, object]]) -> float:
    values = [float(row["exposure_offset_stops"]) for row in rows if row["exposure_offset_stops"] is not None]
    if not values:
        return 0.5
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / max(len(values), 1)
    return max(0.15, 2.0 * math.sqrt(variance))


def build_table_rows(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    threshold = _exposure_outlier_threshold(rows)
    formatted = []
    for row in rows:
        exposure = row["exposure_offset_stops"]
        color_residual = row["color_residual"] if row["color_residual"] is not None else 0.0
        outlier = abs(float(exposure or 0.0)) > threshold or float(color_residual) > 0.05
        formatted.append(
            {
                "clip_id": row["clip_id"],
                "camera_id": row["camera_id"],
                "group_key": row["group_key"],
                "measured_log2_luminance": row["measured_log2_luminance"],
                "exposure_offset_stops": row["exposure_offset_stops"],
                "rgb_gains": row["rgb_gains"],
                "confidence": row["confidence"],
                "flags": row["flags"],
                "outlier": "YES" if outlier else "",
            }
        )
    return formatted


def render_review_app(output_dir: str) -> None:
    import streamlit as st

    bundle = load_review_bundle(output_dir)
    rows = bundle["rows"]
    st.set_page_config(page_title="R3DMatch Review", layout="wide")
    st.title("R3DMatch Calibration Review")
    st.caption(bundle["root"])

    array_calibration = bundle["array_calibration"] or {}
    target = dict(array_calibration.get("target", {}))
    exposure_target = dict(target.get("exposure", {}))
    color_target = dict(target.get("color", {}))

    col1, col2, col3 = st.columns(3)
    col1.metric("Clip Count", bundle["summary_text"]["clip_count"])
    col2.metric(
        "Target Exposure (log2)",
        f"{exposure_target.get('log2_luminance_target', 0.0):.3f}" if exposure_target else "n/a",
    )
    color_value = color_target.get("target_rgb_chromaticity")
    col3.metric(
        "Target Chromaticity",
        ", ".join(f"{float(value):.3f}" for value in color_value) if color_value else "n/a",
    )

    st.subheader("Batch Summary")
    st.write(
        {
            "clip_count": bundle["summary_text"]["clip_count"],
            "min_exposure_offset": bundle["summary_text"]["min_exposure_offset"],
            "max_exposure_offset": bundle["summary_text"]["max_exposure_offset"],
            "largest_color_deviation": bundle["summary_text"]["largest_color_deviation"],
        }
    )

    st.subheader("Exposure Offsets")
    chart_data = {
        row["clip_id"]: float(row["exposure_offset_stops"])
        for row in rows
        if row["exposure_offset_stops"] is not None
    }
    if chart_data:
        st.bar_chart(chart_data)
    else:
        st.info("No exposure offsets found.")

    st.subheader("Camera Table")
    st.dataframe(build_table_rows(rows), use_container_width=True)

    st.subheader("Outliers")
    threshold = _exposure_outlier_threshold(rows)
    outliers = []
    for row in rows:
        exposure = float(row["exposure_offset_stops"] or 0.0)
        color_residual = float(row["color_residual"] or 0.0)
        if abs(exposure) > threshold or color_residual > 0.05:
            color = "#7f1d1d" if abs(exposure) > threshold else "#92400e"
            outliers.append(
                f"<div style='padding:8px;margin:4px 0;background:{color};color:white;border-radius:6px;'>"
                f"{row['clip_id']} | exposure={exposure:.3f} | color_residual={color_residual:.3f} | flags={row['flags']}"
                "</div>"
            )
    if outliers:
        st.markdown("".join(outliers), unsafe_allow_html=True)
    else:
        st.success("No obvious outliers detected.")

    preview_rows = [row for row in rows if row["preview_path"]]
    if preview_rows:
        st.subheader("Preview Stills")
        for row in preview_rows:
            st.markdown(f"**{row['clip_id']}**")
            st.image(row["preview_path"], caption=row["clip_id"])


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output-folder", default=".", help="Analyze output folder containing summary.json and sidecars/")
    return parser.parse_known_args()[0]


def main() -> None:
    args = _parse_args()
    render_review_app(args.output_folder)


if __name__ == "__main__":
    main()
