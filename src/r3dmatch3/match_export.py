"""
match_export.py — offline hand-off of the on-set match.

Auto-written on EVERY run as a network-down fallback: if RCP2 is unreachable
(or the operator chooses not to push), post production still gets the exact
per-camera corrections that were solved on set.

Two artifacts, same numbers:

  • match_export.json        — tool-agnostic, full fidelity. Carries the raw +
                               divider integer form RCP2 itself uses, so any
                               pipeline can reproduce the values without re-
                               deriving them.
  • redcinex_develop.csv     — REDCINE-X develop-panel values (Exposure stops,
                               Color Temp K, Tint), one row per clip. A RED
                               colorist enters/verifies these directly, or uses
                               them to batch-apply RMD develop settings.

Nothing here is gated on a push succeeding. These files are the receipt.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .models import RunResult

# RCP2 integer form (mirrors rcp2._PARAM_ID / _FALLBACK_DIVIDER). Kept here so
# the export is self-describing without importing the live push layer.
_RCP2_PARAM_ID = {
    "exposureAdjust": "EXPOSURE_ADJUST",
    "kelvin":         "COLOR_TEMPERATURE",
    "tint":           "TINT",
}
_RCP2_DIVIDER = {
    "exposureAdjust": 100,   # raw -41  → -0.41 stops
    "kelvin":         1,     # raw 5600 →  5600 K
    "tint":           100,   # raw 12   →  0.12 tint
}


def _rcp2_params(commit) -> dict:
    """The exact raw/divider integers RCP2 would have written, for reproduction."""
    out = {}

    def _put(field: str, value: float):
        div = _RCP2_DIVIDER[field]
        out[_RCP2_PARAM_ID[field]] = {"raw": int(round(value * div)), "divider": div}

    _put("exposureAdjust", commit.exposure_adjust)
    if not commit.exposure_only:
        _put("kelvin", commit.kelvin)
        _put("tint", commit.tint)
    return out


def _cam_code(cr) -> str:
    """Short camera code (GA, GB, … HA, … ID) = reel-group letter + position.

    Matches the codes a REDLine batch bins clips by. Falls back to the label.
    """
    pos = (getattr(cr.metadata, "camera_position", "") or "").strip()
    label = (cr.camera_label or cr.clip_id or "").strip()
    grp = label[:1]
    code = (grp + pos).upper().strip()
    return code or label or cr.clip_id


# Verified against REDLine Build 65.1.3, which exposes --exposureAdjust directly
# (no RMD sidecar needed). --colorSciVersion 3 forces IPP2 explicitly.
_REDLINE_RECIPE = {
    "build_verified": "65.1.3",
    "parameter": "--exposureAdjust",
    "color_science": "--colorSciVersion 3",
    "per_camera_command": (
        'REDline --i "<CLIP.R3D>" --exportPreset "<YOUR PRESET>" '
        '--colorSciVersion 3 --exposureAdjust <EV> --outDir "<OUT>"'
    ),
    "value_meaning": (
        "exposure_adjust_ev is the exact IPP2 Exposure Adjust that was (or would "
        "have been) pushed via RCP2. Passing it to --exposureAdjust reproduces "
        "the on-set match at render time — no RMD sidecar required."
    ),
    "cautions": [
        "Use --exposureAdjust (IPP2 Exposure Adjust), NOT --exposure (a separate control).",
        "Do not bake these offsets into LUTs or CDLs — that is not RAW-domain matching.",
        "This batch applies EXPOSURE only. For white balance, use the RMD / REDCINE-X develop path.",
    ],
    "batching": (
        "redline_batch.sh is camera-wide: it groups clips by camera SUBFOLDER "
        "(media/GA, media/GB, …) under a media root, so each camera's single "
        "calibration EV is applied to EVERY clip that camera shot that day — not "
        "only the calibrated clip. It prompts for the REDLine binary (auto-found "
        "if possible), media root, output dir, and your REDLine flags, and offers "
        "a dry run first."
    ),
    "files": {
        "batch_script": "redline_batch.sh",
        "offsets_csv": "camera_offsets.csv",
        "clips_csv": "clips.csv (calibrated clips, reference only)",
    },
}


def _write_redline_batch(out_root: Path, cameras: list) -> dict:
    """Write camera_offsets.csv, clips.csv (populated), and a runnable redline_batch.sh."""
    # camera_offsets.csv — one EV per camera code
    offsets_path = out_root / "camera_offsets.csv"
    with offsets_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["camera", "exposure_adjust_ev"])
        seen = {}
        for r in cameras:
            code = r["camera_code"]
            seen.setdefault(code, r["exposureAdjust"])  # first clip wins per camera
        for code, ev in seen.items():
            w.writerow([code, f"{ev:.3f}"])

    # clips.csv — every source clip mapped to its camera code (fully populated)
    clips_path = out_root / "clips.csv"
    with clips_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["clip", "camera"])
        for r in cameras:
            w.writerow([r["source_path"], r["camera_code"]])

    # redline_batch.sh — interactive, camera-wide batch
    media_default = _media_root_default(cameras)
    script = _REDLINE_BATCH_TEMPLATE.replace("__MEDIA_ROOT__", media_default)
    script_path = out_root / "redline_batch.sh"
    script_path.write_text(script, encoding="utf-8")
    try:
        script_path.chmod(0o755)
    except OSError:
        pass

    return {"offsets": str(offsets_path), "clips": str(clips_path), "batch": str(script_path)}


def _media_root_default(cameras: list) -> str:
    """Infer the media root (the folder whose children are the camera subfolders)
    from the calibrated clip paths, so the script can offer it as a default."""
    import os
    roots = set()
    for r in cameras:
        parts = r["source_path"].split("/")
        code = (r["camera_code"] or "").upper()
        idx = next((i for i, p in enumerate(parts) if p.upper() == code), None)
        if idx and idx > 0:
            roots.add("/".join(parts[:idx]))
    return roots.pop() if len(roots) == 1 else ""


# Interactive camera-wide batch. Groups by camera SUBFOLDER under a media root,
# so a single calibration's per-camera EV applies to EVERY clip that camera shot
# that day (not just the calibrated clip). Prompts for REDLine path, media root,
# output dir, and post's REDLine flags. Exposure-only (--exposureAdjust direct).
_REDLINE_BATCH_TEMPLATE = r'''#!/usr/bin/env bash
# Auto-generated by R3DMatch — REDLine exposure-match batch (camera-wide).
#
# Each camera's measured IPP2 Exposure Adjust is applied to EVERY clip that
# camera shot, discovered by camera subfolder (media/GA, media/GB, ...) under a
# media root you choose. Exposure-only; REDLine Build 65.1.3+ --exposureAdjust
# (no RMD). Just run:  bash redline_batch.sh   — it prompts for everything.
set -uo pipefail
shopt -s nullglob

HERE="$(cd "$(dirname "$0")" && pwd)"
OFFSETS="$HERE/camera_offsets.csv"
DEFAULT_MEDIA_ROOT="__MEDIA_ROOT__"
DEFAULT_FLAGS="--colorSciVersion 3 --format 2"

ask() {  # ask "Prompt" "default" -> echoes answer on stdout
  local prompt="$1" def="${2:-}" ans
  if [[ -n "$def" ]]; then read -r -p "$prompt [$def]: " ans; echo "${ans:-$def}"
  else read -r -p "$prompt: " ans; echo "$ans"; fi
}

find_redline() {
  local c p g
  for c in REDline REDLine redline; do
    command -v "$c" >/dev/null 2>&1 && { command -v "$c"; return 0; }
  done
  local paths=(
    "/Applications/REDCINE-X PRO.app/Contents/MacOS/REDline"
    /Applications/REDCINE-X*PRO*/REDCINE*X*.app/Contents/MacOS/REDline
    /Applications/REDCINE*X*.app/Contents/MacOS/REDline
    /usr/local/bin/REDline /opt/red/bin/REDline
    /sww/tools/bundles/redline_*/REDline
  )
  for p in "${paths[@]}"; do
    for g in $p; do [[ -x "$g" ]] && { echo "$g"; return 0; }; done
  done
  return 1
}

echo "== R3DMatch — REDLine exposure-match batch (camera-wide) =="

# 1) REDLine binary (auto-find, else ask)
REDLINE_BIN="$(find_redline || true)"
if [[ -n "$REDLINE_BIN" ]]; then
  echo "Found REDLine: $REDLINE_BIN"
  if [[ "$(ask 'Use this REDLine? (y/n)' 'y')" =~ ^[Nn] ]]; then
    REDLINE_BIN="$(ask 'Full path to REDLine binary')"
  fi
else
  echo "REDLine not found automatically."
  REDLINE_BIN="$(ask 'Full path to REDLine binary')"
fi
[[ -x "$REDLINE_BIN" ]] || { echo "Not executable: $REDLINE_BIN" >&2; exit 1; }

# 2) Media root (any folder above the .RDC clips — foldered OR flat layout)
MEDIA_ROOT="$(ask 'Media root (folder above the day media)' "$DEFAULT_MEDIA_ROOT")"
[[ -d "$MEDIA_ROOT" ]] || { echo "Not a folder: $MEDIA_ROOT" >&2; exit 1; }

# 3) Output directory
OUTDIR="$(ask 'Output directory for renders')"
[[ -n "$OUTDIR" ]] || { echo "Output dir is required" >&2; exit 1; }
mkdir -p "$OUTDIR" || { echo "Cannot create $OUTDIR" >&2; exit 1; }

# 4) Post's REDLine flags (everything EXCEPT --i / --outDir / --exposureAdjust)
echo "Enter your REDLine flags. Do NOT include --i, --outDir, or --exposureAdjust"
echo "(the script manages those). Quotes are respected, e.g. --exportPreset \"My Preset\"."
FLAGS="$(ask 'REDLine flags' "$DEFAULT_FLAGS")"
eval "EXTRA=($FLAGS)" 2>/dev/null || EXTRA=($FLAGS)

# 5) Dry run?
DRY="$(ask 'Dry run first (print commands, render nothing)? (y/n)' 'y')"

echo
echo "REDLine : $REDLINE_BIN"
echo "Media   : $MEDIA_ROOT"
echo "Output  : $OUTDIR"
echo "Flags   : $FLAGS"
echo "Dry run : $DRY"
echo

# EV lookup by camera code (case-insensitive, CR-safe) — awk so it works on
# macOS /bin/bash 3.2 (no associative arrays).
get_offset() {
  awk -F, -v cam="$1" '
    NR>1 { c=$1; v=$2; sub(/\r$/,"",c); sub(/\r$/,"",v);
           if (toupper(c)==toupper(cam)) { print v; exit } }' "$OFFSETS"
}

# Camera code from a clip filename: reel-letter + position-letter.
#   G007_B106_0511C3_001.R3D -> reel "G007", pos "B106" -> "GB"
# Works whether or not clips are foldered by camera.
code_from_clip() {
  local stem reel rest pos
  stem="$(basename "$1")"; stem="${stem%.R3D}"; stem="${stem%.r3d}"
  stem="${stem%_[0-9][0-9][0-9]}"          # drop spanning segment _001
  reel="${stem%%_*}"; rest="${stem#*_}"; pos="${rest%%_*}"
  printf '%s%s' "${reel:0:1}" "${pos:0:1}"
}

# Discover every .RDC clip under the media root (recursive: handles
# media/<CAM>/RDM/RDC and the flat media/RDM/RDC layouts alike).
total=0; rendered=0; skipped=0
while IFS= read -r rdc; do
  seg=( "$rdc"/*_[0-9][0-9][0-9].R3D )
  [[ ${#seg[@]} -eq 0 ]] && seg=( "$rdc"/*.R3D )
  [[ ${#seg[@]} -eq 0 ]] && continue
  clip="${seg[0]}"
  code="$(code_from_clip "$clip")"
  ev="$(get_offset "$code")"; ev="${ev%$'\r'}"
  if [[ -z "$ev" ]]; then
    skipped=$((skipped+1))            # camera not in this calibration
    continue
  fi
  total=$((total+1))
  cmd=( "$REDLINE_BIN" --i "$clip" --outDir "$OUTDIR" "${EXTRA[@]}" --exposureAdjust "$ev" )
  if [[ "$DRY" =~ ^[Yy] ]]; then
    printf '   [%s EV %s] DRY:' "$code" "$ev"; printf ' %q' "${cmd[@]}"; echo
  else
    echo "   [$code EV $ev] render: $(basename "$clip")"
    if "${cmd[@]}"; then rendered=$((rendered+1)); else echo "   ! REDLine failed: $clip" >&2; fi
  fi
done < <(find "$MEDIA_ROOT" -type d -name '*.RDC' | sort)

echo
if [[ "$DRY" =~ ^[Yy] ]]; then
  echo "Dry run — $total clip(s) matched a calibrated camera; $skipped clip(s) skipped (no match). Nothing rendered."
  echo "Re-run and answer 'n' to the dry-run prompt to render for real."
else
  echo "Done — $rendered/$total matched clip(s) rendered to $OUTDIR ($skipped skipped, no calibrated camera)."
fi
echo "Tip: validate one clip with --printMeta 2 before a full-day batch."
'''


def write_match_export(out_root: Path, run: "RunResult") -> dict:
    """Write match_export.json + redcinex_develop.csv. Returns paths written."""
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    cameras = []
    for cr in run.cameras:
        if not cr.commit:
            continue
        c = cr.commit
        meta = cr.metadata
        row = {
            "clip_id": cr.clip_id,
            "camera_label": cr.camera_label,
            "camera_code": _cam_code(cr),
            "source_path": cr.source_path,
            "exposureAdjust": round(c.exposure_adjust, 6),   # stops
            "derivation_method": c.derivation_method,
            "exposure_only": bool(c.exposure_only),
            "match_pct": cr.match_pct,
            "as_shot": {
                "kelvin": getattr(meta, "kelvin", None),
                "tint": getattr(meta, "tint", None),
                "iso": getattr(meta, "iso", None),
            },
            "rcp2_params": _rcp2_params(c),
        }
        if not c.exposure_only:
            row["kelvin"] = c.kelvin       # absolute target Kelvin
            row["tint"] = round(c.tint, 2)  # absolute target tint
        cameras.append(row)

    payload = {
        "schema_version": "r3dmatch_match_export_v1",
        "note": "Network-down fallback. Apply these per-clip to reproduce the "
                "on-set match without RCP2. rcp2_params carries the raw/divider "
                "integer form the camera uses.",
        "run_id": run.run_id,
        "created_at": run.created_at,
        "input_path": run.input_path,
        "exposure_only": bool(run.exposure_only),
        "anchor_ire": run.anchor_ire,
        "anchor_domain": "scene-linear" if any(
            getattr(cr.measurement, "hero_log2_lin", None) is not None
            for cr in run.cameras if cr.measurement) else "display",
        "shared_kelvin": run.wb_solve.shared_kelvin if run.wb_solve else None,
        "wb_status": run.wb_solve.status if run.wb_solve else "SKIPPED",
        "array_match_pct": run.array_match_pct,
        "redline": _REDLINE_RECIPE,
        "cameras": cameras,
    }
    json_path = out_root / "match_export.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # --- REDCINE-X develop-panel CSV -------------------------------------------
    csv_path = out_root / "redcinex_develop.csv"
    eo = bool(run.exposure_only)
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow([f"# R3DMatch develop settings for REDCINE-X — run {run.run_id} ({run.created_at})"])
        w.writerow(["# Set each clip's REDCINE-X develop panel to these values to reproduce the on-set match."])
        if eo:
            w.writerow(["# EXPOSURE-ONLY run: white balance unchanged — leave Color Temp / Tint at as-shot."])
        w.writerow(["# Exposure = stops (REDCINE-X 'Exposure'). Color Temp = K. Tint = REDCINE-X tint units."])
        header = ["Clip", "Camera", "Exposure (stops)"]
        if not eo:
            header += ["Color Temp (K)", "Tint"]
        header += ["As-shot K", "As-shot Tint", "Match %", "Source clip"]
        w.writerow(header)
        for r in cameras:
            line = [r["clip_id"], r["camera_label"], f'{r["exposureAdjust"]:.3f}']
            if not eo:
                line += [r.get("kelvin", ""), r.get("tint", "")]
            mp = r["match_pct"]
            line += [
                r["as_shot"]["kelvin"] if r["as_shot"]["kelvin"] is not None else "",
                r["as_shot"]["tint"] if r["as_shot"]["tint"] is not None else "",
                f"{mp:.0f}" if mp is not None else "",
                r["source_path"],
            ]
            w.writerow(line)

    # --- REDLine exposure-match batch (offsets CSV + clips CSV + runnable script) ---
    redline = _write_redline_batch(out_root, cameras)

    return {
        "json": str(json_path),
        "csv": str(csv_path),
        "camera_count": len(cameras),
        **redline,
    }
