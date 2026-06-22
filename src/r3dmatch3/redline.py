"""
R3DMatch v2 — REDLine Interface

All REDLine subprocess calls are here. No SDK. No fallbacks to other tools.

REDLine render produces IPP2 / BT.709 / BT.1886 / Medium Contrast / Medium Rolloff.
REDLine --printMeta 1 provides clip-level metadata (Kelvin, tint, ISO, camera ID).
REDLine --printMeta 5 provides per-frame lens data (optional, merged if non-zero).

Important: REDLine appends ".000000.tif" to the output filename regardless of
the extension you specify. Always use _resolve_output_path() to find the actual file.
"""
from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .models import ClipMetadata
from .colorpipeline import ColorPipeline, REFERENCE_PIPELINE

log = logging.getLogger(__name__)


class REDLineNotFoundError(RuntimeError):
    """Raised when the REDLine executable cannot be located."""

# ---------------------------------------------------------------------------
# REDLine render constants — IPP2 Medium / Medium
# ---------------------------------------------------------------------------

_COLOR_SPACE_BT709 = "13"
_GAMMA_BT1886 = "32"
_OUTPUT_TONEMAP_MEDIUM = "1"
_ROLLOFF_MEDIUM = "3"
_SHADOW_MEDIUM = "0.0"
_COLOR_SCI_IPP2 = "3"


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def resolve_redline_executable(settings_path: str = "") -> str:
    """
    Find REDLine in order of preference:
    1. settings_path argument (from app Settings tab)
    2. REDLINE_PATH environment variable
    3. config/redline.json (project config)
    4. ~/.r3dmatch_config.json (user config)
    5. Known macOS install locations (REDCINE-X PRO)
    6. PATH via shutil.which
    Raises REDLineNotFoundError if not found.
    """
    import shutil

    checked_paths: List[str] = []

    # Known macOS install locations — ordered by likelihood
    REDLINE_CANDIDATES = [
        "/Applications/REDCINE-X PRO.app/Contents/MacOS/REDline",
        "/Applications/REDCINE-X Professional/REDCINE-X PRO.app/Contents/MacOS/REDline",
        "/Applications/REDCINE-X PRO/REDline",
        "/Applications/REDline.app/Contents/MacOS/REDline",
        "/usr/local/bin/REDline",
        "/opt/homebrew/bin/REDline",
    ]

    # Settings override (from app Settings tab — highest priority)
    if settings_path and settings_path.strip():
        checked_paths.append(settings_path.strip())
        if _is_redline_executable(Path(settings_path.strip())):
            return settings_path.strip()
        raise REDLineNotFoundError(_redline_not_found_message(checked_paths))

    env_override = os.environ.get("REDLINE_PATH", "").strip()
    if env_override:
        checked_paths.append(env_override)
        if _is_redline_executable(Path(env_override)):
            return env_override
        raise REDLineNotFoundError(_redline_not_found_message(checked_paths))

    # Project config
    config_path = Path(__file__).resolve().parents[3] / "config" / "redline.json"
    if config_path.exists():
        try:
            payload = json.loads(config_path.read_text(encoding="utf-8"))
            for key in ("redline_executable", "red_runtime_path", "redline_path"):
                candidate = str(payload.get(key) or "").strip()
                if candidate:
                    checked_paths.append(candidate)
                    if _is_redline_executable(Path(candidate)):
                        return candidate
        except Exception:
            pass

    # User config
    user_config = Path.home() / ".r3dmatch_config.json"
    if user_config.exists():
        try:
            payload = json.loads(user_config.read_text(encoding="utf-8"))
            for key in ("redline_path", "red_runtime_path"):
                candidate = str(payload.get(key) or "").strip()
                if candidate:
                    checked_paths.append(candidate)
                    if _is_redline_executable(Path(candidate)):
                        return candidate
        except Exception:
            pass

    # Known install locations
    for candidate in REDLINE_CANDIDATES:
        checked_paths.append(candidate)
        if _is_redline_executable(Path(candidate)):
            return candidate

    # PATH
    found = shutil.which("REDline") or shutil.which("REDLine")
    if found:
        return found
    checked_paths.append("PATH lookup: REDline / REDLine")

    raise REDLineNotFoundError(_redline_not_found_message(checked_paths))


def _is_redline_executable(path: Path) -> bool:
    return path.exists() and path.is_file() and os.access(path, os.X_OK)


def _redline_not_found_message(checked_paths: List[str]) -> str:
    checked = "\n".join(f"  - {path}" for path in checked_paths) or "  - <none>"
    return (
        "REDLine executable not found.\n"
        "Checked paths:\n"
        f"{checked}\n"
        "Install REDCINE-X PRO or set the REDLINE_PATH environment variable."
    )


def check_redline_available(redline: str) -> Dict[str, object]:
    """Quick check that REDLine responds to --help. Returns status dict."""
    try:
        result = subprocess.run(
            [redline, "--help"],
            capture_output=True, text=True, timeout=15
        )
        output = (result.stdout or "") + (result.stderr or "")
        build_match = re.search(r"REDline Build\s+([\d.]+)", output)
        sdk_match = re.search(r"R3DSDK version\s+R3DAPI\s+([\S]+)", output)
        return {
            "ready": "printMeta" in output,
            "build": build_match.group(1) if build_match else "",
            "sdk_version": sdk_match.group(1) if sdk_match else "",
            "error": "" if "printMeta" in output else "REDLine help output did not contain expected flags",
        }
    except Exception as exc:
        return {"ready": False, "build": "", "sdk_version": "", "error": str(exc)}


# ---------------------------------------------------------------------------
# Metadata reading (--printMeta 1 and --printMeta 5)
# ---------------------------------------------------------------------------

def read_clip_metadata(
    r3d_path: str,
    *,
    redline: str,
    read_lens: bool = True,
) -> ClipMetadata:
    """
    Read clip metadata via --printMeta 1, optionally merge lens data from --printMeta 5.
    """
    meta1 = _run_print_meta(r3d_path, mode=1, redline=redline)
    parsed = _parse_meta1(meta1, r3d_path)

    if read_lens:
        try:
            meta5 = _run_print_meta(r3d_path, mode=5, redline=redline)
            lens = _parse_meta5_first_frame(meta5)
            parsed.focal_length_mm = lens.get("focal_length_mm")
            parsed.aperture = lens.get("aperture")
            parsed.focus_distance_mm = lens.get("focus_distance_mm")
        except Exception:
            pass  # lens metadata is optional — failure is silent

    return parsed


def _run_print_meta(r3d_path: str, *, mode: int, redline: str, timeout: int = 30) -> str:
    """Run REDLine --printMeta <mode> and return stdout+stderr."""
    result = subprocess.run(
        [redline, "--i", str(r3d_path), "--printMeta", str(mode)],
        capture_output=True, text=True, timeout=timeout,
    )
    return (result.stdout or "") + (result.stderr or "")


def _parse_meta1(output: str, source_path: str) -> ClipMetadata:
    """Parse --printMeta 1 output into a ClipMetadata."""
    kv: Dict[str, str] = {}
    for line in output.splitlines():
        if ":\t" in line:
            key, _, value = line.partition(":\t")
            kv[key.strip()] = value.strip()
        elif line.startswith("[") or not line.strip():
            continue

    def get(key: str, default: str = "") -> str:
        return kv.get(key, default).strip()

    clip_id = get("ReelID") + "_001"  # e.g. G007_A106_0511R9_001
    reel = get("Reel")
    clip_num = get("Clip")
    camera_position = get("Camera Position")
    camera_label = f"{get('CamReelID')}_{camera_position}{clip_num}"  # G007_A106

    try:
        kelvin = int(float(get("Kelvin", "5600")))
    except ValueError:
        kelvin = 5600

    try:
        tint = float(get("Tint", "0"))
    except ValueError:
        tint = 0.0

    try:
        iso = int(float(get("ISO", "800")))
    except ValueError:
        iso = 800

    try:
        fps = float(get("Record FPS", get("FPS", "24")))
    except ValueError:
        fps = 24.0

    try:
        fw = int(get("Frame Width", "0"))
        fh = int(get("Frame Height", "0"))
    except ValueError:
        fw, fh = 0, 0

    try:
        total_frames = int(get("Total Frames", "1"))
    except ValueError:
        total_frames = 1

    try:
        color_space = int(get("Color Space", "25"))
    except ValueError:
        color_space = 25

    try:
        gamma_space = int(get("Gamma Space", "34"))
    except ValueError:
        gamma_space = 34

    return ClipMetadata(
        clip_id=clip_id,
        source_path=str(Path(source_path).expanduser().resolve()),
        camera_model=get("Camera Model"),
        camera_pin=get("Camera PIN"),
        camera_position=camera_position,
        camera_label=camera_label,
        reel=reel,
        clip_num=clip_num,
        kelvin=kelvin,
        tint=tint,
        iso=iso,
        fps=fps,
        frame_width=fw,
        frame_height=fh,
        total_frames=total_frames,
        timecode=get("Abs TC"),
        color_space=color_space,
        gamma_space=gamma_space,
        image_pipeline=get("Clip Current Image Pipeline", "IPP2"),
        lens_name=get("Lens") or None,
    )


def _parse_meta5_first_frame(output: str) -> Dict[str, Optional[int]]:
    """
    Parse --printMeta 5 CSV output and return lens fields from frame 0.
    Returns zeros if no lens data is present.

    Mode 5 columns:
    FrameNo,Timecode,Timestamp,Aperture,Focus Distance,Focal Length,
    Acceleration X,Acceleration Y,Acceleration Z,Rotation X,Rotation Y,Rotation Z,
    Cooke Metadata
    """
    result: Dict[str, Optional[int]] = {
        "focal_length_mm": None,
        "aperture": None,
        "focus_distance_mm": None,
    }
    lines = [l.strip() for l in output.splitlines() if l.strip() and not l.startswith("[")]
    if len(lines) < 2:
        return result
    header = [h.strip().lower() for h in lines[0].split(",")]
    data = [v.strip() for v in lines[1].split(",")]
    if len(data) < len(header):
        return result

    row = dict(zip(header, data))

    try:
        fl = float(row.get("focal length", "0") or "0")
        if fl > 0:
            result["focal_length_mm"] = int(round(fl))
    except ValueError:
        pass

    try:
        ap = float(row.get("aperture", "0") or "0")
        if ap > 0:
            result["aperture"] = ap
    except ValueError:
        pass

    try:
        fd = float(row.get("focus distance", "0") or "0")
        if fd > 0:
            result["focus_distance_mm"] = int(round(fd))
    except ValueError:
        pass

    return result


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_measurement_frame(
    r3d_path: str,
    output_path: str,
    *,
    redline: str,
    frame_index: int = 0,
    use_as_shot: bool = True,
    exposure_adjust: Optional[float] = None,
    kelvin: Optional[int] = None,
    tint: Optional[float] = None,
    pipeline: Optional[ColorPipeline] = None,
    timeout: int = 120,
) -> Dict[str, object]:
    """
    Render one frame from R3D via REDLine into a TIFF.

    pipeline: ColorPipeline describing the output color science. Defaults to
    REFERENCE_PIPELINE, which emits exactly the v3 flags (IPP2/BT.709/BT.1886/
    Medium/Medium) in the same order — a reference render is byte-identical to v3.
    Pass a project delivery pipeline to render through a different transform / LUT.

    Returns a dict with:
      ok: bool
      output_path: str (actual path, resolved via glob if needed)
      returncode: int
      stderr: str
      command: List[str]
    """
    color_pipeline = pipeline or REFERENCE_PIPELINE
    cmd = [
        redline,
        "--i", str(Path(r3d_path).expanduser().resolve()),
        "--o", str(output_path),
        "--format", "1",           # TIFF
        "--start", str(frame_index),
        "--frameCount", "1",
        *color_pipeline.to_redline_color_args(),
        "--silent",
    ]

    if use_as_shot:
        cmd.append("--useMeta")

    if exposure_adjust is not None:
        cmd += ["--exposureAdjust", f"{float(exposure_adjust):.6f}"]
    if kelvin is not None:
        cmd += ["--kelvin", str(int(kelvin))]
    if tint is not None:
        cmd += ["--tint", f"{float(tint):.2f}"]

    clip_id = Path(r3d_path).stem
    proc = None
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        stdout, stderr = proc.communicate(timeout=timeout)
        if proc.returncode != 0:
            raise subprocess.CalledProcessError(
                proc.returncode,
                cmd,
                output=stdout,
                stderr=stderr,
            )
        actual = _resolve_output_path(output_path)
        ok = actual.exists() and actual.stat().st_size > 10_000
        return {
            "ok": ok,
            "output_path": str(actual),
            "returncode": proc.returncode,
            "stderr": (stderr or "")[:500],
            "status": "OK" if ok else "RENDER_ERROR",
            "command": cmd,
        }
    except subprocess.TimeoutExpired as exc:
        if proc is not None:
            proc.kill()
            _, stderr = proc.communicate()
        else:
            stderr = ""
        log.error("REDLine render timed out for clip %s after %ss", clip_id, timeout)
        return {
            "ok": False,
            "output_path": str(output_path),
            "returncode": -1,
            "stderr": (f"REDLine timed out after {timeout}s. "
                       f"{(stderr or str(exc))[:400]}").strip(),
            "status": "RENDER_TIMEOUT",
            "command": cmd,
        }
    except subprocess.CalledProcessError as exc:
        log.error("REDLine render failed for clip %s with returncode %s", clip_id, exc.returncode)
        return {
            "ok": False,
            "output_path": str(output_path),
            "returncode": exc.returncode,
            "stderr": ((exc.stderr or exc.output or "")[:500]),
            "status": "RENDER_ERROR",
            "command": cmd,
        }
    except Exception as exc:
        log.exception("REDLine render crashed for clip %s", clip_id)
        return {
            "ok": False,
            "output_path": str(output_path),
            "returncode": -1,
            "stderr": str(exc),
            "status": "RENDER_ERROR",
            "command": cmd,
        }


def _resolve_output_path(requested: str) -> Path:
    """
    REDLine appends ".000000.tif" to output paths.
    Given the requested output path, find the actual written file.
    """
    candidate = Path(requested).expanduser().resolve()
    if candidate.exists():
        return candidate
    # REDLine appends .000000.tif
    matches = sorted(candidate.parent.glob(f"{candidate.name}.*"))
    return matches[0] if matches else candidate


# ---------------------------------------------------------------------------
# Lens display formatting
# ---------------------------------------------------------------------------

def format_focus_distance(mm: int) -> Optional[str]:
    """Convert mm to feet+inches string. Returns None if zero."""
    if mm <= 0:
        return None
    total_inches = mm / 25.4
    feet = int(total_inches // 12)
    inches = round(total_inches % 12)
    if inches == 12:
        feet += 1
        inches = 0
    if feet > 0 and inches > 0:
        return f"{feet}ft {inches}in"
    elif feet > 0:
        return f"{feet}ft"
    else:
        return f"{inches}in"


def format_aperture(fstop: float) -> str:
    """Format f-stop as ƒ5.6 etc."""
    if fstop <= 0:
        return ""
    if abs(fstop - round(fstop)) < 0.05:
        return f"f/{int(round(fstop))}"
    return f"f/{fstop:.1f}"


def format_lens_line(meta: ClipMetadata) -> Optional[str]:
    """
    Build the lens display line for the contact sheet card.
    Returns None if no lens data is present (omit the row entirely).
    """
    has_values = meta.has_lens_data()
    has_name = bool(meta.lens_name and meta.lens_name.strip())

    if not has_values and not has_name:
        return None

    parts: List[str] = []
    if has_name:
        parts.append(meta.lens_name)
    if meta.focal_length_mm and meta.focal_length_mm > 0:
        parts.append(f"{meta.focal_length_mm}mm")
    if meta.aperture and meta.aperture > 0:
        parts.append(format_aperture(meta.aperture))
    if meta.focus_distance_mm and meta.focus_distance_mm > 0:
        fd = format_focus_distance(meta.focus_distance_mm)
        if fd:
            parts.append(f"Focus: {fd}")

    return "  ".join(parts) if parts else None
