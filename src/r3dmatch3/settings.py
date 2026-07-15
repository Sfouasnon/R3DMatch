"""
settings.py — R3DMatch v2 persistent application settings.

Stores user preferences to:
  ~/Library/Application Support/R3DMatch_v2/settings.json

Public interface:
    from r3dmatch3.settings import load_settings, save_settings

    settings = load_settings()
    settings["redline_path"] = "/path/to/REDline"
    save_settings(settings)

Keys:
    redline_path   str | ""   — operator-specified REDLine binary path
                                 Empty string means "use auto-discovery"
    default_out_dir str | ""  — default output folder for new runs
                                 Empty string means "derive from input path"

    Capture / FTP ingest (Capture tab). Credentials are NEVER assumed — an empty
    value means "not configured" and the Capture tab blocks the pull step until
    the operator fills them in:
    ftp_user       str | ""   — FTP username on the camera bodies
    ftp_pass       str | ""   — FTP password
    ftp_port       str | "21" — FTP control port (stored as string)
    capture_dest   str | ""   — default local destination for pulled clips
    capture_cidr   str | ""   — preferred CIDR to scan for cameras (optional;
                                 blank falls back to auto interface discovery)
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict

log = logging.getLogger(__name__)

_SETTINGS_DIR  = Path.home() / "Library" / "Application Support" / "R3DMatch_v2"
_SETTINGS_PATH = _SETTINGS_DIR / "settings.json"

_DEFAULTS: Dict[str, str] = {
    "redline_path":    "",
    "default_out_dir": "",
    # Capture / FTP ingest — blank means "not configured" (never assumed).
    "ftp_user":        "",
    "ftp_pass":        "",
    "ftp_port":        "21",
    "capture_dest":    "",
    "capture_cidr":    "",
}


def load_settings() -> Dict[str, str]:
    """
    Load settings from disk. Returns defaults for any missing keys.
    Never raises — returns defaults on any read/parse error.
    """
    settings = dict(_DEFAULTS)
    if not _SETTINGS_PATH.exists():
        return settings
    try:
        raw = json.loads(_SETTINGS_PATH.read_text(encoding="utf-8"))
        for key in _DEFAULTS:
            if key in raw and isinstance(raw[key], str):
                settings[key] = raw[key]
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("Could not read settings file %s: %s", _SETTINGS_PATH, exc)
    return settings


def save_settings(settings: Dict[str, str]) -> None:
    """
    Write settings to disk atomically. Never raises.
    """
    try:
        os.makedirs(_SETTINGS_DIR, exist_ok=True)
        tmp = _SETTINGS_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(settings, indent=2), encoding="utf-8")
        os.replace(tmp, _SETTINGS_PATH)
    except OSError as exc:
        log.warning("Could not save settings to %s: %s", _SETTINGS_PATH, exc)
