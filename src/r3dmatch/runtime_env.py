from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, Optional

from .report import contact_sheet_pdf_export_preflight
from .sdk import red_sdk_configuration_error, resolve_red_sdk_configuration


def recommended_dyld_fallback_library_path() -> Dict[str, str]:
    current = str(os.environ.get("DYLD_FALLBACK_LIBRARY_PATH") or "").strip()
    if current:
        return {"value": current, "source": "environment"}
    if sys.platform == "darwin":
        homebrew_lib = Path("/opt/homebrew/lib")
        if homebrew_lib.exists():
            return {"value": str(homebrew_lib), "source": "auto_homebrew"}
    return {"value": "", "source": "unset"}


def ensure_runtime_environment() -> Dict[str, str]:
    recommendation = recommended_dyld_fallback_library_path()
    if recommendation["source"] == "auto_homebrew" and recommendation["value"]:
        os.environ.setdefault("DYLD_FALLBACK_LIBRARY_PATH", recommendation["value"])
        recommendation["value"] = str(os.environ.get("DYLD_FALLBACK_LIBRARY_PATH") or "")
    return recommendation


def runtime_health_payload(html_path: Optional[str] = None) -> Dict[str, object]:
    dyld = ensure_runtime_environment()
    preflight = contact_sheet_pdf_export_preflight(html_path)
    active_venv = str(os.environ.get("VIRTUAL_ENV") or "")
    if not active_venv and sys.prefix != getattr(sys, "base_prefix", sys.prefix):
        active_venv = sys.prefix
    try:
        sdk_config = resolve_red_sdk_configuration()
        if sdk_config.errors:
            raise RuntimeError(red_sdk_configuration_error(sdk_config))
        red_backend_ready = True
        red_backend_error = ""
        resolved_red_redistributable = str(sdk_config.redistributable_dir)
    except Exception as exc:
        red_backend_ready = False
        red_backend_error = str(exc)
        resolved_red_redistributable = ""
    asset_validation = dict(preflight.get("asset_validation") or {})
    html_pdf_ready = bool(preflight.get("weasyprint_importable")) and (
        not asset_validation or bool(asset_validation.get("all_assets_exist"))
    )
    return {
        "interpreter": sys.executable,
        "virtual_env": active_venv,
        "dyld_fallback_library_path": str(os.environ.get("DYLD_FALLBACK_LIBRARY_PATH") or ""),
        "dyld_fallback_source": dyld["source"],
        "red_sdk_root": str(os.environ.get("RED_SDK_ROOT") or ""),
        "red_sdk_redistributable_dir": str(os.environ.get("RED_SDK_REDISTRIBUTABLE_DIR") or ""),
        "resolved_red_sdk_redistributable_dir": resolved_red_redistributable,
        "red_backend_ready": red_backend_ready,
        "red_backend_error": red_backend_error,
        "weasyprint_importable": bool(preflight.get("weasyprint_importable")),
        "weasyprint_error": str(preflight.get("weasyprint_error") or ""),
        "asset_validation": asset_validation or None,
        "html_pdf_ready": html_pdf_ready,
    }
