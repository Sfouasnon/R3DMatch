from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

from .app_resources import bundled_red_redistributable_dir
from .report import contact_sheet_pdf_export_preflight
from .report import REDLINE_CONFIG_PATH
from .report import resolve_redline_tool_status
from .sdk import load_configured_red_native_module, red_sdk_configuration_error, resolve_red_sdk_configuration

DESKTOP_CONFIG_ENV = "R3DMATCH_USER_CONFIG_PATH"
DESKTOP_REDLINE_SOURCE_ENV = "R3DMATCH_DESKTOP_REDLINE_SOURCE"
DESKTOP_REDLINE_KEY = "redline_path"


def desktop_config_path() -> Path:
    override = str(os.environ.get(DESKTOP_CONFIG_ENV) or "").strip()
    if override:
        return Path(override).expanduser()
    return Path.home() / ".r3dmatch_config.json"


def load_config() -> Dict[str, str]:
    path = desktop_config_path()
    try:
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                return {str(key): str(value) for key, value in payload.items() if str(value or "").strip()}
    except Exception:
        pass
    return {}


def save_config(cfg: Dict[str, str]) -> None:
    path = desktop_config_path()
    sanitized = {str(key): str(value) for key, value in cfg.items() if str(value or "").strip()}
    path.parent.mkdir(parents=True, exist_ok=True)
    if sanitized:
        path.write_text(json.dumps(sanitized, indent=2), encoding="utf-8")
    else:
        path.unlink(missing_ok=True)


def _desktop_configured_redline_path() -> str:
    return str(load_config().get(DESKTOP_REDLINE_KEY) or "").strip()


def _autodetect_redline_installation() -> str:
    candidates = [
        "/Applications/REDCINE-X Professional/REDCINE-X PRO.app/Contents/MacOS/REDline",
        "/Applications/REDCINE-X PRO.app/Contents/MacOS/REDline",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return str(path.resolve())
    return ""


def _restore_redline_environment(previous_env: Dict[str, str]) -> None:
    for key, value in previous_env.items():
        if value:
            os.environ[key] = value
        else:
            os.environ.pop(key, None)


def _apply_persisted_redline_environment() -> Dict[str, str]:
    configured = str(os.environ.get("R3DMATCH_REDLINE_EXECUTABLE") or "").strip()
    if configured:
        os.environ.setdefault("REDLINE_PATH", configured)
        return {"path": configured, "source": "environment"}

    redline_env = str(os.environ.get("REDLINE_PATH") or "").strip()
    if redline_env:
        os.environ["R3DMATCH_REDLINE_EXECUTABLE"] = redline_env
        os.environ[DESKTOP_REDLINE_SOURCE_ENV] = "redline_path_env"
        return {"path": redline_env, "source": "redline_path_env"}

    persisted = _desktop_configured_redline_path()
    if persisted:
        os.environ["REDLINE_PATH"] = persisted
        os.environ["R3DMATCH_REDLINE_EXECUTABLE"] = persisted
        os.environ[DESKTOP_REDLINE_SOURCE_ENV] = "user_config"
        return {"path": persisted, "source": "user_config"}

    autodetected = _autodetect_redline_installation()
    if autodetected:
        os.environ["REDLINE_PATH"] = autodetected
        os.environ["R3DMATCH_REDLINE_EXECUTABLE"] = autodetected
        os.environ[DESKTOP_REDLINE_SOURCE_ENV] = "auto_detected"
        return {"path": autodetected, "source": "auto_detected"}
    return {"path": "", "source": "unset"}


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
    _apply_persisted_redline_environment()
    recommendation = recommended_dyld_fallback_library_path()
    if recommendation["source"] == "auto_homebrew" and recommendation["value"]:
        os.environ.setdefault("DYLD_FALLBACK_LIBRARY_PATH", recommendation["value"])
        recommendation["value"] = str(os.environ.get("DYLD_FALLBACK_LIBRARY_PATH") or "")
    return recommendation


def runtime_cli_prefix() -> List[str]:
    executable = str(sys.executable or "python3")
    if getattr(sys, "frozen", False):
        return [executable, "--cli"]
    return [executable, "-m", "r3dmatch.cli"]


def runtime_subprocess_env(repo_root: str | Path, *, invocation_source: Optional[str] = None) -> Dict[str, str]:
    env = os.environ.copy()
    if not getattr(sys, "frozen", False):
        existing = str(env.get("PYTHONPATH") or "").strip()
        source_path = str(Path(repo_root).expanduser().resolve() / "src")
        env["PYTHONPATH"] = source_path if not existing else f"{source_path}:{existing}"
    if invocation_source:
        env["R3DMATCH_INVOCATION_SOURCE"] = invocation_source
    return env


def read_redline_config() -> Dict[str, str]:
    try:
        if REDLINE_CONFIG_PATH.exists():
            payload = json.loads(REDLINE_CONFIG_PATH.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                return {str(key): str(value) for key, value in payload.items()}
    except Exception:
        pass
    return {}


def write_redline_config(data: Dict[str, str]) -> None:
    sanitized = {str(key): str(value) for key, value in data.items() if str(value or "").strip()}
    REDLINE_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if sanitized:
        REDLINE_CONFIG_PATH.write_text(json.dumps(sanitized, indent=2), encoding="utf-8")
    else:
        REDLINE_CONFIG_PATH.unlink(missing_ok=True)


def redline_configured_path() -> str:
    persisted = _desktop_configured_redline_path()
    if persisted:
        return persisted
    payload = read_redline_config()
    return str(payload.get("redline_executable") or payload.get("redline_path") or "").strip()


def persist_redline_configured_path(candidate_path: str) -> Dict[str, object]:
    configured_path = str(candidate_path or "").strip()
    previous_desktop_config = dict(load_config())
    previous_payload = dict(read_redline_config())
    previous_env = {
        "REDLINE_PATH": str(os.environ.get("REDLINE_PATH") or ""),
        "R3DMATCH_REDLINE_EXECUTABLE": str(os.environ.get("R3DMATCH_REDLINE_EXECUTABLE") or ""),
        DESKTOP_REDLINE_SOURCE_ENV: str(os.environ.get(DESKTOP_REDLINE_SOURCE_ENV) or ""),
    }
    next_payload: Dict[str, str] = {}
    next_desktop_config = dict(previous_desktop_config)
    if configured_path:
        next_payload["redline_executable"] = configured_path
        next_desktop_config[DESKTOP_REDLINE_KEY] = configured_path
    else:
        next_desktop_config.pop(DESKTOP_REDLINE_KEY, None)
    save_config(next_desktop_config)
    project_config_error = ""
    try:
        write_redline_config(next_payload)
    except Exception as exc:
        project_config_error = str(exc)

    if configured_path:
        os.environ["REDLINE_PATH"] = configured_path
        os.environ["R3DMATCH_REDLINE_EXECUTABLE"] = configured_path
        os.environ[DESKTOP_REDLINE_SOURCE_ENV] = "user_config"
    elif previous_env.get(DESKTOP_REDLINE_SOURCE_ENV) in {"user_config", "auto_detected", "redline_path_env"}:
        os.environ.pop("REDLINE_PATH", None)
        os.environ.pop("R3DMATCH_REDLINE_EXECUTABLE", None)
        os.environ.pop(DESKTOP_REDLINE_SOURCE_ENV, None)

    status = dict(resolve_redline_tool_status())
    if configured_path and not bool(status.get("ready")):
        save_config(previous_desktop_config)
        try:
            write_redline_config(previous_payload)
        except Exception:
            pass
        _restore_redline_environment(previous_env)
        return {
            "ok": False,
            "status": dict(resolve_redline_tool_status()),
            "message": "",
            "error": str(status.get("error") or "REDLine is not usable with the selected path."),
        }
    if not configured_path:
        message = "Cleared saved REDLine path. Resolver will now use PATH/environment discovery."
    elif str(status.get("source") or "") == "environment":
        message = (
            "Saved REDLine executable, but the active runtime is still using "
            "R3DMATCH_REDLINE_EXECUTABLE from the environment."
        )
    else:
        message = f"Saved REDLine executable: {status.get('resolved_path') or configured_path}"
    if project_config_error:
        message = f"{message} Project config mirror warning: {project_config_error}"
    return {"ok": True, "status": status, "message": message, "error": ""}


def red_sdk_runtime_health_payload() -> Dict[str, object]:
    try:
        sdk_config = resolve_red_sdk_configuration()
        if sdk_config.errors:
            raise RuntimeError(red_sdk_configuration_error(sdk_config))
        load_configured_red_native_module()
        bundled_dir = bundled_red_redistributable_dir()
        return {
            "ready": True,
            "error": "",
            "redistributable_dir": str(sdk_config.redistributable_dir or ""),
            "source": str(sdk_config.redistributable_source or "missing"),
            "root": str(sdk_config.root or ""),
            "bundled_dir": str(bundled_dir or ""),
        }
    except Exception as exc:
        bundled_dir = bundled_red_redistributable_dir()
        return {
            "ready": False,
            "error": str(exc),
            "redistributable_dir": "",
            "source": "bundled_app" if bundled_dir else "missing",
            "root": str(os.environ.get("RED_SDK_ROOT") or ""),
            "bundled_dir": str(bundled_dir or ""),
        }


def runtime_health_payload(html_path: Optional[str] = None) -> Dict[str, object]:
    dyld = ensure_runtime_environment()
    preflight = contact_sheet_pdf_export_preflight(html_path)
    active_venv = str(os.environ.get("VIRTUAL_ENV") or "")
    if not active_venv and sys.prefix != getattr(sys, "base_prefix", sys.prefix):
        active_venv = sys.prefix
    red_sdk_runtime = red_sdk_runtime_health_payload()
    redline_tool = dict(resolve_redline_tool_status())
    redline_tool["configured_path"] = redline_configured_path()
    redline_tool["configured_config_path"] = str(REDLINE_CONFIG_PATH)
    redline_tool["desktop_config_path"] = str(desktop_config_path())
    redline_tool["desktop_config_source"] = str(os.environ.get(DESKTOP_REDLINE_SOURCE_ENV) or "")
    asset_validation = dict(preflight.get("asset_validation") or {})
    html_pdf_ready = bool(preflight.get("weasyprint_importable")) and (
        not asset_validation or bool(asset_validation.get("all_assets_exist"))
    )
    return {
        "interpreter": sys.executable,
        "frozen_app": bool(getattr(sys, "frozen", False)),
        "virtual_env": active_venv,
        "dyld_fallback_library_path": str(os.environ.get("DYLD_FALLBACK_LIBRARY_PATH") or ""),
        "dyld_fallback_source": dyld["source"],
        "red_sdk_root": str(os.environ.get("RED_SDK_ROOT") or ""),
        "red_sdk_redistributable_dir": str(os.environ.get("RED_SDK_REDISTRIBUTABLE_DIR") or ""),
        "resolved_red_sdk_redistributable_dir": str(red_sdk_runtime.get("redistributable_dir") or ""),
        "red_backend_ready": bool(red_sdk_runtime.get("ready")),
        "red_backend_error": str(red_sdk_runtime.get("error") or ""),
        "red_sdk_runtime": red_sdk_runtime,
        "redline_tool": redline_tool,
        "weasyprint_importable": bool(preflight.get("weasyprint_importable")),
        "weasyprint_error": str(preflight.get("weasyprint_error") or ""),
        "asset_validation": asset_validation or None,
        "html_pdf_ready": html_pdf_ready,
    }
