from __future__ import annotations

import ctypes
import json
import os
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Protocol, Sequence

from .commit_values import KELVIN_MAX, KELVIN_MIN, TINT_MAX, TINT_MIN
from .ftps_ingest import DEFAULT_CAMERA_IP_MAP
from .identity import inventory_camera_label_from_clip_id, inventory_camera_label_from_source_path
from .rcp2_websocket import (
    JsonWebSocketClient,
    PARAMETER_SPECS,
    Rcp2ParameterState,
    Rcp2WebSocketConnectionError,
    Rcp2WebSocketProtocolError,
    Rcp2WebSocketSession,
    Rcp2WebSocketTimeout,
    extract_parameter_state,
)


DEFAULT_RCP2_RAW_PORT = 1112
DEFAULT_RCP2_WEBSOCKET_PORT = 9998
DEFAULT_RCP2_PORT = DEFAULT_RCP2_WEBSOCKET_PORT
DEFAULT_RCP2_SDK_ROOT = ""
DEFAULT_CONNECT_TIMEOUT_MS = 4000
DEFAULT_OPERATION_TIMEOUT_MS = 2500
DEFAULT_CONNECT_RETRIES = 2
DEFAULT_APPLY_RETRY_COUNT = 1
DEFAULT_TRANSPORT_KIND = "websocket"
DEFAULT_SETTLE_TIMEOUT_MS = 300
EXPOSURE_ADJUST_MIN = -4.0
EXPOSURE_ADJUST_MAX = 4.0
EXPOSURE_VERIFY_TOLERANCE = 0.001
KELVIN_VERIFY_TOLERANCE = 50
TINT_VERIFY_TOLERANCE = 0.5
SAFE_FIELD_ORDER = ("kelvin", "tint", "exposureAdjust")
OPERATOR_EXPOSURE_TOLERANCE = 0.01
OPERATOR_KELVIN_TOLERANCE = 50
OPERATOR_TINT_TOLERANCE = 0.01
VERIFICATION_REPORT_SCHEMA = "r3dmatch_rcp2_verification_report_v1"
TRANSACTIONAL_VERIFICATION_SCHEMA = "r3dmatch_rcp2_post_apply_verification_v1"


class Rcp2ApplyError(RuntimeError):
    pass


class Rcp2ConnectionError(Rcp2ApplyError):
    pass


class Rcp2RejectedError(Rcp2ApplyError):
    pass


class Rcp2TimeoutError(Rcp2ApplyError):
    pass


class Rcp2InvalidPayloadError(Rcp2ApplyError):
    pass


def _clamp(value: float, lower: float, upper: float) -> float:
    return float(max(lower, min(upper, value)))


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_rcp2_sdk_root(sdk_root: Optional[str] = None) -> str:
    explicit = str(sdk_root or "").strip()
    if explicit:
        return explicit
    for env_name in ("R3DMATCH_RCP2_SDK_ROOT", "RCP2_SDK_ROOT"):
        value = str(os.environ.get(env_name, "") or "").strip()
        if value:
            return value
    return ""


def _live_bridge_source_path() -> Path:
    return Path(__file__).resolve().parent / "native" / "rcp2_live_bridge.c"


def _live_bridge_output_path() -> Path:
    override = os.environ.get("R3DMATCH_RCP2_BUILD_DIR", "").strip()
    if override:
        return Path(override).expanduser().resolve() / "rcp2_live_bridge.dylib"
    return _project_root() / "build" / "rcp2_live" / "rcp2_live_bridge.dylib"


def _inventory_label_from_payload(entry: Dict[str, object]) -> Optional[str]:
    explicit = str(entry.get("inventory_camera_label") or "").strip().upper()
    if explicit:
        return explicit
    source_path = str(entry.get("source_path") or "").strip()
    if source_path:
        label = inventory_camera_label_from_source_path(source_path)
        if label:
            return label
    for key in ("clip_id", "camera_id"):
        label = inventory_camera_label_from_clip_id(str(entry.get(key) or ""))
        if label:
            return label
    return None


def _sdk_paths(sdk_root: str = DEFAULT_RCP2_SDK_ROOT) -> Dict[str, Path]:
    resolved_root = resolve_rcp2_sdk_root(sdk_root)
    root = Path(resolved_root).expanduser() if resolved_root else Path()
    return {
        "sdk_root_configured": Path(resolved_root).expanduser() if resolved_root else None,
        "sdk_root": root,
        "header_path": root / "rcp_sdk" / "rcp_api" / "rcp_api.h",
        "source_path": root / "rcp_sdk" / "rcp_api" / "rcp_api.c",
        "bridge_source_path": _live_bridge_source_path(),
        "bridge_library_path": _live_bridge_output_path(),
    }


def _compile_live_bridge(sdk_root: str = DEFAULT_RCP2_SDK_ROOT) -> Path:
    paths = _sdk_paths(sdk_root)
    if paths.get("sdk_root_configured") is None:
        raise Rcp2ApplyError(
            "RCP SDK root is not configured. Set R3DMATCH_RCP2_SDK_ROOT or RCP2_SDK_ROOT, "
            "or pass --sdk-root when using the raw-legacy RCP2 transport."
        )
    header_path = paths["header_path"]
    source_path = paths["source_path"]
    bridge_source_path = paths["bridge_source_path"]
    output_path = paths["bridge_library_path"]
    if not header_path.exists() or not source_path.exists():
        raise Rcp2ApplyError(f"RCP SDK source was not found under {paths['sdk_root']}.")
    if not bridge_source_path.exists():
        raise Rcp2ApplyError(f"Live bridge source is missing: {bridge_source_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    needs_rebuild = not output_path.exists()
    if not needs_rebuild:
        output_mtime = output_path.stat().st_mtime
        needs_rebuild = any(path.stat().st_mtime > output_mtime for path in (header_path, source_path, bridge_source_path))
    if not needs_rebuild:
        return output_path

    include_dir = str(header_path.parent)
    command = [
        "cc",
        "-std=c11",
        "-O2",
        "-fPIC",
        "-shared",
        "-DHAVE_STRLCPY",
        "-DHAVE_STRLCAT",
        "-I",
        include_dir,
        str(bridge_source_path),
        str(source_path),
        "-o",
        str(output_path),
        "-lpthread",
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise Rcp2ApplyError(
            "Failed to compile the live RCP2 bridge:\n"
            f"Command: {' '.join(command)}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    return output_path


def probe_rcp2_sdk(sdk_root: str = DEFAULT_RCP2_SDK_ROOT, *, ensure_live_bridge: bool = False) -> Dict[str, object]:
    paths = _sdk_paths(sdk_root)
    bridge_ready = paths["bridge_library_path"].exists()
    bridge_error = None
    if ensure_live_bridge:
        try:
            compiled_path = _compile_live_bridge(sdk_root)
            bridge_ready = compiled_path.exists()
        except Exception as exc:  # pragma: no cover - exercised in live environments
            bridge_error = str(exc)
    return {
        "sdk_root": str(paths.get("sdk_root_configured") or ""),
        "header_path": str(paths["header_path"]),
        "source_path": str(paths["source_path"]),
        "bridge_source_path": str(paths["bridge_source_path"]),
        "bridge_library_path": str(paths["bridge_library_path"]),
        "sdk_present": paths["header_path"].exists() and paths["source_path"].exists(),
        "live_bridge_ready": bridge_ready,
        "bridge_error": bridge_error,
    }


def clamp_calibration_values(calibration: Dict[str, object]) -> Dict[str, object]:
    return {
        "exposureAdjust": round(
            _clamp(float(calibration.get("exposureAdjust", 0.0) or 0.0), EXPOSURE_ADJUST_MIN, EXPOSURE_ADJUST_MAX),
            6,
        ),
        "kelvin": int(round(_clamp(float(calibration.get("kelvin", 5600) or 5600), KELVIN_MIN, KELVIN_MAX))),
        "tint": round(_clamp(float(calibration.get("tint", 0.0) or 0.0), TINT_MIN, TINT_MAX), 1),
    }


def _looks_like_timeout(message: str) -> bool:
    lowered = str(message or "").strip().lower()
    return "timeout" in lowered or "timed out" in lowered


def _prepare_calibration_values(calibration: Dict[str, object]) -> Dict[str, object]:
    raw_exposure = float(calibration.get("exposureAdjust", 0.0) or 0.0)
    raw_kelvin = float(calibration.get("kelvin", 5600) or 5600)
    raw_tint = float(calibration.get("tint", 0.0) or 0.0)
    invalid_reasons: List[str] = []
    warnings: List[str] = []
    if raw_exposure < EXPOSURE_ADJUST_MIN or raw_exposure > EXPOSURE_ADJUST_MAX:
        invalid_reasons.append(
            f"exposureAdjust {raw_exposure:.3f} is outside the safe range {EXPOSURE_ADJUST_MIN:.3f} to {EXPOSURE_ADJUST_MAX:.3f}."
        )
    if raw_kelvin < KELVIN_MIN or raw_kelvin > KELVIN_MAX:
        invalid_reasons.append(f"kelvin {raw_kelvin:.0f} is outside the safe range {KELVIN_MIN} to {KELVIN_MAX}.")
    if raw_tint < TINT_MIN or raw_tint > TINT_MAX:
        invalid_reasons.append(f"tint {raw_tint:.3f} is outside the safe range {TINT_MIN:.3f} to {TINT_MAX:.3f}.")

    normalized = clamp_calibration_values(calibration)
    if not invalid_reasons:
        if abs(float(normalized["exposureAdjust"]) - raw_exposure) > 1e-6:
            warnings.append("exposureAdjust was normalized to the nearest safe precision before sending.")
        if abs(float(normalized["kelvin"]) - raw_kelvin) > 0.5:
            warnings.append("kelvin was normalized to the nearest safe integer before sending.")
        if abs(float(normalized["tint"]) - raw_tint) > 1e-6:
            warnings.append("tint was normalized to the nearest safe precision before sending.")
    return {
        "requested": normalized,
        "raw_requested": {
            "exposureAdjust": raw_exposure,
            "kelvin": raw_kelvin,
            "tint": raw_tint,
        },
        "warnings": warnings,
        "invalid_reasons": invalid_reasons,
    }


def _verification_tolerance_for_field(field_name: str) -> float:
    if field_name == "exposureAdjust":
        return float(EXPOSURE_VERIFY_TOLERANCE)
    if field_name == "kelvin":
        return float(KELVIN_VERIFY_TOLERANCE)
    if field_name == "tint":
        return float(TINT_VERIFY_TOLERANCE)
    raise KeyError(field_name)


def _coerce_field_value(field_name: str, value: object) -> object:
    if value is None:
        return None
    if field_name == "kelvin":
        return int(round(float(value)))
    return round(float(value), 6)


def _field_delta(field_name: str, requested_value: object, applied_value: object) -> object:
    if requested_value is None or applied_value is None:
        return None
    if field_name == "kelvin":
        return int(round(float(applied_value) - float(requested_value)))
    return round(float(applied_value) - float(requested_value), 6)


def _field_verification_status(field_name: str, requested_value: object, applied_value: object) -> str:
    if applied_value is None:
        return "NOT_AVAILABLE"
    delta = _field_delta(field_name, requested_value, applied_value)
    if delta in {0, 0.0}:
        return "EXACT_MATCH"
    if abs(float(delta)) <= _verification_tolerance_for_field(field_name):
        return "WITHIN_TOLERANCE"
    return "MISMATCH"


def _build_field_verification(
    *,
    field_name: str,
    requested_value: object,
    applied_value: object,
    requested_raw_value: Optional[int] = None,
    applied_raw_value: Optional[int] = None,
    parameter_id: str = "",
    verification_status: Optional[str] = None,
    note: str = "",
    set_action: str = "",
) -> Dict[str, object]:
    status = verification_status or _field_verification_status(field_name, requested_value, applied_value)
    return {
        "field_name": field_name,
        "parameter_id": parameter_id or PARAMETER_SPECS[field_name].parameter_id,
        "requested_value": _coerce_field_value(field_name, requested_value),
        "applied_value": _coerce_field_value(field_name, applied_value),
        "delta": _field_delta(field_name, requested_value, applied_value),
        "requested_raw_value": requested_raw_value,
        "applied_raw_value": applied_raw_value,
        "tolerance": _verification_tolerance_for_field(field_name),
        "verification_status": status,
        "note": str(note or ""),
        "set_action": str(set_action or ""),
    }


def _summarize_field_verifications(
    per_field: Dict[str, Dict[str, object]],
    *,
    invalid_input: bool = False,
) -> Dict[str, object]:
    exact_fields: List[str] = []
    tolerance_fields: List[str] = []
    mismatched_fields: List[str] = []
    timeout_fields: List[str] = []
    unavailable_fields: List[str] = []
    for field_name in SAFE_FIELD_ORDER:
        result = dict(per_field.get(field_name) or {})
        status = str(result.get("verification_status") or "NOT_AVAILABLE")
        if status == "EXACT_MATCH":
            exact_fields.append(field_name)
        elif status == "WITHIN_TOLERANCE":
            tolerance_fields.append(field_name)
        elif status == "MISMATCH":
            mismatched_fields.append(field_name)
        elif status == "TIMEOUT":
            timeout_fields.append(field_name)
        else:
            unavailable_fields.append(field_name)
    if invalid_input:
        final_status = "INVALID_INPUT"
    elif timeout_fields:
        final_status = "TIMEOUT"
    elif mismatched_fields or unavailable_fields:
        final_status = "FAILED"
    elif tolerance_fields:
        final_status = "VERIFIED_WITH_TOLERANCE"
    else:
        final_status = "VERIFIED"
    return {
        "field_count": len(SAFE_FIELD_ORDER),
        "exact_match_fields": exact_fields,
        "within_tolerance_fields": tolerance_fields,
        "mismatched_fields": mismatched_fields,
        "timeout_fields": timeout_fields,
        "unavailable_fields": unavailable_fields,
        "all_verified": final_status in {"VERIFIED", "VERIFIED_WITH_TOLERANCE"},
        "final_status": final_status,
    }


def _legacy_verification_from_transaction(
    desired: Dict[str, object],
    readback: Dict[str, object],
    transaction: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    if transaction:
        summary = dict(transaction.get("verification_summary") or {})
        mismatches = list(summary.get("mismatched_fields") or []) + list(summary.get("timeout_fields") or [])
        return {
            "matches": bool(summary.get("all_verified")),
            "mismatched_fields": mismatches,
            "tolerances": {
                "exposureAdjust": EXPOSURE_VERIFY_TOLERANCE,
                "kelvin": KELVIN_VERIFY_TOLERANCE,
                "tint": TINT_VERIFY_TOLERANCE,
            },
            "field_results": dict(transaction.get("per_field") or {}),
            "final_status": str(summary.get("final_status") or ""),
        }
    mismatches: List[str] = []
    if abs(float(readback.get("exposureAdjust", 0.0) or 0.0) - float(desired["exposureAdjust"])) > EXPOSURE_VERIFY_TOLERANCE:
        mismatches.append("exposureAdjust")
    if abs(float(readback.get("kelvin", 0.0) or 0.0) - float(desired["kelvin"])) > KELVIN_VERIFY_TOLERANCE:
        mismatches.append("kelvin")
    if abs(float(readback.get("tint", 0.0) or 0.0) - float(desired["tint"])) > TINT_VERIFY_TOLERANCE:
        mismatches.append("tint")
    return {
        "matches": not mismatches,
        "mismatched_fields": mismatches,
        "tolerances": {
            "exposureAdjust": EXPOSURE_VERIFY_TOLERANCE,
            "kelvin": KELVIN_VERIFY_TOLERANCE,
            "tint": TINT_VERIFY_TOLERANCE,
        },
    }


def _apply_status_from_transaction(transaction: Dict[str, object], *, transport_mode: str) -> str:
    if transport_mode == "dry_run":
        return "dry_run"
    final_status = str((transaction.get("verification_summary") or {}).get("final_status") or transaction.get("final_status") or "")
    if final_status in {"VERIFIED", "VERIFIED_WITH_TOLERANCE"}:
        return "applied_successfully"
    if final_status == "TIMEOUT":
        return "timeout"
    if final_status == "INVALID_INPUT":
        return "invalid_input"
    return "mismatch_after_writeback"


def _failure_reason_from_transaction(transaction: Dict[str, object], *, status: str) -> str:
    if status in {"applied_successfully", "dry_run"}:
        return ""
    if status == "timeout":
        return "timeout"
    if status == "invalid_input":
        return "invalid_input"
    return "mismatch_after_writeback"


def _default_post_apply_verification_path(out_path: str) -> Path:
    report_path = Path(out_path).expanduser().resolve()
    return report_path.with_name("rcp2_post_apply_verification.json")


def _failure_reason_for_exception(exc: Exception) -> str:
    if isinstance(exc, Rcp2InvalidPayloadError):
        return "invalid_payload"
    if isinstance(exc, Rcp2TimeoutError):
        return "timeout"
    if isinstance(exc, Rcp2RejectedError):
        return "rejected_by_camera"
    if isinstance(exc, Rcp2ConnectionError):
        return "failed_to_connect"
    return "apply_error"


def _status_for_failure_reason(reason: str) -> str:
    if reason in {"timeout", "rejected_by_camera", "invalid_payload"}:
        return reason
    if reason == "mismatch_after_writeback":
        return reason
    return "failed_to_connect"


def _should_retry_failure(reason: str, attempt_index: int, retry_count: int) -> bool:
    if attempt_index > retry_count:
        return False
    return reason in {"failed_to_connect", "timeout"}


def _read_json(path: Path) -> Dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise Rcp2ApplyError(f"Calibration payload at {path} is not a JSON object.")
    return payload


def _normalize_selection_token(value: str) -> str:
    token = str(value or "").strip()
    return token.upper()


@dataclass
class ApplyTarget:
    camera_id: str
    clip_id: str
    inventory_camera_label: str
    inventory_camera_ip: str
    source_path: str
    calibration: Dict[str, object]
    confidence: float
    is_hero_camera: bool
    notes: List[str]
    payload_path: str


@dataclass
class LiveRcp2Session:
    handle: int
    host: str
    port: int
    camera_label: str
    connection_state: int
    camera_info: Dict[str, object]


class Rcp2Transport(Protocol):
    mode: str

    def open(self, *, host: str, port: int, camera_label: str) -> object:
        ...

    def close(self, session: object) -> None:
        ...

    def read_state(self, session: object) -> Dict[str, object]:
        ...

    def write_state(self, session: object, values: Dict[str, object]) -> None:
        ...

    def apply_state_transactionally(
        self,
        session: object,
        values: Dict[str, object],
        *,
        stop_on_failure: bool = True,
        enable_histogram_guard: bool = False,
        include_clip_metadata_cross_check: bool = False,
    ) -> Dict[str, object]:
        ...


class DryRunRcp2Transport:
    mode = "dry_run"

    def open(self, *, host: str, port: int, camera_label: str) -> Dict[str, object]:
        return {"host": host, "port": port, "camera_label": camera_label, "state": {}}

    def close(self, session: object) -> None:
        return None

    def read_state(self, session: object) -> Dict[str, object]:
        return dict((session or {}).get("state") or {})

    def write_state(self, session: object, values: Dict[str, object]) -> None:
        if not isinstance(session, dict):
            raise Rcp2ApplyError("Invalid dry-run session.")
        session["state"] = dict(values)

    def apply_state_transactionally(
        self,
        session: object,
        values: Dict[str, object],
        *,
        stop_on_failure: bool = True,
        enable_histogram_guard: bool = False,
        include_clip_metadata_cross_check: bool = False,
    ) -> Dict[str, object]:
        del stop_on_failure, enable_histogram_guard, include_clip_metadata_cross_check
        if not isinstance(session, dict):
            raise Rcp2ApplyError("Invalid dry-run session.")
        before_state = dict((session or {}).get("state") or {})
        original_before_state = dict(before_state)
        requested = clamp_calibration_values(values)
        per_field: Dict[str, Dict[str, object]] = {}
        write_sequence_trace: List[Dict[str, object]] = []
        for index, field_name in enumerate(SAFE_FIELD_ORDER, start=1):
            current_value = before_state.get(field_name, requested[field_name])
            session.setdefault("state", {})
            session["state"][field_name] = requested[field_name]
            result = _build_field_verification(
                field_name=field_name,
                requested_value=requested[field_name],
                applied_value=requested[field_name],
                parameter_id=PARAMETER_SPECS[field_name].parameter_id,
                set_action="simulated_write",
            )
            result["step"] = index
            result["before_value"] = _coerce_field_value(field_name, current_value)
            per_field[field_name] = result
            write_sequence_trace.append(dict(result))
            before_state[field_name] = requested[field_name]
        summary = _summarize_field_verifications(per_field)
        return {
            "before_state": original_before_state,
            "final_readback": dict((session or {}).get("state") or {}),
            "per_field": per_field,
            "write_sequence_trace": write_sequence_trace,
            "verification_summary": summary,
            "final_status": summary["final_status"],
            "histogram_guard": {"enabled": False, "status": "not_run"},
            "clip_metadata_cross_check": {"status": "not_run"},
        }


class MemoryRcp2Transport:
    mode = "mock"

    def __init__(
        self,
        *,
        initial_states: Optional[Dict[str, Dict[str, object]]] = None,
        unreachable_hosts: Optional[Iterable[str]] = None,
        rejected_hosts: Optional[Iterable[str]] = None,
        readback_overrides: Optional[Dict[str, Dict[str, object]]] = None,
    ) -> None:
        self._states = {str(host): dict(state) for host, state in (initial_states or {}).items()}
        self._unreachable = {str(host) for host in (unreachable_hosts or [])}
        self._rejected = {str(host) for host in (rejected_hosts or [])}
        self._overrides = {str(host): dict(values) for host, values in (readback_overrides or {}).items()}

    def open(self, *, host: str, port: int, camera_label: str) -> Dict[str, object]:
        if host in self._unreachable:
            raise Rcp2ConnectionError(f"Failed to connect to {camera_label} at {host}:{port}.")
        return {"host": host, "port": port, "camera_label": camera_label}

    def close(self, session: object) -> None:
        return None

    def read_state(self, session: object) -> Dict[str, object]:
        host = str((session or {}).get("host") or "")
        if host in self._overrides:
            return dict(self._overrides[host])
        return dict(self._states.get(host) or {})

    def write_state(self, session: object, values: Dict[str, object]) -> None:
        host = str((session or {}).get("host") or "")
        if host in self._rejected:
            raise Rcp2RejectedError(f"Camera at {host} rejected the calibration payload.")
        self._states[host] = dict(values)

    def apply_state_transactionally(
        self,
        session: object,
        values: Dict[str, object],
        *,
        stop_on_failure: bool = True,
        enable_histogram_guard: bool = False,
        include_clip_metadata_cross_check: bool = False,
    ) -> Dict[str, object]:
        del enable_histogram_guard, include_clip_metadata_cross_check
        host = str((session or {}).get("host") or "")
        if host in self._rejected:
            raise Rcp2RejectedError(f"Camera at {host} rejected the calibration payload.")
        before_state = dict(self._states.get(host) or {})
        original_before_state = dict(before_state)
        requested = clamp_calibration_values(values)
        working_state = dict(before_state)
        per_field: Dict[str, Dict[str, object]] = {}
        write_sequence_trace: List[Dict[str, object]] = []
        for index, field_name in enumerate(SAFE_FIELD_ORDER, start=1):
            working_state[field_name] = requested[field_name]
            self._states[host] = dict(working_state)
            applied_state = dict(self._overrides.get(host) or self._states.get(host) or {})
            result = _build_field_verification(
                field_name=field_name,
                requested_value=requested[field_name],
                applied_value=applied_state.get(field_name),
                parameter_id=PARAMETER_SPECS[field_name].parameter_id,
                set_action="simulated_write",
            )
            result["step"] = index
            result["before_value"] = _coerce_field_value(field_name, before_state.get(field_name))
            per_field[field_name] = result
            write_sequence_trace.append(dict(result))
            if stop_on_failure and str(result["verification_status"]) not in {"EXACT_MATCH", "WITHIN_TOLERANCE"}:
                break
            before_state[field_name] = applied_state.get(field_name)
        final_readback = dict(self._overrides.get(host) or self._states.get(host) or {})
        summary = _summarize_field_verifications(per_field)
        return {
            "before_state": original_before_state,
            "final_readback": final_readback,
            "per_field": per_field,
            "write_sequence_trace": write_sequence_trace,
            "verification_summary": summary,
            "final_status": summary["final_status"],
            "histogram_guard": {"enabled": False, "status": "not_run"},
            "clip_metadata_cross_check": {"status": "not_run"},
        }


def normalize_transport_kind(value: str) -> str:
    token = str(value or "").strip().lower().replace("_", "-")
    aliases = {
        "websocket": "websocket",
        "ws": "websocket",
        "json": "websocket",
        "legacy": "raw-legacy",
        "raw": "raw-legacy",
        "raw-legacy": "raw-legacy",
        "tcp": "raw-legacy",
        "native": "raw-legacy",
    }
    if token not in aliases:
        raise Rcp2ApplyError("transport must be websocket or raw-legacy")
    return aliases[token]


class WebSocketRcp2Transport:
    mode = "live_websocket"

    def __init__(
        self,
        *,
        connect_timeout_ms: int = DEFAULT_CONNECT_TIMEOUT_MS,
        operation_timeout_ms: int = DEFAULT_OPERATION_TIMEOUT_MS,
        settle_timeout_ms: int = DEFAULT_SETTLE_TIMEOUT_MS,
        strings_decoded: int = 1,
        json_minified: int = 1,
        include_cacheable_flags: int = 0,
        encoding_type: str = "legacy",
        client_name: str = "R3DMatch",
        client_version: str = "0.1",
    ) -> None:
        self.connect_timeout_ms = int(connect_timeout_ms)
        self.operation_timeout_ms = int(operation_timeout_ms)
        self.settle_timeout_ms = int(settle_timeout_ms)
        self.strings_decoded = int(strings_decoded)
        self.json_minified = int(json_minified)
        self.include_cacheable_flags = int(include_cacheable_flags)
        self.encoding_type = str(encoding_type or "legacy")
        self.client_name = str(client_name or "R3DMatch")
        self.client_version = str(client_version or "0.1")

    def _client(self, host: str, port: int) -> JsonWebSocketClient:
        return JsonWebSocketClient(
            host=host,
            port=port,
            connect_timeout_ms=self.connect_timeout_ms,
            operation_timeout_ms=self.operation_timeout_ms,
            settle_timeout_ms=self.settle_timeout_ms,
            raw_logging=bool(os.environ.get("R3DMATCH_RCP2_RAW_LOG")),
        )

    def _config_payload(self) -> Dict[str, object]:
        return {
            "type": "rcp_config",
            "strings_decoded": self.strings_decoded,
            "json_minified": self.json_minified,
            "include_cacheable_flags": self.include_cacheable_flags,
            "encoding_type": self.encoding_type,
            "client": {
                "name": self.client_name,
                "version": self.client_version,
            },
        }

    @staticmethod
    def _message_type_matcher(message_type: str):
        expected = str(message_type or "").strip()

        def _matches(message: Dict[str, object]) -> bool:
            return str(message.get("type") or "").strip() == expected

        return _matches

    @staticmethod
    def _parameter_matcher(parameter_id: str):
        expected = str(parameter_id or "").strip().upper().removeprefix("RCP_PARAM_")

        def _matches(message: Dict[str, object]) -> bool:
            message_id = str(message.get("id") or "").strip().upper().removeprefix("RCP_PARAM_")
            return message_id == expected

        return _matches

    @staticmethod
    def _multi_parameter_matcher(*parameter_ids: str):
        expected = {str(parameter_id or "").strip().upper().removeprefix("RCP_PARAM_") for parameter_id in parameter_ids if str(parameter_id or "").strip()}

        def _matches(message: Dict[str, object]) -> bool:
            message_id = str(message.get("id") or "").strip().upper().removeprefix("RCP_PARAM_")
            return message_id in expected or str(message.get("type") or "") == "rcp_cur_clip_list"

        return _matches

    def _read_parameter_state(self, session: Rcp2WebSocketSession, field_name: str) -> Rcp2ParameterState:
        spec = PARAMETER_SPECS[field_name]
        client = self._client(session.host, session.port)
        try:
            messages = client.request_messages(
                session,
                {"type": "rcp_get", "id": spec.parameter_id},
                matcher=self._parameter_matcher(spec.parameter_id),
                timeout_ms=self.operation_timeout_ms,
                settle_timeout_ms=self.settle_timeout_ms,
            )
        except (Rcp2WebSocketTimeout, Rcp2WebSocketConnectionError) as exc:
            message = f"Timed out while reading {spec.parameter_id} from ws://{session.host}:{session.port}: {exc}"
            raise Rcp2TimeoutError(message) from exc
        except Rcp2WebSocketProtocolError as exc:
            raise Rcp2RejectedError(str(exc)) from exc
        state = extract_parameter_state(messages, parameter_id=spec.parameter_id, fallback_divider=spec.default_divider)
        current = session.parameter_states.get(spec.field_name)
        if current is None or int(state.message_index) >= int(current.message_index):
            session.parameter_states[spec.field_name] = state
        return state

    def _write_parameter_state(self, session: Rcp2WebSocketSession, field_name: str, raw_value: int) -> Rcp2ParameterState:
        spec = PARAMETER_SPECS[field_name]
        client = self._client(session.host, session.port)
        try:
            messages = client.request_messages(
                session,
                {"type": "rcp_set", "id": spec.parameter_id, "value": int(raw_value)},
                matcher=self._parameter_matcher(spec.parameter_id),
                timeout_ms=self.operation_timeout_ms,
                settle_timeout_ms=self.settle_timeout_ms,
            )
        except (Rcp2WebSocketTimeout, Rcp2WebSocketConnectionError) as exc:
            message = f"Timed out while writing {spec.parameter_id} to ws://{session.host}:{session.port}: {exc}"
            raise Rcp2TimeoutError(message) from exc
        except Rcp2WebSocketProtocolError as exc:
            raise Rcp2RejectedError(str(exc)) from exc
        state = extract_parameter_state(messages, parameter_id=spec.parameter_id, fallback_divider=spec.default_divider)
        current = session.parameter_states.get(spec.field_name)
        if current is None or int(state.message_index) >= int(current.message_index):
            session.parameter_states[spec.field_name] = state
        return state

    def _best_effort_histogram_guard(self, session: Rcp2WebSocketSession) -> Dict[str, object]:
        client = self._client(session.host, session.port)
        try:
            messages = client.request_messages(
                session,
                {"type": "rcp_get", "id": "HISTOGRAM"},
                matcher=self._multi_parameter_matcher("HISTOGRAM", "DSHIST"),
                timeout_ms=self.operation_timeout_ms,
                settle_timeout_ms=self.settle_timeout_ms,
            )
        except Exception:
            return {"enabled": True, "status": "not_available", "clipping_detected": False}
        clip_values: Dict[str, float] = {}
        for message in messages:
            if not isinstance(message, dict):
                continue
            for key, value in message.items():
                if key in {"bottom_clip_r", "bottom_clip_g", "bottom_clip_b", "top_clip_r", "top_clip_g", "top_clip_b"}:
                    try:
                        clip_values[key] = float(value)
                    except (TypeError, ValueError):
                        continue
        clipping_detected = any(float(value) > 0.0 for value in clip_values.values())
        payload = {
            "enabled": True,
            "status": "success" if clip_values else "not_available",
            "clipping_detected": clipping_detected,
            "clip_values": clip_values,
        }
        if clipping_detected:
            payload["note"] = "Clipping detected on gray reference — measurement invalid"
        return payload

    def _best_effort_clip_metadata(self, session: Rcp2WebSocketSession, applied_state: Dict[str, object]) -> Dict[str, object]:
        client = self._client(session.host, session.port)
        try:
            messages = client.request_messages(
                session,
                {"type": "rcp_get", "id": "CLIP_LIST"},
                matcher=self._multi_parameter_matcher("CLIP_LIST"),
                timeout_ms=self.operation_timeout_ms,
                settle_timeout_ms=self.settle_timeout_ms,
            )
        except Exception:
            return {"status": "not_available", "metadata_match_status": "NOT_AVAILABLE"}
        clip_entry: Dict[str, object] = {}
        for message in messages:
            if not isinstance(message, dict):
                continue
            clips = message.get("clips")
            if isinstance(clips, list) and clips:
                first = clips[0]
                if isinstance(first, dict):
                    clip_entry = dict(first)
                    break
            if str(message.get("type") or "") == "rcp_cur_clip_list":
                clip_entry = dict(message)
        if not clip_entry:
            return {"status": "not_available", "metadata_match_status": "NOT_AVAILABLE"}
        comparisons = {}
        statuses = []
        for field_name, metadata_keys in {
            "kelvin": ("kelvin", "color_temperature"),
            "tint": ("tint",),
            "exposureAdjust": ("exposure_adjust", "exposureAdjust"),
        }.items():
            metadata_value = None
            for key in metadata_keys:
                if key in clip_entry:
                    metadata_value = clip_entry.get(key)
                    break
            if metadata_value is None:
                comparisons[field_name] = {"status": "NOT_AVAILABLE"}
                statuses.append("NOT_AVAILABLE")
                continue
            status = _field_verification_status(field_name, applied_state.get(field_name), metadata_value)
            comparisons[field_name] = {
                "metadata_value": _coerce_field_value(field_name, metadata_value),
                "applied_value": _coerce_field_value(field_name, applied_state.get(field_name)),
                "delta": _field_delta(field_name, applied_state.get(field_name), metadata_value),
                "status": status,
            }
            statuses.append(status)
        if any(status == "MISMATCH" for status in statuses):
            match_status = "MISMATCH"
        elif any(status == "WITHIN_TOLERANCE" for status in statuses):
            match_status = "WITHIN_TOLERANCE"
        elif all(status == "EXACT_MATCH" for status in statuses if status != "NOT_AVAILABLE") and any(status != "NOT_AVAILABLE" for status in statuses):
            match_status = "EXACT_MATCH"
        else:
            match_status = "NOT_AVAILABLE"
        return {
            "status": "success",
            "metadata_match_status": match_status,
            "fields": comparisons,
        }

    @staticmethod
    def _camera_info_summary(camera_info: Dict[str, object]) -> Dict[str, object]:
        camera_type = dict(camera_info.get("camera_type") or {})
        version = dict(camera_info.get("version") or {})
        return {
            "camera_name": str(camera_info.get("name") or ""),
            "camera_type": str(camera_type.get("str") or camera_type.get("num") or ""),
            "camera_type_code": camera_type.get("num"),
            "serial_number": str(camera_info.get("serial_number") or ""),
            "camera_version": str(version.get("str") or ""),
            "supported_objects": list(camera_info.get("supported_objects") or []),
        }

    def open(self, *, host: str, port: int, camera_label: str) -> Rcp2WebSocketSession:
        client = self._client(host, port)
        try:
            session = client.connect()
            config_messages = client.request_messages(
                session,
                self._config_payload(),
                matcher=self._message_type_matcher("rcp_config"),
                timeout_ms=self.operation_timeout_ms,
                settle_timeout_ms=self.settle_timeout_ms,
            )
            types_messages = client.request_messages(
                session,
                {"type": "rcp_get_types"},
                matcher=self._message_type_matcher("rcp_cur_types"),
                timeout_ms=self.operation_timeout_ms,
                settle_timeout_ms=self.settle_timeout_ms,
            )
            camera_messages = client.request_messages(
                session,
                {"type": "rcp_get", "id": "CAMERA_INFO"},
                matcher=self._message_type_matcher("rcp_cur_cam_info"),
                timeout_ms=self.operation_timeout_ms,
                settle_timeout_ms=self.settle_timeout_ms,
            )
        except Rcp2WebSocketConnectionError as exc:
            raise Rcp2ConnectionError(f"Failed to connect to {camera_label} at ws://{host}:{port}: {exc}") from exc
        except Rcp2WebSocketTimeout as exc:
            raise Rcp2TimeoutError(f"Timed out while initializing {camera_label} at ws://{host}:{port}: {exc}") from exc
        except Rcp2WebSocketProtocolError as exc:
            raise Rcp2RejectedError(f"Camera at ws://{host}:{port} rejected the RCP2 WebSocket session: {exc}") from exc

        session.config = dict(config_messages[-1])
        session.types_payload = dict(types_messages[-1])
        session.camera_info = dict(camera_messages[-1])
        return session

    def close(self, session: object) -> None:
        if not isinstance(session, Rcp2WebSocketSession):
            return
        self._client(session.host, session.port).close(session)

    def read_state(self, session: object) -> Dict[str, object]:
        if not isinstance(session, Rcp2WebSocketSession):
            raise Rcp2ApplyError("Invalid WebSocket RCP2 session.")
        exposure = self._read_parameter_state(session, "exposureAdjust")
        kelvin = self._read_parameter_state(session, "kelvin")
        tint = self._read_parameter_state(session, "tint")
        return {
            "exposureAdjust": round(float(exposure.value), 3),
            "kelvin": int(round(float(kelvin.value))),
            "tint": round(float(tint.value), 3),
            "connection_state": 1,
            "camera_info": self._camera_info_summary(session.camera_info),
            "raw_parameters": {
                "exposureAdjust": {
                    "parameter_id": exposure.parameter_id,
                    "raw_value": exposure.raw_value,
                    "divider": exposure.divider,
                    "display": exposure.display,
                    "edit_info": exposure.edit_info,
                },
                "kelvin": {
                    "parameter_id": kelvin.parameter_id,
                    "raw_value": kelvin.raw_value,
                    "divider": kelvin.divider,
                    "display": kelvin.display,
                    "edit_info": kelvin.edit_info,
                },
                "tint": {
                    "parameter_id": tint.parameter_id,
                    "raw_value": tint.raw_value,
                    "divider": tint.divider,
                    "display": tint.display,
                    "edit_info": tint.edit_info,
                },
            },
        }

    def write_state(self, session: object, values: Dict[str, object]) -> None:
        if not isinstance(session, Rcp2WebSocketSession):
            raise Rcp2ApplyError("Invalid WebSocket RCP2 session.")
        requested = clamp_calibration_values(values)
        for field_name in SAFE_FIELD_ORDER:
            current_state = self._read_parameter_state(session, field_name)
            raw_target = int(round(float(requested[field_name]) * current_state.divider))
            min_value = current_state.edit_info.get("min")
            max_value = current_state.edit_info.get("max")
            if min_value is not None and raw_target < int(min_value):
                raise Rcp2InvalidPayloadError(
                    f"{PARAMETER_SPECS[field_name].parameter_id} target {raw_target} is below the camera minimum {int(min_value)}."
                )
            if max_value is not None and raw_target > int(max_value):
                raise Rcp2InvalidPayloadError(
                    f"{PARAMETER_SPECS[field_name].parameter_id} target {raw_target} is above the camera maximum {int(max_value)}."
                )
            if current_state.raw_value == raw_target:
                continue
            self._write_parameter_state(session, field_name, raw_target)

    def apply_state_transactionally(
        self,
        session: object,
        values: Dict[str, object],
        *,
        stop_on_failure: bool = True,
        enable_histogram_guard: bool = False,
        include_clip_metadata_cross_check: bool = False,
    ) -> Dict[str, object]:
        if not isinstance(session, Rcp2WebSocketSession):
            raise Rcp2ApplyError("Invalid WebSocket RCP2 session.")
        requested = clamp_calibration_values(values)
        before_state = self.read_state(session)
        per_field: Dict[str, Dict[str, object]] = {}
        write_sequence_trace: List[Dict[str, object]] = []
        histogram_guard = {"enabled": False, "status": "not_run", "clipping_detected": False}
        invalid_input = False
        if enable_histogram_guard:
            histogram_guard = self._best_effort_histogram_guard(session)
            invalid_input = bool(histogram_guard.get("clipping_detected"))
        if invalid_input:
            summary = _summarize_field_verifications(per_field, invalid_input=True)
            return {
                "before_state": before_state,
                "final_readback": before_state,
                "per_field": per_field,
                "write_sequence_trace": write_sequence_trace,
                "verification_summary": summary,
                "final_status": summary["final_status"],
                "histogram_guard": histogram_guard,
                "clip_metadata_cross_check": {"status": "not_run"},
            }

        for index, field_name in enumerate(SAFE_FIELD_ORDER, start=1):
            current_state = self._read_parameter_state(session, field_name)
            raw_target = int(round(float(requested[field_name]) * current_state.divider))
            min_value = current_state.edit_info.get("min")
            max_value = current_state.edit_info.get("max")
            if min_value is not None and raw_target < int(min_value):
                raise Rcp2InvalidPayloadError(
                    f"{PARAMETER_SPECS[field_name].parameter_id} target {raw_target} is below the camera minimum {int(min_value)}."
                )
            if max_value is not None and raw_target > int(max_value):
                raise Rcp2InvalidPayloadError(
                    f"{PARAMETER_SPECS[field_name].parameter_id} target {raw_target} is above the camera maximum {int(max_value)}."
                )
            set_action = "skipped_noop"
            try:
                if current_state.raw_value != raw_target:
                    self._write_parameter_state(session, field_name, raw_target)
                    set_action = "rcp_set"
                verified_state = self._read_parameter_state(session, field_name)
                result = _build_field_verification(
                    field_name=field_name,
                    requested_value=requested[field_name],
                    applied_value=verified_state.value,
                    requested_raw_value=raw_target,
                    applied_raw_value=verified_state.raw_value,
                    parameter_id=verified_state.parameter_id,
                    set_action=set_action,
                )
            except Rcp2TimeoutError as exc:
                result = _build_field_verification(
                    field_name=field_name,
                    requested_value=requested[field_name],
                    applied_value=None,
                    requested_raw_value=raw_target,
                    parameter_id=PARAMETER_SPECS[field_name].parameter_id,
                    verification_status="TIMEOUT",
                    note=str(exc),
                    set_action=set_action or "rcp_set",
                )
            result["step"] = index
            result["before_value"] = _coerce_field_value(field_name, current_state.value)
            per_field[field_name] = result
            write_sequence_trace.append(dict(result))
            if stop_on_failure and str(result.get("verification_status") or "") not in {"EXACT_MATCH", "WITHIN_TOLERANCE"}:
                break

        final_readback = self.read_state(session)
        clip_metadata_cross_check = (
            self._best_effort_clip_metadata(session, final_readback) if include_clip_metadata_cross_check else {"status": "not_run"}
        )
        summary = _summarize_field_verifications(per_field, invalid_input=invalid_input)
        return {
            "before_state": before_state,
            "final_readback": final_readback,
            "per_field": per_field,
            "write_sequence_trace": write_sequence_trace,
            "verification_summary": summary,
            "final_status": summary["final_status"],
            "histogram_guard": histogram_guard,
            "clip_metadata_cross_check": clip_metadata_cross_check,
        }


class LiveRcp2Transport:
    mode = "live_raw"

    def __init__(
        self,
        *,
        sdk_root: str = DEFAULT_RCP2_SDK_ROOT,
        connect_timeout_ms: int = DEFAULT_CONNECT_TIMEOUT_MS,
        operation_timeout_ms: int = DEFAULT_OPERATION_TIMEOUT_MS,
        retries: int = DEFAULT_CONNECT_RETRIES,
    ) -> None:
        self.sdk_root = sdk_root
        self.connect_timeout_ms = int(connect_timeout_ms)
        self.operation_timeout_ms = int(operation_timeout_ms)
        self.retries = max(1, int(retries))
        self._lib = self._load_library()

    def _load_library(self) -> ctypes.CDLL:
        lib_path = _compile_live_bridge(self.sdk_root)
        lib = ctypes.CDLL(str(lib_path))
        lib.r3dmatch_rcp_open.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_size_t]
        lib.r3dmatch_rcp_open.restype = ctypes.c_void_p
        lib.r3dmatch_rcp_close.argtypes = [ctypes.c_void_p]
        lib.r3dmatch_rcp_close.restype = None
        lib.r3dmatch_rcp_get_connection_state.argtypes = [ctypes.c_void_p]
        lib.r3dmatch_rcp_get_connection_state.restype = ctypes.c_int
        lib.r3dmatch_rcp_get_camera_info.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_size_t,
            ctypes.c_char_p,
            ctypes.c_size_t,
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        lib.r3dmatch_rcp_get_camera_info.restype = ctypes.c_int
        lib.r3dmatch_rcp_get_supported_exposure_adjust.argtypes = [ctypes.c_void_p]
        lib.r3dmatch_rcp_get_supported_exposure_adjust.restype = ctypes.c_int
        lib.r3dmatch_rcp_get_supported_color_temperature.argtypes = [ctypes.c_void_p]
        lib.r3dmatch_rcp_get_supported_color_temperature.restype = ctypes.c_int
        lib.r3dmatch_rcp_get_supported_tint.argtypes = [ctypes.c_void_p]
        lib.r3dmatch_rcp_get_supported_tint.restype = ctypes.c_int
        for name in (
            "r3dmatch_rcp_set_exposure_adjust",
            "r3dmatch_rcp_set_color_temperature",
            "r3dmatch_rcp_set_tint",
        ):
            getattr(lib, name).argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.c_char_p, ctypes.c_size_t]
            getattr(lib, name).restype = ctypes.c_int
        for name in (
            "r3dmatch_rcp_get_exposure_adjust",
            "r3dmatch_rcp_get_color_temperature",
            "r3dmatch_rcp_get_tint",
        ):
            getattr(lib, name).argtypes = [
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_int32),
                ctypes.c_char_p,
                ctypes.c_size_t,
            ]
            getattr(lib, name).restype = ctypes.c_int
        return lib

    @staticmethod
    def _error_buffer() -> ctypes.Array[ctypes.c_char]:
        return ctypes.create_string_buffer(512)

    @staticmethod
    def _decode_error(buffer: ctypes.Array[ctypes.c_char]) -> str:
        return buffer.value.decode("utf-8", errors="ignore").strip()

    @staticmethod
    def _raise_transport_error(message: str, *, default_exception: type[Exception]) -> None:
        if _looks_like_timeout(message):
            raise Rcp2TimeoutError(message)
        raise default_exception(message)

    def open(self, *, host: str, port: int, camera_label: str) -> LiveRcp2Session:
        last_error = "unknown connection failure"
        for _attempt in range(self.retries):
            error_buf = self._error_buffer()
            handle = self._lib.r3dmatch_rcp_open(host.encode("utf-8"), int(port), self.connect_timeout_ms, error_buf, ctypes.sizeof(error_buf))
            if handle:
                camera_id_buf = ctypes.create_string_buffer(64)
                camera_type_buf = ctypes.create_string_buffer(128)
                camera_version_buf = ctypes.create_string_buffer(128)
                self._lib.r3dmatch_rcp_get_camera_info(
                    handle,
                    camera_id_buf,
                    ctypes.sizeof(camera_id_buf),
                    camera_type_buf,
                    ctypes.sizeof(camera_type_buf),
                    camera_version_buf,
                    ctypes.sizeof(camera_version_buf),
                )
                return LiveRcp2Session(
                    handle=int(handle),
                    host=host,
                    port=int(port),
                    camera_label=camera_label,
                    connection_state=int(self._lib.r3dmatch_rcp_get_connection_state(handle)),
                    camera_info={
                        "camera_id": camera_id_buf.value.decode("utf-8", errors="ignore"),
                        "camera_type": camera_type_buf.value.decode("utf-8", errors="ignore"),
                        "camera_version": camera_version_buf.value.decode("utf-8", errors="ignore"),
                    },
                )
            last_error = self._decode_error(error_buf) or "unknown connection failure"
        self._raise_transport_error(
            f"Failed to connect to {camera_label} at {host}:{port}: {last_error}",
            default_exception=Rcp2ConnectionError,
        )

    def close(self, session: object) -> None:
        if not isinstance(session, LiveRcp2Session):
            return
        self._lib.r3dmatch_rcp_close(ctypes.c_void_p(session.handle))

    def _read_int(self, session: LiveRcp2Session, getter_name: str) -> int:
        value = ctypes.c_int32()
        error_buf = self._error_buffer()
        getter = getattr(self._lib, getter_name)
        if not getter(ctypes.c_void_p(session.handle), self.operation_timeout_ms, ctypes.byref(value), error_buf, ctypes.sizeof(error_buf)):
            message = self._decode_error(error_buf) or f"readback failed for {getter_name}"
            self._raise_transport_error(message, default_exception=Rcp2RejectedError)
        return int(value.value)

    def _write_int(self, session: LiveRcp2Session, setter_name: str, value: int) -> None:
        error_buf = self._error_buffer()
        setter = getattr(self._lib, setter_name)
        if not setter(ctypes.c_void_p(session.handle), int(value), error_buf, ctypes.sizeof(error_buf)):
            message = self._decode_error(error_buf) or f"write failed for {setter_name}"
            self._raise_transport_error(message, default_exception=Rcp2RejectedError)

    def read_state(self, session: object) -> Dict[str, object]:
        if not isinstance(session, LiveRcp2Session):
            raise Rcp2ApplyError("Invalid live RCP2 session.")
        return {
            "exposureAdjust": round(self._read_int(session, "r3dmatch_rcp_get_exposure_adjust") / 1000.0, 3),
            "kelvin": int(self._read_int(session, "r3dmatch_rcp_get_color_temperature")),
            "tint": round(self._read_int(session, "r3dmatch_rcp_get_tint") / 1000.0, 3),
            "connection_state": session.connection_state,
            "camera_info": dict(session.camera_info),
        }

    def write_state(self, session: object, values: Dict[str, object]) -> None:
        if not isinstance(session, LiveRcp2Session):
            raise Rcp2ApplyError("Invalid live RCP2 session.")
        requested = clamp_calibration_values(values)
        self._write_int(session, "r3dmatch_rcp_set_color_temperature", int(requested["kelvin"]))
        self._write_int(session, "r3dmatch_rcp_set_tint", int(round(float(requested["tint"]) * 1000.0)))
        self._write_int(session, "r3dmatch_rcp_set_exposure_adjust", int(round(float(requested["exposureAdjust"]) * 1000.0)))

    def apply_state_transactionally(
        self,
        session: object,
        values: Dict[str, object],
        *,
        stop_on_failure: bool = True,
        enable_histogram_guard: bool = False,
        include_clip_metadata_cross_check: bool = False,
    ) -> Dict[str, object]:
        del enable_histogram_guard, include_clip_metadata_cross_check
        if not isinstance(session, LiveRcp2Session):
            raise Rcp2ApplyError("Invalid live RCP2 session.")
        before_state = self.read_state(session)
        requested = clamp_calibration_values(values)
        per_field: Dict[str, Dict[str, object]] = {}
        write_sequence_trace: List[Dict[str, object]] = []
        io_map = {
            "kelvin": ("r3dmatch_rcp_set_color_temperature", "r3dmatch_rcp_get_color_temperature", 1.0),
            "tint": ("r3dmatch_rcp_set_tint", "r3dmatch_rcp_get_tint", 1000.0),
            "exposureAdjust": ("r3dmatch_rcp_set_exposure_adjust", "r3dmatch_rcp_get_exposure_adjust", 1000.0),
        }
        current_state = dict(before_state)
        for index, field_name in enumerate(SAFE_FIELD_ORDER, start=1):
            setter_name, getter_name, divider = io_map[field_name]
            target_raw_value = int(round(float(requested[field_name]) * float(divider)))
            before_value = current_state.get(field_name)
            set_action = "skipped_noop"
            try:
                current_raw_value = self._read_int(session, getter_name)
                if current_raw_value != target_raw_value:
                    self._write_int(session, setter_name, target_raw_value)
                    set_action = "native_set"
                applied_raw_value = self._read_int(session, getter_name)
                applied_value = float(applied_raw_value) / float(divider)
                if field_name == "kelvin":
                    applied_value = int(round(applied_value))
                result = _build_field_verification(
                    field_name=field_name,
                    requested_value=requested[field_name],
                    applied_value=applied_value,
                    requested_raw_value=target_raw_value,
                    applied_raw_value=applied_raw_value,
                    parameter_id=PARAMETER_SPECS[field_name].parameter_id,
                    set_action=set_action,
                )
            except Rcp2TimeoutError as exc:
                result = _build_field_verification(
                    field_name=field_name,
                    requested_value=requested[field_name],
                    applied_value=None,
                    requested_raw_value=target_raw_value,
                    parameter_id=PARAMETER_SPECS[field_name].parameter_id,
                    verification_status="TIMEOUT",
                    note=str(exc),
                    set_action=set_action,
                )
            result["step"] = index
            result["before_value"] = _coerce_field_value(field_name, before_value)
            per_field[field_name] = result
            write_sequence_trace.append(dict(result))
            current_state[field_name] = result.get("applied_value", current_state.get(field_name))
            if stop_on_failure and str(result.get("verification_status") or "") not in {"EXACT_MATCH", "WITHIN_TOLERANCE"}:
                break
        final_readback = self.read_state(session)
        summary = _summarize_field_verifications(per_field)
        return {
            "before_state": before_state,
            "final_readback": final_readback,
            "per_field": per_field,
            "write_sequence_trace": write_sequence_trace,
            "verification_summary": summary,
            "final_status": summary["final_status"],
            "histogram_guard": {"enabled": False, "status": "not_run"},
            "clip_metadata_cross_check": {"status": "not_run"},
        }


def create_live_transport(
    *,
    transport_kind: str = DEFAULT_TRANSPORT_KIND,
    sdk_root: str = DEFAULT_RCP2_SDK_ROOT,
    connect_timeout_ms: int = DEFAULT_CONNECT_TIMEOUT_MS,
    operation_timeout_ms: int = DEFAULT_OPERATION_TIMEOUT_MS,
) -> Rcp2Transport:
    resolved_kind = normalize_transport_kind(transport_kind)
    if resolved_kind == "websocket":
        return WebSocketRcp2Transport(
            connect_timeout_ms=connect_timeout_ms,
            operation_timeout_ms=operation_timeout_ms,
        )
    return LiveRcp2Transport(
        sdk_root=sdk_root,
        connect_timeout_ms=connect_timeout_ms,
        operation_timeout_ms=operation_timeout_ms,
    )


def read_camera_state(
    *,
    host: str,
    port: int = DEFAULT_RCP2_PORT,
    camera_label: str = "LIVE",
    transport_kind: str = DEFAULT_TRANSPORT_KIND,
    sdk_root: str = DEFAULT_RCP2_SDK_ROOT,
) -> Dict[str, object]:
    transport = create_live_transport(transport_kind=transport_kind, sdk_root=sdk_root)
    session = None
    started_at = datetime.now(timezone.utc).isoformat()
    try:
        session = transport.open(host=host, port=port, camera_label=camera_label)
        state = transport.read_state(session)
        return {
            "schema_version": "r3dmatch_rcp2_camera_state_v1",
            "started_at": started_at,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "transport_mode": getattr(transport, "mode", transport_kind),
            "transport_kind": normalize_transport_kind(transport_kind),
            "host": host,
            "port": int(port),
            "camera_label": camera_label,
            "state": state,
        }
    finally:
        if session is not None:
            transport.close(session)


def test_rcp2_write_smoke(
    *,
    host: str,
    port: int = DEFAULT_RCP2_PORT,
    camera_label: str = "LIVE",
    field_name: str = "exposureAdjust",
    transport_kind: str = DEFAULT_TRANSPORT_KIND,
    sdk_root: str = DEFAULT_RCP2_SDK_ROOT,
) -> Dict[str, object]:
    resolved_kind = normalize_transport_kind(transport_kind)
    if resolved_kind != "websocket":
        raise Rcp2ApplyError("test_rcp2_write_smoke currently supports only the websocket transport.")
    transport = create_live_transport(transport_kind=resolved_kind, sdk_root=sdk_root)
    if not isinstance(transport, WebSocketRcp2Transport):
        raise Rcp2ApplyError("Failed to initialize the websocket RCP2 transport.")
    session = None
    started_at = datetime.now(timezone.utc).isoformat()
    try:
        session = transport.open(host=host, port=port, camera_label=camera_label)
        before_state = transport.read_state(session)
        if field_name not in PARAMETER_SPECS:
            raise Rcp2ApplyError(f"Unsupported smoke-test parameter: {field_name}")
        raw_before = dict((before_state.get("raw_parameters") or {}).get(field_name) or {})
        raw_value = int(raw_before.get("raw_value", 0) or 0)
        edit_info = dict(raw_before.get("edit_info") or {})
        min_value = int(edit_info.get("min", -8000) or -8000)
        max_value = int(edit_info.get("max", 8000) or 8000)
        step = max(1, int(edit_info.get("step", 1) or 1))
        probe_value = raw_value + step if raw_value + step <= max_value else raw_value - step
        if probe_value < min_value or probe_value > max_value:
            raise Rcp2ApplyError(f"No safe {field_name} probe step is available on this camera.")
        applied = transport._write_parameter_state(session, field_name, probe_value)
        after_probe = transport.read_state(session)
        restored = transport._write_parameter_state(session, field_name, raw_value)
        after_restore = transport.read_state(session)
        return {
            "schema_version": "r3dmatch_rcp2_write_smoke_v1",
            "started_at": started_at,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "transport_mode": transport.mode,
            "transport_kind": resolved_kind,
            "host": host,
            "port": int(port),
            "camera_label": camera_label,
            "parameter": field_name,
            "before_state": before_state,
            "probe_raw_value": probe_value,
            "probe_value": round(float(applied.value), 6),
            "after_probe": after_probe,
            "restored_raw_value": raw_value,
            "restored_value": round(float(restored.value), 6),
            "after_restore": after_restore,
            "restored_matches_original": int(((after_restore.get("raw_parameters") or {}).get(field_name) or {}).get("raw_value", 0) or 0)
            == raw_value,
        }
    finally:
        if session is not None:
            transport.close(session)


def load_apply_targets(
    payload_path: str,
    *,
    requested_cameras: Optional[Sequence[str]] = None,
    camera_ip_map: Optional[Dict[str, str]] = None,
) -> List[ApplyTarget]:
    path = Path(payload_path).expanduser().resolve()
    payload = _read_json(path)
    resolved_map = camera_ip_map or DEFAULT_CAMERA_IP_MAP
    targets: List[ApplyTarget] = []

    if payload.get("schema_version") == "r3dmatch_rcp2_ready_v1":
        inventory_label = _inventory_label_from_payload(payload)
        if not inventory_label:
            raise Rcp2ApplyError(f"Could not resolve inventory camera label from {path}.")
        targets.append(
            ApplyTarget(
                camera_id=str(payload.get("camera_id") or inventory_label),
                clip_id=str(payload.get("clip_id") or ""),
                inventory_camera_label=inventory_label,
                inventory_camera_ip=str(payload.get("inventory_camera_ip") or resolved_map.get(inventory_label) or ""),
                source_path=str(payload.get("source_path") or ""),
                calibration=dict(payload.get("calibration") or {}),
                confidence=float(payload.get("confidence", 0.0) or 0.0),
                is_hero_camera=bool(payload.get("is_hero_camera")),
                notes=[str(item) for item in payload.get("notes", []) if str(item).strip()],
                payload_path=str(path),
            )
        )
    elif payload.get("schema_version") == "r3dmatch_calibration_commit_payload_v1":
        for item in payload.get("per_camera_payloads", []):
            per_path = Path(str(item.get("path") or "")).expanduser()
            if not per_path.is_absolute():
                per_path = (path.parent / per_path).resolve()
            per_payload = _read_json(per_path)
            inventory_label = _inventory_label_from_payload(per_payload)
            if not inventory_label:
                raise Rcp2ApplyError(f"Could not resolve inventory camera label from {per_path}.")
            targets.append(
                ApplyTarget(
                    camera_id=str(per_payload.get("camera_id") or inventory_label),
                    clip_id=str(per_payload.get("clip_id") or ""),
                    inventory_camera_label=inventory_label,
                    inventory_camera_ip=str(per_payload.get("inventory_camera_ip") or resolved_map.get(inventory_label) or ""),
                    source_path=str(per_payload.get("source_path") or ""),
                    calibration=dict(per_payload.get("calibration") or {}),
                    confidence=float(per_payload.get("confidence", 0.0) or 0.0),
                    is_hero_camera=bool(per_payload.get("is_hero_camera")),
                    notes=[str(note) for note in per_payload.get("notes", []) if str(note).strip()],
                    payload_path=str(per_path),
                )
            )
    else:
        raise Rcp2ApplyError(f"Unsupported calibration payload schema in {path}.")

    if requested_cameras:
        requested_tokens = {_normalize_selection_token(item) for item in requested_cameras if str(item).strip()}
        selected: List[ApplyTarget] = []
        for target in targets:
            if (
                _normalize_selection_token(target.inventory_camera_label) in requested_tokens
                or _normalize_selection_token(target.camera_id) in requested_tokens
                or _normalize_selection_token(target.clip_id) in requested_tokens
            ):
                selected.append(target)
        missing = sorted(
            requested_tokens
            - {_normalize_selection_token(target.inventory_camera_label) for target in selected}
            - {_normalize_selection_token(target.camera_id) for target in selected}
            - {_normalize_selection_token(target.clip_id) for target in selected}
        )
        if missing:
            raise Rcp2ApplyError(f"Requested camera selections were not found in the commit payload: {', '.join(missing)}")
        targets = selected

    if not targets:
        raise Rcp2ApplyError("No calibration targets were selected for apply.")
    return targets


def _verification_result(requested: Dict[str, object], readback: Dict[str, object]) -> Dict[str, object]:
    return _legacy_verification_from_transaction(requested, readback)


def _generic_apply_state_transactionally(
    transport: Rcp2Transport,
    session: object,
    desired: Dict[str, object],
    *,
    stop_on_failure: bool = True,
) -> Dict[str, object]:
    before_state = dict(transport.read_state(session) or {})
    original_before_state = dict(before_state)
    working_state = dict(before_state)
    per_field: Dict[str, Dict[str, object]] = {}
    write_sequence_trace: List[Dict[str, object]] = []
    for index, field_name in enumerate(SAFE_FIELD_ORDER, start=1):
        next_values = dict(working_state)
        next_values[field_name] = desired[field_name]
        try:
            transport.write_state(session, next_values)
            readback = dict(transport.read_state(session) or {})
            result = _build_field_verification(
                field_name=field_name,
                requested_value=desired[field_name],
                applied_value=readback.get(field_name),
                parameter_id=PARAMETER_SPECS[field_name].parameter_id,
                set_action="transport_write_state",
            )
            working_state = dict(readback)
        except Rcp2TimeoutError as exc:
            result = _build_field_verification(
                field_name=field_name,
                requested_value=desired[field_name],
                applied_value=None,
                parameter_id=PARAMETER_SPECS[field_name].parameter_id,
                verification_status="TIMEOUT",
                note=str(exc),
                set_action="transport_write_state",
            )
        result["step"] = index
        result["before_value"] = _coerce_field_value(field_name, before_state.get(field_name))
        per_field[field_name] = result
        write_sequence_trace.append(dict(result))
        before_state[field_name] = working_state.get(field_name, before_state.get(field_name))
        if stop_on_failure and str(result.get("verification_status") or "") not in {"EXACT_MATCH", "WITHIN_TOLERANCE"}:
            break
    final_readback = dict(transport.read_state(session) or {})
    summary = _summarize_field_verifications(per_field)
    return {
        "before_state": original_before_state,
        "final_readback": final_readback,
        "per_field": per_field,
        "write_sequence_trace": write_sequence_trace,
        "verification_summary": summary,
        "final_status": summary["final_status"],
        "histogram_guard": {"enabled": False, "status": "not_run"},
        "clip_metadata_cross_check": {"status": "not_run"},
    }


def _apply_state_transactionally(
    transport: Rcp2Transport,
    session: object,
    desired: Dict[str, object],
    *,
    stop_on_failure: bool = True,
    enable_histogram_guard: bool = False,
    include_clip_metadata_cross_check: bool = False,
) -> Dict[str, object]:
    transactional = getattr(transport, "apply_state_transactionally", None)
    if callable(transactional):
        return transactional(
            session,
            desired,
            stop_on_failure=stop_on_failure,
            enable_histogram_guard=enable_histogram_guard,
            include_clip_metadata_cross_check=include_clip_metadata_cross_check,
        )
    return _generic_apply_state_transactionally(transport, session, desired, stop_on_failure=stop_on_failure)


def operator_apply_tolerances() -> Dict[str, float]:
    return {
        "exposureAdjust": OPERATOR_EXPOSURE_TOLERANCE,
        "kelvin": OPERATOR_KELVIN_TOLERANCE,
        "tint": OPERATOR_TINT_TOLERANCE,
    }


def verification_tolerances() -> Dict[str, float]:
    return dict(operator_apply_tolerances())


def _state_comparison(
    expected: Dict[str, object],
    actual: Dict[str, object],
    *,
    tolerances: Optional[Dict[str, float]] = None,
) -> Dict[str, object]:
    resolved_tolerances = dict(verification_tolerances())
    if tolerances:
        resolved_tolerances.update({str(key): float(value) for key, value in tolerances.items()})
    deviations = {
        "exposureAdjust": round(float(actual.get("exposureAdjust", 0.0) or 0.0) - float(expected.get("exposureAdjust", 0.0) or 0.0), 6),
        "kelvin": int(round(float(actual.get("kelvin", 0.0) or 0.0) - float(expected.get("kelvin", 0.0) or 0.0))),
        "tint": round(float(actual.get("tint", 0.0) or 0.0) - float(expected.get("tint", 0.0) or 0.0), 6),
    }
    within_tolerance = (
        abs(float(deviations["exposureAdjust"])) <= resolved_tolerances["exposureAdjust"]
        and abs(float(deviations["kelvin"])) <= resolved_tolerances["kelvin"]
        and abs(float(deviations["tint"])) <= resolved_tolerances["tint"]
    )
    exact_match = deviations["exposureAdjust"] == 0.0 and deviations["kelvin"] == 0 and deviations["tint"] == 0.0
    field_statuses = {
        "exposureAdjust": "exact"
        if deviations["exposureAdjust"] == 0.0
        else "within_tolerance"
        if abs(float(deviations["exposureAdjust"])) <= resolved_tolerances["exposureAdjust"]
        else "mismatch",
        "kelvin": "exact"
        if deviations["kelvin"] == 0
        else "within_tolerance"
        if abs(float(deviations["kelvin"])) <= resolved_tolerances["kelvin"]
        else "mismatch",
        "tint": "exact"
        if deviations["tint"] == 0.0
        else "within_tolerance"
        if abs(float(deviations["tint"])) <= resolved_tolerances["tint"]
        else "mismatch",
    }
    deviation_ratios = {
        key: 0.0 if float(resolved_tolerances[key]) == 0.0 else min(abs(float(value)) / float(resolved_tolerances[key]), 1.5)
        for key, value in deviations.items()
    }
    worst_ratio = max((float(value) for value in deviation_ratios.values()), default=0.0)
    return {
        "matches_within_tolerance": within_tolerance,
        "exact_match": exact_match,
        "deviations": deviations,
        "tolerances": resolved_tolerances,
        "field_statuses": field_statuses,
        "deviation_ratios": deviation_ratios,
        "verification_confidence_score": round(max(0.0, 1.0 - min(worst_ratio, 1.0)), 3),
    }


def classify_operator_apply_result(
    result_row: Dict[str, object],
    *,
    tolerances: Optional[Dict[str, float]] = None,
) -> Dict[str, object]:
    raw_status = str(result_row.get("status") or "")
    resolved_tolerances = dict(operator_apply_tolerances())
    if tolerances:
        resolved_tolerances.update({str(key): float(value) for key, value in tolerances.items()})
    requested = dict(result_row.get("requested") or {})
    readback = dict(result_row.get("readback") or {})

    if raw_status in {"dry_run", "failed_to_connect", "timeout", "rejected_by_camera", "invalid_payload", "invalid_input"}:
        return {
            "raw_status": raw_status,
            "display_status": raw_status,
            "matches_within_tolerance": raw_status == "dry_run",
            "deviations": {},
            "tolerances": resolved_tolerances,
        }

    if not requested or not readback:
        return {
            "raw_status": raw_status,
            "display_status": raw_status or "unknown",
            "matches_within_tolerance": False,
            "deviations": {},
            "tolerances": resolved_tolerances,
        }

    comparison = _state_comparison(requested, readback, tolerances=resolved_tolerances)
    deviations = dict(comparison["deviations"])
    within_tolerance = bool(comparison["matches_within_tolerance"])
    exact_match = bool(comparison["exact_match"])
    if within_tolerance and exact_match:
        display_status = "applied_successfully"
    elif within_tolerance:
        display_status = "applied_with_deviation"
    else:
        display_status = "mismatch_after_writeback"
    return {
        "raw_status": raw_status,
        "display_status": display_status,
        "matches_within_tolerance": within_tolerance,
        "deviations": deviations,
        "tolerances": resolved_tolerances,
    }


def build_camera_verification_report(
    payload_path: str,
    *,
    out_path: Optional[str] = None,
    requested_cameras: Optional[Sequence[str]] = None,
    camera_state_report: Optional[Dict[str, object]] = None,
    transport_kind: str = DEFAULT_TRANSPORT_KIND,
    live_read: bool = False,
    port: int = DEFAULT_RCP2_PORT,
    sdk_root: str = DEFAULT_RCP2_SDK_ROOT,
) -> Dict[str, object]:
    resolved_tolerances = verification_tolerances()
    targets = load_apply_targets(payload_path, requested_cameras=requested_cameras)
    started_at = datetime.now(timezone.utc).isoformat()
    actual_rows_by_label: Dict[str, Dict[str, object]] = {}

    verification_mode = "simulated_expected_state"
    if live_read:
        verification_mode = "live_read_compare"
        live_rows: List[Dict[str, object]] = []
        for target in targets:
            host = str(target.inventory_camera_ip or "").strip()
            if not host:
                live_rows.append(
                    {
                        "inventory_camera_label": target.inventory_camera_label,
                        "status": "not_reachable",
                        "error": "No inventory camera IP is mapped for this target.",
                        "state": {},
                    }
                )
                continue
            try:
                snapshot = read_camera_state(
                    host=host,
                    port=port,
                    camera_label=target.inventory_camera_label or target.camera_id or "LIVE",
                    transport_kind=transport_kind,
                    sdk_root=sdk_root,
                )
                live_rows.append(
                    {
                        "inventory_camera_label": target.inventory_camera_label,
                        "status": "success",
                        "error": "",
                        "state": dict(snapshot.get("state") or {}),
                    }
                )
            except Exception as exc:
                live_rows.append(
                    {
                        "inventory_camera_label": target.inventory_camera_label,
                        "status": "not_reachable",
                        "error": str(exc),
                        "state": {},
                    }
                )
        camera_state_report = {
            "schema_version": "r3dmatch_rcp2_camera_state_report_v1",
            "camera_count": len(live_rows),
            "connected_camera_count": sum(1 for row in live_rows if str(row.get("status") or "") == "success"),
            "results": live_rows,
        }
    elif camera_state_report:
        verification_mode = "camera_state_report_compare"

    for row in list((camera_state_report or {}).get("results") or []):
        label = str(row.get("inventory_camera_label") or "")
        if label:
            actual_rows_by_label[label] = dict(row)

    result_rows: List[Dict[str, object]] = []
    verified_camera_count = 0
    within_tolerance_camera_count = 0
    mismatch_camera_count = 0
    unavailable_camera_count = 0
    for target in targets:
        expected = clamp_calibration_values(target.calibration)
        actual_row = dict(actual_rows_by_label.get(target.inventory_camera_label) or {})
        actual_state = dict(actual_row.get("state") or {})
        actual_status = str(actual_row.get("status") or "")
        comparison = _state_comparison(expected, actual_state, tolerances=resolved_tolerances) if actual_status == "success" and actual_state else None
        if comparison is None:
            row_status = "simulated_expected" if verification_mode == "simulated_expected_state" else "not_reachable"
            verification_level = "NOT_AVAILABLE"
            verification_badge = "❓"
            confidence_score = 0.2 if verification_mode == "simulated_expected_state" else 0.0
            confidence_label = "simulated_only" if verification_mode == "simulated_expected_state" else "not_available"
            if row_status == "not_reachable":
                unavailable_camera_count += 1
        elif comparison["exact_match"]:
            row_status = "verified"
            verification_level = "VERIFIED"
            verification_badge = "✅"
            confidence_score = 1.0
            confidence_label = "strong"
            verified_camera_count += 1
        elif comparison["matches_within_tolerance"]:
            row_status = "within_tolerance"
            verification_level = "WITHIN_TOLERANCE"
            verification_badge = "⚠️"
            confidence_score = float(comparison.get("verification_confidence_score", 0.75) or 0.75)
            confidence_label = "tolerance_match"
            within_tolerance_camera_count += 1
        else:
            row_status = "mismatch"
            verification_level = "MISMATCH"
            verification_badge = "❌"
            confidence_score = 0.0
            confidence_label = "failed"
            mismatch_camera_count += 1
        result_rows.append(
            {
                "camera_id": target.camera_id,
                "clip_id": target.clip_id,
                "inventory_camera_label": target.inventory_camera_label,
                "inventory_camera_ip": target.inventory_camera_ip,
                "reference_use": "Excluded" if "excluded" in " ".join(target.notes).lower() else "Included",
                "expected_state": expected,
                "actual_state": actual_state,
                "actual_state_status": actual_status or ("simulated" if verification_mode == "simulated_expected_state" else "not_run"),
                "verification_status": row_status,
                "verification_level": verification_level,
                "verification_badge": verification_badge,
                "verification_confidence_score": confidence_score,
                "verification_confidence_label": confidence_label,
                "comparison": comparison or {},
                "notes": list(target.notes) + ([str(actual_row.get("error") or "").strip()] if str(actual_row.get("error") or "").strip() else []),
            }
        )

    if verification_mode == "simulated_expected_state":
        status = "simulated"
        notes = [
            "This verification report confirms the intended camera targets and tolerance model only.",
            "No fresh camera state was queried, so this report does not prove hardware state.",
        ]
    else:
        if mismatch_camera_count:
            status = "failed" if verified_camera_count == 0 else "partial"
        elif unavailable_camera_count:
            status = "partial" if verified_camera_count else "warning"
        else:
            status = "success"
        notes = [
            "Expected camera targets were compared against read-only camera state.",
            "No write commands were sent during this verification pass.",
        ]

    payload = {
        "schema_version": VERIFICATION_REPORT_SCHEMA,
        "started_at": started_at,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "payload_path": str(Path(payload_path).expanduser().resolve()),
        "verification_mode": verification_mode,
        "transport_kind": normalize_transport_kind(transport_kind),
        "camera_count": len(result_rows),
        "verified_camera_count": verified_camera_count,
        "within_tolerance_camera_count": within_tolerance_camera_count,
        "mismatch_camera_count": mismatch_camera_count,
        "unavailable_camera_count": unavailable_camera_count,
        "tolerances": resolved_tolerances,
        "status": status,
        "notes": notes,
        "results": result_rows,
    }
    if camera_state_report:
        payload["camera_state_report"] = {
            "schema_version": str(camera_state_report.get("schema_version") or ""),
            "camera_count": int(camera_state_report.get("camera_count", 0) or 0),
            "connected_camera_count": int(camera_state_report.get("connected_camera_count", 0) or 0),
            "report_path": str(camera_state_report.get("report_path") or ""),
        }
    if out_path:
        output_path = Path(out_path).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        payload["report_path"] = str(output_path)
    return payload


def summarize_apply_report(
    report: Dict[str, object],
    *,
    tolerances: Optional[Dict[str, float]] = None,
) -> Dict[str, object]:
    rows = [dict(item) for item in report.get("results", []) if isinstance(item, dict)]
    operator_counts: Dict[str, int] = {}
    enhanced_rows: List[Dict[str, object]] = []
    for row in rows:
        operator_result = classify_operator_apply_result(row, tolerances=tolerances)
        row["operator_result"] = operator_result
        display_status = str(operator_result.get("display_status") or "unknown")
        operator_counts[display_status] = operator_counts.get(display_status, 0) + 1
        enhanced_rows.append(row)
    operator_status = "success"
    if operator_counts:
        if any(
            key in {"failed_to_connect", "timeout", "rejected_by_camera", "mismatch_after_writeback", "invalid_payload", "invalid_input"}
            for key in operator_counts
        ):
            operator_status = "partial" if any(key in {"applied_successfully", "applied_with_deviation", "dry_run"} for key in operator_counts) else "failed"
        elif set(operator_counts) == {"dry_run"}:
            operator_status = "dry_run"
    return {
        "operator_status": operator_status,
        "operator_status_counts": operator_counts,
        "operator_tolerances": tolerances or operator_apply_tolerances(),
        "results": enhanced_rows,
    }


def apply_calibration_payload(
    payload_path: str,
    *,
    out_path: Optional[str] = None,
    requested_cameras: Optional[Sequence[str]] = None,
    transport: Optional[Rcp2Transport] = None,
    port: int = DEFAULT_RCP2_PORT,
    camera_ip_map: Optional[Dict[str, str]] = None,
    live: bool = False,
    sdk_root: str = DEFAULT_RCP2_SDK_ROOT,
    transport_kind: str = DEFAULT_TRANSPORT_KIND,
    retry_count: int = DEFAULT_APPLY_RETRY_COUNT,
    enable_histogram_guard: bool = False,
    include_clip_metadata_cross_check: bool = False,
) -> Dict[str, object]:
    resolved_map = camera_ip_map or DEFAULT_CAMERA_IP_MAP
    targets = load_apply_targets(payload_path, requested_cameras=requested_cameras, camera_ip_map=resolved_map)
    active_transport = transport or (create_live_transport(transport_kind=transport_kind, sdk_root=sdk_root) if live else DryRunRcp2Transport())
    resolved_transport_kind = normalize_transport_kind(transport_kind) if live else "dry_run"
    sdk_probe = probe_rcp2_sdk(
        sdk_root=sdk_root,
        ensure_live_bridge=live and getattr(active_transport, "mode", "") == "live_raw",
    )
    started_at = datetime.now(timezone.utc).isoformat()
    results: List[Dict[str, object]] = []

    for target in targets:
        host = target.inventory_camera_ip or resolved_map.get(target.inventory_camera_label, "")
        prepared = _prepare_calibration_values(target.calibration)
        desired = dict(prepared["requested"])
        raw_requested = dict(prepared["raw_requested"])
        safety_warnings = list(prepared["warnings"])
        attempts: List[Dict[str, object]] = []
        if not host:
            results.append(
                {
                    "camera_id": target.camera_id,
                    "inventory_camera_label": target.inventory_camera_label,
                    "status": "failed_to_connect",
                    "failure_reason": "failed_to_connect",
                    "requested": desired,
                    "raw_requested": raw_requested,
                    "warnings": safety_warnings,
                    "error": f"No IP mapping is available for inventory camera {target.inventory_camera_label}.",
                    "attempts": attempts,
                }
            )
            continue
        if prepared["invalid_reasons"]:
            results.append(
                {
                    "camera_id": target.camera_id,
                    "clip_id": target.clip_id,
                    "inventory_camera_label": target.inventory_camera_label,
                    "inventory_camera_ip": host,
                    "status": "invalid_payload",
                    "failure_reason": "invalid_payload",
                    "requested": desired,
                    "raw_requested": raw_requested,
                    "warnings": safety_warnings + list(prepared["invalid_reasons"]),
                    "error": "; ".join(str(item) for item in prepared["invalid_reasons"]),
                    "attempts": attempts,
                    "confidence": target.confidence,
                    "is_hero_camera": target.is_hero_camera,
                    "payload_path": target.payload_path,
                    "notes": list(target.notes),
                }
            )
            continue
        final_row: Optional[Dict[str, object]] = None
        for attempt_number in range(1, retry_count + 2):
            session = None
            try:
                session = active_transport.open(host=host, port=port, camera_label=target.inventory_camera_label)
                transaction = _apply_state_transactionally(
                    active_transport,
                    session,
                    desired,
                    stop_on_failure=True,
                    enable_histogram_guard=enable_histogram_guard,
                    include_clip_metadata_cross_check=include_clip_metadata_cross_check,
                )
                before_state = dict(transaction.get("before_state") or {})
                readback = dict(transaction.get("final_readback") or {})
                verification = _legacy_verification_from_transaction(desired, readback, transaction)
                status = _apply_status_from_transaction(transaction, transport_mode=getattr(active_transport, "mode", ""))
                failure_reason = _failure_reason_from_transaction(transaction, status=status)
                attempts.append(
                    {
                        "attempt": attempt_number,
                        "status": status,
                        "failure_reason": failure_reason,
                    }
                )
                final_row = {
                    "camera_id": target.camera_id,
                    "clip_id": target.clip_id,
                    "inventory_camera_label": target.inventory_camera_label,
                    "inventory_camera_ip": host,
                    "status": status,
                    "failure_reason": failure_reason,
                    "requested": desired,
                    "raw_requested": raw_requested,
                    "before_state": before_state,
                    "readback": readback,
                    "verification": verification,
                    "camera_verification": {
                        "per_field": dict(transaction.get("per_field") or {}),
                        "write_sequence_trace": list(transaction.get("write_sequence_trace") or []),
                        "verification_summary": dict(transaction.get("verification_summary") or {}),
                        "final_status": str(transaction.get("final_status") or ""),
                        "histogram_guard": dict(transaction.get("histogram_guard") or {}),
                        "clip_metadata_cross_check": dict(transaction.get("clip_metadata_cross_check") or {}),
                    },
                    "warnings": safety_warnings,
                    "confidence": target.confidence,
                    "is_hero_camera": target.is_hero_camera,
                    "payload_path": target.payload_path,
                    "notes": list(target.notes),
                    "attempts": attempts,
                }
                break
            except (Rcp2ConnectionError, Rcp2TimeoutError, Rcp2RejectedError, Rcp2InvalidPayloadError) as exc:
                failure_reason = _failure_reason_for_exception(exc)
                attempts.append(
                    {
                        "attempt": attempt_number,
                        "status": _status_for_failure_reason(failure_reason),
                        "failure_reason": failure_reason,
                        "error": str(exc),
                    }
                )
                if _should_retry_failure(failure_reason, attempt_number, retry_count):
                    continue
                final_row = {
                    "camera_id": target.camera_id,
                    "clip_id": target.clip_id,
                    "inventory_camera_label": target.inventory_camera_label,
                    "inventory_camera_ip": host,
                    "status": _status_for_failure_reason(failure_reason),
                    "failure_reason": failure_reason,
                    "requested": desired,
                    "raw_requested": raw_requested,
                    "warnings": safety_warnings,
                    "error": str(exc),
                    "confidence": target.confidence,
                    "is_hero_camera": target.is_hero_camera,
                    "payload_path": target.payload_path,
                    "notes": list(target.notes),
                    "attempts": attempts,
                }
                break
            finally:
                if session is not None:
                    active_transport.close(session)
        if final_row is not None:
            results.append(final_row)

    status_counts: Dict[str, int] = {}
    for item in results:
        key = str(item.get("status") or "unknown")
        status_counts[key] = status_counts.get(key, 0) + 1
    overall_status = "success"
    if any(item.get("status") in {"failed_to_connect", "timeout", "rejected_by_camera", "mismatch_after_writeback", "invalid_payload", "invalid_input"} for item in results):
        overall_status = "partial" if any(item.get("status") in {"applied_successfully", "dry_run"} for item in results) else "failed"
    if active_transport.mode == "dry_run" and overall_status == "success":
        overall_status = "dry_run"

    operator_summary = summarize_apply_report({"results": results})
    summary = {
        "schema_version": "r3dmatch_rcp2_apply_report_v1",
        "started_at": started_at,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "payload_path": str(Path(payload_path).expanduser().resolve()),
        "transport_mode": active_transport.mode,
        "transport_kind": resolved_transport_kind,
        "rcp2_port": int(port),
        "sdk_probe": sdk_probe,
        "status": overall_status,
        "camera_count": len(results),
        "status_counts": status_counts,
        "operator_status": operator_summary["operator_status"],
        "operator_status_counts": operator_summary["operator_status_counts"],
        "operator_tolerances": operator_summary["operator_tolerances"],
        "results": operator_summary["results"],
    }
    transaction_summary_rows: List[Dict[str, object]] = []
    for row in operator_summary["results"]:
        camera_verification = dict(row.get("camera_verification") or {})
        transaction_summary_rows.append(
            {
                "camera_id": row.get("camera_id"),
                "clip_id": row.get("clip_id"),
                "inventory_camera_label": row.get("inventory_camera_label"),
                "requested": row.get("requested"),
                "readback": row.get("readback"),
                "camera_verification": camera_verification,
                "status": row.get("status"),
                "failure_reason": row.get("failure_reason"),
            }
        )
    post_apply_verification_payload = {
        "schema_version": TRANSACTIONAL_VERIFICATION_SCHEMA,
        "started_at": started_at,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "payload_path": str(Path(payload_path).expanduser().resolve()),
        "transport_mode": active_transport.mode,
        "transport_kind": resolved_transport_kind,
        "status": overall_status,
        "camera_count": len(transaction_summary_rows),
        "results": transaction_summary_rows,
    }
    summary["post_apply_verification"] = post_apply_verification_payload
    if out_path:
        output_path = Path(out_path).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        post_apply_path = _default_post_apply_verification_path(str(output_path))
        post_apply_path.write_text(json.dumps(post_apply_verification_payload, indent=2), encoding="utf-8")
        summary["report_path"] = str(output_path)
        summary["post_apply_verification_path"] = str(post_apply_path)
    return summary
