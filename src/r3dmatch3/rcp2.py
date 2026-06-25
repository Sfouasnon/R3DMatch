"""
rcp2.py — R3DMatch v2 RCP2 WebSocket push module
=================================================
Pushes calibration commit values to live RED KOMODO-X cameras
via the RCP2 JSON WebSocket protocol (port 9998).

Architecture:
- Pure Python asyncio + websockets, no C SDK dependency
- Reads dividers from live camera rcp_cur_int_edit_info responses
- Writes in fixed order: kelvin → tint → exposureAdjust
- Per-field verified readback after each write
- Dry-run mode emits full structured report without touching hardware
- Concurrent multi-camera push with per-camera status isolation

Parameter IDs (confirmed from RED KOMODO-X RCP2 Parameters, Rev C, 9-May-2025):
  RCP_PARAM_COLOR_TEMPERATURE   — integer Kelvin, divider=1 (raw = Kelvin)
  RCP_PARAM_TINT                — fixed-point, divider read live (typically 100;
                                   value 150 = tint +1.5)
  RCP_PARAM_EXPOSURE_ADJUST     — fixed-point stops, divider read live (typically 100;
                                   value -41 = -0.41 stops)

All three support: rcp_get, rcp_set, rcp_cur_int, rcp_set_relative, rcp_get_list
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Protocol constants ─────────────────────────────────────────────────────────

RCP2_PORT = 9998
CONNECT_TIMEOUT = 8.0       # seconds to establish WebSocket connection
HANDSHAKE_TIMEOUT = 6.0     # seconds to receive rcp_config acknowledgment
GET_TIMEOUT = 5.0           # seconds to wait for rcp_cur_int response
SET_VERIFY_TIMEOUT = 6.0    # seconds to wait for verified readback after rcp_set
MAX_RETRIES = 2             # per-camera retry attempts on connect/timeout failure

# Verification tolerances — match v1's validated values exactly (rcp2_apply.py:
# KELVIN_VERIFY_TOLERANCE=50, TINT_VERIFY_TOLERANCE=0.5, EXPOSURE_VERIFY_TOLERANCE
# =0.001). v1 was validated on a live KOMODO-X; the previous v4 values (0.02 tint,
# 0.02 exposure) did NOT match and made exposure verification 20x too loose.
TOL_KELVIN = 50             # ±50 K is WITHIN_TOLERANCE
TOL_TINT = 0.5              # ±0.5 tint units
TOL_EXPOSURE = 0.001        # ±0.001 stops

# Write order: camera-stable WB first, then exposure
_WRITE_ORDER = ["kelvin", "tint", "exposureAdjust"]

# Parameter ID map (abbreviated form — camera accepts both RCP_PARAM_X and X)
_PARAM_ID = {
    "kelvin":         "COLOR_TEMPERATURE",
    "tint":           "TINT",
    "exposureAdjust": "EXPOSURE_ADJUST",
}

# Known dividers from v1 validated sessions and RED SDK source.
# The camera reports authoritative dividers via rcp_cur_int_edit_info — we
# read those live and fall back to these only if the get times out.
_FALLBACK_DIVIDER = {
    "kelvin":         1,      # COLOR_TEMPERATURE: raw Kelvin integer, divider=1
    "tint":           1000,   # TINT: confirmed live on KOMODO-X — divider=1000,
                              # e.g. raw 1500 → 1.5 tint
    "exposureAdjust": 1000,   # EXPOSURE_ADJUST: confirmed live on KOMODO-X
                              # (FW 2.2.4): divider=1000,
                              # e.g. raw -410 → -0.41 stops
}

# ── Data classes ───────────────────────────────────────────────────────────────

class VerifyStatus(str, Enum):
    EXACT_MATCH      = "EXACT_MATCH"
    WITHIN_TOLERANCE = "WITHIN_TOLERANCE"
    MISMATCH         = "MISMATCH"
    TIMEOUT          = "TIMEOUT"
    OUT_OF_RANGE     = "OUT_OF_RANGE"   # target outside camera-reported min/max
    NOT_AVAILABLE    = "NOT_AVAILABLE"


@dataclass
class FieldResult:
    """Result for a single parameter write+verify cycle."""
    field: str                          # "kelvin" | "tint" | "exposureAdjust"
    param_id: str                       # e.g. "COLOR_TEMPERATURE"
    requested_value: float              # operator-facing value sent
    raw_sent: int                       # integer wired to camera
    divider: int                        # divider used for conversion
    readback_raw: Optional[int] = None  # raw integer read back
    readback_value: Optional[float] = None
    verify_status: VerifyStatus = VerifyStatus.NOT_AVAILABLE
    delta: Optional[float] = None       # |readback - requested|
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.verify_status in (
            VerifyStatus.EXACT_MATCH,
            VerifyStatus.WITHIN_TOLERANCE,
        )


@dataclass
class CameraStateSnapshot:
    """Camera parameter state before or after a push."""
    kelvin: Optional[float] = None
    tint: Optional[float] = None
    exposure_adjust: Optional[float] = None
    camera_id: Optional[str] = None
    camera_type: Optional[str] = None
    firmware_version: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class PushResult:
    """Full result for one camera push attempt."""
    camera_label: str
    camera_ip: str
    dry_run: bool

    # Timing
    started_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None

    # Camera identity (populated from CAMERA_INFO)
    camera_id: Optional[str] = None
    camera_type: Optional[str] = None
    firmware_version: Optional[str] = None

    # State snapshots
    state_before: Optional[CameraStateSnapshot] = None
    state_after: Optional[CameraStateSnapshot] = None

    # Per-field results (populated as each field is written)
    field_results: List[FieldResult] = field(default_factory=list)

    # Overall outcome
    success: bool = False
    aborted_at_field: Optional[str] = None   # which field caused abort
    error: Optional[str] = None
    retries: int = 0

    @property
    def elapsed_seconds(self) -> float:
        end = self.finished_at or time.time()
        return round(end - self.started_at, 3)

    def to_dict(self) -> Dict[str, Any]:
        def _fr(fr: FieldResult) -> Dict:
            return {
                "field": fr.field,
                "param_id": fr.param_id,
                "requested_value": fr.requested_value,
                "raw_sent": fr.raw_sent,
                "divider": fr.divider,
                "readback_raw": fr.readback_raw,
                "readback_value": fr.readback_value,
                "verify_status": fr.verify_status.value if fr.verify_status else None,
                "delta": fr.delta,
                "ok": fr.ok,
                "error": fr.error,
            }
        def _snap(s: Optional[CameraStateSnapshot]) -> Optional[Dict]:
            if s is None:
                return None
            return {
                "kelvin": s.kelvin,
                "tint": s.tint,
                "exposure_adjust": s.exposure_adjust,
                "camera_id": s.camera_id,
                "camera_type": s.camera_type,
                "firmware_version": s.firmware_version,
                "timestamp": s.timestamp,
            }
        return {
            "camera_label": self.camera_label,
            "camera_ip": self.camera_ip,
            "dry_run": self.dry_run,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "elapsed_seconds": self.elapsed_seconds,
            "camera_id": self.camera_id,
            "camera_type": self.camera_type,
            "firmware_version": self.firmware_version,
            "state_before": _snap(self.state_before),
            "state_after": _snap(self.state_after),
            "field_results": [_fr(f) for f in self.field_results],
            "success": self.success,
            "aborted_at_field": self.aborted_at_field,
            "error": self.error,
            "retries": self.retries,
        }


# ── Low-level WebSocket session ────────────────────────────────────────────────

class RCP2Session:
    """
    Single-camera RCP2 WebSocket session.

    Handles: connect → rcp_config → rcp_get_types → CAMERA_INFO
             → read current state → optionally write fields → verify readback
    """

    def __init__(
        self,
        ip: str,
        label: str,
        *,
        dry_run: bool = False,
        connect_timeout: float = CONNECT_TIMEOUT,
        handshake_timeout: float = HANDSHAKE_TIMEOUT,
    ):
        self.ip = ip
        self.label = label
        self.dry_run = dry_run
        self.connect_timeout = connect_timeout
        self.handshake_timeout = handshake_timeout
        self._ws = None
        self._recv_queue: asyncio.Queue = asyncio.Queue()
        self._listener_task: Optional[asyncio.Task] = None
        self._uri = f"ws://{ip}:{RCP2_PORT}"

    # ── Connection lifecycle ───────────────────────────────────────────────────

    async def connect(self) -> None:
        """Open WebSocket and complete RCP2 initialization handshake."""
        import websockets  # local import keeps module importable without install
        logger.debug("[%s] Connecting to %s", self.label, self._uri)
        self._ws = await asyncio.wait_for(
            websockets.connect(
                self._uri,
                ping_interval=None,   # we don't need WS-level pings
                ping_timeout=None,
                open_timeout=self.connect_timeout,
            ),
            timeout=self.connect_timeout,
        )
        # Start background listener
        self._listener_task = asyncio.create_task(self._listen())

        # 1. Send rcp_config
        await self._send({
            "type": "rcp_config",
            "strings_decoded": 1,
            "json_minified": 1,
            "include_cacheable_flags": 0,
            "encoding_type": "legacy",
            "client": {"name": "R3DMatch v2", "version": "2.0"},
        })

        # 2. Wait for config acknowledgment
        ack = await self._wait_for_type("rcp_config", timeout=self.handshake_timeout)
        logger.debug("[%s] Config ack: lang=%s", self.label, ack.get("lang"))

        # 3. Request types (protocol requirement — avoids hardcoded enum values)
        await self._send({"type": "rcp_get_types"})

        logger.debug("[%s] Session ready", self.label)

    async def close(self) -> None:
        if self._listener_task and not self._listener_task.done():
            self._listener_task.cancel()
            try:
                await self._listener_task
            except asyncio.CancelledError:
                pass
        if self._ws:
            await self._ws.close()
            self._ws = None

    # ── Camera info ───────────────────────────────────────────────────────────

    async def get_camera_info(self) -> Dict[str, Any]:
        """Fetch CAMERA_INFO — returns dict with name, serial_number, version, etc."""
        await self._send({"type": "rcp_get", "id": "CAMERA_INFO"})
        msg = await self._wait_for_type("rcp_cur_cam_info", timeout=GET_TIMEOUT)
        return msg

    # ── Parameter read ────────────────────────────────────────────────────────

    async def _get_param_raw(
        self, field: str
    ) -> Tuple[int, int, Optional[int], Optional[int]]:
        """
        Read a parameter and its edit_info.

        Returns (raw_value, divider, min_raw, max_raw). min/max are the camera's
        authoritative limits when reported, else None.
        """
        param_id = _PARAM_ID[field]
        await self._send({"type": "rcp_get", "id": param_id})

        # Wait for rcp_cur_int (integer param)
        msg = await self._wait_for_type(
            "rcp_cur_int", timeout=GET_TIMEOUT,
            filter_id=param_id
        )

        raw_val = msg.get("cur", {}).get("val", 0)
        edit_info = msg.get("edit_info", {})
        divider = edit_info.get("divider", _FALLBACK_DIVIDER[field])
        if not divider:
            divider = _FALLBACK_DIVIDER[field]
        min_raw = edit_info.get("min")
        max_raw = edit_info.get("max")
        logger.debug(
            "[%s] GET %s: raw=%d divider=%d min=%s max=%s",
            self.label, param_id, raw_val, divider, min_raw, max_raw
        )
        return raw_val, divider, min_raw, max_raw

    async def get_param(self, field: str) -> Tuple[float, int]:
        """
        Read current value of a parameter.

        Returns (operator_value, divider) tuple.
        operator_value = raw_int / divider
        """
        raw_val, divider, _min, _max = await self._get_param_raw(field)
        return raw_val / divider, divider

    async def read_state(self, dividers: Optional[Dict[str, int]] = None) -> CameraStateSnapshot:
        """
        Read current kelvin, tint, exposureAdjust from camera.
        dividers: cached dividers from a prior get_param call (avoids re-reading edit_info)
        """
        snap = CameraStateSnapshot()
        for field_name in _WRITE_ORDER:
            try:
                val, div = await self.get_param(field_name)
                if field_name == "kelvin":
                    snap.kelvin = val
                elif field_name == "tint":
                    snap.tint = val
                elif field_name == "exposureAdjust":
                    snap.exposure_adjust = val
            except asyncio.TimeoutError:
                logger.warning("[%s] Timeout reading %s", self.label, field_name)
            except Exception as exc:
                logger.warning("[%s] Error reading %s: %s", self.label, field_name, exc)
        return snap

    # ── Parameter write + verify ───────────────────────────────────────────────

    async def set_and_verify(
        self,
        field: str,
        requested_value: float,
        divider: int,
        *,
        min_raw: Optional[int] = None,
        max_raw: Optional[int] = None,
    ) -> FieldResult:
        """
        Write one parameter and verify with an INDEPENDENT readback.

        Steps:
        1. Convert to raw integer: round(requested_value * divider)
        2. Enforce the camera-reported min/max (reject out-of-range — do NOT
           silently clamp). Fall back to static safe bounds only when the camera
           did not report limits.
        3. Send rcp_set.
        4. Consume the set echo, then issue a fresh rcp_get and use THAT as the
           authoritative readback (matches v1 — never trust only the set echo).
        5. Classify: EXACT_MATCH / WITHIN_TOLERANCE / MISMATCH / TIMEOUT.
        """
        param_id = _PARAM_ID[field]
        raw_to_send = int(round(requested_value * divider))

        result = FieldResult(
            field=field,
            param_id=param_id,
            requested_value=requested_value,
            raw_sent=raw_to_send,
            divider=divider,
        )

        # ── Enforce limits (reject, don't clamp) ────────────────────────────────
        lo_static, hi_static = _STATIC_RAW_BOUNDS.get(field, (None, None))
        lo = min_raw if min_raw is not None else lo_static
        hi = max_raw if max_raw is not None else hi_static
        if (lo is not None and raw_to_send < lo) or (hi is not None and raw_to_send > hi):
            src = "camera" if (min_raw is not None or max_raw is not None) else "safe"
            result.verify_status = VerifyStatus.OUT_OF_RANGE
            result.error = (
                f"target raw {raw_to_send} ({requested_value:.4f}) outside "
                f"{src} range [{lo}, {hi}]")
            logger.error(
                "[%s] %s target %d outside %s range [%s, %s] — REJECTED",
                self.label, param_id, raw_to_send, src, lo, hi)
            return result

        if self.dry_run:
            # Simulate success without touching hardware
            result.readback_raw = raw_to_send
            result.readback_value = raw_to_send / divider
            result.delta = 0.0
            result.verify_status = VerifyStatus.EXACT_MATCH
            logger.info(
                "[%s] DRY-RUN set %s = %.4f (raw %d, divider %d)",
                self.label, param_id, requested_value, raw_to_send, divider
            )
            return result

        logger.info(
            "[%s] SET %s = %.4f (raw %d, divider %d)",
            self.label, param_id, requested_value, raw_to_send, divider
        )

        # Clear any stale messages for this param_id, then write.
        self._drain_param(param_id)
        await self._send({
            "type": "rcp_set",
            "id": param_id,
            "value": raw_to_send,
        })

        # Consume the camera's set-echo (best effort) — but do NOT trust it as the
        # verification. We re-read with an independent rcp_get below.
        try:
            await self._wait_for_type(
                "rcp_cur_int", timeout=SET_VERIFY_TIMEOUT, filter_id=param_id)
        except asyncio.TimeoutError:
            pass

        # ── Independent confirmation read ───────────────────────────────────────
        self._drain_param(param_id)
        try:
            rb_raw, rb_div, _, _ = await self._get_param_raw(field)
            result.readback_raw = rb_raw
            result.readback_value = rb_raw / rb_div
            intended = raw_to_send / divider          # quantized target we sent
            result.delta = abs(result.readback_value - intended)
            if rb_raw == raw_to_send:
                result.verify_status = VerifyStatus.EXACT_MATCH
            else:
                result.verify_status = _classify(field, result.delta)
        except asyncio.TimeoutError:
            result.verify_status = VerifyStatus.TIMEOUT
            result.error = f"No independent readback within {GET_TIMEOUT}s"
            logger.warning("[%s] TIMEOUT verifying %s", self.label, param_id)

        logger.info(
            "[%s] VERIFY %s: sent=%.4f readback=%.4f Δ=%.4f → %s",
            self.label, param_id,
            requested_value,
            result.readback_value if result.readback_value is not None else float("nan"),
            result.delta if result.delta is not None else float("nan"),
            result.verify_status.value,
        )
        return result

    # ── Internal messaging ────────────────────────────────────────────────────

    async def _send(self, obj: Dict) -> None:
        if self._ws is None:
            raise RuntimeError("WebSocket not connected")
        await self._ws.send(json.dumps(obj))

    async def _listen(self) -> None:
        """Background coroutine — pumps incoming messages into _recv_queue."""
        try:
            async for raw in self._ws:
                try:
                    msg = json.loads(raw)
                    await self._recv_queue.put(msg)
                except json.JSONDecodeError:
                    logger.debug("[%s] Non-JSON message: %r", self.label, raw[:80])
        except Exception as exc:
            logger.debug("[%s] Listener exited: %s", self.label, exc)
            # Signal EOF so waiters unblock
            await self._recv_queue.put({"type": "__eof__"})

    async def _wait_for_type(
        self,
        msg_type: str,
        timeout: float,
        filter_id: Optional[str] = None,
    ) -> Dict:
        """
        Wait for a message of the given type from the queue.
        filter_id: if set, also require msg["id"] == filter_id (case-insensitive,
                   both abbreviated and RCP_PARAM_ prefixed forms accepted).
        Raises asyncio.TimeoutError on deadline.
        """
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise asyncio.TimeoutError(
                    f"Timed out waiting for {msg_type} (filter_id={filter_id})"
                )
            try:
                msg = await asyncio.wait_for(
                    self._recv_queue.get(), timeout=remaining
                )
            except asyncio.TimeoutError:
                raise asyncio.TimeoutError(
                    f"Timed out waiting for {msg_type} (filter_id={filter_id})"
                )

            if msg.get("type") == "__eof__":
                raise ConnectionError("WebSocket closed unexpectedly")

            if msg.get("type") != msg_type:
                # Put back non-matching messages so other waiters see them.
                # (Simple approach — sufficient for sequential single-camera use.)
                await self._recv_queue.put(msg)
                await asyncio.sleep(0.005)
                continue

            if filter_id is not None:
                msg_id = msg.get("id", "")
                # Accept both "COLOR_TEMPERATURE" and "RCP_PARAM_COLOR_TEMPERATURE"
                if not _ids_match(msg_id, filter_id):
                    await self._recv_queue.put(msg)
                    await asyncio.sleep(0.005)
                    continue

            return msg

    def _drain_param(self, param_id: str) -> None:
        """Remove stale rcp_cur_int messages for param_id from queue."""
        stale = []
        while not self._recv_queue.empty():
            try:
                msg = self._recv_queue.get_nowait()
                if msg.get("type") == "rcp_cur_int" and _ids_match(
                    msg.get("id", ""), param_id
                ):
                    continue   # discard
                stale.append(msg)
            except asyncio.QueueEmpty:
                break
        for msg in stale:
            self._recv_queue.put_nowait(msg)


# ── High-level push logic ─────────────────────────────────────────────────────

async def push_camera(
    ip: str,
    camera_label: str,
    commit: "CommitLike",
    *,
    dry_run: bool = False,
    progress_cb=None,
) -> PushResult:
    """
    Push calibration commit values to one camera.

    commit must expose: .kelvin (int), .tint (float), .exposure_adjust (float)

    progress_cb: optional async callable(camera_label, phase, detail)
    Returns PushResult.
    """
    result = PushResult(
        camera_label=camera_label,
        camera_ip=ip,
        dry_run=dry_run,
    )

    async def _cb(phase: str, detail: str = "") -> None:
        if progress_cb:
            try:
                await progress_cb(camera_label, phase, detail)
            except Exception:
                pass

    for attempt in range(MAX_RETRIES + 1):
        if attempt > 0:
            result.retries = attempt
            logger.info("[%s] Retry %d/%d", camera_label, attempt, MAX_RETRIES)
            await asyncio.sleep(1.5)

        session = RCP2Session(ip, camera_label, dry_run=dry_run)
        try:
            # ── Connect ──────────────────────────────────────────────────────
            await _cb("connect", f"ws://{ip}:{RCP2_PORT}")
            await session.connect()

            # ── Camera info ──────────────────────────────────────────────────
            await _cb("camera_info")
            try:
                info = await session.get_camera_info()
                result.camera_id = info.get("serial_number")
                result.camera_type = info.get("camera_type", {}).get("str")
                result.firmware_version = info.get("version", {}).get("str")
                logger.info(
                    "[%s] Camera: %s  FW: %s",
                    camera_label, result.camera_type, result.firmware_version
                )
            except asyncio.TimeoutError:
                logger.warning("[%s] CAMERA_INFO timed out — continuing", camera_label)

            # ── Read state before ─────────────────────────────────────────────
            await _cb("read_before")
            state_before = await session.read_state()
            state_before.camera_id = result.camera_id
            state_before.camera_type = result.camera_type
            state_before.firmware_version = result.firmware_version
            result.state_before = state_before
            logger.info(
                "[%s] Before: K=%s T=%s EA=%s",
                camera_label,
                _fmt(state_before.kelvin), _fmt(state_before.tint),
                _fmt(state_before.exposure_adjust),
            )

            if dry_run:
                logger.info("[%s] DRY-RUN — skipping live writes", camera_label)

            # ── Build value map ───────────────────────────────────────────────
            values = {
                "kelvin":         float(commit.kelvin),
                "tint":           float(commit.tint),
                "exposureAdjust": float(commit.exposure_adjust),
            }

            # Exposure-only commits push ONLY exposureAdjust — white balance on
            # the camera is left untouched.
            write_order = (["exposureAdjust"]
                           if getattr(commit, "exposure_only", False)
                           else _WRITE_ORDER)

            # ── Re-read dividers + limits from live camera (most accurate) ────
            # We already did get_param in read_state but its edit_info wasn't
            # captured. Re-read divider AND min/max so we can enforce the camera's
            # own limits before writing.
            dividers: Dict[str, int] = {}
            bounds: Dict[str, Tuple[Optional[int], Optional[int]]] = {}
            for field_name in write_order:
                try:
                    _, div, min_raw, max_raw = await session._get_param_raw(field_name)
                    dividers[field_name] = div
                    bounds[field_name] = (min_raw, max_raw)
                except asyncio.TimeoutError:
                    # Fail-closed: a wrong divider produces a silent wrong commit
                    # because the camera applies an incorrect raw value and
                    # readback verification passes against the same wrong assumption.
                    # Surface this as a retriable connection-level failure instead.
                    raise asyncio.TimeoutError(
                        f"Divider GET timed out for {field_name} on {camera_label} "
                        f"-- aborting push to prevent silent wrong commit"
                    )

            # ── Write fields in order ─────────────────────────────────────────
            all_ok = True
            for field_name in write_order:
                await _cb("write", field_name)
                min_raw, max_raw = bounds[field_name]
                fr = await session.set_and_verify(
                    field_name,
                    values[field_name],
                    dividers[field_name],
                    min_raw=min_raw,
                    max_raw=max_raw,
                )
                result.field_results.append(fr)

                if not fr.ok:
                    # Abort transaction — do not write remaining fields into
                    # a partially-incorrect camera state
                    all_ok = False
                    result.aborted_at_field = field_name
                    result.error = (
                        f"Field {field_name} failed: {fr.verify_status.value}"
                        + (f" — {fr.error}" if fr.error else "")
                    )
                    logger.error(
                        "[%s] Field %s failed (%s) — ABORTING transaction",
                        camera_label, field_name, fr.verify_status.value
                    )
                    break

            # ── Read state after (even on partial abort — diagnostic) ─────────
            await _cb("read_after")
            state_after = await session.read_state()
            state_after.camera_id = result.camera_id
            result.state_after = state_after
            logger.info(
                "[%s] After: K=%s T=%s EA=%s",
                camera_label,
                _fmt(state_after.kelvin), _fmt(state_after.tint),
                _fmt(state_after.exposure_adjust),
            )

            result.success = all_ok
            result.finished_at = time.time()
            await _cb("done", "success" if all_ok else "failed")
            break   # don't retry on clean exit

        except (ConnectionError, OSError, TimeoutError) as exc:
            result.error = str(exc)
            logger.warning("[%s] Connection error (attempt %d): %s", camera_label, attempt, exc)
            if attempt >= MAX_RETRIES:
                result.finished_at = time.time()
                await _cb("done", "connection_failed")
        except asyncio.TimeoutError as exc:
            result.error = f"Timeout: {exc}"
            logger.warning("[%s] Timeout (attempt %d): %s", camera_label, attempt, exc)
            if attempt >= MAX_RETRIES:
                result.finished_at = time.time()
                await _cb("done", "timeout")
        except Exception as exc:
            result.error = f"{type(exc).__name__}: {exc}"
            logger.exception("[%s] Unexpected error", camera_label)
            result.finished_at = time.time()
            await _cb("done", "error")
            break   # don't retry on unexpected errors
        finally:
            await session.close()

    if result.finished_at is None:
        result.finished_at = time.time()
    return result


async def push_all_cameras(
    targets: List[Dict],
    *,
    dry_run: bool = False,
    max_concurrent: int = 6,
    progress_cb=None,
) -> List[PushResult]:
    """
    Push calibration values to all cameras concurrently.

    targets: list of dicts:
        {
            "camera_label": "G007_A",
            "ip": "10.20.61.191",
            "commit": CommitValues,  # .kelvin, .tint, .exposure_adjust
        }

    Returns list of PushResult in same order as targets.
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    async def _bounded_push(t: Dict) -> PushResult:
        async with semaphore:
            return await push_camera(
                t["ip"],
                t["camera_label"],
                t["commit"],
                dry_run=dry_run,
                progress_cb=progress_cb,
            )

    tasks = [asyncio.create_task(_bounded_push(t)) for t in targets]
    results = await asyncio.gather(*tasks, return_exceptions=False)
    return list(results)



# ── Reset to neutral ───────────────────────────────────────────────────────────

class _NeutralCommit:
    """Neutral camera state: 5600K daylight, zero tint, zero exposure adjust."""
    kelvin: int = 5600
    tint: float = 0.0
    exposure_adjust: float = 0.0

_NEUTRAL = _NeutralCommit()


async def reset_camera(
    ip: str,
    camera_label: str,
    *,
    dry_run: bool = False,
    progress_cb=None,
) -> PushResult:
    """
    Reset one camera to neutral defaults: 5600K, 0 tint, 0 exposureAdjust.

    Uses the full push_camera path — live divider reads, per-field verified
    readback, same retry/error handling as a calibration push.
    """
    return await push_camera(
        ip,
        camera_label,
        _NEUTRAL,
        dry_run=dry_run,
        progress_cb=progress_cb,
    )


async def reset_all_cameras(
    targets: List[Dict],
    *,
    dry_run: bool = False,
    max_concurrent: int = 6,
) -> List[PushResult]:
    """
    Reset all cameras to neutral defaults concurrently.

    targets: same format as push_all_cameras — list of dicts with
             keys: camera_label, ip  (commit key is ignored / not required)
    """
    neutral_targets = [
        {
            "camera_label": t["camera_label"],
            "ip": t["ip"],
            "commit": _NEUTRAL,
        }
        for t in targets
    ]
    return await push_all_cameras(
        neutral_targets,
        dry_run=dry_run,
        max_concurrent=max_concurrent,
    )


def reset_all_cameras_sync(
    targets: List[Dict],
    *,
    dry_run: bool = False,
    max_concurrent: int = 6,
) -> List[PushResult]:
    """Blocking wrapper around reset_all_cameras."""
    return asyncio.run(
        reset_all_cameras(targets, dry_run=dry_run, max_concurrent=max_concurrent)
    )


# ── Sync wrapper (for non-async callers) ──────────────────────────────────────

def push_camera_sync(
    ip: str,
    camera_label: str,
    commit,
    *,
    dry_run: bool = False,
) -> PushResult:
    """Blocking wrapper around push_camera for non-async code."""
    return asyncio.run(push_camera(ip, camera_label, commit, dry_run=dry_run))


def push_all_cameras_sync(
    targets: List[Dict],
    *,
    dry_run: bool = False,
    max_concurrent: int = 6,
) -> List[PushResult]:
    """Blocking wrapper around push_all_cameras."""
    return asyncio.run(
        push_all_cameras(targets, dry_run=dry_run, max_concurrent=max_concurrent)
    )


# ── Verify-only (read-only state check) ───────────────────────────────────────

async def verify_camera_state(
    ip: str,
    camera_label: str,
    expected: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Read current camera state without writing anything.

    expected: optional dict with keys kelvin, tint, exposure_adjust
              to compare against (generates a verification report).

    Returns structured dict — rcp2_camera_state_report compatible.
    """
    report: Dict[str, Any] = {
        "camera_label": camera_label,
        "camera_ip": ip,
        "mode": "verify_only",
        "write_commands_sent": False,
        "timestamp": time.time(),
    }

    session = RCP2Session(ip, camera_label, dry_run=True)
    try:
        await session.connect()
        try:
            info = await session.get_camera_info()
            report["camera_type"] = info.get("camera_type", {}).get("str")
            report["firmware_version"] = info.get("version", {}).get("str")
            report["serial_number"] = info.get("serial_number")
        except asyncio.TimeoutError:
            pass

        state = await session.read_state()
        report["state"] = {
            "kelvin": state.kelvin,
            "tint": state.tint,
            "exposure_adjust": state.exposure_adjust,
        }

        if expected:
            comparisons = {}
            for key in ("kelvin", "tint", "exposure_adjust"):
                exp_val = expected.get(key)
                act_val = report["state"].get(key)
                if exp_val is None or act_val is None:
                    comparisons[key] = {"status": VerifyStatus.NOT_AVAILABLE.value}
                    continue
                delta = abs(act_val - exp_val)
                tol_key = {
                    "kelvin": "kelvin",
                    "tint": "tint",
                    "exposure_adjust": "exposureAdjust",
                }[key]
                status = _classify(tol_key, delta)
                comparisons[key] = {
                    "expected": exp_val,
                    "actual": act_val,
                    "delta": delta,
                    "status": status.value,
                }
            report["expected_comparison"] = comparisons

    finally:
        await session.close()

    return report


# ── Helpers ────────────────────────────────────────────────────────────────────

def _classify(field: str, delta: float) -> VerifyStatus:
    """Classify a readback delta as EXACT / WITHIN / MISMATCH."""
    tolerances = {
        "kelvin":         TOL_KELVIN,
        "tint":           TOL_TINT,
        "exposureAdjust": TOL_EXPOSURE,
    }
    tol = tolerances.get(field, 0.01)
    if delta == 0.0:
        return VerifyStatus.EXACT_MATCH
    if delta <= tol:
        return VerifyStatus.WITHIN_TOLERANCE
    return VerifyStatus.MISMATCH


# Static safe raw bounds — used ONLY as a fallback when the camera does not
# report min/max in edit_info. Ranges from the RED KOMODO-X RCP2 parameter list.
# When the camera reports its own limits, those take precedence (see
# set_and_verify). Targets outside the active range are rejected, not clamped.
_STATIC_RAW_BOUNDS = {
    # COLOR_TEMPERATURE: 1700–10000 K (divider=1)
    "kelvin":         (1700,  10000),
    # TINT: ±100000 raw (±100.0 tint units with divider=1000) — confirmed live
    # on KOMODO-X (FW 2.2.4).
    "tint":           (-100000, 100000),
    # EXPOSURE_ADJUST: ±8000 raw (±8.0 stops with divider=1000) — confirmed live
    # on KOMODO-X (FW 2.2.4).
    "exposureAdjust": (-8000, 8000),
}


def _clamp_raw(field: str, raw: int) -> int:
    """Deprecated: retained for compatibility. Prefer the reject-on-violation
    enforcement in set_and_verify, which uses camera-reported limits."""
    lo, hi = _STATIC_RAW_BOUNDS.get(field, (-999999, 999999))
    return max(lo, min(hi, raw))


def _ids_match(msg_id: str, target_id: str) -> bool:
    """
    Compare RCP parameter IDs — accept both abbreviated and full prefixed forms.
    e.g. "COLOR_TEMPERATURE" matches "RCP_PARAM_COLOR_TEMPERATURE" and vice versa.
    """
    prefix = "RCP_PARAM_"
    a = msg_id.upper().removeprefix(prefix)
    b = target_id.upper().removeprefix(prefix)
    return a == b


def _fmt(v: Optional[float]) -> str:
    if v is None:
        return "N/A"
    return f"{v:.2f}"


# ── CommitLike protocol (duck-typed) ───────────────────────────────────────────

class CommitLike:
    """
    Minimal interface expected from the commit object.
    Compatible with r3dmatch3.models.CommitValues.
    """
    kelvin: int
    tint: float
    exposure_adjust: float


# ── Report helpers ─────────────────────────────────────────────────────────────

def summarize_push_results(results: List[PushResult]) -> Dict[str, Any]:
    """
    Aggregate push results into a summary report dict.
    Matches the rcp2_apply_report.json schema from v1.
    """
    total = len(results)
    succeeded = sum(1 for r in results if r.success)
    failed = total - succeeded

    camera_summaries = []
    for r in results:
        fr_summary = []
        for fr in r.field_results:
            fr_summary.append({
                "field": fr.field,
                "requested": fr.requested_value,
                "readback": fr.readback_value,
                "status": fr.verify_status.value if fr.verify_status else None,
                "delta": fr.delta,
                "ok": fr.ok,
            })
        camera_summaries.append({
            "camera_label": r.camera_label,
            "ip": r.camera_ip,
            "success": r.success,
            "dry_run": r.dry_run,
            "elapsed_seconds": r.elapsed_seconds,
            "camera_type": r.camera_type,
            "firmware": r.firmware_version,
            "state_before": {
                "kelvin": r.state_before.kelvin if r.state_before else None,
                "tint": r.state_before.tint if r.state_before else None,
                "exposure_adjust": r.state_before.exposure_adjust if r.state_before else None,
            },
            "state_after": {
                "kelvin": r.state_after.kelvin if r.state_after else None,
                "tint": r.state_after.tint if r.state_after else None,
                "exposure_adjust": r.state_after.exposure_adjust if r.state_after else None,
            },
            "field_results": fr_summary,
            "aborted_at": r.aborted_at_field,
            "error": r.error,
            "retries": r.retries,
        })

    return {
        "schema": "r3dmatch_rcp2_push_report_v1",
        "timestamp": time.time(),
        "summary": {
            "total_cameras": total,
            "succeeded": succeeded,
            "failed": failed,
            "dry_run": any(r.dry_run for r in results),
        },
        "cameras": camera_summaries,
    }
