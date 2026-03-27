from __future__ import annotations

import base64
import hashlib
import json
import os
import socket
import struct
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Deque, Dict, Iterable, List, Optional


RCP2_WS_PATH = "/"
RCP2_WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


class Rcp2WebSocketError(RuntimeError):
    pass


class Rcp2WebSocketConnectionError(Rcp2WebSocketError):
    pass


class Rcp2WebSocketProtocolError(Rcp2WebSocketError):
    pass


class Rcp2WebSocketTimeout(Rcp2WebSocketError):
    pass


@dataclass(frozen=True)
class Rcp2ParameterSpec:
    field_name: str
    parameter_id: str
    legacy_parameter_id: str
    default_divider: int


PARAMETER_SPECS: Dict[str, Rcp2ParameterSpec] = {
    "exposureAdjust": Rcp2ParameterSpec(
        field_name="exposureAdjust",
        parameter_id="EXPOSURE_ADJUST",
        legacy_parameter_id="RCP_PARAM_EXPOSURE_ADJUST",
        default_divider=1000,
    ),
    "kelvin": Rcp2ParameterSpec(
        field_name="kelvin",
        parameter_id="COLOR_TEMPERATURE",
        legacy_parameter_id="RCP_PARAM_COLOR_TEMPERATURE",
        default_divider=1,
    ),
    "tint": Rcp2ParameterSpec(
        field_name="tint",
        parameter_id="TINT",
        legacy_parameter_id="RCP_PARAM_TINT",
        default_divider=1000,
    ),
}


@dataclass
class Rcp2ParameterState:
    parameter_id: str
    raw_value: int
    divider: int
    value: float
    display: str
    edit_info: Dict[str, object]
    messages: List[Dict[str, object]]
    message_index: int = 0
    received_at: float = 0.0


@dataclass
class Rcp2WebSocketSession:
    sock: socket.socket
    host: str
    port: int
    pending_messages: Deque[Dict[str, object]] = field(default_factory=deque)
    config: Dict[str, object] = field(default_factory=dict)
    types_payload: Dict[str, object] = field(default_factory=dict)
    camera_info: Dict[str, object] = field(default_factory=dict)
    parameter_states: Dict[str, Rcp2ParameterState] = field(default_factory=dict)
    message_counter: int = 0


def build_websocket_upgrade_request(*, host: str, port: int, key: str, path: str = RCP2_WS_PATH) -> bytes:
    return (
        f"GET {path} HTTP/1.1\r\n"
        f"Host: {host}:{port}\r\n"
        "Upgrade: websocket\r\n"
        "Connection: Upgrade\r\n"
        f"Sec-WebSocket-Key: {key}\r\n"
        "Sec-WebSocket-Version: 13\r\n"
        "\r\n"
    ).encode("utf-8")


def expected_websocket_accept(key: str) -> str:
    digest = hashlib.sha1(f"{key}{RCP2_WS_GUID}".encode("utf-8")).digest()
    return base64.b64encode(digest).decode("ascii")


def build_websocket_text_frame(text: str, *, mask_key: Optional[bytes] = None) -> bytes:
    payload = text.encode("utf-8")
    actual_mask = mask_key or os.urandom(4)
    header = bytearray([0x81])
    payload_len = len(payload)
    if payload_len < 126:
        header.append(0x80 | payload_len)
    elif payload_len < 65536:
        header.append(0x80 | 126)
        header.extend(struct.pack("!H", payload_len))
    else:
        header.append(0x80 | 127)
        header.extend(struct.pack("!Q", payload_len))
    header.extend(actual_mask)
    masked_payload = bytes(byte ^ actual_mask[index % 4] for index, byte in enumerate(payload))
    return bytes(header) + masked_payload


def build_periodic_get_payload(*, parameter_id: str, interval_ms: int) -> Dict[str, object]:
    return {
        "type": "rcp_get_periodic_on",
        "id": normalize_parameter_id(parameter_id),
        "interval_ms": int(interval_ms),
    }


def normalize_parameter_id(value: str) -> str:
    token = str(value or "").strip().upper()
    if token.startswith("RCP_PARAM_"):
        token = token[len("RCP_PARAM_") :]
    return token


def extract_parameter_state(
    messages: Iterable[Dict[str, object]],
    *,
    parameter_id: str,
    fallback_divider: int,
) -> Rcp2ParameterState:
    normalized_id = normalize_parameter_id(parameter_id)
    collected: List[Dict[str, object]] = []
    edit_info: Dict[str, object] = {}
    raw_value: Optional[int] = None
    display = ""
    newest_index = 0
    newest_timestamp = 0.0
    for message in messages:
        if normalize_parameter_id(str(message.get("id") or "")) != normalized_id:
            continue
        collected.append(dict(message))
        newest_index = max(newest_index, int(message.get("_r3dmatch_message_index", 0) or 0))
        newest_timestamp = max(newest_timestamp, float(message.get("_r3dmatch_received_at", 0.0) or 0.0))
        message_type = str(message.get("type") or "")
        if message_type.endswith("_edit_info"):
            edit_info = dict(message)
            current_value = message.get("cur")
            if isinstance(current_value, (int, float)):
                raw_value = int(current_value)
        elif message_type in {"rcp_cur_int", "rcp_cur_uint"}:
            current = message.get("cur")
            if isinstance(current, dict) and "val" in current:
                raw_value = int(current["val"])
            elif isinstance(current, (int, float)):
                raw_value = int(current)
            nested_edit = message.get("edit_info")
            if isinstance(nested_edit, dict):
                edit_info = dict(nested_edit)
        elif message_type == "rcp_cur_str":
            display_payload = message.get("display")
            if isinstance(display_payload, dict):
                display = str(display_payload.get("str") or display_payload.get("abbr") or "")
    if raw_value is None:
        raise Rcp2WebSocketProtocolError(f"No integer value was returned for {normalized_id}.")
    divider = int(edit_info.get("divider") or fallback_divider or 1)
    if divider <= 0:
        divider = fallback_divider or 1
    return Rcp2ParameterState(
        parameter_id=normalized_id,
        raw_value=raw_value,
        divider=divider,
        value=float(raw_value) / float(divider),
        display=display,
        edit_info=edit_info,
        messages=collected,
        message_index=newest_index,
        received_at=newest_timestamp,
    )


class JsonWebSocketClient:
    def __init__(
        self,
        *,
        host: str,
        port: int,
        connect_timeout_ms: int,
        operation_timeout_ms: int,
        settle_timeout_ms: int,
        raw_logging: bool = False,
    ) -> None:
        self.host = host
        self.port = int(port)
        self.connect_timeout_ms = int(connect_timeout_ms)
        self.operation_timeout_ms = int(operation_timeout_ms)
        self.settle_timeout_ms = int(settle_timeout_ms)
        self.raw_logging = bool(raw_logging)

    def connect(self) -> Rcp2WebSocketSession:
        try:
            sock = socket.create_connection((self.host, self.port), self.connect_timeout_ms / 1000.0)
        except OSError as exc:
            raise Rcp2WebSocketConnectionError(
                f"Failed to connect to ws://{self.host}:{self.port}: {exc}"
            ) from exc
        sock.settimeout(self.operation_timeout_ms / 1000.0)
        key = base64.b64encode(os.urandom(16)).decode("ascii")
        request = build_websocket_upgrade_request(host=self.host, port=self.port, key=key)
        sock.sendall(request)
        response = self._recv_http_response(sock)
        expected_accept = expected_websocket_accept(key)
        status_line = response.split("\r\n", 1)[0]
        headers = self._parse_http_headers(response)
        if "101" not in status_line:
            sock.close()
            raise Rcp2WebSocketProtocolError(
                f"WebSocket upgrade failed for ws://{self.host}:{self.port}: {status_line or response!r}"
            )
        if headers.get("sec-websocket-accept", "") != expected_accept:
            sock.close()
            raise Rcp2WebSocketProtocolError(
                f"WebSocket accept mismatch for ws://{self.host}:{self.port}."
            )
        return Rcp2WebSocketSession(sock=sock, host=self.host, port=self.port)

    def close(self, session: Rcp2WebSocketSession) -> None:
        try:
            session.sock.sendall(self._build_close_frame())
        except OSError:
            pass
        try:
            session.sock.close()
        except OSError:
            return

    def send_json(self, session: Rcp2WebSocketSession, payload: Dict[str, object]) -> None:
        text = json.dumps(payload, separators=(",", ":"))
        if self.raw_logging:
            print(f"[RCP2 WS OUT] {text}")
        frame = build_websocket_text_frame(text)
        try:
            session.sock.sendall(frame)
        except OSError as exc:
            raise Rcp2WebSocketConnectionError(
                f"Failed to send JSON over ws://{session.host}:{session.port}: {exc}"
            ) from exc

    def request_messages(
        self,
        session: Rcp2WebSocketSession,
        payload: Dict[str, object],
        *,
        matcher: Callable[[Dict[str, object]], bool],
        timeout_ms: Optional[int] = None,
        settle_timeout_ms: Optional[int] = None,
    ) -> List[Dict[str, object]]:
        self.send_json(session, payload)
        minimum_message_index = int(session.message_counter)
        return self.collect_matching_messages(
            session,
            matcher=matcher,
            timeout_ms=timeout_ms,
            settle_timeout_ms=settle_timeout_ms,
            minimum_message_index=minimum_message_index,
        )

    def collect_matching_messages(
        self,
        session: Rcp2WebSocketSession,
        *,
        matcher: Callable[[Dict[str, object]], bool],
        timeout_ms: Optional[int] = None,
        settle_timeout_ms: Optional[int] = None,
        minimum_message_index: int = 0,
    ) -> List[Dict[str, object]]:
        matched: List[Dict[str, object]] = []
        remaining = deque()
        while session.pending_messages:
            message = session.pending_messages.popleft()
            message_index = int(message.get("_r3dmatch_message_index", 0) or 0)
            if matcher(message) and message_index > int(minimum_message_index):
                matched.append(message)
            elif not matcher(message):
                remaining.append(message)
        session.pending_messages = remaining

        total_timeout_s = float(timeout_ms if timeout_ms is not None else self.operation_timeout_ms) / 1000.0
        settle_timeout_s = float(settle_timeout_ms if settle_timeout_ms is not None else self.settle_timeout_ms) / 1000.0
        deadline = time.monotonic() + total_timeout_s
        idle_deadline: Optional[float] = time.monotonic() + settle_timeout_s if matched else None

        while True:
            now = time.monotonic()
            if idle_deadline is not None and now >= idle_deadline:
                break
            if now >= deadline:
                break
            timeout_s = max(0.05, min(deadline - now, (idle_deadline - now) if idle_deadline is not None else total_timeout_s))
            try:
                message = self.recv_json(session, timeout_s)
            except Rcp2WebSocketTimeout:
                if matched:
                    break
                raise
            if matcher(message):
                matched.append(message)
                idle_deadline = min(deadline, time.monotonic() + settle_timeout_s)
            else:
                session.pending_messages.append(message)
                if matched and idle_deadline is None:
                    idle_deadline = min(deadline, time.monotonic() + settle_timeout_s)
        if not matched:
            raise Rcp2WebSocketTimeout("No matching RCP2 message was received before timeout.")
        return matched

    def recv_json(self, session: Rcp2WebSocketSession, timeout_s: float) -> Dict[str, object]:
        while session.pending_messages:
            message = session.pending_messages.popleft()
            if isinstance(message, dict):
                return message
        try:
            payload = self._recv_text_message(session.sock, timeout_s)
        except socket.timeout as exc:
            raise Rcp2WebSocketTimeout("WebSocket receive timed out.") from exc
        except OSError as exc:
            raise Rcp2WebSocketConnectionError(f"WebSocket receive failed: {exc}") from exc
        if payload is None:
            raise Rcp2WebSocketConnectionError(f"Camera at ws://{session.host}:{session.port} closed the WebSocket.")
        if self.raw_logging:
            print(f"[RCP2 WS IN] {payload}")
        try:
            message = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise Rcp2WebSocketProtocolError(f"Received invalid RCP2 JSON: {payload[:200]}") from exc
        if not isinstance(message, dict):
            raise Rcp2WebSocketProtocolError("Received a non-object JSON message from the camera.")
        session.message_counter += 1
        message["_r3dmatch_message_index"] = int(session.message_counter)
        message["_r3dmatch_received_at"] = float(time.time())
        return message

    @staticmethod
    def _parse_http_headers(response: str) -> Dict[str, str]:
        headers: Dict[str, str] = {}
        for line in response.split("\r\n")[1:]:
            if not line or ":" not in line:
                continue
            key, value = line.split(":", 1)
            headers[key.strip().lower()] = value.strip()
        return headers

    @staticmethod
    def _build_close_frame() -> bytes:
        return bytes([0x88, 0x80, 0x00, 0x00, 0x00, 0x00])

    @staticmethod
    def _recv_http_response(sock: socket.socket) -> str:
        buffer = b""
        while b"\r\n\r\n" not in buffer:
            chunk = sock.recv(4096)
            if not chunk:
                break
            buffer += chunk
        return buffer.decode("utf-8", errors="replace")

    @staticmethod
    def _recv_exact(sock: socket.socket, size: int) -> bytes:
        data = b""
        while len(data) < size:
            chunk = sock.recv(size - len(data))
            if not chunk:
                raise EOFError("socket closed")
            data += chunk
        return data

    def _recv_text_message(self, sock: socket.socket, timeout_s: float) -> Optional[str]:
        sock.settimeout(timeout_s)
        fragments: List[bytes] = []
        while True:
            header = self._recv_exact(sock, 2)
            fin = (header[0] >> 7) & 1
            opcode = header[0] & 0x0F
            masked = (header[1] >> 7) & 1
            payload_len = header[1] & 0x7F
            if payload_len == 126:
                payload_len = struct.unpack("!H", self._recv_exact(sock, 2))[0]
            elif payload_len == 127:
                payload_len = struct.unpack("!Q", self._recv_exact(sock, 8))[0]
            mask_key = self._recv_exact(sock, 4) if masked else b""
            payload = self._recv_exact(sock, payload_len) if payload_len else b""
            if masked:
                payload = bytes(byte ^ mask_key[index % 4] for index, byte in enumerate(payload))
            if opcode == 0x8:
                return None
            if opcode == 0x9:
                self._send_pong(sock, payload)
                continue
            if opcode in {0x1, 0x0}:
                fragments.append(payload)
                if fin:
                    return b"".join(fragments).decode("utf-8", errors="replace")
                continue
            raise Rcp2WebSocketProtocolError(f"Unsupported WebSocket opcode {opcode}.")

    @staticmethod
    def _send_pong(sock: socket.socket, payload: bytes) -> None:
        frame = bytearray([0x8A])
        payload_len = len(payload)
        if payload_len < 126:
            frame.append(0x80 | payload_len)
        elif payload_len < 65536:
            frame.append(0x80 | 126)
            frame.extend(struct.pack("!H", payload_len))
        else:
            frame.append(0x80 | 127)
            frame.extend(struct.pack("!Q", payload_len))
        mask_key = os.urandom(4)
        frame.extend(mask_key)
        masked_payload = bytes(byte ^ mask_key[index % 4] for index, byte in enumerate(payload))
        frame.extend(masked_payload)
        sock.sendall(bytes(frame))
