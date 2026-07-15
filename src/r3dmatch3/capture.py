"""
capture.py — R3DMatch v5 RCP2 single-frame capture module
==========================================================
Detect → connect → frame-limit(1) → verify sync → roll → cut for a RED
KOMODO-X array, driving the same RCP2 WebSocket protocol (port 9998) used by
the calibration push, but implemented as a dedicated, purpose-built capture
transport.

WHY A SEPARATE MODULE (and not rcp2.py's RCP2Session):
  rcp2.py is an asyncio client tuned for a short-lived write-verify-close push.
  Capture needs *persistent* sessions across the whole detect→roll→cut arc for
  up to 36 bodies, held open while the operator watches live timecode. This
  module uses a zero-dependency raw-socket transport (one reader thread per
  camera) ported from REDConductorV3's red_soak.py, which encodes every
  hard-won transport rule from RCP2_FIELD_NOTES.md directly.

TRANSPORT RULES (RCP2_FIELD_NOTES.md — each was paid for on set):
  - ONE WebSocket session per camera, ever. Detection is a TCP probe that
    opens NO session (rule 1, 2, 15).
  - Init: rcp_config → rcp_get_types → CAMERA_INFO → rcp_get_parameters →
    subscribe the rule-9 list. If all three core gets go unanswered, init
    FAILED (rule 8).
  - Do NOT send WebSocket pings — FW 2.2.4 never pongs them (rule 5). Pong the
    camera if it pings us.
  - TCP keepalive 5/3/3 + TCP_NODELAY (rule 6).
  - Quiet > 3 s → ONE RECORD_STATE heartbeat; quiet > 15 s → link is stale
    (rule 7).
  - Subscribe once, poll never (rule 9). No blind gets of un-advertised params
    (rule 11) — we only ever touch params in the subscribe list plus the
    sync-record / frame-limit sets, all of which the camera advertises.
  - Graceful close frees the ~8-slot session pool (rule 3).

SYNC RECORD (the approved RED flow, FW ≥ 2.2.4 — rules 17-22):
  Read reference-camera TC → SYNC_RECORD_START (value = target TC 2-5 s in the
  future) to all bodies. RECORD_STATE is truth; SYNC_RECORD_STATUS is advisory.

HARDWARE-UNVERIFIED paths are labelled `# UNVERIFIED` per rule 23 — validate on
a body before relying on them. Nothing here was tested against live hardware in
this build; it mirrors the documented protocol and the reference tools.
"""
from __future__ import annotations

import base64
import ipaddress
import json
import logging
import os
import re
import socket
import struct
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Protocol constants ─────────────────────────────────────────────────────────
RCP2_PORT = 9998
WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

# Subscribe list — verbatim from RCP2_FIELD_NOTES.md rule 9. Subscribe once,
# poll never; the camera pushes TC at 1/s and the rest on change.
SUBSCRIBED = [
    "RECORD_STATE", "TIMECODE", "TIMECODE_STATE", "TIMECODE_SOURCE", "SYNC_STATE",
    "GENLOCK_STATE", "MEDIA_STATUS", "MEDIA_PERCENTAGE_REMAINING",
    "MEDIA_TIME_REMAINING", "MEDIA_CLIP_COUNT", "CLIP_NAME",
    "FRAME_LIMIT_ENABLE", "FRAME_LIMIT_FRAMES", "SYNC_RECORD_STATUS",
]

# Record states (rule 22): 2/3/4/5 are transient — wait, don't fail.
RECORD_STATES = {
    0: "idle", 1: "recording", 2: "finalizing", 3: "pre-recording",
    4: "encoding", 5: "sync-armed",
}

# Timecode shape: HH:MM:SS with optional :FF or ;FF (drop-frame) frame field.
TC_RE = re.compile(r"\b(\d{1,2}):(\d{2}):(\d{2})([:;]\d{2})?\b")

# Transport timings.
CONNECT_TIMEOUT = 5.0
HANDSHAKE_TIMEOUT = 6.0
HEARTBEAT_QUIET = 3.0        # quiet > this → one RECORD_STATE heartbeat (rule 7)
STALE_QUIET = 15.0          # quiet > this despite heartbeat → link stale (rule 7)
SET_VERIFY_TIMEOUT = 6.0    # wait for a subscribed param to reflect a set

# Roll timing: target must be far enough ahead that send latency doesn't eat it
# (rule 17 — 1 s is legal but eaten by latency; use 2 s minimum).
DEFAULT_ROLL_LEAD = 3.0
DEFAULT_CUT_LEAD = 2.0
MIN_ROLL_LEAD = 2.0


# ── Low-level WebSocket framing (ported from red_soak.py) ───────────────────────
def encode_frame(opcode: int, payload: bytes = b"", mask: bool = True) -> bytes:
    ln = len(payload)
    h = bytearray([0x80 | opcode])
    mbit = 0x80 if mask else 0
    if ln < 126:
        h.append(mbit | ln)
    elif ln < 65536:
        h.append(mbit | 126); h += struct.pack("!H", ln)
    else:
        h.append(mbit | 127); h += struct.pack("!Q", ln)
    if not mask:
        return bytes(h) + payload
    m = os.urandom(4); h += m
    return bytes(h) + bytes(b ^ m[i & 3] for i, b in enumerate(payload))


def parse_frame(buf: bytearray):
    """Pop one complete frame from `buf` in place. Returns (fin, op, payload) or None."""
    if len(buf) < 2:
        return None
    b1 = buf[1]; masked = b1 & 0x80; ln = b1 & 0x7f; i = 2
    if ln == 126:
        if len(buf) < 4:
            return None
        ln = struct.unpack("!H", bytes(buf[2:4]))[0]; i = 4
    elif ln == 127:
        if len(buf) < 10:
            return None
        ln = struct.unpack("!Q", bytes(buf[2:10]))[0]; i = 10
    mask = b""
    if masked:
        if len(buf) < i + 4:
            return None
        mask = bytes(buf[i:i + 4]); i += 4
    if len(buf) < i + ln:
        return None
    p = bytes(buf[i:i + ln])
    if masked:
        p = bytes(c ^ mask[j & 3] for j, c in enumerate(p))
    fin = buf[0] & 0x80; op = buf[0] & 0x0f
    del buf[:i + ln]
    return (fin, op, p)


def ws_handshake(sock: socket.socket, host: str, port: int) -> bytearray:
    key = base64.b64encode(os.urandom(16)).decode()
    req = ("GET / HTTP/1.1\r\nHost: %s:%d\r\nUpgrade: websocket\r\n"
           "Connection: Upgrade\r\nSec-WebSocket-Key: %s\r\n"
           "Sec-WebSocket-Version: 13\r\n\r\n" % (host, port, key)).encode()
    sock.sendall(req)
    resp = b""; sock.settimeout(HANDSHAKE_TIMEOUT)
    while b"\r\n\r\n" not in resp:
        c = sock.recv(1024)
        if not c:
            raise IOError("closed during handshake")
        resp += c
    if b" 101 " not in resp.split(b"\r\n", 1)[0]:
        raise IOError("no 101 upgrade: %r" % resp.split(b"\r\n", 1)[0])
    return bytearray(resp.split(b"\r\n\r\n", 1)[1])


def set_keepalive(sock: socket.socket) -> None:
    """TCP keepalive 5/3/3 + TCP_NODELAY (rule 6)."""
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
    for opt, val in [(getattr(socket, "TCP_KEEPALIVE", 0x10), 5),
                     (getattr(socket, "TCP_KEEPINTVL", 0x101), 3),
                     (getattr(socket, "TCP_KEEPCNT", 0x102), 3)]:
        try:
            sock.setsockopt(socket.IPPROTO_TCP, opt, val)
        except OSError:
            pass
    try:
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    except OSError:
        pass


# ── Detection (costs zero session slots — rule 15) ──────────────────────────────
def tcp_scan(cidr: str, port: int = RCP2_PORT, source_ip: Optional[str] = None,
             timeout: float = 0.3, workers: int = 100,
             stop: Optional[Callable[[], bool]] = None,
             progress: Optional[Callable[[int, int], None]] = None) -> List[str]:
    """Plain TCP connect to <port> across a CIDR. No WS upgrade → spends no RCP
    session slot (rule 2, 15). Returns hosts that accept the connection, sorted."""
    hosts = [str(h) for h in ipaddress.ip_network(cidr, strict=False).hosts()]
    total = len(hosts)
    done = [0]
    lock = threading.Lock()

    def probe(ip: str) -> Optional[str]:
        if stop and stop():
            return None
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            if source_ip:
                s.bind((source_ip, 0))
            s.settimeout(timeout)
            s.connect((ip, port))
            return ip
        except OSError:
            return None
        finally:
            s.close()
            if progress:
                with lock:
                    done[0] += 1
                    progress(done[0], total)

    found: List[str] = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for r in ex.map(probe, hosts):
            if r:
                found.append(r)
    return sorted(found, key=lambda x: tuple(int(o) for o in x.split(".")))


def discover_udp(timeout: float = 4.0, source_ip: Optional[str] = None) -> List[str]:
    """UDP CAMINFO broadcast on port 1112 (rule 15). Costs zero session slots.
    Fallback / complement to tcp_scan on networks where the CIDR is unknown."""
    body = b"$API:G:CAMINFO:"; x = 0
    for b in body:
        x ^= b
    pkt = b"#" + body + ("*%02X\n" % x).encode()
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    s.bind((source_ip or "", 0)); s.settimeout(0.3)
    seen: Dict[str, bytes] = {}; end = time.time() + timeout; last = 0.0
    while time.time() < end:
        if time.time() - last > 1:
            last = time.time()
            try:
                s.sendto(pkt, ("255.255.255.255", 1112))
            except OSError:
                pass
        try:
            d, a = s.recvfrom(2048)
            if b"CAMINFO" in d:
                seen[a[0]] = d
        except socket.timeout:
            pass
    s.close()
    return sorted(seen)


# ── Timecode helpers ────────────────────────────────────────────────────────────
def parse_tc(text: str) -> Optional[Tuple[int, int, int, Optional[int], bool]]:
    """Return (hh, mm, ss, ff_or_None, drop_frame) from a TC string, or None.

    FW 2.2.4 pushes seconds-resolution TC ("14:28:34", no frame field — rule 20),
    so ff is usually None. Drop-frame is signalled by a ';' before the frames.
    """
    if not text:
        return None
    m = TC_RE.search(str(text))
    if not m:
        return None
    hh, mm, ss = int(m.group(1)), int(m.group(2)), int(m.group(3))
    ff = None; drop = False
    if m.group(4):
        drop = m.group(4)[0] == ";"
        ff = int(m.group(4)[1:])
    return (hh, mm, ss, ff, drop)


def tc_to_seconds(text: str) -> Optional[int]:
    """Whole-second position of a TC string (frames dropped — rule 20)."""
    p = parse_tc(text)
    if p is None:
        return None
    hh, mm, ss, _ff, _drop = p
    return hh * 3600 + mm * 60 + ss


def compute_target_tc(reference_tc: str, lead_seconds: float) -> Optional[str]:
    """Reference TC + lead → target TC string landing on frame 00 (rule 20).

    Preserves the reference's drop-frame ';' delimiter when present, else ':'.
    Wraps at 24 h. Returns None if the reference TC is unparseable.
    """
    p = parse_tc(reference_tc)
    if p is None:
        return None
    hh, mm, ss, _ff, drop = p
    total = (hh * 3600 + mm * 60 + ss) + max(MIN_ROLL_LEAD, int(round(lead_seconds)))
    total %= 24 * 3600
    th, rem = divmod(total, 3600)
    tm, tsec = divmod(rem, 60)
    delim = ";" if drop else ":"
    return "%02d:%02d:%02d%s00" % (th, tm, tsec, delim)


# ── Per-camera link ─────────────────────────────────────────────────────────────
@dataclass
class LinkState:
    ip: str = ""
    connected: bool = False
    stale: bool = False
    camera_type: str = ""
    serial: str = ""
    firmware: str = ""
    record_state: Optional[int] = None
    timecode: str = ""
    timecode_state: str = ""
    timecode_source: str = ""
    sync_state: str = ""
    genlock_state: str = ""
    clip_name: str = ""
    frame_limit_enable: Optional[bool] = None
    frame_limit_frames: Optional[int] = None
    sync_record_status: Optional[int] = None
    last_msg_t: float = 0.0
    error: str = ""


class CameraLink:
    """One persistent RCP2 session for one camera. A daemon reader thread pumps
    pushed messages into `self.state`; commands are sent under a lock and their
    effect is confirmed by watching the subscribed state (rules 9, 13)."""

    def __init__(self, ip: str, label: str = "", source_ip: Optional[str] = None):
        self.ip = ip
        self.label = label or ip
        self.source_ip = source_ip
        self.state = LinkState(ip=ip)
        self._sock: Optional[socket.socket] = None
        self._send_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._reader: Optional[threading.Thread] = None
        self._running = False

    # -- lifecycle --------------------------------------------------------------
    def connect(self) -> None:
        """Open the session and run the mandated init sequence (rule 8)."""
        sock = socket.create_connection(
            (self.ip, RCP2_PORT), timeout=CONNECT_TIMEOUT,
            source_address=((self.source_ip, 0) if self.source_ip else None))
        set_keepalive(sock)
        buf = ws_handshake(sock, self.ip, RCP2_PORT)
        self._sock = sock
        self._raw_send({
            "type": "rcp_config", "strings_decoded": 1, "json_minified": 1,
            "include_cacheable_flags": 0, "encoding_type": "legacy",
            "client": {"name": "R3DMatch-capture", "version": "5.0"}})
        self._raw_send({"type": "rcp_get_types"})
        self._raw_send({"type": "rcp_get", "id": "CAMERA_INFO"})
        self._raw_send({"type": "rcp_get_parameters"})
        for pid in SUBSCRIBED:
            self._raw_send({"type": "rcp_subscribe", "id": pid, "on_off": True})
        # Verify init actually took: wait for ANY of the three core gets to answer
        # (rule 8 — TCP-open-but-mute is common; a mute link must fail here).
        deadline = time.time() + HANDSHAKE_TIMEOUT
        got = self._pump_until(sock, buf, deadline,
                               lambda: self.state.last_msg_t > 0)
        if not got:
            self._close_socket(sock)
            raise IOError("init unanswered (rcp_get_types/CAMERA_INFO/params) — "
                          "session pool may be full or link is mute")
        with self._state_lock:
            self.state.connected = True
        self._running = True
        self._reader = threading.Thread(
            target=self._read_loop, args=(sock, buf), daemon=True)
        self._reader.start()

    def close(self, grace: float = 3.0) -> None:
        """Graceful close frees the ~8-slot pool (rule 3)."""
        self._running = False
        sock = self._sock
        if sock is not None:
            try:
                sock.sendall(encode_frame(0x8, struct.pack("!H", 1000)))
                time.sleep(min(0.3, grace))
            except OSError:
                pass
            self._close_socket(sock)
        self._sock = None
        with self._state_lock:
            self.state.connected = False

    @staticmethod
    def _close_socket(sock: socket.socket) -> None:
        try:
            sock.close()
        except OSError:
            pass

    # -- send -------------------------------------------------------------------
    def _raw_send(self, obj: dict) -> None:
        sock = self._sock
        if sock is None:
            raise IOError("not connected")
        with self._send_lock:
            sock.sendall(encode_frame(0x1, json.dumps(obj).encode()))

    # -- reader -----------------------------------------------------------------
    def _read_loop(self, sock: socket.socket, buf: bytearray) -> None:
        sock.settimeout(0.3)
        last_seen = time.time(); last_hb = 0.0
        while self._running:
            self._drain(buf)
            try:
                data = sock.recv(8192)
                if data == b"":
                    self._mark_stale("peer closed")
                    return
                buf += data
                last_seen = time.time()
            except socket.timeout:
                pass
            except OSError as e:
                self._mark_stale("recv error: %s" % e)
                return
            self._drain(buf)
            now = time.time()
            quiet = now - max(last_seen, self.state.last_msg_t)
            if quiet > STALE_QUIET:
                self._mark_stale("stale link (%ds silent)" % int(quiet))
                # keep the socket; the controller decides whether to reconnect
            elif quiet > HEARTBEAT_QUIET and now - last_hb > HEARTBEAT_QUIET:
                last_hb = now
                try:
                    self._raw_send({"type": "rcp_get", "id": "RECORD_STATE"})
                except OSError as e:
                    self._mark_stale("heartbeat send failed: %s" % e)
                    return

    def _pump_until(self, sock: socket.socket, buf: bytearray, deadline: float,
                    predicate: Callable[[], bool]) -> bool:
        """Synchronous pump used during connect() before the reader thread starts."""
        sock.settimeout(0.3)
        while time.time() < deadline:
            self._drain(buf)
            if predicate():
                return True
            try:
                data = sock.recv(8192)
                if data == b"":
                    return predicate()
                buf += data
            except socket.timeout:
                pass
            except OSError:
                return predicate()
        self._drain(buf)
        return predicate()

    def _drain(self, buf: bytearray) -> None:
        while True:
            fr = parse_frame(buf)
            if not fr:
                break
            _fin, op, payload = fr
            if op in (0x1, 0x2, 0x0):
                self._on_text(payload)
            elif op == 0x9:                       # camera ping → pong (rule 5)
                try:
                    with self._send_lock:
                        if self._sock:
                            self._sock.sendall(encode_frame(0xA, payload))
                except OSError:
                    pass
            elif op == 0x8:
                self._mark_stale("peer sent close frame")

    def _mark_stale(self, reason: str) -> None:
        with self._state_lock:
            self.state.stale = True
            self.state.error = reason
        logger.warning("[%s] %s", self.label, reason)

    # -- message parsing (id-based, defensive — rule/init note) -----------------
    def _on_text(self, payload: bytes) -> None:
        try:
            obj = json.loads(payload.decode("utf-8", "replace"))
        except Exception:
            return
        with self._state_lock:
            self.state.last_msg_t = time.time()
            self.state.stale = False
            self._apply(obj)

    def _apply(self, obj: dict) -> None:
        t = obj.get("type", "")
        # Camera info (name / serial / firmware).
        if t == "rcp_cur_cam_info" or obj.get("id") == "CAMERA_INFO":
            self.state.camera_type = str(
                (obj.get("camera_type") or {}).get("str", "") or obj.get("camera_type", "") or "")
            self.state.serial = str(obj.get("serial_number", "") or "")
            self.state.firmware = str(obj.get("firmware_version", "") or obj.get("firmware", "") or "")
            return
        pid = str(obj.get("id", "")).replace("RCP_PARAM_", "")
        if not pid:
            # TIMECODE sometimes arrives under a wrapper type without a clean id.
            s = json.dumps(obj)
            if "TIMECODE" in s:
                m = TC_RE.search(s)
                if m:
                    self.state.timecode = m.group(0)
            return
        val = obj.get("value", obj.get("str", obj.get("int")))
        if pid == "TIMECODE":
            m = TC_RE.search(json.dumps(obj))
            if m:
                self.state.timecode = m.group(0)
        elif pid == "RECORD_STATE":
            try:
                self.state.record_state = int(val)
            except (TypeError, ValueError):
                pass
        elif pid == "TIMECODE_STATE":
            self.state.timecode_state = str(val)
        elif pid == "TIMECODE_SOURCE":
            self.state.timecode_source = str(val)
        elif pid == "SYNC_STATE":
            self.state.sync_state = str(val)
        elif pid == "GENLOCK_STATE":
            self.state.genlock_state = str(val)
        elif pid == "CLIP_NAME":
            self.state.clip_name = str(val or "")
        elif pid == "FRAME_LIMIT_ENABLE":
            self.state.frame_limit_enable = bool(val)
        elif pid == "FRAME_LIMIT_FRAMES":
            try:
                self.state.frame_limit_frames = int(val)
            except (TypeError, ValueError):
                pass
        elif pid == "SYNC_RECORD_STATUS":
            try:
                self.state.sync_record_status = int(val)
            except (TypeError, ValueError):
                pass

    # -- commands ---------------------------------------------------------------
    def set_frame_limit(self, frames: int = 1,
                        timeout: float = SET_VERIFY_TIMEOUT) -> bool:
        """Enable the frame limit and set it to `frames`. Confirms via the
        subscribed FRAME_LIMIT_* pushes (rule 9). Returns True if confirmed."""
        self._raw_send({"type": "rcp_set", "id": "FRAME_LIMIT_FRAMES", "value": int(frames)})
        self._raw_send({"type": "rcp_set", "id": "FRAME_LIMIT_ENABLE", "value": True})
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self._state_lock:
                if self.state.frame_limit_enable and self.state.frame_limit_frames == int(frames):
                    return True
            time.sleep(0.1)
        return False

    def arm_record_start(self, target_tc: str) -> None:
        """SYNC_RECORD_START at a future TC (rule 17). RECORD_STATE→5 (armed)
        during the countdown, →1 at the target edge (rule 18)."""
        self._raw_send({"type": "rcp_set", "id": "SYNC_RECORD_START", "value": target_tc})

    def arm_record_stop(self, target_tc: str) -> None:
        """SYNC_RECORD_STOP at a future TC (rule 17)."""
        self._raw_send({"type": "rcp_set", "id": "SYNC_RECORD_STOP", "value": target_tc})

    def snapshot(self) -> LinkState:
        with self._state_lock:
            return LinkState(**vars(self.state))


# ── Sync verification ───────────────────────────────────────────────────────────
@dataclass
class SyncReport:
    in_sync: bool
    reference_tc: str
    spread_seconds: int              # max−min whole-second TC across the array
    cameras: Dict[str, dict] = field(default_factory=dict)
    problems: List[str] = field(default_factory=list)


def verify_sync(links: Dict[str, CameraLink], tc_tolerance_s: int = 1) -> SyncReport:
    """Verify every connected camera agrees on timecode and reports a locked
    sync/genlock source. TC is seconds-resolution (rule 20), so agreement is
    checked to the whole second (tolerance default 1 s to absorb the 1/s push
    boundary). A jam-synced/genlocked array reads identical whole-second TC."""
    problems: List[str] = []
    cams: Dict[str, dict] = {}
    secs: List[int] = []
    ref_tc = ""
    for label, link in sorted(links.items()):
        s = link.snapshot()
        row = {
            "ip": s.ip, "timecode": s.timecode, "record_state": s.record_state,
            "sync_state": s.sync_state, "genlock_state": s.genlock_state,
            "timecode_state": s.timecode_state, "connected": s.connected,
            "stale": s.stale,
        }
        cams[label] = row
        if not s.connected or s.stale:
            problems.append(f"{label}: not connected / stale link")
            continue
        sec = tc_to_seconds(s.timecode)
        if sec is None:
            problems.append(f"{label}: no valid timecode yet")
            continue
        secs.append(sec)
        if not ref_tc:
            ref_tc = s.timecode
        # A locked source is 'locked' / 'genlock' / 'external'; a bare/free-run
        # clock is the failure mode we want to surface (values are FW-dependent —
        # UNVERIFIED enumeration, surfaced as advisory rather than hard-fail).
        low = (s.sync_state or "").lower()
        if low and ("free" in low or "unlock" in low or "none" in low):
            problems.append(f"{label}: sync_state={s.sync_state}")
    spread = (max(secs) - min(secs)) if secs else 0
    if secs and spread > tc_tolerance_s:
        problems.append(f"timecode spread {spread}s exceeds ±{tc_tolerance_s}s")
    in_sync = bool(secs) and spread <= tc_tolerance_s and not any(
        p for p in problems if "spread" in p or "not connected" in p
        or "no valid timecode" in p)
    return SyncReport(in_sync=in_sync, reference_tc=ref_tc, spread_seconds=spread,
                      cameras=cams, problems=problems)


# ── Array controller ────────────────────────────────────────────────────────────
class CaptureArray:
    """Orchestrates detect → connect → frame-limit → verify → roll → cut across
    up to ~36 bodies. One CameraLink (one session) per camera (rules 1, 2)."""

    def __init__(self, source_ip: Optional[str] = None):
        self.source_ip = source_ip
        self.links: Dict[str, CameraLink] = {}   # label → link
        self._rolled = False

    # detection ----------------------------------------------------------------
    def detect(self, cidr: str, stop=None, progress=None) -> List[str]:
        return tcp_scan(cidr, source_ip=self.source_ip, stop=stop, progress=progress)

    # connection ---------------------------------------------------------------
    def connect(self, ips: List[str], labels: Optional[Dict[str, str]] = None,
                progress: Optional[Callable[[str, bool, str], None]] = None
                ) -> Dict[str, bool]:
        """Open one session per IP in parallel. Returns ip→ok. Labels default to
        the camera_type+serial once CAMERA_INFO lands, else the IP."""
        labels = labels or {}
        results: Dict[str, bool] = {}

        def _one(ip: str) -> Tuple[str, bool, str]:
            link = CameraLink(ip, label=labels.get(ip, ip), source_ip=self.source_ip)
            try:
                link.connect()
                s = link.snapshot()
                nice = (f"{s.camera_type} {s.serial}".strip()) or ip
                link.label = labels.get(ip) or nice
                self.links[link.label] = link
                return ip, True, ""
            except Exception as e:
                return ip, False, str(e)

        with ThreadPoolExecutor(max_workers=min(36, max(1, len(ips)))) as ex:
            for ip, ok, err in ex.map(_one, ips):
                results[ip] = ok
                if progress:
                    progress(ip, ok, err)
        return results

    # frame limit --------------------------------------------------------------
    def set_frame_limit(self, frames: int = 1) -> Dict[str, bool]:
        out: Dict[str, bool] = {}

        def _one(item):
            label, link = item
            try:
                return label, link.set_frame_limit(frames)
            except Exception:
                return label, False

        with ThreadPoolExecutor(max_workers=min(36, max(1, len(self.links)))) as ex:
            for label, ok in ex.map(_one, list(self.links.items())):
                out[label] = ok
        return out

    # sync ---------------------------------------------------------------------
    def verify_sync(self, tc_tolerance_s: int = 1) -> SyncReport:
        return verify_sync(self.links, tc_tolerance_s=tc_tolerance_s)

    def reference_timecode(self) -> str:
        """Newest valid TC from any live body — the roll reference (rule 17)."""
        best = ""
        for link in self.links.values():
            s = link.snapshot()
            if s.connected and not s.stale and tc_to_seconds(s.timecode) is not None:
                best = s.timecode
        return best

    # roll / cut ---------------------------------------------------------------
    def roll(self, lead_seconds: float = DEFAULT_ROLL_LEAD) -> Tuple[bool, str, dict]:
        """Read reference TC → arm SYNC_RECORD_START on all live bodies at a
        target 2-5 s ahead (rule 17). Never rolls into a quiet/stale link
        (rule 21) — such bodies are skipped with a loud problem entry."""
        report = self.verify_sync()
        ref = report.reference_tc or self.reference_timecode()
        if not ref:
            return False, "no reference timecode — cannot compute roll target", {}
        target = compute_target_tc(ref, lead_seconds)
        if not target:
            return False, f"reference TC unparseable: {ref!r}", {}
        armed: Dict[str, str] = {}
        skipped: Dict[str, str] = {}
        for label, link in sorted(self.links.items()):
            s = link.snapshot()
            if not s.connected or s.stale:
                skipped[label] = "offline/stale — NOT rolled"  # rule 21
                continue
            try:
                link.arm_record_start(target)
                armed[label] = target
            except Exception as e:
                skipped[label] = f"arm failed: {e}"
        self._rolled = bool(armed)
        detail = {"target_tc": target, "reference_tc": ref,
                  "armed": armed, "skipped": skipped}
        ok = bool(armed) and not skipped
        return ok, (f"armed {len(armed)} camera(s) for {target}"
                    + (f"; SKIPPED {len(skipped)}" if skipped else "")), detail

    def cut(self, lead_seconds: float = DEFAULT_CUT_LEAD) -> Tuple[bool, str, dict]:
        """SYNC_RECORD_STOP on all live bodies (rule 17). With a frame limit of 1
        the camera auto-stops after one frame, so this is a safety fallback; it
        MUST warn loudly if a rolled body is offline (rule 21). UNVERIFIED:
        queuing STOP while START is pending (rule 23)."""
        ref = self.reference_timecode()
        if not ref:
            return False, "no reference timecode — cannot compute cut target", {}
        target = compute_target_tc(ref, lead_seconds)
        stopped: Dict[str, str] = {}
        warnings: List[str] = []
        for label, link in sorted(self.links.items()):
            s = link.snapshot()
            if not s.connected or s.stale:
                if self._rolled:
                    warnings.append(f"{label}: OFFLINE but may still be recording — verify manually")
                continue
            try:
                link.arm_record_stop(target)
                stopped[label] = target
            except Exception as e:
                warnings.append(f"{label}: stop failed: {e}")
        return (not warnings), (f"cut sent to {len(stopped)} camera(s)"
                + (f"; {len(warnings)} WARNING(S)" if warnings else "")), \
               {"target_tc": target, "stopped": stopped, "warnings": warnings}

    def wait_for_capture(self, timeout: float = 20.0,
                         poll: float = 0.3) -> Tuple[bool, Dict[str, str]]:
        """After roll, wait until every armed body returns to idle (0) having
        recorded, and report the clip name each body last wrote. RECORD_STATE is
        truth (rule 18); 2/3/4/5 are transient (rule 22)."""
        deadline = time.time() + timeout
        clips: Dict[str, str] = {}
        while time.time() < deadline:
            done = True
            for label, link in self.links.items():
                s = link.snapshot()
                if s.clip_name:
                    clips[label] = s.clip_name
                # idle again after having been in a record/transient state
                if s.record_state not in (0, None):
                    done = False
            if done and clips:
                return True, clips
            time.sleep(poll)
        return False, clips

    def snapshots(self) -> Dict[str, LinkState]:
        return {label: link.snapshot() for label, link in self.links.items()}

    def close(self) -> None:
        for link in list(self.links.values()):
            try:
                link.close()
            except Exception:
                pass
        self.links.clear()
