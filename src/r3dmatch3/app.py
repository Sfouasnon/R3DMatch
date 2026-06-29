"""
app.py — R3DMatch v4 PySide6 Desktop Application
=================================================
6 screens via persistent tab bar:
  0  Setup          — folder picker, strategy
  1  Progress       — live pipeline phases + camera log
  2  Sphere QC      — verify/correct sphere solves, gate assessment
  3  Results        — HTML contact sheet in WebEngineView
  4  Push           — per-camera RCP2 push with dry-run
  5  Camera Network — IP map + connection test

Flow:
  Setup → Run → Progress → (auto-advance) → Sphere QC
  Sphere QC → Accept All / Correct individual cameras → Create Assessment
  Create Assessment → Results → Push

Sphere QC re-measure:
  Operator places circle on thumbnail via drag gesture.
  App calls workflow_qc.remeasure_cameras() in a QThread with corrected ROIs.
  RunResult is patched in-place, Results screen reloads.
"""

from __future__ import annotations

import inspect
import ipaddress
import json
import re
import socket
import subprocess
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PySide6.QtCore import (
    QPointF, QRectF, QThread, Qt, QUrl, Signal, Slot,
)
from PySide6.QtGui import (
    QColor, QCursor, QPainter, QPen, QPalette,
)
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDialog, QFileDialog,
    QFrame, QGraphicsEllipseItem, QGraphicsPixmapItem,
    QGraphicsScene, QGraphicsView, QHBoxLayout, QHeaderView,
    QLabel, QLineEdit, QMainWindow, QMessageBox,
    QPushButton, QScrollArea, QSizePolicy, QSlider, QStackedWidget,
    QTableWidget, QTableWidgetItem, QTextEdit, QVBoxLayout,
    QWidget, QGridLayout,
)
try:
    from PySide6.QtWebEngineWidgets import QWebEngineView
    _WEB_OK = True
except ImportError:
    _WEB_OK = False

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from r3dmatch3.workflow import run_analysis, verify_run
from r3dmatch3.colorpipeline import ColorPipeline
from r3dmatch3.workflow_qc import remeasure_cameras
from r3dmatch3.report import build_report
from r3dmatch3.rcp2 import (push_all_cameras_sync, reset_all_cameras_sync,
                            summarize_push_results)
from r3dmatch3.redline import REDLineNotFoundError, resolve_redline_executable, check_redline_available
from r3dmatch3.models import SphereROI
from r3dmatch3.settings import load_settings, save_settings

# ── Constants ──────────────────────────────────────────────────────────────────

APP_NAME    = "R3DMatch v4"
APP_VERSION = "4.0"
CONFIG_DIR  = Path.home() / ".config" / "R3DMatch_v3"
IP_MAP_PATH = CONFIG_DIR / "camera_ips.json"

# IPP2 delivery-transform options (label, REDLine code). Reference = first/marked.
_IPP2_COLOR_SPACES = [
    ("BT.709", "13"), ("BT.2020", "24"), ("DCI-P3", "26"),
    ("DCI-P3 D65", "27"), ("REDWideGamutRGB", "25")]
_IPP2_GAMMAS = [
    ("BT.1886", "32"), ("Gamma 2.2", "36"), ("Gamma 2.6", "37"), ("sRGB", "2"),
    ("Hybrid Log-Gamma", "35"), ("ST.2084 (PQ)", "31"), ("Log3G10", "34")]
_IPP2_TONEMAPS = [("Low", "0"), ("Medium", "1"), ("High", "2"), ("None", "3")]
_IPP2_ROLLOFFS = [("None", "0"), ("Hard", "1"), ("Default", "2"),
                  ("Medium", "3"), ("Soft", "4")]

# ── Theme: warm graphite + RED crimson (paired with docs/ui_mockup_v3.html) ──
DARK_BG       = "#100d0a"   # app background
PANEL_BG      = "#1b1612"   # panels / cards level 1
CARD_BG       = "#231c16"   # cards level 2 / hover
BORDER_COLOR  = "#322920"
BORDER_STRONG = "#3f352a"
# Settings-card surfaces: a clearly lighter card on a dark inset input, so each
# section reads as a distinct block instead of bleeding into the background.
_CARD_SURFACE = "#241d15"   # section card fill (lighter than page)
_CARD_BORDER  = "#5a4a34"   # stronger visible edge
_INSET_BG     = "#15110c"   # input/control fill (darker — reads as inset)
ACCENT        = "#e3242b"   # RED crimson — actions, current step
ACCENT_HOVER  = "#ff4d4d"
ACCENT_DEEP   = "#8f1318"   # gradient tail / pressed
ACCENT_DIM    = "#e3242b33"
SUCCESS       = "#34d399"
WARNING       = "#fbbf24"
DANGER        = "#ff7a8a"   # low-match coral — never confused with accent
C_BLUE        = "#e8d9b5"   # info/anchor cream (legacy name kept)
TEXT_PRIMARY  = "#f0e9df"
TEXT_MUTED    = "#ab9f8f"
TEXT_DIM      = "#766a59"
MONO_FONT     = "SF Mono, IBM Plex Mono, Menlo, Courier New, monospace"
BG            = DARK_BG   # alias used in dialogs

STRATEGIES = [
    ("median",      "Median  —  robust, prevents outlier distortion"),
    ("optimal_ire", "Optimal IRE  —  target a specific gray level"),
]

_RUN_ANALYSIS_PARAMS = set(inspect.signature(run_analysis).parameters.keys())


# ── App state ──────────────────────────────────────────────────────────────────

class AppState:
    def __init__(self):
        self.input_path:   Optional[str]  = None
        self.out_dir:      Optional[str]  = None
        self.strategy:     str            = "median"
        self.disable_priors: bool         = False
        # Delivery look — must be explicitly chosen before each run.
        # None = reference-only; a ColorPipeline = score through that look too.
        self.delivery_pipeline: Optional[ColorPipeline] = None
        self.delivery_label: str          = ""
        self.delivery_chosen: bool        = False
        # White-balance mode: "match" (group neutral, proven) | "neutral"
        self.wb_mode: str                 = "match"
        self.run_result                   = None
        self.report_path:  Optional[str]  = None
        self.camera_ips:   Dict[str, str] = {}
        self.push_results: List           = []
        # Calibration commit — set when the operator commits on the Match screen,
        # carrying the per-camera CommitValues forward to the Push screen.
        self.calibration_committed: bool  = False
        self.committed_at: Optional[str]  = None
        # Sphere QC state
        self.qc_rois:      Dict[str, SphereROI] = {}  # clip_id → corrected ROI
        self.qc_accepted:  Dict[str, bool]       = {}  # clip_id → operator accepted
        # Persistent settings
        _s = load_settings()
        self.redline_path:    str = _s.get("redline_path", "")
        self.default_out_dir: str = _s.get("default_out_dir", "")


# ── Workers ────────────────────────────────────────────────────────────────────

class AnalysisWorker(QThread):
    progress = Signal(dict)
    finished = Signal(object)
    errored  = Signal(str)

    def __init__(
        self,
        input_path: str,
        out_dir: str,
        strategy: str,
        disable_priors: bool = False,
        wb_mode: str = "match",
    ):
        super().__init__()
        self.input_path = input_path
        self.out_dir    = out_dir
        self.strategy   = strategy
        self.disable_priors = disable_priors
        self.wb_mode    = wb_mode

    def run(self):
        import io
        import sys as _sys

        # Intercept stdout so workflow JSON progress lines reach the UI
        class _ProgressCapture(io.TextIOBase):
            def __init__(self, signal, original):
                self._signal   = signal
                self._original = original
                self._buf      = ""

            def write(self, s):
                try:
                    self._original.write(s)
                except (BrokenPipeError, OSError):
                    pass
                self._buf += s
                while "\n" in self._buf:
                    line, self._buf = self._buf.split("\n", 1)
                    line = line.strip()
                    if line.startswith("{"):
                        try:
                            self._signal.emit(json.loads(line))
                        except Exception:
                            pass
                return len(s)

            def flush(self):
                try:
                    self._original.flush()
                except (BrokenPipeError, OSError):
                    pass

        original_stdout = _sys.stdout
        _sys.stdout = _ProgressCapture(self.progress, original_stdout)
        try:
            kwargs = {"out_dir": self.out_dir, "render_corrected": False}
            if "strategy" in _RUN_ANALYSIS_PARAMS:
                kwargs["strategy"] = self.strategy
            if "anchor_source" in _RUN_ANALYSIS_PARAMS:
                kwargs["anchor_source"] = self.strategy
            if "disable_priors" in _RUN_ANALYSIS_PARAMS:
                kwargs["disable_priors"] = self.disable_priors
            if "wb_mode" in _RUN_ANALYSIS_PARAMS:
                kwargs["wb_mode"] = self.wb_mode
            result = run_analysis(self.input_path, **kwargs)
            self.finished.emit(result)
        except Exception as exc:
            self.errored.emit(f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}")
        finally:
            _sys.stdout = original_stdout


class RemeasureWorker(QThread):
    progress = Signal(str, str)   # phase, detail
    finished = Signal(object)     # patched RunResult
    errored  = Signal(str)

    def __init__(self, run_result, corrected_rois: Dict[str, SphereROI]):
        super().__init__()
        self.run_result     = run_result
        self.corrected_rois = corrected_rois

    def run(self):
        try:
            def _cb(phase, detail, clip_id=""):
                self.progress.emit(phase, f"{clip_id + ': ' if clip_id else ''}{detail}")
            result = remeasure_cameras(
                self.run_result,
                self.corrected_rois,
                progress_callback=_cb,
            )
            self.finished.emit(result)
        except Exception as exc:
            self.errored.emit(f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}")


class VerifyWorker(QThread):
    """Closed-loop verification post-QC: corrected renders → measure →
    match % → report. Runs before the Results screen is shown."""
    progress = Signal(dict)
    finished = Signal(object, object)   # run_result, report_path (str | None)
    errored  = Signal(str)

    def __init__(self, run_result, out_dir: str, redline_path: str = "",
                 delivery_pipeline=None):
        super().__init__()
        self.run_result   = run_result
        self.out_dir      = out_dir
        self.redline_path = redline_path
        self.delivery_pipeline = delivery_pipeline

    def run(self):
        import io
        import sys as _sys

        worker = self

        class _Capture(io.TextIOBase):
            def __init__(self, original):
                self._original = original
                self._buf = ""

            def write(self, s):
                try:
                    self._original.write(s)
                except (BrokenPipeError, OSError):
                    pass
                self._buf += s
                while "\n" in self._buf:
                    line, self._buf = self._buf.split("\n", 1)
                    line = line.strip()
                    if line.startswith("{"):
                        try:
                            worker.progress.emit(json.loads(line))
                        except Exception:
                            pass
                return len(s)

            def flush(self):
                try:
                    self._original.flush()
                except (BrokenPipeError, OSError):
                    pass

        original_stdout = _sys.stdout
        _sys.stdout = _Capture(original_stdout)
        try:
            rr = verify_run(self.run_result, redline_path=self.redline_path,
                            delivery_pipeline=self.delivery_pipeline)
            report_path = None
            try:
                report_path = build_report(rr, self.out_dir)
                if report_path and not Path(report_path).exists():
                    report_path = None
            except Exception:
                report_path = None
            self.finished.emit(rr, report_path)
        except Exception as exc:
            self.errored.emit(f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}")
        finally:
            _sys.stdout = original_stdout


class PushWorker(QThread):
    camera_update = Signal(str, bool, str)
    finished      = Signal(list)
    errored       = Signal(str)

    def __init__(self, targets: list, dry_run: bool = False, mode: str = "push"):
        super().__init__()
        self.targets = targets
        self.dry_run = dry_run
        self.mode = mode          # "push" → calibration values; "reset" → neutral defaults

    def run(self):
        try:
            if self.mode == "reset":
                results = reset_all_cameras_sync(self.targets, dry_run=self.dry_run)
                ok_word = "reset to default"
            else:
                results = push_all_cameras_sync(self.targets, dry_run=self.dry_run)
                ok_word = "pushed"
            for r in results:
                self.camera_update.emit(
                    r.camera_label, r.success,
                    r.error or ("dry-run OK" if r.dry_run else ok_word),
                )
            self.finished.emit(results)
        except Exception as exc:
            self.errored.emit(f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}")


# ── Delivery look picker ─────────────────────────────────────────────────────

def prompt_delivery_look(parent):
    """Modal, required-before-each-run delivery-look picker.

    Returns (ok, pipeline, label):
      ok=False        -> operator cancelled; caller must abort the run.
      pipeline=None   -> 'Reference only' (score in BT.709 reference look).
      pipeline=ColorPipeline -> score also through the chosen project look.

    There is no default — the operator must pick every run so a run never
    proceeds on an assumed look.
    """
    dlg = QDialog(parent)
    dlg.setWindowTitle("Select Delivery Look")
    dlg.setModal(True)
    lay = QVBoxLayout(dlg)
    lay.setContentsMargins(20, 18, 20, 16)
    lay.setSpacing(10)
    lay.addWidget(_label("How should this run be scored?", size=14, bold=True))
    lay.addWidget(_label(
        "Exposure / Kelvin / tint commits come from the BT.709 reference solve "
        "either way. A project look adds an operator-facing delivery match %.",
        size=11, color=TEXT_MUTED))

    combo = QComboBox()
    combo.addItem("Reference only (BT.709)", userData="reference")
    combo.addItem("Project show LUT (.cube)…", userData="lut")
    lay.addWidget(combo)

    lut_row = QHBoxLayout()
    lut_path_lbl = _label("No LUT selected", size=11, color=TEXT_MUTED)
    browse_btn = QPushButton("Browse…")
    lut_row.addWidget(lut_path_lbl, 1)
    lut_row.addWidget(browse_btn)
    lut_widget = QWidget()
    lut_widget.setLayout(lut_row)
    lut_widget.setVisible(False)
    lay.addWidget(lut_widget)

    state = {"lut_path": ""}

    def _on_combo(_idx):
        lut_widget.setVisible(combo.currentData() == "lut")
        dlg.adjustSize()
    combo.currentIndexChanged.connect(_on_combo)

    def _browse():
        p, _ = QFileDialog.getOpenFileName(
            dlg, "Select Project Show LUT", "", "3D LUT (*.cube *.CUBE)")
        if p:
            state["lut_path"] = p
            lut_path_lbl.setText(Path(p).name)
    browse_btn.clicked.connect(_browse)

    btns = QHBoxLayout()
    btns.addStretch()
    cancel = QPushButton("Cancel")
    ok = QPushButton("Start Run")
    ok.setDefault(True)
    btns.addWidget(cancel)
    btns.addWidget(ok)
    lay.addLayout(btns)
    cancel.clicked.connect(dlg.reject)
    ok.clicked.connect(dlg.accept)

    if dlg.exec() != QDialog.Accepted:
        return (False, None, "")

    if combo.currentData() == "reference":
        return (True, None, "Reference only (BT.709)")
    if not state["lut_path"]:
        QMessageBox.warning(parent, "No LUT",
                            "Choose a .cube LUT or select 'Reference only'.")
        return (False, None, "")
    name = Path(state["lut_path"]).stem
    pipeline = ColorPipeline(creative_lut_path=state["lut_path"], name=f"LUT: {name}")
    return (True, pipeline, f"LUT: {name}")


# ── Style helpers ──────────────────────────────────────────────────────────────

def _label(text: str, size: int = 13, color: str = TEXT_PRIMARY,
           bold: bool = False, mono: bool = False) -> QLabel:
    lbl = QLabel(text)
    f = lbl.font()
    f.setPointSize(size)
    if bold: f.setBold(True)
    if mono: f.setFamily(MONO_FONT)
    lbl.setFont(f)
    lbl.setStyleSheet(f"color:{color};background:transparent;")
    return lbl


def _button(text: str, primary: bool = False, danger: bool = False,
            warning: bool = False) -> QPushButton:
    btn = QPushButton(text)
    if primary:
        s = (f"QPushButton{{background:{ACCENT};color:#fff;border:none;"
             f"border-radius:6px;padding:8px 18px;font-weight:600;font-size:13px;}}"
             f"QPushButton:hover{{background:#c41d23;}}"
             f"QPushButton:pressed{{background:#8f1318;}}"
             f"QPushButton:disabled{{background:{CARD_BG};color:{TEXT_MUTED};"
             f"border:1px solid {BORDER_COLOR};}}")
    elif danger:
        s = (f"QPushButton{{background:{DANGER};color:#fff;border:none;"
             f"border-radius:6px;padding:8px 18px;font-weight:600;font-size:13px;}}"
             f"QPushButton:hover{{background:#8f1318;}}"
             f"QPushButton:disabled{{background:{CARD_BG};color:{TEXT_MUTED};}}")
    elif warning:
        s = (f"QPushButton{{background:{WARNING};color:#000;border:none;"
             f"border-radius:6px;padding:8px 18px;font-weight:600;font-size:13px;}}"
             f"QPushButton:hover{{background:#b8151c;}}"
             f"QPushButton:disabled{{background:{CARD_BG};color:{TEXT_MUTED};}}")
    else:
        s = (f"QPushButton{{background:{CARD_BG};color:{TEXT_PRIMARY};"
             f"border:1px solid {BORDER_STRONG};border-radius:6px;"
             f"padding:8px 18px;font-size:13px;}}"
             f"QPushButton:hover{{background:{BORDER_COLOR};border-color:{BORDER_STRONG};}}"
             f"QPushButton:disabled{{color:{TEXT_MUTED};}}")
    btn.setStyleSheet(s)
    return btn


def _field_style(mono: bool = False) -> str:
    fp = f"font-family:{MONO_FONT};" if mono else ""
    return (f"background:{CARD_BG};color:{TEXT_PRIMARY};"
            f"border:1px solid {BORDER_STRONG};border-radius:6px;"
            f"padding:8px 10px;font-size:13px;{fp}"
            f"selection-background-color:{ACCENT_DIM};")


def _sep() -> QFrame:
    f = QFrame()
    f.setFrameShape(QFrame.Shape.HLine)
    f.setStyleSheet(f"color:{BORDER_COLOR};")
    return f


def _table_style() -> str:
    return (f"QTableWidget{{background:{PANEL_BG};color:{TEXT_PRIMARY};"
            f"border:1px solid {BORDER_COLOR};border-radius:6px;"
            f"gridline-color:{BORDER_COLOR};}}"
            f"QTableWidget::item:selected{{background:{ACCENT_DIM};color:{TEXT_PRIMARY};}}"
            f"QHeaderView::section{{background:{CARD_BG};color:{TEXT_MUTED};"
            f"border:none;border-bottom:1px solid {BORDER_COLOR};"
            f"padding:6px 8px;font-size:10px;font-weight:700;"
            f"text-transform:uppercase;letter-spacing:0.05em;}}")


# ── Camera-network discovery helpers ────────────────────────────────────────────
#
# Reliable detection model ported from R3DMatch v1 (camera_network.py), which was
# validated against a live KOMODO-X. We enumerate real interfaces via `ifconfig`
# instead of socket.gethostbyname(socket.gethostname()) — the latter returns
# 127.0.0.1 (or raises) on macOS and can never surface a direct-attached camera on
# a 169.254.x.x link-local interface.

DEFAULT_CAMERA_CIDR = "172.20.114.0/24"   # current stage network (255.255.255.0)
RCP2_SCAN_PORT = 9998
MAX_SCAN_HOSTS = 1024

_IP_RE = re.compile(r"^\d{1,3}(?:\.\d{1,3}){3}$")


def _normalize_mask(token: str) -> str:
    """Convert a netmask token (dotted or 0x-hex, as ifconfig prints) to dotted."""
    token = str(token or "").strip()
    if not token:
        return ""
    if token.startswith("0x"):
        try:
            return socket.inet_ntoa(int(token, 16).to_bytes(4, "big"))
        except Exception:
            return ""
    return token


def _list_ipv4_interfaces() -> List[Dict[str, str]]:
    """Enumerate non-loopback IPv4 interfaces via `ifconfig` (macOS/BSD)."""
    try:
        out = subprocess.run(
            ["/sbin/ifconfig"], check=True, capture_output=True, text=True
        ).stdout
    except Exception:
        return []
    interfaces: List[Dict[str, str]] = []
    name = ""
    for raw in out.splitlines():
        if raw and not raw[0].isspace():
            name = raw.split(":", 1)[0].strip()
            continue
        line = raw.strip()
        if not line.startswith("inet "):
            continue
        parts = line.split()
        addr = parts[1].strip()
        if addr.startswith("127."):
            continue
        mask = ""
        if "netmask" in parts:
            try:
                mask = _normalize_mask(parts[parts.index("netmask") + 1])
            except Exception:
                mask = ""
        try:
            net = ipaddress.IPv4Network(f"{addr}/{mask}", strict=False) if mask \
                else ipaddress.IPv4Network(f"{addr}/24", strict=False)
        except Exception:
            continue
        interfaces.append({
            "name": name,
            "ipv4": addr,
            "mask": mask,
            "cidr": str(net),
            "link_local": addr.startswith("169.254."),
        })
    return interfaces


def _interface_scan_cidr(itf: Dict[str, str]) -> str:
    """Scan target for one interface, capped to a /24-sized sweep.

    Camera networks are frequently link-local /16 (255.255.0.0) — scanning the
    whole /16 is 65,534 hosts. We scan the /24 block that contains the host's own
    address instead (e.g. NIC 169.245.5.x → 169.245.5.0/24), which keeps the
    sweep <=254 hosts and covers a directly-attached camera on the same block.
    A wider/narrower range can still be typed manually in the dialog.
    """
    try:
        prefix = ipaddress.IPv4Network(itf["cidr"]).prefixlen
    except Exception:
        prefix = 24
    if prefix < 24:
        return str(ipaddress.ip_network(f"{itf['ipv4']}/24", strict=False))
    return itf["cidr"]


def _candidate_scan_cidrs() -> List[str]:
    """Ordered, de-duplicated scan targets: stage network first, then live NICs.

    Link-local interfaces (direct-attached cameras, no DHCP) are promoted ahead
    of ordinary LAN interfaces.
    """
    ordered: List[str] = [DEFAULT_CAMERA_CIDR]
    link_local: List[str] = []
    normal: List[str] = []
    for itf in _list_ipv4_interfaces():
        cidr = _interface_scan_cidr(itf)
        (link_local if itf["link_local"] else normal).append(cidr)
    seen = set()
    result: List[str] = []
    for cidr in ordered + link_local + normal:
        if cidr not in seen:
            seen.add(cidr)
            result.append(cidr)
    return result


def _hosts_for_cidr(cidr: str) -> List[str]:
    """Host addresses for a CIDR, capped at MAX_SCAN_HOSTS."""
    net = ipaddress.IPv4Network(cidr, strict=False)
    hosts = [str(h) for h in net.hosts()] or [str(net.network_address)]
    if len(hosts) > MAX_SCAN_HOSTS:
        raise ValueError(
            f"Scan range too large ({len(hosts)} hosts). "
            f"Limit to {MAX_SCAN_HOSTS} or fewer.")
    return hosts


def _parse_bulk_pairs(text: str) -> List[Tuple[str, str]]:
    """Parse pasted label/IP rows tolerantly.

    Accepts tab, comma, semicolon, or whitespace separators, either column
    order (label-first or IP-first), and skips blank/header lines that contain
    no IP address.
    """
    pairs: List[Tuple[str, str]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        tokens = [t for t in re.split(r"[\t,;]+|\s+", line) if t]
        if not tokens:
            continue
        ip = next((t for t in tokens if _IP_RE.match(t)), None)
        if ip is None:
            continue   # header row or junk — no IP present
        labels = [t for t in tokens if t != ip]
        pairs.append((labels[0] if labels else ip, ip))
    return pairs


class _NetworkScanWorker(QThread):
    """Background RCP2 camera scan — keeps the UI thread responsive.

    Phase 1: fast concurrent TCP probe of port 9998 across the host list.
    Phase 2: confirm each open host is a camera via a real RCP2 session
             (reads CAMERA_INFO), so an unrelated service on 9998 is not
             mistaken for a camera.
    """
    found = Signal(str, str)        # ip, info string ("" until confirmed)
    progress = Signal(int, int)     # done, total
    done = Signal(list)             # [(ip, info), ...] confirmed/open

    def __init__(self, hosts: List[str], port: int = RCP2_SCAN_PORT,
                 timeout: float = 0.3):
        super().__init__()
        self._hosts = hosts
        self._port = port
        self._timeout = timeout
        self._stop = False

    def stop(self):
        self._stop = True

    def _probe(self, ip: str) -> Optional[str]:
        if self._stop:
            return None
        try:
            socket.create_connection((ip, self._port), timeout=self._timeout).close()
            return ip
        except Exception:
            return None

    def _confirm(self, ip: str) -> str:
        """Best-effort RCP2 identity read. Returns a label or '' if unconfirmed."""
        import asyncio
        from r3dmatch3.rcp2 import RCP2Session

        async def _ping():
            s = RCP2Session(ip, ip, dry_run=False,
                            connect_timeout=3.0, handshake_timeout=3.0)
            try:
                await s.connect()
                info = await s.get_camera_info()
                cam = (info.get("camera_type") or {}).get("str", "")
                sn = info.get("serial_number", "")
                return f"{cam} {sn}".strip()
            finally:
                await s.close()
        try:
            return asyncio.run(_ping())
        except Exception:
            return ""

    def run(self):
        import concurrent.futures
        open_hosts: List[str] = []
        total = len(self._hosts)
        done_n = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=64) as ex:
            futs = {ex.submit(self._probe, h): h for h in self._hosts}
            for fut in concurrent.futures.as_completed(futs):
                if self._stop:
                    break
                done_n += 1
                ip = fut.result()
                if ip:
                    open_hosts.append(ip)
                    self.found.emit(ip, "")
                self.progress.emit(done_n, total)
        open_hosts.sort(key=lambda x: tuple(int(p) for p in x.split(".")))
        confirmed: List[Tuple[str, str]] = []
        for ip in open_hosts:
            if self._stop:
                break
            info = self._confirm(ip)
            confirmed.append((ip, info))
            self.found.emit(ip, info)
        self.done.emit(confirmed)


# ── Screen 0: Setup ────────────────────────────────────────────────────────────

class SetupScreen(QWidget):
    run_requested = Signal(str, str, str)

    def __init__(self, state: AppState):
        super().__init__()
        self.state = state
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(0)
        root.setContentsMargins(0, 0, 0, 0)

        # ── Top header band ───────────────────────────────────────────────────
        hdr = QWidget()
        hdr.setFixedHeight(64)
        hdr.setStyleSheet(
            f"background:{DARK_BG};border-bottom:1px solid {BORDER_COLOR};")
        hl = QHBoxLayout(hdr)
        hl.setContentsMargins(32, 0, 32, 0)
        title_lbl = _label("Ingest", size=18, bold=True)
        sub_lbl = _label(
            "Point at the card folder — one R3D frame per camera is enough. "
            "Labels and metadata extract automatically.",
            size=12, color=TEXT_MUTED)
        vl = QVBoxLayout()
        vl.setSpacing(2)
        vl.addWidget(title_lbl)
        vl.addWidget(sub_lbl)
        hl.addLayout(vl)
        hl.addStretch()
        root.addWidget(hdr)

        # ── Scrollable body ───────────────────────────────────────────────────
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(
            f"QScrollArea{{border:none;background:{DARK_BG};}}"
            f"QScrollBar:vertical{{background:{PANEL_BG};width:8px;border-radius:4px;}}"
            f"QScrollBar::handle:vertical{{background:{BORDER_STRONG};border-radius:4px;}}")
        body = QWidget()
        body.setStyleSheet(f"background:{DARK_BG};")
        bl = QVBoxLayout(body)
        bl.setSpacing(16)
        bl.setContentsMargins(32, 24, 32, 32)

        # Each configuration section is ONE unified card: header + description +
        # its control(s) on a single lighter surface, with darker inset inputs.
        # This stops the sections from bleeding into the monotone background.
        def inset_field():
            return (f"background:{_INSET_BG};color:{TEXT_PRIMARY};"
                    f"border:1px solid {_CARD_BORDER};border-radius:6px;"
                    f"padding:8px 10px;font-size:13px;"
                    f"selection-background-color:{ACCENT_DIM};")

        def inset_combo():
            return (f"QComboBox{{background:{_INSET_BG};color:{TEXT_PRIMARY};"
                    f"border:1px solid {_CARD_BORDER};border-radius:6px;"
                    f"padding:8px 10px;font-size:13px;}}"
                    f"QComboBox:hover{{border-color:{ACCENT};}}"
                    f"QComboBox::drop-down{{border:none;width:24px;}}"
                    f"QComboBox QAbstractItemView{{background:{PANEL_BG};"
                    f"color:{TEXT_PRIMARY};border:1px solid {BORDER_COLOR};"
                    f"selection-background-color:{ACCENT_DIM};}}")

        def section(title, desc):
            card = QWidget()
            card.setObjectName("secCard")
            card.setStyleSheet(
                f"QWidget#secCard{{background:{_CARD_SURFACE};"
                f"border:1px solid {_CARD_BORDER};border-left:3px solid {ACCENT};"
                f"border-radius:10px;}}")
            v = QVBoxLayout(card)
            v.setContentsMargins(16, 13, 16, 15)
            v.setSpacing(10)
            tl = _label(title, size=10, bold=True, color=TEXT_PRIMARY)
            tl.setStyleSheet(
                f"color:{TEXT_PRIMARY};background:transparent;border:none;"
                f"font-weight:800;letter-spacing:0.1em;font-size:11px;")
            v.addWidget(tl)
            dl = _label(desc, size=11, color=TEXT_MUTED)
            dl.setWordWrap(True)
            dl.setStyleSheet(
                f"color:{TEXT_MUTED};background:transparent;border:none;font-size:11px;")
            v.addWidget(dl)
            bl.addWidget(card)
            return v

        # ── Input folder ───────────────────────────────────────────────────
        v = section("INPUT FOLDER",
                    "Path to folder containing .R3D clips from all cameras in the array")
        row = QHBoxLayout(); row.setSpacing(8); row.setContentsMargins(0, 0, 0, 0)
        self._folder = QLineEdit()
        self._folder.setPlaceholderText("e.g. /Volumes/RAW/A007_Jun06/")
        self._folder.setStyleSheet(inset_field())
        b = _button("Browse…"); b.clicked.connect(self._browse_folder)
        row.addWidget(self._folder); row.addWidget(b)
        v.addLayout(row)

        # ── Output folder ──────────────────────────────────────────────────
        v = section("OUTPUT FOLDER",
                    "Where renders, the calibration report, and pipeline logs are written")
        row2 = QHBoxLayout(); row2.setSpacing(8); row2.setContentsMargins(0, 0, 0, 0)
        self._out = QLineEdit()
        self._out.setPlaceholderText("e.g. /Volumes/RAW/A007_Jun06_r3dmatch_out/")
        self._out.setStyleSheet(inset_field())
        if self.state.default_out_dir:
            self._out.setText(self.state.default_out_dir)
        b2 = _button("Browse…"); b2.clicked.connect(self._browse_out)
        row2.addWidget(self._out); row2.addWidget(b2)
        v.addLayout(row2)

        # ── Matching strategy ──────────────────────────────────────────────
        v = section("MATCHING STRATEGY",
                    "How the exposure anchor IRE is derived from the camera array")
        self._strategy = QComboBox()
        self._strategy.setStyleSheet(inset_combo())
        for key, lbl in STRATEGIES:
            self._strategy.addItem(lbl, userData=key)
        self._strategy.currentIndexChanged.connect(self._on_strategy_changed)
        v.addWidget(self._strategy)
        self._strat_desc = _label("", size=11, color=TEXT_MUTED)
        self._strat_desc.setWordWrap(True)
        self._strat_desc.setStyleSheet(
            f"color:{TEXT_MUTED};background:transparent;border:none;font-size:11px;")
        v.addWidget(self._strat_desc)
        self._on_strategy_changed(0)

        # ── Delivery look — explicit, required before each run ─────────────
        v = section("DELIVERY LOOK",
                    "The IPP2 output the operator-facing match % is scored through. "
                    "Exposure / Kelvin / tint commits always come from the BT.709 reference solve.")
        self._delivery_lut_path = ""
        self._delivery_combo = QComboBox()
        self._delivery_combo.setStyleSheet(inset_combo())
        self._delivery_combo.addItem("— Select delivery look —", userData="")
        self._delivery_combo.addItem("Reference (BT.709 / BT.1886 / Med / Med)", userData="reference")
        self._delivery_combo.addItem("Custom IPP2 transform…", userData="custom")
        self._delivery_combo.addItem("IPP2 transform + show LUT (.cube)…", userData="lut")
        v.addWidget(self._delivery_combo)

        # Transform dropdowns (shown for custom / lut) — default to the reference look.
        self._delivery_xform = QWidget()
        self._delivery_xform.setStyleSheet("background:transparent;")
        _xl = QVBoxLayout(self._delivery_xform)
        _xl.setContentsMargins(0, 2, 0, 0)
        _xl.setSpacing(6)

        def _mk_xform(label, options, default_idx):
            row = QHBoxLayout(); row.setContentsMargins(0, 0, 0, 0)
            lab = _label(label, size=11, color=TEXT_MUTED)
            lab.setFixedWidth(96)
            cb = QComboBox()
            cb.setStyleSheet(inset_combo())
            for _t, _d in options:
                cb.addItem(_t, userData=_d)
            cb.setCurrentIndex(default_idx)
            row.addWidget(lab); row.addWidget(cb, 1)
            w = QWidget(); w.setStyleSheet("background:transparent;"); w.setLayout(row)
            _xl.addWidget(w)
            return cb

        self._dx_colorspace = _mk_xform("Color space", _IPP2_COLOR_SPACES, 0)  # BT.709
        self._dx_gamma      = _mk_xform("Gamma",       _IPP2_GAMMAS, 0)        # BT.1886
        self._dx_tonemap    = _mk_xform("Tone map",    _IPP2_TONEMAPS, 1)      # Medium
        self._dx_rolloff    = _mk_xform("Roll-off",    _IPP2_ROLLOFFS, 3)      # Medium
        self._delivery_xform.setVisible(False)
        v.addWidget(self._delivery_xform)

        # LUT browse row (shown for lut)
        deliv_row = QHBoxLayout(); deliv_row.setContentsMargins(0, 0, 0, 0)
        self._delivery_lut_lbl = _label("No LUT selected", size=11, color=TEXT_MUTED)
        self._delivery_browse = _button("Browse LUT…")
        deliv_row.addWidget(self._delivery_lut_lbl, 1)
        deliv_row.addWidget(self._delivery_browse)
        self._delivery_lut_row = QWidget()
        self._delivery_lut_row.setStyleSheet("background:transparent;")
        self._delivery_lut_row.setLayout(deliv_row)
        self._delivery_lut_row.setVisible(False)
        v.addWidget(self._delivery_lut_row)

        self._delivery_combo.currentIndexChanged.connect(self._on_delivery_changed)
        self._delivery_browse.clicked.connect(self._browse_delivery_lut)

        # ── White balance ──────────────────────────────────────────────────
        v = section("WHITE BALANCE", "How the array's white balance is solved.")
        self._wb_combo = QComboBox()
        self._wb_combo.setStyleSheet(inset_combo())
        self._wb_combo.addItem("Match at scene temp — per-camera Kelvin (recommended)", userData="scene_match")
        self._wb_combo.addItem("Match — shared Kelvin (v3 legacy)", userData="match")
        self._wb_combo.addItem("Match to neutral — per-camera (shifts off scene temp)", userData="neutral")
        self._wb_combo.addItem("Exposure only — no white balance", userData="exposure_only")
        v.addWidget(self._wb_combo)
        self._wb_desc = _label("", size=11, color=TEXT_MUTED)
        self._wb_desc.setWordWrap(True)
        self._wb_desc.setStyleSheet(
            f"color:{TEXT_MUTED};background:transparent;border:none;font-size:11px;")
        v.addWidget(self._wb_desc)
        self._wb_combo.currentIndexChanged.connect(self._on_wb_changed)
        self._on_wb_changed(0)

        hint = _label(
            "Configure camera IPs in the Camera Network tab before pushing.",
            size=11, color=TEXT_MUTED)
        hint.setWordWrap(True)
        bl.addWidget(hint)
        bl.addStretch()

        rr = QHBoxLayout()
        rr.addStretch()
        self._run_btn = _button("Analyze Cameras →", primary=True)
        self._run_btn.setFixedHeight(42)
        self._run_btn.setMinimumWidth(200)
        self._run_btn.clicked.connect(self._on_run)
        rr.addWidget(self._run_btn)
        bl.addLayout(rr)

        scroll.setWidget(body)
        root.addWidget(scroll)

    def _make_card(self, title: str, desc: str) -> QWidget:
        w = QWidget()
        # Lighter fill + stronger border + accent left-edge so each section
        # header reads as a distinct block instead of bleeding into the page.
        w.setStyleSheet(
            f"QWidget{{background:{CARD_BG};border:1px solid {BORDER_STRONG};"
            f"border-left:3px solid {ACCENT};"
            f"border-top-left-radius:8px;border-top-right-radius:8px;"
            f"border-bottom-left-radius:8px;border-bottom-right-radius:8px;"
            f"padding:10px 14px;}}")
        vl = QVBoxLayout(w)
        vl.setSpacing(3)
        vl.setContentsMargins(0, 0, 0, 0)
        tl = _label(title, size=10, bold=True, color=TEXT_PRIMARY)
        tl.setStyleSheet(
            f"color:{TEXT_PRIMARY};background:transparent;border:none;font-weight:800;"
            f"letter-spacing:0.1em;font-size:10px;")
        dl = _label(desc, size=11, color=TEXT_MUTED)
        dl.setWordWrap(True)
        dl.setStyleSheet(f"color:{TEXT_MUTED};background:transparent;border:none;font-size:11px;")
        vl.addWidget(tl)
        vl.addWidget(dl)
        return w

    def _on_strategy_changed(self, _index: int = 0):
        key = self._strategy.currentData()
        descs = {
            "median": "Median of all camera IRE values — robust against outliers. Recommended for most multi-camera arrays.",
            "optimal_ire": "Target a specific gray card IRE level. Use when you have a reference gray value from the gaffer.",
        }
        self._strat_desc.setText(descs.get(key, ""))

    def _on_delivery_changed(self, _index: int = 0):
        choice = self._delivery_combo.currentData()
        self._delivery_xform.setVisible(choice in ("custom", "lut"))
        self._delivery_lut_row.setVisible(choice == "lut")

    def _on_wb_changed(self, _index: int = 0):
        descs = {
            "scene_match": "Each camera gets its own Kelvin and its own Tint, but "
                           "the array stays anchored on the as-shot scene Kelvin.",
            "match": "All cameras forced to the same Kelvin; only Tint is solved "
                     "per camera.",
            "neutral": "Per-camera Kelvin and Tint, but pushes the correction to "
                       "perfectly neutral.",
            "exposure_only": "Exposure matching only — white balance is left "
                             "untouched (no Kelvin or Tint solved or pushed).",
        }
        self._wb_desc.setText(descs.get(self._wb_combo.currentData(), ""))

    def _browse_delivery_lut(self):
        p, _ = QFileDialog.getOpenFileName(
            self, "Select Project Show LUT", "", "3D LUT (*.cube *.CUBE)")
        if p:
            self._delivery_lut_path = p
            self._delivery_lut_lbl.setText(Path(p).name)

    def _browse_folder(self):
        p = QFileDialog.getExistingDirectory(self, "Select Input Folder")
        if p:
            self._folder.setText(p)
            if not self._out.text():
                self._out.setText(str(Path(p).parent / f"{Path(p).name}_r3dmatch_out"))

    def _browse_out(self):
        p = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if p: self._out.setText(p)

    def _on_run(self):
        ip = self._folder.text().strip()
        od = self._out.text().strip()
        st = self._strategy.currentData()
        if not ip or not Path(ip).exists():
            QMessageBox.warning(self, "No Input", "Please select a valid input folder.")
            return
        if not od:
            QMessageBox.warning(self, "No Output", "Please set an output folder.")
            return
        # Required, explicit per-run delivery-look choice — no assumed default.
        dchoice = self._delivery_combo.currentData()
        if dchoice == "":
            QMessageBox.warning(self, "Select Delivery Look",
                                "Choose a delivery look (Reference, a custom IPP2 "
                                "transform, or a project LUT) before analyzing.")
            return
        if dchoice == "reference":
            self.state.delivery_pipeline = None
            self.state.delivery_label = "Reference (BT.709 / BT.1886)"
        else:
            lut = None
            if dchoice == "lut":
                if not self._delivery_lut_path:
                    QMessageBox.warning(self, "No LUT",
                                        "Browse to a .cube LUT, or pick another option.")
                    return
                lut = self._delivery_lut_path
            name = (f"{self._dx_colorspace.currentText()} / "
                    f"{self._dx_gamma.currentText()} / "
                    f"TM:{self._dx_tonemap.currentText()} / "
                    f"RO:{self._dx_rolloff.currentText()}")
            if lut:
                name += f" + LUT:{Path(lut).stem}"
            self.state.delivery_pipeline = ColorPipeline(
                color_space=self._dx_colorspace.currentData(),
                gamma_curve=self._dx_gamma.currentData(),
                output_tone_map=self._dx_tonemap.currentData(),
                roll_off=self._dx_rolloff.currentData(),
                creative_lut_path=lut,
                name=name)
            self.state.delivery_label = name
        self.state.delivery_chosen   = True
        self.state.wb_mode = self._wb_combo.currentData() or "match"
        self.state.input_path = ip
        self.state.out_dir    = od
        self.state.strategy   = st
        self.run_requested.emit(ip, od, st)


# ── Screen 1: Progress ────────────────────────────────────────────────────────

class ProgressScreen(QWidget):
    def __init__(self, state: AppState):
        super().__init__()
        self.state        = state
        self._phase_rows: Dict[str, tuple] = {}
        self._camera_rows: Dict[str, int]  = {}
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(0)
        root.setContentsMargins(0, 0, 0, 0)

        # ── Header band ───────────────────────────────────────────────────────
        hdr = QWidget()
        hdr.setFixedHeight(52)
        hdr.setStyleSheet(
            f"background:{DARK_BG};border-bottom:1px solid {BORDER_COLOR};")
        hl = QHBoxLayout(hdr)
        hl.setContentsMargins(24, 0, 24, 0)
        hl.addWidget(_label("Analyzing", size=15, bold=True))
        hl.addSpacing(12)
        self._subtitle = _label("", size=12, color=TEXT_MUTED)
        hl.addWidget(self._subtitle)
        hl.addStretch()
        root.addWidget(hdr)

        # ── Body ─────────────────────────────────────────────────────────────
        body = QVBoxLayout()
        body.setSpacing(10)
        body.setContentsMargins(24, 16, 24, 16)

        # Pipeline step row — 6 cells with dividers
        phases = [
            ("scan",    "●", "Scan",    "Discover clips"),
            ("render",  "●", "Render",  "REDLine frames"),
            ("detect",  "●", "Detect",  "Hough / LoG"),
            ("measure", "●", "Measure", "IRE + WB zones"),
            ("solve",   "●", "Solve",   "Exposure + WB"),
            ("verify",  "●", "Verify",  "Closed-loop match"),
            ("report",  "●", "Report",  "Write output"),
        ]
        ph_outer = QFrame()
        ph_outer.setStyleSheet(
            f"QFrame{{background:{PANEL_BG};border:1px solid {BORDER_COLOR};"
            f"border-radius:8px;}}")
        ph_row = QHBoxLayout(ph_outer)
        ph_row.setContentsMargins(0, 0, 0, 0)
        ph_row.setSpacing(0)

        for i, (key, ico, name, desc) in enumerate(phases):
            cell = QWidget()
            cell.setStyleSheet("background:transparent;")
            cl = QVBoxLayout(cell)
            cl.setContentsMargins(16, 12, 16, 12)
            cl.setSpacing(3)
            icon_lbl = _label(ico, size=14, color=TEXT_MUTED, mono=True)
            name_lbl = _label(name, size=12, bold=True, color=TEXT_MUTED)
            desc_lbl = _label(desc, size=10, color="#6e6557")
            cl.addWidget(icon_lbl)
            cl.addWidget(name_lbl)
            cl.addWidget(desc_lbl)
            ph_row.addWidget(cell, stretch=1)
            self._phase_rows[key] = (icon_lbl, name_lbl, desc_lbl)
            if i < len(phases) - 1:
                div = QFrame()
                div.setFrameShape(QFrame.Shape.VLine)
                div.setStyleSheet(f"color:{BORDER_COLOR};")
                ph_row.addWidget(div)

        body.addWidget(ph_outer)

        # Camera table
        cam_lbl = _label("CAMERAS", size=10, bold=True, color=TEXT_MUTED)
        cam_lbl.setStyleSheet(
            f"color:{TEXT_MUTED};font-size:10px;font-weight:800;"
            f"letter-spacing:0.08em;background:transparent;")
        body.addWidget(cam_lbl)
        self._cam_table = QTableWidget(0, 5)
        self._cam_table.setHorizontalHeaderLabels(["Camera", "Stage", "IRE", "FPS", "Resolution"])
        hdr2 = self._cam_table.horizontalHeader()
        hdr2.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        hdr2.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        hdr2.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        hdr2.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        hdr2.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        self._cam_table.setStyleSheet(_table_style())
        self._cam_table.setMaximumHeight(240)
        self._cam_table.verticalHeader().setVisible(False)
        body.addWidget(self._cam_table)

        # Log
        log_lbl = _label("LOG", size=10, bold=True, color=TEXT_MUTED)
        log_lbl.setStyleSheet(
            f"color:{TEXT_MUTED};font-size:10px;font-weight:800;"
            f"letter-spacing:0.08em;background:transparent;")
        body.addWidget(log_lbl)
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setStyleSheet(
            f"background:#0b0907;color:#34d399;font-family:{MONO_FONT};"
            f"font-size:11px;border:1px solid {BORDER_COLOR};border-radius:6px;"
            f"padding:8px;")
        self._log.setMaximumHeight(150)
        body.addWidget(self._log)
        body.addStretch()

        root.addLayout(body)

    def reset(self, input_path: str):
        self._subtitle.setText(f"Input: {input_path}")
        self._log.clear()
        self._cam_table.setRowCount(0)
        self._camera_rows.clear()
        for icon, _, detail in self._phase_rows.values():
            icon.setText("○")
            icon.setStyleSheet(f"color:{TEXT_MUTED};background:transparent;")
            detail.setText("")

    def on_progress(self, data: dict):
        phase   = data.get("phase", "")
        clip_id = data.get("clip_id", "")
        detail  = data.get("detail", "")

        # Map workflow phase names to display phases
        phase_map = {
            "scan": "scan", "scan_complete": "scan",
            "clip_render": "render", "clip_render_reused": "render",
            "clip_detect": "detect", "clip_detection_failed": "detect",
            "clip_measure": "measure", "clip_complete": "measure",
            "solve": "solve", "solve_exposure": "solve",
            "solve_wb": "solve", "solve_wb_complete": "solve",
            "corrected_renders": "verify", "corrected_render_done": "verify",
            "wb_closed_loop": "verify", "assessing": "verify",
            "profile_saved": "report",
            "writing": "report", "complete": "report",
        }
        display_phase = phase_map.get(phase, "")
        if display_phase and display_phase in self._phase_rows:
            icon_lbl, name_lbl, det_lbl = self._phase_rows[display_phase]
            if "error" in phase or "failed" in phase:
                icon_lbl.setText("✗")
                icon_lbl.setStyleSheet(f"color:{DANGER};background:transparent;")
                name_lbl.setStyleSheet(f"color:{DANGER};background:transparent;font-weight:700;")
            elif phase in ("complete", "scan_complete", "solve_wb_complete"):
                icon_lbl.setText("✓")
                icon_lbl.setStyleSheet(f"color:{SUCCESS};background:transparent;")
                name_lbl.setStyleSheet(f"color:{SUCCESS};background:transparent;font-weight:700;")
            else:
                icon_lbl.setText("●")
                icon_lbl.setStyleSheet(f"color:{ACCENT};background:transparent;")
                name_lbl.setStyleSheet(f"color:{TEXT_PRIMARY};background:transparent;font-weight:700;")
            det_lbl.setText(detail[:40] if detail else "")

        if clip_id:
            camera_label = "_".join(clip_id.split("_")[:2]) if clip_id else ""
            if clip_id not in self._camera_rows:
                row = self._cam_table.rowCount()
                self._cam_table.insertRow(row)
                self._camera_rows[clip_id] = row
                self._cam_table.setItem(row, 0, QTableWidgetItem(camera_label))
                self._cam_table.setItem(row, 1, QTableWidgetItem(""))
                self._cam_table.setItem(row, 2, QTableWidgetItem(""))
                self._cam_table.setItem(row, 3, QTableWidgetItem(""))
                self._cam_table.setItem(row, 4, QTableWidgetItem(""))
            row = self._camera_rows[clip_id]
            # Live metadata — fps/resolution arrive with the render event
            if data.get("clip_fps"):
                self._cam_table.setItem(row, 3, QTableWidgetItem(str(data["clip_fps"])))
            if data.get("clip_res"):
                self._cam_table.setItem(row, 4, QTableWidgetItem(str(data["clip_res"])))
            stage_item = QTableWidgetItem(phase)
            # Stage pill color
            if "complete" in phase or "done" in phase:
                stage_item.setForeground(QColor(SUCCESS))
            elif "error" in phase or "failed" in phase:
                stage_item.setForeground(QColor(DANGER))
            else:
                stage_item.setForeground(QColor(ACCENT))
            self._cam_table.setItem(row, 1, stage_item)
            if "complete" in phase and "IRE" in detail:
                try:
                    ire = detail.split("IRE")[0].strip().split()[-1]
                    self._cam_table.setItem(row, 2, QTableWidgetItem(f"{ire} IRE"))
                except Exception:
                    pass

        if detail:
            self._log.append(
                f'<span style="color:{TEXT_MUTED};">[{phase}]</span> '
                f'<span style="color:#34d399;">{detail}</span>')

    def on_error(self, msg: str):
        self._log.append(f"\n[ERROR]\n{msg}")

    def populate_metadata(self, run_result):
        """Fill FPS and resolution columns from run_result metadata."""
        for cr in run_result.cameras:
            clip_id = cr.clip_id
            if clip_id not in self._camera_rows:
                continue
            row = self._camera_rows[clip_id]
            if cr.metadata:
                fps = f"{cr.metadata.fps:.2f}".rstrip('0').rstrip('.') if cr.metadata.fps else "—"
                w, h = cr.metadata.frame_width, cr.metadata.frame_height
                res = f"{w}×{h}" if w and h else "—"
                self._cam_table.setItem(row, 3, QTableWidgetItem(fps))
                self._cam_table.setItem(row, 4, QTableWidgetItem(res))


# ── Screen 2: Sphere QC ────────────────────────────────────────────────────────

class SphereEditor(QGraphicsView):
    """
    Shows a camera render thumbnail with an editable sphere circle overlay.
    Operator drags to move center; scroll wheel resizes radius.
    Emits roi_changed(cx, cy, r) in full-resolution coordinates.
    """
    roi_changed = Signal(float, float, float)  # cx, cy, r (full-res coords)

    def __init__(self):
        super().__init__()
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setStyleSheet(f"background:{DARK_BG};border:1px solid {BORDER_COLOR};border-radius:6px;")
        self.setMinimumHeight(320)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)

        self._pixmap_item: Optional[QGraphicsPixmapItem] = None
        self._ellipse: Optional[QGraphicsEllipseItem] = None
        self._overlay_items: list = []
        self._scale   = 1.0     # display_pixels / full_res_pixels
        self._dragging = False
        self._drag_start: Optional[QPointF] = None
        self._cx = 0.0   # full-res
        self._cy = 0.0
        self._r  = 100.0

    def load_image(self, path: str, roi: Optional[SphereROI] = None):
        from PySide6.QtGui import QPixmap
        self._scene.clear()
        # scene.clear() destroys all items — reset our references
        self._ellipse = None
        self._overlay_items = []
        self._pixmap_item = None

        pix = QPixmap(path)
        if pix.isNull():
            return
        self._pixmap_item = self._scene.addPixmap(pix)
        self._scene.setSceneRect(self._pixmap_item.boundingRect())
        self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

        self._scale = 1.0

        # Operator must ALWAYS get a usable, on-image circle to drag — even if the
        # detector handed back a corrupt/off-image ROI (e.g. a bad prior blend).
        w, h = pix.width(), pix.height()
        default_r = min(w, h) * 0.10
        if roi is not None:
            cx, cy, r = float(roi.cx), float(roi.cy), float(roi.r)
            on_image = (0.0 <= cx <= w) and (0.0 <= cy <= h)
            sane_r = 4.0 <= r <= min(w, h) * 0.9
            if not (on_image and sane_r):
                cx, cy, r = w / 2.0, h / 2.0, default_r
            self._cx, self._cy, self._r = cx, cy, r
        else:
            self._cx, self._cy, self._r = w / 2.0, h / 2.0, default_r

        self._draw_ellipse()

    def _draw_ellipse(self):
        # Remove previous overlay items
        if self._ellipse:
            self._scene.removeItem(self._ellipse)
        for item in getattr(self, '_overlay_items', []):
            self._scene.removeItem(item)
        self._overlay_items = []

        cx, cy, r = self._cx, self._cy, self._r

        # Outer circle — orange, 3px — marks detected/placed sphere boundary
        self._ellipse = self._scene.addEllipse(
            QRectF(cx - r, cy - r, r * 2, r * 2),
            QPen(QColor(255, 80, 0), 3),
        )
        self._ellipse.setFlag(QGraphicsEllipseItem.GraphicsItemFlag.ItemIsMovable, False)

        # Inner sampling zone — cyan dashed — 0.6r, exact measurement area
        inner_r = r * 0.6
        inner = self._scene.addEllipse(
            QRectF(cx - inner_r, cy - inner_r, inner_r * 2, inner_r * 2),
            QPen(QColor(0, 229, 255), 1, Qt.PenStyle.DashLine),
        )
        self._overlay_items.append(inner)

        # Centre crosshair — cyan, 10px arms
        arm = 10
        h_line = self._scene.addLine(cx - arm, cy, cx + arm, cy,
                                     QPen(QColor(0, 229, 255), 1))
        v_line = self._scene.addLine(cx, cy - arm, cx, cy + arm,
                                     QPen(QColor(0, 229, 255), 1))
        self._overlay_items.extend([h_line, v_line])

    def _emit(self):
        self.roi_changed.emit(self._cx, self._cy, self._r)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging  = True
            self._drag_start = self.mapToScene(event.position().toPoint())
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._dragging and self._drag_start is not None:
            pos = self.mapToScene(event.position().toPoint())
            self._cx += pos.x() - self._drag_start.x()
            self._cy += pos.y() - self._drag_start.y()
            self._drag_start = pos
            self._draw_ellipse()
            self._emit()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self._dragging   = False
        self._drag_start = None
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event):
        # Scroll wheel pans the view — do NOT resize the sphere
        super().wheelEvent(event)

    def set_radius(self, r: float):
        """Called by the external radius slider."""
        self._r = max(10.0, float(r))
        self._draw_ellipse()
        self._emit()

    def get_radius(self) -> float:
        return self._r

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._scene.sceneRect().isValid():
            self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def current_roi(self) -> SphereROI:
        return SphereROI(cx=self._cx, cy=self._cy, r=self._r)


# ── Camera thumbnail grid cell ──────────────────────────────────────────────
THUMB_W = 230
THUMB_H = 178


def _thumb_pixmap(path: str, roi: Optional[SphereROI]):
    """Downscaled frame with the sphere overlay painted, so a mis-placed solve
    is obvious at a glance. Returns a QPixmap (or None if the frame won't load)."""
    from PySide6.QtGui import QPixmap
    pix = QPixmap(path)
    if pix.isNull():
        return None
    ow = pix.width()
    scaled = pix.scaled(
        THUMB_W, THUMB_H,
        Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    )
    sf = (scaled.width() / ow) if ow else 1.0
    if roi is not None:
        p = QPainter(scaled)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        cx, cy, r = roi.cx * sf, roi.cy * sf, roi.r * sf
        p.setPen(QPen(QColor(255, 80, 0), 2))
        p.drawEllipse(QPointF(cx, cy), r, r)
        p.setPen(QPen(QColor(0, 229, 255), 1))
        p.drawLine(int(cx - 5), int(cy), int(cx + 5), int(cy))
        p.drawLine(int(cx), int(cy - 5), int(cx), int(cy + 5))
        p.end()
    return scaled


class CameraThumb(QFrame):
    """Grid cell — frame thumbnail with sphere overlay, status-colored edge,
    camera label + IRE underneath. Clicking selects the camera."""

    def __init__(self, cr, cat: str, color: str, on_click):
        super().__init__()
        self.clip_id = cr.clip_id
        self._color = color
        self._on_click = on_click
        self._selected = False
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedWidth(THUMB_W + 14)

        v = QVBoxLayout(self)
        v.setContentsMargins(6, 6, 6, 7)
        v.setSpacing(4)

        self._img = QLabel()
        self._img.setFixedSize(THUMB_W, THUMB_H)
        self._img.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._img.setStyleSheet("background:#0b0907;border-radius:3px;color:#5a4f42;")
        self._img.setText("no frame")
        v.addWidget(self._img)

        name = QLabel(cr.camera_label)
        name.setAlignment(Qt.AlignmentFlag.AlignCenter)
        name.setStyleSheet(
            f"color:{TEXT_PRIMARY};background:transparent;font-weight:800;"
            f"font-size:11px;font-family:{MONO_FONT};")
        v.addWidget(name)

        ire_str = f"{cr.measurement.hero_ire:.1f} IRE" if cr.measurement else "— IRE"
        meta = QLabel(f"{ire_str}  ·  {cat.upper()}")
        meta.setAlignment(Qt.AlignmentFlag.AlignCenter)
        meta.setStyleSheet(
            f"color:{color};background:transparent;font-size:9px;"
            f"font-weight:700;letter-spacing:0.04em;")
        v.addWidget(meta)

        self._apply_style()

    def set_thumb(self, pix):
        if pix is not None:
            self._img.setText("")
            self._img.setPixmap(pix)

    def set_selected(self, sel: bool):
        if sel != self._selected:
            self._selected = sel
            self._apply_style()

    def _apply_style(self):
        if self._selected:
            self.setStyleSheet(
                f"CameraThumb{{background:{CARD_BG};border:2px solid {ACCENT};"
                f"border-radius:8px;}}")
        else:
            self.setStyleSheet(
                f"CameraThumb{{background:{PANEL_BG};border:1px solid {self._color}55;"
                f"border-top:3px solid {self._color};border-radius:8px;}}"
                f"CameraThumb:hover{{background:{CARD_BG};border:1px solid {self._color};"
                f"border-top:3px solid {self._color};}}")

    def mousePressEvent(self, event):
        self._on_click()
        super().mousePressEvent(event)


class SphereQCScreen(QWidget):
    """
    Post-run QC: operator verifies sphere solves or corrects them.

    Left panel: camera grid — green=AUTO_OK, yellow=NEEDS_ASSIST, red=FAILED
    Right panel: selected camera detail with SphereEditor + gate status + metrics
    Bottom: "Create Assessment" gate (enabled when all cameras resolved)
    """
    assessment_ready = Signal(object)  # emits RunResult when assessment created

    def __init__(self, state: AppState):
        super().__init__()
        self.state          = state
        self._selected_clip: Optional[str] = None
        self._remeasure_worker: Optional[RemeasureWorker] = None
        self._camera_btns: Dict[str, "CameraThumb"] = {}
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(0)
        root.setContentsMargins(0, 0, 0, 0)

        # ── Header ────────────────────────────────────────────────────────────
        hdr = QWidget()
        hdr.setFixedHeight(52)
        hdr.setStyleSheet(f"background:{DARK_BG};border-bottom:1px solid {BORDER_COLOR};")
        hl = QHBoxLayout(hdr)
        hl.setContentsMargins(20, 0, 20, 0)
        hl.addWidget(_label("Assist", size=16, bold=True))
        hl.addStretch()
        self._status_lbl = _label("Run analysis first", size=12, color=TEXT_MUTED)
        hl.addWidget(self._status_lbl)
        self._assess_btn = _button("Create Assessment →", primary=True)
        self._assess_btn.setEnabled(False)
        self._assess_btn.clicked.connect(self._create_assessment)
        hl.addWidget(self._assess_btn)
        root.addWidget(hdr)

        # ── Summary strip ─────────────────────────────────────────────────────
        self._summary = QWidget()
        self._summary.setFixedHeight(40)
        self._summary.setStyleSheet(
            f"background:{PANEL_BG};border-bottom:1px solid {BORDER_COLOR};")
        self._sum_layout = QHBoxLayout(self._summary)
        self._sum_layout.setContentsMargins(20, 0, 20, 0)
        self._sum_layout.setSpacing(10)
        root.addWidget(self._summary)

        # ── Main split: camera grid + detail panel ─────────────────────────────
        body = QHBoxLayout()
        body.setSpacing(0)
        body.setContentsMargins(0, 0, 0, 0)

        # Camera thumbnail grid (GRID MODE) — full-width gallery of all cameras.
        # Clicking a thumbnail opens the large solve view (SOLVE MODE).
        left = QWidget()
        left.setStyleSheet(f"background:{DARK_BG};")
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(20, 14, 20, 14)
        left_layout.setSpacing(10)
        cam_hdr = _label("CAMERAS", size=10, bold=True, color=TEXT_MUTED)
        cam_hdr.setStyleSheet(
            f"color:{TEXT_MUTED};font-size:10px;font-weight:800;"
            f"letter-spacing:0.08em;background:transparent;")
        cam_hint = _label("Click any camera to open the large solver", size=11, color=TEXT_MUTED)
        cam_hint.setStyleSheet(f"color:{TEXT_DIM};font-size:11px;background:transparent;")
        hdr_row = QHBoxLayout()
        hdr_row.addWidget(cam_hdr)
        hdr_row.addSpacing(10)
        hdr_row.addWidget(cam_hint)
        hdr_row.addStretch()
        left_layout.addLayout(hdr_row)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(
            f"QScrollArea{{border:none;background:transparent;}}"
            f"QScrollBar:vertical{{background:{PANEL_BG};width:8px;border-radius:4px;}}"
            f"QScrollBar::handle:vertical{{background:{BORDER_STRONG};border-radius:4px;}}")
        self._grid_widget = QWidget()
        self._grid_widget.setStyleSheet("background:transparent;")
        self._grid_layout = QGridLayout(self._grid_widget)
        self._grid_layout.setHorizontalSpacing(14)
        self._grid_layout.setVerticalSpacing(14)
        self._grid_layout.setContentsMargins(0, 0, 0, 0)
        self._grid_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        scroll.setWidget(self._grid_widget)
        left_layout.addWidget(scroll)
        self._left = left
        self._thumbs: list = []
        self._grid_cols = 6
        body.addWidget(left, stretch=1)

        # Detail / solve panel (SOLVE MODE) — full width, hidden until a
        # thumbnail is clicked.
        right = QWidget()
        right.setStyleSheet(f"background:{DARK_BG};")
        self._right = right
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(20, 14, 20, 16)
        right_layout.setSpacing(12)

        title_row = QHBoxLayout()
        self._back_btn = _button("← Cameras")
        self._back_btn.clicked.connect(self._show_grid)
        title_row.addWidget(self._back_btn)
        title_row.addSpacing(8)
        self._detail_title = _label("Select a camera", size=15, bold=True)
        title_row.addWidget(self._detail_title)
        title_row.addStretch()
        # Prev / Next — cycle through cameras without returning to the grid.
        self._prev_btn = _button("‹ Prev")
        self._prev_btn.clicked.connect(lambda: self._step_camera(-1))
        self._next_btn = _button("Next ›")
        self._next_btn.clicked.connect(lambda: self._step_camera(1))
        self._cam_pos_lbl = _label("", size=12, color=TEXT_MUTED, mono=True)
        title_row.addWidget(self._cam_pos_lbl)
        title_row.addSpacing(8)
        title_row.addWidget(self._prev_btn)
        title_row.addWidget(self._next_btn)
        right_layout.addLayout(title_row)

        # Sphere editor
        self._editor = SphereEditor()
        self._editor.roi_changed.connect(self._on_roi_changed)
        right_layout.addWidget(self._editor, stretch=3)

        # Instructions
        instr = _label(
            "Drag to move sphere center  ·  Use slider to resize  ·  "
            "Click Accept or Adjust below",
            size=11, color=TEXT_MUTED)
        instr.setWordWrap(True)
        right_layout.addWidget(instr)

        # Radius slider
        slider_row = QHBoxLayout()
        slider_row.addWidget(_label("Radius:", size=11, color=TEXT_MUTED))
        self._radius_slider = QSlider(Qt.Orientation.Horizontal)
        self._radius_slider.setMinimum(10)
        self._radius_slider.setMaximum(800)
        self._radius_slider.setValue(100)
        self._radius_slider.setStyleSheet(
            f"QSlider::groove:horizontal{{background:{PANEL_BG};"
            f"border:1px solid {BORDER_COLOR};height:6px;border-radius:3px;}}"
            f"QSlider::handle:horizontal{{background:{ACCENT};border:none;"
            f"width:16px;height:16px;margin:-5px 0;border-radius:8px;}}"
            f"QSlider::sub-page:horizontal{{background:{ACCENT};border-radius:3px;}}")
        self._radius_slider.valueChanged.connect(self._on_slider_changed)
        self._radius_lbl = _label("r=100", size=11, color=TEXT_MUTED, mono=True)
        self._radius_lbl.setFixedWidth(60)
        slider_row.addWidget(self._radius_slider, stretch=1)
        slider_row.addWidget(self._radius_lbl)
        right_layout.addLayout(slider_row)

        # Metrics row
        metrics_row = QHBoxLayout()
        self._ire_lbl    = _label("IRE: —",    size=12, mono=True)
        self._gate_lbl   = _label("Gates: —",  size=12, mono=True)
        self._source_lbl = _label("Source: —", size=12, color=TEXT_MUTED)
        for w in (self._ire_lbl, self._gate_lbl, self._source_lbl):
            metrics_row.addWidget(w)
        metrics_row.addStretch()
        right_layout.addLayout(metrics_row)

        # Action buttons
        action_row = QHBoxLayout()
        self._accept_btn = _button("✓ Accept", primary=True)
        self._accept_btn.setEnabled(False)
        self._accept_btn.clicked.connect(self._accept_current)
        self._adjust_btn = _button("↺ Apply Adjustment", warning=True)
        self._adjust_btn.setEnabled(False)
        self._adjust_btn.clicked.connect(self._apply_adjustment)
        self._reject_btn = _button("✗ Exclude", danger=True)
        self._reject_btn.setEnabled(False)
        self._reject_btn.clicked.connect(self._reject_current)
        action_row.addWidget(self._accept_btn)
        action_row.addWidget(self._adjust_btn)
        action_row.addWidget(self._reject_btn)
        action_row.addStretch()
        right_layout.addLayout(action_row)

        # Re-measure log
        self._qc_log = QTextEdit()
        self._qc_log.setReadOnly(True)
        self._qc_log.setMaximumHeight(80)
        self._qc_log.setStyleSheet(
            f"background:#0b0907;color:#34d399;font-family:{MONO_FONT};"
            f"font-size:11px;border:1px solid {BORDER_COLOR};border-radius:6px;"
            f"padding:6px;")
        right_layout.addWidget(self._qc_log)

        body.addWidget(right, stretch=1)
        root.addLayout(body, stretch=1)
        self._right.hide()  # start in grid (overview) mode

    # ── Grid ⇄ solve mode ───────────────────────────────────────────────────
    def _enter_solve_mode(self):
        self._left.hide()
        self._right.show()

    def _show_grid(self):
        self._right.hide()
        self._left.show()
        self._reflow_grid()

    def _step_camera(self, delta: int):
        """Cycle to the prev/next camera while staying in the solve view."""
        rr = self.state.run_result
        if rr is None or not rr.cameras:
            return
        clips = [cr.clip_id for cr in rr.cameras]
        try:
            idx = clips.index(self._selected_clip)
        except ValueError:
            idx = 0
        nxt = (idx + delta) % len(clips)
        self._select_camera(rr.cameras[nxt], enter_solve=True)

    def _reflow_grid(self):
        """Lay the thumbnails out in as many columns as the width allows."""
        if not self._thumbs:
            return
        avail = self._grid_widget.width() or self._left.width() or 900
        cols = max(2, int(avail // (THUMB_W + 26)))
        if cols == getattr(self, "_grid_cols_active", None):
            return
        self._grid_cols_active = cols
        for t in self._thumbs:
            self._grid_layout.removeWidget(t)
        for i, t in enumerate(self._thumbs):
            self._grid_layout.addWidget(t, i // cols, i % cols)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if getattr(self, "_left", None) is not None and self._left.isVisible():
            self._reflow_grid()

    def populate(self, run_result):
        """Called after analysis completes. Build camera grid."""
        rr = run_result
        self.state.run_result = rr
        self.state.qc_rois.clear()
        self.state.qc_accepted.clear()

        # Clear grid
        while self._grid_layout.count():
            item = self._grid_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._camera_btns.clear()
        self._thumbs = []

        # Count categories
        auto_ok = needs_assist = failed = 0
        color_map = {"ok": SUCCESS, "assist": WARNING, "fail": DANGER}

        for cr in rr.cameras:
            det = cr.detection
            if det and det.success and cr.measurement and cr.measurement.measurement_valid:
                cat = "ok"
                auto_ok += 1
                self.state.qc_accepted[cr.clip_id] = True
            elif cr.measurement and cr.measurement.measurement_valid:
                cat = "assist"
                needs_assist += 1
            else:
                cat = "fail"
                failed += 1
            color = color_map[cat]

            thumb = CameraThumb(cr, cat, color,
                                on_click=lambda c=cr: self._select_camera(c))
            # Paint the frame + sphere overlay into the cell.
            render_path = (cr.measurement.render_path if cr.measurement
                           else cr.original_render_path)
            roi = cr.detection.roi if cr.detection else None
            if render_path and Path(render_path).exists():
                pm = _thumb_pixmap(render_path, roi)
                thumb.set_thumb(pm)

            self._thumbs.append(thumb)
            self._camera_btns[cr.clip_id] = thumb

        # Lay out the thumbnails responsively across the full width.
        self._grid_cols_active = None
        self._reflow_grid()

        # Update summary strip — styled chips
        while self._sum_layout.count():
            item = self._sum_layout.takeAt(0)
            if item.widget(): item.widget().deleteLater()
        for lbl_text, val, color, bg in [
            ("Auto OK",      str(auto_ok),      SUCCESS, "#0a2e1a"),
            ("Needs Assist", str(needs_assist), WARNING, "#2a1f00"),
            ("Failed",       str(failed),       DANGER,  "#2a0a0a"),
        ]:
            chip = QWidget()
            chip.setStyleSheet(
                f"QWidget{{background:{bg};border:1px solid {color}44;"
                f"border-radius:5px;padding:2px 8px;}}")
            cl = QHBoxLayout(chip)
            cl.setContentsMargins(6, 2, 6, 2)
            cl.setSpacing(5)
            cl.addWidget(_label(lbl_text, size=10, color=TEXT_MUTED))
            cl.addWidget(_label(val, size=11, bold=True, color=color))
            self._sum_layout.addWidget(chip)
        self._sum_layout.addStretch()

        self._update_status()

        # Prime the editor with the first unresolved (or first) camera, but stay
        # on the grid overview — the operator clicks a thumbnail to open it large.
        prime = next((cr for cr in rr.cameras
                      if not self.state.qc_accepted.get(cr.clip_id)), None)
        if prime is None and rr.cameras:
            prime = rr.cameras[0]
        if prime is not None:
            self._select_camera(prime, enter_solve=False)
        self._show_grid()

    def _select_camera(self, cr, enter_solve: bool = True):
        self._selected_clip = cr.clip_id
        if enter_solve:
            self._enter_solve_mode()

        # Position indicator for the prev/next cycler (e.g. "5 / 12")
        rr = self.state.run_result
        if rr and rr.cameras:
            clips = [c.clip_id for c in rr.cameras]
            if cr.clip_id in clips:
                self._cam_pos_lbl.setText(f"{clips.index(cr.clip_id) + 1} / {len(clips)}")

        # Update thumbnail selection states
        for cid, thumb in self._camera_btns.items():
            thumb.set_selected(cid == cr.clip_id)

        # Load render into editor
        render_path = (
            cr.measurement.render_path if cr.measurement
            else cr.original_render_path
        )
        roi = cr.detection.roi if cr.detection else None
        if render_path and Path(render_path).exists():
            self._editor.load_image(render_path, roi)
        else:
            self._editor.load_image("", roi)

        # Sync slider to the editor's ACTUAL (sanitized) radius, not the raw ROI —
        # a corrupt ROI is reset to a sane default inside load_image.
        r_int = max(10, min(800, int(round(self._editor.get_radius()))))
        self._radius_slider.blockSignals(True)
        self._radius_slider.setValue(r_int)
        self._radius_slider.blockSignals(False)
        self._radius_lbl.setText(f"r={r_int}")

        self._detail_title.setText(f"{cr.camera_label}  ·  {cr.clip_id.split('_')[2] if len(cr.clip_id.split('_')) > 2 else cr.clip_id}")

        # Metrics
        if cr.measurement and cr.measurement.hero_ire is not None:
            self._ire_lbl.setText(f"IRE: {cr.measurement.hero_ire:.2f}")
        elif cr.detection and cr.detection.success:
            self._ire_lbl.setText("IRE: detected, not measured")
        else:
            self._ire_lbl.setText("IRE: —")

        if cr.detection:
            passed = sum(1 for g in cr.detection.gates if g.passed)
            total  = len(cr.detection.gates)
            self._gate_lbl.setText(f"Gates: {passed}/{total}")
            self._source_lbl.setText(f"Source: {cr.detection.source}")
        else:
            self._gate_lbl.setText("Gates: —")
            self._source_lbl.setText("Source: —")

        # Buttons
        is_ok = self.state.qc_accepted.get(cr.clip_id, False)
        self._accept_btn.setEnabled(not is_ok)
        self._adjust_btn.setEnabled(True)
        self._reject_btn.setEnabled(True)

    def _on_slider_changed(self, value: int):
        self._editor.set_radius(float(value))
        self._radius_lbl.setText(f"r={value}")

    def _on_roi_changed(self, cx: float, cy: float, r: float):
        if self._selected_clip:
            self.state.qc_rois[self._selected_clip] = SphereROI(cx=cx, cy=cy, r=r)
        # Keep slider in sync when user drags (radius unchanged on drag, but keep consistent)
        r_int = max(10, min(800, int(round(r))))
        self._radius_slider.blockSignals(True)
        self._radius_slider.setValue(r_int)
        self._radius_slider.blockSignals(False)
        self._radius_lbl.setText(f"r={r_int}")

    def _accept_current(self):
        if not self._selected_clip:
            return
        self.state.qc_accepted[self._selected_clip] = True
        self._accept_btn.setEnabled(False)
        self._update_camera_btn_color(self._selected_clip, "ok")
        self._update_status()
        self._qc_log.append(f"[ACCEPT] {self._selected_clip}")
        self._advance_to_next_unresolved()

    def _advance_to_next_unresolved(self):
        """Move to the next camera that hasn't been accepted or excluded."""
        rr = self.state.run_result
        if rr is None:
            return
        clips = [cr.clip_id for cr in rr.cameras]
        if self._selected_clip not in clips:
            return
        idx = clips.index(self._selected_clip)
        # Search forward from current, wrap around
        n = len(clips)
        for offset in range(1, n):
            next_clip = clips[(idx + offset) % n]
            if not self.state.qc_accepted.get(next_clip, False):
                # Find the CameraResult and select it
                for cr in rr.cameras:
                    if cr.clip_id == next_clip:
                        self._select_camera(cr)
                        return

    def _apply_adjustment(self):
        """Re-measure the current camera with the editor's current ROI."""
        if not self._selected_clip:
            return
        roi = self._editor.current_roi()
        self.state.qc_rois[self._selected_clip] = roi
        self._qc_log.append(f"[ADJUST] {self._selected_clip} cx={roi.cx:.0f} cy={roi.cy:.0f} r={roi.r:.0f}")

        self._adjust_btn.setEnabled(False)
        self._accept_btn.setEnabled(False)

        self._remeasure_worker = RemeasureWorker(
            self.state.run_result, {self._selected_clip: roi}
        )
        self._remeasure_worker.progress.connect(
            lambda ph, det: self._qc_log.append(f"[{ph}] {det}")
        )
        self._remeasure_worker.finished.connect(self._on_remeasure_done)
        self._remeasure_worker.errored.connect(self._on_remeasure_error)
        self._remeasure_worker.start()

    def _reject_current(self):
        if not self._selected_clip:
            return
        self.state.qc_accepted[self._selected_clip] = False
        # Mark camera as excluded
        rr = self.state.run_result
        for cr in rr.cameras:
            if cr.clip_id == self._selected_clip:
                cr.status = "NO_DATA"
                cr.failed_stage = "operator_excluded"
                cr.failure_reason = "Excluded by operator in Sphere QC"
                break
        self._update_camera_btn_color(self._selected_clip, "fail")
        self._qc_log.append(f"[EXCLUDE] {self._selected_clip}")
        self._update_status()

    @Slot(object)
    def _on_remeasure_done(self, run_result):
        self.state.run_result = run_result
        self.state.qc_accepted[self._selected_clip] = True
        self._update_camera_btn_color(self._selected_clip, "ok")
        self._adjust_btn.setEnabled(True)

        # Refresh the current camera's thumbnail (overlay + label) and metrics
        for cr in run_result.cameras:
            if cr.clip_id == self._selected_clip:
                self._refresh_thumb(cr, "ok")
                # Update display BEFORE advancing so operator sees the result
                self._select_camera(cr)
                break

        self._update_status()
        # Use a short timer to let the UI repaint before advancing
        from PySide6.QtCore import QTimer
        QTimer.singleShot(800, self._advance_to_next_unresolved)

    @Slot(str)
    def _on_remeasure_error(self, msg: str):
        self._qc_log.append(f"[ERROR] {msg[:200]}")
        self._adjust_btn.setEnabled(True)
        self._accept_btn.setEnabled(True)

    def _update_camera_btn_color(self, clip_id: str, cat: str):
        """Recolor a thumbnail's status edge (e.g. after exclude / re-measure)."""
        thumb = self._camera_btns.get(clip_id)
        if not thumb:
            return
        color = {
            "ok": SUCCESS, "assist": WARNING, "fail": DANGER
        }.get(cat, TEXT_MUTED)
        thumb._color = color
        thumb._apply_style()

    def _refresh_thumb(self, cr, cat: str):
        """Repaint a thumbnail's overlay + recolor its edge after a re-measure."""
        thumb = self._camera_btns.get(cr.clip_id)
        if not thumb:
            return
        render_path = (cr.measurement.render_path if cr.measurement
                       else cr.original_render_path)
        roi = cr.detection.roi if cr.detection else None
        if render_path and Path(render_path).exists():
            pm = _thumb_pixmap(render_path, roi)
            thumb.set_thumb(pm)
        self._update_camera_btn_color(cr.clip_id, cat)

    def _update_status(self):
        rr = self.state.run_result
        if rr is None:
            self._status_lbl.setText("Run analysis first")
            self._assess_btn.setEnabled(False)
            return

        total    = len(rr.cameras)
        accepted = sum(1 for c in rr.cameras
                       if self.state.qc_accepted.get(c.clip_id, False)
                       or c.status == "NO_DATA")
        pending  = total - accepted

        if pending == 0:
            self._status_lbl.setText(f"All {total} cameras resolved — ready")
            self._assess_btn.setEnabled(True)
        else:
            self._status_lbl.setText(f"{pending} camera{'s' if pending > 1 else ''} pending review")
            self._assess_btn.setEnabled(False)

    def _create_assessment(self):
        """Closed-loop verify (corrected renders), then build the report.

        The main analysis runs with render_corrected=False because QC can
        still change ROIs. This is where the loop closes: final commits →
        corrected TIFFs → measured match % → report.
        """
        rr = self.state.run_result
        if rr is None:
            return
        self._assess_btn.setEnabled(False)
        self._qc_log.append(
            "[VERIFY] Closed-loop: rendering corrected frames with final commits…")
        self._verify_worker = VerifyWorker(
            rr, self.state.out_dir, self.state.redline_path,
            delivery_pipeline=self.state.delivery_pipeline)
        self._verify_worker.progress.connect(self._on_verify_progress)
        self._verify_worker.finished.connect(self._on_verify_done)
        self._verify_worker.errored.connect(self._on_verify_error)
        self._verify_worker.start()

    @Slot(dict)
    def _on_verify_progress(self, data: dict):
        detail = data.get("detail", "")
        phase  = data.get("phase", "")
        if detail and phase in ("corrected_renders", "corrected_render_done",
                                "wb_closed_loop", "assessing", "verify_complete"):
            self._qc_log.append(f"[{phase}] {detail}")

    @Slot(object, object)
    def _on_verify_done(self, rr, report_path):
        self.state.run_result  = rr
        self.state.report_path = report_path
        amp = getattr(rr, "array_match_pct", None)
        self._qc_log.append(
            f"[VERIFY] Done — array match {amp:.0f}%." if amp is not None
            else "[VERIFY] Done — match could not be scored (check renders).")
        if report_path is None:
            self._qc_log.append("[WARN] Report build failed — using inline view")
        self._assess_btn.setEnabled(True)
        self.assessment_ready.emit(rr)

    @Slot(str)
    def _on_verify_error(self, msg: str):
        # Verification failed (e.g. REDLine error) — never block the operator.
        # Fall back to the unverified report so the run is still usable.
        self._qc_log.append(f"[ERROR] Closed-loop verify failed: {msg[:300]}")
        self._qc_log.append("[VERIFY] Falling back to unverified assessment.")
        rr = self.state.run_result
        report_path = None
        try:
            report_path = build_report(rr, self.state.out_dir)
            if report_path and not Path(report_path).exists():
                report_path = None
        except Exception:
            report_path = None
        self.state.report_path = report_path
        self._assess_btn.setEnabled(True)
        self.assessment_ready.emit(rr)


# ── Screen 3: Results ─────────────────────────────────────────────────────────

class ResultsScreen(QWidget):
    push_requested   = Signal()
    rerun_requested  = Signal()
    commit_requested = Signal()

    def __init__(self, state: AppState):
        super().__init__()
        self.state = state
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(0)
        root.setContentsMargins(0, 0, 0, 0)

        tb = QWidget()
        tb.setFixedHeight(72)
        tb.setStyleSheet(f"background:{DARK_BG};border-bottom:1px solid {BORDER_COLOR};")
        tbl = QHBoxLayout(tb)
        tbl.setContentsMargins(20, 0, 20, 0)
        tbl.setSpacing(14)
        # Hero: big match % (mockup Match screen)
        self._badge = QLabel("—")
        self._badge.setStyleSheet(
            f"color:{TEXT_MUTED};font-size:30px;font-weight:800;"
            f"font-family:{MONO_FONT};background:transparent;")
        tbl.addWidget(self._badge)
        hero_sub = QVBoxLayout()
        hero_sub.setSpacing(1)
        self._hero_line1 = _label("Match", size=13, bold=True)
        self._hero_line2 = _label("", size=11, color=TEXT_MUTED)
        hero_sub.addWidget(self._hero_line1)
        hero_sub.addWidget(self._hero_line2)
        tbl.addLayout(hero_sub)
        tbl.addStretch()
        for text, slot in [("Open in Browser", self._open_browser),
                            ("← New Run",      self.rerun_requested)]:
            b = _button(text)
            b.clicked.connect(slot)
            tbl.addWidget(b)
        self._push_btn = _button("Push to Cameras →", primary=True)
        self._push_btn.setEnabled(False)
        self._push_btn.clicked.connect(self.push_requested)
        tbl.addWidget(self._push_btn)
        root.addWidget(tb)

        self._sum_bar = QWidget()
        self._sum_bar.setFixedHeight(40)
        self._sum_bar.setStyleSheet(
            f"background:{PANEL_BG};border-bottom:1px solid {BORDER_COLOR};")
        self._sum_layout = QHBoxLayout(self._sum_bar)
        self._sum_layout.setContentsMargins(20, 0, 20, 0)
        self._sum_layout.setSpacing(24)
        root.addWidget(self._sum_bar)

        if _WEB_OK:
            self._web = QWebEngineView()
            root.addWidget(self._web)
        else:
            fb = QLabel("QWebEngineView unavailable. Click 'Open in Browser'.")
            fb.setAlignment(Qt.AlignmentFlag.AlignCenter)
            fb.setStyleSheet(f"color:{TEXT_MUTED};font-size:14px;background:{DARK_BG};")
            root.addWidget(fb)

        # ── Bottom action bar: commit the calibration → Push ──
        footer = QWidget()
        footer.setFixedHeight(64)
        footer.setStyleSheet(
            f"background:{DARK_BG};border-top:1px solid {BORDER_COLOR};")
        fl = QHBoxLayout(footer)
        fl.setContentsMargins(20, 0, 20, 0)
        fl.setSpacing(14)
        self._commit_status = _label(
            "Run a match to compute calibration values.",
            size=12, color=TEXT_MUTED)
        fl.addWidget(self._commit_status)
        fl.addStretch()
        self._commit_btn = _button("Commit Calibration →", primary=True)
        self._commit_btn.setEnabled(False)
        self._commit_btn.clicked.connect(self.commit_requested)
        fl.addWidget(self._commit_btn)
        root.addWidget(footer)

    def load_result(self, run_result, report_path: Optional[str]):
        self.state.run_result  = run_result
        self.state.report_path = report_path

        # Hero: array match % when verified — there is no FAIL
        amp = getattr(run_result, "array_match_pct", None)
        if amp is not None:
            badge_text = f"{amp:.0f}%"
            color = SUCCESS if amp >= 95.0 else WARNING if amp >= 80.0 else DANGER
            self._hero_line1.setText("Array Match — closed-loop verified")
            mmp = getattr(run_result, "min_match_pct", None)
            mclip = getattr(run_result, "min_match_clip_id", "")
            if mmp is not None and mmp < amp - 0.5:
                self._hero_line2.setText(f"Lowest: {mclip} at {mmp:.0f}%")
            else:
                self._hero_line2.setText("All cameras within measurement noise")
        else:
            status = getattr(run_result, "assessment_status", "UNKNOWN").upper()
            badge_text = status
            color = {
                "SOLVED": SUCCESS, "PARTIAL": WARNING, "NO_SOLVE": DANGER,
            }.get(status, TEXT_MUTED)
            self._hero_line1.setText("Match unverified")
            self._hero_line2.setText("No corrected renders measured this run")
        self._badge.setText(badge_text)
        self._badge.setStyleSheet(
            f"color:{color};font-size:30px;font-weight:800;"
            f"font-family:{MONO_FONT};background:transparent;")
        self._push_btn.setEnabled(bool(self.state.camera_ips))

        while self._sum_layout.count():
            item = self._sum_layout.takeAt(0)
            if item.widget(): item.widget().deleteLater()

        cameras  = run_result.cameras
        solved_count = sum(1 for cr in cameras if cr.commit is not None)
        exp_vals   = [cr.commit.exposure_adjust for cr in cameras if cr.commit]
        spread = max(exp_vals) - min(exp_vals) if len(exp_vals) > 1 else 0.0
        wb  = getattr(run_result, "wb_solve", None)
        k   = getattr(wb, "shared_kelvin", None) if wb else None

        _tiles = [
            ("Cameras",    f"{solved_count}/{len(cameras)} solved"),
            ("Exp Spread", f"{spread:+.3f} stops"),
            ("Shared K",   f"{k} K" if k else "—"),
        ]
        # Delivery-look match — present only when a project look was selected.
        _dname = getattr(run_result, "delivery_pipeline_name", "") or ""
        _dpct  = getattr(run_result, "delivery_array_match_pct", None)
        if _dname or _dpct is not None:
            _dval = f"{_dname}" + (f" · {_dpct:.0f}%" if _dpct is not None else "")
            _tiles.append(("Delivery look", _dval))
        else:
            _tiles.append(("Delivery look", "Reference only"))
        for lbl, val in _tiles:
            self._sum_layout.addWidget(_label(f"{lbl}:  ", size=11, color=TEXT_MUTED))
            self._sum_layout.addWidget(_label(val, size=11, mono=True))
        self._sum_layout.addStretch()

        # A new result invalidates any prior commit; re-arm the commit button.
        self.state.calibration_committed = False
        self.state.committed_at = None
        self._commit_btn.setEnabled(solved_count > 0)
        self._commit_btn.setText("Commit Calibration →")
        if solved_count > 0:
            self._commit_status.setText(
                f"{solved_count}/{len(cameras)} cameras calibrated — "
                "commit to carry these values to Push.")
            self._commit_status.setStyleSheet(
                f"color:{TEXT_PRIMARY};font-size:12px;background:transparent;")
        else:
            self._commit_status.setText(
                "No cameras solved — nothing to commit. Check Sphere QC.")
            self._commit_status.setStyleSheet(
                f"color:{TEXT_MUTED};font-size:12px;background:transparent;")

        if report_path and _WEB_OK:
            rp = Path(report_path).expanduser().resolve()
            if rp.exists():
                self._web.load(QUrl.fromLocalFile(str(rp)))
            else:
                # Report file missing — show inline fallback with key stats
                self._web.setHtml(self._build_inline_html(run_result))

    def mark_committed(self):
        """Reflect that the calibration has been committed and sent to Push."""
        self._commit_btn.setText("Committed ✓")
        self._commit_status.setText(
            "Calibration committed — values carried to the Push screen.")
        self._commit_status.setStyleSheet(
            f"color:{SUCCESS};font-size:12px;background:transparent;")

    def _build_inline_html(self, run_result) -> str:
        rows = ""
        for cr in run_result.cameras:
            ire  = f"{cr.measurement.hero_ire:.1f}" if cr.measurement else "—"
            ea   = f"{cr.commit.exposure_adjust:+.3f}" if cr.commit else "—"
            _eo  = getattr(cr.commit, "exposure_only", False) if cr.commit else False
            k    = "—" if _eo else (f"{cr.commit.kelvin}" if cr.commit else "—")
            t    = "—" if _eo else (f"{cr.commit.tint:+.2f}" if cr.commit else "—")
            mp   = getattr(cr, "match_pct", None)
            sc   = f"{mp:.0f}%" if mp is not None else (cr.status or "—")
            # Metadata
            fps  = "—"
            res  = "—"
            ar   = "—"
            iso  = "—"
            if cr.metadata:
                if cr.metadata.fps:
                    fps = f"{cr.metadata.fps:.2f}".rstrip('0').rstrip('.')
                w, h = cr.metadata.frame_width, cr.metadata.frame_height
                if w and h:
                    res = f"{w}×{h}"
                    from math import gcd
                    g = gcd(w, h)
                    ar = f"{w//g}:{h//g}"
                if cr.metadata.iso:
                    iso = f"ISO {cr.metadata.iso}"
            if mp is not None:
                sc_color = "#34d399" if mp >= 95.0 else "#fbbf24" if mp >= 80.0 else "#ff7a8a"
            else:
                sc_color = "#34d399" if sc == "SOLVED" else "#fbbf24" if sc == "NEEDS_ASSIST" else "#ff7a8a"
            rows += (
                f"<tr>"
                f"<td style='color:#f0e9df;font-weight:600'>{cr.camera_label}</td>"
                f"<td>{ire}</td><td>{ea}</td><td>{k}</td><td>{t}</td>"
                f"<td style='color:{sc_color};font-weight:600'>{sc}</td>"
                f"<td style='color:#8d8273'>{fps}</td>"
                f"<td style='color:#8d8273'>{res}</td>"
                f"<td style='color:#8d8273'>{ar}</td>"
                f"<td style='color:#8d8273'>{iso}</td>"
                f"</tr>"
            )
        anchor = getattr(run_result, 'anchor_ire', None)
        wb = getattr(run_result, 'wb_solve', None)
        k_shared = getattr(wb, 'shared_kelvin', '—') if wb else '—'
        spread = getattr(run_result, 'exposure_spread', None)
        spread_str = f"{spread.max_stops - spread.min_stops:+.3f} stops" if spread else "—"
        return f"""<!DOCTYPE html><html><head>
<meta charset="utf-8">
<style>
  body{{font-family:-apple-system,sans-serif;background:{DARK_BG};color:{TEXT_PRIMARY};
        padding:24px;margin:0;}}
  h2{{color:{ACCENT};margin:0 0 4px 0;font-size:18px;}}
  .meta{{color:{TEXT_MUTED};font-size:12px;margin-bottom:20px;}}
  .summary{{display:flex;gap:32px;margin-bottom:20px;padding:12px 16px;
             background:{PANEL_BG};border-radius:8px;border:1px solid {BORDER_COLOR};}}
  .sum-item{{display:flex;flex-direction:column;gap:2px;}}
  .sum-label{{color:{TEXT_MUTED};font-size:11px;}}
  .sum-val{{color:{TEXT_PRIMARY};font-size:14px;font-weight:600;font-family:monospace;}}
  table{{width:100%;border-collapse:collapse;font-size:12px;}}
  th{{text-align:left;color:{TEXT_MUTED};padding:6px 10px;
      border-bottom:1px solid {BORDER_COLOR};font-weight:600;white-space:nowrap;}}
  td{{padding:6px 10px;border-bottom:1px solid {PANEL_BG};
      font-family:monospace;white-space:nowrap;}}
  tr:hover td{{background:{PANEL_BG};}}
</style></head><body>
<h2>R3DMatch v4 — Assessment</h2>
<div class="meta">{run_result.input_path}</div>
<div class="summary">
  <div class="sum-item"><span class="sum-label">Cameras</span>
    <span class="sum-val">{sum(1 for c in run_result.cameras if c.status=='PASS')}/{len(run_result.cameras)} PASS</span></div>
  <div class="sum-item"><span class="sum-label">Anchor IRE</span>
    <span class="sum-val">{f"{anchor:.1f}" if anchor else "—"}</span></div>
  <div class="sum-item"><span class="sum-label">Exp Spread</span>
    <span class="sum-val">{spread_str}</span></div>
  <div class="sum-item"><span class="sum-label">Shared K</span>
    <span class="sum-val">{k_shared} K</span></div>
</div>
<table>
  <tr><th>Camera</th><th>IRE</th><th>Exp Adj</th><th>Kelvin</th>
      <th>Tint</th><th>Status</th><th>FPS</th><th>Resolution</th>
      <th>Aspect</th><th>ISO</th></tr>
  {rows}
</table>
</body></html>"""

    def _open_browser(self):
        if self.state.report_path:
            import webbrowser
            webbrowser.open(f"file://{self.state.report_path}")


# ── Screen 4: Push ────────────────────────────────────────────────────────────

class PushScreen(QWidget):
    back_requested = Signal()

    def __init__(self, state: AppState):
        super().__init__()
        self.state = state
        self._camera_rows: Dict[str, dict] = {}
        self._worker: Optional[PushWorker] = None
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(16)
        root.setContentsMargins(40, 32, 40, 32)

        hdr = QHBoxLayout()
        back = _button("← Results")
        back.clicked.connect(self.back_requested)
        hdr.addWidget(back); hdr.addStretch()
        root.addLayout(hdr)

        root.addWidget(_label("Push to Cameras", size=22, bold=True))
        root.addWidget(_label(
            "Review commit values. Dry-run first — connects, reads, and verifies without writing.",
            size=13, color=TEXT_MUTED))
        root.addWidget(_sep())

        self._table = QTableWidget(0, 7)
        self._table.setHorizontalHeaderLabels(
            ["", "Camera", "Match", "Exp Adj", "Kelvin", "Tint", "Status"])
        hdr2 = self._table.horizontalHeader()
        hdr2.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        for i in range(1, 6):
            hdr2.setSectionResizeMode(i, QHeaderView.ResizeMode.ResizeToContents)
        hdr2.setSectionResizeMode(6, QHeaderView.ResizeMode.Stretch)
        self._table.setColumnWidth(0, 36)
        self._table.setStyleSheet(_table_style())
        self._table.verticalHeader().setVisible(False)
        root.addWidget(self._table)

        root.addWidget(_label("Push Log", size=12, bold=True, color=TEXT_MUTED))
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setStyleSheet(
            f"background:#0b0907;color:#34d399;font-family:{MONO_FONT};"
            f"font-size:11px;border:1px solid {BORDER_COLOR};border-radius:6px;"
            f"padding:8px;")
        self._log.setMaximumHeight(130)
        root.addWidget(self._log)
        root.addStretch()

        br = QHBoxLayout()
        for text, fn in [("Select All", self._sel_all), ("Select None", self._sel_none)]:
            b = _button(text)
            b.clicked.connect(fn)
            br.addWidget(b)
        br.addSpacing(16)
        detect_btn = _button("Detect Cameras…")
        detect_btn.clicked.connect(self._detect_cameras)
        br.addWidget(detect_btn)
        self._reset_btn = _button("Reset to Default")
        self._reset_btn.clicked.connect(self._start_reset)
        br.addWidget(self._reset_btn)
        br.addStretch()
        self._dry_btn  = _button("Dry Run (no write)")
        self._push_btn = _button("Push to Cameras", primary=True)
        self._dry_btn.clicked.connect(lambda: self._start_push(dry_run=True))
        self._push_btn.clicked.connect(lambda: self._start_push(dry_run=False))
        br.addWidget(self._dry_btn); br.addWidget(self._push_btn)
        root.addLayout(br)

    def populate(self):
        self._table.setRowCount(0)
        self._camera_rows.clear()
        self._log.clear()
        rr = self.state.run_result
        if not rr: return
        for cr in rr.cameras:
            if not cr.commit: continue
            ip  = self.state.camera_ips.get(cr.camera_label, "")
            row = self._table.rowCount()
            self._table.insertRow(row)
            cb = QCheckBox()
            cb.setChecked(bool(ip))
            cbw = QWidget()
            cbl = QHBoxLayout(cbw)
            cbl.setContentsMargins(4, 0, 0, 0)
            cbl.addWidget(cb)
            self._table.setCellWidget(row, 0, cbw)
            self._table.setItem(row, 1, QTableWidgetItem(cr.camera_label))
            mp = getattr(cr, "match_pct", None)
            mp_item = QTableWidgetItem(f"{mp:.0f}%" if mp is not None else "—")
            if mp is not None:
                mp_item.setForeground(QColor(
                    SUCCESS if mp >= 95.0 else WARNING if mp >= 80.0 else DANGER))
            else:
                mp_item.setForeground(QColor(TEXT_MUTED))
            self._table.setItem(row, 2, mp_item)
            ea = cr.commit.exposure_adjust
            ea_item = QTableWidgetItem(f"{ea:+.3f}")
            ea_item.setForeground(QColor(
                SUCCESS if abs(ea) <= 0.25 else WARNING if abs(ea) <= 0.75 else DANGER))
            self._table.setItem(row, 3, ea_item)
            _eo = getattr(cr.commit, "exposure_only", False)
            self._table.setItem(row, 4, QTableWidgetItem("—" if _eo else f"{cr.commit.kelvin} K"))
            self._table.setItem(row, 5, QTableWidgetItem("—" if _eo else f"{cr.commit.tint:+.2f}"))
            si = QTableWidgetItem("Ready" if ip else "No IP")
            si.setForeground(QColor(TEXT_MUTED if not ip else TEXT_PRIMARY))
            self._table.setItem(row, 6, si)
            self._camera_rows[cr.camera_label] = {
                "checkbox": cb, "row": row, "commit": cr.commit, "ip": ip}
        if getattr(self.state, "calibration_committed", False):
            n_ip = sum(1 for i in self._camera_rows.values() if i["ip"])
            self._log.append(
                f"[COMMIT] Calibration committed {self.state.committed_at or ''} — "
                f"{len(self._camera_rows)} camera(s) carried over, {n_ip} with IPs."
                + ("" if n_ip else " Use Detect Cameras to map IPs."))

    def _sel_all(self):
        for i in self._camera_rows.values(): i["checkbox"].setChecked(True)

    def _sel_none(self):
        for i in self._camera_rows.values(): i["checkbox"].setChecked(False)

    # ── Detect Cameras (on Push) ────────────────────────────────────────────
    def _detect_cameras(self):
        """Scan the network for RCP2 cameras and auto-fill push rows lacking IPs."""
        try:
            self._open_detect_dialog()
        except Exception as exc:
            QMessageBox.critical(
                self, "Detect Cameras failed",
                f"{type(exc).__name__}: {exc}")

    def _unmapped_labels(self) -> List[str]:
        """Push-row camera labels currently without an IP, in table order."""
        return [lbl for lbl, info in sorted(
                    self._camera_rows.items(), key=lambda kv: kv[1]["row"])
                if not info.get("ip")]

    def _open_detect_dialog(self):
        interfaces = _list_ipv4_interfaces()
        cidrs = _candidate_scan_cidrs()

        dlg = QDialog(self)
        dlg.setWindowTitle("Detect Cameras on Network")
        dlg.setMinimumWidth(460)
        dlg.setStyleSheet(f"background:{BG};color:{TEXT_PRIMARY};")
        vl = QVBoxLayout(dlg)

        if interfaces:
            hint = "  •  ".join(
                f"{i['name']} {i['ipv4']}" + (" (link-local)" if i['link_local'] else "")
                for i in interfaces)
        else:
            hint = "No active network interfaces detected."
        vl.addWidget(_label("Detected interfaces:", size=11, bold=True, color=TEXT_MUTED))
        il = _label(hint, size=11, color=TEXT_MUTED, mono=True)
        il.setWordWrap(True)
        vl.addWidget(il)

        n_unmapped = len(self._unmapped_labels())
        vl.addWidget(_label(
            f"{n_unmapped} camera(s) on this page have no IP. Found cameras are "
            "assigned to those rows in order — verify before pushing.",
            size=11, color=TEXT_MUTED))

        vl.addWidget(_label("Subnet to scan (CIDR):", size=11, bold=True, color=TEXT_MUTED))
        cidr_combo = QComboBox()
        cidr_combo.setEditable(True)
        cidr_combo.addItems(cidrs)
        cidr_combo.setCurrentText(cidrs[0] if cidrs else DEFAULT_CAMERA_CIDR)
        cidr_combo.setStyleSheet(_field_style(mono=True))
        vl.addWidget(cidr_combo)

        status_lbl = _label("Ready.", size=12, color=TEXT_MUTED)
        vl.addWidget(status_lbl)
        result_lbl = _label("", size=12, mono=True)
        result_lbl.setWordWrap(True)
        vl.addWidget(result_lbl)

        btns = QHBoxLayout()
        scan_btn = _button("Scan", primary=True)
        stop_btn = _button("Stop"); stop_btn.setEnabled(False)
        close_btn = _button("Close")
        btns.addWidget(scan_btn); btns.addWidget(stop_btn)
        btns.addStretch(); btns.addWidget(close_btn)
        vl.addLayout(btns)

        found_map: Dict[str, str] = {}

        def _render():
            if not found_map:
                result_lbl.setText("")
                return
            lines = []
            for ip in sorted(found_map, key=lambda x: tuple(int(p) for p in x.split("."))):
                info = found_map[ip]
                lines.append(f"{ip}   {info}" if info else f"{ip}   (port open)")
            result_lbl.setText(f"Found {len(lines)} device(s):\n" + "\n".join(lines))

        def _on_found(ip, info):
            if ip not in found_map or info:
                found_map[ip] = info
            _render()

        def _on_progress(done, total):
            status_lbl.setText(f"Scanning… {done}/{total}")

        def _on_done(confirmed):
            scan_btn.setEnabled(True); stop_btn.setEnabled(False)
            cameras = [ip for ip, info in confirmed if info]
            if not found_map:
                status_lbl.setText("No devices found on port 9998.")
                return
            # Confirmed cameras first, then any remaining open ports.
            ordered = cameras + [ip for ip in sorted(
                found_map, key=lambda x: tuple(int(p) for p in x.split(".")))
                if ip not in cameras]
            assigned = self._assign_detected(ordered)
            if assigned:
                status_lbl.setText(
                    f"Done — assigned {assigned} IP(s) to unmapped camera(s).")
            elif cameras:
                status_lbl.setText(
                    f"Done — {len(cameras)} camera(s) found, but no unmapped rows to fill.")
            else:
                status_lbl.setText("Done — ports open but no camera confirmed.")

        def _start():
            found_map.clear(); _render()
            try:
                hosts = _hosts_for_cidr(cidr_combo.currentText().strip())
            except Exception as exc:
                status_lbl.setText(f"Invalid subnet: {exc}")
                return
            scan_btn.setEnabled(False); stop_btn.setEnabled(True)
            status_lbl.setText(f"Scanning {len(hosts)} hosts on port {RCP2_SCAN_PORT}…")
            self._scan_worker = _NetworkScanWorker(hosts)
            self._scan_worker.found.connect(_on_found)
            self._scan_worker.progress.connect(_on_progress)
            self._scan_worker.done.connect(_on_done)
            self._scan_worker.start()

        def _stop():
            if getattr(self, "_scan_worker", None):
                self._scan_worker.stop()
            stop_btn.setEnabled(False)
            status_lbl.setText("Stopping…")

        def _close():
            w = getattr(self, "_scan_worker", None)
            if w:
                w.stop()
                for sig in (w.found, w.progress, w.done):
                    try:
                        sig.disconnect()
                    except Exception:
                        pass
            dlg.accept()

        scan_btn.clicked.connect(_start)
        stop_btn.clicked.connect(_stop)
        close_btn.clicked.connect(_close)
        dlg.exec()

    def _assign_detected(self, ordered_ips: List[str]) -> int:
        """Fill push rows that have no IP with detected IPs, in table order.

        Returns the number of rows assigned. Updates the table, in-memory
        state, and the persisted IP map so the pairing survives navigation.
        """
        targets = self._unmapped_labels()
        if not targets or not ordered_ips:
            return 0
        # Don't reuse an IP that's already mapped to another row.
        used = {info["ip"] for info in self._camera_rows.values() if info.get("ip")}
        fresh = [ip for ip in ordered_ips if ip not in used]
        new_pairs: Dict[str, str] = {}
        for lbl, ip in zip(targets, fresh):
            info = self._camera_rows[lbl]
            info["ip"] = ip
            row = info["row"]
            self._table.setItem(row, 1, QTableWidgetItem(lbl))
            si = QTableWidgetItem("Ready")
            si.setForeground(QColor(TEXT_PRIMARY))
            self._table.setItem(row, 6, si)
            info["checkbox"].setChecked(True)
            self.state.camera_ips[lbl] = ip
            new_pairs[lbl] = ip
            self._log.append(f"[DETECT] {lbl} → {ip}")
        if new_pairs:
            self._persist_ips()
        return len(new_pairs)

    def _persist_ips(self):
        """Merge the current label→IP map into camera_ips.json."""
        merged: Dict[str, str] = {}
        if IP_MAP_PATH.exists():
            try:
                merged = json.loads(IP_MAP_PATH.read_text())
            except Exception:
                merged = {}
        merged.update({l: i for l, i in self.state.camera_ips.items() if l and i})
        self.state.camera_ips = merged
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        try:
            IP_MAP_PATH.write_text(json.dumps(merged, indent=2))
        except Exception:
            pass

    def _start_push(self, dry_run: bool):
        targets = []
        for lbl, info in self._camera_rows.items():
            if not info["checkbox"].isChecked(): continue
            if not info["ip"]:
                self._log.append(f"[SKIP] {lbl}: no IP")
                continue
            targets.append({"camera_label": lbl, "ip": info["ip"], "commit": info["commit"]})
        if not targets:
            QMessageBox.information(self, "Nothing to Push",
                                    "Select cameras with configured IPs.")
            return
        self._log.append(f"\n[{'DRY-RUN' if dry_run else 'LIVE'}] {len(targets)} cameras…\n")
        self._set_busy(True)
        self._worker = PushWorker(targets, dry_run=dry_run)
        self._worker.camera_update.connect(self._on_cam_update)
        self._worker.finished.connect(self._on_done)
        self._worker.errored.connect(self._on_error)
        self._worker.start()

    def _start_reset(self):
        targets = []
        for lbl, info in self._camera_rows.items():
            if not info["checkbox"].isChecked():
                continue
            if not info["ip"]:
                self._log.append(f"[SKIP] {lbl}: no IP")
                continue
            targets.append({"camera_label": lbl, "ip": info["ip"]})
        if not targets:
            QMessageBox.information(self, "Nothing to Reset",
                                    "Select cameras with configured IPs.")
            return
        resp = QMessageBox.warning(
            self, "Reset to Default",
            f"Reset {len(targets)} camera(s) to neutral defaults?\n\n"
            "This writes 5600K, 0 tint, and 0 exposure adjust to the live "
            "camera(s), discarding the current calibration.",
            QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel)
        if resp != QMessageBox.StandardButton.Ok:
            return
        self._log.append(f"\n[RESET] {len(targets)} cameras → neutral defaults…\n")
        self._set_busy(True)
        self._worker = PushWorker(targets, dry_run=False, mode="reset")
        self._worker.camera_update.connect(self._on_cam_update)
        self._worker.finished.connect(self._on_done)
        self._worker.errored.connect(self._on_error)
        self._worker.start()

    def _set_busy(self, busy: bool):
        for b in (self._push_btn, self._dry_btn, self._reset_btn):
            b.setEnabled(not busy)

    @Slot(str, bool, str)
    def _on_cam_update(self, lbl: str, ok: bool, detail: str):
        if lbl not in self._camera_rows: return
        row  = self._camera_rows[lbl]["row"]
        item = QTableWidgetItem(f"{'✓' if ok else '✗'} {detail}")
        item.setForeground(QColor(SUCCESS if ok else DANGER))
        self._table.setItem(row, 6, item)
        self._log.append(f"[{'OK' if ok else 'FAIL'}] {lbl}: {detail}")

    @Slot(list)
    def _on_done(self, results):
        self.state.push_results = results
        s = summarize_push_results(results)["summary"]
        self._log.append(f"\n[DONE] {s['succeeded']}/{s['total_cameras']} succeeded.")
        self._set_busy(False)

    @Slot(str)
    def _on_error(self, msg: str):
        self._log.append(f"\n[ERROR]\n{msg}")
        self._set_busy(False)


# ── Screen 5: Camera Network ──────────────────────────────────────────────────

class CameraNetworkScreen(QWidget):
    def __init__(self, state: AppState):
        super().__init__()
        self.state     = state
        self._ip_rows: List[tuple] = []
        self._build_ui()
        self._load_ips()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(16)
        root.setContentsMargins(40, 32, 40, 32)
        root.addWidget(_label("Camera Network", size=22, bold=True))
        root.addWidget(_label(
            "Map camera labels to IP addresses. Saved automatically. "
            "Test verifies RCP2 connectivity.",
            size=13, color=TEXT_MUTED))
        root.addWidget(_sep())

        hdr_row = QHBoxLayout()
        for txt, w in [("Camera Label", 130), ("IP Address", 160), ("Connection", 200)]:
            l = _label(txt, size=11, bold=True, color=TEXT_MUTED)
            l.setFixedWidth(w)
            hdr_row.addWidget(l)
        hdr_row.addStretch()
        root.addLayout(hdr_row)

        ip_layout = QVBoxLayout()
        ip_layout.setSpacing(6)
        for _ in range(16):
            row = QHBoxLayout()
            le = QLineEdit(); le.setPlaceholderText("G007_A")
            le.setFixedWidth(130); le.setStyleSheet(_field_style(mono=True))
            ie = QLineEdit(); ie.setPlaceholderText("10.x.x.x")
            ie.setFixedWidth(160); ie.setStyleSheet(_field_style(mono=True))
            tb = _button("Test"); tb.setFixedWidth(60)
            sl = _label("", size=11, color=TEXT_MUTED, mono=True)
            sl.setFixedWidth(240)

            def _make_test(l=le, i=ie, s=sl):
                def _t(): self._test(l.text().strip(), i.text().strip(), s)
                return _t

            tb.clicked.connect(_make_test())
            le.textChanged.connect(self._save_ips)
            ie.textChanged.connect(self._save_ips)
            row.addWidget(le); row.addSpacing(8)
            row.addWidget(ie); row.addSpacing(8)
            row.addWidget(tb); row.addSpacing(8)
            row.addWidget(sl); row.addStretch()
            ip_layout.addLayout(row)
            self._ip_rows.append((le, ie, sl))

        sw = QWidget()
        sw.setLayout(ip_layout)
        sw.setStyleSheet(f"background:{PANEL_BG};")
        self._scroll_widget = sw
        sc = QScrollArea()
        sc.setWidget(sw); sc.setWidgetResizable(True)
        sc.setStyleSheet(
            f"QScrollArea{{background:{PANEL_BG};"
            f"border:1px solid {BORDER_COLOR};border-radius:6px;}}"
            f"QScrollBar:vertical{{background:{DARK_BG};width:8px;border-radius:4px;}}"
            f"QScrollBar::handle:vertical{{background:{BORDER_STRONG};border-radius:4px;}}")
        root.addWidget(sc)

        br = QHBoxLayout()
        ta = _button("Test All")
        ta.clicked.connect(self._test_all)
        ca = _button("Clear All")
        ca.clicked.connect(self._clear_all)
        bi = _button("Bulk Import…")
        bi.clicked.connect(self._bulk_import)
        dn = _button("Detect on Network…")
        dn.clicked.connect(self._detect_on_network)
        br.addWidget(ta); br.addWidget(ca)
        br.addSpacing(16)
        br.addWidget(bi); br.addWidget(dn)
        br.addStretch()
        root.addLayout(br)

    def _bulk_import(self):
        """Paste label/IP pairs from a spreadsheet (tolerant of format)."""
        try:
            self._open_bulk_import_dialog()
        except Exception as exc:
            QMessageBox.critical(
                self, "Bulk Import failed",
                f"{type(exc).__name__}: {exc}")

    def _open_bulk_import_dialog(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("Bulk Import Camera IPs")
        dlg.setMinimumWidth(480)
        dlg.setStyleSheet(f"background:{BG};color:{TEXT_PRIMARY};")
        vl = QVBoxLayout(dlg)
        vl.addWidget(_label(
            "Paste camera label and IP columns — tab, comma, or space separated.\n"
            "Either column order works; lines without an IP are ignored:",
            size=12, color=TEXT_MUTED))
        ex = _label("AA   172.20.114.141\nAB,172.20.114.142", size=11,
                     color=TEXT_MUTED, mono=True)
        ex.setIndent(12)
        vl.addWidget(ex)
        ta = QTextEdit()
        ta.setPlaceholderText("Paste here…")
        ta.setStyleSheet(f"background:{PANEL_BG};color:{TEXT_PRIMARY};"
                         f"border:1px solid {BORDER_COLOR};border-radius:4px;"
                         f"font-family:{MONO_FONT};font-size:12px;padding:8px;")
        ta.setMinimumHeight(200)
        vl.addWidget(ta)
        status = _label("", size=12, color=WARNING)
        status.setWordWrap(True)
        vl.addWidget(status)
        btns = QHBoxLayout()
        ok_btn = _button("Import", primary=True)
        cn_btn = _button("Cancel")
        btns.addWidget(ok_btn); btns.addWidget(cn_btn); btns.addStretch()
        vl.addLayout(btns)
        cn_btn.clicked.connect(dlg.reject)

        def _do_import():
            pairs = _parse_bulk_pairs(ta.toPlainText())
            if not pairs:
                status.setText(
                    "No valid rows found — each line needs an IP address "
                    "(e.g. \"AA 172.20.114.141\").")
                return
            while len(self._ip_rows) < len(pairs):
                self._add_row()
            for i, (label, ip) in enumerate(pairs):
                le, ie, sl = self._ip_rows[i]
                le.setText(label)
                ie.setText(ip)
                sl.clear()
            self._save_ips()
            dlg.accept()

        ok_btn.clicked.connect(_do_import)
        dlg.exec()

    def _add_row(self):
        """Dynamically add a new camera row."""
        ip_layout = self._scroll_widget.layout()
        row = QHBoxLayout()
        le = QLineEdit(); le.setPlaceholderText("G007_A")
        le.setFixedWidth(130); le.setStyleSheet(_field_style(mono=True))
        ie = QLineEdit(); ie.setPlaceholderText("10.x.x.x")
        ie.setFixedWidth(160); ie.setStyleSheet(_field_style(mono=True))
        tb = _button("Test"); tb.setFixedWidth(60)
        sl = _label("", size=11, color=TEXT_MUTED, mono=True)
        sl.setFixedWidth(240)
        def _make_test(l=le, i=ie, s=sl):
            def _t(): self._test(l.text().strip(), i.text().strip(), s)
            return _t
        tb.clicked.connect(_make_test())
        le.textChanged.connect(self._save_ips)
        ie.textChanged.connect(self._save_ips)
        row.addWidget(le); row.addSpacing(8)
        row.addWidget(ie); row.addSpacing(8)
        row.addWidget(tb); row.addSpacing(8)
        row.addWidget(sl); row.addStretch()
        ip_layout.addLayout(row)
        self._ip_rows.append((le, ie, sl))

    def _detect_on_network(self):
        """Scan a subnet for RCP2 cameras on port 9998 (threaded, non-blocking)."""
        try:
            self._open_detect_dialog()
        except Exception as exc:
            QMessageBox.critical(
                self, "Detect on Network failed",
                f"{type(exc).__name__}: {exc}")

    def _open_detect_dialog(self):
        interfaces = _list_ipv4_interfaces()
        cidrs = _candidate_scan_cidrs()

        dlg = QDialog(self)
        dlg.setWindowTitle("Detect Cameras on Network")
        dlg.setMinimumWidth(460)
        dlg.setStyleSheet(f"background:{BG};color:{TEXT_PRIMARY};")
        vl = QVBoxLayout(dlg)

        if interfaces:
            hint = "  •  ".join(
                f"{i['name']} {i['ipv4']}" + (" (link-local)" if i['link_local'] else "")
                for i in interfaces)
        else:
            hint = "No active network interfaces detected."
        vl.addWidget(_label("Detected interfaces:", size=11, bold=True, color=TEXT_MUTED))
        il = _label(hint, size=11, color=TEXT_MUTED, mono=True)
        il.setWordWrap(True)
        vl.addWidget(il)

        vl.addWidget(_label("Subnet to scan (CIDR):", size=11, bold=True, color=TEXT_MUTED))
        cidr_combo = QComboBox()
        cidr_combo.setEditable(True)
        cidr_combo.addItems(cidrs)
        cidr_combo.setCurrentText(cidrs[0] if cidrs else DEFAULT_CAMERA_CIDR)
        cidr_combo.setStyleSheet(_field_style(mono=True))
        vl.addWidget(cidr_combo)

        status_lbl = _label("Ready.", size=12, color=TEXT_MUTED)
        vl.addWidget(status_lbl)
        result_lbl = _label("", size=12, mono=True)
        result_lbl.setWordWrap(True)
        vl.addWidget(result_lbl)

        btns = QHBoxLayout()
        scan_btn = _button("Scan", primary=True)
        stop_btn = _button("Stop"); stop_btn.setEnabled(False)
        close_btn = _button("Close")
        btns.addWidget(scan_btn); btns.addWidget(stop_btn)
        btns.addStretch(); btns.addWidget(close_btn)
        vl.addLayout(btns)

        found_map: Dict[str, str] = {}

        def _render():
            if not found_map:
                result_lbl.setText("")
                return
            lines = []
            for ip in sorted(found_map, key=lambda x: tuple(int(p) for p in x.split("."))):
                info = found_map[ip]
                lines.append(f"{ip}   {info}" if info else f"{ip}   (port open)")
            result_lbl.setText(f"Found {len(lines)} device(s):\n" + "\n".join(lines))

        def _on_found(ip, info):
            if ip not in found_map or info:
                found_map[ip] = info
            _render()

        def _on_progress(done, total):
            status_lbl.setText(f"Scanning… {done}/{total}")

        def _on_done(confirmed):
            scan_btn.setEnabled(True); stop_btn.setEnabled(False)
            cameras = [ip for ip, info in confirmed if info]
            if not found_map:
                status_lbl.setText("No devices found on port 9998.")
            elif cameras:
                status_lbl.setText(f"Done — {len(cameras)} camera(s) confirmed.")
            else:
                status_lbl.setText("Done — ports open but no camera confirmed.")
            # Auto-populate empty rows: confirmed cameras first, then open ports.
            ordered = cameras + [ip for ip in sorted(found_map) if ip not in cameras]
            empty = [(le, ie, sl) for le, ie, sl in self._ip_rows if not ie.text().strip()]
            for (le, ie, sl), ip in zip(empty, ordered):
                ie.setText(ip)
            if ordered:
                self._save_ips()

        def _start():
            found_map.clear(); _render()
            try:
                hosts = _hosts_for_cidr(cidr_combo.currentText().strip())
            except Exception as exc:
                status_lbl.setText(f"Invalid subnet: {exc}")
                return
            scan_btn.setEnabled(False); stop_btn.setEnabled(True)
            status_lbl.setText(f"Scanning {len(hosts)} hosts on port {RCP2_SCAN_PORT}…")
            self._scan_worker = _NetworkScanWorker(hosts)
            self._scan_worker.found.connect(_on_found)
            self._scan_worker.progress.connect(_on_progress)
            self._scan_worker.done.connect(_on_done)
            self._scan_worker.start()

        def _stop():
            if getattr(self, "_scan_worker", None):
                self._scan_worker.stop()
            stop_btn.setEnabled(False)
            status_lbl.setText("Stopping…")

        def _close():
            w = getattr(self, "_scan_worker", None)
            if w:
                w.stop()
                for sig in (w.found, w.progress, w.done):
                    try:
                        sig.disconnect()
                    except Exception:
                        pass
            dlg.accept()

        scan_btn.clicked.connect(_start)
        stop_btn.clicked.connect(_stop)
        close_btn.clicked.connect(_close)
        dlg.exec()

    def _test(self, label: str, ip: str, status_lbl: QLabel):
        if not ip:
            status_lbl.setText("no IP")
            return
        status_lbl.setText("connecting…")
        status_lbl.setStyleSheet(
            f"color:{ACCENT};background:transparent;font-family:{MONO_FONT};font-size:11px;")
        QApplication.processEvents()

        import asyncio
        from r3dmatch3.rcp2 import RCP2Session

        async def _ping():
            s = RCP2Session(ip, label or ip, dry_run=False,
                            connect_timeout=4.0, handshake_timeout=4.0)
            try:
                await s.connect()
                info = await s.get_camera_info()
                cam = (info.get("camera_type") or {}).get("str", "unknown")
                fw  = (info.get("version") or {}).get("str", "")
                return f"✓ {cam}  {fw}"
            except Exception as e:
                return f"✗ {type(e).__name__}"
            finally:
                await s.close()

        try:
            result = asyncio.run(_ping())
            ok = result.startswith("✓")
            status_lbl.setText(result)
            status_lbl.setStyleSheet(
                f"color:{'#34d399' if ok else '#ff7a8a'};"
                f"background:transparent;font-family:{MONO_FONT};font-size:11px;")
        except Exception as e:
            status_lbl.setText(f"✗ {e}")
            status_lbl.setStyleSheet(
                f"color:{DANGER};background:transparent;"
                f"font-family:{MONO_FONT};font-size:11px;")

    def _test_all(self):
        for le, ie, sl in self._ip_rows:
            if ie.text().strip():
                self._test(le.text().strip(), ie.text().strip(), sl)

    def _clear_all(self):
        for le, ie, sl in self._ip_rows:
            le.clear(); ie.clear(); sl.clear()
        self._save_ips()

    def _load_ips(self):
        if IP_MAP_PATH.exists():
            try:
                data = json.loads(IP_MAP_PATH.read_text())
                self.state.camera_ips = data
                for i, (lbl, ip) in enumerate(list(data.items())[:len(self._ip_rows)]):
                    self._ip_rows[i][0].setText(lbl)
                    self._ip_rows[i][1].setText(ip)
            except Exception:
                pass

    def _save_ips(self):
        m = {}
        for le, ie, _ in self._ip_rows:
            l, i = le.text().strip(), ie.text().strip()
            if l and i: m[l] = i
        self.state.camera_ips = m
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        try:
            IP_MAP_PATH.write_text(json.dumps(m, indent=2))
        except Exception:
            pass


# ── Main window ────────────────────────────────────────────────────────────────

# ── Screen 6: Settings ────────────────────────────────────────────────────────

class SettingsScreen(QWidget):
    """
    Application settings — REDLine path and default output directory.
    Changes are saved to ~/Library/Application Support/R3DMatch_v3/settings.json.
    """

    def __init__(self, state: AppState):
        super().__init__()
        self.state = state
        self.setStyleSheet(f"background:{DARK_BG};")
        root = QVBoxLayout(self)
        root.setContentsMargins(32, 28, 32, 28)
        root.setSpacing(20)

        root.addWidget(_label("Settings", size=22, bold=True))

        # ── REDLine section ──────────────────────────────────────────────────
        rl_card = QWidget()
        rl_card.setStyleSheet(
            f"background:{CARD_BG};border-radius:8px;"
            f"border:1px solid {BORDER_COLOR};")
        rl_lay = QVBoxLayout(rl_card)
        rl_lay.setContentsMargins(20, 16, 20, 16)
        rl_lay.setSpacing(10)

        rl_lay.addWidget(_label("REDLine Executable", bold=True))
        rl_lay.addWidget(_label(
            "Leave blank to use auto-discovery. Set manually if REDCINE-X PRO "
            "is installed in a non-standard location.",
            color=TEXT_MUTED, size=11))

        path_row = QHBoxLayout()
        path_row.setSpacing(8)
        self._rl_path = QLineEdit()
        self._rl_path.setPlaceholderText(
            "Auto-detect  (e.g. /Applications/REDCINE-X PRO.app/…/REDline)")
        self._rl_path.setStyleSheet(_field_style())
        self._rl_path.setText(state.redline_path)
        path_row.addWidget(self._rl_path)
        browse_btn = _button("Browse…")
        browse_btn.clicked.connect(self._browse_redline)
        path_row.addWidget(browse_btn)
        rl_lay.addLayout(path_row)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        test_btn = _button("Test")
        test_btn.clicked.connect(self._test_redline)
        btn_row.addWidget(test_btn)
        btn_row.addStretch()
        rl_lay.addLayout(btn_row)

        self._rl_status = QLabel("")
        self._rl_status.setStyleSheet(f"color:{TEXT_MUTED};font-size:12px;")
        self._rl_status.setWordWrap(True)
        rl_lay.addWidget(self._rl_status)

        root.addWidget(rl_card)

        # ── Default output directory section ────────────────────────────────
        out_card = QWidget()
        out_card.setStyleSheet(
            f"background:{CARD_BG};border-radius:8px;"
            f"border:1px solid {BORDER_COLOR};")
        out_lay = QVBoxLayout(out_card)
        out_lay.setContentsMargins(20, 16, 20, 16)
        out_lay.setSpacing(10)

        out_lay.addWidget(_label("Default Output Directory", bold=True))
        out_lay.addWidget(_label(
            "Pre-fills the Output field on the Setup screen. "
            "Leave blank to derive automatically from the input folder.",
            color=TEXT_MUTED, size=11))

        out_row = QHBoxLayout()
        out_row.setSpacing(8)
        self._out_path = QLineEdit()
        self._out_path.setPlaceholderText(
            "e.g. /Volumes/PROJECTS/R3DMatch_Runs/")
        self._out_path.setStyleSheet(_field_style())
        self._out_path.setText(state.default_out_dir)
        out_row.addWidget(self._out_path)
        out_browse = _button("Browse…")
        out_browse.clicked.connect(self._browse_out)
        out_row.addWidget(out_browse)
        out_lay.addLayout(out_row)

        root.addWidget(out_card)

        # ── Validation / diagnostic flags ────────────────────────────────────
        val_card = QWidget()
        val_card.setStyleSheet(
            f"background:{CARD_BG};border-radius:8px;"
            f"border:1px solid {BORDER_COLOR};")
        val_lay = QVBoxLayout(val_card)
        val_lay.setContentsMargins(20, 16, 20, 16)
        val_lay.setSpacing(10)
        val_lay.addWidget(_label("Validation", size=13, color=TEXT_PRIMARY))
        val_lay.addWidget(_label(
            "Disable prior injection for one run to validate raw detection accuracy.",
            size=12, color=TEXT_MUTED))
        self._disable_priors_cb = QCheckBox("Disable prior injection (validation mode)")
        self._disable_priors_cb.setChecked(self.state.disable_priors)
        self._disable_priors_cb.setStyleSheet(f"font-size:13px;color:{TEXT_PRIMARY};")
        self._disable_priors_cb.toggled.connect(
            lambda v: setattr(self.state, "disable_priors", v)
        )
        val_lay.addWidget(self._disable_priors_cb)
        val_lay.addWidget(_label(
            "Resets automatically to off after each analysis run.",
            size=11, color=TEXT_MUTED))
        root.addWidget(val_card)

        # ── Save button ──────────────────────────────────────────────────────
        save_row = QHBoxLayout()
        save_row.addStretch()
        save_btn = _button("Save Settings", primary=True)
        save_btn.clicked.connect(self._save)
        save_row.addWidget(save_btn)
        self._save_status = QLabel("")
        self._save_status.setStyleSheet(f"color:{TEXT_MUTED};font-size:12px;")
        save_row.addWidget(self._save_status)
        root.addLayout(save_row)

        root.addStretch()

        # Auto-test on open if no manual path set
        self._auto_test()

    def _auto_test(self):
        """Run auto-discovery silently on screen open to show current status."""
        try:
            path = resolve_redline_executable(self.state.redline_path)
            info = check_redline_available(path)
            if info["ready"]:
                self._rl_status.setStyleSheet(f"color:#34d399;font-size:12px;")
                build = info.get("build", "")
                sdk = info.get("sdk_version", "")
                self._rl_status.setText(
                    f"✓  Found: {path}"
                    + (f"  ·  Build {build}" if build else "")
                    + (f"  ·  SDK {sdk}" if sdk else ""))
            else:
                self._rl_status.setStyleSheet(f"color:{ACCENT};font-size:12px;")
                self._rl_status.setText(f"⚠  Found but not responding: {path}")
        except Exception as exc:
            self._rl_status.setStyleSheet(f"color:{ACCENT};font-size:12px;")
            self._rl_status.setText(f"✗  Not found — {str(exc).splitlines()[0]}")

    def _browse_redline(self):
        p, _ = QFileDialog.getOpenFileName(
            self, "Select REDLine Executable", "/Applications")
        if p:
            self._rl_path.setText(p)

    def _browse_out(self):
        p = QFileDialog.getExistingDirectory(self, "Select Default Output Directory")
        if p:
            self._out_path.setText(p)

    def _test_redline(self):
        path = self._rl_path.text().strip()
        self._rl_status.setStyleSheet(f"color:{TEXT_MUTED};font-size:12px;")
        self._rl_status.setText("Testing…")
        try:
            resolved = resolve_redline_executable(path)
            info = check_redline_available(resolved)
            if info["ready"]:
                self._rl_status.setStyleSheet(f"color:#34d399;font-size:12px;")
                build = info.get("build", "")
                sdk = info.get("sdk_version", "")
                self._rl_status.setText(
                    f"✓  {resolved}"
                    + (f"  ·  Build {build}" if build else "")
                    + (f"  ·  SDK {sdk}" if sdk else ""))
            else:
                self._rl_status.setStyleSheet(f"color:{ACCENT};font-size:12px;")
                self._rl_status.setText(
                    f"⚠  Found but not responding correctly: {info.get('error','')}")
        except Exception as exc:
            self._rl_status.setStyleSheet(f"color:{ACCENT};font-size:12px;")
            self._rl_status.setText(f"✗  {str(exc).splitlines()[0]}")

    def _save(self):
        self.state.redline_path    = self._rl_path.text().strip()
        self.state.default_out_dir = self._out_path.text().strip()
        save_settings({
            "redline_path":    self.state.redline_path,
            "default_out_dir": self.state.default_out_dir,
        })
        self._save_status.setStyleSheet(f"color:#34d399;font-size:12px;")
        self._save_status.setText("Saved.")


TABS = [
    ("Setup",          0),
    ("Progress",       1),
    ("Sphere QC",      2),
    ("Results",        3),
    ("Push",           4),
    ("Camera Network", 5),
    ("Settings",       6),
]


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} {APP_VERSION}")
        self.resize(1280, 860)
        self.setMinimumSize(960, 640)

        self.state = AppState()

        self._stack    = QStackedWidget()
        self._setup    = SetupScreen(self.state)
        self._progress = ProgressScreen(self.state)
        self._qc       = SphereQCScreen(self.state)
        self._results  = ResultsScreen(self.state)
        self._push     = PushScreen(self.state)
        self._network  = CameraNetworkScreen(self.state)
        self._settings = SettingsScreen(self.state)

        for s in (self._setup, self._progress, self._qc,
                  self._results, self._push, self._network, self._settings):
            self._stack.addWidget(s)

        self._setup.run_requested.connect(self._start_analysis)
        self._qc.assessment_ready.connect(self._on_assessment_ready)
        self._results.push_requested.connect(self._go_push)
        self._results.commit_requested.connect(self._commit_calibration)
        self._results.rerun_requested.connect(lambda: self._go(0))
        self._push.back_requested.connect(lambda: self._go(3))

        self._worker: Optional[AnalysisWorker] = None
        self._build_chrome()

    def _build_chrome(self):
        """Topbar + left workflow rail (paired with docs/ui_mockup_v3.html)."""
        outer = QWidget()
        ol = QVBoxLayout(outer)
        ol.setSpacing(0)
        ol.setContentsMargins(0, 0, 0, 0)

        # ── Topbar: status pills ──
        topbar = QWidget()
        topbar.setFixedHeight(44)
        topbar.setStyleSheet(
            f"background:{DARK_BG};border-bottom:1px solid {BORDER_COLOR};")
        tl = QHBoxLayout(topbar)
        tl.setContentsMargins(16, 0, 16, 0)
        tl.setSpacing(10)

        def _pill(text: str, dot_color: str = "") -> QLabel:
            dot = (f"<span style='color:{dot_color};font-size:9px'>●</span>&nbsp; "
                   if dot_color else "")
            p = QLabel(f"{dot}<span style='color:{TEXT_MUTED}'>{text}</span>")
            p.setStyleSheet(
                f"background:{PANEL_BG};border:1px solid {BORDER_COLOR};"
                f"border-radius:11px;padding:3px 12px;font-size:11px;")
            return p

        self._proj_pill = QLabel("")
        self._proj_pill.setStyleSheet(
            f"background:{PANEL_BG};border:1px solid {BORDER_COLOR};"
            f"border-radius:7px;padding:3px 12px;font-size:11px;"
            f"color:{TEXT_MUTED};font-family:{MONO_FONT};")
        self._proj_pill.setVisible(False)
        tl.addWidget(self._proj_pill)
        tl.addStretch()
        try:
            resolve_redline_executable(self.state.redline_path)
            tl.addWidget(_pill("REDLine", SUCCESS))
        except REDLineNotFoundError:
            tl.addWidget(_pill("REDLine not found", DANGER))
        n_ips = len([v for v in self.state.camera_ips.values() if v])
        tl.addWidget(_pill(
            f"{n_ips} camera IP{'s' if n_ips != 1 else ''} mapped",
            SUCCESS if n_ips else TEXT_DIM))
        ol.addWidget(topbar)

        container = QWidget()
        cl = QHBoxLayout(container)
        cl.setSpacing(0)
        cl.setContentsMargins(0, 0, 0, 0)

        rail = QWidget()
        rail.setFixedWidth(204)
        rail.setStyleSheet(
            f"background:{PANEL_BG};border-right:1px solid {BORDER_COLOR};")
        rl = QVBoxLayout(rail)
        rl.setContentsMargins(10, 14, 10, 14)
        rl.setSpacing(3)

        # ── Logo lockup: [R3D] Match v3.0 ──
        logo_row = QWidget()
        logo_lay = QHBoxLayout(logo_row)
        logo_lay.setContentsMargins(10, 0, 10, 12)
        logo_lay.setSpacing(8)
        mark = QLabel("R3D")
        mark.setFixedHeight(26)
        mark.setStyleSheet(
            f"background:qlineargradient(x1:0,y1:0,x2:1,y2:1,"
            f"stop:0 {ACCENT}, stop:1 {ACCENT_DEEP});"
            f"color:#ffffff;font-weight:900;font-size:12px;"
            f"border-radius:7px;padding:0 9px;letter-spacing:0.5px;")
        word = QLabel(
            f"<span style='color:{TEXT_PRIMARY};font-weight:800;font-size:15px'>Match</span>"
            f" <span style='color:{TEXT_DIM};font-weight:600;font-size:11px'>v{APP_VERSION}</span>")
        word.setStyleSheet("background:transparent;")
        logo_lay.addWidget(mark)
        logo_lay.addWidget(word)
        logo_lay.addStretch()
        rl.addWidget(logo_row)

        def _section(text: str) -> QLabel:
            s = QLabel(text.upper())
            s.setStyleSheet(
                f"color:{TEXT_DIM};font-size:9px;font-weight:800;"
                f"letter-spacing:2px;padding:10px 12px 4px;background:transparent;")
            return s

        def _nav_btn(num: str, name: str, idx: int) -> QPushButton:
            btn = QPushButton(f"  {num}   {name}")
            btn.setCheckable(True)
            btn.setFixedHeight(36)
            btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
            btn.setStyleSheet(
                f"QPushButton{{background:transparent;color:{TEXT_MUTED};"
                f"border:1px solid transparent;border-radius:9px;"
                f"padding:0 8px;font-size:13px;font-weight:600;text-align:left;}}"
                f"QPushButton:hover{{color:{TEXT_PRIMARY};background:{CARD_BG};}}"
                f"QPushButton:checked{{color:{TEXT_PRIMARY};background:{CARD_BG};"
                f"border:1px solid {BORDER_COLOR};border-left:3px solid {ACCENT};}}")
            btn.clicked.connect(lambda _, i=idx: self._go(i))
            return btn

        self._tab_btns: List[QPushButton] = []
        rl.addWidget(_section("Workflow"))
        workflow = [("1", "Ingest", 0), ("2", "Analyze", 1), ("3", "Assist", 2),
                    ("4", "Match", 3), ("5", "Push", 4)]
        for num, name, idx in workflow:
            btn = _nav_btn(num, name, idx)
            rl.addWidget(btn)
            self._tab_btns.append(btn)

        rl.addWidget(_section("System"))
        for num, name, idx in [("◈", "Cameras", 5), ("⚙", "Settings", 6)]:
            btn = _nav_btn(num, name, idx)
            rl.addWidget(btn)
            self._tab_btns.append(btn)

        rl.addStretch()
        foot = QLabel("R3DMatch v4 · KOMODO-X")
        foot.setStyleSheet(
            f"color:{TEXT_DIM};font-size:10px;font-family:{MONO_FONT};"
            f"padding:10px 12px 0;border-top:1px solid {BORDER_COLOR};background:transparent;")
        rl.addWidget(foot)

        cl.addWidget(rail)
        cl.addWidget(self._stack, 1)
        ol.addWidget(container, 1)
        self.setCentralWidget(outer)
        self._tab_btns[0].setChecked(True)

    def set_project(self, input_path: str):
        """Show the active project in the topbar."""
        name = Path(input_path).name or input_path
        self._proj_pill.setText(f"PROJ: {name}")
        self._proj_pill.setVisible(True)

    def _go(self, idx: int):
        self._stack.setCurrentIndex(idx)
        for i, btn in enumerate(self._tab_btns):
            btn.setChecked(i == idx)

    def _go_push(self):
        self._push.populate()
        self._go(4)

    def _commit_calibration(self):
        """Lock in the per-camera calibration from the Match screen and carry
        it forward to the Push screen."""
        rr = self.state.run_result
        n = sum(1 for cr in rr.cameras if cr.commit) if rr else 0
        if not n:
            QMessageBox.information(
                self, "Nothing to Commit",
                "No solved cameras with calibration values. Check Sphere QC and "
                "re-run the match.")
            return
        from datetime import datetime
        self.state.calibration_committed = True
        self.state.committed_at = datetime.now().isoformat(timespec="seconds")
        self._results.mark_committed()
        self._push.populate()
        self._go(4)

    def _start_analysis(self, ip: str, od: str, st: str):
        try:
            resolve_redline_executable(self.state.redline_path)
        except REDLineNotFoundError as exc:
            QMessageBox.critical(self, "REDLine Not Found", str(exc))
            return
        self.set_project(ip)
        self._progress.reset(ip)
        self._go(1)
        self._worker = AnalysisWorker(ip, od, st, disable_priors=self.state.disable_priors,
                                      wb_mode=getattr(self.state, "wb_mode", "match"))
        self._worker.progress.connect(self._progress.on_progress)
        self._worker.finished.connect(self._on_analysis_done)
        self._worker.errored.connect(self._on_analysis_error)
        self._worker.start()

    @Slot(object)
    def _on_analysis_done(self, run_result):
        self.state.run_result = run_result
        # Auto-reset disable_priors — validation mode is one-shot only
        if self.state.disable_priors:
            self.state.disable_priors = False
            if hasattr(self._settings, "_disable_priors_cb"):
                self._settings._disable_priors_cb.setChecked(False)
        self._progress.populate_metadata(run_result)
        self._qc.populate(run_result)
        self._go(2)   # → Sphere QC

    @Slot(object)
    def _on_assessment_ready(self, run_result):
        self._results.load_result(run_result, self.state.report_path)
        self._go(3)   # → Results

    @Slot(str)
    def _on_analysis_error(self, msg: str):
        self._progress.on_error(msg)
        QMessageBox.critical(self, "Analysis Failed",
                             f"Pipeline error:\n\n{msg[:500]}")


# ── Dark palette ──────────────────────────────────────────────────────────────

def _apply_dark_palette(app: QApplication):
    app.setStyle("Fusion")
    p = QPalette()
    p.setColor(QPalette.ColorRole.Window,          QColor(DARK_BG))
    p.setColor(QPalette.ColorRole.WindowText,      QColor(TEXT_PRIMARY))
    p.setColor(QPalette.ColorRole.Base,            QColor(PANEL_BG))
    p.setColor(QPalette.ColorRole.AlternateBase,   QColor(CARD_BG))
    p.setColor(QPalette.ColorRole.Text,            QColor(TEXT_PRIMARY))
    p.setColor(QPalette.ColorRole.Button,          QColor(CARD_BG))
    p.setColor(QPalette.ColorRole.ButtonText,      QColor(TEXT_PRIMARY))
    p.setColor(QPalette.ColorRole.Link,            QColor(ACCENT))
    p.setColor(QPalette.ColorRole.Highlight,       QColor(ACCENT))
    p.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
    p.setColor(QPalette.ColorGroup.Disabled,
               QPalette.ColorRole.Text, QColor(TEXT_MUTED))
    p.setColor(QPalette.ColorGroup.Disabled,
               QPalette.ColorRole.ButtonText, QColor(TEXT_MUTED))
    app.setPalette(p)


def main():
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setApplicationVersion(APP_VERSION)
    _apply_dark_palette(app)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
