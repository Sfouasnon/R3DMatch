"""
capture_ftp.py — R3DMatch v5 single-clip FTP ingest
====================================================
Pulls the clip that was just captured off each RED body over FTP-over-TLS and
lands it in a local destination. A trimmed, self-contained port of MediaRunner's
FTP core (ftp_connect / recursive dir download / .part staging + size verify),
scoped to the one thing the Capture tab needs: *ingest the newest clip we just
recorded* — because a live array can be up to 36 bodies, we never sweep whole
reels here.

RED on-camera media layout (from MediaRunner):
    /media/<REEL>.RDM/<CLIP>.RDC/<frames + sidecars>

Credentials are NEVER assumed. `ftp_connect` raises if user/password are blank —
the Capture tab keeps the pull step disabled until the operator fills them into
Settings.
"""
from __future__ import annotations

import hashlib
import logging
import os
import ssl
import sys
from dataclasses import dataclass
from ftplib import FTP_TLS, error_perm
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class FTPConfigError(ValueError):
    """Raised when FTP credentials are missing — never assume defaults."""


@dataclass
class ClipPullResult:
    label: str
    ip: str
    clip: str = ""
    remote_path: str = ""
    local_path: str = ""
    bytes: int = 0
    files: int = 0
    ok: bool = False
    verified: bool = False
    error: str = ""


# ── Connection ──────────────────────────────────────────────────────────────────
def ftp_connect(ip: str, user: str, password: str, port: int = 21,
                timeout: float = 15.0) -> FTP_TLS:
    """Open an FTP-over-TLS control connection and switch the data channel to
    protected (PROT P). Raises FTPConfigError if credentials are blank."""
    if not user or not password:
        raise FTPConfigError(
            "FTP username/password not configured — set them in Settings → Capture.")
    if sys.platform == "win32":
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        ftp = FTP_TLS(context=ctx)
    else:
        ftp = FTP_TLS()
    ftp.connect(ip, port=int(port), timeout=timeout)
    ftp.login(user, password)
    ftp.prot_p()
    ftp.set_pasv(True)
    return ftp


def _is_dir(ftp: FTP_TLS, path: str) -> bool:
    try:
        cur = ftp.pwd()
        ftp.cwd(path)
        ftp.cwd(cur)
        return True
    except Exception:
        return False


# ── Clip discovery ──────────────────────────────────────────────────────────────
def _clip_sort_key(name: str) -> tuple:
    """Sort .RDC clip folders so the newest capture ranks last. RED clips carry a
    zero-padded clip number (…_C063_…); fall back to the raw name."""
    base = os.path.basename(name)
    digits = "".join(ch for ch in base if ch.isdigit())
    return (int(digits) if digits else -1, base)


def find_captured_clip(ftp: FTP_TLS, clip_hint: str = "",
                       media_root: str = "/media") -> Optional[str]:
    """Return the remote path of the clip to ingest.

    If `clip_hint` (the CLIP_NAME reported by RCP2 at capture time) matches a
    .RDC folder, that wins — it is exactly the clip we just recorded. Otherwise
    fall back to the highest-numbered / newest .RDC across all reels.
    """
    hint_token = "".join(ch for ch in os.path.basename(clip_hint or "") if ch.isalnum()).upper()
    candidates: List[str] = []
    try:
        reels = [r for r in ftp.nlst(media_root) if r.upper().rstrip("/").endswith(".RDM")]
    except error_perm:
        reels = []
    for reel in reels:
        try:
            entries = ftp.nlst(reel)
        except error_perm:
            continue
        for entry in entries:
            if entry.upper().rstrip("/").endswith(".RDC"):
                candidates.append(entry)
                token = "".join(ch for ch in os.path.basename(entry) if ch.isalnum()).upper()
                if hint_token and hint_token in token:
                    return entry
    if not candidates:
        return None
    return sorted(candidates, key=_clip_sort_key)[-1]


# ── Download ────────────────────────────────────────────────────────────────────
def _download_file(ftp: FTP_TLS, remote: str, local: Path,
                   cancel: Optional[Callable[[], bool]] = None,
                   on_bytes: Optional[Callable[[int], None]] = None) -> int:
    """Download one file via a .part staging file, verify size, atomic-commit.
    Returns bytes written."""
    local.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    try:
        total = ftp.size(remote) or 0
    except Exception:
        total = 0
    part = local.with_suffix(local.suffix + ".part")
    written = [0]

    with part.open("wb") as f:
        def cb(data: bytes):
            if cancel and cancel():
                raise IOError(f"cancelled while downloading {remote}")
            f.write(data)
            written[0] += len(data)
            if on_bytes:
                on_bytes(len(data))
        ftp.retrbinary(f"RETR {remote}", cb)
        f.flush()
        os.fsync(f.fileno())

    if total and written[0] != total:
        part.unlink(missing_ok=True)
        raise IOError(f"size mismatch for {remote}: got {written[0]} of {total}")
    os.replace(part, local)
    return written[0]


def _download_dir(ftp: FTP_TLS, remote_dir: str, local_dir: Path,
                  cancel: Optional[Callable[[], bool]] = None,
                  on_bytes: Optional[Callable[[int], None]] = None) -> Tuple[int, int]:
    """Recursively download remote_dir → local_dir. Returns (files, bytes)."""
    local_dir.mkdir(parents=True, exist_ok=True)
    files = 0
    total_bytes = 0
    try:
        items = ftp.nlst(remote_dir)
    except error_perm:
        return files, total_bytes
    for item in items:
        name = os.path.basename(item.rstrip("/"))
        if name in (".", ""):
            continue
        target = local_dir / name
        if _is_dir(ftp, item):
            sub_f, sub_b = _download_dir(ftp, item, target, cancel=cancel, on_bytes=on_bytes)
            files += sub_f
            total_bytes += sub_b
        else:
            total_bytes += _download_file(ftp, item, target, cancel=cancel, on_bytes=on_bytes)
            files += 1
    return files, total_bytes


def _sha1_of_tree(root: Path) -> str:
    """A single rollup SHA-1 over every file in the pulled clip (name+content),
    so the local copy has a verifiable fingerprint even without a camera MHL."""
    h = hashlib.sha1()
    for p in sorted(root.rglob("*")):
        if p.is_file():
            h.update(p.relative_to(root).as_posix().encode())
            with p.open("rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    h.update(chunk)
    return h.hexdigest()


# ── Per-camera pull ─────────────────────────────────────────────────────────────
def pull_captured_clip(label: str, ip: str, dest_root: Path, user: str, password: str,
                       port: int = 21, clip_hint: str = "", verify: bool = True,
                       cancel: Optional[Callable[[], bool]] = None,
                       on_bytes: Optional[Callable[[int], None]] = None
                       ) -> ClipPullResult:
    """Connect to one body, locate the just-captured clip and download it to
    dest_root/<label>/<CLIP>.RDC. Never raises — errors land in result.error."""
    res = ClipPullResult(label=label, ip=ip)
    ftp = None
    try:
        ftp = ftp_connect(ip, user, password, port=port)
        remote = find_captured_clip(ftp, clip_hint=clip_hint)
        if not remote:
            res.error = "no .RDC clip found on camera media"
            return res
        clip_name = os.path.basename(remote.rstrip("/"))
        res.clip = clip_name
        res.remote_path = remote
        local_dir = Path(dest_root) / label / clip_name
        files, nbytes = _download_dir(ftp, remote, local_dir, cancel=cancel, on_bytes=on_bytes)
        res.local_path = str(local_dir)
        res.files = files
        res.bytes = nbytes
        res.ok = files > 0
        if res.ok and verify:
            try:
                digest = _sha1_of_tree(local_dir)
                (local_dir.parent / f"{clip_name}.sha1").write_text(
                    f"{digest}  {clip_name}\n", encoding="utf-8")
                res.verified = True
            except Exception as e:  # verification is best-effort
                res.error = f"downloaded ok; verify failed: {e}"
        return res
    except FTPConfigError as e:
        res.error = str(e)
        return res
    except Exception as e:
        res.error = str(e)
        return res
    finally:
        if ftp is not None:
            try:
                ftp.quit()
            except Exception:
                try:
                    ftp.close()
                except Exception:
                    pass


def pull_array(cameras: Dict[str, str], dest_root: Path, user: str, password: str,
               port: int = 21, clip_hints: Optional[Dict[str, str]] = None,
               verify: bool = True, max_workers: int = 8,
               cancel: Optional[Callable[[], bool]] = None,
               on_result: Optional[Callable[[ClipPullResult], None]] = None,
               on_bytes: Optional[Callable[[str, int], None]] = None
               ) -> List[ClipPullResult]:
    """Pull the captured clip from every camera concurrently. `cameras` maps
    label→ip; `clip_hints` maps label→CLIP_NAME from the capture. Validates
    credentials once up front so a misconfig fails fast for the whole array."""
    if not user or not password:
        raise FTPConfigError(
            "FTP username/password not configured — set them in Settings → Capture.")
    clip_hints = clip_hints or {}
    from concurrent.futures import ThreadPoolExecutor, as_completed

    results: List[ClipPullResult] = []

    def _one(item: Tuple[str, str]) -> ClipPullResult:
        label, ip = item
        return pull_captured_clip(
            label, ip, dest_root, user, password, port=port,
            clip_hint=clip_hints.get(label, ""), verify=verify, cancel=cancel,
            on_bytes=(lambda n, _l=label: on_bytes(_l, n)) if on_bytes else None)

    workers = max(1, min(max_workers, len(cameras)))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_one, item) for item in cameras.items()]
        for fut in as_completed(futs):
            r = fut.result()
            results.append(r)
            if on_result:
                on_result(r)
    results.sort(key=lambda r: r.label)
    return results
