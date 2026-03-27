from __future__ import annotations

import ftplib
import json
import os
import re
import socket
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_FTPS_USERNAME = "ftp1"
DEFAULT_FTPS_PASSWORD = "12345678"
DEFAULT_FTPS_PORT = 21
DEFAULT_FTPS_REMOTE_ROOT = "/"
DEFAULT_FTPS_RETRIES = 2

DEFAULT_CAMERA_IP_MAP: Dict[str, str] = {
    "AA": "172.20.114.141",
    "AB": "172.20.114.142",
    "AC": "172.20.114.143",
    "AD": "172.20.114.144",
    "BA": "172.20.114.145",
    "BB": "172.20.114.146",
    "BC": "172.20.114.147",
    "BD": "172.20.114.148",
    "CA": "172.20.114.149",
    "CB": "172.20.114.150",
    "CC": "172.20.114.151",
    "CD": "172.20.114.152",
    "DA": "172.20.114.153",
    "DB": "172.20.114.154",
    "DC": "172.20.114.155",
    "DD": "172.20.114.156",
    "EA": "172.20.114.157",
    "EB": "172.20.114.158",
    "EC": "172.20.114.159",
    "ED": "172.20.114.160",
    "FA": "172.20.114.161",
    "FB": "172.20.114.162",
    "FC": "172.20.114.163",
    "FD": "172.20.114.164",
    "GA": "172.20.114.165",
    "GB": "172.20.114.166",
    "GC": "172.20.114.167",
    "GD": "172.20.114.168",
    "HA": "172.20.114.169",
    "HB": "172.20.114.170",
    "HC": "172.20.114.171",
    "HD": "172.20.114.172",
    "IA": "172.20.114.173",
    "IB": "172.20.114.174",
    "IC": "172.20.114.175",
    "ID": "172.20.114.176",
    "JA": "172.20.114.177",
    "JB": "172.20.114.178",
    "JC": "172.20.114.179",
    "JD": "172.20.114.180",
    "KA": "172.20.114.181",
    "KB": "172.20.114.182",
}

SOURCE_MODE_LABELS = {
    "local_folder": "Local Folder",
    "ftps_camera_pull": "FTPS Camera Pull",
}


def normalize_source_mode(value: str) -> str:
    normalized = str(value).strip().lower()
    aliases = {
        "local": "local_folder",
        "local_folder": "local_folder",
        "local-folder": "local_folder",
        "folder": "local_folder",
        "ftps": "ftps_camera_pull",
        "ftps_camera_pull": "ftps_camera_pull",
        "ftps-camera-pull": "ftps_camera_pull",
        "camera_pull": "ftps_camera_pull",
    }
    if normalized not in aliases:
        raise ValueError("source mode must be local_folder or ftps_camera_pull")
    return aliases[normalized]


def source_mode_label(value: str) -> str:
    return SOURCE_MODE_LABELS[normalize_source_mode(value)]


def normalize_reel_identifier(value: str) -> str:
    digits = re.sub(r"\D+", "", str(value))
    if not digits:
        raise ValueError("FTPS reel identifier must contain at least one digit.")
    return digits.zfill(3)


def parse_clip_spec(clip_spec: str) -> List[int]:
    text = str(clip_spec or "").strip()
    if not text:
        raise ValueError("FTPS clip numbers are required.")
    values = set()
    for chunk in [item.strip() for item in text.split(",") if item.strip()]:
        if "-" in chunk:
            left, right = [item.strip() for item in chunk.split("-", 1)]
            if not left.isdigit() or not right.isdigit():
                raise ValueError(f"Invalid clip range: {chunk}")
            start = int(left)
            end = int(right)
            if end < start:
                raise ValueError(f"Invalid descending clip range: {chunk}")
            values.update(range(start, end + 1))
        else:
            if not chunk.isdigit():
                raise ValueError(f"Invalid clip number: {chunk}")
            values.add(int(chunk))
    return sorted(values)


def format_clip_spec(clip_numbers: Sequence[int]) -> str:
    if not clip_numbers:
        return ""
    ordered = sorted({int(value) for value in clip_numbers})
    ranges: List[str] = []
    start = ordered[0]
    end = ordered[0]
    for value in ordered[1:]:
        if value == end + 1:
            end = value
            continue
        ranges.append(f"{start}-{end}" if start != end else str(start))
        start = end = value
    ranges.append(f"{start}-{end}" if start != end else str(start))
    return ",".join(ranges)


def normalize_camera_subset(
    requested: Optional[Sequence[str]],
    *,
    camera_map: Optional[Dict[str, str]] = None,
) -> List[str]:
    resolved_map = camera_map or DEFAULT_CAMERA_IP_MAP
    if not requested:
        return sorted(resolved_map)
    selected = []
    invalid = []
    for item in requested:
        token = str(item).strip().upper()
        if not token:
            continue
        if token not in resolved_map:
            invalid.append(token)
        elif token not in selected:
            selected.append(token)
    if invalid:
        raise ValueError(f"Unknown FTPS camera label(s): {', '.join(invalid)}")
    return selected


def plan_ftps_request(
    *,
    reel_identifier: str,
    clip_spec: str,
    requested_cameras: Optional[Sequence[str]] = None,
    camera_map: Optional[Dict[str, str]] = None,
) -> Dict[str, object]:
    resolved_map = camera_map or DEFAULT_CAMERA_IP_MAP
    resolved_reel = normalize_reel_identifier(reel_identifier)
    clip_numbers = parse_clip_spec(clip_spec)
    cameras = normalize_camera_subset(requested_cameras, camera_map=resolved_map)
    return {
        "source_mode": "ftps_camera_pull",
        "source_mode_label": source_mode_label("ftps_camera_pull"),
        "reel_identifier": resolved_reel,
        "clip_numbers": clip_numbers,
        "clip_spec": format_clip_spec(clip_numbers),
        "requested_cameras": cameras,
        "camera_count": len(cameras),
        "camera_ips": {camera: resolved_map[camera] for camera in cameras},
    }


def _clip_filename_matches(name: str, *, reel_identifier: str, camera_label: str, clip_numbers: Sequence[int]) -> bool:
    if not name.lower().endswith(".r3d"):
        return False
    reel_token = normalize_reel_identifier(reel_identifier)
    camera_token = re.escape(camera_label)
    clip_pattern = "|".join(f"{int(number):02d}" for number in clip_numbers)
    pattern = re.compile(rf"G0*{reel_token}_{camera_token}(?:0)?(?:{clip_pattern})_", re.IGNORECASE)
    return bool(pattern.search(name))


def _list_remote_entries(ftp: ftplib.FTP_TLS, remote_dir: str) -> List[Tuple[str, str]]:
    try:
        entries = list(ftp.mlsd(remote_dir))
        return [(name, facts.get("type", "file")) for name, facts in entries]
    except Exception:
        pass
    original = ftp.pwd()
    ftp.cwd(remote_dir)
    try:
        names = ftp.nlst()
    finally:
        ftp.cwd(original)
    entries: List[Tuple[str, str]] = []
    for name in names:
        child = _join_remote(remote_dir, name)
        entry_type = "file"
        try:
            ftp.cwd(child)
        except Exception:
            entry_type = "file"
        else:
            entry_type = "dir"
            ftp.cwd(original)
        entries.append((Path(name).name, entry_type))
    return entries


def _join_remote(base: str, name: str) -> str:
    if base in {"", "/"}:
        return f"/{name.lstrip('/')}"
    return f"{base.rstrip('/')}/{name.lstrip('/')}"


def _discover_remote_matches(
    ftp: ftplib.FTP_TLS,
    *,
    remote_root: str,
    reel_identifier: str,
    camera_label: str,
    clip_numbers: Sequence[int],
    max_depth: int = 4,
) -> List[str]:
    matches: List[str] = []
    queue: List[Tuple[str, int]] = [(remote_root or "/", 0)]
    seen: set[str] = set()
    while queue:
        current, depth = queue.pop(0)
        if current in seen:
            continue
        seen.add(current)
        try:
            entries = _list_remote_entries(ftp, current)
        except Exception:
            continue
        for name, entry_type in entries:
            remote_path = _join_remote(current, name)
            if entry_type == "dir":
                if depth < max_depth:
                    queue.append((remote_path, depth + 1))
                continue
            if _clip_filename_matches(name, reel_identifier=reel_identifier, camera_label=camera_label, clip_numbers=clip_numbers):
                matches.append(remote_path)
    return sorted(set(matches))


def _download_remote_file(ftp: ftplib.FTP_TLS, remote_path: str, local_path: Path) -> int:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    bytes_written = 0
    with local_path.open("wb") as handle:
        def _write(chunk: bytes) -> None:
            nonlocal bytes_written
            handle.write(chunk)
            bytes_written += len(chunk)

        ftp.retrbinary(f"RETR {remote_path}", _write)
    return bytes_written


def ingest_ftps_batch(
    *,
    out_dir: str,
    reel_identifier: str,
    clip_spec: str,
    requested_cameras: Optional[Sequence[str]] = None,
    username: str = DEFAULT_FTPS_USERNAME,
    password: str = DEFAULT_FTPS_PASSWORD,
    camera_map: Optional[Dict[str, str]] = None,
    remote_root: str = DEFAULT_FTPS_REMOTE_ROOT,
    retries: int = DEFAULT_FTPS_RETRIES,
    timeout_seconds: float = 8.0,
    ftp_factory: Optional[Callable[[], ftplib.FTP_TLS]] = None,
) -> Dict[str, object]:
    resolved_map = camera_map or DEFAULT_CAMERA_IP_MAP
    plan = plan_ftps_request(
        reel_identifier=reel_identifier,
        clip_spec=clip_spec,
        requested_cameras=requested_cameras,
        camera_map=resolved_map,
    )
    ingest_root = Path(out_dir).expanduser().resolve()
    ingest_root.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc).isoformat()
    total_bytes = 0
    successful_cameras: List[str] = []
    failed_cameras: List[str] = []
    clips_found = 0
    camera_results: List[Dict[str, object]] = []
    ftp_factory = ftp_factory or ftplib.FTP_TLS

    for camera_label in plan["requested_cameras"]:
        host = resolved_map[camera_label]
        camera_entry: Dict[str, object] = {
            "camera_label": camera_label,
            "host": host,
            "status": "failed",
            "attempts": 0,
            "clips_found": 0,
            "bytes_pulled": 0,
            "downloaded_files": [],
            "errors": [],
        }
        for attempt in range(1, retries + 2):
            camera_entry["attempts"] = attempt
            ftp = None
            try:
                ftp = ftp_factory()
                ftp.connect(host=host, port=DEFAULT_FTPS_PORT, timeout=timeout_seconds)
                ftp.login(user=username, passwd=password)
                if hasattr(ftp, "prot_p"):
                    ftp.prot_p()
                remote_matches = _discover_remote_matches(
                    ftp,
                    remote_root=remote_root,
                    reel_identifier=str(plan["reel_identifier"]),
                    camera_label=str(camera_label),
                    clip_numbers=plan["clip_numbers"],
                )
                if not remote_matches:
                    raise FileNotFoundError(
                        f"No matching R3D files found for reel {plan['reel_identifier']} camera {camera_label} clip(s) {plan['clip_spec']}."
                    )
                for remote_path in remote_matches:
                    relative_remote = remote_path.lstrip("/")
                    local_path = ingest_root / camera_label / relative_remote
                    bytes_written = _download_remote_file(ftp, remote_path, local_path)
                    total_bytes += bytes_written
                    camera_entry["bytes_pulled"] += bytes_written
                    camera_entry["downloaded_files"].append(
                        {
                            "remote_path": remote_path,
                            "local_path": str(local_path),
                            "bytes": bytes_written,
                        }
                    )
                camera_entry["clips_found"] = len(remote_matches)
                camera_entry["status"] = "success"
                successful_cameras.append(camera_label)
                clips_found += len(remote_matches)
                break
            except (OSError, EOFError, ftplib.Error, socket.error) as exc:
                camera_entry["errors"].append(str(exc))
                if attempt > retries:
                    failed_cameras.append(camera_label)
                continue
            finally:
                if ftp is not None:
                    try:
                        ftp.quit()
                    except Exception:
                        try:
                            ftp.close()
                        except Exception:
                            pass
        camera_results.append(camera_entry)

    manifest = {
        "schema_version": "r3dmatch_ftps_ingest_v1",
        "source_mode": "ftps_camera_pull",
        "source_mode_label": source_mode_label("ftps_camera_pull"),
        "started_at": started_at,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "local_ingest_root": str(ingest_root),
        "reel_identifier": plan["reel_identifier"],
        "clips_requested": plan["clip_numbers"],
        "clip_spec": plan["clip_spec"],
        "requested_cameras": plan["requested_cameras"],
        "successful_cameras": successful_cameras,
        "failed_cameras": failed_cameras,
        "requested_camera_count": len(plan["requested_cameras"]),
        "successful_camera_count": len(successful_cameras),
        "failed_camera_count": len(failed_cameras),
        "clips_found": clips_found,
        "bytes_pulled": total_bytes,
        "per_camera_status": camera_results,
    }
    manifest_path = ingest_root / "ingest_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest["manifest_path"] = str(manifest_path)
    manifest["status"] = "success" if not failed_cameras else ("partial" if successful_cameras else "failed")
    return manifest

