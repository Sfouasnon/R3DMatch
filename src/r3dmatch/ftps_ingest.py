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
DEFAULT_FTPS_DISCOVERY_MAX_DEPTH = 4
INGEST_MANIFEST_FILENAME = "ingest_manifest.json"

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


def ingest_manifest_path_for(local_ingest_root: str | Path) -> Path:
    return Path(local_ingest_root).expanduser().resolve() / INGEST_MANIFEST_FILENAME


def load_ingest_manifest(path_or_root: str | Path) -> Dict[str, object]:
    candidate = Path(path_or_root).expanduser().resolve()
    manifest_path = candidate if candidate.name == INGEST_MANIFEST_FILENAME else ingest_manifest_path_for(candidate)
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _remote_file_size(ftp: ftplib.FTP_TLS, remote_path: str) -> Optional[int]:
    try:
        size = ftp.size(remote_path)
    except Exception:
        return None
    if size is None:
        return None
    try:
        return int(size)
    except (TypeError, ValueError):
        return None


def _discover_matches_for_camera(
    *,
    ftp_factory: Callable[[], ftplib.FTP_TLS],
    host: str,
    username: str,
    password: str,
    remote_root: str,
    reel_identifier: str,
    camera_label: str,
    clip_numbers: Sequence[int],
    retries: int,
    timeout_seconds: float,
) -> Dict[str, object]:
    camera_entry: Dict[str, object] = {
        "camera_label": camera_label,
        "host": host,
        "status": "failed",
        "failure_code": "",
        "attempts": 0,
        "reachable": False,
        "matched_files": [],
        "matched_file_count": 0,
        "estimated_bytes": 0,
        "downloaded_files": [],
        "downloaded_file_count": 0,
        "skipped_existing_files": [],
        "skipped_existing_count": 0,
        "failed_files": [],
        "failed_file_count": 0,
        "bytes_transferred": 0,
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
            camera_entry["reachable"] = True
            remote_matches = _discover_remote_matches(
                ftp,
                remote_root=remote_root,
                reel_identifier=reel_identifier,
                camera_label=camera_label,
                clip_numbers=clip_numbers,
                max_depth=DEFAULT_FTPS_DISCOVERY_MAX_DEPTH,
            )
            matched_files = []
            estimated_bytes = 0
            for remote_path in remote_matches:
                remote_bytes = _remote_file_size(ftp, remote_path)
                if remote_bytes is not None:
                    estimated_bytes += remote_bytes
                matched_files.append(
                    {
                        "remote_path": remote_path,
                        "remote_bytes": remote_bytes,
                    }
                )
            camera_entry["matched_files"] = matched_files
            camera_entry["matched_file_count"] = len(matched_files)
            camera_entry["estimated_bytes"] = estimated_bytes
            if matched_files:
                camera_entry["status"] = "matched"
                camera_entry["failure_code"] = ""
            else:
                camera_entry["status"] = "reachable_no_matches"
                camera_entry["failure_code"] = "no_matching_clips"
            break
        except (OSError, EOFError, ftplib.Error, socket.error) as exc:
            camera_entry["errors"].append(str(exc))
            if attempt > retries:
                camera_entry["status"] = "unreachable"
                camera_entry["failure_code"] = "connection_or_transfer_failed"
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
    return camera_entry


def _transfer_matches_for_camera(
    *,
    ftp_factory: Callable[[], ftplib.FTP_TLS],
    host: str,
    username: str,
    password: str,
    local_ingest_root: Path,
    remote_matches: Sequence[Dict[str, object]],
    camera_entry: Dict[str, object],
    retries: int,
    timeout_seconds: float,
    skip_existing: bool,
    overwrite: bool,
) -> Dict[str, object]:
    if not remote_matches:
        return camera_entry
    ftp = None
    try:
        ftp = ftp_factory()
        ftp.connect(host=host, port=DEFAULT_FTPS_PORT, timeout=timeout_seconds)
        ftp.login(user=username, passwd=password)
        if hasattr(ftp, "prot_p"):
            ftp.prot_p()
        for matched in remote_matches:
            remote_path = str(matched.get("remote_path") or "").strip()
            if not remote_path:
                continue
            relative_remote = remote_path.lstrip("/")
            local_path = local_ingest_root / camera_entry["camera_label"] / relative_remote
            existing = local_path.exists()
            if existing and skip_existing and not overwrite:
                bytes_existing = 0
                try:
                    bytes_existing = int(local_path.stat().st_size)
                except OSError:
                    bytes_existing = 0
                camera_entry["skipped_existing_files"].append(
                    {
                        "remote_path": remote_path,
                        "local_path": str(local_path),
                        "bytes": bytes_existing,
                    }
                )
                camera_entry["skipped_existing_count"] = len(camera_entry["skipped_existing_files"])
                continue
            file_errors: List[str] = []
            bytes_written = 0
            for attempt in range(1, retries + 2):
                try:
                    bytes_written = _download_remote_file(ftp, remote_path, local_path)
                    break
                except (OSError, EOFError, ftplib.Error, socket.error) as exc:
                    file_errors.append(str(exc))
                    if attempt > retries:
                        camera_entry["failed_files"].append(
                            {
                                "remote_path": remote_path,
                                "local_path": str(local_path),
                                "errors": file_errors,
                            }
                        )
            if bytes_written:
                camera_entry["bytes_transferred"] += bytes_written
                camera_entry["downloaded_files"].append(
                    {
                        "remote_path": remote_path,
                        "local_path": str(local_path),
                        "bytes": bytes_written,
                    }
                )
        camera_entry["downloaded_file_count"] = len(camera_entry["downloaded_files"])
        camera_entry["skipped_existing_count"] = len(camera_entry["skipped_existing_files"])
        camera_entry["failed_file_count"] = len(camera_entry["failed_files"])
        if camera_entry["failed_file_count"] and (
            camera_entry["downloaded_file_count"] or camera_entry["skipped_existing_count"]
        ):
            camera_entry["status"] = "partial"
            camera_entry["failure_code"] = "partial_transfer_failed"
        elif camera_entry["failed_file_count"]:
            camera_entry["status"] = "failed"
            camera_entry["failure_code"] = "transfer_failed"
        elif camera_entry["downloaded_file_count"] or camera_entry["skipped_existing_count"]:
            camera_entry["status"] = "downloaded"
            camera_entry["failure_code"] = ""
    finally:
        if ftp is not None:
            try:
                ftp.quit()
            except Exception:
                try:
                    ftp.close()
                except Exception:
                    pass
    return camera_entry


def _manifest_status_from_camera_results(camera_results: Sequence[Dict[str, object]], *, action: str) -> str:
    if not camera_results:
        return "failed"
    successes = 0
    failures = 0
    for entry in camera_results:
        status = str(entry.get("status") or "")
        if action == "discover":
            if status == "matched":
                successes += 1
            elif status in {"unreachable"}:
                failures += 1
        else:
            if status in {"downloaded"}:
                successes += 1
            elif status in {"partial", "failed", "unreachable", "reachable_no_matches"}:
                failures += 1
    if failures and successes:
        return "partial"
    if failures and not successes:
        return "failed"
    return "success"


def _summarize_camera_results(
    *,
    plan: Dict[str, object],
    local_ingest_root: Path,
    camera_results: Sequence[Dict[str, object]],
    action: str,
    processing_requested_after_ingest: bool = False,
    prior_manifest_path: Optional[str] = None,
) -> Dict[str, object]:
    reachable_cameras = [str(entry["camera_label"]) for entry in camera_results if bool(entry.get("reachable"))]
    unreachable_cameras = [str(entry["camera_label"]) for entry in camera_results if not bool(entry.get("reachable"))]
    cameras_with_matches = [str(entry["camera_label"]) for entry in camera_results if int(entry.get("matched_file_count", 0) or 0) > 0]
    cameras_without_matches = [
        str(entry["camera_label"])
        for entry in camera_results
        if bool(entry.get("reachable")) and int(entry.get("matched_file_count", 0) or 0) == 0
    ]
    downloaded_files = [
        item
        for entry in camera_results
        for item in (entry.get("downloaded_files") or [])
        if isinstance(item, dict)
    ]
    skipped_existing_files = [
        item
        for entry in camera_results
        for item in (entry.get("skipped_existing_files") or [])
        if isinstance(item, dict)
    ]
    failed_files = [
        item
        for entry in camera_results
        for item in (entry.get("failed_files") or [])
        if isinstance(item, dict)
    ]
    matched_files = [
        item
        for entry in camera_results
        for item in (entry.get("matched_files") or [])
        if isinstance(item, dict)
    ]
    estimated_bytes = sum(int(entry.get("estimated_bytes", 0) or 0) for entry in camera_results)
    bytes_transferred = sum(int(entry.get("bytes_transferred", 0) or 0) for entry in camera_results)
    manifest: Dict[str, object] = {
        "schema_version": "r3dmatch_ftps_ingest_v2",
        "action": action,
        "source_mode": "ftps_camera_pull",
        "source_mode_label": source_mode_label("ftps_camera_pull"),
        "local_ingest_root": str(local_ingest_root),
        "reel_identifier": plan["reel_identifier"],
        "clips_requested": plan["clip_numbers"],
        "clip_spec": plan["clip_spec"],
        "requested_cameras": plan["requested_cameras"],
        "requested_camera_ips": {camera: plan["camera_ips"][camera] for camera in plan["requested_cameras"]},
        "requested_camera_count": len(plan["requested_cameras"]),
        "reachable_cameras": reachable_cameras,
        "unreachable_cameras": unreachable_cameras,
        "reachable_camera_count": len(reachable_cameras),
        "unreachable_camera_count": len(unreachable_cameras),
        "cameras_with_matches": cameras_with_matches,
        "cameras_without_matches": cameras_without_matches,
        "matched_files": matched_files,
        "matched_file_count": len(matched_files),
        "estimated_bytes": estimated_bytes,
        "downloaded_files": downloaded_files,
        "downloaded_file_count": len(downloaded_files),
        "skipped_existing_files": skipped_existing_files,
        "skipped_existing_count": len(skipped_existing_files),
        "failed_files": failed_files,
        "failed_file_count": len(failed_files),
        "bytes_transferred": bytes_transferred,
        "processing_requested_after_ingest": processing_requested_after_ingest,
        "prior_manifest_path": prior_manifest_path,
        "per_camera_status": list(camera_results),
    }
    manifest["successful_cameras"] = [
        str(entry["camera_label"])
        for entry in camera_results
        if str(entry.get("status") or "") in {"matched", "downloaded"}
    ]
    manifest["failed_cameras"] = [
        str(entry["camera_label"])
        for entry in camera_results
        if str(entry.get("status") or "") in {"unreachable", "failed", "partial", "reachable_no_matches"}
    ]
    manifest["successful_camera_count"] = len(manifest["successful_cameras"])
    manifest["failed_camera_count"] = len(manifest["failed_cameras"])
    manifest["clips_found"] = len(matched_files)
    manifest["bytes_pulled"] = bytes_transferred
    manifest["status"] = _manifest_status_from_camera_results(camera_results, action=action)
    return manifest


def discover_ftps_batch(
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
    processing_requested_after_ingest: bool = False,
    prior_manifest_path: Optional[str] = None,
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
    ftp_factory = ftp_factory or ftplib.FTP_TLS
    camera_results = [
        _discover_matches_for_camera(
            ftp_factory=ftp_factory,
            host=resolved_map[str(camera_label)],
            username=username,
            password=password,
            remote_root=remote_root,
            reel_identifier=str(plan["reel_identifier"]),
            camera_label=str(camera_label),
            clip_numbers=plan["clip_numbers"],
            retries=retries,
            timeout_seconds=timeout_seconds,
        )
        for camera_label in plan["requested_cameras"]
    ]
    manifest = _summarize_camera_results(
        plan=plan,
        local_ingest_root=ingest_root,
        camera_results=camera_results,
        action="discover",
        processing_requested_after_ingest=processing_requested_after_ingest,
        prior_manifest_path=prior_manifest_path,
    )
    manifest["started_at"] = started_at
    manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
    manifest_path = ingest_manifest_path_for(ingest_root)
    manifest["manifest_path"] = str(manifest_path)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def download_ftps_batch(
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
    skip_existing: bool = True,
    overwrite: bool = False,
    prior_manifest: Optional[Dict[str, object]] = None,
    processing_requested_after_ingest: bool = False,
    prior_manifest_path: Optional[str] = None,
) -> Dict[str, object]:
    resolved_map = camera_map or DEFAULT_CAMERA_IP_MAP
    ftp_factory = ftp_factory or ftplib.FTP_TLS
    discovery_manifest = prior_manifest or discover_ftps_batch(
        out_dir=out_dir,
        reel_identifier=reel_identifier,
        clip_spec=clip_spec,
        requested_cameras=requested_cameras,
        username=username,
        password=password,
        camera_map=resolved_map,
        remote_root=remote_root,
        retries=retries,
        timeout_seconds=timeout_seconds,
        ftp_factory=ftp_factory,
        processing_requested_after_ingest=processing_requested_after_ingest,
        prior_manifest_path=prior_manifest_path,
    )
    ingest_root = Path(out_dir).expanduser().resolve()
    camera_results = []
    for entry in discovery_manifest.get("per_camera_status", []):
        camera_entry = dict(entry)
        if int(camera_entry.get("matched_file_count", 0) or 0) > 0:
            camera_entry = _transfer_matches_for_camera(
                ftp_factory=ftp_factory,
                host=str(camera_entry.get("host") or ""),
                username=username,
                password=password,
                local_ingest_root=ingest_root,
                remote_matches=list(camera_entry.get("matched_files") or []),
                camera_entry=camera_entry,
                retries=retries,
                timeout_seconds=timeout_seconds,
                skip_existing=skip_existing,
                overwrite=overwrite,
            )
        camera_results.append(camera_entry)
    manifest = _summarize_camera_results(
        plan={
            "reel_identifier": discovery_manifest.get("reel_identifier"),
            "clip_numbers": discovery_manifest.get("clips_requested"),
            "clip_spec": discovery_manifest.get("clip_spec"),
            "requested_cameras": discovery_manifest.get("requested_cameras"),
            "camera_ips": discovery_manifest.get("requested_camera_ips"),
        },
        local_ingest_root=ingest_root,
        camera_results=camera_results,
        action="download",
        processing_requested_after_ingest=processing_requested_after_ingest,
        prior_manifest_path=prior_manifest_path or str(discovery_manifest.get("manifest_path") or ""),
    )
    manifest["started_at"] = str(discovery_manifest.get("started_at") or datetime.now(timezone.utc).isoformat())
    manifest["discovery_manifest_path"] = str(discovery_manifest.get("manifest_path") or "")
    manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
    manifest_path = ingest_manifest_path_for(ingest_root)
    manifest["manifest_path"] = str(manifest_path)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def retry_failed_ftps_batch(
    *,
    out_dir: str,
    username: str = DEFAULT_FTPS_USERNAME,
    password: str = DEFAULT_FTPS_PASSWORD,
    camera_map: Optional[Dict[str, str]] = None,
    remote_root: str = DEFAULT_FTPS_REMOTE_ROOT,
    retries: int = DEFAULT_FTPS_RETRIES,
    timeout_seconds: float = 8.0,
    ftp_factory: Optional[Callable[[], ftplib.FTP_TLS]] = None,
    skip_existing: bool = True,
    overwrite: bool = False,
    manifest_path: Optional[str] = None,
) -> Dict[str, object]:
    previous_manifest = load_ingest_manifest(manifest_path or out_dir)
    requested_cameras = list(previous_manifest.get("failed_cameras") or previous_manifest.get("unreachable_cameras") or [])
    if not requested_cameras:
        requested_cameras = [
            str(entry.get("camera_label") or "")
            for entry in previous_manifest.get("per_camera_status", [])
            if str(entry.get("status") or "") not in {"downloaded", "matched"}
        ]
    requested_cameras = [item for item in requested_cameras if item]
    if not requested_cameras:
        return previous_manifest
    return download_ftps_batch(
        out_dir=out_dir,
        reel_identifier=str(previous_manifest.get("reel_identifier") or ""),
        clip_spec=str(previous_manifest.get("clip_spec") or ""),
        requested_cameras=requested_cameras,
        username=username,
        password=password,
        camera_map=camera_map,
        remote_root=remote_root,
        retries=retries,
        timeout_seconds=timeout_seconds,
        ftp_factory=ftp_factory,
        skip_existing=skip_existing,
        overwrite=overwrite,
        prior_manifest_path=str(previous_manifest.get("manifest_path") or manifest_path or ""),
    )


def run_ftps_ingest_job(
    *,
    action: str,
    out_dir: str,
    reel_identifier: Optional[str] = None,
    clip_spec: Optional[str] = None,
    requested_cameras: Optional[Sequence[str]] = None,
    username: str = DEFAULT_FTPS_USERNAME,
    password: str = DEFAULT_FTPS_PASSWORD,
    camera_map: Optional[Dict[str, str]] = None,
    remote_root: str = DEFAULT_FTPS_REMOTE_ROOT,
    retries: int = DEFAULT_FTPS_RETRIES,
    timeout_seconds: float = 8.0,
    ftp_factory: Optional[Callable[[], ftplib.FTP_TLS]] = None,
    skip_existing: bool = True,
    overwrite: bool = False,
    manifest_path: Optional[str] = None,
    processing_requested_after_ingest: bool = False,
) -> Dict[str, object]:
    normalized_action = str(action or "download").strip().lower().replace("_", "-")
    if normalized_action == "discover":
        if not reel_identifier or not clip_spec:
            raise ValueError("FTPS discovery requires reel identifier and clip spec.")
        return discover_ftps_batch(
            out_dir=out_dir,
            reel_identifier=reel_identifier,
            clip_spec=clip_spec,
            requested_cameras=requested_cameras,
            username=username,
            password=password,
            camera_map=camera_map,
            remote_root=remote_root,
            retries=retries,
            timeout_seconds=timeout_seconds,
            ftp_factory=ftp_factory,
            processing_requested_after_ingest=processing_requested_after_ingest,
            prior_manifest_path=manifest_path,
        )
    if normalized_action == "download":
        if not reel_identifier or not clip_spec:
            raise ValueError("FTPS download requires reel identifier and clip spec.")
        return download_ftps_batch(
            out_dir=out_dir,
            reel_identifier=reel_identifier,
            clip_spec=clip_spec,
            requested_cameras=requested_cameras,
            username=username,
            password=password,
            camera_map=camera_map,
            remote_root=remote_root,
            retries=retries,
            timeout_seconds=timeout_seconds,
            ftp_factory=ftp_factory,
            skip_existing=skip_existing,
            overwrite=overwrite,
            processing_requested_after_ingest=processing_requested_after_ingest,
            prior_manifest_path=manifest_path,
        )
    if normalized_action == "retry-failed":
        return retry_failed_ftps_batch(
            out_dir=out_dir,
            username=username,
            password=password,
            camera_map=camera_map,
            remote_root=remote_root,
            retries=retries,
            timeout_seconds=timeout_seconds,
            ftp_factory=ftp_factory,
            skip_existing=skip_existing,
            overwrite=overwrite,
            manifest_path=manifest_path,
        )
    raise ValueError("FTPS action must be discover, download, or retry-failed.")


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
    return download_ftps_batch(
        out_dir=out_dir,
        reel_identifier=reel_identifier,
        clip_spec=clip_spec,
        requested_cameras=requested_cameras,
        username=username,
        password=password,
        camera_map=camera_map,
        remote_root=remote_root,
        retries=retries,
        timeout_seconds=timeout_seconds,
        ftp_factory=ftp_factory,
    )
