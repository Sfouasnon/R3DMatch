from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Mapping, Optional, Sequence


CANCEL_FILE_ENV = "R3DMATCH_CANCEL_FILE"


class CancellationError(RuntimeError):
    """Raised when a run has been cancelled by the operator."""


def cancellation_file_from_env() -> Optional[Path]:
    raw = os.environ.get(CANCEL_FILE_ENV, "").strip()
    if not raw:
        return None
    return Path(raw).expanduser().resolve()


def cancellation_requested() -> bool:
    cancel_path = cancellation_file_from_env()
    return bool(cancel_path and cancel_path.exists())


def raise_if_cancelled(message: str = "Run cancelled by operator.") -> None:
    if cancellation_requested():
        raise CancellationError(message)


def terminate_process(process: subprocess.Popen[object], *, grace_period: float = 2.0) -> None:
    if process.poll() is not None:
        return
    try:
        process.terminate()
    except ProcessLookupError:
        return
    deadline = time.monotonic() + max(grace_period, 0.0)
    while process.poll() is None and time.monotonic() < deadline:
        time.sleep(0.05)
    if process.poll() is not None:
        return
    try:
        process.kill()
    except ProcessLookupError:
        return


def terminate_process_group(process: subprocess.Popen[object], *, grace_period: float = 2.0) -> None:
    if process.poll() is not None:
        return
    if os.name != "posix":
        terminate_process(process, grace_period=grace_period)
        return
    try:
        pgid = os.getpgid(process.pid)
    except ProcessLookupError:
        return
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + max(grace_period, 0.0)
    while process.poll() is None and time.monotonic() < deadline:
        time.sleep(0.05)
    if process.poll() is not None:
        return
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        return


def run_cancellable_subprocess(
    command: Sequence[str],
    *,
    cwd: Optional[str] = None,
    env: Optional[Mapping[str, str]] = None,
    text: bool = True,
) -> subprocess.CompletedProcess[str]:
    raise_if_cancelled()
    process = subprocess.Popen(
        list(command),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=text,
        cwd=cwd,
        env=dict(env) if env is not None else None,
    )
    try:
        while True:
            try:
                stdout, stderr = process.communicate(timeout=0.1)
                return subprocess.CompletedProcess(list(command), process.returncode or 0, stdout, stderr)
            except subprocess.TimeoutExpired:
                raise_if_cancelled()
    except CancellationError:
        terminate_process(process)
        stdout, stderr = process.communicate()
        raise CancellationError(
            f"Run cancelled while executing: {Path(str(command[0])).name}"
        ) from None
