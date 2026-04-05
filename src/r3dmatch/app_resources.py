from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional


def is_frozen_app() -> bool:
    return bool(getattr(sys, "frozen", False))


def bundle_resource_root() -> Optional[Path]:
    if not is_frozen_app():
        return None
    executable = Path(sys.executable).resolve()
    if executable.parent.name == "MacOS" and executable.parent.parent.name == "Contents":
        return executable.parent.parent / "Resources"
    meipass = str(getattr(sys, "_MEIPASS", "") or "").strip()
    if meipass:
        return Path(meipass).resolve()
    return executable.parent


def bundled_red_redistributable_dir() -> Optional[Path]:
    explicit = str(os.environ.get("R3DMATCH_BUNDLED_RED_REDISTRIBUTABLE_DIR") or "").strip()
    if explicit:
        path = Path(explicit).expanduser()
        if path.exists():
            return path.resolve()
    candidates = []
    root = bundle_resource_root()
    if root is not None:
        candidates.append(root / "red_runtime" / "redistributable")
    executable = Path(sys.executable).resolve()
    if executable.parent.name == "MacOS" and executable.parent.parent.name == "Contents":
        candidates.append(executable.parent.parent / "Frameworks" / "red_runtime" / "redistributable")
    meipass = str(getattr(sys, "_MEIPASS", "") or "").strip()
    if meipass:
        candidates.append(Path(meipass).resolve() / "red_runtime" / "redistributable")
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None
