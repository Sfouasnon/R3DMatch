from __future__ import annotations

import re
from pathlib import Path
from typing import Optional


def clip_id_from_path(source_path: str) -> str:
    return Path(source_path).stem


def original_filename_from_path(source_path: str) -> str:
    return Path(source_path).name


def legacy_camera_group_from_clip_id(clip_id: str) -> str:
    return clip_id.split("_", 1)[0] if "_" in clip_id else clip_id


def group_key_from_clip_id(clip_id: str) -> str:
    tokens = clip_id.split("_")
    return "_".join(tokens[:2]) if len(tokens) >= 2 else clip_id


def subset_key_from_clip_id(clip_id: str) -> str:
    tokens = clip_id.split("_")
    if len(tokens) >= 2:
        match = re.search(r"(\d+)$", tokens[1])
        if match:
            return match.group(1)
    return tokens[-1] if tokens else clip_id


def rmd_name_for_clip_id(clip_id: str) -> str:
    return f"{clip_id}.RMD"


def inventory_camera_label_from_clip_id(clip_id: str) -> Optional[str]:
    tokens = str(clip_id or "").split("_")
    if len(tokens) < 2:
        return None
    first = re.sub(r"[^A-Za-z]+", "", tokens[0])
    second = re.sub(r"[^A-Za-z]+", "", tokens[1])
    if not first or not second:
        return None
    return f"{first[0]}{second[0]}".upper()


def inventory_camera_label_from_source_path(source_path: str) -> Optional[str]:
    path = Path(source_path)
    for parent in [path.parent, *path.parents]:
        name = parent.name.strip().upper()
        if re.fullmatch(r"[A-Z]{2}", name):
            return name
    return inventory_camera_label_from_clip_id(path.stem)
