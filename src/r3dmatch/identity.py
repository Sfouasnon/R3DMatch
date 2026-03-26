from __future__ import annotations

import re
from pathlib import Path


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
