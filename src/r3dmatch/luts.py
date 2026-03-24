from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np


@dataclass
class CubeLut:
    path: str
    title: Optional[str]
    size_1d: Optional[int]
    size_3d: Optional[int]
    domain_min: List[float]
    domain_max: List[float]
    lut_1d: Optional[np.ndarray]
    lut_3d: Optional[np.ndarray]


def load_cube_lut(path: str) -> CubeLut:
    lut_path = Path(path).expanduser().resolve()
    title: Optional[str] = None
    size_1d: Optional[int] = None
    size_3d: Optional[int] = None
    domain_min = [0.0, 0.0, 0.0]
    domain_max = [1.0, 1.0, 1.0]
    rows_1d: list[list[float]] = []
    rows_3d: list[list[float]] = []

    for raw_line in lut_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("TITLE"):
            title = line.split('"', 2)[1] if '"' in line else line.partition(" ")[2].strip()
            continue
        if line.startswith("LUT_1D_SIZE"):
            size_1d = int(line.split()[1])
            continue
        if line.startswith("LUT_3D_SIZE"):
            size_3d = int(line.split()[1])
            continue
        if line.startswith("DOMAIN_MIN"):
            domain_min = [float(item) for item in line.split()[1:4]]
            continue
        if line.startswith("DOMAIN_MAX"):
            domain_max = [float(item) for item in line.split()[1:4]]
            continue
        sample = [float(item) for item in line.split()]
        if len(sample) != 3:
            raise ValueError(f"Unsupported LUT row in {lut_path}: {line}")
        if size_3d is not None:
            rows_3d.append(sample)
        elif size_1d is not None:
            rows_1d.append(sample)
        else:
            raise ValueError(f"Encountered sample before LUT size declaration in {lut_path}")

    lut_1d = np.asarray(rows_1d, dtype=np.float32) if rows_1d else None
    lut_3d = np.asarray(rows_3d, dtype=np.float32).reshape(size_3d, size_3d, size_3d, 3) if rows_3d and size_3d else None
    return CubeLut(
        path=str(lut_path),
        title=title,
        size_1d=size_1d,
        size_3d=size_3d,
        domain_min=domain_min,
        domain_max=domain_max,
        lut_1d=lut_1d,
        lut_3d=lut_3d,
    )


def apply_lut(image: np.ndarray, lut: CubeLut) -> np.ndarray:
    transformed = np.clip(image, 0.0, 1.0)
    if lut.lut_1d is not None:
        transformed = _apply_1d_lut(transformed, lut)
    if lut.lut_3d is not None:
        transformed = _apply_3d_lut(transformed, lut)
    return transformed


def _normalize(image: np.ndarray, lut: CubeLut) -> np.ndarray:
    domain_min = np.asarray(lut.domain_min, dtype=np.float32).reshape(3, 1, 1)
    domain_max = np.asarray(lut.domain_max, dtype=np.float32).reshape(3, 1, 1)
    return np.clip((image - domain_min) / np.maximum(domain_max - domain_min, 1e-6), 0.0, 1.0)


def _apply_1d_lut(image: np.ndarray, lut: CubeLut) -> np.ndarray:
    assert lut.lut_1d is not None
    normalized = _normalize(image, lut)
    size = lut.lut_1d.shape[0]
    positions = normalized * (size - 1)
    lo = np.floor(positions).astype(np.int32)
    hi = np.clip(lo + 1, 0, size - 1)
    frac = positions - lo
    result = np.empty_like(image)
    for channel in range(3):
        lo_values = lut.lut_1d[lo[channel], channel]
        hi_values = lut.lut_1d[hi[channel], channel]
        result[channel] = lo_values + (hi_values - lo_values) * frac[channel]
    return result


def _apply_3d_lut(image: np.ndarray, lut: CubeLut) -> np.ndarray:
    assert lut.lut_3d is not None
    normalized = _normalize(image, lut)
    size = lut.lut_3d.shape[0]
    positions = normalized * (size - 1)
    r_pos, g_pos, b_pos = positions[0], positions[1], positions[2]
    r0 = np.floor(r_pos).astype(np.int32)
    g0 = np.floor(g_pos).astype(np.int32)
    b0 = np.floor(b_pos).astype(np.int32)
    r1 = np.clip(r0 + 1, 0, size - 1)
    g1 = np.clip(g0 + 1, 0, size - 1)
    b1 = np.clip(b0 + 1, 0, size - 1)
    rf = (r_pos - r0)[..., None]
    gf = (g_pos - g0)[..., None]
    bf = (b_pos - b0)[..., None]

    c000 = lut.lut_3d[r0, g0, b0]
    c001 = lut.lut_3d[r0, g0, b1]
    c010 = lut.lut_3d[r0, g1, b0]
    c011 = lut.lut_3d[r0, g1, b1]
    c100 = lut.lut_3d[r1, g0, b0]
    c101 = lut.lut_3d[r1, g0, b1]
    c110 = lut.lut_3d[r1, g1, b0]
    c111 = lut.lut_3d[r1, g1, b1]

    c00 = c000 + (c100 - c000) * rf
    c01 = c001 + (c101 - c001) * rf
    c10 = c010 + (c110 - c010) * rf
    c11 = c011 + (c111 - c011) * rf
    c0 = c00 + (c10 - c00) * gf
    c1 = c01 + (c11 - c01) * gf
    return np.moveaxis(c0 + (c1 - c0) * bf, -1, 0)

