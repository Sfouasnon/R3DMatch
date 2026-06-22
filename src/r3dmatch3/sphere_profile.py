"""
R3DMatch v2 — Sphere Solve Profile Repository

A living, project-scoped store of verified sphere detections. After each
successful auto-detection passes all gates and the measurement is committed,
the detection geometry and photometrics are written into the profile store.

On subsequent runs, the profile is loaded and used to BIAS candidate ranking
inside the Hough pipeline — not to hard-gate. The gates stay hard. The profile
makes the ranking smarter so the correct sphere rises to the top faster.

Scope model:
  - Profiles are stored per PROJECT (identified by the input folder name
    or an explicit project_id).
  - Within a project, cameras move rarely — geometry priors are stable.
  - Across projects, cameras move — profiles do not transfer.
  - A project profile accumulates samples over multiple runs (clips 106,
    107, 108... from the same shoot location).

Store location:
  ~/Library/Application Support/R3DMatch_v2/profiles/<project_id>.json

Profile schema per camera entry:
  {
    "camera_label": "G007_A106",
    "samples": [
      {
        "clip_id": "G007_A106_0511R9_001",
        "run_id": "run_001",
        "recorded_at": "2026-05-21T...",
        "trust": "verified_auto" | "verified_manual",
        "geometry": {
          "cx_norm": 0.704,       # cx / frame_width
          "cy_norm": 0.392,       # cy / frame_height
          "radius_ratio": 0.063   # r / min(frame_h, frame_w)
        },
        "photometrics": {
          "ire_spread": 0.95,
          "chroma_distance": 0.0327,
          "lambertian_score": 1.0,
          "interior_lum_mean": 0.381,
          "interior_lum_stddev": 0.037,
          "hero_ire": 39.09
        }
      }
    ]
  }

How the profile biases detection:
  For each Hough candidate (cx, cy, r), compute its normalized geometry
  (cx_norm, cy_norm, radius_ratio) and compare to the camera's prior
  distribution. Candidates consistent with the prior get a bonus multiplier
  on their accumulator score before gate evaluation. Candidates far from the
  prior are still evaluated — the gates decide — but they rank lower.

  This means: on first run, no profile exists → pure Hough ranking.
  After one confirmed run, the profile exists → sphere candidates near the
  known position rank higher.
"""
from __future__ import annotations

import json
import logging
import math
import os
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Store location
# ---------------------------------------------------------------------------

_PROFILE_DIR = Path.home() / "Library" / "Application Support" / "R3DMatch_v2" / "profiles"
_SCHEMA_VERSION = "r3dmatch3_sphere_profile_v2"

# How many samples to keep per camera (ring buffer — oldest dropped first)
_MAX_SAMPLES_PER_CAMERA = 20

# Minimum photometric samples before derived stats narrow the gates
_MIN_SAMPLES_FOR_NARROWING = 3

# Bonus multiplier applied to candidates consistent with the prior
_PRIOR_MATCH_BONUS = 1.25

# Distance thresholds for "consistent with prior"
# In normalized units (fraction of frame)
_PRIOR_CENTER_MATCH_RADIUS = 0.10   # within 10% of frame in each axis
_PRIOR_RADIUS_MATCH_RATIO  = 0.30   # within 30% of expected radius ratio


# ---------------------------------------------------------------------------
# Project ID derivation
# ---------------------------------------------------------------------------

def project_id_from_path(input_path: str) -> str:
    """
    Derive a stable project ID from the input folder path.
    Uses the last two path components to avoid collisions:
      /Volumes/TProps/CPT → "TProps_CPT"
      /Desktop/Test_Footage/GraySphere_GrayBackdrop → "Test_Footage_GraySphere_GrayBackdrop"
    """
    parts = Path(input_path).expanduser().resolve().parts
    # Take last 2 non-trivial parts, sanitize
    meaningful = [p for p in parts if p not in ("/", "Volumes", "Users", "Desktop", "")]
    label = "_".join(meaningful[-2:]) if len(meaningful) >= 2 else meaningful[-1] if meaningful else "default"
    # Sanitize to filesystem-safe
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in label)
    return safe[:80]


# ---------------------------------------------------------------------------
# Load / save
# ---------------------------------------------------------------------------

def load_project_profile(project_id: str) -> Dict:
    """
    Load the profile for this project. Returns empty profile if not found.
    """
    try:
        os.makedirs(_PROFILE_DIR, exist_ok=True)
    except OSError as exc:
        log.warning("Could not create profile directory %s: %s", _PROFILE_DIR, exc)
        return _empty_profile(project_id)
    path = _profile_path(project_id)
    if not path.exists():
        return _empty_profile(project_id)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("schema_version") != _SCHEMA_VERSION:
            return _empty_profile(project_id)
        return data
    except json.JSONDecodeError:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        corrupt_path = path.with_name(f"{path.name}.corrupt.{stamp}")
        log.warning("Corrupt sphere profile JSON at %s; moving aside to %s", path, corrupt_path)
        try:
            path.rename(corrupt_path)
        except OSError as exc:
            log.warning("Could not quarantine corrupt profile %s: %s", path, exc)
        return _empty_profile(project_id)
    except OSError as exc:
        log.warning("Could not read sphere profile %s: %s", path, exc)
        return _empty_profile(project_id)
    except Exception:
        return _empty_profile(project_id)


def save_project_profile(project_id: str, profile: Dict) -> None:
    """Write the profile to disk."""
    path = _profile_path(project_id)
    tmp_path = path.with_name(f"{path.name}.tmp")
    try:
        os.makedirs(path.parent, exist_ok=True)
    except OSError as exc:
        log.warning("Could not create profile directory %s: %s", path.parent, exc)
        return
    try:
        tmp_path.write_text(json.dumps(profile, indent=2), encoding="utf-8")
        os.replace(tmp_path, path)
    except OSError as exc:
        log.warning("Could not write sphere profile %s: %s", path, exc)
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass


def _profile_path(project_id: str) -> Path:
    return _PROFILE_DIR / f"{project_id}.json"


def _empty_profile(project_id: str) -> Dict:
    return {
        "schema_version": _SCHEMA_VERSION,
        "project_id": project_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "cameras": {},
    }


# ---------------------------------------------------------------------------
# Recording a verified detection
# ---------------------------------------------------------------------------

def record_detection(
    profile: Dict,
    *,
    clip_id: str,
    camera_label: str,
    run_id: str,
    roi_cx: float,
    roi_cy: float,
    roi_r: float,
    frame_width: int,
    frame_height: int,
    ire_spread: float,
    chroma_distance: float,
    lambertian_score: float,
    interior_lum_mean: float,
    interior_lum_stddev: float,
    hero_ire: float,
    ring_lum: Optional[List[float]] = None,
    brdf_score: Optional[float] = None,
    trust: str = "verified_auto",
) -> Dict:
    """
    Add one verified detection to the profile. Returns the updated profile.
    Caller is responsible for saving.
    """
    min_dim = min(frame_width, frame_height)

    sample = {
        "clip_id": clip_id,
        "run_id": run_id,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "trust": trust,
        "geometry": {
            "cx_norm":      round(roi_cx / frame_width,  4),
            "cy_norm":      round(roi_cy / frame_height, 4),
            "radius_ratio": round(roi_r  / min_dim,      4),
        },
        "photometrics": {
            "ire_spread":          round(ire_spread,          3),
            "chroma_distance":     round(chroma_distance,     4),
            "lambertian_score":    round(lambertian_score,    3),
            "interior_lum_mean":   round(interior_lum_mean,   4),
            "interior_lum_stddev": round(interior_lum_stddev, 4),
            "hero_ire":            round(hero_ire,            2),
            "ring_lum":            [round(v, 4) for v in ring_lum] if ring_lum else None,
            "brdf_score":          round(brdf_score, 3) if brdf_score is not None else None,
        },
    }

    cameras = profile.setdefault("cameras", {})
    entry   = cameras.setdefault(camera_label, {"camera_label": camera_label, "samples": []})
    samples = entry["samples"]

    if trust == "verified_manual":
        # Manual ROI always wins — discard ALL previous samples for this camera
        # and replace with this single authoritative measurement.
        entry["samples"] = [sample]
    else:
        # Auto-detect: deduplicate by clip_id + run_id, then ring-buffer
        already = any(s["clip_id"] == clip_id and s["run_id"] == run_id for s in samples)
        if not already:
            samples.append(sample)
            if len(samples) > _MAX_SAMPLES_PER_CAMERA:
                entry["samples"] = samples[-_MAX_SAMPLES_PER_CAMERA:]

    profile["updated_at"] = datetime.now(timezone.utc).isoformat()
    return profile


# ---------------------------------------------------------------------------
# Manual sample photometrics
# ---------------------------------------------------------------------------

def manual_sample_photometrics(
    *,
    detection=None,
    measurement=None,
) -> Dict[str, float]:
    """
    Build the photometrics block for a verified_manual sample.

    If no measurement is available, preserve the historical zero-filled
    behavior so profile writes never block.
    """
    if measurement is None:
        return {
            "ire_spread": 0.0,
            "chroma_distance": 0.0,
            "lambertian_score": 0.0,
            "interior_lum_mean": 0.0,
            "interior_lum_stddev": 0.0,
            "hero_ire": 0.0,
            "ring_lum": None,
            "brdf_score": None,
        }

    return {
        "ire_spread": _manual_ire_spread(measurement),
        "chroma_distance": _detection_metric(
            detection,
            gate_name="gray_material",
            diagnostic_keys=("chroma_distance", "chromaticity_distance"),
            fallback_attr="chromaticity_distance",
        ),
        "lambertian_score": _detection_metric(
            detection,
            gate_name="lambertian",
            diagnostic_keys=("lambertian_score",),
            fallback_attr="lambertian_score",
        ),
        "interior_lum_mean": _detection_metric(
            detection,
            gate_name="interior_stddev",
            diagnostic_keys=("interior_lum_mean", "interior_luminance_mean"),
            fallback_attr="interior_luminance_mean",
        ),
        "interior_lum_stddev": _detection_metric(
            detection,
            gate_name="interior_stddev",
            diagnostic_keys=("interior_lum_stddev", "interior_luminance_stddev"),
            fallback_attr="interior_luminance_stddev",
        ),
        "hero_ire": float(getattr(measurement, "hero_ire", 0.0) or 0.0),
        "ring_lum": _detection_ring_lum(detection),
        "brdf_score": _detection_brdf_score(detection),
    }


def _detection_ring_lum(detection) -> Optional[List[float]]:
    """Extract ring_lum list from a detection result's lambertian gate."""
    if detection is None:
        return None
    for gate in getattr(detection, "gates", []) or []:
        if getattr(gate, "gate", "") == "lambertian":
            diag = getattr(gate, "diagnostics", None) or {}
            rl = diag.get("ring_lum")
            if rl and isinstance(rl, list):
                return [float(v) for v in rl if v is not None]
            # Fall back to parsing reason string
            import re as _re
            reason = getattr(gate, "reason", "") or ""
            m = _re.search(r"ring_lum=\[([^\]]+)\]", reason)
            if m:
                try:
                    vals = []
                    for v in m.group(1).split(","):
                        vals.append(float(v.strip().strip("\'")))
                    return vals
                except ValueError:
                    pass
    return None


def _detection_brdf_score(detection) -> Optional[float]:
    """Extract BRDF score from a detection result."""
    if detection is None:
        return None
    score = getattr(detection, "lambertian_score", None)
    if score is not None:
        try:
            return float(score)
        except (TypeError, ValueError):
            pass
    for gate in getattr(detection, "gates", []) or []:
        if getattr(gate, "gate", "") == "lambertian":
            diag = getattr(gate, "diagnostics", None) or {}
            s = diag.get("lambertian_score") or diag.get("brdf_score")
            if s is not None:
                try:
                    return float(s)
                except (TypeError, ValueError):
                    pass
    return None


def _manual_ire_spread(measurement) -> float:
    zone_bright = getattr(measurement, "zone_bright", None)
    zone_dark = getattr(measurement, "zone_dark", None)
    bright_ire = getattr(zone_bright, "ire", None)
    dark_ire = getattr(zone_dark, "ire", None)
    if bright_ire is None or dark_ire is None:
        return 0.0
    return float(bright_ire - dark_ire)


def _detection_metric(
    detection,
    *,
    gate_name: str,
    diagnostic_keys: Tuple[str, ...],
    fallback_attr: str,
) -> float:
    if detection is None:
        return 0.0

    for gate in getattr(detection, "gates", []) or []:
        if getattr(gate, "gate", "") != gate_name:
            continue
        diagnostics = getattr(gate, "diagnostics", None) or {}
        for key in diagnostic_keys:
            value = diagnostics.get(key)
            if value is not None:
                try:
                    return float(value)
                except (TypeError, ValueError):
                    break

    value = getattr(detection, fallback_attr, 0.0)
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


# ---------------------------------------------------------------------------
# Computing priors from stored samples
# ---------------------------------------------------------------------------

def get_camera_prior(profile: Dict, camera_label: str) -> Optional[Dict]:
    """
    Compute the prior distribution for a camera from its stored samples.

    Priority order:
      1. Use verified_manual samples if any exist.
      2. Otherwise fall back to verified_auto samples.

    Returns a dict with median/mad for geometry fields, or None if the chosen
    tier has no samples.
    """
    cameras = profile.get("cameras", {})
    entry   = cameras.get(camera_label, {})
    all_samples = entry.get("samples", [])
    manual_samples = [s for s in all_samples if s.get("trust") == "verified_manual"]
    auto_samples = [s for s in all_samples if s.get("trust") == "verified_auto"]
    samples = manual_samples if manual_samples else auto_samples

    if len(samples) < 1:
        return None  # not enough data — use pure Hough ranking

    cx_norms      = [s["geometry"]["cx_norm"]      for s in samples]
    cy_norms      = [s["geometry"]["cy_norm"]       for s in samples]
    radius_ratios = [s["geometry"]["radius_ratio"]  for s in samples]
    ire_spreads   = [s["photometrics"]["ire_spread"] for s in samples]
    hero_ires     = [s["photometrics"]["hero_ire"]   for s in samples]

    def _mad(values: List[float]) -> float:
        med = statistics.median(values)
        return statistics.median([abs(v - med) for v in values])

    n = len(samples)
    photo_ready = n >= _MIN_SAMPLES_FOR_NARROWING
    stds = [s["photometrics"].get("interior_lum_stddev", 0.0) for s in samples
            if s["photometrics"].get("interior_lum_stddev")]
    brdf_scores = [s["photometrics"]["brdf_score"] for s in samples
                   if s["photometrics"].get("brdf_score") is not None]
    ring_derived = []
    for ri in range(4):
        ring_vals = [s["photometrics"]["ring_lum"][ri]
                     for s in samples
                     if s["photometrics"].get("ring_lum")
                     and len(s["photometrics"]["ring_lum"]) > ri
                     and s["photometrics"]["ring_lum"][ri] is not None]
        if ring_vals:
            ring_derived.append({"median": round(statistics.median(ring_vals), 4),
                                  "mad":    round(_mad(ring_vals), 4),
                                  "min":    round(min(ring_vals), 4),
                                  "max":    round(max(ring_vals), 4)})
        else:
            ring_derived.append(None)
    return {
        "camera_label":          camera_label,
        "sample_count":          n,
        "sample_tier":           "verified_manual" if manual_samples else "verified_auto",
        "photo_narrowing_ready": photo_ready,
        # Geometry priors
        "cx_norm_mean":          statistics.median(cx_norms),
        "cx_norm_mad":           _mad(cx_norms),
        "cy_norm_mean":          statistics.median(cy_norms),
        "cy_norm_mad":           _mad(cy_norms),
        "radius_ratio_mean":     statistics.median(radius_ratios),
        "radius_ratio_mad":      _mad(radius_ratios),
        # Photometric priors
        "ire_spread_mean":       statistics.median(ire_spreads),
        "ire_spread_mad":        _mad(ire_spreads),
        "hero_ire_mean":         statistics.median(hero_ires),
        "hero_ire_mad":          _mad(hero_ires),
        # v2 gate narrowing (None when photo_narrowing_ready=False)
        "interior_std_median":   statistics.median(stds) if stds else None,
        "interior_std_mad":      _mad(stds) if len(stds) >= 2 else None,
        "ring_lum_derived":      ring_derived if photo_ready else None,
        "brdf_score_median":     statistics.median(brdf_scores) if brdf_scores else None,
        "brdf_score_min":        min(brdf_scores) if brdf_scores else None,
    }


# ---------------------------------------------------------------------------
# Applying the prior to bias candidate ranking
# ---------------------------------------------------------------------------

def apply_prior_bonus(
    candidates: List[Tuple[float, float, float, float]],
    *,
    prior: Optional[Dict],
    frame_width: int,
    frame_height: int,
) -> List[Tuple[float, float, float, float]]:
    """
    Re-score Hough candidates using the camera prior.

    candidates: list of (accumulator, cx, cy, r) at FULL resolution
    prior: from get_camera_prior(), or None
    Returns re-sorted list with prior-consistent candidates ranked higher.

    The bonus is multiplicative on the accumulator score. A candidate
    within the expected position/radius window gets acc * 1.25.
    A candidate outside gets acc * 1.0 (no change).

    This never eliminates candidates — gates do that. It only reorders.
    """
    if prior is None or not candidates:
        return candidates

    min_dim = min(frame_width, frame_height)
    cx_mean   = prior["cx_norm_mean"]
    cy_mean   = prior["cy_norm_mean"]
    r_mean    = prior["radius_ratio_mean"]

    scored: List[Tuple[float, float, float, float]] = []
    for acc, cx, cy, r in candidates:
        cx_norm = cx / frame_width
        cy_norm = cy / frame_height
        r_ratio = r  / min_dim

        # Distance from prior center in normalized units
        center_dist = math.hypot(cx_norm - cx_mean, cy_norm - cy_mean)
        r_delta     = abs(r_ratio - r_mean) / max(r_mean, 1e-6)

        center_match = center_dist <= _PRIOR_CENTER_MATCH_RADIUS
        radius_match = r_delta     <= _PRIOR_RADIUS_MATCH_RATIO

        bonus = _PRIOR_MATCH_BONUS if (center_match and radius_match) else 1.0
        scored.append((acc * bonus, cx, cy, r))

    scored.sort(key=lambda x: x[0], reverse=True)
    # Restore original accumulator values (bonus was just for ranking)
    # Map back by position — the order changed but values should be originals
    orig_by_pos = {(cx, cy, r): acc for acc, cx, cy, r in candidates}
    return [(orig_by_pos.get((cx, cy, r), acc / (_PRIOR_MATCH_BONUS if acc != orig_by_pos.get((cx,cy,r), acc) else 1.0)), cx, cy, r)
            for acc, cx, cy, r in scored]


def apply_prior_bonus_detect_scale(
    candidates: List[Tuple[float, float, float, float]],
    *,
    prior: Optional[Dict],
    frame_width: int,
    frame_height: int,
    scale: float,
) -> List[Tuple[float, float, float, float]]:
    """
    Same as apply_prior_bonus but candidates are at DETECTION scale.
    Converts prior to detection scale for comparison, then returns
    detection-scale candidates re-ordered.
    """
    if prior is None or not candidates:
        return candidates

    # Expected center in detection-scale pixels
    cx_expected = prior["cx_norm_mean"] * frame_width  * scale
    cy_expected = prior["cy_norm_mean"] * frame_height * scale
    r_expected  = prior["radius_ratio_mean"] * min(frame_width, frame_height) * scale

    scored: List[Tuple[float, float, float, float]] = []
    for acc, cx, cy, r in candidates:
        center_dist_px = math.hypot(cx - cx_expected, cy - cy_expected)
        center_dist_norm = center_dist_px / max(min(frame_width, frame_height) * scale, 1.0)
        r_delta = abs(r - r_expected) / max(r_expected, 1.0)

        center_match = center_dist_norm <= _PRIOR_CENTER_MATCH_RADIUS
        radius_match = r_delta          <= _PRIOR_RADIUS_MATCH_RATIO

        bonus = _PRIOR_MATCH_BONUS if (center_match and radius_match) else 1.0
        scored.append((acc * bonus, cx, cy, r))

    scored.sort(key=lambda x: x[0], reverse=True)

    # Restore original accumulators in new order
    orig = {(round(cx, 1), round(cy, 1), round(r, 1)): acc
            for acc, cx, cy, r in candidates}
    result = []
    for bonused_acc, cx, cy, r in scored:
        key = (round(cx, 1), round(cy, 1), round(r, 1))
        original_acc = orig.get(key, bonused_acc)
        result.append((original_acc, cx, cy, r))
    return result


# ---------------------------------------------------------------------------
# Profile summary for reporting
# ---------------------------------------------------------------------------

def profile_summary(profile: Dict) -> Dict:
    """Human-readable summary of the profile for the contact sheet."""
    cameras   = profile.get("cameras", {})
    total_samples = sum(len(e.get("samples", [])) for e in cameras.values())
    camera_count  = len(cameras)
    return {
        "project_id":    profile.get("project_id", ""),
        "camera_count":  camera_count,
        "total_samples": total_samples,
        "cameras": {
            label: {
                "sample_count": len(entry.get("samples", [])),
                "prior": get_camera_prior(profile, label),
            }
            for label, entry in cameras.items()
        },
    }


def format_prior_for_report(prior: Optional[Dict]) -> str:
    """One-line human-readable prior description."""
    if prior is None or prior.get("sample_count", 0) < 1:
        return "No prior (first run)"
    n = prior["sample_count"]
    cx = prior["cx_norm_mean"]
    cy = prior["cy_norm_mean"]
    r  = prior["radius_ratio_mean"]
    return (f"{n} samples · "
            f"center ({cx:.2f}, {cy:.2f}) ± {prior['cx_norm_mad']:.3f} · "
            f"r_ratio {r:.3f} ± {prior['radius_ratio_mad']:.3f}")
