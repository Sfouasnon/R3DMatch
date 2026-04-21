from __future__ import annotations

import numpy as np
import pytest
from skimage import color, feature

from r3dmatch.models import SphereROI
from r3dmatch.report import (
    _detect_sphere_roi_in_region_hwc,
    _evaluate_detected_sphere_roi,
    _refine_sphere_candidate_geometry,
    _sphere_candidate_hard_gate,
    _sphere_candidate_neutral_region_probe,
    _sphere_candidate_profile_plausibility,
    _sphere_candidate_radial_luminance_profile,
    _sphere_radial_gradient_coherence,
)


def _blank_region(height: int = 240, width: int = 240, value: float = 0.08) -> np.ndarray:
    return np.full((height, width, 3), value, dtype=np.float32)


def _circle_mask(height: int, width: int, cx: float, cy: float, radius: float) -> np.ndarray:
    yy, xx = np.meshgrid(np.arange(height, dtype=np.float32), np.arange(width, dtype=np.float32), indexing="ij")
    return np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) <= radius


def _ellipse_mask(height: int, width: int, cx: float, cy: float, rx: float, ry: float) -> np.ndarray:
    yy, xx = np.meshgrid(np.arange(height, dtype=np.float32), np.arange(width, dtype=np.float32), indexing="ij")
    return (((xx - cx) / max(rx, 1.0)) ** 2 + ((yy - cy) / max(ry, 1.0)) ** 2) <= 1.0


def _flat_neutral_circle() -> tuple[np.ndarray, SphereROI]:
    image = _blank_region()
    roi = SphereROI(cx=120.0, cy=118.0, r=54.0)
    mask = _circle_mask(*image.shape[:2], roi.cx, roi.cy, roi.r)
    image[mask] = np.asarray([0.34, 0.34, 0.34], dtype=np.float32)
    return image, roi


def _small_neutral_circle() -> tuple[np.ndarray, SphereROI]:
    image = _blank_region()
    roi = SphereROI(cx=120.0, cy=118.0, r=14.0)
    mask = _circle_mask(*image.shape[:2], roi.cx, roi.cy, roi.r)
    image[mask] = np.asarray([0.34, 0.34, 0.34], dtype=np.float32)
    return image, roi


def _neutral_ellipse() -> tuple[np.ndarray, SphereROI]:
    image = _blank_region()
    roi = SphereROI(cx=120.0, cy=118.0, r=54.0)
    mask = _ellipse_mask(*image.shape[:2], roi.cx, roi.cy, 70.0, 36.0)
    image[mask] = np.asarray([0.34, 0.34, 0.34], dtype=np.float32)
    return image, roi


def _dark_neutral_circle() -> tuple[np.ndarray, SphereROI]:
    image = _blank_region(value=0.02)
    roi = SphereROI(cx=120.0, cy=118.0, r=54.0)
    mask = _circle_mask(*image.shape[:2], roi.cx, roi.cy, roi.r)
    image[mask] = np.asarray([0.03, 0.03, 0.03], dtype=np.float32)
    return image, roi


def _valid_sphere() -> tuple[np.ndarray, SphereROI]:
    image = _blank_region()
    roi = SphereROI(cx=122.0, cy=116.0, r=56.0)
    height, width = image.shape[:2]
    yy, xx = np.meshgrid(np.arange(height, dtype=np.float32), np.arange(width, dtype=np.float32), indexing="ij")
    dx = (xx - roi.cx) / roi.r
    dy = (yy - roi.cy) / roi.r
    radial = dx * dx + dy * dy
    mask = radial <= 1.0
    nz = np.zeros_like(dx, dtype=np.float32)
    nz[mask] = np.sqrt(np.clip(1.0 - radial[mask], 0.0, 1.0))
    light = np.asarray([-0.58, -0.16, 0.80], dtype=np.float32)
    light /= np.linalg.norm(light)
    lambert = np.clip((dx * light[0]) + (dy * light[1]) + (nz * light[2]), 0.0, 1.0)
    base = np.asarray([0.31, 0.315, 0.305], dtype=np.float32)
    shaded = base * (0.58 + (0.48 * lambert[..., None]))
    image[mask] = shaded[mask]
    rim = np.abs(np.sqrt(np.maximum(radial, 0.0)) - 1.0) <= 0.035
    image[rim] = np.asarray([0.48, 0.48, 0.48], dtype=np.float32)
    return image, roi


def _candidate_for(region: np.ndarray, roi: SphereROI, source: str = "primary_detected") -> dict[str, object]:
    edge_map = feature.canny(np.clip(color.rgb2gray(region), 0.0, 1.0), sigma=2.0, low_threshold=0.03, high_threshold=0.12)
    return _evaluate_detected_sphere_roi(
        region,
        roi=roi,
        edge_map=edge_map,
        accumulator=0.0,
        detection_source=source,
        shape_circularity=0.86,
        aspect_ratio=1.0,
    )


@pytest.mark.parametrize(
    ("builder", "expected_reason"),
    [
        (_flat_neutral_circle, "flat_luminance_profile"),
        (_small_neutral_circle, "minimum_size_not_met"),
        (_neutral_ellipse, "non_circular_region"),
        (_dark_neutral_circle, "flat_low_ire_region"),
    ],
)
def test_sphere_hard_gate_rejects_invalid_candidates(builder, expected_reason: str) -> None:
    region, roi = builder()
    candidate = _candidate_for(region, roi)
    probe = _sphere_candidate_profile_plausibility(region, candidate)
    hard_gate = _sphere_candidate_hard_gate(candidate, probe)

    assert hard_gate["passed"] is False
    assert expected_reason in set(hard_gate["reasons"])
    assert probe["score"] == pytest.approx(0.0)


def test_valid_large_neutral_sphere_passes_all_detector_gates() -> None:
    region, roi = _valid_sphere()
    candidate = _refine_sphere_candidate_geometry(region, _candidate_for(region, roi))
    neutral_probe = _sphere_candidate_neutral_region_probe(region, roi)
    radial_profile = _sphere_candidate_radial_luminance_profile(region, roi)
    radial_coherence = _sphere_radial_gradient_coherence(region, roi)
    probe = _sphere_candidate_profile_plausibility(region, candidate)
    hard_gate = _sphere_candidate_hard_gate(candidate, probe)

    assert neutral_probe["score"] > 0.0
    assert neutral_probe["fragmented"] is False
    assert radial_profile["score"] > 0.26
    assert radial_coherence["score"] > 0.56
    assert probe["score"] >= 0.46
    assert probe["label"] in {"MEDIUM", "HIGH"}
    assert hard_gate["passed"] is True


def test_fallback_candidate_cannot_bypass_same_hard_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    region, roi = _flat_neutral_circle()
    flat_candidate = _candidate_for(region, roi, source="opencv_hough_recovery")

    monkeypatch.setattr("r3dmatch.report._resize_region_for_sphere_detection", lambda current: (current, 1.0))
    monkeypatch.setattr(
        "r3dmatch.report._detect_sphere_candidates_in_region_hwc",
        lambda *_args, **kwargs: [],
    )
    monkeypatch.setattr(
        "r3dmatch.report._detect_sphere_candidates_opencv_hough_hwc",
        lambda *_args, **kwargs: [dict(flat_candidate)],
    )
    monkeypatch.setattr("r3dmatch.report._detect_sphere_candidates_opencv_hough_alt_hwc", lambda *_args, **kwargs: [])
    monkeypatch.setattr("r3dmatch.report._detect_sphere_candidates_opencv_contours_hwc", lambda *_args, **kwargs: [])
    monkeypatch.setattr("r3dmatch.report._detect_neutral_blob_candidates_in_region_hwc", lambda *_args, **kwargs: [])
    monkeypatch.setattr("r3dmatch.report._detect_gray_card_in_region_hwc", lambda *_args, **kwargs: {"roi": {}, "confidence": 0.0})

    detected = _detect_sphere_roi_in_region_hwc(region)

    assert detected["sphere_detection_success"] is False
    assert detected["sphere_detection_unresolved"] is True
    assert detected["source"] == "unresolved"
    assert "flat_luminance_profile" in set((detected.get("hard_gate") or {}).get("reasons") or [])


def test_hard_gate_accepts_strong_centered_partial_neutral_sphere_shape_proxy() -> None:
    candidate = {
        "validation": {"radius_sane": True},
        "radius_ratio": 0.13,
    }
    profile_probe = {
        "plausible_samples": True,
        "zone_spread_ire": 28.0,
        "chroma_range": 0.031,
        "valid_pixel_count": 61000,
        "valid_pixel_fraction": 0.11,
        "center_ire": 45.0,
        "interior_stddev": 0.089,
        "measurement_confidence": 0.80,
        "center_margin_ire": 5.1,
        "center_between_extremes": True,
        "sample_range_ire": 28.2,
        "samples_equal": False,
        "flat_high": False,
        "flat_low": False,
        "neutral_consistency_score": 0.76,
        "shape_score": 0.0,
        "exposure_validity_score": 1.0,
        "edge_support": 0.71,
        "fit_residual_ratio": 0.028,
        "radial_gradient_coherence": {"score": 0.68, "support": 172000},
        "radial_luminance_profile": {"score": 0.66},
        "neutral_region": {
            "region_expansion_score": 0.72,
            "seed_area_fraction": 0.70,
            "edge_aligned": False,
            "circularity": 0.036,
            "aspect_ratio": 1.10,
            "radial_cv": 0.34,
            "solidity": 0.76,
            "component_count": 1,
            "fragmented": False,
            "centroid_distance_norm": 0.02,
        },
        "critical_failures": [],
    }

    hard_gate = _sphere_candidate_hard_gate(candidate, profile_probe)

    assert hard_gate["passed"] is True
    assert "non_circular_region" not in set(hard_gate["reasons"])
    assert "circularity_below_min" not in set(hard_gate["reasons"])
    assert "aspect_ratio_above_max" not in set(hard_gate["reasons"])
    assert "shape_solidity_below_min" not in set(hard_gate["reasons"])


def test_hard_gate_accepts_valid_candidate_geometry_when_neutral_mask_shape_is_partial() -> None:
    candidate = {
        "validation": {"radius_sane": True},
        "radius_ratio": 0.10,
        "shape_circularity": 0.72,
        "aspect_ratio": 1.12,
        "opencv_contour": {"area_ratio": 0.03, "circularity": 0.72},
    }
    profile_probe = {
        "plausible_samples": True,
        "zone_spread_ire": 8.1,
        "chroma_range": 0.028,
        "valid_pixel_count": 18943,
        "valid_pixel_fraction": 0.11,
        "center_ire": 47.0,
        "interior_stddev": 0.041,
        "measurement_confidence": 0.81,
        "center_margin_ire": 3.7,
        "center_between_extremes": True,
        "sample_range_ire": 8.2,
        "samples_equal": False,
        "flat_high": False,
        "flat_low": False,
        "neutral_consistency_score": 0.77,
        "shape_score": 0.0,
        "exposure_validity_score": 1.0,
        "edge_support": 0.79,
        "fit_residual_ratio": 0.024,
        "radial_gradient_coherence": {"score": 0.79, "support": 18000},
        "radial_luminance_profile": {"score": 0.75},
        "neutral_region": {
            "region_expansion_score": 0.76,
            "seed_area_fraction": 0.12,
            "edge_aligned": False,
            "circularity": 0.07,
            "aspect_ratio": 1.13,
            "radial_cv": 0.39,
            "solidity": 0.77,
            "component_count": 1,
            "fragmented": False,
            "centroid_distance_norm": 0.04,
        },
        "critical_failures": [],
    }

    hard_gate = _sphere_candidate_hard_gate(candidate, profile_probe)

    assert hard_gate["passed"] is True
    assert "non_circular_region" not in set(hard_gate["reasons"])
    assert "circularity_below_min" not in set(hard_gate["reasons"])
    assert "shape_solidity_below_min" not in set(hard_gate["reasons"])


def test_fallback_runs_when_primary_candidates_exist_but_all_fail_hard_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    valid_region, valid_roi = _valid_sphere()
    primary_invalid = {
        "roi": {"cx": valid_roi.cx, "cy": valid_roi.cy, "r": valid_roi.r},
        "source": "primary_detected",
        "confidence": 0.91,
        "confidence_label": "HIGH",
        "sample_plausibility": "IMPLAUSIBLE",
        "profile_probe": {"score": 0.0},
        "hard_gate": {"passed": False, "reasons": ["non_circular_region"]},
        "sphere_detection_success": False,
    }
    fallback_valid = _candidate_for(valid_region, valid_roi, source="opencv_hough_recovery")
    fallback_valid["hard_gate"] = {"passed": True, "reasons": []}
    fallback_valid["sample_plausibility"] = "HIGH"
    fallback_valid["profile_probe"] = {"score": 0.82}

    monkeypatch.setattr("r3dmatch.report._resize_region_for_sphere_detection", lambda current: (current, 1.0))
    monkeypatch.setattr(
        "r3dmatch.report._detect_sphere_candidates_in_region_hwc",
        lambda *_args, **kwargs: [dict(primary_invalid)] if kwargs.get("detection_source") == "primary_detected" else [],
    )
    monkeypatch.setattr(
        "r3dmatch.report._detect_sphere_candidates_opencv_hough_hwc",
        lambda *_args, **kwargs: [dict(fallback_valid)],
    )
    monkeypatch.setattr("r3dmatch.report._detect_sphere_candidates_opencv_hough_alt_hwc", lambda *_args, **kwargs: [])
    monkeypatch.setattr("r3dmatch.report._detect_sphere_candidates_opencv_contours_hwc", lambda *_args, **kwargs: [])
    monkeypatch.setattr("r3dmatch.report._detect_neutral_blob_candidates_in_region_hwc", lambda *_args, **kwargs: [])
    monkeypatch.setattr("r3dmatch.report._detect_gray_card_in_region_hwc", lambda *_args, **kwargs: {"roi": {}, "confidence": 0.0})
    def _stub_rescore(_region: np.ndarray, candidates: list[dict[str, object]], **_kwargs: object) -> list[dict[str, object]]:
        rescored = []
        for candidate in candidates:
            item = dict(candidate)
            if item.get("source") == "primary_detected":
                item["hard_gate"] = {"passed": False, "reasons": ["non_circular_region"]}
                item["sample_plausibility"] = "IMPLAUSIBLE"
                item["profile_probe"] = {"score": 0.0}
                item["confidence"] = 0.0
            else:
                item["hard_gate"] = {"passed": True, "reasons": []}
                item["sample_plausibility"] = "HIGH"
                item["profile_probe"] = {"score": 0.82}
                item["confidence"] = max(float(item.get("confidence", 0.0) or 0.0), 0.82)
            rescored.append(item)
        return rescored

    monkeypatch.setattr("r3dmatch.report._rescore_sphere_candidates_with_profile", _stub_rescore)

    detected = _detect_sphere_roi_in_region_hwc(valid_region)

    assert detected["sphere_detection_success"] is True
    assert detected["sphere_detection_unresolved"] is False
    assert detected["source"] == "opencv_hough_recovery"
    assert bool((detected.get("candidate_diagnostics") or {}).get("fallback_activated")) is False
