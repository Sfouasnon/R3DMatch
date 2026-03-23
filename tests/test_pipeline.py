import copy
import os
from pathlib import Path

import pytest
import torch
from typer.testing import CliRunner

from r3dsplat.cache import SequenceCache
from r3dsplat.cli import app
from r3dsplat.colmap_bridge import ColmapCliBridge, ColmapConfig, parse_images_txt
from r3dsplat.dataset import TemporalSequenceDataset
from r3dsplat.fiducials import FiducialConfig, solve_fiducials
from r3dsplat.geometry import align_cameras_to_fiducials, align_world, debug_poses
from r3dsplat.ingest_backends import MockIngestBackend, ingest_backend_summary, resolve_ingest_backend
from r3dsplat.metadata import CameraRecord, DatasetManifest
from r3dsplat.model import CanonicalGaussianModel, TimeConditionedDeformationModel
from r3dsplat.r3d_ingest import ingest_clip
from r3dsplat.training import FourDGaussianTrainer, TrainingConfig, _collate, evaluate, render_sequence, train
from r3dsplat.training_bridge import GSplatRendererBridge


pytestmark = []
runner = CliRunner()


def _make_dataset(tmp_path: Path) -> Path:
    dataset_dir = tmp_path / "dataset"
    ingest_clip("synthetic.R3D", str(dataset_dir), backend="mock")
    return dataset_dir


def test_backend_resolution_prefers_explicit_mock() -> None:
    backend = resolve_ingest_backend("mock")
    assert isinstance(backend, MockIngestBackend)
    assert backend.diagnostics()["backend"] == "mock"
    assert ingest_backend_summary("mock")["capability"] == "mock"


def test_red_sdk_backend_is_explicit_when_unavailable() -> None:
    try:
        resolve_ingest_backend("red-sdk")
    except RuntimeError as exc:
        assert "RED SDK native backend is not built" in str(exc)
    else:
        raise AssertionError("expected red-sdk backend resolution to fail without native SDK support")
    summary = ingest_backend_summary("red-sdk")
    assert summary["backend"] == "red-sdk"
    assert summary["capability"] in {"native-stub", "native-compiled-unavailable", "unavailable"}


def test_ingest_builds_temporal_dataset_and_camera_records(tmp_path: Path) -> None:
    dataset_dir = _make_dataset(tmp_path)
    dataset = TemporalSequenceDataset(dataset_dir, window_size=4, stride=2)
    manifest = DatasetManifest.load(dataset_dir)
    sample = dataset[0]
    assert sample["frames"].shape == (4, 3, 64, 64)
    assert sample["timestamps"].shape == (4,)
    assert sample["cameras"]["viewmats"].shape == (4, 4, 4)
    assert len(manifest.camera_records) == manifest.clip.total_frames
    assert dataset.validation_report["frame_count"] == manifest.clip.total_frames
    assert torch.all(sample["timestamps"][1:] > sample["timestamps"][:-1])


def test_cache_manifest_roundtrip_and_backend_report(tmp_path: Path) -> None:
    dataset_dir = _make_dataset(tmp_path)
    manifest = DatasetManifest.load(dataset_dir)
    cache = SequenceCache(dataset_dir)
    assert manifest.manifest_version == 2
    assert manifest.clip.total_frames == 12
    assert manifest.backend_report.ingest_backend == "mock"
    first = cache.read_frame(manifest.frames[0])
    assert tuple(first.shape) == (3, 64, 64)


def test_ingest_subset_controls_and_progress_callback(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "subset"
    progress_events = []
    ingest_clip(
        "synthetic.R3D",
        str(dataset_dir),
        backend="mock",
        start_frame=2,
        max_frames=4,
        frame_step=2,
        progress_callback=progress_events.append,
    )
    manifest = DatasetManifest.load(dataset_dir)
    assert [frame.frame_index for frame in manifest.frames] == [2, 4, 6, 8]
    assert len(progress_events) == 4
    assert progress_events[-1]["completed"] == 4
    assert progress_events[-1]["total"] == 4


def test_ingest_resize_updates_manifest_and_dataset_shape(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "resized"
    ingest_clip(
        "synthetic.R3D",
        str(dataset_dir),
        backend="mock",
        max_frames=4,
        max_width=32,
    )
    manifest = DatasetManifest.load(dataset_dir)
    assert manifest.backend_report.ingest_settings["original_width"] == 64
    assert manifest.backend_report.ingest_settings["cached_width"] == 32
    assert manifest.frames[0].cached_width == 32
    dataset = TemporalSequenceDataset(dataset_dir, window_size=2, stride=1)
    sample = dataset[0]
    assert sample["frames"].shape == (2, 3, 32, 32)
    assert sample["cameras"]["Ks"][0, 0, 0].item() == pytest.approx(30.0)


def test_model_forward_shapes() -> None:
    canonical = CanonicalGaussianModel(num_gaussians=32, feature_dim=3)
    deform = TimeConditionedDeformationModel(feature_dim=3)
    state = canonical.canonical_state()
    timestamps = torch.tensor([0.0, 0.1, 0.2], dtype=torch.float32)
    deformed = deform(state, timestamps)
    assert deformed.means.shape == (3, 32, 3)
    assert deformed.quats.shape == (3, 32, 4)
    assert deformed.scales.shape == (3, 32, 3)
    assert deformed.colors.shape == (3, 32, 3)


def test_fiducial_solve_creates_masks_and_updates_manifest(tmp_path: Path) -> None:
    dataset_dir = _make_dataset(tmp_path)
    result = solve_fiducials(str(dataset_dir), FiducialConfig(backend="mock", exclude_from_training=True))
    manifest = DatasetManifest.load(dataset_dir)
    dataset = TemporalSequenceDataset(dataset_dir, window_size=3, stride=1)
    sample = dataset[0]
    assert result["detected_frames"] == manifest.clip.total_frames
    assert len(manifest.fiducial_solves) == manifest.clip.total_frames
    assert len(manifest.masks) == manifest.clip.total_frames
    assert sample["masks"] is not None
    assert sample["masks"].shape == (3, 64, 64)


def test_colmap_command_construction_and_parsing(tmp_path: Path) -> None:
    bridge = ColmapCliBridge(ColmapConfig(executable="colmap", mode="standard", matching_mode="sequential"))
    commands = bridge.build_commands("/images", str(tmp_path / "project"))
    assert commands[0][1] == "feature_extractor"
    assert commands[1][1] == "sequential_matcher"
    diagnostics = bridge.diagnostics()
    assert diagnostics["backend"] == "colmap-cli"
    assert diagnostics["capability"] == "missing"
    parsed = parse_images_txt(
        "# comment\n1 1 0 0 0 1 2 3 1 000001.png\n0 0 0\n"
    )
    assert len(parsed) == 1
    assert parsed[0].source_of_pose == "colmap"


def test_colmap_export_writes_images_and_manifest(tmp_path: Path) -> None:
    dataset_dir = _make_dataset(tmp_path)
    bridge = ColmapCliBridge(ColmapConfig(executable="colmap"))
    image_dir = bridge.export_frames(str(dataset_dir))
    exported = sorted(image_dir.glob("*.png"))
    export_manifest = Path(dataset_dir) / "colmap" / "export_manifest.json"
    assert len(exported) == 12
    assert export_manifest.exists()


def test_alignment_transform_updates_pose_provenance(tmp_path: Path) -> None:
    record = CameraRecord(
        camera_record_id="cam-1",
        frame_record_id="clip:000000",
        intrinsics={"fx": 1.0, "fy": 1.0, "cx": 0.0, "cy": 0.0},
        source_of_pose="colmap",
        pose_confidence=0.9,
        pose_provenance=["colmap"],
    )
    result = align_cameras_to_fiducials([record], {"clip:000000": [[1.0, 0.0, 0.0, 0.5], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.0, 1.0]]})
    assert result.aligned_count == 1
    assert result.aligned_cameras[0].alignment_status == "aligned_to_fiducial_world"
    assert "world-aligned" in result.aligned_cameras[0].pose_provenance


def test_align_world_updates_manifest(tmp_path: Path) -> None:
    dataset_dir = _make_dataset(tmp_path)
    solve_fiducials(str(dataset_dir), FiducialConfig(backend="mock", exclude_from_training=False))
    result = align_world(str(dataset_dir))
    manifest = DatasetManifest.load(dataset_dir)
    assert result["aligned_cameras"] == manifest.clip.total_frames
    assert all(record.alignment_status == "aligned_to_fiducial_world" for record in manifest.camera_records)


def test_debug_poses_reports_trajectory_summary(tmp_path: Path) -> None:
    dataset_dir = _make_dataset(tmp_path)
    solve_fiducials(str(dataset_dir), FiducialConfig(backend="mock", exclude_from_training=False))
    align_world(str(dataset_dir))
    summary = debug_poses(str(dataset_dir))
    assert summary["camera_count"] == 12
    assert summary["status"] == "ok"
    assert "aligned_to_fiducial_world" in summary["alignment_statuses"]


def test_ingest_cli_reports_progress_and_summary(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "cli-dataset"
    result = runner.invoke(
        app,
        [
            "ingest",
            "synthetic.R3D",
            "--out",
            str(dataset_dir),
            "--backend",
            "mock",
            "--max-frames",
            "3",
            "--frame-step",
            "2",
        ],
    )
    assert result.exit_code == 0
    assert "[ingest]" in result.stdout
    assert "[ingest-settings]" in result.stdout
    assert "[ingest-estimate]" in result.stdout
    assert "[ingest-summary]" in result.stdout
    manifest = DatasetManifest.load(dataset_dir)
    assert len(manifest.frames) == 3


def test_ingest_cli_dry_run_writes_no_dataset(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dry-run"
    result = runner.invoke(
        app,
        [
            "ingest",
            "synthetic.R3D",
            "--out",
            str(dataset_dir),
            "--backend",
            "mock",
            "--preset",
            "quick-test",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0
    assert "[ingest-dry-run]" in result.stdout
    assert not dataset_dir.exists()


def test_training_step_executes_with_masks(tmp_path: Path) -> None:
    dataset_dir = _make_dataset(tmp_path)
    solve_fiducials(str(dataset_dir), FiducialConfig(backend="mock", exclude_from_training=True))
    dataset = TemporalSequenceDataset(dataset_dir, window_size=3, stride=1)
    batch = _collate([dataset[0], dataset[1]])
    trainer = FourDGaussianTrainer(
        TrainingConfig(
            dataset_dir=str(dataset_dir),
            output_dir=str(tmp_path / "runs"),
            window_size=3,
            num_gaussians=32,
            device="cpu",
        )
    )
    pred, losses = trainer(batch)
    assert pred.shape == (2, 3, 3, 64, 64)
    assert losses["total"].item() > 0.0


def test_renderer_raises_if_gsplat_exists_but_fails() -> None:
    class BrokenGsplat:
        @staticmethod
        def rasterization(*args, **kwargs):
            raise RuntimeError("boom")

    renderer = GSplatRendererBridge()
    renderer._gsplat = BrokenGsplat()
    renderer.backend = "gsplat"
    state = CanonicalGaussianModel(num_gaussians=8).canonical_state()
    cameras = {
        "viewmats": torch.eye(4, dtype=torch.float32).unsqueeze(0),
        "Ks": torch.tensor([[[60.0, 0.0, 32.0], [0.0, 60.0, 32.0], [0.0, 0.0, 1.0]]], dtype=torch.float32),
        "backgrounds": torch.zeros((1, 3), dtype=torch.float32),
    }
    try:
        renderer(state, cameras, (64, 64))
    except RuntimeError as exc:
        assert "boom" in str(exc)
    else:
        raise AssertionError("renderer should not silently fall back when gsplat import succeeded")


def test_renderer_diagnostics_report_fallback_state() -> None:
    diagnostics = GSplatRendererBridge().diagnostics()
    assert diagnostics["backend"] in {"torch", "gsplat"}
    assert "production_mode" in diagnostics


def test_train_4d_produces_checkpoint_and_loss_decreases(tmp_path: Path) -> None:
    dataset_dir = _make_dataset(tmp_path)
    solve_fiducials(str(dataset_dir), FiducialConfig(backend="mock", exclude_from_training=True))
    output_dir = tmp_path / "runs"
    checkpoint = train(
        TrainingConfig(
            dataset_dir=str(dataset_dir),
            output_dir=str(output_dir),
            epochs=6,
            window_size=3,
            num_gaussians=64,
            device="cpu",
            seed=11,
        )
    )
    assert checkpoint.exists()
    report = evaluate(str(checkpoint))
    assert "final_losses" in report
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    history = payload["history"]
    assert history[0]["photometric"] > history[-1]["photometric"]
    run_summary = output_dir / "run_summary.json"
    assert run_summary.exists()
    rendered = render_sequence(str(checkpoint), dataset_dir=str(dataset_dir))
    assert isinstance(rendered, torch.Tensor)
    assert rendered.shape[:3] == (1, 3, 3)


def test_checkpoint_roundtrip_matches_prediction_shape(tmp_path: Path) -> None:
    dataset_dir = _make_dataset(tmp_path)
    output_dir = tmp_path / "runs"
    checkpoint = train(
        TrainingConfig(
            dataset_dir=str(dataset_dir),
            output_dir=str(output_dir),
            epochs=2,
            window_size=4,
            num_gaussians=48,
            device="cpu",
            seed=5,
        )
    )
    rendered_a = render_sequence(str(checkpoint), dataset_dir=str(dataset_dir))
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = TrainingConfig(**payload["config"])
    trainer = FourDGaussianTrainer(config)
    trainer.load_state_dict(payload["state_dict"])
    trainer.eval()
    sample = _collate([TemporalSequenceDataset(dataset_dir, window_size=4, stride=1)[0]])
    with torch.no_grad():
        rendered_b, _ = trainer(copy.deepcopy(sample))
    assert tuple(rendered_a.shape) == tuple(rendered_b.shape)


REAL_BACKEND_ENABLED = os.environ.get("R3DSPLAT_ENABLE_REAL_BACKEND_TESTS") == "1"
REAL_GSPLAT_ENABLED = os.environ.get("R3DSPLAT_ENABLE_REAL_GSPLAT_TESTS") == "1"
REAL_R3D_PATH = os.environ.get("R3DSPLAT_REAL_R3D")
REAL_COLMAP_BIN = os.environ.get("R3DSPLAT_COLMAP_BIN", "colmap")


@pytest.mark.real_backend
@pytest.mark.skipif(not REAL_BACKEND_ENABLED or not REAL_R3D_PATH, reason="real RED backend test not configured")
def test_real_red_ingest_gated() -> None:
    summary = ingest_backend_summary("red-sdk")
    assert summary["backend"] == "red-sdk"
    clip = resolve_ingest_backend("red-sdk").inspect_clip(REAL_R3D_PATH)
    assert clip.total_frames > 0


@pytest.mark.real_backend
@pytest.mark.skipif(not REAL_BACKEND_ENABLED, reason="real COLMAP test not configured")
def test_real_colmap_gated(tmp_path: Path) -> None:
    dataset_dir = _make_dataset(tmp_path)
    bridge = ColmapCliBridge(ColmapConfig(executable=REAL_COLMAP_BIN))
    if not bridge.is_available():
        pytest.skip("real COLMAP executable not available")
    result = bridge.run(str(dataset_dir))
    assert result["solve_status"] == "completed"


@pytest.mark.real_backend
@pytest.mark.skipif(not REAL_BACKEND_ENABLED or not REAL_GSPLAT_ENABLED, reason="real gsplat test not configured")
def test_real_gsplat_gated() -> None:
    diagnostics = GSplatRendererBridge(production_mode=True).diagnostics()
    assert diagnostics["backend"] == "gsplat"
