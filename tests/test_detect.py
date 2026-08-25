"""Detector interface, and the parts of the CoreML path that run without a model."""

import numpy as np
import pytest

from geoloc_agent.contracts import Frame, Intrinsics, Pose
from geoloc_agent.detect.coreml import BUDGET_MS, COCO_KEEP, CoreMLDetector, benchmark_stage
from geoloc_agent.io.synthetic import SyntheticScenario, SyntheticSession


def test_missing_model_explains_how_to_export_one():
    detector = CoreMLDetector("/nonexistent/yolo11n.mlpackage")
    with pytest.raises(FileNotFoundError, match="export"):
        detector.warmup()


def test_detector_refuses_a_frame_with_no_pixels():
    """Geometry-only sessions must use StubDetector, and be told so."""
    frame = Frame(
        frame_id=0, timestamp=0.0,
        intrinsics=Intrinsics(fx=100, fy=100, cx=50, cy=50, width=100, height=100),
        pose=Pose(R=np.eye(3), t=np.zeros(3)), image=None,
    )
    with pytest.raises(ValueError, match="StubDetector"):
        CoreMLDetector("/nonexistent/m.mlpackage").detect(frame)


def test_only_relevant_coco_classes_are_kept():
    assert COCO_KEEP[0] == "pedestrian"
    assert COCO_KEEP[2] == "car"
    # COCO has no infrastructure classes; nothing is invented for them.
    assert "traffic_light" not in COCO_KEEP.values()


def test_raw_output_maps_to_original_frame_coordinates():
    """Boxes must scale to the original image, not the 640x640 network input.

    Every bearing downstream is taken through the original intrinsics, so a box
    left in network coordinates produces a plausible-looking, entirely wrong map.
    """
    detector = CoreMLDetector("/unused", score_threshold=0.3)
    frame = Frame(
        frame_id=7, timestamp=0.0,
        intrinsics=Intrinsics(fx=1266, fy=1266, cx=800, cy=450, width=1600, height=900),
        pose=Pose(R=np.eye(3), t=np.zeros(3)), image=np.zeros((900, 1600, 3), dtype=np.uint8),
    )
    raw = {
        "confidence": np.array([[0.0, 0.0, 0.9, 0, 0, 0, 0, 0]]),
        "coordinates": np.array([[0.5, 0.5, 0.1, 0.2]]),
    }
    detections = detector._to_detections(raw, frame, 1600, 900)
    assert len(detections) == 1
    detection = detections[0]
    assert detection.cls == "car"
    assert detection.frame_id == 7
    assert np.allclose(detection.centroid, [800.0, 450.0])
    assert detection.width == pytest.approx(160.0)
    assert detection.height == pytest.approx(180.0)


def test_low_scores_and_irrelevant_classes_are_dropped():
    detector = CoreMLDetector("/unused", score_threshold=0.5)
    frame = Frame(
        frame_id=0, timestamp=0.0,
        intrinsics=Intrinsics(fx=100, fy=100, cx=50, cy=50, width=100, height=100),
        pose=Pose(R=np.eye(3), t=np.zeros(3)), image=np.zeros((100, 100, 3), dtype=np.uint8),
    )
    raw = {
        "confidence": np.array([[0, 0, 0.2, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0.99, 0, 0, 0]]),
        "coordinates": np.array([[0.5, 0.5, 0.1, 0.1], [0.5, 0.5, 0.1, 0.1]]),
    }
    # First is below threshold; second is COCO class 4 (aeroplane), not kept.
    assert detector._to_detections(raw, frame, 100, 100) == []


def test_benchmark_reports_p95_and_checks_it_against_the_budget():
    """p95, not mean: a loop that misses budget one frame in ten drops one in ten."""
    timing = benchmark_stage(lambda: None, iterations=20, stage="perception")
    assert len(timing.samples_ms) == 20
    assert timing.p95_ms >= 0
    assert timing.within_budget
    assert BUDGET_MS["perception"] < BUDGET_MS["ranging"] < BUDGET_MS["decision"]


def test_geometry_and_fuse_stages_fit_the_ranging_budget():
    """The stages that exist today must leave room for a real detector."""
    session = SyntheticSession(SyntheticScenario(path="lateral", n_frames=20))
    from geoloc_agent.pipeline import run_pipeline

    timing = benchmark_stage(lambda: run_pipeline(session), iterations=3, stage="ranging")
    per_frame_ms = timing.mean_ms / 20
    assert per_frame_ms < BUDGET_MS["ranging"], f"{per_frame_ms:.1f} ms/frame"
