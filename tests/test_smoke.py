"""Phase 0 acceptance: the whole stack, end to end, with no ML in the loop."""

import json

import numpy as np
import pytest

from geoloc_agent.detect.stub import StubDetector
from geoloc_agent.io.synthetic import SyntheticScenario, SyntheticSession
from geoloc_agent.noise import NoiseModel
from geoloc_agent.pipeline import run_pipeline
from geoloc_agent.range.triangulation import TriangulationRanger

ACCEPT_METRES = 0.5


def test_end_to_end_geolocates_a_synthetic_object_under_half_a_metre():
    """The Phase 0 gate."""
    session = SyntheticSession(SyntheticScenario(path="lateral", n_frames=40))
    result = run_pipeline(session)
    located = [t for t in result.final_tracks if t.truth_id in result.truth]
    assert located, "no track was matched to a truth object"
    errors = {
        t.truth_id: float(np.linalg.norm(t.mean - result.truth[t.truth_id])) for t in located
    }
    assert min(errors.values()) < ACCEPT_METRES, errors


def test_every_well_conditioned_track_is_under_half_a_metre():
    session = SyntheticSession(SyntheticScenario(path="lateral", n_frames=40))
    result = run_pipeline(session)
    for track in result.final_tracks:
        if track.truth_id in result.truth and not track.degenerate:
            error = float(np.linalg.norm(track.mean - result.truth[track.truth_id]))
            assert error < ACCEPT_METRES, f"{track.truth_id}: {error:.3f} m"


@pytest.mark.parametrize("path", ["lateral", "straight", "arc"])
def test_pipeline_runs_on_every_path_type(path):
    session = SyntheticSession(SyntheticScenario(path=path, n_frames=30, arc_radius_m=40.0))
    result = run_pipeline(session)
    assert result.n_frames == 30
    assert result.final_tracks


def test_pipeline_is_deterministic_for_a_fixed_seed():
    noise = NoiseModel(gps_sigma=0.5, bearing_sigma=2e-3, detection_dropout=0.1)
    means = []
    for _ in range(2):
        session = SyntheticSession(SyntheticScenario(path="lateral", n_frames=30))
        result = run_pipeline(session, noise=noise, seed=7)
        means.append(sorted(tuple(np.round(t.mean, 9)) for t in result.final_tracks))
    assert means[0] == means[1]


def test_stub_detector_round_trips_through_json(tmp_path):
    """'Scripted detections from JSON', literally."""
    session = SyntheticSession(SyntheticScenario(path="lateral", n_frames=10))
    path = session.write_detection_script(tmp_path / "dets.json")
    assert json.loads(path.read_text())["session"] == session.name

    detector = StubDetector.from_json(path)
    for frame in session.frames():
        replayed = detector.detect(frame)
        original = session.scripted_detections()[frame.frame_id]
        assert len(replayed) == len(original)
        for a, b in zip(replayed, original, strict=True):
            assert np.allclose(a.bbox, b.bbox)
            assert a.cls == b.cls


def test_pipeline_accepts_an_external_detector():
    session = SyntheticSession(SyntheticScenario(path="lateral", n_frames=20))
    detector = StubDetector.from_session(session)
    result = run_pipeline(session, detector=detector)
    assert result.final_tracks


def test_ranger_attaches_ranges_without_breaking_the_pipeline():
    session = SyntheticSession(SyntheticScenario(path="lateral", n_frames=30))
    result = run_pipeline(session, ranger=TriangulationRanger(), range_every_n=3)
    located = [t for t in result.final_tracks if t.truth_id in result.truth]
    assert located
    assert min(float(np.linalg.norm(t.mean - result.truth[t.truth_id])) for t in located) < 1.0


def test_degenerate_and_good_geometry_are_reported_separately():
    """Forward motion must be flagged; lateral motion must not be."""
    straight = run_pipeline(SyntheticSession(SyntheticScenario(path="straight", n_frames=40)))
    lateral = run_pipeline(SyntheticSession(SyntheticScenario(path="lateral", n_frames=40)))
    assert any(t.degenerate for t in straight.final_tracks)
    good = [t for t in lateral.final_tracks if not t.degenerate]
    assert good
    # A degenerate track must carry visibly worse uncertainty, not just a flag.
    worst_good = max(t.sigma_horizontal for t in good)
    degenerate = [t for t in straight.final_tracks if t.degenerate]
    assert min(t.sigma_horizontal for t in degenerate) > worst_good


def test_noise_degrades_accuracy_monotonically():
    """Median, not RMSE.

    The error distribution is heavy-tailed by nature: a short track on a small
    perpendicular baseline can land badly wrong while reporting a small
    covariance, because the covariance is linearised about the (wrong) estimate.
    RMSE on a dozen seeds is then a measurement of how many outliers happened to
    land in the sample, not of accuracy. The tail is real and is asserted
    separately in the next test rather than averaged into silence here.
    """
    medians = []
    for bearing_sigma in (0.0, 2e-3, 8e-3):
        samples = []
        for seed in range(12):
            session = SyntheticSession(SyntheticScenario(path="lateral", n_frames=40))
            result = run_pipeline(
                session, noise=NoiseModel(bearing_sigma=bearing_sigma), seed=seed,
                bearing_sigma_px=max(bearing_sigma * 1266.4, 0.5),
            )
            samples += [
                float(np.linalg.norm(t.mean - result.truth[t.truth_id]))
                for t in result.final_tracks
                if t.truth_id in result.truth and not t.degenerate
            ]
        medians.append(float(np.median(samples)))
    assert medians[0] < medians[1] < medians[2], medians


def test_gross_error_rate_stays_bounded_at_moderate_noise():
    """The tail, stated explicitly rather than hidden inside an RMSE."""
    gross = total = 0
    for seed in range(20):
        session = SyntheticSession(SyntheticScenario(path="lateral", n_frames=40))
        result = run_pipeline(
            session, noise=NoiseModel(bearing_sigma=2e-3), seed=seed, bearing_sigma_px=2.53
        )
        for track in result.final_tracks:
            if track.truth_id in result.truth and not track.degenerate:
                total += 1
                if float(np.linalg.norm(track.mean - result.truth[track.truth_id])) > 5.0:
                    gross += 1
    assert total > 0
    assert gross / total < 0.05, f"{gross}/{total} tracks were grossly wrong"


def test_no_ml_dependencies_are_imported():
    """Phase 0 runs with numpy and scipy only."""
    import sys

    for module in ("torch", "ultralytics", "coremltools", "cv2"):
        assert module not in sys.modules
