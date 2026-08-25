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
    """Forward motion must be flagged; lateral motion must not be.

    The slow approach is deliberate. Inside a 50 m operating envelope, driving
    forward 20 m closes 40% of the distance to a 40 m object, and that alone
    makes its range observable (sigma 1.1 m). Degeneracy is not a property of
    driving forward as such -- it is a property of how little the geometry
    changes over the observation, and at short range a long approach changes it
    a lot. 12 m of travel is the regime where it genuinely stays unobservable.
    """
    straight = run_pipeline(
        SyntheticSession(SyntheticScenario(path="straight", n_frames=40, speed_mps=3.0))
    )
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
    """Phase 0 runs with numpy and scipy only.

    Checked in a subprocess. Asserting on this process's `sys.modules` would pass
    or fail depending on whether some other test had already imported
    coremltools, which measures test ordering rather than the property.
    """
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable, "-c",
            "import geoloc_agent, geoloc_agent.pipeline, geoloc_agent.detect.stub, sys; "
            "print(sorted(m for m in ('torch','ultralytics','coremltools','cv2') "
            "if m in sys.modules))",
        ],
        capture_output=True, text=True, check=True,
    )
    assert result.stdout.strip() == "[]", result.stdout


def test_operating_envelope_derives_every_range_default():
    """One number drives the rest. Changing it must move all of them together."""
    from geoloc_agent.envelope import OperatingEnvelope
    from geoloc_agent.fuse.tracker import TrackerConfig

    small, large = OperatingEnvelope(max_range_m=50.0), OperatingEnvelope(max_range_m=200.0)
    for attr in ("prior_range", "init_range_sigma", "track_max_range", "detector_max_range"):
        assert getattr(large, attr) > getattr(small, attr), attr

    # The prior must actually cover its envelope: the far edge inside ~2 sigma,
    # or new tracks at range are born outside their own prior.
    for envelope in (small, large):
        z = (envelope.max_range_m - envelope.prior_range) / envelope.init_range_sigma
        assert 1.0 < z < 2.5, (envelope.max_range_m, z)
        assert envelope.track_max_range > envelope.max_range_m  # headroom before "broken"

    with pytest.raises(ValueError, match="must exceed"):
        OperatingEnvelope(max_range_m=1.0, min_range_m=2.0)

    # And the tracker actually uses them rather than keeping its own numbers.
    config = TrackerConfig()
    assert config.prior_range == pytest.approx(OperatingEnvelope().prior_range)
    assert config.max_range == pytest.approx(OperatingEnvelope().track_max_range)


def test_unreachable_objects_are_flagged_somehow_rather_than_reported():
    """A 300 m object seen from a 12 m baseline must never come back as a fix.

    Deliberately does not assert *which* guard catches it. The estimate collapses
    toward the prior rather than running out to 300 m, so it is the relative
    range-sigma test that fires, not the envelope one -- and pinning the specific
    reason would make this test about the internals rather than the property.
    """
    import numpy as np

    from geoloc_agent.contracts import Observation
    from geoloc_agent.fuse.tracker import Tracker, TrackerConfig

    tracker = Tracker(TrackerConfig())
    target = np.array([0.0, 300.0, 0.0])
    for i in range(12):
        origin = np.array([-6.0 + i * 1.0, 0.0, 0.0])
        tracker.step(
            [Observation(t=i * 0.1, frame_id=i, origin=origin, bearing=target - origin,
                         bearing_sigma=1e-3)],
            t=i * 0.1,
        )
    states = tracker.live_states()
    assert states
    assert states[0].degenerate
    assert states[0].degeneracy_reason.strip()


def test_envelope_guard_catches_an_estimate_that_escapes_the_envelope():
    """The guard itself: an estimate past the envelope is broken, not long-range."""
    import numpy as np

    from geoloc_agent.contracts import Observation
    from geoloc_agent.envelope import DEFAULT_ENVELOPE
    from geoloc_agent.fuse.tracker import Tracker, TrackerConfig

    tracker = Tracker(TrackerConfig())
    target = np.array([0.0, 30.0, 0.0])
    for i in range(6):
        origin = np.array([-5.0 + i * 2.0, 0.0, 0.0])
        tracker.step(
            [Observation(t=i * 0.1, frame_id=i, origin=origin, bearing=target - origin,
                         bearing_sigma=1e-3)],
            t=i * 0.1,
        )
    track = next(iter(tracker.tracks.values()))
    assert not track.state.degenerate  # a good fix inside the envelope

    # Now shove the estimate past the envelope and re-assess.
    beyond = DEFAULT_ENVELOPE.track_max_range * 2.0
    track.state.mean = np.array([0.0, beyond, 0.0])
    tracker._refresh_geometry(track)
    assert track.state.degenerate
    assert "envelope" in track.state.degeneracy_reason
