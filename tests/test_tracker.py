"""Track management: birth, association, localisation, degeneracy, death."""

import numpy as np
import pytest

from geoloc_agent.contracts import Observation, TrackStatus
from geoloc_agent.fuse.degenerate import assess_geometry, max_perpendicular_baseline
from geoloc_agent.fuse.tracker import Tracker, TrackerConfig


def obs_to(origin, target, t=0.0, frame_id=0, sigma=1e-3, cls="car", score=0.9, truth_id=None):
    bearing = np.asarray(target, float) - np.asarray(origin, float)
    return Observation(
        t=t, frame_id=frame_id, origin=np.asarray(origin, float), bearing=bearing,
        bearing_sigma=sigma, cls=cls, score=score, truth_id=truth_id,
    )


def lateral_run(target, n=12, span=20.0, **kw):
    """Camera strafing across the line of sight: well-conditioned geometry."""
    tracker = Tracker(TrackerConfig(**kw))
    for i in range(n):
        origin = np.array([-span / 2 + span * i / (n - 1), 0.0, 0.0])
        tracker.step([obs_to(origin, target, t=i * 0.1, frame_id=i)], t=i * 0.1)
    return tracker


def test_track_is_born_confirmed_and_localised_with_good_geometry():
    target = np.array([0.0, 40.0, 0.0])
    tracker = lateral_run(target)
    assert len(tracker.tracks) == 1
    state = tracker.live_states()[0]
    assert state.status is TrackStatus.CONFIRMED
    assert state.n_obs == 12
    assert np.allclose(state.mean, target, atol=1e-3)
    assert not state.degenerate


def test_one_object_produces_exactly_one_track():
    """The duplicate-track failure: a track that rejects its own observations."""
    tracker = lateral_run(np.array([0.0, 95.0, 0.0]), n=30, span=20.0)
    assert len(tracker.tracks) == 1
    assert tracker.live_states()[0].n_obs == 30


def test_two_separated_objects_produce_two_tracks():
    target_a = np.array([-15.0, 40.0, 0.0])
    target_b = np.array([15.0, 40.0, 0.0])
    tracker = Tracker(TrackerConfig())
    for i in range(12):
        origin = np.array([-10 + 20 * i / 11, 0.0, 0.0])
        tracker.step(
            [
                obs_to(origin, target_a, t=i * 0.1, frame_id=i, truth_id="a"),
                obs_to(origin, target_b, t=i * 0.1, frame_id=i, truth_id="b"),
            ],
            t=i * 0.1,
        )
    assert len(tracker.tracks) == 2
    by_truth = {t.truth_id: t for t in tracker.live_states()}
    assert np.allclose(by_truth["a"].mean, target_a, atol=1e-3)
    assert np.allclose(by_truth["b"].mean, target_b, atol=1e-3)


def test_forward_motion_is_flagged_degenerate():
    """The headline failure mode: no perpendicular baseline, so no range."""
    target = np.array([0.0, 80.0, 0.0])
    tracker = Tracker(TrackerConfig())
    for i in range(20):
        origin = np.array([0.0, 0.5 * i, 0.0])  # straight at the target
        tracker.step([obs_to(origin, target, t=i * 0.1, frame_id=i)], t=i * 0.1)
    state = tracker.live_states()[0]
    assert state.degenerate
    assert "parallax" in state.degeneracy_reason or "unobservable" in state.degeneracy_reason
    # And the covariance must be honest about it rather than merely flagged.
    assert state.sigma_horizontal > 5.0


def test_degenerate_geometry_still_produces_a_track():
    """A flagged track is still reported -- the operator decides, not the filter."""
    target = np.array([0.0, 80.0, 0.0])
    tracker = Tracker(TrackerConfig())
    for i in range(20):
        obs = obs_to(np.array([0.0, 0.5 * i, 0.0]), target, t=i * 0.1, frame_id=i)
        tracker.step([obs], i * 0.1)
    assert len(tracker.live_states()) == 1


def test_track_dies_after_max_misses():
    target = np.array([0.0, 40.0, 0.0])
    tracker = lateral_run(target, n=6, span=10.0, max_misses=3)
    assert len(tracker.tracks) == 1
    for i in range(10):
        tracker.step([], t=1.0 + i * 0.1)
    assert len(tracker.tracks) == 0


def test_coasting_then_death_is_reported_in_status():
    tracker = lateral_run(np.array([0.0, 40.0, 0.0]), n=6, span=10.0, max_misses=3)
    tracker.step([], t=2.0)
    assert tracker.live_states()[0].status is TrackStatus.COASTING


def test_class_posterior_concentrates_but_never_collapses():
    target = np.array([0.0, 40.0, 0.0])
    tracker = Tracker(TrackerConfig())
    for i in range(20):
        origin = np.array([-10 + 20 * i / 19, 0.0, 0.0])
        tracker.step([obs_to(origin, target, t=i * 0.1, frame_id=i, cls="car", score=0.9)], i * 0.1)
    posterior = tracker.live_states()[0].class_posterior
    assert posterior["car"] > 0.9
    assert sum(posterior.values()) == pytest.approx(1.0)
    # Floored, so entropy stays a usable signal for the agent layer.
    assert all(p > 0 for p in posterior.values())


def test_ambiguous_class_evidence_keeps_entropy_high():
    target = np.array([0.0, 40.0, 0.0])
    tracker = Tracker(TrackerConfig())
    for i in range(20):
        origin = np.array([-10 + 20 * i / 19, 0.0, 0.0])
        cls = "car" if i % 2 == 0 else "pedestrian"
        tracker.step([obs_to(origin, target, t=i * 0.1, frame_id=i, cls=cls, score=0.55)], i * 0.1)
    state = tracker.live_states()[0]
    assert state.class_entropy > 0.4
    assert state.top_class[1] < 0.8


def test_max_perpendicular_baseline_ignores_along_ray_motion():
    target = np.array([0.0, 100.0, 0.0])
    forward = [np.array([0.0, y, 0.0]) for y in (0.0, 10.0, 20.0)]
    assert max_perpendicular_baseline(forward, target) == pytest.approx(0.0, abs=1e-9)
    lateral = [np.array([x, 0.0, 0.0]) for x in (-5.0, 0.0, 5.0)]
    assert max_perpendicular_baseline(lateral, target) == pytest.approx(10.0, rel=1e-6)


def test_assess_geometry_flags_a_single_view():
    report = assess_geometry([np.zeros(3)], np.array([0.0, 50.0, 0.0]), np.eye(3) * 100)
    assert report.degenerate
    assert "single view" in report.reason
