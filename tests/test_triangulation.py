"""Triangulation correctness, including the Phase 1 acceptance criteria."""

import numpy as np
import pytest

from geoloc_agent.contracts import Observation
from geoloc_agent.geometry import expected_range_sigma
from geoloc_agent.range.triangulation import TriangulationRanger, triangulate


def bearings_to(target, origins, sigma=1e-3):
    obs = []
    for origin in origins:
        bearing = np.asarray(target, dtype=float) - np.asarray(origin, dtype=float)
        obs.append(
            Observation(
                t=0.0, frame_id=0, origin=np.asarray(origin, dtype=float),
                bearing=bearing, bearing_sigma=sigma,
            )
        )
    return obs


def test_two_clean_bearings_recover_the_point_exactly():
    target = np.array([3.0, 40.0, 1.5])
    result = triangulate(bearings_to(target, [np.array([-5.0, 0, 0]), np.array([5.0, 0, 0])]))
    assert result.ok
    assert np.allclose(result.position, target, atol=1e-6)


def test_single_bearing_cannot_triangulate():
    result = triangulate(bearings_to(np.array([0.0, 30.0, 0.0]), [np.zeros(3)]))
    assert not result.ok
    assert "two bearings" in result.reason


def test_parallel_bearings_are_flagged_degenerate_not_answered_confidently():
    """Pure forward motion toward the target: range is unobservable."""
    target = np.array([0.0, 100.0, 0.0])
    origins = [np.array([0.0, y, 0.0]) for y in (0.0, 2.0, 4.0)]
    result = triangulate(bearings_to(target, origins))
    assert not result.ok
    assert "degenerate" in result.reason


def test_covariance_matches_the_analytic_two_view_prediction():
    """Phase 1 acceptance: error matches R^2 sigma / B (exact form carries sqrt(2))."""
    sigma, range_m = 1e-3, 50.0
    for baseline in (2.0, 5.0, 10.0):
        target = np.array([0.0, range_m, 0.0])
        origins = [np.array([-baseline / 2, 0.0, 0.0]), np.array([baseline / 2, 0.0, 0.0])]
        result = triangulate(bearings_to(target, origins, sigma), prior_range=range_m)
        assert result.ok
        # Sigma along the line of sight is the dominant error direction.
        along = np.array([0.0, 1.0, 0.0])
        measured = np.sqrt(along @ result.cov @ along)
        analytic = expected_range_sigma(range_m, sigma, baseline)
        assert measured == pytest.approx(analytic, rel=0.20)
        # And the familiar rule of thumb is the same thing without the sqrt(2).
        assert measured == pytest.approx(np.sqrt(2) * range_m**2 * sigma / baseline, rel=0.20)


def test_covariance_shrinks_monotonically_as_baseline_grows():
    """Phase 1 acceptance: more perpendicular baseline is always more information."""
    sigma, range_m = 1e-3, 50.0
    previous = np.inf
    for baseline in (1.0, 2.0, 4.0, 8.0, 16.0):
        target = np.array([0.0, range_m, 0.0])
        origins = [np.array([-baseline / 2, 0.0, 0.0]), np.array([baseline / 2, 0.0, 0.0])]
        result = triangulate(bearings_to(target, origins, sigma), prior_range=range_m)
        assert result.ok
        along = np.array([0.0, 1.0, 0.0])
        current = float(np.sqrt(along @ result.cov @ along))
        assert current < previous, f"covariance grew at baseline {baseline}"
        previous = current


def test_more_observations_shrink_the_covariance():
    target = np.array([0.0, 40.0, 0.0])
    previous = np.inf
    for count in (2, 4, 8, 16):
        origins = [np.array([x, 0.0, 0.0]) for x in np.linspace(-8, 8, count)]
        result = triangulate(bearings_to(target, origins), prior_range=40.0)
        assert result.ok
        current = float(np.trace(result.cov))
        assert current < previous
        previous = current


def test_solution_behind_the_camera_is_rejected():
    # Bearings that diverge rather than converge intersect behind the cameras.
    obs = [
        Observation(t=0.0, frame_id=0, origin=np.array([-5.0, 0, 0]),
                    bearing=np.array([-1.0, 1.0, 0.0]), bearing_sigma=1e-3),
        Observation(t=0.0, frame_id=1, origin=np.array([5.0, 0, 0]),
                    bearing=np.array([1.0, 1.0, 0.0]), bearing_sigma=1e-3),
    ]
    result = triangulate(obs)
    assert not result.ok


def test_ranger_returns_invalid_rather_than_guessing_at_low_parallax():
    target = np.array([0.0, 100.0, 0.0])
    origins = [np.array([0.0, y, 0.0]) for y in (0.0, 1.0)]
    obs = bearings_to(target, origins)
    ranger = TriangulationRanger()
    meas = ranger.range_for(obs[-1], obs[:-1])
    assert not meas.valid
    assert meas.reason


def test_ranger_produces_a_range_with_a_sigma_when_geometry_is_good():
    target = np.array([0.0, 40.0, 0.0])
    obs = bearings_to(target, [np.array([-10.0, 0, 0]), np.array([10.0, 0, 0])])
    meas = TriangulationRanger().range_for(obs[-1], obs[:-1])
    assert meas.valid
    assert meas.value == pytest.approx(np.linalg.norm(target - obs[-1].origin), rel=1e-4)
    assert meas.sigma > 0
